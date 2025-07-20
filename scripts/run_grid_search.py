# Copyright 2025 Igor Bogdanov & Olga Manakina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import pandas as pd
import numpy as np
import argparse
import datetime
import time
import threading
import queue
from typing import Tuple
import traceback
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from settings import DEFAULT_DATASET_FILE_PATH
from grader.essay_grader import EssayGraderAgent
from grader.grid_search_logger import (
    GridSearchLogger,
    METHOD_PREFIXES_GS,
)
from utils.llm_connector.connector.connector_settings import (
    MODEL_PRICING,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
)


def get_price(provider: str, model: str) -> Tuple[float, float]:
    return MODEL_PRICING.get((provider, model), (0.0, 0.0))


stop_workers_grid = False


def grid_worker_function(
    worker_id: int,
    task_queue: queue.Queue,
    results_queue: queue.Queue,
    provider_config: tuple,
    debug: bool,
    use_detailed_prompts_for_worker: bool,
):
    print(f"GridWorker {worker_id} started.")
    try:
        agent = EssayGraderAgent(provider=provider_config)
    except Exception as agent_e:
        print(
            f"GridWorker {worker_id}: Failed to initialize EssayGraderAgent: {agent_e}"
        )
        return

    while not stop_workers_grid:
        try:
            task_data = task_queue.get(timeout=1.0)
            if task_data is None:
                task_queue.put(None)
                break

            original_csv_index, question, essay_text, human_score_str = task_data
            essay_run_results = {"index": original_csv_index, "human": human_score_str}
            price_in, price_out = get_price(provider_config[0], provider_config[1])

            for prefix in METHOD_PREFIXES_GS:
                essay_run_results[f"{prefix}_pred"] = None
                essay_run_results[f"{prefix}_err"] = np.nan
                essay_run_results[f"{prefix}_prompt_tokens"] = 0
                essay_run_results[f"{prefix}_completion_tokens"] = 0
                essay_run_results[f"{prefix}_total_tokens"] = 0
                essay_run_results[f"{prefix}_latency"] = 0.0
                essay_run_results[f"{prefix}_estimated_cost"] = 0.0

            essay_agg_prompt_tokens = 0
            essay_agg_completion_tokens = 0
            essay_agg_total_tokens = 0
            essay_agg_latency = 0.0
            essay_agg_estimated_cost = 0.0

            method_calls_map = {
                "multi_ex": lambda: agent.grade_essay_multi_step(
                    question,
                    essay_text,
                    use_examples=True,
                    use_detailed_criteria_prompts=use_detailed_prompts_for_worker,
                    debug=debug,
                ),
                "multi_noex": lambda: agent.grade_essay_multi_step(
                    question,
                    essay_text,
                    use_examples=False,
                    use_detailed_criteria_prompts=use_detailed_prompts_for_worker,
                    debug=debug,
                ),
                "single_noex": lambda: agent.grade_essay_single_direct_poc(
                    question,
                    essay_text,
                    use_detailed_criteria_prompts=use_detailed_prompts_for_worker,
                    debug=debug,
                ),
                "single_ex": lambda: agent.grade_essay_single_direct_with_examples_poc(
                    question,
                    essay_text,
                    use_detailed_criteria_prompts=use_detailed_prompts_for_worker,
                    debug=debug,
                ),
            }

            for prefix in METHOD_PREFIXES_GS:
                try:
                    method_to_call = method_calls_map[prefix]

                    if prefix.startswith("single"):
                        parsed_score, raw_llm_text, p, c, t, lat = method_to_call()
                        essay_run_results[f"{prefix}_pred"] = parsed_score
                    else:
                        response_dict, p, c, t, lat = method_to_call()
                        essay_run_results[f"{prefix}_pred"] = response_dict.get(
                            "overall", {}
                        ).get("score", "Error")

                    essay_run_results[f"{prefix}_prompt_tokens"] = p
                    essay_run_results[f"{prefix}_completion_tokens"] = c
                    essay_run_results[f"{prefix}_total_tokens"] = t
                    essay_run_results[f"{prefix}_latency"] = lat

                    cost = 0.0
                    if p > 0 and price_in > 0.0:
                        cost += (p / 1_000_000) * price_in
                    if c > 0 and price_out > 0.0:
                        cost += (c / 1_000_000) * price_out
                    essay_run_results[f"{prefix}_estimated_cost"] = cost

                    essay_agg_prompt_tokens += p
                    essay_agg_completion_tokens += c
                    essay_agg_total_tokens += t
                    essay_agg_latency += lat
                    essay_agg_estimated_cost += cost

                except Exception as e:
                    print(
                        f"GridWorker {worker_id} Error [{prefix}] Essay {original_csv_index}: {e}"
                    )
                    essay_run_results[f"{prefix}_pred"] = "Exception"

            try:
                human_float = float(human_score_str)
                for prefix in METHOD_PREFIXES_GS:
                    pred_val = essay_run_results[f"{prefix}_pred"]
                    if pred_val not in [
                        None,
                        "Error",
                        "Exception",
                        "Calculation Error",
                        "Parsing Error",
                    ]:
                        try:
                            essay_run_results[f"{prefix}_err"] = abs(
                                float(pred_val) - human_float
                            )
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                print(
                    f"GridWorker {worker_id}: Cannot convert human score '{human_score_str}' for Essay {original_csv_index}"
                )

            essay_run_results["total_prompt_tokens"] = essay_agg_prompt_tokens
            essay_run_results["total_completion_tokens"] = essay_agg_completion_tokens
            essay_run_results["total_tokens"] = essay_agg_total_tokens
            essay_run_results["total_latency"] = essay_agg_latency
            essay_run_results["estimated_cost"] = essay_agg_estimated_cost

            results_queue.put(essay_run_results)
            task_queue.task_done()

        except queue.Empty:
            if stop_workers_grid:
                break
            else:
                time.sleep(0.1)
                continue
        except Exception as e:
            print(f"GridWorker {worker_id}: Unhandled exception in worker loop: {e}")
            traceback.print_exc()
            task_queue.task_done()

    print(f"GridWorker {worker_id} finished.")


def main(args):
    print(
        f"--- Starting Concurrent Grid Search: {datetime.datetime.now().isoformat()} ---"
    )
    print(f"Args: {args}")
    global stop_workers_grid
    stop_workers_grid = False

    if args.results_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.model_name)
        prompt_type_str = "_basicP" if args.basic_prompts else "_detailP"
        args.results_file = (
            f"logs/grid_log_{args.provider_name}_{sanitized_model_name}_"
            f"{args.num_essays if args.num_essays else 'all'}essays{prompt_type_str}_{timestamp}.csv"
        )
        print(f"No results file specified, generated: {args.results_file}")

    print(f"Loading dataset from: {args.dataset}")
    try:
        data_df_full = pd.read_csv(args.dataset)
        required_columns = ["Question", "Essay", "Overall"]
        if not all(col in data_df_full.columns for col in required_columns):
            print(f"Error: CSV missing one of required columns: {required_columns}")
            sys.exit(1)

        data_df = data_df_full[required_columns].copy()
        data_df.dropna(inplace=True)

        if args.num_essays is not None and args.num_essays > 0:
            if args.num_essays > len(data_df):
                print(
                    f"Warning: Requested {args.num_essays} essays, but only {len(data_df)} available after cleaning. Processing {len(data_df)}."
                )
                essays_to_process_df = data_df
            else:
                essays_to_process_df = data_df.head(args.num_essays)
                print(
                    f"Selected first {len(essays_to_process_df)} essays for processing."
                )
        else:
            print("Processing all available valid essays.")
            essays_to_process_df = data_df

        if essays_to_process_df.empty:
            print("Error: No essays selected for processing.")
            sys.exit(1)

        essay_tasks = [
            (index, row["Question"], row["Essay"], str(row["Overall"]))
            for index, row in essays_to_process_df.iterrows()
        ]
        print(f"Prepared {len(essay_tasks)} essay tasks.")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    task_queue = queue.Queue()
    results_queue = queue.Queue()
    threads = []
    num_workers = args.workers
    provider_config = (args.provider_name, args.model_name)

    try:
        with GridSearchLogger(args.results_file) as logger:
            print(f"Logger initialized. Logging to: {args.results_file}")

            print(f"Starting {num_workers} worker threads...")
            for i in range(num_workers):
                thread = threading.Thread(
                    target=grid_worker_function,
                    args=(
                        i + 1,
                        task_queue,
                        results_queue,
                        provider_config,
                        args.debug,
                        not args.basic_prompts,
                    ),
                    daemon=True,
                )
                threads.append(thread)
                thread.start()

            print(f"Populating task queue with {len(essay_tasks)} essays...")
            start_time = time.time()
            for task in essay_tasks:
                task_queue.put(task)

            completed_tasks = 0
            while completed_tasks < len(essay_tasks):
                try:
                    essay_result = results_queue.get(timeout=1.0)
                    logger.log_essay_results(essay_result)
                    results_queue.task_done()
                    completed_tasks += 1
                    if completed_tasks % 20 == 0 or completed_tasks == len(essay_tasks):
                        print(
                            f"Logged results for {completed_tasks}/{len(essay_tasks)} essays."
                        )
                except queue.Empty:
                    if all(not t.is_alive() for t in threads) and task_queue.empty():
                        print(
                            "Workers finished and task queue empty, but not all results processed. Breaking."
                        )
                        break
                    continue

            print("All results processed from queue and logged.")

            task_queue.join()
            print("All essay tasks consumed by workers.")

            print("Signaling workers to stop...")
            stop_workers_grid = True
            for _ in range(num_workers):
                task_queue.put(None)

            print("Waiting for worker threads to finish...")
            for thread in threads:
                thread.join(timeout=10.0)
                if thread.is_alive():
                    print(
                        f"Warning: GridWorker thread {thread.ident} did not finish cleanly."
                    )
            print("Worker threads finished.")

            print("\n--- Concurrent Grid Search Finished ---")
            end_time = time.time()
            print(f"Total execution time: {end_time - start_time:.2f} seconds")

            try:
                df_results = pd.read_csv(args.results_file)
                for prefix in METHOD_PREFIXES_GS:
                    df_results[f"{prefix}_err"] = pd.to_numeric(
                        df_results[f"{prefix}_err"], errors="coerce"
                    )

                df_results.sort_values(by="EssayIndex", inplace=True)

                print("\nFinal Mean Absolute Errors (MAE) per Approach:")
                approach_maes = {}
                for prefix in METHOD_PREFIXES_GS:
                    mae = df_results[f"{prefix}_err"].mean()
                    approach_maes[prefix] = mae
                    approach_name = (
                        prefix.replace("_", " ")
                        .replace("ex", "(Ex)")
                        .replace("noex", "(NoEx)")
                        .replace("multi", "Multi-Step")
                        .replace("single", "Single-Step")
                    )
                    approach_name_parts = [
                        part.capitalize() for part in approach_name.split(" ")
                    ]
                    approach_name_formatted = " ".join(approach_name_parts)
                    approach_name_formatted = approach_name_formatted.replace(
                        "( Ex )", "(Ex)"
                    ).replace("( No Ex )", "(NoEx)")
                    print(f"{approach_name_formatted:<30} MAE: {mae:.3f}")

                print("\nAverage Usage Statistics per Approach (from log file):")
                agg_metrics_from_log = {}
                for prefix in METHOD_PREFIXES_GS:
                    agg_metrics_from_log[f"{prefix}_mae"] = approach_maes.get(
                        prefix, np.nan
                    )
                    agg_metrics_from_log[f"{prefix}_avg_prompt_tokens"] = df_results[
                        f"{prefix}_PromptTokens"
                    ].mean()
                    agg_metrics_from_log[f"{prefix}_avg_completion_tokens"] = (
                        df_results[f"{prefix}_CompletionTokens"].mean()
                    )
                    agg_metrics_from_log[f"{prefix}_avg_total_tokens"] = df_results[
                        f"{prefix}_TotalTokens"
                    ].mean()
                    agg_metrics_from_log[f"{prefix}_avg_latency"] = df_results[
                        f"{prefix}_Latency"
                    ].mean()
                    agg_metrics_from_log[f"{prefix}_avg_estimated_cost"] = df_results[
                        f"{prefix}_EstimatedCost"
                    ].mean()

                header_approach = "Approach"
                header_mae_col = "MAE"
                header_prompt_tk = "AvgPromptTk"
                header_compl_tk = "AvgComplTk"
                header_total_tk = "AvgTotalTk"
                header_latency = "AvgLatency"
                header_cost = "AvgCost"

                print(
                    f"{header_approach:<30} {header_mae_col:>7} {header_prompt_tk:>12} {header_compl_tk:>12} {header_total_tk:>12} {header_latency:>11} {header_cost:>15}"
                )
                print("-" * 105)
                for prefix in METHOD_PREFIXES_GS:
                    approach_name = (
                        prefix.replace("_", " ")
                        .replace("ex", "(Ex)")
                        .replace("noex", "(NoEx)")
                        .replace("multi", "Multi-Step")
                        .replace("single", "Single-Step")
                    )
                    approach_name_parts = [
                        part.capitalize() for part in approach_name.split(" ")
                    ]
                    approach_name_formatted = " ".join(approach_name_parts)
                    approach_name_formatted = approach_name_formatted.replace(
                        "( Ex )", "(Ex)"
                    ).replace("( No Ex )", "(NoEx)")
                    print(
                        (
                            f"{approach_name_formatted:<30} "
                            f"{agg_metrics_from_log[f'{prefix}_mae']:>7.3f} "
                            f"{agg_metrics_from_log[f'{prefix}_avg_prompt_tokens']:>12,.1f} "
                            f"{agg_metrics_from_log[f'{prefix}_avg_completion_tokens']:>12,.1f} "
                            f"{agg_metrics_from_log[f'{prefix}_avg_total_tokens']:>12,.1f} "
                            f"{agg_metrics_from_log[f'{prefix}_avg_latency']:>10.3f}s "
                            f"{agg_metrics_from_log[f'{prefix}_avg_estimated_cost']:>15.8f}"
                        )
                    )

                total_prompt_tokens_run = df_results["Essay_Agg_PromptTokens"].sum()
                total_completion_tokens_run = df_results[
                    "Essay_Agg_CompletionTokens"
                ].sum()
                total_tokens_run = df_results["Essay_Agg_TotalTokens"].sum()
                total_latency_run = df_results["Essay_Agg_TotalLatency"].sum()
                total_cost_run = df_results["Essay_Agg_EstimatedCost"].sum()
                num_processed_essays = len(df_results)

                print(
                    "\nOverall Usage Statistics for the Entire Run (Sum of per-essay totals from log):"
                )
                print(f"  Essays Processed:        {num_processed_essays}")
                print(f"  Total Prompt Tokens:     {total_prompt_tokens_run:,.0f}")
                print(f"  Total Completion Tokens: {total_completion_tokens_run:,.0f}")
                print(f"  Total Tokens:            {total_tokens_run:,.0f}")
                print(f"  Total Latency (sum):     {total_latency_run:.2f} seconds")
                print(f"  Total Estimated Cost:    ${total_cost_run:.4f}")

            except FileNotFoundError:
                print(f"Log file {args.results_file} not found for final analysis.")
            except Exception as analysis_e:
                print(
                    f"Error during final analysis of {args.results_file}: {analysis_e}"
                )
                traceback.print_exc()
    except Exception as main_e:
        print(f"Critical error in main execution: {main_e}")
        traceback.print_exc()
    finally:
        print("Ensuring workers are signaled to stop (grid search)...")
        stop_workers_grid = True
        if "task_queue" in locals():
            while not task_queue.empty():
                try:
                    task_queue.get_nowait()
                except queue.Empty:
                    break
                task_queue.task_done()
            for _ in range(num_workers):
                try:
                    task_queue.put(None, block=False)
                except queue.Full:
                    pass
        if "threads" in locals():
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Concurrent Grid Search for IELTS Essay Grading Recipes"
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_FILE_PATH,
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--num-essays",
        type=int,
        default=100,
        help="Number of essays from the start of the dataset to process (set to 0 or None for all)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to save detailed CSV results. If None, a filename is generated.",
    )
    parser.add_argument(
        "--provider-name",
        type=str,
        default=DEFAULT_PROVIDER,
        help="LLM provider name (e.g., openrouter, ollama, local)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Specific LLM model name for the chosen provider",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output from grading steps",
    )
    parser.add_argument(
        "--basic-prompts",
        action="store_true",
        help="Use basic prompts (without detailed criteria definitions). Default is to use detailed prompts.",
    )

    parsed_args = parser.parse_args()
    if parsed_args.workers < 1:
        print("Error: Number of workers must be at least 1.")
        sys.exit(1)
    if parsed_args.num_essays == 0:
        parsed_args.num_essays = None

    main(parsed_args)
