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
import random
import argparse
import datetime
import time
import threading
import queue
import numpy as np
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from settings import DEFAULT_DATASET_FILE_PATH
from grader.essay_grader import EssayGraderAgent
from grader.recipe import (
    MultiStepWithExamplesRecipe,
    MultiStepNoExamplesRecipe,
    SingleStepNoExamplesRecipe,
    SingleStepWithExamplesRecipe,
    GradingResult,
)
from grader.reward import calculate_reward
from grader.mab_controller import EpsilonGreedyMAB
from grader.mab_logger import MABLogger
from utils.llm_connector.connector.connector_settings import (
    MODEL_PRICING,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
)


def get_price(provider: str, model: str) -> tuple[float, float]:
    return MODEL_PRICING.get((provider, model), (0.0, 0.0))


stop_workers = False


def worker_function(
    worker_id: int,
    task_queue: queue.Queue,
    mab_controller: EpsilonGreedyMAB,
    logger: MABLogger,
    recipe_definitions: dict,
    essay_data: list,
    args: argparse.Namespace,
):
    print(f"Worker {worker_id} started.")
    provider_name = args.provider_name
    model_name = args.model_name

    try:
        agent = EssayGraderAgent(provider=(provider_name, model_name))
        worker_recipe_map = {
            rec_id: rec_class(
                agent,
                use_detailed_criteria_prompts=not args.basic_prompts,
                debug=args.debug,
            )
            for rec_id, rec_class in recipe_definitions.items()
        }
    except Exception as e:
        print(f"Worker {worker_id}: Failed to initialize agent or recipes: {e}")
        return
    while not stop_workers:
        try:
            step = task_queue.get(timeout=1.0)

            selected_essay_index_in_list = random.randrange(len(essay_data))
            original_csv_index, question, essay_text, human_score_str = essay_data[
                selected_essay_index_in_list
            ]

            chosen_recipe_id = mab_controller.select_arm()
            recipe_to_run = worker_recipe_map[chosen_recipe_id]
            predicted_score = None
            error_message = None
            p_tokens, c_tokens, t_tokens, latency = 0, 0, 0, 0.0
            try:
                grading_result: GradingResult = recipe_to_run.grade(
                    question, essay_text
                )
                predicted_score = grading_result.predicted_score
                error_message = grading_result.error_message
                p_tokens = grading_result.prompt_tokens or 0
                c_tokens = grading_result.completion_tokens or 0
                t_tokens = grading_result.total_tokens or 0
                latency = grading_result.latency or 0.0

            except Exception as grade_e:
                print(
                    f"Worker {worker_id}: Critical Error during grading step {step}, recipe {chosen_recipe_id}: {grade_e}"
                )
                error_message = f"Caught exception: {str(grade_e)}"

            shaped_reward = calculate_reward(
                predicted_score,
                human_score_str,
                total_tokens=t_tokens,
                token_penalty_weight=args.token_penalty,
            )

            price_in, price_out = get_price(provider_name, model_name)
            estimated_cost = 0.0
            if p_tokens > 0 and price_in > 0.0:
                estimated_cost += (p_tokens / 1_000_000) * price_in
            if c_tokens > 0 and price_out > 0.0:
                estimated_cost += (c_tokens / 1_000_000) * price_out

            mab_controller.update(chosen_recipe_id, shaped_reward)

            logger.log_step(
                step=step,
                essay_index=original_csv_index,
                chosen_recipe_id=chosen_recipe_id,
                human_score=human_score_str,
                predicted_score=predicted_score,
                reward=shaped_reward,
                epsilon=mab_controller.epsilon,
                error_message=error_message,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                latency=latency,
                estimated_cost=estimated_cost,
            )
            task_queue.task_done()

        except queue.Empty:
            if stop_workers:
                break
            else:
                time.sleep(0.5)
                continue
        except Exception as e:
            print(f"Worker {worker_id}: Unhandled exception in worker loop: {e}")
            task_queue.task_done()

    print(f"Worker {worker_id} finished.")


def main(args):
    print(
        f"--- Starting Concurrent MAB Experiment: {datetime.datetime.now().isoformat()} ---"
    )
    print(f"Args: {args}")
    global stop_workers
    stop_workers = False

    print(f"Loading dataset from: {args.dataset}")
    try:
        data_df_full = pd.read_csv(args.dataset)
        required_columns = ["Question", "Essay", "Overall"]
        if not all(col in data_df_full.columns for col in required_columns):
            print(f"Error: CSV missing one of required columns: {required_columns}")
            sys.exit(1)
        data_df = data_df_full[required_columns].copy()
        data_df.dropna(inplace=True)
        if data_df.empty:
            print("Error: No valid data rows found after cleaning.")
            sys.exit(1)
        essay_data = [
            (index, row["Question"], row["Essay"], str(row["Overall"]))
            for index, row in data_df.iterrows()
        ]
        print(f"Loaded and prepared {len(essay_data)} essays.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        sys.exit(1)

    print("Initializing components...")
    recipe_definitions = {
        MultiStepWithExamplesRecipe.identifier: MultiStepWithExamplesRecipe,
        MultiStepNoExamplesRecipe.identifier: MultiStepNoExamplesRecipe,
        SingleStepNoExamplesRecipe.identifier: SingleStepNoExamplesRecipe,
        SingleStepWithExamplesRecipe.identifier: SingleStepWithExamplesRecipe,
    }
    recipe_ids = list(recipe_definitions.keys())
    print(f"Defined {len(recipe_ids)} recipes: {recipe_ids}")

    mab_controller = EpsilonGreedyMAB(recipe_ids=recipe_ids, epsilon=args.epsilon)

    try:
        with MABLogger(args.log_file) as logger:
            print(f"Logger initialized. Logging to: {args.log_file}")

            task_queue = queue.Queue()
            threads = []
            num_workers = args.workers

            print(f"Starting {num_workers} worker threads...")
            for i in range(num_workers):
                thread = threading.Thread(
                    target=worker_function,
                    args=(
                        i + 1,
                        task_queue,
                        mab_controller,
                        logger,
                        recipe_definitions,
                        essay_data,
                        args,
                    ),
                    daemon=True,
                )
                threads.append(thread)
                thread.start()

            print(f"Populating task queue with {args.steps} steps...")
            start_time = time.time()
            for step_num in range(1, args.steps + 1):
                task_queue.put(step_num)

            print("Waiting for all tasks to complete...")
            task_queue.join()
            print("All tasks completed.")

            print("Signaling workers to stop...")
            stop_workers = True
            print("Waiting for worker threads to finish...")
            for thread in threads:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print(f"Warning: Thread {thread.ident} did not finish cleanly.")
            print("Worker threads finished.")

            print("\n--- Concurrent MAB Experiment Finished ---")
            end_time = time.time()
            print(f"Total execution time: {end_time - start_time:.2f} seconds")
            final_state = mab_controller.get_state()
            print("\nFinal Arm Counts:")
            for arm_id, count in final_state["counts"].items():
                print(f"  {arm_id}: {count}")
            print("\nFinal Average Rewards (Values):")
            sorted_values = sorted(
                final_state["values"].items(), key=lambda item: item[1], reverse=True
            )
            for arm_id, value in sorted_values:
                print(f"  {arm_id}: {value:.4f}")
            champion_arm, champion_value = mab_controller.get_best_arm()
            print(
                f"\nChampion Arm (Highest Average Reward): {champion_arm} (Value: {champion_value:.4f})"
            )
            print(f"Experiment log saved to: {args.log_file}")

    except Exception as e:
        print(f"An error occurred during the MAB experiment setup or loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Ensuring workers are signaled to stop...")
        stop_workers = True
        if "task_queue" in locals():
            for _ in range(args.workers):
                try:
                    task_queue.put(None, block=False)
                except queue.Full:
                    pass
        if "threads" in locals():
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)

    print("\n--- Analyzing Log File for Per-Arm Statistics ---")
    try:
        log_df = pd.read_csv(args.log_file)

        numeric_cols = [
            "PromptTokens",
            "CompletionTokens",
            "TotalTokens",
            "Latency",
            "EstimatedCost",
            "Reward",
        ]
        for col in numeric_cols:
            log_df[col] = pd.to_numeric(log_df[col], errors="coerce")

        total_prompt_tokens_mab = log_df["PromptTokens"].sum()
        total_completion_tokens_mab = log_df["CompletionTokens"].sum()
        total_tokens_logged_mab = log_df["TotalTokens"].sum()
        total_latency_mab = log_df["Latency"].sum()
        total_estimated_cost_mab = log_df["EstimatedCost"].sum()
        num_mab_steps_logged = len(log_df)

        print("\nOverall MAB Experiment Totals (Actual Operations):")
        print(f"  MAB Steps Logged:        {num_mab_steps_logged}")
        print(f"  Total Prompt Tokens:     {total_prompt_tokens_mab:,.0f}")
        print(f"  Total Completion Tokens: {total_completion_tokens_mab:,.0f}")
        print(f"  Total Tokens:            {total_tokens_logged_mab:,.0f}")
        print(f"  Total Latency (sum):     {total_latency_mab:.2f} seconds")
        print(f"  Total Estimated Cost:    ${total_estimated_cost_mab:.4f}")

        def calculate_accuracy_error(row):
            try:
                pred = float(row["PredictedScore"])
                human = float(row["HumanScore"])
                if not (1.0 <= pred <= 9.0 and pred * 10 % 5 == 0):
                    return np.nan
                if not (1.0 <= human <= 9.0 and human * 10 % 5 == 0):
                    return np.nan
                return abs(pred - human)
            except (ValueError, TypeError):
                return np.nan

        log_df["AccuracyError"] = log_df.apply(calculate_accuracy_error, axis=1)

        print("\nPer-Arm Statistics (from MAB Log):")
        final_state = mab_controller.get_state()

        arm_stats = log_df.groupby("ChosenRecipe").agg(
            MAE=("AccuracyError", "mean"),
            AvgShapedReward=("Reward", "mean"),
            AvgEstimatedCost=("EstimatedCost", "mean"),
            AvgPromptTokens=("PromptTokens", "mean"),
            AvgCompletionTokens=("CompletionTokens", "mean"),
            AvgTotalTokens=("TotalTokens", "mean"),
            AvgLatency=("Latency", "mean"),
        )
        arm_stats["PullCount"] = arm_stats.index.map(final_state["counts"])

        mab_display_cols = [
            "PullCount",
            "MAE",
            "AvgShapedReward",
            "AvgPromptTokens",
            "AvgCompletionTokens",
            "AvgTotalTokens",
            "AvgLatency",
            "AvgEstimatedCost",
        ]
        for col in mab_display_cols:
            if col not in arm_stats.columns:
                arm_stats[col] = 0.0
        arm_stats = arm_stats[mab_display_cols]

        arm_stats.sort_values(
            by=["MAE", "AvgShapedReward"], ascending=[True, False], inplace=True
        )

        header_approach = "Approach (Arm)"
        header_pulls = "Pulls"
        header_mae = "MAE"
        header_avg_shaped_reward = "AvgShapedRew"
        header_avg_prompt_tk = "AvgPromptTk"
        header_avg_compl_tk = "AvgComplTk"
        header_avg_total_tk = "AvgTotalTk"
        header_avg_latency = "AvgLatency"
        header_avg_cost = "AvgCost"

        print(
            f"{header_approach:<25} {header_pulls:>7} {header_mae:>7} {header_avg_shaped_reward:>12} "
            f"{header_avg_prompt_tk:>12} {header_avg_compl_tk:>12} {header_avg_total_tk:>12} "
            f"{header_avg_latency:>11} {header_avg_cost:>15}"
        )
        print("-" * 130)

        for arm_id, row_data in arm_stats.iterrows():
            approach_name = (
                arm_id.replace("_", " ")
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
                f"{approach_name_formatted:<25} "
                f"{int(row_data['PullCount']):>7} "
                f"{row_data['MAE']:>7.3f} "
                f"{row_data['AvgShapedReward']:>12.4f} "
                f"{row_data['AvgPromptTokens']:>12,.1f} "
                f"{row_data['AvgCompletionTokens']:>12,.1f} "
                f"{row_data['AvgTotalTokens']:>12,.1f} "
                f"{row_data['AvgLatency']:>10.3f}s "
                f"{row_data['AvgEstimatedCost']:>15.8f}"
            )

    except FileNotFoundError:
        print(f"Log file {args.log_file} not found for analysis.")
    except Exception as analysis_e:
        print(f"Error analyzing log file {args.log_file}: {analysis_e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Concurrent Multi-Armed Bandit Experiment for IELTS Essay Grading Recipes"
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
        "--log-file",
        type=str,
        default=None,
        help="Path to save the concurrent experiment log CSV file. If None, a filename is generated.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon value for Epsilon-Greedy exploration (0.0 to 1.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Total number of MAB steps (arm pulls) to run",
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
        "--token-penalty",
        type=float,
        default=0.0,
        help="Weight for penalizing total tokens in reward shaping (e.g., 0.0001). 0.0 to disable.",
    )
    parser.add_argument(
        "--basic-prompts",
        action="store_true",
        help="Use basic prompts (without detailed criteria definitions). Default is to use detailed prompts.",
    )

    parsed_args = parser.parse_args()

    if parsed_args.log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", parsed_args.model_name)
        prompt_type_str = "_basicP" if parsed_args.basic_prompts else "_detailP"
        parsed_args.log_file = (
            f"logs/mab_log_{parsed_args.provider_name}_{sanitized_model_name}_"
            f"{parsed_args.steps}s_{parsed_args.epsilon}eps_{parsed_args.token_penalty}pen{prompt_type_str}_{timestamp}.csv"
        )
        print(f"No log file specified, generated: {parsed_args.log_file}")

    if parsed_args.workers < 1:
        print("Error: Number of workers must be at least 1.")
        sys.exit(1)

    main(parsed_args)
