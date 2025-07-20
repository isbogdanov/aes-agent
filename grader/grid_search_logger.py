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

import csv
import datetime
import os
import threading
from typing import Dict, List, Any
import numpy as np
import pandas as pd

METHOD_PREFIXES_GS = ["multi_ex", "multi_noex", "single_noex", "single_ex"]


class GridSearchLogger:
    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        self.file_handle = None
        self.csv_writer = None
        self._lock = threading.Lock()

        self.header = ["Timestamp", "EssayIndex", "HumanScore"]
        for prefix in METHOD_PREFIXES_GS:
            self.header.extend(
                [
                    f"{prefix}_PredictedScore",
                    f"{prefix}_err",
                    f"{prefix}_PromptTokens",
                    f"{prefix}_CompletionTokens",
                    f"{prefix}_TotalTokens",
                    f"{prefix}_Latency",
                    f"{prefix}_EstimatedCost",
                ]
            )
        self.header.extend(
            [
                "Essay_Agg_PromptTokens",
                "Essay_Agg_CompletionTokens",
                "Essay_Agg_TotalTokens",
                "Essay_Agg_TotalLatency",
                "Essay_Agg_EstimatedCost",
            ]
        )

        try:
            log_dir = os.path.dirname(self.log_filepath)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self.file_handle = open(
                self.log_filepath, "w", newline="", encoding="utf-8"
            )
            self.csv_writer = csv.writer(self.file_handle)
            self.csv_writer.writerow(self.header)
            self.file_handle.flush()
            print(f"Initialized new grid search log file: {self.log_filepath}")

        except Exception as e:
            print(
                f"Error initializing GridSearchLogger for file {self.log_filepath}: {e}"
            )
            if self.file_handle:
                self.file_handle.close()
            raise

    def log_essay_results(self, essay_results_dict: Dict[str, Any]):
        if self.csv_writer is None:
            print("Error: GridSearchLogger not initialized properly.")
            return

        timestamp = datetime.datetime.now().isoformat()
        row = [
            timestamp,
            essay_results_dict.get("index"),
            essay_results_dict.get("human"),
        ]

        for prefix in METHOD_PREFIXES_GS:
            err_val = essay_results_dict.get(f"{prefix}_err", np.nan)
            err_str = f"{err_val:.4f}" if not pd.isna(err_val) else "NaN"
            row.extend(
                [
                    str(essay_results_dict.get(f"{prefix}_pred", "")),
                    err_str,
                    str(essay_results_dict.get(f"{prefix}_prompt_tokens", 0)),
                    str(essay_results_dict.get(f"{prefix}_completion_tokens", 0)),
                    str(essay_results_dict.get(f"{prefix}_total_tokens", 0)),
                    f"{essay_results_dict.get(f'{prefix}_latency', 0.0):.4f}",
                    f"{essay_results_dict.get(f'{prefix}_estimated_cost', 0.0):.8f}",
                ]
            )

        total_prompt_tokens_val = essay_results_dict.get("total_prompt_tokens", 0)
        total_completion_tokens_val = essay_results_dict.get(
            "total_completion_tokens", 0
        )
        total_tokens_val = essay_results_dict.get("total_tokens", 0)
        total_latency_val = essay_results_dict.get("total_latency", 0.0)
        estimated_cost_val = essay_results_dict.get("estimated_cost", 0.0)

        row.extend(
            [
                str(total_prompt_tokens_val),
                str(total_completion_tokens_val),
                str(total_tokens_val),
                f"{total_latency_val:.4f}",
                f"{estimated_cost_val:.8f}",
            ]
        )

        try:
            with self._lock:
                self.csv_writer.writerow(row)
                self.file_handle.flush()
        except Exception as e:
            print(
                f"Error writing essay result (Index: {essay_results_dict.get('index')}) to {self.log_filepath}: {e}"
            )

    def close(self):
        if self.file_handle:
            try:
                self.file_handle.close()
                self.file_handle = None
                self.csv_writer = None
            except Exception as e:
                print(f"Error closing log file {self.log_filepath}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
