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
from typing import Optional


class MABLogger:
    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        self.file_handle = None
        self.csv_writer = None
        self._lock = threading.Lock()

        self.header = [
            "Timestamp",
            "Step",
            "EssayIndex",
            "ChosenRecipe",
            "HumanScore",
            "PredictedScore",
            "Reward",
            "Epsilon",
            "ErrorMessage",
            "PromptTokens",
            "CompletionTokens",
            "TotalTokens",
            "Latency",
            "EstimatedCost",
        ]

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
            print(f"Initialized new log file: {self.log_filepath}")

        except Exception as e:
            print(f"Error initializing MABLogger for file {self.log_filepath}: {e}")
            if self.file_handle:
                self.file_handle.close()
            raise

    def log_step(
        self,
        step: int,
        essay_index: int,
        chosen_recipe_id: str,
        human_score: str,
        predicted_score: Optional[str],
        reward: float,
        epsilon: float,
        error_message: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency: Optional[float] = None,
        estimated_cost: Optional[float] = None,
    ):
        if self.csv_writer is None:
            print("Error: Logger not initialized properly.")
            return

        timestamp = datetime.datetime.now().isoformat()
        pred_score_str = str(predicted_score) if predicted_score is not None else ""
        error_msg_str = str(error_message) if error_message is not None else ""
        p_tokens_str = str(prompt_tokens) if prompt_tokens is not None else "0"
        c_tokens_str = str(completion_tokens) if completion_tokens is not None else "0"
        t_tokens_str = str(total_tokens) if total_tokens is not None else "0"
        latency_str = f"{latency:.4f}" if latency is not None else "0.0000"
        cost_str = (
            f"{estimated_cost:.8f}" if estimated_cost is not None else "0.00000000"
        )

        try:
            self._lock.acquire()
            row = [
                timestamp,
                step,
                essay_index,
                chosen_recipe_id,
                human_score,
                pred_score_str,
                f"{reward:.4f}",
                f"{epsilon:.4f}",
                error_msg_str,
                p_tokens_str,
                c_tokens_str,
                t_tokens_str,
                latency_str,
                cost_str,
            ]
            self.csv_writer.writerow(row)
            self.file_handle.flush()
        except Exception as e:
            print(f"Error writing log step {step} to {self.log_filepath}: {e}")
        finally:
            self._lock.release()

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
