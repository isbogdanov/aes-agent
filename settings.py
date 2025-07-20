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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR_NAME = "datasets"

DATASETS_PATH = os.path.join(PROJECT_ROOT, DATASETS_DIR_NAME)

DEFAULT_DATASET_FILE = "ielts_writing_dataset_task2.csv"

DEFAULT_DATASET_FILE_PATH = os.path.join(DATASETS_PATH, DEFAULT_DATASET_FILE)
