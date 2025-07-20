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

from typing import Optional
import numpy as np

DEFAULT_LOW_REWARD = -10.0
CATASTROPHIC_LOW_REWARD = -20.0


def calculate_reward(
    predicted_score_str: Optional[str],
    human_score_str: str,
    total_tokens: int = 0,
    token_penalty_weight: float = 0.0,
) -> float:
    accuracy_reward: float

    if predicted_score_str is None:
        accuracy_reward = DEFAULT_LOW_REWARD
    else:
        try:
            predicted_float = float(predicted_score_str)
            human_float = float(human_score_str)

            if not (1.0 <= predicted_float <= 9.0 and predicted_float * 10 % 5 == 0):
                accuracy_reward = DEFAULT_LOW_REWARD
            elif not (1.0 <= human_float <= 9.0 and human_float * 10 % 5 == 0):
                accuracy_reward = CATASTROPHIC_LOW_REWARD
            else:
                accuracy_reward = -abs(predicted_float - human_float)

        except (ValueError, TypeError) as e:
            accuracy_reward = DEFAULT_LOW_REWARD

    token_penalty = 0.0
    if token_penalty_weight > 0 and total_tokens > 0:
        token_penalty = token_penalty_weight * total_tokens

    shaped_reward = accuracy_reward - token_penalty
    return shaped_reward
