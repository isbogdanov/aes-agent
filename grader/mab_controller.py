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

import random
import threading
from typing import List, Dict, Tuple


class EpsilonGreedyMAB:
    def __init__(self, recipe_ids: List[str], epsilon: float = 0.1):
        if not (0 <= epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1.")
        if not recipe_ids:
            raise ValueError("Recipe IDs list cannot be empty.")
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("Recipe IDs must be unique.")

        self.recipe_ids = recipe_ids
        self.epsilon = epsilon
        self._lock = threading.Lock()

        self.arm_counts: Dict[str, int] = {recipe_id: 0 for recipe_id in recipe_ids}
        self.arm_values: Dict[str, float] = {recipe_id: 0.0 for recipe_id in recipe_ids}

        print(
            f"Initialized EpsilonGreedyMAB with {len(recipe_ids)} arms and epsilon={epsilon}"
        )

    def select_arm(self) -> str:
        with self._lock:
            if random.random() < self.epsilon:
                chosen_arm = random.choice(self.recipe_ids)
            else:
                max_value = -float("inf")
                if all(v == 0.0 for v in self.arm_values.values()) and all(
                    c == 0 for c in self.arm_counts.values()
                ):
                    chosen_arm = random.choice(self.recipe_ids)
                else:
                    for value in self.arm_values.values():
                        if value > max_value:
                            max_value = value
                    best_arms = [
                        arm
                        for arm, value in self.arm_values.items()
                        if value == max_value
                    ]
                    chosen_arm = random.choice(best_arms)
        return chosen_arm

    def update(self, chosen_recipe_id: str, reward: float):
        if chosen_recipe_id not in self.recipe_ids:
            raise ValueError(f"Unknown recipe ID: {chosen_recipe_id}")

        with self._lock:
            self.arm_counts[chosen_recipe_id] += 1
            count = self.arm_counts[chosen_recipe_id]

            old_value = self.arm_values[chosen_recipe_id]
            new_value = old_value + (reward - old_value) / count
            self.arm_values[chosen_recipe_id] = new_value

    def get_best_arm(self) -> Tuple[str, float]:
        with self._lock:
            if all(v == 0.0 for v in self.arm_values.values()):
                first_arm = self.recipe_ids[0]
                return first_arm, self.arm_values[first_arm]

            best_arm_id = max(self.arm_values, key=self.arm_values.get)
            return best_arm_id, self.arm_values[best_arm_id]

    def get_state(self) -> Dict:
        with self._lock:
            return {
                "counts": self.arm_counts.copy(),
                "values": self.arm_values.copy(),
                "epsilon": self.epsilon,
                "recipe_ids": self.recipe_ids,
            }
