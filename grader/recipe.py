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

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import traceback

if TYPE_CHECKING:
    from .essay_grader import EssayGraderAgent


@dataclass
class GradingResult:
    predicted_score: Optional[str] = field(default=None)
    prompt_tokens: Optional[int] = field(default=None)
    completion_tokens: Optional[int] = field(default=None)
    total_tokens: Optional[int] = field(default=None)
    latency: Optional[float] = field(default=None)
    raw_llm_response: Optional[str] = field(default=None)
    error_message: Optional[str] = field(default=None)


class BaseRecipe:
    identifier: str = "base_recipe"

    def __init__(
        self,
        agent: "EssayGraderAgent",
        use_detailed_criteria_prompts: bool = True,
        debug: bool = False,
    ):
        self.agent = agent
        self.use_detailed_criteria_prompts = use_detailed_criteria_prompts
        self.debug = debug

    def grade(self, question: str, essay_text: str) -> GradingResult:
        raise NotImplementedError("Subclasses must implement the grade method.")


class MultiStepWithExamplesRecipe(BaseRecipe):
    identifier: str = "MultiStep+Ex"

    def grade(self, question: str, essay_text: str) -> GradingResult:
        result = GradingResult()
        try:
            multi_step_results, p_tokens, c_tokens, t_tokens, latency = (
                self.agent.grade_essay_multi_step(
                    question=question,
                    essay_text=essay_text,
                    use_examples=True,
                    use_detailed_criteria_prompts=self.use_detailed_criteria_prompts,
                    debug=self.debug,
                )
            )
            overall_info = multi_step_results.get("overall", {})
            predicted = overall_info.get("score")

            if predicted in ["Error", "Calculation Error", "Parsing Error", None]:
                result.error_message = (
                    f"Multi-step(+Ex) grading returned status: {predicted}"
                )
                result.predicted_score = None
            else:
                result.predicted_score = predicted
            result.prompt_tokens = p_tokens
            result.completion_tokens = c_tokens
            result.total_tokens = t_tokens
            result.latency = latency

        except Exception as e:
            print(
                f"Exception during {self.identifier} grading for question '{question[:50]}...': {e}"
            )
            traceback.print_exc()
            result.error_message = f"Exception in {self.identifier}: {str(e)}"
        return result


class MultiStepNoExamplesRecipe(BaseRecipe):
    identifier: str = "MultiStep-NoEx"

    def grade(self, question: str, essay_text: str) -> GradingResult:
        result = GradingResult()
        try:
            multi_step_results, p_tokens, c_tokens, t_tokens, latency = (
                self.agent.grade_essay_multi_step(
                    question=question,
                    essay_text=essay_text,
                    use_examples=False,
                    use_detailed_criteria_prompts=self.use_detailed_criteria_prompts,
                    debug=self.debug,
                )
            )
            overall_info = multi_step_results.get("overall", {})
            predicted = overall_info.get("score")

            if predicted in ["Error", "Calculation Error", "Parsing Error", None]:
                result.error_message = (
                    f"Multi-step(-NoEx) grading returned status: {predicted}"
                )
                result.predicted_score = None
            else:
                result.predicted_score = predicted

            result.prompt_tokens = p_tokens
            result.completion_tokens = c_tokens
            result.total_tokens = t_tokens
            result.latency = latency

        except Exception as e:
            print(
                f"Exception during {self.identifier} grading for question '{question[:50]}...': {e}"
            )
            traceback.print_exc()
            result.error_message = f"Exception in {self.identifier}: {str(e)}"
        return result


class SingleStepNoExamplesRecipe(BaseRecipe):
    identifier: str = "SingleStep-NoEx"

    def grade(self, question: str, essay_text: str) -> GradingResult:
        result = GradingResult()
        try:
            parsed_score, raw_text, p_tokens, c_tokens, t_tokens, latency = (
                self.agent.grade_essay_single_direct_poc(
                    question=question,
                    essay_text=essay_text,
                    use_detailed_criteria_prompts=self.use_detailed_criteria_prompts,
                    debug=self.debug,
                )
            )
            result.raw_llm_response = raw_text

            if parsed_score is None or (
                isinstance(parsed_score, str) and "Error" in parsed_score
            ):
                result.error_message = (
                    f"Single-step(-NoEx) grading returned: {parsed_score}"
                )
                result.predicted_score = None
            else:
                result.predicted_score = parsed_score

            result.prompt_tokens = p_tokens
            result.completion_tokens = c_tokens
            result.total_tokens = t_tokens
            result.latency = latency

        except Exception as e:
            print(
                f"Exception during {self.identifier} grading for question '{question[:50]}...': {e}"
            )
            traceback.print_exc()
            result.error_message = f"Exception in {self.identifier}: {str(e)}"
        return result


class SingleStepWithExamplesRecipe(BaseRecipe):
    identifier: str = "SingleStep+Ex"

    def grade(self, question: str, essay_text: str) -> GradingResult:
        result = GradingResult()
        try:
            parsed_score, raw_text, p_tokens, c_tokens, t_tokens, latency = (
                self.agent.grade_essay_single_direct_with_examples_poc(
                    question=question,
                    essay_text=essay_text,
                    use_detailed_criteria_prompts=self.use_detailed_criteria_prompts,
                    debug=self.debug,
                )
            )
            result.raw_llm_response = raw_text

            if parsed_score is None or (
                isinstance(parsed_score, str) and "Error" in parsed_score
            ):
                result.error_message = (
                    f"Single-step(+Ex) grading returned: {parsed_score}"
                )
                result.predicted_score = None
            else:
                result.predicted_score = parsed_score

            result.prompt_tokens = p_tokens
            result.completion_tokens = c_tokens
            result.total_tokens = t_tokens
            result.latency = latency

        except Exception as e:
            print(
                f"Exception during {self.identifier} grading for question '{question[:50]}...': {e}"
            )
            traceback.print_exc()
            result.error_message = f"Exception in {self.identifier}: {str(e)}"
        return result
