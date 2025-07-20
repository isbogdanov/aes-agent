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

import re
import math
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.llm_connector.connector.connector import chat_completion
except ImportError:
    print(
        "Error: chat_completion not found. Please ensure the submodule structure is 'utils/llm_connector/connector/connector.py'."
    )

    def chat_completion(
        messages, provider, temperature, max_tokens, top_p, debug=False
    ):
        raise NotImplementedError("llm_connector.chat_completion is not available.")


from .prompts import *


class EssayGraderAgent:
    def __init__(
        self,
        provider: tuple[str, str],
        model_params: Optional[Dict[str, Any]] = None,
    ):
        if provider is None:
            raise ValueError("A provider tuple (name, model) is required.")
        self.provider = provider
        self.model_params = model_params if model_params is not None else {}
        self._current_prompt_tokens = 0
        self._current_completion_tokens = 0
        self._current_total_tokens = 0
        self._current_latency = 0.0

    def _prepare_messages(
        self,
        system_prompt: str,
        user_content: str,
    ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    _current_prompt_tokens: int
    _current_completion_tokens: int
    _current_total_tokens: int
    _current_latency: float

    def _get_llm_response(
        self,
        messages: List[Dict[str, str]],
        debug: bool = False,
        max_tokens_override: Optional[int] = None,
    ) -> Tuple[str, int, int, int, float]:
        response_text = f"Error: LLM call did not execute"
        p_tokens, c_tokens, t_tokens, latency = 0, 0, 0, 0.0

        temperature = self.model_params.get("temperature", 0.2)
        max_tokens = self.model_params.get("max_tokens", 200)
        top_p = self.model_params.get("top_p", 0.7)

        if max_tokens_override is not None:
            max_tokens = max_tokens_override
            if debug:
                print(f"    _get_llm_response: max_tokens overridden to: {max_tokens}")

        try:
            response_text, p_tokens, c_tokens, t_tokens, latency = chat_completion(
                messages=messages,
                provider=self.provider,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                debug=debug,
            )
            self._current_prompt_tokens += p_tokens
            self._current_completion_tokens += c_tokens
            self._current_total_tokens += t_tokens
            self._current_latency += latency
            return response_text, p_tokens, c_tokens, t_tokens, latency

        except Exception as e:
            print(f"Error during LLM call: {e}")
            self._current_latency += latency
            return f"Error communicating with LLM: {e}", 0, 0, 0, latency

    def _parse_criterion_score_only(self, llm_response: str) -> Optional[str]:
        return self._parse_direct_score(llm_response)

    def _parse_direct_score(self, llm_response: str) -> Optional[str]:
        try:
            match = re.search(r"(\d+(?:\.\d+)?)", llm_response)
            if match:
                score_str = match.group(1)
                score_float = float(score_str)
                if 1.0 <= score_float <= 9.0:
                    if score_float * 10 % 5 == 0:
                        return score_str
                print(
                    f"Warning: Parsed score '{score_str}' out of valid IELTS range/format."
                )
                return None
            else:
                print(
                    f"Warning: Could not parse score from direct response: {llm_response}"
                )
                return None
        except Exception as e:
            print(f"Error parsing direct score: {e}. Response: {llm_response}")
            return "Parsing Error"

    def _grade_criterion_step(
        self,
        question: str,
        essay_text: str,
        system_prompt: str,
        user_template: str,
        criterion_prefix: str,
        debug: bool = False,
    ) -> Tuple[Optional[str], int, int, int, float]:
        if not isinstance(question, str) or not isinstance(essay_text, str):
            error_msg = f"Type error in _grade_criterion_step for {criterion_prefix}: Question or Essay is not a string. Q: {type(question)}, E: {type(essay_text)}"
            print(error_msg)
            return "Input Error", 0, 0, 0, 0.0

        try:
            user_content = user_template.format(question=question, essay=essay_text)
        except KeyError as ke:
            print(
                f"KeyError in _grade_criterion_step for {criterion_prefix} formatting user_template: {ke}"
            )
            return "Template Error", 0, 0, 0, 0.0
        except Exception as fmt_e:
            print(
                f"Unexpected error formatting user_template for {criterion_prefix}: {fmt_e}"
            )
            return "Format Error", 0, 0, 0, 0.0

        messages = self._prepare_messages(system_prompt, user_content)
        if debug:
            print(f"--- Grading Step (Score Only): {criterion_prefix} --- ")

        llm_response_text, p_tokens, c_tokens, t_tokens, latency = (
            self._get_llm_response(messages, debug, max_tokens_override=10)
        )

        essay_snippet = essay_text[:30].replace("\n", " ")
        print(
            f"AGENT_INTERNAL_RAW_RESPONSE ({criterion_prefix} for essay_text starting with '{essay_snippet}...'): '{llm_response_text}'"
        )

        if debug:
            print(
                f"LLM Response for {criterion_prefix} (Score Only): {llm_response_text}"
            )
            print(
                f"Tokens (P/C/T): {p_tokens}/{c_tokens}/{t_tokens}, Latency: {latency:.3f}s"
            )

        if llm_response_text.startswith("Error communicating with LLM"):
            return "LLM Error", p_tokens, c_tokens, t_tokens, latency

        score = self._parse_criterion_score_only(llm_response_text)
        return score, p_tokens, c_tokens, t_tokens, latency

    def _calculate_and_round_overall_score(
        self, scores: List[Optional[str]]
    ) -> Optional[float]:
        valid_scores = []
        for s_str in scores:
            if s_str is None or s_str in [
                "Not found",
                "Parsing Error",
                "LLM Error",
                "Input Error",
                "Template Error",
                "Format Error",
            ]:
                return None
            try:
                num_match = re.search(r"(\d+(?:\.\d+)?)", s_str)
                if num_match:
                    valid_scores.append(float(num_match.group(1)))
                else:
                    print(
                        f"Warning: Could not convert score '{s_str}' to float for mean calculation."
                    )
                    return None
            except ValueError:
                print(
                    f"Warning: Could not convert score '{s_str}' to float for mean calculation."
                )
                return None
        if not valid_scores or len(valid_scores) < 4:
            return None
        mean_score = sum(valid_scores) / len(valid_scores)
        return round(mean_score * 2) / 2

    def grade_essay_multi_step(
        self,
        question: str,
        essay_text: str,
        use_examples: bool = True,
        use_detailed_criteria_prompts: bool = True,
        debug: bool = False,
    ) -> Tuple[Dict[str, Any], int, int, int, float]:
        self._current_prompt_tokens = 0
        self._current_completion_tokens = 0
        self._current_total_tokens = 0
        self._current_latency = 0.0

        results = {}
        component_scores_for_mean = []
        user_template = CRITERION_USER_TEMPLATE_SCORE_ONLY

        question_snippet = question[:30].replace("\n", " ")

        if use_detailed_criteria_prompts:
            tr_system = (
                TR_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_EX
                if use_examples
                else TR_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_NOEX
            )
        else:
            tr_system = (
                TR_SYSTEM_PROMPT_SCORE_ONLY_BASIC_EX
                if use_examples
                else TR_SYSTEM_PROMPT_SCORE_ONLY_BASIC_NOEX
            )
        tr_score_str, _, _, _, _ = self._grade_criterion_step(
            question, essay_text, tr_system, user_template, "TR", debug
        )
        results["tr"] = {
            "score": tr_score_str,
            "justification": "",
        }
        component_scores_for_mean.append(tr_score_str)

        if use_detailed_criteria_prompts:
            cc_system = (
                CC_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_EX
                if use_examples
                else CC_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_NOEX
            )
        else:
            cc_system = (
                CC_SYSTEM_PROMPT_SCORE_ONLY_BASIC_EX
                if use_examples
                else CC_SYSTEM_PROMPT_SCORE_ONLY_BASIC_NOEX
            )
        cc_score_str, _, _, _, _ = self._grade_criterion_step(
            question, essay_text, cc_system, user_template, "CC", debug
        )
        results["cc"] = {"score": cc_score_str, "justification": ""}
        component_scores_for_mean.append(cc_score_str)

        if use_detailed_criteria_prompts:
            lr_system = (
                LR_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_EX
                if use_examples
                else LR_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_NOEX
            )
        else:
            lr_system = (
                LR_SYSTEM_PROMPT_SCORE_ONLY_BASIC_EX
                if use_examples
                else LR_SYSTEM_PROMPT_SCORE_ONLY_BASIC_NOEX
            )
        lr_score_str, _, _, _, _ = self._grade_criterion_step(
            question, essay_text, lr_system, user_template, "LR", debug
        )
        results["lr"] = {"score": lr_score_str, "justification": ""}
        component_scores_for_mean.append(lr_score_str)

        if use_detailed_criteria_prompts:
            gra_system = (
                GRA_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_EX
                if use_examples
                else GRA_SYSTEM_PROMPT_SCORE_ONLY_DETAIL_NOEX
            )
        else:
            gra_system = (
                GRA_SYSTEM_PROMPT_SCORE_ONLY_BASIC_EX
                if use_examples
                else GRA_SYSTEM_PROMPT_SCORE_ONLY_BASIC_NOEX
            )
        gra_score_str, _, _, _, _ = self._grade_criterion_step(
            question, essay_text, gra_system, user_template, "GRA", debug
        )
        results["gra"] = {"score": gra_score_str, "justification": ""}
        component_scores_for_mean.append(gra_score_str)

        calculated_overall_score = self._calculate_and_round_overall_score(
            component_scores_for_mean
        )
        results["overall"] = {
            "score": (
                str(calculated_overall_score)
                if calculated_overall_score is not None
                else "Calculation Error"
            ),
            "justification": "Programmatically calculated; no LLM justification for overall score in this mode.",
        }

        return (
            results,
            self._current_prompt_tokens,
            self._current_completion_tokens,
            self._current_total_tokens,
            self._current_latency,
        )

    def grade_essay_single_direct_poc(
        self,
        question: str,
        essay_text: str,
        use_detailed_criteria_prompts: bool = True,
        debug: bool = False,
    ) -> Tuple[Optional[str], str, int, int, int, float]:
        self._current_prompt_tokens = 0
        self._current_completion_tokens = 0
        self._current_total_tokens = 0
        self._current_latency = 0.0
        user_content = SIMPLE_OVERALL_USER_TEMPLATE.format(
            question=question, essay=essay_text
        )

        system_prompt = (
            SIMPLE_OVERALL_SYSTEM_PROMPT_DETAIL
            if use_detailed_criteria_prompts
            else SIMPLE_OVERALL_SYSTEM_PROMPT_BASIC
        )

        messages = self._prepare_messages(system_prompt, user_content)
        if debug:
            print(
                f"--- Grading Step: Single Direct POC (Detailed Prompts: {use_detailed_criteria_prompts}) ---"
            )
        essay_snippet_noex = essay_text[:30].replace("\n", " ")
        llm_response_text, p, c, t, lat = self._get_llm_response(
            messages, debug, max_tokens_override=10
        )

        if debug:
            print(
                f"LLM Raw Response (Single Direct POC - No Examples): {llm_response_text}"
            )
            print(f"Tokens (P/C/T): {p}/{c}/{t}, Latency: {lat:.3f}s")
        if llm_response_text.startswith("Error"):
            return f"LLM Error: {llm_response_text}", llm_response_text, p, c, t, lat
        parsed_score = self._parse_direct_score(llm_response_text)
        return parsed_score, llm_response_text, p, c, t, lat

    def grade_essay_single_direct_with_examples_poc(
        self,
        question: str,
        essay_text: str,
        use_detailed_criteria_prompts: bool = True,
        debug: bool = False,
    ) -> Tuple[Optional[str], str, int, int, int, float]:
        self._current_prompt_tokens = 0
        self._current_completion_tokens = 0
        self._current_total_tokens = 0
        self._current_latency = 0.0
        user_content = SIMPLE_OVERALL_USER_TEMPLATE.format(
            question=question, essay=essay_text
        )

        system_prompt = (
            SIMPLE_OVERALL_WITH_EXAMPLES_SYSTEM_PROMPT_DETAIL
            if use_detailed_criteria_prompts
            else SIMPLE_OVERALL_WITH_EXAMPLES_SYSTEM_PROMPT_BASIC
        )

        messages = self._prepare_messages(system_prompt, user_content)
        if debug:
            print(
                f"--- Grading Step: Single Direct WITH EXAMPLES POC (Detailed Prompts: {use_detailed_criteria_prompts}) ---"
            )
        essay_snippet_ex = essay_text[:30].replace("\n", " ")
        llm_response_text, p, c, t, lat = self._get_llm_response(
            messages, debug, max_tokens_override=10
        )

        if debug:
            print(
                f"LLM Raw Response (Single Direct WITH EXAMPLES POC): {llm_response_text}"
            )
            print(f"Tokens (P/C/T): {p}/{c}/{t}, Latency: {lat:.3f}s")
        if llm_response_text.startswith("Error"):
            return f"LLM Error: {llm_response_text}", llm_response_text, p, c, t, lat
        parsed_score = self._parse_direct_score(llm_response_text)
        return parsed_score, llm_response_text, p, c, t, lat


if __name__ == "__main__":
    print("EssayGraderAgent module loaded. For usage, see run_essay_grader.py.")

    try:
        agent = EssayGraderAgent(("openrouter", "google/gemini-2.5-flash-preview"))
        print(
            f"Agent initialized with provider: {agent.provider} and params: {agent.model_params}"
        )

        test_question = "Discuss the impact of social media."
        test_essay = "Social media has changed the world..."
        test_system_prompt = "You are a grader."
        test_user_template = "Question: {question}\nEssay: {essay}\nGrade this."

        print("\nSimulating one-shot call structure (no actual LLM call here):")
        messages = agent._prepare_messages(
            test_system_prompt,
            test_user_template.format(question=test_question, essay=test_essay),
        )

    except Exception as e:
        print(f"Error during basic agent test: {e}")

    print("\nTo run full grading tests, execute run_essay_grader.py.")
