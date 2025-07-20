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

CRITERION_USER_TEMPLATE_SCORE_ONLY = """
Essay Question:
'''
{question}
'''

Essay Text:
'''
{essay}
'''

Please evaluate the specified criterion and provide ONLY its band score.
"""

SIMPLE_OVERALL_USER_TEMPLATE = CRITERION_USER_TEMPLATE_SCORE_ONLY
