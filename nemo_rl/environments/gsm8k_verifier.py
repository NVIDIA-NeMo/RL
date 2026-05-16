# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Adapted from https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """Extract the numeric answer from a GSM8K-style solution string.

    Args:
        solution_str: The model's response string.
        method: 'strict' requires #### prefix or <answer> tags; 'flexible' finds any number.

    Returns:
        The extracted numeric answer string, or None if not found.
    """
    assert method in ["strict", "flexible"]

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # Try <answer> tags first (matches gsm8k.txt prompt format)
        xml_match = re.findall(r"<answer>\s*(\-?[0-9\.\,]+)\s*</answer>", solution_str)
        if xml_match:
            final_answer = xml_match[-1].replace(",", "").replace("$", "")
            return final_answer
        # Fall back to #### prefix (original GSM8K format)
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            return None
        final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answer) > 0:
            invalid_str = ["", "."]
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> dict:
    """Compute the reward score for a GSM8K solution.

    Args:
        solution_str: The model's response string.
        ground_truth: The ground truth numeric answer.
        method: 'strict' or 'flexible' extraction.
        format_score: Score when format is correct but answer is wrong.
        score: Score when answer is correct.

    Returns:
        Dict with 'score' and 'pred' keys.
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return {"score": 0.0, "pred": None}
    elif answer == ground_truth:
        return {"score": score, "pred": answer}
    else:
        return {"score": format_score, "pred": answer}
