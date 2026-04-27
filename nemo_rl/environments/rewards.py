# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import re
from typing import Callable, Optional

import numpy as np
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from mathruler.grader import extract_boxed_content, grade_answer

# initialize math_verify_func once
math_verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    ),
)

boxed = lambda x: "\\boxed{" + x + "}" if not x.startswith("\\boxed{") else x


def math_expression_reward(
    ground_truth: str, response: str, tag: str = "answer"
) -> tuple[float, bool]:
    """Reward the agent when the answer within the <{tag}> tags is the same expression as the ground truth.

    The `tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{tag}>([\s\S]*)</{tag}>", response)
    if match:
        answer = match.group(1)
        try:
            score, _ = math_verify_func([boxed(ground_truth)], [boxed(answer)])
            return float(score), score > 0.1
        except (Exception, TimeoutException) as e:
            return 0.0, False
    return 0.0, False


def format_reward(
    ground_truth: str,
    response: str,
    think_tag: str = "think",
    answer_tag: str = "answer",
) -> tuple[float, Optional[bool]]:
    """Reward the agent when the response follows the format: (.*) <think> (.*) </think> <answer> (.*) </answer>.

    The `think_tag` and `answer_tag` are customizable and must be specified as part of the user COT prompt text file.
    """
    rew = 0.0
    if re.search(rf"<{think_tag}>[\s\S]*</{think_tag}>", response):
        rew += 0.25  # 0.25 points for having think tags
    if re.search(rf"<{answer_tag}>[\s\S]*</{answer_tag}>", response):
        rew += 0.75  # 0.75 points for having answer tags
    return rew, None


def exact_answer_alphanumeric_reward(
    ground_truth: str, response: str, answer_tag: str = "answer"
) -> tuple[float, bool]:
    """Reward the agent when the answer within the <{answer_tag}> tags is the same as the ground truth (case-insensitive).

    The `answer_tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
        # Remove all non-alphanumeric characters (including whitespace, punctuation, etc.)
        answer_clean = "".join(c for c in answer if c.isalnum()).lower()
        ground_truth_clean = "".join(c for c in ground_truth if c.isalnum()).lower()
        if answer_clean == ground_truth_clean:
            return 1.0, True
    return 0.0, False


def bbox_giou_reward(
    ground_truth: str,
    response: str,
    giou_penalty_thres: float = 10.0,
    answer_tag: str = "answer",
) -> tuple[float, bool]:
    """Given [x1, y1, x2, y2] normalized bounding box coordinates within the <{answer_tag}> tags, compute the GIoU between the ground truth and the response.

    The `answer_tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
    else:
        return 0.0, False

    try:
        x1g, y1g, x2g, y2g = [
            float(x) for x in ground_truth.replace("[", "").replace("]", "").split(",")
        ]
        x1r, y1r, x2r, y2r = [
            float(x) for x in answer.replace("[", "").replace("]", "").split(",")
        ]
    except ValueError:
        return 0.0, False

    # compute iou function
    # compute the area of the ground truth and response bounding boxes
    area_g = (x2g - x1g) * (y2g - y1g)
    area_r = (x2r - x1r) * (y2r - y1r)
    # compute the intersection of the ground truth and response bounding boxes
    x1i = max(x1g, x1r)
    y1i = max(y1g, y1r)
    x2i = min(x2g, x2r)
    y2i = min(y2g, y2r)
    # compute the area of the intersection
    area_i = max(0.0, x2i - x1i) * max(0.0, y2i - y1i)
    # compute the area of the union
    area_u = max(1e-3, area_g + area_r - area_i)
    # compute the iou
    iou = area_i / area_u
    # if iou is too low, introduce a giou term to compensate
    if iou < giou_penalty_thres:
        # compute convex hull as min
        x1c = min(x1g, x1r)
        y1c = min(y1g, y1r)
        x2c = max(x2g, x2r)
        y2c = max(y2g, y2r)
        # compute the area of the convex hull
        area_c = max(1e-3, (x2c - x1c) * (y2c - y1c))
        # compute the giou
        giou = iou - (area_c - area_u) / area_c
    else:
        giou = iou
    return giou, giou > 0.5


def combine_reward_functions(
    reward_functions: list[tuple[Callable[[str, str], tuple[float, bool]], float]],
) -> Callable[[str, str], tuple[float, bool]]:
    """Returns a callable function that takes (ground_truth, response) and collects multiple reward functions in sequence.

    The reward functions are weighted by the second element of the tuple.
    This information can be provided in the YAML config file and resolved in the VLMEnvironment class.

    Args:
        reward_functions: list[tuple[Callable[[str, str], tuple[float, bool]], float]]. A list of reward functions and their weights.

    Returns:
        Callable[[str, str], tuple[float, bool]]: A callable function that takes (ground_truth, response) and collects multiple reward functions in sequence
    """
    weights = [weight for _, weight in reward_functions]
    weights = np.array(weights) / np.sum(weights)  # renormalize weights to 1

    def combined_reward_func(ground_truth: str, response: str) -> tuple[float, bool]:
        reward_env_score = [
            reward_func(ground_truth, response) for reward_func, _ in reward_functions
        ]
        rewards = [x[0] for x in reward_env_score]
        is_correct = [
            x[1] for x in reward_env_score if x[1] is not None
        ]  # skip None values, because they do not contribute to the "correctness" of the response (e.g. format_reward, because the answer can still be correct without <think> tags)
        is_correct = all(is_correct)
        return np.sum(np.array(rewards) * weights), is_correct

    return combined_reward_func

def vision_r1_reward(
    ground_truth: str,
    response: str,
    format_reward: float = 0.5,
    correct_reward: float = 1.0,
) -> tuple[float, bool]:
    """Reward function for the Vision-R1 task.

    Inputs:
        - ground_truth: str - The correct answer (e.g., "A", "42", or a mathematical expression).
        - response: str - The model's response containing <answer> tags.
        - format_reward: float - Reward for correct format but incorrect answer (default 0.5).
        - correct_reward: float - Reward for correct answer (default 1.0).

    Outputs:
        - tuple[float, bool]: (reward_score, is_correct)
            - reward_score: float - The computed reward.
            - is_correct: bool - True if the answer is correct, False otherwise.

    Based on empirical analysis of Vision-R1 answers:
    - 66% multiple choice (A-E only)
    - 30% integers/decimals
    - 4% mathematical expressions
    """
    # Extract answer from <answer> tags
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if matches:
        # Take the last match if multiple answers
        final_answer = matches[-1].group(1).strip()
    else:
        # Did not format the answer correctly
        return 0.0, False

    golden_answer = str(ground_truth).strip()

    # Special handling for multiple choice (A-E only)
    if golden_answer.upper() in ['A', 'B', 'C', 'D', 'E']:
        if final_answer.upper() == golden_answer.upper():
            return correct_reward, True
        else:
            # For multiple choice, we're strict - must be exact letter match
            return format_reward, False

    # For all other answers, use math_verify
    try:
        from math_verify import parse, verify
        parsed_answer = parse(final_answer)
        correct_answer = verify(golden_answer, parsed_answer)
        if correct_answer:
            return correct_reward, True
    except Exception:
        # math_verify couldn't parse - try exact string match as fallback
        if final_answer.lower() == golden_answer.lower():
            return correct_reward, True

    return format_reward, False


def verl_geo3k_reward(
    ground_truth: str,
    response: str,
    format_score: float = 0.1,
) -> tuple[float, bool]:
    """Reward function for MMPR-Tiny task following verl's geo3k implementation.
    
    Exact replication of: https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/geo3k.py
    
    Args:
        ground_truth: The correct answer
        response: Model's complete response (with <think> and \\boxed{})
        format_score: Weight for format check (default 0.1 = 10%)
    
    Returns:
        (reward, is_correct) where reward = (1-format_score)*accuracy + format_score*format
    """
    # Format check (relaxed to accept missing opening <think> tag)
    # Original verl regex used fullmatch: r"<think>.*</think>.*\\boxed\{.*\}.*"
    # Modified to use search() to handle nested braces like \boxed{\dfrac{3}{20}}
    # Only requires </think> tag (not opening <think>) and \boxed{} with closing brace
    format_pattern = re.compile(r"</think>.*\\boxed\{.*\}", re.DOTALL)
    has_format = bool(re.search(format_pattern, response))
    format_reward_value = 1.0 if has_format else 0.0
    
    # Accuracy check (verl's exact approach)
    try:
        answer = extract_boxed_content(response)
        is_correct = grade_answer(answer, ground_truth)
        acc_reward_value = 1.0 if is_correct else 0.0
    except Exception as e:
        print("=" * 80)
        print("MATHRULER FAILED")
        print("=" * 80)
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"Ground truth: '{ground_truth}'")
        print(f"Response snippet: {response[:500]}...")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        print("=" * 80)
        acc_reward_value = 0.0
        is_correct = False
    
    # Weighted combination (verl's exact formula)
    final_reward = (1.0 - format_score) * acc_reward_value + format_score * format_reward_value

    return final_reward, is_correct


def progressive_geo3k_reward(
    ground_truth: str,
    response: str,
    format_score: float = 0.1,
) -> tuple[float, bool]:
    r"""Progressive reward function for MMPR-Tiny task.

    Gives partial reward if there are multiple \\boxed{} answers.

    Args:
        ground_truth: The correct answer
        response: Model's complete response (with <think> and \\boxed{})
        format_score: Weight for format check (default 0.1 = 10%)

    Returns:
        (reward, is_correct) where reward = (1-format_score)*accuracy + format_score*format
    """
    # Format check
    if "</think>" in response and "<think>" not in response:
        # XXX nano-v2 tokenizer adds <think> in reasoning mode, but that is omitted from prediction
        response = "<think>\n" + response
    think_blocks = list(re.finditer(r"<think>.*?</think>", response, re.DOTALL))
    if think_blocks:
        # remove first think block
        response = response[think_blocks[0].end() :].strip()

    boxed_answers = extract_all_boxed(response)
    format_reward_value = 1.0 if len(think_blocks) == 1 and len(boxed_answers) == 1 else 0.0

    # Accuracy check
    if boxed_answers:
        grades = []
        for boxed_answer in boxed_answers[:5]:
            try:
                ok = grade_answer(boxed_answer, ground_truth)
                grades.append(1.0 if ok else 0.0)
            except Exception:
                logging.exception("Mathruler failed to grade answer: %s", boxed_answer)
                grades.append(0.0)
        if len(boxed_answers) > 5:
            grades.extend([0.0] * (len(boxed_answers) - 5))
        acc_reward_value = float(np.mean(grades))
        is_correct = any(x > 0.0 for x in grades)
    else:
        acc_reward_value = 0.0
        is_correct = False

    # penalize 10% for any additional syntax error
    extra_thinks = response.count("<think>") + response.count("</think>")
    extra_boxeds = max(0, len(boxed_answers) - 1)
    acc_reward_value = acc_reward_value * (0.9 ** (extra_thinks + extra_boxeds))

    # Weighted combination (verl's exact formula)
    final_reward = (1.0 - format_score) * acc_reward_value + format_score * format_reward_value
    return final_reward, is_correct


def extract_all_boxed(text: str) -> list[str]:
    if "\\boxed{" not in text:
        return []
    results = []
    # require that boxed can't be nested
    parts = text.split("\\boxed{")[1:]
    for part in parts:
        depth = 1
        for i, char in enumerate(part):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                results.append(part[:i])
                break
        # if parens are not balanced, the answer is ignored
    return results
