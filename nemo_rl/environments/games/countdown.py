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

"""Countdown Game environment for GRPO training.

Given a list of numbers and a target, combine numbers using +, -, *, /
to reach the target. Each number may be used at most once.
"""

import re
from collections import Counter
from typing import Optional

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.games.twenty_four_game import _safe_eval_expr
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


def _extract_numbers_from_expr(expr_str: str) -> list[int]:
    """Extract all integer literals from an expression string."""
    return [int(m) for m in re.findall(r"\d+", expr_str)]


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_answer(response: str) -> Optional[str]:
    """Extract expression from <answer>...</answer> tags in the response."""
    response = _strip_think_blocks(response)
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _check_countdown_expression(
    expr_str: str, numbers: list[int], target: int, tol: float = 1e-6
) -> bool:
    """Check if an expression is a valid Countdown solution.

    Validates that:
    1. Each number in the expression appears at most as many times as in the given numbers.
    2. The expression evaluates to the target.
    """
    used_numbers = _extract_numbers_from_expr(expr_str)
    available = Counter(numbers)
    used = Counter(used_numbers)
    for num, count in used.items():
        if count > available.get(num, 0):
            return False

    result = _safe_eval_expr(expr_str)
    if result is None:
        return False

    return abs(result - target) < tol


@ray.remote  # pragma: no cover
class CountdownEnv(EnvironmentInterface[dict]):
    """Countdown Game environment (Ray Actor).

    Single-turn: model receives a list of numbers and a target,
    must produce an expression using +, -, *, / that evaluates to the target.
    Each number may be used at most once.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn[dict]:
        rewards = []
        answers = []

        for message_log, meta in zip(message_log_batch, metadata):
            response = message_log[-1]["content"]
            numbers = meta["numbers"]
            target = meta["target"]

            extracted = _extract_answer(response)
            if extracted and _check_countdown_expression(extracted, numbers, target):
                reward = 1.0
            else:
                reward = 0.0

            rewards.append(reward)
            answers.append(extracted)

        batch_size = len(message_log_batch)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""}] * batch_size,
            metadata=[None] * batch_size,
            next_stop_strings=[None] * batch_size,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(batch_size, dtype=torch.bool),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get("total_reward", torch.tensor([0.0]))
        accuracy = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"accuracy": accuracy}
