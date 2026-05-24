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

import re
from typing import TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class CodeTaskMetadata(TypedDict):
    test_cases: list[dict[str, str]]


def _extract_code(response: str) -> str | None:
    """Extract Python code from markdown code blocks or raw response."""
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "def " in response:
        return response.strip()
    return None


def _run_test_cases(code: str, test_cases: list[dict[str, str]]) -> bool:
    """Execute code and run all test cases. Returns True if all pass."""
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception:
        return False

    for tc in test_cases:
        try:
            result = eval(tc["input"], namespace)  # noqa: S307
            if str(result) != tc["expected"]:
                return False
        except Exception:
            return False
    return True


@ray.remote  # pragma: no cover
class CodeTaskEnvironment(EnvironmentInterface[CodeTaskMetadata]):
    """Single-turn code task environment.

    The model writes a Python function. The environment extracts the code,
    executes it against test cases, and returns 1.0 if all pass, 0.0 otherwise.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CodeTaskMetadata],
    ) -> EnvironmentReturn[CodeTaskMetadata]:
        rewards = []
        answers = []

        for message_log, meta in zip(message_log_batch, metadata):
            response = str(message_log[-1]["content"])
            test_cases = meta["test_cases"]
            code = _extract_code(response)

            if code and _run_test_cases(code, test_cases):
                reward = 1.0
            else:
                reward = 0.0

            rewards.append(reward)
            answers.append(code)

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
