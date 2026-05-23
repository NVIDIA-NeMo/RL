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
from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.games.twenty_four_game import _safe_eval_expr
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

STOP_STRINGS = ["</tool_call>", "</answer>"]


class CalculatorMetadata(TypedDict):
    target_answer: float
    num_tool_calls: int
    max_tool_calls: int


def _parse_tool_call(text: str) -> Optional[str]:
    """Extract the expression from the last <tool_call>...</tool_call> block."""
    match = re.search(
        r'<tool_call>\s*\{\s*"name"\s*:\s*"calculate"\s*,\s*"arguments"\s*:\s*\{\s*"expression"\s*:\s*"([^"]+)"\s*\}\s*\}\s*</tool_call>',
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


def _parse_answer(text: str) -> Optional[str]:
    """Extract the answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class CalculatorToolRunner:
    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: CalculatorMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[CalculatorMetadata],
        Optional[list[str]],
    ]:
        """Process a single turn for the calculator tool-call task."""
        target_answer = metadata["target_answer"]
        num_tool_calls = metadata["num_tool_calls"]
        max_tool_calls = metadata["max_tool_calls"]

        last_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_content = message_log[-1]["content"].strip()

        # Check for tool call
        tool_expr = _parse_tool_call(last_content)
        if tool_expr is not None:
            if num_tool_calls >= max_tool_calls:
                return (
                    {
                        "role": "environment",
                        "content": f"<result>Error: Maximum tool calls ({max_tool_calls}) exceeded.</result>\nPlease give your final answer in <answer></answer> tags.",
                    },
                    0.0,
                    True,
                    None,
                    None,
                    None,
                )

            result = _safe_eval_expr(tool_expr)
            if result is not None:
                result_str = f"{result:g}"
            else:
                result_str = "Error: Invalid expression"

            next_metadata: CalculatorMetadata = {
                "target_answer": target_answer,
                "num_tool_calls": num_tool_calls + 1,
                "max_tool_calls": max_tool_calls,
            }
            return (
                {
                    "role": "environment",
                    "content": f"<result>{result_str}</result>\nYou can make more calculations or give your final answer in <answer></answer> tags.",
                },
                0.0,
                False,
                STOP_STRINGS,
                next_metadata,
                None,
            )

        # Check for final answer
        answer_str = _parse_answer(last_content)
        if answer_str is not None:
            try:
                answer_val = float(answer_str)
            except ValueError:
                return (
                    {"role": "environment", "content": ""},
                    0.0,
                    True,
                    None,
                    None,
                    [answer_str],
                )

            reward = 1.0 if abs(answer_val - target_answer) < 0.01 else 0.0
            return (
                {"role": "environment", "content": ""},
                reward,
                True,
                None,
                None,
                [answer_str],
            )

        # Neither tool call nor answer — nudge the model
        next_metadata = {
            "target_answer": target_answer,
            "num_tool_calls": num_tool_calls,
            "max_tool_calls": max_tool_calls,
        }
        return (
            {
                "role": "environment",
                "content": 'Please use <tool_call>{"name":"calculate","arguments":{"expression":"EXPR"}}</tool_call> to compute, or <answer>NUMBER</answer> to give your final answer.',
            },
            0.0,
            False,
            STOP_STRINGS,
            next_metadata,
            None,
        )


@ray.remote  # pragma: no cover
class CalculatorToolEnv(EnvironmentInterface[CalculatorMetadata]):
    """Calculator tool-call environment (Ray Actor)."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.runner = CalculatorToolRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CalculatorMetadata],
    ) -> EnvironmentReturn[CalculatorMetadata]:
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata)
        ]

        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []
        all_answers = []

        for obs, rew, term, stops, meta, answ in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)
            all_answers.append(answ)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        accuracy = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        return batch, {"calculator_tool_accuracy": accuracy}
