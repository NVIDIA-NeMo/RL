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

from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class MultiStepCalcMetadata(TypedDict):
    problem: str
    expected_final_answer: float
    max_steps: int
    current_step: int


class _MultiStepCalculatorLogic:
    def __init__(self):
        pass

    def _parse_tool_call(self, text: str) -> Optional[tuple[float, float, str]]:
        """Parses '[opA, opB, operation]<call: calculator>'."""
        # Use a more distinct tool call suffix
        tool_call_suffix = "<call: calculator>"
        if not text.strip().endswith(tool_call_suffix):
            return None

        content = text.strip()[: -len(tool_call_suffix)].strip()
        if not (content.startswith("[") and content.endswith("]")):
            return None
        parts = content[1:-1].split(",")
        if len(parts) != 3:
            return None
        try:
            op_a = float(parts[0].strip())
            op_b = float(parts[1].strip())
            operation = parts[2].strip().lower()
            return op_a, op_b, operation
        except ValueError:
            return None

    def _calculate(self, op_a: float, op_b: float, operation: str) -> Optional[float]:
        """Performs the calculation."""
        # (Reusing the calculation logic)
        if operation == "sum":
            return op_a + op_b
        elif operation == "diff":
            return op_a - op_b
        elif operation == "prod":
            return op_a * op_b
        elif operation == "div":
            if abs(op_b) < 1e-6:
                return None  # Division by zero error
            return op_a / op_b
        else:
            return None  # Unknown operation

    def _is_final_answer(self, text: str) -> Optional[float]:
        """Checks if the text is just a final numerical answer."""
        try:
            # Allow potential formatting like <final_answer>16.0</final_answer>
            # or just the number itself.
            processed_text = text.strip()
            if processed_text.startswith("<final_answer>") and processed_text.endswith(
                "</final_answer>"
            ):
                processed_text = processed_text[
                    len("<final_answer>") : -len("</final_answer>")
                ]

            return float(processed_text)
        except ValueError:
            return None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: MultiStepCalcMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[MultiStepCalcMetadata],
    ]:
        """Processes a single turn for the multi-step calculator task."""
        last_assistant_msg = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_msg = message_log[-1]["content"].strip()

        current_step = metadata["current_step"]
        max_steps = metadata["max_steps"]
        expected_final_answer = metadata["expected_final_answer"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = [
            "<call: calculator>"
        ]  # Let model generate tool call or final answer freely
        next_metadata = metadata.copy()
        next_observation_content = ""

        # Check if max steps reached
        if current_step >= max_steps:
            is_terminated = True
            next_observation_content = "<error>Maximum steps reached."
            next_metadata = None
            return (
                {"role": "environment", "content": next_observation_content},
                0.0,
                is_terminated,
                None,
                next_metadata,
            )

        # Check for final answer first
        final_answer = self._is_final_answer(last_assistant_msg)
        if final_answer is not None:
            is_terminated = True
            next_metadata = None  # End of episode
            if abs(final_answer - expected_final_answer) < 1e-6:
                turn_reward = 1.0  # Correct final answer
                next_observation_content = (
                    f"Correct! The final answer is {final_answer:.2f}."
                )
            else:
                turn_reward = 0.0  # Incorrect final answer
                next_observation_content = f"Incorrect final answer. Expected {expected_final_answer:.2f}, got {final_answer:.2f}."
        else:
            # Check for tool call
            parsed_call = self._parse_tool_call(last_assistant_msg)
            if parsed_call:
                req_op_a, req_op_b, req_op = parsed_call
                result = self._calculate(req_op_a, req_op_b, req_op)
                if result is not None:
                    # Tool call success, provide result
                    next_observation_content = f"<result>{result:.5f}</result>"
                    next_metadata["current_step"] += 1
                    is_terminated = False
                else:  # Calculation failed
                    is_terminated = True
                    next_observation_content = "<error>Calculation failed."
                    next_metadata = None
            else:  # No final answer and no valid tool call
                is_terminated = True
                next_observation_content = (
                    "<error>Invalid response. Expected tool call or final answer."
                )
                next_metadata = None

        next_observation = {"role": "environment", "content": next_observation_content}
        return (
            next_observation,
            turn_reward,
            is_terminated,
            next_stop_strings,
            next_metadata,
        )


@ray.remote
class MultiStepCalculatorEnv(EnvironmentInterface):
    """Multi-step calculator environment (Ray Actor)."""

    def __init__(self, cfg: Optional[dict] = None):
        self.logic = _MultiStepCalculatorLogic()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[MultiStepCalcMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of interactions using the calculator logic."""
        futures = [
            self.logic.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata_batch)
        ]
        results = futures

        # Unpack results and format according to EnvironmentReturn tuple
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []  # List of Lists or Nones
        all_next_metadata = []

        for obs, rew, term, stops, meta in results:
            observations.append(obs)  # obs is already dict[str, str]
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)

        # Convert to tensors where needed
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # Done flag combines termination and truncation (truncation not used here)
        done_tensor = torch.tensor(terminateds, dtype=torch.bool)

        # Return tuple matching EnvironmentReturn NamedTuple
        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=rewards_tensor,
            terminateds=done_tensor,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        # Example: could calculate success rate based on final reward
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"success_rate": success_rate}
