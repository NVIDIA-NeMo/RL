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

import json
import logging
import re
from typing import Any, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

logger = logging.getLogger(__name__)

# Regex to extract tool calls in Qwen3 native format: <tool_call>...</tool_call>
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class FunctionCallEnvConfig(TypedDict):
    num_workers: NotRequired[int]


def _parse_tool_calls(response: str) -> list[dict]:
    """Extract tool calls from model output using <tool_call> tags."""
    matches = _TOOL_CALL_PATTERN.findall(response)
    calls = []
    for match in matches:
        try:
            parsed = json.loads(match)
            calls.append(parsed)
        except json.JSONDecodeError:
            continue
    return calls


def _normalize_value(v: Any) -> Any:
    """Normalize a value for comparison (e.g., numeric strings to numbers)."""
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v.strip().lower()
    return v


def _compare_arguments(pred_args: dict, gold_args: dict) -> bool:
    """Compare predicted and gold arguments with type-flexible matching."""
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    for key in gold_args:
        if _normalize_value(pred_args[key]) != _normalize_value(gold_args[key]):
            return False
    return True


def _score_single(pred_calls: list[dict], gold_calls: list[dict]) -> tuple[float, str]:
    """Score a single sample. Returns (reward, extracted_answer_str)."""
    if not pred_calls:
        return 0.0, ""

    extracted_str = json.dumps(pred_calls)

    if len(pred_calls) != len(gold_calls):
        # Check if at least one function name matches for partial credit
        pred_names = {c.get("name") for c in pred_calls}
        gold_names = {c.get("name") for c in gold_calls}
        if pred_names & gold_names:
            return 0.5, extracted_str
        return 0.0, extracted_str

    # Match predicted calls to gold calls greedily
    matched = 0
    name_matched = 0
    used = set()
    for pred in pred_calls:
        for i, gold in enumerate(gold_calls):
            if i in used:
                continue
            if pred.get("name") == gold.get("name"):
                name_matched += 1
                pred_args = pred.get("arguments", {})
                gold_args = gold.get("arguments", {})
                if _compare_arguments(pred_args, gold_args):
                    matched += 1
                    used.add(i)
                    break
                else:
                    used.add(i)
                    break

    if matched == len(gold_calls):
        return 1.0, extracted_str
    if name_matched > 0:
        return 0.5, extracted_str
    return 0.0, extracted_str


@ray.remote(
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class FunctionCallEnvironment(EnvironmentInterface[dict]):
    def __init__(self, cfg: FunctionCallEnvConfig):
        self.cfg = cfg

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn[dict]:
        rewards = []
        answers: list[str | None] = []

        for message_log, meta in zip(message_log_batch, metadata):
            # Extract the assistant's response
            response = ""
            for msg in message_log:
                if msg["role"] == "assistant":
                    response += str(msg["content"])

            # Parse gold answer from metadata
            gold_calls = json.loads(meta["gold_answer"])

            # Parse predicted tool calls
            pred_calls = _parse_tool_calls(response)

            reward, extracted = _score_single(pred_calls, gold_calls)
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
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        rewards = batch.get("total_reward", torch.tensor([0.0]))
        rewards = rewards * batch["is_end"]

        exact_match = (rewards == 1.0).float().mean().item()
        partial_match = (rewards == 0.5).float().mean().item()
        accuracy = rewards.mean().item()

        metrics = {
            "accuracy": accuracy,
            "exact_match": exact_match,
            "partial_match": partial_match,
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
