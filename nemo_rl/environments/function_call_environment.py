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
from typing import Any, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

logger = logging.getLogger(__name__)


class FunctionCallEnvConfig(TypedDict):
    num_workers: int


class FunctionCallMetadata(TypedDict):
    gold_answers: str  # JSON string of gold function calls


def _parse_tool_calls(response: str) -> list[dict[str, Any]]:
    """Parse tool calls from model output in Qwen3 native format.

    Looks for <tool_call>{"name": ..., "arguments": ...}</tool_call> blocks.
    """
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, response, re.DOTALL)

    tool_calls = []
    for match in matches:
        try:
            call = json.loads(match)
            tool_calls.append(call)
        except json.JSONDecodeError:
            continue
    return tool_calls


def _normalize_value(v: Any) -> Any:
    """Normalize a value for comparison (convert numeric strings, etc.)."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v


def _compare_arguments(pred_args: dict, gold_args: dict) -> bool:
    """Compare predicted arguments against gold arguments."""
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    for key in gold_args:
        if _normalize_value(pred_args[key]) != _normalize_value(gold_args[key]):
            return False
    return True


def _score_tool_call(
    pred_calls: list[dict[str, Any]], gold_answers: list[dict[str, Any]]
) -> tuple[float, str | None]:
    """Score predicted tool calls against gold answers.

    Returns:
        (reward, extracted_answer_str)
        - 1.0: exact match (correct name and arguments)
        - 0.5: correct function name but wrong arguments
        - 0.0: wrong function name or no valid tool call
    """
    if not pred_calls:
        return 0.0, None

    if not gold_answers:
        return 0.0, None

    # Compare first predicted call against first gold answer
    pred = pred_calls[0]
    gold = gold_answers[0]

    pred_name = pred.get("name", "")
    gold_name = gold.get("name", "")

    extracted = json.dumps(pred, ensure_ascii=False)

    if pred_name != gold_name:
        return 0.0, extracted

    pred_args = pred.get("arguments", {})
    gold_args = gold.get("arguments", {})

    if isinstance(pred_args, str):
        try:
            pred_args = json.loads(pred_args)
        except json.JSONDecodeError:
            return 0.5, extracted

    if isinstance(gold_args, str):
        try:
            gold_args = json.loads(gold_args)
        except json.JSONDecodeError:
            return 0.5, extracted

    if _compare_arguments(pred_args, gold_args):
        return 1.0, extracted

    return 0.5, extracted


@ray.remote(
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class FunctionCallEnvironment(EnvironmentInterface[FunctionCallMetadata]):
    def __init__(self, cfg: FunctionCallEnvConfig):
        self.cfg = cfg

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[FunctionCallMetadata],
    ) -> EnvironmentReturn[FunctionCallMetadata]:
        rewards = []
        answers = []

        for message_log, meta in zip(message_log_batch, metadata):
            # Extract assistant response
            response = ""
            for msg in message_log:
                if msg["role"] == "assistant":
                    response += str(msg["content"])

            # Parse gold answers
            try:
                gold_answers = json.loads(meta["gold_answers"])
            except json.JSONDecodeError:
                gold_answers = []

            # Parse predicted tool calls from response
            pred_calls = _parse_tool_calls(response)

            # If no tool_call tags found, try parsing the whole response as JSON
            if not pred_calls:
                try:
                    parsed = json.loads(response.strip())
                    if isinstance(parsed, dict):
                        pred_calls = [parsed]
                    elif isinstance(parsed, list):
                        pred_calls = parsed
                except json.JSONDecodeError:
                    pass

            reward, extracted = _score_tool_call(pred_calls, gold_answers)
            rewards.append(reward)
            answers.append(extracted)

        batch_size = len(message_log_batch)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""}] * batch_size,
            metadata=metadata,
            next_stop_strings=[None] * batch_size,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(batch_size, dtype=torch.bool),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        rewards = batch.get("total_reward", torch.tensor([0.0]))

        # Exact match = reward 1.0, partial = 0.5
        exact_match = (
            (rewards == 1.0).float().mean().item() if len(rewards) > 0 else 0.0
        )
        partial_match = (
            (rewards == 0.5).float().mean().item() if len(rewards) > 0 else 0.0
        )

        metrics = {
            "accuracy": rewards.mean().item() if len(rewards) > 0 else 0.0,
            "exact_match": exact_match,
            "partial_match": partial_match,
            "num_problems_in_batch": len(rewards),
        }

        return batch, metrics
