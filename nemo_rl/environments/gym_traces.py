# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Gym trace custody and logical-rollout GRPO batch expansion.

No tokenizer is used to construct training IDs. Each physical row has its own
causal context; only sampled spans owned by that row receive loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator


@dataclass(frozen=True)
class GymTrainingTrace:
    rollout_id: str
    trace_id: str
    token_ids: tuple[int, ...]
    generation_logprobs: tuple[float, ...]
    loss_mask: tuple[int, ...]


def parse_gym_training_traces(envelope: dict[str, Any]) -> list[GymTrainingTrace]:
    """Validate the public Gym v1 wire contract, including sampled-span ownership.

    Invalid custody is an error, never a reason to retokenize or silently fall
    back to a legacy terminal response.
    """
    if (
        type(envelope.get("schema_version")) is not int
        or envelope["schema_version"] != 1
    ):
        raise ValueError("Gym training_traces requires schema_version=1")
    rollout_id = envelope.get("rollout_id")
    if not isinstance(rollout_id, str) or not rollout_id:
        raise ValueError("Gym training_traces requires a nonempty rollout_id")
    if envelope.get("builder") not in {"per_request", "prefix_merging"}:
        raise ValueError("Unsupported Gym training_traces builder")
    rows = envelope.get("traces")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Gym training_traces.traces must be a list")
    trace_ids: set[str] = set()
    owned_call_ids: set[str] = set()
    traces = []
    for row in rows:
        trace_id = row.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id or trace_id in trace_ids:
            raise ValueError("Gym trace_id must be nonempty and unique per rollout")
        trace_ids.add(trace_id)
        tokens = row.get("token_ids")
        logprobs = row.get("generation_logprobs")
        mask = row.get("loss_mask")
        if not all(isinstance(value, list) for value in (tokens, logprobs, mask)):
            raise ValueError("Gym trace vectors must be lists")
        if not tokens or len(tokens) != len(logprobs) or len(tokens) != len(mask):
            raise ValueError("Gym trace token/logprob/mask vectors must align")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Gym trace token IDs must be nonnegative integers")
        if any(type(bit) is not int or bit not in (0, 1) for bit in mask) or mask[0]:
            raise ValueError("Gym loss_mask must be binary with first token masked")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in logprobs
        ):
            raise ValueError("Gym generation_logprobs must be finite")
        call_ids = row.get("model_call_ids")
        spans = row.get("sampled_spans")
        if not isinstance(call_ids, list) or not isinstance(spans, list):
            raise ValueError("Gym trace requires model_call_ids and sampled_spans")
        if any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
            raise ValueError("Gym model_call_ids must be nonempty strings")
        covered = [0] * len(tokens)
        for span in spans:
            call_id, start, end = (
                span.get("model_call_id"),
                span.get("start"),
                span.get("end"),
            )
            if call_id not in call_ids or call_id in owned_call_ids:
                raise ValueError(
                    "Gym sampled model call must have exactly one trace owner"
                )
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 < start < end <= len(tokens)
            ):
                raise ValueError("Gym sampled span is outside the trace")
            owned_call_ids.add(call_id)
            for position in range(start, end):
                if covered[position]:
                    raise ValueError("Gym sampled spans overlap")
                covered[position] = 1
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("Gym model_call_ids must be unique within a trace")
        if covered != mask or not any(mask):
            raise ValueError("Gym sampled_spans must exactly cover loss_mask")
        traces.append(
            GymTrainingTrace(
                rollout_id, trace_id, tuple(tokens), tuple(logprobs), tuple(mask)
            )
        )
    return traces


def gym_trace_message_log(trace: GymTrainingTrace) -> LLMMessageLogType:
    """Represent one independently conditioned trace with exact sampled masks."""
    messages = []
    start = 0
    while start < len(trace.token_ids):
        trainable = trace.loss_mask[start]
        end = start + 1
        while end < len(trace.token_ids) and trace.loss_mask[end] == trainable:
            end += 1
        message = {
            "role": "assistant" if trainable else "user",
            "content": "",
            "token_ids": torch.tensor(trace.token_ids[start:end], dtype=torch.long),
        }
        if trainable:
            message["generation_logprobs"] = torch.tensor(
                trace.generation_logprobs[start:end], dtype=torch.float32
            )
        messages.append(message)
        start = end
    return messages


def gym_masked_message_log(pad_token_id: int) -> LLMMessageLogType:
    """Return an explicit inert row for invalid rollouts and batch padding."""
    return [
        {
            "role": "user",
            "content": "",
            "token_ids": torch.tensor([pad_token_id, pad_token_id], dtype=torch.long),
        }
    ]


@dataclass
class GymTraceBatch:
    batch: BatchedDataDict
    advantages: torch.Tensor
    logical_indices: torch.Tensor
    logical_count: int
    physical_count: int
    padding_count: int
    overlong_trace_count: int
    invalid_rollout_count: int


def prepare_gym_trace_batch(
    logical_batch: BatchedDataDict,
    *,
    estimator: GRPOAdvantageEstimator,
    pad_token_id: int,
    max_sequence_length: int,
    row_multiple: int,
) -> GymTraceBatch:
    """Compute logical GRPO advantages, then expand and zero-pad physical rows.

    A rollout votes in its prompt group's baseline only if its episode mask is
    positive and it has a trainable trace within the policy context limit.
    An overlong trace is replaced by an inert row, preserving valid siblings.
    Padding and invalid rows never contribute reward votes, loss, or tokens.
    """
    if row_multiple < 1 or max_sequence_length < 2:
        raise ValueError("Gym trace row multiple and context limit must be positive")
    logical_count = logical_batch.size
    traces_by_rollout = logical_batch["gym_training_traces"]
    groups = logical_batch["gym_task_group_id"]
    rewards = logical_batch["total_reward"]
    if not torch.isfinite(rewards).all():
        raise ValueError("Gym logical rewards must be finite")
    valid_rollouts = torch.zeros(logical_count, dtype=torch.float32)
    indices, message_logs, trace_ids, rollout_ids, row_validity = [], [], [], [], []
    overlong_count = 0
    known_rollout_ids: set[str] = set()
    for logical_index, traces in enumerate(traces_by_rollout):
        rollout_id = logical_batch["gym_rollout_id"][logical_index]
        if rollout_id in known_rollout_ids:
            raise ValueError("Duplicate Gym rollout_id in a logical batch")
        known_rollout_ids.add(rollout_id)
        if any(trace.rollout_id != rollout_id for trace in traces):
            raise ValueError("Gym trace does not belong to its logical rollout")
        for trace in traces:
            overlong = len(trace.token_ids) > max_sequence_length
            overlong_count += int(overlong)
            valid = (
                bool(sum(trace.loss_mask))
                and not overlong
                and bool(logical_batch["loss_multiplier"][logical_index] > 0)
            )
            if valid:
                valid_rollouts[logical_index] = 1.0
            indices.append(logical_index)
            message_logs.append(
                gym_trace_message_log(trace)
                if valid
                else gym_masked_message_log(pad_token_id)
            )
            row_validity.append(float(valid))
            trace_ids.append(trace.trace_id)
            rollout_ids.append(trace.rollout_id)
        if not traces:
            indices.append(logical_index)
            message_logs.append(gym_masked_message_log(pad_token_id))
            row_validity.append(0.0)
            trace_ids.append("")
            rollout_ids.append(rollout_id)
    if not valid_rollouts.any():
        raise ValueError("Gym logical batch has no eligible training tokens")
    logical_advantages = estimator.compute_advantage(
        prompt_ids=groups.unsqueeze(-1),
        rewards=rewards,
        mask=valid_rollouts.unsqueeze(-1),
        valid_mask=valid_rollouts,
    ).squeeze(-1)
    physical_count = len(indices)
    padding_count = (-physical_count) % row_multiple
    for _ in range(padding_count):
        indices.append(0)
        message_logs.append(gym_masked_message_log(pad_token_id))
        row_validity.append(0.0)
        trace_ids.append("")
        rollout_ids.append("")
    index_tensor = torch.tensor(indices, dtype=torch.long)
    batch = logical_batch.select_indices(index_tensor)
    batch["message_log"] = message_logs
    batch["length"] = torch.tensor(
        [len(messages[0]["token_ids"]) for messages in message_logs]
    )
    batch["loss_multiplier"] = batch["loss_multiplier"] * torch.tensor(row_validity)
    batch["gym_trace_id"] = trace_ids
    batch["gym_rollout_id"] = rollout_ids
    batch["gym_logical_index"] = index_tensor
    return GymTraceBatch(
        batch,
        logical_advantages[index_tensor],
        index_tensor,
        logical_count,
        physical_count,
        padding_count,
        overlong_count,
        int((valid_rollouts == 0).sum()),
    )
