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

"""Producer-side payload helpers for the async-RL TQ path."""

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data.interfaces import LLMMessageLogType, VLMMessageLogType
from nemo_rl.data_plane.codec import pack_jagged_fields
from nemo_rl.data_plane.column_io import TOKEN_ALIGNED_FIELDS
from nemo_rl.data_plane.schema import (
    INVALID_TOOL_CALL_MASK,
    MALFORMED_THINKING_MASK,
    MASK_SAMPLE,
    ROUTED_EXPERTS_FIELD,
    TRUNCATED,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord

VIOLATION_TAG_KEYS = (
    "num_invalid_tool_calls",
    "num_malformed_thinking",
    "num_assistant_messages",
)
# Per-row violation counts ride ``tags`` rather than the tensor fields, so this
# key is carried on the train batch and consumed by pack_payload.
_VIOLATION_COUNTS_KEY = "violation_counts"


def _violation_counts(
    message_log: LLMMessageLogType | VLMMessageLogType,
) -> dict[str, int]:
    """Count invalid tool calls / malformed thinking over flagged assistant turns."""
    counts = dict.fromkeys(VIOLATION_TAG_KEYS, 0)
    for message in message_log:
        if message["role"] != "assistant" or "generation_logprobs" not in message:
            continue
        counts["num_assistant_messages"] += 1
        if message.get("is_invalid_tool_call", False):
            counts["num_invalid_tool_calls"] += 1
        if message.get("has_malformed_thinking", False):
            counts["num_malformed_thinking"] += 1
    return counts


def _add_message_violation_masks(
    message_logs: list[LLMMessageLogType | VLMMessageLogType],
) -> None:
    """Attach token-aligned masks for generated assistant violations.

    This must run before the generic message normalizer fills missing
    ``generation_logprobs`` on prompt and environment messages, because field
    presence distinguishes generated assistant turns.
    """
    for message_log in message_logs:
        for message in message_log:
            token_ids = cast(torch.Tensor, message["token_ids"])
            is_generated_assistant = (
                message["role"] == "assistant" and "generation_logprobs" in message
            )
            is_invalid = is_generated_assistant and bool(
                message.get("is_invalid_tool_call", False)
            )
            is_malformed = is_generated_assistant and bool(
                message.get("has_malformed_thinking", False)
            )
            message[INVALID_TOOL_CALL_MASK] = torch.full_like(
                token_ids, is_invalid, dtype=torch.bool
            )
            message[MALFORMED_THINKING_MASK] = torch.full_like(
                token_ids, is_malformed, dtype=torch.bool
            )


def record_to_train_batch(
    record: PromptGroupRecord,
    *,
    pad_value_dict: Mapping[str, int],
    include_message_violation_fields: bool,
) -> BatchedDataDict[Any]:
    """Convert one prompt group's record into a packed BatchedDataDict of N rows.

    Args:
        record: Rollout's PromptGroupRecord with N completions to flatten into rows.
        pad_value_dict: Field-name → pad value used by batched_message_log_to_flat_message.
        include_message_violation_fields: Whether to tensorize message violation
            flags for configured advantage penalties.

    Returns:
        BatchedDataDict with input_ids, input_lengths, generation_logprobs,
        token_mask, an all-ones sample_mask, the raw mask_sample and truncated
        flags, prompt_ids_for_adv, total_reward, violation counts, and optional
        routed experts and message-violation masks.
    """
    # Lazy imports: grpo and llm_message_utils transitively pull
    # experience.rollouts, so importing at module top risks a cycle.
    from nemo_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
        extract_initial_prompt_messages,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.experience.rollouts import (
        _mask_sample_flags,
        backfill_missing_routed_experts,
    )

    completions = record.completions
    n = len(completions)
    assert n > 0, "PromptGroupRecord has no completions"

    message_logs = [c.message_log for c in completions]
    violation_counts = [_violation_counts(message_log) for message_log in message_logs]
    prompt_token_count = sum(len(m["token_ids"]) for m in record.prompt)
    if include_message_violation_fields:
        _add_message_violation_masks(message_logs)
    prompt_lengths = torch.full((n,), prompt_token_count, dtype=torch.long)

    # Must precede the prompt extraction: it reuses the same message dicts, so
    # backfilling here also covers the prompt flatten below. Doing it only inside
    # add_grpo_token_loss_masks_and_generation_logprobs would be too late.
    backfill_missing_routed_experts(message_logs)

    prompt_message_logs = extract_initial_prompt_messages(message_logs, prompt_lengths)
    prompt_flat, _ = batched_message_log_to_flat_message(
        prompt_message_logs,
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,  # type: ignore
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    total_reward = torch.tensor(
        [float(c.reward) for c in completions], dtype=torch.float32
    )
    mask_sample = _mask_sample_flags(c.env_extras for c in completions)
    truncated = torch.tensor([c.truncated for c in completions], dtype=torch.bool)
    sample_mask = torch.ones(n, dtype=torch.float32)

    train_data: dict[str, Any] = {
        "input_ids": flat["token_ids"],
        "input_lengths": input_lengths,
        "generation_logprobs": flat["generation_logprobs"],
        "token_mask": flat["token_loss_mask"],
        "sample_mask": sample_mask,
        "prompt_ids_for_adv": prompt_flat["token_ids"],
        MASK_SAMPLE: mask_sample,
        TRUNCATED: truncated,
        "total_reward": total_reward,
        _VIOLATION_COUNTS_KEY: violation_counts,
    }
    if ROUTED_EXPERTS_FIELD in flat:
        train_data[ROUTED_EXPERTS_FIELD] = flat[ROUTED_EXPERTS_FIELD]
    if include_message_violation_fields:
        train_data[INVALID_TOOL_CALL_MASK] = flat[INVALID_TOOL_CALL_MASK]
        train_data[MALFORMED_THINKING_MASK] = flat[MALFORMED_THINKING_MASK]
    return BatchedDataDict[Any](train_data)


def pack_payload(
    train_batch: Mapping[str, Any],
    *,
    weight_version: int,
    group_id: str,
    prompt_idx: int,
) -> tuple[list[str], TensorDict, list[dict[str, Any]]]:
    """Pack a producer batch into (sample_ids, fields, tags) for put_samples.

    Args:
        train_batch: Mapping with at least input_lengths plus the tensor/object fields to send.
        weight_version: Trainer weight version stamped on every row's tag.
        group_id: Per-group identifier used as the sample_id prefix; the caller owns uniqueness.
        prompt_idx: Stable dataset prompt index stamped on every row's tag.

    Returns:
        sample_ids of the form {group_id}_g{i}, a jagged-packed TensorDict, and per-row
        tags carrying weight_version plus any per-row violation counts.
    """
    lengths = train_batch["input_lengths"]
    n = int(lengths.shape[0])
    tensor_fields: dict[str, torch.Tensor | np.ndarray] = {
        k: v
        for k, v in train_batch.items()
        if isinstance(v, torch.Tensor)
        or (isinstance(v, np.ndarray) and v.dtype == object)
    }
    fields_td = pack_jagged_fields(
        tensor_fields, lengths=lengths, token_aligned_fields=TOKEN_ALIGNED_FIELDS
    )
    sample_ids = [f"{group_id}_g{i}" for i in range(n)]
    violations = train_batch.get(_VIOLATION_COUNTS_KEY, [{}] * n)
    tags = [
        {
            "weight_version": weight_version,
            "prompt_idx": prompt_idx,
            **violations[i],
        }
        for i in range(n)
    ]
    return sample_ids, fields_td, tags
