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
from nemo_rl.data.multimodal_utils import (
    encode_multimodal_for_wire,
    multimodal_row_tags,
)
from nemo_rl.data.packed_rollouts import (
    PACKED_ATTENTION_SEGMENT_LENGTHS,
    TREE_ATTENTION_EDGE_LENGTHS,
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_LAYOUTS,
    TREE_EDGE_ALIGNED_FIELDS,
    TREE_EDGE_SHIFTED_FIELDS,
    TREE_EDGE_UNSHIFTED_FIELDS,
)
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
    "num_routed_experts_backfilled",
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
        BatchedDataDict with input IDs and lengths, generation log probabilities,
        token and prompt-level sample masks, raw ``mask_sample`` and ``truncated``
        flags, prompt IDs for advantage computation, rewards, and violation
        counts. Optional fields include routed experts, message-violation masks,
        and any packed or per-token multimodal model inputs carried by the
        completions.
    """
    # Lazy imports: grpo and llm_message_utils transitively pull
    # experience.rollouts, so importing at module top risks a cycle.
    from nemo_rl.algorithms.grpo import (
        _apply_exact_nemo_gym_call_trees,
        _flatten_tree_model_inputs,
        _use_exact_nemo_gym_call_sequences,
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

    original_message_logs = [c.message_log for c in completions]
    rollout_batch = BatchedDataDict[Any]({"message_log": original_message_logs})
    exact_call_logs = [c.training_message_logs for c in completions]
    exact_call_trees = [c.exact_call_tree for c in completions]
    if any(tree is not None for tree in exact_call_trees):
        if any(tree is None for tree in exact_call_trees):
            raise ValueError(
                "precompacted exact-call metadata must be present for every "
                "completion in a prompt group"
            )
        if any(call_logs is not None for call_logs in exact_call_logs):
            raise ValueError(
                "a completion cannot carry both raw and precompacted exact-call metadata"
            )
        _apply_exact_nemo_gym_call_trees(
            rollout_batch,
            [tree for tree in exact_call_trees if tree is not None],
        )
    elif any(call_logs is not None for call_logs in exact_call_logs):
        if any(call_logs is None for call_logs in exact_call_logs):
            raise ValueError(
                "exact NeMo-Gym call metadata must be present for every completion "
                "in a prompt group"
            )
        rollout_batch["training_message_logs"] = [
            call_logs for call_logs in exact_call_logs if call_logs is not None
        ]
        _use_exact_nemo_gym_call_sequences(rollout_batch)

    message_logs = rollout_batch["message_log"]
    violation_counts = [_violation_counts(message_log) for message_log in message_logs]
    prompt_token_count = sum(len(m["token_ids"]) for m in record.prompt)
    if include_message_violation_fields:
        _add_message_violation_masks(message_logs)
    prompt_lengths = torch.full((n,), prompt_token_count, dtype=torch.long)

    # Must precede the prompt extraction: it reuses the same message dicts, so
    # backfilling here also covers the prompt flatten below. Doing it only inside
    # add_grpo_token_loss_masks_and_generation_logprobs would be too late.
    routed_experts_backfilled = backfill_missing_routed_experts(message_logs)
    for counts, backfilled in zip(violation_counts, routed_experts_backfilled):
        counts["num_routed_experts_backfilled"] = backfilled

    prompt_message_logs = extract_initial_prompt_messages(
        original_message_logs, prompt_lengths
    )
    prompt_flat, _ = batched_message_log_to_flat_message(
        prompt_message_logs,
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,  # type: ignore
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    model_flat = flat
    model_input_lengths = input_lengths
    if TREE_ATTENTION_LAYOUTS in rollout_batch:
        model_flat, model_input_lengths = _flatten_tree_model_inputs(
            rollout_batch,
            flat,
            input_lengths,
            pad_token_id=int(pad_value_dict.get("token_ids", 0)),
            make_sequence_length_divisible_by=1,
        )

    total_reward = torch.tensor(
        [float(c.reward) for c in completions], dtype=torch.float32
    )
    mask_sample = _mask_sample_flags(c.env_extras for c in completions)
    truncated = torch.tensor([c.truncated for c in completions], dtype=torch.bool)
    sample_mask = torch.full((n,), float(record.loss_multiplier), dtype=torch.float32)

    train_data: dict[str, Any] = {
        "input_ids": model_flat["token_ids"],
        "input_lengths": model_input_lengths,
        "generation_logprobs": flat["generation_logprobs"],
        "token_mask": flat["token_loss_mask"],
        "sample_mask": sample_mask,
        "prompt_ids_for_adv": prompt_flat["token_ids"],
        MASK_SAMPLE: mask_sample,
        TRUNCATED: truncated,
        "total_reward": total_reward,
        _VIOLATION_COUNTS_KEY: violation_counts,
    }
    if include_message_violation_fields:
        train_data[INVALID_TOOL_CALL_MASK] = flat[INVALID_TOOL_CALL_MASK]
        train_data[MALFORMED_THINKING_MASK] = flat[MALFORMED_THINKING_MASK]
    train_data.update(model_flat.get_multimodal_dict(as_tensors=False))
    if ROUTED_EXPERTS_FIELD in model_flat:
        train_data[ROUTED_EXPERTS_FIELD] = model_flat[ROUTED_EXPERTS_FIELD]
    if TREE_ATTENTION_LAYOUTS in rollout_batch:
        layouts = rollout_batch[TREE_ATTENTION_LAYOUTS]
        edge_width = flat["token_ids"].shape[1] - 1
        edge_sources = torch.full((n, edge_width), -1, dtype=torch.long)
        for row, layout in enumerate(layouts):
            edge_sources[row, : len(layout.edge_source_indices)] = torch.tensor(
                layout.edge_source_indices, dtype=torch.long
            )
        train_data[TREE_ATTENTION_EDGE_SOURCE_INDICES] = edge_sources
        train_data[TREE_ATTENTION_EDGE_TARGET_IDS] = flat["token_ids"][:, 1:]
        train_data[TREE_ATTENTION_EDGE_LENGTHS] = torch.tensor(
            [len(layout.edge_source_indices) for layout in layouts],
            dtype=torch.long,
        )
        train_data[TREE_ATTENTION_LAYOUTS] = layouts
    if PACKED_ATTENTION_SEGMENT_LENGTHS in rollout_batch:
        train_data[PACKED_ATTENTION_SEGMENT_LENGTHS] = rollout_batch[
            PACKED_ATTENTION_SEGMENT_LENGTHS
        ]
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
        Sample IDs of the form ``{group_id}_g{i}``, a jagged-packed TensorDict
        containing tensor fields and encoded multimodal wire fields, and
        per-row tags. Tags carry the weight version, prompt index, violation
        counts, and ``<field>__row_shapes`` metadata required to reconstruct
        packed multimodal rows.
    """
    lengths = train_batch["input_lengths"]
    n = int(lengths.shape[0])
    tensor_fields: dict[str, torch.Tensor | np.ndarray] = {
        k: v
        for k, v in train_batch.items()
        if isinstance(v, torch.Tensor)
        or (isinstance(v, np.ndarray) and v.dtype == object)
    }
    multimodal = BatchedDataDict[Any](train_batch).get_multimodal_dict(as_tensors=False)
    for key, value in multimodal.items():
        wire_value = encode_multimodal_for_wire(key, value)
        if wire_value is not None:
            tensor_fields[key] = wire_value
    token_aligned_fields = TOKEN_ALIGNED_FIELDS
    lengths_by_field = None
    if TREE_ATTENTION_LAYOUTS in train_batch:
        edge_lengths = train_batch[TREE_ATTENTION_EDGE_LENGTHS]
        token_aligned_fields = TOKEN_ALIGNED_FIELDS - TREE_EDGE_ALIGNED_FIELDS
        lengths_by_field = {
            **{field: edge_lengths + 1 for field in TREE_EDGE_SHIFTED_FIELDS},
            **{field: edge_lengths for field in TREE_EDGE_UNSHIFTED_FIELDS},
        }
    fields_td = pack_jagged_fields(
        tensor_fields,
        lengths=lengths,
        token_aligned_fields=token_aligned_fields,
        lengths_by_field=lengths_by_field,
    )
    sample_ids = [f"{group_id}_g{i}" for i in range(n)]
    violations = train_batch.get(_VIOLATION_COUNTS_KEY, [{}] * n)
    multimodal_tags = multimodal_row_tags(multimodal, n) or [{} for _ in range(n)]
    tags = [
        {
            "weight_version": weight_version,
            "prompt_idx": prompt_idx,
            "group_id": group_id,
            "rollout_index": i,
            **violations[i],
            **multimodal_tags[i],
        }
        for i in range(n)
    ]
    return sample_ids, fields_td, tags
