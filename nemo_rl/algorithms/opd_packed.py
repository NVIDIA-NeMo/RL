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

"""Per-sample object-ref transport for teacher top-k OPD tensors.

The replay/driver path must not materialize a globally padded ``[B, S, K]``
tensor.  Teacher workers therefore publish one unpadded ``[S_i - 1, K]``
object per sample, and Megatron training workers resolve only the entries in
their current microbatch.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import ray
import torch


OPD_TEACHER_TOPK_PACKED_KEY = "opd_teacher_topk_per_sample_packed"


def resolve_packed_field(
    entries: Sequence[dict[str, Any]], field_name: str
) -> list[torch.Tensor]:
    """Resolve one packed field, accepting direct tensors in unit tests."""
    if not entries:
        return []

    direct = [field_name in entry for entry in entries]
    ref_name = f"{field_name}_ref"
    refs = [ref_name in entry for entry in entries]
    if all(direct):
        tensors = [entry[field_name] for entry in entries]
    elif all(refs):
        tensors = ray.get([entry[ref_name] for entry in entries])
    else:
        raise KeyError(
            f"Every packed entry must contain either {field_name!r} or "
            f"{ref_name!r}; mixed or missing fields are unsupported."
        )

    if not all(torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError(f"Packed field {field_name!r} must resolve to tensors.")
    return tensors


def pack_teacher_topk_for_replay(
    topk_indices: torch.Tensor,
    topk_logprobs: torch.Tensor,
    input_lengths: torch.Tensor,
) -> list[dict[str, Any]]:
    """Publish dense teacher output as unpadded per-sample object refs.

    This is the compatibility fallback for a non-packed teacher. Packed
    Megatron teachers publish the same entries directly from their workers.
    """
    if topk_indices.ndim != 3 or topk_logprobs.ndim != 3:
        raise ValueError(
            "Teacher top-k tensors must both have shape [batch, sequence, k], "
            f"got {tuple(topk_indices.shape)} and {tuple(topk_logprobs.shape)}."
        )
    if topk_indices.shape != topk_logprobs.shape:
        raise ValueError(
            "Teacher top-k indices and logprobs must have identical shapes, got "
            f"{tuple(topk_indices.shape)} and {tuple(topk_logprobs.shape)}."
        )

    batch_size, sequence_length, topk = topk_indices.shape
    if tuple(input_lengths.shape) != (batch_size,):
        raise ValueError(
            "input_lengths must contain one value per teacher top-k row, got "
            f"{tuple(input_lengths.shape)} for batch size {batch_size}."
        )

    entries: list[dict[str, Any]] = []
    lengths_cpu = input_lengths.detach().cpu()
    for sample_idx in range(batch_size):
        input_length = int(lengths_cpu[sample_idx].item())
        if input_length < 0 or input_length > sequence_length:
            raise ValueError(
                f"input_lengths[{sample_idx}]={input_length} is outside "
                f"[0, {sequence_length}]."
            )
        # Logits at positions [0, input_length - 1) predict the real next
        # tokens. The final input position predicts padding/wrap-around and is
        # masked by the training loss, so do not store it.
        next_token_length = max(input_length - 1, 0)
        indices = (
            topk_indices[sample_idx, :next_token_length]
            .detach()
            .to(device="cpu", dtype=torch.int32)
            .contiguous()
        )
        logprobs = (
            topk_logprobs[sample_idx, :next_token_length].detach().cpu().contiguous()
        )
        entries.append(
            {
                "seq_len": next_token_length,
                "topk": int(topk),
                "topk_indices_ref": ray.put(indices),
                "topk_logprobs_ref": ray.put(logprobs),
            }
        )
    return entries


def materialize_teacher_topk_microbatch(
    data: MutableMapping[str, Any],
) -> None:
    """Resolve packed teacher support for one student training microbatch."""
    if OPD_TEACHER_TOPK_PACKED_KEY not in data:
        return
    if "opd_support_indices" in data or "teacher_support_logprobs" in data:
        raise ValueError(
            "Packed and dense teacher top-k fields cannot be present together."
        )
    if "input_ids" not in data or not torch.is_tensor(data["input_ids"]):
        raise ValueError("Packed teacher top-k materialization requires input_ids.")

    entries = data[OPD_TEACHER_TOPK_PACKED_KEY]
    if not isinstance(entries, list):
        raise TypeError(
            f"{OPD_TEACHER_TOPK_PACKED_KEY} must be a list, got "
            f"{type(entries).__name__}."
        )
    batch_size, sequence_length = data["input_ids"].shape[:2]
    if len(entries) != batch_size:
        raise ValueError(
            "Packed teacher top-k must have one entry per microbatch row, got "
            f"{len(entries)} entries for batch size {batch_size}."
        )
    if not entries:
        raise ValueError("Cannot materialize an empty packed teacher top-k batch.")

    index_tensors = resolve_packed_field(entries, "topk_indices")
    logprob_tensors = resolve_packed_field(entries, "topk_logprobs")
    first_indices = index_tensors[0]
    first_logprobs = logprob_tensors[0]
    if first_indices.ndim != 2 or first_logprobs.ndim != 2:
        raise ValueError("Packed teacher top-k fields must have shape [seq_len, k].")
    topk = int(first_indices.shape[-1])
    if topk < 1:
        raise ValueError(f"Packed teacher top-k width must be positive, got {topk}.")

    # Keep the object-store payload compact (the producer stores int32), but
    # materialize the training input directly in the dtype consumed by the
    # loss. Otherwise ``prepare_*_loss_input`` would temporarily retain this
    # full int32 tensor while allocating a second int64 copy on the GPU.
    dense_indices = torch.zeros((batch_size, sequence_length, topk), dtype=torch.int64)
    dense_logprobs = torch.zeros(
        (batch_size, sequence_length, topk), dtype=first_logprobs.dtype
    )
    max_next_token_length = max(sequence_length - 1, 0)
    for sample_idx, (entry, indices, logprobs) in enumerate(
        zip(entries, index_tensors, logprob_tensors)
    ):
        seq_len = int(entry.get("seq_len", -1))
        expected_shape = (seq_len, topk)
        if seq_len < 0 or seq_len > max_next_token_length:
            raise ValueError(
                f"Packed teacher top-k sample {sample_idx} has invalid "
                f"seq_len={seq_len} for sequence length {sequence_length}."
            )
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Packed topk_indices sample {sample_idx} has shape "
                f"{tuple(indices.shape)}, expected {expected_shape}."
            )
        if tuple(logprobs.shape) != expected_shape:
            raise ValueError(
                f"Packed topk_logprobs sample {sample_idx} has shape "
                f"{tuple(logprobs.shape)}, expected {expected_shape}."
            )
        if int(entry.get("topk", topk)) != topk:
            raise ValueError(
                f"Packed teacher top-k sample {sample_idx} disagrees on topk width."
            )
        if seq_len:
            dense_indices[sample_idx, :seq_len].copy_(indices.to(dtype=torch.int64))
            dense_logprobs[sample_idx, :seq_len].copy_(logprobs)

    del data[OPD_TEACHER_TOPK_PACKED_KEY]
    data["opd_support_indices"] = dense_indices
    data["teacher_support_logprobs"] = dense_logprobs
