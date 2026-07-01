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

"""Helpers for logging tensor transport volume."""

from __future__ import annotations

from math import prod
from typing import Any, MutableMapping, Sequence

import torch


def tensor_nbytes(value: Any) -> int:
    """Return tensor payload bytes, or zero for non-tensors."""
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def valid_token_tensor_nbytes(
    tensor: torch.Tensor,
    sequence_lengths: Sequence[int] | None,
) -> int:
    """Return bytes for the valid token prefix of a batched sequence tensor.

    Args:
        tensor: Tensor with shape ``[B, S, ...]``.
        sequence_lengths: Per-row valid sequence lengths. When absent or
            incompatible with ``tensor``, the full tensor byte count is used.

    Returns:
        Byte count for ``sum(sequence_lengths)`` token positions times the
        trailing per-token element count.
    """
    if sequence_lengths is None or tensor.dim() < 2:
        return tensor_nbytes(tensor)
    if tensor.shape[0] != len(sequence_lengths):
        return tensor_nbytes(tensor)

    max_seq_len = int(tensor.shape[1])
    valid_tokens = sum(
        min(max(int(length), 0), max_seq_len) for length in sequence_lengths
    )
    trailing_elems = (
        prod(int(size) for size in tensor.shape[2:]) if tensor.dim() > 2 else 1
    )
    return int(valid_tokens * trailing_elems * tensor.element_size())


def topk_payload_nbytes(
    topk_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    sequence_lengths: Sequence[int] | None = None,
) -> int:
    """Return combined top-k logits and indices payload bytes."""
    if sequence_lengths is None:
        return tensor_nbytes(topk_logits) + tensor_nbytes(topk_indices)
    return valid_token_tensor_nbytes(
        topk_logits, sequence_lengths
    ) + valid_token_tensor_nbytes(topk_indices, sequence_lengths)


def add_byte_metric_derivatives(
    metrics: MutableMapping[str, float | int],
    *,
    token_count: float | int,
) -> None:
    """Add GiB and per-token derivatives for every ``*_bytes`` metric."""
    denom = float(token_count)
    for name, value in list(metrics.items()):
        if not name.endswith("_bytes"):
            continue
        byte_value = float(value)
        stem = name[: -len("_bytes")]
        metrics[f"{stem}_gib"] = byte_value / float(1024**3)
        if denom > 0:
            metrics[f"{name}_per_token"] = byte_value / denom
