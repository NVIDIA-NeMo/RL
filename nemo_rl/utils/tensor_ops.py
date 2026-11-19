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
from typing import Optional

import torch


def pad_and_concat(
    tensors: list[torch.Tensor], *, target_len: int, pad_value: float = 0.0
) -> torch.Tensor:
    """Right-pad each tensor along dim 1 to `target_len`, then concatenate along dim 0.

    Args:
        tensors: Tensors with the sequence on dim 1.
        target_len: Sequence length every tensor is padded up to.
        pad_value: Fill value for the padded positions.

    Returns:
        The padded tensors concatenated along dim 0.
    """
    padded: list[torch.Tensor] = []
    for t in tensors:
        padding_needed = target_len - t.shape[1]
        if padding_needed > 0:
            # F.pad's spec runs from the last dim backwards; leave trailing dims alone.
            pad_spec = [0, 0] * (t.dim() - 2) + [0, padding_needed]
            t = torch.nn.functional.pad(t, pad_spec, mode="constant", value=pad_value)
        padded.append(t)
    return torch.cat(padded, dim=0)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    global_normalization_factor: Optional[torch.Tensor | float] = None,
):
    """Computes the mean of a microbatch, using a global statistic as the normalization factor."""
    normalization_factor = (
        torch.sum(mask, dim=dim)
        if global_normalization_factor is None
        else global_normalization_factor
    )
    return torch.sum(values * mask, dim=dim) / (normalization_factor + 1e-8)


def masked_var(
    values: torch.Tensor,
    mask: torch.Tensor,
    mean: Optional[torch.Tensor | float] = None,
    unbiased: bool = True,
) -> torch.Tensor:
    if mean is None:
        mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values.pow(2), mask)

    if unbiased:
        normalization_factor = torch.sum(mask)
        # A singleton has zero centered variance; avoid an undefined correction.
        variance = variance * (
            normalization_factor / (normalization_factor - 1).clamp(min=1)
        )
    return variance


def masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Min of the entries of values where mask is set.

    Args:
        values: Tensor to reduce.
        mask: Mask broadcastable to values; nonzero entries are kept.

    Returns:
        A 0-dim tensor in the dtype of values; +inf when mask selects nothing.
    """
    # min() raises on an empty tensor.
    if values.numel() == 0:
        return values.new_full((), float("inf"))
    return torch.where(mask.bool(), values, float("inf")).min()


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max of the entries of values where mask is set.

    Args:
        values: Tensor to reduce.
        mask: Mask broadcastable to values; nonzero entries are kept.

    Returns:
        A 0-dim tensor in the dtype of values; -inf when mask selects nothing.
    """
    # max() raises on an empty tensor.
    if values.numel() == 0:
        return values.new_full((), float("-inf"))
    return torch.where(mask.bool(), values, float("-inf")).max()
