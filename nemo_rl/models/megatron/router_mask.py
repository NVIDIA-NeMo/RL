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

"""Router participation derived from input provenance, never prediction loss."""

import torch


def get_router_padding_mask(
    input_lengths: torch.Tensor,
    sequence_length: int,
    *,
    artificial_inputs: torch.Tensor | None = None,
    cu_seqlens_padded: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exclude artificial rows and physical gaps in a dense or packed layout.

    Args:
        input_lengths: Genuine physical lengths, including borrowed layouts.
        sequence_length: Width of the actual model input after alignment.
        artificial_inputs: One boolean per source row; True excludes the row.
        cu_seqlens_padded: Packed row boundaries, including alignment gaps.
            None selects a dense [batch, sequence] layout.

    Returns:
        A boolean mask in the full input layout. True excludes router statistics.
        The caller owns subsequent context/sequence-parallel slicing.
    """
    if input_lengths.ndim != 1:
        raise ValueError("input_lengths must contain one length per input")
    if artificial_inputs is not None and (
        artificial_inputs.shape != input_lengths.shape
        or artificial_inputs.dtype != torch.bool
    ):
        raise ValueError("artificial_inputs must contain one boolean per input")
    positions = torch.arange(sequence_length, device=input_lengths.device)
    if cu_seqlens_padded is None:
        mask = positions.unsqueeze(0) >= input_lengths.unsqueeze(1)
        if artificial_inputs is not None:
            mask = mask | artificial_inputs.unsqueeze(1)
        return mask

    if cu_seqlens_padded.numel() != input_lengths.numel() + 1:
        raise ValueError("packed boundaries must contain one interval per input")
    rows = torch.searchsorted(cu_seqlens_padded[1:].contiguous(), positions, right=True)
    offsets = positions - cu_seqlens_padded[:-1].index_select(0, rows)
    mask = offsets >= input_lengths.index_select(0, rows)
    if artificial_inputs is not None:
        mask = mask | artificial_inputs.index_select(0, rows)
    return mask.unsqueeze(0)
