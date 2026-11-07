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
"""Shared response-relative block canvases for diffusion schedules and generation."""

from typing import Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class BlockDiffusionLayout:
    """Response-relative noisy canvas and the original, unmodified clean context."""

    def __init__(
        self,
        data: BatchedDataDict[Any],
        *,
        block_size: int,
        pad_token_id: int,
        padded_width: int | None = None,
    ) -> None:
        ids, lengths = data["input_ids"], data["input_lengths"].long()
        valid = data["token_mask"].bool()
        if (
            ids.ndim != 2
            or valid.shape != ids.shape
            or lengths.shape != (ids.shape[0],)
        ):
            raise ValueError("Expected matching [N,S] IDs/masks and [N] lengths")
        if block_size < 1 or ((lengths < 0) | (lengths > ids.shape[1])).any():
            raise ValueError("Invalid block size or input lengths")
        valid = valid & (
            torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        )
        selected = (
            data["training_token_mask"].bool()
            if "training_token_mask" in data
            else valid
        )
        if selected.shape != valid.shape or (selected & ~valid).any():
            raise ValueError(
                "training_token_mask must be a subset of valid response tokens"
            )
        starts = lengths.clone()
        response_lengths = torch.zeros_like(lengths)
        for row, active in enumerate(valid):
            positions = active.nonzero().flatten()
            if positions.numel():
                start, end = int(positions[0]), int(positions[-1]) + 1
                if end - start != positions.numel():
                    raise ValueError(
                        "Block diffusion currently requires one contiguous response span"
                    )
                starts[row], response_lengths[row] = start, end - start
        canvas_lengths = (response_lengths + block_size - 1) // block_size * block_size
        width = max(block_size, int(canvas_lengths.max()))
        clean_width = ids.shape[1]
        if padded_width is not None:
            if padded_width < clean_width or padded_width % block_size:
                raise ValueError("padded_width must fit the batch and align to a block")
            # Fixed widths keep TP/DP collective and compilation shapes stable.
            width = clean_width = padded_width
        positions = torch.arange(width, device=ids.device)[None].expand(
            ids.shape[0], -1
        )
        original = starts[:, None] + positions
        gather = original.clamp_max(ids.shape[1] - 1)
        response = positions < response_lengths[:, None]
        clean = torch.nn.functional.pad(
            ids, (0, clean_width - ids.shape[1]), value=pad_token_id
        )
        noisy = ids.gather(1, gather).masked_fill(~response, pad_token_id)
        included = data["sample_mask"].bool()[:, None]
        self.response_mask = response & included
        self.canvas_mask = (positions < canvas_lengths[:, None]) & included
        self.original_width = ids.shape[1]
        self.base = BatchedDataDict(
            input_ids=noisy,
            target_ids=noisy,
            clean_input_ids=clean,
            prompt_lengths=starts,
            response_offsets=starts + (-starts) % block_size,
            response_lengths=response_lengths,
            noisy_valid_lengths=canvas_lengths,
            clean_lengths=lengths,
            token_mask=selected.gather(1, gather) & response & included,
            position_ids=positions,
            original_positions=gather,
            sample_indices=torch.arange(ids.shape[0], device=ids.device),
        )


def prepare_block_inputs(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    *,
    prompt_lengths: torch.Tensor,
    noisy_valid_lengths: torch.Tensor,
    clean_lengths: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align equal [xt | x0] halves without exposing padding to real queries.

    The clean half retains the original tokens and logical positions. The noisy
    response starts on a block boundary; its final block keeps MASK tail tokens.
    """
    padding = (-prompt_lengths) % block_size
    width = (clean.shape[1] + 2 * block_size - 2) // block_size * block_size
    columns = torch.arange(width, device=clean.device)[None]
    clean_positions = columns - padding[:, None]
    noisy_positions = clean_positions - prompt_lengths[:, None]
    clean_valid = (clean_positions >= 0) & (clean_positions < clean_lengths[:, None])
    noisy_valid = (noisy_positions >= 0) & (
        noisy_positions < noisy_valid_lengths[:, None]
    )
    clean_half = clean.gather(1, clean_positions.clamp(0, clean.shape[1] - 1))
    noisy_half = noisy.gather(1, noisy_positions.clamp(0, noisy.shape[1] - 1))
    inputs = torch.cat(
        [
            noisy_half.masked_fill(~noisy_valid, mask_token_id),
            clean_half.masked_fill(~clean_valid, mask_token_id),
        ],
        dim=1,
    )
    positions = clean_positions.clamp_min(0).repeat(1, 2)
    attention_mask = ~torch.cat([noisy_valid, clean_valid], dim=1)[:, None, None]
    return inputs, positions, attention_mask
