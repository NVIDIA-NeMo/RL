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
"""Leftmost corruption for the checkpoint's native block_diff training path."""

import math
from typing import Any

import torch

from just_grpo.config import ScheduleConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from just_grpo.diffusion.denoising_schedule import DenoisingSchedule


def align_prompt(
    tokens: list[int], *, block_size: int, mask_token_id: int
) -> list[int]:
    """Left-pad with fixed MASK context; use these exact IDs for generation too."""
    return [mask_token_id] * (-len(tokens) % block_size) + tokens


@torch.no_grad()
def select_low_confidence_tokens(
    data: BatchedDataDict[Any], *, fraction: float, block_size: int | None = None
) -> torch.Tensor:
    """Keep low-confidence positions per block, or per completion without a block.

    Rank independently in each sample/block when block_size is supplied.
    Ties prefer earlier positions. Prompts,
    padding, and excluded samples are never selected. This is sampled-token
    confidence, not entropy or the maximum probability over the vocabulary.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Training token fraction must be in (0, 1]")
    logprobs = data["generation_logprobs"].detach()
    valid = data["token_mask"].bool() & data["sample_mask"].bool()[:, None]
    valid &= (
        torch.arange(valid.shape[1], device=valid.device)[None]
        < data["input_lengths"][:, None]
    )
    if logprobs.shape != valid.shape or not torch.isfinite(logprobs[valid]).all():
        raise ValueError(
            "Require finite rollout logprobs at every valid response position"
        )
    if fraction == 1:
        return valid
    if block_size is not None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        width = valid.shape[1]
        padding = -width % block_size
        block_valid = torch.nn.functional.pad(valid, (0, padding)).reshape(
            valid.shape[0], -1, block_size
        )
        scores = torch.nn.functional.pad(logprobs, (0, padding), value=float("inf"))
        scores = scores.reshape_as(block_valid).masked_fill(~block_valid, float("inf"))
        count = math.ceil(fraction * block_size)
        order = scores.argsort(dim=-1, stable=True)[..., :count]
        selected = torch.zeros_like(block_valid).scatter_(-1, order, True)
        return (selected & block_valid).reshape(valid.shape[0], -1)[:, :width]
    counts = (valid.sum(dim=1) * fraction).ceil().long()
    order = logprobs.masked_fill(~valid, float("inf")).argsort(dim=1, stable=True)
    keep = torch.arange(valid.shape[1], device=valid.device)[None] < counts[:, None]
    return torch.zeros_like(valid).scatter_(1, order, keep) & valid


class BlockJustGRPOSchedule(DenoisingSchedule):
    def __init__(
        self,
        data: BatchedDataDict[Any],
        config: ScheduleConfig,
        *,
        pad_token_id: int,
        padded_width: int | None = None,
        selected_positions_per_block: int | None = None,
    ) -> None:
        ids = data["input_ids"]
        valid = data["token_mask"].bool()
        lengths = data["input_lengths"]
        if (
            ids.ndim != 2
            or valid.shape != ids.shape
            or lengths.shape != (ids.shape[0],)
        ):
            raise ValueError("Expected matching [N,S] IDs/masks and [N] lengths")
        if ((lengths < 0) | (lengths > ids.shape[1])).any():
            raise ValueError("Invalid input lengths")
        valid = valid & (
            torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        )
        valid = valid & data["sample_mask"].bool()[:, None]
        block = config.block_size
        width = max(block, (ids.shape[1] + block - 1) // block * block)
        if padded_width is not None:
            if padded_width < width or padded_width % block:
                raise ValueError("padded_width must fit the batch and align to a block")
            width = padded_width
        clean = torch.nn.functional.pad(
            ids, (0, width - ids.shape[1]), value=pad_token_id
        )
        selected = data.get("training_token_mask")
        if selected is None:
            selected = valid
        elif selected.shape != valid.shape or (selected.bool() & ~valid).any():
            raise ValueError(
                "training_token_mask must be a subset of valid response tokens"
            )
        score_mask = torch.nn.functional.pad(selected.bool(), (0, width - ids.shape[1]))
        self.response_mask = torch.nn.functional.pad(valid, (0, width - ids.shape[1]))
        canvas_mask = torch.zeros_like(clean, dtype=torch.bool)
        for row, active in enumerate(valid):
            positions = active.nonzero().flatten()
            if not positions.numel():
                continue
            start, end = int(positions[0]), int(positions[-1]) + 1
            if end - start != positions.numel():
                raise ValueError(
                    "Block JustGRPO currently requires one contiguous response span"
                )
            if start % block:
                raise ValueError(
                    "Prompt must be block-aligned before generation; use align_prompt"
                )
            canvas_mask[row, start : (end + block - 1) // block * block] = True
        self.config = config
        self.original_width = ids.shape[1]
        self.canvas_mask = canvas_mask
        position_ids = torch.arange(width, device=ids.device)[None].expand_as(clean)
        self.base = BatchedDataDict(
            input_ids=clean,
            target_ids=clean,
            token_mask=score_mask,
            position_ids=position_ids,
            original_positions=position_ids.clamp_max(ids.shape[1] - 1),
            sample_indices=torch.arange(ids.shape[0], device=ids.device),
        )
        self.selected_offsets = None
        if selected_positions_per_block is not None:
            if config.reveal_tokens_per_step != 1:
                raise ValueError("Compact partial views require one token per reveal")
            if not 1 <= selected_positions_per_block <= block:
                raise ValueError("Invalid selected positions per block")
            block_mask = score_mask.reshape(ids.shape[0], -1, block)
            if (block_mask.sum(-1) > selected_positions_per_block).any():
                raise ValueError("Selection exceeds the partial-view budget")
            offsets = torch.arange(block, device=ids.device).expand_as(block_mask)
            self.selected_offsets = (
                offsets.masked_fill(~block_mask, block)
                .sort(-1)
                .values[..., :selected_positions_per_block]
            )
        super().__init__(
            num_steps=selected_positions_per_block
            or (block + config.reveal_tokens_per_step - 1)
            // config.reveal_tokens_per_step,
            num_samples=ids.shape[0],
        )

    def generate_single_trajectory(self, time_step: int) -> BatchedDataDict[Any]:
        if not 0 <= time_step < self.num_steps:
            raise IndexError(time_step)
        if self.selected_offsets is not None:
            # Each (sample, block) supplies its own next selected position. The
            # prefix includes every earlier token, including unselected tokens.
            threshold = self.selected_offsets[..., time_step].repeat_interleave(
                self.config.block_size, dim=1
            )
            offsets = self.base["position_ids"] % self.config.block_size
            result = BatchedDataDict(self.base)
            result["masked_indices"] = self.canvas_mask & (
                ~self.response_mask | (offsets >= threshold)
            )
            result["token_mask"] = self.base["token_mask"] & (offsets == threshold)
            return result
        prefix = time_step * self.config.reveal_tokens_per_step
        offsets = self.base["position_ids"] % self.config.block_size
        valid = self.base["token_mask"]
        result = BatchedDataDict(self.base)
        # Tail tokens after EOS stay MASK at every reveal level. Prefix MASK
        # tokens belong to fixed context and are never part of masked_indices.
        result["masked_indices"] = self.canvas_mask & (
            ~self.response_mask | (offsets >= prefix)
        )
        result["token_mask"] = (
            valid
            & (offsets >= prefix)
            & (offsets < prefix + self.config.reveal_tokens_per_step)
        )
        return result

    def scatter_logprobs(
        self, trajectory: BatchedDataDict[Any], logprobs: torch.Tensor
    ) -> torch.Tensor:
        output = logprobs.new_zeros((logprobs.shape[0], self.original_width))
        return output.scatter_add(
            1, trajectory["original_positions"], logprobs * trajectory["token_mask"]
        )
