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

"""JustGRPO *block-reveal* logprob estimation.

Computes the same leftmost-reveal token logprobs as
``just_grpo_logprobs.build_leftmost_reveal_batch`` but in ``block_size`` forward
passes instead of one-pass-per-response-token, by reusing DiffuGRPO's asymmetric
``[noisy | clean]`` block-diffusion layout (see
``diffu_grpo_logprobs.build_fully_masked_completion_batch`` and
``NemotronLabsDiffusionAttention.set_asymmetric_ar_metadata``).

Design: a single fully-masked completion ``base`` (N rows) is built once. Each
*reveal level* ``j`` is then a cheap derived **view** of that base in which every
block reveals its first ``j`` tokens (the rest stay MASK); the view contributes
the logprob of each block's ``j``-th token (its "harvest" positions). The worker
iterates levels ``j = 0 .. num_levels-1`` (a Python for-loop), so only one level
(N rows) is materialized at a time -- the number of forward passes is
``block_size`` (capped by ``max_reveal_levels``), independent of sequence length.

The worker uses ``build_block_reveal_base`` (one fully-masked base) +
``make_reveal_level_view`` (per-level reveal) in an explicit loop for logprobs,
and ``BlockJustGRPORevealSchedule`` (which emits those level views as Megatron
microbatches) for training. ``scatter_block_reveal_logprobs`` maps the harvested
per-level logprobs back to NemoRL's ``[N, S]`` layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

from nemo_rl.algorithms.diffu_grpo_logprobs import (
    build_fully_masked_completion_batch,
    build_fully_masked_completion_loss_batch,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.models.policy import BlockJustGRPOLogprobEstimationConfig


def get_block_reveal_logprob_estimation_cfg(
    cfg: dict[str, Any],
) -> "BlockJustGRPOLogprobEstimationConfig":
    estimator_cfg = cfg.get("logprob_estimation", None)
    if estimator_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    if estimator_cfg["type"] != "just_grpo_block_reveal":
        raise ValueError(
            "policy.logprob_estimation.type must be 'just_grpo_block_reveal'"
        )
    return estimator_cfg


def count_reveal_levels(
    block_size: int,
    response_lengths: torch.Tensor,
    max_reveal_levels: int | None,
) -> int:
    """Number of reveal-level passes.

    MUST be identical across all data-parallel ranks. The worker issues one
    forward pass (and its collectives) per reveal level, so if ranks computed
    different counts they would desync and deadlock at the DP process group
    (observed as a 60-min NCCL collective timeout at 16-node scale). It is
    therefore purely ``block_size`` (capped by ``max_reveal_levels``) and does
    NOT depend on ``response_lengths`` — a per-rank, batch-dependent quantity.
    Levels past a sample's response length simply harvest nothing (zero mask),
    which is correct and keeps all ranks in lockstep.

    ``response_lengths`` is accepted for signature stability but only used to
    short-circuit the empty case.
    """
    if response_lengths.numel() == 0:
        return 0
    num_levels = int(block_size)
    if max_reveal_levels is not None:
        num_levels = min(num_levels, int(max_reveal_levels))
    return int(num_levels)


def build_block_reveal_base(
    data: BatchedDataDict[Any],
    mask_token_id: int,
    pad_token_id: int,
    block_size: int,
    pad_to_length: int | None = None,
    include_loss: bool = False,
    max_reveal_levels: int | None = None,
) -> tuple[BatchedDataDict[Any], int, int]:
    """Build the fully-masked completion ``base`` (N rows) + level count.

    The noisy side is *not* block-padded (``block_size=None`` is forwarded to the
    completion-batch builder) so the final partial block is truncated at the last
    response token -- matching the per-token leftmost-reveal attention length.
    Block structure for the attention mask comes from the model module's
    ``block_size``, not from noisy padding. Returns ``(base, num_samples,
    num_levels)``.
    """
    if include_loss:
        base = build_fully_masked_completion_loss_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=None,
        )
    else:
        base = build_fully_masked_completion_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=None,
        )
    num_samples = base["input_ids"].shape[0]
    num_levels = count_reveal_levels(
        block_size, base["diffu_grpo_response_lengths"], max_reveal_levels
    )
    return base, num_samples, num_levels


def make_reveal_level_view(
    base: BatchedDataDict[Any],
    level: int,
    block_size: int,
    harvest_keys: tuple[str, ...],
) -> BatchedDataDict[Any]:
    """Derive the reveal-level-``level`` view (N rows) from a fully-masked base.

    Reveals the first ``level`` tokens of every block (real tokens) and leaves the
    rest MASK; ``harvest_keys`` (score and/or loss mask) are set to the per-row
    harvest mask (1 only at within-block offset ``level`` of valid response
    tokens). All other base fields (asymmetric-AR metadata, target ids, scattered
    advantages, ...) are passed through unchanged.
    """
    device = base["input_ids"].device
    num_samples, total_len = base["input_ids"].shape
    target_ids = base["diffu_grpo_target_ids"].to(device)
    score_mask = base["diffu_grpo_score_mask"].to(device)
    response_lengths = base["diffu_grpo_response_lengths"].to(device)
    noisy_offset = (
        int(base["diffu_grpo_noisy_response_offsets"][0].item())
        if num_samples
        else 0
    )

    col = torch.arange(total_len, device=device).unsqueeze(0)
    rel = col - noisy_offset
    within_block_off = torch.remainder(rel.clamp_min(0), int(block_size))
    in_response = (rel >= 0) & (rel < response_lengths.unsqueeze(1))

    reveal = in_response & (within_block_off < level)
    ids = torch.where(reveal, target_ids, base["input_ids"].to(device))
    harvest = (
        in_response & (within_block_off == level) & (score_mask > 0.5)
    ).to(dtype=score_mask.dtype)

    view = BatchedDataDict[Any]()
    for key, value in base.items():
        view[key] = value
    view["input_ids"] = ids
    for key in harvest_keys:
        view[key] = harvest
    view["token_mask"] = harvest
    view["block_reveal_harvest_mask"] = harvest
    view["block_reveal_reveal_level"] = torch.full(
        (num_samples,), int(level), device=device, dtype=torch.long
    )
    view["block_reveal_sample_index"] = torch.arange(
        num_samples, device=device, dtype=torch.long
    )
    return view


class BlockJustGRPORevealSchedule(BatchedDataDict[Any]):
    """The reveal-level schedule, presented to Megatron as a microbatch source.

    Holds the fully-masked ``base`` (N samples) and emits the model inputs for
    each reveal level in turn: for level ``j`` it reveals the first ``j`` tokens of
    every block, then microbatches the N samples the *standard* way (delegating to
    ``BatchedDataDict.make_microbatch_iterator``). The only block-reveal-specific
    structure is the outer loop over reveal levels; sample microbatching is
    ordinary.

    Used as the training "batch" so a single ``megatron_forward_backward``
    accumulates gradients across all reveal levels before one optimizer step. Only
    one reveal level is materialized at a time.
    """

    def configure(
        self, *, num_levels: int, block_size: int, harvest_keys: tuple[str, ...]
    ) -> "BlockJustGRPORevealSchedule":
        self._br_num_levels = int(num_levels)
        self._br_block_size = int(block_size)
        self._br_harvest_keys = tuple(harvest_keys)
        return self

    def _sample_count(self) -> int:
        if not self.data:
            return 0
        value = self.data[next(iter(self.data))]
        return value.shape[0] if torch.is_tensor(value) else len(value)

    @property
    def size(self) -> int:
        # Total microbatch-equivalents Megatron will see: one full sample batch
        # per reveal level. get_microbatch_iterator divides this by the microbatch
        # size to get num_microbatches.
        return self._br_num_levels * self._sample_count()

    def make_microbatch_iterator(
        self, microbatch_size: int
    ) -> Iterator[BatchedDataDict[Any]]:
        for level in range(self._br_num_levels):
            level_view = make_reveal_level_view(
                self, level, self._br_block_size, self._br_harvest_keys
            )
            # Sample microbatching is the standard BatchedDataDict mechanism.
            yield from level_view.make_microbatch_iterator(microbatch_size)


def scatter_block_reveal_logprobs(
    flat_logprobs: torch.Tensor,
    harvest_mask: torch.Tensor,
    sample_index: torch.Tensor,
    completion_starts: torch.Tensor,
    noisy_response_offset: int,
    original_seq_len: int,
    num_samples: int,
) -> torch.Tensor:
    """Scatter per-row harvested logprobs back to NemoRL's ``[N, S]`` convention.

    ``flat_logprobs`` is ``[rows, noisy_len]`` (logprob of the target token at each
    noisy position). Only harvest positions are kept; each ``(sample, original
    position)`` is harvested by exactly one row, so a scatter-add reconstructs the
    full per-token logprob vector without double counting.
    """
    device = flat_logprobs.device
    noisy_len = flat_logprobs.shape[1]
    output = flat_logprobs.new_zeros((num_samples, original_seq_len))
    if flat_logprobs.numel() == 0:
        return output

    harvest = harvest_mask.to(device=device)[:, :noisy_len]
    contrib = flat_logprobs * harvest
    rel = torch.arange(noisy_len, device=device) - int(noisy_response_offset)
    sample_index = sample_index.to(device=device)
    completion_starts = completion_starts.to(device=device)

    for row in range(flat_logprobs.shape[0]):
        keep = harvest[row] > 0.5
        if not bool(keep.any()):
            continue
        sample = int(sample_index[row].item())
        if sample < 0 or sample >= num_samples:
            continue
        start = int(completion_starts[row].item())
        positions = (start + rel[keep]).to(dtype=torch.long)
        valid = (positions >= 0) & (positions < original_seq_len)
        if not bool(valid.any()):
            continue
        output[sample, positions[valid]] += contrib[row, keep][valid]
    return output
