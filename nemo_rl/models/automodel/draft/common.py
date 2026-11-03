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
#
# Extends nemo_automodel.components.speculative.dspark.common (3rdparty
# Automodel r0.6.0). Unchanged helpers are re-exported from there; only the
# RL-co-training deltas live here:
# - DSparkForwardOutput: adds ``first_supervised_slot`` for the dflash
#   bonus-anchor layout (slot 0 carries no loss).
# - build_anchor_candidate_mask: gates anchors on the FIRST TARGET token's
#   loss mask rather than the anchor token's own mask, so RL rollouts
#   supervise the prompt-to-first-response transition (where inference-time
#   drafting starts) and one-token responses still produce draft signal.
# - sample_anchor_positions: same algorithm as upstream, kept local because
#   the upstream function hard-binds upstream ``build_anchor_candidate_mask``
#   internally — importing it would silently drop the RL anchor gating.
# - build_eval_mask: generalizes the per-slot label offset to
#   ``label_indices - anchor_positions`` and adds ``supervised_from_slot``
#   handling for the dflash at-position layout.
from dataclasses import dataclass
from typing import Optional

import torch
from nemo_automodel.components.speculative.dspark.common import (  # noqa: F401
    AcceptRatePredictor,
    context_doc_ids,
    create_noise_embed,
    create_position_ids,
    extract_context_feature,
    pin_rope_inv_freq_fp32,
    validate_target_layer_ids,
)


@dataclass
class DSparkForwardOutput:
    """Outputs for one DSpark training forward.

    Shape symbols:
        batch_size: number of samples in the batch
        seq_len: source sequence length
        num_anchors: sampled anchor blocks per sample
        block_size: number of draft positions per anchor
        vocab_size: vocabulary size

    The sampler keeps anchors whose first draft target is enabled by
    ``loss_mask``. Later slots are supervised only while they remain inside
    ``seq_len`` and form a contiguous enabled prefix. Dummy anchors can still
    appear when a sample has too few valid anchors; they are masked out by
    ``block_keep_mask`` and ``eval_mask``.
    """

    # [batch_size, num_anchors, block_size, vocab_size]
    draft_logits: torch.Tensor
    # [batch_size, num_anchors, block_size]
    target_ids: torch.Tensor
    # [batch_size, num_anchors, block_size]
    eval_mask: torch.Tensor
    # [batch_size, num_anchors]
    block_keep_mask: torch.Tensor
    # [batch_size, num_anchors, block_size]
    confidence_pred: Optional[torch.Tensor] = None
    # [batch_size, num_anchors, block_size, vocab_size]
    aligned_target_logits: Optional[torch.Tensor] = None
    # First supervised block slot: 0 for the next-token (dspark) layout, 1
    # for the bonus-anchor (dflash) layout where slot 0 carries no loss.
    # Loss-position weighting (loss_decay_gamma) counts proposals from here.
    first_supervised_slot: int = 0


def build_anchor_candidate_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    doc_remaining: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask of anchor positions whose first predicted token is supervised.

    Args:
        seq_len: source sequence length.
        loss_mask: [batch_size, seq_len] per-token supervision mask.
        doc_remaining: [batch_size, seq_len] tokens remaining in the current
            packed document, or None when unpacked.

    Returns:
        [batch_size, seq_len - 1] boolean mask; True at anchor position p
        when token p + 1 is supervised (and, when packing, stays inside the
        anchor's document).
    """
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        return loss_mask[:, :0].bool()

    # An anchor is valid when the token it predicts (anchor + 1) is inside the
    # supervised region; the anchor token itself only seeds the block and may
    # sit outside it. In RL rollouts the response's first token is anchored at
    # the final prompt token (loss mask 0) — exactly where speculative
    # drafting starts at inference — so gating on the anchor's own mask (as
    # the upstream pretraining sampler does) would drop the prompt-to-response
    # transition and leave one-token responses with no draft supervision.
    valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
    if doc_remaining is not None:
        # Packing: the anchor's first target (anchor + 1) must stay in the anchor's
        # document, i.e. at least one real token follows the anchor in its document.
        valid = valid & (doc_remaining[:, :num_candidates] >= 1)
    return valid


def sample_anchor_positions(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    device: torch.device,
    doc_remaining: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample up to num_anchors valid anchor positions per sample.

    Args:
        seq_len: source sequence length.
        loss_mask: [batch_size, seq_len] per-token supervision mask.
        num_anchors: maximum anchors to sample per sample.
        device: device to allocate outputs on.
        doc_remaining: packed-document remaining-token counts, or None.
        generator: optional RNG for a reproducible draw (must be seeded
            identically across TP peers sharing the same draft replica).

    Returns:
        (anchors, keep_mask): anchors is [batch_size, num_anchors] sorted
        anchor token positions (zero-padded past the valid count); keep_mask
        marks which of those are real vs. padding.
    """
    valid = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
        doc_remaining=doc_remaining,
    )
    valid_counts = valid.sum(dim=1)
    bsz = loss_mask.shape[0]
    num_candidates = valid.shape[1]
    max_n = int(num_anchors)
    if num_candidates == 0:
        anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
        keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
        return anchors, keep_mask

    indices = (
        torch.arange(num_candidates, device=device)
        .unsqueeze(0)
        .expand(
            bsz,
            -1,
        )
    )
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    )
    # An explicit generator makes the draw a pure function of its seed: the
    # draft is replicated across TP peers (FSDP over dp_cp only), so every
    # peer must sample identical anchors or the replicas drift apart and the
    # refit exports rank-dependent draft weights.
    random_vals = torch.rand(bsz, num_candidates, device=device, generator=generator)
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)
    if num_candidates < max_n:
        pad = torch.full(
            (bsz, max_n - num_candidates),
            seq_len + 1,
            dtype=gathered.dtype,
            device=device,
        )
        gathered = torch.cat([gathered, pad], dim=1)
    anchors = gathered[:, :max_n].sort(dim=1).values
    keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
        valid_counts.unsqueeze(1).clamp(max=max_n)
    )
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
    return anchors, keep_mask


def build_eval_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    label_indices: torch.Tensor,
    safe_label_indices: torch.Tensor,
    block_keep_mask: torch.Tensor,
    doc_remaining: Optional[torch.Tensor] = None,
    anchor_positions: Optional[torch.Tensor] = None,
    supervised_from_slot: int = 0,
) -> torch.Tensor:
    """Per-slot supervision mask for the anchored blocks.

    ``supervised_from_slot`` marks the first supervised block slot: 0 when the
    anchor slot itself predicts (dspark next-token layout), 1 when the anchor is
    an unsupervised bonus token (dflash at-position layout). Leading
    unsupervised slots are treated as valid links for the trailing cumprod
    (their label — the anchor's own token — may sit outside the loss mask
    without invalidating the rest of the block) and are zeroed afterwards.
    """
    target_valid = label_indices < seq_len
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
        2,
        safe_label_indices,
    )
    eval_mask = target_valid & (target_loss_mask > 0.5)
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
    if doc_remaining is not None:
        # Packing: truncate each block at its anchor's document boundary. Slot k
        # predicts the token at ``anchor + offset_k`` (offset_k = label_indices
        # - anchor), which stays in the document iff ``offset_k <=
        # doc_remaining[anchor]``. The trailing cumprod then drops every slot at
        # or beyond the boundary (a partial in-document block), matching how
        # the block is truncated at the sequence end.
        step = label_indices - anchor_positions.unsqueeze(-1)
        anchor_remaining = torch.gather(doc_remaining, 1, anchor_positions).unsqueeze(
            -1
        )
        eval_mask = eval_mask & (step <= anchor_remaining)
    if supervised_from_slot > 0:
        eval_mask[..., :supervised_from_slot] = True
    eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
    if supervised_from_slot > 0:
        eval_mask[..., :supervised_from_slot] = False
    return eval_mask


__all__ = [
    "DSparkForwardOutput",
    "AcceptRatePredictor",
    "extract_context_feature",
    "validate_target_layer_ids",
    "context_doc_ids",
    "build_anchor_candidate_mask",
    "sample_anchor_positions",
    "build_eval_mask",
    "create_position_ids",
    "create_noise_embed",
    "pin_rope_inv_freq_fp32",
]
