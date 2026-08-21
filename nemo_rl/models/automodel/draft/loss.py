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
# Extends nemo_automodel/components/speculative/dspark/loss.py (3rdparty
# Automodel r0.6.0, byte-identical to vendor base 6f423feb0). Kept as a local
# module because the loss normalization contract differs:
# denominators are all-reduced over an explicit data-parallel process group and
# the loss is returned globally normalized WITHOUT the upstream world-size
# multiply, because nemo-rl's automodel_forward_backward already scales each
# microbatch loss by dp_size * cp_size before backward to undo FSDP's gradient
# averaging. Every rank must call compute_dspark_loss the same number of times
# per step (nemo-rl pads ranks with dummy microbatches to keep FSDP2
# collectives in lockstep, and those dummy calls must also reach this loss).
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import DSparkForwardOutput

_PROBABILITY_CHUNK_TOKENS = 128


def _all_reduce_loss_denominators(
    loss_terms: dict[str, torch.Tensor],
    *,
    process_group: Optional[dist.ProcessGroup],
) -> dict[str, torch.Tensor]:
    denominators = {}
    for key in ("ce_loss_den", "l1_loss_den", "confidence_loss_den"):
        tensor = loss_terms[key].detach().clone()
        if dist.is_initialized() and dist.get_world_size(process_group) > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
        denominators[key] = tensor
    return denominators


def _build_loss_weight_mask(
    *,
    eval_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
    loss_decay_gamma: Optional[float],
    first_supervised_slot: int = 0,
) -> torch.Tensor:
    """Per-slot loss weights: eval mask times the positional decay.

    Decay positions count PROPOSALS, not raw block slots: the k-th supervised
    slot gets ``exp(-k / gamma)`` so the first proposed token always has
    weight 1. Under the dflash bonus-anchor layout slot 0 is unsupervised
    (``first_supervised_slot = 1``); its clamped weight is irrelevant because
    the eval mask zeroes it.
    """
    loss_weight_mask = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = (
            torch.arange(block_size, device=device).view(1, 1, -1)
            - int(first_supervised_slot)
        ).clamp(min=0)
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights
    return loss_weight_mask


def _l1_probability_distance_chunk(
    *,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    target_probs = torch.softmax(target_logits.float(), dim=-1)
    return (draft_probs - target_probs).abs().sum(dim=-1)


def _compute_l1_dist_per_token(
    *,
    draft_logits: torch.Tensor,
    aligned_target_logits: torch.Tensor,
    chunk_size: int = _PROBABILITY_CHUNK_TOKENS,
) -> torch.Tensor:
    """Compute exact FP32 probability L1 distances without full-vocab temporaries."""
    output_shape = draft_logits.shape[:-1]
    vocab_size = draft_logits.shape[-1]
    flat_draft = draft_logits.reshape(-1, vocab_size)
    flat_target = aligned_target_logits.reshape(-1, vocab_size)
    distances = []
    for start in range(0, flat_draft.shape[0], chunk_size):
        draft_chunk = flat_draft[start : start + chunk_size]
        target_chunk = flat_target[start : start + chunk_size]
        if torch.is_grad_enabled() and draft_chunk.requires_grad:
            distance = checkpoint(
                _l1_probability_distance_chunk,
                draft_logits=draft_chunk,
                target_logits=target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            distance = _l1_probability_distance_chunk(
                draft_logits=draft_chunk,
                target_logits=target_chunk,
            )
        distances.append(distance)
    return torch.cat(distances).reshape(output_shape)


def _compute_accept_rate_3d(
    l1_dist_per_token: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if l1_dist_per_token is None:
        return None
    return (1.0 - 0.5 * l1_dist_per_token).clamp_(0.0, 1.0)


def _compute_local_l1_term(
    *,
    l1_dist_per_token: Optional[torch.Tensor],
    loss_weight_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = loss_weight_mask.new_zeros(())
    if l1_dist_per_token is None:
        return zero, zero
    l1_loss_num = (l1_dist_per_token * loss_weight_mask).sum()
    l1_loss_den = loss_weight_mask.sum()
    return l1_loss_num, l1_loss_den


def _collect_acceptance_diagnostics(
    *,
    outputs: DSparkForwardOutput,
    accept_rate_3d: Optional[torch.Tensor],
    loss_weight_mask: torch.Tensor,
    has_confidence: bool,
) -> dict[str, torch.Tensor]:
    """Per-batch numerator/denominator sums for the acceptance diagnostics.

    ``accept_rate_3d`` is the TV-derived per-token acceptance probability. Every
    diagnostic is returned as an unreduced ``(num, den)`` sum; the recipe sums both
    across the log window and the data-parallel group and forms the global ratio
    once (``sum(num) / sum(den)``), so per-micro-batch token-count imbalance never
    biases the reported value. Returns zero sums when no teacher signal is
    available (``accept_rate_3d is None``).
    """
    zero = outputs.draft_logits.new_zeros((), dtype=torch.float32)
    # Diagnostics index PROPOSALS: the dflash bonus-anchor slot
    # (first_supervised_slot = 1) carries no acceptance signal — its
    # always-false eval-mask entry would zero every cumprod prefix and shift
    # the per-position rates — so all slot-indexed tensors are sliced to the
    # supervised region first. For dspark (slot offset 0) this is a no-op.
    first = int(outputs.first_supervised_slot)
    block_size = outputs.draft_logits.shape[2]
    num_proposals = block_size - first
    pos_zero = outputs.draft_logits.new_zeros((num_proposals,), dtype=torch.float32)
    terms = {
        "accept_rate_pos_num": pos_zero,
        "accept_rate_pos_den": pos_zero,
        "tau_num": zero,
        "tau_den": zero,
        "confidence_abs_error_num": zero,
        "confidence_bias_num": zero,
        "confidence_cumprod_bias_num": zero,
        "confidence_diag_den": zero,
    }
    if accept_rate_3d is None:
        return terms

    eval_mask = outputs.eval_mask[..., first:].to(torch.float32)
    accept_rate = accept_rate_3d[..., first:]
    weight_mask = loss_weight_mask[..., first:]
    valid_accept_rate = accept_rate * eval_mask
    # Per-proposal acceptance (accept_rate@k): sum over (batch, blocks) so the
    # recipe can DP-all-reduce the numerator/denominator and form a global per-k
    # ratio; the aggregate accept_rate is just sum(num)/sum(den) over positions.
    terms["accept_rate_pos_num"] = valid_accept_rate.sum(dim=(0, 1))
    terms["accept_rate_pos_den"] = eval_mask.sum(dim=(0, 1))

    # Expected accepted prefix length per block: a draft token survives only when
    # every earlier PROPOSAL in its block is also accepted, hence the running
    # product over the supervised slots. The +1 counts the verified token that
    # seeds the block (the anchor prediction for dspark, the bonus token for
    # dflash).
    valid_blocks = (outputs.block_keep_mask & (eval_mask > 0).any(dim=-1)).to(
        torch.float32
    )
    tau_per_block = valid_accept_rate.cumprod(dim=-1).sum(dim=-1) + 1.0
    terms["tau_num"] = (tau_per_block * valid_blocks).sum()
    terms["tau_den"] = valid_blocks.sum()

    if has_confidence and outputs.confidence_pred is not None:
        confidence_probs = outputs.confidence_pred.float().sigmoid()[..., first:]
        confidence_error = confidence_probs - accept_rate
        terms["confidence_abs_error_num"] = (confidence_error.abs() * weight_mask).sum()
        terms["confidence_bias_num"] = (confidence_error * weight_mask).sum()
        confidence_prefix = (confidence_probs * eval_mask).cumprod(dim=-1)
        target_prefix = (accept_rate * eval_mask).cumprod(dim=-1)
        terms["confidence_cumprod_bias_num"] = (
            (confidence_prefix - target_prefix) * weight_mask
        ).sum()
        terms["confidence_diag_den"] = weight_mask.sum()
    return terms


def _collect_local_terms(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],
    l1_loss_alpha: float,
) -> tuple[dict[str, torch.Tensor], bool]:
    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    _, _, block_size, vocab_size = draft_logits.shape
    device = draft_logits.device

    loss_weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=block_size,
        device=device,
        loss_decay_gamma=loss_decay_gamma,
        first_supervised_slot=outputs.first_supervised_slot,
    )
    flat_logits = draft_logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1)
    flat_weights = loss_weight_mask.reshape(-1)
    loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    ce_loss_num = (loss_per_token * flat_weights).sum()
    ce_loss_den = flat_weights.sum()
    aligned_target_logits = outputs.aligned_target_logits
    if aligned_target_logits is not None:
        # Teacher signal: never backprop into the (possibly trainable) lm_head through it.
        aligned_target_logits = aligned_target_logits.detach()
    zero = ce_loss_num.new_zeros(())
    has_confidence = outputs.confidence_pred is not None
    assert (
        l1_loss_alpha <= 0 and not has_confidence
    ) or aligned_target_logits is not None, (
        "aligned_target_logits is required for the L1 loss or confidence head."
    )
    l1_dist_per_token = None
    if aligned_target_logits is not None and (l1_loss_alpha > 0 or has_confidence):
        l1_dist_per_token = _compute_l1_dist_per_token(
            draft_logits=draft_logits,
            aligned_target_logits=aligned_target_logits,
        )
    accept_rate_3d = _compute_accept_rate_3d(l1_dist_per_token)
    if l1_loss_alpha > 0:
        l1_loss_num, l1_loss_den = _compute_local_l1_term(
            l1_dist_per_token=l1_dist_per_token,
            loss_weight_mask=loss_weight_mask,
        )
    else:
        l1_loss_num = zero
        l1_loss_den = zero

    confidence_loss_num = zero
    confidence_loss_den = zero
    if has_confidence:
        assert accept_rate_3d is not None, (
            "aligned_target_logits is required when the confidence head is enabled."
        )
        confidence_targets = accept_rate_3d.detach()
        confidence_errors = (
            F.binary_cross_entropy_with_logits(
                outputs.confidence_pred.float(),
                confidence_targets,
                reduction="none",
            )
            * loss_weight_mask
        )
        confidence_loss_num = confidence_errors.sum()
        confidence_loss_den = loss_weight_mask.sum()

    loss_terms = {
        "ce_loss_num": ce_loss_num,
        "ce_loss_den": ce_loss_den,
        "l1_loss_num": l1_loss_num,
        "l1_loss_den": l1_loss_den,
        "confidence_loss_num": confidence_loss_num,
        "confidence_loss_den": confidence_loss_den,
    }
    with torch.no_grad():
        loss_terms.update(
            _collect_acceptance_diagnostics(
                outputs=outputs,
                accept_rate_3d=accept_rate_3d,
                loss_weight_mask=loss_weight_mask,
                has_confidence=has_confidence,
            )
        )
    return loss_terms, has_confidence


def _build_loss(
    *,
    loss_terms: dict[str, torch.Tensor],
    global_denominators: dict[str, torch.Tensor],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
    has_confidence: bool,
) -> torch.Tensor:
    # Globally normalized sum; the caller's forward/backward loop is responsible
    # for undoing gradient averaging across data-parallel ranks (nemo-rl
    # multiplies each microbatch loss by dp_size * cp_size before backward).
    ce_loss = loss_terms["ce_loss_num"] / (global_denominators["ce_loss_den"] + 1e-6)
    l1_loss = ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:
        l1_loss = loss_terms["l1_loss_num"] / (
            global_denominators["l1_loss_den"] + 1e-6
        )
    confidence_loss = ce_loss.new_zeros(())
    if has_confidence:
        confidence_loss = loss_terms["confidence_loss_num"] / (
            global_denominators["confidence_loss_den"] + 1e-6
        )
    return (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    )


def compute_dspark_loss(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
    process_group: Optional[dist.ProcessGroup] = None,
    return_terms: bool = False,
):
    loss_terms, has_confidence = _collect_local_terms(
        outputs=outputs,
        loss_decay_gamma=loss_decay_gamma,
        l1_loss_alpha=float(l1_loss_alpha),
    )
    global_denominators = _all_reduce_loss_denominators(
        loss_terms,
        process_group=process_group,
    )
    ce_loss_alpha = float(ce_loss_alpha)
    l1_loss_alpha = float(l1_loss_alpha)
    confidence_head_alpha = float(confidence_head_alpha)

    local_ce_loss = loss_terms["ce_loss_num"] / (loss_terms["ce_loss_den"] + 1e-6)
    local_l1_loss = local_ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:
        local_l1_loss = loss_terms["l1_loss_num"] / (loss_terms["l1_loss_den"] + 1e-6)
    local_confidence_loss = local_ce_loss.new_zeros(())
    if has_confidence:
        local_confidence_loss = loss_terms["confidence_loss_num"] / (
            loss_terms["confidence_loss_den"] + 1e-6
        )
    local_loss = (
        ce_loss_alpha * local_ce_loss
        + l1_loss_alpha * local_l1_loss
        + confidence_head_alpha * local_confidence_loss
    )

    backward_loss = _build_loss(
        loss_terms=loss_terms,
        global_denominators=global_denominators,
        ce_loss_alpha=ce_loss_alpha,
        l1_loss_alpha=l1_loss_alpha,
        confidence_head_alpha=confidence_head_alpha,
        has_confidence=has_confidence,
    )
    if return_terms:
        # The losses are already-normalized scalars, so they are logged as a
        # window mean. The acceptance diagnostics are returned as unreduced
        # (num, den) sums instead: the recipe reduces both across the window and
        # the DP group and divides once, giving the exact global ratio (and a
        # tau that never dips below its definitional floor of 1).
        terms = {
            "loss": local_loss.detach(),
            "ce_loss": local_ce_loss.detach(),
            "l1_loss": local_l1_loss.detach(),
            "confidence_loss": local_confidence_loss.detach(),
            "accept_rate_per_pos_num": loss_terms["accept_rate_pos_num"],
            "accept_rate_per_pos_den": loss_terms["accept_rate_pos_den"],
            "tau_num": loss_terms["tau_num"],
            "tau_den": loss_terms["tau_den"],
            "confidence_abs_error_num": loss_terms["confidence_abs_error_num"],
            "confidence_bias_num": loss_terms["confidence_bias_num"],
            "confidence_cumprod_bias_num": loss_terms["confidence_cumprod_bias_num"],
            "confidence_diag_den": loss_terms["confidence_diag_den"],
        }
        return backward_loss, terms
    return backward_loss


__all__ = [
    "compute_dspark_loss",
]
