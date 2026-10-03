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

"""Hybrid AR + diffusion batch construction.

Trains both generation modes of a block-diffusion checkpoint (e.g.
Nemotron-Labs-Diffusion) from a single forward pass over DiffuGRPO's
asymmetric ``[noisy | clean]`` layout:

* the **clean** half uses ordinary causal attention over the clean side only,
  so its logits are exactly a teacher-forced autoregressive forward over
  ``prompt + response``. It carries the **RL** term (GRPO clipped policy
  gradient).
* the **noisy** half holds the response with a random subset of tokens replaced
  by MASK. It carries the **cross-entropy** term -- the plain masked-diffusion
  (MDM) pretraining objective, *not* an RL objective: no ratio, no clipping, no
  advantage weighting.

Because the clean half never attends to the noisy half, the clean-side logits
are completely independent of the mask realization. ``prev_logprobs`` (clean
side) therefore stays valid no matter how the noisy mask is redrawn, so a fresh
mask may be drawn every step.

**Dual alignment via one gather.** The two halves score different things: the
noisy half predicts the token *at* each masked position, the clean half predicts
the *next* token. Rather than gather twice, ``hybrid_target_ids`` bakes the AR
shift into the target tensor -- clean position ``i`` stores clean token ``i+1``
-- so a single same-position gather serves both halves. The two terms are then
separated by ``hybrid_ce_mask`` (noisy) and ``hybrid_pg_mask`` (clean).

The loss that consumes this batch is
``nemo_rl.algorithms.loss.HybridARDiffusionLossFn``. The forward pass itself
needs an attention mask that is block-bidirectional on the noisy half and causal
on the clean half; that lives in the model/worker, not here.

**Usage.** Add these two blocks to any Megatron GRPO config. Setting
``logprob_estimation`` is what switches the loss; the worker is what builds the
batch, and GRPO refuses to start without it::

    policy:
      worker_extension_cls_fqn: nemo_rl.models.policy.workers.hybrid_ar_diffusion_megatron_policy_worker.HybridARDiffusionMegatronPolicyWorker
      logprob_estimation:
        type: hybrid_ar_diffusion
        mask_token_id: <the model's MASK id>
        ce_loss_weight: 0.1        # 0.0 recovers the RL-only baseline
        pg_loss_weight: 1.0        # 0.0 trains on CE alone
        mask_ratio_min: 0.2
        mask_ratio_max: 0.8
        elbo_weight_ce: true
        noisy_tail_mode: mask
        exclude_mask_token_from_logits: true
        seed_base: 0

The worker requires ``policy.sequence_packing.enabled=false`` and
``policy.megatron_cfg.context_parallel_size=1``; both are rejected at config
time rather than mid-rollout.
"""

from collections.abc import Mapping
from typing import Any

import torch

from nemo_rl.algorithms.diffu_grpo_logprobs import build_fully_masked_completion_batch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import (
    HybridARDiffusionLogprobEstimationConfig,
    NoisyTailMode,
)

__all__ = [
    "HYBRID_AR_DIFFUSION_WORKER_FQN",
    "HYBRID_SEED_STEP_STRIDE",
    "build_hybrid_ar_diffusion_batch",
    "draw_hybrid_noisy_mask",
    "get_hybrid_ar_diffusion_cfg",
    "maybe_set_hybrid_mask_seed",
    "unscatter_clean_aligned",
]

# The one shipped policy worker that builds this batch; set it on
# ``policy.worker_extension_cls_fqn``. Spelled out here rather than imported
# from the worker module so config validation does not pull Megatron into the
# driver process.
HYBRID_AR_DIFFUSION_WORKER_FQN = (
    "nemo_rl.models.policy.workers.hybrid_ar_diffusion_megatron_policy_worker"
    ".HybridARDiffusionMegatronPolicyWorker"
)

# Stride between per-step seed blocks. The per-row offset spans ``[0, N)``, so as
# long as the batch size stays below this stride one step's seeds cannot alias
# the next step's block.
HYBRID_SEED_STEP_STRIDE = 1_000_003


def get_hybrid_ar_diffusion_cfg(
    policy_cfg: Mapping[str, Any],
) -> HybridARDiffusionLogprobEstimationConfig:
    """Fetch and validate the hybrid estimator config off a policy config.

    Args:
        policy_cfg: The policy config (``master_config.policy``). Its
            ``logprob_estimation`` entry may be a plain dict (as loaded from
            YAML) or an already-validated config.

    Returns:
        The validated ``logprob_estimation`` sub-config.

    Raises:
        ValueError: If ``logprob_estimation`` is absent or invalid.
    """
    estimation_cfg = policy_cfg.get("logprob_estimation")
    if estimation_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    return HybridARDiffusionLogprobEstimationConfig.model_validate(estimation_cfg)


def maybe_set_hybrid_mask_seed(
    data: BatchedDataDict[Any],
    policy_cfg: Mapping[str, Any],
    step: int,
) -> None:
    """Attach the per-row noisy-mask seed to ``data`` in place.

    No-op unless ``policy_cfg`` carries a ``logprob_estimation`` config.

    This seed is **not** required for correctness -- the clean-side logprobs
    that feed the RL ratio do not depend on the noisy mask, so nothing has to
    agree across passes. It exists so a run is reproducible and resumable: the
    same step always draws the same mask.

    Args:
        data: Batch to annotate with ``hybrid_mask_seed``.
        policy_cfg: The policy config.
        step: Current GRPO step, folded into the seed so masks vary per step.
    """
    if policy_cfg.get("logprob_estimation") is None:
        return
    seed_base = get_hybrid_ar_diffusion_cfg(policy_cfg).seed_base
    data["hybrid_mask_seed"] = (
        seed_base
        + step * HYBRID_SEED_STEP_STRIDE
        + torch.arange(data["input_ids"].shape[0], dtype=torch.long)
    )


def draw_hybrid_noisy_mask(
    base: BatchedDataDict[Any],
    seed: torch.Tensor,
    *,
    mask_ratio_min: float,
    mask_ratio_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the per-sample masking ratio ``t`` and the noisy-half mask.

    For sample ``i`` the RNG is seeded from ``seed[i]``: draw
    ``t ~ U(mask_ratio_min, mask_ratio_max)``, then a per-position
    ``u ~ U(0, 1)``; a valid response position is masked iff ``u < t``. The RNG
    stream spans the full sequence length (not just the response) so the
    realization is a deterministic function of the seed alone, independent of
    the sample's response length.

    Args:
        base: The ``[noisy | clean]`` base batch.
        seed: ``[N]`` per-row seeds.
        mask_ratio_min: Lower bound on ``t``.
        mask_ratio_max: Upper bound on ``t``.

    Returns:
        Tuple ``(mask, valid, ratios)`` where ``mask`` is ``[N, T]`` bool (the
        masked/CE-scored positions), ``valid`` is ``[N, T]`` bool (all valid
        noisy response positions), and ``ratios`` is ``[N]`` float (the drawn
        ``t``, for optional ``1/t`` ELBO weighting).

    Raises:
        ValueError: If the ratio bounds are not ``0 < min <= max < 1``.
    """
    if not 0.0 < mask_ratio_min <= mask_ratio_max < 1.0:
        raise ValueError(
            "mask_ratio bounds must satisfy 0 < mask_ratio_min <= mask_ratio_max "
            f"< 1; got min={mask_ratio_min}, max={mask_ratio_max}"
        )
    device = base["input_ids"].device
    num_samples, total_len = base["input_ids"].shape
    valid = _valid_noisy_response_mask(base)

    seed_cpu = seed.detach().to("cpu").to(torch.int64)
    mask = torch.zeros((num_samples, total_len), dtype=torch.bool, device=device)
    ratios = torch.zeros(num_samples, dtype=torch.float32, device=device)
    span = mask_ratio_max - mask_ratio_min
    for i in range(num_samples):
        gen = torch.Generator(device="cpu")
        # manual_seed requires a non-negative value; mask off the sign bit.
        gen.manual_seed(int(seed_cpu[i].item()) & 0x7FFFFFFFFFFFFFFF)
        t = mask_ratio_min + span * float(torch.rand((), generator=gen).item())
        u = torch.rand(total_len, generator=gen).to(device)
        mask[i] = (u < t) & valid[i]
        ratios[i] = t
    return mask, valid, ratios


def _valid_noisy_response_mask(base: BatchedDataDict[Any]) -> torch.Tensor:
    """Return the ``[N, T]`` bool mask of scoreable noisy response positions.

    Excludes the block-padding tail: only positions holding a real generated
    response token are eligible.
    """
    device = base["input_ids"].device
    _, total_len = base["input_ids"].shape
    score_mask = base["diffu_grpo_score_mask"].to(device)
    response_lengths = base["diffu_grpo_response_lengths"].to(device)
    noisy_offset = int(base["diffu_grpo_noisy_response_offsets"][0].item())

    col = torch.arange(total_len, device=device).unsqueeze(0)
    rel = col - noisy_offset
    in_response = (rel >= 0) & (rel < response_lengths.unsqueeze(1))
    return in_response & (score_mask > 0.5)


def _build_clean_side_tensors(
    base: BatchedDataDict[Any],
    data: BatchedDataDict[Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the clean-half next-token targets and the PG loss mask.

    The clean half stores ``prompt + response`` verbatim at offset
    ``noisy_length``. Position ``noisy_length + i`` predicts clean token
    ``i + 1``, so the target tensor is the clean span shifted left by one and
    the PG mask marks positions whose *next* clean token is a response token.
    The final clean position has no successor and is always excluded.

    Args:
        base: The ``[noisy | clean]`` base batch.
        data: The original rollout batch (for ``token_mask``).

    Returns:
        Tuple ``(clean_targets, pg_mask)``, both ``[N, T]``; ``clean_targets``
        is zero outside the clean half and ``pg_mask`` is bool.
    """
    device = base["input_ids"].device
    num_samples, total_len = base["input_ids"].shape
    noisy_length = int(base["diffu_grpo_noisy_lengths"][0].item())
    clean_lengths = base["diffu_grpo_clean_lengths"].to(device)
    base_targets = base["diffu_grpo_target_ids"].to(device)
    orig_token_mask = data["token_mask"].to(device) > 0.5

    clean_targets = torch.zeros_like(base_targets)
    pg_mask = torch.zeros((num_samples, total_len), dtype=torch.bool, device=device)
    for i in range(num_samples):
        clean_len = int(clean_lengths[i].item())
        if clean_len <= 1:
            continue
        # Position noisy_length + j predicts clean token j + 1, for
        # j in [0, clean_len - 1).
        span = slice(noisy_length, noisy_length + clean_len - 1)
        clean_targets[i, span] = base_targets[
            i, noisy_length + 1 : noisy_length + clean_len
        ]
        # Score only where the predicted token (j + 1) is a response token.
        pg_mask[i, span] = orig_token_mask[i, 1:clean_len]
    return clean_targets, pg_mask


def _scatter_clean_aligned(
    values: torch.Tensor,
    base: BatchedDataDict[Any],
) -> torch.Tensor:
    """Shift an original ``[N, S]`` per-token tensor into clean-half alignment.

    ``advantages`` and ``prev_logprobs`` are indexed by the token they describe,
    so the value for clean target ``i + 1`` must land on the position that
    predicts it: ``noisy_length + i``.

    Args:
        values: ``[N, S]`` per-token tensor in the original rollout layout.
        base: The ``[noisy | clean]`` base batch.

    Returns:
        ``[N, T]`` tensor aligned with the clean-half prediction positions.
    """
    device = base["input_ids"].device
    num_samples, total_len = base["input_ids"].shape
    noisy_length = int(base["diffu_grpo_noisy_lengths"][0].item())
    clean_lengths = base["diffu_grpo_clean_lengths"].to(device)
    values = values.to(device)

    out = torch.zeros((num_samples, total_len), dtype=values.dtype, device=device)
    for i in range(num_samples):
        clean_len = min(int(clean_lengths[i].item()), values.shape[1])
        if clean_len <= 1:
            continue
        out[i, noisy_length : noisy_length + clean_len - 1] = values[i, 1:clean_len]
    return out


def unscatter_clean_aligned(
    values: torch.Tensor,
    base: BatchedDataDict[Any],
    original_seq_len: int,
) -> torch.Tensor:
    """Invert the clean-half alignment, mapping clean positions back to ``[N, S]``.

    Clean position ``noisy_length + j`` scores original token ``j + 1``, so the
    inverse shifts the clean slice right by one. Index 0 stays zero: the first
    token has no predecessor to condition on, matching the autoregressive
    convention where ``prev_logprobs[:, 0]`` is unused.

    Kept next to the forward direction (rather than in a worker) so the
    round-trip is unit-testable without importing Megatron.

    Args:
        values: ``[N, T]`` per-position tensor in the ``[noisy | clean]`` layout.
        base: The ``[noisy | clean]`` batch the values were produced from.
        original_seq_len: Sequence length ``S`` of the original rollout layout.

    Returns:
        ``[N, S]`` tensor in the original per-token convention.
    """
    device = values.device
    noisy_length = int(base["diffu_grpo_noisy_lengths"][0].item())
    clean_lengths = base["diffu_grpo_clean_lengths"].to(device)

    output = torch.zeros(
        (values.shape[0], original_seq_len), dtype=values.dtype, device=device
    )
    for batch_idx in range(values.shape[0]):
        clean_len = min(int(clean_lengths[batch_idx].item()), original_seq_len)
        if clean_len <= 1:
            continue
        output[batch_idx, 1:clean_len] = values[
            batch_idx, noisy_length : noisy_length + clean_len - 1
        ]
    return output


def build_hybrid_ar_diffusion_batch(
    data: BatchedDataDict[Any],
    *,
    mask_token_id: int,
    pad_token_id: int,
    mask_ratio_min: float,
    mask_ratio_max: float,
    noisy_tail_mode: NoisyTailMode,
    pad_to_length: int | None = None,
    block_size: int | None = None,
    eos_token_id: int | None = None,
) -> BatchedDataDict[Any]:
    """Build the hybrid AR + diffusion training batch.

    Produces one ``[noisy | clean]`` sequence per sample where the noisy half is
    randomly masked (CE target) and the clean half is a verbatim teacher-forcing
    copy (RL target). A single same-position gather against
    ``hybrid_target_ids`` serves both halves; ``hybrid_ce_mask`` and
    ``hybrid_pg_mask`` select which positions feed which term.

    Args:
        data: Rollout batch. Must carry ``input_ids``, ``input_lengths``,
            ``token_mask`` and ``hybrid_mask_seed`` (see
            ``maybe_set_hybrid_mask_seed``). ``advantages``, ``prev_logprobs``
            and ``generation_logprobs`` are re-aligned to the clean half when
            present.
        mask_token_id: Token id used for masked positions.
        pad_token_id: Token id used for padding.
        mask_ratio_min: Lower bound on the per-sample masking ratio ``t``.
        mask_ratio_max: Upper bound on the per-sample masking ratio ``t``.
        noisy_tail_mode: How to fill the noisy block-padding tail.
        pad_to_length: Round the total sequence length up to this multiple.
        block_size: Block size used to pad the noisy side to a whole block.
        eos_token_id: Required when ``noisy_tail_mode`` is ``"eos"``.

    Returns:
        The training batch, carrying the base ``diffu_grpo_*`` layout fields plus
        ``hybrid_target_ids``, ``hybrid_ce_mask``, ``hybrid_pg_mask``,
        ``hybrid_mask_ratio``, and clean-aligned ``advantages`` /
        ``prev_logprobs`` / ``generation_logprobs`` when those were present on
        ``data``.

    Raises:
        ValueError: If ``hybrid_mask_seed`` is absent from ``data`` or the mask
            ratio bounds are invalid.
    """
    base = build_fully_masked_completion_batch(
        data,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        noisy_tail_mode=noisy_tail_mode,
        pad_to_length=pad_to_length,
        block_size=block_size,
        eos_token_id=eos_token_id,
    )
    num_samples = base["input_ids"].shape[0]
    if num_samples == 0:
        return base
    if "hybrid_mask_seed" not in data:
        raise ValueError(
            "build_hybrid_ar_diffusion_batch requires data['hybrid_mask_seed']; "
            "call maybe_set_hybrid_mask_seed in the GRPO loop first"
        )

    device = base["input_ids"].device
    noisy_length = int(base["diffu_grpo_noisy_lengths"][0].item())
    base_targets = base["diffu_grpo_target_ids"].to(device)

    ce_mask, valid, ratios = draw_hybrid_noisy_mask(
        base,
        data["hybrid_mask_seed"],
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
    )
    clean_targets, pg_mask = _build_clean_side_tensors(base, data)

    # Reveal every valid response token that was NOT selected for masking; the
    # selected ones keep the MASK id the fully-masked base already placed there.
    reveal = valid & (~ce_mask)
    base["input_ids"] = torch.where(reveal, base_targets, base["input_ids"].to(device))

    # Noisy half predicts the token at its own position; clean half predicts the
    # next token. One tensor, so one gather covers both.
    hybrid_targets = torch.zeros_like(base_targets)
    hybrid_targets[:, :noisy_length] = base_targets[:, :noisy_length]
    hybrid_targets[:, noisy_length:] = clean_targets[:, noisy_length:]

    base["hybrid_target_ids"] = hybrid_targets
    base["hybrid_ce_mask"] = ce_mask.to(dtype=base["diffu_grpo_score_mask"].dtype)
    base["hybrid_pg_mask"] = pg_mask.to(dtype=base["diffu_grpo_score_mask"].dtype)
    base["hybrid_mask_ratio"] = ratios
    # Both loss terms are normalized by ``global_valid_toks``, which a worker
    # reduces from the *original* batch's ``token_mask`` (as
    # ``sum(token_mask[:, 1:] * sample_mask)``). That count equals
    # ``hybrid_pg_mask.sum()``: every response token yields exactly one PG
    # position, and the ``[:, 1:]`` slice can only drop index 0, which is always
    # a prompt token. Aligning ``token_mask`` here keeps any consumer that reads
    # it off the transformed batch pointed at the scored positions.
    base["token_mask"] = base["hybrid_pg_mask"]

    # generation_logprobs rides along for the token_mult_prob_error diagnostic;
    # like advantages/prev_logprobs it is indexed by the token it scores, so the
    # same clean-side alignment applies.
    for key in ("advantages", "prev_logprobs", "generation_logprobs"):
        if key in data:
            base[key] = _scatter_clean_aligned(data[key], base)
    return base
