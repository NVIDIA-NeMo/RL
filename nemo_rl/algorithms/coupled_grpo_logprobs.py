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
"""CoupledGRPO logprob estimation (antithetic complementary masking).

Estimates per-token response logprobs with the LLaDA-1.5 "Coupled-GRPO" scheme:
two complementary masked forward passes over DiffuGRPO's asymmetric
``[noisy | clean]`` layout (see
``diffu_grpo_logprobs.build_fully_masked_completion_batch``). For each sample a
masking ratio ``t ~ U(0, 1)`` is drawn and every valid response token is placed
in the level-0 mask ``M`` with probability ``t``; level 1 masks the exact
complement ``valid \\ M``. Each valid response token is therefore masked in
*exactly one* of the two levels, so summing the two levels' harvested logprobs
reconstructs the full per-token logprob vector -- with lower variance than two
independent random masks (the coupling is antithetic).

The mask ``M`` is generated deterministically from a per-row seed carried in
``data["coupled_grpo_seed"]`` so that the SAME realization is used for
``prev_logprobs`` (old policy), ``reference_policy_logprobs``, and the
training-time forward of a given step -- otherwise the GRPO importance ratio is
invalid. The seed is set in ``nemo_rl/algorithms/grpo.py`` as
``seed_base + step * LARGE_PRIME + row_index`` so it also resamples every step.

The number of forward passes is the constant ``2`` on every data-parallel rank
(the mask is per-sample *data*, not control flow), so this is free of the
DP-uniform pass-count deadlock that block-reveal must guard against.

Reuses DiffuGRPO's layout/attention/post-processors wholesale and
``block_just_grpo_logprobs.scatter_block_reveal_logprobs`` for the
harvest->``[N, S]`` scatter (it is generic over harvest mask + sample index).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.algorithms.diffu_grpo_logprobs import (
    RevealLevelSchedule,
    _scatter_original_response_values,
    build_fully_masked_completion_batch,
    build_fully_masked_completion_loss_batch,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.models.policy import CoupledGRPOLogprobEstimationConfig

__all__ = [
    "get_coupled_grpo_logprob_estimation_cfg",
    "maybe_set_coupled_grpo_seed",
    "build_coupled_base",
    "make_coupled_level_view",
    "CoupledGRPORevealSchedule",
    "COUPLED_NUM_LEVELS",
    "COUPLED_PAIR_SEED_STRIDE",
]

# CoupledGRPO always runs exactly two complementary forward passes. Keeping this a
# constant (independent of per-rank response lengths) is what makes the estimator
# data-parallel safe -- every rank issues the same number of collectives. ESPO may
# draw K > 1 independent coupled pairs (2*num_pairs levels), computed in the ESPO
# worker; this DP-safety constant is NOT mutated for that.
COUPLED_NUM_LEVELS = 2

# Stride between the per-pair sub-seeds when ESPO draws K > 1 independent coupled
# pairs: pair p is seeded from ``coupled_grpo_seed + p * COUPLED_PAIR_SEED_STRIDE``.
# Distinct from the ``1_000_003`` per-step stride in ``maybe_set_coupled_grpo_seed``
# (which spans ``[0, N)`` per step) so ``(K - 1) * stride`` cannot alias a neighbour
# steps seed block for any realistic K. At num_pairs == 1 pair 0s seed is the
# raw ``coupled_grpo_seed`` -- so the RNG stream is byte-for-byte identical to today.
COUPLED_PAIR_SEED_STRIDE = 2_000_003


def get_coupled_grpo_logprob_estimation_cfg(
    cfg: dict[str, Any],
) -> "CoupledGRPOLogprobEstimationConfig":
    estimator_cfg = cfg.get("logprob_estimation", None)
    if estimator_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    if estimator_cfg["type"] != "coupled_grpo":
        raise ValueError("policy.logprob_estimation.type must be 'coupled_grpo'")
    return estimator_cfg


def maybe_set_coupled_grpo_seed(
    data: BatchedDataDict[Any],
    policy_cfg: dict[str, Any],
    step: int,
) -> None:
    """Attach the per-row CoupledGRPO mask seed to ``data`` in place.

    No-op unless ``policy_cfg`` selects the ``coupled_grpo`` or
    ``espo_block_aware`` estimator (both build CoupledGRPO's complementary mask
    pair from this seed). The seed
    is shared across a step's prev / reference / training forwards so the same
    random+complement mask realization is used everywhere (required for a valid
    GRPO ratio) and varies by ``(row, step)`` so the mask resamples each step.
    The ``1_000_003`` stride keeps each step's seeds collision-free as long as
    the batch size ``N < 1_000_003``: the per-row offset spans ``[0, N)``, so it
    must not reach into the next step's block. Call once per step on the batch
    that feeds prev/reference/training logprobs (see ``nemo_rl/algorithms/grpo.py``).
    """
    estimation_cfg = policy_cfg.get("logprob_estimation", {})
    # Block-aware ESPO reuses CoupledGRPO's complementary mask pair, built from the
    # same ``coupled_grpo_seed``; fire for it under the identical shared-seed
    # contract (one realization across prev / reference / training).
    if estimation_cfg.get("type") not in ("coupled_grpo", "espo_block_aware"):
        return
    seed_base = int(estimation_cfg.get("seed_base", 0))
    data["coupled_grpo_seed"] = (
        seed_base
        + step * 1_000_003
        + torch.arange(data["input_ids"].shape[0], dtype=torch.long)
    )


def _build_level0_mask(
    base: BatchedDataDict[Any],
    seed: torch.Tensor | None,
    level0_override: torch.Tensor | None = None,
    num_pairs: int = 1,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Draw the ``num_pairs`` per-sample level-0 masks and the valid-response mask.

    For pair ``p`` and sample ``i`` the RNG is seeded from the sub-seed
    ``s_i + p * COUPLED_PAIR_SEED_STRIDE``: draw ``t ~ U(0, 1)`` then a per-position
    ``u ~ U(0, 1)``; a valid response position is in that pair's ``M`` iff ``u < t``.
    The RNG stream is drawn over the full sequence length (not just the response) so
    the realization is a deterministic function of the sub-seed alone, independent of
    the per-sample response length. At ``num_pairs == 1`` pair 0 uses the raw seed --
    identical to the CoupledGRPO stream. Returns ``(level0_masks, valid)`` where
    ``level0_masks`` is a list of ``num_pairs`` boolean tensors ``[N, total_len]`` and
    ``valid`` is one boolean tensor ``[N, total_len]``.
    """
    device = base["input_ids"].device
    num_samples, total_len = base["input_ids"].shape
    score_mask = base["diffu_grpo_score_mask"].to(device)
    response_lengths = base["diffu_grpo_response_lengths"].to(device)
    noisy_offset = (
        int(base["diffu_grpo_noisy_response_offsets"][0].item())
        if num_samples
        else 0
    )

    col = torch.arange(total_len, device=device).unsqueeze(0)
    rel = col - noisy_offset
    in_response = (rel >= 0) & (rel < response_lengths.unsqueeze(1))
    valid = in_response & (score_mask > 0.5)

    if level0_override is not None:
        # Caller pinned M directly (already mapped to the noisy layout); use
        # it verbatim, intersected with valid. No per-row RNG / seed consumed.
        # The override path is single-pair only (verification / KL check).
        level0 = (level0_override.to(device) > 0.5) & valid
        return [level0], valid
    if seed is None:
        raise ValueError(
            "_build_level0_mask needs a seed when no level0_override is given"
        )

    seed_cpu = seed.detach().to("cpu").to(torch.int64)
    level0_masks: list[torch.Tensor] = []
    for pair in range(num_pairs):
        level0 = torch.zeros(
            (num_samples, total_len), dtype=torch.bool, device=device
        )
        for i in range(num_samples):
            gen = torch.Generator(device="cpu")
            # Pair p draws from a distinct sub-seed; at pair 0 this is the raw seed
            # (byte-for-byte the CoupledGRPO stream). manual_seed requires a
            # non-negative value; mask off the sign bit.
            sub_seed = int(seed_cpu[i].item()) + pair * COUPLED_PAIR_SEED_STRIDE
            gen.manual_seed(sub_seed & 0x7FFFFFFFFFFFFFFF)
            # Bound the per-sample masking ratio to [0.2, 0.8] (DiffuCoder coupled-GRPO
            # default). Full-range t~U(0,1) lets a level mask ~0% or ~100% of the
            # response, giving extreme/high-variance per-token conditioning.
            t = 0.2 + 0.6 * float(torch.rand((), generator=gen).item())
            u = torch.rand(total_len, generator=gen).to(device)
            level0[i] = (u < t) & valid[i]
        level0_masks.append(level0)
    return level0_masks, valid


def build_coupled_base(
    data: BatchedDataDict[Any],
    mask_token_id: int,
    pad_token_id: int,
    noisy_block_size: int | None,
    pad_to_length: int | None = None,
    include_loss: bool = False,
    num_pairs: int = 1,
) -> tuple[BatchedDataDict[Any], int, int]:
    """Build the DiffuGRPO ``[noisy | clean]`` base + the coupled level-0 mask(s).

    Unlike block-reveal, the noisy side IS block-padded (``noisy_block_size`` is
    forwarded to the completion builder, mirroring DiffuGRPO) so the partial-mask
    forward aligns to the model's block-diffusion attention windows. The
    ``num_pairs`` level-0 masks are computed once here from
    ``data["coupled_grpo_seed"]`` (pair ``p`` from a distinct sub-seed) and stored on
    the base as ``coupled_grpo_level0_mask_pair{p}``; ``coupled_grpo_level0_mask`` is
    kept as an alias for pair 0. Level view ``level`` derives from pair
    ``level // 2``. If ``data["coupled_grpo_level0_mask_override"]`` (``[N, S]``,
    1 = masked) is present it pins pair 0's ``M`` directly (single-pair verification
    path) and no seed is needed. Returns ``(base, num_samples, num_levels)`` where
    ``num_levels`` is ``2 * num_pairs`` (``0`` only for an empty batch). At
    ``num_pairs == 1`` this is byte-for-byte the CoupledGRPO base.
    """
    if include_loss:
        base = build_fully_masked_completion_loss_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=noisy_block_size,
        )
    else:
        base = build_fully_masked_completion_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=noisy_block_size,
        )
    num_samples = base["input_ids"].shape[0]
    zeros = base["diffu_grpo_score_mask"].new_zeros(base["input_ids"].shape)
    if num_samples == 0:
        base["coupled_grpo_level0_mask"] = zeros
        base["coupled_grpo_valid_mask"] = zeros
        return base, 0, 0

    override = data.get("coupled_grpo_level0_mask_override", None)
    if override is not None:
        # Verification/diagnostic path: the caller pins M (the level-0 mask)
        # instead of drawing it from the seed. The override is in the original
        # [N, S] layout (1 = masked); map it into the noisy layout the same way
        # prev_logprobs / advantages are scattered. Single-pair only: the override
        # pins exactly one mask, so multi-pair (num_pairs > 1) is unsupported here
        # rather than silently downgraded to a single pair.
        assert num_pairs == 1, (
            "coupled_grpo_level0_mask_override (provided-mask path) supports only "
            f"num_pairs == 1; got num_pairs={num_pairs}."
        )
        override_noisy = _scatter_original_response_values(
            override.to(device=base["input_ids"].device, dtype=zeros.dtype),
            total_length=base["input_ids"].shape[1],
            completion_starts=base["diffu_grpo_completion_starts"],
            response_lengths=base["diffu_grpo_response_lengths"],
        )
        level0_masks, valid = _build_level0_mask(
            base, seed=None, level0_override=override_noisy
        )
    else:
        if "coupled_grpo_seed" not in data:
            raise ValueError(
                "CoupledGRPO requires a per-row 'coupled_grpo_seed' tensor in "
                "the data batch (set in nemo_rl/algorithms/grpo.py before "
                "logprob/train) unless a 'coupled_grpo_level0_mask_override' is "
                "supplied."
            )
        level0_masks, valid = _build_level0_mask(
            base, data["coupled_grpo_seed"], num_pairs=num_pairs
        )
    for pair, level0 in enumerate(level0_masks):
        base[f"coupled_grpo_level0_mask_pair{pair}"] = level0.to(dtype=zeros.dtype)
    # Keep the un-suffixed name as an alias for pair 0 (CoupledGRPO / all existing
    # single-pair consumers read this name).
    base["coupled_grpo_level0_mask"] = base["coupled_grpo_level0_mask_pair0"]
    base["coupled_grpo_valid_mask"] = valid.to(dtype=zeros.dtype)
    return base, num_samples, COUPLED_NUM_LEVELS * len(level0_masks)


def make_coupled_level_view(
    base: BatchedDataDict[Any],
    level: int,
    harvest_keys: tuple[str, ...],
) -> BatchedDataDict[Any]:
    """Derive the level-``level`` view (N rows) from a coupled base.

    Level ``level`` belongs to pair ``level // 2``; its even level masks that pair's
    ``M`` and reveals ``valid \\ M`` as real context, its odd level masks the exact
    complement ``valid \\ M`` and reveals ``M``. ``harvest_keys`` (score and/or loss
    mask) are set to the level's harvested positions (the masked set), so within a
    pair each valid response token is harvested in exactly one of its two levels. All
    other base fields (asymmetric-AR metadata, target ids, scattered advantages, ...)
    are passed through unchanged. Reuses block-reveal's ``block_reveal_*`` field names
    so ``scatter_block_reveal_logprobs`` and the post-processors work untouched. At a
    single pair (levels 0/1) this is byte-for-byte the CoupledGRPO view.
    """
    device = base["input_ids"].device
    num_samples = base["input_ids"].shape[0]
    target_ids = base["diffu_grpo_target_ids"].to(device)
    pair = int(level) // COUPLED_NUM_LEVELS
    is_complement = int(level) % COUPLED_NUM_LEVELS
    level0 = base[f"coupled_grpo_level0_mask_pair{pair}"].to(device) > 0.5
    valid = base["coupled_grpo_valid_mask"].to(device) > 0.5

    if is_complement == 0:
        mask_set = level0 & valid
    else:
        mask_set = valid & (~level0)
    reveal = valid & (~mask_set)

    ids = torch.where(reveal, target_ids, base["input_ids"].to(device))
    harvest = mask_set.to(dtype=base["diffu_grpo_score_mask"].dtype)

    view = BatchedDataDict[Any]()
    for key, value in base.items():
        view[key] = value
    view["input_ids"] = ids
    for key in harvest_keys:
        view[key] = harvest
    view["token_mask"] = harvest
    view["block_reveal_harvest_mask"] = harvest
    view["block_reveal_sample_index"] = torch.arange(
        num_samples, device=device, dtype=torch.long
    )
    view["coupled_grpo_level"] = torch.full(
        (num_samples,), int(level), device=device, dtype=torch.long
    )
    # Route this level's pair (= level // 2) prev / reference logprobs. When the base
    # carries per-pair fields (ESPO / CoupledGRPO with K > 1 coupled pairs), overwrite
    # the standard field copied above with pair p's summed [N, S] logprobs so the loss
    # reshape [K_seq, num_masks, S] has row 2p / 2p+1 carry pair p's tensor. Pair 0 /
    # a missing key leaves the standard field untouched -- byte-for-byte at one pair
    # (no ``_pair{p > 0}`` keys exist). ``generation_logprobs`` is never routed: it is
    # the real per-token sampling logprob, correct at any harvest.
    for prefix in ("prev_logprobs", "reference_policy_logprobs"):
        pair_key = f"{prefix}_pair{pair}"
        if pair_key in base:
            view[prefix] = base[pair_key]
    return view


class CoupledGRPORevealSchedule(RevealLevelSchedule):
    """The two-level coupled schedule, presented to Megatron as a microbatch source.

    For level 0 it masks ``M`` and reveals the complement; for level 1 the reverse
    (see ``make_coupled_level_view``). Sample microbatching is the standard
    ``BatchedDataDict`` mechanism (see ``RevealLevelSchedule``). Used as the training
    "batch" so a single ``megatron_forward_backward`` accumulates gradients across
    both levels before one optimizer step.
    """

    def configure(
        self, *, num_levels: int, harvest_keys: tuple[str, ...]
    ) -> "CoupledGRPORevealSchedule":
        self._configure_levels(num_levels=num_levels, harvest_keys=harvest_keys)
        return self

    def _make_level_view(self, level: int) -> BatchedDataDict[Any]:
        return make_coupled_level_view(self, level, self._rl_harvest_keys)
