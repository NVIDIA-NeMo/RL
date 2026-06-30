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
"""Block-aware ESPO ELBO reduction (antithetic coupled mask pair).

ESPO (arXiv:2512.03759) trains a diffusion LLM with a SEQUENCE-LEVEL clipped
objective whose likelihood proxy is an ELBO. This module adapts that ELBO to
NemotronLabs *block-diffusion* (sequential block decoding): the ELBO factorizes
over the response's blocks (clean previous blocks, partial current block) and
each block is reweighted by its OWN realized masking ratio.

ESPO scheme (b) uses the antithetic coupled pair: level 0 masks a random subset
``M`` of the response, level 1 masks its complement ``Mbar`` (every response
token is masked in exactly one level). The two masks' ELBOs are AVERAGED into one
``L_hat`` per sequence BEFORE the ratio (``compute_coupled_block_aware_elbo``).
Because the clipped ratio is nonlinear in ``L_hat`` and ``L_hat`` needs both
masks, all of a sequence's mask-variant rows must be combined before the ratio --
the ESPO worker packs every level-expanded row into ONE microbatch (level-major),
so the loss reshapes ``[M*N] -> [M, N]``, reduces each level with its own harvest
mask, averages, and backprops through all rows in one graph.

Per sequence ``n`` with per-token response logprobs ``lp[n, p]`` over the noisy
region, the harvested (masked) set, and block size ``B``::

    for each block b (contiguous B-token span of the response, last may be partial):
        Lb       = number of real response tokens in block b
        Mb       = harvested (masked) positions in block b
        if |Mb| == 0: block contributes 0   (guard div-by-zero)
        L_hat_b  = (Lb / |Mb|) * sum_{p in Mb} lp[n, p]
    L_hat[n]     = sum_b L_hat_b
    Lnorm[n]     = L_hat[n] / L_n            # L_n = total real response length

``Lnorm[n]`` is fed as a per-sequence scalar to the loss (Route B): the GSPO
sequence-ratio path gives ``ratio = exp(Lnorm_theta - Lnorm_old)`` and the k2 KL
gives a genuine ``0.5 * (Lnorm_theta - Lnorm_ref)^2`` per sequence. The same mask
realization (shared seed) is used for curr / prev / ref so the ratio is valid.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "get_espo_logprob_estimation_cfg",
    "compute_block_aware_elbo",
    "compute_coupled_block_aware_elbo",
]


def get_espo_logprob_estimation_cfg(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    estimator_cfg = cfg.get("logprob_estimation", None)
    if estimator_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    if estimator_cfg["type"] != "espo_block_aware":
        raise ValueError(
            "policy.logprob_estimation.type must be 'espo_block_aware'"
        )
    return estimator_cfg


def compute_block_aware_elbo(
    token_logprobs: torch.Tensor,
    harvest_mask: torch.Tensor,
    response_lengths: torch.Tensor,
    noisy_response_offset: int,
    block_size: int,
) -> torch.Tensor:
    """Reduce per-token response logprobs to the block-aware ELBO scalar.

    Vectorized over blocks and differentiable in ``token_logprobs`` (the harvest
    mask, response lengths, offset and block size are constants). Each block is
    reweighted by its realized ratio ``Lb / |Mb|`` (lower variance than the
    nominal masking ratio ``t``); a block with no harvested token contributes 0.

    Args:
        token_logprobs: ``[N, noisy_len]`` per-token response logprobs over the
            noisy region (curr policy with grad, or prev / ref detached).
        harvest_mask: ``[N, noisy_len]`` 1 at masked/scored positions (the single
            mask's harvested set), 0 elsewhere.
        response_lengths: ``[N]`` total real response length ``L_n`` per sequence.
        noisy_response_offset: column where the response starts in the noisy
            layout (uniform across rows).
        block_size: block-diffusion block size ``B``.

    Returns:
        ``elbo_per_seq`` ``[N]`` = ``L_hat[n]``. The ``/L`` normalization is applied
        by ``compute_coupled_block_aware_elbo`` on the averaged ELBO, not per mask.
    """
    device = token_logprobs.device
    num_samples, noisy_len = token_logprobs.shape
    response_lengths = response_lengths.to(device=device, dtype=torch.long)
    harvest = harvest_mask.to(device=device, dtype=token_logprobs.dtype)

    # Per-position block index relative to the response start; positions outside
    # the response get a sentinel block index that is never scattered into.
    col = torch.arange(noisy_len, device=device).unsqueeze(0)
    rel = col - noisy_response_offset
    in_response = (rel >= 0) & (rel < response_lengths.unsqueeze(1))
    num_blocks = max(
        (int(response_lengths.max().item()) + block_size - 1) // block_size, 1
    )
    block_idx = torch.where(
        in_response,
        torch.div(rel, block_size, rounding_mode="floor"),
        torch.full_like(rel, num_blocks),  # sentinel -> dropped below
    ).clamp_max(num_blocks)

    in_resp_f = in_response.to(token_logprobs.dtype)
    harvested = harvest * in_resp_f  # harvested positions inside the response

    # Per-(row, block) reductions via scatter-add over the block dimension.
    # The extra (num_blocks) slot absorbs the sentinel and is dropped.
    def _block_sum(values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            (num_samples, num_blocks + 1), dtype=values.dtype, device=device
        )
        out.scatter_add_(1, block_idx, values)
        return out[:, :num_blocks]

    Lb = _block_sum(in_resp_f)  # real tokens per block
    Mb = _block_sum(harvested)  # harvested positions per block
    lp_sum = _block_sum(token_logprobs * harvested)  # sum lp over harvested

    has_mask = Mb > 0
    reweight = torch.where(has_mask, Lb / Mb.clamp_min(1.0), torch.zeros_like(Mb))
    elbo_b = torch.where(has_mask, reweight * lp_sum, torch.zeros_like(lp_sum))

    elbo_per_seq = elbo_b.sum(dim=1)
    return elbo_per_seq


def compute_coupled_block_aware_elbo(
    level_logprobs: list[torch.Tensor],
    level_harvest_masks: list[torch.Tensor],
    response_lengths: torch.Tensor,
    noisy_response_offset: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average several masks' block-aware ELBOs into one per-sequence scalar.

    ESPO scheme (b): each mask ``j`` is reduced by ``compute_block_aware_elbo``
    with its OWN harvested set (so each block keeps its realized per-mask reweight
    ``Lb / |Mb^j|``); the per-mask ELBOs are then averaged BEFORE forming the
    ratio. ESPO is the coupled pair (``M == 2``), but the reduction stays generic
    in the number of masks for a future MC > 2 extension.

    Args:
        level_logprobs: ``M`` tensors ``[N, noisy_len]``, one per mask, aligned on
            the same ``N`` sequences (curr with grad, or prev / ref detached).
        level_harvest_masks: ``M`` tensors ``[N, noisy_len]``, mask ``j``'s
            harvested (masked/scored) positions.
        response_lengths: ``[N]`` total real response length ``L_n`` per sequence.
        noisy_response_offset: column where the response starts in the noisy layout.
        block_size: block-diffusion block size ``B``.

    Returns:
        ``(elbo_per_seq, lengthnorm_per_seq)``, both ``[N]``:
        ``L_hat[n] = mean_j ELBO_j[n]`` and ``Lnorm[n] = L_hat[n] / L_n``.
    """
    if not level_logprobs:
        raise ValueError("compute_coupled_block_aware_elbo needs at least one mask")
    num_masks = len(level_logprobs)
    elbo_sum: torch.Tensor | None = None
    for lp, harvest in zip(level_logprobs, level_harvest_masks):
        elbo_j = compute_block_aware_elbo(
            lp, harvest, response_lengths, noisy_response_offset, block_size
        )
        elbo_sum = elbo_j if elbo_sum is None else elbo_sum + elbo_j
    elbo_per_seq = elbo_sum / num_masks
    L_n = response_lengths.to(elbo_per_seq.dtype).clamp_min(1.0)
    return elbo_per_seq, elbo_per_seq / L_n

