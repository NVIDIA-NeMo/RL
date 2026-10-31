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

"""Score-centering corrections shared by policy-gradient objectives."""

import torch


def compute_score_centering_term(
    policy_logprobs_at_sampler_topk: torch.Tensor,
    *,
    sampler_topk_logprobs: torch.Tensor,
    importance_sampling_mode: str,
    is_bounds: tuple[float, float] | None,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``sum_v sg(c_v) * log p_v`` whose gradient is the sampler-expected score.

    Score centering (arXiv:2609.20807, Eq. 14) with the paper's top-k estimator:
    the sampler's head ``q`` is exact and its tail is modeled as ``rho * p`` with
    ``rho`` = sampler tail mass / trainer tail mass. Without importance sampling
    the centering mass is ``q`` on the head and ``rho * p`` on the tail. With
    token-level TIS, ``w = clamp(p / q, lo, hi)``, it is
    ``q * w = clamp(p, lo * q, hi * q)`` on the head and ``alpha * p`` on the tail
    with ``alpha = clamp(1, lo * rho, hi * rho)``. Since ``sum_v p_v grad log p_v``
    is zero, the tail contributes ``-alpha * p_v`` on the head ids, so the
    coefficient is ``c_v = mass_v - alpha * p_v``.
    Untruncated IS makes this coefficient zero. ICEPOP instead retains ``p_v``
    only where ``p_v/q_v`` is in bounds, and retains the unit tail coefficient
    only where ``1/rho`` is in bounds. These rules apply to every head/tail
    candidate independently of the sampled token's weight.

    Under sampler replay the trainer renormalizes over the sampler's
    support S_q, which the head holds: slots the sampler filtered out are ``-inf``
    for both and get no coefficient, and the head holds all of p, so there is no
    tail and ``rho`` is zero. (miles uses ``rho = 1``; the two differ by a multiple
    of ``sum_{S_q} p grad log p = 0``.)

    This utility leaves the base actor objective, advantage weighting, masking,
    and normalization to the caller, so PPO and GRPO can share the estimator.
    Adding this term does not center an arbitrary clipped PPO gradient exactly.

    Args:
        policy_logprobs_at_sampler_topk: Trainer log-probs at the sampler's top-k ids, ``[B, S, k]``,
            carrying the gradient; ``-inf`` where the sampler filtered the id out.
        sampler_topk_logprobs: The sampler's log-probs at those ids, ``[B, S, k]``;
            ``-inf`` where its filters removed the id.
        importance_sampling_mode: ``off``, ``is``, ``tis`` or ``icepop``.
        is_bounds: ``(lo, hi)`` for TIS/ICEPOP, otherwise ``None``.
        token_mask: Active token/sample positions, before IS filtering. A zero
            sampled-token ICEPOP weight must not mask the correction.

    Returns:
        The per-token term ``[B, S]`` and the sampler mass the head covers ``[B, S]``.
    """
    active = token_mask.bool()
    slots = active.unsqueeze(-1) & ~torch.isneginf(sampler_topk_logprobs)
    p_log = torch.where(slots, policy_logprobs_at_sampler_topk, 0.0)
    q_log = torch.where(slots, sampler_topk_logprobs.detach(), 0.0)
    p = torch.where(slots, p_log.detach().exp(), 0.0)
    q = torch.where(slots, q_log.exp(), 0.0)
    head_mass = q.sum(-1)
    # With no trainer tail (the head holds all of p, as under replay) there is
    # nothing to model the sampler's tail on.
    trainer_tail = 1.0 - p.sum(-1)
    rho = torch.where(
        trainer_tail > 1e-6,
        (1.0 - head_mass).clamp(min=0.0) / trainer_tail.clamp(min=1e-6),
        0.0,
    )
    if importance_sampling_mode == "off":
        mass, alpha = q, rho
    elif importance_sampling_mode == "is":
        # q * (p/q) == p on the head and tail: exact IS needs no correction.
        mass, alpha = p, torch.ones_like(rho)
    elif importance_sampling_mode == "tis":
        assert is_bounds is not None
        lo, hi = is_bounds
        mass = torch.minimum(torch.maximum(p, lo * q), hi * q)
        alpha = torch.minimum(torch.maximum(torch.ones_like(rho), lo * rho), hi * rho)
    elif importance_sampling_mode == "icepop":
        assert is_bounds is not None
        lo, hi = is_bounds
        ratio = (p_log.detach() - q_log).exp()
        mass = torch.where((ratio >= lo) & (ratio <= hi), p, 0.0)
        tail_ratio = rho.reciprocal()
        # rho * (1/rho) == 1 on retained tails. Avoid 0 * inf at rho=0.
        alpha = ((tail_ratio >= lo) & (tail_ratio <= hi)).to(p.dtype)
    else:
        raise ValueError(
            f"Unsupported score-centering IS mode {importance_sampling_mode!r}"
        )
    coefficients = mass - alpha.unsqueeze(-1) * p
    term = (coefficients.detach() * p_log).sum(-1)
    return torch.where(active, term, 0.0), torch.where(active, head_mass, 0.0)
