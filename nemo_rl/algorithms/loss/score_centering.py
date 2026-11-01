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

from nemo_rl.algorithms.loss.clipped_pg import adaptive_ppo_clip_kept_mask


def compute_score_centering_term(
    policy_logprobs_at_sampler_topk: torch.Tensor,
    *,
    sampler_topk_logprobs: torch.Tensor,
    importance_sampling_mode: str,
    is_bounds: tuple[float, float] | None,
    token_mask: torch.Tensor,
    adaptive_ppo_clip_advantages: torch.Tensor | None = None,
    adaptive_ppo_clip_outer_bound: float | None = None,
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
    only where ``1/rho`` is in bounds. adaptive-ppo-clip follows the same rule
    with the bounds its filter picks by the sign of the position's advantage
    (``adaptive_ppo_clip_kept_mask``). These rules apply to every head/tail
    candidate independently of the sampled token's weight.

    This utility leaves the base actor objective, advantage weighting, masking,
    and normalization to the caller, so PPO and GRPO can share the estimator;
    adaptive-ppo-clip reads only the advantage's sign.
    Adding this term does not center an arbitrary clipped PPO gradient exactly.

    Args:
        policy_logprobs_at_sampler_topk: Trainer log-probs at the sampler's top-k ids, ``[B, S, k]``,
            carrying the gradient.
        sampler_topk_logprobs: The sampler's log-probs at those ids, ``[B, S, k]``.
        importance_sampling_mode: ``off``, ``is``, ``tis``, ``icepop`` or
            ``adaptive-ppo-clip``.
        is_bounds: ``(lo, hi)`` for TIS/ICEPOP; for adaptive-ppo-clip, the
            negative-advantage lower bound and the positive-advantage upper
            bound including the controller's offset ``k``. Otherwise ``None``.
        token_mask: Active token/sample positions, before IS filtering. A zero
            sampled-token ICEPOP or adaptive-ppo-clip weight must not mask the
            correction.
        adaptive_ppo_clip_advantages: Per-token advantages ``[B, S]``, whose
            sign selects adaptive-ppo-clip's bound. Read by that mode only.
        adaptive_ppo_clip_outer_bound: adaptive-ppo-clip's upper bound for
            either advantage sign. Read by that mode only.

    Returns:
        The per-token term ``[B, S]`` and the sampler mass the head covers ``[B, S]``.
    """
    active = token_mask.bool()
    p_log = torch.where(active.unsqueeze(-1), policy_logprobs_at_sampler_topk, 0.0)
    q_log = torch.where(active.unsqueeze(-1), sampler_topk_logprobs.detach(), 0.0)
    p = p_log.detach().exp()
    q = q_log.exp()
    head_mass = q.sum(-1)
    rho = (1.0 - head_mass).clamp(min=0.0) / (1.0 - p.sum(-1)).clamp(min=1e-6)
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
    elif importance_sampling_mode == "adaptive-ppo-clip":
        assert is_bounds is not None
        assert adaptive_ppo_clip_advantages is not None
        assert adaptive_ppo_clip_outer_bound is not None
        lo, hi = is_bounds
        ratio = (p_log.detach() - q_log).exp()
        head_kept = adaptive_ppo_clip_kept_mask(
            ratio,
            advantages=adaptive_ppo_clip_advantages.unsqueeze(-1),
            lower_bound=lo,
            upper_bound=hi,
            outer_bound=adaptive_ppo_clip_outer_bound,
        )
        mass = torch.where(head_kept, p, 0.0)
        # As with ICEPOP, the tail is kept or dropped whole at its shared ratio
        # 1/rho. At rho=0 that ratio is inf, which a finite outer bound drops.
        alpha = adaptive_ppo_clip_kept_mask(
            rho.reciprocal(),
            advantages=adaptive_ppo_clip_advantages,
            lower_bound=lo,
            upper_bound=hi,
            outer_bound=adaptive_ppo_clip_outer_bound,
        ).to(p.dtype)
    else:
        raise ValueError(
            f"Unsupported score-centering IS mode {importance_sampling_mode!r}"
        )
    coefficients = mass - alpha.unsqueeze(-1) * p
    term = (coefficients.detach() * p_log).sum(-1)
    return torch.where(active, term, 0.0), torch.where(active, head_mass, 0.0)
