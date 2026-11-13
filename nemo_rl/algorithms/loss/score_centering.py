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

from typing import NamedTuple

import torch

from nemo_rl.algorithms.loss.clipped_pg import adaptive_ppo_clip_kept_mask


class _SamplerHead(NamedTuple):
    active: torch.Tensor
    p_log: torch.Tensor
    q_log: torch.Tensor
    p: torch.Tensor
    q: torch.Tensor


def _sampler_head(
    policy_logprobs_at_sampler_topk: torch.Tensor,
    *,
    sampler_topk_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
) -> _SamplerHead:
    """Trainer (``p``) and sampler (``q``) log-probs and probabilities on the head.

    Every field is zero on slots of inactive positions and on slots the sampler
    filtered out (``-inf``), so no ``-inf`` or NaN reaches a sum over the head.
    ``p_log`` keeps the gradient; the other fields are detached.
    """
    active = token_mask.bool()
    slots = active.unsqueeze(-1) & ~torch.isneginf(sampler_topk_logprobs)
    p_log = torch.where(slots, policy_logprobs_at_sampler_topk, 0.0)
    q_log = torch.where(slots, sampler_topk_logprobs.detach(), 0.0)
    p = torch.where(slots, p_log.detach().exp(), 0.0)
    q = torch.where(slots, q_log.exp(), 0.0)
    return _SamplerHead(active=active, p_log=p_log, q_log=q_log, p=p, q=q)


def compute_score_centering_term(
    policy_logprobs_at_sampler_topk: torch.Tensor,
    *,
    sampler_topk_logprobs: torch.Tensor,
    importance_sampling_mode: str,
    is_bounds: tuple[float, float] | None,
    token_mask: torch.Tensor,
    sampler_replay: bool,
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

    Under sampler replay the trainer renormalizes over the sampler's
    support S_q, which the head holds: slots the sampler filtered out are ``-inf``
    for both and get no coefficient, and the head holds all of p, so there is no
    tail and ``rho`` is zero. (miles uses ``rho = 1``; the two differ by a multiple
    of ``sum_{S_q} p grad log p = 0``.)

    This utility leaves the base actor objective, advantage weighting, masking,
    and normalization to the caller, so PPO and GRPO can share the estimator;
    adaptive-ppo-clip reads only the advantage's sign.
    Adding this term does not center an arbitrary clipped PPO gradient exactly.

    Args:
        policy_logprobs_at_sampler_topk: Trainer log-probs at the sampler's top-k ids, ``[B, S, k]``,
            carrying the gradient; ``-inf`` where the sampler filtered the id out.
        sampler_topk_logprobs: The sampler's log-probs at those ids, ``[B, S, k]``;
            ``-inf`` where its filters removed the id.
        importance_sampling_mode: ``off``, ``is``, ``tis``, ``icepop`` or
            ``adaptive-ppo-clip``.
        is_bounds: ``(lo, hi)`` for TIS/ICEPOP; for adaptive-ppo-clip, the
            negative-advantage lower bound and the positive-advantage upper
            bound including the controller's offset ``k``. Otherwise ``None``.
        token_mask: Active token/sample positions, before IS filtering. A zero
            sampled-token ICEPOP or adaptive-ppo-clip weight must not mask the
            correction.
        sampler_replay: Whether the trainer log-probs are renormalized over the
            sampler's support, so the head holds all of p and ``rho`` is zero.
            Otherwise ``rho`` models the sampler's tail on the trainer's tail,
            however small that tail is.
        adaptive_ppo_clip_advantages: Per-token advantages ``[B, S]``, whose
            sign selects adaptive-ppo-clip's bound. Read by that mode only.
        adaptive_ppo_clip_outer_bound: adaptive-ppo-clip's upper bound for
            either advantage sign. Read by that mode only.

    Returns:
        The per-token term ``[B, S]`` and the sampler mass the head covers ``[B, S]``.
    """
    active, p_log, q_log, p, q = _sampler_head(
        policy_logprobs_at_sampler_topk,
        sampler_topk_logprobs=sampler_topk_logprobs,
        token_mask=token_mask,
    )
    head_mass = q.sum(-1)
    if sampler_replay:
        rho = torch.zeros_like(head_mass)
    else:
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


class ScoreCenteringDiagnostics(NamedTuple):
    """Per-token sampler-trainer mismatch on the captured head, ``[B, S]`` each.

    Both fields are zero at inactive positions.

    Attributes:
        policy_head_mass: The trainer's mass on the sampler's head, ``sum_H p``.
            With the sampler's head mass ``sum_H q`` it gives the tail ratio
            ``rho``.
        tv_distance: Total variation distance between the sampler, its tail
            modeled as ``rho * p``, and the trainer:
            ``(sum_H |q - p| + |t_q - t_p|) / 2`` with ``t = max(1 - sum_H, 0)``.
            It lies in ``[0, 1]`` and is zero when ``p == q``.
    """

    policy_head_mass: torch.Tensor
    tv_distance: torch.Tensor


@torch.no_grad()
def compute_score_centering_diagnostics(
    policy_logprobs_at_sampler_topk: torch.Tensor,
    *,
    sampler_topk_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    sampler_replay: bool,
) -> ScoreCenteringDiagnostics:
    """Measure the drift ``compute_score_centering_term`` removes, on the same head.

    With importance sampling off, the gradient of the centering term with
    respect to the logits is ``q_hat - p``, where ``q_hat`` is the sampler with
    its tail modeled as ``rho * p``; ``tv_distance`` is half its L1 norm. It
    measures the mismatch itself, so it does not depend on the
    importance-sampling mode. Under sampler replay the head holds the support
    of both distributions, so neither has a tail.

    Args:
        policy_logprobs_at_sampler_topk: Trainer log-probs at the sampler's
            top-k ids, ``[B, S, k]``.
        sampler_topk_logprobs: The sampler's log-probs at those ids,
            ``[B, S, k]``; ``-inf`` where its filters removed the id.
        token_mask: Active token/sample positions, ``[B, S]``.
        sampler_replay: Whether the trainer log-probs are renormalized over the
            sampler's support.

    Returns:
        The per-token diagnostics.
    """
    head = _sampler_head(
        policy_logprobs_at_sampler_topk,
        sampler_topk_logprobs=sampler_topk_logprobs,
        token_mask=token_mask,
    )
    policy_head_mass = head.p.sum(-1)
    head_distance = (head.q - head.p).abs().sum(-1)
    if sampler_replay:
        tail_distance = torch.zeros_like(head_distance)
    else:
        sampler_tail = (1.0 - head.q.sum(-1)).clamp(min=0.0)
        policy_tail = (1.0 - policy_head_mass).clamp(min=0.0)
        tail_distance = (sampler_tail - policy_tail).abs()
    return ScoreCenteringDiagnostics(
        policy_head_mass=torch.where(head.active, policy_head_mass, 0.0),
        tv_distance=torch.where(
            head.active, 0.5 * (head_distance + tail_distance), 0.0
        ),
    )
