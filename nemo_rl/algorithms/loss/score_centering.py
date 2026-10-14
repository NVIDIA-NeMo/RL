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
"""Score centering (Marek and Ryabinin, arXiv:2609.20807) for the clipped policy gradient.

Enabled by ``loss_fn.score_centering`` with the sampler's top-k captured per
token (``token_capture.top_logprobs``); ``ClippedPGLossFn`` adds
``advantages * score_centering_term(...)`` to its per-token loss.
"""

from typing import Optional

import torch


def score_centering_term(
    topk_logprobs: torch.Tensor,
    sampler_topk_logprobs: torch.Tensor,
    is_bounds: Optional[tuple[float, float]],
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

    Args:
        topk_logprobs: Trainer log-probs at the sampler's top-k ids, ``[B, S, k]``,
            carrying the gradient.
        sampler_topk_logprobs: The sampler's log-probs at those ids, ``[B, S, k]``.
        is_bounds: ``(lo, hi)`` of the TIS clamp, or ``None`` when IS is off.

    Returns:
        The per-token term ``[B, S]`` and the sampler mass the head covers ``[B, S]``.
    """
    p = topk_logprobs.detach().exp()
    q = sampler_topk_logprobs.exp()
    head_mass = q.sum(-1)
    rho = (1.0 - head_mass).clamp(min=0.0) / (1.0 - p.sum(-1)).clamp(min=1e-6)
    if is_bounds is None:
        mass, alpha = q, rho
    else:
        lo, hi = is_bounds
        mass = torch.minimum(torch.maximum(p, lo * q), hi * q)
        alpha = torch.minimum(torch.maximum(torch.ones_like(rho), lo * rho), hi * rho)
    coefficients = mass - alpha.unsqueeze(-1) * p
    return (coefficients * topk_logprobs).sum(-1), head_mass
