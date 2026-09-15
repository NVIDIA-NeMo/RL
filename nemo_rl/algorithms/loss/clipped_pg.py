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

from typing import Optional

import torch

from nemo_rl.algorithms.utils import calculate_kl, masked_mean


def materialize_scalar_metrics(
    metrics: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, float]:
    """Copy scalar metrics to the host in one synchronization."""
    values: list[float] = (
        torch.stack(
            tuple(
                value.detach().to(dtype=dtype).reshape(()) for value in metrics.values()
            )
        )
        .cpu()
        .tolist()
    )
    return dict(zip(metrics, values, strict=True))


@torch.no_grad()
def clipped_pg_diagnostic_metrics(
    *,
    curr_logprobs: torch.Tensor,
    prev_logprobs: torch.Tensor,
    generation_logprobs: torch.Tensor,
    ratios: torch.Tensor,
    ratios_clamped: torch.Tensor,
    mask: torch.Tensor,
    global_valid_toks: torch.Tensor,
    reference_policy_kl_type: str,
) -> dict[str, torch.Tensor]:
    """Compute metrics that do not contribute to the actor objective."""
    lp_error = torch.abs(generation_logprobs - prev_logprobs)
    mult_prob_error = masked_mean(
        torch.exp(lp_error * mask),
        mask,
        global_normalization_factor=global_valid_toks,
    )
    gen_kl_error = masked_mean(
        calculate_kl(
            logprobs=generation_logprobs,
            logprobs_reference=prev_logprobs,
            kl_type=reference_policy_kl_type,
            input_clamp_value=None,
            output_clamp_value=None,
        ),
        mask,
        global_normalization_factor=global_valid_toks,
    )
    policy_kl_error = masked_mean(
        calculate_kl(
            logprobs=prev_logprobs,
            logprobs_reference=generation_logprobs,
            kl_type=reference_policy_kl_type,
            input_clamp_value=None,
            output_clamp_value=None,
        ),
        mask,
        global_normalization_factor=global_valid_toks,
    )

    # Jensen-Shannon divergence via KL(P_train || M) and KL(P_gen || M).
    log_mixture = torch.log(
        0.5 * torch.exp(prev_logprobs) + 0.5 * torch.exp(generation_logprobs)
    )
    prev_to_mixture = prev_logprobs - log_mixture
    gen_to_mixture = generation_logprobs - log_mixture
    kl_prev_to_mixture = torch.exp(prev_to_mixture) - prev_to_mixture - 1
    kl_gen_to_mixture = torch.exp(gen_to_mixture) - gen_to_mixture - 1
    js_divergence_error = masked_mean(
        0.5 * kl_prev_to_mixture + 0.5 * kl_gen_to_mixture,
        mask,
        global_normalization_factor=global_valid_toks,
    )
    seq_entropy_approx = -masked_mean(
        torch.exp(curr_logprobs - generation_logprobs) * curr_logprobs,
        mask,
        global_normalization_factor=global_valid_toks,
    )

    valid_mask = mask.bool()
    inf = torch.full_like(ratios, float("inf"))
    neg_inf = torch.full_like(ratios, float("-inf"))
    return {
        "probs_ratio": masked_mean(
            ratios, mask, global_normalization_factor=global_valid_toks
        ),
        "probs_ratio_clamped": masked_mean(
            ratios_clamped, mask, global_normalization_factor=global_valid_toks
        ),
        "probs_ratio_min": torch.where(valid_mask, ratios, inf).min(),
        "probs_ratio_max": torch.where(valid_mask, ratios, neg_inf).max(),
        "probs_ratio_clamped_min": torch.where(valid_mask, ratios_clamped, inf).min(),
        "probs_ratio_clamped_max": torch.where(
            valid_mask, ratios_clamped, neg_inf
        ).max(),
        "token_mult_prob_error": mult_prob_error,
        "gen_kl_error": gen_kl_error,
        "policy_kl_error": policy_kl_error,
        "js_divergence_error": js_divergence_error,
        "approx_entropy": seq_entropy_approx,
    }


def clipped_pg_actor_objective(
    curr_logprobs: torch.Tensor,
    *,
    prev_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    mask: torch.Tensor,
    sample_mask: torch.Tensor,
    importance_weights: Optional[torch.Tensor],
    global_valid_toks: torch.Tensor,
    global_valid_seqs: torch.Tensor,
    ratio_clip_min: float,
    ratio_clip_max: float,
    ratio_clip_c: float,
    disable_ppo_ratio: bool,
    force_on_policy_ratio: bool,
    sequence_level_importance_ratios: bool,
    use_cispo: bool,
    token_level_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute and reduce the actor objective in one compilable region."""
    if force_on_policy_ratio:
        # Ratio is one in the forward pass, but gradients still flow.
        ratios = (curr_logprobs - curr_logprobs.detach()).exp()
        ratios_clamped = ratios
    elif not disable_ppo_ratio:
        log_ratios = curr_logprobs - prev_logprobs
        if sequence_level_importance_ratios:
            seq_ratio = masked_mean(log_ratios, token_mask, dim=-1).unsqueeze(-1).exp()
            ratios = seq_ratio.expand(-1, advantages.shape[1])
        else:
            ratios = log_ratios.exp()
        ratios_clamped = ratios.clamp(1.0 - ratio_clip_min, 1.0 + ratio_clip_max)
    else:
        ratios = curr_logprobs
        ratios_clamped = curr_logprobs

    if use_cispo:
        clip_loss = -advantages * ratios_clamped.detach() * curr_logprobs
    else:
        clip_loss = torch.maximum(-advantages * ratios, -advantages * ratios_clamped)

    # Dual-clipping; see https://arxiv.org/pdf/1912.09729.
    if ratio_clip_c > 0:
        clip_loss = torch.where(
            advantages < 0,
            torch.minimum(clip_loss, -advantages * ratio_clip_c),
            clip_loss,
        )

    actor_values = (
        clip_loss if importance_weights is None else importance_weights * clip_loss
    )
    if token_level_loss:
        actor_loss = masked_mean(
            actor_values, mask, global_normalization_factor=global_valid_toks
        )
    else:
        actor_loss = masked_mean(
            masked_mean(actor_values, token_mask, dim=-1),
            sample_mask,
            global_normalization_factor=global_valid_seqs,
        )
    return actor_loss, ratios, ratios_clamped
