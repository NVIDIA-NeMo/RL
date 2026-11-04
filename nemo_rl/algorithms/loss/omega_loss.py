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

"""Omega loss: a GRPO-like objective with softmax weights over each prompt group.

For a batch of B prompts x_i with k responses y^1_i..y^k_i and rewards r^j_i::

    L = -1/B sum_i 1/k sum_j w^j_i (r^j_i - b_i)
    w^j_i = pi(y^j_i|x_i)^h(x_i,y^j_i) / sum_l pi(y^l_i|x_i)^h(x_i,y^l_i)
    b_i   = 1/k sum_j r^j_i

Unlike GRPO there is no pi_old / ratio / clipping: every pi is the *current*
policy and the gradient flows through the weights w. Only the baseline
advantage (r - b) is treated as a constant.

Implementation notes:
  * ``w`` is a softmax over the group of ``h * log pi(y)``, computed in fp32.
  * ``(r - b)`` is read from ``data["advantages"]`` (the GRPO advantage
    estimator with ``normalize_rewards=false`` and
    ``use_leave_one_out_baseline=false`` produces exactly ``r - mean(r)``).
  * Groups are the ``k`` consecutive rows of the microbatch. This requires
    that each microbatch holds whole groups (no sequence packing / dynamic
    batching, micro batch size a multiple of ``k``); ``validate_omega_setup``
    checks this at setup time.
"""

from typing import Any, Callable, Literal

import torch
from pydantic import BaseModel

from nemo_rl.algorithms.loss.interfaces import (
    LossFunction,
    LossInputType,
    LossType,
    MetricNormalizer,
)
from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# h(x, y): per-sequence exponent applied to the sequence probability.
# Signature: (response_token_mask [N, T-1], data) -> exponent [N] (fp32).
HFunction = Callable[[torch.Tensor, BatchedDataDict[Any]], torch.Tensor]


def _h_inverse_length(
    response_token_mask: torch.Tensor, data: BatchedDataDict[Any]
) -> torch.Tensor:
    """h(x, y) = 1 / |y|, with |y| the number of model-generated tokens."""
    return 1.0 / response_token_mask.sum(dim=-1).clamp(min=1)


# Register new choices of h here and select them with ``omega_loss.h``.
H_FUNCTIONS: dict[str, HFunction] = {
    "inverse_length": _h_inverse_length,
}


def _token_mult_prob_error(
    data: BatchedDataDict[Any],
    sample_mask: torch.Tensor,
    global_valid_toks: torch.Tensor | None,
) -> float:
    """vLLM-vs-training logprob mismatch, computed exactly as ClippedPGLossFn does.

    mean over valid response tokens of exp(|generation_logprobs - prev_logprobs|).
    The GRPO loop reads ``metrics["token_mult_prob_error"]`` after every step
    (and plots samples above 1.05), so every policy loss must report it; it
    also keeps this diagnostic comparable between GRPO and Omega runs.
    """
    token_mask = data["token_mask"][:, 1:].to(torch.float32)
    mask = token_mask * sample_mask.to(torch.float32).unsqueeze(-1)
    gen = data["generation_logprobs"][:, 1:].to(torch.float32)
    prev = data["prev_logprobs"][:, 1:].to(torch.float32)
    gen = torch.where(mask.bool(), gen, 0.0)
    prev = torch.where(mask.bool(), prev, 0.0)
    lp_error = torch.abs(gen - prev)
    return masked_mean(
        torch.exp(lp_error * mask),
        mask,
        global_normalization_factor=global_valid_toks,
    ).item()


class OmegaLossConfig(BaseModel, extra="allow"):
    # Name of the exponent function h(x, y); one of ``H_FUNCTIONS``.
    h: str = "inverse_length"
    # "group": the loss computes the softmax itself; each microbatch must hold
    #   whole groups (no sequence packing); any number of updates per rollout.
    # "per_sequence": group weights are computed once per rollout from the
    #   current policy's logprobs (``compute_omega_coefficients``) and the loss
    #   is a per-sequence term, so sequence packing / CP work. Exact only with
    #   ONE optimizer update per rollout (train_global_batch_size ==
    #   num_prompts_per_step * num_generations_per_prompt).
    form: Literal["group", "per_sequence"] = "group"


class OmegaLossFn(LossFunction):
    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(
        self,
        cfg: OmegaLossConfig,
        num_generations_per_prompt: int,
        use_fused_linear_logprobs: bool = False,
    ):
        if cfg.h not in H_FUNCTIONS:
            raise ValueError(
                f"Unknown omega_loss.h '{cfg.h}'. Options: {sorted(H_FUNCTIONS)}"
            )
        if num_generations_per_prompt < 2:
            raise ValueError("Omega loss needs num_generations_per_prompt >= 2")
        self.h_fn = H_FUNCTIONS[cfg.h]
        self.k = num_generations_per_prompt
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        self.metric_normalizations = {
            "loss": MetricNormalizer.SEQUENCES,
            "weight_concentration": MetricNormalizer.SEQUENCES,
            "response_len_mean": MetricNormalizer.SEQUENCES,
            "h_mean": MetricNormalizer.SEQUENCES,
            "num_empty_responses": MetricNormalizer.NONE,
            "token_mult_prob_error": MetricNormalizer.TOKENS,
        }

    def __call__(
        self,
        next_token_logprobs: torch.Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        k = self.k
        n = next_token_logprobs.shape[0]
        assert n % k == 0, (
            f"Omega loss microbatch has {n} rows, not a multiple of "
            f"num_generations_per_prompt={k}; groups would be split."
        )

        token_mask = data["token_mask"][:, 1:].to(torch.float32)  # [N, T-1]
        # A response with no model-generated tokens (|y| = 0) has log pi(y) = 0,
        # the *maximum* possible score, so left alone it would dominate the
        # softmax. Treat it like an invalid sample: no weight, no gradient.
        has_tokens = token_mask.sum(-1) > 0
        sample_mask = data["sample_mask"].to(torch.float32) * has_tokens  # [N]
        num_empty = ((data["sample_mask"] > 0) & ~has_tokens).sum()
        logp = next_token_logprobs.to(torch.float32)

        # log pi(y|x) of the whole response under the *current* policy (has grad).
        seq_logp = (logp * token_mask).sum(dim=-1)  # [N]
        h = self.h_fn(token_mask, data).to(torch.float32)  # [N]

        # w = softmax_j(h * log pi(y^j)) within each group; invalid rows excluded.
        scores = (h * seq_logp).view(-1, k)
        valid = sample_mask.view(-1, k) > 0
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=-1) * valid  # all-invalid group -> all 0
        w = w.reshape(-1)  # [N]

        # r - b, constant w.r.t. theta. Advantages are broadcast over tokens.
        adv_tok = data["advantages"][:, 1:].to(torch.float32)
        adv = (
            (adv_tok * token_mask).sum(-1) / token_mask.sum(-1).clamp(min=1)
        ).detach()

        # Sum_i Sum_j w*A / (B*k) with B*k = number of valid sequences.
        loss = -masked_mean(
            w * adv,
            sample_mask,
            global_normalization_factor=global_valid_seqs,
        )

        with torch.no_grad():
            w_d = w.detach()
            # sum_j w^2 per group (broadcast to its rows): 1/k when uniform,
            # 1 when all weight sits on one response.
            concentration = (w_d.view(-1, k) ** 2).sum(-1, keepdim=True).expand(-1, k)
            concentration = concentration.reshape(-1)
            metrics = {
                "loss": loss.item(),
                "weight_concentration": masked_mean(
                    concentration,
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                ).item(),
                "response_len_mean": masked_mean(
                    token_mask.sum(-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                ).item(),
                # Valid-by-mask rollouts dropped for having no generated tokens.
                "num_empty_responses": num_empty.item(),
                "h_mean": masked_mean(
                    h, sample_mask, global_normalization_factor=global_valid_seqs
                ).item(),
                "token_mult_prob_error": _token_mult_prob_error(
                    data, data["sample_mask"], global_valid_toks
                ),
            }
        return loss, metrics


def compute_omega_coefficients(
    prompt_ids: torch.Tensor,
    prev_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    advantages: torch.Tensor,
    h_fn: HFunction,
    data: BatchedDataDict[Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-sequence coefficients that make the Omega gradient decomposable.

    With s_j = h_j log pi(y_j), w = softmax_j(s) per prompt group and
    L = -sum_j w_j A_j, the exact gradient is::

        dL/dtheta = -sum_j  w_j (A_j - sum_l w_l A_l) * h_j * grad log pi(y_j)

    so ``coef_j = w_j (A_j - Abar_w) h_j`` (no grad) turns the loss into the
    per-sequence term ``-coef_j * log pi(y_j)`` that packing handles like any
    other sequence-level loss. The logprobs must come from the weights that
    are about to be differentiated (one update per rollout).

    Groups are found by prompt content (like GRPO's baseline), not by row
    position. Invalid rows (sample_mask == 0) and rows with no generated
    tokens get coefficient 0 and are excluded from the softmax.

    Args:
        prompt_ids: [N, Lp] prompt token ids identifying each row's group.
        prev_logprobs: [N, T] current-policy logprobs (GRPO's ``prev_logprobs``).
        token_mask: [N, T] model-generated-token mask.
        sample_mask: [N] 1 for valid samples.
        advantages: [N, T] token-broadcast (r - mean r).

    Returns:
        coef: [N] float32, and scalar metrics (to be logged by the caller).
    """
    device = prev_logprobs.device
    tm = token_mask[:, 1:].to(device=device, dtype=torch.float32)
    lp = prev_logprobs[:, 1:].to(torch.float32)
    n_tok = tm.sum(-1)
    has_tokens = n_tok > 0
    valid = (sample_mask.to(device) > 0) & has_tokens

    seq_logp = (lp * tm).sum(-1)
    h = h_fn(tm, data).to(torch.float32)
    scores = h * seq_logp

    group = torch.unique(prompt_ids.to(device), dim=0, return_inverse=True)[1]
    num_groups = int(group.max().item()) + 1

    neg_inf = torch.full_like(scores, float("-inf"))
    masked_scores = torch.where(valid, scores, neg_inf)
    gmax = torch.full((num_groups,), float("-inf"), device=device)
    gmax = gmax.scatter_reduce(0, group, masked_scores, reduce="amax")
    gmax = torch.where(torch.isfinite(gmax), gmax, torch.zeros_like(gmax))
    e = torch.where(valid, torch.exp(scores - gmax[group]), torch.zeros_like(scores))
    z = torch.zeros(num_groups, device=device).index_add_(0, group, e)
    w = e / z[group].clamp(min=torch.finfo(torch.float32).tiny)

    adv = (advantages[:, 1:].to(device=device, dtype=torch.float32) * tm).sum(
        -1
    ) / n_tok.clamp(min=1)
    abar = torch.zeros(num_groups, device=device).index_add_(0, group, w * adv)
    coef = torch.where(valid, w * (adv - abar[group]) * h, torch.zeros_like(w))

    n_valid = valid.sum().clamp(min=1)
    w_sq = torch.zeros(num_groups, device=device).index_add_(0, group, w * w)
    group_valid = torch.zeros(num_groups, device=device).index_add_(
        0, group, valid.float()
    )
    has_valid_group = group_valid > 0
    metrics = {
        # Value of the Omega objective, -1/N sum_j w_j A_j (same scale as the "group" form's loss).
        "objective": float((-(w * adv).sum() / n_valid).item()),
        # Mean over groups of sum_j w_j^2: 1/k if uniform, 1 if collapsed.
        "weight_concentration": float(w_sq[has_valid_group].mean().item())
        if bool(has_valid_group.any())
        else 0.0,
        "num_empty_responses": float(
            ((sample_mask.to(device) > 0) & ~has_tokens).sum().item()
        ),
        "h_mean": float(h[valid].mean().item()) if bool(valid.any()) else 0.0,
    }
    return coef.detach(), metrics


class OmegaPerSequenceLossFn(LossFunction):
    """Per-sequence form of the Omega loss; see ``compute_omega_coefficients``.

    Loss = -sum_j data["omega_coef"]_j * log pi(y_j) / global_valid_seqs.
    Each sequence is independent, so sequence packing, dynamic batching, CP and
    the fused packed-loss path all work unchanged.
    """

    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, use_fused_linear_logprobs: bool = False):
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        self.metric_normalizations = {
            "loss": MetricNormalizer.SEQUENCES,
            "seq_logprob_mean": MetricNormalizer.SEQUENCES,
            "token_mult_prob_error": MetricNormalizer.TOKENS,
        }

    def __call__(
        self,
        next_token_logprobs: torch.Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        token_mask = data["token_mask"][:, 1:].to(torch.float32)
        seq_logp = (next_token_logprobs.to(torch.float32) * token_mask).sum(-1)  # [N]
        coef = data["omega_coef"].to(device=seq_logp.device, dtype=torch.float32)
        sample_mask = data["sample_mask"].to(torch.float32)

        loss = -masked_mean(
            coef * seq_logp,
            sample_mask,
            global_normalization_factor=global_valid_seqs,
        )
        metrics = {
            "loss": loss.item(),
            "seq_logprob_mean": masked_mean(
                seq_logp.detach(),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            ).item(),
            "token_mult_prob_error": _token_mult_prob_error(
                data, sample_mask, global_valid_toks
            ),
        }
        return loss, metrics


def validate_omega_setup(master_config: Any) -> None:
    """Fail fast on configs the selected Omega form cannot handle."""
    policy = master_config.policy
    grpo = master_config.grpo
    k = grpo.num_generations_per_prompt
    form = grpo.omega_loss.form

    # Common: advantages must be exactly r - mean(r), with no post-processing.
    assert not grpo.normalize_rewards, (
        "Omega loss needs advantages = r - mean(r): set grpo.normalize_rewards=false"
    )
    assert not grpo.use_leave_one_out_baseline, (
        "Omega loss needs b_i = mean(r): set grpo.use_leave_one_out_baseline=false"
    )
    assert grpo.advantage_clip_low is None and grpo.advantage_clip_high is None, (
        "Omega loss does not support grpo.advantage_clip_low/high"
    )
    assert grpo.async_grpo is None or not grpo.async_grpo.enabled, (
        "Omega loss is only wired into the synchronous grpo_train loop "
        "(set grpo.async_grpo.enabled=false)"
    )
    data_plane = getattr(master_config, "data_plane", None) or {}
    assert not data_plane.get("enabled"), (
        "Omega loss is not supported with data_plane.enabled=true (the data plane "
        "only ships a fixed set of train fields, so omega_coef would be dropped)"
    )
    generation = policy["generation"]
    assert generation["top_k"] is None and generation["top_p"] == 1.0, (
        "Omega loss does not support top-k/top-p training-time filtering "
        "(logprobs could be -inf): set policy.generation.top_k=null, top_p=1.0"
    )
    assert policy["train_global_batch_size"] % k == 0, (
        "policy.train_global_batch_size must be a multiple of "
        "grpo.num_generations_per_prompt"
    )

    if form == "group":
        assert not policy["sequence_packing"]["enabled"], (
            "omega_loss.form=group requires policy.sequence_packing.enabled=false "
            "(rows get reordered); use omega_loss.form=per_sequence for packing"
        )
        assert not policy["dynamic_batching"]["enabled"], (
            "omega_loss.form=group requires policy.dynamic_batching.enabled=false "
            "(rows get reordered); use omega_loss.form=per_sequence"
        )
        assert policy["train_micro_batch_size"] % k == 0, (
            f"policy.train_micro_batch_size ({policy['train_micro_batch_size']}) "
            f"must be a multiple of grpo.num_generations_per_prompt ({k})"
        )
    else:
        # w is computed once per rollout from prev_logprobs, so it is only the
        # exact current-policy weight for the first (and only) optimizer update.
        assert policy["train_global_batch_size"] == grpo.num_prompts_per_step * k, (
            "omega_loss.form=per_sequence needs exactly one optimizer update per "
            f"rollout: set policy.train_global_batch_size={grpo.num_prompts_per_step * k} "
            "(= num_prompts_per_step * num_generations_per_prompt)"
        )
        assert not master_config.loss_fn.force_on_policy_ratio, (
            "omega_loss.form=per_sequence needs real prev_logprobs: set "
            "loss_fn.force_on_policy_ratio=false"
        )
        assert not grpo.use_dynamic_sampling, (
            "omega_loss.form=per_sequence does not support dynamic sampling "
            "(the batch must stay num_prompts_per_step * num_generations_per_prompt)"
        )
