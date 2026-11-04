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

from typing import Any, Callable

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


class OmegaLossConfig(BaseModel, extra="allow"):
    # Name of the exponent function h(x, y); one of ``H_FUNCTIONS``.
    h: str = "inverse_length"


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
            }
        return loss, metrics


def validate_omega_setup(master_config: Any) -> None:
    """Fail fast if batching could split a prompt's k responses across microbatches."""
    policy = master_config.policy
    k = master_config.grpo.num_generations_per_prompt
    assert not policy["sequence_packing"]["enabled"], (
        "Omega loss requires policy.sequence_packing.enabled=false (rows get reordered)"
    )
    assert not policy["dynamic_batching"]["enabled"], (
        "Omega loss requires policy.dynamic_batching.enabled=false (rows get reordered)"
    )
    assert policy["train_micro_batch_size"] % k == 0, (
        f"policy.train_micro_batch_size ({policy['train_micro_batch_size']}) must be "
        f"a multiple of grpo.num_generations_per_prompt ({k})"
    )
    assert policy["train_global_batch_size"] % k == 0, (
        "policy.train_global_batch_size must be a multiple of "
        "grpo.num_generations_per_prompt"
    )
    assert not master_config.grpo.normalize_rewards, (
        "Omega loss needs advantages = r - mean(r): set grpo.normalize_rewards=false"
    )
    assert not master_config.grpo.use_leave_one_out_baseline, (
        "Omega loss needs b_i = mean(r): set grpo.use_leave_one_out_baseline=false"
    )
