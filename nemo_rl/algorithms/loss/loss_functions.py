# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import os
from typing import Any, NotRequired, Optional, TypedDict, TypeVar

import numpy as np
import torch
from pydantic import BaseModel

from nemo_rl.algorithms.loss.interfaces import (
    LossFunction,
    LossInputType,
    LossType,
    MetricNormalizer,
)
from nemo_rl.algorithms.utils import calculate_kl, masked_mean
from nemo_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    build_exact_token_map,
    ce_label_mask,
    next_token_accuracy,
    select_teacher_topk_indices,
    student_next_token_ce,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    DistributedCrossEntropy,
    allgather_cp_contiguous_tensor,
    cp_shift_next,
    group_all_reduce_sum,
    vocab_parallel_full_log_softmax,
    vocab_parallel_gather_logits,
)
from nemo_rl.models.dtensor.parallelize import to_local_if_dtensor

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class DraftCrossEntropyLossConfig(TypedDict):
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup]


class DraftCrossEntropyLossDataDict(TypedDict):
    teacher_logits: Tensor
    student_logits: Tensor
    token_mask: Tensor
    sample_mask: Tensor
    student_vocab_indices: NotRequired[Tensor]


class DraftCrossEntropyLossFn(LossFunction):
    """Compute the auxiliary soft-target cross-entropy used for draft-model training."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DRAFT

    def __init__(
        self,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.vocab_parallel_group = vocab_parallel_group

    def __call__(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        token_mask: Tensor,
        data: BatchedDataDict[DraftCrossEntropyLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the masked per-token draft loss to a scalar."""
        if self.vocab_parallel_group is not None:
            # Soft cross entropy matches the forward-KL student gradient.
            per_token_loss = DistributedCrossEntropy.apply(
                student_logits,
                teacher_logits,
                self.vocab_parallel_group,
                False,
            )
        else:
            teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
            student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
            per_token_loss = -(teacher_probs * student_log_probs).sum(dim=-1)

        mask = token_mask * data["sample_mask"].unsqueeze(-1)
        return masked_mean(
            per_token_loss,
            mask,
            global_normalization_factor=global_valid_toks,
        )


class ClippedPGLossConfig(BaseModel, extra="allow"):
    # --- Loss type ---
    disable_ppo_ratio: bool = False
    token_level_loss: bool = True
    # If True, apply the off-policy importance-sampling correction at the
    # sequence level (one weight per generated sample), as in GSPO.
    # If False (default), correction is applied at the token level as in the
    # original GRPO paper.
    sequence_level_importance_ratios: bool = False

    # --- Clipping ---
    ratio_clip_min: float = 0.2
    ratio_clip_max: float = 0.2
    # Dual-clipping value (should be >1 if enabled; usually set to 3 empirically). None to disable.
    ratio_clip_c: Optional[float] = None

    # --- KL regularization ---
    reference_policy_kl_penalty: float = 0.01
    # Can be set to k1, k2, k3
    # For more details, see http://joschu.net/blog/kl-approx.html
    reference_policy_kl_type: str = "k3"
    kl_input_clamp_value: Optional[float] = 20.0
    kl_output_clamp_value: Optional[float] = 10.0
    # If True, add KL penalty to reward instead of loss (used by Reinforce++)
    use_kl_in_reward: bool = False

    # --- Importance sampling correction ---
    # Async GRPO requires importance sampling correction enabled
    # Set to true when async_grpo.enabled is true
    use_importance_sampling_correction: bool = False
    # --- Truncated importance sampling ---
    # Type of truncated importance sampling:
    #   "tis"          – clamp IS weights to [min, max], where min defaults to 0
    #   "icepop"       – zero out tokens with IS weight outside [min, max]
    #   "seq-mask-tis" – zero out sequences by geometric-mean IS ratio, non-truncated token IS correction
    truncated_importance_sampling_type: Optional[str] = None
    truncated_importance_sampling_ratio: Optional[float] = None
    # Lower bound for TIS clipping, ICE-POP filtering, or seq-mask-tis filtering
    truncated_importance_sampling_ratio_min: Optional[float] = None

    # --- On-policy ---
    # (default off) loss formulation improvements (docs/guides/grpo.md#loss)
    use_on_policy_kl_approximation: bool = False
    # If True, force the ratio to 1.0 for truly on-policy behavior,
    # eliminating any importance sampling effects.
    # NOTE: This should only be used when doing exactly one update per rollout
    # (i.e., num_prompts_per_step * num_generations_per_prompt == train_global_batch_size)
    force_on_policy_ratio: bool = False
    # If True, use CISPO (Clipped IS-weight Policy Optimization) from MiniMax-M1.
    use_cispo: bool = False
    # VAPO: weight μ for positive-example NLL loss on correct samples.
    # L = L_PPO + μ·L_NLL(correct)   (arXiv:2504.05118, Eq. 10)
    # Set to 0 to disable.
    positive_example_nll_weight: float = 0.0


class ClippedPGLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    __extra__: Any


class ClippedPGLossFn(LossFunction):
    """Generalized Clipped Policy Gradient loss function w/ KL regularization.

    This implements:

    - PPO (Clipped) - https://arxiv.org/abs/1707.06347
    - GRPO - https://arxiv.org/abs/2402.03300
    - REINFORCE/RLOO (set disable_ppo_ratio = True and ignores ratio_clip_min/ratio_clip_max) - https://arxiv.org/abs/2402.14740
    - GSPO (set sequence_level_importance_ratios = True and token_level_loss = False) - https://arxiv.org/abs/2507.18071
    - CISPO (set use_cispo = True) - https://arxiv.org/abs/2506.13585
    - Truly on-policy (set force_on_policy_ratio = True to force ratio = 1.0, requires one update per rollout)

    Formula:
    L(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
    - A_t is the advantage estimate
    - ε is the clip parameter (ratio_clip_min/ratio_clip_max)
        - As proposed in the DAPO paper (https://arxiv.org/pdf/2503.14476),
          we allow setting a distinct minimum and maximum value for the clip parameter (set to the same value for PPO/GRPO/etc.)
            - ratio_clip_min: minimum value for the clip parameter
            - ratio_clip_max: maximum value for the clip parameter
    - β is the KL penalty coefficient (reference_policy_kl_penalty)
    - KL(π_θ || π_ref) is the KL divergence between the current policy and reference policy (Schulman Approx.)

    For REINFORCE/RLOO (when disable_ppo_ratio=True), the formula simplifies to:
    L(θ) = E_t [ π_θ(a_t|s_t) * A_t ] - β * KL(π_θ || π_ref)

    Formula (CISPO):
    L(θ) = E_t [ sg(clip(r_t(θ), 1-ε_low, 1+ε_high)) * A_t * log π_θ(a_t|s_t) ]


    Also supports "Dual-Clipping" from https://arxiv.org/pdf/1912.09729, which
    imposes an additional upper bound on the probability ratio when advantages are negative.
    This prevents excessive policy updates. $rA << 0$ -> $cA$(clipped)
    The loss function is modified to the following when A_t < 0:
    L(θ) = E_t [ max(min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t), c * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - c is the dual-clip parameter (ratio_clip_c), which must be greater than 1 and is
      usually set as 3 empirically.

    Due to potential numerical instability, we cast the logits to float32 before computing the loss.
    """

    input_type = LossInputType.LOGPROB

    def __init__(
        self, cfg: ClippedPGLossConfig, use_fused_linear_logprobs: bool = False
    ):
        # When True, the model forward is patched to return precomputed next-token
        # logprobs (via chunked linear CE fusion) instead of full logits. This is
        # consumed by prepare_loss_input, which short-circuits the logits->logprobs
        # conversion. See nemo_rl/distributed/model_utils.py for the fused forward.
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        self.disable_ppo_ratio = cfg.disable_ppo_ratio
        self.ratio_clip_min = cfg.ratio_clip_min
        self.ratio_clip_max = cfg.ratio_clip_max
        self.ratio_clip_c = cfg.ratio_clip_c  # set to None to disable dual-clipping
        self.reference_policy_kl_penalty = (
            cfg.reference_policy_kl_penalty if not cfg.use_kl_in_reward else 0
        )
        self.reference_policy_kl_type = cfg.reference_policy_kl_type
        self.kl_input_clamp_value = cfg.kl_input_clamp_value
        self.kl_output_clamp_value = cfg.kl_output_clamp_value
        self.use_importance_sampling_correction = cfg.use_importance_sampling_correction
        # Type of truncated importance sampling: "tis" | "icepop" | "seq-mask-tis"
        self.truncated_importance_sampling_type = cfg.truncated_importance_sampling_type
        self.truncated_importance_sampling_ratio = (
            cfg.truncated_importance_sampling_ratio
        )
        # Lower bound for TIS clipping, ICE-POP filtering, or seq-mask-tis filtering
        self.truncated_importance_sampling_ratio_min = (
            cfg.truncated_importance_sampling_ratio_min
        )
        self.use_on_policy_kl_approximation = cfg.use_on_policy_kl_approximation
        self.force_on_policy_ratio = cfg.force_on_policy_ratio  # Force ratio to 1.0

        # Whether to compute importance weights per-sequence instead of per-token.
        self.sequence_level_importance_ratios = cfg.sequence_level_importance_ratios
        self.positive_example_nll_weight = cfg.positive_example_nll_weight
        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg.token_level_loss else LossType.SEQUENCE_LEVEL
        )
        if self.sequence_level_importance_ratios:
            assert self.loss_type == LossType.SEQUENCE_LEVEL, (
                "sequence-level importance sampling (e.g. GSPO) is mutually exclusive with token-level loss"
            )

        self.use_cispo = cfg.use_cispo
        if self.use_cispo:
            assert not self.disable_ppo_ratio, (
                "use_cispo is incompatible with disable_ppo_ratio; "
                "CISPO needs the pi_theta/pi_theta_old ratio but disable_ppo_ratio removes it"
            )
            assert not self.force_on_policy_ratio, (
                "use_cispo is incompatible with force_on_policy_ratio; "
                "forcing ratio=1 removes the clipped IS-weight that CISPO optimizes"
            )
            assert not self.sequence_level_importance_ratios, (
                "use_cispo is incompatible with sequence_level_importance_ratios; "
                "CISPO uses token-level importance weights"
            )
            assert self.ratio_clip_c is None, (
                "use_cispo is incompatible with dual clipping (ratio_clip_c); "
                "the dual-clip block runs after the CISPO loss assembly and would "
                "silently overwrite it. Set ratio_clip_c=null when use_cispo=True."
            )
            assert self.loss_type == LossType.TOKEN_LEVEL, (
                "use_cispo requires token_level_loss=True (LossType.TOKEN_LEVEL)."
            )
        if self.truncated_importance_sampling_type is not None:
            assert self.use_importance_sampling_correction, (
                "truncated importance sampling is only supported when use_importance_sampling_correction is True"
            )
            assert self.truncated_importance_sampling_type in (
                "tis",
                "icepop",
                "seq-mask-tis",
            ), (
                f"truncated_importance_sampling_type must be 'tis', 'icepop', or 'seq-mask-tis', "
                f"got {self.truncated_importance_sampling_type}"
            )
            assert (
                self.truncated_importance_sampling_ratio is not None
                and self.truncated_importance_sampling_ratio > 0
            ), "truncated_importance_sampling_ratio should be positive"
            if self.truncated_importance_sampling_ratio_min is not None:
                assert (
                    self.truncated_importance_sampling_ratio_min
                    <= self.truncated_importance_sampling_ratio
                ), (
                    "truncated_importance_sampling_ratio_min must be <= "
                    "truncated_importance_sampling_ratio"
                )
            if self.truncated_importance_sampling_type in ("icepop", "seq-mask-tis"):
                assert self.truncated_importance_sampling_ratio_min is not None, (
                    "truncated_importance_sampling_ratio_min should be set when truncated_importance_sampling_type is 'icepop' or 'seq-mask-tis'"
                )
            if self.truncated_importance_sampling_type == "seq-mask-tis":
                assert not self.sequence_level_importance_ratios, (
                    "seq-mask-tis uses token-level IS correction with sequence-level masking, "
                    "and is incompatible with sequence_level_importance_ratios=True"
                )

        # Advertise, per returned metric, the global denominator it was
        # normalized by (see MetricNormalizer). Built here — next to the flags
        # that pick the denominators — so split-API trainers can undo the
        # placeholder global_valid_*=1 normalization without maintaining a
        # consumer-side table. Keep in sync with __call__'s return dict.
        grad_normalizer = (
            MetricNormalizer.TOKENS
            if self.loss_type == LossType.TOKEN_LEVEL
            else MetricNormalizer.SEQUENCES
        )
        self.metric_normalizations: dict[str, MetricNormalizer] = {
            # Normalized like the gradient (loss_type-dependent).
            "loss": grad_normalizer,
            "kl_penalty": grad_normalizer,
            # Token-normalized diagnostics, independent of loss_type.
            "probs_ratio": MetricNormalizer.TOKENS,
            "probs_ratio_clamped": MetricNormalizer.TOKENS,
            "token_mult_prob_error": MetricNormalizer.TOKENS,
            "gen_kl_error": MetricNormalizer.TOKENS,
            "policy_kl_error": MetricNormalizer.TOKENS,
            "js_divergence_error": MetricNormalizer.TOKENS,
            "approx_entropy": MetricNormalizer.TOKENS,
            # Keyed on sequence_level_importance_ratios, NOT loss_type.
            "sampling_importance_ratio": (
                MetricNormalizer.SEQUENCES
                if self.sequence_level_importance_ratios
                else MetricNormalizer.TOKENS
            ),
            # Raw count — the downstream per-microbatch sum IS the value.
            "num_valid_samples": MetricNormalizer.NONE,
            # Normalized by the microbatch's own correct-token count, not a
            # global factor — already a per-microbatch mean.
            "positive_nll_loss": MetricNormalizer.NONE,
            # Extrema — combined downstream with min/max, never scaled.
            "probs_ratio_min": MetricNormalizer.NONE,
            "probs_ratio_max": MetricNormalizer.NONE,
            "probs_ratio_clamped_min": MetricNormalizer.NONE,
            "probs_ratio_clamped_max": MetricNormalizer.NONE,
        }
        if self.truncated_importance_sampling_type is not None:
            # Keyed on the TIS type, NOT loss_type: seq-mask-tis masks whole
            # sequences (÷ global_valid_seqs); tis/icepop are token-level.
            self.metric_normalizations["is_oob_ratio"] = (
                MetricNormalizer.SEQUENCES
                if self.truncated_importance_sampling_type == "seq-mask-tis"
                else MetricNormalizer.TOKENS
            )

    def __call__(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        curr_logprobs = next_token_logprobs
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"][:, 1:]
        # Skip loading prev_logprobs when force_on_policy_ratio=True (will use curr_logprobs instead)
        prev_logprobs = (
            None if self.force_on_policy_ratio else data["prev_logprobs"][:, 1:]
        )
        generation_logprobs = data["generation_logprobs"][:, 1:]
        if self.reference_policy_kl_penalty != 0:
            reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
            curr_logprobs_unfiltered = data.get(
                "curr_logprobs_unfiltered", curr_logprobs
            )

        mask = token_mask * sample_mask.unsqueeze(-1)

        # For truly on-policy training, use curr_logprobs as prev_logprobs
        # This avoids computing prev_logprobs upstream
        if self.force_on_policy_ratio:
            prev_logprobs = curr_logprobs.detach()

        # token_mult_prob_error
        # See more details and other metrics in docs/guides/grpo.md#metrics
        lp_error = torch.abs(generation_logprobs - prev_logprobs)  # noqa: F841  (precommit ignore for now)
        # average over all tokens in the microbatch
        mult_prob_error = masked_mean(
            torch.exp(lp_error * mask),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # gen-kl: kl(P_gen || P_train)
        # where log_ratio = prev_logprobs - generation_logprobs
        gen_kl_error = calculate_kl(
            logprobs=generation_logprobs,
            logprobs_reference=prev_logprobs,
            kl_type=self.reference_policy_kl_type,
            input_clamp_value=None,
            output_clamp_value=None,
        )
        gen_kl_error = masked_mean(
            gen_kl_error,
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # policy-kl: kl(P_train || P_gen)
        # where log_ratio = generation_logprobs - prev_logprobs
        policy_kl_error = calculate_kl(
            logprobs=prev_logprobs,
            logprobs_reference=generation_logprobs,
            kl_type=self.reference_policy_kl_type,
            input_clamp_value=None,
            output_clamp_value=None,
        )
        policy_kl_error = masked_mean(
            policy_kl_error,
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # Jensen-Shannon divergence
        # M = 0.5 * (P_train + P_gen)
        # JSD = 0.5 * KL(P_train || M) + 0.5 * KL(P_gen || M)
        log_mixture = torch.log(
            0.5 * torch.exp(prev_logprobs) + 0.5 * torch.exp(generation_logprobs)
        )
        # KL(P_train || M)
        kl_prev_to_mixture = (
            torch.exp(prev_logprobs - log_mixture) - (prev_logprobs - log_mixture) - 1
        )

        # KL(P_gen || M)
        kl_gen_to_mixture = (
            torch.exp(generation_logprobs - log_mixture)
            - (generation_logprobs - log_mixture)
            - 1
        )

        js_divergence_error = masked_mean(
            0.5 * kl_prev_to_mixture + 0.5 * kl_gen_to_mixture,
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # Calculate KL regularization.
        if self.reference_policy_kl_penalty != 0:
            # When top-k/top-p filtering is enabled, we need special handling for KL:
            # - reference_policy_logprobs is computed **without** filtering (see use_reference_model)
            # - curr_logprobs/prev_logprobs are computed **with** filtering (for actor loss compatibility)
            # - For KL, we need curr_logprobs **without** filtering to be consistent with ref logprobs
            # - For importance weights, we also use unfiltered curr_logprobs_unfiltered since we're
            #   reweighting samples from π_gen_filtered to π_curr_unfiltered

            # On-policy KL approximation
            # KL samples come from the optimized policy, so the KL loss must include
            # the score-function gradient through the sampling probability; see
            # https://arxiv.org/abs/2506.09477v1. In the non-IS case,
            # exp(x - x.detach()) has forward value 1 while preserving that gradient.
            if self.use_on_policy_kl_approximation:
                # See: docs/guides/grpo.md#on-policy-kl-approximation
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - generation_logprobs
                )
            else:
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - curr_logprobs_unfiltered.detach()
                )
            kl_importance_weights = torch.nan_to_num(
                kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Compute KL loss
            kl = self.reference_policy_kl_penalty * calculate_kl(
                logprobs=curr_logprobs_unfiltered,
                logprobs_reference=reference_policy_logprobs,
                kl_type=self.reference_policy_kl_type,
                input_clamp_value=self.kl_input_clamp_value,
                output_clamp_value=self.kl_output_clamp_value,
                importance_sampling_weights=kl_importance_weights,
            )

            # Reduce KL loss
            if self.loss_type == LossType.TOKEN_LEVEL:
                kl = masked_mean(
                    kl, mask, global_normalization_factor=global_valid_toks
                )
            else:
                kl = masked_mean(
                    masked_mean(kl, token_mask, dim=-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                )
        else:
            kl = torch.tensor(0.0)

        # Calculate clipped loss function if ppo ratio is enabled.
        if self.force_on_policy_ratio:
            # Force ratio to 1.0 for truly on-policy behavior
            # Use curr_logprobs twice so ratio=1 but gradients still flow
            log_ratios = curr_logprobs - curr_logprobs.detach()
            ratios = log_ratios.exp()  # = exp(0) = 1.0, but depends on curr_logprobs
            ratios_clamped = ratios
        elif not self.disable_ppo_ratio:
            log_ratios = curr_logprobs - prev_logprobs
            if self.sequence_level_importance_ratios:
                seq_log_ratio_mean = masked_mean(
                    log_ratios,
                    token_mask,
                    dim=-1,
                ).unsqueeze(-1)
                seq_ratio = seq_log_ratio_mean.exp()
                ratios = seq_ratio.repeat(1, advantages.shape[1])
            else:
                ratios = log_ratios.exp()
            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_clip_min, 1.0 + self.ratio_clip_max
            )
        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        if self.use_cispo:
            clip_loss = -advantages * ratios_clamped.detach() * curr_logprobs
        else:
            loss1 = -advantages * ratios
            loss2 = -advantages * ratios_clamped

            # Determine which value to use for clipping (max for pessimistic estimate)
            clip_loss = torch.max(loss1, loss2)
        # Dual-clipping see https://arxiv.org/pdf/1912.09729
        if self.ratio_clip_c is not None:
            assert self.ratio_clip_c > 1, (
                f"ratio_clip_c must exceed 1 representing a lower bound of the ratios, got {self.ratio_clip_c}."
            )
            loss3 = -advantages * self.ratio_clip_c
            clip_loss = torch.where(
                advantages < 0, torch.min(clip_loss, loss3), clip_loss
            )

        # -------------------------------------------------------------
        # Off-policy (actor) importance-sampling correction
        # -------------------------------------------------------------
        _is_filter_metrics: dict = {}  # populated for icepop / seq-mask-tis
        # See: docs/guides/grpo.md#importance-sampling-correction
        if self.sequence_level_importance_ratios:
            # importance weight w_i = exp(Σ_t (log π_actor − log π_behaviour))
            seq_lp_diff = ((prev_logprobs - generation_logprobs) * mask).sum(dim=-1)
            actor_importance_weights = torch.exp(seq_lp_diff).detach()
            actor_importance_weights = torch.nan_to_num(
                actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Broadcast to token dimension so we can reuse existing reduction
            actor_importance_weights_expanded = actor_importance_weights.unsqueeze(-1)
        else:
            # Token-level correction
            actor_importance_weights_expanded = torch.exp(
                prev_logprobs - generation_logprobs
            )
            actor_importance_weights_expanded = torch.nan_to_num(
                actor_importance_weights_expanded, nan=0.0, posinf=0.0, neginf=0.0
            )
        # ---- Truncated Importance Sampling ----
        # "tis"          – clamp IS weights to [min, max], where min defaults to 0
        # "icepop"       – zero out tokens whose IS weight ∉ [min, max]   (ref bounds: 0.5–5)
        # "seq-mask-tis" – zero out entire sequences whose geometric-mean
        #                  IS ratio ∉ [min, max]; retained sequences keep
        #                  raw (non-truncated) token-level IS weights      (ref bounds: 0.999–1.002)
        #   Blog: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
        # is_oob_ratio: fraction of tokens (tis/icepop) or sequences (seq-mask-tis)
        # whose importance weight falls outside the truncation bounds. Each microbatch
        # contributes its out-of-bounds count divided by the *global* valid token/seq
        # count, so the np.sum aggregation in grpo.py recovers the correct global fraction.
        if self.truncated_importance_sampling_ratio is not None:
            if self.truncated_importance_sampling_type == "tis":
                tis_min = self.truncated_importance_sampling_ratio_min
                if tis_min is None:
                    tis_min = 0.0
                token_oob_mask = (
                    actor_importance_weights_expanded
                    > self.truncated_importance_sampling_ratio
                ) | (actor_importance_weights_expanded < tis_min)
                _is_filter_metrics = {
                    "is_oob_ratio": masked_mean(
                        token_oob_mask.float(),
                        mask,
                        global_normalization_factor=global_valid_toks,
                    ).item(),
                }
                actor_importance_weights_expanded = torch.clamp(
                    actor_importance_weights_expanded,
                    min=tis_min,
                    max=self.truncated_importance_sampling_ratio,
                )
            elif self.truncated_importance_sampling_type == "icepop":
                token_kept_mask = (
                    actor_importance_weights_expanded
                    >= self.truncated_importance_sampling_ratio_min
                ) & (
                    actor_importance_weights_expanded
                    <= self.truncated_importance_sampling_ratio
                )
                _is_filter_metrics = {
                    "is_oob_ratio": masked_mean(
                        (~token_kept_mask).float(),
                        mask,
                        global_normalization_factor=global_valid_toks,
                    ).item(),
                }
                actor_importance_weights_expanded = torch.where(
                    token_kept_mask,
                    actor_importance_weights_expanded,
                    torch.zeros_like(actor_importance_weights_expanded),
                )
            elif self.truncated_importance_sampling_type == "seq-mask-tis":
                # geo_mean_i = exp( mean_t( log(π_prev / π_gen) ) )
                log_is_ratio = torch.nan_to_num(
                    prev_logprobs - generation_logprobs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                seq_log_is_ratio_mean = masked_mean(
                    log_is_ratio, token_mask, dim=-1
                )  # [B]
                seq_geomean_is_ratio = torch.exp(seq_log_is_ratio_mean).detach()  # [B]
                seq_kept_mask = (
                    (
                        seq_geomean_is_ratio
                        >= self.truncated_importance_sampling_ratio_min
                    )
                    & (seq_geomean_is_ratio <= self.truncated_importance_sampling_ratio)
                ).float()  # [B]
                _is_filter_metrics = {
                    "is_oob_ratio": masked_mean(
                        1.0 - seq_kept_mask,
                        sample_mask,
                        global_normalization_factor=global_valid_seqs,
                    ).item(),
                }
                actor_importance_weights_expanded = (
                    actor_importance_weights_expanded * seq_kept_mask.unsqueeze(-1)
                )
            else:
                raise ValueError(
                    f"Invalid truncated importance sampling type: {self.truncated_importance_sampling_type}"
                )

        actor_importance_weights = actor_importance_weights_expanded
        del actor_importance_weights_expanded
        if self.use_importance_sampling_correction:
            importance_weights_to_use = actor_importance_weights
        else:
            importance_weights_to_use = torch.ones_like(prev_logprobs)

        if self.loss_type == LossType.TOKEN_LEVEL:
            actor_loss = masked_mean(
                importance_weights_to_use * clip_loss,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            actor_loss = masked_mean(
                masked_mean(
                    importance_weights_to_use * clip_loss,
                    token_mask,
                    dim=-1,
                ),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )

        # Metric: sampling importance ratio (mean over samples)
        # See: docs/guides/grpo.md#sampling-importance-ratio
        if self.sequence_level_importance_ratios:
            sample_importance_ratio = masked_mean(
                actor_importance_weights,
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )
        else:
            sample_importance_ratio = masked_mean(
                actor_importance_weights,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        # Approximating entropy as E_{s ~ \pi_{gen}(s)}[-(\pi_{curr}/\pi_{gen})log(\pi_{curr}(s))]
        # See more details and other metrics in docs/guides/grpo.md#metrics
        with torch.no_grad():
            seq_entropy_approx = -masked_mean(
                torch.exp(curr_logprobs - generation_logprobs) * curr_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        # -----------------------------------------------------------------
        # VAPO: positive-example NLL loss on correct samples (reward > 0)
        # L = L_PPO + μ · L_NLL(correct)
        # -----------------------------------------------------------------
        nll_loss = torch.tensor(0.0, device=mask.device)
        if self.positive_example_nll_weight > 0 and "rewards" in data:
            correct_sample_mask = (data["rewards"] > 0).float()  # [batch]
            correct_mask = mask * correct_sample_mask.unsqueeze(-1)
            correct_valid_toks = correct_mask.sum()
            if correct_valid_toks > 0:
                nll_loss = masked_mean(
                    -curr_logprobs,
                    correct_mask,
                    global_normalization_factor=correct_valid_toks,
                )

        loss = actor_loss + kl + self.positive_example_nll_weight * nll_loss
        with torch.no_grad():
            probs_ratio = masked_mean(
                ratios.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            probs_ratio_clamped = masked_mean(
                ratios_clamped.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

            # Calculate min/max values for ratios (only for valid tokens)
            masked_ratios = ratios.detach()[mask.bool()]
            masked_ratios_clamped = ratios_clamped.detach()[mask.bool()]

            # Handle edge case where there might be no valid tokens
            if masked_ratios.numel() > 0:
                probs_ratio_min = masked_ratios.min().item()
                probs_ratio_max = masked_ratios.max().item()
                probs_ratio_clamped_min = masked_ratios_clamped.min().item()
                probs_ratio_clamped_max = masked_ratios_clamped.max().item()
            else:
                probs_ratio_min = float("inf")
                probs_ratio_max = float("-inf")
                probs_ratio_clamped_min = float("inf")
                probs_ratio_clamped_max = float("-inf")

        # If you provided a global_valid_{seqs/toks}, all metrics here are globally normalized
        # by either sequence or token count, depending on particular metric.
        # To get the true metric, you'll need to sum over the microbatch.
        return (
            loss,
            {
                "loss": loss.item(),
                "probs_ratio": probs_ratio,
                "probs_ratio_clamped": probs_ratio_clamped,
                "probs_ratio_min": probs_ratio_min,
                "probs_ratio_max": probs_ratio_max,
                "probs_ratio_clamped_min": probs_ratio_clamped_min,
                "probs_ratio_clamped_max": probs_ratio_clamped_max,
                "kl_penalty": kl.item() / self.reference_policy_kl_penalty if kl else 0,
                "token_mult_prob_error": mult_prob_error,
                "gen_kl_error": gen_kl_error,
                "policy_kl_error": policy_kl_error,
                "js_divergence_error": js_divergence_error,
                "sampling_importance_ratio": sample_importance_ratio.item(),
                "num_valid_samples": sample_mask.sum().item(),
                "approx_entropy": seq_entropy_approx.item(),
                **_is_filter_metrics,
                "positive_nll_loss": nll_loss.item(),
            },
        )


class NLLLossFn(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, use_fused_linear_logprobs: bool = False):
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        # See MetricNormalizer — split-API trainers use this to undo the
        # placeholder global_valid_*=1 normalization per metric.
        self.metric_normalizations: dict[str, MetricNormalizer] = {
            "loss": MetricNormalizer.TOKENS,
            "num_unmasked_tokens": MetricNormalizer.NONE,
            "num_valid_samples": MetricNormalizer.NONE,
        }

    def __call__(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor,
        dpo_loss: bool = False,
        dpo_average_log_probs: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        if dpo_loss:
            ## shape: [batch_size]
            num_unmasked_tokens = torch.sum(mask, -1)
            ## multiply by sample_mask to zero out invalid samples
            loss = -torch.sum(next_token_logprobs * mask, dim=-1)
            if dpo_average_log_probs:
                loss = loss / num_unmasked_tokens.clamp(min=1)
        else:
            ## single scalar loss
            ## scale by the total number of tokens in the batch
            loss = -masked_mean(
                next_token_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens": mask.sum().item(),
            "num_valid_samples": sample_mask.sum().item(),
        }


class PreferenceLossDataDict(TypedDict):
    """Required keys for the preference loss function."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class PreferenceLossFn(LossFunction):
    """Preference Loss function.

    Optimizes the model to prefer chosen responses over rejected ones

    The preference loss is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is a scaling factor (ex: `reference_policy_kl_penalty` in DPO)
    - r_chosen and r_rejected are the rewards for chosen and rejected responses

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The preference loss value
            - A dictionary with metrics including:
                - loss: Preference loss
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGIT

    def split_output_tensor(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        # tensor is of shape (2*micro_batch_size,)
        return tensor[::2], tensor[1::2]

    def _preference_loss(
        self,
        rewards: Tensor,
        sample_mask: Tensor,
        global_valid_seqs: Tensor,
        beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(beta * rewards_delta) * sample_mask[::2]
        )  ## zero out invalid samples

        ## divide by 2 because each preference example corresponds to 2 samples (chosen, rejected)
        return (
            masked_mean(
                per_sample_loss,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen > rewards_rejected,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_rejected,
                sample_mask[1::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
        )

    def __call__(
        self,
        logits: Tensor,
        data: BatchedDataDict[PreferenceLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sample_mask = data["sample_mask"]

        rewards = logits.squeeze(-1)

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._preference_loss(rewards, sample_mask, global_valid_seqs)

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = sample_mask.sum() / 2

        return preference_loss, {
            "loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class DPOLossConfig(BaseModel, extra="allow"):
    reference_policy_kl_penalty: float = 0.05
    preference_loss_weight: float = 1.0
    sft_loss_weight: float = 0.0
    preference_average_log_probs: bool = False
    sft_average_log_probs: bool = False


class DPOLossDataDict(TypedDict):
    """Required keys for the DPO loss function."""

    input_ids: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class DPOLossFn(PreferenceLossFn):
    """Direct Preference Optimization (DPO) loss function.

    This loss function implements the DPO algorithm as described in:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)

    The loss combines two main components:
    1. Preference Loss: Optimizes the model to prefer chosen responses over rejected ones
    2. SFT Loss (optional): Auxiliary supervised fine-tuning loss on chosen responses

    The total loss is computed as:
    L(θ) = w_p * L_pref(θ) + w_s * L_sft(θ)

    where:
    - w_p is the preference_loss_weight
    - w_s is the sft_loss_weight
    - L_pref(θ) is the preference loss term
    - L_sft(θ) is the supervised fine-tuning loss term

    The preference loss term is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is the reference_policy_kl_penalty
    - r_chosen and r_rejected are the rewards for chosen and rejected responses
    - The rewards are computed as the sum of log probability differences between
      the current policy and reference policy

    If preference_average_log_probs is True, the rewards are averaged over tokens:
    r = (1/n) * Σ_t (log π_θ(a_t|s_t) - log π_ref(a_t|s_t))

    Otherwise, the rewards are summed over tokens.

    The SFT loss term is a standard negative log likelihood loss on the chosen responses.
    If sft_average_log_probs is True, the loss is averaged over tokens.

    Args:
        cfg (DPOLossConfig): Configuration dictionary containing:
            - reference_policy_kl_penalty (float): Strength of the KL penalty term (β)
            - preference_loss_weight (float): Weight for the preference loss term (w_p)
            - sft_loss_weight (float): Weight for the SFT loss term (w_s)
            - preference_average_log_probs (bool): Whether to average log probs across tokens in preference loss
            - sft_average_log_probs (bool): Whether to average log probs across tokens in SFT loss

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The total loss value
            - A dictionary with metrics including:
                - loss: Total loss value
                - sft_loss: SFT loss component
                - preference_loss: Preference loss component
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: DPOLossConfig, use_fused_linear_logprobs: bool = False):
        self.reference_policy_kl_penalty = cfg.reference_policy_kl_penalty
        self.preference_loss_weight = cfg.preference_loss_weight
        self.sft_loss_weight = cfg.sft_loss_weight
        self.preference_average_log_probs = cfg.preference_average_log_probs
        self.sft_average_log_probs = cfg.sft_average_log_probs
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        self.sft_loss = NLLLossFn(use_fused_linear_logprobs=use_fused_linear_logprobs)

    def _dpo_loss(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ## TODO(@ashors): there's some duplicate code here with the NLLLossFn function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]

        ref_logprobs = data["reference_policy_logprobs"][:, :-1]
        diff = (next_token_logprobs - ref_logprobs) * token_mask

        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)

        return self._preference_loss(
            rewards, sample_mask, global_valid_seqs, self.reference_policy_kl_penalty
        )

    # TODO a cleaner typing fix would be required (probably that DPOLossFn should not inherit from PreferenceLossFn)
    def __call__(  # type: ignore
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            assert global_valid_toks is not None, (
                "global_valid_toks must be provided for SFT loss"
            )
            sft_loss, _ = self.sft_loss(
                next_token_logprobs,
                data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,  ## unused because sft loss returned is at the sample level
                dpo_loss=True,
                dpo_average_log_probs=self.sft_average_log_probs,
            )
            sft_loss_chosen, sft_loss_rejected = self.split_output_tensor(sft_loss)
            sft_loss_chosen = masked_mean(
                sft_loss_chosen,
                data["sample_mask"][::2],
                global_normalization_factor=global_valid_seqs / 2,
            )

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._dpo_loss(next_token_logprobs, data, global_valid_seqs)

        dpo_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = data["sample_mask"].sum() / 2

        return dpo_loss, {
            "loss": dpo_loss.item(),
            "sft_loss": sft_loss_chosen.item(),
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class DistillationLossConfig(TypedDict):
    kl_type: str
    mixed_kl_weight: float
    zero_outside_topk: bool


class DistillationLossDataDict(TypedDict):
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    teacher_topk_logits: torch.Tensor
    teacher_topk_indices: torch.Tensor


class DistillationLossFn(LossFunction):
    """Distillation loss function."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION

    def __init__(self, cfg: DistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg["mixed_kl_weight"]
        self.zero_outside_topk = cfg["zero_outside_topk"]
        self.log_infinitesimal = -100

        assert self.kl_type in ["forward", "reverse", "mixed"], "Invalid KL type"
        assert self.mixed_kl_weight >= 0 and self.mixed_kl_weight <= 1, (
            "Invalid mixed KL weight"
        )

    def __call__(
        self,
        student_topk_logprobs: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        H_all: torch.Tensor | None,
        data: DistillationLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute distillation loss between teacher and student logits."""
        student_probs = student_topk_logprobs.exp()  # [B, S-1, k]
        teacher_probs = teacher_topk_logprobs.exp()  # [B, S-1, k]

        loss_correction_term = torch.zeros_like(student_probs[..., 0])  # [B, S-1]
        if self.zero_outside_topk and self.kl_type != "forward":
            H_rest = H_all - (student_probs * student_topk_logprobs).sum(-1)
            P_rest = 1 - (student_probs.sum(-1))
            # The entropy and prob of the rest of the tokens [B, S-1]
            loss_correction_term = H_rest - self.log_infinitesimal * P_rest  # [B, S-1]
            if self.kl_type == "mixed":
                loss_correction_term = loss_correction_term * (
                    1.0 - self.mixed_kl_weight
                )

        if self.kl_type == "forward":
            per_token_kl = teacher_probs * (
                teacher_topk_logprobs - student_topk_logprobs
            )
        elif self.kl_type == "reverse":
            per_token_kl = student_probs * (
                student_topk_logprobs - teacher_topk_logprobs
            )
        else:
            # mixed KL
            kl_forward = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
            kl_reverse = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
            per_token_kl = (
                self.mixed_kl_weight * kl_forward
                + (1.0 - self.mixed_kl_weight) * kl_reverse
            )

        per_token_kl = per_token_kl.sum(dim=-1) + loss_correction_term  # [B, S-1]

        # Masking and reduction
        if "token_mask" in data and "sample_mask" in data:
            token_mask = data["token_mask"][:, 1:]
            sample_mask = data["sample_mask"]
            # Align mask length to current per_token_kl
            max_len = per_token_kl.shape[1]
            token_mask = token_mask[:, :max_len]
            mask = token_mask * sample_mask.unsqueeze(-1)  # [B, S-1]
            # align mask shape to per_token_kl
            kl_loss = masked_mean(
                per_token_kl,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            kl_loss = per_token_kl.mean()

        metrics = {
            "loss": float(kl_loss.item()) if kl_loss.ndim == 0 else kl_loss,
            "num_valid_samples": data["input_ids"].shape[0],
        }

        return kl_loss, metrics


class MseValueLossConfig(BaseModel, extra="forbid"):
    """Config for the MSE value loss used by PPO's value model."""

    # Scaling factor applied to the value loss before it is added to the policy loss.
    scale: float = 1.0
    # Clipping range for value predictions (PPO-style). Set to None to disable clipping.
    cliprange: Optional[float] = None


class MseValueLossFn(LossFunction):
    """Mean Squared Error value loss function with optional clipping (PPO-style).

    When ``cliprange`` is set, value predictions are clipped to
    ``[old_values - cliprange, old_values + cliprange]`` and the loss is
    ``0.5 * max(mse(vpred, returns), mse(vpred_clipped, returns))``.
    This prevents the value function from changing too drastically in a
    single update, mirroring the policy ratio clipping in PPO.
    """

    input_type = LossInputType.LOGIT

    def __init__(self, cfg: MseValueLossConfig):
        self.scale = cfg.scale
        self.cliprange = cfg.cliprange
        self.loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute Mean Squared Error value loss, optionally with clipping."""
        # Squeeze trailing singleton from value head output: [B, S, 1] -> [B, S]
        if logits.ndim > 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        values = logits

        token_mask = data["token_mask"]
        sample_mask = data["sample_mask"]
        returns = data["returns"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        if self.cliprange and self.cliprange > 0:
            old_values = data["values"]
            vpred_clipped = torch.clamp(
                values,
                old_values - self.cliprange,
                old_values + self.cliprange,
            )
            vf_losses_unclipped = (values - returns) ** 2
            vf_losses_clipped = (vpred_clipped - returns) ** 2
            vf_losses = torch.max(vf_losses_unclipped, vf_losses_clipped)
            loss = (
                0.5
                * self.scale
                * masked_mean(
                    vf_losses,
                    mask,
                    global_normalization_factor=global_valid_toks,
                )
            )
            vf_clipfrac = masked_mean(
                (vf_losses_clipped > vf_losses_unclipped).float(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
        else:
            loss = torch.nn.functional.mse_loss(values, returns, reduction="none")
            loss = (
                0.5
                * self.scale
                * masked_mean(
                    loss,
                    mask,
                    global_normalization_factor=global_valid_toks,
                )
            )
            vf_clipfrac = 0.0

        with torch.no_grad():
            # Use global_valid_toks so each MB contributes local_sum/global_total.
            # Summing across MBs in ppo.py then gives the correct global mean.
            returns_mean = masked_mean(
                returns,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            values_mean = masked_mean(
                values,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

            # Min/max are per-MB; ppo.py takes min/max across MBs.
            masked_values = values[mask.bool()]
            values_min = (
                masked_values.min().item() if masked_values.numel() > 0 else 0.0
            )
            values_max = (
                masked_values.max().item() if masked_values.numel() > 0 else 0.0
            )

            # Explained variance sufficient statistics.
            # EV = 1 - Var(residual) / Var(returns)
            # We export E[r²] and E[(r-v)²] (both / global_valid_toks so they
            # sum correctly across MBs).  ppo.py combines them with returns_mean
            # and values_mean to compute exact global EV.
            returns_sq_mean = masked_mean(
                returns**2,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            residual_sq_mean = masked_mean(
                (returns - values) ** 2,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

        metrics = {
            "loss": float(loss.item()),
            "vf_clipfrac": vf_clipfrac,
            "returns_mean": returns_mean,
            "values_mean": values_mean,
            "values_min": values_min,
            "values_max": values_max,
            "returns_sq_mean": returns_sq_mean,
            "residual_sq_mean": residual_sq_mean,
            "num_valid_samples": int(values.shape[0]),
        }

        return loss, metrics


# =====================================================================
# Cross-tokenizer distillation
# =====================================================================


def _generalized_jsd(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Per-element generalized Jensen-Shannon Divergence on log-probabilities.

    Mirrors TRL's ``GOLDTrainer.generalized_jsd_loss`` (huggingface/trl @v1.0.0
    ``trl/experimental/gold/gold_trainer.py``). Inputs are log-probs (may be
    sub-distribution slices - no normalization is applied here, so the
    divergence is on raw probability mass).

    beta semantics (matches TRL):
      * beta = 0   -> forward KL : KL(teacher || student)   (mean-seeking)
      * beta = 1   -> reverse KL : KL(student || teacher)   (mode-seeking; TRL's default)
      * beta = 0.5 -> symmetric JSD
      * 0 < beta < 1 -> blended KL on a (1-beta).student + beta.teacher mixture

    Numerics: computation is forced to fp32 even when inputs are bf16/fp16.
    bf16's ~7.8e-3 epsilon causes (student_log_probs - teacher_log_probs) to
    underflow as the student converges, collapsing the matched-term loss to
    exactly 0 mid-training. Cast back to the input dtype on return.

    Returns a tensor with the same shape as the inputs (per-element divergence,
    matching F.kl_div(reduction='none')). Caller does the chunk/vocab sum.
    """
    in_dtype = student_log_probs.dtype
    s_lp = student_log_probs.to(torch.float32)
    t_lp = teacher_log_probs.to(torch.float32)

    if beta == 0.0:
        out = torch.nn.functional.kl_div(
            s_lp,
            t_lp,
            reduction="none",
            log_target=True,
        )
        return out.to(in_dtype)
    if beta == 1.0:
        out = torch.nn.functional.kl_div(
            t_lp,
            s_lp,
            reduction="none",
            log_target=True,
        )
        return out.to(in_dtype)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=s_lp.device)
    # mixture_log_probs[v] = log( (1-beta)*p_s(v) + beta*p_t(v) )
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                s_lp + torch.log1p(-beta_t),
                t_lp + torch.log(beta_t),
            ]
        ),
        dim=0,
    )
    kl_teacher = torch.nn.functional.kl_div(
        mixture_log_probs,
        t_lp,
        reduction="none",
        log_target=True,
    )
    kl_student = torch.nn.functional.kl_div(
        mixture_log_probs,
        s_lp,
        reduction="none",
        log_target=True,
    )
    out = beta_t * kl_teacher + (1 - beta_t) * kl_student
    return out.to(in_dtype)


def _combine_v3_mismatch_terms(
    last_loss: torch.Tensor,
    pos0_loss: torch.Tensor,
    *,
    pos0_coefficient: float,
    loss_multiplier: float,
    convex: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine the two non-1-to-1 v6 loss terms.

    New configs use ``beta * (L_last + alpha * L_pos0)``. ``convex=True``
    retains the historical formula for configs using the legacy lambda key.
    """
    if convex:
        combined = (1.0 - pos0_coefficient) * last_loss + pos0_coefficient * pos0_loss
    else:
        combined = last_loss + pos0_coefficient * pos0_loss
    return combined, combined * loss_multiplier


class CrossTokenizerDistillationLossConfig(TypedDict):
    """Config for cross-tokenizer distillation loss.

    Attributes:
        projection_matrix_paths: Per-teacher list of filesystem paths to the
            .pt projection file (``None`` marks a same-tokenizer teacher: direct
            KL, no projection). Each .pt holds either the dense top-k projection
            (dict with 'indices' and 'likelihoods' tensors of shape
            [V_student, top_k]) or the sparse multi-token format
            (dict[(student_id, teacher_id)] -> count), loaded lazily on first
            call by each worker process. Runtime-injected by
            ``xtoken_off_policy_distillation.setup`` from ``teachers[i]``; not a
            user loss_fn key in YAML.
        temperature: Softmax temperature applied symmetrically to student
            and teacher logits before KL.
        vocab_topk: Microbatch-global top-k size used by the same-vocab
            direct-KL path (``_direct_topk_kl``); selected student-side over
            the reassembled full-vocab teacher logits.
        reverse_kl: If True, compute KL(student || teacher) instead of
            KL(teacher || student).
        kl_loss_weight: Scalar multiplier on the distillation (KD) term in
            fixed-weight mode (``dynamic_loss_scaling=False``).
        ce_loss_scale: Scalar multiplier on the next-token CE term in
            fixed-weight mode.
        dynamic_loss_scaling: If True, rescale the KD term each step so its
            detached magnitude matches CE, then add CE; ``kl_loss_weight`` /
            ``ce_loss_scale`` are ignored in this mode.
        student_vocab_size: Full student tokenizer vocab size, used to size
            the projection matrix's student-side (V_s) axis. Runtime-injected
            by ``xtoken_off_policy_distillation.setup`` from ``len(student_tokenizer)``;
            not a user knob in YAML. Sizing V_s from the configured tokenizer
            vocab (rather than ``max(observed student_id) + 1`` from the
            sparse projection file) keeps V_s in lockstep with
            ``logits.shape[-1]`` when the file's highest student ids happen
            to be absent.
        teacher_vocab_sizes: Per-teacher list of full teacher tokenizer vocab
            sizes, used to size each projection matrix's teacher-side (V_t) axis.
            Runtime-injected symmetrically to ``student_vocab_size`` from each
            ``len(teacher_tokenizer)``; not a user loss_fn key in YAML.
    """

    temperature: float
    vocab_topk: int  # same-vocab direct-KL path only
    reverse_kl: bool
    kl_loss_weight: float
    ce_loss_scale: float
    dynamic_loss_scaling: bool
    # Multi-teacher aggregation (user loss_fn knobs).
    kd_loss_mode: str  # "sum" | "averaged_logits" | "select_teacher"
    normalize_teacher_by_vocab: (
        bool  # sum-mode only: scale each teacher's KD by log(V_t_i)/log(min_j V_t_j)
    )
    alpha: float  # softmax temperature on dynamic teacher-weight scores (sum_weights_metric)
    sum_weights_metric: NotRequired[
        Optional[str]
    ]  # "ce" | "entropy" | "max_prob"; None => static teacher_weights. sum-mode only.
    # Runtime-injected by xtoken_off_policy_distillation.setup (parallel
    # per-teacher lists + the student vocab size); not user loss_fn keys.
    student_vocab_size: NotRequired[int]
    teacher_vocab_sizes: NotRequired[list[int]]
    projection_matrix_paths: NotRequired[list[Optional[str]]]
    teacher_weights: NotRequired[list[float]]
    # ------------------------------------------------------------------ #
    # v6 (prefix_bidir_partition_kl_v3) bundle. The cross-tokenizer KD    #
    # term is the prefix-bidir partition KL (MtoM-as-ALM + rest bucket).  #
    # All fields are NotRequired so existing YAMLs keep validating.       #
    # ------------------------------------------------------------------ #
    # Per-chunk "logits[t-1] predicts token t" shift, applied only to chunks
    # not at a sequence start (v6 default False).
    kl_chunk_shift: NotRequired[bool]
    # Derive the (common_student, common_teacher) exact-match set from the
    # forward subtoks table (length-1 chains) instead of the projection matrix.
    # Lets common-only v6 runs skip the projection matrix entirely.
    common_indices_from_subtoks: NotRequired[bool]
    # Adds the position-0 partition KL on common-vocab for each multi-token
    # chunk (v6 default False; the preset sets True).
    prefix_bidir_v3_position_0_kl: NotRequired[bool]
    # Loss fn for the position-0 common-vocab term AND the 1-to-1 common-vocab
    # KL. Choices: "kl" | "jsd" | "bce" (v6 default "kl").
    prefix_bidir_v3_loss_fn: NotRequired[str]
    # beta argument for _generalized_jsd (0.5 -> symmetric JSD; v6 default 0.5).
    prefix_bidir_v3_jsd_beta: NotRequired[float]
    # Overrides loss fn for the position-N-1 partition KL only. None means
    # "inherit from prefix_bidir_v3_loss_fn".
    prefix_bidir_v3_last_pos_loss_fn: NotRequired[Optional[str]]
    # Additive coefficient alpha for non-1-to-1 chunks: the mismatch loss is
    # beta * (position-N-1 + alpha * position-0). Non-negative; defaults 1.0.
    prefix_bidir_v3_mismatch_pos0_alpha: NotRequired[float]
    # Outer multiplier beta in the additive non-1-to-1 formula. Distinct from
    # prefix_bidir_v3_jsd_beta (which controls the JSD mixture itself).
    prefix_bidir_v3_mismatch_loss_beta: NotRequired[float]
    # Deprecated convex-blend key: lambda * position-0 + (1-lambda) *
    # position-N-1. Cannot be combined with the alpha/beta keys.
    prefix_bidir_v3_mismatch_pos0_weight: NotRequired[Optional[float]]
    # Deprecated compatibility scalar for the old convex combination formula.
    prefix_bidir_v3_mismatch_loss_scale: NotRequired[float]
    # Noise filter: when > 0, chunks whose realized teacher labels were not in
    # the teacher's true top-k are dropped from KD (keep only the CE anchor).
    prefix_bidir_v3_noise_filter_topk: NotRequired[int]
    # Pure-ALM baseline: every aligned chunk uses only the realized
    # student/teacher token-chain probability plus one rest bucket. Disables
    # the 1-to-1 common-vocab KL/JSD and the position-0 auxiliary term.
    prefix_bidir_v3_pure_alm: NotRequired[bool]
    # Temperature for pure-ALM BCE. When the v3 loss fn is "bce", the binary
    # target/input are p_teacher ** (1/tau) and p_student ** (1/tau).
    prefix_bidir_v3_alm_bce_tau: NotRequired[float]
    # Sparse teacher transport: when > 0, the teacher exports top-k logits +
    # ids + full-vocab logZ via IPC and v6 computes KL/JSD over the explicit
    # top-k support plus one rest bucket instead of full teacher logits.
    teacher_topk_ipc_k: NotRequired[int]
    # Sparse teacher IPC support policy: "row_topk" | "microbatch_global_topk".
    teacher_topk_ipc_support_mode: NotRequired[str]
    # Reserve the realized teacher label as support and use the remaining k-1
    # slots for teacher alternatives (sparse path).
    teacher_topk_ipc_keep_realized: NotRequired[bool]
    # Per-teacher forward pseudo-target table .pt paths (keys ``subtoks`` /
    # ``lengths``: student->teacher sub-token chains), needed for each cross-tok
    # teacher's prefix-support index unless pure_alm=True. ``None`` for
    # same-vocab teachers. Runtime-injected by
    # ``xtoken_off_policy_distillation.setup`` from ``teachers[i]``.
    pseudo_target_paths: NotRequired[list[Optional[str]]]
    # Per-teacher reverse pseudo-target table .pt paths (teacher->student
    # sub-token chains), needed unless pure_alm=True.
    reverse_pseudo_target_paths: NotRequired[list[Optional[str]]]


class CrossTokenizerDistillationLossDataDict(TypedDict):
    """Student-side keys are fixed; teacher-side keys are teacher-indexed.

    Only the student keys below are static. Each teacher ``i`` contributes a
    dynamic set of keys produced by ``CrossTokenizerCollator`` / the trainer and
    so cannot be enumerated here:

    - Every teacher: ``teacher_{i}_full_logits_ipc`` — List[B] of CUDA IPC handle
      dicts (``payload_ipc`` + ``buf_idx``/``sample_index_in_buf`` + TP/CP shard
      metadata) from ``Policy.get_full_logits_ipc``.
      ``rebuild_teacher_full_logits_from_ipc`` (in ``prepare_loss_input``)
      P2P-reads and reassembles full-vocab teacher logits, routing across
      heterogeneous teacher/student TP/CP.
    - Cross-tokenizer teacher only: ``teacher_{i}_input_ids`` /
      ``teacher_{i}_token_mask`` ``[B, T_t]`` and ``alignment_{i}_*``
      (``pair_valid`` / ``pair_is_correct`` ``[B, max_pairs]``;
      ``student_chunk_id`` ``[B, T_s]``; ``teacher_chunk_id`` ``[B, T_t]``;
      partition masks; ``num_chunks``).
    - Same-tokenizer teacher: no ``teacher_{i}_input_ids`` / ``alignment_{i}_*``;
      it reuses the student tokenization (identity 1:1 aligned).
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class CrossTokenizerDistillationLossFn(LossFunction):
    """Cross-tokenizer distillation loss.

    A cross-tokenizer teacher's KD term is the v6 prefix-bidir partition KL
    (``_compute_prefix_bidir_partition_kl_v3``): a common-vocab partition KL/JSD
    on the exact 1-to-1 token map plus, for non-1-to-1 chunks, a batched
    mismatch partition KL over prefix-support alternatives (MtoM-as-ALM + rest
    bucket) and an optional position-0 partition KL. It is combined with a
    next-token student CE term as ``kl_loss_weight * kd + ce_loss_scale * ce`` —
    or, when ``dynamic_loss_scaling`` is set, with the KD term rescaled each
    step to match the detached CE magnitude.

    Multi-teacher: ``setup`` injects per-teacher metadata (projection paths,
    weights, vocab sizes, pseudo-target-table paths). The per-teacher KD terms
    are aggregated by ``kd_loss_mode`` (``sum`` / ``averaged_logits`` /
    ``select_teacher``) and combined with a single student CE term. A teacher
    with a ``None`` projection path is a *same-tokenizer* teacher: projection and
    alignment are skipped and its KD term is a direct top-k per-position KL on the
    shared vocab (top-k selected student-side over the reassembled full-vocab
    teacher logits). The single-teacher path is just ``num_teachers == 1``.

    Inputs (via ``LossInputType.DISTILLATION_CROSS_TOKENIZER``):
        logits: ``[B, T_s, V_s]`` raw student logits from the worker forward.
        student_logits_contig: CP-relaid contiguous student logits shared by
            every teacher's KD term.
        teacher_full_logits_by_idx: ``dict[int, [B, T, V_t]]`` full-vocab teacher
            logits per teacher, rebuilt from the CUDA IPC handles by
            ``prepare_loss_input`` (see
            :func:`nemo_rl.algorithms.x_token.loss_utils.rebuild_teacher_full_logits_from_ipc`).
        aligns_by_idx: ``dict[int, LocalizedAlignment]`` per teacher (cross-tok:
            localized chunk alignment; same-tok: thin, student fields only).

    Inputs (via ``data: BatchedDataDict``):
        See :class:`CrossTokenizerDistillationLossDataDict`.

    Returns:
        ``(loss, metrics)``. Aggregate metrics: ``loss``, ``kl_loss`` (the
        aggregated KD term), ``ce_loss``, ``kl_loss_scale``, ``accuracy``,
        ``num_valid_samples``. Per-teacher metrics are suffixed ``_t{i}`` (e.g.
        ``kl_loss_t0``, ``top1_acc_per_chunk_t0``, ``weight_t0``);
        ``select_teacher`` additionally reports ``selected_teacher``.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION_CROSS_TOKENIZER

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        # Dynamic teacher weighting (sum_weights_metric) and normalize_teacher_by_vocab
        # are only applied in kd_loss_mode="sum"; reject the combo instead of
        # silently ignoring them under the other modes.
        if cfg.get("sum_weights_metric") is not None and cfg["kd_loss_mode"] != "sum":
            raise ValueError(
                f"sum_weights_metric={cfg['sum_weights_metric']!r} is only applied "
                f"in kd_loss_mode='sum'; it is ignored by '{cfg['kd_loss_mode']}'. "
                "Unset one of them."
            )
        if cfg.get("normalize_teacher_by_vocab") and cfg["kd_loss_mode"] != "sum":
            raise ValueError(
                "normalize_teacher_by_vocab is only applied in kd_loss_mode='sum'; "
                f"it is ignored by '{cfg['kd_loss_mode']}'. Unset one of them."
            )
        # averaged_logits forms a convex combination of teacher logits
        # (weight_i / sum(weights)); a zero weight-sum (all zeros, or a signed
        # set cancelling to 0) makes that division undefined. Reject it here
        # rather than fail with a deep ZeroDivisionError mid-step.
        _weights = cfg.get("teacher_weights")
        if (
            cfg["kd_loss_mode"] == "averaged_logits"
            and _weights is not None
            and sum(_weights) == 0
        ):
            raise ValueError(
                "teacher_weights must not sum to zero in "
                "kd_loss_mode='averaged_logits' (they form a convex combination "
                f"of teacher logits); got teacher_weights={list(_weights)}."
            )
        # Global loss knobs (shared across all teachers).
        self.temperature = cfg["temperature"]
        # vocab_topk / reverse_kl are consumed by the same-vocab direct-KL path
        # (_direct_topk_kl); the cross-tokenizer path is v6.
        self.vocab_topk = cfg["vocab_topk"]
        self.reverse_kl = cfg["reverse_kl"]
        self.kl_loss_weight = cfg["kl_loss_weight"]
        self.ce_loss_scale = cfg["ce_loss_scale"]
        self.dynamic_loss_scaling = cfg["dynamic_loss_scaling"]
        self.student_vocab_size = cfg["student_vocab_size"]
        # Multi-teacher aggregation knobs.
        self.kd_loss_mode = cfg["kd_loss_mode"]
        self.normalize_teacher_by_vocab = cfg["normalize_teacher_by_vocab"]
        self.alpha = cfg["alpha"]
        # sum_weights_metric is NotRequired -> None means static teacher_weights.
        self.sum_weights_metric = cfg.get("sum_weights_metric")
        # Per-teacher metadata: parallel lists (one entry per ``teachers[i]``),
        # injected by ``xtoken_off_policy_distillation.setup``. Every teacher ships
        # full-vocab logits (the loss derives the top-k subset student-side), so
        # there is no per-teacher ``send_full_logits`` flag.
        self.projection_matrix_paths = list(cfg["projection_matrix_paths"])
        self.teacher_vocab_sizes = list(cfg["teacher_vocab_sizes"])
        self.teacher_weights = list(cfg["teacher_weights"])
        # Every per-teacher list must have the same length (one entry per
        # teacher); a mismatch would otherwise surface as a deep IndexError
        # mid-training instead of a clear error here.
        per_teacher_lens = {
            "projection_matrix_paths": len(self.projection_matrix_paths),
            "teacher_vocab_sizes": len(self.teacher_vocab_sizes),
            "teacher_weights": len(self.teacher_weights),
        }
        if len(set(per_teacher_lens.values())) != 1:
            raise ValueError(
                f"per-teacher lists must be equal length, got {per_teacher_lens}"
            )
        self.num_teachers = len(self.projection_matrix_paths)
        # ------------------------------------------------------------------
        # v6 (prefix_bidir_partition_kl_v3) state. Each cross-tokenizer teacher's
        # KD term is the prefix-bidir partition KL; same-vocab teachers keep the
        # direct-KL path and hold no v6 state. Per-teacher artifact caches
        # (common-index set, prefix-support index, pseudo-target tables) are
        # keyed by teacher index so any number of cross-tok teachers compose.
        # ------------------------------------------------------------------
        self.common_indices_from_subtoks = bool(
            cfg.get("common_indices_from_subtoks", False)
        )
        # Per-teacher pseudo-target table paths (forward: student->teacher
        # chains; reverse: teacher->student chains). None for same-vocab
        # teachers. Runtime-injected symmetrically to projection_matrix_paths.
        self.pseudo_target_paths: list[Optional[str]] = list(
            cfg.get("pseudo_target_paths", [None] * self.num_teachers)
        )
        self.reverse_pseudo_target_paths: list[Optional[str]] = list(
            cfg.get("reverse_pseudo_target_paths", [None] * self.num_teachers)
        )
        # Optional per-microbatch loss dump for parity comparison
        # (NRL_XTOKEN_LOSS_DUMP_DIR). Raw floats from the loss-compute site.
        self._loss_dump_dir = os.environ.get("NRL_XTOKEN_LOSS_DUMP_DIR")
        self._loss_dump_records: list[dict[str, Any]] = []
        self._loss_dump_call_idx = 0
        # Retain the raw config: the v6 (prefix_bidir_partition_kl_v3) entry
        # reads ~20 global knobs (temperature, reverse_kl, prefix_bidir_v3_*,
        # teacher_topk_ipc_*, ...) via ``cfg.get(...)``. Per-teacher state
        # (vocab size, projection / pseudo-target paths) is threaded separately.
        self.cfg = cfg
        # Lazy v6 caches, populated on first call, keyed by (device, teacher_idx).
        self._v3_common_indices_per_device: dict[
            tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._v3_teacher_to_common_student_per_device: dict[
            tuple[torch.device, int], torch.Tensor
        ] = {}
        self._v3_prefix_index_per_device: dict[
            tuple[torch.device, int], dict[str, Any]
        ] = {}
        # Per-teacher pseudo-target tables, loaded lazily. idx -> (subtoks,
        # lengths); forward = student->teacher chains, reverse = teacher->student.
        self._v3_fwd_subtoks_per_teacher: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._v3_rev_subtoks_per_teacher: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # The materialized projection matrix and the derived exact-map
        # partition both live in process-local caches in
        # ``x_token.loss_utils`` (see ``get_sparse_projection_matrix``,
        # ``get_topk_projection``, ``build_exact_token_map``), not on
        # this instance. That keeps the driver-side ``loss_fn`` free of
        # any large CUDA tensors and lets multiple loss instances on
        # the same worker share one load.

    def _teacher_is_same_vocab(self, i: int) -> bool:
        """A teacher is same-vocab (direct KL, no projection) iff its path is None."""
        return self.projection_matrix_paths[i] is None

    def __call__(
        self,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        logits: torch.Tensor,
        student_logits_contig: torch.Tensor,
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        *,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the (multi-teacher) cross-tokenizer distillation loss.

        Per-teacher KD terms are aggregated per ``kd_loss_mode`` and combined
        with a single student next-token CE term — dynamic-scaled when
        ``dynamic_loss_scaling`` is set (KD rescaled to match the detached CE
        magnitude, ``kl_loss_weight`` / ``ce_loss_scale`` ignored), else
        fixed-weighted. The single-teacher path is just ``num_teachers == 1``.

        ``student_logits_contig`` (CP-relaid) and the per-teacher ``aligns_by_idx``
        / ``teacher_full_logits_by_idx`` are precomputed in ``prepare_loss_input``;
        the raw ``logits`` is kept for the CE term.
        """
        ce_loss = self._compute_ce(logits, data, global_valid_toks)

        if self.kd_loss_mode == "sum":
            total_kd, per_teacher_metrics = self._sum_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "averaged_logits":
            total_kd, per_teacher_metrics = self._averaged_logits_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "select_teacher":
            total_kd, per_teacher_metrics = self._select_teacher_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        else:
            raise ValueError(f"Unknown kd_loss_mode: {self.kd_loss_mode!r}")

        # Combine the aggregated KD term with the single student CE term.
        if self.dynamic_loss_scaling:
            # loss = sg(ce/kd) * kd + ce; user kl_loss_weight / ce_loss_scale
            # are intentionally ignored in this branch.
            kd_detached = total_kd.detach().abs()
            ce_detached = ce_loss.detach().abs()
            kl_scale = torch.where(
                kd_detached > 0,
                ce_detached / kd_detached,
                torch.ones_like(kd_detached),
            )
            loss = kl_scale * total_kd + ce_loss
        else:
            kl_scale = torch.tensor(1.0, device=total_kd.device, dtype=total_kd.dtype)
            loss = self.kl_loss_weight * total_kd + self.ce_loss_scale * ce_loss

        # Next-token accuracy on the student side (quick per-step signal), masked
        # to valid tokens. Computed once on the student from the shared CP-relaid
        # fields (carried on every teacher's align); the CP-aware shift pairs
        # predictors with the right labels under load-balanced sharding.
        align0 = aligns_by_idx[0]
        accuracy = next_token_accuracy(
            student_logits_contig,
            input_ids=align0.student_input_ids,
            token_mask=align0.student_token_mask,
            sample_mask=data["sample_mask"],
            tp_group=tp_group,
            cp_group=cp_group,
        )

        metrics: dict[str, Any] = {
            "loss": loss.item(),
            # Aggregate KD term (kept under ``kl_loss`` so existing trainer
            # metric handling continues to work); per-teacher terms are suffixed
            # ``_t{i}``.
            "kl_loss": total_kd.item(),
            "ce_loss": ce_loss.item(),
            "kl_loss_scale": kl_scale.item(),
            "accuracy": accuracy.item(),
            "num_valid_samples": data["input_ids"].shape[0],
        }
        metrics.update(per_teacher_metrics)
        return loss, metrics

    # ------------------------------------------------------------------ #
    # Multi-teacher aggregation
    # ------------------------------------------------------------------ #
    def _compute_teacher_kd(
        self,
        i: int,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """KD term for teacher ``i`` plus its (unsuffixed) metrics.

        Dispatches on tokenizer kind: same-vocab -> direct top-k per-position KL;
        cross-tokenizer -> the v6 prefix-bidir partition KL over teacher ``i``'s
        localized alignment. Both consume the shared CP-relaid student logits and
        route TP/CP through the parameterized loss-mode helpers.
        """
        if self._teacher_is_same_vocab(i):
            return self._compute_same_vocab_kl(
                i,
                student_logits_contig,
                teacher_full_logits_by_idx[i],
                aligns_by_idx[i],
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )

        kd, v6_metrics = self._compute_prefix_bidir_partition_kl_v3(
            i,
            student_logits_contig,
            teacher_full_logits_by_idx[i],
            aligns_by_idx[i],
            teacher_vocab_size=self.teacher_vocab_sizes[i],
            tp_group=tp_group,
            cp_group=cp_group,
        )
        # Surface the KD value under the shared per-teacher metric key and keep
        # the v6 chunk diagnostics; the dispatcher-level loss keys are dropped
        # (the aggregate ``loss`` / ``kl_loss`` are set by ``__call__``).
        metrics: dict[str, Any] = {"kl_loss": kd.item()}
        for key in (
            "kl_common_per_chunk",
            "kl_partition_last_per_chunk",
            "kl_partition_first_per_chunk",
            "top1_acc_per_chunk",
            "num_common_chunks",
            "num_mismatch_chunks",
        ):
            if key in v6_metrics:
                metrics[key] = v6_metrics[key]
        return kd, metrics

    def _compute_same_vocab_kl(
        self,
        i: int,
        student_logits_contig: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Direct top-k per-position KL for a same-tokenizer teacher.

        Identical tokenizer => teacher tokens == student tokens (identity
        position alignment), so no projection / no chunk-averaging. The reduction
        matches CE: masked next-token mean normalized by ``global_valid_toks``,
        scaled by ``T**2``.
        """
        kd = self._direct_topk_kl(
            student_logits_contig,
            teacher_full_logits,
            align,
            global_valid_toks,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kd, {"kl_loss": kd.item()}

    def _direct_topk_kl(
        self,
        student_logits: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Top-K per-position KL on a shared vocab (same tokenizer), TP/CP-aware.

        Top-K columns are selected at the student from the reassembled full-vocab
        teacher logits (``select_teacher_topk_indices`` MAX-reduces across CP so
        every CP rank agrees on the same columns). The student is gathered to full
        vocab across TP (``vocab_parallel_full_log_softmax``) *before* slicing —
        slicing a TP-local shard would pick the wrong columns. Both sides are
        renormalized within the K-subset (the teacher's subset-softmax and the
        student's full-then-renorm are mathematically identical, the full-vocab
        partition function cancels). The masked next-token mean is normalized by
        the CP/DP-global valid-token count, exactly like the CE term; at CP=1 the
        ``cp_shift_next`` mask reduces to the ``token_mask[:, 1:]`` next-token
        shift.
        """
        T = self.temperature
        # Drop HF lm_head padding beyond the shared tokenizer vocab.
        v_s = self.student_vocab_size
        teacher = teacher_full_logits
        if teacher.shape[-1] > v_s:
            teacher = teacher[..., :v_s]
        vocab_topk = min(self.vocab_topk, teacher.shape[-1])
        topk_idx = select_teacher_topk_indices(teacher, vocab_topk, cp_group=cp_group)

        student_log_probs = vocab_parallel_full_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        student_gathered = student_log_probs[..., topk_idx]
        student_log_probs_k = student_gathered - torch.logsumexp(
            student_gathered, dim=-1, keepdim=True
        )
        teacher_log_probs_k = torch.log_softmax(
            teacher[..., topk_idx].float() / T, dim=-1
        )
        if self.reverse_kl:
            per_pos = torch.nn.functional.kl_div(
                teacher_log_probs_k,
                student_log_probs_k,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
        else:
            per_pos = torch.nn.functional.kl_div(
                student_log_probs_k,
                teacher_log_probs_k,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
        return self._same_vocab_masked_kl(per_pos, align, global_valid_toks, cp_group)

    def _direct_full_vocab_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Full-vocab per-position KL on a shared vocab (same tokenizer), TP/CP-aware.

        Used by ``averaged_logits`` over the convex-averaged teacher logits. The
        student is gathered to full vocab across TP; the teacher (already full
        vocab) is sliced to the student width to drop HF lm_head padding.
        """
        T = self.temperature
        student_log_probs = vocab_parallel_full_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        v_s = student_log_probs.shape[-1]
        teacher = teacher_logits.float()
        if teacher.shape[-1] > v_s:
            teacher = teacher[..., :v_s]
        teacher_log_probs = torch.log_softmax(teacher / T, dim=-1)
        if self.reverse_kl:
            per_pos = torch.nn.functional.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
        else:
            per_pos = torch.nn.functional.kl_div(
                student_log_probs, teacher_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
        return self._same_vocab_masked_kl(per_pos, align, global_valid_toks, cp_group)

    def _same_vocab_masked_kl(
        self,
        per_pos: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Masked next-token mean of a per-position KL (same-tokenizer reduction).

        ``per_pos`` is the per-position KL on this CP rank's contiguous window
        (student and teacher are position-aligned). The CP-aware next-token shift
        (:func:`cp_shift_next`) selects positions whose target token (p+1) is a
        valid label — at CP=1 this is the plain ``token_mask[:, 1:]`` shift, the
        global-last position dropped via ``fill=0``. Reduction matches CE:
        ``masked_mean`` over ``global_valid_toks``, scaled by ``T**2``.
        """
        T = self.temperature
        next_mask = cp_shift_next(
            to_local_if_dtensor(align.student_token_mask), cp_group, fill=0
        )
        sample_mask = to_local_if_dtensor(align.sample_mask)
        mask = next_mask.float() * sample_mask.unsqueeze(-1).float()
        return (
            masked_mean(per_pos, mask, global_normalization_factor=global_valid_toks)
            * T
            * T
        )

    def _sum_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Weighted sum: ``total_kd = Σ_i weight_i · KD_i``.

        Weights are static (config ``weight``) or dynamic (``sum_weights_metric``).
        When ``normalize_teacher_by_vocab`` is set, each teacher's KD is
        additionally scaled by ``log(V_t_i) / log(min_j V_t_j)``.
        """
        device = student_logits_contig.device
        if self.sum_weights_metric is not None:
            weights = self._compute_dynamic_weights(
                data, teacher_full_logits_by_idx, aligns_by_idx
            )
        else:
            weights = [
                torch.tensor(
                    self.teacher_weights[i],
                    device=device,
                    dtype=student_logits_contig.dtype,
                )
                for i in range(self.num_teachers)
            ]

        if self.normalize_teacher_by_vocab:
            temp_weight = torch.log(
                torch.tensor(float(min(self.teacher_vocab_sizes)), device=device)
            )

        total_kd: Optional[torch.Tensor] = None
        per_metrics: dict[str, Any] = {}
        # Deterministic teacher order: each teacher's KD fires its own
        # collectives, so the order must match across ranks.
        for i in range(self.num_teachers):
            kd_i, m_i = self._compute_teacher_kd(
                i,
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            weighted = kd_i * weights[i]
            if self.normalize_teacher_by_vocab:
                v_scale = (
                    torch.log(
                        torch.tensor(float(self.teacher_vocab_sizes[i]), device=device)
                    )
                    / temp_weight
                )
                weighted = weighted * v_scale
            total_kd = weighted if total_kd is None else total_kd + weighted
            for k, v in m_i.items():
                per_metrics[f"{k}_t{i}"] = v
            per_metrics[f"weight_t{i}"] = float(weights[i].item())
        assert total_kd is not None
        return total_kd, per_metrics

    def _averaged_logits_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convex-weighted average of teacher logits, then one direct KL.

        Valid only when all teachers are same-tokenizer (no projection) and ship
        full logits of identical shape. Otherwise falls back to a plain
        static-weight sum (no dynamic weights, no ``normalize_teacher_by_vocab``).
        """
        full = [teacher_full_logits_by_idx.get(i) for i in range(self.num_teachers)]
        # Direct per-position KL is only valid when every teacher shares the
        # student's tokenizer (no projection matrix) *and* ships full logits of
        # identical shape. Two cross-tokenizer teachers can have matching shapes
        # yet still need the projection/alignment path, so the shape check alone
        # is insufficient.
        same_tokenizer = all(p is None for p in self.projection_matrix_paths)
        same_shape = all(f is not None for f in full) and (
            len({tuple(f.shape) for f in full if f is not None}) == 1
        )
        if not (same_tokenizer and same_shape):
            total_kd: Optional[torch.Tensor] = None
            per_metrics: dict[str, Any] = {}
            for i in range(self.num_teachers):
                kd_i, m_i = self._compute_teacher_kd(
                    i,
                    student_logits_contig,
                    data,
                    teacher_full_logits_by_idx,
                    aligns_by_idx,
                    global_valid_toks,
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
                w = self.teacher_weights[i]
                weighted = kd_i * w
                total_kd = weighted if total_kd is None else total_kd + weighted
                for k, v in m_i.items():
                    per_metrics[f"{k}_t{i}"] = v
                per_metrics[f"weight_t{i}"] = float(w)
            assert total_kd is not None
            return total_kd, per_metrics

        total_w = sum(self.teacher_weights)
        avg: Optional[torch.Tensor] = None
        for i, f in enumerate(full):
            assert f is not None
            contrib = f.float() * (self.teacher_weights[i] / total_w)
            avg = contrib if avg is None else avg + contrib
        assert avg is not None
        kd = self._direct_full_vocab_kl(
            student_logits_contig,
            avg,
            aligns_by_idx[0],
            global_valid_toks,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kd, {"kl_loss": kd.item()}

    def _dp_global_masked_mean(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked mean of ``values`` over the *process-global* valid count.

        Teacher selection / dynamic weighting must be identical on every rank: a
        rank-local mean lets ranks pick a different teacher / different weights,
        and the per-teacher KD's collectives then see divergent participation
        (deadlock when one rank's choice fires a collective another's does not).
        All-reduce the masked sum and the mask count over the full group so every
        rank gets the same score (the WORLD-reduced-denominator convention). The
        result is detached (it gates selection / weighting and is not
        back-propagated).
        """
        num = group_all_reduce_sum(
            (values * mask).sum(), group=torch.distributed.group.WORLD
        )
        den = group_all_reduce_sum(
            mask.sum(), group=torch.distributed.group.WORLD
        ).clamp(min=1.0)
        return num / den

    def _select_teacher_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Use only the teacher with the lowest next-token CE on its own tokens."""
        with torch.no_grad():
            ces: list[float] = []
            for i in range(self.num_teachers):
                t_logits, t_ids, t_mask = self._teacher_score_inputs(
                    i, data, teacher_full_logits_by_idx, aligns_by_idx
                )
                ce_pos = torch.nn.functional.cross_entropy(
                    t_logits[:, :-1].reshape(-1, t_logits.shape[-1]).float(),
                    t_ids[:, 1:].reshape(-1),
                    reduction="none",
                )
                mask = (
                    t_mask[:, 1:].float() * data["sample_mask"].unsqueeze(-1).float()
                ).reshape(-1)
                ces.append(self._dp_global_masked_mean(ce_pos, mask).item())
            best = int(min(range(self.num_teachers), key=lambda j: ces[j]))

        kd, m = self._compute_teacher_kd(
            best,
            student_logits_contig,
            data,
            teacher_full_logits_by_idx,
            aligns_by_idx,
            global_valid_toks,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        per_metrics: dict[str, Any] = {f"{k}_t{best}": v for k, v in m.items()}
        per_metrics["selected_teacher"] = best
        return kd, per_metrics

    def _teacher_score_inputs(
        self,
        i: int,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, input_ids, token_mask)`` for teacher ``i``'s CE / weight-metric scores.

        The token mask is over the tokenization the score is computed on: the
        shared student tokens (CP-relaid) for a same-vocab teacher, teacher ``i``'s
        own otherwise. Every teacher ships full logits, so the full distribution
        is always available.
        """
        if self._teacher_is_same_vocab(i):
            align = aligns_by_idx[i]
            ids = to_local_if_dtensor(align.student_input_ids)
            token_mask = to_local_if_dtensor(align.student_token_mask)
        else:
            ids = to_local_if_dtensor(data[f"teacher_{i}_input_ids"])
            token_mask = to_local_if_dtensor(data[f"teacher_{i}_token_mask"])
        return teacher_full_logits_by_idx[i], ids, token_mask

    def _compute_dynamic_weights(
        self,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
    ) -> list[torch.Tensor]:
        """Sequence-level dynamic teacher weights via ``sum_weights_metric``.

        Per teacher computes a scalar score (``ce`` -> -CE, ``entropy`` ->
        -entropy, ``max_prob`` -> max prob; higher = more trusted), optionally
        rescaled by ``log(V_t_i)/log(min_j V_t_j)``, then ``softmax(alpha *
        scores)`` across teachers.
        """
        device = data["input_ids"].device
        if self.normalize_teacher_by_vocab:
            temp_weight = torch.log(
                torch.tensor(float(min(self.teacher_vocab_sizes)), device=device)
            )
        scores: list[torch.Tensor] = []
        for i in range(self.num_teachers):
            t_logits, t_ids, t_mask = self._teacher_score_inputs(
                i, data, teacher_full_logits_by_idx, aligns_by_idx
            )
            score = self._teacher_weight_score(
                t_logits, t_ids, t_mask, data["sample_mask"]
            )
            if self.normalize_teacher_by_vocab:
                v_log = torch.log(
                    torch.tensor(float(self.teacher_vocab_sizes[i]), device=device)
                )
                score = score * (v_log / temp_weight)
            scores.append(score)
        weights = torch.softmax(self.alpha * torch.stack(scores), dim=0)
        return [weights[i] for i in range(self.num_teachers)]

    def _teacher_weight_score(
        self,
        t_logits: torch.Tensor,
        t_ids: torch.Tensor,
        t_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar weight-metric score for one teacher (higher = more trusted).

        Padded positions and masked-out samples are excluded, so long-padded
        batches don't let near-uniform padding logits dominate the score.
        """
        samp = to_local_if_dtensor(sample_mask).unsqueeze(-1).float()
        if self.sum_weights_metric == "ce":
            ce_pos = torch.nn.functional.cross_entropy(
                t_logits[:, :-1].reshape(-1, t_logits.shape[-1]).float(),
                t_ids[:, 1:].reshape(-1),
                reduction="none",
            )
            mask = (t_mask[:, 1:].float() * samp).reshape(-1)
            return -self._dp_global_masked_mean(ce_pos, mask)
        mask = t_mask.float() * samp
        if self.sum_weights_metric == "entropy":
            probs = torch.softmax(t_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            return -self._dp_global_masked_mean(entropy, mask)
        if self.sum_weights_metric == "max_prob":
            probs = torch.softmax(t_logits.float(), dim=-1)
            return self._dp_global_masked_mean(probs.max(dim=-1).values, mask)
        raise ValueError(f"Unknown sum_weights_metric: {self.sum_weights_metric!r}")

    # ------------------------------------------------------------------ #
    # Loss-mode implementations
    # ------------------------------------------------------------------ #
    def _compute_ce(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        global_valid_toks: torch.Tensor,
    ) -> torch.Tensor:
        """Next-token CE on the student side (TP/CP handled by the helpers)."""
        per_token_ce = student_next_token_ce(
            logits, input_ids=data["input_ids"], seq_index=data.get("seq_index")
        )
        label_mask = ce_label_mask(
            token_mask=data["token_mask"],
            sample_mask=data["sample_mask"],
            ce_seq_len=per_token_ce.shape[1],
            dtype=per_token_ce.dtype,
        )
        return masked_mean(
            per_token_ce,
            label_mask,
            global_normalization_factor=global_valid_toks,
        )

    # === v6 (prefix_bidir_partition_kl_v3) ported methods ===
    @staticmethod
    def _rebuild_teacher_full_logits(
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
    ) -> torch.Tensor:
        """Unpack ``teacher_full_logits_ipc`` to a stacked ``[B, T_t, V_t]`` CUDA tensor.

        The IPC handles point at views the teacher worker stashed in its
        ``_teacher_ipc_buffer``; rebuilding does not allocate new memory
        on the producer side. Casts to ``float32`` to match the loss math
        (the producer also writes FP32 via :class:`FullLogitsPostProcessor`).
        """
        from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

        handles = data["teacher_full_logits_ipc"]
        consumer_device = torch.cuda.current_device()
        per_sample = [
            rebuild_cuda_tensor_from_ipc(h["logits_ipc"], consumer_device)
            for h in handles
        ]
        return torch.stack(per_sample, dim=0).float()

    @staticmethod
    def _rebuild_teacher_sparse_logits(
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Unpack sparse teacher-logit IPC handles.

        Returns:
            topk_logits:  ``[B, T_t, k]`` raw teacher logits, fp32
            topk_indices: ``[B, T_t, k]`` sorted teacher token ids, int32
            log_z:        ``[B, T_t]`` logsumexp(logits / temperature), fp32
            gt_in_topk:   optional ``[B, T_t]`` bool mask recording whether
                          the realized teacher label was in the true teacher
                          top-k before forced insertion.
        """
        from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

        handles = data["teacher_sparse_logits_ipc"]
        consumer_device = torch.cuda.current_device()
        per_sample_logits = []
        per_sample_indices = []
        per_sample_log_z = []
        per_sample_gt_in_topk = []
        first_shards = handles[0].get("teacher_shards", [handles[0]]) if handles else []
        has_gt_in_topk = bool(first_shards) and "gt_in_topk_ipc" in first_shards[0]
        for sample_entry in handles:
            shards = sample_entry.get("teacher_shards", [sample_entry])
            shards = sorted(shards, key=lambda h: int(h.get("global_seq_start", 0)))
            if any(("gt_in_topk_ipc" in h) != has_gt_in_topk for h in shards):
                raise ValueError(
                    "Sparse teacher IPC handles mix gt_in_topk and "
                    "non-gt_in_topk payloads."
                )
            if len(shards) > 1:
                expected_start = 0
                for h in shards:
                    start = int(h["global_seq_start"])
                    if start != expected_start:
                        raise ValueError(
                            "Sparse teacher CP shards must cover the sequence "
                            f"contiguously; expected start {expected_start}, got {start}."
                        )
                    expected_start += int(h["topk_shape"][0])
                full_seq_len = int(shards[0]["full_seq_len"])
                if expected_start != full_seq_len:
                    raise ValueError(
                        "Sparse teacher CP shards do not cover the full sequence: "
                        f"covered={expected_start}, full_seq_len={full_seq_len}."
                    )

            sample_logits = [
                rebuild_cuda_tensor_from_ipc(h["topk_logits_ipc"], consumer_device)
                for h in shards
            ]
            sample_indices = [
                rebuild_cuda_tensor_from_ipc(h["topk_indices_ipc"], consumer_device)
                for h in shards
            ]
            sample_log_z = [
                rebuild_cuda_tensor_from_ipc(h["log_z_ipc"], consumer_device)
                for h in shards
            ]
            per_sample_logits.append(torch.cat(sample_logits, dim=0))
            per_sample_indices.append(torch.cat(sample_indices, dim=0))
            per_sample_log_z.append(torch.cat(sample_log_z, dim=0))
            if has_gt_in_topk:
                per_sample_gt_in_topk.append(
                    torch.cat(
                        [
                            rebuild_cuda_tensor_from_ipc(
                                h["gt_in_topk_ipc"], consumer_device
                            )
                            for h in shards
                        ],
                        dim=0,
                    )
                )
        gt_in_topk = (
            torch.stack(per_sample_gt_in_topk, dim=0).to(torch.bool)
            if has_gt_in_topk
            else None
        )
        return (
            torch.stack(per_sample_logits, dim=0).float(),
            torch.stack(per_sample_indices, dim=0).to(torch.int32),
            torch.stack(per_sample_log_z, dim=0).float(),
            gt_in_topk,
        )

    def _get_v3_teacher_to_common_student(
        self,
        device: torch.device,
        i: int,
        v_t: int,
        common_student_idx_t: torch.Tensor,
        common_teacher_idx_t: torch.Tensor,
    ) -> torch.Tensor:
        cache_key = (device, i)
        cached = self._v3_teacher_to_common_student_per_device.get(cache_key)
        if cached is not None:
            return cached
        mapping = torch.full(
            (int(v_t),),
            -1,
            dtype=torch.long,
            device=device,
        )
        valid = common_teacher_idx_t < int(v_t)
        mapping[common_teacher_idx_t[valid]] = common_student_idx_t[valid]
        self._v3_teacher_to_common_student_per_device[cache_key] = mapping
        return mapping

    @staticmethod
    def _lookup_sparse_teacher_logp(
        topk_logits: torch.Tensor,
        topk_indices: torch.Tensor,
        teacher_log_z: torch.Tensor,
        b_idx: torch.Tensor,
        pos: torch.Tensor,
        token_ids: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lookup teacher log-probs for explicit token ids in sparse top-k rows.

        ``topk_indices`` must be sorted ascending along the last dimension.
        ``pos`` and ``token_ids`` may be either ``[K]`` or ``[K, N]``.
        Missing ids return an arbitrary log-prob value and ``found=False``;
        callers must mask them into the rest bucket.
        """
        squeeze = False
        if pos.dim() == 1:
            pos = pos.unsqueeze(-1)
            token_ids = token_ids.unsqueeze(-1)
            squeeze = True

        original_shape = token_ids.shape
        b_flat = b_idx.unsqueeze(-1).expand_as(pos).reshape(-1)
        pos_flat = pos.reshape(-1)
        token_flat = token_ids.reshape(-1).to(topk_indices.dtype)
        out_logp = torch.empty(
            (token_flat.numel(),),
            device=topk_logits.device,
            dtype=topk_logits.dtype,
        )
        out_found = torch.empty(
            (token_flat.numel(),),
            device=topk_indices.device,
            dtype=torch.bool,
        )

        k_top = topk_indices.shape[-1]
        row_chunk = int(os.environ.get("TOKENALIGN_SPARSE_LOOKUP_ROWS", "2048"))
        row_chunk = max(row_chunk, 1)
        for start in range(0, token_flat.numel(), row_chunk):
            end = min(start + row_chunk, token_flat.numel())
            rows_idx = topk_indices[b_flat[start:end], pos_flat[start:end]]
            rows_vals = topk_logits[b_flat[start:end], pos_flat[start:end]]
            targets = token_flat[start:end]

            insert = torch.searchsorted(rows_idx, targets.unsqueeze(-1))
            insert = insert.squeeze(-1)
            in_bounds = insert < k_top
            safe_insert = insert.clamp(max=k_top - 1)
            row_arange = torch.arange(
                rows_idx.shape[0],
                device=rows_idx.device,
                dtype=torch.long,
            )
            matched_idx = rows_idx[row_arange, safe_insert]
            found = in_bounds & (matched_idx == targets)
            raw_logits = rows_vals[row_arange, safe_insert]
            flat_log_z = teacher_log_z[b_flat[start:end], pos_flat[start:end]]

            out_logp[start:end] = raw_logits / float(temperature) - flat_log_z
            out_found[start:end] = found

        logp = out_logp.reshape(original_shape)
        found = out_found.reshape(original_shape)

        if squeeze:
            logp = logp.squeeze(-1)
            found = found.squeeze(-1)
        return logp, found

    def _maybe_dump_loss(self, metrics: dict[str, Any]) -> None:
        """Append per-call raw loss values to a per-rank dump file.

        Activated by ``NRL_XTOKEN_LOSS_DUMP_DIR``. One file per rank,
        rewritten on each call with the full record list. Records are raw
        ``loss.item()`` values from the loss-compute site — not scaled,
        aggregated, or DP-summed — matching the dump protocol used for
        PT-vs-NRL parity comparisons (cf. ``feedback_sanity_loss_dump``).
        """
        if not self._loss_dump_dir:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # The P-KL path emits kl_loss/ce_loss/kl_loss_scale/num_valid_pairs;
        # the gold-loss path emits kl_common/l1_uncommon/num_valid_chunks;
        # the v3 path emits ce_loss_per_token/kl_loss_per_token plus the
        # per-chunk diagnostics. Record everything that's present so the
        # same dump file format serves all — downstream comparison scripts
        # read by key.
        record: dict[str, Any] = {
            "call_idx": self._loss_dump_call_idx,
            "loss": metrics["loss"],
        }
        for k in (
            "kl_loss",
            "ce_loss",
            "kl_loss_per_chunk",
            "ce_loss_per_token",
            "kl_loss_scale",
            "num_valid_pairs",
            "kl_common",
            "kl_common_per_chunk",
            "kl_partition_first_per_chunk",
            "kl_partition_last_per_chunk",
            "l1_uncommon",
            "num_valid_chunks",
        ):
            if k in metrics:
                record[k] = metrics[k]
        self._loss_dump_records.append(record)
        self._loss_dump_call_idx += 1
        os.makedirs(self._loss_dump_dir, exist_ok=True)
        torch.save(
            self._loss_dump_records,
            os.path.join(self._loss_dump_dir, f"rank{rank}.pt"),
        )

    # ------------------------------------------------------------------ #
    # Loss-mode implementations
    # ------------------------------------------------------------------ #

    def _v3_fwd_table(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily load+cache teacher ``i``'s forward subtoks table.

        Forward = student->teacher sub-token chains (``subtoks`` / ``lengths``
        keys). Small CPU tensors, shared by ``_common_indices_from_subtoks`` and
        the forward branch of ``_ensure_bidir_prefix_support_index``.
        """
        cached = self._v3_fwd_subtoks_per_teacher.get(i)
        if cached is not None:
            return cached
        fwd_path = self.pseudo_target_paths[i] or ""
        if not fwd_path or not os.path.exists(fwd_path):
            raise RuntimeError(
                f"teacher {i}: forward pseudo-target table required "
                f"(pseudo_target_paths[{i}]); got {fwd_path!r}."
            )
        fwd = torch.load(fwd_path, map_location="cpu", weights_only=False)
        table = (fwd["subtoks"].long(), fwd["lengths"].long())
        self._v3_fwd_subtoks_per_teacher[i] = table
        return table

    def _v3_rev_table(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily load+cache teacher ``i``'s reverse subtoks table.

        Reverse = teacher->student sub-token chains, consumed by the reverse
        branch of ``_ensure_bidir_prefix_support_index``.
        """
        cached = self._v3_rev_subtoks_per_teacher.get(i)
        if cached is not None:
            return cached
        rev_path = self.reverse_pseudo_target_paths[i] or ""
        if not rev_path or not os.path.exists(rev_path):
            raise RuntimeError(
                f"teacher {i}: reverse pseudo-target table required "
                f"(reverse_pseudo_target_paths[{i}]); got {rev_path!r}."
            )
        rev = torch.load(rev_path, map_location="cpu", weights_only=False)
        table = (rev["subtoks"].long(), rev["lengths"].long())
        self._v3_rev_subtoks_per_teacher[i] = table
        return table

    def _get_common_indices_v3(
        self,
        device: torch.device,
        i: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached (common_student, common_teacher) for teacher ``i``.

        The exact 1-to-1 common set is sourced either from the forward subtoks
        table (``common_indices_from_subtoks=True``, the v6-preset path, no
        projection matrix needed) or from the strict exact-token map of the
        teacher's projection matrix. Cached per ``(device, i)``.
        """
        cache_key = (device, i)
        if cache_key in self._v3_common_indices_per_device:
            return self._v3_common_indices_per_device[cache_key]
        if self.common_indices_from_subtoks:
            common_s, common_t = self._common_indices_from_subtoks(device, i)
        else:
            # v6 has no gold/xtoken modifier -> strict exact map
            # (``xtoken_loss=False``), matching every v6 preset.
            exact_map = build_exact_token_map(
                self.projection_matrix_paths[i],
                device,
                xtoken_loss=False,
                teacher_vocab_size=self.teacher_vocab_sizes[i],
            )
            common_s = exact_map["common_student"]
            common_t = exact_map["common_teacher"]
        self._v3_common_indices_per_device[cache_key] = (common_s, common_t)
        return common_s, common_t

    def _common_indices_from_subtoks(
        self,
        device: torch.device,
        i: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive (common_student, common_teacher) from teacher ``i``'s forward subtoks table.

        A student token is exactly "common" (1-to-1 with the teacher vocab) iff
        its forward sub-token chain has length 1; the paired teacher id is
        ``subtoks_t[s, 0]``. This reads the same forward table the prefix
        support index uses (``pseudo_target_paths[i]``, keys
        ``subtoks``/``lengths``), so a common-only run needs no projection
        matrix. Teacher collisions (rare for true exact matches) are deduped
        keeping the lowest student id, so ``common_teacher`` is unique. Result
        is sorted by student index, like the projection strict exact-token map.
        """
        subtoks_t, lengths_t = self._v3_fwd_table(i)
        v_teacher = int(self.teacher_vocab_sizes[i])

        common_s = (lengths_t == 1).nonzero(as_tuple=True)[0]
        common_t = subtoks_t[common_s, 0]
        in_bounds = (common_t >= 0) & (common_t < v_teacher)
        common_s = common_s[in_bounds]
        common_t = common_t[in_bounds]

        if common_t.numel() > 0:
            # Sort by student id, then keep the first (lowest-student) entry per
            # teacher id so common_teacher is unique.
            order = torch.argsort(common_s)
            common_s, common_t = common_s[order], common_t[order]
            sort_t, t_order = torch.sort(common_t, stable=True)
            first_of_t = torch.ones_like(sort_t, dtype=torch.bool)
            first_of_t[1:] = sort_t[1:] != sort_t[:-1]
            keep = torch.zeros(common_t.shape[0], dtype=torch.bool)
            keep[t_order[first_of_t]] = True
            common_s, common_t = common_s[keep], common_t[keep]
            re_order = torch.argsort(common_s)
            common_s, common_t = common_s[re_order], common_t[re_order]

        return common_s.to(device), common_t.to(device)

    def _ensure_bidir_prefix_support_index(
        self,
        device: torch.device,
        i: int,
    ) -> dict[str, Any]:
        """Lazy-build teacher ``i``'s bidirectional prefix support index.

        The cache maps ``(length, prefix_tuple) -> tuple[(s_id, t_id), ...]``
        for both the forward (``pseudo_target_paths[i]``) and reverse
        (``reverse_pseudo_target_paths[i]``) pseudo-target tables. Built once
        per ``(device, i)``; the table loaders raise if a path is missing.
        """
        cache_key = (device, i)
        if cache_key in self._v3_prefix_index_per_device:
            return self._v3_prefix_index_per_device[cache_key]

        # Small CPU tensors (shape [V, max_chain]); slices move to device on
        # demand.
        subtoks_t_cpu, lengths_t_cpu = self._v3_fwd_table(i)
        subtoks_s_cpu, lengths_s_cpu = self._v3_rev_table(i)

        # Build the prefix-keyed dicts.
        forward: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = (
            collections.defaultdict(list)
        )
        for s_id, length in enumerate(lengths_t_cpu.tolist()):
            length = int(length)
            if length <= 1 or length > subtoks_t_cpu.size(1):
                continue
            row = subtoks_t_cpu[s_id, :length]
            if bool((row < 0).any().item()):
                continue
            prefix = tuple(int(x) for x in row[: length - 1].tolist())
            final_t = int(row[length - 1].item())
            forward[(length, prefix)].append((int(s_id), final_t))

        reverse: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = (
            collections.defaultdict(list)
        )
        for t_id, length in enumerate(lengths_s_cpu.tolist()):
            length = int(length)
            if length <= 1 or length > subtoks_s_cpu.size(1):
                continue
            row = subtoks_s_cpu[t_id, :length]
            if bool((row < 0).any().item()):
                continue
            prefix = tuple(int(x) for x in row[: length - 1].tolist())
            final_s = int(row[length - 1].item())
            reverse[(length, prefix)].append((int(t_id), final_s))

        # Pre-dedupe (tokenalign.py:5984-6017). The same env override
        # gate is preserved so A/B verification stays available.
        _prededuped = os.environ.get("TOKENALIGN_PREDEDUP_PREFIX_INDEX", "1") == "1"

        def _dedup_by_second(pairs_list):
            seen_second = set()
            out = []
            for p in pairs_list:
                second = p[1]
                if second in seen_second:
                    continue
                seen_second.add(second)
                out.append(p)
            return tuple(out)

        if _prededuped:
            cache: dict[str, Any] = {
                "forward": {
                    key: _dedup_by_second(value) for key, value in forward.items()
                },
                "reverse": {
                    key: _dedup_by_second(value) for key, value in reverse.items()
                },
                "_prededuped": True,
            }
        else:
            cache = {
                "forward": {key: tuple(value) for key, value in forward.items()},
                "reverse": {key: tuple(value) for key, value in reverse.items()},
                "_prededuped": False,
            }
        # Cache the table sizes used by the chunk-classification skip rules.
        cache["max_chain_t"] = int(subtoks_t_cpu.size(1))
        cache["max_chain_s"] = int(subtoks_s_cpu.size(1))
        self._v3_prefix_index_per_device[cache_key] = cache
        return cache

    @staticmethod
    def _append_rest_bucket_logp(
        logp: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Append log probability for the complement of a support set.

        Verbatim port of tokenalign.py:_append_rest_bucket_logp (line 6083).
        ``eps=1e-12`` (per the porting spec's faithful-numerics rule) is
        the clamp floor for the complement mass.
        """
        if logp.numel() == 0:
            return logp
        work = logp.float() if logp.dtype in (torch.float16, torch.bfloat16) else logp
        support_mass = work.exp().sum(dim=-1, keepdim=True)
        rest_mass = (1.0 - support_mass).clamp(min=eps)
        return torch.cat([work, rest_mass.log()], dim=-1)

    @staticmethod
    def _binary_power_bce_from_logp(
        student_logp_with_rest: torch.Tensor,
        teacher_logp_with_rest: torch.Tensor,
        tau: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Soft-label BCE on ALM's binary realized-vs-rest distribution.

        ``student_logp_with_rest`` and ``teacher_logp_with_rest`` must have
        exactly two columns: realized-chain log-probability and rest-bucket
        log-probability. The BCE input and target follow:

            input  = p_student_realized ** (1 / tau)
            target = p_teacher_realized ** (1 / tau)

        The returned tensor has the same trailing width of 2, with the realized
        and rest BCE terms split so existing support/rest masking still applies.
        """
        if tau <= 0.0:
            raise ValueError(f"prefix_bidir_v3_alm_bce_tau must be > 0, got {tau}")
        if (
            student_logp_with_rest.shape[-1] != 2
            or teacher_logp_with_rest.shape[-1] != 2
        ):
            raise ValueError(
                "ALM BCE expects binary realized-vs-rest support; got "
                f"student shape {tuple(student_logp_with_rest.shape)} and "
                f"teacher shape {tuple(teacher_logp_with_rest.shape)}"
            )

        work_dtype = (
            torch.float32
            if student_logp_with_rest.dtype in (torch.float16, torch.bfloat16)
            else student_logp_with_rest.dtype
        )
        eps = max(float(eps), float(torch.finfo(work_dtype).eps))
        exponent = 1.0 / float(tau)
        student_real = (
            student_logp_with_rest[..., 0]
            .to(work_dtype)
            .exp()
            .clamp(min=0.0, max=1.0)
            .pow(exponent)
            .clamp(min=eps, max=1.0 - eps)
        )
        teacher_real = (
            teacher_logp_with_rest[..., 0]
            .to(work_dtype)
            .exp()
            .clamp(min=0.0, max=1.0)
            .pow(exponent)
            .detach()
        )
        realized_term = -teacher_real * student_real.log()
        rest_term = -(1.0 - teacher_real) * torch.log1p(-student_real)
        return torch.stack([realized_term, rest_term], dim=-1)

    @staticmethod
    def _topk_mask_by_score(
        candidate_mask: torch.Tensor,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Keep the row-wise top-k valid candidates by score."""
        k = int(k)
        if k <= 0:
            return torch.zeros_like(candidate_mask, dtype=torch.bool)
        candidate_mask = candidate_mask.to(torch.bool)
        if candidate_mask.shape[-1] <= k:
            return candidate_mask
        masked_scores = torch.where(
            candidate_mask,
            scores,
            torch.full_like(scores, -1.0e30),
        )
        _, keep_idx = torch.topk(masked_scores, k=k, dim=-1)
        keep_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=keep_idx, value=True)
        return keep_mask & candidate_mask

    @staticmethod
    def _unique_bidir_pairs_cpu(
        pairs,
        swap: bool = False,
        assume_unique: bool = False,
    ) -> tuple[list[int], list[int]]:
        """CPU dedupe of ``(s_id, t_id)`` pairs.

        Verbatim port of tokenalign.py:_unique_bidir_pairs_cpu (line 6122).
        When ``assume_unique=True`` we skip the Python set dedupe loop and
        unpack via numpy. ``swap=True`` treats input as ``(t_id, s_id)``
        (used for the reverse prefix-index branch).
        """
        if not pairs:
            return [], []
        if assume_unique:
            arr = np.asarray(pairs, dtype=np.int64)
            if swap:
                return arr[:, 1].tolist(), arr[:, 0].tolist()
            return arr[:, 0].tolist(), arr[:, 1].tolist()
        seen_s: set[int] = set()
        seen_t: set[int] = set()
        s_list: list[int] = []
        t_list: list[int] = []
        for p in pairs:
            if swap:
                t_id = int(p[0])
                s_id = int(p[1])
            else:
                s_id = int(p[0])
                t_id = int(p[1])
            if s_id in seen_s or t_id in seen_t:
                continue
            seen_s.add(s_id)
            seen_t.add(t_id)
            s_list.append(s_id)
            t_list.append(t_id)
        return s_list, t_list

    def _partition_kl_mismatch_batched(
        self,
        chunk_records,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        student_log_z: torch.Tensor,
        teacher_log_z: torch.Tensor,
        temperature: float,
        reverse_kl: bool,
        loss_fn: str = "kl",
        jsd_beta: float = 0.5,
        alm_bce_tau: float = 1.0,
        teacher_sparse: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        teacher_sparse_keep_realized: bool = False,
    ) -> tuple[torch.Tensor, int, int, list[int], list[bool]]:
        """Batched partition-KL over a flat list of mismatch chunk records.

        Verbatim port of tokenalign.py:_partition_kl_mismatch_batched
        (line 6161). Each record is the 11-tuple
        ``(b, s_last, t_last, s_prefix_pred_pos, s_prefix_label_id,
        t_prefix_pred_pos, t_prefix_label_id, s_ids_list, t_ids_list,
        realized_s_last_id, realized_t_last_id)``.
        Returns ``(loss_sum, used_count, matches, support_sizes, valid_flags)``.
        """
        device = student_logits.device
        accum_dtype = (
            torch.float32
            if student_logits.dtype in (torch.float16, torch.bfloat16)
            else student_logits.dtype
        )
        if not chunk_records:
            zero = torch.tensor(0.0, device=device, dtype=accum_dtype)
            return zero, 0, 0, [], []

        K = len(chunk_records)
        Pmax_s = max(len(r[3]) for r in chunk_records)
        Pmax_t = max(len(r[5]) for r in chunk_records)
        Smax = max(len(r[7]) for r in chunk_records)
        # Sentinel that survives kl_div: exp(-1e30) underflows to 0 in fp32 so
        # the masked entry contributes 0 to support-mass and to the per-row KL
        # sum, while target - input stays finite (no 0*nan).
        NEG_INF = -1.0e30

        # Numpy fast path mirrors the source's `_use_np` default-on
        # branch (tokenalign.py:6211-6268). The legacy Python-list branch
        # is preserved for A/B verification via the same env override.
        _use_np = os.environ.get("TOKENALIGN_BATCHED_HELPER_NUMPY", "1") == "1"

        b_idx_list = [r[0] for r in chunk_records]
        s_last_list = [r[1] for r in chunk_records]
        t_last_list = [r[2] for r in chunk_records]
        s_realized_list = [r[9] if len(r) > 9 else -1 for r in chunk_records]
        t_realized_list = [r[10] if len(r) > 10 else -1 for r in chunk_records]
        support_sizes = [len(r[7]) for r in chunk_records]

        np_accum_dtype = np.float32 if accum_dtype == torch.float32 else np.float64

        s_pred_pos = None
        s_label_id = None
        s_pmask = None
        t_pred_pos = None
        t_label_id = None
        t_pmask = None

        if _use_np:
            s_ids_np = np.zeros((K, Smax), dtype=np.int64)
            t_ids_np = np.zeros((K, Smax), dtype=np.int64)
            smask_np = np.zeros((K, Smax), dtype=np_accum_dtype)
            if Pmax_s > 0:
                s_pred_pos_np = np.zeros((K, Pmax_s), dtype=np.int64)
                s_label_id_np = np.zeros((K, Pmax_s), dtype=np.int64)
                s_pmask_np = np.zeros((K, Pmax_s), dtype=np_accum_dtype)
            if Pmax_t > 0:
                t_pred_pos_np = np.zeros((K, Pmax_t), dtype=np.int64)
                t_label_id_np = np.zeros((K, Pmax_t), dtype=np.int64)
                t_pmask_np = np.zeros((K, Pmax_t), dtype=np_accum_dtype)
            for i, r in enumerate(chunk_records):
                s_pp, s_lid = r[3], r[4]
                t_pp, t_lid = r[5], r[6]
                s_list, t_list = r[7], r[8]
                ns_p = len(s_pp)
                if ns_p > 0:
                    s_pred_pos_np[i, :ns_p] = s_pp
                    s_label_id_np[i, :ns_p] = s_lid
                    s_pmask_np[i, :ns_p] = 1.0
                nt_p = len(t_pp)
                if nt_p > 0:
                    t_pred_pos_np[i, :nt_p] = t_pp
                    t_label_id_np[i, :nt_p] = t_lid
                    t_pmask_np[i, :nt_p] = 1.0
                n_sup = len(s_list)
                if n_sup > 0:
                    s_ids_np[i, :n_sup] = s_list
                    t_ids_np[i, :n_sup] = t_list
                    smask_np[i, :n_sup] = 1.0

            b_idx = torch.from_numpy(np.asarray(b_idx_list, dtype=np.int64)).to(
                device, non_blocking=True
            )
            s_last = torch.from_numpy(np.asarray(s_last_list, dtype=np.int64)).to(
                device, non_blocking=True
            )
            t_last = torch.from_numpy(np.asarray(t_last_list, dtype=np.int64)).to(
                device, non_blocking=True
            )
            s_realized_last = torch.from_numpy(
                np.asarray(s_realized_list, dtype=np.int64)
            ).to(device, non_blocking=True)
            t_realized_last = torch.from_numpy(
                np.asarray(t_realized_list, dtype=np.int64)
            ).to(device, non_blocking=True)
            s_ids = torch.from_numpy(s_ids_np).to(device, non_blocking=True)
            t_ids = torch.from_numpy(t_ids_np).to(device, non_blocking=True)
            support_mask = torch.from_numpy(smask_np).to(
                device,
                dtype=accum_dtype,
                non_blocking=True,
            )
            if Pmax_s > 0:
                s_pred_pos = torch.from_numpy(s_pred_pos_np).to(
                    device,
                    non_blocking=True,
                )
                s_label_id = torch.from_numpy(s_label_id_np).to(
                    device,
                    non_blocking=True,
                )
                s_pmask = torch.from_numpy(s_pmask_np).to(
                    device,
                    dtype=accum_dtype,
                    non_blocking=True,
                )
            if Pmax_t > 0:
                t_pred_pos = torch.from_numpy(t_pred_pos_np).to(
                    device,
                    non_blocking=True,
                )
                t_label_id = torch.from_numpy(t_label_id_np).to(
                    device,
                    non_blocking=True,
                )
                t_pmask = torch.from_numpy(t_pmask_np).to(
                    device,
                    dtype=accum_dtype,
                    non_blocking=True,
                )
        else:
            s_pred_pos_buf = [[0] * Pmax_s for _ in range(K)] if Pmax_s > 0 else None
            s_label_id_buf = [[0] * Pmax_s for _ in range(K)] if Pmax_s > 0 else None
            s_pmask_buf = [[0.0] * Pmax_s for _ in range(K)] if Pmax_s > 0 else None
            t_pred_pos_buf = [[0] * Pmax_t for _ in range(K)] if Pmax_t > 0 else None
            t_label_id_buf = [[0] * Pmax_t for _ in range(K)] if Pmax_t > 0 else None
            t_pmask_buf = [[0.0] * Pmax_t for _ in range(K)] if Pmax_t > 0 else None
            s_ids_buf = [[0] * Smax for _ in range(K)]
            t_ids_buf = [[0] * Smax for _ in range(K)]
            smask_buf = [[0.0] * Smax for _ in range(K)]

            for i, r in enumerate(chunk_records):
                s_pp, s_lid = r[3], r[4]
                t_pp, t_lid = r[5], r[6]
                s_list, t_list = r[7], r[8]
                for j, (p, lid) in enumerate(zip(s_pp, s_lid)):
                    s_pred_pos_buf[i][j] = p
                    s_label_id_buf[i][j] = lid
                    s_pmask_buf[i][j] = 1.0
                for j, (p, lid) in enumerate(zip(t_pp, t_lid)):
                    t_pred_pos_buf[i][j] = p
                    t_label_id_buf[i][j] = lid
                    t_pmask_buf[i][j] = 1.0
                n_sup = len(s_list)
                for j in range(n_sup):
                    s_ids_buf[i][j] = s_list[j]
                    t_ids_buf[i][j] = t_list[j]
                    smask_buf[i][j] = 1.0

            b_idx = torch.tensor(b_idx_list, device=device, dtype=torch.long)
            s_last = torch.tensor(
                s_last_list,
                device=device,
                dtype=torch.long,
            )
            t_last = torch.tensor(
                t_last_list,
                device=device,
                dtype=torch.long,
            )
            s_realized_last = torch.tensor(
                s_realized_list,
                device=device,
                dtype=torch.long,
            )
            t_realized_last = torch.tensor(
                t_realized_list,
                device=device,
                dtype=torch.long,
            )
            s_ids = torch.tensor(s_ids_buf, device=device, dtype=torch.long)
            t_ids = torch.tensor(t_ids_buf, device=device, dtype=torch.long)
            support_mask = torch.tensor(
                smask_buf,
                device=device,
                dtype=accum_dtype,
            )
            if Pmax_s > 0:
                s_pred_pos = torch.tensor(
                    s_pred_pos_buf,
                    device=device,
                    dtype=torch.long,
                )
                s_label_id = torch.tensor(
                    s_label_id_buf,
                    device=device,
                    dtype=torch.long,
                )
                s_pmask = torch.tensor(
                    s_pmask_buf,
                    device=device,
                    dtype=accum_dtype,
                )
            if Pmax_t > 0:
                t_pred_pos = torch.tensor(
                    t_pred_pos_buf,
                    device=device,
                    dtype=torch.long,
                )
                t_label_id = torch.tensor(
                    t_label_id_buf,
                    device=device,
                    dtype=torch.long,
                )
                t_pmask = torch.tensor(
                    t_pmask_buf,
                    device=device,
                    dtype=accum_dtype,
                )

        # ---- Student prefix chain (batched gather) ----
        if Pmax_s > 0:
            s_prefix_logits = (
                student_logits[b_idx[:, None], s_pred_pos, s_label_id] / temperature
            )
            s_prefix_scores = (
                s_prefix_logits.to(accum_dtype)
                - student_log_z[b_idx[:, None], s_pred_pos]
            )
            s_prefix_logp = (s_prefix_scores * s_pmask).sum(dim=-1)  # (K,)
        else:
            s_prefix_logp = torch.zeros(
                (K,),
                device=device,
                dtype=accum_dtype,
            )

        # ---- Teacher prefix chain (batched gather; detached) ----
        prefix_found = torch.ones((K,), device=device, dtype=torch.bool)
        if Pmax_t > 0:
            if teacher_sparse is not None:
                teacher_topk_logits, teacher_topk_indices = teacher_sparse
                t_prefix_scores, t_prefix_found = self._lookup_sparse_teacher_logp(
                    teacher_topk_logits,
                    teacher_topk_indices,
                    teacher_log_z,
                    b_idx,
                    t_pred_pos,
                    t_label_id,
                    temperature,
                )
                t_prefix_scores = torch.where(
                    t_prefix_found,
                    t_prefix_scores.to(accum_dtype),
                    torch.zeros_like(t_prefix_scores, dtype=accum_dtype),
                )
                prefix_found = (t_prefix_found | (t_pmask <= 0)).all(dim=-1)
            else:
                assert teacher_logits is not None
                t_prefix_logits = (
                    teacher_logits[b_idx[:, None], t_pred_pos, t_label_id] / temperature
                )
                t_prefix_scores = (
                    t_prefix_logits.to(accum_dtype)
                    - teacher_log_z[b_idx[:, None], t_pred_pos]
                )
            t_prefix_logp = (t_prefix_scores * t_pmask).sum(dim=-1)
        else:
            t_prefix_logp = torch.zeros(
                (K,),
                device=device,
                dtype=accum_dtype,
            )

        # ---- Final-position support scores (batched gather) ----
        s_final_logits = (
            student_logits[b_idx[:, None], s_last[:, None], s_ids] / temperature
        )
        s_final_scores = s_final_logits.to(accum_dtype) - student_log_z[
            b_idx, s_last
        ].unsqueeze(-1)
        s_full_scores = s_final_scores + s_prefix_logp.unsqueeze(-1)
        s_full_scores = torch.where(
            support_mask > 0,
            s_full_scores,
            torch.full_like(s_full_scores, NEG_INF),
        )

        with torch.no_grad():
            if teacher_sparse is not None:
                teacher_topk_logits, teacher_topk_indices = teacher_sparse
                t_last_pos = t_last.unsqueeze(-1).expand_as(t_ids)
                t_final_scores, t_final_found = self._lookup_sparse_teacher_logp(
                    teacher_topk_logits,
                    teacher_topk_indices,
                    teacher_log_z,
                    b_idx,
                    t_last_pos,
                    t_ids,
                    temperature,
                )
                if teacher_sparse_keep_realized:
                    topk_width = int(teacher_topk_indices.shape[-1])
                    alt_budget = max(topk_width - 1, 0)
                    realized_pair_mask = (
                        (support_mask > 0)
                        & (s_ids == s_realized_last[:, None])
                        & (t_ids == t_realized_last[:, None])
                    )
                    alt_candidate_mask = (
                        (support_mask > 0) & t_final_found & (~realized_pair_mask)
                    )
                    alt_keep_mask = self._topk_mask_by_score(
                        alt_candidate_mask,
                        t_final_scores.to(accum_dtype),
                        alt_budget,
                    )
                    final_keep_mask = (
                        realized_pair_mask & t_final_found
                    ) | alt_keep_mask
                else:
                    final_keep_mask = (support_mask > 0) & t_final_found
                teacher_support_mask = final_keep_mask & prefix_found.unsqueeze(-1)
                t_full_scores = t_final_scores.to(
                    accum_dtype
                ) + t_prefix_logp.unsqueeze(-1)
                support_mask = support_mask * teacher_support_mask.to(accum_dtype)
            else:
                assert teacher_logits is not None
                t_final_logits = (
                    teacher_logits[b_idx[:, None], t_last[:, None], t_ids] / temperature
                )
                t_final_scores = t_final_logits.to(accum_dtype) - teacher_log_z[
                    b_idx, t_last
                ].unsqueeze(-1)
                t_full_scores = t_final_scores + t_prefix_logp.unsqueeze(-1)
            t_full_scores = torch.where(
                support_mask > 0,
                t_full_scores,
                torch.full_like(t_full_scores, NEG_INF),
            )

        # Teacher sparse support pruning is computed without gradients, but the
        # student scores must be masked with autograd enabled.
        s_full_scores = torch.where(
            support_mask > 0,
            s_full_scores,
            torch.full_like(s_full_scores, NEG_INF),
        )

        # ---- Append REST bucket (per-row valid-mass complement) ----
        s_logp_with_rest = self._append_rest_bucket_logp(s_full_scores)
        t_logp_with_rest = self._append_rest_bucket_logp(t_full_scores)

        # Validity: row contributes if it has at least 1 support position.
        valid_row = support_mask.sum(dim=-1) >= 1.0  # (K,) bool

        # ---- Batched partition loss ----
        if loss_fn == "bce":
            kl_per_pos = self._binary_power_bce_from_logp(
                s_logp_with_rest,
                t_logp_with_rest,
                tau=alm_bce_tau,
            )
        elif loss_fn == "jsd":
            kl_per_pos = _generalized_jsd(
                s_logp_with_rest,
                t_logp_with_rest,
                jsd_beta,
            )
        elif not reverse_kl:
            kl_per_pos = torch.nn.functional.kl_div(
                s_logp_with_rest,
                t_logp_with_rest,
                reduction="none",
                log_target=True,
            )
        else:
            kl_per_pos = torch.nn.functional.kl_div(
                t_logp_with_rest,
                s_logp_with_rest,
                reduction="none",
                log_target=True,
            )
        # Mask padded support columns to zero (REST column stays).
        rest_one = torch.ones((K, 1), device=device, dtype=accum_dtype)
        full_mask = torch.cat([support_mask, rest_one], dim=-1)
        per_row_kl = (kl_per_pos * full_mask).sum(dim=-1)  # (K,)
        per_row_kl = per_row_kl * valid_row.to(per_row_kl.dtype)
        loss_sum = per_row_kl.sum()

        with torch.no_grad():
            s_arg = s_full_scores.argmax(dim=-1)
            t_arg = t_full_scores.argmax(dim=-1)
            match_row = (s_arg == t_arg) & valid_row

        stats_cpu = torch.stack([valid_row, match_row], dim=0).detach().cpu()
        valid_flags = [bool(x) for x in stats_cpu[0].tolist()]
        matches = int(stats_cpu[1].sum().item())
        used_count = int(sum(valid_flags))
        effective_support_cpu = support_mask.sum(dim=-1).detach().cpu().tolist()
        support_sizes_used = [
            int(size)
            for size, is_valid in zip(effective_support_cpu, valid_flags)
            if is_valid
        ]

        return loss_sum, used_count, matches, support_sizes_used, valid_flags

    def _compute_prefix_bidir_partition_kl_v3(
        self,
        i: int,
        student_logits: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        *,
        teacher_vocab_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        global_valid_chunks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """v6 (prefix_bidir_partition_kl_v3) KD term for cross-tokenizer teacher ``i``.

        Prefix-bidir partition KL with MtoM-as-ALM + rest bucket and optional
        position-0 partition KL. ``mtom_as_alm``/``rest_bucket`` are baked in
        for the v6 preset; the v1/v2/v5 surface-match path is not ported.

        Per-teacher contract (mirrors ``_compute_same_vocab_kl``):
        ``student_logits`` is this rank's CP-contiguous window,
        ``teacher_full_logits`` is the reassembled full-vocab teacher window,
        and ``align`` carries the localized student/teacher input ids, per-chunk
        global position spans, ``pair_valid`` and ``num_chunks``. The body is
        native TP/CP-correct: ``student_logits`` is vocab-gathered across
        ``tp_group`` and sequence-gathered across ``cp_group`` (teacher likewise
        on the sequence) so the arbitrary vocab-column / global-span indexing is
        materialized full; ``global_valid_chunks`` is WORLD-reduced in-loss. All
        gathers are no-ops at world size 1 (single-GPU byte-exact).

        Returns ``(loss, metrics_dict)``.
        """
        cfg = self.cfg
        device = student_logits.device
        # ----- Native TP/CP gather (Phase 2) -----------------------------
        # v6 indexes arbitrary vocab columns (common/support token ids) and
        # arbitrary sequence positions (global chunk spans), and takes a full-
        # vocab logsumexp per position. Under TP the vocab is sharded and under
        # CP the sequence is windowed, so both must be materialized full before
        # any indexing. CP is a contiguous sequence all-gather (the student is
        # relaid to a contiguous window and the teacher is contiguous-block
        # sharded); TP is a raw-logit vocab all-gather. The axes are orthogonal
        # so they compose. Both are no-ops at world size 1 -> the single-GPU
        # path (and CPU parity) is byte-exact. Spans / input_ids arrive global
        # from prepare_xtoken_cross_tokenizer_loss_input.
        student_logits = allgather_cp_contiguous_tensor(student_logits, cp_group)
        student_logits = vocab_parallel_gather_logits(student_logits, tp_group=tp_group)
        with torch.no_grad():
            teacher_full_logits = allgather_cp_contiguous_tensor(
                teacher_full_logits, cp_group
            )
        # B is the per-sample loop bound; S is the student-side
        # predictor-position bound used in the classify-loop range check.
        # V_s (student vocab) is implicitly bounded by the logits gather
        # indices (common_student_idx_t) so we don't bind it here.
        B, S, _ = student_logits.shape

        # v6 preset knobs (with safe defaults matching tokenalign args
        # defaults at line 6420-6434):
        kl_chunk_shift = bool(cfg.get("kl_chunk_shift", False))
        temperature = float(cfg["temperature"])
        reverse_kl = bool(cfg.get("reverse_kl", False))
        v3_position_0_kl = bool(cfg.get("prefix_bidir_v3_position_0_kl", False))
        v3_loss_fn = str(cfg.get("prefix_bidir_v3_loss_fn", "kl") or "kl")
        v3_jsd_beta = float(cfg.get("prefix_bidir_v3_jsd_beta", 0.5))
        v3_alm_bce_tau = float(cfg.get("prefix_bidir_v3_alm_bce_tau", 1.0))
        v3_last_pos_loss_fn_cfg = cfg.get("prefix_bidir_v3_last_pos_loss_fn")
        pure_alm = bool(cfg.get("prefix_bidir_v3_pure_alm", False))
        if pure_alm:
            # Position-0 is a common-vocab auxiliary partition term. Pure ALM
            # should only compare realized chain probability vs rest bucket.
            v3_position_0_kl = False
        # v6 hybrid: None means "inherit v3_loss_fn"; "kl"/"jsd" overrides
        # only the position-N-1 (last-position) partition KL.
        effective_last_pos_loss_fn = (
            v3_last_pos_loss_fn_cfg
            if v3_last_pos_loss_fn_cfg is not None
            else v3_loss_fn
        )
        mismatch_pos0_alpha_raw = cfg.get(
            "prefix_bidir_v3_mismatch_pos0_alpha",
        )
        mismatch_loss_beta_raw = cfg.get(
            "prefix_bidir_v3_mismatch_loss_beta",
        )
        mismatch_pos0_weight_raw = cfg.get(
            "prefix_bidir_v3_mismatch_pos0_weight",
        )
        mismatch_loss_scale_raw = cfg.get(
            "prefix_bidir_v3_mismatch_loss_scale",
        )
        uses_additive_coefficients = (
            mismatch_pos0_alpha_raw is not None or mismatch_loss_beta_raw is not None
        )
        if uses_additive_coefficients and (
            mismatch_pos0_weight_raw is not None or mismatch_loss_scale_raw is not None
        ):
            raise ValueError(
                "New mismatch alpha/beta keys cannot be combined with the "
                "deprecated mismatch weight/scale keys."
            )

        mismatch_pos0_alpha: Optional[float] = None
        mismatch_loss_beta: Optional[float] = None
        mismatch_pos0_weight: Optional[float] = None
        mismatch_loss_scale = 1.0
        requires_pos0_support = False
        if uses_additive_coefficients:
            mismatch_pos0_alpha = float(
                1.0 if mismatch_pos0_alpha_raw is None else mismatch_pos0_alpha_raw
            )
            mismatch_loss_beta = float(
                1.0 if mismatch_loss_beta_raw is None else mismatch_loss_beta_raw
            )
            if mismatch_pos0_alpha < 0.0:
                raise ValueError(
                    "prefix_bidir_v3_mismatch_pos0_alpha must be >= 0, "
                    f"got {mismatch_pos0_alpha}"
                )
            if mismatch_loss_beta < 0.0:
                raise ValueError(
                    "prefix_bidir_v3_mismatch_loss_beta must be >= 0, "
                    f"got {mismatch_loss_beta}"
                )
            requires_pos0_support = mismatch_pos0_alpha > 0.0
            mismatch_pos0_coefficient = mismatch_pos0_alpha
            mismatch_loss_multiplier = mismatch_loss_beta
            mismatch_combination_is_convex = False
        else:
            if mismatch_pos0_weight_raw is not None:
                mismatch_pos0_weight = float(mismatch_pos0_weight_raw)
                if not 0.0 <= mismatch_pos0_weight <= 1.0:
                    raise ValueError(
                        "prefix_bidir_v3_mismatch_pos0_weight must be in "
                        f"[0, 1], got {mismatch_pos0_weight}"
                    )
                requires_pos0_support = mismatch_pos0_weight > 0.0
            mismatch_loss_scale = float(
                1.0 if mismatch_loss_scale_raw is None else mismatch_loss_scale_raw
            )
            if mismatch_loss_scale < 0.0:
                raise ValueError(
                    "prefix_bidir_v3_mismatch_loss_scale must be >= 0, "
                    f"got {mismatch_loss_scale}"
                )
            mismatch_pos0_coefficient = (
                1.0 if mismatch_pos0_weight is None else mismatch_pos0_weight
            )
            mismatch_loss_multiplier = mismatch_loss_scale
            mismatch_combination_is_convex = mismatch_pos0_weight is not None
        if requires_pos0_support and not v3_position_0_kl:
            coefficient_name = (
                "prefix_bidir_v3_mismatch_pos0_alpha"
                if uses_additive_coefficients
                else "prefix_bidir_v3_mismatch_pos0_weight"
            )
            raise ValueError(
                f"{coefficient_name} > 0 requires prefix_bidir_v3_position_0_kl=true."
            )
        noise_filter_topk = int(cfg.get("prefix_bidir_v3_noise_filter_topk", 0) or 0)
        if noise_filter_topk < 0:
            raise ValueError(
                "prefix_bidir_v3_noise_filter_topk must be >= 0, "
                f"got {noise_filter_topk}"
            )
        teacher_topk_ipc_k = int(cfg.get("teacher_topk_ipc_k", 0) or 0)
        teacher_topk_keep_realized = teacher_topk_ipc_k > 0 and (
            pure_alm
            or teacher_topk_ipc_k == 1
            or bool(cfg.get("teacher_topk_ipc_keep_realized", True))
        )
        # mtom_as_alm=True and rest_bucket=True are baked in for v6 per
        # the porting spec (Section 8 hard constraints 7 + 8).
        mtom_as_alm = True
        rest_bucket = True

        v_t = int(teacher_vocab_size)
        # RL uses always-full teacher transport: ``teacher_full_logits`` is the
        # reassembled full-vocab teacher logits (rebuilt from IPC in
        # prepare_xtoken_cross_tokenizer_loss_input). The sparse-IPC top-k path
        # from the upstream port is left unwired — ``teacher_sparse_payload``
        # stays None and its downstream branches are inert.
        teacher_sparse_payload: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = None
        teacher_gt_in_topk: Optional[torch.Tensor] = None
        # Slice teacher logits to the real tokenizer vocab (drop any padded
        # lm_head columns beyond len(tokenizer)) BEFORE indexing.
        if teacher_full_logits.shape[-1] > v_t:
            teacher_full_logits = teacher_full_logits[..., :v_t]
        teacher_logits: Optional[torch.Tensor] = teacher_full_logits
        # V_t is implicit (common_teacher_idx_t bounds it); bind T (teacher-side
        # predictor bound) and B2 (sanity check).
        B2, T, _ = teacher_logits.shape
        assert B == B2, f"batch size mismatch: student={B} teacher={B2}"

        # ----- Lazy caches (per-device) ----------------------------------
        if pure_alm:
            common_student_idx_t = torch.empty(
                (0,),
                device=device,
                dtype=torch.long,
            )
            common_teacher_idx_t = torch.empty(
                (0,),
                device=device,
                dtype=torch.long,
            )
            has_common_support = True
        else:
            common_student_idx_t, common_teacher_idx_t = self._get_common_indices_v3(
                device,
                i,
            )
            has_common_support = common_student_idx_t.numel() > 0
        if requires_pos0_support and not has_common_support:
            coefficient_name = (
                "prefix_bidir_v3_mismatch_pos0_alpha"
                if uses_additive_coefficients
                else "prefix_bidir_v3_mismatch_pos0_weight"
            )
            raise ValueError(
                f"{coefficient_name} > 0 requires non-empty common-vocab support."
            )

        # ----- Single-shot host pull of input_ids ------------------------
        # One .detach().cpu().numpy() per tensor, then per-chunk numpy indexing
        # inside the classify/collect loop. The ids ride on the localized
        # alignment as this CP rank's contiguous window (both unshifted; the
        # classify loop applies its own per-chunk kl_chunk_shift); they are
        # CP-gathered to the full sequence to match the gathered logits and the
        # global spans (no-op at CP=1).
        input_ids_student = to_local_if_dtensor(align.student_input_ids)
        input_ids_teacher = to_local_if_dtensor(align.teacher_input_ids)
        with torch.no_grad():
            input_ids_student = allgather_cp_contiguous_tensor(
                input_ids_student, cp_group
            )
            input_ids_teacher = allgather_cp_contiguous_tensor(
                input_ids_teacher, cp_group
            )
        inp_s_np = input_ids_student.detach().cpu().numpy()
        inp_t_np = input_ids_teacher.detach().cpu().numpy()

        teacher_gt_in_topk_np: Optional[np.ndarray] = None
        teacher_filter_topk_indices_np: Optional[np.ndarray] = None
        if noise_filter_topk > 0:
            if teacher_sparse_payload is not None:
                if teacher_gt_in_topk is None:
                    raise ValueError(
                        "prefix_bidir_v3_noise_filter_topk requires sparse "
                        "teacher IPC handles with gt_in_topk metadata. Ensure "
                        "the teacher IPC path receives force_include_token_ids."
                    )
                teacher_gt_in_topk_np = (
                    teacher_gt_in_topk.detach().cpu().numpy().astype(bool)
                )
            else:
                assert teacher_logits is not None
                filter_k = min(noise_filter_topk, int(teacher_logits.shape[-1]))
                if filter_k > 0:
                    with torch.no_grad():
                        teacher_filter_topk_indices_np = (
                            torch.topk(
                                teacher_logits,
                                k=filter_k,
                                dim=-1,
                            )
                            .indices.detach()
                            .cpu()
                            .numpy()
                        )

        def _teacher_targets_pass_noise_filter(
            b_idx: int,
            pred_positions: list[int],
            label_positions: list[int],
        ) -> bool:
            if noise_filter_topk <= 0:
                return True
            if len(pred_positions) != len(label_positions):
                return False
            if teacher_gt_in_topk_np is not None:
                for pred_pos in pred_positions:
                    if (
                        pred_pos < 0
                        or pred_pos >= teacher_gt_in_topk_np.shape[1]
                        or not bool(teacher_gt_in_topk_np[b_idx, pred_pos])
                    ):
                        return False
                return True
            if teacher_filter_topk_indices_np is None:
                return False
            for pred_pos, label_pos in zip(pred_positions, label_positions):
                if pred_pos < 0 or pred_pos >= teacher_filter_topk_indices_np.shape[1]:
                    return False
                if label_pos < 0 or label_pos >= inp_t_np.shape[1]:
                    return False
                label_id = int(inp_t_np[b_idx, label_pos])
                if not bool(
                    np.any(teacher_filter_topk_indices_np[b_idx, pred_pos] == label_id)
                ):
                    return False
            return True

        # ----- Alignment payload (padded tensors -> per-chunk records) ---
        # The collator ships padded per-chunk position spans + a validity mask
        # (dense over max_pairs); a sentinel / OOB pair keeps its slot and is
        # zeroed by ``pair_valid``. Spans / pair_valid / num_chunks ride on the
        # localized alignment.
        s_spans = to_local_if_dtensor(align.student_spans)  # [B, max_pairs, 2]
        t_spans = to_local_if_dtensor(align.teacher_spans)  # [B, max_pairs, 2]
        pair_valid = to_local_if_dtensor(align.pair_valid)  # [B, max_pairs]
        num_chunks = to_local_if_dtensor(align.num_chunks)  # [B]
        # Host-side copies for the classification loop (no GPU sync per
        # chunk).
        s_spans_np = s_spans.detach().cpu().numpy()
        t_spans_np = t_spans.detach().cpu().numpy()
        pair_valid_np = pair_valid.detach().cpu().numpy().astype(bool)
        num_chunks_np = num_chunks.detach().cpu().numpy().astype(np.int64)

        # ----- Classify chunks (tokenalign.py:6590-6627) ----------------
        common_chunks: list[tuple[int, int, int, int, int]] = []
        mismatch_chunks: list[
            tuple[int, int, int, list[int], list[int], list[int], list[int]]
        ] = []
        noise_filtered_common_chunks = 0
        noise_filtered_mismatch_chunks = 0
        T_s_label = input_ids_student.shape[1]
        T_t_label = input_ids_teacher.shape[1]
        for b in range(B):
            n = int(num_chunks_np[b])
            for k in range(n):
                if not bool(pair_valid_np[b, k]):
                    # tokenalign's `start1 == -1` skip (tokenalign.py:6597)
                    # is replaced by pair_valid check per the Adapter shim.
                    continue
                s_start = int(s_spans_np[b, k, 0])
                s_end = int(s_spans_np[b, k, 1])
                t_start = int(t_spans_np[b, k, 0])
                t_end = int(t_spans_np[b, k, 1])
                # tokenalign.py:6597 also skips when explicit -1 sentinels
                # appear. pair_valid handles padding, but guard anyway.
                if s_start == -1 or t_start == -1:
                    continue
                M = s_end - s_start
                N = t_end - t_start
                if M <= 0 or N <= 0:
                    continue
                # NOTE: preserve per-chunk kl_chunk_shift (tokenalign.py:6603).
                # Do NOT reuse nemo_rl's tensor-level global shift —
                # v3 needs the conditional per-chunk shift (chunks at
                # sequence-start are NOT shifted).
                if kl_chunk_shift and s_start > 0 and t_start > 0:
                    s_pred = list(range(s_start - 1, s_end - 1))
                    t_pred = list(range(t_start - 1, t_end - 1))
                    s_labels = list(range(s_start, s_end))
                    t_labels = list(range(t_start, t_end))
                else:
                    s_pred = list(range(s_start, s_end))
                    t_pred = list(range(t_start, t_end))
                    s_labels = s_pred
                    t_labels = t_pred

                if any(pos < 0 or pos >= S for pos in s_pred):
                    continue
                if any(pos < 0 or pos >= T for pos in t_pred):
                    continue
                if any(pos < 0 or pos >= T_s_label for pos in s_labels):
                    continue
                if any(pos < 0 or pos >= T_t_label for pos in t_labels):
                    continue

                if M == 1 and N == 1 and (has_common_support or pure_alm):
                    if not _teacher_targets_pass_noise_filter(
                        b,
                        t_pred,
                        t_labels,
                    ):
                        noise_filtered_common_chunks += 1
                        continue
                    common_chunks.append(
                        (
                            b,
                            s_pred[0],
                            t_pred[0],
                            int(inp_s_np[b, s_labels[0]]),
                            int(inp_t_np[b, t_labels[0]]),
                        )
                    )
                else:
                    if not _teacher_targets_pass_noise_filter(
                        b,
                        t_pred,
                        t_labels,
                    ):
                        noise_filtered_mismatch_chunks += 1
                        continue
                    mismatch_chunks.append(
                        (b, M, N, s_pred, t_pred, s_labels, t_labels)
                    )

        # ----- No early-out on empty chunks -----------------------------
        # A microbatch with no valid chunks (empty common + mismatch) is NOT
        # returned early: it must still reach the ``global_valid_chunks``
        # WORLD-reduce below so a rank with zero local chunks participates in
        # the collective (other DP/CP ranks may have chunks). The empty
        # common/mismatch loops leave both partition losses at zero and the
        # combine step returns a gradient-connected zero.
        accum_dtype = (
            torch.float32
            if student_logits.dtype in (torch.float16, torch.bfloat16)
            else student_logits.dtype
        )
        common_loss = torch.tensor(0.0, device=device, dtype=accum_dtype)
        mismatch_loss = torch.tensor(0.0, device=device, dtype=accum_dtype)
        common_matches = 0
        mismatch_matches = 0

        # ----- Pre-compute log_Z's per (b, predictor-position) ----------
        # Partition-KL (rest_bucket=True) needs log Z[b, s] for every
        # (b, predictor) position the chunk loops reference. Chunked along
        # the seq axis to bound peak fp32 temporary (tokenalign.py:6647-6661).
        def _precompute_log_z(_logits: torch.Tensor, _chunk: int = 256):
            _B, _S, _V = _logits.shape
            _out = torch.empty(
                (_B, _S),
                dtype=accum_dtype,
                device=_logits.device,
            )
            for _s in range(0, _S, _chunk):
                _end = min(_s + _chunk, _S)
                _chunk_t = _logits[:, _s:_end, :].to(accum_dtype) / temperature
                _out[:, _s:_end] = torch.logsumexp(_chunk_t, dim=-1)
            return _out

        student_log_z = _precompute_log_z(student_logits)
        if teacher_sparse_payload is None:
            assert teacher_logits is not None
            with torch.no_grad():
                teacher_log_z = _precompute_log_z(teacher_logits)

        # ----- Common-vocab KL (tokenalign.py:6663-6743) ----------------
        if common_chunks:
            sb_full = torch.tensor(
                [c[0] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            sp_full = torch.tensor(
                [c[1] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            tp_full = torch.tensor(
                [c[2] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            rs_full = torch.tensor(
                [c[3] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            rt_full = torch.tensor(
                [c[4] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )

            common_loss_sum = torch.tensor(
                0.0,
                device=device,
                dtype=accum_dtype,
            )
            common_matches_sum_t = torch.zeros(
                (),
                device=device,
                dtype=torch.long,
            )

            _env_common_mb = os.environ.get("TOKENALIGN_COMMON_LOOP_MB")
            common_mb = int(_env_common_mb) if _env_common_mb else len(common_chunks)
            if teacher_sparse_payload is not None and not _env_common_mb:
                common_mb = min(common_mb, 64)
            teacher_to_common_student = None
            if teacher_sparse_payload is not None and not pure_alm:
                teacher_to_common_student = self._get_v3_teacher_to_common_student(
                    device,
                    i,
                    v_t,
                    common_student_idx_t,
                    common_teacher_idx_t,
                )

            for mb_start in range(0, len(common_chunks), common_mb):
                mb_end = min(mb_start + common_mb, len(common_chunks))
                sb = sb_full[mb_start:mb_end]
                sp = sp_full[mb_start:mb_end]
                tp = tp_full[mb_start:mb_end]
                rs = rs_full[mb_start:mb_end]
                rt = rt_full[mb_start:mb_end]
                support_mask = None
                common_valid_row = None
                if pure_alm:
                    student_common_logp = (
                        student_logits[sb, sp, rs].to(accum_dtype) / temperature
                        - student_log_z[sb, sp]
                    ).unsqueeze(-1)
                    if teacher_sparse_payload is None:
                        assert teacher_logits is not None
                        teacher_common_logp = (
                            teacher_logits[sb, tp, rt].to(accum_dtype) / temperature
                            - teacher_log_z[sb, tp]
                        ).unsqueeze(-1)
                        common_valid_row = torch.ones(
                            (sb.shape[0],),
                            device=device,
                            dtype=torch.bool,
                        )
                    else:
                        teacher_topk_logits, teacher_topk_indices, _, _ = (
                            teacher_sparse_payload
                        )
                        t_real_logp, t_real_found = self._lookup_sparse_teacher_logp(
                            teacher_topk_logits,
                            teacher_topk_indices,
                            teacher_log_z,
                            sb,
                            tp,
                            rt,
                            temperature,
                        )
                        teacher_common_logp = t_real_logp.to(accum_dtype).unsqueeze(-1)
                        common_valid_row = t_real_found
                    support_mask = common_valid_row.unsqueeze(-1)
                    neg_inf = torch.full_like(student_common_logp, -1.0e30)
                    student_common_logp = torch.where(
                        support_mask,
                        student_common_logp,
                        neg_inf,
                    )
                    teacher_common_logp = torch.where(
                        support_mask,
                        teacher_common_logp,
                        neg_inf,
                    )
                elif teacher_sparse_payload is None:
                    assert teacher_logits is not None
                    student_common_logits = (
                        student_logits[
                            sb[:, None],
                            sp[:, None],
                            common_student_idx_t[None, :],
                        ]
                        / temperature
                    )
                    teacher_common_logits = (
                        teacher_logits[
                            sb[:, None],
                            tp[:, None],
                            common_teacher_idx_t[None, :],
                        ]
                        / temperature
                    )
                    # rest_bucket=True path (the v3/v6 default — baked in).
                    s_log_z_mb = student_log_z[sb, sp].unsqueeze(-1)
                    t_log_z_mb = teacher_log_z[sb, tp].unsqueeze(-1)
                    student_common_logp = (
                        student_common_logits.to(accum_dtype) - s_log_z_mb
                    )
                    teacher_common_logp = (
                        teacher_common_logits.to(accum_dtype) - t_log_z_mb
                    )
                else:
                    assert teacher_to_common_student is not None
                    teacher_topk_logits, teacher_topk_indices, _, _ = (
                        teacher_sparse_payload
                    )
                    topk_ids = teacher_topk_indices[sb, tp]
                    topk_vals = teacher_topk_logits[sb, tp]
                    mapped_student = teacher_to_common_student[topk_ids.long()]
                    support_mask = mapped_student >= 0
                    if teacher_topk_keep_realized:
                        topk_width = int(topk_ids.shape[-1])
                        alt_budget = max(topk_width - 1, 0)
                        support_mask = (
                            support_mask
                            & (mapped_student != rs[:, None])
                            & (topk_ids.long() != rt[:, None])
                        )
                        support_mask = self._topk_mask_by_score(
                            support_mask,
                            topk_vals.to(accum_dtype),
                            alt_budget,
                        )
                    safe_student_ids = mapped_student.clamp_min(0)
                    student_common_logits = (
                        student_logits[sb[:, None], sp[:, None], safe_student_ids]
                        / temperature
                    )
                    student_common_logp = student_common_logits.to(
                        accum_dtype
                    ) - student_log_z[sb, sp].unsqueeze(-1)
                    teacher_common_logp = topk_vals.to(
                        accum_dtype
                    ) / temperature - teacher_log_z[sb, tp].unsqueeze(-1)
                    neg_inf = torch.full_like(student_common_logp, -1.0e30)
                    student_common_logp = torch.where(
                        support_mask,
                        student_common_logp,
                        neg_inf,
                    )
                    teacher_common_logp = torch.where(
                        support_mask,
                        teacher_common_logp,
                        neg_inf,
                    )
                    if teacher_topk_keep_realized:
                        t_real_logp, t_real_found = self._lookup_sparse_teacher_logp(
                            teacher_topk_logits,
                            teacher_topk_indices,
                            teacher_log_z,
                            sb,
                            tp,
                            rt,
                            temperature,
                        )
                        realized_common_student = teacher_to_common_student[rt.long()]
                        realized_common_mask = t_real_found & (
                            realized_common_student == rs
                        )
                        s_real_logp = (
                            student_logits[sb, sp, rs].to(accum_dtype) / temperature
                            - student_log_z[sb, sp]
                        )
                        student_common_logp = torch.cat(
                            [
                                student_common_logp,
                                torch.where(
                                    realized_common_mask,
                                    s_real_logp,
                                    torch.full_like(s_real_logp, -1.0e30),
                                ).unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        teacher_common_logp = torch.cat(
                            [
                                teacher_common_logp,
                                torch.where(
                                    realized_common_mask,
                                    t_real_logp.to(accum_dtype),
                                    torch.full_like(t_real_logp, -1.0e30),
                                ).unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        support_mask = torch.cat(
                            [
                                support_mask,
                                realized_common_mask.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                    common_valid_row = support_mask.any(dim=-1)
                student_loss_logp = self._append_rest_bucket_logp(
                    student_common_logp,
                )
                teacher_loss_logp = self._append_rest_bucket_logp(
                    teacher_common_logp,
                )
                if v3_loss_fn == "bce":
                    per_elem = self._binary_power_bce_from_logp(
                        student_loss_logp,
                        teacher_loss_logp,
                        tau=v3_alm_bce_tau,
                    )
                elif v3_loss_fn == "jsd":
                    per_elem = _generalized_jsd(
                        student_loss_logp,
                        teacher_loss_logp,
                        v3_jsd_beta,
                    )
                elif not reverse_kl:
                    per_elem = torch.nn.functional.kl_div(
                        student_loss_logp,
                        teacher_loss_logp,
                        reduction="none",
                        log_target=True,
                    )
                else:
                    per_elem = torch.nn.functional.kl_div(
                        teacher_loss_logp,
                        student_loss_logp,
                        reduction="none",
                        log_target=True,
                    )
                if support_mask is not None:
                    rest_one = torch.ones(
                        (support_mask.shape[0], 1),
                        device=device,
                        dtype=per_elem.dtype,
                    )
                    full_mask = torch.cat(
                        [support_mask.to(per_elem.dtype), rest_one],
                        dim=-1,
                    )
                    per_row = (per_elem * full_mask).sum(dim=-1)
                    per_row = per_row * common_valid_row.to(per_row.dtype)
                    common_loss_sum = common_loss_sum + per_row.sum().to(
                        accum_dtype,
                    )
                else:
                    common_loss_sum = common_loss_sum + per_elem.sum().to(
                        accum_dtype,
                    )
                with torch.no_grad():
                    row_match = student_common_logp.argmax(
                        dim=-1
                    ) == teacher_common_logp.argmax(dim=-1)
                    if common_valid_row is not None:
                        row_match = row_match & common_valid_row
                    common_matches_sum_t = common_matches_sum_t + row_match.sum().to(
                        torch.long
                    )
            common_loss = common_loss_sum / float(len(common_chunks))
            common_matches = int(common_matches_sum_t.item())

        # ----- Mismatch loss (tokenalign.py:6745-7127) ------------------
        skipped_no_support = 0
        support_sizes: list[int] = []
        v3_pos0_records: list[tuple[int, int, int, int, int, frozenset[int]]] = []
        prefix_index = None
        if mismatch_chunks:
            if pure_alm:
                _prefix_index_prededuped = False
                max_chain_t = 0
                max_chain_s = 0
            else:
                prefix_index = self._ensure_bidir_prefix_support_index(device, i)
                _prefix_index_prededuped = bool(prefix_index.get("_prededuped", False))
                max_chain_t = int(prefix_index.get("max_chain_t", 0))
                max_chain_s = int(prefix_index.get("max_chain_s", 0))

            chunk_records: list[
                tuple[
                    int,
                    int,
                    int,
                    list[int],
                    list[int],
                    list[int],
                    list[int],
                    list[int],
                    list[int],
                    int,
                    int,
                ]
            ] = []
            for b, M, N, s_pred, t_pred, s_labels, t_labels in mismatch_chunks:
                if pure_alm:
                    s_realized_id = int(inp_s_np[b, s_labels[-1]])
                    t_realized_id = int(inp_t_np[b, t_labels[-1]])
                    s_prefix_pred = list(s_pred[:-1])
                    s_prefix_lab_ids = [int(inp_s_np[b, pos]) for pos in s_labels[:-1]]
                    t_prefix_pred = list(t_pred[:-1])
                    t_prefix_lab_ids = [int(inp_t_np[b, pos]) for pos in t_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[-1]),
                            int(t_pred[-1]),
                            s_prefix_pred,
                            s_prefix_lab_ids,
                            t_prefix_pred,
                            t_prefix_lab_ids,
                            [s_realized_id],
                            [t_realized_id],
                            s_realized_id,
                            t_realized_id,
                        )
                    )
                elif M == 1 and N > 1:
                    # 1-to-many (tokenalign.py:6792-6832)
                    s_realized_id = int(inp_s_np[b, s_labels[0]])
                    t_realized_id = int(inp_t_np[b, t_labels[-1]])
                    if N > max_chain_t and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    pairs = ()
                    if N <= max_chain_t:
                        teacher_prefix = tuple(
                            int(inp_t_np[b, pos]) for pos in t_labels[: N - 1]
                        )
                        pairs = prefix_index["forward"].get(
                            (N, teacher_prefix),
                            (),
                        )
                    if not pairs and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    s_list, t_list = self._unique_bidir_pairs_cpu(
                        pairs,
                        swap=False,
                        assume_unique=_prefix_index_prededuped,
                    )
                    if teacher_topk_keep_realized and (
                        s_realized_id,
                        t_realized_id,
                    ) not in set(zip(s_list, t_list)):
                        s_list.append(s_realized_id)
                        t_list.append(t_realized_id)
                    min_support = 1 if teacher_topk_keep_realized else 2
                    if len(s_list) < min_support:
                        skipped_no_support += 1
                        continue
                    t_prefix_pred = list(t_pred[:-1])
                    t_prefix_lab_ids = [int(inp_t_np[b, pos]) for pos in t_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[0]),
                            int(t_pred[-1]),
                            [],
                            [],
                            t_prefix_pred,
                            t_prefix_lab_ids,
                            s_list,
                            t_list,
                            s_realized_id,
                            t_realized_id,
                        )
                    )
                    if v3_position_0_kl:
                        v3_pos0_records.append(
                            (
                                int(b),
                                int(s_pred[0]),
                                int(t_pred[0]),
                                int(inp_s_np[b, s_labels[0]]),
                                int(inp_t_np[b, t_labels[0]]),
                                frozenset(int(x) for x in s_list),
                            )
                        )
                elif M > 1 and N == 1:
                    # Many-to-1 (tokenalign.py:6833-6874)
                    s_realized_id = int(inp_s_np[b, s_labels[-1]])
                    t_realized_id = int(inp_t_np[b, t_labels[0]])
                    if M > max_chain_s and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    pairs = ()
                    if M <= max_chain_s:
                        student_prefix = tuple(
                            int(inp_s_np[b, pos]) for pos in s_labels[: M - 1]
                        )
                        pairs = prefix_index["reverse"].get(
                            (M, student_prefix),
                            (),
                        )
                    if not pairs and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    s_list, t_list = self._unique_bidir_pairs_cpu(
                        pairs,
                        swap=True,
                        assume_unique=_prefix_index_prededuped,
                    )
                    if teacher_topk_keep_realized and (
                        s_realized_id,
                        t_realized_id,
                    ) not in set(zip(s_list, t_list)):
                        s_list.append(s_realized_id)
                        t_list.append(t_realized_id)
                    min_support = 1 if teacher_topk_keep_realized else 2
                    if len(s_list) < min_support:
                        skipped_no_support += 1
                        continue
                    s_prefix_pred = list(s_pred[:-1])
                    s_prefix_lab_ids = [int(inp_s_np[b, pos]) for pos in s_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[-1]),
                            int(t_pred[0]),
                            s_prefix_pred,
                            s_prefix_lab_ids,
                            [],
                            [],
                            s_list,
                            t_list,
                            s_realized_id,
                            t_realized_id,
                        )
                    )
                    if v3_position_0_kl:
                        v3_pos0_records.append(
                            (
                                int(b),
                                int(s_pred[0]),
                                int(t_pred[0]),
                                int(inp_s_np[b, s_labels[0]]),
                                int(inp_t_np[b, t_labels[0]]),
                                frozenset(int(x) for x in s_list),
                            )
                        )
                else:
                    # Many-to-many (tokenalign.py:6875-6933).
                    # v6 hardcodes mtom_as_alm=True (see Section 8 hard
                    # constraint 7), so the ALM single-pair branch is the
                    # only path taken here. Surface-match enumeration
                    # (`_bidir_surface_support_pairs`) is intentionally not
                    # ported.
                    assert mtom_as_alm, (
                        "v6 requires mtom_as_alm=True; surface-match path "
                        "is not ported."
                    )
                    s_realized_id = int(inp_s_np[b, s_labels[-1]])
                    t_realized_id = int(inp_t_np[b, t_labels[-1]])
                    s_list = [s_realized_id]
                    t_list = [t_realized_id]
                    s_prefix_pred = list(s_pred[:-1])
                    s_prefix_lab_ids = [int(inp_s_np[b, pos]) for pos in s_labels[:-1]]
                    t_prefix_pred = list(t_pred[:-1])
                    t_prefix_lab_ids = [int(inp_t_np[b, pos]) for pos in t_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[-1]),
                            int(t_pred[-1]),
                            s_prefix_pred,
                            s_prefix_lab_ids,
                            t_prefix_pred,
                            t_prefix_lab_ids,
                            s_list,
                            t_list,
                            s_realized_id,
                            t_realized_id,
                        )
                    )
                    if v3_position_0_kl:
                        v3_pos0_records.append(
                            (
                                int(b),
                                int(s_pred[0]),
                                int(t_pred[0]),
                                int(inp_s_np[b, s_labels[0]]),
                                int(inp_t_np[b, t_labels[0]]),
                                frozenset(int(x) for x in s_list),
                            )
                        )

            if chunk_records:
                (
                    loss_sum_b,
                    used_b,
                    matches_b,
                    support_sizes_b,
                    valid_flags_b,
                ) = self._partition_kl_mismatch_batched(
                    chunk_records,
                    student_logits,
                    teacher_logits,
                    student_log_z,
                    teacher_log_z,
                    temperature,
                    reverse_kl,
                    loss_fn=effective_last_pos_loss_fn,
                    jsd_beta=v3_jsd_beta,
                    alm_bce_tau=v3_alm_bce_tau,
                    teacher_sparse=(
                        None
                        if teacher_sparse_payload is None
                        else (
                            teacher_sparse_payload[0],
                            teacher_sparse_payload[1],
                        )
                    ),
                    teacher_sparse_keep_realized=teacher_topk_keep_realized,
                )
                if v3_position_0_kl:
                    v3_pos0_records = [
                        rec
                        for rec, is_valid in zip(v3_pos0_records, valid_flags_b)
                        if is_valid
                    ]
                if used_b > 0:
                    mismatch_loss = loss_sum_b / float(used_b)
                    mismatch_matches += matches_b
                    support_sizes = support_sizes_b

        # ----- v3 position-0 partition KL (tokenalign.py:7129-7360) -----
        v3_pos0_loss_term = torch.tensor(
            0.0,
            device=device,
            dtype=accum_dtype,
        )
        if (
            v3_position_0_kl
            and has_common_support
            and len(v3_pos0_records) > 0
            and rest_bucket
        ):
            K = len(v3_pos0_records)
            b_idx_np = np.zeros((K,), dtype=np.int64)
            s_first_np = np.zeros((K,), dtype=np.int64)
            t_first_np = np.zeros((K,), dtype=np.int64)
            valid_row_np = np.zeros((K,), dtype=bool)

            for k, rec in enumerate(v3_pos0_records):
                b_i, s_first_pred, t_first_pred, *_ = rec
                b_idx_np[k] = b_i
                s_first_np[k] = s_first_pred
                t_first_np[k] = t_first_pred
            b_idx_v3 = torch.from_numpy(b_idx_np).to(device, non_blocking=True)
            s_first = torch.from_numpy(s_first_np).to(device, non_blocking=True)
            t_first = torch.from_numpy(t_first_np).to(device, non_blocking=True)

            NEG_INF = -1.0e30
            if teacher_sparse_payload is None:
                common_t_cpu = common_teacher_idx_t.detach().cpu().tolist()
                common_s_cpu = common_student_idx_t.detach().cpu().tolist()
                V_common = len(common_s_cpu)
                t_to_pair_slot = {int(t): i for i, t in enumerate(common_t_cpu)}
                s_to_pair_slot = {int(s): i for i, s in enumerate(common_s_cpu)}
                exclude_mask_np = np.zeros((K, V_common), dtype=bool)

                for k, rec in enumerate(v3_pos0_records):
                    (
                        _b_i,
                        _s_first_pred,
                        _t_first_pred,
                        r_s_id,
                        r_t_id,
                        slast_set,
                    ) = rec
                    n_excluded = 0
                    slot = s_to_pair_slot.get(int(r_s_id))
                    if slot is not None and not exclude_mask_np[k, slot]:
                        exclude_mask_np[k, slot] = True
                        n_excluded += 1
                    slot = t_to_pair_slot.get(int(r_t_id))
                    if slot is not None and not exclude_mask_np[k, slot]:
                        exclude_mask_np[k, slot] = True
                        n_excluded += 1
                    for sid in slast_set:
                        slot = s_to_pair_slot.get(int(sid))
                        if slot is not None and not exclude_mask_np[k, slot]:
                            exclude_mask_np[k, slot] = True
                            n_excluded += 1
                    valid_row_np[k] = (V_common - n_excluded) >= 1

                exclude_mask = torch.from_numpy(exclude_mask_np).to(
                    device,
                    non_blocking=True,
                )
                valid_row_v3 = torch.from_numpy(valid_row_np).to(
                    device,
                    non_blocking=True,
                )
                s_full_logits = (
                    student_logits[
                        b_idx_v3[:, None],
                        s_first[:, None],
                        common_student_idx_t[None, :],
                    ].to(accum_dtype)
                    / temperature
                )
                s_log_z_v3 = student_log_z[b_idx_v3, s_first].unsqueeze(-1)
                s_logp_v3 = s_full_logits - s_log_z_v3
                s_logp_v3 = torch.where(
                    exclude_mask,
                    torch.full_like(s_logp_v3, NEG_INF),
                    s_logp_v3,
                )

                with torch.no_grad():
                    assert teacher_logits is not None
                    t_full_logits = (
                        teacher_logits[
                            b_idx_v3[:, None],
                            t_first[:, None],
                            common_teacher_idx_t[None, :],
                        ].to(accum_dtype)
                        / temperature
                    )
                    t_log_z_v3 = teacher_log_z[b_idx_v3, t_first].unsqueeze(-1)
                    t_logp_v3 = t_full_logits - t_log_z_v3
                    t_logp_v3 = torch.where(
                        exclude_mask,
                        torch.full_like(t_logp_v3, NEG_INF),
                        t_logp_v3,
                    )
            else:
                teacher_topk_logits, teacher_topk_indices, _, _ = teacher_sparse_payload
                teacher_to_common_student = self._get_v3_teacher_to_common_student(
                    device,
                    i,
                    v_t,
                    common_student_idx_t,
                    common_teacher_idx_t,
                )
                topk_ids = teacher_topk_indices[b_idx_v3, t_first]
                topk_vals = teacher_topk_logits[b_idx_v3, t_first]
                mapped_student = teacher_to_common_student[topk_ids.long()]
                support_mask = mapped_student >= 0

                for k, rec in enumerate(v3_pos0_records):
                    (
                        _b_i,
                        _s_first_pred,
                        _t_first_pred,
                        r_s_id,
                        r_t_id,
                        slast_set,
                    ) = rec
                    exclude_s_ids = [int(r_s_id), *(int(x) for x in slast_set)]
                    if exclude_s_ids:
                        exclude_s_t = torch.tensor(
                            exclude_s_ids,
                            device=device,
                            dtype=torch.long,
                        )
                        support_mask[k] = support_mask[k] & ~torch.isin(
                            mapped_student[k],
                            exclude_s_t,
                        )
                    support_mask[k] = support_mask[k] & (topk_ids[k] != int(r_t_id))

                valid_row_v3 = support_mask.any(dim=-1)
                exclude_mask = ~support_mask
                safe_student_ids = mapped_student.clamp_min(0)
                s_full_logits = (
                    student_logits[
                        b_idx_v3[:, None], s_first[:, None], safe_student_ids
                    ].to(accum_dtype)
                    / temperature
                )
                s_logp_v3 = s_full_logits - student_log_z[b_idx_v3, s_first].unsqueeze(
                    -1
                )
                t_logp_v3 = topk_vals.to(accum_dtype) / temperature - teacher_log_z[
                    b_idx_v3, t_first
                ].unsqueeze(-1)
                s_logp_v3 = torch.where(
                    support_mask,
                    s_logp_v3,
                    torch.full_like(s_logp_v3, NEG_INF),
                )
                t_logp_v3 = torch.where(
                    support_mask,
                    t_logp_v3,
                    torch.full_like(t_logp_v3, NEG_INF),
                )

            s_logp_with_rest = self._append_rest_bucket_logp(s_logp_v3)
            t_logp_with_rest = self._append_rest_bucket_logp(t_logp_v3)

            if v3_loss_fn == "bce":
                per_pos = self._binary_power_bce_from_logp(
                    s_logp_with_rest,
                    t_logp_with_rest,
                    tau=v3_alm_bce_tau,
                )
            elif v3_loss_fn == "jsd":
                per_pos = _generalized_jsd(
                    s_logp_with_rest,
                    t_logp_with_rest,
                    v3_jsd_beta,
                )
            elif not reverse_kl:
                per_pos = torch.nn.functional.kl_div(
                    s_logp_with_rest,
                    t_logp_with_rest,
                    reduction="none",
                    log_target=True,
                )
            else:
                per_pos = torch.nn.functional.kl_div(
                    t_logp_with_rest,
                    s_logp_with_rest,
                    reduction="none",
                    log_target=True,
                )

            keep_mask_support = (~exclude_mask).to(per_pos.dtype)
            rest_keep = torch.ones(
                (K, 1),
                device=device,
                dtype=per_pos.dtype,
            )
            full_keep_mask = torch.cat(
                [keep_mask_support, rest_keep],
                dim=-1,
            )
            per_chunk_sum = (per_pos * full_keep_mask).sum(dim=-1)  # (K,)
            per_chunk_sum = per_chunk_sum * valid_row_v3.to(per_chunk_sum.dtype)
            denom = max(len(support_sizes), 1)
            v3_pos0_loss_term = per_chunk_sum.sum().to(accum_dtype) / float(denom)

        # ----- Combine + T^2 scale (tokenalign.py:7362-7425) -----------
        effective_mismatch_count = len(support_sizes)
        effective_count = len(common_chunks) + effective_mismatch_count
        # DP×CP-global valid-chunk count, computed in-loss. Reduce over the full
        # mesh (WORLD), then divide out the TP replication: the chunk count is
        # identical on every TP rank (chunks are vocab-independent), so
        # ``WORLD-sum == tp_world × (DP×CP-sum)`` exactly, and dividing by
        # ``tp_world`` recovers the DP×CP count. This matches the CE term's
        # ``global_valid_toks`` (which carries no TP factor), so the KD keeps its
        # configured weight relative to CE at TP>1 — unlike the deleted P-KL/gold
        # WORLD-reduce, which under-weighted the KD by 1/tp. The reduce is
        # unconditional so a no-chunk rank still fires the collective (the
        # early-return was removed for exactly this). At world size 1 (and CPU
        # parity) it is a no-op, so ``global_valid_chunks`` collapses to the local
        # ``effective_count`` and the normalization is byte-exact.
        if global_valid_chunks is None:
            global_valid_chunks = group_all_reduce_sum(
                torch.tensor(
                    float(effective_count), device=device, dtype=torch.float32
                ),
                group=torch.distributed.group.WORLD,
            )
            tp_world = (
                torch.distributed.get_world_size(tp_group)
                if tp_group is not None
                else 1
            )
            if tp_world > 1:
                global_valid_chunks = global_valid_chunks / float(tp_world)
        mismatch_last_loss = mismatch_loss
        mismatch_combined_loss, mismatch_loss = _combine_v3_mismatch_terms(
            mismatch_last_loss,
            v3_pos0_loss_term,
            pos0_coefficient=mismatch_pos0_coefficient,
            loss_multiplier=mismatch_loss_multiplier,
            convex=mismatch_combination_is_convex,
        )
        # v3_average_non_1to1 is a v7/v8/v9 knob; v6 leaves it False so
        # we omit the *0.5 rescale per the porting spec's v6-only scope.
        if effective_count > 0:
            # Sum of all chunk-level KLs for this microbatch. common_loss /
            # mismatch_loss are local per-chunk averages over their
            # respective partitions; multiplying by chunk counts recovers
            # the un-normalised partition sums.
            chunk_sum = common_loss * float(len(common_chunks)) + mismatch_loss * float(
                effective_mismatch_count
            )
            # Per-chunk normalisation. ``global_valid_chunks`` is either the
            # in-loss WORLD-reduce above or a caller-supplied count (both
            # >= this rank's ``effective_count`` > 0 here, so no div-by-zero).
            # The trainer's sum-reduction across microbatches and ranks then
            # yields `total_chunk_KL × T² / global_valid_chunks` — the natural
            # per-chunk KL averaged over the global batch, matching the
            # per-token convention used by _compute_ce (each loss normalises by
            # its own natural global denominator).
            loss_total = chunk_sum / global_valid_chunks.to(chunk_sum.dtype)
        else:
            # No usable chunks. Keep gradient-connected zero.
            loss_total = (student_logits.sum() * 0.0).to(accum_dtype)
        top1_accuracy = (common_matches + mismatch_matches) / max(
            effective_count,
            1,
        )

        # Final loss × T^2 (tokenalign.py:7425).
        final_loss = loss_total * (temperature**2)

        # Metric naming convention for wandb clarity:
        #   - `loss` / `train:loss`: sum-reduced by the trainer. When
        #     global_valid_chunks is provided, final wandb value is the
        #     per-chunk KL averaged over the global batch
        #     (`total_chunk_KL × T² / global_valid_chunks`). The dispatcher
        #     additionally overwrites `loss` with the dynamic-scaled
        #     `kl_scale * kl + ce` combination — that final value mixes
        #     a per-chunk KL term and a per-token CE term (consistent
        #     gradient balance by construction of kl_scale). `train:loss`
        #     is also the checkpoint-selection metric.
        #   - `*_per_chunk`: mean-reduced by the trainer (whitelisted in
        #     xtoken_distillation.py); final wandb value is the average
        #     per-mb local chunk-average — a clean per-chunk quantity when
        #     chunk counts are roughly uniform across microbatches.
        #   - `num_*`: sum-reduced; total count across the step.
        metrics: dict[str, Any] = {
            "loss": final_loss.item(),
            "train:loss": final_loss.item(),
            "kl_common_per_chunk": common_loss.item(),
            "kl_partition_last_per_chunk": mismatch_last_loss.item(),
            "kl_mismatch_combined_per_chunk": mismatch_combined_loss.item(),
            "kl_mismatch_scaled_per_chunk": mismatch_loss.item(),
            "num_common_chunks": len(common_chunks),
            "num_mismatch_chunks": effective_mismatch_count,
            "num_noise_filtered_common_chunks": noise_filtered_common_chunks,
            "num_noise_filtered_mismatch_chunks": noise_filtered_mismatch_chunks,
            "top1_acc_per_chunk": float(top1_accuracy),
            "num_valid_samples": B,
        }
        if uses_additive_coefficients:
            assert mismatch_pos0_alpha is not None
            assert mismatch_loss_beta is not None
            metrics["prefix_bidir_v3_mismatch_pos0_alpha"] = mismatch_pos0_alpha
            metrics["prefix_bidir_v3_mismatch_loss_beta"] = mismatch_loss_beta
        else:
            metrics["prefix_bidir_v3_mismatch_loss_scale"] = mismatch_loss_scale
        if not uses_additive_coefficients and mismatch_pos0_weight is not None:
            metrics["prefix_bidir_v3_mismatch_pos0_weight"] = mismatch_pos0_weight
        if noise_filter_topk > 0:
            metrics["prefix_bidir_v3_noise_filter_topk"] = noise_filter_topk
        if v3_position_0_kl:
            metrics["kl_partition_first_per_chunk"] = v3_pos0_loss_term.item()
        return final_loss, metrics
