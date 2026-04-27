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

import math
import time
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, Union

import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
from nemo_rl.algorithms.utils import calculate_kl, masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import DistributedCrossEntropy

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


class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    reference_policy_kl_type: str
    kl_input_clamp_value: float | None
    kl_output_clamp_value: float | None
    ratio_clip_min: float
    ratio_clip_max: float
    # Dual-clipping value (should be >1 if enabled; usually set to 3 empirically). None to disable.
    ratio_clip_c: float | None
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    truncated_importance_sampling_ratio: float | None
    # Type of truncated importance sampling:
    #   "tis"          – clamp IS weights to max
    #   "icepop"       – zero out tokens with IS weight outside [min, max]
    #   "seq-mask-tis" – zero out sequences by geometric-mean IS ratio, non-truncated token IS correction
    truncated_importance_sampling_type: NotRequired[str | None]
    # Lower bound for ICE-POP / seq-mask-tis filtering
    truncated_importance_sampling_ratio_min: NotRequired[float | None]
    token_level_loss: bool
    # If True, apply the off-policy importance-sampling correction at the
    # sequence level (one weight per generated sample), as in GSPO.
    # If False (default), correction is applied at the token level as in the
    # original GRPO paper.
    sequence_level_importance_ratios: NotRequired[bool]
    disable_ppo_ratio: NotRequired[bool]
    # If True, force the ratio to 1.0 for truly on-policy behavior,
    # eliminating any importance sampling effects.
    # NOTE: This should only be used when doing exactly one update per rollout
    # (i.e., num_prompts_per_step * num_generations_per_prompt == train_global_batch_size)
    force_on_policy_ratio: NotRequired[bool]
    # If True, add KL penalty to reward instead of loss (used by Reinforce++)
    use_kl_in_reward: NotRequired[bool]


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

    def __init__(self, cfg: ClippedPGLossConfig):
        self.ratio_clip_min = cfg["ratio_clip_min"]
        self.ratio_clip_max = cfg["ratio_clip_max"]
        self.ratio_clip_c = cfg["ratio_clip_c"]  # set to None to disable dual-clipping
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.reference_policy_kl_type = cfg["reference_policy_kl_type"]
        self.kl_input_clamp_value = cfg["kl_input_clamp_value"]
        self.kl_output_clamp_value = cfg["kl_output_clamp_value"]
        self.disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)
        self.force_on_policy_ratio = cfg.get(
            "force_on_policy_ratio", False
        )  # Force ratio to 1.0
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.use_importance_sampling_correction = cfg[
            "use_importance_sampling_correction"
        ]
        self.truncated_importance_sampling_ratio = cfg[
            "truncated_importance_sampling_ratio"
        ]
        # Type of truncated importance sampling: "tis" | "icepop" | "seq-mask-tis"
        self.truncated_importance_sampling_type = cfg.get(
            "truncated_importance_sampling_type"
        )
        # Lower bound for ICE-POP / seq-mask-tis filtering
        self.truncated_importance_sampling_ratio_min = cfg.get(
            "truncated_importance_sampling_ratio_min"
        )
        # Whether to compute importance weights per-sequence instead of per-token.
        self.sequence_level_importance_ratios = cfg.get(
            "sequence_level_importance_ratios",
            False,
        )
        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg["token_level_loss"] else LossType.SEQUENCE_LEVEL
        )
        if self.sequence_level_importance_ratios:
            assert self.loss_type == LossType.SEQUENCE_LEVEL, (
                "sequence-level importance sampling (e.g. GSPO) is mutually exclusive with token-level loss"
            )
        if self.truncated_importance_sampling_ratio is not None:
            assert self.use_importance_sampling_correction, (
                "truncated_importance_sampling_ratio is only supported when use_importance_sampling_correction is True"
            )
            assert self.truncated_importance_sampling_ratio > 0, (
                "truncated_importance_sampling_ratio should be positive"
            )
            assert self.truncated_importance_sampling_type in (
                "tis",
                "icepop",
                "seq-mask-tis",
            ), (
                f"truncated_importance_sampling_type must be 'tis', 'icepop', or 'seq-mask-tis', "
                f"got {self.truncated_importance_sampling_type}"
            )
            if self.truncated_importance_sampling_type == "seq-mask-tis":
                assert not self.sequence_level_importance_ratios, (
                    "seq-mask-tis uses token-level IS correction with sequence-level masking, "
                    "and is incompatible with sequence_level_importance_ratios=True"
                )
        else:
            # Warn user that TIS-related parameters are ignored when truncated_importance_sampling_ratio is not set
            ignored_params = []
            if cfg.get("truncated_importance_sampling_type") is not None:
                ignored_params.append("truncated_importance_sampling_type")
            if cfg.get("truncated_importance_sampling_ratio_min") is not None:
                ignored_params.append("truncated_importance_sampling_ratio_min")
            if ignored_params:
                print(
                    f"[WARN] truncated_importance_sampling_ratio is not set, so the following "
                    f"parameters are ignored: {', '.join(ignored_params)}. "
                    f"Set truncated_importance_sampling_ratio to enable truncated importance sampling.",
                    flush=True,
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
        prev_logprobs = data["prev_logprobs"][:, 1:]
        generation_logprobs = data["generation_logprobs"][:, 1:]
        if self.reference_policy_kl_penalty != 0:
            reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
            curr_logprobs_unfiltered = data.get(
                "curr_logprobs_unfiltered", curr_logprobs
            )

        mask = token_mask * sample_mask.unsqueeze(-1)

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
            if self.use_on_policy_kl_approximation:
                # See: docs/guides/grpo.md#on-policy-kl-approximation
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - generation_logprobs
                ).detach()
                kl_importance_weights = torch.nan_to_num(
                    kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                kl_importance_weights = torch.ones_like(curr_logprobs_unfiltered)

            # Compute KL loss
            kl = (
                kl_importance_weights
                * self.reference_policy_kl_penalty
                * calculate_kl(
                    logprobs=curr_logprobs_unfiltered,
                    logprobs_reference=reference_policy_logprobs,
                    kl_type=self.reference_policy_kl_type,
                    input_clamp_value=self.kl_input_clamp_value,
                    output_clamp_value=self.kl_output_clamp_value,
                )
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
        # "tis"          – clamp IS weights to [0, max]
        # "icepop"       – zero out tokens whose IS weight ∉ [min, max]   (ref bounds: 0.5–5)
        # "seq-mask-tis" – zero out entire sequences whose geometric-mean
        #                  IS ratio ∉ [min, max]; retained sequences keep
        #                  raw (non-truncated) token-level IS weights      (ref bounds: 0.999–1.002)
        #   Blog: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
        if self.truncated_importance_sampling_ratio is not None:
            if self.truncated_importance_sampling_type == "tis":
                token_in_bounds = (
                    actor_importance_weights_expanded
                    <= self.truncated_importance_sampling_ratio
                )
                _is_filter_metrics = {
                    "is_oob_ratio": 1.0
                    - masked_mean(
                        token_in_bounds.float(),
                        mask,
                        global_normalization_factor=global_valid_toks,
                    ).item(),
                }
                actor_importance_weights_expanded = torch.clamp(
                    actor_importance_weights_expanded,
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
                    "is_oob_ratio": 1.0
                    - masked_mean(
                        token_kept_mask.float(),
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
                    "is_oob_ratio": 1.0
                    - masked_mean(
                        seq_kept_mask,
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

        loss = actor_loss + kl
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
            },
        )


class NLLLossFn(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, use_linear_ce_fusion: bool = False):
        self.use_linear_ce_fusion = use_linear_ce_fusion

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


class DPOLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    preference_loss_weight: float
    sft_loss_weight: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool


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

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False):
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight = cfg["preference_loss_weight"]
        self.sft_loss_weight = cfg["sft_loss_weight"]
        self.preference_average_log_probs = cfg["preference_average_log_probs"]
        self.sft_average_log_probs = cfg["sft_average_log_probs"]
        self.use_linear_ce_fusion = use_linear_ce_fusion
        self.sft_loss = NLLLossFn(use_linear_ce_fusion=use_linear_ce_fusion)

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


class CrossTokenizerDistillationLossConfig(TypedDict):
    """Configuration for cross-tokenizer distillation loss."""
    loss_type: str                        # 'KL', 'cross_entropy', or 'chunked_ce'
    temperature: float                    # Softmax temperature
    vocab_topk: int                       # Reduce teacher vocab to top-k (0 = all)
    exact_token_match_only: bool          # Only use 1:1 aligned positions
    reverse_kl: bool                      # Reverse KL direction
    project_teacher_to_student: NotRequired[bool]
    gold_loss: NotRequired[bool]          # Use gold loss (common KL + uncommon L1, no projection)
    xtoken_loss: NotRequired[bool]        # Relaxed exact-map threshold (>=0.6 instead of ==1.0)
    ce_loss_scale: NotRequired[float]     # Scale for additional CE (next-token) loss (0.0 = disabled)
    dynamic_loss_scaling: NotRequired[bool]  # Scale KL loss to match CE magnitude


class CrossTokenizerDistillationLossDataDict(TypedDict):
    """Data dict for cross-tokenizer distillation.

    Only contains student-side tensors (same sequence dimension).
    Teacher-side data (teacher_input_ids, aligned_pairs) is stored on the
    loss function instance via set_cross_tokenizer_data() to avoid
    sequence-length mismatches in the worker's shape validation.
    """
    input_ids: torch.Tensor               # Student token IDs (B, S_student)
    input_lengths: torch.Tensor
    token_mask: torch.Tensor              # (B, S_student)
    sample_mask: torch.Tensor             # (B,)


def _scatter_chunk_mask_from_coo(
    coo_list: list[torch.Tensor],
    batch_size: int,
    seq_len: int,
    total_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialize a dense ``(batch_size, seq_len, total_chunks)`` bool mask.

    ``coo_list[b]`` is a CPU ``LongTensor (N_b, 2)`` with rows ``[pos, chunk_id]``
    precomputed by ``CrossTokenizerCollator._build_chunk_coo`` (already
    filtered for ``exact_match_only`` / ``-1`` sentinels / padded-length bounds).

    ``chunk_id`` in each sample's COO is in ``[0, num_chunks_per_sample)``;
    samples with fewer chunks than ``total_chunks`` leave the padded columns
    all-False, which the downstream ``chunk_valid = (chunk_sizes > 0)`` gate
    already drops.
    """
    parts: list[torch.Tensor] = []
    for b, coo in enumerate(coo_list):
        if coo.shape[0] == 0:
            continue
        bcol = torch.full((coo.shape[0], 1), b, dtype=torch.int64)
        parts.append(torch.cat([bcol, coo], dim=1))
    mask = torch.zeros(
        batch_size, seq_len, total_chunks, dtype=torch.bool, device=device,
    )
    if parts:
        idx = torch.cat(parts, dim=0).to(device)
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


class CrossTokenizerDistillationLossFn(LossFunction):
    """Cross-tokenizer distillation loss using TokenAligner's projection matrix.

    Computes per-token KL divergence between projected student probabilities
    (in teacher vocab space) and teacher probabilities, only at positions where
    the two tokenizations have 1:1 aligned tokens. Uses NeMo RL's standard
    masked_mean normalization so loss magnitude is comparable to same-tokenizer
    distillation.

    Teacher-specific data (teacher_input_ids, aligned_pairs) is stored on
    this object via set_cross_tokenizer_data() before each training step,
    rather than in the data dict, because teacher and student sequences
    have different lengths and the worker validates that all tensors in
    the data dict share the same sequence dimension.
    """

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig, token_aligner):
        from nemo_rl.algorithms.x_token.tokenalign import TokenAligner
        assert isinstance(token_aligner, TokenAligner)
        self.token_aligner = token_aligner
        self.cfg = cfg
        self.loss_type = LossType.TOKEN_LEVEL
        self._teacher_input_ids = None
        self._aligned_pairs = None
        self._chunk_indices: Optional[dict[str, list]] = None

    def set_cross_tokenizer_data(
        self,
        teacher_input_ids: torch.Tensor,
        aligned_pairs: list,
        chunk_indices: Optional[dict[str, list]] = None,
    ):
        """Store teacher-side data before each training step.

        Called from the training loop before student_policy.train().
        The worker never sees these tensors in shape validation.

        ``chunk_indices`` carries the per-sample COO chunk-mask indices that
        used to be rebuilt inside ``__call__`` every microbatch. When set, it
        is a dict with keys ``student_chunk_coo``, ``teacher_chunk_coo``,
        ``num_chunks``, each a DP-sharded list of length ``batch_size``.
        """
        self._teacher_input_ids = teacher_input_ids
        self._aligned_pairs = aligned_pairs
        self._chunk_indices = chunk_indices

    def _project_student_to_teacher(
        self,
        student_logits: torch.Tensor,
        teacher_vocab_size: int,
        temperature: float,
        global_top_indices: torch.Tensor,
        device: torch.device,
        precomputed_student_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project student logits into the reduced teacher vocabulary space.

        Returns projected student probabilities of shape (B, S_student, K)
        where K = len(global_top_indices).

        If `precomputed_student_probs` is provided, it is used directly instead
        of recomputing softmax(student_logits / temperature). The caller is
        responsible for ensuring it was computed with the same temperature.
        """
        if precomputed_student_probs is not None:
            student_probs = precomputed_student_probs
        else:
            student_probs = torch.softmax(student_logits / temperature, dim=-1)

        has_sparse = (
            hasattr(self.token_aligner, 'sparse_transformation_matrix')
            and self.token_aligner.sparse_transformation_matrix is not None
        )
        if has_sparse:
            sparse_mat = self.token_aligner.sparse_transformation_matrix
            reduced_sparse = sparse_mat.index_select(1, global_top_indices).coalesce()
            projected = self.token_aligner.project_token_likelihoods_instance(
                student_probs, None, None, None, device,
                use_sparse_format=True,
                sparse_matrix=reduced_sparse,
            )
            return projected

        proj_values = self.token_aligner.likelihood_projection_matrix
        if getattr(self.token_aligner, 'learnable', False):
            proj_values = self.token_aligner.transform_learned_matrix_instance(proj_values)
        projected_full = self.token_aligner.project_token_likelihoods_instance(
            student_probs, self.token_aligner.likelihood_projection_indices,
            proj_values, teacher_vocab_size, device,
            use_sparse_format=False,
        )
        return projected_full[:, :, global_top_indices]

    def _compute_gold_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_chunk_mask: torch.Tensor,
        teacher_chunk_mask: torch.Tensor,
        batch_size: int,
        student_seq_len: int,
        teacher_seq_len: int,
        teacher_vocab_size: int,
        temperature: float,
        reverse_kl: bool,
        xtoken_loss: bool,
        device: torch.device,
        precomputed_student_log_probs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float]:
        """Gold loss: common-vocab KL + uncommon-vocab sorted L1.

        Splits the vocabulary into tokens with exact 1:1 projection mappings
        ("common") and the rest ("uncommon"). Common tokens are compared
        directly via KL on their native log-probs (no projection needed).
        Uncommon tokens are compared via L1 on sorted probability vectors
        (Universal Likelihood Distillation).

        Matches tokenalign.py compute_KL_loss_optimized gold_loss branch.
        """
        partition = self.token_aligner.build_vocab_partition(
            xtoken_loss=xtoken_loss,
            teacher_vocab_size=teacher_vocab_size,
        )
        common_student_indices = partition.common_student_indices.to(device)
        common_teacher_indices = partition.common_teacher_indices.to(device)
        uncommon_student_indices = partition.uncommon_student_indices.to(device)
        uncommon_teacher_indices = partition.uncommon_teacher_indices.to(device)

        # student_chunk_mask / teacher_chunk_mask are precomputed by the
        # caller (from collator-emitted per-sample COO) and shared with the
        # non-gold path — shape (B, seq_len, total_chunks), dtype bool.
        total_chunks = student_chunk_mask.shape[-1]

        # log_softmax on full original logits BEFORE chunk averaging
        if precomputed_student_log_probs is not None:
            student_log_probs = precomputed_student_log_probs
        else:
            student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher_logits / temperature, dim=-1)

        # Chunk-average log-probs over full vocabularies
        student_chunk_lp = torch.bmm(
            student_chunk_mask.transpose(1, 2).to(student_log_probs.dtype), student_log_probs,
        )
        teacher_chunk_lp = torch.bmm(
            teacher_chunk_mask.transpose(1, 2).to(teacher_log_probs.dtype), teacher_log_probs,
        )
        del student_log_probs, teacher_log_probs

        student_chunk_sizes = student_chunk_mask.sum(dim=1, keepdim=True).float().transpose(1, 2)
        teacher_chunk_sizes = teacher_chunk_mask.sum(dim=1, keepdim=True).float().transpose(1, 2)

        student_chunk_lp = student_chunk_lp / (student_chunk_sizes + 1e-10)
        teacher_chunk_lp = teacher_chunk_lp / (teacher_chunk_sizes + 1e-10)

        chunk_valid = (student_chunk_sizes.squeeze(-1) > 0) & (teacher_chunk_sizes.squeeze(-1) > 0)

        if not chunk_valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0

        # --- Part 1: KL on common (exactly-mapped) vocab ---
        loss_kl_common = torch.tensor(0.0, device=device, requires_grad=True)
        if common_student_indices.numel() > 0:
            s_common = student_chunk_lp[:, :, common_student_indices]
            t_common = teacher_chunk_lp[:, :, common_teacher_indices]

            if not reverse_kl:
                kl_elem = torch.nn.functional.kl_div(
                    s_common, t_common, reduction="none", log_target=True,
                )
            else:
                kl_elem = torch.nn.functional.kl_div(
                    t_common, s_common, reduction="none", log_target=True,
                )
            kl_per_chunk = kl_elem.sum(dim=-1) * chunk_valid
            if chunk_valid.sum() > 0:
                loss_kl_common = kl_per_chunk.sum() / chunk_valid.sum()

        # --- Part 2: L1 on uncommon (unaligned) vocab ---
        loss_l1_uncommon = torch.tensor(0.0, device=device, requires_grad=True)
        if uncommon_student_indices.numel() > 0 or uncommon_teacher_indices.numel() > 0:
            s_uncommon = (
                student_chunk_lp[:, :, uncommon_student_indices]
                if uncommon_student_indices.numel() > 0
                else torch.empty(batch_size, total_chunks, 0, device=device)
            )
            t_uncommon = (
                teacher_chunk_lp[:, :, uncommon_teacher_indices]
                if uncommon_teacher_indices.numel() > 0
                else torch.empty(batch_size, total_chunks, 0, device=device)
            )

            s_valid = s_uncommon[chunk_valid]
            t_valid = t_uncommon[chunk_valid]

            if s_valid.shape[0] > 0:
                with torch.no_grad():
                    max_uncommon_vocab = min(s_valid.shape[-1], t_valid.shape[-1], 8192)

                if max_uncommon_vocab > 0:
                    s_probs = torch.exp(s_valid)
                    t_probs = torch.exp(t_valid)

                    if s_probs.shape[-1] > max_uncommon_vocab:
                        s_sorted, _ = torch.topk(s_probs, k=max_uncommon_vocab, dim=-1, largest=True)
                    else:
                        s_sorted = torch.sort(s_probs, dim=-1, descending=True)[0]

                    if t_probs.shape[-1] > max_uncommon_vocab:
                        t_sorted, _ = torch.topk(t_probs, k=max_uncommon_vocab, dim=-1, largest=True)
                    else:
                        t_sorted = torch.sort(t_probs, dim=-1, descending=True)[0]

                    del s_probs, t_probs
                    min_len = min(s_sorted.shape[-1], t_sorted.shape[-1])
                    if min_len > 0:
                        loss_l1_per_chunk = torch.nn.functional.l1_loss(
                            s_sorted[:, :min_len], t_sorted[:, :min_len], reduction='none',
                        ).sum(dim=-1)
                        loss_l1_uncommon = loss_l1_per_chunk.mean()
                        del loss_l1_per_chunk
                    del s_sorted, t_sorted

        loss_total = (loss_kl_common + loss_l1_uncommon) * (temperature ** 2)

        # Top-1 accuracy on common vocab. Computed as a 0-d GPU tensor so the
        # caller can include it in the batched .cpu().tolist() sync at the end
        # of the step (instead of forcing two CPU<->GPU stalls per teacher per
        # microbatch here).
        top1_accuracy: Union[float, torch.Tensor] = 0.0
        with torch.no_grad():
            if common_student_indices.numel() > 0 and chunk_valid.any():
                s_valid_lp = student_chunk_lp[chunk_valid][:, common_student_indices]
                t_valid_lp = teacher_chunk_lp[chunk_valid][:, common_teacher_indices]
                matches = (
                    s_valid_lp.argmax(dim=-1) == t_valid_lp.argmax(dim=-1)
                ).sum()
                denom = chunk_valid.sum().clamp(min=1)
                top1_accuracy = matches.to(torch.float32) / denom.to(torch.float32)

        del student_chunk_lp, teacher_chunk_lp
        return loss_total, top1_accuracy

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: CrossTokenizerDistillationLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        mb_idx: Optional[int] = None,
        mbs: Optional[int] = None,
        teacher_topk_indices_ipc: Optional[torch.Tensor] = None,
        _return_raw_kl: bool = False,
        precomputed_student_logits_f32: Optional[torch.Tensor] = None,
        precomputed_student_probs: Optional[torch.Tensor] = None,
        precomputed_student_log_probs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute cross-tokenizer distillation loss via chunk-averaged KL.

        For each alignment chunk (1:1, 1:many, many:1, or many:many), the
        projected student and teacher distributions are averaged over their
        respective spans, renormalized, and compared via KL divergence.
        The per-chunk KL is then distributed back to student positions
        and normalized with the standard NeMo RL masked_mean.

        The three ``precomputed_*`` kwargs let an outer aggregator hoist
        student-side work that does not depend on the teacher (fp32 cast,
        softmax, log_softmax) out of the per-teacher loop and share it
        across multiple teachers with the same temperature.
        """
        input_ids_student = data["input_ids"]
        batch_size = input_ids_student.shape[0]

        # Keep logits in their native dtype (typically bf16). The downstream
        # log_softmax / softmax / bmm ops on CUDA upcast to fp32 internally for
        # numerics while storing activations in bf16, which roughly halves the
        # working-set memory of the gold-loss / projection paths.
        if precomputed_student_logits_f32 is not None:
            student_logits = precomputed_student_logits_f32
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            student_logits = next_token_logits.full_tensor()
        else:
            student_logits = next_token_logits

        if teacher_logits is None:
            raise ValueError(
                "CrossTokenizerDistillationLossFn requires teacher_logits via IPC. "
                "Set use_ipc=True in the distillation config."
            )
        if self._aligned_pairs is None or self._teacher_input_ids is None:
            raise ValueError(
                "Cross-tokenizer data not set. "
                "Call loss_fn.set_cross_tokenizer_data() before training."
            )

        # IPC contract (XTokenTeacherIPCExportPostProcessor, full-vocab branch):
        # raw logits in their native dtype (bf16 in practice). The log_softmax
        # calls below in _compute_gold_loss / the projection path are the only
        # ones in the teacher pipeline. Variable name kept for diff stability.
        if isinstance(teacher_logits, torch.distributed.tensor.DTensor):
            teacher_logits_f32 = teacher_logits.full_tensor()
        else:
            teacher_logits_f32 = teacher_logits

        if teacher_logits_f32.shape[-1] == 0:
            raise ValueError(
                f"Teacher logits have vocab dimension 0 (shape={teacher_logits_f32.shape}). "
                "This typically means topk_logits=0 was passed instead of None "
                "for the teacher forward pass. Cross-tokenizer distillation "
                "requires full teacher logits (topk_logits=None)."
            )

        if mb_idx is not None and mbs is not None:
            mb_start = mb_idx * mbs
            mb_end = mb_start + batch_size
        else:
            mb_start = 0
            mb_end = batch_size

        self.token_aligner = self.token_aligner.to(student_logits.device)
        device = student_logits.device

        temperature = self.cfg.get("temperature", 1.0)
        vocab_topk = self.cfg.get("vocab_topk", 8192)
        reverse_kl = self.cfg.get("reverse_kl", False)
        use_gold_loss = self.cfg.get("gold_loss", False)
        use_xtoken_loss = self.cfg.get("xtoken_loss", False)
        student_seq_len = student_logits.shape[1]
        teacher_seq_len = teacher_logits_f32.shape[1]
        teacher_vocab_size = teacher_logits_f32.shape[-1]

        # The collator pre-applies exact_match_only, sentinel (-1), and
        # padded-length bounds filters, and emits the chunk mask in COO form
        # per sample. The loss fn just slices the MB range and scatters.
        if self._chunk_indices is None:
            raise ValueError(
                "CrossTokenizerDistillationLossFn requires chunk_indices. "
                "CrossTokenizerCollator should have precomputed them; verify "
                "the training loop forwards them through update_cross_tokenizer_data()."
            )
        student_coo_list = self._chunk_indices["student_chunk_coo"][mb_start:mb_end]
        teacher_coo_list = self._chunk_indices["teacher_chunk_coo"][mb_start:mb_end]
        num_chunks_list = self._chunk_indices["num_chunks"][mb_start:mb_end]
        total_chunks = max(num_chunks_list) if num_chunks_list else 0

        if total_chunks == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return loss, {"loss": 0.0, "topk_accuracy": 0.0, "num_chunks": 0}

        proj_mask = _scatter_chunk_mask_from_coo(
            student_coo_list, batch_size, student_seq_len, total_chunks, device,
        )
        tgt_mask = _scatter_chunk_mask_from_coo(
            teacher_coo_list, batch_size, teacher_seq_len, total_chunks, device,
        )
        num_valid_chunks_total = int(sum(num_chunks_list))

        # ================================================================
        # Gold loss path: common-vocab KL + uncommon-vocab sorted L1.
        # Bypasses the projection matrix for tokens with exact 1:1 mappings.
        # Matches tokenalign.py compute_KL_loss_optimized gold_loss branch.
        # ================================================================
        if use_gold_loss:
            loss, top1_accuracy = self._compute_gold_loss(
                student_logits, teacher_logits_f32, proj_mask, tgt_mask,
                batch_size, student_seq_len, teacher_seq_len,
                teacher_vocab_size,
                temperature, reverse_kl, use_xtoken_loss, device,
                precomputed_student_log_probs=precomputed_student_log_probs,
            )
        else:
            # ================================================================
            # Standard projection-based path
            # ================================================================

            # -- 3. Global vocabulary filtering (top-k teacher tokens) --
            with torch.no_grad():
                if vocab_topk == 0 or vocab_topk >= teacher_vocab_size:
                    global_top_indices = torch.arange(teacher_vocab_size, device=device)
                else:
                    teacher_flat = teacher_logits_f32.view(-1, teacher_vocab_size)
                    importance = teacher_flat.max(dim=0)[0]
                    _, global_top_indices = torch.topk(
                        importance, k=min(vocab_topk, teacher_vocab_size), dim=-1,
                    )
                    global_top_indices = global_top_indices.sort()[0]

            # -- 4. Project student probs to teacher vocab --
            projected_student = self._project_student_to_teacher(
                student_logits, teacher_vocab_size, temperature, global_top_indices, device,
                precomputed_student_probs=precomputed_student_probs,
            )

            # -- 5. Teacher log-probs in reduced vocab --
            teacher_logits_reduced = teacher_logits_f32[:, :, global_top_indices]
            teacher_log_probs = torch.log_softmax(teacher_logits_reduced / temperature, dim=-1)
            del teacher_logits_reduced

            # -- 6. Chunk-averaged distributions --
            proj_chunks = torch.bmm(
                proj_mask.transpose(1, 2).to(projected_student.dtype), projected_student,
            )
            tgt_log_chunks = torch.bmm(
                tgt_mask.transpose(1, 2).to(teacher_log_probs.dtype), teacher_log_probs,
            )
            del projected_student, teacher_log_probs

            proj_sizes = proj_mask.sum(dim=1).unsqueeze(-1).to(proj_chunks.dtype)
            tgt_sizes = tgt_mask.sum(dim=1).unsqueeze(-1).to(tgt_log_chunks.dtype)

            proj_chunks = proj_chunks / (proj_sizes + 1e-10)
            tgt_log_chunks = tgt_log_chunks / (tgt_sizes + 1e-10)

            proj_chunks = proj_chunks / (proj_chunks.sum(dim=-1, keepdim=True) + 1e-10)
            proj_log_chunks = torch.log(proj_chunks + 1e-10)

            chunk_valid = (proj_sizes.squeeze(-1) > 0) & (tgt_sizes.squeeze(-1) > 0)

            # -- 7. KL divergence per chunk --
            if reverse_kl:
                kl_per_elem = torch.nn.functional.kl_div(
                    tgt_log_chunks, proj_log_chunks, reduction="none", log_target=True,
                )
            else:
                kl_per_elem = torch.nn.functional.kl_div(
                    proj_log_chunks, tgt_log_chunks, reduction="none", log_target=True,
                )
            kl_per_chunk = kl_per_elem.sum(dim=-1) * (temperature ** 2)
            kl_per_chunk = kl_per_chunk * chunk_valid
            del proj_chunks, tgt_log_chunks, proj_log_chunks, kl_per_elem

            # -- 8. Scalar loss --
            num_valid_chunks = chunk_valid.sum()
            if num_valid_chunks > 0:
                loss = kl_per_chunk.sum() / num_valid_chunks
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            top1_accuracy = 0.0

        # Keep a stable alias for raw-KL return path and optional CE fusion path.
        kl_loss = loss
        if _return_raw_kl:
            # Return scalar metrics as GPU tensors so the outer aggregator can
            # batch the CPU sync for all teachers / metrics into a single
            # .cpu().tolist() call. The aggregator unwraps tensor entries.
            raw_metrics: dict[str, Any] = {
                "loss": kl_loss if isinstance(kl_loss, torch.Tensor) else kl_loss,
                "kl_loss": kl_loss if isinstance(kl_loss, torch.Tensor) else kl_loss,
                "topk_accuracy": top1_accuracy,
                "num_valid_samples": int(batch_size),
                "num_chunks": num_valid_chunks_total,
            }
            return kl_loss, raw_metrics

        # ================================================================
        # Optional CE (next-token prediction) loss, matching the DDP
        # train_distillation_ddp.py logic:
        #   without dynamic scaling: loss = kl * kl_weight + ce * ce_scale
        #   with dynamic scaling:    loss = kl * (ce/kl)   + ce
        # ================================================================
        ce_loss_scale = self.cfg.get("ce_loss_scale", 0.0)
        dynamic_loss_scaling = self.cfg.get("dynamic_loss_scaling", False)
        ce_loss_value = 0.0

        if ce_loss_scale > 0.0 or dynamic_loss_scaling:
            # Mask padding positions so CE loss only covers real tokens.
            # token_mask[:, 1:] marks valid next-token targets (shifted by 1).
            token_mask = data["token_mask"]
            ce_mask = token_mask[:, 1 : student_seq_len].to(torch.bool)
            ce_targets = input_ids_student[:, 1:student_seq_len].clone()
            ce_targets[~ce_mask] = -100
            ce_loss = torch.nn.functional.cross_entropy(
                student_logits[:, :student_seq_len - 1].reshape(-1, student_logits.shape[-1]),
                ce_targets.reshape(-1),
                ignore_index=-100,
            )
            ce_loss_value = float(ce_loss.item())

            if dynamic_loss_scaling and kl_loss.item() > 0:
                dls_scale = ce_loss.item() / kl_loss.item()
                loss = kl_loss * dls_scale + ce_loss
            else:
                loss = kl_loss + ce_loss * ce_loss_scale

        # Scale for NeMo RL distributed training
        token_mask = data["token_mask"]
        sample_mask = data["sample_mask"]
        max_len = min(token_mask.shape[1] - 1, student_seq_len)
        local_mask = token_mask[:, 1 : max_len + 1] * sample_mask.unsqueeze(-1)
        local_valid_toks = local_mask.sum()

        if local_valid_toks > 0 and global_valid_toks > 0:
            tok_scale = float(local_valid_toks / global_valid_toks)
            loss = loss * tok_scale
        else:
            tok_scale = 0.0
            loss = loss * 0.0

        num_valid = num_valid_chunks_total
        metrics = {
            "loss": float(loss.item()) if loss.ndim == 0 else loss,
            "kl_loss": float(kl_loss.item()) * tok_scale,
            "ce_loss": ce_loss_value * tok_scale,
            "topk_accuracy": top1_accuracy,
            "num_valid_samples": int(batch_size),
            "num_chunks": num_valid,
            "alignment_density": num_valid / max(1, batch_size * student_seq_len),
        }

        return loss, metrics


class MultiTeacherLossAggregator(LossFunction):
    """Aggregate weighted losses from multiple teachers in a unified path."""

    def __init__(
        self,
        loss_fns: list[Optional[CrossTokenizerDistillationLossFn]],
        weights: list[float],
        normalize_by_vocab: bool = False,
        cfg: Optional[dict[str, Any]] = None,
    ):
        assert len(loss_fns) == len(weights), (
            f"loss_fns ({len(loss_fns)}) and weights ({len(weights)}) length mismatch"
        )
        self.loss_fns = loss_fns
        self.weights = weights
        self.normalize_by_vocab = normalize_by_vocab
        self.cfg = cfg or {}
        self.teacher_aggregation_mode = self.cfg.get("teacher_aggregation_mode", "weighted")
        if self.teacher_aggregation_mode not in {"weighted", "routing", "average"}:
            raise ValueError(
                "teacher_aggregation_mode must be one of {'weighted', 'routing', 'average'}, "
                f"got '{self.teacher_aggregation_mode}'"
            )
        self.loss_type = LossType.TOKEN_LEVEL
        # Marks this loss fn as expecting student logits as its primary input.
        # Required by `prepare_loss_input` so the unified single-/multi-teacher
        # cross-tokenizer path can flow through this aggregator.
        self.input_type = LossInputType.LOGIT

    def set_cross_tokenizer_data(
        self,
        teacher_input_ids: torch.Tensor,
        aligned_pairs: list,
        teacher_idx: Optional[int] = None,
        chunk_indices: Optional[dict[str, list]] = None,
    ) -> None:
        # When called from the single-teacher dispatch (no explicit teacher_idx),
        # default to the only teacher slot so the unified worker path keeps
        # working without per-call branching upstream.
        if teacher_idx is None:
            if len(self.loss_fns) != 1:
                raise ValueError(
                    "set_cross_tokenizer_data requires teacher_idx when "
                    f"len(loss_fns) > 1 (got {len(self.loss_fns)})"
                )
            teacher_idx = 0
        fn = self.loss_fns[teacher_idx]
        if fn is not None:
            fn.set_cross_tokenizer_data(
                teacher_input_ids, aligned_pairs, chunk_indices=chunk_indices,
            )

    def _compute_same_tokenizer_kl(
        self,
        next_token_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        data: CrossTokenizerDistillationLossDataDict,
        teacher_topk_indices_ipc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        student_logits = next_token_logits.to(torch.float32)
        t_logits = teacher_logits.to(student_logits.device, dtype=torch.float32)

        seq_len = student_logits.shape[1] - 1
        student_shifted = student_logits[:, :-1]

        if teacher_topk_indices_ipc is None:
            teacher_logprobs = torch.nn.functional.log_softmax(t_logits[:, :seq_len], dim=-1)
            student_logprobs = torch.nn.functional.log_softmax(student_shifted, dim=-1)
            per_token_kl = (
                teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)
            ).sum(dim=-1)
        else:
            topk_idx = teacher_topk_indices_ipc[:, :seq_len].to(student_shifted.device)
            teacher_topk = t_logits[:, :seq_len]
            student_logprobs = torch.nn.functional.log_softmax(student_shifted, dim=-1)
            student_topk = torch.gather(student_logprobs, dim=-1, index=topk_idx)
            teacher_topk_probs = teacher_topk.exp()
            teacher_rest = (1.0 - teacher_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
            teacher_probs_full = torch.cat([teacher_topk_probs, teacher_rest], dim=-1)
            teacher_logprobs_full = torch.cat([teacher_topk, teacher_rest.log()], dim=-1)
            student_topk_probs = student_topk.exp()
            student_rest = (1.0 - student_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
            student_logprobs_full = torch.cat([student_topk, student_rest.log()], dim=-1)
            per_token_kl = (
                teacher_probs_full * (teacher_logprobs_full - student_logprobs_full)
            ).sum(dim=-1)

        token_mask = data["token_mask"][:, 1 : seq_len + 1]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)
        valid_toks = mask.sum().clamp(min=1.0)
        return (per_token_kl * mask).sum() / valid_toks

    def __call__(
        self,
        next_token_logits: Optional[torch.Tensor] = None,
        data: Optional[CrossTokenizerDistillationLossDataDict] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        mb_idx: Optional[int] = None,
        mbs: Optional[int] = None,
        teacher_topk_indices_ipc: Optional[torch.Tensor] = None,
        teacher_logits_list: Optional[list[torch.Tensor]] = None,
        teacher_topk_indices_list: Optional[list[Optional[torch.Tensor]]] = None,
        teacher_routing_indices: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Accept `logits=` as an alias for `next_token_logits=` so this aggregator
        # is a drop-in for `prepare_loss_input`, which emits {"logits": ...} for
        # LossInputType.LOGIT. This removes the need for a separate Compat shim.
        if next_token_logits is None:
            next_token_logits = logits
        if next_token_logits is None:
            raise ValueError(
                "MultiTeacherLossAggregator requires either `next_token_logits` "
                "or `logits` to be provided."
            )

        if teacher_logits_list is None:
            teacher_logits_list = [teacher_logits] if teacher_logits is not None else []
        if teacher_topk_indices_list is None:
            teacher_topk_indices_list = [teacher_topk_indices_ipc] * len(teacher_logits_list)

        if len(teacher_logits_list) == 0:
            zero = torch.tensor(0.0, device=next_token_logits.device, requires_grad=True)
            return zero, {"loss": 0.0, "num_valid_samples": 0}

        if len(teacher_logits_list) != len(self.loss_fns):
            raise ValueError(
                "teacher_logits_list length must match number of configured teachers: "
                f"{len(teacher_logits_list)} != {len(self.loss_fns)}"
            )
        if len(teacher_topk_indices_list) != len(teacher_logits_list):
            raise ValueError(
                "teacher_topk_indices_list length must match teacher_logits_list length: "
                f"{len(teacher_topk_indices_list)} != {len(teacher_logits_list)}"
            )

        vocab_sizes = [int(t.shape[-1]) for t in teacher_logits_list]
        min_log_vocab = math.log(max(2, min(vocab_sizes))) if self.normalize_by_vocab else 1.0

        total_kl = torch.tensor(0.0, device=next_token_logits.device, requires_grad=True)
        metrics: dict[str, Any] = {}
        # GPU-side scalar tensors accumulated during the loop, synced once
        # (as a single batched D2H copy) at the end of the call. This keeps the
        # CPU from stalling between teachers and between micro-steps, which is
        # critical for letting the GPU pipeline kernels back-to-back.
        gpu_scalar_metrics: dict[str, torch.Tensor] = {}
        routing_indices = teacher_routing_indices
        if routing_indices is None and isinstance(data, dict):
            routing_indices = data.get("teacher_routing_indices", None)
        if self.teacher_aggregation_mode == "routing":
            if routing_indices is None:
                raise ValueError(
                    "teacher_aggregation_mode='routing' requires teacher_routing_indices "
                    "either as an argument or in data['teacher_routing_indices']"
                )
            if routing_indices.ndim != 1:
                raise ValueError(
                    "teacher_routing_indices must be a rank-1 tensor with one teacher index per sample"
                )
            if routing_indices.shape[0] != data["sample_mask"].shape[0]:
                raise ValueError(
                    "teacher_routing_indices length must match batch size: "
                    f"{routing_indices.shape[0]} != {data['sample_mask'].shape[0]}"
                )

        original_sample_mask = data["sample_mask"]
        active_teachers = sum(1 for t_logits in teacher_logits_list if t_logits is not None)
        average_weight = 1.0 / max(1, active_teachers)

        # ===== Hoist student-only work out of the per-teacher loop =====
        # The fp32 cast and softmax/log_softmax of student_logits depend only
        # on student_logits and (optionally) the per-teacher temperature.
        # When two or more teachers share the same temperature and code path
        # (gold_loss vs. projection), we can compute the corresponding
        # student tensor exactly once and reuse it across those teachers.
        per_teacher_share_keys: list[Optional[tuple[float, str]]] = []
        share_key_counts: dict[tuple[float, str], int] = {}
        for teacher_idx, (loss_fn, t_logits) in enumerate(
            zip(self.loss_fns, teacher_logits_list)
        ):
            if t_logits is None or loss_fn is None:
                per_teacher_share_keys.append(None)
                continue
            cfg = getattr(loss_fn, "cfg", {}) or {}
            temp = float(cfg.get("temperature", 1.0))
            kind = "log_probs" if cfg.get("gold_loss", False) else "probs"
            key = (temp, kind)
            per_teacher_share_keys.append(key)
            share_key_counts[key] = share_key_counts.get(key, 0) + 1

        has_sharing = any(c >= 2 for c in share_key_counts.values())
        shared_softmax_cache: dict[tuple[float, str], torch.Tensor] = {}
        if has_sharing:
            if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
                full_logits = next_token_logits.full_tensor()
            else:
                full_logits = next_token_logits
            # Keep the cached softmax / log_softmax in the input dtype
            # (typically bf16). The CUDA softmax / log_softmax kernels already
            # accumulate in fp32 internally regardless of input dtype, so
            # numerics match the previous explicit fp32 cast while halving
            # the cache memory and avoiding two transient full-vocab fp32
            # buffers (the .to(fp32) result and the / temp intermediate)
            # that were causing OOM at mbs > 1.
            for key, count in share_key_counts.items():
                if count < 2:
                    continue
                temp, kind = key
                # Skip the divide-by-temperature kernel (and its transient
                # buffer) when temperature is the no-op default.
                scaled_logits = full_logits if temp == 1.0 else full_logits / temp
                if kind == "probs":
                    shared_softmax_cache[key] = torch.softmax(scaled_logits, dim=-1)
                else:
                    shared_softmax_cache[key] = torch.log_softmax(scaled_logits, dim=-1)
                if scaled_logits is not full_logits:
                    del scaled_logits
            del full_logits
        # ===============================================================

        for teacher_idx, (loss_fn, weight, t_logits, t_topk_idx) in enumerate(
            zip(self.loss_fns, self.weights, teacher_logits_list, teacher_topk_indices_list)
        ):
            if t_logits is None:
                continue
            if self.teacher_aggregation_mode == "routing":
                routed_sample_mask = original_sample_mask * (
                    routing_indices.to(original_sample_mask.device) == teacher_idx
                ).to(original_sample_mask.dtype)
                routed_samples = int((routed_sample_mask > 0).sum().item())
                metrics[f"teacher_{teacher_idx}/routed_samples"] = routed_samples
                if routed_samples == 0:
                    continue
                data["sample_mask"] = routed_sample_mask
            teacher_compute_start = time.perf_counter()
            if loss_fn is not None:
                share_key = per_teacher_share_keys[teacher_idx]
                shared_softmax = (
                    shared_softmax_cache.get(share_key)
                    if share_key is not None
                    else None
                )
                shared_kwargs: dict[str, Any] = {}
                if shared_softmax is not None and share_key is not None:
                    if share_key[1] == "probs":
                        shared_kwargs["precomputed_student_probs"] = shared_softmax
                    else:
                        shared_kwargs["precomputed_student_log_probs"] = shared_softmax
                teacher_kl, teacher_metrics = loss_fn(
                    next_token_logits=next_token_logits,
                    data=data,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                    vocab_parallel_rank=vocab_parallel_rank,
                    vocab_parallel_group=vocab_parallel_group,
                    context_parallel_group=context_parallel_group,
                    teacher_logits=t_logits,
                    mb_idx=mb_idx,
                    mbs=mbs,
                    teacher_topk_indices_ipc=t_topk_idx,
                    _return_raw_kl=True,
                    **shared_kwargs,
                )
            else:
                teacher_kl = self._compute_same_tokenizer_kl(
                    next_token_logits, t_logits, data, t_topk_idx
                )
                # Defer this scalar's sync to the batched .cpu().tolist() below.
                teacher_metrics = {"kl_loss": teacher_kl}
            data["sample_mask"] = original_sample_mask
            teacher_compute_elapsed = time.perf_counter() - teacher_compute_start

            vocab_scale = 1.0
            if self.normalize_by_vocab:
                vocab_scale = math.log(max(2, int(t_logits.shape[-1]))) / min_log_vocab

            if self.teacher_aggregation_mode == "weighted":
                effective_weight = weight
            elif self.teacher_aggregation_mode == "average":
                effective_weight = average_weight
            else:
                effective_weight = 1.0
            weighted_teacher_kl = teacher_kl * effective_weight * vocab_scale
            total_kl = total_kl + weighted_teacher_kl

            # Stash GPU scalars; sync once at the end. Non-tensor metadata
            # (weights, vocab_scale, elapsed time) goes straight into metrics.
            gpu_scalar_metrics[f"teacher_{teacher_idx}/raw_kl"] = teacher_kl.detach()
            metrics[f"teacher_{teacher_idx}/weight"] = float(effective_weight)
            metrics[f"teacher_{teacher_idx}/vocab_scale"] = float(vocab_scale)
            gpu_scalar_metrics[f"teacher_{teacher_idx}/weighted_kl"] = (
                weighted_teacher_kl.detach()
            )
            metrics[f"teacher_{teacher_idx}/loss_compute"] = float(teacher_compute_elapsed)
            for key, value in teacher_metrics.items():
                metric_key = f"teacher_{teacher_idx}/{key}"
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    gpu_scalar_metrics[metric_key] = value.detach()
                else:
                    metrics[metric_key] = value

        # Release shared student tensors before CE loss / masking work to keep
        # peak memory close to the original per-teacher implementation.
        shared_softmax_cache.clear()
        del shared_softmax_cache

        ce_loss_scale = self.cfg.get("ce_loss_scale", 0.0)
        dynamic_loss_scaling = self.cfg.get("dynamic_loss_scaling", False)
        loss = total_kl
        ce_loss_tensor: Optional[torch.Tensor] = None
        if ce_loss_scale > 0.0 or dynamic_loss_scaling:
            # Pass logits in their native dtype; cross_entropy internally
            # promotes to fp32 for the log_softmax/NLL reduction. Avoids
            # materializing a second full-vocab fp32 tensor here.
            ce_logits = next_token_logits
            student_seq_len = ce_logits.shape[1]
            # Mask padding positions so CE loss only covers real tokens.
            token_mask_ce = data["token_mask"][:, 1:student_seq_len].to(torch.bool)
            ce_targets = data["input_ids"][:, 1:student_seq_len].clone()
            ce_targets[~token_mask_ce] = -100
            ce_loss = torch.nn.functional.cross_entropy(
                ce_logits[:, :student_seq_len - 1].reshape(-1, ce_logits.shape[-1]),
                ce_targets.reshape(-1),
                ignore_index=-100,
            )
            ce_loss_tensor = ce_loss
            if dynamic_loss_scaling:
                # Avoid the per-microbatch CPU<->GPU sync by computing the
                # scaling factor entirely on-device. The scale is detached so
                # gradient flow matches the original (where dls_scale was a
                # Python scalar treated as a constant). The clamp guards
                # against the degenerate total_kl==0 case; in practice KL is
                # strictly positive, so this matches the original branch.
                dls_scale = (
                    ce_loss.detach() / total_kl.detach().clamp(min=1e-10)
                )
                loss = total_kl * dls_scale + ce_loss
            else:
                loss = total_kl + ce_loss * ce_loss_scale

        token_mask = data["token_mask"]
        sample_mask = data["sample_mask"]
        student_seq_len = next_token_logits.shape[1]
        max_len = min(token_mask.shape[1] - 1, student_seq_len)
        local_mask = token_mask[:, 1 : max_len + 1] * sample_mask.unsqueeze(-1)
        local_valid_toks = local_mask.sum()
        if local_valid_toks > 0 and global_valid_toks > 0:
            tok_scale = float(local_valid_toks / global_valid_toks)
            loss = loss * tok_scale
        else:
            tok_scale = 0.0
            loss = loss * 0.0

        # Defer the loss / kl_loss / ce_loss sync to the single batched
        # transfer below. Apply tok_scale to kl_loss and ce_loss so wandb
        # reports the correct global mean (not N_ranks × mean).
        if isinstance(loss, torch.Tensor) and loss.ndim == 0:
            gpu_scalar_metrics["loss"] = loss.detach()
        else:
            metrics["loss"] = loss
        if isinstance(total_kl, torch.Tensor) and total_kl.ndim == 0:
            gpu_scalar_metrics["kl_loss"] = total_kl.detach() * tok_scale
        else:
            metrics["kl_loss"] = total_kl * tok_scale if total_kl else 0.0
        if ce_loss_tensor is not None:
            gpu_scalar_metrics["ce_loss"] = ce_loss_tensor.detach() * tok_scale
        else:
            metrics["ce_loss"] = 0.0
        metrics["num_valid_samples"] = int(data["input_ids"].shape[0])

        # Single batched D2H copy for every scalar metric we collected above.
        # This replaces the 8-14 individual .item() syncs that were previously
        # interspersed throughout the per-teacher loop and the CE / dynamic
        # loss scaling branches, which forced the CPU to wait between teachers
        # and prevented the GPU from pipelining their kernels.
        if gpu_scalar_metrics:
            keys = list(gpu_scalar_metrics.keys())
            stacked = torch.stack(
                [t.reshape(()) for t in gpu_scalar_metrics.values()]
            )
            values = stacked.cpu().tolist()
            for k, v in zip(keys, values):
                metrics[k] = float(v)

        return loss, metrics
