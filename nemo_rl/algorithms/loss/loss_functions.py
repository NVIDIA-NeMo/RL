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

from typing import Any, NotRequired, Optional, TypedDict, TypeVar

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
        self.force_on_policy_ratio = cfg.get("force_on_policy_ratio", False)  # Force ratio to 1.0
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.use_importance_sampling_correction = cfg["use_importance_sampling_correction"]
        self.truncated_importance_sampling_ratio = cfg["truncated_importance_sampling_ratio"]
        # Type of truncated importance sampling: "tis" | "icepop" | "seq-mask-tis"
        self.truncated_importance_sampling_type = cfg.get("truncated_importance_sampling_type")
        # Lower bound for ICE-POP / seq-mask-tis filtering
        self.truncated_importance_sampling_ratio_min = cfg.get("truncated_importance_sampling_ratio_min")
        # Whether to compute importance weights per-sequence instead of per-token.
        self.sequence_level_importance_ratios = cfg.get(
            "sequence_level_importance_ratios",
            False,
        )
        self.loss_type = LossType.TOKEN_LEVEL if cfg["token_level_loss"] else LossType.SEQUENCE_LEVEL
        if self.sequence_level_importance_ratios:
            assert (
                self.loss_type == LossType.SEQUENCE_LEVEL
            ), "sequence-level importance sampling (e.g. GSPO) is mutually exclusive with token-level loss"
        if self.truncated_importance_sampling_ratio is not None:
            assert (
                self.use_importance_sampling_correction
            ), "truncated_importance_sampling_ratio is only supported when use_importance_sampling_correction is True"
            assert (
                self.truncated_importance_sampling_ratio > 0
            ), "truncated_importance_sampling_ratio should be positive"
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
            curr_logprobs_unfiltered = data.get("curr_logprobs_unfiltered", curr_logprobs)

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
        log_mixture = torch.log(0.5 * torch.exp(prev_logprobs) + 0.5 * torch.exp(generation_logprobs))
        # KL(P_train || M)
        kl_prev_to_mixture = torch.exp(prev_logprobs - log_mixture) - (prev_logprobs - log_mixture) - 1

        # KL(P_gen || M)
        kl_gen_to_mixture = torch.exp(generation_logprobs - log_mixture) - (generation_logprobs - log_mixture) - 1

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
                kl_importance_weights = torch.exp(curr_logprobs_unfiltered - generation_logprobs).detach()
                kl_importance_weights = torch.nan_to_num(kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0)
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
                kl = masked_mean(kl, mask, global_normalization_factor=global_valid_toks)
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
            ratios_clamped = ratios.clamp(1.0 - self.ratio_clip_min, 1.0 + self.ratio_clip_max)
        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped

        # Determine which value to use for clipping (max for pessimistic estimate)
        clip_loss = torch.max(loss1, loss2)

        # Dual-clipping see https://arxiv.org/pdf/1912.09729
        if self.ratio_clip_c is not None:
            assert (
                self.ratio_clip_c > 1
            ), f"ratio_clip_c must exceed 1 representing a lower bound of the ratios, got {self.ratio_clip_c}."
            loss3 = -advantages * self.ratio_clip_c
            clip_loss = torch.where(advantages < 0, torch.min(clip_loss, loss3), clip_loss)

        # -------------------------------------------------------------
        # Off-policy (actor) importance-sampling correction
        # -------------------------------------------------------------
        _is_filter_metrics: dict = {}  # populated for icepop / seq-mask-tis
        # See: docs/guides/grpo.md#importance-sampling-correction
        if self.sequence_level_importance_ratios:
            # importance weight w_i = exp(Σ_t (log π_actor − log π_behaviour))
            seq_lp_diff = ((prev_logprobs - generation_logprobs) * mask).sum(dim=-1)
            actor_importance_weights = torch.exp(seq_lp_diff).detach()
            actor_importance_weights = torch.nan_to_num(actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0)
            # Broadcast to token dimension so we can reuse existing reduction
            actor_importance_weights_expanded = actor_importance_weights.unsqueeze(-1)
        else:
            # Token-level correction
            actor_importance_weights_expanded = torch.exp(prev_logprobs - generation_logprobs)
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
                token_in_bounds = actor_importance_weights_expanded <= self.truncated_importance_sampling_ratio
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
                    actor_importance_weights_expanded >= self.truncated_importance_sampling_ratio_min
                ) & (actor_importance_weights_expanded <= self.truncated_importance_sampling_ratio)
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
                seq_log_is_ratio_mean = masked_mean(log_is_ratio, token_mask, dim=-1)  # [B]
                seq_geomean_is_ratio = torch.exp(seq_log_is_ratio_mean).detach()  # [B]
                seq_kept_mask = (
                    (seq_geomean_is_ratio >= self.truncated_importance_sampling_ratio_min)
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
                actor_importance_weights_expanded = actor_importance_weights_expanded * seq_kept_mask.unsqueeze(-1)
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

        return self._preference_loss(rewards, sample_mask, global_valid_seqs, self.reference_policy_kl_penalty)

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
            assert global_valid_toks is not None, "global_valid_toks must be provided for SFT loss"
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

        dpo_loss = self.sft_loss_weight * sft_loss_chosen + self.preference_loss_weight * preference_loss

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
        assert self.mixed_kl_weight >= 0 and self.mixed_kl_weight <= 1, "Invalid mixed KL weight"

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
                loss_correction_term = loss_correction_term * (1.0 - self.mixed_kl_weight)

        if self.kl_type == "forward":
            per_token_kl = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
        elif self.kl_type == "reverse":
            per_token_kl = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
        else:
            # mixed KL
            kl_forward = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
            kl_reverse = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
            per_token_kl = self.mixed_kl_weight * kl_forward + (1.0 - self.mixed_kl_weight) * kl_reverse

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


# ============================================================================
# SDPO Loss
# ============================================================================


class SDPOLossConfig(TypedDict):
    """Configuration for the SDPO (Self-Distilled Policy Optimization) loss.

    SDPO computes a logit-level KL between a student (current policy on the
    original prompt) and a self-teacher (same policy conditioned on a successful
    demonstration), summed over the full vocabulary at every response position
    (top-k approximation, paper Eq. 1).

    Defaults:
        kl_type: "reverse"            - paper Eq. 1 form; "forward", "mixed", or "js" (symmetric
                                        Jensen-Shannon, bounded in [0, log 2] per token) also supported
        mixed_kl_weight: 0.5          - weight on forward-KL when kl_type="mixed"
        zero_outside_topk: True       - model the teacher as having ~0 mass outside top-k
                                        and add the corresponding tail-correction term to the loss
                                        (paper §A.2). Setting False omits the tail and keeps only
                                        the top-k sum (cheaper, less accurate).
        success_reward_threshold: 1.0 - minimum reward to count as "successful" (used by orchestration)
    """

    kl_type: NotRequired[str]
    mixed_kl_weight: NotRequired[float]
    zero_outside_topk: NotRequired[bool]
    success_reward_threshold: float
    # Trust-region anchor to a frozen-init reference policy (paper Table 4).
    # When penalty > 0, the loss adds beta * KL(student || ref) summed over
    # response positions, where the KL is estimated by one of Schulman's
    # k1/k2/k3 estimators at the sampled tokens.
    reference_policy_kl_penalty: NotRequired[float]
    reference_policy_kl_type: NotRequired[str]


class SDPOLossDataDict(TypedDict):
    """Required keys in the data BatchedDataDict for SDPOLossFn."""

    input_ids: torch.Tensor  # [B, S]
    token_mask: torch.Tensor  # [B, S]      1 = response token
    sample_mask: torch.Tensor  # [B]         1 = valid sample
    sdpo_mask: torch.Tensor  # [B]         1 = sample has a demonstration
    teacher_topk_logits: torch.Tensor  # [B, S, K]   aligned to student positions
    teacher_topk_indices: torch.Tensor  # [B, S, K]


class SDPOLossFn(LossFunction):
    """Self-Distilled Policy Optimization loss (logit-level KL, paper Eq. 1).

    Trains the student (current policy on the original prompt) to match the
    self-teacher (current policy conditioned on a successful demonstration) via
    a top-k logit-level KL summed over response positions:

        L(θ) = Σ_t  D_KL( π_θ(·|x, y_<t)  ‖  stopgrad π_θ(·|x, f, y_<t) )
             ≈ Σ_t  Σ_{ŷ ∈ topK}  π_θ(ŷ|x,y_<t) · [log π_θ(ŷ|x,y_<t) − log π_T(ŷ|x,f,y_<t)]
               + tail_correction_t       (when zero_outside_topk=True)

    Top-k indices are chosen by the teacher (paper §A.2). Tail correction uses the
    full-vocab student entropy H_all returned by the training forward, so the gather
    over top-k preserves an unbiased estimate of the full-vocabulary KL even with
    K << |V|.

    Samples without a demonstration (sdpo_mask=0) contribute zero to the loss.

    References:
        Hübotter et al. (2026) "Reinforcement Learning via Self-Distillation"
        arXiv:2601.20802
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION

    def __init__(self, cfg: SDPOLossConfig):
        self.kl_type: str = cfg.get("kl_type", "reverse")
        self.mixed_kl_weight: float = cfg.get("mixed_kl_weight", 0.5)
        self.zero_outside_topk: bool = cfg.get("zero_outside_topk", True)
        self.log_infinitesimal: float = -100.0
        self.reference_policy_kl_penalty: float = cfg.get("reference_policy_kl_penalty", 0.0)
        self.reference_policy_kl_type: str = cfg.get("reference_policy_kl_type", "k3")

        if self.kl_type not in {"forward", "reverse", "mixed", "js"}:
            raise ValueError(f"SDPOLossFn: kl_type must be one of forward/reverse/mixed/js, got {self.kl_type}")
        if not 0.0 <= self.mixed_kl_weight <= 1.0:
            raise ValueError(f"SDPOLossFn: mixed_kl_weight must be in [0, 1], got {self.mixed_kl_weight}")
        if self.reference_policy_kl_penalty < 0.0:
            raise ValueError(
                f"SDPOLossFn: reference_policy_kl_penalty must be >= 0, got {self.reference_policy_kl_penalty}"
            )
        if self.reference_policy_kl_type not in {"k1", "k2", "k3"}:
            raise ValueError(
                f"SDPOLossFn: reference_policy_kl_type must be one of k1/k2/k3, got {self.reference_policy_kl_type}"
            )

    def __call__(
        self,
        student_topk_logprobs: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        H_all: Optional[torch.Tensor],
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the SDPO logit-level KL loss.

        Args:
            student_topk_logprobs: log π_θ at teacher's top-k indices, shape [B, S-1, K].
            teacher_topk_logprobs: log π_T at teacher's top-k indices, shape [B, S-1, K].
            H_all: full-vocab student entropy, shape [B, S-1] (or None when not needed).
            data: must contain keys defined in SDPOLossDataDict.
            global_valid_seqs: number of valid sequences in this microbatch.
            global_valid_toks: number of valid tokens in this microbatch.

        Returns:
            (loss, metrics)
        """
        student_probs = student_topk_logprobs.exp()  # [B, S-1, K]
        teacher_probs = teacher_topk_logprobs.exp()  # [B, S-1, K]

        if self.kl_type == "forward":
            per_token_kl = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
        elif self.kl_type == "reverse":
            per_token_kl = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
        elif self.kl_type == "js":
            # Symmetric Jensen-Shannon divergence at top-k positions, bounded in
            # [0, log 2] per (sample, position). Per the paper (Appendix E, §A),
            # this is the recommended SDPO loss for stability.
            m_probs = 0.5 * (student_probs + teacher_probs)
            log_m = m_probs.clamp_min(1e-12).log()
            per_token_kl = 0.5 * student_probs * (student_topk_logprobs - log_m) + 0.5 * teacher_probs * (
                teacher_topk_logprobs - log_m
            )
        else:
            kl_forward = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
            kl_reverse = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
            per_token_kl = self.mixed_kl_weight * kl_forward + (1.0 - self.mixed_kl_weight) * kl_reverse

        per_token_kl = per_token_kl.sum(dim=-1)  # [B, S-1]

        # Tail correction for tokens outside top-k (mirrors DistillationLossFn at
        # loss_functions.py:986). Added when we treat the teacher as having ~0
        # mass outside its top-k (zero_outside_topk=True): the student-weighted
        # tail entropy is added back so the approximation stays unbiased.
        # Skipped for "forward" (teacher's near-zero tail mass already implies
        # ~zero contribution) and "js" (the JS midpoint distribution doesn't
        # factor into a clean single-direction tail; top-K truncation already
        # bounds the per-token loss and JS is bounded in [0, log 2] regardless).
        if self.zero_outside_topk and self.kl_type not in {"forward", "js"}:
            assert H_all is not None, (
                "SDPOLossFn requires H_all when zero_outside_topk=True; "
                "the policy training forward must compute full-vocab entropy."
            )
            H_rest = H_all - (student_probs * student_topk_logprobs).sum(-1)
            P_rest = 1.0 - student_probs.sum(-1)
            tail = H_rest - self.log_infinitesimal * P_rest  # [B, S-1]
            if self.kl_type == "mixed":
                tail = tail * (1.0 - self.mixed_kl_weight)
            per_token_kl = per_token_kl + tail

        # Trust-region anchor to a frozen-init reference policy (paper Table 4).
        # We use prev_logprobs (snapshot at step start) as the student side; at
        # LR=3e-7 the within-step drift is small. ref_kl is added regardless of
        # sdpo_mask so that even samples without a teacher demonstration are
        # still anchored to the init policy.
        ref_kl_per_token: Optional[torch.Tensor] = None
        if self.reference_policy_kl_penalty > 0.0 and "prev_logprobs" in data and "reference_policy_logprobs" in data:
            max_len = per_token_kl.shape[1]
            student_lp = data["prev_logprobs"][:, 1:][:, :max_len]  # [B, S-1]
            ref_lp = data["reference_policy_logprobs"][:, 1:][:, :max_len]
            log_ratio = ref_lp - student_lp  # log(p_ref / p_student) at sampled tokens
            if self.reference_policy_kl_type == "k1":
                ref_kl_per_token = -log_ratio
            elif self.reference_policy_kl_type == "k2":
                ref_kl_per_token = 0.5 * log_ratio.pow(2)
            else:  # "k3" — Schulman, unbiased, low-variance, always >= 0
                ref_kl_per_token = torch.exp(log_ratio) - 1.0 - log_ratio
            per_token_kl = per_token_kl + self.reference_policy_kl_penalty * ref_kl_per_token

        # Effective mask: response token AND sample has demo AND sample is valid.
        # token_mask is [B, S]; align to [B, S-1] and trim to per_token_kl length.
        token_mask = data["token_mask"][:, 1:]
        sdpo_mask = data["sdpo_mask"]
        sample_mask = data["sample_mask"]
        max_len = per_token_kl.shape[1]
        token_mask = token_mask[:, :max_len]
        effective_mask = token_mask * sdpo_mask.unsqueeze(-1).float() * sample_mask.unsqueeze(-1)

        loss = masked_mean(
            per_token_kl,
            effective_mask,
            global_normalization_factor=global_valid_toks,
        )

        metrics = {
            "num_valid_samples": sample_mask.sum().item(),
            "sdpo/per_pos_kl": masked_mean(per_token_kl, effective_mask).item(),
            # sdpo/frac_with_demo removed: the per-microbatch fraction breaks
            # under dtensor_policy_worker's metric aggregation (divides by
            # num_global_batches, not num_microbatches, producing a value >> 1
            # when it should be in [0, 1]). Use sdpo/frac_with_demo_pre_train
            # (logged once per step in sdpo_train) for the correct fraction.
        }
        if ref_kl_per_token is not None:
            # Response-position mask without sdpo_mask: ref-KL applies to every
            # sample regardless of whether a teacher demonstration is available.
            ref_mask = token_mask * sample_mask.unsqueeze(-1)
            metrics["sdpo/ref_kl"] = masked_mean(ref_kl_per_token, ref_mask).item()

        return loss, metrics


# ============================================================================
# SDPO + GRPO hybrid loss (paper §4.5)
# ============================================================================


class SDPOHybridLossConfig(TypedDict):
    """Configuration for the SDPO+GRPO hybrid loss (paper §4.5).

    The hybrid blends a clipped policy-gradient (GRPO) term with the SDPO
    logit-level KL distillation term:

        L(θ) = grpo_weight · L_GRPO(θ)  +  (1 − grpo_weight) · L_SDPO(θ)

    Note on fidelity: the paper combines the two *advantages*
    (A = λ·A_GRPO + (1−λ)·A_SDPO). Because this codebase realizes SDPO as a KL
    distillation loss rather than an advantage, we blend at the *loss* level
    instead. Mixing the losses mixes their gradients, which captures the same
    bias/variance trade-off the paper describes, but is not bit-identical to
    the advantage-space formula.

    The two component configs are nested (rather than flattened) so the
    `reference_policy_kl_penalty` key — which means different things for SDPO
    (anchor to the frozen-init policy) and GRPO (KL penalty in the PG loss) —
    does not collide.

    Fields:
        grpo_weight: λ ∈ [0, 1]. 0 ⇒ pure SDPO, 1 ⇒ pure GRPO clipped-PG.
        sdpo: an SDPOLossConfig (see SDPOLossFn).
        grpo: a ClippedPGLossConfig (see ClippedPGLossFn).
    """

    grpo_weight: float
    sdpo: SDPOLossConfig
    grpo: ClippedPGLossConfig


class SDPOHybridLossFn(LossFunction):
    """SDPO+GRPO hybrid loss (paper §4.5), blended at the loss level.

    Composes an inner SDPOLossFn and ClippedPGLossFn and returns

        grpo_weight · L_GRPO + (1 − grpo_weight) · L_SDPO

    The combined training forward (LossInputType.DISTILLATION_AND_LOGPROB)
    produces both the distillation top-k logprobs (for the SDPO term) and the
    next-token logprobs (for the clipped-PG term); see
    nemo_rl/algorithms/loss/utils.py:prepare_loss_input.

    The SDPO term only affects samples that have a teacher demonstration
    (sdpo_mask=1); the GRPO term affects every valid sample. This means even
    demonstration-less samples still receive the policy-gradient signal.

    References:
        Hübotter et al. (2026) "Reinforcement Learning via Self-Distillation"
        arXiv:2601.20802 §4.5
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION_AND_LOGPROB

    def __init__(self, cfg: SDPOHybridLossConfig):
        self.grpo_weight: float = float(cfg["grpo_weight"])
        if not 0.0 <= self.grpo_weight <= 1.0:
            raise ValueError(f"SDPOHybridLossFn: grpo_weight must be in [0, 1], got {self.grpo_weight}")
        # prepare_loss_input reads .sdpo_loss / .grpo_loss off this object.
        self.sdpo_loss = SDPOLossFn(cfg["sdpo"])
        self.grpo_loss = ClippedPGLossFn(cfg["grpo"])

    def __call__(
        self,
        student_topk_logprobs: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        H_all: Optional[torch.Tensor],
        next_token_logprobs: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sdpo_loss, sdpo_metrics = self.sdpo_loss(
            student_topk_logprobs,
            teacher_topk_logprobs,
            H_all,
            data,
            global_valid_seqs,
            global_valid_toks,
        )
        grpo_loss, grpo_metrics = self.grpo_loss(
            next_token_logprobs,
            data,
            global_valid_seqs,
            global_valid_toks,
        )

        loss = self.grpo_weight * grpo_loss + (1.0 - self.grpo_weight) * sdpo_loss

        metrics: dict[str, Any] = {
            "loss": loss.item(),
            # Both component losses mask by sample_mask, so either count works.
            "num_valid_samples": grpo_metrics["num_valid_samples"],
            # loss-like values aggregate correctly under the worker's
            # divide-by-num_global_batches-then-sum scheme (like "loss").
            # grpo_weight is a constant and would be corrupted by it, so it is
            # logged once per step in sdpo_train instead.
            "hybrid/loss_grpo": grpo_loss.item(),
            "hybrid/loss_sdpo": sdpo_loss.item(),
        }
        # Namespace the GRPO component metrics; keep SDPO's (already sdpo/*).
        for k, v in grpo_metrics.items():
            if k in ("loss", "num_valid_samples"):
                continue
            metrics[f"grpo/{k}"] = v
        for k, v in sdpo_metrics.items():
            if k in ("loss", "num_valid_samples"):
                continue
            metrics[k] = v

        return loss, metrics
