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


# =============================================================================
# Cross-Tokenizer Distillation Loss (via TokenAligner)
# =============================================================================


class CrossTokenizerDistillationLossConfig(TypedDict):
    """Configuration for cross-tokenizer distillation loss."""

    loss_type: str  # 'KL', 'cross_entropy', or 'chunked_ce'
    temperature: float  # softmax temperature
    vocab_topk: int  # reduce teacher vocab to top-k (0 = all)
    exact_token_match_only: bool  # only use 1:1 aligned positions
    reverse_kl: bool  # reverse KL direction
    project_teacher_to_student: NotRequired[bool]
    gold_loss: NotRequired[bool]  # common-KL + uncommon-L1 (no projection)
    xtoken_loss: NotRequired[bool]  # relaxed exact-map threshold (>= 0.6)
    ce_loss_scale: NotRequired[float]  # aux next-token CE weight (0.0 = off)
    dynamic_loss_scaling: NotRequired[bool]  # rescale KL to match CE magnitude


class CrossTokenizerDistillationLossDataDict(TypedDict):
    """Student-side tensors for cross-tokenizer distillation.

    Teacher-side data (``teacher_input_ids``, ``aligned_pairs``) is stored on
    the loss instance via :meth:`CrossTokenizerDistillationLossFn.set_cross_tokenizer_data`
    instead of being passed through here, because teacher and student
    sequences have different lengths and the worker validates that all
    tensors in a data dict share the same sequence dimension.
    """

    input_ids: torch.Tensor  # (B, S_student)
    input_lengths: torch.Tensor
    token_mask: torch.Tensor  # (B, S_student)
    sample_mask: torch.Tensor  # (B,)


class CrossTokenizerDistillationLossFn(LossFunction):
    """Cross-tokenizer distillation via TokenAligner projection matrices.

    For every alignment chunk (1:1, 1:many, many:1, many:many), projected
    student and teacher distributions are averaged over their spans,
    renormalized, and compared via KL. Per-chunk KLs are averaged over the
    valid chunks of the microbatch, then re-weighted by the local DP token
    share so the averaged-across-DP loss matches a single-device run.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION

    def __init__(
        self,
        cfg: CrossTokenizerDistillationLossConfig,
        token_aligner,
    ):
        from nemo_rl.algorithms.x_token import TokenAligner

        assert isinstance(token_aligner, TokenAligner)

        self.token_aligner = token_aligner

        # Unpack cfg (mirrors DistillationLossFn style).
        self.temperature = cfg.get("temperature", 1.0)
        self.vocab_topk = cfg.get("vocab_topk", 8192)
        self.reverse_kl = cfg.get("reverse_kl", False)
        self.exact_match_only = cfg.get("exact_token_match_only", False)
        self.use_gold_loss = cfg.get("gold_loss", False)
        self.use_xtoken_loss = cfg.get("xtoken_loss", False)
        self.ce_loss_scale = cfg.get("ce_loss_scale", 0.0)
        self.dynamic_loss_scaling = cfg.get("dynamic_loss_scaling", False)

        # Teacher-side tensors; populated before each training step.
        self._teacher_input_ids: Optional[torch.Tensor] = None
        self._aligned_pairs: Optional[list] = None

    # -------- data attachment --------------------------------------------------

    def set_cross_tokenizer_data(
        self,
        teacher_input_ids: torch.Tensor,
        aligned_pairs: list,
    ) -> None:
        """Attach teacher-side data before ``student_policy.train()``.

        Kept off the data dict because the worker's shape validator rejects
        mixed sequence dimensions across the batch.
        """
        self._teacher_input_ids = teacher_input_ids
        self._aligned_pairs = aligned_pairs

    # -------- alignment prep ---------------------------------------------------

    @staticmethod
    def _filter_pairs(
        aligned_pairs: list,
        batch_size: int,
        student_seq_len: int,
        teacher_seq_len: int,
        exact_match_only: bool,
    ) -> tuple[list[list[tuple]], int]:
        """Drop alignment pairs unusable for this microbatch.

        Rejects chunks with an unmapped (-1) position, that run past the
        truncated sequence length, or — under ``exact_match_only`` — that
        are not strict 1:1. Returns surviving pairs plus the max chunk
        count across the batch (used to size chunk masks).
        """
        filtered: list[list[tuple]] = []
        total_chunks = 0
        for batch_idx in range(batch_size):
            batch_pairs: list[tuple] = []
            for pair in aligned_pairs[batch_idx]:
                _, _, s1_start, s1_end, s2_start, s2_end = pair[:6]
                if exact_match_only and (
                    s1_end - s1_start != 1 or s2_end - s2_start != 1
                ):
                    continue
                if s1_start == -1 or s2_start == -1:
                    continue
                if s1_end > student_seq_len or s2_end > teacher_seq_len:
                    continue
                batch_pairs.append(pair)
            filtered.append(batch_pairs)
            total_chunks = max(total_chunks, len(batch_pairs))
        return filtered, total_chunks

    @staticmethod
    def _build_chunk_masks(
        pairs_per_batch: list[list[tuple]],
        batch_size: int,
        student_seq_len: int,
        teacher_seq_len: int,
        total_chunks: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Boolean chunk masks ``(B, S, C)`` for both sides.

        Pairs with unmapped (-1) positions are skipped; callers that already
        ran :meth:`_filter_pairs` never hit that branch. The gold-loss path
        relies on it because it consumes the unfiltered alignment.
        """
        student_mask = torch.zeros(
            batch_size,
            student_seq_len,
            total_chunks,
            dtype=torch.bool,
            device=device,
        )
        teacher_mask = torch.zeros(
            batch_size,
            teacher_seq_len,
            total_chunks,
            dtype=torch.bool,
            device=device,
        )
        for batch_idx in range(batch_size):
            for chunk_idx, pair in enumerate(pairs_per_batch[batch_idx][:total_chunks]):
                _, _, s1_start, s1_end, s2_start, s2_end = pair[:6]
                if s1_start == -1 or s2_start == -1:
                    continue
                student_mask[batch_idx, s1_start:s1_end, chunk_idx] = True
                teacher_mask[batch_idx, s2_start:s2_end, chunk_idx] = True
        return student_mask, teacher_mask

    # -------- projection -------------------------------------------------------

    def _project_student_to_teacher(
        self,
        student_logits: torch.Tensor,
        teacher_vocab_size: int,
        global_top_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Map student logits ``(B, S, V_s)`` into the reduced teacher vocab.

        Uses the sparse projection matrix when available (memory-friendly),
        falling back to the dense matrix otherwise. Only the columns in
        ``global_top_indices`` are materialized.
        """
        student_probs = torch.softmax(student_logits / self.temperature, dim=-1)

        sparse_mat = getattr(self.token_aligner, "sparse_transformation_matrix", None)
        if sparse_mat is not None:
            reduced_sparse = sparse_mat.index_select(1, global_top_indices).coalesce()
            return self.token_aligner.project_token_likelihoods_instance(
                student_probs,
                None,
                None,
                None,
                device,
                use_sparse_format=True,
                sparse_matrix=reduced_sparse,
            )

        proj_values = self.token_aligner.likelihood_projection_matrix
        if getattr(self.token_aligner, "learnable", False):
            proj_values = self.token_aligner.transform_learned_matrix_instance(
                proj_values
            )
        projected_full = self.token_aligner.project_token_likelihoods_instance(
            student_probs,
            self.token_aligner.likelihood_projection_indices,
            proj_values,
            teacher_vocab_size,
            device,
            use_sparse_format=False,
        )
        return projected_full[:, :, global_top_indices]

    # -------- projection-based KL path ----------------------------------------

    def _projection_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
        teacher_vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Chunk-averaged KL in a reduced teacher vocab.

        1. pick top-``vocab_topk`` teacher columns by global-max logit;
        2. project student into those columns;
        3. log-softmax teacher over the same columns;
        4. BMM-average both sides over chunk spans, renormalize student;
        5. per-chunk KL scaled by ``T^2``, mean over valid chunks.
        """
        # 1. Global teacher-vocab selection.
        with torch.no_grad():
            if self.vocab_topk == 0 or self.vocab_topk >= teacher_vocab_size:
                global_top_indices = torch.arange(teacher_vocab_size, device=device)
            else:
                teacher_flat = teacher_logits.view(-1, teacher_vocab_size)
                importance = teacher_flat.max(dim=0)[0]
                _, global_top_indices = torch.topk(
                    importance,
                    k=min(self.vocab_topk, teacher_vocab_size),
                    dim=-1,
                )
                global_top_indices = global_top_indices.sort()[0]

        # 2. Project student and 3. log-softmax teacher in same vocab.
        student_proj = self._project_student_to_teacher(
            student_logits,
            teacher_vocab_size,
            global_top_indices,
            device,
        )
        teacher_reduced = teacher_logits[:, :, global_top_indices]
        teacher_log_probs = torch.log_softmax(
            teacher_reduced / self.temperature, dim=-1
        )
        del teacher_reduced

        # 4. Chunk-average, normalize.
        student_chunks = torch.bmm(
            student_mask.transpose(1, 2).to(student_proj.dtype),
            student_proj,
        )
        teacher_chunks = torch.bmm(
            teacher_mask.transpose(1, 2).to(teacher_log_probs.dtype),
            teacher_log_probs,
        )
        del student_proj, teacher_log_probs

        student_sizes = student_mask.sum(dim=1).unsqueeze(-1).to(student_chunks.dtype)
        teacher_sizes = teacher_mask.sum(dim=1).unsqueeze(-1).to(teacher_chunks.dtype)
        student_chunks = student_chunks / (student_sizes + 1e-10)
        teacher_chunks = teacher_chunks / (teacher_sizes + 1e-10)

        # Student: renormalize (BMM-avg of probs is sub-stochastic), then log-space
        # so we can feed it to kl_div(log_target=True).
        student_chunks = student_chunks / (
            student_chunks.sum(dim=-1, keepdim=True) + 1e-10
        )
        student_log_chunks = torch.log(student_chunks + 1e-10)

        chunk_valid = (student_sizes.squeeze(-1) > 0) & (teacher_sizes.squeeze(-1) > 0)

        # 5. Per-chunk KL scaled by T^2, mean over valid chunks.
        if self.reverse_kl:
            kl_elem = torch.nn.functional.kl_div(
                teacher_chunks,
                student_log_chunks,
                reduction="none",
                log_target=True,
            )
        else:
            kl_elem = torch.nn.functional.kl_div(
                student_log_chunks,
                teacher_chunks,
                reduction="none",
                log_target=True,
            )
        kl_per_chunk = kl_elem.sum(dim=-1) * (self.temperature**2) * chunk_valid

        num_valid_chunks = chunk_valid.sum()
        if num_valid_chunks > 0:
            return kl_per_chunk.sum() / num_valid_chunks
        return torch.tensor(0.0, device=device, requires_grad=True)

    # -------- gold-loss path ---------------------------------------------------

    def _compute_gold_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
        student_vocab_size: int,
        teacher_vocab_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        """Gold loss: common-vocab KL + uncommon-vocab sorted L1.

        Partitions the vocabulary into tokens with exact 1:1 projection edges
        ("common") and the rest ("uncommon"). Common tokens use KL on native
        log-probs (no projection). Uncommon tokens use L1 on sorted probs
        (Universal Likelihood Distillation). Mirrors
        ``tokenalign.py::compute_KL_loss_optimized`` gold-loss branch.
        """
        aligner = self.token_aligner
        if getattr(aligner, "likelihood_projection_indices", None) is None:
            raise ValueError(
                "gold_loss requires likelihood_projection_indices to be loaded"
            )

        projection_indices = aligner.likelihood_projection_indices
        projection_matrix = aligner.likelihood_projection_matrix
        if getattr(aligner, "learnable", False):
            projection_matrix = aligner.transform_learned_matrix_instance(
                projection_matrix
            )

        # Identify student tokens with an exact-map teacher target.
        sorted_values, sorted_indices_in_topk = torch.sort(
            projection_matrix,
            dim=-1,
            descending=True,
        )
        if self.use_xtoken_loss:
            has_exact_map = sorted_values[:, 0] >= 0.6
        else:
            has_exact_map = (sorted_values[:, 0] == 1.0) & (
                projection_indices[:, 1] == -1
            )

        student_indices_with_exact_map = torch.where(has_exact_map)[0]
        teacher_indices_for_exact_map = projection_indices[
            student_indices_with_exact_map,
            sorted_indices_in_topk[student_indices_with_exact_map, 0],
        ]

        # Resolve duplicate teacher targets: keep the strongest edge.
        student_to_teacher: dict[int, int] = {}
        teacher_to_student: dict[int, int] = {}
        for s_idx, t_idx in zip(
            student_indices_with_exact_map.tolist(),
            teacher_indices_for_exact_map.tolist(),
        ):
            if not (0 <= t_idx < teacher_vocab_size):
                continue
            if t_idx not in teacher_to_student or self.use_xtoken_loss:
                if t_idx in teacher_to_student:
                    prev_s = teacher_to_student[t_idx]
                    if sorted_values[prev_s, 0] >= sorted_values[s_idx, 0]:
                        continue
                    del student_to_teacher[prev_s]
                student_to_teacher[s_idx] = t_idx
                teacher_to_student[t_idx] = s_idx

        common_student_indices = sorted(student_to_teacher.keys())
        common_teacher_indices = [student_to_teacher[s] for s in common_student_indices]
        uncommon_student_indices = sorted(
            set(range(student_vocab_size)) - set(common_student_indices)
        )
        uncommon_teacher_indices = sorted(
            set(range(teacher_vocab_size)) - set(common_teacher_indices)
        )

        # log_softmax on full vocab BEFORE chunk averaging (matches tokenalign.py).
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        student_chunk_lp = torch.bmm(
            student_mask.transpose(1, 2).to(student_log_probs.dtype),
            student_log_probs,
        )
        teacher_chunk_lp = torch.bmm(
            teacher_mask.transpose(1, 2).to(teacher_log_probs.dtype),
            teacher_log_probs,
        )
        del student_log_probs, teacher_log_probs

        student_sizes = student_mask.sum(dim=1, keepdim=True).float().transpose(1, 2)
        teacher_sizes = teacher_mask.sum(dim=1, keepdim=True).float().transpose(1, 2)
        student_chunk_lp = student_chunk_lp / (student_sizes + 1e-10)
        teacher_chunk_lp = teacher_chunk_lp / (teacher_sizes + 1e-10)

        chunk_valid = (student_sizes.squeeze(-1) > 0) & (teacher_sizes.squeeze(-1) > 0)
        if not chunk_valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0

        # Tensorize common indices once (reused for loss and top-1 accuracy).
        cs_tensor = (
            torch.tensor(common_student_indices, device=device)
            if common_student_indices
            else None
        )
        ct_tensor = (
            torch.tensor(common_teacher_indices, device=device)
            if common_teacher_indices
            else None
        )

        # Part 1: KL on common (exact-map) vocab.
        loss_kl_common = torch.tensor(0.0, device=device, requires_grad=True)
        if cs_tensor is not None:
            s_common = student_chunk_lp[:, :, cs_tensor]
            t_common = teacher_chunk_lp[:, :, ct_tensor]
            if self.reverse_kl:
                kl_elem = torch.nn.functional.kl_div(
                    t_common,
                    s_common,
                    reduction="none",
                    log_target=True,
                )
            else:
                kl_elem = torch.nn.functional.kl_div(
                    s_common,
                    t_common,
                    reduction="none",
                    log_target=True,
                )
            kl_per_chunk = kl_elem.sum(dim=-1) * chunk_valid
            if chunk_valid.sum() > 0:
                loss_kl_common = kl_per_chunk.sum() / chunk_valid.sum()

        # Part 2: sorted-L1 on uncommon vocab.
        loss_l1_uncommon = torch.tensor(0.0, device=device, requires_grad=True)
        if uncommon_student_indices or uncommon_teacher_indices:
            bsz, n_chunks = student_chunk_lp.shape[0], student_chunk_lp.shape[1]
            s_uncommon = (
                student_chunk_lp[
                    :, :, torch.tensor(uncommon_student_indices, device=device)
                ]
                if uncommon_student_indices
                else torch.empty(bsz, n_chunks, 0, device=device)
            )
            t_uncommon = (
                teacher_chunk_lp[
                    :, :, torch.tensor(uncommon_teacher_indices, device=device)
                ]
                if uncommon_teacher_indices
                else torch.empty(bsz, n_chunks, 0, device=device)
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
                        s_sorted, _ = torch.topk(
                            s_probs,
                            k=max_uncommon_vocab,
                            dim=-1,
                            largest=True,
                        )
                    else:
                        s_sorted = torch.sort(s_probs, dim=-1, descending=True)[0]
                    if t_probs.shape[-1] > max_uncommon_vocab:
                        t_sorted, _ = torch.topk(
                            t_probs,
                            k=max_uncommon_vocab,
                            dim=-1,
                            largest=True,
                        )
                    else:
                        t_sorted = torch.sort(t_probs, dim=-1, descending=True)[0]
                    del s_probs, t_probs

                    min_len = min(s_sorted.shape[-1], t_sorted.shape[-1])
                    if min_len > 0:
                        l1_per_chunk = torch.nn.functional.l1_loss(
                            s_sorted[:, :min_len],
                            t_sorted[:, :min_len],
                            reduction="none",
                        ).sum(dim=-1)
                        loss_l1_uncommon = l1_per_chunk.mean()

        loss_total = (loss_kl_common + loss_l1_uncommon) * (self.temperature**2)

        # Top-1 accuracy on the common vocab (telemetry only).
        top1_accuracy = 0.0
        with torch.no_grad():
            if cs_tensor is not None and chunk_valid.any():
                s_valid_lp = student_chunk_lp[chunk_valid][:, cs_tensor]
                t_valid_lp = teacher_chunk_lp[chunk_valid][:, ct_tensor]
                matches = (
                    (s_valid_lp.argmax(dim=-1) == t_valid_lp.argmax(dim=-1))
                    .sum()
                    .item()
                )
                top1_accuracy = matches / chunk_valid.sum().item()

        return loss_total, top1_accuracy

    # -------- optional next-token CE auxiliary --------------------------------

    def _apply_ce_auxiliary(
        self,
        kl_loss: torch.Tensor,
        student_logits: torch.Tensor,
        input_ids_student: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Fold a next-token CE term into the KL loss when configured.

        - ``dynamic_loss_scaling``: rescale KL to match CE magnitude, then add
          CE (``kl*(ce/kl) + ce``).
        - otherwise: ``kl + ce_loss_scale * ce``.
        Returns ``(loss, ce_value)``; ``ce_value`` is 0.0 when CE is disabled.
        """
        if self.ce_loss_scale <= 0.0 and not self.dynamic_loss_scaling:
            return kl_loss, 0.0

        ce_loss = torch.nn.functional.cross_entropy(
            student_logits[:, :-1].reshape(-1, student_logits.shape[-1]),
            input_ids_student[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        ce_value = float(ce_loss.item())
        if self.dynamic_loss_scaling and kl_loss.item() > 0:
            dls_scale = ce_loss.item() / kl_loss.item()
            return kl_loss * dls_scale + ce_loss, ce_value
        return kl_loss + ce_loss * self.ce_loss_scale, ce_value

    # -------- DP-aware rescale -------------------------------------------------

    @staticmethod
    def _rescale_for_dp(
        loss: torch.Tensor,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        student_seq_len: int,
        global_valid_toks: torch.Tensor,
    ) -> torch.Tensor:
        """Re-weight a scalar chunk-mean loss by the local DP token share."""
        max_len = min(token_mask.shape[1] - 1, student_seq_len)
        local_mask = token_mask[:, 1 : max_len + 1] * sample_mask.unsqueeze(-1)
        local_valid_toks = local_mask.sum()
        if local_valid_toks > 0 and global_valid_toks > 0:
            return loss * local_valid_toks / global_valid_toks
        return loss * 0.0

    # -------- entry point ------------------------------------------------------

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
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute cross-tokenizer chunk-averaged KL (+ optional CE)."""
        # --- 0. inputs & pre-conditions ---
        input_ids_student = data["input_ids"]
        batch_size = input_ids_student.shape[0]

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

        # DTensor unwrap + float32 upcast for numerical stability.
        student_logits = (
            next_token_logits.full_tensor()
            if isinstance(next_token_logits, torch.distributed.tensor.DTensor)
            else next_token_logits
        ).to(torch.float32)
        teacher_logits_f32 = (
            teacher_logits.full_tensor()
            if isinstance(teacher_logits, torch.distributed.tensor.DTensor)
            else teacher_logits
        ).to(torch.float32)

        if teacher_logits_f32.shape[-1] == 0:
            raise ValueError(
                f"Teacher logits have vocab dimension 0 (shape={teacher_logits_f32.shape}). "
                "Cross-tokenizer distillation requires full teacher logits "
                "(topk_logits=None on the teacher forward pass)."
            )

        # --- 1. microbatch slice of aligned pairs ---
        aligned_pairs = self._aligned_pairs
        if mb_idx is not None and mbs is not None:
            aligned_pairs = aligned_pairs[mb_idx * mbs : mb_idx * mbs + batch_size]

        device = student_logits.device
        self.token_aligner = self.token_aligner.to(device)

        student_seq_len = student_logits.shape[1]
        teacher_seq_len = teacher_logits_f32.shape[1]
        student_vocab_size = student_logits.shape[-1]
        teacher_vocab_size = teacher_logits_f32.shape[-1]

        # --- 2. filter pairs and early-exit if nothing survives ---
        filtered_pairs, total_chunks = self._filter_pairs(
            aligned_pairs,
            batch_size,
            student_seq_len,
            teacher_seq_len,
            self.exact_match_only,
        )
        if total_chunks == 0:
            # Zero loss that is still connected to student_logits via the
            # autograd graph. If we returned an unconnected
            # ``torch.tensor(0.0, requires_grad=True)`` instead, the
            # subsequent ``backward()`` on this rank would NOT trigger any
            # DDP all-reduce, while peer ranks with non-zero chunks still
            # do — causing a cross-rank collective mismatch and an NCCL
            # hang on the very next synchronization.
            zero_loss = (student_logits.to(torch.float32) * 0.0).sum()
            # Match the shape of the success-path metrics so downstream
            # microbatch aggregation (which indexes ``loss_metrics[...]``
            # without a default) doesn't KeyError. ``num_valid_samples=0``
            # tells the worker to skip this microbatch in its logging loop.
            return zero_loss, {
                "loss": 0.0,
                "kl_loss": 0.0,
                "ce_loss": 0.0,
                "topk_accuracy": 0.0,
                "num_valid_samples": 0,
                "num_chunks": 0,
                "alignment_density": 0.0,
            }

        # --- 3. core KL: gold-loss vs. projection path ---
        if self.use_gold_loss:
            # Gold loss uses the UNFILTERED alignment, capped at
            # min(S_s, S_t) chunks (matches tokenalign.py exactly).
            gold_student_mask, gold_teacher_mask = self._build_chunk_masks(
                aligned_pairs,
                batch_size,
                student_seq_len,
                teacher_seq_len,
                min(student_seq_len, teacher_seq_len),
                device,
            )
            kl_loss, top1_accuracy = self._compute_gold_loss(
                student_logits,
                teacher_logits_f32,
                gold_student_mask,
                gold_teacher_mask,
                student_vocab_size,
                teacher_vocab_size,
                device,
            )
        else:
            student_mask, teacher_mask = self._build_chunk_masks(
                filtered_pairs,
                batch_size,
                student_seq_len,
                teacher_seq_len,
                total_chunks,
                device,
            )
            kl_loss = self._projection_kl(
                student_logits,
                teacher_logits_f32,
                student_mask,
                teacher_mask,
                teacher_vocab_size,
                device,
            )
            top1_accuracy = 0.0

        # --- 4. optional next-token CE auxiliary ---
        loss, ce_loss_value = self._apply_ce_auxiliary(
            kl_loss,
            student_logits,
            input_ids_student,
        )

        # --- 5. DP-aware rescale ---
        loss = self._rescale_for_dp(
            loss,
            data["token_mask"],
            data["sample_mask"],
            student_seq_len,
            global_valid_toks,
        )

        num_valid = sum(len(fp) for fp in filtered_pairs)
        metrics = {
            "loss": float(loss.item()) if loss.ndim == 0 else loss,
            "kl_loss": float(kl_loss.item()) if kl_loss.ndim == 0 else kl_loss,
            "ce_loss": ce_loss_value,
            "topk_accuracy": top1_accuracy,
            "num_valid_samples": int(batch_size),
            "num_chunks": num_valid,
            "alignment_density": num_valid / max(1, batch_size * student_seq_len),
        }
        return loss, metrics
