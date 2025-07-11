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
import enum
from typing import Any, Tuple, TypedDict

import torch

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.utils import (
    calculate_kl_penalty_joschu2020,
    compute_token_level_entropy,
    compute_token_logprobs,
    masked_mean,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
)


class LossType(enum.Enum):
    TOKEN_LEVEL = "token_level"
    SEQUENCE_LEVEL = "sequence_level"


class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    token_level_loss: bool
    use_generation_logprobs_in_ppo_baseline: bool


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

    def __init__(self, cfg: ClippedPGLossConfig):
        self.ratio_clip_min = cfg["ratio_clip_min"]
        self.ratio_clip_max = cfg["ratio_clip_max"]
        self.ratio_clip_c = cfg["ratio_clip_c"]  # set to None to disable dual-clipping
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)
        self.use_kl_fix = cfg.get("use_kl_fix", False)
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.use_importance_sampling_correction = cfg[
            "use_importance_sampling_correction"
        ]

        self.use_generation_logprobs_in_ppo_baseline = cfg[
            "use_generation_logprobs_in_ppo_baseline"
        ]

        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg["token_level_loss"] else LossType.SEQUENCE_LEVEL
        )
        self.use_stable_loss = cfg.get("use_stable_loss", False)

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"][:, 1:]
        prev_logprobs = data["prev_logprobs"][:, 1:]
        generation_logprobs = data["generation_logprobs"][:, 1:]
        reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]

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

        next_token_logits = next_token_logits.to(torch.float32)
        curr_logprobs = compute_token_logprobs(next_token_logits, data["input_ids"])

        # Calculate KL regularization.
        if self.use_on_policy_kl_approximation:
            # See: docs/guides/grpo.md#on-policy-kl-approximation
            kl_importance_weights = torch.exp(
                curr_logprobs - generation_logprobs
            ).detach()
            kl_importance_weights = torch.nan_to_num(
                kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            kl_importance_weights = torch.ones_like(curr_logprobs)

        if self.use_kl_fix:
            with torch.no_grad():
                p_t = (curr_logprobs.detach() - reference_policy_logprobs) * mask
                p_t = p_t.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

                kl_for_metric = kl_importance_weights * calculate_kl_penalty_joschu2020(
                    logprobs_policy=curr_logprobs,
                    logprobs_reference=reference_policy_logprobs,
                )

            kl = p_t * curr_logprobs
        else:
            kl = kl_importance_weights * calculate_kl_penalty_joschu2020(
                logprobs_policy=curr_logprobs,
                logprobs_reference=reference_policy_logprobs,
            )
            kl_for_metric = kl

        if self.loss_type == LossType.TOKEN_LEVEL:
            kl = masked_mean(kl, mask, global_normalization_factor=global_valid_toks)
            kl_for_metric = masked_mean(
                kl_for_metric, mask, global_normalization_factor=global_valid_toks
            )
        else:
            kl = masked_mean(
                masked_mean(kl, token_mask, dim=-1),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )
            kl_for_metric = masked_mean(
                masked_mean(kl_for_metric, token_mask, dim=-1),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )

        # Calculate clipped loss function if ppo ratio is enabled.
        if not self.disable_ppo_ratio:
            if self.use_generation_logprobs_in_ppo_baseline:
                ratios = (curr_logprobs - generation_logprobs).exp()
            else:
                ratios = (curr_logprobs - prev_logprobs).exp()

            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_clip_min, 1.0 + self.ratio_clip_max
            )

        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        if self.use_stable_loss:
            loss_stable = -(advantages * ratios_clamped.detach() * curr_logprobs)
            clip_loss = loss_stable

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

        # See: docs/guides/grpo.md#importance-sampling-correction
        actor_importance_weights = torch.exp(prev_logprobs - generation_logprobs)
        actor_importance_weights = torch.nan_to_num(
            actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
        )
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

        # See: docs/guides/grpo.md#sampling-importance-ratio
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
            log_probs_mean = masked_mean(
                curr_logprobs.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            )

            ratio_to_compare = (
                generation_logprobs
                if self.use_generation_logprobs_in_ppo_baseline
                else prev_logprobs
            ).detach()
            sequence_level_ratios = (
                ((curr_logprobs.detach() - ratio_to_compare) * mask).sum(-1).exp().sum()
            )

        kl_for_loss = self.reference_policy_kl_penalty * kl
        loss = actor_loss + kl_for_loss

        with torch.no_grad():
            token_level_entropy = masked_mean(
                compute_token_level_entropy(next_token_logits),
                mask,
                global_normalization_factor=global_valid_toks,
            )
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

            if self.disable_ppo_ratio:
                clipped_min = 0
                clipped_max = 0
            else:
                advantages_neg = advantages < 0
                advantages_pos = advantages > 0

                clipped_min_mask = (
                    (ratios < 1 - self.ratio_clip_min) * advantages_neg * mask
                )
                clipped_max_mask = (
                    (ratios > 1 + self.ratio_clip_max) * advantages_pos * mask
                )

                clipped_min = clipped_min_mask.sum().item()
                clipped_max = clipped_max_mask.sum().item()

                clamped_min_ratios, clamped_max_ratios = 0, 0
                if clipped_min > 0:
                    clamped_min_ratios = ratios[clipped_min_mask.bool()].sum().item()
                if clipped_max > 0:
                    clamped_max_ratios = ratios[clipped_max_mask.bool()].sum().item()

        # If you provided a global_valid_{seqs/toks}, all metrics here are globally normalized
        # by either sequence or token count, depending on particular metric.
        # To get the true metric, you'll need to sum over the microbatch.
        return (
            loss,
            {
                "loss": loss.item(),
                "log_probs_mean": log_probs_mean.item(),
                "sequence_level_ratios": sequence_level_ratios.item(),
                "tokens_min_clipped": clipped_min,
                "tokens_max_clipped": clipped_max,
                "clamped_min_ratios": clamped_min_ratios,
                "clamped_max_ratios": clamped_max_ratios,
                "probs_ratio": probs_ratio,
                "probs_ratio_clamped": probs_ratio_clamped,
                "kl_penalty": kl.item(),
                "kl_penalty_for_metric": kl_for_metric.item(),
                "kl_penalty_for_loss": kl_for_loss.item(),
                "token_mult_prob_error": mult_prob_error,
                "sampling_importance_ratio": sample_importance_ratio.item(),
                "num_valid_samples": sample_mask.sum().item(),
                "seq_approx_entropy": seq_entropy_approx.item(),
                "token_level_entropy": token_level_entropy.item(),
            },
        )


class NLLLoss(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor,
        dpo_loss: bool = False,
        dpo_average_log_probs: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        next_token_logits = next_token_logits.to(torch.float32)

        # Gather the logprobs for the actual next tokens
        if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"]
            )
        else:
            next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        if dpo_loss:
            ## shape: [batch_size]
            num_unmasked_tokens = torch.sum(mask, -1)
            ## multiply by sample_mask to zero out invalid samples
            loss = -torch.sum(token_logprobs * mask, dim=-1)
            if dpo_average_log_probs:
                loss = loss / num_unmasked_tokens.clamp(min=1)
        else:
            ## single scalar loss
            ## scale by the total number of tokens in the batch
            loss = -masked_mean(
                token_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens": mask.sum().item(),
            "num_valid_samples": sample_mask.sum().item(),
        }


class DPOLossConfig(TypedDict):
    reference_policy_kl_penalty: float
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


class DPOLossFn(LossFunction):
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
        Tuple[torch.Tensor, dict]: A tuple containing:
            - The total loss value
            - A dictionary with metrics including:
                - loss: Total loss value
                - sft_loss: SFT loss component
                - preference_loss: Preference loss component
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    def __init__(self, cfg: DPOLossConfig):
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight = cfg["preference_loss_weight"]
        self.sft_loss_weight = cfg["sft_loss_weight"]
        self.preference_average_log_probs = cfg["preference_average_log_probs"]
        self.sft_average_log_probs = cfg["sft_average_log_probs"]
        self.sft_loss = NLLLoss()

        self.loss_type = LossType.SEQUENCE_LEVEL

    def split_output_tensor(self, tensor: torch.Tensor):
        return tensor[::2], tensor[1::2]

    def preference_loss(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: torch.Tensor,
    ) -> torch.Tensor:
        ## TODO(@ashors): there's some duplicate code here with the NLLLoss function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]

        next_token_logits = next_token_logits.to(torch.float32)
        if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"]
            )
        else:
            next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        ref_logprobs = data["reference_policy_logprobs"][:, :-1]

        diff = (token_logprobs - ref_logprobs) * token_mask

        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)

        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(
                self.reference_policy_kl_penalty * rewards_delta
            )
            * sample_mask[::2]
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
        next_token_logits: torch.Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, dict]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            assert global_valid_toks is not None, (
                "global_valid_toks must be provided for SFT loss"
            )
            sft_loss, _ = self.sft_loss(
                next_token_logits,
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
        ) = self.preference_loss(next_token_logits, data, global_valid_seqs)

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
