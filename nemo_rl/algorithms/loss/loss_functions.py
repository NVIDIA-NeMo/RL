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
import sys
if sys.version_info >= (3, 11):
    from typing import Any, NotRequired, Optional, TypedDict, TypeVar
else:
    from typing import Any, Optional, TypedDict, TypeVar
    from typing_extensions import NotRequired

import torch
import torch.distributed

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.algorithms.utils import calculate_kl, masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    ChunkedDistributedEntropy,
    ChunkedDistributedGatherLogprob,
    _compute_distributed_log_softmax,
    _get_tokens_on_this_cp_rank,
    allgather_cp_sharded_tensor,
    from_parallel_logits_to_logprobs,
    gather_logits_at_global_indices,
    get_logprobs_from_vocab_parallel_logits,
)
from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

Tensor = TypeVar("Tensor", bound=torch.Tensor)


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

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"][:, 1:]
        prev_logprobs = data["prev_logprobs"][:, 1:]
        generation_logprobs = data["generation_logprobs"][:, 1:]
        reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
        seq_index = data.get("seq_index", None)

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

        next_token_logits = next_token_logits.to(torch.float32)

        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            curr_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            curr_logprobs = curr_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_token_logits_wo_last = next_token_logits[
                :, :-1
            ]  # Remove last position's logits
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits_wo_last, dim=-1
            )
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            curr_logprobs = next_token_logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # Calculate KL regularization.
        if self.reference_policy_kl_penalty != 0:
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
            kl = (
                kl_importance_weights
                * self.reference_policy_kl_penalty
                * calculate_kl(
                    logprobs=curr_logprobs,
                    logprobs_reference=reference_policy_logprobs,
                    kl_type=self.reference_policy_kl_type,
                    input_clamp_value=self.kl_input_clamp_value,
                    output_clamp_value=self.kl_output_clamp_value,
                )
            )
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
        # TIS see https://fengyao.notion.site/off-policy-rl
        if self.truncated_importance_sampling_ratio is not None:
            actor_importance_weights_expanded = torch.clamp(
                actor_importance_weights_expanded,
                max=self.truncated_importance_sampling_ratio,
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
            },
        )


class NLLLoss(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        dpo_loss: bool = False,
        dpo_average_log_probs: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)
        seq_index = data.get("seq_index", None)

        next_token_logits = next_token_logits.to(torch.float32)

        # Gather the logprobs for the actual next tokens
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
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


class PreferenceLossDataDict(TypedDict):
    """Required keys for the preference loss function."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class PreferenceLoss(LossFunction):
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

    def __init__(self):
        self.loss_type = LossType.SEQUENCE_LEVEL

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
        rewards: Tensor,
        data: BatchedDataDict[PreferenceLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sample_mask = data["sample_mask"]

        rewards = rewards.squeeze(-1)

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


class DPOLossFn(PreferenceLoss):
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

    def __init__(self, cfg: DPOLossConfig):
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight = cfg["preference_loss_weight"]
        self.sft_loss_weight = cfg["sft_loss_weight"]
        self.preference_average_log_probs = cfg["preference_average_log_probs"]
        self.sft_average_log_probs = cfg["sft_average_log_probs"]
        self.sft_loss = NLLLoss()

        self.loss_type = LossType.SEQUENCE_LEVEL

    def _dpo_loss(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ## TODO(@ashors): there's some duplicate code here with the NLLLoss function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        seq_index = data.get("seq_index", None)

        next_token_logits = next_token_logits.to(torch.float32)
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
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

        return self._preference_loss(
            rewards, sample_mask, global_valid_seqs, self.reference_policy_kl_penalty
        )

    # TODO a cleaner typing fix would be required (probably that DPOLossFn should not inherit from PreferenceLoss)
    def __call__(  # type: ignore
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
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
        ) = self._dpo_loss(
            next_token_logits,
            data,
            global_valid_seqs,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )

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


class SequencePackingLossWrapper:
    def __init__(
        self,
        loss_fn: LossFunction,
        cu_seqlens_q: Tensor,
        cu_seqlens_q_padded: Optional[Tensor] = None,
    ):
        self.loss_fn = loss_fn
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_q_padded = cu_seqlens_q_padded

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Wraps a loss function to handle sequence packing by doing one sequence at a time to avoid excessive padding."""
        unpadded_cu_seqlens = self.cu_seqlens_q
        unpadded_seq_lengths = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
        if self.cu_seqlens_q_padded is not None:
            padded_cu_seqlens = self.cu_seqlens_q_padded
            padded_seq_lengths = (
                self.cu_seqlens_q_padded[1:] - self.cu_seqlens_q_padded[:-1]
            )
        else:
            padded_cu_seqlens = unpadded_cu_seqlens
            padded_seq_lengths = unpadded_seq_lengths
        seq_starts = padded_cu_seqlens[:-1]
        seq_ends = padded_cu_seqlens[1:]

        loss_accum = 0
        metrics_accum = {}
        for seq_idx in range(len(seq_starts)):
            seq_start = seq_starts[seq_idx].item()
            seq_end = seq_ends[seq_idx].item()

            # get sequence and unpad all 'data' tensors. The data dict is a BatchedDataDict of unpacked tensors
            seq_data = data.slice(seq_idx, seq_idx + 1)
            unpadded_seq_data = {}
            for k, v in seq_data.items():
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[1] > 1:
                    unpadded_seq_data[k] = v[:, : unpadded_seq_lengths[seq_idx]]
                else:
                    unpadded_seq_data[k] = v

            # get next_token_logits
            cp_size = (
                1
                if context_parallel_group is None
                else torch.distributed.get_world_size(context_parallel_group)
            )
            logit_start = seq_start // cp_size
            logit_end = (seq_start + padded_seq_lengths[seq_idx]) // cp_size
            logit_length = logit_end - logit_start
            next_token_logits_slice = next_token_logits.narrow(
                1, logit_start, logit_length
            )

            loss, metrics = self.loss_fn(
                next_token_logits_slice,
                unpadded_seq_data,
                global_valid_seqs,
                global_valid_toks,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
            loss_accum += loss
            for k, v in metrics.items():
                if k not in metrics_accum:
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        metrics_accum[k] = float("inf")
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        metrics_accum[k] = float("-inf")
                    else:
                        metrics_accum[k] = 0

                val = v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v

                # Skip inf/-inf sentinel values (from sequences with no valid tokens)
                if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                    if not math.isinf(val):
                        metrics_accum[k] = min(metrics_accum[k], val)
                elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                    if not math.isinf(val):
                        metrics_accum[k] = max(metrics_accum[k], val)
                else:
                    metrics_accum[k] += val

        return loss_accum, metrics_accum


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

    def __init__(self, cfg: DistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg.get("mixed_kl_weight", 0.5)
        self.zero_outside_topk = cfg["zero_outside_topk"]
        self.log_infinitesimal = -100
        self.loss_type = LossType.TOKEN_LEVEL

        assert self.kl_type in ["forward", "reverse", "mixed"], "Invalid KL type"
        assert 0 <= self.mixed_kl_weight <= 1, (
            "Invalid mixed KL weight"
        )

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: DistillationLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        teacher_logits: Optional = None,
        mb_idx: Optional[int] = None,
        mbs: Optional[int] = None,
        teacher_topk_indices_ipc: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute distillation loss between teacher and student logits."""
        # Basic shapes
        input_ids = data["input_ids"]
        batch_size = input_ids.shape[0]

        # CP support: get CP group and size.
        # Prefer the explicitly-passed group; fall back to the DTensor device
        # mesh so the IPC path works even when the caller doesn't pass it.
        cp_group = context_parallel_group
        if cp_group is None and isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            mesh = next_token_logits.device_mesh
            if mesh.mesh_dim_names is not None and "cp" in mesh.mesh_dim_names:
                cp_group = mesh.get_group("cp")
        cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)

        # Ensure float32 for stability (match other losses)
        next_token_logits = next_token_logits.to(torch.float32)
        per_token_kl = None

        # ===== IPC PATH: teacher logits passed as pre-reconstructed tensor =====
        if teacher_logits is not None and teacher_topk_indices_ipc is not None:
            # Resolve TP-local student logits
            if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
                device_mesh = next_token_logits.device_mesh
                tp_group = device_mesh.get_group("tp")
                tp_rank = tp_group.rank()
                local_student_logits = next_token_logits.to_local()
                V_local = int(local_student_logits.shape[-1])
                vocab_start_index = tp_rank * V_local
                vocab_end_index = (tp_rank + 1) * V_local
            else:
                tp_group = None
                tp_rank = 0
                local_student_logits = next_token_logits
                V_local = int(local_student_logits.shape[-1])
                vocab_start_index = 0
                vocab_end_index = V_local

            with torch.no_grad():
                if mb_idx is not None and mbs is not None:
                    mb_start = mb_idx * mbs
                    mb_end = mb_start + mbs
                    teacher_topk_logprobs = teacher_logits[mb_start:mb_end, :, :].clone().detach()
                    topk_indices = teacher_topk_indices_ipc[mb_start:mb_end, :, :].clone().detach()
                else:
                    teacher_topk_logprobs = teacher_logits.clone().detach()
                    topk_indices = teacher_topk_indices_ipc.clone().detach()
                teacher_topk_logprobs = teacher_topk_logprobs.to(device=local_student_logits.device)
                topk_indices = topk_indices.to(device=local_student_logits.device)

            # Gather student log probs at teacher's top-k global indices
            if tp_group is not None:
                S_local = int(local_student_logits.shape[1])
                chunk_size = max(1, min(S_local, 1024))
                student_topk_logprobs = ChunkedDistributedGatherLogprob.apply(
                    local_student_logits,
                    topk_indices,
                    vocab_start_index,
                    vocab_end_index,
                    chunk_size,
                    tp_group,
                    False,
                )
            else:
                student_logprobs = torch.nn.functional.log_softmax(
                    local_student_logits, dim=-1
                )
                student_topk_logprobs = torch.gather(
                    student_logprobs, dim=-1, index=topk_indices
                )
                del student_logprobs
            del local_student_logits

            if self.kl_type == "reverse":
                teacher_topk_logprobs, student_topk_logprobs = student_topk_logprobs, teacher_topk_logprobs

            # Build (k+1)-dim distributions with a "rest" bucket
            teacher_topk_probs = teacher_topk_logprobs.exp()
            teacher_rest_prob = (1.0 - teacher_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
            teacher_probs_full = torch.cat([teacher_topk_probs, teacher_rest_prob], dim=-1)
            teacher_logprobs_full = torch.cat([teacher_topk_logprobs, teacher_rest_prob.log()], dim=-1)

            student_topk_probs = student_topk_logprobs.exp()
            student_rest_prob = (1.0 - student_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
            student_logprobs_full = torch.cat([student_topk_logprobs, student_rest_prob.log()], dim=-1)

            per_token_kl = (teacher_probs_full * (teacher_logprobs_full - student_logprobs_full)).sum(dim=-1)

            del teacher_topk_logprobs, teacher_topk_probs, teacher_rest_prob
            del teacher_probs_full, teacher_logprobs_full
            del student_topk_logprobs, student_topk_probs, student_rest_prob
            del student_logprobs_full, topk_indices

            # Next-token alignment
            per_token_kl = per_token_kl[:, :-1]

        # ===== FULL-LOGPROB IPC PATH: teacher provides full vocab logprobs via IPC =====
        elif teacher_logits is not None:
            # Resolve TP-local student logits
            if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
                device_mesh = next_token_logits.device_mesh
                tp_group = device_mesh.get_group("tp")
                local_student_logits = next_token_logits.to_local()
            else:
                tp_group = None
                local_student_logits = next_token_logits

            with torch.no_grad():
                if mb_idx is not None and mbs is not None:
                    mb_start_index = mb_idx * mbs
                    mb_end_index = mb_start_index + mbs
                    teacher_logprobs_local = teacher_logits[mb_start_index:mb_end_index, :, :].clone().detach()
                else:
                    teacher_logprobs_local = teacher_logits.clone().detach()
                teacher_logprobs_local = teacher_logprobs_local.to(device=local_student_logits.device)

            if tp_group is not None:
                # Differentiable distributed log-softmax for student logits.
                # The normalization constants are computed under no_grad,
                # but the final log-softmax is built with differentiable ops
                # so autograd can back-propagate through the student model.
                with torch.no_grad():
                    logits_max = torch.amax(local_student_logits, dim=-1, keepdim=True)
                    torch.distributed.all_reduce(
                        logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
                    )
                shifted = local_student_logits - logits_max
                local_sum_exp = shifted.exp().sum(-1, keepdim=True)
                global_sum_exp = torch.distributed.nn.functional.all_reduce(
                    local_sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group
                )
                student_logprobs_local = shifted - global_sum_exp.log()
                del shifted, local_sum_exp, global_sum_exp
            else:
                student_logprobs_local = torch.nn.functional.log_softmax(local_student_logits, dim=-1)

            per_token_kl = teacher_logprobs_local.exp() * (teacher_logprobs_local - student_logprobs_local)
            per_token_kl = per_token_kl.sum(-1)
            del teacher_logprobs_local, student_logprobs_local, local_student_logits

            if tp_group is not None:
                per_token_kl = torch.distributed.nn.functional.all_reduce(
                    per_token_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group
                )

            # Next-token alignment
            per_token_kl = per_token_kl[:, :-1]

        # ===== STANDARD PATH: teacher top-k in data dict =====
        else:
            teacher_topk_logits = data["teacher_topk_logits"]  # [B, S, k]
            teacher_topk_indices = data["teacher_topk_indices"]  # [B, S, k]

            if teacher_topk_indices.shape[-1] <= 0:
                raise ValueError(
                    f"topk must be positive, got {teacher_topk_indices.shape[-1]}. "
                    "topk=0 is not supported as it would result in empty tensor operations."
                )

            # Determine processing path and setup variables
            if vocab_parallel_group is not None:
                assert vocab_parallel_rank is not None, (
                    "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
                )
                V_local = int(next_token_logits.shape[-1])
                vocab_start_index = vocab_parallel_rank * V_local
                vocab_end_index = (vocab_parallel_rank + 1) * V_local
                parallel_group = vocab_parallel_group
                logits_tensor = next_token_logits
            elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
                device_mesh = next_token_logits.device_mesh
                tp_group = device_mesh.get_group("tp")
                tp_rank = tp_group.rank()
                local_student_logits = next_token_logits.to_local()
                V_local = int(local_student_logits.shape[-1])
                vocab_start_index = tp_rank * V_local
                vocab_end_index = (tp_rank + 1) * V_local
                parallel_group = tp_group
                logits_tensor = local_student_logits
                teacher_topk_indices = teacher_topk_indices.to(local_student_logits.device)
                if (
                    device_mesh.mesh_dim_names is not None
                    and "cp" in device_mesh.mesh_dim_names
                ):
                    cp_group = device_mesh.get_group("cp")
                    cp_size = cp_group.size()
                else:
                    cp_group = None
                    cp_size = 1
            else:
                parallel_group = None
                logits_tensor = next_token_logits

            # Process based on zero_outside_topk setting
            if self.zero_outside_topk and parallel_group is not None:
                indices_local = teacher_topk_indices
                pad_len = 0
                if cp_size > 1:
                    pad_len = logits_tensor.shape[1] * cp_size - indices_local.shape[1]
                    if pad_len > 0:
                        indices_local = torch.nn.functional.pad(
                            indices_local, (0, 0, 0, pad_len), value=0
                        )
                    cp_rank = torch.distributed.get_rank(cp_group)
                    indices_local = _get_tokens_on_this_cp_rank(
                        indices_local, cp_rank, cp_size, seq_dim=1
                    )

                S_local = int(logits_tensor.shape[1])
                chunk_size = max(1, min(S_local, 1024))
                student_topk_logprobs = ChunkedDistributedGatherLogprob.apply(  # type: ignore
                    logits_tensor,
                    indices_local,
                    vocab_start_index,
                    vocab_end_index,
                    chunk_size,
                    parallel_group,
                    False,
                )

                if self.kl_type != "forward":
                    H_all = ChunkedDistributedEntropy.apply(  # type: ignore
                        logits_tensor,
                        chunk_size,
                        parallel_group,
                        False,
                    )

                if cp_size > 1:
                    student_topk_logprobs = allgather_cp_sharded_tensor(
                        student_topk_logprobs, cp_group, seq_dim=1
                    )
                    if self.kl_type != "forward":
                        H_all = allgather_cp_sharded_tensor(H_all, cp_group, seq_dim=1)
                    if pad_len > 0:
                        student_topk_logprobs = student_topk_logprobs[:, :-pad_len, :]
                        if self.kl_type != "forward":
                            H_all = H_all[:, :-pad_len]
            elif self.zero_outside_topk:
                student_logprobs = torch.nn.functional.log_softmax(logits_tensor, dim=-1)
                student_topk_logprobs = student_logprobs.gather(
                    dim=-1, index=teacher_topk_indices.to(student_logprobs.device)
                )
                if self.kl_type != "forward":
                    H_all = (student_logprobs.exp() * student_logprobs).sum(-1)
            else:
                if (parallel_group is not None) or (cp_size > 1):
                    student_topk_logits = gather_logits_at_global_indices(
                        logits_tensor,
                        teacher_topk_indices,
                        tp_group=parallel_group,
                        cp_group=cp_group,
                        vocab_start_index=(
                            vocab_start_index if parallel_group is not None else 0
                        ),
                        vocab_end_index=(
                            vocab_end_index
                            if parallel_group is not None
                            else int(logits_tensor.shape[-1])
                        ),
                    )
                else:
                    student_topk_logits = logits_tensor.gather(
                        dim=-1, index=teacher_topk_indices.to(logits_tensor.device)
                    )
                student_topk_logprobs = torch.nn.functional.log_softmax(
                    student_topk_logits, dim=-1
                )

            teacher_topk_logits = teacher_topk_logits.to(
                student_topk_logprobs.device, dtype=student_topk_logprobs.dtype
            )

            # Use the teacher's top-k values as log-probabilities directly
            # (get_topk_logits returns log-probs when using DTensor/TP).
            # Build (k+1)-dim distributions with a "rest" bucket to match
            # the IPC path and preserve the true probability mass outside top-k.
            teacher_topk_logprobs = teacher_topk_logits

            # Single point of next-token alignment after TP/CP processing
            teacher_topk_logprobs = teacher_topk_logprobs[:, :-1, :]
            student_topk_logprobs = student_topk_logprobs[:, :-1, :]

            if self.kl_type == "reverse":
                teacher_topk_logprobs, student_topk_logprobs = student_topk_logprobs, teacher_topk_logprobs

            teacher_topk_probs = teacher_topk_logprobs.exp()
            teacher_rest_prob = (1.0 - teacher_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
            teacher_probs_full = torch.cat([teacher_topk_probs, teacher_rest_prob], dim=-1)
            teacher_logprobs_full = torch.cat([teacher_topk_logprobs, teacher_rest_prob.log()], dim=-1)

            student_topk_probs = student_topk_logprobs.exp()
            student_rest_prob = (1.0 - student_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
            student_logprobs_full = torch.cat([student_topk_logprobs, student_rest_prob.log()], dim=-1)

            per_token_kl = (teacher_probs_full * (teacher_logprobs_full - student_logprobs_full)).sum(dim=-1)

            del teacher_topk_probs, teacher_rest_prob, teacher_probs_full, teacher_logprobs_full
            del student_topk_probs, student_rest_prob, student_logprobs_full

        # Masking and reduction
        if "token_mask" in data and "sample_mask" in data:
            token_mask = data["token_mask"][:, 1:]
            sample_mask = data["sample_mask"]
            max_len = per_token_kl.shape[1]
            if cp_size > 1:
                cp_rank = torch.distributed.get_rank(cp_group)
                S_local = max_len + 1
                start = cp_rank * S_local
                token_mask = token_mask[:, start:start + max_len]
            else:
                token_mask = token_mask[:, :max_len]
            mask = token_mask * sample_mask.unsqueeze(-1)
            kl_loss = masked_mean(
                per_token_kl,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            kl_loss = per_token_kl.mean()

        metrics = {
            "loss": float(kl_loss.item()) if kl_loss.ndim == 0 else kl_loss,
            "num_valid_samples": int(batch_size),
        }

        return kl_loss, metrics


# =============================================================================
# Cross-Tokenizer Distillation Loss (via TokenAligner)
# =============================================================================


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
        from nemo_rl.algorithms.x_token import TokenAligner
        assert isinstance(token_aligner, TokenAligner)
        self.token_aligner = token_aligner
        self.cfg = cfg
        self.loss_type = LossType.TOKEN_LEVEL
        self._teacher_input_ids = None
        self._aligned_pairs = None

    def set_cross_tokenizer_data(
        self,
        teacher_input_ids: torch.Tensor,
        aligned_pairs: list,
    ):
        """Store teacher-side data before each training step.

        Called from the training loop before student_policy.train().
        The worker never sees these tensors in shape validation.
        """
        self._teacher_input_ids = teacher_input_ids
        self._aligned_pairs = aligned_pairs

    def _project_student_to_teacher(
        self,
        student_logits: torch.Tensor,
        teacher_vocab_size: int,
        temperature: float,
        global_top_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Project student logits into the reduced teacher vocabulary space.

        Returns projected student probabilities of shape (B, S_student, K)
        where K = len(global_top_indices).
        """
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
        aligned_pairs: list,
        batch_size: int,
        student_seq_len: int,
        teacher_seq_len: int,
        student_vocab_size: int,
        teacher_vocab_size: int,
        temperature: float,
        reverse_kl: bool,
        xtoken_loss: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        """Gold loss: common-vocab KL + uncommon-vocab sorted L1.

        Splits the vocabulary into tokens with exact 1:1 projection mappings
        ("common") and the rest ("uncommon"). Common tokens are compared
        directly via KL on their native log-probs (no projection needed).
        Uncommon tokens are compared via L1 on sorted probability vectors
        (Universal Likelihood Distillation).

        Matches tokenalign.py compute_KL_loss_optimized gold_loss branch.
        """
        aligner = self.token_aligner
        if not hasattr(aligner, 'likelihood_projection_indices') or aligner.likelihood_projection_indices is None:
            raise ValueError("gold_loss requires likelihood_projection_indices to be loaded")

        projection_indices = aligner.likelihood_projection_indices
        projection_matrix = (
            aligner.transform_learned_matrix_instance(aligner.likelihood_projection_matrix)
            if getattr(aligner, 'learnable', False)
            else aligner.likelihood_projection_matrix
        )

        sorted_values, sorted_indices_in_topk = torch.sort(projection_matrix, dim=-1, descending=True)

        if xtoken_loss:
            has_exact_map = (sorted_values[:, 0] >= 0.6)
        else:
            has_exact_map = (sorted_values[:, 0] == 1.0) & (projection_indices[:, 1] == -1)

        student_indices_with_exact_map = torch.where(has_exact_map)[0]
        teacher_indices_for_exact_map = projection_indices[
            student_indices_with_exact_map,
            sorted_indices_in_topk[student_indices_with_exact_map, 0],
        ]

        student_to_teacher_exact_map: dict[int, int] = {}
        teacher_to_student_exact_map: dict[int, int] = {}
        for s_idx, t_idx in zip(
            student_indices_with_exact_map.tolist(),
            teacher_indices_for_exact_map.tolist(),
        ):
            if 0 <= t_idx < teacher_vocab_size:
                if t_idx not in teacher_to_student_exact_map or xtoken_loss:
                    if t_idx in teacher_to_student_exact_map:
                        prev_student_token = teacher_to_student_exact_map[t_idx]
                        if sorted_values[prev_student_token, 0] >= sorted_values[s_idx, 0]:
                            continue
                        del student_to_teacher_exact_map[prev_student_token]
                    student_to_teacher_exact_map[s_idx] = t_idx
                    teacher_to_student_exact_map[t_idx] = s_idx

        common_student_indices = sorted(student_to_teacher_exact_map.keys())
        common_teacher_indices = [student_to_teacher_exact_map[s] for s in common_student_indices]
        uncommon_student_indices = sorted(set(range(student_vocab_size)) - set(common_student_indices))
        uncommon_teacher_indices = sorted(set(range(teacher_vocab_size)) - set(common_teacher_indices))

        # Build chunk masks from alignment pairs (matching tokenalign.py exactly)
        max_n_chunks = min(student_seq_len, teacher_seq_len)

        student_chunk_mask = torch.zeros(
            (batch_size, student_seq_len, max_n_chunks), dtype=torch.bool, device=device,
        )
        teacher_chunk_mask = torch.zeros(
            (batch_size, teacher_seq_len, max_n_chunks), dtype=torch.bool, device=device,
        )

        for batch_idx in range(batch_size):
            for chunk_idx, alignment_pair in enumerate(aligned_pairs[batch_idx][:max_n_chunks]):
                s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                if start1 != -1 and start2 != -1:
                    student_chunk_mask[batch_idx, start1:end1, chunk_idx] = True
                    teacher_chunk_mask[batch_idx, start2:end2, chunk_idx] = True

        # log_softmax on full original logits BEFORE chunk averaging
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
        if len(common_student_indices) > 0:
            cs = torch.tensor(common_student_indices, device=device)
            ct = torch.tensor(common_teacher_indices, device=device)
            s_common = student_chunk_lp[:, :, cs]
            t_common = teacher_chunk_lp[:, :, ct]

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
        if len(uncommon_student_indices) > 0 or len(uncommon_teacher_indices) > 0:
            s_uncommon = student_chunk_lp[:, :, torch.tensor(uncommon_student_indices, device=device)] if uncommon_student_indices else torch.empty(batch_size, max_n_chunks, 0, device=device)
            t_uncommon = teacher_chunk_lp[:, :, torch.tensor(uncommon_teacher_indices, device=device)] if uncommon_teacher_indices else torch.empty(batch_size, max_n_chunks, 0, device=device)

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

        # Top-1 accuracy on common vocab
        top1_accuracy = 0.0
        with torch.no_grad():
            if len(common_student_indices) > 0 and chunk_valid.any():
                cs = torch.tensor(common_student_indices, device=device)
                ct = torch.tensor(common_teacher_indices, device=device)
                s_valid_lp = student_chunk_lp[chunk_valid][:, cs]
                t_valid_lp = teacher_chunk_lp[chunk_valid][:, ct]
                matches = (s_valid_lp.argmax(dim=-1) == t_valid_lp.argmax(dim=-1)).sum().item()
                top1_accuracy = matches / chunk_valid.sum().item()

        del student_chunk_lp, teacher_chunk_lp, student_chunk_mask, teacher_chunk_mask
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
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute cross-tokenizer distillation loss via chunk-averaged KL.

        For each alignment chunk (1:1, 1:many, many:1, or many:many), the
        projected student and teacher distributions are averaged over their
        respective spans, renormalized, and compared via KL divergence.
        The per-chunk KL is then distributed back to student positions
        and normalized with the standard NeMo RL masked_mean.
        """
        input_ids_student = data["input_ids"]
        batch_size = input_ids_student.shape[0]

        if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            student_logits = next_token_logits.full_tensor().to(torch.float32)
        else:
            student_logits = next_token_logits.to(torch.float32)

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

        if isinstance(teacher_logits, torch.distributed.tensor.DTensor):
            teacher_logits_f32 = teacher_logits.full_tensor().to(torch.float32)
        else:
            teacher_logits_f32 = teacher_logits.to(torch.float32)

        if teacher_logits_f32.shape[-1] == 0:
            raise ValueError(
                f"Teacher logits have vocab dimension 0 (shape={teacher_logits_f32.shape}). "
                "This typically means topk_logits=0 was passed instead of None "
                "for the teacher forward pass. Cross-tokenizer distillation "
                "requires full teacher logits (topk_logits=None)."
            )

        aligned_pairs = self._aligned_pairs
        if mb_idx is not None and mbs is not None:
            mb_start = mb_idx * mbs
            mb_end = mb_start + batch_size
            aligned_pairs = aligned_pairs[mb_start:mb_end]

        self.token_aligner = self.token_aligner.to(student_logits.device)
        device = student_logits.device

        temperature = self.cfg.get("temperature", 1.0)
        vocab_topk = self.cfg.get("vocab_topk", 8192)
        reverse_kl = self.cfg.get("reverse_kl", False)
        exact_match_only = self.cfg.get("exact_token_match_only", False)
        use_gold_loss = self.cfg.get("gold_loss", False)
        use_xtoken_loss = self.cfg.get("xtoken_loss", False)
        student_seq_len = student_logits.shape[1]
        teacher_seq_len = teacher_logits_f32.shape[1]
        student_vocab_size = student_logits.shape[-1]
        teacher_vocab_size = teacher_logits_f32.shape[-1]

        # -- 1. Filter alignment pairs and count chunks --
        filtered_pairs: list[list[tuple]] = []
        total_chunks = 0
        for batch_idx in range(batch_size):
            batch_pairs = []
            for pair in aligned_pairs[batch_idx]:
                s1text, s2text, s1_start, s1_end, s2_start, s2_end = pair[:6]
                if exact_match_only and (s1_end - s1_start != 1 or s2_end - s2_start != 1):
                    continue
                if s1_start == -1 or s2_start == -1:
                    continue
                if s1_end > student_seq_len or s2_end > teacher_seq_len:
                    continue
                batch_pairs.append(pair)
            filtered_pairs.append(batch_pairs)
            total_chunks = max(total_chunks, len(batch_pairs))

        if total_chunks == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return loss, {"loss": 0.0, "topk_accuracy": 0.0, "num_chunks": 0}

        # -- 2. Build chunk masks (B, seq_len, num_chunks) --
        proj_mask = torch.zeros(
            batch_size, student_seq_len, total_chunks, dtype=torch.bool, device=device,
        )
        tgt_mask = torch.zeros(
            batch_size, teacher_seq_len, total_chunks, dtype=torch.bool, device=device,
        )
        for batch_idx in range(batch_size):
            for chunk_idx, pair in enumerate(filtered_pairs[batch_idx]):
                _, _, s1_start, s1_end, s2_start, s2_end = pair[:6]
                proj_mask[batch_idx, s1_start:s1_end, chunk_idx] = True
                tgt_mask[batch_idx, s2_start:s2_end, chunk_idx] = True

        # ================================================================
        # Gold loss path: common-vocab KL + uncommon-vocab sorted L1.
        # Bypasses the projection matrix for tokens with exact 1:1 mappings.
        # Matches tokenalign.py compute_KL_loss_optimized gold_loss branch.
        # ================================================================
        if use_gold_loss:
            loss, top1_accuracy = self._compute_gold_loss(
                student_logits, teacher_logits_f32, aligned_pairs,
                batch_size, student_seq_len, teacher_seq_len,
                student_vocab_size, teacher_vocab_size,
                temperature, reverse_kl, use_xtoken_loss, device,
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

        # ================================================================
        # Optional CE (next-token prediction) loss, matching the DDP
        # train_distillation_ddp.py logic:
        #   without dynamic scaling: loss = kl * kl_weight + ce * ce_scale
        #   with dynamic scaling:    loss = kl * (ce/kl)   + ce
        # ================================================================
        kl_loss = loss
        ce_loss_scale = self.cfg.get("ce_loss_scale", 0.0)
        dynamic_loss_scaling = self.cfg.get("dynamic_loss_scaling", False)
        ce_loss_value = 0.0

        if ce_loss_scale > 0.0 or dynamic_loss_scaling:
            ce_loss = torch.nn.functional.cross_entropy(
                student_logits[:, :-1].reshape(-1, student_logits.shape[-1]),
                input_ids_student[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            ce_loss_value = float(ce_loss.item())

            if dynamic_loss_scaling and kl_loss.item() > 0:
                dls_scale = ce_loss.item() / kl_loss.item()
                loss = kl_loss * dls_scale + ce_loss
            else:
                loss = kl_loss + ce_loss * ce_loss_scale

        # One-time debug dump for sanity check comparison with standalone TokenAligner
        if not getattr(self, '_debug_dumped', False):
            self._debug_dumped = True
            raw_loss = float(loss.item()) if loss.ndim == 0 else float(loss)
            print(f"[CrossTokenKL DEBUG] raw_chunk_loss={raw_loss:.6f}, "
                  f"gold_loss={use_gold_loss}, "
                  f"student_shape={student_logits.shape}, "
                  f"teacher_shape={teacher_logits_f32.shape}, "
                  f"total_filtered_pairs={sum(len(fp) for fp in filtered_pairs)}", flush=True)
            try:
                import os
                dump_dir = os.environ.get("CROSS_TOK_DEBUG_DIR", "/tmp/cross_tok_debug")
                os.makedirs(dump_dir, exist_ok=True)
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                teacher_ids = self._teacher_input_ids
                if mb_idx is not None and mbs is not None:
                    teacher_ids = teacher_ids[mb_idx * mbs : mb_idx * mbs + batch_size]
                torch.save({
                    "student_logits": student_logits.cpu(),
                    "teacher_logits": teacher_logits_f32.cpu(),
                    "input_ids_student": input_ids_student.cpu(),
                    "input_ids_teacher": teacher_ids.cpu(),
                    "aligned_pairs": aligned_pairs,
                    "config": dict(self.cfg),
                }, os.path.join(dump_dir, f"debug_rank{rank}.pt"))
                print(f"[CrossTokenKL DEBUG] Saved debug tensors to {dump_dir}/debug_rank{rank}.pt", flush=True)
            except Exception as e:
                print(f"[CrossTokenKL DEBUG] Failed to save debug tensors: {e}", flush=True)

        # Scale for NeMo RL distributed training
        token_mask = data["token_mask"]
        sample_mask = data["sample_mask"]
        max_len = min(token_mask.shape[1] - 1, student_seq_len)
        local_mask = token_mask[:, 1 : max_len + 1] * sample_mask.unsqueeze(-1)
        local_valid_toks = local_mask.sum()

        if local_valid_toks > 0 and global_valid_toks > 0:
            loss = loss * local_valid_toks / global_valid_toks
        else:
            loss = loss * 0.0

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
