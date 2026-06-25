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

"""Advantage Estimators for RL algorithms.

This module provides different advantage estimation strategies:
- GRPOAdvantageEstimator: Standard GRPO advantage with leave-one-out baseline
- ReinforcePlusPlusAdvantageEstimator: Reinforce++ with optional baseline subtraction (minus_baseline) and KL penalty in reward
- OPDAdvantageEstimator: On-Policy Distillation (MOPD) token-level distillation advantages
Reference papers:
- ProRLv2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
- Reinforce++: https://arxiv.org/abs/2501.03262
- MOPD: https://arxiv.org/abs/2601.02780
"""

import torch

from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt, calculate_kl


class GRPOAdvantageEstimator:
    """GRPO-style advantage estimator with leave-one-out baseline.

    Note: GRPO computes advantages over all responses for each prompt.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.use_leave_one_out_baseline = estimator_config["use_leave_one_out_baseline"]
        self.normalize_rewards = estimator_config["normalize_rewards"]

    def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
        """Compute GRPO advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
                  Used only for expanding advantages to token-level shape.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards,
            torch.ones_like(rewards),
            leave_one_out_baseline=self.use_leave_one_out_baseline,
        )
        advantages = (rewards - baseline).unsqueeze(-1)

        if self.normalize_rewards:
            # don't sharpen the ones with no variation
            epsilon = 1e-6
            non_zero_std_mask = std > 0
            advantages[non_zero_std_mask] = advantages[non_zero_std_mask] / (
                std.unsqueeze(-1)[non_zero_std_mask] + epsilon
            )

        return advantages.expand(mask.shape)


class ReinforcePlusPlusAdvantageEstimator:
    """Reinforce++ advantage estimator with optional baseline subtraction and KL penalty in reward.

    Args:
        minus_baseline: If True, subtract per-prompt mean baseline from rewards.
        use_kl_in_reward: If True, add KL penalty to reward instead of loss.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.minus_baseline = estimator_config["minus_baseline"]
        self.use_kl_in_reward = loss_config["use_kl_in_reward"]
        self.kl_coef = loss_config["reference_policy_kl_penalty"]
        self.kl_type = loss_config["reference_policy_kl_type"]

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        logprobs_policy=None,
        logprobs_reference=None,
        **kwargs,
    ):
        """Compute Reinforce++ advantages with optional KL penalty.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
                  Used for: (1) expanding advantages to token-level shape, (2) global normalization
                  that only considers valid tokens.
            logprobs_policy: Policy log probabilities of shape [batch_size, seq_len], required if use_kl_in_reward.
            logprobs_reference: Reference policy log probabilities of shape [batch_size, seq_len], required if use_kl_in_reward.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len], globally normalized across valid tokens.
        """
        # minus baseline
        if self.minus_baseline:
            mean, _ = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                rewards,
                torch.ones_like(rewards),
                leave_one_out_baseline=False,
            )
            adv = rewards - mean
        else:
            adv = rewards

        adv = adv.unsqueeze(-1)
        adv = adv.expand(mask.shape)

        # add kl penalty to reward (token-level)
        if (
            self.use_kl_in_reward
            and logprobs_policy is not None
            and logprobs_reference is not None
        ):
            kl = calculate_kl(
                logprobs_policy,
                logprobs_reference,
                kl_type=self.kl_type,
            )
            adv = adv - self.kl_coef * kl

        # global normalization across the batch
        adv_mean = (adv * mask).sum() / mask.sum()
        adv_var = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum()
        adv_rstd = adv_var.clamp(min=1e-8).rsqrt()
        adv = (adv - adv_mean) * adv_rstd

        return adv


class OPDAdvantageEstimator:
    """On-Policy Distillation advantage estimator (MOPD, arXiv:2601.02780).

    Computes token-level distillation advantages:
        Â_MOPD,t = sg[log π_teacher - log π_student]

    This is Equation 8 from the MOPD paper. The IS truncation (w_t, the
    hard gate on the training-to-inference ratio) is handled separately by
    ICE-POP mode in ClippedPGLoss — not here.

    The loss function should be configured with:
        disable_ppo_ratio: true               (REINFORCE, no PPO ratio)
        use_importance_sampling_correction: true
        truncated_importance_sampling_type: icepop
        truncated_importance_sampling_ratio_min: <eps_low>
        truncated_importance_sampling_ratio: <eps_high>

    Required kwargs in compute_advantage:
        teacher_logprobs: [B, S] teacher model log probabilities
        prev_logprobs: [B, S] student training-engine log probabilities
    Optional kwargs:
        orm_advantages: [B, S] ORM advantages to blend in (Equation 9)
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.use_orm_advantage = bool(estimator_config.get("use_orm_advantage", False))
        self.orm_advantage_weight = float(estimator_config.get("orm_advantage_weight", 0.0))
        self.pi_mode = estimator_config.get("pi_mode")
        self.pi_beta = float(estimator_config.get("pi_beta", 1.0))
        self.pi_alpha = float(estimator_config.get("pi_alpha", 0.5))
        self.pi_task_reward_weight = float(
            estimator_config.get("pi_task_reward_weight", 0.0)
        )
        self.use_leave_one_out_baseline = bool(
            estimator_config.get("use_leave_one_out_baseline", True)
        )
        self.normalize_rewards = bool(estimator_config.get("normalize_rewards", True))
        self.last_metrics: dict[str, float] = {}

    def _compute_reward_advantages(self, prompt_ids, rewards, mask, sample_mask=None):
        if sample_mask is None:
            sample_mask = torch.ones_like(rewards)
        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards,
            sample_mask,
            leave_one_out_baseline=self.use_leave_one_out_baseline,
        )
        reward_advantages = rewards - baseline
        if self.normalize_rewards:
            epsilon = 1e-6
            non_zero_std_mask = std > 0
            reward_advantages[non_zero_std_mask] = (
                reward_advantages[non_zero_std_mask]
                / (std[non_zero_std_mask] + epsilon)
            )
        return reward_advantages.unsqueeze(-1).expand(mask.shape)

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=None,
        prev_logprobs=None,
        orm_advantages=None,
        sample_mask=None,
        pi_loss_role=None,
        pi_logprob_gap=None,
        **kwargs,
    ):
        """Compute OPD distillation advantages.

        Args:
            prompt_ids: [B] prompt IDs (unused, kept for interface compatibility)
            rewards: [B] rewards (unused for pure distillation)
            mask: [B, S] token mask
            teacher_logprobs: [B, S] teacher model logprobs (required)
            prev_logprobs: [B, S] student training-engine logprobs (required)
            orm_advantages: [B, S] ORM advantages (optional, Equation 9)

        Returns:
            [B, S] token-level distillation advantages (stop-gradient)
        """
        if self.pi_mode == "pi_distill":
            if pi_loss_role is None:
                raise ValueError("PI distill requires pi_loss_role")
            if pi_logprob_gap is None:
                raise ValueError("PI distill requires pi_logprob_gap")
            reward_term = (
                self.pi_task_reward_weight
                * self._compute_reward_advantages(
                    prompt_ids,
                    rewards,
                    mask,
                    sample_mask=sample_mask,
                ).detach()
            )
            kl_term = self.pi_beta * pi_logprob_gap.detach()
            roles = pi_loss_role.unsqueeze(-1).to(mask.device)
            teacher_advantages = self.pi_alpha * (reward_term - kl_term)
            student_advantages = (1.0 - self.pi_alpha) * (reward_term + kl_term)
            advantages = torch.where(roles == 0, teacher_advantages, student_advantages)
            advantages = advantages * mask
            self._compute_metrics(kl_term, advantages, mask)
            return advantages

        if teacher_logprobs is None:
            raise ValueError("OPD requires teacher_logprobs")
        if prev_logprobs is None:
            raise ValueError("OPD requires prev_logprobs")

        # Reverse-KL dense reward: log teacher minus log student.
        distill_advantages = self.pi_beta * (teacher_logprobs - prev_logprobs).detach()

        if self.pi_mode == "opsd" and self.pi_task_reward_weight != 0:
            reward_advantages = self._compute_reward_advantages(
                prompt_ids,
                rewards,
                mask,
                sample_mask=sample_mask,
            )
            combined = (
                self.pi_task_reward_weight * reward_advantages.detach()
                + distill_advantages
            )
        elif self.use_orm_advantage and orm_advantages is not None:
            combined = (
                distill_advantages
                + self.orm_advantage_weight * orm_advantages.detach()
            )
        else:
            combined = distill_advantages

        # Apply mask
        advantages = combined * mask

        # Metrics
        self._compute_metrics(distill_advantages, advantages, mask)

        return advantages

    def _compute_metrics(self, distill_advantages, advantages, mask):
        """Compute OPD logging metrics and store in self.last_metrics."""
        valid_bool = mask.bool()
        distill_valid = torch.masked_select(distill_advantages, valid_bool)
        adv_valid = torch.masked_select(advantages, valid_bool)

        distill_mean = distill_valid.mean().item() if distill_valid.numel() > 0 else 0.0
        adv_mean = adv_valid.mean().item() if adv_valid.numel() > 0 else 0.0
        adv_std = adv_valid.std().item() if adv_valid.numel() > 1 else 0.0

        self.last_metrics = {
            "on_policy_distillation/teacher_student_logprob_gap_mean": distill_mean,
            "on_policy_distillation/adv_mean": adv_mean,
            "on_policy_distillation/adv_std": adv_std,
        }
