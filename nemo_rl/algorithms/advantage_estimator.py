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
- GAEAdvantageEstimator: Generalized Advantage Estimation (GAE) with temporal bootstrapping
Reference papers:
- ProRLv2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
- Reinforce++: https://arxiv.org/abs/2501.03262
- GAE: https://arxiv.org/abs/1506.02438 (High-Dimensional Continuous Control Using Generalized Advantage Estimation)
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


class GeneralizedAdvantageEstimator:
    """Generalized Advantage Estimation (GAE) with temporal bootstrapping.

    GAE computes advantages using temporal difference (TD) and exponentially-weighted averages:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

    This is computed recursively backwards:
        A_t = δ_t + γλ * (1 - done_t) * A_{t+1}

    Args:
        gae_lambda: GAE λ parameter (decay factor for advantage estimation, typically 0.95-0.98)
        gae_gamma: Discount factor γ (typically 0.99)
        normalize_advantages: If True, normalize advantages globally across batch
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.gae_lambda = estimator_config.get("gae_lambda", 0.95)
        self.gae_gamma = estimator_config.get("gae_gamma", 0.99)
        self.normalize_advantages = estimator_config.get("normalize_advantages", True)

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        lengths,
        values,
        **kwargs,
    ):
        """Compute GAE advantages with temporal bootstrapping.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
                    In PPO, this is typically the final reward at the end of the trajectory.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
            lengths: Input lengths of shape [batch_size].
            values: Value predictions of shape [batch_size, seq_len]. Required for GAE.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        rewards_expanded = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        values_next = torch.zeros_like(values)
        values_next[:, :-1] = values[:, 1:]

        for i, l in enumerate(lengths):
            rewards_expanded[i, l] = rewards[i]

        deltas = rewards_expanded + self.gae_gamma * values_next - values

        last_advantage = 0
        for t in reversed(range(values.shape[1])):
            advantages[:, t] = (
                deltas[:, t] + self.gae_gamma * self.gae_lambda * last_advantage
            )
            last_advantage = advantages[:, t]

        return advantages
