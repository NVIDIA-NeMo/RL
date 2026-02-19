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


class GAEAdvantageEstimator:
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
        values,
        **kwargs,
    ):
        """Compute GAE advantages with temporal bootstrapping.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
                       For GAE, this is used to identify episode boundaries across samples.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
                    In PPO, this is typically the final reward at the end of the trajectory.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
            values: Value predictions of shape [batch_size, seq_len]. Required for GAE.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        batch_size, seq_len = mask.shape
        device = values.device

        # Compute temporal differences: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        deltas = rewards + self.gae_gamma * next_values * (1 - dones) - values

        # Compute GAE recursively backwards: A_t = δ_t + γλ * (1 - done) * A_{t+1}
        advantages = torch.zeros_like(values)
        gae = torch.zeros(batch_size, device=device)

        # Iterate backwards through time
        for t in reversed(range(seq_len)):
            # Only compute GAE for valid tokens
            valid_mask_t = mask[:, t]

            # GAE recursive formula
            gae = (
                deltas[:, t]
                + self.gae_gamma * self.gae_lambda * (1 - dones[:, t]) * gae
            )

            # Store advantages only for valid positions
            advantages[:, t] = gae * valid_mask_t

            # Reset GAE to 0 for sequences that ended (done=1) or are invalid (mask=0)
            gae = gae * (1 - dones[:, t]) * valid_mask_t

        # Normalize advantages globally across batch if requested
        if self.normalize_advantages:
            valid_advantages = advantages[mask.bool()]
            if valid_advantages.numel() > 0:
                adv_mean = valid_advantages.mean()
                adv_std = valid_advantages.std()
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                # Ensure padding tokens remain zero
                advantages = advantages * mask

        return advantages

    def compute_value_targets(
        self,
        rewards,
        values,
        next_values,
        mask,
        dones=None,
    ):
        """Compute value function targets for training the critic.

        Value targets are computed as the discounted sum of rewards with bootstrapping:
            V_target(s_t) = r_t + γ * V(s_{t+1}) * (1 - done_t)

        Alternatively, they can be computed from advantages:
            V_target(s_t) = A_t + V(s_t)

        Args:
            rewards: Tensor of shape [batch_size] or [batch_size, seq_len] containing rewards.
            values: Value predictions of shape [batch_size, seq_len].
            next_values: Next state value predictions of shape [batch_size, seq_len].
            mask: Response token mask of shape [batch_size, seq_len].
            dones: Episode termination flags of shape [batch_size, seq_len].

        Returns:
            Value targets tensor of shape [batch_size, seq_len].
        """
        batch_size, seq_len = mask.shape
        device = values.device

        # Expand rewards to token-level if needed
        if rewards.dim() == 1:
            rewards_expanded = torch.zeros_like(values)
            for i in range(batch_size):
                valid_positions = torch.nonzero(mask[i], as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1]
                    rewards_expanded[i, last_valid_pos] = rewards[i]
            rewards = rewards_expanded

        # Create done masks if not provided
        if dones is None:
            dones = torch.zeros_like(mask, dtype=torch.float32, device=device)
            for i in range(batch_size):
                valid_positions = torch.nonzero(mask[i], as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1]
                    dones[i, last_valid_pos] = 1.0

        # Compute value targets with bootstrapping
        value_targets = rewards + self.gae_gamma * next_values * (1 - dones)

        # Apply mask to ensure padding positions have zero targets
        value_targets = value_targets * mask

        return value_targets
