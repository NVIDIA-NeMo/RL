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

from string import whitespace

import torch

from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    calculate_kl,
    masked_mean,
    masked_var,
)


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

    KL penalty is applied to token-level rewards *externally* before calling the
    pure GAE computation, following veRL's separation-of-concerns approach.  This
    keeps the core GAE loop agnostic to reward construction and makes it easy to
    swap in different reward signals (process reward models, no KL, etc.) without
    touching the advantage estimator.

    The GAE loop uses carry-forward masking: at masked positions the running
    accumulators (next_values, last_gae_lam) are preserved from the last valid
    token rather than being zeroed.  This correctly skips over non-response tokens
    (padding, separators in multi-turn) without introducing phantom TD errors.

    Args:
        gae_lambda: GAE λ parameter (decay factor for advantage estimation, typically 0.95-0.98)
        gae_gamma: Discount factor γ (typically 0.99)
        normalize_advantages: If True, normalize advantages globally across batch
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.gae_lambda = estimator_config.get("gae_lambda", 0.95)
        self.gae_gamma = estimator_config.get("gae_gamma", 0.99)
        self.normalize_advantages = estimator_config.get("normalize_advantages", True)

        self.kl_coef = loss_config.get("reference_policy_kl_penalty", 0.1)
        self.kl_type = loss_config.get("reference_policy_kl_type", "k1")

    def _reward_whiten(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        shift_mean: bool = True,
    ) -> torch.Tensor:
        mean = masked_mean(rewards, mask)
        var = masked_var(rewards, mask, mean)

        whitened_rewards = (rewards - mean) * torch.rsqrt(var + 1e-8)

        if not shift_mean:
            whitened_rewards = whitened_rewards + mean
        return whitened_rewards

    def _build_token_level_rewards(
        self,
        rewards: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        logprobs: torch.Tensor | None = None,
        reference_logprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-token reward tensor with optional KL penalty.

        Constructs token_level_rewards = -kl_coef * KL  (at every response token)
                                        + terminal_reward  (at last valid token)

        Args:
            rewards: Scalar reward per sample, shape [batch_size].
            lengths: Total sequence lengths, shape [batch_size].
            mask: Response token mask, shape [batch_size, seq_len].
            logprobs: Current policy log probs, shape [batch_size, seq_len].
            reference_logprobs: Reference policy log probs, shape [batch_size, seq_len].

        Returns:
            token_level_rewards: shape [batch_size, seq_len].
        """
        seq_len = mask.shape[1]
        token_level_rewards = torch.zeros(
            rewards.shape[0], seq_len, device=rewards.device, dtype=rewards.dtype
        )

        # Apply KL penalty at every response token
        if (
            self.kl_coef > 0
            and logprobs is not None
            and reference_logprobs is not None
        ):
            kl = calculate_kl(logprobs, reference_logprobs, self.kl_type)
            token_level_rewards = token_level_rewards - self.kl_coef * kl

        # Place terminal reward at last valid token for each sample
        for i in range(rewards.shape[0]):
            L = int(lengths[i])
            if L > 0:
                token_level_rewards[i, L - 1] += rewards[i]

        # Zero out prompt/padding positions
        token_level_rewards = token_level_rewards * mask

        return token_level_rewards

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        lengths,
        values,
        reference_logprobs=None,
        logprobs=None,
        **kwargs,
    ):
        """Compute GAE advantages with temporal bootstrapping.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
            lengths: Input lengths of shape [batch_size].
            values: Value predictions of shape [batch_size, seq_len]. Required for GAE.
            reference_logprobs: Reference policy log probabilities of shape [batch_size, seq_len].
            logprobs: Current policy log probabilities of shape [batch_size, seq_len].
            **kwargs: Additional arguments (unused).

        Returns:
            Tuple of (advantages, returns), each of shape [batch_size, seq_len].
        """
        # Step 1: Build token-level rewards externally (KL penalty applied here)
        token_level_rewards = self._build_token_level_rewards(
            rewards, lengths, mask, logprobs, reference_logprobs,
        )

        # Step 2: Pure GAE computation over (rewards, values, mask)
        advantages, returns = self._compute_gae(token_level_rewards, values, mask)

        # Step 3: Whiten advantages and zero out masked positions
        advantages = torch.masked_fill(
            self._reward_whiten(advantages, mask),
            ~(mask.bool()),
            0,
        )
        return advantages, returns

    def _compute_gae(
        self,
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure GAE computation with carry-forward masking.

        At masked positions the running accumulators (next_values, last_gae_lam)
        are preserved from the last valid token rather than being corrupted by
        zeroed-out values.  This correctly handles non-contiguous response masks
        (multi-turn conversations, tool-use delimiters, packed sequences).

        Args:
            token_level_rewards: Per-token rewards, shape [batch_size, response_length].
            values: Value predictions, shape [batch_size, response_length].
            mask: Response token mask, shape [batch_size, response_length].

        Returns:
            advantages: shape [batch_size, response_length].
            returns: advantages + values, shape [batch_size, response_length].
        """
        gen_len = token_level_rewards.shape[-1]
        next_values: torch.Tensor = torch.zeros(
            values.shape[0], device=values.device, dtype=values.dtype
        )
        last_gae_lam: torch.Tensor = torch.zeros_like(next_values)
        advantages_reversed = []

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + self.gae_gamma * next_values - values[:, t]
            new_gae_lam = delta + self.gae_gamma * self.gae_lambda * last_gae_lam

            # Carry-forward: at masked positions, preserve accumulators from
            # the last valid token instead of updating them.
            m = mask[:, t]
            next_values = values[:, t] * m + (1 - m) * next_values
            last_gae_lam = new_gae_lam * m + (1 - m) * last_gae_lam

            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages, returns
