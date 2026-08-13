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
- GDPOAdvantageEstimator: Multi-reward GDPO (per-component baselines, sum then normalize)
- ReinforcePlusPlusAdvantageEstimator: Reinforce++ with optional baseline subtraction (minus_baseline) and KL penalty in reward
- RawRewardAdvantageEstimator: Raw reward as advantage with optional batch normalization (no baseline, no value model)
- GeneralizedAdvantageEstimator: Generalized Advantage Estimation (GAE) with temporal bootstrapping
- TurnLevelGeneralizedAdvantageEstimator: GAE over agent TURNS instead of tokens
- ResidualBaselineEstimator: wraps a value-based estimator so the group supplies the
  task baseline B(X) and the critic learns only the within-task residual C(s)
- OPDAdvantageEstimator: Multi-Teacher On-Policy Distillation (MOPD) token-level distillation advantages
Reference papers:
- ProRLv2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
- Reinforce++: https://arxiv.org/abs/2501.03262
- GAE: https://arxiv.org/abs/1506.02438 (High-Dimensional Continuous Control Using Generalized Advantage Estimation)
- MOPD: https://arxiv.org/abs/2601.02780
"""

from typing import Any, Optional

import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    calculate_kl,
    get_gdpo_reward_component_keys,
    masked_mean,
    masked_var,
)


class GRPOAdvantageEstimator:
    """GRPO-style advantage estimator with leave-one-out baseline.

    Note: GRPO computes advantages over all responses for each prompt.
    """

    def __init__(self, estimator_config: dict, loss_config: ClippedPGLossConfig):
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


class GDPOAdvantageEstimator:
    """GDPO-style advantage estimator with leave-one-out baseline.

    Note: GDPO computes advantages for each reward separately over all responses for each prompt.
    """

    def __init__(self, estimator_config: dict, loss_config: ClippedPGLossConfig):
        self.use_leave_one_out_baseline = estimator_config["use_leave_one_out_baseline"]
        self.normalize_rewards = estimator_config["normalize_rewards"]

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        repeated_batch,
        **kwargs,
    ):
        """Compute GDPO advantages.

        Args:
            prompt_ids: Tensor identifying which prompt each sample belongs to (for per-prompt baselines).
            rewards: Unused; for interface consistency.
            repeated_batch: Batch containing named reward component keys (e.g. reward/correctness, reward/format).
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        reward_component_keys = get_gdpo_reward_component_keys(repeated_batch)
        if len(reward_component_keys) < 2:
            raise ValueError(
                f"GDPO requires multiple reward components (reward/name1, reward/name2, ...). "
                f"This batch has {len(reward_component_keys)} component(s): {reward_component_keys}. "
                "Switch to GRPO by setting grpo.adv_estimator.name to 'grpo' in your config."
            )
        valid = torch.ones_like(repeated_batch[reward_component_keys[0]])
        leave_one_out = self.use_leave_one_out_baseline
        assert prompt_ids.shape[0] == valid.shape[0], (
            "prompt_ids must match reward batch size; "
            f"got {prompt_ids.shape[0]} vs {valid.shape[0]}"
        )
        advantage_parts = []
        for key in reward_component_keys:
            r = repeated_batch[key]
            base, std_k = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                r,
                valid,
                leave_one_out_baseline=leave_one_out,
            )
            adv_k = (r - base).unsqueeze(-1)
            if self.normalize_rewards:
                epsilon = 1e-6
                non_zero_std_mask = std_k > 0
                adv_k[non_zero_std_mask] = adv_k[non_zero_std_mask] / (
                    std_k.unsqueeze(-1)[non_zero_std_mask] + epsilon
                )

            advantage_parts.append(adv_k)

        advantages = sum(advantage_parts)
        # Normalize combined advantage to zero mean and unit std
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - advantages.mean()) / adv_std
        else:
            advantages = advantages - advantages.mean()

        return advantages.expand(mask.shape)


class ReinforcePlusPlusAdvantageEstimator:
    """Reinforce++ advantage estimator with optional baseline subtraction and KL penalty in reward.

    Args:
        minus_baseline: If True, subtract per-prompt mean baseline from rewards.
        use_kl_in_reward: If True, add KL penalty to reward instead of loss.
    """

    def __init__(self, estimator_config: dict, loss_config: ClippedPGLossConfig):
        self.minus_baseline = estimator_config["minus_baseline"]
        self.use_kl_in_reward = loss_config.use_kl_in_reward
        self.kl_coef = loss_config.reference_policy_kl_penalty
        self.kl_type = loss_config.reference_policy_kl_type

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


class RawRewardAdvantageEstimator:
    """Advantage estimator that uses the raw reward directly as the advantage.

    No value model, no baselines. Optionally normalizes across the batch.
    """

    def __init__(self, estimator_config: dict, loss_config: ClippedPGLossConfig):
        self.normalize_advantages = estimator_config["normalize_advantages"]

    def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
        """Compute advantages as raw rewards expanded to token-level shape.

        Args:
            prompt_ids: Tensor of shape [batch_size] (unused).
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len].
            **kwargs: Additional arguments (unused).

        Returns:
            Tuple of (advantages, returns) where returns is None.
        """
        adv = rewards.unsqueeze(-1).expand(mask.shape)

        if self.normalize_advantages:
            adv_mean = (adv * mask).sum() / mask.sum().clamp(min=1)
            adv_var = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum().clamp(min=1)
            adv_rstd = adv_var.clamp(min=1e-8).rsqrt()
            adv = (adv - adv_mean) * adv_rstd

        return adv, None


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

    def __init__(self, estimator_config: dict, loss_config: ClippedPGLossConfig):
        self.gae_lambda = estimator_config["gae_lambda"]
        self.gae_gamma = estimator_config["gae_gamma"]
        self.normalize_advantages = estimator_config["normalize_advantages"]

        # VAPO decoupled GAE: separate λ for value returns vs policy advantages.
        # None for both = standard GAE (use gae_lambda everywhere, no decoupling).
        self.gae_lambda_value = estimator_config["gae_lambda_value"]
        self.gae_lambda_policy = estimator_config["gae_lambda_policy"]
        # Length-adaptive λ_policy = 1 - 1/(α·l). 0 = disabled (use fixed λ).
        self.length_adaptive_alpha = estimator_config["length_adaptive_alpha"]

        self.use_kl_in_reward = loss_config.use_kl_in_reward
        self.kl_coef = loss_config.reference_policy_kl_penalty
        self.kl_type = loss_config.reference_policy_kl_type

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
        mask: torch.Tensor,
        logprobs: torch.Tensor | None = None,
        reference_logprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-token reward tensor with optional KL penalty.

        Constructs token_level_rewards = -kl_coef * KL  (at every response token)
                                        + terminal_reward  (at last valid token)

        Args:
            rewards: Scalar reward per sample, shape [batch_size].
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

        # Apply KL penalty at every response token (gated by use_kl_in_reward).
        if (
            self.use_kl_in_reward
            and self.kl_coef > 0
            and logprobs is not None
            and reference_logprobs is not None
        ):
            kl = calculate_kl(logprobs, reference_logprobs, self.kl_type)
            token_level_rewards = token_level_rewards - self.kl_coef * kl

        # Place terminal reward at the last response token (last mask=1
        # position) for each sample. Using mask (not a separate `lengths`
        # tensor) ensures the reward lands on an assistant token even in
        # multi-turn scenarios where the sequence may end with a non-assistant
        # message.
        last_response_idx = mask.shape[1] - 1 - mask.fliplr().argmax(dim=1)
        has_response = mask.any(dim=1)
        token_level_rewards[has_response, last_response_idx[has_response]] += rewards[
            has_response
        ]

        # Zero out prompt/padding positions
        token_level_rewards = token_level_rewards * mask

        return token_level_rewards

    def _resolve_lambda_policy(self, mask: torch.Tensor) -> float | torch.Tensor:
        """Return the λ to use for policy advantages.

        Priority: length_adaptive_alpha > gae_lambda_policy > gae_lambda.
        """
        if self.length_adaptive_alpha > 0:
            resp_lens = mask.sum(dim=1).float()  # [batch_size]
            lam = 1.0 - 1.0 / (self.length_adaptive_alpha * resp_lens).clamp(min=1.0)
            return lam.clamp(min=0.0, max=1.0)
        if self.gae_lambda_policy is not None:
            return self.gae_lambda_policy
        return self.gae_lambda

    def _resolve_lambda_value(self) -> float:
        """Return the λ to use for value returns.

        Returns gae_lambda_value if set, else gae_lambda (standard GAE).
        """
        if self.gae_lambda_value is not None:
            return self.gae_lambda_value
        return self.gae_lambda

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        values,
        reference_logprobs=None,
        logprobs=None,
        **kwargs,
    ):
        """Compute GAE advantages with temporal bootstrapping.

        Supports VAPO-style decoupled GAE when gae_lambda_value or
        gae_lambda_policy or length_adaptive_alpha are set:
        - Value returns use gae_lambda_value (default: gae_lambda)
        - Policy advantages use gae_lambda_policy or length-adaptive λ
          (default: gae_lambda)
        When none of these are set, this is standard GAE.

        Returns:
            Tuple of (advantages, returns), each of shape [batch_size, seq_len].
        """
        token_level_rewards = self._build_token_level_rewards(
            rewards,
            mask,
            logprobs,
            reference_logprobs,
        )

        lam_value = self._resolve_lambda_value()
        lam_policy = self._resolve_lambda_policy(mask)

        # If lambdas differ, compute GAE twice (decoupled); otherwise once.
        need_decouple = (
            self.gae_lambda_value is not None
            or self.gae_lambda_policy is not None
            or self.length_adaptive_alpha > 0
        )
        if need_decouple:
            _, returns = self._compute_gae(
                token_level_rewards,
                values,
                mask,
                gae_lambda=lam_value,
            )
            advantages, _ = self._compute_gae(
                token_level_rewards,
                values,
                mask,
                gae_lambda=lam_policy,
            )
        else:
            advantages, returns = self._compute_gae(
                token_level_rewards,
                values,
                mask,
            )

        # Whiten advantages (optional) and zero out masked positions (always)
        if self.normalize_advantages:
            advantages = self._reward_whiten(advantages, mask)
        advantages = torch.masked_fill(advantages, ~(mask.bool()), 0)
        return advantages, returns

    def _compute_gae(
        self,
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        gae_lambda: torch.Tensor | float | None = None,
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
            gae_lambda: Override for self.gae_lambda.  Can be a scalar or a
                per-sample tensor of shape [batch_size] (VAPO length-adaptive).

        Returns:
            advantages: shape [batch_size, response_length].
            returns: advantages + values, shape [batch_size, response_length].
        """
        lam = gae_lambda if gae_lambda is not None else self.gae_lambda

        gen_len = token_level_rewards.shape[-1]
        next_values: torch.Tensor = torch.zeros(
            values.shape[0], device=values.device, dtype=values.dtype
        )
        last_gae_lam: torch.Tensor = torch.zeros_like(next_values)
        advantages_reversed = []

        for t in reversed(range(gen_len)):
            delta = (
                token_level_rewards[:, t] + self.gae_gamma * next_values - values[:, t]
            )
            new_gae_lam = delta + self.gae_gamma * lam * last_gae_lam

            # Carry-forward: at masked positions, preserve accumulators from
            # the last valid token instead of updating them.
            m = mask[:, t]
            next_values = values[:, t] * m + (1 - m) * next_values
            last_gae_lam = new_gae_lam * m + (1 - m) * last_gae_lam

            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages, returns


class TurnLevelGeneralizedAdvantageEstimator:
    """GAE over agent turns instead of tokens (see nemo_rl.algorithms.turn_level).

    The turn MDP treats one assistant message as one action:

        δ_k = r_k + γ V(s_{k+1}) - V(s_k),   A_k = δ_k + γλ A_{k+1},   G_k = A_k + V(s_k)

    with ``V(s_k)`` read at the FIRST token of assistant message k (the value
    head is right-shifted, so that position sees the whole preceding observation
    and none of the action) and ``V(s_{K+1}) = 0``.

    Why not just run the token-level estimator: at the token level λ has
    effective horizon 1/(1-λ) TOKENS, so on a 45k-token rollout any λ that is not
    ≈1 severs the terminal reward entirely — which is exactly why the production
    config lands at λ = 1 - 1.5e-5 and GAE collapses to the pure baseline
    ``A_t = R - V(s_t)`` with no temporal credit assignment at all. Over ~92
    turns, λ=0.97 is a 33-turn horizon: a usable knob.

    A second, structural benefit: at λ<1 the advantage is built from TD
    increments, in which any constant per-trajectory offset cancels identically.
    The critic's measured weakness on this workload is precisely the constant
    part (it cannot read task difficulty from the prompt); the part it is good at
    is the terminal ``δ_K = R - V(s_K)``, which is what survives.

    Outputs:
        advantages: ``[B, S]``, constant across the tokens of one assistant
            message (the standard treatment of a multi-token action).
        returns: ``[B, S]``, ``G_k`` placed at the turn ANCHOR only, zero
            elsewhere — paired with ``token_mask = anchor_mask`` in the critic's
            batch so the value loss is an equal-weighted mean over decision
            points instead of a token-count-weighted one.

    Requires ``turn_spans`` (a :class:`~nemo_rl.algorithms.turn_level.TurnSpans`)
    in ``compute_advantage``; the caller builds it once per step from the batch's
    message logs.
    """

    def __init__(self, estimator_config: dict, loss_config: ClippedPGLossConfig):
        # No silent defaults: a λ that quietly falls back to 1.0 would make a
        # sweep look like it ran when it did not.
        for key in (
            "turn_gae_gamma",
            "turn_gae_lambda_value",
            "turn_gae_lambda_policy",
        ):
            if estimator_config.get(key) is None:
                raise ValueError(
                    f"adv_estimator.{key} must be set explicitly when "
                    "adv_estimator.name='turn_gae' (no default is assumed). "
                    "See research/ppo/turn_level_critic_plan.md."
                )
        self.gamma = float(estimator_config["turn_gae_gamma"])
        self.lambda_value = float(estimator_config["turn_gae_lambda_value"])
        self.lambda_policy = float(estimator_config["turn_gae_lambda_policy"])
        self.normalize_advantages = estimator_config["normalize_advantages"]

        self.use_kl_in_reward = loss_config.use_kl_in_reward
        self.kl_coef = loss_config.reference_policy_kl_penalty
        self.kl_type = loss_config.reference_policy_kl_type

        self.last_metrics: dict[str, float] = {}

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        values,
        turn_spans=None,
        reference_logprobs=None,
        logprobs=None,
        sample_mask=None,
        **kwargs,
    ):
        """Compute turn-level GAE advantages and critic targets.

        Args:
            prompt_ids: unused (kept for interface parity).
            rewards: ``[B]`` terminal reward per sample.
            mask: ``[B, S]`` response token mask.
            values: ``[B, S]`` per-token values from a fresh critic forward.
            turn_spans: :class:`TurnSpans` for this batch (required).
            reference_logprobs / logprobs: ``[B, S]``, only used when
                ``use_kl_in_reward`` is on; the per-token penalty is summed into
                its turn's reward.
            sample_mask: ``[B]``, used for metrics only.

        Returns:
            ``(advantages, returns)``, both ``[B, S]``.
        """
        from nemo_rl.algorithms.turn_level import (
            build_turn_rewards,
            gather_turn_values,
            scatter_turns_to_anchors,
            scatter_turns_to_tokens,
            turn_gae,
            turn_level_metrics,
        )

        if turn_spans is None:
            raise ValueError(
                "TurnLevelGeneralizedAdvantageEstimator requires turn_spans; "
                "build it with nemo_rl.algorithms.turn_level.build_turn_spans() "
                "from the batch's message logs."
            )

        seq_len = mask.shape[1]
        turn_values = gather_turn_values(values, turn_spans)

        token_penalty = None
        if (
            self.use_kl_in_reward
            and self.kl_coef > 0
            and logprobs is not None
            and reference_logprobs is not None
        ):
            kl = calculate_kl(logprobs, reference_logprobs, self.kl_type)
            token_penalty = -self.kl_coef * kl * mask
        turn_rewards = build_turn_rewards(rewards, turn_spans, token_penalty)

        # Decoupled λ (VAPO-style): critic targets and policy advantages may use
        # different horizons. Skip the second pass when they agree.
        _, turn_returns = turn_gae(
            turn_values,
            turn_rewards,
            turn_spans.turn_valid,
            self.gamma,
            self.lambda_value,
        )
        if self.lambda_policy == self.lambda_value:
            turn_advantages = turn_returns - turn_values
        else:
            turn_advantages, _ = turn_gae(
                turn_values,
                turn_rewards,
                turn_spans.turn_valid,
                self.gamma,
                self.lambda_policy,
            )

        advantages = scatter_turns_to_tokens(turn_advantages, turn_spans, seq_len)
        # Anchor layout, matched by token_mask=anchor_mask in the critic's batch
        # (build_turn_value_batch). The two MUST agree: returns placed anywhere
        # the critic's mask does not cover are silently discarded.
        returns = scatter_turns_to_anchors(turn_returns, turn_spans, seq_len)

        if self.normalize_advantages:
            adv_mean = masked_mean(advantages, mask)
            adv_var = masked_var(advantages, mask, adv_mean)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8)
        advantages = torch.masked_fill(advantages, ~(mask.bool()), 0)

        self.last_metrics = turn_level_metrics(
            turn_values, turn_advantages, turn_spans, sample_mask
        )
        return advantages, returns


class ResidualBaselineEstimator:
    """Decomposed group baseline + residual critic (research/ppo/residual_critic_report.md).

    The Bayes-optimal value splits exactly into a between-task and a within-task
    part, ``V*(s) = B(X) + C(s)`` with ``E[C | X] = 0``.  An absolute critic has
    to fit both.  On this SWE workload it overwhelmingly fits the first and fits
    it badly: 88-93% of its output variance is between-task, yet its EV (~0.157)
    is below even a two-value "all-fail vs rest" lookup (0.330) and far below the
    free leave-one-out group baseline (~0.553).  Substituting the critic for that
    baseline *raises* advantage variance 1.67-2.09x.

    So hand ``B`` to the rollout group and let the critic learn only ``C``.  This
    wrapper keeps the critic in RESIDUAL space end to end:

        values  in the batch = C(s)
        returns in the batch = the residual lambda-return (target for C)

    and reconstructs the absolute value ``V~ = B_LOO + C`` only for the duration
    of the inner GAE call.  Keeping the batch residual is what makes the PPO
    value clip (``old_values`` vs ``returns``) self-consistent, and it is why
    this is a wrapper rather than a post-hoc transform of the returns: the report
    (S26.10) says "reconstruct V~ and feed the existing GAE", but ``returns``
    flows straight into :class:`MseValueLossFn`, and at lambda=1 GAE returns are
    ``R`` -- following that literally would train ``C`` against ``R``.

    With gamma = 1 the group baseline cancels from every nonterminal TD error
    (``delta_t = C_{t+1} - C_t``), and at lambda = 1 the advantage telescopes to
    ``A_t = R - B_LOO - C_t``: the critic becomes a state-dependent control
    variate on top of the known-good group advantage, and ``C = 0`` recovers it
    exactly.  gamma < 1 breaks the cancellation, so it is rejected outright.

    Wrapping (rather than subclassing) means both ``gae`` and ``turn_gae`` are
    covered without duplicating the leave-one-out logic.

    Args:
        inner: the value-based estimator to wrap (``gae`` or ``turn_gae``).
        residual_target: when False, ``B_LOO`` is still computed and exported for
            metrics but the targets are left in absolute space.  That is what
            lets an absolute-critic run log ``critic/ev_res`` on the same axis as
            a residual run -- there ``1 - ev_res`` is exactly the report's
            advantage-variance ratio.
    """

    def __init__(self, inner: Any, residual_target: bool):
        self.inner = inner
        self.residual_target = residual_target

        # GeneralizedAdvantageEstimator names it gae_gamma; the turn-level one
        # names it gamma. Neither has a default worth guessing at.
        gamma = getattr(inner, "gae_gamma", getattr(inner, "gamma", None))
        if gamma is None:
            raise ValueError(
                f"{type(inner).__name__} exposes no discount factor, so the "
                "residual baseline cannot verify the gamma = 1 condition it "
                "depends on."
            )
        self.gamma = float(gamma)
        if residual_target and abs(self.gamma - 1.0) > 1e-8:
            raise ValueError(
                f"adv_estimator.residual_baseline requires gamma == 1, got {self.gamma}. "
                "The task baseline only cancels from nonterminal TD errors at "
                "gamma = 1; at gamma < 1 the residual value would silently pick "
                "up a (gamma - 1) * B term (report S11.5)."
            )

        self.last_metrics: dict[str, float] = {}
        # Per-sample offsets that move `returns` into each space. Exactly one is
        # zero. Consumed by MseValueLossFn so it can report BOTH explained
        # variances without knowing which mode it is in.
        self.last_returns_to_abs: torch.Tensor | None = None
        self.last_returns_to_res: torch.Tensor | None = None
        # ``[B]`` float, 1.0 where the rollout's group is all-fail or all-pass.
        self.last_group_homogeneous: torch.Tensor | None = None
        # ``[B]`` long, sibling-group index. Exposed so within-group diagnostics
        # partition trajectories exactly the way the baseline did.
        self.last_group_ids: torch.Tensor | None = None

        # NOTE: this ``last_*`` side-channel deliberately mirrors the existing
        # ``last_metrics`` contract that ppo.py and critic_pretrain.py already
        # read via getattr on the estimator; keeping one convention beats adding
        # a second way to hand per-step tensors back to the caller.

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        values: torch.Tensor,
        turn_spans: Optional[Any] = None,
        sample_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the inner estimator against ``V~ = B_LOO + C`` and return residual targets.

        Returns:
            ``(advantages, returns)``. ``advantages`` are byte-for-byte what the
            inner estimator produces for a critic predicting ``V~`` -- only the
            returns are moved back into residual space.
        """
        baseline = self._leave_one_out_baseline(prompt_ids, rewards)
        b_col = baseline.unsqueeze(-1).to(values.dtype)

        inner_values = values + b_col if self.residual_target else values
        adv_kwargs = dict(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            values=inner_values,
            sample_mask=sample_mask,
            **kwargs,
        )
        if turn_spans is not None:
            adv_kwargs["turn_spans"] = turn_spans
        advantages, returns = self.inner.compute_advantage(**adv_kwargs)

        # Turn-level returns live at ONE anchor per turn and are structurally
        # zero elsewhere (scatter_turns_to_anchors), matched by token_mask =
        # anchor_mask in the critic's batch. Subtracting the baseline unmasked
        # would write -B into every non-anchor position.
        return_mask = (
            turn_spans.anchor_mask.to(returns.dtype)
            if turn_spans is not None
            else mask.to(returns.dtype)
        )
        if self.residual_target:
            returns = returns - b_col * return_mask
            self.last_returns_to_abs = baseline
            self.last_returns_to_res = torch.zeros_like(baseline)
        else:
            self.last_returns_to_abs = torch.zeros_like(baseline)
            self.last_returns_to_res = -baseline

        self.last_metrics = dict(getattr(self.inner, "last_metrics", None) or {})
        self.last_metrics.update(
            self._group_metrics(
                prompt_ids, rewards, baseline, returns, return_mask, sample_mask
            )
        )
        return advantages, returns

    def _leave_one_out_baseline(
        self, prompt_ids: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """``B_LOO[i,j] = (sum_k R[i,k] - R[i,j]) / (G_i - 1)``, in fp32.

        ``valid_mask`` is deliberately all-ones, matching
        :class:`GRPOAdvantageEstimator` exactly: a rollout dropped from the
        critic loss by ``sample_mask`` still feeds its siblings' baselines.  That
        keeps the PPO actor A/B against the working DAPO/GRPO control on
        identical group semantics, and keeps critic pretraining (stage B) aligned
        with what PPO (stage C) will feed the same checkpoint.  The divergence is
        made visible by ``residual/frac_traj_sample_masked`` rather than hidden.
        """
        rewards_f32 = rewards.float()
        baseline, _ = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards_f32,
            torch.ones_like(rewards_f32),
            leave_one_out_baseline=True,
        )
        return baseline

    @staticmethod
    def _group_ids(prompt_ids: torch.Tensor) -> torch.Tensor:
        """``[B]`` group index, using the same identity rule as the LOO baseline."""
        _, inverse = torch.unique(prompt_ids, dim=0, return_inverse=True)
        return inverse.reshape(-1)

    def _group_metrics(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        baseline: torch.Tensor,
        returns: torch.Tensor,
        return_mask: torch.Tensor,
        sample_mask: Optional[torch.Tensor],
    ) -> dict[str, float]:
        """Group-composition diagnostics for the residual target.

        ``frac_groups_mixed`` is the load-bearing one: homogeneous groups have
        ``Y = 0`` for every sibling and therefore contribute EXACTLY zero target
        variance, so this fraction -- not the dataset size -- bounds what the
        residual critic can learn.  On the pi0 SWE pool it is 0.435.

        "Homogeneous" is defined as ZERO WITHIN-GROUP REWARD VARIANCE, not as a
        group sum of 0 or G. Rewards arriving here are post-scaling, post-shaping
        and post-penalty: ``reward_scaling`` maps ``[0,1] -> [-1,1]`` in the math
        configs, and judge rewards can be fractional. A sum-based test would then
        call an all-fail group "mixed" and report ``frac_groups_mixed = 1.0`` on a
        pool that is mostly homogeneous. Zero within-group variance is exactly the
        ``Y = 0`` condition being claimed, under any reward scale.
        """
        with torch.no_grad():
            group_ids = self._group_ids(prompt_ids).to(rewards.device)
            self.last_group_ids = group_ids
            rewards_f32 = rewards.float()
            n_groups = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
            if n_groups == 0:
                return {}

            ones = torch.ones_like(rewards_f32)
            counts = torch.zeros(n_groups, device=rewards.device).index_add_(
                0, group_ids, ones
            )
            sums = torch.zeros(n_groups, device=rewards.device).index_add_(
                0, group_ids, rewards_f32
            )
            means = sums / counts.clamp(min=1)
            sq_dev = torch.zeros(n_groups, device=rewards.device).index_add_(
                0, group_ids, (rewards_f32 - means[group_ids]) ** 2
            )
            homogeneous = (sq_dev <= 1e-12).float()
            mixed = 1.0 - homogeneous
            # all-fail / all-pass are only well defined for a binary reward; they
            # are reported as "homogeneous at the group min / max reward" so they
            # stay meaningful (and still sum to `homogeneous`) under any scaling.
            r_min, r_max = rewards_f32.min(), rewards_f32.max()
            all_fail = homogeneous * (means <= r_min + 1e-12).float()
            all_pass = homogeneous * (means >= r_max - 1e-12).float()
            self.last_group_homogeneous = homogeneous[group_ids]
            # <2 valid siblings: calculate_baseline_and_std_per_prompt falls back
            # to baseline = reward, i.e. a silent Y = 0. Surface it.
            singletons = int((counts < 2).sum().item())

            m = return_mask.bool()
            target_var = (
                returns[m].float().var(unbiased=False).item()
                if int(m.sum()) > 1
                else 0.0
            )
            metrics = {
                "residual/b_loo_mean": baseline.mean().item(),
                "residual/b_loo_std": baseline.std(unbiased=False).item(),
                "residual/target_var": target_var,
                "residual/frac_groups_mixed": mixed.mean().item(),
                "residual/frac_groups_all_fail": all_fail.mean().item(),
                "residual/frac_groups_all_pass": all_pass.mean().item(),
                "residual/n_singleton_groups": float(singletons),
                "residual/group_size_min": counts.min().item(),
                "residual/group_size_max": counts.max().item(),
            }
            if sample_mask is not None:
                metrics["residual/frac_traj_sample_masked"] = (
                    1.0 - sample_mask.float().mean().item()
                )
        if singletons:
            print(
                f"  ⚠️ residual baseline: {singletons} group(s) have <2 rollouts; "
                "their targets collapse to 0 (baseline = own reward)."
            )
        return metrics


def attach_value_baseline_keys(batch: Any, adv_estimator: Any) -> None:
    """Copy the estimator's per-sample return-space offsets onto a critic batch.

    No-op unless a :class:`ResidualBaselineEstimator` produced them.  These let
    :class:`MseValueLossFn` report explained variance in BOTH absolute and
    residual space from one pass, without the loss needing to know which space
    ``returns`` is in.
    """
    to_abs = getattr(adv_estimator, "last_returns_to_abs", None)
    to_res = getattr(adv_estimator, "last_returns_to_res", None)
    if to_abs is None or to_res is None:
        return
    n = batch["returns"].shape[0]
    if to_abs.shape[0] != n:
        raise ValueError(
            f"Return-space offsets have batch size {to_abs.shape[0]} but the "
            f"critic batch has {n}. These are per-sample and would misalign "
            "silently, reporting explained variance against the wrong baseline."
        )
    batch["returns_to_abs"] = to_abs.to(batch["returns"].dtype)
    batch["returns_to_res"] = to_res.to(batch["returns"].dtype)


def homogeneous_group_sample_mask(
    sample_mask: torch.Tensor, adv_estimator: Any, weight: float
) -> torch.Tensor | None:
    """``sample_mask`` rescaled so homogeneous groups carry ``weight``.

    Returns None when there is nothing to do (weight 1.0, or no residual
    estimator), so callers can skip building a separate critic batch entirely.

    Under a residual target, all-fail and all-pass groups have ``Y = 0`` for every
    sibling and so contribute exactly zero target variance -- on the pi0 SWE pool
    that is 56.5% of groups and ~58% of critic FLOPs.  They are still worth
    keeping at some weight: the shrinkage-toward-zero they impose is the
    mechanism enforcing ``E[C | X] = 0``, i.e. the regulariser against the
    between-task leakage this whole change exists to remove.

    Rescaling ``sample_mask`` is a correctly renormalised weighted mean with no
    effective-LR confound, because ``global_valid_toks`` is itself computed as
    ``sum(token_mask * sample_mask)`` (nemo_rl/models/megatron/data.py) -- so the
    denominator moves with the numerator.

    The caller MUST apply this to a separate critic batch: in token-level mode
    the actor shares ``train_data['sample_mask']``.
    """
    if weight == 1.0:
        return None
    if weight < 0.0:
        raise ValueError(
            f"value_loss_fn.homogeneous_group_weight must be >= 0, got {weight}."
        )
    homogeneous = getattr(adv_estimator, "last_group_homogeneous", None)
    if homogeneous is None:
        raise ValueError(
            "value_loss_fn.homogeneous_group_weight != 1.0 requires the residual "
            "baseline estimator (it is what identifies homogeneous groups), but "
            f"{type(adv_estimator).__name__} exposes no group composition."
        )
    # The wrapper is installed even for an absolute-critic run (it computes
    # B_LOO for metrics), so group composition is available there too -- but
    # downweighting must NOT apply. Under an absolute target those groups have
    # Y = R != 0 and do carry target variance, so silently reweighting them
    # would change the objective of the control arm in an A/B that sets this
    # knob in both arms to hold wall-clock fixed.
    if not getattr(adv_estimator, "residual_target", False):
        raise ValueError(
            "value_loss_fn.homogeneous_group_weight != 1.0 is only meaningful "
            "with ppo.adv_estimator.residual_baseline=true. Under an absolute "
            "critic target, homogeneous groups still carry target variance "
            "(Y = R != 0), so downweighting them would silently change the "
            "objective rather than skip empty targets."
        )
    homogeneous = homogeneous.to(sample_mask.device, sample_mask.dtype)
    return sample_mask * (1.0 - homogeneous * (1.0 - weight))


class OPDAdvantageEstimator:
    """Multi-Teacher On-Policy Distillation (MOPD) advantage estimator (arXiv:2601.02780).

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
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.last_metrics: dict[str, float] = {}

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=None,
        prev_logprobs=None,
        **kwargs,
    ):
        """Compute OPD distillation advantages.

        Args:
            prompt_ids: [B] prompt IDs (unused, kept for interface compatibility)
            rewards: [B] rewards (unused for pure distillation)
            mask: [B, S] token mask
            teacher_logprobs: [B, S] teacher model logprobs (required)
            prev_logprobs: [B, S] student training-engine logprobs (required)

        Returns:
            [B, S] token-level distillation advantages (stop-gradient)
        """
        if teacher_logprobs is None:
            raise ValueError("OPD requires teacher_logprobs")
        if prev_logprobs is None:
            raise ValueError("OPD requires prev_logprobs")

        # Â_MOPD,t = sg[log π_teacher - log π_student]  (Equation 8)
        distill_advantages = (teacher_logprobs - prev_logprobs).detach()

        # Apply mask
        advantages = distill_advantages * mask

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
