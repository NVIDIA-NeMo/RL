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
Reference papers:
- ProRLv2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
- Reinforce++: https://arxiv.org/abs/2501.03262
"""

import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    calculate_kl,
    get_gdpo_reward_component_keys,
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
        repeated_batch = kwargs.get("repeated_batch")
        chain_hashes = (
            repeated_batch.get("chain_hash") if repeated_batch is not None else None
        )
        if chain_hashes is None:
            baseline, std = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                rewards,
                torch.ones_like(rewards),
                leave_one_out_baseline=self.use_leave_one_out_baseline,
            )
        else:
            baseline, std = _calculate_chain_aware_baseline_and_std(
                prompt_ids,
                rewards,
                chain_hashes,
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

        if chain_hashes is not None:
            # A rollout chain is expanded into a fixed number of round slots. Without
            # this normalization, chains that use more real rounds contribute the same
            # sequence advantage multiple times (while early-success padding is masked),
            # biasing the policy toward multi-round outcomes. Keep sample_mask binary
            # for validity/error filtering and normalize the policy signal here instead.
            round_weights = _calculate_chain_round_weights(
                chain_hashes,
                mask,
                dtype=advantages.dtype,
                device=advantages.device,
            )
            advantages = advantages * round_weights.unsqueeze(-1)

        return advantages.expand(mask.shape)


def _calculate_chain_aware_baseline_and_std(
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    chain_hashes: list[str],
    *,
    leave_one_out_baseline: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one GRPO statistic per rollout chain, then expand to round slots."""
    if len(chain_hashes) != len(rewards):
        raise ValueError(
            f"chain_hash length {len(chain_hashes)} does not match rewards {len(rewards)}"
        )

    representative_rows: list[int] = []
    chain_to_representative: dict[str, int] = {}
    inverse: list[int] = []
    for row, chain_hash in enumerate(chain_hashes):
        if chain_hash not in chain_to_representative:
            chain_to_representative[chain_hash] = len(representative_rows)
            representative_rows.append(row)
        representative = representative_rows[chain_to_representative[chain_hash]]
        if not torch.equal(prompt_ids[row], prompt_ids[representative]):
            raise ValueError(f"chain_hash {chain_hash!r} spans multiple prompt groups")
        if not torch.equal(rewards[row], rewards[representative]):
            raise ValueError(f"chain_hash {chain_hash!r} has inconsistent rewards")
        inverse.append(chain_to_representative[chain_hash])

    representative_index = torch.tensor(
        representative_rows, dtype=torch.long, device=rewards.device
    )
    representative_prompts = prompt_ids[representative_index]
    representative_rewards = rewards[representative_index]
    chain_baseline, chain_std = calculate_baseline_and_std_per_prompt(
        representative_prompts,
        representative_rewards,
        torch.ones_like(representative_rewards),
        leave_one_out_baseline=leave_one_out_baseline,
    )
    inverse_index = torch.tensor(inverse, dtype=torch.long, device=rewards.device)
    return chain_baseline[inverse_index], chain_std[inverse_index]


def _calculate_chain_round_weights(
    chain_hashes: list[str],
    mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Give each active round an equal share of its rollout chain's loss."""
    if len(chain_hashes) != mask.shape[0]:
        raise ValueError(
            f"chain_hash length {len(chain_hashes)} does not match mask batch "
            f"size {mask.shape[0]}"
        )

    active_rows = mask.reshape(mask.shape[0], -1).any(dim=1).tolist()
    active_rounds_per_chain: dict[str, int] = {}
    for chain_hash, is_active in zip(chain_hashes, active_rows, strict=True):
        if is_active:
            active_rounds_per_chain[chain_hash] = (
                active_rounds_per_chain.get(chain_hash, 0) + 1
            )

    weights = [
        1.0 / active_rounds_per_chain[chain_hash] if is_active else 0.0
        for chain_hash, is_active in zip(chain_hashes, active_rows, strict=True)
    ]
    return torch.tensor(weights, dtype=dtype, device=device)


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
            repeated_batch: Batch containing reward1, reward2, ... keys.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        reward_component_keys = get_gdpo_reward_component_keys(repeated_batch)
        if len(reward_component_keys) < 2:
            raise ValueError(
                f"GDPO requires multiple reward components (reward1, reward2, ...). "
                f"This batch has {len(reward_component_keys)} component(s). "
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
