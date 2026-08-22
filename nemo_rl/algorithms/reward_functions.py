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
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import torch
from pydantic import BaseModel, model_validator

from nemo_rl.data.interfaces import LLMMessageLogType, VLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class LengthAwareRewardConfig(BaseModel, extra="allow"):
    """Configure reasoning-chain length reward for NeMo-Gym rollouts.

    The component reward is one through ``tau1``, decreases linearly to zero
    at ``tau2``, and remains zero for longer reasoning chains. The end-token ID
    is model/tokenizer specific and must be provided whenever the feature is
    enabled.
    """

    enabled: bool = False
    tau1: int = 4096
    tau2: int = 13000
    weight: float = 0.1
    composition: Literal["additive", "correctness_gated"] = "additive"
    reasoning_end_token_id: int | None = None

    @model_validator(mode="after")
    def _validate_length_aware_reward(self) -> "LengthAwareRewardConfig":
        if self.tau1 < 0:
            raise ValueError(f"tau1 must be non-negative, got {self.tau1}")
        if self.tau2 <= self.tau1:
            raise ValueError(
                f"tau2 must be greater than tau1, got tau1={self.tau1}, "
                f"tau2={self.tau2}"
            )
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        if self.composition == "correctness_gated" and self.weight > 1:
            raise ValueError(
                "correctness-gated length-aware reward weight must be at most 1, "
                f"got {self.weight}"
            )
        if self.enabled and self.reasoning_end_token_id is None:
            raise ValueError(
                "reasoning_end_token_id must be set when length-aware reward is enabled"
            )
        if self.reasoning_end_token_id is not None and self.reasoning_end_token_id < 0:
            raise ValueError(
                "reasoning_end_token_id must be non-negative, got "
                f"{self.reasoning_end_token_id}"
            )
        return self


@dataclass(frozen=True)
class LengthAwareRewardMetrics:
    component_rewards: list[float]
    reasoning_chain_lengths: list[int]


def length_aware_regularization_reward(
    chain_length: int, *, tau1: int, tau2: int
) -> float:
    """Return the piecewise-linear reward for a reasoning-chain length."""
    if chain_length < 0:
        raise ValueError(f"chain_length must be non-negative, got {chain_length}")
    if tau1 < 0:
        raise ValueError(f"tau1 must be non-negative, got {tau1}")
    if tau2 <= tau1:
        raise ValueError(
            f"tau2 must be greater than tau1, got tau1={tau1}, tau2={tau2}"
        )

    if chain_length <= tau1:
        return 1.0
    if chain_length >= tau2:
        return 0.0
    return 1.0 - (chain_length - tau1) / (tau2 - tau1)


def reasoning_chain_token_length(
    message_log: LLMMessageLogType | VLMMessageLogType,
    *,
    reasoning_end_token_id: int,
) -> int:
    """Count assistant reasoning tokens, excluding each reasoning end token.

    An assistant turn without the configured end token is treated as unfinished
    reasoning, so its complete generated token sequence is counted.
    """
    total = 0
    for message in message_log:
        if message.get("role") != "assistant":
            continue
        token_ids = message.get("token_ids")
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError(
                "Assistant message token_ids must be a torch.Tensor when applying "
                "length-aware reward"
            )
        if token_ids.ndim != 1:
            raise ValueError(
                "Assistant message token_ids must be one-dimensional when applying "
                f"length-aware reward, got shape={tuple(token_ids.shape)}"
            )
        token_id_list = token_ids.tolist()
        try:
            total += token_id_list.index(reasoning_end_token_id)
        except ValueError:
            total += len(token_id_list)
    return total


def apply_length_aware_reward(
    results: list[dict[str, Any]], config: LengthAwareRewardConfig | None
) -> LengthAwareRewardMetrics:
    """Compose reasoning-chain length reward into NeMo-Gym scalar rewards.

    ``additive`` uses ``base + weight * component``. ``correctness_gated`` uses
    ``base * (1 - weight * (1 - component))``, which prevents short incorrect
    responses from receiving positive reward.

    Multi-component Gym rewards are rejected because a non-additive transform
    would otherwise violate the ``reward == sum(reward_components)`` contract.
    """
    if config is None or not config.enabled:
        return LengthAwareRewardMetrics([], [])

    reasoning_end_token_id = config.reasoning_end_token_id
    if reasoning_end_token_id is None:
        raise ValueError(
            "reasoning_end_token_id must be set when length-aware reward is enabled"
        )

    for result in results:
        full_result = result["full_result"]
        if full_result.get("reward_components"):
            raise ValueError(
                "Length-aware reward does not support NeMo-Gym results with "
                "reward_components"
            )
    reasoning_chain_lengths = [
        reasoning_chain_token_length(
            result["message_log"],
            reasoning_end_token_id=reasoning_end_token_id,
        )
        for result in results
    ]
    component_rewards = [
        length_aware_regularization_reward(
            chain_length,
            tau1=config.tau1,
            tau2=config.tau2,
        )
        for chain_length in reasoning_chain_lengths
    ]

    adjusted_rewards = []
    for result, component_reward in zip(results, component_rewards):
        full_result = result["full_result"]
        base_reward = float(full_result["reward"])
        if config.composition == "correctness_gated":
            adjusted_reward = base_reward * (
                1.0 - config.weight * (1.0 - component_reward)
            )
        else:
            adjusted_reward = base_reward + config.weight * component_reward
        adjusted_rewards.append(adjusted_reward)

    for result, adjusted_reward in zip(results, adjusted_rewards):
        result["full_result"]["reward"] = adjusted_reward

    return LengthAwareRewardMetrics(component_rewards, reasoning_chain_lengths)


class RewardShapingConfig(BaseModel, extra="allow"):
    """Configuration for reward function processing.

    This configuration enables custom reward shaping, currently supporting DAPO-style
    penalties for responses that exceed the maximum response length threshold.
    """

    enabled: bool = False

    # The length of the buffer to penalize responses that exceed the maximum response length threshold.
    # Responses of length greater than overlong_buffer_length + max_response_length will
    # receive the maximum penalty.
    overlong_buffer_length: int | None = None

    # The penalty for responses that exceed the maximum response length threshold.
    overlong_buffer_penalty: float | None = None

    # The maximum response length threshold. Responses exceeding this length will be penalized.
    max_response_length: int | None = None

    # Stop properly penalty: scale factor for rewards of truncated responses (0-1).
    # When set to 0, truncated responses get zero reward.
    # When set to 1, no penalty is applied (default behavior).
    stop_properly_penalty_coef: float | None = None


def apply_reward_shaping(
    batch: BatchedDataDict, cfg: RewardShapingConfig
) -> BatchedDataDict:
    """Process rewards by applying penalties for responses exceeding max_response_length. Currently, this function only supports DAPO reward shaping as illustrated in the DAPO paper : https://arxiv.org/pdf/2503.14476.

    Nonetheless, it can be potentially extended to support any custom reward logic.
    """
    rewards = batch["total_reward"]
    if not cfg.enabled:
        return batch

    # Preserve the pre-shaping reward so downstream consumers (e.g. DAPO
    # dynamic sampling) can filter prompt groups on the raw task metric
    # rather than on length-dependent shaped rewards.
    batch["unshaped_total_reward"] = rewards.clone()

    # Apply stop properly penalty if configured
    if cfg.stop_properly_penalty_coef is not None:
        stop_properly_penalty_coef = cfg.stop_properly_penalty_coef
        assert 0 <= stop_properly_penalty_coef <= 1, (
            f"stop_properly_penalty_coef must be in [0, 1], got {stop_properly_penalty_coef}"
        )
        # Warn user that DAPO overlong parameters are ignored when stop_properly_penalty_coef is set
        ignored_params = []
        if cfg.overlong_buffer_length is not None:
            ignored_params.append("overlong_buffer_length")
        if cfg.overlong_buffer_penalty is not None:
            ignored_params.append("overlong_buffer_penalty")
        if cfg.max_response_length is not None:
            ignored_params.append("max_response_length")
        if ignored_params:
            print(
                f"[WARN] stop_properly_penalty_coef is set, so the following DAPO overlong "
                f"parameters are ignored: {', '.join(ignored_params)}. "
                f"Set stop_properly_penalty_coef=null to use DAPO overlong reward shaping instead.",
                flush=True,
            )
        truncated = batch.get("truncated")
        assert truncated is not None, "truncated field not found in batch"
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=rewards.device)
        else:
            truncated = truncated.to(device=rewards.device)

        num_truncated = truncated.sum().item()
        if num_truncated > 0:
            original_rewards = rewards.clone()
            # For truncated samples, scale the reward by stop_properly_penalty_coef
            rewards = torch.where(
                truncated, rewards * stop_properly_penalty_coef, rewards
            )
            batch["total_reward"] = rewards
            print(
                f"[INFO] stop properly penalty applied: {num_truncated}/{len(truncated)} samples truncated, "
                f"coef={stop_properly_penalty_coef}, "
                f"original_reward_mean={original_rewards[truncated].mean().item():.4f}, "
                f"shaped_reward_mean={rewards[truncated].mean().item():.4f}",
                flush=True,
            )
        else:
            print(
                "[INFO] stop properly penalty: no truncated samples (truncation_rate=0)",
                flush=True,
            )

        return batch

    # DAPO reward shaping requires overlong_buffer_length, overlong_buffer_penalty, and max_response_length to be set.
    overlong_buffer_length = cfg.overlong_buffer_length
    overlong_buffer_penalty = cfg.overlong_buffer_penalty
    max_response_length = cfg.max_response_length
    if (
        overlong_buffer_length is None
        or overlong_buffer_penalty is None
        or max_response_length is None
    ):
        raise ValueError(
            "Reward function is enabled but only DAPO reward shaping is currently supported. Please ensure overlong_buffer_length, overlong_buffer_penalty, and max_response_length are properly configured."
        )

    assert overlong_buffer_penalty >= 0, f"{overlong_buffer_penalty=} must be >=0"
    # Calculate the expected response length
    expected_response_length = max_response_length - overlong_buffer_length

    # Prefer slim per-sample tensor (data-plane path: message_log lives in
    # TQ, slice carries response_token_lengths). Fall back to scanning
    # message_log for the legacy non-data-plane caller.
    response_token_lengths = batch.get("response_token_lengths")
    if response_token_lengths is not None:
        if isinstance(response_token_lengths, torch.Tensor):
            response_lengths = response_token_lengths.tolist()
        else:
            response_lengths = list(response_token_lengths)
    else:
        response_lengths = []
        for message_log in batch["message_log"]:
            length = None
            for message in message_log:
                if message["role"] == "assistant":
                    length = message["token_ids"].shape[0]
                    break
            assert length is not None, (
                "Assistant response not found during reward shaping"
            )
            response_lengths.append(length)

    assert len(response_lengths) == len(rewards), (
        "The number of messages in the batch must match the number of rewards"
    )

    updated_rewards = torch.zeros_like(rewards)
    for i, message_response_length in enumerate(response_lengths):
        # Calculate the exceed length and the corresponding reward penalty
        exceed_length = message_response_length - expected_response_length
        overlong_reward = min(
            -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
        )
        updated_rewards[i] = rewards[i] + overlong_reward

    # Update the rewards in the batch
    batch["total_reward"] = updated_rewards

    return batch
