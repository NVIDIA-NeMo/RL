# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare context-discontinuous rollouts for physical-trace training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.trace_batch_materialization import (
    PreparedTraceBatch,
    prepare_trace_batch,
)
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig


def compute_logical_grpo_advantages(
    advantage_estimator: GRPOAdvantageEstimator,
    *,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Compute one GRPO advantage before physical-row expansion."""
    if not isinstance(advantage_estimator, GRPOAdvantageEstimator):
        raise TypeError(
            "Physical-trace training currently supports only the standard "
            "GRPOAdvantageEstimator"
        )
    if (
        prompt_ids.ndim != 2
        or rewards.ndim != 1
        or prompt_ids.shape[0] != rewards.shape[0]
        or not torch.isfinite(rewards).all()
    ):
        raise ValueError(
            "Physical-trace prompt IDs and rewards must be logical-rollout aligned"
        )
    scalar_mask = torch.ones(
        (rewards.shape[0], 1),
        dtype=rewards.dtype,
        device=rewards.device,
    )
    advantages = advantage_estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=scalar_mask,
    )
    if (
        not isinstance(advantages, torch.Tensor)
        or advantages.shape != scalar_mask.shape
        or not torch.isfinite(advantages).all()
    ):
        raise ValueError(
            "GRPO did not produce one finite scalar advantage per logical rollout"
        )
    return advantages


def physical_trace_materialization_required(
    rollout_batch: Mapping[str, Any],
) -> bool:
    """Report whether any logical rollout contains multiple physical traces.

    Identity batches carry no physical sidecar and stay on the ordinary GRPO
    tensorization path. Split batches carry minimal ownership metadata and are
    fully validated by ``prepare_trace_batch`` before materialization.
    """
    physical_message_logs = rollout_batch.get("physical_message_logs")
    if physical_message_logs is None:
        return False
    message_logs = rollout_batch.get("message_log")
    if (
        not isinstance(physical_message_logs, list)
        or not isinstance(message_logs, list)
        or len(physical_message_logs) != len(message_logs)
    ):
        raise ValueError(
            "Physical message logs must stay aligned with logical message logs"
        )
    for index, rollout_logs in enumerate(physical_message_logs):
        if not isinstance(rollout_logs, list) or not rollout_logs:
            raise ValueError(f"Rollout {index} has no physical message logs")
        if len(rollout_logs) > 1:
            return True
    raise ValueError(
        "Identity-only batches must not carry physical trace training metadata"
    )


def validate_physical_trace_training_config(master_config: "MasterConfig") -> None:
    """Reject semantics that have not been qualified for physical trace rows."""
    errors: list[str] = []
    grpo_config = master_config.grpo
    policy_config = master_config.policy
    loss_config = master_config.loss_fn
    async_grpo_enabled = grpo_config.async_grpo.enabled

    if not master_config.env.get("should_use_nemo_gym"):
        errors.append("env.should_use_nemo_gym must be true")
    if grpo_config.use_dynamic_sampling:
        errors.append("dynamic sampling is not supported")
    if grpo_config.reward_scaling.enabled:
        errors.append("post-rollout reward scaling is not supported")
    if grpo_config.reward_shaping.enabled:
        errors.append("post-rollout reward shaping is not supported")
    if grpo_config.seq_logprob_error_threshold is not None:
        errors.append("sequence-level logprob-error masking is not supported")
    if grpo_config.calculate_advantages_on_gpu:
        errors.append("GPU-side advantage calculation is not supported")
    if (
        grpo_config.advantage_clip_low is not None
        or grpo_config.advantage_clip_high is not None
    ):
        errors.append("post-normalization advantage clipping is not supported")
    if grpo_config.invalid_tool_call_advantage is not None:
        errors.append(
            "message-level invalid-tool advantage overrides are not supported"
        )
    if grpo_config.malformed_thinking_advantage is not None:
        errors.append(
            "message-level malformed-thinking advantage overrides are not supported"
        )

    if grpo_config.adv_estimator.name != "grpo":
        errors.append("only the standard GRPO advantage estimator is supported")
    if loss_config.sequence_level_importance_ratios:
        errors.append("sequence-level importance ratios are not supported")
    if not loss_config.token_level_loss:
        errors.append("sequence-level loss reduction is not supported")
    if loss_config.truncated_importance_sampling_type == "seq-mask-tis":
        errors.append("sequence-level TIS masking is not supported")
    if loss_config.use_kl_in_reward:
        errors.append("KL-in-reward advantage rewriting is not supported")
    if loss_config.positive_example_nll_weight != 0:
        errors.append("positive-example sequence weighting is not supported")
    if async_grpo_enabled and loss_config.force_on_policy_ratio:
        errors.append(
            "async replay requires measured token-level importance ratios; "
            "force_on_policy_ratio is not supported"
        )
    if async_grpo_enabled and not loss_config.use_importance_sampling_correction:
        errors.append(
            "async replay requires token-level importance sampling correction"
        )

    megatron_enabled = policy_config["megatron_cfg"]["enabled"]
    dtensor_enabled = policy_config["dtensor_cfg"]["enabled"]
    if not megatron_enabled or dtensor_enabled:
        errors.append("the initial implementation requires the Megatron policy backend")
    if master_config.data_plane is not None and master_config.data_plane["enabled"]:
        errors.append("the data-plane/TQ training path is not supported")

    logical_rollout_count = (
        grpo_config.num_prompts_per_step * grpo_config.num_generations_per_prompt
    )
    if policy_config["train_global_batch_size"] != logical_rollout_count:
        errors.append(
            "policy.train_global_batch_size must equal the logical rollout count "
            "(num_prompts_per_step * num_generations_per_prompt)"
        )
    if errors:
        raise ValueError(
            "This batch requires physical-trace training, which is incompatible with:\n- "
            + "\n- ".join(errors)
        )


def physical_trace_batch_quantum(
    policy: ColocatablePolicyInterface,
    master_config: "MasterConfig",
) -> int:
    """Return the physical-row divisibility required by policy training."""
    data_parallel_size = getattr(policy, "data_parallel_size", None)
    if (
        isinstance(data_parallel_size, bool)
        or not isinstance(data_parallel_size, int)
        or data_parallel_size <= 0
    ):
        raise ValueError(
            "Physical-trace training requires a positive policy data_parallel_size"
        )
    micro_batch_size = master_config.policy["train_micro_batch_size"]
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size <= 0
    ):
        raise ValueError("policy.train_micro_batch_size must be positive")
    return data_parallel_size * micro_batch_size


@dataclass(frozen=True)
class PhysicalTraceTrainingBatch:
    """Prepared physical rows and their logical-rollout ownership."""

    prepared: PreparedTraceBatch
    logical_advantages: torch.Tensor
    micro_batch_size: int

    @property
    def train_data(self) -> BatchedDataDict[Any]:
        return self.prepared.train_data

    @property
    def logprob_data(self) -> BatchedDataDict[Any]:
        return self.prepared.logprob_data

    @property
    def input_lengths(self) -> torch.Tensor:
        return self.train_data["input_lengths"]

    @property
    def row_rewards(self) -> torch.Tensor:
        return self.prepared.row_rewards

    @property
    def row_count(self) -> int:
        return self.prepared.total_row_count

    @property
    def logical_rollout_count(self) -> int:
        return self.prepared.logical_rollout_count

    @property
    def content(self) -> list[str]:
        return [
            "".join(str(message.get("content", "")) for message in message_log)
            for message_log in self.prepared.materialized_message_logs
        ]

    def metrics(self) -> dict[str, int]:
        return self.prepared.metrics()

    def train_overrides(self) -> dict[str, int]:
        return self.prepared.train_overrides(micro_batch_size=self.micro_batch_size)

    def project_logical_rows(
        self,
        values: list[Any],
        *,
        padding_value: Any = None,
    ) -> list[Any]:
        return self.prepared.project_logical_rows(
            values,
            padding_value=padding_value,
        )


def maybe_prepare_physical_trace_training_batch(
    rollout_batch: Mapping[str, Any],
    *,
    advantage_estimator: GRPOAdvantageEstimator,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    policy: ColocatablePolicyInterface,
    master_config: "MasterConfig",
    pad_token_id: int,
) -> PhysicalTraceTrainingBatch | None:
    """Prepare split trace rows, or return ``None`` for the identity path."""
    if not physical_trace_materialization_required(rollout_batch):
        return None
    validate_physical_trace_training_config(master_config)
    logical_advantages = compute_logical_grpo_advantages(
        advantage_estimator,
        prompt_ids=prompt_ids,
        rewards=rewards,
    )
    prepared = prepare_trace_batch(
        rollout_batch,
        prompt_ids=prompt_ids,
        logical_advantages=logical_advantages,
        expected_rollouts_per_group=(master_config.grpo.num_generations_per_prompt),
        batch_quantum=physical_trace_batch_quantum(policy, master_config),
        pad_token_id=pad_token_id,
        mask_truncated=master_config.grpo.overlong_filtering,
        make_sequence_length_divisible_by=master_config.policy[
            "make_sequence_length_divisible_by"
        ],
        require_generation_policy_version=True,
    )
    return PhysicalTraceTrainingBatch(
        prepared=prepared,
        logical_advantages=logical_advantages,
        micro_batch_size=master_config.policy["train_micro_batch_size"],
    )
