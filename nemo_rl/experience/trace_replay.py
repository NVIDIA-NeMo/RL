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

"""Identity and validation for split physical-trace replay groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


_REPLAY_CONTENT_FIELDS = (
    "rollout_id",
    "group_id",
    "generation_policy_version",
    "message_log",
    "physical_message_logs",
    "physical_trace_ids",
    "total_reward",
    "loss_multiplier",
    "mask_sample",
    "truncated",
)


def _exact_value_equal(left: Any, right: Any) -> bool:
    """Compare replay content without serializing tensor or media payloads."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, PackedTensor) or isinstance(right, PackedTensor):
        return (
            isinstance(left, PackedTensor)
            and isinstance(right, PackedTensor)
            and left.dim_to_pack == right.dim_to_pack
            and left.pad_to_max_shape == right.pad_to_max_shape
            and _exact_value_equal(left.tensors, right.tensors)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_exact_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) or isinstance(right, Sequence):
        if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
            return left == right
        return (
            isinstance(left, Sequence)
            and isinstance(right, Sequence)
            and len(left) == len(right)
            and all(
                _exact_value_equal(left_value, right_value)
                for left_value, right_value in zip(left, right, strict=True)
            )
        )
    return bool(left == right)


def trace_replay_group_identity(
    trajectory: Mapping[str, Any],
) -> frozenset[str] | None:
    """Return caller-owned rollout IDs for a split replay group."""
    batch = trajectory.get("batch")
    rollout_ids_column = (
        batch.get("rollout_id") if isinstance(batch, BatchedDataDict) else None
    )
    if rollout_ids_column is None:
        return None
    if not isinstance(rollout_ids_column, list) or not rollout_ids_column:
        raise ValueError("Replay group has no rollout identities")
    rollout_ids = frozenset(str(rollout_id) for rollout_id in rollout_ids_column)
    if len(rollout_ids) != len(rollout_ids_column) or "" in rollout_ids:
        raise ValueError("Replay group has incomplete or duplicate identities")
    return rollout_ids


def normalize_mixed_physical_trace_groups(
    trajectories: Sequence[Mapping[str, Any]],
) -> list[BatchedDataDict[Any]] | None:
    """Return optimizer-step copies with identity rows added beside split groups."""
    batches = [trajectory.get("batch") for trajectory in trajectories]
    if any(not isinstance(batch, BatchedDataDict) for batch in batches):
        raise TypeError("Replay trajectories must carry BatchedDataDict batches")
    typed_batches = [batch for batch in batches if isinstance(batch, BatchedDataDict)]
    if not any("physical_message_logs" in batch for batch in typed_batches):
        return None

    prepared_batches: list[BatchedDataDict[Any]] = []
    for batch in typed_batches:
        if "physical_message_logs" in batch:
            if "physical_trace_ids" not in batch:
                raise ValueError("Split replay group has no physical trace IDs")
            prepared_batches.append(batch)
            continue
        message_logs = batch.get("message_log")
        rollout_ids = batch.get("rollout_id")
        if (
            not isinstance(message_logs, list)
            or not isinstance(rollout_ids, list)
            or len(message_logs) != len(rollout_ids)
        ):
            raise ValueError("Identity replay group cannot be expanded safely")
        prepared_batch = BatchedDataDict(dict(batch.items()))
        prepared_batch["physical_message_logs"] = [
            [message_log] for message_log in message_logs
        ]
        prepared_batch["physical_trace_ids"] = [
            [f"{rollout_id}:trace-000000"] for rollout_id in rollout_ids
        ]
        prepared_batches.append(prepared_batch)
    return prepared_batches


def trace_replay_content_equal(
    left_trajectory: Mapping[str, Any],
    right_trajectory: Mapping[str, Any],
) -> bool:
    """Compare canonical training content after rollout identities match."""
    left_batch = left_trajectory.get("batch")
    right_batch = right_trajectory.get("batch")
    if not isinstance(left_batch, BatchedDataDict) or not isinstance(
        right_batch, BatchedDataDict
    ):
        return False
    return all(
        field in left_batch
        and field in right_batch
        and _exact_value_equal(left_batch[field], right_batch[field])
        for field in _REPLAY_CONTENT_FIELDS
        if field in left_batch or field in right_batch
    )


def validate_trace_replay_groups(
    trajectories: list[dict[str, Any]],
    *,
    expected_prompt_groups: int,
    expected_rollouts_per_group: int,
    current_weight_version: int,
    max_age_steps: int,
) -> None:
    """Prove replay selected complete logical groups with stable provenance."""
    if len(trajectories) != expected_prompt_groups:
        raise ValueError(
            "Async split replay must return exactly "
            f"{expected_prompt_groups} complete prompt groups"
        )

    observed_rollout_ids: set[str] = set()
    for group_index, trajectory in enumerate(trajectories):
        batch = trajectory.get("batch")
        if (
            not isinstance(batch, BatchedDataDict)
            or batch.size != expected_rollouts_per_group
        ):
            raise ValueError(
                f"Async split replay group {group_index} must contain exactly "
                f"{expected_rollouts_per_group} logical rollouts"
            )

        generation_weight_version = trajectory.get("generation_weight_version")
        target_weight_version = trajectory.get("target_weight_version")
        if (
            isinstance(generation_weight_version, bool)
            or not isinstance(generation_weight_version, int)
            or isinstance(target_weight_version, bool)
            or not isinstance(target_weight_version, int)
        ):
            raise ValueError(
                f"Async split replay group {group_index} is missing integer "
                "generation/target weight provenance"
            )
        if target_weight_version != current_weight_version:
            raise ValueError(
                f"Async split replay group {group_index} targets weight "
                f"{target_weight_version}, expected {current_weight_version}"
            )
        trajectory_age = current_weight_version - generation_weight_version
        if trajectory_age < 0 or trajectory_age > max_age_steps:
            raise ValueError(
                f"Async split replay group {group_index} has invalid trajectory age "
                f"{trajectory_age}"
            )

        rollout_ids = batch.get("rollout_id")
        group_ids = batch.get("group_id")
        policy_versions = batch.get("generation_policy_version")
        if (
            not isinstance(rollout_ids, list)
            or len(rollout_ids) != batch.size
            or not isinstance(group_ids, list)
            or len(group_ids) != batch.size
            or not isinstance(policy_versions, list)
            or len(policy_versions) != batch.size
        ):
            raise ValueError(
                f"Async split replay group {group_index} has no rollout-aligned "
                "identity or provenance"
            )
        observed_group_ids: set[str] = set()
        expected_policy_version = f"async-policy-weight-{generation_weight_version:08d}"
        for rollout_id, group_id, policy_version in zip(
            rollout_ids, group_ids, policy_versions, strict=True
        ):
            if not isinstance(rollout_id, str) or not rollout_id:
                raise ValueError("Async replay row has no rollout identity")
            if rollout_id in observed_rollout_ids:
                raise ValueError(
                    "Async split replay selected duplicate logical rollout ID "
                    f"{rollout_id!r}"
                )
            observed_rollout_ids.add(rollout_id)
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(
                    f"Async replay row {rollout_id!r} has no group identity"
                )
            observed_group_ids.add(group_id)
            if policy_version != expected_policy_version:
                raise ValueError(
                    f"Async replay row {rollout_id!r} policy provenance "
                    "disagrees with its replay entry"
                )
        if len(observed_group_ids) != 1:
            raise ValueError(
                f"Async split replay group {group_index} mixes comparison groups "
                f"{sorted(observed_group_ids)!r}"
            )
