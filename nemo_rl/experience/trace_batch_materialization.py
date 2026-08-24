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

"""Project logical rollouts onto exact physical training rows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
)


@dataclass(frozen=True)
class PreparedTraceBatch:
    """Worker tensors and the minimal logical-to-physical row projection."""

    train_data: BatchedDataDict[Any]
    logprob_data: BatchedDataDict[Any]
    materialized_message_logs: list[list[dict[str, Any]]]
    parent_indices: torch.Tensor
    row_rewards: torch.Tensor
    row_rollout_ids: list[str | None]
    row_trace_ids: list[str | None]
    rollout_advantages: dict[str, float]
    logical_rollout_count: int
    physical_trace_count: int
    padding_row_count: int
    eligible_action_token_count: int
    declared_boundary_count: int
    inferred_boundary_count: int

    @property
    def total_row_count(self) -> int:
        """Return real physical rows plus synthetic padding rows."""
        return self.physical_trace_count + self.padding_row_count

    def metrics(self) -> dict[str, int]:
        """Return logical-to-physical expansion metrics."""
        return {
            "physical_trace_training/logical_rollouts": self.logical_rollout_count,
            "physical_trace_training/physical_traces": self.physical_trace_count,
            "physical_trace_training/padding_rows": self.padding_row_count,
            "physical_trace_training/physical_rows": self.total_row_count,
            "physical_trace_training/eligible_action_tokens": (
                self.eligible_action_token_count
            ),
            "physical_trace_training/declared_boundaries": (
                self.declared_boundary_count
            ),
            "physical_trace_training/inferred_discontinuities": (
                self.inferred_boundary_count
            ),
        }

    def project_logical_rows(
        self,
        values: Sequence[Any],
        *,
        padding_value: Any = None,
    ) -> list[Any]:
        """Project rollout-aligned values onto physical and padding rows."""
        if len(values) != self.logical_rollout_count:
            raise ValueError("Logical values do not match the prepared trace batch")
        return [
            values[parent_index] if parent_index >= 0 else padding_value
            for parent_index in self.parent_indices.tolist()
        ]

    def train_overrides(self, *, micro_batch_size: int) -> dict[str, int]:
        """Keep physical optimizer rows and logical scheduler progress explicit."""
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        return {
            "gbs": self.total_row_count,
            "mbs": micro_batch_size,
            "scheduler_step_increment": self.logical_rollout_count,
        }


@dataclass(frozen=True)
class _PhysicalRow:
    parent_index: int
    rollout_id: str
    trace_id: str
    reward: float
    advantage: float
    sample_mask: float
    message_log: list[dict[str, Any]]


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _copy_canonical_message_log(
    message_log: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
) -> list[dict[str, Any]]:
    """Validate and copy one canonical physical message log."""
    if not message_log:
        raise ValueError(f"Physical trace {trace_id!r} has no messages")

    copied_log: list[dict[str, Any]] = []
    for message_index, message in enumerate(message_log):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"Physical trace {trace_id!r} message {message_index} is not a mapping"
            )
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"Physical trace {trace_id!r} message {message_index} has invalid role"
            )
        token_ids = message.get("token_ids")
        token_loss_mask = message.get("token_loss_mask")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError(
                f"Physical trace {trace_id!r} message {message_index} has invalid tokens"
            )
        if (
            not isinstance(token_loss_mask, torch.Tensor)
            or token_loss_mask.shape != token_ids.shape
        ):
            raise ValueError(
                f"Physical trace {trace_id!r} message {message_index} has an "
                "unaligned token loss mask"
            )

        copied = dict(message)
        generation_logprobs = message.get("generation_logprobs")
        if generation_logprobs is None:
            copied["generation_logprobs"] = torch.zeros(
                token_ids.shape,
                dtype=torch.float32,
                device=token_ids.device,
            )
        elif (
            not isinstance(generation_logprobs, torch.Tensor)
            or generation_logprobs.shape != token_ids.shape
        ):
            raise ValueError(
                f"Physical trace {trace_id!r} message {message_index} has "
                "unaligned generation logprobs"
            )

        routed_experts = message.get("routed_experts")
        if routed_experts is not None and (
            not isinstance(routed_experts, torch.Tensor)
            or routed_experts.ndim != 3
            or routed_experts.shape[0] != token_ids.shape[0]
        ):
            raise ValueError(
                f"Physical trace {trace_id!r} message {message_index} has "
                "misaligned routed_experts"
            )
        copied_log.append(copied)
    return copied_log


def _packed_tensor_specs(
    message_logs: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[int, bool]]:
    specs: dict[str, tuple[int, bool]] = {}
    for message_log in message_logs:
        for message in message_log:
            for key, value in message.items():
                if not isinstance(value, PackedTensor):
                    continue
                spec = (value.dim_to_pack, value.pad_to_max_shape)
                previous = specs.setdefault(key, spec)
                if previous != spec:
                    raise ValueError(
                        f"Multimodal key {key!r} has inconsistent packing: "
                        f"{previous!r} versus {spec!r}"
                    )
    return specs


def _normalize_packed_tensor_rows(
    message_logs: list[list[dict[str, Any]]],
) -> None:
    specs = _packed_tensor_specs(message_logs)
    for row_index, message_log in enumerate(message_logs):
        if not message_log:
            raise ValueError(f"Materialized row {row_index} has no messages")
        for key, (dim_to_pack, pad_to_max_shape) in specs.items():
            values = [message[key] for message in message_log if key in message]
            if any(not isinstance(value, PackedTensor) for value in values):
                raise TypeError(
                    f"Materialized row {row_index} key {key!r} mixes packed "
                    "and non-packed values"
                )
            if not values:
                message_log[0][key] = PackedTensor(
                    [None],
                    dim_to_pack=dim_to_pack,
                    pad_to_max_shape=pad_to_max_shape,
                )


def _normalize_routed_expert_rows(
    message_logs: list[list[dict[str, Any]]],
    *,
    physical_trace_count: int,
) -> None:
    physical_messages = [
        message
        for message_log in message_logs[:physical_trace_count]
        for message in message_log
    ]
    routed_values = [message.get("routed_experts") for message in physical_messages]
    if not any(value is not None for value in routed_values):
        return
    if any(value is None for value in routed_values):
        raise ValueError(
            "routed_experts must cover every physical token-bearing message"
        )
    routed_tensors = [
        value for value in routed_values if isinstance(value, torch.Tensor)
    ]
    if len(routed_tensors) != len(routed_values):
        raise TypeError("routed_experts values must be tensors")
    first = routed_tensors[0]
    trailing_shape = first.shape[1:]
    if any(
        value.shape[1:] != trailing_shape
        or value.dtype != first.dtype
        or value.device != first.device
        for value in routed_tensors
    ):
        raise ValueError("routed_experts tensors have inconsistent worker shapes")
    for message_log in message_logs[physical_trace_count:]:
        message_log[0]["routed_experts"] = torch.full(
            (1, *trailing_shape),
            ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
            dtype=first.dtype,
            device=first.device,
        )


def _rollout_aligned_values(
    rollout_batch: Mapping[str, Any],
    key: str,
    *,
    rollout_count: int,
) -> list[Any] | None:
    value = rollout_batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.shape[0] != rollout_count:
            raise ValueError(
                f"Physical-trace batch field {key!r} is not rollout-aligned"
            )
        return value.tolist()
    if isinstance(value, list) and len(value) == rollout_count:
        return value
    raise ValueError(f"Physical-trace batch field {key!r} is not rollout-aligned")


def _logical_sample_masks(
    rollout_batch: Mapping[str, Any],
    *,
    rollout_count: int,
    mask_truncated: bool,
) -> list[float]:
    loss_multipliers = _rollout_aligned_values(
        rollout_batch,
        "loss_multiplier",
        rollout_count=rollout_count,
    )
    if loss_multipliers is None:
        raise ValueError("Physical-trace training requires loss_multiplier")
    sample_masks = [
        _finite_float(value, field=f"loss_multiplier[{index}]")
        for index, value in enumerate(loss_multipliers)
    ]

    mask_sample = _rollout_aligned_values(
        rollout_batch,
        "mask_sample",
        rollout_count=rollout_count,
    )
    if mask_sample is not None:
        sample_masks = [
            0.0 if bool(masked) else value
            for value, masked in zip(sample_masks, mask_sample, strict=True)
        ]

    truncated = _rollout_aligned_values(
        rollout_batch,
        "truncated",
        rollout_count=rollout_count,
    )
    if mask_truncated and truncated is not None:
        sample_masks = [
            0.0 if bool(is_truncated) else value
            for value, is_truncated in zip(sample_masks, truncated, strict=True)
        ]
    return sample_masks


def _validate_prepared_tensors(
    prepared: PreparedTraceBatch,
    *,
    pad_token_id: int,
) -> None:
    train_data = prepared.train_data
    input_ids = train_data["input_ids"]
    input_lengths = train_data["input_lengths"]
    generation_logprobs = train_data["generation_logprobs"]
    token_mask = train_data["token_mask"]
    sample_mask = train_data["sample_mask"]
    advantages = train_data["advantages"]
    row_count = prepared.total_row_count
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != row_count
        or generation_logprobs.shape != input_ids.shape
        or token_mask.shape != input_ids.shape
        or advantages.shape != input_ids.shape
        or input_lengths.shape != (row_count,)
        or sample_mask.shape != (row_count,)
        or prepared.parent_indices.shape != (row_count,)
        or prepared.row_rewards.shape != (row_count,)
    ):
        raise ValueError(
            "Prepared physical-trace worker fields have inconsistent shapes"
        )
    if not torch.isfinite(sample_mask).all() or not torch.isfinite(advantages).all():
        raise ValueError("Prepared physical-trace masks or advantages are not finite")

    for row_index in range(prepared.physical_trace_count, row_count):
        if (
            int(input_lengths[row_index].item()) != 1
            or torch.any(input_ids[row_index] != pad_token_id)
            or torch.count_nonzero(token_mask[row_index]).item() != 0
            or torch.count_nonzero(generation_logprobs[row_index]).item() != 0
            or torch.count_nonzero(advantages[row_index]).item() != 0
            or sample_mask[row_index].item() != 0.0
            or prepared.parent_indices[row_index].item() != -1
            or prepared.row_rewards[row_index].item() != 0.0
        ):
            raise ValueError(f"Prepared padding row {row_index} is not fully masked")


def prepare_trace_batch(
    rollout_batch: Mapping[str, Any],
    *,
    prompt_ids: torch.Tensor,
    logical_advantages: torch.Tensor,
    expected_rollouts_per_group: int,
    batch_quantum: int,
    pad_token_id: int,
    mask_truncated: bool,
    make_sequence_length_divisible_by: int,
    require_generation_policy_version: bool,
) -> PreparedTraceBatch:
    """Validate logical ownership and materialize exact physical training rows."""
    expected_rollouts_per_group = _require_positive_int(
        expected_rollouts_per_group,
        field="expected_rollouts_per_group",
    )
    batch_quantum = _require_positive_int(batch_quantum, field="batch_quantum")
    make_sequence_length_divisible_by = _require_positive_int(
        make_sequence_length_divisible_by,
        field="make_sequence_length_divisible_by",
    )
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer")

    rollout_ids_column = rollout_batch.get("rollout_id")
    group_ids_column = rollout_batch.get("group_id")
    policy_versions_column = rollout_batch.get("generation_policy_version")
    physical_trace_ids = rollout_batch.get("physical_trace_ids")
    physical_message_logs = rollout_batch.get("physical_message_logs")
    logical_message_logs = rollout_batch.get("message_log")
    if not isinstance(rollout_ids_column, list) or not rollout_ids_column:
        raise ValueError("Physical-trace training requires rollout identities")
    rollout_count = len(rollout_ids_column)
    if (
        not isinstance(group_ids_column, list)
        or len(group_ids_column) != rollout_count
        or not isinstance(policy_versions_column, list)
        or len(policy_versions_column) != rollout_count
        or not isinstance(physical_trace_ids, list)
        or len(physical_trace_ids) != rollout_count
        or not isinstance(physical_message_logs, list)
        or len(physical_message_logs) != rollout_count
    ):
        raise ValueError("Physical-trace fields must be rollout-aligned")
    if (
        not isinstance(logical_message_logs, list)
        or len(logical_message_logs) != rollout_count
    ):
        raise ValueError("Logical message logs must be rollout-aligned")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != rollout_count:
        raise ValueError("Physical-trace prompt IDs must be rollout-aligned")
    if (
        logical_advantages.shape != (rollout_count, 1)
        or not logical_advantages.is_floating_point()
        or not torch.isfinite(logical_advantages).all()
    ):
        raise ValueError(
            "Physical-trace training requires one finite advantage per rollout"
        )
    rewards = rollout_batch.get("total_reward")
    if (
        not isinstance(rewards, torch.Tensor)
        or rewards.shape != (rollout_count,)
        or not torch.isfinite(rewards).all()
    ):
        raise ValueError(
            "Physical-trace training requires one finite reward per rollout"
        )

    logical_sample_masks = _logical_sample_masks(
        rollout_batch,
        rollout_count=rollout_count,
        mask_truncated=mask_truncated,
    )
    group_prompts: dict[str, torch.Tensor] = {}
    group_counts: dict[str, int] = {}
    rollout_ids: set[str] = set()
    trace_ids: set[str] = set()
    generation_policy_versions: set[str] = set()
    rows: list[_PhysicalRow] = []
    rollout_advantages: dict[str, float] = {}
    declared_boundary_count = 0
    inferred_boundary_count = 0

    for parent_index, (
        rollout_id,
        group_id,
        policy_version,
        raw_trace_ids,
        rollout_logs,
    ) in enumerate(
        zip(
            rollout_ids_column,
            group_ids_column,
            policy_versions_column,
            physical_trace_ids,
            physical_message_logs,
            strict=True,
        )
    ):
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError(f"Physical row {parent_index} has no rollout identity")
        if rollout_id in rollout_ids:
            raise ValueError(f"Duplicate logical rollout ID {rollout_id!r}")
        rollout_ids.add(rollout_id)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"Physical row {parent_index} has no group identity")
        group_prompt = group_prompts.setdefault(group_id, prompt_ids[parent_index])
        if not torch.equal(group_prompt, prompt_ids[parent_index]):
            raise ValueError(
                f"Comparison group {group_id!r} owns more than one tokenized prompt"
            )
        group_counts[group_id] = group_counts.get(group_id, 0) + 1

        if policy_version is not None:
            if not isinstance(policy_version, str) or not policy_version:
                raise ValueError(
                    f"Trace metadata {rollout_id!r} has invalid policy provenance"
                )
            generation_policy_versions.add(policy_version)

        rollout_reward = float(rewards[parent_index].item())
        advantage = float(logical_advantages[parent_index, 0].item())
        rollout_advantages[rollout_id] = advantage

        if (
            not isinstance(raw_trace_ids, list)
            or not raw_trace_ids
            or any(
                not isinstance(trace_id, str) or not trace_id
                for trace_id in raw_trace_ids
            )
        ):
            raise ValueError(f"Trace metadata {rollout_id!r} has invalid trace IDs")
        if not isinstance(rollout_logs, list) or len(rollout_logs) != len(
            raw_trace_ids
        ):
            raise ValueError(
                f"Rollout {rollout_id!r} physical message logs are incomplete"
            )
        for trace_index, (trace_id, message_log) in enumerate(
            zip(raw_trace_ids, rollout_logs, strict=True)
        ):
            if trace_id in trace_ids:
                raise ValueError(f"Duplicate physical trace ID {trace_id!r}")
            trace_ids.add(trace_id)
            if not isinstance(message_log, Sequence):
                raise TypeError(f"Physical trace {trace_id!r} has no message log")
            rows.append(
                _PhysicalRow(
                    parent_index=parent_index,
                    rollout_id=rollout_id,
                    trace_id=trace_id,
                    reward=rollout_reward,
                    advantage=advantage,
                    sample_mask=logical_sample_masks[parent_index],
                    message_log=_copy_canonical_message_log(
                        message_log,
                        trace_id=trace_id,
                    ),
                )
            )
        declared_boundary_count += len(raw_trace_ids) - 1

    incomplete_groups = {
        group_id: count
        for group_id, count in group_counts.items()
        if count != expected_rollouts_per_group
    }
    if incomplete_groups:
        raise ValueError(
            "Physical-trace comparison groups are incomplete: "
            f"expected={expected_rollouts_per_group}, observed={incomplete_groups!r}"
        )
    if require_generation_policy_version and len(generation_policy_versions) != 1:
        raise ValueError(
            "One physical optimizer batch requires one generation policy version"
        )
    if inferred_boundary_count:
        raise ValueError("Physical training cannot consume inferred discontinuities")

    physical_trace_count = len(rows)
    padding_row_count = (-physical_trace_count) % batch_quantum
    message_logs = [row.message_log for row in rows]
    message_logs.extend(
        [
            {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor([pad_token_id], dtype=torch.int64),
                "token_loss_mask": torch.zeros(1, dtype=torch.int64),
                "generation_logprobs": torch.zeros(1, dtype=torch.float32),
            }
        ]
        for _ in range(padding_row_count)
    )
    # Router capture is attached only to model calls. Physical logs are copied
    # before the ordinary logical-log backfill runs, so normalize their missing
    # prompt/tool rows independently before flattening.
    from nemo_rl.experience.rollouts import backfill_missing_routed_experts

    backfill_missing_routed_experts(message_logs[:physical_trace_count])
    _normalize_packed_tensor_rows(message_logs)
    _normalize_routed_expert_rows(
        message_logs,
        physical_trace_count=physical_trace_count,
    )
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        pad_value_dict={"token_ids": pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    for key in ("token_ids", "token_loss_mask", "generation_logprobs"):
        if key not in flat:
            raise ValueError(f"Materialized physical rows are missing {key!r}")

    row_count, sequence_length = flat["token_ids"].shape
    if row_count != physical_trace_count + padding_row_count:
        raise ValueError("Materialized physical row count is inconsistent")
    parent_indices = torch.tensor(
        [row.parent_index for row in rows] + [-1] * padding_row_count,
        dtype=torch.int64,
        device=flat["token_ids"].device,
    )
    row_advantages = torch.tensor(
        [row.advantage for row in rows] + [0.0] * padding_row_count,
        dtype=torch.float32,
        device=flat["token_ids"].device,
    )
    row_rewards = torch.tensor(
        [row.reward for row in rows] + [0.0] * padding_row_count,
        dtype=torch.float32,
        device=flat["token_ids"].device,
    )
    train_data = BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "generation_logprobs": flat["generation_logprobs"],
            "token_mask": flat["token_loss_mask"],
            "sample_mask": torch.tensor(
                [row.sample_mask for row in rows] + [0.0] * padding_row_count,
                dtype=torch.float32,
                device=flat["token_ids"].device,
            ),
            "advantages": row_advantages.unsqueeze(-1)
            .expand(row_count, sequence_length)
            .clone(),
        }
    )
    train_data.update(flat.get_multimodal_dict(as_tensors=False))
    if "routed_experts" in flat:
        train_data["routed_experts"] = flat["routed_experts"]

    logprob_data = BatchedDataDict(
        {
            "input_ids": train_data["input_ids"],
            "input_lengths": train_data["input_lengths"],
            "token_mask": train_data["token_mask"],
            "sample_mask": train_data["sample_mask"],
        }
    )
    logprob_data.update(train_data.get_multimodal_dict(as_tensors=False))
    if "routed_experts" in train_data:
        logprob_data["routed_experts"] = train_data["routed_experts"]

    prepared = PreparedTraceBatch(
        train_data=train_data,
        logprob_data=logprob_data,
        materialized_message_logs=message_logs,
        parent_indices=parent_indices,
        row_rewards=row_rewards,
        row_rollout_ids=[row.rollout_id for row in rows] + [None] * padding_row_count,
        row_trace_ids=[row.trace_id for row in rows] + [None] * padding_row_count,
        rollout_advantages=rollout_advantages,
        logical_rollout_count=rollout_count,
        physical_trace_count=physical_trace_count,
        padding_row_count=padding_row_count,
        eligible_action_token_count=int(train_data["token_mask"].sum().item()),
        declared_boundary_count=declared_boundary_count,
        inferred_boundary_count=inferred_boundary_count,
    )
    if prepared.eligible_action_token_count <= 0:
        raise ValueError("Prepared physical batch has no eligible action tokens")
    _validate_prepared_tensors(prepared, pad_token_id=pad_token_id)
    return prepared
