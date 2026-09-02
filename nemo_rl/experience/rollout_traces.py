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

"""Logical-rollout to physical-trace batch planning.

This module deliberately stops before tensor collation, policy/reference
scoring, or training.  It validates the statistical ownership of exact
physical traces and emits a JSON-serializable plan that a later materializer
can use to build worker-facing rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence, TypedDict

from nemo_rl.environments.nemo_gym_trace import (
    validate_rollout_trace_bundle,
    validate_rollout_trace_group,
)


_TRACE_BATCH_PLAN_SCHEMA_VERSION = 1
_LOSS_NORMALIZATION = "global_action_token_mean"
_SUPPORTED_ADVANTAGE_ESTIMATORS = frozenset({"grpo", "reinforce_baseline"})


class TraceBatchRowPlan(TypedDict):
    """One physical trace row or a fully masked padding row."""

    row_index: int
    row_kind: str
    parent_rollout_index: int
    rollout_id: str | None
    source_row_index: int | None
    group_id: str | None
    trace_id: str | None
    trace_index: int
    reward: float
    advantage: float
    sample_mask: float
    token_count: int
    eligible_token_count: int
    completion_ids: list[str]
    ordered_media_ids: list[str]


class TraceBatchPlan(TypedDict):
    """JSON-serializable ownership and padding plan for one optimizer step."""

    schema_version: int
    plan_id: str
    optimizer_step_id: str
    generation_contract_id: str
    training_admission_contract_id: str | None
    training_admission_contract_ids: list[str]
    advantage_estimator_name: str
    loss_normalization: str
    training_admitted: bool
    sequence_level_ratios_enabled: bool
    sequence_level_clipping_enabled: bool
    expected_rollouts_per_group: int
    batch_quantum: int
    comparison_group_count: int
    logical_rollout_count: int
    physical_trace_count: int
    padding_row_count: int
    total_row_count: int
    eligible_action_token_count: int
    duplicate_retry_count: int
    group_ids: list[str]
    rollout_ids: list[str]
    parent_indices: list[int]
    rollout_to_rows: list[list[int]]
    rows: list[TraceBatchRowPlan]


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def build_trace_batch_plan(
    bundles: Sequence[Mapping[str, Any]],
    *,
    rollout_advantages: Mapping[str, float],
    expected_rollouts_per_group: int,
    batch_quantum: int,
    optimizer_step_id: str,
    training_admission: bool = False,
    advantage_estimator_name: str = "grpo",
    sequence_level_ratios_enabled: bool = False,
    sequence_level_clipping_enabled: bool = False,
) -> TraceBatchPlan:
    """Build a fail-closed physical-row plan from complete logical rollouts.

    ``rollout_advantages`` must already contain one scalar advantage per unique
    logical rollout. The planner never recomputes group-relative statistics
    after expanding a rollout into multiple rows.
    """
    expected_rollouts_per_group = _require_positive_int(
        expected_rollouts_per_group,
        field="expected_rollouts_per_group",
    )
    batch_quantum = _require_positive_int(batch_quantum, field="batch_quantum")
    if not isinstance(optimizer_step_id, str) or not optimizer_step_id:
        raise ValueError("optimizer_step_id must be a non-empty string")
    if advantage_estimator_name not in _SUPPORTED_ADVANTAGE_ESTIMATORS:
        raise ValueError(
            "Multi-trace planning does not support advantage estimator "
            f"{advantage_estimator_name!r}; supported="
            f"{sorted(_SUPPORTED_ADVANTAGE_ESTIMATORS)!r}"
        )
    if sequence_level_ratios_enabled or sequence_level_clipping_enabled:
        raise ValueError(
            "Sequence-level ratios and clipping are disabled until their "
            "multi-trace semantics are explicitly qualified"
        )
    if not bundles:
        raise ValueError("TraceBatchPlan requires at least one rollout bundle")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for bundle in bundles:
        group_id = bundle.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Every rollout bundle must have a non-empty group_id")
        grouped.setdefault(group_id, []).append(bundle)

    duplicate_retry_count = 0
    generation_contract_ids: set[str] = set()
    training_admission_contract_ids: set[str] = set()
    for group_id, group_bundles in grouped.items():
        summary = validate_rollout_trace_group(
            group_bundles,
            expected_group_id=group_id,
            training_admission=training_admission,
        )
        if summary["unique_rollout_count"] != expected_rollouts_per_group:
            raise ValueError(
                f"Comparison group {group_id!r} is incomplete: expected "
                f"{expected_rollouts_per_group} unique rollouts, observed "
                f"{summary['unique_rollout_count']}"
            )
        duplicate_retry_count += int(summary["duplicate_retry_count"])
        generation_contract_ids.add(str(summary["generation_contract_id"]))
        if summary["training_admission_contract_id"] is not None:
            training_admission_contract_ids.add(
                str(summary["training_admission_contract_id"])
            )

    if len(generation_contract_ids) != 1:
        raise ValueError(
            "One TraceBatchPlan cannot mix generation contracts: "
            f"{sorted(generation_contract_ids)!r}"
        )
    sorted_training_admission_contract_ids = sorted(
        training_admission_contract_ids
    )
    if training_admission and not sorted_training_admission_contract_ids:
        raise ValueError("Training-admitted TraceBatchPlan has no admission identity")
    if len(sorted_training_admission_contract_ids) == 1:
        training_admission_contract_id = (
            sorted_training_admission_contract_ids[0]
        )
    elif sorted_training_admission_contract_ids:
        training_admission_contract_id = (
            "training-admission-batch-contract-"
            + _canonical_digest(
                {"contract_ids": sorted_training_admission_contract_ids}
            )[:24]
        )
    else:
        training_admission_contract_id = None

    unique_bundles: list[Mapping[str, Any]] = []
    seen_rollout_groups: dict[str, str] = {}
    for bundle in bundles:
        rollout_id = bundle.get("rollout_id")
        group_id = bundle.get("group_id")
        assert isinstance(rollout_id, str)
        assert isinstance(group_id, str)
        previous_group = seen_rollout_groups.get(rollout_id)
        if previous_group is not None:
            if previous_group != group_id:
                raise ValueError(
                    f"Rollout ID {rollout_id!r} appears in multiple groups"
                )
            # The group validator already proved that repeated content with
            # this identity is byte-for-byte equivalent.
            continue
        seen_rollout_groups[rollout_id] = group_id
        unique_bundles.append(bundle)

    expected_advantage_ids = set(seen_rollout_groups)
    observed_advantage_ids = set(rollout_advantages)
    if observed_advantage_ids != expected_advantage_ids:
        missing = sorted(expected_advantage_ids - observed_advantage_ids)
        extra = sorted(observed_advantage_ids - expected_advantage_ids)
        raise ValueError(
            "Rollout advantages must match unique logical rollouts exactly: "
            f"missing={missing!r}, extra={extra!r}"
        )
    advantages = {
        rollout_id: _finite_float(
            rollout_advantages[rollout_id],
            field=f"advantage[{rollout_id!r}]",
        )
        for rollout_id in seen_rollout_groups
    }

    rows: list[TraceBatchRowPlan] = []
    rollout_ids: list[str] = []
    rollout_to_rows: list[list[int]] = []
    eligible_action_token_count = 0
    for parent_rollout_index, bundle in enumerate(unique_bundles):
        rollout_id = str(bundle["rollout_id"])
        group_id = str(bundle["group_id"])
        reward = _finite_float(bundle.get("reward"), field=f"reward[{rollout_id!r}]")
        rollout_ids.append(rollout_id)
        owned_rows: list[int] = []
        for trace in bundle["physical_traces"]:
            row_index = len(rows)
            token_count = int(trace["token_count"])
            eligible_token_count = int(trace["trainable_token_count"])
            completion_ids = [
                str(span["completion_id"]) for span in trace["completion_spans"]
            ]
            row: TraceBatchRowPlan = {
                "row_index": row_index,
                "row_kind": "physical_trace",
                "parent_rollout_index": parent_rollout_index,
                "rollout_id": rollout_id,
                "source_row_index": bundle.get("source_row_index"),
                "group_id": group_id,
                "trace_id": str(trace["trace_id"]),
                "trace_index": int(trace["trace_index"]),
                "reward": reward,
                "advantage": advantages[rollout_id],
                "sample_mask": 1.0,
                "token_count": token_count,
                "eligible_token_count": eligible_token_count,
                "completion_ids": completion_ids,
                "ordered_media_ids": [
                    str(media_id) for media_id in trace["ordered_media_ids"]
                ],
            }
            rows.append(row)
            owned_rows.append(row_index)
            eligible_action_token_count += eligible_token_count
        rollout_to_rows.append(owned_rows)

    physical_trace_count = len(rows)
    if eligible_action_token_count <= 0:
        raise ValueError("TraceBatchPlan has no eligible action tokens")
    padding_row_count = (-physical_trace_count) % batch_quantum
    for _ in range(padding_row_count):
        rows.append(
            {
                "row_index": len(rows),
                "row_kind": "padding",
                "parent_rollout_index": -1,
                "rollout_id": None,
                "source_row_index": None,
                "group_id": None,
                "trace_id": None,
                "trace_index": -1,
                "reward": 0.0,
                "advantage": 0.0,
                "sample_mask": 0.0,
                "token_count": 0,
                "eligible_token_count": 0,
                "completion_ids": [],
                "ordered_media_ids": [],
            }
        )

    plan_without_id: dict[str, Any] = {
        "schema_version": _TRACE_BATCH_PLAN_SCHEMA_VERSION,
        "optimizer_step_id": optimizer_step_id,
        "generation_contract_id": next(iter(generation_contract_ids)),
        "training_admission_contract_id": training_admission_contract_id,
        "training_admission_contract_ids": sorted_training_admission_contract_ids,
        "advantage_estimator_name": advantage_estimator_name,
        "loss_normalization": _LOSS_NORMALIZATION,
        "training_admitted": training_admission,
        "sequence_level_ratios_enabled": False,
        "sequence_level_clipping_enabled": False,
        "expected_rollouts_per_group": expected_rollouts_per_group,
        "batch_quantum": batch_quantum,
        "comparison_group_count": len(grouped),
        "logical_rollout_count": len(unique_bundles),
        "physical_trace_count": physical_trace_count,
        "padding_row_count": padding_row_count,
        "total_row_count": len(rows),
        "eligible_action_token_count": eligible_action_token_count,
        "duplicate_retry_count": duplicate_retry_count,
        "group_ids": list(grouped),
        "rollout_ids": rollout_ids,
        "parent_indices": [row["parent_rollout_index"] for row in rows],
        "rollout_to_rows": rollout_to_rows,
        "rows": rows,
    }
    plan = TraceBatchPlan(
        plan_id=_canonical_digest(plan_without_id),
        **plan_without_id,
    )
    validate_trace_batch_plan(plan, bundles=unique_bundles)
    return plan


def validate_trace_batch_plan(
    plan: Mapping[str, Any],
    *,
    bundles: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Independently validate a serialized TraceBatchPlan."""
    if plan.get("schema_version") != _TRACE_BATCH_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported TraceBatchPlan schema version")
    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("TraceBatchPlan has no plan identity")
    digest_input = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan_id != _canonical_digest(digest_input):
        raise ValueError("TraceBatchPlan identity does not match its contents")

    batch_quantum = _require_positive_int(
        plan.get("batch_quantum"),
        field="batch_quantum",
    )
    expected_rollouts_per_group = _require_positive_int(
        plan.get("expected_rollouts_per_group"),
        field="expected_rollouts_per_group",
    )
    if plan.get("advantage_estimator_name") not in _SUPPORTED_ADVANTAGE_ESTIMATORS:
        raise ValueError("TraceBatchPlan has an unsupported advantage estimator")
    if plan.get("loss_normalization") != _LOSS_NORMALIZATION:
        raise ValueError("TraceBatchPlan has an unsupported loss normalization")
    if plan.get("sequence_level_ratios_enabled") is not False:
        raise ValueError("TraceBatchPlan unexpectedly enables sequence-level ratios")
    if plan.get("sequence_level_clipping_enabled") is not False:
        raise ValueError("TraceBatchPlan unexpectedly enables sequence-level clipping")
    if not isinstance(plan.get("training_admitted"), bool):
        raise ValueError("TraceBatchPlan training_admitted must be boolean")
    if not isinstance(plan.get("optimizer_step_id"), str) or not plan.get(
        "optimizer_step_id"
    ):
        raise ValueError("TraceBatchPlan has no optimizer-step identity")
    if not isinstance(plan.get("generation_contract_id"), str) or not plan.get(
        "generation_contract_id"
    ):
        raise ValueError("TraceBatchPlan has no generation-contract identity")
    admission_contract_id = plan.get("training_admission_contract_id")
    admission_contract_ids = plan.get("training_admission_contract_ids")
    if not isinstance(admission_contract_ids, list) or any(
        not isinstance(value, str) or not value
        for value in admission_contract_ids
    ):
        raise ValueError(
            "TraceBatchPlan training admission identities must be a list of strings"
        )
    if plan["training_admitted"]:
        if not isinstance(admission_contract_id, str) or not admission_contract_id:
            raise ValueError(
                "Training-admitted TraceBatchPlan has no admission identity"
            )
        if not admission_contract_ids:
            raise ValueError(
                "Training-admitted TraceBatchPlan has no source admission identities"
            )
    elif admission_contract_id is not None or admission_contract_ids:
        raise ValueError(
            "Generation-only TraceBatchPlan unexpectedly has an admission identity"
        )
    duplicate_retry_count = plan.get("duplicate_retry_count")
    if (
        isinstance(duplicate_retry_count, bool)
        or not isinstance(duplicate_retry_count, int)
        or duplicate_retry_count < 0
    ):
        raise ValueError("TraceBatchPlan duplicate-retry count is invalid")

    rows = plan.get("rows")
    rollout_ids = plan.get("rollout_ids")
    rollout_to_rows = plan.get("rollout_to_rows")
    parent_indices = plan.get("parent_indices")
    group_ids = plan.get("group_ids")
    if not isinstance(rows, list):
        raise ValueError("TraceBatchPlan rows must be a list")
    if not isinstance(rollout_ids, list) or len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("TraceBatchPlan rollout IDs must be a unique list")
    if not isinstance(rollout_to_rows, list):
        raise ValueError("TraceBatchPlan rollout_to_rows must be a list")
    if not isinstance(parent_indices, list):
        raise ValueError("TraceBatchPlan parent_indices must be a list")
    if not isinstance(group_ids, list) or len(set(group_ids)) != len(group_ids):
        raise ValueError("TraceBatchPlan group IDs must be a unique list")

    logical_rollout_count = len(rollout_ids)
    if plan.get("logical_rollout_count") != logical_rollout_count:
        raise ValueError("TraceBatchPlan logical rollout count is corrupted")
    if len(rollout_to_rows) != logical_rollout_count:
        raise ValueError("TraceBatchPlan rollout-to-row mapping is incomplete")
    if plan.get("comparison_group_count") != len(group_ids):
        raise ValueError("TraceBatchPlan comparison group count is corrupted")
    if len(rows) != plan.get("total_row_count"):
        raise ValueError("TraceBatchPlan total row count is corrupted")
    if len(parent_indices) != len(rows):
        raise ValueError("TraceBatchPlan parent indices are incomplete")
    if len(rows) % batch_quantum:
        raise ValueError("TraceBatchPlan row count is not batch-quantum aligned")

    derived_rollout_to_rows: list[list[int]] = [
        [] for _ in range(logical_rollout_count)
    ]
    derived_parent_indices: list[int] = []
    seen_trace_ids: set[str] = set()
    seen_completion_ids: set[tuple[str, str]] = set()
    rollout_groups: list[str | None] = [None] * logical_rollout_count
    rollout_rewards: list[float | None] = [None] * logical_rollout_count
    rollout_advantages: list[float | None] = [None] * logical_rollout_count
    rollout_trace_indices: list[list[int]] = [[] for _ in range(logical_rollout_count)]
    physical_trace_count = 0
    padding_row_count = 0
    eligible_action_token_count = 0
    padding_started = False
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("row_index") != row_index:
            raise ValueError(f"TraceBatchPlan row {row_index} has an invalid index")
        row_kind = row.get("row_kind")
        parent_index = row.get("parent_rollout_index")
        derived_parent_indices.append(parent_index)
        if row_kind == "physical_trace":
            if padding_started:
                raise ValueError("Physical trace appears after padding rows")
            if (
                isinstance(parent_index, bool)
                or not isinstance(parent_index, int)
                or not 0 <= parent_index < logical_rollout_count
            ):
                raise ValueError(
                    f"Physical row {row_index} has an invalid parent rollout"
                )
            if row.get("rollout_id") != rollout_ids[parent_index]:
                raise ValueError(
                    f"Physical row {row_index} rollout identity is corrupted"
                )
            group_id = row.get("group_id")
            if not isinstance(group_id, str) or group_id not in group_ids:
                raise ValueError(
                    f"Physical row {row_index} group identity is corrupted"
                )
            if rollout_groups[parent_index] not in (None, group_id):
                raise ValueError(
                    f"Rollout {rollout_ids[parent_index]!r} changed comparison groups"
                )
            rollout_groups[parent_index] = group_id
            trace_id = row.get("trace_id")
            if not isinstance(trace_id, str) or not trace_id:
                raise ValueError(f"Physical row {row_index} has no trace identity")
            if trace_id in seen_trace_ids:
                raise ValueError(f"Duplicate physical trace ID {trace_id!r}")
            seen_trace_ids.add(trace_id)
            token_count = _require_positive_int(
                row.get("token_count"),
                field=f"rows[{row_index}].token_count",
            )
            eligible_token_count = row.get("eligible_token_count")
            if (
                isinstance(eligible_token_count, bool)
                or not isinstance(eligible_token_count, int)
                or eligible_token_count < 0
                or eligible_token_count > token_count
            ):
                raise ValueError(
                    f"Physical row {row_index} has an invalid eligible-token count"
                )
            if row.get("sample_mask") != 1.0:
                raise ValueError(f"Physical row {row_index} is sample-masked")
            reward = _finite_float(
                row.get("reward"),
                field=f"rows[{row_index}].reward",
            )
            advantage = _finite_float(
                row.get("advantage"),
                field=f"rows[{row_index}].advantage",
            )
            if rollout_rewards[parent_index] not in (None, reward):
                raise ValueError(
                    f"Rollout {rollout_ids[parent_index]!r} has inconsistent rewards"
                )
            if rollout_advantages[parent_index] not in (None, advantage):
                raise ValueError(
                    f"Rollout {rollout_ids[parent_index]!r} has inconsistent advantages"
                )
            rollout_rewards[parent_index] = reward
            rollout_advantages[parent_index] = advantage
            if (
                not isinstance(row.get("completion_ids"), list)
                or not row["completion_ids"]
            ):
                raise ValueError(f"Physical row {row_index} has no completions")
            completion_keys = {
                (row["rollout_id"], completion_id)
                for completion_id in row["completion_ids"]
            }
            duplicate_completion_ids = seen_completion_ids.intersection(completion_keys)
            if duplicate_completion_ids:
                raise ValueError(
                    "Completion IDs appear in multiple physical rows: "
                    f"{sorted(duplicate_completion_ids)!r}"
                )
            seen_completion_ids.update(completion_keys)
            if not isinstance(row.get("ordered_media_ids"), list):
                raise ValueError(
                    f"Physical row {row_index} has invalid ordered media IDs"
                )
            derived_rollout_to_rows[parent_index].append(row_index)
            rollout_trace_indices[parent_index].append(row.get("trace_index"))
            eligible_action_token_count += eligible_token_count
            physical_trace_count += 1
        elif row_kind == "padding":
            padding_started = True
            if (
                parent_index != -1
                or row.get("rollout_id") is not None
                or row.get("source_row_index") is not None
                or row.get("group_id") is not None
                or row.get("trace_id") is not None
                or row.get("trace_index") != -1
                or row.get("reward") != 0.0
                or row.get("advantage") != 0.0
                or row.get("sample_mask") != 0.0
                or row.get("token_count") != 0
                or row.get("eligible_token_count") != 0
                or row.get("completion_ids") != []
                or row.get("ordered_media_ids") != []
            ):
                raise ValueError(f"Padding row {row_index} is not fully masked")
            padding_row_count += 1
        else:
            raise ValueError(f"TraceBatchPlan row {row_index} has unknown kind")

    if derived_parent_indices != parent_indices:
        raise ValueError("TraceBatchPlan parent indices are corrupted")
    if derived_rollout_to_rows != rollout_to_rows:
        raise ValueError("TraceBatchPlan rollout-to-row mapping is corrupted")
    if any(not row_indices for row_indices in rollout_to_rows):
        raise ValueError("A logical rollout owns no physical trace rows")
    if any(group_id is None for group_id in rollout_groups):
        raise ValueError("TraceBatchPlan has a rollout with no comparison group")
    if any(
        trace_indices != list(range(len(trace_indices)))
        for trace_indices in rollout_trace_indices
    ):
        raise ValueError(
            "TraceBatchPlan trace indices are not rollout-local and consecutive"
        )
    group_cardinalities = {
        group_id: sum(observed_group == group_id for observed_group in rollout_groups)
        for group_id in group_ids
    }
    if any(
        cardinality != expected_rollouts_per_group
        for cardinality in group_cardinalities.values()
    ):
        raise ValueError("TraceBatchPlan comparison group is incomplete")
    if plan.get("physical_trace_count") != physical_trace_count:
        raise ValueError("TraceBatchPlan physical trace count is corrupted")
    if plan.get("padding_row_count") != padding_row_count:
        raise ValueError("TraceBatchPlan padding row count is corrupted")
    if padding_row_count != (-physical_trace_count) % batch_quantum:
        raise ValueError("TraceBatchPlan does not use minimal batch-quantum padding")
    if plan.get("eligible_action_token_count") != eligible_action_token_count:
        raise ValueError("TraceBatchPlan eligible-token count is corrupted")
    if eligible_action_token_count <= 0:
        raise ValueError("TraceBatchPlan has no eligible action tokens")

    if bundles is not None:
        if len(bundles) != logical_rollout_count:
            raise ValueError("TraceBatchPlan bundle count is corrupted")
        expected_rows: list[tuple[Any, ...]] = []
        observed_contract_ids: set[str] = set()
        observed_admission_ids: set[str] = set()
        for parent_rollout_index, bundle in enumerate(bundles):
            validate_rollout_trace_bundle(bundle, strict=True)
            rollout_id = str(bundle["rollout_id"])
            if rollout_id != rollout_ids[parent_rollout_index]:
                raise ValueError("TraceBatchPlan bundle ordering is corrupted")
            contract = bundle["generation_contract"]
            observed_contract_ids.add(str(contract["generation_contract_id"]))
            if plan["training_admitted"]:
                admission = bundle.get("training_admission")
                if not isinstance(admission, Mapping):
                    raise ValueError(
                        "TraceBatchPlan admits a rollout without NeMo-RL training "
                        "admission"
                    )
                admission_id = admission.get("admission_contract_id")
                if not isinstance(admission_id, str) or not admission_id:
                    raise ValueError(
                        "TraceBatchPlan rollout has no training admission identity"
                    )
                observed_admission_ids.add(admission_id)
            reward = _finite_float(
                bundle.get("reward"),
                field=f"reward[{rollout_id!r}]",
            )
            for trace in bundle["physical_traces"]:
                expected_rows.append(
                    (
                        parent_rollout_index,
                        rollout_id,
                        str(bundle["group_id"]),
                        str(trace["trace_id"]),
                        int(trace["trace_index"]),
                        reward,
                        int(trace["token_count"]),
                        int(trace["trainable_token_count"]),
                        [
                            str(span["completion_id"])
                            for span in trace["completion_spans"]
                        ],
                        [str(value) for value in trace["ordered_media_ids"]],
                    )
                )
        observed_rows = [
            (
                row["parent_rollout_index"],
                row["rollout_id"],
                row["group_id"],
                row["trace_id"],
                row["trace_index"],
                row["reward"],
                row["token_count"],
                row["eligible_token_count"],
                row["completion_ids"],
                row["ordered_media_ids"],
            )
            for row in rows[:physical_trace_count]
        ]
        if observed_rows != expected_rows:
            raise ValueError(
                "TraceBatchPlan physical rows disagree with rollout bundles"
            )
        if observed_contract_ids != {plan.get("generation_contract_id")}:
            raise ValueError(
                "TraceBatchPlan generation contract disagrees with its bundles"
            )
        if plan["training_admitted"] and observed_admission_ids != set(
            plan.get("training_admission_contract_ids", [])
        ):
            raise ValueError(
                "TraceBatchPlan training admission disagrees with its bundles"
            )
        observed_groups = list(
            dict.fromkeys(str(bundle["group_id"]) for bundle in bundles)
        )
        if observed_groups != group_ids:
            raise ValueError("TraceBatchPlan group ordering is corrupted")
        if any(
            sum(1 for bundle in bundles if bundle["group_id"] == group_id)
            != expected_rollouts_per_group
            for group_id in group_ids
        ):
            raise ValueError("TraceBatchPlan comparison group is incomplete")

    return {
        "logical_rollout_count": logical_rollout_count,
        "physical_trace_count": physical_trace_count,
        "padding_row_count": padding_row_count,
        "eligible_action_token_count": eligible_action_token_count,
        "duplicate_retry_count": int(plan.get("duplicate_retry_count", 0)),
    }
