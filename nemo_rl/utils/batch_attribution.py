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

"""Payload-free, opt-in attribution of SingleController training batches.

Times are wall-clock Unix seconds so events can be joined to Gym/checkpoint
logs. Unknown historical times stay null: restoring a ready group must not
invent a new completion time. Selection records describe a claim, not a
successful optimizer update; ``step_committed`` is emitted separately.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

TASK_TAG = "batch_attribution_task"
TASK_FIELD = "batch_attribution_task_names"
READY_TAG = "batch_attribution_ready_unix_s"
DISPATCH_TAG = "batch_attribution_dispatch_unix_s"
METRIC_PREFIX = "batch_attribution/tasks/"


def attach_task_labels(
    data: MutableMapping[str, Any], tags: list[dict[str, Any]] | None
) -> None:
    """Attach task strings locally, after any tensor-only replica broadcast."""
    if tags and any(tag.get("batch_attribution_enabled") for tag in tags):
        if not all(tag.get("batch_attribution_enabled") for tag in tags):
            raise ValueError("mixed enabled/disabled batch attribution rows")
        data[TASK_FIELD] = [tag[TASK_TAG] for tag in tags]


def task_metric_prefix(task_name: str) -> str:
    """Escape user-supplied task names without merging distinct names."""
    return METRIC_PREFIX + quote(task_name, safe="-_.") + "/"


@dataclass(frozen=True)
class GroupTrainingStats:
    """One selected group's final-mask statistics, never token payloads."""

    group_id: str
    task_name: str
    selected_samples: int
    valid_sequence_weight: float
    valid_tokens: float
    reward_sum: float
    reward_weight: float
    response_lengths: tuple[int, ...]


def reduce_group_stats(groups: list[GroupTrainingStats]) -> dict[str, float]:
    """Reduce across all chunks/shards using sums, not means of means."""
    by_task: dict[str, list[GroupTrainingStats]] = defaultdict(list)
    for group in groups:
        by_task[group.task_name].append(group)
    total_tokens = sum(group.valid_tokens for group in groups)
    total_weight = sum(group.reward_weight for group in groups)
    metrics: dict[str, float] = {}
    for task_name, task_groups in by_task.items():
        prefix = task_metric_prefix(task_name)
        lengths = sorted(n for group in task_groups for n in group.response_lengths)
        tokens = sum(group.valid_tokens for group in task_groups)
        reward_weight = sum(group.reward_weight for group in task_groups)
        metrics.update(
            {
                prefix + "selected_groups": float(len(task_groups)),
                prefix + "selected_samples": float(
                    sum(g.selected_samples for g in task_groups)
                ),
                prefix + "global_valid_seqs": sum(
                    g.valid_sequence_weight for g in task_groups
                ),
                prefix + "global_valid_toks": tokens,
                prefix + "token_fraction": tokens / total_tokens
                if total_tokens
                else 0.0,
                prefix + "reward_weight": reward_weight,
                prefix + "sample_fraction": reward_weight / total_weight
                if total_weight
                else 0.0,
                prefix + "reward_mean": sum(g.reward_sum for g in task_groups)
                / reward_weight
                if reward_weight
                else 0.0,
                prefix + "response_length_mean": sum(lengths) / len(lengths)
                if lengths
                else 0.0,
                prefix + "response_length_p95": float(
                    lengths[max(0, math.ceil(0.95 * len(lengths)) - 1)]
                )
                if lengths
                else 0.0,
            }
        )
    return metrics


def finish_entropy_metrics(metrics: dict[str, Any]) -> None:
    """Derive per-task entropy from globally reduced loss contributions."""
    for key in tuple(metrics):
        if key.startswith(METRIC_PREFIX) and key.endswith("/entropy_contribution"):
            prefix = key.removesuffix("entropy_contribution")
            fraction = metrics[prefix + "entropy_token_fraction"]
            metrics[prefix + "approx_entropy"] = (
                metrics[key] / fraction if fraction else 0.0
            )


def reconciliation_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Expose attribution coverage/errors without changing training decisions."""
    result = {}
    for suffix, global_name in (
        ("global_valid_toks", "global_valid_toks"),
        ("global_valid_seqs", "global_valid_seqs"),
        ("entropy_contribution", "approx_entropy"),
    ):
        values = [
            float(v)
            for k, v in metrics.items()
            if k.startswith(METRIC_PREFIX) and k.endswith("/" + suffix)
        ]
        if values and global_name in metrics:
            result[f"batch_attribution/reconcile/{global_name}_delta"] = sum(
                values
            ) - float(metrics[global_name])
    weight = sum(
        float(v)
        for k, v in metrics.items()
        if k.startswith(METRIC_PREFIX) and k.endswith("/reward_weight")
    )
    if weight and "reward" in metrics:
        reward = (
            sum(
                float(v)
                * float(metrics[k.removesuffix("reward_mean") + "reward_weight"])
                for k, v in metrics.items()
                if k.startswith(METRIC_PREFIX) and k.endswith("/reward_mean")
            )
            / weight
        )
        result["batch_attribution/reconcile/reward_delta"] = reward - float(
            metrics["reward"]
        )
    return result


@dataclass
class GroupLifecycle:
    """Current-attempt provenance; historical dispatch/ready times are optional."""

    task_name: str
    origin: str
    dispatch_unix_s: float | None
    ready_unix_s: float | None
    restore_details: dict[str, Any]


class BatchAttribution:
    """Controller-local trace, flushed in batches off the event loop.

    No diagnostic state participates in admission or checkpoint ownership.
    Existing metadata timestamps are persisted with completed replay groups;
    unfinished groups are identified from the recovery ledger on each restore.
    """

    def __init__(self, log_dir: str) -> None:
        self.attempt_id = uuid.uuid4().hex
        self.attempt_started_unix_s = time.time()
        self.path = (
            Path(log_dir) / "batch_attribution" / f"groups-{self.attempt_id}.jsonl"
        )
        self.groups: dict[str, GroupLifecycle] = {}
        self.step_groups: list[GroupTrainingStats] = []
        self.events: list[dict[str, Any]] = []
        self._sequence = 0
        self._write_lock = threading.Lock()
        self.record("attempt_started", pid=os.getpid())

    def record(self, event: str, **details: Any) -> None:
        """Queue a metadata-only event; callers supply no prompts or outputs."""
        self.events.append(
            dict(
                schema_version=1,
                attempt_id=self.attempt_id,
                sequence=self._sequence,
                event=event,
                unix_s=time.time(),
                **details,
            )
        )
        self._sequence += 1

    def restore_group(
        self,
        group_id: str,
        *,
        origin: str,
        task_name: str,
        tags: dict[str, Any] | None = None,
        **details: Any,
    ) -> None:
        """Classify restored ownership without treating it as fresh generation."""
        tags = tags or {}
        self.groups[group_id] = GroupLifecycle(
            task_name=task_name,
            origin=origin,
            dispatch_unix_s=tags.get(DISPATCH_TAG),
            ready_unix_s=tags.get(READY_TAG),
            restore_details=details,
        )
        self.record(
            "group_restored",
            group_id=group_id,
            task_name=task_name,
            origin=origin,
            historical_ready_unix_s=tags.get(READY_TAG),
            **details,
        )

    def dispatch(
        self,
        group_id: str,
        *,
        task_name: str,
        trainer_version: int,
        target_step: int | None,
        prompt_idx: int | None = None,
    ) -> None:
        """Record the start of current-attempt work, preserving restore origin."""
        group = self.groups.get(group_id)
        if group is None:
            group = GroupLifecycle(task_name, "fresh", None, None, {})
            self.groups[group_id] = group
        group.task_name = task_name
        group.dispatch_unix_s = time.time()
        self.record(
            "group_dispatch",
            group_id=group_id,
            task_name=task_name,
            origin=group.origin,
            trainer_version=trainer_version,
            target_step=target_step,
            prompt_idx=prompt_idx,
        )

    def ready(
        self, group_id: str, *, task_name: str, prompt_idx: int | None = None
    ) -> dict[str, Any]:
        """Stamp replay metadata only after canonical publication succeeds."""
        group = self.groups.get(group_id)
        if group is None:
            group = GroupLifecycle(task_name, "untracked", None, None, {})
            self.groups[group_id] = group
        group.task_name = task_name
        group.ready_unix_s = time.time()
        self.record(
            "group_ready",
            group_id=group_id,
            task_name=task_name,
            origin=group.origin,
            dispatch_unix_s=group.dispatch_unix_s,
            prompt_idx=prompt_idx,
        )
        return {
            TASK_TAG: task_name,
            READY_TAG: group.ready_unix_s,
            DISPATCH_TAG: group.dispatch_unix_s,
        }

    def selected(
        self,
        *,
        group_ids: list[str],
        step: int,
        trainer_version: int,
        chunk: int,
        sampler: str,
        inventory: list[dict[str, Any]],
        min_groups: int,
        max_groups: int,
        sample_ids_by_group: dict[str, list[str]] | None = None,
    ) -> None:
        """Describe selection and the competing ready/unready buffer population."""
        now = time.time()
        self.record(
            "selection_inventory",
            train_step=step,
            chunk=chunk,
            trainer_version=trainer_version,
            sampler=sampler,
            min_groups=min_groups,
            max_groups=max_groups,
            selected_group_ids=group_ids,
            inventory=inventory,
        )
        positions = {row["group_id"]: row for row in inventory}
        for group_id in group_ids:
            group = self.groups.get(group_id)
            ready = group.ready_unix_s if group else None
            self.record(
                "group_selected",
                group_id=group_id,
                train_step=step,
                chunk=chunk,
                trainer_version=trainer_version,
                sampler=sampler,
                origin=group.origin if group else "unknown_restored",
                task_name=group.task_name if group else "unknown_restored",
                sample_ids=(sample_ids_by_group or {}).get(group_id),
                dispatch_unix_s=group.dispatch_unix_s if group else None,
                ready_unix_s=ready,
                ready_wait_current_attempt_s=(
                    now - max(ready, self.attempt_started_unix_s)
                )
                if ready is not None
                else None,
                selection=positions.get(group_id),
            )

    def add_training_stats(self, groups: tuple[GroupTrainingStats, ...]) -> None:
        """Retain only scalar/length summaries until this optimizer step closes."""
        self.step_groups.extend(groups)

    def finish_step(self, step: int) -> dict[str, float]:
        """Publish actual consumed groups separately from pre-training claims."""
        metrics = reduce_group_stats(self.step_groups)
        by_origin: dict[str, float] = defaultdict(float)
        groups_by_origin: dict[str, int] = defaultdict(int)
        for group in self.step_groups:
            lifecycle = self.groups.pop(group.group_id, None)
            origin = lifecycle.origin if lifecycle else "unknown_restored"
            by_origin[origin] += group.valid_tokens
            groups_by_origin[origin] += 1
            details = asdict(group)
            # Lengths are used for exact per-task quantiles, not repeated in JSONL.
            details.pop("response_lengths")
            self.record("group_consumed", train_step=step, origin=origin, **details)
        metrics.update(
            {
                f"batch_attribution/origins/{origin}/valid_tokens": value
                for origin, value in by_origin.items()
            }
        )
        metrics.update(
            {
                f"batch_attribution/origins/{origin}/groups": float(value)
                for origin, value in groups_by_origin.items()
            }
        )
        self.record(
            "step_committed",
            train_step=step,
            groups=len(self.step_groups),
            valid_tokens=sum(g.valid_tokens for g in self.step_groups),
        )
        self.step_groups.clear()
        return metrics

    def drain(self) -> list[dict[str, Any]]:
        """Swap buffers on the controller thread before writing off-thread."""
        events, self.events = self.events, []
        return events

    def write(self, events: list[dict[str, Any]]) -> None:
        """Append one event batch, never one filesystem write per token/group."""
        if not events:
            return
        encoded = "".join(
            json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n"
            for event in events
        )
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
