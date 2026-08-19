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

import asyncio
import hashlib
import json
import math
import statistics
import threading as _threading
import time
import uuid
from collections import Counter, deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Literal, Optional, TypedDict

import ray

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG, ROUTED_EXPERTS_FIELD
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.payload import pack_payload, record_to_train_batch
from nemo_rl.experience.row_dump import maybe_dump_train_rows
from nemo_rl.utils.r3_trace import trace_rollout_payload

DATA_PLANE_CHECKPOINT_DIR = "data_plane"
REPLAY_BUFFER_METADATA_FILENAME = "replay_buffer_metadata.pt"
LEGACY_REPLAY_BUFFER_FILENAME = "replay_buffer.pt"
REPLAY_BUFFER_METADATA_SCHEMA_VERSION = 1
REPLAY_BUFFER_METADATA_STORAGE: Literal["tq_checkpoint"] = "tq_checkpoint"

CheckpointMutationKind = Literal[
    "group_commits",
    "group_removals",
    "other",
    "prompt_reservations",
    "recovery_retries",
    "sample_clears",
    "sibling_seals",
]
CHECKPOINT_MUTATION_KINDS: tuple[CheckpointMutationKind, ...] = (
    "group_commits",
    "group_removals",
    "prompt_reservations",
    "recovery_retries",
    "sample_clears",
    "sibling_seals",
    "other",
)


@dataclass(frozen=True)
class DataPlaneCheckpointBarrierTelemetry:
    """Bounded interval telemetry for checkpoint-induced mutation waits."""

    blocked_by_kind: dict[CheckpointMutationKind, int]
    wait_durations_s: tuple[float, ...]
    active_mutations: int
    waiting_mutations: int
    max_waiting_mutations: int
    checkpoint_active: bool


# These TypedDicts describe the versioned, plain-mapping checkpoint wire
# format. They are intentionally not dataclass instances: persisting a
# dataclass would couple recovery to its Python import path and class layout.
# Runtime objects such as KVBatchMeta remain explicitly represented as fields
# inside this schema.


class TQReplayGroupMetadata(TypedDict):
    """Controller-local index for one training-ready group stored in TQ."""

    meta: KVBatchMeta
    start_weight: int
    end_weight: int
    target_step: Optional[int]
    group_id: str


class TQReplayMetadataState(TypedDict):
    """Versioned metadata-only replay sidecar paired with a TQ snapshot."""

    schema_version: int
    storage: Literal["tq_checkpoint"]
    partition_id: str
    saved_capacity: int
    manifest_digest: str
    groups: list[TQReplayGroupMetadata]


def _canonical_manifest_value(value: Any, *, path: str) -> Any:
    """Return a deterministic JSON value or reject unsupported metadata."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        float_value = float(value)
        if not math.isfinite(float_value):
            raise TypeError(f"Replay metadata at {path} must be finite")
        return float_value
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Replay metadata at {path} has non-string key {key!r}")
            canonical[key] = _canonical_manifest_value(
                item,
                path=f"{path}.{key}",
            )
        return canonical
    if isinstance(value, (list, tuple)):
        return [
            _canonical_manifest_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Replay metadata at {path} has unsupported type "
        f"{type(value).__name__}; expected JSON-compatible primitive values"
    )


def replay_manifest_digest(groups: list[TQReplayGroupMetadata]) -> str:
    """Return a stable digest binding replay metadata to a TQ checkpoint."""
    digest_input = [
        {
            "group_id": group["group_id"],
            "start_weight": group["start_weight"],
            "end_weight": group["end_weight"],
            "target_step": group["target_step"],
            "meta": {
                "partition_id": group["meta"].partition_id,
                "task_name": group["meta"].task_name,
                "sample_ids": list(group["meta"].sample_ids),
                "fields": (
                    list(group["meta"].fields)
                    if group["meta"].fields is not None
                    else None
                ),
                "sequence_lengths": (
                    list(group["meta"].sequence_lengths)
                    if group["meta"].sequence_lengths is not None
                    else None
                ),
                "tags": _canonical_manifest_value(
                    group["meta"].tags,
                    path=f"groups[{group_index}].meta.tags",
                ),
                "extra_info": _canonical_manifest_value(
                    group["meta"].extra_info,
                    path=f"groups[{group_index}].meta.extra_info",
                ),
            },
        }
        for group_index, group in enumerate(groups)
    ]
    encoded = json.dumps(
        digest_input,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DataPlaneCheckpointBarrier:
    """Allow concurrent mutations while giving live checkpoints exclusivity.

    At most one checkpoint holder is active. New mutations queue behind it,
    and a checkpoint waits for all active mutations before yielding. Every
    live canonical TQ commit/clear and native save must use this barrier so the
    snapshot and controller replay index describe the same rows.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._checkpoint_active = False
        self._active_mutations = 0
        self._mutation_depth_by_task: dict[asyncio.Task[Any], int] = {}
        self._mutation_version = 0
        self._waiting_mutations = 0
        self._max_waiting_mutations = 0
        self._blocked_by_kind: Counter[CheckpointMutationKind] = Counter()
        self._wait_durations_s: deque[float] = deque(maxlen=10_000)

    @property
    def mutation_version(self) -> int:
        """Return a monotonic marker for completed outer mutation sections."""
        return self._mutation_version

    @asynccontextmanager
    async def mutation(
        self, kind: CheckpointMutationKind = "other"
    ) -> AsyncIterator[None]:
        """Enter a commit/clear section, waiting only for an active checkpoint."""
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("data-plane mutation must run inside an asyncio task")
        depth = self._mutation_depth_by_task.get(task, 0)
        if depth:
            # Replay-buffer helpers may join a controller-owned mutation. Do not
            # wait behind a checkpoint that is already waiting for this outer
            # mutation, or the two tasks deadlock.
            self._mutation_depth_by_task[task] = depth + 1
            try:
                yield
            finally:
                self._mutation_depth_by_task[task] -= 1
            return
        async with self._condition:
            wait_started: Optional[float] = None
            if self._checkpoint_active:
                wait_started = time.monotonic()
                self._waiting_mutations += 1
                self._max_waiting_mutations = max(
                    self._max_waiting_mutations, self._waiting_mutations
                )
            try:
                await self._condition.wait_for(lambda: not self._checkpoint_active)
            finally:
                if wait_started is not None:
                    self._waiting_mutations -= 1
                    self._blocked_by_kind[kind] += 1
                    self._wait_durations_s.append(time.monotonic() - wait_started)
            self._active_mutations += 1
            self._mutation_depth_by_task[task] = 1
        try:
            yield
        finally:
            del self._mutation_depth_by_task[task]
            async with self._condition:
                self._active_mutations -= 1
                # Count completed mutation sections even when their body raised.
                # A redundant periodic snapshot is safe; missing a mutation that
                # partially changed TQ or controller metadata is not.
                self._mutation_version += 1
                if self._active_mutations == 0:
                    self._condition.notify_all()

    async def drain_telemetry(self) -> DataPlaneCheckpointBarrierTelemetry:
        """Return and reset interval waits while preserving current state."""
        async with self._condition:
            telemetry = DataPlaneCheckpointBarrierTelemetry(
                blocked_by_kind={
                    kind: self._blocked_by_kind.get(kind, 0)
                    for kind in CHECKPOINT_MUTATION_KINDS
                },
                wait_durations_s=tuple(self._wait_durations_s),
                active_mutations=self._active_mutations,
                waiting_mutations=self._waiting_mutations,
                max_waiting_mutations=self._max_waiting_mutations,
                checkpoint_active=self._checkpoint_active,
            )
            self._blocked_by_kind.clear()
            self._wait_durations_s.clear()
            self._max_waiting_mutations = self._waiting_mutations
            return telemetry

    @asynccontextmanager
    async def checkpoint(self) -> AsyncIterator[None]:
        """Block new mutations and wait for active ones before snapshotting."""
        async with self._condition:
            await self._condition.wait_for(lambda: not self._checkpoint_active)
            self._checkpoint_active = True
            try:
                await self._condition.wait_for(lambda: self._active_mutations == 0)
            except BaseException:
                self._checkpoint_active = False
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            async with self._condition:
                self._checkpoint_active = False
                self._condition.notify_all()


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class ReplayBufferImpl(ReplayBufferProtocol):
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.trajectories = []  # List[dict[str, Any]]
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

    @staticmethod
    def _rollout_metrics_turn_count_for_diagnostics(
        rm: dict[str, Any],
    ) -> Optional[float]:
        """One scalar turn-depth per buffered trajectory for starvation diagnostics.

        Supports sync multi-turn rollouts (`max_turns_per_sample` / `avg_turns_per_sample`)
        and NeMo Gym (`turns_per_sample/max` / `turns_per_sample/mean`).
        """
        if "max_turns_per_sample" in rm:
            return float(rm["max_turns_per_sample"])
        if "avg_turns_per_sample" in rm:
            return float(rm["avg_turns_per_sample"])
        if "turns_per_sample/max" in rm:
            return float(rm["turns_per_sample/max"])
        if "turns_per_sample/mean" in rm:
            return float(rm["turns_per_sample/mean"])
        return None

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("🔍 ReplayBuffer.add: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            # Do not advance last_target_weight_already_generated here. A target
            # is only safe to skip once training consumes a complete batch for it.
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}, targets={self.target_weight_versions}, last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        info: dict[str, Any] = {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
        }
        if self.trajectories:
            durations = []
            max_gen_tokens_per_turn_list = []
            turn_counts_list = []
            for t in self.trajectories:
                rm = t.get("rollout_metrics", {})
                if "trajectory_duration_s" in rm:
                    durations.append(rm["trajectory_duration_s"])
                if "max_gen_tokens_per_turn/max" in rm:
                    max_gen_tokens_per_turn_list.append(
                        rm["max_gen_tokens_per_turn/max"]
                    )
                elif "max_gen_tokens_per_turn" in rm:
                    max_gen_tokens_per_turn_list.append(rm["max_gen_tokens_per_turn"])
                tc = self._rollout_metrics_turn_count_for_diagnostics(rm)
                if tc is not None:
                    turn_counts_list.append(tc)

            def _pct(values: list[float], p: float) -> float:
                if not values:
                    return 0.0
                sorted_v = sorted(values)
                idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
                return float(sorted_v[idx])

            info["starvation_diagnostics"] = {
                "trajectory_duration_s": {
                    "mean": sum(durations) / len(durations) if durations else 0,
                    "median": statistics.median(durations) if durations else 0,
                    "max": max(durations) if durations else 0,
                    "p95": _pct(durations, 95),
                },
                "max_gen_tokens_per_turn_in_buffer": {
                    "mean": sum(max_gen_tokens_per_turn_list)
                    / len(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "median": statistics.median(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "max": max(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "p95": _pct(max_gen_tokens_per_turn_list, 95),
                },
                "turns_per_sample_in_buffer": {
                    "mean": sum(turn_counts_list) / len(turn_counts_list)
                    if turn_counts_list
                    else 0,
                    "median": statistics.median(turn_counts_list)
                    if turn_counts_list
                    else 0,
                    "max": max(turn_counts_list) if turn_counts_list else 0,
                    "p95": _pct(turn_counts_list, 95),
                },
                "num_trajectories_sampled": len(self.trajectories),
            }
        return info

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        with self._lock:
            return set(self.target_weight_versions)

    def _remove_indices(self, indices: Iterable[int]) -> None:
        """Remove trajectories at the given indices."""
        for idx in sorted(indices, reverse=True):
            self.trajectory_versions.pop(idx)
            self.target_weight_versions.pop(idx)
            self.trajectories.pop(idx)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            # For debugging: check for unexpected old trajectories
            version_counts = Counter(self.trajectory_versions)
            print(f"   {version_counts=}")

            # Compute minimum valid version based on age window
            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Evict old trajectories that are beyond the age window. This can
            # happen after checkpoint restore when old trajectories remain.
            old_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if v < min_valid_version
            ]
            if old_indices:
                print(
                    f"   Evicting {len(old_indices)} stale trajectories "
                    f"(version < {min_valid_version})"
                )
                self._remove_indices(old_indices)
                total_trajectories = len(self.trajectories)

            # Filter for valid trajectories without modifying the buffer
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
            )
            if not valid_indices:
                print("No trajectories available for sampling.")
                return None

            # Enforce exact number of groups if available; otherwise, signal to wait
            if len(valid_indices) < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step
            # This ensures no trajectory loses its "last chance" to be used for its intended step
            intended_indices = [
                i
                for i in valid_indices
                if self.target_weight_versions[i] == current_weight_version
            ]

            print(
                f"   🎯 Found {len(intended_indices)} trajectories intended for current step {current_weight_version}"
            )

            # Stall training if we don't have enough trajectories intended for this step
            if len(intended_indices) < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} trajectories to be generated"
                )
                return None

            # Select exactly the trajectories intended for this step (FIFO within same target)
            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
            )

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            print(
                f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            print(
                f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
            )

            # Remove selected items in reverse order to maintain correct indices
            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)

            old_last_target = self.last_target_weight_already_generated
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated,
                current_weight_version,
            )
            if self.last_target_weight_already_generated > old_last_target:
                print(
                    "Advanced last_target_weight_already_generated: "
                    f"{old_last_target} -> "
                    f"{self.last_target_weight_already_generated} "
                    f"(consumed batch for step {current_weight_version})"
                )

            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        with self._lock:
            return {
                "trajectories": list(self.trajectories),
                "trajectory_versions": list(self.trajectory_versions),
                "target_weight_versions": list(self.target_weight_versions),
                "last_target_weight_already_generated": (
                    self.last_target_weight_already_generated
                ),
                "max_size": self.max_size,
            }

    def load_state_dict(
        self,
        state: dict[str, Any],
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
        max_age_steps: int | None = None,
    ) -> None:
        """Restore replay buffer state from a checkpoint.

        Args:
            state: State returned by ``state_dict``.
            num_prompts_per_step: Number of prompt groups required for one
                training step. When provided, incomplete target steps can be
                removed or prepared for gap filling.
            current_training_step: Step being resumed. When provided with
                ``num_prompts_per_step``, past target steps are dropped and
                incomplete current/future target steps are kept for gap filling.
            max_age_steps: Maximum allowed age for restored trajectories. When
                provided, stale trajectories are removed during restore.

        Raises:
            ValueError: If the checkpoint is missing required fields or has
                inconsistent parallel list lengths.
        """
        with self._lock:
            required_keys = {
                "trajectories",
                "trajectory_versions",
                "target_weight_versions",
                "last_target_weight_already_generated",
            }
            missing_keys = required_keys - set(state)
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            trajectories = list(state["trajectories"])
            trajectory_versions = list(state["trajectory_versions"])
            target_weight_versions = list(state["target_weight_versions"])
            if not (
                len(trajectories)
                == len(trajectory_versions)
                == len(target_weight_versions)
            ):
                raise ValueError(
                    "Checkpoint has inconsistent replay buffer lengths: "
                    f"trajectories={len(trajectories)}, "
                    f"trajectory_versions={len(trajectory_versions)}, "
                    f"target_weight_versions={len(target_weight_versions)}"
                )

            if "max_size" in state and state["max_size"] != self.max_size:
                print(
                    "ReplayBuffer max_size changed: "
                    f"checkpoint={state['max_size']}, current={self.max_size}. "
                    "Using current config value."
                )

            self.trajectories = trajectories
            self.trajectory_versions = trajectory_versions
            self.target_weight_versions = target_weight_versions
            self.last_target_weight_already_generated = state[
                "last_target_weight_already_generated"
            ]

            if current_training_step is not None and num_prompts_per_step is not None:
                self._prepare_for_training_step(
                    current_step=current_training_step,
                    num_prompts_per_step=num_prompts_per_step,
                )
            elif num_prompts_per_step is not None and self.trajectories:
                self._remove_incomplete_target_steps(num_prompts_per_step)

            if max_age_steps is not None and self.trajectories:
                self._remove_stale_trajectories(max_age_steps)
                if current_training_step is None and num_prompts_per_step is not None:
                    self._remove_incomplete_target_steps(num_prompts_per_step)

            self._truncate_to_max_size(current_training_step)

            print(
                f"ReplayBuffer restored: {len(self.trajectories)} trajectories, "
                "last_target_weight_already_generated="
                f"{self.last_target_weight_already_generated}"
            )

    def _prepare_for_training_step(
        self, current_step: int, num_prompts_per_step: int
    ) -> None:
        """Prepare restored state so training can resume at ``current_step``."""
        print(f"   Preparing replay buffer for training step {current_step}...")

        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target >= current_step
        ]

        if len(indices_to_keep) < original_count:
            removed_past = original_count - len(indices_to_keep)
            self.trajectories = [self.trajectories[i] for i in indices_to_keep]
            self.trajectory_versions = [
                self.trajectory_versions[i] for i in indices_to_keep
            ]
            self.target_weight_versions = [
                self.target_weight_versions[i] for i in indices_to_keep
            ]
            print(
                f"   Removed {removed_past} trajectories for past steps "
                f"(target < {current_step})"
            )

        if not self.trajectories:
            self.last_target_weight_already_generated = current_step - 1
            print(
                "   No restored trajectories remain; collector will generate "
                f"from step {current_step}"
            )
            return

        target_counts = Counter(self.target_weight_versions)
        complete_targets = {
            target
            for target, count in target_counts.items()
            if count >= num_prompts_per_step
        }
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }

        print(
            "   Complete targets: "
            f"{sorted(complete_targets) if complete_targets else 'none'}"
        )
        for target in sorted(incomplete_targets):
            print(
                f"   Incomplete target {target}: "
                f"{target_counts[target]}/{num_prompts_per_step}"
            )

        # Let the collector ask each target from current_step onward how many
        # trajectories are still needed, so incomplete restored batches can be
        # gap-filled and complete batches can be skipped.
        self.last_target_weight_already_generated = current_step - 1

    @staticmethod
    def _is_valid_for_target(
        trajectory_version: int, target_step: int, max_age_steps: int | None
    ) -> bool:
        if max_age_steps is None:
            return True
        min_valid_version = max(0, target_step - max_age_steps)
        return min_valid_version <= trajectory_version <= target_step

    def _remove_stale_trajectories(self, max_age_steps: int) -> None:
        """Remove restored trajectories that are stale for their target step.

        Must be called while holding ``self._lock``.
        """
        indices_to_remove = [
            i
            for i, (trajectory_version, target) in enumerate(
                zip(self.trajectory_versions, self.target_weight_versions)
            )
            if not self._is_valid_for_target(trajectory_version, target, max_age_steps)
        ]
        if not indices_to_remove:
            return

        print(
            f"   Removing {len(indices_to_remove)} stale restored trajectories "
            f"(max_age_steps={max_age_steps})"
        )
        self._remove_indices(indices_to_remove)

    def _count_for_target(
        self, target_step: int, max_age_steps: int | None = None
    ) -> int:
        """Count trajectories usable for ``target_step``.

        Must be called while holding ``self._lock``.
        """
        return sum(
            1
            for trajectory_version, target in zip(
                self.trajectory_versions, self.target_weight_versions
            )
            if target == target_step
            and self._is_valid_for_target(
                trajectory_version, target_step, max_age_steps
            )
        )

    def _truncate_to_max_size(self, current_training_step: int | None = None) -> None:
        """Truncate restored state to ``max_size`` after resume cleanup.

        Must be called while holding ``self._lock``.
        """
        if len(self.trajectories) <= self.max_size:
            return

        print(
            f"Truncating restored buffer from {len(self.trajectories)} "
            f"to max_size={self.max_size}"
        )
        if current_training_step is None:
            indices_to_keep = list(
                range(len(self.trajectories) - self.max_size, len(self.trajectories))
            )
        else:
            prioritized_indices = sorted(
                range(len(self.trajectories)),
                key=lambda i: (self.target_weight_versions[i], i),
            )
            indices_to_keep = sorted(prioritized_indices[: self.max_size])

        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]

    def get_trajectories_needed(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> int:
        """Return additional trajectories needed for ``target_step``."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return max(0, num_prompts_per_step - current_count)

    def has_complete_batch(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> bool:
        """Return whether ``target_step`` has enough trajectories to train."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return current_count >= num_prompts_per_step

    def _remove_incomplete_target_steps(self, num_prompts_per_step: int) -> None:
        """Remove target steps without a complete batch.

        Must be called while holding ``self._lock``.
        """
        target_counts = Counter(self.target_weight_versions)
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }
        if not incomplete_targets:
            print(f"   All target steps have complete batches ({num_prompts_per_step})")
            return

        print(f"   Removing incomplete target steps: {sorted(incomplete_targets)}")
        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target not in incomplete_targets
        ]
        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]
        print(
            f"   Removed {original_count - len(self.trajectories)} trajectories "
            "from incomplete target steps"
        )

        if self.target_weight_versions:
            first_remaining_target = min(self.target_weight_versions)
            self.last_target_weight_already_generated = min(
                self.last_target_weight_already_generated,
                first_remaining_target - 1,
            )
        else:
            self.last_target_weight_already_generated = -1


@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass


class TQReplayBuffer:
    """Meta cache + TQ writer with reserve-then-commit slot semantics.

    meta_list, weight_list, ready_list, _group_ids are parallel; a slot stays
    ready=False until commit fills it.
    """

    def __init__(
        self,
        dp_client: Any,
        partition_id: str,
        *,
        pad_value_dict: Mapping[str, int],
        staging_partition_id: Optional[str] = None,
        require_routed_experts: bool = False,
    ):
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._pad_value_dict = dict(pad_value_dict)
        # Token-capture mode only (docs/design-docs/token-capture-ledger.md):
        # the staging partition whose per-call delta rows `remove` must clear
        # alongside the canonical rows. None on the legacy path.
        self._staging_partition_id = staging_partition_id
        self._require_routed_experts = require_routed_experts
        self.meta_list: list[Optional[KVBatchMeta]] = []
        self.start_weight_list: list[int] = []
        self.end_weight_list: list[int] = []
        # Per-slot target training step (set when force_in_order=True, else None).
        self.target_step_list: list[Optional[int]] = []
        self.ready_list: list[bool] = []
        self._group_ids: list[str] = []
        # Parallel to the lists above; populated only in token-capture mode.
        self._rollout_ids_list: list[Optional[list[str]]] = []
        self._staging_keys_list: list[Optional[list[str]]] = []
        self._data_plane_checkpoint_barrier: Optional[DataPlaneCheckpointBarrier] = None

    def set_data_plane_checkpoint_barrier(
        self, barrier: DataPlaneCheckpointBarrier
    ) -> None:
        """Bind the controller's shared checkpoint/mutation barrier once.

        A private fallback barrier would not coordinate with controller-owned
        saves and clears, so destructive operations fail loudly until the SC
        actor supplies its barrier.
        """
        if self._data_plane_checkpoint_barrier is not None:
            raise RuntimeError("data-plane checkpoint barrier is already configured")
        self._data_plane_checkpoint_barrier = barrier

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Return a stable snapshot of controller-local replay ownership."""
        return tuple(self._group_ids)

    def reserve(
        self,
        *,
        weight_version: int,
        target_step: Optional[int] = None,
        group_id: Optional[str] = None,
        rollout_ids: Optional[list[str]] = None,
    ) -> str:
        """Append an unready slot tagged with weight_version.

        Args:
            weight_version: Weight version stamped on the slot.
            target_step: Training step this slot targets; only consulted by StalenessSampler.force_in_order.
            group_id: Per-group sample_id prefix; defaults to a fresh uuid4.
            rollout_ids: Token-capture mode: the ledger-registered rollout ids
                this slot dispatched, recorded so cleanup can name what it
                owns even before a receipt exists.

        Returns:
            group_id used by the matching commit.
        """
        if group_id is None:
            group_id = str(uuid.uuid4())
        self.meta_list.append(None)
        self.start_weight_list.append(weight_version)
        self.end_weight_list.append(-1)
        self.target_step_list.append(target_step)
        self.ready_list.append(False)
        self._group_ids.append(group_id)
        self._rollout_ids_list.append(
            list(rollout_ids) if rollout_ids is not None else None
        )
        self._staging_keys_list.append(None)
        return group_id

    async def commit(
        self,
        group_id: str,
        record: PromptGroupRecord,
        start_weight_version: int,
        end_weight_version: int,
    ) -> KVBatchMeta:
        """Tensorize record, write N rows to TQ, and mark the slot ready.

        Args:
            group_id: group_id returned by the matching reserve call.
            record: PromptGroupRecord to tensorize.
            start_weight_version: Weight version stamped on the slot before rollout.
                The same as the one from reserve, passed again to avoid race condition when lookup.
            end_weight_version: Weight version stamped on the slot after rollout.

        Returns:
            KVBatchMeta for the committed group.

        Raises:
            ValueError: group_id has no live slot (removed or never reserved).
            RuntimeError: router replay is enabled but the payload has no routes.
        """
        # Check the slot is still live BEFORE writing: a slot evicted while
        # its rollout was in flight must not orphan rows into the partition.
        if group_id not in self._group_ids:
            raise ValueError(
                f"TQReplayBuffer.commit: group {group_id} has no live slot "
                "(evicted or never reserved); nothing written"
            )
        if self._data_plane_checkpoint_barrier is None:
            raise RuntimeError(
                "TQReplayBuffer must be bound to the controller data-plane "
                "checkpoint barrier before committing samples"
            )
        train_batch = record_to_train_batch(record, pad_value_dict=self._pad_value_dict)
        sample_ids, fields, tags = pack_payload(
            train_batch, weight_version=start_weight_version, group_id=group_id
        )
        if self._require_routed_experts and ROUTED_EXPERTS_FIELD not in fields:
            raise RuntimeError(
                "policy.router_replay.enabled=true requires routed_experts in "
                "the SingleController rollout payload, but payload packing did "
                "not produce that field. Check vLLM routed-expert capture and "
                "the async message-log flattening path."
            )
        maybe_dump_train_rows(
            source="legacy_commit",
            group_id=group_id,
            sample_ids=list(sample_ids),
            train_batch=train_batch,
            weight_version=start_weight_version,
        )
        trace_rollout_payload(keys=sample_ids, data=train_batch)
        async with self._data_plane_checkpoint_barrier.mutation("group_commits"):
            try:
                await call_data_plane(
                    self._dp_client,
                    "put_samples",
                    sample_ids=sample_ids,
                    partition_id=self._partition_id,
                    fields=fields,
                    tags=tags,
                )

                # mirrors kv_first_write
                lengths = train_batch["input_lengths"]
                meta = KVBatchMeta(
                    partition_id=self._partition_id,
                    task_name="train",
                    sample_ids=list(sample_ids),
                    fields=list(fields.keys()),
                    sequence_lengths=[int(s) for s in lengths.tolist()],
                    tags=[dict(t) for t in tags],
                )

                try:
                    idx = self._group_ids.index(group_id)
                except ValueError:
                    raise ValueError(
                        f"TQReplayBuffer.commit: group {group_id} was evicted "
                        "during the write; rows cleared"
                    ) from None
                self.meta_list[idx] = meta
                self.end_weight_list[idx] = end_weight_version
                self.ready_list[idx] = True
                return meta
            except BaseException as commit_error:
                # put_samples may have written rows before raising. Roll back by the
                # deterministic IDs while retaining the barrier mutation slot.
                try:
                    await self._clear_samples_unlocked(
                        sample_ids=list(sample_ids),
                    )
                except BaseException as rollback_error:
                    if isinstance(commit_error, asyncio.CancelledError):
                        raise commit_error from rollback_error
                    raise BaseExceptionGroup(
                        f"commit and rollback both failed for group_id={group_id!r}",
                        [commit_error, rollback_error],
                    )
                raise

    async def remove_group(self, group_id: str, *, remove_in_dp: bool = False) -> int:
        """Remove the live slot identified by ``group_id``.

        Args:
            group_id: Group identifier returned by :meth:`reserve`.
            remove_in_dp: Whether to clear rows referenced by a committed slot.

        Returns:
            Number of removed slots (always one on success).

        Raises:
            ValueError: ``group_id`` has no live slot.
        """
        if self._data_plane_checkpoint_barrier is None:
            raise RuntimeError(
                "TQReplayBuffer must be bound to the controller data-plane "
                "checkpoint barrier before removing a group"
            )
        async with self._data_plane_checkpoint_barrier.mutation("group_removals"):
            try:
                idx = self._group_ids.index(group_id)
            except ValueError as error:
                raise ValueError(f"unknown group_id={group_id!r}") from error
            return await self._remove_unlocked([idx], clear_data_plane=remove_in_dp)

    async def clear_staging_keys(self, staging_keys: list[str]) -> None:
        """Clear known token-capture staging rows under the checkpoint barrier."""
        if not staging_keys:
            return
        if self._staging_partition_id is None:
            raise RuntimeError(
                "cannot clear token-capture staging keys without a staging partition"
            )
        if self._data_plane_checkpoint_barrier is None:
            raise RuntimeError(
                "TQReplayBuffer must be bound to the controller data-plane "
                "checkpoint barrier before clearing staging samples"
            )
        unique_keys = list(dict.fromkeys(staging_keys))
        async with self._data_plane_checkpoint_barrier.mutation("sample_clears"):
            await call_data_plane(
                self._dp_client,
                "clear_samples",
                offload_sync=True,
                sample_ids=unique_keys,
                partition_id=self._staging_partition_id,
            )

    async def commit_finalized(
        self,
        group_id: str,
        meta: KVBatchMeta,
        group_min_wv: int,
        group_max_wv: int,
        *,
        staging_keys: Optional[list[str]] = None,
    ) -> KVBatchMeta:
        """Mark a slot ready from finalizer output (token-capture mode).

        Unlike :meth:`commit`, the canonical rows are already in TQ — the
        finalizer tensorized and put them — so this only fills the slot.
        The slot's effective version is the group's OLDEST call version
        (``group_min_wv``): staleness accounting stays conservative when a
        rollout straddles a refit.

        Args:
            group_id: group_id returned by the matching reserve call.
            meta: KVBatchMeta the finalizer built over its published rows.
            group_min_wv: Oldest weight version any call in the group used.
            group_max_wv: Newest weight version any call in the group used.
            staging_keys: The group's staged delta keys, recorded so
                :meth:`remove` can clear the staging partition too.

        Raises:
            ValueError: group_id has no live slot (removed or never reserved).
        """
        if self._data_plane_checkpoint_barrier is None:
            raise RuntimeError(
                "TQReplayBuffer must be bound to the controller data-plane "
                "checkpoint barrier before committing finalized samples"
            )

        async with self._data_plane_checkpoint_barrier.mutation("group_commits"):
            try:
                idx = self._group_ids.index(group_id)
            except ValueError:
                raise ValueError(
                    f"TQReplayBuffer.commit_finalized: group {group_id} has no "
                    "live slot (evicted or never reserved)"
                ) from None
            tagged_plans = [
                tag[ROUTE_PLAN_TAG]
                for tag in (meta.tags or [])
                if ROUTE_PLAN_TAG in tag
            ]
            if tagged_plans:
                if len(tagged_plans) != len(meta.sample_ids):
                    raise ValueError(
                        "commit_finalized received mixed deferred/direct route plans"
                    )
                from nemo_rl.experience.route_plan import decode_route_plan

                plan_cleanup_keys = {
                    key
                    for encoded in tagged_plans
                    for key in decode_route_plan(encoded).cleanup_staging_keys
                }
                provided_staging_keys = list(staging_keys or [])
                if len(provided_staging_keys) != len(set(provided_staging_keys)):
                    raise ValueError(
                        "commit_finalized staging_keys contains duplicates"
                    )
                if set(provided_staging_keys) != plan_cleanup_keys:
                    raise ValueError(
                        "commit_finalized staging ownership does not match route plans: "
                        f"provided={sorted(provided_staging_keys)!r}, "
                        f"planned={sorted(plan_cleanup_keys)!r}"
                    )
            self.meta_list[idx] = meta
            self.start_weight_list[idx] = group_min_wv
            self.end_weight_list[idx] = group_max_wv
            self.ready_list[idx] = True
            self._staging_keys_list[idx] = (
                list(staging_keys) if staging_keys is not None else None
            )
            return meta

    def abort(self, group_id: str) -> bool:
        """Drop an unready slot whose dispatch failed or was cancelled.

        Token-capture mode; called from the failed dispatch path.
        No DataPlane rows are cleared here. Callers with sealed receipts must
        clear their deterministic canonical IDs and full staging manifests
        before dropping this ownership record. Before receipt sealing, orphan
        cleanup remains an explicit controlled-validation limitation.

        Returns:
            True when a slot was dropped; False when the group_id has no
            live slot (already committed+consumed or never reserved).
        """
        try:
            idx = self._group_ids.index(group_id)
        except ValueError:
            return False
        if self.ready_list[idx]:
            return False
        self._delete_slot(idx)
        return True

    def _delete_slot(self, idx: int) -> None:
        del self.meta_list[idx]
        del self.start_weight_list[idx]
        del self.end_weight_list[idx]
        del self.target_step_list[idx]
        del self.ready_list[idx]
        del self._group_ids[idx]
        del self._rollout_ids_list[idx]
        del self._staging_keys_list[idx]

    async def remove(self, idxs: list[int], remove_in_dp: bool) -> int:
        """Drop entries at the given indices and optionally clear them from DataPlane.

        In token-capture mode (``staging_partition_id`` set), clearing a
        group also clears its recorded staged delta rows, so eviction leaves
        neither canonical nor staging bytes behind.

        Args:
            idxs: Entry indices to drop. Must be within [0, size).
            remove_in_dp: If True, also clear the dropped rows from DataPlane.

        Returns:
            Number of group entries removed from the buffer.
        """
        if len(idxs) == 0:
            return 0
        if self._data_plane_checkpoint_barrier is None:
            raise RuntimeError(
                "TQReplayBuffer must be bound to the controller data-plane "
                "checkpoint barrier before removing groups"
            )
        async with self._data_plane_checkpoint_barrier.mutation("group_removals"):
            drop_idxs = sorted(idxs, reverse=True)
            if drop_idxs[0] >= len(self.meta_list):
                raise IndexError(
                    f"TQReplayBuffer.remove: indices out of range: {drop_idxs[0]}; "
                    f"size={len(self.meta_list)}"
                )
            return await self._remove_unlocked(drop_idxs, clear_data_plane=remove_in_dp)

    async def _remove_unlocked(
        self, drop_idxs: list[int], *, clear_data_plane: bool
    ) -> int:
        """Remove validated indices while the caller owns any required lock."""
        dropped_sample_ids: list[str] = []
        dropped_staging_keys: list[str] = []
        for i in drop_idxs:
            meta = self.meta_list[i]
            if meta is not None:
                dropped_sample_ids.extend(meta.sample_ids)
            staging_keys = self._staging_keys_list[i]
            if staging_keys:
                dropped_staging_keys.extend(staging_keys)

        if clear_data_plane:
            await self._clear_samples_unlocked(
                sample_ids=dropped_sample_ids,
            )
            if dropped_staging_keys and self._staging_partition_id is not None:
                try:
                    await self._call_dp(
                        "clear_samples",
                        sample_ids=dropped_staging_keys,
                        partition_id=self._staging_partition_id,
                    )
                except Exception as error:
                    raise RuntimeError(
                        "staging cleanup failed; retained replay-buffer ownership "
                        f"partition={self._staging_partition_id!r}, "
                        f"staging_keys={dropped_staging_keys!r}; canonical rows "
                        "may already be cleared"
                    ) from error

        for i in drop_idxs:
            self._delete_slot(i)

        return len(drop_idxs)

    def metadata_state_dict(
        self,
        *,
        saved_capacity: int,
        additional_groups: Optional[list[TQReplayGroupMetadata]] = None,
    ) -> TQReplayMetadataState:
        """Capture the controller index for ready groups without tensor payloads.

        The caller must hold the exclusive side of the shared data-plane
        checkpoint barrier through this capture and the matching TQ save.
        Commits and destructive clears use shared mutation slots, so the sidecar
        and native snapshot describe one exact set of training-ready groups.
        Every operation that mutates the canonical rollout partition or its
        controller-local replay membership must participate in that barrier
        across the complete publish/index or clear/remove transition. This
        includes future finalizer paths; canonical writes are not required to
        originate specifically from :meth:`commit`.
        In-flight reservations are intentionally omitted. ``additional_groups``
        is used only by periodic snapshots to re-index canonical groups already
        claimed by the current, uncheckpointed optimizer step. Their TQ rows are
        still present, but normal sampler selection has removed their live slots.
        """
        groups: list[TQReplayGroupMetadata] = []
        for i, ready in enumerate(self.ready_list):
            if not ready:
                continue
            meta = self.meta_list[i]
            assert meta is not None  # commit sets meta before ready=True
            groups.append(
                {
                    "meta": meta,
                    "start_weight": self.start_weight_list[i],
                    "end_weight": self.end_weight_list[i],
                    "target_step": self.target_step_list[i],
                    "group_id": self._group_ids[i],
                }
            )
        existing_group_ids = {group["group_id"] for group in groups}
        existing_sample_ids = {
            sample_id for group in groups for sample_id in group["meta"].sample_ids
        }
        for group in additional_groups or []:
            group_id = group["group_id"]
            if group_id in existing_group_ids:
                raise ValueError(
                    f"additional replay metadata duplicates group_id={group_id!r}"
                )
            duplicate_sample_ids = existing_sample_ids.intersection(
                group["meta"].sample_ids
            )
            if duplicate_sample_ids:
                raise ValueError(
                    "additional replay metadata duplicates sample IDs: "
                    f"{sorted(duplicate_sample_ids)!r}"
                )
            groups.append(group)
            existing_group_ids.add(group_id)
            existing_sample_ids.update(group["meta"].sample_ids)
        return {
            "schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
            "storage": REPLAY_BUFFER_METADATA_STORAGE,
            "partition_id": self._partition_id,
            "saved_capacity": saved_capacity,
            "manifest_digest": replay_manifest_digest(groups),
            "groups": groups,
        }

    async def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        max_groups: int,
        expected_partition_id: str,
        expected_group_size: int,
        expected_manifest_digest: str,
    ) -> int:
        """Restore the local replay index for an already-restored TQ snapshot.

        The sidecar never contains tensor payloads and this method never writes
        to the DataPlane. TQ must be restored first; the caller binds the two
        artifacts by passing the manifest digest returned by TQ checkpoint
        loading.

        Staleness is intentionally NOT handled here — load only loads. The
        train pump's first ``sampler.evict`` drops any restored group that is
        outside the staleness window and releases its capacity permit, keeping
        eviction in one place.

        Args:
            state: Envelope produced by ``metadata_state_dict``.
            max_groups: Current max_buffered_rollouts; the restored count
                never exceeds it.
            expected_partition_id: Partition this buffer writes to; must
                match the envelope.
            expected_group_size: num_generations_per_prompt; every group must
                hold exactly this many rows (a changed group size silently
                breaks the group-relative baseline).
            expected_manifest_digest: Digest returned by the matching native
                TQ checkpoint load. It must match the metadata sidecar.

        Returns:
            Number of groups restored into the buffer.

        Raises:
            ValueError: If the envelope is malformed (missing keys, partition
                mismatch, misaligned or wrongly sized groups, duplicate
                sample_ids), disagrees with the native TQ snapshot, or exceeds
                ``max_groups``.
        """
        if self.meta_list or self._group_ids:
            raise RuntimeError(
                "Replay-buffer checkpoint loading requires an empty local buffer"
            )
        required_keys = {
            "schema_version",
            "storage",
            "partition_id",
            "saved_capacity",
            "manifest_digest",
            "groups",
        }
        missing_keys = required_keys - set(state)
        if missing_keys:
            raise ValueError(
                f"Replay buffer checkpoint missing required keys: {missing_keys}"
            )
        if state["schema_version"] != REPLAY_BUFFER_METADATA_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported replay-buffer metadata schema version: "
                f"{state['schema_version']!r}"
            )
        if state["storage"] != REPLAY_BUFFER_METADATA_STORAGE:
            raise ValueError(
                f"Replay-buffer metadata has unsupported storage: {state['storage']!r}"
            )
        if state["partition_id"] != expected_partition_id:
            raise ValueError(
                "Replay buffer checkpoint partition_id mismatch: "
                f"checkpoint={state['partition_id']!r}, "
                f"expected={expected_partition_id!r}"
            )

        groups = list(state["groups"])
        group_keys = {
            "meta",
            "start_weight",
            "end_weight",
            "target_step",
            "group_id",
        }
        seen_sample_ids: set[str] = set()
        for group in groups:
            if "fields_data" in group:
                raise ValueError(
                    "Metadata-only replay checkpoint must not contain fields_data"
                )
            missing_group_keys = group_keys - set(group)
            if missing_group_keys:
                raise ValueError(
                    f"Replay buffer checkpoint group missing keys: {missing_group_keys}"
                )
            meta = group["meta"]
            if meta.partition_id != expected_partition_id:
                raise ValueError(
                    "Replay buffer checkpoint group partition_id mismatch: "
                    f"checkpoint={meta.partition_id!r}, "
                    f"expected={expected_partition_id!r}"
                )
            num_tags = len(meta.tags) if meta.tags is not None else -1
            num_lengths = (
                len(meta.sequence_lengths) if meta.sequence_lengths is not None else -1
            )
            if not (
                len(meta.sample_ids) == num_tags == num_lengths == expected_group_size
            ):
                raise ValueError(
                    "Replay buffer checkpoint group misaligned: "
                    f"sample_ids={len(meta.sample_ids)}, tags={num_tags}, "
                    f"sequence_lengths={num_lengths}, "
                    f"expected_group_size={expected_group_size}"
                )
            for sid in meta.sample_ids:
                if sid in seen_sample_ids:
                    raise ValueError(
                        f"Replay buffer checkpoint has duplicate sample_id: {sid!r}"
                    )
                seen_sample_ids.add(sid)

        actual_digest = replay_manifest_digest(groups)
        if state["manifest_digest"] != actual_digest:
            raise ValueError(
                "Replay-buffer metadata digest does not match its contents"
            )
        if expected_manifest_digest != actual_digest:
            raise ValueError(
                "Replay-buffer metadata does not match the loaded TQ checkpoint"
            )

        if state["saved_capacity"] != max_groups:
            print(
                "TQReplayBuffer capacity changed: "
                f"checkpoint={state['saved_capacity']}, current={max_groups}. "
                "Using current config value."
            )
        if len(groups) > max_groups:
            raise ValueError(
                "Native TQ checkpoint contains more replay groups than the current "
                f"buffer capacity: checkpoint={len(groups)}, current={max_groups}"
            )

        for group in groups:
            meta = group["meta"]
            self.meta_list.append(meta)
            self.start_weight_list.append(group["start_weight"])
            self.end_weight_list.append(group["end_weight"])
            self.target_step_list.append(group["target_step"])
            self.ready_list.append(True)
            self._group_ids.append(group["group_id"])

        print(
            f"📦 Restored {len(groups)} replay group(s) from checkpoint",
            flush=True,
        )
        return len(groups)

    def count_for_target_step(self, target_step: int) -> int:
        """Return how many slots are stamped with ``target_step``."""
        return sum(1 for target in self.target_step_list if target == target_step)

    def size(self) -> int:
        """Return the number of prompt-group entries currently held."""
        return len(self.meta_list)

    def __len__(self) -> int:
        return len(self.meta_list)

    async def _clear_samples(self, *, sample_ids: list[str]) -> None:
        """Clear rows without overlapping a bound data-plane checkpoint."""
        if self._data_plane_checkpoint_barrier is None:
            raise RuntimeError(
                "TQReplayBuffer must be bound to the controller data-plane "
                "checkpoint barrier before clearing samples"
            )
        async with self._data_plane_checkpoint_barrier.mutation("sample_clears"):
            await self._clear_samples_unlocked(sample_ids=sample_ids)

    async def _clear_samples_unlocked(self, *, sample_ids: list[str]) -> None:
        """Clear rows while the caller holds a barrier mutation slot."""
        await call_data_plane(
            self._dp_client,
            "clear_samples",
            offload_sync=True,
            sample_ids=sample_ids,
            partition_id=self._partition_id,
        )
