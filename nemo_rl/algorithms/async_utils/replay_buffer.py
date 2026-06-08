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
import threading as _threading
from collections import Counter
from typing import TYPE_CHECKING, Any, Iterable, Optional

import ray

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol
from nemo_rl.data_plane import KVBatchMeta

if TYPE_CHECKING:
    from tensordict import TensorDict


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
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
        }

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


# WIP: DO NOT USE - This class is WIP and may be changed without notice, please DO NOT USE it.
class TQReplayBuffer:
    """Plain (non-Ray) replay buffer holding prompt-group meta + a TQ proxy.

    -- WIP: DO NOT USE --

    Lives in the SingleController process. Each entry corresponds to one
    prompt group (one ``RolloutManager.generate_and_push`` writes one
    entry). Tensor payloads are stored in TransferQueue (TQ) via the
    :class:`DataPlaneClient`; only per-group :class:`KVBatchMeta` and the
    generating ``weight_version`` are cached here for selection by
    :class:`StalenessSampler`.

    Concurrency: single asyncio loop in the SC process; list appends and
    pops are not protected by a lock — callers must ``await`` the TQ
    round-trip in :meth:`add` / :meth:`remove` before the local state
    mutation, so the TQ state is the source of truth when an ``await``
    suspends the coroutine.
    """

    def __init__(self, dp_client: Any, partition_id: str):
        self._dp_client = dp_client
        self._partition_id = partition_id
        self.meta_list: list[KVBatchMeta] = []
        self.weight_list: list[int] = []

    async def add(
        self,
        sample_ids: list[str],
        fields: "TensorDict | None",
        tags: list[dict[str, Any]] | None,
        weight_version: int,
    ) -> KVBatchMeta:
        """Write one prompt group to TQ and append its meta locally.

        The TQ ``put_samples`` must complete before the local lists are
        mutated; otherwise the local view would lead the data plane and a
        concurrent sampler could hand out meta for samples TQ has not yet
        accepted.

        Args:
            sample_ids: Per-sample uids for this prompt group.
            fields: Tensor / NonTensorStack leaves to write into TQ.
            tags: Per-sample primitive metadata stamped on TQ.
            weight_version: Trainer weight version this group was
                generated under. Must be an ``int``; the legacy ``None``
                path is intentionally not supported.

        Returns:
            ``KVBatchMeta`` covering ``sample_ids``.
        """
        if not isinstance(weight_version, int) or isinstance(weight_version, bool):
            raise TypeError(
                f"TQReplayBuffer.add: weight_version must be int, got "
                f"{type(weight_version).__name__}"
            )
        meta = await self._call_dp(
            "put_samples",
            sample_ids=list(sample_ids),
            partition_id=self._partition_id,
            fields=fields,
            tags=tags,
        )
        self.meta_list.append(meta)
        self.weight_list.append(weight_version)
        return meta

    async def remove(self, sample_ids: list[str]) -> int:
        """Drop entries whose sample_ids are fully covered by the input.

        Removal is entry-level: an entry is dropped iff every one of its
        ``sample_ids`` appears in the input. Callers that only know
        partial-group ``sample_ids`` (a logic bug under the entry-as-group
        contract) cause a ``ValueError`` so the mismatch surfaces early.

        Order: TQ ``clear_samples`` runs first; local lists are mutated
        only after the data plane has acknowledged the drop.

        Args:
            sample_ids: Uids to clear; must align with whole-entry
                boundaries.

        Returns:
            Number of group entries dropped.
        """
        ids_set = set(sample_ids)
        keep_meta: list[KVBatchMeta] = []
        keep_weights: list[int] = []
        hit_ids: set[str] = set()
        n_removed = 0
        for meta, weight in zip(self.meta_list, self.weight_list):
            entry_ids = set(meta.sample_ids)
            if entry_ids <= ids_set:
                hit_ids.update(entry_ids)
                n_removed += 1
            else:
                keep_meta.append(meta)
                keep_weights.append(weight)
        leftover = ids_set - hit_ids
        if leftover:
            raise ValueError(
                f"TQReplayBuffer.remove: sample_ids did not align with whole-entry "
                f"boundaries; unmatched ids={sorted(leftover)}"
            )
        await self._call_dp(
            "clear_samples",
            sample_ids=list(sample_ids),
            partition_id=self._partition_id,
        )
        self.meta_list = keep_meta
        self.weight_list = keep_weights
        return n_removed

    def size(self) -> int:
        """Number of group entries currently held."""
        return len(self.meta_list)

    def __len__(self) -> int:
        return len(self.meta_list)

    async def _call_dp(self, method_name: str, **kwargs: Any) -> Any:
        """Invoke a ``DataPlaneClient`` method on either a plain client or a Ray actor."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await remote(**kwargs)
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
