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
from typing import Any, Iterable, Optional

import ray
from tensordict import TensorDict

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol
from nemo_rl.data_plane import KVBatchMeta


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
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated, target_weight_version
            )
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

            # Check for unexpected old trajectories
            old_trajectories = [
                v for v in self.trajectory_versions if v < min_valid_version
            ]
            if old_trajectories:
                raise ValueError(
                    f"Found {len(old_trajectories)} trajectories older than min_valid_version {min_valid_version}"
                )

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
        fields: TensorDict | None,
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
