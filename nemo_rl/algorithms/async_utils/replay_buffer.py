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
import statistics
import threading as _threading
import uuid
from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import ray

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.payload import pack_payload, record_to_train_batch
from nemo_rl.utils.r3_trace import trace_rollout_payload


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
        # Admission order is the scheduling order.  A version is the oldest
        # weight that may have contributed tokens to the prompt group.
        self.trajectory_versions = []
        # Kept only to read/write older checkpoints and satisfy old diagnostics.
        # It has no scheduling meaning.
        self.target_weight_versions = []
        self.last_target_weight_already_generated = -1
        self._reserved_slots = 0
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
        target_weight_version: int | None = None,
        *,
        reserved: bool = False,
    ) -> str:
        """Commit one complete prompt group at the FIFO tail."""
        with self._lock:
            if reserved:
                if self._reserved_slots <= 0:
                    raise RuntimeError("add(reserved=True) without a reserved slot")
                self._reserved_slots -= 1
            elif len(self.trajectories) + self._reserved_slots >= self.max_size:
                return "full"

            self.trajectories.append(trajectory)
            self.trajectory_versions.append(int(weight_version))
            self.target_weight_versions.append(target_weight_version)
            return "success"

    def reserve(self, num_prompt_groups: int) -> int:
        """Atomically reserve FIFO capacity for admitted rollout workers."""
        if num_prompt_groups < 0:
            raise ValueError("num_prompt_groups must be non-negative")
        if num_prompt_groups > self.max_size:
            raise ValueError(
                f"cannot reserve batch of {num_prompt_groups} groups in "
                f"max_size={self.max_size} FIFO"
            )
        with self._lock:
            available = self.max_size - len(self.trajectories) - self._reserved_slots
            granted = num_prompt_groups if available >= num_prompt_groups else 0
            self._reserved_slots += granted
            return granted

    def release_reserved(self, num_prompt_groups: int) -> None:
        """Release capacity for admitted groups that did not complete."""
        if num_prompt_groups < 0:
            raise ValueError("num_prompt_groups must be non-negative")
        with self._lock:
            if num_prompt_groups > self._reserved_slots:
                raise RuntimeError(
                    f"cannot release {num_prompt_groups} slots; "
                    f"only {self._reserved_slots} are reserved"
                )
            self._reserved_slots -= num_prompt_groups

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        info: dict[str, Any] = {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
            "reserved_slots": self._reserved_slots,
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
        """Consume one complete FIFO batch, evicting stale entries at the head.

        Returns:
            Dictionary with 'trajectories', 'avg_trajectory_age',
            'buffer_size_before_sample', and 'evicted_stale_count' keys,
            or None if insufficient data.
        """
        with self._lock:
            min_valid_version = max(0, current_weight_version - max_age_steps)
            evicted_stale_count = 0
            while (
                self.trajectory_versions
                and self.trajectory_versions[0] < min_valid_version
            ):
                self._remove_indices([0])
                evicted_stale_count += 1

            total_trajectories = len(self.trajectories)
            if total_trajectories < num_prompt_groups:
                return None

            selected = list(range(num_prompt_groups))
            sampled_weights = self.trajectory_versions[:num_prompt_groups]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            sampled_items = self.trajectories[:num_prompt_groups]
            self._remove_indices(selected)

            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated,
                current_weight_version,
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
                "buffer_size_before_sample": total_trajectories,
                "evicted_stale_count": evicted_stale_count,
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
            self._reserved_slots = 0

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        with self._lock:
            return {
                "format_version": 2,
                "trajectories": list(self.trajectories),
                "trajectory_versions": list(self.trajectory_versions),
                # Retained for checkpoint readers from the target-pinned era.
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
        """Restore FIFO order, migrating target-pinned checkpoints in place.

        Legacy target versions are accepted but never used to reorder or select.
        Staleness is intentionally evaluated only when the trainer dequeues.
        """
        with self._lock:
            required_keys = {
                "trajectories",
                "trajectory_versions",
            }
            missing_keys = required_keys - set(state)
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            trajectories = list(state["trajectories"])
            trajectory_versions = list(state["trajectory_versions"])
            target_weight_versions = list(
                state.get("target_weight_versions", [None] * len(trajectories))
            )
            if not len(trajectories) == len(trajectory_versions):
                raise ValueError(
                    "Checkpoint has inconsistent replay buffer lengths: "
                    f"trajectories={len(trajectories)}, "
                    f"trajectory_versions={len(trajectory_versions)}"
                )
            if len(target_weight_versions) != len(trajectories):
                # Old target metadata is dispensable.  Do not reject otherwise
                # usable FIFO data because that parallel list was malformed.
                target_weight_versions = [None] * len(trajectories)

            if "max_size" in state and state["max_size"] != self.max_size:
                print(
                    "ReplayBuffer max_size changed: "
                    f"checkpoint={state['max_size']}, current={self.max_size}. "
                    "Using current config value."
                )

            self.trajectories = trajectories
            self.trajectory_versions = trajectory_versions
            self.target_weight_versions = target_weight_versions
            self.last_target_weight_already_generated = int(
                state.get("last_target_weight_already_generated", -1)
            )
            self._reserved_slots = 0
            self._truncate_to_max_size()

            print(f"ReplayBuffer restored: {len(self.trajectories)} FIFO groups")

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
        indices_to_keep = list(range(self.max_size))

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
        """Compatibility readiness helper for the unpinned FIFO queue."""
        with self._lock:
            current_count = self._ready_fifo_count(target_step, max_age_steps)
            return max(0, num_prompts_per_step - current_count)

    def has_complete_batch(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> bool:
        """Return whether one complete FIFO batch can be consumed."""
        with self._lock:
            current_count = self._ready_fifo_count(target_step, max_age_steps)
            return current_count >= num_prompts_per_step

    def _ready_fifo_count(
        self, current_weight_version: int, max_age_steps: int | None
    ) -> int:
        """Count entries after the stale prefix without mutating the queue."""
        if max_age_steps is None:
            return len(self.trajectories)
        min_valid_version = max(0, current_weight_version - max_age_steps)
        first_valid = 0
        while (
            first_valid < len(self.trajectory_versions)
            and self.trajectory_versions[first_valid] < min_valid_version
        ):
            first_valid += 1
        return len(self.trajectories) - first_valid

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
        require_routed_experts: bool = False,
    ):
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._pad_value_dict = dict(pad_value_dict)
        self._require_routed_experts = require_routed_experts
        self.meta_list: list[Optional[KVBatchMeta]] = []
        self.start_weight_list: list[int] = []
        self.end_weight_list: list[int] = []
        # Per-slot target training step (set when force_in_order=True, else None).
        self.target_step_list: list[Optional[int]] = []
        self.ready_list: list[bool] = []
        self._group_ids: list[str] = []

    def reserve(
        self,
        *,
        weight_version: int,
        target_step: Optional[int] = None,
        group_id: Optional[str] = None,
    ) -> str:
        """Append an unready slot tagged with weight_version.

        Args:
            weight_version: Weight version stamped on the slot.
            target_step: Training step this slot targets; only consulted by StalenessSampler.force_in_order.
            group_id: Per-group sample_id prefix; defaults to a fresh uuid4.

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
        # Precondition: reserve() must have registered this group_id. Raise
        # before any side effects so a stray commit doesn't leak orphan DP rows.
        if group_id not in self._group_ids:
            raise ValueError(
                f"commit called with unknown group_id={group_id!r}; "
                f"reserve() must precede commit() (or the slot was already removed)"
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
        trace_rollout_payload(keys=sample_ids, data=train_batch)
        try:
            await self._call_dp(
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

            idx = self._group_ids.index(group_id)
            self.meta_list[idx] = meta
            self.end_weight_list[idx] = end_weight_version
            self.ready_list[idx] = True
            return meta
        except BaseException as commit_error:
            # put_samples may have written rows before raising. Roll back by the
            # deterministic IDs known here; the caller removes the reserved slot.
            try:
                await self._call_dp(
                    "clear_samples",
                    sample_ids=list(sample_ids),
                    partition_id=self._partition_id,
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
        try:
            idx = self._group_ids.index(group_id)
        except ValueError as error:
            raise ValueError(f"unknown group_id={group_id!r}") from error
        return await self.remove([idx], remove_in_dp=remove_in_dp)

    async def remove(self, idxs: list[int], remove_in_dp: bool) -> int:
        """Drop entries at the given indices and optionally clear them from DataPlane.

        Args:
            idxs: Entry indices to drop. Must be within [0, size).
            remove_in_dp: If True, also clear the dropped rows from DataPlane.

        Returns:
            Number of group entries removed from the buffer.
        """
        if len(idxs) == 0:
            return 0

        drop_idxs = sorted(idxs, reverse=True)
        if drop_idxs[0] >= len(self.meta_list):
            raise IndexError(
                f"TQReplayBuffer.remove: indices out of range: {drop_idxs[0]}; "
                f"size={len(self.meta_list)}"
            )

        dropped_sample_ids: list[str] = []
        for i in drop_idxs:
            meta = self.meta_list[i]
            if meta is not None:
                dropped_sample_ids.extend(meta.sample_ids)
            del self.meta_list[i]
            del self.start_weight_list[i]
            del self.end_weight_list[i]
            del self.target_step_list[i]
            del self.ready_list[i]
            del self._group_ids[i]

        if remove_in_dp:
            await self._call_dp(
                "clear_samples",
                sample_ids=dropped_sample_ids,
                partition_id=self._partition_id,
            )

        return len(drop_idxs)

    def size(self) -> int:
        """Return the number of prompt-group entries currently held."""
        return len(self.meta_list)

    def __len__(self) -> int:
        return len(self.meta_list)

    async def _call_dp(self, method_name: str, **kwargs: Any) -> Any:
        """Call a DataPlaneClient method, awaiting Ray remotes if needed."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await remote(**kwargs)
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
