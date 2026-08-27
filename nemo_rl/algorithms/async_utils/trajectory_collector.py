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

from __future__ import annotations

import asyncio
import concurrent.futures
import threading as _threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from typing import Any, Optional, cast

import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    AsyncGRPOConfig,
    GRPOConfig,
    _get_effort_config,
)
from nemo_rl.algorithms.grpo import (
    MasterConfig as GRPOMasterConfig,
)
from nemo_rl.algorithms.opd import resolve_reference_aliases, teacher_seq_pad_multiple
from nemo_rl.algorithms.ppo import (
    AsyncPPOConfig,
    PPOConfig,
)
from nemo_rl.algorithms.ppo import (
    MasterConfig as PPOMasterConfig,
)
from nemo_rl.data.dataloader import CyclingDataLoader
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import should_use_nemo_gym
from nemo_rl.experience.interfaces import (
    NEMO_GYM_TASK_INDEX_KEY,
    NEXT_NEMO_GYM_TASK_INDEX_KEY,
    PENDING_PROMPTS_KEY,
)
from nemo_rl.experience.rollouts import (
    RolloutGroupResult,
    attach_initial_nemo_gym_image_payloads,
    run_async_multi_turn_rollout_groups,
)
from nemo_rl.models.generation.interfaces import GenerationConfig, GenerationInterface
from nemo_rl.utils.logger import should_log_nemo_gym_full_result_tables
from nemo_rl.utils.multimodal_payload_metrics import (
    collect_multimodal_payload_metrics,
    drain_multimodal_payload_metrics,
    print_multimodal_payload_metrics,
)
from nemo_rl.utils.timer import ThreadSafeTimer

TokenizerType = PreTrainedTokenizerBase
_MAX_NEMO_GYM_STREAM_RETRIES = 3
_NEMO_GYM_RETRY_DELAY_BASE_SECONDS = 1.0
_REPLAY_BUFFER_MAX_BACKOFF_SECONDS = 0.5


def _stamped_task_indices(batch: BatchedDataDict[DatumSpec]) -> list[int]:
    """Stamped per-group ordinals of a pre-repeat slice ([] when unstamped)."""
    rows = batch.get("extra_env_info")
    if not isinstance(rows, list):
        return []
    raw_ordinals = [
        row.get(NEMO_GYM_TASK_INDEX_KEY) if isinstance(row, dict) else None
        for row in rows
    ]
    if not raw_ordinals or any(ordinal is None for ordinal in raw_ordinals):
        return []
    return [int(ordinal) for ordinal in cast("list[int]", raw_ordinals)]


def _unanimous_task_index(rows: list[Any]) -> Optional[int]:
    """Return the task index every row agrees on, or ``None``.

    A prompt group is one prompt repeated ``num_generations`` times, so its
    stamped rows all carry the same ordinal. Returns ``None`` for unstamped
    rows (legacy runs) rather than guessing.
    """
    if not rows:
        return None
    ordinals = {
        row.get(NEMO_GYM_TASK_INDEX_KEY) if isinstance(row, dict) else None
        for row in rows
    }
    if len(ordinals) != 1:
        return None
    (ordinal,) = ordinals
    return int(ordinal) if ordinal is not None else None


@ray.remote  # pragma: no cover
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to replay buffer."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: GRPOMasterConfig | PPOMasterConfig,
        replay_buffer: Any,
        start_step: int = 0,
        teacher_worker_groups: Optional[dict[str, Any]] = None,
        alias_to_group_alias: Optional[dict[str, str]] = None,
        on_policy_distillation_cfg: Optional[dict[str, Any]] = None,
        next_nemo_gym_task_index: int = 0,
        processor: Any = None,
        pending_batch: Optional[BatchedDataDict[DatumSpec]] = None,
        ordinals_frontier_aligned: bool = True,
        resume_frontier_ordinal: Optional[int] = None,
        resume_covered_task_indices: Optional[list[int]] = None,
    ) -> None:
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        algorithm_config: GRPOConfig | PPOConfig
        async_config: AsyncGRPOConfig | AsyncPPOConfig
        if isinstance(master_config, GRPOMasterConfig):
            algorithm_config = master_config.grpo
            async_config = algorithm_config.async_grpo  # type: ignore
            self._deduplicate_multimodal_data = (
                algorithm_config.deduplicate_multimodal_data
            )
            self._debug_payload_metrics = algorithm_config.debug_payload_metrics
            self._max_generation_failures = async_config.max_generation_failures
        elif isinstance(master_config, PPOMasterConfig):
            algorithm_config = master_config.ppo
            async_config = algorithm_config.async_ppo  # type: ignore
            self._deduplicate_multimodal_data = False
            self._debug_payload_metrics = False
            self._max_generation_failures = 0
        else:
            raise TypeError(
                "master_config must be a GRPO or PPO MasterConfig, got "
                f"{type(master_config).__name__}"
            )
        self.algorithm_config = algorithm_config
        self.async_config = async_config
        self._num_prompts_per_step = int(algorithm_config.num_prompts_per_step)
        self._num_generations_per_prompt = algorithm_config.num_generations_per_prompt
        self._max_rollout_turns = algorithm_config.max_rollout_turns
        self.replay_buffer = replay_buffer
        self.teacher_worker_groups = teacher_worker_groups or {}
        self.alias_to_group_alias = alias_to_group_alias or {}
        self.on_policy_distillation_cfg = on_policy_distillation_cfg or {}
        self.processor = processor
        self._has_distillation_teachers = bool(self.teacher_worker_groups)
        self._teacher_seq_pad_multiple = teacher_seq_pad_multiple(
            self.teacher_worker_groups,
            self.master_config.policy["make_sequence_length_divisible_by"],
        )
        # Per-teacher locks to serialize get_logprobs calls. Concurrent calls
        # to the same teacher cause NCCL collective desync across workers
        # (different workers may receive requests in different order → SeqNum
        # mismatch → 600s timeout → crash). Different teachers can still run
        # in parallel since they use separate NCCL groups on separate nodes.
        self._teacher_locks: dict[str, _threading.Lock] = {
            k: _threading.Lock() for k in self.teacher_worker_groups
        }
        self.running = False
        self.data_exhausted = False
        self.collection_failed = False
        self.collection_error: Optional[str] = None
        self._failure_lock: _threading.Lock = _threading.Lock()

        self._pg_lock: _threading.Lock = _threading.Lock()

        # Event for manual pause/resume control
        self._manual_pause_cleared = _threading.Event()
        self._manual_pause_cleared.set()

        self._refit_pause_cleared = _threading.Event()
        self._refit_pause_cleared.set()  # Start in cleared state

        self.current_weight_version: int = start_step
        self.initial_weight_version: int = start_step
        self.dataloader: StatefulDataLoader | CyclingDataLoader | None = None
        self.collection_thread: _threading.Thread | None = None
        self._generation_lead_steps = self.async_config.max_trajectory_age_steps
        self._max_trajectory_age_steps = self.async_config.max_trajectory_age_steps

        # Track when generation limits cause collection to pause
        self._last_limit_warning_version: int | None = None

        # Event to signal when generation limits are cleared (more efficient than polling)
        self._generation_limit_cleared = _threading.Event()
        self._generation_limit_cleared.set()  # Start in cleared state

        # Track threads
        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()
        self._next_nemo_gym_task_index = next_nemo_gym_task_index

        # Unconsumed suffix of the last gap-fill batch, consumed before the
        # next dataloader pull and serialized into fallback checkpoints.
        # Written by the collection thread, read at checkpoint time from the
        # actor thread, hence the lock.
        self._pending_lock: _threading.Lock = _threading.Lock()
        self._pending_batch: Optional[BatchedDataDict[DatumSpec]] = pending_batch

        # Frontier-aligned checkpointing. Every yielded prompt is stamped
        # with a monotonic ordinal equal to its position in the dataloader
        # stream; this ring of pre-pull dataloader snapshots, keyed by that
        # ordinal, is what checkpoints save instead of the live cursor. On a
        # frontier restore, re-yielded rows that are already covered (below
        # the cut, or in resume_covered_task_indices) are dropped at yield
        # and the gap-fill path regenerates the rest.
        self._snapshot_lock: _threading.Lock = _threading.Lock()
        self._dataloader_snapshots: deque[tuple[int, dict]] = deque(maxlen=512)
        # False when ordinals cannot be trusted to equal the trained-prompt
        # position (legacy-checkpoint resume, or unstampable batches); the
        # checkpoint then falls back to saving the live dataloader cursor.
        self._ordinals_frontier_aligned = ordinals_frontier_aligned
        self._resume_frontier_ordinal = resume_frontier_ordinal
        self._covered_task_indices: set[int] = set(resume_covered_task_indices or [])
        self._skip_horizon = max(
            [*self._covered_task_indices, (resume_frontier_ordinal or 0) - 1]
        )

        # Group ordinals dispatched to a rollout worker and not yet buffered.
        # Target interleaving can leave ordinals below the trained frontier
        # in flight; the checkpoint cut must not sit above the minimum of
        # this set (see get_checkpoint_state).
        self._outstanding_lock: _threading.Lock = _threading.Lock()
        self._outstanding_task_indices: set[int] = set()

        # Timer for efficiency metrics
        self._efficiency_timer = ThreadSafeTimer(context={"worker": "collector"})

        # Failure tracking for rollout batch workers.
        self._failure_count: int = 0
        self._fatal_error_message: str | None = None

    def _calculate_target_weights(self, generation_weight_version: int) -> list[int]:
        """Calculate target weight versions for given generation weight version.

        The list of versions returned enumerate the possible version a generation
        server can target. These versions are looped over to see what training
        step they can target. If all target versions are exhausted, this generation
        server will remain idle until the next weight update.

        Example:
        generation_weight_version = 10
        generation_lead_steps = 4

        Generation lead usually equals maximum trajectory age, but PPO critic
        warmup can temporarily configure them independently.

        Returns:
            [11, 12, 13, 14]  # Meaning this generation server can create trajectories for training step 11, 12, 13, 14
        """
        generation_lead = self._generation_lead_steps
        if generation_weight_version == self.initial_weight_version:
            return [
                i
                for i in range(
                    self.initial_weight_version,
                    self.initial_weight_version + generation_lead + 1,
                )
            ]

        return [generation_weight_version + i for i in range(1, generation_lead + 1)]

    def _get_next_target_for_generation(
        self, generation_weight_version: int
    ) -> Optional[int]:
        """Get the next target weight that needs generation (if any)."""
        target_weights = self._calculate_target_weights(generation_weight_version)
        num_prompts = self._num_prompts_per_step
        max_age_steps = self._max_trajectory_age_steps
        last_consumed_target = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )

        with self._generation_check_lock:
            for target_weight in target_weights:
                if target_weight <= last_consumed_target:
                    continue
                if target_weight in self._generating_targets:
                    continue

                trajectories_needed = ray.get(
                    self.replay_buffer.get_trajectories_needed.remote(
                        target_weight, num_prompts, max_age_steps
                    )
                )
                if trajectories_needed <= 0:
                    continue

                self._generating_targets.add(target_weight)
                if trajectories_needed < num_prompts:
                    print(
                        f"🎯 Reserved target weight {target_weight} for gap-filling "
                        f"(need {trajectories_needed}/{num_prompts} more trajectories)"
                    )
                else:
                    print(f"🎯 Reserved target weight {target_weight} for generation")
                return target_weight

        return None

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

        # Resume collection if it was paused due to generation limits
        was_paused = not self._generation_limit_cleared.is_set()
        if was_paused:
            self._generation_limit_cleared.set()  # Signal that collection can resume
            print(f"🔄 Updated weight version to {version}, resuming collection")
        else:
            print(f"🔄 Updated weight version to {version}")

    def set_generation_window(
        self,
        *,
        weight_version: int,
        generation_lead_steps: int,
        max_trajectory_age_steps: int,
    ) -> None:
        """Update the PPO generation version, lead, and buffer-validity age."""
        if generation_lead_steps < 1:
            raise ValueError("generation_lead_steps must be at least 1")
        if max_trajectory_age_steps < generation_lead_steps:
            raise ValueError(
                "max_trajectory_age_steps must be greater than or equal to "
                "generation_lead_steps"
            )

        with self._generation_check_lock:
            self.current_weight_version = weight_version
            self._generation_lead_steps = generation_lead_steps
            self._max_trajectory_age_steps = max_trajectory_age_steps

        self._generation_limit_cleared.set()
        print(
            f"🔄 Updated generation window: version={weight_version}, "
            f"lead={generation_lead_steps}, max_age={max_trajectory_age_steps}"
        )

    def _should_pause_for_generation_limits(self) -> bool:
        """Check if collection should be paused due to generation limits."""
        try:
            target_weights = self._calculate_target_weights(self.current_weight_version)
            num_prompts = self._num_prompts_per_step
            max_age_steps = self._max_trajectory_age_steps
            last_consumed_target = ray.get(
                self.replay_buffer.get_last_target_weight_already_generated.remote()
            )

            with self._generation_check_lock:
                # Check if any target weight in our range needs generation
                for target_weight in target_weights:
                    if target_weight <= last_consumed_target:
                        continue
                    if target_weight in self._generating_targets:
                        continue
                    trajectories_needed = ray.get(
                        self.replay_buffer.get_trajectories_needed.remote(
                            target_weight, num_prompts, max_age_steps
                        )
                    )
                    if trajectories_needed > 0:
                        return False  # Found a target that needs generation

            print(
                f"⏸️ All target weights {target_weights} already generated or in progress, pausing"
            )
            return True
        except Exception:
            return False

    def start_collection(
        self, dataloader: StatefulDataLoader | CyclingDataLoader
    ) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader

        print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("Collection thread started, start_collection returning")

    def is_data_exhausted(self) -> bool:
        """Check if collection stopped because the dataloader ran out of data."""
        return self.data_exhausted

    def get_status(self) -> dict:
        """Return a snapshot of the collector's internal state for driver-side diagnostics."""
        with self._threads_lock:
            inflight_workers = len(self._inflight_threads)
        with self._failure_lock:
            collection_failed = self.collection_failed
            collection_error = self.collection_error
        return {
            "running": self.running,
            "data_exhausted": self.data_exhausted,
            "errored": collection_failed,
            "error": collection_error,
            "inflight_workers": inflight_workers,
        }

    def _mark_collection_failed(self, error: Exception) -> None:
        """Record the first collection-loop failure."""
        with self._failure_lock:
            if not self.collection_failed:
                self.collection_failed = True
                self.collection_error = f"{type(error).__name__}: {error}"

    def _collection_loop(self):
        """Run the collection loop in background thread.

        Prompts left over from a gap-fill slice (``_pending_batch``) are
        consumed before the next dataloader pull, so a partially used batch is
        never discarded. The dataloader counts as exhausted only when the
        iterator drains with no pending prompts remaining.
        """
        dataloader_exhausted = False
        if self.dataloader is None:
            raise RuntimeError(
                "start_collection must set a dataloader before collection"
            )
        dataloader_iter = iter(self.dataloader)
        try:
            while self.running:
                # Check if manually paused and wait
                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                # Check if refit is in progress and wait
                if not self._refit_pause_cleared.is_set() and self.running:
                    print("⏸️ Pausing collection for refit...")
                    with self._efficiency_timer.time("idle/refit_event_wait"):
                        self._refit_pause_cleared.wait()
                    print("▶️ Refit completed, resuming collection")

                # Check if generation limits require pausing collection
                if self._should_pause_for_generation_limits() and self.running:
                    self._generation_limit_cleared.clear()

                    # Only log warning once per weight version
                    if self._last_limit_warning_version != self.current_weight_version:
                        target_weights = self._calculate_target_weights(
                            self.current_weight_version
                        )
                        print(
                            f"⏸️ Pausing collection: all target weights {target_weights} for weight version {self.current_weight_version} "
                            f"already exist in buffer. Waiting for weight update..."
                        )
                        self._last_limit_warning_version = self.current_weight_version

                    # Efficiently wait for generation limits to be cleared (no polling!)
                    with self._efficiency_timer.time("idle/generation_limit_pause"):
                        self._generation_limit_cleared.wait()

                    # Double-check we're still running after being woken up
                    if not self.running:
                        break

                if not self.running:
                    break

                # Carried-over prompts take priority over a fresh pull so the
                # dataloader stream stays strictly ordered and lossless. They
                # were stamped when first pulled, so only fresh batches are
                # stamped and snapshotted here.
                with self._pending_lock:
                    batch = self._pending_batch
                    self._pending_batch = None
                if batch is None:
                    pre_pull_state = self._capture_dataloader_state()
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        dataloader_exhausted = True
                        break
                    first_ordinal = self._next_nemo_gym_task_index
                    if self._stamp_task_indices(batch) and pre_pull_state is not None:
                        with self._snapshot_lock:
                            self._dataloader_snapshots.append(
                                (first_ordinal, pre_pull_state)
                            )
                    batch = self._filter_covered_rows(batch)
                    if batch is None:
                        continue

                leftover = self._process_batch(batch)
                if leftover is not None:
                    with self._pending_lock:
                        self._pending_batch = leftover
                    if leftover is batch:
                        # Nothing was consumed (e.g. no target needed
                        # generation). Yield briefly so the retry does not
                        # busy-spin against the replay buffer.
                        time.sleep(0.05)

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
            self._mark_collection_failed(e)
        finally:
            self.running = False
            if dataloader_exhausted:
                self.data_exhausted = True
                print(
                    "❌ Trajectory collection stopped: dataloader exhausted "
                    "(max_num_epochs reached). No more data available for generation. "
                    "Increase max_num_epochs or use a larger dataset."
                )
            else:
                print("🛑 Trajectory collection stopped")

    def _stamp_task_indices(self, batch: BatchedDataDict[DatumSpec]) -> bool:
        """Assign one stable, monotonic task index to every prompt in a batch.

        The ordinal equals the prompt's global position in the dataloader
        stream, which is what frontier-aligned checkpointing keys on. Batches
        whose ``extra_env_info`` rows are not dicts cannot carry a stamp; the
        collector then permanently falls back to live-cursor checkpoints.

        Returns:
            Whether the batch was stamped.
        """
        rows = batch.get("extra_env_info")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            if self._ordinals_frontier_aligned:
                self._ordinals_frontier_aligned = False
                print(
                    "⚠️ Batch rows cannot carry task indices; checkpoints "
                    "fall back to the live dataloader cursor."
                )
            return False

        stamped_rows = []
        first_task_index = self._next_nemo_gym_task_index
        for offset, row in enumerate(rows):
            stamped_row = dict(row)
            stamped_row[NEMO_GYM_TASK_INDEX_KEY] = first_task_index + offset
            stamped_rows.append(stamped_row)

        batch["extra_env_info"] = stamped_rows
        self._next_nemo_gym_task_index += len(stamped_rows)
        return True

    def _capture_dataloader_state(self) -> Optional[dict]:
        """Snapshot the dataloader state, or None when it has no state_dict."""
        if self.dataloader is not None and hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return None

    def _filter_covered_rows(
        self, batch: BatchedDataDict[DatumSpec]
    ) -> Optional[BatchedDataDict[DatumSpec]]:
        """Drop re-yielded rows a frontier restore already accounts for.

        After a frontier-aligned restore the dataloader re-yields the window
        between the cut and the old cursor. Rows already covered (ordinal
        below the cut, or in the covered set) are dropped; the normal
        gap-fill path regenerates the rest.

        Returns:
            The (possibly row-filtered) batch, or ``None`` when every row was
            already covered.
        """
        if self._resume_frontier_ordinal is None:
            return batch

        rows = batch.get("extra_env_info")
        raw_ordinals = [
            row.get(NEMO_GYM_TASK_INDEX_KEY) if isinstance(row, dict) else None
            for row in (rows if isinstance(rows, list) else [])
        ]
        if not raw_ordinals or any(ordinal is None for ordinal in raw_ordinals):
            raise ValueError(
                "Frontier-aligned resume requires task indices on every row, "
                "but a re-yielded batch is missing them. The checkpoint was "
                "written by a run whose batches carried task indices, so this "
                "indicates the dataset or task-data processor changed since "
                "the checkpoint was written; silently continuing could "
                "re-train retained prompts. Resume with the original data "
                "configuration, or start a fresh run."
            )
        ordinals = cast(list[int], raw_ordinals)

        if min(ordinals) > self._skip_horizon:
            # The covered window has been fully re-yielded; stop filtering.
            self._resume_frontier_ordinal = None
            self._covered_task_indices = set()
            return batch

        frontier = self._resume_frontier_ordinal
        keep = [
            position
            for position, ordinal in enumerate(ordinals)
            if ordinal >= frontier and ordinal not in self._covered_task_indices
        ]
        if len(keep) == len(ordinals):
            return batch
        dropped = len(ordinals) - len(keep)
        print(
            f"🔁 Frontier restore: dropping {dropped} already-covered prompts "
            f"(ordinals {ordinals[0]}–{ordinals[-1]}), regenerating {len(keep)}"
        )
        if not keep:
            return None
        return batch.select_indices(keep)

    def get_checkpoint_dataloader_state(self, frontier_ordinal: int) -> dict[str, Any]:
        """Return the dataloader state a checkpoint should persist.

        Returns the newest ring snapshot at or below ``frontier_ordinal``, so
        a resume re-yields everything past it. Falls back to the live cursor
        when ordinals are not frontier-aligned or the snapshot ring no longer
        covers the frontier.

        Args:
            frontier_ordinal: The trained-prompt count (consumed_samples).

        Returns:
            Mapping with ``dataloader_state``, ``base_ordinal`` (the ordinal
            the saved state resumes yielding from; ``None`` on fallback), and
            ``frontier_aligned``.
        """
        if self._ordinals_frontier_aligned:
            best: Optional[tuple[int, dict]] = None
            with self._snapshot_lock:
                ring_bases = [base for base, _ in self._dataloader_snapshots]
                for base_ordinal, state in self._dataloader_snapshots:
                    if base_ordinal <= frontier_ordinal and (
                        best is None or base_ordinal > best[0]
                    ):
                        best = (base_ordinal, state)
            if best is not None:
                return {
                    "dataloader_state": best[1],
                    "base_ordinal": best[0],
                    "frontier_aligned": True,
                }
            print(
                "⚠️ No dataloader snapshot at or below frontier ordinal "
                f"{frontier_ordinal} (snapshot ring covers "
                f"{f'[{min(ring_bases)}, {max(ring_bases)}]' if ring_bases else 'nothing'}); "
                "this checkpoint falls back to the live cursor and a resume "
                "from it may skip in-flight prompts."
            )
        return {
            "dataloader_state": self.get_dataloader_state(),
            "base_ordinal": None,
            "frontier_aligned": False,
        }

    def get_checkpoint_state(self, frontier_ordinal: int) -> dict[str, Any]:
        """Return the dataloader snapshot and rollout state as one consistent pair.

        Both are read while holding the pending lock so the collection loop
        cannot consume or replace the pending batch in between; a torn pair
        (cursor newer than its pending suffix) would duplicate prompts on a
        fallback resume.

        The pending remainder is only included on fallback snapshots. On a
        frontier-aligned snapshot the rewound dataloader re-yields it (the
        restore path forces ``pending_batch=None``), so persisting it would
        be dead weight in every frontier checkpoint.

        The snapshot is taken at the conservative **cut** — the minimum of
        the trained frontier, the lowest ordinal still out with a rollout
        worker, and the lowest ordinal held in the replay buffer (buffered-
        but-untrained groups are covered only by the buffer, which a
        ``load_replay_buffer=false`` resume discards). Normally everything
        in flight or buffered sits at or above the frontier and the cut
        equals it; when a target is refilled from later prompts (after a
        tolerated generation failure, or when gap-filling an incomplete
        target restored from a checkpoint), training it can advance past
        another target's in-flight groups, and cutting at the frontier
        would strand those prompts. Cutting below the frontier instead
        re-yields a window that includes already-trained prompts; the
        driver persists those trained ordinals in the checkpoint
        (``TRAINED_TASK_INDICES_KEY``) so the resume covers them like
        retained groups — nothing is skipped and nothing is re-trained.

        Returns:
            Mapping with ``dataloader`` (shape of
            :meth:`get_checkpoint_dataloader_state`, plus ``frontier_ordinal``
            = the cut, which the checkpoint must persist as its filter
            threshold) and ``rollouts`` (shape of :meth:`get_rollouts_state`).
        """
        with self._outstanding_lock:
            outstanding_min = min(self._outstanding_task_indices, default=None)
        # The cut is also bounded by buffered-untrained ordinals: their only
        # record is the buffer, which load_replay_buffer=false discards on
        # resume. Read after the outstanding set — an ordinal leaves it only
        # once its buffer add succeeded, so no group is missed by both reads.
        held_task_indices = ray.get(self.replay_buffer.get_held_task_indices.remote())
        buffered_min = min(held_task_indices, default=None)
        cut = frontier_ordinal
        for candidate in (outstanding_min, buffered_min):
            if candidate is not None and candidate < cut:
                cut = candidate
        if cut < frontier_ordinal:
            print(
                f"⚠️ Checkpoint cut lowered from trained frontier "
                f"{frontier_ordinal} to {cut}: ordinals below the frontier "
                "are still in flight or buffered-untrained (a gap-filled "
                "target was refilled from later prompts — after a tolerated "
                "generation failure, or when resuming with an incomplete "
                "restored target). A resume from this checkpoint regenerates "
                "the window instead of skipping it; trained prompts above "
                "the cut are persisted in rollouts.pt and stay covered."
            )
        with self._pending_lock:
            dataloader_snapshot = self.get_checkpoint_dataloader_state(cut)
            dataloader_snapshot["frontier_ordinal"] = cut
            rollouts_state = self._build_rollouts_state(
                include_pending=not dataloader_snapshot["frontier_aligned"]
            )
        return {"dataloader": dataloader_snapshot, "rollouts": rollouts_state}

    def _process_batch(
        self, batch: BatchedDataDict[DatumSpec]
    ) -> Optional[BatchedDataDict[DatumSpec]]:
        """Process a batch, generating for one target weight.

        Args:
            batch: Prompt batch pulled from the dataloader (or carried over
                from a previous gap-fill remainder).

        Returns:
            The unconsumed remainder of ``batch`` — the whole batch when no
            target currently needs generation, the sliced-off suffix when this
            target needed fewer prompts than the batch holds, or ``None`` when
            every prompt was consumed. The caller re-queues it so no yielded
            prompt is ever discarded.
        """
        target_weight: Optional[int] = None
        worker_started = False
        leftover: Optional[BatchedDataDict[DatumSpec]] = None
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self._num_generations_per_prompt
            num_prompts_in_batch = batch.size
            num_prompts_per_step = self._num_prompts_per_step
            max_age_steps = self._max_trajectory_age_steps

            # Get the next target weight that needs generation
            target_weight = self._get_next_target_for_generation(
                generation_weight_version
            )

            if target_weight is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                return batch
            reserved_target = target_weight

            print(
                f"🎯 Generating for target weight {reserved_target} from generation_weight_version {generation_weight_version}"
            )

            trajectories_needed = ray.get(
                self.replay_buffer.get_trajectories_needed.remote(
                    reserved_target, num_prompts_per_step, max_age_steps
                )
            )
            num_prompts_to_generate = min(num_prompts_in_batch, trajectories_needed)
            if num_prompts_to_generate == 0:
                print(
                    f"🔄 Target {reserved_target} already has enough trajectories, skipping"
                )
                self._release_target(reserved_target)
                return batch

            if num_prompts_to_generate < num_prompts_in_batch:
                # Keep the unused suffix instead of discarding it: the caller
                # re-queues it ahead of the next dataloader pull.
                leftover = batch.slice(num_prompts_to_generate, num_prompts_in_batch)
                print(
                    f"🎯 Gap-filling for target weight {reserved_target}: "
                    f"generating {num_prompts_to_generate}/{num_prompts_in_batch} "
                    f"prompts (need {trajectories_needed} more trajectories); "
                    f"carrying over the remaining "
                    f"{num_prompts_in_batch - num_prompts_to_generate}"
                )

            # Generate all prompt groups needed for this target in one batched worker.
            use_nemo_gym = should_use_nemo_gym(self.master_config)

            if not self._refit_pause_cleared.is_set() and self.running:
                with self._threads_lock:
                    active_threads = len(self._inflight_threads)
                print(
                    "⏸️ Waiting for refit to complete before starting new "
                    f"generation ({active_threads} threads still active)"
                )
                with self._efficiency_timer.time("idle/refit_event_wait"):
                    self._refit_pause_cleared.wait()
                generation_weight_version = self.current_weight_version

            # Task indices are stamped at yield time in _collection_loop, so
            # slices and carried-over remainders keep their original ordinals.
            rollout_batch = batch.slice(0, num_prompts_to_generate)
            # Rows are stamped at yield time, so this slice carries its
            # original stream ordinals; record them for the outstanding set.
            dispatched_task_indices = _stamped_task_indices(rollout_batch)
            if use_nemo_gym and self._deduplicate_multimodal_data:
                attach_initial_nemo_gym_image_payloads(
                    rollout_batch,
                    self.processor,
                    env_config=self.master_config.env,
                )
            repeated_batch = rollout_batch.repeat_interleave(
                num_generations,
                share_immutable_media=self._deduplicate_multimodal_data,
            )
            print_multimodal_payload_metrics(
                collect_multimodal_payload_metrics(
                    repeated_batch,
                    "prompt_repeat_async",
                    enabled=self._debug_payload_metrics,
                )
            )

            def _run_rollout_batch() -> None:
                asyncio.run(
                    self._run_rollout_batch_worker(
                        repeated_batch=repeated_batch,
                        generation_weight_version=generation_weight_version,
                        target_weight_version=reserved_target,
                        num_generations=num_generations,
                        use_nemo_gym=use_nemo_gym,
                        dispatched_task_indices=dispatched_task_indices,
                    )
                )

            worker = _threading.Thread(target=_run_rollout_batch, daemon=True)
            try:
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                if dispatched_task_indices:
                    with self._outstanding_lock:
                        self._outstanding_task_indices.update(dispatched_task_indices)
                worker.start()
                worker_started = True
            except Exception:
                with self._threads_lock:
                    self._inflight_threads.discard(worker)
                if dispatched_task_indices:
                    with self._outstanding_lock:
                        self._outstanding_task_indices.difference_update(
                            dispatched_task_indices
                        )
                raise

            backend = "NeMo-Gym" if use_nemo_gym else "native"
            print(
                f"📊 Started one {backend} batch worker for "
                f"{num_prompts_to_generate} prompt groups at "
                f"target_weight={reserved_target}"
            )

            self._cleanup_finished_threads()
            return leftover

        except Exception as e:
            if target_weight is not None and not worker_started:
                self._release_target(target_weight)
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()
            # No worker was started, so nothing in `batch` was consumed: hand
            # the whole batch back rather than only the gap-fill tail.
            return batch if not worker_started else leftover

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def check_health(self) -> None:
        """Raise the stored fatal worker error, if any.

        Called by the trainer between sampling iterations. When a generation
        worker has recorded a fatal failure (consecutive count exceeded
        max_generation_failures), this raises it so the training job dies
        instead of stalling on an empty replay buffer. Safe to call
        repeatedly: returns silently when no fatal error is set, and raises
        every time once one is.
        """
        with self._failure_lock:
            error_message = self._fatal_error_message
        if error_message is not None:
            raise RuntimeError(error_message)

    def pause(self) -> None:
        """Pause trajectory collection."""
        self._manual_pause_cleared.clear()  # Signal collection to pause
        print("Trajectory collection paused")

    def resume(self) -> None:
        """Resume trajectory collection."""
        self._manual_pause_cleared.set()  # Signal collection to resume
        print("Trajectory collection resumed")

    def prepare_for_refit(self) -> None:
        """Pause new generation starts and optionally wait for pending generations.

        For backends with an async engine in-flight weight updates allows ongoing generations
        to continue with their current KV caches while weights are updated.
        This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Check if we're using async engine
        generation_cfg = self.master_config.policy.get("generation", {})
        backend = generation_cfg.get("backend", "")
        if backend == "vllm":
            is_async_engine = generation_cfg.get("vllm_cfg", {}).get(
                "async_engine", False
            )
        elif backend == "megatron":
            is_async_engine = True
        elif backend == "trtllm":
            assert generation_cfg.get("trtllm_cfg", {}).get("async_engine", False), (
                "TRT-LLM backend requires trtllm_cfg.async_engine=true; the "
                "synchronous engine path (async_engine=false) is no longer supported."
            )
            is_async_engine = True
        elif backend == "dynamo":
            # Dynamo's native layerwise reload temporarily materializes model
            # parameters while the NCCL update is in progress.  It is not safe
            # to execute an already-issued vLLM request concurrently with that
            # reload (in particular for NemotronH/Mamba parameters), even when
            # the update route accepts allow_unpaused=True.  Stop new trajectory
            # starts above and drain every active trajectory before refitting.
            is_async_engine = False
        else:
            is_async_engine = False
        in_flight_weight_updates = self.async_config.in_flight_weight_updates

        if is_async_engine and in_flight_weight_updates:
            # async engines support in-flight weight updates
            # Ongoing generations will continue with their current KV caches
            # New generations (after weight update) will use the updated weights
            print(
                f"🚀 Using {backend} in-flight weight update - skipping wait for pending generations"
            )
            print(
                f"   {len(self._inflight_threads)} ongoing generations will complete with current weights"
            )
        else:
            # For non-async engines, wait for all pending generations to complete
            print(
                "⏸️ Non-async engine: waiting for all pending generations to complete..."
            )
            self.wait_for_pending_generations()

        elapsed = time.time() - start_time
        print(f"✅ Ready for refit (took {elapsed:.2f}s)")

    def resume_after_refit(self) -> None:
        """Resume new generation starts after refit is complete."""
        print("🔄 Resuming generation starts after refit")

        # Invalidate&recompute vLLM caches after the weight updates (in-flight or not) if
        # recompute_kv_cache_after_weight_updates is True (AREAL-style implementation).
        # Otherwise, keep using the stale KV caches (Magistral-style implementation).
        if self.async_config.recompute_kv_cache_after_weight_updates:
            try:
                print(
                    "🔄 Invalidating generation backend KV caches after weight update"
                )
                invalidated = self.policy_generation.invalidate_kv_cache()
                if invalidated:
                    print(
                        "✅ Invalidated generation backend KV caches after weight update"
                    )
                else:
                    print(
                        "⚠️ KV cache invalidation not supported or only partially applied by the generation backend"
                    )
            except Exception as e:
                print(f"⚠️ Failed to invalidate generation backend KV caches: {e}")
                if (
                    "generation" in self.master_config.policy
                    and self.master_config.policy["generation"]["backend"] == "dynamo"
                ):
                    raise RuntimeError(
                        "Managed Dynamo KV cache invalidation failed after refit"
                    ) from e
            finally:
                self._refit_pause_cleared.set()
        else:
            self._refit_pause_cleared.set()

    def wait_for_pending_generations(self) -> None:
        """Wait for all in-flight generation threads to complete."""
        start_time = time.time()

        while True:
            with self._threads_lock:
                finished = {t for t in self._inflight_threads if not t.is_alive()}
                for t in finished:
                    self._inflight_threads.remove(t)

                pending_count = len(self._inflight_threads)

            if pending_count == 0:
                print("✅ All generation threads completed")
                break

            elapsed = time.time() - start_time
            print(
                f"⏳ Waiting for {pending_count} pending generation threads... ({elapsed:.1f}s elapsed)"
            )
            time.sleep(0.5)

    def get_dataloader_state(self) -> dict:
        """Get the current dataloader state for checkpointing."""
        if self.dataloader is not None:
            return self.dataloader.state_dict()
        return {}

    def get_efficiency_metrics(self) -> dict[str, float]:
        """Return accumulated efficiency metrics (sum of durations per category).

        Called by the driver process each step to merge collector-side metrics.
        """
        return cast(
            dict[str, float],
            self._efficiency_timer.get_timing_metrics(reduction_op="sum"),
        )

    async def drain_payload_metrics(self) -> dict[str, int | float]:
        """Close one drain-to-drain collector/Gym telemetry interval.

        Rollout collection is concurrent with training, so the interval is not
        claimed to own the sampled training batch. Call-normalized metrics make
        intervals comparable even when their background transfer counts differ.
        """
        return drain_multimodal_payload_metrics()

    def _build_rollouts_state(self, *, include_pending: bool) -> dict[str, Any]:
        """Build the rollout-state mapping. Caller must hold ``_pending_lock``.

        The mapping carries the next task index and, when requested and
        present, the pending prompt batch under ``PENDING_PROMPTS_KEY``.
        Serializing the remainder keeps yielded prompts recoverable on
        live-cursor checkpoints: the dataloader cursor has already advanced
        past them, so a checkpoint that dropped them would skip those prompts
        for the rest of the run.
        """
        state: dict[str, Any] = {
            NEXT_NEMO_GYM_TASK_INDEX_KEY: self._next_nemo_gym_task_index
        }
        if include_pending and self._pending_batch is not None:
            state[PENDING_PROMPTS_KEY] = self._pending_batch
        return state

    def get_rollouts_state(self) -> dict[str, Any]:
        """Get collector-side rollout state (always including any pending batch).

        The driver's save path reads this through
        :meth:`get_checkpoint_state`, which pairs it with the dataloader
        snapshot under one lock; this standalone accessor serves tests and
        diagnostics.
        """
        with self._pending_lock:
            return self._build_rollouts_state(include_pending=True)

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _release_target(self, target_weight_version: int) -> None:
        """Release the reservation owned by a completed batch worker."""
        with self._generation_check_lock:
            if target_weight_version in self._generating_targets:
                self._generating_targets.discard(target_weight_version)
                print(
                    f"🧹 Released reservation for target weight {target_weight_version}"
                )

    def _compute_teacher_logprobs(
        self,
        input_ids: torch.Tensor,
        agent_refs: list[dict[str, Any]],
        input_lengths: Optional[torch.Tensor] = None,
        multimodal_data: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, float]:
        """Compute teacher logprobs for non-colocated teachers.

        Groups samples by teacher, fans out in parallel, stitches results.

        Args:
            input_ids: [B, S] tokenized input tensor
            agent_refs: list of B agent reference dicts
            input_lengths: [B] per-sample lengths (required for sequence packing)
            multimodal_data: batch-level multimodal inputs, row-aligned with
                ``input_ids`` and sliced per teacher

        Returns:
            ([B, S] teacher logprobs tensor, total_time_seconds)
        """
        opd_cfg = self.on_policy_distillation_cfg
        teacher_model_by_agent_name = opd_cfg.get("teacher_model_by_agent_name", {})
        default_teacher_alias = opd_cfg.get("default_teacher_alias")
        strict = opd_cfg.get("strict_agent_name_match", False)

        # Resolve each sample's agent -> the teacher alias it should be distilled
        # from: the agent name is looked up in teacher_model_by_agent_name; unmapped
        # agents fall back to default_teacher_alias (or raise if strict_agent_name_match).
        # Returns one alias per sample, index-aligned with agent_refs.
        reference_aliases = resolve_reference_aliases(
            agent_refs,
            teacher_model_by_agent_name,
            default_teacher_alias=default_teacher_alias,
            strict_agent_name_match=strict,
        )

        # Map aliases to actual group keys via deduplication mapping
        group_keys = [self.alias_to_group_alias.get(a, a) for a in reference_aliases]

        # Group sample indices by teacher group
        group_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, gk in enumerate(group_keys):
            group_to_indices[gk].append(i)

        B, S = input_ids.shape
        result = torch.zeros(B, S, dtype=torch.float32)
        if (
            not group_to_indices
        ):  # 0-sample batch: nothing to route (avoid max_workers=0)
            return result, 0.0

        def _get_logprobs_for_group(group_key, indices):
            twg = self.teacher_worker_groups[group_key]
            sub_input_ids = input_ids[indices]
            sub_lengths = input_lengths[indices] if input_lengths is not None else None
            row_indices = list(indices)

            # Pad batch to multiple of dp_size (required for DP sharding)
            dp_size = twg.sharding_annotations.get_axis_size("data_parallel")
            actual_batch_size = sub_input_ids.shape[0]
            remainder = actual_batch_size % dp_size
            if remainder != 0:
                pad_count = dp_size - remainder
                # Repeat last row to fill — can't slice [:pad_count] when
                # actual_batch_size < pad_count (e.g., 1 sample, dp_size=4)
                pad_rows = sub_input_ids[-1:].expand(pad_count, -1)
                sub_input_ids = torch.cat([sub_input_ids, pad_rows], dim=0)
                if sub_lengths is not None:
                    sub_lengths = torch.cat(
                        [sub_lengths, sub_lengths[-1:].expand(pad_count)], dim=0
                    )
                row_indices.extend([row_indices[-1]] * pad_count)

            sub_data = BatchedDataDict({"input_ids": sub_input_ids})
            if sub_lengths is not None:
                sub_data["input_lengths"] = sub_lengths
            if multimodal_data:
                selected_multimodal = BatchedDataDict(multimodal_data).select_indices(
                    row_indices
                )
                sub_data.update(
                    {
                        key: value
                        for key, value in selected_multimodal.items()
                        if value is not None
                        and not (
                            isinstance(value, PackedTensor)
                            and not any(value.logical_segment_counts_by_row())
                        )
                    }
                )

            # Serialize calls per teacher to prevent NCCL collective desync
            t_lock_start = time.time()
            with self._teacher_locks[group_key]:
                t_inference_start = time.time()
                logprobs_result = twg.get_logprobs(sub_data)
            t_done = time.time()
            lock_wait = t_inference_start - t_lock_start
            inference_time = t_done - t_inference_start
            print(
                f"[teacher_logprob] group={group_key} samples={actual_batch_size} "
                f"lock_wait={lock_wait:.2f}s inference={inference_time:.2f}s"
            )
            logprobs = logprobs_result["reference_logprobs"]

            # Trim DP padding
            logprobs = logprobs[:actual_batch_size]

            return indices, logprobs

        # Fan out to teachers in parallel
        t_total_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(group_to_indices)
        ) as executor:
            futures = {
                executor.submit(_get_logprobs_for_group, gk, idxs): gk
                for gk, idxs in group_to_indices.items()
            }
            for future in concurrent.futures.as_completed(futures):
                indices, logprobs = future.result()
                result[indices] = logprobs
        total_time = time.time() - t_total_start
        print(
            f"[teacher_logprob] total={total_time:.2f}s for {B} samples across {len(group_to_indices)} teacher(s)"
        )

        return result, total_time

    async def _iter_rollout_groups(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        num_generations: int,
        use_nemo_gym: bool,
        task_index_to_group_index: dict[int, int],
    ) -> AsyncGenerator[RolloutGroupResult, None]:
        """Yield prompt groups from either backend through one result type."""
        if use_nemo_gym:
            # Import here to keep the NeMo-Gym dependency local to its backend.
            from nemo_rl.experience.rollouts import (
                get_nemo_gym_thinking_tags,
                run_async_nemo_gym_rollout,
                should_mask_flagged_samples,
            )

            # NeMo-Gym owns stop criteria. Configuration fills the policy EOS token
            # automatically, so clear the derived values before validation.
            generation_config: GenerationConfig = {
                **self.master_config.policy["generation"],
                "stop_token_ids": None,
                "stop_strings": None,
            }
            async for rollout_result in run_async_nemo_gym_rollout(
                policy_generation=self.policy_generation,
                input_batch=repeated_batch,
                tokenizer=self.tokenizer,
                task_to_env=self.task_to_env,
                max_seq_len=self.master_config.policy["max_total_sequence_length"],
                generation_config=generation_config,
                num_generations=num_generations,
                log_full_result_tables=should_log_nemo_gym_full_result_tables(
                    wandb_enabled=self.master_config.logger["wandb_enabled"],
                    wandb_config=self.master_config.logger["wandb"],
                ),
                max_rollout_turns=None,
                greedy=False,
                effort_config=(
                    _get_effort_config(self.master_config)
                    if isinstance(self.master_config, GRPOMasterConfig)
                    else None
                ),
                reward_penalty_config=self.master_config.reward_penalties,
                thinking_tags=get_nemo_gym_thinking_tags(self.master_config.env),
                mask_env_flagged_samples=should_mask_flagged_samples(
                    self.master_config.env
                ),
                deduplicate_multimodal_data=self._deduplicate_multimodal_data,
                debug_payload_metrics=self._debug_payload_metrics,
            ):
                task_index = rollout_result.task_index
                if task_index is None:
                    raise ValueError("NeMo-Gym prompt group is missing _ng_task_index")
                task_index = int(task_index)
                if task_index not in task_index_to_group_index:
                    raise ValueError(f"Unexpected _ng_task_index {task_index}")
                yield RolloutGroupResult(
                    group_index=task_index_to_group_index[task_index],
                    final_batch=rollout_result.final_batch,
                    rollout_metrics=rollout_result.rollout_metrics,
                    task_index=task_index,
                )
            return

        async for rollout_result in run_async_multi_turn_rollout_groups(
            policy_generation=self.policy_generation,
            input_batch=repeated_batch,
            tokenizer=self.tokenizer,
            task_to_env=self.task_to_env,
            max_seq_len=self.master_config.policy["max_total_sequence_length"],
            num_generations=num_generations,
            max_rollout_turns=self._max_rollout_turns,
            greedy=False,
            deduplicate_multimodal_data=self._deduplicate_multimodal_data,
        ):
            yield rollout_result

    async def _run_rollout_batch_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        num_generations: int,
        use_nemo_gym: bool,
        dispatched_task_indices: Optional[list[int]] = None,
    ) -> None:
        """Own one target reservation while collecting its rollout batch."""
        worker_start = time.perf_counter()
        wake_generation_limits_after_cleanup = False
        try:
            await self._collect_rollout_batch(
                repeated_batch=repeated_batch,
                generation_weight_version=generation_weight_version,
                target_weight_version=target_weight_version,
                num_generations=num_generations,
                use_nemo_gym=use_nemo_gym,
            )
            with self._failure_lock:
                if self._fatal_error_message is None:
                    self._failure_count = 0
        except Exception as error:
            if not self.running:
                return

            self._efficiency_timer.record(
                "wasted/failed_trajectory", time.perf_counter() - worker_start
            )
            backend = "NeMo-Gym" if use_nemo_gym else "native"
            import traceback

            failure_traceback = traceback.format_exc()
            with self._failure_lock:
                self._failure_count += 1
                failure_count = self._failure_count
                failure_limit = self._max_generation_failures
                is_fatal = failure_count > failure_limit
                if is_fatal and self._fatal_error_message is None:
                    self._fatal_error_message = (
                        "AsyncTrajectoryCollector aborting: "
                        f"{failure_count} batch-worker failure(s) exceeded "
                        f"max_generation_failures={failure_limit}. "
                        f"Last failure in {backend} batch worker for "
                        f"generation_weight={generation_weight_version}, "
                        f"target_weight={target_weight_version}: {error!r}\n"
                        f"Worker traceback:\n{failure_traceback}"
                    )
            wake_generation_limits_after_cleanup = True
            print(
                f"[AsyncTrajectoryCollector] {backend} batch worker FAILED "
                f"(failure {failure_count}, tolerating {failure_limit}) "
                f"generation_weight={generation_weight_version} "
                f"target_weight={target_weight_version}\n{failure_traceback}",
                flush=True,
            )
            if is_fatal:
                print(
                    f"[AsyncTrajectoryCollector] FATAL: failure count "
                    f"{failure_count} exceeds threshold {failure_limit}; trainer "
                    "will be notified on the next check_health() call.",
                    flush=True,
                )
        finally:
            if dispatched_task_indices:
                # This worker's unbuffered ordinals are a permanent loss and
                # must not keep holding the checkpoint cut down.
                with self._outstanding_lock:
                    self._outstanding_task_indices.difference_update(
                        dispatched_task_indices
                    )
            self._release_target(target_weight_version)
            with self._threads_lock:
                self._inflight_threads.discard(_threading.current_thread())
            if wake_generation_limits_after_cleanup:
                self._generation_limit_cleared.set()

    @staticmethod
    def _build_task_index_map(
        repeated_batch: BatchedDataDict[DatumSpec],
        num_generations: int,
    ) -> dict[int, int]:
        """Map each Gym task index to its repeated prompt-group position."""
        task_index_to_group_index: dict[int, int] = {}
        prompt_group_count = repeated_batch.size // num_generations
        for group_index in range(prompt_group_count):
            start = group_index * num_generations
            rows = repeated_batch["extra_env_info"][start : start + num_generations]
            raw_task_indices = [row.get(NEMO_GYM_TASK_INDEX_KEY) for row in rows]
            if any(task_index is None for task_index in raw_task_indices):
                raise ValueError(
                    "Every NeMo-Gym row must include _ng_task_index, got "
                    f"{raw_task_indices} for group {group_index}"
                )

            task_indices = {int(task_index) for task_index in raw_task_indices}
            if len(task_indices) != 1:
                raise ValueError(
                    "Expected one _ng_task_index per repeated prompt group, got "
                    f"{sorted(task_indices)} for group {group_index}"
                )
            task_index = task_indices.pop()
            if task_index in task_index_to_group_index:
                raise ValueError(f"Duplicate _ng_task_index {task_index}")
            task_index_to_group_index[task_index] = group_index

        return task_index_to_group_index

    async def _enqueue_rollout_group(
        self,
        rollout_result: RolloutGroupResult,
        generation_weight_version: int,
        target_weight_version: int,
        expected_prompt_groups: int,
        buffered_group_indices: set[int],
        collection_started_at: float,
        input_task_index: Optional[int] = None,
    ) -> None:
        """Push one prompt group to the replay buffer with bounded backoff."""
        final_batch_cpu = rollout_result.final_batch.to("cpu")
        rollout_metrics = rollout_result.rollout_metrics

        # Teacher inference is blocking. Keep it off this worker's event loop so
        # other completed prompt groups can continue moving toward the buffer.
        if self._has_distillation_teachers and "agent_ref" in final_batch_cpu:
            agent_refs = final_batch_cpu["agent_ref"]
            if isinstance(agent_refs, list):
                from nemo_rl.data.llm_message_utils import (
                    batched_message_log_to_flat_message,
                )

                flat_for_teacher, teacher_input_lengths = (
                    batched_message_log_to_flat_message(
                        final_batch_cpu["message_log"],
                        pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=self._teacher_seq_pad_multiple,
                    )
                )
                teacher_logprobs, teacher_logprob_time = await asyncio.to_thread(
                    self._compute_teacher_logprobs,
                    flat_for_teacher["token_ids"],
                    agent_refs,
                    input_lengths=teacher_input_lengths,
                    multimodal_data=flat_for_teacher.get_multimodal_dict(
                        as_tensors=False
                    ),
                )
                # Keep the tensor inside the batch so replay-buffer collation can
                # pad variable-length prompt groups correctly.
                final_batch_cpu["teacher_reference_logprobs"] = teacher_logprobs
                rollout_metrics = dict(rollout_metrics)
                rollout_metrics["teacher_logprob_time"] = teacher_logprob_time

        rollout_metrics = dict(rollout_metrics)
        rollout_metrics["trajectory_duration_s"] = (
            time.perf_counter() - collection_started_at
        )
        trajectory_group = {
            "batch": final_batch_cpu,
            "rollout_metrics": rollout_metrics,
            "timestamp": time.time(),
        }
        group_task_index = rollout_result.task_index
        if group_task_index is None:
            # Native groups carry their ordinal on the stamped *input* rows,
            # captured before the rollout ran; recording it on the group lets
            # a frontier restore report which prompts the retained buffer
            # already covers.
            group_task_index = input_task_index
        if group_task_index is not None:
            trajectory_group[NEMO_GYM_TASK_INDEX_KEY] = group_task_index
        backoff_delay = 0.01
        backoff_started_at: float | None = None
        try:
            while self.running:
                # Every retry is a distinct Ray submission of the full payload.
                print_multimodal_payload_metrics(
                    collect_multimodal_payload_metrics(
                        (
                            trajectory_group,
                            generation_weight_version,
                            target_weight_version,
                        ),
                        "replay_push",
                        enabled=self._debug_payload_metrics,
                    )
                )
                status = await self.replay_buffer.add.remote(
                    trajectory_group,
                    generation_weight_version,
                    target_weight_version,
                )
                if status == "success":
                    buffered_group_indices.add(rollout_result.group_index)
                    if group_task_index is not None:
                        # The cut now covers this ordinal via the buffer's
                        # held-ordinal report instead of the outstanding set.
                        with self._outstanding_lock:
                            self._outstanding_task_indices.discard(
                                int(group_task_index)
                            )
                    group_description = f"group_index={rollout_result.group_index}"
                    if rollout_result.task_index is not None:
                        group_description = (
                            f"_ng_task_index={rollout_result.task_index}"
                        )
                    print(
                        "📦 Buffered prompt group "
                        f"({group_description}, target_weight={target_weight_version}) "
                        f"[{len(buffered_group_indices)}/{expected_prompt_groups} buffered]"
                    )
                    return
                if status != "full":
                    raise RuntimeError(
                        f"Replay buffer returned unexpected add status {status!r}"
                    )

                if backoff_started_at is None:
                    backoff_started_at = time.perf_counter()
                await asyncio.sleep(
                    min(backoff_delay, _REPLAY_BUFFER_MAX_BACKOFF_SECONDS)
                )
                backoff_delay *= 1.5

            raise RuntimeError("Trajectory collection stopped before enqueue completed")
        finally:
            if backoff_started_at is not None:
                self._efficiency_timer.record(
                    "idle/buffer_full_backoff",
                    time.perf_counter() - backoff_started_at,
                )

    async def _collect_rollout_batch(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        num_generations: int,
        use_nemo_gym: bool,
    ) -> None:
        """Run one backend batch and enqueue every completed prompt group."""
        collection_started_at = time.perf_counter()
        if num_generations <= 0 or repeated_batch.size % num_generations != 0:
            raise ValueError(
                "Rollout batch size must be divisible by a positive num_generations"
            )
        expected_prompt_groups = repeated_batch.size // num_generations
        expected_group_indices = set(range(expected_prompt_groups))
        task_index_to_group_index = (
            self._build_task_index_map(repeated_batch, num_generations)
            if use_nemo_gym
            else {}
        )
        # Capture each group's ordinal from the *input* rows up front: the
        # rollout may replace extra_env_info wholesale (multi-turn envs can
        # build fresh metadata dicts), so the output rows are not a reliable
        # carrier for the stamp.
        input_rows = repeated_batch.get("extra_env_info")
        group_input_task_indices: list[Optional[int]] = [
            _unanimous_task_index(
                input_rows[
                    group_index * num_generations : (group_index + 1) * num_generations
                ]
                if isinstance(input_rows, list)
                else []
            )
            for group_index in range(expected_prompt_groups)
        ]
        buffered_group_indices: set[int] = set()
        last_error: Exception | None = None
        max_attempts = 1 + (_MAX_NEMO_GYM_STREAM_RETRIES if use_nemo_gym else 0)
        for attempt in range(1, max_attempts + 1):
            push_tasks: list[asyncio.Task[None]] = []
            scheduled_group_indices: set[int] = set()
            stream_error: Exception | None = None
            try:
                async for rollout_result in self._iter_rollout_groups(
                    repeated_batch=repeated_batch,
                    num_generations=num_generations,
                    use_nemo_gym=use_nemo_gym,
                    task_index_to_group_index=task_index_to_group_index,
                ):
                    group_index = rollout_result.group_index
                    if group_index not in expected_group_indices:
                        raise ValueError(f"Unexpected prompt group index {group_index}")
                    if rollout_result.final_batch.size != num_generations:
                        raise ValueError(
                            f"Prompt group {group_index} contains "
                            f"{rollout_result.final_batch.size} rollouts; expected "
                            f"{num_generations}"
                        )
                    if group_index in buffered_group_indices:
                        continue
                    if group_index in scheduled_group_indices:
                        raise ValueError(
                            f"Rollout stream yielded prompt group {group_index} twice"
                        )
                    scheduled_group_indices.add(group_index)
                    push_tasks.append(
                        asyncio.create_task(
                            self._enqueue_rollout_group(
                                rollout_result=rollout_result,
                                generation_weight_version=generation_weight_version,
                                target_weight_version=target_weight_version,
                                expected_prompt_groups=expected_prompt_groups,
                                buffered_group_indices=buffered_group_indices,
                                collection_started_at=collection_started_at,
                                input_task_index=group_input_task_indices[group_index],
                            )
                        )
                    )
            except Exception as error:
                stream_error = error

            push_results = await asyncio.gather(*push_tasks, return_exceptions=True)
            push_errors = [
                result for result in push_results if isinstance(result, Exception)
            ]
            pending_group_indices = expected_group_indices - buffered_group_indices
            if not pending_group_indices:
                return

            last_error = stream_error or (push_errors[0] if push_errors else None)
            if last_error is None:
                last_error = RuntimeError(
                    "Rollout stream ended before yielding prompt groups "
                    f"{sorted(pending_group_indices)}"
                )
            if attempt == max_attempts or not self.running:
                break

            retry_delay = _NEMO_GYM_RETRY_DELAY_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                "❌ NeMo-Gym batch did not complete prompt groups "
                f"{sorted(pending_group_indices)}; retrying in "
                f"{retry_delay:.1f}s "
                f"(attempt {attempt + 1}/{max_attempts})"
            )
            await asyncio.sleep(retry_delay)

        batch_error = RuntimeError(
            "Rollout batch failed to buffer prompt groups "
            f"{sorted(expected_group_indices - buffered_group_indices)}"
        )
        if last_error is not None:
            raise batch_error from last_error
        raise batch_error
