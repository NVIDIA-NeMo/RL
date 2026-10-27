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

"""Rollout checkpointing that saves the data plane (TQ) beside controller state.

Writes periodic rollout snapshots anchored to the latest durable trainer
checkpoint, and the TQ part of every trainer checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import shutil
import time
import warnings
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    DataPlaneCheckpointBarrier,
    DataPlaneCheckpointMetadata,
    DataPlaneMutationCut,
    TQReplayMetadataState,
)
from nemo_rl.algorithms.rollout_checkpointing.controller_state import (
    ControllerRolloutState,
    ControllerSidecars,
    write_controller_sidecars,
)
from nemo_rl.algorithms.rollout_checkpointing.interface import (
    RolloutCheckpointing,
    RolloutCheckpointingHost,
)
from nemo_rl.algorithms.rollout_checkpointing.inventory import (
    capture_reward_settings,
    validate_replay_inventory,
    validate_rollout_recovery_inventory,
)
from nemo_rl.algorithms.metric_utils import (
    ROLLOUT_CHECKPOINT_ATTEMPT_DURATION_KEY,
    ROLLOUT_CHECKPOINT_ATTEMPT_OUTCOMES,
    ROLLOUT_CHECKPOINT_ATTEMPT_REASONS,
    ROLLOUT_CHECKPOINT_OUTCOME_PREFIX,
    ROLLOUT_CHECKPOINT_SECONDS_SINCE_LAST_SUCCESS_KEY,
    RolloutCheckpointAttemptOutcome,
    RolloutCheckpointAttemptReason,
    rollout_checkpoint_reason_key,
)
from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    ROLLOUT_SNAPSHOT_MANIFEST_FILENAME,
    ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
    BootstrapCompatibilityIdentity,
    RolloutSnapshotManifest,
    commit_snapshot,
    ensure_bootstrap_anchor,
    prepare_snapshot_paths,
    prune_bootstrap_snapshots,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.experience.environment_checkpoint import (
    EnvironmentCheckpoint,
    EnvironmentSnapshot,
)
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    build_rollout_recovery_state,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
    from nemo_rl.algorithms.opd import TQTeacherLogprobCoordinator
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.experience.rollout_manager import RolloutManager
    from nemo_rl.utils.checkpoint import CheckpointManager, PathLike
    from nemo_rl.utils.timer import TimeoutChecker


@dataclass(frozen=True)
class RolloutCheckpointSaveResult:
    """Outcome returned by one rollout checkpoint save attempt."""

    saved: bool
    reason: RolloutCheckpointAttemptReason


@dataclass(frozen=True)
class RolloutCheckpointCut:
    """Controller sidecars captured with one native TQ snapshot."""

    dataloader_state: dict[str, Any]
    sampler_dispatch_index: int
    replacement_reserve: list[DatumSpec]
    replay_metadata: Optional[TQReplayMetadataState]
    rollout_recovery_payload: Optional[bytes]
    rollout_recovery_group_count: Optional[int]
    replay_row_count: int
    staging_row_count: int
    rolled_back_train_group_count: int
    mutation_version: int
    tq_save_seconds: float


@dataclass(frozen=True)
class _DataPlaneSave:
    """What saving TQ and the recovery ledger under one cut produced."""

    rollout_recovery_payload: Optional[bytes]
    rollout_recovery_group_count: Optional[int]
    staging_row_count: int
    tq_save_seconds: float


class DataPlaneRolloutCheckpointing(RolloutCheckpointing):
    """Save rollout state as a native TQ snapshot plus controller sidecars."""

    def __init__(
        self,
        *,
        host: RolloutCheckpointingHost,
        master_config: MasterConfig,
        dp_client: Any,
        partition_id: str,
        buffer: TQReplayBuffer,
        rollout_manager: RolloutManager,
        checkpointer: CheckpointManager,
        timeout: TimeoutChecker,
        data_plane_barrier: DataPlaneCheckpointBarrier,
        environment_checkpoint: EnvironmentCheckpoint,
        bootstrap_identity: Optional[BootstrapCompatibilityIdentity],
        teacher_coordinator: Optional[TQTeacherLogprobCoordinator],
    ) -> None:
        """Store the parts this component saves and coordinates with.

        Args:
            host: The controller, through the calls this component may make.
            master_config: The run's config.
            dp_client: The controller's data-plane client.
            partition_id: The partition that holds finished rollouts.
            buffer: The replay buffer, which owns the replay index.
            rollout_manager: Owns the recovery ledger.
            checkpointer: Owns the trainer checkpoint directory.
            timeout: The run's checkpoint deadline, shared with the train pump.
            data_plane_barrier: The barrier every TQ mutation and save goes through.
            environment_checkpoint: The environment backend's rollout state.
            bootstrap_identity: Identifies the run before its first trainer
                checkpoint; needed for snapshots taken before step 1.
            teacher_coordinator: Records which teacher tagged the rows, if any.
        """
        self._host = host
        self._master_config = master_config
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._buffer = buffer
        self._rollout_manager = rollout_manager
        self._checkpointer = checkpointer
        self._timeout = timeout
        self._data_plane_checkpoint_barrier = data_plane_barrier
        self._environment_checkpoint = environment_checkpoint
        self._bootstrap_identity = bootstrap_identity
        self._teacher_coordinator = teacher_coordinator

        # Full trainer checkpoints and lightweight rollout snapshots share one
        # namespace and must never publish concurrently.
        self._checkpoint_save_lock = asyncio.Lock()
        self._last_rollout_snapshot_mutation_version: Optional[int] = None
        self._last_missing_rollout_snapshot_anchor: Optional[tuple[int, int]] = None
        self._last_successful_rollout_checkpoint_time: Optional[float] = None
        self._rollout_checkpoint_stop_requested = asyncio.Event()
        # Set by a trainer checkpoint, which holds no environment state, so the
        # next rollout snapshot captures the environment soon after it.
        self._rollout_checkpoint_wakeup = asyncio.Event()
        # Narrow unsafe window after an optimizer mutates model state and before
        # SC publishes the matching TQ cleanup and trainer counters. Gradient
        # accumulation remains snapshot-safe because selected rows stay owned by
        # the replay buffer and are re-indexed in periodic snapshots.
        self._optimizer_commit_in_progress = False
        # Set whenever a step starts committing. A rollout snapshot clears it when
        # it starts and stops its environment phase once it is set, since the
        # snapshot can no longer be kept.
        self._optimizer_commit_started = asyncio.Event()

    # ── RolloutCheckpointing ───────────────────────────────────────────────

    def background_tasks(self) -> list[Coroutine[Any, Any, None]]:
        if (
            self._master_config.rollout_checkpointing.snapshot_attempt_interval_s
            is None
        ):
            return []
        return [self._rollout_checkpoint_pump()]

    @property
    def stop_requested(self) -> bool:
        return self._rollout_checkpoint_stop_requested.is_set()

    @contextlib.asynccontextmanager
    async def optimizer_commit(self, *, drain_snapshot: bool) -> AsyncIterator[None]:
        if drain_snapshot:
            async with self._checkpoint_save_lock:
                self._begin_optimizer_commit()
        else:
            self._begin_optimizer_commit()
        yield
        self._optimizer_commit_in_progress = False

    @contextlib.asynccontextmanager
    async def trainer_checkpoint(self) -> AsyncIterator[None]:
        async with self._checkpoint_save_lock:
            yield
            if self._environment_checkpoint.holds_rollout_state:
                # The rollout snapshot waits for this checkpoint to be durable, off
                # the training loop, and then captures the environment anchored to it.
                self._rollout_checkpoint_wakeup.set()

    async def save_with_trainer_checkpoint(
        self,
        cut: DataPlaneMutationCut,
        checkpoint_path: PathLike,
        state: ControllerRolloutState,
    ) -> ControllerSidecars:
        replay_metadata: Optional[TQReplayMetadataState] = None
        rollout_recovery_payload: Optional[bytes] = None
        if self._master_config.checkpointing.get("save_data_plane"):
            training_owned_groups = self._buffer.training_owned_replay_groups()
            if training_owned_groups:
                raise RuntimeError(
                    "full trainer checkpoint still owns streamed training rows: "
                    f"groups={[group['group_id'] for group in training_owned_groups]!r}"
                )
            if state.sampler_supports_buffer_checkpoint:
                replay_metadata = self._buffer.metadata_state_dict(
                    saved_capacity=self._master_config.async_rl.max_buffered_rollouts
                )
            data_plane_save = await self._save_data_plane_with_recovery_state(
                cut,
                checkpoint_path,
                state,
                replay_metadata=replay_metadata,
                save_recovery_state=state.rollout_recovery_enabled,
            )
            rollout_recovery_payload = data_plane_save.rollout_recovery_payload
            # A trainer checkpoint holds no environment state, so when the
            # environment keeps some it does not stand in for the next
            # rollout snapshot.
            if not self._environment_checkpoint.holds_rollout_state:
                self._last_rollout_snapshot_mutation_version = (
                    self._data_plane_checkpoint_barrier.mutation_version
                )
        return ControllerSidecars(
            dataloader_state=state.dataloader_state,
            replacement_reserve=state.replacement_reserve,
            replay_metadata=replay_metadata,
            rollout_recovery_payload=rollout_recovery_payload,
        )

    # ── data-plane save ────────────────────────────────────────────────────

    async def _validate_replay_inventory(
        self, replay_metadata: TQReplayMetadataState
    ) -> None:
        """Require the canonical TQ keys to match the SC replay index exactly."""
        await validate_replay_inventory(
            replay_metadata,
            dp_client=self._dp_client,
            partition_id=self._partition_id,
        )

    async def _save_data_plane_checkpoint(
        self,
        checkpoint_path: PathLike,
        *,
        train_steps: int,
        trainer_version: int,
        current_epoch: int,
        replay_metadata: Optional[TQReplayMetadataState] = None,
        rollout_recovery_payload_sha256: Optional[str] = None,
        rollout_recovery_group_count: Optional[int] = None,
    ) -> None:
        """Save a required TQ snapshot inside an SC checkpoint bundle.

        A sampler with replay-buffer recovery writes an authoritative native
        TQ snapshot bound to its metadata-only replay index by a digest. Other
        samplers retain shadow-mode snapshots until their recovery contract is
        defined. Failures propagate so a finalized bundle never silently omits
        the advertised data-plane component.
        """
        checkpoint_dir = os.path.join(
            checkpoint_path,
            DATA_PLANE_CHECKPOINT_DIR,
        )
        metadata: DataPlaneCheckpointMetadata = {
            "data_plane_checkpoint_schema_version": (
                DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
            ),
            "single_controller_train_steps": train_steps,
            "single_controller_trainer_version": trainer_version,
            "single_controller_epoch": current_epoch,
            "partition_id": self._partition_id,
            "sampler_name": self._master_config.async_rl.sampler.name,
            "mode": "authoritative" if replay_metadata is not None else "shadow",
        }
        if replay_metadata is not None:
            metadata["replay_metadata_schema_version"] = (
                REPLAY_BUFFER_METADATA_SCHEMA_VERSION
            )
            metadata["replay_manifest_digest"] = replay_metadata["manifest_digest"]
            metadata["replay_group_count"] = len(replay_metadata["groups"])
        if rollout_recovery_payload_sha256 is not None:
            if rollout_recovery_group_count is None:
                raise ValueError("rollout recovery payload hash requires a group count")
            metadata["rollout_recovery_schema_version"] = (
                ROLLOUT_RECOVERY_SCHEMA_VERSION
            )
            metadata["rollout_recovery_payload_sha256"] = (
                rollout_recovery_payload_sha256
            )
            metadata["rollout_recovery_group_count"] = rollout_recovery_group_count
        elif rollout_recovery_group_count is not None:
            raise ValueError("rollout recovery group count requires a payload hash")
        if self._teacher_coordinator is not None:
            # The rows in this checkpoint are tagged with a teacher_index each;
            # record what those indices meant so a restore can reject a config
            # that would renumber them.
            teacher_checkpoints = (
                self._teacher_coordinator.teacher_checkpoints_by_index()
            )
            if teacher_checkpoints is not None:
                metadata["opd_full_teacher_checkpoints"] = teacher_checkpoints
        started = time.monotonic()
        print(f"data-plane checkpoint save started: {checkpoint_dir}", flush=True)
        try:
            await call_data_plane(
                self._dp_client,
                "save_checkpoint",
                offload_sync=True,
                checkpoint_dir=checkpoint_dir,
                metadata=metadata,
            )
        except Exception as error:
            print(
                "data-plane checkpoint save failed: "
                f"{checkpoint_dir} ({type(error).__name__}: {error})",
                flush=True,
            )
            raise
        print(
            "data-plane checkpoint save completed: "
            f"{checkpoint_dir} ({time.monotonic() - started:.2f}s)",
            flush=True,
        )

    async def _save_data_plane_with_recovery_state(
        self,
        cut: DataPlaneMutationCut,
        checkpoint_path: PathLike,
        state: ControllerRolloutState,
        *,
        replay_metadata: Optional[TQReplayMetadataState],
        save_recovery_state: bool,
    ) -> _DataPlaneSave:
        """Save TQ, with the recovery ledger it matches, under one exclusive cut.

        Trainer checkpoints and rollout snapshots both save this way. The
        ledger leaves out groups already in ``replay_metadata``, so no group
        has two durable owners.
        """
        if replay_metadata is not None:
            await self._validate_replay_inventory(replay_metadata)

        recovery_payload: Optional[bytes] = None
        recovery_digest: Optional[str] = None
        recovery_group_count: Optional[int] = None
        if save_recovery_state:
            canonical_group_ids = (
                {group["group_id"] for group in replay_metadata["groups"]}
                if replay_metadata is not None
                else set()
            )
            recovery_state = build_rollout_recovery_state(
                self._rollout_manager.recovery_ledger,
                batch_shortfall=state.batch_shortfall,
                sampler_stamps_target_steps=state.sampler_stamps_target_steps,
                finalizer_metrics_by_group=state.finalizer_metrics_by_group,
                finalizer_rewards_by_group=state.finalizer_rewards_by_group,
                reward_settings=capture_reward_settings(self._master_config),
                canonical_group_ids=canonical_group_ids,
            )
            payload_buffer = io.BytesIO()
            await asyncio.to_thread(torch.save, recovery_state, payload_buffer)
            recovery_payload = payload_buffer.getvalue()
            recovery_digest = hashlib.sha256(recovery_payload).hexdigest()
            recovery_group_count = len(recovery_state["groups"])

        staging_row_count = 0
        if self._master_config.token_capture.enabled:
            staging_row_count = await validate_rollout_recovery_inventory(
                cut,
                dp_client=self._dp_client,
                recovery_ledger=self._rollout_manager.recovery_ledger,
                staging_partition=self._master_config.token_capture.staging_partition,
                replay_metadata=replay_metadata,
                clear_unreferenced=False,
            )
        tq_save_started = time.monotonic()
        await self._save_data_plane_checkpoint(
            checkpoint_path,
            train_steps=state.train_steps,
            trainer_version=state.trainer_version,
            current_epoch=state.current_epoch,
            replay_metadata=replay_metadata,
            rollout_recovery_payload_sha256=recovery_digest,
            rollout_recovery_group_count=recovery_group_count,
        )
        return _DataPlaneSave(
            rollout_recovery_payload=recovery_payload,
            rollout_recovery_group_count=recovery_group_count,
            staging_row_count=staging_row_count,
            tq_save_seconds=time.monotonic() - tq_save_started,
        )

    # ── rollout snapshots ──────────────────────────────────────────────────

    async def _capture_rollout_checkpoint_cut(
        self,
        cut: DataPlaneMutationCut,
        checkpoint_path: PathLike,
    ) -> RolloutCheckpointCut:
        """Save TQ and capture matching restart state under the barrier.

        Groups selected by an unfinished streamed step are absent from the live
        replay index but remain in TQ. Re-index them only in this persisted cut;
        the live trainer keeps accumulating gradients without modification.
        """
        cut.require_live()
        state = self._host.rollout_state()
        training_owned_groups = self._buffer.training_owned_replay_groups()
        replay_metadata = self._buffer.metadata_state_dict(
            saved_capacity=self._master_config.async_rl.max_buffered_rollouts,
            additional_groups=training_owned_groups,
        )
        replay_row_count = sum(
            len(group["meta"].sample_ids) for group in replay_metadata["groups"]
        )
        data_plane_save = await self._save_data_plane_with_recovery_state(
            cut,
            checkpoint_path,
            state,
            replay_metadata=replay_metadata,
            save_recovery_state=True,
        )
        return RolloutCheckpointCut(
            dataloader_state=state.dataloader_state,
            sampler_dispatch_index=state.sampler_dispatch_index,
            replacement_reserve=state.replacement_reserve,
            replay_metadata=replay_metadata,
            rollout_recovery_payload=data_plane_save.rollout_recovery_payload,
            rollout_recovery_group_count=data_plane_save.rollout_recovery_group_count,
            replay_row_count=replay_row_count,
            staging_row_count=data_plane_save.staging_row_count,
            rolled_back_train_group_count=len(training_owned_groups),
            mutation_version=self._data_plane_checkpoint_barrier.mutation_version,
            tq_save_seconds=data_plane_save.tq_save_seconds,
        )

    async def _write_rollout_checkpoint_sidecars(
        self,
        checkpoint_path: Path,
        cut: RolloutCheckpointCut,
    ) -> int:
        """Write controller state beside TQ and return its on-disk byte size."""
        written_paths = await write_controller_sidecars(
            checkpoint_path,
            ControllerSidecars(
                dataloader_state=cut.dataloader_state,
                replacement_reserve=cut.replacement_reserve,
                replay_metadata=cut.replay_metadata,
                rollout_recovery_payload=cut.rollout_recovery_payload,
            ),
        )

        config_path = checkpoint_path / "config.yaml"

        def _write_config() -> None:
            import yaml

            dumped = self._master_config.model_dump(mode="json")
            with config_path.open("w") as config_file:
                yaml.safe_dump(dumped, config_file)

        await asyncio.to_thread(_write_config)
        written_paths.append(str(config_path))
        return await asyncio.to_thread(
            lambda: sum(Path(path).stat().st_size for path in written_paths)
        )

    def _begin_optimizer_commit(self) -> None:
        """Mark the step's optimizer update as applied until its counters publish."""
        self._optimizer_commit_in_progress = True
        self._optimizer_commit_started.set()

    async def _enter_environment_phase_unless_a_step_commits(
        self,
        stack: contextlib.AsyncExitStack,
        environment_phase: contextlib.AbstractAsyncContextManager[
            Optional[EnvironmentSnapshot]
        ],
    ) -> tuple[bool, Optional[EnvironmentSnapshot]]:
        """Enter the environment phase on ``stack``, or stop it once a step commits.

        The snapshot would be thrown away at the trainer-state check anyway, so
        the environment is not kept paused while it saves for it. Cancelling the
        entry leaves no cleanup: the phase resumes the environment and reopens
        dispatch on any exception. Returns whether the phase was entered; a phase
        that finished entering is on ``stack``, and the trainer-state check
        decides whether its snapshot is kept.
        """
        enter = asyncio.ensure_future(stack.enter_async_context(environment_phase))
        commit_started = asyncio.ensure_future(self._optimizer_commit_started.wait())
        try:
            await asyncio.wait(
                {enter, commit_started}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            commit_started.cancel()
            if not enter.done():
                enter.cancel()
                await asyncio.gather(enter, return_exceptions=True)
        if enter.cancelled():
            return False, None
        return True, enter.result()

    async def _save_rollout_checkpoint(
        self, *, force: bool = False
    ) -> RolloutCheckpointSaveResult:
        """Publish one rollout-only snapshot anchored to durable trainer state."""
        async with self._checkpoint_save_lock:
            if self._optimizer_commit_in_progress:
                return RolloutCheckpointSaveResult(
                    saved=False,
                    reason="optimizer_commit_in_progress",
                )
            # Only a step that starts committing during this attempt stops it.
            self._optimizer_commit_started.clear()
            if (
                not force
                and self._last_rollout_snapshot_mutation_version
                == self._data_plane_checkpoint_barrier.mutation_version
            ):
                return RolloutCheckpointSaveResult(
                    saved=False,
                    reason="no_data_plane_mutations",
                )

            save_started = time.monotonic()
            await asyncio.to_thread(self._checkpointer.finalize_pending)
            train_steps = self._host.train_steps
            trainer_version = self._host.trainer_version
            if train_steps == 0:
                if trainer_version != 0:
                    raise RuntimeError(
                        "bootstrap rollout snapshot requires trainer version zero"
                    )
                if self._bootstrap_identity is None:
                    raise RuntimeError(
                        "rollout snapshotting requires a bootstrap identity"
                    )
                anchor = await asyncio.to_thread(
                    ensure_bootstrap_anchor,
                    self._checkpointer.checkpoint_dir,
                    identity=self._bootstrap_identity,
                )
                snapshot_fingerprint = self._bootstrap_identity.fingerprint()
            else:
                if trainer_version != train_steps:
                    raise RuntimeError(
                        "rollout snapshot trainer identity is ambiguous: "
                        f"step={train_steps}, "
                        f"trainer_version={trainer_version}"
                    )
                anchor = self._checkpointer.checkpoint_dir / f"step_{train_steps}"
                if not anchor.is_dir():
                    skip_key = (train_steps, trainer_version)
                    if self._last_missing_rollout_snapshot_anchor != skip_key:
                        print(
                            "rollout checkpoint skipped: matching trainer "
                            f"checkpoint is not durable yet: {anchor}",
                            flush=True,
                        )
                    self._last_missing_rollout_snapshot_anchor = skip_key
                    return RolloutCheckpointSaveResult(
                        saved=False,
                        reason="missing_trainer_anchor",
                    )
                try:
                    await asyncio.to_thread(
                        prune_bootstrap_snapshots,
                        self._checkpointer.checkpoint_dir,
                        durable_trainer_checkpoint=anchor,
                    )
                except OSError as error:
                    warnings.warn(
                        "Failed to prune obsolete bootstrap rollout snapshots: "
                        f"{type(error).__name__}: {error}",
                        stacklevel=2,
                    )
                snapshot_fingerprint = None

            expected_train_step = train_steps
            expected_trainer_version = trainer_version
            tmp_path, final_path, snapshot_sequence = await asyncio.to_thread(
                prepare_snapshot_paths, anchor
            )
            try:
                environment_pause_started = time.monotonic()
                async with contextlib.AsyncExitStack() as environment_phase:
                    (
                        entered,
                        environment_snapshot,
                    ) = await self._enter_environment_phase_unless_a_step_commits(
                        environment_phase,
                        self._environment_checkpoint.snapshot(tmp_path),
                    )
                    if not entered:
                        await asyncio.to_thread(partial(shutil.rmtree, tmp_path))
                        return RolloutCheckpointSaveResult(
                            saved=False,
                            reason="trainer_state_changed",
                        )
                    barrier_requested = time.monotonic()
                    async with self._data_plane_checkpoint_barrier.checkpoint() as cut:
                        barrier_acquired = time.monotonic()
                        if (
                            self._optimizer_commit_in_progress
                            or self._host.train_steps != expected_train_step
                            or self._host.trainer_version != expected_trainer_version
                        ):
                            await asyncio.to_thread(partial(shutil.rmtree, tmp_path))
                            return RolloutCheckpointSaveResult(
                                saved=False,
                                reason="trainer_state_changed",
                            )
                        snapshot_epoch = self._host.current_epoch
                        if environment_snapshot is not None:
                            await environment_snapshot.check_against_cut(cut)
                        snapshot_cut = await self._capture_rollout_checkpoint_cut(
                            cut, tmp_path
                        )
                    barrier_released = time.monotonic()
                # Dispatch reopens and the environment resumes when that block
                # exits, so this spans every second new rollouts could not start.
                environment_paused_seconds = (
                    time.monotonic() - environment_pause_started
                )

                sidecar_save_started = time.monotonic()
                controller_sidecar_bytes = (
                    await self._write_rollout_checkpoint_sidecars(
                        tmp_path,
                        snapshot_cut,
                    )
                )
                manifest = RolloutSnapshotManifest(
                    schema_version=ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
                    base_train_step=expected_train_step,
                    trainer_version=expected_trainer_version,
                    current_epoch=snapshot_epoch,
                    sampler_dispatch_index=snapshot_cut.sampler_dispatch_index,
                    mutation_version=snapshot_cut.mutation_version,
                    rolled_back_train_group_count=(
                        snapshot_cut.rolled_back_train_group_count
                    ),
                    bootstrap_fingerprint=snapshot_fingerprint,
                )
                manifest_text = (
                    json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n"
                )
                await asyncio.to_thread(
                    (tmp_path / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text,
                    manifest_text,
                )
                controller_sidecar_bytes += len(manifest_text.encode())
                sidecar_save_seconds = time.monotonic() - sidecar_save_started
                snapshot_commit_started = time.monotonic()
                await asyncio.to_thread(
                    commit_snapshot,
                    tmp_path,
                    final_path,
                    keep_latest_k=(
                        self._master_config.rollout_checkpointing.keep_latest_k
                    ),
                )
                snapshot_commit_seconds = time.monotonic() - snapshot_commit_started
            except BaseException:
                if tmp_path.exists():
                    await asyncio.to_thread(partial(shutil.rmtree, tmp_path))
                raise

            self._last_rollout_snapshot_mutation_version = snapshot_cut.mutation_version
            self._last_missing_rollout_snapshot_anchor = None
            save_completed = time.monotonic()
            checkpoint_metrics = {
                "snapshot_sequence": float(snapshot_sequence),
                "total_save_seconds": save_completed - save_started,
                "tq_save_seconds": snapshot_cut.tq_save_seconds,
                "barrier_wait_seconds": barrier_acquired - barrier_requested,
                "exclusive_hold_seconds": barrier_released - barrier_acquired,
                "sidecar_save_seconds": sidecar_save_seconds,
                "snapshot_commit_seconds": snapshot_commit_seconds,
                "replay_groups": float(
                    len(snapshot_cut.replay_metadata["groups"])
                    if snapshot_cut.replay_metadata is not None
                    else 0
                ),
                "ledger_groups": float(snapshot_cut.rollout_recovery_group_count or 0),
                "replay_rows": float(snapshot_cut.replay_row_count),
                "staging_rows": float(snapshot_cut.staging_row_count),
                "snapshot_rows": float(
                    snapshot_cut.replay_row_count + snapshot_cut.staging_row_count
                ),
                "controller_sidecar_bytes": float(controller_sidecar_bytes),
            }
            restarted_rollouts = 0
            if environment_snapshot is not None:
                restarted_rollouts = environment_snapshot.restarted_rollouts
                checkpoint_metrics.update(
                    environment_snapshot.metrics(environment_paused_seconds)
                )
                self._warn_if_environment_paused_past_interval(
                    environment_paused_seconds
                )
            self._host.log_telemetry_metrics(
                checkpoint_metrics,
                step=expected_train_step,
                prefix="timing/rollout_checkpoint",
            )
            print(
                "rollout checkpoint save completed: "
                f"{final_path} (step={expected_train_step}, "
                f"trainer_version={expected_trainer_version}, "
                f"ledger_groups={snapshot_cut.rollout_recovery_group_count or 0}, "
                f"restarted_rollouts={restarted_rollouts}, "
                f"total_save_seconds={checkpoint_metrics['total_save_seconds']:.2f}, "
                "exclusive_hold_seconds="
                f"{checkpoint_metrics['exclusive_hold_seconds']:.2f})",
                flush=True,
            )
            return RolloutCheckpointSaveResult(saved=True, reason="completed")

    def _warn_if_snapshots_keep_losing_to_train_steps(
        self, consecutive_discards: int
    ) -> None:
        """Warn once per streak when train steps keep discarding turn snapshots."""
        if not self._environment_checkpoint.holds_rollout_state:
            return
        limit = self._master_config.rollout_checkpointing.max_consecutive_failures
        if consecutive_discards != limit:
            return
        warnings.warn(
            f"{consecutive_discards} turn snapshots in a row were thrown away "
            "because a train step finished while the environment was saving; "
            "turn-level recovery is saving no environment state. Shorten "
            "environment steps or use rollout_recovery.target_level: sibling.",
            stacklevel=2,
        )

    def _warn_if_environment_paused_past_interval(self, paused_seconds: float) -> None:
        """Warn when a snapshot held rollouts longer than the snapshot interval.

        Rollouts then spend more time paused than running between snapshots.
        """
        interval_s = (
            self._master_config.rollout_checkpointing.snapshot_attempt_interval_s
        )
        if interval_s is None or paused_seconds <= interval_s:
            return
        warnings.warn(
            f"Environment rollout snapshot held new rollout dispatch for "
            f"{paused_seconds:.1f}s, longer than "
            f"rollout_checkpointing.snapshot_attempt_interval_s={interval_s:g}s; "
            "rollouts spend more time paused than running between snapshots. "
            "Raise the interval or check the timing/rollout_checkpoint metrics.",
            stacklevel=2,
        )

    def _log_rollout_checkpoint_outcome(
        self,
        *,
        outcome: RolloutCheckpointAttemptOutcome,
        reason: RolloutCheckpointAttemptReason,
        attempt_duration_seconds: float,
    ) -> None:
        """Record every scheduled checkpoint attempt, including no-op cuts."""
        metrics = {
            "attempt": 1.0,
            ROLLOUT_CHECKPOINT_ATTEMPT_DURATION_KEY: attempt_duration_seconds,
            "configured_interval_seconds": float(
                self._master_config.rollout_checkpointing.snapshot_attempt_interval_s
                or 0.0
            ),
            **{
                rollout_checkpoint_reason_key(candidate): float(reason == candidate)
                for candidate in ROLLOUT_CHECKPOINT_ATTEMPT_REASONS
            },
            **{
                candidate: float(outcome == candidate)
                for candidate in ROLLOUT_CHECKPOINT_ATTEMPT_OUTCOMES
            },
        }
        completed_at = time.monotonic()
        if outcome == "completed":
            if self._last_successful_rollout_checkpoint_time is not None:
                metrics["seconds_since_previous_success"] = (
                    completed_at - self._last_successful_rollout_checkpoint_time
                )
            self._last_successful_rollout_checkpoint_time = completed_at
        elif self._last_successful_rollout_checkpoint_time is not None:
            metrics[ROLLOUT_CHECKPOINT_SECONDS_SINCE_LAST_SUCCESS_KEY] = (
                completed_at - self._last_successful_rollout_checkpoint_time
            )
        self._host.log_telemetry_metrics(
            metrics,
            step=self._host.train_steps,
            prefix=ROLLOUT_CHECKPOINT_OUTCOME_PREFIX,
        )

    async def _rollout_checkpoint_pump(self) -> None:
        """Persist rollout state periodically, including during streamed train."""
        snapshot_attempt_interval_s = (
            self._master_config.rollout_checkpointing.snapshot_attempt_interval_s
        )
        if snapshot_attempt_interval_s is None:
            raise RuntimeError("rollout checkpoint pump started while disabled")
        consecutive_failures = 0
        consecutive_discards = 0
        while True:
            try:
                await asyncio.wait_for(
                    self._rollout_checkpoint_wakeup.wait(),
                    timeout=snapshot_attempt_interval_s,
                )
            except TimeoutError:
                pass
            self._rollout_checkpoint_wakeup.clear()
            attempt_started = time.monotonic()
            deadline_due = self._host.train_steps == 0 and self._timeout.would_save()
            try:
                result = await self._save_rollout_checkpoint(force=deadline_due)
            except Exception as error:
                if isinstance(error, TimeoutError):
                    failure_reason: RolloutCheckpointAttemptReason = "timeout"
                elif isinstance(error, OSError):
                    failure_reason = "io_error"
                elif self._environment_checkpoint.is_unavailable(error):
                    # Nothing RL holds changed, so the next attempt can retry.
                    failure_reason = "environment_unavailable"
                else:
                    failure_reason = "invariant_error"
                self._log_rollout_checkpoint_outcome(
                    outcome="failed",
                    reason=failure_reason,
                    attempt_duration_seconds=time.monotonic() - attempt_started,
                )
                if failure_reason == "invariant_error":
                    raise
                if deadline_due:
                    raise RuntimeError(
                        "failed to save the required pre-step rollout checkpoint"
                    ) from error
                consecutive_failures += 1
                print(
                    "Periodic rollout checkpoint failed; retaining the previous "
                    "committed snapshot: "
                    f"consecutive_failures={consecutive_failures}, "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if (
                    consecutive_failures
                    >= self._master_config.rollout_checkpointing.max_consecutive_failures
                ):
                    raise RuntimeError(
                        "periodic rollout checkpoint failed "
                        f"{consecutive_failures} consecutive times"
                    ) from error
                continue
            consecutive_failures = 0
            self._log_rollout_checkpoint_outcome(
                outcome="completed" if result.saved else "skipped",
                reason=result.reason,
                attempt_duration_seconds=time.monotonic() - attempt_started,
            )
            consecutive_discards = (
                consecutive_discards + 1
                if result.reason == "trainer_state_changed"
                else 0
            )
            self._warn_if_snapshots_keep_losing_to_train_steps(consecutive_discards)
            if deadline_due and result.saved and self._timeout.check_save():
                print(
                    "Checkpoint deadline reached before the first train step; "
                    "stopping after a durable rollout snapshot",
                    flush=True,
                )
                self._rollout_checkpoint_stop_requested.set()
                return
