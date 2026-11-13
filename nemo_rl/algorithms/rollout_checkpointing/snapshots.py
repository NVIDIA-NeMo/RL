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

"""Rollout snapshots and the rollout state of trainer checkpoints.

The single controller calls ``RolloutSnapshotter`` to save rollout state;
it never builds TQ cuts, sidecars or snapshot manifests itself. Trainer
checkpoints and rollout snapshots share one capture and one sidecar writer,
and one lock keeps them from publishing at the same time.
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
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
)

import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    REPLACEMENT_RESERVE_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    DataPlaneCheckpointMetadata,
    DataPlaneMutationCut,
    TQReplayGroupMetadata,
    TQReplayMetadataState,
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
from nemo_rl.algorithms.rollout_checkpointing.inventory import (
    capture_reward_settings,
    validate_replay_inventory,
    validate_rollout_recovery_inventory,
)
from nemo_rl.algorithms.rollout_checkpointing.manifest import (
    ROLLOUT_SNAPSHOT_MANIFEST_FILENAME,
    ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
    BootstrapCompatibilityIdentity,
    RolloutSnapshotManifest,
    adopt_staged_snapshot,
    commit_snapshot,
    ensure_bootstrap_anchor,
    prepare_snapshot_paths,
    prepare_staged_snapshot_path,
    prune_bootstrap_snapshots,
)
from nemo_rl.algorithms.rollout_checkpointing.state import (
    CapturedRolloutState,
    CheckpointKind,
    RolloutCheckpointSaveResult,
    RolloutParts,
    _UsesRolloutParts,
)
from nemo_rl.data.utils import save_dataloader_state
from nemo_rl.data_plane import (
    DATA_PLANE_CHECKPOINT_SCHEMA_VERSION,
)
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.environments.checkpointing.interfaces import (
    EnvironmentSnapshot,
)
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    build_rollout_recovery_state,
)
from nemo_rl.utils.checkpoint import (
    CheckpointManager,
    PathLike,
)
from nemo_rl.utils.path_utils import (
    save_checkpoint_object,
)
from nemo_rl.utils.timer import TimeoutChecker


async def write_controller_sidecars(
    checkpoint_path: PathLike,
    captured: CapturedRolloutState,
) -> list[str]:
    """Write the captured controller state beside TQ; return the files written."""
    written_paths = [os.path.join(checkpoint_path, "train_dataloader.pt")]
    await asyncio.to_thread(
        save_dataloader_state, captured.dataloader_state, checkpoint_path
    )
    objects = {
        REPLACEMENT_RESERVE_FILENAME: captured.replacement_reserve,
        REPLAY_BUFFER_METADATA_FILENAME: captured.replay_metadata,
    }
    for filename, value in objects.items():
        if value is not None:
            path = os.path.join(checkpoint_path, filename)
            await asyncio.to_thread(save_checkpoint_object, value, path)
            written_paths.append(path)
    if captured.rollout_recovery_payload is not None:
        path = os.path.join(checkpoint_path, ROLLOUT_RECOVERY_STATE_FILENAME)
        await asyncio.to_thread(
            Path(path).write_bytes, captured.rollout_recovery_payload
        )
        written_paths.append(path)
    return written_paths


def _warn_if_environment_paused_past_interval(
    paused_seconds: float, interval_s: Optional[float]
) -> None:
    """Warn when a snapshot held rollouts longer than the snapshot interval.

    Rollouts then spend more time paused than running between snapshots.
    """
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


class RolloutSnapshotter(_UsesRolloutParts):
    """Save rollout state for the single controller.

    Owns the save lock shared by trainer checkpoints and rollout snapshots,
    the optimizer-commit window that rollout snapshots skip, and the periodic
    snapshot pump.
    """

    def __init__(
        self,
        parts: RolloutParts,
        *,
        checkpointer: CheckpointManager,
        timeout: TimeoutChecker,
        bootstrap_identity: Optional[BootstrapCompatibilityIdentity],
        teacher_coordinator: Optional[Any],
        records_prompt_groups: Callable[[], bool],
    ) -> None:
        """Store the parts this saves and coordinates with.

        Args:
            parts: The controller's parts shared with restore.
            checkpointer: Owns the trainer checkpoint directory.
            timeout: The run's checkpoint deadline, shared with the train pump.
            bootstrap_identity: Identifies the run before its first trainer
                checkpoint; needed for snapshots taken before step 1.
            teacher_coordinator: Records which teacher tagged the rows, if any.
            records_prompt_groups: Whether dispatch records prompt groups in
                the recovery ledger, which decides what a trainer checkpoint saves.
        """
        super().__init__(parts)
        self._checkpointer = checkpointer
        self._timeout = timeout
        self._bootstrap_identity = bootstrap_identity
        self._teacher_coordinator = teacher_coordinator
        self._records_prompt_groups = records_prompt_groups

        # Full trainer checkpoints and lightweight rollout snapshots share one
        # namespace and must never publish concurrently.
        self._save_lock = asyncio.Lock()
        self._last_snapshot_mutation_version: Optional[int] = None
        self._last_missing_anchor: Optional[tuple[int, int]] = None
        self._last_success_time: Optional[float] = None
        self._stop_event = asyncio.Event()
        # Set by a trainer checkpoint, which holds no environment state, so the
        # next rollout snapshot captures the environment soon after it.
        self._wakeup = asyncio.Event()
        # Narrow unsafe window after an optimizer mutates model state and before
        # SC publishes the matching TQ cleanup and trainer counters. Gradient
        # accumulation remains snapshot-safe because selected rows stay owned by
        # the replay buffer and are re-indexed in periodic snapshots.
        self._optimizer_commit_in_progress = False
        # Set whenever a step starts committing. A rollout snapshot clears it when
        # it starts and stops its environment phase once it is set, since the
        # snapshot can no longer be kept.
        self._optimizer_commit_started = asyncio.Event()

    @property
    def stop_requested(self) -> bool:
        """Whether the pump stopped the run after the pre-step deadline snapshot."""
        return self._stop_event.is_set()

    async def begin_optimizer_commit(self, *, drain_snapshot: bool) -> None:
        """Open the window rollout snapshots skip, until ``end_optimizer_commit``.

        Args:
            drain_snapshot: First wait for a snapshot that is already saving the
                old trainer identity, as a step that publishes a new trainer
                version before its cleanup must.
        """
        if drain_snapshot:
            async with self._save_lock:
                self._begin_optimizer_commit()
        else:
            self._begin_optimizer_commit()

    def end_optimizer_commit(self) -> None:
        """Close the window once the step's TQ cleanup and counters are published."""
        self._optimizer_commit_in_progress = False

    @contextlib.asynccontextmanager
    async def trainer_checkpoint(self) -> AsyncIterator[None]:
        """Hold off rollout snapshots while a trainer checkpoint saves or publishes."""
        async with self._save_lock:
            yield

    def trainer_checkpoint_published(self, step: int) -> None:
        """Wake the pump once the trainer checkpoint for ``step`` is published.

        A trainer checkpoint holds no environment state, so the pump waits for
        it to be durable, off the training loop, and then captures the
        environment anchored to it, unless
        ``rollout_checkpointing.snapshot_on_trainer_checkpoint`` is off.
        """
        self._trace.emit("trainer_checkpoint_published", step=step)
        if (
            self._environment_checkpoint.holds_rollout_state
            and self._master_config.rollout_checkpointing.snapshot_on_trainer_checkpoint
        ):
            self._wakeup.set()

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
            "sampler_name": self._async_cfg.sampler.name,
            "mode": "authoritative" if replay_metadata is not None else "shadow",
            "forest_training": self._master_config.token_capture.forest_training,
            "forest_execution_row_multiple": (
                self._rollout_manager.execution_row_multiple
                if self._master_config.token_capture.forest_training
                else None
            ),
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

    async def capture(
        self,
        cut: DataPlaneMutationCut,
        checkpoint_path: PathLike,
        kind: CheckpointKind,
        *,
        additional_expected_staging_keys: Optional[set[str]] = None,
    ) -> CapturedRolloutState:
        """Capture restart-facing controller state and save TQ under the barrier.

        A trainer checkpoint saves TQ only with ``checkpointing.save_data_plane``,
        and refuses rows an unfinished streamed step still owns. A rollout
        snapshot always saves TQ and re-indexes those rows in the persisted cut
        only; the live trainer keeps accumulating gradients without modification.
        """
        cut.require_live()
        snapshot = kind == "rollout_snapshot"
        dataloader_state = self._dataloader.state_dict()
        next_nemo_gym_task_index = self._rollout_manager.get_next_nemo_gym_task_index()
        # The spare pool and dataloader advance together under the same mutation
        # cut in recovery-enabled dispatch, so preserve them in this cut too.
        replacement_reserve = {
            "replacement": list(self._rollout_runtime.replacement_reserve),
            "filtered": list(self._rollout_runtime.filtered_reserve),
        }
        replay_metadata: Optional[TQReplayMetadataState] = None
        recovery_payload: Optional[bytes] = None
        recovery_group_count: Optional[int] = None
        training_owned_groups: list[TQReplayGroupMetadata] = []
        staging_row_count = 0
        tq_save_seconds = 0.0
        if snapshot or self._master_config.checkpointing.get("save_data_plane"):
            training_owned_groups = self._buffer.training_owned_replay_groups()
            if training_owned_groups and not snapshot:
                raise RuntimeError(
                    "full trainer checkpoint still owns streamed training rows: "
                    f"groups={[group['group_id'] for group in training_owned_groups]!r}"
                )
            if snapshot or self._sampler.supports_buffer_checkpoint:
                replay_metadata = self._buffer.metadata_state_dict(
                    saved_capacity=self._async_cfg.max_buffered_rollouts,
                    additional_groups=training_owned_groups,
                )
                await validate_replay_inventory(
                    replay_metadata,
                    dp_client=self._dp_client,
                    partition_id=self._partition_id,
                )

            recovery_digest: Optional[str] = None
            if snapshot or self._records_prompt_groups():
                recovery_state = build_rollout_recovery_state(
                    self._rollout_manager.recovery_ledger,
                    batch_shortfall=self._rollout_runtime.batch_shortfall,
                    steps_allowed_short=self._rollout_runtime.steps_allowed_short,
                    sampler_stamps_target_steps=self._rollout_runtime.sampler_stamps_target_steps,
                    finalizer_metrics_by_group=self._rollout_runtime.finalizer_metrics_by_group,
                    finalizer_rewards_by_group=self._rollout_runtime.finalizer_rewards_by_group,
                    reward_settings=capture_reward_settings(self._master_config),
                    canonical_group_ids={
                        group["group_id"] for group in replay_metadata["groups"]
                    }
                    if replay_metadata is not None
                    else set(),
                )
                payload_buffer = io.BytesIO()
                await asyncio.to_thread(torch.save, recovery_state, payload_buffer)
                recovery_payload = payload_buffer.getvalue()
                recovery_digest = hashlib.sha256(recovery_payload).hexdigest()
                recovery_group_count = len(recovery_state["groups"])

            if self._master_config.token_capture.enabled:
                staging_row_count = await validate_rollout_recovery_inventory(
                    cut,
                    dp_client=self._dp_client,
                    recovery_ledger=self._rollout_recovery_ledger,
                    staging_partition=self._master_config.token_capture.staging_partition,
                    replay_metadata=replay_metadata,
                    clear_unreferenced=False,
                    additional_expected_staging_keys=additional_expected_staging_keys,
                )
            tq_save_started = time.monotonic()
            await self._save_data_plane_checkpoint(
                checkpoint_path,
                train_steps=self._read_progress().train_steps,
                trainer_version=self._read_progress().trainer_version,
                current_epoch=self._read_progress().current_epoch,
                replay_metadata=replay_metadata,
                rollout_recovery_payload_sha256=recovery_digest,
                rollout_recovery_group_count=recovery_group_count,
            )
            tq_save_seconds = time.monotonic() - tq_save_started
            # A trainer checkpoint holds no environment state, so when the
            # environment keeps some it does not stand in for the next rollout
            # snapshot.
            if not snapshot and not self._environment_checkpoint.holds_rollout_state:
                self._last_snapshot_mutation_version = (
                    self._data_plane_checkpoint_barrier.mutation_version
                )
        return CapturedRolloutState(
            dataloader_state=dataloader_state,
            sampler_dispatch_index=self._sampler.dispatch_index,
            next_nemo_gym_task_index=next_nemo_gym_task_index,
            replacement_reserve=replacement_reserve,
            replay_metadata=replay_metadata,
            rollout_recovery_payload=recovery_payload,
            rollout_recovery_group_count=recovery_group_count,
            replay_row_count=sum(
                len(group["meta"].sample_ids) for group in replay_metadata["groups"]
            )
            if replay_metadata is not None
            else 0,
            staging_row_count=staging_row_count,
            rolled_back_train_group_count=len(training_owned_groups),
            mutation_version=self._data_plane_checkpoint_barrier.mutation_version,
            tq_save_seconds=tq_save_seconds,
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

    async def _resolve_anchor(self) -> Optional[tuple[Path, Optional[str]]]:
        """Choose the durable trainer checkpoint this snapshot attaches to.

        Before the first train step that is the bootstrap anchor, identified by
        this run's fingerprint; after it, ``step_<N>`` for the current step.

        Returns:
            The anchor directory and the bootstrap fingerprint (``None`` past
            step 0), or ``None`` when the matching trainer checkpoint is not
            durable yet and the attempt is skipped.
        """
        progress = self._read_progress()
        if progress.train_steps == 0:
            if progress.trainer_version != 0:
                raise RuntimeError(
                    "bootstrap rollout snapshot requires trainer version zero"
                )
            if self._bootstrap_identity is None:
                raise RuntimeError("rollout snapshotting requires a bootstrap identity")
            anchor = await asyncio.to_thread(
                ensure_bootstrap_anchor,
                self._checkpointer.checkpoint_dir,
                identity=self._bootstrap_identity,
            )
            snapshot_fingerprint = self._bootstrap_identity.fingerprint()
        else:
            if progress.trainer_version != progress.train_steps:
                raise RuntimeError(
                    "rollout snapshot trainer identity is ambiguous: "
                    f"step={progress.train_steps}, "
                    f"trainer_version={progress.trainer_version}"
                )
            anchor = self._checkpointer.checkpoint_dir / f"step_{progress.train_steps}"
            if not anchor.is_dir():
                skip_key = (
                    progress.train_steps,
                    progress.trainer_version,
                )
                if self._last_missing_anchor != skip_key:
                    print(
                        "rollout checkpoint skipped: matching trainer "
                        f"checkpoint is not durable yet: {anchor}",
                        flush=True,
                    )
                self._last_missing_anchor = skip_key
                return None
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
        return anchor, snapshot_fingerprint

    async def _publish(
        self,
        tmp_path: Path,
        final_path: Path,
        captured: CapturedRolloutState,
        manifest: RolloutSnapshotManifest,
    ) -> tuple[int, float, float]:
        """Write the snapshot's files, then commit it atomically under retention.

        Returns:
            The controller files' on-disk bytes, and the seconds spent writing
            them and committing the snapshot.
        """
        sidecar_save_started = time.monotonic()
        written_paths = await write_controller_sidecars(tmp_path, captured)
        config_path = tmp_path / "config.yaml"

        def _write_config() -> None:
            import yaml

            dumped = self._master_config.model_dump(mode="json")
            with config_path.open("w") as config_file:
                yaml.safe_dump(dumped, config_file)

        await asyncio.to_thread(_write_config)
        written_paths.append(str(config_path))
        controller_sidecar_bytes = await asyncio.to_thread(
            lambda: sum(Path(path).stat().st_size for path in written_paths)
        )
        manifest_text = json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n"
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
            keep_latest_k=self._master_config.rollout_checkpointing.keep_latest_k,
        )
        return (
            controller_sidecar_bytes,
            sidecar_save_seconds,
            time.monotonic() - snapshot_commit_started,
        )

    async def save_snapshot(
        self, *, force: bool = False, staged: bool = False
    ) -> RolloutCheckpointSaveResult:
        """Publish one rollout-only snapshot anchored to durable trainer state.

        Args:
            force: Save even when the data plane has not changed since the last
                snapshot.
            staged: The trainer checkpoint for the current step is still being
                written. Write the snapshot beside it meanwhile, then wait for
                step_N to be published and move the snapshot in; skip it if
                step_N never appears.
        """
        async with self._save_lock:
            if self._optimizer_commit_in_progress:
                return RolloutCheckpointSaveResult(
                    saved=False,
                    reason="optimizer_commit_in_progress",
                )
            # Only a step that starts committing during this attempt stops it.
            self._optimizer_commit_started.clear()
            if (
                not force
                and self._last_snapshot_mutation_version
                == self._data_plane_checkpoint_barrier.mutation_version
                # Decoding grows a live call's recoverable prefix without
                # mutating the data plane, so keep cutting while one is active.
                and not (
                    self._master_config.rollout_recovery.generation_prefix_cuts_enabled
                    and self._rollout_recovery_ledger.has_dispatched_generation_prefix()
                )
            ):
                return RolloutCheckpointSaveResult(
                    saved=False,
                    reason="no_data_plane_mutations",
                )

            save_started = time.monotonic()
            expected_train_step = self._read_progress().train_steps
            expected_trainer_version = self._read_progress().trainer_version
            final_path: Optional[Path] = None
            if staged:
                if expected_train_step == 0 or (
                    expected_trainer_version != expected_train_step
                ):
                    raise RuntimeError(
                        "a staged rollout snapshot needs a trainer checkpoint: "
                        f"step={expected_train_step}, "
                        f"trainer_version={expected_trainer_version}"
                    )
                checkpoint_dir = self._checkpointer.checkpoint_dir
                anchor_path = checkpoint_dir / f"step_{expected_train_step}"
                snapshot_fingerprint = None
                tmp_path = await asyncio.to_thread(
                    prepare_staged_snapshot_path, checkpoint_dir, expected_train_step
                )
            else:
                await asyncio.to_thread(self._checkpointer.finalize_pending)
                anchor = await self._resolve_anchor()
                if anchor is None:
                    return RolloutCheckpointSaveResult(
                        saved=False,
                        reason="missing_trainer_anchor",
                    )
                anchor_path, snapshot_fingerprint = anchor
                tmp_path, final_path, snapshot_sequence = await asyncio.to_thread(
                    prepare_snapshot_paths, anchor_path
                )
            trainer_publish_wait_seconds = 0.0
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
                        cut_wall_time = time.time()
                        if (
                            self._optimizer_commit_in_progress
                            or self._read_progress().train_steps != expected_train_step
                            or self._read_progress().trainer_version
                            != expected_trainer_version
                        ):
                            await asyncio.to_thread(partial(shutil.rmtree, tmp_path))
                            return RolloutCheckpointSaveResult(
                                saved=False,
                                reason="trainer_state_changed",
                            )
                        snapshot_epoch = self._read_progress().current_epoch
                        environment_staging_keys: frozenset[str] = frozenset()
                        if environment_snapshot is not None:
                            # The environment's saved state can reference rows
                            # RL's ledger does not own, such as a generation
                            # prefix; the TQ capture must hold those too.
                            environment_staging_keys = (
                                await environment_snapshot.check_against_cut(cut)
                            )
                        snapshot_cut = await self.capture(
                            cut,
                            tmp_path,
                            "rollout_snapshot",
                            additional_expected_staging_keys=set(
                                environment_staging_keys
                            ),
                        )
                    barrier_released = time.monotonic()
                # Dispatch reopens and the environment resumes when that block
                # exits, so this spans every second new rollouts could not start.
                environment_paused_seconds = (
                    time.monotonic() - environment_pause_started
                )
                if final_path is None:
                    # Only now wait for step_N's policy write and rename; a failed
                    # write re-raises here.
                    publish_wait_started = time.monotonic()
                    await asyncio.to_thread(self._checkpointer.finalize_pending)
                    trainer_publish_wait_seconds = (
                        time.monotonic() - publish_wait_started
                    )
                    if not anchor_path.is_dir():
                        await asyncio.to_thread(partial(shutil.rmtree, tmp_path))
                        return RolloutCheckpointSaveResult(
                            saved=False,
                            reason="missing_trainer_anchor",
                        )
                    tmp_path, final_path, snapshot_sequence = await asyncio.to_thread(
                        adopt_staged_snapshot, tmp_path, anchor_path
                    )

                (
                    controller_sidecar_bytes,
                    sidecar_save_seconds,
                    snapshot_commit_seconds,
                ) = await self._publish(
                    tmp_path,
                    final_path,
                    snapshot_cut,
                    RolloutSnapshotManifest(
                        schema_version=ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
                        base_train_step=expected_train_step,
                        trainer_version=expected_trainer_version,
                        current_epoch=snapshot_epoch,
                        sampler_dispatch_index=snapshot_cut.sampler_dispatch_index,
                        next_nemo_gym_task_index=snapshot_cut.next_nemo_gym_task_index,
                        mutation_version=snapshot_cut.mutation_version,
                        rolled_back_train_group_count=(
                            snapshot_cut.rolled_back_train_group_count
                        ),
                        bootstrap_fingerprint=snapshot_fingerprint,
                    ),
                )
            except BaseException:
                if tmp_path.exists():
                    await asyncio.to_thread(partial(shutil.rmtree, tmp_path))
                raise

            self._last_snapshot_mutation_version = snapshot_cut.mutation_version
            self._last_missing_anchor = None
            save_completed = time.monotonic()
            checkpoint_metrics = {
                "snapshot_sequence": float(snapshot_sequence),
                "total_save_seconds": save_completed - save_started,
                "tq_save_seconds": snapshot_cut.tq_save_seconds,
                "barrier_wait_seconds": barrier_acquired - barrier_requested,
                "exclusive_hold_seconds": barrier_released - barrier_acquired,
                "sidecar_save_seconds": sidecar_save_seconds,
                "snapshot_commit_seconds": snapshot_commit_seconds,
                # How long a staged snapshot then waited for step_N; 0 otherwise.
                "trainer_publish_wait_seconds": trainer_publish_wait_seconds,
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
            environment_metrics: dict[str, float] = {}
            if environment_snapshot is not None:
                restarted_rollouts = environment_snapshot.restarted_rollouts
                environment_metrics = environment_snapshot.metrics(
                    environment_paused_seconds
                )
                checkpoint_metrics.update(environment_metrics)
                _warn_if_environment_paused_past_interval(
                    environment_paused_seconds,
                    self._master_config.rollout_checkpointing.snapshot_attempt_interval_s,
                )
            self._log_metrics(
                checkpoint_metrics,
                step=expected_train_step,
                prefix="timing/rollout_checkpoint",
            )
            self._trace.emit(
                "snapshot_published",
                path=str(final_path),
                step=expected_train_step,
                trainer_version=expected_trainer_version,
                # When the saved rollout state was captured.
                cut_time=cut_wall_time,
            )
            print(
                "rollout checkpoint save completed: "
                f"{final_path} (step={expected_train_step}, "
                f"trainer_version={expected_trainer_version}, "
                f"ledger_groups={snapshot_cut.rollout_recovery_group_count or 0}, "
                f"restarted_rollouts={restarted_rollouts}, "
                f"total_save_seconds={checkpoint_metrics['total_save_seconds']:.2f}, "
                "exclusive_hold_seconds="
                f"{checkpoint_metrics['exclusive_hold_seconds']:.2f}"
                # The environment's own figures, e.g. a prefix cut's fence hold.
                + "".join(
                    f", {name}={value:.2f}"
                    for name, value in environment_metrics.items()
                )
                + ")",
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

    def _log_outcome(
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
            if self._last_success_time is not None:
                metrics["seconds_since_previous_success"] = (
                    completed_at - self._last_success_time
                )
            self._last_success_time = completed_at
        elif self._last_success_time is not None:
            metrics[ROLLOUT_CHECKPOINT_SECONDS_SINCE_LAST_SUCCESS_KEY] = (
                completed_at - self._last_success_time
            )
        self._log_metrics(
            metrics,
            step=self._read_progress().train_steps,
            prefix=ROLLOUT_CHECKPOINT_OUTCOME_PREFIX,
        )

    def _failure_reason(self, error: Exception) -> RolloutCheckpointAttemptReason:
        """Classify a failed snapshot; only ``invariant_error`` is not retryable."""
        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, OSError):
            return "io_error"
        if self._environment_checkpoint.is_unavailable(error):
            # Nothing RL holds changed, so the next attempt can retry.
            return "gym_unavailable"
        return "invariant_error"

    async def _attempt(
        self, *, force: bool, staged: bool = False
    ) -> RolloutCheckpointSaveResult | Exception:
        """Run and log one snapshot attempt; return a retryable error, raise the rest."""
        started = time.monotonic()
        try:
            result = await self.save_snapshot(force=force, staged=staged)
        except Exception as error:
            reason = self._failure_reason(error)
            self._log_outcome(
                outcome="failed",
                reason=reason,
                attempt_duration_seconds=time.monotonic() - started,
            )
            if reason == "invariant_error":
                raise
            return error
        self._log_outcome(
            outcome="completed" if result.saved else "skipped",
            reason=result.reason,
            attempt_duration_seconds=time.monotonic() - started,
        )
        return result

    def _seconds_until_next_attempt(self) -> Optional[float]:
        """How long the pump waits for a wakeup before attempting anyway.

        ``None`` waits only for a trainer checkpoint's wakeup. Before the first
        train step the pump also wakes at the checkpoint deadline, since no
        train step will reach it.
        """
        waits = []
        interval_s = (
            self._master_config.rollout_checkpointing.snapshot_attempt_interval_s
        )
        if interval_s is not None:
            waits.append(interval_s)
        if self._read_progress().train_steps == 0:
            until_deadline = self._timeout.seconds_until_save()
            if until_deadline is not None:
                # A deadline snapshot that was skipped retries once a second.
                waits.append(max(until_deadline, 1.0))
        return min(waits, default=None)

    async def run(self) -> None:
        """Save rollout snapshots on schedule, including during streamed train.

        Attempts one every ``snapshot_attempt_interval_s`` when set, soon after
        each trainer checkpoint that asks for one, and at the checkpoint
        deadline when it arrives before the first train step.
        """
        consecutive_failures = 0
        consecutive_discards = 0
        while True:
            try:
                await asyncio.wait_for(
                    self._wakeup.wait(),
                    timeout=self._seconds_until_next_attempt(),
                )
            except TimeoutError:
                pass
            woken = self._wakeup.is_set()
            self._wakeup.clear()
            deadline_due = (
                self._read_progress().train_steps == 0 and self._timeout.would_save()
            )
            if (
                not woken
                and not deadline_due
                and self._master_config.rollout_checkpointing.snapshot_attempt_interval_s
                is None
            ):
                # A wait timed for a deadline that the first train step now handles.
                continue
            result = await self._attempt(force=deadline_due)
            if isinstance(result, Exception):
                if deadline_due:
                    raise RuntimeError(
                        "failed to save the required pre-step rollout checkpoint"
                    ) from result
                consecutive_failures += 1
                print(
                    "Periodic rollout checkpoint failed; retaining the previous "
                    "committed snapshot: "
                    f"consecutive_failures={consecutive_failures}, "
                    f"{type(result).__name__}: {result}",
                    flush=True,
                )
                if (
                    consecutive_failures
                    >= self._master_config.rollout_checkpointing.max_consecutive_failures
                ):
                    raise RuntimeError(
                        "periodic rollout checkpoint failed "
                        f"{consecutive_failures} consecutive times"
                    ) from result
                continue
            consecutive_failures = 0
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
                self._stop_event.set()
                return

    async def save_at_deadline(self) -> None:
        """Save the environment state the deadline's trainer checkpoint lacks.

        The run stops after this, so nothing later would snapshot it. A failure
        that leaves checkpoint state consistent only warns: the trainer
        checkpoint is durable, and a restore from it redoes the unfinished turns.
        """
        if not self._environment_checkpoint.holds_rollout_state:
            return
        # The pump need not also answer the trainer checkpoint's wakeup.
        self._wakeup.clear()
        # Runs while step_N's policy write is still landing; the snapshot is
        # moved under step_N once it is published.
        outcome = await self._attempt(force=True, staged=True)
        if isinstance(outcome, Exception):
            problem = f"failed: {type(outcome).__name__}: {outcome}"
        elif not outcome.saved:
            problem = f"was skipped ({outcome.reason})"
        else:
            return
        warnings.warn(
            f"rollout snapshot at the checkpoint deadline {problem}; a restore "
            "redoes the unfinished turns from the trainer checkpoint",
            stacklevel=2,
        )
