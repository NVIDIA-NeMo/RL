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

"""SingleController: asyncio orchestrator for the RL training loop.

CPU-only Ray actor that runs rollout/train pumps plus an optional periodic
checkpoint pump and coordinates the
other actors via lightweight RPCs. SC sends control signals and reads
metadata only — model tensors still move through DataPlane or NCCL.

Data flow:
  _rollout_pump  → gen.generate_and_push(prompt, dp_client) ← RPC to GenWorker
                     GenWorker → dp_client.put_samples(...)
  _train_pump    → sampler.evict/select against TQReplayBuffer
                 → _advantage_stage(meta) → dp_client.get_samples(...)
                                        → adv_estimator.compute_advantage(...)
                                        → dp_client.put_samples(...)
                 → trainer.begin/train_microbatches/finish_train_step (split API,
                     driver-side TQPolicy via asyncio.to_thread)
                     Trainer → dp_client.get_samples(...)   (via its own client)
                 → dp_client.clear_samples(...)             ← SC clears after train
  _sync_weights  → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import ray
import torch
import yaml

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    DATA_PLANE_CHECKPOINT_SCHEMA_VERSION,
    DataPlaneCheckpointBarrier,
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    TQReplayMetadataState,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    CheckpointDispatchSampler,
    TwoPhaseAdmissionSampler,
    create_sampler,
)
from nemo_rl.algorithms.grpo import GRPOSaveState, _write_latest_checkpoint_status
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    MasterConfig,
    required_rollout_recovery_capacity,
    validate_sampler_buffer_capacity,
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    ROLLOUT_SNAPSHOT_MANIFEST_FILENAME,
    ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
    RolloutSnapshotManifest,
    commit_snapshot,
    ensure_bootstrap_anchor,
    prepare_snapshot_paths,
    prune_bootstrap_snapshots,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.data_plane.schema import DP_CALIB_INPUT_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    RolloutAttemptStatus,
    RolloutRecoveryState,
)
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import TimeoutChecker, Timer

Generation = Union[VllmGeneration, SGLangGeneration]


@dataclass(frozen=True)
class _RolloutWorkItem:
    """One capacity-owned prompt group awaiting generation or recovery."""

    prompt: DatumSpec
    target_step: Optional[int]
    group_id: Optional[str]


@dataclass(frozen=True)
class _RolloutCheckpointCut:
    """In-memory sidecars captured with one native TQ snapshot."""

    dataloader_state: dict[str, Any]
    replay_metadata: Optional[TQReplayMetadataState]
    rollout_recovery_payload: Optional[bytes]
    rollout_recovery_group_count: Optional[int]
    mutation_version: int


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns two primary asyncio tasks and an optional checkpoint task:
      - _rollout_pump: dispatches prompts to GenerationWorkerActor
      - _train_pump:   claims DataPlane meta, trains, clears consumed rows,
                       then runs _sync_weights (drain gate + weight
                       synchronization) inline after each optimizer step
      - _rollout_checkpoint_pump: periodically snapshots TQ, replay metadata,
                       partial-rollout receipts, and the dataloader cursor

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        actor_args: SingleControllerActorArgs,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            actor_args: Pre-built actor args from setup_single_controller.
        """
        validate_single_controller_config(master_config)

        self._advantage_cfg = AdvantageConfig()
        self._partition_id: str = actor_args.partition_id

        self._master_config = master_config
        self._async_cfg = master_config.async_rl
        self._policy_logprobs_required = not (
            master_config.loss_fn.force_on_policy_ratio
            and master_config.grpo.get("seq_logprob_error_threshold") is None
        )
        self._reference_logprobs_required = not bool(
            master_config.grpo.get("skip_reference_policy_logprobs_calculation")
        )
        self._dp_client = actor_args.dp_client
        self._gen: Generation = actor_args.gen_handle
        self._trainer: TQPolicy = actor_args.trainer_handle
        self._dataloader = actor_args.dataloader
        self._weight_synchronizer = actor_args.weight_synchronizer
        self._advantage_estimator = actor_args.advantage_estimator
        self._loss_fn = actor_args.loss_fn
        self._buffer = actor_args.tq_buffer
        self._rollout_manager = actor_args.rollout_manager
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._timer = Timer()

        # Also built here, not on the driver: TimeoutChecker must capture
        # wall-clock start times inside the actor, not at driver setup time.
        # actor_args only carries the driver-side restore products
        # (save_state, last_checkpoint_path).
        self._checkpointer = CheckpointManager(master_config.checkpointing)
        self._timeout = TimeoutChecker(
            timeout=master_config.checkpointing["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        self._timeout.start_iterations()

        # Loaded (or default) GRPOSaveState; keys SC does not own
        # (val_reward, ...) pass through to saved checkpoints untouched.
        self._save_state: GRPOSaveState = actor_args.save_state
        self._last_checkpoint_path: Optional[str] = actor_args.last_checkpoint_path
        self._data_plane_checkpoint_metadata: Optional[dict[str, Any]] = (
            actor_args.data_plane_checkpoint_metadata
        )
        self._consumed_samples: int = actor_args.save_state["consumed_samples"]
        self._total_valid_tokens: int = actor_args.save_state.get(
            "total_valid_tokens", 0
        )  # Default to 0 for backward compatibility with older checkpoints

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = actor_args.train_cluster
        self._inference_cluster = actor_args.inference_cluster

        num_prompts_per_step = self._master_config.grpo["num_prompts_per_step"]
        self._sampler = create_sampler(
            self._buffer,
            self._async_cfg.sampler,
            resume_from_step=actor_args.save_state.get(
                "trainer_version", actor_args.save_state["current_step"]
            ),
        )
        if (
            self._master_config.checkpointing["enabled"]
            and self._sampler.supports_buffer_checkpoint
            and not self._master_config.data_plane.get("checkpointing_enabled")
        ):
            raise ValueError(
                "SingleController checkpointing with a replay-checkpoint-capable "
                "sampler requires data_plane.checkpointing_enabled=true so "
                "completed, unconsumed rollouts are recoverable."
            )
        required_capacity = self._sampler.required_buffer_capacity(num_prompts_per_step)
        validate_sampler_buffer_capacity(
            self._async_cfg,
            required_capacity=required_capacity,
            sampler_name=type(self._sampler).__name__,
        )

        # ── asyncio state ──────────────────────────────────────────────────
        # Commits and destructive clears use this lock with TQ snapshots. This
        # makes the native snapshot match the controller's metadata-only replay
        # index exactly. Generation may continue, but completed rollouts wait at
        # commit; _buffer_capacity bounds reservations and eventually stalls
        # dispatch instead of allowing unbounded TQ growth.
        # A future staging/finalizer path must join the same barrier before
        # native restore can be authoritative.
        self._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
        # Full trainer checkpoints and lightweight rollout snapshots share the
        # same filesystem namespace and must never finalize concurrently.
        self._checkpoint_save_lock = asyncio.Lock()
        self._last_rollout_snapshot_mutation_version: Optional[int] = None
        self._last_missing_rollout_snapshot_anchor: Optional[tuple[int, int]] = None
        self._bootstrap_fingerprint = actor_args.bootstrap_fingerprint
        self._train_step_active = False
        self._rollout_checkpoint_stop_requested = asyncio.Event()
        if self._buffer is not None:
            self._buffer.set_data_plane_checkpoint_barrier(
                self._data_plane_checkpoint_barrier
            )
        if self._master_config.token_capture.enabled:
            self._rollout_manager.set_data_plane_checkpoint_barrier(
                self._data_plane_checkpoint_barrier
            )

        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Set only after _rollout_pump exhausts its configured epochs and all
        # dispatched tasks finish successfully. Rollout failures propagate
        # through run() instead of being reported as normal exhaustion.
        self._rollout_exhausted: asyncio.Event = asyncio.Event()
        # The dataloader can exhaust while restored groups are still running.
        # Training may declare the rollout stream drained only after both
        # sources are finished.
        self._rollout_recovery_complete: asyncio.Event = asyncio.Event()
        self._rollout_recovery_complete.set()

        # Count of work items currently executing generation/finalization.
        self._inflight_rollouts: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[bool]] = set()

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )
        # One prompt-group concurrency budget shared by startup recovery and
        # fresh rollout dispatch. Restored and fresh groups share this limiter,
        # so startup recovery can overlap the dataloader without overwhelming
        # generation workers.
        self._rollout_slots: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_inflight_prompts
        )

        self._trainer_version: int = actor_args.save_state.get(
            "trainer_version", actor_args.save_state["current_step"]
        )
        self._train_steps: int = actor_args.save_state["current_step"]
        self._current_epoch: int = actor_args.save_state["current_epoch"]
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
        }

        print(
            f"SingleControllerActor: "
            f"sampler={self._async_cfg.sampler.name} "
            f"buffer={self._async_cfg.max_buffered_rollouts} "
            f"inflight={self._async_cfg.max_inflight_prompts} "
            f"weight_sync={type(self._weight_synchronizer).__name__}",
            flush=True,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        # Synchronize weights before starting the pumps
        await self._sync_weights()

        restored_replay_groups = await self._maybe_restore_replay_buffer()
        recovery_work_items = await self._prepare_rollout_recovery(
            restored_replay_groups=restored_replay_groups
        )

        # Start recovery and the normal pumps together. Recovery groups already
        # own their restored buffer permits, so the dataloader may reserve fresh
        # work only from genuine spare capacity. All dispatches share
        # _rollout_slots and _rollout_permitted.
        recovery_task = (
            asyncio.create_task(
                self._recover_prepared_rollout_groups(recovery_work_items)
            )
            if recovery_work_items
            else None
        )
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        rollout_checkpoint_task = (
            asyncio.create_task(self._rollout_checkpoint_pump())
            if self._master_config.rollout_checkpointing.interval_s is not None
            else None
        )
        tasks = {rollout_task, train_task}
        if recovery_task is not None:
            tasks.add(recovery_task)
        if rollout_checkpoint_task is not None:
            tasks.add(rollout_checkpoint_task)
        try:
            while tasks:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    # Propagate failures from every pump immediately.
                    await task
                if train_task in done:
                    break
                if rollout_checkpoint_task in done:
                    if self._rollout_checkpoint_stop_requested.is_set():
                        break
                    raise RuntimeError(
                        "rollout checkpoint pump exited without requesting stop"
                    )
                # A normally exhausted rollout pump leaves the train pump to
                # drain already committed groups. Keep waiting for train or a
                # pre-step checkpoint deadline.
        finally:
            all_tasks = [rollout_task, train_task]
            if recovery_task is not None:
                all_tasks.append(recovery_task)
            if rollout_checkpoint_task is not None:
                all_tasks.append(rollout_checkpoint_task)
            for task in all_tasks:
                task.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
            # Flush the last checkpoint's background finalization; a failure
            # raises on a clean exit but never masks a propagating exception.
            propagating = sys.exc_info()[0] is not None
            shutdown_succeeded = False
            try:
                await asyncio.to_thread(self._checkpointer.shutdown)
                shutdown_succeeded = True
            except Exception:
                if not propagating:
                    raise
                warnings.warn(
                    "Checkpoint finalization failed while handling an "
                    "exception; the original exception will be re-raised.",
                    stacklevel=2,
                )
            finally:
                if shutdown_succeeded:
                    try:
                        durable_anchor_path = await asyncio.to_thread(
                            self._checkpointer.get_latest_checkpoint_path
                        )
                        if durable_anchor_path is not None:
                            await asyncio.to_thread(
                                prune_bootstrap_snapshots,
                                self._checkpointer.checkpoint_dir,
                                durable_trainer_checkpoint=Path(durable_anchor_path),
                            )
                    except OSError as error:
                        warnings.warn(
                            "Failed to prune obsolete bootstrap rollout "
                            f"snapshots: {type(error).__name__}: {error}",
                            stacklevel=2,
                        )
                self._logger.finish()

        return {
            "train_steps": self._train_steps,
            "trainer_version": self._trainer_version,
        }

    async def ping(self) -> dict[str, Any]:
        """Liveness check — returns immediately if event loop is running."""
        return {
            "alive": True,
            "trainer_version": self._trainer_version,
            "train_steps": self._train_steps,
            "inflight_rollouts": self._inflight_rollouts,
            "rollout_permitted": self._rollout_permitted.is_set(),
            "epoch": self._current_epoch,
        }

    # ── internal helpers ───────────────────────────────────────────────────

    async def _maybe_restore_replay_buffer(self) -> int:
        """Restore replay-buffer groups from the previous run's checkpoint.

        No-op unless the sampler supports buffered-group recovery.

        Returns:
            Number of canonical replay groups restored.
        """
        if self._last_checkpoint_path is None:
            return 0
        metadata_path = os.path.join(
            self._last_checkpoint_path, REPLAY_BUFFER_METADATA_FILENAME
        )
        if (
            os.path.exists(metadata_path)
            and not self._sampler.supports_buffer_checkpoint
        ):
            raise RuntimeError(
                "The checkpoint contains native replay state, but the configured "
                f"sampler {self._async_cfg.sampler.name!r} does not support "
                "replay-buffer recovery"
            )
        if not self._sampler.supports_buffer_checkpoint:
            return 0
        if not os.path.exists(metadata_path):
            legacy_path = os.path.join(
                self._last_checkpoint_path, LEGACY_REPLAY_BUFFER_FILENAME
            )
            if os.path.exists(legacy_path):
                raise RuntimeError(
                    "Checkpoint contains legacy replay_buffer.pt state, which "
                    "predates authoritative native TQ replay recovery. Resume it "
                    "with the older implementation or explicitly start without "
                    "restoring buffered rollouts."
                )
            print(
                f"⚠️ No native replay metadata found at {metadata_path}. "
                "Starting with an empty replay buffer.",
                flush=True,
            )
            return 0
        print(f"📦 Restoring replay buffer metadata: {metadata_path}")
        # weights_only=False: the metadata sidecar contains pickled KVBatchMeta
        # objects but no rollout tensor payloads. It is a trusted same-job artifact.
        buffer_state = await asyncio.to_thread(
            torch.load, metadata_path, weights_only=False
        )
        if self._data_plane_checkpoint_metadata is None:
            raise RuntimeError(
                "Found metadata-only replay checkpoint, but the matching "
                "native TQ checkpoint was not restored during setup"
            )
        expected_manifest_digest_value = self._data_plane_checkpoint_metadata.get(
            "replay_manifest_digest"
        )
        if not isinstance(expected_manifest_digest_value, str):
            raise ValueError(
                "Restored TQ checkpoint metadata is missing a replay manifest digest"
            )
        expected_group_count = self._data_plane_checkpoint_metadata.get(
            "replay_group_count"
        )
        groups = buffer_state.get("groups")
        if (
            not isinstance(expected_group_count, int)
            or not isinstance(groups, list)
            or len(groups) != expected_group_count
        ):
            raise ValueError(
                "Replay-buffer metadata group count does not match the "
                "loaded TQ checkpoint metadata"
            )
        restored = await self._buffer.load_state_dict(
            buffer_state,
            max_groups=self._async_cfg.max_buffered_rollouts,
            expected_partition_id=self._partition_id,
            expected_group_size=self._master_config.grpo["num_generations_per_prompt"],
            expected_manifest_digest=expected_manifest_digest_value,
        )
        await self._validate_replay_inventory(buffer_state)

        # Each buffered group holds one _buffer_capacity permit. Restore fails
        # above if the saved group count exceeds current capacity.
        assert restored <= self._async_cfg.max_buffered_rollouts
        for _ in range(restored):
            await self._buffer_capacity.acquire()
        recovery_ledger = self._rollout_manager.recovery_ledger
        if recovery_ledger is not None:
            canonical_group_ids = {
                group["group_id"] for group in buffer_state["groups"]
            }
            discarded = recovery_ledger.discard_canonical_groups(canonical_group_ids)
            if discarded:
                print(
                    "rollout recovery reconciled canonical groups: "
                    f"discarded={discarded}",
                    flush=True,
                )
        return restored

    async def _validate_replay_inventory(
        self, replay_metadata: TQReplayMetadataState
    ) -> None:
        """Require the canonical TQ keys to match the SC replay index exactly."""
        expected_sample_ids = {
            sample_id
            for group in replay_metadata["groups"]
            for sample_id in group["meta"].sample_ids
        }
        actual_sample_ids = set(
            await call_data_plane(
                self._dp_client,
                "list_sample_ids",
                offload_sync=True,
                partition_id=self._partition_id,
            )
        )
        missing_sample_ids = sorted(expected_sample_ids - actual_sample_ids)
        unexpected_sample_ids = sorted(actual_sample_ids - expected_sample_ids)
        if missing_sample_ids or unexpected_sample_ids:
            raise RuntimeError(
                "Native TQ checkpoint inventory does not match the replay "
                "metadata sidecar: "
                f"missing={missing_sample_ids[:10]!r} "
                f"(total={len(missing_sample_ids)}), "
                f"unexpected={unexpected_sample_ids[:10]!r} "
                f"(total={len(unexpected_sample_ids)})"
            )
        print(
            "📦 Native TQ replay inventory validated: "
            f"samples={len(actual_sample_ids)}",
            flush=True,
        )

    async def _validate_rollout_recovery_inventory(
        self,
        *,
        clear_unreferenced: bool,
        recovery_state: Optional[RolloutRecoveryState] = None,
    ) -> None:
        """Require every sealed receipt to retain its staged TQ rows."""
        recovery_ledger = self._rollout_manager.recovery_ledger
        if recovery_ledger is None:
            return
        staging_partition = self._master_config.token_capture.staging_partition
        if recovery_state is None:
            expected_staging_keys = recovery_ledger.expected_staging_keys()
        else:
            expected_staging_keys = {
                staging_key
                for group in recovery_state["groups"]
                for sibling in group["siblings"]
                for attempt in sibling["attempts"][-1:]
                if attempt["status"] == RolloutAttemptStatus.SEALED.value
                for staging_key in attempt["staging_keys"]
            }
        actual_staging_keys = set(
            await call_data_plane(
                self._dp_client,
                "list_sample_ids",
                offload_sync=True,
                partition_id=staging_partition,
            )
        )
        missing_staging_keys = sorted(expected_staging_keys - actual_staging_keys)
        if missing_staging_keys:
            raise RuntimeError(
                "Rollout-recovery receipts reference staging rows missing from "
                "the native TQ checkpoint: "
                f"missing={missing_staging_keys[:10]!r} "
                f"(total={len(missing_staging_keys)})"
            )

        unreferenced_staging_keys = sorted(actual_staging_keys - expected_staging_keys)
        if clear_unreferenced and unreferenced_staging_keys:
            await call_data_plane(
                self._dp_client,
                "clear_samples",
                offload_sync=True,
                sample_ids=unreferenced_staging_keys,
                partition_id=staging_partition,
            )
            print(
                "rollout recovery cleared unreferenced staging rows: "
                f"count={len(unreferenced_staging_keys)}",
                flush=True,
            )
        print(
            "rollout recovery staging inventory validated: "
            f"referenced={len(expected_staging_keys)}",
            flush=True,
        )

    async def _prepare_rollout_recovery(
        self, *, restored_replay_groups: int
    ) -> list[_RolloutWorkItem]:
        """Validate restored groups and reclaim their durable buffer permits."""
        self._rollout_recovery_complete.set()
        metadata = self._data_plane_checkpoint_metadata or {}
        if "rollout_recovery_payload_sha256" not in metadata:
            if restored_replay_groups:
                self._restore_sampler_dispatch_state(
                    [], restored_replay_groups=restored_replay_groups
                )
            return []
        recovery_ledger = self._rollout_manager.recovery_ledger
        if recovery_ledger is None:
            raise RuntimeError(
                "Native TQ checkpoint advertises rollout recovery, but token "
                "capture did not construct a recovery ledger"
            )

        recovery_ledger.prepare_for_restart()
        groups = recovery_ledger.groups()
        if groups:
            self._rollout_recovery_complete.clear()
        else:
            self._rollout_recovery_complete.set()
        await self._validate_rollout_recovery_inventory(clear_unreferenced=True)
        if restored_replay_groups + len(groups) > self._async_cfg.max_buffered_rollouts:
            raise RuntimeError(
                "Restored canonical and unfinished rollout groups exceed current "
                "buffer capacity: "
                f"canonical={restored_replay_groups}, unfinished={len(groups)}, "
                f"capacity={self._async_cfg.max_buffered_rollouts}"
            )
        restored_group_count = restored_replay_groups + len(groups)
        configured_batch_size = self._master_config.grpo["num_prompts_per_step"]
        min_streaming_groups = self._async_cfg.min_groups_for_streaming_train
        free_capacity = self._async_cfg.max_buffered_rollouts - restored_group_count
        if (
            0 < restored_group_count < min_streaming_groups
            and free_capacity < configured_batch_size
        ):
            required_capacity = required_rollout_recovery_capacity(
                num_prompts_per_step=configured_batch_size,
                min_groups_for_streaming_train=min_streaming_groups,
            )
            raise RuntimeError(
                "Restored rollout state cannot make progress with the current "
                "buffer capacity: "
                f"restored_groups={restored_group_count}, "
                f"free_capacity={free_capacity}, "
                f"fresh_batch_size={configured_batch_size}, "
                f"min_groups_for_streaming_train={min_streaming_groups}. "
                "Increase async_rl.max_buffered_rollouts to at least "
                f"{required_capacity}."
            )

        work_items = [
            _RolloutWorkItem(
                prompt=group.prompt_payload,
                target_step=group.target_step,
                group_id=group.group_id,
            )
            for group in groups
        ]
        self._restore_sampler_dispatch_state(
            work_items, restored_replay_groups=restored_replay_groups
        )

        # Reclaim every restored group's saved capacity before fresh dataloader
        # admission begins. The guard above makes these acquisitions immediate;
        # doing them here prevents fresh work from consuming recovery's budget.
        acquired = 0
        try:
            for _ in groups:
                await self._buffer_capacity.acquire()
                acquired += 1
        except BaseException:
            for _ in range(acquired):
                self._buffer_capacity.release()
            raise
        return work_items

    def _restore_sampler_dispatch_state(
        self,
        recovery_work_items: list[_RolloutWorkItem],
        *,
        restored_replay_groups: int,
    ) -> None:
        """Reconstruct gated admission from canonical and unfinished groups."""
        if not isinstance(self._sampler, CheckpointDispatchSampler):
            raise RuntimeError(
                f"checkpoint-capable sampler {type(self._sampler).__name__} "
                "does not implement dispatch-state recovery"
            )
        restored_target_steps = list(self._buffer.target_step_list)
        if len(restored_target_steps) != restored_replay_groups:
            raise RuntimeError(
                "restored replay group count does not match the sampler target "
                "index: "
                f"groups={restored_replay_groups}, "
                f"targets={len(restored_target_steps)}"
            )
        restored_target_steps.extend(
            work_item.target_step for work_item in recovery_work_items
        )
        self._sampler.restore_dispatch_state(
            current_train_weight=self._trainer_version,
            restored_target_steps=restored_target_steps,
            groups_per_step=self._master_config.grpo["num_prompts_per_step"],
        )
        stamped_targets = [
            target for target in restored_target_steps if target is not None
        ]
        if stamped_targets:
            print(
                "📦 Restored sampler dispatch state: "
                f"sampler={type(self._sampler).__name__}, "
                f"oldest_target={min(stamped_targets)}, "
                f"newest_target={max(stamped_targets)}, "
                f"groups={len(stamped_targets)}",
                flush=True,
            )

    def _release_buffer_permit_if_task_not_started(
        self,
        _: asyncio.Task[Any],
        *,
        task_started_event: asyncio.Event,
        dispatch_admitted_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Return caller-owned capacity if a cancelled task never took it."""
        if not task_started_event.is_set():
            self._buffer_capacity.release()
        if dispatch_admitted_event is not None:
            dispatch_admitted_event.set()

    async def _execute_rollout_work_item(
        self,
        work_item: _RolloutWorkItem,
        *,
        task_started_event: asyncio.Event,
        dispatch_admitted_event: Optional[asyncio.Event] = None,
    ) -> bool:
        """Run one capacity-owned fresh or restored prompt group.

        The controller owns admission, concurrency, and permit lifetime. The
        rollout manager inspects durable ledger state to choose fresh
        generation, selective recovery, or finalizer-only replay.
        """
        # Set this before the first await. From here on, this coroutine owns the
        # buffer permit and must either transfer it to training or release it.
        task_started_event.set()
        buffer_permit_owned = True
        rollout_slot_owned = False
        counted_inflight = False
        try:
            await self._rollout_slots.acquire()
            rollout_slot_owned = True
            await self._rollout_permitted.wait()
            if dispatch_admitted_event is not None:
                dispatch_admitted_event.set()

            self._inflight_rollouts += 1
            counted_inflight = True
            if work_item.group_id is None:
                # Native, non-token-capture rollouts do not have durable
                # prompt-group lineage for RolloutManager to inspect.
                await self._rollout_manager.generate_and_push(
                    work_item.prompt,
                    target_step=work_item.target_step,
                )
                committed = True
            else:
                committed = await self._rollout_manager.ensure_rollout_group(
                    work_item.prompt,
                    group_id=work_item.group_id,
                    target_step=work_item.target_step,
                )

            if committed:
                # The canonical replay group retains the buffer permit until
                # the train pump consumes or evicts it.
                buffer_permit_owned = False

            if committed and self._async_cfg.diagnostics:
                content = ""
                for message in work_item.prompt["message_log"]:
                    if message["role"] == "user":
                        content = message["content"]
                        break
                print(f"  rollout done for prompt='{content[:20]}...'", flush=True)
            return committed
        finally:
            if dispatch_admitted_event is not None:
                # Unblock a fresh producer if cancellation/error happened
                # before this work item passed the generation gates.
                dispatch_admitted_event.set()
            if counted_inflight:
                self._inflight_rollouts -= 1
            if rollout_slot_owned:
                self._rollout_slots.release()
            if buffer_permit_owned:
                self._buffer_capacity.release()

    async def _recover_prepared_rollout_groups(
        self, work_items: list[_RolloutWorkItem]
    ) -> None:
        """Complete prepared groups while normal rollout/train pumps run."""
        try:
            recovery_tasks: list[asyncio.Task[bool]] = []
            for work_item in work_items:
                task_started_event = asyncio.Event()
                task = asyncio.create_task(
                    self._execute_rollout_work_item(
                        work_item,
                        task_started_event=task_started_event,
                    )
                )
                task.add_done_callback(
                    partial(
                        self._release_buffer_permit_if_task_not_started,
                        task_started_event=task_started_event,
                    )
                )
                recovery_tasks.append(task)
            recovery_results = await asyncio.gather(
                *recovery_tasks,
                return_exceptions=True,
            )
            recovery_errors = [
                result
                for result in recovery_results
                if isinstance(result, BaseException)
            ]
            if recovery_errors:
                raise BaseExceptionGroup(
                    "one or more restored rollout groups failed recovery",
                    recovery_errors,
                )

            if work_items:
                print(
                    "rollout recovery replay completed: "
                    f"groups={len(work_items)}, "
                    f"max_concurrency={self._async_cfg.max_inflight_prompts}",
                    flush=True,
                )
        finally:
            self._rollout_recovery_complete.set()

    async def _maybe_restore_rollout_recovery(
        self, *, restored_replay_groups: int
    ) -> None:
        """Prepare and complete restored groups without starting normal pumps.

        This compatibility helper keeps focused tests and direct callers simple;
        ``run`` uses the split prepare/recover methods to overlap recovery with
        fresh dataloader work.
        """
        work_items = await self._prepare_rollout_recovery(
            restored_replay_groups=restored_replay_groups
        )
        await self._recover_prepared_rollout_groups(work_items)

    async def _clear_data_plane_samples(self, sample_ids: list[str]) -> None:
        """Clear consumed rows without overlapping a data-plane checkpoint."""
        async with self._data_plane_checkpoint_barrier.mutation():
            await call_data_plane(
                self._dp_client,
                "clear_samples",
                offload_sync=True,
                sample_ids=sample_ids,
                partition_id=self._partition_id,
            )

    async def _save_data_plane_checkpoint(
        self,
        checkpoint_path: os.PathLike[str] | str,
        *,
        replay_metadata: Optional[TQReplayMetadataState] = None,
        rollout_recovery_payload_sha256: Optional[str] = None,
        rollout_recovery_group_count: Optional[int] = None,
    ) -> None:
        """Save a required TQ snapshot inside an SC checkpoint bundle.

        A sampler with replay-buffer recovery writes an authoritative native
        TQ snapshot bound to its metadata-only sidecar by a digest. Other
        samplers retain shadow-mode snapshots until their recovery contract is
        defined. Failures propagate so a finalized bundle never silently omits
        the advertised data-plane component.
        """
        checkpoint_dir = os.path.join(
            checkpoint_path,
            DATA_PLANE_CHECKPOINT_DIR,
        )
        metadata = {
            "data_plane_checkpoint_schema_version": (
                DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
            ),
            "single_controller_train_steps": self._train_steps,
            "single_controller_trainer_version": self._trainer_version,
            "single_controller_epoch": self._current_epoch,
            "partition_id": self._partition_id,
            "sampler_name": self._async_cfg.sampler.name,
            "mode": (
                "authoritative"
                if replay_metadata is not None
                or rollout_recovery_payload_sha256 is not None
                else "shadow"
            ),
        }
        if replay_metadata is not None:
            metadata.update(
                {
                    "replay_metadata_schema_version": (
                        REPLAY_BUFFER_METADATA_SCHEMA_VERSION
                    ),
                    "replay_manifest_digest": replay_metadata["manifest_digest"],
                    "replay_group_count": len(replay_metadata["groups"]),
                }
            )
        if rollout_recovery_payload_sha256 is not None:
            if rollout_recovery_group_count is None:
                raise ValueError(
                    "rollout recovery group count is required with its payload digest"
                )
            metadata.update(
                {
                    "rollout_recovery_schema_version": (
                        ROLLOUT_RECOVERY_SCHEMA_VERSION
                    ),
                    "rollout_recovery_payload_sha256": (
                        rollout_recovery_payload_sha256
                    ),
                    "rollout_recovery_group_count": rollout_recovery_group_count,
                }
            )
        elif rollout_recovery_group_count is not None:
            raise ValueError(
                "rollout recovery payload digest is required with its group count"
            )
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

    async def _capture_rollout_checkpoint_cut(
        self, checkpoint_path: os.PathLike[str] | str
    ) -> _RolloutCheckpointCut:
        """Capture cursor, replay index, ledger, and TQ in one barrier cut.

        The caller must hold ``DataPlaneCheckpointBarrier.checkpoint()``.
        This helper performs the native TQ save before returning, so the
        returned sidecars describe that exact immutable snapshot.
        """
        replay_metadata: Optional[TQReplayMetadataState] = None
        rollout_recovery_state: Optional[RolloutRecoveryState] = None
        rollout_recovery_payload: Optional[bytes] = None
        rollout_recovery_payload_sha256: Optional[str] = None
        dataloader_state = self._dataloader.state_dict()

        if self._master_config.data_plane.get("checkpointing_enabled"):
            if self._sampler.supports_buffer_checkpoint:
                replay_metadata = self._buffer.metadata_state_dict(
                    saved_capacity=self._async_cfg.max_buffered_rollouts
                )
            if self._master_config.token_capture.enabled:
                recovery_ledger = self._rollout_manager.recovery_ledger
                if recovery_ledger is None:
                    raise RuntimeError(
                        "token capture checkpointing requires a rollout recovery ledger"
                    )
                rollout_recovery_state = recovery_ledger.state_dict(
                    staging_partition=self._master_config.token_capture.staging_partition
                )
                if replay_metadata is not None:
                    # A canonical commit and the following ledger release
                    # straddle one event-loop yield. Canonical replay wins if
                    # a checkpoint lands in that narrow interval.
                    canonical_group_ids = {
                        group["group_id"] for group in replay_metadata["groups"]
                    }
                    rollout_recovery_state = {
                        "schema_version": rollout_recovery_state["schema_version"],
                        "staging_partition": rollout_recovery_state[
                            "staging_partition"
                        ],
                        "groups": [
                            group
                            for group in rollout_recovery_state["groups"]
                            if group["group_id"] not in canonical_group_ids
                        ],
                    }
                payload_buffer = io.BytesIO()
                torch.save(rollout_recovery_state, payload_buffer)
                rollout_recovery_payload = payload_buffer.getvalue()
                rollout_recovery_payload_sha256 = hashlib.sha256(
                    rollout_recovery_payload
                ).hexdigest()
                await self._validate_rollout_recovery_inventory(
                    clear_unreferenced=False,
                    recovery_state=rollout_recovery_state,
                )
            await self._save_data_plane_checkpoint(
                checkpoint_path,
                replay_metadata=replay_metadata,
                rollout_recovery_payload_sha256=rollout_recovery_payload_sha256,
                rollout_recovery_group_count=(
                    len(rollout_recovery_state["groups"])
                    if rollout_recovery_state is not None
                    else None
                ),
            )
            if replay_metadata is not None:
                await self._validate_replay_inventory(replay_metadata)

        return _RolloutCheckpointCut(
            dataloader_state=dataloader_state,
            replay_metadata=replay_metadata,
            rollout_recovery_payload=rollout_recovery_payload,
            rollout_recovery_group_count=(
                len(rollout_recovery_state["groups"])
                if rollout_recovery_state is not None
                else None
            ),
            mutation_version=self._data_plane_checkpoint_barrier.mutation_version,
        )

    async def _write_rollout_checkpoint_sidecars(
        self,
        checkpoint_path: os.PathLike[str] | str,
        cut: _RolloutCheckpointCut,
        *,
        include_config: bool,
    ) -> None:
        """Write the small controller-side files matching a native TQ cut."""
        checkpoint_path = Path(checkpoint_path)
        await asyncio.to_thread(
            torch.save,
            cut.dataloader_state,
            checkpoint_path / "train_dataloader.pt",
        )
        if cut.replay_metadata is not None:
            await asyncio.to_thread(
                torch.save,
                cut.replay_metadata,
                checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME,
            )
        if cut.rollout_recovery_payload is not None:
            await asyncio.to_thread(
                (checkpoint_path / ROLLOUT_RECOVERY_STATE_FILENAME).write_bytes,
                cut.rollout_recovery_payload,
            )
        if include_config:
            dumped_config = self._master_config.model_dump()

            def _write_config() -> None:
                with (checkpoint_path / "config.yaml").open("w") as config_file:
                    yaml.safe_dump(dumped_config, config_file)

            await asyncio.to_thread(_write_config)

    async def _save_rollout_checkpoint(self, *, force: bool = False) -> bool:
        """Publish a lightweight rollout snapshot when trainer state is anchored.

        Returns ``True`` when a new snapshot was committed and ``False`` when
        there was no new mutation or the controller was in an unsafe window.
        """
        async with self._checkpoint_save_lock:
            if self._train_step_active:
                return False
            if (
                not force
                and self._last_rollout_snapshot_mutation_version
                == self._data_plane_checkpoint_barrier.mutation_version
            ):
                return False

            await asyncio.to_thread(self._checkpointer.finalize_pending)
            if self._train_steps == 0:
                if self._trainer_version != 0:
                    raise RuntimeError(
                        "bootstrap rollout snapshot requires trainer version zero"
                    )
                if self._bootstrap_fingerprint is None:
                    raise RuntimeError(
                        "rollout snapshotting requires a bootstrap fingerprint"
                    )
                anchor = await asyncio.to_thread(
                    ensure_bootstrap_anchor,
                    self._checkpointer.checkpoint_dir,
                    fingerprint=self._bootstrap_fingerprint,
                )
                snapshot_fingerprint = self._bootstrap_fingerprint
            else:
                if self._trainer_version != self._train_steps:
                    raise RuntimeError(
                        "rollout snapshot trainer identity is ambiguous: "
                        f"step={self._train_steps}, "
                        f"trainer_version={self._trainer_version}"
                    )
                anchor = self._checkpointer.checkpoint_dir / f"step_{self._train_steps}"
                if not anchor.is_dir():
                    # The policy/optimizer for this version has not been made
                    # durable. Saving only TQ would create an unrestorable cut.
                    skip_key = (self._train_steps, self._trainer_version)
                    if self._last_missing_rollout_snapshot_anchor != skip_key:
                        print(
                            "rollout checkpoint skipped: matching trainer "
                            f"checkpoint is not durable yet: {anchor}",
                            flush=True,
                        )
                        self._last_missing_rollout_snapshot_anchor = skip_key
                    return False
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

            expected_train_step = self._train_steps
            expected_trainer_version = self._trainer_version

            tmp_path, final_path, _ = await asyncio.to_thread(
                prepare_snapshot_paths, anchor
            )
            try:
                async with self._data_plane_checkpoint_barrier.checkpoint():
                    # The train pump may have selected its first batch while
                    # this writer waited for active mutations to drain.
                    if (
                        self._train_step_active
                        or self._train_steps != expected_train_step
                        or self._trainer_version != expected_trainer_version
                    ):
                        await asyncio.to_thread(shutil.rmtree, tmp_path)
                        return False
                    snapshot_epoch = self._current_epoch
                    cut = await self._capture_rollout_checkpoint_cut(tmp_path)

                await self._write_rollout_checkpoint_sidecars(
                    tmp_path,
                    cut,
                    include_config=True,
                )
                manifest = RolloutSnapshotManifest(
                    schema_version=ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
                    base_train_step=expected_train_step,
                    trainer_version=expected_trainer_version,
                    current_epoch=snapshot_epoch,
                    mutation_version=cut.mutation_version,
                    bootstrap_fingerprint=snapshot_fingerprint,
                )
                await asyncio.to_thread(
                    (tmp_path / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text,
                    json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
                )
                await asyncio.to_thread(
                    commit_snapshot,
                    tmp_path,
                    final_path,
                    keep_latest_k=(
                        self._master_config.rollout_checkpointing.keep_latest_k
                    ),
                )
            except BaseException:
                if tmp_path.exists():
                    await asyncio.to_thread(shutil.rmtree, tmp_path)
                raise

            self._last_rollout_snapshot_mutation_version = cut.mutation_version
            self._last_missing_rollout_snapshot_anchor = None
            print(
                "rollout checkpoint save completed: "
                f"{final_path} "
                f"(step={expected_train_step}, "
                f"trainer_version={expected_trainer_version}, "
                f"ledger_groups={cut.rollout_recovery_group_count or 0})",
                flush=True,
            )
            return True

    async def _rollout_checkpoint_pump(self) -> None:
        """Periodically persist rollout state, including before train step one."""
        interval_s = self._master_config.rollout_checkpointing.interval_s
        if interval_s is None:
            raise RuntimeError("rollout checkpoint pump started while disabled")

        while True:
            await asyncio.sleep(interval_s)
            deadline_due = (
                self._train_steps == 0
                and not self._train_step_active
                and self._timeout.would_save()
            )
            try:
                saved = await self._save_rollout_checkpoint(force=deadline_due)
            except Exception as error:
                if deadline_due:
                    raise RuntimeError(
                        "failed to save the required pre-step rollout checkpoint"
                    ) from error
                warnings.warn(
                    "Periodic rollout checkpoint failed; retaining the previous "
                    f"committed snapshot: {type(error).__name__}: {error}",
                    stacklevel=2,
                )
                continue

            if deadline_due:
                if not saved:
                    print(
                        "Pre-step checkpoint deadline reached during an unsafe "
                        "snapshot window; retrying without consuming the trainer "
                        "checkpoint deadline",
                        flush=True,
                    )
                    continue
                # Claim the shared one-shot deadline only after the rollout
                # snapshot is durable. If the train pump claimed it while this
                # save was in progress, it owns the required full checkpoint.
                if not self._timeout.check_save():
                    continue
                print(
                    "Checkpoint deadline reached before the first train step; "
                    "stopping after a durable rollout snapshot",
                    flush=True,
                )
                self._rollout_checkpoint_stop_requested.set()
                return

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch:
          0. On the token-capture path, acquire capacity for the whole prompt
             batch before advancing the dataloader and reserving durable
             lineage. This keeps every checkpointed group capacity-owned.
          1. await sampler.admit(...) to wait until the batch may dispatch and
             obtain its target_step stamp.

        Per prompt:
          1. Acquire _buffer_capacity slot unless the token-capture batch
             already owns it (backpressure)
          2. Submit one capacity-owned work item to the common executor
          3. The executor applies _rollout_slots and _rollout_permitted
          4. RolloutManager routes durable groups to fresh generation,
             selective recovery, or finalizer-only replay
        """
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        def _release_buffer_capacity(permits: int) -> None:
            for _ in range(permits):
                self._buffer_capacity.release()

        async def _acquire_buffer_capacity(permits: int) -> None:
            acquired = 0
            try:
                for _ in range(permits):
                    await self._buffer_capacity.acquire()
                    acquired += 1
            except BaseException:
                _release_buffer_capacity(acquired)
                raise

        max_epochs = self._master_config.grpo["max_num_epochs"]
        token_capture_enabled = self._master_config.token_capture.enabled
        if token_capture_enabled and not isinstance(
            self._sampler, TwoPhaseAdmissionSampler
        ):
            raise RuntimeError(
                f"token-capture sampler {type(self._sampler).__name__} must "
                "implement two-phase admission so checkpointing never waits "
                "while holding the mutation barrier"
            )
        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
                dataloader_iterator = iter(self._dataloader)
                while True:
                    preacquired_capacity = 0
                    if token_capture_enabled:
                        # Gated samplers may wait for the trainer for a long
                        # time. Wait before taking capacity or the checkpoint
                        # barrier; the non-blocking admission commit below is
                        # atomic with dataloader advancement and ledger
                        # reservation.
                        await self._sampler.wait_until_admissible(
                            trainer_version_fn=lambda: self._trainer_version
                        )
                        # Capacity is acquired before the durable reservation,
                        # but never while holding the checkpoint barrier. A
                        # snapshot therefore cannot contain more recoverable
                        # groups than the buffer can reconstruct at startup.
                        configured_batch_size = self._master_config.grpo[
                            "num_prompts_per_step"
                        ]
                        await _acquire_buffer_capacity(configured_batch_size)
                        preacquired_capacity = configured_batch_size
                        end_of_epoch = False
                        try:
                            # Advancing the dataloader and reserving every prompt
                            # in that batch is one checkpoint mutation. A snapshot
                            # therefore sees either neither operation or both,
                            # preventing skipped/duplicated prompts after restart.
                            async with self._data_plane_checkpoint_barrier.mutation():
                                try:
                                    prompt_batch = next(dataloader_iterator)
                                except StopIteration:
                                    end_of_epoch = True
                                if not end_of_epoch:
                                    if prompt_batch.size > preacquired_capacity:
                                        raise RuntimeError(
                                            "dataloader batch exceeds its reserved "
                                            "rollout capacity: "
                                            f"batch={prompt_batch.size}, "
                                            f"reserved={preacquired_capacity}"
                                        )
                                    unused_capacity = (
                                        preacquired_capacity - prompt_batch.size
                                    )
                                    _release_buffer_capacity(unused_capacity)
                                    preacquired_capacity = prompt_batch.size
                                    target_step = self._sampler.commit_admission(
                                        trainer_version=self._trainer_version
                                    )
                                    rollout_work_items: list[_RolloutWorkItem] = []
                                    for prompt_idx in range(prompt_batch.size):
                                        prompt: DatumSpec = {  # type: ignore
                                            k: v[prompt_idx]
                                            for k, v in prompt_batch.items()
                                        }
                                        recovery_group_id = (
                                            self._rollout_manager.reserve_prompt_group(
                                                prompt, target_step=target_step
                                            )
                                        )
                                        if recovery_group_id is None:
                                            raise RuntimeError(
                                                "token-capture dispatch must reserve "
                                                "durable prompt-group lineage"
                                            )
                                        rollout_work_items.append(
                                            _RolloutWorkItem(
                                                prompt=prompt,
                                                target_step=target_step,
                                                group_id=recovery_group_id,
                                            )
                                        )
                        except BaseException:
                            _release_buffer_capacity(preacquired_capacity)
                            raise
                        if end_of_epoch:
                            _release_buffer_capacity(preacquired_capacity)
                            preacquired_capacity = 0
                            self._current_epoch += 1
                            break
                    else:
                        try:
                            prompt_batch = next(dataloader_iterator)
                        except StopIteration:
                            self._current_epoch += 1
                            break
                        target_step = await self._sampler.admit(
                            trainer_version_fn=lambda: self._trainer_version
                        )
                        rollout_work_items = []
                        for prompt_idx in range(prompt_batch.size):
                            prompt = {  # type: ignore
                                k: v[prompt_idx] for k, v in prompt_batch.items()
                            }
                            rollout_work_items.append(
                                _RolloutWorkItem(
                                    prompt=prompt,
                                    target_step=target_step,
                                    group_id=None,
                                )
                            )

                    pending_preacquired_capacity = preacquired_capacity
                    try:
                        for work_item in rollout_work_items:
                            capacity_acquired_here = False
                            if not token_capture_enabled:
                                # Token-capture groups already own capacity from
                                # their durable batch reservation above.
                                await self._buffer_capacity.acquire()
                                capacity_acquired_here = True
                            task_started_event = asyncio.Event()
                            dispatch_admitted_event = asyncio.Event()
                            try:
                                task = rollout_tasks.create_task(
                                    self._execute_rollout_work_item(
                                        work_item,
                                        task_started_event=task_started_event,
                                        dispatch_admitted_event=dispatch_admitted_event,
                                    )
                                )
                            except BaseException:
                                if capacity_acquired_here:
                                    self._buffer_capacity.release()
                                elif token_capture_enabled:
                                    pending_preacquired_capacity -= 1
                                    self._buffer_capacity.release()
                                raise

                            if token_capture_enabled:
                                pending_preacquired_capacity -= 1
                            self._dispatched_rollouts.add(task)
                            task.add_done_callback(self._dispatched_rollouts.discard)
                            task.add_done_callback(
                                partial(
                                    self._release_buffer_permit_if_task_not_started,
                                    task_started_event=task_started_event,
                                    dispatch_admitted_event=dispatch_admitted_event,
                                )
                            )
                            # Preserve the old producer pacing: do not reserve
                            # more dataloader work while this item is still
                            # waiting for a generation slot or weight sync.
                            await dispatch_admitted_event.wait()
                    finally:
                        _release_buffer_capacity(pending_preacquired_capacity)

        # Drain fresh dataloader work owned by this pump. Restored work is
        # owned by _recover_prepared_rollout_groups and tracked by run().
        inflight = list(self._dispatched_rollouts)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        self._rollout_exhausted.set()
        print(f"rollout_pump: completed {self._current_epoch} epoch(s)", flush=True)

    async def _train_pump(self) -> None:
        """Per-prompt-group streaming train loop.

        Per step:
          1. sampler.evict drops stale groups from the buffer and clears their TQ rows.
          2. sampler.select returns K prompt groups (or None) and drops them from the
             buffer; DP rows survive so the trainer can read them. Already trainable —
             buffer wrote training-shaped rows at rollout time.
          3. _advantage_stage(train_meta).
          4. trainer.train_microbatches_from_meta + finish_train_step.
          5. dp_client.clear_samples on consumed sample_ids; release _buffer_capacity
             per dropped group, then sync.
        """
        grpo_cfg = self._master_config.grpo

        while self._train_steps < grpo_cfg["max_num_steps"]:
            version_during_step = self._trainer_version
            groups_dispatched = 0
            min_sample_version = None
            step_open = False
            calibration_batches: list[BatchedDataDict[Any]] = []

            with self._timer.time("total_step_time"):
                while groups_dispatched < grpo_cfg["num_prompts_per_step"]:
                    # Wait for a selectable batch
                    with self._timer.time("exposed_generation"):
                        await asyncio.sleep(0)

                        # Evict stale groups
                        evicted = await self._sampler.evict(
                            current_train_weight=self._trainer_version,
                        )
                        if evicted:
                            print(
                                f"  evicted {evicted} stale prompt group(s)",
                                flush=True,
                            )
                            for _ in range(evicted):
                                self._buffer_capacity.release()

                        # Select a batch
                        max_prompt_groups = (
                            grpo_cfg["num_prompts_per_step"] - groups_dispatched
                        )
                        min_prompt_groups = min(
                            self._async_cfg.min_groups_for_streaming_train,
                            max_prompt_groups,
                        )
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=min_prompt_groups,
                            max_prompt_groups=max_prompt_groups,
                        )

                        # If no batch is selectable, sleep and retry
                        if train_meta is None:
                            if (
                                self._rollout_exhausted.is_set()
                                and self._rollout_recovery_complete.is_set()
                            ):
                                buffered_groups = len(self._buffer)
                                if groups_dispatched == 0 and buffered_groups == 0:
                                    print(
                                        "train_pump: rollout exhausted and "
                                        "buffer drained",
                                        flush=True,
                                    )
                                    return
                                raise RuntimeError(
                                    "rollout exhausted before a complete training "
                                    f"step was assembled: dispatched "
                                    f"{groups_dispatched}/"
                                    f"{grpo_cfg['num_prompts_per_step']} prompt "
                                    f"groups with {buffered_groups} group(s) "
                                    f"remaining in the buffer"
                                )
                            await asyncio.sleep(0.005)
                            continue

                        # Once the sampler removes the first group from its
                        # replay index, the current optimizer step owns rows
                        # that are intentionally absent from that index. A
                        # rollout-only snapshot is unsafe until the step and
                        # any matching full trainer checkpoint complete.
                        self._train_step_active = True

                        # Release buffer capacity
                        for _ in range(num_groups):
                            self._buffer_capacity.release()

                    # Compute prev_logprobs / ref_logprobs
                    if (
                        self._policy_logprobs_required
                        or self._reference_logprobs_required
                    ):
                        with self._timer.time("logprob_inference_prep"):
                            await asyncio.to_thread(
                                self._trainer.prepare_for_lp_inference
                            )
                        with self._timer.time("policy_and_reference_logprobs"):
                            if self._policy_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_logprobs_from_meta, train_meta
                                )
                            if self._reference_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_reference_policy_logprobs_from_meta,
                                    train_meta,
                                )

                    # Compute advantages
                    with self._timer.time("advantage_calculation"):
                        train_meta = await self._advantage_stage(train_meta)

                    # Train
                    with self._timer.time("training_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_training)
                    with self._timer.time("policy_training"):
                        if not step_open:
                            await asyncio.to_thread(
                                self._trainer.begin_train_step,
                                self._loss_fn,
                            )
                            step_open = True
                        await asyncio.to_thread(
                            self._trainer.train_microbatches_from_meta,
                            train_meta,
                        )

                    if train_meta.sequence_lengths:
                        self._step_log_dict["sequence_lengths"].extend(
                            int(s) for s in train_meta.sequence_lengths
                        )

                    if getattr(self._gen, "requires_kv_scale_sync", False):
                        calibration_fields = [
                            field
                            for field in (train_meta.fields or [])
                            if field in DP_CALIB_INPUT_FIELDS
                        ]
                        calibration_batches.append(
                            await asyncio.to_thread(
                                self._trainer.read_from_dataplane,
                                train_meta,
                                select_fields=calibration_fields,
                            )
                        )

                    # Refresh min_sample_version
                    curr_min_sample_version = min(
                        t["weight_version"]
                        for t in train_meta.tags  # type: ignore
                    )
                    if min_sample_version is not None:
                        min_sample_version = min(
                            min_sample_version, curr_min_sample_version
                        )
                    else:
                        min_sample_version = curr_min_sample_version

                    # Remove consumed sample_ids from the buffer
                    await self._clear_data_plane_samples(list(train_meta.sample_ids))

                    groups_dispatched += num_groups

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(self._trainer.finish_train_step)

                step_metrics = aggregate_step_metrics(result)
                step_metrics.update(
                    reduce_advantage_pump_metrics(**self._step_log_dict)
                )
                self._step_log_dict = {k: [] for k in self._step_log_dict}

                self._trainer_version += 1
                self._train_steps += 1
                with self._timer.time("weight_sync"):
                    calibration_data = (
                        BatchedDataDict.from_batches(calibration_batches)
                        if calibration_batches
                        else None
                    )
                    await self._sync_weights(calibration_data=calibration_data)

                # Checkpointing (mirrors async_grpo_train's save block).
                self._consumed_samples += grpo_cfg["num_prompts_per_step"]
                self._total_valid_tokens += step_metrics.get("global_valid_toks", 0)
                self._timeout.mark_iteration()

                is_last_step = self._train_steps >= grpo_cfg["max_num_steps"]
                # _train_steps was already incremented above, so it equals
                # the legacy loop's 1-indexed `step + 1`.
                should_save_by_step = (
                    is_last_step
                    or self._train_steps
                    % self._master_config.checkpointing["save_period"]
                    == 0
                )
                should_save_by_timeout = self._timeout.check_save()

                if self._master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    with self._timer.time("checkpointing"):
                        await self._save_checkpoint(step_metrics)

                self._train_step_active = False

            timing_metrics: dict[str, float] = self._timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            total_time = timing_metrics.get("total_step_time", 0.0)
            total_num_gpus = int(ray.cluster_resources().get("GPU", 0))
            if (
                total_time > 0
                and total_num_gpus > 0
                and "global_valid_toks" in step_metrics
            ):
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    step_metrics["global_valid_toks"] / total_time / total_num_gpus
                )

            print("\n⏱️  Timing:")
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k == "total_step_time":
                    continue
                percent = (v / total_time * 100) if total_time > 0 else 0.0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            # TODO: per-step train_data jsonl dump, vllm metrics logger,
            #   histogram log, rollout_metrics, seq_logprob_error_metrics,
            #   pretty-print "Training Results" block, print_performance_metrics.
            print(f"step_metrics={step_metrics}", flush=True)
            self._logger.log_metrics(
                step_metrics, step=self._train_steps, prefix="train"
            )
            self._logger.log_metrics(
                timing_metrics, step=self._train_steps, prefix="timing/train"
            )
            if self._master_config.token_capture.enabled:
                await self._log_gate_metrics()
            self._timer.reset()

            # min sample version refers to the version each consumed sample was
            # generated with; lag = training version - oldest sample version.
            lag = version_during_step - min_sample_version  # type: ignore
            print(
                f"train step {self._train_steps}/{grpo_cfg['max_num_steps']}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                break

    async def _save_checkpoint(
        self,
        step_metrics: dict[str, Any],
        val_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Serialize full and rollout-only checkpoint publication."""
        async with self._checkpoint_save_lock:
            await self._save_checkpoint_locked(step_metrics, val_metrics)

    async def _save_checkpoint_locked(
        self,
        step_metrics: dict[str, Any],
        val_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write a full checkpoint for the just-finished train step.

        Everything except the (possibly async) policy weight write must be on
        disk before begin_finalization. In-flight generation continues, while
        the short dataloader/ledger/TQ cut blocks new batch reservation and
        completed-row publication. ``val_metrics`` stays None until SC grows a
        validation loop.
        """
        save_state = self._save_state
        save_state["current_step"] = self._train_steps
        save_state["total_steps"] = self._train_steps
        save_state["trainer_version"] = self._trainer_version
        save_state["consumed_samples"] = self._consumed_samples
        save_state["total_valid_tokens"] = self._total_valid_tokens
        if val_metrics is not None:
            save_state["val_reward"] = val_metrics["accuracy"]
        elif "val_reward" in save_state:
            del save_state["val_reward"]

        full_metric_name = self._master_config.checkpointing["metric_name"]
        if full_metric_name is not None:
            assert full_metric_name.startswith("train:") or full_metric_name.startswith(
                "val:"
            ), (
                f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                f" e.g. 'val_reward --> 'val:accuracy'"
            )
            prefix, metric_name = full_metric_name.split(":", 1)
            metrics_source = step_metrics if prefix == "train" else val_metrics
            if not metrics_source:
                warnings.warn(
                    f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                    "This checkpoint will not be saved as top-k.",
                    stacklevel=2,
                )
                if full_metric_name in save_state:
                    del save_state[full_metric_name]
            elif metric_name not in metrics_source:
                raise ValueError(f"Metric {metric_name} not found in {prefix} metrics")
            else:
                save_state[full_metric_name] = metrics_source[metric_name]

        # Flush the previous checkpoint's background finalization first;
        # re-raises a failure from it.
        await asyncio.to_thread(self._checkpointer.finalize_pending)

        print(f"Saving checkpoint for step {self._train_steps}...")
        # The rollout pump advances the dataloader and reserves durable lineage
        # under the mutation side of this barrier. Capturing the cursor, ledger,
        # and TQ under the exclusive side makes them one restart-consistent cut.
        async with self._data_plane_checkpoint_barrier.checkpoint():
            save_state["current_epoch"] = self._current_epoch
            checkpoint_path = await asyncio.to_thread(
                self._checkpointer.init_tmp_checkpoint,
                self._train_steps,
                save_state,
                self._master_config,
            )
            cut = await self._capture_rollout_checkpoint_cut(checkpoint_path)

        # With async_save this returns after D2H staging; disk writes finish
        # in the background. New rollout dispatch may resume after the durable
        # data-plane cut because the trainer is not mutated during this save.
        await asyncio.to_thread(
            self._trainer.save_checkpoint,
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer")
            if self._checkpointer.save_optimizer
            else None,
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            checkpointing_cfg=self._master_config.checkpointing,
        )
        await self._write_rollout_checkpoint_sidecars(
            checkpoint_path,
            cut,
            include_config=False,
        )
        # Rename happens in the background once the async weight writes
        # finish; flushed at the next save or on exit.
        self._checkpointer.begin_finalization(
            checkpoint_path,
            wait_fn=self._trainer.finalize_async_save,
        )
        _write_latest_checkpoint_status(
            self._checkpointer, last_checkpoint_step=self._train_steps
        )
        self._last_rollout_snapshot_mutation_version = cut.mutation_version

    async def _sync_weights(
        self,
        *,
        calibration_data: Optional[BatchedDataDict[Any]] = None,
    ) -> None:
        """Pause new rollout dispatches, synchronize weights, resume.

        SC owns the pause gate; in-flight generations continue through the
        refit — vLLM V1 async engine supports weight updates during pending
        requests.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. Optionally calibrate FP8 KV-cache scales.
          3. weight_synchronizer.sync_weights(kv_scales=...)
          4. _rollout_permitted.set()   — resume
        """
        self._rollout_permitted.clear()

        # TODO(#2625): Add drain-gate support during refit.

        t0 = time.monotonic()
        kv_scales = None
        if (
            getattr(self._gen, "requires_kv_scale_sync", False)
            and calibration_data is not None
        ):
            print("▶ Computing KV cache scales...", flush=True)
            calibration_result = await asyncio.to_thread(
                self._trainer.calibrate_qkv_fp8_scales,
                calibration_data,
                include_q=True,
            )
            kv_scales = calibration_result["layers"]

        await asyncio.to_thread(
            self._weight_synchronizer.sync_weights,
            kv_scales=kv_scales,
        )
        if self._async_cfg.recompute_kv_cache_after_weight_updates:
            self._gen.invalidate_kv_cache()
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        if self._master_config.token_capture.enabled:
            # Rotate the version vLLM workers stamp on captured model calls
            # (per-call tagging; group staleness = min over the group's calls).
            await asyncio.to_thread(
                self._gen.set_rollout_weight_version, self._trainer_version
            )
        self._rollout_permitted.set()

    async def _log_gate_metrics(self) -> None:
        """Log the capture gate's cumulative § 8 counters + token_in_rate.

        Counters are cumulative over the run; ``gate/token_in_rate`` is the
        cumulative marker-hit rate over all admitted model calls. Fetch
        failures are logged and swallowed — metrics must never kill a step.
        """
        try:
            counters = await self._rollout_manager.gate_metrics()
        except (RuntimeError, OSError) as error:
            print(f"gate metrics fetch failed: {error}", flush=True)
            return
        if not counters:
            return
        calls = counters["token_in"] + sum(
            v for k, v in counters.items() if k.startswith("fallback_")
        )
        gate_metrics: dict[str, float] = {k: float(v) for k, v in counters.items()}
        if calls:
            gate_metrics["token_in_rate"] = counters["token_in"] / calls
        self._logger.log_metrics(gate_metrics, step=self._train_steps, prefix="gate")
        print(f"gate_metrics={gate_metrics}", flush=True)

    async def _advantage_stage(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.
        """
        if self._advantage_estimator is None:
            return meta
        adv_cfg = self._advantage_cfg

        data = await call_data_plane(
            self._dp_client,
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        prompt_ids = tensor_field(data, adv_cfg.prompt_ids_field)
        rewards = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.reward_field)
        ).float()
        token_mask = tensor_field(data, adv_cfg.token_mask_field).float()
        sample_mask = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.sample_mask_field)
        ).float()
        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in adv_cfg.repeated_batch_fields:
            repeated_batch[field_name] = squeeze_trailing_unit_dim(
                tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if self._policy_logprobs_required:
            kwargs["logprobs_policy"] = tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
        if self._reference_logprobs_required:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )

        advantages = self._advantage_estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            repeated_batch=repeated_batch,
            # Real validity (token-capture placeholders carry sample_mask 0)
            # instead of the hardwired all-ones — § 9.1, advantage_estimator.
            valid_mask=sample_mask,
            **kwargs,
        )
        response_advantages = torch.masked_select(advantages, mask.bool())
        self._step_log_dict["rewards"].append(rewards.detach().cpu())
        self._step_log_dict["masked_advantages"].append(
            response_advantages.detach().cpu()
        )

        await call_data_plane(
            self._dp_client,
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(
                meta,
                {adv_cfg.output_field: advantages},
            ),
        )
        return meta.with_fields([adv_cfg.output_field])

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        adv_cfg = self._advantage_cfg
        fields = [
            adv_cfg.prompt_ids_field,
            adv_cfg.reward_field,
            adv_cfg.token_mask_field,
            adv_cfg.sample_mask_field,
            *adv_cfg.repeated_batch_fields,
        ]
        if self._policy_logprobs_required:
            fields.append(adv_cfg.policy_logprobs_field)
        if self._reference_logprobs_required:
            fields.append(adv_cfg.reference_logprobs_field)
        return list(dict.fromkeys(fields))
