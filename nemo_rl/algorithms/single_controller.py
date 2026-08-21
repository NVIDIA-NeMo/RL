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

CPU-only Ray actor that runs two concurrent pumps plus a watchdog, and
coordinates the other actors via lightweight RPCs. SC sends control signals
and reads metadata only — model tensors still move through DataPlane or NCCL.

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
import math
import os
import shutil
import statistics
import time
import warnings
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import ray
import torch
import yaml

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    CHECKPOINT_MUTATION_KINDS,
    DataPlaneCheckpointBarrier,
    TQReplayGroupMetadata,
    TQReplayMetadataState,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    CheckpointDispatchSampler,
    TwoPhaseAdmissionSampler,
    create_sampler,
)
from nemo_rl.algorithms.grpo import GRPOSaveState, _write_latest_checkpoint_status
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
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
from nemo_rl.data_plane import DATA_PLANE_CHECKPOINT_SCHEMA_VERSION, KVBatchMeta
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.data_plane.schema import DP_CALIB_INPUT_FIELDS
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.failures import RolloutStall
from nemo_rl.experience.rollout_manager import RolloutOutcome
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    PromptGroupRecoverySummary,
    PromptGroupStatus,
)
from nemo_rl.experience.route_plan import decode_route_plan
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager, PathLike
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import TimeoutChecker, Timer

if TYPE_CHECKING:
    from nemo_rl.experience.finalizer_actor import FinalizationRequest

Generation = Union[VllmGeneration, SGLangGeneration, MegatronGeneration]

_TELEMETRY_WALL_TIME_METRIC = "telemetry/wall_time_seconds"
_TELEMETRY_PREFIXES = (
    "timing/rollout_checkpoint",
    "timing/rollout_recovery",
    "rollout/checkpoint_outcome",
    "rollout/throughput",
)


@dataclass(frozen=True)
class _RolloutWorkItem:
    """One capacity-owned prompt group awaiting generation."""

    prompt: DatumSpec
    target_step: Optional[int]
    group_id: Optional[str]


@dataclass(frozen=True)
class _RolloutCheckpointCut:
    """Controller sidecars captured with one native TQ snapshot."""

    dataloader_state: dict[str, Any]
    replay_metadata: Optional[TQReplayMetadataState]
    rollout_recovery_payload: Optional[bytes]
    rollout_recovery_group_count: Optional[int]
    rolled_back_train_group_count: int
    mutation_version: int
    tq_save_seconds: float


def _latest_vllm_values(metrics: dict[str, Any], metric_name: str) -> dict[int, float]:
    """Return the latest value keyed by vLLM data-parallel worker."""
    per_worker = metrics.get(metric_name)
    if not isinstance(per_worker, dict):
        return {}
    return {
        worker_id: float(values[-1])
        for worker_id, values in per_worker.items()
        if isinstance(worker_id, int)
        and isinstance(values, list)
        and values
        and isinstance(values[-1], (int, float))
    }


def _percentile(values: list[float], quantile: float) -> float:
    """Return a deterministic nearest-rank percentile for telemetry."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns three concurrent asyncio tasks:
      - _rollout_pump:  dispatches prompts to GenerationWorkerActor
      - _train_pump:    claims DataPlane meta, trains, clears consumed rows,
                        then runs _sync_weights (drain gate + weight
                        synchronization) inline after each optimizer step
      - _watchdog_pump: publishes rollout counters and reports stalls or
                        unhealthy environments, which are the failures that
                        otherwise produce no signal at all

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        actor_args: SingleControllerActorArgs,
        setup_timing_metrics: SetupTimingMetrics,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            actor_args: Pre-built actor args from setup_single_controller.
            setup_timing_metrics: Driver-side setup timings; logged here (Logger isn't cloudpickleable).
        """
        validate_single_controller_config(master_config)

        self._advantage_cfg = AdvantageConfig()
        self._partition_id: str = actor_args.partition_id

        self._master_config = master_config
        self._async_cfg = master_config.async_rl
        self._policy_logprobs_required = not (
            master_config.loss_fn.force_on_policy_ratio
            and master_config.grpo.seq_logprob_error_threshold is None
        )
        self._reference_logprobs_required = not bool(
            master_config.grpo.skip_reference_policy_logprobs_calculation
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
        self._rollout_recovery_ledger = self._rollout_manager.recovery_ledger
        # Direct access, deliberately. A getattr default here reads as defensive but
        # buys a silent failure mode: rename or drop the field and
        # watchdog.gym_subprocess_check: true degrades to a health check that iterates
        # nothing and reports nothing -- the exact class of silent failure this work
        # exists to remove. A missing field should break loudly at construction, where
        # it costs five minutes, not quietly at hour three of a run.
        self._env_handles = actor_args.env_handles
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer
        self._finalizer_actors = list(actor_args.finalizer_actors)
        if self._finalizer_actors and self._rollout_recovery_ledger is None:
            raise RuntimeError(
                "token-capture finalizer actors require a rollout recovery ledger"
            )
        self._available_finalizers: asyncio.Queue[Any] = asyncio.Queue()
        for actor in self._finalizer_actors:
            self._available_finalizers.put_nowait(actor)
        self._active_finalizers = 0
        self._finalizer_waiters = 0
        self._finalizer_unknown_outcomes = 0
        self._finalizer_metrics_by_group: dict[str, dict[str, float]] = {}

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._logger.log_hyperparams(master_config.model_dump())
        self._logger.log_metrics(
            setup_timing_metrics.to_metrics_dict(), step=0, prefix="timing/setup"
        )
        self._timer = Timer()
        self._telemetry_started_at = time.monotonic()
        self._telemetry_sample_index = 0
        try:
            for prefix in _TELEMETRY_PREFIXES:
                self._logger.define_metric(
                    f"{prefix}/*", step_metric=_TELEMETRY_WALL_TIME_METRIC
                )
        except Exception as error:
            warnings.warn(
                "Failed to define rollout benchmark telemetry series: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
        self._throughput_sample_time: Optional[float] = None
        self._throughput_generation_tokens_by_worker: Optional[dict[int, int]] = None
        self._throughput_rollout_counters: Optional[dict[str, int]] = None
        self._rollout_telemetry_lock = asyncio.Lock()
        self._rollout_completion_durations_s: deque[float] = deque(maxlen=10_000)
        self._rollout_queue_wait_durations_s: deque[float] = deque(maxlen=10_000)
        self._last_successful_rollout_checkpoint_time: Optional[float] = None

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

        # Loaded (or initial) GRPOSaveState from setup; _get_grpo_save_state
        # already defaulted any fields missing from older checkpoints.
        self._save_state: GRPOSaveState = actor_args.save_state
        self._last_checkpoint_path: Optional[str] = actor_args.last_checkpoint_path
        self._data_plane_checkpoint_metadata: Optional[dict[str, Any]] = (
            actor_args.data_plane_checkpoint_metadata
        )
        self._rollout_checkpoint_load_metrics = (
            actor_args.rollout_checkpoint_load_metrics
        )
        self._consumed_samples: int = actor_args.save_state.consumed_samples
        self._total_valid_tokens: int = actor_args.save_state.total_valid_tokens

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = actor_args.train_cluster
        self._inference_cluster = actor_args.inference_cluster

        restored_trainer_version = (
            actor_args.save_state.trainer_version
            if actor_args.save_state.trainer_version is not None
            else actor_args.save_state.current_step
        )
        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        self._sampler = create_sampler(self._buffer, self._async_cfg.sampler)
        self._sampler.set_dispatch_index(restored_trainer_version)
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
        # Full trainer checkpoints and lightweight rollout snapshots share a
        # filesystem namespace and must never publish concurrently.
        self._checkpoint_save_lock = asyncio.Lock()
        self._last_rollout_snapshot_mutation_version: Optional[int] = None
        self._last_missing_rollout_snapshot_anchor: Optional[tuple[int, int]] = None
        self._bootstrap_fingerprint = actor_args.bootstrap_fingerprint
        self._rollout_checkpoint_stop_requested = asyncio.Event()
        # Narrow unsafe window after optimizer apply and before SC publishes
        # the matching trainer counters. Streaming accumulation itself remains
        # snapshot-safe and is represented by the ledger's open train step.
        self._optimizer_commit_in_progress = False
        if self._buffer is not None:
            self._buffer.set_data_plane_checkpoint_barrier(
                self._data_plane_checkpoint_barrier
            )
        if self._rollout_recovery_ledger is not None:
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
        # Restored groups run beside the normal pumps. Dataloader exhaustion is
        # not terminal until this source has also drained.
        self._rollout_recovery_complete: asyncio.Event = asyncio.Event()
        self._rollout_recovery_complete.set()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0
        self._rollout_slot_waiters: int = 0
        self._rollout_permitted_waiters: int = 0
        self._buffer_capacity_waiters: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        self._inflight_by_group_id: dict[str, tuple[asyncio.Task[None], int]] = {}

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )
        # Recovery and fresh dispatch share one generation budget. Restored
        # groups reserve buffer ownership first, then compete for worker slots
        # without blocking genuinely spare capacity from serving fresh prompts.
        self._rollout_slots: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_inflight_prompts
        )

        self._trainer_version: int = restored_trainer_version
        self._train_steps: int = actor_args.save_state.current_step
        self._current_epoch: int = actor_args.save_state.current_epoch
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

        replay_restore_started = time.monotonic()
        restored_replay_groups = await self._maybe_restore_replay_buffer()
        replay_restore_seconds = time.monotonic() - replay_restore_started
        recovery_prepare_started = time.monotonic()
        recovery_groups = await self._prepare_rollout_recovery(
            restored_replay_groups=restored_replay_groups
        )
        recovery_prepare_seconds = time.monotonic() - recovery_prepare_started
        if self._rollout_checkpoint_load_metrics is not None:
            load_metrics = dict(self._rollout_checkpoint_load_metrics)
            load_metrics["replay_metadata_load_seconds"] = replay_restore_seconds
            load_metrics["recovery_prepare_seconds"] = recovery_prepare_seconds
            load_metrics["total_load_seconds"] = sum(load_metrics.values())
            self._log_telemetry_metrics(
                load_metrics,
                step=self._train_steps,
                prefix="timing/rollout_checkpoint",
            )

        telemetry_interval_s = (
            self._master_config.rollout_checkpointing.telemetry_interval_s
        )
        if telemetry_interval_s is not None:
            # Establish a process-local rate baseline before restored and fresh
            # work begin competing for the same generation capacity.
            await self._log_rollout_throughput_metrics(emit=False)

        # Start restored work and normal pumps together. Preparation above has
        # already reconstructed sampler state and reserved restored capacity,
        # so fresh dispatch can use only genuine headroom.
        recovery_task = (
            asyncio.create_task(self._recover_prepared_rollout_groups(recovery_groups))
            if recovery_groups
            else None
        )
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        watchdog_task = asyncio.create_task(self._watchdog_pump())
        rollout_checkpoint_task = (
            asyncio.create_task(self._rollout_checkpoint_pump())
            if self._master_config.rollout_checkpointing.interval_s is not None
            else None
        )
        rollout_telemetry_task = (
            asyncio.create_task(self._rollout_telemetry_pump())
            if telemetry_interval_s is not None
            else None
        )
        tasks = {rollout_task, train_task, watchdog_task}
        if recovery_task is not None:
            tasks.add(recovery_task)
        if rollout_checkpoint_task is not None:
            tasks.add(rollout_checkpoint_task)
        if rollout_telemetry_task is not None:
            tasks.add(rollout_telemetry_task)
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    # Propagate recovery, rollout, training, and watchdog
                    # failures as soon as their owning task reports them.
                    await task
                if train_task in done:
                    # A max-step exit can happen before recovery drains. Do not
                    # leave restored lineage half-mutated merely because enough
                    # other groups completed the requested training steps.
                    if recovery_task is not None and not recovery_task.done():
                        rollout_task.cancel()
                        await recovery_task
                    break
                if (
                    rollout_checkpoint_task is not None
                    and rollout_checkpoint_task in done
                ):
                    if self._rollout_checkpoint_stop_requested.is_set():
                        break
                    raise RuntimeError(
                        "rollout checkpoint pump exited without requesting stop"
                    )
                if (
                    rollout_telemetry_task is not None
                    and rollout_telemetry_task in done
                ):
                    raise RuntimeError("rollout telemetry pump exited unexpectedly")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for actor in self._finalizer_actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as error:
                    print(f"finalizer actor termination failed: {error}", flush=True)
            try:
                self._weight_synchronizer.shutdown()
            except Exception as e:  # teardown must not mask the original failure
                print(f"Error during weight-synchronizer shutdown: {e}", flush=True)
            finally:
                self._logger.finish()
                # Flush the last checkpoint's background finalization before exit.
                await asyncio.to_thread(self._checkpointer.shutdown)

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
            "active_finalizers": self._active_finalizers,
            "finalizer_waiters": self._finalizer_waiters,
            "finalizer_queue_depth": self._available_finalizers.qsize(),
            "finalizer_unknown_outcomes": self._finalizer_unknown_outcomes,
        }

    def _log_telemetry_metrics(
        self, metrics: dict[str, float], *, step: int, prefix: str
    ) -> None:
        """Log benchmark telemetry on an axis independent of trainer steps."""
        try:
            self._telemetry_sample_index += 1
            event_metrics = dict(metrics)
            event_metrics["train_step"] = float(step)
            event_metrics["sample_index"] = float(self._telemetry_sample_index)
            event_metrics[_TELEMETRY_WALL_TIME_METRIC] = (
                time.monotonic() - self._telemetry_started_at
            )
            self._logger.log_metrics(
                event_metrics,
                step=self._telemetry_sample_index,
                prefix=prefix,
                step_metric=_TELEMETRY_WALL_TIME_METRIC,
            )
        except Exception as error:
            warnings.warn(
                f"Failed to log {prefix} telemetry: {type(error).__name__}: {error}",
                stacklevel=2,
            )

    def _record_rollout_timing(
        self, *, work_started: float, dispatch_started: float
    ) -> None:
        """Record one committed group's queue and execution durations."""
        if not hasattr(self, "_rollout_queue_wait_durations_s"):
            self._rollout_queue_wait_durations_s = deque(maxlen=10_000)
        if not hasattr(self, "_rollout_completion_durations_s"):
            self._rollout_completion_durations_s = deque(maxlen=10_000)
        self._rollout_queue_wait_durations_s.append(dispatch_started - work_started)
        self._rollout_completion_durations_s.append(time.monotonic() - dispatch_started)

    # ── internal helpers ───────────────────────────────────────────────────

    async def _maybe_restore_replay_buffer(self) -> int:
        """Restore the local replay index for the native TQ checkpoint.

        Recovery is authoritative only for samplers that explicitly support
        buffered-group restoration. The native snapshot and metadata sidecar
        must both be present and agree on their manifest and group count.
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
            expected_group_size=self._master_config.grpo.num_generations_per_prompt,
            expected_manifest_digest=expected_manifest_digest_value,
        )
        await self._validate_replay_inventory(buffer_state)

        # Each buffered group holds one _buffer_capacity permit. Restore fails
        # above if the saved group count exceeds current capacity.
        assert restored <= self._async_cfg.max_buffered_rollouts
        for _ in range(restored):
            await self._buffer_capacity.acquire()
        return restored

    async def _ray_get(self, obj_ref: Any) -> Any:
        """Await a Ray ObjectRef without blocking the asyncio event loop."""
        return await obj_ref

    async def _call_dp(
        self,
        method_name: str,
        *,
        offload_sync: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Call a local DataPlaneClient or a Ray actor exposing its methods.

        Args:
            method_name: DataPlaneClient method to invoke.
            offload_sync: Run a synchronous local implementation in a worker
                thread. Use for blocking filesystem or RPC operations; Ray
                methods are already asynchronous and ignore this setting.
            **kwargs: Keyword arguments forwarded to the data-plane method.

        Returns:
            The method result after awaiting Ray or coroutine results.
        """
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await self._ray_get(remote(**kwargs))
        if offload_sync:
            result = await asyncio.to_thread(method, **kwargs)
        else:
            result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @staticmethod
    def _request_staging_keys(request: "FinalizationRequest") -> list[str]:
        """Return the full receipt-manifest staging ownership for a request."""
        keys: list[str] = []
        for receipt in request.receipts:
            if receipt is None:
                continue
            manifest = receipt.get("manifest")
            if not isinstance(manifest, list):
                continue
            for record in manifest:
                if isinstance(record, dict) and isinstance(
                    record.get("staging_key"), str
                ):
                    keys.append(record["staging_key"])
        return list(dict.fromkeys(keys))

    async def _cleanup_known_finalization_request_unlocked(
        self, request: "FinalizationRequest"
    ) -> None:
        """Clear known ownership while holding a barrier mutation slot."""
        errors: list[BaseException] = []
        try:
            await self._call_dp(
                "clear_samples",
                sample_ids=list(request.canonical_sample_ids),
                partition_id=self._partition_id,
            )
        except Exception as error:
            errors.append(
                RuntimeError(
                    "pre-publication canonical cleanup failed for "
                    f"group={request.group_id!r}, "
                    f"ids={request.canonical_sample_ids!r}"
                )
            )
            errors[-1].__cause__ = error
        staging_keys = self._request_staging_keys(request)
        if staging_keys:
            try:
                await self._call_dp(
                    "clear_samples",
                    sample_ids=staging_keys,
                    partition_id=self._master_config.token_capture.staging_partition,
                )
            except Exception as error:
                errors.append(
                    RuntimeError(
                        "pre-publication staging cleanup failed for "
                        f"group={request.group_id!r}, keys={staging_keys!r}"
                    )
                )
                errors[-1].__cause__ = error
        if errors:
            raise BaseExceptionGroup(
                f"known-outcome cleanup failed for group {request.group_id}",
                errors,
            )
        self._buffer.abort(request.group_id)
        if (
            self._rollout_recovery_ledger is not None
            and request.group_id in self._rollout_recovery_ledger
        ):
            self._rollout_recovery_ledger.discard_group(request.group_id)

    async def _cleanup_known_finalization_request(
        self, request: "FinalizationRequest"
    ) -> None:
        """Clear known request ownership without racing a TQ checkpoint."""
        async with self._data_plane_checkpoint_barrier.mutation("group_removals"):
            await self._cleanup_known_finalization_request_unlocked(request)

    async def _finalize_with_actor(self, request: "FinalizationRequest") -> None:
        """Finalize and index one group atomically with respect to TQ saves."""
        self._finalizer_waiters += 1
        queue_depth = max(
            0,
            self._finalizer_waiters - self._available_finalizers.qsize(),
        )
        queue_start = time.perf_counter()
        try:
            actor = await self._available_finalizers.get()
        except asyncio.CancelledError:
            await self._cleanup_known_finalization_request(request)
            raise
        finally:
            self._finalizer_waiters -= 1
        queue_wait_ms = (time.perf_counter() - queue_start) * 1000.0
        self._active_finalizers += 1
        active_actor_count = self._active_finalizers
        finalize_start = time.perf_counter()
        rpc_submitted = False
        actor_reusable = False
        try:
            # The actor publishes canonical rows before returning metadata.
            # Hold one local mutation slot across both that remote write and
            # the matching replay-index transition. A checkpoint therefore
            # observes either the complete group or neither half of it while
            # tensor payloads remain outside the SingleController process.
            async with self._data_plane_checkpoint_barrier.mutation("group_commits"):
                ledger = self._rollout_recovery_ledger
                assert ledger is not None
                ledger.mark_finalization_started(request.group_id)
                try:
                    rpc_submitted = True
                    finalized = await actor.finalize.remote(request)
                except BaseException:
                    self._finalizer_unknown_outcomes += 1
                    ledger.mark_finalization_unknown(request.group_id)
                    print(
                        "FATAL: finalizer actor RPC failed after submission; canonical "
                        f"publication outcome is unknown for group {request.group_id}. "
                        "Stopping validation without actor replacement or retry.",
                        flush=True,
                    )
                    raise
                else:
                    actor_reusable = True

                if finalized.dropped:
                    try:
                        await self._cleanup_known_finalization_request_unlocked(request)
                    except BaseException as cleanup_error:
                        raise RuntimeError(
                            "finalizer dropped the group and known-key cleanup failed "
                            f"for group {request.group_id}"
                        ) from cleanup_error
                    raise RuntimeError(
                        f"token capture: group {request.group_id} dropped "
                        "(min_valid_fraction_per_group)"
                    )
                if finalized.meta is None:
                    try:
                        await self._cleanup_known_finalization_request_unlocked(request)
                    except BaseException as cleanup_error:
                        raise RuntimeError(
                            "finalizer returned no metadata and known-key cleanup failed "
                            f"for group {request.group_id}"
                        ) from cleanup_error
                    raise RuntimeError(
                        "finalizer returned no metadata for non-dropped group "
                        f"{request.group_id}"
                    )
                try:
                    await self._buffer.commit_finalized(
                        request.group_id,
                        finalized.meta,
                        finalized.group_min_wv,
                        finalized.group_max_wv,
                        staging_keys=finalized.staging_keys,
                    )
                except BaseException as commit_error:
                    try:
                        await self._cleanup_known_finalization_request_unlocked(request)
                    except BaseException as cleanup_error:
                        raise BaseExceptionGroup(
                            "finalizer commit and known-key cleanup failed for "
                            f"group {request.group_id}",
                            [commit_error, cleanup_error],
                        )
                    raise
                ledger.mark_group_finalized(
                    request.group_id,
                    meta=finalized.meta,
                    group_min_weight_version=finalized.group_min_wv,
                    group_max_weight_version=finalized.group_max_wv,
                )
                self._rollout_manager.record_canonical_publication(
                    finalized.canonical_output_tokens
                )
        finally:
            self._active_finalizers -= 1
            # Cancellation before RPC submission leaves the actor untouched;
            # a failed submitted RPC has unknown publication state and must
            # not be reused. Every successful RPC makes the actor reusable,
            # including when later validation or cleanup rejects the group.
            if actor_reusable or not rpc_submitted:
                self._available_finalizers.put_nowait(actor)
        finalize_total_ms = (time.perf_counter() - finalize_start) * 1000.0
        finalized.metrics.update(
            {
                "finalize/queue_wait_ms": queue_wait_ms,
                "finalize/total_ms": finalize_total_ms,
                "finalize/queue_depth": float(queue_depth),
                "finalize/active_actor_count": float(active_actor_count),
            }
        )
        self._finalizer_metrics_by_group[request.group_id] = dict(finalized.metrics)

    async def _cleanup_consumed_metas_unlocked(self, metas: list[KVBatchMeta]) -> None:
        """Clear consumed ownership while holding a barrier mutation slot."""
        canonical_by_partition: dict[str, list[str]] = {}
        staging_by_partition: dict[str, list[str]] = {}
        for meta in metas:
            canonical_by_partition.setdefault(meta.partition_id, []).extend(
                meta.sample_ids
            )
            for tag in meta.tags or []:
                encoded_plan = tag.get(ROUTE_PLAN_TAG)
                if encoded_plan is None:
                    continue
                plan = decode_route_plan(encoded_plan)
                staging_by_partition.setdefault(plan.staging_partition, []).extend(
                    plan.cleanup_staging_keys
                )

        errors: list[BaseException] = []
        for partition_id, sample_ids in canonical_by_partition.items():
            unique_ids = list(dict.fromkeys(sample_ids))
            try:
                await self._call_dp(
                    "clear_samples",
                    sample_ids=unique_ids,
                    partition_id=partition_id,
                )
            except Exception as error:
                cleanup_error = RuntimeError(
                    "post-train canonical cleanup failed: "
                    f"partition={partition_id!r}, sample_ids={unique_ids!r}"
                )
                cleanup_error.__cause__ = error
                errors.append(cleanup_error)
        for partition_id, staging_keys in staging_by_partition.items():
            unique_keys = list(dict.fromkeys(staging_keys))
            try:
                await self._call_dp(
                    "clear_samples",
                    sample_ids=unique_keys,
                    partition_id=partition_id,
                )
            except Exception as error:
                cleanup_error = RuntimeError(
                    "post-train staging cleanup failed: "
                    f"partition={partition_id!r}, staging_keys={unique_keys!r}"
                )
                cleanup_error.__cause__ = error
                errors.append(cleanup_error)
        if errors:
            raise BaseExceptionGroup("post-train DataPlane cleanup failed", errors)

    async def _cleanup_consumed_metas(self, metas: list[KVBatchMeta]) -> None:
        """Clear consumed rows without racing a native TQ checkpoint."""
        async with self._data_plane_checkpoint_barrier.mutation("sample_clears"):
            await self._cleanup_consumed_metas_unlocked(metas)

    async def _validate_replay_inventory(
        self, replay_metadata: TQReplayMetadataState
    ) -> None:
        """Require the canonical TQ keys to match the SC replay index exactly.

        Live checkpoint callers must hold the exclusive data-plane barrier so
        commits and clears cannot race this inventory read. Restore calls are
        also safe before the rollout and train pumps start any live writers.
        """
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
        self, *, clear_unreferenced: bool
    ) -> None:
        """Require every unfinished receipt or deferred route to retain staging."""
        ledger = self._rollout_recovery_ledger
        if ledger is None:
            return
        expected_staging_keys = ledger.expected_staging_keys()
        staging_partition = self._master_config.token_capture.staging_partition
        actual_staging_keys = set(
            await self._call_dp(
                "list_sample_ids",
                partition_id=staging_partition,
            )
        )
        missing = sorted(expected_staging_keys - actual_staging_keys)
        if missing:
            raise RuntimeError(
                "rollout-recovery ownership references staging rows missing "
                f"from live TQ state: missing={missing[:10]!r} "
                f"(total={len(missing)})"
            )
        unreferenced = sorted(actual_staging_keys - expected_staging_keys)
        if clear_unreferenced and unreferenced:
            await self._call_dp(
                "clear_samples",
                sample_ids=unreferenced,
                partition_id=staging_partition,
            )
            print(
                "rollout recovery cleared unreferenced staging rows: "
                f"count={len(unreferenced)}",
                flush=True,
            )
        print(
            "📦 Rollout-recovery staging inventory validated: "
            f"referenced={len(expected_staging_keys)}",
            flush=True,
        )

    async def _prepare_rollout_recovery(
        self, *, restored_replay_groups: int
    ) -> list[PromptGroupRecoverySummary]:
        """Validate restored lineage and reclaim its durable buffer permits."""
        self._rollout_recovery_complete.set()
        metadata = self._data_plane_checkpoint_metadata or {}
        if "rollout_recovery_payload_sha256" not in metadata:
            self._restore_sampler_dispatch_state(
                [], restored_replay_groups=restored_replay_groups
            )
            return []
        ledger = self._rollout_recovery_ledger
        if ledger is None:
            raise RuntimeError(
                "native TQ checkpoint advertises rollout recovery, but token "
                "capture did not construct a recovery ledger"
            )
        ledger.prepare_for_restart()
        groups = ledger.summaries()
        unfinished = [
            group
            for group in groups
            if group.status
            in {PromptGroupStatus.GENERATING, PromptGroupStatus.READY_TO_FINALIZE}
        ]
        finalized_group_ids = {
            group.group_id
            for group in groups
            if group.status == PromptGroupStatus.FINALIZED
        }
        restored_group_ids = set(self._buffer.group_ids)
        if restored_group_ids != finalized_group_ids:
            raise RuntimeError(
                "restored replay groups do not match finalized rollout lineage: "
                f"replay_only={sorted(restored_group_ids - finalized_group_ids)!r}, "
                f"lineage_only={sorted(finalized_group_ids - restored_group_ids)!r}"
            )
        self._restore_sampler_dispatch_state(
            unfinished,
            restored_replay_groups=restored_replay_groups,
        )
        await self._validate_rollout_recovery_inventory(clear_unreferenced=True)
        stale_unfinished = [
            group
            for group in unfinished
            if self._sampler.should_abort_inflight(
                start_weight_version=group.start_weight_version,
                current_train_weight=self._trainer_version,
            )
        ]
        stale_group_ids = {group.group_id for group in stale_unfinished}
        for group in stale_unfinished:
            await self._rollout_manager.discard_recovery_group(group.group_id)
        if stale_unfinished:
            print(
                "📦 Rollout recovery dropped stale unfinished groups: "
                f"count={len(stale_unfinished)}",
                flush=True,
            )
        unfinished = [
            group for group in unfinished if group.group_id not in stale_group_ids
        ]
        if (
            restored_replay_groups + len(unfinished)
            > self._async_cfg.max_buffered_rollouts
        ):
            raise RuntimeError(
                "restored canonical and unfinished rollout groups exceed current "
                "buffer capacity: "
                f"canonical={restored_replay_groups}, "
                f"unfinished={len(unfinished)}, "
                f"capacity={self._async_cfg.max_buffered_rollouts}"
            )
        restored_group_count = restored_replay_groups + len(unfinished)
        min_streaming_groups = self._async_cfg.min_groups_for_streaming_train
        free_capacity = self._async_cfg.max_buffered_rollouts - restored_group_count
        if (
            0 < restored_group_count < min_streaming_groups
            and free_capacity < self._master_config.grpo.num_prompts_per_step
        ):
            required_capacity = required_rollout_recovery_capacity(
                num_prompts_per_step=(self._master_config.grpo.num_prompts_per_step),
                min_groups_for_streaming_train=min_streaming_groups,
            )
            raise RuntimeError(
                "restored rollout state cannot fill a streaming batch with the "
                "configured capacity; increase async_rl.max_buffered_rollouts "
                f"to at least {required_capacity}"
            )

        # Reclaim all saved permits before fresh dataloader work can run. The
        # capacity check above makes these acquisitions immediate and prevents
        # new work from stealing ownership represented by the durable ledger.
        acquired = 0
        try:
            for _ in unfinished:
                await self._buffer_capacity.acquire()
                acquired += 1
        except BaseException:
            for _ in range(acquired):
                self._buffer_capacity.release()
            raise

        if unfinished:
            self._rollout_recovery_complete.clear()
        return unfinished

    @staticmethod
    def _release_recovery_permit_if_task_not_started(
        _: asyncio.Task[Any],
        *,
        task_started_event: asyncio.Event,
        buffer_capacity: asyncio.Semaphore,
    ) -> None:
        """Return preclaimed capacity if cancellation beats coroutine startup."""
        if not task_started_event.is_set():
            buffer_capacity.release()

    async def _recover_one_prepared_group(
        self,
        group: PromptGroupRecoverySummary,
        *,
        task_started_event: asyncio.Event,
    ) -> bool:
        """Recover one capacity-owned group through the shared worker budget."""
        task_started_event.set()
        buffer_permit_owned = True
        rollout_slot_owned = False
        counted_inflight = False
        ledger = self._rollout_recovery_ledger
        assert ledger is not None
        recovery_record = ledger.get_group(group.group_id)
        reused_siblings = len(recovery_record.sealed_generation_indices)
        redispatched_siblings = recovery_record.expected_generations - reused_siblings
        work_started = time.monotonic()
        try:
            self._rollout_slot_waiters = getattr(self, "_rollout_slot_waiters", 0) + 1
            try:
                await self._rollout_slots.acquire()
            finally:
                self._rollout_slot_waiters -= 1
            rollout_slot_owned = True
            self._rollout_permitted_waiters = (
                getattr(self, "_rollout_permitted_waiters", 0) + 1
            )
            try:
                await self._rollout_permitted.wait()
            finally:
                self._rollout_permitted_waiters -= 1
            dispatch_started = time.monotonic()
            self._inflight_rollouts += 1
            counted_inflight = True

            request = await self._rollout_manager.recover_for_finalization(
                group.group_id,
                inflight_registry=self._inflight_by_group_id,
            )

            # Generation capacity protects vLLM/Gym work, not the independent
            # finalizer pool. Release it as soon as token capture is complete.
            self._inflight_rollouts -= 1
            counted_inflight = False
            self._rollout_slots.release()
            rollout_slot_owned = False

            if request is None:
                return False
            await self._finalize_with_actor(request)
            self._rollout_manager.record_recovery_siblings(
                reused=reused_siblings,
                redispatched=redispatched_siblings,
            )
            self._record_rollout_timing(
                work_started=work_started,
                dispatch_started=dispatch_started,
            )
            print(
                "rollout recovery finalized group: "
                f"group={group.group_id} reused={reused_siblings} "
                f"redispatched={redispatched_siblings}",
                flush=True,
            )
            buffer_permit_owned = False
            return True
        finally:
            if counted_inflight:
                self._inflight_rollouts -= 1
            if rollout_slot_owned:
                self._rollout_slots.release()
            if buffer_permit_owned:
                self._buffer_capacity.release()

    async def _recover_prepared_rollout_groups(
        self, groups: list[PromptGroupRecoverySummary]
    ) -> None:
        """Recover restored groups concurrently beside fresh rollout dispatch."""
        recovery_started = time.monotonic()
        counters_before = self._rollout_manager.telemetry_snapshot()
        recovery_tasks: list[asyncio.Task[bool]] = []
        try:
            async with asyncio.TaskGroup() as task_group:
                for group in groups:
                    task_started_event = asyncio.Event()
                    task = task_group.create_task(
                        self._recover_one_prepared_group(
                            group,
                            task_started_event=task_started_event,
                        )
                    )
                    task.add_done_callback(
                        partial(
                            self._release_recovery_permit_if_task_not_started,
                            task_started_event=task_started_event,
                            buffer_capacity=self._buffer_capacity,
                        )
                    )
                    recovery_tasks.append(task)

            completed = sum(task.result() for task in recovery_tasks)
            print(
                "📦 Rollout recovery completed: "
                f"groups={len(groups)}, finalized={completed}, "
                f"dropped={len(groups) - completed}",
                flush=True,
            )
            counters_after = self._rollout_manager.telemetry_snapshot()
            self._log_telemetry_metrics(
                {
                    "total_recovery_seconds": time.monotonic() - recovery_started,
                    "groups_considered": float(len(groups)),
                    "groups_finalized": float(completed),
                    "groups_dropped": float(len(groups) - completed),
                    "siblings_reused": float(
                        counters_after["recovery_siblings_reused"]
                        - counters_before["recovery_siblings_reused"]
                    ),
                    "siblings_redispatched": float(
                        counters_after["recovery_siblings_redispatched"]
                        - counters_before["recovery_siblings_redispatched"]
                    ),
                },
                step=self._train_steps,
                prefix="timing/rollout_recovery",
            )
        finally:
            self._rollout_recovery_complete.set()

    async def _maybe_restore_rollout_recovery(
        self, *, restored_replay_groups: int
    ) -> None:
        """Compatibility helper for focused recovery callers and tests."""
        groups = await self._prepare_rollout_recovery(
            restored_replay_groups=restored_replay_groups
        )
        await self._recover_prepared_rollout_groups(groups)

    def _restore_sampler_dispatch_state(
        self,
        unfinished_groups: list[PromptGroupRecoverySummary],
        *,
        restored_replay_groups: int,
    ) -> None:
        """Reconstruct gated admission from canonical and unfinished groups."""
        if not self._sampler.supports_buffer_checkpoint:
            return
        if not isinstance(self._sampler, CheckpointDispatchSampler):
            raise RuntimeError(
                f"checkpoint-capable sampler {type(self._sampler).__name__} "
                "does not implement dispatch-state recovery"
            )
        restored_target_steps = list(self._buffer.target_step_list)
        if len(restored_target_steps) != restored_replay_groups:
            raise RuntimeError(
                "restored replay group count does not match the sampler target "
                f"index: groups={restored_replay_groups}, "
                f"targets={len(restored_target_steps)}"
            )
        restored_target_steps.extend(group.target_step for group in unfinished_groups)
        self._sampler.restore_dispatch_state(
            current_train_weight=self._trainer_version,
            restored_target_steps=restored_target_steps,
            groups_per_step=self._master_config.grpo.num_prompts_per_step,
        )
        stamped_targets = [
            target for target in restored_target_steps if target is not None
        ]
        if stamped_targets:
            print(
                "📦 Restored sampler dispatch state: "
                f"highest_target_step={max(stamped_targets)}",
                flush=True,
            )

    async def _clear_data_plane_samples(self, sample_ids: list[str]) -> None:
        """Clear consumed rows without overlapping a data-plane checkpoint."""
        async with self._data_plane_checkpoint_barrier.mutation("sample_clears"):
            await call_data_plane(
                self._dp_client,
                "clear_samples",
                offload_sync=True,
                sample_ids=sample_ids,
                partition_id=self._partition_id,
            )

    async def _save_data_plane_checkpoint(
        self,
        checkpoint_path: PathLike,
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
                    "rollout recovery group count is required with its digest"
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
        self,
        checkpoint_path: PathLike,
        *,
        periodic: bool,
    ) -> _RolloutCheckpointCut:
        """Save TQ and capture matching controller sidecars under the barrier.

        For a periodic cut, groups already claimed by an open optimizer step
        are added back to the persisted replay index and rolled back to
        ``FINALIZED`` in the persisted lineage. The live ledger and trainer are
        not modified; only restart semantics discard those uncommitted gradients.
        """
        dataloader_state = self._dataloader.state_dict()
        replay_metadata: Optional[TQReplayMetadataState] = None
        rollout_recovery_payload: Optional[bytes] = None
        rollout_recovery_digest: Optional[str] = None
        rollout_recovery_group_count: Optional[int] = None
        rollout_recovery_state: Optional[dict[str, Any]] = None
        ledger = self._rollout_recovery_ledger

        additional_groups: list[TQReplayGroupMetadata] = []
        if ledger is not None:
            if periodic:
                additional_groups = ledger.training_owned_replay_groups()
                rollout_recovery_state = ledger.periodic_snapshot_state_dict()
            else:
                ledger.assert_full_step_checkpoint_safe()
                rollout_recovery_state = ledger.state_dict()
            payload_buffer = io.BytesIO()
            torch.save(rollout_recovery_state, payload_buffer)
            rollout_recovery_payload = payload_buffer.getvalue()
            rollout_recovery_digest = hashlib.sha256(
                rollout_recovery_payload
            ).hexdigest()
            rollout_recovery_group_count = len(rollout_recovery_state["groups"])

        if self._sampler.supports_buffer_checkpoint:
            replay_metadata = self._buffer.metadata_state_dict(
                saved_capacity=self._async_cfg.max_buffered_rollouts,
                additional_groups=additional_groups,
            )
        if replay_metadata is not None and rollout_recovery_state is not None:
            replay_group_ids = {
                group["group_id"] for group in replay_metadata["groups"]
            }
            finalized_group_ids = {
                group["group_id"]
                for group in rollout_recovery_state["groups"]
                if group["status"] == PromptGroupStatus.FINALIZED.value
            }
            if replay_group_ids != finalized_group_ids:
                raise RuntimeError(
                    "rollout snapshot replay metadata does not match finalized "
                    "lineage ownership: "
                    f"replay_only={sorted(replay_group_ids - finalized_group_ids)!r}, "
                    f"lineage_only={sorted(finalized_group_ids - replay_group_ids)!r}"
                )
        if replay_metadata is not None:
            await self._validate_replay_inventory(replay_metadata)
        if rollout_recovery_payload is not None:
            await self._validate_rollout_recovery_inventory(clear_unreferenced=False)
        tq_save_started = time.monotonic()
        await self._save_data_plane_checkpoint(
            checkpoint_path,
            replay_metadata=replay_metadata,
            rollout_recovery_payload_sha256=rollout_recovery_digest,
            rollout_recovery_group_count=rollout_recovery_group_count,
        )
        tq_save_seconds = time.monotonic() - tq_save_started
        return _RolloutCheckpointCut(
            dataloader_state=dataloader_state,
            replay_metadata=replay_metadata,
            rollout_recovery_payload=rollout_recovery_payload,
            rollout_recovery_group_count=rollout_recovery_group_count,
            rolled_back_train_group_count=len(additional_groups),
            mutation_version=self._data_plane_checkpoint_barrier.mutation_version,
            tq_save_seconds=tq_save_seconds,
        )

    async def _write_rollout_checkpoint_sidecars(
        self,
        checkpoint_path: PathLike,
        cut: _RolloutCheckpointCut,
        *,
        include_config: bool,
    ) -> None:
        """Write controller files that belong to a captured native TQ cut."""
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
            dumped_config = self._master_config.model_dump(mode="json")

            def _write_config() -> None:
                with (checkpoint_path / "config.yaml").open("w") as config_file:
                    yaml.safe_dump(dumped_config, config_file)

            await asyncio.to_thread(_write_config)

    async def _save_rollout_checkpoint(self, *, force: bool = False) -> bool:
        """Publish one lightweight snapshot anchored to durable trainer state."""
        async with self._checkpoint_save_lock:
            if self._optimizer_commit_in_progress:
                return False
            if (
                not force
                and self._last_rollout_snapshot_mutation_version
                == self._data_plane_checkpoint_barrier.mutation_version
            ):
                return False

            save_started = time.monotonic()
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
            tmp_path, final_path, snapshot_sequence = await asyncio.to_thread(
                prepare_snapshot_paths, anchor
            )
            try:
                barrier_requested = time.monotonic()
                async with self._data_plane_checkpoint_barrier.checkpoint():
                    barrier_acquired = time.monotonic()
                    if (
                        self._optimizer_commit_in_progress
                        or self._train_steps != expected_train_step
                        or self._trainer_version != expected_trainer_version
                    ):
                        await asyncio.to_thread(shutil.rmtree, tmp_path)
                        return False
                    snapshot_epoch = self._current_epoch
                    cut = await self._capture_rollout_checkpoint_cut(
                        tmp_path,
                        periodic=True,
                    )
                barrier_released = time.monotonic()

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
                    rolled_back_train_group_count=(cut.rolled_back_train_group_count),
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
            save_completed = time.monotonic()
            checkpoint_metrics = {
                "snapshot_sequence": float(snapshot_sequence),
                "total_save_seconds": save_completed - save_started,
                "tq_save_seconds": cut.tq_save_seconds,
                "barrier_wait_seconds": barrier_acquired - barrier_requested,
                "exclusive_hold_seconds": barrier_released - barrier_acquired,
                "replay_groups": float(
                    len(cut.replay_metadata["groups"])
                    if cut.replay_metadata is not None
                    else 0
                ),
                "ledger_groups": float(cut.rollout_recovery_group_count or 0),
                "rolled_back_train_groups": float(cut.rolled_back_train_group_count),
            }
            self._log_telemetry_metrics(
                checkpoint_metrics,
                step=expected_train_step,
                prefix="timing/rollout_checkpoint",
            )
            print(
                "rollout checkpoint save completed: "
                f"{final_path} (step={expected_train_step}, "
                f"trainer_version={expected_trainer_version}, "
                f"ledger_groups={cut.rollout_recovery_group_count or 0}, "
                f"total_save_seconds={checkpoint_metrics['total_save_seconds']:.2f}, "
                "exclusive_hold_seconds="
                f"{checkpoint_metrics['exclusive_hold_seconds']:.2f})",
                flush=True,
            )
            return True

    def _log_rollout_checkpoint_outcome(
        self, *, outcome: str, attempt_duration_seconds: float
    ) -> None:
        """Record every scheduled checkpoint attempt, including no-op cuts."""
        metrics = {
            "attempt": 1.0,
            "attempt_duration_seconds": attempt_duration_seconds,
            "completed": float(outcome == "completed"),
            "skipped": float(outcome == "skipped"),
            "failed": float(outcome == "failed"),
            "configured_interval_seconds": float(
                self._master_config.rollout_checkpointing.interval_s or 0.0
            ),
        }
        if outcome == "completed":
            completed_at = time.monotonic()
            if self._last_successful_rollout_checkpoint_time is not None:
                metrics["seconds_since_previous_success"] = (
                    completed_at - self._last_successful_rollout_checkpoint_time
                )
            self._last_successful_rollout_checkpoint_time = completed_at
        self._log_telemetry_metrics(
            metrics,
            step=self._train_steps,
            prefix="rollout/checkpoint_outcome",
        )

    async def _rollout_checkpoint_pump(self) -> None:
        """Persist rollout state periodically, including during streamed train."""
        interval_s = self._master_config.rollout_checkpointing.interval_s
        if interval_s is None:
            raise RuntimeError("rollout checkpoint pump started while disabled")
        while True:
            await asyncio.sleep(interval_s)
            attempt_started = time.monotonic()
            deadline_due = self._train_steps == 0 and self._timeout.would_save()
            try:
                saved = await self._save_rollout_checkpoint(force=deadline_due)
            except Exception as error:
                self._log_rollout_checkpoint_outcome(
                    outcome="failed",
                    attempt_duration_seconds=time.monotonic() - attempt_started,
                )
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
            self._log_rollout_checkpoint_outcome(
                outcome="completed" if saved else "skipped",
                attempt_duration_seconds=time.monotonic() - attempt_started,
            )
            if deadline_due:
                if not saved:
                    continue
                if not self._timeout.check_save():
                    continue
                print(
                    "Checkpoint deadline reached before the first train step; "
                    "stopping after a durable rollout snapshot",
                    flush=True,
                )
                self._rollout_checkpoint_stop_requested.set()
                return

    async def _rollout_telemetry_pump(self) -> None:
        """Sample generation and publication throughput at a fixed cadence."""
        interval_s = self._master_config.rollout_checkpointing.telemetry_interval_s
        if interval_s is None:
            raise RuntimeError("rollout telemetry pump started while disabled")
        while True:
            await asyncio.sleep(interval_s)
            await self._log_rollout_throughput_metrics()

    @staticmethod
    def _group_ids_from_meta(meta: KVBatchMeta) -> list[str]:
        """Return stable prompt-group IDs in their canonical sample order."""
        group_ids: list[str] = []
        for sample_id in meta.sample_ids:
            if "_g" not in sample_id:
                raise ValueError(
                    f"canonical rollout sample ID has no generation suffix: "
                    f"{sample_id!r}"
                )
            group_id, generation_index = sample_id.rsplit("_g", 1)
            if not group_id or not generation_index.isdigit():
                raise ValueError(f"invalid canonical rollout sample ID: {sample_id!r}")
            if not group_ids or group_ids[-1] != group_id:
                group_ids.append(group_id)
        return group_ids

    def _claim_train_meta(self, meta: KVBatchMeta, *, num_groups: int) -> list[str]:
        """Bind a sampler selection to the ledger's current optimizer step."""
        ledger = self._rollout_recovery_ledger
        if ledger is None:
            return self._group_ids_from_meta(meta)
        group_ids = self._group_ids_from_meta(meta)
        if len(group_ids) != num_groups:
            raise ValueError(
                f"sampler selected {num_groups} groups but canonical metadata "
                f"contains {len(group_ids)} group IDs"
            )

        missing_group_ids = [
            group_id for group_id in group_ids if group_id not in ledger
        ]
        if missing_group_ids:
            raise RuntimeError(
                "sampler selected canonical rollout group(s) missing from the "
                "rollout recovery ledger: "
                f"{missing_group_ids!r}. Restore TQ, replay metadata, and rollout "
                "lineage from the same checkpoint."
            )
        ledger.claim_groups_for_training(
            group_ids,
            train_step=self._train_steps,
            trainer_version=self._trainer_version,
            expected_group_count=self._master_config.grpo.num_prompts_per_step,
        )
        return group_ids

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch:
          0. await sampler.admit(...) to wait until the batch may dispatch and
             obtain its target_step stamp.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire _rollout_slots shared with startup recovery
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Run the rollout, then either commit it directly or submit its
             metadata-only request to the finalizer actor pool.
          5. Decrement _inflight_rollouts
        """
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            recovery_group_id: Optional[str],
            task_started_event: asyncio.Event,
            dispatch_admitted_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            work_started = time.monotonic()
            generation_permit_released = False
            inflight_count_released = False
            counted_inflight = False
            ownership_transferred = False
            rollout_slot_owned = False
            try:
                self._rollout_slot_waiters = (
                    getattr(self, "_rollout_slot_waiters", 0) + 1
                )
                try:
                    await self._rollout_slots.acquire()
                finally:
                    self._rollout_slot_waiters -= 1
                rollout_slot_owned = True
                self._rollout_permitted_waiters = (
                    getattr(self, "_rollout_permitted_waiters", 0) + 1
                )
                try:
                    await self._rollout_permitted.wait()
                finally:
                    self._rollout_permitted_waiters -= 1
                dispatch_started = time.monotonic()
                dispatch_admitted_event.set()
                self._inflight_rollouts += 1
                counted_inflight = True
                if self._finalizer_actors:
                    finalization_kwargs: dict[str, Any] = {
                        "target_step": target_step,
                        "inflight_registry": self._inflight_by_group_id,
                    }
                    if recovery_group_id is not None:
                        finalization_kwargs["recovery_group_id"] = recovery_group_id
                    request = await self._rollout_manager.generate_for_finalization(
                        prompt,
                        **finalization_kwargs,
                    )
                    if request is None:
                        outcome = RolloutOutcome.SKIPPED
                    else:
                        self._inflight_rollouts -= 1
                        inflight_count_released = True
                        self._rollout_slots.release()
                        rollout_slot_owned = False
                        generation_permit_released = True
                        await self._finalize_with_actor(request)
                        self._rollout_manager.stats.committed += 1
                        outcome = RolloutOutcome.COMMITTED
                else:
                    outcome = await self._rollout_manager.generate_and_push(
                        prompt,
                        target_step=target_step,
                        inflight_registry=self._inflight_by_group_id,
                    )
                ownership_transferred = True
            except BaseException:
                # On success ownership transfers to the train pump, which
                # releases this permit after consuming the committed group.
                if not ownership_transferred:
                    self._buffer_capacity.release()
                raise
            finally:
                # Never leave the producer waiting if this task fails or is
                # cancelled before it passes the generation gates.
                dispatch_admitted_event.set()
                if counted_inflight and not inflight_count_released:
                    self._inflight_rollouts -= 1
                if not generation_permit_released and rollout_slot_owned:
                    self._rollout_slots.release()

            if outcome is RolloutOutcome.SKIPPED:
                # Nothing was committed, so the train pump will never see this group
                # and never release its permit on our behalf.
                self._buffer_capacity.release()
                return

            self._record_rollout_timing(
                work_started=work_started,
                dispatch_started=dispatch_started,
            )

            if self._async_cfg.diagnostics:
                content = ""
                for i in range(len(prompt["message_log"])):
                    if prompt["message_log"][i]["role"] == "user":
                        content = prompt["message_log"][i]["content"]
                        break
                print(f"  rollout done for prompt='{content[:20]}...'", flush=True)

        def _release_buffer_if_task_not_started(
            _: asyncio.Task[Any],
            *,
            task_started_event: asyncio.Event,
            dispatch_admitted_event: asyncio.Event,
        ) -> None:
            if not task_started_event.is_set():
                self._buffer_capacity.release()
                dispatch_admitted_event.set()

        def _release_capacity(permits: int) -> None:
            for _ in range(permits):
                self._buffer_capacity.release()

        async def _acquire_capacity(permits: int) -> None:
            acquired = 0
            try:
                for _ in range(permits):
                    self._buffer_capacity_waiters = (
                        getattr(self, "_buffer_capacity_waiters", 0) + 1
                    )
                    try:
                        await self._buffer_capacity.acquire()
                    finally:
                        self._buffer_capacity_waiters -= 1
                    acquired += 1
            except BaseException:
                _release_capacity(acquired)
                raise

        def _num_prompts_to_dispatch(
            batch_size: int, target_step: Optional[int]
        ) -> int:
            """Return only the target-step shortfall after replay restoration."""
            if target_step is None:
                return batch_size
            buffered = self._buffer.count_for_target_step(target_step)
            return max(0, batch_size - buffered)

        max_epochs = self._master_config.grpo.max_num_epochs
        token_capture_config = getattr(self._master_config, "token_capture", None)
        token_capture_enabled = bool(
            token_capture_config is not None and token_capture_config.enabled
        )
        if token_capture_enabled and not isinstance(
            self._sampler, TwoPhaseAdmissionSampler
        ):
            raise RuntimeError(
                f"token-capture sampler {type(self._sampler).__name__} must "
                "support two-phase admission"
            )

        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
                dataloader_iterator = iter(self._dataloader)
                while True:
                    preacquired_capacity = 0
                    if token_capture_enabled:
                        await self._sampler.wait_until_admissible(
                            trainer_version_fn=lambda: self._trainer_version
                        )
                        configured_batch_size = (
                            self._master_config.grpo.num_prompts_per_step
                        )
                        await _acquire_capacity(configured_batch_size)
                        preacquired_capacity = configured_batch_size
                        end_of_epoch = False
                        try:
                            # The cursor and all durable prompt reservations move
                            # together. A snapshot therefore cannot skip a batch
                            # whose lineage was not yet made recoverable.
                            async with self._data_plane_checkpoint_barrier.mutation(
                                "prompt_reservations"
                            ):
                                try:
                                    prompt_batch = next(dataloader_iterator)
                                except StopIteration:
                                    end_of_epoch = True
                                if end_of_epoch:
                                    # Keep the outer epoch counter in the same
                                    # checkpoint cut as the exhausted loader
                                    # cursor. Otherwise a snapshot can observe
                                    # end-of-epoch data with the previous epoch
                                    # number and replay the dataset on restart.
                                    self._current_epoch += 1
                                else:
                                    if prompt_batch.size > preacquired_capacity:
                                        raise RuntimeError(
                                            "dataloader batch exceeds reserved "
                                            "rollout capacity"
                                        )
                                    target_step = self._sampler.commit_admission(
                                        trainer_version=self._trainer_version
                                    )
                                    num_prompts = _num_prompts_to_dispatch(
                                        prompt_batch.size, target_step
                                    )
                                    unused_capacity = preacquired_capacity - num_prompts
                                    _release_capacity(unused_capacity)
                                    preacquired_capacity = num_prompts
                                    rollout_work_items = []
                                    for prompt_idx in range(num_prompts):
                                        prompt: DatumSpec = {  # type: ignore
                                            key: value[prompt_idx]
                                            for key, value in prompt_batch.items()
                                        }
                                        group_id = (
                                            self._rollout_manager.reserve_prompt_group(
                                                prompt,
                                                target_step=target_step,
                                            )
                                        )
                                        rollout_work_items.append(
                                            _RolloutWorkItem(
                                                prompt=prompt,
                                                target_step=target_step,
                                                group_id=group_id,
                                            )
                                        )
                        except BaseException:
                            _release_capacity(preacquired_capacity)
                            raise
                        if end_of_epoch:
                            _release_capacity(preacquired_capacity)
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
                        num_prompts = _num_prompts_to_dispatch(
                            prompt_batch.size, target_step
                        )
                        rollout_work_items = []
                        for prompt_idx in range(num_prompts):
                            prompt = {  # type: ignore
                                key: value[prompt_idx]
                                for key, value in prompt_batch.items()
                            }
                            rollout_work_items.append(
                                _RolloutWorkItem(prompt, target_step, None)
                            )

                    pending_preacquired_capacity = preacquired_capacity
                    try:
                        for work_item in rollout_work_items:
                            capacity_acquired_here = False
                            if not token_capture_enabled:
                                self._buffer_capacity_waiters = (
                                    getattr(self, "_buffer_capacity_waiters", 0) + 1
                                )
                                try:
                                    await self._buffer_capacity.acquire()
                                finally:
                                    self._buffer_capacity_waiters -= 1
                                capacity_acquired_here = True
                            task_started_event = asyncio.Event()
                            dispatch_admitted_event = asyncio.Event()
                            try:
                                task = rollout_tasks.create_task(
                                    _dispatch_one_prompt(
                                        work_item.prompt,
                                        work_item.target_step,
                                        work_item.group_id,
                                        task_started_event,
                                        dispatch_admitted_event,
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
                                    _release_buffer_if_task_not_started,
                                    task_started_event=task_started_event,
                                    dispatch_admitted_event=dispatch_admitted_event,
                                )
                            )
                            await dispatch_admitted_event.wait()
                    finally:
                        _release_capacity(pending_preacquired_capacity)

        # Drain in-flight so return implies "all rollouts in TQ".
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

        while self._train_steps < grpo_cfg.max_num_steps:
            version_during_step = self._trainer_version
            groups_dispatched = 0
            evicted_stale_prompt_groups = 0
            min_sample_version = None
            step_open = False
            generation_released = False
            calibration_batches: list[BatchedDataDict[Any]] = []
            consumed_metas: list[KVBatchMeta] = []
            consumed_group_count = 0
            step_finalizer_metrics: dict[str, list[float]] = {}

            with self._timer.time("total_step_time"):
                while groups_dispatched < grpo_cfg.num_prompts_per_step:
                    # Wait for a selectable batch
                    with self._timer.time("exposed_generation"):
                        await asyncio.sleep(0)

                        # Evict stale groups
                        if self._rollout_recovery_ledger is None:
                            evicted = await self._sampler.evict(
                                current_train_weight=self._trainer_version,
                            )
                        else:
                            async with self._data_plane_checkpoint_barrier.mutation(
                                "group_removals"
                            ):
                                group_ids_before_evict = set(self._buffer.group_ids)
                                evicted = await self._sampler.evict(
                                    current_train_weight=self._trainer_version,
                                )
                                evicted_group_ids = group_ids_before_evict - set(
                                    self._buffer.group_ids
                                )
                                for group_id in evicted_group_ids:
                                    if group_id in self._rollout_recovery_ledger:
                                        self._rollout_recovery_ledger.discard_group(
                                            group_id
                                        )
                        evicted_stale_prompt_groups += evicted
                        if evicted:
                            print(
                                f"  evicted {evicted} stale prompt group(s)",
                                flush=True,
                            )
                            for _ in range(evicted):
                                self._buffer_capacity.release()

                        # Select a batch
                        max_prompt_groups = (
                            grpo_cfg.num_prompts_per_step - groups_dispatched
                        )
                        min_prompt_groups = min(
                            self._async_cfg.min_groups_for_streaming_train,
                            max_prompt_groups,
                        )
                        if self._gen.blocks_training():
                            # Always assemble a whole batch in colocated mode.
                            min_prompt_groups = max_prompt_groups
                        if self._rollout_recovery_ledger is None:
                            train_meta, num_groups = await self._sampler.select(
                                current_train_weight=self._trainer_version,
                                min_prompt_groups=min_prompt_groups,
                                max_prompt_groups=max_prompt_groups,
                            )
                        else:
                            # Selection removes local replay ownership. Keep that
                            # removal and the matching ledger claim in one mutation
                            # boundary so a later periodic checkpoint cannot split them.
                            async with self._data_plane_checkpoint_barrier.mutation(
                                "group_removals"
                            ):
                                train_meta, num_groups = await self._sampler.select(
                                    current_train_weight=self._trainer_version,
                                    min_prompt_groups=min_prompt_groups,
                                    max_prompt_groups=max_prompt_groups,
                                )
                                if train_meta is not None:
                                    self._claim_train_meta(
                                        train_meta, num_groups=num_groups
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
                                    f"{grpo_cfg.num_prompts_per_step} prompt "
                                    f"groups with {buffered_groups} group(s) "
                                    f"remaining in the buffer"
                                )
                            await asyncio.sleep(0.005)
                            continue

                        consumed_metas.append(train_meta)
                        consumed_group_count += num_groups
                        selected_group_ids = self._group_ids_from_meta(train_meta)
                        for group_id in selected_group_ids:
                            for name, value in self._finalizer_metrics_by_group.pop(
                                group_id, {}
                            ).items():
                                step_finalizer_metrics.setdefault(name, []).append(
                                    float(value)
                                )

                    # A generation engine that blocks training must stand down;
                    # sleep the engine to allow for the trainer to perform GPU work.
                    # Safe because the select above assembles whole steps for
                    # blocking engines (min == max), so nothing below waits on
                    # rollouts the sleeping engine could not deliver.
                    # In-flight requests freeze, then continue on fresh weights.
                    # The post-step `_sync_weights` wake reopens the gate,
                    # except on save-bound steps, where the wake is deferred until after the save.
                    if not generation_released and self._gen.blocks_training():
                        self._rollout_permitted.clear()
                        await asyncio.to_thread(self._gen.finish_generation)
                        generation_released = True

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

                    groups_dispatched += num_groups

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(self._trainer.finish_train_step)

                self._optimizer_commit_in_progress = True
                async with self._data_plane_checkpoint_barrier.mutation(
                    "sample_clears"
                ):
                    if self._rollout_recovery_ledger is not None:
                        self._rollout_recovery_ledger.mark_train_step_applied(
                            self._train_steps
                        )
                    await self._cleanup_consumed_metas(consumed_metas)
                    if self._rollout_recovery_ledger is not None:
                        self._rollout_recovery_ledger.release_applied_train_step(
                            self._train_steps
                        )
                for _ in range(consumed_group_count):
                    self._buffer_capacity.release()

                step_metrics = aggregate_step_metrics(result)
                step_metrics.update(
                    {
                        name: statistics.fmean(values)
                        for name, values in step_finalizer_metrics.items()
                        if values
                    }
                )
                step_metrics.update(
                    reduce_advantage_pump_metrics(**self._step_log_dict)
                )
                self._step_log_dict = {k: [] for k in self._step_log_dict}

                self._trainer_version += 1
                self._train_steps += 1
                self._optimizer_commit_in_progress = False
                self._consumed_samples += grpo_cfg.num_prompts_per_step
                self._total_valid_tokens += step_metrics.get("global_valid_toks", 0)
                self._timeout.mark_iteration()

                is_last_step = self._train_steps >= grpo_cfg.max_num_steps or (
                    self._rollout_exhausted.is_set()
                    and self._rollout_recovery_complete.is_set()
                    and len(self._buffer) == 0
                )
                ft_save_period = self._master_config.checkpointing.get("ft_save_period")
                # _train_steps was already incremented above, so it equals
                # the legacy loop's 1-indexed `step + 1`.
                should_save_by_step = (
                    is_last_step
                    or self._train_steps
                    % self._master_config.checkpointing["save_period"]
                    == 0
                    or (
                        ft_save_period is not None
                        and self._train_steps % ft_save_period == 0
                    )
                )
                # Latched (timer.last_saved): call once per step and reuse the bool.
                # A second call would consume the latch, returning False and silently
                # dropping both the save and the early-stop break below.
                should_save_by_timeout = self._timeout.check_save()
                will_save_checkpoint = self._master_config.checkpointing[
                    "enabled"
                ] and (should_save_by_step or should_save_by_timeout)
                # A save-bound colocated step defers the wake until after the save.
                defer_wake_for_save = (
                    will_save_checkpoint
                    and self._gen.blocks_training()
                    and self._gen.wake_carries_weight_updates()
                )

                with self._timer.time("weight_sync"):
                    calibration_data = (
                        BatchedDataDict.from_batches(calibration_batches)
                        if calibration_batches
                        else None
                    )
                    aborted_stale_inflight_groups = await self._sync_weights(
                        calibration_data=calibration_data,
                        defer_engine_wake=defer_wake_for_save,
                    )
                    step_metrics.update(
                        {
                            "evicted_stale_prompt_groups": evicted_stale_prompt_groups,
                            "aborted_stale_inflight_groups": aborted_stale_inflight_groups,
                        }
                    )

                if will_save_checkpoint:
                    with self._timer.time("checkpointing"):
                        await self._save_checkpoint(step_metrics)
                    if defer_wake_for_save:
                        # The save is done; wake the engine unless the loop is about to exit.
                        loop_will_exit = (
                            self._train_steps >= grpo_cfg.max_num_steps
                            or should_save_by_timeout
                            or (
                                self._rollout_exhausted.is_set()
                                and len(self._buffer) == 0
                            )
                        )
                        if not loop_will_exit:
                            with self._timer.time("weight_sync"):
                                # The save onloaded model+optimizer; generation needs offload.
                                await asyncio.to_thread(
                                    self._trainer.offload_after_refit
                                )
                                await asyncio.to_thread(
                                    self._gen.prepare_for_generation
                                )
                                self._rollout_permitted.set()

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
            if (
                self._master_config.rollout_checkpointing.telemetry_interval_s
                is not None
            ):
                await self._log_rollout_throughput_metrics()
            self._timer.reset()

            # min sample version refers to the version each consumed sample was
            # generated with; lag = training version - oldest sample version.
            lag = version_during_step - min_sample_version  # type: ignore
            print(
                f"train step {self._train_steps}/{grpo_cfg.max_num_steps}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                break

    async def _watchdog_pump(self) -> None:
        """Report rollout health, and detect stalls nothing else catches.

        Progress is the pair (committed groups, completed train steps) rather than a
        timestamp: both counters already exist, and "neither has moved" is the property
        that actually matters.

        Deliberately *not* conditioned on rollouts being in flight. An earlier version
        required that, on the reasoning that an idle controller has legitimately no
        work -- and a fault-injection run walked straight through the gap. Killing a
        generation worker wedged the loop with zero rollouts in flight and zero
        failures recorded: the rollout pump was blocked on backpressure behind a train
        pump that could no longer finish a step, so nothing was in flight to count.
        The watchdog observed six minutes of idleness and said nothing.

        What separates a real stall from an idle gap is whether work remains, so that
        is what is checked instead.
        """
        watchdog_cfg = self._async_cfg.watchdog
        max_num_steps = self._master_config.grpo.max_num_steps
        last_progress = (-1, -1)
        last_progress_at = time.monotonic()

        while True:
            await asyncio.sleep(watchdog_cfg.interval_s)
            now = time.monotonic()

            stats = self._rollout_manager.stats
            progress = (stats.committed, self._train_steps)
            if progress != last_progress:
                last_progress = progress
                last_progress_at = now
            idle_s = now - last_progress_at

            metrics = dict(stats.as_metrics())
            metrics["rollout/inflight"] = float(self._inflight_rollouts)
            metrics["rollout/idle_s"] = idle_s
            metrics["rollout/train_steps"] = float(self._train_steps)
            self._logger.log_metrics(metrics, step=self._train_steps)

            if watchdog_cfg.gym_subprocess_check:
                # Bounded by one tick so a wedged environment cannot stop the pump, and
                # routed through stall_action so "warn" means warn -- see
                # _check_env_health.
                problems = await self._check_env_health(watchdog_cfg.interval_s)
                if problems:
                    detail = "; ".join(problems)
                    if watchdog_cfg.stall_action == "abort":
                        raise RuntimeError(
                            f"environment health check failed -- {detail}"
                        )
                    print(f"WARNING: environment health -- {detail}", flush=True)

            work_remains = self._train_steps < max_num_steps
            if work_remains and idle_s > watchdog_cfg.stall_timeout_s:
                message = (
                    f"no rollout committed and no train step completed in "
                    f"{idle_s:.0f}s ({self._inflight_rollouts} rollouts in flight, "
                    f"{stats.committed} groups committed, step "
                    f"{self._train_steps}/{max_num_steps}, "
                    f"stall_timeout_s={watchdog_cfg.stall_timeout_s})"
                )
                if watchdog_cfg.stall_action == "abort":
                    raise RolloutStall(message)
                print(f"WARNING: rollout stall -- {message}", flush=True)

    async def _check_env_health(self, timeout_s: float) -> list[str]:
        """Ask each environment actor that exposes a health check whether it is whole.

        Returns the problems found, empty when everything is well. It *reports* rather
        than raises so the caller can route the verdict through ``stall_action``, the
        same way the stall path does. Raising here bypassed ``stall_action`` entirely:
        under the documented default (``"warn"``, which promises to "only report"), and
        with ``gym_subprocess_check`` defaulting to true, an unhealthy environment killed
        the run -- a run-ending path switched on by default, in a feature whose whole
        posture is inert-by-default.

        Each probe is bounded. ``NemoGym`` is an asyncio actor, so a *wedged* environment
        -- precisely the case this check exists to catch -- left the await hanging
        forever, the pump never ticked again, and stall detection was dead exactly when
        it was needed. A probe that does not answer within one tick IS the unhealthy
        signal; it is not a reason to stop watching.

        Environments without the method are skipped rather than treated as unhealthy;
        only NeMo-Gym has subprocess servers to lose.
        """
        problems: list[str] = []
        for env_name, handle in self._env_handles.items():
            health_check = getattr(handle, "health_check", None)
            if health_check is None:
                continue
            try:
                await asyncio.wait_for(
                    self._ray_get(health_check.remote()), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                problems.append(
                    f"environment {env_name!r} did not answer its health check within "
                    f"{timeout_s}s"
                )
            except Exception as error:
                problems.append(f"environment {env_name!r} reported unhealthy: {error}")
        return problems

    async def _abort_stale_inflight(self) -> int:
        """Abort in-flight rollouts that the sampler can no longer select."""
        stale_tasks = [
            task
            for task, start_version in self._inflight_by_group_id.values()
            if self._sampler.should_abort_inflight(
                start_weight_version=start_version,
                current_train_weight=self._trainer_version,
            )
        ]
        if not stale_tasks:
            return 0

        for task in stale_tasks:
            task.cancel()

        results = await asyncio.gather(*stale_tasks, return_exceptions=True)
        failures = [
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]
        if failures:
            raise BaseExceptionGroup(
                "stale in-flight rollout cleanup failed",
                failures,
            )

        print(
            f"  aborted {len(stale_tasks)} stale in-flight rollout(s)",
            flush=True,
        )
        return len(stale_tasks)

    async def _save_checkpoint(self, step_metrics: dict[str, Any]) -> None:
        """Serialize full and rollout-only checkpoint publication."""
        async with self._checkpoint_save_lock:
            await self._save_checkpoint_locked(step_metrics)

    async def _save_checkpoint_locked(self, step_metrics: dict[str, Any]) -> None:
        """Write a full checkpoint for the just-finished train step.

        Everything except the (possibly async) policy weight write must be on disk
        before begin_finalization. Non-colocated engines keep serving rollouts throughout;
        colocated engine is held stood-down through the save.
        """
        save_state = self._save_state
        save_state.current_step = self._train_steps
        save_state.total_steps = self._train_steps
        save_state.trainer_version = self._trainer_version
        save_state.current_epoch = self._current_epoch
        save_state.consumed_samples = self._consumed_samples
        save_state.total_valid_tokens = self._total_valid_tokens
        save_state.sampler_name = self._async_cfg.sampler.name
        dataloader_state: Optional[dict[str, Any]] = None
        # SC has no validation loop yet; drop the default sentinel instead of
        # persisting a bogus val_reward.
        if hasattr(save_state, "val_reward"):
            delattr(save_state, "val_reward")

        # validate_single_controller_config already rejected anything but a
        # "train:" prefix, so step_metrics is the only source to consult.
        full_metric_name = self._master_config.checkpointing["metric_name"]
        if full_metric_name is not None:
            metric_name = full_metric_name.split(":", 1)[1]
            if metric_name not in step_metrics:
                raise ValueError(f"Metric {metric_name} not found in train metrics")
            setattr(save_state, full_metric_name, step_metrics[metric_name])

        # Flush the previous checkpoint's background finalization first;
        # re-raises a failure from it.
        await asyncio.to_thread(self._checkpointer.finalize_pending)

        print(f"Saving checkpoint for step {self._train_steps}...")
        checkpoint_path: PathLike
        replay_metadata: Optional[TQReplayMetadataState] = None
        rollout_recovery_payload: Optional[bytes] = None
        rollout_recovery_digest: Optional[str] = None
        rollout_recovery_group_count: Optional[int] = None
        if self._master_config.data_plane.get("checkpointing_enabled"):
            if self._finalizer_unknown_outcomes:
                raise RuntimeError(
                    "Refusing to save a native TQ checkpoint after an unknown "
                    "finalizer publication outcome"
                )
            # Commits and destructive clears take the same barrier. Generation
            # may continue while a snapshot is written, but completed groups
            # wait at commit, so TQ and the metadata sidecar describe exactly
            # the same set of training-ready groups.
            async with self._data_plane_checkpoint_barrier.checkpoint():
                # Dataloader advancement and prompt-lineage reservation use the
                # mutation side of this barrier, so this cursor belongs to the
                # same exact cut as replay metadata, lineage, and native TQ.
                save_state.current_epoch = self._current_epoch
                checkpoint_path = await asyncio.to_thread(
                    self._checkpointer.init_tmp_checkpoint,
                    self._train_steps,
                    vars(save_state),
                    self._master_config,
                )
                dataloader_state = self._dataloader.state_dict()
                if self._sampler.supports_buffer_checkpoint:
                    replay_metadata = self._buffer.metadata_state_dict(
                        saved_capacity=self._async_cfg.max_buffered_rollouts
                    )
                if self._rollout_recovery_ledger is not None:
                    self._rollout_recovery_ledger.assert_full_step_checkpoint_safe()
                    rollout_recovery_state = self._rollout_recovery_ledger.state_dict()
                    payload_buffer = io.BytesIO()
                    torch.save(rollout_recovery_state, payload_buffer)
                    rollout_recovery_payload = payload_buffer.getvalue()
                    rollout_recovery_digest = hashlib.sha256(
                        rollout_recovery_payload
                    ).hexdigest()
                    rollout_recovery_group_count = len(rollout_recovery_state["groups"])
                if replay_metadata is not None:
                    await self._validate_replay_inventory(replay_metadata)
                if rollout_recovery_payload is not None:
                    await self._validate_rollout_recovery_inventory(
                        clear_unreferenced=False
                    )
                await self._save_data_plane_checkpoint(
                    checkpoint_path,
                    replay_metadata=replay_metadata,
                    rollout_recovery_payload_sha256=rollout_recovery_digest,
                    rollout_recovery_group_count=rollout_recovery_group_count,
                )
                self._last_rollout_snapshot_mutation_version = (
                    self._data_plane_checkpoint_barrier.mutation_version
                )
        else:
            save_state.current_epoch = self._current_epoch
            checkpoint_path = await asyncio.to_thread(
                self._checkpointer.init_tmp_checkpoint,
                self._train_steps,
                vars(save_state),
                self._master_config,
            )
            dataloader_state = self._dataloader.state_dict()
        # With async_save this returns after D2H staging; disk writes finish
        # in the background.
        await asyncio.to_thread(
            self._trainer.save_checkpoint,
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer")
            if self._checkpointer.save_optimizer
            else None,
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            checkpointing_cfg=self._master_config.checkpointing,
        )
        assert dataloader_state is not None
        await asyncio.to_thread(
            torch.save,
            dataloader_state,
            os.path.join(checkpoint_path, "train_dataloader.pt"),
        )
        if replay_metadata is not None:
            await asyncio.to_thread(
                torch.save,
                replay_metadata,
                os.path.join(checkpoint_path, REPLAY_BUFFER_METADATA_FILENAME),
            )
        if rollout_recovery_payload is not None:
            recovery_path = os.path.join(
                checkpoint_path, ROLLOUT_RECOVERY_STATE_FILENAME
            )
            await asyncio.to_thread(
                Path(recovery_path).write_bytes,
                rollout_recovery_payload,
            )
        # Rename happens in the background once the async weight writes
        # finish; flushed at the next save or on exit.
        self._checkpointer.begin_finalization(
            checkpoint_path,
            wait_fn=self._trainer.finalize_async_save,
        )
        await asyncio.to_thread(
            _write_latest_checkpoint_status,
            self._checkpointer,
            last_checkpoint_step=self._train_steps,
        )

    async def _sync_weights(
        self,
        *,
        calibration_data: Optional[BatchedDataDict[Any]] = None,
        defer_engine_wake: bool = False,
    ) -> int:
        """Pause new rollout dispatches, synchronize weights, resume.

        SC owns the pause gate. vLLM serves through the refit;
        its async engine supports weight updates during pending requests.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. Optionally calibrate FP8 KV-cache scales.
          3. weight_synchronizer.sync_weights(kv_scales=...)
          4. _rollout_permitted.set()   — resume

        `defer_engine_wake` is used to skip refit and wake on save-bound steps for colocated cases.

        Args:
            calibration_data: Optional data used to calibrate FP8 KV-cache
                scales before synchronizing weights.
            defer_engine_wake: Keep the engine asleep: offload the trainer's
                refit buffers, stamp the version, leave the gate closed.

        Returns:
            The number of stale in-flight rollout groups aborted before the
            weight synchronization.
        """
        self._rollout_permitted.clear()

        # TODO(#2625): abort unconditionally once gym-path abort is validated;
        # for now only the native path aborts. Local import dodges the grpo.py
        # circular dep (as in async_utils/trajectory_collector.py).
        from nemo_rl.algorithms.grpo import MasterConfig as GrpoMasterConfig
        from nemo_rl.algorithms.grpo import _should_use_nemo_gym

        aborted_stale_inflight_groups = (
            0
            if _should_use_nemo_gym(cast(GrpoMasterConfig, self._master_config))
            else await self._abort_stale_inflight()
        )

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

        if defer_engine_wake:
            await asyncio.to_thread(self._trainer.offload_before_refit)
        else:
            await asyncio.to_thread(
                self._weight_synchronizer.sync_weights,
                kv_scales=kv_scales,
            )
        if self._async_cfg.recompute_kv_cache_after_weight_updates:
            await asyncio.to_thread(self._gen.invalidate_kv_cache)
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        if self._master_config.token_capture.enabled:
            # Rotate the version vLLM workers stamp on captured model calls
            # (per-call tagging; group staleness = min over the group's calls).
            await asyncio.to_thread(
                self._gen.set_rollout_weight_version, self._trainer_version
            )
        if not defer_engine_wake:
            self._rollout_permitted.set()
        return aborted_stale_inflight_groups

    async def _log_rollout_throughput_metrics(self, *, emit: bool = True) -> None:
        """Serialize backend sampling across telemetry and train-step paths."""
        async with self._rollout_telemetry_lock:
            await self._collect_and_log_rollout_throughput_metrics(emit=emit)

    async def _collect_and_log_rollout_throughput_metrics(
        self, *, emit: bool = True
    ) -> None:
        """Compare generation-engine output with canonical TQ publication."""
        now = time.monotonic()
        counters = self._rollout_manager.telemetry_snapshot()
        barrier = await self._data_plane_checkpoint_barrier.drain_telemetry()
        generation_metrics: dict[str, Any] = {}
        try:
            generation_metrics = await asyncio.to_thread(
                self._gen.drain_latest_logger_metrics
            )
        except Exception as error:
            warnings.warn(
                "Failed to collect generation throughput telemetry: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )

        generated_values = _latest_vllm_values(generation_metrics, "generation_tokens")
        generated_by_worker = (
            {worker_id: int(value) for worker_id, value in generated_values.items()}
            if generated_values
            else None
        )
        generated_tokens = (
            sum(generated_by_worker.values())
            if generated_by_worker is not None
            else None
        )
        metrics: dict[str, float] = {
            key: float(value) for key, value in counters.items()
        }
        metrics.update(
            {
                "controller_dispatched_tasks": float(len(self._dispatched_rollouts)),
                "controller_inflight_rollouts": float(self._inflight_rollouts),
                "controller_rollout_capacity": float(
                    self._async_cfg.max_inflight_prompts
                ),
                "controller_rollout_slot_waiters": float(self._rollout_slot_waiters),
                "controller_rollout_permitted_waiters": float(
                    self._rollout_permitted_waiters
                ),
                "controller_buffer_capacity_waiters": float(
                    self._buffer_capacity_waiters
                ),
                "checkpoint_barrier_active": float(barrier.checkpoint_active),
                "checkpoint_active_mutations": float(barrier.active_mutations),
                "checkpoint_waiting_mutations": float(barrier.waiting_mutations),
                "checkpoint_max_simultaneous_waiters": float(
                    barrier.max_waiting_mutations
                ),
                "checkpoint_blocked_mutations": float(
                    sum(barrier.blocked_by_kind.values())
                ),
                "buffer_occupancy_groups": float(len(self._buffer)),
            }
        )
        for kind in CHECKPOINT_MUTATION_KINDS:
            metrics[f"checkpoint_blocked_{kind}"] = float(barrier.blocked_by_kind[kind])
        mutation_waits = list(barrier.wait_durations_s)
        if mutation_waits:
            metrics["checkpoint_mutation_wait_seconds_total"] = sum(mutation_waits)
            metrics["checkpoint_mutation_wait_seconds_mean"] = sum(
                mutation_waits
            ) / len(mutation_waits)
            metrics["checkpoint_mutation_wait_seconds_p95"] = _percentile(
                mutation_waits, 0.95
            )
            metrics["checkpoint_mutation_wait_seconds_max"] = max(mutation_waits)

        running = _latest_vllm_values(generation_metrics, "inflight_batch_sizes")
        waiting = _latest_vllm_values(generation_metrics, "num_pending_samples")
        kv_usage = _latest_vllm_values(generation_metrics, "kv_cache_usage_perc")
        if running:
            metrics["vllm_requests_running"] = sum(running.values())
        if waiting:
            metrics["vllm_requests_waiting"] = sum(waiting.values())
        if kv_usage:
            metrics["vllm_kv_cache_usage_mean"] = sum(kv_usage.values()) / len(kv_usage)
        if generated_tokens is not None:
            metrics["vllm_output_tokens"] = float(generated_tokens)

        completion_durations = list(self._rollout_completion_durations_s)
        queue_wait_durations = list(self._rollout_queue_wait_durations_s)
        self._rollout_completion_durations_s.clear()
        self._rollout_queue_wait_durations_s.clear()
        if completion_durations:
            metrics["group_completion_samples"] = float(len(completion_durations))
            metrics["group_completion_seconds_mean"] = sum(completion_durations) / len(
                completion_durations
            )
            metrics["group_completion_seconds_p50"] = _percentile(
                completion_durations, 0.50
            )
            metrics["group_completion_seconds_p95"] = _percentile(
                completion_durations, 0.95
            )
        if queue_wait_durations:
            metrics["group_queue_wait_seconds_mean"] = sum(queue_wait_durations) / len(
                queue_wait_durations
            )
            metrics["group_queue_wait_seconds_p95"] = _percentile(
                queue_wait_durations, 0.95
            )

        previous_time = self._throughput_sample_time
        previous_counters = self._throughput_rollout_counters
        previous_generated = self._throughput_generation_tokens_by_worker
        if previous_time is not None and previous_counters is not None:
            elapsed = now - previous_time
            if elapsed > 0:
                metrics["sample_elapsed_seconds"] = elapsed
                metrics["canonical_groups_per_second"] = (
                    counters["canonical_groups_finalized"]
                    - previous_counters["canonical_groups_finalized"]
                ) / elapsed
                metrics["canonical_output_tokens_per_second"] = (
                    counters["canonical_output_tokens"]
                    - previous_counters["canonical_output_tokens"]
                ) / elapsed
                if generated_by_worker is not None and previous_generated is not None:
                    current_workers = set(generated_by_worker)
                    previous_workers = set(previous_generated)
                    monotonic = all(
                        generated_by_worker[worker_id] >= previous_generated[worker_id]
                        for worker_id in current_workers & previous_workers
                    )
                    if current_workers == previous_workers and monotonic:
                        generated_delta = sum(
                            generated_by_worker[worker_id]
                            - previous_generated[worker_id]
                            for worker_id in current_workers
                        )
                        metrics["vllm_output_tokens_per_second"] = (
                            generated_delta / elapsed
                        )
                    else:
                        metrics["vllm_counter_discontinuity"] = 1.0

        self._throughput_sample_time = now
        self._throughput_generation_tokens_by_worker = generated_by_worker
        self._throughput_rollout_counters = counters
        if emit:
            self._log_telemetry_metrics(
                metrics,
                step=self._train_steps,
                prefix="rollout/throughput",
            )
            print(f"rollout_throughput_metrics={metrics}", flush=True)

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
