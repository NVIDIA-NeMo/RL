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
                 → _value_stage(meta) (PPO only) → value.get_values_from_meta(...)
                     Value → dp_client.get/put_samples(...) (via its own client)
                 → _advantage_stage(meta) → dp_client.get_samples(...)
                                        → adv_estimator.compute_advantage(...)
                                        → dp_client.put_samples(...)
                 → _value_train_epochs(meta) (PPO only)
                     → value.train_from_meta(...)
                     Value → dp_client.get_samples(...)     (via its own client)
                 → trainer.begin/train_microbatches/finish_train_step (split API,
                     driver-side TQPolicy via asyncio.to_thread)
                     Trainer → dp_client.get_samples(...)   (via its own client)
                 → dp_client.clear_samples(...)             ← SC clears after train
  _sync_weights  → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import math
import os
import threading
import time
import uuid
import warnings
from collections import deque
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union, cast

import ray
import torch
from ray.exceptions import RayActorError

from nemo_rl.algorithms import opd as opd_module
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLACEMENT_RESERVE_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    DataPlaneCheckpointBarrier,
    DataPlaneCheckpointMetadata,
    DataPlaneMutationCut,
    TQReplayMetadataState,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    TransactionalAdmissionSampler,
    create_sampler,
)
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _write_latest_checkpoint_status,
    aggregate_rollout_metrics,
    compute_and_apply_seq_logprob_error_masking,
)
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.ppo import _compute_critic_metrics
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    MasterConfig,
    algo_config,
    is_ppo_run,
    validate_sampler_buffer_capacity,
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    apply_message_level_advantage_penalties,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import DATA_PLANE_CHECKPOINT_SCHEMA_VERSION, KVBatchMeta
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.data_plane.schema import (
    DP_CALIB_INPUT_FIELDS,
    DP_TRAIN_FIELDS,
    ROLLOUT_METRICS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.refit_watchdog import RefitAborted, is_refit_context_lost
from nemo_rl.environments.nemo_gym import should_use_nemo_gym
from nemo_rl.experience.failures import RolloutStall
from nemo_rl.experience.payload import VIOLATION_TAG_KEYS
from nemo_rl.experience.rollout_manager import RolloutOutcome
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    PromptGroupPhase,
    RolloutRecoveryState,
    build_rollout_recovery_state,
    parse_rollout_recovery_state,
)
from nemo_rl.models.generation.fleet_health import ShardState
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.models.value.tq_value import TQValue
from nemo_rl.utils.checkpoint import CheckpointManager, PathLike
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import TimeoutChecker, Timer

Generation = Union[VllmGeneration, SGLangGeneration, MegatronGeneration]

# Named `log` rather than `logger` to keep it distinct from the experiment
# Logger this module also uses as `self._logger`.
log = logging.getLogger(__name__)


def _pooled_opd_metrics(
    stat_sum: float, stat_sumsq: float, count: int
) -> dict[str, float]:
    """Compute whole-step OPD metrics from exact pooled sufficient statistics."""
    if count <= 0:
        return {}
    mean = stat_sum / count
    # OPDAdvantageEstimator uses torch.std's default unbiased estimator.
    variance = (stat_sumsq - count * mean * mean) / (count - 1) if count > 1 else 0.0
    return {
        "on_policy_distillation/teacher_student_logprob_gap_mean": mean,
        "on_policy_distillation/adv_mean": mean,
        "on_policy_distillation/adv_std": math.sqrt(max(variance, 0.0)),
    }


def _train_fields_for_step(
    *,
    policy_logprobs_required: bool,
    reference_logprobs_required: bool,
) -> tuple[str, ...]:
    """Return only the data-plane columns produced for this train step."""
    return tuple(
        field
        for field in DP_TRAIN_FIELDS
        if (policy_logprobs_required or field != "prev_logprobs")
        and (reference_logprobs_required or field != "reference_policy_logprobs")
    )


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns three concurrent asyncio tasks:
      - _rollout_pump:  dispatches prompts to GenerationWorkerActor
      - _train_pump:    claims DataPlane meta, trains, clears consumed rows,
                        then runs _sync_weights (drain gate + weight
                        synchronization) inline after each optimizer step
      - _stall_watchdog_pump: publishes rollout counters and reports stalls or
                        unhealthy environments, which are the failures that
                        otherwise produce no signal at all

    Plus _gen_fleet_probe_pump when fleet health is enabled, which probes generation
    shard liveness on its own, much shorter clock.

    All other actors are passive — they expose methods and wait to be called.
    """

    # True from the moment a failed refit pulls every shard out of service until the
    # retry puts them back; the exhaustion check stands down for that window. See
    # _recovery_window.
    #
    # Declared on the class, not assigned in __init__, for the same reason
    # AbstractPolicyWorker.model_update_group is: the stall watchdog reads it on every
    # tick, and it must exist on any instance the watchdog can reach.
    _recovering_from_refit: bool = False

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
        self._algo_cfg = algo_config(master_config)
        self._async_cfg = master_config.async_rl
        self._is_ppo: bool = is_ppo_run(master_config)
        # GRPO has no epoch knob: it makes one optimizer step per RL step.
        self._ppo_epochs: int = self._algo_cfg.ppo_epochs if self._is_ppo else 1
        self._critic_ppo_epochs: int = (
            self._algo_cfg.critic_ppo_epochs if self._is_ppo else 1
        )
        self._message_level_advantage_penalties_enabled = (
            self._algo_cfg.invalid_tool_call_advantage is not None
            or self._algo_cfg.malformed_thinking_advantage is not None
        )

        self._policy_logprobs_required = not (
            master_config.loss_fn.force_on_policy_ratio
            and self._algo_cfg.seq_logprob_error_threshold is None
        )
        # _build_trainer initializes the reference model only for a positive KL
        # penalty, so the controller must use the same gate before requesting it.
        self._reference_logprobs_required = bool(
            master_config.loss_fn.reference_policy_kl_penalty > 0
            and not self._algo_cfg.skip_reference_policy_logprobs_calculation
        )
        self._teacher_logprobs_required = opd_module.is_opd_enabled(master_config)
        self._train_fields = _train_fields_for_step(
            policy_logprobs_required=self._policy_logprobs_required,
            reference_logprobs_required=self._reference_logprobs_required,
        )
        self._dp_client = actor_args.dp_client
        self._gen: Generation = actor_args.gen_handle
        self._trainer: TQPolicy = actor_args.trainer_handle
        self._value: Optional[TQValue] = getattr(actor_args, "value_handle", None)
        self._dataloader = actor_args.dataloader
        self._weight_synchronizer = actor_args.weight_synchronizer
        self._advantage_estimator = actor_args.advantage_estimator
        self._loss_fn = actor_args.loss_fn
        self._value_loss_fn = getattr(actor_args, "value_loss_fn", None)
        self._buffer = actor_args.tq_buffer
        self._rollout_manager = actor_args.rollout_manager
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer

        # Direct access, deliberately. A getattr default here reads as defensive but
        # buys a silent failure mode: rename or drop the field and
        # watchdog.gym_subprocess_check: true degrades to a health check that iterates
        # nothing and reports nothing -- the exact class of silent failure this work
        # exists to remove. A missing field should break loudly at construction, where
        # it costs five minutes, not quietly at hour three of a run.
        self._env_handles = actor_args.env_handles
        # These two keep the getattr for a genuinely different reason: None is a
        # meaningful value meaning "feature off", and it is also their default. Absence
        # therefore degrades to the documented off state rather than to a broken one.
        self._gen_fleet = getattr(actor_args, "fleet_monitor", None)
        self._generation_router = getattr(actor_args, "generation_router", None)
        teacher_worker_groups = getattr(actor_args, "teacher_worker_groups", None) or {}
        if teacher_worker_groups:
            self._teacher_coordinator: Optional[
                opd_module.TQTeacherLogprobCoordinator
            ] = opd_module.TQTeacherLogprobCoordinator(
                dp_client=self._dp_client,
                teacher_worker_groups=teacher_worker_groups,
                alias_to_group_alias=(
                    getattr(actor_args, "alias_to_group_alias", None) or {}
                ),
                on_policy_distillation_cfg=opd_module._opd_cfg(master_config),
            )
            self._buffer.set_post_write_enricher(self._teacher_coordinator.enrich)
        else:
            self._teacher_coordinator = None

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._logger.log_hyperparams(master_config.model_dump())
        self._logger.log_metrics(
            setup_timing_metrics.to_metrics_dict(), step=0, prefix="timing/setup"
        )
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

        # Loaded (or initial) GRPOSaveState from setup; _get_grpo_save_state
        # already defaulted any fields missing from older checkpoints.
        self._save_state: GRPOSaveState = actor_args.save_state
        self._last_checkpoint_path: Optional[str] = actor_args.last_checkpoint_path
        self._data_plane_checkpoint_metadata: Optional[DataPlaneCheckpointMetadata] = (
            actor_args.data_plane_checkpoint_metadata
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
        num_prompts_per_step = self._algo_cfg.num_prompts_per_step
        self._sampler = create_sampler(self._buffer, self._async_cfg.sampler)
        restored_dispatch_index = actor_args.save_state.sampler_dispatch_index
        if restored_dispatch_index is None:
            # Checkpoints predating exact sampler state reconstruct the original
            # fresh-step invariant from the restored trainer version.
            self._sampler.set_dispatch_index(restored_trainer_version)
        else:
            self._sampler.restore_dispatch_index(restored_dispatch_index)
        if (
            self._master_config.checkpointing["enabled"]
            and self._sampler.supports_buffer_checkpoint
            and not self._master_config.checkpointing.get("save_data_plane")
        ):
            raise ValueError(
                "SingleController checkpointing with a replay-checkpoint-capable "
                "sampler requires checkpointing.save_data_plane=true so "
                "completed, unconsumed rollouts are recoverable."
            )
        restoring_rollout_recovery = bool(
            self._data_plane_checkpoint_metadata is not None
            and self._data_plane_checkpoint_metadata.get(
                "rollout_recovery_payload_sha256"
            )
            is not None
        )
        self._rollout_recovery_enabled = bool(
            restoring_rollout_recovery
            or (
                self._master_config.checkpointing["enabled"]
                and self._master_config.checkpointing.get("save_data_plane")
                and self._sampler.supports_buffer_checkpoint
            )
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
        self._buffer.set_data_plane_checkpoint_barrier(
            self._data_plane_checkpoint_barrier
        )

        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Set only after _rollout_pump exhausts its configured epochs and all
        # dispatched tasks finish successfully. Rollout failures propagate
        # through run() instead of being reported as normal exhaustion.
        self._rollout_exhausted: asyncio.Event = asyncio.Event()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        self._inflight_by_group_id: dict[str, tuple[asyncio.Task[None], int]] = {}

        # Groups that will never arrive, keyed by the training step they were stamped
        # for. A sampler that matches batches to steps exactly (InOrderSampler) can only
        # ever select num_prompts_per_step groups carrying that stamp, so a dropped
        # prompt leaves that step permanently one short and the train pump waits on a
        # group no one is generating. The pump subtracts these to close the step short.
        # Only stamped prompts appear here: with an unstamped sampler the batch fills
        # from whatever is ready, so a drop costs throughput but strands nothing.
        self._batch_shortfall: dict[int, int] = {}

        # Spare prompts that on_dropped_prompt="replace" substitutes for dropped ones,
        # and the per-step counts of how each step got made whole (logged so a step's
        # batch size stays explainable after the fact). The reserve is filled only by the
        # rollout pump, which owns the dataloader iterator; see the config docstring for
        # why a dispatch task cannot pull from it directly.
        #
        # The two counters read from opposite ends of a borrow: a step that filled a
        # hole with a later step's finished group counts a promotion, and the step it
        # borrowed from counts the replacement that repaid it.
        self._replacement_reserve: deque[DatumSpec] = deque()
        self._batch_replacements: dict[int, int] = {}
        self._batch_promotions: dict[int, int] = {}
        # Whether the sampler has ever handed back a target step. Only stamped prompts
        # can strand a step, so this gates the pool: filling it for a sampler that never
        # stamps would divert a batch of prompts that nothing is ever able to draw on.
        # Learned from admit rather than the sampler's type, because a custom sampler's
        # stamping is not knowable until it answers.
        self._sampler_stamps_target_steps: bool = False

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )

        self._trainer_version: int = restored_trainer_version
        self._train_steps: int = actor_args.save_state.current_step
        self._current_epoch: int = actor_args.save_state.current_epoch
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
            "seq_logprob_error_metrics": [],
            **{key: [] for key in VIOLATION_TAG_KEYS},
        }
        self._opd_stat_sum = 0.0
        self._opd_stat_sumsq = 0.0
        self._opd_stat_count = 0

        # Seeded here rather than in run(): on resume _trainer_version is the
        # checkpoint's step, so a run resuming mid-warmup needs the widened
        # window before the first dispatch.
        self._retune_lookahead_versions()

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
        # Synchronize weights before starting the pumps, unless setup already delivered them.
        if self._weight_synchronizer.is_stale:
            await self._sync_weights()
        self._rollout_manager.set_weight_version(self._trainer_version)

        restored_replay_groups = await self._maybe_restore_replay_buffer()
        await self._maybe_restore_rollout_recovery(
            restored_replay_groups=restored_replay_groups
        )
        await self._maybe_restore_replacement_reserve()

        # Start the rollout and train pumps, plus the watchdog
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        watchdog_task = asyncio.create_task(self._stall_watchdog_pump())
        tasks = [rollout_task, train_task, watchdog_task]
        # Only with fleet health on. Created unconditionally it would be a timer firing
        # every probe_interval_s for every run that does not use the feature, which is
        # the default.
        probe_task = (
            asyncio.create_task(self._gen_fleet_probe_pump())
            if self._gen_fleet is not None
            else None
        )
        if probe_task is not None:
            tasks.append(probe_task)
        try:
            done, _ = await asyncio.wait(
                set(tasks), return_when=asyncio.FIRST_COMPLETED
            )
            if probe_task is not None and probe_task in done:
                # Loops forever like the watchdog, so finishing at all means it raised.
                await probe_task
            if watchdog_task in done:
                # The watchdog loops forever, so finishing at all means it raised --
                # a stall or an unhealthy environment. Surface that ahead of the
                # pumps, whose own symptom would just be "waiting".
                await watchdog_task
            if rollout_task in done:
                # Propagate rollout failures immediately. A normally exhausted
                # rollout pump leaves the train pump to drain committed groups.
                await rollout_task
            await train_task
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                self._weight_synchronizer.shutdown()
            except Exception as e:  # teardown must not mask the original failure
                print(f"Error during weight-synchronizer shutdown: {e}", flush=True)
            finally:
                self._logger.finish()
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
        }

    # ── internal helpers ───────────────────────────────────────────────────

    async def _maybe_restore_replay_buffer(self) -> int:
        """Restore the local replay index for the native TQ checkpoint.

        Recovery is authoritative only for samplers that explicitly support
        buffered-group restoration. The native snapshot and replay metadata file
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
        # weights_only=False: the replay metadata file contains pickled KVBatchMeta
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
            expected_group_size=self._algo_cfg.num_generations_per_prompt,
            expected_manifest_digest=expected_manifest_digest_value,
        )
        await self._validate_replay_inventory(buffer_state)

        # Each buffered group holds one _buffer_capacity permit. Restore fails
        # above if the saved group count exceeds current capacity.
        assert restored <= self._async_cfg.max_buffered_rollouts
        for _ in range(restored):
            await self._buffer_capacity.acquire()
        return restored

    async def _maybe_restore_rollout_recovery(
        self,
        *,
        restored_replay_groups: int,
    ) -> None:
        """Restore unfinished ownership for prioritized rollout-pump redispatch."""
        if self._last_checkpoint_path is None:
            return
        recovery_path = Path(
            self._last_checkpoint_path,
            ROLLOUT_RECOVERY_STATE_FILENAME,
        )
        metadata = self._data_plane_checkpoint_metadata or {}
        expected_payload_sha256 = metadata.get("rollout_recovery_payload_sha256")
        if expected_payload_sha256 is None:
            if recovery_path.is_file():
                raise RuntimeError(
                    f"{ROLLOUT_RECOVERY_STATE_FILENAME} exists, but the matching "
                    "native TQ checkpoint does not advertise rollout recovery"
                )
            return
        if not isinstance(expected_payload_sha256, str):
            raise TypeError(
                "rollout_recovery_payload_sha256 must be a string in native "
                "TQ checkpoint metadata"
            )
        expected_schema_version = metadata.get("rollout_recovery_schema_version")
        if (
            isinstance(expected_schema_version, bool)
            or expected_schema_version != ROLLOUT_RECOVERY_SCHEMA_VERSION
        ):
            raise ValueError(
                "native TQ checkpoint rollout recovery schema mismatch: "
                f"checkpoint={expected_schema_version!r}, "
                f"expected={ROLLOUT_RECOVERY_SCHEMA_VERSION}"
            )
        expected_group_count = metadata.get("rollout_recovery_group_count")
        if (
            isinstance(expected_group_count, bool)
            or not isinstance(expected_group_count, int)
            or expected_group_count < 0
        ):
            raise TypeError(
                "rollout_recovery_group_count must be an integer in native "
                "TQ checkpoint metadata"
            )
        if not recovery_path.is_file():
            raise FileNotFoundError(
                "native TQ checkpoint advertises rollout recovery, but the "
                f"sidecar is missing at {recovery_path}"
            )

        payload = await asyncio.to_thread(recovery_path.read_bytes)
        actual_payload_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_payload_sha256 != expected_payload_sha256:
            raise ValueError(
                "rollout recovery sidecar checksum mismatch: "
                f"checkpoint={expected_payload_sha256}, "
                f"actual={actual_payload_sha256}"
            )
        state = await asyncio.to_thread(
            torch.load,
            io.BytesIO(payload),
            weights_only=True,
        )
        parsed_state = parse_rollout_recovery_state(state)
        if len(parsed_state.ledger_state["groups"]) != expected_group_count:
            raise ValueError(
                "rollout recovery sidecar group count does not match native "
                "TQ checkpoint metadata"
            )

        recovery_ledger = self._rollout_manager.recovery_ledger
        async with self._data_plane_checkpoint_barrier.mutation() as cut:
            recovery_ledger.load_state_dict(cut, parsed_state.ledger_state)
            self._batch_shortfall = parsed_state.batch_shortfall
            canonical_state = self._buffer.metadata_state_dict(
                saved_capacity=self._async_cfg.max_buffered_rollouts
            )
            canonical_group_ids = {
                group["group_id"] for group in canonical_state["groups"]
            }
            recovery_ledger.discard_canonical_groups(cut, canonical_group_ids)
            await self._rehydrate_rollout_recovery_prompts(cut)
        self._sampler_stamps_target_steps = (
            parsed_state.sampler_stamps_target_steps
            if parsed_state.sampler_stamps_target_steps is not None
            else any(
                group.target_step is not None for group in recovery_ledger.groups()
            )
            or any(
                group.get("target_step") is not None
                for group in canonical_state["groups"]
            )
        )

        groups_to_recover = recovery_ledger.groups()
        if groups_to_recover:
            print(
                f"📦 Loaded {len(groups_to_recover)} unfinished rollout "
                f"group(s) next to {restored_replay_groups} canonical group(s); "
                "the rollout pump will redispatch them before new dataloader work",
                flush=True,
            )

    async def _rehydrate_rollout_recovery_prompts(
        self,
        cut: DataPlaneMutationCut,
    ) -> None:
        """Resolve positional prompt references against a stable map-style dataset.

        This assumes the dataset exposes integer ``__getitem__`` and retains the
        same ordering across checkpoint and restart.
        """
        recovery_ledger = self._rollout_manager.recovery_ledger
        groups = recovery_ledger.groups()
        if not groups:
            return

        dataset = getattr(self._dataloader, "dataset", None)
        if dataset is None:
            raise RuntimeError(
                "cannot restore unfinished rollouts because the dataloader does "
                "not expose its source dataset"
            )

        resolved_prompts: dict[str, DatumSpec] = {}
        for group in groups:
            sample_id = group.prompt_ref.sample_id
            try:
                sample_index = int(sample_id)
            except ValueError as error:
                raise ValueError(
                    f"recovery group {group.group_id!r} has a non-integer "
                    f"dataset sample_id={sample_id!r}"
                ) from error
            if sample_index < 0 or str(sample_index) != sample_id:
                raise ValueError(
                    f"recovery group {group.group_id!r} has a non-canonical "
                    f"dataset sample_id={sample_id!r}"
                )

            prompt = resolved_prompts.get(sample_id)
            if prompt is None:
                try:
                    dataset_prompt = await asyncio.to_thread(
                        dataset.__getitem__, sample_index
                    )
                except (IndexError, KeyError) as error:
                    raise RuntimeError(
                        f"cannot rehydrate recovery group {group.group_id!r}: "
                        f"dataset sample_id={sample_id!r} is unavailable"
                    ) from error
                if not isinstance(dataset_prompt, dict):
                    raise TypeError(
                        f"dataset sample_id={sample_id!r} resolved to "
                        f"{type(dataset_prompt).__name__}, expected a DatumSpec "
                        "dictionary"
                    )

                # Re-run one-row collation to reconstruct the tensor scalars,
                # optional fields, and multimodal wrappers expected by RolloutManager.
                collate_fn = getattr(self._dataloader, "collate_fn", None)
                if collate_fn is None:
                    prompt = dataset_prompt
                else:
                    prompt_batch = await asyncio.to_thread(
                        collate_fn,
                        [dataset_prompt],
                    )
                    if isinstance(prompt_batch, BatchedDataDict):
                        if prompt_batch.size != 1:
                            raise ValueError(
                                "recovery collation must return exactly one prompt; "
                                f"sample_id={sample_id!r}, size={prompt_batch.size}"
                            )
                        prompt = {key: value[0] for key, value in prompt_batch.items()}
                    elif isinstance(prompt_batch, dict):
                        # Identity-style collators used by lightweight/custom
                        # dataloaders may return the DatumSpec directly.
                        prompt = prompt_batch
                    else:
                        raise TypeError(
                            "recovery collation for "
                            f"sample_id={sample_id!r} returned "
                            f"{type(prompt_batch).__name__}, expected a mapping"
                        )
                resolved_prompts[sample_id] = cast(DatumSpec, prompt)
            recovery_ledger.bind_runtime_prompt(
                cut,
                group.group_id,
                cast(DatumSpec, prompt),
            )

    async def _admit_reserved_prompt_groups(
        self,
        group_ids: list[str],
    ) -> tuple[Optional[int], list[str], int]:
        """Commit one admission and atomically reconcile restored canonical groups.

        Returns:
            The target-step stamp, IDs that still require rollout dispatch, and the
            number of already-canonical groups that replaced reservations in this
            admission.
        """
        if not group_ids:
            raise ValueError("sampler admission requires at least one prompt group")

        def _commit(
            cut: DataPlaneMutationCut,
            target_step: Optional[int],
        ) -> tuple[Optional[int], list[str], int]:
            if target_step is not None:
                self._sampler_stamps_target_steps = True
            for group_id in group_ids:
                self._rollout_manager.mark_prompt_group_admitted(
                    cut,
                    group_id,
                    target_step=target_step,
                )

            buffered = 0
            dispatch_group_ids = group_ids
            if target_step is not None:
                buffered = self._buffer.count_for_target_step(target_step)
                if buffered:
                    dispatch_count = max(0, len(group_ids) - buffered)
                    dispatch_group_ids = group_ids[:dispatch_count]
                    for group_id in group_ids[dispatch_count:]:
                        self._rollout_manager.discard_prompt_group(cut, group_id)
            return target_step, dispatch_group_ids, buffered

        if isinstance(self._sampler, TransactionalAdmissionSampler):
            await self._sampler.wait_until_admissible(
                trainer_version_fn=lambda: self._trainer_version
            )
            async with self._data_plane_checkpoint_barrier.mutation() as cut:
                target_step = self._sampler.commit_admission(cut)
                return _commit(cut, target_step)

        # Custom samplers retain their existing monolithic admission API. Hold
        # the mutation cut across it for correctness. Contract: a custom admit()
        # must not wait on anything beyond a single trainer_version increment --
        # a checkpoint drains mutation slots while blocking the train pump, so a
        # longer wait deadlocks the run. Implement TransactionalAdmissionSampler
        # to keep the gate wait outside the mutation cut entirely.
        async with self._data_plane_checkpoint_barrier.mutation() as cut:
            target_step = await self._sampler.admit(
                trainer_version_fn=lambda: self._trainer_version
            )
            return _commit(cut, target_step)

    async def _redispatch_restored_rollouts(
        self,
        launch: Callable[[DatumSpec, Optional[int], str], Awaitable[None]],
    ) -> None:
        """Prioritize durable unfinished groups while the train pump drains TQ.

        Launching happens inside the ordinary rollout pump so restored groups use
        the same in-flight and replay-capacity semaphores as new work. The train
        pump runs concurrently and releases replay capacity as it consumes
        canonical or newly recovered groups; therefore recovery cannot deadlock
        merely because the checkpoint contained more unfinished ownership records
        than free replay slots.
        """
        recovery_ledger = self._rollout_manager.recovery_ledger
        groups_to_recover = recovery_ledger.groups()
        if not groups_to_recover:
            return

        recognized_phases = (
            PromptGroupPhase.ADMITTED,
            PromptGroupPhase.RESERVED,
        )
        unhandled_groups = [
            group for group in groups_to_recover if group.phase not in recognized_phases
        ]
        if unhandled_groups:
            details = ", ".join(
                f"{group.group_id}={group.phase!r}" for group in unhandled_groups
            )
            raise RuntimeError(f"unrecognized rollout recovery phase(s): {details}")

        # ADMITTED groups may be the only work capable of advancing the trainer and
        # opening the sampler gate. Launch them before waiting to re-admit RESERVED
        # groups, or restore can deadlock with the trainer waiting for recovered work
        # that this method has not launched yet.
        redispatched = 0
        for group in groups_to_recover:
            if group.phase is PromptGroupPhase.ADMITTED:
                await launch(
                    group.prompt_payload,
                    group.target_step,
                    group.group_id,
                )
                redispatched += 1

        # A checkpoint may land after dataloader ownership is recorded but before
        # sampler admission commits. Re-admit each original dataloader batch once and
        # launch it immediately; do not wait for every reserved batch to pass its gate.
        reserved_admissions: dict[str, list[str]] = {}
        for group in groups_to_recover:
            if group.phase is PromptGroupPhase.RESERVED:
                reserved_admissions.setdefault(group.admission_id, []).append(
                    group.group_id
                )
        for group_ids in reserved_admissions.values():
            _, dispatch_group_ids, _ = await self._admit_reserved_prompt_groups(
                group_ids
            )
            for group_id in dispatch_group_ids:
                group = recovery_ledger.get_group(group_id)
                await launch(
                    group.prompt_payload,
                    group.target_step,
                    group.group_id,
                )
                redispatched += 1

        print(
            f"📦 Redispatched {redispatched} unfinished rollout "
            "group(s) before new dataloader work",
            flush=True,
        )

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
                "Native TQ checkpoint inventory does not match "
                f"{REPLAY_BUFFER_METADATA_FILENAME}: "
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

    async def _maybe_restore_replacement_reserve(self) -> None:
        """Restore spare prompts diverted before the previous run's checkpoint.

        These were pulled from the dataloader and never dispatched, so the restored
        dataloader resumes past them. Nothing else in the checkpoint holds them, and
        without this they are simply gone: one batch of the dataset per divert, plus
        the training step ``_clamp_max_num_steps`` had budgeted for it.

        No sampler-name guard, unlike the buffer restore. Spares carry no stamp -- they
        are prompts that never reached ``admit`` -- so nothing about them depends on
        which sampler wrote the checkpoint. They are restored even into a run that has
        since switched to ``on_dropped_prompt="shrink"``, where the pool is never drawn
        on but is still drained back into training at the end of the dataloader.
        """
        if self._last_checkpoint_path is None:
            return
        reserve_path = os.path.join(
            self._last_checkpoint_path, REPLACEMENT_RESERVE_FILENAME
        )
        # Absent for every run that never diverted a batch, which is every run that
        # does not use "replace" -- so silence here rather than the buffer restore's
        # warning, since this is the ordinary case rather than a lost artifact.
        if not os.path.exists(reserve_path):
            return
        # weights_only=False: spares are pickled DatumSpecs, and the checkpoint is a
        # trusted same-job artifact (the replay buffer restore loads on the same terms).
        reserve_state = await asyncio.to_thread(
            torch.load, reserve_path, weights_only=False
        )
        self._replacement_reserve.extend(reserve_state)
        print(
            f"📦 Restored {len(reserve_state)} pooled spare prompt(s) from checkpoint: "
            f"{reserve_path}",
            flush=True,
        )

    async def _ray_get(self, obj_ref: Any) -> Any:
        """Await a Ray ObjectRef without blocking the asyncio event loop."""
        return await obj_ref

    async def _call_dp(self, method_name: str, **kwargs) -> Any:
        """Call a DataPlaneClient method or a Ray actor exposing that method."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await self._ray_get(remote(**kwargs))
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

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
        checkpoint_path: PathLike,
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
        save_state = self._save_state
        checkpoint_trainer_version = save_state.trainer_version
        if checkpoint_trainer_version is None:
            raise RuntimeError(
                "Cannot save a data-plane checkpoint before trainer_version "
                "is captured in the controller save state"
            )
        metadata: DataPlaneCheckpointMetadata = {
            "data_plane_checkpoint_schema_version": (
                DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
            ),
            "single_controller_train_steps": save_state.current_step,
            "single_controller_trainer_version": checkpoint_trainer_version,
            "single_controller_epoch": save_state.current_epoch,
            "partition_id": self._partition_id,
            "sampler_name": self._async_cfg.sampler.name,
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

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch:
          0. Under on_dropped_prompt="replace", divert the batch into the spare pool
             if the pool is below its low-water mark, and skip admission entirely.
             Otherwise await sampler.admit(...) to wait until the batch may dispatch
             and obtain its target_step stamp.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager reserves a slot, runs the rollout, then commits the
             group via TQReplayBuffer (→ dp_client.put_samples + mark ready)
          5. If the prompt was dropped, substitute a spare and repeat step 4 -- for this
             step, or for whichever step lends this one a finished group in its place
             (see _take_replacement, _promote_into_step) -- or credit the step short so
             the train pump can close it
          6. Decrement _inflight_rollouts

        Once every epoch is done, whatever is left in the spare pool is dispatched as
        ordinary steps rather than discarded (see _drain_reserve_into_steps).
        """
        sem = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            lineage_group_id: Optional[str],
            task_started_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            self._inflight_rollouts += 1
            # This task owns one slot of a step, which can outlive both the prompt it
            # started with and the step it started on: a dropped prompt is substituted in
            # place and the loop runs again, and the slot is re-aimed at whichever step
            # lends this one a finished group. Both permits are held across
            # substitutions because the slot stays occupied either way -- they are
            # released once something commits, or once a step is credited short.
            replacements = 0
            try:
                while True:
                    try:
                        if lineage_group_id is None:
                            outcome = await self._rollout_manager.generate_and_push(
                                prompt,
                                target_step=target_step,
                                inflight_registry=self._inflight_by_group_id,
                            )
                        else:
                            outcome = await self._rollout_manager.generate_and_push(
                                prompt,
                                target_step=target_step,
                                inflight_registry=self._inflight_by_group_id,
                                lineage_group_id=lineage_group_id,
                            )
                    except BaseException:
                        # On success ownership transfers to the train pump, which
                        # releases this permit after consuming the committed group.
                        self._buffer_capacity.release()
                        raise

                    if outcome is not RolloutOutcome.SKIPPED:
                        break

                    if self._rollout_recovery_enabled:
                        assert lineage_group_id is not None
                        async with (
                            self._data_plane_checkpoint_barrier.mutation()
                        ) as cut:
                            replacement = self._take_replacement(
                                target_step, replacements
                            )
                            # A skipped tracked prompt remains ledger-owned until this
                            # controller transition. Dropping the old owner, reserving
                            # a replacement, or crediting the target step short must be
                            # one checkpoint-atomic decision.
                            self._rollout_manager.discard_prompt_group(
                                cut, lineage_group_id
                            )
                            if replacement is not None:
                                lender_step = self._promote_into_step(target_step)
                                if lender_step is not None:
                                    target_step = lender_step
                                lineage_group_id = (
                                    self._rollout_manager.reserve_prompt_group(
                                        cut,
                                        replacement,
                                        target_step=target_step,
                                    )
                                )
                            else:
                                self._credit_shortfall(target_step)
                    else:
                        replacement = self._take_replacement(target_step, replacements)
                    if replacement is None:
                        # Nothing was committed, so the train pump will never see this
                        # group and never release its permit on our behalf.
                        self._buffer_capacity.release()
                        if not self._rollout_recovery_enabled:
                            self._credit_shortfall(target_step)
                        return

                    replacements += 1
                    prompt = replacement
                    print(
                        f"  target_step={target_step}: substituting a spare prompt for "
                        f"the dropped group (replacement {replacements}/"
                        f"{self._async_cfg.rollout_failure.max_replacement_attempts}, "
                        f"{len(self._replacement_reserve)} spare(s) left)",
                        flush=True,
                    )
                    # Attempted only now that a spare is in hand, because the borrow is a
                    # debt and the spare is what repays it. Borrowing without one would
                    # leave the lender short instead: the same hole, one step later.
                    if not self._rollout_recovery_enabled:
                        lender_step = self._promote_into_step(target_step)
                        if lender_step is not None:
                            target_step = lender_step
                    # A substitution is a fresh rollout, not a continuation of the one
                    # that failed, so it observes the same pause a first dispatch does
                    # instead of pushing new generation into a weight-sync window.
                    await self._rollout_permitted.wait()
            finally:
                self._inflight_rollouts -= 1
                sem.release()

            if replacements and target_step is not None:
                # Counted per slot, not per attempt: a step got its group back, which is
                # the fact that explains why its batch is full despite a drop. Recorded
                # against the step the spare actually committed to, which after a borrow
                # is the lender rather than the step that was dropped from.
                self._batch_replacements[target_step] = (
                    self._batch_replacements.get(target_step, 0) + 1
                )

            if self._async_cfg.diagnostics:
                content = ""
                for i in range(len(prompt["message_log"])):
                    if prompt["message_log"][i]["role"] == "user":
                        content = prompt["message_log"][i]["content"]
                        break
                print(f"  rollout done for prompt='{content[:20]}...'", flush=True)

        def _release_permits_if_task_not_started(
            _: asyncio.Task[Any],
            *,
            task_started_event: asyncio.Event,
        ) -> None:
            if not task_started_event.is_set():
                self._buffer_capacity.release()
                sem.release()

        async def _launch(
            prompt: DatumSpec,
            target_step: Optional[int],
            lineage_group_id: Optional[str],
        ) -> None:
            if self._rollout_recovery_enabled and lineage_group_id is None:
                raise RuntimeError(
                    "recovery-enabled rollout dispatch requires a pre-reserved "
                    "prompt-group ID"
                )
            # check if buffer is full
            await self._buffer_capacity.acquire()
            # check if inflight rollouts is full
            await sem.acquire()
            # wait for rollout to be permitted
            await self._rollout_permitted.wait()

            task_started_event = asyncio.Event()
            # dispatch rollout
            task = rollout_tasks.create_task(
                _dispatch_one_prompt(
                    prompt,
                    target_step,
                    lineage_group_id,
                    task_started_event,
                )
            )
            self._dispatched_rollouts.add(task)
            task.add_done_callback(self._dispatched_rollouts.discard)
            task.add_done_callback(
                partial(
                    _release_permits_if_task_not_started,
                    task_started_event=task_started_event,
                )
            )

        max_epochs = self._algo_cfg.max_num_epochs
        async with asyncio.TaskGroup() as rollout_tasks:
            if self._rollout_recovery_enabled:
                await self._redispatch_restored_rollouts(_launch)
            while max_epochs is None or self._current_epoch < max_epochs:
                if not self._rollout_recovery_enabled:
                    for prompt_batch in self._dataloader:
                        if self._divert_batch_to_reserve(prompt_batch):
                            continue
                        target_step = await self._sampler.admit(
                            trainer_version_fn=lambda: self._trainer_version
                        )
                        if target_step is not None:
                            self._sampler_stamps_target_steps = True
                        num_prompts = prompt_batch.size
                        if target_step is not None:
                            buffered = self._buffer.count_for_target_step(target_step)
                            if buffered:
                                num_prompts = max(0, prompt_batch.size - buffered)
                                print(
                                    f"  target_step={target_step}: {buffered} group(s) "
                                    f"already buffered; dispatching {num_prompts} of "
                                    f"{prompt_batch.size} prompt(s), dropping the rest",
                                    flush=True,
                                )
                        for prompt_idx in range(num_prompts):
                            prompt: DatumSpec = {  # type: ignore
                                k: v[prompt_idx] for k, v in prompt_batch.items()
                            }
                            await _launch(prompt, target_step, None)
                    self._current_epoch += 1
                    continue

                dataloader_iterator = iter(self._dataloader)
                while True:
                    prompt_dispatches: list[tuple[DatumSpec, str]] = []
                    async with self._data_plane_checkpoint_barrier.mutation() as cut:
                        try:
                            prompt_batch = next(dataloader_iterator)
                        except StopIteration:
                            self._current_epoch += 1
                            break
                        if self._divert_batch_to_reserve(prompt_batch):
                            continue
                        admission_id = str(uuid.uuid4())
                        for prompt_idx in range(prompt_batch.size):
                            prompt = {  # type: ignore
                                k: v[prompt_idx] for k, v in prompt_batch.items()
                            }
                            group_id = self._rollout_manager.reserve_prompt_group(
                                cut,
                                prompt,
                                target_step=None,
                                admitted=False,
                                admission_id=admission_id,
                            )
                            prompt_dispatches.append((prompt, group_id))

                    (
                        target_step,
                        dispatch_group_ids,
                        buffered,
                    ) = await self._admit_reserved_prompt_groups(
                        [group_id for _, group_id in prompt_dispatches]
                    )

                    if target_step is not None:
                        if buffered:
                            print(
                                f"  target_step={target_step}: {buffered} group(s) "
                                f"already buffered; dispatching "
                                f"{len(dispatch_group_ids)} of "
                                f"{len(prompt_dispatches)} prompt(s), dropping the rest",
                                flush=True,
                            )
                            dispatch_group_id_set = set(dispatch_group_ids)
                            prompt_dispatches = [
                                (prompt, group_id)
                                for prompt, group_id in prompt_dispatches
                                if group_id in dispatch_group_id_set
                            ]

                    for prompt, group_id in prompt_dispatches:
                        await _launch(prompt, target_step, group_id)

        # Only now that every dispatched rollout has settled is the pool genuinely
        # spare. Draining it inside the group above would race them for it, and a
        # rollout that was about to be dropped has the better claim: it needs a spare to
        # keep its step whole, whereas an extra step is only worth having if one is
        # left over. A second group because the first is closed to new tasks.
        async with asyncio.TaskGroup() as rollout_tasks:
            await self._drain_reserve_into_steps(_launch)

        # Drain in-flight so return implies "all rollouts in TQ".
        inflight = list(self._dispatched_rollouts)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        self._rollout_exhausted.set()
        print(f"rollout_pump: completed {self._current_epoch} epoch(s)", flush=True)

    def _divert_batch_to_reserve(
        self, prompt_batch: BatchedDataDict[DatumSpec]
    ) -> bool:
        """Consume a whole batch as spare prompts instead of admitting it as a step.

        Returns whether the batch was taken, in which case the caller must not admit it.
        Diverting before ``admit`` is what keeps the stamp sequence honest: admitting a
        batch and then dispatching nothing for it would leave a target step that no
        group is ever generated for, which is exactly the hang the shortfall accounting
        exists to prevent.

        A whole batch at a time because the dataloader only yields batches. The spares
        that go unused are not wasted work -- nothing has been generated for them -- and
        they stay in the pool for later steps.

        Nothing is diverted until the sampler has actually stamped a batch, so a run
        whose sampler never stamps does not lose a batch of prompts to a pool it can
        never draw on. The cost is that the first batch is always admitted rather than
        diverted; in practice the pool is filled while that first batch's rollouts are
        still running, so it is available by the time any of them can be given up on.
        """
        failure_cfg = self._async_cfg.rollout_failure
        if failure_cfg.on_dropped_prompt != "replace":
            return False
        if not self._sampler_stamps_target_steps:
            return False
        if len(self._replacement_reserve) >= failure_cfg.replacement_reserve_prompts:
            return False

        for prompt_idx in range(prompt_batch.size):
            spare: DatumSpec = {  # type: ignore
                k: v[prompt_idx] for k, v in prompt_batch.items()
            }
            self._replacement_reserve.append(spare)
        print(
            f"  spare pool refilled with {prompt_batch.size} prompt(s) "
            f"(low-water mark {failure_cfg.replacement_reserve_prompts}); this batch is "
            "not admitted as a training step",
            flush=True,
        )
        return True

    async def _drain_reserve_into_steps(
        self,
        launch: Callable[[DatumSpec, Optional[int], Optional[str]], Awaitable[None]],
    ) -> None:
        """Train on the leftover spares once the dataloader has nothing more to give.

        Spares were consumed from the dataset like any other prompt, so leaving them in
        the pool at the end of the last epoch throws away data the run already paid for.

        It also restores the step count. ``_clamp_max_num_steps`` derives
        ``max_num_steps`` from ``len(dataloader)``, and every diverted batch is one
        fewer batch the loop can admit -- so without this a replace-mode run quietly
        finishes one step short of the budget it was configured with, per divert.

        Whole steps only. A partial pool dispatched as a step is short by construction,
        and ``min_step_batch_fraction`` would then reject it and fail a run that had
        otherwise completed cleanly. In the ordinary case the pool holds exactly one
        batch (the dataloader uses ``batch_size=num_prompts_per_step``), so the common
        outcome is that the whole thing is recovered.

        Not gated on ``on_dropped_prompt``: an empty pool makes this a no-op anyway, and
        only "replace" ever fills one, so the gate would buy nothing while stranding a
        pool restored from a checkpoint into a run that has since switched to "shrink".
        """
        num_prompts_per_step = self._algo_cfg.num_prompts_per_step
        while len(self._replacement_reserve) >= num_prompts_per_step:
            if self._rollout_recovery_enabled:
                prompt_dispatches: list[tuple[DatumSpec, str]] = []
                async with self._data_plane_checkpoint_barrier.mutation() as cut:
                    step_prompts = [
                        self._replacement_reserve.popleft()
                        for _ in range(num_prompts_per_step)
                    ]
                    admission_id = str(uuid.uuid4())
                    for prompt in step_prompts:
                        group_id = self._rollout_manager.reserve_prompt_group(
                            cut,
                            prompt,
                            target_step=None,
                            admitted=False,
                            admission_id=admission_id,
                        )
                        prompt_dispatches.append((prompt, group_id))
                (
                    target_step,
                    dispatch_group_ids,
                    buffered,
                ) = await self._admit_reserved_prompt_groups(
                    [group_id for _, group_id in prompt_dispatches]
                )
                dispatch_group_id_set = set(dispatch_group_ids)
                prompt_dispatches = [
                    (prompt, group_id)
                    for prompt, group_id in prompt_dispatches
                    if group_id in dispatch_group_id_set
                ]
                print(
                    f"  dataloader exhausted; training on {len(prompt_dispatches)} "
                    f"pooled spare(s) as target_step={target_step}"
                    + (f" ({buffered} group(s) already buffered)" if buffered else ""),
                    flush=True,
                )
                for prompt, group_id in prompt_dispatches:
                    await launch(prompt, target_step, group_id)
                continue

            # Take the step's prompts out before the first await. A drop resolving
            # concurrently draws from this same pool, and could otherwise claim one of
            # them and leave the step it is filling one group short.
            step_prompts = [
                self._replacement_reserve.popleft() for _ in range(num_prompts_per_step)
            ]
            target_step = await self._sampler.admit(
                trainer_version_fn=lambda: self._trainer_version
            )
            print(
                f"  dataloader exhausted; training on {len(step_prompts)} pooled "
                f"spare(s) as target_step={target_step}",
                flush=True,
            )
            for prompt in step_prompts:
                await launch(prompt, target_step, None)

        if self._replacement_reserve:
            print(
                f"  {len(self._replacement_reserve)} pooled spare(s) left over, fewer "
                f"than the {num_prompts_per_step} a step needs; they are not trained on",
                flush=True,
            )

    def _take_replacement(
        self, target_step: Optional[int], replacements_used: int
    ) -> Optional[DatumSpec]:
        """A spare prompt to stand in for a dropped group, or None to shrink instead.

        None covers the four ways a replacement can be unavailable: it was not asked
        for, the sampler did not stamp this prompt so no step is waiting on it, the
        per-slot budget is spent, or the pool is empty because the dataloader is
        exhausted. Every one of them falls back to ``on_dropped_prompt="shrink"`` rather
        than waiting, because a step whose replacements keep failing still has to close.
        """
        failure_cfg = self._async_cfg.rollout_failure
        if failure_cfg.on_dropped_prompt != "replace":
            return None
        if target_step is None:
            return None
        if replacements_used >= failure_cfg.max_replacement_attempts:
            return None
        if not self._replacement_reserve:
            return None
        return self._replacement_reserve.popleft()

    def _promote_into_step(self, target_step: Optional[int]) -> Optional[int]:
        """Fill a dropped step from a later step's finished work, and name the lender.

        Where a replacement goes, rather than whether one happens. The lost step closes
        on generation that already exists instead of waiting out a rollout with the
        trainer idle, and the caller redirects its spare prompt to the lender, which is
        due a training step later and has the slack to absorb the wait. The same prompt
        is generated either way.

        Only ever reached with a spare already in hand, which is what makes the borrow
        safe to take: an unrepaid loan is the same hole one step later.

        Returns None -- leaving the caller filling the dropped step directly -- when
        nothing is stamped so no step is stranded, when the trainer has already moved
        past this step (a second drop can land after the first one closed it short, and
        a group stamped for a finished step would only be evicted), or when no later
        step has a finished group to lend. The last is always the case at
        ``in_order.max_lookahead_versions=0``, where the next batch is not dispatched
        until this step trains.

        Returns:
            The step that lent the group, which the caller now owes a rollout, or None.
        """
        if target_step is None:
            return None
        if target_step < self._trainer_version:
            return None
        lender_step = self._buffer.promote_ready_group(to_target_step=target_step)
        if lender_step is None:
            return None
        self._batch_promotions[target_step] = (
            self._batch_promotions.get(target_step, 0) + 1
        )
        print(
            f"  target_step={target_step}: filled the dropped group by promoting a "
            f"finished group from target_step={lender_step}; the spare prompt is "
            "dispatched to repay that step instead",
            flush=True,
        )
        return lender_step

    def _credit_shortfall(self, target_step: Optional[int]) -> None:
        """Record that a stamped step will never receive a group it is waiting for."""
        if target_step is None:
            return
        self._batch_shortfall[target_step] = (
            self._batch_shortfall.get(target_step, 0) + 1
        )
        print(
            f"  target_step={target_step} is one group short "
            f"({self._batch_shortfall[target_step]} total); the train pump "
            "will close that step early",
            flush=True,
        )

    def _target_groups_for_step(self, step: int) -> int:
        """How many prompt groups this step should train on, after dropped prompts.

        ``num_prompts_per_step`` is the target; groups stamped for this step that were
        given up on are subtracted, because they are never arriving and a sampler that
        matches batches to steps exactly cannot substitute another step's groups for
        them. Without this the pump waits on a group no one is generating.

        The step trains on fewer samples than configured, which is the point: a smaller
        step beats a stalled run. The count is logged as ``dropped_prompt_groups`` so
        the batch size a step actually used is recoverable afterwards.

        How much smaller is bounded by ``min_step_batch_fraction``, and that bound has
        to live here because neither drop budget provides it. Both budgets are
        run-scoped -- the consecutive counter is cleared by any commit, including
        commits for other steps -- so drops landing on one step while other steps
        succeed can shrink it without ever tripping them.

        Raises:
            RuntimeError: The step fell below ``min_step_batch_fraction`` of
                ``num_prompts_per_step``. Training a fraction of a batch is a silent
                change to the gradient estimate, so it is refused rather than absorbed.
        """
        num_prompts_per_step = self._algo_cfg.num_prompts_per_step
        dropped = self._batch_shortfall.get(step, 0)
        target = num_prompts_per_step - dropped
        fraction = self._async_cfg.rollout_failure.min_step_batch_fraction
        # ceil, so the floor is never rounded down into allowing one more drop than the
        # fraction states. With fraction > 0 this is always >= 1, which also rules out
        # the empty step.
        floor = math.ceil(num_prompts_per_step * fraction)
        if target < floor:
            raise RuntimeError(
                f"training step {step} lost {dropped} of {num_prompts_per_step} prompt "
                f"group(s), leaving {target}, below the floor of {floor} set by "
                f"async_rl.rollout_failure.min_step_batch_fraction={fraction}. "
                "Either the generation fleet is failing a whole step's worth of "
                "prompts, or the drop budgets are set too high to catch it: they are "
                "run-scoped and cannot bound how short a single step gets."
            )
        return target

    async def _train_pump(self) -> None:
        """Per-prompt-group streaming train loop.

        Per step, with 1-4 running per streaming chunk and 5-6 once the chunk
        loop closes:
            1. Select the rollouts to train on.
                a. sampler.evict drops stale groups from the buffer and clears their
                    TQ rows.
                b. sampler.select returns K prompt groups, or None, and removes them from
                    the buffer. The DP rows survive, already training-shaped because the
                    buffer wrote them that way at rollout time.
                c. One _buffer_capacity permit is released per group that left the buffer.
            2. Prepare the batch.
                a. Policy and reference logprobs.
                b. Value model forward (PPO only), with the policy parked on CPU
                    so the critic never shares the training GPUs with it.
                c. _advantage_stage.
            3. Train on the chunk.
                a. Value model (PPO only): train_from_meta, which is a whole
                    optimizer step. That is why a PPO step is pinned to a single
                    chunk -- the value workers have no split train API yet (#2625).
                b. Policy model: train_microbatches_from_meta, which only
                    accumulates gradients.
                c. PPO only: all critic updates run before all policy updates.
                    Their counts are ppo.critic_ppo_epochs and ppo.ppo_epochs,
                    respectively. Each policy optimizer step closes here rather
                    than in 5 -- a PPO step is one chunk, so there is nothing to
                    accumulate across chunks.
            4. Clear the batch. dp_client.clear_samples on the consumed sample_ids.
            5. Train the policy model (GRPO) -- finish_train_step all_reduces the
                accumulated gradients, rescales, and runs optimizer.step.
            6. Refit the model. Sync the new policy weights to generation.

        PPO critic warmup (ppo.policy_training_start_step > 0) changes which of those
        run. For the first N steps 3a still trains the critic every step, but 3b and 6
        are skipped, so the policy is neither trained nor refit. The trainer version
        still advances, and the sampler's lookahead is widened while the policy is
        frozen.
        """
        policy_training_start_step = (
            self._algo_cfg.policy_training_start_step if self._is_ppo else 0
        )

        while self._train_steps < self._algo_cfg.max_num_steps:
            version_during_step = self._trainer_version
            groups_dispatched = 0
            evicted_stale_prompt_groups = 0
            min_sample_version = None
            step_open = False
            chunks_dispatched = 0
            calibration_batches: list[BatchedDataDict[Any]] = []
            selected_rollout_metrics: list[dict[str, Any]] = []
            # One chunk per step on the PPO path, so these are the step's own
            # model updates -- the last epoch's, when there is more than one.
            policy_result: Optional[dict[str, Any]] = None
            value_result: Optional[dict[str, Any]] = None
            # Always True off the PPO path: the start step is pinned to 0 there.
            is_policy_training_step = self._train_steps >= policy_training_start_step

            with self._timer.time("total_step_time"):
                # Re-read on every iteration rather than once: a prompt stamped for this
                # step can be dropped while the pump is already waiting for it, which is
                # precisely the case that would otherwise wait forever.
                while groups_dispatched < self._target_groups_for_step(
                    version_during_step
                ):
                    # ---- 1. Select the rollouts to train on ----
                    with self._timer.time("exposed_generation"):
                        await asyncio.sleep(0)

                        # Evict stale groups
                        evicted = await self._sampler.evict(
                            current_train_weight=self._trainer_version,
                        )
                        evicted_stale_prompt_groups += evicted
                        if evicted:
                            print(
                                f"  evicted {evicted} stale prompt group(s)",
                                flush=True,
                            )
                            for _ in range(evicted):
                                self._buffer_capacity.release()

                        # Select a batch. Read the target again rather than reusing
                        # the loop condition's value: the awaits above are a window in
                        # which a prompt stamped for this step can be dropped, and the
                        # target would then be stale by the time it is subtracted. It
                        # can also have fallen to what is already dispatched, which is
                        # not a batch the sampler can be asked for -- select() rejects
                        # a min below 1 -- so close the step instead.
                        target_groups = self._target_groups_for_step(
                            version_during_step
                        )
                        max_prompt_groups = target_groups - groups_dispatched
                        if max_prompt_groups <= 0:
                            break
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
                            if self._rollout_exhausted.is_set():
                                buffered_groups = len(self._buffer)
                                if groups_dispatched == 0 and buffered_groups == 0:
                                    print(
                                        "train_pump: rollout exhausted and "
                                        "buffer drained",
                                        flush=True,
                                    )
                                    return
                                # Against the step's own target, not the configured
                                # batch size: a step that legitimately shrank would
                                # otherwise be reported as missing groups it was
                                # already excused from.
                                raise RuntimeError(
                                    "rollout exhausted before a complete training "
                                    f"step was assembled: dispatched "
                                    f"{groups_dispatched}/{target_groups} prompt "
                                    f"groups with {buffered_groups} group(s) "
                                    f"remaining in the buffer"
                                )
                            await asyncio.sleep(0.005)
                            continue

                        # Release buffer capacity
                        for _ in range(num_groups):
                            self._buffer_capacity.release()

                        selected_rollout_metrics.extend(
                            train_meta.extra_info.pop(ROLLOUT_METRICS, [])
                        )

                    if groups_dispatched == 0 and self._gen is not None:
                        await asyncio.to_thread(self._gen.snapshot_step_metrics)

                    # ---- 2. Prepare the batch ----
                    # Compute prev_logprobs / ref_logprobs
                    if (
                        self._policy_logprobs_required
                        or self._reference_logprobs_required
                    ):
                        with self._timer.time("logprob_inference_prep"):
                            # Once the step is open, gradients are accumulating
                            # in the trainer's grad buffers across chunks. The
                            # Megatron buffer offload frees that storage outright
                            # and its reload zeroes it, so offloading here would
                            # discard every chunk but the last while the 1/N
                            # normalizer still counts all of them.
                            await asyncio.to_thread(
                                self._trainer.prepare_for_lp_inference,
                                keep_train_buffers=step_open,
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
                    elif self._is_ppo:
                        # prepare_for_lp_inference is skipped here, and it is the only
                        # other call that parks the policy optimizer before the critic.
                        with self._timer.time("value_inference_prep"):
                            await asyncio.to_thread(self._trainer.offload_to_cpu)

                    # Value model forward
                    if self._is_ppo:
                        with self._timer.time("value_inference"):
                            await asyncio.to_thread(self._trainer.finish_inference)
                            train_meta = await self._value_stage(train_meta)

                    # Compute advantages
                    with self._timer.time("advantage_calculation"):
                        (
                            train_meta,
                            has_valid_training_tokens,
                        ) = await self._advantage_stage(train_meta)

                    # A PPO step is this one chunk, so a chunk with nothing left
                    # after filtering is a step that trains neither model.
                    if self._is_ppo and not has_valid_training_tokens:
                        raise RuntimeError(
                            "SingleController has no valid response tokens after "
                            "filtering. Check ppo.seq_logprob_error_threshold to "
                            "avoid an optimizer step with an empty batch."
                        )

                    # ---- 3. Train the model -- train_microbatches_from_meta ----
                    # Filtering can leave a GRPO streaming chunk with no training
                    # tokens. Consume that chunk without F/B, then continue the same
                    # optimizer step with the next chunk.

                    # GRPO runs one iteration: F/B only, its optimizer step is in 5.
                    # For PPO, each actor epoch and critic epoch is a full optimizer
                    # step. Group each model's epochs under one residency cycle so
                    # the colocated models do not move between CPU and GPU per epoch.
                    # TODO(#2625): value_result, policy_result only record the last epoch's metrics.
                    # That matches ppo.py for the losses; total_flops is additive and undercounted.
                    if self._is_ppo:
                        with self._timer.time("value_training"):
                            value_result = await self._value_train_epochs(
                                train_meta,
                                num_epochs=self._critic_ppo_epochs,
                            )

                    if is_policy_training_step:
                        if (
                            self._is_ppo
                            and self._train_steps == policy_training_start_step
                            and policy_training_start_step > 0
                        ):
                            print(
                                f"  ✓ Critic warmup complete ({policy_training_start_step} "
                                "steps). Starting policy training.",
                                flush=True,
                            )
                        # Always restore training mode because log-prob inference may have
                        # switched the model to inference mode. Keep it resident
                        # across every PPO actor epoch.
                        with self._timer.time("training_prep"):
                            await asyncio.to_thread(self._trainer.prepare_for_training)

                        if has_valid_training_tokens:
                            for _ in range(self._ppo_epochs):
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
                                        train_fields=self._train_fields,
                                    )
                                    # A PPO step is one chunk: nothing to
                                    # accumulate, so close every epoch here.
                                    if self._is_ppo:
                                        policy_result = await asyncio.to_thread(
                                            self._trainer.finish_train_step
                                        )
                                        step_open = False

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

                    # ---- 4. Clear the batch ----
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
                    chunks_dispatched += 1
                    # How many chunks a step is split into decides how many times
                    # gradients accumulate before the single reduce, so record it
                    # rather than leaving it to be inferred from phase timings.
                    #
                    # These reach a run's output only because nemo_rl/__init__.py
                    # sets the `nemo_rl` logger to NRL_LOG_LEVEL (INFO by
                    # default); the bare basicConfig() there pins the root logger
                    # at WARNING and no later basicConfig can raise it. Note that
                    # the handler writes to stderr while the progress prints
                    # around this write to stdout, so the two are not guaranteed
                    # to interleave in order in a Ray driver log.
                    log.info(
                        "train_pump: step %d chunk %d: %d group(s), %d/%d dispatched",
                        version_during_step,
                        chunks_dispatched,
                        num_groups,
                        groups_dispatched,
                        self._algo_cfg.num_prompts_per_step,
                    )

                # ---- 5. Train the policy model -- finish_train_step ----
                log.info(
                    "train_pump: step %d closing on %d chunk(s), %d group(s)",
                    version_during_step,
                    chunks_dispatched,
                    groups_dispatched,
                )

                # Only the streaming path has anything left open: a PPO step is one
                # chunk, so each epoch already closed its own optimizer step above.
                if not self._is_ppo:
                    if not step_open:
                        raise RuntimeError(
                            "SingleController has no valid response tokens after "
                            "filtering. Check grpo.seq_logprob_error_threshold to "
                            "avoid an optimizer step with an empty batch."
                        )

                    with self._timer.time("policy_training"):
                        policy_result = await asyncio.to_thread(
                            self._trainer.finish_train_step
                        )

                # Aggregate step metrics
                step_metrics = {}
                if policy_result is not None:
                    step_metrics.update(aggregate_step_metrics(policy_result))
                if value_result is not None:
                    step_metrics.update(_compute_critic_metrics(value_result))
                step_metrics.update(
                    reduce_advantage_pump_metrics(**self._step_log_dict)
                )
                per_group_rollout_metrics: dict[str, list[Any]] = {}
                for group_metrics in selected_rollout_metrics:
                    for metric_name, value in group_metrics.items():
                        per_group_rollout_metrics.setdefault(metric_name, []).append(
                            value
                        )
                step_metrics.update(
                    aggregate_rollout_metrics(per_group_rollout_metrics)
                )
                if self._gen is not None:
                    step_metrics.update(
                        await asyncio.to_thread(self._gen.get_step_metrics)
                    )
                self._step_log_dict = {k: [] for k in self._step_log_dict}
                step_metrics.update(
                    _pooled_opd_metrics(
                        self._opd_stat_sum,
                        self._opd_stat_sumsq,
                        self._opd_stat_count,
                    )
                )
                self._opd_stat_sum = 0.0
                self._opd_stat_sumsq = 0.0
                self._opd_stat_count = 0
                if self._teacher_coordinator is not None:
                    step_metrics.update(self._teacher_coordinator.drain_metrics())

                self._trainer_version += 1
                self._train_steps += 1
                dropped_prompt_groups = self._batch_shortfall.get(
                    version_during_step, 0
                )
                replaced_prompt_groups = self._batch_replacements.get(
                    version_during_step, 0
                )
                promoted_prompt_groups = self._batch_promotions.get(
                    version_during_step, 0
                )
                # Prune every stamp this step or older. Popping only this step's entry
                # would leak the ones belonging to a step that was already closed when
                # a straggler stamped for it was finally given up on.
                self._batch_shortfall = {
                    step: dropped
                    for step, dropped in self._batch_shortfall.items()
                    if step > version_during_step
                }
                self._batch_replacements = {
                    step: replaced
                    for step, replaced in self._batch_replacements.items()
                    if step > version_during_step
                }
                self._batch_promotions = {
                    step: promoted
                    for step, promoted in self._batch_promotions.items()
                    if step > version_during_step
                }

                # ---- 6. Refit the model ----
                with self._timer.time("weight_sync"):
                    calibration_data = (
                        BatchedDataDict.from_batches(calibration_batches)
                        if calibration_batches
                        else None
                    )
                    # Critic warmup doesn't need refit, and the version still advances.
                    aborted_stale_inflight_groups = 0
                    if is_policy_training_step:
                        aborted_stale_inflight_groups = await self._sync_weights(
                            calibration_data=calibration_data
                        )
                    self._retune_lookahead_versions()
                    self._rollout_manager.set_weight_version(self._trainer_version)
                    step_metrics.update(
                        {
                            "evicted_stale_prompt_groups": evicted_stale_prompt_groups,
                            "aborted_stale_inflight_groups": aborted_stale_inflight_groups,
                            # Non-zero means this step trained on a smaller batch than
                            # num_prompts_per_step, which any comparison of step metrics
                            # across steps has to account for.
                            "dropped_prompt_groups": dropped_prompt_groups,
                            # Groups filled by a spare prompt this step waited on --
                            # either one it lost itself, or one it lent to an earlier
                            # step and was repaid for. Non-zero here with zero above is
                            # the healthy shape of on_dropped_prompt="replace": the
                            # batch stayed whole, and the cost was the wall-clock spent
                            # waiting on the spare.
                            "replaced_prompt_groups": replaced_prompt_groups,
                            # Groups this step lost and filled by borrowing finished work
                            # from a later step. The better shape of the same thing: the
                            # batch stayed whole and nothing waited for it, with the
                            # repayment showing up as a replacement on the lender.
                            "promoted_prompt_groups": promoted_prompt_groups,
                        }
                    )

                # Checkpointing (mirrors async_grpo_train's save block).
                # What the step actually trained on, which is num_prompts_per_step only
                # when nothing was dropped. Counted from the dispatch tally rather than
                # derived from the shortfall so the figure does not depend on the
                # bookkeeping staying exact; this lands in the checkpoint.
                self._consumed_samples += groups_dispatched
                self._total_valid_tokens += step_metrics.get("global_valid_toks", 0)
                self._timeout.mark_iteration()

                is_last_step = self._train_steps >= self._algo_cfg.max_num_steps or (
                    self._rollout_exhausted.is_set() and len(self._buffer) == 0
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
                should_save_by_timeout = self._timeout.check_save()

                if self._master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    with self._timer.time("checkpointing"):
                        await self._save_checkpoint(
                            step_metrics,
                            is_policy_training_step=is_policy_training_step,
                        )

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
            #   histogram log, pretty-print "Training Results" block,
            #   print_performance_metrics.
            printable_step_metrics = {
                name: value
                for name, value in step_metrics.items()
                if not isinstance(value, list)
            }
            print(f"step_metrics={printable_step_metrics}", flush=True)
            self._logger.log_metrics(
                step_metrics, step=self._train_steps, prefix="train"
            )
            # step_finished=True here since this is the final log of our current step.
            self._logger.log_metrics(
                timing_metrics,
                step=self._train_steps,
                prefix="timing/train",
                step_finished=True,
            )
            self._timer.reset()

            # min sample version refers to the version each consumed sample was
            # generated with; lag = training version - oldest sample version.
            lag = version_during_step - min_sample_version  # type: ignore
            print(
                f"train step {self._train_steps}/{self._algo_cfg.max_num_steps}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                break

    async def _stall_watchdog_pump(self) -> None:
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
        watchdog_cfg = self._async_cfg.stall_watchdog
        max_num_steps = self._algo_cfg.max_num_steps
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
            if self._gen_fleet is not None:
                metrics.update(self._gen_fleet.as_metrics())
            if self._generation_router is not None:
                # router/* counters are exactly what you want when a backend starts
                # failing; computed since P2 landed but never published until now.
                # Best-effort like the membership push: a router being recreated must
                # not cost a metrics tick.
                try:
                    metrics.update(
                        await self._ray_get(self._generation_router.metrics.remote())
                    )
                except Exception as error:  # noqa: BLE001 - metrics are advisory
                    print(
                        f"watchdog: router metrics unavailable this tick: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
            self._logger.log_metrics(
                metrics, step=self._train_steps, step_metric="rollout/train_steps"
            )

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

            if self._gen_fleet is not None and not self._recovering_from_refit:
                # Raises once too few shards remain for the run to be worth continuing.
                # Checked after publishing so the final state is on record.
                #
                # Skipped mid-recovery: the serving set is empty by construction there,
                # not because the fleet is gone. See _recovery_window.
                self._gen_fleet.raise_if_exhausted()

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

    async def _gen_fleet_probe_pump(self) -> None:
        """Probe the generation fleet on its own clock.

        Separate from the watchdog because the two cadences answer different questions.
        The watchdog publishes counters and notices a stalled run, which is a
        minutes-scale concern; liveness detection is the input to every recovery
        decision and has to be seconds-scale.

        Sharing the watchdog's loop made ``probe_interval_s`` decorative -- probes ran at
        ``watchdog.interval_s`` and nothing read the configured value. With the shipped
        defaults that put detection at ``30s * unhealthy_threshold``, i.e. 60-90s, which
        is *longer* than the refit deadline: by the time a hung refit aborted, the monitor
        still had the dead shard as SUSPECT, so the rebuild that abort exists to trigger
        saw an empty absent set and did nothing. Arithmetic, not a race -- it could never
        have worked. Job 5925668.
        """
        interval_s = self._async_cfg.generation_fleet_health.probe_interval_s
        while True:
            await asyncio.sleep(interval_s)
            await self._probe_generation_fleet()
            # Both of these are best-effort: they talk to a max_restarts=-1 actor that
            # may be mid-recreation, and run() awaits this task and re-raises, so an
            # unguarded RayActorError here would end the training job over a push that
            # the next tick would have retried anyway. GenerationFleetExhausted from the
            # watchdog stays the only fatal path -- the same bounded-failure contract
            # _check_env_health follows.
            try:
                await self._drain_router_failures()
                # Pushed here rather than on the watchdog's clock so a membership change
                # reaches the router at detection speed.
                await self._push_router_membership()
            except Exception as error:  # noqa: BLE001 - best-effort, retried next tick
                print(
                    f"fleet probe: router update failed, retrying next tick: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

    async def _probe_generation_fleet(self) -> None:
        """Ask every serving generation shard whether it is still alive.

        Ray actor liveness is the cheap authoritative signal for "the process is gone",
        and it is what the probe uses. It does not catch every failure -- a vLLM engine
        core can die while the worker process and its HTTP thread survive -- which is
        why the routing adapters also report the failures they observe. The two signals
        feed the same counters.

        Only serving shards are probed: a quarantined shard answering again says nothing
        about whether its weights are current, and the monitor ignores such probes
        anyway.

        Shards are probed concurrently. Sequentially, a tick costs up to
        ``probe_timeout_s`` per shard, so a fleet of four would take 8s to complete a
        round the config promises every 5s -- and config validation only checks
        ``probe_timeout_s < probe_interval_s``, which silently assumes one probe per
        tick. Concurrent, a round is bounded by ``probe_timeout_s`` at any fleet size.
        """
        if self._gen_fleet is None:
            return

        fleet_cfg = self._async_cfg.generation_fleet_health
        worker_group = self._gen.worker_group

        async def probe(shard_idx: int) -> None:
            worker_idx = worker_group.get_dp_leader_worker_idx(shard_idx)
            try:
                await asyncio.wait_for(
                    self._ray_get(worker_group.workers[worker_idx].is_alive.remote()),
                    timeout=fleet_cfg.probe_timeout_s,
                )
            except RayActorError as error:
                # Conclusive, unlike a timeout: Ray only reports this once the actor
                # process is actually gone. Counting it as one more ambiguous failure
                # would delay the verdict by unhealthy_threshold intervals for no gain,
                # and the refit deadline can expire inside that delay.
                self._gen_fleet.record_actor_death(
                    shard_idx, error=f"{type(error).__name__}: {error}"
                )
                # A DEATH, not a silence -- so stand the trainers' refit deadline down.
                # A dead peer closes its sockets and NCCL unblocks the survivors by
                # itself; the deadline would abort first and orphan kernels on the
                # trainers' streams, leaving a CUDA context no rebuild can use. See
                # stand_down_armed_watchdogs.
                self._stand_down_refit_deadline(shard_idx)
            except (Exception, asyncio.TimeoutError) as error:
                self._gen_fleet.record_probe(
                    shard_idx, ok=False, error=f"{type(error).__name__}: {error}"
                )
            else:
                self._gen_fleet.record_probe(shard_idx, ok=True)

        await asyncio.gather(*(probe(idx) for idx in self._gen_fleet.serving_shards()))

    async def _push_router_membership(self) -> None:
        """Tell the NeMo-Gym router which backends are currently serving.

        Pushed as the full set rather than a delta, so a dropped or reordered update --
        or a restarted router, which comes up believing every backend serves -- converges
        on the next tick without sequence numbers or replay.

        Pushed unconditionally, not gated on the membership epoch moving. The gate looked
        free -- an unchanged serving set costs nothing to skip -- but it made the router's
        own restart unrecoverable: a recreated actor rebuilds ``_serving`` as *every*
        backend, while the epoch it was last pushed at has not moved, so the gate blocked
        every corrective push and Gym routed to a quarantined shard for the rest of the
        run. The payload is a short list of strings on a probe-interval timer; the gate
        bought nothing and cost the guarantee both docstrings advertised.

        It is also what makes the router's reflex drop safe: dropping a failing backend
        locally is only correct because a later push puts it back.
        """
        if self._generation_router is None or self._gen_fleet is None:
            return
        await self._ray_get(
            self._generation_router.set_serving_backends.remote(
                self._gen_fleet.serving_base_urls()
            )
        )

    async def _drain_router_failures(self) -> None:
        """Fold the router's observed backend outcomes into the fleet ledger.

        The router is the only component that sees a *wedged* engine: it answers
        ``is_alive`` from a healthy worker process, so no probe can condemn it. The
        router holds no monitor reference by design -- membership flows one way -- so it
        counts per backend URL and this drains them here, on the tick that already talks
        to it.

        BOTH halves, and the successes first. ``consecutive_reported_failures`` is the only
        counter that can condemn a wedged engine, and it is a *streak* -- but on the router
        path nothing ever cleared it. ``report_success`` has exactly one caller, on the
        native adapter, so a router run made the streak monotonic and every shard reached
        ``unhealthy_threshold`` eventually, however healthy. Three unrelated blips days
        apart, with thousands of successes between them, condemned a shard.

        Successes first because they describe the same window: replaying failures onto a
        streak that a success in that window should already have cleared is what made the
        count monotonic in the first place. A genuinely wedged shard produces no successes,
        so its condemnation timing is unchanged.

        Deliberately not a reset on a clean *probe*: the reported streak is kept separate
        from the probe streak precisely because a wedged engine still answers ``is_alive``.
        """
        if self._generation_router is None or self._gen_fleet is None:
            return
        outcomes: dict[str, tuple[int, int]] = await self._ray_get(
            self._generation_router.drain_backend_outcomes.remote()
        )
        for url, (successes, failures) in outcomes.items():
            shard_idx = self._gen_fleet.shard_for_base_url(url)
            if shard_idx is None:
                continue
            if successes:
                self._gen_fleet.report_success(shard_idx)
            for _ in range(failures):
                self._gen_fleet.report_failure(
                    shard_idx,
                    RuntimeError(f"router: {failures} failed request(s) to {url}"),
                )

    def _stand_down_refit_deadline(self, shard_idx: int) -> None:
        """Tell every policy worker to cancel an in-flight refit deadline.

        Fire-and-forget, and deliberately not awaited: this runs inside a probe whose job
        is to keep the ledger current, and a worker that cannot answer is already the
        larger problem. Reaching the worker at all depends on the refit running off its
        event loop -- see await_off_loop.

        Only ever called for a CONFIRMED actor death. A frozen rank never produces one, so
        its deadline still fires and still ends the run attributably.
        """
        try:
            for worker in self._trainer.worker_group.workers:
                worker.stand_down_refit_watchdog.remote()
        except Exception as error:  # noqa: BLE001 - the probe must not die of this
            print(
                f"  refit: could not stand the deadline down after shard {shard_idx} "
                f"died: {type(error).__name__}: {error}",
                flush=True,
            )
        else:
            print(
                f"  refit: shard {shard_idx} is confirmed gone; standing the trainers' "
                "refit deadline down so NCCL's own dead-peer path can unblock them",
                flush=True,
            )

    async def _reconcile_refit_membership(self, force: bool = False) -> Optional[bool]:
        """Ask the weight transport to match the live fleet before the refit runs.

        A no-op without fleet health: with no monitor there is no notion of a shard being
        gone, so the transport keeps the membership it was built with -- which is the
        pre-existing behaviour, and why this is inert by default.

        Returns whether the communicator was actually rebuilt, or None when the
        transport owns no membership at all. The recovery path needs all three: after an
        abort the old communicator is gone, so "nothing to reconcile" means there is
        nothing to retry with either -- but "this transport has no membership" is a
        different refusal and deserves a different message.

        ``force`` says the communicator is gone rather than merely unchanged. The
        synchronizers skip a rebuild when the absent set matches what they last built with,
        which is what stops a lost shard costing two full rebuilds on every subsequent step
        -- but after an abort the membership is identical and the communicator is dead, so
        the recovery path has to override that or it retries over nothing.
        """
        if self._gen_fleet is None:
            return False
        absent = self._gen_fleet.absent_shards()
        # to_thread like every other call that reaches the workers: this can rebuild
        # communicators via blocking Ray calls, and running it on the loop would freeze
        # the watchdog, which is an asyncio task on that same loop.
        rebuilt = await asyncio.to_thread(
            self._weight_synchronizer.reconcile_communicator, absent, force
        )
        # Log unconditionally, not just on a rebuild.
        #
        # Job 5898311 wedged with both policy workers stuck in the refit broadcast and no
        # rebuild logged, and "no rebuild logged" could not distinguish two causes needing
        # opposite fixes: reconcile ran BEFORE the death was recorded (absent empty, so
        # correctly did nothing, and the race is the problem), or it ran after and
        # reconcile_communicator wrongly returned False. The absent set is the whole
        # difference and it was not being printed.
        print(
            f"  _sync_weights: refit membership absent={sorted(absent)} "
            f"rebuilt={rebuilt}{' (forced)' if force else ''}",
            flush=True,
        )
        return rebuilt

    @contextlib.contextmanager
    def _recovery_window(self) -> Iterator[None]:
        """Mark the span where the serving set is deliberately empty.

        _recover_from_failed_refit marks every serving shard partial, so they all go
        STALE and serving_shards() is empty until _promote_refit_shards runs -- after a
        rebuild and a full retry refit, both of which await and yield the event loop.

        _stall_watchdog_pump is a task on that same loop and calls raise_if_exhausted()
        on every tick, defaulting to one every 30s. With min_healthy_shards=1 and zero
        serving, any tick landing in this window ends the run over an exhausted fleet
        while the retry that would have refilled it is still in flight -- killing the
        recovery that was about to succeed, and blaming the wrong thing in the log.

        A flag rather than deferring mark_weights_partial until after the rebuild: that
        would also close the window, but it would leave shards holding a mix of old and
        new weights in the serving set while it did.
        """
        self._recovering_from_refit = True
        try:
            yield
        finally:
            # finally, not a trailing assignment: the retry refit inside this window is
            # expected to raise sometimes, and a leaked flag would disable the
            # exhaustion check for the rest of the run.
            self._recovering_from_refit = False

    async def _recover_from_failed_refit(self, failure: BaseException) -> None:
        """Drop whatever stopped participating, rebuild the communicator, allow a retry.

        Two failures arrive here and they are not the same event:

        ``RefitAborted`` -- a rank went silent *inside* the collective and a worker's
        watchdog broke it. Every engine that was receiving is left holding a mix of old
        and new weights, so none of them may serve until a refit completes.

        ``RayActorError`` -- the collective finished and a shard died in the epilogue,
        before its RPC returned. Nothing is partial; the survivors have complete weights.
        Left uncaught this killed a run whose data transfer had *already succeeded*.

        Both need the same repair, because both leave a communicator that no longer
        matches the fleet, and in the abort case no communicator at all.

        The probe here is the point. Waiting for the health monitor to reach its own
        conclusion is what failed before: its verdict is paced by probe rounds while this
        is an event, so the abort arrived first and the rebuild saw an empty absent set
        and did nothing. Asking now, on this thread, turns a race into a lookup.
        """
        print(f"  _sync_weights: {failure}; rebuilding and retrying once", flush=True)
        if self._gen_fleet is None:
            # Without fleet health there is no notion of a shard being gone and nothing
            # to rebuild against. Failing here is the pre-existing behaviour.
            raise failure

        # 1. Establish who is actually gone, now, rather than on the probe's clock.
        await self._probe_generation_fleet()

        # 2. Only an abort leaves partial weights behind. Marking survivors stale after
        #    a completed broadcast would pull a healthy fleet out of service over a
        #    transfer that succeeded.
        if isinstance(failure, RefitAborted):
            for shard_idx in self._gen_fleet.serving_shards():
                self._gen_fleet.mark_weights_partial(shard_idx)

        # 3. Rebuild without the dead.
        rebuilt = await self._reconcile_refit_membership(force=True)
        if rebuilt is None:
            # This transport owns no NCCL world -- IPC, HTTP, checkpoint-engine. There is
            # nothing to rebuild, and saying "no shard was absent" here would be a wrong
            # diagnosis of a right refusal.
            raise RuntimeError(
                "refit failed on a transport that owns no communicator membership, so "
                "there is nothing to rebuild over and the run cannot continue. Only the "
                "NCCL transports (collective, nccl_reshard) can recover this way."
            ) from failure
        if not rebuilt:
            # Nothing is absent, but that is not the same as nothing being known.
            #
            # The engine this is really about is wedged rather than dead: is_alive() is
            # answered by the Ray actor and never touches the engine, so the probe can only
            # ever see a dead PROCESS. A wedged one stays serving, is never absent, and
            # takes the run down here -- while its own generations have been timing out and
            # driving it to SUSPECT the whole time. The abort is independent evidence that
            # something in the collective stopped participating; SUSPECT says which.
            #
            # Exactly one, or not at all. With several suspects there is no way to tell
            # which broke the collective, and condemning the wrong one costs a healthy shard
            # and leaves the real culprit in the rebuilt communicator -- the same hang, one
            # shard smaller. With none there is nothing to go on. Both keep the raise.
            suspects = self._gen_fleet.suspected_shards()
            if len(suspects) == 1:
                culprit = suspects[0]
                print(
                    f"  _sync_weights: no shard is absent, but shard {culprit} was already "
                    "suspect; condemning it as the silent participant and retrying",
                    flush=True,
                )
                self._gen_fleet.condemn_silent_participant(
                    culprit,
                    reason=(
                        "did not participate in a refit that had to be aborted, while "
                        "already suspect from failing generations"
                    ),
                )
                rebuilt = await self._reconcile_refit_membership(force=True)

            if not rebuilt:
                # Either nothing was suspect, or too many were. Retrying would die on the
                # aborted communicator or -- worse -- rebuild over the full fleet and hang
                # on the same silent rank, which is the wedge this path exists to remove.
                raise RuntimeError(
                    "refit failed and no generation shard could be identified as absent, "
                    "so the communicator cannot be safely rebuilt; the run cannot "
                    "continue. A rank that is alive but not participating would produce "
                    f"this. Shards already suspect: {suspects or 'none'} (exactly one is "
                    "needed to attribute the failure)."
                ) from failure

    def _promote_refit_shards(self) -> None:
        """Return shards holding current weights to the serving set.

        The exit from STALE, and the reason marking partial weights is safe rather than
        terminal. An aborted refit leaves every engine that was receiving with a mix of
        old and new weights, so they are pulled out of service -- but nothing else moves
        a shard out of STALE, so without this the recovery would succeed and then leave
        the fleet empty, which ``raise_if_exhausted`` would end the run over. A worse
        failure than the one being recovered from, and reached only on the recovery path.

        Only STALE shards are promoted. A SUSPECT shard also took part in the refit, but
        it is failing probes for its own reasons and promoting it here would reset the
        failure count that is supposed to condemn it.
        """
        if self._gen_fleet is None:
            return
        for health in self._gen_fleet.snapshot():
            if health.state is ShardState.STALE:
                self._gen_fleet.report_refit(
                    health.dp_shard_idx, weight_version=self._trainer_version
                )

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

        def _stale_groups() -> list[tuple[str, asyncio.Task[None]]]:
            stale_groups: list[tuple[str, asyncio.Task[None]]] = []
            for group_id, inflight in self._inflight_by_group_id.items():
                task, start_version = inflight
                if self._sampler.should_abort_inflight(
                    start_weight_version=start_version,
                    current_train_weight=self._trainer_version,
                ):
                    stale_groups.append((group_id, task))
            return stale_groups

        if self._rollout_recovery_enabled:
            async with self._data_plane_checkpoint_barrier.mutation() as cut:
                # Re-evaluate after acquiring the cut: a rollout may have completed
                # while a checkpoint holder delayed this mutation.
                stale_groups = _stale_groups()
                for group_id, _ in stale_groups:
                    # This is an intentional live abort, not a process failure. Remove
                    # durable ownership before cancellation cleanup removes the unready
                    # TQ slot, so a concurrent checkpoint cannot resurrect the prompt.
                    self._rollout_manager.discard_prompt_group(cut, group_id)
                for _, task in stale_groups:
                    task.cancel()
        else:
            stale_groups = _stale_groups()
            for _, task in stale_groups:
                task.cancel()

        if not stale_groups:
            return 0

        stale_tasks = [task for _, task in stale_groups]
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

    async def _save_checkpoint(
        self,
        step_metrics: dict[str, Any],
        *,
        is_policy_training_step: bool,
    ) -> None:
        """Write a full checkpoint for the just-finished train step.

        Everything except the (possibly async) policy weight write must be
        on disk before begin_finalization; rollouts keep running throughout.
        The policy optimizer is skipped during critic warmup -- it has never
        stepped.
        """
        save_state = self._save_state
        # SC has no validation loop yet; drop the default sentinel instead of
        # persisting a bogus val_reward.
        if hasattr(save_state, "val_reward"):
            delattr(save_state, "val_reward")

        # validate_single_controller_config already rejected anything but a
        # "train:" prefix, so step_metrics is the only source to consult.
        full_metric_name = self._master_config.checkpointing["metric_name"]
        if full_metric_name is not None:
            metric_name = full_metric_name.split(":", 1)[1]
            if not is_policy_training_step:
                warnings.warn(
                    f"checkpointing.metric_name={full_metric_name!r} is not "
                    "available during PPO critic warmup; this checkpoint will "
                    "not be saved as top-k.",
                    stacklevel=2,
                )
                if hasattr(save_state, full_metric_name):
                    delattr(save_state, full_metric_name)
            elif metric_name not in step_metrics:
                raise ValueError(f"Metric {metric_name} not found in train metrics")
            else:
                setattr(save_state, full_metric_name, step_metrics[metric_name])

        # Flush the previous checkpoint's background finalization first;
        # re-raises a failure from it.
        await asyncio.to_thread(self._checkpointer.finalize_pending)

        print(f"Saving checkpoint for step {self._train_steps}...")
        replay_metadata: Optional[TQReplayMetadataState] = None
        rollout_recovery_state: Optional[RolloutRecoveryState] = None
        rollout_recovery_payload: Optional[bytes] = None
        rollout_recovery_payload_sha256: Optional[str] = None

        # Admission, dataloader movement, replay mutations, and canonical TQ writes
        # all take the mutation side of this barrier. Capture every restart-facing
        # controller artifact under the exclusive side so the checkpoint cannot
        # contain a cursor without its prompt owner, or two durable owners for one
        # canonical group.
        async with self._data_plane_checkpoint_barrier.checkpoint():
            save_state.current_step = self._train_steps
            save_state.total_steps = self._train_steps
            save_state.trainer_version = self._trainer_version
            save_state.current_epoch = self._current_epoch
            save_state.consumed_samples = self._consumed_samples
            save_state.total_valid_tokens = self._total_valid_tokens
            save_state.sampler_name = self._async_cfg.sampler.name
            save_state.sampler_dispatch_index = self._sampler.dispatch_index
            dataloader_state = self._dataloader.state_dict()
            # The spare pool and dataloader advance together under the same mutation
            # cut in recovery-enabled dispatch, so preserve them in this cut too.
            reserve_state = list(self._replacement_reserve)

            checkpoint_path: PathLike = await asyncio.to_thread(  # pyrefly: ignore[bad-assignment]  the PathLike alias resolves inconsistently under pyrefly's import-cycle breaking
                self._checkpointer.init_tmp_checkpoint,
                self._train_steps,
                vars(save_state),
                self._master_config,
            )

            if self._master_config.checkpointing.get("save_data_plane"):
                if self._sampler.supports_buffer_checkpoint:
                    replay_metadata = self._buffer.metadata_state_dict(
                        saved_capacity=self._async_cfg.max_buffered_rollouts
                    )
                if replay_metadata is not None:
                    await self._validate_replay_inventory(replay_metadata)

                if self._rollout_recovery_enabled:
                    rollout_recovery_state = build_rollout_recovery_state(
                        self._rollout_manager.recovery_ledger,
                        batch_shortfall=self._batch_shortfall,
                        sampler_stamps_target_steps=(self._sampler_stamps_target_steps),
                    )
                    if replay_metadata is not None:
                        canonical_group_ids = {
                            group["group_id"] for group in replay_metadata["groups"]
                        }
                        rollout_recovery_state["groups"] = [
                            group
                            for group in rollout_recovery_state["groups"]
                            if group["group_id"] not in canonical_group_ids
                        ]
                    payload_buffer = io.BytesIO()
                    await asyncio.to_thread(
                        torch.save,
                        rollout_recovery_state,
                        payload_buffer,
                    )
                    rollout_recovery_payload = payload_buffer.getvalue()
                    rollout_recovery_payload_sha256 = hashlib.sha256(
                        rollout_recovery_payload
                    ).hexdigest()

                await self._save_data_plane_checkpoint(
                    checkpoint_path,
                    replay_metadata=replay_metadata,
                    rollout_recovery_payload_sha256=(rollout_recovery_payload_sha256),
                    rollout_recovery_group_count=(
                        len(rollout_recovery_state["groups"])
                        if rollout_recovery_state is not None
                        else None
                    ),
                )

        # Save value model
        if self._is_ppo:
            # The critic shares the training GPUs, so the two weight saves are
            # serialized. The critic goes first because offloading the policy runs
            # finalize_async_save, which would block on the policy's own write if
            # that had already been staged.
            await asyncio.to_thread(self._trainer.offload_to_cpu)
            # The value model writes synchronously, so unlike the policy below it is
            # fully on disk when this returns and needs no finalize wait.
            await asyncio.to_thread(self._value.prepare_for_training)
            await asyncio.to_thread(
                self._value.save_checkpoint,
                weights_path=os.path.join(checkpoint_path, "value", "weights"),
                optimizer_path=os.path.join(checkpoint_path, "value", "optimizer")
                if self._checkpointer.save_optimizer
                else None,
                tokenizer_path=os.path.join(checkpoint_path, "value", "tokenizer"),
                checkpointing_cfg=self._master_config.checkpointing,
            )
            await asyncio.to_thread(self._value.finish_training)
            # Also covers a warmup step, which never ran prepare_for_training in
            # the pump and would otherwise save from CPU-resident params.
            await asyncio.to_thread(self._trainer.prepare_for_training)

        # Save policy model
        # With async_save this returns after D2H staging; disk writes finish
        # in the background.
        await asyncio.to_thread(
            self._trainer.save_checkpoint,
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            # Always save policy weights so every PPO checkpoint has
            # the same component layout. Before the first real policy
            # update, omit optimizer and scheduler state because their
            # lazily initialized state is not yet safe to checkpoint.
            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer")
            if self._checkpointer.save_optimizer and is_policy_training_step
            else None,
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            checkpointing_cfg=self._master_config.checkpointing,
        )

        await asyncio.to_thread(
            torch.save,
            dataloader_state,
            os.path.join(checkpoint_path, "train_dataloader.pt"),
        )
        if reserve_state:
            await asyncio.to_thread(
                torch.save,
                reserve_state,
                os.path.join(checkpoint_path, REPLACEMENT_RESERVE_FILENAME),
            )
        if replay_metadata is not None:
            await asyncio.to_thread(
                torch.save,
                replay_metadata,
                os.path.join(checkpoint_path, REPLAY_BUFFER_METADATA_FILENAME),
            )
        if rollout_recovery_payload is not None:
            await asyncio.to_thread(
                Path(
                    checkpoint_path,
                    ROLLOUT_RECOVERY_STATE_FILENAME,
                ).write_bytes,
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

    # Grace on top of the workers' own refit deadline, so their attributable error wins
    # the race against this bound. They need to hit the deadline, abort, unwind, raise,
    # and have Ray deliver it -- seconds, not minutes. If this fires first we lose their
    # diagnosis and report only "the refit never came back", which is true but less useful.
    _REFIT_UNWIND_GRACE_S = 60.0

    def _refit_await_budget_s(self) -> Optional[float]:
        """How long to wait for the refit before giving up, or None to wait forever."""
        deadline = self._async_cfg.generation_fleet_health.refit_timeout_s
        return None if deadline is None else deadline + self._REFIT_UNWIND_GRACE_S

    async def _sync_weights_within(self, kv_scales, what: str) -> None:
        """Run the refit off-loop, and stop waiting if it outlives the deadline.

        WHY THIS IS NEEDED ON TOP OF EVERY WORKER-SIDE BOUND. A frozen-but-alive rank is a
        Ray actor that never answers, and Ray puts no timeout on an actor call. So the
        controller's await never resolves however well the workers behave. Job 6508251
        measured that end state: the deadline fired, the workers aborted, the trainers
        returned, every actor was idle -- and the run still sat for 1800s because this
        await had no bound. It is the last unbounded wait on the refit path, and unlike the
        others it is not in NCCL or CUDA.

        Timing out raises RefitAborted so this joins the existing recovery path rather than
        inventing a second one: the caller reconciles membership and retries once. By then
        the fleet probe has usually condemned the silent shard, so the rebuild can exclude
        it and the retry may genuinely succeed.

        A DEDICATED DAEMON THREAD, not asyncio.to_thread. to_thread runs on the default
        ThreadPoolExecutor, whose workers are non-daemon and are joined at interpreter
        exit -- so a thread still blocked on the frozen actor would hang shutdown, trading
        a wedge in the refit for a wedge on the way out. asyncio.wait_for cannot cancel a
        running thread either way; this only controls whether the orphan can block exit.

        The orphan is a real consequence, not a free win. If the retry succeeds the run
        continues with a thread still parked inside the old sync_weights, and if that rank
        were ever resumed it would wake up holding a communicator that has since been
        replaced. Acceptable against a guaranteed 30-minute stall, and it is why this path
        is bounded-failure-first rather than resume-and-forget.
        """
        budget_s = self._refit_await_budget_s()
        if budget_s is None:
            await asyncio.to_thread(
                self._weight_synchronizer.sync_weights, kv_scales=kv_scales
            )
            return

        loop = asyncio.get_running_loop()
        settled: asyncio.Future = loop.create_future()

        def _settle(setter, value) -> None:
            # wait_for cancels `settled` on timeout, and setting a result on a cancelled
            # future raises InvalidStateError inside the loop callback.
            if not settled.done():
                setter(value)

        def _run() -> None:
            try:
                self._weight_synchronizer.sync_weights(kv_scales=kv_scales)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the loop below
                loop.call_soon_threadsafe(_settle, settled.set_exception, exc)
            else:
                loop.call_soon_threadsafe(_settle, settled.set_result, None)

        threading.Thread(target=_run, name=f"sc-refit-{what}", daemon=True).start()

        try:
            await asyncio.wait_for(settled, budget_s)
        except asyncio.TimeoutError:
            raise RefitAborted(
                f"the {what} refit did not return within {budget_s}s. Every worker-side "
                "bound has passed, so a generation rank is most likely alive but not "
                "answering -- a Ray actor call has no timeout of its own, so waiting here "
                "is unbounded. Giving up so the fleet can be reconciled and retried."
            ) from None

    async def _sync_weights(
        self,
        *,
        calibration_data: Optional[BatchedDataDict[Any]] = None,
    ) -> int:
        """Pause new rollout dispatches, synchronize weights, resume.

        SC owns the pause gate; in-flight generations continue through the
        refit — vLLM V1 async engine supports weight updates during pending
        requests.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. Optionally calibrate FP8 KV-cache scales.
          3. weight_synchronizer.sync_weights(kv_scales=...)
          4. _rollout_permitted.set()   — resume

        Args:
            calibration_data: Optional data used to calibrate FP8 KV-cache
                scales before synchronizing weights.

        Returns:
            The number of stale in-flight rollout groups aborted before the
            weight synchronization.
        """
        self._rollout_permitted.clear()

        # TODO(#2625): Abort unconditionally once Gym-path abort is validated;
        # for now only the native path aborts stale in-flight requests.
        aborted_stale_inflight_groups = (
            0
            if should_use_nemo_gym(self._master_config)
            else await self._abort_stale_inflight()
        )

        # TODO(#2625): Add drain-gate support during refit.

        # Reconcile before the refit, not on a death event. The refit group is provably
        # idle here and every rank is synchronized, which is required because the
        # operations that change membership are themselves collectives. Doing it every
        # time is also idempotent, so a missed or reordered health update converges on
        # the next step instead of needing replay. Cheap rather than free: the
        # synchronizer skips the rebuild when the absent set matches what it last built
        # with, so the steady state after a loss is a set comparison, not a rebuild.
        await self._reconcile_refit_membership()

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

        # Reconcile once more, immediately before the collective.
        #
        # The reconcile above runs before calibration and two to_thread hops; a death
        # recorded in that gap would otherwise be ignored until the NEXT step, by which
        # time this broadcast is already hanging on the missing rank. Idempotent, and a
        # set comparison in the common case -- it used to be a full rebuild on every call
        # once a shard was gone, because absent_shards() never empties again.
        await self._reconcile_refit_membership()

        try:
            await self._sync_weights_within(kv_scales, "first")
        except (RefitAborted, RayActorError) as failure:
            # DETECT AND FAIL FAST, because this one cannot be recovered from.
            #
            # sync_stream_within gives up on kernels already enqueued on THIS trainer's
            # stream, and aborting a communicator does not retire them. Its CUDA context is
            # unusable afterwards: ncclCommAbort never returns and no rebuild on that device
            # can bootstrap, so entering the recovery here does not fail -- it wedges, for
            # the full harness deadline, with no attribution. Jobs 6521181, 6523731, 6582457
            # and 6584636 each eliminated one candidate explanation and left this one.
            #
            # Narrow by construction: the token is applied only by sync_stream_within, which
            # is reachable only from _nccl_reshard_refit. The packed-broadcast transport
            # takes the same fault, fires the same deadline, aborts and recovers, and so
            # does reshard when the fault lands at a step boundary. Recovering this last
            # case is deliberately left to a future change; see the design doc, 8.5.7.
            if is_refit_context_lost(failure):
                print(
                    "  _sync_weights: the refit aborted mid-transfer on the nccl_reshard "
                    "bulk path, which orphans GPU work on the trainers and leaves their "
                    "CUDA contexts unusable. Recovery is not possible from here, so the "
                    "run ends now rather than wedging in a rebuild that cannot complete.",
                    flush=True,
                )
                raise
            with self._recovery_window():
                await self._recover_from_failed_refit(failure)
                # Once only: a second failure is a real fault, not a membership problem,
                # and retrying forever would recreate the wedge this exists to remove.
                await self._sync_weights_within(kv_scales, "retry")
                # Inside the window: this is what refills the serving set, so releasing
                # the flag before it runs would reopen the gap it exists to close.
                self._promote_refit_shards()
        else:
            # A completed refit is what makes an engine's weights current, so this is
            # where a shard pulled out of service for holding partial ones earns its way
            # back. else, not a trailing statement: the recovery path above already
            # promoted inside its window, and everything below this must still run on
            # both paths.
            self._promote_refit_shards()
        if self._async_cfg.recompute_kv_cache_after_weight_updates:
            # to_thread, like every other call into the workers here. Run directly on
            # the loop this is a blocking Ray call, and a wedged generation worker would
            # freeze the event loop itself -- taking the watchdog, which is an asyncio
            # task on that same loop, down with it.
            await asyncio.to_thread(self._gen.invalidate_kv_cache)
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_permitted.set()
        return aborted_stale_inflight_groups

    async def _value_stage(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Run the PPO value model's forward pass over the selected chunk.

        Tensors never touch SC: workers fetch the sequence columns from
        DataPlane and commit the per-token prediction back under ``values``,
        which the advantage stage then reads alongside the rewards. The value model
        is loaded and offloaded around the call, so it holds the training GPUs
        only for the duration of the forward.

        Returns:
            The batch metadata with the ``values`` column recorded on it.
        """
        await asyncio.to_thread(self._value.prepare_for_inference)
        await asyncio.to_thread(self._value.get_values_from_meta, meta)
        await asyncio.to_thread(self._value.finish_inference)
        return meta.with_fields([self._advantage_cfg.values_field])

    async def _value_train_epochs(
        self, meta: KVBatchMeta, *, num_epochs: int
    ) -> dict[str, Any]:
        """Run consecutive critic epochs under one model onload/offload cycle.

        Returns:
            The final epoch's ``train_from_meta`` output; earlier epochs'
            results are discarded.
        """
        await asyncio.to_thread(self._value.prepare_for_training)
        result: dict[str, Any] | None = None
        for _ in range(num_epochs):
            result = await asyncio.to_thread(
                self._value.train_from_meta,
                meta,
                self._value_loss_fn,  # pyrefly: ignore
            )
        await asyncio.to_thread(self._value.finish_training)
        assert result is not None
        return result

    async def _advantage_stage(self, meta: KVBatchMeta) -> tuple[KVBatchMeta, bool]:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.

        Returns:
            The updated batch metadata and whether the batch contains at least
            one valid training token.
        """
        for tag in meta.tags or []:
            for key in VIOLATION_TAG_KEYS:
                self._step_log_dict.setdefault(key, []).append(int(tag.get(key, 0)))

        if self._advantage_estimator is None:
            return meta, True
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

        seq_logprob_error_threshold = self._algo_cfg.seq_logprob_error_threshold
        # Match the legacy path: whenever real policy logprobs are available,
        # report sequence-level generation/training mismatch. A threshold adds
        # masking; leaving it unset keeps this metrics-only.
        if self._policy_logprobs_required:
            masking_data = BatchedDataDict(
                {
                    "token_mask": token_mask,
                    "sample_mask": sample_mask,
                    "prev_logprobs": tensor_field(
                        data,
                        adv_cfg.policy_logprobs_field,
                    ),
                    "generation_logprobs": tensor_field(
                        data,
                        adv_cfg.generation_logprobs_field,
                    ),
                }
            )
            num_valid_seqs_before = float(
                ((token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(dim=-1) > 0)
                .sum()
                .item()
            )
            seq_error_metrics = compute_and_apply_seq_logprob_error_masking(
                train_data=masking_data,
                rewards=rewards,
                seq_logprob_error_threshold=seq_logprob_error_threshold,
            )
            sample_mask = masking_data["sample_mask"]
            num_valid_seqs_after = float(
                ((token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(dim=-1) > 0)
                .sum()
                .item()
            )
            seq_error_metrics["num_masked_seqs_by_logprob_error"] = (
                seq_error_metrics.pop("num_masked_seqs")
            )
            seq_error_metrics["_num_valid_seqs_before"] = num_valid_seqs_before
            seq_error_metrics["_num_valid_seqs_after"] = num_valid_seqs_after
            self._step_log_dict["seq_logprob_error_metrics"].append(seq_error_metrics)

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
            policy_logprobs = tensor_field(data, adv_cfg.policy_logprobs_field)
            if self._teacher_logprobs_required:
                kwargs["prev_logprobs"] = policy_logprobs
            else:
                kwargs["logprobs_policy"] = policy_logprobs
        if self._reference_logprobs_required:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )
        if self._teacher_logprobs_required:
            kwargs["teacher_logprobs"] = tensor_field(
                data,
                adv_cfg.teacher_logprobs_field,
            )
        if self._is_ppo:
            kwargs["values"] = tensor_field(data, adv_cfg.values_field)

        # Training predicts token t from position t - 1, so token_mask[:, 1:]
        # is the exact mask used when global_valid_toks and the loss are built.
        has_valid_training_tokens = bool(mask[:, 1:].bool().any().item())
        # Value-model estimators (GAE) hand back the regression target alongside
        # the advantages; the group-relative ones return a bare tensor.
        returns: Optional[torch.Tensor] = None
        if has_valid_training_tokens:
            result = self._advantage_estimator.compute_advantage(
                prompt_ids=prompt_ids,
                rewards=rewards,
                mask=mask,
                repeated_batch=repeated_batch,
                **kwargs,
            )
            if self._is_ppo:
                advantages, returns = result
            else:
                advantages = result
        else:
            advantages = torch.zeros_like(mask)
            if self._is_ppo:
                returns = torch.zeros_like(mask)

        if self._message_level_advantage_penalties_enabled:
            # Sequence-error filtering and the pre-existing sample mask remain
            # authoritative: a message penalty must not make a filtered token
            # trainable again.
            valid_tokens = mask.bool()
            advantages = apply_message_level_advantage_penalties(
                advantages,
                invalid_tool_call_mask=(
                    tensor_field(data, adv_cfg.invalid_tool_call_mask_field).bool()
                    & valid_tokens
                ),
                malformed_thinking_mask=(
                    tensor_field(data, adv_cfg.malformed_thinking_mask_field).bool()
                    & valid_tokens
                ),
                invalid_tool_call_advantage=self._algo_cfg.invalid_tool_call_advantage,
                malformed_thinking_advantage=(
                    self._algo_cfg.malformed_thinking_advantage
                ),
            )

        response_advantages = torch.masked_select(advantages, mask.bool())
        self._step_log_dict["rewards"].append(rewards.detach().cpu())
        self._step_log_dict["masked_advantages"].append(
            response_advantages.detach().cpu()
        )
        if self._teacher_logprobs_required:
            valid = response_advantages.detach().double()
            self._opd_stat_sum += float(valid.sum())
            self._opd_stat_sumsq += float((valid * valid).sum())
            self._opd_stat_count += int(valid.numel())

        fields_to_put = {adv_cfg.output_field: advantages}
        if seq_logprob_error_threshold is not None:
            fields_to_put[adv_cfg.sample_mask_field] = sample_mask
        new_fields = [adv_cfg.output_field]
        if returns is not None:
            fields_to_put[adv_cfg.returns_field] = returns
            new_fields.append(adv_cfg.returns_field)

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(meta, fields_to_put),
        )
        return (
            meta.with_fields(new_fields),
            has_valid_training_tokens,
        )

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
        if self._message_level_advantage_penalties_enabled:
            fields.extend(
                [
                    adv_cfg.invalid_tool_call_mask_field,
                    adv_cfg.malformed_thinking_mask_field,
                ]
            )
        if self._policy_logprobs_required:
            fields.append(adv_cfg.policy_logprobs_field)
        if self._policy_logprobs_required:
            fields.append(adv_cfg.generation_logprobs_field)
        if self._reference_logprobs_required:
            fields.append(adv_cfg.reference_logprobs_field)
        if self._teacher_logprobs_required:
            fields.append(adv_cfg.teacher_logprobs_field)
        if self._is_ppo:
            fields.append(adv_cfg.values_field)
        return list(dict.fromkeys(fields))

    def _retune_lookahead_versions(self) -> None:
        """Widen the sampler's lookahead while the policy is frozen, then shrink it back.

        Port of ppo.py's _async_ppo_generation_lead_steps.
        """
        if not self._is_ppo:
            return
        steady = self._async_cfg.sampler.max_lookahead_versions
        warmup = self._async_cfg.sampler.warmup_lookahead_versions
        start = self._algo_cfg.policy_training_start_step
        if warmup is None or self._trainer_version >= start:
            window = steady
        else:
            remaining_to_frontier = start + steady - self._trainer_version
            window = max(steady, min(warmup, remaining_to_frontier))
        self._sampler.set_gate_window(window)
