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

CPU-only Ray actor that runs two concurrent pumps and coordinates the
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
import statistics
import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import ray
import torch

from nemo_rl.algorithms.async_utils.staleness_sampler import create_sampler
from nemo_rl.algorithms.grpo import GRPOSaveState, _write_latest_checkpoint_status
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    MasterConfig,
    validate_sampler_buffer_capacity,
    validate_single_controller_config,
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
from nemo_rl.data_plane.schema import DP_CALIB_INPUT_FIELDS
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.route_plan import decode_route_plan
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager, PathLike
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import TimeoutChecker, Timer

if TYPE_CHECKING:
    from nemo_rl.experience.finalizer_actor import FinalizationRequest

Generation = Union[VllmGeneration, SGLangGeneration]


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns two concurrent asyncio tasks:
      - _rollout_pump: dispatches prompts to GenerationWorkerActor
      - _train_pump:   claims DataPlane meta, trains, clears consumed rows,
                       then runs _sync_weights (drain gate + weight
                       synchronization) inline after each optimizer step

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
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer
        self._finalizer_actors = list(actor_args.finalizer_actors)
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
        self._consumed_samples: int = actor_args.save_state.consumed_samples
        self._total_valid_tokens: int = actor_args.save_state.total_valid_tokens

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = actor_args.train_cluster
        self._inference_cluster = actor_args.inference_cluster

        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        self._sampler = create_sampler(self._buffer, self._async_cfg.sampler)
        self._sampler.set_dispatch_index(actor_args.save_state.current_step)
        required_capacity = self._sampler.required_buffer_capacity(num_prompts_per_step)
        validate_sampler_buffer_capacity(
            self._async_cfg,
            required_capacity=required_capacity,
            sampler_name=type(self._sampler).__name__,
        )

        # ── asyncio state ──────────────────────────────────────────────────
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

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )

        self._trainer_version: int = actor_args.save_state.current_step
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

        await self._maybe_restore_replay_buffer()

        # Start the rollout and train pumps
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        try:
            done, _ = await asyncio.wait(
                {rollout_task, train_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if rollout_task in done:
                # Propagate rollout failures immediately. A normally exhausted
                # rollout pump leaves the train pump to drain committed groups.
                await rollout_task
            await train_task
        finally:
            rollout_task.cancel()
            train_task.cancel()
            await asyncio.gather(rollout_task, train_task, return_exceptions=True)
            for actor in self._finalizer_actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as error:
                    print(f"finalizer actor termination failed: {error}", flush=True)
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

    # ── internal helpers ───────────────────────────────────────────────────

    async def _maybe_restore_replay_buffer(self) -> None:
        """Restore replay-buffer groups from the previous run's checkpoint.

        Skipped with a warning when the checkpoint was written under a
        different sampler: restored groups carry the saving sampler's
        weight/target-step stamps, which another policy may never select.
        """
        if self._last_checkpoint_path is None:
            return
        buffer_path = os.path.join(self._last_checkpoint_path, "replay_buffer.pt")
        if not os.path.exists(buffer_path):
            print(
                f"⚠️ No replay buffer checkpoint found at {buffer_path}. "
                "Starting with an empty replay buffer.",
                flush=True,
            )
            return
        saved_sampler_name = self._save_state.sampler_name
        current_sampler_name = self._async_cfg.sampler.name
        if saved_sampler_name != current_sampler_name:
            print(
                f"⚠️ Replay buffer checkpoint was saved with sampler "
                f"{saved_sampler_name!r} but this run uses "
                f"{current_sampler_name!r}; skipping the buffer restore.",
                flush=True,
            )
            return
        print(f"📦 Restoring replay buffer from checkpoint: {buffer_path}")
        # weights_only=False: groups hold pickled KVBatchMeta/TensorDicts,
        # not plain tensors. The checkpoint is a trusted same-job artifact.
        buffer_state = await asyncio.to_thread(
            torch.load, buffer_path, weights_only=False
        )
        restored = await self._buffer.load_state_dict(
            buffer_state,
            max_groups=self._async_cfg.max_buffered_rollouts,
            expected_partition_id=self._partition_id,
            expected_group_size=self._master_config.grpo.num_generations_per_prompt,
        )
        # Each buffered group holds one _buffer_capacity permit; the load
        # truncation guarantees restored <= capacity, so this never blocks.
        assert restored <= self._async_cfg.max_buffered_rollouts
        for _ in range(restored):
            await self._buffer_capacity.acquire()

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

    async def _cleanup_known_finalization_request(
        self, request: "FinalizationRequest"
    ) -> None:
        """Clear known request ownership after a pre-publication/known outcome."""
        errors: list[BaseException] = []
        try:
            await self._call_dp(
                "clear_samples",
                sample_ids=list(request.rollout_ids),
                partition_id=self._partition_id,
            )
        except Exception as error:
            errors.append(
                RuntimeError(
                    "pre-publication canonical cleanup failed for "
                    f"group={request.group_id!r}, ids={request.rollout_ids!r}"
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

    async def _finalize_with_actor(self, request: "FinalizationRequest") -> None:
        """Submit one metadata request to the bounded fixed actor pool."""
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
        try:
            finalized = await actor.finalize.remote(request)
        except BaseException:
            self._finalizer_unknown_outcomes += 1
            print(
                "FATAL: finalizer actor RPC failed after submission; canonical "
                f"publication outcome is unknown for group {request.group_id}. "
                "Stopping validation without actor replacement or retry.",
                flush=True,
            )
            raise
        else:
            self._available_finalizers.put_nowait(actor)
        finally:
            self._active_finalizers -= 1
        finalize_total_ms = (time.perf_counter() - finalize_start) * 1000.0

        if finalized.dropped:
            try:
                await self._cleanup_known_finalization_request(request)
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
                await self._cleanup_known_finalization_request(request)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "finalizer returned no metadata and known-key cleanup failed "
                    f"for group {request.group_id}"
                ) from cleanup_error
            raise RuntimeError(
                f"finalizer returned no metadata for non-dropped group {request.group_id}"
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
                await self._cleanup_known_finalization_request(request)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    f"finalizer commit and known-key cleanup failed for "
                    f"group {request.group_id}",
                    [commit_error, cleanup_error],
                )
            raise
        finalized.metrics.update(
            {
                "finalize/queue_wait_ms": queue_wait_ms,
                "finalize/total_ms": finalize_total_ms,
                "finalize/queue_depth": float(queue_depth),
                "finalize/active_actor_count": float(active_actor_count),
            }
        )
        self._finalizer_metrics_by_group[request.group_id] = dict(finalized.metrics)

    async def _cleanup_consumed_metas(self, metas: list[KVBatchMeta]) -> None:
        """Clear canonical rows and full-manifest staging keys after train success."""
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

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch:
          0. await sampler.admit(...) to wait until the batch may dispatch and
             obtain its target_step stamp.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Run the rollout, then either commit it directly or submit its
             metadata-only request to the finalizer actor pool.
          5. Decrement _inflight_rollouts
        """
        sem = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            task_started_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            self._inflight_rollouts += 1
            generation_permit_released = False
            inflight_count_released = False
            ownership_transferred = False
            try:
                if self._finalizer_actors:
                    request = await self._rollout_manager.generate_for_finalization(
                        prompt,
                        target_step=target_step,
                        inflight_registry=self._inflight_by_group_id,
                    )
                    self._inflight_rollouts -= 1
                    inflight_count_released = True
                    sem.release()
                    generation_permit_released = True
                    await self._finalize_with_actor(request)
                else:
                    await self._rollout_manager.generate_and_push(
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
                if not inflight_count_released:
                    self._inflight_rollouts -= 1
                if not generation_permit_released:
                    sem.release()

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

        max_epochs = self._master_config.grpo.max_num_epochs
        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
                for prompt_batch in self._dataloader:
                    target_step = await self._sampler.admit(
                        trainer_version_fn=lambda: self._trainer_version
                    )

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
                                prompt, target_step, task_started_event
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

                self._current_epoch += 1

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

                        # Select a batch
                        max_prompt_groups = (
                            grpo_cfg.num_prompts_per_step - groups_dispatched
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
                            if self._rollout_exhausted.is_set():
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
                        selected_group_ids = {
                            sample_id.rsplit("_g", 1)[0]
                            for sample_id in train_meta.sample_ids
                        }
                        for group_id in selected_group_ids:
                            for name, value in self._finalizer_metrics_by_group.pop(
                                group_id, {}
                            ).items():
                                step_finalizer_metrics.setdefault(name, []).append(
                                    float(value)
                                )

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

                await self._cleanup_consumed_metas(consumed_metas)
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
                with self._timer.time("weight_sync"):
                    calibration_data = (
                        BatchedDataDict.from_batches(calibration_batches)
                        if calibration_batches
                        else None
                    )
                    aborted_stale_inflight_groups = await self._sync_weights(
                        calibration_data=calibration_data
                    )
                    step_metrics.update(
                        {
                            "evicted_stale_prompt_groups": evicted_stale_prompt_groups,
                            "aborted_stale_inflight_groups": aborted_stale_inflight_groups,
                        }
                    )

                # Checkpointing (mirrors async_grpo_train's save block).
                self._consumed_samples += grpo_cfg.num_prompts_per_step
                self._total_valid_tokens += step_metrics.get("global_valid_toks", 0)
                self._timeout.mark_iteration()

                is_last_step = self._train_steps >= grpo_cfg.max_num_steps or (
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
                        await self._save_checkpoint(step_metrics)

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
        """Write a full checkpoint for the just-finished train step.

        Everything except the (possibly async) policy weight write must be
        on disk before begin_finalization; rollouts keep running throughout.
        """
        save_state = self._save_state
        save_state.current_step = self._train_steps
        save_state.total_steps = self._train_steps
        save_state.current_epoch = self._current_epoch
        save_state.consumed_samples = self._consumed_samples
        save_state.total_valid_tokens = self._total_valid_tokens
        # The restore skips the replay buffer when the resuming run uses a
        # different sampler (its stamps may never be selectable there).
        save_state.sampler_name = self._async_cfg.sampler.name
        # Snapshot before any await so it can't interleave with
        # _rollout_pump iterating this same dataloader.
        dataloader_state = self._dataloader.state_dict()
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
        checkpoint_path: PathLike = await asyncio.to_thread(  # pyrefly: ignore[bad-assignment]  the PathLike alias resolves inconsistently under pyrefly's import-cycle breaking
            self._checkpointer.init_tmp_checkpoint,
            self._train_steps,
            vars(save_state),
            self._master_config,
        )
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
        await asyncio.to_thread(
            torch.save,
            dataloader_state,
            os.path.join(checkpoint_path, "train_dataloader.pt"),
        )
        buffer_state = await self._buffer.state_dict(
            saved_capacity=self._async_cfg.max_buffered_rollouts
        )
        await asyncio.to_thread(
            torch.save,
            buffer_state,
            os.path.join(checkpoint_path, "replay_buffer.pt"),
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
        return aborted_stale_inflight_groups

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

        data = await self._call_dp(
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

        await self._call_dp(
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
