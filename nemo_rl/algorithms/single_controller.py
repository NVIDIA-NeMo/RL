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
import logging
import math
import os
import time
from collections import deque
from functools import partial
from typing import Any, Awaitable, Callable, Optional, Union

import ray
import torch

from nemo_rl.algorithms.async_utils.staleness_sampler import create_sampler
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _write_latest_checkpoint_status,
    compute_and_apply_seq_logprob_error_masking,
)
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
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.nemo_gym import should_use_nemo_gym
from nemo_rl.experience.failures import RolloutStall
from nemo_rl.experience.rollout_manager import RolloutOutcome
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager, PathLike
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import TimeoutChecker, Timer

Generation = Union[VllmGeneration, SGLangGeneration]

# Named `log` rather than `logger` to keep it distinct from the experiment
# Logger this module also uses as `self._logger`.
log = logging.getLogger(__name__)


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
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer

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

        self._trainer_version: int = actor_args.save_state.current_step
        self._train_steps: int = actor_args.save_state.current_step
        self._current_epoch: int = actor_args.save_state.current_epoch
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
            "seq_logprob_error_metrics": [],
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
            self._last_checkpoint_path, "replacement_reserve.pt"
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
                        outcome = await self._rollout_manager.generate_and_push(
                            prompt,
                            target_step=target_step,
                            inflight_registry=self._inflight_by_group_id,
                        )
                    except BaseException:
                        # On success ownership transfers to the train pump, which
                        # releases this permit after consuming the committed group.
                        self._buffer_capacity.release()
                        raise

                    if outcome is not RolloutOutcome.SKIPPED:
                        break

                    replacement = self._take_replacement(target_step, replacements)
                    if replacement is None:
                        # Nothing was committed, so the train pump will never see this
                        # group and never release its permit on our behalf.
                        self._buffer_capacity.release()
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

        async def _launch(prompt: DatumSpec, target_step: Optional[int]) -> None:
            # check if buffer is full
            await self._buffer_capacity.acquire()
            # check if inflight rollouts is full
            await sem.acquire()
            # wait for rollout to be permitted
            await self._rollout_permitted.wait()

            task_started_event = asyncio.Event()
            # dispatch rollout
            task = rollout_tasks.create_task(
                _dispatch_one_prompt(prompt, target_step, task_started_event)
            )
            self._dispatched_rollouts.add(task)
            task.add_done_callback(self._dispatched_rollouts.discard)
            task.add_done_callback(
                partial(
                    _release_permits_if_task_not_started,
                    task_started_event=task_started_event,
                )
            )

        max_epochs = self._master_config.grpo.max_num_epochs
        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
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
                        await _launch(prompt, target_step)

                self._current_epoch += 1

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
        self, launch: Callable[[DatumSpec, Optional[int]], Awaitable[None]]
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
        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        while len(self._replacement_reserve) >= num_prompts_per_step:
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
                await launch(prompt, target_step)

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
        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
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
            chunks_dispatched = 0
            calibration_batches: list[BatchedDataDict[Any]] = []

            with self._timer.time("total_step_time"):
                # Re-read on every iteration rather than once: a prompt stamped for this
                # step can be dropped while the pump is already waiting for it, which is
                # precisely the case that would otherwise wait forever.
                while groups_dispatched < self._target_groups_for_step(
                    version_during_step
                ):
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

                    # Compute advantages
                    with self._timer.time("advantage_calculation"):
                        (
                            train_meta,
                            has_valid_training_tokens,
                        ) = await self._advantage_stage(train_meta)

                    # Filtering can leave a streaming chunk with no training tokens.
                    # Consume that chunk without F/B, then continue the same optimizer
                    # step with the next chunk. Always restore training mode because
                    # log-prob inference may have switched the model to inference mode.
                    with self._timer.time("training_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_training)
                    if has_valid_training_tokens:
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
                    await self._call_dp(
                        "clear_samples",
                        sample_ids=list(train_meta.sample_ids),
                        partition_id=self._partition_id,
                    )

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
                        grpo_cfg.num_prompts_per_step,
                    )

                log.info(
                    "train_pump: step %d closing on %d chunk(s), %d group(s)",
                    version_during_step,
                    chunks_dispatched,
                    groups_dispatched,
                )
                if not step_open:
                    raise RuntimeError(
                        "SingleController has no valid response tokens after "
                        "filtering. Check grpo.seq_logprob_error_threshold to "
                        "avoid an optimizer step with an empty batch."
                    )

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(self._trainer.finish_train_step)

                step_metrics = aggregate_step_metrics(result)
                step_metrics.update(
                    reduce_advantage_pump_metrics(**self._step_log_dict)
                )
                self._step_log_dict = {k: [] for k in self._step_log_dict}

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
            #   histogram log, rollout_metrics, pretty-print "Training Results"
            #   block, print_performance_metrics.
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

            if self._gen_fleet is not None:
                # Raises once too few shards remain for the run to be worth continuing.
                # Checked after publishing so the final state is on record.
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
        """Fold the router's observed backend failures into the fleet ledger.

        The router is the only component that sees a *wedged* engine: it answers
        ``is_alive`` from a healthy worker process, so no probe can condemn it. The
        router holds no monitor reference by design -- membership flows one way -- so it
        counts failures per backend URL and this drains them here, on the tick that
        already talks to it.
        """
        if self._generation_router is None or self._gen_fleet is None:
            return
        counts: dict[str, int] = await self._ray_get(
            self._generation_router.drain_backend_failures.remote()
        )
        for url, count in counts.items():
            shard_idx = self._gen_fleet.shard_for_base_url(url)
            if shard_idx is None:
                continue
            for _ in range(count):
                self._gen_fleet.report_failure(
                    shard_idx,
                    RuntimeError(f"router: {count} failed request(s) to {url}"),
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
        # The spare pool has to be saved with that snapshot, not left out of the
        # checkpoint: diverting a batch already advanced the iterator, so the state
        # above records those prompts as consumed while they are still only in memory.
        # Without this a resumed replace-mode run comes back with an empty pool and a
        # dataloader positioned past the diverted batch, silently losing it -- and
        # losing it for good, since _drain_reserve_into_steps only ever recovers spares
        # held by the process that diverted them. Snapshotted here, in the same
        # await-free window, so the pair cannot disagree.
        reserve_state = list(self._replacement_reserve)
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
        if reserve_state:
            await asyncio.to_thread(
                torch.save,
                reserve_state,
                os.path.join(checkpoint_path, "replacement_reserve.pt"),
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

        # TODO(#2625): Abort unconditionally once Gym-path abort is validated;
        # for now only the native path aborts stale in-flight requests.
        aborted_stale_inflight_groups = (
            0
            if should_use_nemo_gym(self._master_config)
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
            # to_thread, like every other call into the workers here. Run directly on
            # the loop this is a blocking Ray call, and a wedged generation worker would
            # freeze the event loop itself -- taking the watchdog, which is an asyncio
            # task on that same loop, down with it.
            await asyncio.to_thread(self._gen.invalidate_kv_cache)
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        self._rollout_permitted.set()
        return aborted_stale_inflight_groups

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
        if self._advantage_estimator is None:
            return meta, True
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

        seq_logprob_error_threshold = (
            self._master_config.grpo.seq_logprob_error_threshold
        )
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
            kwargs["logprobs_policy"] = tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
        if self._reference_logprobs_required:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )

        # Training predicts token t from position t - 1, so token_mask[:, 1:]
        # is the exact mask used when global_valid_toks and the loss are built.
        has_valid_training_tokens = bool(mask[:, 1:].bool().any().item())
        if has_valid_training_tokens:
            advantages = self._advantage_estimator.compute_advantage(
                prompt_ids=prompt_ids,
                rewards=rewards,
                mask=mask,
                repeated_batch=repeated_batch,
                **kwargs,
            )
        else:
            advantages = torch.zeros_like(mask)
        response_advantages = torch.masked_select(advantages, mask.bool())
        self._step_log_dict["rewards"].append(rewards.detach().cpu())
        self._step_log_dict["masked_advantages"].append(
            response_advantages.detach().cpu()
        )

        fields_to_put = {adv_cfg.output_field: advantages}
        if seq_logprob_error_threshold is not None:
            fields_to_put[adv_cfg.sample_mask_field] = sample_mask

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(meta, fields_to_put),
        )
        return (
            meta.with_fields([adv_cfg.output_field]),
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
        if self._policy_logprobs_required:
            fields.append(adv_cfg.policy_logprobs_field)
        if self._policy_logprobs_required:
            fields.append(adv_cfg.generation_logprobs_field)
        if self._reference_logprobs_required:
            fields.append(adv_cfg.reference_logprobs_field)
        return list(dict.fromkeys(fields))
