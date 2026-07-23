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

"""SingleController: asyncio-based orchestrator for the RL training loop.

SingleController is a CPU-only Ray actor that owns two concurrent asyncio
pumps and coordinates the other components: gen and DataPlane via
lightweight actor RPCs, and the trainer as a driver-side ``TQPolicy``
object invoked through ``asyncio.to_thread``.

Key invariant: SC does not run model work. It sends control signals
(``KVBatchMeta`` and actor handles) and reads metadata. When advantage
calculation is enabled, SC fetches only the configured advantage input
columns, computes advantages, and writes that small derived column back to
DataPlane. Model tensors still move through DataPlane or NCCL.

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
  _sync_weights  → drain _inflight_rollouts → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import Any, Literal, Optional

import ray
import torch
from pydantic import BaseModel, Field

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSamplerConfig,
    SamplerConfig,
    create_sampler,
)
from nemo_rl.algorithms.grpo import GRPOLoggerConfig
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer

# TQ partition schema field names — cross-component protocol with the rollout
# actor. These are not user-tunable: changing them in SC would also need to
# change the rollout writer. Treated as fixed conventions, not config.
ADVANTAGE_OUTPUT_FIELD = "advantages"
PROMPT_IDS_FIELD = "prompt_ids_for_adv"
REWARD_FIELD = "total_reward"
TOKEN_MASK_FIELD = "token_mask"
SAMPLE_MASK_FIELD = "sample_mask"


class StubRefitConfig(BaseModel):
    """No-op refit backend for dry runs and tests."""

    impl: Literal["stub"] = "stub"


class NcclRefitConfig(BaseModel):
    """NCCL weight-broadcast refit backend.

    NOTE: addr/port currently have no readers; they're reserved for the
    NCCL transport that lands with weight_sync wiring in a follow-up PR.
    """

    impl: Literal["nccl"] = "nccl"
    addr: str = "127.0.0.1"
    port: Optional[int] = None


class SingleControllerConfig(BaseModel, extra="allow"):
    """Configuration for SingleController.

    ``extra="allow"`` accepts unknown keys so YAML-loaded configs can carry
    forward fields the runtime doesn't (yet) consume. Literal-typed fields
    are validated at construction by pydantic — no runtime assert needed.
    """

    # Group geometry. A "group" is the atomic training unit: group_size
    # samples sharing one source prompt. GRPO consumers set group_size =
    # num generations per prompt; SFT/OPD-style consumers degenerate
    # cleanly to group_size=1 (every sample its own group).
    min_groups_per_batch: int = 2
    target_groups_per_step: Optional[int] = None
    group_size: int = 4

    # Concurrency limits
    max_inflight_prompts: int = 8
    max_buffered_rollouts: int = 8  # _buffer_capacity semaphore size

    # Staleness policy — the single source of truth for how rollouts are gated
    # (rollout pump) and selected/evicted (train pump). Discriminated on
    # ``name`` (windowed | weight_fifo | in_order | custom); each variant carries
    # only its own args, so invalid combinations are unrepresentable and there
    # are no cross-field knobs to validate at runtime. ``custom`` takes a
    # ``module:ClassName`` target to load a PromptGroupSampler from outside this
    # repo. Default mirrors the exemplar (exact batch->step matching).
    sampler: SamplerConfig = Field(default_factory=InOrderSamplerConfig)

    # Training
    max_train_steps: int = 10
    # Bounded dataset passes, mirroring grpo.py's max_num_epochs loop. One
    # epoch = one pass over the dataloader. None = unbounded: the dataloader
    # is iterated indefinitely until _train_pump reaches max_train_steps.
    max_num_epochs: Optional[int] = None
    # Worker-side optimizer mini-batch size, in samples. SC opens one
    # begin / microbatch×K / finish cycle (= one opt.step) per
    # ``train_global_batch_size`` worth of samples. Number of opt.steps per
    # outer SC step is
    # ``target_groups_per_step * group_size // train_global_batch_size``.
    # If None, coerced at __init__ to one mini-batch per outer step
    # (samples_per_step), preserving single-opt.step-per-outer-step behavior.
    train_global_batch_size: Optional[int] = None

    # DataPlane partition
    partition_id: str = "rollout_data"

    # Advantage calculation. The TQ partition column names for prompt_ids /
    # reward / token_mask / sample_mask / advantages are fixed conventions
    # (see module-level *_FIELD constants); only the real toggles live here.
    advantage_enabled: bool = False
    advantage_repeated_batch_fields: list[str] = Field(default_factory=list)
    advantage_policy_logprobs_field: str | None = None
    advantage_reference_logprobs_field: str | None = None

    # Diagnostics
    diagnostics: bool = False

    # Refit (weight transport) backend. Discriminated on ``impl`` so each
    # backend carries only its own knobs and pydantic narrows the type at
    # construction — not every algorithm/deployment uses NCCL.
    refit_cfg: StubRefitConfig | NcclRefitConfig = Field(
        default_factory=StubRefitConfig, discriminator="impl"
    )

    # Logger
    logger: GRPOLoggerConfig



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
        cfg: SingleControllerConfig,
        dp_client_handle: Any,
        gen_handle: Any,
        trainer_handle: Any,
        weight_synchronizer: Any,
        loss_fn: Any,
        rollout_manager: Any,
        advantage_estimator: Any | None = None,
        dataloader: Any = None,
        tq_buffer: Any = None,
    ) -> None:
        self._cfg = cfg
        self._dp_client = dp_client_handle
        self._gen = gen_handle
        self._trainer = trainer_handle
        self._weight_synchronizer = weight_synchronizer
        self._loss_fn = loss_fn
        self._advantage_estimator = advantage_estimator
        self._dataloader = dataloader
        self._buffer = tq_buffer
        self._rollout_manager = rollout_manager
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(cfg.logger)  # type: ignore
        self._timer = Timer()

        if cfg.advantage_enabled and self._advantage_estimator is None:
            raise ValueError(
                "advantage_enabled=True requires an advantage_estimator instance"
            )

        # Staleness-mode validation that used to live here (strict_on_policy
        # coercion, over_sampling/force_in_order mutual-exclusion raises) is gone:
        # the sampler config's discriminated union makes those states
        # unrepresentable — see ``cfg.sampler`` / ``create_sampler``.

        if cfg.target_groups_per_step is None:
            cfg.target_groups_per_step = cfg.min_groups_per_batch
        if cfg.target_groups_per_step < cfg.min_groups_per_batch:
            raise ValueError(
                f"target_groups_per_step ({cfg.target_groups_per_step}) "
                f"must be >= min_groups_per_batch ({cfg.min_groups_per_batch})"
            )

        # Mini-batching contract: SC opens one begin / microbatch×N / finish
        # cycle (= one opt.step) per train_global_batch_size worth of samples.
        # Matches the sync path's gb_idx loop in the worker (one opt.step
        # per gbs slice). When unset, coerce to a single mini-batch per
        # outer step so behavior matches the pre-mini-batch design.
        samples_per_step = cfg.target_groups_per_step * cfg.group_size
        if cfg.train_global_batch_size is None:
            cfg.train_global_batch_size = samples_per_step
        if cfg.train_global_batch_size <= 0:
            raise ValueError(
                f"train_global_batch_size must be > 0, "
                f"got {cfg.train_global_batch_size}"
            )
        if samples_per_step % cfg.train_global_batch_size != 0:
            raise ValueError(
                f"target_groups_per_step ({cfg.target_groups_per_step}) "
                f"* group_size ({cfg.group_size}) "
                f"= {samples_per_step} samples must be divisible by "
                f"train_global_batch_size ({cfg.train_global_batch_size})"
            )
        if cfg.train_global_batch_size % cfg.group_size != 0:
            raise ValueError(
                f"train_global_batch_size ({cfg.train_global_batch_size}) must "
                f"be divisible by group_size "
                f"({cfg.group_size}) so a mini-batch contains "
                f"whole prompt groups"
            )
        # Staleness policy owns admit (rollout side) + select/evict (train side)
        # + is_on_policy / required_buffer_capacity (derived facts). ``name`` in
        # the config is the single switch for which behavior runs.
        self._sampler = create_sampler(self._buffer, cfg.sampler)

        # Capacity requirement is a derived fact the sampler owns, so this check
        # can't drift from the admission logic. None means the policy self-bounds
        # via backpressure (e.g. the over-sampled windowed policy).
        required_capacity = self._sampler.required_buffer_capacity(
            cfg.target_groups_per_step
        )
        if (
            required_capacity is not None
            and cfg.max_buffered_rollouts < required_capacity
        ):
            raise ValueError(
                f"max_buffered_rollouts ({cfg.max_buffered_rollouts}) is below "
                f"the {type(self._sampler).__name__}'s required capacity "
                f"({required_capacity} = target_groups_per_step "
                f"{cfg.target_groups_per_step} * (lookahead + 1)); the rollout "
                f"pump would deadlock waiting for buffer slots."
            )

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # The rollout batch counter (dispatch index) is now owned by the sampler
        # (``BaseSampler._dispatch_index``), which gates admission and stamps
        # target_step — see ``_rollout_pump`` / ``PromptGroupSampler.admit``.

        # Active rollout tasks used by downstream synchronization/drain paths.
        # TaskGroup remains responsible for task ownership and cancellation.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released after clear_samples.
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            cfg.max_buffered_rollouts
        )

        self._trainer_version: int = 0
        self._train_steps: int = 0
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
        }

        # Completed prompt-list passes; only advances when
        # cfg.max_num_epochs is set (see _rollout_pump).
        self._current_epoch: int = 0

        print(
            f"SingleControllerActor: sampler={cfg.sampler.name} "
            f"buffer={cfg.max_buffered_rollouts} "
            f"inflight={cfg.max_inflight_prompts} "
            f"transport={cfg.refit_cfg.impl}",
            flush=True,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
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

        return {
            "train_steps": self._train_steps,
            "trainer_version": self._trainer_version,
            "epochs": self._current_epoch,
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
          0. await sampler.admit(...) — blocks until the batch may dispatch and
             returns the target_step stamp (the staleness gate lives in the
             sampler, not here).

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager reserves a slot, runs the rollout, then commits the
             group via TQReplayBuffer (→ dp_client.put_samples + mark ready)
          5. Decrement _inflight_rollouts
        """
        sem = asyncio.Semaphore(self._cfg.max_inflight_prompts)
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            task_started_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            self._inflight_rollouts += 1
            try:
                await self._rollout_manager.generate_and_push(
                    prompt, target_step=target_step
                )
            except BaseException:
                # On success ownership transfers to the train pump, which
                # releases this permit after consuming the committed group.
                self._buffer_capacity.release()
                raise
            finally:
                self._inflight_rollouts -= 1
                sem.release()

            if self._cfg.diagnostics:
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

        max_epochs = self._cfg.max_num_epochs
        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
                for prompt_batch in self._dataloader:
                    # The sampler owns admission: it blocks until this batch may
                    # dispatch (per its own staleness algorithm) and returns the
                    # target_step to stamp (None when the policy doesn't stamp).
                    # Keeping the gate here means the rollout pump follows
                    # whichever sampler is selected without a second copy of the
                    # gating logic to keep in sync.
                    target_step = await self._sampler.admit(
                        trainer_version_fn=lambda: self._trainer_version
                    )

                    for prompt_idx in range(prompt_batch.size):
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
        target_groups = (
            self._cfg.target_groups_per_step or self._cfg.min_groups_per_batch
        )

        # TODO: fix the prev_logprobs_required and reference_logprobs_required logic
        prev_logprobs_required = self._cfg.advantage_policy_logprobs_field is not None
        reference_logprobs_required = (
            self._cfg.advantage_reference_logprobs_field is not None
        )

        while self._train_steps < self._cfg.max_train_steps:
            version_during_step = self._trainer_version
            groups_dispatched = 0
            min_sample_version = None
            step_open = False

            with self._timer.time("total_step_time"):
                while groups_dispatched < target_groups:
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
                        max_prompt_groups = target_groups - groups_dispatched
                        min_prompt_groups = min(
                            self._cfg.min_groups_per_batch,
                            max_prompt_groups,
                        )
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=min_prompt_groups,
                            max_prompt_groups=max_prompt_groups,
                        )

                        # If no batch is selectable, sleep and retry
                        if train_meta is None:
                            await asyncio.sleep(0.005)
                            continue

                        # Release buffer capacity
                        for _ in range(num_groups):
                            self._buffer_capacity.release()

                    # Compute prev_logprobs / ref_logprobs
                    with self._timer.time("logprob_inference_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_lp_inference)
                    with self._timer.time("policy_and_reference_logprobs"):
                        if prev_logprobs_required:
                            await asyncio.to_thread(
                                self._trainer.get_logprobs_from_meta, train_meta
                            )
                        if reference_logprobs_required:
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
                        partition_id=self._cfg.partition_id,
                    )

                    groups_dispatched += num_groups

                if not step_open:
                    print(
                        "train_pump: rollout exhausted before any group ready",
                        flush=True,
                    )
                    break

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
                    await self._sync_weights()

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

            # TODO: checkpointing (save_period/top-k metric_name,
            #   policy.save_checkpoint, dataloader state, TQReplayBuffer state).
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
                f"train step {self._train_steps}/{self._cfg.max_train_steps}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

    async def _sync_weights(self) -> None:
        """Drain in-flight rollouts then synchronize weights.

        SC owns the drain gate (when to sync); WeightSynchronizer owns how.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. drain _inflight_rollouts → 0  (5ms poll)
          3. weight_synchronizer.sync_weights(trainer_version)
          4. _rollout_permitted.set()   — resume
        """
        self._rollout_permitted.clear()

        # Drain: wait for all in-flight rollouts to complete before NCCL
        # Critical: if GenWorker has queued calls when NCCL init is dispatched,
        # the init sits behind them — trainer blocks in rendezvous → deadlock
        drain_start = time.monotonic()
        while self._inflight_rollouts > 0:
            await asyncio.sleep(0.005)

        drain_elapsed = time.monotonic() - drain_start
        print(
            f"  _sync_weights: drained in {drain_elapsed:.3f}s, "
            f"syncing weights v{self._trainer_version}",
            flush=True,
        )

        t0 = time.monotonic()
        await self._weight_synchronizer.sync_weights(self._trainer_version)
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_permitted.set()

    async def _advantage_stage(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.
        """
        if not self._cfg.advantage_enabled:
            return meta
        assert self._advantage_estimator is not None

        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        prompt_ids = tensor_field(data, PROMPT_IDS_FIELD)
        rewards = squeeze_trailing_unit_dim(tensor_field(data, REWARD_FIELD)).float()
        token_mask = tensor_field(data, TOKEN_MASK_FIELD).float()
        sample_mask = squeeze_trailing_unit_dim(
            tensor_field(data, SAMPLE_MASK_FIELD)
        ).float()
        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in self._cfg.advantage_repeated_batch_fields:
            repeated_batch[field_name] = squeeze_trailing_unit_dim(
                tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if self._cfg.advantage_policy_logprobs_field is not None:
            kwargs["logprobs_policy"] = tensor_field(
                data,
                self._cfg.advantage_policy_logprobs_field,
            )
        if self._cfg.advantage_reference_logprobs_field is not None:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                self._cfg.advantage_reference_logprobs_field,
            )

        advantages = self._advantage_estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            repeated_batch=repeated_batch,
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
                {ADVANTAGE_OUTPUT_FIELD: advantages},
            ),
        )
        return meta.with_fields([ADVANTAGE_OUTPUT_FIELD])

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        fields = [
            PROMPT_IDS_FIELD,
            REWARD_FIELD,
            TOKEN_MASK_FIELD,
            SAMPLE_MASK_FIELD,
            *self._cfg.advantage_repeated_batch_fields,
        ]
        if self._cfg.advantage_policy_logprobs_field is not None:
            fields.append(self._cfg.advantage_policy_logprobs_field)
        if self._cfg.advantage_reference_logprobs_field is not None:
            fields.append(self._cfg.advantage_reference_logprobs_field)
        return list(dict.fromkeys(fields))
