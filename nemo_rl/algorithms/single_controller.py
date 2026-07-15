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
  _train_pump    → dp_client.claim_meta(...) → StalenessSampler
                 → _advantage_stage(meta) → dp_client.get_samples(...)
                                        → adv_estimator.compute_advantage(...)
                                        → dp_client.put_samples(...)
                 → trainer.begin/train_microbatch/finish_train_step (split API,
                     driver-side TQPolicy via asyncio.to_thread)
                     Trainer → dp_client.get_samples(...)   (via its own client)
                 → dp_client.clear_samples(...)             ← SC clears after train
  _sync_weights  → drain _inflight_rollouts → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from typing import Any, Literal, Optional

import ray
import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict

from nemo_rl.algorithms.grpo import GRPOLoggerConfig
from nemo_rl.algorithms.staleness_sampler import (
    StalenessSampler,
    count_groups,
    incomplete_group_indices,
    min_weight_version,
)
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

    # Staleness. A "group" is the atomic training unit: group_size
    # samples sharing one source prompt. GRPO consumers set group_size =
    # num generations per prompt; SFT/OPD-style consumers degenerate
    # cleanly to group_size=1 (every sample its own group).
    max_weight_staleness_versions: int = 1
    min_groups_per_batch: int = 2
    target_groups_per_step: Optional[int] = None
    group_size: int = 4
    batch_selection_strategy: Literal[
        "strict_on_policy",
        "staleness_window",
    ] = "strict_on_policy"

    # Concurrency limits
    max_inflight_prompts: int = 8
    max_buffered_rollouts: int = 8  # _buffer_capacity semaphore size

    # Training
    max_train_steps: int = 10
    max_rollout_prompts: int = 32
    # Bounded dataset passes, mirroring grpo.py's max_num_epochs loop. One
    # epoch = one pass over the prompt list. None preserves the unbounded
    # behavior: max_rollout_prompts alone caps dispatch (cycling through
    # the prompt list) and no epoch accounting is performed.
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
    consumer_task_name: str = "train"
    claim_required_fields: list[str] = Field(default_factory=lambda: ["input_ids"])
    max_claim_groups: int = 8

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


def warn_if_staleness_window_below_minibatches(
    cfg: SingleControllerConfig,
) -> None:
    """Warn when the staleness window is too small to cover one outer step.

    ``_train_pump`` runs ``num_minibatches`` begin/finish cycles per outer
    step, bumping ``_trainer_version`` after each ``finish_train_step``
    (one opt.step) but refreshing generation weights (``_sync_weights``)
    only once, at the end of the outer step. A group produced at the version
    the step started on therefore ages by up to ``num_minibatches - 1``
    versions before the next sync. When the effective staleness window is
    smaller than that, such groups become unselectable mid-step — the
    sampler skips them and ``_evict_stale_claimed`` may drop them as stale —
    so the pump can spin or thrash instead of consuming them.

    Warns rather than raises: a run may intentionally accept the resulting
    eviction churn. Self-contained (does not assume ``cfg`` has been coerced
    by ``__init__``) so it is unit-testable without constructing the actor.

    Args:
        cfg: SingleController config. ``strict_on_policy`` pins the effective
            window to 0 regardless of ``max_weight_staleness_versions``.
    """
    effective_window = (
        0
        if cfg.batch_selection_strategy == "strict_on_policy"
        else cfg.max_weight_staleness_versions
    )
    target_groups = cfg.target_groups_per_step or cfg.min_groups_per_batch
    samples_per_step = target_groups * cfg.group_size
    train_global_batch_size = cfg.train_global_batch_size or samples_per_step
    groups_per_minibatch = train_global_batch_size // cfg.group_size
    if groups_per_minibatch <= 0:
        return
    num_minibatches = target_groups // groups_per_minibatch
    if num_minibatches > 1 and effective_window < num_minibatches - 1:
        print(
            f"WARNING: max_weight_staleness_versions (effective window "
            f"{effective_window}) is smaller than num_minibatches - 1 "
            f"({num_minibatches - 1}): each outer step runs {num_minibatches} "
            f"optimizer steps but syncs generation weights only once, at the "
            f"end, so groups produced at the version the step began on age by "
            f"up to {num_minibatches - 1} versions before the next sync and "
            f"become unselectable (skipped, then evicted as stale) mid-step "
            f"— the train pump may spin or thrash. Remedy: raise "
            f"max_weight_staleness_versions to at least {num_minibatches - 1}, "
            f"or lower target_groups_per_step * group_size / "
            f"train_global_batch_size to reduce num_minibatches.",
            flush=True,
        )


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
        prompts: list[str],
        dp_client_handle: Any,
        gen_handle: Any,
        trainer_handle: Any,
        weight_synchronizer: Any,
        loss_fn: Any,
        advantage_estimator: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._prompts = prompts
        self._dp_client = dp_client_handle
        self._gen = gen_handle
        self._trainer = trainer_handle
        self._weight_synchronizer = weight_synchronizer
        self._loss_fn = loss_fn
        self._advantage_estimator = advantage_estimator

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(cfg.logger)  # type: ignore
        self._timer = Timer()

        if cfg.advantage_enabled and self._advantage_estimator is None:
            raise ValueError(
                "advantage_enabled=True requires an advantage_estimator instance"
            )

        # batch_selection_strategy is Literal-typed; pydantic rejects unknown
        # values at config construction, so no runtime assert is needed here.
        if cfg.batch_selection_strategy == "strict_on_policy":
            cfg.max_weight_staleness_versions = 0
            print(
                "Using strict_on_policy, auto setting "
                "max_weight_staleness_versions to 0.",
                flush=True,
            )
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
        # num_minibatches > 1 runs multiple opt.steps (each bumps
        # trainer_version) between weight syncs; warn if the staleness window
        # cannot keep same-step groups selectable across those bumps.
        warn_if_staleness_window_below_minibatches(cfg)
        self._sampler = StalenessSampler(cfg.max_weight_staleness_versions)

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released after clear_samples.
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            cfg.max_buffered_rollouts
        )

        self._trainer_version: int = 0
        self._train_steps: int = 0
        self._rollout_done: bool = False
        # Completed prompt-list passes; only advances when
        # cfg.max_num_epochs is set (see _rollout_pump).
        self._current_epoch: int = 0
        self._claimed_meta: KVBatchMeta | None = None
        self._step_consumed_sample_ids: list[str] = []

        print(
            f"SingleControllerActor: staleness_cap="
            f"{cfg.max_weight_staleness_versions} "
            f"buffer={cfg.max_buffered_rollouts} "
            f"inflight={cfg.max_inflight_prompts} "
            f"transport={cfg.refit_cfg.impl}",
            flush=True,
        )

    def _timed(self, label: str) -> Any:
        """Time a code block with the SC Timer when a logger is attached.

        No-op when no logger is configured. Use as
        ``with self._timed("phase"): ...``.
        """
        return self._timer.time(label) if self._timer is not None else nullcontext()

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())

        await train_task

        rollout_task.cancel()
        try:
            await rollout_task
        except asyncio.CancelledError:
            pass

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
        """Dispatch prompts as concurrent coroutines, one per prompt group.

        Flow per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Wait for _rollout_permitted (paused during weight sync)
          3. Call gen.generate_and_push(prompt, dp_client) — RPC to GenWorker
             GenWorker generates and calls DataPlane put_samples directly
          4. Decrement _inflight_rollouts
        """
        n = self._cfg.max_rollout_prompts
        max_epochs = self._cfg.max_num_epochs
        sem = asyncio.Semaphore(self._cfg.max_inflight_prompts)

        start = time.monotonic()
        print(f"rollout_pump: dispatching {n} prompts", flush=True)

        async def _one_group(prompt: str) -> None:
            await self._buffer_capacity.acquire()
            await self._rollout_permitted.wait()
            async with sem:
                self._inflight_rollouts += 1
                try:
                    await self._ray_get(
                        self._gen.generate_and_push.remote(prompt, self._dp_client)
                    )
                    if self._cfg.diagnostics:
                        print(
                            f"  rollout done for prompt='{prompt[:20]}...'",
                            flush=True,
                        )
                finally:
                    self._inflight_rollouts -= 1

        dispatched = 0
        if max_epochs is None:
            # Unbounded-epoch path: max_rollout_prompts alone caps dispatch,
            # all prompts in flight together (cycling through the list).
            tasks = [
                asyncio.ensure_future(_one_group(self._prompts[i % len(self._prompts)]))
                for i in range(n)
            ]
            await asyncio.gather(*tasks)
            dispatched = n
        else:
            # Epoch-bounded path: one gather per pass over the prompt list,
            # mirroring grpo.py's per-epoch dataset iteration. Ends at
            # whichever bound is hit first (epochs or total prompt budget).
            while dispatched < n and self._current_epoch < max_epochs:
                k = min(len(self._prompts), n - dispatched)
                tasks = [
                    asyncio.ensure_future(_one_group(self._prompts[i]))
                    for i in range(k)
                ]
                await asyncio.gather(*tasks)
                dispatched += k
                self._current_epoch += 1
                print(
                    f"rollout_pump: epoch {self._current_epoch}/{max_epochs} "
                    f"complete ({dispatched}/{n} prompts)",
                    flush=True,
                )

        self._rollout_done = True
        print(
            f"rollout_pump: finished {dispatched} prompts in "
            f"{time.monotonic() - start:.2f}s",
            flush=True,
        )

    async def _train_pump(self) -> None:
        """Per-prompt-group streaming train loop.

        Per step:
          - Lazy ``begin_train_step`` on first ready group.
          - Per ready group: optional logprob refresh
            (``get_logprobs_from_meta`` / ``get_reference_policy_logprobs_from_meta``) →
            ``_advantage_stage`` → ``train_microbatches_from_meta``.
          - End-of-step: ``finish_train_step`` →
            single ``clear_samples`` → ``_sync_weights``.

        **Concurrency contract:**
        ``self._trainer`` is a driver-side ``TQPolicy`` object, not a Ray
        actor. Trainer calls run via
        ``asyncio.to_thread`` and are awaited sequentially, so the worker's
        ``_train_step_state`` accumulators (``local_valid_seqs +=``,
        ``mb_losses.append``, ``all_mb_metrics.append``), which are not
        concurrency-safe, see exactly one call at a time. ``to_thread``
        keeps the fan-out + internal ``ray.get`` off the event loop so the
        rollout pump makes progress during trainer calls; exceptions
        surface at the corresponding ``await``.
        """
        # TODO: fix the prev_logprobs_required and reference_logprobs_required logic
        prev_logprobs_required = self._cfg.advantage_policy_logprobs_field is not None
        reference_logprobs_required = (
            self._cfg.advantage_reference_logprobs_field is not None
        )

        while self._train_steps < self._cfg.max_train_steps:
            # __init__ coerces None → min_groups_per_batch (int);
            # the assert narrows the Optional[int] type for pyrefly.
            assert self._cfg.target_groups_per_step is not None
            assert self._cfg.train_global_batch_size is not None
            target_groups: int = self._cfg.target_groups_per_step
            # Mini-batch aggregation: K groups per begin/finish cycle, where
            # K * group_size == train_global_batch_size. Each
            # cycle is one opt.step. Mirrors sync's gb_idx loop in the worker.
            groups_per_minibatch = (
                self._cfg.train_global_batch_size // self._cfg.group_size
            )
            num_minibatches = target_groups // groups_per_minibatch
            rollout_exhausted = False

            for mb_idx in range(num_minibatches):
                groups_dispatched = 0
                step_open = False
                step_min_weight_version: int | None = None

                # No SC-side error handling: a mid-cycle worker failure
                # propagates out of run(). The worker restores its own hooks
                # on failure (see megatron_policy_worker); abort_train_step
                # + a retry policy return with fault-tolerance support.
                while groups_dispatched < groups_per_minibatch:
                    await asyncio.sleep(0)
                    await self._claim_available_meta()
                    evicted_meta = await self._evict_stale_claimed()
                    if evicted_meta is not None:
                        evicted_groups = count_groups(
                            evicted_meta,
                            group_size=self._cfg.group_size,
                        )
                        for _ in range(evicted_groups):
                            self._buffer_capacity.release()

                    group_indices = None
                    if self._claimed_meta is not None and self._claimed_meta.size > 0:
                        group_indices = self._sampler.select_one_group(
                            self._claimed_meta,
                            trainer_version=self._trainer_version,
                            group_size=self._cfg.group_size,
                        )

                    if group_indices is None:
                        if self._rollout_done:
                            # No group is selectable and no more samples
                            # will arrive: flush groups that can never
                            # complete so the emptiness check fires
                            # instead of spinning forever.
                            await self._flush_incomplete_groups()
                            if (
                                self._claimed_meta is None
                                or self._claimed_meta.size == 0
                            ):
                                rollout_exhausted = True
                                break
                        await asyncio.sleep(0.005)
                        continue

                    group_meta = self._claimed_meta.subset(group_indices)
                    self._claimed_meta = self._claimed_meta.drop(group_indices)

                    if prev_logprobs_required or reference_logprobs_required:
                        with self._timed("policy_and_reference_logprobs"):
                            if prev_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_logprobs_from_meta,
                                    group_meta,
                                )
                            if reference_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_reference_policy_logprobs_from_meta,
                                    group_meta,
                                )

                    # Advantage stage — inline in the train pump, not a
                    # standalone pump task.
                    with self._timed("advantage_stage"):
                        group_meta = await self._advantage_stage(group_meta)

                    if not step_open:
                        # gbs/mbs default to worker-side cfg when None; SC
                        # owns only loss_fn (stable across the whole run).
                        await asyncio.to_thread(
                            self._trainer.begin_train_step,
                            self._loss_fn,
                        )
                        step_open = True

                    await asyncio.to_thread(
                        self._trainer.train_microbatches_from_meta,
                        group_meta,
                    )
                    groups_dispatched += 1
                    self._buffer_capacity.release()
                    self._step_consumed_sample_ids.extend(group_meta.sample_ids)
                    group_min_v = min_weight_version(group_meta)
                    if group_min_v is not None:
                        step_min_weight_version = (
                            group_min_v
                            if step_min_weight_version is None
                            else min(step_min_weight_version, group_min_v)
                        )

                if not step_open:
                    # No groups consumed this mini-batch — either rollouts
                    # exhausted before any group arrived, or the outer
                    # loop broke without dispatching. Skip finish/cleanup.
                    print(
                        f"train_pump: rollout exhausted at mb {mb_idx} "
                        f"(no groups for this opt.step)",
                        flush=True,
                    )
                    break

                # finish_train_step returns step metrics. trainer_version
                # is driver-owned (workers don't emit it) and bumps after
                # this call succeeds. The new value propagates to rollouts
                # via _sync_weights at end of the outer step. Capture
                # train_results for the logger emit below.
                with self._timed("policy_training"):
                    train_results = await asyncio.to_thread(
                        self._trainer.finish_train_step
                    )

                # finish_train_step succeeded → opt.step ran on the worker,
                # model weights are advanced. Bump the version immediately so
                # SC's counter reflects worker state even if clear_samples
                # below raises.
                prev_trainer_version = self._trainer_version
                self._trainer_version += 1

                consumed_ids = list(self._step_consumed_sample_ids)
                self._step_consumed_sample_ids = []
                with self._timed("clear_samples"):
                    await self._call_dp(
                        "clear_samples",
                        sample_ids=consumed_ids,
                        partition_id=self._cfg.partition_id,
                    )
                lag = (
                    prev_trainer_version - step_min_weight_version
                    if step_min_weight_version is not None
                    else 0
                )
                print(
                    f"train step {self._train_steps + 1}/"
                    f"{self._cfg.max_train_steps}  "
                    f"mb {mb_idx + 1}/{num_minibatches}  "
                    f"trainer_v={self._trainer_version}  lag={lag}  "
                    f"batch_size={len(consumed_ids)}",
                    flush=True,
                )

                # Log metrics
                self._logger.log_metrics(
                    train_results, step=self._trainer_version, prefix="train"
                )
                self._logger.log_metrics(
                    self._timer.get_timing_metrics(),
                    step=self._trainer_version,
                    prefix="timing/train",
                    step_finished=True,
                )
                self._timer.reset()

                if rollout_exhausted:
                    # Inner loop terminated early due to rollout exhaustion;
                    # the cycle still completed (step_open was True), so we
                    # finished and logged above. Exit the mini-batch loop now
                    # — no more groups are coming.
                    break

            if rollout_exhausted:
                # No more rollouts; stop the outer training loop entirely
                # rather than running empty mini-batches.
                break

            # One sync per outer step covers all opt.steps in this iteration.
            with self._timed("sync_weights"):
                await self._sync_weights()
            self._train_steps += 1

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

        prompt_ids = _tensor_field(data, PROMPT_IDS_FIELD)
        rewards = _squeeze_trailing_unit_dim(_tensor_field(data, REWARD_FIELD)).float()
        token_mask = _tensor_field(data, TOKEN_MASK_FIELD).float()
        sample_mask = _squeeze_trailing_unit_dim(
            _tensor_field(data, SAMPLE_MASK_FIELD)
        ).float()
        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in self._cfg.advantage_repeated_batch_fields:
            repeated_batch[field_name] = _squeeze_trailing_unit_dim(
                _tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if self._cfg.advantage_policy_logprobs_field is not None:
            kwargs["logprobs_policy"] = _tensor_field(
                data,
                self._cfg.advantage_policy_logprobs_field,
            )
        if self._cfg.advantage_reference_logprobs_field is not None:
            kwargs["logprobs_reference"] = _tensor_field(
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

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=_fields_for_put(
                meta,
                {ADVANTAGE_OUTPUT_FIELD: advantages},
            ),
        )
        return meta.with_fields([ADVANTAGE_OUTPUT_FIELD])

    # ── utility helpers ────────────────────────────────────────────────────

    async def _claim_available_meta(self) -> None:
        """Claim currently-ready rows and append them to the local scheduler cache.

        TODO: replace this with a non-consuming metadata listing API.
        ``claim_meta`` advances TQ's per-task cursor, so SC must keep a
        local cache of claimed-but-not-yet-trained samples for now.
        """
        batch_size = self._cfg.max_claim_groups * self._cfg.group_size
        meta = await self._call_dp(
            "claim_meta",
            partition_id=self._cfg.partition_id,
            task_name=self._cfg.consumer_task_name,
            required_fields=self._claim_required_fields(),
            batch_size=batch_size,
            blocking=False,
            timeout_s=0.0,
        )
        if meta.size == 0:
            return
        if self._claimed_meta is None or self._claimed_meta.size == 0:
            self._claimed_meta = meta
        else:
            self._claimed_meta = self._claimed_meta.concat(meta)

    def _claim_required_fields(self) -> list[str]:
        fields = list(self._cfg.claim_required_fields)
        if self._cfg.advantage_enabled:
            fields.extend(self._advantage_input_fields())
        return list(dict.fromkeys(fields))

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

    async def _evict_stale_claimed(self) -> KVBatchMeta | None:
        if self._claimed_meta is None or self._claimed_meta.size == 0:
            return None
        indices = self._sampler.evictable_indices(
            self._claimed_meta,
            trainer_version=self._trainer_version,
            group_size=self._cfg.group_size,
        )
        if not indices:
            return None
        evicted_meta = self._claimed_meta.subset(indices)
        print(
            f"  evicting {evicted_meta.size} stale samples from "
            f"{count_groups(evicted_meta, group_size=self._cfg.group_size)} "
            f"prompt group(s)",
            flush=True,
        )
        await self._call_dp(
            "clear_samples",
            sample_ids=evicted_meta.sample_ids,
            partition_id=evicted_meta.partition_id,
        )
        self._claimed_meta = self._claimed_meta.drop(indices)
        return evicted_meta

    async def _flush_incomplete_groups(self) -> None:
        """Drop groups that can never become selectable after rollout shutdown.

        With ``_rollout_done`` set, a group that is uncommitted or short of
        ``expected_num_samples`` will never receive more rows, yet the sampler
        neither selects nor evicts it — without this flush the train pump
        spins forever and ``run()`` hangs. Drain rows still unclaimed at
        DataPlane first: a group can straddle a ``claim_meta`` batch boundary,
        so incomplete-in-``_claimed_meta`` does not yet prove
        incomplete-in-partition.
        """
        while True:
            before = self._claimed_meta.size if self._claimed_meta is not None else 0
            await self._claim_available_meta()
            after = self._claimed_meta.size if self._claimed_meta is not None else 0
            if after == before:
                break
        if self._claimed_meta is None or self._claimed_meta.size == 0:
            return
        indices = incomplete_group_indices(
            self._claimed_meta,
            group_size=self._cfg.group_size,
        )
        if not indices:
            return
        dropped = self._claimed_meta.subset(indices)
        print(
            f"WARNING: rollout done: dropping {dropped.size} sample(s) from "
            f"incomplete prompt group(s) that can no longer complete",
            flush=True,
        )
        await self._call_dp(
            "clear_samples",
            sample_ids=dropped.sample_ids,
            partition_id=dropped.partition_id,
        )
        self._claimed_meta = self._claimed_meta.drop(indices)
        # No _buffer_capacity release: the rollout pump has exited, so no
        # dispatcher will acquire again this run.


def _tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"advantage stage expected tensor field {field_name!r}; got {type(value)}"
        )
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def _squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def _fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])
