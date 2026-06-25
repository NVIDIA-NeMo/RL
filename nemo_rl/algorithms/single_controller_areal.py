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

CPU-only Ray actor that runs three concurrent pumps and coordinates the
other actors via lightweight RPCs. SC sends control signals and reads
metadata only — model tensors still move through DataPlane or NCCL.

Data flow:
  _rollout_pump  → rollout_manager.generate_and_push(prompt)
                     → TQReplayBuffer.reserve claims a slot at dispatch time;
                       run_rollout runs; TQReplayBuffer.commit tensorizes the
                       record, writes N training rows to TQ, and marks ready.
  _train_pump    → sampler.evict → buffer.remove (stale groups, with DP clear).
                 → sampler.select → drops chosen groups from buffer, returns
                   KVBatchMeta of K groups (or None); meta is already trainable.
                 → _advantage_pump (get → compute → put).
                 → trainer.train_on_meta.
                 → dp_client.clear_samples (trained groups; buffer already dropped).
  _sync_weights  → drain _inflight_rollouts → WeightSynchronizer.sync_weights.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterator, Optional, Union

import ray
import torch

from nemo_rl.algorithms.async_utils.staleness_sampler import StalenessSampler
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    MasterConfig,
    WeightSyncConfig,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerBundle
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.models.generation.sglang import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer

Generation = Union[VllmGeneration, SGLangGeneration]


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerArealActor:
    """CPU-only Ray actor that orchestrates the AReaL decoupled-PPO loop.

    Standalone fork of :class:`SingleControllerActor` (no inheritance / mixin):
    the base controller's machinery is copied verbatim so the AReaL two-phase
    train loop and interruptible refit can diverge without modifying the base. 
    At defaults (``num_minibatches=1``) this actor behaves identically to the 
    base controller.

    Owns three concurrent asyncio tasks:
      - _rollout_pump: dispatches prompts via RolloutManager; reserve+commit in
        TQReplayBuffer preserves dispatch order
      - _train_pump:   evicts stale groups, samples a batch, trains, drops it
      - _sync_weights: drain gate + weight synchronization

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        bundle: SingleControllerBundle,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            bundle: Pre-built bundle from setup_single_controller. Tests can
                construct a bundle by hand (or with fakes) to bypass the real factories.
        """
        self._advantage_cfg = AdvantageConfig()
        self._weight_sync_cfg = WeightSyncConfig()
        self._partition_id: str = bundle.partition_id
        self._diagnostics: bool = False

        self._master_config = master_config
        self._async_cfg = master_config.async_rl
        self._dp_client = bundle.dp_client
        self._gen: Generation = bundle.gen_handle
        self._trainer: TQPolicy = bundle.trainer_handle
        self._dataloader = bundle.dataloader
        self._weight_synchronizer = bundle.weight_synchronizer
        self._advantage_estimator = bundle.advantage_estimator
        self._loss_fn = bundle.loss_fn
        self._buffer = bundle.tq_buffer
        self._rollout_manager = bundle.rollout_manager
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._timer = Timer()

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = bundle.train_cluster
        self._inference_cluster = bundle.inference_cluster

        if self._async_cfg.target_prompt_groups_per_step is None:
            self._async_cfg.target_prompt_groups_per_step = (
                self._async_cfg.min_prompt_groups_per_batch
            )
        if (
            self._async_cfg.target_prompt_groups_per_step
            < self._async_cfg.min_prompt_groups_per_batch
        ):
            raise ValueError(
                f"target_prompt_groups_per_step ({self._async_cfg.target_prompt_groups_per_step}) "
                f"must be >= min_prompt_groups_per_batch ({self._async_cfg.min_prompt_groups_per_batch})"
            )

        if self._async_cfg.batch_selection_strategy == "strict_on_policy":
            self._async_cfg.max_weight_staleness_versions = 0
            self._async_cfg.over_sampling = False
            print(
                "Using strict_on_policy, auto setting max_weight_staleness_versions to 0 and over_sampling to False.",
                flush=True,
            )

        # AReaL is over_sampling=False ONLY. Off-policyness is controlled by the
        # staleness window (η) + the decoupled-PPO π_prox/π_behav reweighting, NOT
        # by over-producing and discarding stragglers that age out (an orthogonal
        # async strategy AReaL avoids). Validate + derive the buffer size from η.
        self._validate_buffer_config(self._async_cfg)
        # force_in_order needs over_sampling=False, which _validate_buffer_config
        # now guarantees for the AReaL path, so no separate check is needed.

        # SC split path does one optimizer.step per RL step.
        # TODO: support multi-mini-step (legacy train() does gbs-sized
        # mini-steps with shared prev_logprobs).
        rl_step_samples = (
            self._master_config.grpo["num_prompts_per_step"]
            * self._master_config.grpo["num_generations_per_prompt"]
        )
        train_gbs = self._master_config.policy["train_global_batch_size"]
        if rl_step_samples != train_gbs:
            raise ValueError(
                f"num_prompts_per_step * num_generations_per_prompt "
                f"({rl_step_samples}) must equal policy.train_global_batch_size "
                f"({train_gbs}) so that one RL step maps to exactly one "
                f"optimizer.step. Multi-mini-step inside a single RL step is "
                f"not supported on the SC split path."
            )

        self._sampler = StalenessSampler(
            self._buffer,
            max_staleness_versions=self._async_cfg.max_weight_staleness_versions,
            require_order=not self._async_cfg.over_sampling,
            force_in_order=self._async_cfg.force_in_order,
        )

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        # Set True by _rollout_pump when its epoch loop completes. Makes
        # _collect_full_batch tail-safe (§4, §7.10): exhaustion is only declared
        # once the dispatch loop is done AND no rollouts are still in flight, so
        # the last in-flight rollouts are trained rather than dropped.
        self._rollout_done: bool = False

        # over_sampling=False batch gate: farthest trainer_version covered by
        # already-dispatched batches.
        self._max_rollout_version: int = -1

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )

        self._trainer_version: int = 0
        self._train_steps: int = 0
        self._step_consumed_sample_ids: list[str] = []
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
        }

        print(
            f"SingleControllerArealActor: "
            f"staleness_cap={self._async_cfg.max_weight_staleness_versions} "
            f"buffer={self._async_cfg.max_buffered_rollouts} "
            f"inflight={self._async_cfg.max_inflight_prompts} "
            f"over_sampling={self._async_cfg.over_sampling} "
            f"num_minibatches={self._async_cfg.num_minibatches} "
            f"refit_invalidate_kv_cache={self._async_cfg.refit_invalidate_kv_cache} "
            f"transport={self._weight_sync_cfg.transport}",
            flush=True,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        # Synchronize weights before starting the pumps
        await self._sync_weights()

        # Start the rollout and train pumps
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())

        # Wait until the train pump is done
        await train_task
        self._logger.finish()

        # Cancel the rollout pump and any in-flight dispatches so we exit immediately.
        rollout_task.cancel()
        try:
            await rollout_task
        except asyncio.CancelledError:
            pass
        inflight = list(self._dispatched_rollouts)
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

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

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch (over_sampling=False):
          0. Wait while _max_rollout_version >= trainer_version + max_staleness,
             then claim the next step by incrementing _max_rollout_version.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager reserves a slot, runs the rollout, then commits the
             group via TQReplayBuffer (→ dp_client.put_samples + mark ready)
          5. Decrement _inflight_rollouts
        """
        sem = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        over_sampling = self._async_cfg.over_sampling
        max_staleness = self._async_cfg.max_weight_staleness_versions
        force_in_order = self._async_cfg.force_in_order
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec, target_step: Optional[int]
        ) -> None:
            self._inflight_rollouts += 1
            try:
                await self._rollout_manager.generate_and_push(
                    prompt, target_step=target_step
                )
                if self._diagnostics:
                    content = ""
                    for i in range(len(prompt["message_log"])):
                        if prompt["message_log"][i]["role"] == "user":
                            content = prompt["message_log"][i]["content"]
                            break
                    print(f"  rollout done for prompt='{content[:20]}...'", flush=True)
            finally:
                self._inflight_rollouts -= 1
                sem.release()

        max_epochs = self._master_config.grpo["max_num_epochs"]
        epoch = 0
        while max_epochs is None or epoch < max_epochs:
            for prompt_batch in self._dataloader:
                # over_sampling=False: batch-level gate on max_rollout_version.
                if not over_sampling:
                    while (
                        self._max_rollout_version
                        >= self._trainer_version + max_staleness
                    ):
                        await asyncio.sleep(0.005)
                    self._max_rollout_version += 1

                # target_step = batch dispatch index when force_in_order is on.
                target_step = self._max_rollout_version if force_in_order else None

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

                    # dispatch rollout
                    task = asyncio.create_task(
                        _dispatch_one_prompt(prompt, target_step)
                    )
                    self._dispatched_rollouts.add(task)
                    task.add_done_callback(self._dispatched_rollouts.discard)
            epoch += 1

        # Mark dispatch loop exhausted so _collect_full_batch can declare the
        # tail batch ready once the remaining in-flight rollouts commit (§7.10).
        self._rollout_done = True
        print(f"rollout_pump: completed {epoch} epoch(s)", flush=True)

    async def _train_pump(self) -> None:
        """Drain stale groups, sample, train, drop.

        Per step:
          1. sampler.evict drops stale groups from the buffer and clears their TQ rows.
          2. sampler.select returns K prompt groups (or None) and drops them from the
             buffer; DP rows survive so the trainer can read them. Already trainable —
             buffer wrote training-shaped rows at rollout time.
          3. _advantage_pump(train_meta).
          4. trainer.train_microbatch_from_meta + finish_train_step.
          5. dp_client.clear_samples on consumed sample_ids; release _buffer_capacity
             per dropped group, then sync.

        P0 note: this is a verbatim fork of the base controller loop with the
        AReaL ``step_results`` scaffolding added — each ``finish_train_step``
        result is collected into ``step_results`` and reduced through the EXISTING
        ``aggregate_step_metrics`` helper. At ``num_minibatches=1`` there is a
        single optimizer step per training step, so ``step_results`` holds exactly
        one result and the reduced metrics are byte-for-byte identical to the base
        controller (no behavior change at defaults). P2 replaces this single-step
        body with the two-phase Phase-1 barrier + per-minibatch Phase-2 loop that
        appends one result per minibatch to the SAME ``step_results`` list.
        """
        adv_cfg = self._advantage_cfg
        grpo_cfg = self._master_config.grpo

        # TODO: fix the compute_prev_logprobs and compute_reference_logprobs logic
        compute_prev_logprobs = adv_cfg.policy_logprobs_field is not None
        compute_reference_logprobs = adv_cfg.reference_logprobs_field is not None

        while self._train_steps < grpo_cfg["max_num_steps"]:
            step_id = f"sc-step-{self._train_steps:06d}"
            # __init__ coerces None → min_prompt_groups_per_batch (int);
            # the assert narrows the Optional[int] type for pyrefly.
            assert self._async_cfg.target_prompt_groups_per_step is not None
            target_groups: int = self._async_cfg.target_prompt_groups_per_step
            groups_dispatched = 0
            step_open = False
            # AReaL scaffolding: one finish_train_step result per optimizer step.
            # P0 (num_minibatches=1) appends exactly one; P2's minibatch loop
            # appends one per minibatch. Reduced via the existing helper below.
            step_results: list[dict[str, Any]] = []

            with self._timer.time("total_step_time"):
                while groups_dispatched < target_groups:
                    await asyncio.sleep(0)

                    # evict stale groups
                    evicted = await self._sampler.evict(
                        current_train_weight=self._trainer_version,
                    )
                    if evicted:
                        print(f"  evicted {evicted} stale prompt group(s)", flush=True)
                        for _ in range(evicted):
                            self._buffer_capacity.release()

                    # TODO @yukih: wait train pump merged, now always return min_prompt_groups_per_batch
                    # need to add a max_prompt_groups_per_batch
                    with self._timer.time("exposed_generation"):
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=self._async_cfg.min_prompt_groups_per_batch,
                        )

                    if train_meta is None:
                        await asyncio.sleep(0.05)
                        continue

                    for _ in range(num_groups):
                        self._buffer_capacity.release()

                    # Compute prev_logprobs / ref_logprobs
                    with self._timer.time("logprob_inference_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_lp_inference)
                    with self._timer.time("policy_and_reference_logprobs"):
                        if compute_prev_logprobs:
                            await asyncio.to_thread(
                                self._trainer.get_logprobs_from_meta, train_meta
                            )
                        if compute_reference_logprobs:
                            await asyncio.to_thread(
                                self._trainer.get_reference_policy_logprobs_from_meta,
                                train_meta,
                            )

                    with self._timer.time("advantage_calculation"):
                        train_meta = await self._advantage_pump(train_meta)

                    # Train
                    with self._timer.time("training_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_training)
                    with self._timer.time("policy_training"):
                        if not step_open:
                            await asyncio.to_thread(
                                self._trainer.begin_train_step,
                                step_id,
                                loss_fn=self._loss_fn,
                            )
                            step_open = True
                        await asyncio.to_thread(
                            self._trainer.train_microbatch_from_meta,
                            step_id,
                            train_meta,
                        )

                    groups_dispatched += num_groups
                    self._step_consumed_sample_ids.extend(train_meta.sample_ids)
                    if train_meta.sequence_lengths:
                        self._step_log_dict["sequence_lengths"].extend(
                            int(s) for s in train_meta.sequence_lengths
                        )

                if not step_open:
                    print(
                        "train_pump: rollout exhausted before any group ready",
                        flush=True,
                    )
                    break

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(
                        self._trainer.finish_train_step, step_id
                    )
                step_results.append(result)
                consumed_ids = list(self._step_consumed_sample_ids)
                self._step_consumed_sample_ids = []
                await self._call_dp(
                    "clear_samples",
                    sample_ids=list(consumed_ids),
                    partition_id=self._partition_id,
                )

                # Reduce the per-optimizer-step finish results through the EXISTING
                # metrics path (no new helper). At num_minibatches=1 step_results
                # holds one result, so aggregate_step_metrics(step_results[-1])
                # reproduces the base controller exactly. P2 reduces all M results.
                step_metrics = aggregate_step_metrics(step_results[-1])
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
            cluster_cfg = self._master_config.cluster
            total_num_gpus = cluster_cfg["num_nodes"] * cluster_cfg["gpus_per_node"]
            if total_time > 0 and "global_valid_toks" in step_metrics:
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
            # generated with; lag = current trainer version - oldest sample version.
            min_sample_version = min(t["weight_version"] for t in train_meta.tags)  # type: ignore
            lag = self._trainer_version - min_sample_version
            print(
                f"train step {self._train_steps}/{grpo_cfg['max_num_steps']}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  "
                f"batch_size={len(consumed_ids)}",
                flush=True,
            )

    # ── AReaL helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_buffer_config(async_cfg: AsyncRLConfig) -> None:
        """Require over_sampling=False and DERIVE the buffer size from the window.

        AReaL controls off-policyness through the staleness window (η) plus the
        decoupled-PPO π_prox/π_behav reweighting — NOT through over_sampling
        ("over-generate and waste stragglers that age out"), which is an
        orthogonal async strategy AReaL deliberately avoids. So this controller
        is scoped to ``over_sampling=False``.

        With ``over_sampling=False`` the buffer size is fully derivable: at most
        one in-window batch per live weight version, i.e.

            max_buffered_rollouts = target_prompt_groups_per_step
                                    * (max_weight_staleness_versions + 1)

        rather than a user knob to set and assert against. This method computes
        that value and writes it back onto ``async_cfg.max_buffered_rollouts`` so
        the ``_buffer_capacity`` semaphore and the startup banner both use the
        derived (auditable) number. If the user also set a conflicting value, we
        log a one-line note and use the derived value — we do NOT error.

        ``target_prompt_groups_per_step`` must already be coerced from ``None``.

        Raises:
            ValueError: if ``over_sampling`` is True (unsupported for AReaL).
        """
        assert async_cfg.target_prompt_groups_per_step is not None
        if async_cfg.over_sampling:
            raise ValueError(
                "AReaL controls off-policyness via the staleness window (η) + "
                "decoupled-PPO reweighting, not over_sampling; set "
                "async_rl.over_sampling=false."
            )

        derived_buffer = async_cfg.target_prompt_groups_per_step * (
            async_cfg.max_weight_staleness_versions + 1
        )
        if async_cfg.max_buffered_rollouts != derived_buffer:
            print(
                f"AReaL: deriving max_buffered_rollouts={derived_buffer} "
                f"(= target_prompt_groups_per_step * "
                f"(max_weight_staleness_versions + 1)); ignoring the configured "
                f"value {async_cfg.max_buffered_rollouts}.",
                flush=True,
            )
        async_cfg.max_buffered_rollouts = derived_buffer

    async def _collect_full_batch(self) -> "KVBatchMeta | None":
        """Phase-1 barrier: drain the staleness sampler until the full batch is claimed.

        Accumulates whole prompt groups until ``target_prompt_groups_per_step``
        groups are in hand, then merges them into a single ``KVBatchMeta``. This
        is the structural difference from the base controller's interleaved
        loop: the *whole* batch is gathered before any training so that π_prox
        (``get_logprobs_from_meta``) and the GRPO advantages can be defined over
        a single, consistent batch (§3, §4).

        ACCUMULATE + MERGE ONLY — this method performs no training, no logprob
        inference, and no advantage computation. P2 wires the call site in
        ``_train_pump`` and adds those phases around it.

        Each loop iteration yields to ``_rollout_pump`` (``asyncio.sleep(0)``) so
        generation of batches i+1..i+η keeps streaming in the background — that
        overlap is exactly what η buys. Staleness is enforced entirely by the
        ``StalenessSampler`` window ``[v-η, v]`` via ``evict``/``select`` (no
        hand-rolled lag check) and there is NO in-flight drain.

        Buffer-capacity slots are released as groups leave the buffer (per evicted
        group and per claimed group) so the rollout pump keeps producing during
        Phase 2; the DataPlane rows stay alive until ``clear_samples`` at step end
        (§7.4), so the frozen π_prox/advantages survive the whole minibatch loop.

        Exhaustion is tail-safe (§7.10): ``select`` returning ``None`` only ends
        the loop when the dispatch loop is done (``_rollout_done``) AND nothing is
        still in flight (``_dispatched_rollouts`` empty) — otherwise the last
        in-flight rollouts would be dropped.

        Returns:
            The merged full-batch ``KVBatchMeta`` (concat of all claimed groups),
            or ``None`` if rollout is exhausted with nothing left to train.
        """
        # __init__ coerces None → min_prompt_groups_per_batch (int); the assert
        # narrows the Optional[int] type for pyrefly.
        assert self._async_cfg.target_prompt_groups_per_step is not None
        target: int = self._async_cfg.target_prompt_groups_per_step
        claimed: list[KVBatchMeta] = []
        n = 0
        while n < target:
            await asyncio.sleep(0)  # yield to _rollout_pump

            # Drop groups that aged past the staleness window [v-η, v]; free
            # their buffer slots so the rollout pump can keep producing.
            evicted = await self._sampler.evict(
                current_train_weight=self._trainer_version,
            )
            for _ in range(evicted):
                self._buffer_capacity.release()

            # Claim in-window groups (lag ≤ η). select() returns (meta|None, num).
            train_meta, num = await self._sampler.select(
                current_train_weight=self._trainer_version,
                min_prompt_groups=self._async_cfg.min_prompt_groups_per_batch,
            )

            if train_meta is None:
                # True exhaustion = dispatch loop done AND nothing still in
                # flight; otherwise the last in-flight rollouts would be dropped.
                if self._rollout_done and not self._dispatched_rollouts:
                    break
                await asyncio.sleep(0.05)
                continue

            # Free buffer slots for the claimed groups (decoupled from the DP-row
            # clear, which happens at step end).
            for _ in range(num):
                self._buffer_capacity.release()
            claimed.append(train_meta)
            n += num

        if not claimed:
            return None
        return claimed[0].concat(*claimed[1:])

    def _iter_minibatches(
        self, batch_meta: "KVBatchMeta"
    ) -> Iterator["KVBatchMeta"]:
        """Yield ``num_minibatches`` contiguous slices of the batch, in order.

        Each yielded meta is ONE optimizer step (one begin/train_microbatch/
        finish cycle in Phase 2). Deterministic, no shuffle — matches AReaL,
        which splits the flat batch into ``ppo_n_minibatches`` in order (single
        pass, each sample used exactly once). The split is even; the leading
        minibatches absorb the remainder. ``num_minibatches=1`` yields the whole
        batch unchanged (one optimizer step per training step).

        Args:
            batch_meta: the frozen full-batch ``KVBatchMeta`` (Phase 1 output).

        Yields:
            Contiguous, non-overlapping ``KVBatchMeta`` slices whose union is the
            full batch. Empty slices (when ``num_minibatches > batch size``) are
            skipped.
        """
        n = batch_meta.size  # total sequences in the batch
        m = self._async_cfg.num_minibatches
        if m < 1:
            raise ValueError(f"num_minibatches must be >= 1, got {m}")
        # Even split; the first (n % m) minibatches each take one extra sample.
        sizes = [n // m + (1 if i < n % m else 0) for i in range(m)]
        start = 0
        for sz in sizes:
            if sz == 0:
                continue
            yield batch_meta.slice(start, start + sz)
            start += sz


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

        # TODO: currently sync_weights is not implemented, comment out for now
        # # Drain: wait for all in-flight rollouts to complete before NCCL
        # # Critical: if GenWorker has queued calls when NCCL init is dispatched,
        # # the init sits behind them — trainer blocks in rendezvous → deadlock
        # drain_start = time.monotonic()
        # while self._inflight_rollouts > 0:
        #     await asyncio.sleep(0.005)

        # drain_elapsed = time.monotonic() - drain_start
        # print(
        #     f"  _sync_weights: drained in {drain_elapsed:.3f}s, "
        #     f"syncing weights v{self._trainer_version}",
        #     flush=True,
        # )

        t0 = time.monotonic()
        await asyncio.to_thread(self._weight_synchronizer.sync_weights)
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        self._rollout_permitted.set()

    async def _advantage_pump(self, meta: KVBatchMeta) -> KVBatchMeta:
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
        self._step_log_dict["rewards"].append(rewards.detach())
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
        if adv_cfg.policy_logprobs_field is not None:
            kwargs["logprobs_policy"] = tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
        if adv_cfg.reference_logprobs_field is not None:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )

        advantages = self._advantage_estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            repeated_batch=repeated_batch,
            **kwargs,
        )
        self._step_log_dict["masked_advantages"].append(
            torch.masked_select(advantages.detach(), mask.bool())
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
        if adv_cfg.policy_logprobs_field is not None:
            fields.append(adv_cfg.policy_logprobs_field)
        if adv_cfg.reference_logprobs_field is not None:
            fields.append(adv_cfg.reference_logprobs_field)
        return list(dict.fromkeys(fields))
