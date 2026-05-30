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

SingleController is a CPU-only Ray actor that owns three concurrent asyncio
pumps and coordinates all other actors via lightweight RPCs. Other actors
expose methods and wait to be called.

Key invariant: SC never sees tensor data. It sends control signals
(``KVBatchMeta`` and actor handles) and reads metadata. All tensor movement
happens directly between actors via the DataPlane path or NCCL.

Data flow:
  _rollout_pump  → gen.generate_and_push(prompt, dp_client) ← RPC to GenWorker
                     GenWorker → dp_client.put_samples(...)
  _train_pump    → dp_client.claim_meta(...) → BatchSelectionStrategy
                 → trainer.train_on_meta(meta)
                     Trainer → dp_client.get_samples(...)
                 → dp_client.clear_samples(...)             ← SC clears after train
  _sync_weights  → drain _inflight_rollouts → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import ray

from nemo_rl.algorithms.staleness_sampler import (
    BatchSelectionStrategy,
    StalenessWindowSelection,
    StrictOnPolicyBatchSampler,
    count_prompt_groups,
)
from nemo_rl.data_plane import KVBatchMeta

log = logging.getLogger(__name__)


@dataclass
class SingleControllerConfig:
    """Configuration for SingleController."""

    # Staleness
    max_weight_staleness_versions: int = 1
    min_prompt_groups_per_batch: int = 2
    generations_per_prompt: int = 4
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

    # DataPlane partition
    partition_id: str = "rollout_data"
    consumer_task_name: str = "train"
    claim_required_fields: list[str] = field(default_factory=lambda: ["input_ids"])
    max_claim_prompt_groups: int = 8

    # Diagnostics
    diagnostics: bool = False

    # Weight transport backend ("stub" for dry-run, "nccl" for production)
    weight_transport: str = "stub"
    weight_nccl_addr: str = "127.0.0.1"
    weight_nccl_port: Optional[int] = None

    # Extra fields passed through to avoid TypedDict issues
    extra: dict = field(default_factory=dict)


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns three concurrent asyncio tasks:
      - _rollout_pump: dispatches prompts to GenerationWorkerActor
      - _train_pump:   claims DataPlane meta, trains, clears consumed rows
      - _sync_weights: drain gate + weight synchronization

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
    ) -> None:
        import logging as _logging

        _logging.basicConfig(
            level=_logging.INFO,
            format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        )

        self._cfg = cfg
        self._prompts = prompts
        self._dp_client = dp_client_handle
        self._gen = gen_handle
        self._trainer = trainer_handle
        self._weight_synchronizer = weight_synchronizer

        if cfg.batch_selection_strategy == "strict_on_policy":
            self._sampler: BatchSelectionStrategy = StrictOnPolicyBatchSampler()
        elif cfg.batch_selection_strategy == "staleness_window":
            self._sampler = StalenessWindowSelection(cfg.max_weight_staleness_versions)
        else:
            raise ValueError(
                f"Unknown batch_selection_strategy: {cfg.batch_selection_strategy}"
            )

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
        self._claimed_meta: KVBatchMeta | None = None

        log.info(
            "SingleControllerActor: staleness_cap=%d buffer=%d inflight=%d transport=%s",
            cfg.max_weight_staleness_versions,
            cfg.max_buffered_rollouts,
            cfg.max_inflight_prompts,
            cfg.weight_transport,
        )

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

    # ── the three pumps ────────────────────────────────────────────────────

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
        sem = asyncio.Semaphore(self._cfg.max_inflight_prompts)

        start = time.monotonic()
        log.info("rollout_pump: dispatching %d prompts", n)

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
                        log.info("  rollout done for prompt='%s...'", prompt[:20])
                finally:
                    self._inflight_rollouts -= 1

        tasks = [
            asyncio.ensure_future(
                _one_group(self._prompts[i % len(self._prompts)])
            )
            for i in range(n)
        ]
        await asyncio.gather(*tasks)

        self._rollout_done = True
        log.info(
            "rollout_pump: finished %d prompts in %.2fs",
            n,
            time.monotonic() - start,
        )

    async def _train_pump(self) -> None:
        """Claim DataPlane metadata, select a batch, train, clear, sync weights.

        Flow per step:
          1. claim_meta() — temporary consuming metadata acquisition
          2. Evict sample_ids no longer eligible under the selected policy
          3. BatchSelectionStrategy.select_indices() — choose full prompt groups
          4. trainer.train_on_meta(KVBatchMeta, dp_client) — trainer fetches tensors
          5. clear_samples(sample_ids) — SC removes consumed rows
          6. _buffer_capacity.release() — unblock _rollout_pump
          7. _sync_weights()
        """
        while self._train_steps < self._cfg.max_train_steps:
            await self._claim_available_meta()

            if self._claimed_meta is None or self._claimed_meta.size == 0:
                if self._rollout_done:
                    log.info("train_pump: rollout done and buffer drained, exiting")
                    break
                await asyncio.sleep(0.05)
                continue

            evicted_meta = await self._evict_stale_claimed()
            if evicted_meta is not None:
                evicted_groups = count_prompt_groups(
                    evicted_meta,
                    generations_per_prompt=self._cfg.generations_per_prompt,
                )
                for _ in range(evicted_groups):
                    self._buffer_capacity.release()

            if self._claimed_meta is None or self._claimed_meta.size == 0:
                continue

            selected_indices = self._sampler.select_indices(
                self._claimed_meta,
                trainer_version=self._trainer_version,
                min_prompt_groups=self._cfg.min_prompt_groups_per_batch,
                generations_per_prompt=self._cfg.generations_per_prompt,
            )

            if selected_indices is None:
                if (
                    self._rollout_done
                    and count_prompt_groups(
                        self._claimed_meta,
                        generations_per_prompt=self._cfg.generations_per_prompt,
                    )
                    < self._cfg.min_prompt_groups_per_batch
                ):
                    log.info("train_pump: rollout done and no full batch remains")
                    break
                await asyncio.sleep(0.05)
                continue

            selected_meta = self._claimed_meta.subset(selected_indices)
            selected_weight = _min_weight_version(selected_meta)
            lag = (
                self._trainer_version - selected_weight
                if selected_weight is not None
                else 0
            )

            log.info(
                "train step %d/%d  trainer_v=%d  lag=%d  batch_size=%d",
                self._train_steps + 1,
                self._cfg.max_train_steps,
                self._trainer_version,
                lag,
                selected_meta.size,
            )

            t0 = time.perf_counter()
            result = await self._ray_get(
                self._trainer.train_on_meta.remote(selected_meta, self._dp_client)
            )
            elapsed = time.perf_counter() - t0

            if elapsed > 5.0:
                log.info("  train_on_meta returned in %.1fs", elapsed)

            if result.get("clear_samples", True):
                await self._call_dp(
                    "clear_samples",
                    sample_ids=selected_meta.sample_ids,
                    partition_id=selected_meta.partition_id,
                )
                self._claimed_meta = _drop_indices(
                    self._claimed_meta, selected_indices
                )
                self._trainer_version = result["trainer_version"]
                selected_groups = count_prompt_groups(
                    selected_meta,
                    generations_per_prompt=self._cfg.generations_per_prompt,
                )
                for _ in range(selected_groups):
                    self._buffer_capacity.release()

                await self._sync_weights()

                log.info(
                    "  → done  loss=%.6g  trainer_version=%d",
                    result["loss"],
                    self._trainer_version,
                )
                self._train_steps += 1

    async def _claim_available_meta(self) -> None:
        """Claim currently-ready rows and append them to the local scheduler cache.

        TODO: replace this with a non-consuming metadata listing API.
        ``claim_meta`` advances TQ's per-task cursor, so SC must keep a
        local cache of claimed-but-not-yet-trained samples for now.
        """
        batch_size = (
            self._cfg.max_claim_prompt_groups * self._cfg.generations_per_prompt
        )
        meta = await self._call_dp(
            "claim_meta",
            partition_id=self._cfg.partition_id,
            task_name=self._cfg.consumer_task_name,
            required_fields=list(self._cfg.claim_required_fields),
            batch_size=batch_size,
            blocking=False,
            timeout_s=0.0,
        )
        if meta.size == 0:
            return
        self._claimed_meta = _concat_meta(self._claimed_meta, meta)

    async def _evict_stale_claimed(self) -> KVBatchMeta | None:
        if self._claimed_meta is None or self._claimed_meta.size == 0:
            return None
        indices = self._sampler.evictable_indices(
            self._claimed_meta,
            trainer_version=self._trainer_version,
            generations_per_prompt=self._cfg.generations_per_prompt,
        )
        if not indices:
            return None
        evicted_meta = self._claimed_meta.subset(indices)
        log.info(
            "  evicting %d stale samples from %d prompt group(s)",
            evicted_meta.size,
            count_prompt_groups(
                evicted_meta,
                generations_per_prompt=self._cfg.generations_per_prompt,
            ),
        )
        await self._call_dp(
            "clear_samples",
            sample_ids=evicted_meta.sample_ids,
            partition_id=evicted_meta.partition_id,
        )
        self._claimed_meta = _drop_indices(self._claimed_meta, indices)
        return evicted_meta

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
        log.info(
            "  _sync_weights: drained in %.3fs, syncing weights v%d",
            drain_elapsed,
            self._trainer_version,
        )

        t0 = time.monotonic()
        await self._weight_synchronizer.sync_weights(self._trainer_version)
        elapsed = time.monotonic() - t0

        log.info("  _sync_weights: sync done in %.3fs", elapsed)
        self._rollout_permitted.set()


def _concat_meta(left: KVBatchMeta | None, right: KVBatchMeta) -> KVBatchMeta:
    if left is None or left.size == 0:
        return right
    return left.concat(right)


def _drop_indices(meta: KVBatchMeta, indices: list[int]) -> KVBatchMeta | None:
    dropped = set(indices)
    keep = [i for i in range(meta.size) if i not in dropped]
    if not keep:
        return None
    return meta.subset(keep)


def _min_weight_version(meta: KVBatchMeta) -> int | None:
    versions = []
    for tag in meta.tags or []:
        value = tag.get("weight_version", tag.get("version"))
        if value is None:
            continue
        versions.append(int(value))
    return min(versions) if versions else None
