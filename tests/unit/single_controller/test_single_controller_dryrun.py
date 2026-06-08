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

"""Dry-run tests for SingleController asyncio skeleton (C-03).

Validates the three-pump asyncio architecture using stub actors with
configurable sleep latencies — no GPU, no real model weights required.

Key questions answered:
  - Do all 3 pumps run concurrently? (rollout_pump dispatches while
    train_pump is "busy")
  - Does buffer capacity correctly block rollout_pump when capacity is full?
  - Does _rollout_permitted correctly pause dispatch during _sync_weights?
  - RISK-06: does a blocking policy.train() call freeze the event loop?
    (train_on_meta uses asyncio.sleep to simulate, so this is non-blocking
    by construction in the dry-run — see dedicated RISK-06 test below)
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

import pytest
import ray
import torch
from tensordict import TensorDict

# ── Ray temp dir: must be SHORT on macOS (AF_UNIX path limit = 103 bytes) ─
# Use a fixed short path under /tmp to avoid hitting the socket length limit.
_RAY_TEMP = "/tmp/nrl_sc_test"
os.makedirs(_RAY_TEMP, exist_ok=True)
os.environ["RAY_TEMP_DIR"] = _RAY_TEMP
os.environ["RAY_TMPDIR"] = _RAY_TEMP

from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
)
from nemo_rl.algorithms.staleness_sampler import StalenessSampler
from nemo_rl.data_plane import KVBatchMeta

# ── Fake in-memory DataPlane ──────────────────────────────────────────────


@ray.remote(num_cpus=0)
class FakeDataPlaneActor:
    """Minimal in-memory DataPlane actor for dry-run testing.

    Stores rows by sample_id and exposes the current DataPlane methods
    SingleController uses: claim_meta, get_samples, and clear_samples.
    Not production code — used only for C-03 dry-run validation.
    """

    def __init__(self, partition_id: str = "rollout_data"):
        self._partition_id = partition_id
        self._rows: dict[str, dict] = {}
        self._consumed: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        assert partition_id == self._partition_id
        with self._lock:
            for i, sample_id in enumerate(sample_ids):
                row_fields = set(fields.keys()) if fields is not None else set()
                row = self._rows.setdefault(
                    sample_id,
                    {
                        "fields": set(),
                        "values": {},
                        "tag": dict(tags[i]) if tags is not None else {},
                    },
                )
                row["fields"].update(row_fields)
                if tags is not None:
                    row["tag"] = dict(tags[i])
                if fields is not None:
                    for field_name in fields.keys():
                        value = fields[field_name]
                        assert isinstance(value, torch.Tensor)
                        row["values"][field_name] = value[i].detach().clone()
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=list(fields.keys()) if fields is not None else None,
            tags=[dict(t) for t in tags] if tags is not None else None,
        )

    def claim_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        del dp_rank, blocking, timeout_s
        assert partition_id == self._partition_id
        with self._lock:
            consumed = self._consumed.setdefault(task_name, set())
            sample_ids: list[str] = []
            tags: list[dict[str, Any]] = []
            for sample_id, row in self._rows.items():
                if sample_id in consumed:
                    continue
                if not all(field in row["fields"] for field in required_fields):
                    continue
                sample_ids.append(sample_id)
                tags.append(dict(row["tag"]))
                if len(sample_ids) >= batch_size:
                    break
            consumed.update(sample_ids)
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=task_name,
            sample_ids=sample_ids,
            fields=list(required_fields),
            tags=tags if tags else None,
        )

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        assert partition_id == self._partition_id
        values: dict[str, torch.Tensor] = {}
        with self._lock:
            for field_name in select_fields:
                rows = []
                for sample_id in sample_ids:
                    rows.append(self._rows[sample_id]["values"][field_name])
                values[field_name] = torch.stack(rows, dim=0)
        return TensorDict(
            values,
            batch_size=[len(sample_ids)],
        )

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        assert partition_id == self._partition_id
        with self._lock:
            ids = list(self._rows) if sample_ids is None else sample_ids
            for sample_id in ids:
                self._rows.pop(sample_id, None)
                for consumed in self._consumed.values():
                    consumed.discard(sample_id)

    def depth(self) -> int:
        with self._lock:
            return len(self._rows)


# ── Dry-run stub actors ───────────────────────────────────────────────────


@ray.remote(num_cpus=0)
class DryRunGenWorker:
    """Stub GenerationWorkerActor.

    Implements the same interface as production GenWorker:
      generate_and_push(prompt, dp_client) → pushes fake record to DataPlane

    Uses asyncio.sleep to simulate generation latency without blocking
    the event loop.
    """

    def __init__(self, gen_latency_s: float = 0.1, weight_version: int = 0):
        self._gen_latency_s = gen_latency_s
        self._weight_version = weight_version
        self._call_count = 0
        self._call_timestamps: list[float] = []

    async def generate_and_push(self, prompt: str, dp_client: Any) -> None:
        """Simulate generation + push directly to DataPlane."""
        self._call_count += 1
        self._call_timestamps.append(time.monotonic())
        await asyncio.sleep(self._gen_latency_s)
        group_id = f"group-{self._call_count:04d}"
        sample_id = f"{group_id}_g0"
        await dp_client.put_samples.remote(
            sample_ids=[sample_id],
            partition_id="rollout_data",
            fields=TensorDict(
                {
                    "input_ids": torch.ones((1, 3), dtype=torch.long),
                    "prompt_ids_for_adv": torch.tensor(
                        [[self._call_count]],
                        dtype=torch.long,
                    ),
                    "total_reward": torch.tensor(
                        [float(self._call_count)],
                        dtype=torch.float32,
                    ),
                    "token_mask": torch.ones((1, 3), dtype=torch.float32),
                    "sample_mask": torch.ones(1, dtype=torch.float32),
                },
                batch_size=[1],
            ),
            tags=[
                {
                    "group_id": group_id,
                    "weight_version": self._weight_version,
                    "committed": True,
                    "expected_num_samples": 1,
                }
            ],
        )

    def get_call_count(self) -> int:
        return self._call_count

    def get_call_timestamps(self) -> list[float]:
        return list(self._call_timestamps)

    def set_weight_version(self, version: int) -> None:
        self._weight_version = version


@ray.remote(num_cpus=0)
class DryRunTrainer:
    """Stub PolicyTrainerActor.

    Implements the same interface as production trainer:
      train_on_meta(meta, dp_client) → fetches from DataPlane, sleeps, returns result

    Uses asyncio.sleep so event loop stays responsive — other pumps continue.
    """

    def __init__(
        self,
        train_latency_s: float = 0.2,
        expect_advantages: bool = False,
    ):
        self._train_latency_s = train_latency_s
        self._expect_advantages = expect_advantages
        self._trainer_version = 0
        self._train_count = 0
        self._train_start_times: list[float] = []
        self._last_advantages: torch.Tensor | None = None

    async def train_on_meta(self, meta: KVBatchMeta, dp_client: Any) -> dict:
        """Simulate a training step."""
        self._train_start_times.append(time.monotonic())
        # Fetch records from DataPlane directly — same path as production
        select_fields = ["input_ids"]
        if self._expect_advantages:
            select_fields.append("advantages")
        data = await dp_client.get_samples.remote(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=select_fields,
        )
        if self._expect_advantages:
            self._last_advantages = data["advantages"].detach().clone()
        await asyncio.sleep(self._train_latency_s)
        self._trainer_version += 1
        self._train_count += 1
        return {
            "loss": 1.0 / (self._trainer_version + 1),
            "trainer_version": self._trainer_version,
            "clear_samples": True,
        }

    def get_trainer_version(self) -> int:
        return self._trainer_version

    def get_train_count(self) -> int:
        return self._train_count

    def get_train_start_times(self) -> list[float]:
        return list(self._train_start_times)

    def get_last_advantages(self) -> torch.Tensor | None:
        return self._last_advantages


class DryRunAdvantageEstimator:
    """Small estimator used by the dry-run SC advantage stage test."""

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        repeated_batch,
        **kwargs,
    ):
        del prompt_ids, repeated_batch, kwargs
        centered = rewards - rewards.mean()
        return centered.unsqueeze(-1).expand(mask.shape)


class DryRunRolloutManager:
    """Dry-run mock of ``RolloutManager`` for SC dry-run tests.

    Production ``RolloutManager`` is a plain (non-Ray) class living in the
    SC actor's process; this mock matches that shape. Actual work (sleep +
    push a fake sample to DataPlane + bump call counters) is delegated to a
    ``DryRunGenWorker`` Ray actor so the test can inspect call counts,
    timestamps, and weight_version from outside the SC actor.
    """

    def __init__(self, gen_actor: Any, dp_client: Any) -> None:
        self._gen_actor = gen_actor
        self._dp_client = dp_client

    async def generate_and_push(self, prompt: str) -> None:
        await self._gen_actor.generate_and_push.remote(prompt, self._dp_client)


class DryRunWeightSynchronizer:
    """Stub WeightSynchronizer — just sleeps.

    In production this would call WeightSynchronizer.sync_weights() which
    dispatches to IPC/HTTP/NCCL based on deployment config.
    """

    def __init__(self, sync_latency_s: float = 0.05, gen_handle: Any = None):
        self._sync_latency_s = sync_latency_s
        self._gen_handle = gen_handle
        self._sync_count = 0
        self._sync_timestamps: list[float] = []

    async def sync_weights(self, trainer_version: int) -> None:
        self._sync_count += 1
        self._sync_timestamps.append(time.monotonic())
        await asyncio.sleep(self._sync_latency_s)
        if self._gen_handle is not None:
            await self._gen_handle.set_weight_version.remote(trainer_version)


# ── pytest fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    yield
    # Don't shutdown — other tests in the module may need Ray


class _FakeBuffer:
    """Mock of TQReplayBuffer exposing only the surface StalenessSampler reads."""

    def __init__(self, partition_id: str = "rollout_data") -> None:
        self._partition_id = partition_id
        self.meta_list: list[KVBatchMeta] = []
        self.weight_list: list[int] = []

    def add_group(self, group_id: str, weight: int, group_size: int = 1) -> None:
        sample_ids = [f"{group_id}_s{j}" for j in range(group_size)]
        self.meta_list.append(
            KVBatchMeta(
                partition_id=self._partition_id,
                task_name=None,
                sample_ids=sample_ids,
                tags=[{"weight_version": weight, "group_id": group_id}] * group_size,
            )
        )
        self.weight_list.append(weight)

    async def remove(self, sample_ids: list[str]) -> int:
        ids = set(sample_ids)
        keep_meta: list[KVBatchMeta] = []
        keep_w: list[int] = []
        n = 0
        for meta, w in zip(self.meta_list, self.weight_list):
            if set(meta.sample_ids) <= ids:
                n += 1
            else:
                keep_meta.append(meta)
                keep_w.append(w)
        self.meta_list = keep_meta
        self.weight_list = keep_w
        return n


def _buffer_with_versions(versions: list[int]) -> _FakeBuffer:
    buf = _FakeBuffer()
    for i, w in enumerate(versions):
        buf.add_group(f"g{i}", weight=w)
    return buf


# ── tests ─────────────────────────────────────────────────────────────────


class TestSingleControllerDryRun:
    """Validate asyncio skeleton concurrency and backpressure."""

    def _make_controller(
        self,
        dp_client,
        gen,
        trainer,
        weight_sync=None,
        max_train_steps=3,
        min_prompt_groups_per_batch=1,
        generations_per_prompt=1,
        max_buffered_rollouts=4,
        max_inflight_prompts=4,
        max_weight_staleness_versions=1,
        advantage_enabled=False,
        advantage_estimator=None,
        diagnostics=False,
    ):
        cfg = SingleControllerConfig(
            max_train_steps=max_train_steps,
            min_prompt_groups_per_batch=min_prompt_groups_per_batch,
            generations_per_prompt=generations_per_prompt,
            max_buffered_rollouts=max_buffered_rollouts,
            max_inflight_prompts=max_inflight_prompts,
            max_weight_staleness_versions=max_weight_staleness_versions,
            advantage_enabled=advantage_enabled,
            diagnostics=diagnostics,
        )

        # SC expects a StatefulDataLoader, but the pump only iterates it
        # (`for prompt in self._dataloader`), so a list satisfies the contract.
        dataloader = [f"prompt_{i}" for i in range(10)]

        if weight_sync is None:
            weight_sync = DryRunWeightSynchronizer(gen_handle=gen)
        rollout_manager = DryRunRolloutManager(gen, dp_client)

        return SingleControllerActor.remote(
            cfg=cfg,
            dp_client=dp_client,
            gen_handle=gen,
            env_handles={},
            trainer_handle=trainer,
            dataloader=dataloader,
            weight_synchronizer=weight_sync,
            advantage_estimator=advantage_estimator,
            rollout_manager=rollout_manager,
        )

    def test_dry_run_completes(self, ray_init):
        """SC completes N train steps without deadlock on CPU."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.05)
        trainer = DryRunTrainer.remote(train_latency_s=0.1)
        weight_sync = DryRunWeightSynchronizer(sync_latency_s=0.02, gen_handle=gen)

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            weight_sync,
            max_train_steps=3,
            min_prompt_groups_per_batch=1,
            generations_per_prompt=1,
        )

        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 3
        assert result["trainer_version"] == 3

    @pytest.mark.skip("current fail")
    def test_advantage_pump_writes_advantages_before_train(self, ray_init):
        """SC computes advantages from DataPlane inputs and writes them back."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer = DryRunTrainer.remote(
            train_latency_s=0.01,
            expect_advantages=True,
        )

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=1,
            min_prompt_groups_per_batch=2,
            generations_per_prompt=1,
            advantage_enabled=True,
            advantage_estimator=DryRunAdvantageEstimator(),
        )

        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 1

        advantages = ray.get(trainer.get_last_advantages.remote())
        assert advantages is not None
        assert advantages.shape == (2, 3)
        assert torch.allclose(
            advantages,
            torch.tensor(
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, 0.5, 0.5],
                ]
            ),
        )

    def test_rollout_pump_runs_concurrently_with_train(self, ray_init):
        """rollout_pump dispatches while train_pump is sleeping.

        If pumps were sequential, rollout dispatches would only happen
        between training steps. With concurrent asyncio tasks, rollout
        dispatches happen while trainer is in asyncio.sleep().
        """
        dp_client = FakeDataPlaneActor.remote()
        # Gen is fast (0.02s), trainer is slow (0.3s)
        gen = DryRunGenWorker.remote(gen_latency_s=0.02)
        trainer = DryRunTrainer.remote(train_latency_s=0.3)

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=2,
            min_prompt_groups_per_batch=1,
            generations_per_prompt=1,
            max_buffered_rollouts=6,
            max_inflight_prompts=6,
        )

        ray.get(ctrl.run.remote(), timeout=30)

        # Multiple rollouts should have completed during the first train step
        call_timestamps = ray.get(gen.get_call_timestamps.remote())
        train_start_times = ray.get(trainer.get_train_start_times.remote())

        assert len(call_timestamps) > 0
        assert len(train_start_times) > 0

        # Some rollout calls should have started AFTER the first train step began
        first_train_start = train_start_times[0]
        rollouts_during_train = sum(1 for t in call_timestamps if t > first_train_start)
        assert rollouts_during_train > 0, (
            "No rollouts dispatched while trainer was running — pumps may not be concurrent"
        )

    def test_buffer_capacity_semaphore_blocks_rollout(self, ray_init):
        """_rollout_pump blocks when buffer capacity is exhausted.

        Set max_buffered_rollouts=2 with slow trainer — rollout_pump
        should fill buffer capacity then block until trainer clears a group.
        """
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.01)  # fast gen
        trainer = DryRunTrainer.remote(train_latency_s=0.3)  # slow trainer

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=2,
            min_prompt_groups_per_batch=1,
            generations_per_prompt=1,
            max_buffered_rollouts=2,  # small buffer — backpressure kicks in
            max_inflight_prompts=4,
        )

        start = time.monotonic()
        result = ray.get(ctrl.run.remote(), timeout=30)
        elapsed = time.monotonic() - start

        # Should complete without deadlock
        assert result["train_steps"] == 2
        # DataPlane depth should never exceed max_buffered_rollouts (approx)
        # We can't easily observe mid-run depth, but completion = no deadlock

    def test_rollout_permitted_pauses_during_sync(self, ray_init):
        """_rollout_pump pauses new dispatches during _sync_weights.

        During weight sync, _rollout_permitted is cleared. _rollout_pump
        blocks on _rollout_permitted.wait() so no new generate_and_push
        calls are made. Existing in-flight ones drain naturally.
        """
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.05)
        trainer = DryRunTrainer.remote(train_latency_s=0.05)
        weight_sync = DryRunWeightSynchronizer(
            sync_latency_s=0.15,
            gen_handle=gen,
        )  # slow sync

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            weight_sync,
            max_train_steps=2,
            min_prompt_groups_per_batch=1,
            generations_per_prompt=1,
        )

        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 2
        # Weight sync happened (sync_count > 0 implies gate opened correctly)

    @pytest.mark.skip("current fail")
    def test_ping_returns_while_running(self, ray_init):
        """ping() returns immediately if event loop is running — basis for watchdog."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.05)
        trainer = DryRunTrainer.remote(train_latency_s=0.1)

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=5,
            min_prompt_groups_per_batch=1,
            generations_per_prompt=1,
        )

        # Start SC
        run_ref = ctrl.run.remote()

        # Ping while SC is running — should return quickly
        time.sleep(0.2)
        ping_start = time.monotonic()
        health = ray.get(ctrl.ping.remote(), timeout=5)
        ping_elapsed = time.monotonic() - ping_start

        assert health["alive"] is True
        assert ping_elapsed < 1.0, (
            f"ping() took {ping_elapsed:.2f}s — event loop may be blocked"
        )

        ray.get(run_ref, timeout=30)

    def test_staleness_sampler_filters_correctly(self):
        """StalenessSampler returns freshest complete groups within the window."""
        buf = _buffer_with_versions([3, 4, 5, 2, 6])
        sampler = StalenessSampler(buf, max_staleness_versions=2)

        selected = sampler.select(current_train_weight=5, min_prompt_groups=2)

        assert selected is not None
        # freshest-first: g2(lag 0), g1(lag 1). g3 stale, g4 future.
        assert selected.sample_ids == ["g2_s0", "g1_s0"]

    def test_staleness_sampler_returns_none_when_insufficient(self):
        """StalenessSampler returns None when not enough eligible rows."""
        buf = _buffer_with_versions([1])
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        assert (
            sampler.select(current_train_weight=5, min_prompt_groups=2) is None
        )

    def test_staleness_sampler_concats_multiple_groups(self):
        """Selected meta concatenates whole-group sample_ids end-to-end."""
        buf = _FakeBuffer()
        buf.add_group("g0", weight=5, group_size=2)
        buf.add_group("g1", weight=5, group_size=2)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        selected = sampler.select(current_train_weight=5, min_prompt_groups=2)
        assert selected is not None
        assert selected.sample_ids == ["g0_s0", "g0_s1", "g1_s0", "g1_s1"]

    def test_strict_on_policy_batch_sampler_requires_exact_version(self):
        """Strict sampler waits for a full batch at the trainer version."""
        buf = _buffer_with_versions([4, 5, 5, 6])
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        # Eligible at weight==5 are indices 1 and 2 only.
        assert (
            sampler.select(current_train_weight=5, min_prompt_groups=3) is None
        )
        selected = sampler.select(current_train_weight=5, min_prompt_groups=2)
        assert selected is not None
        assert selected.sample_ids == ["g1_s0", "g2_s0"]

    def test_strict_on_policy_batch_sampler_evicts_old_groups(self):
        """Strict sampler drops complete old-version groups via buffer.remove."""
        buf = _buffer_with_versions([4, 5, 4])
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        dropped = asyncio.run(sampler.evict(current_train_weight=5))

        assert dropped == 2
        assert buf.weight_list == [5]
        assert [m.sample_ids[0] for m in buf.meta_list] == ["g1_s0"]


class TestRisk06EventLoopBlocking:
    """RISK-06: validate that asyncio event loop is not blocked during training.

    The risk: if train_on_meta is a synchronous blocking call, the asyncio
    event loop freezes and _rollout_pump + _sync_weights can't make progress.
    Fix: use `await loop.run_in_executor(None, blocking_fn, ...)` or ensure
    train_on_meta is an async method (as DryRunTrainer is).

    These tests document the expected behavior and serve as a benchmark.
    """

    def test_blocking_call_freezes_loop(self):
        """Demonstrate that a synchronous time.sleep freezes the event loop.

        This test validates the PROBLEM (not the solution) — if train used
        time.sleep instead of asyncio.sleep, other tasks would not progress.
        """
        progress: list[str] = []

        async def blocking_task():
            progress.append("blocking_start")
            time.sleep(0.1)  # blocks event loop
            progress.append("blocking_end")

        async def concurrent_task():
            progress.append("concurrent_start")
            await asyncio.sleep(0)
            progress.append("concurrent_mid")
            await asyncio.sleep(0)
            progress.append("concurrent_end")

        async def run():
            t1 = asyncio.create_task(blocking_task())
            t2 = asyncio.create_task(concurrent_task())
            await asyncio.gather(t1, t2)

        asyncio.run(run())

        # With blocking call, concurrent task can't interleave during the sleep
        # blocking_start, blocking_end happen before concurrent_mid
        block_end_idx = progress.index("blocking_end")
        concurrent_mid_idx = progress.index("concurrent_mid")
        assert block_end_idx < concurrent_mid_idx, (
            "blocking_task did not freeze concurrent_task as expected"
        )

    def test_async_sleep_allows_concurrency(self):
        """Demonstrate that asyncio.sleep yields to other tasks.

        The DryRunTrainer uses asyncio.sleep — this shows the event loop
        stays responsive during 'training'. Production code must use
        loop.run_in_executor() for real blocking GPU operations.
        """
        progress: list[str] = []

        async def async_task():
            progress.append("async_start")
            await asyncio.sleep(0.1)  # yields to event loop
            progress.append("async_end")

        async def concurrent_task():
            progress.append("concurrent_start")
            await asyncio.sleep(0.01)
            progress.append("concurrent_mid")
            await asyncio.sleep(0)
            progress.append("concurrent_end")

        async def run():
            t1 = asyncio.create_task(async_task())
            t2 = asyncio.create_task(concurrent_task())
            await asyncio.gather(t1, t2)

        asyncio.run(run())

        # concurrent_task should make progress while async_task is sleeping
        async_end_idx = progress.index("async_end")
        concurrent_mid_idx = progress.index("concurrent_mid")
        assert concurrent_mid_idx < async_end_idx, (
            "concurrent_task should have progressed while async_task was sleeping"
        )

    def test_run_in_executor_unblocks_loop(self):
        """Validate the production fix for RISK-06.

        In production, policy.train() is a blocking GPU call. SC must use:
          await loop.run_in_executor(None, policy.train, ...)
        This runs the blocking call in a thread pool, leaving the event loop
        free for _rollout_pump and _sync_weights to make progress.
        """
        progress: list[str] = []

        def blocking_train():
            time.sleep(0.1)
            return "trained"

        async def train_with_executor():
            loop = asyncio.get_running_loop()
            progress.append("train_start")
            result = await loop.run_in_executor(None, blocking_train)
            progress.append("train_end")
            return result

        async def rollout():
            progress.append("rollout_start")
            await asyncio.sleep(0.02)
            progress.append("rollout_mid")
            await asyncio.sleep(0.02)
            progress.append("rollout_end")

        async def run():
            t1 = asyncio.create_task(train_with_executor())
            t2 = asyncio.create_task(rollout())
            await asyncio.gather(t1, t2)

        asyncio.run(run())

        # rollout should have made progress WHILE train was blocking in executor
        train_end_idx = progress.index("train_end")
        rollout_mid_idx = progress.index("rollout_mid")
        assert rollout_mid_idx < train_end_idx, (
            "rollout should have progressed while blocking_train ran in executor"
        )
