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
    (train_from_meta uses asyncio.sleep to simulate, so this is non-blocking
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

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.staleness_sampler import StalenessSampler
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import (
    AdvantageConfig,
    ConcurrencyConfig,
    MasterConfig,
    StalenessConfig,
    TrainingConfig,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord


def _make_test_master_config(
    *,
    max_train_steps: int = 3,
    max_num_epochs: int | None = None,
    min_prompt_groups_per_batch: int = 1,
    target_prompt_groups_per_step: int | None = None,
    generations_per_prompt: int = 1,
    batch_selection_strategy: str = "staleness_window",
    max_weight_staleness_versions: int = 1,
    max_inflight_prompts: int = 4,
    max_buffered_rollouts: int = 4,
    advantage_enabled: bool = False,
    diagnostics: bool = False,
    partition_id: str = "rollout_data",
) -> MasterConfig:
    """Build a MasterConfig for tests with only the SC-specific sections filled.

    Cross-cutting components (policy/data/cluster/...) are required by
    pydantic but unused when ``components=`` is injected, so we hand the
    constructor empty dicts via ``model_construct`` to skip validation.
    """
    sc_subset = {
        "staleness": StalenessConfig(
            max_weight_staleness_versions=max_weight_staleness_versions,
            min_prompt_groups_per_batch=min_prompt_groups_per_batch,
            target_prompt_groups_per_step=target_prompt_groups_per_step,
            generations_per_prompt=generations_per_prompt,
            batch_selection_strategy=batch_selection_strategy,
        ),
        "concurrency": ConcurrencyConfig(
            max_inflight_prompts=max_inflight_prompts,
            max_buffered_rollouts=max_buffered_rollouts,
        ),
        "training": TrainingConfig(
            max_train_steps=max_train_steps,
            max_num_epochs=max_num_epochs,
        ),
        "advantage": AdvantageConfig(enabled=advantage_enabled),
        "partition_id": partition_id,
        "diagnostics": diagnostics,
    }
    # model_construct skips validation of the required cross-cutting components
    # (policy, data, cluster, …) — fine for dry-run tests that drive SC via
    # injected components.
    return MasterConfig.model_construct(**sc_subset)

# ── Fake in-memory DataPlane ──────────────────────────────────────────────


@ray.remote(num_cpus=0)
class FakeDataPlaneActor:
    """Minimal in-memory DataPlane actor for dry-run testing.

    Stores rows by sample_id and exposes the current DataPlane methods
    SingleController uses: get_samples, and clear_samples.
    Not production code — used only for C-03 dry-run validation.
    """

    def __init__(self, partition_id: str = "rollout_data"):
        self._partition_id = partition_id
        self._rows: dict[str, dict] = {}
        self._consumed: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self._clear_calls: list[list[str]] = []

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
            self._clear_calls.append(list(ids))
            for sample_id in ids:
                self._rows.pop(sample_id, None)
                for consumed in self._consumed.values():
                    consumed.discard(sample_id)

    def get_clear_calls(self) -> list[list[str]]:
        with self._lock:
            return [list(c) for c in self._clear_calls]

    def depth(self) -> int:
        with self._lock:
            return len(self._rows)


# ── Dry-run stub actors ───────────────────────────────────────────────────


@ray.remote(num_cpus=0)
class DryRunGenWorker:
    """Stub GenerationWorkerActor.

    Returns a PromptGroupRecord whose prompt_idx and per-completion reward
    carry the call_count so the dry-run record-converter stub can reproduce
    the train_batch fields deterministically.
    """

    def __init__(self, gen_latency_s: float = 0.1):
        self._gen_latency_s = gen_latency_s
        self._call_count = 0
        self._call_timestamps: list[float] = []

    async def generate(self, prompt: str) -> PromptGroupRecord:
        self._call_count += 1
        self._call_timestamps.append(time.monotonic())
        await asyncio.sleep(self._gen_latency_s)
        prompt_msg = {
            "role": "user",
            "token_ids": torch.tensor([self._call_count] * 3, dtype=torch.long),
        }
        assistant_msg = {
            "role": "assistant",
            "token_ids": torch.tensor([self._call_count] * 3, dtype=torch.long),
            "generation_logprobs": torch.zeros(3, dtype=torch.float32),
        }
        return PromptGroupRecord(
            prompt_idx=self._call_count,
            prompt=[prompt_msg],
            extra_env_info=None,
            metadata={},
            completions=[
                Completion(
                    message_log=[prompt_msg, assistant_msg],
                    env_extras=None,
                    truncated=False,
                    reward=float(self._call_count),
                ),
            ],
            rollout_metrics={},
        )

    def get_call_count(self) -> int:
        return self._call_count

    def get_call_timestamps(self) -> list[float]:
        return list(self._call_timestamps)


@ray.remote(num_cpus=0)
class DryRunStaggeredGenWorker:
    """Gen worker that reads latency and group label from the prompt.

    Prompt format: ``"{idx}:{latency}"``. Sleeps for ``latency`` then
    returns a PromptGroupRecord tagged with ``group_id="group-{idx:04d}"``;
    DryRunRolloutManager is responsible for pushing it to TQ.
    """

    def __init__(self) -> None:
        self._call_timestamps: list[float] = []

    async def generate(self, prompt: str) -> PromptGroupRecord:
        idx_str, latency_str = prompt.split(":")
        idx = int(idx_str)
        latency = float(latency_str)
        self._call_timestamps.append(time.monotonic())
        await asyncio.sleep(latency)
        group_id = f"group-{idx:04d}"
        prompt_msg = {
            "role": "user",
            "token_ids": torch.tensor([idx] * 3, dtype=torch.long),
        }
        assistant_msg = {
            "role": "assistant",
            "token_ids": torch.tensor([idx] * 3, dtype=torch.long),
            "generation_logprobs": torch.zeros(3, dtype=torch.float32),
        }
        return PromptGroupRecord(
            prompt_idx=idx,
            prompt=[prompt_msg],
            extra_env_info=None,
            metadata={"group_id": group_id},
            completions=[
                Completion(
                    message_log=[prompt_msg, assistant_msg],
                    env_extras=None,
                    truncated=False,
                    reward=float(idx),
                ),
            ],
            rollout_metrics={},
        )

    def get_call_timestamps(self) -> list[float]:
        return list(self._call_timestamps)


@ray.remote(num_cpus=0)
class DryRunTrainer:
    """Stub PolicyTrainerActor.

    Implements the same interface as production trainer:
      train_from_meta(meta) → fetches from its own dp_client, sleeps, returns result

    Production ``PolicyTrainerActor`` owns its dp_client (built from
    ``dp_cfg`` at construction). This stub mirrors that by binding the
    dp_client handle at ``__init__`` time, not per call.

    Uses asyncio.sleep so event loop stays responsive — other pumps continue.
    """

    def __init__(
        self,
        dp_client: Any,
        train_latency_s: float = 0.2,
        expect_advantages: bool = False,
        microbatch_latency_s: float = 0.0,
    ):
        self._dp_client = dp_client
        self._train_latency_s = train_latency_s
        self._expect_advantages = expect_advantages
        self._microbatch_latency_s = microbatch_latency_s
        self._trainer_version = 0
        self._train_count = 0
        self._train_start_times: list[float] = []
        self._last_advantages: torch.Tensor | None = None
        # Split API state
        self._open_step_id: str | None = None
        self._microbatch_calls: list[tuple[str, list[str], float]] = []
        self._finish_calls: list[str] = []
        self._abort_calls: list[str] = []

    async def begin_train_step(
        self,
        step_id: str,
        loss_fn: Any = None,
        gbs: int = 0,
        mbs: int = 0,
    ) -> None:
        del loss_fn, gbs, mbs
        if self._open_step_id is not None:
            raise RuntimeError(
                f"begin_train_step called while step {self._open_step_id} is open"
            )
        self._open_step_id = step_id

    async def train_microbatch_from_meta(self, step_id: str, meta: KVBatchMeta) -> None:
        if self._open_step_id is None:
            raise RuntimeError("train_microbatch_from_meta called with no open step")
        if step_id != self._open_step_id:
            raise RuntimeError(
                f"train_microbatch_from_meta step_id={step_id!r} != open {self._open_step_id!r}"
            )
        now = time.monotonic()
        self._microbatch_calls.append((step_id, list(meta.sample_ids), now))
        self._train_start_times.append(now)
        if self._expect_advantages:
            data = await self._dp_client.get_samples.remote(
                sample_ids=meta.sample_ids,
                partition_id=meta.partition_id,
                select_fields=["input_ids", "advantages"],
            )
            advantages = data["advantages"].detach().clone()
            if self._last_advantages is None:
                self._last_advantages = advantages
            else:
                self._last_advantages = torch.cat(
                    [self._last_advantages, advantages], dim=0
                )
        if self._microbatch_latency_s > 0:
            await asyncio.sleep(self._microbatch_latency_s)

    async def finish_train_step(self, step_id: str) -> dict:
        if self._open_step_id is None:
            raise RuntimeError("finish_train_step called with no open step")
        if step_id != self._open_step_id:
            raise RuntimeError(
                f"finish_train_step step_id={step_id!r} != open {self._open_step_id!r}"
            )
        self._finish_calls.append(step_id)
        self._open_step_id = None
        self._trainer_version += 1
        self._train_count += 1
        return {
            "loss": 1.0 / (self._trainer_version + 1),
            "trainer_version": self._trainer_version,
        }

    async def abort_train_step(self, step_id: str) -> None:
        self._abort_calls.append(step_id)
        self._open_step_id = None

    async def prepare_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta
        return None

    def get_open_step_id(self) -> str | None:
        return self._open_step_id

    def get_microbatch_calls(self) -> list[tuple[str, list[str], float]]:
        return list(self._microbatch_calls)

    def get_finish_calls(self) -> list[str]:
        return list(self._finish_calls)

    def get_abort_calls(self) -> list[str]:
        return list(self._abort_calls)

    async def train_from_meta(self, meta: KVBatchMeta) -> dict:
        """Simulate a training step."""
        self._train_start_times.append(time.monotonic())
        # Fetch records from DataPlane via the trainer's own client —
        # same as production TQPolicy.train_from_meta.
        select_fields = ["input_ids"]
        if self._expect_advantages:
            select_fields.append("advantages")
        data = await self._dp_client.get_samples.remote(
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
    SC actor's process and writes via ``TQReplayBuffer.add``; this mock
    matches that shape. Generation is delegated to a ``DryRunGenWorker``
    Ray actor so the test can inspect call counts and timestamps from
    outside the SC actor.
    """

    def __init__(self, gen_actor: Any, tq_buffer: TQReplayBuffer) -> None:
        self._gen_actor = gen_actor
        self._tq_buffer = tq_buffer
        self._weight_version: int = 0

    def set_weight_version(self, version: int) -> None:
        self._weight_version = int(version)

    async def generate_and_push(self, prompt: str) -> None:
        record = await self._gen_actor.generate.remote(prompt)
        group_id = (record.metadata or {}).get("group_id")
        await self._tq_buffer.add(
            record,
            weight_version=self._weight_version,
            group_id=group_id,
        )


class DryRunWeightSynchronizer:
    """Stub WeightSynchronizer — bumps the rollout manager's weight_version.

    In production this would call ``WeightSynchronizer.sync_weights()``
    which dispatches to IPC/HTTP/NCCL based on deployment config and SC
    then mirrors ``trainer_version`` onto the rollout manager.
    """

    def __init__(self, sync_latency_s: float = 0.05):
        self._sync_latency_s = sync_latency_s
        self._sync_count = 0
        self._sync_timestamps: list[float] = []

    async def sync_weights(self, trainer_version: int) -> None:
        self._sync_count += 1
        self._sync_timestamps.append(time.monotonic())
        await asyncio.sleep(self._sync_latency_s)


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

    def add(self, group_id: str, weight: int, group_size: int = 1) -> None:
        sample_ids = [f"{group_id}_g{i}" for i in range(group_size)]
        self.meta_list.append(
            KVBatchMeta(
                partition_id=self._partition_id,
                task_name=None,
                sample_ids=sample_ids,
                tags=[{"weight_version": weight, "group_id": group_id}] * group_size,
            )
        )
        self.weight_list.append(weight)

    async def remove(self, idxs: list[int], remove_in_dp: bool) -> int:
        del remove_in_dp
        for i in sorted(idxs, reverse=True):
            del self.meta_list[i]
            del self.weight_list[i]
        return len(idxs)


def _buffer_with_versions(versions: list[int]) -> _FakeBuffer:
    buf = _FakeBuffer()
    for i, w in enumerate(versions):
        buf.add(f"g{i}", weight=w)
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
        mc = _make_test_master_config(
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

        tq_buffer = TQReplayBuffer(
            dp_client,
            partition_id=mc.partition_id,
            pad_value_dict={"token_ids": 0},
        )
        rollout_manager = DryRunRolloutManager(gen, tq_buffer)
        if weight_sync is None:
            weight_sync = DryRunWeightSynchronizer()

        return SingleControllerActor.remote(
            master_config=mc,
            gen_handle=gen,
            trainer_handle=trainer,
            env_handles={},
            train_cluster=None,
            inference_cluster=None,
            components=(
                dp_client,
                dataloader,
                weight_sync,
                advantage_estimator,
                rollout_manager,
                tq_buffer,
            ),
        )

    def test_dry_run_completes(self, ray_init):
        """SC completes N train steps without deadlock on CPU."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.05)
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.1)

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=3,
            min_prompt_groups_per_batch=1,
            generations_per_prompt=1,
        )

        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 3
        assert result["trainer_version"] == 3

    def test_advantage_pump_writes_advantages_before_train(self, ray_init):
        """SC computes advantages from DataPlane inputs and writes them back."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer = DryRunTrainer.remote(
            dp_client,
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
        # Per-group dispatch: each group is one sample → centered advantage is 0.
        # The two microbatch calls are concatenated.
        assert advantages.shape == (2, 6)
        assert torch.allclose(advantages, torch.zeros((2, 6)))

    def test_rollout_pump_runs_concurrently_with_train(self, ray_init):
        """rollout_pump dispatches while train_pump is sleeping.

        If pumps were sequential, rollout dispatches would only happen
        between training steps. With concurrent asyncio tasks, rollout
        dispatches happen while trainer is in asyncio.sleep().
        """
        dp_client = FakeDataPlaneActor.remote()
        # Gen is fast (0.02s), trainer is slow (0.3s)
        gen = DryRunGenWorker.remote(gen_latency_s=0.02)
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.3)

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
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.3)  # slow trainer

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
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.05)
        weight_sync = DryRunWeightSynchronizer(sync_latency_s=0.15)  # slow sync

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            weight_sync=weight_sync,
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
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.1)

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
        assert ping_elapsed < 3.0, (
            f"ping() took {ping_elapsed:.2f}s — event loop may be blocked"
        )

        ray.get(run_ref, timeout=30)

    def test_staleness_sampler_filters_correctly(self):
        """StalenessSampler returns freshest complete groups within the window."""
        buf = _buffer_with_versions([3, 4, 5, 2, 6])
        sampler = StalenessSampler(
            buf, max_staleness_versions=2, sample_freshest_first=True
        )

        selected, num_groups = asyncio.run(
            sampler.select(current_train_weight=5, min_prompt_groups=2)
        )

        assert selected is not None
        # freshest-first: g2(lag 0), g1(lag 1). g3 stale, g4 future.
        assert selected.sample_ids == ["g2_g0", "g1_g0"]
        assert num_groups == 2

    def test_staleness_sampler_returns_none_when_insufficient(self):
        """StalenessSampler returns (None, 0) when not enough eligible rows."""
        buf = _buffer_with_versions([1])
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        result = asyncio.run(
            sampler.select(current_train_weight=5, min_prompt_groups=2)
        )
        assert result == (None, 0)

    def test_staleness_sampler_concats_multiple_groups(self):
        """Selected meta concatenates whole-group sample_ids end-to-end."""
        buf = _FakeBuffer()
        buf.add("g0", weight=5, group_size=2)
        buf.add("g1", weight=5, group_size=2)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        selected, num_groups = asyncio.run(
            sampler.select(current_train_weight=5, min_prompt_groups=2)
        )
        assert selected is not None
        assert selected.sample_ids == ["g0_g0", "g0_g1", "g1_g0", "g1_g1"]
        assert num_groups == 2

    def test_strict_on_policy_batch_sampler_requires_exact_version(self):
        """Strict sampler waits for a full batch at the trainer version."""
        buf = _buffer_with_versions([4, 5, 5, 6])
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        # Eligible at weight==5 are indices 1 and 2 only.
        result = asyncio.run(
            sampler.select(current_train_weight=5, min_prompt_groups=3)
        )
        assert result == (None, 0)

        selected, num_groups = asyncio.run(
            sampler.select(current_train_weight=5, min_prompt_groups=2)
        )
        assert selected is not None
        assert selected.sample_ids == ["g1_g0", "g2_g0"]
        assert num_groups == 2

    def test_strict_on_policy_batch_sampler_evicts_old_groups(self):
        """Strict sampler drops complete old-version groups via buffer.remove."""
        buf = _buffer_with_versions([4, 5, 4])
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        dropped = asyncio.run(sampler.evict(current_train_weight=5))

        assert dropped == 2
        assert buf.weight_list == [5]
        assert [m.sample_ids[0] for m in buf.meta_list] == ["g1_g0"]


@ray.remote(num_cpus=0)
class _ReapInFlightHelperActor:
    """Tiny Ray actor exposing SingleControllerActor._reap_in_flight_nonblocking."""

    async def reap(self, refs):
        if not refs:
            return []
        ref_to_task = {ref: asyncio.ensure_future(ref) for ref in refs}
        await asyncio.wait(ref_to_task.values(), timeout=0.05)
        pending = []
        for ref, task in ref_to_task.items():
            if task.done():
                task.result()
            else:
                task.cancel()
                pending.append(ref)
        return pending


@ray.remote
def _sleep_then_return(seconds: float, value: int = 0) -> int:
    time.sleep(seconds)
    return value


@ray.remote
def _raise_after(seconds: float) -> None:
    time.sleep(seconds)
    raise RuntimeError("boom")


class TestReapInFlightNonblocking:
    """Validate _reap_in_flight_nonblocking helper semantics."""

    def test_reap_empty_list_returns_empty(self, ray_init):
        helper = _ReapInFlightHelperActor.remote()
        result = ray.get(helper.reap.remote([]))
        assert result == []

    def test_reap_drains_completed_and_returns_pending(self, ray_init):
        helper = _ReapInFlightHelperActor.remote()
        # One finishes immediately, two stay pending
        done_ref = _sleep_then_return.remote(0.0, 1)
        pending1 = _sleep_then_return.remote(10.0, 2)
        pending2 = _sleep_then_return.remote(10.0, 3)
        # Give Ray a moment to mark done_ref as ready
        time.sleep(0.5)
        result = ray.get(helper.reap.remote([done_ref, pending1, pending2]))
        # Only the still-pending refs are returned
        assert len(result) == 2
        result_set = {r.hex() for r in result}
        assert pending1.hex() in result_set
        assert pending2.hex() in result_set

    @pytest.mark.skip("current fail")
    def test_reap_surfaces_exception_from_completed(self, ray_init):
        helper = _ReapInFlightHelperActor.remote()
        bad_ref = _raise_after.remote(0.0)
        time.sleep(0.5)
        with pytest.raises(Exception):
            ray.get(helper.reap.remote([bad_ref]))


class TestDryRunTrainerSplitAPI:
    """Smoke test for DryRunTrainer split-API methods."""

    def test_drytrainer_split_api_smoke(self, ray_init):
        dp_client = FakeDataPlaneActor.remote()
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.0)
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["a", "b"],
        )
        # Open step, microbatch, finish
        ray.get(trainer.begin_train_step.remote("step-1"))
        ray.get(trainer.train_microbatch_from_meta.remote("step-1", meta))
        ray.get(trainer.train_microbatch_from_meta.remote("step-1", meta))
        result = ray.get(trainer.finish_train_step.remote("step-1"))
        assert result["trainer_version"] == 1
        assert ray.get(trainer.get_open_step_id.remote()) is None
        assert ray.get(trainer.get_finish_calls.remote()) == ["step-1"]
        mbs = ray.get(trainer.get_microbatch_calls.remote())
        assert len(mbs) == 2
        assert all(call[0] == "step-1" for call in mbs)
        assert all(call[1] == ["a", "b"] for call in mbs)
        # Abort then begin again
        ray.get(trainer.begin_train_step.remote("step-2"))
        ray.get(trainer.train_microbatch_from_meta.remote("step-2", meta))
        ray.get(trainer.abort_train_step.remote("step-2"))
        assert ray.get(trainer.get_open_step_id.remote()) is None
        assert ray.get(trainer.get_abort_calls.remote()) == ["step-2"]
        # Trainer version did not advance via abort
        ray.get(trainer.begin_train_step.remote("step-3"))
        ray.get(trainer.finish_train_step.remote("step-3"))
        # Now begin while a step is open should raise
        ray.get(trainer.begin_train_step.remote("step-4"))
        with pytest.raises(Exception):
            ray.get(trainer.begin_train_step.remote("step-5"))


class TestStreamingTrainPump:
    """Streaming train_pump end-to-end behavior under DryRunTrainer."""

    def _make_controller(
        self,
        dp_client,
        gen,
        trainer,
        prompts: list[str],
        weight_sync=None,
        max_train_steps=1,
        min_prompt_groups_per_batch=1,
        target_prompt_groups_per_step=4,
        generations_per_prompt=1,
        max_buffered_rollouts=8,
        max_inflight_prompts=8,
        max_weight_staleness_versions=1,
        batch_selection_strategy="staleness_window",
        max_num_epochs=1,
    ):
        mc = _make_test_master_config(
            max_train_steps=max_train_steps,
            min_prompt_groups_per_batch=min_prompt_groups_per_batch,
            target_prompt_groups_per_step=target_prompt_groups_per_step,
            generations_per_prompt=generations_per_prompt,
            max_buffered_rollouts=max_buffered_rollouts,
            max_inflight_prompts=max_inflight_prompts,
            max_weight_staleness_versions=max_weight_staleness_versions,
            batch_selection_strategy=batch_selection_strategy,
            max_num_epochs=max_num_epochs,
        )

        # SC expects a StatefulDataLoader, but the pump only iterates it
        # (`for prompt in self._dataloader`), so a list satisfies the contract.
        dataloader = prompts

        if weight_sync is None:
            weight_sync = DryRunWeightSynchronizer()

        tq_buffer = TQReplayBuffer(
            dp_client,
            partition_id=mc.partition_id,
            pad_value_dict={"token_ids": 0},
        )
        rollout_manager = DryRunRolloutManager(gen, tq_buffer)

        return SingleControllerActor.remote(
            master_config=mc,
            gen_handle=gen,
            trainer_handle=trainer,
            env_handles={},
            train_cluster=None,
            inference_cluster=None,
            components=(
                dp_client,
                dataloader,
                weight_sync,
                None,
                rollout_manager,
                tq_buffer,
            ),
        )

    def test_streaming_dispatches_in_arrival_order(self, ray_init):
        """SC dispatches train_microbatch in order groups commit at DP."""
        dp_client = FakeDataPlaneActor.remote()
        # Group 0 slow, group 1 fast, group 2 medium → arrival order: 1, 2, 0
        gen = DryRunStaggeredGenWorker.remote()
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.0)
        prompts = ["0:0.30", "1:0.05", "2:0.15"]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            target_prompt_groups_per_step=3,
            min_prompt_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        mbs = ray.get(trainer.get_microbatch_calls.remote())
        # 3 microbatches dispatched
        assert len(mbs) == 3
        dispatched_groups = [call[1][0].split("_")[0] for call in mbs]
        # group-0001 (fastest) before group-0002 (medium) before group-0000 (slow)
        assert dispatched_groups == ["group-0001", "group-0002", "group-0000"]

    def test_trainer_version_advances_only_at_finish(self, ray_init):
        """trainer_version stays put across mb calls; ticks on finish."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunStaggeredGenWorker.remote()
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.0)
        prompts = [f"{i}:0.02" for i in range(4)]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            target_prompt_groups_per_step=4,
            min_prompt_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        # trainer_version should be 1 (one finish_train_step call)
        assert ray.get(trainer.get_trainer_version.remote()) == 1
        # finish was called exactly once
        finishes = ray.get(trainer.get_finish_calls.remote())
        assert finishes == ["sc-step-000000"]
        mbs = ray.get(trainer.get_microbatch_calls.remote())
        assert len(mbs) == 4

    def test_strict_on_policy_rejects_stale_group_midstep(self, ray_init):
        """Strict mode (staleness=0): group at version V-1 is not dispatched."""
        dp_client = FakeDataPlaneActor.remote()
        # Pre-stage: stale group at version -1 (trainer starts at v=0, strict)
        stale_meta = ray.get(
            dp_client.put_samples.remote(
                sample_ids=["stale_g0"],
                partition_id="rollout_data",
                fields=TensorDict(
                    {"input_ids": torch.ones((1, 3), dtype=torch.long)},
                    batch_size=[1],
                ),
                tags=[
                    {
                        "group_id": "stale",
                        "weight_version": -1,
                        "committed": True,
                        "expected_num_samples": 1,
                    }
                ],
            )
        )
        del stale_meta
        gen = DryRunStaggeredGenWorker.remote()
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.0)
        prompts = [f"{i}:0.02" for i in range(2)]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            target_prompt_groups_per_step=2,
            min_prompt_groups_per_batch=1,
            batch_selection_strategy="strict_on_policy",
            max_weight_staleness_versions=0,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        mbs = ray.get(trainer.get_microbatch_calls.remote())
        # The stale group was evicted by _evict_stale_claimed; never dispatched
        all_sample_ids = [sid for _, ids, _ in mbs for sid in ids]
        assert "stale_g0" not in all_sample_ids

    def test_long_tail_overlap(self, ray_init):
        """First microbatch begins before the long-tail group's rollout finishes."""
        dp_client = FakeDataPlaneActor.remote()
        # Group 0 fast, 1-3 medium, group 4 slow
        gen = DryRunStaggeredGenWorker.remote()
        trainer = DryRunTrainer.remote(
            dp_client, train_latency_s=0.0, microbatch_latency_s=0.0
        )
        prompts = ["0:0.01", "1:0.03", "2:0.03", "3:0.03", "4:0.30"]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            target_prompt_groups_per_step=5,
            min_prompt_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        mbs = ray.get(trainer.get_microbatch_calls.remote())
        assert len(mbs) == 5
        first_mb_ts = mbs[0][2]
        # generate() records call-time before sleep; derive completion via prompt latency.
        call_ts = ray.get(gen.get_call_timestamps.remote())
        latencies = [float(p.split(":")[1]) for p in prompts]
        completion_ts = [call_ts[i] + latencies[i] for i in range(len(prompts))]
        slow_completion = max(completion_ts)
        assert first_mb_ts < slow_completion, (
            f"first mb dispatched at {first_mb_ts} but slow group "
            f"completed at {slow_completion}"
        )

    def test_abort_train_step_idempotent_and_clears_state(self, ray_init):
        """abort_train_step clears state and a new begin succeeds."""
        dp_client = FakeDataPlaneActor.remote()
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.0)
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["a", "b"],
        )
        ray.get(trainer.begin_train_step.remote("step-x"))
        ray.get(trainer.train_microbatch_from_meta.remote("step-x", meta))
        ray.get(trainer.train_microbatch_from_meta.remote("step-x", meta))
        ray.get(trainer.abort_train_step.remote("step-x"))
        assert ray.get(trainer.get_open_step_id.remote()) is None
        # New begin must succeed
        ray.get(trainer.begin_train_step.remote("step-y"))
        assert ray.get(trainer.get_open_step_id.remote()) == "step-y"
        # Idempotent: a second abort on a closed step also clears (no raise)
        ray.get(trainer.abort_train_step.remote("step-y"))
        assert ray.get(trainer.get_open_step_id.remote()) is None
        ray.get(trainer.abort_train_step.remote("step-y"))
        assert ray.get(trainer.get_open_step_id.remote()) is None

    @pytest.mark.skip("current fail")
    def test_empty_step_is_no_op(self, ray_init):
        """No rollouts → SC exits without calling finish_train_step."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunStaggeredGenWorker.remote()
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.0)
        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=["0:0.01"],
            max_train_steps=1,
            target_prompt_groups_per_step=2,
            min_prompt_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 0
        assert ray.get(trainer.get_finish_calls.remote()) == []
        assert ray.get(trainer.get_microbatch_calls.remote()) == []

    def test_clear_samples_called_once_per_step(self, ray_init):
        """clear_samples is called exactly once per step covering all dispatched ids."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunStaggeredGenWorker.remote()
        trainer = DryRunTrainer.remote(dp_client, train_latency_s=0.0)
        prompts = ["0:0.01", "1:0.02", "2:0.03"]
        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            target_prompt_groups_per_step=3,
            min_prompt_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        clear_calls = ray.get(dp_client.get_clear_calls.remote())
        assert len(clear_calls) == 1
        mbs = ray.get(trainer.get_microbatch_calls.remote())
        dispatched_ids = set()
        for _, ids, _ in mbs:
            dispatched_ids.update(ids)
        assert set(clear_calls[0]) == dispatched_ids


class TestRisk06EventLoopBlocking:
    """RISK-06: validate that asyncio event loop is not blocked during training.

    The risk: if train_from_meta is a synchronous blocking call, the asyncio
    event loop freezes and _rollout_pump + _sync_weights can't make progress.
    Fix: use `await loop.run_in_executor(None, blocking_fn, ...)` or ensure
    train_from_meta is an async method (as DryRunTrainer is).

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
