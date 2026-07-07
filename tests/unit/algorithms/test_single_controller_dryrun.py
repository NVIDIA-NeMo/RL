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

Validates the pump orchestration using stub actors with configurable sleep
latencies — no GPU, no real model weights required.

Coverage is intentionally limited to invariants that are hard to exercise
on real hardware:
  - Ordering contracts: logprob refresh before advantage stage, advantages
    written to the DataPlane before train dispatch.
  - Concurrency: rollout_pump dispatches while train_pump is busy.
  - Backpressure: buffer capacity blocks rollout_pump when full;
    _rollout_permitted pauses dispatch during _sync_weights.
  - StalenessSampler selection/eviction logic (pure, no fakes).
  - Streaming train_pump state machine: arrival-order dispatch, version
    advance only at finish, strict-on-policy mid-step rejection, abort
    idempotency, exactly-once clear_samples.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import pytest
import ray
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
    warn_if_staleness_window_below_minibatches,
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
        call_idx = self._call_count
        self._call_timestamps.append(time.monotonic())
        await asyncio.sleep(self._gen_latency_s)
        group_id = f"group-{call_idx:04d}"
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
class _TrainerStateRecorder:
    """Shared call/state ledger for DryRunTrainer.

    DryRunTrainer is a plain object; passing it into the SC actor
    cloudpickles a copy, so recorded state must live in an actor that both
    the test-side and SC-side copies can reach. The step-protocol
    invariants are enforced here so violations raise inside the trainer
    call and surface through SC's error path.
    """

    def __init__(self):
        self._open_step_id: str | None = None
        self._trainer_version = 0
        self._train_start_times: list[float] = []
        self._microbatch_calls: list[tuple[str, list[str], float]] = []
        self._finish_calls: list[str] = []
        self._abort_calls: list[str] = []
        self._prepare_logprobs_calls: list[tuple[list[str], bool, bool]] = []
        self._last_advantages: torch.Tensor | None = None

    def begin(self, step_id: str) -> None:
        if self._open_step_id is not None:
            raise RuntimeError(
                f"begin_train_step called while step {self._open_step_id} is open"
            )
        self._open_step_id = step_id

    def microbatch(self, step_id: str, sample_ids: list[str]) -> None:
        if self._open_step_id is None:
            raise RuntimeError("train_microbatch_from_meta called with no open step")
        if step_id != self._open_step_id:
            raise RuntimeError(
                f"train_microbatch_from_meta step_id={step_id!r} != open {self._open_step_id!r}"
            )
        now = time.monotonic()
        self._microbatch_calls.append((step_id, list(sample_ids), now))
        self._train_start_times.append(now)

    def finish(self, step_id: str) -> dict:
        if self._open_step_id is None:
            raise RuntimeError("finish_train_step called with no open step")
        if step_id != self._open_step_id:
            raise RuntimeError(
                f"finish_train_step step_id={step_id!r} != open {self._open_step_id!r}"
            )
        self._finish_calls.append(step_id)
        self._open_step_id = None
        self._trainer_version += 1
        # NOTE: real backends (TQPolicy, MegatronPolicyWorker) do not emit a
        # trainer_version key — SC owns that counter. Stub keeps its own
        # _trainer_version only for assertions in tests.
        return {"loss": 1.0 / (self._trainer_version + 1)}

    def abort(self, step_id: str) -> None:
        self._abort_calls.append(step_id)
        self._open_step_id = None

    def prepare_logprobs(
        self,
        sample_ids: list[str],
        refresh_policy_logprobs: bool,
        refresh_reference_logprobs: bool,
    ) -> None:
        self._prepare_logprobs_calls.append(
            (list(sample_ids), refresh_policy_logprobs, refresh_reference_logprobs)
        )

    def append_advantages(self, advantages: torch.Tensor) -> None:
        if self._last_advantages is None:
            self._last_advantages = advantages
        else:
            self._last_advantages = torch.cat(
                [self._last_advantages, advantages], dim=0
            )

    def get_open_step_id(self) -> str | None:
        return self._open_step_id

    def get_trainer_version(self) -> int:
        return self._trainer_version

    def get_train_start_times(self) -> list[float]:
        return list(self._train_start_times)

    def get_microbatch_calls(self) -> list[tuple[str, list[str], float]]:
        return list(self._microbatch_calls)

    def get_finish_calls(self) -> list[str]:
        return list(self._finish_calls)

    def get_abort_calls(self) -> list[str]:
        return list(self._abort_calls)

    def get_prepare_logprobs_calls(self) -> list[tuple[list[str], bool, bool]]:
        return list(self._prepare_logprobs_calls)

    def get_last_advantages(self) -> torch.Tensor | None:
        return self._last_advantages


class DryRunTrainer:
    """Stub trainer with the driver-side ``TQPolicy`` calling convention.

    Plain object (no actor wrapper), mirroring production where SC holds a
    ``TQPolicy`` instance and invokes it via ``asyncio.to_thread``. Methods
    are synchronous; latency is simulated with ``time.sleep``, which runs
    on SC's to_thread worker so the event loop stays responsive.

    Passing this object into the SC actor copies it; all recorded state
    lives in :class:`_TrainerStateRecorder` so the test-side copy and the
    SC-side copy observe the same ledger.
    """

    def __init__(
        self,
        dp_client: Any,
        train_latency_s: float = 0.2,
        expect_advantages: bool = False,
        microbatch_latency_s: float = 0.0,
    ):
        self._dp_client = dp_client
        # Retained for constructor parity with older tests; the split API
        # only consumes microbatch_latency_s.
        self._train_latency_s = train_latency_s
        self._expect_advantages = expect_advantages
        self._microbatch_latency_s = microbatch_latency_s
        self._rec = _TrainerStateRecorder.remote()

    def begin_train_step(
        self,
        step_id: str,
        loss_fn: Any = None,
        gbs: int = 0,
        mbs: int = 0,
    ) -> None:
        del loss_fn, gbs, mbs
        ray.get(self._rec.begin.remote(step_id))

    def train_microbatch_from_meta(self, step_id: str, meta: KVBatchMeta) -> None:
        ray.get(self._rec.microbatch.remote(step_id, list(meta.sample_ids)))
        if self._expect_advantages:
            data = ray.get(
                self._dp_client.get_samples.remote(
                    sample_ids=meta.sample_ids,
                    partition_id=meta.partition_id,
                    select_fields=["input_ids", "advantages"],
                )
            )
            ray.get(
                self._rec.append_advantages.remote(data["advantages"].detach().clone())
            )
        if self._microbatch_latency_s > 0:
            time.sleep(self._microbatch_latency_s)

    def finish_train_step(self, step_id: str) -> dict:
        return ray.get(self._rec.finish.remote(step_id))

    def abort_train_step(self, step_id: str) -> None:
        ray.get(self._rec.abort.remote(step_id))

    def prepare_logprobs_from_meta(
        self,
        meta: KVBatchMeta,
        *,
        refresh_policy_logprobs: bool = False,
        refresh_reference_logprobs: bool = False,
    ) -> None:
        ray.get(
            self._rec.prepare_logprobs.remote(
                list(meta.sample_ids),
                refresh_policy_logprobs,
                refresh_reference_logprobs,
            )
        )

    # ── test-side inspection (reads the shared recorder) ────────────────

    def get_open_step_id(self) -> str | None:
        return ray.get(self._rec.get_open_step_id.remote())

    def get_trainer_version(self) -> int:
        return ray.get(self._rec.get_trainer_version.remote())

    def get_train_start_times(self) -> list[float]:
        return ray.get(self._rec.get_train_start_times.remote())

    def get_microbatch_calls(self) -> list[tuple[str, list[str], float]]:
        return ray.get(self._rec.get_microbatch_calls.remote())

    def get_finish_calls(self) -> list[str]:
        return ray.get(self._rec.get_finish_calls.remote())

    def get_abort_calls(self) -> list[str]:
        return ray.get(self._rec.get_abort_calls.remote())

    def get_prepare_logprobs_calls(self) -> list[tuple[list[str], bool, bool]]:
        return ray.get(self._rec.get_prepare_logprobs_calls.remote())

    def get_last_advantages(self) -> torch.Tensor | None:
        return ray.get(self._rec.get_last_advantages.remote())


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


def _meta_with_versions(versions: list[int]) -> KVBatchMeta:
    sample_ids = [f"g{i}_g0" for i in range(len(versions))]
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids,
        tags=[
            {
                "group_id": f"g{i}",
                "weight_version": version,
                "committed": True,
                "expected_num_samples": 1,
            }
            for i, version in enumerate(versions)
        ],
    )


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
        max_rollout_prompts=12,
        min_groups_per_batch=1,
        group_size=1,
        max_buffered_rollouts=4,
        max_inflight_prompts=4,
        max_weight_staleness_versions=1,
        advantage_enabled=False,
        advantage_estimator=None,
        advantage_policy_logprobs_field=None,
        advantage_reference_logprobs_field=None,
        refresh_prev_logprobs_for_loss=False,
        refresh_reference_logprobs_for_loss=False,
        diagnostics=False,
    ):
        cfg = SingleControllerConfig(
            max_train_steps=max_train_steps,
            max_rollout_prompts=max_rollout_prompts,
            min_groups_per_batch=min_groups_per_batch,
            group_size=group_size,
            max_buffered_rollouts=max_buffered_rollouts,
            max_inflight_prompts=max_inflight_prompts,
            max_weight_staleness_versions=max_weight_staleness_versions,
            advantage_enabled=advantage_enabled,
            advantage_policy_logprobs_field=advantage_policy_logprobs_field,
            advantage_reference_logprobs_field=advantage_reference_logprobs_field,
            refresh_prev_logprobs_for_loss=refresh_prev_logprobs_for_loss,
            refresh_reference_logprobs_for_loss=refresh_reference_logprobs_for_loss,
            diagnostics=diagnostics,
        )
        if weight_sync is None:
            weight_sync = DryRunWeightSynchronizer(gen_handle=gen)
        prompts = [f"prompt_{i}" for i in range(10)]
        return SingleControllerActor.remote(
            cfg,
            prompts,
            dp_client,
            gen,
            trainer,
            weight_sync,
            None,  # loss_fn stub — DryRunTrainer ignores it
            advantage_estimator,
        )

    def test_prepare_logprobs_called_before_advantage_stage(self, ray_init):
        """SC fires prepare_logprobs_from_meta with config-driven flags.

        When ``advantage_policy_logprobs_field`` / ``advantage_reference_logprobs_field``
        are set on the config, the train_pump must call
        ``trainer.prepare_logprobs_from_meta(meta, refresh_policy_logprobs=...,
        refresh_reference_logprobs=...)`` exactly once per consumed group,
        before advantage estimation. With both fields None, the hook must
        not fire at all.
        """
        # Case 1: both flags set
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer = DryRunTrainer(dp_client, train_latency_s=0.01)

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=1,
            max_rollout_prompts=2,
            min_groups_per_batch=2,
            group_size=1,
            advantage_policy_logprobs_field="policy_logprobs",
            advantage_reference_logprobs_field="reference_logprobs",
        )
        ray.get(ctrl.run.remote(), timeout=30)
        calls = trainer.get_prepare_logprobs_calls()
        assert len(calls) == 2, f"expected 2 logprob refresh calls, got {calls}"
        for sample_ids, refresh_policy, refresh_ref in calls:
            assert refresh_policy is True
            assert refresh_ref is True
            assert len(sample_ids) > 0

        # Case 2: only policy field set
        dp_client2 = FakeDataPlaneActor.remote()
        gen2 = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer2 = DryRunTrainer(dp_client2, train_latency_s=0.01)
        ctrl2 = self._make_controller(
            dp_client2,
            gen2,
            trainer2,
            max_train_steps=1,
            max_rollout_prompts=2,
            min_groups_per_batch=2,
            group_size=1,
            advantage_policy_logprobs_field="policy_logprobs",
        )
        ray.get(ctrl2.run.remote(), timeout=30)
        calls2 = trainer2.get_prepare_logprobs_calls()
        assert len(calls2) == 2
        for _, refresh_policy, refresh_ref in calls2:
            assert refresh_policy is True
            assert refresh_ref is False

        # Case 3: neither field set → hook must not fire
        dp_client3 = FakeDataPlaneActor.remote()
        gen3 = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer3 = DryRunTrainer(dp_client3, train_latency_s=0.01)
        ctrl3 = self._make_controller(
            dp_client3,
            gen3,
            trainer3,
            max_train_steps=1,
            max_rollout_prompts=2,
            min_groups_per_batch=2,
            group_size=1,
        )
        ray.get(ctrl3.run.remote(), timeout=30)
        calls3 = trainer3.get_prepare_logprobs_calls()
        assert calls3 == [], (
            "prepare_logprobs_from_meta must not be called when both "
            f"advantage_*_logprobs_field are None; got {calls3}"
        )

    def test_loss_driven_logprob_refresh_without_advantage_fields(self, ray_init):
        """Loss flags alone must trigger prepare_logprobs_from_meta.

        The loss can consume prev/reference logprob columns even when the
        advantage estimator does not (sync GRPO refreshes reference
        logprobs iff ``reference_policy_kl_penalty != 0``), so
        ``refresh_*_for_loss`` gates the refresh independently of the
        ``advantage_*_logprobs_field`` triggers.
        """
        # Case 1: both loss flags set, no advantage fields
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer = DryRunTrainer(dp_client, train_latency_s=0.01)
        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=1,
            max_rollout_prompts=2,
            min_groups_per_batch=2,
            group_size=1,
            refresh_prev_logprobs_for_loss=True,
            refresh_reference_logprobs_for_loss=True,
        )
        ray.get(ctrl.run.remote(), timeout=30)
        calls = trainer.get_prepare_logprobs_calls()
        assert len(calls) == 2, f"expected 2 logprob refresh calls, got {calls}"
        for sample_ids, refresh_policy, refresh_ref in calls:
            assert refresh_policy is True
            assert refresh_ref is True
            assert len(sample_ids) > 0

        # Case 2: reference-only loss flag (KL penalty without IS correction)
        dp_client2 = FakeDataPlaneActor.remote()
        gen2 = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer2 = DryRunTrainer(dp_client2, train_latency_s=0.01)
        ctrl2 = self._make_controller(
            dp_client2,
            gen2,
            trainer2,
            max_train_steps=1,
            max_rollout_prompts=2,
            min_groups_per_batch=2,
            group_size=1,
            refresh_reference_logprobs_for_loss=True,
        )
        ray.get(ctrl2.run.remote(), timeout=30)
        calls2 = trainer2.get_prepare_logprobs_calls()
        assert len(calls2) == 2
        for _, refresh_policy, refresh_ref in calls2:
            assert refresh_policy is False
            assert refresh_ref is True

    def test_advantage_stage_writes_advantages_before_train(self, ray_init):
        """SC computes advantages from DataPlane inputs and writes them back."""
        dp_client = FakeDataPlaneActor.remote()
        gen = DryRunGenWorker.remote(gen_latency_s=0.01)
        trainer = DryRunTrainer(
            dp_client,
            train_latency_s=0.01,
            expect_advantages=True,
        )

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=1,
            max_rollout_prompts=2,
            min_groups_per_batch=2,
            group_size=1,
            advantage_enabled=True,
            advantage_estimator=DryRunAdvantageEstimator(),
        )

        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 1

        advantages = trainer.get_last_advantages()
        assert advantages is not None
        # Per-group dispatch: each group is one sample → centered advantage is 0.
        # The two microbatch calls are concatenated.
        assert advantages.shape == (2, 3)
        assert torch.allclose(advantages, torch.zeros((2, 3)))

    def test_rollout_pump_runs_concurrently_with_train(self, ray_init):
        """rollout_pump dispatches while train_pump is sleeping.

        If pumps were sequential, rollout dispatches would only happen
        between training steps. With concurrent asyncio tasks, rollout
        dispatches happen while trainer is in asyncio.sleep().
        """
        dp_client = FakeDataPlaneActor.remote()
        # Gen is fast (0.02s), trainer is slow (0.3s)
        gen = DryRunGenWorker.remote(gen_latency_s=0.02)
        trainer = DryRunTrainer(dp_client, train_latency_s=0.3)

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=2,
            max_rollout_prompts=10,
            min_groups_per_batch=1,
            group_size=1,
            max_buffered_rollouts=6,
            max_inflight_prompts=6,
        )

        ray.get(ctrl.run.remote(), timeout=30)

        # Multiple rollouts should have completed during the first train step
        call_timestamps = ray.get(gen.get_call_timestamps.remote())
        train_start_times = trainer.get_train_start_times()

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
        trainer = DryRunTrainer(dp_client, train_latency_s=0.3)  # slow trainer

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            max_train_steps=2,
            max_rollout_prompts=8,
            min_groups_per_batch=1,
            group_size=1,
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
        trainer = DryRunTrainer(dp_client, train_latency_s=0.05)
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
            max_rollout_prompts=8,
            min_groups_per_batch=1,
            group_size=1,
        )

        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 2
        # Weight sync happened (sync_count > 0 implies gate opened correctly)

    def test_staleness_sampler_filters_correctly(self):
        """StalenessSampler returns freshest complete groups within the window."""
        sampler = StalenessSampler(max_staleness_versions=2)

        meta = _meta_with_versions([3, 4, 5, 2, 6])

        indices = sampler.select_indices(
            meta,
            trainer_version=5,
            min_groups=2,
            group_size=1,
        )
        assert indices == [2, 1]

    def test_staleness_sampler_returns_none_when_insufficient(self):
        """StalenessSampler returns None when not enough eligible rows."""
        sampler = StalenessSampler(max_staleness_versions=1)
        meta = _meta_with_versions([1])
        result = sampler.select_indices(
            meta,
            trainer_version=5,
            min_groups=2,
            group_size=1,
        )
        assert result is None

    def test_staleness_sampler_requires_complete_groups(self):
        """Staleness sampler skips incomplete prompt groups."""
        sampler = StalenessSampler(max_staleness_versions=2)
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["p0_g0", "p1_g0", "p1_g1"],
            tags=[
                {"group_id": "p0", "weight_version": 5, "expected_num_samples": 2},
                {"group_id": "p1", "weight_version": 5, "expected_num_samples": 2},
                {"group_id": "p1", "weight_version": 5, "expected_num_samples": 2},
            ],
        )

        assert sampler.select_indices(
            meta,
            trainer_version=5,
            min_groups=1,
            group_size=2,
        ) == [1, 2]

    def test_strict_on_policy_batch_sampler_requires_exact_version(self):
        """Strict sampler waits for a full batch at the trainer version."""
        sampler = StalenessSampler(max_staleness_versions=0)
        meta = _meta_with_versions([4, 5, 5, 6])

        assert (
            sampler.select_indices(
                meta,
                trainer_version=5,
                min_groups=3,
                group_size=1,
            )
            is None
        )
        assert sampler.select_indices(
            meta,
            trainer_version=5,
            min_groups=2,
            group_size=1,
        ) == [1, 2]

    def test_strict_on_policy_batch_sampler_evicts_old_groups(self):
        """Strict sampler marks complete old-version groups for eviction."""
        sampler = StalenessSampler(max_staleness_versions=0)
        meta = _meta_with_versions([4, 5, 4])

        assert sampler.evictable_indices(
            meta,
            trainer_version=5,
            group_size=1,
        ) == [0, 2]


@ray.remote(num_cpus=0)
class StaggeredGenWorker:
    """Gen worker that reads latency and group label from the prompt.

    Prompt format: ``"{idx}:{latency}"``. Sleeps for ``latency`` then
    pushes a row with ``group_id="group-{idx}"``.
    """

    def __init__(self, weight_version: int = 0, unique_ids: bool = False) -> None:
        self._weight_version = weight_version
        self._completion_timestamps: list[float] = []
        # When True, suffix ids with a per-call counter so re-dispatching
        # the same prompt (epoch loops) yields distinct groups instead of
        # colliding on already-claimed DataPlane rows.
        self._unique_ids = unique_ids
        self._calls = 0

    async def generate_and_push(self, prompt: str, dp_client: Any) -> None:
        idx_str, latency_str = prompt.split(":")
        idx = int(idx_str)
        latency = float(latency_str)
        self._calls += 1
        call_idx = self._calls
        await asyncio.sleep(latency)
        group_id = f"group-{idx:04d}"
        if self._unique_ids:
            group_id = f"{group_id}-c{call_idx:04d}"
        sample_id = f"{group_id}_g0"
        await dp_client.put_samples.remote(
            sample_ids=[sample_id],
            partition_id="rollout_data",
            fields=TensorDict(
                {
                    "input_ids": torch.ones((1, 3), dtype=torch.long),
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
        self._completion_timestamps.append(time.monotonic())

    def get_completion_timestamps(self) -> list[float]:
        return list(self._completion_timestamps)

    def set_weight_version(self, version: int) -> None:
        self._weight_version = version


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
        max_rollout_prompts=4,
        min_groups_per_batch=1,
        target_groups_per_step=4,
        group_size=1,
        max_buffered_rollouts=8,
        max_inflight_prompts=8,
        max_weight_staleness_versions=1,
        batch_selection_strategy="staleness_window",
        max_num_epochs=None,
        train_global_batch_size=None,
    ):
        cfg = SingleControllerConfig(
            max_train_steps=max_train_steps,
            max_rollout_prompts=max_rollout_prompts,
            min_groups_per_batch=min_groups_per_batch,
            target_groups_per_step=target_groups_per_step,
            group_size=group_size,
            max_buffered_rollouts=max_buffered_rollouts,
            max_inflight_prompts=max_inflight_prompts,
            max_weight_staleness_versions=max_weight_staleness_versions,
            batch_selection_strategy=batch_selection_strategy,
            max_num_epochs=max_num_epochs,
            train_global_batch_size=train_global_batch_size,
        )
        if weight_sync is None:
            weight_sync = DryRunWeightSynchronizer(gen_handle=gen)
        return SingleControllerActor.remote(
            cfg,
            prompts if prompts else ["unused"],
            dp_client,
            gen,
            trainer,
            weight_sync,
            None,  # loss_fn stub — DryRunTrainer ignores it
            None,
        )

    def test_streaming_dispatches_in_arrival_order(self, ray_init):
        """SC dispatches train_microbatch in order groups commit at DP."""
        dp_client = FakeDataPlaneActor.remote()
        # Group 0 slow, group 1 fast, group 2 medium → arrival order: 1, 2, 0
        gen = StaggeredGenWorker.remote()
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        prompts = ["0:0.30", "1:0.05", "2:0.15"]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            max_rollout_prompts=3,
            target_groups_per_step=3,
            min_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        mbs = trainer.get_microbatch_calls()
        # 3 microbatches dispatched
        assert len(mbs) == 3
        dispatched_groups = [call[1][0].split("_")[0] for call in mbs]
        # group-0001 (fastest) before group-0002 (medium) before group-0000 (slow)
        assert dispatched_groups == ["group-0001", "group-0002", "group-0000"]

    def test_trainer_version_advances_only_at_finish(self, ray_init):
        """trainer_version stays put across mb calls; ticks on finish."""
        dp_client = FakeDataPlaneActor.remote()
        gen = StaggeredGenWorker.remote()
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        prompts = [f"{i}:0.02" for i in range(4)]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            max_rollout_prompts=4,
            target_groups_per_step=4,
            min_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        # trainer_version should be 1 (one finish_train_step call)
        assert trainer.get_trainer_version() == 1
        # finish was called exactly once
        finishes = trainer.get_finish_calls()
        # SC step_id now carries a mini-batch suffix. With the default
        # train_global_batch_size (coerced to samples_per_step), there's
        # exactly one mini-batch per outer step → suffix "-mb-00".
        assert finishes == ["sc-step-000000-mb-00"]
        mbs = trainer.get_microbatch_calls()
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
        gen = StaggeredGenWorker.remote(weight_version=0)
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        prompts = [f"{i}:0.02" for i in range(2)]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            max_rollout_prompts=2,
            target_groups_per_step=2,
            min_groups_per_batch=1,
            batch_selection_strategy="strict_on_policy",
            max_weight_staleness_versions=0,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        mbs = trainer.get_microbatch_calls()
        # The stale group was evicted by _evict_stale_claimed; never dispatched
        all_sample_ids = [sid for _, ids, _ in mbs for sid in ids]
        assert "stale_g0" not in all_sample_ids

    def test_long_tail_overlap(self, ray_init):
        """First microbatch begins before the long-tail group's rollout finishes."""
        dp_client = FakeDataPlaneActor.remote()
        # Group 0 fast, 1-3 medium, group 4 slow
        gen = StaggeredGenWorker.remote()
        trainer = DryRunTrainer(
            dp_client, train_latency_s=0.0, microbatch_latency_s=0.0
        )
        prompts = ["0:0.01", "1:0.03", "2:0.03", "3:0.03", "4:0.30"]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            max_rollout_prompts=5,
            target_groups_per_step=5,
            min_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        mbs = trainer.get_microbatch_calls()
        assert len(mbs) == 5
        first_mb_ts = mbs[0][2]
        completion_ts = ray.get(gen.get_completion_timestamps.remote())
        # 5 completions; the slow group is the last one to finish
        slow_completion = max(completion_ts)
        assert first_mb_ts < slow_completion, (
            f"first mb dispatched at {first_mb_ts} but slow group "
            f"completed at {slow_completion}"
        )

    def test_abort_train_step_idempotent_and_clears_state(self, ray_init):
        """abort_train_step clears state and a new begin succeeds."""
        dp_client = FakeDataPlaneActor.remote()
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["a", "b"],
        )
        trainer.begin_train_step("step-x")
        trainer.train_microbatch_from_meta("step-x", meta)
        trainer.train_microbatch_from_meta("step-x", meta)
        trainer.abort_train_step("step-x")
        assert trainer.get_open_step_id() is None
        # New begin must succeed
        trainer.begin_train_step("step-y")
        assert trainer.get_open_step_id() == "step-y"
        # Idempotent: a second abort on a closed step also clears (no raise)
        trainer.abort_train_step("step-y")
        assert trainer.get_open_step_id() is None
        trainer.abort_train_step("step-y")
        assert trainer.get_open_step_id() is None

    def test_empty_step_is_no_op(self, ray_init):
        """No rollouts → SC exits without calling finish_train_step."""
        dp_client = FakeDataPlaneActor.remote()
        gen = StaggeredGenWorker.remote()
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=["0:0.01"],
            max_train_steps=1,
            max_rollout_prompts=0,
            target_groups_per_step=2,
            min_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=30)
        assert result["train_steps"] == 0
        assert trainer.get_finish_calls() == []
        assert trainer.get_microbatch_calls() == []

    def test_clear_samples_called_once_per_step(self, ray_init):
        """clear_samples is called exactly once per step covering all dispatched ids."""
        dp_client = FakeDataPlaneActor.remote()
        gen = StaggeredGenWorker.remote()
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        prompts = ["0:0.01", "1:0.02", "2:0.03"]
        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            max_rollout_prompts=3,
            target_groups_per_step=3,
            min_groups_per_batch=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        clear_calls = ray.get(dp_client.get_clear_calls.remote())
        assert len(clear_calls) == 1
        mbs = trainer.get_microbatch_calls()
        dispatched_ids = set()
        for _, ids, _ in mbs:
            dispatched_ids.update(ids)
        assert set(clear_calls[0]) == dispatched_ids

    def test_epoch_bound_limits_dataset_passes(self, ray_init):
        """max_num_epochs bounds dispatch to N passes over the prompt list."""
        dp_client = FakeDataPlaneActor.remote()
        gen = StaggeredGenWorker.remote(unique_ids=True)
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        prompts = ["0:0.01", "1:0.01"]
        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            # Budget far above the epoch bound: epochs must be the binding
            # constraint (2 epochs x 2 prompts = 4 dispatches, not 100).
            max_rollout_prompts=100,
            target_groups_per_step=4,
            min_groups_per_batch=1,
            max_num_epochs=2,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["epochs"] == 2
        assert result["train_steps"] == 1
        mbs = trainer.get_microbatch_calls()
        dispatched_ids = set()
        for _, ids, _ in mbs:
            dispatched_ids.update(ids)
        assert len(dispatched_ids) == len(prompts) * 2, (
            f"expected 4 unique sample ids (2 epochs x 2 prompts), got "
            f"{sorted(dispatched_ids)}"
        )

    def test_incomplete_group_does_not_hang_train_pump(self, ray_init):
        """A permanently-incomplete group must not deadlock the train pump.

        Seed DataPlane with 3 of ``expected_num_samples=4`` rows for one
        group and dispatch no rollouts: the group is never selectable and
        never evictable, so without the post-``_rollout_done`` flush the
        exhaustion check never fires and ``run()`` hangs.
        """
        dp = FakeDataPlaneActor.remote()
        ray.get(
            dp.put_samples.remote(
                sample_ids=["g0_s0", "g0_s1", "g0_s2"],
                partition_id="rollout_data",
                fields=TensorDict(
                    {"input_ids": torch.ones((3, 3), dtype=torch.long)},
                    batch_size=[3],
                ),
                tags=[
                    {
                        "group_id": "g0",
                        "weight_version": 0,
                        "committed": True,
                        "expected_num_samples": 4,
                    }
                ]
                * 3,
            )
        )
        ctrl = self._make_controller(
            dp,
            StaggeredGenWorker.remote(weight_version=0),
            DryRunTrainer(dp, train_latency_s=0.0),
            prompts=[],
            max_train_steps=1,
            max_rollout_prompts=0,
            target_groups_per_step=1,
            min_groups_per_batch=1,
            group_size=4,
            batch_selection_strategy="strict_on_policy",
            max_weight_staleness_versions=0,
        )
        result = ray.get(ctrl.run.remote(), timeout=15)
        assert result["train_steps"] == 0
        # The stuck rows must be cleared from the partition, not leaked.
        clear_calls = ray.get(dp.get_clear_calls.remote())
        assert set(clear_calls[-1]) == {"g0_s0", "g0_s1", "g0_s2"}

    def test_multi_minibatch_step_consumes_all_groups(self, ray_init):
        """num_minibatches > 1 runs multiple opt.steps per outer step.

        ``target_groups_per_step=4`` with ``train_global_batch_size=2`` and
        ``group_size=1`` gives 2 groups per mini-batch → 2 opt.steps per
        outer step. With a staleness window of 1, groups produced at the
        step's starting version survive the mid-step ``trainer_version`` bump
        (lag 1 <= window 1) and remain selectable for the second mini-batch,
        so the step completes and every dispatched sample is consumed.
        """
        dp_client = FakeDataPlaneActor.remote()
        gen = StaggeredGenWorker.remote(weight_version=0)
        trainer = DryRunTrainer(dp_client, train_latency_s=0.0)
        prompts = [f"{i}:0.01" for i in range(4)]

        ctrl = self._make_controller(
            dp_client,
            gen,
            trainer,
            prompts=prompts,
            max_train_steps=1,
            max_rollout_prompts=4,
            target_groups_per_step=4,
            train_global_batch_size=2,  # 2 groups/mb → 2 mini-batches
            min_groups_per_batch=1,
            group_size=1,
            batch_selection_strategy="staleness_window",
            max_weight_staleness_versions=1,
        )
        result = ray.get(ctrl.run.remote(), timeout=60)
        assert result["train_steps"] == 1
        # 2 opt.steps → 2 finish calls → trainer_version advanced twice.
        assert result["trainer_version"] == 2
        assert trainer.get_finish_calls() == [
            "sc-step-000000-mb-00",
            "sc-step-000000-mb-01",
        ]
        mbs = trainer.get_microbatch_calls()
        assert len(mbs) == 4
        dispatched_ids = set()
        for _, ids, _ in mbs:
            dispatched_ids.update(ids)
        # Every dispatched sample must be cleared (one clear per opt.step).
        clear_calls = ray.get(dp_client.get_clear_calls.remote())
        cleared_ids = set()
        for call in clear_calls:
            cleared_ids.update(call)
        assert cleared_ids == dispatched_ids
        assert len(dispatched_ids) == 4

    def test_warns_when_staleness_window_below_minibatches(self, caplog):
        """__init__'s validation warns when the window can't cover mini-batches.

        The check is factored into a module-level helper so it is unit-
        testable without constructing the Ray actor (which would run
        ``__init__`` remotely and make caplog useless).
        """
        # num_minibatches = 4 // (2 // 1) = 2; strict_on_policy pins the
        # effective window to 0 < num_minibatches - 1 (=1) → warn.
        cfg = SingleControllerConfig(
            target_groups_per_step=4,
            group_size=1,
            train_global_batch_size=2,
            min_groups_per_batch=1,
            batch_selection_strategy="strict_on_policy",
            max_weight_staleness_versions=0,
        )
        with caplog.at_level(
            logging.WARNING, logger="nemo_rl.algorithms.single_controller"
        ):
            warn_if_staleness_window_below_minibatches(cfg)
        assert any("num_minibatches" in record.message for record in caplog.records), (
            f"expected a staleness-window warning, got {caplog.records}"
        )

        # A window of 1 covers the single mid-step version bump → no warning.
        caplog.clear()
        cfg_ok = SingleControllerConfig(
            target_groups_per_step=4,
            group_size=1,
            train_global_batch_size=2,
            min_groups_per_batch=1,
            batch_selection_strategy="staleness_window",
            max_weight_staleness_versions=1,
        )
        with caplog.at_level(
            logging.WARNING, logger="nemo_rl.algorithms.single_controller"
        ):
            warn_if_staleness_window_below_minibatches(cfg_ok)
        assert not any(
            "num_minibatches" in record.message for record in caplog.records
        ), f"unexpected staleness-window warning, got {caplog.records}"
