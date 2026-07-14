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

"""Focused unit tests for SingleController validation rollout draining."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import nemo_rl.algorithms.single_controller as sc_mod
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from tests.unit.single_controller.test_sc_checkpointing import (
    _FakeSampler,
    _FakeTrainer,
    _actor_master_config,
    _make_bundle,
    _step_dir_names,
)


_ACTOR_CLS = SingleControllerActor.__ray_metadata__.modified_class


class _ObservedEvent(asyncio.Event):
    """Event that exposes when the controller first closes admission."""

    def __init__(self) -> None:
        super().__init__()
        self.cleared = asyncio.Event()

    def clear(self) -> None:
        super().clear()
        self.cleared.set()


class _FakeWeightSynchronizer:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.sync_count = 0

    def sync_weights(self) -> None:
        self._order.append("sync")
        self.sync_count += 1


class _FakeRolloutManager:
    def __init__(self) -> None:
        self.weight_versions: list[int] = []

    def set_weight_version(self, version: int) -> None:
        self.weight_versions.append(version)


def _make_actor(
    order: list[str],
) -> tuple[Any, _ObservedEvent, _FakeWeightSynchronizer, _FakeRolloutManager]:
    actor = object.__new__(_ACTOR_CLS)
    gate = _ObservedEvent()
    gate.set()
    synchronizer = _FakeWeightSynchronizer(order)
    rollout_manager = _FakeRolloutManager()

    actor._rollout_permitted = gate
    actor._dispatched_rollouts = set()
    actor._weight_synchronizer = synchronizer
    actor._rollout_manager = rollout_manager
    actor._trainer_version = 7
    return actor, gate, synchronizer, rollout_manager


def test_regular_sync_does_not_wait_for_dispatched_rollout() -> None:
    async def _main() -> None:
        order: list[str] = []
        actor, gate, synchronizer, rollout_manager = _make_actor(order)
        rollout_started = asyncio.Event()
        release_rollout = asyncio.Event()

        async def _blocked_rollout() -> None:
            rollout_started.set()
            await release_rollout.wait()
            order.append("rollout_finished")

        rollout_task = asyncio.create_task(_blocked_rollout())
        actor._dispatched_rollouts.add(rollout_task)
        rollout_task.add_done_callback(actor._dispatched_rollouts.discard)
        await rollout_started.wait()

        try:
            await asyncio.wait_for(actor._sync_weights(), timeout=5.0)

            assert synchronizer.sync_count == 1
            assert rollout_manager.weight_versions == [7]
            assert gate.is_set()
            assert not rollout_task.done()
            assert order == ["sync"]
        finally:
            release_rollout.set()
            await rollout_task

    asyncio.run(_main())


def test_validation_drain_waits_for_dispatched_rollout() -> None:
    async def _main() -> None:
        order: list[str] = []
        actor, gate, _, _ = _make_actor(order)
        rollout_started = asyncio.Event()
        release_rollout = asyncio.Event()

        async def _blocked_rollout() -> None:
            rollout_started.set()
            await release_rollout.wait()
            order.append("rollout_finished")

        rollout_task = asyncio.create_task(_blocked_rollout())
        actor._dispatched_rollouts.add(rollout_task)
        rollout_task.add_done_callback(actor._dispatched_rollouts.discard)
        await rollout_started.wait()

        drain_task = asyncio.create_task(actor._pause_and_drain_rollouts())
        await gate.cleared.wait()

        assert not drain_task.done()
        assert not gate.is_set()

        release_rollout.set()
        await drain_task

        assert rollout_task.done()
        assert order == ["rollout_finished"]
        assert not gate.is_set()

    asyncio.run(_main())


def test_validation_drain_settles_snapshot_before_propagating_cancellation() -> None:
    async def _main() -> None:
        order: list[str] = []
        actor, gate, _, _ = _make_actor(order)
        cancelled_started = asyncio.Event()
        blocked_started = asyncio.Event()
        cancellation_observed = asyncio.Event()
        hold_cancelled = asyncio.Event()
        release_blocked = asyncio.Event()

        async def _cancelled_rollout() -> None:
            cancelled_started.set()
            try:
                await hold_cancelled.wait()
            finally:
                cancellation_observed.set()

        async def _blocked_rollout() -> None:
            blocked_started.set()
            await release_blocked.wait()
            order.append("blocked_finished")

        cancelled_task = asyncio.create_task(_cancelled_rollout())
        blocked_task = asyncio.create_task(_blocked_rollout())
        rollout_tasks = (cancelled_task, blocked_task)
        for task in rollout_tasks:
            actor._dispatched_rollouts.add(task)
            task.add_done_callback(actor._dispatched_rollouts.discard)

        drain_task: asyncio.Task[None] | None = None
        try:
            await asyncio.gather(
                cancelled_started.wait(),
                blocked_started.wait(),
            )
            drain_task = asyncio.create_task(actor._pause_and_drain_rollouts())
            await gate.cleared.wait()

            # clear() and the snapshot have no await between them, so the
            # cancelled task is already part of this drain.
            cancelled_task.cancel()
            await cancellation_observed.wait()

            assert not drain_task.done()
            assert not gate.is_set()

            release_blocked.set()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("drain did not propagate rollout cancellation")

            assert order == ["blocked_finished"]
            assert not gate.is_set()
        finally:
            release_blocked.set()
            cleanup_tasks = [*rollout_tasks]
            if drain_task is not None:
                cleanup_tasks.append(drain_task)
            for task in cleanup_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    asyncio.run(_main())


def test_sync_while_paused_runs_after_drain_and_stays_paused() -> None:
    async def _main() -> None:
        order: list[str] = []
        actor, gate, synchronizer, rollout_manager = _make_actor(order)
        rollout_started = asyncio.Event()
        release_rollout = asyncio.Event()

        async def _blocked_rollout() -> None:
            rollout_started.set()
            await release_rollout.wait()
            order.append("rollout_finished")

        rollout_task = asyncio.create_task(_blocked_rollout())
        actor._dispatched_rollouts.add(rollout_task)
        rollout_task.add_done_callback(actor._dispatched_rollouts.discard)
        await rollout_started.wait()

        drain_task = asyncio.create_task(actor._pause_and_drain_rollouts())
        await gate.cleared.wait()
        assert synchronizer.sync_count == 0

        release_rollout.set()
        await drain_task
        await actor._sync_weights(reopen_rollouts=False)

        assert order == ["rollout_finished", "sync"]
        assert synchronizer.sync_count == 1
        assert rollout_manager.weight_versions == [7]
        assert not gate.is_set()

    asyncio.run(_main())


class _ValidationLogger:
    def __init__(self) -> None:
        self.metric_calls: list[tuple[dict[str, Any], int, str]] = []
        self.jsonl_calls: list[tuple[dict[str, Any], str]] = []
        self.finish_count = 0

    def log_metrics(
        self, metrics: dict[str, Any], step: int, prefix: str
    ) -> None:
        self.metric_calls.append((dict(metrics), step, prefix))

    def log_batched_dict_as_jsonl(
        self, data: dict[str, Any], filename: str
    ) -> None:
        self.jsonl_calls.append(
            ({key: list(value) for key, value in data.items()}, filename)
        )

    def finish(self) -> None:
        self.finish_count += 1


def _validation_batch(indices: list[int]) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "idx": indices,
            "message_log": [
                [{"role": "user", "content": f"question-{idx}"}]
                for idx in indices
            ],
            "extra_env_info": [None for _ in indices],
            "task_name": ["math" for _ in indices],
        }
    )


def _validation_record(
    idx: int,
    *,
    reward: float,
    length: float,
    num_completions: int = 1,
) -> PromptGroupRecord:
    completion = Completion(
        message_log=[
            {"role": "user", "content": f"question-{idx}"},
            {
                "role": "assistant",
                "content": f"answer-{idx}",
                "token_ids": [idx],
            },
        ],
        env_extras=None,
        truncated=False,
        reward=reward,
    )
    return PromptGroupRecord(
        prompt_idx=idx,
        prompt=[{"role": "user", "content": f"question-{idx}"}],
        extra_env_info=None,
        metadata={"task_name": "math"},
        completions=[completion for _ in range(num_completions)],
        rollout_metrics={"mean_gen_tokens_per_sample": length},
    )


class _ValidationRolloutManager:
    def __init__(
        self,
        outcomes: dict[int, PromptGroupRecord | BaseException],
        *,
        releases: dict[int, asyncio.Event] | None = None,
    ) -> None:
        self._outcomes = outcomes
        self._releases = releases or {}
        self.calls: list[dict[str, Any]] = []
        self.started: dict[int, asyncio.Event] = {
            idx: asyncio.Event() for idx in outcomes
        }
        self.finished: dict[int, asyncio.Event] = {
            idx: asyncio.Event() for idx in outcomes
        }

    async def run_rollout(self, prompt: dict[str, Any]) -> PromptGroupRecord:
        self.calls.append(prompt)
        idx = prompt["idx"]
        self.started[idx].set()
        try:
            if idx in self._releases:
                await self._releases[idx].wait()
            outcome = self._outcomes[idx]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.finished[idx].set()

    async def generate_and_push(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("validation must not use generate_and_push")


def _make_validation_actor(
    *,
    batches: list[BatchedDataDict],
    manager: _ValidationRolloutManager,
    max_val_samples: int | None,
) -> tuple[Any, _ValidationLogger]:
    actor = object.__new__(_ACTOR_CLS)
    logger = _ValidationLogger()
    actor._val_dataloader = batches
    actor._validation_rollout_manager = manager
    actor._master_config = SimpleNamespace(
        grpo={"max_val_samples": max_val_samples},
        logger={
            "num_val_samples_to_print": (
                max_val_samples if max_val_samples is not None else 0
            )
        },
    )
    actor._logger = logger
    return actor, logger


def test_validation_executor_aggregates_and_preserves_input_order(monkeypatch) -> None:
    async def _main() -> None:
        releases = {idx: asyncio.Event() for idx in range(3)}
        outcomes = {
            0: _validation_record(0, reward=1.0, length=4.0),
            1: _validation_record(1, reward=0.0, length=8.0),
            2: _validation_record(2, reward=0.5, length=6.0),
        }
        manager = _ValidationRolloutManager(outcomes, releases=releases)
        actor, logger = _make_validation_actor(
            batches=[_validation_batch([0, 1, 2])],
            manager=manager,
            max_val_samples=3,
        )
        printed: list[tuple[list[Any], list[float], int, int]] = []

        def _capture_samples(
            message_logs: list[Any],
            rewards: list[float],
            num_samples: int,
            step: int,
        ) -> None:
            printed.append((list(message_logs), list(rewards), num_samples, step))

        monkeypatch.setattr(sc_mod, "print_message_log_samples", _capture_samples)

        validation_task = asyncio.create_task(actor._run_validation(step=7))
        await asyncio.gather(*(event.wait() for event in manager.started.values()))
        for idx in reversed(range(3)):
            releases[idx].set()
        metrics = await validation_task

        assert metrics == {
            "accuracy": 0.5,
            "avg_length": 6.0,
            "num_samples": 3,
        }
        assert [prompt["idx"] for prompt in manager.calls] == [0, 1, 2]
        assert set(manager.calls[0]) == {
            "idx",
            "message_log",
            "extra_env_info",
            "task_name",
        }
        jsonl_message_logs = logger.jsonl_calls[0][0]["content"]
        assert [
            message_log[-1]["content"]
            for message_log in jsonl_message_logs
        ] == ["answer-0", "answer-1", "answer-2"]
        assert all(
            set(message) == {"role", "content"}
            for message_log in jsonl_message_logs
            for message in message_log
        )
        assert logger.jsonl_calls[0][0]["rewards"] == [1.0, 0.0, 0.5]
        assert logger.jsonl_calls[0][1] == "val_data_step7.jsonl"
        assert all(
            set(message) == {"role", "content"}
            for message_log in printed[0][0]
            for message in message_log
        )
        assert printed[0][1:] == ([1.0, 0.0, 0.5], 3, 7)
        assert [call[2] for call in logger.metric_calls] == [
            "validation",
            "timing/validation",
        ]

    asyncio.run(_main())


@pytest.mark.parametrize(
    ("sample_limit", "expected_indices"),
    [
        (2, [0, 1]),
        (4, [0, 1, 2, 3]),
    ],
)
def test_validation_sample_limit_is_applied_by_row(
    monkeypatch,
    sample_limit,
    expected_indices,
) -> None:
    async def _main() -> None:
        outcomes = {
            idx: _validation_record(idx, reward=float(idx), length=float(idx + 1))
            for idx in range(6)
        }
        manager = _ValidationRolloutManager(outcomes)
        actor, _ = _make_validation_actor(
            batches=[_validation_batch([0, 1, 2]), _validation_batch([3, 4, 5])],
            manager=manager,
            max_val_samples=sample_limit,
        )
        monkeypatch.setattr(sc_mod, "print_message_log_samples", lambda *a, **k: None)

        metrics = await actor._run_validation(step=1)

        assert [prompt["idx"] for prompt in manager.calls] == expected_indices
        assert metrics["num_samples"] == sample_limit

    asyncio.run(_main())


def test_nemo_gym_validation_none_limit_consumes_complete_dataset(
    monkeypatch,
) -> None:
    async def _main() -> None:
        outcomes = {
            idx: _validation_record(idx, reward=float(idx), length=float(idx + 1))
            for idx in range(5)
        }
        manager = _ValidationRolloutManager(outcomes)
        actor, _ = _make_validation_actor(
            batches=[
                _validation_batch([0, 1]),
                _validation_batch([2, 3]),
                _validation_batch([4]),
            ],
            manager=manager,
            max_val_samples=None,
        )
        monkeypatch.setattr(sc_mod, "print_message_log_samples", lambda *a, **k: None)

        metrics = await actor._run_validation(step=1)

        assert [prompt["idx"] for prompt in manager.calls] == list(range(5))
        assert metrics == {"accuracy": 2.0, "avg_length": 3.0, "num_samples": 5}

    asyncio.run(_main())


def test_repeated_validation_starts_from_first_row(monkeypatch) -> None:
    async def _main() -> None:
        outcomes = {
            idx: _validation_record(idx, reward=1.0, length=2.0)
            for idx in range(3)
        }
        manager = _ValidationRolloutManager(outcomes)
        actor, _ = _make_validation_actor(
            batches=[_validation_batch([0, 1, 2])],
            manager=manager,
            max_val_samples=2,
        )
        monkeypatch.setattr(sc_mod, "print_message_log_samples", lambda *a, **k: None)

        await actor._run_validation(step=1)
        await actor._run_validation(step=2)

        assert [prompt["idx"] for prompt in manager.calls] == [0, 1, 0, 1]

    asyncio.run(_main())


def test_validation_settles_batch_before_propagating_error(monkeypatch) -> None:
    async def _main() -> None:
        releases = {idx: asyncio.Event() for idx in range(2)}
        error = RuntimeError("validation rollout failed")
        manager = _ValidationRolloutManager(
            {
                0: error,
                1: _validation_record(1, reward=1.0, length=2.0),
            },
            releases=releases,
        )
        actor, logger = _make_validation_actor(
            batches=[_validation_batch([0, 1])],
            manager=manager,
            max_val_samples=2,
        )
        monkeypatch.setattr(sc_mod, "print_message_log_samples", lambda *a, **k: None)

        validation_task = asyncio.create_task(actor._run_validation(step=1))
        await asyncio.gather(*(event.wait() for event in manager.started.values()))
        releases[0].set()
        await manager.finished[0].wait()
        assert not validation_task.done()

        releases[1].set()
        with pytest.raises(RuntimeError, match="validation rollout failed"):
            await validation_task
        assert logger.metric_calls == []
        assert logger.jsonl_calls == []

    asyncio.run(_main())


def test_validation_requires_exactly_one_completion(monkeypatch) -> None:
    async def _main() -> None:
        manager = _ValidationRolloutManager(
            {0: _validation_record(0, reward=1.0, length=2.0, num_completions=2)}
        )
        actor, _ = _make_validation_actor(
            batches=[_validation_batch([0])],
            manager=manager,
            max_val_samples=1,
        )
        monkeypatch.setattr(sc_mod, "print_message_log_samples", lambda *a, **k: None)

        with pytest.raises(ValueError, match="exactly one completion"):
            await actor._run_validation(step=1)

    asyncio.run(_main())


def _make_run_actor(
    *,
    current_step: int,
    validation_error: BaseException | None = None,
) -> tuple[Any, list[str], _ValidationLogger]:
    actor = object.__new__(_ACTOR_CLS)
    order: list[str] = []
    logger = _ValidationLogger()
    rollout_started = asyncio.Event()

    async def _sync_weights() -> None:
        order.append("sync")

    async def _run_validation(*, step: int) -> dict[str, Any]:
        order.append(f"validation-{step}")
        if validation_error is not None:
            raise validation_error
        return {}

    async def _rollout_pump() -> None:
        order.append("rollout-pump")
        rollout_started.set()
        await asyncio.Event().wait()

    async def _train_pump() -> None:
        await rollout_started.wait()
        order.append("train-pump")

    actor._sync_weights = _sync_weights
    actor._run_validation = _run_validation
    actor._rollout_pump = _rollout_pump
    actor._train_pump = _train_pump
    actor._master_config = SimpleNamespace(grpo={"val_at_start": True})
    actor._async_cfg = SimpleNamespace(over_sampling=False)
    actor._last_checkpoint_path = None
    actor._train_steps = current_step
    actor._trainer_version = current_step
    actor._dispatched_rollouts = set()
    actor._logger = logger
    return actor, order, logger


@pytest.mark.parametrize(
    ("current_step", "validation_expected"),
    [(0, True), (3, False)],
)
def test_val_at_start_runs_only_for_fresh_state(
    current_step,
    validation_expected,
) -> None:
    async def _main() -> None:
        actor, order, logger = _make_run_actor(current_step=current_step)

        await actor.run()

        assert order[0] == "sync"
        assert ("validation-0" in order) is validation_expected
        pump_positions = [
            order.index(name) for name in ("rollout-pump", "train-pump")
        ]
        if validation_expected:
            assert order.index("validation-0") < min(pump_positions)
        assert logger.finish_count == 1

    asyncio.run(_main())


def test_initial_validation_failure_propagates_before_pumps() -> None:
    async def _main() -> None:
        actor, order, _ = _make_run_actor(
            current_step=0,
            validation_error=RuntimeError("initial validation failed"),
        )

        with pytest.raises(RuntimeError, match="initial validation failed"):
            await actor.run()

        assert order == ["sync", "validation-0"]

    asyncio.run(_main())


@pytest.mark.parametrize(
    ("val_period", "val_at_end", "expected_validation_steps"),
    [
        (2, False, [2, 4]),
        (0, True, [4]),
        (2, True, [2, 4]),
    ],
)
def test_validation_cadence_and_non_validation_sync(
    tmp_path: Path,
    val_period: int,
    val_at_end: bool,
    expected_validation_steps: list[int],
) -> None:
    async def _main() -> tuple[Any, list[tuple[Any, ...]]]:
        actor = _ACTOR_CLS(
            _actor_master_config(
                tmp_path,
                max_num_steps=4,
                val_period=val_period,
                val_at_end=val_at_end,
                enabled=False,
            ),
            _make_bundle(),
        )
        actor._sampler = _FakeSampler()
        events: list[tuple[Any, ...]] = []

        async def _pause_and_drain_rollouts() -> None:
            events.append(("drain", actor._train_steps))
            actor._rollout_permitted.clear()

        async def _sync_weights(*, reopen_rollouts: bool = True) -> None:
            if reopen_rollouts:
                assert actor._rollout_permitted.is_set()
            else:
                assert not actor._rollout_permitted.is_set()
            events.append(("sync", actor._train_steps, reopen_rollouts))
            actor._rollout_permitted.clear()
            if reopen_rollouts:
                actor._rollout_permitted.set()

        async def _run_validation(*, step: int) -> dict[str, Any]:
            assert not actor._rollout_permitted.is_set()
            events.append(("validation", step))
            return {"accuracy": 1.0}

        actor._pause_and_drain_rollouts = _pause_and_drain_rollouts
        actor._sync_weights = _sync_weights
        actor._run_validation = _run_validation
        await actor._train_pump()
        return actor, events

    actor, events = asyncio.run(_main())

    assert [event[1] for event in events if event[0] == "validation"] == (
        expected_validation_steps
    )
    assert [event[1] for event in events if event[0] == "drain"] == (
        expected_validation_steps
    )
    for step in range(1, 5):
        step_events = [event for event in events if event[1] == step]
        if step in expected_validation_steps:
            assert step_events == [
                ("drain", step),
                ("sync", step, False),
                ("validation", step),
            ]
        else:
            assert step_events == [("sync", step, True)]
    assert actor._rollout_permitted.is_set()


def test_periodic_validation_drains_before_sync_and_keeps_gate_closed(
    tmp_path: Path,
) -> None:
    async def _main() -> tuple[Any, list[str], _FakeWeightSynchronizer]:
        actor = _ACTOR_CLS(
            _actor_master_config(
                tmp_path,
                max_num_steps=1,
                val_at_end=True,
                enabled=False,
            ),
            _make_bundle(),
        )
        actor._sampler = _FakeSampler()
        order: list[str] = []
        gate = _ObservedEvent()
        gate.set()
        actor._rollout_permitted = gate
        synchronizer = _FakeWeightSynchronizer(order)
        actor._weight_synchronizer = synchronizer
        actor._rollout_manager = _FakeRolloutManager()

        rollout_started = asyncio.Event()
        release_rollout = asyncio.Event()

        async def _old_rollout() -> None:
            rollout_started.set()
            await release_rollout.wait()
            order.append("rollout_finished")

        rollout_task = asyncio.create_task(_old_rollout())
        actor._dispatched_rollouts.add(rollout_task)
        rollout_task.add_done_callback(actor._dispatched_rollouts.discard)

        async def _release_after_gate_closes() -> None:
            await rollout_started.wait()
            await gate.cleared.wait()
            assert synchronizer.sync_count == 0
            release_rollout.set()

        release_task = asyncio.create_task(_release_after_gate_closes())

        async def _run_validation(*, step: int) -> dict[str, Any]:
            assert step == 1
            assert synchronizer.sync_count == 1
            assert not gate.is_set()
            order.append("validation")
            return {"accuracy": 1.0}

        actor._run_validation = _run_validation
        await actor._train_pump()
        await release_task
        await rollout_task
        return actor, order, synchronizer

    actor, order, synchronizer = asyncio.run(_main())

    assert order == ["rollout_finished", "sync", "validation"]
    assert synchronizer.sync_count == 1
    assert actor._rollout_permitted.is_set()


def test_periodic_validation_failure_keeps_gate_closed_and_skips_checkpoint(
    tmp_path: Path,
) -> None:
    async def _main() -> tuple[Any, _FakeTrainer]:
        trainer = _FakeTrainer()
        actor = _ACTOR_CLS(
            _actor_master_config(
                tmp_path,
                max_num_steps=1,
                val_at_end=True,
                save_period=1,
            ),
            _make_bundle(trainer=trainer),
        )
        actor._sampler = _FakeSampler()

        async def _run_validation(*, step: int) -> dict[str, Any]:
            raise RuntimeError(f"validation failed at step {step}")

        actor._run_validation = _run_validation
        with pytest.raises(RuntimeError, match="validation failed at step 1"):
            await actor._train_pump()
        return actor, trainer

    actor, trainer = asyncio.run(_main())

    assert not actor._rollout_permitted.is_set()
    assert trainer.save_calls == []
    assert _step_dir_names(tmp_path / "checkpoints") == set()
