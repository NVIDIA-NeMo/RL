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

"""Supervise failures until training drains, including after rollout exhaustion."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.experience.failures import RolloutStall


async def _blocked() -> None:
    await asyncio.Event().wait()


def _bare_actor() -> Any:
    """Use the real pump supervisor without starting models or Ray workers."""
    actor_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(actor_cls)
    ctrl._gen_fleet = None
    ctrl._engine_supervisor = None
    ctrl._train_steps = 7
    ctrl._trainer_version = 7
    ctrl._master_config = SimpleNamespace(
        rollout_checkpointing=SimpleNamespace(
            snapshot_attempt_interval_s=None, telemetry_interval_s=None
        )
    )
    ctrl._rollout_checkpoint_stop_requested = asyncio.Event()
    ctrl._weight_synchronizer = SimpleNamespace(is_stale=False, shutdown=Mock())
    ctrl._logger = SimpleNamespace(finish=Mock())
    ctrl._checkpointer = SimpleNamespace(shutdown=Mock())
    ctrl._finalizer_actors = []
    ctrl._storage_units = []
    ctrl._train_data_dump = None
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=Mock())
    ctrl._sync_weights = AsyncMock()
    ctrl._maybe_restore_replay_buffer = AsyncMock(return_value=0)
    ctrl._maybe_restore_rollout_recovery = AsyncMock()
    ctrl._maybe_restore_replacement_reserve = AsyncMock()
    ctrl._validate_restored_sampler_cursor = Mock()
    ctrl._log_rollout_restore_metrics = Mock()
    ctrl._log_rollout_throughput_metrics = AsyncMock()
    ctrl._finish_pending_save = AsyncMock()
    ctrl._rollout_pump = _blocked
    ctrl._train_pump = _blocked
    ctrl._stall_watchdog_pump = _blocked
    return ctrl


async def _start_run(ctrl: Any, *, draining: bool) -> asyncio.Task:
    exhausted = asyncio.Event()
    if draining:

        async def _exhaust() -> None:
            exhausted.set()

        ctrl._rollout_pump = _exhaust
    task = asyncio.create_task(ctrl._run_pumps())
    if draining:
        await asyncio.wait_for(exhausted.wait(), timeout=2)
    # Let normal rollout completion reach the supervisor before releasing a
    # failure. Event-loop yields avoid depending on a wall-clock sleep delay.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return task


def _assert_cleanup(ctrl: Any) -> None:
    ctrl._finish_pending_save.assert_awaited_once_with(blocking=True)
    ctrl._weight_synchronizer.shutdown.assert_called_once_with()
    ctrl._logger.finish.assert_called_once_with()
    ctrl._checkpointer.shutdown.assert_called_once_with()
    assert asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.parametrize("draining", [False, True], ids=["generating", "draining"])
@pytest.mark.parametrize("monitor", ["watchdog", "fleet", "telemetry", "checkpoint"])
@pytest.mark.parametrize("finalize_fails", [False, True])
def test_monitor_failure_reaches_caller(
    monitor: str, draining: bool, finalize_fails: bool
) -> None:
    async def _scenario() -> None:
        ctrl = _bare_actor()
        if finalize_fails:
            ctrl._finish_pending_save.side_effect = OSError("checkpoint publish failed")
        release = asyncio.Event()
        error = (
            RolloutStall("watchdog failed")
            if monitor == "watchdog"
            else RuntimeError(f"{monitor} failed")
        )

        async def _fail() -> None:
            await release.wait()
            raise error

        if monitor == "watchdog":
            ctrl._stall_watchdog_pump = _fail
        elif monitor == "fleet":
            ctrl._gen_fleet = object()
            ctrl._gen_fleet_probe_pump = _fail
        elif monitor == "telemetry":
            ctrl._master_config.rollout_checkpointing.telemetry_interval_s = 1.0
            ctrl._rollout_telemetry_pump = _fail
        else:
            ctrl._master_config.rollout_checkpointing.snapshot_attempt_interval_s = 1.0
            ctrl._rollout_checkpoint_pump = _fail

        run_task = await _start_run(ctrl, draining=draining)
        release.set()
        with pytest.raises(type(error)) as caught:
            await asyncio.wait_for(run_task, timeout=1)
        assert caught.value is error
        _assert_cleanup(ctrl)

    asyncio.run(_scenario())


@pytest.mark.parametrize("pump", ["rollout", "train"])
def test_pump_failure_reaches_caller(pump: str) -> None:
    async def _scenario() -> None:
        ctrl = _bare_actor()
        release = asyncio.Event()
        error = RuntimeError(f"{pump} failed")

        async def _fail() -> None:
            await release.wait()
            raise error

        if pump == "rollout":
            ctrl._rollout_pump = _fail
        else:
            ctrl._train_pump = _fail
        run_task = await _start_run(ctrl, draining=False)
        release.set()
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(run_task, timeout=1)
        assert caught.value is error
        _assert_cleanup(ctrl)

    asyncio.run(_scenario())


@pytest.mark.parametrize("draining", [False, True], ids=["generating", "draining"])
@pytest.mark.parametrize("requested_stop", [False, True])
def test_checkpoint_completion_requires_stop_request(
    draining: bool, requested_stop: bool
) -> None:
    async def _scenario() -> None:
        ctrl = _bare_actor()
        ctrl._master_config.rollout_checkpointing.snapshot_attempt_interval_s = 1.0
        release = asyncio.Event()

        async def _checkpoint() -> None:
            await release.wait()
            if requested_stop:
                ctrl._rollout_checkpoint_stop_requested.set()

        ctrl._rollout_checkpoint_pump = _checkpoint
        run_task = await _start_run(ctrl, draining=draining)
        release.set()
        if requested_stop:
            assert await asyncio.wait_for(run_task, timeout=1) == {
                "train_steps": 7,
                "trainer_version": 7,
            }
        else:
            with pytest.raises(RuntimeError, match="without requesting stop"):
                await asyncio.wait_for(run_task, timeout=1)
        _assert_cleanup(ctrl)

    asyncio.run(_scenario())


@pytest.mark.parametrize(
    "draining", [False, True], ids=["train-first", "rollout-first"]
)
def test_training_completion_returns_summary_and_cancels_monitors(
    draining: bool,
) -> None:
    async def _scenario() -> None:
        ctrl = _bare_actor()
        finish_training = asyncio.Event()
        ctrl._train_pump = finish_training.wait
        run_task = await _start_run(ctrl, draining=draining)
        finish_training.set()
        assert await asyncio.wait_for(run_task, timeout=1) == {
            "train_steps": 7,
            "trainer_version": 7,
        }
        _assert_cleanup(ctrl)

    asyncio.run(_scenario())


@pytest.mark.parametrize("completion", ["checkpoint-stop", "training-done"])
def test_normal_completion_does_not_hide_simultaneous_watchdog_failure(
    completion: str,
) -> None:
    async def _scenario() -> None:
        ctrl = _bare_actor()
        release = asyncio.Event()

        async def _checkpoint() -> None:
            await release.wait()
            ctrl._rollout_checkpoint_stop_requested.set()

        async def _fail() -> None:
            await release.wait()
            raise RolloutStall("concurrent stall")

        if completion == "checkpoint-stop":
            ctrl._master_config.rollout_checkpointing.snapshot_attempt_interval_s = 1.0
            ctrl._rollout_checkpoint_pump = _checkpoint
        else:
            ctrl._train_pump = release.wait
        ctrl._stall_watchdog_pump = _fail
        run_task = await _start_run(ctrl, draining=True)
        release.set()
        with pytest.raises(RolloutStall, match="concurrent stall"):
            await asyncio.wait_for(run_task, timeout=1)
        _assert_cleanup(ctrl)

    asyncio.run(_scenario())


@pytest.mark.parametrize("pump_fails_on_cancel", [False, True])
def test_external_cancellation_stops_pumps_and_runs_cleanup(
    pump_fails_on_cancel: bool,
) -> None:
    async def _scenario() -> None:
        ctrl = _bare_actor()
        if pump_fails_on_cancel:

            async def _fail_on_cancel() -> None:
                try:
                    await _blocked()
                finally:
                    raise RuntimeError("train pump failed during cancellation")

            ctrl._train_pump = _fail_on_cancel
        run_task = await _start_run(ctrl, draining=True)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        _assert_cleanup(ctrl)

    asyncio.run(_scenario())
