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

"""Watchdog: the last line of defence for failures nothing else catches.

Every other guard in this phase reacts to something raising. The wedge described in the
resiliency report raises nothing at all -- rollouts sit in NeMo-Gym's uncapped retry loop
while the train pump spins -- so the only way to see it is to notice that committed
groups stopped moving while rollouts are still in flight.

Progress is measured by the committed counter rather than a timestamp because that is the
property that matters: "no group has landed" is the symptom, whatever the cause.
"""

import asyncio
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.experience.failures import RolloutStall
from nemo_rl.experience.rollout_manager import RolloutStats


class _RecordingLogger:
    def __init__(self) -> None:
        self.metrics: list[dict] = []

    def log_metrics(self, metrics, step=0, prefix="", **kwargs) -> None:
        del step, prefix, kwargs
        self.metrics.append(dict(metrics))


def _make_controller(
    *,
    stats: RolloutStats,
    inflight: int,
    stall_timeout_s: float,
    stall_action: str = "warn",
    gym_subprocess_check: bool = False,
    env_handles=None,
    train_steps: int = 0,
    max_num_steps: int = 100,
):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = SimpleNamespace(
        watchdog=SimpleNamespace(
            # Tiny tick so the loop runs immediately; the stall threshold is what the
            # tests actually vary.
            interval_s=0.001,
            stall_timeout_s=stall_timeout_s,
            stall_action=stall_action,
            gym_subprocess_check=gym_subprocess_check,
        )
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(max_num_steps=max_num_steps)
    )
    ctrl._rollout_manager = SimpleNamespace(stats=stats)
    ctrl._inflight_rollouts = inflight
    ctrl._train_steps = train_steps
    ctrl._logger = _RecordingLogger()
    ctrl._env_handles = env_handles if env_handles is not None else {}
    return ctrl


async def _run_ticks(ctrl, ticks: int):
    """Run the watchdog for a bounded number of ticks, then cancel it."""
    task = asyncio.ensure_future(ctrl._watchdog_pump())
    # Each tick sleeps interval_s (1ms); give it room for `ticks` of them.
    await asyncio.sleep(0.005 * ticks)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task


class TestStallDetection:
    def test_no_stall_is_reported_while_groups_keep_landing(self):
        stats = RolloutStats()

        async def _main():
            ctrl = _make_controller(stats=stats, inflight=4, stall_timeout_s=0.0)
            task = asyncio.ensure_future(ctrl._watchdog_pump())
            for _ in range(5):
                await asyncio.sleep(0.003)
                stats.committed += 1  # progress on every tick
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # stall_timeout_s=0 would fire instantly if progress were not being seen.
        asyncio.run(_main())

    def test_no_progress_while_work_remains_aborts_when_configured(self):
        stats = RolloutStats()
        ctrl = _make_controller(
            stats=stats, inflight=8, stall_timeout_s=0.0, stall_action="abort"
        )
        with pytest.raises(RolloutStall, match="8 rollouts in flight"):
            asyncio.run(ctrl._watchdog_pump())

    def test_a_wedge_with_nothing_in_flight_is_still_a_stall(self):
        """Regression guard for the gap a fault-injection run walked straight through.

        Killing a generation worker wedged the loop with zero rollouts in flight and
        zero failures recorded: the rollout pump sat on backpressure behind a train
        pump that could no longer finish a step, so there was nothing in flight to
        count. The earlier `inflight > 0` condition meant the watchdog watched six
        minutes of idleness and said nothing.
        """
        stats = RolloutStats()
        stats.committed = 10  # groups landed before the wedge, then stopped
        ctrl = _make_controller(
            stats=stats,
            inflight=0,
            stall_timeout_s=0.0,
            stall_action="abort",
            train_steps=4,
            max_num_steps=50,
        )
        with pytest.raises(RolloutStall, match="0 rollouts in flight"):
            asyncio.run(ctrl._watchdog_pump())

    def test_warn_mode_reports_without_ending_the_run(self, capsys):
        stats = RolloutStats()
        ctrl = _make_controller(
            stats=stats, inflight=3, stall_timeout_s=0.0, stall_action="warn"
        )
        asyncio.run(_run_ticks(ctrl, 3))
        assert "rollout stall" in capsys.readouterr().out

    def test_a_finished_run_is_not_a_stall(self):
        """With every step done there is nothing left to wait for."""
        stats = RolloutStats()
        ctrl = _make_controller(
            stats=stats,
            inflight=0,
            stall_timeout_s=0.0,
            stall_action="abort",
            train_steps=50,
            max_num_steps=50,
        )
        asyncio.run(_run_ticks(ctrl, 3))

    def test_train_step_progress_counts_even_without_new_commits(self):
        """A step draining already-buffered groups is progress, not a stall."""
        stats = RolloutStats()

        async def _main():
            # Threshold comfortably above the progress cadence below, so only a real
            # gap in progress can trip it.
            ctrl = _make_controller(
                stats=stats,
                inflight=0,
                stall_timeout_s=0.05,
                stall_action="abort",
                max_num_steps=100,
            )
            task = asyncio.ensure_future(ctrl._watchdog_pump())
            for _ in range(5):
                await asyncio.sleep(0.003)
                ctrl._train_steps += 1  # commits frozen, steps advancing
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_main())


class TestMetrics:
    def test_rollout_counters_and_inflight_are_published(self):
        stats = RolloutStats()
        stats.committed = 7
        stats.record_redispatch("GenerationUnavailable")
        ctrl = _make_controller(stats=stats, inflight=2, stall_timeout_s=1000.0)

        asyncio.run(_run_ticks(ctrl, 2))

        assert ctrl._logger.metrics, "the watchdog must publish something"
        published = ctrl._logger.metrics[-1]
        assert published["rollout/committed_total"] == 7.0
        assert published["rollout/redispatch_total"] == 1.0
        assert published["rollout/inflight"] == 2.0
        # The leading indicator: idle time rises before a wedge becomes a stall.
        assert "rollout/idle_s" in published


class TestEnvHealthCheck:
    def test_a_healthy_environment_passes(self):
        calls = []

        class _Handle:
            health_check = SimpleNamespace(
                remote=lambda: _completed(calls.append("checked"))
            )

        ctrl = _make_controller(
            stats=RolloutStats(),
            inflight=0,
            stall_timeout_s=1000.0,
            gym_subprocess_check=True,
            env_handles={"nemo_gym": _Handle()},
        )
        asyncio.run(_run_ticks(ctrl, 2))
        assert calls, "health_check should have been polled"

    def test_an_unhealthy_environment_is_named_in_the_error(self):
        """Gym's poll() names the dead process; the env name says which actor it was."""

        class _Handle:
            health_check = SimpleNamespace(
                remote=lambda: _failed(
                    RuntimeError("Process `workplace_assistant` finished unexpectedly!")
                )
            )

        ctrl = _make_controller(
            stats=RolloutStats(),
            inflight=0,
            stall_timeout_s=1000.0,
            stall_action="abort",
            gym_subprocess_check=True,
            env_handles={"nemo_gym": _Handle()},
        )
        with pytest.raises(RuntimeError, match="'nemo_gym' reported unhealthy"):
            asyncio.run(ctrl._watchdog_pump())

    def test_an_unhealthy_environment_only_warns_under_the_default_action(self, capsys):
        """stall_action="warn" promises to "only report". It has to mean that here too.

        This path raised unconditionally, so with gym_subprocess_check defaulting to
        true, an unhealthy environment ended the run under the documented default -- a
        run-killing path switched on by default in a feature that is meant to be
        inert until configured.
        """

        class _Handle:
            health_check = SimpleNamespace(
                remote=lambda: _failed(RuntimeError("subprocess died"))
            )

        ctrl = _make_controller(
            stats=RolloutStats(),
            inflight=0,
            stall_timeout_s=1000.0,
            stall_action="warn",
            gym_subprocess_check=True,
            env_handles={"nemo_gym": _Handle()},
        )
        # Must complete its ticks rather than blowing up.
        asyncio.run(_run_ticks(ctrl, 2))
        out = capsys.readouterr().out
        assert "environment health" in out
        assert "nemo_gym" in out

    def test_a_wedged_environment_does_not_stop_the_pump(self):
        """The failure this whole check exists for must not disable the check.

        NemoGym is an asyncio actor: a wedged one never answers, and an unbounded await
        meant the pump stopped ticking and stall detection died exactly when it was
        needed. A probe that does not answer within a tick IS the unhealthy signal.
        """
        never_resolves: list[asyncio.Future] = []

        def _hang():
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            never_resolves.append(future)
            return future

        class _Handle:
            health_check = SimpleNamespace(remote=_hang)

        ctrl = _make_controller(
            stats=RolloutStats(),
            inflight=0,
            stall_timeout_s=1000.0,
            stall_action="warn",
            gym_subprocess_check=True,
            env_handles={"nemo_gym": _Handle()},
        )
        asyncio.run(_run_ticks(ctrl, 4))
        # The pump kept ticking despite the environment never answering.
        assert len(ctrl._logger.metrics) >= 2, (
            "watchdog stopped ticking while an environment was wedged"
        )
        assert never_resolves, "the health check was never actually polled"

    def test_environments_without_a_health_check_are_skipped(self):
        """Only NeMo-Gym has subprocess servers to lose; math envs must not trip this."""
        ctrl = _make_controller(
            stats=RolloutStats(),
            inflight=0,
            stall_timeout_s=1000.0,
            gym_subprocess_check=True,
            env_handles={"math": SimpleNamespace()},
        )
        asyncio.run(_run_ticks(ctrl, 2))

    def test_the_check_can_be_disabled(self):
        class _Handle:
            health_check = SimpleNamespace(
                remote=lambda: _failed(RuntimeError("would fail if polled"))
            )

        ctrl = _make_controller(
            stats=RolloutStats(),
            inflight=0,
            stall_timeout_s=1000.0,
            gym_subprocess_check=False,
            env_handles={"nemo_gym": _Handle()},
        )
        asyncio.run(_run_ticks(ctrl, 2))


def _completed(_value=None):
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    future.set_result(None)
    return future


def _failed(error: BaseException):
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    future.set_exception(error)
    return future
