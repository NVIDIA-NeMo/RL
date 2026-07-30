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
import ray.exceptions

from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.experience.failures import RolloutStall
from nemo_rl.experience.rollout_manager import RolloutStats
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetExhausted,
    GenerationFleetMonitor,
    ShardState,
)


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
        ),
        fleet_health=SimpleNamespace(probe_timeout_s=1.0),
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(max_num_steps=max_num_steps)
    )
    ctrl._rollout_manager = SimpleNamespace(stats=stats)
    ctrl._inflight_rollouts = inflight
    ctrl._train_steps = train_steps
    ctrl._logger = _RecordingLogger()
    ctrl._env_handles = env_handles if env_handles is not None else {}
    # These tests cover stall detection, not fleet health.
    ctrl._fleet_monitor = None
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


class TestGenerationFleetProbe:
    """The probe is the proactive half; the routing adapters supply the reactive half.

    Ray liveness is cheap and authoritative for "the process is gone", which is what
    this checks. It does not catch a vLLM engine core dying under a live worker, which
    is why observed failures are reported separately into the same counters.
    """

    @staticmethod
    def _with_fleet(monitor, *, worker_alive):
        ctrl = _make_controller(
            stats=RolloutStats(), inflight=0, stall_timeout_s=1000.0
        )
        ctrl._fleet_monitor = monitor
        ctrl._gen = SimpleNamespace(
            worker_group=SimpleNamespace(
                get_dp_leader_worker_idx=lambda shard: shard,
                workers=[
                    SimpleNamespace(
                        is_alive=SimpleNamespace(
                            remote=(lambda alive=alive: _completed())
                            if alive
                            else (lambda: _failed(ray.exceptions.ActorDiedError()))
                        )
                    )
                    for alive in worker_alive
                ],
            )
        )
        return ctrl

    def test_a_live_fleet_stays_serving(self):
        monitor = GenerationFleetMonitor(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, True])
        asyncio.run(_run_ticks(ctrl, 3))
        assert monitor.serving_shards() == [0, 1]

    def test_a_dead_worker_is_quarantined_by_the_probe(self):
        monitor = GenerationFleetMonitor(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, False])
        asyncio.run(_run_ticks(ctrl, 3))
        assert monitor.state_of(1) is ShardState.DEAD
        assert monitor.serving_shards() == [0]

    def test_fleet_state_is_published(self):
        monitor = GenerationFleetMonitor(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, False])
        asyncio.run(_run_ticks(ctrl, 3))
        published = ctrl._logger.metrics[-1]
        assert published["fleet/shards/dead"] == 1.0
        assert published["fleet/serving_shards"] == 1.0

    def test_losing_the_whole_fleet_ends_the_run(self):
        """Below the floor there is nothing left to generate with."""
        monitor = GenerationFleetMonitor(
            shard_count=1,
            policy=FleetHealthPolicy(unhealthy_threshold=1, min_healthy_shards=1),
        )
        ctrl = self._with_fleet(monitor, worker_alive=[False])
        with pytest.raises(GenerationFleetExhausted):
            asyncio.run(ctrl._watchdog_pump())

    def test_no_monitor_means_no_probing(self):
        """Fleet health off must leave the watchdog exactly as it was."""
        ctrl = _make_controller(
            stats=RolloutStats(), inflight=0, stall_timeout_s=1000.0
        )
        ctrl._gen = SimpleNamespace()  # would AttributeError if probed
        asyncio.run(_run_ticks(ctrl, 3))


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
            gym_subprocess_check=True,
            env_handles={"nemo_gym": _Handle()},
        )
        with pytest.raises(RuntimeError, match="'nemo_gym' reported unhealthy"):
            asyncio.run(ctrl._watchdog_pump())

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
