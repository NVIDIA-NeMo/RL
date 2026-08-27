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
    GenerationFleetHealth,
    ShardState,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.metrics: list[dict] = []
        self.step_metrics: list[str | None] = []

    def log_metrics(
        self, metrics, step=0, prefix="", step_metric=None, **kwargs
    ) -> None:
        del step, prefix, kwargs
        self.metrics.append(dict(metrics))
        self.step_metrics.append(step_metric)


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
    watchdog_interval_s: float = 0.001,
    probe_interval_s: float = 0.001,
):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = SimpleNamespace(
        stall_watchdog=SimpleNamespace(
            # Tiny tick so the loop runs immediately; the stall threshold is what the
            # tests actually vary.
            interval_s=watchdog_interval_s,
            stall_timeout_s=stall_timeout_s,
            stall_action=stall_action,
            gym_subprocess_check=gym_subprocess_check,
        ),
        generation_fleet_health=SimpleNamespace(
            probe_timeout_s=1.0, probe_interval_s=probe_interval_s
        ),
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(max_num_steps=max_num_steps)
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._rollout_manager = SimpleNamespace(stats=stats)
    ctrl._inflight_rollouts = inflight
    ctrl._train_steps = train_steps
    ctrl._logger = _RecordingLogger()
    ctrl._env_handles = env_handles if env_handles is not None else {}
    # These tests cover stall detection, not fleet health or gym routing.
    ctrl._gen_fleet = None
    ctrl._generation_router = None
    return ctrl


async def _run_pump(pump, ticks: int):
    """Run one pump coroutine for a bounded number of ticks, then cancel it."""
    task = asyncio.ensure_future(pump)
    # Each tick sleeps its interval (1ms); give it room for `ticks` of them.
    await asyncio.sleep(0.005 * ticks)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task


async def _run_ticks(ctrl, ticks: int):
    """Run the watchdog for a bounded number of ticks, then cancel it."""
    return await _run_pump(ctrl._stall_watchdog_pump(), ticks)


async def _run_probe_ticks(ctrl, ticks: int):
    """Run the fleet probe pump for a bounded number of ticks, then cancel it."""
    return await _run_pump(ctrl._gen_fleet_probe_pump(), ticks)


class TestStallDetection:
    def test_no_stall_is_reported_while_groups_keep_landing(self):
        stats = RolloutStats()

        async def _main():
            ctrl = _make_controller(stats=stats, inflight=4, stall_timeout_s=0.0)
            task = asyncio.ensure_future(ctrl._stall_watchdog_pump())
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
            asyncio.run(ctrl._stall_watchdog_pump())

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
            asyncio.run(ctrl._stall_watchdog_pump())

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
            task = asyncio.ensure_future(ctrl._stall_watchdog_pump())
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

    def test_ticks_never_name_the_committed_step(self):
        """A tick naming _train_steps is dropped: _train_pump already committed it."""
        ctrl = _make_controller(
            stats=RolloutStats(), inflight=0, stall_timeout_s=1000.0
        )

        asyncio.run(_run_ticks(ctrl, 2))

        assert ctrl._logger.step_metrics, "the watchdog never ticked"
        assert set(ctrl._logger.step_metrics) == {"rollout/train_steps"}
        # The no-step branch needs the key in the payload too, not just the argument.
        assert all("rollout/train_steps" in m for m in ctrl._logger.metrics)


class TestGenerationFleetProbe:
    """The probe is the proactive half; the routing adapters supply the reactive half.

    Ray liveness is cheap and authoritative for "the process is gone", which is what
    this checks. It does not catch a vLLM engine core dying under a live worker, which
    is why observed failures are reported separately into the same counters.
    """

    @staticmethod
    def _with_fleet(monitor, *, worker_alive, **kwargs):
        ctrl = _make_controller(
            stats=RolloutStats(), inflight=0, stall_timeout_s=1000.0, **kwargs
        )
        ctrl._gen_fleet = monitor
        # A confirmed death stands the trainers' refit deadline down, so the fixture has to
        # carry the trainer handle it fans out over. Without it the probe's except clause
        # would swallow an AttributeError and the feature would silently no-op -- which is
        # how a wrong attribute name survived until a test looked for the calls.
        ctrl._stood_down = []
        ctrl._trainer = SimpleNamespace(
            worker_group=SimpleNamespace(
                workers=[
                    SimpleNamespace(
                        stand_down_refit_watchdog=SimpleNamespace(
                            remote=(lambda i=i: ctrl._stood_down.append(i))
                        )
                    )
                    for i in range(2)
                ]
            )
        )
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
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, True])
        asyncio.run(_run_probe_ticks(ctrl, 3))
        assert monitor.serving_shards() == [0, 1]

    def test_a_dead_worker_is_quarantined_by_the_probe(self):
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, False])
        asyncio.run(_run_probe_ticks(ctrl, 3))
        assert monitor.state_of(1) is ShardState.DEAD
        assert monitor.serving_shards() == [0]

    def test_a_dead_actor_is_conclusive_on_the_first_round(self):
        """A RayActorError is proof, not another ambiguous data point.

        A probe timeout cannot tell a slow shard from a dead one, which is what the
        counters are for. Ray reporting the actor dead can, so waiting for
        unhealthy_threshold rounds only delays the verdict -- and the refit deadline can
        expire inside that delay, which is exactly how job 5925668 aborted a hung refit
        while the corpse was still SUSPECT. A threshold of 99 makes counting impossible.
        """
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=99)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, False])
        asyncio.run(_run_probe_ticks(ctrl, 2))
        assert monitor.state_of(1) is ShardState.DEAD
        assert monitor.absent_shards() == [1]

    def test_a_confirmed_death_stands_the_trainers_deadline_down(self):
        """A dead peer needs NCCL's own error path, not the refit deadline.

        Job 6405953 recovered the reshard kill variant with RefitAborted appearing zero
        times, before any deadline existed. Once it existed it started winning that race:
        it aborts, sync_stream_within orphans kernels on the trainers' streams, and the
        rebuild that would have worked cannot. recovery-reshard-refit has failed
        continuously since job 6512153 for exactly that reason.
        """
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=99)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, False])
        asyncio.run(_run_probe_ticks(ctrl, 1))
        assert ctrl._stood_down == [0, 1], (
            "every policy worker must be told to stand its refit deadline down once a "
            f"generation shard is confirmed gone; saw {ctrl._stood_down}"
        )

    def test_a_timeout_leaves_the_deadline_armed(self):
        """The frozen case, and the reason this is keyed on DEATH rather than silence.

        A frozen rank is alive and simply stops answering. No death is ever recorded, the
        deadline still fires, and the run still ends attributably -- which is the only
        outcome available on the reshard transport once the bulk transfer has aborted.
        """
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=99)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, True])
        ctrl._async_cfg.generation_fleet_health.probe_timeout_s = 0.001
        ctrl._gen.worker_group.workers[1].is_alive.remote = lambda: asyncio.sleep(10.0)
        asyncio.run(_run_probe_ticks(ctrl, 2))
        assert ctrl._stood_down == [], (
            "a probe timeout is not proof of death; standing the deadline down on one "
            "would leave a frozen rank able to wedge the run forever"
        )

    def test_a_timeout_is_still_counted_rather_than_trusted(self):
        """The other half: an ambiguous failure must keep its benefit of the doubt.

        A shard busy inside a refit stops answering probes without being dead. Treating
        that like proof of death would condemn a healthy fleet on one slow round.
        """
        monitor = GenerationFleetHealth(
            shard_count=1, policy=FleetHealthPolicy(unhealthy_threshold=99)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True])
        ctrl._async_cfg.generation_fleet_health.probe_timeout_s = 0.001
        ctrl._gen.worker_group.workers = [
            SimpleNamespace(
                is_alive=SimpleNamespace(remote=lambda: asyncio.sleep(10.0))
            )
        ]
        asyncio.run(_run_probe_ticks(ctrl, 3))
        assert monitor.state_of(0) is ShardState.SUSPECT
        assert monitor.absent_shards() == []

    def test_fleet_state_is_published(self):
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(monitor, worker_alive=[True, False])

        async def _main():
            await _run_probe_ticks(ctrl, 3)
            # The watchdog still owns publishing; it reports whatever the probe found.
            await _run_ticks(ctrl, 2)

        asyncio.run(_main())
        published = ctrl._logger.metrics[-1]
        assert published["gen_fleet/shards/dead"] == 1.0
        assert published["gen_fleet/serving_shards"] == 1.0

    def test_losing_the_whole_fleet_ends_the_run(self):
        """Below the floor there is nothing left to generate with."""
        monitor = GenerationFleetHealth(
            shard_count=1,
            policy=FleetHealthPolicy(unhealthy_threshold=1, min_healthy_shards=1),
        )
        ctrl = self._with_fleet(monitor, worker_alive=[False])

        async def _main():
            await _run_probe_ticks(ctrl, 3)
            await ctrl._stall_watchdog_pump()

        with pytest.raises(GenerationFleetExhausted):
            asyncio.run(_main())

    def test_a_tick_inside_a_recovery_does_not_end_the_run(self):
        """Regression: the exhaustion check killed the recovery that was about to succeed.

        _recover_from_failed_refit marks every serving shard partial, so the serving set
        is empty by construction until the retry refit promotes them back -- across two
        awaits that yield this loop. A watchdog tick landing in that window used to see
        zero serving shards, raise GenerationFleetExhausted, and end the run while the
        retry was still in flight, blaming an exhausted fleet for a recovery in progress.

        The default tick is 30s and a rebuild plus a full refit can easily outlast it.
        """
        monitor = GenerationFleetHealth(
            shard_count=1,
            policy=FleetHealthPolicy(unhealthy_threshold=1, min_healthy_shards=1),
        )
        ctrl = self._with_fleet(monitor, worker_alive=[False])

        async def _main():
            await _run_probe_ticks(ctrl, 3)
            with ctrl._recovery_window():
                # _run_ticks, not the raw pump: the pump is `while True` and only ever
                # returns by raising, so awaiting it directly here would hang forever --
                # which is precisely what the fix makes it do, since the exhaustion check
                # it used to raise from is now skipped inside this window.
                await _run_ticks(ctrl, 3)

        asyncio.run(_main())  # must not raise

    def test_the_window_reopens_the_check_even_if_the_retry_raises(self):
        """A leaked flag would disable the exhaustion check for the rest of the run."""
        monitor = GenerationFleetHealth(
            shard_count=1,
            policy=FleetHealthPolicy(unhealthy_threshold=1, min_healthy_shards=1),
        )
        ctrl = self._with_fleet(monitor, worker_alive=[False])

        with pytest.raises(RuntimeError, match="retry blew up"):
            with ctrl._recovery_window():
                raise RuntimeError("retry blew up")
        assert ctrl._recovering_from_refit is False

        async def _main():
            await _run_probe_ticks(ctrl, 3)
            await ctrl._stall_watchdog_pump()

        with pytest.raises(GenerationFleetExhausted):
            asyncio.run(_main())

    def test_no_monitor_means_no_probing(self):
        """Fleet health off must leave the watchdog exactly as it was."""
        ctrl = _make_controller(
            stats=RolloutStats(), inflight=0, stall_timeout_s=1000.0
        )
        ctrl._gen = SimpleNamespace()  # would AttributeError if probed
        asyncio.run(_run_ticks(ctrl, 3))

    def test_the_watchdog_no_longer_probes(self):
        """The probe must not ride the watchdog's clock.

        Sharing it made probe_interval_s decorative: probes ran at watchdog.interval_s,
        so with the shipped defaults detection took unhealthy_threshold * 30s = 60-90s
        rather than the ~15s the config documents. That is longer than the refit
        deadline, so a refit hung on a dead rank always aborted before the monitor knew
        which rank to drop, and the rebuild the abort exists to trigger saw an empty
        absent set. Job 5925668.

        The watchdog ticks fast here *on purpose*: a slow interval would make this pass
        because the loop never ran, which is no evidence at all.
        """
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(
            monitor, worker_alive=[True, False], watchdog_interval_s=0.001
        )
        asyncio.run(_run_ticks(ctrl, 5))
        # The watchdog demonstrably ran -- it published -- and still did not probe.
        assert ctrl._logger.metrics, (
            "the watchdog never ticked; the test proves nothing"
        )
        assert monitor.state_of(1) is ShardState.HEALTHY

    def test_the_probe_runs_on_its_own_interval(self):
        """The other half: a slow watchdog must not slow detection down."""
        monitor = GenerationFleetHealth(
            shard_count=2, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        ctrl = self._with_fleet(
            monitor,
            worker_alive=[True, False],
            watchdog_interval_s=30.0,
            probe_interval_s=0.001,
        )
        asyncio.run(_run_probe_ticks(ctrl, 3))
        assert monitor.state_of(1) is ShardState.DEAD

    def test_shards_are_probed_concurrently(self):
        """A round must cost one probe_timeout_s, not one per shard.

        Sequentially, a fleet larger than probe_interval_s / probe_timeout_s could never
        complete a round within its own interval -- and the config validator only checks
        those two against each other, silently assuming a single probe per tick.
        """
        monitor = GenerationFleetHealth(
            shard_count=4, policy=FleetHealthPolicy(unhealthy_threshold=1)
        )
        started = asyncio.Event()
        concurrent = 0
        peak = 0

        async def _slow():
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            started.set()
            try:
                await asyncio.sleep(0.05)
            finally:
                concurrent -= 1

        ctrl = self._with_fleet(monitor, worker_alive=[True] * 4)
        ctrl._gen.worker_group.workers = [
            SimpleNamespace(is_alive=SimpleNamespace(remote=lambda: _slow()))
            for _ in range(4)
        ]

        async def _main():
            task = asyncio.ensure_future(ctrl._probe_generation_fleet())
            await started.wait()
            await asyncio.sleep(0.01)
            observed = peak
            await task
            return observed

        assert asyncio.run(_main()) == 4


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
            asyncio.run(ctrl._stall_watchdog_pump())

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
