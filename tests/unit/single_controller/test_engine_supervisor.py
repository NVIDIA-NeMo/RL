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

"""Restarting dead generation shards and handing them back to the refit path.

The handover is entirely through fleet-health states, so that is what these pin:

    DEAD --(restart starts)--> RESTARTING --(engine up)--> STALE --(next refit)--> HEALTHY

Getting a state wrong here is not loud. Landing in HEALTHY instead of STALE would put a
shard holding disk weights straight back into rollouts; staying in RESTARTING would keep
it out of the refit that is supposed to fix it.
"""

import asyncio

from nemo_rl.models.generation.engine_supervisor import EngineSupervisor
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetHealth,
    ShardState,
)


def _monitor(shard_count=3, **policy_kwargs) -> GenerationFleetHealth:
    return GenerationFleetHealth(
        shard_count=shard_count,
        policy=FleetHealthPolicy(**policy_kwargs),
        base_urls=[f"http://h:{8000 + i}/v1" for i in range(shard_count)],
    )


def _condemn(monitor, shard_idx, policy=None):
    policy = policy or FleetHealthPolicy()
    for _ in range(policy.unhealthy_threshold):
        monitor.report_failure(shard_idx, RuntimeError("actor died"))
    assert monitor.state_of(shard_idx) is ShardState.DEAD


class _Generation:
    """Records restarts; the replacement reports a new URL, as a real one would."""

    def __init__(self, *, fail=False, block=None):
        self.restarted = []
        self.gpu_reads = []
        self.fail = fail
        self._block = block

    def log_shard_gpu_state(self, shard_idx, *, label, timeout_s=30.0):
        del timeout_s
        self.gpu_reads.append((shard_idx, label))

    def restart_shard(self, shard_idx):
        if self._block is not None:
            self._block.wait()
        self.restarted.append(shard_idx)
        if self.fail:
            raise RuntimeError("engine did not come up")
        return f"http://h:{9000 + shard_idx}/v1"


async def _tick_and_settle(supervisor):
    supervisor.tick()
    await supervisor.drain(timeout_s=5)
    # to_thread completions land on the loop; give the callbacks a turn.
    await asyncio.sleep(0)


class TestWhichShardsGetRestarted:
    def test_a_healthy_fleet_restarts_nothing(self):
        monitor, gen = _monitor(), _Generation()
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert gen.restarted == []

    def test_a_dead_shard_is_restarted(self):
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 1)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert gen.restarted == [1]

    def test_a_suspect_shard_is_not_restarted(self):
        """Restarting on a single failed probe would cost minutes of reload for a blip."""
        monitor, gen = _monitor(), _Generation()
        monitor.record_probe(0, ok=False, error="timeout")
        assert monitor.state_of(0) is ShardState.SUSPECT
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert gen.restarted == []

    def test_a_shard_already_restarting_is_not_restarted_again(self):
        """tick() runs on every watchdog beat; a restart takes minutes."""
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 2)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        async def _main():
            supervisor.tick()
            supervisor.tick()
            supervisor.tick()
            await supervisor.drain(timeout_s=5)

        asyncio.run(_main())

        assert gen.restarted == [2]


class TestStateHandover:
    def test_a_restarted_shard_lands_in_stale_not_healthy(self):
        """STALE is what keeps disk weights out of rollouts until a refit lands."""
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert monitor.state_of(0) is ShardState.STALE

    def test_a_stale_shard_is_present_for_the_refit_but_not_serving(self):
        """The two facts that together make re-admission work."""
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert 0 not in monitor.absent_shards(), "must join the next refit"
        assert 0 not in monitor.serving_shards(), "must not take traffic yet"

    def test_a_refit_returns_it_to_service(self):
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)
        asyncio.run(_tick_and_settle(supervisor))

        monitor.report_refit(0, weight_version=7)

        assert monitor.state_of(0) is ShardState.HEALTHY
        assert 0 in monitor.serving_shards()

    def test_the_replacements_url_replaces_the_dead_one(self):
        """A new engine binds a new port; the router is fed from these URLs."""
        monitor, gen = _monitor(), _Generation()
        old_url = monitor.snapshot()[1].base_url
        _condemn(monitor, 1)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))
        monitor.report_refit(1, weight_version=1)

        assert monitor.snapshot()[1].base_url != old_url
        assert monitor.snapshot()[1].base_url in monitor.serving_base_urls()

    def test_probe_history_does_not_survive_the_restart(self):
        """Otherwise one unlucky probe re-condemns a fresh engine immediately."""
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)
        asyncio.run(_tick_and_settle(supervisor))

        monitor.record_probe(0, ok=False, error="one blip")

        assert monitor.state_of(0) is not ShardState.DEAD

    def test_a_replacement_that_dies_again_is_restarted_again(self):
        """One restart per shard would make the second failure permanent."""
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 1)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)
        asyncio.run(_tick_and_settle(supervisor))
        assert monitor.state_of(1) is ShardState.STALE

        monitor.record_actor_death(1, error="replacement died")
        asyncio.run(_tick_and_settle(supervisor))

        assert gen.restarted == [1, 1]
        assert monitor.state_of(1) is ShardState.STALE


class TestFailedRestarts:
    def test_a_failed_restart_returns_the_shard_to_dead(self):
        """Left in RESTARTING it would never be retried and never be retired."""
        monitor, gen = _monitor(), _Generation(fail=True)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert monitor.state_of(0) is ShardState.DEAD
        assert supervisor.as_metrics()["gen_fleet/restarts_failed"] == 1.0

    def test_a_failed_restart_does_not_propagate(self):
        """A restart is best-effort; the run continues on the surviving shards."""
        monitor, gen = _monitor(), _Generation(fail=True)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))  # must not raise

    def test_attempts_are_capped_and_the_shard_is_retired(self):
        """A node that is never coming back must stop consuming restarts."""
        policy = FleetHealthPolicy(max_restart_attempts_per_shard=2)
        monitor = _monitor(shard_count=2, max_restart_attempts_per_shard=2)
        gen = _Generation(fail=True)
        # No cooldown: the subject here is the attempt cap, and the default backoff would
        # make every tick after the first a no-op. Spacing has its own tests.
        supervisor = EngineSupervisor(
            generation=gen, monitor=monitor, restart_backoff_s=0.0
        )

        async def _main():
            # Each round: the shard is DEAD, a restart is attempted, it fails, and the
            # shard returns to DEAD -- until the budget runs out and it is retired.
            for _ in range(5):
                if monitor.state_of(0) is ShardState.RETIRED:
                    break
                if monitor.state_of(0) is not ShardState.DEAD:
                    _condemn(monitor, 0, policy)
                await _tick_and_settle(supervisor)

        asyncio.run(_main())

        assert monitor.state_of(0) is ShardState.RETIRED
        # Not `len(...) <=`: that also holds at zero, which is the failure this exists to
        # catch. The fake appends before it raises, so every attempt is recorded.
        assert gen.restarted == [0] * policy.max_restart_attempts_per_shard


class TestARestartThatNeverReturns:
    """The failure the timeout exists for: nothing in the restart chain has a bound.

    ``restart_shard`` ends in `ray.get`s with no timeout, and `create_worker` returns
    before the actor is scheduled -- so a placement-group bundle that can never be filled,
    which is what a lost node looks like, blocks inside `post_init` rather than raising.
    Without a bound the `except` never fires: the shard sits in RESTARTING for the rest of
    the run, never retried because it is no longer DEAD, and never retired because
    retirement is driven by restart *attempts*.
    """

    def test_it_becomes_a_failed_restart_rather_than_a_permanent_restarting(self):
        import threading

        gate = threading.Event()
        monitor, gen = _monitor(), _Generation(block=gate)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(
            generation=gen, monitor=monitor, restart_timeout_s=0.05
        )

        async def _main():
            supervisor.tick()
            await supervisor.drain(timeout_s=5)
            await asyncio.sleep(0)

        try:
            asyncio.run(_main())
            assert monitor.state_of(0) is ShardState.DEAD
            assert supervisor.as_metrics()["gen_fleet/restarts_failed"] == 1.0
        finally:
            # Let the parked thread finish so it does not outlive the test.
            gate.set()

    def test_the_reason_records_the_timeout(self):
        import threading

        gate = threading.Event()
        monitor, gen = _monitor(), _Generation(block=gate)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(
            generation=gen, monitor=monitor, restart_timeout_s=0.05
        )

        async def _main():
            supervisor.tick()
            await supervisor.drain(timeout_s=5)
            await asyncio.sleep(0)

        try:
            asyncio.run(_main())
            assert "0.05" in monitor.snapshot()[0].last_error
        finally:
            gate.set()

    def test_the_worker_thread_is_a_daemon(self):
        """A non-daemon thread stuck in a reload is joined at interpreter exit, with no
        timeout, so it holds the whole process open. That is why this is not to_thread."""
        import threading

        gate = threading.Event()
        seen: list[threading.Thread] = []
        monitor = _monitor()
        _condemn(monitor, 0)

        class _Recording(_Generation):
            def restart_shard(self, shard_idx):
                seen.append(threading.current_thread())
                return super().restart_shard(shard_idx)

        gen = _Recording(block=gate)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        async def _main():
            supervisor.tick()
            gate.set()
            await supervisor.drain(timeout_s=5)

        asyncio.run(_main())

        assert seen and all(t.daemon for t in seen), (
            f"restart ran on non-daemon thread(s) {seen}; a reload that never returns "
            "would hang interpreter exit"
        )


class TestAttemptsAreSpacedOut:
    """Five attempts spent inside 25s is five attempts wasted on one transient cause.

    A failed restart returns the shard to DEAD and the next probe tick picks it straight
    back up, so without a cooldown the whole budget can burn before whatever broke the
    restart -- a GPU still held by an orphaned EngineCore, say, which took 370s to come
    back in job 6720618 -- has had any chance to clear.
    """

    def test_a_second_attempt_waits_for_the_backoff(self):
        monitor, gen = _monitor(), _Generation(fail=True)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(
            generation=gen, monitor=monitor, restart_backoff_s=1e6
        )

        async def _main():
            await _tick_and_settle(supervisor)
            await _tick_and_settle(supervisor)

        asyncio.run(_main())

        assert gen.restarted == [0], "the second tick must not spend another attempt"
        assert monitor.snapshot()[0].restart_attempts == 1

    def test_the_backoff_expires(self):
        monitor, gen = _monitor(), _Generation(fail=True)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(
            generation=gen, monitor=monitor, restart_backoff_s=0.0
        )

        async def _main():
            await _tick_and_settle(supervisor)
            await _tick_and_settle(supervisor)

        asyncio.run(_main())

        assert gen.restarted == [0, 0]

    def test_a_successful_restart_is_not_delayed_by_a_previous_failure(self):
        """The cooldown gates retries of a *failed* restart, not the first attempt."""
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 1)
        supervisor = EngineSupervisor(
            generation=gen, monitor=monitor, restart_backoff_s=1e6
        )

        asyncio.run(_tick_and_settle(supervisor))

        assert gen.restarted == [1]
        assert monitor.state_of(1) is ShardState.STALE


class TestItDoesNotBlockTheControlLoop:
    def test_tick_returns_before_the_restart_finishes(self):
        """A model reload takes minutes; the loop also drives rollouts and the watchdog."""
        import threading

        gate = threading.Event()
        monitor, gen = _monitor(), _Generation(block=gate)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        async def _main():
            supervisor.tick()
            # The restart is still blocked, yet control is back here.
            assert monitor.state_of(0) is ShardState.RESTARTING
            assert gen.restarted == []
            gate.set()
            await supervisor.drain(timeout_s=5)

        asyncio.run(_main())

        assert gen.restarted == [0]


def test_a_retired_shard_is_never_restarted():
    monitor, gen = _monitor(), _Generation()
    monitor.retire(0, reason="node gone")
    supervisor = EngineSupervisor(generation=gen, monitor=monitor)

    asyncio.run(_tick_and_settle(supervisor))

    assert gen.restarted == []
    assert monitor.state_of(0) is ShardState.RETIRED


class TestPromotionIsWiredUp:
    """The step that turns a restart into recovered throughput.

    Nothing except a completed refit moves a shard out of STALE, and the supervisor
    deliberately does not do it -- the refit has to have actually happened. So the
    controller must promote, and these run through the controller rather than calling
    report_refit by hand, which is what let this stay unwired: every earlier test in this
    file promoted manually and passed against a controller that never did.
    """

    @staticmethod
    def _controller(monitor, trainer_version=5):
        from nemo_rl.algorithms.single_controller import SingleControllerActor

        ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
        ctrl._gen_fleet = monitor
        ctrl._trainer_version = trainer_version
        return ctrl

    def test_a_restarted_shard_is_returned_to_service_after_a_refit(self):
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 1)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)
        asyncio.run(_tick_and_settle(supervisor))
        assert monitor.state_of(1) is ShardState.STALE

        ctrl = self._controller(monitor)
        ctrl._record_refit_landed(ctrl._refit_participants())

        assert monitor.state_of(1) is ShardState.HEALTHY
        assert 1 in monitor.serving_shards()

    def test_the_promoted_shard_carries_the_current_weight_version(self):
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)
        asyncio.run(_tick_and_settle(supervisor))

        ctrl = self._controller(monitor, trainer_version=11)
        ctrl._record_refit_landed(ctrl._refit_participants())

        assert monitor.snapshot()[0].weight_version == 11

    def test_a_suspect_shard_is_not_promoted_by_a_refit(self):
        """It took part in the refit, but it is failing probes for its own reasons and
        promoting it would reset the count that is meant to condemn it."""
        monitor = _monitor()
        monitor.record_probe(2, ok=False, error="timeout")
        assert monitor.state_of(2) is ShardState.SUSPECT

        ctrl = self._controller(monitor)
        ctrl._record_refit_landed(ctrl._refit_participants())

        assert monitor.state_of(2) is ShardState.SUSPECT

    def test_promotion_is_inert_without_fleet_health(self):
        ctrl = self._controller(None)
        ctrl._record_refit_landed(set())  # must not raise


class TestOnlyTheShardsThatWereRefitArePromoted:
    """A refit may only return to service the shards it actually wrote weights to.

    A restart takes minutes and nothing blocks it, so ``mark_loaded`` can turn a shard
    STALE *during* a refit whose membership was settled while that shard was still
    RESTARTING -- absent, and correctly left out of the communicator. Promoting on "is
    STALE" alone then returns a shard to service holding the checkpoint it read off disk.

    Before restart existed, nothing could turn a shard STALE mid-refit, so "is STALE" and
    "was in the refit group" described the same set and no check was needed.
    """

    @staticmethod
    def _controller(monitor, trainer_version=5):
        from nemo_rl.algorithms.single_controller import SingleControllerActor

        ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
        ctrl._gen_fleet = monitor
        ctrl._trainer_version = trainer_version
        return ctrl

    def test_a_shard_that_finished_restarting_mid_refit_stays_stale(self):
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 2)
        monitor.mark_restarting(2)
        # Membership is settled here: shard 2 is RESTARTING, so it is absent and the
        # communicator is built without it.
        participants = self._controller(monitor)._refit_participants()
        assert 2 not in participants

        # The reload lands mid-transfer. Nothing blocks it.
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)
        monitor.mark_loaded(2, base_url="http://replacement:9000/v1")
        assert monitor.state_of(2) is ShardState.STALE
        del supervisor

        self._controller(monitor)._record_refit_landed(participants)

        assert monitor.state_of(2) is ShardState.STALE, (
            "shard 2 received no weights from this refit and must not serve"
        )
        assert 2 not in monitor.serving_shards()

    def test_and_the_next_refit_picks_it_up(self):
        """STALE is not absent, so the following refit includes it and promotes it."""
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 2)
        monitor.mark_restarting(2)
        monitor.mark_loaded(2, base_url="http://replacement:9000/v1")

        participants = self._controller(monitor)._refit_participants()
        assert 2 in participants
        self._controller(monitor, trainer_version=9)._record_refit_landed(participants)

        assert monitor.state_of(2) is ShardState.HEALTHY
        assert monitor.snapshot()[2].weight_version == 9

    def test_the_shards_that_were_refit_are_still_promoted(self):
        """The filter must not strand a survivor that legitimately holds partial weights."""
        monitor, gen = _monitor(), _Generation()
        monitor.mark_weights_partial(0)
        assert monitor.state_of(0) is ShardState.STALE

        participants = self._controller(monitor)._refit_participants()
        self._controller(monitor)._record_refit_landed(participants)

        assert monitor.state_of(0) is ShardState.HEALTHY


class TestTheWeightVersionSaysWhatEachShardHolds:
    """``gen_fleet/shard_weight_version`` has to be written where the weights land.

    Writing it in the promotion instead makes it wrong in both directions at once. Nothing
    turns a shard STALE on a refit that succeeds, so a fleet that never lost a shard is
    never promoted and reports version 0 forever, however many refits it received -- which
    is enough on its own to guarantee nobody looks at the metric. And the one shard that
    reports the current version is the one the promotion touched, which after the
    participants filter can be a shard that received nothing.
    """

    @staticmethod
    def _controller(monitor, trainer_version=5):
        from nemo_rl.algorithms.single_controller import SingleControllerActor

        ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
        ctrl._gen_fleet = monitor
        ctrl._trainer_version = trainer_version
        return ctrl

    def test_an_ordinary_refit_stamps_every_shard_it_reached(self):
        """The common case: nothing died, nothing is STALE, nothing gets promoted."""
        monitor = _monitor()
        ctrl = self._controller(monitor, trainer_version=7)

        ctrl._record_refit_landed(ctrl._refit_participants())

        assert [s.weight_version for s in monitor.snapshot()] == [7, 7, 7]
        assert all(s.state is ShardState.HEALTHY for s in monitor.snapshot())

    def test_a_shard_that_was_not_in_the_refit_keeps_its_old_version(self):
        monitor = _monitor()
        ctrl = self._controller(monitor, trainer_version=7)
        ctrl._record_refit_landed(ctrl._refit_participants())

        _condemn(monitor, 1)
        monitor.mark_restarting(1)
        later = self._controller(monitor, trainer_version=8)
        later._record_refit_landed(later._refit_participants())

        assert monitor.snapshot()[1].weight_version == 7, (
            "shard 1 was absent for the version-8 refit; reporting 8 would say it holds "
            "weights it never received"
        )
        assert monitor.snapshot()[0].weight_version == 8

    def test_a_suspect_shard_is_stamped_but_still_not_promoted(self):
        """The version is a fact about the engine's weights, not a verdict on its health."""
        monitor = _monitor()
        monitor.record_probe(2, ok=False, error="timeout")
        assert monitor.state_of(2) is ShardState.SUSPECT

        ctrl = self._controller(monitor, trainer_version=4)
        ctrl._record_refit_landed(ctrl._refit_participants())

        assert monitor.snapshot()[2].weight_version == 4
        assert monitor.state_of(2) is ShardState.SUSPECT

    def test_a_retired_shard_is_left_alone(self):
        monitor = _monitor()
        monitor.retire(1, reason="attempts exhausted")
        assert monitor.state_of(1) is ShardState.RETIRED

        ctrl = self._controller(monitor, trainer_version=3)
        ctrl._record_refit_landed(ctrl._refit_participants())

        assert monitor.snapshot()[1].weight_version == 0
        assert monitor.state_of(1) is ShardState.RETIRED


class TestTheGpuIsReadBeforeTheRestart:
    """The failure this feature was built around is only visible before the attempt.

    An orphaned EngineCore still holding the memory is what makes a restart fail, and the
    reading that would show it is taken inside the *new* worker at `_load_model` -- too
    late to be a signal, and not taken at all when the replacement never gets scheduled,
    which is exactly the case where you most want to know what is on that GPU.
    """

    def test_the_reading_is_taken_before_restart_shard_runs(self):
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 1)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert [idx for idx, _ in gen.gpu_reads] == [1]
        assert gen.restarted == [1]

    def test_the_label_names_the_shard_and_the_attempt(self):
        """Otherwise five [GPU_DIAG] blocks from one shard are indistinguishable."""
        monitor, gen = _monitor(), _Generation(fail=True)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(
            generation=gen, monitor=monitor, restart_backoff_s=0.0
        )

        async def _main():
            await _tick_and_settle(supervisor)
            await _tick_and_settle(supervisor)

        asyncio.run(_main())

        assert [label for _, label in gen.gpu_reads] == [
            "pre_restart_shard0_attempt1",
            "pre_restart_shard0_attempt2",
        ]

    def test_a_backend_without_the_diagnostic_still_restarts(self):
        """The interface default is a no-op, not NotImplementedError, on purpose."""

        class _Bare:
            def __init__(self):
                self.restarted = []

            def log_shard_gpu_state(self, shard_idx, *, label, timeout_s=30.0):
                del shard_idx, label, timeout_s

            def restart_shard(self, shard_idx):
                self.restarted.append(shard_idx)
                return None

        monitor, gen = _monitor(), _Bare()
        _condemn(monitor, 2)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert gen.restarted == [2]
        assert monitor.state_of(2) is ShardState.STALE


class TestDrainIsActuallyWiredUp:
    """A helper only the tests call is a safety net that is not there.

    ``drain`` reads as "shutdown is handled", and the next person will believe it. But the
    supervisor's restart tasks are created on demand, one per shard, so they are not in the
    task list ``run()``'s teardown cancels -- nothing cancelled them and nothing waited for
    them, and an in-flight restart at shutdown was simply abandoned mid-way.

    Asserted against the source because reaching the real teardown means constructing the
    whole controller; the same shape as the ordering check in test_engine_reaping_env.py.
    """

    def test_the_controller_teardown_drains_the_supervisor(self):
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3]
            / "nemo_rl"
            / "algorithms"
            / "single_controller.py"
        ).read_text()
        drains = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "drain"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_engine_supervisor"
        ]
        assert drains, (
            "single_controller.py never calls self._engine_supervisor.drain(); a restart "
            "in flight when the run ends is abandoned, and drain() is left as a safety "
            "net nothing uses"
        )
        assert all(call.keywords for call in drains), (
            "drain() must be given a timeout: restart_shard can block on a bundle that "
            "will never be filled again, and an unbounded wait in teardown would trade a "
            "lost shard for a process that cannot exit"
        )

    def test_it_reports_what_it_gave_up_on(self, capsys):
        import threading

        gate = threading.Event()
        monitor, gen = _monitor(), _Generation(block=gate)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        async def _main():
            supervisor.tick()
            await supervisor.drain(timeout_s=0.05)

        try:
            asyncio.run(_main())
            assert "still running after" in capsys.readouterr().out
        finally:
            gate.set()


def test_the_supervisors_backend_contract_is_declared_and_implemented():
    """The supervisor calls restart_shard by name, so a backend missing it degrades silently.

    That is not hypothetical: restart_shard was once lost to a merge that resolved a
    conflict in the same region of vllm_generation.py by taking one side wholesale. The
    call site and the test fake both survived, so nothing looked broken -- and _restart
    catches the AttributeError, counts a failure, and retries until the attempt budget
    retires the shard. Restart never worked and never said so.

    Parsed rather than imported: VllmGeneration pulls in vllm, which is not installed in the
    default test venv, and the point is the methods' existence rather than their behaviour.
    Named for what it asserts -- vLLM is the only supervised backend today, so "every
    backend" would be claiming more than one hardcoded path can check.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    called = "restart_shard"

    supervisor = ast.parse(
        (repo_root / "nemo_rl/models/generation/engine_supervisor.py").read_text()
    )
    assert any(
        isinstance(n, ast.Attribute) and n.attr == called for n in ast.walk(supervisor)
    ), "the supervisor no longer calls restart_shard; this test is guarding nothing"

    backend = repo_root / "nemo_rl/models/generation/vllm/vllm_generation.py"
    defined = {
        n.name
        for n in ast.walk(ast.parse(backend.read_text()))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert called in defined, (
        f"{backend.name} does not define {called}, but EngineSupervisor calls it on every "
        "restart. The failure is silent: _restart swallows the AttributeError and the "
        "shard is retried until its attempt budget retires it."
    )

    # And declared, not merely present on one backend. The supervisor is handed whatever
    # GenerationInterface the config selected; a method that exists only on vLLM is a
    # contract kept by coincidence.
    interface = ast.parse(
        (repo_root / "nemo_rl/models/generation/interfaces.py").read_text()
    )
    declared = {
        n.name
        for cls in ast.walk(interface)
        if isinstance(cls, ast.ClassDef) and cls.name == "GenerationInterface"
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {called, "log_shard_gpu_state"} <= declared, (
        "GenerationInterface does not declare "
        f"{sorted({called, 'log_shard_gpu_state'} - declared)}, so a backend that omits "
        "it fails as an AttributeError at restart time rather than as an unsupported "
        "operation"
    )
