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

import pytest

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
        self.fail = fail
        self._block = block

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


class TestFailedRestarts:
    def test_a_failed_restart_returns_the_shard_to_dead(self):
        """Left in RESTARTING it would never be retried and never be retired."""
        monitor, gen = _monitor(), _Generation(fail=True)
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

        asyncio.run(_tick_and_settle(supervisor))

        assert monitor.state_of(0) is ShardState.DEAD
        assert supervisor.metrics()["supervisor/restarts_failed"] == 1.0

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
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)

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
        assert len(gen.restarted) <= policy.max_restart_attempts_per_shard


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


@pytest.mark.parametrize("state", [ShardState.RETIRED])
def test_a_retired_shard_is_never_restarted(state):
    monitor, gen = _monitor(), _Generation()
    monitor.retire(0, reason="node gone")
    supervisor = EngineSupervisor(generation=gen, monitor=monitor)

    asyncio.run(_tick_and_settle(supervisor))

    assert gen.restarted == []
    assert monitor.state_of(0) is state


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

        self._controller(monitor)._promote_refit_shards()

        assert monitor.state_of(1) is ShardState.HEALTHY
        assert 1 in monitor.serving_shards()

    def test_the_promoted_shard_carries_the_current_weight_version(self):
        monitor, gen = _monitor(), _Generation()
        _condemn(monitor, 0)
        supervisor = EngineSupervisor(generation=gen, monitor=monitor)
        asyncio.run(_tick_and_settle(supervisor))

        self._controller(monitor, trainer_version=11)._promote_refit_shards()

        assert monitor.snapshot()[0].weight_version == 11

    def test_a_suspect_shard_is_not_promoted_by_a_refit(self):
        """It took part in the refit, but it is failing probes for its own reasons and
        promoting it would reset the count that is meant to condemn it."""
        monitor = _monitor()
        monitor.record_probe(2, ok=False, error="timeout")
        assert monitor.state_of(2) is ShardState.SUSPECT

        self._controller(monitor)._promote_refit_shards()

        assert monitor.state_of(2) is ShardState.SUSPECT

    def test_promotion_is_inert_without_fleet_health(self):
        ctrl = self._controller(None)
        ctrl._promote_refit_shards()  # must not raise
