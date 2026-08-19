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

"""The one-way link from the fleet ledger to the NeMo-Gym router.

Two directions meet here. Membership flows out -- which shards are serving -- and failure
counts flow back in, because the router sees failures no liveness probe can: a wedged
engine answers ``is_alive`` from a healthy worker process.

The push is deliberately unconditional. Gating it on the membership epoch looked free,
and made a router restart unrecoverable: a recreated actor rebuilds its serving set as
*every* backend while the epoch has not moved, so the gate blocked the one push that
would have corrected it and NeMo-Gym kept routing to a quarantined shard for the rest of
the run.
"""

import asyncio
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetHealth,
    ShardState,
)


class _Remote:
    """Wraps a plain callable so it looks like a Ray actor method."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        async def _call():
            return self._fn(*args, **kwargs)

        return _call()


class _FakeRouter:
    """Stands in for the GenerationRouterActor handle."""

    def __init__(self, *, failures=None, push_error=None):
        self.pushes: list[list[str]] = []
        self._failures = dict(failures or {})
        self._push_error = push_error
        self.set_serving_backends = _Remote(self._set_serving_backends)
        self.drain_backend_failures = _Remote(self._drain)
        self.metrics = _Remote(lambda: {"router/requests_total": 1.0})

    def _set_serving_backends(self, urls):
        if self._push_error is not None:
            raise self._push_error
        self.pushes.append(list(urls))

    def _drain(self):
        drained, self._failures = self._failures, {}
        return drained


def _urls(n):
    return [f"http://shard{i}:8000/v1" for i in range(n)]


def _controller(*, shard_count=3, router=None, unhealthy_threshold=3):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._gen_fleet = GenerationFleetHealth(
        shard_count=shard_count,
        policy=FleetHealthPolicy(unhealthy_threshold=unhealthy_threshold),
        base_urls=_urls(shard_count),
    )
    ctrl._generation_router = router
    ctrl._async_cfg = SimpleNamespace(
        generation_fleet_health=SimpleNamespace(
            probe_interval_s=0.001, probe_timeout_s=1.0
        )
    )
    return ctrl


class TestMembershipPush:
    def test_the_first_push_carries_every_serving_backend(self):
        router = _FakeRouter()
        ctrl = _controller(router=router)
        asyncio.run(ctrl._push_router_membership())
        assert router.pushes == [_urls(3)]

    def test_the_set_is_pushed_on_every_tick_even_when_unchanged(self):
        """Regression: the epoch gate skipped these, and a restarted router stayed wrong.

        The payload is a short list of strings on a probe-interval timer, so the gate
        bought nothing and cost the self-correction both docstrings advertised.
        """
        router = _FakeRouter()
        ctrl = _controller(router=router)
        for _ in range(3):
            asyncio.run(ctrl._push_router_membership())
        assert router.pushes == [_urls(3)] * 3

    def test_a_restarted_router_is_corrected_by_the_next_push(self):
        """A recreated actor comes up serving everything; the next push must fix it."""
        router = _FakeRouter()
        ctrl = _controller(router=router)
        for _ in range(3):
            ctrl._gen_fleet.record_probe(1, ok=False, error="gone")
        assert ctrl._gen_fleet.state_of(1) is ShardState.DEAD
        asyncio.run(ctrl._push_router_membership())
        # The router restarts here -- Ray re-runs __init__, serving = every backend.
        # The corrective push must still arrive, which is the whole regression.
        asyncio.run(ctrl._push_router_membership())
        assert router.pushes[-1] == [_urls(3)[0], _urls(3)[2]]

    def test_a_dead_shard_leaves_the_pushed_set(self):
        router = _FakeRouter()
        ctrl = _controller(router=router)
        for _ in range(3):
            ctrl._gen_fleet.record_probe(2, ok=False, error="gone")
        asyncio.run(ctrl._push_router_membership())
        assert _urls(3)[2] not in router.pushes[-1]

    def test_a_suspect_shard_still_appears(self):
        """SUSPECT is 'failing probes, not yet condemned' -- it still takes traffic."""
        router = _FakeRouter()
        ctrl = _controller(router=router)
        ctrl._gen_fleet.record_probe(0, ok=False, error="blip")
        assert ctrl._gen_fleet.state_of(0) is ShardState.SUSPECT
        asyncio.run(ctrl._push_router_membership())
        assert router.pushes[-1] == _urls(3)

    def test_no_router_means_no_push(self):
        ctrl = _controller(router=None)
        asyncio.run(ctrl._push_router_membership())  # must not raise


class TestFailureDrain:
    def test_router_failures_reach_the_ledger(self):
        """The router counts; this is where the ledger learns about a wedged engine."""
        router = _FakeRouter(failures={_urls(3)[1]: 1})
        ctrl = _controller(router=router)
        asyncio.run(ctrl._drain_router_failures())
        assert ctrl._gen_fleet.state_of(1) is ShardState.SUSPECT

    def test_enough_router_failures_condemn_the_shard(self):
        router = _FakeRouter(failures={_urls(3)[1]: 3})
        ctrl = _controller(router=router, unhealthy_threshold=3)
        asyncio.run(ctrl._drain_router_failures())
        assert ctrl._gen_fleet.state_of(1) is ShardState.DEAD

    def test_an_unknown_backend_url_is_skipped(self):
        """The router may still hold a URL the ledger no longer tracks."""
        router = _FakeRouter(failures={"http://stale:1/v1": 5})
        ctrl = _controller(router=router)
        asyncio.run(ctrl._drain_router_failures())  # must not raise
        assert ctrl._gen_fleet.serving_shards() == [0, 1, 2]


class TestPumpContainment:
    def test_a_push_failure_does_not_end_the_run(self):
        """run() awaits this task and re-raises, so an unguarded error kills training.

        The router is a max_restarts=-1 actor; a RayActorError while it is being
        recreated is exactly the transient the next tick would have healed.
        """
        router = _FakeRouter(push_error=RuntimeError("router is being recreated"))
        ctrl = _controller(router=router)
        # The probe itself is covered in test_watchdog_pump; stub it out so this test is
        # only about the push failing.
        ctrl._probe_generation_fleet = _noop

        async def _main():
            task = asyncio.ensure_future(ctrl._gen_fleet_probe_pump())
            await asyncio.sleep(0.02)
            still_running = not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return still_running

        assert asyncio.run(_main()), "the pump must survive a failed push"


async def _noop() -> None:
    return None
