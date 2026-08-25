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

"""Reconciling refit-communicator membership before each weight sync.

The failure being prevented: a NCCL broadcast requires every rank in the communicator to
take part, so when a generation rank dies the refit blocks forever *inside NCCL* -- no
exception, no progress, and Ray still reporting every actor healthy. These pin that the
check fires when it should, stays out of the way when it should not, and leaves the
transports that own no NCCL world alone.
"""

import asyncio
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetHealth,
    ShardState,
)
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.membership import NoSurvivingShards
from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
    NcclReshardWeightSynchronizer,
)


def _collective() -> CollectiveWeightSynchronizer:
    return CollectiveWeightSynchronizer(
        policy=object(), generation=object(), train_cluster=None, inference_cluster=None
    )


def _reshard() -> NcclReshardWeightSynchronizer:
    return NcclReshardWeightSynchronizer(
        policy=object(), generation=object(), train_cluster=None, inference_cluster=None
    )


@pytest.fixture(params=["collective", "nccl_reshard"])
def synchronizer(request):
    """Both NCCL transports. They diverge once a shard is gone -- collective rebuilds,
    reshard rebuilds too and additionally regenerates its refit plan -- but the
    no-op path must be identical for both."""
    return _collective() if request.param == "collective" else _reshard()


class TestNothingAbsent:
    def test_reconcile_is_a_no_op_when_the_fleet_is_whole(self, synchronizer):
        """The overwhelmingly common path: called before every refit, does nothing."""
        assert synchronizer.reconcile_communicator([]) is False

    def test_repeated_calls_stay_no_ops(self, synchronizer):
        for _ in range(5):
            assert synchronizer.reconcile_communicator([]) is False


class _FakeWorker:
    """One Ray actor handle. Records the init_collective it was asked to run."""

    def __init__(self, idx: int, *, dead: bool = False) -> None:
        self.idx = idx
        self.dead = dead
        self.calls: list[dict] = []

    class _Method:
        def __init__(self, worker: "_FakeWorker") -> None:
            self._worker = worker

        def remote(self, **kwargs):
            if self._worker.dead:
                raise AssertionError(
                    f"worker {self._worker.idx} is gone; dispatching to it is the hang "
                    "this rebuild exists to avoid"
                )
            self._worker.calls.append(kwargs)
            return f"future-{self._worker.idx}"

    def __getattr__(self, name):
        if name.startswith("init_collective"):
            return _FakeWorker._Method(self)
        raise AttributeError(name)


def _rebuildable(dp_size=4, workers_per_shard=1, dead_shards=(), train_world_size=8):
    """A CollectiveWeightSynchronizer over fake Ray handles."""
    workers = []
    for shard in range(dp_size):
        for _ in range(workers_per_shard):
            workers.append(_FakeWorker(len(workers), dead=shard in set(dead_shards)))
    generation = SimpleNamespace(
        cfg={"vllm_cfg": {"async_engine": True}},
        worker_group=SimpleNamespace(workers=workers, dp_size=dp_size),
        # The rebuild bootstraps with the same peer protocol as the first build, so it
        # asks the backend which one that is. A stand-in has to carry every hook the code
        # under test reads, or it tests a shape the product never has.
        get_collective_sender_spec=lambda: SimpleNamespace(nccl_peer="nemo"),
    )
    from nemo_rl.models.generation.vllm import vllm_generation

    generation.set_refit_membership = lambda membership: setattr(
        generation, "_refit_membership", membership
    )
    generation.rebuild_collective = (
        lambda membership, ip, port: vllm_generation.VllmGeneration.rebuild_collective(
            generation, membership, ip, port
        )
    )
    policy_calls = []
    policy = SimpleNamespace(
        init_collective=lambda ip,
        port,
        world_size,
        *,
        train_world_size,
        nccl_peer=None: (
            policy_calls.append(
                {
                    "ip": ip,
                    "port": port,
                    "world_size": world_size,
                    "train_world_size": train_world_size,
                }
            )
            or ["train-future"]
        )
    )
    ports = iter(range(7001, 7100))
    train_cluster = SimpleNamespace(
        world_size=lambda: train_world_size,
        get_master_address_and_port=lambda: ("10.0.0.1", next(ports)),
    )
    sync = CollectiveWeightSynchronizer(
        policy=policy,
        generation=generation,
        train_cluster=train_cluster,
        inference_cluster=None,
    )
    return sync, workers, policy_calls


class TestRebuildDispatch:
    """Where a wrong answer is silent rather than loud."""

    def test_the_dead_shard_is_never_dispatched_to(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, workers, _ = _rebuildable(dead_shards=(2,))

        assert sync.reconcile_communicator([2]) is True
        # _FakeWorker raises if touched, so reaching here is the assertion; confirm the
        # survivors really were called.
        assert [w.idx for w in workers if w.calls] == [0, 1, 3]

    def test_survivors_get_compacted_prefixes(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, workers, _ = _rebuildable(dead_shards=(1,))

        sync.reconcile_communicator([1])

        prefixes = {w.idx: w.calls[0]["rank_prefix"] for w in workers if w.calls}
        assert prefixes == {0: 0, 2: 1, 3: 2}, "survivors must be contiguous, not holed"

    def test_world_size_shrinks_by_the_lost_shard(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, workers, policy_calls = _rebuildable(
            dp_size=4, workers_per_shard=2, dead_shards=(0,), train_world_size=8
        )

        sync.reconcile_communicator([0])

        assert policy_calls[0]["world_size"] == 8 + 6
        assert all(c["world_size"] == 8 + 6 for w in workers for c in w.calls)

    def test_both_sides_rendezvous_on_the_same_address(self, monkeypatch):
        """A mismatch here hangs in the TCPStore instead of erroring."""
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, workers, policy_calls = _rebuildable(dead_shards=(3,))

        sync.reconcile_communicator([3])

        gen_ports = {c["port"] for w in workers for c in w.calls}
        assert gen_ports == {policy_calls[0]["port"]}

    def test_each_rebuild_takes_a_fresh_port(self, monkeypatch):
        """The previous world's store may still be bound."""
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, _, policy_calls = _rebuildable(dead_shards=(3,))

        sync.reconcile_communicator([3])
        # force, because an unchanged absent set is now skipped. What is under test is that
        # each rebuild that DOES happen takes a fresh port, not that every call rebuilds.
        sync.reconcile_communicator([3], force=True)

        assert policy_calls[0]["port"] != policy_calls[1]["port"]

    def test_trainers_are_all_kept(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, _, policy_calls = _rebuildable(dead_shards=(1,), train_world_size=64)

        sync.reconcile_communicator([1])

        assert policy_calls[0]["train_world_size"] == 64

    def test_losing_every_shard_refuses_rather_than_building_an_empty_world(
        self, monkeypatch
    ):
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, _, _ = _rebuildable(dp_size=2, dead_shards=(0, 1))

        with pytest.raises(NoSurvivingShards):
            sync.reconcile_communicator([0, 1])


class TestOtherTransportsAreUnaffected:
    def test_the_default_is_a_no_op_even_with_absent_shards(self):
        """IPC/HTTP/checkpoint-engine own no NCCL world, so there is nothing to break."""
        from nemo_rl.weight_sync.interfaces import WeightSynchronizer

        class _Transport(WeightSynchronizer):
            def sync_weights(self, *, timer=None, kv_scales=None):
                return None

            @property
            def is_stale(self):
                return False

            def init_communicator(self):
                pass

            def shutdown(self):
                pass

        # None, not False: False means "nothing was absent", which the controller
        # reports as a silent non-participating rank. For a transport that owns no
        # membership at all that diagnosis is simply wrong, and it sends the reader
        # hunting for a rank that does not exist.
        assert _Transport().reconcile_communicator([0, 1]) is None


def _monitor(shard_count: int = 3) -> GenerationFleetHealth:
    return GenerationFleetHealth(
        shard_count=shard_count,
        policy=FleetHealthPolicy(),
        base_urls=[f"http://h:{8000 + i}/v1" for i in range(shard_count)],
    )


def _condemn(monitor: GenerationFleetHealth, shard_idx: int) -> None:
    """Drive a shard to DEAD the way the fleet actually does.

    One failure only makes a shard SUSPECT -- reaching DEAD takes
    ``unhealthy_threshold`` consecutive ones, deliberately, so a single blip cannot cost
    a shard.
    """
    for _ in range(FleetHealthPolicy().unhealthy_threshold):
        monitor.report_failure(shard_idx, RuntimeError("actor died"))
    assert monitor.state_of(shard_idx) == ShardState.DEAD


class TestAbsentIsNotTheComplementOfServing:
    """The distinction the whole hook turns on.

    A shard withheld from traffic is not necessarily gone. Treating "not serving" as
    "absent" would abort a run on a single failed probe, and would abort it precisely
    when a STALE shard is waiting to be refit -- which is the recovery, not the failure.
    """

    def test_a_whole_fleet_has_nothing_absent(self):
        assert _monitor().absent_shards() == []

    def test_a_suspect_shard_is_withheld_from_traffic_but_still_in_the_collective(self):
        monitor = _monitor()
        policy = FleetHealthPolicy()
        for _ in range(policy.unhealthy_threshold - 1):
            monitor.record_probe(0, ok=False, error="timeout")

        assert monitor.state_of(0) == ShardState.SUSPECT
        assert 0 not in monitor.absent_shards(), "a probe blip must not abort the refit"

    def test_a_dead_shard_is_absent(self):
        monitor = _monitor()
        _condemn(monitor, 0)

        assert monitor.absent_shards() == [0]

    def test_a_restarting_shard_is_absent(self):
        monitor = _monitor()
        _condemn(monitor, 1)
        monitor.mark_restarting(1)

        assert monitor.absent_shards() == [1]

    def test_a_stale_shard_is_present_because_refitting_it_is_the_recovery(self):
        monitor = _monitor()
        _condemn(monitor, 2)
        monitor.mark_restarting(2)
        monitor.mark_loaded(2)

        assert monitor.state_of(2) == ShardState.STALE
        assert monitor.absent_shards() == [], (
            "a reloaded shard must be allowed into the refit; that is how it stops "
            "being stale"
        )


class TestControllerCallSite:
    """The hook has to be reached, and has to stay inert without fleet health."""

    @staticmethod
    def _controller(monitor, synchronizer):
        from nemo_rl.algorithms.single_controller import SingleControllerActor

        ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
        ctrl._gen_fleet = monitor
        ctrl._weight_synchronizer = synchronizer
        return ctrl

    def test_without_fleet_health_the_transport_is_never_consulted(self):
        calls = []
        synchronizer = SimpleNamespace(
            reconcile_communicator=lambda absent: calls.append(absent) or False
        )
        ctrl = self._controller(None, synchronizer)

        asyncio.run(ctrl._reconcile_refit_membership())

        assert calls == [], "fleet health is off; behaviour must be unchanged"

    def test_the_absent_set_is_forwarded_to_the_transport(self):
        monitor = _monitor()
        _condemn(monitor, 1)
        calls = []
        synchronizer = SimpleNamespace(
            reconcile_communicator=lambda absent, force=False: calls.append(
                list(absent)
            )
            or False
        )
        ctrl = self._controller(monitor, synchronizer)

        asyncio.run(ctrl._reconcile_refit_membership())

        assert calls == [[1]]

    def test_a_refusal_propagates_rather_than_being_swallowed(self, monkeypatch):
        """If this were swallowed the job would proceed into the failure it prevents."""
        monkeypatch.setattr("ray.get", lambda futures: futures)
        monitor = _monitor(shard_count=2)
        _condemn(monitor, 0)
        _condemn(monitor, 1)
        sync, _, _ = _rebuildable(dp_size=2, dead_shards=(0, 1))
        ctrl = self._controller(monitor, sync)

        with pytest.raises(NoSurvivingShards):
            asyncio.run(ctrl._reconcile_refit_membership())

    def test_a_rebuild_is_driven_all_the_way_from_the_controller(self, monkeypatch):
        """End to end through the hook: monitor -> absent set -> rebuilt communicator."""
        monkeypatch.setattr("ray.get", lambda futures: futures)
        monitor = _monitor(shard_count=4)
        _condemn(monitor, 2)
        sync, workers, policy_calls = _rebuildable(dead_shards=(2,))
        ctrl = self._controller(monitor, sync)

        asyncio.run(ctrl._reconcile_refit_membership())

        assert [w.idx for w in workers if w.calls] == [0, 1, 3]
        assert policy_calls[0]["world_size"] == 8 + 3


class TestAnUnchangedMembershipIsNotRebuiltAgain:
    """A dead shard used to cost two full rebuilds on every subsequent step, forever.

    ``absent_shards()`` never empties again -- nothing in production calls
    ``mark_restarting`` or ``mark_loaded`` -- and ``_sync_weights`` reconciles twice per
    step. So a run that lost a shard at step 10 and trained to 10,000 paid ~20,000
    rebuilds, each a fresh port, a fresh TCPStore and a fresh NCCL bootstrap across every
    train and inference rank. The steady state this feature exists to produce was the
    expensive one, and three comments claimed the opposite.
    """

    def test_the_second_call_with_the_same_absent_set_is_skipped(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, workers, _ = _rebuildable(dead_shards=(2,))

        assert sync.reconcile_communicator([2]) is True
        calls_after_first = sum(len(w.calls) for w in workers)

        assert sync.reconcile_communicator([2]) is False, "membership did not change"
        assert sum(len(w.calls) for w in workers) == calls_after_first, (
            "the skip must be a real skip -- no worker may be touched again"
        )

    def test_a_changed_membership_still_rebuilds(self, monkeypatch):
        """The skip must not swallow a second loss."""
        monkeypatch.setattr("ray.get", lambda futures: futures)
        # Only 2 is unreachable: the first rebuild still dispatches to 3, which is only
        # lost by the time of the second.
        sync, _, _ = _rebuildable(dead_shards=(2,))

        assert sync.reconcile_communicator([2]) is True
        assert sync.reconcile_communicator([2, 3]) is True, "a new loss must rebuild"

    def test_order_does_not_defeat_the_comparison(self, monkeypatch):
        """absent_shards() returns a sequence; the same set in another order is the same."""
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, _, _ = _rebuildable(dead_shards=(2, 3))

        assert sync.reconcile_communicator([2, 3]) is True
        assert sync.reconcile_communicator([3, 2]) is False

    def test_force_rebuilds_over_an_unchanged_membership(self, monkeypatch):
        """The recovery path's case, and the one that breaks if the skip has no override.

        After an abort the communicator is gone while the absent set is identical. Skipping
        there would make the retry run over a communicator that no longer exists, and the
        recovery would fail with "no generation shard could be identified as absent".
        """
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, workers, _ = _rebuildable(dead_shards=(2,))

        assert sync.reconcile_communicator([2]) is True
        calls_after_first = sum(len(w.calls) for w in workers)

        assert sync.reconcile_communicator([2], force=True) is True
        assert sum(len(w.calls) for w in workers) > calls_after_first, (
            "force must actually rebuild, not just return True"
        )

    def test_nothing_absent_still_short_circuits_first(self, monkeypatch):
        """An empty absent set is 'nothing to do', not 'same as last time'."""
        monkeypatch.setattr("ray.get", lambda futures: futures)
        sync, _, _ = _rebuildable(dead_shards=(2,))

        assert sync.reconcile_communicator([]) is False
        assert sync.reconcile_communicator([], force=True) is False


class TestStragglersAreWaitedForBeforeARebuild:
    """A communicator rebuild is a collective, so it needs every rank out of the old one.

    ray.get raises on the FIRST future that fails and leaves the rest running. Job 6512153
    measured what that costs on the reshard kill variant: rank 0 gave up on its own
    deadline, the controller went into the recovery, and the rebuild began two log lines
    BEFORE rank 1's watchdog fired. The surviving generation worker then spent 300s twice
    failing to reach a store that never came up, and the run died at 690s.
    """

    @staticmethod
    def _futures(monkeypatch, *, pending):
        """Records what ray.wait was asked to wait for."""
        waited = {}

        def _fake_wait(futures, *, num_returns, timeout):
            waited["futures"] = list(futures)
            waited["num_returns"] = num_returns
            waited["timeout"] = timeout
            return ([], list(futures)) if pending else (list(futures), [])

        monkeypatch.setattr("ray.wait", _fake_wait)
        return waited

    def test_every_train_rank_is_waited_for(self, monkeypatch):
        from nemo_rl.weight_sync import nccl_reshard_weight_synchronizer as mod

        waited = self._futures(monkeypatch, pending=False)
        mod._settle_before_propagating(["a", "b", "c"], 90.0, "train")

        assert waited["futures"] == ["a", "b", "c"]
        assert waited["num_returns"] == 3, "all of them, not just the first"
        assert waited["timeout"] == 90.0

    def test_it_is_bounded_rather_than_blocking_the_recovery(self, monkeypatch):
        """A caller stuck here is a worse wedge than the one being recovered from."""
        from nemo_rl.weight_sync import nccl_reshard_weight_synchronizer as mod

        self._futures(monkeypatch, pending=True)
        mod._settle_before_propagating(["a"], 0.1, "train")  # must return, not hang

    def test_a_stragglers_error_does_not_replace_the_caller_s(self, monkeypatch):
        """They are unwinding from the same failure the caller already holds."""
        from nemo_rl.weight_sync import nccl_reshard_weight_synchronizer as mod

        def _boom(*_args, **_kwargs):
            raise RuntimeError("straggler blew up while unwinding")

        monkeypatch.setattr("ray.wait", _boom)
        mod._settle_before_propagating(["a"], 1.0, "train")  # must not raise

    def test_nothing_to_wait_for_is_a_no_op(self, monkeypatch):
        from nemo_rl.weight_sync import nccl_reshard_weight_synchronizer as mod

        def _never(*_args, **_kwargs):
            raise AssertionError("ray.wait must not be called with no futures")

        monkeypatch.setattr("ray.wait", _never)
        mod._settle_before_propagating([], 1.0, "train")

    def test_the_budget_outlasts_the_deadline_the_ranks_were_armed_with(self):
        """A straggler gives up when its own watchdog fires; wait past that."""
        sync = _collective()
        sync._refit_timeout_s = 60.0
        assert sync._settle_budget_s() > 60.0

    def test_an_unconfigured_deadline_still_bounds_the_wait(self):
        """Nothing bounds the ranks, so this must not wait forever either."""
        sync = _collective()
        sync._refit_timeout_s = None
        assert 0 < sync._settle_budget_s() < 600
