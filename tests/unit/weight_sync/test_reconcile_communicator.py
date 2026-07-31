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
    GenerationFleetMonitor,
    ShardState,
)
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.interfaces import RefitMembershipChanged
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
    """Both NCCL transports; both must refuse a refit into a broken world."""
    return _collective() if request.param == "collective" else _reshard()


class TestNothingAbsent:
    def test_reconcile_is_a_no_op_when_the_fleet_is_whole(self, synchronizer):
        """The overwhelmingly common path: called before every refit, does nothing."""
        assert synchronizer.reconcile_communicator([]) is False

    def test_repeated_calls_stay_no_ops(self, synchronizer):
        for _ in range(5):
            assert synchronizer.reconcile_communicator([]) is False


class TestAbsentRanks:
    def test_a_missing_shard_raises_instead_of_hanging(self, synchronizer):
        with pytest.raises(RefitMembershipChanged, match=r"\[2\]"):
            synchronizer.reconcile_communicator([2])

    def test_the_message_names_every_absent_shard(self, synchronizer):
        with pytest.raises(RefitMembershipChanged, match=r"\[1, 3\]"):
            synchronizer.reconcile_communicator([3, 1])

    def test_the_reshard_message_explains_the_extra_constraint(self):
        """Resizing a reshard world without regenerating the plan corrupts weights.

        Worth a distinct message: the plain broadcast could in principle just drop a
        receiver, but the reshard's destination placements are derived from
        gen_world_size, so survivors would hold slices nobody wrote.
        """
        with pytest.raises(RefitMembershipChanged, match="gen_world_size"):
            _reshard().reconcile_communicator([0])


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

            def mark_stale(self):
                pass

            def init_communicator(self):
                pass

            def shutdown(self):
                pass

        assert _Transport().reconcile_communicator([0, 1]) is False


def _monitor(shard_count: int = 3) -> GenerationFleetMonitor:
    return GenerationFleetMonitor(
        shard_count=shard_count,
        policy=FleetHealthPolicy(),
        base_urls=[f"http://h:{8000 + i}/v1" for i in range(shard_count)],
    )


def _condemn(monitor: GenerationFleetMonitor, shard_idx: int) -> None:
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
        ctrl._fleet_monitor = monitor
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
            reconcile_communicator=lambda absent: calls.append(list(absent)) or False
        )
        ctrl = self._controller(monitor, synchronizer)

        asyncio.run(ctrl._reconcile_refit_membership())

        assert calls == [[1]]

    def test_a_refusal_propagates_rather_than_being_swallowed(self):
        """If this were swallowed the job would proceed into the hang it prevents."""
        monitor = _monitor()
        _condemn(monitor, 0)
        ctrl = self._controller(monitor, _collective())

        with pytest.raises(RefitMembershipChanged):
            asyncio.run(ctrl._reconcile_refit_membership())
