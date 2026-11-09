# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A restart between refits must rebuild even when the shard layout is unchanged."""

from typing import Any

import pytest

from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
    NcclReshardWeightSynchronizer,
)
from tests.unit.weight_sync.test_reconcile_communicator import _rebuildable
from tests.unit.weight_sync.test_reshard_rebuild import _reshard

_Fleet = tuple[CollectiveWeightSynchronizer | NcclReshardWeightSynchronizer, list[Any]]


@pytest.fixture(params=["collective", "nccl_reshard"])
def fleet(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _Fleet:
    monkeypatch.setattr("ray.get", lambda futures: futures)
    if request.param == "collective":
        sync, workers, _ = _rebuildable(dp_size=3, workers_per_shard=2)
    else:
        sync, _, workers, _, _ = _reshard(dp_size=3, workers_per_shard=2)
    sync.init_communicator()
    assert sync.reconcile_communicator([]) is False
    return sync, workers


@pytest.mark.parametrize("victims", [(2,), (3,), (2, 3, 4, 5)])
def test_fast_replacement_rebuilds_metadata_and_communicators(
    fleet: _Fleet, victims: tuple[int, ...]
) -> None:
    """Cover a leader, a non-leader TP rank, and two complete shards, twice."""
    sync, workers = fleet
    for _ in range(2):
        previous = sync._built_membership
        for idx in victims:
            workers[idx] = type(workers[idx])(idx)
        assert sync.reconcile_communicator([]) is True
        assert sync._built_membership.shard_prefixes == previous.shard_prefixes
        assert sync._built_membership.actor_ids != previous.actor_ids
        # A TP non-leader is bootstrapped through its shard's leader.
        for shard in {idx // 2 for idx in victims}:
            assert workers[shard * 2].calls
        if isinstance(sync, CollectiveWeightSynchronizer):
            assert len(sync._generation.refit_info_pushes) >= 2
        else:
            for shard in {idx // 2 for idx in victims}:
                names = {name for name, _ in workers[shard * 2].calls}
                assert "prepare_nccl_reshard_refit_info" in names
                assert "init_nccl_reshard_comm_group" in names
        assert sync.reconcile_communicator([]) is False
        assert sync.reconcile_communicator([], force=True) is False


def test_replacement_in_a_reduced_fleet_is_detected(fleet: _Fleet) -> None:
    sync, workers = fleet
    assert sync.reconcile_communicator([2]) is True
    workers[2] = type(workers[2])(2)
    assert sync.reconcile_communicator([2]) is True
    assert sync.reconcile_communicator([2]) is False
    # Replacements which remain excluded are not refit participants yet.
    workers[4] = type(workers[4])(4)
    assert sync.reconcile_communicator([2]) is False
    assert sync.reconcile_communicator([]) is True


def test_failed_bootstrap_does_not_commit_new_identities(
    fleet: _Fleet, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync, workers = fleet
    previous = sync._built_membership
    workers[2] = type(workers[2])(2)

    def fail(_futures: list[Any]) -> None:
        raise RuntimeError("bootstrap failed")

    with monkeypatch.context() as patch:
        patch.setattr("ray.get", fail)
        with pytest.raises(RuntimeError, match="bootstrap failed"):
            sync.reconcile_communicator([])
    assert sync._built_membership == previous
    assert sync.reconcile_communicator([]) is True


@pytest.mark.parametrize("initial", [False, True])
def test_replacement_during_bootstrap_is_not_recorded_as_initialized(
    fleet: _Fleet, monkeypatch: pytest.MonkeyPatch, initial: bool
) -> None:
    sync, workers = fleet
    previous = sync._built_membership
    workers[2] = type(workers[2])(2)

    def replace_during_wait(futures: list[Any]) -> list[Any]:
        workers[3] = type(workers[3])(3)
        return futures

    with monkeypatch.context() as patch:
        patch.setattr("ray.get", replace_during_wait)
        with pytest.raises(RuntimeError, match="actors changed during"):
            if initial:
                sync.init_communicator()
            else:
                sync.reconcile_communicator([])
    assert sync._built_membership == previous
    assert sync.reconcile_communicator([]) is True
