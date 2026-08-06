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

"""Lifecycle of the refit process group.

Elastic recovery rebuilds this group whenever the generation fleet's membership
changes, so it runs many times per job rather than once. That makes two things load
bearing: releasing a group must never hang (its peer may be dead, which is the whole
reason we are rebuilding), and it must actually release, or every recovery strands a
NCCL communicator and a TCPStore for the life of the worker.

Building a real communicator needs CUDA and peers, so these drive the lifecycle against
a stub communicator. The runtime behaviour of abort() against a genuinely dead peer was
verified separately on 2xA6000 (design doc 8.4.1).
"""

from typing import Optional

import pytest
import torch

import nemo_rl.distributed.stateless_process_group as spg_module
from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup


class _FakeCommunicator:
    def __init__(self) -> None:
        self.aborts = 0
        self.broadcasts = 0

    def abort(self) -> None:
        self.aborts += 1

    def broadcast(self, **kwargs) -> None:
        del kwargs
        self.broadcasts += 1


def _group(communicator: Optional[_FakeCommunicator] = None) -> StatelessProcessGroup:
    """A group as it exists after construction, without binding a TCPStore port."""
    group = object.__new__(StatelessProcessGroup)
    group.master_address = "127.0.0.1"
    group.port = 12345
    group.rank = 0
    group.world_size = 2
    group.nccl_communicator = communicator
    group.tcp_store = object()
    return group


class TestAbort:
    def test_abort_before_the_communicator_exists_is_a_no_op(self):
        """A group can be constructed and then abandoned before init ever runs."""
        group = _group(None)
        group.abort()  # must not raise
        assert group.nccl_communicator is None

    def test_abort_releases_the_communicator(self):
        communicator = _FakeCommunicator()
        group = _group(communicator)

        group.abort()

        assert communicator.aborts == 1
        assert group.nccl_communicator is None

    def test_abort_is_idempotent(self):
        """Reconciliation may abort a group that a previous pass already released."""
        communicator = _FakeCommunicator()
        group = _group(communicator)

        group.abort()
        group.abort()

        assert communicator.aborts == 1

    def test_the_reference_is_dropped_before_abort_is_called(self):
        """If abort() itself raises, the group must not keep serving a dead comm.

        Otherwise a failed release leaves broadcast() pointing at a communicator whose
        peer is gone, which is exactly the hang this path exists to prevent.
        """

        class _Exploding(_FakeCommunicator):
            def abort(self) -> None:
                raise RuntimeError("nccl abort failed")

        group = _group(_Exploding())

        with pytest.raises(RuntimeError, match="nccl abort failed"):
            group.abort()

        assert group.nccl_communicator is None


class TestBroadcastGuard:
    def test_broadcast_without_a_communicator_names_the_cause(self):
        group = _group(None)

        with pytest.raises(RuntimeError, match="aborted and not rebuilt"):
            group.broadcast(tensor=object(), src=0, stream=object())

    def test_broadcast_after_abort_raises_rather_than_crashing(self):
        """The failure a rebuild bug would produce: using a released group."""
        group = _group(_FakeCommunicator())
        group.abort()

        with pytest.raises(RuntimeError, match="no communicator"):
            group.broadcast(tensor=object(), src=0, stream=object())


class _RecordingGroup:
    """Stands in for StatelessProcessGroup so no port is bound and no CUDA is touched."""

    instances: list["_RecordingGroup"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.aborts = 0
        _RecordingGroup.instances.append(self)

    def init_nccl_communicator(self, device) -> None:
        del device

    def abort(self) -> None:
        self.aborts += 1


@pytest.fixture
def recording_group(monkeypatch):
    _RecordingGroup.instances = []
    monkeypatch.setattr(spg_module, "StatelessProcessGroup", _RecordingGroup)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return _RecordingGroup


class TestRebuildReleasesThePreviousGroup:
    """Both sides of the refit rebuild the same group; both must release first.

    A rebuild that only overwrites the attribute leaks the old communicator and its
    TCPStore. That is invisible in a one-shot job -- which is why it survived until
    membership became dynamic -- and unbounded once recovery can happen repeatedly.
    """

    def test_policy_worker_releases_before_rebuilding(self, recording_group):
        from nemo_rl.models.policy.workers.base_policy_worker import (
            AbstractPolicyWorker,
        )

        worker = object.__new__(AbstractPolicyWorker)
        worker.rank = 0

        worker.init_collective("10.0.0.1", 5000, world_size=4, train_world_size=2)
        worker.init_collective("10.0.0.1", 5001, world_size=3, train_world_size=2)

        first, second = recording_group.instances
        assert first.aborts == 1, "first group was not released on rebuild"
        assert second.aborts == 0
        assert worker.model_update_group is second
        # The rebuild must carry the new membership, not resurrect the old world size.
        assert second.kwargs["world_size"] == 3

    def test_first_init_has_nothing_to_release(self, recording_group):
        from nemo_rl.models.policy.workers.base_policy_worker import (
            AbstractPolicyWorker,
        )

        worker = object.__new__(AbstractPolicyWorker)
        worker.rank = 0
        worker.init_collective("10.0.0.1", 5000, world_size=4, train_world_size=2)

        assert len(recording_group.instances) == 1
        assert recording_group.instances[0].aborts == 0

    # The generation-side half of this invariant lives in
    # tests/unit/models/generation/test_vllm_backend.py, because vllm_backend imports
    # `vllm` eagerly and is only collectable in the vllm test lane.
