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

"""Two independent concerns that share this module.

Peer protocol: which NCCL metadata and warmup a receiver expects, "nemo" or "vllm".

Lifecycle of the refit process group: elastic recovery rebuilds this group whenever the
generation fleet's membership changes, so it runs many times per job rather than once.
That makes two things load bearing -- releasing a group must never hang (its peer may be
dead, which is the whole reason we are rebuilding), and it must actually release, or
every recovery strands a NCCL communicator and a TCPStore for the life of the worker.

Building a real communicator needs CUDA and peers, so these drive the lifecycle against
a stub communicator. The runtime behaviour of abort() against a genuinely dead peer was
verified separately on 2xA6000 (design doc 8.4.1).
"""

import ctypes
import pickle
import sys
import types
from contextlib import contextmanager, nullcontext
from typing import Optional

import pytest
import torch

from nemo_rl.distributed import stateless_process_group as spg
from nemo_rl.distributed.refit_watchdog import RefitAborted
from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup


@contextmanager
def _vllm_unique_id_type():
    module_names = [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.device_communicators",
        spg._VLLM_NCCL_MODULE,
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    modules = {name: types.ModuleType(name) for name in module_names}
    for name in module_names[:-1]:
        modules[name].__path__ = []

    modules["vllm"].distributed = modules["vllm.distributed"]
    modules["vllm.distributed"].device_communicators = modules[
        "vllm.distributed.device_communicators"
    ]
    modules["vllm.distributed.device_communicators"].pynccl_wrapper = modules[
        spg._VLLM_NCCL_MODULE
    ]
    unique_id_type = type(
        "ncclUniqueId",
        (ctypes.Structure,),
        {
            "__module__": spg._VLLM_NCCL_MODULE,
            "_fields_": [("internal", ctypes.c_byte * 128)],
        },
    )
    modules[spg._VLLM_NCCL_MODULE].ncclUniqueId = unique_id_type

    try:
        sys.modules.update(modules)
        yield unique_id_type
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def test_vllm_unique_id_pickle_unpickles_as_vllm_ctypes_type():
    unique_id_bytes = bytes(range(128))
    payload = spg._pickle_vllm_unique_id(unique_id_bytes)

    with _vllm_unique_id_type() as unique_id_type:
        unique_id = pickle.loads(payload)

    assert isinstance(unique_id, unique_id_type)
    assert bytes(unique_id) == unique_id_bytes


def test_vllm_unique_id_pickle_requires_nccl_id_size():
    with pytest.raises(ValueError, match="128-byte NCCL unique ID"):
        spg._pickle_vllm_unique_id(b"too short")


class _Store:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value


class _Stream:
    cuda_stream = 123

    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _Communicator:
    def __init__(self):
        self.allreduce_calls = []
        self.broadcast_calls = []

    def allreduce(self, **kwargs):
        self.allreduce_calls.append(kwargs)

    def broadcast(self, **kwargs):
        self.broadcast_calls.append(kwargs)


def _make_process_group(monkeypatch):
    store = _Store()
    stream = _Stream()
    communicator = _Communicator()
    unique_id_bytes = bytes(range(128))
    unique_id = types.SimpleNamespace(as_bytes=unique_id_bytes)

    monkeypatch.setattr(spg.torch.distributed, "TCPStore", lambda **_kwargs: store)
    monkeypatch.setattr(spg, "get_unique_id", lambda: unique_id)
    monkeypatch.setattr(spg.Communicator, "init", lambda **_kwargs: communicator)
    monkeypatch.setattr(spg.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(spg.torch.cuda, "current_stream", lambda: stream)

    group = spg.StatelessProcessGroup(
        master_address="127.0.0.1", port=1234, rank=0, world_size=2
    )
    return group, store, stream, communicator, unique_id_bytes


def test_vllm_peer_uses_vllm_metadata_and_allreduce_warmup(monkeypatch):
    # Coupled to vLLM 0.23.0's stateless process-group wire protocol. Reverify
    # this literal whenever the isolated Dynamo vLLM pin changes.
    assert spg._VLLM_UNIQUE_ID_KEY == "broadcast_from/0/0"
    group, store, stream, communicator, unique_id_bytes = _make_process_group(
        monkeypatch
    )
    warmup_tensor = object()
    monkeypatch.setattr(spg.torch, "zeros", lambda *_args, **_kwargs: warmup_tensor)

    group.init_nccl_communicator(device=0, peer="vllm")

    assert store.data[spg._NEMO_UNIQUE_ID_KEY] == unique_id_bytes
    with _vllm_unique_id_type():
        stored_unique_id = pickle.loads(store.data[spg._VLLM_UNIQUE_ID_KEY])
    assert bytes(stored_unique_id) == unique_id_bytes
    assert communicator.allreduce_calls == [
        {
            "sendbuf": warmup_tensor,
            "recvbuf": warmup_tensor,
            "op": spg.SUM,
            "stream": 123,
        }
    ]
    assert communicator.broadcast_calls == []
    assert stream.synchronized


def test_nemo_peer_preserves_broadcast_warmup(monkeypatch):
    group, store, stream, communicator, unique_id_bytes = _make_process_group(
        monkeypatch
    )
    warmup_tensor = object()
    expected_tensor = object()
    monkeypatch.setattr(spg.torch, "ones", lambda *_args, **_kwargs: warmup_tensor)
    monkeypatch.setattr(spg.torch, "allclose", lambda actual, expected: True)
    monkeypatch.setattr(spg.torch, "zeros", lambda *_args, **_kwargs: expected_tensor)

    group.init_nccl_communicator(device=0)

    assert store.data == {spg._NEMO_UNIQUE_ID_KEY: unique_id_bytes}
    assert communicator.broadcast_calls == [
        {
            "sendbuf": warmup_tensor,
            "recvbuf": warmup_tensor,
            "root": 0,
            "stream": 123,
        }
    ]
    assert communicator.allreduce_calls == []
    assert stream.synchronized


# ---------------------------------------------------------------------------
# Refit process group lifecycle
# ---------------------------------------------------------------------------


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
    # Set here because object.__new__ skips __init__. A fixture that stands in for a
    # constructed group has to carry every field the methods under test read, or it
    # tests a shape the product never has.
    group._aborted = False
    return group


class _Stub:
    """A stream stub. object() is enough until a test actually reaches the broadcast."""

    cuda_stream = 0


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
        assert group.tcp_store is None, "the rendezvous store must be dropped too"

    def test_abort_is_idempotent(self):
        """Reconciliation may abort a group that a previous pass already released."""
        communicator = _FakeCommunicator()
        group = _group(communicator)

        group.abort()
        group.abort()

        assert communicator.aborts == 1

    def test_the_split_children_are_aborted_and_dropped(self):
        """The third communicator family, which aborting the parent does not reach.

        NCCL gives a split child its own abort flag unless splitShare is set, and it
        defaults to 0 -- so a rank blocked on a child survives the parent's abort. It
        never returns, so the watchdog's block never exits to have `fired` read: a hang
        no exception translation can reach.
        """
        from nemo_rl.weight_sync import xferdtensor_python as xp

        communicator = _FakeCommunicator()
        group = _group(communicator)
        child, stale = _FakeCommunicator(), _FakeCommunicator()
        key = (id(communicator), "replica-0")
        xp._SUBCOMM_CACHE[key] = child
        xp._INACTIVE_SUBCOMM_CACHE[key] = stale
        try:
            group.abort()
            # Read inside the try: the cleanup below removes these keys either way, so
            # asserting after it would pass whether or not abort() dropped them.
            still_active = key in xp._SUBCOMM_CACHE
            still_inactive = key in xp._INACTIVE_SUBCOMM_CACHE
        finally:
            xp._SUBCOMM_CACHE.pop(key, None)
            xp._INACTIVE_SUBCOMM_CACHE.pop(key, None)

        assert child.aborts == 1, (
            "a blocked rank on the child is otherwise never released"
        )
        assert stale.aborts == 1
        assert not still_active, (
            "the cache is keyed on id(nccl_communicator), so a rebuild would otherwise "
            "strand these for the life of the worker"
        )
        assert not still_inactive

    def test_the_children_are_aborted_not_destroyed(self):
        """destroy()/finalize() are collective; a dead peer cannot join them.

        Using the existing clear_xferdtensor_python_caches here would have moved the hang
        from the child collective to the child teardown rather than removing it.
        """
        from nemo_rl.weight_sync import xferdtensor_python as xp

        class _Child(_FakeCommunicator):
            def __init__(self) -> None:
                super().__init__()
                self.destroys = 0
                self.finalizes = 0

            def destroy(self) -> None:
                self.destroys += 1

            def finalize(self) -> None:
                self.finalizes += 1

        communicator = _FakeCommunicator()
        group = _group(communicator)
        child = _Child()
        key = (id(communicator), "replica-0")
        xp._SUBCOMM_CACHE[key] = child
        try:
            group.abort()
        finally:
            xp._SUBCOMM_CACHE.pop(key, None)

        assert child.aborts == 1
        assert child.destroys == 0, "destroy() is collective and hangs on a dead peer"
        assert child.finalizes == 0

    def test_another_groups_children_are_left_alone(self):
        from nemo_rl.weight_sync import xferdtensor_python as xp

        mine, theirs = _FakeCommunicator(), _FakeCommunicator()
        group = _group(mine)
        other_child = _FakeCommunicator()
        other_key = (id(theirs), "replica-0")
        xp._SUBCOMM_CACHE[other_key] = other_child
        try:
            group.abort()
            still_cached = other_key in xp._SUBCOMM_CACHE
        finally:
            xp._SUBCOMM_CACHE.pop(other_key, None)

        assert other_child.aborts == 0
        assert still_cached, "another group's children are not ours to evict"

    def test_a_failing_child_abort_does_not_stop_the_parent(self):
        """The caller may be blocked on the parent, or on a sibling that would release."""
        from nemo_rl.weight_sync import xferdtensor_python as xp

        class _Exploding(_FakeCommunicator):
            def abort(self) -> None:
                raise RuntimeError("child abort failed")

        communicator = _FakeCommunicator()
        group = _group(communicator)
        bad, good = _Exploding(), _FakeCommunicator()
        keys = ((id(communicator), "a"), (id(communicator), "b"))
        xp._SUBCOMM_CACHE[keys[0]] = bad
        xp._SUBCOMM_CACHE[keys[1]] = good
        try:
            group.abort()
        finally:
            for k in keys:
                xp._SUBCOMM_CACHE.pop(k, None)

        assert good.aborts == 1
        assert communicator.aborts == 1, "the parent must still be released"

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
    def test_a_group_that_was_never_built_says_so(self):
        """Never initialized is a different failure from aborted, and stays a RuntimeError.

        Collapsing the two would make an ordinary setup mistake look like a mid-refit
        abort and send the reader to the watchdog.
        """
        group = _group(None)

        with pytest.raises(RuntimeError, match="never called") as raised:
            group.broadcast(tensor=object(), src=0, stream=_Stub())
        assert not isinstance(raised.value, RefitAborted)

    def test_every_broadcast_after_an_abort_reports_the_abort(self):
        """The whole point: a refit is many broadcasts, not one.

        The watchdog aborts the collective in flight; that one returns cleanly and the
        caller reads guard.fired. The NEXT buffer lands here. If it raised a bare
        RuntimeError it would escape the caller's ``with`` block before guard.fired is
        read -- no RefitAborted, no recovery, and the run dies blaming a missing
        communicator instead of the abort that caused it.
        """
        group = _group(_FakeCommunicator())
        group.abort()

        for _ in range(3):  # every subsequent buffer, not just the first
            with pytest.raises(RefitAborted, match="aborted mid-collective"):
                group.broadcast(tensor=object(), src=0, stream=_Stub())

    def test_the_abort_survives_the_watchdog_nulling_the_field_mid_call(self):
        """broadcast() must not read self.nccl_communicator twice.

        The watchdog thread nulls it from under the caller, so a check-then-call on the
        attribute can pass the check and then raise AttributeError: 'NoneType'.
        """
        group = _group(_FakeCommunicator())

        class _NullsOnUse(_FakeCommunicator):
            def broadcast(self, **kwargs):
                group.abort()  # the watchdog firing between the check and the call
                super().broadcast(**kwargs)

        nulling = _NullsOnUse()
        group.nccl_communicator = nulling

        group.broadcast(
            tensor=object(), src=0, stream=_Stub()
        )  # must not AttributeError

        assert nulling.broadcasts == 1, (
            "the snapshot must survive the field being nulled"
        )


class _RecordingGroup:
    """Stands in for StatelessProcessGroup so no port is bound and no CUDA is touched."""

    instances: list["_RecordingGroup"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.aborts = 0
        _RecordingGroup.instances.append(self)

    def init_nccl_communicator(self, device, *, peer: str = "nemo") -> None:
        # Mirrors the real signature including ``peer``, which init_collective now
        # forwards. A stub that omits a kwarg its caller sends fails at the call with a
        # TypeError, which is the same shape of break this branch has hit three times in
        # the product code -- worth not reproducing in the fakes as well.
        del device, peer

    def abort(self) -> None:
        self.aborts += 1


@pytest.fixture
def recording_group(monkeypatch):
    _RecordingGroup.instances = []
    monkeypatch.setattr(spg, "StatelessProcessGroup", _RecordingGroup)
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
