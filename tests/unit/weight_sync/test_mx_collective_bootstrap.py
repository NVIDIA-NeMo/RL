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

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from nemo_rl.weight_sync import mx_collective_bootstrap as bootstrap


def rendezvous_kwargs(**overrides):
    kwargs = {
        "mx_server_url": "mx:8001",
        "model_name": "model",
        "role": "TRAINER",
        "index_in_role": 0,
        "slot_id": "train/0",
        "worker_id": "worker-0",
        "trainer_slots": ["train/0"],
        "generator_slots": ["gen/0"],
        "source_partition_count": 1,
    }
    kwargs.update(overrides)
    return kwargs


def bootstrap_state(*, rank=0, epoch=7):
    lane = SimpleNamespace(lane_id=1, rank_in_lane=rank, world_size=2, kind="BROADCAST")
    membership = SimpleNamespace(group_id="group", epoch=epoch, lanes=(lane,))
    return bootstrap.MxBootstrapState(
        membership, {}, device=0, worker_id=f"worker-{rank}"
    )


class FakeRendezvous:
    def __init__(self):
        self.published = []

    def publish_bootstrap(self, **kwargs):
        self.published.append(kwargs)


class RootUniqueId:
    def __init__(self, raw):
        self.as_bytes = raw


def install_unique_id_module(monkeypatch, unique_id):
    utils = ModuleType("nccl.core.utils")
    utils.get_unique_id = lambda: unique_id
    core = ModuleType("nccl.core")
    nccl = ModuleType("nccl")
    monkeypatch.setitem(sys.modules, "nccl", nccl)
    monkeypatch.setitem(sys.modules, "nccl.core", core)
    monkeypatch.setitem(sys.modules, "nccl.core.utils", utils)


def test_rendezvous_rejects_an_unknown_role_before_joining():
    with pytest.raises(ValueError, match="unsupported MX collective role"):
        bootstrap.mx_rendezvous(**rendezvous_kwargs(role="OBSERVER"))


def test_rendezvous_rejects_nccl_comm_id_override(monkeypatch):
    monkeypatch.setenv("NCCL_COMM_ID", "10.0.0.1:1234")
    with pytest.raises(RuntimeError, match="must be unset"):
        bootstrap.mx_rendezvous(**rendezvous_kwargs())


def test_lane_root_rejects_a_uid_different_from_what_it_published(monkeypatch):
    minted = b"a" * 128
    install_unique_id_module(monkeypatch, RootUniqueId(minted))
    state = bootstrap_state()
    state.rz = FakeRendezvous()
    monkeypatch.setattr(bootstrap, "_await_lane_id", lambda *_: b"b" * 128)

    with pytest.raises(RuntimeError, match="different ncclUniqueId"):
        bootstrap.mx_init_lane(state, 1)
    assert state.rz.published[0]["nccl_unique_id"] == minted


def test_lane_root_keeps_the_minted_uid_object_through_communicator_init(monkeypatch):
    minted = b"a" * 128
    root_unique_id = RootUniqueId(minted)
    install_unique_id_module(monkeypatch, root_unique_id)
    state = bootstrap_state()
    state.rz = FakeRendezvous()
    monkeypatch.setattr(bootstrap, "_await_lane_id", lambda *_: minted)
    observed = {}

    class FakeProcessGroup:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def init_nccl_communicator(self, device):
            observed["device"] = device

    monkeypatch.setattr(bootstrap, "MxProcessGroup", FakeProcessGroup)
    bootstrap.mx_init_lane(state, 1)

    assert observed["root_unique_id"] is root_unique_id
    assert observed["unique_id"] == minted
    assert state.ids[1] == minted


def test_lane_fetch_requires_a_bootstrap_stamp_from_the_current_epoch(monkeypatch):
    from modelexpress_rl import refit_collective_pb2_grpc as pb_grpc

    stale = SimpleNamespace(lane_id=1, bootstrap_epoch=6, nccl_unique_id=b"s" * 128)
    current = SimpleNamespace(lane_id=1, bootstrap_epoch=7, nccl_unique_id=b"c" * 128)

    class Stub:
        def __init__(self):
            self.responses = [
                SimpleNamespace(epoch=7, lanes=[stale]),
                SimpleNamespace(epoch=7, lanes=[current]),
            ]

        def GetCollectiveGroup(self, request, timeout):
            return self.responses.pop(0)

    stub = Stub()
    monkeypatch.setattr(pb_grpc, "RefitCollectiveServiceStub", lambda channel: stub)
    monkeypatch.setattr("time.sleep", lambda _: None)
    state = bootstrap_state()
    state.channel = object()
    state.timeout_s = 1.0

    assert bootstrap._await_lane_id(state, 1) == b"c" * 128


def test_process_group_uses_root_uid_and_warms_up_the_communicator(monkeypatch):
    root_unique_id = object()
    events = []

    class FakeTensor:
        def __init__(self, value):
            self.value = value

    class FakeStream:
        cuda_stream = 17

        def synchronize(self):
            events.append("sync")

    class FakeCommunicator:
        @staticmethod
        def init(*, nranks, rank, unique_id):
            events.append(("init", nranks, rank, unique_id))
            return FakeCommunicator()

        def broadcast(self, *, sendbuf, recvbuf, root, stream):
            recvbuf.value = 1
            events.append(("broadcast", root, stream))

    class FakeUniqueId:
        @staticmethod
        def from_bytes(raw):
            raise AssertionError("rank zero must use the original unique-id object")

    communicator = ModuleType("nccl.core.communicator")
    communicator.Communicator = FakeCommunicator
    utils = ModuleType("nccl.core.utils")
    utils.UniqueId = FakeUniqueId
    monkeypatch.setitem(sys.modules, "nccl", ModuleType("nccl"))
    monkeypatch.setitem(sys.modules, "nccl.core", ModuleType("nccl.core"))
    monkeypatch.setitem(sys.modules, "nccl.core.communicator", communicator)
    monkeypatch.setitem(sys.modules, "nccl.core.utils", utils)
    monkeypatch.setattr(bootstrap.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(bootstrap.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(bootstrap.torch.cuda, "current_stream", FakeStream)
    monkeypatch.setattr(bootstrap.torch, "ones", lambda *args, **kwargs: FakeTensor(1))
    monkeypatch.setattr(bootstrap.torch, "zeros", lambda *args, **kwargs: FakeTensor(0))
    monkeypatch.setattr(
        bootstrap.torch, "allclose", lambda left, right: left.value == right.value
    )

    group = bootstrap.MxProcessGroup(
        unique_id=b"u" * 128,
        rank=0,
        world_size=2,
        root_unique_id=root_unique_id,
    )
    group.init_nccl_communicator(device=0)

    assert events == [
        ("init", 2, 0, root_unique_id),
        ("broadcast", 0, 17),
        "sync",
    ]
