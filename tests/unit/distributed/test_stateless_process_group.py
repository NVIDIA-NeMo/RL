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

import ctypes
import pickle
import sys
import types
from contextlib import contextmanager, nullcontext

import pytest

from nemo_rl.distributed import stateless_process_group as spg


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
