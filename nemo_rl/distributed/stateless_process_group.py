# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
import threading
import types
from typing import Optional

import torch
from nccl.core import SUM
from nccl.core.communicator import Communicator
from nccl.core.utils import UniqueId, get_unique_id

# Mirrors vLLM 0.20.0's StatelessProcessGroup.broadcast_obj key format and
# PyNcclCommunicator ncclUniqueId pickle type. Revalidate both when bumping vLLM:
# vllm/distributed/utils.py
# vllm/distributed/device_communicators/pynccl.py
# vllm/distributed/device_communicators/pynccl_wrapper.py
# https://github.com/vllm-project/vllm/tree/v0.20.0
_NEMO_UNIQUE_ID_KEY = "nccl_unique_id"
_VLLM_UNIQUE_ID_KEY = "broadcast_from/0/0"
_VLLM_NCCL_MODULE = "vllm.distributed.device_communicators.pynccl_wrapper"
_VLLM_PICKLE_LOCK = threading.Lock()


class _VllmNcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


_VllmNcclUniqueId.__module__ = _VLLM_NCCL_MODULE
_VllmNcclUniqueId.__name__ = "ncclUniqueId"
_VllmNcclUniqueId.__qualname__ = "ncclUniqueId"


def _pickle_vllm_unique_id(unique_id_bytes: bytes) -> bytes:
    """Serialize an NCCL unique ID in vLLM's metadata wire format.

    vLLM's stateless process group pickles its ``ncclUniqueId`` ctypes
    structure. Training workers do not install vLLM, so construct the same
    ctypes type under its canonical module name only while serializing.
    """
    if len(unique_id_bytes) != 128:
        raise ValueError(
            f"Expected a 128-byte NCCL unique ID, got {len(unique_id_bytes)} bytes."
        )

    module_names = [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.device_communicators",
        _VLLM_NCCL_MODULE,
    ]
    with _VLLM_PICKLE_LOCK:
        previous_modules = {name: sys.modules.get(name) for name in module_names}
        modules = {name: types.ModuleType(name) for name in module_names}
        for name in module_names[:-1]:
            modules[name].__path__ = []

        modules["vllm"].distributed = modules["vllm.distributed"]
        modules["vllm.distributed"].device_communicators = modules[
            "vllm.distributed.device_communicators"
        ]
        modules["vllm.distributed.device_communicators"].pynccl_wrapper = modules[
            _VLLM_NCCL_MODULE
        ]

        modules[_VLLM_NCCL_MODULE].ncclUniqueId = _VllmNcclUniqueId

        try:
            sys.modules.update(modules)
            unique_id = _VllmNcclUniqueId.from_buffer_copy(unique_id_bytes)
            payload = pickle.dumps(unique_id)
        finally:
            for name, previous_module in previous_modules.items():
                if previous_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous_module
        return payload


class StatelessProcessGroup:
    def __init__(self, master_address: str, port: int, rank: int, world_size: int):
        self.master_address = master_address
        self.port = port
        self.rank = rank
        self.world_size = world_size
        self.tcp_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
        )

    def init_nccl_communicator(self, device: int, *, peer: str = "nemo") -> None:
        """Initialize NCCL using the metadata and warmup protocol of the peer."""
        if peer not in ("nemo", "vllm"):
            raise ValueError(f"Unsupported NCCL peer protocol: {peer!r}.")

        if self.rank == 0:
            unique_id = get_unique_id()
            unique_id_bytes = unique_id.as_bytes
            self.tcp_store.set(_NEMO_UNIQUE_ID_KEY, unique_id_bytes)
            if peer == "vllm":
                self.tcp_store.set(
                    _VLLM_UNIQUE_ID_KEY,
                    _pickle_vllm_unique_id(unique_id_bytes),
                )
        else:
            self.tcp_store.wait([_NEMO_UNIQUE_ID_KEY])
            unique_id_bytes = self.tcp_store.get(_NEMO_UNIQUE_ID_KEY)
            unique_id = UniqueId.from_bytes(unique_id_bytes)

        with torch.cuda.device(device):
            self.nccl_communicator = Communicator.init(
                nranks=self.world_size,
                rank=self.rank,
                unique_id=unique_id,
            )
            stream = torch.cuda.current_stream()
            if peer == "vllm":
                # Match PyNcclCommunicator's first collective exactly.
                data = torch.zeros(1, device=device)
                self.nccl_communicator.allreduce(
                    sendbuf=data,
                    recvbuf=data,
                    op=SUM,
                    stream=int(stream.cuda_stream),
                )
            else:
                if self.rank == 0:
                    data = torch.ones(1, device=device)
                else:
                    data = torch.zeros(1, device=device)
                self.broadcast(data, 0, stream=stream)
            stream.synchronize()
            if peer == "nemo":
                assert torch.allclose(data, torch.ones(1, device=device))

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )
