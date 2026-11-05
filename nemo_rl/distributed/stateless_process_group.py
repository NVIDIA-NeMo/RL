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

from nemo_rl.distributed.refit_watchdog import RefitAborted

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
        # Declared here rather than sprung into existence by init_nccl_communicator, so
        # abort() can tell "never initialized" from "initialized" without hasattr.
        self.nccl_communicator: Optional[Communicator] = None
        # Whether this group was aborted, as opposed to never built. Both leave
        # nccl_communicator None, but they are different failures and the collectives
        # below have to report them differently -- see broadcast().
        self._aborted = False
        # Optional because abort() releases it: a run that recovers repeatedly would
        # otherwise hold a bound store per recovery for the life of the worker.
        self.tcp_store: Optional[torch.distributed.TCPStore] = (
            torch.distributed.TCPStore(
                host_name=self.master_address,
                port=self.port,
                world_size=self.world_size,
                is_master=(self.rank == 0),
            )
        )

    def abort(self) -> None:
        """Terminate in-flight operations and release the communicator.

        Idempotent, and safe on a group whose communicator was never built.

        **`abort()`, not `destroy()`, is the correct teardown here.** NCCL documents
        `destroy` as an intra-node collective that every rank must call or it hangs --
        precisely what a rank whose process has died cannot do. `abort` terminates
        outstanding operations instead, so it works whether or not the peers are alive,
        which makes it the only safe choice on a path that exists to handle dead peers.

        Verified on 2xA6000: with a peer SIGKILLed mid-broadcast, a survivor blocked in
        the collective was released 0.15s after another thread called abort().

        The rendezvous store is dropped too. Each rebuild gets a fresh port, so holding
        the old one costs nothing functionally, but a run that recovers repeatedly would
        otherwise accumulate a bound TCPStore per recovery for the life of the worker.

        **The split children are aborted first, and they are a third communicator family.**
        The Python reshard path splits this communicator per replica group and caches the
        children; NCCL gives a split child its own abort flag unless ``splitShare`` is set
        (it defaults to 0), so aborting this communicator does not reach them. A rank
        blocked on a child would never be released, and since it never returns, the
        watchdog's guarded block never exits to have ``fired`` read -- a hang no
        exception-translation fix can reach.

        Imported locally to keep this module free of a ``weight_sync`` dependency at module
        scope.
        """
        # Before the parent: once nccl_communicator is None the cache keys derived from it
        # cannot be recovered, and the children would be stranded as well as un-aborted.
        from nemo_rl.weight_sync.xferdtensor_python import (
            abort_xferdtensor_python_subcommunicators,
        )

        abort_xferdtensor_python_subcommunicators(self)

        communicator, self.nccl_communicator = self.nccl_communicator, None
        self.tcp_store = None
        self._aborted = True
        if communicator is not None:
            communicator.abort()

    def init_nccl_communicator(self, device: int, *, peer: str = "nemo") -> None:
        """Initialize NCCL using the metadata and warmup protocol of the peer.

        ``peer="nemo"`` publishes the raw 128-byte unique ID under
        ``nccl_unique_id`` and warms up with a rank-zero broadcast.
        ``peer="vllm"`` additionally publishes vLLM's pickled ``ncclUniqueId``
        under ``broadcast_from/0/0`` and warms up with an all-reduce, matching
        ``PyNcclCommunicator``. The receiver protocol is not negotiable, so a
        generation backend must select the peer it implements.
        """
        if peer not in ("nemo", "vllm"):
            raise ValueError(f"Unsupported NCCL peer protocol: {peer!r}.")

        if self.tcp_store is None:
            raise RuntimeError(
                "StatelessProcessGroup has no rendezvous store: the group was aborted. "
                "Construct a new one rather than re-initializing this."
            )

        if self.rank == 0:
            unique_id = get_unique_id()
            unique_id_bytes = unique_id.as_bytes
            # The torch stub types `value` as str, but TCPStore.set accepts bytes and
            # round-trips them byte-for-byte (verified directly). Bytes is also required
            # here, not incidental: a NCCL UniqueId is binary and would not survive a
            # str round trip. Surfaced when this file entered pyrefly's scope -- main
            # does not check this file, which is why upstream carries no annotation.
            self.tcp_store.set(
                _NEMO_UNIQUE_ID_KEY,
                unique_id_bytes,  # pyrefly: ignore[bad-argument-type]
            )
            if peer == "vllm":
                self.tcp_store.set(
                    _VLLM_UNIQUE_ID_KEY,
                    _pickle_vllm_unique_id(
                        unique_id_bytes
                    ),  # pyrefly: ignore[bad-argument-type]
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
        # Snapshotted, not read twice. The watchdog thread nulls this field from under
        # us, so a check-then-call on the attribute can pass the check and then raise
        # AttributeError: 'NoneType' has no attribute 'broadcast'.
        communicator = self.nccl_communicator
        if communicator is None:
            if self._aborted:
                # A refit is many broadcasts, not one. The watchdog aborts the collective
                # that is in flight -- that one returns cleanly, by NCCL's contract, and
                # the caller learns about it from ``guard.fired``. But the *next* buffer
                # in the same refit arrives here, and if it raised a bare RuntimeError it
                # would escape the caller's ``with`` block before ``guard.fired`` is ever
                # read: no RefitAborted, no recovery, and the run dies reporting a
                # missing communicator instead of the abort that caused it.
                #
                # Naming the real cause here fixes every guarded site at once, which is
                # why it lives in the group rather than in a try/except around each of
                # the four callers.
                raise RefitAborted(
                    "the refit process group was aborted mid-collective, so this and "
                    "every later operation on it fails; the abort is the cause, not "
                    "this call"
                )
            raise RuntimeError(
                "StatelessProcessGroup has no communicator: "
                "init_nccl_communicator() was never called."
            )
        if stream is None:
            stream = torch.cuda.current_stream()
        communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )
