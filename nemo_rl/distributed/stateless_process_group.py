# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import torch
from nccl.core.communicator import Communicator
from nccl.core.utils import UniqueId, get_unique_id


class StatelessProcessGroup:
    def __init__(self, master_address: str, port: int, rank: int, world_size: int):
        self.master_address = master_address
        self.port = port
        self.rank = rank
        self.world_size = world_size
        # Declared here rather than sprung into existence by init_nccl_communicator, so
        # abort() can tell "never initialized" from "initialized" without hasattr.
        self.nccl_communicator: Optional[Communicator] = None
        self.tcp_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
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
        """
        communicator, self.nccl_communicator = self.nccl_communicator, None
        if communicator is not None:
            communicator.abort()

    def init_nccl_communicator(self, device: int):
        UNIQUE_ID_KEY = "nccl_unique_id"

        if self.rank == 0:
            unique_id = get_unique_id()
            unique_id_bytes = unique_id.as_bytes
            # Rank 0: store unique_id to TCPStore
            # The torch stub types `value` as str, but TCPStore.set accepts bytes and
            # round-trips them byte-for-byte (verified directly). Bytes is also required
            # here, not incidental: a NCCL UniqueId is binary and would not survive a
            # str round trip. Surfaced when this file entered pyrefly's scope.
            self.tcp_store.set(
                UNIQUE_ID_KEY,
                unique_id_bytes,  # pyrefly: ignore[bad-argument-type]
            )
        else:
            # Other ranks: get unique_id from TCPStore
            self.tcp_store.wait([UNIQUE_ID_KEY])
            unique_id_bytes = self.tcp_store.get(UNIQUE_ID_KEY)
            unique_id = UniqueId.from_bytes(unique_id_bytes)

        with torch.cuda.device(device):
            self.nccl_communicator = Communicator.init(
                nranks=self.world_size,
                rank=self.rank,
                unique_id=unique_id,
            )
            # warmup and check if broadcast is working
            stream = torch.cuda.current_stream()
            if self.rank == 0:
                data = torch.ones(1, device=device)
            else:
                data = torch.zeros(1, device=device)
            self.broadcast(data, 0, stream=stream)
            torch.cuda.current_stream().synchronize()
            assert torch.allclose(data, torch.ones(1, device=device))

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        if self.nccl_communicator is None:
            raise RuntimeError(
                "StatelessProcessGroup has no communicator: init_nccl_communicator() "
                "was never called, or the group was aborted and not rebuilt."
            )
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )
