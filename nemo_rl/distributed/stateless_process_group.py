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
from nccl.core.communicator import NCCLConfig, Communicator
from nccl.core.utils import UniqueId, get_unique_id


class StatelessProcessGroup:
    def __init__(self, master_address: str, port: int, rank: int, world_size: int):
        self.master_address = master_address
        self.port = port
        self.rank = rank
        self.world_size = world_size
        self.device = 0
        self._retired_nccl_communicators: list[Communicator] = []
        self.tcp_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
        )

    def init_nccl_communicator(self, device: int):
        UNIQUE_ID_KEY = "nccl_unique_id"
        self.device = device

        if self.rank == 0:
            unique_id = get_unique_id()
            unique_id_bytes = unique_id.as_bytes
            # Rank 0: store unique_id to TCPStore
            self.tcp_store.set(UNIQUE_ID_KEY, unique_id_bytes)
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

    def shrink(self, exclude_ranks: list[int]) -> tuple[int, int, int]:
        """Shrink the NCCL communicator around failed ranks."""
        # CommShrinkFlag.DEFAULT requires all outstanding communicator work to
        # be complete. packed_tensor uses multiple CUDA streams, so quiesce the
        # device before entering this collective operation.
        torch.cuda.synchronize()

        old_rank = self.rank
        existing_communicator = self.nccl_communicator
        self.nccl_communicator = existing_communicator.shrink(
            exclude_ranks=exclude_ranks,
            config=NCCLConfig(shrink_share=True),
        )

        # The child shares resources with its parent. Keep the parent alive for
        # the process lifetime instead of destroying and recreating resources.
        self._retired_nccl_communicators.append(existing_communicator)
        self.rank -= sum(excluded_rank < old_rank for excluded_rank in exclude_ranks)
        self.world_size -= len(exclude_ranks)

        return old_rank, self.rank, self.world_size

    def get_grow_unique_id(self) -> bytes:
        """Generate a one-use identifier for growing this communicator."""
        torch.cuda.synchronize()
        return self.nccl_communicator.get_unique_id().as_bytes

    def grow(
        self,
        new_world_size: int,
        *,
        unique_id_bytes: bytes | None = None,
        new_rank: int | None = None,
    ) -> tuple[int, int, int]:
        """Grow the communicator with existing or newly added ranks."""
        torch.cuda.synchronize()
        unique_id = (
            UniqueId.from_bytes(unique_id_bytes)
            if unique_id_bytes is not None
            else None
        )
        old_rank = self.rank
        existing_communicator = self.nccl_communicator

        with torch.cuda.device(self.device):
            # newly added ranks
            if new_rank is not None:
                self.nccl_communicator = Communicator().grow(
                    nranks=new_world_size,
                    unique_id=unique_id,
                    rank=new_rank,
                )
            else:
                # existing ranks
                self.nccl_communicator = existing_communicator.grow(
                    nranks=new_world_size,
                    unique_id=unique_id,
                )

        if new_rank is not None:
            self._retired_nccl_communicators.append(existing_communicator)
            self.rank = new_rank
        else:
            existing_communicator.destroy()

        self.world_size = new_world_size
        return old_rank, self.rank, self.world_size

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )
