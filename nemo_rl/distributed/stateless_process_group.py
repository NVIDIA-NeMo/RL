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
        self._retired_nccl_communicators: list[Communicator] = []
        self.tcp_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
        )

    def init_nccl_communicator(self, device: int):
        UNIQUE_ID_KEY = "nccl_unique_id"

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
        """Shrink the NCCL communicator around failed ranks.

        Every rank that remains in the communicator must call this method with
        the same ``exclude_ranks``. Excluded ranks must not call it.

        Args:
            exclude_ranks: Ranks in the current communicator to remove.

        Returns:
            The old rank, compacted new rank, and new communicator world size.
        """
        if not exclude_ranks:
            raise ValueError("exclude_ranks must contain at least one rank")

        excluded = sorted(set(exclude_ranks))
        if len(excluded) != len(exclude_ranks):
            raise ValueError(f"exclude_ranks contains duplicates: {exclude_ranks}")
        if excluded[0] < 0 or excluded[-1] >= self.world_size:
            raise ValueError(
                f"exclude_ranks must be in [0, {self.world_size}), got {excluded}"
            )
        if self.rank in excluded:
            raise ValueError(
                f"Excluded rank {self.rank} must not call communicator shrink"
            )

        # CommShrinkFlag.DEFAULT requires all outstanding communicator work to
        # be complete. packed_tensor uses multiple CUDA streams, so quiesce the
        # device before entering this collective operation.
        torch.cuda.synchronize()

        old_rank = self.rank
        old_communicator = self.nccl_communicator
        self.nccl_communicator = old_communicator.shrink(
            exclude_ranks=excluded,
            config=NCCLConfig(shrink_share=True),
        )

        # The child shares resources with its parent. Keep the parent alive for
        # the process lifetime instead of destroying and recreating resources.
        self._retired_nccl_communicators.append(old_communicator)
        self.rank -= sum(excluded_rank < old_rank for excluded_rank in excluded)
        self.world_size -= len(excluded)

        return old_rank, self.rank, self.world_size

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )
