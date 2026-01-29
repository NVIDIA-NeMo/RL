from typing import Optional

import torch

try:
    from nccl.core.communicator import Communicator
    from nccl.core.utils import UniqueId, get_unique_id
except ImportError:
    raise ImportError(
        "nccl is not installed. Please install nccl using `pip install nccl4py`."
    )


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

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=stream
        )
