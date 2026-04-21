from typing import Any

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor


def get_handle_from_tensor(tensor: torch.Tensor) -> tuple[Any]:
    """Get IPC handle from a tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    # Skip serializing the function for better refit performance.
    return reduce_tensor(tensor.detach())[1:]


def rebuild_cuda_tensor_from_ipc(
    cuda_ipc_handle: tuple, device_id: int
) -> torch.Tensor:
    """Rebuild a CUDA tensor from an IPC handle."""
    args = cuda_ipc_handle[0]
    list_args = list(args)
    list_args[6] = device_id
    return rebuild_cuda_tensor(*list_args)
