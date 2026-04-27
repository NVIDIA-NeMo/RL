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
"""IPC helpers for sharing CUDA tensors across processes.

Used by cross-tokenizer off-policy distillation to ship teacher logits
from the teacher policy worker to the student worker without an extra
host roundtrip.
"""

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
    """Rebuild a CUDA tensor from an IPC handle on ``device_id``."""
    args = cuda_ipc_handle[0]
    list_args = list(args)
    list_args[6] = device_id
    return rebuild_cuda_tensor(*list_args)
