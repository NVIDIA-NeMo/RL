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
from typing import Optional, TypeVar

import torch

T = TypeVar("T")


def rebalance_nd_tensor(
    tensor: torch.Tensor, group: Optional[torch.distributed.ProcessGroup] = None
) -> torch.Tensor:
    """Takes tensors with variable leading sizes (at dim=0) and stacks them into a single tensor.

    This function handles the case where different GPUs have tensors with different batch sizes
    and combines them into a single balanced tensor across all ranks.

    For example, with 3 GPUs:
        GPU0: tensor of shape [3, D]
        GPU1: tensor of shape [5, D]
        GPU2: tensor of shape [2, D]

    After rebalancing:
        All GPUs will have the same tensor of shape [10, D] (3+5+2=10)

    NOTE: assumes all other (i.e., non-zero) dimensions are equal.
    """
    num_samples = torch.as_tensor(
        tensor.size(0), dtype=torch.int64, device=torch.cuda.current_device()
    )
    batch_num_per_rank = torch.zeros(
        torch.distributed.get_world_size(group),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_gather_into_tensor(
        batch_num_per_rank, num_samples, group=group
    )

    B = batch_num_per_rank.sum()
    other_dims = tensor.shape[1:]
    dims = (B, *other_dims)

    indices = batch_num_per_rank.cumsum(dim=0)
    output_tensor = torch.zeros(
        *dims, dtype=tensor.dtype, device=torch.cuda.current_device()
    )

    # tensor_split is a view we can copy into
    output_tensor.tensor_split(indices[0:-1].cpu())[
        torch.distributed.get_rank(group=group)
    ].copy_(tensor)
    torch.distributed.all_reduce(output_tensor, group=group)
    return output_tensor


def gather_jagged_object_lists(
    local_objects: list[T], group: Optional[torch.distributed.ProcessGroup] = None
) -> list[T]:
    """Gathers jagged lists of picklable objects from all ranks and flattens them into a single list.

    This function handles the case where different GPUs have lists of different lengths
    and combines them into a single list containing all objects from all ranks.

    For example, with 3 GPUs:
        GPU0: [obj0, obj1]
        GPU1: [obj2, obj3, obj4]
        GPU2: [obj5]

    After gathering:
        All GPUs will have: [obj0, obj1, obj2, obj3, obj4, obj5]

    WARNING: synchronous

    Args:
        local_objects: List of objects to gather from current rank
        group: Optional process group

    Returns:
        Flattened list of all objects from all ranks in order [rank0, rank1, ...]
    """
    # Gather all lists across ranks
    world_size = torch.distributed.get_world_size(group=group)
    gathered_lists: list[list[T]] = [None] * world_size  # type: ignore
    torch.distributed.all_gather_object(gathered_lists, local_objects, group=group)

    # Flatten into single list while preserving order
    return [obj for sublist in gathered_lists for obj in sublist]
