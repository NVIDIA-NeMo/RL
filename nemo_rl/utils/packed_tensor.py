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

import math
import os
from functools import lru_cache
from typing import Any, List, Tuple

import torch


@lru_cache(maxsize=1)
def get_target_packed_tensor_size():
    memory_ratio = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.01")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_memory_bytes = props.total_memory
    # max size is 5GB
    target_size = min(int(total_memory_bytes * float(memory_ratio)), 5 * 1024**3)
    return target_size


def packed_broadcast_producer(iterator, group, src, post_iter_func):
    """Broadcast a list of tensors in a packed manner.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_iter_func: function to apply to each tensor before packing, should return a tensor

    Returns:
        None

    """
    target_packed_tensor_size = get_target_packed_tensor_size()

    while True:
        # Form a packed tensor
        packing_tensor_list = []
        packing_tensor_sizes = 0
        try:
            while True:
                # Apply backend specific post processing and then convert to linearized uint8 tensor
                tensor = post_iter_func(next(iterator)).view(torch.uint8).view(-1)
                packing_tensor_list.append(tensor)
                packing_tensor_sizes += tensor.view(torch.uint8).numel()
                if packing_tensor_sizes > target_packed_tensor_size:
                    break
            # Pack the tensors and call broadcast collective
            packed_tensor = torch.cat(packing_tensor_list, dim=0)
            group.broadcast(packed_tensor, src=src)
        except StopIteration:
            # do the last broadcast if there are remaining tensors
            if len(packing_tensor_list) > 0:
                packed_tensor = torch.cat(packing_tensor_list, dim=0)
                group.broadcast(packed_tensor, src=src)
            break


def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
    """Consume a packed tensor and unpack it into a list of tensors.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_unpack_func: function to apply to each tensor after unpacking

    Returns:
        None

    """

    def unpack_tensor(
        packed_tensor: torch.Tensor, meta_data_list: list[Any]
    ) -> List[Tuple[str, torch.Tensor]]:
        """Unpack a single tensor into a list of tensors.

        Args:
            packed_tensor: the packed torch.uint8 tensor to unpack
            meta_data_list: List[(name, shape, dtype, offset, tensor_size)]

        Returns:
            unpacked List[(name, tensor)]
        """
        unpacked_list = []
        # Perform batched split with torch.split_with_sizes
        packed_tensor_sizes = list(map(lambda x: x[4], meta_data_list))
        unpacked_tensor = packed_tensor.split_with_sizes(packed_tensor_sizes)

        # unpacked_list = List[(name, torch.Tensor.view(dtype).view(*shape))]
        unpacked_list = [
            (
                meta_data_list[i][0],
                tensor.view(meta_data_list[i][2]).view(*meta_data_list[i][1]),
            )
            for i, tensor in enumerate(unpacked_tensor)
        ]

        return unpacked_list

    target_packed_tensor_size = get_target_packed_tensor_size()

    while True:
        # Form a packed tensor
        packing_tensor_meta_data = []
        packing_tensor_sizes = 0
        offset = 0
        try:
            while True:
                # Form a packed tensor
                name, (shape, dtype) = next(iterator)
                tensor_size = math.prod(shape) * dtype.itemsize
                packing_tensor_meta_data.append(
                    (name, shape, dtype, offset, tensor_size)
                )
                packing_tensor_sizes += tensor_size
                offset += tensor_size
                if packing_tensor_sizes > target_packed_tensor_size:
                    break
            # Create a packed tensor and broadcast it
            packed_tensor = torch.empty(
                packing_tensor_sizes, dtype=torch.uint8, device="cuda"
            )
            group.broadcast(packed_tensor, src=src)
            # Load the packed tensor into the model
            post_unpack_func(unpack_tensor(packed_tensor, packing_tensor_meta_data))
        except StopIteration:
            # do the last broadcast if there are remaining tensors
            if len(packing_tensor_meta_data) > 0:
                # Create a packed tensor and broadcast it
                packed_tensor = torch.empty(
                    packing_tensor_sizes, dtype=torch.uint8, device="cuda"
                )
                group.broadcast(packed_tensor, src=src)
                # Load the packed tensor into the model
                post_unpack_func(unpack_tensor(packed_tensor, packing_tensor_meta_data))
            break
