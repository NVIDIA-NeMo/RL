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

"""Pipeline parallel utilities for Megatron models."""

from typing import Any, Optional

import torch
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    is_pipeline_last_stage,
)


def broadcast_obj_from_pp_rank(obj: Any) -> Any:
    """Broadcast an object across pipeline parallel ranks.
    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def broadcast_loss_metrics_from_last_stage(loss_metrics: Optional[list] = None) -> list:
    """Broadcast loss metrics from the last pipeline stage to all stages.

    This utility handles the common pattern where loss computation happens on the last
    pipeline stage and needs to be broadcast to all other stages.

    Args:
        loss_metrics: List of loss metrics if on last stage, None otherwise

    Returns:
        List of loss metrics on all ranks
    """
    pp_group = get_pipeline_model_parallel_group()
    last_rank = get_pipeline_model_parallel_last_rank()

    if is_pipeline_last_stage(ignore_virtual=True):
        metrics_to_broadcast = [loss_metrics]
        torch.distributed.broadcast_object_list(
            metrics_to_broadcast,
            src=last_rank,
            group=pp_group,
        )
        return loss_metrics
    else:
        metrics_to_broadcast = [None]
        torch.distributed.broadcast_object_list(
            metrics_to_broadcast,
            src=last_rank,
            group=pp_group,
        )
        return metrics_to_broadcast[0]


def broadcast_tensors_from_last_stage(
    tensors: dict[str, Optional[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Broadcast multiple tensors from the last pipeline stage to all stages.

    Args:
        tensors: Dictionary mapping tensor names to tensors (None on non-last stages)
        pp_group: Pipeline parallel group (auto-detected if None)

    Returns:
        Dictionary of broadcasted tensors on all ranks
    """
    pp_group = get_pipeline_model_parallel_group()

    from nemo_rl.models.megatron.common import broadcast_tensor

    last_rank = get_pipeline_model_parallel_last_rank()
    current_rank = torch.distributed.get_rank()

    broadcasted_tensors = {}

    if is_pipeline_last_stage(ignore_virtual=True):
        # Broadcast tensors from last stage
        for name, tensor in tensors.items():
            if tensor is not None:
                broadcasted_tensors[name] = broadcast_tensor(
                    tensor, current_rank, pp_group
                )
            else:
                broadcasted_tensors[name] = None
    else:
        # Receive tensors on other stages
        for name in tensors.keys():
            broadcasted_tensors[name] = broadcast_tensor(None, last_rank, pp_group)

    return broadcasted_tensors
