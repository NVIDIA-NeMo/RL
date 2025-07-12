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
import re
import time
from typing import Dict, List
from collections import defaultdict

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from nemo_rl.utils.nvml import get_free_memory_bytes

def get_tp_dim(model, param_name, named_modules_dict):
    # pass in named_modules_dict so we can get it ahead of time instead
    # of once for each param
    pattern = re.compile(r"\.(?:weight|bias)\d*$")
    if not pattern.search(param_name):
        return None

    prefix = ""
    if hasattr(model, "module"):
        prefix = "module."
        if hasattr(model.module, "module"):
            prefix = "module.module."
    key = prefix + ".".join(param_name.split(".")[:-1])
    module = named_modules_dict.get(key)
    if module is None:
        return None
    if hasattr(module, "parallel_mode") and module.parallel_mode is not None:
        # TE layers sometimes have parallel_mode we can check directly
        if module.parallel_mode == "column":
            return 0
        elif module.parallel_mode == "row":
            return 1
        else:
            return None
    elif isinstance(
        module,
        (
            VocabParallelEmbedding,
            ColumnParallelLinear,
            TEColumnParallelGroupedLinear,
            TEColumnParallelLinear,
        ),
    ):
        return 0
    elif isinstance(
        module, (RowParallelLinear, TERowParallelGroupedLinear, TERowParallelLinear)
    ):
        return 1
    else:
        return None


@torch.no_grad()
def gather_params(model, keys, key_to_global_keys: Dict[str, List[str]]):
    st = time.perf_counter()

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = torch.distributed.get_world_size(tp_group)
    etp_group = parallel_state.get_expert_tensor_parallel_group()
    etp_world_size = torch.distributed.get_world_size(etp_group)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_global_ranks = torch.distributed.get_process_group_ranks(group=pp_group)
    pp_local_rank_id = parallel_state.get_pipeline_model_parallel_rank()
    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_world_size = torch.distributed.get_world_size(ep_group)

    named_modules_dict = dict(model.named_modules())
    state_dict = model.state_dict()
    gathered_params = {}
    ep_pattern = re.compile(r"mlp\.experts.*\.weight\d*$")

    for local_key, owner_pp_local_rank_id, shape, dtype in sorted(keys):
        if local_key in state_dict and owner_pp_local_rank_id == pp_local_rank_id:
            param = state_dict[local_key]

            tp_dim = get_tp_dim(model, local_key, named_modules_dict)

            # If the parameter is TP-sharded, gather its slices on GPU.
            if tp_dim is not None:
                if ep_pattern.search(local_key):
                    world_size = etp_world_size
                    group = etp_group
                else:
                    world_size = tp_world_size
                    group = tp_group

                gathered_slices = [torch.empty_like(param) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_slices, param, group=group)
                full_param = torch.cat(gathered_slices, dim=tp_dim)
            else:
                full_param = param
        else:
            full_param = torch.empty(
                *shape, dtype=dtype, device=torch.cuda.current_device()
            )

        # Broadcast across PP group.
        src_global_rank = pp_global_ranks[owner_pp_local_rank_id]

        # Broadcast from the rank that has the parameter
        torch.distributed.broadcast(full_param, src=src_global_rank, group=pp_group)
        pp_gathered_params = [full_param]

        # gather across EP group
        if ep_pattern.search(local_key):
            stacked_pp_gathered_params = torch.stack(pp_gathered_params)

            ep_gathered_params = [
                torch.empty(
                    stacked_pp_gathered_params.shape,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                )
                for _ in range(ep_world_size)
            ]
            torch.distributed.all_gather(
                ep_gathered_params, stacked_pp_gathered_params, group=ep_group
            )
            flat_gathered_params = [
                x for y in ep_gathered_params for x in torch.unbind(y)
            ]

        else:
            flat_gathered_params = pp_gathered_params

        flat_gathered_global_keys = key_to_global_keys[
            (local_key, owner_pp_local_rank_id)
        ]
        for k, p in zip(flat_gathered_global_keys, flat_gathered_params):
            if k is not None:
                gathered_params[k] = p

    print(f"Time taken to gather params: {time.perf_counter() - st}")
    return gathered_params

def get_param_info(model, precision, cached_param_info):

    # get parallel info
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = torch.distributed.get_world_size(tp_group)
    tp_group_rank_ids = torch.distributed.get_process_group_ranks(group=tp_group)
    etp_group = parallel_state.get_expert_tensor_parallel_group()
    etp_world_size = torch.distributed.get_world_size(etp_group)
    etp_group_rank_ids = torch.distributed.get_process_group_ranks(group=etp_group)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_group_rank_ids = torch.distributed.get_process_group_ranks(group=pp_group)
    pp_local_rank_id = parallel_state.get_pipeline_model_parallel_rank()
    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_world_size = torch.distributed.get_world_size(ep_group)
    ep_group_rank_ids = torch.distributed.get_process_group_ranks(group=ep_group)

    # collect parameter info
    param_info = []
    named_modules_dict = dict(model.named_modules())
    ep_pattern = re.compile(r"mlp\.experts.*\.weight\d*$")

    # process each parameter in the model
    need_update_cache = False
    cached_param_info = iter(cached_param_info)
    for name, param in model.state_dict().items():
        if "_extra_state" in name:
            continue

        is_expert = ep_pattern.search(name)
        if is_expert:
            tensor_mp_rank_ids = etp_group_rank_ids
        else:
            tensor_mp_rank_ids = tp_group_rank_ids

        shape = list(param.shape)
        tp_dim = get_tp_dim(model, name, named_modules_dict)
        if tp_dim is not None:
            tp_rank_ids = tuple(sorted(tensor_mp_rank_ids))
            shape[tp_dim] *= len(tp_rank_ids)
        else:
            tp_rank_ids = (torch.distributed.get_rank(),)

        if is_expert:
            ep_rank_ids = tuple(sorted(ep_group_rank_ids))
        else:
            ep_rank_ids = (torch.distributed.get_rank(),)

        prec_to_bytes = {
            torch.bfloat16: 2,
            torch.float16: 2,
            torch.float32: 4,
        }
        scale = prec_to_bytes[precision] / prec_to_bytes[param.dtype]
        size_in_bytes = (
            param.element_size() * param.numel() * len(tensor_mp_rank_ids) * len(ep_rank_ids) * scale
        )
        if is_expert:
            param_info.append(
                (
                    (
                        name, 
                        pp_local_rank_id,
                        tuple(shape),
                        param.dtype,
                    ),
                    size_in_bytes,
                )
            )   
        else:
            param_info.append(
                (
                    (
                        name, 
                        pp_local_rank_id,
                        tuple(shape),
                        param.dtype,
                    ),
                    size_in_bytes,
                )
            )
        
        # check against cached param info. If it has next element, check if it equals to the new appended entry
        if (not need_update_cache) and (next(cached_param_info, None) != param_info[-1]):
            need_update_cache = True

    # gather parameter info from all PP ranks to ensure complete coverage
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_gathered_param_infos = [None] * pp_world_size
    torch.distributed.all_gather_object(
        pp_gathered_param_infos, 
        param_info, 
        group=pp_group
    )
    pp_gathered_param_infos = [x for y in pp_gathered_param_infos for x in y]

    # Merge all parameter infos, keeping only unique parameter names
    merged_param_info = []
    seen_params = set()

    for name, size in pp_gathered_param_infos:
        if name not in seen_params:
            merged_param_info.append((name, size))
            seen_params.add(name)

    param_info = merged_param_info

    return param_info, need_update_cache

def bucketize_params(param_info, refit_buffer_size_gb=None):
    if refit_buffer_size_gb is None:
        refit_buffer_size_gb = get_free_memory_bytes(torch.cuda.current_device()) * 0.1
    refit_buffer_size_bytes = refit_buffer_size_gb * 1024**3

    num_buckets = 0
    max_bucket_size = -1
    mem_budget_for_this_bucket = refit_buffer_size_bytes
    assert refit_buffer_size_bytes > 0, "Refit buffer size must be greater than 0"

    buckets_of_param_info = []
    current_bucket_of_param_info = []

    for info, size_in_bytes in param_info:
        if size_in_bytes > mem_budget_for_this_bucket:
            if len(current_bucket_of_param_info) == 0:
                raise ValueError("Refit buffer size is smaller than the largest required full param size.")
            else:
                # otherwise, commit the previous bucket and start a new one
                buckets_of_param_info.append(current_bucket_of_param_info)
                current_bucket_of_param_info = [info]
                mem_budget_for_this_bucket = refit_buffer_size_bytes - size_in_bytes
        else:
            current_bucket_of_param_info.append(info)
            mem_budget_for_this_bucket -= size_in_bytes

    # commit the last bucket
    if len(current_bucket_of_param_info) > 0:
        buckets_of_param_info.append(current_bucket_of_param_info)
    
    num_buckets = len(buckets_of_param_info)
    max_bucket_size = defaultdict(lambda: 0)
    for bucket in buckets_of_param_info:
        current_bucket_size = defaultdict(lambda: 0)
        for (name, pp_local_rank_id, shape, param_dtype) in bucket:
            current_bucket_size[param_dtype] += np.prod(shape)
        for dtype, size in current_bucket_size.items():
            if size > max_bucket_size[dtype]:
                max_bucket_size[dtype] = size

    return buckets_of_param_info, num_buckets, max_bucket_size

        
        