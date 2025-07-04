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
from torch.distributed import get_process_group_ranks

from nemo_rl.models.megatron.converters.common import get_global_key_from_local_key


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
        print(f"Module {key} not found in named_modules_dict")
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
def gather_params(model, keys):
    st = time.time()

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = torch.distributed.get_world_size(tp_group)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)

    named_modules_dict = dict(model.named_modules())
    state_dict = model.state_dict()
    gathered_params = {}
    for local_key, shape, dtype in sorted(keys):
        if local_key in state_dict:
            param = state_dict[local_key]

            # Check if param is TP-sharded
            tp_dim = get_tp_dim(model, local_key, named_modules_dict)

            # If the parameter is TP-sharded, gather its slices on GPU.
            if tp_dim is not None:
                gathered_slices = [
                    torch.empty_like(param) for _ in range(tp_world_size)
                ]
                torch.distributed.all_gather(gathered_slices, param, group=tp_group)
                # TODO: why cast to torch.bfloat16 instead of param.dtype?
                full_param = torch.cat(gathered_slices, dim=tp_dim)
            else:
                # TODO: why do we need to clone?
                full_param = param
            global_key = get_global_key_from_local_key(local_key, model.config)
        else:
            #  params that may not be on every rank, e.g. the embedding layer
            global_key = None
            full_param = torch.empty(
                *shape, dtype=dtype, device=torch.cuda.current_device()
            )

        # gather across PP group
        pp_gathered_global_keys = [None] * pp_world_size
        torch.distributed.all_gather_object(
            pp_gathered_global_keys, global_key, group=pp_group
        )
        # To test no gather:
        # pp_gathered_global_keys = [global_key] * pp_world_size

        pp_gathered_params = [
            torch.empty(*shape, dtype=dtype, device=torch.cuda.current_device())
            for _ in range(pp_world_size)
        ]
        torch.distributed.all_gather(pp_gathered_params, full_param, group=pp_group)

        flat_gathered_global_keys = pp_gathered_global_keys
        flat_gathered_params = pp_gathered_params

        for k, p in zip(flat_gathered_global_keys, flat_gathered_params):
            if k is not None:
                gathered_params[k] = p

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Time taken to gather params: {time.time() - st}")
    return gathered_params


@torch.no_grad()
def get_param_info(model, dtype):
    # Get parallel info
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_group_rank_ids = get_process_group_ranks(tp_group)

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_group_rank_ids = get_process_group_ranks(pp_group)

    # Collect parameter info
    param_info = []

    # Dictionary of modules we can quickly look up to check if a module has TP
    named_modules_dict = dict(model.named_modules())

    # Process each parameter in the model
    # state_dict includes parameters and persistent buffers
    for name, param in model.state_dict().items():
        # Skip _extra_state entries (these are metadata, not actual weights)
        if "_extra_state" in name:
            continue

        shape = list(param.shape)
        tp_dim = get_tp_dim(model, name, named_modules_dict)
        if tp_dim is not None:
            tp_rank_ids = tuple(sorted(tp_group_rank_ids))
            shape[tp_dim] *= len(tp_rank_ids)
        else:
            tp_rank_ids = (torch.distributed.get_rank(),)

        pp_rank_ids = tuple(sorted(pp_group_rank_ids))

        # Calculate size for this parameter
        prec_to_bytes = {
            torch.bfloat16: 2,
            torch.float16: 2,
            torch.float32: 4,
        }
        scale = prec_to_bytes[dtype] / prec_to_bytes[param.dtype]
        size_in_bytes = (
            param.element_size()
            * param.numel()
            * len(tp_rank_ids)
            * len(pp_rank_ids)
            * scale
        )
        param_info.append(
            (
                (
                    name,
                    tuple(shape),
                    param.dtype,
                ),
                size_in_bytes,
            )
        )

    # Gather all parameter info from all PP ranks
    pp_gathered_param_infos = [None] * pp_world_size
    torch.distributed.all_gather_object(
        pp_gathered_param_infos, param_info, group=pp_group
    )
    pp_gathered_param_infos = [x for y in pp_gathered_param_infos for x in y]  # type: ignore

    all_param_infos = pp_gathered_param_infos

    # Merge all parameter infos, keeping only unique parameter names
    merged_param_info = []
    seen_params = set()

    for name, size in all_param_infos:
        if name not in seen_params:
            merged_param_info.append((name, size))
            seen_params.add(name)

    # Update param_info with the merged information
    param_info = merged_param_info
    print(f"Prepared {len(param_info)} tensors for refit")

    return param_info
