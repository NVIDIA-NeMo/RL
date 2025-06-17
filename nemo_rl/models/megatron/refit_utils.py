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
def gather_params(
    model,
    keys,
):
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
