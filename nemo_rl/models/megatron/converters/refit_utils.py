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
from typing import Dict, List, Tuple

import torch
from megatron.core import parallel_state
from nemo.collections.llm.gpt.model.base import GPTConfig

import nemo_rl.models.megatron.converters as model_converters


def get_param_conversion_recipe_dict(
    name, converter_type: model_converters.ModelType, model_cfg: GPTConfig
):
    converter_dict = model_converters.REGISTRY[converter_type]

    local_layer = model_converters.get_local_layer_num(name)
    global_layer = (
        model_converters.get_global_layer_num(name, model_cfg)
        if local_layer is not None
        else None
    )
    format_dict = model_converters.SafeDict(l=local_layer, gl=global_layer)

    formatted_mapping = {
        k.format_map(format_dict): rec for k, rec in converter_dict.items()
    }
    return formatted_mapping, format_dict


@torch.no_grad()
def get_global_param_key_to_local_key_map(
    model, model_cfg: GPTConfig, keys: List[Tuple[str, str]]
) -> Dict[str, Tuple[int, str]]:
    """Get a mapping from global parameter keys to local parameter keys.

    Args:
        model: The model to get the mapping for.
        model_cfg: The model configuration.
        keys: The keys to get the mapping for. Tuple of (local_key, global_hf_key)

    Returns:
        A dictionary mapping global parameter keys to a tuple of (rank, local parameter key).
    """
    # Initialize pipeline parallel group information.
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_global_rank_ids = model_converters.get_all_rank_ids_in_group(pp_group)

    # Build a mapping on each PP rank from a computed global key to the raw state dict key.
    # The global key is computed by replacing the local layer number (after "layers.")
    # with its corresponding global layer number (if applicable).
    local_map = {}
    state_dict = model.state_dict()
    for local_key, _ in keys:
        if local_key not in state_dict:
            continue
        local_layer = model_converters.get_local_layer_num(local_key)
        if local_layer is not None:
            global_layer = model_converters.get_global_layer_num(local_key, model_cfg)
            # Replace the first occurrence of the digits after "layers." with the global layer number.
            global_key = re.sub(
                r"(?<=layers\.)\d+", str(global_layer), local_key, count=1
            )
        else:
            global_key = local_key
        local_map[global_key] = local_key

    # Gather the local maps from all PP ranks (only lightweight key info is gathered).
    all_maps = [None] * pp_world_size
    torch.distributed.all_gather_object(all_maps, local_map, group=pp_group)

    # Build the union over global keys and assign an owner (the rank with the smallest PP rank).
    union_global_map = {}
    for pp_rank, omap in enumerate(all_maps):
        for gk, raw_key in omap.items():
            if (
                gk not in union_global_map
                or pp_global_rank_ids[pp_rank] < union_global_map[gk][0]
            ):
                union_global_map[gk] = (pp_global_rank_ids[pp_rank], raw_key)
            else:
                print(
                    f"WARNING: {gk} already in union_global_map when gathering keys",
                    flush=True,
                )

    return union_global_map


@torch.no_grad()
def gather_and_convert_params(
    model,
    converter_type: model_converters.ModelType,
    model_cfg: GPTConfig,
    param_name_to_rank_and_key,
):
    import time

    st = time.time()
    # Process each parameter (by its unique global key) one at a time.
    gathered_params = {}
    state_dict = model.state_dict()
    for gk in sorted(param_name_to_rank_and_key.keys()):
        owner_pp_global_rank, owner_raw_key = param_name_to_rank_and_key[gk]

        # Only the owner PP rank has the parameter locally.
        if torch.distributed.get_rank() == owner_pp_global_rank:
            param = state_dict[owner_raw_key]

            # Use the conversion dict to get the appropriate recipe for this parameter.
            recipe_dict, format_dict = get_param_conversion_recipe_dict(
                owner_raw_key, converter_type, model_cfg
            )
            recipe = recipe_dict.get(owner_raw_key, None)
            if recipe is None and "_extra_state" not in owner_raw_key:
                print(
                    f"WARNING: {owner_raw_key} has no recipe mapping for conversion",
                    flush=True,
                )
                hf_mapping = {"None": None}
            else:
                # If the parameter is TP-sharded, gather its slices on GPU.
                if recipe.get("tp", None) is not None:
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    tp_world_size = torch.distributed.get_world_size(tp_group)
                    gathered_slices = [
                        torch.empty_like(param) for _ in range(tp_world_size)
                    ]
                    torch.distributed.all_gather(gathered_slices, param, group=tp_group)
                    full_param = torch.cat(gathered_slices, dim=recipe["tp"]).to(
                        torch.bfloat16
                    )
                else:
                    full_param = torch.clone(param).to(torch.bfloat16)

                # Convert the parameter using the provided function or mapping.
                if recipe.get("hf_func", None) is not None:
                    hf_mapping = recipe["hf_func"](full_param, model_cfg)
                    hf_mapping = {
                        k.format_map(format_dict): v for k, v in hf_mapping.items()
                    }
                elif recipe.get("hf", None) is not None:
                    hf_mapping = {recipe["hf"].format_map(format_dict): full_param}
                else:
                    raise NotImplementedError(
                        f"No conversion recipe found for {owner_raw_key}"
                    )
        else:
            hf_mapping = None  # Non-owner ranks will receive the converted tensors.

        # Broadcast the list of target HF parameter keys from the owner.
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if torch.distributed.get_rank() == owner_pp_global_rank:
            target_keys = [list(hf_mapping.keys())]
        else:
            target_keys = [None]  # Placeholder to be filled by broadcast.

        torch.distributed.broadcast_object_list(
            target_keys, src=owner_pp_global_rank, group=pp_group
        )
        if "None" in target_keys[0]:
            continue

        # For each converted tensor (could be more than one per original parameter), broadcast it individually.
        for target_key in target_keys[0]:
            if torch.distributed.get_rank() == owner_pp_global_rank:
                tensor_to_send = hf_mapping[target_key]
            else:
                tensor_to_send = None
            # Broadcast tensor metadata (shape and dtype) to allocate GPU buffer on receiving ranks.
            meta = [None]
            if torch.distributed.get_rank() == owner_pp_global_rank:
                meta[0] = (tensor_to_send.shape, str(tensor_to_send.dtype))
            torch.distributed.broadcast_object_list(
                meta, src=owner_pp_global_rank, group=pp_group
            )
            shape, dtype_str = meta[0]
            dtype = getattr(torch, dtype_str.split(".")[-1])
            if torch.distributed.get_rank() != owner_pp_global_rank:
                tensor_to_send = torch.empty(
                    *shape, dtype=dtype, device=torch.cuda.current_device()
                )
            torch.distributed.broadcast(
                tensor_to_send, src=owner_pp_global_rank, group=pp_group
            )
            gathered_params[target_key] = tensor_to_send

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Time taken to gather and convert params: {time.time() - st}")
    return gathered_params
