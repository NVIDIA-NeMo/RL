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
from collections import defaultdict
from typing import Any

import einops
import numpy as np
import torch
from megatron.core import parallel_state
from nemo.lightning.io.state import (
    StateDictTransform,
    TransformCTX,
    _match_keys,
    _ModelState,
)
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations.accelerate import init_empty_weights

import nemo_rl.models.megatron.converters.llama as llama_converter
import nemo_rl.models.megatron.converters.qwen2 as qwen2_converter
import nemo_rl.models.megatron.converters.deepseek as deepseek_converter


def get_local_layer_num(s):
    """Assumes layer number is preceeded by 'layers.'."""
    segments = s.split(".")
    number = None
    for i, segment in enumerate(segments):
        if segment == "layers":
            if segments[i + 1].isdigit():
                number = int(segments[i + 1])
                break
    return number


def get_local_expert_num(s):
    """Assumes experts have 'experts.' in their name. Expert num succeeds '.weight'."""
    segments = s.split(".")
    if "experts" not in segments or segments[-1] == "_extra_state":
        return None
    number = int(segments[-1].strip("weight"))
    return number


def get_global_layer_num(s, cfg):
    """Assumes layer number is preceeded by 'layers.'.

    Assumes pipeline model parallel size is set.
    In the state dict, the layer number is the local layer number (PP local).
    This function converts the local layer number to the global layer number.
    """
    local_layer_num = get_local_layer_num(s)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    first_stage_layers = cfg.num_layers_in_first_pipeline_stage
    last_stage_layers = cfg.num_layers_in_last_pipeline_stage

    if first_stage_layers is None and last_stage_layers is None:
        first_stage_layers = last_stage_layers = cfg.num_layers // pp_size
    elif first_stage_layers is None:
        first_stage_layers = (cfg.num_layers - last_stage_layers) // (pp_size - 1)
    elif last_stage_layers is None:
        last_stage_layers = (cfg.num_layers - first_stage_layers) // (pp_size - 1)

    # Calculate global offset based on rank
    if pp_rank == 0:
        global_offset = 0
    elif pp_rank == pp_size - 1:
        global_offset = cfg.num_layers - last_stage_layers
    else:
        middle_layers = cfg.num_layers - first_stage_layers - last_stage_layers
        layers_per_middle_stage = middle_layers // (pp_size - 2)
        global_offset = first_stage_layers + (pp_rank - 1) * layers_per_middle_stage

    return global_offset + local_layer_num


def get_global_expert_num(s, cfg):
    """Assumes experts have 'experts.' in their name. Expert num succeeds '.weight'.

    Assumes expert model parallel size is set.
    In the state dict, the expert number is the local expert number (expert local).
    This function converts the local expert number to the global expert number.
    """
    local_expert_num = get_local_expert_num(s)
    global_expert_num = (
        parallel_state.get_expert_model_parallel_rank()
        * cfg.num_moe_experts
        // parallel_state.get_expert_model_parallel_world_size()
        + local_expert_num
    )
    return global_expert_num


def get_global_key_from_local_key(local_key, model_cfg):
    local_layer = get_local_layer_num(local_key)
    if local_layer is not None:
        global_layer = get_global_layer_num(local_key, model_cfg)
        # Replace the first occurrence of the digits after "layers." with the global layer number.
        global_key = re.sub(r"(?<=layers\.)\d+", str(global_layer), local_key, count=1)
    else:
        global_key = local_key
    local_expert = get_local_expert_num(global_key)
    if local_expert is not None:
        global_expert = get_global_expert_num(global_key, model_cfg)
        # Replace the last occurrence of the digits after "weight" with the global expert number.
        global_key = re.sub(r"(?<=weight)\d+", str(global_expert), global_key)
    return global_key


def split_fc1_tp(ctx: TransformCTX, linear_fc1: torch.Tensor):
    # gate proj and up proj are mixed right now, and we need to reshape them
    # [ gate_tp0 ]     [ gate_tp0 ]
    # [  up_tp0  ] --\ [ gate_tp1 ] --\ (split gate)
    # [ gate_tp1 ] --/ [  up_tp0  ] --/ (split  up)
    # [  up_tp1  ]     [  up_tp1  ]
    megatron_config = ctx.source.config
    tp = megatron_config.tensor_model_parallel_size
    linear_fc1 = einops.rearrange(linear_fc1, "(t c d) a1 ->  c (t d) a1", c=2, t=tp)
    mlp_gate_proj_weight = linear_fc1[0]
    mlp_up_proj_weight = linear_fc1[1]
    return mlp_gate_proj_weight, mlp_up_proj_weight


def split_fc1_etp(ctx: TransformCTX, linear_fc1: torch.Tensor):
    # gate proj and up proj are mixed right now, and we need to reshape them
    # [ gate_tp0 ]     [ gate_tp0 ]
    # [  up_tp0  ] --\ [ gate_tp1 ] --\ (split gate)
    # [ gate_tp1 ] --/ [  up_tp0  ] --/ (split  up)
    # [  up_tp1  ]     [  up_tp1  ]
    megatron_config = ctx.source.config
    etp = megatron_config.expert_tensor_parallel_size
    linear_fc1 = einops.rearrange(linear_fc1, "(t c d) a1 ->  c (t d) a1", c=2, t=etp)
    mlp_gate_proj_weight = linear_fc1[0]
    mlp_up_proj_weight = linear_fc1[1]
    return mlp_gate_proj_weight, mlp_up_proj_weight


def split_qkv_gpu(ctx: TransformCTX, linear_qkv: torch.Tensor):
    """Split interleave-concatenated qkv to q, k, v.

    Example: export layer linear_qkv to HF {q|k|v}_proj
    """
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    # hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
    # when converting base model (linear_qkv), hidden size = megatron_config.hidden_size
    # when converting lora (linear_qkv.adapter.linear_out), hidden size = lora_r
    hidden_size = linear_qkv.size(-1)
    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size)
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size)
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size)

    return q_proj, k_proj, v_proj


def split_qkv_bias_gpu(ctx: TransformCTX, qkv_bias: torch.Tensor):
    """Split interleave-concatenated qkv bias to separate q, k, v bias.

    Example: export layer linear_qkv bias to HF {q|k|v}_proj bias
    """
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])
    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_bias = qkv_bias[q_slice].reshape(-1)
    k_bias = qkv_bias[k_slice].reshape(-1)
    v_bias = qkv_bias[v_slice].reshape(-1)

    return q_bias, k_bias, v_bias


def update_transforms_for_nemorl(export_transforms):
    # In place update
    for transform in export_transforms:
        if transform.transform.__name__ == "split_fc1":
            if (
                "experts" in transform.source_key
                and "shared_experts" not in transform.source_key
            ):
                transform.transform = split_fc1_etp
            else:
                transform.transform = split_fc1_tp
        elif transform.transform.__name__ == "split_qkv":
            # This transform previously moved qkv weights to cpu
            transform.transform = split_qkv_gpu
        elif transform.transform.__name__ == "split_qkv_bias":
            # This transform previously moved qkv weights to cpu
            transform.transform = split_qkv_bias_gpu
    return export_transforms


class MegatronToHFConverter:
    def __init__(self, hf_model_name, megatron_model):
        # We only care about the state_dict keys and the config, so we
        # don't need to load the model weights
        config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        with init_empty_weights():
            self.target_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True
            )

        local_keys = list(megatron_model.state_dict().keys())
        global_keys = [
            get_global_key_from_local_key(k, megatron_model.config) for k in local_keys
        ]

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_world_size = torch.distributed.get_world_size(pp_group)
        pp_gathered_global_keys = [None] * pp_world_size
        torch.distributed.all_gather_object(
            pp_gathered_global_keys, global_keys, group=pp_group
        )
        pp_gathered_global_keys = list({k for l in pp_gathered_global_keys for k in l})  # type: ignore

        ep_group = parallel_state.get_expert_model_parallel_group()
        ep_world_size = parallel_state.get_expert_model_parallel_world_size()
        ep_gathered_global_keys = [None] * ep_world_size
        torch.distributed.all_gather_object(
            ep_gathered_global_keys, pp_gathered_global_keys, group=ep_group
        )
        ep_gathered_global_keys = list({k for l in ep_gathered_global_keys for k in l})

        global_keys = ep_gathered_global_keys
        global_keys_map = {k: None for k in global_keys}

        if "qwen" in hf_model_name.lower():
            self.export_mapping = qwen2_converter.get_export_mapping(megatron_model)
            self.export_transforms = qwen2_converter.get_export_transforms(config)
            self.get_source_fn = lambda source_state_dict, _: _ModelState(
                source_state_dict
            )
        elif "llama" in hf_model_name.lower():
            self.export_mapping = llama_converter.get_export_mapping()
            self.export_transforms = llama_converter.get_export_transforms(config)
            self.get_source_fn = lambda source_state_dict, _: _ModelState(
                source_state_dict
            )
        elif (
            "deepseek" in hf_model_name.lower()
            or "Moonlight-16B-A3B" in hf_model_name
            or hf_model_name in ("ByteDance-Seed/academic-ds-9B",)
        ):
            self.export_mapping = deepseek_converter.get_export_mapping(
                source=global_keys_map,
                source_config=megatron_model.config.__dict__,
            )
            self.export_transforms = deepseek_converter.get_export_transforms()
            self.get_source_fn = deepseek_converter.get_source_fn
        else:
            raise ValueError(
                f"No converter mapping and transforms found for {hf_model_name}"
            )

        self.export_transforms = update_transforms_for_nemorl(self.export_transforms)

        updated_global_keys_map = self.get_source_fn(
            global_keys_map, megatron_model.config.__dict__
        ).state_dict()

        # Set the value of the state_dict to the megatron key name so that
        # StateDictTransform will set the value of the target state dict to
        # the megatron key name
        dummy_source = _ModelState({k: k for k in updated_global_keys_map.keys()})

        ctx = TransformCTX(
            source=dummy_source,
            source_state=dummy_source.state_dict(),
            target=self.target_model,
            target_state=self._get_empty_state_dict(),
        )
        for key, val in self.export_mapping.items():
            ctx = StateDictTransform(key, val)(ctx)

        for transform in self.export_transforms:
            if type(transform.target_key) == tuple:
                for t in transform.target_key:
                    ctx = StateDictTransform(transform.source_key, t)(ctx)
            else:
                ctx = StateDictTransform(transform.source_key, transform.target_key)(
                    ctx
                )

        hf_keys_to_megatron_keys = ctx.target_state
        megatron_keys_to_hf_keys = defaultdict(set)
        for hf_key, megatron_key in hf_keys_to_megatron_keys.items():
            if isinstance(megatron_key, list):
                for k in megatron_key:
                    megatron_keys_to_hf_keys[k].add(hf_key)
            else:
                megatron_keys_to_hf_keys[megatron_key].add(hf_key)
        self.megatron_keys_to_hf_keys = dict(megatron_keys_to_hf_keys)

    def _get_empty_state_dict(self, source_keys=None):
        if source_keys is None:
            # If source_keys is None, then we use all the target model keys
            target_keys = self.target_model.state_dict().keys()
        else:
            # Otherwise, we only use the target keys corresponding to the source_keys
            target_keys = set()
            for k in source_keys:
                target_keys = target_keys.union(self.megatron_keys_to_hf_keys[k])

        state_dict = {k: None for k in target_keys}
        return state_dict

    def _group(
        self,
        state_dict,
        key,
        item,
        main_state_dict_keys,
        main_items,
        exception_state_dict_keys_list,
        exception_items,
    ):
        source_matches = _match_keys(list(state_dict.keys()), key)
        if source_matches.size == 1 and source_matches == np.array(None):
            # no match, don't include these keys
            return
        elif source_matches.ndim == 1:
            # normal case
            main_state_dict_keys.extend(source_matches)
            main_items.append(item)
        elif source_matches.ndim == 2:
            for source_match in source_matches:
                if None in source_match:
                    # partial wildcard match case (e.g. an MoE layer with missing experts in this batch)
                    non_none_sources = [s for s in source_match if s is not None]
                    exception_state_dict_keys_list.append(non_none_sources)
                    exception_items.append(item)
                else:
                    # normal case
                    main_state_dict_keys.extend(source_match)
                    main_items.append(item)
        else:
            raise NotImplementedError(
                f"source_matches.ndim = {source_matches.ndim}. Expressions with more than 2 wildcard expressions are not supported."
            )

    def _get_groups(self, state_dict):
        """This function is used to group mappings and transforms together.

        Goes through the mappings and transforms once to collect mapping and transform groups
        [(mapping, state_dict_keys)], [(transforms, state_dict_keys)] that can be converted
        together.

        This is necessary because:
        1. If the mapping or transform expression has 2 wildcard expressions,
           _match_keys assumes the matches for each wildcard are the same size. For example,
           if the mapping is "layers.*.mlp.experts.*.linear_fc1.weight", where the first wildcard
           matches the layer number and the second wildcard matches the expert number, it assumes
           the number of experts is the same for each layer. This will fail in the case we're doing
           batched streaming refit and the current state dict is missing experts from some layers.
           To handle this, we separate out the partial keys (e.g. the ones corresponding to less experts)
           in a separate group and run them through the mapping and transforms separately.

           NOTE: this function currently only handles expressions with up to 2 wildcard expressions
           and will fail if the mapping or transform expression has more than 2 wildcard expressions.

        2. An expression matches 0 keys in the current state dict. This can happen during batched
           streaming refit if the current state dict doesn't have any keys that match the expression.
           To handle this, we skip these mapping/transforms.

        """
        # Most of the keys will be able to converted together (main)
        # For the keys that can't be converted together (exception), we need to handle them separately
        main_state_dict_keys: list[str] = []
        exception_mappings_state_dict_keys_list: list[list[str]] = []
        exception_transforms_state_dict_keys_list: list[list[str]] = []

        main_mappings: list[tuple[str, Any]] = []
        exception_mappings: list[tuple[str, Any]] = []
        for key, val in self.export_mapping.items():
            self._group(
                state_dict,
                key,
                (key, val),
                main_state_dict_keys,
                main_mappings,
                exception_mappings_state_dict_keys_list,
                exception_mappings,
            )

        main_transforms = []
        exception_transforms = []
        for transform in self.export_transforms:
            if type(transform.source_key) == tuple:
                source_keys = transform.source_key
            else:
                source_keys = (transform.source_key,)
            for source_key in source_keys:
                self._group(
                    state_dict,
                    source_key,
                    transform,
                    main_state_dict_keys,
                    main_transforms,
                    exception_transforms_state_dict_keys_list,
                    exception_transforms,
                )

        mapping_groups = [({k: v for k, v in main_mappings}, main_state_dict_keys)]
        for (k, v), exception_state_dict_keys in zip(
            exception_mappings, exception_mappings_state_dict_keys_list
        ):
            mapping_groups.append(({k: v}, exception_state_dict_keys))
        transform_groups = [(main_transforms, main_state_dict_keys)]
        for exception_transform, exception_state_dict_keys in zip(
            exception_transforms, exception_transforms_state_dict_keys_list
        ):
            transform_groups.append(([exception_transform], exception_state_dict_keys))

        return mapping_groups, transform_groups

    def convert(self, state_dict, megatron_config):
        state_dict = self.get_source_fn(
            state_dict, megatron_config.__dict__
        ).state_dict()

        mapping_groups, transform_groups = self._get_groups(state_dict)

        converted_state_dict = {}
        for mapping, state_dict_keys in mapping_groups:
            source = _ModelState({k: state_dict[k] for k in state_dict_keys})
            source.config = megatron_config
            ctx = TransformCTX(
                source=source,
                source_state=source.state_dict(),
                target=self.target_model,
                target_state=self._get_empty_state_dict(list(state_dict_keys)),
            )

            for key, val in mapping.items():
                ctx = StateDictTransform(key, val)(ctx)

            for k, v in ctx.target_state.items():
                if v is not None:
                    converted_state_dict[k] = v

        for transforms, state_dict_keys in transform_groups:
            source = _ModelState({k: state_dict[k] for k in state_dict_keys})
            source.config = megatron_config
            ctx = TransformCTX(
                source=source,
                source_state=source.state_dict(),
                target=self.target_model,
                target_state=self._get_empty_state_dict(list(state_dict_keys)),
            )
            for transform in transforms:
                ctx = transform(ctx)

            for k, v in ctx.target_state.items():
                if v is not None:
                    converted_state_dict[k] = v

        return converted_state_dict
