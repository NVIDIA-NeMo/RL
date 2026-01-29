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


import os
import re
from collections import defaultdict

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.transformer.moe.router import TopKRouter
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from modelopt.torch.utils.plugins import megatron_prefill
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from nemo_rl.models.modelopt.quant_config import CUSTOM_CONFIG

MAX_SEQ_LEN = 2048
MAX_OUTPUT_LEN = 512

# nano3_config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
# # disable all attention layers
# nano3_config['quant_cfg']["*.[q|k|v|o]_proj.*"] = {"enable": False}
# # vllm
# nano3_config['quant_cfg']["*.qkv_proj.*"] = {"enable": False}
# # disable all preceding layers of attention layers
# bf16_layers = [4, 11, 18, 25, 32, 41]
# for i in bf16_layers:
#     attention_preceding_layer_spec = "*.layers." + str(i) +".*"
#     nano3_config["quant_cfg"][attention_preceding_layer_spec] = {"enable": False}
#     # print_rank_0(f"The layer {i} with {hybrid_model_config[i]} that precedes a self-attention layer {hybrid_model_config[i+1]} is kept unquantized")

# nano3_config["quant_cfg"]["*mixer.conv1d*"] = {"enable": False} # quantize only linear layers within mamba

# # This is an example to customize the quantization config.
# # Modify your custom config for debugging or research purposes.
# CUSTOM_CONFIG = {
#     "MY_QUANT_CONFIG": {
#         "quant_cfg": {
#             "*weight_quantizer": {
#                 "num_bits": 4,
#                 "block_sizes": {-1: 128},
#                 "enable": True,
#             },
#             "*input_quantizer": {
#                 "num_bits": 8,
#                 "type": "dynamic",
#                 "block_sizes": {-1: None},
#             },
#             # Disable sensitive layers such as \`lm_head\`, gate layers in MoE etc.
#             **mtq.config._default_disabled_quantizer_cfg,
#         },
#         "algorithm": "max",
#     },
#     "NANO3_NVFP4_CFG": nano3_config
# }


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, trust_remote_code=False):
    """Returns the tokenizer from the model ckpt_path."""
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if type(tokenizer).__name__ == "QWenTokenizer":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def merge_qkv_gate_up_proj_amaxs(model):
    """Merge amax values for QKV and Gate/Up projections in the model.

    This function merges:
    - q_proj, k_proj, v_proj amax values by taking the max of the 3
    - gate_proj, up_proj amax values by taking the max of the 2

    Args:
        model: The model containing quantized layers with weight_quantizer modules
    """
    # Group quantizers by their base pattern (layer + projection type)
    qkv_groups = defaultdict(list)  # base_pattern -> [(proj_type, quantizer)]
    gate_up_groups = defaultdict(list)  # base_pattern -> [(proj_type, quantizer)]
    qkv_input_groups = defaultdict(list)  # base_pattern -> [(proj_type, quantizer)]
    gate_up_input_groups = defaultdict(list)  # base_pattern -> [(proj_type, quantizer)]

    num_layers = 0

    for name, module in model.named_modules():
        if "layers" in name and name.endswith("q_proj"):
            num_layers += 1

        if hasattr(module, "weight_quantizer") and module.weight_quantizer.is_enabled:
            # Check if this is a q/k/v projection
            qkv_match = re.search(r"(.*\.)([qkv])_proj$", name)
            if qkv_match:
                base_pattern = qkv_match.group(1)  # e.g., "model.layers.0.self_attn."
                proj_type = qkv_match.group(2)  # q, k, or v
                qkv_groups[base_pattern].append(
                    (proj_type + "_weight", module.weight_quantizer)
                )
                if (
                    hasattr(module, "input_quantizer")
                    and module.input_quantizer.is_enabled
                ):
                    qkv_input_groups[base_pattern].append(
                        (proj_type + "_input", module.input_quantizer)
                    )
                continue

            # Check if this is a gate/up projection
            gate_up_match = re.search(r"(.*\.)(gate|up)_proj$", name)
            if gate_up_match:
                base_pattern = gate_up_match.group(1)  # e.g., "model.layers.0.mlp."
                proj_type = gate_up_match.group(2)  # gate or up
                gate_up_groups[base_pattern].append(
                    (proj_type + "_weight", module.weight_quantizer)
                )
                if (
                    hasattr(module, "input_quantizer")
                    and module.input_quantizer.is_enabled
                ):
                    gate_up_input_groups[base_pattern].append(
                        (proj_type + "_input", module.input_quantizer)
                    )

    print(f"Found {num_layers} layers")
    print(
        f"Found {len(qkv_groups)} qkv groups, {len(qkv_input_groups)} qkv input groups, found {len(gate_up_groups)} gate/up groups, {len(gate_up_input_groups)} gate/up input groups"
    )
    assert (
        num_layers
        == len(qkv_groups)
        == len(qkv_input_groups)
        == len(gate_up_groups)
        == len(gate_up_input_groups)
    )

    # Merge QKV amax values (max of q, k, v)
    for qkv_group in (qkv_groups, qkv_input_groups):
        for base_pattern, quantizers in qkv_group.items():
            # if len(quantizers) == 3:  # Should have q, k, v
            # Extract amax values
            assert len(quantizers) == 3, (
                f"Expected 3 quantizers for {base_pattern}, got {len(quantizers)}"
            )
            amax_values = []
            for proj_type, quantizer in quantizers:
                if hasattr(quantizer, "amax") and quantizer.amax is not None:
                    amax_values.append(quantizer.amax)

            # if len(amax_values) == 3:
            assert len(amax_values) == 3, (
                f"Expected 3 amax values for {base_pattern}, got {len(amax_values)}"
            )
            # Take the maximum across all three amax values
            merged_amax = torch.stack(amax_values).max(dim=0)[0]

            # Update all three quantizers with the merged amax
            for proj_type, quantizer in quantizers:
                # if hasattr(quantizer, "amax") and quantizer.amax is not None:
                quantizer.amax.copy_(merged_amax)

            print(f"Merged QKV amax for {base_pattern} (q,k,v)")

    # Merge Gate/Up amax values (max of gate, up)
    for gate_up_group in (gate_up_groups, gate_up_input_groups):
        for base_pattern, quantizers in gate_up_group.items():
            # if len(quantizers) == 2:  # Should have gate, up
            assert len(quantizers) == 2, (
                f"Expected 2 quantizers for {base_pattern}, got {len(quantizers)}"
            )
            # Extract amax values
            amax_values = []
            for proj_type, quantizer in quantizers:
                if hasattr(quantizer, "amax") and quantizer.amax is not None:
                    amax_values.append(quantizer.amax)

            # if len(amax_values) == 2:
            assert len(amax_values) == 2, (
                f"Expected 2 amax values for {base_pattern}, got {len(amax_values)}"
            )
            # Take the maximum across both amax values
            merged_amax = torch.stack(amax_values).max(dim=0)[0]

            # Update both quantizers with the merged amax
            for proj_type, quantizer in quantizers:
                # if hasattr(quantizer, "amax") and quantizer.amax is not None:
                quantizer.amax.copy_(merged_amax)

            print(f"Merged Gate/Up amax for {base_pattern} (gate,up)")


class _DictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))


def get_forward_loop_func(
    is_megatron: bool,
    calib_dataloader: DataLoader,
    force_all_expert_routing: bool = True,
):
    """Gets the forward loop function for the model."""
    if not is_megatron:
        return create_forward_loop(dataloader=calib_dataloader)

    def _forward_loop(model):
        original_topk_values = {}
        original_config_topk = None

        if force_all_expert_routing:
            for name, module in model.named_modules():
                if isinstance(module, TopKRouter):
                    # Store original values
                    original_topk_values[name] = module.topk
                    if original_config_topk is None:
                        original_config_topk = module.config.moe_router_topk

                    # Set router topk to route to all experts
                    module.topk = module.num_experts
                    # IMPORTANT: Also update config.moe_router_topk so the token dispatcher
                    # computes the correct num_out_tokens for all_to_all communication.
                    # Without this, the token dispatcher uses the original topk value
                    # which causes "Split sizes doesn't match total dim 0 size" error.
                    module.config.moe_router_topk = module.num_experts

        for batch in calib_dataloader:
            megatron_prefill(model, batch["input_ids"], skip_return_logits=True)

        if force_all_expert_routing:
            for name, module in model.named_modules():
                if isinstance(module, TopKRouter):
                    # Restore original values
                    module.topk = original_topk_values.get(name, module.topk)
                    if original_config_topk is not None:
                        module.config.moe_router_topk = original_config_topk

    return _forward_loop


def quantize_model(
    model: nn.Module,
    quant_cfg: str,
    tokenizer,
    calib_size,
    is_megatron: bool = False,
    batch_size=32,
    data="cnn_dailymail",
    force_all_expert_routing: bool = True,
    max_sample_length=1024,
):
    """Quantizes the model with the provided calibration dataset.

    Args:
        model: the model to be quantized.
        quant_cfg: the quantization algorithm config name if simple quantization is used.
                   the list of quantization algorithm config names if auto quantization is used.
        tokenizer: the tokenizer.
        batch_size: the calibration batch size for each calibration inference run.
        calib_size: the total calibration dataset size.
        auto_quantize_bits: The effective bits constraint for auto_quantize.
        data: the name of the calibration dataset.
    """
    device = (
        model.device if hasattr(model, "device") else next(model.parameters()).device
    )
    if data == "random":
        calib_size = 1
        calib_dataloader = DataLoader(
            _DictDataset({"input_ids": torch.randint(0, 100, (1, 5))}), batch_size=1
        )
    else:
        calib_dataloader = get_dataset_dataloader(
            dataset_name=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=calib_size,
            device=device,
            include_labels=False,
            max_sample_length=max_sample_length,
        )

    mtq_cfg = CUSTOM_CONFIG.get(quant_cfg)  # type: ignore [arg-type]
    if mtq_cfg is None:
        mtq_cfg = getattr(mtq, quant_cfg)  # type: ignore [arg-type]

    forward_loop = None
    use_calibration = need_calibration(mtq_cfg)
    if not use_calibration:
        print("Dynamic quantization. Calibration skipped.")
    else:
        forward_loop = get_forward_loop_func(
            is_megatron, calib_dataloader, force_all_expert_routing
        )

    model = mtq.quantize(model, mtq_cfg, forward_loop)
    if not is_megatron:
        merge_qkv_gate_up_proj_amaxs(model)
    mtq.print_quant_summary(model)


def get_modelopt_checkpoint_dir() -> str:
    """Gets the default modelopt checkpoint directory.

    1. Use NRL_MODELOPT_CHECKPOINT_DIR environment variable if set.
    2. Use HF_HOME/nemo_rl if HF_HOME is set.
    3. Use ~/.cache/huggingface/nemo_rl if neither are set.
    """
    nrl_modelopt_checkpoint_dir = os.environ.get("NRL_MODELOPT_CHECKPOINT_DIR")
    if nrl_modelopt_checkpoint_dir is not None and nrl_modelopt_checkpoint_dir.strip():
        modelopt_checkpoint_dir = nrl_modelopt_checkpoint_dir
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is not None and hf_home.strip():
            modelopt_checkpoint_dir = os.path.join(hf_home, "nemo_rl")
        else:
            modelopt_checkpoint_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "nemo_rl"
            )
    print(f"Using default modelopt checkpoint dir: {modelopt_checkpoint_dir}")
    return modelopt_checkpoint_dir


def quantization_layer_spec(config):
    """Layer specification for quantization with ModelOpt.

    We need to disable arbitrary attention mask for sequence packing.
    """
    disable_modelopt_layer_spec = int(
        os.environ.get("DISABLE_MODELOPT_LAYER_SPEC", "0")
    )
    if disable_modelopt_layer_spec:
        return transformer_engine_layer_spec(config)
    return get_gpt_modelopt_spec(
        config=config,
        local_core_attention=False,
        remap_te_layernorm=True,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=False,
    )
