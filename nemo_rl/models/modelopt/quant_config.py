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


import copy

import modelopt.torch.quantization as mtq

nano3_config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
# disable all attention layers
nano3_config["quant_cfg"]["*.[q|k|v|o]_proj.*"] = {"enable": False}
# vllm
nano3_config["quant_cfg"]["*.qkv_proj.*"] = {"enable": False}
# megatron
nano3_config["quant_cfg"]["*.linear_proj.*"] = {"enable": False}
nano3_config["quant_cfg"]["*.linear_qkv.*"] = {"enable": False}
# disable all preceding layers of attention layers
bf16_layers = [4, 11, 18, 25, 32, 41]
for i in bf16_layers:
    attention_preceding_layer_spec = "*.layers." + str(i) + ".*"
    nano3_config["quant_cfg"][attention_preceding_layer_spec] = {"enable": False}
    # print_rank_0(f"The layer {i} with {hybrid_model_config[i]} that precedes a self-attention layer {hybrid_model_config[i+1]} is kept unquantized")

nano3_config["quant_cfg"]["*mixer.conv1d*"] = {
    "enable": False
}  # quantize only linear layers within mamba

# This is an example to customize the quantization config.
# Modify your custom config for debugging or research purposes.
CUSTOM_CONFIG = {
    "MY_QUANT_CONFIG": {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": 4,
                "block_sizes": {-1: 128},
                "enable": True,
            },
            "*input_quantizer": {
                "num_bits": 8,
                "type": "dynamic",
                "block_sizes": {-1: None},
            },
            # Disable sensitive layers such as \`lm_head\`, gate layers in MoE etc.
            **mtq.config._default_disabled_quantizer_cfg,
        },
        "algorithm": "max",
    },
    "NANO3_NVFP4_CFG": nano3_config,
}
