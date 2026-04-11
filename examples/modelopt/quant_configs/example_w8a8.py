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

"""Example custom quantization config for QARL.

This demonstrates how to define a custom quantization config that can be
referenced from a YAML config file by providing a file path and variable name:

    policy:
      quant_cfg: "examples/modelopt/quant_configs/example_w8a8.py:W8A8_CUSTOM_CFG"

This example defines a W8A8 config (FP8 weights, FP8 activations) using
per-tensor quantization. Sensitive layers (lm_head, MoE gate layers, etc.)
are disabled by default.

See https://github.com/NVIDIA/TensorRT-Model-Optimizer for more quantization
config examples and supported options.
"""

import modelopt.torch.quantization as mtq

W8A8_CUSTOM_CFG = {
    "quant_cfg": [
        *mtq.config._base_disable_all,
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
        },
        *mtq.config._default_disabled_quantizer_cfg,
    ],
    "algorithm": "max",
}
