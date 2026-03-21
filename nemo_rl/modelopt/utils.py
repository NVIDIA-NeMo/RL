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

"""Lightweight quantization config resolver usable by both Megatron and vLLM workers."""

import importlib
import importlib.util
import os

import modelopt.torch.quantization as mtq


def resolve_quant_cfg(quant_cfg):
    """Resolves a quantization config from a built-in name or a Python file path.

    Resolution order:
    1. Built-in ModelOpt config name (e.g. "NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG")
    2. File path with variable name: "/path/to/config.py:VAR_NAME"

    For custom configs, the target variable must be a dict with the ModelOpt
    quantization config format. See examples/modelopt/quant_configs/ for examples.
    """
    file_path, sep, attr_name = quant_cfg.rpartition(":")
    if sep and file_path.endswith(".py"):
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise ValueError(f"quant_cfg file not found: '{file_path}'")
        spec = importlib.util.spec_from_file_location("_custom_quant_cfg", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = getattr(module, attr_name, None)
        if cfg is None:
            raise ValueError(f"File '{file_path}' has no attribute '{attr_name}'.")
        if not isinstance(cfg, dict):
            raise ValueError(
                f"quant_cfg '{attr_name}' in '{file_path}' is {type(cfg).__name__}, expected a dict."
            )
        return cfg

    builtin = getattr(mtq, quant_cfg, None)
    if builtin is not None:
        return builtin

    raise ValueError(
        f"Unknown quant_cfg '{quant_cfg}'. Must be a built-in ModelOpt config name "
        f"(e.g. 'NVFP4_DEFAULT_CFG') or a file path with variable name "
        f"(e.g. '/path/to/config.py:MY_CONFIG')."
    )
