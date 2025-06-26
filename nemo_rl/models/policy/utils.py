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

import importlib
import os
from typing import Any

import torch
from transformers import AutoConfig


def import_class_from_path(name: str) -> Any:
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


def get_gpu_info(model: torch.nn.Module) -> dict[str, Any]:
    """Return information about the GPU being used by this worker."""
    import torch

    # Get distributed training info
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get device info from CUDA
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # in MB

    # Try to get the real global device ID (not the local one)
    # In distributed training, each process only sees its assigned GPU as device 0
    local_device_id = device
    global_device_id = local_device_id

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if local_rank < len(cuda_visible_devices):
            global_device_id = int(cuda_visible_devices[local_rank])

    # Get a parameter from the model to verify CUDA device placement
    # This confirms tensors are actually on the appropriate device
    param_info = {}
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None and param.requires_grad:
                full_name = f"{module_name}.{param_name}"
                param_info[full_name] = {
                    "device": str(param.device),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                }
                # Just grab one parameter for verification
                break
        if param_info:
            break

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_device_id": local_device_id,
        "global_device_id": global_device_id,
        "device_count": device_count,
        "device_name": device_name,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "peak_memory_allocated_mb": peak_memory,
        "peak_memory_reserved_mb": peak_reserved,
        "parameter_sample": param_info,
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
        },
    }


def sliding_window_overwrite(model_name: str) -> dict[str, Any]:
    """Returns configuration overrides to handle sliding window settings based on model rules.

    Args:
        model_name: The HuggingFace model name or path to load configuration from

    Returns:
        dict: Dictionary with overwrite values, or empty dict if no overwrites needed
    """
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    overwrite_dict = {}

    # Override sliding_window setting to address a HF mismatch relevant to use_sliding_window
    # TODO(@zhiyul): remove this once the bug is fixed https://github.com/huggingface/transformers/issues/38002
    if (
        hasattr(hf_config, "use_sliding_window")
        and hf_config.use_sliding_window == False
    ):
        assert hasattr(hf_config, "sliding_window")
        overwrite_dict = {
            "sliding_window": None,
        }
        print(
            f"use_sliding_window=False in config - overriding sliding_window parameter to None: {overwrite_dict}"
        )

    return overwrite_dict
