# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from megatron.core.inference.utils import device_memory_summary


def resolve_torch_dtype(val):
    """Convert a value to `torch.dtype`."""
    if isinstance(val, torch.dtype):
        return val
    if isinstance(val, str):
        name = val.replace("torch.", "")
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    raise ValueError(
        f"Cannot resolve torch dtype from {val!r} (type {type(val).__name__}). "
        f"Expected a torch.dtype or a string like 'torch.float32' / 'float32'."
    )


def get_lang_module(model):
    """Return the underlying language module, unwrapping precision wrappers."""
    return model.module.module if hasattr(model.module, "module") else model.module


def log_gpu_memory(tag: str) -> None:
    """Print a one-line GPU-memory summary for the calling rank."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"[GPU Rank {rank}] {tag} | {device_memory_summary()}")
