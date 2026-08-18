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

import warnings
from contextlib import nullcontext
from typing import ContextManager

try:
    from torch_memory_saver import (  # pyrefly: ignore[import-error]
        torch_memory_saver,
    )

    torch_memory_saver.hook_mode = "torch"

    HAVE_TORCH_MEMORY_SAVER = True
except ImportError:
    HAVE_TORCH_MEMORY_SAVER = False

# torch_memory_saver region tag for the colocated inference model's weights.
_INFERENCE_MODEL_OFFLOAD_TAG = "nemo_rl_megatron_inference_model"


def inference_model_alloc_region() -> ContextManager[None]:
    """Allocation region to build the colocated inference model under.

    Returns a CPU-backup-enabled torch_memory_saver region, or a null context.
    """
    if HAVE_TORCH_MEMORY_SAVER:
        return torch_memory_saver.region(
            tag=_INFERENCE_MODEL_OFFLOAD_TAG, enable_cpu_backup=True
        )
    warnings.warn(
        "torch_memory_saver is unavailable; the colocated inference model will stay "
        "GPU-resident alongside the training model (higher peak memory). Install "
        "torch_memory_saver to enable inference-weight offload.",
        stacklevel=2,
    )
    return nullcontext()


def pause_inference_weights() -> None:
    """Back the colocated inference weights to CPU (no-op without torch_memory_saver)."""
    if HAVE_TORCH_MEMORY_SAVER:
        torch_memory_saver.pause(_INFERENCE_MODEL_OFFLOAD_TAG)


def resume_inference_weights() -> None:
    """Restore the inference weights to their GPU addresses (no-op without torch_memory_saver)."""
    if HAVE_TORCH_MEMORY_SAVER:
        torch_memory_saver.resume(_INFERENCE_MODEL_OFFLOAD_TAG)
