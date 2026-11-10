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
from typing import Any, Dict

# Side-effect import: installs the resolver hook that routes FP8-native Mistral 3.5
# configs to Mistral3FP8VLM. Without it, HF's stock FP8Linear path runs and produces
# 0-d weight_scale_inv params that FSDP2 rejects.
import nemo_automodel.components.models.mistral3_vlm  # noqa: F401
import torch
from nemo_automodel._transformers.auto_model import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    NeMoAutoModelForTextToWaveform,
)
from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
)
from torch import nn

# Add an entry whenever a model's architecture isn't loadable via
# NeMoAutoModelForCausalLM (e.g. VLMs using ForConditionalGeneration /
# ForImageTextToText). Check MODEL_ARCH_MAPPING in the NeMo automodel registry to see
# which architectures have custom impls:
# https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/registry.py#L32-L146
AUTOMODEL_FACTORY: Dict[str, Any] = {
    "qwen2_5_vl": NeMoAutoModelForImageTextToText,
    "qwen2_vl": NeMoAutoModelForImageTextToText,
    "qwen2_5_omni": NeMoAutoModelForTextToWaveform,
    "qwen3_5": NeMoAutoModelForImageTextToText,
    "llava": NeMoAutoModelForImageTextToText,
    "internvl": NeMoAutoModelForImageTextToText,
    "gemma3": NeMoAutoModelForImageTextToText,
    "gemma4": NeMoAutoModelForImageTextToText,
    "gemma4_unified": NeMoAutoModelForImageTextToText,
    "smolvlm": NeMoAutoModelForImageTextToText,
    "mistral3": NeMoAutoModelForImageTextToText,
    "llama4": NeMoAutoModelForImageTextToText,
}


def move_model_buffers_to_device(
    model: nn.Module, device: str | torch.device
) -> nn.Module:
    """Move registered buffers while preserving their tensor identity and aliases."""
    for buffer in model.buffers():
        # Swapping also updates device metadata; .data assignment can leave stale storage.
        torch.utils.swap_tensors(buffer, buffer.to(device))
    return model


def initialize_mamba_kernels(model: nn.Module) -> bool:
    """Initialize native Mamba extensions and linear kernels before FSDP overlap."""
    # Keep Mamba's optional CUDA dependencies out of unrelated model paths.
    from nemo_automodel.components.models.nemotron_v3.layers import (
        NemotronV3Mamba2Mixer,
    )

    if not any(isinstance(module, NemotronV3Mamba2Mixer) for module in model.modules()):
        return False
    # CUDA registration can wait for work that another rank has not enqueued yet.
    import mamba_ssm.ops.triton.ssd_combined  # noqa: F401

    device_index = torch.cuda.current_device()
    # The first cuBLAS handle can synchronize outstanding asynchronous collectives.
    with torch.random.fork_rng(devices=[device_index]), torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            inputs = torch.zeros((16, 16), device=device_index, dtype=dtype)
            weight = torch.zeros_like(inputs)
            bias = torch.zeros(16, device=device_index, dtype=dtype)
            nn.functional.linear(inputs, weight)
            nn.functional.linear(inputs, weight, bias)
        torch.cuda.synchronize(device_index)
    return True


def warmup_compiled_rms_norm(model: nn.Module) -> int:
    """Initialize native compiled norms without changing model state or RNG."""
    from nemo_automodel.components.models.common.utils import Float32RMSNorm

    variants = list(
        dict.fromkeys(
            (module.weight.shape[0], module.eps, module.weight.dtype)
            for module in model.modules()
            if isinstance(module, Float32RMSNorm)
        )
    )
    if not variants:
        return 0
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    # Lazy compiler initialization allocates CUDA constants and can synchronize
    # against outstanding FSDP collectives when ranks reach their first norm apart.
    with torch.random.fork_rng(devices=[device_index]):
        for width, eps, dtype in variants:
            norm = Float32RMSNorm(width, eps=eps, device=device, dtype=dtype)
            inputs = torch.zeros((1, 2, width), device=device, dtype=dtype)
            with torch.no_grad():
                norm(inputs)
            with torch.enable_grad():
                norm(inputs.requires_grad_()).sum().backward()
            torch.cuda.synchronize(device)
    return len(variants)


def resolve_model_class(model_name: str) -> Any:
    """Resolve the model class for a model type.

    Args:
        model_name: Model type to resolve.
    """
    return AUTOMODEL_FACTORY.get(model_name.lower(), NeMoAutoModelForCausalLM)


# Per-dispatch tensors that Automodel's flex token-dispatch managers (DeepEP,
# HybridEP, UCCL-EP) keep as attributes. Each dispatch reassigns them, so none
# is needed across steps.
# TODO(https://github.com/NVIDIA-NeMo/Automodel/issues/4401): drop this workaround
# once Automodel releases this state itself.
_MOE_DISPATCH_STATE_ATTRS = (
    "token_probs",
    "token_indices",
    "routing_map",
    "handle",
    "tokens_per_expert",
    "num_permuted_tokens",
    "dispatched_indices",
    "dispatched_probs",
    "dispatched_routing_map",
    "reversed_mapping_for_combine",
)


def release_moe_dispatcher_state(model: nn.Module) -> int:
    """Drop the per-dispatch tensors cached on the model's MoE dispatch managers.

    The managers keep the last dispatch's routing tensors as attributes;
    ``combine`` clears only ``handle`` (and HybridEP's ``num_permuted_tokens``).
    After a backward with activation checkpointing, a manager's last dispatch
    is a recompute, so ``token_probs`` and ``dispatched_probs`` carry a
    ``grad_fn`` that keeps that recompute graph and its saved activations
    alive. With the default manager shared by all layers that is the first MoE
    layer (about 1 GB for Qwen3.5-35B-A3B at 9216 tokens per micro-batch); with
    per-layer managers every layer keeps one. The live blocks pin cached
    allocator segments, so ``torch.cuda.empty_cache()`` cannot return them
    before colocated generation wakes up.

    Args:
        model: Model whose ``token_dispatcher`` attributes are scanned. A
            manager shared across layers is cleared once.

    Returns:
        The number of attributes set to ``None``.
    """
    managers = {}
    for module in model.modules():
        # MoEFlexTokenDispatcher is a plain object held by the experts module,
        # not a submodule, so model.modules() does not yield it.
        dispatcher = getattr(module, "token_dispatcher", None)
        if isinstance(dispatcher, MoEFlexTokenDispatcher):
            manager = getattr(dispatcher, "_comm_manager", None)
            if manager is not None:
                managers[id(manager)] = manager

    cleared = 0
    for manager in managers.values():
        for attr in _MOE_DISPATCH_STATE_ATTRS:
            if getattr(manager, attr, None) is not None:
                setattr(manager, attr, None)
                cleared += 1
    return cleared
