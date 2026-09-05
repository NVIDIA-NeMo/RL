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
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

_PACKED_MODULES = {
    "attn.fused_wqa_wkv": ["attn.wq_a", "attn.wkv"],
    "compressor.fused_wkv_wgate": ["compressor.wkv", "compressor.wgate"],
    "shared_experts.gate_up_proj": ["shared_experts.w1", "shared_experts.w3"],
}

_EXPERT_REFIT_PARAMS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)


def is_model(model: torch.nn.Module) -> bool:
    """Return whether a constructed vLLM model is a DeepSeek V4 causal LM."""
    return model.config.model_type == "deepseek_v4"


def map_checkpoint_name(model: torch.nn.Module, name: str) -> str:
    """Apply DeepSeek V4's regex and suffix-aware HF-to-vLLM mapping."""
    if not is_model(model):
        return name
    return model.hf_to_vllm_mapper.apply_list([name])[0]


def remap_packed_module_path(
    model: torch.nn.Module, module_path: list[str]
) -> list[str]:
    """Map separately exported DeepSeek V4 weights to fused vLLM modules."""
    if not is_model(model):
        return module_path

    for fused_name, original_names in _PACKED_MODULES.items():
        for original_name in original_names:
            original_path = original_name.split(".")
            if module_path[-len(original_path) :] == original_path:
                module_path[-len(original_path) :] = fused_name.split(".")
                break
    return module_path


def _block_fp8_routed_experts(model: torch.nn.Module):
    """Yield routed-expert modules using checkpoint-style block FP8 weights."""
    for layer in model.modules():
        quant_method = getattr(layer, "quant_method", None)
        if (
            isinstance(layer, RoutedExperts)
            and isinstance(quant_method, Fp8MoEMethod)
            and quant_method.block_quant
        ):
            yield layer


def _reset_routed_experts_for_refit(model: torch.nn.Module) -> None:
    """Restore FP8 expert tensors to the checkpoint-loadable layout."""
    for layer in _block_fp8_routed_experts(model):
        restore_params, _ = get_layerwise_info(layer).restore_metadata
        missing = set(_EXPERT_REFIT_PARAMS) - set(restore_params)
        if missing:
            raise RuntimeError(
                "Missing checkpoint-layout reload metadata for DeepSeek V4 FP8 "
                f"expert tensors: {sorted(missing)}"
            )

        for name in _EXPERT_REFIT_PARAMS:
            param = getattr(layer, name)
            metadata = restore_params[name]
            if (
                param.shape != metadata.shape
                or param.stride() != metadata.stride()
                or param.dtype != metadata.dtype
            ):
                param.data = torch.empty_strided(
                    metadata.shape,
                    metadata.stride(),
                    dtype=metadata.dtype,
                    device=param.device,
                )


def prepare_refit(model: torch.nn.Module) -> set[str]:
    """Prepare DeepSeek V4 parameters for layerwise FP8 refit.

    Returns the process-global skip names added by this invocation. The caller
    must pass them to :func:`restore_refit` after the layerwise lifecycle,
    including on failure.
    """
    from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS

    # DeepSeek V4 loads attn_sink with a direct copy_ instead of its Parameter
    # weight_loader. Keep its kernel storage materialized during reload.
    required_skip_tensors = {"attn_sink"}

    for layer in _block_fp8_routed_experts(model):
        # Load local experts immediately instead of retaining every source
        # expert in layerwise reload until the fused parameter is complete.
        required_skip_tensors.update(_EXPERT_REFIT_PARAMS)

    added_skip_tensors = required_skip_tensors - SKIP_TENSORS
    SKIP_TENSORS.update(required_skip_tensors)
    try:
        _reset_routed_experts_for_refit(model)
    except Exception:
        SKIP_TENSORS.difference_update(added_skip_tensors)
        raise
    return added_skip_tensors


def restore_refit(added_skip_tensors: set[str]) -> None:
    """Remove process-global layerwise skip names added for one refit."""
    from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS

    SKIP_TENSORS.difference_update(added_skip_tensors)


@torch.no_grad()
def finalize_refit(model: torch.nn.Module) -> None:
    """Convert immediately loaded expert tensors back to their kernel layout."""
    for layer in _block_fp8_routed_experts(model):
        layer.quant_method.process_weights_after_loading(layer)
        if layer.w13_weight.is_cuda:
            # Requantization uses sizeable per-expert float32 temporaries.
            torch.cuda.empty_cache()
