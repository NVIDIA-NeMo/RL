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

import re

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


_PACKED_MODULES = {
    "attn.fused_wqa_wkv": ["attn.wq_a", "attn.wkv"],
    "compressor.fused_wkv_wgate": ["compressor.wkv", "compressor.wgate"],
    "shared_experts.gate_up_proj": ["shared_experts.w1", "shared_experts.w3"],
}

_EXPERT_WEIGHT_RE = re.compile(
    r"(?:^|\.)ffn\.experts\.(?P<expert_id>\d+)\.w[123]\.weight$"
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


def get_exported_expert_id(model: torch.nn.Module, name: str) -> int | None:
    """Return the global expert ID encoded in a DeepSeek V4 export name."""
    if not is_model(model):
        return None
    match = _EXPERT_WEIGHT_RE.search(name)
    return int(match.group("expert_id")) if match is not None else None


def is_nonlocal_expert(module: torch.nn.Module | None, expert_id: int) -> bool:
    """Return whether a routed-expert module does not own an exported expert."""
    assert isinstance(module, RoutedExperts)
    return module._map_global_expert_id_to_local_expert_id(expert_id) == -1


def _reset_routed_experts_for_refit(model: torch.nn.Module) -> None:
    """Restore FP8 expert tensors to the checkpoint-loadable layout."""
    for layer in model.modules():
        if not isinstance(layer, RoutedExperts):
            continue
        weight_block_size = layer.weight_block_size

        raw_weight_shapes = {
            "w13": (
                int(layer.num_experts),
                2 * int(layer.intermediate_size_per_partition),
                int(layer.hidden_size),
            ),
            "w2": (
                int(layer.num_experts),
                int(layer.hidden_size),
                int(layer.intermediate_size_per_partition),
            ),
        }
        for prefix, raw_shape in raw_weight_shapes.items():
            weight = getattr(layer, f"{prefix}_weight")
            if weight.shape != raw_shape or weight.dtype != torch.float8_e4m3fn:
                weight.data = torch.empty(
                    raw_shape,
                    dtype=torch.float8_e4m3fn,
                    device=weight.device,
                )

        block_m, block_n = (int(size) for size in weight_block_size)
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            raw_scale_shape = (
                weight.shape[0],
                (weight.shape[1] + block_m - 1) // block_m,
                (weight.shape[2] + block_n - 1) // block_n,
            )
            scale = getattr(layer, f"{prefix}_weight_scale_inv")
            if scale.shape != raw_scale_shape or scale.dtype != torch.float32:
                scale.data = torch.empty(
                    raw_scale_shape,
                    dtype=torch.float32,
                    device=scale.device,
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

    for layer in model.modules():
        if not isinstance(layer, RoutedExperts):
            continue
        # Load local experts immediately instead of retaining every source
        # expert in layerwise reload until the fused parameter is complete.
        required_skip_tensors.update(layer._parameters)

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
    for layer in model.modules():
        if not isinstance(layer, RoutedExperts):
            continue
        layer.quant_method.process_weights_after_loading(layer)
        if layer.w13_weight.is_cuda:
            # Requantization uses sizeable per-expert float32 temporaries.
            torch.cuda.empty_cache()
