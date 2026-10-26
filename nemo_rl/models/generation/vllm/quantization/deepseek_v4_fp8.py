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

from dataclasses import dataclass, field
from typing import cast

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.models.utils import WeightsMapper

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

_MODEL_POST_LOAD_HOOK = "process_weights_after_loading"


def _skip_model_post_load() -> None:
    """Stand-in for the model-level post-load hook while a refit streams."""


def suspend_model_post_load(model: torch.nn.Module) -> None:
    """Shadow the model-level post-load hook until :func:`finalize_refit` runs it.

    Since vLLM 0.29, ``DeepseekV4ForCausalLM.load_weights`` ends with
    ``self.process_weights_after_loading()``, which recomputes the first
    layer's hyper-connection broadcast from ``hc_attn_fn``. NeMo-RL streams a
    refit through buffer-sized ``load_weights`` calls while layerwise reload
    still holds that parameter on the meta device, so the hook fails with
    ``Cannot copy out of meta tensor``. An instance attribute shadows the class
    method until every layer is materialized.
    """
    if not callable(getattr(type(model), _MODEL_POST_LOAD_HOOK, None)):
        return
    if _MODEL_POST_LOAD_HOOK not in vars(model):
        setattr(model, _MODEL_POST_LOAD_HOOK, _skip_model_post_load)


def resume_model_post_load(model: torch.nn.Module) -> None:
    """Drop the shadow installed by :func:`suspend_model_post_load`."""
    if vars(model).get(_MODEL_POST_LOAD_HOOK) is _skip_model_post_load:
        delattr(model, _MODEL_POST_LOAD_HOOK)


def is_model(model: torch.nn.Module) -> bool:
    """Return whether a constructed vLLM model is a DeepSeek V4 causal LM."""
    config = getattr(model, "config", None)
    return getattr(config, "model_type", None) == "deepseek_v4"


def map_checkpoint_name(model: torch.nn.Module, name: str) -> str:
    """Apply DeepSeek V4's regex and suffix-aware HF-to-vLLM mapping."""
    if not is_model(model):
        return name
    mapper = cast(WeightsMapper, model.hf_to_vllm_mapper)
    return mapper.apply_list([name])[0]


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


@dataclass
class SkipNames:
    """Skip-list names one :func:`prepare_refit` call added, per vLLM set."""

    tensors: set[str] = field(default_factory=set)
    """Names added to ``SKIP_TENSORS`` (kept off the meta device)."""
    load: set[str] = field(default_factory=set)
    """Names added to ``SKIP_LOAD_TENSORS`` (kept out of load accounting)."""


def _layerwise_skip_sets() -> list[set[str]]:
    """The process-global skip sets an immediately loaded tensor must be in.

    vLLM 0.25 had one set, ``SKIP_TENSORS``: a listed tensor stays off the meta
    device *and* its weight loader is neither wrapped nor counted, so the
    layer is never processed by the layerwise machinery. vLLM 0.29 split that
    into ``SKIP_LOAD_TENSORS`` (never loaded through ``weight_loader``: not
    wrapped, not counted) and ``SKIP_TENSORS`` (additionally never moved to
    meta), and only ``SKIP_LOAD_TENSORS`` keeps a tensor out of the layer's
    load accounting. The expert tensors below are loaded immediately through
    their original loaders and converted once in :func:`finalize_refit`; if
    they were counted, vLLM would run ``process_weights_after_loading`` on the
    layer as soon as its last expert arrived and :func:`finalize_refit` would
    convert the already kernel-layout tensors a second time.
    """
    from vllm.model_executor.model_loader.reload import meta

    sets = [meta.SKIP_TENSORS]
    skip_load = getattr(meta, "SKIP_LOAD_TENSORS", None)
    if isinstance(skip_load, set) and skip_load is not meta.SKIP_TENSORS:
        sets.append(skip_load)
    return sets


def prepare_refit(model: torch.nn.Module) -> SkipNames:
    """Prepare DeepSeek V4 parameters for layerwise FP8 refit.

    Returns the process-global skip names added by this invocation, per set.
    The caller must pass them to :func:`restore_refit` after the layerwise
    lifecycle, including on failure.
    """
    # DeepSeek V4 loads attn_sink with a direct copy_ instead of its Parameter
    # weight_loader. Keep its kernel storage materialized during reload.
    required_skip_tensors = {"attn_sink"}

    for layer in _block_fp8_routed_experts(model):
        # Load local experts immediately instead of retaining every source
        # expert in layerwise reload until the fused parameter is complete.
        required_skip_tensors.update(_EXPERT_REFIT_PARAMS)

    skip_sets = _layerwise_skip_sets()
    added = SkipNames(
        tensors=required_skip_tensors - skip_sets[0],
        load=required_skip_tensors - skip_sets[-1],
    )
    for skip_set in skip_sets:
        skip_set.update(required_skip_tensors)
    try:
        _reset_routed_experts_for_refit(model)
    except Exception:
        _remove_skip_names(added)
        raise
    suspend_model_post_load(model)
    return added


def _remove_skip_names(added: SkipNames) -> None:
    skip_sets = _layerwise_skip_sets()
    skip_sets[0].difference_update(added.tensors)
    skip_sets[-1].difference_update(added.load)


def restore_refit(
    added_skip_tensors: SkipNames, model: torch.nn.Module | None = None
) -> None:
    """Undo :func:`prepare_refit` after the layerwise lifecycle, including on failure.

    Removes the process-global skip names added for one refit and, when the
    model is given, lifts the post-load shadow a failed refit may have left.
    """
    _remove_skip_names(added_skip_tensors)
    if model is not None:
        resume_model_post_load(model)


@torch.no_grad()
def finalize_refit(model: torch.nn.Module) -> None:
    """Finish a layerwise refit once every layer is materialized.

    Converts immediately loaded expert tensors back to their kernel layout,
    then runs the model-level post-load hook that
    :func:`suspend_model_post_load` kept off while the refit streamed.
    """
    for layer in _block_fp8_routed_experts(model):
        layer.quant_method.process_weights_after_loading(layer)
        if layer.w13_weight.is_cuda:
            # Requantization uses sizeable per-expert float32 temporaries.
            torch.cuda.empty_cache()
    resume_model_post_load(model)
    post_load = getattr(model, _MODEL_POST_LOAD_HOOK, None)
    if callable(post_load):
        post_load()
