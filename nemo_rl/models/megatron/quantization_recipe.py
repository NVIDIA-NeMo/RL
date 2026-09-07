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

from typing import Any


def first_last_bf16_local_layers(
    *,
    total_layers: int,
    global_layer_offset: int,
    local_layer_count: int,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
) -> tuple[int, ...]:
    """Return pipeline-local indices whose global layers must stay in BF16."""
    if total_layers < 0 or global_layer_offset < 0 or local_layer_count < 0:
        raise ValueError("Layer counts and offsets must be non-negative")
    if not 0 <= num_layers_at_start_in_bf16 <= total_layers:
        raise ValueError("Invalid leading BF16 layer count")
    if not 0 <= num_layers_at_end_in_bf16 <= total_layers:
        raise ValueError("Invalid trailing BF16 layer count")
    if num_layers_at_start_in_bf16 + num_layers_at_end_in_bf16 > total_layers:
        raise ValueError("Leading and trailing BF16 layer ranges overlap")

    trailing_start = total_layers - num_layers_at_end_in_bf16
    return tuple(
        local_layer
        for local_layer in range(local_layer_count)
        if (global_layer := global_layer_offset + local_layer)
        < num_layers_at_start_in_bf16
        or global_layer >= trailing_start
    )


def _find_bf16_config_key(recipe: Any) -> str:
    candidates = []
    for config_key, config in recipe.configs.items():
        training_recipe = config.get("training_recipe", {})
        evaluation_recipe = config.get("evaluation_recipe", training_recipe)
        if all(
            phase_recipe.get("fp8_quantization_recipe") is None
            and phase_recipe.get("fp4_quantization_recipe") is None
            for phase_recipe in (training_recipe, evaluation_recipe)
        ):
            candidates.append(str(config_key))
    if "bf16" in candidates:
        return "bf16"
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "A first/last BF16 override found multiple fully non-quantized "
            f"configs without a 'bf16' key: {candidates!r}"
        )
    raise ValueError(
        "A first/last BF16 override requires a config that is non-quantized "
        "during both training and evaluation"
    )


def _specialize_first_last_bf16_quant_recipe(
    model_config: Any,
    *,
    global_layer_offset: int,
    local_layer_count: int,
) -> None:
    recipe = getattr(model_config, "quant_recipe", None)
    if recipe is None or not getattr(model_config, "first_last_layers_bf16", False):
        return

    local_layers = first_last_bf16_local_layers(
        total_layers=int(model_config.num_layers),
        global_layer_offset=global_layer_offset,
        local_layer_count=local_layer_count,
        num_layers_at_start_in_bf16=int(model_config.num_layers_at_start_in_bf16),
        num_layers_at_end_in_bf16=int(model_config.num_layers_at_end_in_bf16),
    )
    if not local_layers:
        return

    from megatron.core.quantization.quant_config import GlobMatcher

    config_key = _find_bf16_config_key(recipe)
    override_patterns = tuple(f"*.layers.{layer}.*" for layer in local_layers)
    existing_patterns = {
        matcher.pattern
        for matcher in recipe.matchers
        if isinstance(matcher, GlobMatcher) and matcher.config_key == config_key
    }
    overrides = [
        GlobMatcher(pattern=pattern, config_key=config_key)
        for pattern in override_patterns
        if pattern not in existing_patterns
    ]
    recipe.matchers = overrides + recipe.matchers


def specialize_first_last_bf16_quant_recipe_for_current_pipeline_rank(
    model_config: Any,
) -> None:
    """Map global first/last BF16 policy onto pipeline-local module names."""
    recipe = getattr(model_config, "quant_recipe", None)
    if recipe is None or not getattr(model_config, "first_last_layers_bf16", False):
        return
    if getattr(model_config, "virtual_pipeline_model_parallel_size", None) is not None:
        raise NotImplementedError(
            "Per-module TE recipes with first_last_layers_bf16 do not yet support "
            "virtual pipeline parallelism"
        )

    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    _specialize_first_last_bf16_quant_recipe(
        model_config,
        global_layer_offset=get_transformer_layer_offset(model_config),
        local_layer_count=get_num_layers_to_build(model_config),
    )
