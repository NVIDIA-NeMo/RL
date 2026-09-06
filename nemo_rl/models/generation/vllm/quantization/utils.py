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
"""Shared helpers for resolving HF checkpoint names in vLLM model trees."""

from dataclasses import dataclass

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


@dataclass(frozen=True)
class ModuleResolution:
    """Successful checkpoint-name resolution.

    ``mapped_path`` records the vLLM-side module path after packed-module and
    HF-to-vLLM mapping. ``consumed_parts`` is useful when a walk stops at the
    routed-expert ownership boundary before per-expert checkpoint components.
    """

    module: torch.nn.Module
    mapped_path: tuple[str, ...]
    consumed_parts: int


def resolve_module_from_param_name(
    model: torch.nn.Module, name: str
) -> ModuleResolution | None:
    """Resolve the vLLM module owning an HF-named checkpoint parameter.

    Resolution deliberately returns ``None`` after any failed path component;
    returning the last traversed parent can misclassify an unresolved target as
    an ordinary BF16 tensor. Routed-expert walks stop at ``RoutedExperts`` (or
    its ``MoERunner`` owner), because per-expert checkpoint components are not
    represented as child modules in vLLM.
    """
    mapped_name = name
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    for original, replacement in getattr(mapper, "orig_to_new_prefix", {}).items():
        if mapped_name.startswith(original):
            if replacement is None:
                return None
            mapped_name = f"{replacement}{mapped_name[len(original) :]}"
    for original, replacement in getattr(mapper, "orig_to_new_substr", {}).items():
        if original in mapped_name:
            if replacement is None:
                return None
            mapped_name = mapped_name.replace(original, replacement)

    path_parts = mapped_name.split(".")
    if len(path_parts) < 2:
        return None
    module_path = path_parts[:-1]

    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names in getattr(
            model, "packed_modules_mapping", {}
        ).items()
        for original_name in original_names
    }
    if module_path[-1] in reversed_mapping:
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module: torch.nn.Module = model
    for index, part in enumerate(module_path):
        if isinstance(current_module, MoERunner):
            return ModuleResolution(
                current_module.routed_experts, tuple(module_path), index
            )
        if isinstance(current_module, RoutedExperts):
            return ModuleResolution(current_module, tuple(module_path), index)
        if part == "model" and not hasattr(current_module, part):
            continue
        if part == "layers" and not hasattr(current_module, part):
            wrapped_model = getattr(current_module, "model", None)
            if wrapped_model is not None and hasattr(wrapped_model, part):
                current_module = wrapped_model
        try:
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None

    if isinstance(current_module, MoERunner):
        current_module = current_module.routed_experts
    if not isinstance(current_module, torch.nn.Module):
        return None
    return ModuleResolution(current_module, tuple(module_path), len(module_path))
