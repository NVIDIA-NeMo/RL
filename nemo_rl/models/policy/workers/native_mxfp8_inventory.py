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

"""Bounded native-MXFP8 refit storage inventory checks for smoke runs."""

import json
import re
from collections.abc import Mapping
from typing import Any, Literal


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ROUTED_EXPERT_RE = re.compile(r"\.(?:mlp|mixer)\.experts\.")
_SHARED_EXPERT_RE = re.compile(r"\.(?:mlp|mixer)\.shared_experts?\.")
_ROUTER_RE = re.compile(r"\.(?:mlp|mixer)\.(?:gate|router)\.")
_QKVO_RE = re.compile(
    r"\.(?:self_attn|self_attention|attention)\.|"
    r"\.mixer\.(?:qkv_proj|o_proj|in_proj|out_proj)\."
)

_BF16_ONLY_SCOPES = ("shared_experts", "router", "qkvo", "lm_head")
_OPTIONAL_BF16_SCOPES = {"qwen30": frozenset(("shared_experts",)), "nano": frozenset()}
_MODEL_ROUTED_LAYER_COUNTS = {"qwen30": 48, "nano": 52}
_ROUTED_MODULES = frozenset(("FC1", "FC2"))
_EXPECTED_ROUTED_MODULES = {
    model_scope: frozenset(
        (layer, module)
        for layer in range(layer_count)
        for module in _ROUTED_MODULES
    )
    for model_scope, layer_count in _MODEL_ROUTED_LAYER_COUNTS.items()
}


def _scope_for_name(name: str) -> str:
    if name.startswith("lm_head.") or ".lm_head." in name:
        return "lm_head"
    if _SHARED_EXPERT_RE.search(name):
        return "shared_experts"
    if _ROUTED_EXPERT_RE.search(name):
        return "routed_experts"
    if _ROUTER_RE.search(name):
        return "router"
    if _QKVO_RE.search(name):
        return "qkvo"
    return "other"


def _layer_number(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match is not None else None


def _routed_module_key(name: str) -> tuple[int, str]:
    layer = _layer_number(name)
    if layer is None:
        raise ValueError(f"Routed-expert entry {name!r} has no layer number")
    if name.endswith((".gate_proj.weight", ".up_proj.weight", ".linear_fc1.weight")):
        return layer, "FC1"
    if name.endswith((".down_proj.weight", ".linear_fc2.weight")):
        return layer, "FC2"
    raise ValueError(f"Routed-expert entry {name!r} is not an FC1 or FC2 weight")


def _format_routed_module(module: tuple[int, str]) -> str:
    return f"layer {module[0]} {module[1]}"


def _validate_native_components(name: str, metadata: Mapping[str, Any]) -> None:
    components = metadata.get("components")
    if not isinstance(components, list):
        raise ValueError(f"Native MXFP8 inventory entry {name!r} has no component list")
    roles = [component.get("role") for component in components if isinstance(component, Mapping)]
    dtypes = [component.get("dtype") for component in components if isinstance(component, Mapping)]
    if roles != ["weight", "weight_scale"] or dtypes != [
        "torch.float8_e4m3fn",
        "torch.uint8",
    ]:
        raise ValueError(
            f"Native MXFP8 inventory entry {name!r} must expose E4M3 weight and uint8 scale"
        )


def assert_native_mxfp8_storage_inventory(
    *,
    native_metadata: Mapping[str, Mapping[str, Any]],
    misc_metadata: Mapping[str, Mapping[str, Any]],
    model_scope: Literal["qwen30", "nano"],
    num_layers_at_end_in_bf16: int,
) -> dict[str, dict[str, int | list[int]]]:
    """Validate and emit a compact inventory of native and BF16 refit entries.

    The caller passes metadata already produced for the actual refit operation;
    this routine never reads module tensors or allocates payload-sized buffers.
    """
    scope_counts: dict[str, dict[str, int]] = {
        scope: {"native": 0, "bf16": 0}
        for scope in (*_BF16_ONLY_SCOPES, "routed_experts", "other")
    }
    routed_native_modules: set[tuple[int, str]] = set()
    routed_bf16_modules: set[tuple[int, str]] = set()

    for name, metadata in native_metadata.items():
        scope = _scope_for_name(name)
        _validate_native_components(name, metadata)
        scope_counts[scope]["native"] += 1
        if scope == "routed_experts":
            routed_native_modules.add(_routed_module_key(name))

    for name, metadata in misc_metadata.items():
        scope = _scope_for_name(name)
        if metadata.get("dtype") != "torch.bfloat16":
            continue
        scope_counts[scope]["bf16"] += 1
        if scope == "routed_experts":
            routed_bf16_modules.add(_routed_module_key(name))

    if scope_counts["other"]["native"]:
        raise ValueError("Native MXFP8 inventory contains an unsupported module scope")
    if not scope_counts["routed_experts"]["native"]:
        raise ValueError("Native MXFP8 inventory has no routed experts")

    for scope in _BF16_ONLY_SCOPES:
        if scope_counts[scope]["native"]:
            raise ValueError(f"Native MXFP8 inventory {scope} entries must remain BF16")
        if (
            scope not in _OPTIONAL_BF16_SCOPES[model_scope]
            and not scope_counts[scope]["bf16"]
        ):
            raise ValueError(f"Native MXFP8 inventory {scope} BF16 entries are missing")

    expected_modules = _EXPECTED_ROUTED_MODULES[model_scope]
    layer_count = _MODEL_ROUTED_LAYER_COUNTS[model_scope]
    if not 0 <= num_layers_at_end_in_bf16 <= layer_count:
        raise ValueError(
            f"Native MXFP8 inventory final BF16 layer count must be in [0, {layer_count}]"
        )
    expected_last_layers = frozenset(
        range(layer_count - num_layers_at_end_in_bf16, layer_count)
    )
    expected_bf16_modules = frozenset(
        (layer, module)
        for layer in expected_last_layers
        for module in _ROUTED_MODULES
    )
    expected_native_modules = expected_modules - expected_bf16_modules
    observed_modules = routed_native_modules | routed_bf16_modules
    unexpected_modules = observed_modules - expected_modules
    if unexpected_modules:
        raise ValueError(
            "Native MXFP8 inventory has an unexpected routed module "
            f"{_format_routed_module(min(unexpected_modules))}"
        )

    for expected_storage, required_modules, observed_modules_for_storage in (
        ("native", expected_native_modules, routed_native_modules),
        ("BF16", expected_bf16_modules, routed_bf16_modules),
    ):
        missing_modules = required_modules - observed_modules_for_storage
        if missing_modules:
            raise ValueError(
                "Native MXFP8 inventory routed "
                f"{_format_routed_module(min(missing_modules))} must be {expected_storage}"
            )

    for forbidden_storage, forbidden_modules, observed_modules_for_storage in (
        ("BF16", expected_native_modules, routed_bf16_modules),
        ("native", expected_bf16_modules, routed_native_modules),
    ):
        wrong_modules = forbidden_modules & observed_modules_for_storage
        if wrong_modules:
            raise ValueError(
                "Native MXFP8 inventory routed "
                f"{_format_routed_module(min(wrong_modules))} must not be {forbidden_storage}"
            )

    inventory: dict[str, dict[str, int | list[int]]] = {}
    for scope, counts in scope_counts.items():
        inventory[scope] = {"native": counts["native"], "bf16": counts["bf16"]}
    inventory["last_layers_bf16"] = {"layers": sorted(expected_last_layers)}
    inventory.pop("other")
    print(f"[native-mxfp8-inventory] {json.dumps(inventory, sort_keys=True)}", flush=True)
    return inventory
