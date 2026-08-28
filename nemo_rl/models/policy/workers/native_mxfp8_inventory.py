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
    routed_native_layers: set[int] = set()
    routed_bf16_layers: set[int] = set()

    for name, metadata in native_metadata.items():
        scope = _scope_for_name(name)
        _validate_native_components(name, metadata)
        scope_counts[scope]["native"] += 1
        if scope == "routed_experts":
            layer = _layer_number(name)
            if layer is None:
                raise ValueError(f"Native routed-expert entry {name!r} has no layer number")
            routed_native_layers.add(layer)

    for name, metadata in misc_metadata.items():
        scope = _scope_for_name(name)
        if metadata.get("dtype") != "torch.bfloat16":
            continue
        scope_counts[scope]["bf16"] += 1
        if scope == "routed_experts":
            layer = _layer_number(name)
            if layer is None:
                raise ValueError(f"BF16 routed-expert entry {name!r} has no layer number")
            routed_bf16_layers.add(layer)

    if scope_counts["other"]["native"]:
        raise ValueError("Native MXFP8 inventory contains an unsupported module scope")
    if not scope_counts["routed_experts"]["native"]:
        raise ValueError("Native MXFP8 inventory has no routed experts")

    required_bf16_scopes = set(_BF16_ONLY_SCOPES)
    if model_scope == "qwen30":
        required_bf16_scopes.remove("shared_experts")
    for scope in required_bf16_scopes:
        if scope_counts[scope]["native"]:
            raise ValueError(f"Native MXFP8 inventory {scope} entries must remain BF16")
        if not scope_counts[scope]["bf16"]:
            raise ValueError(f"Native MXFP8 inventory {scope} BF16 entries are missing")

    all_routed_layers = routed_native_layers | routed_bf16_layers
    if num_layers_at_end_in_bf16:
        if not all_routed_layers:
            raise ValueError("Native MXFP8 inventory cannot determine routed-expert layers")
        last_layer = max(all_routed_layers)
        expected_last_layers = set(
            range(last_layer - num_layers_at_end_in_bf16 + 1, last_layer + 1)
        )
        if expected_last_layers & routed_native_layers:
            raise ValueError("Native MXFP8 inventory final routed-expert layers must remain BF16")
        if not expected_last_layers <= routed_bf16_layers:
            raise ValueError("Native MXFP8 inventory final BF16 routed-expert layers are missing")
    else:
        expected_last_layers = set()

    inventory: dict[str, dict[str, int | list[int]]] = {}
    for scope, counts in scope_counts.items():
        inventory[scope] = {"native": counts["native"], "bf16": counts["bf16"]}
    inventory["last_layers_bf16"] = {"layers": sorted(expected_last_layers)}
    inventory.pop("other")
    print(f"[native-mxfp8-inventory] {json.dumps(inventory, sort_keys=True)}", flush=True)
    return inventory
