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

"""Shared configuration helpers for quantized SGLang checkpoints and refits."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nemo_rl.models.generation.sglang.config import (
    SglangQuantizationScheme,
    get_sglang_quantization_scheme,
)

HF_MOE_EXPERT_NAME_MARKERS = (
    ".experts.",
    ".shared_expert.",
    ".shared_experts.",
    "block_sparse_moe.experts.",
    ".moe.experts.",
)
HF_FUSED_LINEAR_WEIGHT_GROUPS = (
    (
        "qkv_proj",
        (".q_proj.weight", ".k_proj.weight", ".v_proj.weight"),
    ),
    (
        "gate_up_proj",
        (".gate_proj.weight", ".up_proj.weight"),
    ),
    (
        "gate_up_proj",
        (".w1.weight", ".w3.weight"),
    ),
)

def ensure_sglang_quantized_checkpoint(
    *,
    model_path: str,
    quantization_config: dict[str, Any],
) -> str:
    """Resolve the startup checkpoint for the configured refit precision."""
    scheme = get_sglang_quantization_scheme(quantization_config)
    if scheme == "bf16":
        return model_path

    from nemo_rl.models.generation.sglang.mxfp8_setup import (
        ensure_mxfp8_checkpoint,
    )

    return ensure_mxfp8_checkpoint(
        model_path=model_path,
        quantization_cfg=quantization_config,
    )


def validate_sglang_quantized_refit_backend(
    *,
    scheme: SglangQuantizationScheme,
    use_megatron: bool,
) -> None:
    """Reject quantized refits on backends that cannot emit quantized weights."""
    if scheme != "bf16" and not use_megatron:
        raise NotImplementedError(
            f"SGLang {scheme} weight refit requires a Megatron policy because "
            "the DTensor/FSDP refit path only supports BF16."
        )


def validate_checkpoint_high_precision_layout(
    *,
    checkpoint_path: str,
    scheme: str,
    weight_names: Iterable[object],
    high_precision_substrings: tuple[str, ...],
    quantized_companion_suffixes: tuple[str, ...],
) -> None:
    """Ensure a reused checkpoint already honors the requested BF16 policy.

    Changing the online refit skip rules cannot change tensors already loaded
    by SGLang. A matching weight with quantization companions proves that the
    existing checkpoint still stores that tensor in the quantized layout.
    """
    names: set[str] = set()
    for name in weight_names:
        if not isinstance(name, str):
            raise TypeError("HF checkpoint weight names must be strings.")
        names.add(name)

    conflicts: list[str] = []
    for name in sorted(names):
        if not name.endswith(".weight") or not any(
            substring in name for substring in high_precision_substrings
        ):
            continue
        base_name = name.removesuffix(".weight")
        if any(base_name + suffix in names for suffix in quantized_companion_suffixes):
            conflicts.append(name)

    if conflicts:
        preview = ", ".join(repr(name) for name in conflicts[:5])
        if len(conflicts) > 5:
            preview += f", ... ({len(conflicts)} total)"
        raise ValueError(
            f"The existing {scheme} checkpoint at {checkpoint_path!r} contains "
            "quantized tensors selected for high precision by the current "
            f"configuration: {preview}. Reconvert the original HF checkpoint "
            "with the requested extra/head/tail skip policy."
        )


def get_hf_moe_expert_container(name: str) -> str | None:
    """Return the fused SGLang MoE module that owns an HF expert tensor."""
    for marker in sorted(HF_MOE_EXPERT_NAME_MARKERS, key=len, reverse=True):
        marker_index = name.find(marker)
        if marker_index >= 0:
            return name[:marker_index] + marker.rstrip(".")
    return None


def expand_fused_moe_high_precision_substrings(
    *,
    weight_names: Iterable[object],
    skip_weight_substrings: tuple[str, ...],
) -> tuple[str, ...]:
    """Expand any skipped expert tensor to its whole fused MoE container.

    SGLang chooses one quantization method for a complete ``FusedMoE`` module,
    so a checkpoint cannot mix quantized and high-precision expert tensors
    within that container.
    """
    skipped_containers: set[str] = set()
    for name in weight_names:
        if not isinstance(name, str):
            raise TypeError("HF checkpoint weight names must be strings.")
        container = get_hf_moe_expert_container(name)
        if container is not None and any(
            substring in name for substring in skip_weight_substrings
        ):
            skipped_containers.add(container)
    return tuple(dict.fromkeys((*skip_weight_substrings, *sorted(skipped_containers))))


def expand_sglang_atomic_high_precision_substrings(
    *,
    weight_names: Iterable[object],
    skip_weight_substrings: tuple[str, ...],
) -> tuple[str, ...]:
    """Expand skip rules across SGLang's fused linear and MoE boundaries."""
    names: set[str] = set()
    for name in weight_names:
        if not isinstance(name, str):
            raise TypeError("HF checkpoint weight names must be strings.")
        names.add(name)

    added_modules: set[str] = set()
    visited_groups: set[tuple[str, tuple[str, ...]]] = set()
    for name in names:
        for fused_name, suffixes in HF_FUSED_LINEAR_WEIGHT_GROUPS:
            matching_suffix = next(
                (suffix for suffix in suffixes if name.endswith(suffix)),
                None,
            )
            if matching_suffix is None:
                continue
            prefix = name[: -len(matching_suffix)]
            group_key = (prefix, suffixes)
            if group_key in visited_groups:
                continue
            visited_groups.add(group_key)

            members = tuple(
                prefix + suffix for suffix in suffixes if prefix + suffix in names
            )
            fused_module = f"{prefix}.{fused_name}"
            if len(members) >= 2 and (
                any(
                    substring in member
                    for member in members
                    for substring in skip_weight_substrings
                )
                or any(
                    substring in fused_module for substring in skip_weight_substrings
                )
            ):
                added_modules.update(
                    member.removesuffix(".weight") for member in members
                )

    expanded = tuple(dict.fromkeys((*skip_weight_substrings, *sorted(added_modules))))
    return expand_fused_moe_high_precision_substrings(
        weight_names=names,
        skip_weight_substrings=expanded,
    )


def _optional_string_tuple(
    quantization_config: dict[str, Any],
    key: str,
) -> tuple[str, ...]:
    value = quantization_config.get(key)
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"SGLang quantization.{key} must be a list of strings.")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"SGLang quantization.{key} entries must be non-empty strings."
            )
        result.append(item.strip())
    return tuple(result)


def _optional_nonnegative_int(
    quantization_config: dict[str, Any],
    key: str,
) -> int:
    value = quantization_config.get(key)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"SGLang quantization.{key} must be an integer.")
    if value < 0:
        raise ValueError(f"SGLang quantization.{key} must be non-negative.")
    return value


def get_dynamic_high_precision_substrings(
    *,
    quantization_config: dict[str, Any],
    num_hidden_layers: int,
) -> tuple[str, ...]:
    """Build HF-name substrings that must remain in high precision.

    The result combines explicit HF-name patterns, checkpoint loader ignore
    rules, and the configured BF16 bands at the start and end of the decoder.
    """
    extra_high_precision_layers_hf = _optional_string_tuple(
        quantization_config,
        "extra_high_precision_layers_hf",
    )
    modules_to_not_convert = _optional_string_tuple(
        quantization_config,
        "modules_to_not_convert",
    )
    num_layers_at_start_in_bf16 = _optional_nonnegative_int(
        quantization_config,
        "num_layers_at_start_in_bf16",
    )
    num_layers_at_end_in_bf16 = _optional_nonnegative_int(
        quantization_config,
        "num_layers_at_end_in_bf16",
    )

    if num_layers_at_start_in_bf16 or num_layers_at_end_in_bf16:
        if num_hidden_layers <= 0:
            raise ValueError(
                "num_hidden_layers must be positive when BF16 head/tail layers "
                "are configured."
            )
        if num_layers_at_start_in_bf16 + num_layers_at_end_in_bf16 > num_hidden_layers:
            raise ValueError(
                "The configured BF16 head/tail layer counts exceed the model's "
                f"{num_hidden_layers} decoder layers."
            )

    tail_start_idx = num_hidden_layers - num_layers_at_end_in_bf16
    dynamic_layer_prefixes = [
        *(f"model.layers.{i}." for i in range(num_layers_at_start_in_bf16)),
        *(f"model.layers.{i}." for i in range(tail_start_idx, num_hidden_layers)),
    ]

    # Preserve user order while removing duplicates so matching and cache
    # fingerprints remain deterministic.
    return tuple(
        dict.fromkeys(
            (
                *extra_high_precision_layers_hf,
                *modules_to_not_convert,
                *dynamic_layer_prefixes,
            )
        )
    )


def build_dynamic_skip_substrings(
    *,
    quantization_config: dict[str, Any],
    num_hidden_layers: int,
    static_skip_substrings: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Combine static exclusions with the configured high-precision rules."""
    dynamic_substrings = get_dynamic_high_precision_substrings(
        quantization_config=quantization_config,
        num_hidden_layers=num_hidden_layers,
    )
    return tuple(dict.fromkeys((*static_skip_substrings, *dynamic_substrings)))
