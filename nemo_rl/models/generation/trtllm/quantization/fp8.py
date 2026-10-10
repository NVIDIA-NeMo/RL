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

"""Routed-expert block-FP8 / MXFP8 refit support for TRT-LLM (MoE models)."""

import fnmatch
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

FP8_BLOCK_SIZE = (128, 128)
FP8_EXPERT_CHUNK_SIZE = 16

# MXFP8: E4M3 weights, one UE8M0 (power-of-two exponent byte) scale per 32 elements
# along K. Shares this module's plumbing with block-FP8 but not the caster.
MXFP8_BLOCK_SIZE = 32
UE8M0_BIAS = 127
E4M3_MAX = 448.0

# Never quantized, whatever the user lists (same as the vLLM path's lm_head rule).
ALWAYS_BF16_PATTERNS = ("lm_head", "*embed_tokens*")

# Default scope when ``trtllm_cfg.quantization_ignore_patterns`` is unset: routed
# experts only. TRT-LLM only has a negative module filter, so everything else
# is listed here and stays BF16.
ROUTED_EXPERTS_ONLY_IGNORE_PATTERNS = (
    "model.layers.*.self_attn*",
    "model.layers.*.linear_attn*",
    "model.layers.*.mlp.gate",
    "model.layers.*.mlp.shared_expert*",
    "model.embed_tokens",
    "model.norm",
    "lm_head",
    # MTP layer: same split as above (experts quantized, rest BF16).
    "mtp.layers.0.self_attn*",
    "mtp.layers.0.linear_attn*",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert*",
    "mtp.fc*",
    "mtp.norm*",
    "mtp.pre_fc_norm_embedding*",
    "mtp.pre_fc_norm_hidden*",
)


def validate_ignore_patterns(patterns: Any) -> list[str]:
    """Validate ``quantization_ignore_patterns`` (same rules as the vLLM path)."""
    if isinstance(patterns, (str, bytes)) or not isinstance(patterns, Sequence):
        raise ValueError("quantization_ignore_patterns must be a list of strings")
    if any(not isinstance(p, str) or not p.strip() for p in patterns):
        raise ValueError("quantization_ignore_patterns must contain non-empty strings")
    return [p.strip() for p in patterns]


def build_quant_config(
    is_mx: bool, ignore_patterns: Sequence[str] | None = None
) -> dict[str, Any]:
    """Build the TRT-LLM ``quantization_config`` for block-FP8 or MXFP8.

    ``ignore_patterns=None`` selects the routed-experts-only default scope;
    a list replaces it (``ALWAYS_BF16_PATTERNS`` are always added).
    """
    if ignore_patterns is None:
        not_converted = list(ROUTED_EXPERTS_ONLY_IGNORE_PATTERNS)
    else:
        not_converted = list(dict.fromkeys([*ignore_patterns, *ALWAYS_BF16_PATTERNS]))
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "mxfp8" if is_mx else "fp8",
        "weight_block_size": [1, MXFP8_BLOCK_SIZE] if is_mx else list(FP8_BLOCK_SIZE),
        "modules_to_not_convert": not_converted,
    }


FP8_BLOCK_QUANT_KWARGS: dict[str, Any] = build_quant_config(is_mx=False)
MXFP8_BLOCK_QUANT_KWARGS: dict[str, Any] = build_quant_config(is_mx=True)


def is_module_ignored(module_name: str, patterns: Sequence[str]) -> bool:
    """Whether the module or any ancestor matches an ignore pattern.

    Mirrors TRT-LLM's ``exclude_modules`` matching: ``fnmatch`` or ``re:`` regex.
    """
    parts = module_name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        for pattern in patterns:
            if pattern.startswith("re:"):
                if re.fullmatch(pattern[3:], candidate):
                    return True
            elif fnmatch.fnmatchcase(candidate, pattern):
                return True
            elif pattern.endswith(".*") and candidate == pattern[:-2]:
                return True
    return False


_EXPERT_PREFIX: str = (
    r"(?:"
    r"(?:(?:model\.)?(?:language_model\.)?)layers\.\d+"
    r"|mtp\.layers\.\d+"
    r")\.mlp\.experts"
)
_FUSED_EXPERT_RE = re.compile(
    rf"^(?P<prefix>{_EXPERT_PREFIX})\."
    r"(?P<projection>gate_up_proj|down_proj)$"
)
_SPLIT_EXPERT_RE = re.compile(
    rf"^(?P<prefix>{_EXPERT_PREFIX})\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_DENSE_MLP_RE = re.compile(
    r"^(?:(?:model\.)?(?:language_model\.)?)layers\.\d+\.mlp\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_SPLIT_LINEAR_ATTN_RE = re.compile(
    r"^(?:.*\.)?layers\.\d+\.linear_attn\."
    r"in_proj_(?:qkv|q|k|v|z|a|b)\..+$"
)


def validate_fused_expert_layout(
    state_dict_info: dict[str, Any],
) -> None:
    """Validate Qwen3.5's fused routed-expert layout before the first refit."""
    layouts: dict[str, dict[str, tuple[int, ...]]] = {}
    for name, metadata in state_dict_info.items():
        match = _FUSED_EXPERT_RE.fullmatch(name)
        if match is None:
            continue
        shape, _ = metadata
        layouts.setdefault(match.group("prefix"), {})[match.group("projection")] = (
            tuple(shape)
        )

    if not layouts:
        raise ValueError(
            "No Qwen3.5 fused routed-expert weights were found; expected "
            "gate_up_proj=[E,2I,H] and down_proj=[E,H,I]."
        )

    for prefix, projections in layouts.items():
        if set(projections) != {"gate_up_proj", "down_proj"}:
            raise ValueError(
                f"Qwen3.5 fused experts require both gate_up_proj and down_proj: "
                f"{prefix} has {sorted(projections)}"
            )
        gate_up_shape = projections["gate_up_proj"]
        down_shape = projections["down_proj"]
        valid = (
            len(gate_up_shape) == 3
            and len(down_shape) == 3
            and gate_up_shape[0] == down_shape[0]
            and gate_up_shape[1] == 2 * down_shape[2]
            and gate_up_shape[2] == down_shape[1]
        )
        if not valid:
            raise ValueError(
                "Qwen3.5 fused expert weights must use gate_up_proj=[E,2I,H] "
                f"and down_proj=[E,H,I]; {prefix} has "
                f"gate_up_proj={gate_up_shape}, down_proj={down_shape}"
            )


def _with_always_bf16(patterns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*patterns, *ALWAYS_BF16_PATTERNS]))


def _is_quantized_weight(name: str, ndim: int, ignore_patterns: Sequence[str]) -> bool:
    """Whether pattern mode quantizes this weight.

    True for a fused expert stack or a 2-D ``.weight`` whose module is not ignored.
    """
    ignore_patterns = _with_always_bf16(ignore_patterns)
    fused = _FUSED_EXPERT_RE.fullmatch(name)
    if fused is not None:
        return not is_module_ignored(fused.group("prefix"), ignore_patterns)
    if not name.endswith(".weight") or ndim != 2:
        return False
    return not is_module_ignored(name.removesuffix(".weight"), ignore_patterns)


def validate_routed_experts(
    state_dict_info: dict[str, Any], *, ignore_patterns: Sequence[str] | None = None
) -> None:
    """Fail setup unless the FP8 filter will quantize something, correctly.

    Routed experts are either fused (``mlp.experts.{gate_up,down}_proj``, Qwen3.5)
    or per-expert (``mlp.experts.{i}.{gate,up,down}_proj.weight``, Qwen3 MoE).
    With ``ignore_patterns=None`` (default scope) only experts are quantized, so
    at least one must exist and dense MLP weights are rejected. With patterns,
    any weight that is not ignored is quantized, so at least one must remain.
    """
    names = [str(name) for name in state_dict_info]
    has_fused = any(_FUSED_EXPERT_RE.fullmatch(name) for name in names)
    has_split = any(_SPLIT_EXPERT_RE.fullmatch(name) for name in names)
    if ignore_patterns is None:
        if not (has_fused or has_split):
            raise ValueError(
                "precision='fp8' found no routed-expert weights to quantize; "
                "expected mlp.experts.{gate_up_proj,down_proj} or "
                "mlp.experts.{i}.{gate,up,down}_proj.weight"
            )
        dense = [name for name in names if _DENSE_MLP_RE.fullmatch(name)]
        if dense:
            raise ValueError(
                "precision='fp8' quantizes only routed experts by default, but the "
                f"model has dense MLP weights that would stay BF16 (e.g. {dense[0]}); "
                "set trtllm_cfg.quantization_ignore_patterns to choose the scope"
            )
    elif not any(
        _is_quantized_weight(name, len(state_dict_info[name][0]), ignore_patterns)
        for name in names
    ):
        raise ValueError(
            "quantization_ignore_patterns ignores every weight; nothing would be "
            "quantized"
        )
    if has_fused:
        validate_fused_expert_layout(state_dict_info)


def configure_fp8_llm_kwargs(
    llm_kwargs: dict[str, Any],
    *,
    is_mx: bool = False,
    ignore_patterns: Sequence[str] | None = None,
) -> None:
    """Apply the block-FP8 / MXFP8 contract to TRT-LLM args.

    ``ignore_patterns`` (``trtllm_cfg.quantization_ignore_patterns``) sets which
    modules stay BF16; ``None`` keeps the routed-experts-only default scope.
    Conflicting quantization or load-format overrides raise at setup.
    """
    if ignore_patterns is not None:
        ignore_patterns = validate_ignore_patterns(ignore_patterns)
    base_kwargs = build_quant_config(is_mx, ignore_patterns)
    label = "MXFP8" if is_mx else "block-FP8"

    model_kwargs = dict(llm_kwargs.get("model_kwargs") or {})
    existing_quant_config = model_kwargs.get("quantization_config")
    if existing_quant_config is not None and dict(existing_quant_config) != base_kwargs:
        raise ValueError(
            f"precision='fp8' requires NeMo-RL's {label} quantization_config; "
            "set trtllm_cfg.quantization_ignore_patterns instead of overriding it"
        )

    load_format = llm_kwargs.get("load_format")
    if load_format is not None and load_format != "dummy":
        raise ValueError(
            "precision='fp8' requires load_format='dummy'; the initial BF16 "
            "trainer refit populates the FP8 weights and scales"
        )

    quantization_config = dict(base_kwargs)
    quantization_config["modules_to_not_convert"] = list(
        base_kwargs["modules_to_not_convert"]
    )
    model_kwargs["quantization_config"] = quantization_config
    llm_kwargs["model_kwargs"] = model_kwargs
    llm_kwargs["load_format"] = "dummy"
    llm_kwargs["dtype"] = "bfloat16"
    # Keep the block-FP8 Linear fallback on FP32 scales (not DeepGEMM's E8M0).
    llm_kwargs["use_cute_dsl_blockscaling_mm"] = True


def configure_fp8_moe_backend(
    llm_kwargs: dict[str, Any], moe_config_type: type[Any], *, is_mx: bool = False
) -> None:
    """Force (or check) the MoE backend for the requested scale format.

    Block-FP8 needs TRTLLM (keeps FP32 scales). MXFP8 uses CUTLASS (default) or
    CUTEDSL (Rubin).
    """
    allowed = ("CUTLASS", "CUTEDSL") if is_mx else ("TRTLLM",)
    default = allowed[0]
    reason = (
        "precision='fp8' with is_mx=true (MXFP8 routed-expert scales) requires "
        if is_mx
        else "precision='fp8' with FP32 routed-expert scales requires "
    )
    expected = " or ".join(
        f"trtllm_kwargs.moe_config.backend={backend!r}" for backend in allowed
    )

    moe_config = llm_kwargs.get("moe_config")
    if moe_config is None:
        llm_kwargs["moe_config"] = moe_config_type(backend=default)
        return

    if isinstance(moe_config, dict):
        moe_config_kwargs = dict(moe_config)
        configured_backend = str(moe_config_kwargs.get("backend", default)).upper()
        if configured_backend not in allowed:
            raise ValueError(f"{reason}{expected}, got {configured_backend!r}")
        moe_config_kwargs["backend"] = configured_backend
        llm_kwargs["moe_config"] = moe_config_type(**moe_config_kwargs)
        return

    if isinstance(moe_config, moe_config_type):
        configured_backend = str(moe_config.backend).upper()
        if configured_backend not in allowed:
            raise ValueError(f"{reason}{expected}, got {moe_config.backend!r}")
        return

    raise TypeError(
        "trtllm_kwargs.moe_config must be a dict or MoeConfig, got "
        f"{type(moe_config).__name__}"
    )


def _has_fp8_block_scales(quant_config: Any) -> bool:
    if quant_config is None:
        return False
    layer_mode = getattr(quant_config, "layer_quant_mode", None)
    if layer_mode is None:
        return False
    return layer_mode.has_fp8_block_scales() is True


def is_fp8_model(quant_config: Any) -> bool:
    """Return whether a TRT-LLM model uses 128x128 block FP8."""
    return _has_fp8_block_scales(quant_config)


def is_mxfp8_model(quant_config: Any) -> bool:
    """Return whether a TRT-LLM model uses MXFP8 (E4M3 + UE8M0 1x32)."""
    if quant_config is None:
        return False
    layer_mode = getattr(quant_config, "layer_quant_mode", None)
    if layer_mode is None:
        return False
    has_mxfp8 = getattr(layer_mode, "has_mxfp8", None)
    return has_mxfp8 is not None and has_mxfp8() is True


def is_quantized_expert_refit(quant_config: Any) -> bool:
    """Whether refit must quantize routed experts before loading."""
    return is_fp8_model(quant_config) or is_mxfp8_model(quant_config)


def cast_tensor_to_fp8_blockwise(
    data_hp: torch.Tensor,
    weight_block_size: Sequence[int] = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last two dims into E4M3 with FP32 block scales.

    Leading dims are independent. ``weight_scale_inv`` is ``[..., out_block,
    in_block]``; dequantize as ``fp8.float() * weight_scale_inv``.
    """
    if data_hp.dim() < 2:
        raise ValueError(
            "cast_tensor_to_fp8_blockwise expects at least a 2-D tensor, got "
            f"shape {tuple(data_hp.shape)}"
        )
    if len(weight_block_size) != 2:
        raise ValueError(
            f"weight_block_size must contain two dimensions, got {weight_block_size}"
        )
    block_m, block_n = (int(weight_block_size[0]), int(weight_block_size[1]))
    if (block_m, block_n) != FP8_BLOCK_SIZE:
        raise ValueError(
            "TRT-LLM FP8_BLOCK_SCALES requires weight_block_size=[128, 128], "
            f"got {[block_m, block_n]}"
        )

    batch_shape = tuple(data_hp.shape[:-2])
    rows, columns = data_hp.shape[-2:]
    pad_rows = (-rows) % block_m
    pad_columns = (-columns) % block_n

    data_fp32 = data_hp.to(torch.float32)
    if pad_rows or pad_columns:
        data_fp32 = F.pad(data_fp32, (0, pad_columns, 0, pad_rows), value=0.0)

    padded_rows, padded_columns = data_fp32.shape[-2:]
    row_blocks = padded_rows // block_m
    column_blocks = padded_columns // block_n
    batch_size = math.prod(batch_shape) if batch_shape else 1

    blocked = (
        data_fp32.reshape(
            batch_size,
            row_blocks,
            block_m,
            column_blocks,
            block_n,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .flatten(start_dim=3)
    )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    max_abs = torch.amax(torch.abs(blocked), dim=-1, keepdim=True)
    valid_scale = torch.isfinite(max_abs) & (max_abs > 0)
    scale_inv = torch.where(
        valid_scale,
        max_abs / fp8_max,
        torch.ones_like(max_abs),
    )
    quant_scale = torch.where(
        valid_scale,
        torch.reciprocal(scale_inv),
        torch.ones_like(scale_inv),
    )
    fp8_data = torch.clamp(
        blocked * quant_scale,
        min=-fp8_max,
        max=fp8_max,
    ).to(torch.float8_e4m3fn)

    fp8_data = (
        fp8_data.reshape(
            batch_size,
            row_blocks,
            column_blocks,
            block_m,
            block_n,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(*batch_shape, padded_rows, padded_columns)
    )
    fp8_data = fp8_data[..., :rows, :columns].contiguous()
    scale_inv = scale_inv.squeeze(-1).reshape(*batch_shape, row_blocks, column_blocks)
    return fp8_data, scale_inv.contiguous()


def cast_tensor_to_mxfp8_blockwise(
    data_hp: torch.Tensor,
    block_size: int = MXFP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim into E4M3 with UE8M0 1x{block} scales.

    Mirrors TRT-LLM's ``quant_bf16_to_mxfp8``: the power-of-two exponent maps
    each block's amax just under the E4M3 max. Blocks span K only.

    Returns ``(e4m3 [..., K], ue8m0 uint8 [..., K // block])``.
    """
    if data_hp.dim() < 2:
        raise ValueError(
            "cast_tensor_to_mxfp8_blockwise expects at least a 2-D tensor, got "
            f"shape {tuple(data_hp.shape)}"
        )
    columns = data_hp.shape[-1]
    if columns % block_size != 0:
        raise ValueError(
            f"MXFP8 requires the last dim to be a multiple of {block_size}, got "
            f"{columns}"
        )

    shape = tuple(data_hp.shape)
    flat = data_hp.float().reshape(-1, columns)
    blocked = flat.view(-1, columns // block_size, block_size)
    amax = blocked.abs().amax(dim=-1).clamp_min(1e-12)
    exponent = torch.ceil(torch.log2(amax / E4M3_MAX))
    scale_ue8m0 = (exponent + UE8M0_BIAS).clamp(0, 255).to(torch.uint8)
    scale = torch.exp2(exponent).unsqueeze(-1)
    quantized = (blocked / scale).to(torch.float8_e4m3fn).view(-1, columns)
    return (
        quantized.reshape(shape).contiguous(),
        scale_ue8m0.reshape(*shape[:-1], columns // block_size).contiguous(),
    )


def _insert_unique(
    output: dict[str, torch.Tensor], name: str, tensor: torch.Tensor
) -> None:
    if name in output:
        raise ValueError(f"Duplicate refit weight after FP8 conversion: {name}")
    output[name] = tensor


def _insert_quantized_projection(
    output: dict[str, torch.Tensor],
    name: str,
    tensor: torch.Tensor,
    *,
    is_mx: bool = False,
) -> None:
    if is_mx:
        data, scale = cast_tensor_to_mxfp8_blockwise(tensor)
    else:
        data, scale = cast_tensor_to_fp8_blockwise(tensor)
    _insert_unique(output, name, data)
    # Both formats use `.weight_scale_inv` (TRT-LLM's MXFP8 loader probes it first).
    scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
    _insert_unique(output, scale_name, scale)


def clone_mapper_staging_weights(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clone tensors the Qwen3.5 mapper may retain across reload calls.

    IPC buffers are reused after each ACK, so retained views must own storage.
    """
    return {
        name: tensor.clone()
        if _SPLIT_LINEAR_ATTN_RE.fullmatch(name) is not None
        else tensor
        for name, tensor in weights.items()
    }


def _convert_fused_expert_weight(
    output: dict[str, torch.Tensor],
    *,
    name: str,
    tensor: torch.Tensor,
    prefix: str,
    projection: str,
    is_mx: bool = False,
) -> None:
    if tensor.dim() != 3:
        raise ValueError(
            f"Qwen3.5 fused expert weight {name} must be 3-D, got {tuple(tensor.shape)}"
        )

    num_experts = tensor.shape[0]
    if projection == "gate_up_proj" and tensor.shape[1] % 2 != 0:
        raise ValueError(
            f"Qwen3.5 gate_up_proj dimension must be even, got {tuple(tensor.shape)}"
        )

    for start in range(0, num_experts, FP8_EXPERT_CHUNK_SIZE):
        end = min(start + FP8_EXPERT_CHUNK_SIZE, num_experts)
        if projection == "gate_up_proj":
            intermediate_size = tensor.shape[1] // 2
            projections = (
                ("gate_proj", tensor[start:end, :intermediate_size, :]),
                ("up_proj", tensor[start:end, intermediate_size:, :]),
            )
        else:
            projections = (("down_proj", tensor[start:end]),)

        for projection_name, projection_tensor in projections:
            if is_mx:
                fp8_data, scale_inv = cast_tensor_to_mxfp8_blockwise(projection_tensor)
            else:
                fp8_data, scale_inv = cast_tensor_to_fp8_blockwise(projection_tensor)
            for chunk_index, expert_index in enumerate(range(start, end)):
                weight_name = f"{prefix}.{expert_index}.{projection_name}.weight"
                scale_name = weight_name.removesuffix(".weight") + ".weight_scale_inv"
                _insert_unique(output, weight_name, fp8_data[chunk_index])
                _insert_unique(output, scale_name, scale_inv[chunk_index])


def load_weights(
    weight_list: Iterable[tuple[str, torch.Tensor]],
    *,
    is_mx: bool = False,
    ignore_patterns: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert BF16 weights to block-FP8 or MXFP8 for the TRT-LLM engine.

    With ``ignore_patterns=None`` only routed experts are quantized. With
    patterns, every 2-D ``.weight`` (and fused expert stack) whose module is not
    ignored is quantized, matching the engine's ``modules_to_not_convert``.
    Fused expert tensors (Qwen3.5) are expanded to per-expert HF names; other
    weights pass through unchanged.
    """
    output: dict[str, torch.Tensor] = {}
    for name, tensor in weight_list:
        if not isinstance(name, str):
            raise TypeError(
                f"TRT-LLM refit weight names must be strings, got {type(name).__name__}"
            )
        weight_name = str(name)
        fused_match = _FUSED_EXPERT_RE.fullmatch(weight_name)
        is_split_expert = _SPLIT_EXPERT_RE.fullmatch(weight_name) is not None
        if ignore_patterns is not None and not _is_quantized_weight(
            weight_name, tensor.dim(), ignore_patterns
        ):
            _insert_unique(output, weight_name, tensor)
        elif fused_match is not None:
            _convert_fused_expert_weight(
                output,
                name=weight_name,
                tensor=tensor,
                prefix=fused_match.group("prefix"),
                projection=fused_match.group("projection"),
                is_mx=is_mx,
            )
        elif is_split_expert or ignore_patterns is not None:
            _insert_quantized_projection(output, weight_name, tensor, is_mx=is_mx)
        else:
            _insert_unique(output, weight_name, tensor)
    return output
