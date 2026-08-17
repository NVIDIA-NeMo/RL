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

"""Qwen3.5 routed-expert block-FP8 refit support for TRT-LLM."""

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F


FP8_BLOCK_SIZE = (128, 128)
FP8_EXPERT_CHUNK_SIZE = 16

# MXFP8: E4M3 weights with one UE8M0 (power-of-two) scale per 32 elements along
# K. Unlike 128x128 block FP8 the scale is an exponent byte, not an FP32
# multiplier, so the two paths share this module's plumbing but not the caster.
MXFP8_BLOCK_SIZE = 32
UE8M0_BIAS = 127
E4M3_MAX = 448.0

# TRT-LLM's HF quantization config only has a negative module filter. Keep
# every Qwen3.5 linear outside the routed-expert subtree in BF16.
FP8_BLOCK_QUANT_KWARGS: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": list(FP8_BLOCK_SIZE),
    "modules_to_not_convert": [
        "model.layers.*.self_attn*",
        "model.layers.*.linear_attn*",
        "model.layers.*.mlp.gate",
        "model.layers.*.mlp.shared_expert*",
        "model.embed_tokens",
        "model.norm",
        "lm_head",
        # Qwen3.5 has one MTP layer. Keep its attention/shared paths in BF16,
        # but quantize its routed experts under the same contract.
        "mtp.layers.0.self_attn*",
        "mtp.layers.0.linear_attn*",
        "mtp.layers.0.mlp.gate",
        "mtp.layers.0.mlp.shared_expert*",
        "mtp.fc*",
        "mtp.norm*",
        "mtp.pre_fc_norm_embedding*",
        "mtp.pre_fc_norm_hidden*",
    ],
}

# Same experts-only scope as the block-FP8 contract; only the format differs.
# model_config.py maps quant_method="mxfp8" to QuantAlgo.MXFP8, asserts the
# block size is exactly [1, 32], and folds modules_to_not_convert into
# exclude_modules just as it does for the fp8 branch.
MXFP8_BLOCK_QUANT_KWARGS: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "mxfp8",
    "weight_block_size": [1, MXFP8_BLOCK_SIZE],
    "modules_to_not_convert": list(FP8_BLOCK_QUANT_KWARGS["modules_to_not_convert"]),
}


_QWEN35_PREFIX = (
    r"(?:"
    r"(?:(?:model\.)?(?:language_model\.)?)layers\.\d+"
    r"|mtp\.layers\.\d+"
    r")\.mlp\.experts"
)
_FUSED_EXPERT_RE = re.compile(
    rf"^(?P<prefix>{_QWEN35_PREFIX})\."
    r"(?P<projection>gate_up_proj|down_proj)$"
)
_SPLIT_EXPERT_RE = re.compile(
    rf"^(?P<prefix>{_QWEN35_PREFIX})\.\d+\."
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


def configure_fp8_llm_kwargs(
    llm_kwargs: dict[str, Any], *, model_type: str, is_mx: bool = False
) -> None:
    """Apply the experts-only block-FP8 / MXFP8 contract to TRT-LLM args.

    Existing Qwen3.5 ``model_kwargs`` entries are preserved. Conflicting
    quantization or load-format overrides fail at setup instead of silently
    creating a runtime whose refit schema differs from this converter.
    """
    if model_type != "qwen3_5_moe":
        raise ValueError(
            "precision='fp8' currently supports only Qwen3.5 MoE, got "
            f"model_type={model_type!r}"
        )

    base_kwargs = MXFP8_BLOCK_QUANT_KWARGS if is_mx else FP8_BLOCK_QUANT_KWARGS
    label = "MXFP8" if is_mx else "block-FP8"

    model_kwargs = dict(llm_kwargs.get("model_kwargs") or {})
    existing_quant_config = model_kwargs.get("quantization_config")
    if (
        existing_quant_config is not None
        and dict(existing_quant_config) != base_kwargs
    ):
        raise ValueError(
            "precision='fp8' requires NeMo-RL's Qwen3.5 routed-experts-only "
            f"{label} quantization_config"
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
    # This keeps any block-FP8 Linear fallback on the FP32-scale path. Routed
    # experts additionally require MoeConfig(backend="TRTLLM"), configured by
    # the worker.
    # llm_kwargs["use_cute_dsl_blockscaling_mm"] = True


def configure_fp8_moe_backend(
    llm_kwargs: dict[str, Any], moe_config_type: type[Any], *, is_mx: bool = False
) -> None:
    """Force the MoE backend that implements the requested scale format.

    Block-FP8 routed experts need TRTLLMGen (it keeps the FP32 scale buffers
    DeepGEMM would resmooth to E8M0). MXFP8 is the other way round: only
    ``MXFP8CutlassFusedMoEMethod`` exists, wired in ``fused_moe_cutlass.py``,
    so the CUTLASS backend is the only one that can serve it.
    """
    required = "CUTLASS" if is_mx else "TRTLLM"
    reason = (
        "precision='fp8' with is_mx=true (MXFP8 routed-expert scales) requires "
        if is_mx
        else "precision='fp8' with FP32 routed-expert scales requires "
    )

    moe_config = llm_kwargs.get("moe_config")
    if moe_config is None:
        llm_kwargs["moe_config"] = moe_config_type(backend=required)
        return

    if isinstance(moe_config, dict):
        moe_config_kwargs = dict(moe_config)
        configured_backend = str(moe_config_kwargs.get("backend", required)).upper()
        if configured_backend != required:
            raise ValueError(
                f"{reason}trtllm_kwargs.moe_config.backend={required!r}, got "
                f"{configured_backend!r}"
            )
        moe_config_kwargs["backend"] = required
        llm_kwargs["moe_config"] = moe_config_type(**moe_config_kwargs)
        return

    if isinstance(moe_config, moe_config_type):
        configured_backend = str(moe_config.backend).upper()
        if configured_backend != required:
            raise ValueError(
                f"{reason}trtllm_kwargs.moe_config.backend={required!r}, got "
                f"{moe_config.backend!r}"
            )
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
    """Whether the refit path must quantize routed experts before loading.

    Both block-FP8 and MXFP8 run the experts-only converter; only the caster
    and the scale dtype differ.
    """
    return is_fp8_model(quant_config) or is_mxfp8_model(quant_config)


def cast_tensor_to_fp8_blockwise(
    data_hp: torch.Tensor,
    weight_block_size: Sequence[int] = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the final two dimensions into E4M3 with FP32 block scales.

    Leading dimensions are treated as independent matrices. The returned
    ``weight_scale_inv`` uses ``[..., out_block, in_block]`` orientation and
    dequantizes as ``fp8.float() * weight_scale_inv`` per block.
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

    Mirrors TRT-LLM's ``quant_bf16_to_mxfp8``: pick the power-of-two exponent
    that maps each block's amax just under the E4M3 max, and ship the biased
    exponent as a uint8. Leading dims are independent, so an ``[E, N, K]``
    expert stack folds into rows -- blocks span K only.

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
    # Both formats ship the scale under `.weight_scale_inv`; TRT-LLM's MXFP8
    # loader probes weight_scale_inv before weight_scale.
    scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
    _insert_unique(output, scale_name, scale)


def clone_mapper_staging_weights(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Detach split linear-attention tensors that the mapper may retain.

    Qwen3.5's mapper stages partial QKVZ and BA groups across reload calls. IPC
    transport buffers are acknowledged and reused after each call, so retained
    views must own their storage before the sender can overwrite that buffer.
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
                fp8_data, scale_inv = cast_tensor_to_mxfp8_blockwise(
                    projection_tensor
                )
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
) -> dict[str, torch.Tensor]:
    """Convert only Qwen3.5 routed experts from BF16 to HF block-FP8 or MXFP8.

    Fused Transformers/Megatron expert tensors are expanded into the standard
    per-expert HF names consumed by TRT-LLM's Qwen3.5 mapper. All non-routed
    weights pass through unchanged.
    """
    output: dict[str, torch.Tensor] = {}
    for name, tensor in weight_list:
        if not isinstance(name, str):
            raise TypeError(
                f"TRT-LLM refit weight names must be strings, got {type(name).__name__}"
            )
        weight_name = str(name)
        fused_match = (
            _FUSED_EXPERT_RE.fullmatch(  # pyrefly: ignore[no-matching-overload]
                weight_name
            )
        )
        if fused_match is not None:
            _convert_fused_expert_weight(
                output,
                name=weight_name,
                tensor=tensor,
                prefix=fused_match.group("prefix"),
                projection=fused_match.group("projection"),
                is_mx=is_mx,
            )
        elif (
            _SPLIT_EXPERT_RE.fullmatch(  # pyrefly: ignore[no-matching-overload]
                weight_name
            )
            is not None
        ):
            _insert_quantized_projection(output, weight_name, tensor, is_mx=is_mx)
        else:
            _insert_unique(output, weight_name, tensor)
    return output
