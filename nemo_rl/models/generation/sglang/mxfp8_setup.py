# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Offline HF -> MXFP8 conversion + startup helper for SGLang.

Wraps NeMo-RL's quantization core so SGLang can boot from an MXFP8 HF
checkpoint and the online weight-update path can reuse the exact same
per-tensor decisions.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
from typing import Any

import torch

from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    MXFP8_QUANTIZATION_CONFIG,
    SKIP_WEIGHT_SUBSTRINGS,
    SOURCE_FP8_BLOCK_SIZE,
    SOURCE_FP8_DTYPES,
    SOURCE_FP8_SCALE_KEY_SUFFIX,
    TARGET_MXFP8_BLOCK_SIZE,
    is_bf16_source_checkpoint,
    is_mxfp8_quantization_config,
    is_source_block_fp8_ue8m0_checkpoint,
    quantize_mxfp8,
    should_quantize,
    source_fp8_to_mxfp8_scale_u8,
    strip_weight_suffix,
)

logger = logging.getLogger(__name__)

CONVERTER_VERSION: str = "1"


class _ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(
        self,
        filename: str,
        q_weights: dict[str, torch.Tensor],
        module_names: list[str],
    ) -> None:
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def _load_source_scale_u8(
    weights: dict[str, torch.Tensor],
    weight_key: str,
    weight: torch.Tensor,
    *,
    source_scale_index: dict[str, str],
    input_path: str,
    device: str,
    current_filename: str,
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    import safetensors

    scale_key = strip_weight_suffix(weight_key) + SOURCE_FP8_SCALE_KEY_SUFFIX
    scale_file = source_scale_index[scale_key]
    if scale_file == current_filename and scale_key in weights:
        scale = weights[scale_key]
    else:
        with safetensors.safe_open(
            os.path.join(input_path, scale_file), framework="pt", device=device
        ) as f:
            scale = f.get_tensor(scale_key)

    if scale.dtype == torch.uint8:
        scale_u8: torch.Tensor | None = scale
    else:
        if scale.dtype != torch.float32:
            raise ValueError(
                f"Unexpected source FP8 scale dtype {scale.dtype} for {scale_key}"
            )
        n, k = weight.shape[-2], weight.shape[-1]
        n_tiles = (n + SOURCE_FP8_BLOCK_SIZE[0] - 1) // SOURCE_FP8_BLOCK_SIZE[0]
        k_tiles = (k + SOURCE_FP8_BLOCK_SIZE[1] - 1) // SOURCE_FP8_BLOCK_SIZE[1]
        scale_fp32 = scale[..., :n_tiles, :k_tiles].contiguous()
        bits = scale_fp32.contiguous().view(torch.int32)
        mantissa_all_zero = not torch.any((bits & 0x007FFFFF) != 0).item()
        non_negative = not torch.any(bits < 0).item()
        if mantissa_all_zero and non_negative:
            scale_u8 = ((bits >> 23) & 0xFF).to(torch.uint8)
        else:
            scale_u8 = None
        return scale_fp32, scale_u8, scale_key

    n, k = weight.shape[-2], weight.shape[-1]
    n_tiles = (n + SOURCE_FP8_BLOCK_SIZE[0] - 1) // SOURCE_FP8_BLOCK_SIZE[0]
    k_tiles = (k + SOURCE_FP8_BLOCK_SIZE[1] - 1) // SOURCE_FP8_BLOCK_SIZE[1]
    scale_u8 = scale_u8[..., :n_tiles, :k_tiles].contiguous()
    scale_fp32 = (scale_u8.to(torch.int32) << 23).view(torch.float32)
    return scale_fp32, scale_u8, scale_key


def _process_file(
    input_path: str,
    output_path: str,
    filename: str,
    *,
    result_collector: _ConversionResult,
    device: str,
    num_hidden_layers: int,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
    source_is_block_fp8_ue8m0: bool,
    extra_high_precision_layers_hf: tuple[str, ...],
    source_scale_index: dict[str, str],
) -> None:
    import safetensors
    import safetensors.torch
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    weights: dict[str, torch.Tensor] = {}
    q_weights: dict[str, torch.Tensor] = {}

    with safetensors.safe_open(
        os.path.join(input_path, filename), framework="pt", device=device
    ) as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    modules_to_not_convert: list[str] = []
    head_end_idx = num_layers_at_start_in_bf16
    tail_start_idx = num_hidden_layers - num_layers_at_end_in_bf16
    dynamic_skip_layer_prefixes: set[str] = set()
    dynamic_skip_layer_prefixes.update(
        f"model.layers.{i}." for i in range(0, head_end_idx)
    )
    dynamic_skip_layer_prefixes.update(
        f"model.layers.{i}." for i in range(tail_start_idx, num_hidden_layers)
    )

    if num_layers_at_end_in_bf16 > 0 or num_layers_at_start_in_bf16 > 0:
        modules_to_not_convert.extend(sorted(dynamic_skip_layer_prefixes))

    dynamic_skip_substrings = (
        *SKIP_WEIGHT_SUBSTRINGS,
        *extra_high_precision_layers_hf,
        *sorted(dynamic_skip_layer_prefixes),
    )

    for key, tensor in weights.items():
        if not key.endswith(".weight"):
            continue

        should_quant = should_quantize(
            key,
            tensor,
            skip_weight_substrings=dynamic_skip_substrings,
            allow_source_fp8=source_is_block_fp8_ue8m0,
        )

        if should_quant:
            if source_is_block_fp8_ue8m0 and tensor.dtype in SOURCE_FP8_DTYPES:
                source_scale_fp32, source_scale_u8, scale_key = _load_source_scale_u8(
                    weights,
                    key,
                    tensor,
                    source_scale_index=source_scale_index,
                    input_path=input_path,
                    device=device,
                    current_filename=filename,
                )
                if source_scale_u8 is not None:
                    qweight = tensor.contiguous()
                    scale = source_fp8_to_mxfp8_scale_u8(tensor, source_scale_u8)
                else:
                    weight_fp32 = block_quant_dequant(
                        tensor,
                        source_scale_fp32,
                        SOURCE_FP8_BLOCK_SIZE,
                        torch.float32,
                    ).contiguous()
                    qweight, scale = quantize_mxfp8(weight_fp32)
                q_weights[key] = qweight
                q_weights[scale_key] = scale
            else:
                qweight, scale = quantize_mxfp8(tensor)
                q_weights[key] = qweight
                q_weights[strip_weight_suffix(key) + SOURCE_FP8_SCALE_KEY_SUFFIX] = (
                    scale
                )
        else:
            if ".experts." not in key:
                modules_to_not_convert.append(strip_weight_suffix(key))
            if source_is_block_fp8_ue8m0 and tensor.dtype in SOURCE_FP8_DTYPES:
                source_scale_fp32, _, _ = _load_source_scale_u8(
                    weights,
                    key,
                    tensor,
                    source_scale_index=source_scale_index,
                    input_path=input_path,
                    device=device,
                    current_filename=filename,
                )
                q_weights[key] = block_quant_dequant(
                    tensor,
                    source_scale_fp32,
                    SOURCE_FP8_BLOCK_SIZE,
                    torch.bfloat16,
                ).contiguous()
            else:
                q_weights[key] = tensor

    for key, tensor in weights.items():
        if key.endswith(".weight"):
            continue
        if source_is_block_fp8_ue8m0 and key.endswith(SOURCE_FP8_SCALE_KEY_SUFFIX):
            continue
        q_weights[key] = tensor

    safetensors.torch.save_file(
        q_weights, os.path.join(output_path, filename), metadata={"format": "pt"}
    )
    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_mxfp8(
    model_dir: str,
    save_dir: str,
    *,
    device: str = "cuda",
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    extra_high_precision_layers_hf: tuple[str, ...] = (),
) -> None:
    """Convert an HF safetensors checkpoint to MXFP8 with UE8M0 scales.

    Uses the shared quantization core in ``mxfp8_quantization_core``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run MXFP8 quantization.")

    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(input_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    num_hidden_layers = int(cfg["num_hidden_layers"])
    if is_source_block_fp8_ue8m0_checkpoint(cfg):
        source_is_block_fp8_ue8m0 = True
    elif is_bf16_source_checkpoint(cfg):
        source_is_block_fp8_ue8m0 = False
    else:
        raise ValueError(
            "Unsupported source quantization_config. "
            "Only BF16/FP16/FP32 sources and "
            "{quant_method=fp8, weight_block_size=[128, 128], scale_fmt=ue8m0} sources are supported."
        )

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(
            os.path.join(input_path, filename)
        ):
            shutil.copyfile(
                os.path.join(input_path, filename),
                os.path.join(output_path, filename),
            )

    index_path = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    safetensors_files = sorted(set(weight_map.values()))
    source_scale_index: dict[str, str] = {}
    if source_is_block_fp8_ue8m0:
        source_scale_index = {
            key: filename
            for key, filename in weight_map.items()
            if key.endswith(SOURCE_FP8_SCALE_KEY_SUFFIX)
        }

    result_collector = _ConversionResult()
    for filename in safetensors_files:
        logger.info(f"[mxfp8] Processing {filename}")
        _process_file(
            input_path,
            output_path,
            filename,
            result_collector=result_collector,
            device=device,
            num_hidden_layers=num_hidden_layers,
            num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
            source_is_block_fp8_ue8m0=source_is_block_fp8_ue8m0,
            extra_high_precision_layers_hf=extra_high_precision_layers_hf,
            source_scale_index=source_scale_index,
        )
        gc.collect()
        torch.cuda.empty_cache()

    quantization_config: dict[str, Any] = dict(MXFP8_QUANTIZATION_CONFIG)
    if len(result_collector.modules_to_not_convert) > 0:

        def natural_key(s: str) -> list[Any]:
            return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]

        quantization_config["modules_to_not_convert"] = sorted(
            list(set(result_collector.modules_to_not_convert)), key=natural_key
        )

    cfg["quantization_config"] = quantization_config
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    index_dict = {
        "weight_map": result_collector.weight_map,
        "metadata": {"total_size": result_collector.total_size},
    }
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_dict, f, indent=2)

    gc.collect()
    torch.cuda.empty_cache()


def _read_source_config(model_dir: str) -> dict[str, Any]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path) as f:
        return json.load(f)


def _quantization_fingerprint(quantization_cfg: dict[str, Any]) -> str:
    relevant_keys = (
        "extra_high_precision_layers_hf",
        "modules_to_not_convert",
        "num_layers_at_start_in_bf16",
        "num_layers_at_end_in_bf16",
        "weight_block_size",
        "scale_fmt",
    )
    payload = {k: quantization_cfg.get(k) for k in relevant_keys}
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def _hash_qualified_save_dir(
    *, model_dir: str, cache_root: str, quantization_cfg: dict[str, Any]
) -> str:
    abs_model = os.path.abspath(model_dir)
    src_cfg = _read_source_config(model_dir)
    src_fingerprint = hashlib.sha1(
        json.dumps(src_cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    quant_fingerprint = _quantization_fingerprint(quantization_cfg)
    payload = (
        f"{abs_model}|{src_fingerprint}|{quant_fingerprint}|v{CONVERTER_VERSION}"
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(os.path.normpath(abs_model)) or "hf"
    return os.path.join(os.path.abspath(cache_root), f"{base}-mxfp8-{digest}")


def is_existing_mxfp8_checkpoint(path: str) -> bool:
    cfg = _read_source_config(path)
    qcfg = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    return is_mxfp8_quantization_config(qcfg)


def ensure_mxfp8_checkpoint(
    *,
    model_path: str,
    quantization_cfg: dict[str, Any],
) -> str:
    """Return a path to an MXFP8-loadable HF checkpoint for SGLang.

    - If ``model_path`` is already an MXFP8 checkpoint, return it as-is.
    - If ``quantization_cfg.converted_model_path`` is an MXFP8 checkpoint,
      return it.
    - Otherwise convert ``model_path`` into a hash-qualified subdirectory
      under ``quantization_cfg.cache_root`` (or ``$NRL_MXFP8_CACHE`` /
      ``~/.cache/nemo_rl/mxfp8`` if not set) and return that path.

    The hash includes absolute model path, source config fingerprint,
    quantization config fingerprint and converter version, so different
    sources / settings never collide.
    """
    if is_existing_mxfp8_checkpoint(model_path):
        return model_path

    converted = quantization_cfg.get("converted_model_path")
    if converted and is_existing_mxfp8_checkpoint(converted):
        return converted

    cache_root = (
        quantization_cfg.get("cache_root")
        or os.environ.get("NRL_MXFP8_CACHE")
        or os.path.join(
            os.path.expanduser("~"), ".cache", "nemo_rl", "mxfp8"
        )
    )
    save_dir = converted or _hash_qualified_save_dir(
        model_dir=model_path,
        cache_root=cache_root,
        quantization_cfg=quantization_cfg,
    )

    if is_existing_mxfp8_checkpoint(save_dir):
        return save_dir

    extra_high_precision_layers_hf = tuple(
        quantization_cfg.get("extra_high_precision_layers_hf", ()) or ()
    )
    num_layers_at_start_in_bf16 = int(
        quantization_cfg.get("num_layers_at_start_in_bf16", 0) or 0
    )
    num_layers_at_end_in_bf16 = int(
        quantization_cfg.get("num_layers_at_end_in_bf16", 0) or 0
    )

    logger.info(
        f"[mxfp8] Converting {model_path} -> {save_dir} "
        f"(start_bf16={num_layers_at_start_in_bf16}, "
        f"end_bf16={num_layers_at_end_in_bf16}, "
        f"extra_hp={extra_high_precision_layers_hf})"
    )
    convert_mxfp8(
        model_dir=model_path,
        save_dir=save_dir,
        num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
        extra_high_precision_layers_hf=extra_high_precision_layers_hf,
    )
    return save_dir
