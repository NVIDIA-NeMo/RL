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
    MXFP8_SCALE_KEY_SUFFIX,
    is_bf16_source_checkpoint,
    is_mxfp8_quantization_config,
    quantize_mxfp8,
    should_quantize,
    strip_weight_suffix,
)
from nemo_rl.models.generation.sglang.quantization_utils import (
    build_dynamic_skip_substrings,
    expand_sglang_atomic_high_precision_substrings,
    get_dynamic_high_precision_substrings,
    validate_checkpoint_high_precision_layout,
)

logger = logging.getLogger(__name__)

CONVERTER_VERSION: str = "4"


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


def _process_file(
    input_path: str,
    output_path: str,
    filename: str,
    *,
    result_collector: _ConversionResult,
    device: str,
    skip_weight_substrings: tuple[str, ...],
) -> None:
    import safetensors
    import safetensors.torch

    weights: dict[str, torch.Tensor] = {}
    q_weights: dict[str, torch.Tensor] = {}

    with safetensors.safe_open(
        os.path.join(input_path, filename), framework="pt", device=device
    ) as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    modules_to_not_convert: list[str] = []

    for key, tensor in weights.items():
        if not key.endswith(".weight"):
            continue

        if should_quantize(key, tensor, skip_weight_substrings=skip_weight_substrings):
            qweight, scale = quantize_mxfp8(tensor)
            q_weights[key] = qweight
            q_weights[strip_weight_suffix(key) + MXFP8_SCALE_KEY_SUFFIX] = scale
        else:
            if ".experts." not in key:
                modules_to_not_convert.append(strip_weight_suffix(key))
            q_weights[key] = tensor

    for key, tensor in weights.items():
        if not key.endswith(".weight"):
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
    modules_to_not_convert: tuple[str, ...] = (),
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
    num_hidden_layers = _get_num_hidden_layers(cfg, config_path=config_path)
    selection_config = {
        "extra_high_precision_layers_hf": extra_high_precision_layers_hf,
        "modules_to_not_convert": modules_to_not_convert,
        "num_layers_at_start_in_bf16": num_layers_at_start_in_bf16,
        "num_layers_at_end_in_bf16": num_layers_at_end_in_bf16,
    }
    configured_high_precision_substrings = get_dynamic_high_precision_substrings(
        quantization_config=selection_config,
        num_hidden_layers=num_hidden_layers,
    )
    skip_weight_substrings = build_dynamic_skip_substrings(
        quantization_config=selection_config,
        num_hidden_layers=num_hidden_layers,
        static_skip_substrings=SKIP_WEIGHT_SUBSTRINGS,
    )
    if not is_bf16_source_checkpoint(cfg):
        raise ValueError(
            "MXFP8 conversion only supports BF16/FP16/FP32 source checkpoints. "
            "Both cold start and weight refit go through the Megatron bridge, "
            "which always hands over high-precision HF tensors."
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
    expanded_skip_weight_substrings = expand_sglang_atomic_high_precision_substrings(
        weight_names=weight_map,
        skip_weight_substrings=skip_weight_substrings,
    )
    concrete_atomic_modules = tuple(
        substring
        for substring in expanded_skip_weight_substrings
        if substring not in skip_weight_substrings
    )
    skip_weight_substrings = expanded_skip_weight_substrings
    safetensors_files = sorted(set(weight_map.values()))

    result_collector = _ConversionResult()
    result_collector.modules_to_not_convert.extend(
        (*configured_high_precision_substrings, *concrete_atomic_modules)
    )
    for filename in safetensors_files:
        logger.info(f"[mxfp8] Processing {filename}")
        _process_file(
            input_path,
            output_path,
            filename,
            result_collector=result_collector,
            device=device,
            skip_weight_substrings=skip_weight_substrings,
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


def _checkpoint_weight_names(model_dir: str) -> tuple[str, ...]:
    """Read tensor names from an indexed or single-file safetensors checkpoint."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as file:
            index = json.load(file)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Missing non-empty weight_map in {index_path}.")
        if any(
            not isinstance(name, str) or not isinstance(filename, str)
            for name, filename in weight_map.items()
        ):
            raise ValueError(f"{index_path} weight_map must map strings to strings.")
        return tuple(weight_map)

    import safetensors

    names: set[str] = set()
    shard_names = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
        and os.path.isfile(os.path.join(model_dir, filename))
    )
    if not shard_names:
        raise ValueError(f"No safetensors weights found in {model_dir}.")
    for filename in shard_names:
        with safetensors.safe_open(
            os.path.join(model_dir, filename),
            framework="pt",
            device="cpu",
        ) as file:
            for name in file.keys():
                if name in names:
                    raise ValueError(
                        f"Duplicate tensor {name!r} in checkpoint {model_dir!r}."
                    )
                names.add(name)
    return tuple(names)


def _get_num_hidden_layers(cfg: dict[str, Any], *, config_path: str) -> int:
    num_hidden_layers = cfg.get("num_hidden_layers")
    text_config = cfg.get("text_config")
    if num_hidden_layers is None and isinstance(text_config, dict):
        num_hidden_layers = text_config.get("num_hidden_layers")
    if (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
    ):
        raise ValueError(
            f"{config_path} must define a positive integer num_hidden_layers."
        )
    return num_hidden_layers


def _validated_conversion_options(
    quantization_cfg: dict[str, Any],
    *,
    num_hidden_layers: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    """Validate and normalize the shared offline/online selection options."""
    get_dynamic_high_precision_substrings(
        quantization_config=quantization_cfg,
        num_hidden_layers=num_hidden_layers,
    )

    extra_value = quantization_cfg.get("extra_high_precision_layers_hf")
    modules_value = quantization_cfg.get("modules_to_not_convert")
    start_value = quantization_cfg.get("num_layers_at_start_in_bf16")
    end_value = quantization_cfg.get("num_layers_at_end_in_bf16")
    return (
        () if extra_value is None else tuple(item.strip() for item in extra_value),
        () if modules_value is None else tuple(item.strip() for item in modules_value),
        0 if start_value is None else start_value,
        0 if end_value is None else end_value,
    )


def _merge_checkpoint_modules_to_not_convert(
    *,
    checkpoint_path: str,
    quantization_cfg: dict[str, Any],
) -> None:
    """Merge the checkpoint's concrete ignore list into the refit config."""
    checkpoint_cfg = _read_source_config(checkpoint_path)
    checkpoint_quantization_cfg = checkpoint_cfg.get("quantization_config")
    if not isinstance(checkpoint_quantization_cfg, dict):
        raise ValueError(f"{checkpoint_path}/config.json has no quantization_config.")

    num_hidden_layers = _get_num_hidden_layers(
        checkpoint_cfg,
        config_path=os.path.join(checkpoint_path, "config.json"),
    )
    _, user_modules, _, _ = _validated_conversion_options(
        quantization_cfg,
        num_hidden_layers=num_hidden_layers,
    )
    requested_high_precision = get_dynamic_high_precision_substrings(
        quantization_config=quantization_cfg,
        num_hidden_layers=num_hidden_layers,
    )
    checkpoint_modules = get_dynamic_high_precision_substrings(
        quantization_config={
            "modules_to_not_convert": checkpoint_quantization_cfg.get(
                "modules_to_not_convert"
            )
        },
        num_hidden_layers=num_hidden_layers,
    )
    checkpoint_weight_names = _checkpoint_weight_names(checkpoint_path)
    configured_high_precision = tuple(
        dict.fromkeys((*requested_high_precision, *checkpoint_modules))
    )
    expanded_high_precision = expand_sglang_atomic_high_precision_substrings(
        weight_names=checkpoint_weight_names,
        skip_weight_substrings=configured_high_precision,
    )
    validate_checkpoint_high_precision_layout(
        checkpoint_path=checkpoint_path,
        scheme="MXFP8",
        weight_names=checkpoint_weight_names,
        high_precision_substrings=expanded_high_precision,
        quantized_companion_suffixes=(MXFP8_SCALE_KEY_SUFFIX,),
    )
    concrete_atomic_modules = tuple(
        substring
        for substring in expanded_high_precision
        if substring not in configured_high_precision
    )
    quantization_cfg["modules_to_not_convert"] = list(
        dict.fromkeys((*user_modules, *checkpoint_modules, *concrete_atomic_modules))
    )


def _quantization_fingerprint(quantization_cfg: dict[str, Any]) -> str:
    relevant_keys = (
        "extra_high_precision_layers_hf",
        "modules_to_not_convert",
        "num_layers_at_start_in_bf16",
        "num_layers_at_end_in_bf16",
        # Legacy fingerprint inputs. No longer settable via
        # SglangQuantizationConfig; kept so existing cache dirs stay valid.
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
    payload = f"{abs_model}|{src_fingerprint}|{quant_fingerprint}|v{CONVERTER_VERSION}"
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
    """Return an MXFP8 checkpoint path and synchronize its ignore policy."""
    if is_existing_mxfp8_checkpoint(model_path):
        _merge_checkpoint_modules_to_not_convert(
            checkpoint_path=model_path,
            quantization_cfg=quantization_cfg,
        )
        return model_path

    converted = quantization_cfg.get("converted_model_path")
    if converted and is_existing_mxfp8_checkpoint(converted):
        _merge_checkpoint_modules_to_not_convert(
            checkpoint_path=converted,
            quantization_cfg=quantization_cfg,
        )
        return converted

    cache_root = (
        quantization_cfg.get("cache_root")
        or os.environ.get("NRL_MXFP8_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "nemo_rl", "mxfp8")
    )
    save_dir = converted or _hash_qualified_save_dir(
        model_dir=model_path,
        cache_root=cache_root,
        quantization_cfg=quantization_cfg,
    )

    if is_existing_mxfp8_checkpoint(save_dir):
        _merge_checkpoint_modules_to_not_convert(
            checkpoint_path=save_dir,
            quantization_cfg=quantization_cfg,
        )
        return save_dir

    source_cfg = _read_source_config(model_path)
    num_hidden_layers = _get_num_hidden_layers(
        source_cfg,
        config_path=os.path.join(model_path, "config.json"),
    )
    (
        extra_high_precision_layers_hf,
        modules_to_not_convert,
        num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16,
    ) = _validated_conversion_options(
        quantization_cfg,
        num_hidden_layers=num_hidden_layers,
    )

    logger.info(
        f"[mxfp8] Converting {model_path} -> {save_dir} "
        f"(start_bf16={num_layers_at_start_in_bf16}, "
        f"end_bf16={num_layers_at_end_in_bf16}, "
        f"extra_hp={extra_high_precision_layers_hf}, "
        f"modules_to_not_convert={modules_to_not_convert})"
    )
    convert_mxfp8(
        model_dir=model_path,
        save_dir=save_dir,
        num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
        extra_high_precision_layers_hf=extra_high_precision_layers_hf,
        modules_to_not_convert=modules_to_not_convert,
    )
    _merge_checkpoint_modules_to_not_convert(
        checkpoint_path=save_dir,
        quantization_cfg=quantization_cfg,
    )
    return save_dir
