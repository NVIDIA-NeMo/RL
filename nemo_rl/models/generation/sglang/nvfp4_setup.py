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

"""Offline HF-to-NVFP4 conversion and SGLang startup checkpoint setup."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Mapping
from typing import Any

import torch

from nemo_rl.models.generation.sglang.config import SglangQuantizationConfig
from nemo_rl.models.generation.sglang.nvfp4_quantization_core import (
    NVFP4_GROUP_SIZE,
    NVFP4_QUANTIZATION_CONFIG,
    is_bf16_source_checkpoint,
    is_moe_expert_weight_name,
    is_nvfp4_quantization_config,
    nvfp4_quantized_entries,
    quantize_nvfp4,
    quantize_nvfp4_pair,
    should_quantize_nvfp4,
    split_gated_pair_name,
    strip_weight_suffix,
)
from nemo_rl.models.generation.sglang.quantization_utils import (
    build_dynamic_skip_substrings,
    expand_sglang_atomic_high_precision_substrings,
    get_dynamic_high_precision_substrings,
    get_hf_moe_expert_container,
    validate_checkpoint_high_precision_layout,
)

logger = logging.getLogger(__name__)

# v2 stamps the NeMo-RL MoE-only layout marker required for cache reuse.
CONVERTER_VERSION = "2"
NEMO_RL_NVFP4_LAYOUT = "sglang_moe_only_v1"
NEMO_RL_NVFP4_LAYOUT_KEY = "nemo_rl_nvfp4_layout"
_NVFP4_ENV_KEYS = (
    "NVTE_NVFP4_4OVER6",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256",
    "NVTE_NVFP4_4OVER6_ERR_MODE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH",
)


class _ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        self.modules_to_not_convert: set[str] = set()

    def add_tensor(self, filename: str, key: str, tensor: torch.Tensor) -> None:
        existing = self.weight_map.get(key)
        if existing is not None:
            raise ValueError(
                f"Duplicate output tensor {key!r} in {existing!r} and {filename!r}."
            )
        self.weight_map[key] = filename
        self.total_size += tensor.numel() * tensor.element_size()


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_source_config(model_dir: str) -> dict[str, Any]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}
    return _read_json(config_path)


def _read_external_nvfp4_quantization_config(
    model_dir: str,
) -> dict[str, Any] | None:
    """Read NVFP4 metadata from ModelOpt's companion file."""
    hf_quant_path = os.path.join(model_dir, "hf_quant_config.json")
    if not os.path.isfile(hf_quant_path):
        return None
    hf_quant = _read_json(hf_quant_path)
    return next(
        (
            candidate
            for candidate in (
                hf_quant.get("quantization"),
                hf_quant.get("quantization_config"),
                hf_quant,
            )
            if is_nvfp4_quantization_config(candidate)
        ),
        None,
    )


def _num_hidden_layers(config: dict[str, Any]) -> int:
    value = config.get("num_hidden_layers")
    if value is None and isinstance(config.get("text_config"), dict):
        value = config["text_config"].get("num_hidden_layers")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "The source config must define a positive num_hidden_layers, either "
            "at the top level or under text_config."
        )
    return value


def _source_weight_map(model_dir: str) -> dict[str, str]:
    """Return a validated tensor-to-shard map for indexed or single-file HF models."""
    import safetensors

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        index = _read_json(index_path)
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError(f"Missing non-empty weight_map in {index_path}.")
        weight_map = {
            str(key): str(filename) for key, filename in raw_weight_map.items()
        }
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = sorted(
            filename
            for filename in os.listdir(model_dir)
            if filename.endswith(".safetensors")
            and os.path.isfile(os.path.join(model_dir, filename))
        )
        if not shard_names:
            raise ValueError(f"No safetensors weights found in {model_dir}.")
        weight_map: dict[str, str] = {}

    actual_weight_map: dict[str, str] = {}
    for filename in shard_names:
        shard_path = os.path.join(model_dir, filename)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Checkpoint shard not found: {shard_path}")
        with safetensors.safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as file:
            for key in file.keys():
                existing = actual_weight_map.get(key)
                if existing is not None:
                    raise ValueError(
                        f"Duplicate source tensor {key!r} in {existing!r} and "
                        f"{filename!r}."
                    )
                actual_weight_map[key] = filename

    if os.path.isfile(index_path):
        if weight_map != actual_weight_map:
            missing = sorted(set(weight_map) - set(actual_weight_map))
            unindexed = sorted(set(actual_weight_map) - set(weight_map))
            misplaced = sorted(
                key
                for key in set(weight_map) & set(actual_weight_map)
                if weight_map[key] != actual_weight_map[key]
            )
            raise ValueError(
                "The safetensors index does not match the checkpoint shards: "
                f"missing={missing[:5]}, unindexed={unindexed[:5]}, "
                f"misplaced={misplaced[:5]}."
            )
        return weight_map
    return actual_weight_map


def _natural_key(value: str) -> list[Any]:
    return [
        int(token) if token.isdigit() else token
        for token in re.findall(r"\d+|\D+", value)
    ]


def _expanded_ignore_modules(modules: set[str]) -> list[str]:
    expanded = set(modules)
    for module in tuple(expanded):
        if module.endswith((".q_proj", ".k_proj", ".v_proj")):
            expanded.add(module.rsplit(".", 1)[0] + ".qkv_proj")
        container = get_hf_moe_expert_container(module + ".weight")
        if container is not None:
            expanded.add(container)

    return sorted(expanded, key=_natural_key)


def _build_conversion_policy(
    *,
    weight_map: dict[str, str],
    skip_weight_substrings: tuple[str, ...],
) -> tuple[
    set[str],
    set[str],
    dict[str, tuple[str, str]],
]:
    """Select quantized expert tensors and complete gate/up pairs atomically."""
    skipped_containers: set[str] = set()
    expert_keys_by_container: dict[str, list[str]] = {}

    for key in weight_map:
        if not key.endswith(".weight"):
            continue
        container = get_hf_moe_expert_container(key)
        if container is None:
            continue
        expert_keys_by_container.setdefault(container, []).append(key)
        if not is_moe_expert_weight_name(key) or any(
            pattern in key for pattern in skip_weight_substrings
        ):
            skipped_containers.add(container)

    quantized_keys = {
        key
        for container, keys in expert_keys_by_container.items()
        if container not in skipped_containers
        for key in keys
    }

    pair_roles: dict[str, dict[str, str]] = {}
    for key in sorted(quantized_keys):
        pair_base, pair_role = split_gated_pair_name(key)
        if pair_base is None or pair_role is None:
            continue
        roles = pair_roles.setdefault(pair_base, {})
        if pair_role in roles:
            raise ValueError(
                f"Duplicate NVFP4 {pair_role} weight for pair {pair_base!r}: "
                f"{roles[pair_role]!r}, {key!r}."
            )
        roles[pair_role] = key

    incomplete = {
        base: sorted(roles)
        for base, roles in pair_roles.items()
        if set(roles) != {"gate", "up"}
    }
    if incomplete:
        raise ValueError(
            "NVFP4 gate/up weights must be converted together; incomplete "
            f"checkpoint pairs: {incomplete}."
        )

    pair_by_key: dict[str, tuple[str, str]] = {}
    for base, roles in pair_roles.items():
        gate_key = roles["gate"]
        up_key = roles["up"]
        pair_by_key[gate_key] = (base, up_key)
        pair_by_key[up_key] = (base, gate_key)
    return quantized_keys, skipped_containers, pair_by_key


def _load_tensor(
    model_dir: str,
    filename: str,
    key: str,
    *,
    device: str,
) -> torch.Tensor:
    import safetensors

    with safetensors.safe_open(
        os.path.join(model_dir, filename),
        framework="pt",
        device=device,
    ) as file:
        return file.get_tensor(key)


def _add_output_entries(
    destination: dict[str, torch.Tensor],
    entries: list[tuple[str, torch.Tensor]],
) -> None:
    for key, tensor in entries:
        if key in destination:
            raise ValueError(f"Duplicate NVFP4 output tensor {key!r}.")
        destination[key] = tensor


def _convert_shards(
    *,
    model_dir: str,
    save_dir: str,
    weight_map: dict[str, str],
    quantized_keys: set[str],
    pair_by_key: dict[str, tuple[str, str]],
    skipped_containers: set[str],
    device: str,
) -> _ConversionResult:
    import safetensors
    import safetensors.torch

    result = _ConversionResult()
    result.modules_to_not_convert.update(skipped_containers)
    deferred: dict[str, dict[str, torch.Tensor]] = {}
    processed_pairs: set[str] = set()
    processed_source_keys: set[str] = set()

    for filename in sorted(set(weight_map.values())):
        logger.info("[nvfp4] Processing %s", filename)
        shard_path = os.path.join(model_dir, filename)
        with safetensors.safe_open(
            shard_path,
            framework="pt",
            device=device,
        ) as file:
            weights = {key: file.get_tensor(key) for key in file.keys()}

        output = deferred.pop(filename, {})
        for key, tensor in weights.items():
            if key in processed_source_keys:
                continue

            if key in quantized_keys:
                if not should_quantize_nvfp4(
                    key,
                    tensor,
                    skip_weight_substrings=(),
                ):
                    raise ValueError(
                        f"Expert weight {key!r} is not a supported NVFP4 source "
                        f"(dtype={tensor.dtype}, shape={tuple(tensor.shape)})."
                    )

                pair_info = pair_by_key.get(key)
                if pair_info is None:
                    _add_output_entries(
                        output,
                        nvfp4_quantized_entries(
                            key,
                            quantize_nvfp4(tensor),
                            include_input_scale=True,
                        ),
                    )
                    processed_source_keys.add(key)
                    continue

                pair_base, counterpart_key = pair_info
                if pair_base in processed_pairs:
                    processed_source_keys.add(key)
                    continue
                counterpart_filename = weight_map[counterpart_key]
                counterpart = (
                    weights[counterpart_key]
                    if counterpart_filename == filename
                    else _load_tensor(
                        model_dir,
                        counterpart_filename,
                        counterpart_key,
                        device=device,
                    )
                )
                if not should_quantize_nvfp4(
                    counterpart_key,
                    counterpart,
                    skip_weight_substrings=(),
                ):
                    raise ValueError(
                        f"Expert weight {counterpart_key!r} is not a supported "
                        f"NVFP4 source (dtype={counterpart.dtype}, "
                        f"shape={tuple(counterpart.shape)})."
                    )

                current_base, current_role = split_gated_pair_name(key)
                if current_base != pair_base or current_role is None:
                    raise RuntimeError(f"Invalid cached NVFP4 pair for {key!r}.")
                if current_role == "gate":
                    gate_key, gate_weight = key, tensor
                    up_key, up_weight = counterpart_key, counterpart
                else:
                    gate_key, gate_weight = counterpart_key, counterpart
                    up_key, up_weight = key, tensor

                gate_output, up_output = quantize_nvfp4_pair(
                    gate_weight,
                    up_weight,
                )
                pair_entries = (
                    (
                        gate_key,
                        nvfp4_quantized_entries(
                            gate_key,
                            gate_output,
                            include_input_scale=True,
                        ),
                    ),
                    (
                        up_key,
                        nvfp4_quantized_entries(
                            up_key,
                            up_output,
                            include_input_scale=True,
                        ),
                    ),
                )
                for source_key, entries in pair_entries:
                    target_filename = weight_map[source_key]
                    target = (
                        output
                        if target_filename == filename
                        else deferred.setdefault(target_filename, {})
                    )
                    _add_output_entries(target, entries)

                processed_pairs.add(pair_base)
                processed_source_keys.update((gate_key, up_key))
                continue

            output[key] = tensor
            processed_source_keys.add(key)
            if key.endswith(".weight"):
                result.modules_to_not_convert.add(strip_weight_suffix(key))

        output_path = os.path.join(save_dir, filename)
        safetensors.torch.save_file(
            output,
            output_path,
            metadata={"format": "pt"},
        )
        for key, tensor in output.items():
            result.add_tensor(filename, key, tensor)

        del output, weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if deferred:
        raise RuntimeError(
            "Deferred NVFP4 pair outputs were not written because their target "
            f"shards were missing: {sorted(deferred)}."
        )
    if processed_source_keys != set(weight_map):
        missing = sorted(set(weight_map) - processed_source_keys)
        raise RuntimeError(
            f"NVFP4 conversion did not process source keys: {missing[:10]}."
        )
    return result


def _copy_checkpoint_metadata(model_dir: str, save_dir: str) -> None:
    for filename in os.listdir(model_dir):
        source = os.path.join(model_dir, filename)
        if os.path.isdir(source) or filename.endswith(".safetensors"):
            continue
        shutil.copyfile(source, os.path.join(save_dir, filename))


def _write_output_metadata(
    *,
    model_dir: str,
    save_dir: str,
    config: dict[str, Any],
    result: _ConversionResult,
) -> list[str]:
    ignore = _expanded_ignore_modules(result.modules_to_not_convert)
    quantization_config = dict(NVFP4_QUANTIZATION_CONFIG)
    source_quantization = config.get("quantization_config")
    if isinstance(source_quantization, dict):
        kv_cache_scheme = source_quantization.get("kv_cache_scheme")
        if isinstance(kv_cache_scheme, dict):
            quantization_config["kv_cache_scheme"] = kv_cache_scheme
    quantization_config["ignore"] = ignore
    config["quantization_config"] = quantization_config
    config[NEMO_RL_NVFP4_LAYOUT_KEY] = NEMO_RL_NVFP4_LAYOUT

    with open(os.path.join(save_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=2)

    source_hf_quant_path = os.path.join(model_dir, "hf_quant_config.json")
    hf_quant_config = (
        _read_json(source_hf_quant_path)
        if os.path.isfile(source_hf_quant_path)
        else {"producer": {"name": "modelopt", "version": "nemo-rl"}}
    )
    raw_hf_quantization = hf_quant_config.get("quantization")
    hf_quantization: dict[str, Any] = (
        dict(raw_hf_quantization) if isinstance(raw_hf_quantization, dict) else {}
    )
    hf_quant_config["quantization"] = hf_quantization
    hf_quantization.update(
        {
            "exclude_modules": ignore,
            "group_size": NVFP4_GROUP_SIZE,
            "quant_algo": "NVFP4",
        }
    )
    output_kv_cache_scheme = quantization_config.get("kv_cache_scheme")
    if (
        isinstance(output_kv_cache_scheme, dict)
        and output_kv_cache_scheme.get("type") == "float"
        and output_kv_cache_scheme.get("num_bits") == 8
    ):
        hf_quantization["kv_cache_quant_algo"] = "FP8"
    else:
        hf_quantization.pop("kv_cache_quant_algo", None)
    with open(os.path.join(save_dir, "hf_quant_config.json"), "w") as file:
        json.dump(hf_quant_config, file, indent=2)

    index = {
        "weight_map": result.weight_map,
        "metadata": {"total_size": result.total_size},
    }
    with open(
        os.path.join(save_dir, "model.safetensors.index.json"),
        "w",
    ) as file:
        json.dump(index, file, indent=2)
    return ignore


def convert_nvfp4(
    model_dir: str,
    save_dir: str,
    *,
    device: str = "cuda",
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    extra_high_precision_layers_hf: tuple[str, ...] = (),
    modules_to_not_convert: tuple[str, ...] = (),
) -> list[str]:
    """Convert an HF checkpoint to ModelOpt NVFP4 for SGLang.

    Megatron name conversion is intentionally absent: both this converter and
    the live refit iterator operate on finalized HF tensor names.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run NVFP4 quantization.")

    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    config_path = os.path.join(input_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"HF config not found: {config_path}")
    config = _read_json(config_path)
    if not is_bf16_source_checkpoint(config):
        raise ValueError(
            "NVFP4 conversion only supports BF16/FP16/FP32 source checkpoints."
        )

    num_hidden_layers = _num_hidden_layers(config)
    policy_config = {
        "extra_high_precision_layers_hf": list(extra_high_precision_layers_hf),
        "modules_to_not_convert": list(modules_to_not_convert),
        "num_layers_at_start_in_bf16": num_layers_at_start_in_bf16,
        "num_layers_at_end_in_bf16": num_layers_at_end_in_bf16,
    }
    skip_weight_substrings = build_dynamic_skip_substrings(
        quantization_config=policy_config,
        num_hidden_layers=num_hidden_layers,
    )
    weight_map = _source_weight_map(input_path)
    quantized_keys, skipped_containers, pair_by_key = _build_conversion_policy(
        weight_map=weight_map,
        skip_weight_substrings=skip_weight_substrings,
    )
    if not quantized_keys:
        raise ValueError(
            "No eligible MoE expert weights remain after applying NVFP4 skip rules."
        )

    os.makedirs(output_path, exist_ok=True)
    _copy_checkpoint_metadata(input_path, output_path)
    result = _convert_shards(
        model_dir=input_path,
        save_dir=output_path,
        weight_map=weight_map,
        quantized_keys=quantized_keys,
        pair_by_key=pair_by_key,
        skipped_containers=skipped_containers,
        device=device,
    )
    ignore = _write_output_metadata(
        model_dir=input_path,
        save_dir=output_path,
        config=config,
        result=result,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ignore


def _quantization_fingerprint(quantization_cfg: Mapping[str, Any]) -> str:
    relevant_keys = (
        "extra_high_precision_layers_hf",
        "modules_to_not_convert",
        "num_layers_at_start_in_bf16",
        "num_layers_at_end_in_bf16",
    )
    payload = {
        "config": {key: quantization_cfg.get(key) for key in relevant_keys},
        "environment": {key: os.environ.get(key) for key in _NVFP4_ENV_KEYS},
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def _validated_conversion_options(
    quantization_cfg: Mapping[str, Any],
    *,
    num_hidden_layers: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    """Validate and normalize the shared offline/online selection options."""
    build_dynamic_skip_substrings(
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


def _hash_qualified_save_dir(
    *,
    model_dir: str,
    cache_root: str,
    quantization_cfg: Mapping[str, Any],
) -> str:
    absolute_model = os.path.abspath(model_dir)
    source_config = _read_source_config(model_dir)
    source_fingerprint = hashlib.sha1(
        json.dumps(source_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    quantization_fingerprint = _quantization_fingerprint(quantization_cfg)
    payload = (
        f"{absolute_model}|{source_fingerprint}|{quantization_fingerprint}|"
        f"v{CONVERTER_VERSION}"
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(os.path.normpath(absolute_model)) or "hf"
    return os.path.join(os.path.abspath(cache_root), f"{base}-nvfp4-{digest}")


def is_existing_nvfp4_checkpoint(path: str) -> bool:
    config = _read_source_config(path)
    quantization_config = config.get("quantization_config")
    return (
        is_nvfp4_quantization_config(quantization_config)
        and config.get(NEMO_RL_NVFP4_LAYOUT_KEY) == NEMO_RL_NVFP4_LAYOUT
    )


def _reject_unsupported_external_nvfp4_checkpoint(path: str) -> None:
    config = _read_source_config(path)
    if (
        not is_nvfp4_quantization_config(config.get("quantization_config"))
        and _read_external_nvfp4_quantization_config(path) is None
    ):
        return
    raise ValueError(
        f"Externally quantized NVFP4 checkpoint at {path} is not supported for "
        "SGLang online refit because its tensor layout may differ from NeMo-RL's "
        "MoE-only refit layout. Use a BF16 source checkpoint or a checkpoint "
        "converted by NeMo-RL."
    )


def _sync_checkpoint_ignore(
    checkpoint_path: str,
    quantization_cfg: SglangQuantizationConfig,
) -> None:
    config = _read_source_config(checkpoint_path)
    checkpoint_quantization = config.get("quantization_config")
    if not isinstance(checkpoint_quantization, dict):
        return
    ignore = checkpoint_quantization.get("ignore")
    if ignore is None:
        ignore = []
    if not isinstance(ignore, list):
        raise ValueError(
            f"{checkpoint_path}/config.json quantization_config.ignore must "
            "be a list of strings."
        )
    if any(not isinstance(item, str) or not item.strip() for item in ignore):
        raise ValueError(
            f"{checkpoint_path}/config.json quantization_config.ignore must "
            "contain non-empty strings."
        )
    num_hidden_layers = _num_hidden_layers(config)
    _, configured, _, _ = _validated_conversion_options(
        quantization_cfg,
        num_hidden_layers=num_hidden_layers,
    )
    requested_high_precision = get_dynamic_high_precision_substrings(
        quantization_config=quantization_cfg,
        num_hidden_layers=num_hidden_layers,
    )
    checkpoint_weight_names = tuple(_source_weight_map(checkpoint_path))
    configured_high_precision = tuple(
        dict.fromkeys((*requested_high_precision, *(str(item) for item in ignore)))
    )
    expanded_high_precision = expand_sglang_atomic_high_precision_substrings(
        weight_names=checkpoint_weight_names,
        skip_weight_substrings=configured_high_precision,
    )
    validate_checkpoint_high_precision_layout(
        checkpoint_path=checkpoint_path,
        scheme="NVFP4",
        weight_names=checkpoint_weight_names,
        high_precision_substrings=expanded_high_precision,
        quantized_companion_suffixes=(".weight_scale", ".weight_scale_2"),
    )
    concrete_atomic_modules = tuple(
        substring
        for substring in expanded_high_precision
        if substring not in configured_high_precision
    )
    merged = list(
        dict.fromkeys(
            [
                *(str(item) for item in configured),
                *(str(item) for item in ignore),
                *concrete_atomic_modules,
            ]
        )
    )
    quantization_cfg["modules_to_not_convert"] = merged


def ensure_nvfp4_checkpoint(
    *,
    model_path: str,
    quantization_cfg: SglangQuantizationConfig,
) -> str:
    """Return a local preconverted NVFP4 checkpoint with synchronized skips.

    Startup deliberately does not convert in the Ray driver: its base venv has
    no Transformer Engine and the process is not guaranteed a GPU.
    ``convert_nvfp4`` remains the explicit offline GPU conversion entry point.
    """
    local_model_path = os.path.abspath(os.path.expanduser(model_path))
    if os.path.isdir(local_model_path) and is_existing_nvfp4_checkpoint(
        local_model_path
    ):
        _sync_checkpoint_ignore(local_model_path, quantization_cfg)
        return local_model_path
    _reject_unsupported_external_nvfp4_checkpoint(local_model_path)

    converted = quantization_cfg.get("converted_model_path")
    if converted is not None:
        if not isinstance(converted, str) or not converted.strip():
            raise ValueError(
                "SGLang quantization.converted_model_path must be a non-empty "
                "local directory path."
            )
        converted_path = os.path.abspath(os.path.expanduser(converted))
        if not os.path.isdir(converted_path):
            raise ValueError(
                "SGLang NVFP4 startup requires a local preconverted checkpoint, "
                "but quantization.converted_model_path is not a directory: "
                f"{converted!r}."
            )
        if not is_existing_nvfp4_checkpoint(converted_path):
            _reject_unsupported_external_nvfp4_checkpoint(converted_path)
            raise ValueError(
                "SGLang NVFP4 startup requires a checkpoint converted by "
                "NeMo-RL, but quantization.converted_model_path does not declare "
                "its supported MoE-only NVFP4 layout marker in config.json: "
                f"{converted_path!r}."
            )
        _sync_checkpoint_ignore(converted_path, quantization_cfg)
        return converted_path

    if not os.path.isdir(local_model_path):
        raise ValueError(
            "SGLang NVFP4 startup cannot auto-convert an HF repo id in the Ray "
            f"driver: {model_path!r}. Download and convert the checkpoint offline "
            "under the mcore environment on a GPU, then set "
            "quantization.converted_model_path to a local directory visible to "
            "every rollout worker."
        )

    cache_root = (
        quantization_cfg.get("cache_root")
        or os.environ.get("NRL_NVFP4_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "nemo_rl", "nvfp4")
    )
    save_dir = _hash_qualified_save_dir(
        model_dir=local_model_path,
        cache_root=cache_root,
        quantization_cfg=quantization_cfg,
    )
    if is_existing_nvfp4_checkpoint(save_dir):
        _sync_checkpoint_ignore(save_dir, quantization_cfg)
        return save_dir
    _reject_unsupported_external_nvfp4_checkpoint(save_dir)

    raise ValueError(
        "SGLang NVFP4 startup does not quantize in the Ray driver because its "
        "base environment has no Transformer Engine and may have no assigned "
        "GPU. Run convert_nvfp4 offline under the mcore environment on a GPU, "
        "then set quantization.converted_model_path to the output directory. "
        f"The deterministic cache path for this source and policy is {save_dir!r}."
    )
