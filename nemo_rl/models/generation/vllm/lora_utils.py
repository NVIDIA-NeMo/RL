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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

LoraRefitMode = Literal["native", "merged"]

NATIVE_LORA_ADAPTER_ID = 1
NATIVE_LORA_ADAPTER_NAME = "nemo-rl-online"
NATIVE_LORA_ADAPTER_PATH = "/nemo-rl/in-memory-online-lora"
NATIVE_LORA_CONFIG_KEY = "nemo_rl_native_lora"


@dataclass(frozen=True)
class NativeLoraRefitSettings:
    """Internal native-LoRA settings forwarded to each vLLM worker."""

    rank: int
    alpha: int


def native_lora_refit_settings(vllm_config: Any) -> NativeLoraRefitSettings | None:
    """Read native-LoRA settings from a realized vLLM worker config."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return None
    raw_settings = additional_config.get(NATIVE_LORA_CONFIG_KEY)
    if raw_settings is None:
        return None
    return NativeLoraRefitSettings(
        rank=int(raw_settings["rank"]),
        alpha=int(raw_settings["alpha"]),
    )


def validate_vllm_lora_refit(policy_config: Mapping[str, Any]) -> None:
    """Validate vLLM-specific requirements for a native LoRA refit."""
    generation_config = policy_config["generation"]
    if generation_config["backend"] != "vllm":
        raise ValueError("Native LoRA refit is currently supported only by vLLM.")

    if generation_config.get("refit_transport") is not None:
        raise ValueError(
            "Native LoRA refit currently supports only the topology-default "
            "collective/non-colocated or CUDA-IPC/colocated transport. Set "
            "policy.generation.refit_transport=null, or opt into merged refit."
        )
    if generation_config["vllm_cfg"]["async_engine"]:
        raise ValueError(
            "Native LoRA refit does not yet support vllm_cfg.async_engine=true."
        )
    if generation_config["vllm_cfg"].get("expose_http_server"):
        raise ValueError(
            "Native LoRA refit does not yet support vLLM's exposed HTTP server "
            "because every request must carry the in-memory LoRARequest."
        )
    if generation_config.get("real_quant") or generation_config.get("quant_cfg"):
        raise ValueError(
            "Native LoRA refit does not yet support quantized rollout models."
        )

    vllm_kwargs = generation_config.get("vllm_kwargs") or {}
    speculative_config = vllm_kwargs.get("speculative_config")
    speculative_decoding_enabled = speculative_config is not None and not (
        isinstance(speculative_config, Mapping)
        and int(speculative_config.get("num_speculative_tokens", 0)) == 0
    )
    if speculative_decoding_enabled:
        raise ValueError("Native LoRA refit does not yet support speculative decoding.")
    if vllm_kwargs.get("enable_lora") is False:
        raise ValueError(
            "Native LoRA refit requires policy.generation.vllm_kwargs.enable_lora=true."
        )

    lora_config = policy_config["automodel_cfg"]["lora_cfg"]
    rank = int(lora_config["dim"])
    max_lora_rank = vllm_kwargs.get("max_lora_rank")
    if max_lora_rank is not None and int(max_lora_rank) < rank:
        raise ValueError(
            f"vllm_kwargs.max_lora_rank={max_lora_rank} is smaller than the "
            f"trainer LoRA rank {rank}."
        )

    precision = policy_config["precision"]
    lora_dtype_by_precision = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
    }
    lora_dtype = lora_dtype_by_precision[precision]
    rollout_precision = generation_config["vllm_cfg"].get("precision")
    if (
        not isinstance(rollout_precision, str)
        or lora_dtype_by_precision.get(rollout_precision) != lora_dtype
    ):
        raise ValueError(
            "Native LoRA refit requires matching policy and vLLM precision, got "
            f"policy={precision!r} and vLLM={rollout_precision!r}."
        )
    configured_lora_dtype = vllm_kwargs.get("lora_dtype")
    if configured_lora_dtype not in (None, "auto", lora_dtype):
        raise ValueError(
            f"vllm_kwargs.lora_dtype={configured_lora_dtype!r} does not match "
            f"policy precision {precision!r}."
        )

    max_loras = int(vllm_kwargs.get("max_loras", 1))
    if max_loras < 1:
        raise ValueError("vllm_kwargs.max_loras must be at least 1 for native LoRA.")
    max_cpu_loras = int(vllm_kwargs.get("max_cpu_loras", max_loras))
    if max_cpu_loras < max_loras:
        raise ValueError(
            "vllm_kwargs.max_cpu_loras must be greater than or equal to "
            "vllm_kwargs.max_loras."
        )


def configure_vllm_lora_refit(policy_config: Mapping[str, Any]) -> None:
    """Materialize vLLM adapter settings after native-refit validation."""
    generation_config = policy_config["generation"]
    lora_config = policy_config["automodel_cfg"]["lora_cfg"]
    rank = int(lora_config["dim"])
    alpha = int(lora_config["alpha"])

    precision = policy_config["precision"]
    lora_dtype = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
    }[precision]

    vllm_kwargs = generation_config.setdefault("vllm_kwargs", {})
    vllm_kwargs["enable_lora"] = True
    vllm_kwargs.setdefault("max_lora_rank", rank)
    max_loras = int(vllm_kwargs.setdefault("max_loras", 1))
    vllm_kwargs.setdefault("max_cpu_loras", max_loras)
    vllm_kwargs.setdefault("lora_dtype", lora_dtype)
    additional_config = dict(vllm_kwargs.get("additional_config") or {})
    additional_config[NATIVE_LORA_CONFIG_KEY] = {"rank": rank, "alpha": alpha}
    vllm_kwargs["additional_config"] = additional_config
    generation_config["vllm_cfg"]["load_format"] = "auto"


def make_native_lora_request(config: Mapping[str, Any]) -> Any:
    """Build the stable in-memory adapter request selected by native refit."""
    vllm_kwargs = config.get("vllm_kwargs") or {}
    additional_config = vllm_kwargs.get("additional_config") or {}
    if (
        config.get("lora_refit_mode") != "native"
        or not vllm_kwargs.get("enable_lora")
        or NATIVE_LORA_CONFIG_KEY not in additional_config
    ):
        return None

    # Optional vLLM dependency: import only in the vLLM worker environment.
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=NATIVE_LORA_ADAPTER_NAME,
        lora_int_id=NATIVE_LORA_ADAPTER_ID,
        lora_path=NATIVE_LORA_ADAPTER_PATH,
    )


def collect_native_lora(
    destination: dict[str, torch.Tensor],
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Stage adapter tensors off the transport's reusable receive buffer.

    Both native transports hand out views whose storage may be reused after the
    current batch is released, while the adapter is installed only after the
    final batch arrives.
    """
    for name, tensor in weights:
        destination[name] = tensor.to(device="cpu", copy=True)
