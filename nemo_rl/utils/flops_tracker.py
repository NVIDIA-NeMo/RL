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

from dataclasses import asdict
from typing import Any, Callable, Optional

import torch
from packaging.version import Version as PkgVersion
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from nemo_rl.utils.flops_formulas import (
    FLOPSConfig,
    deepseekv3,
    gpt_oss,
    llama,
    qwen2,
    qwen3,
)


def get_default_hf_config(model_name: str) -> PretrainedConfig:
    """Get the default Hugging Face config for a model.

    Both the DTensor and MCore paths use the same default config, we initialize the model config
    here to allow computation of theoretical flops which is agnostic to the backend.
    """
    return AutoConfig.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )


# =============================================================================
# Config Builders: Functions that extract FLOPSConfig from HuggingFace configs
# =============================================================================


def _build_qwen2_config(config: Any) -> FLOPSConfig:
    """Build FLOPSConfig for Qwen2 models."""
    return FLOPSConfig(
        gbs=0,
        hs=config.hidden_size,
        layers=config.num_hidden_layers,
        ffn_hs=config.intermediate_size,
        vocab_size=config.vocab_size,
    )


def _build_qwen3_config(config: Any) -> FLOPSConfig:
    """Build FLOPSConfig for Qwen3/Qwen3MoE models."""
    return FLOPSConfig(
        gbs=0,
        hs=config.hidden_size,
        layers=config.num_hidden_layers,
        ffn_hs=config.intermediate_size,
        vocab_size=config.vocab_size,
        query_groups=config.num_key_value_heads,
        attention_heads=config.num_attention_heads,
        moe_ffn_hidden_size=config.intermediate_size,
        moe_router_topk=1,
    )


def _build_llama_config(config: Any) -> FLOPSConfig:
    """Build FLOPSConfig for Llama models."""
    return FLOPSConfig(
        gbs=0,
        hs=config.hidden_size,
        layers=config.num_hidden_layers,
        ffn_hs=config.intermediate_size,
        query_groups=config.num_key_value_heads,
        attention_heads=config.num_attention_heads,
        vocab_size=config.vocab_size,
    )


def _build_deepseekv3_config(config: Any) -> FLOPSConfig:
    """Build FLOPSConfig for DeepSeek V3 models."""
    return FLOPSConfig(
        gbs=0,
        hs=config.hidden_size,
        layers=config.num_hidden_layers,
        ffn_hs=config.intermediate_size,
        attention_heads=config.num_attention_heads,
        moe_router_topk=config.num_experts_per_tok,
        query_groups=config.num_key_value_heads,
        vocab_size=config.vocab_size,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_head_dim=config.qk_nope_head_dim,
        qk_pos_emb_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        moe_layer_freq=1,
        moe_shared_expert_intermediate_size=config.moe_intermediate_size,
        moe_ffn_hidden_size=config.moe_intermediate_size,
        mtp_num_layers=0,
        causal_self_attn=True,
    )


def _build_gpt_oss_config(config: Any) -> FLOPSConfig:
    """Build FLOPSConfig for GPT-OSS models (SWA + MoE)."""
    return FLOPSConfig(
        gbs=0,
        hs=config.hidden_size,
        layers=config.num_hidden_layers,
        ffn_hs=config.intermediate_size,
        attention_heads=config.num_attention_heads,
        query_groups=getattr(config, "num_key_value_heads", config.num_attention_heads),
        vocab_size=config.vocab_size,
        moe_router_topk=getattr(config, "num_experts_per_tok", 1),
        moe_ffn_hidden_size=getattr(config, "moe_ffn_hidden_size", config.intermediate_size),
        kv_channels=getattr(config, "kv_channels", None),
        swa_window_size=config.window_size[0] if hasattr(config, "window_size") and config.window_size else 128,
        window_attn_skip_freq=getattr(config, "window_attn_skip_freq", 2),
    )


# =============================================================================
# Registry: Maps config class names to (config_builder, flops_formula) tuples
# =============================================================================

# Registry mapping class names to (config_builder, formula) tuples
_FLOPS_REGISTRY: dict[str, tuple[Callable[[Any], FLOPSConfig], Callable]] = {
    # Qwen family
    "Qwen2Config": (_build_qwen2_config, qwen2),
    "Qwen3Config": (_build_qwen3_config, qwen3),
    "Qwen3MoeConfig": (_build_qwen3_config, qwen3),
    # Llama family
    "LlamaConfig": (_build_llama_config, llama),
    # DeepSeek V3
    "DeepseekV3Config": (_build_deepseekv3_config, deepseekv3),
    # GPT-OSS
    "GptOssConfig": (_build_gpt_oss_config, gpt_oss),
}

# Fallback registry by model_type (for configs loaded via AutoConfig with trust_remote_code)
_MODEL_TYPE_REGISTRY: dict[str, tuple[Callable[[Any], FLOPSConfig], Callable]] = {
    "qwen2": (_build_qwen2_config, qwen2),
    "qwen3": (_build_qwen3_config, qwen3),
    "qwen3_moe": (_build_qwen3_config, qwen3),
    "llama": (_build_llama_config, llama),
    "deepseek_v3": (_build_deepseekv3_config, deepseekv3),
    "gpt_oss": (_build_gpt_oss_config, gpt_oss),
}


def convert_config_to_flops_config(
    config: PretrainedConfig,
) -> tuple[FLOPSConfig, Callable]:
    """Convert a pretrained config to a tuple containing a FLOPSConfig and a flops formula.

    Uses a registry-based approach for extensibility:
    1. First tries to match by config class name
    2. Falls back to matching by model_type attribute
    """
    config_class_name = config.__class__.__name__

    # Try exact class name match first
    if config_class_name in _FLOPS_REGISTRY:
        config_builder, formula = _FLOPS_REGISTRY[config_class_name]
        return config_builder(config), formula

    # Fallback: try model_type attribute
    model_type = getattr(config, "model_type", None)
    if model_type and model_type in _MODEL_TYPE_REGISTRY:
        config_builder, formula = _MODEL_TYPE_REGISTRY[model_type]
        return config_builder(config), formula

    raise ValueError(f"Unsupported config type: {type(config)}")


def is_using_tf32() -> bool:
    """Check if the current device is using TF32."""
    if PkgVersion(torch.__version__) < PkgVersion("2.9.0a0"):
        return torch.backends.cuda.matmul.allow_tf32
    else:
        return torch.backends.cuda.matmul.fp32_precision == "tf32"


THEORETICAL_TFLOPS = {
    ("NVIDIA A100 80GB PCIe", torch.bfloat16): 624 / 2,
    ("NVIDIA A100 80GB PCIe", torch.float32): 312 / 2 if is_using_tf32() else 19.5,
    ("NVIDIA H100 80GB HBM3", torch.bfloat16): 1979 / 2,
    ("NVIDIA H100 80GB HBM3", torch.float32): 989 / 2 if is_using_tf32() else 67.0,
    ("NVIDIA H200", torch.bfloat16): 1979 / 2,
    ("NVIDIA H200", torch.float32): 989 / 2 if is_using_tf32() else 67.0,
    ("NVIDIA B200", torch.bfloat16): 4500 / 2,
    ("NVIDIA B200", torch.float32): 2200 / 2 if is_using_tf32() else 80.0,
    ("NVIDIA B300", torch.bfloat16): 4500 / 2,
    ("NVIDIA B300", torch.float32): 2200 / 2 if is_using_tf32() else 80.0,
    ("NVIDIA GB200", torch.bfloat16): 4900 / 2,
    ("NVIDIA GB200", torch.float32): 2500 / 2 if is_using_tf32() else 80.0,
    ("NVIDIA GB300", torch.bfloat16): 4900 / 2,
    ("NVIDIA GB300", torch.float32): 2500 / 2 if is_using_tf32() else 80.0,
}


def get_theoretical_tflops(device_name: str, model_dtype: torch.dtype) -> float:
    """Get the theoretical total flops for a device name."""
    if (device_name, model_dtype) in THEORETICAL_TFLOPS:
        return THEORETICAL_TFLOPS[(device_name, model_dtype)]
    else:
        raise ValueError(
            f"Unknown device name: {device_name} and dtype name: {model_dtype}"
        )


class FLOPTracker:
    def __init__(
        self,
        model_name: str,
        base_config: FLOPSConfig | None = None,
        flops_formula: Callable[[FLOPSConfig], float] | None = None,
    ):
        self.model_name = model_name
        self.base_config = base_config
        self.total_flops = 0
        self.flops_formula: Optional[Callable[[FLOPSConfig], float]] = flops_formula

    @classmethod
    def from_config(cls, model_name: str, config: PretrainedConfig) -> "FLOPTracker":
        flops_config, flops_formula = convert_config_to_flops_config(config)
        return cls(
            model_name=model_name, base_config=flops_config, flops_formula=flops_formula
        )

    def track(self, n_samples: int, padded_seq_len: int):
        if self.flops_formula is None:
            raise ValueError("Flops formula is not set")

        base_config_dict = (
            asdict(self.base_config) if self.base_config is not None else {}
        )

        # Override gbs and enc_seq_len with current values
        config_dict = {
            **base_config_dict,
            "gbs": n_samples,
            "enc_seq_len": padded_seq_len,
        }

        # Compute and accumulate flops
        flops = self.flops_formula(FLOPSConfig(**config_dict))
        self.total_flops += flops

    def track_batch(self, sequence_lengths: list[int]):
        """Track the flops for a batch of sequences."""
        for seq_len in sequence_lengths:
            self.track(n_samples=1, padded_seq_len=seq_len)

    def reset(self):
        self.total_flops = 0
