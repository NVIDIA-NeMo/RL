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
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_rl.utils.flops_formulas import (
    FLOPSConfig,
    deepseekv3,
    glm_moe_dsa,
    llama,
    qwen2,
    qwen3,
)


def get_hf_config(model_name: str, **overrides: Any) -> PretrainedConfig:
    """Get the effective Hugging Face config for a model.

    Both the DTensor and MCore paths use the same config, which allows backend-agnostic
    theoretical FLOPs computation. Policy overrides must be included so the tracker describes
    the model that is actually trained.
    """
    return AutoConfig.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        **overrides,
    )


def _get_glm_moe_layer_pattern(config: PretrainedConfig) -> list[int]:
    """Return the GLM dense/MoE layer pattern from the model config."""
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    if mlp_layer_types is None:
        first_k_dense = config.first_k_dense_replace
        if not 0 <= first_k_dense <= config.num_hidden_layers:
            raise ValueError(
                "GLM first_k_dense_replace must be between zero and the number "
                f"of hidden layers, got {first_k_dense}"
            )
        return [0] * first_k_dense + [1] * (config.num_hidden_layers - first_k_dense)

    if len(mlp_layer_types) != config.num_hidden_layers:
        raise ValueError(
            "GLM mlp_layer_types must contain one entry per hidden layer, got "
            f"{len(mlp_layer_types)} entries for {config.num_hidden_layers} layers"
        )

    valid_layer_types = {"dense", "sparse"}
    invalid_layer_types = set(mlp_layer_types) - valid_layer_types
    if invalid_layer_types:
        raise ValueError(f"Unsupported GLM MLP layer types: {invalid_layer_types}")
    return [0 if layer_type == "dense" else 1 for layer_type in mlp_layer_types]


def _get_glm_index_compute_layers(config: PretrainedConfig) -> int:
    """Return how many GLM layers compute, rather than reuse, DSA indices."""
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != config.num_hidden_layers:
            raise ValueError(
                "GLM indexer_types must contain one entry per hidden layer, got "
                f"{len(indexer_types)} entries for {config.num_hidden_layers} layers"
            )

        valid_indexer_types = {"full", "shared"}
        invalid_indexer_types = set(indexer_types) - valid_indexer_types
        if invalid_indexer_types:
            raise ValueError(
                f"Unsupported GLM DSA indexer types: {invalid_indexer_types}"
            )
        return sum(indexer_type == "full" for indexer_type in indexer_types)

    topk_freq = getattr(config, "index_topk_freq", 1) or 1
    skip_topk_offset = getattr(config, "index_skip_topk_offset", 0) or 0
    if topk_freq < 1:
        raise ValueError(f"GLM index_topk_freq must be positive, got {topk_freq}")
    if skip_topk_offset < 0:
        raise ValueError(
            f"GLM index_skip_topk_offset must be non-negative, got {skip_topk_offset}"
        )
    skip_topk_offset = max(skip_topk_offset, 1)
    return sum(
        max(layer_number - skip_topk_offset, 0) % topk_freq == 0
        for layer_number in range(1, config.num_hidden_layers + 1)
    )


def convert_config_to_flops_config(
    config: PretrainedConfig,
) -> tuple[FLOPSConfig, Callable]:
    """Convert a pretrained config to a tuple containing a FLOPSConfig and a flops formula."""
    if isinstance(config, Qwen2Config):
        return FLOPSConfig(
            gbs=0,
            hs=config.hidden_size,
            layers=config.num_hidden_layers,
            ffn_hs=config.intermediate_size,
            vocab_size=config.vocab_size,
        ), qwen2
    elif isinstance(config, (Qwen3Config, Qwen3MoeConfig)):
        return FLOPSConfig(
            gbs=0,
            hs=config.hidden_size,
            layers=config.num_hidden_layers,
            ffn_hs=config.intermediate_size,
            vocab_size=config.vocab_size,
            query_groups=config.num_key_value_heads,
            attention_heads=config.num_attention_heads,
            # head_dim is set explicitly by Qwen3 and is NOT always hidden_size/num_heads
            # (e.g. Qwen3-235B-A22B: head_dim=128, num_heads=64, hidden=4096 -> "wide" attention).
            head_dim=getattr(config, "head_dim", None),
            # for non-MoE models, we use the intermediate size as the ffn hidden size
            moe_ffn_hidden_size=config.intermediate_size,
            moe_router_topk=1,
        ), qwen3
    elif isinstance(config, LlamaConfig):
        return FLOPSConfig(
            gbs=0,
            hs=config.hidden_size,
            layers=config.num_hidden_layers,
            ffn_hs=config.intermediate_size,
            query_groups=config.num_key_value_heads,
            attention_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
        ), llama
    elif config.__class__.model_type == "deepseek_v3":
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
        ), deepseekv3
    elif config.__class__.model_type == "glm_moe_dsa":
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
            moe_layer_freq=_get_glm_moe_layer_pattern(config),
            moe_shared_expert_intermediate_size=(
                config.moe_intermediate_size * config.n_shared_experts
            ),
            moe_ffn_hidden_size=config.moe_intermediate_size,
            # GLM-5 HF config may expose next-token-prediction heads, but the
            # MCore Bridge path used by these runs disables MTP mappings.
            mtp_num_layers=None,
            causal_self_attn=True,
            dsa_indexer_n_heads=config.index_n_heads,
            dsa_indexer_head_dim=config.index_head_dim,
            dsa_indexer_topk=config.index_topk,
            dsa_indexer_compute_layers=_get_glm_index_compute_layers(config),
        ), glm_moe_dsa
    else:
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
