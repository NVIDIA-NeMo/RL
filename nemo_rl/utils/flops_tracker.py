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
from typing import Callable, Optional

import torch
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_rl.models.policy.utils import sliding_window_overwrite
from nemo_rl.utils.flops_formulas import FLOPSConfig, llama2, llama3, qwen2, qwen3


def get_default_hf_config(model_name: str) -> PretrainedConfig:
    """Get the default Hugging Face config for a model.

    Both the DTensor and MCore paths use the same default config, we initialize the model config
    here to allow computation of theoretical flops which is agnostic to the backend.
    """
    return AutoConfig.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        **sliding_window_overwrite(model_name),
    )


def convert_config_to_flops_config(
    model_name: str, config: PretrainedConfig
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
    elif isinstance(config, Qwen3Config):
        return FLOPSConfig(
            gbs=0,
            hs=config.hidden_size,
            layers=config.num_hidden_layers,
            ffn_hs=config.intermediate_size,
            vocab_size=config.vocab_size,
            query_groups=config.num_attention_heads / config.num_key_value_heads,
            attention_heads=config.num_attention_heads,
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
            query_groups=config.num_attention_heads / config.num_key_value_heads,
            attention_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
        ), llama3 if "llama3" in model_name.lower() else llama2
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")


THEORETICAL_TFLOPS = {
    ("NVIDIA H100 80GB HBM3", torch.bfloat16): 1979 / 2,
    ("NVIDIA H100 80GB HBM3", torch.float32): 67.0,
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
        flops_config, flops_formula = convert_config_to_flops_config(model_name, config)
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
