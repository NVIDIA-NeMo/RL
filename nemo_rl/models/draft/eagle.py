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

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer import MegatronModule, TransformerConfig
from modelopt.torch.speculative.plugins.megatron_eagle import EagleModule
from torch import Tensor


def build_default_causal_attention_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )


def shift_attention_mask_for_next_token_inputs(attention_mask: Tensor) -> Tensor:
    """Align a causal mask with left-shifted next-token draft inputs."""
    if attention_mask.ndim != 4 or attention_mask.shape[-1] != attention_mask.shape[-2]:
        raise ValueError(
            "EagleModel.forward expects a square attention mask with shape [b, 1, s, s]."
        )

    shifted_attention_mask = attention_mask.clone()
    shifted_attention_mask[:, :, :-1, :-1] = attention_mask[:, :, 1:, 1:]
    shifted_attention_mask[:, :, -1, :] = True
    shifted_attention_mask[:, :, :, -1] = True
    return shifted_attention_mask.contiguous()


class EagleModel(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config

        rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            seq_len_interpolation_factor=None,
            rotary_base=getattr(config, "rotary_base", 10000),
            rope_scaling=getattr(config, "rope_scaling", False),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", 8.0),
            use_cpu_initialization=getattr(
                config,
                "use_cpu_initialization",
                not torch.cuda.is_available(),
            ),
        )
        self.eagle_module = EagleModule(
            config=config, rotary_pos_emb=rotary_pos_emb, bias=False
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            skip_weight_param_allocation=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        bootstrap_hidden_states: bool = True,
        lm_head_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if lm_head_weight is None:
            raise ValueError("EagleModel.forward requires an LM head weight tensor.")

        if bootstrap_hidden_states:
            hidden_states = self.eagle_module.fc(hidden_states)[0]
        elif hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden states with size {self.config.hidden_size} when "
                f"`bootstrap_hidden_states=False`, got {hidden_states.shape[-1]}."
            )

        attention_mask = shift_attention_mask_for_next_token_inputs(attention_mask)

        hidden_states, _ = self.eagle_module(
            embeddings=input_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        logits, _ = self.lm_head(hidden_states, weight=lm_head_weight.detach())
        logits = logits.transpose(0, 1).contiguous()
        return logits
