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

from typing import Optional, Tuple

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)
from modelopt.torch.speculative.plugins.megatron_eagle import EagleModule
from torch import Tensor


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

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Override to fix a bug in modelopt < 0.42.0.

        In modelopt < 0.42.0, EagleTransformerBlock.sharded_state_dict omits
        tp_group when calling sharded_state_dict_default for non-layer children
        (e.g. final_layernorm). This causes make_sharded_tensors_for_checkpoint
        to receive tp_group=None while dp_cp_group is set, so the
        ``tp_group is None and dp_cp_group is None`` guard never fires, and
        get_pg_rank(None)=0 is used for all TP ranks. With TP>1 and DP>1, two
        ranks end up with replica_id=(0,0,0), triggering CheckpointingException.
        """
        sd = super().sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

        decoder = self.eagle_module.decoder
        if not hasattr(decoder, "layers"):
            return sd

        metadata = ensure_metadata_has_dp_cp_group(metadata)

        # Regenerate all non-layer children of the decoder with the correct
        # tp_group. EagleTransformerBlock asserts sharded_offsets=() so we
        # always use () here too.
        for name, module in decoder.named_children():
            if module is decoder.layers:
                continue
            child_prefix = f"{prefix}eagle_module.decoder.{name}."
            for k in list(sd):
                if k.startswith(child_prefix):
                    del sd[k]
            sd.update(
                sharded_state_dict_default(
                    module,
                    child_prefix,
                    (),
                    metadata,
                    tp_group=decoder.tp_group,
                )
            )

        return sd

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        bootstrap_hidden_states: bool = True,
    ) -> Tensor:
        if bootstrap_hidden_states:
            hidden_states = self.eagle_module.fc(hidden_states)[0]
        elif hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden states with size {self.config.hidden_size} when "
                f"`bootstrap_hidden_states=False`, got {hidden_states.shape[-1]}."
            )

        hidden_states, _ = self.eagle_module(
            embeddings=input_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        logits, _ = self.eagle_module.eagle_output_layer(hidden_states)
        logits = logits.transpose(0, 1).contiguous()
        return logits
