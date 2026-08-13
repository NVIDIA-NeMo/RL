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
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)
from torch import Tensor

from nemo_rl.models.megatron.draft.ttt_attention import TTTDraftCoreAttention


class EagleModel(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        ttt_steps: int = 1,
    ):
        super().__init__(config=config)
        self.config = config
        self.ttt_steps = int(ttt_steps)
        if self.ttt_steps < 1:
            raise ValueError(f"ttt_steps must be >= 1, got {self.ttt_steps}.")

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
        # Prevent modelopt import from breaking unrelated functionality.
        # TODO: Investigate the circular import chain inside `modelopt.torch.quantization`:
        # backends/__init__.py -> from .nvfp4_gemm import * -> nvfp4_gemm.py ->
        # from ...quant_linear import RealQuantLinear -> quant_linear.py -> from ... import backends
        from modelopt.torch.speculative.plugins.megatron_eagle import EagleModule

        # Many specdec libraries use LlamaForCausalLMEagle3 class by default so rope is hardcoded
        self.eagle_module = EagleModule(
            config=config, rotary_pos_emb=rotary_pos_emb, bias=False
        )

        # ModelOpt builds the Eagle decoder with the `arbitrary` attention-mask
        # type, needed only for its multi-step-TTT staircase masks. NeMo-RL
        # trains the draft with a single TTT step whose mask is plain causal,
        # and `arbitrary` routes TE to the unfused backend that materializes
        # the full O(seq^2) fp32 attention-probs tensor (OOM at long sequence
        # lengths). Force the fused/flash causal path; the forward must then
        # pass attention_mask=None.
        for layer in self.eagle_module.decoder.layers:
            layer.self_attention.attn_mask_type = AttnMaskType.causal

        self._ttt_attn_modules: list[TTTDraftCoreAttention] = []
        self._ttt_prenorm_hidden: Optional[Tensor] = None
        if self.ttt_steps > 1:
            if getattr(config, "recompute_granularity", None) is not None:
                # The checkpointed core-attention path re-enters forward during
                # backward, which would corrupt the stateful per-pass KV stash.
                raise ValueError(
                    "TTT draft training (ttt_steps > 1) is incompatible with "
                    "activation recomputation on the draft config."
                )
            for layer in self.eagle_module.decoder.layers:
                ttt_attention = TTTDraftCoreAttention(config)
                layer.self_attention.core_attention = ttt_attention
                self._ttt_attn_modules.append(ttt_attention)
            # modelopt's own last-layer hook detaches its capture (built for
            # its recursion-free single-step use); TTT needs the pre-final-norm
            # hidden state WITH gradients as the next pass's input, so register
            # a separate non-detaching hook.
            self.eagle_module.decoder.layers[-1].register_forward_hook(
                self._capture_prenorm_hidden_hook
            )

    def _capture_prenorm_hidden_hook(
        self, _module: torch.nn.Module, _args: Tuple, output: Tensor | Tuple
    ) -> None:
        hidden_states = output[0] if isinstance(output, tuple) else output
        self._ttt_prenorm_hidden = hidden_states

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

    def _ttt_rotary_pos_emb(self, ttt_pass: int, seq_len: int) -> Optional[Tensor]:
        """Per-pass RoPE positions ``[d-1, S+d-2]``.

        Pass-d anchor ``i`` sits at absolute position ``i + d - 1`` — both at
        inference (the draft's KV cache advances one slot per speculated
        token) and in modelopt's reference TTT (its repeated rotary plus the
        one-row-per-pass stream shift produce exactly this offset). Pass 1 is
        positions ``[0, S-1]``, which the eagle module computes itself from a
        ``None`` rotary input.
        """
        if ttt_pass == 1:
            return None
        rotary = self.eagle_module.rotary_pos_emb(seq_len + ttt_pass - 1)
        return rotary[ttt_pass - 1 :]

    def forward_ttt(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
    ) -> list[Tensor]:
        """Run ``ttt_steps`` sequential draft passes and return per-pass logits.

        Args:
            hidden_states: Captured policy aux hidden states ``[S, B, 3h]``
                (bootstrapped through ``fc`` for pass 1 only).
            input_embeds: Pass-1 input embeddings ``e(x_{i+1})`` ``[S, B, h]``
                (the caller has already rolled the captured embeddings by -1).

        Returns:
            One ``[B, S, draft_vocab]`` logits tensor per pass; pass ``d``
            position ``i`` predicts token ``x_{i+d+1}``.

        Pass ``d >= 2`` feeds the previous pass's pre-final-norm hidden state
        (non-detached: the h-chain must carry gradients) and the embeddings
        rolled one more step. The per-layer attention modules stash pass-1 KV
        as the trunk and later passes' KV as branch diagonals (see
        ``TTTDraftCoreAttention``).
        """
        if self.ttt_steps < 2:
            raise RuntimeError(
                "forward_ttt requires ttt_steps >= 2; use forward() for the "
                "single-pass draft."
            )
        # Deferred import mirrors train.py: mcore's MTP module is heavy and
        # only needed on the draft training path.
        from megatron.core.transformer.multi_token_prediction import roll_tensor

        hidden = self.eagle_module.fc(hidden_states)[0]
        embeds = input_embeds
        logits_by_pass: list[Tensor] = []
        try:
            for ttt_pass in range(1, self.ttt_steps + 1):
                for ttt_attention in self._ttt_attn_modules:
                    ttt_attention.begin_pass(ttt_pass)
                self._ttt_prenorm_hidden = None
                decoder_hidden, _ = self.eagle_module(
                    embeddings=embeds,
                    hidden_states=hidden,
                    # The TTT attention modules build their own trunk/branch
                    # masking; an explicit mask tensor is rejected there.
                    attention_mask=None,
                    rotary_pos_emb=self._ttt_rotary_pos_emb(ttt_pass, embeds.shape[0]),
                )
                logits, _ = self.eagle_module.eagle_output_layer(decoder_hidden)
                logits_by_pass.append(logits.transpose(0, 1).contiguous())

                if ttt_pass < self.ttt_steps:
                    if self._ttt_prenorm_hidden is None:
                        raise RuntimeError(
                            "TTT pre-norm hidden-state capture hook did not "
                            "fire; cannot feed the next draft pass."
                        )
                    hidden = self._ttt_prenorm_hidden
                    embeds = roll_tensor(
                        embeds,
                        shifts=-1,
                        dims=0,
                        cp_group=parallel_state.get_context_parallel_group(),
                    )[0]
        finally:
            # Never leak stashed KV / captured hidden into the next microbatch,
            # even if a pass raises mid-loop.
            for ttt_attention in self._ttt_attn_modules:
                ttt_attention.reset()
            self._ttt_prenorm_hidden = None
        return logits_by_pass
