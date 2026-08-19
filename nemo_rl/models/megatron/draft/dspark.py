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

"""Megatron DSpark draft model for online co-training.

Training-side counterpart of vLLM 0.26's
``vllm/model_executor/models/qwen3_dspark.py``. DSpark blocks have no bonus
slot: ``W = gamma`` and every slot (the anchor included) predicts the *next*
token (``sample_pos = query_pos + 1``) — the same labels
``x_{p+1} .. x_{p+gamma}`` as DFlash. On top of the shared machinery, the
official architecture adds two fixed heads (deepspec/modeling/dspark):

- Markov head: ``markov_w2(markov_w1(prev_token))`` is a logit bias added to
  each slot's base logits, teacher-forced with the ground-truth previous
  token during training.
- Confidence head: ``Linear(cat(hidden, markov_w1(prev_token)))`` predicts
  the per-slot TV acceptance rate (BCE-trained); serving-side vLLM ignores
  it, DeepSpec's evaluator uses it for dynamic draft length.

Everything else (trunk K/V projection, block attention, mask embedding,
LM head) is inherited from
:class:`~nemo_rl.models.megatron.draft.dflash.DFlashDraftModel`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer import TransformerConfig
from torch import Tensor

from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel


class DSparkDraftModel(DFlashDraftModel):
    """DSpark block draft: ``W = gamma``, plus the Markov and confidence heads."""

    method = "dspark"

    def __init__(
        self,
        config: TransformerConfig,
        *,
        gamma: int,
        mask_token_id: int,
        num_aux_hidden_states: int,
        target_hidden_size: Optional[int] = None,
        markov_rank: int = 64,
        trunk_chunk: int = 1024,
    ):
        super().__init__(
            config,
            gamma=gamma,
            mask_token_id=mask_token_id,
            num_aux_hidden_states=num_aux_hidden_states,
            target_hidden_size=target_hidden_size,
            trunk_chunk=trunk_chunk,
            block_width=gamma,  # no bonus anchor slot
        )
        if markov_rank < 1:
            raise ValueError(f"markov_rank must be >= 1, got {markov_rank}.")
        self.markov_rank = int(markov_rank)
        self.markov_w1 = torch.nn.Embedding(
            config.vocab_size, self.markov_rank, dtype=config.params_dtype
        )
        self.markov_w2 = tensor_parallel.ColumnParallelLinear(
            self.markov_rank,
            config.draft_vocab_size,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
            skip_weight_param_allocation=False,
        )
        # Replicated on purpose (like fc): every TP rank computes the same
        # full confidence loss, so grads are complete without a collective.
        self.confidence_head = torch.nn.Linear(
            config.hidden_size + self.markov_rank,
            1,
            bias=True,
            dtype=config.params_dtype,
        )

    def markov_bias(self, prev_tokens: Tensor) -> Tensor:
        """DSpark transition bias for ground-truth previous tokens.

        Args:
            prev_tokens: ``[B, N, gamma]`` the token preceding each predicted
                position (teacher forcing; slot 0's prev is the anchor token).

        Returns:
            ``[B, N, gamma, draft_vocab_local]`` logit bias (vocab-parallel).
        """
        embedded = self.markov_w1(prev_tokens)
        # Megatron's ColumnParallelLinear expects the [s, b, h] layout.
        bias, _ = self.markov_w2(embedded.reshape(-1, 1, self.markov_rank))
        return bias.reshape(*prev_tokens.shape, -1)

    def confidence_logits(self, hidden: Tensor, prev_tokens: Tensor) -> Tensor:
        """Pre-sigmoid acceptance-rate prediction, ``[B, N, gamma]`` fp32."""
        prev_embeddings = self.markov_w1(prev_tokens).to(dtype=hidden.dtype)
        features = torch.cat([hidden, prev_embeddings], dim=-1)
        return self.confidence_head(features).squeeze(-1).float()

    def forward(
        self,
        *,
        taps: Tensor,
        input_embeds: Tensor,
        anchors: Tensor,
        anchor_valid: Tensor,
        lm_head_weight: Tensor,
        mask_embedding: Tensor,
        input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Copy of :meth:`DFlashDraftModel.forward` plus the DSpark heads.

        ``input_ids`` ``[B, S]`` teacher-force the Markov/confidence heads:
        slot j predicts ``x_{p+1+j}``, so its prev token is ``x_{p+j}`` (slot
        0's prev is the anchor token itself). Returns the Markov-biased
        logits ``[B, N, W, V_local]`` and the confidence prediction
        ``[B, N, W]``. The confidence head consumes the decoder hidden BEFORE
        the LM head's copy-to-TP-region: its loss is computed in full on
        every TP rank, so d(hidden) is already complete — the copy region
        would TP-SUM it into an overcount.
        """
        seq_len, batch = taps.shape[0], taps.shape[1]
        num_anchors = anchors.shape[1]
        num_blocks = batch * num_anchors
        block_width = self.block_width
        device = taps.device

        if anchors.shape[0] != batch:
            raise ValueError(f"anchors batch {anchors.shape[0]} != taps batch {batch}.")
        if int(anchors.max().item()) >= seq_len:
            raise ValueError("anchor position exceeds sequence length.")

        # ---- Trunk stream: fc -> hidden_norm -> per-layer K/V + RoPE ----
        trunk_hidden = self.hidden_norm(self.fc(taps))
        trunk_hidden = tensor_parallel.copy_to_tensor_model_parallel_region(
            trunk_hidden
        )
        rotary_table = self.rotary_pos_emb(seq_len + block_width)
        trunk_freqs = rotary_table[:seq_len]

        block_row = torch.arange(batch, device=device).repeat_interleave(num_anchors)
        anchors_flat = anchors.reshape(-1)
        vis_len = anchors_flat

        for layer, core in zip(self.decoder.layers, self._block_attn_modules):
            key, value = self._project_trunk_kv(layer.self_attention, trunk_hidden)
            key = apply_rotary_pos_emb(key, trunk_freqs, config=self.config)
            core.stage_trunk(
                key.permute(1, 0, 2, 3).contiguous(),
                value.permute(1, 0, 2, 3).contiguous(),
                block_row,
                vis_len,
                block_width,
            )

        # ---- Block stream: anchor embedding + mask embeddings ----
        embeds_flat = input_embeds.permute(1, 0, 2).reshape(batch * seq_len, -1)
        anchor_embeds = embeds_flat[block_row * seq_len + anchors_flat]
        hidden = (
            mask_embedding.to(anchor_embeds.dtype)
            .expand(num_blocks, block_width, -1)
            .clone()
        )
        hidden[:, 0] = anchor_embeds
        hidden = hidden.reshape(num_blocks * block_width, 1, -1)

        positions = (
            anchors_flat.unsqueeze(1)
            + torch.arange(block_width, device=device).unsqueeze(0)
        ).reshape(-1)
        block_freqs = rotary_table[positions]

        try:
            decoder_hidden = self.decoder(
                hidden_states=hidden,
                attention_mask=None,
                rotary_pos_emb=block_freqs,
            )
        finally:
            for core in self._block_attn_modules:
                core.reset()

        head_input = tensor_parallel.copy_to_tensor_model_parallel_region(
            decoder_hidden
        )
        logits = F.linear(head_input, lm_head_weight)
        logits = logits.reshape(batch, num_anchors, block_width, -1)
        hidden = decoder_hidden.reshape(batch, num_anchors, block_width, -1)

        prev_pos = (
            anchors.unsqueeze(-1)
            + torch.arange(block_width, device=device).view(1, 1, -1)
        ).clamp(max=seq_len - 1)
        prev_ids = torch.gather(input_ids, 1, prev_pos.reshape(batch, -1)).reshape(
            batch, num_anchors, block_width
        )
        return (
            logits + self.markov_bias(prev_ids),
            self.confidence_logits(hidden, prev_ids),
        )
