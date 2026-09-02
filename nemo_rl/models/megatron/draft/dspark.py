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

"""Megatron training implementation of the DSpark draft model.

DSpark reuses the DFlash trunk and block-attention code, but changes the block
output and adds two prediction heads:

- A block has exactly ``W = gamma`` slots. For an anchor at position ``p``,
  slot ``j`` reads token ``x_{p+j}`` and predicts ``x_{p+j+1}``. Therefore,
  every slot makes a prediction, including slot 0 that contains the anchor.
- The Markov head converts the previous ground-truth token into a vocabulary
  bias and adds it to the decoder logits. Training uses teacher forcing.
- The confidence head combines the decoder hidden state with the Markov
  embedding and predicts whether each drafted token will be accepted. This
  score is trained with BCE and can be used to choose a dynamic draft length.

Trunk K/V construction, block attention, mask embeddings, and LM-head sharing
come from :class:`~nemo_rl.models.megatron.draft.dflash.DFlashDraftModel`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    copy_to_tensor_model_parallel_region,
)
from megatron.core.transformer import TransformerConfig
from torch import Tensor

from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel


class DSparkDraftModel(DFlashDraftModel):
    """DFlash-style block drafter with Markov and confidence heads.

    DSpark uses ``W = gamma`` because its anchor slot predicts a token instead
    of serving only as context.
    """

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
        self.markov_w2 = ColumnParallelLinear(
            self.markov_rank,
            config.draft_vocab_size,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            gather_output=False,
        )
        # Keep this head replicated. Every TP rank computes the same complete
        # confidence loss, so its gradients need no cross-rank reduction.
        self.confidence_head = torch.nn.Linear(
            config.hidden_size + self.markov_rank,
            1,
            bias=True,
            dtype=config.params_dtype,
        )

    def markov_bias(self, prev_tokens: Tensor) -> Tensor:
        """Compute the vocabulary bias from each slot's previous token.

        Training passes the ground-truth previous tokens here, so this is the
        teacher-forced Markov contribution to the final logits.

        Args:
            prev_tokens: Token preceding each prediction, with shape
                ``[B, N, gamma]``. For slot 0, this is the anchor token.

        Returns:
            Vocab-parallel logit bias with shape
            ``[B, N, gamma, draft_vocab_local]``.
        """
        embedded = self.markov_w1(prev_tokens)
        # ColumnParallelLinear expects sequence, batch, hidden dimensions.
        bias, _ = self.markov_w2(embedded.reshape(-1, 1, self.markov_rank))
        return bias.reshape(*prev_tokens.shape, -1)

    def confidence_logits(self, hidden: Tensor, prev_tokens: Tensor) -> Tensor:
        """Predict an acceptance logit for every draft slot.

        Args:
            hidden: Decoder states with shape ``[B, N, gamma, h]``.
            prev_tokens: Previous-token IDs with shape ``[B, N, gamma]``.

        Returns:
            Pre-sigmoid fp32 logits with shape ``[B, N, gamma]``.
        """
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
        """Run the DSpark drafter and its two additional heads.

        For an anchor at ``p``, slot ``j`` predicts ``x_{p+j+1}``. Its Markov
        and confidence inputs therefore use the ground-truth token ``x_{p+j}``,
        which for slot 0 is the anchor token itself.

        The confidence head reads ``decoder_hidden`` before the LM-head TP
        gradient wrapper. Its loss is already computed in full on every TP
        rank; applying the wrapper would sum that complete gradient again.

        Args:
            taps: Target auxiliary hidden states with shape ``[S, B, h_aux]``.
            input_embeds: Unshifted target embeddings with shape ``[S, B, h]``.
            anchors: Anchor positions with shape ``[B, N]``.
            anchor_valid: Validity mask with shape ``[B, N]``. Invalid blocks
                are ignored by the loss.
            lm_head_weight: Detached target LM-head shard with shape
                ``[V_local, h]``.
            mask_embedding: Detached target mask-token embedding with shape
                ``[h]``.
            input_ids: Ground-truth token IDs with shape ``[B, S]``.

        Returns:
            Markov-biased vocabulary logits with shape ``[B, N, W, V_local]``
            and pre-sigmoid confidence logits with shape ``[B, N, W]``.
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

        # Build the target-derived trunk K/V used by every draft block.
        trunk_hidden = self.hidden_norm(self.fc(taps))
        trunk_hidden = copy_to_tensor_model_parallel_region(trunk_hidden)
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

        # Create each block from its anchor embedding followed by mask vectors.
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

        head_input = copy_to_tensor_model_parallel_region(decoder_hidden)
        logits = F.linear(head_input, lm_head_weight)
        logits = logits.reshape(batch, num_anchors, block_width, -1)
        hidden = decoder_hidden.reshape(batch, num_anchors, block_width, -1)

        # Slot j uses the ground-truth token at p + j as its previous token.
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
