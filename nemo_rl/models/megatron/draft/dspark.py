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

from typing import Any, Optional

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

    speculator_type = "dspark"

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
        layer_windows: Optional[list[int]] = None,
    ):
        super().__init__(
            config,
            gamma=gamma,
            mask_token_id=mask_token_id,
            num_aux_hidden_states=num_aux_hidden_states,
            target_hidden_size=target_hidden_size,
            trunk_chunk=trunk_chunk,
            layer_windows=layer_windows,
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
                ``[..., gamma]``. For slot 0, this is the anchor token.

        Returns:
            Vocab-parallel logit bias with shape
            ``[..., gamma, draft_vocab_local]``.
        """
        embedded = self.markov_w1(prev_tokens)
        # ColumnParallelLinear expects sequence, batch, hidden dimensions.
        bias, _ = self.markov_w2(embedded.reshape(-1, 1, self.markov_rank))
        return bias.reshape(*prev_tokens.shape, -1)

    def confidence_logits(self, hidden: Tensor, prev_tokens: Tensor) -> Tensor:
        """Predict an acceptance logit for every draft slot.

        Args:
            hidden: Decoder states with shape ``[..., gamma, h]``.
            prev_tokens: Previous-token IDs with shape ``[..., gamma]``.

        Returns:
            Pre-sigmoid fp32 logits with shape ``[..., gamma]``.
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
        packed_seq_params: Optional[Any] = None,
        block_seq_idx: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the DSpark drafter and its two additional heads.

        For an anchor at ``p``, slot ``j`` predicts ``x_{p+j+1}``. Its Markov
        and confidence inputs therefore use the ground-truth token
        ``x_{p+j}``. ``input_ids`` remains an unpacked, replicated ``[B, S]``
        tensor even when the model trunk is packed, so these tokens can be
        gathered locally without extra CP communication.

        The confidence head reads ``decoder_hidden`` before the LM-head TP
        gradient wrapper. Its loss is already computed in full on every TP
        rank; applying the wrapper would sum that complete gradient again.

        Args:
            taps: Target auxiliary hidden states in padded or packed layout.
            input_embeds: Unshifted target embeddings in the same layout as
                ``taps``.
            anchors: Anchor positions. Shape is ``[B, N]`` for padded input or
                ``[NB]`` for packed blocks owned by this rank.
            anchor_valid: Validity mask with the same layout as ``anchors``.
                Invalid blocks are ignored by the loss.
            lm_head_weight: Detached target LM-head shard with shape
                ``[V_local, h]``.
            mask_embedding: Detached target mask-token embedding with shape
                ``[h]``.
            input_ids: Replicated ground-truth token IDs with shape ``[B, S]``.
            packed_seq_params: Global THD packing metadata, or ``None`` for
                padded input.
            block_seq_idx: For packed input, the subsequence index of each
                local block, with shape ``[NB]``.

        Returns:
            Markov-biased vocabulary logits and pre-sigmoid confidence logits.
            Their padded shapes are ``[B, N, W, V_local]`` and ``[B, N, W]``;
            packed shapes are ``[NB, W, V_local]`` and ``[NB, W]``.
        """
        device = taps.device
        block_width = self.block_width
        (
            taps_flat,
            embeds_flat,
            block_seq,
            anchors_flat,
            cu_local,
            pos_in_seq,
            max_len,
            cp_group,
            out_shape,
        ) = self._flatten_to_thd(
            taps, input_embeds, anchors, packed_seq_params, block_seq_idx
        )
        num_blocks = block_seq.shape[0]
        if int(anchors_flat.max().item()) >= max_len:
            raise ValueError("anchor position exceeds sequence length.")

        # Build the target-derived trunk K/V used by every draft block.
        trunk_hidden = self.hidden_norm(self.fc(taps_flat))
        trunk_hidden = copy_to_tensor_model_parallel_region(trunk_hidden)
        # Build one full RoPE table and index it with explicit global positions;
        # this avoids RotaryEmbedding slicing it a second time for CP.
        rotary_table = self.rotary_pos_emb(max_len + block_width, packed_seq=True)
        trunk_freqs = rotary_table[pos_in_seq]

        vis_len = anchors_flat

        for layer, core in zip(self.decoder.layers, self._block_attn_modules):
            key, value = self._project_trunk_kv(layer.self_attention, trunk_hidden)
            key = apply_rotary_pos_emb(key, trunk_freqs, config=self.config)
            core.stage_trunk(
                key.squeeze(1).contiguous(),
                value.squeeze(1).contiguous(),
                block_seq,
                vis_len,
                cu_local,
                block_width,
                cp_group,
            )

        # Create each block from its anchor embedding followed by mask vectors.
        rows = self._anchor_embed_index(block_seq, anchors_flat, cu_local, cp_group)
        mask_row = mask_embedding.to(embeds_flat.dtype)
        hidden = mask_row.expand(num_blocks, block_width, -1).clone()
        hidden[:, 0] = embeds_flat[rows]
        hidden = hidden.reshape(num_blocks * block_width, 1, -1)

        offsets = torch.arange(block_width, device=device)
        positions = (anchors_flat.unsqueeze(1) + offsets).reshape(-1)
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
        logits = logits.reshape(*out_shape)
        hidden = decoder_hidden.reshape(*out_shape[:-1], -1)

        # Slot j uses the ground-truth token at p + j as its previous token.
        seq_len = input_ids.shape[1]
        prev_pos = positions.reshape(num_blocks, block_width).clamp(max=seq_len - 1)
        prev_ids = input_ids[
            block_seq.unsqueeze(1).expand(-1, block_width), prev_pos
        ].reshape(*out_shape[:-1])
        return (
            logits + self.markov_bias(prev_ids),
            self.confidence_logits(hidden, prev_ids),
        )
