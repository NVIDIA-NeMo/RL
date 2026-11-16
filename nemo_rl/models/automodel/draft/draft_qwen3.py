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
#
# Extends nemo_automodel/components/speculative/dspark/draft_qwen3.py
# (3rdparty Automodel r0.6.0, byte-identical to vendor base 6f423feb0). Kept
# as a local module because the RL deltas are interleaved through forward();
# byte-identical helpers are imported from nemo_automodel instead.
# Local modifications:
# forward() additionally accepts an optional teacher_logits tensor [B, S, V].
# When provided, aligned_target_logits are gathered from it (detached) at the
# block prediction positions instead of being recomputed through this model's
# lm_head from target_last_hidden_states. RL co-training trains the draft
# lm_head, so distilling against the policy's own raw logits keeps the teacher
# distribution well-defined rather than self-referenced.
from typing import Callable, Optional

import torch
from nemo_automodel.components.attention.dflash_mask import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from nemo_automodel.components.speculative.dspark._sampling import sample_tokens
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from nemo_rl.models.automodel.draft.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    context_doc_ids,
    create_noise_embed,
    create_position_ids,
    pin_rope_inv_freq_fp32,
    sample_anchor_positions,
)
from nemo_rl.models.automodel.draft.markov_head import build_markov_head


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DSparkAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        if (
            self.config._attn_implementation == "flex_attention"
            and self.num_key_value_groups > 1
        ):
            kv_seq_len = k.shape[-2]
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            k = k.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
            v = v.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_is_causal = bool(kwargs.get("is_causal", False))
        # The SDPA path may consult module.is_causal when dispatching kernels,
        # so keep the per-call value mirrored on the module before invoking it.
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output), attn_weights


class Qwen3DSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3DSparkModel(Qwen3PreTrainedModel):
    _no_split_modules = ["Qwen3DSparkDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), (
                "config.markov_head_type must be provided when markov_rank > 0."
            )
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when enable_confidence_head is true."
            )
        self.target_layer_ids = config.target_layer_ids

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DSparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Reduced-draft-vocab checkpoints (e.g. RedHat speculators): lm_head
        # predicts over draft_vocab_size tokens; d2t maps draft index ->
        # target token id and t2d marks target-vocab membership. Both load
        # from the checkpoint as persistent buffers. Full-vocab checkpoints
        # (draft_vocab_size unset) keep the original behavior.
        self.draft_vocab_size = getattr(config, "draft_vocab_size", None)
        if self.draft_vocab_size is not None:
            self.draft_vocab_size = int(self.draft_vocab_size)
            self.register_buffer(
                "d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long)
            )
            self.register_buffer(
                "t2d", torch.zeros(config.vocab_size, dtype=torch.bool)
            )
            self._t2d_index: Optional[torch.Tensor] = None
        self.lm_head = nn.Linear(
            config.hidden_size,
            self.draft_vocab_size
            if self.draft_vocab_size is not None
            else config.vocab_size,
            bias=False,
        )
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)
        # Supervision layout. True (dspark, and the flat deepseek checkpoints
        # this trainer was vendored for): every block slot predicts the NEXT
        # token (slot at position p emits the distribution for p + 1). False
        # (speculators dflash): the anchor slot is an unsupervised bonus token
        # and each mask slot predicts the token AT its own position, matching
        # vLLM's dflash speculator (``sample_pos = query_pos``).
        self.sample_from_anchor = bool(getattr(config, "sample_from_anchor", True))

        # Markov head.
        self.markov_head = build_markov_head(config)

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.post_init()

    def _get_d2t_target_ids(self) -> torch.Tensor:
        """Absolute target token id for each draft index.

        The checkpoint's ``d2t`` buffer stores OFFSETS (eagle3/speculators
        convention): ``target_id = draft_idx + d2t[draft_idx]`` — the same
        interpretation as vLLM's ``arange(draft_vocab) + d2t`` scatter index.
        """
        assert self.draft_vocab_size is not None
        return (
            torch.arange(self.d2t.numel(), device=self.d2t.device, dtype=torch.long)
            + self.d2t
        )

    def _get_t2d_index(self) -> torch.Tensor:
        """Target token id -> draft index, -1 when absent.

        Built lazily so it reflects the checkpoint-loaded ``d2t`` buffer
        (from_pretrained fills buffers after ``__init__``).
        """
        assert self.draft_vocab_size is not None
        if self._t2d_index is None or self._t2d_index.device != self.d2t.device:
            target_ids = self._get_d2t_target_ids()
            index = torch.full(
                (self.config.vocab_size,),
                -1,
                dtype=torch.long,
                device=self.d2t.device,
            )
            index[target_ids] = torch.arange(target_ids.numel(), device=self.d2t.device)
            self._t2d_index = index
        return self._t2d_index

    def _apply(self, fn, recurse: bool = True):
        """Keep the RoPE ``inv_freq`` buffer in fp32 across dtype casts.

        ``model.to(bfloat16)`` (the training build path) would otherwise round
        ``inv_freq`` to bf16 and dephase RoPE with absolute position, eroding
        draft acceptance (see ``pin_rope_inv_freq_fp32``).
        """
        module = super()._apply(fn, recurse=recurse)
        pin_rope_inv_freq_fp32(getattr(self, "rotary_emb", None))
        return module

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2, (
            f"sample_draft_token_step expects base_logits shaped [batch, vocab], got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        doc_remaining: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        anchor_generator: Optional[torch.Generator] = None,
    ) -> DSparkForwardOutput:
        """Run one DSpark training forward.

        Sequence packing (``position_ids`` ``[B, S]`` per-document reset positions,
        ``seq_lens`` ``[B, max_docs]``, ``doc_remaining`` ``[B, S]``) keeps every block
        inside its anchor's document: the anchor's first target must be in-document,
        the block's context prefix and supervision are restricted to that document,
        and the draft's RoPE uses the per-document positions.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        packed = seq_lens is not None

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
            doc_remaining=doc_remaining if packed else None,
            generator=anchor_generator,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        if packed:
            context_position_ids = position_ids
            ctx_doc_id = context_doc_ids(seq_lens, seq_len, device)
            anchor_doc_id = torch.gather(ctx_doc_id, 1, anchor_positions)
        else:
            context_position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            )
            ctx_doc_id = None
            anchor_doc_id = None
        draft_position_ids = create_position_ids(
            anchor_positions, self.block_size, context_position_ids if packed else None
        )
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        if self.config._attn_implementation == "flex_attention":
            dspark_attn_mask = create_dflash_block_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
                ctx_doc_id=ctx_doc_id,
                anchor_doc_id=anchor_doc_id,
            )
        else:
            dspark_attn_mask = create_dflash_sdpa_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
                noise_embedding.dtype,
                ctx_doc_id=ctx_doc_id,
                anchor_doc_id=anchor_doc_id,
            )
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)

        # Next-token layout (dspark): slot k is supervised on the token at
        # anchor + k + 1. At-position layout (dflash): slot k is supervised on
        # the token at anchor + k, with the anchor slot (k = 0) unsupervised.
        first_label_offset = 1 if self.sample_from_anchor else 0
        label_offsets = torch.arange(
            first_label_offset, first_label_offset + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        # Markov prev-token embeddings index the FULL target vocab, so keep the
        # unmapped ids for prev_token_ids even under a reduced draft vocab.
        prev_source_token_ids = target_ids
        in_draft_vocab = None
        if self.draft_vocab_size is not None:
            # CE labels must be draft-vocab indices; block positions whose
            # target token has no draft-vocab entry are masked out of the loss.
            t2d_index = self._get_t2d_index()
            mapped_target_ids = t2d_index[target_ids]
            in_draft_vocab = mapped_target_ids >= 0
            target_ids = mapped_target_ids.clamp(min=0)
        aligned_target_logits = None
        if teacher_logits is not None:
            # Teacher distribution comes from the target/policy model's own raw
            # logits at the position that predicts each block token (offset k of
            # anchor a is predicted at position a + k). Under a reduced draft
            # vocab the caller pre-maps the teacher to draft-vocab order
            # (teacher_logits[..., d2t]) so the vocab dims line up here.
            assert teacher_logits.size(-1) == self.lm_head.out_features, (
                f"teacher_logits vocab dim {teacher_logits.size(-1)} does not "
                f"match the draft lm_head vocab {self.lm_head.out_features}; "
                "reduced-vocab drafts require the teacher pre-mapped via d2t."
            )
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_logits = torch.gather(
                teacher_logits.detach()
                .unsqueeze(1)
                .expand(-1, anchor_positions.size(1), -1, -1),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1, -1, -1, teacher_logits.size(-1)
                ),
            )
        elif target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
            doc_remaining=doc_remaining if packed else None,
            anchor_positions=anchor_positions if packed else None,
            supervised_from_slot=0 if self.sample_from_anchor else 1,
        )
        if in_draft_vocab is not None:
            eval_mask = eval_mask & in_draft_vocab
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), prev_source_token_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(
                    prev_token_ids
                ).to(dtype=output_hidden_4d.dtype)
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
            first_supervised_slot=0 if self.sample_from_anchor else 1,
        )


__all__ = [
    "Qwen3DSparkModel",
    "Qwen3DSparkAttention",
    "Qwen3DSparkDecoderLayer",
]
