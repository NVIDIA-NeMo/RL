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
# Vendored from vllm-project/speculators @ 0b08a89
# Sources: src/speculators/models/eagle3/{core.py, model_definitions.py,
#          metrics.py, attention.py}, src/speculators/losses/{eager.py, utils.py},
#          src/speculators/models/metrics.py
# Local modifications:
# - Self-contained single file: only the Qwen3 EAGLE3 drafter and the pieces of
#   the training forward it needs are vendored (no speculators framework base
#   classes, registries, or fused-kernel loss variants; the eager kl_div loss
#   is used).
# - The teacher distribution comes from a ``teacher_logits`` tensor
#   [1, T, draft_vocab] supplied by the caller (the RL runtime stashes the
#   policy's raw logits and pre-maps them into draft-vocab order via the d2t
#   offsets) instead of frozen verifier norm/lm_head copies. This is
#   mathematically the original construction, evaluated with the *current*
#   policy weights, which is exactly what RL co-training wants.
# - Dense SDPA attention masks replace flex-attention block masks (same mask
#   semantics: causal AND same-document, with per-TTT-step diagonal
#   extensions); the training sequences here are short enough that dense
#   masks are cheap and avoid the flex-attention dependency.
# - The embedding lookup in the TTT loop is not wrapped in torch.no_grad()
#   (the upstream trainer's no_grad silently blocks embedding gradients even
#   when embed_requires_grad is set); trainability is governed by
#   set_embedding_head_trainable() alone.
# - The forward returns per-step loss numerators/denominators (plus accuracy
#   counts) instead of a locally normalized scalar so the RL runtime can apply
#   NeMo-RL's global (DP-reduced, microbatch-slot) normalization. With a
#   single rank and one microbatch, num/den reproduce the original
#   speculators loss exactly (see tools/draft_verification/verify_eagle3_parity.py).
"""Vendored Qwen3 EAGLE3 drafter with speculators-style TTT training forward."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from nemo_rl.models.automodel.draft.common import pin_rope_inv_freq_fp32

_LOSS_REDUCTION_EPS = 1e-5


class _Eagle3FirstLayerMixin:
    """EAGLE3 first-layer modifications shared across decoder-layer families.

    The q/k/v projections take the 2*hidden concatenation of
    [token embeddings, fused aux hidden states]; the embeddings half is
    normalized by ``input_layernorm`` and the hidden half by ``hidden_norm``.
    With ``norm_before_residual`` the residual is the normalized hidden half.
    """

    def _patch_eagle3_projections(self, config, norm_class, norm_before_residual):
        self.norm_before_residual = norm_before_residual
        self.hidden_norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.self_attn.q_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.k_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.v_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden
        hidden_states = torch.cat([embeds, hidden], dim=-1)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Eagle3FirstLayer(_Eagle3FirstLayerMixin, Qwen3DecoderLayer):
    def __init__(self, config, layer_idx: int, norm_before_residual: bool):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, Qwen3RMSNorm, norm_before_residual)


class LlamaEagle3FirstLayer(_Eagle3FirstLayerMixin, LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int, norm_before_residual: bool):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, LlamaRMSNorm, norm_before_residual)


# Decoder-layer family per transformer_layer_config.model_type, mirroring the
# speculators model_classes registry AND the pinned vLLM's serving choice:
# speculators publishes qwen3-target eagle3 drafts with a LLAMA layer config
# (no q/k norms), and vLLM serves them through llama-style eagle3 layers, so
# the trainer must build the same family or it would stream q/k-norm weights
# the serving drafter does not have.
_EAGLE3_LAYER_FAMILIES = {
    "qwen3": (
        Qwen3Eagle3FirstLayer,
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    ),
    "llama": (
        LlamaEagle3FirstLayer,
        LlamaDecoderLayer,
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
    ),
}


@dataclass
class Eagle3ForwardTerms:
    """Per-TTT-step loss numerators/denominators and accuracy counts.

    Numerators carry gradients; denominators are detached scalars. The
    caller normalizes as sum_k decay^k * num_k / den_k with den_k optionally
    reduced across the data-parallel group.
    """

    loss_nums: list[torch.Tensor]
    loss_dens: list[torch.Tensor]
    full_acc_nums: list[torch.Tensor]
    full_acc_dens: list[torch.Tensor]
    cond_acc_nums: list[torch.Tensor]
    cond_acc_dens: list[torch.Tensor]


def _kl_div_per_position(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_p = F.softmax(targets, dim=-1, dtype=torch.float32)
    return F.kl_div(log_p, target_p, reduction="none", log_target=False).sum(dim=-1)


def _align_for_step(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    prev_correct: torch.Tensor,
    ttt_step: int,
):
    """Trim logits/targets/masks so step-k logits line up with their labels.

    Step-k logits at position t predict the token at t + 1 + k, so the
    targets (and loss mask) shift left by k while the last k logit positions
    have no labels. Mirrors the speculators ``align_for_step``.
    """
    if ttt_step > 0:
        logits = logits[:, :-ttt_step]
        prev_correct = prev_correct[:, :-ttt_step]
    targets = targets[:, ttt_step:]
    loss_mask = loss_mask[:, ttt_step:]
    return logits, targets, loss_mask, prev_correct


class Qwen3Eagle3DraftModel(Qwen3PreTrainedModel):
    """EAGLE3 drafter over Qwen3 layers with the TTT training forward.

    Weight layout matches the speculators/RedHat checkpoints: full-vocab
    ``embed_tokens``, ``fc`` fusing num_aux target hidden states, ``layers.0``
    with 2H q/k/v projections and ``hidden_norm``, reduced-vocab ``lm_head``,
    and the ``d2t`` offset / ``t2d`` membership vocab maps. The decoder-layer
    family follows ``config.model_type`` (see _EAGLE3_LAYER_FAMILIES).
    """

    _no_split_modules = [
        "Qwen3Eagle3FirstLayer",
        "Qwen3DecoderLayer",
        "LlamaEagle3FirstLayer",
        "LlamaDecoderLayer",
    ]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "draft_vocab_size",
            "norm_before_residual",
        )
        for field_name in required_fields:
            assert hasattr(config, field_name), f"config.{field_name} must be provided."

        self.target_layer_ids = [int(i) for i in config.target_layer_ids]
        self.draft_vocab_size = int(config.draft_vocab_size)
        num_aux = len(self.target_layer_ids)

        if config.model_type not in _EAGLE3_LAYER_FAMILIES:
            raise ValueError(
                f"Unsupported eagle3 layer family {config.model_type!r}; "
                f"supported: {sorted(_EAGLE3_LAYER_FAMILIES)}."
            )
        first_layer_class, decoder_layer_class, norm_class, rotary_class = (
            _EAGLE3_LAYER_FAMILIES[config.model_type]
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.fc = nn.Linear(
            num_aux * config.hidden_size, config.hidden_size, bias=False
        )
        layers: list[nn.Module] = [
            first_layer_class(
                config, layer_idx=0, norm_before_residual=config.norm_before_residual
            )
        ]
        layers.extend(
            decoder_layer_class(config, layer_idx)
            for layer_idx in range(1, config.num_hidden_layers)
        )
        self.layers = nn.ModuleList(layers)
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_class(config)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        # d2t stores offsets (target_id = draft_idx + d2t[draft_idx]); t2d is a
        # target-vocab membership mask. Both load from the checkpoint.
        self.register_buffer(
            "d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long)
        )
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))
        self._t2d_index: Optional[torch.Tensor] = None

        self.post_init()

    def _apply(self, fn, recurse: bool = True):
        """Keep the RoPE ``inv_freq`` buffer in fp32 across dtype casts.

        ``model.to(bfloat16)`` (the training build path) would otherwise round
        ``inv_freq`` to bf16 and dephase RoPE against vLLM serving's fp32
        cache, eroding draft acceptance (same pin as the dspark drafter; see
        ``pin_rope_inv_freq_fp32``).
        """
        module = super()._apply(fn, recurse=recurse)
        pin_rope_inv_freq_fp32(getattr(self, "rotary_emb", None))
        return module

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def get_d2t_target_ids(self) -> torch.Tensor:
        """Absolute target token id for each draft index (offset convention)."""
        return (
            torch.arange(self.d2t.numel(), device=self.d2t.device, dtype=torch.long)
            + self.d2t
        )

    def _build_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Dense additive mask: causal AND same-document, padding excluded.

        Matches the speculators mask_mod semantics (documents_ids == -1 marks
        padding; queries in padding attend nowhere in the base block and only
        their own diagonal in draft-token extensions, which is harmless since
        their loss is masked).
        """
        doc = document_ids.view(-1)
        idx = torch.arange(total_seq_len, device=device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        same_doc = (doc.unsqueeze(1) == doc.unsqueeze(0)) & (doc.unsqueeze(1) != -1)
        allowed = causal & same_doc
        mask = torch.zeros(
            (1, 1, total_seq_len, total_seq_len), dtype=dtype, device=device
        )
        mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
        return mask

    @staticmethod
    def _extend_mask_for_draft_tokens(
        mask: torch.Tensor, total_seq_len: int
    ) -> torch.Tensor:
        """Append a diagonal draft-token block (kv % T == q), speculators-style."""
        idx = torch.arange(total_seq_len, device=mask.device)
        diag = idx.unsqueeze(1) == idx.unsqueeze(0)
        block = torch.zeros(
            (1, 1, total_seq_len, total_seq_len), dtype=mask.dtype, device=mask.device
        )
        block.masked_fill_(~diag.unsqueeze(0).unsqueeze(0), torch.finfo(mask.dtype).min)
        return torch.cat([mask, block], dim=-1)

    def forward(
        self,
        fused_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_logits: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        ttt_steps: int = 3,
    ) -> Eagle3ForwardTerms:
        """TTT training forward over one packed row.

        Args:
            fused_hidden_states: [1, T, num_aux * H] concatenated aux hiddens.
            input_ids: [1, T] rollout token ids (target vocab).
            document_ids: [1, T] per-position document index, -1 for padding.
            loss_mask: [1, T] boolean positions that carry loss.
            teacher_logits: [1, T, draft_vocab] policy logits pre-mapped into
                draft-vocab order (detached by the caller).
            position_ids: [1, T]; defaults to the speculators 1 + arange.
            ttt_steps: unroll depth.
        """
        device = fused_hidden_states.device
        total_seq_len = fused_hidden_states.shape[1]
        assert teacher_logits.size(-1) == self.draft_vocab_size, (
            f"teacher_logits vocab dim {teacher_logits.size(-1)} != draft vocab "
            f"{self.draft_vocab_size}; pre-map the teacher via the d2t offsets."
        )

        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        past_key_values = DynamicCache()
        dtype = self.fc.weight.dtype
        attn_mask = self._build_attention_mask(
            document_ids, total_seq_len, dtype, device
        )

        hidden_states = self.fc(fused_hidden_states)

        original_input_ids = input_ids.detach().clone()
        targets = teacher_logits.detach()
        loss_mask = loss_mask.to(torch.bool)
        prev_correct = loss_mask.clone()

        terms = Eagle3ForwardTerms([], [], [], [], [], [])
        for ttt_step in range(ttt_steps):
            # Unlike the speculators trainer, the embedding lookup is NOT
            # wrapped in torch.no_grad(): train_embed_and_head must actually
            # train the embedding table, so trainability is controlled solely
            # by requires_grad via set_embedding_head_trainable(). Forward
            # values are identical either way (verified by the parity script).
            input_embeds = self.embed_tokens(input_ids)
            cache_position = torch.arange(
                ttt_step * total_seq_len,
                (ttt_step + 1) * total_seq_len,
                dtype=torch.long,
                device=device,
            )
            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            logits = self.lm_head(self.norm(hidden_states))

            s_logits, s_targets, s_mask, s_prev = _align_for_step(
                logits, targets, loss_mask, prev_correct, ttt_step
            )
            elementwise = _kl_div_per_position(s_logits, s_targets)
            mask_f = s_mask.to(elementwise.dtype)
            loss_num = (elementwise * mask_f).sum(dim=1)
            loss_den = mask_f.sum(dim=1).detach() + _LOSS_REDUCTION_EPS
            terms.loss_nums.append(loss_num.sum())
            terms.loss_dens.append(loss_den.sum())

            with torch.no_grad():
                pred_ids = torch.argmax(s_logits, dim=-1)
                target_ids = torch.argmax(s_targets, dim=-1)
                correct = pred_ids == target_ids
                cond_total = s_prev.sum().float()
                # s_prev is a view of prev_correct, so the in-place AND
                # persists into the next step (speculators semantics).
                correct = torch.logical_and(s_prev, correct, out=s_prev)
                masked_correct = torch.masked_select(correct, s_mask)
                correct_sum = masked_correct.float().sum()
                terms.full_acc_nums.append(correct_sum)
                terms.full_acc_dens.append(
                    torch.tensor(float(masked_correct.numel()), device=device)
                )
                terms.cond_acc_nums.append(correct_sum.clone())
                terms.cond_acc_dens.append(cond_total)

            # Teacher forcing: next step consumes ground-truth ids shifted
            # left by 1 + ttt_step (right-padded with 0); the tail positions
            # fall out of the loss via _align_for_step, matching the
            # speculators trainer's global-shift semantics (shifts may cross
            # packed-document boundaries there too).
            input_ids = torch.cat(
                [
                    original_input_ids[:, 1 + ttt_step :],
                    original_input_ids.new_zeros(1, 1 + ttt_step),
                ],
                dim=-1,
            )
            attn_mask = self._extend_mask_for_draft_tokens(attn_mask, total_seq_len)
            position_ids = position_ids + 1

        return terms
