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
# Built on nemo_automodel.components.speculative.eagle.draft_llama
# (LlamaEagle3DraftModel / Eagle3LlamaAttention / Eagle3LlamaDecoderLayer):
# the actual drafter network (attention, decoder layer, embedding/fc/lm_head)
# is imported unchanged, not vendored. That module's TTT attention is
# meaningfully better than what used to be hand-rolled here: step 0's causal
# block runs once (dense [T, T], optionally FlashAttention-2) and every later
# TTT step attends only to its OWN previous-step K/V at the SAME position (an
# O(T) einsum "diagonal" via a shared `cache_hidden` list) instead of
# re-attending against the whole growing KV cache (an O(T^2 * step) matmul
# per step) the way the previous from-scratch implementation did.
#
# What lives in THIS file are the RL-specific deltas that don't fit
# automodel's `Eagle3TrainerModule` (core.py) API:
# - Per-step loss numerators/denominators (plus accuracy counts) instead of
#   `Eagle3TrainerModule`'s pre-averaged scalar loss/accuracy, so NeMo-RL's
#   runtime can apply its own DP-group-reduced, microbatch-slot-normalized
#   combination (`Eagle3Runtime.compute_loss`) instead of the plain
#   CP-only-reduced mean `Eagle3TrainerModule.forward` computes.
# - The document-packed dense attention mask (`_build_attention_mask`): the
#   RL runtime packs a whole training microbatch (multiple rollout sequences)
#   into ONE row via `document_ids`, which needs to be threaded straight into
#   the attention layer's `attention_mask` argument; `LlamaEagle3DraftModel`'s
#   own `forward()` wrapper only builds a mask from a 2D padding mask or a
#   `seq_lens` packing path, so this file drives the model's per-piece public
#   API (`embed_input_ids` / `project_hidden_states` / `model.layers[0]` /
#   `compute_logits`) directly instead of going through that wrapper.
# - Chunked `compute_logits` + KL along the sequence, to avoid materializing
#   the full ``[1, T, draft_vocab]`` logits tensor at once (~2.4 GiB at
#   20k-token sequences with a 64000 draft vocab -- see `_kl_div_per_position`).
"""EAGLE3 TTT training forward, built on automodel's LlamaEagle3DraftModel."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.speculative.eagle.draft_llama import (
    Eagle3LlamaDecoderLayer,
    LlamaEagle3DraftModel,
    _seq_lens_to_cu_seqlens,
)

_LOSS_REDUCTION_EPS = 1e-5
_KL_CHUNK_TOKENS = 512


class _Eagle3LlamaDecoderLayerWithResidualToggle(Eagle3LlamaDecoderLayer):
    """Restores the ``norm_before_residual`` toggle automodel's layer lacks.

    ``Eagle3LlamaDecoderLayer.forward`` always uses the PRE-normalization
    hidden state as the residual (``residual = hidden_states`` before
    ``hidden_norm`` runs) -- equivalent to ``norm_before_residual=False``,
    with no toggle at all. Speculators-format checkpoints (e.g.
    RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3) explicitly
    declare ``norm_before_residual: true``; loading them through the
    unmodified automodel layer would silently use the wrong residual
    convention. Only ``forward`` differs from the parent; state dict keys
    and shapes are unchanged.
    """

    def __init__(self, config, layer_id: int = 0):
        super().__init__(config, layer_id=layer_id)
        self.norm_before_residual = bool(getattr(config, "norm_before_residual", False))

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: list[list[torch.Tensor]],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_input_embeds = self.input_layernorm(input_embeds)
        norm_hidden_states = self.hidden_norm(hidden_states)
        if self.norm_before_residual:
            residual = norm_hidden_states
        combined_states = torch.cat((norm_input_embeds, norm_hidden_states), dim=-1)
        hidden_states = residual + self.self_attn(
            combined_states,
            attention_mask,
            position_ids,
            cache_hidden=cache_hidden,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


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


def _kl_div_per_position_chunk(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_p = F.softmax(targets, dim=-1, dtype=torch.float32)
    return F.kl_div(log_p, target_p, reduction="none", log_target=False).sum(dim=-1)


def _kl_div_per_position(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-position KL over the draft vocab, chunked along the sequence.

    Each position's KL is independent, so chunking is mathematically exact.
    The fp32 ``[*, chunk, V]`` transients (log_softmax, softmax, elementwise
    kl) would otherwise materialize for the FULL row at once -- at 18k-token
    generations with a 32000 draft vocab that is ~2.3 GiB per tensor per TTT
    step, which OOMs the tp=2 co-training layout. Checkpointing recomputes
    the transients per chunk in backward (same pattern as the dspark loss's
    chunked fp32 probability distance).
    """
    seq_len = logits.size(1)
    if seq_len <= _KL_CHUNK_TOKENS:
        return _kl_div_per_position_chunk(logits, targets)
    pieces = []
    for start in range(0, seq_len, _KL_CHUNK_TOKENS):
        logit_chunk = logits[:, start : start + _KL_CHUNK_TOKENS]
        target_chunk = targets[:, start : start + _KL_CHUNK_TOKENS]
        if logits.requires_grad:
            piece = checkpoint(
                _kl_div_per_position_chunk,
                logit_chunk,
                target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            piece = _kl_div_per_position_chunk(logit_chunk, target_chunk)
        pieces.append(piece)
    return torch.cat(pieces, dim=1)


def _build_attention_mask(
    document_ids: torch.Tensor,
    total_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Dense additive mask: causal AND same-document, padding excluded.

    Used only for the TTT loop's step-0 causal block (``Eagle3LlamaAttention``
    keeps this mask constant across TTT steps -- later steps attend only to
    their own previous-step K/V at the same position, which needs no mask).
    documents_ids == -1 marks padding; queries in padding attend nowhere.
    """
    doc = document_ids.view(-1)
    idx = torch.arange(total_seq_len, device=device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
    same_doc = (doc.unsqueeze(1) == doc.unsqueeze(0)) & (doc.unsqueeze(1) != -1)
    allowed = causal & same_doc
    mask = torch.zeros((1, 1, total_seq_len, total_seq_len), dtype=dtype, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
    return mask


class Eagle3DraftModel(LlamaEagle3DraftModel):
    """``LlamaEagle3DraftModel`` with an RL-shaped TTT ``forward``.

    Everything except ``forward`` (embedding/fc/lm_head, ``d2t``/``t2d``,
    ``set_vocab_mapping``, ``copy_embeddings_from_target``) is inherited
    unchanged from automodel. ``model.layers[0]`` is swapped for
    :class:`_Eagle3LlamaDecoderLayerWithResidualToggle` right after
    construction (before any weights load) -- same parameter names/shapes,
    so checkpoints round-trip unchanged.

    ``forward`` runs the WHOLE TTT unroll as a single top-level call
    (matching the parent's calling convention) rather than driving
    ``embed_input_ids`` / ``project_hidden_states`` / ``compute_logits`` as
    separate calls: FSDP2's ``fully_shard`` unshard/reshard hooks fire on
    ``nn.Module.__call__``, and only the top-level model and each decoder
    layer are individually ``fully_shard``-ed (see
    ``build_eagle3_draft_model``) -- ``embed_tokens``/``fc``/``lm_head``/
    ``norm`` are NOT, so calling their owning methods directly (bypassing
    ``draft_model(...)``) would run them against still-sharded ``DTensor``
    parameters.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.model.layers[0] = _Eagle3LlamaDecoderLayerWithResidualToggle(
            config, layer_id=0
        )
        # Redundant with config.target_layer_ids, but Eagle3Runtime reads it
        # off the model directly (mirrors DSparkRuntime's
        # config.target_layer_ids access pattern).
        self.target_layer_ids = [int(i) for i in config.target_layer_ids]

    def set_embedding_head_trainable(self, trainable: bool) -> None:
        self.model.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def get_d2t_target_ids(self) -> torch.Tensor:
        """Absolute target token id for each draft index (offset convention)."""
        return (
            torch.arange(self.d2t.numel(), device=self.d2t.device, dtype=torch.long)
            + self.d2t
        )

    def forward(
        self,
        fused_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_logits: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        ttt_steps: int = 3,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Eagle3ForwardTerms:
        """TTT training forward over one packed row.

        Args:
            fused_hidden_states: [1, T, num_aux * H_target] concatenated aux hiddens.
            input_ids: [1, T] rollout token ids (target vocab).
            document_ids: [1, T] per-position document index, -1 for padding.
            loss_mask: [1, T] boolean positions that carry loss.
            teacher_logits: [1, T, draft_vocab] policy logits pre-mapped into
                draft-vocab order (detached by the caller).
            position_ids: [1, T]; defaults to the speculators ``1 + arange``.
            ttt_steps: unroll depth.
            seq_lens: [1, batch_size] per-sample block width (each entry the
                fixed ``seq_len`` every sample's slot occupies in the packed
                row, INCLUDING that sample's trailing padding -- see
                ``Eagle3Runtime.compute_loss``). Required when
                ``attn_implementation="flash_attention_2"``, unused
                otherwise. Declaring each fixed-size block as one
                FlashAttention-varlen "document" is safe even though
                padding sits inside the block: padding is placed at each
                block's TAIL, so causal masking already keeps every real
                token from attending to any padding position (padding
                comes chronologically after), and padding positions'
                own outputs never enter the loss (loss_mask excludes them).
        """
        device = fused_hidden_states.device
        total_seq_len = fused_hidden_states.shape[1]
        if teacher_logits.size(-1) != self.draft_vocab_size:
            raise ValueError(
                f"teacher_logits vocab dim {teacher_logits.size(-1)} != draft "
                f"vocab {self.draft_vocab_size}; pre-map the teacher via the "
                "d2t offsets."
            )

        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        dtype = self.model.fc.weight.dtype
        # Step-0's causal block stays CONSTANT across the whole TTT unroll --
        # later steps attend only to their own previous-step K/V at the same
        # position (see Eagle3LlamaAttention), which needs no mask at all.
        #
        # attn_implementation="flash_attention_2": build cu_seqlens/max_seqlen
        # instead of the dense [B, H, T, T] additive mask -- at H=32 heads
        # and a 10k-token packed row that dense mask is a ~12 GiB fp32
        # allocation inside eager attention's softmax (OOMs at full 4n8g
        # scale). FlashAttention-2's varlen kernel never materializes it.
        is_fa2 = (
            self.model.layers[0].self_attn.attn_implementation == "flash_attention_2"
        )
        if is_fa2:
            if seq_lens is None:
                raise ValueError(
                    "Eagle3DraftModel.forward requires seq_lens when "
                    "attn_implementation='flash_attention_2'."
                )
            attn_mask = None
            cu_seqlens, max_seqlen = _seq_lens_to_cu_seqlens(
                seq_lens, seq_length=total_seq_len
            )
        else:
            attn_mask = _build_attention_mask(document_ids, total_seq_len, dtype, device)
            cu_seqlens = None
            max_seqlen = None

        hidden_states = self.project_hidden_states(fused_hidden_states.to(dtype))

        original_input_ids = input_ids.detach().clone()
        targets = teacher_logits.detach()
        loss_mask = loss_mask.to(torch.bool)
        prev_correct = loss_mask.clone()

        # Shared K/V cache the attention layer appends to every TTT step; see
        # Eagle3LlamaAttention's docstring for the mixed causal/diagonal
        # pattern this drives.
        cache_hidden: list[list[torch.Tensor]] = [[], []]

        terms = Eagle3ForwardTerms([], [], [], [], [], [])
        for ttt_step in range(ttt_steps):
            # embed_input_ids is NOT wrapped in torch.no_grad():
            # train_embed_and_head must actually train the embedding table,
            # so trainability is controlled solely by requires_grad via
            # set_embedding_head_trainable().
            input_embeds = self.embed_input_ids(input_ids)
            hidden_states = self.model.layers[0](
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            # Step-k logits at position t predict the token at t + 1 + k, so
            # targets/loss_mask shift left by k while the last k positions
            # have no label (mirrors the speculators align_for_step).
            #
            # Chunked along the sequence: compute_logits is per-position
            # (RMSNorm has no cross-position stats, the Linear is
            # pointwise), so slicing hidden_states before it is identical to
            # slicing the resulting logits afterward, but never materializes
            # the full [1, T, draft_vocab] logits tensor -- at 20k-token
            # sequences with a 64000 draft vocab that tensor alone is ~2.4
            # GiB per TTT step, which OOMs at this model's memory budget.
            effective_len = total_seq_len - ttt_step
            hidden_for_logits = (
                hidden_states[:, :effective_len] if ttt_step > 0 else hidden_states
            )
            targets_for_step = targets[:, ttt_step:]
            mask_for_step = loss_mask[:, ttt_step:]
            prev_for_step = (
                prev_correct[:, :effective_len] if ttt_step > 0 else prev_correct
            )

            loss_num = hidden_for_logits.new_zeros(())
            loss_den = hidden_for_logits.new_zeros(())
            correct_sum = hidden_for_logits.new_zeros(())
            correct_count = hidden_for_logits.new_zeros(())
            cond_total = hidden_for_logits.new_zeros(())
            for start in range(0, effective_len, _KL_CHUNK_TOKENS):
                end = start + _KL_CHUNK_TOKENS
                h_chunk = hidden_for_logits[:, start:end]
                t_chunk = targets_for_step[:, start:end]
                m_chunk = mask_for_step[:, start:end]
                # A view into prev_correct's storage: the in-place AND below
                # must persist into the next step (speculators semantics),
                # same as the pre-chunking s_prev view did.
                p_chunk = prev_for_step[:, start:end]

                logits_chunk = self.compute_logits(h_chunk)
                elementwise = _kl_div_per_position_chunk(logits_chunk, t_chunk)
                mask_f = m_chunk.to(elementwise.dtype)
                loss_num = loss_num + (elementwise * mask_f).sum()
                loss_den = loss_den + mask_f.sum().detach()

                with torch.no_grad():
                    pred_ids = torch.argmax(logits_chunk, dim=-1)
                    target_ids = torch.argmax(t_chunk, dim=-1)
                    correct = pred_ids == target_ids
                    cond_total = cond_total + p_chunk.sum().float()
                    correct = torch.logical_and(p_chunk, correct, out=p_chunk)
                    masked_correct = torch.masked_select(correct, m_chunk)
                    correct_sum = correct_sum + masked_correct.float().sum()
                    correct_count = correct_count + masked_correct.numel()

            terms.loss_nums.append(loss_num)
            terms.loss_dens.append(loss_den + _LOSS_REDUCTION_EPS)
            terms.full_acc_nums.append(correct_sum)
            terms.full_acc_dens.append(correct_count)
            terms.cond_acc_nums.append(correct_sum.clone())
            terms.cond_acc_dens.append(cond_total)

            if ttt_step + 1 < ttt_steps:
                # Teacher forcing: next step consumes ground-truth ids
                # shifted left by 1 + ttt_step (right-padded with 0); the
                # tail positions fall out of the loss via the alignment
                # slices above, matching the speculators trainer's
                # global-shift semantics (shifts may cross packed-document
                # boundaries there too).
                input_ids = torch.cat(
                    [
                        original_input_ids[:, 1 + ttt_step :],
                        original_input_ids.new_zeros(1, 1 + ttt_step),
                    ],
                    dim=-1,
                )
                # position_ids/attn_mask are NOT advanced here:
                # Eagle3LlamaAttention derives the rotary phase shift from
                # `len(cache_hidden[0])` (the TTT step index) internally,
                # and the step-0 causal block (attn_mask) never changes
                # across steps.

        return terms
