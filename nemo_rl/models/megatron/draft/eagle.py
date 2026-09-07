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

"""Megatron training implementation of the EAGLE-3 draft model.

This file wraps ModelOpt's ``EagleModule`` and adds two training paths:

- :meth:`EagleModel.forward` runs one causal draft pass when ``ttt_steps == 1``.
- :meth:`EagleModel.forward_ttt` runs several sequential TTT passes.
  Each pass feeds its hidden state and shifted token embeddings into the next
  pass, matching the draft model's self-conditioning during serving.

For a query at anchor ``i`` in TTT pass ``d``, attention can read:

- the pass-1 trunk at positions ``0..i``; and
- the entry at anchor ``i`` from every pass ``2..d``.

It cannot read a later trunk position or another anchor's branch. This matches
the serving cache: causal prefill entries followed by one draft-generated entry
for each speculation depth. :class:`TwoPartTTTAttention` shows the full mask.

The causal trunk uses FlashAttention. The short per-anchor branch uses an
einsum. Their output and log-sum-exp values are merged to produce the exact
softmax over both sets of keys; backward uses the same joint values to produce
the exact gradients.

The implementation uses private FlashAttention functions because the merge
needs their log-sum-exp values. Their interface is pinned to the compatible
FlashAttention 2.x versions checked below.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Tuple

import flash_attn
import torch
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    flash_attn_func,
)
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)
from torch import Tensor

# These are private FlashAttention functions, so their Python signatures
# cannot be inspected reliably. This file uses the interface from version
# 2.8.1, which is compatible with FlashAttention 2.7 and later 2.x releases.
# Reject other versions early instead of failing with an unclear argument error.
_fa_major, _fa_minor = (int(x) for x in flash_attn.__version__.split(".")[:2])
if not (_fa_major == 2 and _fa_minor >= 7):
    raise RuntimeError(
        f"flash-attn {flash_attn.__version__} does not match the private "
        "interface vendored for flash-attn 2.8.1; update "
        "nemo_rl/models/megatron/draft/eagle.py."
    )


def _fa_forward_causal(
    q: Tensor, k: Tensor, v: Tensor, softmax_scale: float
) -> tuple[Tensor, Tensor]:
    """Run causal FlashAttention on batch-first tensors.

    Returns the output with shape ``[B, S, Hq, D]`` and fp32 log-sum-exp values
    with shape ``[B, Hq, S]``.
    """
    out, softmax_lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        0.0,  # dropout_p
        softmax_scale,
        True,  # causal
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # return_softmax
    )
    return out, softmax_lse


def _fa_backward_causal(
    *,
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    softmax_scale: float,
) -> None:
    """Run causal FlashAttention backward into existing buffers.

    ``out`` and ``lse`` describe attention over all merged key sets. Passing
    them to this call produces the exact part of that joint-softmax gradient
    that belongs to this call's keys.
    """
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        0.0,  # dropout_p
        softmax_scale,
        True,  # causal
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # deterministic
        None,  # rng_state
    )


class TwoPartTTTAttention(torch.autograd.Function):
    """Combine a causal pass-1 trunk with each anchor's own branch keys.

    In TTT pass ``d``, the query at anchor ``i`` can attend to:

    - pass-1 trunk keys at positions ``0..i``; and
    - the key at anchor ``i`` from every pass ``2..d``.

    It cannot attend to later trunk positions or branch keys from another
    anchor. Teacher forcing treats every sequence position as an independent
    anchor, so all anchors from one pass can be processed together.

    The example below has four anchors and three passes. ``q2@1`` means the
    query for anchor 1 during pass 2. ``A`` marks trunk attention computed by
    FlashAttention, ``x`` marks the small same-anchor branch computed by an
    fp32 einsum, and blank cells are masked. Each group of query rows is one
    :class:`TwoPartTTTAttention` call; pass 1 has no branch::

                 | pass-1 KV(trunk) |    pass-2 KV    |    pass-3 KV    |
       anchor j: |  0   1   2   3   |  0   1   2   3  |  0   1   2   3  |
       ==================================================================
       q1@0      |  A               |                 |                 |
       q1@1      |  A   A           |                 |                 |
       q1@2      |  A   A   A       |                 |                 |
       q1@3      |  A   A   A   A   |                 |                 |
       ------------------------------------------------------------------
       q2@0      |  A               |  x              |                 |
       q2@1      |  A   A           |      x          |                 |
       q2@2      |  A   A   A       |          x      |                 |
       q2@3      |  A   A   A   A   |              x  |                 |
       ------------------------------------------------------------------
       q3@0      |  A               |  x              |  x              |
       q3@1      |  A   A           |      x          |      x          |
       q3@2      |  A   A   A       |          x      |          x      |
       q3@3      |  A   A   A   A   |              x  |              x  |
       ==================================================================

    The RoPE positions follow the serving cache. For ``q_d@i``, the trunk uses
    positions ``0..i`` and branch keys use ``i+1..i+d-1``. The query and its
    current-pass key both use ``i+d-1``. Thus the cache contains the causal
    prefill plus the ``d - 1`` entries generated by this speculation chain.

    For pass ``d`` and anchor ``i``:

    - Input: ``concat(e(x_{i+d}), h^{d-1}_i)``. Here ``h^0`` is the projected
      target auxiliary state, and later ``h`` values are pre-norm draft states.
    - RoPE position: ``i + d - 1``.
    - Training target: ``x_{i+d+1}``.
    - Saved state: ``(k^d_i, v^d_i)``, which becomes a branch entry for the
      next pass.

    The trunk and branch are evaluated separately, then merged with their
    log-sum-exp values. This gives the exact softmax over the union of keys
    without creating an ``S x S`` score matrix. Backward passes the merged
    output and log-sum-exp to FlashAttention for the trunk and computes the
    branch gradient directly.

    Inputs to ``apply`` use the batch-first FlashAttention layout:

    - ``q``: Current-pass queries with shape ``[B, S, Hq, D]``.
    - ``k1`` and ``v1``: Pass-1 trunk K/V with shape ``[B, S, Hkv, D]``.
    - ``kb`` and ``vb``: Branch K/V with shape ``[B, S, Hkv, P, D]``. The branch
      axis stores passes ``2..d`` for the same anchor, so ``P = d - 1``.
    - ``softmax_scale``: Scale applied to attention scores.

    The returned tensor has shape ``[B, S, Hq, D]``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: Tensor,
        k1: Tensor,
        v1: Tensor,
        kb: Tensor,
        vb: Tensor,
        softmax_scale: float,
    ) -> Tensor:
        batch, seqlen, num_q_heads, head_dim = q.shape
        num_kv_heads = k1.shape[2]
        group = num_q_heads // num_kv_heads

        out_t, lse_t = _fa_forward_causal(q, k1, v1, softmax_scale)

        # Branch part in fp32: scores/probs over at most P keys per query.
        q_grouped = q.view(batch, seqlen, num_kv_heads, group, head_dim).float()
        # [B, S, Hkv, G, P]
        scores_b = (
            torch.einsum("bskgd,bskpd->bskgp", q_grouped, kb.float()) * softmax_scale
        )
        lse_b = torch.logsumexp(scores_b, dim=-1)  # [B, S, Hkv, G]
        probs_b = torch.exp(scores_b - lse_b.unsqueeze(-1))
        out_b = torch.einsum("bskgp,bskpd->bskgd", probs_b, vb.float())

        # Exact LSE merge of the two disjoint key sets.
        lse_t_grouped = (
            lse_t.permute(0, 2, 1).view(batch, seqlen, num_kv_heads, group).float()
        )
        lse_joint = torch.logaddexp(lse_t_grouped, lse_b)  # [B, S, Hkv, G]
        w_t = torch.exp(lse_t_grouped - lse_joint).unsqueeze(-1)
        w_b = torch.exp(lse_b - lse_joint).unsqueeze(-1)
        out_f32 = (
            w_t * out_t.view(batch, seqlen, num_kv_heads, group, head_dim).float()
            + w_b * out_b
        )
        out = out_f32.view(batch, seqlen, num_q_heads, head_dim).to(q.dtype)

        # FlashAttention backward expects lse as [B, Hq, S] fp32.
        lse_joint_hbs = (
            lse_joint.view(batch, seqlen, num_q_heads).permute(0, 2, 1).contiguous()
        )

        ctx.save_for_backward(q, k1, v1, kb, vb, out, lse_joint_hbs)
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, dout: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        q, k1, v1, kb, vb, out, lse_joint_hbs = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        batch, seqlen, num_q_heads, head_dim = q.shape
        num_kv_heads = k1.shape[2]
        group = num_q_heads // num_kv_heads

        dout = dout.contiguous()

        # Trunk gradient: FA backward with the joint (out, lse) returns the
        # exact joint-softmax gradient restricted to the trunk key set.
        dq_t = torch.empty_like(q)
        dk1 = torch.empty_like(k1)
        dv1 = torch.empty_like(v1)
        _fa_backward_causal(
            dout=dout,
            q=q,
            k=k1,
            v=v1,
            out=out,
            lse=lse_joint_hbs,
            dq=dq_t,
            dk=dk1,
            dv=dv1,
            softmax_scale=softmax_scale,
        )

        # Branch gradient in closed form (fp32): probs are joint-normalized,
        # and the D_i = rowsum(dout * out_joint) term carries the cross-part
        # softmax correction (same identity the FA kernel applies internally).
        q_grouped = q.view(batch, seqlen, num_kv_heads, group, head_dim).float()
        scores_b = (
            torch.einsum("bskgd,bskpd->bskgp", q_grouped, kb.float()) * softmax_scale
        )
        lse_joint_grouped = (
            lse_joint_hbs.permute(0, 2, 1)
            .reshape(batch, seqlen, num_kv_heads, group)
            .float()
        )
        probs_bj = torch.exp(scores_b - lse_joint_grouped.unsqueeze(-1))

        dout_f32 = dout.float()
        d_row = (
            (dout_f32 * out.float())
            .sum(dim=-1)
            .view(batch, seqlen, num_kv_heads, group)
        )
        dout_grouped = dout_f32.view(batch, seqlen, num_kv_heads, group, head_dim)

        dvb = torch.einsum("bskgp,bskgd->bskpd", probs_bj, dout_grouped)
        d_probs = torch.einsum("bskgd,bskpd->bskgp", dout_grouped, vb.float())
        d_scores = probs_bj * (d_probs - d_row.unsqueeze(-1)) * softmax_scale
        dq_b = torch.einsum("bskgp,bskpd->bskgd", d_scores, kb.float()).view(
            batch, seqlen, num_q_heads, head_dim
        )
        dkb = torch.einsum("bskgp,bskgd->bskpd", d_scores, q_grouped)

        dq = (dq_t.float() + dq_b).to(q.dtype)
        return dq, dk1, dv1, dkb.to(kb.dtype), dvb.to(vb.dtype), None


class TTTDraftCoreAttention(torch.nn.Module):
    """Drop-in ``core_attention`` for the draft decoder's multi-pass training.

    Replaces the layer's TE core attention (see ``EagleModel``); receives the
    post-RoPE q/k/v that MCore's ``SelfAttention.forward`` hands to
    ``core_attention`` in sbhd layout and returns ``[S, B, Hq * D]``.

    Statefulness contract: the TTT driver calls :meth:`begin_pass` before each
    draft decoder pass and :meth:`reset` when the multi-pass loop finishes (or
    aborts). Pass 1 runs plain causal FlashAttention and stashes its KV
    (non-detached — cross-pass reuse must carry gradients back to the pass-1
    projections); later passes stash their own KV as branch entries and run
    :class:`TwoPartTTTAttention` against the trunk. Because the stash holds
    the exact tensors fed to the kernels, a single backward over the combined
    loss accumulates every pass's contribution automatically.
    """

    def __init__(self, config: Any):
        super().__init__()
        attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        if attention_dropout != 0.0:
            raise ValueError(
                "TTTDraftCoreAttention does not support attention dropout "
                f"(got attention_dropout={attention_dropout})."
            )
        if int(getattr(config, "context_parallel_size", 1) or 1) != 1:
            raise ValueError(
                "TTTDraftCoreAttention requires context_parallel_size == 1."
            )
        # Match MCore's DotProductAttention convention: explicit config
        # softmax_scale, else 1/sqrt(head_dim).
        self.softmax_scale: Optional[float] = getattr(config, "softmax_scale", None)
        self._pass_idx = 0
        self._kv_by_pass: list[tuple[Tensor, Tensor]] = []

    def begin_pass(self, pass_idx: int) -> None:
        """Arm the module for TTT pass ``pass_idx`` (1-indexed)."""
        if pass_idx == 1:
            self._kv_by_pass = []
        elif pass_idx != self._pass_idx + 1:
            raise RuntimeError(
                f"TTT passes must run in order; got begin_pass({pass_idx}) "
                f"after pass {self._pass_idx}."
            )
        self._pass_idx = pass_idx

    def reset(self) -> None:
        """Clear saved K/V and release the cross-pass autograd graph."""
        self._pass_idx = 0
        self._kv_by_pass = []

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[Any] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[Any] = None,
    ) -> Tensor:
        if packed_seq_params is not None:
            raise NotImplementedError(
                "TTT draft training does not support sequence packing; "
                "disable policy.sequence_packing."
            )
        if self._pass_idx < 1:
            raise RuntimeError(
                "TTTDraftCoreAttention.forward called without begin_pass(); "
                "the TTT driver must arm the pass index before each pass."
            )

        seqlen, batch = query.shape[0], query.shape[1]
        softmax_scale = self.softmax_scale or query.shape[-1] ** -0.5

        # sbhd -> bshd (FlashAttention layout).
        q = query.transpose(0, 1).contiguous()
        k = key.transpose(0, 1).contiguous()
        v = value.transpose(0, 1).contiguous()

        self._kv_by_pass.append((k, v))

        if self._pass_idx == 1:
            out = flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=True)
        else:
            k1, v1 = self._kv_by_pass[0]
            kb = torch.stack([kv[0] for kv in self._kv_by_pass[1:]], dim=3)
            vb = torch.stack([kv[1] for kv in self._kv_by_pass[1:]], dim=3)
            out = TwoPartTTTAttention.apply(q, k1, v1, kb, vb, softmax_scale)

        # bshd -> [S, B, Hq * D] (core_attention output contract).
        return out.transpose(0, 1).reshape(seqlen, batch, -1)


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

        self._ttt_attn_modules: list[TTTDraftCoreAttention] = []
        self._ttt_prenorm_hidden: Optional[Tensor] = None
        if self.ttt_steps > 1:
            if getattr(config, "recompute_granularity", None) is not None:
                # Activation recomputation would call this stateful attention
                # again during backward and corrupt its per-pass K/V list.
                raise ValueError(
                    "TTT draft training (ttt_steps > 1) is incompatible with "
                    "activation recomputation on the draft config."
                )
            for layer in self.eagle_module.decoder.layers:
                ttt_attention = TTTDraftCoreAttention(config)
                layer.self_attention.core_attention = ttt_attention
                self._ttt_attn_modules.append(ttt_attention)
            # ModelOpt's hook saves a detached value. TTT instead needs the
            # pre-final-norm hidden state with gradients for the next pass.
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

    @contextmanager
    def _thd_attention_mask_type(self):
        """Temporarily run the eagle layers with a padding_causal mask type.

        modelopt builds the eagle decoder with ``AttnMaskType.arbitrary`` (its
        own multi-step training manipulates masks explicitly), but TE's THD
        kernels only accept ``padding``/``padding_causal``, and mcore's
        auto-conversion covers only ``causal``/``no_mask``. The packed draft
        forward wants plain block-diagonal causal attention, which is exactly
        ``padding_causal`` + ``cu_seqlens``.
        """
        layers = self.eagle_module.decoder.layers
        original_mask_types = [layer.self_attention.attn_mask_type for layer in layers]
        for layer in layers:
            layer.self_attention.attn_mask_type = AttnMaskType.padding_causal
        try:
            yield
        finally:
            for layer, mask_type in zip(layers, original_mask_types):
                layer.self_attention.attn_mask_type = mask_type

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        bootstrap_hidden_states: bool = True,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        if bootstrap_hidden_states:
            hidden_states = self.eagle_module.fc(hidden_states)[0]
        elif hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden states with size {self.config.hidden_size} when "
                f"`bootstrap_hidden_states=False`, got {hidden_states.shape[-1]}."
            )

        # packed_seq_params drives the draft decoder's THD attention (segment
        # boundaries + per-sequence RoPE restarts) when the trainer runs with
        # sequence packing; None keeps the dense [s, b, h] path.
        mask_type_ctx = (
            self._thd_attention_mask_type()
            if packed_seq_params is not None
            else nullcontext()
        )
        with mask_type_ctx:
            hidden_states, _ = self.eagle_module(
                embeddings=input_embeds,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
        logits, _ = self.eagle_module.eagle_output_layer(hidden_states)
        logits = logits.transpose(0, 1).contiguous()
        return logits

    def _ttt_rotary_pos_emb(self, ttt_pass: int, seq_len: int) -> Optional[Tensor]:
        """Build RoPE frequencies for one TTT pass.

        Pass ``d`` shifts every position by ``d - 1``. A query at anchor ``i``
        therefore uses position ``i + d - 1``, matching the extra cache entry
        created at each speculation depth during serving. Pass 1 returns
        ``None`` because EagleModule's default table is already correct.

        Args:
            ttt_pass: One-indexed TTT pass ``d``.
            seq_len: Length of the input sequence.

        Returns:
            Shifted RoPE frequencies for this pass, or ``None`` for pass 1.
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
        """Run the configured TTT passes in sequence.

        Pass 1 projects the target auxiliary hidden states. Every later pass
        receives the previous pass's pre-final-norm hidden state without
        detaching it, so gradients flow through the entire pass chain. Token
        embeddings shift left by one position before each new pass.

        Args:
            hidden_states: Target auxiliary hidden states with shape
                ``[S, B, 3h]``.
            input_embeds: Pass-1 token embeddings ``e(x_{i+1})`` with shape
                ``[S, B, h]``. They have already been shifted left once.

        Returns:
            One ``[B, S, draft_vocab]`` logits tensor per pass. At pass ``d``,
            position ``i`` predicts token ``x_{i+d+1}``.
        """
        if self.ttt_steps < 2:
            raise RuntimeError(
                "forward_ttt requires ttt_steps >= 2; use forward() for the "
                "single-pass draft."
            )
        # Needed only by the multi-pass training path.
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
                    # The custom attention modules create the trunk/branch mask.
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
            # Clear saved K/V and hidden states even if a pass raises an error.
            for ttt_attention in self._ttt_attn_modules:
                ttt_attention.reset()
            self._ttt_prenorm_hidden = None
        return logits_by_pass
