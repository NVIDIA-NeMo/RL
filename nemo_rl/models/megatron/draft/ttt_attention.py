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

"""Two-part attention for EAGLE-3 multi-pass (TTT) draft training.

TTT pass ``d`` (1-indexed) runs the draft decoder once per pass over the full
sequence. The query at anchor position ``i`` of pass ``d`` attends to:

- the *trunk*: pass-1 keys/values at positions ``j <= i`` (causal), and
- the *branch*: the same-anchor diagonal key/value of every later pass
  ``p in [2, d]`` (including its own pass).

This mirrors inference, where the draft's KV cache holds the prefill (pass-1)
entries plus one self-generated entry per speculation depth.
:class:`TwoPartTTTAttention` has the mask diagram.

The two parts are fused exactly via the standard log-sum-exp merge: the trunk
runs through the FlashAttention kernel, the branch (at most ``d - 1`` keys per
query) is a small einsum, and the joint softmax is recovered from the two
partial (out, lse) pairs. The backward pass feeds the FlashAttention backward
kernel the *joint* (out, lse) — which yields the exact trunk-restricted
gradient of the joint softmax (the same mechanism ring attention uses for
context parallelism) — and computes the branch-restricted gradient in closed
form.

FlashAttention is called through its private ``_flash_attn_forward`` /
``_flash_attn_backward`` interface (needed to obtain and feed the lse). The
call signatures are vendored for flash-attn 2.8.1 (pinned in uv.lock), the
same pattern ring-flash-attention uses; ``_load_flash_attn`` gates on the
flash-attn version at import time so a version bump fails loudly.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor

_FLASH_ATTN_FUNCS: Optional[
    tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]
] = None


def _load_flash_attn() -> tuple[
    Callable[..., Any], Callable[..., Any], Callable[..., Any]
]:
    """Return ``(flash_attn_func, _flash_attn_forward, _flash_attn_backward)``.

    Deferred import: flash-attn is a GPU-only dependency and must not load on
    CPU-only paths (unit tests, config validation).
    """
    global _FLASH_ATTN_FUNCS
    if _FLASH_ATTN_FUNCS is not None:
        return _FLASH_ATTN_FUNCS

    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_backward,
        _flash_attn_forward,
        flash_attn_func,
    )

    # The private call contract this module vendors (split window_size_left/
    # right args + softcap) exists in flash-attn 2.7-2.x. The functions are
    # torch custom ops on torch >= 2.4, so their python signature cannot be
    # introspected — gate on the package version and fail loudly outside the
    # vendored range instead of passing misaligned positional args.
    version_parts = flash_attn.__version__.split(".")
    major, minor = int(version_parts[0]), int(version_parts[1])
    if not (major == 2 and minor >= 7):
        raise RuntimeError(
            f"flash-attn {flash_attn.__version__} does not match the private "
            "interface vendored for flash-attn 2.8.1; update "
            "nemo_rl/models/megatron/draft/ttt_attention.py."
        )

    _FLASH_ATTN_FUNCS = (flash_attn_func, _flash_attn_forward, _flash_attn_backward)
    return _FLASH_ATTN_FUNCS


def _fa_forward_causal(
    q: Tensor, k: Tensor, v: Tensor, softmax_scale: float
) -> tuple[Tensor, Tensor]:
    """Causal FlashAttention forward returning ``(out, lse)``.

    Args:
        q: Queries ``[B, S, Hq, D]`` (fp16/bf16).
        k: Keys ``[B, S, Hkv, D]``.
        v: Values ``[B, S, Hkv, D]``.
        softmax_scale: Score scale (``1/sqrt(D)`` convention).

    Returns:
        ``out`` ``[B, S, Hq, D]`` in the input dtype and ``lse`` ``[B, Hq, S]``
        in fp32 (natural log).
    """
    _, fa_fwd, _ = _load_flash_attn()
    out, softmax_lse, _, _ = fa_fwd(
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
    """Causal FlashAttention backward writing into preallocated ``dq/dk/dv``.

    Feeding the *joint* ``(out, lse)`` makes the kernel return the exact
    trunk-restricted gradient of the joint two-part softmax.
    """
    _, _, fa_bwd = _load_flash_attn()
    fa_bwd(
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
    """Exact joint attention over a causal trunk plus per-anchor branch keys.

    The mask (never materialized), for ``S = 4`` positions and ``K = 3`` TTT
    passes. Teacher forcing makes every position an anchor, so pass ``d`` is
    one decoder forward whose query at anchor ``i`` replays depth ``d`` of the
    speculation chain rooted at ``i``. Row block ``d`` below holds that pass's
    queries: block 1 is the plain causal FlashAttention call in
    :class:`TTTDraftCoreAttention` (drawn to complete the picture); each later
    block is one ``TwoPartTTTAttention`` call. ``A`` keys ride the causal
    FlashAttention kernel (part one), ``x`` keys the small fp32 einsum
    (part two)::

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

    A pass-``d`` query at anchor ``i`` sees the trunk causally *including*
    ``i`` and, in each pass group ``p in [2, d]``, only its own anchor's
    diagonal entry — never another anchor's branch, never a later trunk
    position. Chains rooted at different anchors share nothing but the trunk,
    which is what lets all ``S`` anchors of a pass run as one kernel call. In
    RoPE terms the union key set of ``q_d@i`` covers positions ``0..i``
    (trunk) then ``i+1 .. i+d-1`` (branch diagonals, own key last) with the
    query itself at ``i+d-1`` — exactly the KV cache state of the depth-``d``
    speculation step at serving: prefill entries plus the ``d-1`` entries the
    draft generated itself.

    Per pass ``d``, anchor ``i`` (``EagleModel.forward_ttt`` is the driver;
    ``DraftCrossEntropyLossFn`` applies the shifts)::

        input   concat(e(x_{i+d}), h^{d-1}_i)    h^0 = fc(target aux taps),
                                                 h^p = pass-p pre-norm hidden
        RoPE    i + d - 1
        target  x_{i+d+1}   (teacher: the policy's logits at i + d)
        stash   (k^d_i, v^d_i) -> the next pass's newest branch column

    Inputs are batch-first (FlashAttention layout):

    - ``q``: pass-d queries ``[B, S, Hq, D]``
    - ``k1``/``v1``: pass-1 (trunk) keys/values ``[B, S, Hkv, D]``
    - ``kb``/``vb``: branch keys/values ``[B, S, Hkv, P, D]`` where slot ``p``
      holds the pass-(p+2) diagonal entry for the same anchor position
      (``P = d - 1``, including pass d itself).

    The output equals softmax over the union key set
    ``{k1[:, :i+1]} ∪ {kb[:, i, :, p] for all p}`` per query ``i`` — i.e. the
    dense staircase mask — computed without materializing any ``S x S`` score
    matrix.
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
        """Drop stashed KV references (frees the autograd graph between steps)."""
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
        if attention_bias is not None:
            raise NotImplementedError(
                "TTT draft training does not support attention_bias."
            )
        if attention_mask is not None:
            raise ValueError(
                "TTTDraftCoreAttention builds its own causal/branch masking; "
                "pass attention_mask=None (see EagleModel.forward)."
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
        if len(self._kv_by_pass) != self._pass_idx:
            raise RuntimeError(
                f"TTT pass bookkeeping out of sync: pass {self._pass_idx} but "
                f"{len(self._kv_by_pass)} stashed KV entries."
            )

        if self._pass_idx == 1:
            flash_attn_func, _, _ = _load_flash_attn()
            out = flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=True)
        else:
            k1, v1 = self._kv_by_pass[0]
            kb = torch.stack([kv[0] for kv in self._kv_by_pass[1:]], dim=3)
            vb = torch.stack([kv[1] for kv in self._kv_by_pass[1:]], dim=3)
            out = TwoPartTTTAttention.apply(q, k1, v1, kb, vb, softmax_scale)

        # bshd -> [S, B, Hq * D] (core_attention output contract).
        return out.transpose(0, 1).reshape(seqlen, batch, -1)
