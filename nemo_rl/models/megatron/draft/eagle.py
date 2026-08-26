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

Packed and padded inputs share one flat THD code path. With context
parallelism (CP), queries and branch K/V stay local while pass-1 trunk K/V
moves around the zigzag ring. Partial attention results are merged at each
step, and trunk gradients finish by returning to the rank that owns them.

The implementation uses private FlashAttention functions because the merge
needs their log-sum-exp values. Their interface is pinned to the compatible
FlashAttention 2.x versions checked below.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import flash_attn
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)
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


def _check_flash_attn_version() -> None:
    """Reject flash-attn versions whose private interface may not match.

    ``_flash_attn_varlen_forward``/``_flash_attn_varlen_backward`` are private
    FlashAttention functions, so their signatures cannot be inspected reliably;
    this file uses the interface from version 2.8.1, compatible with
    FlashAttention 2.7 and later 2.x releases. Checked when the multi-pass
    attention is built — not at import — so a future flash-attn bump surfaces
    here as a clear error for TTT users instead of an import failure for every
    Megatron run (this module is imported on the refit path even with the
    draft disabled).
    """
    major, minor = (int(x) for x in flash_attn.__version__.split(".")[:2])
    if not (major == 2 and minor >= 7):
        raise RuntimeError(
            f"flash-attn {flash_attn.__version__} does not match the private "
            "interface vendored for flash-attn 2.8.1; update "
            "nemo_rl/models/megatron/draft/eagle.py."
        )


def _fa_varlen_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[Tensor, Tensor]:
    """Run variable-length FlashAttention on flat THD tensors.

    Returns the output with shape ``[T, Hq, D]`` and fp32 log-sum-exp values
    with shape ``[Hq, T]``.
    """
    out, softmax_lse, _, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        0.0,  # dropout_p
        softmax_scale,
        causal,
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # return_softmax
    )
    return out, softmax_lse


def _fa_varlen_backward(
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
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    causal: bool,
) -> None:
    """Run variable-length FlashAttention backward into existing buffers.

    ``out`` and ``lse`` describe attention over all merged key sets. Passing
    them to this call produces the exact part of that joint-softmax gradient
    that belongs to this call's keys.
    """
    _flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        0.0,  # dropout_p
        softmax_scale,
        causal,
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # deterministic
        None,  # rng_state
    )


def _global_cp_group() -> Optional[dist.ProcessGroup]:
    """The global CP group, or None before model-parallel init (unit tests)."""
    if not parallel_state.model_parallel_is_initialized():
        return None
    return parallel_state.get_context_parallel_group()


def _zigzag_half_split(cu_seqlens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split every local zigzag subsequence into its front and back halves.

    A CP rank stores two equally sized chunks from each global subsequence.
    Rank ``r`` owns front chunk ``r`` and back chunk ``2 * cp_size - 1 - r``.

    Args:
        cu_seqlens: Boundaries of the rank-local THD subsequences.

    Returns:
        Indices of all front rows, indices of all back rows, and THD
        boundaries for a tensor containing one half of each subsequence.
    """
    device = cu_seqlens.device
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
    half = lens // 2
    total = int(cu_seqlens[-1].item())
    seq_id = torch.repeat_interleave(torch.arange(lens.numel(), device=device), lens)
    pos_local = (
        torch.arange(total, device=device) - cu_seqlens[:-1].to(torch.long)[seq_id]
    )
    front_mask = pos_local < half[seq_id]
    front_index = front_mask.nonzero(as_tuple=True)[0]
    back_index = (~front_mask).nonzero(as_tuple=True)[0]
    cu_half = torch.zeros_like(cu_seqlens)
    cu_half[1:] = torch.cumsum(half, dim=0).to(cu_seqlens.dtype)
    return front_index, back_index, cu_half


def _ring_send_recv(
    tensors: list[Tensor],
    send_rank: int,
    recv_rank: int,
    cp_group: dist.ProcessGroup,
) -> tuple[list[Tensor], list[Any]]:
    """Move a list of tensors one step around the CP ring.

    Each tensor is sent to the next rank while a same-shaped tensor is received
    from the previous rank. The caller must keep the sent tensors alive until
    all returned requests have completed.
    """
    recvs = [torch.empty_like(t) for t in tensors]
    ops = []
    for send_t, recv_t in zip(tensors, recvs):
        ops.append(dist.P2POp(dist.isend, send_t.contiguous(), send_rank, cp_group))
        ops.append(dist.P2POp(dist.irecv, recv_t, recv_rank, cp_group))
    return recvs, dist.batch_isend_irecv(ops)


def _lse_merge_update(
    out_acc: Tensor,
    lse_acc: Tensor,
    out_step: Tensor,
    lse_step: Tensor,
    rows: Optional[Tensor] = None,
) -> None:
    """Merge attention over another set of keys into fp32 accumulators.

    ``out_step`` is normalized only over the keys handled by this step. Its
    log-sum-exp value supplies the exact weight needed to combine it with the
    earlier results. The accumulators start at 0 and ``-inf``, so the first
    merge acts like a copy. ``rows`` limits the merge when only the local back
    half contains valid queries.
    """
    if rows is None:
        lse_new = torch.logaddexp(lse_acc, lse_step)
        w_old = torch.exp(lse_acc - lse_new).transpose(0, 1).unsqueeze(-1)
        w_new = torch.exp(lse_step - lse_new).transpose(0, 1).unsqueeze(-1)
        out_acc.mul_(w_old).add_(out_step.float() * w_new)
        lse_acc.copy_(lse_new)
    else:
        lse_old = lse_acc[:, rows]
        lse_new = torch.logaddexp(lse_old, lse_step)
        w_old = torch.exp(lse_old - lse_new).transpose(0, 1).unsqueeze(-1)
        w_new = torch.exp(lse_step - lse_new).transpose(0, 1).unsqueeze(-1)
        out_acc[rows] = out_acc[rows] * w_old + out_step.float() * w_new
        lse_acc[:, rows] = lse_new


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
    without creating a ``T x T`` score matrix. Backward passes the merged
    output and log-sum-exp to FlashAttention for the trunk and computes the
    branch gradient directly.

    Inputs to ``apply`` use flat THD layout:

    - ``q``: Current-pass queries with shape ``[T, Hq, D]``.
    - ``k1`` and ``v1``: Pass-1 trunk K/V with shape ``[T, Hkv, D]``.
    - ``kb`` and ``vb``: Branch K/V with shape ``[T, Hkv, P, D]``. The branch
      axis stores passes ``2..d`` for the same local anchor, so ``P = d - 1``.
      Both tensors are ``None`` during pass 1.
    - ``cu_seqlens``: Rank-local subsequence boundaries with shape ``[N + 1]``.
    - ``max_seqlen``: Maximum rank-local subsequence length.
    - ``softmax_scale``: Scale applied to attention scores.
    - ``cp_group``: CP process group, or ``None`` on a single rank.

    Packing boundaries are respected by variable-length FlashAttention. The
    causal trunk restarts at every subsequence, while each branch stays at one
    local row and therefore cannot cross into another subsequence. Padded input
    is converted to ``B`` equal-length THD subsequences by the caller.

    With CP, rank ``r`` owns a front chunk ``r`` and a back chunk
    ``2 * cp_size - 1 - r`` from every subsequence. Queries remain local while
    trunk K/V moves around the ring. Each ring step uses one of three masks:

    - Step 0 uses causal attention over this rank's concatenated front and back
      chunks.
    - At steps ``s <= r``, all local queries see the arriving front chunk and
      none see its back chunk.
    - At steps ``s > r``, only local back-chunk queries see the entire arriving
      K/V shard.

    The local branch is merged as one additional step. Backward replays the
    ring, and trunk K/V gradients make a final hop back to their owner rank.

    The returned tensor has shape ``[T, Hq, D]``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: Tensor,
        k1: Tensor,
        v1: Tensor,
        kb: Optional[Tensor],
        vb: Optional[Tensor],
        cu_seqlens: Tensor,
        max_seqlen: int,
        softmax_scale: float,
        cp_group: Optional[dist.ProcessGroup],
    ) -> Tensor:
        total, num_q_heads, head_dim = q.shape
        num_kv_heads = k1.shape[1]
        group = num_q_heads // num_kv_heads
        has_branch = kb is not None

        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            cp_rank = cp_group.rank()
            global_ranks = dist.get_process_group_ranks(cp_group)
            send_rank = global_ranks[(cp_rank + 1) % cp_size]
            recv_rank = global_ranks[(cp_rank - 1) % cp_size]
            front_index, back_index, cu_half = _zigzag_half_split(cu_seqlens)
            max_half = max(max_seqlen // 2, 1)
            q_back = q[back_index]
        else:
            cp_rank = 0
            front_index = back_index = cu_half = None

        out_acc = q.new_zeros((total, num_q_heads, head_dim), dtype=torch.float32)
        lse_acc = q.new_full((num_q_heads, total), float("-inf"), dtype=torch.float32)

        k_cur, v_cur = k1, v1
        reqs: list[Any] = []
        for step in range(cp_size):
            if step + 1 < cp_size:
                (k_next, v_next), reqs = _ring_send_recv(
                    [k_cur, v_cur], send_rank, recv_rank, cp_group
                )
            if step == 0:
                out_s, lse_s = _fa_varlen_forward(
                    q,
                    k_cur,
                    v_cur,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    softmax_scale,
                    causal=True,
                )
                _lse_merge_update(out_acc, lse_acc, out_s, lse_s)
            elif step <= cp_rank:
                out_s, lse_s = _fa_varlen_forward(
                    q,
                    k_cur[front_index],
                    v_cur[front_index],
                    cu_seqlens,
                    cu_half,
                    max_seqlen,
                    max_half,
                    softmax_scale,
                    causal=False,
                )
                _lse_merge_update(out_acc, lse_acc, out_s, lse_s)
            else:
                out_s, lse_s = _fa_varlen_forward(
                    q_back,
                    k_cur,
                    v_cur,
                    cu_half,
                    cu_seqlens,
                    max_half,
                    max_seqlen,
                    softmax_scale,
                    causal=False,
                )
                _lse_merge_update(out_acc, lse_acc, out_s, lse_s, rows=back_index)
            if step + 1 < cp_size:
                for req in reqs:
                    req.wait()
                k_cur, v_cur = k_next, v_next

        if has_branch:
            # Each query has at most P branch keys, so compute this small part
            # directly in fp32.
            q_grouped = q.view(total, num_kv_heads, group, head_dim).float()
            scores_b = (
                torch.einsum("tkgd,tkpd->tkgp", q_grouped, kb.float()) * softmax_scale
            )
            lse_b = torch.logsumexp(scores_b, dim=-1)  # [T, Hkv, G]
            probs_b = torch.exp(scores_b - lse_b.unsqueeze(-1))
            out_b = torch.einsum("tkgp,tkpd->tkgd", probs_b, vb.float())

            # Merge in place. An einsum result may have a permuted KV-major
            # layout; replacing ``out_acc`` with a broadcast expression could
            # inherit that layout and break the final view.
            lse_t = lse_acc.transpose(0, 1).reshape(total, num_kv_heads, group)
            lse_joint = torch.logaddexp(lse_t, lse_b)
            w_t = torch.exp(lse_t - lse_joint).unsqueeze(-1)
            w_b = torch.exp(lse_b - lse_joint).unsqueeze(-1)
            out_acc_grouped = out_acc.view(total, num_kv_heads, group, head_dim)
            out_acc_grouped.mul_(w_t).add_(w_b * out_b)
            out = out_acc.view(total, num_q_heads, head_dim).to(q.dtype)
            # FlashAttention backward expects fp32 LSE in [Hq, T] layout.
            lse_final = lse_joint.view(total, num_q_heads).transpose(0, 1).contiguous()
        else:
            out = out_acc.to(q.dtype)
            lse_final = lse_acc

        saved = [q, k1, v1, out, lse_final, cu_seqlens]
        if has_branch:
            saved += [kb, vb]
        if cp_size > 1:
            saved += [front_index, back_index, cu_half]
        ctx.save_for_backward(*saved)
        ctx.has_branch = has_branch
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.max_seqlen = max_seqlen
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, dout: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        saved = list(ctx.saved_tensors)
        q, k1, v1, out, lse_joint, cu_seqlens = saved[:6]
        cursor = 6
        if ctx.has_branch:
            kb, vb = saved[cursor : cursor + 2]
            cursor += 2
        if ctx.cp_size > 1:
            front_index, back_index, cu_half = saved[cursor : cursor + 3]
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        softmax_scale = ctx.softmax_scale
        max_seqlen = ctx.max_seqlen

        total, num_q_heads, head_dim = q.shape
        num_kv_heads = k1.shape[1]
        group = num_q_heads // num_kv_heads

        dout = dout.contiguous()
        # Accumulate K/V gradients in fp32 while they travel with their shard;
        # each FlashAttention call returns lower-precision step gradients.
        dq_acc = q.new_zeros((total, num_q_heads, head_dim), dtype=torch.float32)
        dk_cur = q.new_zeros(k1.shape, dtype=torch.float32)
        dv_cur = torch.zeros_like(dk_cur)

        if cp_size > 1:
            cp_rank = cp_group.rank()
            global_ranks = dist.get_process_group_ranks(cp_group)
            send_rank = global_ranks[(cp_rank + 1) % cp_size]
            recv_rank = global_ranks[(cp_rank - 1) % cp_size]
            max_half = max(max_seqlen // 2, 1)
            q_back = q[back_index]
            dout_back = dout[back_index]
            out_back = out[back_index]
            lse_back = lse_joint[:, back_index].contiguous()
        else:
            cp_rank = 0

        # Replay the forward ring. Add each step's K/V gradient before moving
        # the shard and its accumulator together. A final hop returns the full
        # gradient to the rank that owns the shard.
        k_cur, v_cur = k1, v1
        for step in range(cp_size):
            kv_reqs: list[Any] = []
            if step + 1 < cp_size:
                (k_next, v_next), kv_reqs = _ring_send_recv(
                    [k_cur, v_cur], send_rank, recv_rank, cp_group
                )
            if step == 0:
                dq_s = torch.empty_like(q)
                dk_s = torch.empty_like(k_cur)
                dv_s = torch.empty_like(v_cur)
                _fa_varlen_backward(
                    dout=dout,
                    q=q,
                    k=k_cur,
                    v=v_cur,
                    out=out,
                    lse=lse_joint,
                    dq=dq_s,
                    dk=dk_s,
                    dv=dv_s,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
                dq_acc += dq_s.float()
                dk_cur += dk_s.float()
                dv_cur += dv_s.float()
            elif step <= cp_rank:
                k_half = k_cur[front_index]
                v_half = v_cur[front_index]
                dq_s = torch.empty_like(q)
                dk_s = torch.empty_like(k_half)
                dv_s = torch.empty_like(v_half)
                _fa_varlen_backward(
                    dout=dout,
                    q=q,
                    k=k_half,
                    v=v_half,
                    out=out,
                    lse=lse_joint,
                    dq=dq_s,
                    dk=dk_s,
                    dv=dv_s,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_half,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_half,
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                dq_acc += dq_s.float()
                dk_cur.index_add_(0, front_index, dk_s.float())
                dv_cur.index_add_(0, front_index, dv_s.float())
            else:
                dq_s = torch.empty_like(q_back)
                dk_s = torch.empty_like(k_cur)
                dv_s = torch.empty_like(v_cur)
                _fa_varlen_backward(
                    dout=dout_back,
                    q=q_back,
                    k=k_cur,
                    v=v_cur,
                    out=out_back,
                    lse=lse_back,
                    dq=dq_s,
                    dk=dk_s,
                    dv=dv_s,
                    cu_seqlens_q=cu_half,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=max_half,
                    max_seqlen_kv=max_seqlen,
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                dq_acc.index_add_(0, back_index, dq_s.float())
                dk_cur += dk_s.float()
                dv_cur += dv_s.float()
            if step + 1 < cp_size:
                for req in kv_reqs:
                    req.wait()
                # Move gradients only after adding this step's contribution.
                (dk_next, dv_next), dkv_reqs = _ring_send_recv(
                    [dk_cur, dv_cur], send_rank, recv_rank, cp_group
                )
                for req in dkv_reqs:
                    req.wait()
                k_cur, v_cur = k_next, v_next
                dk_cur, dv_cur = dk_next, dv_next
        if cp_size > 1:
            (dk_own, dv_own), dkv_reqs = _ring_send_recv(
                [dk_cur, dv_cur], send_rank, recv_rank, cp_group
            )
            for req in dkv_reqs:
                req.wait()
            dk_cur, dv_cur = dk_own, dv_own

        if ctx.has_branch:
            # Compute the branch gradient directly in fp32. ``probs_bj`` uses
            # the joint normalization, and ``d_row`` supplies the correction
            # caused by sharing one softmax with the trunk.
            q_grouped = q.view(total, num_kv_heads, group, head_dim).float()
            scores_b = (
                torch.einsum("tkgd,tkpd->tkgp", q_grouped, kb.float()) * softmax_scale
            )
            lse_joint_grouped = (
                lse_joint.transpose(0, 1).reshape(total, num_kv_heads, group).float()
            )
            probs_bj = torch.exp(scores_b - lse_joint_grouped.unsqueeze(-1))

            dout_f32 = dout.float()
            d_row = (
                (dout_f32 * out.float()).sum(dim=-1).view(total, num_kv_heads, group)
            )
            dout_grouped = dout_f32.view(total, num_kv_heads, group, head_dim)

            dvb = torch.einsum("tkgp,tkgd->tkpd", probs_bj, dout_grouped)
            d_probs = torch.einsum("tkgd,tkpd->tkgp", dout_grouped, vb.float())
            d_scores = probs_bj * (d_probs - d_row.unsqueeze(-1)) * softmax_scale
            # Use reshape because einsum may return a permuted layout.
            dq_acc += torch.einsum("tkgp,tkpd->tkgd", d_scores, kb.float()).reshape(
                total, num_q_heads, head_dim
            )
            dkb = torch.einsum("tkgp,tkgd->tkpd", d_scores, q_grouped).to(kb.dtype)
            dvb = dvb.to(vb.dtype)
        else:
            dkb = dvb = None

        return (
            dq_acc.to(q.dtype),
            dk_cur.to(k1.dtype),
            dv_cur.to(v1.dtype),
            dkb,
            dvb,
            None,
            None,
            None,
            None,
        )


class TTTDraftCoreAttention(torch.nn.Module):
    """Adapt MCore self-attention to the multi-pass TTT mask.

    This module replaces each draft decoder layer's TE core attention. MCore
    passes Q/K/V after RoPE in one of two layouts:

    - padded: ``[S, B, H, D]``;
    - packed THD: ``[T, H, D]``, with the batch dimension removed.

    The returned layouts are ``[S, B, Hq * D]`` and ``[T, Hq * D]``.

    The module keeps K/V from earlier passes. :meth:`begin_pass` selects the
    current pass. Pass 1 becomes the causal trunk; later passes are appended as
    branch entries. These tensors are not detached because later-pass losses
    must also update the projections that created earlier K/V. :meth:`reset`
    clears all saved tensors after the TTT loop, including error paths.

    ``pg_collection`` supplies the CP process group. Multiple CP ranks require
    packed THD input because padded input has no zigzag position metadata.
    """

    def __init__(self, config: Any):
        super().__init__()
        _check_flash_attn_version()
        attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        if attention_dropout != 0.0:
            # The two-part flash calls hard-code dropout_p=0.0; ignoring the
            # config value would silently train without the requested dropout.
            raise ValueError(
                "TTTDraftCoreAttention does not support attention dropout "
                f"(got attention_dropout={attention_dropout})."
            )
        # Match MCore: use the configured scale, or default to 1 / sqrt(D).
        self.softmax_scale: Optional[float] = getattr(config, "softmax_scale", None)
        # ``build_draft_model`` injects this collection. The draft module itself
        # is replicated rather than sharded across CP ranks.
        self.pg_collection: Optional[Any] = None
        self._pass_idx = 0
        self._kv_by_pass: list[tuple[Tensor, Tensor]] = []

    def begin_pass(self, pass_idx: int) -> None:
        """Start the one-indexed TTT pass ``pass_idx``."""
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

    def _cp_group(self) -> Optional[dist.ProcessGroup]:
        group = getattr(self.pg_collection, "cp", None)
        if group is None or group.size() == 1:
            return None
        return group

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
        if attention_mask is not None:
            # The layer spec is AttnMaskType.arbitrary, so MCore may hand in a
            # mask; this module builds its own trunk/branch structure and
            # silently dropping a padding mask would be wrong.
            raise ValueError(
                "TTTDraftCoreAttention builds its own trunk/branch attention_mask "
                "structure and cannot honour an external attention_mask."
            )
        if attention_bias is not None:
            raise ValueError("TTTDraftCoreAttention does not support attention_bias.")
        if self._pass_idx < 1:
            raise RuntimeError(
                "TTTDraftCoreAttention.forward called without begin_pass(); "
                "the TTT driver must arm the pass index before each pass."
            )

        cp_group = self._cp_group()
        softmax_scale = self.softmax_scale or query.shape[-1] ** -0.5

        if packed_seq_params is not None:
            if getattr(packed_seq_params, "qkv_format", None) != "thd":
                raise NotImplementedError(
                    "TTT draft training supports packed sequences only in "
                    f"'thd' format, got {getattr(packed_seq_params, 'qkv_format', None)!r}."
                )
            # MCore removes the singleton batch dimension in THD mode.
            q, k, v = query, key, value
            cu_global = packed_seq_params.cu_seqlens_q_padded
            if cu_global is None:
                cu_global = packed_seq_params.cu_seqlens_q
            cp_size = 1 if cp_group is None else cp_group.size()
            cu_local = torch.div(cu_global, cp_size, rounding_mode="floor").to(
                torch.int32
            )
            max_local = int(packed_seq_params.max_seqlen_q) // cp_size
            packed = True
        else:
            if cp_group is not None:
                raise NotImplementedError(
                    "Context-parallel TTT draft training requires sequence "
                    "packing (THD input); enable policy.sequence_packing."
                )
            # Treat a padded batch as B equal-length THD subsequences.
            seqlen, batch = query.shape[0], query.shape[1]
            q = (
                query.transpose(0, 1)
                .contiguous()
                .view(batch * seqlen, *query.shape[2:])
            )
            k = key.transpose(0, 1).contiguous().view(batch * seqlen, *key.shape[2:])
            v = (
                value.transpose(0, 1)
                .contiguous()
                .view(batch * seqlen, *value.shape[2:])
            )
            cu_local = (
                torch.arange(0, batch + 1, device=query.device, dtype=torch.int32)
                * seqlen
            )
            max_local = seqlen
            packed = False

        self._kv_by_pass.append((k, v))

        if self._pass_idx == 1:
            out = TwoPartTTTAttention.apply(
                q, k, v, None, None, cu_local, max_local, softmax_scale, cp_group
            )
        else:
            k1, v1 = self._kv_by_pass[0]
            kb = torch.stack([kv[0] for kv in self._kv_by_pass[1:]], dim=2)
            vb = torch.stack([kv[1] for kv in self._kv_by_pass[1:]], dim=2)
            out = TwoPartTTTAttention.apply(
                q, k1, v1, kb, vb, cu_local, max_local, softmax_scale, cp_group
            )

        if packed:
            # MCore expects [T, Hq * D] here and restores batch size 1 itself.
            return out.reshape(out.shape[0], -1)
        # Restore the padded core-attention output layout [S, B, Hq * D].
        return out.view(batch, seqlen, -1).transpose(0, 1)


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
                config, "use_cpu_initialization", not torch.cuda.is_available()
            ),
        )
        # Import here to avoid a circular dependency inside ModelOpt's
        # quantization backends.
        from modelopt.torch.speculative.plugins.megatron_eagle import EagleModule

        # Supply RoPE explicitly instead of relying on the Llama-style defaults
        # used by many speculative-decoding wrappers.
        self.eagle_module = EagleModule(
            config=config, rotary_pos_emb=rotary_pos_emb, bias=False
        )

        # ModelOpt defaults to an arbitrary mask, which selects TE's unfused
        # quadratic fp32 path. This file builds the mask itself, so the decoder
        # can use the fused causal path and receive ``attention_mask=None``.
        for layer in self.eagle_module.decoder.layers:
            layer.self_attention.attn_mask_type = AttnMaskType.causal

        # Multi-pass TTT needs the branch mask. CP also needs this implementation
        # because TE's CP path does not expose the LSE required for merging.
        cp_group = _global_cp_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        self._needs_ttt_attention = self.ttt_steps > 1 or cp_size > 1

        self._ttt_attn_modules: list[TTTDraftCoreAttention] = []
        self._ttt_prenorm_hidden: Optional[Tensor] = None
        if self._needs_ttt_attention:
            if getattr(config, "recompute_granularity", None) is not None:
                # Activation recomputation would call this stateful attention
                # again during backward and corrupt its per-pass K/V list.
                raise ValueError(
                    "TTT draft training (ttt_steps > 1 or CP > 1) is "
                    "incompatible with activation recomputation on the draft "
                    "config."
                )
            for layer in self.eagle_module.decoder.layers:
                ttt_attention = TTTDraftCoreAttention(config)
                layer.self_attention.core_attention = ttt_attention
                self._ttt_attn_modules.append(ttt_attention)
        if self.ttt_steps > 1:
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
        """Build a sharded state dict with correct TP metadata.

        ModelOpt versions before 0.42.0 omit ``tp_group`` for decoder children
        outside the layer list, such as the final layer norm. With both TP and
        DP enabled, this can assign the same checkpoint replica ID to multiple
        ranks and raise ``CheckpointingException``.

        This method first builds the normal state dict, then rebuilds only the
        affected child entries with the decoder's TP group.

        Args:
            prefix: Prefix added to every state-dict key.
            sharded_offsets: Parent-provided sharding offsets.
            metadata: Distributed checkpoint metadata.

        Returns:
            Sharded state dict with corrected entries for non-layer decoder
            children.
        """
        sd = super().sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

        decoder = self.eagle_module.decoder
        if not hasattr(decoder, "layers"):
            return sd

        metadata = ensure_metadata_has_dp_cp_group(metadata)

        # Rebuild non-layer children with the correct TP group. ModelOpt's
        # EagleTransformerBlock requires empty offsets for these entries.
        for name, module in decoder.named_children():
            if module is decoder.layers:
                continue
            child_prefix = f"{prefix}eagle_module.decoder.{name}."
            for k in list(sd):
                if k.startswith(child_prefix):
                    del sd[k]
            sd.update(
                sharded_state_dict_default(
                    module, child_prefix, (), metadata, tp_group=decoder.tp_group
                )
            )

        return sd

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        bootstrap_hidden_states: bool = True,
        packed_seq_params: Optional[Any] = None,
    ) -> Tensor:
        if bootstrap_hidden_states:
            hidden_states = self.eagle_module.fc(hidden_states)[0]
        elif hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden states with size {self.config.hidden_size} when "
                f"`bootstrap_hidden_states=False`, got {hidden_states.shape[-1]}."
            )

        # Packed THD input needs explicit RoPE. EagleModule's fallback sizes the
        # table from total packed tokens and slices it for CP too early.
        rotary_pos_emb = None
        if packed_seq_params is not None:
            rotary_pos_emb = self._rotary_for_pass(
                1, packed_seq_params, input_embeds.shape[0]
            )

        for ttt_attention in self._ttt_attn_modules:
            ttt_attention.begin_pass(1)
        try:
            hidden_states, _ = self.eagle_module(
                embeddings=input_embeds,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
        finally:
            for ttt_attention in self._ttt_attn_modules:
                ttt_attention.reset()
        logits, _ = self.eagle_module.eagle_output_layer(hidden_states)
        logits = logits.transpose(0, 1).contiguous()
        return logits

    def _rotary_for_pass(
        self,
        ttt_pass: int,
        packed_seq_params: Optional[Any],
        local_seq_len: int,
    ) -> Optional[Tensor]:
        """Build RoPE frequencies for one TTT pass.

        Pass ``d`` shifts every position by ``d - 1``. A query at anchor ``i``
        therefore uses position ``i + d - 1``, matching the extra cache entry
        created at each speculation depth during serving.

        For packed THD input, the table is based on the maximum subsequence
        length rather than the total number of packed tokens. Positions restart
        in every subsequence, and MCore performs the rank-local zigzag slice.
        Using ``packed_seq=True`` prevents this method from slicing the table
        before MCore receives it.

        For padded input, pass 1 returns ``None`` because EagleModule's default
        table is already correct. Later passes build a shifted global table;
        when CP is active, this method then selects the current rank's zigzag
        rows because the padded path applies the returned frequencies directly.

        Args:
            ttt_pass: One-indexed TTT pass ``d``.
            packed_seq_params: THD packing metadata, or ``None`` for padded
                input.
            local_seq_len: Number of local rows in the padded input.

        Returns:
            Shifted RoPE frequencies for this pass, or ``None`` when the
            single-pass padded default can be used.
        """
        if packed_seq_params is not None:
            max_seqlen = int(packed_seq_params.max_seqlen_q)
            rotary = self.eagle_module.rotary_pos_emb(
                max_seqlen + ttt_pass - 1, packed_seq=True
            )
            return rotary[ttt_pass - 1 :]

        if ttt_pass == 1:
            return None
        cp_group = _global_cp_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        global_seq_len = local_seq_len * cp_size
        rotary = self.eagle_module.rotary_pos_emb(
            global_seq_len + ttt_pass - 1, packed_seq=True
        )
        rotary = rotary[ttt_pass - 1 :]
        if cp_size > 1:
            from megatron.core.models.common.embeddings.rope_utils import (
                get_pos_emb_on_this_cp_rank,
            )

            rotary = get_pos_emb_on_this_cp_rank(rotary, 0, cp_group)
        return rotary

    def forward_ttt(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        packed_seq_params: Optional[Any] = None,
    ) -> list[Tensor]:
        """Run the configured TTT passes in sequence.

        Pass 1 projects the target auxiliary hidden states. Every later pass
        receives the previous pass's pre-final-norm hidden state without
        detaching it, so gradients flow through the entire pass chain. Token
        embeddings shift left by one position before each new pass. Packed
        shifts stop at subsequence boundaries and exchange boundary values
        between CP chunks when needed.

        Args:
            hidden_states: Target auxiliary hidden states with padded shape
                ``[S, B, 3h]`` or packed local shape ``[T, 1, 3h]``.
            input_embeds: Pass-1 token embeddings ``e(x_{i+1})`` with padded
                shape ``[S, B, h]`` or packed shape ``[T, 1, h]``. They have
                already been shifted left once within each subsequence.
            packed_seq_params: Global THD packing metadata, or ``None`` for
                padded input.

        Returns:
            One logits tensor per pass. Padded tensors have shape
            ``[B, S, draft_vocab]`` and packed tensors have shape
            ``[1, T, draft_vocab]``. At pass ``d``, position ``i`` predicts
            token ``x_{i+d+1}`` in its subsequence.
        """
        if self.ttt_steps < 2:
            raise RuntimeError(
                "forward_ttt requires ttt_steps >= 2; use forward() for the "
                "single-pass draft."
            )
        # These helpers are needed only by the multi-pass training path.
        from megatron.core.transformer.multi_token_prediction import roll_tensor

        from nemo_rl.algorithms.loss.utils import roll_packed_left_cp

        cp_group = _global_cp_group()
        cu_local = None
        if packed_seq_params is not None:
            cu_global = packed_seq_params.cu_seqlens_q_padded
            if cu_global is None:
                cu_global = packed_seq_params.cu_seqlens_q
            cp_size = 1 if cp_group is None else cp_group.size()
            cu_local = torch.div(cu_global, cp_size, rounding_mode="floor").to(
                torch.int32
            )

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
                    rotary_pos_emb=self._rotary_for_pass(
                        ttt_pass, packed_seq_params, embeds.shape[0]
                    ),
                    packed_seq_params=packed_seq_params,
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
                    if packed_seq_params is not None:
                        embeds = roll_packed_left_cp(embeds, cu_local, cp_group)
                    else:
                        embeds = roll_tensor(
                            embeds, shifts=-1, dims=0, cp_group=cp_group
                        )[0]
        finally:
            # Clear saved K/V and hidden states even if a pass raises an error.
            for ttt_attention in self._ttt_attn_modules:
                ttt_attention.reset()
            self._ttt_prenorm_hidden = None
        return logits_by_pass
