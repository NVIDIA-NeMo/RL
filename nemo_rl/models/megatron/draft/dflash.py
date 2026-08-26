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

"""Megatron training implementation of the DFlash draft model.

This file matches the behavior of vLLM 0.26's
``vllm/model_executor/models/qwen3_dflash.py``. The main ideas are:

- The draft model does not run a causal decoder over the target sequence.
  It combines the target model's auxiliary hidden states once, then uses each
  draft layer's K/V weights to build that layer's trunk keys and values.
- An anchor at position ``p`` creates a block of ``W = gamma + 1`` tokens.
  Slot 0 contains the embedding of token ``x_p`` and provides context only.
  The other ``gamma`` slots predict ``x_{p+1} .. x_{p+gamma}``.
- A block can attend to target tokens before ``p`` and to every token in its
  own block. It cannot attend to other blocks. See
  :class:`BlockDraftAttention` for an example mask.
- Layers marked ``sliding_attention`` in the checkpoint's ``layer_types``
  are causal with a per-query ``sliding_window`` (vLLM's resolution for
  mixed-type checkpoints) and run as one windowed varlen call.
- Packed and unpacked inputs both use a flat THD trunk internally. With
  context parallelism (CP), full-attention layers move trunk K/V shards
  around a zigzag ring while each block stays on the rank that owns its
  anchor; sliding layers instead all-gather the layer's trunk K/V back to
  global order and keep their single windowed call.
- The draft model has no separate mask embedding or LM head. ``forward``
  receives detached references to the target model's mask-token embedding and
  current LM head, matching how serving shares those target-model weights.

The decoder is a standard MCore ``TransformerBlock`` whose ``core_attention``
modules are replaced by :class:`BlockDraftCoreAttention`. The DSpark model in
``draft/dspark.py`` subclasses :class:`DFlashDraftModel` and reuses this code.
"""

from __future__ import annotations

from typing import Any, Optional

import flash_attn
import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel import copy_to_tensor_model_parallel_region
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
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
        "nemo_rl/models/megatron/draft/dflash.py."
    )


SUPPORTED_BLOCK_SPECULATOR_TYPES = ("dflash", "dspark")


def sample_block_anchors(
    *,
    token_mask: Tensor,
    sample_mask: Tensor,
    input_ids: Tensor,
    num_anchors: int,
    generation_only: bool = True,
) -> tuple[Tensor, Tensor]:
    """Choose a fixed number of draft anchors for each sequence.

    An anchor at position ``p`` uses ``x_p`` as context and starts predicting
    at ``x_{p+1}``. By default, ``p`` is eligible only when ``x_{p+1}`` is a
    trained generation token. This excludes prompt, user, and tool spans that
    are handled as prefill during serving.

    If a sequence has too few eligible positions, positions are sampled with
    replacement. If it has none, position 0 is returned as a dummy anchor and
    marked invalid. The batch's token IDs seed a local CPU generator so all TP
    ranks choose the same anchors without changing the global random state.

    Args:
        token_mask: ``[B, S]`` mask. A value of 1 marks a trained target token.
        sample_mask: ``[B]`` mask. A value of 1 marks a valid sequence.
        input_ids: ``[B, S]`` token IDs used to create the sampling seed.
        num_anchors: Number of anchors ``N`` to return per sequence.
        generation_only: If true, choose anchors only before trained targets.

    Returns:
        A pair containing ``anchors`` with shape ``[B, N]`` and a Boolean
        ``anchor_valid`` mask with the same shape.
    """
    if num_anchors < 1:
        raise ValueError(f"num_anchors must be >= 1, got {num_anchors}.")
    batch_size = token_mask.shape[0]
    device = token_mask.device

    if generation_only:
        candidate_mask = token_mask[:, 1:] > 0.5  # p such that token_mask[p+1]==1
    else:
        candidate_mask = torch.ones_like(token_mask[:, 1:], dtype=torch.bool)
    candidate_mask = candidate_mask & (sample_mask.view(-1, 1) > 0.5)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(input_ids.sum().item()) % (2**63 - 1))

    anchors = torch.zeros((batch_size, num_anchors), dtype=torch.int64)
    anchor_valid = torch.zeros((batch_size, num_anchors), dtype=torch.bool)
    candidate_mask_cpu = candidate_mask.cpu()
    for row in range(batch_size):
        candidates = torch.nonzero(candidate_mask_cpu[row], as_tuple=True)[0]
        if candidates.numel() == 0:
            continue
        if candidates.numel() >= num_anchors:
            pick = torch.randperm(candidates.numel(), generator=gen)[:num_anchors]
        else:
            pick = torch.randint(candidates.numel(), (num_anchors,), generator=gen)
        anchors[row] = candidates[pick]
        anchor_valid[row] = True
    return anchors.to(device), anchor_valid.to(device)


def anchors_to_count_map(anchors: Tensor, anchor_valid: Tensor, seq_len: int) -> Tensor:
    """Convert anchor lists into a sequence-shaped count map.

    Dynamic batching knows how to truncate and reorder tensors shaped like
    ``[B, S]``, but it cannot safely transform the anchor layout ``[B, N]``.
    The count map solves that problem: each value tells how many blocks use
    that sequence position. Counts greater than one preserve anchors that were
    sampled more than once.

    Args:
        anchors: Anchor positions with shape ``[B, N]``.
        anchor_valid: Boolean validity mask with shape ``[B, N]``.
        seq_len: Sequence length ``S`` of the returned map.

    Returns:
        Per-position anchor counts with shape ``[B, S]``.
    """
    count_map = anchors.new_zeros((anchors.shape[0], seq_len), dtype=torch.int32)
    count_map.scatter_add_(1, anchors, anchor_valid.to(torch.int32))
    return count_map


def count_map_to_anchors(count_map: Tensor) -> tuple[Tensor, Tensor]:
    """Rebuild padded anchor lists from a count map.

    A position is repeated according to its count. Rows are padded to the
    largest number of anchors in the microbatch; padding uses position 0 and
    is marked invalid. If the entire microbatch is empty, one invalid column
    is still returned so downstream tensors never have a zero-width dimension.
    The order within a row is not preserved because blocks are independent.

    Args:
        count_map: Per-position anchor counts with shape ``[B, S]``.

    Returns:
        A pair containing padded ``anchors`` and its Boolean ``anchor_valid``
        mask, both with shape ``[B, N_max]``.
    """
    batch_size = count_map.shape[0]
    device = count_map.device
    max_blocks = max(int(count_map.sum(dim=1).max().item()), 1)

    anchors = torch.zeros((batch_size, max_blocks), dtype=torch.int64, device=device)
    anchor_valid = torch.zeros_like(anchors, dtype=torch.bool)
    for row in range(batch_size):
        positions = torch.nonzero(count_map[row], as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        repeated = torch.repeat_interleave(positions, count_map[row, positions].long())
        anchors[row, : repeated.numel()] = repeated
        anchor_valid[row, : repeated.numel()] = True
    return anchors, anchor_valid


# ---------------------------------------------------------------------------
# Exact two-part block attention (mask diagram on BlockDraftAttention).
# ---------------------------------------------------------------------------


def _fa_dense_forward(
    q: Tensor, k: Tensor, v: Tensor, softmax_scale: float
) -> tuple[Tensor, Tensor]:
    """Dense non-causal FA forward; returns (out [1,Sq,Hq,D], lse [1,Hq,Sq] fp32)."""
    out, softmax_lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        0.0,  # dropout_p
        softmax_scale,
        False,  # causal
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # return_softmax
    )
    return out, softmax_lse


def _fa_dense_backward(
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
    """Dense FA backward into preallocated dq/dk/dv (fed the joint (out, lse))."""
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
        False,  # causal
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # deterministic
        None,  # rng_state
    )


def _fa_varlen_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_left: int,
    window_right: int,
) -> tuple[Tensor, Tensor]:
    """Varlen non-causal FA forward; returns (out [Tq,Hq,D], lse [Hq,Tq] fp32)."""
    out, softmax_lse, _, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,  # dropout_p
        softmax_scale,
        False,  # causal
        window_left,
        window_right,
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # return_softmax
        None,  # block_table
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
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_left: int,
    window_right: int,
) -> None:
    """Varlen FA backward into preallocated dq/dk/dv (fed the joint (out, lse))."""
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
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,  # dropout_p
        softmax_scale,
        False,  # causal
        window_left,
        window_right,
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # deterministic
        None,  # rng_state
    )


def _ragged_arange(lengths: Tensor) -> Tensor:
    """``concat(arange(l) for l in lengths)`` without a python loop."""
    total = int(lengths.sum().item())
    if total == 0:
        return lengths.new_zeros((0,))
    starts = torch.cumsum(lengths, dim=0) - lengths
    return torch.arange(total, device=lengths.device) - starts.repeat_interleave(
        lengths
    )


def _global_cp_group() -> Optional[dist.ProcessGroup]:
    """The global CP group, or None before model-parallel init (unit tests)."""
    if not parallel_state.model_parallel_is_initialized():
        return None
    return parallel_state.get_context_parallel_group()


def _ring_send_recv(
    tensors: list[Tensor],
    send_rank: int,
    recv_rank: int,
    cp_group: dist.ProcessGroup,
) -> tuple[list[Tensor], list[Any]]:
    """Batched P2P hop: send each tensor to next, receive a like one from prev.

    The caller must keep the sent tensors alive until it waits on the requests.
    """
    recvs = [torch.empty_like(t) for t in tensors]
    ops = []
    for send_t, recv_t in zip(tensors, recvs):
        ops.append(dist.P2POp(dist.isend, send_t.contiguous(), send_rank, cp_group))
        ops.append(dist.P2POp(dist.irecv, recv_t, recv_rank, cp_group))
    return recvs, dist.batch_isend_irecv(ops)


def _merge_rows(
    out_acc: Tensor,
    lse_acc: Tensor,
    rows: Tensor,
    out_step: Tensor,
    lse_step: Tensor,
) -> None:
    """Merge attention over another set of keys into selected output rows.

    ``out_step`` is normalized only over the keys handled by this step. Its
    log-sum-exp value, ``lse_step``, provides the exact weight needed to merge
    it with earlier steps. The fp32 accumulators start at 0 and ``-inf``, so
    the first merge acts like a copy.

    Shapes are ``[NB, W, Hq, D]`` for ``out_acc``, ``[NB, Hq, W]`` for
    ``lse_acc``, ``[nb, W, Hq, D]`` for ``out_step``, and ``[nb, Hq, W]`` for
    ``lse_step``.
    """
    lse_old = lse_acc[rows]
    lse_new = torch.logaddexp(lse_old, lse_step)
    w_old = torch.exp(lse_old - lse_new).permute(0, 2, 1).unsqueeze(-1)
    w_new = torch.exp(lse_step - lse_new).permute(0, 2, 1).unsqueeze(-1)
    out_acc[rows] = out_acc[rows] * w_old + out_step.float() * w_new
    lse_acc[rows] = lse_new


class _RingStepGeometry:
    """Describe the visible keys for one context-parallel ring step.

    Forward and backward need the same gather and scatter indices. They are
    cheap to compute, so backward creates this object again instead of saving
    all index tensors from forward. :meth:`gather` builds the temporary K/V
    buffer, and :meth:`scatter_grads` maps its gradients back.

    At a ring step, the local trunk buffer contains two zigzag chunks from
    source rank ``src``: a front chunk ``c_f = src`` and a back chunk
    ``c_b = 2 * cp_size - 1 - src``. For an anchor at position ``p``, the
    number of visible tokens in this buffer is shown below. ``half`` is the
    number of tokens in either half of a rank's local zigzag shard.

    ``clamp(p - c_f * half, 0, half) + clamp(p - c_b * half, 0, half)``.

    These tokens form a local prefix. The largest whole multiple of ``chunk``
    becomes the shared dense-attention part. The remaining trunk tokens use
    variable-length attention. On the first ring step, that variable-length
    part also includes the block's own ``W`` keys.
    """

    def __init__(
        self,
        block_seq: Tensor,
        vis_len: Tensor,
        cu_local: Tensor,
        chunk: int,
        src: int,
        cp_size: int,
        block_width: int,
        include_own: bool,
    ):
        device = block_seq.device
        cu = cu_local.to(torch.long)
        half = ((cu[1:] - cu[:-1]) // 2)[block_seq]
        c_front, c_back = src, 2 * cp_size - 1 - src
        vis_front = torch.minimum((vis_len - c_front * half).clamp(min=0), half)
        vis_back = torch.minimum((vis_len - c_back * half).clamp(min=0), half)
        vis = vis_front + vis_back
        self.part_a_len = torch.div(vis, chunk, rounding_mode="floor") * chunk
        rem_len = vis - self.part_a_len
        seg_start = cu[:-1][block_seq]

        own_w = block_width if include_own else 0
        kv_len = rem_len + own_w
        self.active = torch.nonzero(kv_len > 0, as_tuple=True)[0]
        self.own_w = own_w
        num_active = self.active.numel()
        rem_active = rem_len[self.active]
        self.cu_k = torch.zeros(num_active + 1, device=device, dtype=torch.int32)
        self.cu_k[1:] = torch.cumsum(kv_len[self.active], dim=0).to(torch.int32)
        self.total_k = int(self.cu_k[-1].item()) if num_active else 0
        self.max_seqlen_k = int(kv_len[self.active].max().item()) if num_active else 0

        starts = self.cu_k[:-1].to(torch.long)
        rem_offsets = _ragged_arange(rem_active)
        rem_block = torch.arange(num_active, device=device).repeat_interleave(
            rem_active
        )
        # Map the visible remainder from the current trunk shard into the
        # temporary variable-length K/V buffer.
        act = self.active
        self.rem_dest = starts[rem_block] + rem_offsets
        self.rem_src = (seg_start[act] + self.part_a_len[act])[rem_block] + rem_offsets
        if own_w:
            own_start = (starts + rem_active).unsqueeze(1)
            self.own_dest = (own_start + torch.arange(own_w, device=device)).reshape(-1)
        else:
            self.own_dest = None

    def gather(self, trunk_step: Tensor, own: Tensor) -> Tensor:
        """Gather visible trunk rows and, on step 0, the block's own keys."""
        gathered = trunk_step.new_empty((self.total_k, *trunk_step.shape[1:]))
        gathered[self.rem_dest] = trunk_step[self.rem_src]
        if self.own_w:
            gathered[self.own_dest] = own[self.active].reshape(-1, *own.shape[2:])
        return gathered

    def scatter_grads(
        self, d_gathered: Tensor, d_trunk_step: Tensor, d_own: Optional[Tensor]
    ) -> None:
        """Scatter gathered-buffer gradients back to trunk and block tensors."""
        d_trunk_step.index_add_(0, self.rem_src, d_gathered[self.rem_dest])
        if self.own_w:
            d_own[self.active] += d_gathered[self.own_dest].reshape(
                self.active.numel(), self.own_w, *d_gathered.shape[1:]
            )


def _part_a_buckets(
    block_seq: Tensor, part_a_len: Tensor
) -> list[tuple[int, int, Tensor]]:
    """Group blocks by ``(subsequence, part_a_len)``; skip empty prefixes."""
    buckets: list[tuple[int, int, Tensor]] = []
    pairs = torch.stack([block_seq, part_a_len], dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    for pair_index in range(unique_pairs.shape[0]):
        seq = int(unique_pairs[pair_index, 0].item())
        prefix_len = int(unique_pairs[pair_index, 1].item())
        if prefix_len == 0:
            continue
        idx = torch.nonzero(inverse == pair_index, as_tuple=True)[0]
        buckets.append((seq, prefix_len, idx))
    return buckets


def _sliding_gather(
    trunk_k: Tensor,
    trunk_v: Tensor,
    k_own: Tensor,
    v_own: Tensor,
    block_seq: Tensor,
    vis_len: Tensor,
    cu_local: Tensor,
    window: int,
) -> tuple[Tensor, Tensor, Tensor, int, Tensor, Tensor, Tensor]:
    """Key buffers + maps for the sliding path's single windowed varlen call.

    Per block the visible keys are trunk ``[max(0, p - window + 1), p)`` plus
    the own ``W`` keys — contiguous positions, so FA's native window applies
    the per-slot mask exactly. Returns ``(kb, vb, cu_k, max_seqlen_k, rem_src,
    rem_dest, own_dest)``; backward recalls this instead of saving the maps
    (cheap integer math, same policy as :class:`_RingStepGeometry`).
    """
    num_blocks, block_width = k_own.shape[0], k_own.shape[1]
    device = block_seq.device
    cu = cu_local.to(torch.long)
    lo = (vis_len - (window - 1)).clamp(min=0)
    rem_len = vis_len - lo
    kv_len = rem_len + block_width
    cu_k = torch.zeros(num_blocks + 1, device=device, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(kv_len, dim=0).to(torch.int32)
    starts = cu_k[:-1].to(torch.long)
    rem_offsets = _ragged_arange(rem_len)
    rem_block = torch.arange(num_blocks, device=device).repeat_interleave(rem_len)
    rem_dest = starts[rem_block] + rem_offsets
    rem_src = (cu[block_seq] + lo)[rem_block] + rem_offsets
    own_start = (starts + rem_len).unsqueeze(1)
    own_dest = (own_start + torch.arange(block_width, device=device)).reshape(-1)

    kb = trunk_k.new_empty((int(cu_k[-1].item()), *trunk_k.shape[1:]))
    vb = torch.empty_like(kb)
    kb[rem_dest] = trunk_k[rem_src]
    vb[rem_dest] = trunk_v[rem_src]
    kb[own_dest] = k_own.reshape(-1, *k_own.shape[2:])
    vb[own_dest] = v_own.reshape(-1, *v_own.shape[2:])
    return kb, vb, cu_k, int(kv_len.max().item()), rem_src, rem_dest, own_dest


def _unzigzag_rows(cu_local: Tensor, cp_size: int, t_local: int) -> Tensor:
    """Row map from global THD order into the rank-major all-gathered buffer.

    Rank ``r``'s shard holds, per sequence, zigzag chunks ``r`` and
    ``2 * cp_size - 1 - r`` back to back, so global token ``g`` of a sequence
    lives at buffer row ``src_rank * t_local + local_offset``. Indexing the
    all-gathered ``[cp_size * t_local, ...]`` buffer with the result yields
    the sequence-packed trunk in global order; backward reuses the same map
    (a bijection) to lay gradient rows out for ``reduce_scatter_tensor``.
    """
    cu = cu_local.to(torch.long)
    device = cu_local.device
    rows = []
    for seq in range(int(cu.numel()) - 1):
        half = (int(cu[seq + 1]) - int(cu[seq])) // 2
        if half == 0:
            continue
        g = torch.arange(2 * cp_size * half, device=device)
        chunk_idx = torch.div(g, half, rounding_mode="floor")
        back = chunk_idx >= cp_size
        src_rank = torch.where(back, 2 * cp_size - 1 - chunk_idx, chunk_idx)
        local = int(cu[seq]) + back.long() * half + g % half
        rows.append(src_rank * t_local + local)
    return torch.cat(rows)


def _allgather_trunk(
    trunk_k: Tensor,
    trunk_v: Tensor,
    cu_local: Tensor,
    cp_size: int,
    cp_group: dist.ProcessGroup,
) -> tuple[Tensor, Tensor, Tensor]:
    """All-gather zigzag trunk K/V shards into global THD order.

    Transient by design: forward and backward each call this instead of
    saving the gathered buffers, the same recompute policy as
    :class:`_RingStepGeometry`.
    """
    t_local = trunk_k.shape[0]
    k_all = trunk_k.new_empty((cp_size * t_local, *trunk_k.shape[1:]))
    v_all = torch.empty_like(k_all)
    dist.all_gather_into_tensor(k_all, trunk_k.contiguous(), group=cp_group)
    dist.all_gather_into_tensor(v_all, trunk_v.contiguous(), group=cp_group)
    unzig = _unzigzag_rows(cu_local, cp_size, t_local)
    return k_all[unzig], v_all[unzig], unzig


class BlockDraftAttention(torch.autograd.Function):
    """Attend each draft block to the earlier trunk and to its own tokens.

    A block anchored at position ``p`` can attend to exactly two sets of keys:

    - Trunk positions ``[0, p)``. The anchor itself is not included.
    - All ``W`` positions in the same block, with bidirectional attention.

    It cannot attend to any other draft block. Only block positions are
    queries; trunk positions provide keys and values only.

    The example below uses prompt tokens ``p1..p4``, response tokens
    ``r1..r3``, one four-token block (``gamma = 3``, ``W = 4``) at each
    response token, and ``chunk = 4``. ``A`` marks a shared, full-chunk trunk
    prefix. ``x`` marks the remaining visible trunk keys or the block's own
    keys. A blank cell is masked::

                   |      trunk keys      | r1's block  | r2's block  | r3's block  |
                   | p1 p2 p3 p4 r1 r2 r3 | r1  m  m  m | r2  m  m  m | r3  m  m  m |
        =============================================================================
        r1 (pos 4) |  A  A  A  A          |  x  x  x  x |             |             |
        m  (pos 5) |  A  A  A  A          |  x  x  x  x |             |             |
        m  (pos 6) |  A  A  A  A          |  x  x  x  x |             |             |
        m  (pos 7) |  A  A  A  A          |  x  x  x  x |             |             |
        -----------------------------------------------------------------------------
        r2 (pos 5) |  A  A  A  A  x       |             |  x  x  x  x |             |
        m  (pos 6) |  A  A  A  A  x       |             |  x  x  x  x |             |
        m  (pos 7) |  A  A  A  A  x       |             |  x  x  x  x |             |
        m  (pos 8) |  A  A  A  A  x       |             |  x  x  x  x |             |
        -----------------------------------------------------------------------------
        r3 (pos 6) |  A  A  A  A  x  x    |             |             |  x  x  x  x |
        m  (pos 7) |  A  A  A  A  x  x    |             |             |  x  x  x  x |
        m  (pos 8) |  A  A  A  A  x  x    |             |             |  x  x  x  x |
        m  (pos 9) |  A  A  A  A  x  x    |             |             |  x  x  x  x |
        =============================================================================

    The anchor is represented by its embedding in block slot 0. This matters
    during serving: the anchor is the newest accepted token, so its target
    hidden state is not available yet. Earlier response tokens can later
    appear in the trunk, but a block never reads its own anchor from the trunk.

    The computation is split without changing the result:

    1. With CP, blocks stay on their owner rank while trunk K/V shards move
       from rank to rank for ``cp_size`` steps. With one rank, this is one step.
    2. At each step, a block's visible local trunk keys form a prefix. The
       prefix is split at ``part_a_len = (vis // chunk) * chunk``.
    3. Blocks with the same sequence and ``part_a_len`` share one dense
       FlashAttention call for the full-chunk prefix (the ``A`` cells).
       One variable-length call handles the remainder and, on step 0, each
       block's own keys (the ``x`` cells).
    4. These calls cover disjoint key sets. Their outputs and log-sum-exp
       values are merged to recover the exact softmax over all visible keys.

    Backward repeats the same ring steps and kernel split. It uses the final
    output and log-sum-exp so each call computes its part of the joint-softmax
    gradient. Trunk K/V gradients travel with their shard and finish with one
    last hop back to the rank that owns them.

    ``window > 0`` selects sliding-window attention for the layer instead
    (official 35B-style checkpoints: ``layer_types`` + ``sliding_window``).
    vLLM resolves such layers as CAUSAL sliding windows
    (``qwen3_dflash._resolve_layer_attention``: mixed-type checkpoints are
    causal on sliding layers, non-causal on full layers), so key ``k`` is
    visible to query ``q`` iff ``0 <= q - k <= window - 1`` — block-internal
    attention becomes lower-triangular too. The visible keys
    ``[max(0, p - window + 1), p)`` plus the own block form one contiguous
    position run, so the whole layer becomes a single varlen call with FA's
    native ``(window - 1, 0)`` window: the per-slot mask is exact, with no
    kernel split, no LSE merge, and no ring. Under CP the sliding layer
    all-gathers its trunk K/V shard back to global order first and keeps
    that single call; backward re-gathers and returns trunk gradients home
    with one reduce-scatter. Windowing the ring instead would break FA's
    bottom-right causal alignment on mid-ring key segments (per-slot lower
    bounds need a staircase split the full path never has), and a block only
    ever reads ``window + W`` keys, so the transient gather is the cheaper
    exact form.

    Floating-point inputs use contiguous bf16/fp16 FlashAttention layouts.
    Tensor shapes passed to ``apply`` are:

    - ``q``: ``[NB, W, Hq, D]`` block queries after RoPE.
    - ``k_own`` and ``v_own``: ``[NB, W, Hkv, D]`` block-local K/V.
    - ``trunk_k`` and ``trunk_v``: ``[T, Hkv, D]`` local trunk K/V shard.
    - ``block_seq``: ``[NB]`` subsequence index for each block.
    - ``vis_len``: ``[NB]`` anchor position, which is also the visible trunk
      length in global subsequence coordinates.
    - ``cu_seqlens_local``: ``[B + 1]`` local subsequence boundaries.
    - ``chunk``: Size of the shared dense-attention prefix buckets.
    - ``window``: Sliding window size for this layer; 0 means full attention.
    - ``softmax_scale``: Scale applied to attention scores.
    - ``cp_group``: CP process group, or ``None`` on a single rank.

    The result has shape ``[NB, W, Hq, D]`` and equals attention over
    ``trunk[seq, :p]`` together with the block's own keys.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: Tensor,
        k_own: Tensor,
        v_own: Tensor,
        trunk_k: Tensor,
        trunk_v: Tensor,
        block_seq: Tensor,
        vis_len: Tensor,
        cu_seqlens_local: Tensor,
        chunk: int,
        window: int,
        softmax_scale: float,
        cp_group: Optional[dist.ProcessGroup],
    ) -> Tensor:
        num_blocks, block_width, num_q_heads, head_dim = q.shape
        device = q.device
        cp_size = 1 if cp_group is None else cp_group.size()
        cp_rank = 0 if cp_group is None else cp_group.rank()
        if cp_size > 1:
            global_ranks = dist.get_process_group_ranks(cp_group)
            send_rank = global_ranks[(cp_rank + 1) % cp_size]
            recv_rank = global_ranks[(cp_rank - 1) % cp_size]

        cu = cu_seqlens_local.to(torch.long)
        seq_lens = (cu[1:] - cu[:-1]) * cp_size
        if bool((vis_len > seq_lens[block_seq]).any().item()):
            raise ValueError("vis_len exceeds its subsequence's trunk length.")

        cu_q = block_width * torch.arange(
            num_blocks + 1, device=device, dtype=torch.int32
        )

        if window > 0:
            if cp_size > 1:
                k_attn, v_attn, _ = _allgather_trunk(
                    trunk_k, trunk_v, cu, cp_size, cp_group
                )
                cu_attn = cu * cp_size
            else:
                k_attn, v_attn, cu_attn = trunk_k, trunk_v, cu
            kb, vb, cu_k, max_k, _, _, _ = _sliding_gather(
                k_attn, v_attn, k_own, v_own, block_seq, vis_len, cu_attn, window
            )
            out_b, lse = _fa_varlen_forward(
                q.reshape(-1, num_q_heads, head_dim),
                kb,
                vb,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=block_width,
                max_seqlen_k=max_k,
                softmax_scale=softmax_scale,
                window_left=window - 1,
                window_right=0,
            )
            out = out_b.view(num_blocks, block_width, num_q_heads, head_dim)
            ctx.save_for_backward(
                q,
                k_own,
                v_own,
                trunk_k,
                trunk_v,
                block_seq,
                vis_len,
                cu_seqlens_local,
                out,
                lse,
            )
            ctx.chunk = chunk
            ctx.window = window
            ctx.softmax_scale = softmax_scale
            ctx.cp_group = cp_group
            ctx.cp_size = cp_size
            return out

        out_acc = torch.zeros_like(q, dtype=torch.float32)
        lse_acc = q.new_full(
            (num_blocks, num_q_heads, block_width), float("-inf"), dtype=torch.float32
        )

        k_cur, v_cur = trunk_k, trunk_v
        reqs: list[Any] = []
        for step in range(cp_size):
            if step + 1 < cp_size:
                (k_next, v_next), reqs = _ring_send_recv(
                    [k_cur, v_cur], send_rank, recv_rank, cp_group
                )
            geo = _RingStepGeometry(
                block_seq,
                vis_len,
                cu,
                chunk,
                (cp_rank - step) % cp_size,
                cp_size,
                block_width,
                include_own=step == 0,
            )
            if geo.active.numel():
                kb = geo.gather(k_cur, k_own)
                vb = geo.gather(v_cur, v_own)
                nb = geo.active.numel()
                q_act = q[geo.active].reshape(-1, num_q_heads, head_dim)
                out_b, lse_b = _fa_varlen_forward(
                    q_act,
                    kb,
                    vb,
                    cu_seqlens_q=cu_q[: nb + 1],
                    cu_seqlens_k=geo.cu_k,
                    max_seqlen_q=block_width,
                    max_seqlen_k=geo.max_seqlen_k,
                    softmax_scale=softmax_scale,
                    window_left=-1,
                    window_right=-1,
                )
                _merge_rows(
                    out_acc,
                    lse_acc,
                    geo.active,
                    out_b.view(nb, block_width, num_q_heads, head_dim),
                    lse_b.view(num_q_heads, nb, block_width).permute(1, 0, 2).float(),
                )
            for seq, prefix_len, idx in _part_a_buckets(block_seq, geo.part_a_len):
                seg_start = int(cu[seq].item())
                nbk = idx.shape[0]
                q_bucket = q[idx].reshape(1, -1, num_q_heads, head_dim)
                out_a, lse_a = _fa_dense_forward(
                    q_bucket,
                    k_cur[seg_start : seg_start + prefix_len].unsqueeze(0),
                    v_cur[seg_start : seg_start + prefix_len].unsqueeze(0),
                    softmax_scale,
                )
                _merge_rows(
                    out_acc,
                    lse_acc,
                    idx,
                    out_a.view(nbk, block_width, num_q_heads, head_dim),
                    lse_a.view(num_q_heads, nbk, block_width).permute(1, 0, 2).float(),
                )
            if step + 1 < cp_size:
                for req in reqs:
                    req.wait()
                k_cur, v_cur = k_next, v_next

        out = out_acc.to(q.dtype)
        ctx.save_for_backward(
            q,
            k_own,
            v_own,
            trunk_k,
            trunk_v,
            block_seq,
            vis_len,
            cu_seqlens_local,
            out,
            lse_acc,
        )
        ctx.chunk = chunk
        ctx.window = window
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, dout: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        (
            q,
            k_own,
            v_own,
            trunk_k,
            trunk_v,
            block_seq,
            vis_len,
            cu_seqlens_local,
            out,
            lse_joint,
        ) = ctx.saved_tensors
        chunk = ctx.chunk
        window = ctx.window
        softmax_scale = ctx.softmax_scale
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size

        num_blocks, block_width, num_q_heads, head_dim = q.shape
        device = q.device
        dout = dout.contiguous()
        cp_rank = 0 if cp_group is None else cp_group.rank()
        if cp_size > 1:
            global_ranks = dist.get_process_group_ranks(cp_group)
            send_rank = global_ranks[(cp_rank + 1) % cp_size]
            recv_rank = global_ranks[(cp_rank - 1) % cp_size]

        cu = cu_seqlens_local.to(torch.long)
        cu_q = block_width * torch.arange(
            num_blocks + 1, device=device, dtype=torch.int32
        )

        if window > 0:
            # Replay the single windowed varlen call (saved lse is [Hq, NB*W]).
            if cp_size > 1:
                k_attn, v_attn, unzig = _allgather_trunk(
                    trunk_k, trunk_v, cu, cp_size, cp_group
                )
                cu_attn = cu * cp_size
            else:
                k_attn, v_attn, cu_attn = trunk_k, trunk_v, cu
            kb, vb, cu_k, max_k, rem_src, rem_dest, own_dest = _sliding_gather(
                k_attn, v_attn, k_own, v_own, block_seq, vis_len, cu_attn, window
            )
            q_flat, out_flat, dout_flat = (
                t.reshape(-1, num_q_heads, head_dim) for t in (q, out, dout)
            )
            dq_flat = torch.empty_like(q_flat)
            dkb = torch.empty_like(kb)
            dvb = torch.empty_like(vb)
            _fa_varlen_backward(
                dout=dout_flat,
                q=q_flat,
                k=kb,
                v=vb,
                out=out_flat,
                lse=lse_joint,
                dq=dq_flat,
                dk=dkb,
                dv=dvb,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=block_width,
                max_seqlen_k=max_k,
                softmax_scale=softmax_scale,
                window_left=window - 1,
                window_right=0,
            )
            # Blocks can share trunk rows, so trunk grads accumulate in fp32.
            dk_trunk = torch.zeros_like(k_attn, dtype=torch.float32)
            dv_trunk = torch.zeros_like(dk_trunk)
            dk_trunk.index_add_(0, rem_src, dkb[rem_dest].float())
            dv_trunk.index_add_(0, rem_src, dvb[rem_dest].float())
            if cp_size > 1:
                # unzig is a bijection, so scattering through it lays the
                # global-order grads out rank-major; reduce-scatter then sums
                # every rank's contribution into the owner's local shard.
                t_local = trunk_k.shape[0]
                dk_buf = dk_trunk.new_zeros((cp_size * t_local, *dk_trunk.shape[1:]))
                dv_buf = torch.zeros_like(dk_buf)
                dk_buf[unzig] = dk_trunk
                dv_buf[unzig] = dv_trunk
                dk_trunk = dk_trunk.new_empty((t_local, *dk_trunk.shape[1:]))
                dv_trunk = torch.empty_like(dk_trunk)
                dist.reduce_scatter_tensor(dk_trunk, dk_buf, group=cp_group)
                dist.reduce_scatter_tensor(dv_trunk, dv_buf, group=cp_group)
            return (
                dq_flat.view_as(q),
                dkb[own_dest].view_as(k_own),
                dvb[own_dest].view_as(v_own),
                dk_trunk.to(trunk_k.dtype),
                dv_trunk.to(trunk_v.dtype),
                None,  # block_seq
                None,  # vis_len
                None,  # cu_seqlens_local
                None,  # chunk
                None,  # window
                None,  # softmax_scale
                None,  # cp_group
            )

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk_own = torch.zeros_like(k_own, dtype=torch.float32)
        dv_own = torch.zeros_like(v_own, dtype=torch.float32)
        # dK/dV accumulate (and travel the ring) in fp32, paired with the KV
        # in hand; after the loop one final hop delivers them home.
        dk_cur = torch.zeros_like(trunk_k, dtype=torch.float32)
        dv_cur = torch.zeros_like(dk_cur)

        k_cur, v_cur = trunk_k, trunk_v
        for step in range(cp_size):
            kv_reqs: list[Any] = []
            if step + 1 < cp_size:
                (k_next, v_next), kv_reqs = _ring_send_recv(
                    [k_cur, v_cur], send_rank, recv_rank, cp_group
                )
            geo = _RingStepGeometry(
                block_seq,
                vis_len,
                cu,
                chunk,
                (cp_rank - step) % cp_size,
                cp_size,
                block_width,
                include_own=step == 0,
            )
            if geo.active.numel():
                kb = geo.gather(k_cur, k_own)
                vb = geo.gather(v_cur, v_own)
                nb = geo.active.numel()
                q_act, out_act, dout_act = (
                    t[geo.active].reshape(-1, num_q_heads, head_dim)
                    for t in (q, out, dout)
                )
                lse_act = lse_joint[geo.active].permute(1, 0, 2)
                lse_act = lse_act.reshape(num_q_heads, -1).contiguous()
                dq_b = torch.empty_like(q_act)
                dkb = torch.empty_like(kb)
                dvb = torch.empty_like(vb)
                _fa_varlen_backward(
                    dout=dout_act,
                    q=q_act,
                    k=kb,
                    v=vb,
                    out=out_act,
                    lse=lse_act,
                    dq=dq_b,
                    dk=dkb,
                    dv=dvb,
                    cu_seqlens_q=cu_q[: nb + 1],
                    cu_seqlens_k=geo.cu_k,
                    max_seqlen_q=block_width,
                    max_seqlen_k=geo.max_seqlen_k,
                    softmax_scale=softmax_scale,
                    window_left=-1,
                    window_right=-1,
                )
                dq[geo.active] += dq_b.float().view(nb, -1, num_q_heads, head_dim)
                geo.scatter_grads(dkb.float(), dk_cur, dk_own)
                geo.scatter_grads(dvb.float(), dv_cur, dv_own)
            for seq, prefix_len, idx in _part_a_buckets(block_seq, geo.part_a_len):
                seg_start = int(cu[seq].item())
                nbk = idx.shape[0]
                q_bucket, out_bucket, dout_bucket = (
                    t[idx].reshape(1, -1, num_q_heads, head_dim) for t in (q, out, dout)
                )
                lse_bucket = lse_joint[idx].permute(1, 0, 2)
                lse_bucket = lse_bucket.reshape(1, num_q_heads, -1).contiguous()
                k_bucket = k_cur[seg_start : seg_start + prefix_len].unsqueeze(0)
                v_bucket = v_cur[seg_start : seg_start + prefix_len].unsqueeze(0)
                dq_a = torch.empty_like(q_bucket)
                dk_a = torch.empty_like(k_bucket)
                dv_a = torch.empty_like(v_bucket)
                _fa_dense_backward(
                    dout=dout_bucket,
                    q=q_bucket,
                    k=k_bucket,
                    v=v_bucket,
                    out=out_bucket,
                    lse=lse_bucket,
                    dq=dq_a,
                    dk=dk_a,
                    dv=dv_a,
                    softmax_scale=softmax_scale,
                )
                dq[idx] += dq_a.view(nbk, block_width, num_q_heads, head_dim).float()
                dk_cur[seg_start : seg_start + prefix_len] += dk_a[0].float()
                dv_cur[seg_start : seg_start + prefix_len] += dv_a[0].float()
            if step + 1 < cp_size:
                for req in kv_reqs:
                    req.wait()
                # dK/dV hop only AFTER this step's contribution is folded in.
                (dk_next, dv_next), dkv_reqs = _ring_send_recv(
                    [dk_cur, dv_cur], send_rank, recv_rank, cp_group
                )
                for req in dkv_reqs:
                    req.wait()
                k_cur, v_cur = k_next, v_next
                dk_cur, dv_cur = dk_next, dv_next
        if cp_size > 1:
            (dk_home, dv_home), dkv_reqs = _ring_send_recv(
                [dk_cur, dv_cur], send_rank, recv_rank, cp_group
            )
            for req in dkv_reqs:
                req.wait()
            dk_cur, dv_cur = dk_home, dv_home

        return (
            dq.to(q.dtype),
            dk_own.to(k_own.dtype),
            dv_own.to(v_own.dtype),
            dk_cur.to(trunk_k.dtype),
            dv_cur.to(trunk_v.dtype),
            None,  # block_seq
            None,  # vis_len
            None,  # cu_seqlens_local
            None,  # chunk
            None,  # window
            None,  # softmax_scale
            None,  # cp_group
        )


class BlockDraftCoreAttention(torch.nn.Module):
    """Adapt an MCore decoder layer to block-draft attention.

    MCore supplies the block Q/K/V as one flattened stream with shape
    ``[NB * W, 1, H, D]``. Before running the decoder, the model calls
    :meth:`stage_trunk` to store this layer's trunk K/V and block metadata.
    :meth:`forward` reshapes the stream and calls :class:`BlockDraftAttention`.

    The standard MCore mask, bias, and packing arguments are accepted only to
    match its interface. They are ignored because block attention creates its
    own mask. ``window > 0`` runs this layer with sliding-window attention.
    The model must call :meth:`reset` after every decoder run so tensors and
    autograd graphs do not leak into the next microbatch.
    """

    def __init__(self, config: TransformerConfig, chunk: int, window: int):
        super().__init__()
        if getattr(config, "attention_dropout", 0.0):
            raise ValueError(
                "BlockDraftCoreAttention does not support attention dropout."
            )
        if int(getattr(config, "context_parallel_size", 1) or 1) != 1:
            raise ValueError(
                "BlockDraftCoreAttention requires context_parallel_size == 1."
            )
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}.")
        if window < 0:
            raise ValueError(f"window must be >= 0 (0 = full attention), got {window}.")
        self.chunk = int(chunk)
        self.window = int(window)
        self.softmax_scale: Optional[float] = getattr(config, "softmax_scale", None)
        self._trunk_k: Optional[Tensor] = None
        self._trunk_v: Optional[Tensor] = None
        self._block_seq: Optional[Tensor] = None
        self._vis_len: Optional[Tensor] = None
        self._cu_seqlens_local: Optional[Tensor] = None
        self._cp_group: Optional[dist.ProcessGroup] = None
        self._block_width: int = 0

    def stage_trunk(
        self,
        trunk_k: Tensor,
        trunk_v: Tensor,
        block_seq: Tensor,
        vis_len: Tensor,
        cu_seqlens_local: Tensor,
        block_width: int,
        cp_group: Optional[dist.ProcessGroup],
    ) -> None:
        """Stage this layer's THD trunk K/V ``[T, Hkv, D]`` and block metadata."""
        self._trunk_k = trunk_k
        self._trunk_v = trunk_v
        self._block_seq = block_seq
        self._vis_len = vis_len
        self._cu_seqlens_local = cu_seqlens_local
        self._cp_group = cp_group
        self._block_width = int(block_width)

    def reset(self) -> None:
        """Drop staged trunk references (frees the autograd graph)."""
        self._trunk_k = None
        self._trunk_v = None
        self._block_seq = None
        self._vis_len = None
        self._cu_seqlens_local = None
        self._cp_group = None
        self._block_width = 0

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
        if self._trunk_k is None:
            raise RuntimeError(
                "BlockDraftCoreAttention.forward called without staged trunk "
                "K/V; the draft model must call stage_trunk() first."
            )
        seqlen, block_width = query.shape[0], self._block_width
        softmax_scale = self.softmax_scale or query.shape[-1] ** -0.5

        # sbhd (b=1) -> [NB, W, H, D]; the reshapes themselves reject any
        # stream that is not batch-1 block-major.
        num_blocks = seqlen // block_width
        q = query.reshape(num_blocks, block_width, *query.shape[2:])
        k_own = key.reshape(num_blocks, block_width, *key.shape[2:])
        v_own = value.reshape(num_blocks, block_width, *value.shape[2:])

        out = BlockDraftAttention.apply(
            q,
            k_own,
            v_own,
            self._trunk_k,
            self._trunk_v,
            self._block_seq,
            self._vis_len,
            self._cu_seqlens_local,
            self.chunk,
            self.window,
            softmax_scale,
            self._cp_group,
        )
        # [NB, W, Hq, D] -> [S, 1, Hq * D] (core_attention output contract).
        return out.reshape(seqlen, 1, -1)


class DFlashDraftModel(MegatronModule):
    """Training model shared by the DFlash and DSpark block drafters.

    A DFlash block has ``W = gamma + 1`` slots. Slot 0 contains the anchor and
    supplies context, but its logits are not trained. The remaining ``gamma``
    slots predict new tokens. DSpark subclasses this model and changes
    ``method`` and ``block_width`` for its serving and loss behavior.

    ``layer_windows`` gives each decoder layer its sliding window (0 = full
    attention), mirroring the checkpoint's ``layer_types``/``sliding_window``;
    ``None`` means all layers use full attention.
    """

    speculator_type = "dflash"

    def __init__(
        self,
        config: TransformerConfig,
        *,
        gamma: int,
        mask_token_id: int,
        num_aux_hidden_states: int,
        target_hidden_size: Optional[int] = None,
        trunk_chunk: int = 1024,
        layer_windows: Optional[list[int]] = None,
        block_width: Optional[int] = None,
    ):
        super().__init__(config=config)
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {gamma}.")
        if bool(getattr(config, "add_bias_linear", False)):
            raise NotImplementedError(
                "Block draft trunk projection assumes bias-free qkv (Qwen3 style)."
            )
        tp_size = int(config.tensor_model_parallel_size or 1)
        if int(config.num_query_groups or config.num_attention_heads) % tp_size != 0:
            raise NotImplementedError(
                "Block draft trunk projection requires num_query_groups % TP == 0."
            )
        self.config = config
        self.gamma = int(gamma)
        # DFlash adds one context-only anchor slot. DSpark overrides the width
        # because its anchor slot also makes a prediction.
        self.block_width = int(block_width) if block_width is not None else gamma + 1
        self.mask_token_id = int(mask_token_id)
        self.num_aux_hidden_states = int(num_aux_hidden_states)
        self.trunk_chunk = int(trunk_chunk)
        if layer_windows is None:
            layer_windows = [0] * int(config.num_layers)
        if len(layer_windows) != int(config.num_layers):
            raise ValueError(
                f"layer_windows has {len(layer_windows)} entries for "
                f"{config.num_layers} decoder layers."
            )
        self.layer_windows = [int(w) for w in layer_windows]

        target_hidden = int(target_hidden_size or config.hidden_size)
        # Keep this Linear replicated across TP ranks. The input is already
        # replicated, and every rank needs the complete output to build its
        # local K/V heads. Sharding it would add a collective and would not
        # match the dense ``fc.weight`` stored in existing checkpoints.
        self.fc = torch.nn.Linear(
            target_hidden * self.num_aux_hidden_states,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
        )
        # There is no trainable mask embedding here. Mask slots reuse the
        # detached mask-token embedding supplied by the target model.
        self.hidden_norm = TENorm(config, config.hidden_size, config.layernorm_epsilon)

        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            qk_layernorm=bool(getattr(config, "qk_layernorm", False))
        )
        self.decoder = TransformerBlock(
            config=config,
            spec=layer_spec,
            post_layer_norm=True,
            pre_process=True,
            post_process=True,
        )

        self.rotary_pos_emb = RotaryEmbedding(
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

        self._block_attn_modules: list[BlockDraftCoreAttention] = []
        for layer, layer_window in zip(self.decoder.layers, self.layer_windows):
            core = BlockDraftCoreAttention(config, self.trunk_chunk, layer_window)
            layer.self_attention.core_attention = core
            self._block_attn_modules.append(core)

        # There is no draft LM head. ``forward`` uses the detached target LM
        # head, which matches both existing checkpoints and vLLM serving.

    def _project_trunk_kv(
        self, self_attention: torch.nn.Module, trunk_hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Build one decoder layer's trunk keys and values.

        Megatron stores each TP rank's Q/K/V weights in one fused tensor. For
        every query group, its rows are ordered as ``[Q rows, K row, V row]``.
        This method selects only the K and V rows and applies them to the
        already-normalized trunk. It intentionally skips the decoder layer's
        input normalization because the serving context path skips it too.

        Args:
            self_attention: Decoder self-attention module that owns the fused
                Q/K/V weight and optional K normalization.
            trunk_hidden: Normalized trunk states with shape ``[T, B, H]``.

        Returns:
            The layer's key and value tensors, each with shape
            ``[T, B, Hkv_local, D]``.
        """
        num_groups = self_attention.num_query_groups_per_partition
        heads_per_group = self_attention.num_attention_heads_per_partition // num_groups
        head_dim = self_attention.hidden_size_per_attention_head
        weight = self_attention.linear_qkv.weight
        grouped = weight.view(num_groups, (heads_per_group + 2) * head_dim, -1)
        q_rows = heads_per_group * head_dim
        k_weight = grouped[:, q_rows : q_rows + head_dim].reshape(-1, weight.shape[1])
        v_weight = grouped[:, q_rows + head_dim :].reshape(-1, weight.shape[1])

        seq_len, batch = trunk_hidden.shape[0], trunk_hidden.shape[1]
        key = F.linear(trunk_hidden, k_weight).view(seq_len, batch, num_groups, -1)
        value = F.linear(trunk_hidden, v_weight).view(seq_len, batch, num_groups, -1)
        k_layernorm = getattr(self_attention, "k_layernorm", None)
        if k_layernorm is not None:
            key = k_layernorm(key)
        return key, value

    def _flatten_to_thd(
        self,
        taps: Tensor,
        input_embeds: Tensor,
        anchors: Tensor,
        packed_seq_params: Optional[Any],
        block_seq_idx: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, Any, tuple]:
        """Convert packed or padded inputs to the same flat THD layout.

        Packed inputs are already a rank-local zigzag shard. Their anchors are
        a flat list of blocks owned by this rank. For padded inputs, the batch
        dimension is flattened into ``B`` equal-length subsequences. After
        this conversion, the rest of the model can use one code path.

        Args:
            taps: Concatenated target hidden states in packed or padded layout.
            input_embeds: Target token embeddings in the same layout as
                ``taps``.
            anchors: Anchor positions for all local blocks.
            packed_seq_params: Packing metadata, or ``None`` for padded input.
            block_seq_idx: For packed input, the subsequence index of each
                local block.

        Returns:
            ``(taps_flat, embeds_flat, block_seq, anchors_flat, cu_local,
            pos_in_seq, max_len, cp_group, out_shape)``. These values contain
            the flat trunk tensors, block-to-sequence mapping, local sequence
            boundaries, global token positions, CP group, and final logit
            shape.
        """
        device = taps.device
        if packed_seq_params is not None:
            from nemo_rl.algorithms.loss.utils import packed_zigzag_token_coords

            cp_group = _global_cp_group()
            cp_size = 1 if cp_group is None else cp_group.size()
            cp_rank = 0 if cp_group is None else cp_group.rank()
            cu_global = packed_seq_params.cu_seqlens_q_padded
            if cu_global is None:
                cu_global = packed_seq_params.cu_seqlens_q
            cu_local = (cu_global // cp_size).to(torch.long)
            _, pos_in_seq = packed_zigzag_token_coords(cu_global, cp_rank, cp_size)
            block_seq = block_seq_idx.to(torch.long)
            anchors_flat = anchors.reshape(-1).to(torch.long)
            max_len = int((cu_local[1:] - cu_local[:-1]).max().item()) * cp_size
            embeds_flat = input_embeds.reshape(-1, input_embeds.shape[-1])
            out_shape = (anchors_flat.shape[0], self.block_width, -1)
            return (
                taps,
                embeds_flat,
                block_seq,
                anchors_flat,
                cu_local,
                pos_in_seq.to(device),
                max_len,
                cp_group,
                out_shape,
            )

        seq_len, batch = taps.shape[0], taps.shape[1]
        num_anchors = anchors.shape[1]
        if anchors.shape[0] != batch:
            raise ValueError(f"anchors batch {anchors.shape[0]} != taps batch {batch}.")
        taps_flat = taps.permute(1, 0, 2).reshape(batch * seq_len, 1, -1)
        embeds_flat = input_embeds.permute(1, 0, 2).reshape(batch * seq_len, -1)
        block_seq = torch.arange(batch, device=device).repeat_interleave(num_anchors)
        anchors_flat = anchors.reshape(-1)
        cu_local = torch.arange(0, batch + 1, device=device, dtype=torch.long) * seq_len
        pos_in_seq = torch.arange(seq_len, device=device).repeat(batch)
        out_shape = (batch, num_anchors, self.block_width, -1)
        return (
            taps_flat,
            embeds_flat,
            block_seq,
            anchors_flat,
            cu_local,
            pos_in_seq,
            seq_len,
            None,
            out_shape,
        )

    def forward(
        self,
        *,
        taps: Tensor,
        input_embeds: Tensor,
        anchors: Tensor,
        anchor_valid: Tensor,
        lm_head_weight: Tensor,
        mask_embedding: Tensor,
        packed_seq_params: Optional[Any] = None,
        block_seq_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the block drafter for one batch.

        The method builds per-layer trunk K/V, creates each block from one
        anchor embedding plus mask embeddings, runs the decoder, and projects
        its outputs with the target model's LM head.

        Args:
            taps: Target auxiliary hidden states. Padded shape is
                ``[S, B, k * target_h]``; packed shape is the local zigzag
                shard ``[T_local, 1, k * target_h]``.
            input_embeds: Unshifted target embeddings. Padded shape is
                ``[S, B, h]``; packed shape is ``[T_local, 1, h]``.
            anchors: Anchor positions. Padded shape is ``[B, N]``; packed
                shape is ``[NB]`` for the blocks owned by this rank.
            anchor_valid: Validity mask with the same layout as ``anchors``.
                Invalid blocks keep tensor shapes static and are masked by the
                loss.
            lm_head_weight: Detached target LM-head shard with shape
                ``[V_local, h]``. It produces logits but is not trained here.
            mask_embedding: Detached target mask-token embedding with shape
                ``[h]``. Every mask slot starts from this same vector.
            packed_seq_params: Global THD packing metadata, or ``None`` for
                padded input.
            block_seq_idx: For packed input, the subsequence index of each
                local block, with shape ``[NB]``.

        Returns:
            Vocab-parallel logits. Padded shape is ``[B, N, W, V_local]``;
            packed shape is ``[NB, W, V_local]``. Slot 0 is the anchor slot.
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
        # ``F.linear`` bypasses the input-gradient reduction normally provided
        # by ``ColumnParallelLinear``. This wrapper is an identity in forward
        # and sums gradients across TP ranks in backward, keeping the
        # replicated ``fc`` and ``hidden_norm`` parameters in sync.
        trunk_hidden = copy_to_tensor_model_parallel_region(trunk_hidden)
        # Build one full RoPE table and index it with explicit global positions;
        # this avoids RotaryEmbedding slicing it a second time for CP.
        rotary_table = self.rotary_pos_emb(max_len + block_width, packed_seq=True)
        trunk_freqs = rotary_table[pos_in_seq]

        # An anchor at p can see trunk positions [0, p). Its own token is
        # supplied separately as block slot 0.
        vis_len = anchors_flat

        for layer, core in zip(self.decoder.layers, self._block_attn_modules):
            key, value = self._project_trunk_kv(layer.self_attention, trunk_hidden)
            key = apply_rotary_pos_emb(key, trunk_freqs, config=self.config)
            # [T, 1, Hkv, D] -> [T, Hkv, D]
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

        # The manual projection through the sharded LM head has the same TP
        # gradient issue as the trunk projection above. Sum its input gradient.
        decoder_hidden = copy_to_tensor_model_parallel_region(decoder_hidden)
        logits = F.linear(decoder_hidden, lm_head_weight)
        return logits.reshape(out_shape)

    @staticmethod
    def _anchor_embed_index(
        block_seq: Tensor,
        anchors_flat: Tensor,
        cu_local: Tensor,
        cp_group: Optional[dist.ProcessGroup],
    ) -> Tensor:
        """Row of each block's anchor embedding in the flat local embeds."""
        if cp_group is None or cp_group.size() == 1:
            return cu_local[block_seq] + anchors_flat
        from nemo_rl.algorithms.loss.utils import packed_zigzag_local_index

        return packed_zigzag_local_index(
            block_seq, anchors_flat, cu_local, cp_group.rank(), cp_group.size()
        )
