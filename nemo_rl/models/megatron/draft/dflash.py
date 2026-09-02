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
import torch.nn.functional as F
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)
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


SUPPORTED_BLOCK_DRAFT_METHODS = ("dflash", "dspark")


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
        -1,  # window_size_left
        -1,  # window_size_right
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
        -1,  # window_size_left
        -1,  # window_size_right
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


class _PartBGather:
    """Index bookkeeping for the part-B (varlen) gathered key/value buffer.

    Block ``b`` owns the buffer slice ``[cu_k[b], cu_k[b+1])`` laid out as its
    trunk remainder ``[part_a_len_b, vis_len_b)`` followed by its own ``W``
    keys. Recomputed in backward (cheap integer math) instead of saving the
    index tensors.
    """

    def __init__(
        self,
        block_row: Tensor,
        vis_len: Tensor,
        part_a_len: Tensor,
        block_width: int,
        trunk_seqlen: int,
    ):
        rem_len = vis_len - part_a_len
        kv_len = rem_len + block_width
        num_blocks = block_row.shape[0]
        device = block_row.device

        self.kv_len = kv_len
        self.cu_k = torch.zeros(num_blocks + 1, device=device, dtype=torch.int32)
        self.cu_k[1:] = torch.cumsum(kv_len, dim=0).to(torch.int32)
        self.total_k = int(self.cu_k[-1].item())
        self.max_seqlen_k = int(kv_len.max().item()) if num_blocks else 0

        starts = (self.cu_k[:-1]).to(torch.long)
        rem_offsets = _ragged_arange(rem_len)
        rem_block = torch.repeat_interleave(
            torch.arange(num_blocks, device=device), rem_len
        )
        # Destination rows in the gathered buffer / source rows in the
        # flattened [B * S] trunk for each remainder key.
        self.rem_dest = starts[rem_block] + rem_offsets
        self.rem_src = (
            block_row[rem_block] * trunk_seqlen + part_a_len[rem_block] + rem_offsets
        )
        own_offsets = torch.arange(block_width, device=device)
        self.own_dest = (
            (starts + rem_len).unsqueeze(1) + own_offsets.unsqueeze(0)
        ).reshape(-1)

    def gather(self, trunk_flat: Tensor, own: Tensor) -> Tensor:
        """Build the gathered buffer from ``trunk_flat`` [B*S, Hkv, D] and ``own`` [NB, W, Hkv, D]."""
        num_blocks, block_width = own.shape[0], own.shape[1]
        gathered = trunk_flat.new_empty((self.total_k, own.shape[2], own.shape[3]))
        gathered[self.rem_dest] = trunk_flat[self.rem_src]
        gathered[self.own_dest] = own.reshape(num_blocks * block_width, *own.shape[2:])
        return gathered

    def scatter_grads(
        self, d_gathered: Tensor, d_trunk_flat: Tensor, own_shape: tuple[int, ...]
    ) -> Tensor:
        """Accumulate remainder grads into ``d_trunk_flat`` and return the own-block grads."""
        d_trunk_flat.index_add_(0, self.rem_src, d_gathered[self.rem_dest])
        return d_gathered[self.own_dest].reshape(own_shape)


def _part_a_buckets(
    block_row: Tensor, part_a_len: Tensor
) -> list[tuple[int, int, Tensor]]:
    """Group blocks by ``(batch_row, part_a_len)``; skip empty prefixes."""
    buckets: list[tuple[int, int, Tensor]] = []
    pairs = torch.stack([block_row, part_a_len], dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    for pair_index in range(unique_pairs.shape[0]):
        row = int(unique_pairs[pair_index, 0].item())
        prefix_len = int(unique_pairs[pair_index, 1].item())
        if prefix_len == 0:
            continue
        buckets.append(
            (row, prefix_len, torch.nonzero(inverse == pair_index, as_tuple=True)[0])
        )
    return buckets


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

    1. A block's visible trunk keys form a prefix, split at
       ``part_a_len = (vis_len // chunk) * chunk``.
    2. Blocks in the same row that share ``part_a_len`` share one dense
       FlashAttention call for the full-chunk prefix (the ``A`` cells). One
       variable-length call handles the remainder plus each block's own keys
       (the ``x`` cells).
    3. These calls cover disjoint key sets. Their outputs and log-sum-exp
       values are merged to recover the exact softmax over all visible keys.

    Backward repeats the same kernel split. It uses the final output and
    log-sum-exp so each call computes its part of the joint-softmax gradient,
    the same mechanism as :class:`~nemo_rl.models.megatron.draft.eagle.TwoPartTTTAttention`;
    trunk-key gradients are scattered back with ``index_add``.

    Floating-point inputs use contiguous bf16/fp16 FlashAttention layouts.
    Tensor shapes passed to ``apply`` are:

    - ``q``: ``[NB, W, Hq, D]`` block queries after RoPE.
    - ``k_own`` and ``v_own``: ``[NB, W, Hkv, D]`` block-local K/V.
    - ``trunk_k`` and ``trunk_v``: ``[B, S, Hkv, D]`` per-layer projected taps.
    - ``block_row``: ``[NB]`` batch row of each block.
    - ``vis_len``: ``[NB]`` anchor position, which is also the visible trunk
      length.
    - ``chunk``: Size of the shared dense-attention prefix buckets.
    - ``softmax_scale``: Scale applied to attention scores.

    The result has shape ``[NB, W, Hq, D]`` and equals attention over
    ``trunk[row, :p]`` together with the block's own keys.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: Tensor,
        k_own: Tensor,
        v_own: Tensor,
        trunk_k: Tensor,
        trunk_v: Tensor,
        block_row: Tensor,
        vis_len: Tensor,
        chunk: int,
        softmax_scale: float,
    ) -> Tensor:
        num_blocks, block_width, num_q_heads, head_dim = q.shape
        trunk_seqlen = trunk_k.shape[1]
        device = q.device

        if int(vis_len.max().item()) > trunk_seqlen:
            raise ValueError(
                f"vis_len max {int(vis_len.max().item())} exceeds trunk length "
                f"{trunk_seqlen}."
            )

        part_a_len = torch.div(vis_len, chunk, rounding_mode="floor") * chunk

        # ---- Part B: trunk remainder + own block, one varlen call ----
        gather = _PartBGather(block_row, vis_len, part_a_len, block_width, trunk_seqlen)
        trunk_k_flat = trunk_k.reshape(-1, *trunk_k.shape[2:])
        trunk_v_flat = trunk_v.reshape(-1, *trunk_v.shape[2:])
        kb = gather.gather(trunk_k_flat, k_own)
        vb = gather.gather(trunk_v_flat, v_own)
        q_flat = q.reshape(num_blocks * block_width, num_q_heads, head_dim)
        cu_q = torch.arange(
            0,
            (num_blocks + 1) * block_width,
            block_width,
            device=device,
            dtype=torch.int32,
        )
        out_b_flat, lse_b_flat = _fa_varlen_forward(
            q_flat,
            kb,
            vb,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=gather.cu_k,
            max_seqlen_q=block_width,
            max_seqlen_k=gather.max_seqlen_k,
            softmax_scale=softmax_scale,
        )
        # [Hq, NB*W] -> [NB, Hq, W] (uniform q length per block).
        lse_b = (
            lse_b_flat.view(num_q_heads, num_blocks, block_width)
            .permute(1, 0, 2)
            .float()
        )
        out_b = out_b_flat.view(num_blocks, block_width, num_q_heads, head_dim)

        # ---- Part A: shared full-chunk prefixes, one dense call per bucket ----
        lse_a = torch.full(
            (num_blocks, num_q_heads, block_width),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        out_a = torch.zeros_like(out_b, dtype=torch.float32)
        buckets = _part_a_buckets(block_row, part_a_len)
        for row, prefix_len, idx in buckets:
            q_bucket = q[idx].reshape(1, -1, num_q_heads, head_dim)
            out_bucket, lse_bucket = _fa_dense_forward(
                q_bucket,
                trunk_k[row : row + 1, :prefix_len],
                trunk_v[row : row + 1, :prefix_len],
                softmax_scale,
            )
            lse_a[idx] = (
                lse_bucket.view(num_q_heads, idx.shape[0], block_width)
                .permute(1, 0, 2)
                .float()
            )
            out_a[idx] = out_bucket.view(
                idx.shape[0], block_width, num_q_heads, head_dim
            ).float()

        # ---- Exact LSE merge of the two disjoint key sets ----
        lse_joint = torch.logaddexp(lse_a, lse_b)  # [NB, Hq, W]
        # [NB, Hq, W] -> [NB, W, Hq, 1] weights.
        w_a = torch.exp(lse_a - lse_joint).permute(0, 2, 1).unsqueeze(-1)
        w_b = torch.exp(lse_b - lse_joint).permute(0, 2, 1).unsqueeze(-1)
        out = (w_a * out_a + w_b * out_b.float()).to(q.dtype)

        ctx.save_for_backward(
            q, k_own, v_own, trunk_k, trunk_v, block_row, vis_len, out, lse_joint
        )
        ctx.chunk = chunk
        ctx.softmax_scale = softmax_scale
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
            block_row,
            vis_len,
            out,
            lse_joint,
        ) = ctx.saved_tensors
        chunk = ctx.chunk
        softmax_scale = ctx.softmax_scale

        num_blocks, block_width, num_q_heads, head_dim = q.shape
        trunk_seqlen = trunk_k.shape[1]
        device = q.device
        dout = dout.contiguous()

        part_a_len = torch.div(vis_len, chunk, rounding_mode="floor") * chunk
        dq = torch.zeros_like(q, dtype=torch.float32)
        d_trunk_k = torch.zeros_like(trunk_k, dtype=torch.float32)
        d_trunk_v = torch.zeros_like(trunk_v, dtype=torch.float32)

        # ---- Part B backward (joint lse -> exact joint gradient on B keys) ----
        gather = _PartBGather(block_row, vis_len, part_a_len, block_width, trunk_seqlen)
        trunk_k_flat = trunk_k.reshape(-1, *trunk_k.shape[2:])
        trunk_v_flat = trunk_v.reshape(-1, *trunk_v.shape[2:])
        kb = gather.gather(trunk_k_flat, k_own)
        vb = gather.gather(trunk_v_flat, v_own)
        q_flat = q.reshape(num_blocks * block_width, num_q_heads, head_dim)
        cu_q = torch.arange(
            0,
            (num_blocks + 1) * block_width,
            block_width,
            device=device,
            dtype=torch.int32,
        )
        lse_joint_flat = (
            lse_joint.permute(1, 0, 2)
            .reshape(num_q_heads, num_blocks * block_width)
            .contiguous()
        )
        dq_b = torch.empty_like(q_flat)
        dkb = torch.empty_like(kb)
        dvb = torch.empty_like(vb)
        _fa_varlen_backward(
            dout=dout.reshape(num_blocks * block_width, num_q_heads, head_dim),
            q=q_flat,
            k=kb,
            v=vb,
            out=out.reshape(num_blocks * block_width, num_q_heads, head_dim),
            lse=lse_joint_flat,
            dq=dq_b,
            dk=dkb,
            dv=dvb,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=gather.cu_k,
            max_seqlen_q=block_width,
            max_seqlen_k=gather.max_seqlen_k,
            softmax_scale=softmax_scale,
        )
        dq += dq_b.view_as(q).float()
        dk_own = gather.scatter_grads(
            dkb.float(), d_trunk_k.view(-1, *trunk_k.shape[2:]), k_own.shape
        )
        dv_own = gather.scatter_grads(
            dvb.float(), d_trunk_v.view(-1, *trunk_v.shape[2:]), v_own.shape
        )

        # ---- Part A backward, one dense call per bucket ----
        for row, prefix_len, idx in _part_a_buckets(block_row, part_a_len):
            nb = idx.shape[0]
            q_bucket = q[idx].reshape(1, nb * block_width, num_q_heads, head_dim)
            out_bucket = out[idx].reshape(1, nb * block_width, num_q_heads, head_dim)
            dout_bucket = (
                dout[idx]
                .reshape(1, nb * block_width, num_q_heads, head_dim)
                .contiguous()
            )
            lse_bucket = (
                lse_joint[idx]
                .permute(1, 0, 2)
                .reshape(1, num_q_heads, nb * block_width)
                .contiguous()
            )
            k_bucket = trunk_k[row : row + 1, :prefix_len].contiguous()
            v_bucket = trunk_v[row : row + 1, :prefix_len].contiguous()
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
            dq[idx] += dq_a.view(nb, block_width, num_q_heads, head_dim).float()
            d_trunk_k[row, :prefix_len] += dk_a[0].float()
            d_trunk_v[row, :prefix_len] += dv_a[0].float()

        return (
            dq.to(q.dtype),
            dk_own.to(k_own.dtype),
            dv_own.to(v_own.dtype),
            d_trunk_k.to(trunk_k.dtype),
            d_trunk_v.to(trunk_v.dtype),
            None,  # block_row
            None,  # vis_len
            None,  # chunk
            None,  # softmax_scale
        )


def block_draft_attention(
    q: Tensor,
    k_own: Tensor,
    v_own: Tensor,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_row: Tensor,
    vis_len: Tensor,
    *,
    chunk: int = 1024,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """Functional wrapper for :class:`BlockDraftAttention` (see class docstring)."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}.")
    return BlockDraftAttention.apply(
        q,
        k_own,
        v_own,
        trunk_k.contiguous(),
        trunk_v.contiguous(),
        block_row,
        vis_len,
        chunk,
        softmax_scale,
    )


class BlockDraftCoreAttention(torch.nn.Module):
    """Drop-in ``core_attention`` for the block-draft decoder layers.

    Receives the post-RoPE q/k/v of the flattened block stream (sbhd with
    batch 1, seq = ``NB * W`` in block-major order) and attends against the
    per-layer trunk K/V staged by the model via :meth:`stage_trunk` before the
    decoder call. Statefulness contract mirrors ``TTTDraftCoreAttention``:
    the model stages fresh trunk tensors every forward and :meth:`reset`
    clears them (never leaks across microbatches).
    """

    def __init__(self, config: TransformerConfig, chunk: int):
        super().__init__()
        attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        if attention_dropout != 0.0:
            raise ValueError(
                "BlockDraftCoreAttention does not support attention dropout "
                f"(got attention_dropout={attention_dropout})."
            )
        if int(getattr(config, "context_parallel_size", 1) or 1) != 1:
            raise ValueError(
                "BlockDraftCoreAttention requires context_parallel_size == 1."
            )
        self.chunk = int(chunk)
        self.softmax_scale: Optional[float] = getattr(config, "softmax_scale", None)
        self._trunk_k: Optional[Tensor] = None
        self._trunk_v: Optional[Tensor] = None
        self._block_row: Optional[Tensor] = None
        self._vis_len: Optional[Tensor] = None
        self._block_width: int = 0

    def stage_trunk(
        self,
        trunk_k: Tensor,
        trunk_v: Tensor,
        block_row: Tensor,
        vis_len: Tensor,
        block_width: int,
    ) -> None:
        """Stage this layer's trunk K/V ``[B, S, Hkv, D]`` and block metadata."""
        self._trunk_k = trunk_k
        self._trunk_v = trunk_v
        self._block_row = block_row
        self._vis_len = vis_len
        self._block_width = int(block_width)

    def reset(self) -> None:
        """Drop staged trunk references (frees the autograd graph)."""
        self._trunk_k = None
        self._trunk_v = None
        self._block_row = None
        self._vis_len = None
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
        if packed_seq_params is not None:
            raise NotImplementedError(
                "Block draft training does not support sequence packing; "
                "disable policy.sequence_packing."
            )
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

        out = block_draft_attention(
            q,
            k_own,
            v_own,
            self._trunk_k,
            self._trunk_v,
            self._block_row,
            self._vis_len,
            chunk=self.chunk,
            softmax_scale=softmax_scale,
        )
        # [NB, W, Hq, D] -> [S, 1, Hq * D] (core_attention output contract).
        return out.reshape(seqlen, 1, -1)


class DFlashDraftModel(MegatronModule):
    """Training model shared by the DFlash and DSpark block drafters.

    A DFlash block has ``W = gamma + 1`` slots. Slot 0 contains the anchor and
    supplies context, but its logits are not trained. The remaining ``gamma``
    slots predict new tokens. DSpark subclasses this model and changes
    ``method`` and ``block_width`` for its serving and loss behavior.
    """

    method = "dflash"

    def __init__(
        self,
        config: TransformerConfig,
        *,
        gamma: int,
        mask_token_id: int,
        num_aux_hidden_states: int,
        target_hidden_size: Optional[int] = None,
        trunk_chunk: int = 1024,
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
        for layer in self.decoder.layers:
            core = BlockDraftCoreAttention(config, self.trunk_chunk)
            layer.self_attention.core_attention = core
            self._block_attn_modules.append(core)

        # NO lm_head (official contract; z-lab ckpts ship only
        # layers/fc/hidden_norm/norm): logits project through the target's
        # live head, passed detached into forward(). A head-less ckpt makes
        # vLLM share the target's lm_head with the drafter unconditionally.

    def _project_trunk_kv(
        self, self_attention: torch.nn.Module, trunk_hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Project trunk hidden states through one layer's K/V rows.

        Slices the K/V rows out of the layer's fused, TP-local
        ``linear_qkv.weight`` (Megatron interleaves per query group as
        ``[q.. k v | q.. k v | ...]``) and applies them to the shared
        ``hidden_norm``-ed trunk — bypassing the fused input layernorm, which
        inference does *not* apply on the context path.
        """
        num_groups = self_attention.num_query_groups_per_partition
        heads_per_group = self_attention.num_attention_heads_per_partition // num_groups
        head_dim = self_attention.hidden_size_per_attention_head
        weight = self_attention.linear_qkv.weight
        grouped = weight.view(num_groups, (heads_per_group + 2) * head_dim, -1)
        k_weight = grouped[
            :, heads_per_group * head_dim : (heads_per_group + 1) * head_dim
        ].reshape(num_groups * head_dim, -1)
        v_weight = grouped[:, (heads_per_group + 1) * head_dim :].reshape(
            num_groups * head_dim, -1
        )

        seq_len, batch = trunk_hidden.shape[0], trunk_hidden.shape[1]
        key = F.linear(trunk_hidden, k_weight).view(
            seq_len, batch, num_groups, head_dim
        )
        value = F.linear(trunk_hidden, v_weight).view(
            seq_len, batch, num_groups, head_dim
        )
        k_layernorm = getattr(self_attention, "k_layernorm", None)
        if k_layernorm is not None:
            key = k_layernorm(key)
        return key, value

    def forward(
        self,
        *,
        taps: Tensor,
        input_embeds: Tensor,
        anchors: Tensor,
        anchor_valid: Tensor,
        lm_head_weight: Tensor,
        mask_embedding: Tensor,
    ) -> Tensor:
        """Run the block drafter for one batch.

        The method builds per-layer trunk K/V, creates each block from one
        anchor embedding plus mask embeddings, runs the decoder, and projects
        its outputs with the target model's LM head.

        Args:
            taps: Target auxiliary hidden states with shape
                ``[S, B, k * target_h]``.
            input_embeds: Unshifted target embeddings with shape ``[S, B, h]``.
            anchors: Anchor positions with shape ``[B, N]``.
            anchor_valid: Validity mask with shape ``[B, N]``. Invalid blocks
                keep tensor shapes static and are masked by the loss.
            lm_head_weight: Detached target LM-head shard with shape
                ``[V_local, h]``. It produces logits but is not trained here.
            mask_embedding: Detached target mask-token embedding with shape
                ``[h]``. Every mask slot starts from this same vector.

        Returns:
            Vocab-parallel logits with shape ``[B, N, W, V_local]``. Slot 0 is
            the anchor slot.
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
        # ``F.linear`` bypasses the input-gradient reduction normally provided
        # by ``ColumnParallelLinear``. This wrapper is an identity in forward
        # and sums gradients across TP ranks in backward, keeping the
        # replicated ``fc`` and ``hidden_norm`` parameters in sync.
        trunk_hidden = copy_to_tensor_model_parallel_region(trunk_hidden)
        rotary_table = self.rotary_pos_emb(seq_len + block_width)
        trunk_freqs = rotary_table[:seq_len]

        block_row = torch.arange(batch, device=device).repeat_interleave(num_anchors)
        anchors_flat = anchors.reshape(-1)
        # An anchor at p can see trunk positions [0, p). Its own token is
        # supplied separately as block slot 0.
        vis_len = anchors_flat

        for layer, core in zip(self.decoder.layers, self._block_attn_modules):
            key, value = self._project_trunk_kv(layer.self_attention, trunk_hidden)
            key = apply_rotary_pos_emb(key, trunk_freqs, config=self.config)
            # [S, B, Hkv, D] -> [B, S, Hkv, D]
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
        # [NB * W, 1, V_local] -> [B, N, W, V_local]
        return logits.reshape(batch, num_anchors, block_width, -1)
