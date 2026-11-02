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
"""SM100 (Blackwell) paged decode for BF16 queries over a FP8 per-token-head KV cache.

One CTA handles one KV head group (all of its query heads) of one sequence for
one KV split. K/V tiles are dequantized to BF16 with their FP32 per-token-head
scales before ``tcgen05`` QK and PV MMAs; partial results of a split-KV launch
are merged by ``_reduce``. Supported per-rank shapes are 32 Q / 4 KV heads
(Qwen3-30B-A3B TP1) and 16 Q / 1 KV head (Nemotron 3 Super TP2), head size 128,
64-token pages. The split scratch buffer is shared by all layers, so calls must
run serially on one CUDA stream (vLLM's default execution).
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    get_tmem_reg_layout,
    mbarrier,
    tcgen05_mma,
)

PAGE_SIZE = 64
HEAD_SIZE = 128
SUPPORTED_HEADS = ((32, 4), (16, 1))  # (query heads, KV heads) per rank
MAX_ROWS = 512  # decode rows (sequences) covered by the shared split scratch


@gluon.jit
def _decode_kernel(
    Q, K, V, KS, VS, Table, Lengths, Starts, O, Partial, Maxima, Sums,
    strides: gl.constexpr, qstride: gl.constexpr, ostride: gl.constexpr,
    SCALE: gl.constexpr, QH: gl.constexpr, HEADS: gl.constexpr,
    BLOCK: gl.constexpr, SPLITS: gl.constexpr,
):  # fmt: skip
    kb: gl.constexpr = strides[0]
    kt: gl.constexpr = strides[1]
    kh: gl.constexpr = strides[2]
    kd: gl.constexpr = strides[3]
    vb: gl.constexpr = strides[4]
    vt: gl.constexpr = strides[5]
    vh: gl.constexpr = strides[6]
    vd: gl.constexpr = strides[7]
    sb: gl.constexpr = strides[8]
    st: gl.constexpr = strides[9]
    sh: gl.constexpr = strides[10]
    vsb: gl.constexpr = strides[11]
    vst: gl.constexpr = strides[12]
    vsh: gl.constexpr = strides[13]
    bt: gl.constexpr = strides[14]
    shared: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16, transposed=False, rank=2
    )
    # One BF16 K/V tile buffer: QK completes before it is reused for V.
    kvbuf = gl.allocate_shared_memory(gl.bfloat16, [1, BLOCK, 128], shared).index(0)

    seq = gl.program_id(0)
    head = gl.program_id(1)
    split = gl.program_id(2)
    qrow = gl.load(Starts + seq)
    active = gl.load(Starts + seq + 1) > qrow  # padded requests have no query
    length = gl.where(active, gl.load(Lengths + seq), 0)
    chunks = gl.cdiv(length, SPLITS * BLOCK)
    begin = split * chunks * BLOCK
    end = gl.minimum(begin + chunks * BLOCK, length)

    qlayout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [2, 2], [1, 0])
    hh = gl.arange(0, QH, layout=gl.SliceLayout(1, qlayout))
    dd = gl.arange(0, 128, layout=gl.SliceLayout(0, qlayout))
    q = gl.load(
        Q + qrow * qstride + (head * QH + hh[:, None]) * 128 + dd[None, :],
        active,
        other=0.0,
    )
    qsmem = gl.allocate_shared_memory(gl.bfloat16, [QH, 128], shared)
    psmem = gl.allocate_shared_memory(gl.bfloat16, [QH, BLOCK], shared)
    qsmem.store(q)
    fence_async_shared()

    slayout: gl.constexpr = TensorMemoryLayout([BLOCK, QH], col_stride=1)
    olayout: gl.constexpr = TensorMemoryLayout([128, QH], col_stride=1)
    score = allocate_tensor_memory(gl.float32, [BLOCK, QH], slayout)
    output = allocate_tensor_memory(gl.float32, [128, QH], olayout)
    sr: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 4], [0, 1])
    oo: gl.constexpr = get_tmem_reg_layout(gl.float32, (128, QH), olayout, 4)
    hr: gl.constexpr = gl.SliceLayout(0, sr)
    ho: gl.constexpr = gl.SliceLayout(0, oo)
    maximum = gl.full([QH], float("-inf"), gl.float32, hr)
    denominator = gl.full([QH], 0, gl.float32, hr)
    acc = gl.full([128, QH], 0, gl.float32, oo)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)

    tokens = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, sr))
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    rr = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))
    dd = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    for start in range(begin, end, BLOCK):
        token = start + rr
        page = gl.load(Table + seq * bt + token // 64, token < length, other=0).to(
            gl.int64
        )
        slot = token % 64
        ko = page[:, None] * kb + slot[:, None] * kt + head * kh + dd[None, :] * kd
        vo = page[:, None] * vb + slot[:, None] * vt + head * vh + dd[None, :] * vd
        sk = gl.load(KS + page * sb + slot * st + head * sh, token < length, other=1.0)[
            :, None
        ]
        sv = gl.load(
            VS + page * vsb + slot * vst + head * vsh, token < length, other=1.0
        )[:, None]
        kval = gl.load(K + ko, token[:, None] < length, other=0.0)
        vval = gl.load(V + vo, token[:, None] < length, other=0.0)
        # Early dequantization: FP8 * FP32 scale, rounded to BF16 for the MMA.
        kval = (kval.to(gl.float32) * sk).to(gl.bfloat16)
        vval = (vval.to(gl.float32) * sv).to(gl.bfloat16)

        kvbuf.store(kval)
        fence_async_shared()
        tcgen05_mma(
            kvbuf, qsmem.permute((1, 0)), score, use_acc=False, mbarriers=[mma_bar]
        )
        mbarrier.wait(mma_bar, 0)
        kvbuf.store(vval)
        s = gl.convert_layout(
            score.load(get_tmem_reg_layout(gl.float32, (BLOCK, QH), slayout, 4)), sr
        )
        s = s * (SCALE * 1.4426950408889634)
        s = gl.where((start + tokens[:, None]) < length, s, float("-inf"))
        new_max = gl.maximum(maximum, gl.max(s, 0))
        alpha = gl.exp2(maximum - new_max)
        p = gl.exp2(s - new_max[None, :])
        denominator = denominator * alpha + gl.sum(p, 0)
        psmem.store(p.trans().to(gl.bfloat16))
        fence_async_shared()
        tcgen05_mma(
            kvbuf.permute((1, 0)),
            psmem.permute((1, 0)),
            output,
            use_acc=False,
            mbarriers=[mma_bar],
        )
        mbarrier.wait(mma_bar, 1)
        acc = acc * gl.convert_layout(alpha, ho)[None, :] + output.load(oo)
        maximum = new_max
    mbarrier.invalidate(mma_bar)

    out_d = gl.arange(0, 128, layout=gl.SliceLayout(1, oo))
    out_h = gl.arange(0, QH, layout=ho) + head * QH
    if SPLITS == 1:
        denom = gl.convert_layout(denominator, ho)
        value = acc / gl.where(denom > 0, denom, 1)[None, :]
        gl.store(
            O + qrow * ostride + out_h[None, :] * 128 + out_d[:, None], value, active
        )
    else:
        offsets = ((qrow * HEADS + out_h[None, :]) * SPLITS + split) * 128 + out_d[
            :, None
        ]
        gl.store(Partial + offsets, acc, active)
        mh = gl.arange(0, QH, layout=hr) + head * QH
        gl.store(Maxima + (qrow * HEADS + mh) * SPLITS + split, maximum, active)
        gl.store(Sums + (qrow * HEADS + mh) * SPLITS + split, denominator, active)


@triton.jit
def _reduce(
    Partial, Maxima, Sums, O, Starts,
    NSEQ: tl.constexpr, SPLITS: tl.constexpr, OSTRIDE: tl.constexpr, QH: tl.constexpr, HEADS: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0)
    head = tl.program_id(1) * QH + tl.arange(0, QH)
    sp = tl.arange(0, SPLITS)
    d = tl.arange(0, 128)
    ix = (row * HEADS + head[:, None]) * SPLITS + sp[None, :]
    active = row < tl.load(Starts + NSEQ)
    m = tl.load(Maxima + ix, active, other=-float("inf"))
    l = tl.load(Sums + ix, active, other=0.0)  # noqa: E741
    top = tl.max(m, 1)
    w = tl.exp2(m - top[:, None])
    w = tl.where(l > 0, w, 0.0)
    denom = tl.sum(l * w, 1)
    val = tl.load(Partial + ix[:, :, None] * 128 + d[None, None, :], active, other=0.0)
    result = tl.sum(val * w[:, :, None], 1) / tl.where(denom > 0, denom, 1)[:, None]
    tl.store(O + row * OSTRIDE + head[:, None] * 128 + d[None, :], result, active)


def num_splits(rows: int, kv_heads: int) -> int:
    """Adaptive KV split count measured on GB200 for 1-512 rows and 2K-20K context."""
    if kv_heads == 4:
        return 4 if rows <= 8 or rows > 256 else 16 if rows <= 64 else 8
    # One KV head gives a quarter of the CTAs per sequence, so split deeper.
    return 16 if rows <= 128 else 8


G_DECODE_SCRATCH: dict[
    tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def reserve_scratch(
    device: torch.device, num_heads: int, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate (once) the split-KV scratch shared by every layer on ``device``."""
    key = (torch.device(device), num_heads)
    if key not in G_DECODE_SCRATCH:
        # Largest rows * splits product of num_splits() for rows <= MAX_ROWS.
        slots = MAX_ROWS * num_heads * (4 if num_kv_heads == 4 else 16)
        G_DECODE_SCRATCH[key] = (
            torch.empty(slots * HEAD_SIZE, device=device, dtype=torch.float32),
            torch.empty(slots, device=device, dtype=torch.float32),
            torch.empty(slots, device=device, dtype=torch.float32),
        )
    return G_DECODE_SCRATCH[key]


def decode(
    q: torch.Tensor,
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Paged decode attention (one query token per sequence) into ``out``.

    Args:
        q, out: ``[rows, num_heads, 128]`` BF16.
        k, v: FP8 E4M3 page views ``[num_blocks, 64, num_kv_heads, 128]``.
        k_scale, v_scale: FP32 per-token-head scales ``[num_blocks, 64, num_kv_heads]``.
        cu_seqlens_q: ``[num_seqs + 1]`` query start offsets (padded sequences repeat the end).
        seq_lens: ``[num_seqs]`` KV lengths; block_table: ``[num_seqs, max_pages]``.
    """
    rows, heads = q.shape[0], q.shape[1]
    kv_heads = k.shape[2]
    assert (heads, kv_heads) in SUPPORTED_HEADS, (heads, kv_heads)
    assert (
        q.dtype == torch.bfloat16
        and q.shape[2] == HEAD_SIZE
        and q.stride(2) == 1
        and q.stride(1) == HEAD_SIZE
    )
    assert out.stride(2) == 1 and out.stride(1) == HEAD_SIZE
    assert (
        k.shape[1:] == (PAGE_SIZE, kv_heads, HEAD_SIZE)
        and k.dtype == torch.float8_e4m3fn
    )
    assert k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32
    splits = num_splits(rows, kv_heads)
    if splits > 1:
        assert rows <= MAX_ROWS, (
            f"SM100 FP8 KV decode supports at most {MAX_ROWS} rows, got {rows}"
        )
        partial, maxima, sums = reserve_scratch(q.device, heads, kv_heads)
    else:
        partial = maxima = sums = None
    num_seqs = seq_lens.shape[0]
    strides = (
        *k.stride(),
        *v.stride(),
        *k_scale.stride(),
        *v_scale.stride(),
        block_table.stride(0),
    )
    qh = heads // kv_heads
    _decode_kernel[(num_seqs, kv_heads, splits)](
        q, k, v, k_scale, v_scale, block_table, seq_lens, cu_seqlens_q, out, partial, maxima, sums,
        strides, q.stride(0), out.stride(0), softmax_scale, qh, heads, PAGE_SIZE, splits,
        num_warps=4, maxnreg=128,
    )  # fmt: skip
    if splits > 1:
        _reduce[(rows, kv_heads)](
            partial,
            maxima,
            sums,
            out,
            cu_seqlens_q,
            num_seqs,
            splits,
            out.stride(0),
            qh,
            heads,
            num_warps=4,
        )
