#!/usr/bin/env python3
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
"""GPU self-check for ``nemo_rl.models.generation.vllm.fp8_kv_attention`` (one SM100 GPU).

    uv run --extra vllm python tools/check_fp8_kv_attention.py [--timing]

For each supported head shape (32/4 and 16/1) on seeded random ragged paged inputs:

1. Cache writes through the backend give bit-identical FP8 codes and FP32
   scales to vLLM's native per-token-head writer, for LBNHC and LBHNC physical
   layouts (page bytes must also be identical across layouts), and match a
   PyTorch ``max|x|/448`` reference; copying a whole block moves its scales.
2. Prefill/mixed batches (Triton, num_stages=1) match native TRITON_ATTN on the
   native layout and an FP64 reference.
3. Decode (Gluon, adaptive splits, padded requests) matches native and the FP64
   reference; CUDA-graph replay is bitwise equal to eager.

Exits non-zero on the first failed gate; prints one JSON line per case.
"""

from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from nemo_rl.models.generation.vllm.fp8_kv_attention.backend import (
    Fp8KvAttentionImpl,
    page_views,
)
from nemo_rl.models.generation.vllm.fp8_kv_attention.decode import (
    HEAD_SIZE,
    PAGE_SIZE,
    num_splits,
)

CONTENT = 2 * (HEAD_SIZE + 4)
SEGMENTS = 16
# (batch, max context, query tokens per sequence, padded requests); query 1 => decode.
DECODE_CASES = [
    (1, 2048, 1, 0),
    (5, 4097, 1, 2),
    (16, 20479, 1, 0),
    (128, 8191, 1, 3),
    (512, 2048, 1, 0),
]
PREFILL_CASES = [
    (2, 2047, 64, 0),
    (2, 8191, 128, 0),
    (6, 3000, -96, 1),
]  # negative: mixed 1/96 tokens


def alloc_cache(blocks: int, kv_heads: int, layout: str) -> torch.Tensor:
    """Logical [B, H, N, C] byte cache with vLLM 0.29's physical LBNHC or LBHNC order."""
    if layout == "LBNHC":
        return torch.zeros(
            blocks, PAGE_SIZE, kv_heads, CONTENT, dtype=torch.uint8, device="cuda"
        ).transpose(1, 2)
    return torch.zeros(
        blocks, kv_heads, PAGE_SIZE, CONTENT, dtype=torch.uint8, device="cuda"
    )


def make_impl(cls: type, heads: int, kv_heads: int) -> TritonAttentionImpl:
    return cls(
        num_heads=heads,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_per_token_head",
    )


def make_case(
    batch: int, max_len: int, q_len: int, pad: int, heads: int, kv_heads: int
) -> SimpleNamespace:
    """Random ragged KV history, queries for each sequence's last tokens, and metadata."""
    torch.manual_seed(17)
    q_lens = [1 if (q_len < 0 and i % 2) else abs(q_len) for i in range(batch)]
    lengths = torch.linspace(max(max(q_lens), max_len // 4), max_len, batch).long()
    lengths[-1] = max_len
    pages = math.ceil(max_len / PAGE_SIZE)
    blocks = batch * pages + 1  # last block is a spare page-copy target
    table = (
        torch.randperm(blocks - 1, device="cuda")[: batch * pages]
        .view(batch, pages)
        .int()
    )
    positions = torch.cat([torch.arange(int(n), device="cuda") for n in lengths])
    owner = torch.repeat_interleave(torch.arange(batch, device="cuda"), lengths.cuda())
    slots = (
        table[owner, positions // PAGE_SIZE].long() * PAGE_SIZE + positions % PAGE_SIZE
    )
    slots = torch.cat(
        [slots, torch.full((7,), -1, device="cuda")]
    )  # padded tokens are skipped

    def kv():
        x = torch.randn(len(slots), kv_heads, HEAD_SIZE, device="cuda")
        x *= torch.exp2(
            torch.randint(-3, 4, (len(slots), kv_heads, 1), device="cuda").float()
        )
        x[..., 0] *= 4  # channel outlier
        return x.to(torch.bfloat16)

    cu = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()] + [sum(q_lens)] * pad,
        device="cuda",
    )
    rows = sum(q_lens) + pad  # padded requests contribute no query rows
    meta = SimpleNamespace(
        num_actual_tokens=sum(q_lens),
        max_query_len=max(q_lens),
        query_start_loc=cu.int(),
        max_seq_len=max_len,
        seq_lens=torch.cat(
            [lengths.cuda(), torch.zeros(pad, dtype=torch.long, device="cuda")]
        ).int(),
        block_table=torch.cat(
            [table, torch.zeros(pad, pages, dtype=torch.int32, device="cuda")]
        ),
        use_cascade=False,
        causal=True,
        mm_prefix_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        seq_threshold_3D=128 // kv_heads,
        num_par_softmax_segments=SEGMENTS,
        softmax_segm_output=torch.empty(
            rows, heads, SEGMENTS, HEAD_SIZE, device="cuda"
        ),
        softmax_segm_max=torch.empty(rows, heads, SEGMENTS, device="cuda"),
        softmax_segm_expsum=torch.empty(rows, heads, SEGMENTS, device="cuda"),
    )
    query = (0.125 * torch.randn(rows, heads, HEAD_SIZE, device="cuda")).to(
        torch.bfloat16
    )
    return SimpleNamespace(
        key=kv(),
        value=kv(),
        slots=slots,
        query=query,
        meta=meta,
        blocks=blocks,
        batch=batch,
    )


def errors(result: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = (result.float() - reference.float()).abs()
    return dict(
        max_abs=diff.max().item(),
        relative_l2=(
            diff.square().sum() / reference.float().square().sum().clamp_min(1e-30)
        )
        .sqrt()
        .item(),
        unequal_fraction=(result != reference).float().mean().item(),
        finite=bool(torch.isfinite(result).all()),
    )


def reference(
    case: SimpleNamespace, k, v, k_scale, v_scale, heads: int, kv_heads: int
) -> torch.Tensor:
    """FP64 causal attention over the dequantized paged cache (page-local views)."""
    meta, out = case.meta, torch.zeros_like(case.query, dtype=torch.float32)
    for b in range(case.batch):
        length, start, stop = (
            int(meta.seq_lens[b]),
            int(meta.query_start_loc[b]),
            int(meta.query_start_loc[b + 1]),
        )
        pages = meta.block_table[b, : math.ceil(length / PAGE_SIZE)].long()
        kk = (k[pages].float().double() * k_scale[pages, ..., None].double()).reshape(
            -1, kv_heads, HEAD_SIZE
        )[:length]
        vv = (v[pages].float().double() * v_scale[pages, ..., None].double()).reshape(
            -1, kv_heads, HEAD_SIZE
        )[:length]
        q = (
            case.query[start:stop]
            .double()
            .view(stop - start, kv_heads, heads // kv_heads, HEAD_SIZE)
        )
        scores = torch.einsum("qhgd,nhd->qhgn", q, kk) * HEAD_SIZE**-0.5
        q_pos = length - (stop - start) + torch.arange(stop - start, device="cuda")
        future = torch.arange(length, device="cuda")[None, :] > q_pos[:, None]
        scores.masked_fill_(future[:, None, None, :], float("-inf"))
        out[start:stop] = (
            torch.einsum("qhgn,nhd->qhgd", scores.softmax(-1), vv)
            .reshape(-1, heads, HEAD_SIZE)
            .float()
        )
    return out


def gate(name: str, ok: bool, row: dict) -> None:
    if not ok:
        raise SystemExit(f"FAILED {name}: {json.dumps(row)}")


def graph_us(fn, repeats: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(repeats):
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1000)
    return sorted(times)[len(times) // 2]


def check_case(spec: tuple, heads: int, kv_heads: int, timing: bool) -> dict:
    case = make_case(*spec, heads=heads, kv_heads=kv_heads)
    native = make_impl(TritonAttentionImpl, heads, kv_heads)
    # One impl per physical layout: an impl binds its scale views to the first
    # cache it sees and zero-initialises them (scale 1.0), as vLLM does.
    impls = {
        layout: make_impl(Fp8KvAttentionImpl, heads, kv_heads)
        for layout in ("LBNHC", "LBHNC")
    }
    impl = impls["LBNHC"]
    row = dict(
        heads=f"{heads}/{kv_heads}",
        batch=spec[0],
        max_len=spec[1],
        q_len=spec[2],
        pad=spec[3],
    )

    # 1. Cache writes: page-local (two physical layouts) vs native, all from vLLM's writer.
    native_cache = alloc_cache(case.blocks, kv_heads, "LBNHC")
    native.do_kv_cache_update(None, case.key, case.value, native_cache, case.slots)
    # Native views keep the 4 scale bytes of each 132-byte head row; drop them.
    nk, nv = (x[..., :HEAD_SIZE] for x in native._pth_key_value_caches(native_cache))
    ns, nt = native._k_scale_cache, native._v_scale_cache
    page_caches = {}
    for layout in ("LBNHC", "LBHNC"):
        cache = alloc_cache(case.blocks, kv_heads, layout)
        impls[layout].do_kv_cache_update(None, case.key, case.value, cache, case.slots)
        page_caches[layout] = cache
        pk, pv, ps, pt = page_views(cache)
        exact = (
            torch.equal(pk.view(torch.uint8), nk.view(torch.uint8))
            and torch.equal(pv.view(torch.uint8), nv.view(torch.uint8))
            and torch.equal(ps, ns)
            and torch.equal(pt, nt)
        )
        gate(f"cache write {layout}", exact, row)
    same_bytes = torch.equal(
        page_caches["LBNHC"].transpose(1, 2).contiguous().view(case.blocks, -1),
        page_caches["LBHNC"].reshape(case.blocks, -1),
    )
    gate("layout-independent page bytes", same_bytes, row)

    # PyTorch reference for the written slots: scale = max(max|x|/448, 1e-6), code = x * (1/scale).
    valid = case.slots >= 0
    block, slot = case.slots[valid] // PAGE_SIZE, case.slots[valid] % PAGE_SIZE
    pk, pv, ps, pt = page_views(page_caches["LBNHC"])
    for x, codes, scales in ((case.key, pk, ps), (case.value, pv, pt)):
        x = x[valid].float()
        ref_scale = (x.abs().amax(-1) / 448.0).clamp_min(1e-6)
        ref_codes = (
            (x * (1.0 / ref_scale)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        scale_err = ((scales[block, slot] - ref_scale).abs() / ref_scale).max().item()
        code_mismatch = (
            (codes[block, slot].view(torch.uint8) != ref_codes.view(torch.uint8))
            .float()
            .mean()
            .item()
        )
        row.update(scale_rel_err=max(row.get("scale_rel_err", 0.0), scale_err))
        row.update(code_mismatch=max(row.get("code_mismatch", 0.0), code_mismatch))
    gate(
        "writer vs torch reference",
        row["scale_rel_err"] < 1e-6 and row["code_mismatch"] < 1e-3,
        row,
    )

    # Whole-block copy (prefix caching / block copies) must carry the scales with the data.
    src, spare = int(case.meta.block_table[0, 0]), case.blocks - 1
    cache = page_caches["LBNHC"]
    cache[spare].copy_(cache[src])
    copied = torch.equal(
        pk[spare].view(torch.uint8), pk[src].view(torch.uint8)
    ) and torch.equal(ps[spare], ps[src])
    gate("page copy", copied, row)

    # 2./3. Attention outputs.
    rows = case.meta.num_actual_tokens
    native_out = torch.zeros_like(case.query)
    native.forward(
        SimpleNamespace(), case.query, None, None, native_cache, case.meta, native_out
    )
    outputs = {}
    for layout, cache in page_caches.items():
        outputs[layout] = torch.zeros_like(case.query)
        impls[layout].forward(
            None, case.query, None, None, cache, case.meta, outputs[layout]
        )
    out, cache = outputs["LBNHC"], page_caches["LBNHC"]
    gate(
        "layout-independent output",
        torch.equal(out[:rows], outputs["LBHNC"][:rows]),
        row,
    )
    ref = reference(case, pk, pv, ps, pt, heads, kv_heads)
    row["vs_reference"] = errors(out[:rows], ref[:rows])
    row["native_vs_reference"] = errors(native_out[:rows], ref[:rows])
    row["vs_native"] = errors(out[:rows], native_out[:rows])
    decode = case.meta.max_query_len == 1
    if decode:
        row["splits"] = num_splits(rows, kv_heads)
        gate(
            "decode vs reference",
            row["vs_reference"]["finite"] and row["vs_reference"]["relative_l2"] < 0.02,
            row,
        )
        gate("decode vs native", row["vs_native"]["relative_l2"] < 0.02, row)
        graph_out = torch.zeros_like(case.query)
        page_us = graph_us(
            lambda: impl.forward(
                None, case.query, None, None, cache, case.meta, graph_out
            )
        )
        gate("decode graph replay", torch.equal(graph_out[:rows], out[:rows]), row)
    else:
        gate(
            "prefill vs reference",
            row["vs_reference"]["finite"] and row["vs_reference"]["relative_l2"] < 0.02,
            row,
        )
        gate(
            "prefill vs native",
            row["vs_native"]["relative_l2"] < 3e-3
            and row["vs_native"]["max_abs"] < 5e-3,
            row,
        )
        page_us = (
            graph_us(
                lambda: impl.forward(
                    None, case.query, None, None, cache, case.meta, out
                )
            )
            if timing
            else 0
        )
    if timing:
        layer = SimpleNamespace()
        native_us = graph_us(
            lambda: native.forward(
                layer, case.query, None, None, native_cache, case.meta, native_out
            )
        )
        row.update(page_us=page_us, native_us=native_us, speedup=native_us / page_us)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--timing", action="store_true", help="also report CUDA-graph timings vs native"
    )
    parser.add_argument(
        "--shapes", default="32/4,16/1", help="comma list of Q/KV heads per rank"
    )
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (10, 0), (
        "requires an SM100 (GB200/B200) GPU"
    )
    for shape in args.shapes.split(","):
        heads, kv_heads = map(int, shape.split("/"))
        for spec in DECODE_CASES + PREFILL_CASES:
            print(
                json.dumps(check_case(spec, heads, kv_heads, args.timing)), flush=True
            )
            torch.cuda.empty_cache()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
