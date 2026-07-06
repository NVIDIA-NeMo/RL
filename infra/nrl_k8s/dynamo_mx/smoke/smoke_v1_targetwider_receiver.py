"""Target-wider mixed-TP cluster smoke — proves the v1 sliced-pull primitive
on real NIXL RDMA bytes.

Shape: source_tp=1 → target_tp=2 (each "target rank" pulls only HALF of
each tp-sharded source tensor). One publisher publishes full-size global
tensors; this receiver acts as BOTH rank 0 and rank 1 of a TP=2 layout
sequentially, pulling the relevant slice for each rank via
``MxRefitReceiver.pull_to`` and asserting byte-identity vs the global
ground truth on the PVC.

This is the smallest viable validation that:
  * SlicedTransferRequest math is right (source_offset_bytes + slice_bytes)
  * pull_to wrapper computes byte offsets correctly from element ranges
  * NIXL transfer with offset+size lands data in the correct dest memory
  * The combined transfer pattern works (N slices in one wire call)

Reuses the existing TP=1 smoke publisher (smoke_megatron_publisher.py)
running on the trainer pod. Receiver runs in the DGD worker pod.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Any

import torch

from modelexpress import MxV2RefitReceiver
from modelexpress.megatron_helpers import (
    MegatronTransformerConfig, split_qkv_weights, split_gated_mlp_tp,
    split_gated_mlp,
)


# Same MODEL_NAME and GT_PATH as smoke_megatron_publisher.py.
MODEL_NAME = "smoke/megatron-mx-toy"
GT_PATH = "/mnt/rl-workspace/kavink/smoke-megatron-mx-groundtruth.pt"

# Target layout: TP=2.
TARGET_TP = 2


def _validate_rank(
    rcv,
    chosen_ref,
    tp_rank: int,
    cfg: MegatronTransformerConfig,
    gt: dict,
) -> tuple[int, int, int]:
    """Pull rank-tp_rank's slice of each role via pull_to + verify
    byte-identity. Returns (n_ok, n_drift, total_bytes_pulled)."""
    hidden = cfg.hidden_size
    intermediate = 1024  # matches the publisher
    head_dim = cfg.kv_channels
    nh = cfg.num_attention_heads
    nkv = cfg.num_query_groups

    # Per-rank slice sizes
    nh_per = nh // TARGET_TP
    nkv_per = nkv // TARGET_TP
    inter_per = intermediate // TARGET_TP
    h_per = hidden // TARGET_TP

    # ------ Build pull requests ------
    # Need contiguous axis-0 narrows for the v1 path:
    #   * QKV column (axis 0): take rank's q heads + k heads + v heads
    #     interleaved by head. Global QKV layout: [q0,k0,v0, q1,k1,v1, ...]
    #     by query group. Rank 0 wants first nkv/TP query groups; rank 1
    #     wants the second nkv/TP. Per group, slice contains
    #     (nh/nkv)*head_dim + head_dim + head_dim = (nh/nkv + 2)*head_dim
    #     rows.
    #   * gated_mlp (axis 0): per-rank tensor is [gate_local; up_local].
    #     For target-wider, each rank pulls SUB-range of source's
    #     [gate_global; up_global]. Rank 0 wants rows
    #     [0, inter_per) + [intermediate, intermediate+inter_per), which is
    #     NOT contiguous on the source side — gate_global's first half is
    #     followed by gate_global's second half, then up_global. Means
    #     two separate slice requests per rank (one for gate, one for up)
    #     OR we route gated_mlp to v0 fallback.
    #   * dense column (axis 0): rank pulls rows [tp_rank*inter_per : (tp_rank+1)*inter_per).
    #
    # Row-parallel: dest narrow is axis-1 → non-contiguous → v0 fallback.
    # Replicated: full tensor, single source.

    # Pre-allocate dest tensors (per-rank shapes) on GPU.
    device = torch.device("cuda:0")
    dt = torch.bfloat16

    # QKV: per-rank shape = ((nh_per + 2 * nkv_per) * head_dim, hidden)
    qkv_per_rank_rows = (nh_per + 2 * nkv_per) * head_dim
    qkv_dest = torch.empty((qkv_per_rank_rows, hidden), dtype=dt, device=device)
    # Gated MLP: per-rank shape = (2 * inter_per, hidden). Use TWO slice
    # requests (one for gate half, one for up half), each contiguous.
    gated_dest = torch.empty((2 * inter_per, hidden), dtype=dt, device=device)
    # Dense column: per-rank shape = (inter_per, hidden), contiguous narrow.
    dense_col_dest = torch.empty((inter_per, hidden), dtype=dt, device=device)
    # Row proj: per-rank shape = (hidden, h_per). Axis-1 narrow on full
    # row tensor → non-contiguous. Allocate full and v0 the slice via host copy.
    row_dest = torch.empty((hidden, h_per), dtype=dt, device=device)
    # Replicated norm: full tensor.
    norm_dest = torch.empty((hidden,), dtype=dt, device=device)

    buffers = {
        f"qkv_r{tp_rank}": qkv_dest,
        f"gated_r{tp_rank}": gated_dest,
        f"dense_col_r{tp_rank}": dense_col_dest,
        f"row_r{tp_rank}": row_dest,
        f"norm_r{tp_rank}": norm_dest,
    }

    # Register with NIXL.
    rcv._receiver._nixl.register_tensors(buffers)
    print(f"  [r{tp_rank}] registered {len(buffers)} dest buffers")

    # ------ Build pull_to requests ------
    # QKV: rank's slice is rows [tp_rank * nkv_per * (nh/nkv + 2) * head_dim,
    # (tp_rank+1) * ...]. Actually it's contiguous: each query group's
    # (q + k + v) heads are stored together, and ranks partition by group.
    qkv_rows_total = (nh + 2 * nkv) * head_dim
    qkv_rows_per_rank = qkv_per_rank_rows  # = (nh_per + 2*nkv_per)*head_dim
    qkv_lo = tp_rank * qkv_rows_per_rank
    qkv_hi = (tp_rank + 1) * qkv_rows_per_rank
    qkv_elem_lo = qkv_lo * hidden  # element offset
    qkv_elem_hi = qkv_hi * hidden

    # Gated MLP: source's global layout is [gate_global; up_global], each
    # `intermediate` rows. Rank 0 wants gate rows [0, inter_per) +
    # up rows [0, inter_per). Rank 1 wants gate rows [inter_per, 2*inter_per)
    # + up rows [inter_per, 2*inter_per). Need TWO requests per rank: one
    # for the gate slice, one for the up slice. The dest buffer is
    # gated_dest = [gate_rank; up_rank] (2 * inter_per rows).
    # Split gated_dest into two contiguous halves.
    gate_dest_view = gated_dest.narrow(0, 0, inter_per)
    up_dest_view = gated_dest.narrow(0, inter_per, inter_per)
    gate_src_elem_lo = tp_rank * inter_per * hidden
    gate_src_elem_hi = (tp_rank + 1) * inter_per * hidden
    up_src_elem_lo = (intermediate + tp_rank * inter_per) * hidden
    up_src_elem_hi = (intermediate + (tp_rank + 1) * inter_per) * hidden

    # Dense column: source rows are contiguous [tp_rank*inter_per : (tp_rank+1)*inter_per).
    dense_col_elem_lo = tp_rank * inter_per * hidden
    dense_col_elem_hi = (tp_rank + 1) * inter_per * hidden

    # Replicated norm: full tensor.
    norm_elements = hidden

    # Build requests (use the v1 pull_to API). The publisher publishes
    # global names; the receiver pulls the per-rank slices into per-rank
    # dest buffers using a name-mapping (publisher's name → receiver's local name).
    publisher_name_for_local = {
        f"qkv_r{tp_rank}": "decoder.layers.0.self_attention.linear_qkv.weight",
        f"gated_r{tp_rank}": "decoder.layers.0.mlp.linear_fc1.weight",
        f"dense_col_r{tp_rank}": "decoder.layers.0.mlp.linear_fc2.weight",
        f"row_r{tp_rank}": "decoder.layers.0.self_attention.linear_proj.weight",
        f"norm_r{tp_rank}": "decoder.layers.0.input_layernorm.weight",
    }

    # The v1 primitive matches by NAME, so we pass the publisher's name
    # but the dest_view is OUR buffer.
    # row_proj is non-contiguous in the receiver's view (axis-1 narrow)
    # — would normally fall back to v0 scratch+copy. For this smoke we
    # pull the FULL row tensor (contiguous on source side) and
    # host-side-slice afterwards.
    requests = [
        # QKV column-parallel: contiguous slice
        ("decoder.layers.0.self_attention.linear_qkv.weight",
         (qkv_elem_lo, qkv_elem_hi), qkv_dest),
        # Gated MLP — gate half: contiguous slice of [gate_global; up_global]
        ("decoder.layers.0.mlp.linear_fc1.weight",
         (gate_src_elem_lo, gate_src_elem_hi), gate_dest_view),
        # Gated MLP — up half: contiguous slice
        ("decoder.layers.0.mlp.linear_fc1.weight",
         (up_src_elem_lo, up_src_elem_hi), up_dest_view),
        # Dense column-parallel: contiguous slice
        ("decoder.layers.0.mlp.linear_fc2.weight",
         (dense_col_elem_lo, dense_col_elem_hi), dense_col_dest),
        # Replicated norm: full tensor
        ("decoder.layers.0.input_layernorm.weight", None, norm_dest),
        # Row-parallel: pull full (will host-slice after)
        ("decoder.layers.0.self_attention.linear_proj.weight",
         None, torch.empty(hidden, hidden, dtype=dt, device=device)),
    ]
    # NOTE: the row-parallel request's dest is a FULL-size scratch buffer
    # because the axis-1 narrow would be non-contiguous. After the pull
    # lands, we host-slice the column range we want into row_dest.
    row_full_dest = requests[-1][2]
    # Register the row_full_dest too.
    rcv._receiver._nixl.register_tensors({**buffers, "row_full_scratch": row_full_dest})

    # ------ Issue the sliced pull ------
    print(f"  [r{tp_rank}] issuing pull_to with {len(requests)} slice requests...")
    t0 = time.perf_counter()
    total_bytes, n_slices, elapsed = rcv._receiver.pull_to(
        chosen_ref, requests, timeout_seconds=120.0,
    )
    print(f"  [r{tp_rank}] pulled {n_slices} slices, {total_bytes / 1e6:.2f} MB, "
          f"{elapsed:.3f}s")

    # Post-process row-parallel: take the column range we want.
    h_lo, h_hi = tp_rank * h_per, (tp_rank + 1) * h_per
    row_dest.copy_(row_full_dest[:, h_lo:h_hi].contiguous())

    # ------ Verify against ground truth ------
    n_ok = 0
    n_drift = 0
    failures = []

    # QKV: split the per-rank packed tensor via the receiver's translator
    # (using a half-config) and compare against the corresponding half of
    # global q, k, v.
    half_cfg = MegatronTransformerConfig(
        num_attention_heads=nh_per, num_query_groups=nkv_per,
        kv_channels=head_dim, hidden_size=hidden,
    )
    q_local, k_local, v_local = split_qkv_weights(half_cfg, qkv_dest.cpu().float())
    # Ground-truth per-rank slices of q, k, v
    q_gt_rank = gt["q"][tp_rank * nh_per * head_dim : (tp_rank + 1) * nh_per * head_dim]
    k_gt_rank = gt["k"][tp_rank * nkv_per * head_dim : (tp_rank + 1) * nkv_per * head_dim]
    v_gt_rank = gt["v"][tp_rank * nkv_per * head_dim : (tp_rank + 1) * nkv_per * head_dim]
    for name, got, exp in [
        ("q_proj", q_local, q_gt_rank.float()),
        ("k_proj", k_local, k_gt_rank.float()),
        ("v_proj", v_local, v_gt_rank.float()),
    ]:
        if torch.equal(got, exp):
            n_ok += 1
            print(f"    OK  r{tp_rank} {name:8s} {tuple(got.shape)}")
        else:
            n_drift += 1
            diff = (got - exp).abs()
            failures.append((name, f"max={diff.max():.4e}", f"mean={diff.mean():.4e}"))

    # Gated MLP: per-rank dest is [gate_rank; up_rank]. With TARGET_TP=2
    # and the slice being the SOURCE's intermediate rank's slice, the
    # un-interleave is the matched-TP case (each rank's gated is
    # [gate_local; up_local], not interleaved across multiple sources).
    gate_local, up_local = split_gated_mlp(gated_dest.cpu().float())
    gate_gt_rank = gt["gate"][tp_rank * inter_per : (tp_rank + 1) * inter_per]
    up_gt_rank = gt["up"][tp_rank * inter_per : (tp_rank + 1) * inter_per]
    for name, got, exp in [
        ("gate", gate_local, gate_gt_rank.float()),
        ("up", up_local, up_gt_rank.float()),
    ]:
        if torch.equal(got, exp):
            n_ok += 1
            print(f"    OK  r{tp_rank} {name:8s} {tuple(got.shape)}")
        else:
            n_drift += 1
            diff = (got - exp).abs()
            failures.append((name, f"max={diff.max():.4e}", f"mean={diff.mean():.4e}"))

    # Dense column (axis 0)
    dc_gt = gt["dense_col"][tp_rank * inter_per : (tp_rank + 1) * inter_per].float()
    if torch.equal(dense_col_dest.cpu().float(), dc_gt):
        n_ok += 1
        print(f"    OK  r{tp_rank} {'dense_col':8s} {tuple(dc_gt.shape)}")
    else:
        n_drift += 1
        diff = (dense_col_dest.cpu().float() - dc_gt).abs()
        failures.append(("dense_col", f"max={diff.max():.4e}"))

    # Row (axis 1 — host-sliced from full pull)
    row_gt = gt["o_proj"][:, tp_rank * h_per : (tp_rank + 1) * h_per].float()
    if torch.equal(row_dest.cpu().float(), row_gt):
        n_ok += 1
        print(f"    OK  r{tp_rank} {'o_proj':8s} {tuple(row_gt.shape)}")
    else:
        n_drift += 1
        diff = (row_dest.cpu().float() - row_gt).abs()
        failures.append(("o_proj", f"max={diff.max():.4e}"))

    # Norm (replicated)
    norm_gt = gt["norm_w"].float()
    if torch.equal(norm_dest.cpu().float(), norm_gt):
        n_ok += 1
        print(f"    OK  r{tp_rank} {'norm':8s} {tuple(norm_gt.shape)}")
    else:
        n_drift += 1
        diff = (norm_dest.cpu().float() - norm_gt).abs()
        failures.append(("norm", f"max={diff.max():.4e}"))

    if failures:
        print(f"  [r{tp_rank}] FAILURES:")
        for f in failures:
            print(f"    {f}")
    return n_ok, n_drift, total_bytes


def main() -> int:
    print(f"[rcv] loading ground truth from {GT_PATH}")
    gt = torch.load(GT_PATH, weights_only=False)
    cfg = MegatronTransformerConfig(**gt["cfg"])
    print(f"  cfg={cfg}")

    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-v1-targetwider-rcv",
        device_id=0,
        mx_server_url=os.environ.get(
            "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"
        ),
        worker_rank=0,
    )
    rcv.initialize(model_tensors=None)

    print(f"\n[rcv] discovering source for {MODEL_NAME}...")
    deadline = time.time() + 120
    cands = []
    while time.time() < deadline:
        cands = rcv.discover_v2_sources(
            model_name=MODEL_NAME, min_version=1,
            same_rank_only=False, include_replicas=True,
        )
        megatron_cands = [c for c in cands if c.megatron_meta is not None]
        if megatron_cands:
            break
        print(f"  waiting (found {len(cands)} cands, {len(megatron_cands)} megatron)")
        time.sleep(2)
    megatron_cands = [c for c in cands if c.megatron_meta is not None]
    if not megatron_cands:
        print("[rcv] ERROR: no Megatron source found")
        return 3
    megatron_cands.sort(key=lambda c: -c.updated_at)
    chosen = megatron_cands[0]
    print(f"  chosen sid={chosen.ref.mx_source_id} "
          f"tp_rank={chosen.megatron_meta.tp_rank}/{chosen.megatron_meta.tp_size}")

    if chosen.megatron_meta.tp_size != 1:
        print(f"[rcv] WARN: expected source_tp=1 for target-wider smoke; "
              f"got source_tp={chosen.megatron_meta.tp_size}. Continuing.")

    # Run both target ranks sequentially.
    total_ok = 0
    total_drift = 0
    total_bytes = 0
    for tp_rank in range(TARGET_TP):
        print(f"\n[rcv] === Target TP=2 rank {tp_rank} ===")
        n_ok, n_drift, b = _validate_rank(rcv, chosen.ref, tp_rank, cfg, gt)
        total_ok += n_ok
        total_drift += n_drift
        total_bytes += b

    print(f"\n=== SUMMARY ===")
    print(f"  byte-identical: {total_ok}")
    print(f"  drift:          {total_drift}")
    print(f"  total bytes pulled: {total_bytes / 1e6:.2f} MB across {TARGET_TP} ranks")

    if total_drift == 0 and total_ok > 0:
        print(f"\n*** V1 SLICED-PULL TARGET-WIDER CLUSTER SMOKE VALIDATED: "
              f"{total_ok}/{total_ok + total_drift} byte-identical ***")
        return 0
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
