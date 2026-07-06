"""Phase E shape 2 receiver — Qwen3-MoE-30B-A3B (real MoE model).

Pairs with smoke_qwen3_moe_publisher.py. Pulls all 12 627 Megatron-shaped
tensors (190 dense + ~12 288 grouped per-expert) via NIXL RDMA from the
matched-TP source, runs the full A → B → C → D translator pipeline
(including the new grouped per-expert path), and asserts byte-identity
against the ~18 867 HF tensors that bridge.export_hf_weights produced.

Memory plan (single 189 GB GPU on the worker pod):
  pre-allocated Megatron-shape buffers     ~60 GB
  receive_from bulk pull into those         (same buffers)
  per-plan translation produces HF tensor   (small, free after compare)
  GT loaded via torch.load mmap=True        (no host RAM blow-up)
Peak GPU mem usage: ~60 GB. Peak host RAM: ~few GB.

Validation: for each yielded (hf_name, hf_tensor), compare to GT[hf_name]
and accumulate counters; free the GPU tensor after compare. Final report
covers byte-identical / drift / missing / extras across all 18 867 HF
tensors.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Any

import torch

from modelexpress import MxV2RefitReceiver
from modelexpress.nemo_rl_v2 import TargetTpLayout, MegatronTensorSpec
from modelexpress.megatron_helpers import MegatronTransformerConfig
from modelexpress.megatron_translator import (
    MegatronReceiverContext,
    ReceiveSpec,
    discover_megatron_context,
    run_refit_cycle,
)


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
GT_PATH = "/mnt/rl-workspace/kavink/phase-e-shape-2-qwen3-moe-groundtruth.pt"


SHARD_AXIS_BY_ROLE = {
    "column": 0, "qkv_column": 0, "gated_mlp_column": 0,
    "vocab_parallel": 0, "row": 1,
    "expert_column": 0, "expert_row": 0,
    "replicated": 0,
}


def main() -> int:
    print(f"[rcv] loading ground truth metadata + name_map from {GT_PATH}")
    t0 = time.perf_counter()
    gt_data = torch.load(GT_PATH, weights_only=False, mmap=True)
    gt_hf: dict[str, torch.Tensor] = gt_data["hf_weights"]
    name_map_gt = gt_data["name_map"]
    print(f"  loaded in {time.perf_counter() - t0:.1f}s; "
          f"{len(gt_hf)} HF tensors, {len(name_map_gt)} name_map entries (mmap)")

    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-phase-e-shape-2-rcv",
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
            same_rank_only=True, include_replicas=True,
        )
        if cands:
            break
        print("  waiting...")
        time.sleep(3)
    megatron_cands = [c for c in cands if c.megatron_meta is not None]
    if not megatron_cands:
        print("[rcv] ERROR: no Megatron candidate discovered")
        return 3
    # Pick the freshest source.
    megatron_cands.sort(key=lambda c: -c.updated_at)
    chosen = megatron_cands[0]
    print(f"  chosen sid={chosen.ref.mx_source_id} "
          f"tp_rank={chosen.megatron_meta.tp_rank}/{chosen.megatron_meta.tp_size}")

    # Sidecar
    sidecar_cfg, name_map = discover_megatron_context(megatron_cands)
    if sidecar_cfg is None:
        print("[rcv] ERROR: no sidecar config")
        return 4
    print(f"  sidecar cfg={sidecar_cfg}; name_map has {len(name_map)} entries")

    # Build receive_specs from the registry.
    print(f"\n[rcv] building ReceiveSpecs from source registry...")
    receive_specs: dict[str, ReceiveSpec] = {}
    role_counts: dict[str, int] = {}
    for td in chosen.registry.get("tensors", []) if chosen.registry else []:
        if not td.megatron_role:
            continue
        role = td.megatron_role
        role_counts[role] = role_counts.get(role, 0) + 1
        # Source TP=1, target TP=1 → matched. Strip "module." prefix
        # the way the production receiver wire-up does.
        lookup_name = td.name[len("module."):] if td.name.startswith("module.") else td.name
        hf_names = name_map.get(lookup_name, [td.name])
        shard_axis = SHARD_AXIS_BY_ROLE.get(role, int(td.shard_axis))
        receive_specs[td.name] = ReceiveSpec(
            megatron_name=td.name,
            hf_names=list(hf_names),
            role=role,
            target_shape=tuple(int(s) for s in td.global_shape),
            target_dtype=td.dtype or "bfloat16",
            shard_axis=shard_axis,
            pp_rank=chosen.megatron_meta.pp_rank,
            role_descriptor=dict(td.megatron_extras or {}),
        )
    print(f"  {len(receive_specs)} receive_specs; role distribution:")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"    {role:25s} {count:6d}")

    # Pre-allocate GPU buffers for matched-TP bulk pull.
    print(f"\n[rcv] pre-allocating {len(receive_specs)} buffers...")
    device = torch.device("cuda:0")
    dt_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    buffers: dict[str, torch.Tensor] = {}
    total_bytes = 0
    t0 = time.perf_counter()
    for m_name, spec in receive_specs.items():
        dt = dt_map.get(spec.target_dtype, torch.bfloat16)
        b = torch.empty(spec.target_shape, dtype=dt, device=device)
        buffers[m_name] = b
        total_bytes += b.numel() * b.element_size()
    print(f"  allocated in {time.perf_counter() - t0:.1f}s; "
          f"{total_bytes / 1e9:.2f} GB on GPU")

    print(f"\n[rcv] registering buffers with NIXL...")
    t0 = time.perf_counter()
    rcv._receiver._nixl.register_tensors(buffers)
    print(f"  registered in {time.perf_counter() - t0:.1f}s")

    # Bulk pull from the source.
    print(f"\n[rcv] bulk receive_from (RDMA pull)...")
    t0 = time.perf_counter()
    n_yielded = 0
    for name, t in rcv.receive_from(chosen, timeout_seconds=300.0):
        n_yielded += 1
    elapsed = time.perf_counter() - t0
    print(f"  done: {n_yielded} tensors, {total_bytes / 1e9:.2f} GB, "
          f"{elapsed:.2f}s, {total_bytes * 8 / elapsed / 1e9:.1f} Gbps")

    # Build context + run translator. Per-plan compare + free.
    layout = TargetTpLayout(tp_size=1, tp_rank=0)
    ctx = MegatronReceiverContext(
        target_tp_layout=layout,
        transformer_config=sidecar_cfg,
        hf_name_map=name_map,
        receive_specs=receive_specs,
    )

    print(f"\n[rcv] running run_refit_cycle + per-tensor byte-identity compare...")
    t0 = time.perf_counter()
    n_ok = 0
    n_drift = 0
    n_missing = 0
    n_extra = 0
    drift_examples: list[tuple] = []
    missing_examples: list[tuple] = []
    per_role_ok: dict[str, int] = {}

    for hf_name, hf_tensor in run_refit_cycle(
        rcv,
        candidates=megatron_cands,
        context=ctx,
        pull=lambda src, dest: None,  # matched-TP — buffers pre-filled
        device=device,
        pre_assembled_buffers=buffers,
    ):
        # Move to CPU for compare; don't keep on GPU.
        got = hf_tensor.detach().cpu()
        exp = gt_hf.get(hf_name)
        if exp is None:
            n_extra += 1
            continue
        # GT was saved in bfloat16; align dtypes.
        if got.dtype != exp.dtype:
            exp_aligned = exp.to(got.dtype)
        else:
            exp_aligned = exp
        if got.shape != exp_aligned.shape:
            n_drift += 1
            if len(drift_examples) < 3:
                drift_examples.append((hf_name, "shape", tuple(got.shape), tuple(exp_aligned.shape)))
            continue
        if torch.equal(got, exp_aligned):
            n_ok += 1
            # Rough role attribution for the summary table.
            if ".experts." in hf_name and ".gate_proj." in hf_name:
                per_role_ok["expert_gate"] = per_role_ok.get("expert_gate", 0) + 1
            elif ".experts." in hf_name and ".up_proj." in hf_name:
                per_role_ok["expert_up"] = per_role_ok.get("expert_up", 0) + 1
            elif ".experts." in hf_name and ".down_proj." in hf_name:
                per_role_ok["expert_down"] = per_role_ok.get("expert_down", 0) + 1
            elif ".q_proj.weight" in hf_name or ".k_proj.weight" in hf_name or ".v_proj.weight" in hf_name:
                per_role_ok["qkv"] = per_role_ok.get("qkv", 0) + 1
            elif ".o_proj.weight" in hf_name:
                per_role_ok["o_proj"] = per_role_ok.get("o_proj", 0) + 1
            elif "norm" in hf_name or "router" in hf_name or "gate.weight" in hf_name:
                per_role_ok["replicated"] = per_role_ok.get("replicated", 0) + 1
            elif "embed" in hf_name or "lm_head" in hf_name:
                per_role_ok["vocab"] = per_role_ok.get("vocab", 0) + 1
            else:
                per_role_ok["other"] = per_role_ok.get("other", 0) + 1
        else:
            n_drift += 1
            if len(drift_examples) < 3:
                diff = (got.float() - exp_aligned.float()).abs()
                drift_examples.append(
                    (hf_name, "values",
                     f"max_diff={diff.max().item():.4e}",
                     f"mean_diff={diff.mean().item():.4e}")
                )

    for hf_name in gt_hf.keys():
        # n_missing only counted if translator didn't yield it. Hard to
        # know without iterating but cheap to check now.
        pass  # n_missing tracked implicitly below.

    elapsed = time.perf_counter() - t0
    n_seen = n_ok + n_drift
    n_missing = max(0, len(gt_hf) - n_seen - n_extra)

    print(f"  cycle complete in {elapsed:.2f}s")
    print(f"\n  byte-identical:  {n_ok} / {len(gt_hf)}")
    print(f"  drift:           {n_drift}")
    print(f"  missing:         {n_missing}")
    print(f"  extras:          {n_extra}")
    if drift_examples:
        print(f"\n  drift examples:")
        for ex in drift_examples:
            print(f"    {ex}")
    if missing_examples:
        print(f"\n  missing examples:")
        for ex in missing_examples:
            print(f"    {ex}")
    print(f"\n  byte-identical by role:")
    for role, count in sorted(per_role_ok.items(), key=lambda x: -x[1]):
        print(f"    {role:18s} {count:6d}")

    if n_ok == len(gt_hf):
        print(f"\n*** PHASE E SHAPE 2 VALIDATED: {n_ok}/{len(gt_hf)} HF TENSORS "
              f"BYTE-IDENTICAL TO BRIDGE GROUND TRUTH ***")
        return 0
    print(f"\n[rcv] {n_ok}/{len(gt_hf)} byte-identical; "
          f"{n_drift} drift, {n_missing} missing, {n_extra} extras")
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
