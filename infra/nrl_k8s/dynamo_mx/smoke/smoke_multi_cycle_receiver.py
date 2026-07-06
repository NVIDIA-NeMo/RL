"""Multi-cycle Megatron-MX receiver — measure per-cycle timing.

Runs N back-to-back refit cycles against a single trainer source (a real
Bridge-loaded Qwen3-4B-Thinking model). Captures per-cycle wall time
broken down by phase (allocation, NIXL registration, RDMA pull, translate,
load_weights). The point is to surface whether per-cycle setup (the
buffer allocation + NIXL register_tensors) dominates steady-state cycle
cost — which is the bug John's 6s number on Llama 3.1 surfaces.

Two cycles is enough to compare cold (cycle 1) vs warm (cycle 2). The
difference between cycle 1 and cycle 2 is the "would-be cached" overhead.

Designed to be run from inside a single pod (no separate trainer / DGD).
Uses the publish_self_as_source pattern? No — uses the existing trainer
publisher's output, just re-pulls it multiple times.
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


GT_PATH = "/mnt/rl-workspace/kavink/phase-e-shape-1-groundtruth.pt"
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
N_CYCLES = 3  # repeat refit cycles to surface cold vs warm timing

# Whether to apply the buffer-caching fix (matches the proposed code change)
# at receiver-side. Toggled via env so we can compare A/B without redeploying.
CACHE_BUFFERS = os.environ.get("MX_CACHE_BUFFERS", "0") == "1"


def main() -> int:
    print(f"[rcv] CACHE_BUFFERS={CACHE_BUFFERS} (env MX_CACHE_BUFFERS={os.environ.get('MX_CACHE_BUFFERS','0')})")
    print(f"[rcv] running {N_CYCLES} back-to-back refit cycles")
    print(f"[rcv] loading GT from {GT_PATH}...")
    t0 = time.perf_counter()
    gt = torch.load(GT_PATH, weights_only=False)
    gt_hf = gt["hf_weights"]
    name_map = gt["name_map"]
    print(f"  loaded in {time.perf_counter() - t0:.1f}s ({len(gt_hf)} HF tensors)")

    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-multi-cycle-rcv",
        device_id=0,
        mx_server_url=os.environ.get(
            "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"
        ),
        worker_rank=0,
    )
    rcv.initialize(model_tensors=None)

    # Discover the trainer source (publisher must be running).
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
        print("[rcv] ERROR: no Megatron source found")
        return 3
    chosen = megatron_cands[0]
    print(f"  chosen sid={chosen.ref.mx_source_id}")

    sidecar_cfg, name_map_received = discover_megatron_context(megatron_cands)
    print(f"  sidecar cfg={sidecar_cfg}")

    # Build receive specs (once — these don't change cycle-to-cycle in
    # matched-TP).
    dev = torch.device("cuda:0")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    specs: dict[str, ReceiveSpec] = {}
    for td in chosen.registry.get("tensors", []) if chosen.registry else []:
        if not td.megatron_role:
            continue
        lookup = td.name[len("module."):] if td.name.startswith("module.") else td.name
        specs[td.name] = ReceiveSpec(
            megatron_name=td.name,
            hf_names=list(name_map_received.get(lookup, [td.name])),
            role=td.megatron_role,
            target_shape=tuple(int(s) for s in td.global_shape),
            target_dtype=td.dtype if td.dtype else "bfloat16",
            shard_axis=int(td.shard_axis),
            pp_rank=chosen.megatron_meta.pp_rank,
            role_descriptor=dict(td.megatron_extras or {}),
        )

    print(f"\n[rcv] {len(specs)} ReceiveSpecs built")

    # Cycle loop — measure per-cycle phase timing.
    layout = TargetTpLayout(tp_size=1, tp_rank=0)
    ctx = MegatronReceiverContext(
        target_tp_layout=layout,
        transformer_config=sidecar_cfg,
        hf_name_map=name_map_received,
        receive_specs=specs,
    )

    # Cached buffers (only if CACHE_BUFFERS=1) — survives across cycles.
    cached_buffers: dict[str, torch.Tensor] | None = None

    cycle_timings = []
    for cycle in range(1, N_CYCLES + 1):
        print(f"\n[rcv] === CYCLE {cycle} ===")
        cycle_start = time.perf_counter()
        t_alloc = 0.0
        t_register = 0.0

        # Phase 1: buffer allocation + NIXL registration.
        if not (CACHE_BUFFERS and cached_buffers is not None):
            t_a0 = time.perf_counter()
            buffers: dict[str, torch.Tensor] = {}
            total_bytes = 0
            for spec in specs.values():
                dt = dtype_map.get(spec.target_dtype, torch.bfloat16)
                b = torch.empty(spec.target_shape, dtype=dt, device=dev)
                buffers[spec.megatron_name] = b
                total_bytes += b.numel() * b.element_size()
            t_alloc = time.perf_counter() - t_a0

            t_r0 = time.perf_counter()
            rcv._receiver._nixl.register_tensors(buffers)
            t_register = time.perf_counter() - t_r0

            if CACHE_BUFFERS:
                cached_buffers = buffers
            print(f"  Phase 1 (alloc+register): {t_alloc:.3f}s alloc + {t_register:.3f}s register "
                  f"({total_bytes/1e9:.2f} GB, {len(buffers)} buffers)")
        else:
            buffers = cached_buffers
            print(f"  Phase 1 SKIPPED — using cached {len(buffers)} buffers")

        # Phase 2: RDMA pull.
        t_p0 = time.perf_counter()
        n_pulled = 0
        for _name, _t in rcv.receive_from(chosen, timeout_seconds=120.0):
            n_pulled += 1
        t_pull = time.perf_counter() - t_p0
        pulled_bytes = sum(b.numel() * b.element_size() for b in buffers.values())
        print(f"  Phase 2 (RDMA pull):       {t_pull:.3f}s ({pulled_bytes / 1e9:.2f} GB, "
              f"{pulled_bytes * 8 / t_pull / 1e9:.1f} Gbps)")

        # Phase 3: translate.
        t_t0 = time.perf_counter()
        n_hf = 0
        for _hf_name, _hf_tensor in run_refit_cycle(
            rcv, candidates=megatron_cands, context=ctx,
            pull=lambda src, dest: None,  # no-op (matched-TP pre-pulled)
            device=dev,
            pre_assembled_buffers=buffers,
        ):
            n_hf += 1
        t_translate = time.perf_counter() - t_t0
        print(f"  Phase 3 (translate):       {t_translate:.3f}s ({n_hf} HF tensors)")

        cycle_total = time.perf_counter() - cycle_start
        cycle_timings.append({
            "cycle": cycle,
            "alloc_s": t_alloc,
            "register_s": t_register,
            "pull_s": t_pull,
            "translate_s": t_translate,
            "total_s": cycle_total,
        })
        print(f"  ----- cycle total: {cycle_total:.3f}s -----")

        # Brief pause between cycles (let any background work settle).
        time.sleep(2)

    # Summary
    print(f"\n[rcv] === SUMMARY ({N_CYCLES} cycles, CACHE_BUFFERS={CACHE_BUFFERS}) ===")
    print(f"{'cycle':>6} {'alloc':>8} {'register':>10} {'pull':>8} {'translate':>10} {'total':>8}")
    for r in cycle_timings:
        print(f"{r['cycle']:>6} {r['alloc_s']:>8.3f} {r['register_s']:>10.3f} "
              f"{r['pull_s']:>8.3f} {r['translate_s']:>10.3f} {r['total_s']:>8.3f}")

    if N_CYCLES >= 2:
        c1 = cycle_timings[0]['total_s']
        c2 = cycle_timings[1]['total_s']
        savings = c1 - c2
        print(f"\n  cycle 1 → cycle 2 delta: {savings:.3f}s "
              f"(cycle 1 = {c1:.3f}s, cycle 2 = {c2:.3f}s)")
        if CACHE_BUFFERS:
            print(f"  with caching: cycle 2 should be ≈ pull + translate only")
        else:
            print(f"  without caching: cycle 2 still pays alloc+register cost on every refit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
