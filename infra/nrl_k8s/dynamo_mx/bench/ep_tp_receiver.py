"""EP-trainer -> TP-rollout receiver for the EP->TP2 first-party refit benchmark.

Pairs with ep_publisher.py (EP=N Megatron publisher). Discovers all N EP sources,
gathers each source's local experts + the replicated/attention tensors, reshards
dense layers to the target TP, translates Megatron->HF, and (for TP1) verifies
byte-identity vs the phase-e ground truth. Reports per-source pull bytes/time
(Istvan's "pull balance across trainers" concern) + aggregate transfer rate.

Runs on the vLLM WORKER image (full CUDA+IB NIXL; no Megatron needed on receive
side). Env: MODEL_EXPRESS_URL, EP_SIZE (default 4), TARGET_TP (default 1),
TARGET_TP_RANK (default 0), GT_PATH (optional, for TP1 byte-identity).

NOTE: EP>1 grouped-expert local->global remap is validated only at EP1 upstream;
this harness is the first EP>1 exercise, so expect to iterate if experts collide.
"""
from __future__ import annotations
import os, socket, time
from collections import Counter
import torch

from modelexpress import MxV2RefitReceiver
from modelexpress.nemo_rl_v2 import TargetTpLayout, MegatronTensorSpec
from modelexpress.megatron_translator import (
    MegatronReceiverContext, ReceiveSpec, discover_megatron_context, run_refit_cycle,
)

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
EP_SIZE = int(os.environ.get("EP_SIZE", "4"))
TARGET_TP = int(os.environ.get("TARGET_TP", "1"))
TARGET_TP_RANK = int(os.environ.get("TARGET_TP_RANK", "0"))
GT_PATH = os.environ.get("GT_PATH", "/mnt/rl-workspace/kavink/phase-e-shape-2-qwen3-moe-groundtruth.pt")
SHARD_AXIS_BY_ROLE = {"column": 0, "qkv_column": 0, "gated_mlp_column": 0,
                      "vocab_parallel": 0, "row": 1, "expert_column": 0,
                      "expert_row": 0, "replicated": 0}


def main() -> int:
    device = torch.device("cuda:0")
    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-ep{EP_SIZE}-tp{TARGET_TP}-rcv",
        device_id=0,
        mx_server_url=os.environ.get("MODEL_EXPRESS_URL",
                                     "modelexpress-server.kavin.svc.cluster.local:8001"),
        worker_rank=TARGET_TP_RANK,
    )
    rcv.initialize(model_tensors=None)

    # ---- discover N EP sources, dedupe by ep_rank ----
    print(f"[rcv] discovering {EP_SIZE} EP sources for {MODEL} ...", flush=True)
    deadline = time.time() + 120
    by_ep = {}
    while time.time() < deadline:
        cands = rcv.discover_v2_sources(model_name=MODEL, min_version=1,
                                        same_rank_only=False, include_replicas=True)
        by_ep = {}
        for c in sorted([c for c in cands if c.megatron_meta is not None],
                        key=lambda x: -x.updated_at):
            by_ep.setdefault(c.megatron_meta.ep_rank, c)
        if len(by_ep) >= EP_SIZE:
            break
        print(f"  {len(by_ep)}/{EP_SIZE} EP ranks ...", flush=True)
        time.sleep(3)
    if len(by_ep) < EP_SIZE:
        print(f"[rcv] ERROR: only {len(by_ep)}/{EP_SIZE} EP sources"); return 3
    megatron_cands = [by_ep[e] for e in sorted(by_ep)]
    print(f"  {len(megatron_cands)} EP sources: " +
          ", ".join(f"ep{c.megatron_meta.ep_rank}={c.ref.mx_source_id[:8]}" for c in megatron_cands),
          flush=True)

    sidecar_cfg, name_map = discover_megatron_context(megatron_cands)
    if sidecar_cfg is None:
        print("[rcv] ERROR: no sidecar"); return 4

    # ---- build receive_specs: union across all EP sources ----
    receive_specs: dict[str, ReceiveSpec] = {}
    roles = Counter()
    for c in megatron_cands:
        for td in (c.registry.get("tensors", []) if c.registry else []):
            if not td.megatron_role or td.name in receive_specs:
                continue
            role = td.megatron_role
            roles[role] += 1
            axis = SHARD_AXIS_BY_ROLE.get(role, int(td.shard_axis))
            lookup = td.name[len("module."):] if td.name.startswith("module.") else td.name
            receive_specs[td.name] = ReceiveSpec(
                megatron_name=td.name, hf_names=list(name_map.get(lookup, [td.name])),
                role=role, target_shape=tuple(int(s) for s in td.global_shape),
                target_dtype=td.dtype or "bfloat16", shard_axis=axis,
                pp_rank=c.megatron_meta.pp_rank,
                role_descriptor=dict(td.megatron_extras or {}),
            )
    print(f"[rcv] {len(receive_specs)} receive_specs; roles={dict(roles)}", flush=True)

    # ---- per-source scratch pull (records per-source bytes for balance) ----
    print(f"[rcv] pulling {len(megatron_cands)} EP sources (scratch) ...", flush=True)
    scratch: dict[str, dict[str, torch.Tensor]] = {}
    per_src = {}
    t_all = time.perf_counter()
    for c in megatron_cands:
        shp = {td.name: tuple(int(s) for s in td.global_shape)
               for td in (c.registry.get("tensors", []) if c.registry else [])
               if not td.name.startswith("__mx_") and tuple(td.global_shape)}
        bufs, nb = {}, 0
        t0 = time.perf_counter()
        for name, t in rcv._receiver.receive_weights_scratch(c.ref, timeout_seconds=300.0, tensor_shapes=shp):
            bufs[name] = t; nb += t.numel() * t.element_size()
        dt = time.perf_counter() - t0
        scratch[c.ref.mx_source_id] = bufs
        per_src[c.megatron_meta.ep_rank] = (nb, dt)
        print(f"  ep{c.megatron_meta.ep_rank}: {len(bufs)} tensors, {nb/1e9:.2f} GB, "
              f"{dt:.2f}s, {nb*8/dt/1e9:.1f} Gbps", flush=True)
    tot = sum(nb for nb, _ in per_src.values()); dt_all = time.perf_counter() - t_all
    print(f"[rcv] PULL BALANCE (per EP source): {dict((k, round(v[0]/1e9,2)) for k,v in per_src.items())} GB", flush=True)
    print(f"[rcv] aggregate: {tot/1e9:.2f} GB in {dt_all:.2f}s = {tot*8/dt_all/1e9:.1f} Gbps", flush=True)

    # ---- plan + assemble + translate ----
    layout = TargetTpLayout(tp_size=TARGET_TP, tp_rank=TARGET_TP_RANK)
    ctx = MegatronReceiverContext(target_tp_layout=layout, transformer_config=sidecar_cfg,
                                  hf_name_map=name_map, receive_specs=receive_specs)

    def _pull(src, dest):
        # find the tensor in the owning source's scratch by matching plan name
        for sid, bufs in scratch.items():
            if src.mx_source_id == sid:
                return  # scratch pre-filled; run_refit_cycle assembles from pre_assembled
        return

    print(f"[rcv] planning + translating (EP{EP_SIZE} -> TP{TARGET_TP}) ...", flush=True)
    # flatten scratch into one pre-assembled buffer dict (name -> tensor from owning source)
    pre = {}
    for bufs in scratch.values():
        for n, t in bufs.items():
            pre.setdefault(n, t)
    t0 = time.perf_counter()
    gt = None
    if TARGET_TP == 1 and os.path.exists(GT_PATH):
        gt = torch.load(GT_PATH, weights_only=False, mmap=True).get("hf_weights")
    n_ok = n_seen = n_drift = 0
    for hf_name, hf_t in run_refit_cycle(rcv, candidates=megatron_cands, context=ctx,
                                         pull=lambda s, d: None, device=device,
                                         pre_assembled_buffers=pre):
        n_seen += 1
        if gt is not None:
            exp = gt.get(hf_name)
            if exp is not None:
                got = hf_t.detach().cpu()
                e = exp.to(got.dtype) if got.dtype != exp.dtype else exp
                if got.shape == e.shape and torch.equal(got, e):
                    n_ok += 1
                else:
                    n_drift += 1
    dt = time.perf_counter() - t0
    print(f"[rcv] refit cycle: {n_seen} HF tensors in {dt:.2f}s", flush=True)
    if gt is not None:
        print(f"[rcv] byte-identity: {n_ok} ok / {n_drift} drift / {len(gt)} GT", flush=True)
        print("RESULT:", "PASS - EP->TP1 refit byte-identical" if n_ok == len(gt)
              else f"PARTIAL - {n_ok}/{len(gt)} (EP expert remap likely needs work)", flush=True)
    else:
        print(f"RESULT: EP{EP_SIZE}->TP{TARGET_TP} refit produced {n_seen} HF tensors (no GT compare)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
