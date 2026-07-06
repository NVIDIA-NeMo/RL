"""Tree fan-out verification / Stage-1 emergent-tree experiment.

Proves rollout-to-rollout fan-out: a seed rollout pulls the trainer's
Megatron weights, converts to inference (HF) format, and republishes
itself as an inference_replica; follower rollouts then pull the
already-converted HF weights FROM THE SEED (not the trainer), asserting
their chosen source role is 'inference_replica'.

This is the real RL-refit fan-out (not the HF-download cold-start
analogy): the data path from seed->follower is inference-format
(§4.7 "workers republish in inference format"), so the follower does a
plain by-name HF pull with NO Megatron translation.

Roles:
  seed:     discover trainer -> pull Megatron -> translate to HF ->
            register HF buffers -> publish_self_as_source
  follower: discover(prefer_replicas=True) -> MUST pick a seed replica ->
            pull HF tensors by name -> byte-identity vs GT

Env:
  FANOUT_MODEL   default 'smoke/megatron-mx-toy'
  FANOUT_GT      default '/mnt/rl-workspace/kavink/smoke-megatron-mx-groundtruth.pt'
  FANOUT_HOLD    seed republish hold seconds (default 300)

Usage: python smoke_fanout_receiver.py {seed|follower}
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time

import torch

# Surface modelexpress INFO timing logs (match_tensors, prep_xfer_dlist,
# "RDMA transfer complete: ... Gbps") so we can separate pure wire BW from
# registration + D2H-copy overhead in the follower measurement.
if os.environ.get("FANOUT_VERBOSE", "0") == "1":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    logging.getLogger("modelexpress.nixl_transfer").setLevel(logging.INFO)

from modelexpress import MxV2RefitReceiver
from modelexpress.nemo_rl_v2 import (
    TargetTpLayout, MegatronTensorSpec, ROLE_INFERENCE_REPLICA, ROLE_TRAINER,
)
from modelexpress.megatron_translator import (
    ReceiveSpec, assemble_into_destination, discover_megatron_context,
    translate_megatron_to_hf,
)

MODEL = os.environ.get("FANOUT_MODEL", "smoke/megatron-mx-toy")
GT = os.environ.get("FANOUT_GT", "/mnt/rl-workspace/kavink/smoke-megatron-mx-groundtruth.pt")
HOLD = int(os.environ.get("FANOUT_HOLD", "300"))
SHARD_AXIS = {"column": 0, "qkv_column": 0, "gated_mlp_column": 0,
              "vocab_parallel": 0, "row": 1, "replicated": 0}


def _gt_hf(gt):
    return gt["hf_weights"] if isinstance(gt, dict) and "hf_weights" in gt else gt


def _pull_from_trainer(rcv, device, tag):
    """Discover trainer, pull Megatron manifest (timed), translate to HF.
    Returns (hf_gpu dict, pull_seconds, pull_bytes)."""
    print(f"[{tag}] discovering TRAINER...")
    cand = None
    deadline = time.time() + 120
    while time.time() < deadline:
        cands = rcv.discover_v2_sources(model_name=MODEL, min_version=1,
                                        same_rank_only=False, include_replicas=False)
        m = [c for c in cands if c.megatron_meta is not None and c.role == ROLE_TRAINER]
        if m:
            cand = m[0]; break
        time.sleep(2)
    if cand is None:
        raise RuntimeError(f"[{tag}] no trainer")
    print(f"[{tag}] pull+translate from TRAINER sid={cand.ref.mx_source_id[:16]}")
    sidecar_cfg, name_map = discover_megatron_context([cand])

    shape_table = {td.name: tuple(int(s) for s in td.global_shape)
                   for td in (cand.registry.get("tensors", []) if cand.registry else [])
                   if not td.name.startswith("__mx_") and tuple(td.global_shape)}
    scratch = {}
    _t0 = time.perf_counter()
    pull_bytes = 0
    for name, t in rcv._receiver.receive_weights_scratch(
            cand.ref, timeout_seconds=300.0, tensor_shapes=shape_table):
        scratch[name] = t
        pull_bytes += t.numel() * t.element_size()
    pull_s = time.perf_counter() - _t0

    specs = {}
    for td in (cand.registry.get("tensors", []) if cand.registry else []):
        if not td.megatron_role:
            continue
        specs[td.name] = ReceiveSpec(
            megatron_name=td.name, hf_names=list(name_map.get(td.name, [td.name])),
            role=td.megatron_role, target_shape=tuple(int(s) for s in td.global_shape),
            target_dtype=td.dtype or "bfloat16",
            shard_axis=SHARD_AXIS.get(td.megatron_role, int(td.shard_axis)),
            pp_rank=cand.megatron_meta.pp_rank if cand.megatron_meta else 0,
            role_descriptor=dict(td.megatron_extras or {}))
    tspecs = {m: MegatronTensorSpec(role=rs.role, target_shape=rs.target_shape,
              target_dtype=rs.target_dtype, shard_axis=rs.shard_axis, pp_rank=rs.pp_rank,
              role_descriptor=dict(rs.role_descriptor or {})) for m, rs in specs.items()}
    plans = rcv.pick_megatron_slice_plans(
        [cand], target_tp_layout=TargetTpLayout(tp_size=1, tp_rank=0),
        target_tensor_specs=tspecs)
    hf_gpu = {}
    for plan in plans:
        if not plan.sources:
            continue
        rs = specs[plan.tensor_name]
        def _pull(src, dest, _n=plan.tensor_name):
            dest.copy_(scratch[_n], non_blocking=True)
        assembled = assemble_into_destination(plan, pull=_pull, device=device)
        for hf_name, hf_t in translate_megatron_to_hf(
                plan, assembled, transformer_config=sidecar_cfg, hf_names=list(rs.hf_names)):
            hf_gpu[hf_name] = hf_t.to(device).contiguous()
    bw = pull_bytes * 8 / pull_s / 1e9 if pull_s > 0 else 0
    print(f"[{tag}] TRAINER pull: {pull_bytes/1e9:.2f} GB in {pull_s:.2f}s ({bw:.1f} Gbps)")
    return hf_gpu, pull_s, pull_bytes


def _decoupled(rcv, device):
    """Baseline: pull directly from trainer (no fan-out). Reports timing."""
    hf_gpu, pull_s, pull_bytes = _pull_from_trainer(rcv, device, "decoupled")
    print(f"\n[decoupled] DONE: pulled {pull_bytes/1e9:.2f} GB from TRAINER in {pull_s:.2f}s "
          f"({pull_bytes*8/pull_s/1e9:.1f} Gbps)")
    return 0


def _seed(rcv, device):
    if os.environ.get("FANOUT_SEED_SYNTHETIC", "0") == "1":
        # ISOLATION TEST: skip the trainer pull entirely. Load the GT HF
        # weights straight onto the GPU, register, publish, serve. This
        # tells us whether serving from a RECEIVER-created agent is broken
        # on its own, or only after the agent has acted as an initiator
        # (pull-then-serve). No prior add_remote_agent / transfer state.
        print("[seed] SYNTHETIC mode: loading GT to GPU (NO trainer pull)")
        raw = torch.load(GT, weights_only=False)
        gt = _gt_hf(raw)
        hf_gpu = {k: v.to(device).contiguous() for k, v in gt.items()}
        print(f"[seed] loaded {len(hf_gpu)} HF tensors to GPU "
              f"({sum(t.numel()*t.element_size() for t in hf_gpu.values())/1e9:.2f} GB)")
    else:
        hf_gpu, _pull_s, _pull_bytes = _pull_from_trainer(rcv, device, "seed")
    use_arena_mode = os.environ.get("FANOUT_SEED_ARENA", "0") == "1"

    if use_arena_mode:
        # Arena mode: re-allocate the HF buffers inside a single VMM arena
        # so all N tensors share ONE contiguous VA range, registered as a
        # single NIXL region (register_arena). This is the fix for the
        # seed-serving bandwidth gap — per-tensor registration of ~398
        # small HF tensors served followers at ~half trainer BW; a single
        # dmabuf region should let a seed serve at trainer-parity.
        from modelexpress.vmm import (
            VmmArena, CudaVmmBackend, use_arena, install_pluggable_allocator,
        )
        install_pluggable_allocator()
        backend = CudaVmmBackend(device=0)
        arena = VmmArena(backend=backend, device=0)
        print(f"[seed] ARENA mode: re-allocating {len(hf_gpu)} HF tensors into one VMM range")
        hf_arena = {}
        with use_arena(arena, device):
            for k, v in hf_gpu.items():
                hf_arena[k] = v.contiguous().clone()  # allocation lands in arena
        del hf_gpu
        rcv._receiver._nixl.register_arena(arena, hf_arena)
        rcv._registered_buffers = hf_arena
        rcv._mx_arena = arena  # keep alive
        print(f"[seed] registered arena ({arena.used_bytes/1e9:.2f} GB, 1 NIXL region)")
    else:
        print(f"[seed] translated {len(hf_gpu)} HF tensors; per-tensor register...")
        rcv._receiver._nixl.register_tensors(hf_gpu)
        rcv._registered_buffers = hf_gpu
    sid = rcv.publish_self_as_source(version=1, model_name=MODEL)
    if sid is None:
        print("[seed] ERROR publish_self_as_source returned None (no buffers?)"); return 4
    print(f"[seed] published_self_as_source sid={str(sid)[:16]} — now an inference_replica. "
          f"Holding {HOLD}s (progress loop).")
    # Bare time.sleep() starves the seed's UCX worker: connection wireup +
    # UD keepalive aren't serviced, so a follower's larger (multi-hundred-ms)
    # READ trips NIXL_ERR_REMOTE_DISCONNECT mid-transfer. Tiny toy transfers
    # (~4 MB, 0.1s) finish before the timeout and masked this. A live vLLM
    # replica has its own event loop; the smoke seed must poll the agent
    # itself. get_new_notifs() drives ucp_worker_progress.
    agent = rcv._receiver._nixl._agent
    deadline = time.time() + HOLD
    while time.time() < deadline:
        try:
            agent.get_new_notifs()
        except Exception:
            pass
        time.sleep(0.002)
    return 0


def _follower(rcv, device):
    print("[follower] discovering with prefer_replicas=True (want seed replica)...")
    cand = None
    deadline = time.time() + 180
    while time.time() < deadline:
        cands = rcv.discover_v2_sources(model_name=MODEL, min_version=1,
                                        same_rank_only=False, include_replicas=True,
                                        prefer_replicas=True)
        reps = [c for c in cands if c.role == ROLE_INFERENCE_REPLICA]
        if reps:
            cand = cands[0]  # prefer_replicas sorts replica first
            break
        print(f"  waiting; visible={[(c.role, c.ref.mx_source_id[:10]) for c in cands]}")
        time.sleep(3)
    if cand is None:
        print("[follower] ERROR no replica appeared"); return 3
    print(f"[follower] chosen source role={cand.role} sid={cand.ref.mx_source_id[:16]}")
    if cand.role != ROLE_INFERENCE_REPLICA:
        print(f"[follower] FAIL: picked role={cand.role}, not a replica — tree did NOT form")
        return 5

    # Replica publishes HF-format tensors named by HF name, NO shape_registry.
    # Build the expected {hf_name: tensor} from the GT. The toy GT uses
    # short keys; map them to the HF names the seed republished (mirrors
    # the toy publisher's name_map). In the real path the inference model
    # knows its own param names/shapes directly.
    raw = torch.load(GT, weights_only=False)
    if isinstance(raw, dict) and "hf_weights" in raw:
        gt = raw["hf_weights"]  # already HF-named
    elif isinstance(raw, dict) and "q" in raw:  # toy short-key GT
        gt = {
            "model.layers.0.self_attn.q_proj.weight": raw["q"],
            "model.layers.0.self_attn.k_proj.weight": raw["k"],
            "model.layers.0.self_attn.v_proj.weight": raw["v"],
            "model.layers.0.mlp.gate_proj.weight": raw["gate"],
            "model.layers.0.mlp.up_proj.weight": raw["up"],
            "model.layers.0.self_attn.o_proj.weight": raw["o_proj"],
            "model.layers.0.mlp.down_proj.weight": raw["dense_col"],
            "model.layers.0.input_layernorm.weight": raw["norm_w"],
        }
    else:
        gt = raw
    shape_table = {k: tuple(v.shape) for k, v in gt.items()}
    _limit = int(os.environ.get("FANOUT_LIMIT", "0"))
    if _limit > 0:
        keep = list(shape_table)[:_limit]
        shape_table = {k: shape_table[k] for k in keep}
        gt = {k: gt[k] for k in keep}
        print(f"[follower] FANOUT_LIMIT={_limit}: requesting only {len(shape_table)} tensors")
    t0 = time.perf_counter()
    got = {}
    total_bytes = 0
    for name, t in rcv._receiver.receive_weights_scratch(
            cand.ref, timeout_seconds=120.0, tensor_shapes=shape_table):
        got[name] = t.cpu()
        total_bytes += t.numel() * t.element_size()
    dt = time.perf_counter() - t0
    print(f"[follower] pulled {len(got)} HF tensors FROM THE SEED REPLICA "
          f"({total_bytes/1e6:.2f} MB, {dt:.2f}s, {total_bytes*8/dt/1e9:.1f} Gbps) — no translation")

    n_ok = sum(1 for k, v in gt.items()
               if k in got and got[k].shape == v.shape and torch.equal(got[k].cpu(), v.cpu()))
    print(f"\n  byte-identical vs GT: {n_ok}/{len(gt)}")
    if n_ok == len(gt):
        print(f"\n*** TREE FAN-OUT VERIFIED: follower pulled {n_ok}/{len(gt)} byte-identical "
              f"HF tensors FROM THE SEED REPLICA (role=inference_replica), NOT the trainer ***")
        return 0
    for k, v in gt.items():
        if k not in got:
            print(f"   MISSING {k}")
        elif not torch.equal(got[k].cpu(), v.cpu()):
            print(f"   DRIFT {k}")
    return 5


def main() -> int:
    role_arg = sys.argv[1] if len(sys.argv) > 1 else "seed"
    device = torch.device("cuda:0")
    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-fanout-{role_arg}-{os.getpid()}", device_id=0,
        mx_server_url=os.environ.get(
            "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"),
        worker_rank=0)
    rcv.initialize(model_tensors=None)
    if role_arg == "seed":
        return _seed(rcv, device)
    if role_arg == "decoupled":
        return _decoupled(rcv, device)
    return _follower(rcv, device)


if __name__ == "__main__":
    raise SystemExit(main())
