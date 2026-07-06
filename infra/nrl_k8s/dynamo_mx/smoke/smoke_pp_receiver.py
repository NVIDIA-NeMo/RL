"""PP=2 cluster smoke — receiver side.

PP=1 receiver that unions two pipeline stages (each owning disjoint
layers). Discovers both PP-stage sources, builds ReceiveSpecs tagged
with each source's pp_rank, and the planner routes each layer's tensors
to the stage that owns it (matching source.pp_rank == spec.pp_rank).
TP=1 throughout, so every plan is single-source (no concat) — this
isolates the PP-routing axis. Verifies byte-identity for all 4 layers.
"""

from __future__ import annotations

import os
import socket
import time

import torch

from modelexpress import MxV2RefitReceiver
from modelexpress.nemo_rl_v2 import TargetTpLayout, MegatronTensorSpec
from modelexpress.megatron_translator import (
    MegatronReceiverContext, ReceiveSpec, assemble_into_destination,
    discover_megatron_context, translate_megatron_to_hf,
)

MODEL_NAME = "smoke/pp-toy"
GT_PATH = "/mnt/rl-workspace/kavink/smoke-pp-groundtruth.pt"
PP_SIZE = 2


def main() -> int:
    print(f"[pp-rcv] loading GT from {GT_PATH}")
    gt = torch.load(GT_PATH, weights_only=False)
    n_layers = int(gt["n_layers"])
    layout = TargetTpLayout(tp_size=1, tp_rank=0)
    device = torch.device("cuda:0")

    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-pp-rcv", device_id=0,
        mx_server_url=os.environ.get(
            "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"),
        worker_rank=0,
    )
    rcv.initialize(model_tensors=None)

    print(f"[pp-rcv] discovering {PP_SIZE} PP-stage sources for {MODEL_NAME}...")
    deadline = time.time() + 120
    megatron_cands = []
    while time.time() < deadline:
        cands = rcv.discover_v2_sources(
            model_name=MODEL_NAME, min_version=1,
            same_rank_only=False, include_replicas=True,
        )
        allm = [c for c in cands if c.megatron_meta is not None]
        # dedupe by pp_rank, freshest wins
        by_pp = {}
        for c in sorted(allm, key=lambda x: -x.updated_at):
            if c.megatron_meta.pp_rank not in by_pp:
                by_pp[c.megatron_meta.pp_rank] = c
        if len(by_pp) >= PP_SIZE:
            megatron_cands = [by_pp[r] for r in sorted(by_pp.keys())]
            break
        print(f"  waiting (found {len(by_pp)}/{PP_SIZE} pp stages)")
        time.sleep(2)
    if len(megatron_cands) < PP_SIZE:
        print(f"[pp-rcv] ERROR: only {len(megatron_cands)} pp stages found")
        return 3
    for c in megatron_cands:
        mm = c.megatron_meta
        print(f"  source sid={c.ref.mx_source_id[:16]} pp_rank={mm.pp_rank}/{mm.pp_size}")

    sidecar_cfg, name_map = discover_megatron_context(megatron_cands)
    if sidecar_cfg is None:
        print("[pp-rcv] ERROR: no sidecar")
        return 4
    print(f"  sidecar cfg={sidecar_cfg}; name_map has {len(name_map)} entries")

    # Pull each stage's manifest into scratch.
    print(f"[pp-rcv] pulling each stage via receive_weights_scratch...")
    scratch = {}
    t0 = time.perf_counter()
    total_bytes = 0
    for c in megatron_cands:
        shape_table = {
            td.name: tuple(int(s) for s in td.global_shape)
            for td in (c.registry.get("tensors", []) if c.registry else [])
            if not td.name.startswith("__mx_") and tuple(td.global_shape)
        }
        bufs = {}
        for name, t in rcv._receiver.receive_weights_scratch(
            c.ref, timeout_seconds=120.0, tensor_shapes=shape_table,
        ):
            bufs[name] = t
            total_bytes += t.numel() * t.element_size()
        scratch[c.ref.mx_source_id] = bufs
        print(f"  pp_rank={c.megatron_meta.pp_rank}: {len(bufs)} tensors")
    elapsed = time.perf_counter() - t0
    print(f"  pulled {sum(len(d) for d in scratch.values())} tensors, "
          f"{total_bytes/1e6:.2f} MB, {elapsed:.2f}s")

    SHARD_AXIS_BY_ROLE = {
        "column": 0, "qkv_column": 0, "gated_mlp_column": 0,
        "vocab_parallel": 0, "row": 1, "replicated": 0,
    }
    # Union all stages' registries; each spec tagged with its source pp_rank.
    receive_specs = {}
    for c in megatron_cands:
        for td in (c.registry.get("tensors", []) if c.registry else []):
            if not td.megatron_role or td.name in receive_specs:
                continue
            role = td.megatron_role
            shard_axis = SHARD_AXIS_BY_ROLE.get(role, int(td.shard_axis))
            # TP=1: global shape == per-rank shape (no expansion).
            receive_specs[td.name] = ReceiveSpec(
                megatron_name=td.name,
                hf_names=list(name_map.get(td.name, [td.name])),
                role=role,
                target_shape=tuple(int(s) for s in td.global_shape),
                target_dtype=td.dtype or "bfloat16",
                shard_axis=shard_axis,
                pp_rank=c.megatron_meta.pp_rank,
                role_descriptor=dict(td.megatron_extras or {}),
            )
    print(f"[pp-rcv] built {len(receive_specs)} ReceiveSpecs across "
          f"{len(set(s.pp_rank for s in receive_specs.values()))} pp stages")

    target_specs = {
        m: MegatronTensorSpec(
            role=rs.role, target_shape=rs.target_shape, target_dtype=rs.target_dtype,
            shard_axis=rs.shard_axis, pp_rank=rs.pp_rank,
            role_descriptor=dict(rs.role_descriptor or {}),
        )
        for m, rs in receive_specs.items()
    }
    plans = rcv.pick_megatron_slice_plans(
        megatron_cands, target_tp_layout=layout, target_tensor_specs=target_specs,
    )
    # Confirm each plan routed to exactly the owning stage.
    multi = [p for p in plans if len(p.sources) != 1]
    print(f"[pp-rcv] planner produced {len(plans)} plans; "
          f"{len(multi)} with !=1 source (expect 0 for pure PP)")

    hf_results = {}
    for plan in plans:
        if not plan.sources:
            print(f"  WARN empty sources for {plan.tensor_name}")
            continue
        rs = receive_specs[plan.tensor_name]

        def _pull_factory(name=plan.tensor_name, assembly=plan.assembly):
            def _pull(src, dest):
                full = scratch.get(src.mx_source_id, {}).get(name)
                if full is None:
                    raise RuntimeError(f"PP: scratch missing {name!r} from {src.mx_source_id}")
                axis = 1 if assembly == "concat_dim1" else 0
                if src.source_subslice is not None:
                    slo, shi = src.source_subslice
                    full = full.narrow(axis, slo, shi - slo)
                dest.copy_(full, non_blocking=True)
            return _pull

        assembled = assemble_into_destination(plan, pull=_pull_factory(), device=device)
        for hf_name, hf_tensor in translate_megatron_to_hf(
            plan, assembled, transformer_config=sidecar_cfg, hf_names=list(rs.hf_names),
        ):
            hf_results[hf_name] = hf_tensor.cpu()
    print(f"[pp-rcv] translated {len(hf_results)} HF tensors")

    # Build expected from GT (all layers).
    expected = {}
    for L in range(n_layers):
        t = gt["layers"][L]
        expected[f"model.layers.{L}.self_attn.q_proj.weight"] = t["q"]
        expected[f"model.layers.{L}.self_attn.k_proj.weight"] = t["k"]
        expected[f"model.layers.{L}.self_attn.v_proj.weight"] = t["v"]
        expected[f"model.layers.{L}.mlp.gate_proj.weight"] = t["gate"]
        expected[f"model.layers.{L}.mlp.up_proj.weight"] = t["up"]
        expected[f"model.layers.{L}.self_attn.o_proj.weight"] = t["o_proj"]
        expected[f"model.layers.{L}.mlp.down_proj.weight"] = t["fc2_col"]
        expected[f"model.layers.{L}.input_layernorm.weight"] = t["norm"]

    print(f"[pp-rcv] validating byte-identity vs GT ({len(expected)} tensors)...")
    n_ok = n_drift = n_missing = 0
    per_layer_ok = {L: 0 for L in range(n_layers)}
    for hf_name, exp in expected.items():
        L = int(hf_name.split(".")[2])
        got = hf_results.get(hf_name)
        if got is None:
            n_missing += 1
            print(f"  MISSING {hf_name}")
            continue
        if got.shape == exp.shape and torch.equal(got.cpu(), exp.cpu()):
            n_ok += 1
            per_layer_ok[L] += 1
        else:
            n_drift += 1
            print(f"  DRIFT {hf_name} got={tuple(got.shape)} exp={tuple(exp.shape)}")

    print(f"\n  byte-identical: {n_ok}/{len(expected)}  drift={n_drift} missing={n_missing}")
    print(f"  per-layer OK: {per_layer_ok}  (layers 0,1 from stage 0; 2,3 from stage 1)")
    if n_ok == len(expected):
        print(f"\n*** PP=2 CLUSTER SMOKE VALIDATED: {n_ok}/{len(expected)} BYTE-IDENTICAL "
              f"across {n_layers} layers / {PP_SIZE} pipeline stages ***")
        return 0
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
