"""Mixed-TP cluster smoke — receiver side.

Target: TP=1 (target-narrower: vLLM TP < source TP, each receiver
concatenates multiple source ranks' full shards along the role's
shard axis). v0 host-side scratch+slice path is BANDWIDTH-OPTIMAL
in this direction.

Discovers BOTH source ranks (tp_rank=0 + tp_rank=1 of a TP=2 trainer),
pulls each one's full manifest into per-source scratch dicts via
``receive_weights_scratch``, then for each plan slice-copies from
scratch into the planner's pre-narrowed destination view. Translates
to HF and asserts byte-identity against the global ground truth.

This replays the exact code path in
``VllmInternalWorkerExtension._update_weights_via_mx_megatron``'s
mixed-TP branch.
"""

from __future__ import annotations

import os
import socket
import sys
import time

import torch

from modelexpress import MxV2RefitReceiver
from modelexpress.nemo_rl_v2 import TargetTpLayout, MegatronTensorSpec
from modelexpress.megatron_helpers import MegatronTransformerConfig
from modelexpress.megatron_translator import (
    MegatronReceiverContext,
    ReceiveSpec,
    assemble_into_destination,
    discover_megatron_context,
    translate_megatron_to_hf,
)


MODEL_NAME = "smoke/mixed-tp-toy"
GT_PATH = "/mnt/rl-workspace/kavink/smoke-mixed-tp-groundtruth.pt"


def main() -> int:
    print(f"[rcv] loading ground truth from {GT_PATH}")
    gt = torch.load(GT_PATH, weights_only=False)
    cfg = MegatronTransformerConfig(**gt["cfg"])
    source_tp = int(gt["source_tp"])
    print(f"  source_tp={source_tp} cfg={cfg}")

    target_tp = 1
    target_rank = 0
    layout = TargetTpLayout(tp_size=target_tp, tp_rank=target_rank)
    device = torch.device("cuda:0")

    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-mixed-tp-rcv",
        device_id=0,
        mx_server_url=os.environ.get(
            "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"
        ),
        worker_rank=0,
    )
    rcv.initialize(model_tensors=None)

    print(f"\n[rcv] discovering Megatron sources for model={MODEL_NAME}...")
    deadline = time.time() + 120
    cands = []
    while time.time() < deadline:
        cands = rcv.discover_v2_sources(
            model_name=MODEL_NAME, min_version=1,
            same_rank_only=False, include_replicas=True,
        )
        all_megatron = [c for c in cands if c.megatron_meta is not None]
        if len(all_megatron) >= source_tp:
            break
        print(f"  waiting (found {len(all_megatron)}/{source_tp} megatron sources)")
        time.sleep(2)
    all_megatron = [c for c in cands if c.megatron_meta is not None]
    if len(all_megatron) < source_tp:
        print(f"[rcv] ERROR: only found {len(all_megatron)} sources, want >= {source_tp}")
        return 3

    # Dedupe by tp_rank, picking the freshest per rank (handles stale
    # entries from previous runs in the MX catalog).
    by_rank: dict[int, object] = {}
    for c in sorted(all_megatron, key=lambda x: -x.updated_at):
        if c.megatron_meta.tp_rank not in by_rank:
            by_rank[c.megatron_meta.tp_rank] = c
    megatron_cands = [by_rank[r] for r in sorted(by_rank.keys())]
    if len(megatron_cands) != source_tp:
        print(f"[rcv] ERROR: after dedup got {len(megatron_cands)} ranks, "
              f"want {source_tp}")
        return 3
    for c in megatron_cands:
        mm = c.megatron_meta
        print(f"  source sid={c.ref.mx_source_id} tp_rank={mm.tp_rank}/{mm.tp_size}")

    # Pull the sidecar (config + name map) from one of the candidates.
    sidecar_cfg, name_map = discover_megatron_context(megatron_cands)
    if sidecar_cfg is None:
        print("[rcv] ERROR: no Megatron sidecar found")
        return 4
    print(f"  sidecar cfg={sidecar_cfg}; name_map has {len(name_map)} entries")

    # ---- Pull each source's full manifest into a scratch dict. ----
    # This is exactly what the mixed-TP branch of
    # _update_weights_via_mx_megatron does.
    print(f"\n[rcv] pulling each source via receive_weights_scratch...")
    scratch: dict[str, dict[str, torch.Tensor]] = {}
    t0 = time.perf_counter()
    total_bytes = 0
    for c in megatron_cands:
        # Build a per-source shape table from its registry so we can
        # reshape the 1-D scratch buffers back to the source's per-rank
        # shape. receive_weights_scratch returns flat numel buffers by
        # default.
        shape_table = {
            td.name: tuple(int(s) for s in td.global_shape)
            for td in (c.registry.get("tensors", []) if c.registry else [])
            if not td.name.startswith("__mx_") and tuple(td.global_shape)
        }
        bufs: dict[str, torch.Tensor] = {}
        for name, t in rcv._receiver.receive_weights_scratch(
            c.ref, timeout_seconds=120.0, tensor_shapes=shape_table,
        ):
            bufs[name] = t
            total_bytes += t.numel() * t.element_size()
        scratch[c.ref.mx_source_id] = bufs
        print(f"  sid={c.ref.mx_source_id[:16]} tp_rank={c.megatron_meta.tp_rank}: "
              f"{len(bufs)} tensors, shapes via registry")
    elapsed = time.perf_counter() - t0
    print(f"  pulled {sum(len(d) for d in scratch.values())} tensors across "
          f"{len(scratch)} sources, {total_bytes / 1e6:.2f} MB, {elapsed:.2f}s, "
          f"{total_bytes * 8 / elapsed / 1e9:.2f} Gbps aggregate")

    # ---- Build receive specs from the source registries. ----
    # The name_map sidecar tells us which HF names each Megatron tensor
    # expands into; the per-tensor megatron_role lives on each source's
    # TensorDescriptorV2. Read them off the first source.
    # Receiver-side knowledge of shard_axis per Megatron role. In the
    # production receiver path this comes from the inference model's
    # structure (each parameter has a known parallelism kind). For this
    # smoke we just hardcode the role → axis mapping.
    SHARD_AXIS_BY_ROLE = {
        "column": 0, "qkv_column": 0, "gated_mlp_column": 0,
        "vocab_parallel": 0, "row": 1,
        "expert_column": 0, "expert_row": 0,
        "replicated": 0,  # unused for replicated
    }
    receive_specs: dict[str, ReceiveSpec] = {}
    # Union all candidate registries — replicated tensors are only
    # published by rank 0, so the spec set has to come from all sources.
    for c in megatron_cands:
        for td in (c.registry.get("tensors", []) if c.registry else []):
            if not td.megatron_role or td.name in receive_specs:
                continue
            role = td.megatron_role
            shard_axis = SHARD_AXIS_BY_ROLE.get(role, int(td.shard_axis))
            per_rank_shape = list(td.global_shape)
            global_shape = list(per_rank_shape)
            # For replicated, no expansion. For tp-sharded (column / row /
            # qkv / gated / vocab), the per-source manifest shape is
            # global // source_tp on the shard axis.
            if role != "replicated":
                global_shape[shard_axis] = per_rank_shape[shard_axis] * source_tp
            receive_specs[td.name] = ReceiveSpec(
                megatron_name=td.name,
                hf_names=list(name_map.get(td.name, [td.name])),
                role=role,
                target_shape=tuple(int(s) for s in global_shape),
                target_dtype=td.dtype or "bfloat16",
                shard_axis=shard_axis,
                pp_rank=c.megatron_meta.pp_rank,
                role_descriptor=dict(td.megatron_extras or {}),
            )
    print(f"\n[rcv] built {len(receive_specs)} ReceiveSpecs")
    for name, spec in receive_specs.items():
        print(f"  {name[:60]:60s} role={spec.role:25s} target={spec.target_shape}")

    # ---- Run the slice planner. ----
    target_specs = {
        m_name: MegatronTensorSpec(
            role=rs.role, target_shape=rs.target_shape,
            target_dtype=rs.target_dtype, shard_axis=rs.shard_axis,
            pp_rank=rs.pp_rank, role_descriptor=dict(rs.role_descriptor or {}),
        )
        for m_name, rs in receive_specs.items()
    }
    plans = rcv.pick_megatron_slice_plans(
        megatron_cands, target_tp_layout=layout, target_tensor_specs=target_specs,
    )
    print(f"\n[rcv] planner produced {len(plans)} plans:")
    for p in plans:
        print(f"  {p.tensor_name[:50]:50s} assembly={p.assembly:18s} sources={len(p.sources)}")

    # ---- Translate each plan: scratch -> assembled -> HF tensors. ----
    print(f"\n[rcv] assembling + translating...")
    hf_results: dict[str, torch.Tensor] = {}
    for plan in plans:
        if not plan.sources:
            print(f"  WARN: empty sources for {plan.tensor_name}")
            continue
        rs = receive_specs[plan.tensor_name]

        def _pull_factory(name=plan.tensor_name, assembly=plan.assembly):
            def _pull(src, dest):
                full = scratch.get(src.mx_source_id, {}).get(name)
                if full is None:
                    raise RuntimeError(
                        f"mixed-TP: scratch missing {name!r} from "
                        f"source {src.mx_source_id}"
                    )
                axis = 1 if assembly == "concat_dim1" else 0
                if src.source_subslice is not None:
                    slo, shi = src.source_subslice
                    slice_src = full.narrow(axis, slo, shi - slo)
                else:
                    slice_src = full
                if slice_src.shape != dest.shape:
                    raise RuntimeError(
                        f"shape mismatch on {name}: src={tuple(slice_src.shape)} "
                        f"dest={tuple(dest.shape)} axis={axis} "
                        f"subslice={src.source_subslice}"
                    )
                dest.copy_(slice_src, non_blocking=True)
            return _pull

        assembled = assemble_into_destination(
            plan, pull=_pull_factory(), device=device,
        )
        for hf_name, hf_tensor in translate_megatron_to_hf(
            plan, assembled,
            transformer_config=sidecar_cfg,
            hf_names=list(rs.hf_names),
        ):
            hf_results[hf_name] = hf_tensor.cpu()

    print(f"[rcv] translated {len(hf_results)} HF tensors")

    # ---- Compare against ground truth. ----
    print(f"\n[rcv] validating byte-identity vs ground truth...")
    expected = {
        "model.layers.0.self_attn.q_proj.weight": gt["q"],
        "model.layers.0.self_attn.k_proj.weight": gt["k"],
        "model.layers.0.self_attn.v_proj.weight": gt["v"],
        "model.layers.0.mlp.gate_proj.weight": gt["gate"],
        "model.layers.0.mlp.up_proj.weight": gt["up"],
        "model.layers.0.self_attn.o_proj.weight": gt["o_proj"],
        "model.layers.0.mlp.down_proj.weight": gt["dense_col"],
        "model.layers.0.input_layernorm.weight": gt["norm_w"],
    }
    n_ok = 0
    n_drift = 0
    n_missing = 0
    drift_examples = []
    for hf_name, exp in expected.items():
        got = hf_results.get(hf_name)
        if got is None:
            n_missing += 1
            print(f"  MISSING  {hf_name}")
            continue
        if got.shape != exp.shape:
            n_drift += 1
            drift_examples.append((hf_name, "shape", tuple(got.shape), tuple(exp.shape)))
            continue
        if torch.equal(got.cpu(), exp.cpu()):
            n_ok += 1
            print(f"  OK       {hf_name:55s} {tuple(got.shape)}")
        else:
            n_drift += 1
            diff = (got.cpu().float() - exp.cpu().float()).abs()
            drift_examples.append((hf_name, "values",
                                   f"max={diff.max().item():.4e}",
                                   f"mean={diff.mean().item():.4e}"))

    print(f"\n  byte-identical:  {n_ok}/{len(expected)}")
    print(f"  drift:           {n_drift}")
    print(f"  missing:         {n_missing}")
    if drift_examples:
        print(f"\n  drift examples:")
        for ex in drift_examples:
            print(f"    {ex}")

    if n_ok == len(expected):
        print(f"\n*** MIXED-TP CLUSTER SMOKE VALIDATED: "
              f"{n_ok}/{len(expected)} BYTE-IDENTICAL ***")
        return 0
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
