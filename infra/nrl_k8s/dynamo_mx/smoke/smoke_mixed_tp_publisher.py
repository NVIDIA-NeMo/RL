"""Mixed-TP cluster smoke — publisher side (one rank).

Runs as a single Megatron-TP rank in a TP=2 world. Two of these are launched
in parallel inside the trainer pod (one per GPU? — here we run both on GPU
0 since the trainer has 1 GPU; multi-GPU CUDA context sharing is fine for
the smoke since each rank's publish buffers are independently registered
with NIXL).

Each rank publishes its own slice of the synthetic ground-truth tensors:
  * column-parallel (axis 0): half the rows
  * row-parallel    (axis 1): half the columns
  * fused QKV       (axis 0): half the heads (each rank holds its share
                              of q/k/v heads interleaved by-head)
  * fused gated-MLP (axis 0): half the intermediate dim of gate AND up
  * replicated     : rank 0 only

The matching receiver runs as target_tp=1 → reads BOTH sources and
concatenates along the right axis, then translates to HF and asserts
byte-identity against the global ground truth.

This validates the mixed-TP receiver path (host-side scratch+slice in
``_update_weights_via_mx_megatron``) for the target-narrower direction
where v0 is already bandwidth-optimal.

Args:
    sys.argv[1] = rank (0 or 1)

Usage:
    /opt/.../bin/python smoke_mixed_tp_publisher.py 0 &
    /opt/.../bin/python smoke_mixed_tp_publisher.py 1 &
"""

from __future__ import annotations

import os
import socket
import sys
import time

import torch

from modelexpress import MxV2TrainingPublisher, TrainerWorldLayout
from modelexpress.megatron_helpers import (
    MegatronTransformerConfig,
    merge_qkv_weights,
    merge_gated_mlp,
)
from modelexpress.nemo_rl_v2 import (
    ROLE_MEGATRON_COLUMN,
    ROLE_MEGATRON_GATED_MLP_COLUMN,
    ROLE_MEGATRON_QKV_COLUMN,
    ROLE_MEGATRON_REPLICATED,
    ROLE_MEGATRON_ROW,
)


SOURCE_TP = 2
GT_PATH = "/mnt/rl-workspace/kavink/smoke-mixed-tp-groundtruth.pt"


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <rank>")
        return 2
    tp_rank = int(sys.argv[1])
    assert 0 <= tp_rank < SOURCE_TP

    # Same seed so both ranks produce IDENTICAL global tensors before
    # slicing. This is the "global ground truth" the receiver compares
    # against.
    torch.manual_seed(2026_06_10)

    # GQA 4:1, sized so TP=2 divides cleanly.
    cfg = MegatronTransformerConfig(
        num_attention_heads=8, num_query_groups=2,
        kv_channels=64, hidden_size=512,
    )
    hidden = cfg.hidden_size
    intermediate = 1024

    # Global ground-truth HF tensors.
    q = torch.randn(cfg.num_attention_heads * cfg.head_size, hidden, dtype=torch.bfloat16)
    k = torch.randn(cfg.num_query_groups * cfg.head_size, hidden, dtype=torch.bfloat16)
    v = torch.randn(cfg.num_query_groups * cfg.head_size, hidden, dtype=torch.bfloat16)
    gate = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    up = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    o_proj = torch.randn(hidden, hidden, dtype=torch.bfloat16)
    dense_col = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    norm_w = torch.randn(hidden, dtype=torch.bfloat16)

    # Megatron-global packed forms.
    qkv_global = merge_qkv_weights(cfg, q, k, v)         # rows = (nh + 2 nkv) * head_size
    gated_global = merge_gated_mlp(gate, up)             # rows = 2 * intermediate

    # Slice each global into this rank's TP shard.
    # qkv: by_head interleave puts q/k/v heads consecutively per chunk;
    # split along axis 0 evenly across TP=2 → each rank gets nh/2 q heads + nkv/2 kv heads
    # interleaved in the same way as the global. The trick: merge_qkv_weights at
    # cfg with HALVED head counts (per-rank cfg) produces the exact same byte
    # layout as slicing the global at the half-row boundary, IFF nh and nkv are
    # both divisible by tp.
    rank_cfg = MegatronTransformerConfig(
        num_attention_heads=cfg.num_attention_heads // SOURCE_TP,
        num_query_groups=cfg.num_query_groups // SOURCE_TP,
        kv_channels=cfg.kv_channels,
        hidden_size=cfg.hidden_size,
    )

    nh_per_rank = cfg.num_attention_heads // SOURCE_TP
    nkv_per_rank = cfg.num_query_groups // SOURCE_TP
    head_size = cfg.head_size

    q_rank = q[tp_rank * nh_per_rank * head_size : (tp_rank + 1) * nh_per_rank * head_size]
    k_rank = k[tp_rank * nkv_per_rank * head_size : (tp_rank + 1) * nkv_per_rank * head_size]
    v_rank = v[tp_rank * nkv_per_rank * head_size : (tp_rank + 1) * nkv_per_rank * head_size]
    qkv_rank = merge_qkv_weights(rank_cfg, q_rank, k_rank, v_rank)

    # Gated MLP: split each of gate and up along axis 0 (per_rank intermediate
    # = intermediate // tp), then merge per-rank.
    inter_per_rank = intermediate // SOURCE_TP
    gate_rank = gate[tp_rank * inter_per_rank : (tp_rank + 1) * inter_per_rank]
    up_rank = up[tp_rank * inter_per_rank : (tp_rank + 1) * inter_per_rank]
    gated_rank = merge_gated_mlp(gate_rank, up_rank)

    # Column-parallel dense_col: split along axis 0.
    dense_col_rank = dense_col[tp_rank * inter_per_rank : (tp_rank + 1) * inter_per_rank]

    # Row-parallel o_proj: split along axis 1.
    h_per_rank = hidden // SOURCE_TP
    o_proj_rank = o_proj[:, tp_rank * h_per_rank : (tp_rank + 1) * h_per_rank].contiguous()

    print(f"[pub-r{tp_rank}] per-rank shapes:")
    print(f"  qkv {tuple(qkv_rank.shape)}  gated {tuple(gated_rank.shape)}")
    print(f"  o_proj_row {tuple(o_proj_rank.shape)}  dense_col {tuple(dense_col_rank.shape)}")
    print(f"  global cfg: nh={cfg.num_attention_heads} nkv={cfg.num_query_groups} "
          f"head_size={cfg.head_size} hidden={cfg.hidden_size}")

    # Persist ground truth ONCE (rank 0 only; rank 1 just waits a beat).
    if tp_rank == 0:
        torch.save({
            "cfg": cfg.to_dict(),
            "q": q, "k": k, "v": v, "gate": gate, "up": up,
            "o_proj": o_proj, "dense_col": dense_col, "norm_w": norm_w,
            "qkv_global": qkv_global, "gated_global": gated_global,
            "source_tp": SOURCE_TP,
        }, GT_PATH)
        print(f"[pub-r0] persisted ground truth → {GT_PATH}")

    # Move shards to GPU for NIXL registration. Both ranks share GPU 0.
    device = torch.device("cuda:0")
    qkv_rank_gpu = qkv_rank.to(device).contiguous()
    gated_rank_gpu = gated_rank.to(device).contiguous()
    o_proj_rank_gpu = o_proj_rank.to(device).contiguous()
    dense_col_rank_gpu = dense_col_rank.to(device).contiguous()
    norm_w_gpu = norm_w.to(device).contiguous()  # replicated; only rank 0 publishes

    pub = MxV2TrainingPublisher(
        agent_name=f"{socket.gethostname()}-mixed-tp-pub-r{tp_rank}",
        device_id=0,
        mx_server_url=os.environ.get(
            "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"
        ),
        worker_rank=tp_rank,
        world_layout=TrainerWorldLayout(
            fsdp_world_size=1, tp_world_size=SOURCE_TP,
            pp_world_size=1, ep_world_size=1,
        ),
    )
    pub.initialize(model_name="smoke/mixed-tp-toy", dtype="bfloat16")
    pub.set_megatron_mesh_position(tp_rank=tp_rank, pp_rank=0, ep_rank=0)

    # Each rank emits the same sidecar (name_map is global).
    pub.set_megatron_sidecar({
        "megatron_transformer_config": cfg.to_dict(),
        "megatron_hf_name_map": [
            ("decoder.layers.0.self_attention.linear_qkv.weight",
             ["model.layers.0.self_attn.q_proj.weight",
              "model.layers.0.self_attn.k_proj.weight",
              "model.layers.0.self_attn.v_proj.weight"]),
            ("decoder.layers.0.mlp.linear_fc1.weight",
             ["model.layers.0.mlp.gate_proj.weight",
              "model.layers.0.mlp.up_proj.weight"]),
            ("decoder.layers.0.self_attention.linear_proj.weight",
             ["model.layers.0.self_attn.o_proj.weight"]),
            ("decoder.layers.0.mlp.linear_fc2.weight",
             ["model.layers.0.mlp.down_proj.weight"]),
            ("decoder.layers.0.input_layernorm.weight",
             ["model.layers.0.input_layernorm.weight"]),
        ],
    })

    pub.add_tensor(
        name="decoder.layers.0.self_attention.linear_qkv.weight",
        tensor=qkv_rank_gpu,
        megatron_role=ROLE_MEGATRON_QKV_COLUMN,
        megatron_extras={
            "qkv_interleave": "by_head",
            "num_heads_local": str(nh_per_rank),
            "num_kv_heads_local": str(nkv_per_rank),
            "head_dim": str(head_size),
        },
    )
    pub.add_tensor(
        name="decoder.layers.0.mlp.linear_fc1.weight",
        tensor=gated_rank_gpu,
        megatron_role=ROLE_MEGATRON_GATED_MLP_COLUMN,
        megatron_extras={"gated_mlp_order": "gate_then_up"},
    )
    pub.add_tensor(
        name="decoder.layers.0.self_attention.linear_proj.weight",
        tensor=o_proj_rank_gpu,
        megatron_role=ROLE_MEGATRON_ROW,
    )
    pub.add_tensor(
        name="decoder.layers.0.mlp.linear_fc2.weight",
        tensor=dense_col_rank_gpu,
        megatron_role=ROLE_MEGATRON_COLUMN,
    )
    # Replicated: both ranks publish so the receiver planner can pick
    # any source. (The production publisher's "rank 0 only" optimization
    # depends on a planner fix that filters to sources whose registry
    # actually contains the tensor — separate work; tracked as a
    # follow-up but not in scope for this mixed-TP smoke.)
    pub.add_tensor(
        name="decoder.layers.0.input_layernorm.weight",
        tensor=norm_w_gpu,
        megatron_role=ROLE_MEGATRON_REPLICATED,
    )

    print(f"[pub-r{tp_rank}] publishing version=1")
    sid = pub.publish(version=1)
    pub.mark_ready()
    print(f"[pub-r{tp_rank}] published source_id={sid}, ready=True")

    hold = 600
    print(f"[pub-r{tp_rank}] sleeping {hold}s")
    time.sleep(hold)
    pub.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
