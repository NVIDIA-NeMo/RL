"""PP=2 cluster smoke — publisher side (one pipeline stage).

Isolates the pipeline-parallel axis: a 4-layer toy model split across
PP=2, stage 0 owns layers [0,1], stage 1 owns layers [2,3]. TP=1 (no
within-layer sharding) so the ONLY thing under test is PP-stage routing:
each stage publishes its layers tagged with pp_rank, and a PP=1 receiver
must union both stages, pulling each layer from the stage that owns it.

This is the least-exercised Megatron axis (all prior validation was
PP=1). Mirrors the mixed-TP smoke's structure but splits by-layer
instead of within-layer.

Usage inside trainer pod:
    python smoke_pp_publisher.py 0    # stage 0 -> layers 0,1
    python smoke_pp_publisher.py 1    # stage 1 -> layers 2,3
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

PP_SIZE = 2
LAYERS_PER_STAGE = 2
N_LAYERS = PP_SIZE * LAYERS_PER_STAGE
GT_PATH = "/mnt/rl-workspace/kavink/smoke-pp-groundtruth.pt"


def _layer_tensors(layer_id: int, cfg, hidden, intermediate):
    """Deterministic per-layer global tensors (seeded by layer id)."""
    g = torch.Generator().manual_seed(2026_07_03 + layer_id)
    hd = cfg.head_size
    q = torch.randn(cfg.num_attention_heads * hd, hidden, generator=g, dtype=torch.float32).bfloat16()
    k = torch.randn(cfg.num_query_groups * hd, hidden, generator=g, dtype=torch.float32).bfloat16()
    v = torch.randn(cfg.num_query_groups * hd, hidden, generator=g, dtype=torch.float32).bfloat16()
    gate = torch.randn(intermediate, hidden, generator=g, dtype=torch.float32).bfloat16()
    up = torch.randn(intermediate, hidden, generator=g, dtype=torch.float32).bfloat16()
    o_proj = torch.randn(hidden, hidden, generator=g, dtype=torch.float32).bfloat16()
    fc2_col = torch.randn(intermediate, hidden, generator=g, dtype=torch.float32).bfloat16()
    norm = torch.randn(hidden, generator=g, dtype=torch.float32).bfloat16()
    return dict(q=q, k=k, v=v, gate=gate, up=up, o_proj=o_proj, fc2_col=fc2_col, norm=norm)


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <pp_rank>")
        return 2
    pp_rank = int(sys.argv[1])
    assert 0 <= pp_rank < PP_SIZE

    cfg = MegatronTransformerConfig(
        num_attention_heads=8, num_query_groups=2, kv_channels=64, hidden_size=512,
    )
    hidden = cfg.hidden_size
    intermediate = 1024
    device = torch.device("cuda:0")

    my_layers = list(range(pp_rank * LAYERS_PER_STAGE, (pp_rank + 1) * LAYERS_PER_STAGE))
    print(f"[pp-pub-s{pp_rank}] owns layers {my_layers}")

    # Persist full ground truth (all layers) once, from stage 0.
    if pp_rank == 0:
        gt = {"cfg": cfg.to_dict(), "n_layers": N_LAYERS, "layers": {}}
        for L in range(N_LAYERS):
            t = _layer_tensors(L, cfg, hidden, intermediate)
            gt["layers"][L] = {k: v for k, v in t.items()}
        torch.save(gt, GT_PATH)
        print(f"[pp-pub-s0] persisted GT ({N_LAYERS} layers) -> {GT_PATH}")

    pub = MxV2TrainingPublisher(
        agent_name=f"{socket.gethostname()}-pp-pub-s{pp_rank}",
        device_id=0,
        mx_server_url=os.environ.get(
            "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"
        ),
        worker_rank=pp_rank,
        world_layout=TrainerWorldLayout(
            fsdp_world_size=1, tp_world_size=1,
            pp_world_size=PP_SIZE, ep_world_size=1,
        ),
    )
    pub.initialize(model_name="smoke/pp-toy", dtype="bfloat16")
    pub.set_megatron_mesh_position(tp_rank=0, pp_rank=pp_rank, ep_rank=0)

    # Sidecar name_map is GLOBAL metadata (Bridge derives it from the
    # model config), so every stage emits the full map for all layers —
    # the receiver reads one stage's sidecar and gets every layer's names.
    name_map = []
    for L in range(N_LAYERS):
        name_map += [
            (f"decoder.layers.{L}.self_attention.linear_qkv.weight",
             [f"model.layers.{L}.self_attn.q_proj.weight",
              f"model.layers.{L}.self_attn.k_proj.weight",
              f"model.layers.{L}.self_attn.v_proj.weight"]),
            (f"decoder.layers.{L}.mlp.linear_fc1.weight",
             [f"model.layers.{L}.mlp.gate_proj.weight",
              f"model.layers.{L}.mlp.up_proj.weight"]),
            (f"decoder.layers.{L}.self_attention.linear_proj.weight",
             [f"model.layers.{L}.self_attn.o_proj.weight"]),
            (f"decoder.layers.{L}.mlp.linear_fc2.weight",
             [f"model.layers.{L}.mlp.down_proj.weight"]),
            (f"decoder.layers.{L}.input_layernorm.weight",
             [f"model.layers.{L}.input_layernorm.weight"]),
        ]
    pub.set_megatron_sidecar({
        "megatron_transformer_config": cfg.to_dict(),
        "megatron_hf_name_map": name_map,
    })

    for L in my_layers:
        t = _layer_tensors(L, cfg, hidden, intermediate)
        qkv = merge_qkv_weights(cfg, t["q"], t["k"], t["v"]).to(device).contiguous()
        gated = merge_gated_mlp(t["gate"], t["up"]).to(device).contiguous()
        pub.add_tensor(
            name=f"decoder.layers.{L}.self_attention.linear_qkv.weight",
            tensor=qkv, megatron_role=ROLE_MEGATRON_QKV_COLUMN,
            megatron_extras={
                "qkv_interleave": "by_head",
                "num_heads_local": str(cfg.num_attention_heads),
                "num_kv_heads_local": str(cfg.num_query_groups),
                "head_dim": str(cfg.head_size),
            },
        )
        pub.add_tensor(
            name=f"decoder.layers.{L}.mlp.linear_fc1.weight",
            tensor=gated, megatron_role=ROLE_MEGATRON_GATED_MLP_COLUMN,
            megatron_extras={"gated_mlp_order": "gate_then_up"},
        )
        pub.add_tensor(
            name=f"decoder.layers.{L}.self_attention.linear_proj.weight",
            tensor=t["o_proj"].to(device).contiguous(), megatron_role=ROLE_MEGATRON_ROW,
        )
        pub.add_tensor(
            name=f"decoder.layers.{L}.mlp.linear_fc2.weight",
            tensor=t["fc2_col"].to(device).contiguous(), megatron_role=ROLE_MEGATRON_COLUMN,
        )
        pub.add_tensor(
            name=f"decoder.layers.{L}.input_layernorm.weight",
            tensor=t["norm"].to(device).contiguous(), megatron_role=ROLE_MEGATRON_REPLICATED,
        )

    print(f"[pp-pub-s{pp_rank}] publishing version=1 ({len(my_layers)} layers)")
    sid = pub.publish(version=1)
    pub.mark_ready()
    print(f"[pp-pub-s{pp_rank}] published source_id={sid}, ready=True")

    hold = 600
    print(f"[pp-pub-s{pp_rank}] sleeping {hold}s")
    time.sleep(hold)
    pub.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
