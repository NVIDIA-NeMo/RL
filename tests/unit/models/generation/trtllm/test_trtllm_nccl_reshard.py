"""Unit tests for the TRT-LLM shard-to-shard (nccl_reshard) refit plumbing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.distributed._tensor import Replicate, Shard

from nemo_rl.models.generation.trtllm.quantization.fp8 import (
    cast_tensor_to_mxfp8_blockwise,
    convert_expert_projection_stack,
    load_weights,
)
from nemo_rl.models.generation.trtllm.trtllm_backend import NcclExtension
from nemo_rl.models.generation.trtllm.trtllm_generation import TrtllmGeneration
from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo

pytestmark = pytest.mark.trtllm


def test_projection_stack_matches_fused_conversion():
    prefix = "model.layers.5.mlp.experts"
    num_experts, inter, hidden = 6, 64, 128
    gate_up = torch.randn(num_experts, 2 * inter, hidden, dtype=torch.bfloat16)
    fused = load_weights([(f"{prefix}.gate_up_proj", gate_up)], is_mx=True)
    stacked: dict = {}
    convert_expert_projection_stack(
        stacked,
        prefix=prefix,
        projection="up_proj",
        tensor=gate_up[2:5, inter:, :],
        expert_ids=[2, 3, 4],
        is_mx=True,
    )
    assert set(stacked) == {
        f"{prefix}.{e}.up_proj.{leaf}"
        for e in (2, 3, 4)
        for leaf in ("weight", "weight_scale_inv")
    }
    for name, tensor in stacked.items():
        assert torch.equal(tensor.float(), fused[name].float())
    weight, scale = cast_tensor_to_mxfp8_blockwise(gate_up[3, inter:, :])
    assert torch.equal(stacked[f"{prefix}.3.up_proj.weight"].float(), weight.float())
    assert torch.equal(stacked[f"{prefix}.3.up_proj.weight_scale_inv"], scale)


def test_projection_stack_rejects_bad_shapes():
    with pytest.raises(ValueError, match="Unsupported"):
        convert_expert_projection_stack(
            {}, prefix="p", projection="gate_up_proj", tensor=torch.zeros(1, 2, 32), expert_ids=[0]
        )
    with pytest.raises(ValueError, match="must be"):
        convert_expert_projection_stack(
            {}, prefix="p", projection="down_proj", tensor=torch.zeros(2, 2, 32), expert_ids=[0]
        )


def _generation_stub(roles, tps, disagg, ctx_kwargs=None, gen_kwargs=None, base=None):
    stub = SimpleNamespace()
    stub._engine_roles = roles
    stub._engine_tps = tps
    stub._disagg_cfg = disagg
    stub.cfg = {"trtllm_cfg": base or {"tensor_parallel_size": tps[0], "moe_expert_parallel_size": tps[0]}}
    overrides = {"ctx": ctx_kwargs or {}, "gen": gen_kwargs or {}}
    stub._role_kwargs = lambda role: {**stub.cfg["trtllm_cfg"], **overrides[role]}
    stub.get_nccl_reshard_layouts = lambda: TrtllmGeneration.get_nccl_reshard_layouts(stub)
    return stub


def test_reshard_layouts_disaggregated():
    stub = _generation_stub(
        ["context", "generation", "context", "generation"],
        [8, 16, 8, 16],
        {"enabled": True},
        ctx_kwargs={"tensor_parallel_size": 8, "moe_expert_parallel_size": 8},
        gen_kwargs={"tensor_parallel_size": 16, "moe_expert_parallel_size": 16},
    )
    layouts = stub.get_nccl_reshard_layouts()
    assert [layout["role"] for layout in layouts] == ["context", "generation"]
    ctx, gen = layouts
    assert (ctx["tp_size"], ctx["ep_size"], ctx["etp_size"], ctx["world_size"]) == (8, 8, 1, 16)
    assert ctx["engine_indices"] == [0, 2] and ctx["rank_prefixes"] == [0, 8]
    assert (gen["tp_size"], gen["ep_size"], gen["etp_size"], gen["world_size"]) == (16, 16, 1, 32)
    assert gen["engine_indices"] == [1, 3] and gen["rank_prefixes"] == [0, 16]


def test_reshard_layouts_aggregated_and_expert_tp():
    stub = _generation_stub(["generation"] * 2, [8, 8], {"enabled": False})
    (layout,) = stub.get_nccl_reshard_layouts()
    assert layout["engine_indices"] == [0, 1] and layout["rank_prefixes"] == [0, 8]
    assert layout["world_size"] == 16 and layout["etp_size"] == 1
    stub = _generation_stub(
        ["generation"], [8], {"enabled": False},
        base={"tensor_parallel_size": 8, "moe_expert_parallel_size": 4},
    )
    (layout,) = stub.get_nccl_reshard_layouts()
    assert layout["etp_size"] == 2


def _param_info(name, global_shape, proj, ep_size, rank_offset, dtype="torch.bfloat16"):
    mesh = MeshInfo(torch.arange(rank_offset, rank_offset + ep_size).view(1, ep_size))
    return {
        "name": name,
        "global_shape": tuple(global_shape),
        "dtype": dtype,
        "src_mesh_info": mesh,
        "src_placements": [Replicate(), Shard(0)],
        "dst_mesh_info": mesh,
        "dst_placements": [Replicate(), Shard(0)],
        "grouped_expert_proj": proj,
    }


def _extension(rank, local_ids_by_layer, quantized=False, device="cpu"):
    ext = NcclExtension.__new__(NcclExtension)
    ext.device_id = 0
    ext.pp_comm_groups = {0: SimpleNamespace(rank=rank)}
    ext._local_expert_lookup = lambda prefix: local_ids_by_layer.get(prefix)
    quant_config = SimpleNamespace()
    model = SimpleNamespace(model_config=SimpleNamespace(quant_config=quant_config))
    model_loader = MagicMock()
    ext.engine = SimpleNamespace(model_engine=SimpleNamespace(model=model, model_loader=model_loader))
    return ext, model_loader


def test_local_shard_slices_and_expert_specs(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    prefix = "model.layers.0.mlp.experts"
    info = _param_info(f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2)
    # rank 3 in the group = mesh coordinate 1 -> experts 2..3
    ext, model_loader = _extension(rank=3, local_ids_by_layer={prefix: [2, 3]})
    slices = NcclExtension._local_shard_slices(info, 3)
    assert slices[0] == slice(2, 4)
    param_map = ext._build_expert_local_param_map(
        {"layer_names": ["layer0"], "per_layer_params": {"layer0": [info]}}
    )
    spec = param_map.get(info["name"])
    assert spec is not None
    ctx = spec.pre(spec.base)
    assert tuple(ctx.buf.shape) == (2, 16, 4) and ctx.buf.dtype == torch.bfloat16
    ctx.buf.copy_(torch.arange(2 * 16 * 4, dtype=torch.bfloat16).view(2, 16, 4))
    spec.post(ctx)
    model_loader.reload.assert_called_once()
    _model, weights = model_loader.reload.call_args.args[:2]
    assert model_loader.reload.call_args.kwargs == {"allow_partial_loading": True}
    assert set(weights) == {f"{prefix}.2.down_proj.weight", f"{prefix}.3.down_proj.weight"}
    assert torch.equal(weights[f"{prefix}.3.down_proj.weight"], ctx.buf[1])


def test_expert_spec_quantizes_locally(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: True
    )
    monkeypatch.setattr(trtllm_backend.fp8_quantization, "is_mxfp8_model", lambda cfg: True)
    prefix = "model.language_model.layers.1.mlp.experts"
    info = _param_info(f"{prefix}.gate_proj.weight", (4, 8, 64), "gate_proj", ep_size=2, rank_offset=32)
    ext, model_loader = _extension(rank=33, local_ids_by_layer={prefix: [2, 3]})
    spec = ext._build_expert_local_param_map(
        {"layer_names": ["l"], "per_layer_params": {"l": [info]}}
    ).get(info["name"])
    ctx = spec.pre(spec.base)
    ctx.buf.copy_(torch.randn(2, 8, 64, dtype=torch.bfloat16))
    spec.post(ctx)
    weights = model_loader.reload.call_args.args[1]
    assert set(weights) == {
        f"{prefix}.{e}.gate_proj.{leaf}" for e in (2, 3) for leaf in ("weight", "weight_scale_inv")
    }
    expected, expected_scale = cast_tensor_to_mxfp8_blockwise(ctx.buf[0])
    assert torch.equal(weights[f"{prefix}.2.gate_proj.weight"].float(), expected.float())
    assert torch.equal(weights[f"{prefix}.2.gate_proj.weight_scale_inv"], expected_scale)


def test_expert_spec_rejects_slot_mismatch_and_non_expert_bulk(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    prefix = "model.layers.0.mlp.experts"
    info = _param_info(f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2)
    ext, _ = _extension(rank=3, local_ids_by_layer={prefix: [4, 5]})
    with pytest.raises(RuntimeError, match="differs from TRT-LLM"):
        ext._build_expert_local_param_map(
            {"layer_names": ["l"], "per_layer_params": {"l": [info]}}
        )
    ext, _ = _extension(rank=3, local_ids_by_layer={})
    with pytest.raises(RuntimeError, match="no MoE module"):
        ext._build_expert_local_param_map(
            {"layer_names": ["l"], "per_layer_params": {"l": [info]}}
        )
    dense = dict(info, name="model.layers.0.mlp.gate_proj.weight")
    dense.pop("grouped_expert_proj")
    ext, _ = _extension(rank=3, local_ids_by_layer={prefix: [2, 3]})
    with pytest.raises(NotImplementedError, match="grouped routed experts only"):
        ext._build_expert_local_param_map(
            {"layer_names": ["l"], "per_layer_params": {"l": [dense]}}
        )


def test_synchronizer_layouts_for_trtllm():
    from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
        NcclReshardWeightSynchronizer,
    )

    sync = NcclReshardWeightSynchronizer.__new__(NcclReshardWeightSynchronizer)
    sync._policy = SimpleNamespace(cfg={"generation": {"backend": "trtllm"}})
    sync._generation = _generation_stub(
        ["context", "generation"], [8, 16], {"enabled": True},
        ctx_kwargs={"tensor_parallel_size": 8, "moe_expert_parallel_size": 8},
        gen_kwargs={"tensor_parallel_size": 16, "moe_expert_parallel_size": 16},
    )
    layouts = sync._gen_layouts()
    assert [(l["role"], l["ep_size"], l["world_size"]) for l in layouts] == [
        ("context", 8, 8),
        ("generation", 16, 16),
    ]
    with pytest.raises(ValueError, match="several layouts"):
        sync._gen_parallelism()
    sync._generation = _generation_stub(
        ["generation"], [8], {"enabled": False},
        base={"tensor_parallel_size": 8, "moe_expert_parallel_size": 4},
    )
    with pytest.raises(ValueError, match="pure expert parallelism"):
        sync._gen_layouts()


def test_support_check_accepts_pure_ep_trtllm():
    from nemo_rl.weight_sync.nccl_reshard_utils import check_nccl_reshard_refit_support

    def config(ep):
        return SimpleNamespace(
            policy={
                "precision": "bfloat16",
                "megatron_cfg": {"enabled": True, "expert_tensor_parallel_size": 1},
                "dtensor_cfg": {"enabled": False},
                "generation": {
                    "backend": "trtllm",
                    "colocated": {"enabled": False},
                    "trtllm_cfg": {"tensor_parallel_size": 8, "moe_expert_parallel_size": ep},
                },
            }
        )

    check_nccl_reshard_refit_support(config(8))
    with pytest.raises(ValueError, match="pure expert parallelism"):
        check_nccl_reshard_refit_support(config(4))
