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
            {},
            prefix="p",
            projection="gate_up_proj",
            tensor=torch.zeros(1, 2, 32),
            expert_ids=[0],
        )
    with pytest.raises(ValueError, match="must be"):
        convert_expert_projection_stack(
            {},
            prefix="p",
            projection="down_proj",
            tensor=torch.zeros(2, 2, 32),
            expert_ids=[0],
        )


def _generation_stub(roles, tps, disagg, ctx_kwargs=None, gen_kwargs=None, base=None):
    stub = SimpleNamespace()
    stub._engine_roles = roles
    stub._engine_tps = tps
    stub._disagg_cfg = disagg
    stub.cfg = {
        "trtllm_cfg": base
        or {"tensor_parallel_size": tps[0], "moe_expert_parallel_size": tps[0]}
    }
    overrides = {"ctx": ctx_kwargs or {}, "gen": gen_kwargs or {}}
    stub._role_kwargs = lambda role: {**stub.cfg["trtllm_cfg"], **overrides[role]}
    stub.get_nccl_reshard_layouts = lambda: TrtllmGeneration.get_nccl_reshard_layouts(
        stub
    )
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
    assert (ctx["tp_size"], ctx["ep_size"], ctx["etp_size"], ctx["world_size"]) == (
        8,
        8,
        1,
        16,
    )
    assert ctx["engine_indices"] == [0, 2] and ctx["rank_prefixes"] == [0, 8]
    assert (gen["tp_size"], gen["ep_size"], gen["etp_size"], gen["world_size"]) == (
        16,
        16,
        1,
        32,
    )
    assert gen["engine_indices"] == [1, 3] and gen["rank_prefixes"] == [0, 16]


def test_reshard_layouts_aggregated_and_expert_tp():
    stub = _generation_stub(["generation"] * 2, [8, 8], {"enabled": False})
    (layout,) = stub.get_nccl_reshard_layouts()
    assert layout["engine_indices"] == [0, 1] and layout["rank_prefixes"] == [0, 8]
    assert layout["world_size"] == 16 and layout["etp_size"] == 1
    stub = _generation_stub(
        ["generation"],
        [8],
        {"enabled": False},
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
    ext.engine = SimpleNamespace(
        model_engine=SimpleNamespace(model=model, model_loader=model_loader)
    )
    return ext, model_loader


def test_local_shard_slices_and_expert_specs(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setenv("NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD", "0")

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    prefix = "model.layers.0.mlp.experts"
    info = _param_info(
        f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2
    )
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
    # post queues the converted stack; the receive loop flushes per stage.
    ext._flush_received_experts()
    model_loader.reload.assert_called_once()
    _model, weights = model_loader.reload.call_args.args[:2]
    assert model_loader.reload.call_args.kwargs == {"allow_partial_loading": True}
    assert set(weights) == {
        f"{prefix}.2.down_proj.weight",
        f"{prefix}.3.down_proj.weight",
    }
    assert torch.equal(weights[f"{prefix}.3.down_proj.weight"], ctx.buf[1])


def test_expert_spec_quantizes_locally(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setenv("NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD", "0")

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: True
    )
    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_mxfp8_model", lambda cfg: True
    )
    prefix = "model.language_model.layers.1.mlp.experts"
    info = _param_info(
        f"{prefix}.gate_proj.weight", (4, 8, 64), "gate_proj", ep_size=2, rank_offset=32
    )
    ext, model_loader = _extension(rank=33, local_ids_by_layer={prefix: [2, 3]})
    spec = ext._build_expert_local_param_map(
        {"layer_names": ["l"], "per_layer_params": {"l": [info]}}
    ).get(info["name"])
    ctx = spec.pre(spec.base)
    ctx.buf.copy_(torch.randn(2, 8, 64, dtype=torch.bfloat16))
    spec.post(ctx)
    ext._flush_received_experts()
    weights = model_loader.reload.call_args.args[1]
    assert set(weights) == {
        f"{prefix}.{e}.gate_proj.{leaf}"
        for e in (2, 3)
        for leaf in ("weight", "weight_scale_inv")
    }
    expected, expected_scale = cast_tensor_to_mxfp8_blockwise(ctx.buf[0])
    assert torch.equal(
        weights[f"{prefix}.2.gate_proj.weight"].float(), expected.float()
    )
    assert torch.equal(
        weights[f"{prefix}.2.gate_proj.weight_scale_inv"], expected_scale
    )


def test_received_experts_reload_in_batches(monkeypatch):
    """Stacks accumulate until the byte budget, then load in one reload call."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setenv("NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD", "0")

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    prefix = "model.layers.0.mlp.experts"
    infos = [
        _param_info(
            f"{prefix}.{proj}.weight", (8, 16, 4), proj, ep_size=4, rank_offset=2
        )
        for proj in ("gate_proj", "up_proj")
    ]
    ext, model_loader = _extension(rank=3, local_ids_by_layer={prefix: [2, 3]})
    param_map = ext._build_expert_local_param_map(
        {"layer_names": ["layer0"], "per_layer_params": {"layer0": infos}}
    )
    # Large budget: both stacks wait for the flush and share one reload.
    monkeypatch.setenv("NRL_TRTLLM_REFIT_RELOAD_BATCH_BYTES", str(1 << 40))
    for info in infos:
        spec = param_map.get(info["name"])
        ctx = spec.pre(spec.base)
        ctx.buf.zero_()
        spec.post(ctx)
    model_loader.reload.assert_not_called()
    ext._flush_received_experts()
    model_loader.reload.assert_called_once()
    weights = model_loader.reload.call_args.args[1]
    assert set(weights) == {
        f"{prefix}.{e}.{proj}.weight"
        for e in (2, 3)
        for proj in ("gate_proj", "up_proj")
    }
    assert ext._refit_stats["bulk_reload_calls"] == 1
    # Budget 0: every stack loads on the spot, as before batching.
    model_loader.reload.reset_mock()
    monkeypatch.setenv("NRL_TRTLLM_REFIT_RELOAD_BATCH_BYTES", "0")
    spec = param_map.get(infos[0]["name"])
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    model_loader.reload.assert_called_once()
    ext._flush_received_experts()
    model_loader.reload.assert_called_once()


def test_received_experts_load_directly_into_moe_modules(monkeypatch):
    """The direct path replays the loader's MoE branch per module, no global reload."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    monkeypatch.setenv("NRL_TRTLLM_REFIT_RELOAD_BATCH_BYTES", str(1 << 40))
    monkeypatch.delenv("NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD", raising=False)
    prefix = "model.layers.0.mlp.experts"
    info = _param_info(
        f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2
    )
    ext, model_loader = _extension(rank=3, local_ids_by_layer={prefix: [2, 3]})
    calls = []
    mapper = MagicMock()
    mapper.preprocess_weights.side_effect = lambda w, allow_partial_loading=False: dict(
        w
    )
    mapper.filter_weights.side_effect = lambda pre, w: {
        k[len(pre) + 1 :]: v for k, v in w.items() if k.startswith(pre + ".")
    }
    mapper.is_special_instance_module.return_value = True
    mapper.handle_special_instance_module.side_effect = (
        lambda module, name, weights, allow_partial_loading=False: calls.append(
            (module, name, sorted(weights), allow_partial_loading)
        )
    )
    model_loader.weight_mapper = mapper
    moe_module = object()
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {prefix: moe_module})
    spec = ext._build_expert_local_param_map(
        {"layer_names": ["layer0"], "per_layer_params": {"layer0": [info]}}
    ).get(info["name"])
    ctx = spec.pre(spec.base)
    ctx.buf.zero_()
    spec.post(ctx)
    ext._flush_received_experts()
    assert calls == [
        (moe_module, "experts", ["2.down_proj.weight", "3.down_proj.weight"], True)
    ]
    model_loader.reload.assert_not_called()
    assert ext._refit_stats["bulk_reload_calls"] == 1
    # A prefix no MoE module owns falls back to the generic loader.
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {})
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    ext._flush_received_experts()
    model_loader.reload.assert_called_once()


def test_expert_spec_rejects_slot_mismatch_and_non_expert_bulk(monkeypatch):
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    prefix = "model.layers.0.mlp.experts"
    info = _param_info(
        f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2
    )
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
        ["context", "generation"],
        [8, 16],
        {"enabled": True},
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
        ["generation"],
        [8],
        {"enabled": False},
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
                    "trtllm_cfg": {
                        "tensor_parallel_size": 8,
                        "moe_expert_parallel_size": ep,
                    },
                },
            }
        )

    check_nccl_reshard_refit_support(config(8))
    with pytest.raises(ValueError, match="pure expert parallelism"):
        check_nccl_reshard_refit_support(config(4))


def _stack_loader_module(supports: bool = True):
    moe = MagicMock()
    moe.supports_expert_stack_loading.return_value = supports
    return moe


def test_received_experts_load_as_whole_stacks(monkeypatch):
    """A MoE module that takes load_expert_stacks gets the received stack as is."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    monkeypatch.delenv("NRL_TRTLLM_REFIT_STACK_LOAD", raising=False)
    prefix = "model.layers.0.mlp.experts"
    down = _param_info(
        f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2
    )
    gate = _param_info(
        f"{prefix}.gate_proj.weight", (8, 4, 16), "gate_proj", ep_size=4, rank_offset=2
    )
    up = _param_info(
        f"{prefix}.up_proj.weight", (8, 4, 16), "up_proj", ep_size=4, rank_offset=2
    )
    gate_up = _param_info(
        f"{prefix}.gate_up_proj.weight",
        (8, 8, 4),
        "gate_up_proj",
        ep_size=4,
        rank_offset=2,
    )
    ext, model_loader = _extension(rank=3, local_ids_by_layer={prefix: [2, 3]})
    moe = _stack_loader_module()
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {prefix: moe})
    param_map = ext._build_expert_local_param_map(
        {
            "layer_names": ["layer0"],
            "per_layer_params": {"layer0": [down, gate, up, gate_up]},
        }
    )

    spec = param_map.get(down["name"])
    ctx = spec.pre(spec.base)
    ctx.buf.copy_(torch.arange(2 * 16 * 4, dtype=torch.bfloat16).view(2, 16, 4))
    spec.post(ctx)
    moe.load_expert_stacks.assert_called_once()
    (ids,), kwargs = moe.load_expert_stacks.call_args
    assert ids == [2, 3] and set(kwargs) == {"w2"}
    assert torch.equal(kwargs["w2"], ctx.buf)

    # The reshard plan's per-projection stacks: gate_proj -> w1, up_proj -> w3.
    for info, key in ((gate, "w1"), (up, "w3")):
        moe.load_expert_stacks.reset_mock()
        spec = param_map.get(info["name"])
        ctx = spec.pre(spec.base)
        ctx.buf.copy_(torch.arange(2 * 4 * 16, dtype=torch.bfloat16).view(2, 4, 16))
        spec.post(ctx)
        (ids,), kwargs = moe.load_expert_stacks.call_args
        assert ids == [2, 3] and set(kwargs) == {key}
        assert torch.equal(kwargs[key], ctx.buf)

    # A fused gate_up stack splits into the gate (w1) and up (w3) halves along dim 1.
    moe.load_expert_stacks.reset_mock()
    spec = param_map.get(gate_up["name"])
    ctx = spec.pre(spec.base)
    ctx.buf.copy_(torch.arange(2 * 8 * 4, dtype=torch.bfloat16).view(2, 8, 4))
    spec.post(ctx)
    (ids,), kwargs = moe.load_expert_stacks.call_args
    assert ids == [2, 3] and set(kwargs) == {"w1", "w3"}
    assert torch.equal(kwargs["w1"], ctx.buf[:, :4])
    assert torch.equal(kwargs["w3"], ctx.buf[:, 4:])

    ext._flush_received_experts()
    model_loader.reload.assert_not_called()
    assert ext._refit_stats["bulk_stack_calls"] == 4
    assert ext._refit_stats["bulk_reload_calls"] == 0


def test_received_experts_stack_load_quantizes_mxfp8(monkeypatch):
    """MXFP8 engines quantize the stack in expert chunks and pass e4m3 + UE8M0."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: True
    )
    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_mxfp8_model", lambda cfg: True
    )
    monkeypatch.setattr(trtllm_backend.fp8_quantization, "FP8_EXPERT_CHUNK_SIZE", 1)
    monkeypatch.delenv("NRL_TRTLLM_REFIT_STACK_LOAD", raising=False)
    prefix = "model.layers.1.mlp.experts"
    info = _param_info(
        f"{prefix}.down_proj.weight", (4, 8, 64), "down_proj", ep_size=2, rank_offset=32
    )
    ext, model_loader = _extension(rank=33, local_ids_by_layer={prefix: [2, 3]})
    moe = _stack_loader_module()
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {prefix: moe})
    spec = ext._build_expert_local_param_map(
        {"layer_names": ["l"], "per_layer_params": {"l": [info]}}
    ).get(info["name"])
    ctx = spec.pre(spec.base)
    ctx.buf.copy_(torch.randn(2, 8, 64, dtype=torch.bfloat16))
    spec.post(ctx)
    assert moe.load_expert_stacks.call_count == 2
    for call, expert_id in zip(moe.load_expert_stacks.call_args_list, (2, 3)):
        (ids,), kwargs = call
        assert ids == [expert_id] and set(kwargs) == {"w2", "w2_scale"}
        expected, expected_scale = cast_tensor_to_mxfp8_blockwise(
            ctx.buf[expert_id - 2 : expert_id - 1]
        )
        assert kwargs["w2"].dtype == torch.float8_e4m3fn
        assert torch.equal(kwargs["w2"].view(torch.uint8), expected.view(torch.uint8))
        assert kwargs["w2_scale"].dtype == torch.uint8
        assert torch.equal(kwargs["w2_scale"], expected_scale)
    model_loader.reload.assert_not_called()
    assert ext._refit_stats["bulk_stack_calls"] == 2


def test_received_experts_stack_load_falls_back(monkeypatch):
    """No stack loader (unsupported method, missing API, or env off) -> per-expert path."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    monkeypatch.setenv("NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD", "0")
    monkeypatch.setenv("NRL_TRTLLM_REFIT_RELOAD_BATCH_BYTES", "0")
    prefix = "model.layers.0.mlp.experts"
    info = _param_info(
        f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2
    )
    ext, model_loader = _extension(rank=3, local_ids_by_layer={prefix: [2, 3]})
    spec = ext._build_expert_local_param_map(
        {"layer_names": ["layer0"], "per_layer_params": {"layer0": [info]}}
    ).get(info["name"])

    # Quant method without a stack path.
    moe = _stack_loader_module(supports=False)
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {prefix: moe})
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    moe.load_expert_stacks.assert_not_called()
    model_loader.reload.assert_called_once()

    # Older TRT-LLM: no load_expert_stacks attribute at all.
    ext._expert_stack_loader_map.clear()
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {prefix: object()})
    model_loader.reload.reset_mock()
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    model_loader.reload.assert_called_once()

    # Knob off.
    ext._expert_stack_loader_map.clear()
    moe = _stack_loader_module()
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {prefix: moe})
    monkeypatch.setenv("NRL_TRTLLM_REFIT_STACK_LOAD", "0")
    model_loader.reload.reset_mock()
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    moe.load_expert_stacks.assert_not_called()
    model_loader.reload.assert_called_once()
    assert ext._refit_stats["bulk_stack_calls"] == 0


def test_received_experts_stack_loader_normalizes_hf_prefix(monkeypatch):
    """Trainer HF prefixes (model.language_model.*) resolve to TRT-LLM's module path."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    monkeypatch.delenv("NRL_TRTLLM_REFIT_STACK_LOAD", raising=False)
    hf_prefix = "model.language_model.layers.1.mlp.experts"
    module_prefix = "model.layers.1.mlp.experts"
    info = _param_info(
        f"{hf_prefix}.down_proj.weight",
        (8, 16, 4),
        "down_proj",
        ep_size=4,
        rank_offset=2,
    )
    ext, model_loader = _extension(rank=3, local_ids_by_layer={hf_prefix: [2, 3]})
    moe = _stack_loader_module()
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {module_prefix: moe})

    # The mapper renames like TRT-LLM's Qwen3.5 mapper does.
    mapper = MagicMock()
    mapper.preprocess_weights.side_effect = lambda w, allow_partial_loading=False: {
        k.replace("model.language_model.", "model."): v for k, v in w.items()
    }
    model_loader.weight_mapper = mapper
    spec = ext._build_expert_local_param_map(
        {"layer_names": ["l"], "per_layer_params": {"l": [info]}}
    ).get(info["name"])
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    moe.load_expert_stacks.assert_called_once()
    model_loader.reload.assert_not_called()

    # Without a usable mapper the generic language_model strip still resolves it.
    ext._expert_stack_loader_map.clear()
    moe.load_expert_stacks.reset_mock()
    model_loader.weight_mapper = None
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    moe.load_expert_stacks.assert_called_once()


def test_received_experts_stack_loader_resolves_by_layer_idx(monkeypatch):
    """When no name matches, the module's layer_idx pairs it with the prefix."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    monkeypatch.delenv("NRL_TRTLLM_REFIT_STACK_LOAD", raising=False)
    hf_prefix = "model.language_model.layers.7.mlp.experts"
    info = _param_info(
        f"{hf_prefix}.down_proj.weight",
        (8, 16, 4),
        "down_proj",
        ep_size=4,
        rank_offset=2,
    )
    ext, model_loader = _extension(rank=3, local_ids_by_layer={hf_prefix: [2, 3]})
    model_loader.weight_mapper = None
    moe = _stack_loader_module()
    moe.layer_idx = 7
    other = _stack_loader_module()
    other.layer_idx = 8
    # Module names TRT-LLM might use that no normalization of the HF prefix produces.
    monkeypatch.setattr(
        ext,
        "_moe_weight_owners",
        lambda: {"llm.blocks.7.moe": moe, "llm.blocks.8.moe": other},
    )
    spec = ext._build_expert_local_param_map(
        {"layer_names": ["l"], "per_layer_params": {"l": [info]}}
    ).get(info["name"])
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    moe.load_expert_stacks.assert_called_once()
    other.load_expert_stacks.assert_not_called()
    model_loader.reload.assert_not_called()
