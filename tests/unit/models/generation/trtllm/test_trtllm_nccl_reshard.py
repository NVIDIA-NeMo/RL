# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the TRT-LLM shard-to-shard (nccl_reshard) refit plumbing."""

# trtllm_backend imports tensorrt_llm eagerly, so import it only inside tests.

import contextlib
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
from nemo_rl.models.generation.trtllm.trtllm_generation import TrtllmGeneration
from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo
from nemo_rl.weight_sync.xferdtensor_python import local_shard_slices

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


def _generation_stub(
    roles, tps, disagg, prefill_kwargs=None, decode_kwargs=None, base=None
):
    stub = SimpleNamespace()
    stub._engine_roles = roles
    stub._engine_tps = tps
    stub._disagg_cfg = SimpleNamespace(**disagg)
    trtllm_cfg = dict(
        base or {"tensor_parallel_size": tps[0], "moe_expert_parallel_size": tps[0]}
    )
    if disagg["enabled"]:
        # Role-resolved config, as role_engine_config expects.
        trtllm_cfg["prefill_engine"] = {**trtllm_cfg, **(prefill_kwargs or {})}
        trtllm_cfg["decode_engine"] = {**trtllm_cfg, **(decode_kwargs or {})}
    stub.cfg = {"trtllm_cfg": trtllm_cfg}
    stub.get_nccl_reshard_layouts = lambda: TrtllmGeneration.get_nccl_reshard_layouts(
        stub
    )
    return stub


def test_reshard_layouts_disaggregated():
    stub = _generation_stub(
        ["prefill", "decode", "prefill", "decode"],
        [8, 16, 8, 16],
        {"enabled": True},
        prefill_kwargs={"tensor_parallel_size": 8, "moe_expert_parallel_size": 8},
        decode_kwargs={"tensor_parallel_size": 16, "moe_expert_parallel_size": 16},
    )
    layouts = stub.get_nccl_reshard_layouts()
    assert [layout["role"] for layout in layouts] == ["prefill", "decode"]
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
    stub = _generation_stub(["aggregated"] * 2, [8, 8], {"enabled": False})
    (layout,) = stub.get_nccl_reshard_layouts()
    assert layout["engine_indices"] == [0, 1] and layout["rank_prefixes"] == [0, 8]
    assert layout["world_size"] == 16 and layout["etp_size"] == 1
    stub = _generation_stub(
        ["aggregated"],
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
    from nemo_rl.models.generation.trtllm.trtllm_backend import NcclExtension

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
    slices = local_shard_slices(info, 3)
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
    ext._reload_batch_bytes = 1 << 40
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
    # Budget 0: every stack loads on the spot, as before batching.
    model_loader.reload.reset_mock()
    ext._reload_batch_bytes = 0
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
    monkeypatch.delenv("NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD", raising=False)
    prefix = "model.layers.0.mlp.experts"
    info = _param_info(
        f"{prefix}.down_proj.weight", (8, 16, 4), "down_proj", ep_size=4, rank_offset=2
    )
    ext, model_loader = _extension(rank=3, local_ids_by_layer={prefix: [2, 3]})
    ext._reload_batch_bytes = 1 << 40
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
    # Every queued weight is a routed expert, so a prefix no MoE module owns
    # means the names drifted: fail loudly instead of a silent generic reload.
    monkeypatch.setattr(ext, "_moe_weight_owners", lambda: {})
    ctx = spec.pre(spec.base)
    spec.post(ctx)
    with pytest.raises(RuntimeError, match="expected one TRT-LLM MoE weight owner"):
        ext._flush_received_experts()
    model_loader.reload.assert_not_called()


def test_moe_weight_owner_matches_prefixes_under_a_vlm_wrapper(monkeypatch):
    """Owners are named from the engine root; VLM wrappers nest the decoder under ``llm.``."""
    from tensorrt_llm._torch.moe.fused_moe import weight_owner

    layer0, layer1, mtp0 = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    owners = (layer0, layer1, mtp0)
    # The engine model is the Qwen3.5 VLM wrapper; on the first refit the weight
    # mapper is still bound to it, so the module tree is the only reliable name source.
    wrapper = SimpleNamespace(
        named_modules=lambda: iter(
            [
                ("llm.model.layers.0.mlp", SimpleNamespace()),
                ("llm.model.layers.0.mlp.experts.backend", layer0),
                ("llm.model.layers.1.mlp.experts", layer1),
                ("llm.mtp.layers.0.mlp.experts", mtp0),
            ]
        )
    )
    ext, _ = _extension(rank=0, local_ids_by_layer={})
    ext.engine.model_engine.model = wrapper
    ext._moe_weight_owner_map = None
    monkeypatch.setattr(
        weight_owner, "is_moe_weight_owner", lambda module: module in owners
    )
    assert ext._moe_weight_owner("model.layers.0.mlp.experts") is layer0
    assert ext._moe_weight_owner("model.layers.1.mlp.experts") is layer1
    assert ext._moe_weight_owner("mtp.layers.0.mlp.experts") is mtp0
    with pytest.raises(RuntimeError, match="found 0"):
        ext._moe_weight_owner("model.layers.2.mlp.experts")

    # Two owners ending in the same prefix are ambiguous: fail, never guess.
    ext._moe_weight_owner_map = {
        "a.model.layers.0.mlp.experts": layer0,
        "b.model.layers.0.mlp.experts": layer1,
    }
    with pytest.raises(RuntimeError, match="found 2"):
        ext._moe_weight_owner("model.layers.0.mlp.experts")


@pytest.mark.parametrize(
    "knob", ["NRL_TRTLLM_REFIT_RELOAD_BATCH_BYTES", "NRL_REFIT_NUM_STREAMS"]
)
def test_malformed_refit_knob_fails_at_prepare(monkeypatch, knob):
    """A bad knob must fail at setup, not mid-receive (which hangs the trainers)."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    monkeypatch.setenv(knob, "2G")
    restore = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.weight_sync.nccl_reshard_utils.restore_refit_info_placements",
        restore,
    )
    ext, _ = _extension(rank=0, local_ids_by_layer={})
    with pytest.raises(ValueError, match="2G"):
        trtllm_backend.NcclExtension.prepare_nccl_reshard_refit_info(
            ext, {"layer_names": []}
        )
    restore.assert_not_called()


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
        ["prefill", "decode"],
        [8, 16],
        {"enabled": True},
        prefill_kwargs={"tensor_parallel_size": 8, "moe_expert_parallel_size": 8},
        decode_kwargs={"tensor_parallel_size": 16, "moe_expert_parallel_size": 16},
    )
    layouts = sync._gen_layouts()
    assert [(l["role"], l["ep_size"], l["world_size"]) for l in layouts] == [
        ("prefill", 8, 8),
        ("decode", 16, 16),
    ]
    sync._generation = _generation_stub(
        ["aggregated"],
        [8],
        {"enabled": False},
        base={"tensor_parallel_size": 8, "moe_expert_parallel_size": 4},
    )
    with pytest.raises(ValueError, match="pure expert parallelism"):
        sync._gen_layouts()


def test_support_check_accepts_pure_ep_trtllm():
    from nemo_rl.weight_sync.nccl_reshard_utils import check_nccl_reshard_refit_support

    def config(ep, disagg=None):
        generation = {
            "backend": "trtllm",
            "colocated": {"enabled": False},
            "trtllm_cfg": {"tensor_parallel_size": 8, "moe_expert_parallel_size": ep},
        }
        if disagg is not None:
            generation["disaggregation"] = {"enabled": True}
            generation["trtllm_cfg"].update(disagg)
        return SimpleNamespace(
            policy={
                "precision": "bfloat16",
                "megatron_cfg": {"enabled": True, "expert_tensor_parallel_size": 1},
                "dtensor_cfg": {"enabled": False},
                "generation": generation,
            }
        )

    check_nccl_reshard_refit_support(config(8))
    with pytest.raises(ValueError, match="pure expert parallelism"):
        check_nccl_reshard_refit_support(config(4))

    # Trainer ETP>1 into TRT-LLM is untested, so setup rejects it.
    etp2 = config(8)
    etp2.policy["megatron_cfg"]["expert_tensor_parallel_size"] = 2
    with pytest.raises(ValueError, match="expert_tensor_parallel_size must be 1"):
        check_nccl_reshard_refit_support(etp2)

    # Disaggregated: each role is checked over the shared trtllm_cfg.
    pure_ep_decode = {
        "decode_engine": {"tensor_parallel_size": 16, "moe_expert_parallel_size": 16}
    }
    check_nccl_reshard_refit_support(config(8, pure_ep_decode))
    with pytest.raises(ValueError, match=r"trtllm_cfg\.decode_engine.*pure expert"):
        check_nccl_reshard_refit_support(
            config(8, {"decode_engine": {"tensor_parallel_size": 16}})
        )


def test_bulk_param_check_rejects_dense_mlp_for_trtllm():
    from nemo_rl.weight_sync.nccl_reshard_utils import (
        check_trtllm_nccl_reshard_bulk_params,
    )

    expert = {
        "name": "model.layers.1.mlp.experts.gate_proj.weight",
        "grouped_expert_proj": "gate_proj",
    }
    dense = {"name": "model.layers.0.mlp.gate_proj.weight"}
    check_trtllm_nccl_reshard_bulk_params(
        {"layer_names": ["l1"], "per_layer_params": {"l1": [expert]}}
    )
    with pytest.raises(ValueError, match="dense MLP layers"):
        check_trtllm_nccl_reshard_bulk_params(
            {
                "layer_names": ["l0", "l1"],
                "per_layer_params": {"l0": [dense], "l1": [expert]},
            }
        )


@pytest.mark.parametrize("fails", [False, True])
def test_nccl_reshard_refit_runs_weight_update_lifecycle(monkeypatch, fails):
    """Bulk, then misc, inside begin/finish; a failure aborts the update."""
    from nemo_rl.models.generation.trtllm import trtllm_backend

    ext = trtllm_backend.NcclExtension.__new__(trtllm_backend.NcclExtension)
    model_loader = MagicMock()
    engine = MagicMock()
    engine.model_engine = SimpleNamespace(model=MagicMock(), model_loader=model_loader)
    engine.control_action.side_effect = lambda **_: contextlib.nullcontext()
    ext.engine = engine
    ext.nccl_reshard_refit_info = {"layer_names": [], "per_layer_params": {}}
    ext.pp_comm_groups = {0: object()}
    ext.hf_to_local_param_map = object()
    calls = []
    ext.begin_weight_update = lambda: calls.append("begin")
    ext._recv_bulk_params = lambda: calls.append("bulk")

    def receive_misc():
        calls.append("misc")
        if fails:
            raise RuntimeError("misc failed")

    ext._receive_and_load_misc_params = receive_misc
    ext._finish_refit = lambda recompute_kv: calls.append(("finish", recompute_kv))
    monkeypatch.setattr(
        trtllm_backend.torch.cuda, "empty_cache", lambda: calls.append("empty_cache")
    )

    if fails:
        with pytest.raises(RuntimeError, match="misc failed"):
            ext.nccl_reshard_refit(drain=False, recompute_kv=True)
        assert calls == ["begin", "bulk", "empty_cache", "misc"]
        model_loader.abort_update_weights.assert_called_once_with()
    else:
        assert ext.nccl_reshard_refit(drain=False, recompute_kv=True) is True
        assert calls == ["begin", "bulk", "empty_cache", "misc", ("finish", True)]
        model_loader.abort_update_weights.assert_not_called()
    engine.control_action.assert_called_once_with(drain=False)


def test_prepare_nccl_reshard_refit_info_keeps_misc_meta_and_needs_ep_slots(
    monkeypatch,
):
    from nemo_rl.models.generation.trtllm import trtllm_backend

    ext, _ = _extension(rank=0, local_ids_by_layer={})
    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "is_quantized_expert_refit", lambda cfg: False
    )
    monkeypatch.setattr(
        trtllm_backend.fp8_quantization, "build_local_expert_lookup", lambda m: None
    )
    with pytest.raises(RuntimeError, match="initial_local_expert_ids"):
        ext.prepare_nccl_reshard_refit_info({"layer_names": [], "per_layer_params": {}})

    monkeypatch.setattr(
        trtllm_backend.fp8_quantization,
        "build_local_expert_lookup",
        lambda m: lambda prefix: None,
    )
    ext._build_expert_local_param_map = MagicMock()
    # Before init_nccl_reshard_comm_group: the slot map waits for the groups.
    ext.pp_comm_groups = None
    ext.prepare_nccl_reshard_refit_info(
        {
            "layer_names": [],
            "per_layer_params": {},
            "misc_meta": {
                "model.norm.weight": {"shape": [4], "dtype": "torch.bfloat16"},
                "lm_head.weight": {"shape": [2, 4], "dtype": "torch.float32"},
            },
        }
    )
    assert ext.misc_state_dict_info == {
        "model.norm.weight": (torch.Size([4]), torch.bfloat16),
        "lm_head.weight": (torch.Size([2, 4]), torch.float32),
    }
    ext._build_expert_local_param_map.assert_not_called()
    ext.pp_comm_groups = {0: SimpleNamespace(rank=0)}
    ext.prepare_nccl_reshard_refit_info({"layer_names": [], "per_layer_params": {}})
    ext._build_expert_local_param_map.assert_called_once_with(
        ext.nccl_reshard_refit_info
    )
    assert ext.hf_to_local_param_map is ext._build_expert_local_param_map.return_value


def test_synchronizer_builds_one_bulk_group_and_plan_per_trtllm_layout(monkeypatch):
    from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
        NcclReshardWeightSynchronizer,
    )

    monkeypatch.setattr("ray.get", lambda futures: futures)

    def build(plan, world_size=8 + 24):
        calls = []
        generation = _generation_stub(
            ["prefill", "decode"],
            [8, 16],
            {"enabled": True},
            prefill_kwargs={"tensor_parallel_size": 8, "moe_expert_parallel_size": 8},
            decode_kwargs={"tensor_parallel_size": 16, "moe_expert_parallel_size": 16},
        )
        generation.set_refit_membership = lambda membership: None
        generation.get_collective_sender_spec = lambda: SimpleNamespace(
            nccl_peer="nemo"
        )
        generation.rebuild_collective = lambda *a, **k: []
        generation.rebuild_nccl_reshard_comm_group = lambda membership, **k: (
            calls.append(("gen_group", k["layout_index"], k["sub_world_size"])) or []
        )
        generation.prepare_nccl_reshard_refit_info = lambda plans: calls.append(
            ("gen_plans", len(plans))
        )
        policy = SimpleNamespace(
            cfg={
                "megatron_cfg": {
                    "tensor_model_parallel_size": 2,
                    "expert_model_parallel_size": 4,
                    "expert_tensor_parallel_size": 1,
                },
                "generation": {"backend": "trtllm"},
            },
            init_collective=lambda *a, **k: [],
            init_nccl_reshard_comm_group=lambda **k: (
                calls.append(("train_group", k["layout_index"], k["sub_world_size"]))
                or []
            ),
            prepare_nccl_reshard_refit_info=lambda tp, gp, tws, gws, **k: (
                calls.append(("train_plan", k["layout_index"], gp["ep_size"], gws))
                or dict(plan)
            ),
        )
        ports = iter(range(9001, 9100))
        sync = NcclReshardWeightSynchronizer(
            policy=policy,
            generation=generation,
            train_cluster=SimpleNamespace(
                num_gpus_per_node=4,
                get_master_address_and_port=lambda: ("10.0.0.1", next(ports)),
                get_available_address_and_port=lambda pg_idx, bundle_idx: (
                    "10.0.0.1",
                    next(ports),
                ),
            ),
            inference_cluster=None,
        )
        sync._build(SimpleNamespace(train_world_size=8, world_size=world_size))
        return calls

    empty = {"layer_names": [], "per_layer_params": {}, "misc_meta": {}}
    # Each layout gets its own bulk group (8 train + its ranks) and its own plan.
    assert build(empty) == [
        ("train_group", 0, 8 + 8),
        ("gen_group", 0, 8 + 8),
        ("train_group", 1, 8 + 16),
        ("gen_group", 1, 8 + 16),
        ("train_plan", 0, 8, 8),
        ("train_plan", 1, 16, 16),
        ("gen_plans", 2),
    ]
    dense = {
        "layer_names": ["l0"],
        "per_layer_params": {"l0": [{"name": "model.layers.0.mlp.gate_proj.weight"}]},
        "misc_meta": {},
    }
    with pytest.raises(ValueError, match="dense MLP layers"):
        build(dense)
    with pytest.raises(RuntimeError, match="cover 24 ranks but the inference world"):
        build(empty, world_size=8 + 20)
