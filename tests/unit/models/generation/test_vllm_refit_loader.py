# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for destination-local vLLM refit loading."""

from types import SimpleNamespace

import pytest
import torch


class _FakeUnquantizedMethod:
    def __init__(self, backend_name="TRITON"):
        self.unquantized_backend = SimpleNamespace(name=backend_name)


class _FakeExpertOwner:
    def __init__(
        self,
        *,
        use_ep,
        expert_map=None,
        tp_rank=0,
        tp_size=1,
        backend_name="TRITON",
    ):
        self.use_ep = use_ep
        self._expert_map = expert_map
        self.global_num_experts = 4
        self.moe_config = SimpleNamespace(
            tp_rank=tp_rank,
            tp_size=tp_size,
            num_logical_experts=4,
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
        )
        self.quant_config = None
        self.quant_method = _FakeUnquantizedMethod(backend_name)

    def weight_loader(self, *args, **kwargs):
        raise AssertionError("The fake weight loader should not be called")

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        if self._expert_map is None:
            return expert_id
        return int(self._expert_map[expert_id])


def _expert_param(shape, owner):
    param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
    param.weight_loader = owner.weight_loader
    return param


@pytest.mark.vllm
def test_destination_local_copy_matches_vllm_full_weight_tp_loading():
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    owner = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(owner)
    # vLLM 0.25's _load_w13/_load_w2 take tp_rank as an argument and read the
    # TP size from moe_config.moe_parallel_config to slice the checkpoint
    # weight per rank.
    owner.moe_config = SimpleNamespace(
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )
    reference_w13 = _expert_param((2, 8, 6), owner)
    reference_w2 = _expert_param((2, 6, 4), owner)
    destination_w13 = _expert_param((2, 8, 6), owner)
    destination_w2 = _expert_param((2, 6, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    full_w1 = torch.arange(64, dtype=torch.float32).reshape(2, 8, 4)
    full_w3 = full_w1 + 100
    full_w2 = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8) + 200

    owner._load_w13(
        expert_data=reference_w13.data,
        shard_dim=1,
        shard_id="w1",
        loaded_weight=full_w1,
        tp_rank=1,
    )
    owner._load_w13(
        expert_data=reference_w13.data,
        shard_dim=1,
        shard_id="w3",
        loaded_weight=full_w3,
        tp_rank=1,
    )
    owner._load_w2(
        expert_data=reference_w2.data,
        shard_dim=2,
        loaded_weight=full_w2,
        tp_rank=1,
    )

    ext._load_destination_local_expert_group(
        "w13", destination_w13, "w1", list(enumerate(full_w1[:, 4:]))
    )
    ext._load_destination_local_expert_group(
        "w13", destination_w13, "w3", list(enumerate(full_w3[:, 4:]))
    )
    ext._load_destination_local_expert_group(
        "w2", destination_w2, "w2", list(enumerate(full_w2[:, :, 4:]))
    )

    torch.testing.assert_close(destination_w13, reference_w13)
    torch.testing.assert_close(destination_w2, reference_w2)


@pytest.mark.vllm
def test_refit_load_weights_uses_full_weight_path_by_default():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    loaded = []
    model = SimpleNamespace(load_weights=lambda *, weights: loaded.extend(weights))
    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.model_runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[])),
    )
    weight = torch.ones(2, 2)

    ext._load_weights([("model.weight", weight)])

    assert loaded == [("model.weight", weight)]


@pytest.mark.vllm
def test_refit_load_weights_dispatches_to_sharded_path_when_enabled():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    loaded = []
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.checkpoint_engine = SimpleNamespace(shard_expert_weights=True)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            load_weights=lambda **_kwargs: pytest.fail("used full loader")
        ),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[])),
    )
    ext._load_sharded_expert_weights = lambda weights: loaded.extend(weights)
    weight = torch.ones(2, 2)

    ext._load_weights([("model.weight", weight)])

    assert loaded == [("model.weight", weight)]


@pytest.mark.vllm
def test_checkpoint_refit_worker_falls_back_to_full_loader():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    loaded = []
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.checkpoint_engine = SimpleNamespace(shard_expert_weights=False)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(load_weights=lambda *, weights: loaded.extend(weights)),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[])),
    )
    weight = torch.ones(2, 2)

    ext._load_weights([("model.weight", weight)])

    assert loaded == [("model.weight", weight)]


@pytest.mark.vllm
def test_checkpoint_refit_preserves_nonsharded_fp8_path(monkeypatch):
    from nemo_rl.models.generation.vllm.quantization import fp8
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    loaded = []
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.checkpoint_engine = SimpleNamespace(shard_expert_weights=False)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            load_weights=lambda **_kwargs: pytest.fail("used full loader")
        ),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[])),
    )
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _config: True)

    def load_fp8_weights(weights, model_runner):
        assert model_runner is ext.model_runner
        loaded.extend(weights)

    monkeypatch.setattr(fp8, "load_weights", load_fp8_weights)
    weight = torch.ones(2, 2)

    ext._load_weights([("model.weight", weight)])

    assert loaded == [("model.weight", weight)]


@pytest.mark.vllm
def test_refit_load_weights_rejects_sharded_fp8_path(monkeypatch):
    from nemo_rl.models.generation.vllm.quantization import fp8
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.checkpoint_engine = SimpleNamespace(shard_expert_weights=True)
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            load_weights=lambda **_kwargs: pytest.fail("used full loader")
        ),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[])),
    )
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _config: True)

    with pytest.raises(ValueError, match="not supported for FP8"):
        ext._load_weights([("model.weight", torch.ones(2, 2))])


@pytest.mark.vllm
def test_checkpoint_engine_weight_layout_reports_ep_and_pp_ownership(monkeypatch):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    owner = _FakeExpertOwner(
        use_ep=True, expert_map=torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    )
    w13 = _expert_param((2, 8, 4), owner)
    w2 = _expert_param((2, 4, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("model.layers.0.mlp.experts.routed_experts.w13_weight", w13),
                ("model.layers.0.mlp.experts.routed_experts.w2_weight", w2),
            ]
        )
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.utils.get_pp_missing_layer_names",
        lambda model: ["model.layers.1."],
    )

    layout = ext._checkpoint_engine_weight_layout()

    assert layout["expert_params"][
        "model.layers.0.mlp.experts.routed_experts.w13_weight"
    ] == {
        "tp_rank": 0,
        "tp_size": 1,
        "local_expert_ids": [1, 3],
    }
    assert layout["missing_weight_prefixes"] == ["model.layers.1."]


@pytest.mark.vllm
def test_checkpoint_engine_weight_layout_rejects_shuffled_backend():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    owner = _FakeExpertOwner(use_ep=True, backend_name="FLASHINFER_TRTLLM")
    param = _expert_param((2, 8, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("model.layers.0.mlp.experts.routed_experts.w13_weight", param)
            ]
        )
    )
    with pytest.raises(ValueError, match="canonical unquantized Triton"):
        ext._checkpoint_engine_weight_layout()


@pytest.mark.vllm
def test_checkpoint_engine_weight_layout_rejects_transposed_experts():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    owner = _FakeExpertOwner(use_ep=True)
    param = _expert_param((2, 8, 4), owner)
    param.is_transposed = True
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("model.layers.0.mlp.experts.routed_experts.w13_weight", param)
            ]
        )
    )
    with pytest.raises(ValueError, match="canonical expert-weight orientation"):
        ext._checkpoint_engine_weight_layout()


@pytest.mark.vllm
def test_sharded_refit_directly_loads_full_ep_owned_experts():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    owner = _FakeExpertOwner(
        use_ep=True, expert_map=torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    )
    w13 = _expert_param((2, 8, 4), owner)
    w2 = _expert_param((2, 4, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("model.layers.0.mlp.experts.routed_experts.w13_weight", w13),
                ("model.layers.0.mlp.experts.routed_experts.w2_weight", w2),
            ]
        )
    )
    ext.state_dict_info = {
        "model.layers.0.mlp.experts.1.gate_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
        "model.layers.0.mlp.experts.3.gate_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
        "model.layers.0.mlp.experts.1.up_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
        "model.layers.0.mlp.experts.3.up_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
        "model.layers.0.mlp.experts.1.down_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
        "model.layers.0.mlp.experts.3.down_proj.weight": (
            torch.Size([4, 4]),
            torch.float32,
        ),
    }
    weights = {
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((4, 4), 1.0),
        "model.layers.0.mlp.experts.3.gate_proj.weight": torch.full((4, 4), 3.0),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((4, 4), 11.0),
        "model.layers.0.mlp.experts.3.up_proj.weight": torch.full((4, 4), 13.0),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((4, 4), 21.0),
        "model.layers.0.mlp.experts.3.down_proj.weight": torch.full((4, 4), 23.0),
    }

    remaining = ext._load_sharded_expert_weight_groups(list(weights.items()))

    assert remaining == []
    torch.testing.assert_close(
        w13[0, :4], weights["model.layers.0.mlp.experts.1.gate_proj.weight"]
    )
    torch.testing.assert_close(
        w13[1, :4], weights["model.layers.0.mlp.experts.3.gate_proj.weight"]
    )
    torch.testing.assert_close(
        w13[0, 4:], weights["model.layers.0.mlp.experts.1.up_proj.weight"]
    )
    torch.testing.assert_close(
        w13[1, 4:], weights["model.layers.0.mlp.experts.3.up_proj.weight"]
    )
    torch.testing.assert_close(
        w2[0], weights["model.layers.0.mlp.experts.1.down_proj.weight"]
    )
    torch.testing.assert_close(
        w2[1], weights["model.layers.0.mlp.experts.3.down_proj.weight"]
    )


@pytest.mark.vllm
def test_sharded_refit_loads_sparse_local_expert_ids_individually():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    owner = _FakeExpertOwner(use_ep=False)
    param = _expert_param((3, 8, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    expert_0 = torch.full((4, 4), 1.0)
    expert_2 = torch.full((4, 4), 2.0)

    ext._load_destination_local_expert_group(
        "model.layers.0.mlp.experts.routed_experts.w13_weight",
        param,
        "w1",
        [(0, expert_0), (2, expert_2)],
    )

    torch.testing.assert_close(param[0, :4], expert_0)
    torch.testing.assert_close(param[1], torch.zeros(8, 4))
    torch.testing.assert_close(param[2, :4], expert_2)


@pytest.mark.vllm
def test_sharded_refit_keeps_full_tp_weight_for_standard_loader():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    owner = _FakeExpertOwner(use_ep=False, tp_rank=0, tp_size=2)
    param = _expert_param((4, 8, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("model.layers.0.mlp.experts.routed_experts.w13_weight", param)
            ]
        )
    )
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    ext.state_dict_info = {name: (torch.Size([8, 4]), torch.float32)}
    full_weight = torch.ones(8, 4)

    remaining = ext._load_sharded_expert_weight_groups([(name, full_weight)])

    assert len(remaining) == 1
    assert remaining[0][0] == name
    assert remaining[0][1] is full_weight
    torch.testing.assert_close(param, torch.zeros_like(param))


@pytest.mark.vllm
def test_sharded_refit_requires_bound_vllm_expert_loader():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    param_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    weight_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    param = torch.nn.Parameter(torch.zeros(1, 8, 4), requires_grad=False)
    param.weight_loader = lambda *args, **kwargs: None
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext._nrl_named_parameters = {param_name: param}
    ext.state_dict_info = {weight_name: (torch.Size([8, 4]), torch.float32)}

    with pytest.raises(RuntimeError, match="Could not resolve.*w13_weight"):
        ext._load_sharded_expert_weight_groups([(weight_name, torch.zeros(4, 4))])


@pytest.mark.vllm
def test_sharded_refit_rejects_noncanonical_expert_dimensions():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    param_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    weight_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    owner = _FakeExpertOwner(use_ep=False)
    param = _expert_param((8, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext._nrl_named_parameters = {param_name: param}
    ext.state_dict_info = {weight_name: (torch.Size([8, 4]), torch.float32)}

    with pytest.raises(ValueError, match="requires a 3-D vLLM parameter"):
        ext._load_sharded_expert_weight_groups([(weight_name, torch.zeros(4, 4))])


@pytest.mark.vllm
def test_sharded_refit_validates_source_shape_against_reported_tp_size():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtensionWithCheckpointEngine,
    )

    param_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    weight_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    owner = _FakeExpertOwner(use_ep=False, tp_size=2)
    param = _expert_param((1, 8, 4), owner)
    ext = VllmInternalWorkerExtensionWithCheckpointEngine.__new__(
        VllmInternalWorkerExtensionWithCheckpointEngine
    )
    ext._nrl_named_parameters = {param_name: param}
    ext.state_dict_info = {weight_name: (torch.Size([8, 4]), torch.float32)}

    with pytest.raises(ValueError, match=r"expected \(4, 4\).*TP size 2"):
        ext._load_sharded_expert_weight_groups([(weight_name, torch.zeros(2, 4))])
