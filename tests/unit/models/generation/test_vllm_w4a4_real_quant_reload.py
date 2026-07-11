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

import importlib
import sys
import types

import pytest
import torch

from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
    _modelopt_dense_process_weights,
    _modelopt_moe_process_weights,
    _set_or_update_processed_tensor_ref,
    prepare_modelopt_for_weight_reload,
)


class _NoOpNvfp4Kernel:
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        del layer


def test_processed_tensor_ref_materializes_expanded_storage_for_refit() -> None:
    layer = torch.nn.Module()
    first = torch.tensor([1.0, 2.0]).unsqueeze(-1).expand(2, 2)

    _set_or_update_processed_tensor_ref(
        layer,
        "input_scale",
        first,
        is_first_call=True,
    )

    first_ptr = layer.input_scale.data_ptr()
    assert layer.input_scale.stride() != first.stride()
    torch.testing.assert_close(layer.input_scale, first)

    updated = torch.tensor([3.0, 4.0]).unsqueeze(-1).expand(2, 2)
    _set_or_update_processed_tensor_ref(
        layer,
        "input_scale",
        updated,
        is_first_call=False,
    )

    assert layer.input_scale.data_ptr() == first_ptr
    torch.testing.assert_close(layer.input_scale, updated)


def _make_dense_w4a4_layer(
    *,
    input_scales: tuple[float, ...] | None = (0.125, 0.25),
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.tensor([[1.0], [2.0]]),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
        requires_grad=False,
    )
    layer.weight_scale_2 = torch.nn.Parameter(
        torch.tensor([0.5]),
        requires_grad=False,
    )
    if input_scales is not None:
        layer.input_scale = torch.nn.Parameter(
            torch.tensor(input_scales),
            requires_grad=False,
        )
    return layer


def _install_fake_modelopt(monkeypatch: pytest.MonkeyPatch) -> None:
    module_names = (
        "modelopt",
        "modelopt.torch",
        "modelopt.torch.quantization",
        "modelopt.torch.quantization.nn",
        "modelopt.torch.quantization.nn.modules",
    )
    modules: dict[str, types.ModuleType] = {}
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        modules[module_name] = module
        monkeypatch.setitem(sys.modules, module_name, module)

    tensor_quantizer_module = types.ModuleType(
        "modelopt.torch.quantization.nn.modules.tensor_quantizer"
    )

    class FakeTensorQuantizer(torch.nn.Module):
        pass

    tensor_quantizer_module.TensorQuantizer = FakeTensorQuantizer
    monkeypatch.setitem(
        sys.modules,
        "modelopt.torch.quantization.nn.modules.tensor_quantizer",
        tensor_quantizer_module,
    )
    modules["modelopt"].torch = modules["modelopt.torch"]
    modules["modelopt.torch"].quantization = modules["modelopt.torch.quantization"]
    modules["modelopt.torch.quantization"].nn = modules[
        "modelopt.torch.quantization.nn"
    ]
    modules["modelopt.torch.quantization.nn"].modules = modules[
        "modelopt.torch.quantization.nn.modules"
    ]
    modules[
        "modelopt.torch.quantization.nn.modules"
    ].tensor_quantizer = tensor_quantizer_module


def _import_quant_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> types.ModuleType:
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    _install_fake_modelopt(monkeypatch)

    fake_vllm_backend = types.ModuleType("nemo_rl.models.generation.vllm.vllm_backend")

    class FakeVllmInternalWorkerExtension:
        def prepare_refit_info(self, state_dict_info) -> None:
            self.state_dict_info = state_dict_info

    fake_vllm_backend.VllmInternalWorkerExtension = FakeVllmInternalWorkerExtension
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.models.generation.vllm.vllm_backend",
        fake_vllm_backend,
    )

    module_name = "nemo_rl.modelopt.models.generation.vllm_quant_backend"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def _make_moe_model(
    *,
    local_num_experts: int = 2,
    global_num_experts: int = 2,
    expert_map: torch.Tensor | None = None,
    w13_num_shards: int = 2,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    class ModelOptNvFp4FusedMoE:
        pass

    moe = torch.nn.Module()
    moe.quant_method = ModelOptNvFp4FusedMoE()
    moe.local_num_experts = local_num_experts
    moe.global_num_experts = global_num_experts
    moe._expert_map = expert_map
    moe.w13_input_scale = torch.nn.Parameter(
        torch.full((global_num_experts, w13_num_shards), torch.nan),
        requires_grad=False,
    )
    moe.w2_input_scale = torch.nn.Parameter(
        torch.full((global_num_experts,), torch.nan),
        requires_grad=False,
    )
    moe.w13_weight_scale_2 = torch.nn.Parameter(
        torch.full((global_num_experts, w13_num_shards), torch.nan),
        requires_grad=False,
    )
    moe.w2_weight_scale_2 = torch.nn.Parameter(
        torch.full((global_num_experts,), torch.nan),
        requires_grad=False,
    )

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.model.layers[0].mlp = moe
    return root, moe


def test_dense_w4a4_refit_derives_scales_and_preserves_storage() -> None:
    layer = _make_dense_w4a4_layer()
    quant_method = types.SimpleNamespace(kernel=_NoOpNvfp4Kernel())

    _modelopt_dense_process_weights(quant_method, layer)

    assert not hasattr(layer, "input_scale")
    assert not hasattr(layer, "weight_scale_2")
    torch.testing.assert_close(layer.input_global_scale, torch.tensor(0.25))
    torch.testing.assert_close(layer.weight_global_scale, torch.tensor(0.5))
    torch.testing.assert_close(layer.alpha, torch.tensor(0.125))
    torch.testing.assert_close(layer.input_global_scale_inv, torch.tensor(4.0))
    pointer_names = (
        "weight",
        "weight_scale",
        "weight_global_scale",
        "input_global_scale",
        "alpha",
        "input_global_scale_inv",
    )
    first_data_ptrs = {name: getattr(layer, name).data_ptr() for name in pointer_names}
    block_scale = layer.weight_scale.clone()
    block_scale_ptr = layer.weight_scale.data_ptr()

    prepare_modelopt_for_weight_reload(layer, device=torch.device("cpu"))
    assert torch.isnan(layer.weight_scale_2).all()
    assert torch.isnan(layer.input_scale).all()
    assert layer.weight_scale.data_ptr() == block_scale_ptr
    torch.testing.assert_close(layer.weight_scale, block_scale)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[9.0], [8.0]]))
        layer.weight_scale.copy_(torch.tensor([[4.0, 3.0], [2.0, 1.0]]))
        layer.weight_scale_2.copy_(torch.tensor([4.0]))
        layer.input_scale.copy_(torch.tensor([0.5, 0.75]))

    _modelopt_dense_process_weights(quant_method, layer)

    assert {
        name: getattr(layer, name).data_ptr() for name in pointer_names
    } == first_data_ptrs
    torch.testing.assert_close(layer.weight, torch.tensor([[9.0], [8.0]]))
    torch.testing.assert_close(
        layer.weight_scale,
        torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
    )
    torch.testing.assert_close(layer.input_global_scale, torch.tensor(0.75))
    torch.testing.assert_close(layer.weight_global_scale, torch.tensor(4.0))
    torch.testing.assert_close(layer.alpha, torch.tensor(3.0))
    torch.testing.assert_close(
        layer.input_global_scale_inv,
        torch.tensor(4.0 / 3.0),
    )


def test_dense_w4a4_rejects_missing_input_scale() -> None:
    layer = _make_dense_w4a4_layer(input_scales=None)
    quant_method = types.SimpleNamespace(kernel=_NoOpNvfp4Kernel())

    with pytest.raises(RuntimeError, match="requires input_scale"):
        _modelopt_dense_process_weights(quant_method, layer)


@pytest.mark.parametrize("input_scale", [0.0, -0.25])
def test_dense_w4a4_rejects_nonpositive_input_scale(input_scale: float) -> None:
    layer = _make_dense_w4a4_layer(input_scales=(input_scale,))
    layer._nrl_process_weights_call_count = 1
    quant_method = types.SimpleNamespace(kernel=_NoOpNvfp4Kernel())

    with pytest.raises(RuntimeError, match="input_scale must be strictly positive"):
        _modelopt_dense_process_weights(quant_method, layer)


def test_dense_w4a4_sanitizes_signed_dummy_scales() -> None:
    layer = _make_dense_w4a4_layer(input_scales=(-0.25,))
    layer.weight_scale.data[0, 0] = torch.nan
    layer.weight_scale_2.data.zero_()
    quant_method = types.SimpleNamespace(kernel=_NoOpNvfp4Kernel())

    _modelopt_dense_process_weights(quant_method, layer)

    torch.testing.assert_close(layer.input_global_scale, torch.tensor(0.25))
    torch.testing.assert_close(layer.weight_global_scale, torch.tensor(1.0))
    assert torch.isfinite(layer.weight_scale).all()


def test_dense_w4a4_rejects_weight_only_marlin_backend() -> None:
    layer = _make_dense_w4a4_layer()
    quant_method = types.SimpleNamespace(
        backend=types.SimpleNamespace(value="MARLIN"),
    )

    with pytest.raises(RuntimeError, match="Marlin is weight-only"):
        _modelopt_dense_process_weights(quant_method, layer)


@pytest.mark.parametrize(
    ("is_act_and_mul", "w13_num_shards"),
    [(True, 2), (False, 1)],
    ids=("gated", "non_gated"),
)
def test_fused_moe_w4a4_refit_reuses_processed_storage(
    monkeypatch: pytest.MonkeyPatch,
    is_act_and_mul: bool,
    w13_num_shards: int,
) -> None:
    oracle_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4"
    )
    conversion_calls: list[dict[str, torch.Tensor]] = []
    conversion_modes: list[bool] = []
    kernel_calls: list[object] = []
    postprocess_calls: list[torch.nn.Module] = []

    def convert_to_nvfp4_moe_kernel_format(
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        tensor_names = (
            "w13",
            "w13_scale",
            "w13_scale_2",
            "a13_scale",
            "w2",
            "w2_scale",
            "w2_scale_2",
            "a2_scale",
        )
        tensors = tuple(kwargs[name] for name in tensor_names)
        assert all(isinstance(tensor, torch.Tensor) for tensor in tensors)
        conversion_calls.append(
            {
                name: tensor.detach().clone()
                for name, tensor in zip(tensor_names, tensors)
            }
        )
        conversion_modes.append(bool(kwargs["is_act_and_mul"]))
        return tuple(tensor.detach().clone() for tensor in tensors)

    class FakeFusedExperts:
        def __init__(self, quant_config: object) -> None:
            self.quant_config = quant_config

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            postprocess_calls.append(layer)

    def make_nvfp4_moe_kernel(
        *,
        moe_quant_config: object,
        moe_config: object,
        experts_cls: object,
        nvfp4_backend: object,
    ) -> object:
        del moe_config, experts_cls, nvfp4_backend
        kernel_calls.append(moe_quant_config)
        return types.SimpleNamespace(
            fused_experts=FakeFusedExperts(moe_quant_config),
        )

    oracle_module.convert_to_nvfp4_moe_kernel_format = (
        convert_to_nvfp4_moe_kernel_format
    )
    oracle_module.make_nvfp4_moe_kernel = make_nvfp4_moe_kernel
    module_names = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.oracle",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
        oracle_module,
    )

    layer = torch.nn.Module()
    layer.local_num_experts = 2
    layer.global_num_experts = 2
    layer._expert_map = None
    initial_tensors = {
        "w13_weight": torch.arange(8 * w13_num_shards, dtype=torch.float32).reshape(
            2, 2 * w13_num_shards, 2
        )
        + 1,
        "w13_weight_scale": torch.full((2, 2 * w13_num_shards, 1), 0.5),
        "w13_weight_scale_2": torch.ones(2, w13_num_shards),
        "w13_input_scale": torch.arange(
            1, 2 * w13_num_shards + 1, dtype=torch.float32
        ).reshape(2, w13_num_shards)
        / 4,
        "w2_weight": torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) + 1,
        "w2_weight_scale": torch.full((2, 1, 4), 0.25),
        "w2_weight_scale_2": torch.tensor([1.5, 2.0]),
        "w2_input_scale": torch.tensor([1.25, 1.5]),
    }
    for name, tensor in initial_tensors.items():
        setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))

    class ModelOptNvFp4FusedMoE:
        def __init__(self) -> None:
            self.nvfp4_backend = types.SimpleNamespace(value="TRTLLM")
            self.moe = types.SimpleNamespace(is_act_and_mul=is_act_and_mul)
            self.experts_cls = object()
            self.quant_config = object()

        def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> object:
            return types.SimpleNamespace(
                g1_alphas=layer.w13_input_scale.detach().clone(),
                g2_alphas=layer.w2_input_scale.detach().clone(),
                a1_gscale=layer.w13_input_scale.amax(dim=1).detach().clone(),
                a2_gscale=layer.w2_input_scale.detach().clone(),
                w1_scale=layer.w13_weight_scale_2.detach().clone(),
                w2_scale=layer.w2_weight_scale_2.detach().clone(),
            )

    quant_method = ModelOptNvFp4FusedMoE()
    _modelopt_moe_process_weights(quant_method, layer)

    processed_names = tuple(initial_tensors)
    first_data_ptrs = {
        name: getattr(layer, name).data_ptr() for name in processed_names
    }
    quant_config_names = (
        "g1_alphas",
        "g2_alphas",
        "a1_gscale",
        "a2_gscale",
        "w1_scale",
        "w2_scale",
    )
    first_quant_config = quant_method.moe_quant_config
    first_quant_config_ptrs = {
        name: getattr(first_quant_config, name).data_ptr()
        for name in quant_config_names
    }

    prepare_modelopt_for_weight_reload(layer, device=torch.device("cpu"))
    reloaded_w13_scale_2 = torch.tensor([13.0, 14.0])[:, None].expand(
        -1, w13_num_shards
    )
    reloaded_tensors = {
        "w13_weight": torch.full((2, 2 * w13_num_shards, 2), 11.0),
        "w13_weight_scale": torch.full((2, 2 * w13_num_shards, 1), 12.0),
        "w13_weight_scale_2": reloaded_w13_scale_2,
        "w13_input_scale": torch.arange(
            15, 15 + 2 * w13_num_shards, dtype=torch.float32
        ).reshape(2, w13_num_shards),
        "w2_weight": torch.full((2, 2, 4), 19.0),
        "w2_weight_scale": torch.full((2, 1, 4), 20.0),
        "w2_weight_scale_2": torch.tensor([21.0, 22.0]),
        "w2_input_scale": torch.tensor([23.0, 24.0]),
    }
    with torch.no_grad():
        for name, tensor in reloaded_tensors.items():
            getattr(layer, name).copy_(tensor)

    _modelopt_moe_process_weights(quant_method, layer)

    assert len(conversion_calls) == 2
    assert conversion_modes == [is_act_and_mul, is_act_and_mul]
    torch.testing.assert_close(
        conversion_calls[1]["a13_scale"],
        reloaded_tensors["w13_input_scale"],
    )
    torch.testing.assert_close(
        conversion_calls[1]["a2_scale"],
        reloaded_tensors["w2_input_scale"],
    )
    assert {
        name: getattr(layer, name).data_ptr() for name in processed_names
    } == first_data_ptrs
    assert quant_method.moe_quant_config is first_quant_config
    assert {
        name: getattr(first_quant_config, name).data_ptr()
        for name in quant_config_names
    } == first_quant_config_ptrs
    expected_quant_config = {
        "g1_alphas": reloaded_tensors["w13_input_scale"],
        "g2_alphas": reloaded_tensors["w2_input_scale"],
        "a1_gscale": reloaded_tensors["w13_input_scale"].amax(dim=1),
        "a2_gscale": reloaded_tensors["w2_input_scale"],
        "w1_scale": torch.tensor([13.0, 14.0]),
        "w2_scale": reloaded_tensors["w2_weight_scale_2"],
    }
    for name, expected in expected_quant_config.items():
        torch.testing.assert_close(getattr(first_quant_config, name), expected)
    assert len(kernel_calls) == 1
    assert postprocess_calls == [layer, layer]


def test_fused_moe_input_scale_batching_maps_gate_up_and_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    w13_input_scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w2_input_scale = torch.tensor([5.0, 6.0])

    batched = backend._batch_fused_modelopt_moe_weights(
        [
            (
                "model.layers.0.mlp.experts.w13_input_scale",
                w13_input_scale,
            ),
            (
                "model.layers.0.mlp.experts.w2_input_scale",
                w2_input_scale,
            ),
        ],
        w13_num_shards_by_prefix={},
    )

    assert [name for name, _ in batched] == [
        "model.layers.0.mlp.experts.0.gate_proj.input_scale",
        "model.layers.0.mlp.experts.0.up_proj.input_scale",
        "model.layers.0.mlp.experts.1.gate_proj.input_scale",
        "model.layers.0.mlp.experts.1.up_proj.input_scale",
        "model.layers.0.mlp.experts.0.down_proj.input_scale",
        "model.layers.0.mlp.experts.1.down_proj.input_scale",
    ]
    torch.testing.assert_close(
        torch.stack([tensor for _, tensor in batched]),
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )


def test_fused_moe_input_scale_batching_maps_non_gated_up_and_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)

    batched = backend._batch_fused_modelopt_moe_weights(
        [
            (
                "model.layers.0.mlp.experts.up_proj_input_scale",
                torch.tensor([[1.0], [2.0]]),
            ),
            (
                "model.layers.0.mlp.experts.w2_input_scale",
                torch.tensor([3.0, 4.0]),
            ),
        ],
        w13_num_shards_by_prefix={},
    )

    assert [name for name, _ in batched] == [
        "model.layers.0.mlp.experts.0.up_proj.input_scale",
        "model.layers.0.mlp.experts.1.up_proj.input_scale",
        "model.layers.0.mlp.experts.0.down_proj.input_scale",
        "model.layers.0.mlp.experts.1.down_proj.input_scale",
    ]
    torch.testing.assert_close(
        torch.stack([tensor for _, tensor in batched]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )


def test_all_bridge_fused_moe_tensors_map_to_vllm_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    prefix = "model.layers.0.mlp"
    w13_weight = torch.arange(32).reshape(2, 8, 2)
    w13_weight_scale = torch.arange(16).reshape(2, 8, 1)
    tensors = {
        f"{prefix}.experts.w13_weight": w13_weight,
        f"{prefix}.experts.w13_weight_scale": w13_weight_scale,
        f"{prefix}.experts.w13_weight_scale_2": torch.ones(2, 2),
        f"{prefix}.experts.w13_input_scale": torch.ones(2, 2),
        f"{prefix}.experts.w2_weight": torch.zeros(2, 2, 2),
        f"{prefix}.experts.w2_weight_scale": torch.zeros(2, 2, 1),
        f"{prefix}.experts.w2_weight_scale_2": torch.ones(2),
        f"{prefix}.experts.w2_input_scale": torch.ones(2),
    }

    batched = backend._batch_fused_modelopt_moe_weights(
        list(tensors.items()),
        w13_num_shards_by_prefix={prefix: 2},
    )
    names = [name for name, _ in batched]
    mapped = dict(batched)

    assert f"{prefix}.experts.w13_weight" not in names
    assert f"{prefix}.experts.w13_weight_scale" not in names
    assert f"{prefix}.experts.w13_weight_scale_2" not in names
    gate_weight_name = f"{prefix}.experts.0.gate_proj.weight"
    up_weight_name = f"{prefix}.experts.0.up_proj.weight"
    gate_scale_name = f"{prefix}.experts.0.gate_proj.weight_scale"
    up_scale_name = f"{prefix}.experts.0.up_proj.weight_scale"
    assert gate_weight_name in names
    assert up_weight_name in names
    assert gate_scale_name in names
    assert up_scale_name in names
    torch.testing.assert_close(mapped[gate_weight_name], w13_weight[:, :4])
    torch.testing.assert_close(mapped[up_weight_name], w13_weight[:, 4:])
    torch.testing.assert_close(mapped[gate_scale_name], w13_weight_scale[:, :4])
    torch.testing.assert_close(mapped[up_scale_name], w13_weight_scale[:, 4:])
    assert f"{prefix}.experts.0.gate_proj.weight_scale_2" in names
    assert f"{prefix}.experts.1.up_proj.weight_scale_2" in names
    assert f"{prefix}.experts.0.down_proj.weight" in names
    assert f"{prefix}.experts.1.down_proj.input_scale" in names


def test_one_shard_w13_layout_is_reused_across_refit_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    prefix = "backbone.layers.1.mixer"
    state_dict_info = {
        f"{prefix}.experts.w13_weight": ((2, 8, 2), torch.uint8),
        f"{prefix}.experts.w13_weight_scale": ((2, 8, 1), torch.uint8),
        f"{prefix}.experts.w13_weight_scale_2": ((2, 1), torch.float32),
        f"{prefix}.experts.w13_input_scale": ((2, 1), torch.float32),
        f"{prefix}.experts.w2_weight": ((2, 2, 4), torch.uint8),
        f"{prefix}.experts.w2_weight_scale": ((2, 1, 4), torch.uint8),
        f"{prefix}.experts.w2_weight_scale_2": ((2,), torch.float32),
        f"{prefix}.experts.w2_input_scale": ((2,), torch.float32),
    }
    extension = backend.VllmQuantInternalWorkerExtension()
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )
    extension.model_runner = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    quantization_config={"quant_mode": "w4a4_nvfp4"}
                )
            )
        )
    )

    extension.prepare_refit_info(state_dict_info)

    assert extension.state_dict_info is state_dict_info
    assert extension._nrl_w13_num_shards_by_prefix == {prefix: 1}
    w13_weight = torch.arange(32).reshape(2, 8, 2)
    w13_weight_scale = torch.arange(16).reshape(2, 8, 1)
    w13_weight_scale_2 = torch.tensor([[1.0], [2.0]])
    w13_input_scale = torch.tensor([[3.0], [4.0]])
    mapped = dict(
        backend._batch_fused_modelopt_moe_weights(
            [
                (f"{prefix}.experts.w13_weight", w13_weight),
                (f"{prefix}.experts.w13_weight_scale", w13_weight_scale),
            ],
            w13_num_shards_by_prefix=extension._nrl_w13_num_shards_by_prefix,
        )
        + backend._batch_fused_modelopt_moe_weights(
            [
                (f"{prefix}.experts.w13_weight_scale_2", w13_weight_scale_2),
                (f"{prefix}.experts.w13_input_scale", w13_input_scale),
            ],
            w13_num_shards_by_prefix=extension._nrl_w13_num_shards_by_prefix,
        )
    )

    assert mapped[f"{prefix}.experts.0.up_proj.weight"] is w13_weight
    assert mapped[f"{prefix}.experts.0.up_proj.weight_scale"] is w13_weight_scale
    assert not any("gate_proj" in name for name in mapped)
    torch.testing.assert_close(
        mapped[f"{prefix}.experts.1.up_proj.weight_scale_2"],
        w13_weight_scale_2[1, 0],
    )
    torch.testing.assert_close(
        mapped[f"{prefix}.experts.1.up_proj.input_scale"],
        w13_input_scale[1, 0],
    )
    legacy_mapped = dict(
        backend._batch_fused_modelopt_moe_weights(
            [
                (f"{prefix}.experts.w13_weight_scale_2", w13_weight_scale_2[:, 0]),
                (f"{prefix}.experts.w13_input_scale", w13_input_scale[:, 0]),
            ],
            w13_num_shards_by_prefix=extension._nrl_w13_num_shards_by_prefix,
        )
    )
    torch.testing.assert_close(
        legacy_mapped[f"{prefix}.experts.1.up_proj.weight_scale_2"],
        w13_weight_scale_2[1, 0],
    )
    torch.testing.assert_close(
        legacy_mapped[f"{prefix}.experts.1.up_proj.input_scale"],
        w13_input_scale[1, 0],
    )


def test_fused_moe_scales_are_stashed_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    model, moe = _make_moe_model()
    extension = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=model),
    )
    weights = [
        (
            "model.layers.0.mlp.experts.w13_weight_scale_2",
            torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        ),
        (
            "model.layers.0.mlp.experts.w2_weight_scale_2",
            torch.tensor([11.0, 12.0]),
        ),
        (
            "model.layers.0.mlp.experts.w13_input_scale",
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ),
        (
            "model.layers.0.mlp.experts.w2_input_scale",
            torch.tensor([5.0, 6.0]),
        ),
    ]

    backend._stash_fused_modelopt_moe_scales(extension, weights)
    backend._restore_fused_modelopt_moe_scales(extension)

    torch.testing.assert_close(
        moe.w13_input_scale,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(moe.w2_input_scale, torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(
        moe.w13_weight_scale_2,
        torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
    )
    torch.testing.assert_close(
        moe.w2_weight_scale_2,
        torch.tensor([11.0, 12.0]),
    )


def test_non_gated_moe_scales_are_stashed_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    model, moe = _make_moe_model(w13_num_shards=1)
    extension = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=model),
    )
    weights = [
        (
            "model.layers.0.mlp.experts.up_proj_input_scale",
            torch.tensor([[1.0], [2.0]]),
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight_scale_2",
            torch.tensor(3.0),
        ),
        (
            "model.layers.0.mlp.experts.1.up_proj.weight_scale_2",
            torch.tensor(4.0),
        ),
    ]

    backend._stash_fused_modelopt_moe_scales(extension, weights)
    backend._restore_fused_modelopt_moe_scales(extension)

    torch.testing.assert_close(moe.w13_input_scale, torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(
        moe.w13_weight_scale_2,
        torch.tensor([[3.0], [4.0]]),
    )


def test_non_gated_fused_moe_scale_2_is_stashed_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    model, moe = _make_moe_model(w13_num_shards=1)
    extension = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=model),
    )
    prefix = "model.layers.0.mlp"

    backend._stash_fused_modelopt_moe_scales(
        extension,
        [
            (
                f"{prefix}.experts.up_proj_scale_2",
                torch.tensor([3.0, 4.0]),
            )
        ],
    )

    stashed = extension._nrl_fused_moe_scales[(prefix, "w13_weight_scale_2")]
    assert stashed.shape == (2, 1)
    backend._restore_fused_modelopt_moe_scales(extension)
    torch.testing.assert_close(
        moe.w13_weight_scale_2,
        torch.tensor([[3.0], [4.0]]),
    )


def test_nemotron_h_moe_scales_follow_vllm_weight_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    _, moe = _make_moe_model(w13_num_shards=1)

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Module()])
    model.model.layers[1].mixer = torch.nn.Module()
    model.model.layers[1].mixer.experts = moe

    mapper_calls: list[list[str]] = []

    class PrefixMapper:
        def apply_list(self, prefixes: list[str]) -> list[str]:
            mapper_calls.append(prefixes)
            return [prefix.replace("backbone", "model", 1) for prefix in prefixes]

    model.hf_to_vllm_mapper = PrefixMapper()
    extension = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=model),
    )
    prefix = "backbone.layers.1.mixer"

    backend._stash_fused_modelopt_moe_scales(
        extension,
        [(f"{prefix}.experts.up_proj_scale_2", torch.tensor([3.0, 4.0]))],
    )
    backend._restore_fused_modelopt_moe_scales(extension)

    assert mapper_calls == [[prefix]]
    torch.testing.assert_close(
        moe.w13_weight_scale_2,
        torch.tensor([[3.0], [4.0]]),
    )


def test_per_expert_moe_scales_are_stashed_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    model, moe = _make_moe_model()
    extension = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=model),
    )
    weights = [
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
            torch.tensor(7.0),
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight_scale_2",
            torch.tensor(8.0),
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight_scale_2",
            torch.tensor(11.0),
        ),
        (
            "model.layers.0.mlp.experts.1.gate_proj.weight_scale_2",
            torch.tensor(9.0),
        ),
        (
            "model.layers.0.mlp.experts.1.up_proj.weight_scale_2",
            torch.tensor(10.0),
        ),
        (
            "model.layers.0.mlp.experts.1.down_proj.weight_scale_2",
            torch.tensor(12.0),
        ),
        (
            "model.layers.0.mlp.experts.0.gate_proj.input_scale",
            torch.tensor(1.0),
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.input_scale",
            torch.tensor(2.0),
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.input_scale",
            torch.tensor(5.0),
        ),
        (
            "model.layers.0.mlp.experts.1.gate_proj.input_scale",
            torch.tensor(3.0),
        ),
        (
            "model.layers.0.mlp.experts.1.up_proj.input_scale",
            torch.tensor(4.0),
        ),
        (
            "model.layers.0.mlp.experts.1.down_proj.input_scale",
            torch.tensor(6.0),
        ),
    ]

    backend._stash_fused_modelopt_moe_scales(extension, weights)
    backend._restore_fused_modelopt_moe_scales(extension)

    torch.testing.assert_close(
        moe.w13_input_scale,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(moe.w2_input_scale, torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(
        moe.w13_weight_scale_2,
        torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
    )
    torch.testing.assert_close(
        moe.w2_weight_scale_2,
        torch.tensor([11.0, 12.0]),
    )


def test_fused_moe_refit_requires_all_experts_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _import_quant_backend(monkeypatch)
    full_model, _ = _make_moe_model()
    assert backend._supports_batched_modelopt_moe_load(full_model)

    partial_model, _ = _make_moe_model(
        local_num_experts=1,
        expert_map=torch.tensor([0, -1]),
    )
    assert not backend._supports_batched_modelopt_moe_load(partial_model)

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=partial_model,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(quantization_config={"ignore": []})
            )
        ),
    )
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda self: True,
    )

    with pytest.raises(RuntimeError, match="all experts local"):
        extension._load_weights(
            [
                (
                    "model.layers.0.mlp.experts.w13_input_scale",
                    torch.ones(2, 2),
                )
            ]
        )
