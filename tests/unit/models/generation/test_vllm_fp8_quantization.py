# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import types

import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.fixture(autouse=True)
def mock_vllm_tp_metadata(monkeypatch):
    pytest.importorskip("vllm")

    from vllm.model_executor import parameter as vllm_parameter

    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )


@pytest.fixture()
def fp8_module():
    pytest.importorskip("vllm")

    from nemo_rl.models.generation.vllm.quantization import fp8

    old_config = fp8.global_fp8_config
    old_state = fp8.fp8_state
    old_patches_applied = fp8.fp8_patches_applied
    old_run_engine_core = fp8.EngineCoreProc.run_engine_core
    old_core_manager_init = fp8.CoreEngineProcManager.__init__
    fp8.global_fp8_config = None
    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False

    try:
        yield fp8
    finally:
        fp8.global_fp8_config = old_config
        fp8.fp8_state = old_state
        fp8.fp8_patches_applied = old_patches_applied
        fp8.EngineCoreProc.run_engine_core = old_run_engine_core
        fp8.CoreEngineProcManager.__init__ = old_core_manager_init


@pytest.mark.parametrize("async_engine", [False, True])
@pytest.mark.parametrize("refit_with_reload_api", [False, True])
def test_init_fp8_uses_mxfp8_quantization_config(
    fp8_module, monkeypatch, async_engine, refit_with_reload_api
):
    fp8 = fp8_module
    original_run_engine_core = fp8.EngineCoreProc.run_engine_core
    original_core_manager_init = fp8.CoreEngineProcManager.__init__
    applied_configs = []

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(
        fp8,
        "monkey_patch_vllm_ray_executor",
        lambda fp8_config: applied_configs.append(fp8_config),
    )
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": async_engine,
            "is_mx": True,
            "use_deep_gemm": True,
            "refit_with_reload_api": refit_with_reload_api,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    assert vllm_kwargs == {
        "quantization": "fp8",
        "kv_cache_dtype": "auto",
        "hf_overrides": {"quantization_config": fp8.MXFP8_BLOCK_QUANT_KWARGS},
    }
    if async_engine:
        assert applied_configs == []
        assert fp8.EngineCoreProc.run_engine_core is fp8.my_run_engine_core
        assert fp8.CoreEngineProcManager.__init__ is fp8.my_init
    else:
        assert len(applied_configs) == 1
        assert applied_configs[0] is fp8.global_fp8_config
        assert fp8.EngineCoreProc.run_engine_core is original_run_engine_core
        assert fp8.CoreEngineProcManager.__init__ is original_core_manager_init
    assert fp8.global_fp8_config.is_mx is True
    assert "VLLM_USE_DEEP_GEMM" not in fp8.os.environ
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in fp8.os.environ


@pytest.mark.parametrize("async_engine", [False, True])
def test_init_fp8_reload_api_keeps_blockwise_vllm_native(
    fp8_module, monkeypatch, async_engine
):
    fp8 = fp8_module
    original_run_engine_core = fp8.EngineCoreProc.run_engine_core
    original_core_manager_init = fp8.CoreEngineProcManager.__init__
    applied_configs = []
    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(
        fp8,
        "monkey_patch_vllm_ray_executor",
        lambda fp8_config: applied_configs.append(fp8_config),
    )

    fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": async_engine,
            "is_mx": False,
            "refit_with_reload_api": True,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    assert applied_configs == []
    assert fp8.EngineCoreProc.run_engine_core is original_run_engine_core
    assert fp8.CoreEngineProcManager.__init__ is original_core_manager_init


@pytest.mark.parametrize("precision", [None, "auto", "bf16", "bfloat16"])
def test_init_fp8_rejects_mxfp8_without_fp8_precision(
    fp8_module, monkeypatch, precision
):
    fp8 = fp8_module
    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )

    with pytest.raises(ValueError, match="is_mx=True requires precision='fp8'"):
        fp8.init_fp8(
            {
                "precision": precision,
                "kv_cache_dtype": "auto",
                "is_mx": True,
            },
            "dummy-model",
            model_parallel_size=1,
        )


def test_patch_ray_executor_v2_applies_fp8_before_worker_init(fp8_module, monkeypatch):
    import cloudpickle
    from vllm.v1.executor import ray_executor_v2

    fp8 = fp8_module
    events = []

    class FakeRayWorkerProc:
        def initialize_worker(self, *args, **kwargs):
            events.append(("initialize", args, kwargs))
            return "initialized"

    monkeypatch.setattr(ray_executor_v2, "RayWorkerProc", FakeRayWorkerProc)
    monkeypatch.setattr(
        fp8,
        "apply_fp8_patches",
        lambda _worker, config: events.append(("patch", config)),
    )

    first_config = "first-config"
    second_config = "second-config"
    fp8._patch_vllm_ray_executor_v2(first_config)
    first_worker_cls_local = ray_executor_v2.RayWorkerProc
    first_worker_local = first_worker_cls_local()
    first_result_local = first_worker_local.initialize_worker(1, {"A": "B"})

    fp8._patch_vllm_ray_executor_v2(second_config)
    second_worker_cls_local = ray_executor_v2.RayWorkerProc
    second_worker_local = second_worker_cls_local()
    second_result_local = second_worker_local.initialize_worker(2, {"C": "D"})

    assert first_result_local == "initialized"
    assert second_result_local == "initialized"
    assert events == [
        ("patch", first_config),
        ("initialize", (1, {"A": "B"}), {}),
        ("patch", second_config),
        ("initialize", (2, {"C": "D"}), {}),
    ]
    assert first_worker_cls_local.__bases__ == (FakeRayWorkerProc,)
    assert second_worker_cls_local.__bases__ == (FakeRayWorkerProc,)

    first_worker_cls = cloudpickle.loads(cloudpickle.dumps(first_worker_cls_local))
    second_worker_cls = cloudpickle.loads(cloudpickle.dumps(second_worker_cls_local))
    serialized_events = []
    monkeypatch.setitem(
        first_worker_cls.initialize_worker.__globals__,
        "apply_fp8_patches",
        lambda _worker, config: serialized_events.append(("first", config)),
    )
    assert first_worker_cls().initialize_worker() == "initialized"
    monkeypatch.setitem(
        second_worker_cls.initialize_worker.__globals__,
        "apply_fp8_patches",
        lambda _worker, config: serialized_events.append(("second", config)),
    )
    assert second_worker_cls().initialize_worker() == "initialized"
    assert serialized_events == [("first", first_config), ("second", second_config)]


@pytest.mark.parametrize("model_parallel_size", [1, 2])
def test_run_engine_core_removes_serialized_fp8_config(
    fp8_module, monkeypatch, model_parallel_size
):
    fp8 = fp8_module
    fp8_config = types.SimpleNamespace(model_parallel_size=model_parallel_size)
    vllm_config = types.SimpleNamespace(nrl_fp8_cfg=fp8_config)
    applied_configs = []

    monkeypatch.setattr(
        fp8,
        "monkey_patch_vllm_ray_executor",
        lambda config: applied_configs.append(config),
    )
    monkeypatch.setattr(
        fp8,
        "original_run_engine_core",
        lambda **kwargs: kwargs["vllm_config"],
    )

    result = fp8.my_run_engine_core(vllm_config=vllm_config)

    assert result is vllm_config
    assert applied_configs == [fp8_config]
    assert not hasattr(vllm_config, "nrl_fp8_cfg")


@pytest.mark.parametrize("hidden_size", [32, 64])
def test_quantize_mxfp8_weight_restores_grouped_logical_shape(
    fp8_module, monkeypatch, hidden_size
):
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    weight = torch.empty(2, 3, hidden_size, dtype=torch.bfloat16)
    quantized_scale = torch.arange(6 * hidden_size // 32, dtype=torch.uint8).reshape(
        6, hidden_size // 32
    )

    def fake_quantize(tensor):
        assert tensor is weight
        return (
            torch.zeros(6, hidden_size, dtype=torch.float8_e4m3fn),
            quantized_scale,
        )

    monkeypatch.setattr(mxfp8_utils, "mxfp8_e4m3_quantize", fake_quantize)

    value, scale = fp8_module.quantize_mxfp8_weight(weight)

    assert value.shape == (2, 3, hidden_size)
    assert scale.shape == (2, 3, hidden_size // 32)
    torch.testing.assert_close(scale, quantized_scale.reshape(scale.shape))


@pytest.mark.parametrize(
    ("projection", "shape"),
    [("up_proj", (2, 3, 64)), ("down_proj", (2, 4, 32))],
)
def test_load_weights_preserves_grouped_mxfp8_value_and_scale_shapes(
    fp8_module, monkeypatch, projection, shape
):
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    fp8_module.global_fp8_config = types.SimpleNamespace(is_mx=True)
    monkeypatch.setattr(fp8_module, "_is_fp8_weight", lambda *_args: True)

    def fake_quantize(weight):
        flattened_rows = weight.numel() // weight.shape[-1]
        return (
            torch.zeros(
                flattened_rows,
                weight.shape[-1],
                dtype=torch.float8_e4m3fn,
            ),
            torch.zeros(
                flattened_rows,
                weight.shape[-1] // 32,
                dtype=torch.uint8,
            ),
        )

    monkeypatch.setattr(mxfp8_utils, "mxfp8_e4m3_quantize", fake_quantize)
    loaded = []
    model = types.SimpleNamespace(load_weights=lambda weights: loaded.extend(weights))
    name = f"model.layers.0.mixer.experts.0.{projection}.weight"

    fp8_module.load_weights(
        [(name, torch.empty(shape, dtype=torch.bfloat16))],
        types.SimpleNamespace(model=model),
    )

    assert [item[0] for item in loaded] == [
        name,
        f"{name}_scale_from_checkpoint",
    ]
    assert loaded[0][1].shape == shape
    assert loaded[1][1].shape == (*shape[:-1], shape[-1] // 32)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("is_gated", "intermediate_size", "hidden_size"),
    [
        (True, 128, 256),
        (True, 192, 128),
        (False, 128, 256),
    ],
)
def test_batched_moe_shuffle_matches_per_expert(
    fp8_module, monkeypatch, is_gated, intermediate_size, hidden_size
):
    pytest.importorskip("flashinfer")
    fp8 = fp8_module
    torch.manual_seed(0)
    num_experts = 4
    w13_rows = (2 if is_gated else 1) * intermediate_size

    def rand_bytes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    w13_weight = rand_bytes(num_experts, w13_rows, hidden_size).view(
        torch.float8_e4m3fn
    )
    w2_weight = rand_bytes(num_experts, hidden_size, intermediate_size).view(
        torch.float8_e4m3fn
    )
    w13_scale = rand_bytes(num_experts, w13_rows, hidden_size // 32)
    w2_scale = rand_bytes(num_experts, hidden_size, intermediate_size // 32)

    original_index_select = torch.index_select
    index_select_out_tensors = []

    def track_index_select(*args, **kwargs):
        index_select_out_tensors.append(kwargs.get("out"))
        return original_index_select(*args, **kwargs)

    monkeypatch.setattr(torch, "index_select", track_index_select)
    batched = fp8._shuffle_mxfp8_moe_batched(
        types.SimpleNamespace(),
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        is_gated,
        128,
    )
    monkeypatch.setattr(torch, "index_select", original_index_select)

    assert len(index_select_out_tensors) == 4
    assert all(tensor is None for tensor in index_select_out_tensors)

    reference = fp8._shuffle_mxfp8_moe_per_expert(
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        is_gated,
        128,
    )

    for actual, expected in zip(batched, reference):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("is_gated", [True, False])
def test_process_mxfp8_moe_refit_uses_batched_flashinfer_shuffle(
    fp8_module, monkeypatch, is_gated
):
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    fp8 = fp8_module
    fp8.global_fp8_config = fp8.FP8Config(
        use_fp8_weights=True,
        model_parallel_size=1,
        is_mx=True,
    )

    w13_rows = 256 if is_gated else 128
    w13_weight = torch.nn.Parameter(torch.zeros(2, w13_rows, 512), requires_grad=False)
    w2_weight = torch.nn.Parameter(torch.zeros(2, 512, 128), requires_grad=False)
    w13_scale = torch.nn.Parameter(torch.zeros(2, w13_rows, 16), requires_grad=False)
    w2_scale = torch.nn.Parameter(torch.zeros(2, 512, 4), requires_grad=False)
    w13_scale_from_checkpoint = torch.ones_like(w13_scale)
    w2_scale_from_checkpoint = torch.ones_like(w2_scale)
    layer = types.SimpleNamespace(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        w13_weight_scale_from_checkpoint=types.SimpleNamespace(
            data=w13_scale_from_checkpoint
        ),
        w2_weight_scale_from_checkpoint=types.SimpleNamespace(
            data=w2_scale_from_checkpoint
        ),
    )
    moe_kernel = object()
    moe_quant_config = object()
    quant_method = types.SimpleNamespace(
        moe=types.SimpleNamespace(is_act_and_mul=is_gated),
        moe_kernel=moe_kernel,
        moe_quant_config=moe_quant_config,
        mxfp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
    )
    shuffled = (
        torch.full_like(w13_weight, 1),
        torch.full_like(w2_weight, 2),
        torch.full_like(w13_scale, 3),
        torch.full_like(w2_scale, 4),
    )
    calls = []

    def batched_shuffle(*args):
        calls.append(("batched", args))
        return shuffled

    monkeypatch.setattr(fp8, "_shuffle_mxfp8_moe_batched", batched_shuffle)

    from vllm.model_executor.layers.quantization.utils import flashinfer_utils

    swap_calls = []

    def swap_w13_to_w31(tensor):
        swap_calls.append(tensor)
        return tensor

    monkeypatch.setattr(flashinfer_utils, "swap_w13_to_w31", swap_w13_to_w31)

    parameter_ids = tuple(
        id(parameter)
        for parameter in (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
    )
    storage_ptrs = tuple(
        parameter.data_ptr()
        for parameter in (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
    )

    fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    assert len(calls) == 1
    selected_path, args = calls[0]
    assert selected_path == "batched"
    assert args[0] is layer
    args = args[1:]
    assert args[0].data_ptr() == w13_weight.data_ptr()
    assert args[1].data_ptr() == w2_weight.data_ptr()
    assert args[2].data_ptr() == w13_scale_from_checkpoint.data_ptr()
    assert args[3].data_ptr() == w2_scale_from_checkpoint.data_ptr()
    assert args[4:] == (is_gated, 128)
    expected_swap_ptrs = (
        [w13_weight.data_ptr(), w13_scale_from_checkpoint.data_ptr()]
        if is_gated
        else []
    )
    assert [tensor.data_ptr() for tensor in swap_calls] == expected_swap_ptrs

    parameters = (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
    )
    assert tuple(id(parameter) for parameter in parameters) == parameter_ids
    assert tuple(parameter.data_ptr() for parameter in parameters) == storage_ptrs
    assert torch.equal(layer.w13_weight, shuffled[0])
    assert torch.equal(layer.w2_weight, shuffled[1])
    assert torch.equal(layer.w13_weight_scale, shuffled[2])
    assert torch.equal(layer.w2_weight_scale, shuffled[3])
    assert quant_method.moe_kernel is moe_kernel
    assert quant_method.moe_quant_config is moe_quant_config


def test_process_mxfp8_moe_refit_rejects_non_flashinfer_backend(fp8_module):
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    quant_method = types.SimpleNamespace(mxfp8_backend=Fp8MoeBackend.DEEPGEMM)

    with pytest.raises(
        NotImplementedError,
        match="MXFP8 MoE refit layout conversion only supports FLASHINFER_TRTLLM",
    ):
        fp8_module.process_weights_after_loading_mxfp8_moe(quant_method, object())


def test_process_mxfp8_moe_initializes_kernel_once(fp8_module, monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    fp8 = fp8_module
    fp8.global_fp8_config = fp8.FP8Config(
        use_fp8_weights=True,
        model_parallel_size=1,
        is_mx=True,
    )

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.zeros(2, 128, 512), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.zeros(2, 512, 128), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 128, 16), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 512, 4), requires_grad=False
    )
    layer.w13_weight_scale.weight_loader = object()
    layer.w2_weight_scale.weight_loader = object()
    layer._expert_routing_tables = lambda: (None, None, None)
    moe_config = types.SimpleNamespace(is_act_and_mul=False)
    quant_config = object()
    experts_cls = object()
    quant_config_calls = []

    def get_quant_config(_layer):
        quant_config_calls.append(_layer)
        return quant_config

    quant_method = types.SimpleNamespace(
        moe=moe_config,
        moe_kernel=None,
        mxfp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
        experts_cls=experts_cls,
        get_fused_moe_quant_config=get_quant_config,
    )
    kernel = object()
    kernel_calls = []
    shuffle_calls = []

    def shuffle(*args):
        shuffle_calls.append(args)
        fill = len(shuffle_calls)
        return tuple(torch.full_like(tensor, fill) for tensor in args[1:5])

    monkeypatch.setattr(fp8, "_shuffle_mxfp8_moe_batched", shuffle)

    from vllm.model_executor.layers.quantization import fp8 as vllm_fp8

    def make_kernel(**kwargs):
        kernel_calls.append(kwargs)
        return kernel

    monkeypatch.setattr(vllm_fp8, "make_fp8_moe_kernel", make_kernel)

    fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    runtime_parameters = (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
    )
    parameter_ids = tuple(id(parameter) for parameter in runtime_parameters)
    storage_ptrs = tuple(parameter.data_ptr() for parameter in runtime_parameters)

    layer.w13_weight_scale_from_checkpoint.data.fill_(2)
    layer.w2_weight_scale_from_checkpoint.data.fill_(2)
    fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    assert quant_method.moe_kernel is kernel
    assert quant_method.moe_quant_config is quant_config
    assert quant_config_calls == [layer]
    assert len(kernel_calls) == 1
    assert len(shuffle_calls) == 2
    refit_parameters = (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
    )
    assert tuple(id(parameter) for parameter in refit_parameters) == parameter_ids
    assert tuple(parameter.data_ptr() for parameter in refit_parameters) == storage_ptrs
    assert all(torch.all(parameter == 2) for parameter in refit_parameters)
    assert kernel_calls[0] == {
        "moe_quant_config": quant_config,
        "moe_config": moe_config,
        "fp8_backend": Fp8MoeBackend.FLASHINFER_TRTLLM,
        "experts_cls": experts_cls,
        "routing_tables": (None, None, None),
        "layer": layer,
    }


@pytest.mark.parametrize("is_gated", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_process_mxfp8_moe_padding_preserves_refit_tensors(
    fp8_module, monkeypatch, is_gated, tp_size
):
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    fp8 = fp8_module
    fp8.global_fp8_config = fp8.FP8Config(
        use_fp8_weights=True,
        model_parallel_size=1,
        is_mx=True,
    )

    def make_parameter(value):
        parameter = torch.nn.Parameter(value, requires_grad=False)
        parameter.weight_loader = lambda *_args, **_kwargs: None
        return parameter

    layer = torch.nn.Module()
    w13_rows = 64 if is_gated else 32
    w13 = torch.ones(1, w13_rows, 128)
    if is_gated:
        w13[:, 32:].fill_(5)
    layer.register_parameter("w13_weight", make_parameter(w13))
    layer.register_parameter("w2_weight", make_parameter(torch.ones(1, 128, 32)))
    layer.register_parameter(
        "w13_weight_scale",
        make_parameter(torch.full((1, w13_rows, 4), 2, dtype=torch.uint8)),
    )
    layer.register_parameter(
        "w2_weight_scale",
        make_parameter(torch.full((1, 128, 1), 2, dtype=torch.uint8)),
    )
    layer._expert_routing_tables = lambda: (None, None, None)

    moe_config = types.SimpleNamespace(
        is_act_and_mul=is_gated,
        hidden_dim=128,
        hidden_dim_unpadded=128,
        intermediate_size=32 * tp_size,
        intermediate_size_per_partition=32,
        intermediate_size_per_partition_unpadded=32,
        moe_parallel_config=types.SimpleNamespace(tp_size=tp_size),
    )
    kernel = object()
    kernel_configs = []
    quant_method = types.SimpleNamespace(
        moe=moe_config,
        moe_kernel=None,
        mxfp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
        experts_cls=types.SimpleNamespace(is_monolithic=lambda: True),
        get_fused_moe_quant_config=lambda _layer: object(),
    )

    def make_kernel(**kwargs):
        kernel_configs.append(kwargs["moe_config"])
        return kernel

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.make_fp8_moe_kernel",
        make_kernel,
    )
    monkeypatch.setattr(
        fp8,
        "_shuffle_mxfp8_moe_batched",
        lambda _layer, w13, w2, s13, s2, _gated, _tile: (w13, w2, s13, s2),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.utils.flashinfer_utils.swap_w13_to_w31",
        lambda value: value,
    )

    fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    assert tuple(layer.w13_weight.shape) == (1, w13_rows, 128)
    assert tuple(layer.w2_weight.shape) == (1, 128, 32)
    assert tuple(layer.w13_weight_scale_from_checkpoint.shape) == (1, w13_rows, 4)
    assert tuple(layer.w2_weight_scale_from_checkpoint.shape) == (1, 128, 1)
    expected_w13_rows = 256 if is_gated else 128
    assert tuple(layer.w13_weight_for_apply.shape) == (1, expected_w13_rows, 512)
    assert tuple(layer.w2_weight_for_apply.shape) == (1, 512, 128)
    assert tuple(layer.w13_weight_scale.shape) == (1, expected_w13_rows, 16)
    assert tuple(layer.w2_weight_scale.shape) == (1, 512, 4)
    assert torch.count_nonzero(layer.w13_weight_for_apply[:, :, 128:]) == 0
    assert torch.all(layer.w13_weight_scale[:, :, 4:] == 127)
    if is_gated:
        assert torch.count_nonzero(layer.w13_weight_for_apply[:, 32:128, :]) == 0
        assert torch.all(layer.w13_weight_for_apply[:, 128:160, :128] == 5)
        assert torch.count_nonzero(layer.w13_weight_for_apply[:, 160:, :]) == 0
        assert torch.all(layer.w13_weight_scale[:, 32:128, :] == 127)
        assert torch.all(layer.w13_weight_scale[:, 160:, :] == 127)
    else:
        assert torch.count_nonzero(layer.w13_weight_for_apply[:, 32:, :]) == 0
        assert torch.all(layer.w13_weight_scale[:, 32:, :] == 127)
    assert torch.count_nonzero(layer.w2_weight_for_apply[:, 128:, :]) == 0
    assert torch.count_nonzero(layer.w2_weight_for_apply[:, :, 32:]) == 0
    assert torch.all(layer.w2_weight_scale[:, 128:, :] == 127)
    assert torch.all(layer.w2_weight_scale[:, :, 1:] == 127)
    assert kernel_configs[0].hidden_dim == 512
    assert kernel_configs[0].intermediate_size_per_partition == 128
    assert kernel_configs[0].intermediate_size == 128 * tp_size

    x = torch.randn(2, 128)
    padded_x = torch.nn.functional.pad(x, (0, 512 - x.shape[-1]))
    if is_gated:
        reference_hidden = torch.nn.functional.silu(x @ w13[0, :32].T) * (
            x @ w13[0, 32:].T
        )
        padded_hidden = torch.nn.functional.silu(
            padded_x @ layer.w13_weight_for_apply[0, :128].T
        ) * (padded_x @ layer.w13_weight_for_apply[0, 128:].T)
    else:
        reference_hidden = torch.relu(x @ w13[0].T)
        padded_hidden = torch.relu(padded_x @ layer.w13_weight_for_apply[0].T)
    reference_output = reference_hidden @ layer.w2_weight[0].T
    padded_output = padded_hidden @ layer.w2_weight_for_apply[0].T
    torch.testing.assert_close(padded_output[:, :128], reference_output)
    assert torch.count_nonzero(padded_output[:, 128:]) == 0

    apply_parameter_ids = {
        name: id(getattr(layer, name))
        for name in (
            "w13_weight_for_apply",
            "w2_weight_for_apply",
            "w13_weight_scale",
            "w2_weight_scale",
        )
    }
    apply_storage_ptrs = {
        name: getattr(layer, name).data_ptr() for name in apply_parameter_ids
    }
    with torch.no_grad():
        layer.w13_weight.fill_(3)
        layer.w2_weight.fill_(3)
        layer.w13_weight_scale_from_checkpoint.fill_(4)
        layer.w2_weight_scale_from_checkpoint.fill_(4)

    fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    assert all(
        id(getattr(layer, name)) == parameter_id
        for name, parameter_id in apply_parameter_ids.items()
    )
    assert all(
        getattr(layer, name).data_ptr() == data_ptr
        for name, data_ptr in apply_storage_ptrs.items()
    )
    assert torch.all(layer.w13_weight_for_apply[:, :32, :128] == 3)
    assert torch.all(layer.w2_weight_for_apply[:, :128, :32] == 3)
    assert torch.all(layer.w13_weight_scale[:, :32, :4] == 4)
    assert torch.all(layer.w2_weight_scale[:, :128, :1] == 4)
    assert torch.count_nonzero(layer.w13_weight_for_apply[:, :, 128:]) == 0
    assert torch.all(layer.w13_weight_scale[:, :, 4:] == 127)
    if is_gated:
        assert torch.count_nonzero(layer.w13_weight_for_apply[:, 32:128, :]) == 0
        assert torch.all(layer.w13_weight_for_apply[:, 128:160, :128] == 3)
        assert torch.count_nonzero(layer.w13_weight_for_apply[:, 160:, :]) == 0
        assert torch.all(layer.w13_weight_scale[:, 32:128, :] == 127)
        assert torch.all(layer.w13_weight_scale[:, 128:160, :4] == 4)
        assert torch.all(layer.w13_weight_scale[:, 160:, :] == 127)
    else:
        assert torch.count_nonzero(layer.w13_weight_for_apply[:, 32:, :]) == 0
        assert torch.all(layer.w13_weight_scale[:, 32:, :] == 127)
    assert torch.count_nonzero(layer.w2_weight_for_apply[:, 128:, :]) == 0
    assert torch.count_nonzero(layer.w2_weight_for_apply[:, :, 32:]) == 0
    assert torch.all(layer.w2_weight_scale[:, 128:, :] == 127)
    assert torch.all(layer.w2_weight_scale[:, :, 1:] == 127)
    assert quant_method.moe_kernel is kernel
    assert len(kernel_configs) == 1


def test_process_mxfp8_moe_padding_rejects_modular_kernel(fp8_module, monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    fp8 = fp8_module
    layer = torch.nn.Module()
    for name, value in (
        ("w13_weight", torch.ones(1, 32, 128)),
        ("w2_weight", torch.ones(1, 128, 32)),
        ("w13_weight_scale", torch.ones(1, 32, 4, dtype=torch.uint8)),
        ("w2_weight_scale", torch.ones(1, 128, 1, dtype=torch.uint8)),
    ):
        parameter = torch.nn.Parameter(value, requires_grad=False)
        parameter.weight_loader = lambda *_args, **_kwargs: None
        layer.register_parameter(name, parameter)

    method = types.SimpleNamespace(
        moe=types.SimpleNamespace(is_act_and_mul=False),
        moe_kernel=None,
        mxfp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
        experts_cls=types.SimpleNamespace(is_monolithic=lambda: False),
    )
    monkeypatch.setattr(
        fp8,
        "_shuffle_mxfp8_moe_batched",
        lambda _layer, w13, w2, s13, s2, _gated, _tile: (w13, w2, s13, s2),
    )

    with pytest.raises(NotImplementedError, match="requires a monolithic kernel"):
        fp8.process_weights_after_loading_mxfp8_moe(method, layer)


@pytest.mark.parametrize("requires_padding", [False, True])
def test_apply_monolithic_mxfp8_moe_uses_padded_apply_weights(
    fp8_module, requires_padding
):
    fp8 = fp8_module
    calls = []

    class Kernel:
        def apply_monolithic(
            self,
            x,
            w13,
            w2,
            router_logits,
            *,
            activation,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            routed_scaling_factor,
        ):
            kwargs = {
                "activation": activation,
                "global_num_experts": global_num_experts,
                "expert_map": expert_map,
                "apply_router_weight_on_input": apply_router_weight_on_input,
                "num_expert_group": num_expert_group,
                "topk_group": topk_group,
                "e_score_correction_bias": e_score_correction_bias,
                "routed_scaling_factor": routed_scaling_factor,
            }
            calls.append((x, w13, w2, router_logits, kwargs))
            return x + 1

    method = types.SimpleNamespace(is_monolithic=True, moe_kernel=Kernel())
    layer_kwargs = {
        "w13_weight": torch.tensor([130]),
        "w2_weight": torch.tensor([20]),
        "activation": "relu2",
        "global_num_experts": 4,
        "expert_map": "expert-map",
        "apply_router_weight_on_input": True,
        "num_expert_group": 8,
        "topk_group": 2,
        "e_score_correction_bias": "correction-bias",
        "routed_scaling_factor": 1.25,
    }
    if requires_padding:
        layer_kwargs.update(
            {
                "mxfp8_unpadded_hidden_size": 2688,
                "mxfp8_padded_hidden_size": 3072,
                "w13_weight_for_apply": torch.tensor([13]),
                "w2_weight_for_apply": torch.tensor([2]),
            }
        )
    layer = types.SimpleNamespace(**layer_kwargs)
    x = torch.arange(2 * 2688, dtype=torch.float32).reshape(2, 2688)
    router_logits = torch.ones(2, 4)

    output = fp8.apply_monolithic_mxfp8_moe(method, layer, x, router_logits)

    padded_x, w13, w2, actual_logits, _kwargs = calls[0]
    expected_hidden_size = 3072 if requires_padding else 2688
    assert tuple(padded_x.shape) == (2, expected_hidden_size)
    torch.testing.assert_close(padded_x[:, :2688], x)
    if requires_padding:
        assert torch.count_nonzero(padded_x[:, 2688:]) == 0
        assert w13 is layer.w13_weight_for_apply
        assert w2 is layer.w2_weight_for_apply
    else:
        assert w13 is layer.w13_weight
        assert w2 is layer.w2_weight
    assert actual_logits is router_logits
    assert _kwargs == {
        "activation": "relu2",
        "global_num_experts": 4,
        "expert_map": "expert-map",
        "apply_router_weight_on_input": True,
        "num_expert_group": 8,
        "topk_group": 2,
        "e_score_correction_bias": "correction-bias",
        "routed_scaling_factor": 1.25,
    }
    assert tuple(output.shape) == (2, 2688)
    torch.testing.assert_close(output, x + 1)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("pow2_weight_scaling_factors", "only pow2 weight scaling factors"),
        ("pow2_activation_scaling_factors", "only pow2 activation scaling factors"),
    ],
)
def test_init_fp8_rejects_non_pow2_mxfp8_scales(fp8_module, monkeypatch, field, error):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    with pytest.raises(ValueError, match=error):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                "refit_with_reload_api": False,
                field: False,
            },
            "dummy-model",
            model_parallel_size=1,
        )


def test_apply_fp8_patches_registers_modelopt_patches_only_for_mxfp8(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    patched_calls = []

    class FakePatch:
        def __init__(self, path):
            self.path = path
            self.started = False

        def start(self):
            self.started = True

    def fake_patch(path, replacement):
        patched_calls.append((path, replacement))
        return FakePatch(path)

    monkeypatch.setattr(fp8, "patch", fake_patch)

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=False),
    )
    assert not any("ModelOptMxFp8" in path for path, _replacement in patched_calls)
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_calls.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(
            use_fp8_weights=True,
            model_parallel_size=1,
            use_activation_pow2_scale=True,
        ),
    )
    assert any(
        "per_token_group_quant_fp8" in path for path, _replacement in patched_calls
    )
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_calls.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=True),
    )

    expected_mxfp8_calls = [
        (
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptMxFp8LinearMethod.process_weights_after_loading",
            fp8.process_weights_after_loading_mxfp8_linear,
        ),
        (
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptMxFp8FusedMoE.create_weights",
            fp8.create_weights_mxfp8_moe,
        ),
        (
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptMxFp8FusedMoE.process_weights_after_loading",
            fp8.process_weights_after_loading_mxfp8_moe,
        ),
        (
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptMxFp8FusedMoE.apply_monolithic",
            fp8.apply_monolithic_mxfp8_moe,
        ),
    ]
    actual_mxfp8_calls = [call for call in patched_calls if "ModelOptMxFp8" in call[0]]
    assert actual_mxfp8_calls == expected_mxfp8_calls
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)


def test_process_weights_after_loading_copies_in_place_on_refit(monkeypatch):
    """Refit runs this every step; rebinding .data each time fragments memory.

    Regression guard for the CuMemAllocator wake-up OOM (~75 steps into the
    fp8-rollouts nightlies): the 0.25 port rebound weight/weight_scale_inv to
    fresh allocations on every call, where 0.20 copied in place. Nothing in the
    suite pinned that, so a refactor back to .data rebinding would have
    produced no test failure -- just a slow OOM in a nightly days later.
    """
    import torch
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    from nemo_rl.models.generation.vllm.quantization import fp8

    layer = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False),
    )
    # Same shape/dtype back, but a *fresh* tensor each call -- exactly what the
    # real helper returns once the processed layout is stable.
    monkeypatch.setattr(
        fp8_utils,
        "process_fp8_weight_block_strategy",
        lambda w, s: (torch.ones_like(w), torch.ones_like(s)),
    )
    monkeypatch.setattr(fp8, "maybe_post_process_fp8_weight_block", lambda _layer: None)

    method = types.SimpleNamespace(
        block_quant=True,
        quant_config=types.SimpleNamespace(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        ),
    )

    weight_ptr = layer.weight.data.data_ptr()
    scale_ptr = layer.weight_scale_inv.data.data_ptr()
    weight_param, scale_param = layer.weight, layer.weight_scale_inv

    for _ in range(3):  # initial load + two refits
        fp8.process_weights_after_loading(method, layer)

    assert layer.weight.data.data_ptr() == weight_ptr, (
        "weight storage was rebound instead of copied in place; on a real refit "
        "this leaks a fresh allocation every step until wake_up OOMs"
    )
    assert layer.weight_scale_inv.data.data_ptr() == scale_ptr, (
        "weight_scale_inv storage was rebound instead of copied in place"
    )
    # Parameter identity (and therefore weight_loader) must also survive.
    assert layer.weight is weight_param
    assert layer.weight_scale_inv is scale_param
    # The processed values must actually land.
    assert torch.equal(layer.weight.data, torch.ones(4, 4))


def test_mxfp8_load_weights_routes_moe_weights_to_checkpoint_params(
    fp8_module, monkeypatch
):
    import torch
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    fp8 = fp8_module

    class FakeRoutedExperts:
        pass

    captured_weights = []

    def capture_load_weights(weights):
        captured_weights.extend(weights)

    def fake_mxfp8_e4m3_quantize(weight):
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.ones(*weight.shape[:-1], 1, dtype=torch.uint8),
        )

    fake_model = types.SimpleNamespace(load_weights=capture_load_weights)
    fake_runner = types.SimpleNamespace(
        model=fake_model,
        vllm_config=types.SimpleNamespace(),
    )

    monkeypatch.setattr(fp8, "RoutedExperts", FakeRoutedExperts)
    fp8.global_fp8_config = fp8.FP8Config(is_mx=True)
    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    monkeypatch.setattr(
        fp8,
        "_get_module_from_param_name",
        lambda _model, _name: FakeRoutedExperts(),
    )
    monkeypatch.setattr(mxfp8_utils, "mxfp8_e4m3_quantize", fake_mxfp8_e4m3_quantize)

    fp8.load_weights(
        [("model.layers.0.mlp.experts.w13_weight", torch.zeros(2, 32, 32))],
        fake_runner,
    )

    assert [name for name, _ in captured_weights] == [
        "model.layers.0.mlp.experts.w13_weight_from_checkpoint",
        "model.layers.0.mlp.experts.w13_weight_scale_from_checkpoint",
    ]


def test_mxfp8_reload_iterator_emits_upstream_checkpoint_names(fp8_module, monkeypatch):
    import torch
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    fp8 = fp8_module

    def fake_mxfp8_e4m3_quantize(weight):
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.ones(*weight.shape[:-1], 1, dtype=torch.uint8),
        )

    fake_runner = types.SimpleNamespace(
        model=object(),
        vllm_config=types.SimpleNamespace(),
    )
    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    fp8.global_fp8_config = fp8.FP8Config(is_mx=True)
    monkeypatch.setattr(mxfp8_utils, "mxfp8_e4m3_quantize", fake_mxfp8_e4m3_quantize)

    quantized = list(
        fp8.get_quantized_weight_iterator(
            [("model.layers.0.mlp.experts.w13_weight", torch.zeros(2, 32, 32))],
            fake_runner,
        )
    )

    assert [name for name, _ in quantized] == [
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w13_weight_scale",
    ]
    assert quantized[0][1].dtype == torch.float8_e4m3fn
    assert quantized[1][1].dtype == torch.uint8
