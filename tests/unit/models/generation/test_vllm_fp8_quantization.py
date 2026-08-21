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
from typing import Any

import cloudpickle
import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.fixture()
def fp8_module():
    pytest.importorskip("vllm")

    from nemo_rl.models.generation.vllm.quantization import fp8

    old_config = fp8.global_fp8_config
    old_state = fp8.fp8_state
    old_patches_applied = fp8.fp8_patches_applied
    fp8.global_fp8_config = None
    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False

    try:
        yield fp8
    finally:
        fp8.global_fp8_config = old_config
        fp8.fp8_state = old_state
        fp8.fp8_patches_applied = old_patches_applied


def test_init_fp8_uses_mxfp8_quantization_config(fp8_module, monkeypatch):
    fp8 = fp8_module
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
            "async_engine": False,
            "is_mx": True,
            "refit_cache_loader_routes": True,
            "use_deep_gemm": True,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    assert vllm_kwargs == {
        "quantization": "fp8",
        "kv_cache_dtype": "auto",
        "hf_overrides": {
            "quantization_config": {
                **fp8.MXFP8_BLOCK_QUANT_KWARGS,
                "ignored_layers": ["lm_head"],
                "ignore": ["lm_head"],
            }
        },
    }
    assert applied_configs == [fp8.global_fp8_config]
    assert fp8.global_fp8_config.is_mx is True
    assert "VLLM_USE_DEEP_GEMM" not in fp8.os.environ
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in fp8.os.environ


def test_ray_executor_v2_worker_applies_fp8_patches_before_model_load(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    monkeypatch.setattr(fp8, "_test_applied_configs", [], raising=False)
    config = fp8.FP8Config(
        use_fp8_weights=True,
        model_parallel_size=2,
        is_mx=True,
    )

    class FakeRayWorkerProc:
        def initialize_worker(
            self,
            local_rank,
            env_vars,
            driver_env_vars=None,
            assigned_physical_gpu_ids=None,
        ):
            assert fp8.fp8_patches_applied
            return (
                local_rank,
                env_vars,
                driver_env_vars,
                assigned_physical_gpu_ids,
            )

    def fake_apply_fp8_patches(_self, fp8_config):
        fp8._test_applied_configs.append(fp8_config)
        fp8.fp8_patches_applied = True

    monkeypatch.setattr(fp8, "apply_fp8_patches", fake_apply_fp8_patches)
    ray_executor_v2 = types.SimpleNamespace(RayWorkerProc=FakeRayWorkerProc)
    fp8._patch_ray_executor_v2_worker(ray_executor_v2, config)
    patched_worker_cls = cloudpickle.loads(
        cloudpickle.dumps(ray_executor_v2.RayWorkerProc)
    )

    result = patched_worker_cls().initialize_worker(
        1,
        {"WORKER_ENV": "1"},
        {"DRIVER_ENV": "1"},
        assigned_physical_gpu_ids=[2, 3],
    )

    assert fp8._test_applied_configs == [config]
    assert result == (
        1,
        {"WORKER_ENV": "1"},
        {"DRIVER_ENV": "1"},
        [2, 3],
    )


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
    assert all(tensor is not None for tensor in index_select_out_tensors)

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


@pytest.mark.parametrize(
    ("env_value", "expected_path"),
    [(None, "batched"), ("1", "batched"), ("0", "per-expert")],
)
def test_mxfp8_moe_shuffle_ablation_selector(
    fp8_module, monkeypatch, env_value, expected_path
):
    fp8 = fp8_module
    calls = []
    shuffled = tuple(torch.tensor(index) for index in range(4))

    def batched(*args):
        calls.append(("batched", args))
        return shuffled

    def per_expert(*args):
        calls.append(("per-expert", args))
        return shuffled

    monkeypatch.setattr(fp8, "_shuffle_mxfp8_moe_batched", batched)
    monkeypatch.setattr(fp8, "_shuffle_mxfp8_moe_per_expert", per_expert)
    if env_value is None:
        monkeypatch.delenv("NRL_MXFP8_BATCHED_SHUFFLE", raising=False)
    else:
        monkeypatch.setenv("NRL_MXFP8_BATCHED_SHUFFLE", env_value)

    args = (
        object(),
        torch.tensor(1),
        torch.tensor(2),
        torch.tensor(3),
        torch.tensor(4),
        True,
        128,
    )
    assert fp8._shuffle_mxfp8_moe(*args) == shuffled
    assert len(calls) == 1
    selected_path, selected_args = calls[0]
    assert selected_path == expected_path
    if expected_path == "batched":
        assert selected_args == args
    else:
        assert selected_args == args[1:]


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

    hidden_size = 512
    intermediate_size = 128
    w13_rows = intermediate_size * (2 if is_gated else 1)
    w13_weight = torch.nn.Parameter(
        torch.zeros(2, w13_rows, hidden_size, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    w2_weight = torch.nn.Parameter(
        torch.zeros(2, hidden_size, intermediate_size, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    w13_scale = torch.nn.Parameter(
        torch.zeros(2, w13_rows * (hidden_size // 32), dtype=torch.uint8),
        requires_grad=False,
    )
    w2_scale = torch.nn.Parameter(
        torch.zeros(2, hidden_size * (intermediate_size // 32), dtype=torch.uint8),
        requires_grad=False,
    )
    w13_scale_from_checkpoint = torch.ones(
        2, w13_rows, hidden_size // 32, dtype=torch.uint8
    )
    w2_scale_from_checkpoint = torch.ones(
        2, hidden_size, intermediate_size // 32, dtype=torch.uint8
    )
    moe_config = types.SimpleNamespace(
        is_act_and_mul=is_gated,
        intermediate_size_per_partition=intermediate_size,
    )
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
        moe_config=moe_config,
    )
    moe_kernel = object()
    moe_quant_config = types.SimpleNamespace(
        w1_scale=w13_scale,
        w2_scale=w2_scale,
    )
    quant_method = types.SimpleNamespace(
        moe=moe_config,
        moe_kernel=moe_kernel,
        moe_quant_config=moe_quant_config,
        mxfp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
        experts_cls=types.SimpleNamespace(is_monolithic=lambda: True),
        weight_block_size=[32, 32],
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
    layer.w13_weight = torch.nn.Parameter(torch.zeros(2, 4, 3), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.zeros(2, 3, 2), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 4, 1), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 3, 1), requires_grad=False
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

    from vllm.model_executor import parameter as vllm_parameter
    from vllm.model_executor.layers.quantization import fp8 as vllm_fp8

    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )

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
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _fp8_config: None)

    with pytest.raises(ValueError, match=error):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                "refit_cache_loader_routes": False,
                field: False,
            },
            "dummy-model",
            model_parallel_size=1,
        )


def test_apply_fp8_patches_registers_modelopt_patches_only_for_mxfp8(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    patched_paths = []

    class FakePatch:
        def __init__(self, path):
            self.path = path
            self.started = False

        def start(self):
            self.started = True

    def fake_patch(path, _replacement):
        patched_paths.append(path)
        return FakePatch(path)

    monkeypatch.setattr(fp8, "patch", fake_patch)

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=False),
    )
    assert not any("ModelOptMxFp8" in path for path in patched_paths)
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_paths.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(
            use_fp8_weights=True,
            model_parallel_size=1,
            use_activation_pow2_scale=True,
        ),
    )
    assert any("per_token_group_quant_fp8" in path for path in patched_paths)
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_paths.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=True),
    )

    assert any("ModelOptMxFp8LinearMethod" in path for path in patched_paths)
    assert any("ModelOptMxFp8FusedMoE.create_weights" in path for path in patched_paths)
    assert any(
        "ModelOptMxFp8FusedMoE.process_weights_after_loading" in path
        for path in patched_paths
    )
    assert any(
        "ModelOptMxFp8FusedMoE.apply_monolithic" in path for path in patched_paths
    )
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)


def test_get_module_from_param_name_resolves_vllm_025_routed_experts(
    fp8_module: types.ModuleType,
) -> None:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    fp8 = fp8_module
    routed_experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = routed_experts
    model = types.SimpleNamespace(packed_modules_mapping={}, experts=runner)

    assert (
        fp8._get_module_from_param_name(model, "experts.w13_weight") is routed_experts
    )


def test_load_weights_preserves_prequantized_mxfp8_and_clamps_scales(
    fp8_module, monkeypatch
):
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    from nemo_rl.models.generation.vllm import vllm_backend

    fp8 = fp8_module
    fp8.global_fp8_config = types.SimpleNamespace(
        is_mx=True,
        refit_prequantize=True,
    )
    native = torch.ones(2, 2, dtype=torch.bfloat16)
    prequantized = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    prequantized_scales = torch.ones(2, 1, dtype=torch.uint8)
    receiver_quantized = torch.full((2, 64), 2.0, dtype=torch.bfloat16)
    receiver_fp8 = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    receiver_scales = torch.tensor([[0, 7], [3, 0]], dtype=torch.uint8)
    loaded = []

    monkeypatch.setattr(
        fp8,
        "_is_fp8_weight",
        lambda name, _model: name.endswith(".weight"),
    )
    monkeypatch.setattr(
        mxfp8_utils,
        "mxfp8_e4m3_quantize",
        lambda tensor, **_kwargs: (
            (
                receiver_fp8,
                receiver_scales,
            )
            if tensor is receiver_quantized
            else pytest.fail("unexpected receiver quantization input")
        ),
    )
    monkeypatch.setattr(
        vllm_backend,
        "load_weights_maybe_cached",
        lambda model, weights, *, cache_loader_routes: loaded.extend(weights),
    )
    model = object()

    fp8.load_weights(
        [
            ("model.native", native),
            ("model.prequantized.weight", prequantized),
            (
                "model.prequantized.weight_scale_from_checkpoint",
                prequantized_scales,
            ),
            ("model.receiver.weight", receiver_quantized),
        ],
        types.SimpleNamespace(
            model=model,
            vllm_config=types.SimpleNamespace(additional_config={}),
        ),
    )

    assert loaded[0][0] == "model.native"
    assert loaded[0][1] is native
    assert loaded[1][0] == "model.prequantized.weight"
    assert loaded[1][1] is prequantized
    assert loaded[2][0] == "model.prequantized.weight_scale_from_checkpoint"
    assert loaded[2][1] is prequantized_scales
    assert loaded[3][0] == "model.receiver.weight"
    assert loaded[3][1] is receiver_fp8
    assert loaded[4][0] == "model.receiver.weight_scale_from_checkpoint"
    torch.testing.assert_close(
        loaded[4][1],
        torch.tensor([[1, 7], [3, 1]], dtype=torch.uint8),
    )


def test_load_weights_rejects_unnegotiated_mxfp8_payload(fp8_module, monkeypatch):
    fp8 = fp8_module
    fp8.global_fp8_config = types.SimpleNamespace(
        is_mx=True,
        refit_prequantize=False,
    )
    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)

    with pytest.raises(ValueError, match="refit_prequantize=true"):
        fp8.load_weights(
            [("model.weight", torch.ones(2, 2, dtype=torch.float8_e4m3fn))],
            types.SimpleNamespace(
                model=object(),
                vllm_config=types.SimpleNamespace(additional_config={}),
            ),
        )


def test_load_weights_accepts_prequantized_mxfp8_chunk_before_scale(
    fp8_module, monkeypatch
):
    from nemo_rl.models.generation.vllm import vllm_backend

    fp8 = fp8_module
    fp8.global_fp8_config = types.SimpleNamespace(
        is_mx=True,
        refit_prequantize=True,
    )
    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    loaded = []
    monkeypatch.setattr(
        vllm_backend,
        "load_weights_maybe_cached",
        lambda model, weights, *, cache_loader_routes: loaded.extend(weights),
    )
    weight = torch.ones(2, 2, dtype=torch.float8_e4m3fn)

    fp8.load_weights(
        [("model.weight", weight)],
        types.SimpleNamespace(
            model=object(),
            vllm_config=types.SimpleNamespace(additional_config={}),
        ),
    )

    assert len(loaded) == 1
    assert loaded[0][0] == "model.weight"
    assert loaded[0][1] is weight


def test_load_weights_preserves_non_mx_blockwise_fp8_payload(fp8_module, monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    fp8 = fp8_module
    fp8.global_fp8_config = types.SimpleNamespace(
        is_mx=False,
        refit_prequantize=False,
    )
    weight = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32)
    loaded = []
    monkeypatch.setattr(
        fp8,
        "_is_fp8_weight",
        lambda name, _model: name == "model.weight",
    )
    monkeypatch.setattr(
        vllm_backend,
        "load_weights_maybe_cached",
        lambda model, weights, *, cache_loader_routes: loaded.extend(weights),
    )

    fp8.load_weights(
        [("model.weight", weight), ("model.weight_scale_inv", scale)],
        types.SimpleNamespace(
            model=object(),
            vllm_config=types.SimpleNamespace(additional_config={}),
        ),
    )

    assert loaded[0][0] == "model.weight"
    assert loaded[0][1] is weight
    assert loaded[1][0] == "model.weight_scale_inv"
    assert loaded[1][1] is scale


def test_mxfp8_padding_helpers_preserve_values_and_fill_padding(
    fp8_module: types.ModuleType,
) -> None:
    fp8 = fp8_module
    tensor = torch.arange(12).reshape(2, 2, 3)

    assert fp8._round_up(1856, 128) == 1920
    assert fp8._pad_tensor_dim(tensor, 1, 2) is tensor

    padded = fp8._pad_tensor_dim(tensor, 1, 4, pad_value=7)
    torch.testing.assert_close(padded[:, :2], tensor)
    torch.testing.assert_close(padded[:, 2:], torch.full((2, 2, 3), 7))

    w13 = torch.arange(24).reshape(2, 4, 3)
    assert fp8._pad_w13_shards(w13, 2, 2) is w13

    padded_w13 = fp8._pad_w13_shards(w13, 2, 3, pad_value=9)
    expected_w13 = torch.tensor(
        [
            [[0, 1, 2], [3, 4, 5], [9, 9, 9], [6, 7, 8], [9, 10, 11], [9, 9, 9]],
            [
                [12, 13, 14],
                [15, 16, 17],
                [9, 9, 9],
                [18, 19, 20],
                [21, 22, 23],
                [9, 9, 9],
            ],
        ]
    )
    torch.testing.assert_close(padded_w13, expected_w13)
    torch.testing.assert_close(
        fp8._clamp_mxfp8_scale(torch.tensor([0, 2, 0], dtype=torch.uint8)),
        torch.tensor([1, 2, 1], dtype=torch.uint8),
    )


def test_set_mxfp8_apply_tensor_reuses_matching_storage(
    fp8_module: types.ModuleType,
) -> None:
    fp8 = fp8_module
    layer = torch.nn.Module()

    fp8._set_mxfp8_apply_tensor(layer, "weight_for_apply", torch.ones(2, 3))
    first = layer.weight_for_apply
    first_data_ptr = first.data_ptr()

    fp8._set_mxfp8_apply_tensor(layer, "weight_for_apply", torch.full((2, 3), 4.0))

    assert layer.weight_for_apply is first
    assert layer.weight_for_apply.data_ptr() == first_data_ptr
    torch.testing.assert_close(layer.weight_for_apply, torch.full((2, 3), 4.0))


def test_process_mxfp8_moe_pads_kernel_tensors_without_changing_checkpoint_layout(
    fp8_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    fp8 = fp8_module
    captured: dict[str, Any] = {}
    kernel_builds = 0

    def fake_make_quant_config(**kwargs: Any) -> Any:
        captured["quant_config_kwargs"] = kwargs
        return types.SimpleNamespace(
            w1_scale=kwargs["w1_scale"],
            w2_scale=kwargs["w2_scale"],
        )

    def fake_make_kernel(**kwargs: Any) -> Any:
        nonlocal kernel_builds
        kernel_builds += 1
        captured["kernel_kwargs"] = kwargs
        return types.SimpleNamespace()

    def fake_batched_shuffle(
        layer: torch.nn.Module,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        is_gated: bool,
        epilogue_tile_m: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        captured.update(
            {
                "layer": layer,
                "w13_weight": w13_weight,
                "w2_weight": w2_weight,
                "w13_scale": w13_scale,
                "w2_scale": w2_scale,
                "is_gated": is_gated,
                "epilogue_tile_m": epilogue_tile_m,
            }
        )
        return tuple(
            tensor.clone() for tensor in (w13_weight, w2_weight, w13_scale, w2_scale)
        )

    monkeypatch.setattr(fp8, "_shuffle_mxfp8_moe_batched", fake_batched_shuffle)
    monkeypatch.setattr(fp8_oracle, "make_fp8_moe_quant_config", fake_make_quant_config)
    monkeypatch.setattr(fp8_oracle, "make_fp8_moe_kernel", fake_make_kernel)
    fp8.global_fp8_config = fp8.FP8Config()

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(30, dtype=torch.float32).reshape(2, 3, 5),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(30, dtype=torch.float32).reshape(2, 5, 3),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 3, 1, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 5, 1, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w13_weight_scale_from_checkpoint = torch.nn.Parameter(
        torch.zeros(2, 3, 1, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight_scale_from_checkpoint = torch.nn.Parameter(
        torch.zeros(2, 5, 1, dtype=torch.uint8),
        requires_grad=False,
    )
    moe_config = types.SimpleNamespace(
        intermediate_size_per_partition=3,
        is_act_and_mul=False,
    )
    layer.moe_config = moe_config
    layer._expert_routing_tables = lambda: None
    quant_method = types.SimpleNamespace(
        mxfp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
        experts_cls=types.SimpleNamespace(is_monolithic=lambda: True),
        weight_block_size=[1, 32],
        moe=moe_config,
        moe_kernel=None,
        moe_quant_config=None,
    )
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    assert captured["layer"] is layer
    assert captured["is_gated"] is False
    assert captured["epilogue_tile_m"] == 128
    assert captured["w13_weight"].shape == (2, 128, 512)
    assert captured["w2_weight"].shape == (2, 512, 128)
    assert captured["w13_scale"].shape == (2, 128, 16)
    assert captured["w2_scale"].shape == (2, 512, 4)
    assert torch.count_nonzero(captured["w13_scale"] == 0) == 0
    assert torch.count_nonzero(captured["w2_scale"] == 0) == 0

    torch.testing.assert_close(layer.w13_weight, original_w13)
    torch.testing.assert_close(layer.w2_weight, original_w2)
    assert layer.mxfp8_unpadded_hidden_size == 5
    assert layer.mxfp8_padded_hidden_size == 512
    assert layer.mxfp8_unpadded_intermediate_size_per_partition == 3
    assert layer.mxfp8_padded_intermediate_size_per_partition == 128
    assert layer.intermediate_size_per_partition == 128
    assert layer.moe_config.intermediate_size_per_partition == 128
    assert quant_method.moe.intermediate_size_per_partition == 128
    assert layer.w13_weight_for_apply.shape == (2, 128, 512)
    assert layer.w2_weight_for_apply.shape == (2, 512, 128)
    assert layer.w13_scale_for_apply.shape == (2, 128, 16)
    assert layer.w2_scale_for_apply.shape == (2, 512, 4)
    assert layer.weight_block_size == [1, 32]
    assert captured["quant_config_kwargs"]["w1_scale"] is layer.w13_scale_for_apply
    assert captured["quant_config_kwargs"]["w2_scale"] is layer.w2_scale_for_apply
    assert captured["kernel_kwargs"]["moe_config"] is moe_config
    assert captured["kernel_kwargs"]["routing_tables"] is None
    assert kernel_builds == 1

    w13_scale_for_apply = layer.w13_scale_for_apply
    w2_scale_for_apply = layer.w2_scale_for_apply
    fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    assert layer.w13_scale_for_apply is w13_scale_for_apply
    assert layer.w2_scale_for_apply is w2_scale_for_apply
    assert quant_method.moe_quant_config.w1_scale is w13_scale_for_apply
    assert quant_method.moe_quant_config.w2_scale is w2_scale_for_apply
    assert kernel_builds == 1


def test_process_mxfp8_moe_rejects_non_trtllm_backend_before_mutation(
    fp8_module: types.ModuleType,
) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    fp8 = fp8_module
    quant_method = types.SimpleNamespace(
        mxfp8_backend=Fp8MoeBackend.DEEPGEMM,
        experts_cls=types.SimpleNamespace(is_monolithic=lambda: True),
    )
    layer = types.SimpleNamespace(marker=object())

    with pytest.raises(
        NotImplementedError,
        match="requires the monolithic FlashInfer TRTLLM backend",
    ):
        fp8.process_weights_after_loading_mxfp8_moe(quant_method, layer)

    assert not hasattr(layer, "weight_block_size")


def test_apply_monolithic_mxfp8_moe_uses_vllm_025_moe_config(
    fp8_module: types.ModuleType,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    fp8 = fp8_module
    captured: dict[str, Any] = {}

    def fake_apply(
        x: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        captured.update(
            {
                "x": x,
                "w13_weight": w13_weight,
                "w2_weight": w2_weight,
                "router_logits": router_logits,
            }
        )
        captured.update(kwargs)
        return torch.zeros_like(x, dtype=torch.bfloat16)

    kernel = types.SimpleNamespace(apply_monolithic=fake_apply)
    quant_method = types.SimpleNamespace(
        is_monolithic=True,
        moe_kernel=kernel,
    )
    runtime_w13 = torch.empty(4, 128, 512, dtype=torch.float8_e4m3fn)
    runtime_w2 = torch.empty(4, 512, 128, dtype=torch.float8_e4m3fn)
    layer = types.SimpleNamespace(
        activation=MoEActivation.RELU2_NO_MUL,
        global_num_experts=32,
        expert_map=None,
        apply_router_weight_on_input=False,
        num_expert_group=0,
        topk_group=0,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        w13_weight=torch.empty(4, 128, 512, dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty(4, 512, 128, dtype=torch.float8_e4m3fn),
        w13_weight_for_apply=runtime_w13,
        w2_weight_for_apply=runtime_w2,
        mxfp8_padded_hidden_size=512,
    )
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    router_logits = torch.zeros(2, 32, dtype=torch.bfloat16)

    output = fp8.apply_monolithic_mxfp8_moe(
        quant_method,
        layer,
        x,
        router_logits,
    )

    assert captured["x"].shape == (2, 512)
    assert captured["w13_weight"] is runtime_w13
    assert captured["w2_weight"] is runtime_w2
    assert captured["router_logits"] is router_logits
    assert captured["activation"] == MoEActivation.RELU2_NO_MUL
    assert captured["global_num_experts"] == 32
    assert captured["expert_map"] is None
    assert captured["apply_router_weight_on_input"] is False
    assert captured["num_expert_group"] == 0
    assert captured["topk_group"] == 0
    assert captured["e_score_correction_bias"] is None
    assert captured["routed_scaling_factor"] == 1.0
    assert output.shape == x.shape


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
