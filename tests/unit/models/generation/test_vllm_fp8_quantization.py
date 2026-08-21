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
            "use_deep_gemm": True,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    assert vllm_kwargs == {
        "quantization": "fp8",
        "kv_cache_dtype": "auto",
        "hf_overrides": {"quantization_config": fp8.MXFP8_BLOCK_QUANT_KWARGS},
    }
    assert applied_configs == [fp8.global_fp8_config]
    assert fp8.global_fp8_config.is_mx is True
    assert "VLLM_USE_DEEP_GEMM" not in fp8.os.environ
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in fp8.os.environ


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


def test_quantize_mxfp8_weight_restores_grouped_expert_shape(fp8_module, monkeypatch):
    fp8 = fp8_module
    weight = torch.zeros(2, 3, 32, dtype=torch.bfloat16)

    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    def flattened_quantize(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = tensor.numel() // tensor.shape[-1]
        return (
            torch.zeros(rows, tensor.shape[-1], dtype=torch.float8_e4m3fn),
            torch.tensor([0, 2, 0, 127, 255, 5], dtype=torch.uint8),
        )

    monkeypatch.setattr(mxfp8_utils, "mxfp8_e4m3_quantize", flattened_quantize)

    value, scale = fp8.quantize_mxfp8_weight(weight)

    assert value.shape == weight.shape
    assert scale.shape == (2, 3, 1)
    assert torch.equal(
        scale.flatten(), torch.tensor([1, 2, 1, 127, 255, 5], dtype=torch.uint8)
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

    w13_weight = torch.nn.Parameter(torch.zeros(2, 4, 3), requires_grad=False)
    w2_weight = torch.nn.Parameter(torch.zeros(2, 3, 2), requires_grad=False)
    w13_scale = torch.nn.Parameter(torch.zeros(2, 4, 1), requires_grad=False)
    w2_scale = torch.nn.Parameter(torch.zeros(2, 3, 1), requires_grad=False)
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
