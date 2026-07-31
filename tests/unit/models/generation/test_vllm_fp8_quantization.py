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
    fp8.global_fp8_config = types.SimpleNamespace(is_mx=True)
    native = torch.ones(2, 2, dtype=torch.bfloat16)
    prequantized = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    receiver_quantized = torch.full((2, 64), 2.0, dtype=torch.bfloat16)
    receiver_fp8 = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    receiver_scales = torch.tensor([[0, 7], [3, 0]], dtype=torch.uint8)
    loaded = []

    monkeypatch.setattr(
        fp8,
        "_is_fp8_weight",
        lambda name, _model: name != "model.native",
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
        lambda model, weights: loaded.extend(weights),
    )
    model = object()

    fp8.load_weights(
        [
            ("model.native", native),
            ("model.prequantized.weight", prequantized),
            ("model.receiver.weight", receiver_quantized),
        ],
        types.SimpleNamespace(model=model),
    )

    assert loaded[0][0] == "model.native"
    assert loaded[0][1] is native
    assert loaded[1][0] == "model.prequantized.weight"
    assert loaded[1][1] is prequantized
    assert loaded[2][0] == "model.receiver.weight"
    assert loaded[2][1] is receiver_fp8
    assert loaded[3][0] == "model.receiver.weight_scale_from_checkpoint"
    torch.testing.assert_close(
        loaded[3][1],
        torch.tensor([[1, 7], [3, 1]], dtype=torch.uint8),
    )


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
    monkeypatch.delenv("NRL_MXFP8_BATCHED_SHUFFLE", raising=False)

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
