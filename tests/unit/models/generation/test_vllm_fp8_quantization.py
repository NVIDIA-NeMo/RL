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


def test_init_fp8_ignores_lm_head_explicitly(fp8_module, monkeypatch):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(
        fp8.AutoModel,
        "from_config",
        lambda _config: types.SimpleNamespace(named_parameters=lambda: []),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "quantization_ignored_layer_kws": ["lm_head"],
        },
        "dummy-model",
        model_parallel_size=1,
    )

    quantization_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    assert quantization_config["ignored_layers"] == ["lm_head"]
    assert quantization_config["ignore"] == ["lm_head"]


def test_init_fp8_rejects_unmatched_ignored_layer_keyword(fp8_module, monkeypatch):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(
        fp8.AutoModel,
        "from_config",
        lambda _config: types.SimpleNamespace(
            named_parameters=lambda: [("layers.0.q_proj.weight", object())]
        ),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    with pytest.raises(
        AssertionError,
        match="entries that do not match any model layer: \\['missing_proj'\\]",
    ):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                "quantization_ignored_layer_kws": ["q_proj", "missing_proj"],
            },
            "dummy-model",
            model_parallel_size=1,
        )


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


def test_load_weights_gets_mxfp8_mode_from_worker_config(fp8_module, monkeypatch):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    fp8 = fp8_module
    loaded_weights = []
    quantized_weight = object()
    quantized_scale = fp8.torch.ones(2, 1)
    model = types.SimpleNamespace(
        packed_modules_mapping={},
        layer=types.SimpleNamespace(
            weight_scale=fp8.torch.nn.Parameter(
                fp8.torch.empty(2, 1), requires_grad=False
            )
        ),
        load_weights=lambda weights: loaded_weights.extend(weights),
    )
    model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(
            quant_config=object.__new__(ModelOptMxFp8Config)
        ),
    )

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    monkeypatch.setattr(
        mxfp8_utils,
        "mxfp8_e4m3_quantize",
        lambda _weight: (quantized_weight, quantized_scale),
    )
    monkeypatch.setenv(fp8.NRL_VLLM_MXFP8_REFIT_USE_WORKER_CONFIG, "1")

    assert fp8.global_fp8_config is None
    fp8.load_weights([("layer.weight", fp8.torch.ones(2, 2))], model_runner)

    assert loaded_weights[0] == ["layer.weight", quantized_weight]
    assert loaded_weights[1][0] == "layer.weight_scale"
    fp8.torch.testing.assert_close(
        loaded_weights[1][1], fp8.torch.squeeze(quantized_scale, dim=-1)
    )


def test_get_mxfp8_scale_name_prefers_refit_checkpoint_parameter(fp8_module):
    fp8 = fp8_module
    layer = types.SimpleNamespace(
        weight_scale=fp8.torch.nn.Parameter(
            fp8.torch.empty(2, 1), requires_grad=False
        ),
        weight_scale_from_checkpoint=fp8.torch.nn.Parameter(
            fp8.torch.empty(2, 1), requires_grad=False
        ),
    )
    model = types.SimpleNamespace(packed_modules_mapping={}, layer=layer)

    assert (
        fp8._get_mxfp8_scale_name("layer.weight", model)
        == "layer.weight_scale_from_checkpoint"
    )


@pytest.mark.parametrize(
    ("projection_name", "fused_weight_name", "scale_suffix"),
    [
        ("up_proj", "w13_weight", "_scale_from_checkpoint"),
        ("down_proj", "w2_weight", "_scale"),
    ],
)
def test_get_mxfp8_scale_name_maps_per_expert_moe_weight(
    fp8_module,
    monkeypatch,
    projection_name,
    fused_weight_name,
    scale_suffix,
):
    fp8 = fp8_module

    class FakeRoutedExperts:
        def get_expert_mapping(self):
            return [
                ("experts.w13_", "experts.0.up_proj.", 0, "w1"),
                ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
            ]

    routed_experts = FakeRoutedExperts()
    setattr(
        routed_experts,
        fused_weight_name + scale_suffix,
        fp8.torch.nn.Parameter(fp8.torch.empty(2, 1), requires_grad=False),
    )
    monkeypatch.setattr(fp8, "RoutedExperts", FakeRoutedExperts)
    monkeypatch.setattr(
        fp8,
        "_get_module_from_param_name",
        lambda _model, _name: routed_experts,
    )
    name = f"backbone.layers.1.mixer.experts.0.{projection_name}.weight"

    assert fp8._get_mxfp8_scale_name(name, object()) == name + scale_suffix


def test_get_mxfp8_scale_name_reports_unregistered_scale(fp8_module):
    fp8 = fp8_module
    model = types.SimpleNamespace(
        packed_modules_mapping={}, layer=types.SimpleNamespace()
    )

    with pytest.raises(
        RuntimeError,
        match="Expected the resolved SimpleNamespace module to register",
    ):
        fp8._get_mxfp8_scale_name("layer.weight", model)


def test_load_weights_reports_missing_process_local_config(fp8_module, monkeypatch):
    fp8 = fp8_module
    monkeypatch.delenv(fp8.NRL_VLLM_MXFP8_REFIT_USE_WORKER_CONFIG, raising=False)

    model_runner = types.SimpleNamespace(
        model=types.SimpleNamespace(),
        vllm_config=types.SimpleNamespace(),
    )

    with pytest.raises(
        RuntimeError,
        match=fp8.NRL_VLLM_MXFP8_REFIT_USE_WORKER_CONFIG,
    ):
        fp8.load_weights([], model_runner)


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
