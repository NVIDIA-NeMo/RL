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

import os
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
    env_keys = {
        "NRL_FP8_USE_WEIGHT_POW2_SCALE",
        "NRL_FP8_USE_ACTIVATION_POW2_SCALE",
        "NRL_FP8_NUM_FIRST_LAYERS_IN_BF16",
        "NRL_FP8_NUM_LAST_LAYERS_IN_BF16",
        "NRL_FP8_MODEL_PARALLEL_SIZE",
        "NRL_FP8_KV_CACHE_DTYPE",
        "NRL_FP8_USE_WEIGHTS",
        "NRL_FP8_IS_MX",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_USE_DEEP_GEMM_E8M0",
    }
    old_env = {key: os.environ.get(key) for key in env_keys}
    fp8.global_fp8_config = None
    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False

    try:
        yield fp8
    finally:
        fp8.global_fp8_config = old_config
        fp8.fp8_state = old_state
        fp8.fp8_patches_applied = old_patches_applied
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def test_fp8_ds_mla_skips_static_kv_scale_patch(fp8_module, monkeypatch):
    fp8 = fp8_module
    patched_paths = []

    class FakePatch:
        def start(self):
            pass

    def fake_patch(path, _replacement):
        patched_paths.append(path)
        return FakePatch()

    monkeypatch.setattr(fp8, "patch", fake_patch)

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(
            use_fp8_weights=True,
            model_parallel_size=1,
            kv_cache_dtype="fp8_ds_mla",
        ),
    )

    assert not any("BaseKVCacheMethod" in path for path in patched_paths)


def test_init_fp8_accepts_fp8_ds_mla(fp8_module, monkeypatch):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "fp8_ds_mla",
            "async_engine": False,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    assert vllm_kwargs["kv_cache_dtype"] == "fp8_ds_mla"
    assert fp8.global_fp8_config.kv_cache_dtype == "fp8_ds_mla"


def test_init_fp8_preserves_dsv4_scale_format_and_uses_arch_e8m0(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    checkpoint_quantization_config = {
        "scale_fmt": "ue8m0",
        "fmt": "checkpoint-value",
        "weight_block_size": [64, 64],
    }
    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            num_hidden_layers=4,
            quantization_config=checkpoint_quantization_config,
        ),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "fp8_ds_mla",
            "async_engine": False,
            "pow2_weight_scaling_factors": True,
            "use_deep_gemm": True,
        },
        "dummy-model",
        model_parallel_size=8,
    )

    quantization_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    assert quantization_config["scale_fmt"] == "ue8m0"
    assert quantization_config["fmt"] == "e4m3"
    assert quantization_config["weight_block_size"] == [128, 128]
    assert os.environ["VLLM_USE_DEEP_GEMM"] == "1"
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in os.environ
    assert os.environ["NRL_FP8_USE_WEIGHT_POW2_SCALE"] == "1"


def test_resolve_fp8_config_from_worker_environment(fp8_module, monkeypatch):
    fp8 = fp8_module
    fp8.global_fp8_config = None
    monkeypatch.setenv("NRL_FP8_USE_WEIGHT_POW2_SCALE", "1")
    monkeypatch.setenv("NRL_FP8_USE_ACTIVATION_POW2_SCALE", "0")
    monkeypatch.setenv("NRL_FP8_MODEL_PARALLEL_SIZE", "8")
    monkeypatch.setenv("NRL_FP8_KV_CACHE_DTYPE", "fp8_ds_mla")
    monkeypatch.setenv("NRL_FP8_USE_WEIGHTS", "1")
    monkeypatch.setenv("NRL_FP8_IS_MX", "0")

    resolved = fp8._resolve_fp8_config()

    assert resolved.use_weight_pow2_scale is True
    assert resolved.use_activation_pow2_scale is False
    assert resolved.model_parallel_size == 8
    assert resolved.kv_cache_dtype == "fp8_ds_mla"
    assert resolved.is_mx is False


def test_dsv4_module_lookup_maps_checkpoint_attention_to_fused_vllm_module(
    fp8_module,
):
    import torch

    fp8 = fp8_module

    class DeepSeekV4ForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.attn = torch.nn.Module()
            layer.attn.fused_wqa_wkv = torch.nn.Linear(2, 2, bias=False)
            self.model.layers = torch.nn.ModuleList([layer])
            self.packed_modules_mapping = {}

    model = DeepSeekV4ForCausalLM()

    module = fp8._get_module_from_param_name(model, "model.layers.0.attn.wq_a.weight")

    assert module is model.model.layers[0].attn.fused_wqa_wkv


def test_dsv4_bmm_refit_slices_tp_weight_and_transforms_scale(fp8_module, monkeypatch):
    import torch
    from vllm.utils import deep_gemm

    fp8 = fp8_module
    weight = torch.nn.Parameter(torch.zeros(2, 2, 4), requires_grad=False)
    weight.tp_rank = 1
    scale = torch.nn.Parameter(torch.zeros(2, 1, 2), requires_grad=False)
    layer = types.SimpleNamespace(
        weight=weight,
        weight_block_size=[2, 2],
        is_bmm=True,
    )
    loaded_weight = torch.arange(32, dtype=torch.float32).view(8, 4)
    loaded_scale = torch.arange(8, dtype=torch.float32).view(4, 2)
    transform_calls = []

    def transform_sf_into_required_layout(**kwargs):
        transform_calls.append(kwargs)
        return kwargs["sf"] + 10

    monkeypatch.setattr(
        deep_gemm,
        "transform_sf_into_required_layout",
        transform_sf_into_required_layout,
    )

    assert fp8._copy_deepseek_v4_bmm_weight(layer, weight, loaded_weight)
    assert fp8._copy_deepseek_v4_bmm_scale(layer, scale, loaded_scale)

    assert torch.equal(weight, loaded_weight[4:8].view(2, 2, 4))
    assert torch.equal(scale, loaded_scale[2:4].view(2, 1, 2) + 10)
    assert transform_calls[0]["num_groups"] == 2
    assert transform_calls[0]["recipe"] == (1, 2, 2)


def test_dsv4_moe_refit_tp_shards_w13_and_w2_scales(fp8_module, monkeypatch):
    import torch

    fp8 = fp8_module

    class FakeRoutedExperts:
        def __init__(self):
            self.moe_config = types.SimpleNamespace(tp_size=2, tp_rank=1)

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 3 else -1

    monkeypatch.setattr(fp8, "RoutedExperts", FakeRoutedExperts)
    layer = FakeRoutedExperts()
    module_map = {"experts": layer}
    w13_scale = torch.nn.Parameter(torch.zeros(1, 4, 2), requires_grad=False)
    w2_scale = torch.nn.Parameter(torch.zeros(1, 2, 2), requires_grad=False)
    full_w1_scale = torch.arange(8, dtype=torch.float32).view(4, 2)
    full_w3_scale = full_w1_scale + 100
    full_w2_scale = torch.arange(8, dtype=torch.float32).view(2, 4)

    for shard_id, loaded_scale in (
        ("w1", full_w1_scale),
        ("w3", full_w3_scale),
    ):
        assert fp8._try_load_deepseek_v4_moe_block_scale(
            module_map,
            "experts.w13_weight_scale_inv",
            w13_scale,
            loaded_scale,
            "checkpoint.experts.scale",
            shard_id,
            3,
        )
    assert fp8._try_load_deepseek_v4_moe_block_scale(
        module_map,
        "experts.w2_weight_scale_inv",
        w2_scale,
        full_w2_scale,
        "checkpoint.experts.scale",
        "w2",
        3,
    )

    assert torch.equal(w13_scale[0, :2], full_w1_scale[2:4])
    assert torch.equal(w13_scale[0, 2:], full_w3_scale[2:4])
    assert torch.equal(w2_scale[0], full_w2_scale[:, 2:4])


def test_prepare_fp8_refit_restores_raw_e8m0_linear_scale(fp8_module, monkeypatch):
    import torch
    from vllm.utils import deep_gemm

    fp8 = fp8_module

    class FakeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(256, 256), requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(
                torch.zeros(4, 8, dtype=torch.int32), requires_grad=False
            )
            self.weight_block_size = [128, 128]
            self.orig_dtype = torch.bfloat16

    layer = FakeLinear()
    model = torch.nn.Sequential(layer)
    model_runner = types.SimpleNamespace(model=model)
    monkeypatch.setattr(fp8, "LinearBase", FakeLinear)
    monkeypatch.setattr(fp8, "ensure_fp8_patches_applied", lambda _runner: None)
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_e8m0_used", lambda: True)
    monkeypatch.setattr(
        deep_gemm,
        "should_use_deepgemm_for_fp8_linear",
        lambda _dtype, _shape: True,
    )

    fp8.prepare_fp8_model_for_refit(model_runner)

    assert layer.weight_scale_inv.shape == (2, 2)
    assert layer.weight_scale_inv.dtype == torch.float32


def test_dsv4_bmm_post_process_uses_vllm_bmm_layout(fp8_module, monkeypatch):
    import torch
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils import deep_gemm

    fp8 = fp8_module
    weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    scale = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    layer = types.SimpleNamespace(
        weight=weight,
        weight_scale_inv=scale,
        weight_block_size=[2, 2],
        orig_dtype=torch.bfloat16,
        is_bmm=True,
        bmm_batch_size=2,
    )
    calls = []

    monkeypatch.setattr(
        deep_gemm,
        "should_use_deepgemm_for_fp8_linear",
        lambda _dtype, _shape: True,
    )
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_e8m0_used", lambda: True)

    def post_process(**kwargs):
        calls.append(kwargs)
        return torch.ones(2, 2, 4), torch.ones(2, 1, 2, dtype=torch.int32)

    monkeypatch.setattr(
        fp8_utils, "deepgemm_post_process_fp8_weight_block", post_process
    )

    fp8.maybe_post_process_fp8_weight_block(layer)

    assert layer.weight is weight
    assert layer.weight_scale_inv is scale
    assert layer.weight.shape == (2, 2, 4)
    assert layer.weight_scale_inv.shape == (2, 1, 2)
    assert calls[0]["is_bmm"] is True
    assert calls[0]["bmm_batch_size"] == 2


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
