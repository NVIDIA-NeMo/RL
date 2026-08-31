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
from pathlib import Path

import pytest
import torch
import yaml

pytestmark = pytest.mark.vllm

PROJECT_ROOT = Path(__file__).resolve().parents[4]


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


def test_init_fp8_passes_modelopt_ignore_patterns_without_hf_expansion(
    fp8_module, monkeypatch
):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(
        fp8.AutoModel,
        "from_config",
        lambda *_args, **_kwargs: pytest.fail(
            "ModelOpt ignore patterns must not depend on AutoModel names"
        ),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "quantization_ignore_patterns": [
                " model.layers.*.self_attn.* ",
                "model.layers.*.mlp.gate",
                "lm_head",
            ],
        },
        "dummy-model",
        model_parallel_size=1,
    )

    quant_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    assert quant_config["ignore"] == [
        "model.layers.*.self_attn.*",
        "model.layers.*.mlp.gate",
        "lm_head",
    ]
    assert quant_config["ignored_layers"] == ["lm_head"]

    modelopt_config = ModelOptMxFp8Config.from_config(quant_config)
    qwen3_quantizable_families = {
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.experts",
        "lm_head",
    }
    mxfp8_families = {
        name
        for name in qwen3_quantizable_families
        if not modelopt_config.is_layer_excluded(name)
    }
    assert mxfp8_families == {"model.layers.0.mlp.experts"}
    assert not modelopt_config.is_layer_excluded("model.layers.0.mlp.gate_up_proj")


@pytest.mark.parametrize(
    "config",
    [
        types.SimpleNamespace(
            num_hidden_layers=43,
            num_nextn_predict_layers=1,
        ),
        types.SimpleNamespace(
            num_hidden_layers=78,
            num_nextn_predict_layers=1,
        ),
        types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                num_hidden_layers=61,
                num_nextn_predict_layers=0,
            )
        ),
        types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                num_hidden_layers=32,
                mtp_num_hidden_layers=1,
            )
        ),
    ],
)
def test_init_fp8_does_not_add_draft_model_patterns_to_target_config(
    fp8_module, monkeypatch, config
):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    quant_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    assert quant_config["ignore"] == ["lm_head"]


def test_init_fp8_loads_remote_model_config(fp8_module, monkeypatch):
    fp8 = fp8_module
    config_loads = []

    def load_config(*args, **kwargs):
        config_loads.append((args, kwargs))
        return types.SimpleNamespace(num_hidden_layers=61)

    monkeypatch.setattr(fp8.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
        },
        "remote-code-model",
        model_parallel_size=1,
    )

    assert config_loads == [(("remote-code-model",), {"trust_remote_code": True})]


def test_init_fp8_deduplicates_explicit_ignore_pattern(fp8_module, monkeypatch):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            num_hidden_layers=61,
            num_nextn_predict_layers=1,
        ),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "quantization_ignore_patterns": ["lm_head", "lm_head"],
        },
        "dummy-model",
        model_parallel_size=1,
    )

    ignore = vllm_kwargs["hf_overrides"]["quantization_config"]["ignore"]
    assert ignore == ["lm_head"]


@pytest.mark.parametrize(
    ("recipe_name", "quantized_modules", "bf16_modules"),
    [
        (
            "grpo-deepseek-v3-64n4g-mxfp8-rollout.yaml",
            {
                "model.layers.3.mlp.experts",
                "model.layers.60.mlp.experts",
            },
            {
                "model.layers.2.mlp.experts",
                "model.layers.3.self_attn.qkv_proj",
                "model.layers.3.mlp.gate",
                "model.layers.3.mlp.shared_experts.gate_up_proj",
                "model.layers.61.mlp.experts",
                "model.layers.61.mtp.fc",
                "lm_head",
            },
        ),
        (
            "grpo-nemotron3-super-120BA12B-32n4g-mxfp8-rollout.yaml",
            {"model.layers.3.mixer.experts"},
            {
                "model.layers.3.mixer.in_proj",
                "model.layers.3.mixer.out_proj",
                "model.layers.3.mixer.qkv_proj",
                "model.layers.3.mixer.o_proj",
                "model.layers.3.mixer.up_proj",
                "model.layers.3.mixer.down_proj",
                "model.layers.3.mixer.gate",
                "model.layers.3.mixer.shared_experts.up_proj",
                "model.layers.3.mixer.fc1_latent_proj",
                "model.layers.3.mixer.fc2_latent_proj",
                "lm_head",
            },
        ),
    ],
)
def test_mxfp8_recipe_patterns_select_only_routed_experts(
    recipe_name, quantized_modules, bf16_modules
):
    pytest.importorskip("vllm")

    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    recipe_path = (
        PROJECT_ROOT / "examples/configs/recipes/llm/performance" / recipe_name
    )
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    patterns = recipe["policy"]["generation"]["vllm_cfg"][
        "quantization_ignore_patterns"
    ]
    modelopt_config = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "ignore": [*patterns, "lm_head"],
            "ignored_layers": ["lm_head"],
        }
    )

    assert all(
        not modelopt_config.is_layer_excluded(name) for name in quantized_modules
    )
    assert all(modelopt_config.is_layer_excluded(name) for name in bf16_modules)


def test_init_fp8_excludes_lm_head_from_regular_fp8(fp8_module, monkeypatch):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

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
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": False,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    quant_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    assert quant_config == {
        **fp8.FP8_BLOCK_QUANT_KWARGS,
        "ignored_layers": ["lm_head"],
        "ignore": ["lm_head"],
    }
    assert Fp8Config.from_config(quant_config).ignored_layers == ["lm_head"]


@pytest.mark.parametrize("patterns", ["lm_head", 1, {"lm_head"}])
def test_init_fp8_rejects_non_list_modelopt_ignore_patterns(
    fp8_module, monkeypatch, patterns
):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )

    with pytest.raises(ValueError, match="list of strings"):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                "quantization_ignore_patterns": patterns,
            },
            "dummy-model",
            model_parallel_size=1,
        )


@pytest.mark.parametrize("pattern", ["", "   "])
def test_init_fp8_rejects_empty_modelopt_ignore_pattern(
    fp8_module, monkeypatch, pattern
):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    with pytest.raises(ValueError, match="non-empty strings"):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                "quantization_ignore_patterns": [pattern],
            },
            "dummy-model",
            model_parallel_size=1,
        )


def test_init_fp8_combines_legacy_and_modelopt_ignore_patterns(fp8_module, monkeypatch):
    fp8 = fp8_module

    class FakeModel:
        def named_parameters(self):
            return [
                ("layers.0.self_attn.q_proj.weight", object()),
                ("layers.0.mlp.experts.0.gate_proj.weight", object()),
            ]

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(fp8.AutoModel, "from_config", lambda *_args: FakeModel())
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    with pytest.warns(
        DeprecationWarning,
        match="quantization_ignored_layer_kws.*quantization_ignore_patterns",
    ):
        vllm_kwargs = fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                "quantization_ignored_layer_kws": ["q_proj"],
                "quantization_ignore_patterns": ["lm_head"],
            },
            "dummy-model",
            model_parallel_size=1,
        )

    quant_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    assert quant_config["ignore"] == [
        "lm_head",
        "model.layers.0.self_attn.q_proj",
    ]
    assert quant_config["ignore"].count("lm_head") == 1


def test_init_fp8_rejects_modelopt_ignore_patterns_for_regular_fp8(
    fp8_module, monkeypatch
):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    with pytest.raises(ValueError, match="requires is_mx=True"):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": False,
                "quantization_ignore_patterns": ["lm_head"],
            },
            "dummy-model",
            model_parallel_size=1,
        )


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


@pytest.mark.parametrize(
    "use_ray_v2", ["1", "0"], ids=["ray_executor_v2", "ray_executor_v1"]
)
def test_multi_gpu_fp8_patches_before_model_load(fp8_module, monkeypatch, use_ray_v2):
    """Both Ray executors must receive the FP8 patches before worker/model init."""
    from vllm import envs
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.executor.ray_executor import RayDistributedExecutor
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2, RayWorkerProc

    fp8 = fp8_module
    events = []
    fp8_config = fp8.FP8Config(model_parallel_size=2)
    vllm_config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(distributed_executor_backend="ray")
    )

    # vLLM memoizes env lookups once an engine has been built in-process, which
    # would make setenv below a silent no-op and quietly test one branch twice.
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", use_ray_v2)
    uses_v2 = use_ray_v2 == "1"
    assert envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND is uses_v2
    assert Executor.get_class(vllm_config) is (
        RayExecutorV2 if uses_v2 else RayDistributedExecutor
    )

    def fake_apply_fp8_patches(_worker, config):
        events.append(("apply_fp8_patches", config))
        fp8.fp8_patches_applied = True

    def fake_initialize_worker(_worker, *args, **kwargs):
        events.append(("initialize_worker", args))
        assert fp8.fp8_patches_applied, (
            "RayExecutorV2 started worker/model initialization before NeMo-RL "
            "installed its FP8 patches"
        )

    def fake_collective_rpc(_executor, *_args, **_kwargs):
        events.append(("collective_rpc", None))
        assert fp8.fp8_patches_applied, (
            "RayDistributedExecutor started worker/model initialization before "
            "NeMo-RL installed its FP8 patches"
        )

    monkeypatch.setattr(fp8, "apply_fp8_patches", fake_apply_fp8_patches)
    # monkey_patch_vllm_ray_executor() rebinds these by raw class assignment with
    # no cleanup of its own, so register both with monkeypatch to undo the rebind
    # even when this regression test fails.
    monkeypatch.setattr(RayWorkerProc, "initialize_worker", fake_initialize_worker)
    monkeypatch.setattr(RayDistributedExecutor, "collective_rpc", fake_collective_rpc)

    fp8.monkey_patch_vllm_ray_executor(fp8_config)

    if uses_v2:
        assert RayDistributedExecutor.collective_rpc is fake_collective_rpc, (
            "the V1 executor must be left unpatched when the V2 backend is active"
        )
        patched_initialize_worker = RayWorkerProc.initialize_worker
        # cloudpickle reconstructs nested functions with a distinct globals dict.
        worker_initialize_worker = types.FunctionType(
            patched_initialize_worker.__code__,
            patched_initialize_worker.__globals__.copy(),
            closure=patched_initialize_worker.__closure__,
        )
        worker_initialize_worker(object(), 0, {})
        worker_initialize_worker(object(), 0, {})

        assert events == [
            ("apply_fp8_patches", fp8_config),
            ("initialize_worker", (0, {})),
            ("initialize_worker", (0, {})),
        ]
    else:
        assert RayWorkerProc.initialize_worker is fake_initialize_worker, (
            "the V2 worker hook must be left unpatched when the V1 backend is active"
        )

        # execute_method(fn, cfg) ends up calling fn(worker, cfg) upstream, so pass
        # the worker through rather than None to mirror apply_fp8_patches(self, cfg).
        def make_worker():
            worker = types.SimpleNamespace()

            def fake_execute_method_remote(fn, config):
                fn(worker, config)
                return object()

            worker.execute_method = types.SimpleNamespace(
                remote=fake_execute_method_remote
            )
            return worker

        monkeypatch.setattr(fp8, "ray", types.SimpleNamespace(get=lambda _future: None))
        executor = types.SimpleNamespace(workers=[make_worker()])
        RayDistributedExecutor.collective_rpc(executor, "init_device")
        RayDistributedExecutor.collective_rpc(executor, "init_device")

        assert events == [
            ("apply_fp8_patches", fp8_config),
            ("collective_rpc", None),
            ("collective_rpc", None),
        ]


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


def _grouped_expert_model(fp8, monkeypatch, experts_dtype, wrap_language_model=False):
    """Fake model mirroring vLLM's MoERunner -> RoutedExperts layout at
    ``layers.0.mlp.experts``, with expert weights in ``experts_dtype``.

    With ``wrap_language_model=True``, mirrors the Qwen3.5-VL layout instead:
    no top-level ``model``/``layers``; the decoder sits at
    ``language_model.model.layers`` while parameter names still carry the
    synthetic ``model.language_model.layers.`` prefix — the key shape both
    FP8 recipes actually refit at vLLM 0.25.1.
    """
    import torch

    class _RoutedExperts:
        pass

    class _MoERunner:
        pass

    monkeypatch.setattr(fp8, "RoutedExperts", _RoutedExperts)
    monkeypatch.setattr(fp8, "MoERunner", _MoERunner)

    experts = _RoutedExperts()
    experts.w13_weight = torch.zeros(2, 4, 4, dtype=experts_dtype)
    experts.w2_weight = torch.zeros(2, 4, 4, dtype=experts_dtype)
    runner = _MoERunner()
    runner.routed_experts = experts

    layer = torch.nn.Module()
    layer.mlp = types.SimpleNamespace(experts=runner)
    layers = torch.nn.ModuleList([layer])
    if wrap_language_model:
        return types.SimpleNamespace(
            packed_modules_mapping={},
            language_model=types.SimpleNamespace(
                model=types.SimpleNamespace(layers=layers)
            ),
        )
    return types.SimpleNamespace(
        packed_modules_mapping={},
        layers=layers,
    )


GROUPED_EXPERT_KEY_SHAPES = pytest.mark.parametrize(
    "layers_prefix, wrap_language_model",
    [("model.layers", False), ("model.language_model.layers", True)],
    ids=["flat", "vl-wrapper"],
)


@GROUPED_EXPERT_KEY_SHAPES
def test_load_weights_passes_grouped_experts_through_for_ignored_bf16_layers(
    fp8_module, monkeypatch, layers_prefix, wrap_language_model
):
    """Grouped-expert refit must respect ``ignored_layers``.

    Experts covered by num_{first,last}_layers_in_bf16 or
    quantization_ignored_layer_kws are built by vLLM as unquantized bf16 MoE
    without ``*_weight_scale_inv`` params, so emitting per-expert FP8 + scale
    entries for them has nowhere to load. The grouped bf16 slab must pass
    through untouched.
    """
    import torch

    fp8 = fp8_module
    model = _grouped_expert_model(fp8, monkeypatch, torch.bfloat16, wrap_language_model)
    loaded = []
    model.load_weights = lambda pairs: loaded.extend(pairs)

    gate_up = torch.randn(2, 256, 128).to(torch.bfloat16)
    down = torch.randn(2, 128, 128).to(torch.bfloat16)
    fp8.load_weights(
        [
            (f"{layers_prefix}.0.mlp.experts.gate_up_proj", gate_up),
            (f"{layers_prefix}.0.mlp.experts.down_proj", down),
        ],
        types.SimpleNamespace(model=model),
    )

    assert [k for k, _ in loaded] == [
        f"{layers_prefix}.0.mlp.experts.gate_up_proj",
        f"{layers_prefix}.0.mlp.experts.down_proj",
    ]
    assert loaded[0][1] is gate_up
    assert loaded[1][1] is down
    # Pass-through is also what a failed lookup produces, so pin that the
    # bf16 RoutedExperts was actually resolved.
    assert isinstance(
        fp8._get_module_from_param_name(
            model, f"{layers_prefix}.0.mlp.experts.gate_up_proj"
        ),
        fp8.RoutedExperts,
    )


def _assert_dequant_close(weight_fp8, scale_inv, source_bf16):
    """Dequantized FP8 must match the bf16 source within e4m3 half-ULP.

    Worst-case blockwise quantization error is half the e4m3 ULP at the
    block amax: amax * (16 / 448) = amax / 28.
    """
    import torch

    dequant = weight_fp8.to(torch.float32) * scale_inv.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    source = source_bf16.to(torch.float32)
    rows, cols = source.shape
    block_amax = (
        source.abs().reshape(rows // 128, 128, cols // 128, 128).amax(dim=(1, 3))
    )
    atol = block_amax.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1) / 28
    assert torch.all((dequant - source).abs() <= atol + 1e-6 * source.abs())


@GROUPED_EXPERT_KEY_SHAPES
def test_load_weights_expands_grouped_experts_for_fp8_layers(
    fp8_module, monkeypatch, layers_prefix, wrap_language_model
):
    """FP8-built experts keep the per-expert expand+quantize refit path.

    Non-square expert shards (256x384 gate/up, 384x256 down) give a scale
    grid larger than 1x1 so a transposed or misrouted scale cannot pass, and
    the dequantization check pins values, not just names and shapes.
    """
    import torch

    fp8 = fp8_module
    fp8.global_fp8_config = types.SimpleNamespace(
        use_weight_pow2_scale=False, is_mx=False
    )
    model = _grouped_expert_model(
        fp8, monkeypatch, torch.float8_e4m3fn, wrap_language_model
    )
    loaded = []
    model.load_weights = lambda pairs: loaded.extend(pairs)

    intermediate, hidden = 256, 384
    gate_up = torch.randn(2, 2 * intermediate, hidden).to(torch.bfloat16)
    down = torch.randn(2, hidden, intermediate).to(torch.bfloat16)
    fp8.load_weights(
        [
            (f"{layers_prefix}.0.mlp.experts.gate_up_proj", gate_up),
            (f"{layers_prefix}.0.mlp.experts.down_proj", down),
        ],
        types.SimpleNamespace(model=model),
    )

    base = f"{layers_prefix}.0.mlp.experts"
    assert [k for k, _ in loaded] == [
        f"{base}.{eid}.{proj}.weight{suffix}"
        for proj in ("gate_proj", "up_proj")
        for eid in (0, 1)
        for suffix in ("", "_scale_inv")
    ] + [
        f"{base}.{eid}.down_proj.weight{suffix}"
        for eid in (0, 1)
        for suffix in ("", "_scale_inv")
    ]
    entries = dict(loaded)
    shards = {
        "gate_proj": (gate_up[:, :intermediate, :], (intermediate, hidden)),
        "up_proj": (gate_up[:, intermediate:, :], (intermediate, hidden)),
        "down_proj": (down, (hidden, intermediate)),
    }
    for proj, (source, shape) in shards.items():
        for eid in (0, 1):
            weight = entries[f"{base}.{eid}.{proj}.weight"]
            scale = entries[f"{base}.{eid}.{proj}.weight_scale_inv"]
            assert weight.dtype == torch.float8_e4m3fn
            assert weight.shape == shape
            assert scale.shape == (shape[0] // 128, shape[1] // 128)
            _assert_dequant_close(weight, scale, source[eid])


def test_load_weights_rejects_grouped_experts_for_mxfp8(fp8_module, monkeypatch):
    """Grouped MoE expansion only implements blockwise FP8, not MXFP8."""
    import torch

    fp8 = fp8_module
    fp8.global_fp8_config = types.SimpleNamespace(
        use_weight_pow2_scale=False, is_mx=True
    )
    model = _grouped_expert_model(fp8, monkeypatch, torch.float8_e4m3fn)
    model.load_weights = lambda pairs: pytest.fail("must raise before loading")

    with pytest.raises(NotImplementedError, match="MXFP8"):
        fp8.load_weights(
            [
                (
                    "model.layers.0.mlp.experts.gate_up_proj",
                    torch.randn(2, 512, 384).to(torch.bfloat16),
                )
            ],
            types.SimpleNamespace(model=model),
        )
