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
    ("num_first_layers_in_bf16", "num_last_layers_in_bf16"),
    [
        (0, 0),
        (1, 1),
        (2, 6),
        (7, 3),
        (26, 26),
        (30, 30),
    ],
)
@pytest.mark.parametrize(
    "recipe_case",
    [
        pytest.param(
            types.SimpleNamespace(
                model_name="dummy-qwen-model",
                hf_layer_prefix="layers",
                raw_layer_prefix="model.layers",
                mapper_prefixes={},
                hf_target_suffixes=(
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.experts.gate_proj",
                    "mlp.experts.up_proj",
                    "mlp.experts.down_proj",
                ),
                vllm_target_suffixes=(
                    "self_attn.qkv_proj",
                    "self_attn.o_proj",
                    "mlp.experts",
                ),
                non_target_suffixes=("mlp.gate", "mlp.shared_experts.up_proj"),
                ignore_patterns=(
                    "*layers.*.mlp.gate",
                    "*layers.*.mlp.shared_experts.*",
                    "lm_head",
                ),
            ),
            id="qwen",
        ),
        pytest.param(
            types.SimpleNamespace(
                model_name="dummy-nemotron-h-model",
                hf_layer_prefix="backbone.layers",
                raw_layer_prefix="backbone.layers",
                mapper_prefixes={"backbone": "model"},
                hf_target_suffixes=(
                    "mixer.q_proj",
                    "mixer.k_proj",
                    "mixer.v_proj",
                    "mixer.o_proj",
                    "mixer.experts.up_proj",
                    "mixer.experts.down_proj",
                ),
                vllm_target_suffixes=(
                    "mixer.qkv_proj",
                    "mixer.o_proj",
                    "mixer.experts",
                ),
                non_target_suffixes=(
                    "mixer.in_proj",
                    "mixer.shared_experts.up_proj",
                ),
                ignore_patterns=(
                    "*layers.*.mixer.in_proj",
                    "*layers.*.mixer.shared_experts.*",
                    "lm_head",
                ),
            ),
            id="nemotron-h",
        ),
    ],
)
def test_init_fp8_keeps_mixed_recipe_boundary_targets_in_bf16(
    fp8_module,
    monkeypatch,
    num_first_layers_in_bf16,
    num_last_layers_in_bf16,
    recipe_case,
):
    """Keep boundary QKVO and routed experts BF16 across mixed recipes."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.model_executor.models.utils import WeightsMapper

    fp8 = fp8_module
    num_hidden_layers = 52
    param_names = []
    for layer_idx in range(num_hidden_layers):
        param_names.extend(
            f"{recipe_case.hf_layer_prefix}.{layer_idx}.{suffix}.weight"
            for suffix in (
                *recipe_case.hf_target_suffixes,
                *recipe_case.non_target_suffixes,
            )
        )

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            num_hidden_layers=num_hidden_layers
        ),
    )
    monkeypatch.setattr(
        fp8.AutoModel,
        "from_config",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            named_parameters=lambda: [(name, None) for name in param_names]
        ),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "num_first_layers_in_bf16": num_first_layers_in_bf16,
            "num_last_layers_in_bf16": num_last_layers_in_bf16,
            "quantization_ignore_patterns": list(recipe_case.ignore_patterns),
        },
        recipe_case.model_name,
        model_parallel_size=1,
    )

    quant_config = vllm_kwargs["hf_overrides"]["quantization_config"]
    modelopt_config = ModelOptMxFp8Config.from_config(quant_config)
    mapper = WeightsMapper(orig_to_new_prefix=recipe_case.mapper_prefixes)
    modelopt_config.apply_vllm_mapper(mapper.get_unstacked_mapper())
    modelopt_config.packed_modules_mapping.update(
        {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    )
    boundary_layers = {
        layer_idx
        for layer_idx in range(num_hidden_layers)
        if layer_idx < num_first_layers_in_bf16
        or layer_idx >= num_hidden_layers - num_last_layers_in_bf16
    }
    for layer_idx in range(num_hidden_layers):
        if layer_idx in boundary_layers:
            for suffix in recipe_case.hf_target_suffixes:
                module_name = f"{recipe_case.raw_layer_prefix}.{layer_idx}.{suffix}"
                assert module_name in quant_config["ignored_layers"]

            for suffix in recipe_case.vllm_target_suffixes:
                module_name = f"model.layers.{layer_idx}.{suffix}"
                assert modelopt_config.is_layer_excluded(module_name)
        else:
            for suffix in recipe_case.vllm_target_suffixes:
                module_name = f"model.layers.{layer_idx}.{suffix}"
                assert not modelopt_config.is_layer_excluded(module_name)

        for suffix in recipe_case.non_target_suffixes:
            module_name = f"model.layers.{layer_idx}.{suffix}"
            assert modelopt_config.is_layer_excluded(module_name)


def test_init_fp8_reads_layer_count_from_text_config(fp8_module, monkeypatch):
    fp8 = fp8_module
    num_hidden_layers = 8
    param_names = [
        f"layers.{layer_idx}.mlp.experts.up_proj.weight"
        for layer_idx in range(num_hidden_layers)
    ]

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            text_config=types.SimpleNamespace(num_hidden_layers=num_hidden_layers)
        ),
    )
    from_config_calls = []

    def from_config(config, **kwargs):
        from_config_calls.append((config, kwargs))
        return types.SimpleNamespace(
            named_parameters=lambda: [(name, None) for name in param_names]
        )

    monkeypatch.setattr(fp8.AutoModel, "from_config", from_config)
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "num_first_layers_in_bf16": 2,
            "num_last_layers_in_bf16": 2,
        },
        "dummy-model-with-text-config",
        model_parallel_size=1,
    )

    ignored_layers = vllm_kwargs["hf_overrides"]["quantization_config"][
        "ignored_layers"
    ]
    assert "model.layers.0.mlp.experts.up_proj" in ignored_layers
    assert "model.layers.1.mlp.experts.up_proj" in ignored_layers
    assert "model.layers.6.mlp.experts.up_proj" in ignored_layers
    assert "model.layers.7.mlp.experts.up_proj" in ignored_layers
    assert "model.layers.2.mlp.experts.up_proj" not in ignored_layers
    assert from_config_calls[0][1] == {"trust_remote_code": True}


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


def test_init_fp8_defaults_missing_nemotron_h_moe_latent_size(fp8_module, monkeypatch):
    fp8 = fp8_module
    config = types.SimpleNamespace(model_type="nemotron_h", num_hidden_layers=52)

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )

    def create_model(model_config):
        assert hasattr(model_config, "moe_latent_size")
        assert model_config.moe_latent_size is None
        return types.SimpleNamespace(named_parameters=lambda: [])

    monkeypatch.setattr(fp8.AutoModel, "from_config", create_model)
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _config: None)

    fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "num_last_layers_in_bf16": 8,
        },
        "dummy-model",
        model_parallel_size=1,
    )


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


def test_quantize_mxfp8_weight_preserves_grouped_shape_and_scale_bytes(
    fp8_module, monkeypatch
):
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
        scale.flatten(), torch.tensor([0, 2, 0, 127, 255, 5], dtype=torch.uint8)
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


def test_load_weights_uses_supplied_model_loader(fp8_module):
    fp8 = fp8_module
    model = types.SimpleNamespace(
        packed_modules_mapping={},
        load_weights=lambda _weights: pytest.fail("must use the supplied loader"),
    )
    source = torch.ones(2, dtype=torch.bfloat16)
    loaded = []

    fp8.load_weights(
        [("model.norm.weight", source)],
        types.SimpleNamespace(model=model),
        model_load_weights=lambda weights: loaded.extend(weights),
    )

    assert loaded == [("model.norm.weight", source)]


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
