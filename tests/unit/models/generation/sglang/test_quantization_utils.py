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

import pytest

from nemo_rl.models.generation.sglang import quantization_utils
from nemo_rl.models.generation.sglang.quantization_utils import (
    expand_sglang_atomic_high_precision_substrings,
    get_dynamic_high_precision_substrings,
    get_sglang_quantization_scheme,
    prepare_sglang_quantized_generation,
    validate_sglang_quantized_refit_backend,
)


@pytest.mark.parametrize("scheme", ["bf16", "mxfp8"])
def test_get_sglang_quantization_scheme_accepts_supported_values(scheme: str) -> None:
    assert get_sglang_quantization_scheme({"scheme": scheme}) == scheme


@pytest.mark.parametrize("config", [{}, {"modules_to_not_convert": []}])
def test_get_sglang_quantization_scheme_requires_scheme(config: dict) -> None:
    with pytest.raises(KeyError, match="scheme"):
        get_sglang_quantization_scheme(config)


def test_get_sglang_quantization_scheme_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match=r"got 'fp4'"):
        get_sglang_quantization_scheme({"scheme": "fp4"})


def test_high_precision_substrings_combine_extra_head_tail_and_deduplicate() -> None:
    result = get_dynamic_high_precision_substrings(
        quantization_config={
            "extra_high_precision_layers_hf": [
                "model.layers.2.mlp.experts",
                "model.layers.0.",
            ],
            "modules_to_not_convert": [
                "model.layers.2.mlp.experts",
                "lm_head",
            ],
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        },
        num_hidden_layers=4,
    )

    assert result == (
        "model.layers.2.mlp.experts",
        "model.layers.0.",
        "lm_head",
        "layers.0.",
        "layers.3.",
    )


@pytest.mark.parametrize(
    "layer_prefix",
    [
        "model.layers",
        "model.language_model.layers",
        "backbone.layers",
        "layers",
    ],
)
def test_layer_prefix_for_layer_one_does_not_match_layer_ten(
    layer_prefix: str,
) -> None:
    substrings = get_dynamic_high_precision_substrings(
        quantization_config={"num_layers_at_start_in_bf16": 2},
        num_hidden_layers=12,
    )

    layer_one = f"{layer_prefix}.1.mlp.experts.0.down_proj.weight"
    layer_ten = f"{layer_prefix}.10.mlp.experts.0.down_proj.weight"
    assert any(substring in layer_one for substring in substrings)
    assert not any(substring in layer_ten for substring in substrings)


def test_prepare_sglang_quantized_generation_resolves_shared_startup(
    monkeypatch,
) -> None:
    observed: dict = {}

    def fake_ensure(*, model_path, quantization_config):
        observed["model_path"] = model_path
        observed["quantization_config"] = quantization_config
        return "/shared/model-mxfp8"

    monkeypatch.setattr(
        quantization_utils, "ensure_sglang_quantized_checkpoint", fake_ensure
    )
    generation_config = {
        "sglang_cfg": {"quantization": {"scheme": "mxfp8"}},
    }
    policy_config = {
        "model_name": "NVIDIA/model",
        "megatron_cfg": {"enabled": True},
    }

    prepare_sglang_quantized_generation(
        generation_config=generation_config,
        policy_config=policy_config,
    )

    assert generation_config["sglang_cfg"]["model_path"] == "/shared/model-mxfp8"
    assert observed == {
        "model_path": "NVIDIA/model",
        "quantization_config": {"scheme": "mxfp8"},
    }


def test_atomic_high_precision_expands_fused_linear_and_moe_modules() -> None:
    result = expand_sglang_atomic_high_precision_substrings(
        weight_names=[
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.2.mlp.experts.0.gate_proj.weight",
            "model.layers.2.mlp.experts.0.up_proj.weight",
            "model.layers.2.mlp.experts.0.down_proj.weight",
        ],
        skip_weight_substrings=(
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.mlp.gate_proj",
            "model.layers.2.mlp.experts.0.gate_proj",
        ),
    )

    assert "model.layers.1.self_attn.k_proj" in result
    assert "model.layers.1.self_attn.v_proj" in result
    assert "model.layers.1.mlp.up_proj" in result
    assert "model.layers.2.mlp.experts" in result


def test_quantized_refit_requires_megatron_backend() -> None:
    validate_sglang_quantized_refit_backend(scheme="bf16", use_megatron=False)
    validate_sglang_quantized_refit_backend(scheme="mxfp8", use_megatron=True)


def test_quantized_refit_rejects_a_non_megatron_backend() -> None:
    """The rejecting branch is the only reason the guard exists.

    Without this, neutering ``validate_sglang_quantized_refit_backend`` to a
    bare ``return`` leaves the whole suite green -- and that guard is what
    stops a DTensor/FSDP policy from booting SGLang off a quantized checkpoint
    it can only ever refit in BF16.
    """
    with pytest.raises(NotImplementedError, match="requires a Megatron policy"):
        validate_sglang_quantized_refit_backend(scheme="mxfp8", use_megatron=False)


@pytest.mark.parametrize(
    ("quantization_config", "num_hidden_layers", "exception", "match"),
    [
        (
            {"num_layers_at_start_in_bf16": -1},
            4,
            ValueError,
            "non-negative",
        ),
        (
            {"num_layers_at_end_in_bf16": True},
            4,
            TypeError,
            "must be an integer",
        ),
        (
            {"num_layers_at_start_in_bf16": 1.5},
            4,
            TypeError,
            "must be an integer",
        ),
        (
            {
                "num_layers_at_start_in_bf16": 3,
                "num_layers_at_end_in_bf16": 2,
            },
            4,
            ValueError,
            "exceed",
        ),
        (
            {"num_layers_at_end_in_bf16": 1},
            0,
            ValueError,
            "must be positive",
        ),
    ],
)
def test_high_precision_substrings_reject_invalid_layer_counts(
    quantization_config: dict,
    num_hidden_layers: int,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        get_dynamic_high_precision_substrings(
            quantization_config=quantization_config,
            num_hidden_layers=num_hidden_layers,
        )
