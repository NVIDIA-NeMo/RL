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

from types import SimpleNamespace
from typing import cast

import pytest

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm.config import (
    VllmConfig,
    validate_vllm_quantization_config,
)


@pytest.mark.parametrize(
    "generation_config",
    [
        {
            "vllm_cfg": {
                "precision": "fp8",
                "is_mx": False,
                "refit_prequantize": True,
            }
        },
        {
            "vllm_cfg": {
                "precision": "fp8",
                "refit_prequantize": True,
            }
        },
        {
            "vllm_cfg": {
                "precision": "bfloat16",
                "refit_prequantize": True,
            },
            "quant_cfg": "examples/modelopt/quant_configs/nvfp4_a16.yaml",
            "real_quant": True,
        },
    ],
)
def test_refit_prequantize_requires_mxfp8(generation_config: dict) -> None:
    with pytest.raises(
        ValueError,
        match="refit_prequantize requires precision='fp8' and is_mx=true",
    ):
        validate_vllm_quantization_config(cast(VllmConfig, generation_config))


def test_refit_prequantize_must_be_boolean() -> None:
    generation_config = cast(
        VllmConfig,
        {
            "vllm_cfg": {
                "precision": "fp8",
                "is_mx": True,
                "refit_prequantize": "false",
            }
        },
    )

    with pytest.raises(ValueError, match="refit_prequantize must be a boolean"):
        validate_vllm_quantization_config(generation_config)


def test_refit_prequantize_accepts_mxfp8() -> None:
    generation_config = cast(
        VllmConfig,
        {
            "vllm_cfg": {
                "precision": "fp8",
                "is_mx": True,
                "refit_prequantize": True,
            }
        },
    )

    validate_vllm_quantization_config(generation_config)


def test_refit_prequantize_rejects_nccl_reshard() -> None:
    generation_config = cast(
        VllmConfig,
        {
            "refit_transport": "nccl_reshard",
            "vllm_cfg": {
                "precision": "fp8",
                "is_mx": True,
                "refit_prequantize": True,
            },
        },
    )

    with pytest.raises(ValueError, match="not supported with nccl_reshard"):
        validate_vllm_quantization_config(generation_config)


@pytest.mark.parametrize(
    "field",
    ["refit_cache_loader_routes"],
)
def test_refit_optimization_flags_must_be_boolean(field: str) -> None:
    generation_config = cast(
        VllmConfig,
        {
            "vllm_cfg": {
                "precision": "fp8",
                "is_mx": True,
                field: "true",
            }
        },
    )

    with pytest.raises(ValueError, match=rf"{field} must be a boolean"):
        validate_vllm_quantization_config(generation_config)


def test_refit_prequantize_validation_allows_omitted_vllm_cfg() -> None:
    generation_config = cast(VllmConfig, {"quant_cfg": None})

    validate_vllm_quantization_config(generation_config)


def test_configure_generation_config_validates_refit_prequantize() -> None:
    generation_config = cast(
        GenerationConfig,
        {
            "backend": "vllm",
            "stop_token_ids": None,
            "stop_strings": None,
            "vllm_cfg": {
                "precision": "bfloat16",
                "refit_prequantize": True,
            },
        },
    )
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)

    with pytest.raises(ValueError, match="requires precision='fp8' and is_mx=true"):
        configure_generation_config(generation_config, tokenizer)
