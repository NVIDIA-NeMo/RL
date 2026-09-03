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

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config_with_inheritance,
    register_omegaconf_resolvers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RECIPE_NAME = "grpo-qwen3.5-35ba3b-4n4g-megatron-ep16tp2-mxfp8-trtllm.yaml"
SCRIPT_NAME = "grpo-qwen3.5-35ba3b-4n4g-megatron-ep16tp2-mxfp8-trtllm.sh"


def _load_recipe() -> dict[str, Any]:
    register_omegaconf_resolvers()
    recipe_path = PROJECT_ROOT / "examples/configs/recipes/llm" / RECIPE_NAME
    recipe = OmegaConf.to_container(
        load_config_with_inheritance(recipe_path), resolve=True
    )
    assert isinstance(recipe, dict)
    return recipe


def test_qwen35_mxfp8_recipe_uses_sync_full_loader_and_trtllm() -> None:
    recipe = _load_recipe()
    generation = recipe["policy"]["generation"]
    vllm_cfg = generation["vllm_cfg"]

    assert recipe["cluster"]["gpus_per_node"] == 4
    assert recipe["cluster"]["num_nodes"] == 4
    assert recipe["cluster"]["segment_size"] == 4
    assert generation.get("refit_transport") is None
    assert generation["colocated"]["enabled"] is True
    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["enforce_eager"] is False
    assert vllm_cfg["expert_parallel_size"] == 2
    assert generation["vllm_kwargs"] == {
        "moe_backend": "flashinfer_trtllm",
        "expert_placement_strategy": "linear",
    }


def test_qwen35_mxfp8_recipe_keeps_gdn_and_vision_in_bf16() -> None:
    recipe = _load_recipe()
    patterns = recipe["policy"]["generation"]["vllm_cfg"][
        "quantization_ignore_patterns"
    ]

    assert patterns == [
        "language_model.model.layers.*.linear_attn.*",
        "visual.*",
        "lm_head",
    ]


def test_qwen35_mxfp8_test_script_requests_one_topology_segment() -> None:
    script_path = PROJECT_ROOT / "tests/test_suites/llm" / SCRIPT_NAME
    script = script_path.read_text(encoding="utf-8")

    assert "SEGMENT_SIZE=4" in script


@pytest.mark.vllm
def test_qwen35_mxfp8_patterns_match_vllm_module_names() -> None:
    pytest.importorskip("vllm")

    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    recipe = _load_recipe()
    patterns = recipe["policy"]["generation"]["vllm_cfg"][
        "quantization_ignore_patterns"
    ]
    modelopt_config = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "ignore": patterns,
            "ignored_layers": ["lm_head"],
        }
    )

    assert modelopt_config.is_layer_excluded(
        "language_model.model.layers.0.linear_attn.in_proj"
    )
    assert modelopt_config.is_layer_excluded("visual.blocks.0.mlp.linear_fc2")
    assert modelopt_config.is_layer_excluded("lm_head")
    assert not modelopt_config.is_layer_excluded(
        "language_model.model.layers.0.mlp.experts"
    )
