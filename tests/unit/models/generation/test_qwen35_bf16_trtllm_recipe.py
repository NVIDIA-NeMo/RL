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

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config_with_inheritance,
    register_omegaconf_resolvers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RECIPE_NAME = "grpo-qwen3.5-35ba3b-6n4g-async-1off-bf16-trtllm.yaml"


def _load_recipe() -> dict[str, Any]:
    register_omegaconf_resolvers()
    recipe_path = PROJECT_ROOT / "examples/configs/recipes/llm" / RECIPE_NAME
    recipe = OmegaConf.to_container(
        load_config_with_inheritance(recipe_path), resolve=True
    )
    assert isinstance(recipe, dict)
    return recipe


def test_qwen35_bf16_trtllm_recipe_uses_nccl_reshard() -> None:
    recipe = _load_recipe()
    generation = recipe["policy"]["generation"]

    assert recipe["loss_fn"]["use_importance_sampling_correction"] is True
    assert recipe["cluster"]["gpus_per_node"] == 4
    assert recipe["cluster"]["num_nodes"] == 6
    assert recipe["cluster"]["segment_size"] == 2
    async_grpo = recipe["grpo"]["async_grpo"]
    assert async_grpo["enabled"] is True
    assert async_grpo["max_trajectory_age_steps"] == 1
    assert async_grpo["in_flight_weight_updates"] is True
    assert generation["refit_transport"] == "nccl_reshard"
    assert generation["colocated"] == {
        "enabled": False,
        "resources": {"gpus_per_node": 4, "num_nodes": 2},
    }


def test_qwen35_bf16_trtllm_recipe_uses_supported_expert_layout() -> None:
    recipe = _load_recipe()
    generation = recipe["policy"]["generation"]
    vllm_cfg = generation["vllm_cfg"]

    assert vllm_cfg["precision"] == "bfloat16"
    assert vllm_cfg["tensor_parallel_size"] == 4
    assert vllm_cfg["expert_parallel_size"] == 4
    assert vllm_cfg["pipeline_parallel_size"] == 1
    assert vllm_cfg["enforce_eager"] is False
    assert generation["vllm_kwargs"] == {
        "moe_backend": "flashinfer_trtllm",
        "expert_placement_strategy": "linear",
    }
