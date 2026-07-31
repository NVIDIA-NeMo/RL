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
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()

MOE_8G_RECIPES = (
    "grpo-deepseek-v3-32n8g.yaml",
    "grpo-deepseek-v3-64n8g.yaml",
    "grpo-deepseek-v3-64n8g-async-1off.yaml",
    "grpo-deepseek-v3-64n8g-fp8-async-1off.yaml",
    "dapo-deepseek-v3-64n8g.v2.yaml",
    "grpo-nemotron3-super-120BA12B-32n8g.yaml",
    "grpo-nemotron3-super-120BA12B-32n8g-async-1off.yaml",
    "grpo-qwen3-235b-16n8g.yaml",
    "grpo-qwen3-235b-32n8g.yaml",
    "grpo-qwen3-235b-32n8g-async-1off.yaml",
    "grpo-qwen3-30ba3b-4n8g.yaml",
    "grpo-qwen3-30ba3b-4n8g-async-1off.yaml",
    "grpo-qwen3-30ba3b-24n8g-async-8off.yaml",
    "grpo-qwen3-30ba3b-4n8g-40K.yaml",
)

X86_HYBRIDEP_ENVIRONMENT = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "8",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": "128",
    "NVLINK_DOMAIN_SIZE": "8",
    "USE_MNNVL": "0",
}
X86_HYBRIDEP_ENVIRONMENT_KEYS = set(X86_HYBRIDEP_ENVIRONMENT)

DENSE_8G_RECIPES = (
    "grpo-llama3.1-8b-instruct-2n8g.yaml",
    "grpo-llama3.1-8b-instruct-2n8g-async-1off.yaml",
    "grpo-llama3.1-8b-instruct-2n8g-fp8-async-1off.yaml",
    "grpo-qwen3-32b-4n8g.yaml",
    "grpo-qwen3-32b-8n8g-async-1off.yaml",
)

FOUR_GPU_RECIPES = (
    "grpo-deepseek-v3-32n4g.yaml",
    "grpo-deepseek-v3-64n4g-async-1off.yaml",
    "grpo-qwen3-235b-16n4g.yaml",
    "grpo-qwen3-235b-32n4g-async-1off.yaml",
)


def _recipe_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "configs"
        / "recipes"
        / "llm"
        / "performance"
    )


def _resolve_recipe(recipe_name: str) -> dict[str, Any]:
    recipe_path = _recipe_dir() / recipe_name
    assert recipe_path.is_file(), f"Missing recipe: {recipe_path}"
    resolved = OmegaConf.to_container(load_config(recipe_path), resolve=True)
    assert isinstance(resolved, dict)
    return cast(dict[str, Any], resolved)


def _megatron_config(config: dict[str, Any]) -> dict[str, Any]:
    policy = config["policy"]
    assert isinstance(policy, dict)
    megatron_cfg = policy["megatron_cfg"]
    assert isinstance(megatron_cfg, dict)
    return megatron_cfg


def _environment(megatron_cfg: dict[str, Any]) -> dict[str, Any]:
    env_vars = megatron_cfg.get("env_vars")
    if env_vars is None:
        return {}
    assert isinstance(env_vars, dict)
    return env_vars


@pytest.mark.parametrize("recipe_name", MOE_8G_RECIPES)
def test_moe_8g_canonical_recipes_default_to_x86_hybridep(
    recipe_name: str,
) -> None:
    megatron_cfg = _megatron_config(_resolve_recipe(recipe_name))

    assert megatron_cfg["moe_token_dispatcher_type"] == "flex"
    assert megatron_cfg["moe_flex_dispatcher_backend"] == "hybridep"
    assert megatron_cfg["moe_hybridep_num_sms"] == 32
    assert X86_HYBRIDEP_ENVIRONMENT.items() <= _environment(megatron_cfg).items()


@pytest.mark.parametrize("recipe_name", MOE_8G_RECIPES)
def test_moe_8g_recipes_define_hybridep_directly_without_alltoall_peer(
    recipe_name: str,
) -> None:
    recipe_path = _recipe_dir() / recipe_name
    raw_config = OmegaConf.to_container(OmegaConf.load(recipe_path), resolve=False)
    assert isinstance(raw_config, dict)
    megatron_cfg = _megatron_config(cast(dict[str, Any], raw_config))

    assert megatron_cfg["moe_token_dispatcher_type"] == "flex"
    assert megatron_cfg["moe_flex_dispatcher_backend"] == "hybridep"
    assert megatron_cfg["moe_hybridep_num_sms"] == 32
    assert X86_HYBRIDEP_ENVIRONMENT.items() <= _environment(megatron_cfg).items()

    if recipe_name.endswith(".v2.yaml"):
        alltoall_name = recipe_name.replace(".v2.yaml", "-alltoall.v2.yaml")
    else:
        alltoall_name = recipe_name.replace(".yaml", "-alltoall.yaml")
    assert not (_recipe_dir() / alltoall_name).exists()


@pytest.mark.parametrize("recipe_name", DENSE_8G_RECIPES)
def test_dense_8g_recipes_do_not_select_hybridep(recipe_name: str) -> None:
    megatron_cfg = _megatron_config(_resolve_recipe(recipe_name))

    assert megatron_cfg["moe_token_dispatcher_type"] == "alltoall"
    assert "moe_flex_dispatcher_backend" not in megatron_cfg
    assert "moe_hybridep_num_sms" not in megatron_cfg
    assert not X86_HYBRIDEP_ENVIRONMENT_KEYS.intersection(_environment(megatron_cfg))


@pytest.mark.parametrize("recipe_name", FOUR_GPU_RECIPES)
def test_4g_descendants_use_hybridep_without_x86_environment(
    recipe_name: str,
) -> None:
    megatron_cfg = _megatron_config(_resolve_recipe(recipe_name))

    assert megatron_cfg["moe_token_dispatcher_type"] == "flex"
    assert megatron_cfg["moe_flex_dispatcher_backend"] == "hybridep"
    assert megatron_cfg["moe_hybridep_num_sms"] == 32
    assert not X86_HYBRIDEP_ENVIRONMENT_KEYS.intersection(_environment(megatron_cfg))
