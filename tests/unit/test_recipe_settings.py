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

import shlex
from pathlib import Path

import pytest

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

RECIPE_ROOT = Path(__file__).parents[2] / "examples/configs/recipes"


@pytest.mark.parametrize(
    "script_name",
    [
        "grpo_megatron_generation_gym_single_controller.sh",
        "grpo_megatron_generation_colocated_reshard_gym_single_controller.sh",
    ],
)
def test_megatron_single_controller_gym_scripts_recompute_kv(script_name: str) -> None:
    register_omegaconf_resolvers()
    project_root = Path(__file__).parents[2]
    script = (project_root / "tests/functional" / script_name).read_text()
    command = next(
        line
        for line in script.replace("\\\n", " ").splitlines()
        if line.startswith("uv run coverage run ")
    )
    arguments = shlex.split(command)
    config_path = arguments[arguments.index("--config") + 1].replace(
        "$PROJECT_ROOT", str(project_root)
    )
    overrides = [
        argument
        for argument in arguments
        if argument.lstrip("+").startswith(("policy.generation.", "async_rl."))
    ]
    config = parse_hydra_overrides(load_config(config_path), overrides)
    assert config.policy.generation.backend == "megatron"
    assert (
        config.policy.generation.mcore_generation_config.kv_cache_management_mode
        == "recompute"
    )


@pytest.mark.parametrize(
    "recipe_name",
    [
        "llm/sft-nanov3-30BA3B-2n8g-automodel-ep8.yaml",
        "llm/sft-nanov3-30BA3B-2n4g-automodel-ep4.yaml",
        "llm/sft-gpt-oss-20b-1n8g-automodel-ep8.yaml",
        "llm/sft-gpt-oss-20b-1n4g-automodel-ep4.yaml",
        "vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2.yaml",
        "vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml",
    ],
)
def test_hybridep_regression_recipes_keep_legacy_deepep(recipe_name: str) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_ROOT / recipe_name)
    assert config.policy.automodel_cfg.automodel_kwargs.backend.dispatcher == "deepep"
    assert config.policy.make_sequence_length_divisible_by == 1


@pytest.mark.parametrize("context_parallel_size", [2, 4])
def test_gemma4_cp_keeps_local_hybridep_inputs_aligned(
    context_parallel_size: int,
) -> None:
    register_omegaconf_resolvers()
    config = load_config(
        RECIPE_ROOT / "llm/dapo-gemma4-26ba4b-it-4n8g-automodel-ep16cp2.yaml"
    )
    config.policy.automodel_cfg.context_parallel_size = context_parallel_size
    global_alignment = 64 * config.policy.automodel_cfg.context_parallel_size
    assert config.policy.make_sequence_length_divisible_by == global_alignment
    assert config.policy.dynamic_batching.sequence_length_round == global_alignment


@pytest.mark.parametrize(
    ("recipe_name", "dispatcher", "experts"),
    [
        ("llm/dpo-nanov3-30B3AB-1n4g-automodel-ep4.yaml", "torch", "torch_mm"),
        ("llm/grpo-glm47-flash-4n8g-automodel-ep32.yaml", "hybridep", "torch_mm"),
        ("llm/grpo-minimax-m27-dapo-8n8g-automodel-ep64.yaml", "hybridep", "torch_mm"),
        ("llm/grpo-moonlight-16b-automodel-1n8g-ep8.yaml", "hybridep", "torch_mm"),
        (
            "llm/grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.v2.yaml",
            "hybridep",
            "torch_mm",
        ),
        ("llm/dapo-gemma4-26ba4b-it-4n8g-automodel-ep32.yaml", "hybridep", "gmm"),
        ("llm/dapo-nanov3.5-30BA3B-4n8g-automodel-ep8.yaml", "hybridep", "gmm"),
        ("llm/grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml", "hybridep", "gmm"),
        ("llm/grpo-qwen3.5-35ba3b-dapo-4n8g-automodel-ep32.yaml", "hybridep", "gmm"),
        (
            "vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2.yaml",
            "deepep",
            "torch_mm",
        ),
        (
            "vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml",
            "deepep",
            "torch_mm",
        ),
        (
            "vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml",
            "hybridep",
            "gmm",
        ),
        ("vlm/vlm_grpo-gemma4-e4b-geo3k-1n8g-automodel.yaml", "hybridep", "gmm"),
    ],
)
def test_automodel_recipes_preserve_backend_choices(
    recipe_name: str, dispatcher: str, experts: str
) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_ROOT / recipe_name)
    backend = config.policy.automodel_cfg.automodel_kwargs.backend
    assert backend.dispatcher == dispatcher
    assert backend.experts == experts


@pytest.mark.parametrize(
    ("recipe_name", "expected_async_save"),
    [
        ("grpo-deepseek-v3-32n4g.yaml", False),
        ("grpo-deepseek-v3-64n4g.yaml", True),
        ("grpo-deepseek-v3-64n4g-mxfp8-rollout.yaml", True),
    ],
)
def test_deepseek_performance_recipes_preserve_checkpoint_save_mode(
    recipe_name: str, expected_async_save: bool
) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_ROOT / "llm/performance" / recipe_name)
    assert config.policy.megatron_cfg.checkpoint.async_save is expected_async_save
