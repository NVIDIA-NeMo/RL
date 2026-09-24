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

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

RECIPE_DIR = Path(__file__).parents[3] / "examples/configs/recipes/llm"
RECIPES = tuple(sorted(RECIPE_DIR.rglob("*.yaml")))


@pytest.mark.parametrize(
    "recipe_path", RECIPES, ids=lambda path: str(path.relative_to(RECIPE_DIR))
)
def test_automodel_moe_recipes_use_hybridep_or_explicit_torch(
    recipe_path: Path,
) -> None:
    register_omegaconf_resolvers()
    config = load_config(recipe_path)
    dtensor_cfg = OmegaConf.select(config, "policy.dtensor_cfg")

    if (
        dtensor_cfg is None
        or not dtensor_cfg.enabled
        or dtensor_cfg.get("expert_parallel_size", 1) <= 1
    ):
        pytest.skip("Not an AutoModel expert-parallel recipe")

    backend = dtensor_cfg.automodel_kwargs.backend
    assert "enable_deepep" not in backend
    assert backend.get("dispatcher") in {"hybridep", "torch"}
    if backend.dispatcher == "hybridep":
        assert backend.get("experts") is not None, (
            f"{recipe_path.name}: HybridEP must explicitly select an experts backend"
        )
        assert config.policy.make_sequence_length_divisible_by % 64 == 0, (
            f"{recipe_path.name}: HybridEP input width must be padded to a multiple of 64"
        )


def test_gemma4_cp_keeps_local_hybridep_inputs_aligned() -> None:
    register_omegaconf_resolvers()
    config = load_config(
        RECIPE_DIR / "dapo-gemma4-26ba4b-it-4n8g-fsdp2ep16cp2-automodel.yaml"
    )
    # Gemma shards contiguous sequences without padding local widths to 64.
    global_alignment = 64 * config.policy.dtensor_cfg.context_parallel_size
    assert config.policy.make_sequence_length_divisible_by % global_alignment == 0
    assert config.policy.dynamic_batching.sequence_length_round % global_alignment == 0


@pytest.mark.parametrize(
    ("recipe_name", "dispatcher", "experts"),
    [
        ("dpo-nanov3-30B3AB-1n4g-fsdp4ep4-automodel.yaml", "torch", "torch_mm"),
        ("grpo-glm47-flash-4n8g-automodel.yaml", "hybridep", "torch_mm"),
        ("grpo-minimax-m27-dapo-8n8g-automodel.yaml", "hybridep", "torch_mm"),
        ("grpo-moonlight-16b-automodel-1n8g-ep8.yaml", "hybridep", "torch_mm"),
        (
            "grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.v2.yaml",
            "hybridep",
            "torch_mm",
        ),
        ("dapo-gemma4-26ba4b-it-4n8g-fsdp2-automodel.yaml", "hybridep", "gmm"),
        ("dapo-nanov3.5-30BA3B-4n8g-automodel.yaml", "hybridep", "gmm"),
        ("grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml", "hybridep", "gmm"),
        ("grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml", "hybridep", "gmm"),
    ],
)
def test_automodel_recipes_preserve_backend_choices(
    recipe_name: str, dispatcher: str, experts: str
) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_DIR / recipe_name)
    backend = config.policy.dtensor_cfg.automodel_kwargs.backend
    assert backend.dispatcher == dispatcher
    assert backend.experts == experts
