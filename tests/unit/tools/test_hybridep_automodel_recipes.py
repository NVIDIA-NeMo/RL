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
AUTOMODEL_RECIPES = tuple(sorted(RECIPE_DIR.glob("*automodel*.yaml")))


@pytest.mark.parametrize("recipe_path", AUTOMODEL_RECIPES, ids=lambda path: path.name)
def test_automodel_moe_recipes_use_hybridep_or_explicit_torch(
    recipe_path: Path,
) -> None:
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(recipe_path), resolve=True)
    dtensor_cfg = config["policy"].get("dtensor_cfg", {})

    if dtensor_cfg.get("expert_parallel_size", 1) <= 1:
        return

    backend = dtensor_cfg.get("automodel_kwargs", {}).get("backend", {})
    assert backend.get("enable_deepep") is not True
    assert backend.get("dispatcher") in {"hybridep", "torch"}
