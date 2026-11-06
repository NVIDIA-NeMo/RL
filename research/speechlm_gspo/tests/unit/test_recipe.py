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
from speechlm_gspo.actor_environments import GENERATION_WORKER, POLICY_WORKER
from speechlm_gspo.config import SpeechLMNativeConfig

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.outdated_config_checks import check_outdated_config

RECIPES = sorted((Path(__file__).parents[2] / "configs" / "recipes").glob("*.yaml"))


@pytest.mark.parametrize("path", RECIPES, ids=lambda path: path.stem)
def test_recipe_validates_with_research_workers(path):
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(str(path)), resolve=True)
    check_outdated_config(config)
    policy = MasterConfig(**config).policy
    assert policy["worker_extension_cls_fqn"] == POLICY_WORKER
    assert policy["generation"]["worker_extension_cls_fqn"] == GENERATION_WORKER
    native = SpeechLMNativeConfig(
        **policy["automodel_cfg"]["automodel_kwargs"]["native_model"]
    )
    assert len(set(native.trainable_names)) == native.expected_trainable_tensors
    assert native.max_expanded_sequence_length >= policy["max_total_sequence_length"]
