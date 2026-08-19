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
from pydantic import ValidationError

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.models.policy.draft_config import (
    DraftOptimizerConfig,
    Eagle3DraftConfig,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[4]

register_omegaconf_resolvers()


def test_eagle3_draft_config_preserves_legacy_defaults() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    config = Eagle3DraftConfig.model_validate({})

    assert config.model_dump() == {
        "speculator_type": "eagle3",
        "enabled": False,
        "model_name": None,
        "loss_weight": 0.1,
        "num_layers": None,
        "aux_layer_indices": None,
        "optimizer": None,
    }


def test_draft_optimizer_config_is_typed() -> None:
    draft = Eagle3DraftConfig(
        enabled=True,
        optimizer={
            "lr": 1.0e-5,
            "min_lr": 1.0e-6,
            "weight_decay": 0.02,
        },
    )

    assert draft.optimizer == DraftOptimizerConfig(
        lr=1.0e-5,
        min_lr=1.0e-6,
        weight_decay=0.02,
    )


def test_eagle3_draft_config_accepts_legacy_mapping_without_speculator_type() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    config = Eagle3DraftConfig.model_validate(
        {"enabled": True, "model_name": "draft", "loss_weight": 0.25}
    )

    assert config.speculator_type == "eagle3"
    assert config.enabled is True
    assert config.model_name == "draft"
    assert config.loss_weight == 0.25


def test_eagle3_draft_config_preserves_extra_legacy_keys() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    config = Eagle3DraftConfig.model_validate({"legacy_extension": 7})

    assert config.model_dump()["legacy_extension"] == 7


def test_eagle3_draft_config_rejects_unknown_speculator_type() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    with pytest.raises(ValidationError, match="eagle3"):
        Eagle3DraftConfig.model_validate({"speculator_type": "dflash"})


def test_grpo_master_config_parses_nested_draft_model() -> None:
    path = REPO_ROOT / "examples/configs/grpo_math_1B.yaml"
    raw = OmegaConf.to_container(load_config(path), resolve=True)

    config = MasterConfig(**raw)

    assert isinstance(config.policy["draft"], Eagle3DraftConfig)
    assert config.policy["draft"].speculator_type == "eagle3"


def test_policy_config_may_omit_draft_block() -> None:
    path = REPO_ROOT / "examples/configs/grpo_math_1B.yaml"
    raw = OmegaConf.to_container(load_config(path), resolve=True)
    del raw["policy"]["draft"]

    config = MasterConfig(**raw)

    assert "draft" not in config.policy


def test_omitted_draft_config_does_not_request_refit() -> None:
    from nemo_rl.models.policy.draft_config import draft_refit_enabled

    assert draft_refit_enabled(None) is False
