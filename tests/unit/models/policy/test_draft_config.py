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
    DFlashDraftConfig,
    DSparkDraftConfig,
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


def test_dflash_accepts_excluded_legacy_aux_layer_field() -> None:
    config = DFlashDraftConfig(
        enabled=True,
        aux_layer_indices=None,
        gamma=5,
        anchors_per_sample=4,
        mask_token_id=151669,
        target_hidden_state_layer_ids=[1, 17, 33],
    )

    assert "aux_layer_indices" not in config.model_dump()


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


def test_eagle3_draft_config_rejects_near_miss_typo_keys() -> None:
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    with pytest.raises(ValidationError, match="did you mean 'enabled'"):
        Eagle3DraftConfig.model_validate({"enalbed": True})
    with pytest.raises(ValidationError, match="did you mean 'loss_weight'"):
        Eagle3DraftConfig.model_validate({"loss_wieght": 0.5})


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


def test_grpo_master_config_preserves_legacy_missing_method_discriminator() -> None:
    path = REPO_ROOT / "examples/configs/grpo_math_1B.yaml"
    raw = OmegaConf.to_container(load_config(path), resolve=True)
    del raw["policy"]["draft"]["speculator_type"]

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


def test_dflash_config_validates_complete_training_contract() -> None:
    config = DFlashDraftConfig(
        enabled=True,
        gamma=5,
        anchors_per_sample=4,
        mask_token_id=151665,
        target_hidden_state_layer_ids=[1, 17, 33],
    )

    assert config.speculator_type == "dflash"
    assert config.gamma == 5
    assert config.target_hidden_state_layer_ids == [1, 17, 33]
    assert config.max_cp_boundary_exclusion_fraction == 0.25


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_dflash_config_rejects_invalid_cp_boundary_exclusion_fraction(
    value: float,
) -> None:
    """Catches exclusion thresholds that cannot represent a fraction."""
    with pytest.raises(ValidationError):
        DFlashDraftConfig(
            enabled=True,
            gamma=5,
            anchors_per_sample=4,
            mask_token_id=151665,
            target_hidden_state_layer_ids=[1, 17, 33],
            max_cp_boundary_exclusion_fraction=value,
        )


def test_dspark_config_preserves_public_qwen3_8b_contract() -> None:
    config = DSparkDraftConfig(
        enabled=True,
        model_name="deepseek-ai/dspark_qwen3_8b_block7",
        model_revision="03326e5043815da1f81b109078b2889737c26017",  # pragma: allowlist secret
        block_size=7,
        anchors_per_sample=512,
        mask_token_id=151669,
        target_hidden_state_layer_ids=[1, 9, 17, 25, 33],
        markov_rank=256,
        confidence_enabled=True,
        confidence_with_markov=True,
    )

    assert config.speculator_type == "dspark"
    assert config.model_name == "deepseek-ai/dspark_qwen3_8b_block7"
    assert config.model_revision == (
        "03326e5043815da1f81b109078b2889737c26017"  # pragma: allowlist secret
    )
    assert config.block_size == 7
    assert config.draft_vocab_size is None
    assert config.target_hidden_state_layer_ids == [1, 9, 17, 25, 33]


def test_dspark_config_rejects_vocab_different_from_live_target() -> None:
    config = DSparkDraftConfig(
        enabled=True,
        block_size=7,
        anchors_per_sample=2,
        mask_token_id=151669,
        target_hidden_state_layer_ids=[1, 9, 17, 25, 33],
        draft_vocab_size=32_000,
    )

    with pytest.raises(ValueError, match="must match the live target vocabulary"):
        config.resolve_draft_vocab_size(target_vocab_size=151_936)


def test_dspark_config_resolves_target_owned_vocab() -> None:
    values = {
        "enabled": True,
        "block_size": 7,
        "anchors_per_sample": 2,
        "mask_token_id": 151669,
        "target_hidden_state_layer_ids": [1, 9, 17, 25, 33],
    }

    assert (
        DSparkDraftConfig(**values).resolve_draft_vocab_size(
            target_vocab_size=151_936
        )
        == 151_936
    )
    assert (
        DSparkDraftConfig(
            **values,
            draft_vocab_size=151_936,
        ).resolve_draft_vocab_size(target_vocab_size=151_936)
        == 151_936
    )


def test_dspark_update_probe_is_explicitly_opt_in() -> None:
    values = {
        "block_size": 7,
        "anchors_per_sample": 2,
        "mask_token_id": 151669,
        "target_hidden_state_layer_ids": [1, 9, 17, 25, 33],
    }

    assert DSparkDraftConfig(**values).update_probe_enabled is False
    assert DSparkDraftConfig(
        **values, update_probe_enabled=True
    ).update_probe_enabled is True


def test_dspark_config_rejects_dflash_gamma_alias() -> None:
    with pytest.raises(ValidationError, match="gamma"):
        DSparkDraftConfig.model_validate(
            {
                "enabled": True,
                "gamma": 7,
                "anchors_per_sample": 512,
                "mask_token_id": 151669,
                "target_hidden_state_layer_ids": [1, 9, 17, 25, 33],
            }
        )


def test_dflash_config_accepts_only_null_inherited_eagle_taps() -> None:
    values = {
        "enabled": True,
        "gamma": 5,
        "anchors_per_sample": 4,
        "mask_token_id": 151665,
        "target_hidden_state_layer_ids": [1, 17, 33],
        "aux_layer_indices": None,
    }

    config = DFlashDraftConfig.model_validate(values)

    assert "aux_layer_indices" not in config.model_dump()
    with pytest.raises(ValidationError, match="aux_layer_indices"):
        DFlashDraftConfig.model_validate({**values, "aux_layer_indices": [1]})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"gamma": 0}, "greater than 0"),
        ({"anchors_per_sample": 0}, "greater than 0"),
        ({"mask_token_id": -1}, "greater than or equal to 0"),
        ({"target_hidden_state_layer_ids": []}, "at least 1"),
        ({"target_hidden_state_layer_ids": [1, 1]}, "unique"),
    ],
)
def test_dflash_config_rejects_invalid_plan_before_model_build(
    override: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "enabled": True,
        "gamma": 5,
        "anchors_per_sample": 4,
        "mask_token_id": 151665,
        "target_hidden_state_layer_ids": [1, 17, 33],
    }
    values.update(override)

    with pytest.raises(ValidationError, match=message):
        DFlashDraftConfig.model_validate(values)


def test_qwen3_8b_dflash_recipe_pairs_public_drafter_with_exact_target() -> None:
    path = (
        REPO_ROOT
        / "examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash.yaml"
    )
    raw = OmegaConf.to_container(load_config(path), resolve=True)

    config = MasterConfig(**raw)

    assert config.policy["model_name"] == "Qwen/Qwen3-8B"
    assert config.policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert isinstance(config.policy["draft"], DFlashDraftConfig)
    assert config.policy["draft"].model_name == "z-lab/Qwen3-8B-DFlash-b16"
    assert config.policy["draft"].target_hidden_state_layer_ids == [1, 9, 17, 25, 33]
    assert config.policy["generation"]["vllm_kwargs"]["speculative_config"] == {
        "method": "dflash",
        "model": "z-lab/Qwen3-8B-DFlash-b16",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
    }
    assert config.policy["generation"]["vllm_cfg"]["enforce_eager"] is False
    assert config.policy["generation"]["vllm_kwargs"]["compilation_config"] == {
        "backend": "eager",
        "cudagraph_mode": "PIECEWISE",
    }


def test_qwen3_8b_dspark_recipe_keeps_cuda_graphs_with_eager_backend() -> None:
    path = (
        REPO_ROOT
        / "examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dspark.yaml"
    )
    raw = OmegaConf.to_container(load_config(path), resolve=True)

    config = MasterConfig(**raw)

    assert config.policy["generation"]["vllm_cfg"]["enforce_eager"] is False
    assert config.policy["generation"]["vllm_kwargs"]["compilation_config"] == {
        "backend": "eager",
        "cudagraph_mode": "PIECEWISE",
    }
