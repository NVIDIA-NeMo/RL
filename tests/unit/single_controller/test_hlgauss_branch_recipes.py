# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public branch recipes select PPO and compose without cluster-local files."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.critic_training_mask import build_critic_sample_mask
from nemo_rl.algorithms.sc_critic_pretrain import CriticPretrainConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    is_ppo_run,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

RECIPES = Path(__file__).resolve().parents[3] / "examples" / "hlgauss_ppo_sc"


def test_offline_pretraining_recipe_needs_only_model_and_bank_paths():
    register_omegaconf_resolvers()
    config = load_config(RECIPES / "critic_pretrain.yaml")
    config.policy.model_name = "test/nano"
    config.critic_pretrain.bank_dirs = ["/test/rollout-bank"]
    model = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    offline = CriticPretrainConfig.model_validate(model.critic_pretrain)
    assert is_ppo_run(model)
    assert offline.train_on_env_masked and offline.train_on_truncated
    assert model.data["train"]["data_path"] == ""


@pytest.mark.parametrize(
    "name",
    ["nano_ctrunc", "nano_sc", "nano_sc_failures", "nano_sc_token_credit"],
)
def test_recipe_selects_single_controller_ppo(name):
    register_omegaconf_resolvers()
    config = load_config(RECIPES / f"{name}.yaml")
    config.policy.model_name = "test/nano"
    config.data.train.data_path = "/test/train.jsonl"
    model = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    validate_single_controller_config(model)
    assert is_ppo_run(model)
    assert model.ppo.async_ppo is None
    assert model.value["head"]["type"] == "hl_gauss"
    assert model.ppo.critic_train_on_env_masked
    assert model.ppo.critic_train_on_truncated
    assert model.loss_fn.score_centering == (name != "nano_ctrunc")
    assert model.ppo.overlong_filtering == (name in {"nano_ctrunc", "nano_sc"})
    assert model.policy["train_global_batch_size"] == (
        model.ppo.num_prompts_per_step * model.ppo.num_generations_per_prompt
    )
    credit = name == "nano_sc_token_credit"
    assert bool(config.ppo.get("malformed_think_credit")) == credit
    if credit:
        assert not config.reward_penalties.penalize_malformed_think_tag


def test_critic_restoration_preserves_invalid_and_sequence_error_masks():
    # Valid / truncation / timeout / poisoned infrastructure / sequence-error.
    original = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])
    actor = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    result = build_critic_sample_mask(
        original,
        actor,
        torch.tensor([False, False, True, True, False]),
        torch.tensor([False, True, False, False, False]),
        overlong_filtering=True,
        train_on_env_masked=True,
        train_on_truncated=True,
        seq_error_keep=torch.tensor([True, True, True, True, False]),
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(actor, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
