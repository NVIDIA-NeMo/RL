# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_rl.algorithms.dpo import _validate_stop_after_step
from nemo_rl.algorithms.loss import MPOLossConfig, MPOLossFn
from nemo_rl.algorithms.mpo import (
    MasterConfig,
    MPOConfig,
    MPOSaveState,
    _update_reward_shift,
    _validate_pair_safe_packing,
)
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def _save_state() -> MPOSaveState:
    return MPOSaveState(
        epoch=0,
        step=0,
        total_steps=0,
        consumed_samples=0,
        total_valid_tokens=0,
        reward_shift=1.0,
        reward_shift_num_updates=3,
    )


def _loss_fn() -> MPOLossFn:
    return MPOLossFn(
        MPOLossConfig(
            reference_policy_kl_penalty=1.0,
            reward_shift=1.0,
            reward_shift_momentum=0.5,
        )
    )


def test_reward_shift_update_uses_all_microbatch_statistics():
    save_state = _save_state()
    loss_fn = _loss_fn()
    train_results = {
        "all_mb_metrics": {
            "bco_reward_sum": [-2.0, -8.0],
            "bco_reward_count": [1.0, 3.0],
        }
    }

    _update_reward_shift(train_results, loss_fn, save_state)

    assert save_state.reward_shift == pytest.approx(-0.75)
    assert save_state.reward_shift_num_updates == 4
    assert loss_fn.reward_shift == pytest.approx(save_state.reward_shift)
    assert train_results["all_mb_metrics"]["reward_shift"] == pytest.approx(
        [save_state.reward_shift]
    )


def test_reward_shift_update_ignores_empty_optimizer_step():
    save_state = _save_state()
    loss_fn = _loss_fn()

    _update_reward_shift({"all_mb_metrics": {}}, loss_fn, save_state)

    assert save_state.reward_shift == 1.0
    assert save_state.reward_shift_num_updates == 3


def test_mpo_stop_after_step_does_not_change_final_step_limit():
    config = MPOConfig(max_num_steps=100, stop_after_step=50)

    _validate_stop_after_step(config, "mpo")

    assert config.max_num_steps == 100
    assert config.stop_after_step == 50

    for invalid_boundary in (0, 101):
        with pytest.raises(ValueError, match="between 1 and mpo.max_num_steps"):
            _validate_stop_after_step(
                MPOConfig(max_num_steps=100, stop_after_step=invalid_boundary),
                "mpo",
            )


def _packed_policy_config() -> PolicyConfig:
    return cast(
        PolicyConfig,
        {
            "train_global_batch_size": 4,
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {
                "enabled": True,
                "algorithm": "modified_first_fit_decreasing",
                "fuse_loss": True,
                "pair_grouping_key": "pair_index",
                "max_sequences_per_bin": 1,
            },
            "megatron_cfg": {
                "enabled": True,
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            },
        },
    )


def _cluster_config() -> ClusterConfig:
    return ClusterConfig(num_nodes=1, gpus_per_node=4)


def test_mpo_validates_pair_safe_sequence_packing():
    _validate_pair_safe_packing(_packed_policy_config(), _cluster_config())


def test_mpo_rejects_non_pair_grouping_key():
    policy_config = _packed_policy_config()
    sequence_packing = cast(dict[str, Any], policy_config["sequence_packing"])
    sequence_packing["pair_grouping_key"] = "task_name"

    with pytest.raises(ValueError, match="pair_grouping_key='pair_index'"):
        _validate_pair_safe_packing(policy_config, _cluster_config())


def test_mpo_rejects_non_fused_or_automodel_packed_loss():
    policy_config = _packed_policy_config()
    sequence_packing = cast(dict[str, Any], policy_config["sequence_packing"])
    sequence_packing["fuse_loss"] = False
    with pytest.raises(ValueError, match="fuse_loss=True"):
        _validate_pair_safe_packing(policy_config, _cluster_config())

    policy_config = _packed_policy_config()
    megatron_config = cast(dict[str, Any], policy_config["megatron_cfg"])
    megatron_config["enabled"] = False
    with pytest.raises(ValueError, match="requires the Megatron backend"):
        _validate_pair_safe_packing(policy_config, _cluster_config())


def test_mpo_rejects_pair_batch_not_divisible_by_dp():
    policy_config = _packed_policy_config()
    policy_config["train_global_batch_size"] = 3

    with pytest.raises(
        ValueError,
        match=r"train_global_batch_size=3, DP=2.*1 nodes x 4 GPUs",
    ):
        _validate_pair_safe_packing(policy_config, _cluster_config())


def test_vlm_mpo_exemplar_validates_without_dpo_config():
    root = Path(__file__).parents[3]
    register_omegaconf_resolvers()
    config = load_config(root / "examples/configs/vlm_mpo.yaml")
    master_config = MasterConfig.model_validate(
        OmegaConf.to_container(config, resolve=True)
    )

    train_config = master_config.data["train"]
    assert isinstance(train_config, dict)
    assert train_config["dataset_name"] == "MMPRPreference"
    assert master_config.data["validation"] is None
    assert "dpo" not in (master_config.model_extra or {})


def test_nemotron_omni_mpo_recipe_validates():
    root = Path(__file__).parents[3]
    recipe_path = (
        root
        / "examples/configs/recipes/vlm"
        / "vlm_mpo-nemotron-omni-30ba3b-mmpr-1n8g-megatron-tp8.v1.yaml"
    )
    register_omegaconf_resolvers()
    config = load_config(recipe_path)
    MasterConfig.model_validate(OmegaConf.to_container(config, resolve=True))
