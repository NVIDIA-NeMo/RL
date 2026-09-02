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
from nemo_rl.models.policy import MegatronConfig, PolicyConfig, SequencePackingConfig
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
    return {"num_nodes": 1, "gpus_per_node": 4}


def test_mpo_validates_pair_safe_sequence_packing():
    _validate_pair_safe_packing(_packed_policy_config(), _cluster_config())


def test_mpo_rejects_non_pair_grouping_key():
    policy_config = _packed_policy_config()
    sequence_packing = cast(dict[str, Any], policy_config["sequence_packing"])
    sequence_packing["pair_grouping_key"] = "task_name"

    with pytest.raises(ValueError, match="pair_grouping_key='pair_index'"):
        _validate_pair_safe_packing(policy_config, _cluster_config())


def test_mpo_rejects_non_fused_or_dtensor_packed_loss():
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
    master_config = MasterConfig.model_validate(
        OmegaConf.to_container(config, resolve=True)
    )

    assert master_config.policy["offload_optimizer_for_logprob"] is False
    assert master_config.policy["tokenizer"]["fix_mistral_regex"] is True
    megatron_config = cast(MegatronConfig, master_config.policy["megatron_cfg"])
    assert megatron_config["enabled"] is True
    assert megatron_config["context_parallel_size"] == 1
    assert megatron_config["moe_router_dtype"] == "fp64"
    assert megatron_config["moe_router_load_balancing_type"] == "aux_loss"
    assert megatron_config["moe_router_bias_update_rate"] == 1.0e-3
    assert megatron_config["moe_permute_fusion"] is True
    assert megatron_config["freeze_audio_encoder"] is True
    assert megatron_config["freeze_audio_projector"] is True
    sequence_packing = cast(
        SequencePackingConfig, master_config.policy["sequence_packing"]
    )
    assert sequence_packing["enabled"] is True
    assert sequence_packing["pair_grouping_key"] == "pair_index"
    train_config = master_config.data["train"]
    assert isinstance(train_config, dict)
    assert train_config["dataset_name"] == "MMPRPreference"
    assert "max_samples" in train_config
    assert master_config.data["validation"] is None
    assert not master_config.logger["wandb_enabled"]


def test_nemotron_omni_nopack_parity_config_matches_legacy_run():
    root = Path(__file__).parents[3]
    config_path = (
        root
        / "examples/configs/experiments"
        / "vlm_mpo-nemotron-omni-nopack-legacy-parity.yaml"
    )
    register_omegaconf_resolvers()
    config = load_config(config_path)
    master_config = MasterConfig.model_validate(
        OmegaConf.to_container(config, resolve=True)
    )

    assert master_config.mpo.max_num_steps == 100
    assert master_config.mpo.seed == 42
    assert master_config.policy["train_global_batch_size"] == 256
    assert master_config.policy["train_micro_batch_size"] == 1
    assert master_config.policy["max_total_sequence_length"] == 32768
    packing = cast(dict[str, Any], master_config.policy["sequence_packing"])
    assert packing["enabled"] is False
    megatron_config = cast(MegatronConfig, master_config.policy["megatron_cfg"])
    assert megatron_config["tensor_model_parallel_size"] == 8
    assert megatron_config["expert_model_parallel_size"] == 32
    assert megatron_config["context_parallel_size"] == 1
    assert megatron_config["freeze_moe_router"] is True
    assert megatron_config["freeze_audio_encoder"] is True
    assert megatron_config["freeze_audio_projector"] is True
    assert megatron_config["attention_backend"] == "auto"
    assert megatron_config["moe_router_load_balancing_type"] == "none"
    assert megatron_config["optimizer"]["lr"] == 8.0e-7
    assert megatron_config["optimizer"]["optimizer_cpu_offload"] is True
    train_config = cast(dict[str, Any], master_config.data["train"])
    assert train_config["data_path"] == "REPLACE_WITH_MMPR_METADATA_PATH"
    assert train_config["split_validation_size"] == 2000
    assert train_config["legacy_validation_split"] is True
    assert master_config.data["num_workers"] == 0
    assert master_config.cluster["num_nodes"] == 4
    assert not master_config.logger["wandb_enabled"]


@pytest.mark.parametrize(
    ("filename", "packing_enabled", "sequence_length", "context_parallel_size"),
    [
        ("vlm_mpo-nemotron-super-omni-nopack-2k-parity.yaml", False, 2048, 1),
        ("vlm_mpo-nemotron-super-omni-packed-2k-parity.yaml", True, 2048, 1),
        (
            "vlm_mpo-nemotron-super-omni-packed-16k-cp2-parity.yaml",
            True,
            16384,
            2,
        ),
    ],
)
def test_nemotron_super_omni_mpo_parity_configs(
    filename: str,
    packing_enabled: bool,
    sequence_length: int,
    context_parallel_size: int,
):
    root = Path(__file__).parents[3]
    config_path = root / "examples/configs/experiments" / filename
    register_omegaconf_resolvers()
    config = load_config(config_path)
    master_config = MasterConfig.model_validate(
        OmegaConf.to_container(config, resolve=True)
    )

    assert master_config.mpo.max_num_steps == 100
    assert (
        master_config.policy["model_name"] == "REPLACE_WITH_NEMOTRON_SUPER_OMNI_MODEL"
    )
    assert master_config.policy["train_global_batch_size"] == 256
    assert master_config.policy["max_total_sequence_length"] == sequence_length
    assert master_config.data["max_input_seq_length"] == sequence_length
    assert master_config.cluster["num_nodes"] == 4
    assert master_config.cluster["gpus_per_node"] == 8
    expected_divisibility = 32 if context_parallel_size == 2 else 8
    assert (
        master_config.policy["make_sequence_length_divisible_by"]
        == expected_divisibility
    )

    megatron_config = cast(MegatronConfig, master_config.policy["megatron_cfg"])
    assert megatron_config["tensor_model_parallel_size"] == 8
    assert megatron_config["expert_model_parallel_size"] == 32
    assert megatron_config["context_parallel_size"] == context_parallel_size
    assert megatron_config["mtp_num_layers"] == 0
    assert megatron_config["use_fused_linear_logprobs"] is False

    packing = cast(SequencePackingConfig, master_config.policy["sequence_packing"])
    assert packing["enabled"] is packing_enabled
    if packing_enabled:
        assert packing["pair_grouping_key"] == "pair_index"
        assert packing["max_sequences_per_bin"] == 1
