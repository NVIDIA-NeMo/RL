# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE = (
    REPO_ROOT
    / "examples/nemo_gym/grpo_anyterminal_multi_harness_qwen3_0_6b_single_controller.yaml"
)
SUPER_RECIPE = (
    REPO_ROOT
    / "examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_single_controller.yaml"
)


def test_anyterminal_multi_harness_recipe_resolves_async_training_contract():
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(RECIPE), resolve=True)

    assert config["env"]["should_use_nemo_gym"] is True
    assert config["env"]["nemo_gym"]["fan_out"] == {
        "anyterminal_multi_harness": [
            "anyterminal_opencode",
            "anyterminal_openclaw",
            "anyterminal_pi",
            "anyterminal_hermes",
        ]
    }
    assert config["grpo"]["num_prompts_per_step"] == 4
    assert config["grpo"]["num_generations_per_prompt"] == 2
    assert config["policy"]["train_global_batch_size"] == 8
    assert config["data"]["train"]["dataset_name"] == "NemoGymDataset"
    assert set(config["data"]["train"]) == {"dataset_name", "data_path"}
    assert config["grpo"]["async_grpo"] is None
    assert config["async_rl"]["sampler"]["max_lookahead_versions"] == 1
    assert config["data_plane"]["enabled"] is True
    assert config["policy"]["dtensor_cfg"]["enabled"] is False
    assert config["policy"]["megatron_cfg"]["enabled"] is True
    assert config["policy"]["megatron_cfg"]["tensor_model_parallel_size"] == 1
    assert config["policy"]["megatron_cfg"]["pipeline_model_parallel_size"] == 1
    assert config["token_capture"] == {
        "enabled": True,
        "staging_partition": "rollout_staging",
        "min_valid_fraction_per_group": 1.0,
    }
    assert config["env"]["nemo_gym"]["policy_model"]["responses_api_models"][
        "vllm_model"
    ]["sampling_overrides"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    assert config["policy"]["generation"]["top_k"] is None
    assert config["policy"]["generation"]["vllm_cfg"]["logprobs_mode"] == "raw_logprobs"
    assert config["policy"]["max_total_sequence_length"] == 16384
    assert config["policy"]["generation"]["vllm_cfg"]["max_model_len"] == 16384
    assert config["policy"]["generation"]["vllm_cfg"]["expose_http_server"] is True
    assert config["policy"]["generation"]["colocated"]["enabled"] is False


def test_super_omni_anyterminal_recipe_resolves_training_topology():
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(SUPER_RECIPE), resolve=True)

    assert config["env"]["nemo_gym"]["fan_out"] == {
        "anyterminal_multi_harness": [
            "anyterminal_opencode",
            "anyterminal_openclaw",
            "anyterminal_pi",
            "anyterminal_hermes",
        ]
    }
    assert config["grpo"]["num_prompts_per_step"] == 4
    assert config["grpo"]["num_generations_per_prompt"] == 2
    assert config["grpo"]["max_num_steps"] == 1_000_000
    assert config["grpo"]["async_grpo"] is None
    assert config["policy"]["train_global_batch_size"] == 8
    assert config["policy"]["is_vlm"] is True
    assert config["policy"]["megatron_cfg"]["mtp_num_layers"] == 1
    assert config["policy"]["generation"]["vllm_cfg"][
        "http_server_serving_chat_kwargs"
    ]["tool_parser"] == "qwen3_coder"
    assert config["policy"]["generation"]["colocated"] == {
        "enabled": False,
        "resources": {"gpus_per_node": 8, "num_nodes": 8},
    }
    assert config["cluster"] == {
        "gpus_per_node": 8,
        "num_nodes": 16,
        "master_port_range_low": 1400,
        "master_port_range_high": 1999,
    }
    assert config["token_capture"]["enabled"] is True
    assert config["data_plane"]["enabled"] is True
    assert config["checkpointing"]["enabled"] is False
    assert config["logger"]["wandb"]["log_nemo_gym_full_result_tables"] is False
