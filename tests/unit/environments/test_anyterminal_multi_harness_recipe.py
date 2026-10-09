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
SUPER_SYNC_8N_RECIPE = (
    REPO_ROOT
    / "examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_sync_8n_single_controller.yaml"
)
SUPER_SYNC_3N_DEBUG_RECIPE = (
    REPO_ROOT
    / "examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_sync_3n_debug_single_controller.yaml"
)
SUPER_SYNC_4N_DEBUG_RECIPE = (
    REPO_ROOT
    / "examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_sync_4n_debug_single_controller.yaml"
)
NANO_OMNI_SYNC_2N_DEBUG_RECIPE = (
    REPO_ROOT
    / "examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_nano_omni_sync_2n_debug_single_controller.yaml"
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
    assert config["grpo"]["max_num_epochs"] == 1_000_000
    assert config["grpo"]["max_num_steps"] == 1_000_000
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
    assert config["env"]["nemo_gym"]["anyterminal_opencode"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {
        "context_window": 15872,
        "max_input_tokens": 11776,
        "max_output_tokens": 4096,
    }
    assert config["env"]["nemo_gym"]["anyterminal_openclaw"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {
        "context_window": 15872,
        "max_output_tokens": 4096,
    }
    assert config["env"]["nemo_gym"]["anyterminal_hermes"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {
        "max_turns": 3,
        "context_window": 15872,
    }
    assert config["logger"]["wandb_enabled"] is True
    assert config["logger"]["wandb"]["entity"] == "adlr"
    assert config["logger"]["wandb"]["log_nemo_gym_full_result_tables"] is True


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
    assert config["grpo"]["max_num_epochs"] == 1_000_000
    assert config["grpo"]["max_num_steps"] == 1_000_000
    assert config["grpo"]["async_grpo"] is None
    assert config["policy"]["train_global_batch_size"] == 8
    assert config["policy"]["is_vlm"] is True
    assert config["policy"]["megatron_cfg"]["mtp_num_layers"] == 1
    assert (
        config["policy"]["generation"]["vllm_cfg"]["http_server_serving_chat_kwargs"][
            "tool_parser"
        ]
        == "qwen3_coder"
    )
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
    assert config["env"]["nemo_gym"]["anyterminal_opencode"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {
        "context_window": 15872,
        "max_input_tokens": 11776,
        "max_output_tokens": 4096,
    }
    assert config["env"]["nemo_gym"]["anyterminal_openclaw"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {
        "context_window": 15872,
        "max_output_tokens": 4096,
    }
    assert config["env"]["nemo_gym"]["anyterminal_hermes"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {"context_window": 15872}
    assert config["logger"]["wandb_enabled"] is True
    assert config["logger"]["wandb"]["entity"] == "adlr"
    assert config["logger"]["wandb"]["log_nemo_gym_full_result_tables"] is True


def test_super_omni_sync_8n_recipe_resolves_synchronous_topology():
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(SUPER_SYNC_8N_RECIPE), resolve=True)

    assert config["cluster"]["num_nodes"] == 8
    assert config["policy"]["generation"]["colocated"] == {
        "enabled": False,
        "resources": {"gpus_per_node": 8, "num_nodes": 4},
    }
    assert config["grpo"]["async_grpo"] is None
    assert config["async_rl"]["sampler"] == {
        "name": "in_order",
        "max_lookahead_versions": 0,
    }
    assert config["async_rl"]["recompute_kv_cache_after_weight_updates"] is True
    assert config["async_rl"]["min_groups_for_streaming_train"] == 4
    assert config["async_rl"]["max_inflight_prompts"] == 4
    assert config["async_rl"]["max_buffered_rollouts"] == 4
    assert config["env"]["nemo_gym"]["fan_out"] == {
        "anyterminal_multi_harness": [
            "anyterminal_opencode",
            "anyterminal_openclaw",
            "anyterminal_pi",
            "anyterminal_hermes",
        ]
    }


def test_super_omni_sync_debug_recipes_keep_native_context_and_minimum_training_shape():
    register_omegaconf_resolvers()

    for recipe, nodes, generation_nodes in (
        (SUPER_SYNC_3N_DEBUG_RECIPE, 3, 1),
        (SUPER_SYNC_4N_DEBUG_RECIPE, 4, 2),
    ):
        config = OmegaConf.to_container(load_config(recipe), resolve=True)

        assert config["cluster"]["num_nodes"] == nodes
        assert config["policy"]["generation"]["colocated"]["resources"] == {
            "gpus_per_node": 8,
            "num_nodes": generation_nodes,
        }
        assert config["policy"]["max_total_sequence_length"] == 16384
        assert config["policy"]["generation"]["max_new_tokens"] == 16384
        assert config["grpo"]["num_prompts_per_step"] == 4
        assert config["grpo"]["num_generations_per_prompt"] == 2
        assert config["policy"]["train_global_batch_size"] == 8


def test_nano_omni_sync_2n_debug_recipe_resolves_multi_harness_topology():
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(
        load_config(NANO_OMNI_SYNC_2N_DEBUG_RECIPE), resolve=True
    )

    assert config["policy"]["model_name"] == (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    )
    assert config["policy"]["max_total_sequence_length"] == 16384
    assert config["policy"]["generation"]["max_new_tokens"] == 8192
    assert config["cluster"]["num_nodes"] == 2
    assert config["policy"]["generation"]["colocated"] == {
        "enabled": False,
        "resources": {"gpus_per_node": 8, "num_nodes": 1},
    }
    assert config["policy"]["generation"]["vllm_cfg"][
        "reasoning_parser_plugin"
    ].endswith("nano_v3_reasoning_parser.py")
    assert (
        config["policy"]["generation"]["vllm_cfg"]["http_server_serving_chat_kwargs"][
            "reasoning_parser"
        ]
        == "nano_v3"
    )
    assert config["policy"]["generation"]["vllm_kwargs"]["kernel_config"] == {
        "enable_flashinfer_autotune": False,
    }
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
    assert config["grpo"]["max_num_epochs"] == 1_000_000
    assert config["grpo"]["max_num_steps"] == 1_000_000
    assert config["policy"]["train_global_batch_size"] == 8
    assert config["async_rl"]["sampler"]["max_lookahead_versions"] == 0
    assert config["data_plane"]["enabled"] is True
    assert config["token_capture"]["enabled"] is True
    assert config["env"]["nemo_gym"]["anyterminal_opencode"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {
        "context_window": 15872,
        "max_input_tokens": 7680,
        "max_output_tokens": 8192,
    }
    assert config["env"]["nemo_gym"]["anyterminal_openclaw"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {
        "context_window": 15872,
        "max_output_tokens": 4096,
    }
    assert config["env"]["nemo_gym"]["anyterminal_hermes"]["responses_api_agents"][
        "anyterminal_agent"
    ]["agent_kwargs"] == {"context_window": 15872}
    assert config["logger"]["wandb_enabled"] is True
    assert config["logger"]["wandb"]["entity"] == "adlr"
    assert config["logger"]["wandb"]["log_nemo_gym_full_result_tables"] is True
