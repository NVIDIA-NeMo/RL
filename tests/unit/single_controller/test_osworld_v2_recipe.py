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

import sys

import pytest
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from omegaconf import OmegaConf

from examples.nemo_gym.launch_osworld_v2_cc import (
    MOLT_CHECKPOINT_CONFIG_PATH,
    MOLT_CONFIG_PATH,
    RECIPE_CONFIG_PATHS,
    compose_and_validate_config,
    parse_args,
)


def test_osworld_v2_recipe_composes_to_initial_cc_envelope(monkeypatch):
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nemotron-omni")
    monkeypatch.setenv(
        "NANO_OMNI_CHAT_TEMPLATE",
        "/models/nemotron-omni/chat_template.jinja",
    )
    monkeypatch.setenv("OSWORLD_GRPO_TRAIN_DATA", "/data/osworld-train.jsonl")

    config = compose_and_validate_config([])

    assert config.grpo is not None
    assert config.ppo is None
    assert config.grpo.num_prompts_per_step == 1
    assert config.grpo.num_generations_per_prompt == 2
    assert config.grpo.max_rollout_turns == 1
    assert config.grpo.max_num_steps == 1
    assert config.policy["train_global_batch_size"] == 2
    assert config.policy["train_global_batch_size"] == (
        config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt
    )

    assert config.data_plane["enabled"] is True
    assert config.data_plane["impl"] == "transfer_queue"
    assert config.data_plane["backend"] == "simple"
    assert config.token_capture.enabled is True
    assert config.token_capture.context_compaction is True
    assert config.token_capture.defer_routed_experts_to_policy is False

    assert config.async_rl.sampler.name == "in_order"
    assert config.async_rl.sampler.max_lookahead_versions == 0
    assert config.async_rl.sampler.warmup_lookahead_versions is None
    assert config.async_rl.min_groups_for_streaming_train == 1
    assert config.async_rl.max_inflight_prompts == 1
    assert config.async_rl.max_buffered_rollouts == 1
    failure = config.async_rl.rollout_failure
    assert failure.max_infra_attempts_per_prompt == 1
    assert failure.max_data_attempts_per_prompt == 1
    assert failure.max_skipped_prompts == 0
    assert failure.max_consecutive_dropped_prompts == 0
    assert failure.min_step_batch_fraction == 1
    assert failure.on_dropped_prompt == "shrink"
    assert failure.nemo_gym.max_row_attempts == 1

    assert config.grpo.async_grpo is None
    assert config.grpo.adv_estimator.name == "grpo"
    assert config.grpo.advantage_clip_low is None
    assert config.grpo.advantage_clip_high is None
    assert config.grpo.reward_clip_low is None
    assert config.grpo.reward_clip_high is None
    assert config.grpo.seq_logprob_error_threshold is None
    assert config.grpo.use_dynamic_sampling is False
    assert config.grpo.reward_shaping.enabled is False
    assert config.grpo.reward_scaling.enabled is False
    assert config.loss_fn.token_level_loss is True
    assert config.loss_fn.sequence_level_importance_ratios is False
    assert config.loss_fn.truncated_importance_sampling_type is None
    assert config.loss_fn.disable_ppo_ratio is False
    assert config.loss_fn.use_kl_in_reward is False
    assert config.loss_fn.positive_example_nll_weight == 0
    assert config.on_policy_distillation is None

    assert config.checkpointing["enabled"] is False
    assert "pretrained_checkpoint" not in config.checkpointing
    assert config.rollout_checkpointing.snapshot_attempt_interval_s is None
    assert config.grpo.val_period == 0
    assert config.grpo.val_at_start is False
    assert config.grpo.val_at_end is False
    assert config.data["validation"] is None
    assert config.data["use_multiple_dataloader"] is False
    assert config.data["default"]["repeat"] == 1


def test_osworld_v2_recipe_uses_expected_gym_context_management(monkeypatch):
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nemotron-omni")
    monkeypatch.setenv(
        "NANO_OMNI_CHAT_TEMPLATE",
        "/models/nemotron-omni/chat_template.jinja",
    )
    monkeypatch.setenv("OSWORLD_GRPO_TRAIN_DATA", "/data/osworld-train.jsonl")

    config = compose_and_validate_config([])
    gym = config.env["nemo_gym"]

    assert gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model.yaml",
        "responses_api_agents/nemotron_osworld/configs/nemotron_osworld_cc.yaml",
        "resources_servers/osworld/configs/osworld.yaml",
        "resources_servers/osworld/configs/opensandbox_osworld.yaml",
    ]
    assert all("vllm_model_for_training" not in path for path in gym["config_paths"])
    assert (
        gym["policy_model"]["responses_api_models"]["vllm_model"][
            "return_token_id_information"
        ]
        is False
    )

    agent = gym["nemotron_osworld"]["responses_api_agents"]["nemotron_osworld"]
    context_history = agent["context_history"]
    assert agent["max_steps"] == 3
    assert agent["max_parallel_rollouts"] == 2
    assert context_history["enabled"] is True
    assert context_history["schedule"] == {
        "type": "turn_chunked_recency",
        "actions_per_chunk": 2,
    }
    assert context_history["policy"]["config"]["images"]["keep_last_groups"] == 1
    assert context_history["max_response_retries"] == 0

    generation = config.policy["generation"]
    assert generation["backend"] == "vllm"
    assert generation["colocated"]["enabled"] is False
    assert generation["vllm_cfg"]["logprobs_mode"] == "processed_logprobs"
    assert generation["vllm_kwargs"]["limit_mm_per_prompt"] == {"image": 9}


def test_osworld_v2_recipe_resolves_paired_gym_configs(monkeypatch):
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nemotron-omni")
    monkeypatch.setenv(
        "NANO_OMNI_CHAT_TEMPLATE",
        "/models/nemotron-omni/chat_template.jinja",
    )
    monkeypatch.setenv("OSWORLD_GRPO_TRAIN_DATA", "/data/osworld-train.jsonl")

    config = compose_and_validate_config([])
    gym_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                config.env["nemo_gym"]
                | {
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "test-key",
                    "policy_model_name": config.policy["model_name"],
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    agent = gym_config.nemotron_osworld.responses_api_agents.nemotron_osworld
    assert agent.entrypoint == "cc_app.py"
    assert agent.token_id_capture is True
    assert agent.context_history.enabled is True
    assert (
        gym_config.policy_model.responses_api_models.vllm_model.return_token_id_information
        is False
    )


def test_osworld_v2_recipe_selects_agentenv_fork_oversampling(monkeypatch):
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nemotron-omni")
    monkeypatch.setenv(
        "NANO_OMNI_CHAT_TEMPLATE",
        "/models/nemotron-omni/chat_template.jinja",
    )
    monkeypatch.setenv("OSWORLD_GRPO_TRAIN_DATA", "/data/osworld-train.jsonl")
    monkeypatch.setenv(
        "OSWORLD_SANDBOX_CONFIG",
        "nemo_gym/sandbox/providers/agentenv/configs/agentenv.yaml",
    )
    monkeypatch.setenv("AGENTENV_ENDPOINT", "https://agentenv.example")
    monkeypatch.setenv("AGENTENV_API_KEY", "test-key")
    monkeypatch.setenv("AGENTENV_TLS_CA", "/private/agentenv-ca.pem")

    config = compose_and_validate_config([])
    gym_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                config.env["nemo_gym"]
                | {
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "test-key",
                    "policy_model_name": config.policy["model_name"],
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    assert config.env["nemo_gym"]["config_paths"][-1].endswith("agentenv.yaml")
    assert (
        gym_config.sandbox.agentenv.create.template
        == "osworld-slim-pixel-parity-20261001"
    )
    assert (
        gym_config.osworld_resources_server.resources_servers.osworld.fork_oversampling
        is True
    )


def test_osworld_v2_molt_b8k8_recipe_composes_and_validates(monkeypatch):
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nemotron-omni")
    monkeypatch.setenv(
        "NANO_OMNI_CHAT_TEMPLATE",
        "/models/nemotron-omni/chat_template.jinja",
    )
    monkeypatch.setenv("OSWORLD_GRPO_TRAIN_DATA", "/data/osworld-train.jsonl")
    monkeypatch.delenv("OSWORLD_MAX_PARALLEL_ROLLOUTS", raising=False)

    config = compose_and_validate_config([], config_path=MOLT_CONFIG_PATH)

    assert config.grpo.num_prompts_per_step == 8
    assert config.grpo.num_generations_per_prompt == 8
    assert config.grpo.max_num_epochs == 100000
    assert config.grpo.max_num_steps == 300
    assert config.policy["train_global_batch_size"] == 64
    assert config.grpo.adv_estimator.name == "reinforce_baseline"
    assert config.grpo.baseline_population == "all_owners"
    assert config.grpo.reward_clip_low == 0.0
    assert config.grpo.reward_clip_high == 1.0

    assert config.async_rl.sampler.name == "windowed"
    assert config.async_rl.sampler.max_staleness_versions == 1
    assert config.async_rl.sampler.sample_freshest_first is False
    assert config.async_rl.min_groups_for_streaming_train == 8
    assert config.async_rl.stage_full_advantage_window is True
    assert config.async_rl.staged_policy_probe_interval_s == 600
    assert config.async_rl.max_inflight_prompts == 8
    assert config.async_rl.max_buffered_rollouts == 16

    loss = config.loss_fn
    assert loss.use_importance_sampling_correction is True
    assert loss.truncated_importance_sampling_type == "seq-mask-tis"
    assert loss.truncated_importance_sampling_ratio_min == 0.99
    assert loss.truncated_importance_sampling_ratio == 1.01
    assert loss.sequence_level_importance_ratios is False
    assert loss.token_level_loss is True
    assert loss.force_on_policy_ratio is True

    assert config.policy["max_total_sequence_length"] == 49152
    assert config.policy["offload_optimizer_for_logprob"] is True
    assert config.policy["generation"]["max_new_tokens"] == 16384
    assert config.policy["router_replay"]["enabled"] is True
    assert config.policy["sequence_packing"]["enabled"] is True
    assert config.policy["sequence_packing"]["train_mb_tokens"] == 49152
    assert config.policy["generation"]["vllm_cfg"]["enable_prefix_caching"] is True
    assert config.policy["generation"]["colocated"]["resources"] == {
        "gpus_per_node": 8,
        "num_nodes": 3,
    }
    assert config.cluster["num_nodes"] == 4
    assert config.cluster["gpus_per_node"] == 8
    megatron = config.policy["megatron_cfg"]
    assert megatron["freeze_vision_model"] is True
    assert megatron["freeze_vision_projection"] is True
    assert megatron["freeze_sound_encoder"] is True
    assert megatron["freeze_sound_projection"] is True
    assert megatron["optimizer"]["lr"] == 5e-6
    assert megatron["optimizer"]["min_lr"] == 5e-6
    assert megatron["scheduler"]["lr_warmup_iters"] == 9
    assert megatron["scheduler"]["lr_decay_iters"] == 300
    assert megatron["freeze_moe_router"] is True
    assert megatron["moe_router_bias_update_rate"] == 0.0

    agent = config.env["nemo_gym"]["nemotron_osworld"]["responses_api_agents"][
        "nemotron_osworld"
    ]
    assert agent["max_steps"] == 150
    assert agent["max_parallel_rollouts"] == 8
    assert agent["context_history"]["schedule"]["actions_per_chunk"] == 100
    assert agent["context_history"]["schedule"]["actions_per_chunk"] != (
        config.async_rl.max_inflight_prompts
    )
    assert (
        agent["context_history"]["policy"]["config"]["images"]["keep_last_groups"] == 2
    )
    assert agent["context_history"]["guards"]["max_active_images"] == 10
    assert agent["context_history"]["guards"]["reserved_generation_tokens"] == 11152
    assert config.checkpointing["enabled"] is False
    assert config.rollout_checkpointing.snapshot_attempt_interval_s is None

    gym_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                config.env["nemo_gym"]
                | {
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "test-key",
                    "policy_model_name": config.policy["model_name"],
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    resolved_agent = gym_config.nemotron_osworld.responses_api_agents.nemotron_osworld
    assert resolved_agent.max_steps == 150
    assert resolved_agent.max_parallel_rollouts == 8
    assert resolved_agent.context_history.policy.config.images.keep_last_groups == 2
    assert resolved_agent.context_history.guards.max_active_images == 10

    monkeypatch.setenv("OSWORLD_MAX_STEPS", "20")
    monkeypatch.setenv("OSWORLD_MAX_PARALLEL_ROLLOUTS", "4")
    smoke_config = compose_and_validate_config([], config_path=MOLT_CONFIG_PATH)
    smoke_agent = smoke_config.env["nemo_gym"]["nemotron_osworld"][
        "responses_api_agents"
    ]["nemotron_osworld"]
    assert smoke_agent["max_steps"] == 20
    assert smoke_agent["max_parallel_rollouts"] == 4


def test_launcher_selects_molt_recipe_and_preserves_hydra_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_osworld_v2_cc.py",
            "--molt-b8k8",
            "--validate-only",
            "logger.log_dir=/logs/molt",
        ],
    )

    args, overrides = parse_args()

    assert args.molt_b8k8 is True
    assert args.validate_only is True
    assert overrides == ["logger.log_dir=/logs/molt"]


def test_osworld_v2_molt_checkpoint_recipe_composes_and_validates(monkeypatch):
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nemotron-omni")
    monkeypatch.setenv(
        "NANO_OMNI_CHAT_TEMPLATE",
        "/models/nemotron-omni/chat_template.jinja",
    )
    monkeypatch.setenv("OSWORLD_GRPO_TRAIN_DATA", "/data/osworld-train.jsonl")
    monkeypatch.setenv("OSWORLD_CHECKPOINT_DIR", "/checkpoints/osworld-v2")

    config = compose_and_validate_config([], config_path=MOLT_CHECKPOINT_CONFIG_PATH)

    assert config.grpo.max_num_epochs == 100000
    assert config.grpo.max_num_steps == 300
    assert config.checkpointing["enabled"] is True
    assert config.checkpointing["checkpoint_dir"] == "/checkpoints/osworld-v2"
    assert config.checkpointing["save_period"] == 1
    assert config.checkpointing["save_optimizer"] is True
    assert config.checkpointing["save_data_plane"] is True
    assert config.checkpointing["quiesce_rollouts_for_checkpoint"] is True
    assert config.policy["megatron_cfg"]["checkpoint"]["async_save"] is False
    assert config.async_rl.min_groups_for_streaming_train == 8
    assert config.async_rl.stage_full_advantage_window is True
    assert config.async_rl.staged_policy_probe_interval_s == 600
    assert config.async_rl.max_buffered_rollouts == 8
    assert config.rollout_checkpointing.snapshot_attempt_interval_s is None
    assert config.rollout_checkpointing.restore_mode == "trainer_checkpoint"


def test_launcher_selects_molt_checkpoint_recipe(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_osworld_v2_cc.py",
            "--molt-b8k8-checkpoint",
            "--validate-only",
        ],
    )

    args, overrides = parse_args()

    assert args.molt_b8k8 is False
    assert args.molt_b8k8_checkpoint is True
    assert args.validate_only is True
    assert overrides == []


@pytest.mark.parametrize(
    ("recipe", "prompts", "generations", "advantage"),
    [
        ("flash-b8n8-reinforce", 8, 8, "reinforce_baseline"),
        ("flash-b8n8-dr-grpo", 8, 8, "dr_grpo"),
        ("flash-b17n16-dr-grpo", 17, 16, "dr_grpo"),
        ("flash-b17n16-reinforce", 17, 16, "reinforce_baseline"),
    ],
)
def test_flash_recipes_compose_and_validate(
    monkeypatch, recipe, prompts, generations, advantage
):
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nemotron-omni")
    monkeypatch.setenv(
        "NANO_OMNI_CHAT_TEMPLATE",
        "/models/nemotron-omni/chat_template.jinja",
    )
    monkeypatch.setenv("OSWORLD_GRPO_TRAIN_DATA", "/data/rlvr-band/train.jsonl")
    monkeypatch.setenv("OSWORLD_CHECKPOINT_DIR", "/checkpoints/osworld-v2")

    config = compose_and_validate_config([], config_path=RECIPE_CONFIG_PATHS[recipe])

    assert config.grpo.num_prompts_per_step == prompts
    assert config.grpo.num_generations_per_prompt == generations
    assert config.grpo.adv_estimator.name == advantage
    assert config.policy["train_global_batch_size"] == prompts * generations
    assert config.policy["router_replay"]["enabled"] is True
    assert config.policy["megatron_cfg"]["optimizer"]["lr"] == 5e-6
    assert config.async_rl.stage_full_advantage_window is True
    assert config.loss_fn.force_on_policy_ratio is True
    assert config.loss_fn.use_importance_sampling_correction is True
    assert config.loss_fn.truncated_importance_sampling_type == "seq-mask-tis"
    assert config.loss_fn.is_correction_gating == "binary_kl"
    assert config.loss_fn.truncated_importance_sampling_ratio_min == 0.0
    assert config.loss_fn.truncated_importance_sampling_ratio == 0.01
    assert config.loss_fn.loss_agg_mode == "prompt-mean-token-mean"
    if prompts == 17:
        assert config.policy["train_global_batch_size"] > 16 * 16
        assert config.async_rl.min_groups_for_streaming_train == 17
        assert config.async_rl.max_buffered_rollouts == 17


def test_launcher_selects_explicit_flash_recipe(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_osworld_v2_cc.py",
            "--recipe",
            "flash-b8n8-dr-grpo",
            "--validate-only",
            "logger.log_dir=/logs/flash",
        ],
    )

    args, overrides = parse_args()

    assert args.recipe == "flash-b8n8-dr-grpo"
    assert args.validate_only is True
    assert overrides == ["logger.log_dir=/logs/flash"]
