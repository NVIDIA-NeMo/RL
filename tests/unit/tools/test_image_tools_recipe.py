# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from nemo_rl.utils.config import load_config


def test_standalone_hsg_recipe_geometry_and_protocol(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    for key in (
        "MODEL_CHECKPOINT",
        "VLLM_TOKENIZER",
        "TRAIN_MANIFEST",
        "EVAL_MANIFEST",
        "CHECKPOINT_DIR",
        "RUN_LOG_DIR",
        "WANDB_RUN_NAME",
    ):
        monkeypatch.setenv(key, "/fixture/" + key)
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    recipe = (
        root
        / "examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-image-tools-8n4g-megatron-tp8ep16cp2-async.v1.yaml"
    )
    cfg = load_config(recipe)
    generation = cfg.policy.generation.colocated.resources
    learner_gpus = (
        cfg.cluster.num_nodes - generation.num_nodes
    ) * cfg.cluster.gpus_per_node
    parallel = cfg.policy.megatron_cfg
    assert cfg.cluster.gpus_per_node == generation.gpus_per_node == 4
    assert learner_gpus == 16
    assert (
        learner_gpus
        % (parallel.tensor_model_parallel_size * parallel.context_parallel_size)
        == 0
    )
    assert learner_gpus % parallel.expert_model_parallel_size == 0
    assert cfg.policy.generation.vllm_cfg.tensor_parallel_size == 4
    assert (
        cfg.policy.train_global_batch_size
        == cfg.grpo.num_prompts_per_step * cfg.grpo.num_generations_per_prompt
        == 32
    )
    assert cfg.env.nemo_gym.num_samples_in_parallel == 32
    assert cfg.grpo.async_grpo.enabled
    assert not cfg.grpo.async_grpo.in_flight_weight_updates
    assert not cfg.grpo.reward_shaping.enabled
    assert not cfg.grpo.debug_payload_metrics
    assert cfg.loss_fn.token_level_loss
    assert cfg.policy.generation.max_new_tokens == 512
    assert cfg.policy.generation.vllm_kwargs.limit_mm_per_prompt.image == 21
    assert (
        cfg.policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template_content_format
        == "string"
    )
    assert cfg.env.nemo_gym.skip_venv_if_present
    assert cfg.env.nemo_gym.uv_venv_dir == "/opt/gym_venvs"
    assert list(cfg.env.nemo_gym.config_paths) == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "environments/image_tools_grpo/config.yaml",
    ]
    assert cfg.checkpointing.enabled
    assert cfg.checkpointing.ft_save_period == 20
    assert cfg.logger.wandb_enabled


def test_mixed_hsg_recipe_preserves_protocol_and_scales_generation(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    for key in (
        "MODEL_CHECKPOINT",
        "VLLM_TOKENIZER",
        "TRAIN_MANIFEST",
        "EVAL_MANIFEST",
        "CHECKPOINT_DIR",
        "RUN_LOG_DIR",
        "WANDB_RUN_NAME",
    ):
        monkeypatch.setenv(key, "/fixture/" + key)
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    cfg = load_config(
        root
        / "examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-visual-games-image-tools-40n4g-megatron-tp8ep16cp2-async.v1.yaml"
    )
    assert cfg.cluster.num_nodes == 40
    assert cfg.cluster.gpus_per_node == 4
    assert cfg.policy.generation.colocated.resources.num_nodes == 32
    assert cfg.policy.generation.colocated.resources.gpus_per_node == 4
    assert cfg.policy.megatron_cfg.tensor_model_parallel_size == 8
    assert cfg.policy.megatron_cfg.context_parallel_size == 2
    assert cfg.policy.megatron_cfg.expert_model_parallel_size == 16
    assert (
        cfg.policy.train_global_batch_size
        == cfg.grpo.num_prompts_per_step * cfg.grpo.num_generations_per_prompt
        == 128
    )
    assert cfg.env.nemo_gym.num_samples_in_parallel == 128
    assert cfg.policy.generation.max_new_tokens == 512
    assert cfg.policy.generation.vllm_kwargs.limit_mm_per_prompt.image == 32
    assert cfg.data.shuffle
    assert cfg.grpo.val_period == 10
    assert cfg.grpo.val_batch_size == 32
    assert cfg.checkpointing.ft_save_period == 20
    assert cfg.checkpointing.keep_top_k is None
    assert cfg.checkpointing.ft_keep_latest_k is None
    assert not cfg.grpo.async_grpo.in_flight_weight_updates
    assert not cfg.grpo.reward_shaping.enabled
    assert cfg.loss_fn.token_level_loss
    assert cfg.logger.wandb_enabled
    assert list(cfg.env.nemo_gym.config_paths) == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "environments/visual_games_image_tools/config.yaml",
    ]
