# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.models.generation.vllm.config import materialize_vllm_video_config
from nemo_rl.utils.config import load_config

pytestmark = pytest.mark.run_first

RECIPE_DIR = Path(__file__).parents[2] / "examples" / "configs" / "recipes" / "vlm"


def _load_recipe(name: str) -> dict:
    config = load_config(RECIPE_DIR / name)
    resolved = OmegaConf.to_container(config, resolve=False)
    assert isinstance(resolved, dict)
    return resolved


def test_video_grpo_recipes_preserve_unmasked_sync_and_async_contracts():
    async_recipe = _load_recipe(
        "vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml"
    )
    sync_recipe = _load_recipe(
        "vlm_grpo-nemotron-omni-30ba3b-2n8g-megatron-tp4ep4-gym-video.v1.yaml"
    )

    for recipe, async_enabled in ((async_recipe, True), (sync_recipe, False)):
        grpo = recipe["grpo"]
        policy = recipe["policy"]
        materialize_vllm_video_config(policy, recipe["data"])
        vllm_cfg = policy["generation"]["vllm_cfg"]

        assert grpo["max_num_steps"] == 1_000_000
        assert grpo["seq_logprob_error_threshold"] is None
        assert grpo["async_grpo"]["enabled"] is async_enabled
        assert grpo["reward_shaping"]["enabled"] is False
        assert grpo["reward_shaping"]["mode"] == "response_length"
        assert policy["is_vlm"] is True
        assert vllm_cfg["video"] == {
            "sampling_style": "nemotron_vl",
            "num_frames": 32,
            "temporal_patch_size": 2,
        }
        assert vllm_cfg["reset_encoder_cache_after_weight_update"] is False
        assert policy["generation"]["vllm_kwargs"]["media_io_kwargs"]["video"] == {
            "num_frames": 32
        }
        assert not {
            "NRL_VIDEO_BACKEND",
            "NRL_VIDEO_SAMPLING_STYLE",
            "NRL_VIDEO_TEMPORAL_PATCH_SIZE",
        } & set(vllm_cfg.get("env_vars", {}))


def test_video_length_reward_recipes_preserve_arushi_experiment_contract():
    async_recipe = _load_recipe(
        "vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp2ep16-async-gym-video-length-reward.v1.yaml"
    )
    sync_recipe = _load_recipe(
        "vlm_grpo-nemotron-omni-30ba3b-2n8g-megatron-tp4ep4-gym-video-length-reward.v1.yaml"
    )

    for recipe, async_enabled in ((async_recipe, True), (sync_recipe, False)):
        assert recipe["grpo"]["async_grpo"]["enabled"] is async_enabled
        reward_shaping = recipe["grpo"]["reward_shaping"]
        assert reward_shaping["enabled"] is True
        assert reward_shaping["mode"] == "reasoning_length"
        assert reward_shaping["reasoning_length"] == {
            "tau1": 1024,
            "tau2": 4096,
            "weight": 0.05,
            "composition": "correctness_gated",
            "reasoning_end_token_id": 13,
        }

    assert async_recipe["grpo"]["num_prompts_per_step"] == 128
    assert async_recipe["grpo"]["num_generations_per_prompt"] == 16
    assert async_recipe["grpo"]["max_num_steps"] == 1_000_000
    assert async_recipe["grpo"]["max_num_epochs"] == 1_000_000
    assert async_recipe["grpo"]["seq_logprob_error_threshold"] is None
    assert async_recipe["loss_fn"]["reference_policy_kl_penalty"] == 0.0
    assert async_recipe["loss_fn"]["use_importance_sampling_correction"] is True

    policy = async_recipe["policy"]
    assert policy["train_global_batch_size"] == 2048
    assert policy["max_total_sequence_length"] == 16384
    assert policy["sequence_packing"]["enabled"] is True
    assert policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert policy["megatron_cfg"]["expert_model_parallel_size"] == 16
    assert policy["megatron_cfg"]["activation_checkpointing"] is False
    assert policy["megatron_cfg"]["optimizer"]["lr"] == 3.0e-6
    assert policy["megatron_cfg"]["scheduler"]["lr_warmup_iters"] == 10

    generation = policy["generation"]
    assert generation["max_new_tokens"] == 16384
    assert generation["temperature"] == 1.0
    assert generation["top_p"] == 1.0
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 2
    assert generation["vllm_cfg"]["gpu_memory_utilization"] == 0.7
    assert generation["vllm_cfg"]["video"]["num_frames"] == 64
    assert generation["vllm_kwargs"]["max_num_seqs"] == 8
    assert generation["vllm_kwargs"]["max_num_batched_tokens"] == 32768
    assert async_recipe["data"]["shuffle"] is True
    assert async_recipe["data"]["default"]["num_frames"] == 64

    sync_policy = sync_recipe["policy"]
    assert sync_recipe["grpo"]["max_num_epochs"] == 1_000_000
    assert sync_recipe["grpo"]["val_period"] == 0
    assert sync_recipe["data"]["shuffle"] is True
    assert sync_policy["sequence_packing"]["enabled"] is True
    assert sync_policy["generation"]["vllm_cfg"]["video"]["num_frames"] == 64
    assert sync_policy["generation"]["vllm_kwargs"]["mm_processor_cache_gb"] == 4
    assert sync_policy["megatron_cfg"]["scheduler"]["lr_warmup_iters"] == 10


def test_video_recipe_materializes_one_sampling_contract_for_all_consumers():
    recipe = _load_recipe(
        "vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml"
    )

    materialize_vllm_video_config(recipe["policy"], recipe["data"])

    assert recipe["policy"]["tokenizer"]["video"]["num_frames"] == 32
    assert recipe["data"]["default"]["num_frames"] == 32
    assert recipe["data"]["default"]["video_sampling_style"] == "nemotron_vl"
    assert recipe["data"]["default"]["video_temporal_patch_size"] == 2
    assert (
        recipe["policy"]["generation"]["vllm_kwargs"]["limit_mm_per_prompt"]["video"][
            "num_frames"
        ]
        == 32
    )
    assert recipe["policy"]["generation"]["vllm_kwargs"]["media_io_kwargs"] == {
        "video": {"num_frames": 32}
    }
