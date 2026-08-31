# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.models.generation.vllm.config import materialize_vllm_video_config
from nemo_rl.utils.config import load_config

RECIPE_DIR = Path(__file__).parents[2] / "examples" / "configs" / "recipes" / "vlm"


def _load_recipe(name: str) -> dict:
    config = load_config(RECIPE_DIR / name)
    resolved = OmegaConf.to_container(config, resolve=False)
    assert isinstance(resolved, dict)
    return resolved


def _assert_original_reward_contract(recipe: dict) -> None:
    grpo = recipe["grpo"]
    assert grpo["seq_logprob_error_threshold"] == 2
    assert grpo["reward_shaping"]["enabled"] is False
    assert grpo["reward_scaling"]["enabled"] is False
    assert grpo["length_penalty"]["profile_band"]["enabled"] is False
    assert recipe["env"]["should_mask_flagged_samples"] is True
    assert "resources_servers/sav_tracks/configs/sav_tracks.yaml" in recipe["env"][
        "nemo_gym"
    ]["config_paths"]


def test_nano_sync_and_async_combined_data_topologies() -> None:
    sync_recipe = _load_recipe(
        "vlm_grpo-nemotron-nano-vl-v2-svg-b200-2n8g-megatron-"
        "tp4ep4-sync-gym-video-sav-caprl.v1.yaml"
    )
    async_recipe = _load_recipe(
        "vlm_grpo-nemotron-nano-vl-v2-svg-b200-16n8g-megatron-"
        "tp4ep4-async-gym-video-sav-caprl.v1.yaml"
    )

    _assert_original_reward_contract(sync_recipe)
    _assert_original_reward_contract(async_recipe)
    assert sync_recipe["cluster"] == {"num_nodes": 2, "gpus_per_node": 8}
    assert sync_recipe["grpo"]["async_grpo"]["enabled"] is False
    assert sync_recipe["policy"]["generation"]["colocated"]["enabled"] is True

    assert async_recipe["cluster"] == {"num_nodes": 16, "gpus_per_node": 8}
    assert async_recipe["grpo"]["async_grpo"] == {
        "enabled": True,
        "max_trajectory_age_steps": 1,
        "in_flight_weight_updates": True,
        "recompute_kv_cache_after_weight_updates": False,
    }
    assert async_recipe["policy"]["train_global_batch_size"] == 64
    assert async_recipe["policy"]["generation"]["colocated"] == {
        "enabled": False,
        "resources": {"num_nodes": 14, "gpus_per_node": 8},
    }
    assert async_recipe["checkpointing"]["save_replay_buffer"] is True


def test_super_async_combined_data_contract() -> None:
    recipe = _load_recipe(
        "vlm_grpo-nemotron-super-omni-120ba12b-svg-b200-32n8g-megatron-"
        "tp8ep16cp2-async-gym-video-sav-caprl.v1.yaml"
    )
    materialize_vllm_video_config(recipe["policy"], recipe["data"])

    _assert_original_reward_contract(recipe)
    assert recipe["cluster"] == {"num_nodes": 32, "gpus_per_node": 8}
    assert recipe["grpo"]["num_prompts_per_step"] == 128
    assert recipe["grpo"]["num_generations_per_prompt"] == 16
    assert recipe["policy"]["train_global_batch_size"] == 2048
    assert recipe["policy"]["megatron_cfg"]["context_parallel_size"] == 2
    assert recipe["policy"]["generation"]["colocated"] == {
        "enabled": False,
        "resources": {"num_nodes": 16, "gpus_per_node": 8},
    }
    assert recipe["policy"]["generation"]["vllm_cfg"]["video"] == {
        "sampling_style": "nemotron_vl",
        "num_frames": 64,
        "temporal_patch_size": 2,
    }
    assert recipe["policy"]["tokenizer"]["video"]["num_frames"] == 64
    assert recipe["data"]["default"]["num_frames"] == 64
    assert recipe["checkpointing"]["checkpoint_must_save_by"] == "00:03:10:00"
