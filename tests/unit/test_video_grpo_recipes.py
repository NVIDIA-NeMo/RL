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
