# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import yaml


RECIPE_DIR = Path(__file__).parents[2] / "examples" / "nemo_gym"


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_recipe_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    defaults = config.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]

    merged = {}
    for default in defaults:
        merged = _merge_dicts(merged, _load_recipe_file(path.parent / default))
    return _merge_dicts(merged, config)


def _load_recipe(name: str) -> dict:
    return _load_recipe_file(RECIPE_DIR / name)


def test_video_grpo_recipes_keep_validated_raw_tmpe_configuration():
    async_recipe = _load_recipe("grpo_nemotron_omni_30ba3b_video_async.yaml")
    sync_recipe = _load_recipe("grpo_nemotron_omni_30ba3b_video_sync.yaml")

    for recipe, async_enabled in ((async_recipe, True), (sync_recipe, False)):
        grpo = recipe["grpo"]
        policy = recipe["policy"]
        vllm_cfg = recipe["policy"]["generation"]["vllm_cfg"]

        assert grpo["max_num_steps"] == -1
        assert grpo["seq_logprob_error_threshold"] is None
        assert grpo["async_grpo"]["enabled"] is async_enabled
        assert recipe.get("loss_fn", {}).get("force_on_policy_ratio") is not True
        assert policy["is_vlm"] is True
        assert policy["model_name"] != "/path/to/hf_checkpoint"
        assert policy["megatron_cfg"]["pipeline_model_parallel_size"] == 1
        assert recipe["env"]["should_use_nemo_gym"] is True
        assert vllm_cfg["logprobs_mode"] == "raw_logprobs"
        assert vllm_cfg["tensor_parallel_size"] == 4
        assert vllm_cfg["skip_tokenizer_init"] is False
        assert vllm_cfg["env_vars"]["NRL_VIDEO_BACKEND"] == "torchcodec"
        assert vllm_cfg["env_vars"]["NRL_VIDEO_SAMPLING_STYLE"] == "nemotron_vl"
        assert vllm_cfg["env_vars"]["NRL_VIDEO_TEMPORAL_PATCH_SIZE"] == "2"
        assert vllm_cfg["env_vars"]["VLLM_VIDEO_LOADER_BACKEND"] == "nemotron_vl"

        assert policy["train_global_batch_size"] == (
            grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
        )
        if async_enabled:
            # Keep the validated group size while allocating only two of the
            # sixteen nodes to the disaggregated policy. The scheduler's idle
            # reaper triggers at 25%; two idle nodes are 12.5%.
            assert grpo["num_prompts_per_step"] == 4
            assert policy["train_global_batch_size"] == 64
            assert policy["megatron_cfg"]["expert_model_parallel_size"] == 4
            assert policy["megatron_cfg"]["moe_shared_expert_overlap"] is False
            assert policy["megatron_cfg"]["optimizer"]["optimizer_cpu_offload"] is True
            generation_nodes = policy["generation"]["colocated"]["resources"][
                "num_nodes"
            ]
            assert generation_nodes == 14
            assert recipe["cluster"]["num_nodes"] - generation_nodes == 2
