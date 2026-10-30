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

"""Early guards for supported AutoModel single-controller configurations."""

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    _validate_automodel_single_controller_config,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@pytest.fixture
def config():
    register_omegaconf_resolvers()
    raw = load_config("examples/configs/grpo_math_1B_automodel_single_controller.yaml")
    return OmegaConf.to_container(raw, resolve=True)


def test_supported(config):
    assert config["policy"]["automodel_cfg"]["enabled"]
    assert not config["policy"]["megatron_cfg"]["enabled"]
    validate_single_controller_config(MasterConfig(**config))


@pytest.mark.parametrize("prompts,generations", [(2, 4), (8, 2)])
def test_base_batch_settings_follow_prompt_overrides(prompts, generations):
    register_omegaconf_resolvers()
    raw = load_config("examples/configs/grpo_math_1B_automodel_single_controller.yaml")
    raw.grpo.num_prompts_per_step = prompts
    raw.grpo.num_generations_per_prompt = generations
    config = MasterConfig(**OmegaConf.to_container(raw, resolve=True))
    validate_single_controller_config(config)
    assert config.policy["train_global_batch_size"] == prompts * generations
    assert config.async_rl.min_groups_for_streaming_train == prompts
    assert config.async_rl.max_inflight_prompts == prompts
    assert config.async_rl.max_buffered_rollouts == prompts


def test_full_batch_regression_recipe():
    register_omegaconf_resolvers()
    raw = load_config(
        "examples/configs/recipes/llm/"
        "grpo-qwen2.5-1.5b-1n2g-automodel-single-controller-full-batch.yaml"
    )
    config = MasterConfig(**OmegaConf.to_container(raw, resolve=True))
    validate_single_controller_config(config)
    assert config.policy["train_global_batch_size"] == (
        config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("async_rl.sampler.max_lookahead_versions", 1),
        ("async_rl.sampler.name", "ready_first"),
        ("async_rl.min_groups_for_streaming_train", 1),
        ("policy.dynamic_batching.enabled", True),
        ("policy.sequence_packing.enabled", True),
        ("policy.automodel_cfg.context_parallel_size", 2),
        ("policy.automodel_cfg.tensor_parallel_size", 2),
        ("policy.automodel_cfg.dp_replicate_size", 2),
        ("policy.automodel_cfg.cpu_offload", True),
        ("policy.automodel_cfg.lora_cfg.enabled", True),
        ("policy.tokenizer.use_processor", True),
        ("policy.quant_cfg", "fp8"),
        ("policy.generation.colocated.enabled", True),
        ("policy.generation.vllm_cfg.async_engine", False),
        ("async_rl.rollout_failure.max_skipped_prompts", 1),
    ],
)
def test_unsupported(config, path, value):
    raw = OmegaConf.create(config)
    OmegaConf.update(raw, path, value, merge=False)
    with pytest.raises(ValueError, match="AutoModel SC"):
        _validate_automodel_single_controller_config(
            MasterConfig(**OmegaConf.to_container(raw, resolve=True))
        )


def test_megatron_unchanged(config):
    config["policy"]["automodel_cfg"]["enabled"] = False
    config["async_rl"]["sampler"]["max_lookahead_versions"] = 2
    _validate_automodel_single_controller_config(MasterConfig(**config))
