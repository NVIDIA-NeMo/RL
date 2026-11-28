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

from pathlib import Path
from typing import Any

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
    repo_root = Path(__file__).resolve().parents[3]
    raw = load_config(
        str(
            repo_root / "examples/configs/grpo_math_1B_automodel_single_controller.yaml"
        )
    )
    return OmegaConf.to_container(raw, resolve=True)


@pytest.mark.parametrize("lookahead", [1, 2, 4])
def test_full_batch_lookahead_supported(config: dict[str, Any], lookahead: int) -> None:
    prompts = config["grpo"]["num_prompts_per_step"]
    config["async_rl"]["sampler"]["max_lookahead_versions"] = lookahead
    config["async_rl"]["max_inflight_prompts"] = (lookahead + 1) * prompts
    config["async_rl"]["max_buffered_rollouts"] = (lookahead + 1) * prompts
    config["loss_fn"]["use_importance_sampling_correction"] = True
    config["loss_fn"]["force_on_policy_ratio"] = False
    validate_single_controller_config(MasterConfig(**config))


@pytest.mark.parametrize(
    "path,value",
    [
        ("on_policy_distillation.enabled", True),
        ("grpo.use_dynamic_sampling", True),
        ("token_capture.enabled", True),
        ("async_rl.sampler.name", "ready_first"),
        ("async_rl.min_groups_for_streaming_train", 1),
        ("async_rl.rollout_failure.max_skipped_prompts", 1),
        ("async_rl.rollout_failure.max_consecutive_dropped_prompts", 1),
        ("policy.generation.backend", "megatron"),
        ("policy.generation.colocated.enabled", True),
        ("policy.generation.vllm_cfg.async_engine", False),
        ("policy.sequence_packing.enabled", True),
        ("policy.dynamic_batching.enabled", True),
        ("policy.automodel_cfg.tensor_parallel_size", 2),
        ("policy.automodel_cfg.context_parallel_size", 2),
        ("policy.automodel_cfg.dp_replicate_size", 2),
        ("policy.automodel_cfg.cpu_offload", True),
        ("policy.automodel_cfg.custom_parallel_plan", "my.plan"),
        ("policy.automodel_cfg.sequence_parallel", True),
        ("policy.automodel_cfg.lora_cfg.enabled", True),
        ("policy.is_vlm", True),
        ("policy.tokenizer.use_processor", True),
        ("policy.draft.enabled", True),
        ("policy.quant_cfg", "fp8"),
        ("policy.precision", "float16"),
    ],
)
def test_unsupported(config, path, value):
    raw = OmegaConf.create(config)
    OmegaConf.update(raw, path, value, merge=False)
    with pytest.raises(ValueError, match=f"Unsupported: .*{path}"):
        _validate_automodel_single_controller_config(
            MasterConfig(**OmegaConf.to_container(raw, resolve=True))
        )


def test_ppo_unsupported(config):
    config["ppo"] = config.pop("grpo")
    # PPO estimators are gae/raw_reward; the GRPO exemplar's estimator does not parse.
    config["ppo"].pop("adv_estimator")
    with pytest.raises(ValueError, match="Unsupported: ppo"):
        _validate_automodel_single_controller_config(MasterConfig(**config))


def test_disabled_opd_block_supported(config):
    config["on_policy_distillation"] = {"enabled": False}
    _validate_automodel_single_controller_config(MasterConfig(**config))


def test_megatron_unchanged(config):
    config["policy"]["automodel_cfg"]["enabled"] = False
    config["policy"]["sequence_packing"]["enabled"] = True
    _validate_automodel_single_controller_config(MasterConfig(**config))
