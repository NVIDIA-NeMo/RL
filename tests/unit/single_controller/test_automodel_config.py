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
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@pytest.fixture
def config():
    register_omegaconf_resolvers()
    raw = load_config("examples/configs/grpo_math_1B.yaml")
    raw.grpo.async_grpo = None
    raw.grpo.val_period = 0
    raw.policy.dynamic_batching.enabled = False
    raw.policy.sequence_packing.enabled = False
    raw.policy.generation.colocated.enabled = False
    raw.policy.generation.vllm_cfg.async_engine = True
    raw.async_rl = {
        "sampler": {"name": "in_order", "max_lookahead_versions": 0},
        "min_groups_for_streaming_train": raw.grpo.num_prompts_per_step,
    }
    return OmegaConf.to_container(raw, resolve=True)


def test_supported(config):
    _validate_automodel_single_controller_config(MasterConfig(**config))


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
