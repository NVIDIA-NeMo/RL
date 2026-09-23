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
"""Megatron recipes preserve diffusion semantics and reject unsupported layouts."""

from pathlib import Path

import pytest
import torch
from just_grpo.algorithms.block_just_grpo import BlockJustGRPOSchedule
from just_grpo.config import ScheduleConfig, validate_config
from omegaconf import OmegaConf
from test_schedule import batch

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

RECIPES = Path(__file__).parents[2] / "configs/recipes"


def recipe(name="just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml"):
    register_omegaconf_resolvers()
    return load_config(RECIPES / name)


@pytest.mark.parametrize(
    "name,world,fraction,runtime",
    [
        ("just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml", 32, 1.0, "megatron"),
        (
            "just_grpo-sudoku6x6-4n8g-megatron-inference-fast-long.yaml",
            32,
            0.25,
            "megatron",
        ),
    ],
)
def test_megatron_recipes(name, world, fraction, runtime):
    cfg = recipe(name)
    validate_config(cfg)
    assert cfg.cluster.num_nodes * cfg.cluster.gpus_per_node == world
    assert cfg.policy.megatron_cfg.enabled
    assert cfg.policy.megatron_cfg.tensor_model_parallel_size == (
        1 if world == 32 or runtime == "megatron" else 2
    )
    assert cfg.just_grpo.distributed
    assert cfg.just_grpo.training_token_fraction == fraction
    assert cfg.just_grpo.runtime == runtime
    assert (
        cfg.just_grpo.validation_sampling.selection_policy
        == cfg.just_grpo.sampling.selection_policy
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("policy.megatron_cfg.pipeline_model_parallel_size", 2),
        ("policy.megatron_cfg.context_parallel_size", 2),
        ("policy.megatron_cfg.sequence_parallel", True),
        ("policy.megatron_cfg.recompute_granularity", "selective"),
        ("policy.megatron_cfg.tensor_model_parallel_size", 3),
        (
            "policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather",
            True,
        ),
        ("policy.megatron_cfg.optimizer.lr", 0.1),
        ("just_grpo.distributed", False),
    ],
)
def test_reject_unsupported_megatron_settings(key, value):
    cfg = recipe()
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError, match="Megatron"):
        validate_config(cfg)


def test_fixed_canvas_preserves_reveal_positions():
    data = batch()
    cfg = ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=1)
    dynamic = BlockJustGRPOSchedule(data, cfg, pad_token_id=0)
    fixed = BlockJustGRPOSchedule(data, cfg, pad_token_id=0, padded_width=32)
    for step in range(dynamic.num_steps):
        a = dynamic.generate_single_trajectory(step)
        b = fixed.generate_single_trajectory(step)
        width = a["input_ids"].shape[1]
        for key in ("input_ids", "masked_indices", "token_mask"):
            torch.testing.assert_close(a[key], b[key][:, :width])
        assert not b["token_mask"][:, width:].any()
        assert not b["masked_indices"][:, width:].any()
        values = torch.randn_like(b["input_ids"], dtype=torch.float32)
        torch.testing.assert_close(
            dynamic.scatter_logprobs(a, values[:, :width]),
            fixed.scatter_logprobs(b, values),
        )


@pytest.mark.parametrize(
    "path", sorted(RECIPES.glob("*.yaml")), ids=lambda path: path.name
)
def test_every_recipe_uses_megatron(path):
    config = recipe(path.name)
    validate_config(config)
    assert config.policy.megatron_cfg.enabled
    assert not config.policy.dtensor_cfg.enabled
    assert "reference_device" not in config.just_grpo
