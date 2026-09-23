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
import copy
import sys
from pathlib import Path

import pytest
from just_grpo.config import validate_config
from just_grpo.environments.sudoku import make_examples
from just_grpo.environments.sudoku6x6_generator import _count_solutions
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml"
RECIPE = (
    PROJECT
    / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-fast-long.yaml"
)


def answer(grid):
    return (
        "<answer>\n"
        + "\n".join(" ".join(map(str, row)) for row in grid)
        + "\n</answer>"
    )


def test_unique_puzzles_and_blank_only_reward():
    example = make_examples(1, seed=42)[0]
    assert _count_solutions(copy.deepcopy(example.puzzle)) == 1
    assert example.reward(answer(example.solution)) == 1
    assert example.reward(answer(example.solution) + "<|im_end|>") == 1
    assert example.reward(answer(example.solution).replace("\n", "\n\n")) == 1
    assert example.reward(answer(example.puzzle)) == 0
    changed = copy.deepcopy(example.solution)
    blank = next(
        (r, c) for r in range(6) for c in range(6) if example.puzzle[r][c] == 0
    )
    changed[blank[0]][blank[1]] = changed[blank[0]][blank[1]] % 6 + 1
    n = sum(v == 0 for row in example.puzzle for v in row)
    assert example.reward(answer(changed)) == pytest.approx(1 - 1 / n)
    assert example.reward(answer(example.solution) + "extra") == 0
    assert example.reward(answer(example.solution) * 2) == 0
    assert example.reward("<answer>1 2 3</answer>") == 0


def load(path=BASE):
    register_omegaconf_resolvers()
    return load_config(path)


def load_reference_vllm(path=BASE, *, asynchronous=False):
    """Exercise the optional backend without publishing additional recipes."""
    config = load(path)
    config.just_grpo.runtime = "reference_vllm"
    config.policy.generation.backend = "vllm"
    config.policy.generation.refit_transport = None
    worker = "ReferenceVllmAsyncWorker" if asynchronous else "ReferenceVllmWorker"
    config.policy.generation.worker_extension_cls_fqn = (
        f"just_grpo.generation.reference_vllm.{worker}"
    )
    config.policy.generation.vllm_kwargs.diffusion_config = {
        "canvas_length": "${just_grpo.sampling.block_size}",
        "max_denoising_steps": "${just_grpo.sampling.max_steps}",
        "temperature": "${just_grpo.sampling.temperature}",
        "selection_policy": "${just_grpo.sampling.selection_policy}",
        "confidence_threshold": "${just_grpo.sampling.threshold}",
        "return_entropy": "${just_grpo.sampling.returns_entropy}",
        "return_reveal_steps": "${just_grpo.sampling.returns_reveal_steps}",
    }
    config.policy.generation.vllm_cfg.async_engine = asynchronous
    config.grpo.async_grpo.enabled = asynchronous
    if asynchronous:
        config.policy.generation.colocated.enabled = False
        config.policy.generation.colocated.resources.num_nodes = 2
        config.policy.generation.colocated.resources.gpus_per_node = 8
    return config


def test_validation_inherits_training_decoder(tmp_path):
    child = tmp_path / "validation.yaml"
    child.write_text(
        f"defaults: {BASE}\npolicy:\n  generation:\n    temperature: 0.2\n"
        "just_grpo:\n  schedule:\n    block_size: 32\n"
        "  sampling:\n    max_steps: 32\n    threshold: 0.8\n"
    )
    config = validate_config(load(child))
    assert config.sampling.temperature == 0.2
    assert config.validation_sampling.temperature == 0.2
    training = config.sampling.model_dump(exclude={"temperature"})
    validation = config.validation_sampling.model_dump(exclude={"temperature"})
    assert training == validation
    assert validation["selection_policy"] == "leftmost"
    assert validation["block_size"] == validation["max_steps"] == 32
    assert validation["threshold"] == 0.8


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"just_grpo": {"runtime": "upstream_vllm"}}, "Upstream generation"),
        ({"just_grpo": {"runtime": "upstream_sglang"}}, "Upstream generation"),
        (
            {"just_grpo": {"sampling": {"selection_policy": "confidence_threshold"}}},
            "leftmost sampling",
        ),
        ({"just_grpo": {"sampling": {"max_steps": 4}}}, "Generation steps"),
        ({"just_grpo": {"sampling": {"returns_entropy": True}}}, "entropy/reveal"),
        (
            {"just_grpo": {"schedule": {"reveal_tokens_per_step": 3}}},
            "divisible by reveal width",
        ),
        ({"policy": {"dtensor_cfg": {"enabled": True}}}, "DTensor"),
        ({"policy": {"megatron_cfg": {"enabled": False}}}, "Megatron training backend"),
        ({"checkpointing": {"enabled": True}}, "Resumable checkpointing"),
        ({"loss_fn": {"use_kl_in_reward": True}}, "reward-side KL"),
        ({"loss_fn": {"force_on_policy_ratio": True}}, "recomputed prev_logprobs"),
        ({"just_grpo": {"old_logprobs": "generation"}}, "Extra inputs"),
        ({"policy": {"train_global_batch_size": 1}}, "one global optimizer update"),
    ],
)
def test_fail_closed_on_unsupported_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_config(OmegaConf.merge(load(), kwargs))


def test_missing_diffusion_values_do_not_fall_back_to_python():
    raw = load()
    del raw.just_grpo.schedule.mask_token_id
    del raw.just_grpo.generation_python
    with pytest.raises(ValidationError) as error:
        validate_config(raw)
    assert {tuple(item["loc"]) for item in error.value.errors()} == {
        ("generation_python",),
        ("schedule", "mask_token_id"),
    }


def test_submitted_recipe_inherits_upstream_math_grpo():
    parent = load()
    child = load(RECIPE)
    validate_config(child)
    assert parent.grpo.num_prompts_per_step == 128
    assert child.grpo.num_prompts_per_step == 128
    assert child.policy.train_global_batch_size == 1024
    assert child.loss_fn.ratio_clip_min == 0.2
    assert child.policy.optimizer.kwargs.betas == [0.9, 0.999]
    assert child.grpo.adv_estimator.normalize_rewards is False
    assert "num_prompts" not in child and "learning_rate" not in child
    assert "defaults" not in child


def test_standard_overrides_update_interpolated_settings():
    config = OmegaConf.merge(
        load(),
        {
            "grpo": {"normalize_rewards": False},
            "policy": {
                "model_name": "test/model",
                "optimizer": {"kwargs": {"lr": 2e-6}},
            },
            "loss_fn": {"ratio_clip_min": 0.1},
        },
    )
    validate_config(config)
    assert config.policy.tokenizer.name == "test/model"
    assert config.grpo.adv_estimator.normalize_rewards is False
    assert config.policy.optimizer.kwargs.lr == 2e-6
    assert config.loss_fn.ratio_clip_min == 0.1


def test_entry_point_loads_inheritance_and_cli_overrides(monkeypatch, tmp_path):
    import run_just_grpo

    captured = []
    monkeypatch.setattr("just_grpo.train.run", captured.append)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_just_grpo.py",
            "--config",
            str(RECIPE),
            "--model",
            "/test/checkpoint",
            "--output-dir",
            "/test/results",
            "policy.optimizer.kwargs.lr=2e-6",
        ],
    )
    run_just_grpo.main()
    config = captured[0]
    validate_config(config)
    assert (
        config.policy.model_name == config.policy.tokenizer.name == "/test/checkpoint"
    )
    assert config.logger.log_dir == "/test/results"
    assert config.just_grpo.generation_python is None
    assert config.grpo.num_prompts_per_step == 128
    assert config.policy.optimizer.kwargs.lr == 2e-6
    assert (
        config.just_grpo.sampling.block_size
        == config.just_grpo.schedule.block_size
        == 16
    )


def test_long_recipe_enables_wandb_and_periodic_validation():
    config = load(
        PROJECT
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml"
    )
    validate_config(config)
    assert config.logger.wandb_enabled
    assert config.grpo.val_period == 10
    assert config.grpo.max_num_steps == 100
    assert config.policy.train_global_batch_size == 1024


@pytest.mark.parametrize("period", [0, -2])
def test_invalid_validation_period(period):
    config = load()
    config.grpo.val_period = period
    with pytest.raises(ValueError, match="val_period"):
        validate_config(config)


def test_reference_template_hyperparameters():
    config = load(
        PROJECT
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml"
    )
    validate_config(config)
    assert config.policy.optimizer.kwargs.lr == 3e-7
    assert config.policy.optimizer.kwargs.weight_decay == 0.01
    assert config.policy.megatron_cfg.scheduler.lr_warmup_iters == 13
    assert config.policy.megatron_cfg.scheduler.lr_warmup_init == pytest.approx(3e-8)
    assert not config.grpo.adv_estimator.normalize_rewards
    assert config.loss_fn.reference_policy_kl_penalty == 0.01
    assert config.loss_fn.use_importance_sampling_correction
    assert config.loss_fn.truncated_importance_sampling_type == "tis"
    assert config.loss_fn.truncated_importance_sampling_ratio == 2
    assert config.policy.generation.max_new_tokens == 512
    assert config.policy.generation.stop_token_ids == [11]
    assert config.data.train.size == 50000
    assert config.data.train.seed == 1
    assert config.data.validation.size == 256
    assert config.data.validation.seed == 2
    assert config.data.validation.repeat == 4
    assert "validation_modes" not in config.just_grpo
    assert config.just_grpo.validation_sampling == config.just_grpo.sampling


def test_lazy_dataset_matches_reference_generation_and_repeat():
    from just_grpo.environments.sudoku import SudokuDataset, user_prompt

    expected = make_examples(6, seed=1)
    dataset = SudokuDataset(size=6, seed=1, repeat=4)
    assert len(dataset) == 24
    assert [dataset[i] for i in range(24)] == expected * 4
    prompt = user_prompt(dataset[0], style="sudoku_answer_tag")
    assert prompt.startswith("Solve this 6x6 Sudoku puzzle:\n")
    assert prompt.endswith(
        "Put the completed grid inside <answer> </answer> tags, as rows of space-separated digits."
    )
