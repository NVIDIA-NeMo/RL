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
"""Selection must reduce scored positions without changing their conditioning."""

import pytest
import torch
from omegaconf import OmegaConf
from test_schedule import TinyDiffusion, batch
from test_sudoku_and_config import load, PROJECT

from just_grpo.algorithms.block_just_grpo import (
    BlockJustGRPOSchedule,
    select_low_confidence_tokens,
)
from just_grpo.config import ScheduleConfig, validate_config
from just_grpo.diffusion.denoising_schedule import aggregate_diffusion_logprobs


def confidence_batch():
    data = batch()
    # Invalid positions deliberately contain NaN and must never be ranked.
    scores = torch.full_like(data["token_mask"], float("nan"), dtype=torch.float32)
    scores[0, 4:10] = torch.tensor([-0.1, -2, -0.4, -4, -0.2, -0.3])
    scores[1, 8:11] = torch.tensor([-1, -1, -0.2])
    data["generation_logprobs"] = scores.requires_grad_()
    return data


def make_schedule(data):
    return BlockJustGRPOSchedule(
        data,
        ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=1),
        pad_token_id=0,
    )


def test_low_confidence_selection_per_completion_with_rounding_and_ties():
    data = confidence_batch()
    selected = select_low_confidence_tokens(data, fraction=0.25)
    assert selected[0].nonzero().flatten().tolist() == [5, 7]
    assert selected[1].nonzero().flatten().tolist() == [8]
    assert not selected.requires_grad
    assert not (selected & ~data["token_mask"].bool()).any()
    assert select_low_confidence_tokens(data, fraction=0.01).sum(1).tolist() == [1, 1]
    assert torch.equal(
        select_low_confidence_tokens(data, fraction=1), data["token_mask"].bool()
    )


def test_excluded_samples_and_lengths_do_not_participate_in_ranking():
    data = confidence_batch()
    data["sample_mask"][1] = 0
    data["input_lengths"][0] = 6
    selected = select_low_confidence_tokens(data, fraction=0.25)
    assert selected[0].nonzero().flatten().tolist() == [5]
    assert not selected[1].any()
    data["sample_mask"][:] = 0
    assert not select_low_confidence_tokens(data, fraction=0.25).any()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_rollout_logprob_is_rejected(value):
    data = confidence_batch()
    with torch.no_grad():
        data["generation_logprobs"][0, 4] = value
    with pytest.raises(ValueError, match="finite rollout"):
        select_low_confidence_tokens(data, fraction=0.25)


def test_selection_preserves_every_denoising_canvas_and_selected_logprob():
    data = confidence_batch()
    full = make_schedule(data)
    selected = select_low_confidence_tokens(data, fraction=0.25)
    data["training_token_mask"] = selected
    fast = make_schedule(data)
    torch.manual_seed(4)
    model = TinyDiffusion()
    coverage = torch.zeros_like(selected, dtype=torch.int)
    for original, sparse in zip(full.return_schedule(2), fast.return_schedule(2)):
        assert torch.equal(original["masked_indices"], sparse["masked_indices"])
        assert torch.equal(original["input_ids"], sparse["input_ids"])
        active = sparse["token_mask"]
        assert torch.equal(active, original["token_mask"] & selected)
        torch.testing.assert_close(model(original)[active], model(sparse)[active])
        coverage += active
    assert torch.equal(coverage, selected.int())
    sparse_logprobs = aggregate_diffusion_logprobs(
        fast.iter_levels(),
        model,
        original_shape=selected.shape,
        device=torch.device("cpu"),
    )
    assert not sparse_logprobs[~selected].any()


def test_training_mask_must_not_select_prompt_or_padding():
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(data, fraction=0.25)
    data["training_token_mask"][0, 0] = True
    with pytest.raises(ValueError, match="subset"):
        make_schedule(data)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan")])
def test_invalid_fraction_config(fraction):
    with pytest.raises(ValueError):
        validate_config(
            OmegaConf.merge(
                load(), {"just_grpo": {"training_token_fraction": fraction}}
            )
        )


def test_fast_recipe_only_changes_training_selection():
    full = load(
        PROJECT
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml"
    )
    fast = load(
        PROJECT
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-fast-long.yaml"
    )
    assert validate_config(full).training_token_fraction == 1
    assert validate_config(fast).training_token_fraction == 0.25
    for key in ("grpo", "policy", "loss_fn", "data", "env", "cluster"):
        assert full[key] == fast[key]
    assert full.just_grpo.sampling == fast.just_grpo.sampling
    assert full.just_grpo.validation_sampling == fast.just_grpo.validation_sampling


def test_per_block_selection_and_partial_views_preserve_scores_and_gradients():
    data = confidence_batch()
    selected = select_low_confidence_tokens(data, fraction=0.25, block_size=4)
    # Unlike completion-wide ranking, the second response block also contributes.
    assert selected[0].nonzero().flatten().tolist() == [7, 9]
    assert selected[1].nonzero().flatten().tolist() == [8]
    data["training_token_mask"] = selected
    full = make_schedule(data)
    compact = BlockJustGRPOSchedule(
        data, full.config, pad_token_id=0, selected_positions_per_block=1
    )
    assert full.num_steps == 4 and compact.num_steps == 1
    view = compact.generate_single_trajectory(0)
    # Sample 0 blocks select offsets 3 and 1; sample 1 selects offset 0.
    assert view["masked_indices"][0, 4:8].tolist() == [False, False, False, True]
    assert view["masked_indices"][0, 8:12].tolist() == [False, True, True, True]
    assert view["masked_indices"][1, 8:12].all()
    torch.manual_seed(17)
    model = TinyDiffusion()
    dense_values = sum(model(t) * t["token_mask"] for t in full.return_schedule(2))
    compact_values = model(view) * view["token_mask"]
    torch.testing.assert_close(dense_values, compact_values)
    weights = torch.randn_like(dense_values)
    dense_grad = torch.autograd.grad(
        (dense_values * weights).sum(), tuple(model.parameters())
    )
    compact_grad = torch.autograd.grad(
        (compact_values * weights).sum(), tuple(model.parameters())
    )
    for a, b in zip(dense_grad, compact_grad):
        torch.testing.assert_close(a, b)
    assert torch.equal(view["token_mask"], selected)


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_native_streaming_token_count_matches_selected_targets(fraction):
    """Native train_microbatch counts each target once across denoising levels."""
    data = confidence_batch()
    data["sample_mask"][-1] = 0
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    schedule = BlockJustGRPOSchedule(
        data,
        make_schedule(data).config,
        pad_token_id=0,
        selected_positions_per_block=1 if fraction < 1 else None,
    )
    count = sum(
        (level["token_mask"][:, 1:] * level["sample_mask"][:, None]).sum()
        for level in schedule.iter_levels(data)
    )
    expected = (data["training_token_mask"] * data["sample_mask"][:, None]).sum()
    assert expected > 0
    torch.testing.assert_close(count, expected)
