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
"""Context-cost reward shaping for multi-turn rollouts."""

import math

import pytest
import torch

from nemo_rl.algorithms.reward_functions import (
    ContextCostShapingConfig,
    RewardShapingConfig,
    apply_context_cost_shaping,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _batch(context_tokens: list[int], rewards: list[float]) -> BatchedDataDict:
    """One message log per sample; context is split over a prompt and two turns."""
    message_logs = []
    for total in context_tokens:
        prompt = total // 2
        turn = total - prompt
        message_logs.append(
            [
                {"role": "user", "token_ids": torch.zeros(prompt, dtype=torch.long)},
                {"role": "assistant", "token_ids": torch.zeros(turn, dtype=torch.long)},
                {"role": "user", "content": "tool result without token_ids"},
            ]
        )
    return BatchedDataDict(
        {
            "message_log": message_logs,
            "total_reward": torch.tensor(rewards, dtype=torch.float32),
        }
    )


def _factor(ctx: int, ctx_min: int, failure_rate: float, beta=0.5, ref=1000) -> float:
    return failure_rate + (1.0 - failure_rate) * math.exp(
        -beta * max(0, ctx - ctx_min) / ref
    )


def test_disabled_returns_batch_untouched():
    batch = _batch([100, 200], [1.0, 1.0])
    before = batch["total_reward"].clone()
    out = apply_context_cost_shaping(
        batch, ContextCostShapingConfig(enabled=False), num_generations=2
    )
    assert out is batch
    assert torch.equal(out["total_reward"], before)
    assert "unshaped_total_reward" not in out


def test_correct_rollouts_decay_with_context_above_group_minimum():
    cfg = ContextCostShapingConfig(enabled=True, beta=0.5, ref_tokens=1000)
    # One group of 4: three correct at 1000 / 2000 / 4000 tokens, one wrong.
    batch = _batch([1000, 2000, 4000, 500], [1.0, 1.0, 1.0, 0.0])
    out = apply_context_cost_shaping(batch, cfg, num_generations=4)

    d = 0.25  # one of four failed
    expected = torch.tensor(
        [_factor(1000, 1000, d), _factor(2000, 1000, d), _factor(4000, 1000, d), 0.0]
    )
    assert torch.allclose(out["total_reward"], expected, atol=1e-6)
    # The cheapest correct rollout is untouched; costlier ones decay toward d.
    assert out["total_reward"][0] == 1.0
    assert out["total_reward"][1] > out["total_reward"][2] > d
    # Wrong rollouts keep their reward and the raw reward is preserved.
    assert out["total_reward"][3] == 0.0
    assert torch.equal(out["unshaped_total_reward"], torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_groups_are_contiguous_blocks_and_shaped_independently():
    cfg = ContextCostShapingConfig(enabled=True, beta=0.5, ref_tokens=1000)
    # Group A: all correct (d = 0) -> full efficiency pressure.
    # Group B: none correct -> untouched.
    batch = _batch([1000, 3000, 700, 900], [1.0, 1.0, 0.0, 0.0])
    out = apply_context_cost_shaping(batch, cfg, num_generations=2)
    assert out["total_reward"][0] == 1.0
    assert math.isclose(
        out["total_reward"][1].item(), math.exp(-0.5 * 2000 / 1000), rel_tol=1e-6
    )
    assert torch.equal(out["total_reward"][2:], torch.tensor([0.0, 0.0]))


def test_hard_prompts_are_barely_shaped():
    """With most of the group failing, the surviving correct rollout keeps almost all of its reward."""
    cfg = ContextCostShapingConfig(enabled=True, beta=0.5, ref_tokens=1000)
    batch = _batch([1000, 9000, 800, 800, 800, 800, 800, 800], [1.0, 1.0] + [0.0] * 6)
    out = apply_context_cost_shaping(batch, cfg, num_generations=8)
    d = 0.75
    assert out["total_reward"][0] == 1.0
    assert out["total_reward"][1].item() >= d
    assert math.isclose(
        out["total_reward"][1].item(), _factor(9000, 1000, d), rel_tol=1e-6
    )


def test_threshold_and_non_binary_rewards():
    cfg = ContextCostShapingConfig(
        enabled=True, beta=1.0, ref_tokens=1000, correct_reward_threshold=0.5
    )
    batch = _batch([1000, 2000, 1000, 2000], [0.5, 0.8, 0.4, 0.2])
    out = apply_context_cost_shaping(batch, cfg, num_generations=4)
    d = 0.5
    # Rewards at/above 0.5 are scaled multiplicatively; the others are untouched.
    assert out["total_reward"][0].item() == pytest.approx(
        0.5 * _factor(1000, 1000, d, beta=1.0)
    )
    assert out["total_reward"][1].item() == pytest.approx(
        0.8 * _factor(2000, 1000, d, beta=1.0)
    )
    assert out["total_reward"][2].item() == pytest.approx(0.4)
    assert out["total_reward"][3].item() == pytest.approx(0.2)


def test_existing_unshaped_reward_is_not_overwritten():
    cfg = ContextCostShapingConfig(enabled=True)
    batch = _batch([100, 200], [1.0, 1.0])
    batch["unshaped_total_reward"] = torch.tensor([7.0, 7.0])
    out = apply_context_cost_shaping(batch, cfg, num_generations=2)
    assert torch.equal(out["unshaped_total_reward"], torch.tensor([7.0, 7.0]))


def test_validation_errors():
    batch = _batch([100, 200, 300], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="multiple of num_generations"):
        apply_context_cost_shaping(
            batch, ContextCostShapingConfig(enabled=True), num_generations=2
        )
    with pytest.raises(ValueError, match="ref_tokens"):
        apply_context_cost_shaping(
            batch,
            ContextCostShapingConfig(enabled=True, ref_tokens=0),
            num_generations=3,
        )


def test_reward_shaping_config_nests_context_cost_with_defaults():
    cfg = RewardShapingConfig()
    assert cfg.context_cost.enabled is False
    assert cfg.context_cost.beta == 0.5
    assert cfg.context_cost.ref_tokens == 32768
    nested = RewardShapingConfig(context_cost={"enabled": True, "beta": 0.25})
    assert nested.context_cost.enabled and nested.context_cost.beta == 0.25


def test_correctness_uses_the_raw_reward_when_dapo_shaping_ran_first():
    """A length penalty applied earlier must not move a rollout out of the correct set."""
    cfg = ContextCostShapingConfig(enabled=True, beta=0.5, ref_tokens=1000)
    batch = _batch([1000, 3000], [1.0, -1.125])  # DAPO already penalised the long one
    batch["unshaped_total_reward"] = torch.tensor([1.0, 1.0])
    out = apply_context_cost_shaping(batch, cfg, num_generations=2)
    # Both count as correct (raw reward 1.0); d = 0, so the long one decays from
    # its current (penalised) reward.
    assert out["total_reward"][0].item() == pytest.approx(1.0)
    assert out["total_reward"][1].item() == pytest.approx(
        -1.125 * math.exp(-0.5 * 2000 / 1000)
    )
    assert torch.equal(out["unshaped_total_reward"], torch.tensor([1.0, 1.0]))


def test_correct_rollouts_are_floored_at_the_best_incorrect_reward():
    cfg = ContextCostShapingConfig(
        enabled=True, beta=1.0, ref_tokens=100, correct_reward_threshold=0.5
    )
    batch = _batch([1000, 9000, 1000, 1000], [0.9, 0.55, 0.49, 0.49])
    out = apply_context_cost_shaping(batch, cfg, num_generations=4)
    # Without the floor the 0.55 rollout would decay far below the 0.49 ones.
    assert out["total_reward"][1].item() == pytest.approx(0.49)
    assert out["total_reward"][0].item() == pytest.approx(0.9)


def test_missing_message_log_raises_instead_of_silently_doing_nothing():
    cfg = ContextCostShapingConfig(enabled=True)
    batch = BatchedDataDict({"total_reward": torch.tensor([1.0, 1.0])})
    with pytest.raises(ValueError, match="message_log"):
        apply_context_cost_shaping(batch, cfg, num_generations=2)
