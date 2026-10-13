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

"""ArgMaxRL advantage estimator: per-prompt best@k weights over non-negative rewards."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    ArgMaxRLAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import GRPOConfig, _create_advantage_estimator
from nemo_rl.algorithms.loss import ClippedPGLossConfig


def _estimator(
    minus_baseline: bool = False, use_leave_one_out_baseline: bool = False
) -> ArgMaxRLAdvantageEstimator:
    config = AdvEstimatorConfig.model_construct(
        minus_baseline=minus_baseline,
        use_leave_one_out_baseline=use_leave_one_out_baseline,
    )
    return ArgMaxRLAdvantageEstimator(config, ClippedPGLossConfig())


# ── raw weights ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rewards, expected",
    [
        ([1.0, 0.5, 0.0], [0.75, 0.25, 0.0]),
        ([1.0, 1.0, 0.0], [0.5, 0.5, 0.0]),
        ([1.0, 1.0, 1.0], [1 / 3, 1 / 3, 1 / 3]),
        ([0.0, 0.0], [0.0, 0.0]),
        ([2.5], [2.5]),
        ([], []),
    ],
)
def test_weights_known_values(rewards, expected):
    torch.testing.assert_close(
        ArgMaxRLAdvantageEstimator.compute_weights(torch.tensor(rewards)),
        torch.tensor(expected),
    )


def test_weights_reduce_to_maxrl_on_binary_rewards():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(
        ArgMaxRLAdvantageEstimator.compute_weights(rewards), rewards / rewards.sum()
    )


# ── estimator ────────────────────────────────────────────────────────────────


def test_estimator_groups_by_prompt_identity_and_expands_to_mask_shape():
    prompt_ids = torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2]])
    rewards = torch.tensor([1.0, 2.0, 0.5, 0.0, 0.0])  # groups: rows 0,2,4 and 1,3
    mask = torch.ones(5, 4)

    adv = _estimator().compute_advantage(prompt_ids, rewards, mask)

    assert adv.shape == mask.shape
    torch.testing.assert_close(adv[:, 0], torch.tensor([0.75, 2.0, 0.25, 0.0, 0.0]))
    torch.testing.assert_close(adv, adv[:, :1].expand_as(adv))


def test_estimator_skips_rows_without_mask_or_valid_mask_without_reading_rewards():
    prompt_ids = torch.zeros(4, 3, dtype=torch.long)
    rewards = torch.tensor([1.0, float("nan"), 0.5, -7.0])
    mask = torch.ones(4, 5)
    mask[1] = 0.0  # fully masked row: never trained, never inspected
    valid_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])  # token-capture placeholder row

    adv = _estimator().compute_advantage(
        prompt_ids, rewards, mask, valid_mask=valid_mask
    )

    torch.testing.assert_close(adv[:, 0], torch.tensor([0.75, 0.0, 0.25, 0.0]))


def test_estimator_returns_zeros_when_nothing_participates():
    prompt_ids = torch.zeros(2, 1, dtype=torch.long)
    rewards = torch.tensor([float("nan"), -1.0])
    mask = torch.zeros(2, 3)

    adv = _estimator().compute_advantage(prompt_ids, rewards, mask)

    torch.testing.assert_close(adv, torch.zeros(2, 3))


def test_estimator_empty_batch():
    adv = _estimator().compute_advantage(
        torch.zeros(0, 1, dtype=torch.long), torch.zeros(0), torch.zeros(0, 3)
    )
    assert adv.shape == (0, 3)


@pytest.mark.parametrize(
    "minus_baseline, leave_one_out, expected",
    [
        (False, False, [0.75, 0.25, 0.0]),
        (True, False, [5 / 12, -1 / 12, -1 / 3]),
        (True, True, [5 / 8, -1 / 8, -1 / 2]),
    ],
)
def test_estimator_baselines(minus_baseline, leave_one_out, expected):
    """Raw weights [3/4, 1/4, 0]; group mean 1/3; leave-one-out means [1/8, 3/8, 1/2]."""
    prompt_ids = torch.zeros(3, 1, dtype=torch.long)
    adv = _estimator(minus_baseline, leave_one_out).compute_advantage(
        prompt_ids, torch.tensor([1.0, 0.5, 0.0]), torch.ones(3, 2)
    )
    torch.testing.assert_close(adv[:, 0], torch.tensor(expected))


@pytest.mark.parametrize(
    "minus_baseline, leave_one_out, expected",
    [(False, False, 2.0), (True, False, 0.0), (True, True, 2.0)],
)
def test_estimator_singleton_group(minus_baseline, leave_one_out, expected):
    adv = _estimator(minus_baseline, leave_one_out).compute_advantage(
        torch.zeros(1, 1, dtype=torch.long), torch.tensor([2.0]), torch.ones(1, 2)
    )
    torch.testing.assert_close(adv[0, 0], torch.tensor(expected))


@pytest.mark.parametrize("bad_reward", [-0.5, float("nan"), float("inf")])
def test_estimator_rejects_invalid_participating_rewards(bad_reward):
    with pytest.raises(ValueError, match="reward_scaling"):
        _estimator().compute_advantage(
            torch.zeros(2, 1, dtype=torch.long),
            torch.tensor([1.0, bad_reward]),
            torch.ones(2, 2),
        )


def test_estimator_rejects_kl_in_reward():
    with pytest.raises(ValueError, match="use_kl_in_reward"):
        ArgMaxRLAdvantageEstimator(
            AdvEstimatorConfig(name="argmaxrl"),
            ClippedPGLossConfig(use_kl_in_reward=True),
        )


# ── factory ──────────────────────────────────────────────────────────────────


def test_factory_builds_argmaxrl():
    master_config = SimpleNamespace(
        grpo=GRPOConfig(adv_estimator=AdvEstimatorConfig(name="argmaxrl")),
        loss_fn=ClippedPGLossConfig(),
    )
    assert isinstance(
        _create_advantage_estimator(master_config), ArgMaxRLAdvantageEstimator
    )
