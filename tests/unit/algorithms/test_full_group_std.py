import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt


@pytest.mark.parametrize(
    "values",
    [[0.8] * 64, [0.8] * 63 + [0.9], [-2.0, 0.0, 0.25, 1.0], [0.8] * 63 + [0.80001]],
)
def test_full_group_std_preserves_leave_one_out_mean(values):
    rewards = torch.tensor(values)
    groups = torch.zeros((len(values), 1), dtype=torch.long)
    baseline, std = calculate_baseline_and_std_per_prompt(
        groups, rewards, torch.ones_like(rewards), full_group_std=True
    )
    expected_baseline = torch.stack(
        [torch.cat((rewards[:i], rewards[i + 1 :])).mean() for i in range(len(values))]
    )
    torch.testing.assert_close(baseline, expected_baseline)
    torch.testing.assert_close(std, (rewards.std() if min(values) != max(values) else torch.tensor(0.)).expand_as(std))


def test_full_group_std_masks_invalid_rewards_and_distinguishes_groups():
    rewards = torch.tensor([0.8, 0.9, -100.0, 0.8, 0.8])
    groups = torch.tensor([[0], [0], [0], [1], [1]])
    valid = torch.tensor([1, 1, 0, 1, 1])
    baseline, std = calculate_baseline_and_std_per_prompt(
        groups, rewards, valid, full_group_std=True
    )
    torch.testing.assert_close(baseline[:2], torch.tensor([0.9, 0.8]))
    torch.testing.assert_close(std[:3], rewards[:2].std().expand(3))
    assert torch.equal(std[3:], torch.zeros(2))


def test_grpo_full_group_outlier_advantage_and_legacy_default():
    rewards = torch.tensor([0.8] * 63 + [0.9])
    groups = torch.zeros((64, 1), dtype=torch.long)
    mask = torch.ones((64, 3))
    estimator = GRPOAdvantageEstimator(
        {"full_group_std": True, "normalize_rewards": True, "use_leave_one_out_baseline": True}, ClippedPGLossConfig()
    )
    advantage = estimator.compute_advantage(groups, rewards, mask)
    assert advantage[-1, 0].item() == pytest.approx(8.0, abs=0.001)
    torch.testing.assert_close(advantage[:, 0], advantage[:, 2])
    legacy = GRPOAdvantageEstimator({"normalize_rewards": True, "use_leave_one_out_baseline": True}, ClippedPGLossConfig())
    assert legacy.compute_advantage(groups, rewards, mask)[
        -1, 0
    ].item() == pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("valid", [[1] * 64, [0] * 64, [1] + [0] * 63])
def test_full_group_constant_or_undersized_advantages_are_exactly_zero(valid):
    estimator = GRPOAdvantageEstimator(
        {"full_group_std": True, "normalize_rewards": True, "use_leave_one_out_baseline": True}, ClippedPGLossConfig()
    )
    advantage = estimator.compute_advantage(
        torch.zeros((64, 1), dtype=torch.long),
        torch.full((64,), 0.8),
        torch.ones((64, 2)),
        valid_mask=torch.tensor(valid),
    )
    assert torch.equal(advantage, torch.zeros_like(advantage))
