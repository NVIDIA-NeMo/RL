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


def test_explicit_groups_separate_identical_audio_instructions():
    prompts = torch.ones(6, 4, dtype=torch.long)
    rewards = torch.tensor([0., 1., 2., 10., 10., 10.])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    baseline, std = calculate_baseline_and_std_per_prompt(
        prompts, rewards, torch.ones_like(rewards), full_group_std=True,
        prompt_group_ids=groups,
    )
    torch.testing.assert_close(baseline, torch.tensor([1.5, 1., .5, 10., 10., 10.]))
    torch.testing.assert_close(std, torch.tensor([1., 1., 1., 0., 0., 0.]))
    with pytest.raises(ValueError, match="one identity per reward"):
        calculate_baseline_and_std_per_prompt(prompts, rewards, torch.ones_like(rewards), prompt_group_ids=groups[:-1])


def test_group_ids_survive_repetition_filter_refill_and_slice():
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    offered = BatchedDataDict({"prompt_group_ids": torch.arange(4), "source_idx": torch.arange(10, 14)})
    repeated = offered.repeat_interleave(3)
    accepted = repeated.select_indices(torch.tensor([0, 1, 2, 6, 7, 8]))
    refill = BatchedDataDict({"prompt_group_ids": torch.arange(4, 6), "source_idx": torch.arange(14, 16)}).repeat_interleave(3)
    combined = BatchedDataDict.from_batches([accepted, refill]).slice(0, 9)
    assert combined["prompt_group_ids"].tolist() == [0, 0, 0, 2, 2, 2, 4, 4, 4]
    assert combined["source_idx"].tolist() == [10, 10, 10, 12, 12, 12, 14, 14, 14]
