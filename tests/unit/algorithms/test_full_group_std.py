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

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
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
    baseline, std, _ = calculate_baseline_and_std_per_prompt(
        groups, rewards, torch.ones_like(rewards), full_group_std=True
    )
    expected_baseline = torch.stack(
        [torch.cat((rewards[:i], rewards[i + 1 :])).mean() for i in range(len(values))]
    )
    torch.testing.assert_close(baseline, expected_baseline)
    torch.testing.assert_close(
        std,
        (rewards.std() if min(values) != max(values) else torch.tensor(0.0)).expand_as(
            std
        ),
    )


def test_full_group_std_masks_invalid_rewards_and_distinguishes_groups():
    rewards = torch.tensor([0.8, 0.9, -100.0, 0.8, 0.8])
    groups = torch.tensor([[0], [0], [0], [1], [1]])
    valid = torch.tensor([1, 1, 0, 1, 1])
    baseline, std, _ = calculate_baseline_and_std_per_prompt(
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
        AdvEstimatorConfig(
            full_group_std=True, normalize_rewards=True, use_leave_one_out_baseline=True
        ),
        ClippedPGLossConfig(),
    )
    advantage = estimator.compute_advantage(groups, rewards, mask)
    assert advantage[-1, 0].item() == pytest.approx(8.0, abs=0.001)
    torch.testing.assert_close(advantage[:, 0], advantage[:, 2])
    legacy = GRPOAdvantageEstimator(
        AdvEstimatorConfig(normalize_rewards=True, use_leave_one_out_baseline=True),
        ClippedPGLossConfig(),
    )
    assert legacy.compute_advantage(groups, rewards, mask)[
        -1, 0
    ].item() == pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("name", ["gdpo", "opd", "reinforce_plus_plus"])
def test_full_group_std_rejects_non_grpo_estimators(name):
    with pytest.raises(ValueError, match="only supported for name='grpo'"):
        AdvEstimatorConfig(name=name, full_group_std=True)
    AdvEstimatorConfig(name=name)


@pytest.mark.parametrize("valid", [[1] * 64, [0] * 64, [1] + [0] * 63])
def test_full_group_constant_or_undersized_advantages_are_exactly_zero(valid):
    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(
            full_group_std=True, normalize_rewards=True, use_leave_one_out_baseline=True
        ),
        ClippedPGLossConfig(),
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
    rewards = torch.tensor([0.0, 1.0, 2.0, 10.0, 10.0, 10.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    baseline, std, _ = calculate_baseline_and_std_per_prompt(
        groups.unsqueeze(-1),
        rewards,
        torch.ones_like(rewards),
        full_group_std=True,
    )
    torch.testing.assert_close(
        baseline, torch.tensor([1.5, 1.0, 0.5, 10.0, 10.0, 10.0])
    )
    torch.testing.assert_close(std, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_group_ids_survive_repetition_filter_refill_and_slice():
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    offered = BatchedDataDict(
        {"prompt_group_ids": torch.arange(4), "source_idx": torch.arange(10, 14)}
    )
    repeated = offered.repeat_interleave(3)
    accepted = repeated.select_indices(torch.tensor([0, 1, 2, 6, 7, 8]))
    refill = BatchedDataDict(
        {"prompt_group_ids": torch.arange(4, 6), "source_idx": torch.arange(14, 16)}
    ).repeat_interleave(3)
    combined = BatchedDataDict.from_batches([accepted, refill]).slice(0, 9)
    assert combined["prompt_group_ids"].tolist() == [0, 0, 0, 2, 2, 2, 4, 4, 4]
    assert combined["source_idx"].tolist() == [10, 10, 10, 12, 12, 12, 14, 14, 14]


def test_dynamic_sampling_keeps_first_twenty_nonconstant_groups_across_refill():
    from types import SimpleNamespace

    from nemo_rl.algorithms.grpo import dynamic_sampling
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.utils.timer import Timer

    config = SimpleNamespace(
        grpo=SimpleNamespace(
            num_prompts_per_step=20,
            num_generations_per_prompt=64,
            use_dynamic_sampling=True,
            dynamic_sampling_max_gen_batches=10,
            deduplicate_multimodal_data=False,
        )
    )

    def offer(first, count, nonconstant):
        identities = torch.arange(first, first + count).repeat_interleave(64)
        rewards = torch.ones(count * 64)
        for group in nonconstant:
            rewards[(group - first) * 64] = 2.0
        batch = BatchedDataDict(
            {
                "prompt_group_ids": identities,
                "idx": identities.clone(),
                "total_reward": rewards,
            }
        )
        mean, std, trivial = calculate_baseline_and_std_per_prompt(
            identities.unsqueeze(-1),
            rewards,
            torch.ones_like(rewards),
            full_group_std=True,
        )
        return batch, mean, std, trivial

    first, baseline, std, trivial = offer(0, 40, list(range(0, 30, 2)))
    accepted, complete, cache, _ = dynamic_sampling(
        first, std, baseline, 1, config, Timer(), is_trivial_prompt_distribution=trivial
    )
    assert not complete
    assert accepted["prompt_group_ids"].unique_consecutive().tolist() == list(
        range(0, 30, 2)
    )
    refill, baseline, std, trivial = offer(40, 10, list(range(40, 50)))
    accepted, complete, _, _ = dynamic_sampling(
        refill,
        std,
        baseline,
        2,
        config,
        Timer(),
        cache,
        is_trivial_prompt_distribution=trivial,
    )
    assert complete
    assert accepted["prompt_group_ids"].unique_consecutive().tolist() == list(
        range(0, 30, 2)
    ) + list(range(40, 45))
    assert accepted.size == 20 * 64
    assert torch.all(accepted["std"] > 0)
