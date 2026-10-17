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

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.loss import ClippedPGLossConfig


@pytest.mark.parametrize(
    ("use_leave_one_out_baseline", "normalize_rewards", "expected"),
    [
        pytest.param(
            False,
            False,
            [0.6666667, -0.3333333, -0.3333333],
            id="group-mean",
        ),
        pytest.param(
            False,
            True,
            [1.1546985, -0.5773493, -0.5773493],
            id="group-mean-normalized",
        ),
        pytest.param(True, False, [1.0, -0.5, -0.5], id="leave-one-out"),
        pytest.param(
            True,
            True,
            [1.0, -0.7071058, -0.7071058],
            id="leave-one-out-normalized",
        ),
    ],
)
def test_grpo_stats_exclude_samples_removed_by_sample_mask(
    use_leave_one_out_baseline: bool,
    normalize_rewards: bool,
    expected: list[float],
) -> None:
    """Masked responses must not affect per-prompt baselines or normalization."""
    estimator = GRPOAdvantageEstimator(
        {
            "use_leave_one_out_baseline": use_leave_one_out_baseline,
            "normalize_rewards": normalize_rewards,
        },
        ClippedPGLossConfig(),
    )
    prompt_ids = torch.tensor([[7], [7], [7], [7]], dtype=torch.long)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    sample_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    token_mask = torch.ones((4, 1), dtype=torch.float32)

    advantages = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=token_mask * sample_mask.unsqueeze(-1),
    )[:, 0]

    actual = advantages[sample_mask.bool()]
    assert torch.allclose(
        actual,
        torch.tensor(expected),
        atol=1e-6,
        rtol=0.0,
    ), (
        "masked sample changed the per-prompt population: "
        f"expected valid advantages {expected}, observed {actual.tolist()}"
    )
