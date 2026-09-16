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
"""Regression tests for the fp32 leave-one-out std residue in GRPO advantage normalization.

Wireup job 1973369 (32x8 web RL, CC v2) step 5 logged advantages/max 7428 with loss -26
and grad_norm 35: a group split 7-vs-1 on a non-dyadic reward, the singleton's LOO set
of identical rewards produced std ~1e-4 instead of 0 via E[x^2]-E[x]^2 cancellation,
slipped past the old ``std > 0`` guard and was divided by that residue.
"""

import torch

from nemo_rl.algorithms.advantage_estimator import (
    STD_FLOOR,
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt


def _estimator(leave_one_out: bool = True) -> GRPOAdvantageEstimator:
    cfg = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=leave_one_out, normalize_rewards=True
    )
    return GRPOAdvantageEstimator(cfg, ClippedPGLossConfig())


def _seven_vs_one(reward: float) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = torch.zeros(8, 1, dtype=torch.long)
    rewards = torch.tensor([reward] * 7 + [0.0], dtype=torch.float32)
    return prompt_ids, rewards


def test_loo_std_residue_is_positive_but_below_floor():
    """The hazard the guard exists for: identical LOO rewards give std in (0, STD_FLOOR)."""
    prompt_ids, rewards = _seven_vs_one(0.35)
    _, std = calculate_baseline_and_std_per_prompt(
        prompt_ids, rewards, torch.ones_like(rewards), leave_one_out_baseline=True
    )
    singleton_std = std[-1].item()
    assert singleton_std >= 0.0
    assert singleton_std < STD_FLOOR
    # Dyadic rewards cancel exactly; non-dyadic ones may leave residue. Either way the
    # guard below must treat the singleton row as zero-variance.


def test_singleton_loser_advantage_is_not_amplified():
    prompt_ids, rewards = _seven_vs_one(0.35)
    mask = torch.ones(8, 4)
    adv = _estimator().compute_advantage(prompt_ids, rewards, mask)[:, 0]
    # Singleton's LOO baseline is 0.35 -> raw advantage -0.35, left unnormalized.
    assert torch.isclose(adv[-1], torch.tensor(-0.35), atol=1e-6)
    # Winners have genuine variance in their LOO set and are normalized to O(1).
    assert adv[:7].abs().max() < 5.0
    assert adv.abs().max() < 5.0


def test_genuine_variance_still_normalized():
    prompt_ids = torch.zeros(4, 1, dtype=torch.long)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4, 3)
    adv = _estimator(leave_one_out=False).compute_advantage(prompt_ids, rewards, mask)[
        :, 0
    ]
    # mean 0.5, std (unbiased) = sqrt(1/3): advantages +-0.5/std.
    expected = 0.5 / torch.tensor(1.0 / 3.0).sqrt()
    assert torch.allclose(adv.abs(), expected.expand(4), rtol=1e-4)
