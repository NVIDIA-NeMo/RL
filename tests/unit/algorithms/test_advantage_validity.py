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

"""S4: validity-aware GRPO baseline (token-capture placeholder rows)."""

import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)


def _estimator(**overrides) -> GRPOAdvantageEstimator:
    config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False, normalize_rewards=False, **overrides
    )
    return GRPOAdvantageEstimator(config, loss_config=None)


def test_invalid_rows_do_not_bias_the_baseline():
    prompt_ids = torch.zeros(
        4, 3, dtype=torch.long
    )  # one shared prompt (2D, as prompt_ids_for_adv)
    # The last row is a token-capture placeholder: reward 0, sample_mask 0.
    rewards = torch.tensor([1.0, 3.0, 2.0, 0.0])
    valid_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    mask = torch.ones(4, 5)

    adv = _estimator().compute_advantage(
        prompt_ids, rewards, mask, valid_mask=valid_mask
    )
    # Baseline over valid rows only: mean(1,3,2) = 2 (placeholder's 0 excluded).
    assert torch.allclose(adv[0], torch.full((5,), -1.0))
    assert torch.allclose(adv[1], torch.full((5,), 1.0))
    assert torch.allclose(adv[2], torch.full((5,), 0.0))


def test_none_valid_mask_keeps_legacy_all_valid_behavior():
    prompt_ids = torch.zeros(2, 3, dtype=torch.long)
    rewards = torch.tensor([1.0, 3.0])
    mask = torch.ones(2, 3)
    legacy = _estimator().compute_advantage(prompt_ids, rewards, mask)
    explicit = _estimator().compute_advantage(
        prompt_ids, rewards, mask, valid_mask=torch.ones(2)
    )
    assert torch.equal(legacy, explicit)
