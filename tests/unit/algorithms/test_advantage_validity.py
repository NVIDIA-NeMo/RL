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

"""S4: validity-aware group baselines (token-capture placeholder rows)."""

import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GDPOAdvantageEstimator,
    GRPOAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig


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


# ── GDPO ─────────────────────────────────────────────────────────────────────


def _gdpo_estimator() -> GDPOAdvantageEstimator:
    config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False, normalize_rewards=False, reward_weights=None
    )
    return GDPOAdvantageEstimator(config, loss_config=None)


def _gdpo_batch(placeholder_reward: float) -> dict[str, torch.Tensor]:
    # Row 3 is a token-capture placeholder; its copied rewards are an outlier.
    return {
        "reward/a": torch.tensor([1.0, 3.0, 2.0, placeholder_reward]),
        "reward/b": torch.tensor([0.0, 1.0, 1.0, placeholder_reward]),
    }


def test_gdpo_placeholder_reward_does_not_move_valid_rows_when_masked():
    prompt_ids = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.ones(4, 5)
    valid_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    rewards = torch.zeros(4)  # unused by GDPO

    low = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(0.0), valid_mask=valid_mask
    )
    high = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(100.0), valid_mask=valid_mask
    )
    # GDPO's final zero-mean/unit-std normalization runs over all rows, so the
    # placeholder's own (masked-out) advantage shifts the scale slightly, but
    # the valid rows' ordering and relative spacing must be untouched.
    valid_low = low[:3, 0]
    valid_high = high[:3, 0]
    assert torch.allclose(
        valid_low - valid_low.mean(),
        (valid_high - valid_high.mean()) * (valid_low.std() / valid_high.std()),
        atol=1e-5,
    )


def test_gdpo_placeholder_reward_biases_valid_rows_without_mask():
    prompt_ids = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.ones(4, 5)
    rewards = torch.zeros(4)

    low = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(0.0)
    )
    high = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(100.0)
    )
    # Without the mask the outlier enters the per-prompt mean and the valid
    # rows' relative spacing changes (here: their sign pattern flips).
    assert not torch.allclose(torch.sign(low[:3, 0]), torch.sign(high[:3, 0]))


def test_gdpo_none_valid_mask_keeps_legacy_all_valid_behavior():
    prompt_ids = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.ones(4, 5)
    rewards = torch.zeros(4)
    legacy = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(5.0)
    )
    explicit = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(5.0), valid_mask=torch.ones(4)
    )
    assert torch.equal(legacy, explicit)


# ── Reinforce++ ──────────────────────────────────────────────────────────────


def _rpp_estimator() -> ReinforcePlusPlusAdvantageEstimator:
    config = AdvEstimatorConfig.model_construct(minus_baseline=True)
    loss_config = ClippedPGLossConfig.model_construct(
        use_kl_in_reward=False,
        reference_policy_kl_penalty=0.0,
        reference_policy_kl_type="k1",
    )
    return ReinforcePlusPlusAdvantageEstimator(config, loss_config=loss_config)


def test_reinforce_pp_placeholder_reward_does_not_move_valid_rows_when_masked():
    prompt_ids = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.ones(4, 5)
    valid_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

    low = _rpp_estimator().compute_advantage(
        prompt_ids,
        torch.tensor([1.0, 3.0, 2.0, 0.0]),
        mask,
        valid_mask=valid_mask,
    )
    high = _rpp_estimator().compute_advantage(
        prompt_ids,
        torch.tensor([1.0, 3.0, 2.0, 100.0]),
        mask,
        valid_mask=valid_mask,
    )
    # Reinforce++ globally normalizes over all masked tokens, so compare the
    # valid rows after removing that scale: their centered pattern must match.
    valid_low = low[:3, 0]
    valid_high = high[:3, 0]
    assert torch.allclose(
        valid_low - valid_low.mean(),
        (valid_high - valid_high.mean()) * (valid_low.std() / valid_high.std()),
        atol=1e-5,
    )


def test_reinforce_pp_placeholder_reward_biases_valid_rows_without_mask():
    prompt_ids = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.ones(4, 5)

    low = _rpp_estimator().compute_advantage(
        prompt_ids, torch.tensor([1.0, 3.0, 2.0, 0.0]), mask
    )
    high = _rpp_estimator().compute_advantage(
        prompt_ids, torch.tensor([1.0, 3.0, 2.0, 100.0]), mask
    )
    # Without the mask the outlier enters the per-prompt mean; with mean(1,3,2,100)
    # every valid row is now below the baseline, so their sign pattern flips.
    assert not torch.allclose(torch.sign(low[:3, 0]), torch.sign(high[:3, 0]))


def test_reinforce_pp_none_valid_mask_keeps_legacy_all_valid_behavior():
    prompt_ids = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.ones(4, 5)
    rewards = torch.tensor([1.0, 3.0, 2.0, 5.0])
    legacy = _rpp_estimator().compute_advantage(prompt_ids, rewards, mask)
    explicit = _rpp_estimator().compute_advantage(
        prompt_ids, rewards, mask, valid_mask=torch.ones(4)
    )
    assert torch.equal(legacy, explicit)
