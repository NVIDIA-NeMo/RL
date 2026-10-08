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

"""Independent reward-statistics and loss masks for group-relative estimators."""

import pytest
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
    valid_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    mask = valid_mask.unsqueeze(-1).expand(4, 5)
    rewards = torch.zeros(4)  # unused by GDPO

    low = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(0.0), valid_mask=valid_mask
    )
    high = _gdpo_estimator().compute_advantage(
        prompt_ids, rewards, mask, _gdpo_batch(100.0), valid_mask=valid_mask
    )
    torch.testing.assert_close(low[:3], high[:3])


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
    valid_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    mask = valid_mask.unsqueeze(-1).expand(4, 5)

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
    torch.testing.assert_close(low[:3], high[:3])


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


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("leave_one_out", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_gdpo_statistics_match_participating_rewards(include, leave_one_out, normalize):
    """Check actual final values, including weighted components and whitening."""
    prompt_ids = torch.tensor([[0]] * 4 + [[1]] * 4)
    loss_weights = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    valid = torch.ones(8) if include else loss_weights
    loss_mask = loss_weights[:, None].expand(8, 3)
    batch = {
        "reward/a": torch.tensor([1.0, 3.0, 2.0, 20.0, 2.0, 5.0, 4.0, -10.0]),
        "reward/b": torch.tensor([0.0, 1.0, 1.0, -5.0, 1.0, 2.0, 0.0, 30.0]),
    }
    weights = [0.25, 1.5]
    estimator = GDPOAdvantageEstimator(
        AdvEstimatorConfig(
            use_leave_one_out_baseline=leave_one_out,
            normalize_rewards=normalize,
            reward_weights=weights,
        ),
        None,
    )
    actual = estimator.compute_advantage(
        prompt_ids, None, loss_mask, batch, valid_mask=valid
    )
    # Independent scalar reference for each participating row and reward.
    expected = torch.zeros(8)
    for i in range(8):
        if not valid[i]:
            continue
        peers = (prompt_ids[:, 0] == prompt_ids[i, 0]) & valid.bool()
        if leave_one_out:
            peers[i] = False
        for weight, rewards in zip(weights, batch.values()):
            values = rewards[peers]
            advantage = rewards[i] - values.mean()
            if normalize and values.std() > 0:
                advantage /= values.std() + 1e-6
            expected[i] += weight * advantage
    participants = expected[valid.bool()]
    expected = (expected - participants.mean()) / participants.std()
    torch.testing.assert_close(
        actual[loss_weights.bool(), 0], expected[loss_weights.bool()]
    )
    assert torch.isfinite(actual).all()
    assert (actual * loss_mask)[~loss_weights.bool()].count_nonzero() == 0
    changed = {key: value.clone() for key, value in batch.items()}
    for value in changed.values():
        value[~loss_weights.bool()] += 100
    perturbed = estimator.compute_advantage(
        prompt_ids, None, loss_mask, changed, valid_mask=valid
    )
    if include:
        assert not torch.allclose(
            actual[loss_weights.bool()], perturbed[loss_weights.bool()]
        )
    else:
        torch.testing.assert_close(
            actual[loss_weights.bool()], perturbed[loss_weights.bool()]
        )


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("minus_baseline", [False, True])
@pytest.mark.parametrize("use_kl", [False, True])
def test_reinforce_pp_uses_independent_token_statistics(
    include, minus_baseline, use_kl
):
    """Different response lengths and two prompts exercise both normalization stages."""
    prompt_ids = torch.tensor([[0], [0], [0], [1], [1], [1]])
    rewards = torch.tensor([1.0, 3.0, 10.0, 2.0, 5.0, -4.0])
    response_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    loss_weights = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    valid = torch.ones(6) if include else loss_weights
    loss_mask = response_mask * loss_weights[:, None]
    stats_mask = response_mask * valid[:, None]
    policy_logprobs = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    reference_logprobs = torch.zeros_like(policy_logprobs)
    estimator = ReinforcePlusPlusAdvantageEstimator(
        AdvEstimatorConfig(minus_baseline=minus_baseline),
        ClippedPGLossConfig(
            use_kl_in_reward=use_kl,
            reference_policy_kl_penalty=0.2,
            reference_policy_kl_type="k1",
        ),
    )

    def compute(r, logprobs=policy_logprobs):
        return estimator.compute_advantage(
            prompt_ids,
            r,
            loss_mask,
            valid_mask=valid,
            normalization_mask=stats_mask,
            logprobs_policy=logprobs,
            logprobs_reference=reference_logprobs,
        )

    actual = compute(rewards)
    raw = rewards.clone()
    if minus_baseline:
        for group in [0, 1]:
            members = prompt_ids[:, 0] == group
            raw[members] -= rewards[members & valid.bool()].mean()
    raw = raw[:, None].expand_as(loss_mask)
    if use_kl:
        raw = raw - 0.2 * policy_logprobs
    participants = raw[stats_mask.bool()]
    expected = (raw - participants.mean()) / participants.var(correction=0).clamp(
        min=1e-8
    ).sqrt()
    torch.testing.assert_close(actual, expected)
    assert (actual * loss_mask)[~loss_mask.bool()].count_nonzero() == 0
    changed = rewards.clone()
    changed[~loss_weights.bool()] += 100
    if include:
        assert not torch.allclose(
            actual[loss_mask.bool()], compute(changed)[loss_mask.bool()]
        )
    else:
        torch.testing.assert_close(
            actual[loss_mask.bool()], compute(changed)[loss_mask.bool()]
        )
    # Neither policy admits prompt or padding tokens to normalization.
    changed_logprobs = policy_logprobs.clone()
    changed_logprobs[~response_mask.bool()] += 1000
    torch.testing.assert_close(
        actual[loss_mask.bool()], compute(rewards, changed_logprobs)[loss_mask.bool()]
    )


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("name", ["gdpo", "reinforce_plus_plus"])
@pytest.mark.parametrize("num_valid", [0, 1])
@pytest.mark.parametrize("include", [False, True])
def test_masked_estimators_handle_empty_and_singleton_groups(
    name, num_valid, include, mask_dtype
):
    prompt_ids = torch.zeros(3, 1, dtype=torch.long)
    rewards = torch.tensor([1.0, 2.0, 4.0])
    loss_weights = (torch.arange(3) < num_valid).float()
    loss_mask = loss_weights[:, None].expand(3, 2).to(mask_dtype)
    valid = torch.ones(3) if include else loss_weights
    if name == "gdpo":
        estimator = _gdpo_estimator()
    else:
        estimator = _rpp_estimator()
    actual = estimator.compute_advantage(
        prompt_ids,
        rewards,
        loss_mask,
        valid_mask=valid,
        normalization_mask=valid[:, None].expand(3, 2),
        repeated_batch={"reward/a": rewards, "reward/b": rewards.square()},
    )
    assert actual.dtype == rewards.dtype
    assert torch.isfinite(actual).all()
    assert (actual * loss_mask)[~loss_weights.bool()].count_nonzero() == 0
    if not include:
        assert actual.count_nonzero() == 0


def test_reinforce_pp_empty_legacy_normalization_mask_is_finite():
    actual = _rpp_estimator().compute_advantage(
        torch.zeros(3, 1, dtype=torch.long),
        torch.tensor([1.0, 2.0, 4.0]),
        torch.zeros(3, 2),
    )
    assert torch.isfinite(actual).all()
    assert actual.count_nonzero() == 0
