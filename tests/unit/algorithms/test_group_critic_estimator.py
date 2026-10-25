# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""ppo.adv_estimator.name=group_critic: GRPO's leave-one-out advantage minus a sibling-centred critic term."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GAEConfig,
    GeneralizedAdvantageEstimator,
    GroupCriticAdvantageEstimator,
    GroupCriticConfig,
    GRPOAdvantageEstimator,
    group_critic_advantages,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt

PROMPT_OFFSET = 2  # prompt tokens ahead of every response (mask 0)


def _batch(rollouts, seq_len=None):
    """rollouts: list of (group, reward, values per response token). Returns prompt_ids, rewards, mask, values."""
    seq_len = seq_len or PROMPT_OFFSET + max(len(v) for _, _, v in rollouts) + 1
    n = len(rollouts)
    prompt_ids = torch.tensor([[g, 7] for g, _, _ in rollouts], dtype=torch.long)
    rewards = torch.tensor([r for _, r, _ in rollouts], dtype=torch.float32)
    mask = torch.zeros(n, seq_len)
    values = torch.full((n, seq_len), 9.0)  # junk outside the mask must never be read
    for i, (_, _, v) in enumerate(rollouts):
        mask[i, PROMPT_OFFSET : PROMPT_OFFSET + len(v)] = 1
        values[i, PROMPT_OFFSET : PROMPT_OFFSET + len(v)] = torch.tensor(v)
    return prompt_ids, rewards, mask, values


def _run(rollouts, *, step=100, valid=None, train=None, canonical_of_row=None, **cfg):
    prompt_ids, rewards, mask, values = _batch(rollouts)
    config = GroupCriticConfig(**cfg)
    adv, metrics = group_critic_advantages(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        values=values,
        valid_mask=valid,
        train_mask=train,
        canonical_of_row=canonical_of_row,
        config=config,
        step=step,
    )
    return adv, metrics, (prompt_ids, rewards, mask, values)


def _response(adv, mask, row):
    return adv[row][mask[row] > 0]


def _grpo(prompt_ids, rewards, mask, valid=None):
    est = GRPOAdvantageEstimator(AdvEstimatorConfig(), ClippedPGLossConfig())
    return est.compute_advantage(prompt_ids, rewards, mask, valid_mask=valid)


def _mixed_groups():
    torch.manual_seed(0)
    out = []
    reward_sets = [
        [
            1,
            0,
            0,
            0,
            0,
            0,
        ],  # one success: its siblings are uniform (GRPO leaves it unnormalised)
        [0, 0, 0, 0, 0, 0],  # all fail: uniform group
        [1, 1, 0, 1, 0, 0],
        [0.25, 0.5, 0.75, 0.5, 1.0, 0.0],  # GDPval-like fractions
    ]
    for g, rs in enumerate(reward_sets):
        for r in rs:
            length = int(torch.randint(1, 9, ()).item())
            out.append((g, float(r), torch.rand(length).tolist()))
    return out


def test_zero_weight_equals_grpo_exactly():
    rollouts = _mixed_groups()
    valid = torch.ones(len(rollouts))
    valid[7] = 0.0  # a non-voting rollout (e.g. a token-capture placeholder)
    adv, _, (pids, rewards, mask, _) = _run(
        rollouts, weight="constant", constant_weight=0.0, valid=valid
    )
    expected = _grpo(pids, rewards, mask, valid)
    on = mask > 0
    assert torch.equal(adv[on], expected[on])
    assert torch.equal(adv[~on], torch.zeros_like(adv[~on]))


def test_shared_offset_per_group_cancels():
    rollouts = _mixed_groups()
    adv, _, (pids, rewards, mask, values) = _run(rollouts, ramp_per_step=None)
    shifted = values.clone()
    for i, (g, _, _) in enumerate(rollouts):
        shifted[i] += (
            0.37 * (g + 1) - 0.5
        )  # a different shift per group, the same for its siblings
    config = GroupCriticConfig(ramp_per_step=None)
    adv2, _ = group_critic_advantages(
        prompt_ids=pids,
        rewards=rewards,
        mask=mask,
        values=shifted,
        valid_mask=None,
        train_mask=None,
        canonical_of_row=None,
        config=config,
        step=1,
    )
    assert torch.allclose(adv, adv2, atol=1e-5)


def test_worked_example_from_the_design_doc():
    # G = 4, R = (1, 1, 0, 0), critic values at three stages (one token each), w = 1, before the std division.
    rollouts = [
        (0, 1.0, [0.50, 0.80, 0.95]),
        (0, 1.0, [0.50, 0.70, 0.90]),
        (0, 0.0, [0.50, 0.30, 0.10]),
        (0, 0.0, [0.50, 0.30, 0.05]),
    ]
    adv, _, (_, _, mask, _) = _run(
        rollouts,
        weight="constant",
        constant_weight=1.0,
        ramp_per_step=None,
        normalize_rewards=False,
    )
    assert torch.allclose(
        _response(adv, mask, 0), torch.tensor([2 / 3, 0.3, 0.2 / 3]), atol=1e-6
    )
    assert torch.allclose(
        _response(adv, mask, 2),
        torch.tensor([-2 / 3, -2 / 3 + 0.3, -2 / 3 + 1.6 / 3]),
        atol=1e-6,
    )


def test_finished_sibling_holds_its_last_value():
    rollouts = [
        (0, 1.0, [0.2, 0.4, 0.6, 0.8]),
        (0, 0.0, [0.5, 0.1]),  # ends after two tokens: held at 0.1 afterwards
    ]
    adv, _, (_, _, mask, _) = _run(
        rollouts,
        weight="constant",
        constant_weight=1.0,
        ramp_per_step=None,
        normalize_rewards=False,
    )
    sibling = torch.tensor([0.5, 0.1, 0.1, 0.1])
    expected = 1.0 - (torch.tensor([0.2, 0.4, 0.6, 0.8]) - sibling)
    assert torch.allclose(_response(adv, mask, 0), expected, atol=1e-6)


def test_positions_count_response_tokens_only():
    # Row 0 has a 3-token gap (tool output, mask 0) between its 2nd and 3rd response tokens.
    prompt_ids = torch.tensor([[0, 7], [0, 7]])
    rewards = torch.tensor([1.0, 0.0])
    mask = torch.zeros(2, 10)
    values = torch.full((2, 10), 9.0)
    mask[0, [2, 3, 7, 8]] = 1
    values[0, [2, 3, 7, 8]] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    mask[1, 2:6] = 1
    values[1, 2:6] = torch.tensor([0.5, 0.6, 0.7, 0.8])
    adv, _ = group_critic_advantages(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        values=values,
        valid_mask=None,
        train_mask=None,
        canonical_of_row=None,
        config=GroupCriticConfig(
            weight="constant",
            constant_weight=1.0,
            ramp_per_step=None,
            normalize_rewards=False,
        ),
        step=1,
    )
    expected = 1.0 - (
        torch.tensor([0.1, 0.2, 0.3, 0.4]) - torch.tensor([0.5, 0.6, 0.7, 0.8])
    )
    assert torch.allclose(adv[0, [2, 3, 7, 8]], expected, atol=1e-6)
    assert torch.equal(adv[0, [4, 5, 6]], torch.zeros(3))


def test_weight_is_fitted_leaving_the_prompt_group_out():
    # Group 0: the critic term equals R - L exactly (alone it would fit w = 1).
    # Group 1: the critic term is anti-correlated with R - L (alone it would fit w < 0 -> 0).
    g0 = [(0, 1.0, [0.9, 0.9]), (0, 0.0, [0.1, 0.1])]
    g1 = [(1, 1.0, [0.2, 0.2]), (1, 0.0, [0.8, 0.8])]
    adv, metrics, (pids, rewards, mask, values) = _run(
        g0 + g1, ramp_per_step=None, normalize_rewards=False
    )
    # Group 0's weight comes from group 1 only (clipped to 0): its advantages are GRPO's.
    assert torch.allclose(_response(adv, mask, 0), torch.tensor([1.0, 1.0]))
    assert torch.allclose(_response(adv, mask, 1), torch.tensor([-1.0, -1.0]))
    # Group 1's weight comes from group 0 only: term_0 = 0.9 - 0.1 = 0.8 per token, y = 1 -> w = 0.8 / 0.64 > 1 -> 1.
    assert torch.allclose(
        _response(adv, mask, 2), torch.tensor([1.0 - (0.2 - 0.8)] * 2)
    )
    assert torch.allclose(
        _response(adv, mask, 3), torch.tensor([-1.0 - (0.8 - 0.2)] * 2)
    )


def test_ramp_caps_the_weight():
    rollouts = [(0, 1.0, [0.9]), (0, 0.0, [0.1])]
    adv, metrics, (_, _, mask, _) = _run(
        rollouts,
        step=3,
        weight="constant",
        constant_weight=1.0,
        ramp_per_step=0.1,
        normalize_rewards=False,
    )
    assert metrics["hybrid/weight_cap"] == pytest.approx(0.3)
    assert torch.allclose(
        _response(adv, mask, 0), torch.tensor([1.0 - 0.3 * 0.8]), atol=1e-6
    )


@pytest.mark.parametrize("mode", ["mean_normalized", "none"])
def test_uniform_group_scale(mode):
    rollouts = [
        (0, 0.0, [0.6, 0.6]),
        (0, 0.0, [0.2, 0.2]),
        (0, 0.0, [0.4, 0.4]),  # all fail: GRPO advantage 0
        (1, 1.0, [0.5]),
        (1, 0.0, [0.5]),
        (1, 1.0, [0.5]),
    ]
    adv, metrics, (pids, rewards, mask, _) = _run(
        rollouts,
        weight="constant",
        constant_weight=1.0,
        ramp_per_step=None,
        uniform_group_scale=mode,
    )
    # The mean GRPO scale over the normalised rollouts: rows 3 and 5 (row 4's siblings are both 1: GRPO leaves it
    # unnormalised, so it does not count).
    _, std, trivial = calculate_baseline_and_std_per_prompt(
        pids, rewards, torch.ones(6)
    )
    normalized = (std > 0) & ~trivial
    assert normalized.tolist() == [False, False, False, True, False, True]
    scale = (
        float((std[normalized].double() + 1e-6).mean())
        if mode == "mean_normalized"
        else 1.0
    )
    assert metrics["hybrid/uniform_scale"] == pytest.approx(scale)
    term_0 = 0.6 - (0.2 + 0.4) / 2
    assert torch.allclose(
        _response(adv, mask, 0), torch.tensor([-term_0 / scale] * 2), atol=1e-5
    )


def test_segment_rows_get_grpo_and_pad_rows_get_nothing():
    rollouts = [
        (0, 1.0, [0.9, 0.8]),  # row 0: canonical of rollout A
        (0, 0.0, [0.1, 0.2]),  # row 1: canonical of rollout B
        (0, 0.0, [0.3, 0.3]),  # row 2: canonical of rollout C
        (0, 1.0, [0.5, 0.5, 0.5]),  # row 3: a subagent segment row of rollout A
        (0, 1.0, [0.9, 0.8]),  # row 4: a DP pad row (copy of row 0)
    ]
    owner = torch.tensor([0, 1, 2, 0, -1])
    adv, _, (pids, rewards, mask, _) = _run(
        rollouts,
        canonical_of_row=owner,
        weight="constant",
        constant_weight=1.0,
        ramp_per_step=None,
    )
    grpo = _grpo(pids[:3], rewards[:3], mask[:3])  # siblings = the canonical rows only
    assert (
        grpo[0, 0] == 1.0
    )  # rollout A's siblings are uniform (0, 0): GRPO leaves it unnormalised
    assert torch.equal(_response(adv, mask, 3), grpo[0, 0].expand(3))
    assert torch.equal(adv[4], torch.zeros_like(adv[4]))
    # The segment row is not a sibling: row 0's term uses rows 1 and 2 only (scale 1, see above).
    expected = 1.0 - (
        torch.tensor([0.9, 0.8]) - torch.tensor([(0.1 + 0.3) / 2, (0.2 + 0.3) / 2])
    )
    assert torch.allclose(_response(adv, mask, 0), expected, atol=1e-6)


def test_masked_rollout_is_a_sibling_but_not_fitted():
    rollouts = [(0, 1.0, [0.9, 0.9]), (0, 0.0, [0.1, 0.1]), (0, 0.0, [0.5, 0.5])]
    _, _, (pids, rewards, mask, values) = _run(rollouts)
    train = mask.clone()
    train[2] = (
        0  # rollout 2 carries no loss (e.g. context-limited) but its reward votes
    )
    cfg = GroupCriticConfig(
        weight="constant",
        constant_weight=1.0,
        ramp_per_step=None,
        normalize_rewards=False,
    )
    kw = dict(
        prompt_ids=pids,
        rewards=rewards,
        mask=mask,
        valid_mask=None,
        train_mask=train,
        canonical_of_row=None,
        config=cfg,
        step=1,
    )
    adv_a, _ = group_critic_advantages(values=values, **kw)
    values_b = values.clone()
    values_b[2] += 0.2
    adv_b, _ = group_critic_advantages(values=values_b, **kw)
    assert not torch.allclose(_response(adv_a, mask, 0), _response(adv_b, mask, 0))


def test_estimator_returns_gae_returns():
    rollouts = _mixed_groups()
    pids, rewards, mask, values = _batch(rollouts)
    values = (
        values * mask
    )  # GAE reads every position; keep the junk out of the comparison
    cfg = GAEConfig(name="group_critic", gae_lambda=1.0)
    est = GroupCriticAdvantageEstimator(cfg, ClippedPGLossConfig())
    est.current_step = 5
    adv, returns = est.compute_advantage(
        pids, rewards, mask, values, canonical_of_row=torch.arange(len(rollouts))
    )
    _, gae_returns = GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0), ClippedPGLossConfig()
    ).compute_advantage(pids, rewards, mask, values)
    assert torch.equal(returns, gae_returns)
    assert adv.shape == mask.shape
    assert (
        "hybrid/weight_cap" in est.last_metrics and "adv_raw/mean" in est.last_metrics
    )
    assert "adv_raw/whiten_gain" not in est.last_metrics
    with pytest.raises(ValueError, match="row identity"):
        est.compute_advantage(pids, rewards, mask, values)


def test_config_validation_and_factory():
    with pytest.raises(ValueError):
        GroupCriticConfig(position_boundaries=[100, 50])
    with pytest.raises(ValueError):
        GroupCriticAdvantageEstimator(
            GAEConfig(name="group_critic", gae_gamma=0.99), ClippedPGLossConfig()
        )
    from nemo_rl.algorithms.ppo import _create_advantage_estimator

    cfg = GAEConfig(name="group_critic", gae_lambda=1.0)
    master = SimpleNamespace(
        ppo=SimpleNamespace(adv_estimator=cfg, num_generations_per_prompt=16),
        loss_fn=ClippedPGLossConfig(),
    )
    assert isinstance(
        _create_advantage_estimator(master), GroupCriticAdvantageEstimator
    )
    master.ppo.num_generations_per_prompt = 1
    with pytest.raises(ValueError):
        _create_advantage_estimator(master)


def test_lambda_must_not_be_decoupled():
    with pytest.raises(ValueError, match="gae_lambda_policy"):
        GroupCriticAdvantageEstimator(
            GAEConfig(
                name="group_critic", gae_lambda_value=1.0, gae_lambda_policy=0.95
            ),
            ClippedPGLossConfig(),
        )


def test_critic_scale_floor_bounds_near_uniform_fractional_groups():
    # 15 x 0.5 and one 0.52: GRPO's leave-one-out std for the 0.5 rollouts is ~0.005.
    rollouts = [(0, 0.52, [0.5])] + [(0, 0.5, [0.5])] * 14 + [(0, 0.5, [0.6])]
    adv, _, (pids, rewards, mask, _) = _run(
        rollouts, weight="constant", constant_weight=1.0, ramp_per_step=None
    )
    grpo = _grpo(pids, rewards, mask)
    term = (
        0.6 - (0.5 * 14 + 0.5) / 15
    )  # the last rollout looks 0.1 better than its siblings
    critic_part = grpo[15][mask[15] > 0] - _response(adv, mask, 15)
    assert torch.allclose(critic_part, torch.tensor([term / 0.25]), atol=1e-5)


def test_fit_ignores_uniform_groups():
    mixed = [
        (0, 1.0, [0.9, 0.9]),
        (0, 0.0, [0.1, 0.1]),
        (1, 1.0, [0.8]),
        (1, 0.0, [0.3]),
    ]
    uniform = [(2, 0.0, [0.9, 0.9]), (2, 0.0, [0.1, 0.1])]
    _, m1, _ = _run(mixed, ramp_per_step=None)
    _, m2, _ = _run(mixed + uniform, ramp_per_step=None)
    assert m1["hybrid/w_fit_pooled_r0"] == pytest.approx(m2["hybrid/w_fit_pooled_r0"])


def test_nonfinite_values_do_not_break_grpo_exactness():
    rollouts = [
        (0, 1.0, [0.9, 0.9, 0.9]),
        (0, 0.0, [0.1, 0.1, 0.1]),
        (0, 0.0, [0.2, 0.2, 0.2]),
    ]
    pids, rewards, mask, values = _batch(rollouts)
    values[0, PROMPT_OFFSET + 1] = float("nan")
    adv, metrics = group_critic_advantages(
        prompt_ids=pids,
        rewards=rewards,
        mask=mask,
        values=values,
        valid_mask=None,
        train_mask=None,
        canonical_of_row=None,
        config=GroupCriticConfig(weight="constant", constant_weight=0.0),
        step=1,
    )
    grpo = _grpo(pids, rewards, mask)
    on = mask > 0
    assert torch.equal(adv[on], grpo[on])
    assert metrics["hybrid/nonfinite_values"] > 0


def test_canonical_row_of_maps_segment_and_pad_rows():
    from nemo_rl.algorithms.single_controller_utils.segment_stats import (
        RowIdentity,
        canonical_row_of,
    )

    # Real rows (in real_index order): canonical A, canonical B, segment of A. Chunk: [A, pad, B, segA, pad].
    identity = RowIdentity(
        group_pos=torch.tensor([0, 0, 0]),
        rollout_local_idx=torch.tensor([0, 1, 0]),
        trace_in_rollout_idx=torch.tensor([0, 0, 1]),
        trace_kind=["terminal", "terminal", "subagent"],
        rollout_key=torch.tensor([0, 1, 0]),
        is_canonical=torch.tensor([True, True, False]),
        canonical_index=torch.tensor([0, 1]),
        row_to_canonical=torch.tensor([0, 1, 0]),
    )
    real_index = torch.tensor([0, 2, 3])
    assert canonical_row_of(identity, real_index, 5).tolist() == [0, -1, 2, 0, -1]
