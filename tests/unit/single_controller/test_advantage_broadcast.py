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
"""Rollout-level advantage broadcast for token-capture segment rows (pure torch)."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.segment_stats import (
    broadcast_rollout_advantages,
    resolve_row_identity,
    subset_rows,
)


def _loo_grpo(group_pos: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Reference leave-one-out GRPO baseline (what GRPOAdvantageEstimator does).

    One scalar advantage per row: reward minus the mean of the OTHER rows of
    the same group (0 when a group has a single row).
    """
    adv = torch.zeros_like(rewards)
    for g in group_pos.unique().tolist():
        rows = (group_pos == g).nonzero().reshape(-1)
        if rows.numel() < 2:
            continue
        total = rewards[rows].sum()
        for r in rows.tolist():
            adv[r] = rewards[r] - (total - rewards[r]) / (rows.numel() - 1)
    return adv


def test_broadcast_copies_canonical_rows_to_their_rollout():
    seq = 6
    canonical = torch.tensor([[0.5] * seq, [-0.5] * seq, [2.0] * seq])
    out = broadcast_rollout_advantages(canonical, torch.tensor([0, 0, 1, 2, 2]))
    assert out.shape == (5, seq)
    assert torch.equal(out[0], canonical[0]) and torch.equal(out[1], canonical[0])
    assert torch.equal(out[2], canonical[1])
    assert torch.equal(out[3], canonical[2]) and torch.equal(out[4], canonical[2])
    # Identity mapping (no segment rows) is a no-op copy.
    assert torch.equal(
        broadcast_rollout_advantages(canonical, torch.arange(3)), canonical
    )


def test_broadcast_rejects_orphans_bad_indices_and_shapes():
    canonical = torch.ones(2, 3)
    with pytest.raises(ValueError, match="no canonical row"):
        broadcast_rollout_advantages(canonical, torch.tensor([0, -1]))
    with pytest.raises(ValueError, match="only 2 canonical"):
        broadcast_rollout_advantages(canonical, torch.tensor([0, 2]))
    with pytest.raises(ValueError, match="1-D"):
        broadcast_rollout_advantages(canonical, torch.tensor([[0]]))
    empty = broadcast_rollout_advantages(canonical, torch.zeros(0, dtype=torch.long))
    assert empty.shape == (0, 3)


def test_subset_rows_indexes_every_tensor_along_dim0():
    index = torch.tensor([0, 2])
    out = subset_rows(
        {"a": torch.arange(3), "b": torch.arange(6).reshape(3, 2)}, index
    )
    assert out["a"].tolist() == [0, 2]
    assert out["b"].tolist() == [[0, 1], [4, 5]]
    assert subset_rows({}, index) == {}


def test_canonical_only_estimator_then_broadcast_matches_single_row_grpo():
    """The estimator sees one row per rollout; segment rows inherit, never vote.

    Group g has rollouts 0..3 with rewards [1, 0, 0, 1]; rollout 0 publishes
    two extra rows and rollout 3 one. Computing on the canonical rows and
    broadcasting must equal plain single-row GRPO on the four rollouts, while
    computing on all rows (the pre-segment code path) would let rollout 0
    vote three times and bias every baseline in the group.
    """
    sample_ids = [
        "g_g0",
        "g_g0_t1",
        "g_g0_t2",
        "g_g1",
        "g_g2",
        "g_g3",
        "g_g3_t1",
        "h_g0",
        "h_g1",
    ]
    ident = resolve_row_identity(sample_ids)
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    seq = 4
    mask = torch.ones(len(sample_ids), seq)

    # Canonical-only estimator input, exactly what _advantage_stage builds.
    c = ident.canonical_index
    adv_canonical = _loo_grpo(ident.group_pos[c], rewards[c])
    adv_canonical = adv_canonical.unsqueeze(-1).expand(mask[c].shape)
    adv = broadcast_rollout_advantages(adv_canonical, ident.row_to_canonical)

    # Reference: single-row GRPO over the rollouts (no segment rows at all).
    expected_rollout = _loo_grpo(
        torch.tensor([0, 0, 0, 0, 1, 1]), torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    )
    expected = expected_rollout[ident.rollout_key]
    assert torch.allclose(adv[:, 0], expected)
    # All rows of a rollout share its advantage; row shape is preserved.
    assert adv.shape == mask.shape
    assert torch.equal(adv[0], adv[1]) and torch.equal(adv[1], adv[2])
    assert torch.equal(adv[5], adv[6])

    # Naive all-rows GRPO is NOT the same: rollout 0 appears three times.
    naive = _loo_grpo(ident.group_pos, rewards)
    assert not torch.allclose(naive, expected)


def test_all_canonical_rows_reduce_to_the_plain_path():
    ident = resolve_row_identity(["g_g0", "g_g1", "h_g0", "h_g1"])
    assert not ident.has_extra_rows
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv_all = _loo_grpo(ident.group_pos, rewards)
    c = ident.canonical_index
    assert torch.equal(c, torch.arange(4))
    adv_c = _loo_grpo(ident.group_pos[c], rewards[c]).unsqueeze(-1).expand(4, 3)
    assert torch.equal(
        broadcast_rollout_advantages(adv_c, ident.row_to_canonical),
        adv_all.unsqueeze(-1).expand(4, 3),
    )


def _loo_grpo_normalized(
    group_pos: torch.Tensor, rewards: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Reference for GRPOAdvantageEstimator with LOO + normalize_rewards + valid_mask.

    Mirrors ``calculate_baseline_and_std_per_prompt``: for a voting row the
    baseline is the mean of the OTHER valid rows, the std is the unbiased std
    of those rows, and the advantage is ``(r - baseline) / (std + 1e-6)`` when
    ``std > 0``. Only voting rows are meaningful here (the stage masks the
    others).
    """
    adv = torch.zeros_like(rewards)
    for g in group_pos.unique().tolist():
        rows = (group_pos == g).nonzero().reshape(-1)
        for r in rows.tolist():
            others = [o for o in rows.tolist() if o != r and valid[o] > 0]
            if valid[r] <= 0 or not others:
                continue
            vals = rewards[others]
            baseline = vals.mean()
            std = vals.std(unbiased=True) if len(others) > 1 else torch.tensor(0.0)
            adv[r] = rewards[r] - baseline
            if std > 0:
                adv[r] = adv[r] / (std + 1e-6)
    return adv


def test_gated_canonical_with_passing_extra_votes_via_amax_and_gets_loo_baseline():
    """F2 of refute-multirow-grpo / A-1 of pr-rl-4124.

    Group rewards [1, 1, 0, 0]; rollout 0's canonical row is gate-masked, its
    segment row passes. The rollout trains (through the segment row), so its
    reward must be in the baseline: with the amax vote the estimator sees
    valid [1, 1, 1, 1] and rollout 0's LOO baseline is mean(1, 0, 0) = 1/3
    (advantage (1 - 1/3) / 0.57735 = 1.1547). Voting only through the
    canonical row (valid [0, 1, 1, 1]) would hand the trained segment the
    estimator's non-voting-row value sum(valid)/(num_valid - 1) = 0.5
    (advantage 0.7071) and remove rollout 0 from its siblings' baselines.
    """
    from nemo_rl.algorithms.single_controller_utils.segment_stats import (
        rollout_vote_mask,
    )

    sample_ids = ["g_g0", "g_g1", "g_g2", "g_g3", "g_g0_t1"]
    ident = resolve_row_identity(sample_ids)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
    # What the stage builds: canonical row 0 gated, every other row trains.
    final_sample_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    baseline_mask = final_sample_mask  # no masked_sample_rewards_in_baseline

    c = ident.canonical_index
    votes = rollout_vote_mask(baseline_mask, ident.rollout_key, c)
    assert votes.tolist() == [1.0, 1.0, 1.0, 1.0]
    canonical_only = baseline_mask.index_select(0, c)
    assert canonical_only.tolist() == [0.0, 1.0, 1.0, 1.0]

    expected = _loo_grpo_normalized(ident.group_pos[c], rewards[c], votes)
    assert torch.allclose(expected, torch.tensor([1.154698, 1.154698, -1.154699, -1.154699]), atol=1e-5)
    adv = broadcast_rollout_advantages(
        expected.unsqueeze(-1).expand(4, 3), ident.row_to_canonical
    )
    # The trained segment row gets the LOO-correct advantage (baseline 1/3).
    assert adv[4, 0].item() == pytest.approx(1.154698, abs=1e-5)
    assert adv[4, 0].item() != pytest.approx(0.707107, abs=1e-3)
    # Siblings see rollout 0's reward: 1.1547 / -1.1547, not 1.0 / -0.7071.
    assert adv[1, 0].item() == pytest.approx(1.154698, abs=1e-5)
    assert adv[2, 0].item() == pytest.approx(-1.154699, abs=1e-5)


def test_real_estimator_reproduces_the_gated_canonical_numbers():
    """Container-only twin of the test above against GRPOAdvantageEstimator."""
    from nemo_rl.algorithms.single_controller_utils.segment_stats import (
        rollout_vote_mask,
    )

    estimator_module = pytest.importorskip(
        "nemo_rl.algorithms.advantage_estimator",
        reason="needs the trainer stack (nemo.lens) on this host",
    )
    loss_module = pytest.importorskip("nemo_rl.algorithms.loss")
    ident = resolve_row_identity(["g_g0", "g_g1", "g_g2", "g_g3", "g_g0_t1"])
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
    baseline_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    c = ident.canonical_index
    votes = rollout_vote_mask(baseline_mask, ident.rollout_key, c)
    canonical_only = baseline_mask.index_select(0, c)
    expected = _loo_grpo_normalized(ident.group_pos[c], rewards[c], votes)
    estimator = estimator_module.GRPOAdvantageEstimator(
        estimator_module.AdvEstimatorConfig(
            normalize_rewards=True, use_leave_one_out_baseline=True
        ),
        loss_module.ClippedPGLossConfig(),
    )
    prompt_ids = torch.zeros(4, 2, dtype=torch.long)
    with_votes = estimator.compute_advantage(
        prompt_ids=prompt_ids, rewards=rewards[c], mask=torch.ones(4, 3), valid_mask=votes
    )
    torch.testing.assert_close(with_votes[:, 0], expected, atol=1e-5, rtol=0)
    canonical_vote = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards[c],
        mask=torch.ones(4, 3),
        valid_mask=canonical_only,
    )
    # The non-voting row's value the old code broadcast to the trained segment.
    assert canonical_vote[0, 0].item() == pytest.approx(0.707107, abs=1e-5)
    assert canonical_vote[1, 0].item() == pytest.approx(1.0, abs=1e-5)


def test_rollout_with_every_row_masked_still_does_not_vote():
    from nemo_rl.algorithms.single_controller_utils.segment_stats import (
        rollout_vote_mask,
    )

    ident = resolve_row_identity(["g_g0", "g_g0_t1", "g_g1", "g_g2", "g_g3"])
    baseline_mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    votes = rollout_vote_mask(baseline_mask, ident.rollout_key, ident.canonical_index)
    assert votes.tolist() == [0.0, 1.0, 1.0, 1.0]
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    c = ident.canonical_index
    expected = _loo_grpo_normalized(ident.group_pos[c], rewards[c], votes)
    # Rollout 0 is out of its siblings' baselines: rollout 1 sees mean(0, 0).
    assert expected[1].item() == pytest.approx(1.0)
    assert expected[2].item() == pytest.approx(-0.5 / (0.707107 + 1e-6), abs=1e-4)
