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
"""CPU oracle tests for the block-aware ESPO ELBO reductions.

Pure-CPU numerics on small hand-constructed ``[N, noisy_len]`` inputs:
``compute_block_aware_elbo`` (the per-mask primitive) -- per-block realized-ratio
reweight ``Lb/|Mb|``, block boundaries, partial last block, the ``|Mb| == 0``
divide-by-zero guard, ``/L`` normalization, differentiability; the two-mask
``compute_coupled_block_aware_elbo`` (antithetic coupled pair averaged before the
ratio); and the same-mask determinism contract -- all against hand-computed
numbers.
"""

import torch

from nemo_rl.algorithms.espo_logprobs import compute_block_aware_elbo

BLOCK_SIZE = 2
NOISY_OFFSET = 1
NOISY_LEN = 8


def _build_case():
    """Two sequences laid out in the noisy region starting at NOISY_OFFSET.

    Garbage logprobs sit off the response / off the harvest set to prove they
    are ignored. Response col ``c`` corresponds to ``rel = c - NOISY_OFFSET``.

    Sequence 0 (L=5): blocks rel[0,1], rel[2,3], rel[4] (partial, Lb=1).
      b0: harvest rel0 only          -> Mb=1, sum=-1.0  -> (2/1)*-1.0 = -2.0
      b1: harvest rel2, rel3         -> Mb=2, sum=-2.0  -> (2/2)*-2.0 = -2.0
      b2: harvest none               -> Mb=0            -> 0 (guard)
      L_hat = -4.0 ; Lnorm = -4.0 / 5
    Sequence 1 (L=3): blocks rel[0,1], rel[2] (partial, Lb=1).
      b0: harvest rel0, rel1         -> Mb=2, sum=-4.0  -> (2/2)*-4.0 = -4.0
      b1: harvest rel2               -> Mb=1, sum=-3.0  -> (1/1)*-3.0 = -3.0
      L_hat = -7.0 ; Lnorm = -7.0 / 3
    """
    lp = torch.full((2, NOISY_LEN), 99.0)
    harvest = torch.zeros((2, NOISY_LEN))

    # seq 0
    lp[0, NOISY_OFFSET + 0] = -1.0
    lp[0, NOISY_OFFSET + 2] = -0.5
    lp[0, NOISY_OFFSET + 3] = -1.5
    harvest[0, NOISY_OFFSET + 0] = 1.0
    harvest[0, NOISY_OFFSET + 2] = 1.0
    harvest[0, NOISY_OFFSET + 3] = 1.0
    # seq 1
    lp[1, NOISY_OFFSET + 0] = -2.0
    lp[1, NOISY_OFFSET + 1] = -2.0
    lp[1, NOISY_OFFSET + 2] = -3.0
    harvest[1, NOISY_OFFSET + 0] = 1.0
    harvest[1, NOISY_OFFSET + 1] = 1.0
    harvest[1, NOISY_OFFSET + 2] = 1.0

    response_lengths = torch.tensor([5, 3], dtype=torch.long)
    return lp, harvest, response_lengths


def test_block_aware_elbo_matches_hand_computed():
    lp, harvest, response_lengths = _build_case()
    elbo = compute_block_aware_elbo(
        lp, harvest, response_lengths, NOISY_OFFSET, BLOCK_SIZE
    )

    expected_elbo = torch.tensor([-4.0, -7.0])
    torch.testing.assert_close(elbo, expected_elbo)


def test_block_aware_elbo_guards_empty_block():
    # A sequence whose every harvested position is gone -> ELBO is exactly 0,
    # never NaN/inf (the |Mb| == 0 guard).
    lp = torch.full((1, NOISY_LEN), -1.0)
    harvest = torch.zeros((1, NOISY_LEN))
    response_lengths = torch.tensor([4], dtype=torch.long)
    elbo = compute_block_aware_elbo(
        lp, harvest, response_lengths, NOISY_OFFSET, BLOCK_SIZE
    )
    torch.testing.assert_close(elbo, torch.zeros(1))


def test_block_aware_elbo_is_differentiable():
    lp, harvest, response_lengths = _build_case()
    lp = lp.clone().requires_grad_(True)
    elbo = compute_block_aware_elbo(
        lp, harvest, response_lengths, NOISY_OFFSET, BLOCK_SIZE
    )
    elbo.sum().backward()
    assert lp.grad is not None
    # Gradient flows only through harvested positions, scaled by the reweight Lb/|Mb|.
    # seq0 b0 (Lb/|Mb|=2): d ELBO / d lp[rel0] = 2.
    torch.testing.assert_close(
        lp.grad[0, NOISY_OFFSET + 0], torch.tensor(2.0)
    )
    # seq0 b1 (Lb/|Mb|=1): d / d lp[rel2] = 1.
    torch.testing.assert_close(
        lp.grad[0, NOISY_OFFSET + 2], torch.tensor(1.0)
    )
    # off the harvest set -> no gradient.
    torch.testing.assert_close(
        lp.grad[0, NOISY_OFFSET + 4], torch.tensor(0.0)
    )


# --------------------------------------------------------------------------- #
# Two complementary masks (ESPO scheme (b)) and the same-mask determinism contract.
# --------------------------------------------------------------------------- #

import pytest  # noqa: E402

from nemo_rl.algorithms.espo_logprobs import (  # noqa: E402
    compute_coupled_block_aware_elbo,
)


def test_coupled_block_aware_elbo_matches_hand_computed():
    # Sequence 0 (L=5), blocks rel[0,1], rel[2,3], rel[4]. Two complementary
    # masks (every response token harvested in exactly one level):
    #   level 0 (mask M)    = {rel0, rel2, rel3}
    #     b0 {rel0}        Mb=1 -> (2/1)*-1.0 = -2.0
    #     b1 {rel2,rel3}   Mb=2 -> (2/2)*-2.0 = -2.0
    #     b2 {}            Mb=0 -> 0
    #     elbo0 = -4.0
    #   level 1 (compl Mbar) = {rel1, rel4}
    #     b0 {rel1}        Mb=1 -> (2/1)*-2.0 = -4.0
    #     b1 {}            Mb=0 -> 0
    #     b2 {rel4}        Mb=1, Lb=1 -> (1/1)*-1.0 = -1.0
    #     elbo1 = -5.0
    #   L_hat = 0.5*(-4.0 + -5.0) = -4.5 ; Lnorm = -4.5 / 5 = -0.9
    lp0 = torch.full((1, NOISY_LEN), 99.0)
    h0 = torch.zeros((1, NOISY_LEN))
    lp0[0, NOISY_OFFSET + 0] = -1.0
    lp0[0, NOISY_OFFSET + 2] = -0.5
    lp0[0, NOISY_OFFSET + 3] = -1.5
    h0[0, NOISY_OFFSET + 0] = 1.0
    h0[0, NOISY_OFFSET + 2] = 1.0
    h0[0, NOISY_OFFSET + 3] = 1.0

    lp1 = torch.full((1, NOISY_LEN), 99.0)
    h1 = torch.zeros((1, NOISY_LEN))
    lp1[0, NOISY_OFFSET + 1] = -2.0
    lp1[0, NOISY_OFFSET + 4] = -1.0
    h1[0, NOISY_OFFSET + 1] = 1.0
    h1[0, NOISY_OFFSET + 4] = 1.0

    response_lengths = torch.tensor([5], dtype=torch.long)
    elbo, lnorm = compute_coupled_block_aware_elbo(
        [lp0, lp1], [h0, h1], response_lengths, NOISY_OFFSET, BLOCK_SIZE
    )
    torch.testing.assert_close(elbo, torch.tensor([-4.5]))
    torch.testing.assert_close(lnorm, torch.tensor([-0.9]))


def test_coupled_mask_is_deterministic_in_seed():
    # Locks the prev == curr == ref masking contract: the level-0 mask M and its
    # complement Mbar are a deterministic function of ``coupled_grpo_seed`` alone,
    # so two independent base constructions with the same seed (the prev / ref /
    # train forwards of one step) harvest exactly the same positions in each level.
    # Skips on login (build_coupled_base pulls BatchedDataDict's heavy import chain).
    pytest.importorskip("nemo_rl.algorithms.coupled_grpo_logprobs")
    from nemo_rl.algorithms.coupled_grpo_logprobs import (
        build_coupled_base,
        make_coupled_level_view,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    mask_token_id, pad_token_id, seq_len = 200, 0, 12
    layout = [(2, 4), (3, 5), (1, 3)]  # (prompt_len, response_len)

    def _make_data():
        n = len(layout)
        input_ids = torch.full((n, seq_len), pad_token_id, dtype=torch.long)
        token_mask = torch.zeros((n, seq_len), dtype=torch.float32)
        input_lengths = torch.zeros(n, dtype=torch.long)
        for sidx, (plen, rlen) in enumerate(layout):
            total = plen + rlen
            input_ids[sidx, :total] = (
                torch.arange(total, dtype=torch.long) + 10 * (sidx + 1) + 1
            )
            token_mask[sidx, plen:total] = 1.0
            input_lengths[sidx] = total
        return BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "token_mask": token_mask,
                "sample_mask": torch.ones(n, dtype=torch.float32),
                "coupled_grpo_seed": torch.arange(n, dtype=torch.long),
            }
        )

    base_a, _, _ = build_coupled_base(
        _make_data(), mask_token_id, pad_token_id, noisy_block_size=None
    )
    base_b, _, _ = build_coupled_base(
        _make_data(), mask_token_id, pad_token_id, noisy_block_size=None
    )
    # Same seed -> identical level-0 mask M ...
    assert torch.equal(
        base_a["coupled_grpo_level0_mask"], base_b["coupled_grpo_level0_mask"]
    )
    # ... AND identical complement Mbar (the level-1 harvested set).
    for level in range(2):
        ha = make_coupled_level_view(base_a, level, ("diffu_grpo_score_mask",))[
            "block_reveal_harvest_mask"
        ]
        hb = make_coupled_level_view(base_b, level, ("diffu_grpo_score_mask",))[
            "block_reveal_harvest_mask"
        ]
        assert torch.equal(ha, hb)


# --------------------------------------------------------------------------- #
# Per-microbatch SAMPLE-MAJOR reshape + read-only level/order guard (loss branch).
# --------------------------------------------------------------------------- #


def _espo_sample_major_groups(rows, num_masks):
    """Mirror the loss branch: reshape [K*M, ...] -> [K, M, ...], level j = [:, j].

    Returns the per-level row lists (each [K, ...]); the i-th entry of level j is
    sequence i's level-j row.
    """
    k = rows.shape[0] // num_masks
    r = rows.reshape(k, num_masks, *rows.shape[1:])
    return [r[:, j] for j in range(num_masks)]


def _espo_guard_ok(levels, sample_index, num_masks):
    """Mirror the loss branch read-only guard for per-microbatch sample-major rows:
    each K-group's levels are [0..M-1], and a group's M rows share one sample index
    (constant across the LEVEL axis)."""
    k = levels.shape[0] // num_masks
    levels_r = levels.reshape(k, num_masks)
    sidx_r = sample_index.reshape(k, num_masks)
    expected = torch.arange(num_masks, dtype=levels_r.dtype).unsqueeze(0)
    if not torch.equal(levels_r, expected.expand_as(levels_r)):
        return False
    if num_masks > 1 and not bool((sidx_r == sidx_r[:, :1]).all()):
        return False
    return True


def test_sample_major_reshape_groups_levels_per_sequence():
    # K=2 sequences, M=2 levels, sample-major rows [s0L0, s0L1, s1L0, s1L1].
    # reshape [4,1] -> [2,2,1]; level 0 = [s0L0, s1L0], level 1 = [s0L1, s1L1].
    num_masks = 2
    rows = torch.tensor([[0.0], [1.0], [10.0], [11.0]])  # s0L0,s0L1,s1L0,s1L1
    level_lps = _espo_sample_major_groups(rows, num_masks)
    torch.testing.assert_close(level_lps[0].squeeze(-1), torch.tensor([0.0, 10.0]))
    torch.testing.assert_close(level_lps[1].squeeze(-1), torch.tensor([1.0, 11.0]))


def test_sample_major_guard_passes_for_valid_order_raises_for_level_major():
    num_masks = 2
    # Valid sample-major: levels [0,1,0,1], sample index [0,0,1,1].
    levels_sm = torch.tensor([0, 1, 0, 1])
    sidx_sm = torch.tensor([0, 0, 1, 1])
    assert _espo_guard_ok(levels_sm, sidx_sm, num_masks)
    # Level-major order [0,0,1,1] (all level-0 rows, then level-1) must be rejected:
    # the K-group levels would be [0,0] / [1,1], not [0,1].
    levels_lm = torch.tensor([0, 0, 1, 1])
    sidx_lm = torch.tensor([0, 1, 0, 1])
    assert not _espo_guard_ok(levels_lm, sidx_lm, num_masks)


def test_espo_reveal_schedule_yields_sample_major_microbatches():
    # ESPORevealSchedule over N samples with K and M=2 yields N/K microbatches,
    # each K*M rows in sample-major order with correctly-grouped levels. Skips on
    # login (build_coupled_base / the worker module pull the heavy import chain).
    pytest.importorskip("nemo_rl.algorithms.coupled_grpo_logprobs")
    worker_mod = pytest.importorskip(
        "nemo_rl.models.policy.workers.espo_megatron_policy_worker"
    )
    from nemo_rl.algorithms.coupled_grpo_logprobs import build_coupled_base
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    ESPORevealSchedule = worker_mod.ESPORevealSchedule
    mask_token_id, pad_token_id, seq_len = 200, 0, 12
    layout = [(2, 4), (3, 5), (1, 3), (2, 2)]  # N = 4 samples
    n = len(layout)
    input_ids = torch.full((n, seq_len), pad_token_id, dtype=torch.long)
    token_mask = torch.zeros((n, seq_len), dtype=torch.float32)
    input_lengths = torch.zeros(n, dtype=torch.long)
    for sidx, (plen, rlen) in enumerate(layout):
        total = plen + rlen
        input_ids[sidx, :total] = (
            torch.arange(total, dtype=torch.long) + 10 * (sidx + 1) + 1
        )
        token_mask[sidx, plen:total] = 1.0
        input_lengths[sidx] = total
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(n, dtype=torch.float32),
            "coupled_grpo_seed": torch.arange(n, dtype=torch.long),
        }
    )
    base, num_samples, num_levels = build_coupled_base(
        data, mask_token_id, pad_token_id, noisy_block_size=None
    )
    assert num_samples == n and num_levels == 2

    K, M = 2, num_levels
    harvest_keys = ("diffu_grpo_score_mask", "diffu_grpo_loss_mask")
    schedule = ESPORevealSchedule(base).configure(
        num_levels=M, harvest_keys=harvest_keys, num_samples_per_micro_batch=K
    )
    # size = N * M; num_microbatches = size / (K*M) = N / K.
    assert schedule.size == n * M
    mbs = list(schedule.make_microbatch_iterator(K * M))
    assert len(mbs) == n // K  # 4 / 2 = 2 microbatches

    for mb in mbs:
        # K*M rows, sample-major: levels [0,1,0,1], sample index constant per group.
        assert mb["input_ids"].shape[0] == K * M
        levels_r = mb["coupled_grpo_level"].reshape(K, M)
        sidx_r = mb["block_reveal_sample_index"].reshape(K, M)
        torch.testing.assert_close(
            levels_r, torch.arange(M).unsqueeze(0).expand(K, M)
        )
        assert bool((sidx_r == sidx_r[:, :1]).all())  # group shares one sample idx
    # Local per-microbatch sample indices: each microbatch indexes its K samples 0..K-1.
    torch.testing.assert_close(
        mbs[0]["block_reveal_sample_index"], torch.tensor([0, 0, 1, 1])
    )


# --------------------------------------------------------------------------- #
# K = 2 coupled pairs (num_mc_samples = 4): ELBO oracle, per-pair mask
# generation / partition / independence, sample-major schedule, and the
# over-counting regression that locks Approach A (per-pair prev/ref summing).
# --------------------------------------------------------------------------- #


def test_coupled_block_aware_elbo_four_masks_matches_hand_computed():
    # K = 2 pairs -> 4 mask levels; compute_coupled_block_aware_elbo averages all
    # four block-aware ELBOs (the 1/num_mc_samples = 1/4 factor). One sequence
    # (L=5), block_size 2 -> blocks rel[0,1], rel[2,3], rel[4].
    #
    # pair 0 level 0 M0    = {rel0, rel2, rel3}   (as in the two-mask oracle above)
    #   b0 {rel0} (2/1)*-1.0=-2.0 ; b1 {rel2,rel3} (2/2)*-2.0=-2.0 ; b2 {} 0  -> -4.0
    # pair 0 level 1 M0bar = {rel1, rel4}
    #   b0 {rel1} (2/1)*-2.0=-4.0 ; b1 {} 0 ; b2 {rel4} (1/1)*-1.0=-1.0  -> -5.0
    # pair 1 level 0 M1    = {rel1, rel2}
    #   b0 {rel1} (2/1)*-3.0=-6.0 ; b1 {rel2} (2/1)*-0.5=-1.0 ; b2 {} 0  -> -7.0
    # pair 1 level 1 M1bar = {rel0, rel3, rel4}
    #   b0 {rel0} (2/1)*-2.0=-4.0 ; b1 {rel3} (2/1)*-1.5=-3.0 ; b2 {rel4} (1/1)*-4.0=-4.0
    #   -> -11.0
    # L_hat = (-4.0 + -5.0 + -7.0 + -11.0)/4 = -27.0/4 = -6.75 ; Lnorm = -6.75/5 = -1.35
    def _mask_lp(positions):
        lp = torch.full((1, NOISY_LEN), 99.0)
        h = torch.zeros((1, NOISY_LEN))
        for rel, val in positions.items():
            lp[0, NOISY_OFFSET + rel] = val
            h[0, NOISY_OFFSET + rel] = 1.0
        return lp, h

    lp00, h00 = _mask_lp({0: -1.0, 2: -0.5, 3: -1.5})
    lp01, h01 = _mask_lp({1: -2.0, 4: -1.0})
    lp10, h10 = _mask_lp({1: -3.0, 2: -0.5})
    lp11, h11 = _mask_lp({0: -2.0, 3: -1.5, 4: -4.0})

    response_lengths = torch.tensor([5], dtype=torch.long)
    elbo, lnorm = compute_coupled_block_aware_elbo(
        [lp00, lp01, lp10, lp11],
        [h00, h01, h10, h11],
        response_lengths,
        NOISY_OFFSET,
        BLOCK_SIZE,
    )
    torch.testing.assert_close(elbo, torch.tensor([-6.75]))
    torch.testing.assert_close(lnorm, torch.tensor([-1.35]))


def _make_coupled_data(layout, seq_len=12, mask_token_id=200, pad_token_id=0):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    n = len(layout)
    input_ids = torch.full((n, seq_len), pad_token_id, dtype=torch.long)
    token_mask = torch.zeros((n, seq_len), dtype=torch.float32)
    input_lengths = torch.zeros(n, dtype=torch.long)
    for sidx, (plen, rlen) in enumerate(layout):
        total = plen + rlen
        input_ids[sidx, :total] = (
            torch.arange(total, dtype=torch.long) + 10 * (sidx + 1) + 1
        )
        token_mask[sidx, plen:total] = 1.0
        input_lengths[sidx] = total
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(n, dtype=torch.float32),
            "coupled_grpo_seed": torch.arange(n, dtype=torch.long),
        }
    )


def test_build_coupled_base_two_pairs_partition_and_independence():
    # build_coupled_base(num_pairs=2) -> 4 levels, deterministic in the seed, with
    # each pair's two complementary levels partitioning the valid tokens, and the
    # two pairs drawing INDEPENDENT masks (pair 0 != pair 1). Skips on login.
    pytest.importorskip("nemo_rl.algorithms.coupled_grpo_logprobs")
    from nemo_rl.algorithms.coupled_grpo_logprobs import (
        build_coupled_base,
        make_coupled_level_view,
    )

    mask_token_id, pad_token_id = 200, 0
    layout = [(2, 4), (3, 5), (1, 3)]

    base_a, n_a, num_levels_a = build_coupled_base(
        _make_coupled_data(layout), mask_token_id, pad_token_id,
        noisy_block_size=None, num_pairs=2,
    )
    base_b, _, _ = build_coupled_base(
        _make_coupled_data(layout), mask_token_id, pad_token_id,
        noisy_block_size=None, num_pairs=2,
    )
    assert num_levels_a == 4  # 2 * num_pairs

    # Deterministic in the seed: both per-pair level-0 masks reproduce.
    for pair in range(2):
        assert torch.equal(
            base_a[f"coupled_grpo_level0_mask_pair{pair}"],
            base_b[f"coupled_grpo_level0_mask_pair{pair}"],
        )
    # Pair 0 alias.
    assert torch.equal(
        base_a["coupled_grpo_level0_mask"], base_a["coupled_grpo_level0_mask_pair0"]
    )
    # Independent pairs: at least one row's masks differ between pair 0 and pair 1.
    assert not torch.equal(
        base_a["coupled_grpo_level0_mask_pair0"],
        base_a["coupled_grpo_level0_mask_pair1"],
    )

    # Per-pair partition: within each pair, levels {2p, 2p+1} harvest disjoint sets
    # whose union is exactly the valid response tokens.
    valid = base_a["coupled_grpo_valid_mask"] > 0.5
    for pair in range(2):
        h_even = make_coupled_level_view(
            base_a, 2 * pair, ("diffu_grpo_score_mask",)
        )["block_reveal_harvest_mask"] > 0.5
        h_odd = make_coupled_level_view(
            base_a, 2 * pair + 1, ("diffu_grpo_score_mask",)
        )["block_reveal_harvest_mask"] > 0.5
        assert not bool((h_even & h_odd).any())      # disjoint
        assert torch.equal(h_even | h_odd, valid)    # cover all valid tokens


def test_build_coupled_base_single_pair_matches_default():
    # num_pairs=1 must reproduce the default (no-arg) 2-level base byte-for-byte:
    # same level count and identical per-level harvest masks. Locks the CoupledGRPO
    # parity of the K-generalized path. Skips on login.
    pytest.importorskip("nemo_rl.algorithms.coupled_grpo_logprobs")
    from nemo_rl.algorithms.coupled_grpo_logprobs import (
        build_coupled_base,
        make_coupled_level_view,
    )

    mask_token_id, pad_token_id = 200, 0
    layout = [(2, 4), (3, 5), (1, 3)]
    base_def, _, num_def = build_coupled_base(
        _make_coupled_data(layout), mask_token_id, pad_token_id,
        noisy_block_size=None,
    )
    base_1, _, num_1 = build_coupled_base(
        _make_coupled_data(layout), mask_token_id, pad_token_id,
        noisy_block_size=None, num_pairs=1,
    )
    assert num_def == num_1 == 2
    for level in range(2):
        hd = make_coupled_level_view(base_def, level, ("diffu_grpo_score_mask",))[
            "block_reveal_harvest_mask"
        ]
        h1 = make_coupled_level_view(base_1, level, ("diffu_grpo_score_mask",))[
            "block_reveal_harvest_mask"
        ]
        assert torch.equal(hd, h1)


def test_espo_reveal_schedule_four_levels_yields_sample_major_microbatches():
    # ESPORevealSchedule with num_levels=4 (K=2 pairs) over N samples with K_seq
    # yields N/K_seq microbatches of K_seq*4 sample-major rows, levels [0,1,2,3] per
    # group, group sharing one sample index. Skips on login.
    pytest.importorskip("nemo_rl.algorithms.coupled_grpo_logprobs")
    worker_mod = pytest.importorskip(
        "nemo_rl.models.policy.workers.espo_megatron_policy_worker"
    )
    from nemo_rl.algorithms.coupled_grpo_logprobs import build_coupled_base

    ESPORevealSchedule = worker_mod.ESPORevealSchedule
    mask_token_id, pad_token_id = 200, 0
    layout = [(2, 4), (3, 5), (1, 3), (2, 2)]  # N = 4
    n = len(layout)
    base, num_samples, num_levels = build_coupled_base(
        _make_coupled_data(layout), mask_token_id, pad_token_id,
        noisy_block_size=None, num_pairs=2,
    )
    assert num_samples == n and num_levels == 4

    K_seq, M = 2, num_levels
    harvest_keys = ("diffu_grpo_score_mask", "diffu_grpo_loss_mask")
    schedule = ESPORevealSchedule(base).configure(
        num_levels=M, harvest_keys=harvest_keys, num_samples_per_micro_batch=K_seq
    )
    assert schedule.size == n * M
    mbs = list(schedule.make_microbatch_iterator(K_seq * M))
    assert len(mbs) == n // K_seq  # 2 microbatches
    for mb in mbs:
        assert mb["input_ids"].shape[0] == K_seq * M  # 8 rows
        levels_r = mb["coupled_grpo_level"].reshape(K_seq, M)
        sidx_r = mb["block_reveal_sample_index"].reshape(K_seq, M)
        torch.testing.assert_close(
            levels_r, torch.arange(M).unsqueeze(0).expand(K_seq, M)
        )
        assert bool((sidx_r == sidx_r[:, :1]).all())


def _espo_pair_sum(level_lps, level_harvests, pair):
    # Per-pair prev/ref summing (Approach A): sum WITHIN a pair (levels 2p, 2p+1).
    # Each level's per-token logprobs are nonzero only at its harvested positions,
    # so the pair sum carries each of the pair's tokens' logprob once.
    even = level_lps[2 * pair] * level_harvests[2 * pair]
    odd = level_lps[2 * pair + 1] * level_harvests[2 * pair + 1]
    return even + odd


def test_per_pair_sum_recovers_each_level_over_global_sum_overcounts():
    # THE CRUX (Approach A over-counting regression). K=2 pairs, one sequence, L=5,
    # block_size 2. Each level j has its own per-token logprobs (as from distinct
    # forwards); level j is nonzero only on its harvested set. Reducing level j
    # against PAIR (j//2)'s within-pair sum recovers level j's ELBO exactly, because
    # the pair sum equals level j's logprob on level j's harvested positions (the
    # OTHER level of the pair is zero there). Reducing against the GLOBAL 4-level sum
    # DOUBLES (K=2) each token: on level 0's positions the global sum also carries
    # pair 1's logprob at those same positions, corrupting the ELBO.
    def _mask_lp(positions):
        lp = torch.zeros((1, NOISY_LEN))
        h = torch.zeros((1, NOISY_LEN))
        for rel, val in positions.items():
            lp[0, NOISY_OFFSET + rel] = val
            h[0, NOISY_OFFSET + rel] = 1.0
        return lp, h

    # pair 0: M0 = {rel0, rel2, rel3}, complement {rel1, rel4}
    lp00, h00 = _mask_lp({0: -1.0, 2: -0.5, 3: -1.5})
    lp01, h01 = _mask_lp({1: -2.0, 4: -1.0})
    # pair 1: M1 = {rel1, rel2}, complement {rel0, rel3, rel4}
    lp10, h10 = _mask_lp({1: -3.0, 2: -0.5})
    lp11, h11 = _mask_lp({0: -2.0, 3: -1.5, 4: -4.0})

    level_lps = [lp00, lp01, lp10, lp11]
    level_harvests = [h00, h01, h10, h11]
    response_lengths = torch.tensor([5], dtype=torch.long)

    # Ground-truth per-level ELBOs (reduce each level's own logprobs with its own
    # harvest mask).
    per_level_elbo = [
        compute_block_aware_elbo(
            level_lps[j], level_harvests[j], response_lengths, NOISY_OFFSET, BLOCK_SIZE
        )
        for j in range(4)
    ]

    # Per-pair sums (what the ESPO worker returns as prev/ref) and the global sum
    # (the over-counting bug: summing ALL 2K levels).
    pair_sums = [_espo_pair_sum(level_lps, level_harvests, p) for p in range(2)]
    global_sum = sum(level_lps[j] * level_harvests[j] for j in range(4))

    for j in range(4):
        pair = j // 2
        # Approach A: reduce level j against its pair's sum -> exact recovery.
        got = compute_block_aware_elbo(
            pair_sums[pair], level_harvests[j], response_lengths,
            NOISY_OFFSET, BLOCK_SIZE,
        )
        torch.testing.assert_close(got, per_level_elbo[j])
        # Bug: reduce level j against the global 2K-sum -> over-counts (the other
        # pair's logprob at level j's positions leaks in).
        bug = compute_block_aware_elbo(
            global_sum, level_harvests[j], response_lengths,
            NOISY_OFFSET, BLOCK_SIZE,
        )
        assert not torch.allclose(bug, per_level_elbo[j])
