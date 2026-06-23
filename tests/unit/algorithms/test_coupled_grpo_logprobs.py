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
"""Unit tests for the CoupledGRPO (random + complement) logprob estimator.

Covers the pure-CPU masking/scatter logic (no Megatron/Ray): every valid
response token is masked in exactly one of the two complementary levels, the
mask is deterministic given the per-row seed, and summing the two levels'
scattered logprobs reconstructs the full ``[N, S]`` vector.
"""

import torch

from nemo_rl.algorithms.block_just_grpo_logprobs import scatter_block_reveal_logprobs
from nemo_rl.algorithms.coupled_grpo_logprobs import (
    COUPLED_NUM_LEVELS,
    build_coupled_base,
    make_coupled_level_view,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

MASK_TOKEN_ID = 200
PAD_TOKEN_ID = 0

# (prompt_len, response_len) per sample.
LAYOUT = [(2, 4), (3, 5), (1, 3)]
SEQ_LEN = 12


def _make_data(seed_offset: int = 0) -> BatchedDataDict:
    n = len(LAYOUT)
    input_ids = torch.full((n, SEQ_LEN), PAD_TOKEN_ID, dtype=torch.long)
    token_mask = torch.zeros((n, SEQ_LEN), dtype=torch.float32)
    input_lengths = torch.zeros(n, dtype=torch.long)
    for s, (plen, rlen) in enumerate(LAYOUT):
        total = plen + rlen
        # Unique, non-special token ids so target/reveal checks are unambiguous.
        input_ids[s, :total] = torch.arange(total, dtype=torch.long) + 10 * (s + 1) + 1
        token_mask[s, plen:total] = 1.0
        input_lengths[s] = total
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(n, dtype=torch.float32),
            "coupled_grpo_seed": torch.arange(n, dtype=torch.long) + seed_offset,
        }
    )


def _harvest_bool(view: BatchedDataDict) -> torch.Tensor:
    return view["block_reveal_harvest_mask"] > 0.5


def test_two_levels_constant():
    data = _make_data()
    _, n, num_levels = build_coupled_base(
        data, MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=None
    )
    assert n == len(LAYOUT)
    assert num_levels == COUPLED_NUM_LEVELS == 2


def test_coverage_and_complementarity():
    data = _make_data()
    base, _, _ = build_coupled_base(
        data, MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=None
    )
    valid = base["coupled_grpo_valid_mask"] > 0.5
    v0 = make_coupled_level_view(base, 0, ("diffu_grpo_score_mask",))
    v1 = make_coupled_level_view(base, 1, ("diffu_grpo_score_mask",))
    h0, h1 = _harvest_bool(v0), _harvest_bool(v1)

    # Disjoint, and together cover exactly the valid response positions.
    assert not bool((h0 & h1).any())
    assert torch.equal(h0 | h1, valid)
    # Level 1 reveals exactly what level 0 masks, and vice-versa.
    assert torch.equal(h1, valid & (~h0))
    # Revealed tokens carry the true target ids; masked carry MASK.
    target = base["diffu_grpo_target_ids"]
    reveal0 = valid & (~h0)
    assert torch.equal(v0["input_ids"][reveal0], target[reveal0])
    assert bool((v0["input_ids"][h0] == MASK_TOKEN_ID).all())


def test_determinism_and_step_variation():
    base_a, _, _ = build_coupled_base(
        _make_data(seed_offset=0), MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=None
    )
    base_b, _, _ = build_coupled_base(
        _make_data(seed_offset=0), MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=None
    )
    # Same seed -> identical mask realization (prev/ref/train consistency).
    assert torch.equal(
        base_a["coupled_grpo_level0_mask"], base_b["coupled_grpo_level0_mask"]
    )
    base_c, _, _ = build_coupled_base(
        _make_data(seed_offset=997), MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=None
    )
    # Different seed (different step) -> mask changes somewhere.
    assert not torch.equal(
        base_a["coupled_grpo_level0_mask"], base_c["coupled_grpo_level0_mask"]
    )


def test_block_padding_never_harvested():
    # With block padding the noisy side has MASK positions past the response;
    # those must never be harvested (valid is response-only).
    data = _make_data()
    base, _, _ = build_coupled_base(
        data, MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=4
    )
    valid = base["coupled_grpo_valid_mask"] > 0.5
    h0 = _harvest_bool(make_coupled_level_view(base, 0, ("diffu_grpo_score_mask",)))
    h1 = _harvest_bool(make_coupled_level_view(base, 1, ("diffu_grpo_score_mask",)))
    assert torch.equal(h0 | h1, valid)
    # Number of valid positions equals total response tokens (no padding leak).
    assert int(valid.sum()) == sum(rlen for _, rlen in LAYOUT)


def test_scatter_roundtrip_sums_to_full_vector():
    data = _make_data()
    base, n, num_levels = build_coupled_base(
        data, MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=None
    )
    noisy_len = base["input_ids"].shape[1]
    completion_starts = base["diffu_grpo_completion_starts"]
    # Fabricated per-(row, noisy-pos) "logprobs"; identical array fed to both
    # levels. Each (sample, pos) is harvested by exactly one level, so the sum
    # places value (row*100 + q + 1) at original position start + q.
    flat = (
        torch.arange(noisy_len, dtype=torch.float32).unsqueeze(0)
        + 100.0 * torch.arange(n, dtype=torch.float32).unsqueeze(1)
        + 1.0
    )

    out = torch.zeros((n, SEQ_LEN), dtype=torch.float32)
    for level in range(num_levels):
        view = make_coupled_level_view(base, level, ("diffu_grpo_score_mask",))
        out = out + scatter_block_reveal_logprobs(
            flat_logprobs=flat,
            harvest_mask=view["block_reveal_harvest_mask"],
            sample_index=view["block_reveal_sample_index"],
            completion_starts=completion_starts,
            noisy_response_offset=0,
            original_seq_len=SEQ_LEN,
            num_samples=n,
        )

    expected = torch.zeros((n, SEQ_LEN), dtype=torch.float32)
    for s, (plen, rlen) in enumerate(LAYOUT):
        for k in range(rlen):
            expected[s, plen + k] = s * 100 + k + 1
    assert torch.allclose(out, expected), (out, expected)


def test_level0_mask_override_pins_M_without_seed():
    # Verification plumbing: a caller-provided level-0 mask pins M directly,
    # needs no seed, and level 0 harvests exactly the overridden positions.
    data = _make_data()
    n = len(LAYOUT)
    # Mask the first response token of each sample (original [N, S] coords).
    override = torch.zeros((n, SEQ_LEN), dtype=torch.float32)
    for s, (plen, rlen) in enumerate(LAYOUT):
        override[s, plen] = 1.0
    data_no_seed = BatchedDataDict(
        {k: v for k, v in data.items() if k != "coupled_grpo_seed"}
    )
    data_no_seed["coupled_grpo_level0_mask_override"] = override

    base, _, num_levels = build_coupled_base(
        data_no_seed, MASK_TOKEN_ID, PAD_TOKEN_ID, noisy_block_size=None
    )
    assert num_levels == COUPLED_NUM_LEVELS

    # First response token maps to noisy offset 0; level 0 harvests exactly it.
    h0 = _harvest_bool(make_coupled_level_view(base, 0, ("diffu_grpo_score_mask",)))
    expected = torch.zeros_like(h0)
    expected[:, 0] = True
    assert torch.equal(h0, expected)
    # Level 1 harvests the complement of the override over valid positions.
    valid = base["coupled_grpo_valid_mask"] > 0.5
    h1 = _harvest_bool(make_coupled_level_view(base, 1, ("diffu_grpo_score_mask",)))
    assert torch.equal(h1, valid & (~h0))
