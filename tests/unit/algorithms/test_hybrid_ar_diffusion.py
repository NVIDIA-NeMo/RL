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

"""Unit tests for the hybrid AR + diffusion batch builder.

Covers the pure-CPU layout/masking logic (no Megatron/Ray): the noisy half is
randomly masked only on real response tokens, the clean half carries next-token
targets for the RL term, the two halves never overlap, and the clean-side
alignment of advantages / prev_logprobs matches the token each position
predicts.
"""

import pytest
import torch

from nemo_rl.algorithms.diffu_grpo_logprobs import build_fully_masked_completion_batch
from nemo_rl.algorithms.hybrid_ar_diffusion import (
    HybridARDiffusionLogprobEstimationConfig,
    build_hybrid_ar_diffusion_batch,
    draw_hybrid_noisy_mask,
    get_hybrid_ar_diffusion_cfg,
    maybe_set_hybrid_mask_seed,
    unscatter_clean_aligned,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

MASK_TOKEN_ID = 200
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 300
RATIO_MIN = 0.2
RATIO_MAX = 0.8

# (prompt_len, response_len) per sample.
LAYOUT = [(2, 4), (3, 5), (1, 3)]
SEQ_LEN = 12


def _make_data(seed_offset: int = 0, with_rl_fields: bool = True) -> BatchedDataDict:
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
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(n, dtype=torch.float32),
            "hybrid_mask_seed": torch.arange(n, dtype=torch.long) + seed_offset,
        }
    )
    if with_rl_fields:
        # Distinct per-token values so mis-alignment by even one position shows.
        data["advantages"] = torch.arange(SEQ_LEN, dtype=torch.float32).unsqueeze(
            0
        ).repeat(n, 1) + 100.0 * torch.arange(n, dtype=torch.float32).unsqueeze(1)
        data["prev_logprobs"] = -(
            torch.arange(SEQ_LEN, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
        )
    return data


def _build(
    data: BatchedDataDict,
    block_size: int | None = None,
    noisy_tail_mode: str = "mask",
) -> BatchedDataDict:
    return build_hybrid_ar_diffusion_batch(
        data,
        mask_token_id=MASK_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        mask_ratio_min=RATIO_MIN,
        mask_ratio_max=RATIO_MAX,
        noisy_tail_mode=noisy_tail_mode,
        block_size=block_size,
        eos_token_id=EOS_TOKEN_ID,
    )


def _noisy_length(batch: BatchedDataDict) -> int:
    return int(batch["diffu_grpo_noisy_lengths"][0].item())


def _estimation_cfg(**overrides) -> dict:
    cfg = {
        "type": "hybrid_ar_diffusion",
        "mask_token_id": MASK_TOKEN_ID,
        "ce_loss_weight": 0.1,
    }
    cfg.update(overrides)
    return cfg


def test_ce_mask_only_on_real_response_tokens():
    batch = _build(_make_data())
    ce = batch["hybrid_ce_mask"] > 0.5
    noisy_len = _noisy_length(batch)
    # Never in the clean half.
    assert not bool(ce[:, noisy_len:].any())
    # Never past a sample's own response length (no block-padding tail).
    for s, (_, rlen) in enumerate(LAYOUT):
        assert not bool(ce[s, rlen:noisy_len].any())


def test_ce_and_pg_masks_are_disjoint_halves():
    batch = _build(_make_data())
    ce = batch["hybrid_ce_mask"] > 0.5
    pg = batch["hybrid_pg_mask"] > 0.5
    noisy_len = _noisy_length(batch)
    assert not bool((ce & pg).any())
    assert not bool(pg[:, :noisy_len].any())


def test_masked_positions_hold_mask_id_and_revealed_hold_truth():
    data = _make_data()
    batch = _build(data)
    ce = batch["hybrid_ce_mask"] > 0.5
    noisy_len = _noisy_length(batch)
    ids = batch["input_ids"]
    targets = batch["diffu_grpo_target_ids"]

    assert bool((ids[ce] == MASK_TOKEN_ID).all())
    for s, (_, rlen) in enumerate(LAYOUT):
        revealed = ~ce[s, :rlen]
        assert torch.equal(ids[s, :rlen][revealed], targets[s, :rlen][revealed])
    # The clean half is never masked -- it is verbatim teacher forcing.
    assert not bool((ids[:, noisy_len:] == MASK_TOKEN_ID).any())


def test_noisy_targets_are_same_position_response_tokens():
    data = _make_data()
    batch = _build(data)
    hybrid_targets = batch["hybrid_target_ids"]
    for s, (plen, rlen) in enumerate(LAYOUT):
        expected = data["input_ids"][s, plen : plen + rlen]
        assert torch.equal(hybrid_targets[s, :rlen], expected)


def test_clean_targets_are_next_token_shifted():
    data = _make_data()
    batch = _build(data)
    noisy_len = _noisy_length(batch)
    hybrid_targets = batch["hybrid_target_ids"]
    clean_lengths = batch["diffu_grpo_clean_lengths"]
    for s in range(len(LAYOUT)):
        clean_len = int(clean_lengths[s].item())
        got = hybrid_targets[s, noisy_len : noisy_len + clean_len - 1]
        # Position noisy_len + j must predict original token j + 1.
        assert torch.equal(got, data["input_ids"][s, 1:clean_len])


def test_pg_mask_scores_every_response_token_and_no_prompt_token():
    data = _make_data()
    batch = _build(data)
    noisy_len = _noisy_length(batch)
    pg = batch["hybrid_pg_mask"] > 0.5
    for s, (plen, rlen) in enumerate(LAYOUT):
        # Positions predicting response tokens are noisy_len + (plen-1) .. +(plen+rlen-2).
        expected = torch.zeros(pg.shape[1], dtype=torch.bool)
        expected[noisy_len + plen - 1 : noisy_len + plen + rlen - 1] = True
        assert torch.equal(pg[s], expected)
        assert int(pg[s].sum().item()) == rlen


def test_advantages_and_prev_logprobs_align_with_predicted_token():
    data = _make_data()
    batch = _build(data)
    noisy_len = _noisy_length(batch)
    pg = batch["hybrid_pg_mask"] > 0.5
    clean_lengths = batch["diffu_grpo_clean_lengths"]
    for s in range(len(LAYOUT)):
        clean_len = int(clean_lengths[s].item())
        for j in range(clean_len - 1):
            pos = noisy_len + j
            if not bool(pg[s, pos]):
                continue
            # Position pos predicts token j+1, so it must carry that token's
            # advantage and prev_logprob.
            assert batch["advantages"][s, pos] == data["advantages"][s, j + 1]
            assert batch["prev_logprobs"][s, pos] == data["prev_logprobs"][s, j + 1]


def test_determinism_and_step_variation():
    a = _build(_make_data(seed_offset=0))["hybrid_ce_mask"]
    b = _build(_make_data(seed_offset=0))["hybrid_ce_mask"]
    assert torch.equal(a, b)
    c = _build(_make_data(seed_offset=7919))["hybrid_ce_mask"]
    assert not torch.equal(a, c)


def test_clean_half_is_independent_of_the_mask_seed():
    # This is the load-bearing invariant: it is why a fresh mask can be drawn
    # every pass without a shared seed. If anything the RL term reads were
    # mask-dependent, prev_logprobs would be invalidated whenever the mask is
    # redrawn and the importance ratio would compare two different
    # conditionals.
    a = _build(_make_data(seed_offset=0))
    b = _build(_make_data(seed_offset=7919))
    noisy_len = _noisy_length(a)
    assert not torch.equal(a["hybrid_ce_mask"], b["hybrid_ce_mask"]), (
        "masks must actually differ for this test to mean anything"
    )
    assert torch.equal(a["hybrid_pg_mask"], b["hybrid_pg_mask"])
    assert torch.equal(
        a["hybrid_target_ids"][:, noisy_len:], b["hybrid_target_ids"][:, noisy_len:]
    )
    assert torch.equal(a["input_ids"][:, noisy_len:], b["input_ids"][:, noisy_len:])
    assert torch.equal(a["advantages"], b["advantages"])
    assert torch.equal(a["prev_logprobs"], b["prev_logprobs"])


def test_clean_alignment_round_trips():
    # Forward (_scatter_clean_aligned, exercised via the builder) and inverse
    # (unscatter_clean_aligned, used to return prev_logprobs) must compose to
    # the identity on response positions. A one-off in either direction
    # silently shifts every prev_logprob by a token, which would corrupt the
    # importance ratio without ever raising.
    data = _make_data()
    batch = _build(data)
    recovered = unscatter_clean_aligned(
        batch["prev_logprobs"], batch, data["input_ids"].shape[1]
    )
    clean_lengths = batch["diffu_grpo_clean_lengths"]
    for s in range(len(LAYOUT)):
        clean_len = int(clean_lengths[s].item())
        # Positions 1..clean_len-1 are the ones the clean half can score.
        assert torch.equal(
            recovered[s, 1:clean_len], data["prev_logprobs"][s, 1:clean_len]
        )
        # Index 0 has no predecessor and is unused by the AR convention.
        assert recovered[s, 0] == 0.0


def test_round_trip_recovers_every_response_token():
    # The round-trip must specifically preserve the response positions, since
    # those are the ones the policy-gradient term reads.
    data = _make_data()
    batch = _build(data)
    recovered = unscatter_clean_aligned(
        batch["prev_logprobs"], batch, data["input_ids"].shape[1]
    )
    for s, (plen, rlen) in enumerate(LAYOUT):
        assert torch.equal(
            recovered[s, plen : plen + rlen],
            data["prev_logprobs"][s, plen : plen + rlen],
        )


def test_generation_logprobs_are_clean_aligned_when_present():
    data = _make_data()
    data["generation_logprobs"] = -2.0 * torch.arange(
        SEQ_LEN, dtype=torch.float32
    ).unsqueeze(0).repeat(len(LAYOUT), 1)
    batch = _build(data)
    assert "generation_logprobs" in batch
    noisy_len = _noisy_length(batch)
    clean_lengths = batch["diffu_grpo_clean_lengths"]
    for s in range(len(LAYOUT)):
        clean_len = int(clean_lengths[s].item())
        for j in range(clean_len - 1):
            assert (
                batch["generation_logprobs"][s, noisy_len + j]
                == data["generation_logprobs"][s, j + 1]
            )


def test_mask_ratio_within_bounds():
    batch = _build(_make_data())
    ratios = batch["hybrid_mask_ratio"]
    assert bool((ratios >= RATIO_MIN).all())
    assert bool((ratios <= RATIO_MAX).all())


def test_block_padding_tail_never_scored():
    # block_size 4 pads sample 1 (response 5) out to 8 noisy slots.
    batch = _build(_make_data(), block_size=4)
    ce = batch["hybrid_ce_mask"] > 0.5
    noisy_len = _noisy_length(batch)
    assert noisy_len == 8
    for s, (_, rlen) in enumerate(LAYOUT):
        assert not bool(ce[s, rlen:noisy_len].any())


def test_eos_tail_mode_fills_block_padding_with_eos():
    # Sample 0 (response 4) fills its block exactly; sample 2 (response 3) has a
    # one-token tail. With the eos mode that tail holds EOS instead of MASK.
    batch = _build(_make_data(), block_size=4, noisy_tail_mode="eos")
    assert batch["input_ids"][2, 3] == EOS_TOKEN_ID
    assert not bool((batch["hybrid_ce_mask"][2, 3:] > 0.5).any())


def test_none_tail_mode_skips_block_padding():
    batch = _build(_make_data(), block_size=4, noisy_tail_mode="none")
    assert _noisy_length(batch) == max(r for _, r in LAYOUT)


def test_eos_tail_mode_requires_eos_token_id():
    with pytest.raises(ValueError, match="eos_token_id"):
        build_fully_masked_completion_batch(
            _make_data(),
            mask_token_id=MASK_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            noisy_tail_mode="eos",
            block_size=4,
        )


def test_unknown_tail_mode_raises():
    with pytest.raises(ValueError, match="noisy_tail_mode"):
        _build(_make_data(), block_size=4, noisy_tail_mode="Mask")


def test_pg_token_count_matches_the_frameworks_global_valid_toks():
    # global_valid_toks is reduced from the ORIGINAL batch as
    # sum(token_mask[:, 1:] * sample_mask). Both loss terms are normalized by it,
    # so it must equal the number of PG positions the builder produces --
    # otherwise the loss scale is silently wrong.
    data = _make_data()
    batch = _build(data)
    framework_count = (
        data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    ).sum()
    assert batch["hybrid_pg_mask"].sum() == framework_count
    assert framework_count == sum(r for _, r in LAYOUT)


def test_missing_seed_raises():
    data = _make_data()
    del data.data["hybrid_mask_seed"]
    with pytest.raises(ValueError, match="hybrid_mask_seed"):
        _build(data)


def test_rl_fields_optional():
    batch = _build(_make_data(with_rl_fields=False))
    assert "advantages" not in batch
    assert "hybrid_ce_mask" in batch


@pytest.mark.parametrize("lo,hi", [(0.0, 0.8), (0.2, 1.0), (0.8, 0.2), (-0.1, 0.5)])
def test_invalid_ratio_bounds_raise(lo: float, hi: float):
    with pytest.raises(ValueError, match="mask_ratio"):
        build_hybrid_ar_diffusion_batch(
            _make_data(),
            mask_token_id=MASK_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            mask_ratio_min=lo,
            mask_ratio_max=hi,
            noisy_tail_mode="mask",
        )
    with pytest.raises(ValueError, match="mask_ratio"):
        HybridARDiffusionLogprobEstimationConfig.model_validate(
            _estimation_cfg(mask_ratio_min=lo, mask_ratio_max=hi)
        )


def test_draw_mask_respects_ratio_ordering():
    # A larger t must mask at least as much in expectation; check the extremes
    # are separated using a wide, deterministic sweep over many rows.
    n = 64
    layout_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).repeat(n, 1) + 1
    data = BatchedDataDict(
        {
            "input_ids": layout_ids,
            "input_lengths": torch.full((n,), 8, dtype=torch.long),
            "token_mask": torch.tensor([[0.0, 0.0, 1, 1, 1, 1, 1, 1]]).repeat(n, 1),
            "sample_mask": torch.ones(n, dtype=torch.float32),
            "hybrid_mask_seed": torch.arange(n, dtype=torch.long),
        }
    )
    base = build_fully_masked_completion_batch(
        data,
        mask_token_id=MASK_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        noisy_tail_mode="mask",
    )
    low, _, _ = draw_hybrid_noisy_mask(
        base, data["hybrid_mask_seed"], mask_ratio_min=0.05, mask_ratio_max=0.15
    )
    high, _, _ = draw_hybrid_noisy_mask(
        base, data["hybrid_mask_seed"], mask_ratio_min=0.85, mask_ratio_max=0.95
    )
    assert int(low.sum().item()) < int(high.sum().item())


def test_maybe_set_hybrid_mask_seed_gating():
    data = _make_data()
    del data.data["hybrid_mask_seed"]
    maybe_set_hybrid_mask_seed(data, {}, 3)
    assert "hybrid_mask_seed" not in data

    policy_cfg = {"logprob_estimation": _estimation_cfg(seed_base=5)}
    maybe_set_hybrid_mask_seed(data, policy_cfg, 3)
    assert "hybrid_mask_seed" in data
    # Distinct per row, and the step offset separates step blocks.
    seeds = data["hybrid_mask_seed"]
    assert len(set(seeds.tolist())) == len(LAYOUT)

    data2 = _make_data()
    maybe_set_hybrid_mask_seed(data2, policy_cfg, 4)
    assert not torch.equal(seeds, data2["hybrid_mask_seed"])


def test_get_cfg_validates_and_applies_defaults():
    with pytest.raises(ValueError, match="logprob_estimation must be set"):
        get_hybrid_ar_diffusion_cfg({})
    with pytest.raises(ValueError, match="hybrid_ar_diffusion"):
        get_hybrid_ar_diffusion_cfg(
            {"logprob_estimation": _estimation_cfg(type="coupled_grpo")}
        )
    with pytest.raises(ValueError, match="ce_loss_weight"):
        get_hybrid_ar_diffusion_cfg(
            {"logprob_estimation": _estimation_cfg(ce_loss_weight=-0.1)}
        )

    cfg = get_hybrid_ar_diffusion_cfg({"logprob_estimation": _estimation_cfg()})
    assert cfg.ce_loss_weight == pytest.approx(0.1)
    assert cfg.pg_loss_weight == pytest.approx(1.0)
    assert (cfg.mask_ratio_min, cfg.mask_ratio_max) == (0.2, 0.8)
    assert cfg.noisy_tail_mode == "mask"
    assert cfg.elbo_weight_ce is False
    # An already-validated config (as produced by MasterConfig) is accepted too.
    assert get_hybrid_ar_diffusion_cfg({"logprob_estimation": cfg}) == cfg
