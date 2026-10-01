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
"""Pure-torch tests for single_controller_utils.segment_stats (no trainer stack)."""

from __future__ import annotations

import math

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.segment_stats import (
    accumulate_segment_stats,
    new_segment_stats_accumulator,
    reduce_segment_stats,
    resolve_row_identity,
)

# ── resolve_row_identity ───────────────────────────────────────────────────


def test_identity_from_sample_id_grammar_without_tags():
    ident = resolve_row_identity(["g_g0", "g_g0_t1", "g_g1", "h_g0"])
    assert ident.group_pos.tolist() == [0, 0, 0, 1]
    assert ident.rollout_local_idx.tolist() == [0, 0, 1, 0]
    assert ident.trace_in_rollout_idx.tolist() == [0, 1, 0, 0]
    assert ident.trace_kind == ["terminal"] * 4
    assert ident.rollout_key.tolist() == [0, 0, 1, 2]
    assert ident.is_canonical.tolist() == [True, False, True, True]
    assert ident.canonical_index.tolist() == [0, 2, 3]
    assert ident.row_to_canonical.tolist() == [0, 0, 1, 2]
    assert ident.has_extra_rows and ident.orphan_rows == 0
    assert ident.num_rows == 4 and ident.num_rollouts == 3


def test_identity_prefers_tags_and_keeps_group_from_id():
    tags = [
        {"rollout_local_idx": 3, "trace_in_rollout_idx": 0, "trace_kind": "terminal"},
        {
            "rollout_local_idx": 3,
            "trace_in_rollout_idx": 2,
            "trace_kind": "compaction_segment",
        },
        {"rollout_local_idx": 0, "trace_in_rollout_idx": 0, "trace_kind": "placeholder"},
    ]
    ident = resolve_row_identity(["g_g3", "g_g3_t2", "g_g0"], tags)
    assert ident.rollout_local_idx.tolist() == [3, 3, 0]
    assert ident.trace_in_rollout_idx.tolist() == [0, 2, 0]
    assert ident.trace_kind == ["terminal", "compaction_segment", "placeholder"]
    assert ident.rollout_key.tolist() == [0, 0, 1]
    assert ident.row_to_canonical.tolist() == [0, 0, 1]
    # Tag values are read as ints even when they arrive as strings/floats;
    # missing or junk entries fall back to the id.
    ident = resolve_row_identity(
        ["g_g1_t1"], [{"rollout_local_idx": "1", "trace_in_rollout_idx": 1.0}]
    )
    assert ident.rollout_local_idx.tolist() == [1]
    assert ident.trace_in_rollout_idx.tolist() == [1]
    ident = resolve_row_identity(["g_g1_t1"], [{"trace_in_rollout_idx": "x"}])
    assert ident.trace_in_rollout_idx.tolist() == [1]
    ident = resolve_row_identity(["g_g1"], [None])
    assert ident.is_canonical.tolist() == [True]


def test_identity_treats_legacy_ids_as_their_own_canonical_rollouts():
    ident = resolve_row_identity(["sample-0", "sample-1", "sample-2"])
    assert ident.group_pos.tolist() == [0, 1, 2]
    assert ident.rollout_key.tolist() == [0, 1, 2]
    assert ident.is_canonical.all() and not ident.has_extra_rows
    assert ident.row_to_canonical.tolist() == [0, 1, 2]


def test_identity_group_order_is_first_seen_and_empty_is_fine():
    ident = resolve_row_identity(["b_g0", "a_g0", "b_g1"])
    assert ident.group_pos.tolist() == [0, 1, 0]
    empty = resolve_row_identity([])
    assert empty.num_rows == 0 and empty.num_rollouts == 0
    assert not empty.has_extra_rows and empty.row_to_canonical.numel() == 0


def test_identity_orphan_and_duplicate_canonical():
    orphan = resolve_row_identity(["g_g0_t1", "g_g1"])
    assert orphan.row_to_canonical.tolist() == [-1, 0]
    assert orphan.orphan_rows == 1 and orphan.has_extra_rows
    with pytest.raises(ValueError, match="more than one canonical row"):
        resolve_row_identity(["g_g0", "g_g0"])
    with pytest.raises(ValueError, match="align 1:1"):
        resolve_row_identity(["g_g0"], [])


# ── accumulate / reduce ────────────────────────────────────────────────────


def _two_rollout_chunk():
    """Rows: A canonical, A compaction segment (gated), B canonical, C placeholder."""
    sample_ids = ["g1_g0", "g1_g0_t1", "g1_g1", "g2_g0"]
    tags = [
        {"rollout_local_idx": 0, "trace_in_rollout_idx": 0, "trace_kind": "terminal"},
        {
            "rollout_local_idx": 0,
            "trace_in_rollout_idx": 1,
            "trace_kind": "compaction_segment",
        },
        {"rollout_local_idx": 1, "trace_in_rollout_idx": 0, "trace_kind": "terminal"},
        {"rollout_local_idx": 0, "trace_in_rollout_idx": 0, "trace_kind": "placeholder"},
    ]
    ident = resolve_row_identity(sample_ids, tags)
    token_mask = torch.zeros(4, 5)
    token_mask[0, 1:4] = 1.0  # 3 tokens
    token_mask[1, 1:5] = 1.0  # 4 tokens
    token_mask[2, 1:3] = 1.0  # 2 tokens
    prev = torch.zeros(4, 5)
    gen = torch.zeros(4, 5)
    gen[0, 1:4] = -0.1
    gen[1, 1:5] = -1.0
    return ident, dict(
        rollout_key=ident.rollout_key,
        trace_kind=ident.trace_kind,
        final_sample_mask=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        sample_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        token_mask=token_mask,
        generation_logprobs=gen,
        prev_logprobs=prev,
        pre_gate_sample_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        is_canonical=ident.is_canonical,
    )


def test_segment_metrics_counts_kinds_and_by_kind_errors():
    _, kwargs = _two_rollout_chunk()
    acc = new_segment_stats_accumulator()
    accumulate_segment_stats(acc, **kwargs)
    out = reduce_segment_stats(acc)

    assert out["segments/rows"] == 4
    assert out["segments/canonical_rows"] == 3
    assert out["segments/extra_rows"] == 1
    assert out["segments/rollouts"] == 3
    assert out["segments/rollouts_with_segments"] == 1
    assert math.isclose(out["segments/rows_per_rollout_mean"], 4 / 3, rel_tol=1e-6)
    assert out["segments/rows_per_rollout_max"] == 2

    assert out["segments/rows_by_kind/terminal"] == 2
    assert out["segments/rows_by_kind/compaction_segment"] == 1
    assert out["segments/rows_by_kind/placeholder"] == 1
    assert out["segments/trained_rows_by_kind/terminal"] == 2
    assert out["segments/trained_rows_by_kind/compaction_segment"] == 0
    assert out["segments/trained_rows_by_kind/placeholder"] == 0
    assert out["segments/trainable_tokens_by_kind/terminal"] == 5
    assert out["segments/trainable_tokens_by_kind/compaction_segment"] == 0
    assert out["segments/trainable_tokens_by_kind/placeholder"] == 0
    # Rollout A compacted (it has a compaction_segment row): its trainable
    # tokens are the 3 of its terminal row; B's 2 tokens are uncompacted.
    assert math.isclose(
        out["segments/trainable_tokens_compacted_fraction"], 3 / 5, rel_tol=1e-6
    )
    assert out["segments/masked_row_frac_by_kind/terminal"] == 0.0
    assert out["segments/masked_row_frac_by_kind/compaction_segment"] == 1.0
    assert "segments/masked_row_frac_by_kind/placeholder" not in out

    terminal_err = (3 * math.exp(0.1) + 2 * math.exp(0.0)) / 5
    assert math.isclose(
        out["token_mult_prob_error/by_kind/terminal"], terminal_err, rel_tol=1e-6
    )
    # Gated row: no post-gate tokens -> bucket omitted; premask keeps it.
    assert "token_mult_prob_error/by_kind/compaction_segment" not in out
    assert "token_mult_prob_error/by_kind/placeholder" not in out
    assert math.isclose(
        out["token_mult_prob_error_premask/by_kind/terminal"],
        terminal_err,
        rel_tol=1e-6,
    )
    assert math.isclose(
        out["token_mult_prob_error_premask/by_kind/compaction_segment"],
        math.exp(1.0),
        rel_tol=1e-6,
    )
    assert "token_mult_prob_error_premask/by_kind/placeholder" not in out
    assert all(math.isfinite(v) for v in out.values())


def test_segment_metrics_omit_nonfinite_buckets_and_skip_errors_without_logprobs():
    _, kwargs = _two_rollout_chunk()
    kwargs["generation_logprobs"][2, 1] = float("inf")
    acc = new_segment_stats_accumulator()
    accumulate_segment_stats(acc, **kwargs)
    out = reduce_segment_stats(acc)
    assert "token_mult_prob_error/by_kind/terminal" not in out
    assert "token_mult_prob_error_premask/by_kind/terminal" not in out
    assert "token_mult_prob_error_premask/by_kind/compaction_segment" in out
    assert out["segments/rows"] == 4

    _, kwargs = _two_rollout_chunk()
    kwargs["generation_logprobs"] = None
    kwargs["prev_logprobs"] = None
    acc = new_segment_stats_accumulator()
    accumulate_segment_stats(acc, **kwargs)
    out = reduce_segment_stats(acc)
    assert not any(k.startswith("token_mult_prob_error") for k in out)
    assert out["segments/trainable_tokens_by_kind/terminal"] == 5


def test_segment_metrics_concatenate_chunks_with_rollout_offsets():
    _, kwargs = _two_rollout_chunk()
    acc = new_segment_stats_accumulator()
    accumulate_segment_stats(acc, **kwargs)
    accumulate_segment_stats(acc, **kwargs)
    out = reduce_segment_stats(acc)
    assert out["segments/rows"] == 8
    assert out["segments/rollouts"] == 6  # chunk-local keys were offset
    assert out["segments/rollouts_with_segments"] == 2
    assert out["segments/rows_per_rollout_max"] == 2
    assert math.isclose(
        out["segments/trainable_tokens_compacted_fraction"], 6 / 10, rel_tol=1e-6
    )


def test_segment_metrics_default_canonical_and_pre_gate_and_empty():
    _, kwargs = _two_rollout_chunk()
    kwargs.pop("is_canonical")
    kwargs.pop("pre_gate_sample_mask")
    acc = new_segment_stats_accumulator()
    accumulate_segment_stats(acc, **kwargs)
    out = reduce_segment_stats(acc)
    # First row of each rollout is canonical; premask defaults to the final mask.
    assert out["segments/canonical_rows"] == 3
    assert "token_mult_prob_error_premask/by_kind/compaction_segment" not in out
    assert math.isclose(
        out["token_mult_prob_error_premask/by_kind/terminal"],
        out["token_mult_prob_error/by_kind/terminal"],
        rel_tol=1e-9,
    )

    assert reduce_segment_stats(new_segment_stats_accumulator()) == {}
    acc = new_segment_stats_accumulator()
    accumulate_segment_stats(
        acc,
        rollout_key=torch.zeros(0, dtype=torch.long),
        trace_kind=[],
        final_sample_mask=torch.zeros(0),
        sample_mask=torch.zeros(0),
        token_mask=torch.zeros(0, 4),
    )
    assert reduce_segment_stats(acc) == {}


def test_segment_metrics_single_row_rollouts_are_the_degenerate_case():
    ident = resolve_row_identity(["sample-0", "sample-1"])
    acc = new_segment_stats_accumulator()
    token_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.float32)
    accumulate_segment_stats(
        acc,
        rollout_key=ident.rollout_key,
        trace_kind=ident.trace_kind,
        final_sample_mask=torch.ones(2),
        sample_mask=torch.ones(2),
        token_mask=token_mask,
        generation_logprobs=torch.zeros(2, 4),
        prev_logprobs=torch.zeros(2, 4),
        pre_gate_sample_mask=torch.ones(2),
        is_canonical=ident.is_canonical,
    )
    out = reduce_segment_stats(acc)
    assert out["segments/rows"] == out["segments/canonical_rows"] == 2
    assert out["segments/extra_rows"] == 0
    assert out["segments/rollouts_with_segments"] == 0
    assert out["segments/rows_per_rollout_mean"] == 1.0
    assert out["segments/trainable_tokens_compacted_fraction"] == 0.0
    assert out["token_mult_prob_error/by_kind/terminal"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="trace_kind"):
        accumulate_segment_stats(
            acc,
            rollout_key=ident.rollout_key,
            trace_kind=["terminal"],
            final_sample_mask=torch.ones(2),
            sample_mask=torch.ones(2),
            token_mask=token_mask,
        )


# ── rollout vote / rows_in_rollout / chunk completeness ────────────────────


def test_rollout_vote_mask_is_amax_over_the_rollout_rows():
    from nemo_rl.algorithms.single_controller_utils.segment_stats import (
        rollout_vote_mask,
    )

    # Rollout A: canonical gated (0), extra passes (1) -> votes.
    # Rollout B: both rows masked -> does not vote.
    # Rollout C: one row, valid with loss weight 0.25 -> votes with 0.25.
    ident = resolve_row_identity(["g_g0", "g_g0_t1", "g_g1", "g_g1_t1", "g_g2"])
    baseline_mask = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.25])
    votes = rollout_vote_mask(baseline_mask, ident.rollout_key, ident.canonical_index)
    assert votes.tolist() == [1.0, 0.0, 0.25]
    assert votes.dtype == baseline_mask.dtype
    # Row order does not matter: extras before their canonical row.
    ident2 = resolve_row_identity(["g_g0_t1", "g_g0", "g_g1"])
    votes2 = rollout_vote_mask(
        torch.tensor([1.0, 0.0, 0.0]), ident2.rollout_key, ident2.canonical_index
    )
    assert votes2.tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="align"):
        rollout_vote_mask(torch.ones(2), ident.rollout_key, ident.canonical_index)
    empty = resolve_row_identity([])
    assert rollout_vote_mask(torch.zeros(0), empty.rollout_key, empty.canonical_index).numel() == 0


def test_rollout_vote_mask_is_bit_identical_to_canonical_mask_without_extras():
    from nemo_rl.algorithms.single_controller_utils.segment_stats import (
        rollout_vote_mask,
    )

    torch.manual_seed(0)
    ids = [f"g{g}_g{i}" for g in range(5) for i in range(16)]
    ident = resolve_row_identity(ids)
    assert not ident.has_extra_rows
    for _ in range(20):
        baseline_mask = (torch.rand(len(ids)) > 0.3).float() * torch.rand(len(ids))
        votes = rollout_vote_mask(baseline_mask, ident.rollout_key, ident.canonical_index)
        assert torch.equal(votes, baseline_mask.index_select(0, ident.canonical_index))


def test_identity_reads_rows_in_rollout_tag_and_reports_mismatches():
    tags = [
        {"rollout_local_idx": 0, "trace_in_rollout_idx": 0, "rows_in_rollout": 2},
        {"rollout_local_idx": 0, "trace_in_rollout_idx": 1, "rows_in_rollout": 2},
        {"rollout_local_idx": 1, "trace_in_rollout_idx": 0, "rows_in_rollout": "1"},
        {"rollout_local_idx": 2, "trace_in_rollout_idx": 0},
    ]
    ident = resolve_row_identity(["g_g0", "g_g0_t1", "g_g1", "g_g2"], tags)
    assert ident.rows_in_rollout.tolist() == [2, 2, 1, -1]
    assert ident.rows_per_rollout.tolist() == [2, 1, 1]
    assert ident.num_groups == 1
    assert ident.rows_in_rollout_mismatches() == []
    # Untagged rows (legacy checkpoints) are never checked.
    assert resolve_row_identity(["g_g0", "g_g0_t1"]).rows_in_rollout.tolist() == [-1, -1]
    assert resolve_row_identity(["g_g0", "g_g0_t1"]).rows_in_rollout_mismatches() == []
    # A missing row: rollout 0 declares 3 rows but only 2 arrived.
    tags[0]["rows_in_rollout"] = 3
    tags[1]["rows_in_rollout"] = 3
    missing = resolve_row_identity(["g_g0", "g_g0_t1", "g_g1", "g_g2"], tags)
    problems = missing.rows_in_rollout_mismatches()
    assert len(problems) == 1 and "declares rows_in_rollout=3" in problems[0]
    assert "holds 2 row(s)" in problems[0]
    # Conflicting tags inside one rollout.
    tags[1]["rows_in_rollout"] = 2
    conflicting = resolve_row_identity(["g_g0", "g_g0_t1", "g_g1", "g_g2"], tags)
    problems = conflicting.rows_in_rollout_mismatches()
    assert len(problems) == 1 and "conflicting rows_in_rollout tags [2, 3]" in problems[0]
    # Junk / non-positive values count as untagged.
    junk = resolve_row_identity(
        ["g_g0", "g_g1"], [{"rows_in_rollout": 0}, {"rows_in_rollout": "x"}]
    )
    assert junk.rows_in_rollout.tolist() == [-1, -1]


def test_check_chunk_completeness_groups_times_n_and_whole_rollouts():
    from nemo_rl.algorithms.single_controller_utils.segment_stats import (
        check_chunk_completeness,
    )

    def tag(i, j, n):
        return {"rollout_local_idx": i, "trace_in_rollout_idx": j, "rows_in_rollout": n}

    # Two groups x N=2, one rollout with an extra row: complete.
    ids = ["a_g0", "a_g1", "b_g0", "b_g1", "a_g1_t1"]
    tags = [tag(0, 0, 1), tag(1, 0, 2), tag(0, 0, 1), tag(1, 0, 1), tag(1, 1, 2)]
    check_chunk_completeness(resolve_row_identity(ids, tags), num_generations_per_prompt=2)
    # Empty chunk is fine; N must be positive.
    check_chunk_completeness(resolve_row_identity([]), num_generations_per_prompt=2)
    with pytest.raises(ValueError, match="num_generations_per_prompt"):
        check_chunk_completeness(resolve_row_identity(ids, tags), num_generations_per_prompt=0)
    # A group missing a rollout (3 rollouts over 2 groups x 2).
    partial = resolve_row_identity(["a_g0", "a_g1", "b_g0"])
    with pytest.raises(RuntimeError, match="not made of whole groups") as info:
        check_chunk_completeness(partial, num_generations_per_prompt=2)
    assert "3 distinct rollout(s) over 2 group(s)" in str(info.value)
    # A group with an extra rollout (N too small for what arrived).
    with pytest.raises(RuntimeError, match="expected 2"):
        check_chunk_completeness(resolve_row_identity(ids, tags), num_generations_per_prompt=1)
    # Right rollout count, but a rollout's rows were split across chunks.
    split_tags = [tag(0, 0, 1), tag(1, 0, 3), tag(0, 0, 1), tag(1, 0, 1), tag(1, 1, 3)]
    with pytest.raises(RuntimeError, match="does not hold every row") as info:
        check_chunk_completeness(
            resolve_row_identity(ids, split_tags), num_generations_per_prompt=2
        )
    assert "declares rows_in_rollout=3" in str(info.value)
    # Legacy rows without the tag pass the rollout check.
    check_chunk_completeness(resolve_row_identity(ids), num_generations_per_prompt=2)


def test_segment_metrics_bucket_any_kind_including_compaction_summary():
    """By-kind metrics bucket whatever kind the finalizer stamped (no fixed list)."""
    sample_ids = ["g_g0", "g_g0_t1", "g_g0_t2", "g_g0_t3"]
    tags = [
        {"rollout_local_idx": 0, "trace_in_rollout_idx": 0, "trace_kind": "terminal"},
        {
            "rollout_local_idx": 0,
            "trace_in_rollout_idx": 1,
            "trace_kind": "compaction_segment",
        },
        {
            "rollout_local_idx": 0,
            "trace_in_rollout_idx": 2,
            "trace_kind": "compaction_summary",
        },
        {"rollout_local_idx": 0, "trace_in_rollout_idx": 3, "trace_kind": "new_kind"},
    ]
    ident = resolve_row_identity(sample_ids, tags)
    assert ident.trace_kind == [
        "terminal",
        "compaction_segment",
        "compaction_summary",
        "new_kind",
    ]
    token_mask = torch.zeros(4, 4)
    token_mask[:, 1:] = 1.0
    acc = new_segment_stats_accumulator()
    accumulate_segment_stats(
        acc,
        rollout_key=ident.rollout_key,
        trace_kind=ident.trace_kind,
        final_sample_mask=torch.tensor([1.0, 1.0, 0.0, 1.0]),
        sample_mask=torch.ones(4),
        token_mask=token_mask,
        generation_logprobs=torch.full((4, 4), -0.5),
        prev_logprobs=torch.zeros(4, 4),
        pre_gate_sample_mask=torch.ones(4),  # the summary row was gate-masked
        is_canonical=ident.is_canonical,
    )
    out = reduce_segment_stats(acc)
    for kind in ("terminal", "compaction_segment", "compaction_summary", "new_kind"):
        assert out[f"segments/rows_by_kind/{kind}"] == 1
        assert out[f"segments/trainable_tokens_by_kind/{kind}"] == (
            0 if kind == "compaction_summary" else 3
        )
        assert math.isclose(
            out[f"token_mult_prob_error_premask/by_kind/{kind}"],
            math.exp(0.5),
            rel_tol=1e-6,
        )
    assert "token_mult_prob_error/by_kind/compaction_summary" not in out
    assert out["segments/masked_row_frac_by_kind/compaction_summary"] == 1.0
    assert out["segments/rollouts_with_segments"] == 1
    assert out["segments/rows_per_rollout_max"] == 4
    # The rollout has a compaction_segment row, so every trained token is
    # "compacted" (summary row included had it trained).
    assert out["segments/trainable_tokens_compacted_fraction"] == 1.0
