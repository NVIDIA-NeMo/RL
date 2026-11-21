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
"""Tests for the driver-side shared-prefix planner."""

import random

import pytest

from nemo_rl.data.packing.shared_prefix.plan import (
    MAX_TREE_ROWS,
    SharedPrefixShard,
    plan_shared_prefix_shards,
)
from nemo_rl.data.packing.shared_prefix.row_metadata import UNSHARED_ROW


def _plan(
    lengths: list[int],
    groups: list[str],
    prompts: list[int],
    *,
    hashes: list[int] | None = None,
    num_shards: int = 1,
    batch_size: int | None = None,
    budget: int = 1024,
    pad_multiple: int = 1,
    bound_by_full_length: bool = False,
    largest_first: bool = False,
) -> list[SharedPrefixShard]:
    return plan_shared_prefix_shards(
        sequence_lengths=lengths,
        group_ids=groups,
        prompt_lengths=prompts,
        prompt_hashes=hashes if hashes is not None else [0] * len(lengths),
        num_shards=num_shards,
        batch_size=batch_size,
        max_tokens_per_microbatch=budget,
        sequence_length_pad_multiple=pad_multiple,
        algorithm="modified_first_fit_decreasing",
        bound_by_full_length=bound_by_full_length,
        largest_first=largest_first,
    )


def _forwards(
    shard: SharedPrefixShard, global_batch: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Rows and tree ids of each forward of one global batch."""
    offset = sum(shard.elem_counts_per_gb[:global_batch])
    return [
        (
            shard.row_indices[offset + start : offset + end],
            shard.tree_ids[offset + start : offset + end],
        )
        for start, end in shard.micro_batch_indices[global_batch]
    ]


def _kinds(shard: SharedPrefixShard, global_batch: int = 0) -> list[str]:
    """Kind of each forward of one global batch: a pack of trees or a dense bin."""
    kinds = []
    for _, tree_ids in _forwards(shard, global_batch):
        shared = {tree_id != UNSHARED_ROW for tree_id in tree_ids}
        assert len(shared) == 1, "a forward holds trees or unshared rows, never both"
        kinds.append("pack" if shared.pop() else "dense")
    return kinds


def test_rows_of_one_prompt_share_one_forward() -> None:
    (shard,) = _plan([100, 120, 140, 160], ["a"] * 4, [80] * 4)
    assert _forwards(shard) == [([0, 1, 2, 3], [0, 0, 0, 0])]
    # The prompt once, then the four completions.
    assert shard.micro_batch_lengths == [[80 + 20 + 40 + 60 + 80]]
    assert shard.elem_counts_per_gb == [4]


def test_padding_applies_to_each_row_and_to_the_forward() -> None:
    (shard,) = _plan([100, 120], ["a"] * 2, [70] * 2, pad_multiple=16)
    # Rows pad to 112 and 128; the tree is 70 + 42 + 58 = 170, and the forward pads to 176.
    assert shard.micro_batch_lengths == [[176]]


@pytest.mark.parametrize(
    "groups, prompts, hashes",
    [
        (["a", "b"], [80, 80], [0, 0]),
        (["a", "a"], [80, 80], [1, 2]),
        (["a", "a"], [80, 70], [0, 0]),
        (["a", "a"], [0, 0], [0, 0]),
        (["a", "a"], [100, 100], [0, 0]),
    ],
    ids=["groups", "hashes", "prompt-lengths", "no-prompt", "no-completion"],
)
def test_rows_that_differ_run_dense(
    groups: list[str], prompts: list[int], hashes: list[int]
) -> None:
    (shard,) = _plan([100, 100], groups, prompts, hashes=hashes)
    assert _forwards(shard) == [([0, 1], [UNSHARED_ROW, UNSHARED_ROW])]
    assert shard.micro_batch_lengths == [[200]]


def test_packs_and_dense_bins_are_separate_forwards() -> None:
    (shard,) = _plan([100, 100, 100, 50], ["a", "a", "b", "c"], [80, 80, 80, 0])
    assert _forwards(shard) == [
        ([0, 1], [0, 0]),
        ([2, 3], [UNSHARED_ROW, UNSHARED_ROW]),
    ]
    assert shard.micro_batch_lengths == [[120, 150]]


def test_rows_of_one_group_share_only_with_rows_of_the_same_prompt() -> None:
    # The row shape of forest training. Rows 0, 1 and 3 start with the group's prompt, row 2
    # starts with a rewritten prompt, and rows 4 and 5 are padding rows without a prompt.
    (shard,) = _plan(
        [6, 6, 6, 3, 6, 6],
        ["g"] * 6,
        [2, 2, 2, 2, 0, 0],
        hashes=[1, 1, 2, 1, 0, 0],
        budget=64,
    )
    assert _forwards(shard) == [
        ([0, 1, 3], [0, 0, 0]),
        ([2, 4, 5], [UNSHARED_ROW] * 3),
    ]
    assert shard.micro_batch_lengths == [[2 + 4 + 4 + 1, 18]]


def test_several_trees_share_a_forward() -> None:
    (shard,) = _plan([100] * 6, ["a", "a", "a", "b", "b", "b"], [80] * 6)
    ((rows, tree_ids),) = _forwards(shard)
    assert sorted(rows) == [0, 1, 2, 3, 4, 5]
    # Rows of one tree are contiguous, and the two trees have different ids.
    assert tree_ids == [0, 0, 0, 1, 1, 1]
    assert shard.micro_batch_lengths == [[2 * (80 + 3 * 20)]]


def test_a_group_larger_than_the_row_limit_splits_evenly() -> None:
    rows = MAX_TREE_ROWS + 1
    (shard,) = _plan([100] * rows, ["a"] * rows, [80] * rows, budget=10_000)
    ((_, tree_ids),) = _forwards(shard)
    assert sorted(tree_ids.count(tree) for tree in set(tree_ids)) == [8, 9]


def test_a_tree_larger_than_the_budget_splits() -> None:
    # Packed, four rows need 80 + 4 * 120 = 560 tokens; two rows need 320.
    (shard,) = _plan([200] * 4, ["a"] * 4, [80] * 4, budget=400)
    assert [tree_ids for _, tree_ids in _forwards(shard)] == [[0, 0], [0, 0]]
    assert shard.micro_batch_lengths == [[320, 320]]


def test_full_length_bound_counts_the_prompt_in_every_row() -> None:
    lengths, groups, prompts = [100] * 8, ["a"] * 4 + ["b"] * 4, [80] * 8
    (packed,) = _plan(lengths, groups, prompts, budget=400)
    (full,) = _plan(lengths, groups, prompts, budget=400, bound_by_full_length=True)
    # Packed, both trees (160 tokens each) fit one forward. In full they are 400 tokens each.
    assert packed.micro_batch_lengths == [[320]]
    assert full.micro_batch_lengths == [[160, 160]]


def _mixed_batch() -> tuple[list[int], list[str], list[int]]:
    """Four groups of four rows that share a prompt, and five rows that share nothing."""
    lengths = [100] * 16 + [300, 50, 60, 70, 80]
    groups = [group for group in "abcd" for _ in range(4)] + list("efghi")
    prompts = [80] * 16 + [0] * 5
    return lengths, groups, prompts


def _one_tree_per_rank() -> tuple[list[int], list[str], list[int]]:
    """64 groups of 16 rows, for 64 ranks, with no unshared row."""
    lengths = [100 + 10 * (row % 7) for row in range(64 * 16)]
    groups = [f"group-{row // 16}" for row in range(64 * 16)]
    return lengths, groups, [60] * (64 * 16)


def _mostly_unshared_batch() -> tuple[list[int], list[str], list[int]]:
    """Two groups of four rows that share a prompt, and 30 rows that share nothing."""
    lengths = [100] * 8 + [40 + row for row in range(30)]
    groups = ["a"] * 4 + ["b"] * 4 + [f"single-{row}" for row in range(30)]
    return lengths, groups, [80] * 8 + [0] * 30


@pytest.mark.parametrize(
    "batch, num_shards, batch_size",
    [
        (_mixed_batch(), 4, None),
        (_one_tree_per_rank(), 64, None),
        (_mostly_unshared_batch(), 8, None),
        # The first global batch only has trees; the second mixes trees and unshared rows.
        (([100] * 16, ["a"] * 8 + list("bbccdefg"), [80] * 12 + [0] * 4), 2, 8),
    ],
    ids=["mixed", "one-tree-per-rank", "mostly-unshared", "two-global-batches"],
)
def test_every_rank_runs_the_same_kind_at_every_forward_index(
    batch: tuple[list[int], list[str], list[int]],
    num_shards: int,
    batch_size: int | None,
) -> None:
    lengths, groups, prompts = batch
    shards = _plan(
        lengths,
        groups,
        prompts,
        num_shards=num_shards,
        batch_size=batch_size,
        budget=4096,
    )
    for global_batch in range(len(shards[0].elem_counts_per_gb)):
        kinds = _kinds(shards[0], global_batch)
        assert kinds == sorted(kinds, reverse=True), "packs run before dense bins"
        for shard in shards:
            assert _kinds(shard, global_batch) == kinds
    assert sorted(row for shard in shards for row in shard.row_indices) == list(
        range(len(lengths))
    )


def test_a_mixed_batch_gives_every_rank_a_pack_and_a_dense_bin() -> None:
    lengths, groups, prompts = _mixed_batch()
    shards = _plan(lengths, groups, prompts, num_shards=4, budget=400)
    for shard in shards:
        assert _kinds(shard) == ["pack", "dense"]
        ((_, tree_ids), _) = _forwards(shard)
        assert tree_ids == [0, 0, 0, 0]
        assert shard.micro_batch_lengths[0][0] == 80 + 4 * 20
    assert sorted(shard.elem_counts_per_gb[0] for shard in shards) == [5, 5, 5, 6]


def test_one_tree_per_rank_needs_no_dense_bin() -> None:
    lengths, groups, prompts = _one_tree_per_rank()
    shards = _plan(lengths, groups, prompts, num_shards=64, budget=4096)
    for shard in shards:
        assert _kinds(shard) == ["pack"]
        assert shard.tree_ids == [0] * 16


def test_too_few_trees_for_the_ranks_divides_a_tree() -> None:
    shards = _plan([100] * 8, ["a"] * 8, [80] * 8, num_shards=2)
    assert [tree_ids for shard in shards for _, tree_ids in _forwards(shard)] == [
        [0] * 4,
        [0] * 4,
    ]


def test_only_a_tree_of_four_rows_or_more_is_divided() -> None:
    # One tree of eight rows on four ranks becomes four trees of two rows.
    shards = _plan([100] * 8, ["a"] * 8, [80] * 8, num_shards=4)
    assert [shard.tree_ids for shard in shards] == [[0, 0]] * 4
    # Eight ranks would need trees of one row, so the rows run dense.
    shards = _plan([100] * 8, ["a"] * 8, [80] * 8, num_shards=8)
    assert [shard.tree_ids for shard in shards] == [[UNSHARED_ROW]] * 8


def test_trees_that_cannot_fill_a_pack_per_rank_run_dense() -> None:
    # Three trees of two rows cannot give each of the four ranks a pack.
    lengths, groups, prompts = [100] * 10, list("aabbccdefg"), [80] * 6 + [0] * 4
    shards = _plan(lengths, groups, prompts, num_shards=4)
    for shard in shards:
        assert _kinds(shard) == ["dense"]
    assert sorted(row for shard in shards for row in shard.row_indices) == list(
        range(10)
    )


def test_a_small_tree_stops_sharing_when_every_rank_needs_a_forward() -> None:
    shards = _plan([100, 100], ["a"] * 2, [80] * 2, num_shards=2)
    assert [shard.tree_ids for shard in shards] == [[UNSHARED_ROW], [UNSHARED_ROW]]


def test_the_smallest_tree_stops_sharing_when_dense_rows_are_too_few() -> None:
    # Trees of four, two and three rows, and one unshared row, on two ranks.
    lengths, groups, prompts = [100] * 10, list("aaaabbcccd"), [80] * 9 + [0]
    shards = _plan(lengths, groups, prompts, num_shards=2)
    for shard in shards:
        assert _kinds(shard) == ["pack", "dense"]
    tree_id_of_row = {
        row: tree_id
        for shard in shards
        for row, tree_id in zip(shard.row_indices, shard.tree_ids, strict=True)
    }
    # Only the tree of two rows joins the unshared row in the dense bins.
    assert [row for row in range(10) if tree_id_of_row[row] == UNSHARED_ROW] == [
        4,
        5,
        9,
    ]


def test_fewer_rows_than_ranks_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot create 4 forwards with only 2 rows"):
        _plan([100, 100], ["a"] * 2, [80] * 2, num_shards=4)


def test_forwards_are_dealt_to_the_lightest_rank() -> None:
    lengths = [900, 500, 400, 300]
    shards = _plan(lengths, ["a", "b", "c", "d"], [0] * 4, num_shards=2, budget=900)
    assert sorted(
        sorted(lengths[row] for row in shard.row_indices) for shard in shards
    ) == [
        [300, 900],
        [400, 500],
    ]


def test_token_balance_counts_packs_and_dense_bins() -> None:
    # Packs of 900 and 200 tokens, then dense bins of 600 and 300 tokens.
    lengths, groups, prompts = (
        [500, 500, 150, 150, 600, 300],
        list("aabbcd"),
        [100] * 4 + [0] * 2,
    )
    shards = _plan(lengths, groups, prompts, num_shards=2, budget=1000)
    # The rank with the short pack gets the long dense bin.
    assert sorted(shard.micro_batch_lengths[0] for shard in shards) == [
        [200, 600],
        [900, 300],
    ]


def test_largest_first_orders_each_rank() -> None:
    lengths, groups, prompts = [100, 100, 250, 90], ["a", "b", "c", "d"], [0] * 4
    (in_order,) = _plan(lengths, groups, prompts, budget=300)
    (largest,) = _plan(lengths, groups, prompts, budget=300, largest_first=True)
    assert in_order.micro_batch_lengths == [[250, 290]]
    assert largest.micro_batch_lengths == [[290, 250]]
    assert sorted(in_order.row_indices) == sorted(largest.row_indices) == [0, 1, 2, 3]


def test_largest_first_orders_packs_and_dense_bins_separately() -> None:
    # Trees of 140 and 520 tokens, and unshared rows of 500 and 590 tokens.
    lengths, groups, prompts = (
        [100, 100, 300, 300, 500, 590],
        list("aabbcd"),
        [60, 60, 80, 80, 0, 0],
    )
    (shard,) = _plan(lengths, groups, prompts, budget=600, largest_first=True)
    # The longest dense bin still runs after the shortest pack.
    assert shard.micro_batch_lengths == [[520, 140, 590, 500]]
    assert _kinds(shard) == ["pack", "pack", "dense", "dense"]


def test_global_batches_are_planned_separately() -> None:
    lengths, groups, prompts = [100] * 8, ["a"] * 4 + ["b"] * 4, [80] * 8
    shards = _plan(lengths, groups, prompts, num_shards=2, batch_size=4)
    for shard in shards:
        assert shard.elem_counts_per_gb == [2, 2]
        # Row ranges restart at every global batch.
        assert shard.micro_batch_indices == [[[0, 2]], [[0, 2]]]
        assert shard.micro_batch_lengths == [[120], [120]]
    assert [
        sorted(shard.row_indices[:2] + shard.row_indices[2:]) for shard in shards
    ] == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]


@pytest.mark.parametrize("length", [0, 1025])
def test_a_row_that_cannot_fit_a_forward_is_rejected(length: int) -> None:
    with pytest.raises(ValueError, match="must be nonempty and fit one forward"):
        _plan([length], ["a"], [0])


def test_rows_must_divide_into_global_batches() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive and divide"):
        _plan([100] * 3, ["a"] * 3, [0] * 3, batch_size=2)


@pytest.mark.parametrize("seed", range(25))
def test_random_batches_keep_the_plan_invariants(seed: int) -> None:
    rng = random.Random(seed)
    num_shards = rng.choice([1, 2, 4, 8])
    pad_multiple = rng.choice([1, 8])
    bound_by_full_length = rng.choice([False, True])
    budget = 512
    # A multiple of every rank count, so each global batch can always be planned.
    batch_size = 8 * rng.randint(1, 6)
    num_global_batches = rng.randint(1, 3)
    lengths: list[int] = []
    groups: list[str] = []
    prompts: list[int] = []
    for global_batch in range(num_global_batches):
        group_prompts = [rng.randint(1, 120) for _ in range(rng.randint(1, 6))]
        for _ in range(batch_size):
            group = rng.randrange(len(group_prompts))
            groups.append(f"{global_batch}-{group}")
            prompts.append(rng.choice([0, group_prompts[group], group_prompts[group]]))
            lengths.append(group_prompts[group] + rng.randint(0, 200))
    shards = _plan(
        lengths,
        groups,
        prompts,
        num_shards=num_shards,
        batch_size=batch_size,
        budget=budget,
        pad_multiple=pad_multiple,
        bound_by_full_length=bound_by_full_length,
    )

    padded = [-(-length // pad_multiple) * pad_multiple for length in lengths]
    for global_batch in range(num_global_batches):
        kinds = _kinds(shards[0], global_batch)
        assert kinds == sorted(kinds, reverse=True)
        rows_seen: list[int] = []
        for shard in shards:
            assert _kinds(shard, global_batch) == kinds
            forwards = _forwards(shard, global_batch)
            for (rows, tree_ids), forward_length in zip(
                forwards, shard.micro_batch_lengths[global_batch], strict=True
            ):
                rows_seen.extend(rows)
                assert forward_length <= budget
                if tree_ids[0] == UNSHARED_ROW:
                    assert forward_length == sum(padded[row] for row in rows)
                    continue
                # Tree ids count up from 0 and the rows of a tree are contiguous.
                assert tree_ids == sorted(tree_ids)
                assert set(tree_ids) == set(range(tree_ids[-1] + 1))
                packed_length = 0
                for tree_id in set(tree_ids):
                    tree = [
                        row
                        for row, row_tree in zip(rows, tree_ids)
                        if row_tree == tree_id
                    ]
                    assert 2 <= len(tree) <= MAX_TREE_ROWS
                    assert len({(groups[row], prompts[row]) for row in tree}) == 1
                    prompt = prompts[tree[0]]
                    assert all(0 < prompt < lengths[row] for row in tree)
                    packed_length += prompt + sum(padded[row] - prompt for row in tree)
                assert (
                    forward_length == -(-packed_length // pad_multiple) * pad_multiple
                )
                if bound_by_full_length:
                    assert sum(padded[row] for row in rows) <= budget
            # The row ranges tile the rank's rows of this global batch.
            assert [end for _, end in shard.micro_batch_indices[global_batch]][-1] == (
                shard.elem_counts_per_gb[global_batch]
            )
        assert sorted(rows_seen) == list(
            range(global_batch * batch_size, (global_batch + 1) * batch_size)
        )
