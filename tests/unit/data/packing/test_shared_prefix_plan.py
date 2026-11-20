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

import pytest

from nemo_rl.data.packing.shared_prefix.columns import UNSHARED_ROW
from nemo_rl.data.packing.shared_prefix.plan import (
    MAX_TREE_ROWS,
    SharedPrefixShard,
    plan_shared_prefix_shards,
)


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


def test_rows_of_one_prompt_share_one_forward():
    (shard,) = _plan([100, 120, 140, 160], ["a"] * 4, [80] * 4)
    assert _forwards(shard) == [([0, 1, 2, 3], [0, 0, 0, 0])]
    # The prompt once, then the four completions.
    assert shard.micro_batch_lengths == [[80 + 20 + 40 + 60 + 80]]
    assert shard.elem_counts_per_gb == [4]


def test_padding_applies_to_each_row_and_to_the_forward():
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
def test_rows_that_differ_run_dense(groups, prompts, hashes):
    (shard,) = _plan([100, 100], groups, prompts, hashes=hashes)
    assert _forwards(shard) == [([0, 1], [UNSHARED_ROW, UNSHARED_ROW])]
    assert shard.micro_batch_lengths == [[200]]


def test_packs_and_dense_bins_are_separate_forwards():
    (shard,) = _plan([100, 100, 100, 50], ["a", "a", "b", "c"], [80, 80, 80, 0])
    assert _forwards(shard) == [
        ([0, 1], [0, 0]),
        ([2, 3], [UNSHARED_ROW, UNSHARED_ROW]),
    ]
    assert shard.micro_batch_lengths == [[120, 150]]


def test_several_trees_share_a_forward():
    (shard,) = _plan([100] * 6, ["a", "a", "a", "b", "b", "b"], [80] * 6)
    ((rows, tree_ids),) = _forwards(shard)
    assert sorted(rows) == [0, 1, 2, 3, 4, 5]
    # Rows of one tree are contiguous, and the two trees have different ids.
    assert tree_ids == [0, 0, 0, 1, 1, 1]
    assert shard.micro_batch_lengths == [[2 * (80 + 3 * 20)]]


def test_a_group_larger_than_the_row_limit_splits_evenly():
    rows = MAX_TREE_ROWS + 1
    (shard,) = _plan([100] * rows, ["a"] * rows, [80] * rows, budget=10_000)
    ((_, tree_ids),) = _forwards(shard)
    assert sorted(tree_ids.count(tree) for tree in set(tree_ids)) == [8, 9]


def test_a_tree_larger_than_the_budget_splits():
    # Packed, four rows need 80 + 4 * 120 = 560 tokens; two rows need 320.
    (shard,) = _plan([200] * 4, ["a"] * 4, [80] * 4, budget=400)
    assert [tree_ids for _, tree_ids in _forwards(shard)] == [[0, 0], [0, 0]]
    assert shard.micro_batch_lengths == [[320, 320]]


def test_full_length_bound_counts_the_prompt_in_every_row():
    lengths, groups, prompts = [100] * 8, ["a"] * 4 + ["b"] * 4, [80] * 8
    (packed,) = _plan(lengths, groups, prompts, budget=400)
    (full,) = _plan(lengths, groups, prompts, budget=400, bound_by_full_length=True)
    # Packed, both trees (160 tokens each) fit one forward. In full they are 400 tokens each.
    assert packed.micro_batch_lengths == [[320]]
    assert full.micro_batch_lengths == [[160, 160]]


def test_every_rank_gets_the_same_number_of_forwards():
    lengths = [100] * 12 + [300, 50, 60]
    groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d", "e", "f"]
    prompts = [80] * 12 + [0, 0, 0]
    shards = _plan(lengths, groups, prompts, num_shards=4, budget=400)
    assert len({len(shard.micro_batch_indices[0]) for shard in shards}) == 1
    assert sorted(row for shard in shards for row in shard.row_indices) == list(
        range(15)
    )


def test_too_few_trees_for_the_ranks_divides_a_tree():
    shards = _plan([100] * 8, ["a"] * 8, [80] * 8, num_shards=2)
    assert [tree_ids for shard in shards for _, tree_ids in _forwards(shard)] == [
        [0] * 4,
        [0] * 4,
    ]


def test_a_small_tree_stops_sharing_when_every_rank_needs_a_forward():
    shards = _plan([100, 100], ["a"] * 2, [80] * 2, num_shards=2)
    assert [shard.tree_ids for shard in shards] == [[UNSHARED_ROW], [UNSHARED_ROW]]


def test_fewer_rows_than_ranks_is_rejected():
    with pytest.raises(ValueError, match="Cannot create 4 forwards with only 2 rows"):
        _plan([100, 100], ["a"] * 2, [80] * 2, num_shards=4)


def test_forwards_are_dealt_to_the_lightest_rank():
    lengths = [900, 500, 400, 300]
    shards = _plan(lengths, ["a", "b", "c", "d"], [0] * 4, num_shards=2, budget=900)
    assert sorted(
        sorted(lengths[row] for row in shard.row_indices) for shard in shards
    ) == [
        [300, 900],
        [400, 500],
    ]


def test_largest_first_orders_each_rank():
    lengths, groups, prompts = [50, 300, 200, 100], ["a", "b", "c", "d"], [0] * 4
    (in_order,) = _plan(lengths, groups, prompts, budget=300)
    (largest,) = _plan(lengths, groups, prompts, budget=300, largest_first=True)
    assert largest.micro_batch_lengths == [
        sorted(in_order.micro_batch_lengths[0], reverse=True)
    ]
    assert sorted(in_order.row_indices) == sorted(largest.row_indices) == [0, 1, 2, 3]


def test_global_batches_are_planned_separately():
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
def test_a_row_that_cannot_fit_a_forward_is_rejected(length):
    with pytest.raises(ValueError, match="must be nonempty and fit one forward"):
        _plan([length], ["a"], [0])


def test_rows_must_divide_into_global_batches():
    with pytest.raises(ValueError, match="batch_size must be positive and divide"):
        _plan([100] * 3, ["a"] * 3, [0] * 3, batch_size=2)
