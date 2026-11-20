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
"""Plan the forwards of a batch whose rows share prompts.

Rows of one prompt group that start with the same prompt form a prefix tree: a forward runs the
prompt once, followed by each row's completion. A forward is either a pack of trees or a dense
bin of rows that share with nothing. The driver plans every data-parallel rank's forwards here,
from row lengths alone, so all ranks run the same number of forwards.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from nemo_rl.data.packing.algorithms import get_packer
from nemo_rl.data.packing.shared_prefix.columns import UNSHARED_ROW

MAX_TREE_ROWS = 16
"""Most rows behind one prompt copy. A larger group is split into trees of even size."""


@dataclass(frozen=True)
class SharedPrefixShard:
    """One data-parallel rank's rows and forwards, in execution order.

    Attributes:
        row_indices: Batch rows of this rank, forward by forward.
        tree_ids: For each of those rows, its tree within its forward, or ``UNSHARED_ROW``. The
            rows of a tree are contiguous.
        micro_batch_indices: Per global batch, the ``[start, end]`` row range of each forward,
            counted from the first row of that global batch on this rank.
        micro_batch_lengths: Per global batch, the token count each forward runs.
        elem_counts_per_gb: Rows of this rank in each global batch.
    """

    row_indices: list[int]
    tree_ids: list[int]
    micro_batch_indices: list[list[list[int]]]
    micro_batch_lengths: list[list[int]]
    elem_counts_per_gb: list[int]


@dataclass(frozen=True)
class _Forward:
    """One planned forward: the trees of a pack, or the rows of a dense bin."""

    trees: tuple[tuple[int, ...], ...]
    dense_rows: tuple[int, ...]
    length: int


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _split_evenly(
    rows: list[int], lengths: list[int], *, capacity: int
) -> list[list[int]]:
    """Split one prompt's rows into the fewest parts that fit ``capacity`` and ``MAX_TREE_ROWS``.

    First-fit decreasing over ``lengths``. Part sizes differ by at most one when only the row
    limit binds, so 17 rows become 9 + 8 rather than 16 + 1.
    """
    num_parts = -(-len(rows) // MAX_TREE_ROWS)
    base, extra = divmod(len(rows), num_parts)
    large_parts = 0
    parts: list[list[int]] = []
    part_lengths: list[int] = []
    for position in sorted(range(len(rows)), key=lambda p: (-lengths[p], rows[p])):
        for index, part in enumerate(parts):
            has_row_slot = len(part) < base or (
                len(part) == base and large_parts < extra
            )
            if has_row_slot and part_lengths[index] + lengths[position] <= capacity:
                large_parts += len(part) == base
                part.append(rows[position])
                part_lengths[index] += lengths[position]
                break
        else:
            parts.append([rows[position]])
            part_lengths.append(lengths[position])
    return [sorted(part) for part in parts]


def _grow_bins(bins: list[list[int]], extra: int) -> list[list[int]]:
    """Open ``extra`` bins, each seeded with one item taken from the bin that holds the most."""
    bins = [list(items) for items in bins]
    for _ in range(extra):
        source = max((items for items in bins if len(items) > 1), key=len)
        bins.append([source.pop()])
    return bins


def _plan_global_batch(
    *,
    padded_lengths: list[int],
    share_keys: list[tuple[str, int, int] | None],
    num_shards: int,
    capacity: int,
    pad_multiple: int,
    algorithm: str,
    bound_by_full_length: bool,
) -> list[_Forward]:
    """Plan one global batch: a multiple of ``num_shards`` forwards over rows ``0..n-1``."""
    prompt_length = {
        row: key[1] for row, key in enumerate(share_keys) if key is not None
    }

    def tree_lengths(tree: tuple[int, ...]) -> tuple[int, int]:
        """Packed length (prompt once) and full length (prompt in every row) of a tree."""
        full = sum(padded_lengths[row] for row in tree)
        return full - (len(tree) - 1) * prompt_length[tree[0]], full

    def tree_cost(tree: tuple[int, ...]) -> int:
        return tree_lengths(tree)[1 if bound_by_full_length else 0]

    # A pack ends with padding up to pad_multiple, so its trees must fit below the last multiple.
    pack_capacity = capacity - capacity % pad_multiple
    by_key: dict[tuple[str, int, int], list[int]] = {}
    for row, key in enumerate(share_keys):
        if key is not None:
            by_key.setdefault(key, []).append(row)
    trees: list[tuple[int, ...]] = []
    dense_rows = [row for row, key in enumerate(share_keys) if key is None]
    for key, rows in by_key.items():
        # Counted in full, each row costs its whole length. Counted packed, the tree pays the
        # prompt once and each row adds its completion.
        shared_tokens = 0 if bound_by_full_length else key[1]
        parts = _split_evenly(
            rows,
            [padded_lengths[row] - shared_tokens for row in rows],
            capacity=pack_capacity - shared_tokens,
        )
        for part in parts:
            if len(part) > 1:
                trees.append(tuple(part))
            else:
                dense_rows.extend(part)
    dense_rows.sort()

    packer = get_packer(algorithm=algorithm, bin_capacity=capacity)
    pack_packer = get_packer(algorithm=algorithm, bin_capacity=pack_capacity)
    while True:
        tree_bins = (
            pack_packer.pack([tree_cost(tree) for tree in trees]) if trees else []
        )
        dense_bins = (
            packer.pack([padded_lengths[row] for row in dense_rows])
            if dense_rows
            else []
        )
        natural = len(tree_bins) + len(dense_bins)
        target = max(natural, num_shards)
        target += -target % num_shards
        extra = target - natural
        spare_dense = len(dense_rows) - len(dense_bins)
        spare_trees = len(trees) - len(tree_bins)
        if extra <= spare_dense + spare_trees:
            dense_bins = _grow_bins(dense_bins, min(extra, spare_dense))
            tree_bins = _grow_bins(tree_bins, extra - min(extra, spare_dense))
            break
        if not trees:
            raise ValueError(
                f"Cannot create {target} forwards with only {len(padded_lengths)} rows. Each "
                f"of the {num_shards} data-parallel ranks needs at least one row per global batch."
            )
        # Too few items for one forward per rank: divide the largest tree, or stop sharing it.
        largest = max(range(len(trees)), key=lambda index: len(trees[index]))
        rows = list(trees.pop(largest))
        if len(rows) >= 4:
            trees.extend((tuple(rows[0::2]), tuple(rows[1::2])))
        else:
            dense_rows = sorted(dense_rows + rows)

    forwards = [
        _Forward(
            trees=tuple(trees[index] for index in items),
            dense_rows=(),
            length=_round_up(
                sum(tree_lengths(trees[index])[0] for index in items), pad_multiple
            ),
        )
        for items in tree_bins
    ]
    forwards += [
        _Forward(
            trees=(),
            dense_rows=tuple(dense_rows[index] for index in items),
            length=sum(padded_lengths[dense_rows[index]] for index in items),
        )
        for items in dense_bins
    ]
    return forwards


def plan_shared_prefix_shards(
    *,
    sequence_lengths: Sequence[int],
    group_ids: Sequence[str],
    prompt_lengths: Sequence[int],
    prompt_hashes: Sequence[int],
    num_shards: int,
    batch_size: int | None,
    max_tokens_per_microbatch: int,
    sequence_length_pad_multiple: int,
    algorithm: str,
    bound_by_full_length: bool,
    largest_first: bool,
) -> list[SharedPrefixShard]:
    """Plan every rank's forwards for a batch, sharing each prompt among the rows that have it.

    Two rows share when they have the same group, prompt length and prompt hash, and a prompt
    length between 1 and their length minus 1. Every rank gets the same number of forwards, and
    the forwards are dealt longest first to the rank with the fewest tokens so far.

    Args:
        sequence_lengths: Token count of each row.
        group_ids: Prompt group of each row.
        prompt_lengths: Shareable prompt tokens at the start of each row; 0 for none.
        prompt_hashes: Hash of each row's prompt tokens.
        num_shards: Number of data-parallel ranks.
        batch_size: Rows per global batch; ``None`` plans all rows as one.
        max_tokens_per_microbatch: Token budget of one forward.
        sequence_length_pad_multiple: Multiple that each row, and each forward, is padded to.
        algorithm: Packing algorithm name, as for ordinary sequence packing.
        bound_by_full_length: Count a tree at its full length, with the prompt in every row,
            against the budget. Needed when the model also processes the rows in full, as the
            multi-token-prediction heads do.
        largest_first: Run each rank's forwards from longest to shortest.

    Returns:
        One ``SharedPrefixShard`` per rank.

    Raises:
        ValueError: If a row is empty or exceeds the budget, the rows do not divide into global
            batches, or a global batch has fewer rows than ranks.
    """
    num_rows = len(sequence_lengths)
    if batch_size is None:
        batch_size = num_rows
    if batch_size < 1 or num_rows % batch_size != 0:
        raise ValueError(
            f"batch_size must be positive and divide the {num_rows} rows, got {batch_size}"
        )
    padded_lengths = [
        _round_up(int(length), sequence_length_pad_multiple)
        for length in sequence_lengths
    ]
    for row, padded in enumerate(padded_lengths):
        if not 0 < padded <= max_tokens_per_microbatch:
            raise ValueError(
                "A row must be nonempty and fit one forward after padding: "
                f"row={row}, length={sequence_lengths[row]}, padded_length={padded}, "
                f"max_tokens_per_microbatch={max_tokens_per_microbatch}"
            )
    share_keys = [
        (group_ids[row], int(prompt_lengths[row]), int(prompt_hashes[row]))
        if 0 < prompt_lengths[row] < sequence_lengths[row]
        else None
        for row in range(num_rows)
    ]

    row_indices: list[list[int]] = [[] for _ in range(num_shards)]
    tree_ids: list[list[int]] = [[] for _ in range(num_shards)]
    micro_batch_indices: list[list[list[list[int]]]] = [[] for _ in range(num_shards)]
    micro_batch_lengths: list[list[list[int]]] = [[] for _ in range(num_shards)]
    elem_counts: list[list[int]] = [[] for _ in range(num_shards)]
    for start in range(0, num_rows, batch_size):
        forwards = _plan_global_batch(
            padded_lengths=padded_lengths[start : start + batch_size],
            share_keys=share_keys[start : start + batch_size],
            num_shards=num_shards,
            capacity=max_tokens_per_microbatch,
            pad_multiple=sequence_length_pad_multiple,
            algorithm=algorithm,
            bound_by_full_length=bound_by_full_length,
        )
        quota = len(forwards) // num_shards
        dealt: list[list[int]] = [[] for _ in range(num_shards)]
        tokens = [0] * num_shards
        # A stable sort, so forwards of equal length keep the planning order.
        for index in sorted(
            range(len(forwards)), key=lambda index: -forwards[index].length
        ):
            shard = min(
                (shard for shard in range(num_shards) if len(dealt[shard]) < quota),
                key=lambda shard: tokens[shard],
            )
            dealt[shard].append(index)
            tokens[shard] += forwards[index].length
        for shard in range(num_shards):
            indices: list[list[int]] = []
            lengths: list[int] = []
            count = 0
            for index in dealt[shard] if largest_first else sorted(dealt[shard]):
                forward = forwards[index]
                rows = [row for tree in forward.trees for row in tree] or list(
                    forward.dense_rows
                )
                row_indices[shard].extend(start + row for row in rows)
                tree_ids[shard].extend(
                    [
                        tree_id
                        for tree_id, tree in enumerate(forward.trees)
                        for _ in tree
                    ]
                    or [UNSHARED_ROW] * len(rows)
                )
                indices.append([count, count + len(rows)])
                lengths.append(forward.length)
                count += len(rows)
            micro_batch_indices[shard].append(indices)
            micro_batch_lengths[shard].append(lengths)
            elem_counts[shard].append(count)
    return [
        SharedPrefixShard(
            row_indices=row_indices[shard],
            tree_ids=tree_ids[shard],
            micro_batch_indices=micro_batch_indices[shard],
            micro_batch_lengths=micro_batch_lengths[shard],
            elem_counts_per_gb=elem_counts[shard],
        )
        for shard in range(num_shards)
    ]
