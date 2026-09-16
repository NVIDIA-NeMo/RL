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
"""Tests for the rollout first-write helper and the meta-only sharder.

After the sync 1-hop refactor, ``fan_out_per_rank_metas`` was retired in
favor of:

  * ``kv_first_write`` — single flat ``put_samples`` of every tensor
    field in the rollout output (multimodal extras ride along).
  * ``shard_meta_for_dp`` — pure key-list split per DP rank, no I/O.

These tests lock in the schema-extensibility behavior (multimodal
fields propagate) and the meta-sharding contract (no key minting,
identity preserved across shards).
"""

from __future__ import annotations

import torch

from nemo_rl.data.packed_rollouts import (
    PACKED_ATTENTION_SEGMENT_LENGTHS,
    PACKED_ATTENTION_SELECTED_SEGMENTS,
    TREE_ATTENTION_EDGE_LENGTHS,
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_FRAGMENT_SELECTIONS,
    TREE_ATTENTION_LAYOUTS,
    TreeAttentionLayout,
    expand_batched_data_for_packed_attention,
    expand_batched_data_for_tree_attention,
    expand_selected_packed_attention_segments,
    materialize_tree_attention_fragments,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.data_plane.preshard import (
    shard_meta_for_dp,
    split_packed_attention_microbatch_metas,
    split_tree_attention_microbatch_metas,
)
from nemo_rl.data_plane.schema import (
    DP_TRAIN_FIELDS,
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from ._rollout_shapes import (
    keys_from_uids,
    make_rollout_batch,
    register_train_partition,
)


def _final_batch(n_samples: int = 4, *, with_extras: bool = False) -> BatchedDataDict:
    d: BatchedDataDict = BatchedDataDict()
    d["input_ids"] = torch.zeros((n_samples, 8), dtype=torch.long)
    d["input_lengths"] = torch.tensor([8] * n_samples, dtype=torch.long)
    d["token_mask"] = torch.ones((n_samples, 8), dtype=torch.long)
    d["sample_mask"] = torch.ones((n_samples,), dtype=torch.long)
    d["generation_logprobs"] = torch.zeros((n_samples, 8), dtype=torch.float32)
    if with_extras:
        d["pixel_values"] = torch.zeros((n_samples, 3, 4, 4), dtype=torch.float32)
    return d


# ── kv_first_write schema extensibility ────────────────────────────────


def test_kv_first_write_writes_seed_fields():
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(
        fb, sample_ids=keys_from_uids(uids), dp_client=client, partition_id="train"
    )
    # Every tensor field in the input lands in TQ under f"{uid}_g0".
    assert meta.sample_ids == [f"u{i}_g0" for i in range(4)]
    fetched = client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id="train",
        select_fields=["input_ids", "input_lengths", "token_mask", "sample_mask"],
    )
    assert fetched["input_ids"].shape == (4, 8)


def test_kv_first_write_carries_multimodal_extras():
    """VLM extras (pixel_values) ride along with no schema declaration."""
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=4)
    fb = _final_batch(4, with_extras=True)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(
        fb, sample_ids=keys_from_uids(uids), dp_client=client, partition_id="train"
    )
    assert "pixel_values" in (meta.fields or [])
    fetched = client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id="train",
        select_fields=["pixel_values"],
    )
    assert fetched["pixel_values"].shape == (4, 3, 4, 4)


def test_kv_first_write_keys_match_uids_x_ngen():
    """Keys round-trip: caller mints ``f"{uid}_g{i}"``, helper preserves them
    in ``meta.sample_ids`` byte-for-byte."""
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=6)
    fb = _final_batch(6)  # 3 prompts × 2 generations
    uids = ["a", "b", "c"]
    keys = keys_from_uids(uids, n_gen=2)
    meta = kv_first_write(fb, sample_ids=keys, dp_client=client, partition_id="train")
    assert meta.sample_ids == ["a_g0", "a_g1", "b_g0", "b_g1", "c_g0", "c_g1"]


# ── shard_meta_for_dp invariants ──────────────────────────────────────


def _meta(n: int) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=[f"k{i}" for i in range(n)],
        fields=list(DP_TRAIN_FIELDS),
        sequence_lengths=[10 + i for i in range(n)],
        extra_info={},
    )


def test_shard_meta_for_dp_partitions_keys_disjointly():
    n, dp = 8, 4
    metas, _ = shard_meta_for_dp(_meta(n), dp_world=dp, batch_size=n)
    assert len(metas) == dp
    flat = [k for m in metas for k in m.sample_ids]
    assert sorted(flat) == sorted(_meta(n).sample_ids)  # same set, no dups, no minting


def test_shard_meta_for_dp_preserves_partition_id():
    metas, _ = shard_meta_for_dp(_meta(4), dp_world=2, batch_size=4)
    assert all(m.partition_id == "train" for m in metas)


def test_shard_meta_for_dp_preserves_tag_row_identity():
    meta = _meta(8)
    meta.tags = [{"row": sample_id} for sample_id in meta.sample_ids]

    metas, _ = shard_meta_for_dp(meta, dp_world=4, batch_size=8)

    for shard in metas:
        assert shard.tags is not None
        assert [tag["row"] for tag in shard.tags] == shard.sample_ids


def test_shard_meta_for_dp_permutes_tags_with_sample_ids():
    """Tags must ride each sample, not each position.

    ``tags`` carries the per-row multimodal shapes. Sharding sorts by sequence
    length, so a tags list left un-permuted would pair one sample's pixel bytes
    with another sample's shapes — and ``from_wire`` would reshape into it
    silently instead of raising.
    """
    n, dp = 8, 4
    meta = _meta(n)
    # Tag each sample with its own id so a mispairing is directly visible.
    meta.tags = [{"owner": sid} for sid in meta.sample_ids]

    metas, _ = shard_meta_for_dp(meta, dp_world=dp, batch_size=n)

    for shard in metas:
        assert shard.tags is not None
        assert len(shard.tags) == len(shard.sample_ids)
        for sid, tag in zip(shard.sample_ids, shard.tags):
            assert tag["owner"] == sid


def test_shard_meta_for_dp_leaves_tags_none_when_absent():
    """Text-only runs write no tags; sharding must not invent an empty list."""
    metas, _ = shard_meta_for_dp(_meta(4), dp_world=2, batch_size=4)
    assert all(m.tags is None for m in metas)


def test_shard_meta_for_dp_unsorted_round_trip():
    """unsorted_indices must reconstruct the input order from DP-rank concat."""
    n, dp = 8, 4
    metas, unsorted = shard_meta_for_dp(_meta(n), dp_world=dp, batch_size=n)
    if unsorted is None:
        # No reorder happened — DP-rank concat IS the original order.
        return
    # Build a tensor whose row i is i; permute via dispatch order; reorder back.
    flat = [k for m in metas for k in m.sample_ids]
    aggregated = BatchedDataDict(
        {"row": torch.tensor([_meta(n).sample_ids.index(k) for k in flat])}
    )
    aggregated.reorder_data(unsorted)
    assert aggregated["row"].tolist() == list(range(n))


def test_sequence_packed_meta_reorders_worker_rows_with_public_api():
    meta = KVBatchMeta(
        partition_id="train",
        task_name="prev_lp",
        sample_ids=["s0", "s1", "s2", "s3"],
        sequence_lengths=[1, 4, 2, 3],
    )
    metas, unsorted = shard_meta_for_dp(
        meta,
        dp_world=2,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "max_tokens_per_microbatch": 5,
            "sequence_length_pad_multiple": 1,
        },
    )

    dispatched = [
        meta.sample_ids.index(key) for rank in metas for key in rank.sample_ids
    ]
    aggregated = BatchedDataDict({"row": torch.tensor(dispatched)})
    if unsorted is not None:
        aggregated.reorder_data(unsorted)
    assert aggregated["row"].tolist() == [0, 1, 2, 3]


def test_shard_meta_for_dp_balances_exact_calls_and_preserves_logical_keys():
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["rollout-0", "rollout-1"],
        fields=list(DP_TRAIN_FIELDS),
        sequence_lengths=[5, 4],
        extra_info={PACKED_ATTENTION_SEGMENT_LENGTHS: [[3, 2], [4]]},
        tags=[{"reward": 1.0}, {"reward": 2.0}],
    )
    metas, unsorted = shard_meta_for_dp(
        meta,
        dp_world=2,
        batch_size=2,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "max_tokens_per_microbatch": 5,
            "sequence_length_pad_multiple": 1,
        },
    )

    call_offsets = {"rollout-0": 0, "rollout-1": 2}
    dispatched_calls = []
    for rank_meta in metas:
        assert rank_meta.tags == [
            meta.tags[meta.sample_ids.index(sample_id)]
            for sample_id in rank_meta.sample_ids
        ]
        assert len(rank_meta.sample_ids) == len(set(rank_meta.sample_ids))
        assert len(rank_meta.extra_info[PACKED_ATTENTION_SEGMENT_LENGTHS]) == len(
            rank_meta.sample_ids
        )
        assert MICRO_BATCH_INDICES in rank_meta.extra_info
        assert MICRO_BATCH_LENGTHS in rank_meta.extra_info
        for local_row, segment in rank_meta.extra_info[
            PACKED_ATTENTION_SELECTED_SEGMENTS
        ]:
            dispatched_calls.append(
                call_offsets[rank_meta.sample_ids[local_row]] + segment
            )

    assert sorted(dispatched_calls) == [0, 1, 2]
    if unsorted is None:
        assert dispatched_calls == [0, 1, 2]
    else:
        aggregated = BatchedDataDict({"call": torch.tensor(dispatched_calls)})
        aggregated.reorder_data(unsorted)
        assert aggregated["call"].tolist() == [0, 1, 2]


def test_tq_selected_calls_match_legacy_expansion_including_routes() -> None:
    segment_lengths = [[3, 2], [4]]
    logical = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 20, 21], [30, 31, 32, 33, 0]]),
            "input_lengths": torch.tensor([5, 4]),
            "token_mask": torch.tensor([[0, 1, 1, 0, 1], [0, 1, 1, 1, 0]]),
            "generation_logprobs": torch.arange(10, dtype=torch.float32).reshape(2, 5),
            "routed_experts": torch.arange(2 * 5 * 3 * 2, dtype=torch.int16).reshape(
                2, 5, 3, 2
            ),
            PACKED_ATTENTION_SEGMENT_LENGTHS: segment_lengths,
        }
    )
    legacy, _ = expand_batched_data_for_packed_attention(
        logical, output_sequence_length=6
    )
    meta = KVBatchMeta(
        partition_id="train",
        task_name="prev_lp",
        sample_ids=["rollout-0", "rollout-1"],
        fields=[
            "input_ids",
            "input_lengths",
            "token_mask",
            "generation_logprobs",
            "routed_experts",
        ],
        sequence_lengths=[5, 4],
        extra_info={PACKED_ATTENTION_SEGMENT_LENGTHS: segment_lengths},
    )
    rank_metas, unsorted = shard_meta_for_dp(
        meta,
        dp_world=2,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "max_tokens_per_microbatch": 5,
            "sequence_length_pad_multiple": 1,
        },
    )

    sample_index = {"rollout-0": 0, "rollout-1": 1}
    rank_calls = []
    for rank_meta in rank_metas:
        parent_indices = torch.tensor(
            [sample_index[sample_id] for sample_id in rank_meta.sample_ids]
        )
        fetched = BatchedDataDict(
            {
                key: value.index_select(0, parent_indices)
                for key, value in logical.items()
                if key != PACKED_ATTENTION_SEGMENT_LENGTHS
            }
        )
        rank_calls.append(
            expand_selected_packed_attention_segments(
                fetched,
                segment_lengths=rank_meta.extra_info[PACKED_ATTENTION_SEGMENT_LENGTHS],
                selected_segments=rank_meta.extra_info[
                    PACKED_ATTENTION_SELECTED_SEGMENTS
                ],
                output_sequence_length=6,
            )
        )

    tq_calls = BatchedDataDict.from_batches(rank_calls)
    if unsorted is not None:
        tq_calls.reorder_data(unsorted)
    for field in (
        "input_ids",
        "input_lengths",
        "token_mask",
        "generation_logprobs",
        "routed_experts",
    ):
        assert torch.equal(tq_calls[field], legacy[field])
    assert tq_calls["input_ids"].shape == (3, 6)


def test_jagged_tq_selected_calls_match_dense_expansion() -> None:
    segment_lengths = [[3, 2], [4]]
    dense = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 20, 21], [30, 31, 32, 33, 0]]),
            "input_lengths": torch.tensor([5, 4]),
            "token_mask": torch.tensor([[0, 1, 1, 0, 1], [0, 1, 1, 1, 0]]),
            "routed_experts": torch.arange(2 * 5 * 3 * 2, dtype=torch.int16).reshape(
                2, 5, 3, 2
            ),
        }
    )
    jagged = BatchedDataDict(
        {
            "input_ids": torch.nested.nested_tensor(
                [dense["input_ids"][0, :5], dense["input_ids"][1, :4]],
                layout=torch.jagged,
            ),
            "input_lengths": dense["input_lengths"],
            "token_mask": torch.nested.nested_tensor(
                [dense["token_mask"][0, :5], dense["token_mask"][1, :4]],
                layout=torch.jagged,
            ),
            "routed_experts": torch.nested.nested_tensor(
                [
                    dense["routed_experts"][0, :5],
                    dense["routed_experts"][1, :4],
                ],
                layout=torch.jagged,
            ),
        }
    )

    selected = [(1, 0), (0, 1)]
    expected = expand_selected_packed_attention_segments(
        dense,
        segment_lengths=segment_lengths,
        selected_segments=selected,
        output_sequence_length=4,
    )
    actual = expand_selected_packed_attention_segments(
        jagged,
        segment_lengths=segment_lengths,
        selected_segments=selected,
        output_sequence_length=4,
    )

    for field in ("input_ids", "input_lengths", "token_mask", "routed_experts"):
        assert torch.equal(actual[field], expected[field])


def test_split_packed_attention_metas_preserves_packer_bins() -> None:
    meta = KVBatchMeta(
        partition_id="train",
        task_name="prev_lp",
        sample_ids=["rollout-0", "rollout-1", "rollout-2"],
        fields=list(DP_TRAIN_FIELDS),
        sequence_lengths=[8, 7, 6],
        extra_info={
            PACKED_ATTENTION_SEGMENT_LENGTHS: [[5, 3], [7], [4, 2]],
        },
    )
    rank_metas, _ = shard_meta_for_dp(
        meta,
        dp_world=1,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "max_tokens_per_microbatch": 10,
            "sequence_length_pad_multiple": 1,
            "microbatch_order": "largest_first",
        },
    )
    rank_meta = rank_metas[0]
    micro_metas = split_packed_attention_microbatch_metas(rank_meta)

    original_calls = rank_meta.extra_info[PACKED_ATTENTION_SELECTED_SEGMENTS]
    reconstructed_calls = []
    for micro_meta in micro_metas:
        assert micro_meta.extra_info[MICRO_BATCH_INDICES] == [
            [[0, len(micro_meta.extra_info[PACKED_ATTENTION_SELECTED_SEGMENTS])]]
        ]
        assert micro_meta.extra_info[ELEM_COUNTS_PER_GB] == [
            len(micro_meta.extra_info[PACKED_ATTENTION_SELECTED_SEGMENTS])
        ]
        local_calls = micro_meta.extra_info[PACKED_ATTENTION_SELECTED_SEGMENTS]
        source_parent = {
            sample_id: rank_meta.sample_ids.index(sample_id)
            for sample_id in micro_meta.sample_ids
        }
        reconstructed_calls.extend(
            (source_parent[micro_meta.sample_ids[parent]], segment)
            for parent, segment in local_calls
        )
        expected_width = max(
            rank_meta.extra_info[PACKED_ATTENTION_SEGMENT_LENGTHS][parent][segment]
            for parent, segment in reconstructed_calls[-len(local_calls) :]
        )
        assert micro_meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == expected_width

    assert reconstructed_calls == original_calls


def test_tq_tree_fragments_match_legacy_expansion() -> None:
    layouts = [
        TreeAttentionLayout(
            segment_lengths=(3, 2, 2),
            segment_parents=(-1, 0, 0),
            segment_depths=(0, 3, 3),
            edge_source_indices=(1, 3, 5),
            original_token_count=9,
        ),
        TreeAttentionLayout(
            segment_lengths=(4,),
            segment_parents=(-1,),
            segment_depths=(0,),
            edge_source_indices=(1, 2),
            original_token_count=4,
        ),
    ]
    logical = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[10, 11, 12, 20, 21, 30, 31], [40, 41, 42, 43, 0, 0, 0]]
            ),
            "input_lengths": torch.tensor([7, 4]),
            "routed_experts": torch.arange(2 * 7 * 2).reshape(2, 7, 2, 1),
            "generation_logprobs": torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [0.0, 4.0, 5.0, 0.0]]
            ),
            "token_mask": torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]]),
            "invalid_tool_call_mask": torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0]]),
            "malformed_thinking_mask": torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]]),
            "sample_mask": torch.ones(2),
            TREE_ATTENTION_EDGE_SOURCE_INDICES: torch.tensor([[1, 3, 5], [1, 2, -1]]),
            TREE_ATTENTION_EDGE_TARGET_IDS: torch.tensor(
                [[101, 102, 103], [201, 202, 0]]
            ),
            TREE_ATTENTION_EDGE_LENGTHS: torch.tensor([3, 2]),
            TREE_ATTENTION_LAYOUTS: layouts,
        }
    )
    legacy, legacy_layout = expand_batched_data_for_tree_attention(
        logical, max_physical_tokens=5, fragment_count_multiple=2
    )
    assert legacy_layout is not None

    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["rollout-0", "rollout-1"],
        fields=[key for key in logical if key != TREE_ATTENTION_LAYOUTS],
        sequence_lengths=[7, 4],
        extra_info={TREE_ATTENTION_LAYOUTS: layouts},
        tags=[{"owner": "rollout-0"}, {"owner": "rollout-1"}],
    )
    rank_metas, unsorted = shard_meta_for_dp(
        meta,
        dp_world=2,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "max_tokens_per_microbatch": 5,
            "sequence_length_pad_multiple": 1,
        },
    )

    sample_index = {"rollout-0": 0, "rollout-1": 1}
    rank_fragments = []
    dispatched_edges = []
    for rank_meta in rank_metas:
        assert rank_meta.tags == [
            {"owner": sample_id} for sample_id in rank_meta.sample_ids
        ]
        assert len(rank_meta.sample_ids) == len(set(rank_meta.sample_ids))
        selections = rank_meta.extra_info[TREE_ATTENTION_FRAGMENT_SELECTIONS]
        parent_indices = torch.tensor(
            [sample_index[sample_id] for sample_id in rank_meta.sample_ids]
        )
        fetched = BatchedDataDict(
            {
                key: value.index_select(0, parent_indices)
                if torch.is_tensor(value)
                else [value[index] for index in parent_indices.tolist()]
                for key, value in logical.items()
                if key != TREE_ATTENTION_LAYOUTS
            }
        )
        rank_fragments.append(
            materialize_tree_attention_fragments(
                fetched,
                fragments=[fragment for _, fragment in selections],
                parent_indices=[parent for parent, _ in selections],
            )
        )
        dispatched_edges.extend(fragment.edge_indices for _, fragment in selections)

        micro_metas = split_tree_attention_microbatch_metas(rank_meta)
        for micro in micro_metas:
            assert micro.tags == [
                {"owner": sample_id} for sample_id in micro.sample_ids
            ]
        assert sum(
            len(micro.extra_info[TREE_ATTENTION_FRAGMENT_SELECTIONS])
            for micro in micro_metas
        ) == len(selections)
        assert all(
            micro.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] <= 5 for micro in micro_metas
        )

    tq = BatchedDataDict.from_batches(
        rank_fragments,
        pad_value_dict={TREE_ATTENTION_EDGE_SOURCE_INDICES: -1},
    )
    if unsorted is not None:
        tq.reorder_data(unsorted)
    for field in (
        "input_ids",
        "input_lengths",
        "routed_experts",
        "generation_logprobs",
        "token_mask",
        "sample_mask",
        "invalid_tool_call_mask",
        "malformed_thinking_mask",
        TREE_ATTENTION_EDGE_SOURCE_INDICES,
        TREE_ATTENTION_EDGE_TARGET_IDS,
        TREE_ATTENTION_EDGE_LENGTHS,
    ):
        assert torch.equal(tq[field], legacy[field])
    assert tq[TREE_ATTENTION_LAYOUTS] == legacy[TREE_ATTENTION_LAYOUTS]
    assert sorted(edge for edges in dispatched_edges for edge in edges) == [
        0,
        0,
        1,
        1,
        2,
    ]


# ── meta utility helpers ──────────────────────────────────────────────


def test_kvbatchmeta_subset_filters_keys_and_seqlens():
    m = _meta(6)
    sub = m.subset([1, 3, 5])
    assert sub.sample_ids == ["k1", "k3", "k5"]
    assert sub.sequence_lengths == [11, 13, 15]
    assert sub.partition_id == m.partition_id


def test_kvbatchmeta_concat_joins_keys_and_seqlens():
    m1 = _meta(3)
    m2 = _meta(6).subset([3, 4, 5])
    j = m1.concat(m2)
    assert j.sample_ids == ["k0", "k1", "k2", "k3", "k4", "k5"]
    assert j.sequence_lengths == [10, 11, 12, 13, 14, 15]


def test_kvbatchmeta_transforms_keep_exact_call_layout_aligned():
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["a", "b", "c"],
        sequence_lengths=[5, 4, 6],
        extra_info={PACKED_ATTENTION_SEGMENT_LENGTHS: [[3, 2], [4], [1, 5]]},
    )

    subset = meta.subset([2, 0])
    assert subset.extra_info[PACKED_ATTENTION_SEGMENT_LENGTHS] == [[1, 5], [3, 2]]

    joined = meta.slice(0, 1).concat(meta.slice(1, 3))
    assert joined.extra_info[PACKED_ATTENTION_SEGMENT_LENGTHS] == [
        [3, 2],
        [4],
        [1, 5],
    ]


def test_kvbatchmeta_slice_takes_range():
    m = _meta(5)
    s = m.slice(1, 4)
    assert s.sample_ids == ["k1", "k2", "k3"]
    assert s.sequence_lengths == [11, 12, 13]


def test_kvbatchmeta_concat_rejects_partition_mismatch():
    import pytest

    m1 = _meta(2)
    m2 = KVBatchMeta(
        partition_id="other",
        task_name="train",
        sample_ids=["x", "y"],
        fields=None,
        sequence_lengths=[1, 2],
    )
    with pytest.raises(ValueError, match=r"partition_ids must match"):
        m1.concat(m2)


# ── Realistic multimodal extras via the rollout-shapes helper ──


def test_kv_first_write_realistic_multimodal_round_trip() -> None:
    """VLM extras (pixel_values bf16, image_grid_thw int64) flow through
    the wire as flat top-level fields and come back intact."""

    n = 4
    batch = make_rollout_batch(n=n, max_seqlen=64, multimodal=True, seed=33)
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=[
            "input_ids",
            "input_lengths",
            "sample_mask",
            "pixel_values",
            "image_grid_thw",
        ],
        num_samples=n,
        consumer_tasks=["train"],
    )
    final = BatchedDataDict(
        {
            "input_ids": batch["input_ids"],
            "input_lengths": batch["input_lengths"],
            "sample_mask": batch["sample_mask"],
            "pixel_values": batch["pixel_values"],
            "image_grid_thw": batch["image_grid_thw"],
        }
    )
    meta = kv_first_write(
        final,
        sample_ids=[f"u{i}" for i in range(n)],
        dp_client=client,
        partition_id="train",
    )
    out = read_columns(client, meta, select_fields=["pixel_values", "image_grid_thw"])
    # bf16 pixel_values + int64 image_grid_thw survive the wire intact.
    assert out["pixel_values"].dtype == torch.bfloat16
    assert out["image_grid_thw"].dtype == torch.long
    assert out["pixel_values"].shape[0] == n
