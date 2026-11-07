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

import pytest
import torch

from nemo_rl.data.packed_rollouts import (
    PACKED_ATTENTION_SEGMENT_LENGTHS,
    TREE_ATTENTION_EDGE_LENGTHS,
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_LAYOUTS,
    TreeAttentionLayout,
    expand_batched_data_for_packed_attention,
    expand_batched_data_for_tree_attention,
    expand_selected_packed_attention_segments,
    plan_tree_attention_fragments,
    reassemble_packed_attention_segments,
    reassemble_tree_attention_edge_values,
    split_tensor_at_packed_attention_segments,
    validate_packed_attention_segment_lengths,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_split_and_reassemble_independent_attention_segments() -> None:
    logical_rows = torch.tensor(
        [
            [10, 11, 12, 20, 21],
            [30, 31, 32, 33, 0],
        ]
    )
    segment_lengths = [[3, 2], [4]]

    segments, lengths = split_tensor_at_packed_attention_segments(
        logical_rows, segment_lengths
    )

    assert torch.equal(lengths, torch.tensor([3, 2, 4]))
    assert torch.equal(
        segments,
        torch.tensor(
            [
                [10, 11, 12, 0],
                [20, 21, 0, 0],
                [30, 31, 32, 33],
            ]
        ),
    )
    assert torch.equal(
        reassemble_packed_attention_segments(
            segments, segment_lengths, output_sequence_length=5
        ),
        logical_rows,
    )


def test_expand_batched_data_creates_independently_packable_call_rows() -> None:
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [10, 11, 12, 20, 21],
                    [30, 31, 32, 33, 0],
                ]
            ),
            "input_lengths": torch.tensor([5, 4]),
            "token_mask": torch.tensor(
                [
                    [0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 0.5]),
            "problem_id": ["first", "second"],
            PACKED_ATTENTION_SEGMENT_LENGTHS: [[3, 2], [4]],
        }
    )

    expanded, layout = expand_batched_data_for_packed_attention(data)

    assert layout is not None
    assert layout.segment_lengths == [[3, 2], [4]]
    assert layout.output_sequence_length == 5
    assert PACKED_ATTENTION_SEGMENT_LENGTHS not in expanded
    assert torch.equal(expanded["input_lengths"], torch.tensor([3, 2, 4]))
    assert torch.equal(
        expanded["input_ids"],
        torch.tensor(
            [
                [10, 11, 12, 0],
                [20, 21, 0, 0],
                [30, 31, 32, 33],
            ]
        ),
    )
    assert torch.equal(
        expanded["token_mask"],
        torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 1],
            ]
        ),
    )
    assert torch.equal(expanded["sample_mask"], torch.tensor([1.0, 1.0, 0.5]))
    assert expanded["problem_id"] == ["first", "first", "second"]


def test_expanded_calls_can_be_sequence_packed_across_uneven_dp_shards() -> None:
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [10, 11, 12, 20, 21],
                    [30, 31, 32, 33, 0],
                ]
            ),
            "input_lengths": torch.tensor([5, 4]),
            PACKED_ATTENTION_SEGMENT_LENGTHS: [[3, 2], [4]],
        }
    )
    expanded, _ = expand_batched_data_for_packed_attention(data)

    shards, ordered_indices = expanded.shard_by_batch_size(
        shards=2,
        batch_size=None,
        allow_uneven_shards=True,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "max_tokens_per_microbatch": 5,
            "sequence_length_pad_multiple": 1,
        },
    )

    assert sorted(ordered_indices) == [0, 1, 2]
    assert sum(shard.size for shard in shards) == 3
    assert all(shard.elem_counts_per_gb is not None for shard in shards)


def test_expand_selected_segments_preserves_dp_execution_order() -> None:
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [10, 11, 12, 20, 21],
                    [30, 31, 32, 33, 0],
                ]
            ),
            "input_lengths": torch.tensor([5, 4]),
            "token_mask": torch.tensor(
                [
                    [0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 0.5]),
        }
    )

    selected = expand_selected_packed_attention_segments(
        data,
        segment_lengths=[[3, 2], [4]],
        selected_segments=[(1, 0), (0, 1)],
    )

    assert torch.equal(selected["input_lengths"], torch.tensor([4, 2]))
    assert torch.equal(
        selected["input_ids"],
        torch.tensor(
            [
                [30, 31, 32, 33],
                [20, 21, 0, 0],
            ]
        ),
    )
    assert torch.equal(selected["sample_mask"], torch.tensor([0.5, 1.0]))
    assert torch.equal(
        selected["token_mask"],
        torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 1, 0, 0],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("segment_lengths", "input_lengths", "message"),
    [
        ([[2]], [2, 1], "one entry per rollout row"),
        ([[0]], [0], "positive integer"),
        ([[1, 2]], [4], "cover the rollout's unpadded tokens"),
    ],
)
def test_validate_packed_attention_segment_lengths_rejects_invalid_metadata(
    segment_lengths: list[list[int]], input_lengths: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_packed_attention_segment_lengths(segment_lengths, input_lengths)


def _branching_tree_layout() -> TreeAttentionLayout:
    return TreeAttentionLayout(
        segment_lengths=(3, 2, 2),
        segment_parents=(-1, 0, 0),
        segment_depths=(0, 3, 3),
        edge_source_indices=(1, 3, 5),
        original_token_count=9,
    )


def test_plan_tree_fragments_is_bounded_ancestor_closed_and_edge_disjoint() -> None:
    fragments = plan_tree_attention_fragments(
        _branching_tree_layout(), max_physical_tokens=5
    )

    assert len(fragments) == 2
    assert [fragment.node_indices for fragment in fragments] == [
        (0, 1, 2, 3, 4),
        (0, 1, 2, 5, 6),
    ]
    assert [fragment.edge_indices for fragment in fragments] == [(0, 1), (2,)]
    assert [fragment.layout.segment_parents for fragment in fragments] == [
        (-1, 0),
        (-1, 0),
    ]
    assert [fragment.layout.edge_source_indices for fragment in fragments] == [
        (1, 3),
        (3,),
    ]
    assert all(fragment.layout.unique_token_count <= 5 for fragment in fragments)


def test_tree_fragment_expansion_gathers_nodes_and_edges_then_reassembles() -> None:
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 20, 21, 30, 31]]),
            "input_lengths": torch.tensor([7]),
            "routed_experts": torch.arange(14).reshape(1, 7, 2, 1),
            "generation_logprobs": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            "token_mask": torch.tensor([[0, 1, 1, 1]]),
            "sample_mask": torch.tensor([0.5]),
            TREE_ATTENTION_EDGE_SOURCE_INDICES: torch.tensor([[1, 3, 5]]),
            TREE_ATTENTION_EDGE_TARGET_IDS: torch.tensor([[101, 102, 103]]),
            TREE_ATTENTION_EDGE_LENGTHS: torch.tensor([3]),
            TREE_ATTENTION_LAYOUTS: [_branching_tree_layout()],
        }
    )

    expanded, layout = expand_batched_data_for_tree_attention(
        data, max_physical_tokens=5
    )

    assert layout is not None
    assert expanded.size == 2
    assert torch.equal(expanded["input_lengths"], torch.tensor([5, 5]))
    assert torch.equal(
        expanded["input_ids"],
        torch.tensor(
            [
                [10, 11, 12, 20, 21],
                [10, 11, 12, 30, 31],
            ]
        ),
    )
    assert torch.equal(
        expanded[TREE_ATTENTION_EDGE_SOURCE_INDICES],
        torch.tensor([[1, 3], [3, -1]]),
    )
    assert torch.equal(
        expanded[TREE_ATTENTION_EDGE_TARGET_IDS],
        torch.tensor([[101, 102], [103, 0]]),
    )
    assert torch.equal(expanded[TREE_ATTENTION_EDGE_LENGTHS], torch.tensor([2, 1]))
    assert torch.equal(
        expanded["generation_logprobs"],
        torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 0.0]]),
    )
    assert torch.equal(expanded["sample_mask"], torch.tensor([0.5, 0.5]))
    assert torch.equal(
        expanded["routed_experts"][1], data["routed_experts"][0, [0, 1, 2, 5, 6]]
    )

    fragment_values = torch.tensor([[0.0, 10.0, 20.0], [0.0, 30.0, 0.0]])
    assert torch.equal(
        reassemble_tree_attention_edge_values(fragment_values, layout),
        torch.tensor([[0.0, 10.0, 20.0, 30.0]]),
    )


def test_tree_fragment_plan_rejects_a_path_larger_than_budget() -> None:
    with pytest.raises(ValueError, match="logical path exceeds"):
        plan_tree_attention_fragments(_branching_tree_layout(), max_physical_tokens=4)
