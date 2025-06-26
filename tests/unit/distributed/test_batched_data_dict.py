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
import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, DynamicBatchingArgs


def test_shard_by_batch_size_basic():
    """Test basic functionality of shard_by_batch_size with tensor data."""
    # Create a sample batch with tensor data
    batch = BatchedDataDict(
        {
            "tensor_data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            "other_tensor": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
        }
    )

    # Shard with batch_size=4, shards=2
    sharded = batch.shard_by_batch_size(shards=2, batch_size=4)

    # Verify output structure
    assert len(sharded) == 2, f"Expected 2 shards, got {len(sharded)}"

    # Verify first shard content (first elements of each chunk)
    assert torch.equal(sharded[0]["tensor_data"], torch.tensor([0, 1, 4, 5]))
    assert torch.equal(sharded[0]["other_tensor"], torch.tensor([10, 11, 14, 15]))

    # Verify second shard content (second elements of each chunk)
    assert torch.equal(sharded[1]["tensor_data"], torch.tensor([2, 3, 6, 7]))
    assert torch.equal(sharded[1]["other_tensor"], torch.tensor([12, 13, 16, 17]))


def test_shard_by_batch_size_list_data():
    """Test shard_by_batch_size with list data."""
    # Create a sample batch with list data
    batch = BatchedDataDict(
        {
            "list_data": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "tensor_data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        }
    )

    # Shard with batch_size=4, shards=2
    sharded = batch.shard_by_batch_size(shards=2, batch_size=4)

    # Verify output structure
    assert len(sharded) == 2

    # Verify first shard content
    assert sharded[0]["list_data"] == ["A", "B", "E", "F"]
    assert torch.equal(sharded[0]["tensor_data"], torch.tensor([0, 1, 4, 5]))

    # Verify second shard content
    assert sharded[1]["list_data"] == ["C", "D", "G", "H"]
    assert torch.equal(sharded[1]["tensor_data"], torch.tensor([2, 3, 6, 7]))


def test_shard_by_batch_size_larger_example():
    """Test shard_by_batch_size with a larger example with multiple chunks and shards."""
    # Create a batch with 12 elements
    batch = BatchedDataDict(
        {"tensor_data": torch.arange(12), "list_data": [f"item_{i}" for i in range(12)]}
    )

    # Shard with batch_size=3, shards=3
    sharded = batch.shard_by_batch_size(shards=3, batch_size=3)

    # Verify we get 3 shards
    assert len(sharded) == 3

    # Expected results:
    # Chunk 1: [0, 1, 2], Chunk 2: [3, 4, 5], Chunk 3: [6, 7, 8], Chunk 4: [9, 10, 11]
    # Shard 1: [0, 3, 6, 9]
    # Shard 2: [1, 4, 7, 10]
    # Shard 3: [2, 5, 8, 11]

    # Verify tensor content
    assert torch.equal(sharded[0]["tensor_data"], torch.tensor([0, 3, 6, 9]))
    assert torch.equal(sharded[1]["tensor_data"], torch.tensor([1, 4, 7, 10]))
    assert torch.equal(sharded[2]["tensor_data"], torch.tensor([2, 5, 8, 11]))

    # Verify list content
    assert sharded[0]["list_data"] == ["item_0", "item_3", "item_6", "item_9"]
    assert sharded[1]["list_data"] == ["item_1", "item_4", "item_7", "item_10"]
    assert sharded[2]["list_data"] == ["item_2", "item_5", "item_8", "item_11"]


def test_shard_by_batch_size_2d_tensor():
    """Test shard_by_batch_size with 2D tensor data."""
    # Create a batch with 2D tensors
    batch = BatchedDataDict(
        {
            "features": torch.tensor(
                [
                    [1, 2, 3],  # 0
                    [4, 5, 6],  # 1
                    [7, 8, 9],  # 2
                    [10, 11, 12],  # 3
                    [13, 14, 15],  # 4
                    [16, 17, 18],  # 5
                ]
            )
        }
    )

    # Shard with batch_size=3, shards=3
    sharded = batch.shard_by_batch_size(shards=3, batch_size=3)

    # Verify we get 3 shards
    assert len(sharded) == 3

    # Expected results by index:
    # Chunk 1: [0, 1, 2], Chunk 2: [3, 4, 5]
    # Shard 1: [0, 3]
    # Shard 2: [1, 4]
    # Shard 3: [2, 5]

    # Verify tensor content
    expected_0 = torch.tensor([[1, 2, 3], [10, 11, 12]])
    expected_1 = torch.tensor([[4, 5, 6], [13, 14, 15]])
    expected_2 = torch.tensor([[7, 8, 9], [16, 17, 18]])

    assert torch.equal(sharded[0]["features"], expected_0)
    assert torch.equal(sharded[1]["features"], expected_1)
    assert torch.equal(sharded[2]["features"], expected_2)


def test_shard_by_batch_size_edge_cases():
    """Test edge cases for shard_by_batch_size."""
    # Case 1: Single batch, multiple shards
    batch = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3])})

    sharded = batch.shard_by_batch_size(shards=2, batch_size=4)
    assert len(sharded) == 2
    assert torch.equal(sharded[0]["data"], torch.tensor([0, 1]))
    assert torch.equal(sharded[1]["data"], torch.tensor([2, 3]))

    # Case 2: Multiple batches, single shard
    batch = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])})

    sharded = batch.shard_by_batch_size(shards=1, batch_size=2)
    assert len(sharded) == 1
    assert torch.equal(sharded[0]["data"], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))


def test_shard_by_batch_size_validation():
    """Test validation checks in shard_by_batch_size."""
    # Create a batch
    batch = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])})

    # Case 1: batch_size not a divisor of total_batch_size
    with pytest.raises(
        AssertionError, match="Total batch size.*is not a multiple of batch_size"
    ):
        batch.shard_by_batch_size(shards=2, batch_size=3)

    # Case 2: shards not a divisor of batch_size
    # First make a batch that's divisible by batch_size to reach the second assertion
    batch_for_case2 = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])})
    with pytest.raises(AssertionError, match="Batch size.*is not a multiple of shards"):
        batch_for_case2.shard_by_batch_size(shards=3, batch_size=4)

    # Case 3: Different batch sizes across keys
    inconsistent_batch = BatchedDataDict(
        {
            "data1": torch.tensor([0, 1, 2, 3]),
            "data2": torch.tensor([0, 1, 2]),
        }  # Different length
    )

    with pytest.raises(
        AssertionError, match="Batch sizes are not the same across the rollout batch"
    ):
        inconsistent_batch.shard_by_batch_size(shards=2, batch_size=2)


def test_shard_by_batch_size_matches_example():
    """Test that shard_by_batch_size behaves as described in the docstring example."""
    # Create the example data: [A A B B C C D D]
    batch = BatchedDataDict({"data": ["A", "A", "B", "B", "C", "C", "D", "D"]})

    # Shard with batch_size=2, shards=2
    sharded = batch.shard_by_batch_size(shards=2, batch_size=2)

    # Verify output structure
    assert len(sharded) == 2

    # Expected output:
    # Element 0: [A B C D] (first elements from each chunk)
    # Element 1: [A B C D] (second elements from each chunk)
    assert sharded[0]["data"] == ["A", "B", "C", "D"]
    assert sharded[1]["data"] == ["A", "B", "C", "D"]


def test_shard_by_batch_size_dynamic():
    # create a data dict with variable sequence lengths per datum
    batch = BatchedDataDict(
        {
            "data": torch.ones([8, 128]),
            "sequence_lengths": torch.tensor(
                (2, 8, 4, 16, 28, 32, 2, 32), dtype=torch.int
            ),
        }
    )
    dynamic_batching_args: DynamicBatchingArgs = {
        "input_key": "data",
        "input_lengths_key": "sequence_lengths",
        "sequence_length_round": 4,
        "max_tokens_per_microbatch": 32,
    }

    shards, _ = batch.shard_by_batch_size(
        shards=2, dynamic_batching_args=dynamic_batching_args
    )
    # Expected Output: 3 microbatches per shard, of sizes 2, 1, 1
    for shard in shards:
        shard.micro_batch_indices == [[[0, 2], [2, 3], [3, 4]]]

    # test creating dynamic micro_batch iterators
    for shard in shards:
        mb_iterator = shard.make_microbatch_iterator_with_dynamic_shapes()
        # check each microbatch has a valid dynamic sequence length
        for mb in mb_iterator:
            batch_size, seqlen = mb["data"].shape
            assert seqlen % 4 == 0
            assert seqlen <= 32
