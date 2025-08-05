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

from nemo_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    DynamicBatchingArgs,
    SequencePackingArgs,
)


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


def test_sequence_packing_basic():
    """Test basic functionality of sequence packing with modified FFD algorithm."""
    # Create sample data with varying sequence lengths
    batch_size = 8
    max_seq_length = 512

    # Generate random sequence lengths between 50 and 400
    torch.manual_seed(42)
    sequence_lengths = torch.randint(50, 400, (batch_size,))

    # Create input tensors with padding
    input_ids = []
    for seq_len in sequence_lengths:
        # Create a sequence with actual tokens up to seq_len, then padding
        seq = torch.cat(
            [
                torch.randint(1, 1000, (seq_len,)),  # Actual tokens
                torch.zeros(max_seq_length - seq_len, dtype=torch.long),  # Padding
            ]
        )
        input_ids.append(seq)

    input_ids = torch.stack(input_ids)

    # Create batch data dict
    batch_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "sequence_lengths": sequence_lengths,
            "problem_ids": torch.arange(batch_size),
        }
    )

    # Configure sequence packing
    sequence_packing_args = SequencePackingArgs(
        max_tokens_per_microbatch=1024,
        input_key="input_ids",
        input_lengths_key="sequence_lengths",
        algorithm="modified_first_fit_decreasing",
        sequence_length_pad_multiple=1,
    )

    # Shard the batch with sequence packing
    shards = 2
    sharded_batches, sorted_indices = batch_data.shard_by_batch_size(
        shards=shards, sequence_packing_args=sequence_packing_args
    )

    # Verify output structure
    assert len(sharded_batches) == shards
    assert len(sorted_indices) == batch_size

    # Verify each shard has microbatch indices and lengths
    for shard in sharded_batches:
        assert hasattr(shard, "micro_batch_indices")
        assert hasattr(shard, "micro_batch_lengths")
        assert len(shard.micro_batch_indices) > 0
        assert len(shard.micro_batch_lengths) > 0

        problem_ids_seen = set()

        # Verify microbatch structure
        for chunk_indices, chunk_lengths in zip(
            shard.micro_batch_indices, shard.micro_batch_lengths
        ):
            assert len(chunk_indices) == len(chunk_lengths)

            # Verify each microbatch respects the token limit
            for (start_idx, end_idx), packed_len in zip(chunk_indices, chunk_lengths):
                assert packed_len <= sequence_packing_args["max_tokens_per_microbatch"]

        for s in sharded_batches:
            for mb in s.make_microbatch_iterator_for_packable_sequences():
                mb_len = mb["sequence_lengths"].sum().item()
                assert mb_len <= sequence_packing_args["max_tokens_per_microbatch"]
                for i in range(mb["input_ids"].shape[0]):
                    problem_id = mb["problem_ids"][i].item()
                    assert problem_id not in problem_ids_seen, (
                        f"Problem ID {problem_id} seen twice"
                    )
                    problem_ids_seen.add(problem_id)
        assert len(problem_ids_seen) == batch_size


def test_sequence_packing_uniform_lengths():
    """Test sequence packing when all sequences have the same length."""
    batch_size = 16
    seq_length = 256

    batch_data = BatchedDataDict(
        {
            "input_ids": torch.ones(batch_size, seq_length, dtype=torch.long),
            "sequence_lengths": torch.full((batch_size,), seq_length),
            "problem_ids": torch.arange(batch_size),
        }
    )

    sequence_packing_args = SequencePackingArgs(
        max_tokens_per_microbatch=1024,
        input_key="input_ids",
        input_lengths_key="sequence_lengths",
        algorithm="modified_first_fit_decreasing",
        sequence_length_pad_multiple=1,
    )

    sharded_batches, sorted_indices = batch_data.shard_by_batch_size(
        shards=2, sequence_packing_args=sequence_packing_args
    )

    # With uniform lengths, sequences should be efficiently packed
    assert len(sharded_batches) == 2
    len_0 = len(
        list(sharded_batches[0].make_microbatch_iterator_for_packable_sequences())
    )
    len_1 = len(
        list(sharded_batches[1].make_microbatch_iterator_for_packable_sequences())
    )
    assert len_0 + len_1 == 4
    assert min(len_0, len_1) == 2

    # Each microbatch should pack as many sequences as possible
    for shard in sharded_batches:
        for chunk_indices, chunk_lengths in zip(
            shard.micro_batch_indices, shard.micro_batch_lengths
        ):
            for (start_idx, end_idx), packed_len in zip(chunk_indices, chunk_lengths):
                # With 256 tokens per sequence and 1024 max, should pack 4 sequences
                assert packed_len <= 1024
                num_seqs = end_idx - start_idx
                assert num_seqs <= 4  # Can fit at most 4 sequences of length 256

    problem_ids_seen = set()
    for s in sharded_batches:
        for mb in s.make_microbatch_iterator_for_packable_sequences():
            mb_len = mb["sequence_lengths"].sum().item()
            assert mb_len <= sequence_packing_args["max_tokens_per_microbatch"]
            for i in range(mb["input_ids"].shape[0]):
                problem_id = mb["problem_ids"][i].item()
                assert problem_id not in problem_ids_seen, (
                    f"Problem ID {problem_id} seen twice"
                )
                problem_ids_seen.add(problem_id)
    assert len(problem_ids_seen) == batch_size


def test_sequence_packing_long_sequences():
    """Test sequence packing with very long sequences that require individual microbatches."""
    batch_size = 4

    batch_data = BatchedDataDict(
        {
            "input_ids": torch.ones(batch_size, 2048, dtype=torch.long),
            "sequence_lengths": torch.tensor([900, 850, 1000, 950]),
            "problem_ids": torch.arange(batch_size),
        }
    )

    sequence_packing_args = SequencePackingArgs(
        max_tokens_per_microbatch=1024,
        input_key="input_ids",
        input_lengths_key="sequence_lengths",
        algorithm="modified_first_fit_decreasing",
        sequence_length_pad_multiple=1,
    )

    sharded_batches, sorted_indices = batch_data.shard_by_batch_size(
        shards=2, sequence_packing_args=sequence_packing_args
    )

    # Each sequence should be in its own microbatch due to length
    for shard in sharded_batches:
        for chunk_indices, chunk_lengths in zip(
            shard.micro_batch_indices, shard.micro_batch_lengths
        ):
            for (start_idx, end_idx), max_len in zip(chunk_indices, chunk_lengths):
                num_seqs = end_idx - start_idx
                # Each long sequence should be alone in its microbatch
                assert num_seqs == 1

    problem_ids_seen = set()
    for s in sharded_batches:
        for mb in s.make_microbatch_iterator_for_packable_sequences():
            mb_len = mb["sequence_lengths"].sum().item()
            assert mb_len <= sequence_packing_args["max_tokens_per_microbatch"]
            for i in range(mb["input_ids"].shape[0]):
                problem_id = mb["problem_ids"][i].item()
                assert problem_id not in problem_ids_seen, (
                    f"Problem ID {problem_id} seen twice"
                )
                problem_ids_seen.add(problem_id)
    assert len(problem_ids_seen) == batch_size


def test_sequence_packing_with_dynamic_batching_conflict():
    """Test that sequence packing and dynamic batching cannot be used together."""
    batch_data = BatchedDataDict(
        {
            "input_ids": torch.ones(4, 100, dtype=torch.long),
            "sequence_lengths": torch.tensor([50, 60, 70, 80]),
        }
    )

    sequence_packing_args = SequencePackingArgs(
        max_tokens_per_microbatch=1024,
        input_key="input_ids",
        input_lengths_key="sequence_lengths",
        algorithm="modified_first_fit_decreasing",
    )

    dynamic_batching_args: DynamicBatchingArgs = {
        "input_key": "input_ids",
        "input_lengths_key": "sequence_lengths",
        "sequence_length_round": 4,
        "max_tokens_per_microbatch": 1024,
    }

    with pytest.raises(
        AssertionError,
        match="dynamic_batching_args and sequence_packing_args cannot be passed together",
    ):
        batch_data.shard_by_batch_size(
            shards=2,
            sequence_packing_args=sequence_packing_args,
            dynamic_batching_args=dynamic_batching_args,
        )


@pytest.mark.parametrize("pad_to_multiple_of", [1, 32, 64, 256])
def test_sequence_packing_microbatch_boundaries(pad_to_multiple_of):
    """Test that microbatch boundaries are correctly maintained across chunks with random sequences."""
    # Create a large batch with random sequence lengths to test boundary handling
    torch.manual_seed(123)  # For reproducible tests
    batch_size = 1024
    num_global_batches = 4
    max_seq_length = 1024
    max_tokens_per_microbatch = 1200

    def _get_padded_seqlen(seqlen: int) -> int:
        return (seqlen + (pad_to_multiple_of - 1)) // pad_to_multiple_of

    # Generate random sequence lengths with good variety
    sequence_lengths = torch.randint(50, 800, (batch_size,))

    # Create input tensors with padding
    input_ids = []
    for i, seq_len in enumerate(sequence_lengths):
        # Create a sequence with actual tokens up to seq_len, then padding
        seq = torch.cat(
            [
                torch.randint(1, 1000, (seq_len,)),  # Actual tokens
                torch.zeros(max_seq_length - seq_len, dtype=torch.long),  # Padding
            ]
        )
        input_ids.append(seq)

    input_ids = torch.stack(input_ids)

    batch_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "sequence_lengths": sequence_lengths,
            "problem_ids": torch.arange(batch_size),
        }
    )

    sequence_packing_args = SequencePackingArgs(
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        input_key="input_ids",
        input_lengths_key="sequence_lengths",
        algorithm="modified_first_fit_decreasing",
        sequence_length_pad_multiple=pad_to_multiple_of,
    )

    # Test with multiple shards and explicit batch_size to create chunks
    shards = 4
    chunk_batch_size = batch_size // num_global_batches
    sharded_batches, sorted_indices = batch_data.shard_by_batch_size(
        shards=shards,
        batch_size=chunk_batch_size,
        sequence_packing_args=sequence_packing_args,
    )

    # Verify output structure
    assert len(sharded_batches) == shards
    assert len(sorted_indices) == batch_size

    # Track all problem IDs to ensure completeness and no duplicates
    problem_ids_seen = set()

    for gb_idx in range(num_global_batches):
        mb_count_for_gb = 0
        min_mb_count = 100000000  # arbitrary large number
        max_mb_count = 0
        legal_problem_ids = set(
            range(gb_idx * chunk_batch_size, (gb_idx + 1) * chunk_batch_size)
        )
        for shard_idx in range(shards):
            shard_batch = sharded_batches[shard_idx].get_batch(gb_idx)
            mb_count = 0
            for mb in shard_batch.make_microbatch_iterator_for_packable_sequences():
                mb_count += 1
                for i in range(mb["input_ids"].shape[0]):
                    problem_id = mb["problem_ids"][i].item()
                    assert problem_id in legal_problem_ids, (
                        f"Problem ID {problem_id} not in legal problem IDs"
                    )
                    assert problem_id not in problem_ids_seen, (
                        f"Problem ID {problem_id} seen twice"
                    )
                    problem_ids_seen.add(problem_id)
                assert (
                    _get_padded_seqlen(mb["sequence_lengths"]).sum().item()
                    <= max_tokens_per_microbatch
                ), (
                    f"Sequence length {_get_padded_seqlen(mb['sequence_lengths']).sum().item()} is greater than max tokens per microbatch {max_tokens_per_microbatch}"
                )

            min_mb_count = min(min_mb_count, mb_count)
            max_mb_count = max(max_mb_count, mb_count)
            mb_count_for_gb += mb_count
        assert max_mb_count - min_mb_count <= 1

        num_actual_tokens = sum(
            sequence_lengths[
                gb_idx * chunk_batch_size : (gb_idx + 1) * chunk_batch_size
            ]
        )
        packing_efficiency = num_actual_tokens / (
            mb_count_for_gb * max_tokens_per_microbatch
        )

        pack_efficiency_standards = {
            1: (0.955, 1.0),
            32: (0.91, 0.97),
            64: (0.85, 0.92),
            256: (0.60, 0.80),
        }
        assert packing_efficiency >= pack_efficiency_standards[pad_to_multiple_of][0], (
            f"We expect packing efficiency to be above {pack_efficiency_standards[pad_to_multiple_of][0]} for these nice random inputs with padding to multiples of {pad_to_multiple_of}. Got {packing_efficiency}"
        )
        assert packing_efficiency <= pack_efficiency_standards[pad_to_multiple_of][1], (
            f"We expect packing efficiency to be below {pack_efficiency_standards[pad_to_multiple_of][1]} for these nice random inputs with padding to multiples of {pad_to_multiple_of}. Got {packing_efficiency}"
        )

    assert len(problem_ids_seen) == batch_size

    # Finally, test that we can reorder everything back to how it was before
    reconstructed = BatchedDataDict.from_batches(sharded_batches)
    # check that it's different from the original
    assert not torch.all(reconstructed["problem_ids"] == batch_data["problem_ids"])
    assert not torch.all(reconstructed["input_ids"] == batch_data["input_ids"])
    assert not torch.all(
        reconstructed["sequence_lengths"] == batch_data["sequence_lengths"]
    )

    reconstructed.reorder_data(sorted_indices)
    # check that it's the same as the original
    assert torch.all(reconstructed["problem_ids"] == batch_data["problem_ids"])
    assert torch.all(reconstructed["input_ids"] == batch_data["input_ids"])
    assert torch.all(
        reconstructed["sequence_lengths"] == batch_data["sequence_lengths"]
    )
