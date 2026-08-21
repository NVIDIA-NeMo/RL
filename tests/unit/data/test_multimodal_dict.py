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
from copy import deepcopy

import pytest
import ray.cloudpickle as cloudpickle
import torch

from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
)
from nemo_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    DynamicBatchingArgs,
    SequencePackingArgs,
)


def test_packed_data_basic():
    """Test basic functionality of PackedTensor."""
    # Create sample packed items
    tensor1 = torch.randn(16, 3)
    tensor2 = torch.randn(45, 3)

    item1 = PackedTensor(tensor1, dim_to_pack=0)
    item2 = PackedTensor(tensor2, dim_to_pack=0)

    # Test item functionality
    assert torch.equal(item1.as_tensor(), tensor1)
    assert item1.dim_to_pack == 0

    # Test batch creation and concatenation
    batch = PackedTensor([item1.as_tensor(), item2.as_tensor()], dim_to_pack=0)
    assert len(batch) == 2

    # Test as_tensor
    expected_tensor = torch.cat([tensor1, tensor2], dim=0)
    assert torch.equal(batch.as_tensor(), expected_tensor)


def test_shard_by_batch_size_with_packed_data():
    """Test shard_by_batch_size with packed multimodal data."""
    # Create sample data
    text_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    image_tensors = [torch.randn(3 * i + 2, 3, 128, 128) for i in range(4)]

    # Create packed image data
    packed_batch = PackedTensor(image_tensors, dim_to_pack=0)

    # Create BatchedDataDict
    batch = BatchedDataDict(
        {
            "text_ids": text_tensor,
            "image_features": packed_batch,
            "labels": [1, 2, 3, 4],
        }
    )

    # Test sharding
    shards = batch.shard_by_batch_size(shards=2)
    assert len(shards) == 2

    # Verify first shard
    assert torch.equal(shards[0]["text_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert isinstance(shards[0]["image_features"], PackedTensor)
    assert len(shards[0]["image_features"]) == 2
    assert shards[0]["image_features"].as_tensor().shape == (2 + 5, 3, 128, 128)
    assert shards[0]["labels"] == [1, 2]

    # Verify second shard
    assert torch.equal(shards[1]["text_ids"], torch.tensor([[7, 8, 9], [10, 11, 12]]))
    assert isinstance(shards[1]["image_features"], PackedTensor)
    assert len(shards[1]["image_features"]) == 2
    assert shards[1]["image_features"].as_tensor().shape == (8 + 11, 3, 128, 128)
    assert shards[1]["labels"] == [3, 4]


def test_truncate_tensors_with_packed_data():
    """Test truncate_tensors with packed multimodal data."""
    # Create sample data
    text_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    image_tensors = [
        torch.randn(5, 3, 128, 4, 2, 2) for i in range(2)
    ]  # also check a different dim_to_pack

    # Create packed image data
    packed_batch = PackedTensor(image_tensors, dim_to_pack=1)

    # Create BatchedDataDict
    batch = BatchedDataDict({"text_ids": text_tensor, "image_features": packed_batch})

    # Test truncation
    batch.truncate_tensors(dim=1, truncated_len=2)

    # Verify text was truncated
    assert torch.equal(batch["text_ids"], torch.tensor([[1, 2], [5, 6]]))
    # Verify image features were not affected (assumed safe as per comment in truncate_tensors)
    assert isinstance(batch["image_features"], PackedTensor)
    assert batch["image_features"].as_tensor().shape == (5, 6, 128, 4, 2, 2)


def test_multiturn_rollout_with_packed_data():
    """Test multiturn conversations with packed multimodal data."""
    message_log_1 = [
        {
            "role": "user",
            "token_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            "images": PackedTensor(torch.randn(3, 128, 128), dim_to_pack=0),
        },
        {
            "role": "assistant",
            "token_ids": torch.tensor([9, 10, 11, 12, 13, 14, 15, 16]),
        },
        {
            "role": "user",
            "token_ids": torch.tensor([17, 18, 19, 20, 21, 22, 23, 24]),
            "images": PackedTensor(torch.randn(3, 128, 128), dim_to_pack=0),
        },
    ]
    message_log_2 = [
        {
            "role": "user",
            "token_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            "images": PackedTensor(torch.randn(3, 128, 128), dim_to_pack=0),
        },
        {
            "role": "assistant",
            "token_ids": torch.tensor([9, 10, 11, 12, 13, 14, 15, 16]),
        },
        {
            "role": "user",
            "token_ids": torch.tensor([17, 18, 19, 20, 21, 22, 23, 24]),
        },
    ]
    # data spec
    message_logs = BatchedDataDict(
        {
            "message_log": [message_log_1, message_log_2],
        }
    )
    flat_message, input_lengths = batched_message_log_to_flat_message(
        message_logs["message_log"],
        pad_value_dict={
            "token_ids": -1,
        },
    )
    shards = flat_message.shard_by_batch_size(shards=2)
    assert len(shards) == 2
    assert tuple(shards[0]["images"].as_tensor().shape) == (6, 128, 128)
    assert tuple(shards[1]["images"].as_tensor().shape) == (3, 128, 128)


def test_sequence_packing_with_packed_data():
    """Test sequence packing with packed multimodal data."""
    # Create sample data
    text_tensor = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    image_tensors = [torch.randn(2**i, 1176) for i in range(4)]

    # Create packed image data
    packed_batch = PackedTensor(image_tensors, dim_to_pack=0)

    # Create BatchedDataDict
    batch = BatchedDataDict(
        {
            "text_ids": text_tensor,
            "image_features": packed_batch,
            "sequence_lengths": torch.tensor([2, 3, 2, 4]),
        }
    )

    sequence_packing_args = SequencePackingArgs(
        max_tokens_per_microbatch=6,
        input_key="text_ids",
        input_lengths_key="sequence_lengths",
        algorithm="modified_first_fit_decreasing",
        sequence_length_pad_multiple=1,
    )

    # Test sequence packing
    sharded_batches, sorted_indices = batch.shard_by_batch_size(
        shards=2, sequence_packing_args=sequence_packing_args
    )

    # Verify basic structure
    assert len(sharded_batches) == 2
    assert len(sorted_indices) == 4

    print("sequence packing sorted indices", sorted_indices)

    # Verify each shard has the necessary attributes
    for shard in sharded_batches:
        assert hasattr(shard, "micro_batch_indices")
        assert hasattr(shard, "micro_batch_lengths")
        assert isinstance(shard["image_features"], PackedTensor)


def test_dynamic_batching_with_packed_data():
    """Test dynamic batching with packed multimodal data."""
    # Create sample data
    text_tensor = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    image_tensors = [torch.randn(2**i, 1176) for i in range(4)]

    # Create packed image data
    packed_batch = PackedTensor(image_tensors, dim_to_pack=0)

    # Create BatchedDataDict
    batch = BatchedDataDict(
        {
            "text_ids": text_tensor,
            "image_features": packed_batch,
            "sequence_lengths": torch.tensor([2, 3, 2, 4]),
        }
    )

    dynamic_batching_args: DynamicBatchingArgs = {
        "input_key": "text_ids",
        "input_lengths_key": "sequence_lengths",
        "sequence_length_round": 2,
        "max_tokens_per_microbatch": 6,
    }

    # Test dynamic batching
    sharded_batches, sorted_indices = batch.shard_by_batch_size(
        shards=2, dynamic_batching_args=dynamic_batching_args
    )

    print("dynamic batching sorted indices", sorted_indices)

    # Verify basic structure
    assert len(sharded_batches) == 2
    assert len(sorted_indices) == 4

    # Verify each shard has the necessary attributes
    for shard in sharded_batches:
        assert hasattr(shard, "micro_batch_indices")
        assert hasattr(shard, "micro_batch_lengths")
        assert isinstance(shard["image_features"], PackedTensor)


def test_multimodal_specific_functionality():
    """Test functionality specific to multimodal data handling. (length, device movement, as_tensor)"""
    # Create sample data
    text_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    image_tensor = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])

    # Test PackedTensorItem
    mm_data = PackedTensor(image_tensor, dim_to_pack=0)
    assert isinstance(mm_data, PackedTensor)
    assert torch.equal(mm_data.as_tensor(), image_tensor)
    assert len(mm_data) == 1

    # Test device movement
    if torch.cuda.is_available():
        mm_data = mm_data.to("cuda")
        assert mm_data.tensors[0].device.type == "cuda"

    # images differ along a different dimension
    image_tensors = [torch.randn(3, 128, 128 + i) for i in range(2)]

    mm_batch = PackedTensor(image_tensors, dim_to_pack=0)
    with pytest.raises(RuntimeError):
        batch_tensor = mm_batch.as_tensor()

    # check for packing on correct dimension
    image_tensors = [torch.randn(3 + 10**i, 128, 128) for i in range(2)]
    mm_batch = PackedTensor(image_tensors, dim_to_pack=0)
    mm_tensor = mm_batch.as_tensor()

    expected_dim = sum([3 + 10**i for i in range(2)])
    assert mm_tensor.shape == (expected_dim, 128, 128)


def test_get_multimodal_dict():
    """Test the get_multimodal_dict functionality."""
    # Create sample data
    text_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    image_tensor = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    token_type_ids = torch.tensor([[1, 1, 1], [1, 1, 1]])

    # Create packed image data
    packed_image = PackedTensor(image_tensor, dim_to_pack=0)

    # Create BatchedDataDict
    batch = BatchedDataDict(
        {
            "text_ids": text_tensor,
            "image_features": packed_image,
            "token_type_ids": token_type_ids,  # Special key that should be included
        }
    )

    # Test getting multimodal dict as tensors
    mm_dict = batch.get_multimodal_dict(as_tensors=True)
    assert "image_features" in mm_dict
    assert "token_type_ids" in mm_dict
    assert torch.is_tensor(mm_dict["image_features"])
    assert torch.is_tensor(mm_dict["token_type_ids"])
    assert "text_ids" not in mm_dict  # Regular tensors should not be included

    # Test getting multimodal dict as packed items
    mm_dict = batch.get_multimodal_dict(as_tensors=False)
    assert "image_features" in mm_dict
    assert "token_type_ids" in mm_dict
    assert isinstance(mm_dict["image_features"], PackedTensor)
    assert torch.is_tensor(mm_dict["token_type_ids"])


def test_packedtensor_all_none():
    pt = PackedTensor([None, None], dim_to_pack=0)
    assert pt.as_tensor() is None


def test_packedtensor_with_none_entry():
    original = PackedTensor([torch.randn(2, 3), None], dim_to_pack=0)
    empty = PackedTensor.empty_like(original)
    # same logical length
    assert len(empty) == len(original)
    # all entries are None, thus as_tensor returns None
    assert empty.as_tensor() is None


def test_packedtensor_to_with_none_entry():
    t = torch.randn(1, 2)
    pt = PackedTensor([None, t], dim_to_pack=0)
    pt = pt.to("cpu")
    assert pt.tensors[0] is None
    assert isinstance(pt.tensors[1], torch.Tensor)
    assert pt.tensors[1].device.type == "cpu"


def test_packedtensor_as_tensor_with_mixed_none_and_tensors():
    t1 = torch.randn(2, 3)
    t2 = None
    t3 = torch.randn(4, 3)
    pt = PackedTensor([t1, t2, t3], dim_to_pack=0)
    out = pt.as_tensor()
    expected = torch.cat([t1, t3], dim=0)
    assert torch.equal(out, expected)


def test_packedtensor_pads_mixed_dynamic_resolution_images():
    """Raw image batches pad spatial dimensions before packing on dim 0."""
    first = torch.ones(1, 3, 2, 4)
    second = 2 * torch.ones(1, 3, 4, 2)

    packed = PackedTensor(
        [first, second], dim_to_pack=0, pad_to_max_shape=True
    ).as_tensor()

    assert packed.shape == (2, 3, 4, 4)
    torch.testing.assert_close(packed[0, :, :2, :4], first[0])
    torch.testing.assert_close(packed[0, :, 2:, :], torch.zeros(3, 2, 4))
    torch.testing.assert_close(packed[1, :, :4, :2], second[0])
    torch.testing.assert_close(packed[1, :, :, 2:], torch.zeros(3, 4, 2))


@pytest.mark.mcore
def test_dynamic_resolution_padding_is_cropped_before_radio_patchification():
    """Batch-shape padding must not become RADIO image content."""
    from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import (
        NemotronOmniModel,
    )

    generator = torch.Generator().manual_seed(2026)
    small = torch.randn(1, 3, 32, 32, generator=generator)
    large = torch.randn(1, 3, 64, 64, generator=generator)
    imgs_sizes = torch.tensor([[32, 32], [64, 64]], dtype=torch.long)

    padded = PackedTensor(
        [small, large],
        dim_to_pack=0,
        pad_to_max_shape=True,
    ).as_tensor()
    # Use nonzero garbage so this test cannot pass merely because F.pad uses zero.
    padded[0, :, 32:, :] = 123
    padded[0, :, :, 32:] = -456

    class _Patchifier:
        patch_dim = 16

    patchifier = _Patchifier()
    packed_patches = NemotronOmniModel._patchify_dynamic_images(
        patchifier,
        padded,
        imgs_sizes,
    )
    expected_patches = torch.cat(
        [
            NemotronOmniModel._patchify_dynamic_images(
                patchifier,
                small,
                imgs_sizes[:1],
            ),
            NemotronOmniModel._patchify_dynamic_images(
                patchifier,
                large,
                imgs_sizes[1:],
            ),
        ],
        dim=1,
    )

    torch.testing.assert_close(packed_patches, expected_patches)


@pytest.mark.parametrize(
    ("first_shape", "second_shape", "expected_shape"),
    [
        ((1, 2, 3), (2, 4, 3), (3, 4, 3)),
        ((1, 2, 3, 2, 4), (2, 4, 3, 4, 2), (3, 4, 3, 4, 4)),
    ],
)
def test_packedtensor_pad_to_max_shape_supports_audio_and_video(
    first_shape, second_shape, expected_shape
):
    """Padding is generic across non-packing dimensions and tensor ranks."""
    first = torch.ones(first_shape)
    second = 2 * torch.ones(second_shape)

    packed = PackedTensor(
        [first, second], dim_to_pack=0, pad_to_max_shape=True
    ).as_tensor()

    assert packed.shape == expected_shape
    slices = (slice(0, first_shape[0]),) + tuple(
        slice(0, size) for size in first_shape[1:]
    )
    torch.testing.assert_close(packed[slices], first)


def test_pad_to_max_shape_rejects_mismatched_ranks():
    with pytest.raises(ValueError, match="same rank"):
        PackedTensor(
            [torch.ones(1, 3, 4), torch.ones(1, 3)],
            dim_to_pack=0,
            pad_to_max_shape=True,
        ).as_tensor()


def test_pad_to_max_shape_rejects_out_of_range_dim():
    with pytest.raises(IndexError, match="dim_to_pack=3 is invalid"):
        PackedTensor(
            [torch.ones(1, 3, 4), torch.ones(2, 3, 4)],
            dim_to_pack=3,
            pad_to_max_shape=True,
        ).as_tensor()


def test_pad_to_max_shape_supports_negative_pack_dim():
    packed = PackedTensor(
        [torch.ones(2, 3, 1), 2 * torch.ones(4, 3, 1)],
        dim_to_pack=-3,
        pad_to_max_shape=True,
    ).as_tensor()

    assert packed.shape == (6, 3, 1)


def test_slice_preserves_pad_to_max_shape_flag():
    packed = PackedTensor(
        [torch.ones(1, 3, 2, 4), 2 * torch.ones(1, 3, 4, 2)],
        dim_to_pack=0,
        pad_to_max_shape=True,
    )

    sliced = packed.slice([0, 1])

    assert sliced.pad_to_max_shape is True
    assert sliced.as_tensor().shape == (2, 3, 4, 4)


def test_packedtensor_dedup_uses_provenance_not_prompt_position():
    """Only segments descended from the same physical media are compacted."""
    shared = PackedTensor(torch.tensor([[1.0]]), dim_to_pack=0)
    shared.enable_deduplication()
    shared_copy = deepcopy(shared)
    same_prompt_but_different_media = PackedTensor(torch.tensor([[1.0]]), dim_to_pack=0)
    same_prompt_but_different_media.enable_deduplication()

    packed = PackedTensor.concat([shared, shared_copy, same_prompt_but_different_media])

    assert len(packed) == 3
    assert sum(packed.logical_segment_counts_by_row()) == 3
    assert len(packed.tensors) == 2
    torch.testing.assert_close(packed.as_tensor(), torch.tensor([[1.0], [1.0], [1.0]]))


def test_packedtensor_multiturn_csr_preserves_shared_seed_and_unique_media():
    """Diverged rows retain one seed segment plus their own later segment."""
    seed = PackedTensor(torch.tensor([[1.0]]), dim_to_pack=0)
    seed.enable_deduplication()
    row_1 = PackedTensor.merge_segments(
        [deepcopy(seed), PackedTensor(torch.tensor([[2.0]]), dim_to_pack=0)]
    )
    row_2 = PackedTensor.merge_segments(
        [deepcopy(seed), PackedTensor(torch.tensor([[3.0]]), dim_to_pack=0)]
    )

    packed = PackedTensor.flattened_concat([row_1, row_2])

    assert len(packed) == 2
    assert sum(packed.logical_segment_counts_by_row()) == 4
    assert len(packed.tensors) == 3
    torch.testing.assert_close(
        packed.as_tensor(), torch.tensor([[1.0], [2.0], [1.0], [3.0]])
    )

    second_row = packed.slice([1])
    assert len(second_row) == 1
    assert len(second_row.tensors) == 2
    torch.testing.assert_close(second_row.as_tensor(), torch.tensor([[1.0], [3.0]]))


def test_packedtensor_dedup_expands_before_dynamic_shape_padding():
    """Logical order is restored before non-packing dimensions are padded."""
    first = PackedTensor(
        torch.ones(1, 1, 2),
        dim_to_pack=0,
        pad_to_max_shape=True,
    ).enable_deduplication()
    second = PackedTensor(
        2 * torch.ones(1, 2, 1),
        dim_to_pack=0,
        pad_to_max_shape=True,
    ).enable_deduplication()

    packed = PackedTensor.concat([first, deepcopy(first), second])
    materialized = packed.as_tensor()

    assert materialized.shape == (3, 2, 2)
    torch.testing.assert_close(materialized[0], materialized[1])
    torch.testing.assert_close(materialized[2, :, 0], 2 * torch.ones(2))


def test_packedtensor_to_dtype_returns_independent_wrapper_when_dtype_matches():
    packed = PackedTensor(
        torch.ones(1, 2, dtype=torch.bfloat16), dim_to_pack=0
    ).enable_deduplication()
    compact = PackedTensor.concat([packed] * 2)

    unchanged = compact.to_dtype(torch.bfloat16)

    assert unchanged is not compact
    assert unchanged.tensors is not compact.tensors
    assert unchanged.tensors[0] is compact.tensors[0]
    assert unchanged._row_offsets == compact._row_offsets
    assert unchanged._row_offsets is not compact._row_offsets
    assert unchanged._segment_indices == compact._segment_indices
    assert unchanged._segment_indices is not compact._segment_indices
    assert unchanged._segment_provenance == compact._segment_provenance
    assert unchanged._segment_provenance is not compact._segment_provenance


def test_packedtensor_compact_dim_one_slice_empty_and_cloudpickle_roundtrip():
    first = torch.tensor([[1.0], [2.0]])
    second = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    packed = PackedTensor(
        [first, second],
        dim_to_pack=1,
    ).enable_deduplication()
    repeated = PackedTensor.concat(
        [packed.slice([row]) for row in range(len(packed)) for _ in range(2)]
    )

    assert len(repeated) == 4
    assert len(repeated.tensors) == 2
    torch.testing.assert_close(
        repeated.as_tensor(),
        torch.cat([first, first, second, second], dim=1),
    )

    selected = repeated.slice([3, 0, -1])
    assert len(selected) == 3
    assert len(selected.tensors) == 2
    torch.testing.assert_close(
        selected.as_tensor(),
        torch.cat([second, first, second], dim=1),
    )

    restored = cloudpickle.loads(cloudpickle.dumps(selected, protocol=5))
    assert restored.deduplication_enabled
    assert len(restored) == 3
    assert len(restored.tensors) == 2
    torch.testing.assert_close(restored.as_tensor(), selected.as_tensor())

    empty = PackedTensor.empty_rows_like(packed, 0)
    assert len(empty) == 0
    assert sum(empty.logical_segment_counts_by_row()) == 0
    assert empty.as_tensor() is None


def test_packedtensor_unpickles_pre_deduplication_state():
    tensor = torch.tensor([[1.0], [2.0]])
    legacy = PackedTensor.__new__(PackedTensor)
    legacy.__dict__ = {
        "tensors": [tensor],
        "dim_to_pack": 0,
        "pad_to_max_shape": False,
    }

    restored = cloudpickle.loads(cloudpickle.dumps(legacy, protocol=5))

    assert not restored.deduplication_enabled
    assert len(restored) == 1
    assert sum(restored.logical_segment_counts_by_row()) == 1
    torch.testing.assert_close(restored.as_tensor(), tensor)
    restored.enable_deduplication()
    assert restored.deduplication_enabled


def test_packedtensor_empty_legacy_rows_survive_copy_pickle_and_slice():
    legacy = PackedTensor(torch.tensor([[1.0]]), dim_to_pack=0)
    empty = PackedTensor.empty_rows_like(legacy, 0)

    assert len(empty) == 0
    assert not empty.deduplication_enabled
    assert empty.as_tensor() is None

    copied = deepcopy(empty)
    restored = cloudpickle.loads(cloudpickle.dumps(empty, protocol=5))
    sliced = empty.slice([])
    for value in (copied, restored, sliced):
        assert len(value) == 0
        assert sum(value.logical_segment_counts_by_row()) == 0
        assert not value.deduplication_enabled
        assert value.as_tensor() is None
