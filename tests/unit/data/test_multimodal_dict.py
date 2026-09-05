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
    PACKED_MULTIMODAL_FIELDS,
    PER_TOKEN_MULTIMODAL_FIELDS,
    PackedTensor,
    encode_multimodal_for_wire,
    multimodal_row_tags,
    reassemble_packed_multimodal,
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


def test_truncate_tensors_skips_wire_form_multimodal():
    """Dynamic batching narrows dim 1 to the microbatch seqlen. The
    data-plane wire form of a packed multimodal field has patch count on
    dim 1, not seqlen — narrowing it would corrupt the images (or raise
    when patches < seqlen)."""
    batch = BatchedDataDict(
        {
            "input_ids": torch.arange(8).reshape(2, 4),
            # [B, max_patches, feat] — 3 patches, fewer than seqlen=4.
            "pixel_values": torch.randn(2, 3, 16),
            # Per-token multimodal IS sequence-aligned and must truncate.
            "mm_token_type_ids": torch.ones((2, 4), dtype=torch.long),
        }
    )

    batch.truncate_tensors(dim=1, truncated_len=2)

    assert torch.equal(batch["input_ids"], torch.tensor([[0, 1], [4, 5]]))
    assert batch["mm_token_type_ids"].shape == (2, 2)
    assert batch["pixel_values"].shape == (2, 3, 16)


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


def test_to_wire_emits_one_row_per_logical_row_under_dedup():
    """Deduplicated values map one logical row to several shared physical
    segments, so the wire encoder must walk logical rows — iterating
    ``tensors`` would emit the physical segment count as the batch size
    and desync every downstream column."""
    seg_a = torch.ones(2, 3)
    seg_b = 2 * torch.ones(4, 3)
    # 3 logical rows over 2 physical segments: [a], [a, b], [b].
    packed = PackedTensor(
        [seg_a, seg_b],
        dim_to_pack=0,
        _row_offsets=[0, 1, 3, 4],
        _segment_indices=[0, 0, 1, 1],
    ).enable_deduplication()
    assert len(packed) == 3

    nested, shapes = packed.to_wire()
    # Rows are flattened, so the logical row count shows up as one shape entry
    # per row; the per-row element counts follow the 2/6/4 row heights.
    assert len(shapes) == 3
    assert [t.numel() for t in nested.unbind()] == [2 * 3, 6 * 3, 4 * 3]
    assert torch.equal(
        PackedTensor.from_wire(nested, shapes).as_tensor(),
        packed.as_tensor(),
    )


def test_to_wire_does_not_pad_segments_before_concat_under_dedup():
    """A dedup row spanning segments of differing trailing dims.

    ``to_wire`` flattens each segment, so the per-row concat is 1-D and cannot
    hit ``RuntimeError: Sizes of tensors must match except in dimension 0``.
    The padding ``as_tensor`` needs is applied on the read side by
    ``from_wire`` instead, so no padded bytes cross the wire.
    """
    # One logical row referencing two segments: 2x4 and 4x2 spatial dims.
    packed = PackedTensor(
        [torch.ones(1, 3, 2, 4), 2 * torch.ones(1, 3, 4, 2)],
        dim_to_pack=0,
        pad_to_max_shape=True,
        _row_offsets=[0, 2],
        _segment_indices=[0, 1],
    )
    assert len(packed) == 1
    expected = packed.as_tensor()
    assert expected.shape == (2, 3, 4, 4)  # padded to the batch max

    nested, shapes = packed.to_wire()
    # Natural size: 1*3*2*4 + 1*3*4*2 = 48 elements, no padding materialized.
    assert [t.numel() for t in nested.unbind()] == [48]
    assert shapes == [[[1, 3, 2, 4], [1, 3, 4, 2]]]

    restored = PackedTensor.from_wire(nested, shapes, pad_to_max_shape=True).as_tensor()
    assert torch.equal(restored, expected)


def test_from_wire_rejects_dense_input():
    """A dense value here means ``materialize`` padded the field, which
    silently loses the row boundaries. Fail loud instead."""
    with pytest.raises(TypeError, match="expects the nested value"):
        PackedTensor.from_wire(torch.zeros(3, 2, 4), [])


def test_from_wire_empty_rows_match_legacy_none_semantics():
    """An image-free shard must reconstruct as legacy does: ``as_tensor``
    returns ``None`` and the per-row counts are 0, not a ``(0, ...)``
    tensor with counts of 1. A zero-length row is how absence travels."""
    nested = torch.nested.as_nested_tensor(
        [torch.zeros(0, 3, 4), torch.zeros(0, 3, 4)], layout=torch.jagged
    )

    # An empty row contributes no segments, so its shapes entry is empty too
    # — what ``to_wire`` mints for an all-``None`` row.
    restored = PackedTensor.from_wire(nested, [[], []])
    assert restored.as_tensor() is None
    assert restored.logical_segment_counts_by_row() == [0, 0]


def test_to_wire_does_not_materialize_pad_to_max_shape():
    """Dynamic-resolution values travel at natural size.

    Flattening each row to 1-D makes ``torch.jagged`` total, so differing
    trailing dims no longer force the write side to pad up to the batch max.
    The padding ``as_tensor`` returns is reapplied on read from the carried
    shapes, keeping the padded bytes out of the wire and out of TQ storage.
    """
    # Same rank, different trailing dims — nemotron-omni style tiles.
    first = torch.ones(1, 3, 2, 4)
    second = 2 * torch.ones(2, 3, 4, 2)
    packed = PackedTensor([first, second], dim_to_pack=0, pad_to_max_shape=True)

    nested, shapes = packed.to_wire()
    rows = list(nested.unbind())
    # Natural sizes: 1*3*2*4=24 and 2*3*4*2=48. Padding to the batch max
    # (3, 4, 4) would have cost 48 and 96 -- 3x the bytes for this batch.
    assert [t.numel() for t in rows] == [24, 48]
    assert shapes == [[[1, 3, 2, 4]], [[2, 3, 4, 2]]]

    # Padding is reapplied on read, reproducing the pre-wire as_tensor().
    assert torch.equal(
        PackedTensor.from_wire(nested, shapes, pad_to_max_shape=True).as_tensor(),
        packed.as_tensor(),
    )


def test_get_multimodal_dict_rejects_wire_form_field():
    """Either wire form reaching here is unrecoverable, so both fail loud.

    Dense means ``codec.materialize`` padded the field and the row boundaries
    are gone; nested means the shapes companion on ``KVBatchMeta.tags`` was
    never applied, and taking the flat rows as-is would train image-blind.
    Neither is reconstructible from inside ``get_multimodal_dict``.
    """
    dense = BatchedDataDict({"pixel_values": torch.zeros(2, 3, 4, 4)})
    with pytest.raises(ValueError, match=r"wire form \(dense\)"):
        dense.get_multimodal_dict(as_tensors=False)

    nested_value, _ = PackedTensor(
        [torch.ones(3, 4), torch.ones(1, 4)], dim_to_pack=0
    ).to_wire()
    nested = BatchedDataDict({"pixel_values": nested_value})
    with pytest.raises(ValueError, match=r"wire form \(nested\)"):
        nested.get_multimodal_dict(as_tensors=False)


def test_image_free_shard_emits_no_wire_field_and_reads_back_empty():
    """An all-empty packed field never reaches the wire, end to end.

    Replaces an earlier "0-row wire field" guard: that state is now
    unconstructible. ``to_wire`` returns ``None`` for an all-``None``
    value so the field is never emitted, and ``kv_first_write`` rejects a
    zero-row batch outright — so the read side has nothing to skip rather
    than an empty column to tolerate.
    """
    packed = PackedTensor([None, None], dim_to_pack=0)

    assert encode_multimodal_for_wire("pixel_values", packed) is None

    # What the trainer actually receives for an image-free shard.
    data = BatchedDataDict({"input_ids": torch.zeros(2, 4, dtype=torch.long)})
    assert "pixel_values" not in data.get_multimodal_dict(as_tensors=False)


def test_encode_multimodal_for_wire_packed_emits_single_nested_entry():
    """A ``PackedTensor`` arrives on the wire as exactly one entry. Row
    boundaries live in the nested tensor itself and are preserved by TQ
    (one stored entry per row), so no companion field is emitted."""
    packed = PackedTensor(
        [torch.ones(3, 4), torch.ones(1, 4)],
        dim_to_pack=0,
    )

    # One wire value under the field's own key: the payload. Shapes ride on
    # ``KVBatchMeta.tags``, not as a companion column, so the field count on
    # the wire is unchanged.
    value = encode_multimodal_for_wire("pixel_values", packed)

    assert value is not None
    assert value.is_nested
    # Flattened rows: 3*4 and 1*4 elements.
    assert [t.numel() for t in value.unbind()] == [12, 4]

    tags = multimodal_row_tags({"pixel_values": packed}, len(packed))
    assert [t["pixel_values__row_shapes"]["shapes"] for t in tags] == [
        [[3, 4]],
        [[1, 4]],
    ]
    assert tags[0]["pixel_values__row_shapes"]["pad"] is False


def test_multimodal_row_tags_does_not_encode_the_payload():
    """``multimodal_row_tags`` needs geometry only, and must not pay for bytes.

    It used to call ``to_wire``, whose ``torch.cat`` copies the whole column,
    and then throw the nested value away — a full copy of the largest field in
    the batch, discarded, once per rollout step.
    """
    packed = PackedTensor([torch.ones(3, 4), torch.ones(1, 4)], dim_to_pack=0)
    calls = []
    packed.to_wire = lambda: calls.append(1)  # type: ignore[method-assign]

    tags = multimodal_row_tags({"pixel_values": packed}, len(packed))

    assert calls == [], "multimodal_row_tags must not call to_wire()"
    assert [t["pixel_values__row_shapes"]["shapes"] for t in tags] == [
        [[3, 4]],
        [[1, 4]],
    ]


def test_multimodal_row_tags_rejects_row_count_disagreement():
    """A tags list shorter than the batch would leave a trailing sample with no
    companion, which the read side then cannot distinguish from a lost one."""
    packed = PackedTensor([torch.ones(3, 4), torch.ones(1, 4)], dim_to_pack=0)

    with pytest.raises(ValueError, match="logical rows but the batch has"):
        multimodal_row_tags({"pixel_values": packed}, 3)


def test_reassemble_packed_multimodal_raises_without_companion():
    """No companion means the true shapes are gone; reconstructing anyway
    yields 1-D pixels and trains image-blind with no error."""
    packed = PackedTensor([torch.ones(3, 4), torch.ones(1, 4)], dim_to_pack=0)
    nested, _ = packed.to_wire()

    with pytest.raises(ValueError, match="tags=None"):
        reassemble_packed_multimodal({"pixel_values": nested}, None)

    with pytest.raises(ValueError, match="checked 2 tag rows"):
        reassemble_packed_multimodal({"pixel_values": nested}, [{}, {}])


def test_reassemble_packed_multimodal_round_trips_with_companion():
    packed = PackedTensor([torch.ones(3, 4), torch.ones(1, 4)], dim_to_pack=0)
    nested, _ = packed.to_wire()
    tags = multimodal_row_tags({"pixel_values": packed}, len(packed))

    fields = {"pixel_values": nested}
    reassemble_packed_multimodal(fields, tags)

    assert torch.equal(fields["pixel_values"].as_tensor(), packed.as_tensor())


def test_encode_multimodal_for_wire_per_token_passes_through():
    """Per-token fields are rectangular ``[B, S]`` — they ride as plain
    tensors, not nested ones."""
    ids = torch.ones((2, 6), dtype=torch.long)

    assert encode_multimodal_for_wire("mm_token_type_ids", ids) is ids


def test_pixel_dtype_cast_survives_to_the_wire_and_spares_integers():
    """The rollout casts pixels once, at ``get_multimodal_dict``.

    No worker re-applies it, so if the cast did not survive
    ``encode_multimodal_for_wire`` the largest column would cross the wire in
    fp32 where the legacy path shipped bf16 — a silent 2x on the dominant
    field. Integer geometry must not be dragged along with it.
    """
    batch = BatchedDataDict(
        {
            "pixel_values": PackedTensor(
                [torch.ones(3, 4), torch.ones(1, 4)], dim_to_pack=0
            ),
            "image_grid_thw": PackedTensor(
                [torch.ones(1, 3, dtype=torch.long)] * 2, dim_to_pack=0
            ),
        }
    )

    multimodal = batch.get_multimodal_dict(as_tensors=False, pixel_dtype=torch.bfloat16)
    wire = {k: encode_multimodal_for_wire(k, v) for k, v in multimodal.items()}

    assert wire["pixel_values"].dtype == torch.bfloat16
    assert wire["image_grid_thw"].dtype == torch.int64


def test_encode_multimodal_for_wire_skips_all_empty_packed():
    """An all-``None`` packed field has nothing to ship at all."""
    packed = PackedTensor([None, None], dim_to_pack=0)

    assert encode_multimodal_for_wire("pixel_values", packed) is None


def test_encode_multimodal_for_wire_rejects_unregistered_field():
    """A new modality added to ``get_multimodal_dict`` without a registry
    entry must fail loud here — the silent-drop class this PR fixes."""
    with pytest.raises(KeyError, match="unregistered multimodal field"):
        encode_multimodal_for_wire("audio_values", torch.zeros(2, 4))


def test_encode_multimodal_for_wire_rejects_unregistered_packed_field():
    """The realistic drift case, and the one the registries exist to catch.

    ``get_multimodal_dict`` emits *any* ``PackedTensor`` value regardless of
    key, so a processor field that is in neither registry reaches
    ``encode_multimodal_for_wire`` as a ``PackedTensor``. It must raise
    rather than be dropped: a dropped image column is a silently
    image-blind forward, not a crash.

    The key below is a stand-in — the behavior is name-independent. No
    specific model's ``model_input_names`` is asserted here, because that
    could not be verified against the pinned transformers version.
    """
    packed = PackedTensor([torch.ones(2, 4), torch.ones(3, 4)], dim_to_pack=0)

    with pytest.raises(KeyError, match="unregistered multimodal field"):
        encode_multimodal_for_wire("pixel_attention_mask", packed)


def test_unregistered_packed_field_survives_get_multimodal_dict_then_raises():
    """Pin the whole leak path, not just the encoder in isolation.

    ``get_multimodal_dict``'s first branch is ``isinstance(v, PackedTensor)``
    — no registry check — so an unregistered packed field passes straight
    through it. The registry gate is therefore load-bearing only at the wire
    boundary; this test proves the two halves compose into a loud failure.
    """
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(8).reshape(2, 4),
            "pixel_attention_mask": PackedTensor(
                [torch.ones(2, 4), torch.ones(3, 4)], dim_to_pack=0
            ),
        }
    )

    mm = data.get_multimodal_dict(as_tensors=False)
    assert "pixel_attention_mask" in mm  # slipped past the read-side dispatch

    with pytest.raises(KeyError, match="unregistered multimodal field"):
        for k, v in mm.items():
            encode_multimodal_for_wire(k, v)


def test_encode_multimodal_for_wire_rejects_wrong_type_per_registry():
    """Registry membership decides the branch, so a mismatched value type
    is a contract break, not something to coerce."""
    with pytest.raises(AssertionError, match="expected PackedTensor"):
        encode_multimodal_for_wire("pixel_values", torch.zeros(2, 4))

    with pytest.raises(AssertionError, match="expected Tensor"):
        encode_multimodal_for_wire(
            "mm_token_type_ids", PackedTensor([torch.ones(2, 3)], dim_to_pack=0)
        )


def test_multimodal_registries_are_disjoint():
    """A field in both registries would make ``encode_multimodal_for_wire``
    dispatch order-dependent and silently pick the packed branch."""
    assert not (PACKED_MULTIMODAL_FIELDS & PER_TOKEN_MULTIMODAL_FIELDS)


# ── to_wire guard rails ──────────────────────────────────────────


def test_to_wire_rejects_nonzero_dim_to_pack():
    """Only ``dim_to_pack=0`` round-trips; anything else needs
    ``ragged_idx`` threading, so it must raise instead of silently
    encoding along the wrong axis."""
    packed = PackedTensor([torch.ones(2, 3), torch.ones(2, 5)], dim_to_pack=1)

    with pytest.raises(NotImplementedError, match="only supports dim_to_pack=0"):
        packed.to_wire()


def test_to_wire_all_none_returns_none():
    """All-empty batch signals 'skip this field' with ``None`` rather than
    an empty nested tensor the read side cannot interpret."""
    packed = PackedTensor([None, None, None], dim_to_pack=0)

    nested, shapes = packed.to_wire()
    assert nested is None
    assert shapes == []


def test_to_wire_carries_mixed_rank_rows():
    """Rows of differing rank now encode.

    The old encoder rejected these: padding to a batch max is undefined across
    ranks, and no ``torch.nested`` layout holds them. Flattening sidesteps both
    -- every row becomes rank-1 -- so the rank check was removable rather than
    load-bearing. Reshaping on read restores the original ranks.
    """
    rows = [torch.ones(1, 3, 2), torch.ones(2, 3)]
    packed = PackedTensor(list(rows), dim_to_pack=0, pad_to_max_shape=True)

    nested, shapes = packed.to_wire()
    assert [t.numel() for t in nested.unbind()] == [6, 6]
    assert shapes == [[[1, 3, 2]], [[2, 3]]]

    restored = PackedTensor.from_wire(nested, shapes, pad_to_max_shape=True)
    assert [tuple(t.shape) for t in restored.tensors] == [(1, 3, 2), (2, 3)]
