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
"""Sparse media from independent rollout writes must survive worker fetches."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.data.multimodal_utils import (
    MEDIA_ANCHOR_SPANS_TAG,
    PackedTensor,
    multimodal_row_tags,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin, _expand_sparse_packed_rows
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from ._rollout_shapes import register_train_partition


def _independent_rollout_groups(*, equal_image_shapes: bool):
    client = NoOpDataPlaneClient()
    fields = [
        "input_ids",
        "input_lengths",
        "sample_mask",
        "token_mask",
        "rewards",
        "pixel_values",
        "image_grid_thw",
    ]
    register_train_partition(client, num_samples=4, fields=fields)
    lengths = [2, 2, 2] if equal_image_shapes else [2, 3, 1]
    pixels = PackedTensor(
        [torch.full((length, 3), float(i + 1)) for i, length in enumerate(lengths)],
        dim_to_pack=0,
    )
    grids = PackedTensor(
        [torch.tensor([[1, 1, length]]) for length in lengths], dim_to_pack=0
    )
    image_meta = kv_first_write(
        BatchedDataDict(
            {
                "input_ids": torch.tensor([[11, 12], [21, 22], [31, 32]]),
                "input_lengths": torch.tensor([2, 2, 2]),
                "sample_mask": torch.ones(3),
                "token_mask": torch.tensor([[0, 1], [1, 1], [1, 0]]),
                "rewards": torch.tensor([0.25, 0.5, 0.75]),
                "pixel_values": pixels.to_wire()[0],
                "image_grid_thw": grids.to_wire()[0],
            }
        ),
        sample_ids=["image-a", "image-b", "image-c"],
        dp_client=client,
        partition_id="train",
        tags=multimodal_row_tags({"pixel_values": pixels, "image_grid_thw": grids}, 3),
    )
    text_meta = kv_first_write(
        BatchedDataDict(
            {
                "input_ids": torch.tensor([[41, 42]]),
                "input_lengths": torch.tensor([2]),
                "sample_mask": torch.ones(1),
                "token_mask": torch.tensor([[0, 1]]),
                "rewards": torch.tensor([1.0]),
            }
        ),
        sample_ids=["text-only"],
        dp_client=client,
        partition_id="train",
        tags=[{}],
    )
    mixed = image_meta.concat(text_meta).subset([0, 3, 1, 2])
    return client, mixed, pixels, grids


@pytest.mark.parametrize("equal_image_shapes", [False, True])
@pytest.mark.parametrize("leader_broadcast", [False, True])
@pytest.mark.parametrize("captured_media_anchors", [False, True])
def test_worker_fetch_preserves_independently_written_sparse_media(
    monkeypatch, equal_image_shapes, leader_broadcast, captured_media_anchors
):
    """Exercise both worker branches; only the NCCL boundary is mocked on CPU."""
    client, meta, pixels, grids = _independent_rollout_groups(
        equal_image_shapes=equal_image_shapes
    )
    if captured_media_anchors:
        for tag, token_id in zip(meta.tags, [11, None, 21, 31], strict=True):
            tag[MEDIA_ANCHOR_SPANS_TAG] = [] if token_id is None else [(0, 1, token_id)]
    worker = TQWorkerMixin()
    worker._dp_client = client
    broadcast_calls = []
    if leader_broadcast:
        group = SimpleNamespace(size=lambda: 2)
        monkeypatch.setattr(worker, "_get_replica_group", lambda: group)
        monkeypatch.setattr(worker, "_is_replica_leader", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_global_rank", lambda *args: 0)

        def broadcast(data, **kwargs):
            assert kwargs["is_leader"] is True
            broadcast_calls.append(data)
            return data

        monkeypatch.setattr(
            "nemo_rl.data_plane.worker_mixin._broadcast_batched_data_dict", broadcast
        )

    fetched = worker._fetch(
        meta,
        fetch_policy="leader_broadcast" if leader_broadcast else "independent",
        dp_aligned_seq_len=False,
    )

    assert len(broadcast_calls) == int(leader_broadcast)
    assert fetched["input_ids"].tolist() == [[11, 12], [41, 42], [21, 22], [31, 32]]
    assert fetched["token_mask"].tolist() == [[0, 1], [0, 1], [1, 1], [1, 0]]
    assert fetched["sample_mask"].tolist() == [1, 1, 1, 1]
    assert fetched["rewards"].tolist() == [0.25, 1.0, 0.5, 0.75]
    if captured_media_anchors:
        assert fetched["media_token_validity_mask"].tolist() == [
            [True, False],
            [False, False],
            [True, False],
            [True, False],
        ]
    for field, original in (("pixel_values", pixels), ("image_grid_thw", grids)):
        assert isinstance(fetched[field], PackedTensor)
        assert fetched[field].logical_segment_counts_by_row() == [1, 0, 1, 1]
        torch.testing.assert_close(
            fetched[field].as_tensor(), original.as_tensor(), rtol=0, atol=0
        )


def test_text_only_shard_does_not_fetch_union_media_columns():
    client, meta, _, _ = _independent_rollout_groups(equal_image_shapes=False)
    worker = TQWorkerMixin()
    worker._dp_client = client
    fetched = worker._fetch(
        meta.subset([1]), fetch_policy="independent", dp_aligned_seq_len=False
    )
    assert fetched["input_ids"].tolist() == [[41, 42]]
    assert "pixel_values" not in fetched
    assert "image_grid_thw" not in fetched


def test_declared_but_missing_media_still_fails():
    client, meta, _, _ = _independent_rollout_groups(equal_image_shapes=False)
    # A geometry tag is provenance, not permission to manufacture missing pixels.
    meta.tags[1] = dict(meta.tags[0])
    worker = TQWorkerMixin()
    worker._dp_client = client
    with pytest.raises(KeyError, match="not yet produced"):
        worker._fetch(meta, fetch_policy="independent", dp_aligned_seq_len=False)


def test_legacy_rows_without_geometry_do_not_hide_transport_errors():
    client, meta, _, _ = _independent_rollout_groups(equal_image_shapes=False)
    meta.tags = None
    worker = TQWorkerMixin()
    worker._dp_client = client
    with pytest.raises(KeyError, match="not yet produced"):
        worker._fetch(meta, fetch_policy="independent", dp_aligned_seq_len=False)


def test_sparse_row_count_mismatch_fails():
    pixels = PackedTensor([torch.ones(2, 3)], dim_to_pack=0)
    with pytest.raises(ValueError, match="row count"):
        _expand_sparse_packed_rows(pixels, (0, 2), 3)
