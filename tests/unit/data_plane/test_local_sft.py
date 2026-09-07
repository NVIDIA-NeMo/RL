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

from __future__ import annotations

import pytest
import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.data_plane.adapters.local import (
    LocalDataPlaneClient,
    local_batch_to_tensordict,
    materialize_local,
)
from nemo_rl.data_plane.interfaces import KVBatchMeta, LocalDataPlaneConfig
from nemo_rl.data_plane.worker_mixin import _materialize_fetched


def _client(*, max_partitions: int = 2) -> LocalDataPlaneClient:
    return LocalDataPlaneClient(LocalDataPlaneConfig(max_partitions=max_partitions))


def _register(client: LocalDataPlaneClient, partition_id: str = "step-0") -> None:
    client.register_partition(
        partition_id=partition_id,
        fields=["input_ids", "input_lengths", "pixel_values", "source_ids"],
        num_samples=2,
        consumer_tasks=["train"],
    )


def _put_multimodal_batch(
    client: LocalDataPlaneClient, partition_id: str = "step-0"
) -> KVBatchMeta:
    pixels = PackedTensor(
        [torch.full((1, 2), 1.0), torch.full((2, 2), 2.0)],
        dim_to_pack=0,
        pad_to_max_shape=True,
    ).enable_deduplication()
    fields = local_batch_to_tensordict(
        {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "input_lengths": torch.tensor([2, 3]),
            "pixel_values": pixels,
            "source_ids": ["source-a", "source-b"],
        },
        batch_size=2,
    )
    return client.put_samples(
        sample_ids=["a", "b"],
        partition_id=partition_id,
        fields=fields,
        tags=[{"source": "source-a"}, {"source": "source-b"}],
    )


def test_local_round_trip_preserves_tensor_and_packed_tensor_fields() -> None:
    client = _client()
    _register(client)
    meta = _put_multimodal_batch(client)

    assert meta.sequence_lengths == [2, 3]
    fetched = client.get_data(meta)
    batch = materialize_local(fetched)

    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 0], [3, 4, 5]]))
    assert batch["source_ids"] == ["source-a", "source-b"]
    pixels = batch["pixel_values"]
    assert isinstance(pixels, PackedTensor)
    assert pixels.dim_to_pack == 0
    assert pixels.pad_to_max_shape
    assert pixels.deduplication_enabled
    assert pixels.logical_segment_counts_by_row() == [1, 1]
    assert torch.equal(pixels.tensors[0], torch.full((1, 2), 1.0))
    assert torch.equal(pixels.tensors[1], torch.full((2, 2), 2.0))


def test_local_subset_preserves_requested_order_and_packed_rows() -> None:
    client = _client()
    _register(client)
    _put_multimodal_batch(client)

    fetched = client.get_samples(
        sample_ids=["b", "a"],
        partition_id="step-0",
        select_fields=["input_lengths", "pixel_values", "source_ids"],
    )
    batch = materialize_local(fetched)

    assert torch.equal(batch["input_lengths"], torch.tensor([3, 2]))
    assert batch["source_ids"] == ["source-b", "source-a"]
    pixels = batch["pixel_values"]
    assert isinstance(pixels, PackedTensor)
    assert torch.equal(pixels.tensors[0], torch.full((2, 2), 2.0))
    assert torch.equal(pixels.tensors[1], torch.full((1, 2), 1.0))


def test_stock_materialize_is_refused_for_a_local_batch() -> None:
    # materialize() reads a whole-column NonTensorData as a single row, so
    # pairing it with a local fetch would silently turn N rows into 1. The
    # dispatch in worker_mixin picks both from one flag; this pins that a
    # future edit to one branch alone fails loudly.
    client = _client()
    _register(client)
    fetched = client.get_data(_put_multimodal_batch(client))

    with pytest.raises(TypeError, match="materialize_local"):
        _materialize_fetched(
            fetched,
            local_batch=False,
            layout="padded",
            pad_value_dict=None,
            pad_to_seqlen=0,
        )

    batch = _materialize_fetched(
        fetched,
        local_batch=True,
        layout="padded",
        pad_value_dict=None,
        pad_to_seqlen=0,
        tags=[{"source": "source-a"}, {"source": "source-b"}],
    )
    assert batch["source_ids"] == ["source-a", "source-b"]


def test_local_rejects_duplicate_write() -> None:
    client = _client()
    _register(client)
    _put_multimodal_batch(client)

    with pytest.raises(ValueError, match="duplicate writes"):
        _put_multimodal_batch(client)


def test_local_bounds_active_and_prefetched_partitions() -> None:
    client = _client(max_partitions=2)
    _register(client, "active")
    _register(client, "prefetch")

    with pytest.raises(RuntimeError, match="partition limit"):
        _register(client, "too-far-ahead")

    client.clear_samples(sample_ids=None, partition_id="active")
    _register(client, "next")


def test_local_rejects_metadata_from_old_partition_generation() -> None:
    client = _client(max_partitions=1)
    _register(client)
    old_meta = _put_multimodal_batch(client)
    client.clear_samples(sample_ids=None, partition_id="step-0")
    _register(client)
    _put_multimodal_batch(client)

    with pytest.raises(ValueError, match="Stale local metadata"):
        client.get_data(old_meta)


def test_local_rejects_access_after_close_and_close_is_idempotent() -> None:
    client = _client()
    client.close()
    client.close()

    with pytest.raises(RuntimeError, match="closed"):
        _register(client)


def test_factory_resolves_validated_local_config() -> None:
    client = build_data_plane_client(LocalDataPlaneConfig(max_partitions=1))
    assert isinstance(client, LocalDataPlaneClient)
    client.register_partition(
        partition_id="only",
        fields=[],
        num_samples=1,
        consumer_tasks=[],
    )
    with pytest.raises(RuntimeError, match="partition limit"):
        client.register_partition(
            partition_id="overflow",
            fields=[],
            num_samples=1,
            consumer_tasks=[],
        )
    client.close()
