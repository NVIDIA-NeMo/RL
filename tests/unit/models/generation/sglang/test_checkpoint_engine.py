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

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.sglang.checkpoint_engine import (
    SGLangCheckpointEngineMixin,
    _aligned_checkpoint_engine_batches,
    _validate_rank_batches,
)


class _Worker(SGLangCheckpointEngineMixin):
    def __init__(self, *, gpus_per_node=2, num_gpus_per_engine=2):
        self.gpus_per_node = gpus_per_node
        self.num_gpus_per_engine = num_gpus_per_engine
        self.base_gpu_id = 0
        self.node_rank = 0
        self.update_calls = []
        self.invalidated = False

    def _to_local_gpu_id(self, gpu_id):
        return gpu_id

    def update_weights_from_tensor(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"success": True}

    def invalidate_kv_cache(self):
        self.invalidated = True


class _Engine:
    shard_expert_weights = False

    def __init__(self, device="cpu", batches=None):
        self.device = device
        self.batches = batches or []
        self.prepared = False
        self.process_group = None
        self.finalized = False

    def prepare(self):
        self.prepared = True
        return {"agent": self.device}

    def init_rollout_process_group(self, **kwargs):
        self.process_group = kwargs

    async def receive_weight_batches(self):
        for batch in self.batches:
            yield batch

    def finalize(self):
        self.finalized = True


def test_validate_rank_batches_accepts_matching_names_and_dtypes():
    batches = [
        [("a", torch.ones(2)), ("b", torch.ones(1, dtype=torch.bfloat16))],
        [("a", torch.ones(3)), ("b", torch.ones(4, dtype=torch.bfloat16))],
    ]

    grouped = _validate_rank_batches(batches)

    assert list(grouped[0]) == [torch.float32, torch.bfloat16]
    assert [name for name, _tensor in grouped[1][torch.float32]] == ["a"]


def test_validate_rank_batches_rejects_divergent_names():
    batches = [
        [("a", torch.ones(1))],
        [("b", torch.ones(1))],
    ]

    with pytest.raises(RuntimeError, match="diverged"):
        _validate_rank_batches(batches)


def test_checkpoint_engine_lifecycle_uses_one_receiver_per_sglang_rank(monkeypatch):
    engines = []

    def create_engine(_backend, *, bucket_size_bytes, engine_kwargs):
        assert bucket_size_bytes == 1024
        engine = _Engine(device=engine_kwargs["device"])
        engines.append(engine)
        return engine

    monkeypatch.setattr(
        "nemo_rl.utils.checkpoint_engines.base.create_checkpoint_engine",
        create_engine,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.utils.train_utils.monkey_patch_torch_reductions",
        MagicMock(),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())

    worker = _Worker()
    worker.init_checkpoint_engine(
        "nixl", 1024, {"device": "cuda", "release_after_refit": False}, 4
    )

    assert [engine.device for engine in engines] == ["cuda:0", "cuda:1"]
    assert worker.prepare_checkpoint_engine() == [
        {"agent": "cuda:0", "rank": 4},
        {"agent": "cuda:1", "rank": 5},
    ]

    metadata = [{"rank": rank} for rank in range(12)]
    worker.init_checkpoint_engine_process_group(6, 6, metadata)
    assert engines[0].process_group == {
        "rollout_rank": 4,
        "train_world_size": 6,
        "rollout_world_size": 6,
        "metadata": metadata,
    }
    assert engines[1].process_group["rollout_rank"] == 5

    worker.finalize_checkpoint_engine()
    assert all(engine.finalized for engine in engines)


def test_checkpoint_engine_update_streams_each_aligned_batch(monkeypatch):
    rank_0 = [
        [("a", torch.ones(2))],
        [("b", torch.ones(3))],
    ]
    rank_1 = [
        [("a", torch.ones(4))],
        [("b", torch.ones(5))],
    ]
    worker = _Worker()
    worker.checkpoint_engines = [_Engine(batches=rank_0), _Engine(batches=rank_1)]
    worker._checkpoint_engine_target_devices = [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    ]
    serialize = MagicMock(
        side_effect=[
            ([["a-r0", "a-r1"]], [object()]),
            ([["b-r0", "b-r1"]], [object()]),
        ]
    )
    monkeypatch.setattr(worker, "_serialize_checkpoint_engine_batches", serialize)

    assert asyncio.run(worker._update_weights_from_checkpoint_engine_async())
    assert [call["serialized_named_tensors"] for call in worker.update_calls] == [
        ["a-r0", "a-r1"],
        ["b-r0", "b-r1"],
    ]
    assert [call["weight_version"] for call in worker.update_calls] == ["1", "1"]
    assert worker.invalidated
    assert worker._checkpoint_engine_weight_version == 1


def test_checkpoint_engine_alignment_handles_different_transport_buckets():
    engines = [
        _Engine(
            batches=[
                [("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))],
                [("c", torch.tensor([3.0]))],
            ]
        ),
        _Engine(
            batches=[
                [("a", torch.tensor([4.0]))],
                [("b", torch.tensor([5.0])), ("c", torch.tensor([6.0]))],
            ]
        ),
    ]

    async def collect():
        return [
            batches async for batches in _aligned_checkpoint_engine_batches(engines)
        ]

    aligned = asyncio.run(collect())

    # Assert the tensors, not just the names. The aligner itself enforces name
    # equality across ranks, so a name-only assertion holds under any rank
    # permutation and would not catch a shard being handed to the wrong rank.
    assert [
        [
            [(name, tensor.tolist()) for name, tensor in rank_batch]
            for rank_batch in batch
        ]
        for batch in aligned
    ] == [
        [[("a", [1.0])], [("a", [4.0])]],
        [[("b", [2.0])], [("b", [5.0])]],
        [[("c", [3.0])], [("c", [6.0])]],
    ]


def test_checkpoint_engine_serializes_one_flattened_payload_per_rank():
    from nemo_rl.models.generation.sglang.utils.train_utils import (
        FlattenedTensorBucket,
        MultiprocessingSerializer,
    )

    worker = _Worker()
    worker._checkpoint_engine_target_devices = [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    batches = [
        [("a", torch.tensor([1.0, 2.0])), ("b", torch.tensor([3.0]))],
        [("a", torch.tensor([4.0, 5.0])), ("b", torch.tensor([6.0]))],
    ]

    serialized_by_dtype, keepalive = worker._serialize_checkpoint_engine_batches(
        batches
    )

    assert len(keepalive) == 2
    assert len(serialized_by_dtype) == 1
    reconstructed = []
    for payload in serialized_by_dtype[0]:
        decoded = MultiprocessingSerializer.deserialize(payload)
        bucket = FlattenedTensorBucket(
            flattened_tensor=decoded["flattened_tensor"],
            metadata=decoded["metadata"],
        )
        reconstructed.append(dict(bucket.reconstruct_tensors()))
    torch.testing.assert_close(reconstructed[0]["a"], batches[0][0][1])
    torch.testing.assert_close(reconstructed[0]["b"], batches[0][1][1])
    torch.testing.assert_close(reconstructed[1]["a"], batches[1][0][1])
    torch.testing.assert_close(reconstructed[1]["b"], batches[1][1][1])


def test_checkpoint_engine_update_rejects_misaligned_stream_lengths(monkeypatch):
    worker = _Worker()
    worker.checkpoint_engines = [
        _Engine(batches=[[("a", torch.ones(1))]]),
        _Engine(batches=[]),
    ]
    monkeypatch.setattr(
        worker,
        "_serialize_checkpoint_engine_batches",
        MagicMock(return_value=([["payload"]], [object()])),
    )

    with pytest.raises(RuntimeError, match="ended with different weights"):
        asyncio.run(worker._update_weights_from_checkpoint_engine_async())


def test_checkpoint_engine_rejects_cross_node_tensor_parallelism():
    worker = _Worker(gpus_per_node=2, num_gpus_per_engine=4)

    with pytest.raises(NotImplementedError, match="fit on one node"):
        worker._checkpoint_engine_devices()


def test_checkpoint_engine_reports_each_local_device_memory(monkeypatch):
    worker = _Worker()
    get_properties = MagicMock(
        side_effect=[
            SimpleNamespace(total_memory=100),
            SimpleNamespace(total_memory=200),
        ]
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_properties)

    assert worker.checkpoint_engine_total_memory_bytes() == [100, 200]
