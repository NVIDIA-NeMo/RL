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
import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.sglang.checkpoint_engine import (
    SGLangCheckpointEngineMixin,
    _aligned_checkpoint_engine_batches,
    _load_checkpoint_engine_weights,
    _validate_rank_batches,
)

# The message SGLang's scheduler asserts with at the pinned rev when a tensor
# update arrives outside a weight-update session.
_NO_SESSION = "update_weights_from_tensor requires an open begin_weight_update session"


class _Worker(SGLangCheckpointEngineMixin):
    def __init__(self, *, gpus_per_node=2, num_gpus_per_engine=2):
        self.gpus_per_node = gpus_per_node
        self.num_gpus_per_engine = num_gpus_per_engine
        self.base_gpu_id = 0
        self.node_rank = 0
        self.update_calls = []
        self.weight_update_in_progress = False

    def _to_local_gpu_id(self, gpu_id):
        return gpu_id

    def begin_weight_update(self):
        self.weight_update_in_progress = True

    def end_weight_update(self):
        self.weight_update_in_progress = False

    def update_weights_from_tensor(self, **kwargs):
        # Models the pinned rev's own gate rather than rubber-stamping the
        # call: a transfer that escapes the session fails here the way it
        # would against a real server.
        assert self.weight_update_in_progress, _NO_SESSION
        self.update_calls.append(kwargs)
        return {"success": True}


def _refit(worker):
    """Run one refit the way the synchronizer does -- inside the session."""
    worker.begin_weight_update()
    try:
        return asyncio.run(_load_checkpoint_engine_weights(worker))
    finally:
        worker.end_weight_update()


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

    assert _refit(worker)
    assert [call["serialized_named_tensors"] for call in worker.update_calls] == [
        ["a-r0", "a-r1"],
        ["b-r0", "b-r1"],
    ]
    assert [call["weight_version"] for call in worker.update_calls] == ["1", "1"]
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
        _refit(worker)


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


class _RecyclingEngine:
    """Engine whose receive buffers are reused, like NIXL's two-buffer rotation.

    ``nemo_rl/utils/checkpoint_engines/nixl.py`` hands out views into a small
    pool of transfer buffers and rotates through them, so a tensor yielded N
    advances ago has been overwritten. A fake engine that allocates fresh
    tensors every batch cannot catch code that holds those references too long,
    which makes the whole bug class invisible. This one poisons on recycle.
    """

    shard_expert_weights = False

    def __init__(self, batches, rotation=2):
        self._batches = batches
        self._rotation = rotation
        self._issued = []

    async def receive_weight_batches(self):
        for index, batch in enumerate(self._batches):
            recycled = index - self._rotation
            if recycled >= 0:
                for _name, tensor in self._issued[recycled]:
                    tensor.fill_(float("nan"))
            live = [(name, tensor.clone()) for name, tensor in batch]
            self._issued.append(live)
            yield live


def test_checkpoint_engine_alignment_never_outlives_recycled_buffers():
    """The aligner must not hold a batch across more advances than the rotation.

    It only calls ``anext`` for a rank whose deque is already empty, so at most
    one batch per rank is outstanding. Prefetching would break that and surface
    here as NaN.
    """
    engines = [
        _RecyclingEngine(
            batches=[
                [("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))],
                [("c", torch.tensor([3.0]))],
                [("d", torch.tensor([4.0])), ("e", torch.tensor([5.0]))],
            ]
        ),
        _RecyclingEngine(
            batches=[
                [("a", torch.tensor([6.0]))],
                [("b", torch.tensor([7.0])), ("c", torch.tensor([8.0]))],
                [("d", torch.tensor([9.0]))],
                [("e", torch.tensor([10.0]))],
            ]
        ),
    ]

    async def collect():
        seen = []
        async for batch in _aligned_checkpoint_engine_batches(engines):
            # Read the tensors as the consumer would, while the batch is live.
            seen.append(
                [[(name, tensor.clone()) for name, tensor in rb] for rb in batch]
            )
        return seen

    aligned = asyncio.run(collect())

    flat = [t for batch in aligned for rb in batch for _name, t in rb]
    assert flat, "aligner yielded nothing"
    assert not any(torch.isnan(t).any() for t in flat), (
        "a yielded tensor was backed by a recycled buffer"
    )
    assert [[[name for name, _t in rb] for rb in batch] for batch in aligned] == [
        [["a"], ["a"]],
        [["b"], ["b"]],
        [["c"], ["c"]],
        [["d"], ["d"]],
        [["e"], ["e"]],
    ]


def test_checkpoint_engine_update_posts_once_per_dtype_group():
    """NIXL packs buckets by bytes, not dtype, so mixed-dtype batches are normal.

    Each dtype group becomes its own ``update_weights_from_tensor`` call, and
    every call must carry one payload per SGLang rank.
    """
    worker = _Worker()
    worker.checkpoint_engines = [
        _Engine(
            batches=[
                [
                    ("w", torch.ones(2, dtype=torch.bfloat16)),
                    ("norm", torch.ones(1, dtype=torch.float32)),
                ]
            ]
        ),
        _Engine(
            batches=[
                [
                    ("w", torch.full((2,), 2.0, dtype=torch.bfloat16)),
                    ("norm", torch.full((1,), 3.0, dtype=torch.float32)),
                ]
            ]
        ),
    ]
    worker._checkpoint_engine_target_devices = [
        torch.device("cpu"),
        torch.device("cpu"),
    ]

    assert _refit(worker)

    assert len(worker.update_calls) == 2, "expected one POST per dtype group"
    for call in worker.update_calls:
        assert len(call["serialized_named_tensors"]) == 2
        assert call["load_format"] == "flattened_bucket"
        assert call["flush_cache"] is False


def test_checkpoint_engine_weight_version_advances_across_refits():
    """``weight_version`` must advance once per refit, not once per POST."""
    worker = _Worker()
    worker._checkpoint_engine_target_devices = [torch.device("cpu")]

    for expected in (1, 2):
        worker.checkpoint_engines = [_Engine(batches=[[("a", torch.ones(1))]])]
        assert _refit(worker)
        assert worker._checkpoint_engine_weight_version == expected
        assert worker.update_calls[-1]["weight_version"] == str(expected)

    assert [call["weight_version"] for call in worker.update_calls] == ["1", "2"]


class _RemappingWorker(_Worker):
    """Worker using the real ``CUDA_VISIBLE_DEVICES`` remapping semantics."""

    def __init__(self, *, visible_devices, **kwargs):
        super().__init__(**kwargs)
        self._visible_devices = visible_devices

    def _to_local_gpu_id(self, physical_gpu_id):
        return self._visible_devices.index(physical_gpu_id)


def test_checkpoint_engine_devices_use_the_remapped_base_gpu_id():
    """``base_gpu_id`` is physical; the receivers must use the local index.

    With ``base_gpu_id=0`` remapping is indistinguishable from doing nothing,
    which is why the other tests cannot cover this.
    """
    worker = _RemappingWorker(visible_devices=[4, 5], gpus_per_node=2)
    worker.base_gpu_id = 4

    assert worker._checkpoint_engine_devices() == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    ]


def test_checkpoint_engine_payload_index_matches_sglang_rank():
    """Payload i must be rank i's shard.

    SGLang indexes the list by its own TP rank, so a transposed payload list
    would load every shard onto the wrong GPU while still succeeding.
    """
    from nemo_rl.models.generation.sglang.utils.train_utils import (
        FlattenedTensorBucket,
        MultiprocessingSerializer,
    )

    worker = _Worker()
    worker._checkpoint_engine_target_devices = [
        torch.device("cpu"),
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    batches = [[("a", torch.tensor([float(rank)]))] for rank in range(3)]

    serialized_by_dtype, _keepalive = worker._serialize_checkpoint_engine_batches(
        batches
    )

    assert len(serialized_by_dtype) == 1
    for rank, payload in enumerate(serialized_by_dtype[0]):
        decoded = MultiprocessingSerializer.deserialize(payload)
        bucket = FlattenedTensorBucket(
            flattened_tensor=decoded["flattened_tensor"],
            metadata=decoded["metadata"],
        )
        torch.testing.assert_close(
            dict(bucket.reconstruct_tensors())["a"], torch.tensor([float(rank)])
        )


def test_transfer_outside_a_weight_update_session_is_rejected():
    """The double's gate has to bite, or every test above proves nothing."""
    worker = _Worker()
    worker.checkpoint_engines = [_Engine(batches=[[("a", torch.ones(1))]])]
    worker._checkpoint_engine_target_devices = [torch.device("cpu")]

    with pytest.raises(AssertionError, match="begin_weight_update"):
        asyncio.run(_load_checkpoint_engine_weights(worker))


def test_refit_rejects_a_sender_that_shipped_nothing():
    """Zero tensors is a failed refit, not a fast one.

    Reporting success here would bump the weight version and let the run keep
    rolling out on stale weights with nothing pointing at why.
    """
    worker = _Worker()
    worker.checkpoint_engines = [_Engine(batches=[])]
    worker._checkpoint_engine_target_devices = [torch.device("cpu")]

    with pytest.raises(RuntimeError, match="received no tensors"):
        _refit(worker)

    assert not hasattr(worker, "_checkpoint_engine_weight_version")


def test_mixin_defines_no_coroutine_members():
    """No coroutine may become a member of this mixin.

    ``SGLangGenerationWorker`` mixes it in and is a ``@ray.remote`` actor. Ray
    picks between a threaded and an asyncio actor with ``has_async_methods``,
    which is ``inspect.getmembers(cls, is_async_func)`` over the whole MRO --
    so one ``async def`` here flips every pre-existing worker RPC into
    asyncio-actor semantics, and because Ray then runs *sync* methods on that
    event loop, the ``asyncio.run`` in the refit entry point raises
    ``asyncio.run() cannot be called from a running event loop`` on the first
    refit. The receive loop lives at module scope for exactly that reason.
    """
    coroutines = [
        name
        for name, _member in inspect.getmembers(
            SGLangCheckpointEngineMixin, inspect.iscoroutinefunction
        )
    ]

    assert coroutines == [], (
        f"{coroutines} would make SGLangGenerationWorker an asyncio actor; "
        "keep the receive loop at module scope"
    )


def test_public_entry_point_drives_a_full_refit():
    """Cover the wrapper the synchronizer actually calls, not just the loop."""
    worker = _Worker()
    worker.checkpoint_engines = [_Engine(batches=[[("a", torch.ones(1))]])]
    worker._checkpoint_engine_target_devices = [torch.device("cpu")]
    worker.begin_weight_update()

    assert worker.update_weights_from_checkpoint_engine()
    assert worker._checkpoint_engine_weight_version == 1
    assert len(worker.update_calls) == 1
