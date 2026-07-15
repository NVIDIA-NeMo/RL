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

"""Unit tests for checkpoint-engine primitives and policy-worker integration."""

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.checkpoint_engine import (
    maybe_preinit_nixl_checkpoint_engine,
)
from nemo_rl.utils.checkpoint_engines import nixl as nixl_mod
from nemo_rl.utils.checkpoint_engines.base import (
    CheckpointEngine,
    TensorMeta,
    create_checkpoint_engine,
    merge_weight_chunk_batches,
    split_weight_chunks,
)
from nemo_rl.utils.checkpoint_engines.nixl import NIXLCheckpointEngine


class _PluginCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int, marker: str) -> None:
        self.bucket_size, self.marker = bucket_size, marker

    def prepare(self):
        return {"marker": self.marker}

    def init_policy_process_group(
        self,
        *,
        worker_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        pass

    def init_rollout_process_group(
        self,
        *,
        rollout_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        pass

    async def send_weights(self, weights):
        pass

    async def receive_weight_batches(self):
        pass


class _RecordingCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int) -> None:
        self.bucket_size = bucket_size
        self.policy_process_group = None
        self.sent_weights = None
        self.finalized = False

    def prepare(self):
        return {"bucket_size": self.bucket_size}

    def init_policy_process_group(
        self,
        *,
        worker_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        self.policy_process_group = {
            "worker_rank": worker_rank,
            "train_world_size": train_world_size,
            "rollout_world_size": rollout_world_size,
            "metadata": metadata,
        }

    def init_rollout_process_group(
        self,
        *,
        rollout_rank,
        train_world_size,
        rollout_world_size,
        metadata,
    ):
        pass

    def finalize(self) -> None:
        self.finalized = True

    async def send_weights(self, weights):
        self.sent_weights = list(weights)

    async def receive_weight_batches(self):
        pass


class _CheckpointPolicyWorker(AbstractPolicyWorker):
    def __init__(self) -> None:
        self.rank = 3
        self.events = []
        self.kv_scales = None

    def _checkpoint_engine_weight_iterator(self, kv_scales=None):
        self.kv_scales = kv_scales
        yield "weight", torch.tensor([1.0, 2.0])

    def _prepare_checkpoint_engine_weight_send(self) -> None:
        self.events.append("prepare")

    def _finalize_checkpoint_engine_weight_send(self) -> None:
        self.events.append("finalize")


def _run_checkpoint_rpc(
    worker: _CheckpointPolicyWorker,
    checkpoint_method: str,
    method_kwargs: dict | None = None,
):
    return asyncio.run(
        worker.checkpoint_engine_rpc(
            checkpoint_method,
            method_kwargs=method_kwargs,
        )
    )


class TestCheckpointEngineABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CheckpointEngine()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self):
        class IncompleteEngine(CheckpointEngine):
            pass

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]


def test_checkpoint_engine_helpers():
    engine = create_checkpoint_engine(
        f"{__name__}:_PluginCheckpointEngine",
        bucket_size_bytes=16,
        engine_kwargs={"marker": "ok"},
    )
    assert isinstance(engine, _PluginCheckpointEngine)
    assert (engine.bucket_size, engine.marker) == (16, "ok")
    assert engine.cleanup_after_load
    assert not engine.shard_hf_weights
    assert engine.get_target_weight_layout() is None

    async def roundtrip(bucket_size):
        async def batches():
            for chunk in split_weight_chunks(iter([("weight", tensor)]), bucket_size):
                yield [chunk]

        merged = []
        async for batch in merge_weight_chunk_batches(batches()):
            merged.extend(batch)
        return merged

    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    for bucket_size in (17, 1024):
        merged = asyncio.run(roundtrip(bucket_size))
        assert merged[0][0] == "weight"
        torch.testing.assert_close(merged[0][1], tensor)


def test_sharded_plugin_requires_target_weight_layout_accessor():
    engine = _PluginCheckpointEngine(bucket_size=16, marker="sharded")
    engine.shard_hf_weights = True

    with pytest.raises(NotImplementedError, match="get_target_weight_layout"):
        engine.get_target_weight_layout()


def test_nixl_checkpoint_engine_rejects_invalid_bucket_size():
    with pytest.raises(ValueError, match="bucket_size must be >= 1"):
        NIXLCheckpointEngine(bucket_size=0, device="cpu")


def test_maybe_preinit_nixl_checkpoint_engine_defaults_backend(monkeypatch):
    calls = []
    fake_nixl = ModuleType("nemo_rl.utils.checkpoint_engines.nixl")
    fake_nixl.NIXL_DEFAULT_BACKEND_NAME = "UCX"

    def preinit_nixl_agent(**kwargs):
        calls.append(kwargs)
        return "agent"

    def resolve_nixl_backend_kwargs(nixl_kwargs):
        return (
            nixl_kwargs.get("backend_name", "UCX"),
            nixl_kwargs.get("backend_init_params"),
        )

    fake_nixl.preinit_nixl_agent = preinit_nixl_agent
    fake_nixl.resolve_nixl_backend_kwargs = resolve_nixl_backend_kwargs
    monkeypatch.setitem(sys.modules, "nemo_rl.utils.checkpoint_engines.nixl", fake_nixl)

    assert maybe_preinit_nixl_checkpoint_engine({}) is None
    assert (
        maybe_preinit_nixl_checkpoint_engine(
            {
                "generation": {
                    "checkpoint_engine": {
                        "enabled": True,
                        "backend": "nixl",
                        "engine_kwargs": {"nixl": {}},
                    }
                }
            }
        )
        == "agent"
    )
    assert calls == [{"backend_name": "UCX", "backend_init_params": None}]


def test_merge_weight_chunk_batches_uses_aligned_zero_copy_view():
    fp32_w = torch.tensor([7.0, 8.0], dtype=torch.float32)
    bucket = torch.zeros(64, dtype=torch.uint8)
    offset = 8
    raw = fp32_w.view(torch.uint8)
    bucket[offset : offset + fp32_w.nbytes].copy_(raw)
    chunk = bucket[offset : offset + fp32_w.nbytes]
    meta = TensorMeta(
        "fp32_w",
        fp32_w.shape,
        fp32_w.dtype,
        chunk_offset=0,
        chunk_size=fp32_w.nbytes,
        offset=offset,
    )

    async def run():
        async def batches():
            yield [(meta, chunk)]

        merged = []
        async for batch in merge_weight_chunk_batches(batches()):
            merged.extend(batch)
        return dict(merged)

    merged = asyncio.run(run())
    torch.testing.assert_close(merged["fp32_w"], fp32_w)
    assert (
        merged["fp32_w"].untyped_storage().data_ptr()
        == bucket.untyped_storage().data_ptr()
    )


def test_merge_weight_chunk_batches_uses_requested_device(monkeypatch):
    tensor = torch.arange(8, dtype=torch.float32)
    chunks = list(split_weight_chunks(iter([("weight", tensor)]), tensor.nbytes // 2))
    devices = []
    torch_empty = torch.empty

    def record_empty(*args, **kwargs):
        devices.append(kwargs["device"])
        return torch_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", record_empty)

    async def run():
        async def batches():
            for chunk in chunks:
                yield [chunk]

        return [
            batch
            async for batch in merge_weight_chunk_batches(batches(), merge_device="cpu")
        ]

    merged = asyncio.run(run())

    assert devices == ["cpu"]
    torch.testing.assert_close(merged[0][0][1], tensor)


def test_policy_worker_checkpoint_engine_rpc_runs_weight_send():
    worker = _CheckpointPolicyWorker()
    _run_checkpoint_rpc(
        worker,
        "init_checkpoint_engine",
        {
            "backend": f"{__name__}:_RecordingCheckpointEngine",
            "bucket_size_bytes": 32,
            "engine_kwargs": {},
        },
    )

    assert _run_checkpoint_rpc(worker, "prepare_checkpoint_engine") == {
        "bucket_size": 32,
        "rank": 3,
    }
    _run_checkpoint_rpc(
        worker,
        "init_checkpoint_engine_process_group",
        {
            "train_world_size": 2,
            "rollout_world_size": 1,
            "metadata": ["p0", "p1", "g0"],
        },
    )
    _run_checkpoint_rpc(
        worker,
        "send_weights_via_checkpoint_engine",
        {"kv_scales": {"scale": 1.0}},
    )

    assert worker.checkpoint_engine.policy_process_group == {
        "worker_rank": 3,
        "train_world_size": 2,
        "rollout_world_size": 1,
        "metadata": ["p0", "p1", "g0"],
    }
    assert worker.kv_scales == {"scale": 1.0}
    assert worker.events == ["prepare", "finalize"]
    sent_name, sent_tensor = worker.checkpoint_engine.sent_weights[0]
    assert sent_name == "weight"
    torch.testing.assert_close(sent_tensor, torch.tensor([1.0, 2.0]))

    _run_checkpoint_rpc(worker, "finalize_checkpoint_engine")
    assert worker.checkpoint_engine.finalized


def test_policy_worker_checkpoint_engine_rpc_sends_from_running_event_loop():
    worker = _CheckpointPolicyWorker()
    _run_checkpoint_rpc(
        worker,
        "init_checkpoint_engine",
        {
            "backend": f"{__name__}:_RecordingCheckpointEngine",
            "bucket_size_bytes": 32,
            "engine_kwargs": {},
        },
    )

    async def run_send() -> None:
        await worker.checkpoint_engine_rpc("send_weights_via_checkpoint_engine")

    asyncio.run(run_send())

    sent_name, sent_tensor = worker.checkpoint_engine.sent_weights[0]
    assert sent_name == "weight"
    torch.testing.assert_close(sent_tensor, torch.tensor([1.0, 2.0]))


def test_nixl_send_weights_drains_iterator_without_rollout_peer():
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.next_agent = None
    consumed = []

    def weights():
        consumed.append("started")
        yield "weight", torch.tensor([1.0])
        consumed.append("finished")

    asyncio.run(engine.send_weights(weights()))

    assert consumed == ["started", "finished"]


def test_nixl_send_weights_aligns_bucket_offsets_for_dtype_views():
    class FakeAgent:
        def __init__(self):
            self.messages = []

        def send_message(self, _agent_name, message):
            self.messages.append(message)

        async def wait_notification(self, _agent_name, _notify_key):
            return None

    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.next_agent = "rollout"
    engine.buffers = [
        torch.zeros(64, dtype=torch.uint8),
        torch.zeros(64, dtype=torch.uint8),
    ]
    engine.xfer_descs = ["desc0", "desc1"]
    engine.bucket_size = 64
    engine._transfer_device = torch.device("cpu")
    engine.agent = FakeAgent()

    weights = iter(
        [
            ("bf16_w", torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)),
            ("fp32_w", torch.tensor([7.0], dtype=torch.float32)),
        ]
    )

    asyncio.run(engine.send_weights(weights))

    [message] = engine.agent.messages
    assert message["bucket_meta"]["bf16_w"].offset == 0
    assert message["bucket_meta"]["fp32_w"].offset == 8
    assert message["bucket_meta"]["fp32_w"].offset % torch.float32.itemsize == 0


def test_nixl_cuda_transfer_buffer_falls_back_without_cupy(monkeypatch):
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.bucket_size = 16
    engine._transfer_device = torch.device("cuda", 0)
    engine._cupy_buffers = []
    set_device_calls = []
    zeros_calls = []

    monkeypatch.setattr(torch.cuda, "set_device", set_device_calls.append)
    monkeypatch.setattr(
        "nemo_rl.utils.checkpoint_engines.nixl.importlib.import_module",
        MagicMock(side_effect=ImportError("cupy unavailable")),
    )

    def fake_zeros(*args, **kwargs):
        zeros_calls.append((args, kwargs))
        return "buffer"

    monkeypatch.setattr(torch, "zeros", fake_zeros)

    assert engine._allocate_transfer_buffer() == "buffer"
    assert set_device_calls == [torch.device("cuda", 0)]
    assert zeros_calls == [
        (
            (16,),
            {"dtype": torch.uint8, "device": torch.device("cuda", 0)},
        )
    ]
    assert engine._cupy_buffers == []


def test_nixl_finalize_disconnects_peers():
    class FakeAgent:
        def __init__(self):
            self.removed = []

        def remove_remote_agent(self, agent_name):
            self.removed.append(agent_name)

    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = FakeAgent()
    engine.prev_agent = "policy"
    engine.next_agent = "rollout"
    engine._target_weight_layout = {"layer": {}}

    engine.finalize()

    assert engine.agent.removed == ["policy", "rollout"]
    assert engine.prev_agent is None
    assert engine.next_agent is None
    assert engine.get_target_weight_layout() is None


def test_nixl_agent_binds_zmq_socket_atomically(monkeypatch):
    class FakeNixlBackend:
        def get_agent_metadata(self):
            return {"backend": "metadata"}

    class FakePushContext:
        pass

    class FakePullSocket:
        def __init__(self):
            self.bind_endpoints = []

        def bind_to_random_port(self, endpoint):
            self.bind_endpoints.append(endpoint)
            return 45678

    class FakePullContext:
        def __init__(self, socket):
            self._socket = socket

        def socket(self, socket_type):
            assert socket_type == nixl_mod.zmq.PULL
            return self._socket

    pull_socket = FakePullSocket()
    monkeypatch.setattr(
        nixl_mod,
        "_create_nixl_agent",
        lambda agent_name, backend_name, backend_init_params: FakeNixlBackend(),
    )
    monkeypatch.setattr(nixl_mod.ray.util, "get_node_ip_address", lambda: "10.10.0.12")
    monkeypatch.setattr(nixl_mod.zmq, "Context", lambda: FakePushContext())
    monkeypatch.setattr(
        nixl_mod.zmq.asyncio, "Context", lambda: FakePullContext(pull_socket)
    )

    agent = nixl_mod.NixlAgent()

    assert agent.listen_port == 45678
    assert pull_socket.bind_endpoints == ["tcp://10.10.0.12"]
    assert agent.get_agent_metadata()["zmq_port"] == 45678


def test_nixl_process_group_uses_parallel_policy_to_rollout_topology():
    def engine_with_agent():
        engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
        engine.prev_agent = None
        engine.next_agent = None
        engine._target_weight_layout = None
        engine.shard_hf_weights = True
        engine.agent = MagicMock()
        engine.agent.add_remote_agent.side_effect = lambda metadata: metadata["name"]
        return engine

    rollout_0_layout = {"layer.0": {}}
    rollout_1_layout = {"layer.1": {}}
    rollout_2_layout = {"layer.2": {}}
    metadata = [
        {"name": "policy-0"},
        {"name": "policy-1"},
        {"name": "policy-2"},
        {"name": "policy-3"},
        {"name": "rollout-0", "weight_layout": rollout_0_layout},
        {"name": "rollout-1", "weight_layout": rollout_1_layout},
        {"name": "rollout-2", "weight_layout": rollout_2_layout},
    ]

    policy_rank_0 = engine_with_agent()
    policy_rank_0.init_policy_process_group(
        worker_rank=0,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert policy_rank_0.next_agent == "rollout-0"
    assert policy_rank_0.get_target_weight_layout() is rollout_0_layout

    policy_rank_1 = engine_with_agent()
    policy_rank_1.init_policy_process_group(
        worker_rank=1,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert policy_rank_1.next_agent == "rollout-1"
    assert policy_rank_1.get_target_weight_layout() is rollout_1_layout

    policy_rank_3 = engine_with_agent()
    policy_rank_3.init_policy_process_group(
        worker_rank=3,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert policy_rank_3.next_agent is None
    assert policy_rank_3.get_target_weight_layout() is None
    policy_rank_3.agent.add_remote_agent.assert_not_called()

    rollout_rank_0 = engine_with_agent()
    rollout_rank_0.init_rollout_process_group(
        rollout_rank=0,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert (rollout_rank_0.prev_agent, rollout_rank_0.next_agent) == ("policy-0", None)

    rollout_rank_1 = engine_with_agent()
    rollout_rank_1.init_rollout_process_group(
        rollout_rank=1,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert (rollout_rank_1.prev_agent, rollout_rank_1.next_agent) == ("policy-1", None)

    rollout_rank_2 = engine_with_agent()
    rollout_rank_2.init_rollout_process_group(
        rollout_rank=2,
        train_world_size=4,
        rollout_world_size=3,
        metadata=metadata,
    )
    assert (rollout_rank_2.prev_agent, rollout_rank_2.next_agent) == (
        "policy-2",
        None,
    )
