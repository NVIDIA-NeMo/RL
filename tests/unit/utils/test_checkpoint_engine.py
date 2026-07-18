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
import multiprocessing
import sys
import traceback
from collections import defaultdict, deque
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.checkpoint_engine import (
    PolicyCheckpointEngineMixin,
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


class _CheckpointPolicyWorker(PolicyCheckpointEngineMixin, AbstractPolicyWorker):
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


class _FakeNixlTransport:
    def __init__(self) -> None:
        self.agents = {}

    def create_agent(self, agent_name: str):
        agent = _FakeNixlAgent(agent_name, self)
        self.agents[agent_name] = agent
        return agent


class _FakeNixlBackend:
    def __init__(self, owner, transport: _FakeNixlTransport) -> None:
        self.owner = owner
        self.transport = transport
        self.initialize_calls = []
        self.released_handles = []

    def register_memory(self, buffer):
        return buffer

    def deregister_memory(self, _registration):
        return None

    def get_xfer_descs(self, buffer):
        return buffer

    def initialize_xfer(
        self,
        operation,
        local_descs,
        remote_descs,
        remote_agent,
        notify_key,
    ):
        handle = (local_descs, remote_descs, remote_agent, notify_key)
        self.initialize_calls.append((operation, *handle))
        return handle

    def transfer(self, handle):
        local_descs, remote_descs, remote_agent, notify_key = handle
        local_descs.copy_(remote_descs)
        self.transport.agents[remote_agent].notifications[self.owner.agent_name].append(
            notify_key
        )
        return "OK"

    def check_xfer_state(self, _handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released_handles.append(handle)


class _FakeNixlAgent:
    def __init__(self, agent_name: str, transport: _FakeNixlTransport) -> None:
        self.agent_name = agent_name
        self.transport = transport
        self.messages = defaultdict(deque)
        self.notifications = defaultdict(deque)
        self.agent = _FakeNixlBackend(self, transport)

    def get_agent_metadata(self):
        return {"agent_name": self.agent_name}

    def add_remote_agent(self, metadata):
        return metadata["agent_name"]

    def remove_remote_agent(self, _agent_name):
        return None

    def send_message(self, agent_name, message):
        self.transport.agents[agent_name].messages[self.agent_name].append(message)

    async def read_message(self, agent_name):
        while not self.messages[agent_name]:
            await asyncio.sleep(0)
        return self.messages[agent_name].popleft()

    async def wait_notification(self, agent_name, notify_key):
        while notify_key not in self.notifications[agent_name]:
            await asyncio.sleep(0)
        self.notifications[agent_name].remove(notify_key)


def _fake_nixl_pair(monkeypatch, bucket_size: int, device: str = "cpu"):
    transport = _FakeNixlTransport()
    agent_names = iter(("policy", "rollout"))
    monkeypatch.setattr(
        nixl_mod,
        "NixlAgent",
        lambda *_args, **_kwargs: transport.create_agent(next(agent_names)),
    )
    sender = NIXLCheckpointEngine(bucket_size=bucket_size, device=device)
    receiver = NIXLCheckpointEngine(bucket_size=bucket_size, device=device)
    sender.prepare()
    receiver.prepare()
    sender.next_agent = "rollout"
    receiver.prev_agent = "policy"
    return sender, receiver


def _nixl_roundtrip_weights(bucket_size: int) -> list[tuple[str, torch.Tensor]]:
    spanning_numel = 3 * (bucket_size // torch.float32.itemsize) + 5
    return [
        ("small", torch.arange(13, dtype=torch.bfloat16)),
        (
            "spanning",
            torch.arange(spanning_numel, dtype=torch.float32).reshape(-1, 1),
        ),
        ("tail", torch.arange(7, dtype=torch.int64)),
    ]


def _run_nixl_roundtrip_process(
    role: str,
    transfer_device: str,
    bucket_size: int,
    metadata_queue,
    peer_metadata_queue,
    result_queue,
) -> None:
    try:
        torch.cuda.set_device(0)
        engine = NIXLCheckpointEngine(
            bucket_size=bucket_size,
            device=transfer_device,
        )
        metadata_queue.put((role, engine.prepare()))
        metadata = peer_metadata_queue.get(timeout=30)

        if role == "policy":
            engine.init_policy_process_group(
                worker_rank=0,
                train_world_size=1,
                rollout_world_size=1,
                metadata=metadata,
            )
            weights = [
                (name, tensor.cuda())
                for name, tensor in _nixl_roundtrip_weights(bucket_size)
            ]
            asyncio.run(engine.send_weights(iter(weights)))
            details = {}
        else:
            engine.init_rollout_process_group(
                rollout_rank=0,
                train_world_size=1,
                rollout_world_size=1,
                metadata=metadata,
            )

            async def receive_weights():
                received = {}
                async for batch in engine.receive_weight_batches():
                    received.update((name, tensor.clone()) for name, tensor in batch)
                return received

            received = asyncio.run(receive_weights())
            expected = dict(_nixl_roundtrip_weights(bucket_size))
            assert received.keys() == expected.keys()
            for name, tensor in received.items():
                torch.testing.assert_close(tensor.cuda(), expected[name].cuda())
            details = {
                "buffer_devices": [buffer.device.type for buffer in engine.buffers],
                "cupy_buffer_count": len(engine._cupy_buffers),
                "received_devices": {
                    name: tensor.device.type for name, tensor in received.items()
                },
            }

        result_queue.put((role, "success", details))
    except BaseException:
        result_queue.put((role, "error", traceback.format_exc()))
        raise


def _run_nixl_subprocess_roundtrip(
    transfer_device: str, bucket_size: int
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    metadata_queue = context.Queue()
    peer_metadata_queues = {role: context.Queue() for role in ("policy", "rollout")}
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_run_nixl_roundtrip_process,
            args=(
                role,
                transfer_device,
                bucket_size,
                metadata_queue,
                peer_metadata_queues[role],
                result_queue,
            ),
        )
        for role in ("policy", "rollout")
    ]

    try:
        for process in processes:
            process.start()
        metadata = dict(metadata_queue.get(timeout=30) for _ in processes)
        ordered_metadata = [metadata["policy"], metadata["rollout"]]
        for queue in peer_metadata_queues.values():
            queue.put(ordered_metadata)
        for process in processes:
            process.join(timeout=60)

        results = dict(
            (role, (status, details))
            for role, status, details in (
                result_queue.get(timeout=5) for _ in processes
            )
        )
        assert {process.exitcode for process in processes} == {0}, results
        assert {status for status, _details in results.values()} == {"success"}
        return results["rollout"][1]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()


class TestCheckpointEngineABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CheckpointEngine()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self):
        class IncompleteEngine(CheckpointEngine):
            pass

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]


def test_abstract_policy_worker_does_not_enable_checkpoint_engine():
    assert not issubclass(AbstractPolicyWorker, PolicyCheckpointEngineMixin)
    assert not hasattr(AbstractPolicyWorker, "checkpoint_engine_rpc")


def test_checkpoint_engine_helpers():
    engine = create_checkpoint_engine(
        f"{__name__}:_PluginCheckpointEngine",
        bucket_size_bytes=16,
        engine_kwargs={"marker": "ok"},
    )
    assert isinstance(engine, _PluginCheckpointEngine)
    assert (engine.bucket_size, engine.marker) == (16, "ok")
    assert not engine.shard_expert_weights
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
    engine.shard_expert_weights = True

    with pytest.raises(NotImplementedError, match="get_target_weight_layout"):
        engine.get_target_weight_layout()


def test_nixl_checkpoint_engine_rejects_invalid_bucket_size():
    with pytest.raises(ValueError, match="bucket_size must be >= 1"):
        NIXLCheckpointEngine(bucket_size=0, device="cpu")


@pytest.mark.parametrize(
    "rollout_rank,train_world_size,rollout_world_size,error",
    [
        (0, 0, 1, "train_world_size must be >= 1"),
        (0, 1, 0, "rollout_world_size must be >= 1"),
        (-1, 2, 1, "rollout_rank must be in"),
        (1, 2, 1, "rollout_rank must be in"),
        (0, 1, 2, "train_world_size >= rollout_world_size"),
    ],
)
def test_nixl_source_rank_rejects_invalid_topology(
    rollout_rank, train_world_size, rollout_world_size, error
):
    with pytest.raises(ValueError, match=error):
        nixl_mod._source_rank_for_rollout(
            rollout_rank,
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
        )


def test_resolve_nixl_backend_kwargs():
    assert nixl_mod.resolve_nixl_backend_kwargs({}) == ("UCX", None)
    assert nixl_mod.resolve_nixl_backend_kwargs(
        {"backend_name": "GDS", "backend_init_params": {"device": "nvme0"}}
    ) == ("GDS", {"device": "nvme0"})


def test_create_nixl_agent_uses_default_ucx_constructor(monkeypatch):
    backend = MagicMock()
    nixl_api = SimpleNamespace(nixl_agent=MagicMock(return_value=backend))
    monkeypatch.setattr(nixl_mod.importlib, "import_module", lambda _name: nixl_api)

    assert nixl_mod._create_nixl_agent("agent", "UCX") is backend
    nixl_api.nixl_agent.assert_called_once_with("agent")
    backend.create_backend.assert_not_called()


def test_create_nixl_agent_configures_explicit_backend(monkeypatch):
    backend = MagicMock()
    config = object()
    nixl_api = SimpleNamespace(
        nixl_agent=MagicMock(return_value=backend),
        nixl_agent_config=MagicMock(return_value=config),
    )
    monkeypatch.setattr(nixl_mod.importlib, "import_module", lambda _name: nixl_api)

    assert (
        nixl_mod._create_nixl_agent(
            "agent", "UCX", {"engine_config": "MAX_RMA_RAILS=8", "rails": 8}
        )
        is backend
    )
    nixl_api.nixl_agent.assert_called_once_with("agent", config)
    backend.create_backend.assert_called_once_with(
        "UCX", {"engine_config": "MAX_RMA_RAILS=8", "rails": "8"}
    )


def test_create_nixl_agent_reports_missing_dependency(monkeypatch):
    monkeypatch.setattr(
        nixl_mod.importlib,
        "import_module",
        MagicMock(side_effect=ImportError("missing")),
    )

    with pytest.raises(ImportError, match="Install NIXL"):
        nixl_mod._create_nixl_agent("agent", "UCX")


def test_preinit_nixl_agent_initializes_metadata(monkeypatch):
    agent = MagicMock()
    create_agent = MagicMock(return_value=agent)
    monkeypatch.setattr(nixl_mod, "_create_nixl_agent", create_agent)
    monkeypatch.setattr(nixl_mod.uuid, "uuid4", lambda: "id")

    assert (
        nixl_mod.preinit_nixl_agent(
            backend_name="GDS", backend_init_params={"path": "/tmp/device"}
        )
        is agent
    )
    create_agent.assert_called_once_with("preinit-id", "GDS", {"path": "/tmp/device"})
    agent.get_agent_metadata.assert_called_once_with()


def test_sync_device_synchronizes_cuda_device(monkeypatch):
    synchronize = MagicMock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    device = torch.device("cuda", 1)
    nixl_mod._sync_device(device)

    synchronize.assert_called_once_with(device)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_sync_device_flushes_cpu_staging_copy_when_cuda_is_available(
    monkeypatch, cuda_available
):
    stream = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)

    nixl_mod._sync_device(torch.device("cpu"))

    if cuda_available:
        stream.synchronize.assert_called_once_with()
    else:
        stream.synchronize.assert_not_called()


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


def test_policy_worker_checkpoint_engine_rpc_reports_total_memory(monkeypatch):
    worker = _CheckpointPolicyWorker()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    get_device_properties = MagicMock(return_value=SimpleNamespace(total_memory=1234))
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)

    assert _run_checkpoint_rpc(worker, "checkpoint_engine_total_memory_bytes") == 1234
    get_device_properties.assert_called_once_with(2)


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


@pytest.mark.parametrize("bucket_size", [17, 32])
def test_nixl_roundtrip_merges_multibucket_weights_and_alternates_buffers(
    monkeypatch, bucket_size
):
    sender, receiver = _fake_nixl_pair(monkeypatch, bucket_size)
    expected = {
        "small": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
        "spanning": torch.arange(37, dtype=torch.float32).reshape(37, 1),
        "tail": torch.arange(5, dtype=torch.int64),
    }

    async def run_roundtrip():
        send_task = asyncio.create_task(sender.send_weights(iter(expected.items())))
        received = {}
        async for batch in receiver.receive_weight_batches():
            received.update((name, tensor.clone()) for name, tensor in batch)
        await send_task
        return received

    received = asyncio.run(run_roundtrip())

    assert received.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(received[name], tensor)

    calls = receiver.agent.agent.initialize_calls
    assert len(calls) > 2
    assert calls[0][0] == "READ"
    assert calls[0][1] is receiver.buffers[1]
    assert calls[1][1] is receiver.buffers[0]
    assert calls[2][1] is receiver.buffers[1]
    assert len(receiver.agent.agent.released_handles) == len(calls)


def test_nixl_roundtrip_handles_empty_weight_stream(monkeypatch):
    sender, receiver = _fake_nixl_pair(monkeypatch, bucket_size=16)

    async def run_roundtrip():
        send_task = asyncio.create_task(sender.send_weights(iter(())))
        batches = [batch async for batch in receiver.receive_weight_batches()]
        await send_task
        return batches

    assert asyncio.run(run_roundtrip()) == []
    assert len(receiver.agent.agent.initialize_calls) == 1


@pytest.mark.parametrize(
    "transfer_device,bucket_size",
    [("cpu", 4096), ("cpu", 8192), ("cuda", 4096), ("cuda", 8192)],
)
def test_nixl_subprocess_roundtrip(transfer_device, bucket_size):
    pytest.importorskip("nixl._api")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise policy-to-transfer buffer copies")
    if transfer_device == "cuda":
        pytest.importorskip("cupy")

    details = _run_nixl_subprocess_roundtrip(transfer_device, bucket_size)

    assert details["buffer_devices"] == [transfer_device, transfer_device]
    assert details["cupy_buffer_count"] == (2 if transfer_device == "cuda" else 0)
    assert details["received_devices"] == {
        "small": transfer_device,
        "spanning": "cpu",
        "tail": transfer_device,
    }


def test_nixl_cuda_transfer_buffer_falls_back_without_cupy(monkeypatch):
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.bucket_size = 16
    engine._transfer_device = torch.device("cuda", 0)
    engine._cupy_buffers = []
    engine._cupy_memory_pool = None
    engine._uses_torch_cuda_buffers = False
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
    assert engine._uses_torch_cuda_buffers


def test_nixl_cuda_transfer_buffer_uses_private_cupy_pool(monkeypatch):
    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    pool = MagicMock()
    cupy_buffer = object()
    cupy = SimpleNamespace(
        uint8="uint8",
        zeros=MagicMock(return_value=cupy_buffer),
        cuda=SimpleNamespace(
            MemoryPool=MagicMock(return_value=pool),
            Device=MagicMock(return_value=Context()),
            using_allocator=MagicMock(return_value=Context()),
        ),
    )
    torch_buffer = object()
    as_tensor = MagicMock(return_value=torch_buffer)
    set_device = MagicMock()
    monkeypatch.setattr(nixl_mod.importlib, "import_module", lambda _name: cupy)
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch, "as_tensor", as_tensor)

    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.bucket_size = 16
    engine._transfer_device = torch.device("cuda", 0)
    engine._cupy_buffers = []
    engine._cupy_memory_pool = None
    engine._uses_torch_cuda_buffers = False

    assert engine._allocate_transfer_buffer() is torch_buffer
    set_device.assert_called_once_with(torch.device("cuda", 0))
    cupy.cuda.MemoryPool.assert_called_once_with()
    cupy.cuda.Device.assert_called_once_with(0)
    cupy.cuda.using_allocator.assert_called_once_with(pool.malloc)
    cupy.zeros.assert_called_once_with(16, dtype="uint8")
    as_tensor.assert_called_once_with(
        cupy_buffer, dtype=torch.uint8, device=torch.device("cuda", 0)
    )
    assert engine._cupy_buffers == [cupy_buffer]
    assert engine._cupy_memory_pool is pool


def test_nixl_release_transfer_buffers_frees_cupy_pool(monkeypatch):
    backend = MagicMock()
    pool = MagicMock()
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine._transfer_device = torch.device("cpu")
    engine.agent = SimpleNamespace(agent=backend)
    engine.registration_descs = ["registration-0", "registration-1"]
    engine.xfer_descs = ["xfer-0", "xfer-1"]
    engine.buffers = [object(), object()]
    engine._cupy_buffers = [object(), object()]
    engine._cupy_memory_pool = pool
    engine._uses_torch_cuda_buffers = False
    monkeypatch.setattr(nixl_mod, "_sync_device", MagicMock())

    engine._release_transfer_buffers()

    assert [call.args for call in backend.deregister_memory.call_args_list] == [
        ("registration-0",),
        ("registration-1",),
    ]
    assert engine.registration_descs == []
    assert engine.xfer_descs == []
    assert engine.buffers == []
    assert engine._cupy_buffers == []
    pool.free_all_blocks.assert_called_once_with()
    assert engine._cupy_memory_pool is None


def test_nixl_release_torch_cuda_buffers_empties_cache(monkeypatch):
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine._transfer_device = torch.device("cuda", 0)
    engine.agent = SimpleNamespace(agent=MagicMock())
    engine.registration_descs = []
    engine.xfer_descs = []
    engine.buffers = []
    engine._cupy_buffers = []
    engine._cupy_memory_pool = None
    engine._uses_torch_cuda_buffers = True
    empty_cache = MagicMock()
    monkeypatch.setattr(nixl_mod, "_sync_device", MagicMock())
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    engine._release_transfer_buffers()

    empty_cache.assert_called_once_with()
    assert not engine._uses_torch_cuda_buffers


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
    engine.release_after_refit = False

    engine.finalize()

    assert engine.agent.removed == ["policy", "rollout"]
    assert engine.prev_agent is None
    assert engine.next_agent is None
    assert engine.get_target_weight_layout() is None


def test_nixl_release_after_refit_deregisters_and_reregisters_buffers(monkeypatch):
    class FakeBackend:
        def __init__(self):
            self.registered = []
            self.deregistered = []

        def register_memory(self, buffer):
            registration = f"registration-{len(self.registered)}"
            self.registered.append((registration, buffer))
            return registration

        def get_xfer_descs(self, buffer):
            return f"xfer-{len(self.registered)}-{buffer.numel()}"

        def deregister_memory(self, registration):
            self.deregistered.append(registration)

    class FakeAgent:
        instances = []

        def __init__(self, backend_name, backend_init_params):
            self.backend_name = backend_name
            self.backend_init_params = backend_init_params
            self.agent = FakeBackend()
            self.instances.append(self)

        def get_agent_metadata(self):
            return {"agent": len(self.instances)}

    monkeypatch.setattr(nixl_mod, "NixlAgent", FakeAgent)
    engine = NIXLCheckpointEngine(
        bucket_size=16,
        device="cpu",
        backend_name="UCX",
        backend_init_params={"device_list": "mlx5_0:1"},
        release_after_refit=True,
    )

    first_agent = engine.agent
    assert engine.prepare() == {"agent": 1}
    assert len(first_agent.agent.registered) == 2
    assert len(engine.buffers) == len(engine.registration_descs) == 2

    engine.finalize()

    assert first_agent.agent.deregistered == ["registration-0", "registration-1"]
    assert engine.agent is first_agent
    assert engine.buffers == []
    assert engine.registration_descs == []
    assert engine.xfer_descs == []

    assert engine.prepare() == {"agent": 1}
    assert engine.agent is first_agent
    assert len(engine.agent.agent.registered) == 4


def test_nixl_default_finalize_retains_registered_resources(monkeypatch):
    agent = MagicMock()
    agent.get_agent_metadata.return_value = {"agent": "retained"}
    agent.agent.register_memory.side_effect = ["registration-0", "registration-1"]
    agent.agent.get_xfer_descs.side_effect = ["xfer-0", "xfer-1"]
    monkeypatch.setattr(nixl_mod, "NixlAgent", lambda *_args: agent)
    engine = NIXLCheckpointEngine(bucket_size=16, device="cpu")

    engine.prepare()
    buffers = engine.buffers
    engine.finalize()

    assert engine.agent is agent
    assert engine.buffers is buffers
    agent.agent.deregister_memory.assert_not_called()


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


def test_nixl_agent_connects_sends_and_removes_remote():
    backend = MagicMock()
    backend.add_remote_agent.return_value = b"remote"
    socket = MagicMock()
    context = MagicMock()
    context.socket.return_value = socket
    agent = nixl_mod.NixlAgent.__new__(nixl_mod.NixlAgent)
    agent.agent_name = "local"
    agent.agent = backend
    agent.zmq_client_context = context
    agent.zmq_clients = {}

    metadata = {
        "agent_metadata": b"metadata",
        "zmq_ip": "10.0.0.2",
        "zmq_port": 1234,
    }
    assert agent.add_remote_agent(metadata) == "remote"
    socket.connect.assert_called_once_with("tcp://10.0.0.2:1234")

    message = {"key": "value"}
    agent.send_message("remote", message)
    socket.send_pyobj.assert_called_once_with(("local", message), nixl_mod.zmq.DONTWAIT)

    agent.remove_remote_agent("remote")
    backend.remove_remote_agent.assert_called_once_with("remote")
    socket.close.assert_called_once_with(linger=0)
    assert agent.zmq_clients == {}


def test_nixl_agent_reads_messages_and_progresses_backend():
    backend = MagicMock()
    socket = SimpleNamespace(
        recv_pyobj=AsyncMock(
            side_effect=[nixl_mod.zmq.Again(), ("remote", {"value": 1})]
        )
    )
    agent = nixl_mod.NixlAgent.__new__(nixl_mod.NixlAgent)
    agent.agent = backend
    agent.socket = socket
    agent.messages = defaultdict(deque)

    assert asyncio.run(agent.read_message("remote")) == {"value": 1}
    assert backend.progress.call_count == 2
    assert socket.recv_pyobj.await_count == 2


def test_nixl_agent_waits_for_matching_notification():
    backend = MagicMock()
    backend.get_new_notifs.side_effect = [
        {b"remote": [b"other", b"wanted"]},
    ]
    agent = nixl_mod.NixlAgent.__new__(nixl_mod.NixlAgent)
    agent.agent = backend
    agent.notifications = defaultdict(deque)

    asyncio.run(agent.wait_notification("remote", b"wanted"))

    assert list(agent.notifications["remote"]) == [b"other"]
    backend.progress.assert_called_once_with()
    backend.get_new_notifs.assert_called_once_with()


def test_nixl_wait_read_completes_and_releases_handle():
    backend = MagicMock()
    backend.check_xfer_state.side_effect = ["IN_PROGRESS", "DONE"]
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = SimpleNamespace(agent=backend)

    asyncio.run(engine._wait_read("handle", "policy"))

    assert backend.progress.call_count == 2
    assert backend.check_xfer_state.call_count == 2
    backend.release_xfer_handle.assert_called_once_with("handle")


def test_nixl_wait_read_reports_transfer_error():
    backend = MagicMock()
    backend.check_xfer_state.return_value = "ERR"
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = SimpleNamespace(agent=backend)

    with pytest.raises(RuntimeError, match="read from policy failed"):
        asyncio.run(engine._wait_read("handle", "policy"))
    backend.release_xfer_handle.assert_not_called()


def test_nixl_receive_requires_initialized_rollout_process_group():
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.prev_agent = None

    async def receive_one():
        return await anext(engine._receive_weight_chunk_batches())

    with pytest.raises(RuntimeError, match="rollout process group is not initialized"):
        asyncio.run(receive_one())


def test_nixl_receive_reports_transfer_start_error():
    backend = MagicMock()
    backend.initialize_xfer.return_value = "handle"
    backend.transfer.return_value = "ERR"
    agent = SimpleNamespace(
        agent=backend,
        read_message=AsyncMock(
            return_value={
                "remote_descs": "remote",
                "notify_key": b"key",
                "bucket_meta": {},
                "is_last": True,
            }
        ),
    )
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.prev_agent = "policy"
    engine.agent = agent
    engine.buffers = [torch.zeros(8, dtype=torch.uint8) for _ in range(2)]
    engine.xfer_descs = ["local-0", "local-1"]

    async def receive_one():
        return await anext(engine._receive_weight_chunk_batches())

    with pytest.raises(RuntimeError, match="failed to start"):
        asyncio.run(receive_one())
    backend.initialize_xfer.assert_called_once_with(
        "READ", "local-1", "remote", "policy", b"key"
    )


def test_nixl_process_group_uses_parallel_policy_to_rollout_topology():
    def engine_with_agent():
        engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
        engine.prev_agent = None
        engine.next_agent = None
        engine._target_weight_layout = None
        engine.shard_expert_weights = True
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
