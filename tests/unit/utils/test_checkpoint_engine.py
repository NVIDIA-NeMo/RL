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
import sys
import types
from collections import defaultdict, deque
from threading import Lock

import pytest
import torch

import nemo_rl.utils.checkpoint_engines.nixl as nixl_module
from nemo_rl.utils.checkpoint_engine import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunk_batches,
    split_weight_chunks,
)
from nemo_rl.utils.checkpoint_engines.nixl import (
    NixlAgentMetadata,
    NIXLCheckpointEngine,
)


class _DummyNixlAgent:
    def __init__(self):
        self.closed = False
        self.registered = []
        self.xfer_descs = []
        self.released_xfers = []

    def add_remote_agent(self, metadata):
        raise AssertionError(f"idle rank should not add remote agents: {metadata}")

    def remove_remote_agent(self, agent_name):
        raise AssertionError(f"idle rank should not remove remote agents: {agent_name}")

    def register_memory(self, buf):
        desc = f"reg_{len(self.registered)}"
        self.registered.append(buf)
        return desc

    def get_xfer_descs(self, buf):
        desc = f"xfer_{len(self.xfer_descs)}"
        self.xfer_descs.append(buf)
        return desc

    def release_xfer_handle(self, xfer_handle):
        self.released_xfers.append(xfer_handle)

    def get_agent_metadata(self):
        return {
            "agent_name": "dummy",
            "agent_metadata": b"",
            "zmq_ip": "127.0.0.1",
            "zmq_port": 0,
        }

    def close(self):
        self.closed = True


class _RecordingNixlAgent(_DummyNixlAgent):
    def __init__(self):
        super().__init__()
        self.added = []
        self.removed = []
        self.sent_messages = []
        self.sent_payloads = []
        self.notifications = {}
        self.ignored_notifications = []
        self.completions = []

    def add_remote_agent(self, metadata):
        agent_name = metadata["agent_name"]
        self.added.append(agent_name)
        return agent_name

    def remove_remote_agent(self, agent_name):
        self.removed.append(agent_name)

    def send_message(self, agent_name, message):
        self.sent_payloads.append((agent_name, message))
        self.sent_messages.append((agent_name, message))
        if "notify_key" in message:
            self.notifications.setdefault(agent_name, []).append(message["notify_key"])

    def send_messages(self, agent_name, messages):
        if len(messages) == 1:
            self.send_message(agent_name, messages[0])
            return
        self.sent_payloads.append(
            (
                agent_name,
                {
                    "message_type": nixl_module._MESSAGE_BUCKET_METADATA_BATCH,
                    "messages": messages,
                },
            )
        )
        for message in messages:
            self.sent_messages.append((agent_name, message))
            if "notify_key" in message:
                self.notifications.setdefault(agent_name, []).append(
                    message["notify_key"]
                )

    async def get_notification(self, agent_name):
        return self.notifications[agent_name].pop(0)

    def ignore_notification(self, agent_name, notify_key):
        self.ignored_notifications.append((agent_name, notify_key))

    async def get_completion(self, agent_name, notify_key):
        self.completions.append((agent_name, notify_key))

    async def drain_notifications(self):
        return None


class _TransferNixlAgent(_RecordingNixlAgent):
    def __init__(self):
        super().__init__()
        self.initialized_xfers = []
        self.transferred = []
        self.checked = []
        self.progress_calls = 0

    def initialize_xfer(
        self, operation, local_descs, remote_descs, remote_agent, notify_key
    ):
        handle = f"handle_{len(self.initialized_xfers)}"
        self.initialized_xfers.append(
            (operation, local_descs, remote_descs, remote_agent, notify_key)
        )
        return handle

    def transfer(self, xfer_handle):
        self.transferred.append(xfer_handle)
        return "DONE"

    def check_xfer_state(self, xfer_handle):
        return "DONE"

    def progress(self):
        self.progress_calls += 1
        return True


class _StatefulTransferNixlAgent(_TransferNixlAgent):
    def __init__(self, *, transfer_state="DONE", check_states=None):
        super().__init__()
        self.transfer_state = transfer_state
        self.check_states = deque(check_states or ["DONE"])

    def transfer(self, xfer_handle):
        self.transferred.append(xfer_handle)
        return self.transfer_state

    def check_xfer_state(self, xfer_handle):
        self.checked.append(xfer_handle)
        if self.check_states:
            return self.check_states.popleft()
        return "DONE"


class _QueuedTransferNixlAgent(_TransferNixlAgent):
    def __init__(self, messages):
        super().__init__()
        self.messages = deque()
        for message in messages:
            self._queue_message(message)

    def _queue_message(self, message):
        if message.get("message_type") == nixl_module._MESSAGE_BUCKET_METADATA_BATCH:
            self.messages.extend(message["messages"])
            return
        self.messages.append(message)

    async def read_message(self, agent_name):
        assert agent_name == "policy0"
        return self.messages.popleft()

    async def try_read_message(self, agent_name):
        assert agent_name == "policy0"
        if not self.messages:
            return None
        return self.messages.popleft()


class _RawNixlAgent:
    def __init__(self, remote_agent_name=b"remote0"):
        self.remote_agent_name = remote_agent_name
        self.added = []
        self.removed = []
        self.progress_calls = 0
        self.registered = []
        self.xfer_descs = []
        self.initialized_xfers = []
        self.transferred = []
        self.checked = []
        self.released = []
        self.native_notifications = deque()

    def get_agent_metadata(self):
        return b"local-metadata"

    def add_remote_agent(self, metadata):
        self.added.append(metadata)
        return self.remote_agent_name

    def remove_remote_agent(self, agent_name):
        self.removed.append(agent_name)

    def progress(self):
        self.progress_calls += 1

    def get_new_notifs(self):
        if self.native_notifications:
            return self.native_notifications.popleft()
        return {}

    def register_memory(self, buffer):
        self.registered.append(buffer)
        return "registered"

    def get_xfer_descs(self, buffer):
        self.xfer_descs.append(buffer)
        return "descs"

    def initialize_xfer(
        self, operation, local_descs, remote_descs, remote_agent, notify_key
    ):
        self.initialized_xfers.append(
            (operation, local_descs, remote_descs, remote_agent, notify_key)
        )
        return "xfer"

    def transfer(self, xfer_handle):
        self.transferred.append(xfer_handle)
        return "DONE"

    def check_xfer_state(self, xfer_handle):
        self.checked.append(xfer_handle)
        return "DONE"

    def release_xfer_handle(self, xfer_handle):
        self.released.append(xfer_handle)


class _FakeZmqSocket:
    def __init__(self, socket_type):
        self.socket_type = socket_type
        self.options = []
        self.bound = []
        self.connected = []
        self.closed = []
        self.sent = []
        self.recv_queue = deque()

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def bind(self, address):
        self.bound.append(address)

    def connect(self, address):
        self.connected.append(address)

    def close(self, *, linger=0):
        self.closed.append(linger)

    def send_pyobj(self, payload, flags):
        self.sent.append((payload, flags))

    async def poll(self, timeout):
        return bool(self.recv_queue)

    async def recv_pyobj(self):
        return self.recv_queue.popleft()


class _FakeZmqContext:
    def __init__(self):
        self.sockets = []
        self.destroyed = []

    def socket(self, socket_type):
        socket = _FakeZmqSocket(socket_type)
        self.sockets.append(socket)
        return socket

    def destroy(self, *, linger=0):
        self.destroyed.append(linger)


class _PluginCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size, plugin_arg):
        self.bucket_size = bucket_size
        self.plugin_arg = plugin_arg

    def prepare(self):
        return None

    def init_policy_process_group(
        self, *, worker_rank, train_world_size, rollout_world_size, metadata
    ):
        pass

    def init_rollout_process_group(
        self, *, rollout_rank, train_world_size, rollout_world_size, metadata
    ):
        pass

    def finalize(self):
        pass

    async def send_weights(self, weights):
        pass

    async def receive_weight_batches(self):
        if False:
            yield []


def _metadata(agent_name: str) -> NixlAgentMetadata:
    return {
        "agent_name": agent_name,
        "agent_metadata": b"",
        "zmq_ip": "127.0.0.1",
        "zmq_port": 0,
    }


async def _collect_weight_batches(chunks, bucket_size):
    results = []
    async for batch in merge_weight_chunk_batches(chunks, bucket_size):
        results.extend((name, weight.clone()) for name, weight in batch)
    return results


def _expect_value_error(fn, expected_message: str):
    try:
        fn()
    except ValueError as exc:
        assert expected_message in str(exc)
    else:
        raise AssertionError("Expected NIXL configuration validation to fail.")


def test_split_and_merge_weight_chunks_roundtrip():
    weights = [
        ("small", torch.arange(4, dtype=torch.float32)),
        ("large", torch.arange(18, dtype=torch.float32).reshape(3, 6)),
    ]
    bucket_size = 16

    async def run_roundtrip():
        async def chunk_batches():
            async for tensor_meta, chunk in split_weight_chunks(
                iter(weights), bucket_size
            ):
                yield [(tensor_meta, chunk)]

        return await _collect_weight_batches(chunk_batches(), bucket_size)

    merged = asyncio.run(run_roundtrip())

    assert [name for name, _weight in merged] == ["small", "large"]
    assert torch.equal(merged[0][1], weights[0][1])
    assert torch.equal(merged[1][1], weights[1][1])


def test_merge_weight_chunk_batches_preserves_complete_weight_batches():
    small = torch.arange(4, dtype=torch.float32)
    large = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    bucket_size = 16

    async def chunk_batches():
        small_buffer = small.view(-1).view(torch.uint8)
        yield [
            (
                TensorMeta(
                    name="small",
                    shape=small.shape,
                    dtype=small.dtype,
                    chunk_offset=0,
                    chunk_size=small.nbytes,
                    offset=0,
                ),
                small_buffer,
            )
        ]

        async for tensor_meta, chunk in split_weight_chunks(
            iter([("large", large)]), bucket_size
        ):
            yield [(tensor_meta, chunk)]

    async def run_roundtrip():
        results = []
        async for batch in merge_weight_chunk_batches(chunk_batches(), bucket_size):
            results.append([(name, weight.clone()) for name, weight in batch])
        return results

    batches = asyncio.run(run_roundtrip())

    assert [[name for name, _weight in batch] for batch in batches] == [
        ["small"],
        ["large"],
    ]
    assert torch.equal(batches[0][0][1], small)
    assert torch.equal(batches[1][0][1], large)


def test_merge_weight_chunk_batches_unpacks_packed_metadata():
    first = torch.arange(4, dtype=torch.float32)
    second = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    packed = torch.empty(first.nbytes + second.nbytes, dtype=torch.uint8)
    packed[: first.nbytes] = first.view(-1).view(torch.uint8)
    packed[first.nbytes :] = second.view(-1).view(torch.uint8)
    parent = TensorMeta(
        name="__nixl_packed_0",
        shape=packed.shape,
        dtype=packed.dtype,
        chunk_offset=0,
        chunk_size=packed.nbytes,
        offset=0,
        children=[
            TensorMeta(
                name="first",
                shape=first.shape,
                dtype=first.dtype,
                chunk_offset=0,
                chunk_size=first.nbytes,
                offset=0,
            ),
            TensorMeta(
                name="second",
                shape=second.shape,
                dtype=second.dtype,
                chunk_offset=0,
                chunk_size=second.nbytes,
                offset=first.nbytes,
            ),
        ],
    )

    async def chunk_batches():
        yield [(parent, packed)]

    merged = asyncio.run(_collect_weight_batches(chunk_batches(), packed.nbytes))

    assert [name for name, _weight in merged] == ["first", "second"]
    assert torch.equal(merged[0][1], first)
    assert torch.equal(merged[1][1], second)


def test_checkpoint_engine_registry_loads_class_path_plugin():
    engine = CheckpointEngineRegistry.new(
        f"{__name__}:_PluginCheckpointEngine",
        bucket_size=123,
        plugin_arg="ok",
    )

    assert isinstance(engine, _PluginCheckpointEngine)
    assert engine.bucket_size == 123
    assert engine.plugin_arg == "ok"


def test_nixl_helper_functions_cover_optional_and_cuda_paths(monkeypatch):
    assert nixl_module._normalize_non_negative_int(None, default=3, name="x") == 3
    assert nixl_module._normalize_non_negative_int(4, default=3, name="x") == 4
    with pytest.raises(ValueError, match="x must be >= 0"):
        nixl_module._normalize_non_negative_int(-1, default=0, name="x")

    assert nixl_module._normalize_positive_int(None, default=2, name="y") == 2
    assert nixl_module._normalize_positive_int(5, default=2, name="y") == 5
    with pytest.raises(ValueError, match="y must be >= 1"):
        nixl_module._normalize_positive_int(0, default=1, name="y")

    assert nixl_module._normalize_bool(None, default=True) is True
    assert nixl_module._normalize_bool(0, default=True) is False
    assert nixl_module._is_valid_ipv6_address("::1")
    assert not nixl_module._is_valid_ipv6_address("127.0.0.1")
    assert nixl_module._tcp_address("::1", 1234) == "tcp://[::1]:1234"
    assert nixl_module._tcp_address("127.0.0.1", 1234) == "tcp://127.0.0.1:1234"
    assert nixl_module._get_free_port("127.0.0.1") > 0

    assert nixl_module._optional_module("types") is types
    assert nixl_module._optional_module("missing_nixl_test_module") is None

    monkeypatch.setattr(nixl_module.torch.cuda, "current_device", lambda: 1)
    assert nixl_module._normalize_device("cpu") == torch.device("cpu")
    assert nixl_module._normalize_device("cuda") == torch.device("cuda", 1)

    synced = []
    monkeypatch.setattr(
        nixl_module.torch.cuda, "synchronize", lambda device: synced.append(device)
    )
    nixl_module._sync_devices(
        [
            torch.device("cpu"),
            torch.device("cuda"),
            torch.device("cuda", 2),
            torch.device("cuda", 2),
        ]
    )
    assert synced == [torch.device("cuda", 1), torch.device("cuda", 2)]

    monkeypatch.setattr(nixl_module.torch.cuda, "is_available", lambda: False)
    assert nixl_module._cuda_fence_devices(
        [torch.device("cpu"), torch.device("cuda"), torch.device("cuda", 3)]
    ) == (torch.device("cuda", 3),)

    monkeypatch.setattr(nixl_module.torch.cuda, "is_available", lambda: True)
    assert set(
        nixl_module._cuda_fence_devices(
            [torch.device("cuda"), torch.device("cuda", 3)],
            include_current_device=True,
        )
    ) == {torch.device("cuda", 1), torch.device("cuda", 3)}

    class _PendingEvent:
        def __init__(self):
            self.queries = 0

        def query(self):
            self.queries += 1
            return self.queries > 1

    event = _PendingEvent()
    asyncio.run(nixl_module._wait_cuda_events([event]))
    assert event.queries == 2


def test_nixl_import_shims_and_module_loading(monkeypatch):
    real_import_module = nixl_module.importlib.import_module

    def missing_torch_version(module_name):
        if module_name == "torch.version":
            raise ModuleNotFoundError(module_name)
        return real_import_module(module_name)

    monkeypatch.setattr(nixl_module.importlib, "import_module", missing_torch_version)
    monkeypatch.delitem(sys.modules, "torch.version", raising=False)
    monkeypatch.setattr(nixl_module.torch, "version", types.SimpleNamespace())
    monkeypatch.setattr(nixl_module.torch, "__version__", "2.6.0+cu124")
    nixl_module._ensure_torch_version_module()
    assert sys.modules["torch.version"].cuda == "12.4"

    monkeypatch.delitem(sys.modules, "torch.version", raising=False)
    monkeypatch.setattr(nixl_module.torch, "version", types.SimpleNamespace())
    monkeypatch.setattr(nixl_module.torch, "__version__", "2.6.0")
    monkeypatch.setenv("CUDA_VERSION", "12.8.1")
    nixl_module._ensure_torch_version_module()
    assert sys.modules["torch.version"].cuda == "12.8"

    fake_numpy = types.SimpleNamespace()
    monkeypatch.setattr(
        nixl_module.importlib,
        "import_module",
        lambda module_name: fake_numpy
        if module_name == "numpy"
        else real_import_module(module_name),
    )
    nixl_module._ensure_numpy_annotation_attrs()
    assert fake_numpy.ndarray is object

    monkeypatch.setattr(
        nixl_module.importlib,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(ModuleNotFoundError(module_name))
        if module_name == "numpy"
        else real_import_module(module_name),
    )
    nixl_module._ensure_numpy_annotation_attrs()

    calls = []
    module = object()
    monkeypatch.setattr(
        nixl_module, "_ensure_torch_version_module", lambda: calls.append("torch")
    )
    monkeypatch.setattr(
        nixl_module, "_ensure_numpy_annotation_attrs", lambda: calls.append("numpy")
    )
    monkeypatch.setattr(nixl_module.importlib, "import_module", lambda _name: module)
    assert nixl_module._require_module("nixl._api", "hint") is module
    assert calls == ["torch", "numpy"]

    monkeypatch.setattr(
        nixl_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("boom")),
    )
    with pytest.raises(ImportError, match="required for NIXL checkpoint-engine refit"):
        nixl_module._require_module("missing_mod", "install it")


def test_nixl_create_agent_and_preinit_use_backend_params(monkeypatch):
    class _CreatedAgent:
        def __init__(self):
            self.backends = []
            self.metadata_calls = 0

        def create_backend(self, backend_name, init_params):
            self.backends.append((backend_name, init_params))

        def get_agent_metadata(self):
            self.metadata_calls += 1
            return b"metadata"

    class _FakeNixlApi:
        def __init__(self):
            self.config_calls = []
            self.agent_calls = []

        def nixl_agent_config(self, *, backends):
            self.config_calls.append(backends)
            return "config"

        def nixl_agent(self, *args):
            agent = _CreatedAgent()
            self.agent_calls.append((args, agent))
            return agent

    fake_api = _FakeNixlApi()
    monkeypatch.setattr(nixl_module, "_require_module", lambda _name, _hint: fake_api)

    ucx_agent = nixl_module._create_nixl_agent(
        agent_name="agent0", backend_name="UCX", backend_init_params=None
    )
    assert fake_api.agent_calls[0][0] == ("agent0",)
    assert ucx_agent.backends == []

    custom_agent = nixl_module._create_nixl_agent(
        agent_name="agent1",
        backend_name="UCX",
        backend_init_params={"threads": 4, "enabled": True},
    )
    assert fake_api.config_calls == [[]]
    assert fake_api.agent_calls[1][0] == ("agent1", "config")
    assert custom_agent.backends == [("UCX", {"threads": "4", "enabled": "True"})]

    preinit_agent = nixl_module.preinit_nixl_agent(
        backend_name="CUSTOM", backend_init_params={"mode": "test"}
    )
    assert preinit_agent.metadata_calls == 1
    assert preinit_agent.backends == [("CUSTOM", {"mode": "test"})]


def test_nixl_agent_zmq_lifecycle_and_messages(monkeypatch):
    raw_agent = _RawNixlAgent()
    server_context = _FakeZmqContext()
    client_context = _FakeZmqContext()

    monkeypatch.setattr(nixl_module.uuid, "uuid4", lambda: "local-agent")
    monkeypatch.setattr(nixl_module, "_create_nixl_agent", lambda **_kwargs: raw_agent)
    monkeypatch.setattr(nixl_module.ray.util, "get_node_ip_address", lambda: "[::1]")
    monkeypatch.setattr(nixl_module, "_get_free_port", lambda _ip: 4567)
    monkeypatch.setattr(nixl_module.zmq.asyncio, "Context", lambda: server_context)
    monkeypatch.setattr(nixl_module.zmq, "Context", lambda: client_context)

    agent = nixl_module.NixlAgent("UCX")
    server_socket = server_context.sockets[0]
    assert agent.get_agent_metadata() == {
        "agent_name": "local-agent",
        "agent_metadata": b"local-metadata",
        "zmq_ip": "::1",
        "zmq_port": 4567,
    }
    assert server_socket.options == [(nixl_module.zmq.IPV6, 1)]
    assert server_socket.bound == ["tcp://[::1]:4567"]

    remote_metadata = {
        "agent_name": "remote0",
        "agent_metadata": b"remote-metadata",
        "zmq_ip": "::1",
        "zmq_port": 9999,
    }
    assert agent.add_remote_agent(remote_metadata) == "remote0"
    client_socket = client_context.sockets[0]
    assert raw_agent.added == [b"remote-metadata"]
    assert client_socket.options == [(nixl_module.zmq.IPV6, 1)]
    assert client_socket.connected == ["tcp://[::1]:9999"]

    agent.send_message("remote0", {"kind": "single"})
    agent.send_messages("remote0", [{"kind": "one"}])
    agent.send_messages("remote0", [{"kind": "a"}, {"kind": "b"}])
    assert client_socket.sent[0] == (
        ("local-agent", {"kind": "single"}),
        nixl_module.zmq.DONTWAIT,
    )
    assert client_socket.sent[1][0] == ("local-agent", {"kind": "one"})
    assert client_socket.sent[2][0][1] == {
        "message_type": nixl_module._MESSAGE_BUCKET_METADATA_BATCH,
        "messages": [{"kind": "a"}, {"kind": "b"}],
    }

    agent._enqueue_message(
        "remote0",
        {
            "message_type": nixl_module._MESSAGE_BUCKET_METADATA_BATCH,
            "messages": [{"idx": 0}, {"idx": 1}],
        },
    )
    assert asyncio.run(agent.try_read_message("remote0")) == {"idx": 0}
    assert asyncio.run(agent.try_read_message("remote0")) == {"idx": 1}

    server_socket.recv_queue.append(("remote0", {"idx": 2}))
    assert asyncio.run(agent.read_message("remote0")) == {"idx": 2}
    server_socket.recv_queue.append(("remote0", {"idx": 3}))
    assert asyncio.run(agent.try_read_message("remote0")) == {"idx": 3}
    assert asyncio.run(agent.try_read_message("remote0")) is None

    agent._enqueue_message(
        "remote0",
        {"message_type": nixl_module._MESSAGE_BUCKET_COMPLETE, "notify_key": b"done"},
    )
    agent.ignore_notification("remote0", b"ignored")
    raw_agent.native_notifications.append({"remote0": [b"ignored", b"ready"]})
    asyncio.run(agent.drain_notifications())
    assert asyncio.run(agent.get_notification("remote0")) == b"ready"
    asyncio.run(agent.get_completion("remote0", b"done"))

    buffer = torch.empty(1, dtype=torch.uint8)
    assert agent.register_memory(buffer) == "registered"
    assert agent.get_xfer_descs(buffer) == "descs"
    assert agent.initialize_xfer("READ", "local", "remote", "remote0", b"n") == "xfer"
    assert agent.transfer("xfer") == "DONE"
    assert agent.check_xfer_state("xfer") == "DONE"
    agent.release_xfer_handle("xfer")
    assert agent.progress()

    assert agent.refresh_remote_agent(remote_metadata) == "remote0"
    assert raw_agent.removed == ["remote0"]
    agent.remove_remote_agent("remote0")
    assert client_socket.closed == [0]
    agent.close()
    agent.close()
    assert server_socket.closed == [0]
    assert client_context.destroyed == [0]
    assert server_context.destroyed == [0]


def test_nixl_agent_rejects_remote_name_mismatch(monkeypatch):
    raw_agent = _RawNixlAgent(remote_agent_name=b"actual")
    monkeypatch.setattr(nixl_module, "_create_nixl_agent", lambda **_kwargs: raw_agent)
    monkeypatch.setattr(
        nixl_module.ray.util, "get_node_ip_address", lambda: "127.0.0.1"
    )
    monkeypatch.setattr(nixl_module, "_get_free_port", lambda _ip: 4567)
    monkeypatch.setattr(nixl_module.zmq.asyncio, "Context", _FakeZmqContext)
    monkeypatch.setattr(nixl_module.zmq, "Context", _FakeZmqContext)

    agent = nixl_module.NixlAgent("UCX")
    metadata = {
        "agent_name": "expected",
        "agent_metadata": b"remote-metadata",
        "zmq_ip": "127.0.0.1",
        "zmq_port": 9999,
    }
    with pytest.raises(RuntimeError, match="remote agent mismatch"):
        agent.add_remote_agent(metadata)
    with pytest.raises(RuntimeError, match="remote agent mismatch"):
        agent.refresh_remote_agent(metadata)
    agent.close()


def test_nixl_readable_group_publishes_to_each_remote():
    agent = _RecordingNixlAgent()
    metadata = nixl_module._bucket_metadata(
        {}, bucket_bytes=1, buffer_index=0, notify_key=b"n", is_last=True
    )

    group = nixl_module.ReadableOperationGroup(
        agent, ["rollout0", "rollout1"], "local-desc", metadata
    )

    assert len(group.operations) == 2
    assert [remote for remote, _message in agent.sent_messages] == [
        "rollout0",
        "rollout1",
    ]
    assert all(
        message["remote_descs"] == "local-desc" for _, message in agent.sent_messages
    )


def test_nixl_received_chunk_batch_release_is_idempotent():
    calls = []

    async def release(sync_devices):
        calls.append(sync_devices)

    batch = nixl_module._ReceivedChunkBatch([], 1, release)
    asyncio.run(batch.release(sync_devices=False))
    asyncio.run(batch.release(sync_devices=True))

    assert calls == [False]


def test_nixl_read_operation_error_and_background_paths():
    with pytest.raises(RuntimeError, match="entered ERR state"):
        nixl_module.ReadOperation(
            _StatefulTransferNixlAgent(transfer_state="ERR"),
            "policy0",
            "local",
            "remote",
            b"notify",
            1,
        ).begin_read()

    not_started = nixl_module.ReadOperation(
        _StatefulTransferNixlAgent(), "policy0", "local", "remote", b"notify", 1
    )
    with pytest.raises(RuntimeError, match="must be started"):
        asyncio.run(not_started.wait_for_complete())
    not_started._start_background_progress()
    assert not_started._background_done is None

    foreground_agent = _StatefulTransferNixlAgent(check_states=["BUSY", "DONE"])
    foreground = nixl_module.ReadOperation(
        foreground_agent, "policy0", "local", "remote", b"notify", 1
    )
    foreground.begin_read()
    asyncio.run(foreground.wait_for_complete())
    assert foreground_agent.progress_calls == 2

    foreground_error = nixl_module.ReadOperation(
        _StatefulTransferNixlAgent(check_states=["BUSY", "ERR"]),
        "policy0",
        "local",
        "remote",
        b"notify",
        1,
    )
    foreground_error.begin_read()
    with pytest.raises(RuntimeError, match="entered ERR state"):
        asyncio.run(foreground_error.wait_for_complete())

    background_agent = _StatefulTransferNixlAgent(check_states=["BUSY", "DONE"])
    background = nixl_module.ReadOperation(
        background_agent, "policy0", "local", "remote", b"notify", 1
    )
    background.begin_read(background_progress=True)
    background._start_background_progress()
    asyncio.run(background.wait_for_complete())
    assert background._background_done.is_set()

    background_error = nixl_module.ReadOperation(
        _StatefulTransferNixlAgent(check_states=["ERR"]),
        "policy0",
        "local",
        "remote",
        b"notify",
        1,
    )
    background_error.begin_read(background_progress=True)
    with pytest.raises(RuntimeError, match="entered ERR state"):
        asyncio.run(background_error.wait_for_complete())


def test_nixl_engine_buffer_allocation_paths(monkeypatch):
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.bucket_size = 4
    engine.device = torch.device("cpu")
    engine._cupy_buffers = []

    monkeypatch.setattr(nixl_module.torch.cuda, "is_available", lambda: False)
    cpu_buffer = engine._allocate_transfer_buffer()
    assert cpu_buffer.shape == (4,)
    assert cpu_buffer.device.type == "cpu"

    set_devices = []
    monkeypatch.setattr(
        nixl_module.torch.cuda, "set_device", lambda device: set_devices.append(device)
    )
    monkeypatch.setattr(nixl_module, "_optional_module", lambda _name: None)
    monkeypatch.setattr(
        nixl_module.torch,
        "zeros",
        lambda *args, **kwargs: ("torch-buffer", args, kwargs),
    )
    engine.device = torch.device("cuda", 0)
    torch_result = engine._allocate_transfer_buffer()
    assert set_devices == [torch.device("cuda", 0)]
    assert torch_result[0] == "torch-buffer"

    class _FakeCupyDevice:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_cupy = types.SimpleNamespace(
        uint8="cupy-uint8",
        cuda=types.SimpleNamespace(Device=_FakeCupyDevice),
        zeros=lambda size, dtype: ("cupy-buffer", size, dtype),
    )
    monkeypatch.setattr(nixl_module, "_optional_module", lambda _name: fake_cupy)
    monkeypatch.setattr(
        nixl_module.torch,
        "as_tensor",
        lambda cupy_buffer, *, dtype, device: (
            "torch-view",
            cupy_buffer,
            dtype,
            device,
        ),
    )
    cupy_result = engine._allocate_transfer_buffer()
    assert engine._cupy_buffers == [("cupy-buffer", 4, "cupy-uint8")]
    assert cupy_result[0] == "torch-view"


def test_nixl_engine_prepared_buffers_and_direct_caches():
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.bucket_size = 8
    engine.agent = _TransferNixlAgent()
    engine.send_buf = None
    engine.recv_buf = None
    engine.send_descs = None
    engine.recv_descs = None

    with pytest.raises(RuntimeError, match="buffers are not prepared"):
        engine._prepared_buffer_lists()

    engine.send_buf = torch.empty(8, dtype=torch.uint8)
    engine.recv_buf = torch.empty(8, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="descriptors are not prepared"):
        engine._prepared_buffer_lists()

    engine.send_descs = "send-desc"
    engine.recv_descs = "recv-desc"
    assert engine._prepared_buffer_lists() == (
        [engine.send_buf, engine.recv_buf],
        ["send-desc", "recv-desc"],
    )

    engine.transfer_bufs = [torch.empty(8, dtype=torch.uint8)]
    engine.transfer_descs = ["full-desc"]
    assert engine._buffer_xfer_desc(0, 8) == "full-desc"
    assert engine._buffer_xfer_desc(0, 4) == "xfer_0"
    assert engine._buffer_xfer_desc(0, 4) == "xfer_0"
    assert len(engine.agent.xfer_descs) == 1

    buffer = torch.arange(8, dtype=torch.uint8)
    assert engine._direct_xfer_desc(buffer, chunk_offset=2, chunk_size=3) == "xfer_1"
    assert engine._direct_xfer_desc(buffer, chunk_offset=2, chunk_size=3) == "xfer_1"
    assert len(engine.agent.registered) == 1
    assert len(engine.agent.xfer_descs) == 2


def test_nixl_engine_peer_refresh_disconnect_and_direct_source():
    class _RefreshAgent(_RecordingNixlAgent):
        def __init__(self):
            super().__init__()
            self.refreshed = []

        def refresh_remote_agent(self, metadata):
            self.refreshed.append(metadata)
            return metadata["agent_name"]

    agent = _RefreshAgent()
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.device = torch.device("cpu")
    engine.bucket_size = 8
    engine.direct_min_bytes = 16
    engine._read_xfer_handles = {("cache",): "handle"}
    engine._read_notify_keys = {("cache",): b"notify"}
    engine.prev_agent = "prev"
    engine.next_agent = "next"
    engine.next_agents = None

    with pytest.raises(RuntimeError, match="metadata update mismatch"):
        engine._refresh_prev_agent(_metadata("other"))

    engine._refresh_prev_agent(_metadata("prev"))
    assert agent.refreshed == [_metadata("prev")]
    assert agent.released_xfers == ["handle"]
    assert engine.prev_agent == "prev"

    engine._send_agent_metadata_update(["next0", "next1"])
    assert [remote for remote, _message in agent.sent_messages] == ["next0", "next1"]

    engine._read_xfer_handles = {("cache",): "handle2"}
    engine._read_notify_keys = {("cache",): b"notify2"}
    engine._disconnect_peers()
    assert agent.released_xfers[-1] == "handle2"
    assert agent.removed == ["prev", "next"]
    assert engine.prev_agent is None
    assert engine.next_agent is None
    assert engine.next_agents == []

    weight = torch.arange(4, dtype=torch.float32)
    engine.transfer_mode = "staged"
    assert engine._direct_source_buffer(weight) is None

    engine.transfer_mode = "direct"
    engine.device = torch.device("cuda", 0)
    assert engine._direct_source_buffer(weight) is None

    engine.device = torch.device("cpu")
    assert torch.equal(
        engine._direct_source_buffer(weight), weight.view(-1).view(torch.uint8)
    )

    engine.transfer_mode = "auto"
    engine.direct_min_bytes = weight.nbytes + 1
    assert engine._direct_source_buffer(weight) is None
    engine.direct_min_bytes = 1
    assert torch.equal(
        engine._direct_source_buffer(weight), weight.view(-1).view(torch.uint8)
    )


def test_nixl_rejects_two_gib_bucket_size():
    max_bucket_size = (1 << 31) - 1

    assert nixl_module._normalize_bucket_size(max_bucket_size) == max_bucket_size

    try:
        nixl_module._normalize_bucket_size(1 << 31)
    except ValueError as exc:
        assert "below 2 GiB" in str(exc)
    else:
        raise AssertionError("Expected NIXL bucket size validation to fail.")


def test_nixl_rejects_non_positive_bucket_size():
    for bucket_size in (0, -1):
        try:
            nixl_module._normalize_bucket_size(bucket_size)
        except ValueError as exc:
            assert "must be >= 1 byte" in str(exc)
        else:
            raise AssertionError("Expected NIXL bucket size validation to fail.")


def test_nixl_normalizes_public_config_options():
    assert nixl_module._normalize_topology(None) == "auto"
    assert nixl_module._normalize_topology("paired") == "paired"
    assert nixl_module._normalize_transfer_mode(None) == "staged"
    assert nixl_module._normalize_transfer_mode("direct") == "direct"
    assert nixl_module._normalize_transfer_mode("auto") == "auto"
    assert nixl_module._normalize_buffer_count(None) == 2
    assert nixl_module._normalize_buffer_count(4) == 4
    assert (
        nixl_module._normalize_non_negative_int(
            None,
            default=7,
            name="direct_min_bytes",
        )
        == 7
    )
    assert (
        nixl_module._normalize_non_negative_int(
            0,
            default=7,
            name="direct_min_bytes",
        )
        == 0
    )
    assert (
        nixl_module._normalize_positive_int(
            None,
            default=3,
            name="metadata_batch_size",
        )
        == 3
    )
    assert (
        nixl_module._normalize_positive_int(
            2,
            default=3,
            name="metadata_batch_size",
        )
        == 2
    )
    assert nixl_module._normalize_bool(None, default=True) is True
    assert nixl_module._normalize_bool(False, default=True) is False
    assert nixl_module._normalize_bool(True, default=False) is True
    assert nixl_module._tcp_address("127.0.0.1", 5555) == "tcp://127.0.0.1:5555"
    assert nixl_module._tcp_address("::1", 5555) == "tcp://[::1]:5555"


def test_nixl_rejects_invalid_public_config_options():
    _expect_value_error(
        lambda: nixl_module._normalize_topology("ring"),
        "Unsupported NIXL checkpoint-engine topology",
    )
    _expect_value_error(
        lambda: nixl_module._normalize_transfer_mode("remote"),
        "Unsupported NIXL checkpoint-engine transfer mode",
    )
    _expect_value_error(
        lambda: nixl_module._normalize_buffer_count(1),
        "buffer_count must be >= 2",
    )
    _expect_value_error(
        lambda: nixl_module._normalize_non_negative_int(
            -1,
            default=0,
            name="direct_min_bytes",
        ),
        "direct_min_bytes must be >= 0",
    )
    _expect_value_error(
        lambda: nixl_module._normalize_positive_int(
            0,
            default=1,
            name="metadata_batch_size",
        ),
        "metadata_batch_size must be >= 1",
    )


def test_nixl_init_applies_config_defaults_and_overrides():
    class _FakeAgent:
        def __init__(self, *, backend_name, backend_init_params):
            self.backend_name = backend_name
            self.backend_init_params = backend_init_params

    original_agent = nixl_module.NixlAgent
    try:
        nixl_module.NixlAgent = _FakeAgent
        engine = NIXLCheckpointEngine(
            bucket_size=8,
            device="cpu",
            backend_name="UCX",
            cleanup_after_load=False,
            backend_init_params={"device": "mlx5_0"},
            topology="paired",
            transfer_mode="auto",
            buffer_count=4,
            direct_min_bytes=0,
            background_progress=True,
            load_batch_bucket_count=3,
            direct_stripe_count=2,
            metadata_batch_size=5,
        )
    finally:
        nixl_module.NixlAgent = original_agent

    assert engine.bucket_size == 8
    assert engine.device == torch.device("cpu")
    assert engine.cleanup_after_load is False
    assert engine.topology == "paired"
    assert engine.transfer_mode == "auto"
    assert engine.buffer_count == 4
    assert engine.direct_min_bytes == 0
    assert engine.background_progress is True
    assert engine.load_batch_bucket_count == 3
    assert engine.direct_stripe_count == 2
    assert engine.metadata_batch_size == 5
    assert engine.agent.backend_name == "UCX"
    assert engine.agent.backend_init_params == {"device": "mlx5_0"}
    assert len(engine._notify_keys) == 4


def test_nixl_as_contiguous_uint8_marks_copied_views():
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    contiguous, copied = nixl_module._as_contiguous_uint8(matrix)
    assert copied is False
    assert contiguous.dtype == torch.uint8
    assert contiguous.numel() == matrix.nbytes

    transposed = matrix.t()
    contiguous_copy, copied = nixl_module._as_contiguous_uint8(transposed)
    assert copied is True
    assert contiguous_copy.dtype == torch.uint8
    assert contiguous_copy.numel() == transposed.nbytes
    assert contiguous_copy.is_contiguous()


def test_nixl_bucket_chunks_slices_by_metadata_offsets():
    buffer = torch.arange(8, dtype=torch.uint8)
    first = TensorMeta(
        name="first",
        shape=torch.Size([2]),
        dtype=torch.uint8,
        chunk_offset=0,
        chunk_size=2,
        offset=1,
    )
    second = TensorMeta(
        name="second",
        shape=torch.Size([3]),
        dtype=torch.uint8,
        chunk_offset=0,
        chunk_size=3,
        offset=4,
    )
    metadata = {
        "bucket_meta": {"first": first, "second": second},
        "bucket_bytes": 8,
        "buffer_index": 0,
        "notify_key": b"notify",
        "is_last": True,
    }

    chunks = nixl_module._bucket_chunks(metadata, buffer)

    assert [tensor_meta.name for tensor_meta, _chunk in chunks] == ["first", "second"]
    assert torch.equal(chunks[0][1], torch.tensor([1, 2], dtype=torch.uint8))
    assert torch.equal(chunks[1][1], torch.tensor([4, 5, 6], dtype=torch.uint8))


def test_nixl_bucket_chunks_requires_offsets():
    metadata = {
        "bucket_meta": {
            "weight": TensorMeta(
                name="weight",
                shape=torch.Size([1]),
                dtype=torch.uint8,
                chunk_offset=0,
                chunk_size=1,
                offset=None,
            )
        },
        "bucket_bytes": 1,
        "buffer_index": 0,
        "notify_key": b"notify",
        "is_last": True,
    }

    try:
        nixl_module._bucket_chunks(metadata, torch.zeros(1, dtype=torch.uint8))
    except RuntimeError as exc:
        assert "Missing NIXL offset for weight" in str(exc)
    else:
        raise AssertionError("Expected NIXL bucket chunk extraction to fail.")


def test_nixl_readable_operation_message_includes_remote_descs():
    metadata = {
        "bucket_meta": {},
        "bucket_bytes": 1,
        "buffer_index": 0,
        "notify_key": b"notify",
        "is_last": True,
    }

    message = nixl_module._readable_operation_message("local-desc", metadata)

    assert message["remote_descs"] == "local-desc"
    assert message["bucket_bytes"] == 1
    assert message["notify_key"] == b"notify"


def test_nixl_publish_readable_operation_groups_batches_per_remote():
    agent = _RecordingNixlAgent()
    first = {
        "bucket_meta": {},
        "bucket_bytes": 1,
        "buffer_index": 0,
        "notify_key": b"notify-0",
        "is_last": False,
    }
    second = {
        "bucket_meta": {},
        "bucket_bytes": 2,
        "buffer_index": 1,
        "notify_key": b"notify-1",
        "is_last": True,
    }

    groups = nixl_module._publish_readable_operation_groups(
        agent,
        ["rollout0", "rollout1"],
        [("desc0", first), ("desc1", second)],
    )

    assert len(groups) == 2
    assert agent.ignored_notifications == [
        ("rollout0", b"notify-0"),
        ("rollout1", b"notify-0"),
        ("rollout0", b"notify-1"),
        ("rollout1", b"notify-1"),
    ]
    assert [agent_name for agent_name, _payload in agent.sent_payloads] == [
        "rollout0",
        "rollout1",
    ]
    for _agent_name, payload in agent.sent_payloads:
        assert payload["message_type"] == nixl_module._MESSAGE_BUCKET_METADATA_BATCH
        assert [message["remote_descs"] for message in payload["messages"]] == [
            "desc0",
            "desc1",
        ]


def test_nixl_prepare_reuses_registered_buffers():
    agent = _DummyNixlAgent()
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.buffer_count = 2
    engine.transfer_bufs = None
    engine.transfer_reg_descs = None
    engine.transfer_descs = None
    engine.send_buf = None
    engine.recv_buf = None
    engine.send_reg_descs = None
    engine.recv_reg_descs = None
    engine.send_descs = None
    engine.recv_descs = None
    engine._allocate_transfer_buffer = lambda: object()

    assert engine.prepare()["agent_name"] == "dummy"
    first_state = (
        engine.send_buf,
        engine.recv_buf,
        engine.send_reg_descs,
        engine.recv_reg_descs,
        engine.send_descs,
        engine.recv_descs,
    )

    assert engine.prepare()["agent_name"] == "dummy"

    assert (
        engine.send_buf,
        engine.recv_buf,
        engine.send_reg_descs,
        engine.recv_reg_descs,
        engine.send_descs,
        engine.recv_descs,
    ) == first_state
    assert len(agent.registered) == 2
    assert len(agent.xfer_descs) == 2


def test_nixl_prepare_uses_configured_buffer_count():
    agent = _DummyNixlAgent()
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.buffer_count = 3
    engine.transfer_bufs = None
    engine.transfer_reg_descs = None
    engine.transfer_descs = None
    engine.send_buf = None
    engine.recv_buf = None
    engine.send_reg_descs = None
    engine.recv_reg_descs = None
    engine.send_descs = None
    engine.recv_descs = None
    engine._allocate_transfer_buffer = lambda: object()

    engine.prepare()

    assert len(engine.transfer_bufs) == 3
    assert engine.send_buf is engine.transfer_bufs[0]
    assert engine.recv_buf is engine.transfer_bufs[1]
    assert len(agent.registered) == 3
    assert len(agent.xfer_descs) == 3


def test_nixl_idle_rank_finalizes_without_closing_buffers():
    agent = _DummyNixlAgent()
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.rank = None
    engine.world_size = None
    engine.prev_agent = None
    engine.next_agent = None
    engine.send_buf = object()
    engine.recv_buf = object()
    engine.send_reg_descs = "send_desc"
    engine.recv_reg_descs = "recv_desc"
    engine.send_descs = object()
    engine.recv_descs = object()
    engine._cupy_buffers = []

    engine.init_process_group(
        rank=-1,
        world_size=9,
        prev_agent_metadata=None,
        next_agent_metadata=None,
    )
    engine.finalize()

    assert engine.send_buf is not None
    assert engine.recv_buf is not None
    assert engine.send_reg_descs == "send_desc"
    assert engine.recv_reg_descs == "recv_desc"
    assert not agent.closed
    engine.close()
    assert agent.closed


def _make_topology_engine(agent):
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.agent = agent
    engine.topology = "auto"
    engine.transfer_mode = "staged"
    engine.buffer_count = 2
    engine.direct_min_bytes = 1024 * 1024
    engine.direct_stripe_count = 1
    engine.metadata_batch_size = 1
    engine.rank = None
    engine.world_size = None
    engine.prev_agent = None
    engine.next_agent = None
    engine.next_agents = []
    engine._read_xfer_handles = {}
    engine._read_notify_keys = {}
    return engine


def test_nixl_uses_paired_topology_when_policy_can_cover_rollout():
    metadata = [_metadata(f"policy{i}") for i in range(4)] + [
        _metadata(f"rollout{i}") for i in range(4)
    ]

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.init_policy_process_group(
        worker_rank=2,
        train_world_size=4,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert policy_engine.rank == 0
    assert policy_engine.world_size == 2
    assert policy_engine.prev_agent is None
    assert policy_engine.next_agent == "rollout2"
    assert policy_agent.added == ["rollout2"]

    rollout_agent = _RecordingNixlAgent()
    rollout_engine = _make_topology_engine(rollout_agent)
    rollout_engine.init_rollout_process_group(
        rollout_rank=2,
        train_world_size=4,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert rollout_engine.rank == 1
    assert rollout_engine.world_size == 2
    assert rollout_engine.prev_agent == "policy2"
    assert rollout_engine.next_agent is None
    assert rollout_agent.added == ["policy2"]


def test_nixl_reuses_existing_peer_until_topology_changes():
    first = _metadata("rollout0")
    second = _metadata("rollout1")

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.init_process_group(
        rank=0,
        world_size=2,
        prev_agent_metadata=None,
        next_agent_metadata=first,
    )
    policy_engine.finalize()
    policy_engine.init_process_group(
        rank=0,
        world_size=2,
        prev_agent_metadata=None,
        next_agent_metadata=first,
    )

    assert policy_agent.added == ["rollout0"]
    assert policy_agent.removed == []

    policy_engine.init_process_group(
        rank=0,
        world_size=2,
        prev_agent_metadata=None,
        next_agent_metadata=second,
    )

    assert policy_agent.added == ["rollout0", "rollout1"]
    assert policy_agent.removed == ["rollout0"]


def test_nixl_reuses_read_xfer_handle_for_same_ping_pong_buffers():
    agent = _TransferNixlAgent()
    engine = _make_topology_engine(agent)
    engine.prev_agent = "policy0"
    engine.bucket_size = 1024

    notify_key = b"notify-0"
    metadata = {
        "remote_descs": "remote-desc-0",
        "notify_key": notify_key,
        "buffer_index": 0,
        "bucket_meta": {},
        "bucket_bytes": 1,
        "is_last": True,
    }

    _metadata_0, _read_op_0 = engine._read_operation(
        local_descs="local-desc-1",
        local_buffer_index=1,
        metadata=dict(metadata),
    )
    _metadata_1, _read_op_1 = engine._read_operation(
        local_descs="local-desc-1",
        local_buffer_index=1,
        metadata=dict(metadata),
    )

    assert len(agent.initialized_xfers) == 1
    assert agent.transferred == ["handle_0", "handle_0"]
    assert engine._read_xfer_handles[("policy0", 1, 0, 1, 0)] == "handle_0"


def test_nixl_read_wait_uses_native_progress_when_available():
    agent = _TransferNixlAgent()
    engine = _make_topology_engine(agent)
    engine.prev_agent = "policy0"
    engine.bucket_size = 1024

    metadata = {
        "remote_descs": "remote-desc-0",
        "notify_key": b"notify-0",
        "buffer_index": 0,
        "bucket_meta": {},
        "bucket_bytes": 1,
        "is_last": True,
    }

    _metadata, read_op = engine._read_operation(
        local_descs="local-desc-1",
        local_buffer_index=1,
        metadata=dict(metadata),
    )
    asyncio.run(read_op.wait_for_complete())

    assert agent.progress_calls >= 1


def test_nixl_readable_operation_uses_message_completion_for_staged_bucket():
    agent = _RecordingNixlAgent()
    metadata = {
        "remote_descs": "remote-desc-0",
        "notify_key": b"notify-0",
        "buffer_index": 0,
        "bucket_meta": {},
        "bucket_bytes": 1,
        "is_last": True,
    }

    readable_op = nixl_module.ReadableOperation(
        agent,
        "rollout0",
        "local-desc-0",
        metadata,
    )
    asyncio.run(readable_op.wait_for_complete())

    assert agent.ignored_notifications == [("rollout0", b"notify-0")]
    assert agent.completions == [("rollout0", b"notify-0")]


def test_nixl_completion_wait_drains_ignored_native_notifications():
    class _NativeAgent:
        def __init__(self):
            self.notifications = [{"policy0": [b"done", b"next"]}, {}]

        def progress(self):
            return True

        def get_new_notifs(self):
            return self.notifications.pop(0) if self.notifications else {}

    agent = nixl_module.NixlAgent.__new__(nixl_module.NixlAgent)
    agent.agent = _NativeAgent()
    agent._agent_lock = Lock()
    agent.notifications = defaultdict(deque)
    agent.completions = defaultdict(set)
    agent.ignored_notifications = defaultdict(set)
    agent.ignored_notifications["policy0"].add(b"done")

    async def drain_zmq_messages():
        agent.completions["policy0"].add(b"done")

    agent._drain_zmq_messages = drain_zmq_messages

    asyncio.run(agent.get_completion("policy0", b"done"))

    assert b"done" not in agent.ignored_notifications["policy0"]
    assert list(agent.notifications["policy0"]) == [b"next"]


def test_nixl_completion_wait_drains_when_completion_is_prequeued():
    class _NativeAgent:
        def __init__(self):
            self.notifications = [{"policy0": [b"done"]}, {}]

        def progress(self):
            return True

        def get_new_notifs(self):
            return self.notifications.pop(0) if self.notifications else {}

    agent = nixl_module.NixlAgent.__new__(nixl_module.NixlAgent)
    agent.agent = _NativeAgent()
    agent._agent_lock = Lock()
    agent.notifications = defaultdict(deque)
    agent.completions = defaultdict(set)
    agent.ignored_notifications = defaultdict(set)
    agent.completions["policy0"].add(b"done")
    agent.ignored_notifications["policy0"].add(b"done")

    async def drain_zmq_messages():
        return None

    agent._drain_zmq_messages = drain_zmq_messages

    asyncio.run(agent.get_completion("policy0", b"done"))

    assert b"done" not in agent.ignored_notifications["policy0"]
    assert list(agent.notifications["policy0"]) == []


def test_nixl_read_xfer_handle_cache_distinguishes_direct_sources():
    agent = _TransferNixlAgent()
    engine = _make_topology_engine(agent)
    engine.prev_agent = "policy0"
    engine.bucket_size = 1024

    first = {
        "remote_descs": "remote-desc-0",
        "remote_desc_key": ("ptr0", 0, 16),
        "notify_key": b"notify-0",
        "buffer_index": -1,
        "bucket_meta": {},
        "bucket_bytes": 16,
        "is_last": False,
    }
    second = {
        "remote_descs": "remote-desc-1",
        "remote_desc_key": ("ptr1", 0, 16),
        "notify_key": b"notify-1",
        "buffer_index": -1,
        "bucket_meta": {},
        "bucket_bytes": 16,
        "is_last": False,
    }

    engine._read_operation(
        local_descs="local-desc-1",
        local_buffer_index=1,
        metadata=dict(first),
    )
    engine._read_operation(
        local_descs="local-desc-1",
        local_buffer_index=1,
        metadata=dict(second),
    )

    assert len(agent.initialized_xfers) == 2
    assert agent.initialized_xfers[0][2] == "remote-desc-0"
    assert agent.initialized_xfers[1][2] == "remote-desc-1"


def test_nixl_receive_prefetches_with_extra_buffers():
    def bucket_metadata(index, *, is_last):
        name = f"weight{index}"
        return {
            "remote_descs": f"remote-desc-{index}",
            "notify_key": f"notify-{index}".encode(),
            "buffer_index": index,
            "bucket_meta": {
                name: TensorMeta(
                    name=name,
                    shape=torch.Size([4]),
                    dtype=torch.uint8,
                    chunk_offset=0,
                    chunk_size=4,
                    offset=0,
                )
            },
            "bucket_bytes": 4,
            "is_last": is_last,
        }

    agent = _QueuedTransferNixlAgent(
        [
            bucket_metadata(0, is_last=False),
            bucket_metadata(1, is_last=False),
            bucket_metadata(2, is_last=True),
        ]
    )
    engine = _make_topology_engine(agent)
    engine.prev_agent = "policy0"
    engine.bucket_size = 4
    engine.device = torch.device("cpu")
    engine.buffer_count = 3
    engine.transfer_bufs = [
        torch.zeros(4, dtype=torch.uint8),
        torch.zeros(4, dtype=torch.uint8),
        torch.zeros(4, dtype=torch.uint8),
    ]
    engine.transfer_descs = ["desc0", "desc1", "desc2"]

    async def receive():
        batches = []
        generator = engine._receive_weight_chunk_batches()
        first_batch = await generator.__anext__()
        initialized_before_first_yield = list(agent.initialized_xfers)
        completions_before_first_release = [
            payload
            for _agent_name, payload in agent.sent_payloads
            if payload.get("message_type") == nixl_module._MESSAGE_BUCKET_COMPLETE
        ]
        batches.append(first_batch)
        async for batch in generator:
            batches.append(batch)
        return initialized_before_first_yield, completions_before_first_release, batches

    initialized_before_first_yield, completions_before_first_release, batches = (
        asyncio.run(receive())
    )

    assert [entry[1] for entry in initialized_before_first_yield] == [
        "desc1",
        "desc2",
        "desc0",
    ]
    assert completions_before_first_release == [
        {
            "message_type": nixl_module._MESSAGE_BUCKET_COMPLETE,
            "notify_key": b"notify-0",
        }
    ]
    assert [[chunk_meta.name for chunk_meta, _chunk in batch] for batch in batches] == [
        ["weight0"],
        ["weight1"],
        ["weight2"],
    ]


def test_nixl_receive_weight_batches_coalesces_ready_buckets():
    def bucket_metadata(index, *, is_last):
        name = f"weight{index}"
        return {
            "remote_descs": f"remote-desc-{index}",
            "notify_key": f"notify-{index}".encode(),
            "buffer_index": index,
            "bucket_meta": {
                name: TensorMeta(
                    name=name,
                    shape=torch.Size([4]),
                    dtype=torch.uint8,
                    chunk_offset=0,
                    chunk_size=4,
                    offset=0,
                )
            },
            "bucket_bytes": 4,
            "is_last": is_last,
        }

    agent = _QueuedTransferNixlAgent(
        [
            bucket_metadata(0, is_last=False),
            bucket_metadata(1, is_last=False),
            bucket_metadata(2, is_last=True),
        ]
    )
    engine = _make_topology_engine(agent)
    engine.prev_agent = "policy0"
    engine.bucket_size = 4
    engine.device = torch.device("cpu")
    engine.buffer_count = 3
    engine.background_progress = False
    engine.load_batch_bucket_count = 2
    engine.transfer_bufs = [
        torch.full((4,), 1, dtype=torch.uint8),
        torch.full((4,), 2, dtype=torch.uint8),
        torch.full((4,), 3, dtype=torch.uint8),
    ]
    engine.transfer_descs = ["desc0", "desc1", "desc2"]

    async def receive():
        batches = []
        async for batch in engine.receive_weight_batches():
            batches.append([(name, weight.clone()) for name, weight in batch])
        return batches

    batches = asyncio.run(receive())

    assert [[name for name, _weight in batch] for batch in batches] == [
        ["weight0", "weight1"],
        ["weight2"],
    ]


def test_nixl_recorded_load_completion_skips_release_device_sync():
    metadata = {
        "remote_descs": "remote-desc",
        "notify_key": b"notify",
        "buffer_index": 0,
        "bucket_meta": {
            "weight": TensorMeta(
                name="weight",
                shape=torch.Size([4]),
                dtype=torch.uint8,
                chunk_offset=0,
                chunk_size=4,
                offset=0,
            )
        },
        "bucket_bytes": 4,
        "is_last": True,
    }
    agent = _QueuedTransferNixlAgent([metadata])
    engine = _make_topology_engine(agent)
    engine.prev_agent = "policy0"
    engine.bucket_size = 4
    engine.device = torch.device("cpu")
    engine.buffer_count = 2
    engine.background_progress = False
    engine.load_batch_bucket_count = 1
    engine.transfer_bufs = [
        torch.full((4,), 1, dtype=torch.uint8),
        torch.full((4,), 2, dtype=torch.uint8),
    ]
    engine.transfer_descs = ["desc0", "desc1"]

    sync_calls = 0
    original_sync_devices = nixl_module._sync_devices

    def count_sync_calls(devices):
        nonlocal sync_calls
        sync_calls += 1

    async def receive():
        batches = []
        async for batch in engine.receive_weight_batches():
            batch.record_cuda_load_complete()
            batches.append([(name, weight.clone()) for name, weight in batch])
        return batches

    try:
        nixl_module._sync_devices = count_sync_calls
        batches = asyncio.run(receive())
    finally:
        nixl_module._sync_devices = original_sync_devices

    assert [[name for name, _weight in batch] for batch in batches] == [["weight"]]
    assert sync_calls == 0


def test_nixl_agent_unwraps_batched_bucket_metadata():
    agent = nixl_module.NixlAgent.__new__(nixl_module.NixlAgent)
    agent.messages = defaultdict(deque)
    first = {"bucket_bytes": 1}
    second = {"bucket_bytes": 2}

    agent._enqueue_message(
        "policy0",
        {
            "message_type": nixl_module._MESSAGE_BUCKET_METADATA_BATCH,
            "messages": [first, second],
        },
    )

    assert agent.messages["policy0"].popleft() is first
    assert agent.messages["policy0"].popleft() is second


def test_nixl_direct_xfer_desc_reuses_registered_source_buffer():
    agent = _DummyNixlAgent()
    engine = _make_topology_engine(agent)
    engine._direct_reg_cache = {}
    engine._direct_desc_cache = {}
    engine._direct_buffer_refs = {}
    buffer = torch.arange(16, dtype=torch.uint8)

    first = engine._direct_xfer_desc(buffer, chunk_offset=0, chunk_size=8)
    second = engine._direct_xfer_desc(buffer, chunk_offset=0, chunk_size=8)
    third = engine._direct_xfer_desc(buffer, chunk_offset=8, chunk_size=8)

    assert first == second
    assert third != first
    assert len(agent.registered) == 1
    assert len(agent.xfer_descs) == 2


def test_nixl_direct_send_stripes_and_batches_metadata():
    agent = _RecordingNixlAgent()
    engine = _make_topology_engine(agent)
    engine.rank = 0
    engine.prev_agent = None
    engine.next_agents = ["rollout0"]
    engine.next_agent = "rollout0"
    engine.transfer_mode = "direct"
    engine.direct_stripe_count = 2
    engine.metadata_batch_size = 2
    engine.bucket_size = 8
    engine.device = torch.device("cpu")
    engine.buffer_count = 3
    engine.transfer_bufs = [
        torch.zeros(8, dtype=torch.uint8),
        torch.zeros(8, dtype=torch.uint8),
        torch.zeros(8, dtype=torch.uint8),
    ]
    engine.transfer_descs = ["desc0", "desc1", "desc2"]
    engine._direct_reg_cache = {}
    engine._direct_desc_cache = {}
    engine._direct_buffer_refs = {}

    def weights():
        yield "weight", torch.arange(16, dtype=torch.uint8)

    asyncio.run(engine.send_weights(weights()))

    bucket_messages = [
        message
        for _agent_name, message in agent.sent_messages
        if "bucket_meta" in message
    ]
    batch_payloads = [
        payload
        for _agent_name, payload in agent.sent_payloads
        if payload.get("message_type") == nixl_module._MESSAGE_BUCKET_METADATA_BATCH
    ]

    assert [message["bucket_bytes"] for message in bucket_messages] == [
        4,
        4,
        4,
        4,
    ]
    assert [
        next(iter(message["bucket_meta"].values())).chunk_offset
        for message in bucket_messages
    ] == [0, 4, 8, 12]
    assert [len(payload["messages"]) for payload in batch_payloads] == [2, 2]
    assert [message["is_last"] for message in bucket_messages] == [
        False,
        False,
        False,
        True,
    ]


def test_nixl_receive_reconstructs_sub_bucket_stripes():
    def bucket_metadata(index, *, chunk_offset, is_last):
        return {
            "remote_descs": f"remote-desc-{index}",
            "notify_key": f"notify-{index}".encode(),
            "buffer_index": index,
            "bucket_meta": {
                "weight": TensorMeta(
                    name="weight",
                    shape=torch.Size([8]),
                    dtype=torch.uint8,
                    chunk_offset=chunk_offset,
                    chunk_size=4,
                    offset=0,
                )
            },
            "bucket_bytes": 4,
            "is_last": is_last,
        }

    agent = _QueuedTransferNixlAgent(
        [
            {
                "message_type": nixl_module._MESSAGE_BUCKET_METADATA_BATCH,
                "messages": [
                    bucket_metadata(0, chunk_offset=0, is_last=False),
                    bucket_metadata(1, chunk_offset=4, is_last=True),
                ],
            }
        ]
    )
    engine = _make_topology_engine(agent)
    engine.prev_agent = "policy0"
    engine.bucket_size = 8
    engine.device = torch.device("cpu")
    engine.buffer_count = 3
    engine.transfer_bufs = [
        torch.full((8,), 0, dtype=torch.uint8),
        torch.tensor([1, 2, 3, 4, 0, 0, 0, 0], dtype=torch.uint8),
        torch.tensor([5, 6, 7, 8, 0, 0, 0, 0], dtype=torch.uint8),
    ]
    engine.transfer_descs = ["desc0", "desc1", "desc2"]

    async def receive():
        batches = []
        async for batch in engine.receive_weight_batches():
            batches.append([(name, weight.clone()) for name, weight in batch])
        return batches

    batches = asyncio.run(receive())

    assert [[name for name, _weight in batch] for batch in batches] == [["weight"]]
    assert torch.equal(batches[0][0][1], torch.arange(1, 9, dtype=torch.uint8))


def test_nixl_staged_send_preserves_order_for_packed_small_tensors():
    agent = _RecordingNixlAgent()
    engine = _make_topology_engine(agent)
    engine.rank = 0
    engine.prev_agent = None
    engine.next_agents = ["rollout0"]
    engine.next_agent = "rollout0"
    engine.bucket_size = 4
    engine.device = torch.device("cpu")
    engine.buffer_count = 2
    engine.transfer_bufs = [
        torch.zeros(4, dtype=torch.uint8),
        torch.zeros(4, dtype=torch.uint8),
    ]
    engine.transfer_descs = ["desc0", "desc1"]

    def weights():
        yield "weight0", torch.full((4,), 1, dtype=torch.uint8)
        yield "weight1", torch.full((4,), 2, dtype=torch.uint8)

    asyncio.run(engine.send_weights(weights()))

    sent_names = []
    for _agent_name, message in agent.sent_messages:
        tensor_meta = next(iter(message["bucket_meta"].values()))
        if tensor_meta.children:
            sent_names.extend(child.name for child in tensor_meta.children)
        else:
            sent_names.append(tensor_meta.name)

    assert sent_names == ["weight0", "weight1"]


def test_nixl_auto_direct_default_keeps_medium_tensors_staged():
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.transfer_mode = "auto"
    engine.device = torch.device("cpu")
    engine.bucket_size = 128 * 1024 * 1024

    medium = torch.empty(1024 * 1024, dtype=torch.uint8)
    large = torch.empty(64 * 1024 * 1024, dtype=torch.uint8)
    bucket_sized = torch.empty(engine.bucket_size, dtype=torch.uint8)

    assert engine._direct_source_buffer(medium) is None
    assert engine._direct_source_buffer(large) is None
    assert engine._direct_source_buffer(bucket_sized) is not None


def test_nixl_policy_ranks_without_rollout_pair_are_idle():
    metadata = [_metadata(f"policy{i}") for i in range(6)] + [
        _metadata(f"rollout{i}") for i in range(4)
    ]

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.init_policy_process_group(
        worker_rank=5,
        train_world_size=6,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert policy_engine.rank == -1
    assert policy_engine.world_size == 2
    assert policy_engine.prev_agent is None
    assert policy_engine.next_agent is None
    assert policy_agent.added == []


def test_nixl_leader_chain_topology_uses_single_policy_sender():
    metadata = [_metadata(f"policy{i}") for i in range(4)] + [
        _metadata(f"rollout{i}") for i in range(4)
    ]

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.topology = "leader_chain"
    policy_engine.init_policy_process_group(
        worker_rank=0,
        train_world_size=4,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert policy_engine.rank == 0
    assert policy_engine.world_size == 5
    assert policy_engine.prev_agent is None
    assert policy_engine.next_agent == "rollout0"
    assert policy_agent.added == ["rollout0"]

    idle_agent = _RecordingNixlAgent()
    idle_engine = _make_topology_engine(idle_agent)
    idle_engine.topology = "leader_chain"
    idle_engine.init_policy_process_group(
        worker_rank=3,
        train_world_size=4,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert idle_engine.rank == -1
    assert idle_engine.world_size == 5
    assert idle_engine.prev_agent is None
    assert idle_engine.next_agent is None
    assert idle_agent.added == []

    rollout_agent = _RecordingNixlAgent()
    rollout_engine = _make_topology_engine(rollout_agent)
    rollout_engine.topology = "leader_chain"
    rollout_engine.init_rollout_process_group(
        rollout_rank=2,
        train_world_size=4,
        rollout_world_size=4,
        metadata=metadata,
    )

    assert rollout_engine.rank == 3
    assert rollout_engine.world_size == 5
    assert rollout_engine.prev_agent == "rollout1"
    assert rollout_engine.next_agent == "rollout3"
    assert rollout_agent.added == ["rollout1", "rollout3"]


def test_nixl_auto_uses_tree_when_rollout_exceeds_policy_workers():
    metadata = [_metadata("policy0")] + [_metadata(f"rollout{i}") for i in range(5)]

    policy_agent = _RecordingNixlAgent()
    policy_engine = _make_topology_engine(policy_agent)
    policy_engine.init_policy_process_group(
        worker_rank=0,
        train_world_size=1,
        rollout_world_size=5,
        metadata=metadata,
    )

    assert policy_engine.rank == 0
    assert policy_engine.world_size == 6
    assert policy_engine.next_agent == "rollout0"
    assert policy_engine.next_agents == ["rollout0", "rollout1"]
    assert policy_agent.added == ["rollout0", "rollout1"]

    middle_agent = _RecordingNixlAgent()
    middle_engine = _make_topology_engine(middle_agent)
    middle_engine.init_rollout_process_group(
        rollout_rank=0,
        train_world_size=1,
        rollout_world_size=5,
        metadata=metadata,
    )

    assert middle_engine.rank == 1
    assert middle_engine.world_size == 6
    assert middle_engine.prev_agent == "policy0"
    assert middle_engine.next_agent == "rollout2"
    assert middle_engine.next_agents == ["rollout2", "rollout3"]
    assert middle_agent.added == ["policy0", "rollout2", "rollout3"]

    leaf_agent = _RecordingNixlAgent()
    leaf_engine = _make_topology_engine(leaf_agent)
    leaf_engine.init_rollout_process_group(
        rollout_rank=4,
        train_world_size=1,
        rollout_world_size=5,
        metadata=metadata,
    )

    assert leaf_engine.rank == 5
    assert leaf_engine.world_size == 6
    assert leaf_engine.prev_agent == "rollout1"
    assert leaf_engine.next_agent is None
    assert leaf_engine.next_agents == []
    assert leaf_agent.added == ["rollout1"]


def test_nixl_idle_sender_does_not_consume_weight_generator():
    engine = NIXLCheckpointEngine.__new__(NIXLCheckpointEngine)
    engine.rank = -1
    engine.prev_agent = None
    engine.next_agent = None
    consumed = False

    def weights():
        nonlocal consumed
        consumed = True
        yield "weight", torch.ones(1)

    asyncio.run(engine.send_weights(weights()))

    assert not consumed
