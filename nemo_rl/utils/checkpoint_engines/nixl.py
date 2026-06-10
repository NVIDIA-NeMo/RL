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

import asyncio
import importlib
import ipaddress
import logging
import os
import re
import socket
import sys
import threading
import time
import types
import uuid
from collections import defaultdict, deque
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    NamedTuple,
    NotRequired,
    TypedDict,
    cast,
)

import ray
import torch
import zmq
import zmq.asyncio

from nemo_rl.utils.checkpoint_engines.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NIXLCheckpointEngine",
    "NixlAgentMetadata",
    "preinit_nixl_agent",
]


class NixlAgentMetadata(TypedDict):
    """Serializable NIXL agent metadata exchanged through Ray."""

    agent_name: str
    agent_metadata: bytes
    zmq_ip: str
    zmq_port: int


class NixlBucketMetadata(TypedDict):
    """Metadata for one NIXL transfer bucket."""

    bucket_meta: dict[str, TensorMeta]
    bucket_bytes: int
    buffer_index: int
    notify_key: bytes
    is_last: bool
    remote_desc_key: NotRequired[Any]


class _PendingStagedBucket(NamedTuple):
    buffer_index: int
    local_descs: Any
    metadata: NixlBucketMetadata
    copy_events: tuple[torch.cuda.Event, ...]


_SMALL_TENSOR_PACK_THRESHOLD_BYTES = 1024 * 1024
_SMALL_TENSOR_PACK_BYTES = 16 * 1024 * 1024
_TOPOLOGY_AUTO = "auto"
_TOPOLOGY_PAIRED = "paired"
_TOPOLOGY_LEADER_CHAIN = "leader_chain"
_TOPOLOGY_LEADER_TREE = "leader_tree"
_SUPPORTED_TOPOLOGIES = {
    _TOPOLOGY_AUTO,
    _TOPOLOGY_PAIRED,
    _TOPOLOGY_LEADER_CHAIN,
    _TOPOLOGY_LEADER_TREE,
}
_TRANSFER_MODE_STAGED = "staged"
_TRANSFER_MODE_DIRECT = "direct"
_TRANSFER_MODE_AUTO = "auto"
_SUPPORTED_TRANSFER_MODES = {
    _TRANSFER_MODE_STAGED,
    _TRANSFER_MODE_DIRECT,
    _TRANSFER_MODE_AUTO,
}
_MESSAGE_AGENT_METADATA_UPDATE = "agent_metadata_update"
_MESSAGE_BUCKET_METADATA_BATCH = "bucket_metadata_batch"
_MESSAGE_BUCKET_COMPLETE = "bucket_complete"
_DEFAULT_AUTO_DIRECT_MIN_BYTES = 64 * 1024 * 1024
_MAX_NIXL_BUCKET_BYTES = (1 << 31) - 1


def _normalize_topology(topology: str | None) -> str:
    if topology is None:
        return _TOPOLOGY_AUTO
    if topology not in _SUPPORTED_TOPOLOGIES:
        raise ValueError(
            f"Unsupported NIXL checkpoint-engine topology {topology!r}. "
            f"Supported values: {sorted(_SUPPORTED_TOPOLOGIES)}"
        )
    return topology


def _normalize_transfer_mode(transfer_mode: str | None) -> str:
    if transfer_mode is None:
        return _TRANSFER_MODE_STAGED
    if transfer_mode not in _SUPPORTED_TRANSFER_MODES:
        raise ValueError(
            f"Unsupported NIXL checkpoint-engine transfer mode {transfer_mode!r}. "
            f"Supported values: {sorted(_SUPPORTED_TRANSFER_MODES)}"
        )
    return transfer_mode


def _normalize_buffer_count(buffer_count: int | None) -> int:
    if buffer_count is None:
        return 2
    if buffer_count < 2:
        raise ValueError("NIXL checkpoint-engine buffer_count must be >= 2.")
    return buffer_count


def _normalize_bucket_size(bucket_size: int) -> int:
    if bucket_size < 1:
        raise ValueError("NIXL checkpoint-engine bucket_size must be >= 1 byte.")
    if bucket_size > _MAX_NIXL_BUCKET_BYTES:
        raise ValueError(
            "NIXL checkpoint-engine bucket_size must be below 2 GiB "
            f"({_MAX_NIXL_BUCKET_BYTES} bytes max). Set "
            "policy.generation.checkpoint_engine.update_weights_bucket_megabytes "
            "to 1024 or another value below 2048."
        )
    return bucket_size


def _normalize_non_negative_int(value: int | None, *, default: int, name: str) -> int:
    if value is None:
        return default
    if value < 0:
        raise ValueError(f"NIXL checkpoint-engine {name} must be >= 0.")
    return value


def _normalize_positive_int(value: int | None, *, default: int, name: str) -> int:
    if value is None:
        return default
    if value < 1:
        raise ValueError(f"NIXL checkpoint-engine {name} must be >= 1.")
    return value


def _normalize_bool(value: bool | None, *, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _get_free_port(address: str) -> int:
    family = socket.AF_INET6 if _is_valid_ipv6_address(address) else socket.AF_INET
    with socket.socket(family=family, type=socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, 0))
        return int(sock.getsockname()[1])


def _tcp_address(ip_address: str, port: int) -> str:
    if _is_valid_ipv6_address(ip_address):
        return f"tcp://[{ip_address}]:{port}"
    return f"tcp://{ip_address}:{port}"


def _ensure_torch_version_module() -> None:
    if not hasattr(torch, "Tensor"):
        torch.Tensor = object
    try:
        importlib.import_module("torch.version")
        return
    except ModuleNotFoundError:
        pass

    torch_version = getattr(torch, "version", None)
    version_module = types.ModuleType("torch.version")
    for name in ("__version__", "cuda", "debug", "git_version", "hip"):
        if torch_version is not None and hasattr(torch_version, name):
            setattr(version_module, name, getattr(torch_version, name))
    if not hasattr(version_module, "cuda"):
        torch_version_text = getattr(torch, "__version__", "")
        cuda_match = re.search(r"\+cu(\d+)", torch_version_text)
        if cuda_match is not None:
            cuda_digits = cuda_match.group(1)
            if len(cuda_digits) >= 2:
                version_module.cuda = (
                    f"{cuda_digits[:-1]}.{cuda_digits[-1]}"
                    if len(cuda_digits) > 2
                    else cuda_digits
                )
    if not hasattr(version_module, "cuda"):
        cuda_version = os.environ.get("CUDA_VERSION")
        if cuda_version:
            version_module.cuda = ".".join(cuda_version.split(".")[:2])
    sys.modules["torch.version"] = version_module
    setattr(torch, "version", version_module)


def _ensure_numpy_annotation_attrs() -> None:
    try:
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError:
        return
    if not hasattr(numpy, "ndarray"):
        numpy.ndarray = object


def _require_module(module_name: str, install_hint: str) -> Any:
    if module_name.startswith("nixl"):
        _ensure_torch_version_module()
        _ensure_numpy_annotation_attrs()
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name!r} is required for NIXL checkpoint-engine refit. "
            f"{install_hint}"
        ) from exc


def _optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _normalize_device(device: str | torch.device) -> torch.device:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch_device


def _sync_devices(devices: Iterable[torch.device]) -> None:
    synced_cuda_indices: set[int] = set()
    for device in devices:
        if device.type != "cuda":
            continue
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if index in synced_cuda_indices:
            continue
        torch.cuda.synchronize(torch.device("cuda", index))
        synced_cuda_indices.add(index)


def _cuda_fence_devices(
    devices: Iterable[torch.device], *, include_current_device: bool = False
) -> tuple[torch.device, ...]:
    cuda_devices: dict[int, torch.device] = {}
    for device in devices:
        if device.type != "cuda":
            continue
        if device.index is None:
            if not torch.cuda.is_available():
                continue
            index = torch.cuda.current_device()
        else:
            index = device.index
        cuda_devices[index] = torch.device("cuda", index)

    if include_current_device and torch.cuda.is_available():
        current_index = torch.cuda.current_device()
        cuda_devices.setdefault(current_index, torch.device("cuda", current_index))

    return tuple(cuda_devices.values())


def _record_cuda_events(
    devices: Iterable[torch.device],
) -> tuple[torch.cuda.Event, ...]:
    events: list[torch.cuda.Event] = []
    for device in devices:
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        events.append(event)
    return tuple(events)


async def _wait_cuda_events(events: Iterable[torch.cuda.Event]) -> None:
    pending_events = tuple(events)
    while pending_events:
        pending_events = tuple(event for event in pending_events if not event.query())
        if pending_events:
            await asyncio.sleep(0)


def _as_contiguous_uint8(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.is_contiguous():
        return weight.view(-1).view(torch.uint8), False
    return weight.contiguous().view(-1).view(torch.uint8), True


def _create_nixl_agent(
    *,
    agent_name: str,
    backend_name: str,
    backend_init_params: dict[str, Any] | None = None,
) -> Any:
    nixl_api = _require_module(
        "nixl._api",
        "Install NIXL in the runtime environment or disable "
        "policy.generation.checkpoint_engine.enabled.",
    )
    if backend_name == "UCX" and backend_init_params is None:
        return nixl_api.nixl_agent(agent_name)

    nixl_config = nixl_api.nixl_agent_config(backends=[])
    agent = nixl_api.nixl_agent(agent_name, nixl_config)
    init_params = {
        key: str(value) for key, value in (backend_init_params or {}).items()
    }
    agent.create_backend(backend_name, init_params)
    return agent


def preinit_nixl_agent(
    *,
    backend_name: str,
    backend_init_params: dict[str, Any] | None = None,
) -> Any:
    """Create a lightweight NIXL agent to initialize backend plugins early."""
    agent = _create_nixl_agent(
        agent_name=f"preinit-{uuid.uuid4()}",
        backend_name=backend_name,
        backend_init_params=backend_init_params,
    )
    agent.get_agent_metadata()
    return agent


def _bucket_metadata(
    bucket_meta: dict[str, TensorMeta],
    *,
    bucket_bytes: int,
    buffer_index: int,
    notify_key: bytes,
    is_last: bool,
) -> NixlBucketMetadata:
    return {
        "bucket_meta": bucket_meta,
        "bucket_bytes": bucket_bytes,
        "buffer_index": buffer_index,
        "notify_key": notify_key,
        "is_last": is_last,
    }


def _bucket_chunks(
    metadata: NixlBucketMetadata,
    buffer: torch.Tensor,
) -> list[tuple[TensorMeta, torch.Tensor]]:
    chunks = []
    for tensor_meta in metadata["bucket_meta"].values():
        if tensor_meta.offset is None:
            raise RuntimeError(f"Missing NIXL offset for {tensor_meta.name}.")
        tensor = buffer[
            tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size
        ]
        chunks.append((tensor_meta, tensor))
    return chunks


def _readable_operation_message(
    local_descs: Any,
    metadata: NixlBucketMetadata,
) -> dict[str, Any]:
    return {
        "remote_descs": local_descs,
        **metadata,
    }


class NixlAgent:
    """NIXL agent wrapper using ZMQ for bucket metadata notifications."""

    def __init__(
        self,
        backend_name: str,
        backend_init_params: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = str(uuid.uuid4())
        self.agent = _create_nixl_agent(
            agent_name=self.agent_name,
            backend_name=backend_name,
            backend_init_params=backend_init_params,
        )
        self.notifications: dict[str, deque[bytes]] = defaultdict(deque)
        self.completions: dict[str, set[bytes]] = defaultdict(set)
        self.ignored_notifications: dict[str, set[bytes]] = defaultdict(set)
        self.messages: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self.zmq_clients: dict[str, zmq.Socket] = {}
        self.zmq_client_context = zmq.Context()
        self._agent_lock = threading.Lock()
        self._closed = False
        self._start_zmq_server()

    def _progress_locked(self) -> bool:
        progress = getattr(self.agent, "progress", None)
        if not callable(progress):
            return False
        progress()
        return True

    def _start_zmq_server(self) -> None:
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port = _get_free_port(self.ip)

        self.zmq_context = zmq.asyncio.Context()
        self.socket = self.zmq_context.socket(zmq.PULL)
        if _is_valid_ipv6_address(self.ip):
            self.socket.setsockopt(zmq.IPV6, 1)
        self.socket.bind(_tcp_address(self.ip, self.listen_port))

    def get_agent_metadata(self) -> NixlAgentMetadata:
        with self._agent_lock:
            agent_metadata = self.agent.get_agent_metadata()
        return {
            "agent_name": self.agent_name,
            "agent_metadata": agent_metadata,
            "zmq_ip": self.ip,
            "zmq_port": self.listen_port,
        }

    def add_remote_agent(self, metadata: NixlAgentMetadata) -> str:
        with self._agent_lock:
            remote_agent_name = self.agent.add_remote_agent(
                metadata["agent_metadata"]
            ).decode("utf-8")
        if remote_agent_name != metadata["agent_name"]:
            raise RuntimeError(
                f"NIXL remote agent mismatch: expected {metadata['agent_name']}, "
                f"got {remote_agent_name}"
            )

        client_socket = self.zmq_client_context.socket(zmq.PUSH)
        if _is_valid_ipv6_address(metadata["zmq_ip"]):
            client_socket.setsockopt(zmq.IPV6, 1)
        client_socket.connect(_tcp_address(metadata["zmq_ip"], metadata["zmq_port"]))
        self.zmq_clients[remote_agent_name] = client_socket
        return remote_agent_name

    def refresh_remote_agent(self, metadata: NixlAgentMetadata) -> str:
        with self._agent_lock:
            self.agent.remove_remote_agent(metadata["agent_name"])
            remote_agent_name = self.agent.add_remote_agent(
                metadata["agent_metadata"]
            ).decode("utf-8")
        if remote_agent_name != metadata["agent_name"]:
            raise RuntimeError(
                f"NIXL remote agent mismatch: expected {metadata['agent_name']}, "
                f"got {remote_agent_name}"
            )
        return remote_agent_name

    def remove_remote_agent(self, agent_name: str) -> None:
        with self._agent_lock:
            self.agent.remove_remote_agent(agent_name)
        client_socket = self.zmq_clients.pop(agent_name)
        client_socket.close(linger=0)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        for client_socket in self.zmq_clients.values():
            client_socket.close(linger=0)
        self.zmq_clients.clear()
        self.socket.close(linger=0)
        self.zmq_client_context.destroy(linger=0)
        self.zmq_context.destroy(linger=0)

    def send_message(self, agent_name: str, message: dict[str, Any]) -> None:
        self.zmq_clients[agent_name].send_pyobj(
            (self.agent_name, message), zmq.DONTWAIT
        )

    def send_messages(self, agent_name: str, messages: list[dict[str, Any]]) -> None:
        if len(messages) == 1:
            self.send_message(agent_name, messages[0])
            return
        self.send_message(
            agent_name,
            {
                "message_type": _MESSAGE_BUCKET_METADATA_BATCH,
                "messages": messages,
            },
        )

    def _enqueue_message(self, agent_name: str, message: dict[str, Any]) -> None:
        if message.get("message_type") == _MESSAGE_BUCKET_METADATA_BATCH:
            self.messages[agent_name].extend(message["messages"])
            return
        if message.get("message_type") == _MESSAGE_BUCKET_COMPLETE:
            self.completions[agent_name].add(message["notify_key"])
            return
        self.messages[agent_name].append(message)

    async def _drain_zmq_messages(self) -> None:
        while await self.socket.poll(0):
            recv_agent_name, message = await self.socket.recv_pyobj()
            self._enqueue_message(recv_agent_name, message)

    async def read_message(self, agent_name: str) -> dict[str, Any]:
        while len(self.messages[agent_name]) == 0:
            recv_agent_name, message = await self.socket.recv_pyobj()
            self._enqueue_message(recv_agent_name, message)
            await asyncio.sleep(0)
        return self.messages[agent_name].popleft()

    async def try_read_message(self, agent_name: str) -> dict[str, Any] | None:
        if len(self.messages[agent_name]) > 0:
            return self.messages[agent_name].popleft()

        while await self.socket.poll(0):
            recv_agent_name, message = await self.socket.recv_pyobj()
            self._enqueue_message(recv_agent_name, message)
            if len(self.messages[agent_name]) > 0:
                return self.messages[agent_name].popleft()
        return None

    def ignore_notification(self, remote_name: str, notify_key: bytes) -> None:
        self.ignored_notifications[remote_name].add(notify_key)

    def _enqueue_native_notifications(
        self, notifications: dict[str, list[bytes]]
    ) -> None:
        for agent_name, agent_notifications in notifications.items():
            ignored = self.ignored_notifications[agent_name]
            for notification in agent_notifications:
                if notification in ignored:
                    ignored.remove(notification)
                else:
                    self.notifications[agent_name].append(notification)

    async def drain_notifications(self) -> None:
        await self._drain_zmq_messages()
        while True:
            with self._agent_lock:
                self._progress_locked()
                notifications = self.agent.get_new_notifs()
            if not any(notifications.values()):
                return
            self._enqueue_native_notifications(notifications)
            await asyncio.sleep(0)

    async def get_notification(self, remote_name: str) -> bytes:
        while len(self.notifications[remote_name]) == 0:
            await self.drain_notifications()
            await asyncio.sleep(0)
        return self.notifications[remote_name].popleft()

    async def get_completion(self, remote_name: str, notify_key: bytes) -> None:
        await self.drain_notifications()
        while notify_key not in self.completions[remote_name]:
            await self.drain_notifications()
            await asyncio.sleep(0)
        self.completions[remote_name].remove(notify_key)
        await self.drain_notifications()

    def register_memory(self, buffer: torch.Tensor) -> Any:
        with self._agent_lock:
            return self.agent.register_memory(buffer)

    def get_xfer_descs(self, buffer: torch.Tensor) -> Any:
        with self._agent_lock:
            return self.agent.get_xfer_descs(buffer)

    def initialize_xfer(
        self,
        operation: str,
        local_descs: Any,
        remote_descs: Any,
        remote_agent: str,
        notify_key: bytes,
    ) -> Any:
        with self._agent_lock:
            return self.agent.initialize_xfer(
                operation,
                local_descs,
                remote_descs,
                remote_agent,
                notify_key,
            )

    def transfer(self, xfer_handle: Any) -> str:
        with self._agent_lock:
            return self.agent.transfer(xfer_handle)

    def progress(self) -> bool:
        with self._agent_lock:
            return self._progress_locked()

    def check_xfer_state(self, xfer_handle: Any) -> str:
        with self._agent_lock:
            return self.agent.check_xfer_state(xfer_handle)

    def release_xfer_handle(self, xfer_handle: Any) -> None:
        with self._agent_lock:
            self.agent.release_xfer_handle(xfer_handle)


class ReadableOperation:
    """Remote-readable bucket exposed through NIXL."""

    def __init__(
        self,
        agent: NixlAgent,
        remote_agent: str,
        local_descs: Any,
        metadata: NixlBucketMetadata,
        *,
        publish: bool = True,
    ) -> None:
        self.agent = agent
        self.remote_agent = remote_agent
        self.notify_key = metadata["notify_key"]
        self.agent.ignore_notification(self.remote_agent, self.notify_key)
        if publish:
            self.agent.send_message(
                self.remote_agent,
                _readable_operation_message(local_descs, metadata),
            )

    async def wait_for_complete(self) -> None:
        await self.agent.get_completion(self.remote_agent, self.notify_key)


class ReadableOperationGroup:
    """One readable bucket exposed to one or more downstream NIXL peers."""

    def __init__(
        self,
        agent: NixlAgent,
        remote_agents: Iterable[str],
        local_descs: Any,
        metadata: NixlBucketMetadata,
        *,
        publish: bool = True,
    ) -> None:
        remote_agents = tuple(remote_agents)
        self.operations = [
            ReadableOperation(
                agent,
                remote_agent,
                local_descs,
                metadata,
                publish=False,
            )
            for remote_agent in remote_agents
        ]
        if publish:
            message = _readable_operation_message(local_descs, metadata)
            for remote_agent in remote_agents:
                agent.send_message(remote_agent, message)

    async def wait_for_complete(self) -> None:
        for operation in self.operations:
            await operation.wait_for_complete()


def _publish_readable_operation_groups(
    agent: NixlAgent,
    remote_agents: Iterable[str],
    buckets: list[tuple[Any, NixlBucketMetadata]],
) -> list[ReadableOperationGroup]:
    remote_agents = tuple(remote_agents)
    groups = [
        ReadableOperationGroup(
            agent,
            remote_agents,
            local_descs,
            metadata,
            publish=False,
        )
        for local_descs, metadata in buckets
    ]
    for remote_agent in remote_agents:
        agent.send_messages(
            remote_agent,
            [
                _readable_operation_message(local_descs, metadata)
                for local_descs, metadata in buckets
            ],
        )
    return groups


class ReadOperation:
    """NIXL read operation from a remote readable bucket."""

    def __init__(
        self,
        agent: NixlAgent,
        remote_agent: str,
        local_descs: Any,
        remote_descs: Any,
        notify_key: bytes,
        bucket_size: int,
        xfer_handle: Any | None = None,
    ) -> None:
        self.agent = agent
        self.remote_agent = remote_agent
        self.local_descs = local_descs
        self.remote_descs = remote_descs
        self.xfer_handle = xfer_handle
        self.notify_key = notify_key
        self.bucket_size = _normalize_bucket_size(bucket_size)
        self.start_time: float | None = None
        self._background_done: threading.Event | None = None
        self._background_thread: threading.Thread | None = None
        self._background_error: RuntimeError | None = None

    def begin_read(self, *, background_progress: bool = False) -> None:
        if self.xfer_handle is None:
            self.xfer_handle = self.agent.initialize_xfer(
                "READ",
                self.local_descs,
                self.remote_descs,
                self.remote_agent,
                self.notify_key,
            )
        state = self.agent.transfer(self.xfer_handle)
        if state == "ERR":
            raise RuntimeError(f"NIXL read from {self.remote_agent} entered ERR state.")
        self.start_time = time.time()
        if background_progress:
            self._start_background_progress()

    def _start_background_progress(self) -> None:
        if self.xfer_handle is None or self._background_thread is not None:
            return

        self._background_done = threading.Event()
        self._background_thread = threading.Thread(
            target=self._background_wait_for_complete,
            name=f"nixl-read-progress-{self.remote_agent}",
            daemon=True,
        )
        self._background_thread.start()

    def _background_wait_for_complete(self) -> None:
        if self.xfer_handle is None or self._background_done is None:
            return
        try:
            while True:
                self.agent.progress()
                state = self.agent.check_xfer_state(self.xfer_handle)
                if state == "ERR":
                    self._background_error = RuntimeError(
                        f"NIXL read from {self.remote_agent} entered ERR state."
                    )
                    return
                if state == "DONE":
                    return
                time.sleep(0)
        finally:
            self._background_done.set()

    async def wait_for_complete(self) -> None:
        if self.xfer_handle is None or self.start_time is None:
            raise RuntimeError("NIXL read must be started before waiting.")
        if self._background_done is None:
            while True:
                self.agent.progress()
                state = self.agent.check_xfer_state(self.xfer_handle)
                if state == "ERR":
                    raise RuntimeError(
                        f"NIXL read from {self.remote_agent} entered ERR state."
                    )
                if state == "DONE":
                    break
                await asyncio.sleep(0)
        else:
            while not self._background_done.is_set():
                await asyncio.sleep(0)
            if self._background_thread is not None:
                self._background_thread.join(timeout=0)
            if self._background_error is not None:
                raise self._background_error
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            bandwidth = self.bucket_size / elapsed / (1024 * 1024 * 1024)
            logger.debug(
                "NIXL read from %s completed at %.2f GB/s",
                self.remote_agent,
                bandwidth,
            )


class _ReceivedChunkBatch:
    def __init__(
        self,
        chunks: list[tuple[TensorMeta, torch.Tensor]],
        bucket_bytes: int,
        release: Callable[[bool], Awaitable[None]],
        release_blocks_next: bool = False,
    ) -> None:
        self.chunks = chunks
        self.bucket_bytes = bucket_bytes
        self._release = release
        self._released = False
        self.release_blocks_next = release_blocks_next

    async def release(self, *, sync_devices: bool = True) -> None:
        if self._released:
            return
        self._released = True
        await self._release(sync_devices)


class _LoadCompleteWeightBatch(list[tuple[str, Any]]):
    """Weight batch that keeps transfer-buffer leases alive until CUDA load ends."""

    def __init__(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        super().__init__(weights)
        cuda_devices: dict[int, torch.device] = {}
        for _name, weight in self:
            if weight.device.type != "cuda":
                continue
            index = (
                weight.device.index
                if weight.device.index is not None
                else torch.cuda.current_device()
            )
            cuda_devices[index] = torch.device("cuda", index)
        self._cuda_devices = tuple(cuda_devices.values())
        self._cuda_events: tuple[torch.cuda.Event, ...] | None = None
        self._load_complete_waited = False

    def _load_fence_devices(self) -> tuple[torch.device, ...]:
        return _cuda_fence_devices(
            self._cuda_devices,
            include_current_device=True,
        )

    def record_cuda_load_complete(self) -> None:
        if self._cuda_events is not None:
            return
        self._cuda_events = _record_cuda_events(self._load_fence_devices())

    def wait_for_cuda_load_complete(self) -> None:
        if self._load_complete_waited:
            return
        if self._cuda_events is None:
            _sync_devices(self._load_fence_devices())
        else:
            for event in self._cuda_events:
                event.synchronize()
        self._load_complete_waited = True

    async def wait_for_cuda_load_complete_async(self) -> None:
        if self._load_complete_waited:
            return
        if self._cuda_events is None:
            _sync_devices(self._load_fence_devices())
        else:
            await _wait_cuda_events(self._cuda_events)
        self._load_complete_waited = True


@CheckpointEngineRegistry.register("nixl")
class NIXLCheckpointEngine(CheckpointEngine):
    """NIXL checkpoint engine for non-colocated policy-to-generation refit."""

    def __init__(
        self,
        bucket_size: int,
        device: str | torch.device,
        backend_name: str,
        cleanup_after_load: bool,
        backend_init_params: dict[str, Any] | None = None,
        topology: str | None = None,
        transfer_mode: str | None = None,
        buffer_count: int | None = None,
        direct_min_bytes: int | None = None,
        background_progress: bool | None = None,
        load_batch_bucket_count: int | None = None,
        direct_stripe_count: int | None = None,
        metadata_batch_size: int | None = None,
    ) -> None:
        self.bucket_size = _normalize_bucket_size(bucket_size)
        self.device = _normalize_device(device)
        self.cleanup_after_load = cleanup_after_load
        self.topology = _normalize_topology(topology)
        self.transfer_mode = _normalize_transfer_mode(transfer_mode)
        self.buffer_count = _normalize_buffer_count(buffer_count)
        self.direct_min_bytes = _normalize_non_negative_int(
            direct_min_bytes,
            default=_DEFAULT_AUTO_DIRECT_MIN_BYTES,
            name="direct_min_bytes",
        )
        self.background_progress = _normalize_bool(
            background_progress,
            default=False,
        )
        self.load_batch_bucket_count = _normalize_positive_int(
            load_batch_bucket_count,
            default=1,
            name="load_batch_bucket_count",
        )
        self.direct_stripe_count = _normalize_positive_int(
            direct_stripe_count,
            default=1,
            name="direct_stripe_count",
        )
        self.metadata_batch_size = _normalize_positive_int(
            metadata_batch_size,
            default=1,
            name="metadata_batch_size",
        )
        self.agent = NixlAgent(
            backend_name=backend_name,
            backend_init_params=backend_init_params,
        )
        self.rank: int | None = None
        self.world_size: int | None = None
        self.prev_agent: str | None = None
        self.next_agent: str | None = None
        self.next_agents: list[str] = []
        self.transfer_bufs: list[torch.Tensor] | None = None
        self.transfer_reg_descs: list[Any] | None = None
        self.transfer_descs: list[Any] | None = None
        self.send_buf: torch.Tensor | None = None
        self.recv_buf: torch.Tensor | None = None
        self.send_reg_descs: Any | None = None
        self.recv_reg_descs: Any | None = None
        self.send_descs: Any | None = None
        self.recv_descs: Any | None = None
        self._cupy_buffers: list[Any] = []
        self._notify_keys = {
            buffer_index: uuid.uuid4().bytes
            for buffer_index in range(self.buffer_count)
        }
        self._buffer_slice_descs: dict[tuple[int, int], Any] = {}
        self._direct_reg_cache: dict[tuple[int, int], Any] = {}
        self._direct_desc_cache: dict[tuple[int, int, int], Any] = {}
        self._direct_buffer_refs: dict[tuple[int, int], torch.Tensor] = {}
        self._read_xfer_handles: dict[tuple[str, int, int, int, Any], Any] = {}
        self._read_notify_keys: dict[tuple[str, int, int, int, Any], bytes] = {}

    def _allocate_transfer_buffer(self) -> torch.Tensor:
        if self.device.type != "cuda":
            return torch.zeros(
                self.bucket_size,
                dtype=torch.uint8,
                device=self.device,
                pin_memory=torch.cuda.is_available(),
            )

        torch.cuda.set_device(self.device)
        cupy = _optional_module("cupy")
        if cupy is None:
            logger.warning(
                "CuPy is not installed; using torch CUDA buffers for NIXL memory "
                "registration. If registration fails with expandable CUDA segments, "
                "install CuPy or disable expandable segments."
            )
            return torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device)

        with cupy.cuda.Device(self.device.index):
            cupy_buffer = cupy.zeros(self.bucket_size, dtype=cupy.uint8)
        self._cupy_buffers.append(cupy_buffer)
        return torch.as_tensor(cupy_buffer, dtype=torch.uint8, device=self.device)

    def prepare(self) -> NixlAgentMetadata:
        buffer_count = getattr(self, "buffer_count", 2)
        transfer_bufs = getattr(self, "transfer_bufs", None)
        transfer_reg_descs = getattr(self, "transfer_reg_descs", None)
        transfer_descs = getattr(self, "transfer_descs", None)
        if (
            transfer_bufs is not None
            and transfer_reg_descs is not None
            and transfer_descs is not None
            and len(transfer_bufs) == buffer_count
            and len(transfer_reg_descs) == buffer_count
            and len(transfer_descs) == buffer_count
        ):
            return self.agent.get_agent_metadata()

        self.transfer_bufs = [
            self._allocate_transfer_buffer() for _ in range(buffer_count)
        ]
        self.transfer_reg_descs = [
            self.agent.register_memory(buffer) for buffer in self.transfer_bufs
        ]
        self.transfer_descs = [
            self.agent.get_xfer_descs(buffer) for buffer in self.transfer_bufs
        ]
        self.send_buf = self.transfer_bufs[0]
        self.recv_buf = self.transfer_bufs[1]
        self.send_reg_descs = self.transfer_reg_descs[0]
        self.recv_reg_descs = self.transfer_reg_descs[1]
        self.send_descs = self.transfer_descs[0]
        self.recv_descs = self.transfer_descs[1]
        self._buffer_slice_descs = {}
        return self.agent.get_agent_metadata()

    def _prepared_buffer_lists(self) -> tuple[list[torch.Tensor], list[Any]]:
        transfer_bufs = getattr(self, "transfer_bufs", None)
        transfer_descs = getattr(self, "transfer_descs", None)
        if transfer_bufs is not None and transfer_descs is not None:
            return transfer_bufs, transfer_descs

        if self.send_buf is None or self.recv_buf is None:
            raise RuntimeError("NIXL transfer buffers are not prepared.")
        if self.send_descs is None or self.recv_descs is None:
            raise RuntimeError("NIXL transfer descriptors are not prepared.")
        return [self.send_buf, self.recv_buf], [self.send_descs, self.recv_descs]

    def _buffer_xfer_desc(self, buffer_index: int, nbytes: int) -> Any:
        buffers, descs = self._prepared_buffer_lists()
        if nbytes == self.bucket_size:
            return descs[buffer_index]

        cache_key = (buffer_index, nbytes)
        buffer_slice_descs = getattr(self, "_buffer_slice_descs", {})
        if cache_key not in buffer_slice_descs:
            buffer_slice_descs[cache_key] = self.agent.get_xfer_descs(
                buffers[buffer_index][:nbytes]
            )
            self._buffer_slice_descs = buffer_slice_descs
        return buffer_slice_descs[cache_key]

    def _notify_key(self, buffer_index: int) -> bytes:
        if not hasattr(self, "_notify_keys"):
            self._notify_keys = {}
        if buffer_index not in self._notify_keys:
            self._notify_keys[buffer_index] = uuid.uuid4().bytes
        return self._notify_keys[buffer_index]

    def _direct_xfer_desc(
        self, buffer: torch.Tensor, *, chunk_offset: int, chunk_size: int
    ) -> Any:
        if not hasattr(self, "_direct_reg_cache"):
            self._direct_reg_cache = {}
            self._direct_desc_cache = {}
            self._direct_buffer_refs = {}

        reg_key = (buffer.data_ptr(), buffer.nbytes)
        if reg_key not in self._direct_reg_cache:
            self._direct_reg_cache[reg_key] = self.agent.register_memory(buffer)
            self._direct_buffer_refs[reg_key] = buffer

        desc_key = (buffer.data_ptr(), chunk_offset, chunk_size)
        if desc_key not in self._direct_desc_cache:
            self._direct_desc_cache[desc_key] = self.agent.get_xfer_descs(
                buffer[chunk_offset : chunk_offset + chunk_size]
            )
        return self._direct_desc_cache[desc_key]

    def _release_read_xfer_handles(self) -> None:
        for xfer_handle in getattr(self, "_read_xfer_handles", {}).values():
            self.agent.release_xfer_handle(xfer_handle)
        self._read_xfer_handles = {}
        self._read_notify_keys = {}

    def _disconnect_peers(self) -> None:
        self._release_read_xfer_handles()
        if self.prev_agent is not None:
            self.agent.remove_remote_agent(self.prev_agent)
        next_agents = getattr(self, "next_agents", None)
        if next_agents is None:
            next_agents = [self.next_agent] if self.next_agent is not None else []
        for next_agent in next_agents:
            self.agent.remove_remote_agent(next_agent)
        self.prev_agent = None
        self.next_agent = None
        self.next_agents = []

    def _refresh_prev_agent(self, metadata: NixlAgentMetadata) -> None:
        if self.prev_agent != metadata["agent_name"]:
            raise RuntimeError(
                f"NIXL metadata update mismatch: expected {self.prev_agent}, "
                f"got {metadata['agent_name']}"
            )
        self._release_read_xfer_handles()
        self.prev_agent = self.agent.refresh_remote_agent(metadata)

    def _send_agent_metadata_update(self, next_agents: list[str]) -> None:
        message = {
            "message_type": _MESSAGE_AGENT_METADATA_UPDATE,
            "agent_metadata": self.agent.get_agent_metadata(),
        }
        for next_agent in next_agents:
            self.agent.send_message(next_agent, message)

    def _direct_source_buffer(self, weight: torch.Tensor) -> torch.Tensor | None:
        transfer_mode = _normalize_transfer_mode(
            getattr(self, "transfer_mode", _TRANSFER_MODE_STAGED)
        )
        if transfer_mode == _TRANSFER_MODE_STAGED or not weight.is_contiguous():
            return None
        buffer = weight.view(-1).view(torch.uint8)
        if buffer.device != self.device:
            return None
        if transfer_mode == _TRANSFER_MODE_DIRECT:
            return buffer
        if weight.nbytes >= self._effective_auto_direct_min_bytes():
            return buffer
        return None

    def _effective_auto_direct_min_bytes(self) -> int:
        return max(
            getattr(self, "direct_min_bytes", _DEFAULT_AUTO_DIRECT_MIN_BYTES),
            self.bucket_size,
        )

    def _direct_chunk_size(self) -> int:
        stripe_count = getattr(self, "direct_stripe_count", 1)
        return max(1, (self.bucket_size + stripe_count - 1) // stripe_count)

    def _read_operation(
        self,
        *,
        local_descs: Any,
        local_buffer_index: int,
        metadata: dict[str, Any],
        background_progress: bool | None = None,
    ) -> tuple[NixlBucketMetadata, ReadOperation]:
        if self.prev_agent is None:
            raise RuntimeError("NIXL receiver rank has no previous peer.")

        remote_descs = metadata.pop("remote_descs")
        notify_key = metadata["notify_key"]
        remote_buffer_index = metadata["buffer_index"]
        remote_desc_key = metadata.get("remote_desc_key", remote_buffer_index)
        cache_key = (
            self.prev_agent,
            local_buffer_index,
            remote_buffer_index,
            metadata["bucket_bytes"],
            remote_desc_key,
        )

        xfer_handle = self._read_xfer_handles.get(cache_key)
        if xfer_handle is not None and self._read_notify_keys[cache_key] != notify_key:
            self.agent.release_xfer_handle(xfer_handle)
            xfer_handle = None
            self._read_xfer_handles.pop(cache_key)
            self._read_notify_keys.pop(cache_key)

        read_op = ReadOperation(
            self.agent,
            self.prev_agent,
            local_descs,
            remote_descs,
            notify_key,
            metadata["bucket_bytes"],
            xfer_handle=xfer_handle,
        )
        if background_progress is None:
            background_progress = getattr(self, "background_progress", False)
        read_op.begin_read(background_progress=background_progress)
        if xfer_handle is None:
            self._read_xfer_handles[cache_key] = read_op.xfer_handle
            self._read_notify_keys[cache_key] = notify_key
        return metadata, read_op

    def _forward_metadata(
        self, metadata: NixlBucketMetadata, *, buffer_index: int
    ) -> NixlBucketMetadata:
        return _bucket_metadata(
            metadata["bucket_meta"],
            bucket_bytes=metadata["bucket_bytes"],
            buffer_index=buffer_index,
            notify_key=self._notify_key(buffer_index),
            is_last=metadata["is_last"],
        )

    def _effective_topology(
        self, *, train_world_size: int, rollout_world_size: int
    ) -> str:
        topology = _normalize_topology(getattr(self, "topology", _TOPOLOGY_AUTO))
        if topology == _TOPOLOGY_AUTO:
            if train_world_size >= rollout_world_size:
                return _TOPOLOGY_PAIRED
            return _TOPOLOGY_LEADER_TREE
        if topology == _TOPOLOGY_PAIRED and train_world_size < rollout_world_size:
            raise ValueError(
                "NIXL paired topology requires train_world_size >= rollout_world_size."
            )
        return topology

    def _leader_tree_children(self, rollout_rank: int, rollout_world_size: int):
        first_child = 2 * rollout_rank + 2
        second_child = first_child + 1
        for child_rank in (first_child, second_child):
            if child_rank < rollout_world_size:
                yield child_rank

    def init_policy_process_group(
        self,
        *,
        worker_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[NixlAgentMetadata],
    ) -> None:
        topology = self._effective_topology(
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
        )
        if topology == _TOPOLOGY_PAIRED:
            world_size = 2
            rank = 0 if worker_rank < rollout_world_size else -1
            next_agent_metadata = (
                [metadata[train_world_size + worker_rank]] if rank == 0 else []
            )
        elif topology == _TOPOLOGY_LEADER_CHAIN:
            world_size = rollout_world_size + 1
            rank = 0 if worker_rank == 0 else -1
            next_agent_metadata = [metadata[train_world_size]] if rank == 0 else []
        else:
            world_size = rollout_world_size + 1
            rank = 0 if worker_rank == 0 else -1
            next_agent_metadata = (
                [
                    metadata[train_world_size + child_rank]
                    for child_rank in range(min(2, rollout_world_size))
                ]
                if rank == 0
                else []
            )
        self.init_process_group(
            rank=rank,
            world_size=world_size,
            prev_agent_metadata=None,
            next_agent_metadata=next_agent_metadata,
        )

    def init_rollout_process_group(
        self,
        *,
        rollout_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[NixlAgentMetadata],
    ) -> None:
        if rollout_rank < 0 or rollout_rank >= rollout_world_size:
            raise ValueError(
                f"rollout_rank must be in [0, {rollout_world_size}), got {rollout_rank}"
            )

        topology = self._effective_topology(
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
        )
        if topology == _TOPOLOGY_PAIRED:
            world_size = 2
            rank = 1
            prev_agent_metadata = metadata[rollout_rank]
            next_agent_metadata = []
        elif topology == _TOPOLOGY_LEADER_CHAIN:
            world_size = rollout_world_size + 1
            rank = rollout_rank + 1
            prev_agent_metadata = (
                metadata[0]
                if rollout_rank == 0
                else metadata[train_world_size + rollout_rank - 1]
            )
            next_agent_metadata = (
                [metadata[train_world_size + rollout_rank + 1]]
                if rollout_rank < rollout_world_size - 1
                else []
            )
        else:
            world_size = rollout_world_size + 1
            rank = rollout_rank + 1
            parent_rollout_rank = (rollout_rank - 2) // 2
            prev_agent_metadata = (
                metadata[0]
                if rollout_rank < 2
                else metadata[train_world_size + parent_rollout_rank]
            )
            next_agent_metadata = [
                metadata[train_world_size + child_rank]
                for child_rank in self._leader_tree_children(
                    rollout_rank, rollout_world_size
                )
            ]
        self.init_process_group(
            rank=rank,
            world_size=world_size,
            prev_agent_metadata=prev_agent_metadata,
            next_agent_metadata=next_agent_metadata,
        )

    def init_process_group(
        self,
        *,
        rank: int,
        world_size: int,
        prev_agent_metadata: NixlAgentMetadata | None,
        next_agent_metadata: NixlAgentMetadata | list[NixlAgentMetadata] | None,
    ) -> None:
        if next_agent_metadata is None:
            next_agent_metadata_list = []
        elif isinstance(next_agent_metadata, list):
            next_agent_metadata_list = next_agent_metadata
        else:
            next_agent_metadata_list = [next_agent_metadata]

        if rank < 0:
            if prev_agent_metadata is not None or next_agent_metadata_list:
                raise ValueError(f"Idle NIXL rank {rank} should not have peers.")
        elif rank == 0:
            if prev_agent_metadata is not None or not next_agent_metadata_list:
                raise ValueError("NIXL source rank must have downstream peers only.")
        elif rank < world_size - 1:
            if prev_agent_metadata is None:
                raise ValueError("NIXL middle ranks must have a previous peer.")
        elif prev_agent_metadata is None:
            raise ValueError("NIXL final rank must have a previous peer.")

        next_agent_names = [
            next_metadata["agent_name"] for next_metadata in next_agent_metadata_list
        ]
        prev_agent_name = (
            prev_agent_metadata["agent_name"]
            if prev_agent_metadata is not None
            else None
        )
        if (
            self.prev_agent != prev_agent_name
            or getattr(self, "next_agents", []) != next_agent_names
        ):
            self._disconnect_peers()

        self.rank = rank
        self.world_size = world_size

        if prev_agent_metadata is not None and self.prev_agent is None:
            self.prev_agent = self.agent.add_remote_agent(prev_agent_metadata)
        if not getattr(self, "next_agents", []):
            self.next_agents = [
                self.agent.add_remote_agent(next_metadata)
                for next_metadata in next_agent_metadata_list
            ]
            self.next_agent = self.next_agents[0] if self.next_agents else None

    def finalize(self) -> None:
        """Keep long-lived peers and transfer handles available for the next refit."""

    def close(self) -> None:
        """Close the long-lived NIXL agent when the engine is discarded."""
        self._disconnect_peers()
        self.agent.close()

    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> None:
        if self.rank is None:
            raise RuntimeError(
                "NIXL checkpoint engine process group is not initialized."
            )
        if self.prev_agent is not None:
            raise RuntimeError("Only NIXL source ranks may send weights.")
        if self.rank < 0:
            return
        next_agents = getattr(self, "next_agents", [])
        if not next_agents and self.next_agent is not None:
            next_agents = [self.next_agent]
        if not next_agents:
            raise RuntimeError("NIXL source rank has no next peer.")

        await self.agent.drain_notifications()
        buffers, descs = self._prepared_buffer_lists()
        buffer_count = len(buffers)
        start_time = time.perf_counter()
        direct_registration_count = 0
        direct_registration_time = 0.0
        buffer_index = 0
        staged_ops: dict[int, ReadableOperationGroup] = {}
        pending_staged_buckets: deque[_PendingStagedBucket] = deque()
        direct_ops: deque[ReadableOperationGroup] = deque()
        pending_direct: deque[tuple[Any, NixlBucketMetadata]] = deque()
        bucket_meta: dict[str, TensorMeta] = {}
        bucket_copy_streams: set[int] = set()
        copy_streams: dict[int, torch.cuda.Stream] = {}
        offset = 0
        metadata_batch_size = getattr(self, "metadata_batch_size", 1)
        direct_chunk_size = self._direct_chunk_size()
        pack_children: list[TensorMeta] = []
        pack_dtype: torch.dtype | None = None
        pack_start_offset = 0
        pack_bytes = 0
        pack_index = 0
        total_bytes = 0
        tensor_count = 0
        packed_tensor_count = 0
        transfer_entry_count = 0
        bucket_count = 0
        staged_bucket_count = 0
        direct_bucket_count = 0
        staged_bytes = 0
        direct_bytes = 0
        metadata_publish_count = 0
        contiguous_copy_count = 0
        source_view_time = 0.0
        copy_time = 0.0
        sync_time = 0.0
        send_wait_time = 0.0

        def can_direct_transfer(
            weight: torch.Tensor, buffer: torch.Tensor, made_contiguous_copy: bool
        ) -> bool:
            transfer_mode = _normalize_transfer_mode(
                getattr(self, "transfer_mode", _TRANSFER_MODE_STAGED)
            )
            if transfer_mode == _TRANSFER_MODE_STAGED or made_contiguous_copy:
                return False
            if buffer.device != self.device:
                return False
            if transfer_mode == _TRANSFER_MODE_DIRECT:
                return True
            return weight.nbytes >= self._effective_auto_direct_min_bytes()

        def flush_pack() -> None:
            nonlocal pack_children, pack_dtype, pack_start_offset, pack_bytes
            nonlocal pack_index, transfer_entry_count
            if not pack_children:
                return
            name = f"__nixl_packed_{pack_index}"
            bucket_meta[name] = TensorMeta(
                name=name,
                shape=torch.Size([pack_bytes]),
                dtype=torch.uint8,
                chunk_offset=0,
                chunk_size=pack_bytes,
                offset=pack_start_offset,
                children=pack_children,
            )
            transfer_entry_count += 1
            pack_children = []
            pack_dtype = None
            pack_start_offset = 0
            pack_bytes = 0
            pack_index += 1

        def enqueue_staged_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
            nonlocal copy_time
            copy_start = time.perf_counter()
            cuda_device = src.device if src.device.type == "cuda" else dst.device
            if cuda_device.type != "cuda":
                dst.copy_(src)
                copy_time += time.perf_counter() - copy_start
                return

            cuda_device = _normalize_device(cuda_device)
            device_index = cast(int, cuda_device.index)
            copy_stream = copy_streams.get(device_index)
            if copy_stream is None:
                copy_stream = torch.cuda.Stream(device=cuda_device)
                copy_streams[device_index] = copy_stream

            current_stream = torch.cuda.current_stream(cuda_device)
            copy_stream.wait_stream(current_stream)
            with torch.cuda.device(cuda_device), torch.cuda.stream(copy_stream):
                dst.copy_(src, non_blocking=True)
            bucket_copy_streams.add(device_index)
            copy_time += time.perf_counter() - copy_start

        def record_bucket_copy_events() -> tuple[torch.cuda.Event, ...]:
            events: list[torch.cuda.Event] = []
            for device_index in bucket_copy_streams:
                cuda_device = torch.device("cuda", device_index)
                copy_stream = copy_streams[device_index]
                with torch.cuda.device(cuda_device), torch.cuda.stream(copy_stream):
                    event = torch.cuda.Event()
                    event.record(copy_stream)
                events.append(event)
            return tuple(events)

        async def wait_for_operation(operation: ReadableOperationGroup) -> None:
            nonlocal send_wait_time
            wait_start = time.perf_counter()
            await operation.wait_for_complete()
            send_wait_time += time.perf_counter() - wait_start

        async def publish_staged_buckets(
            *, block_until_buffer: int | None = None, block_all: bool = False
        ) -> None:
            nonlocal metadata_publish_count, sync_time
            target_is_pending = block_until_buffer is not None and any(
                bucket.buffer_index == block_until_buffer
                for bucket in pending_staged_buckets
            )
            ready_buckets: list[_PendingStagedBucket] = []
            while pending_staged_buckets:
                pending_bucket = pending_staged_buckets[0]
                should_block = block_all or target_is_pending
                if pending_bucket.copy_events:
                    if not should_block and not all(
                        event.query() for event in pending_bucket.copy_events
                    ):
                        break
                    sync_start = time.perf_counter()
                    await _wait_cuda_events(pending_bucket.copy_events)
                    sync_time += time.perf_counter() - sync_start

                pending_staged_buckets.popleft()
                ready_buckets.append(pending_bucket)
                if pending_bucket.buffer_index == block_until_buffer:
                    break
                if (
                    not block_all
                    and block_until_buffer is None
                    and len(ready_buckets) >= metadata_batch_size
                ):
                    break

            if ready_buckets:
                groups = _publish_readable_operation_groups(
                    self.agent,
                    next_agents,
                    [
                        (pending_bucket.local_descs, pending_bucket.metadata)
                        for pending_bucket in ready_buckets
                    ],
                )
                metadata_publish_count += len(next_agents)
                for pending_bucket, group in zip(ready_buckets, groups):
                    staged_ops[pending_bucket.buffer_index] = group

        async def wait_for_all_outstanding() -> None:
            await publish_staged_buckets(block_all=True)
            for operation in staged_ops.values():
                await wait_for_operation(operation)
            staged_ops.clear()
            while direct_ops:
                await wait_for_operation(direct_ops.popleft())

        async def publish_direct_pending(*, is_last: bool, force: bool = False) -> None:
            nonlocal pending_direct, bucket_count, direct_bucket_count, direct_bytes
            nonlocal metadata_publish_count
            if not pending_direct:
                return
            await publish_staged_buckets(block_all=True)
            if not is_last and not force and len(pending_direct) <= metadata_batch_size:
                return

            publish_count = (
                len(pending_direct)
                if is_last or force
                else min(len(pending_direct) - 1, metadata_batch_size)
            )
            direct_buckets: list[tuple[Any, NixlBucketMetadata]] = []
            for bucket_offset in range(publish_count):
                local_descs, metadata = pending_direct.popleft()
                metadata["is_last"] = (
                    is_last
                    and bucket_offset == publish_count - 1
                    and not pending_direct
                )
                direct_buckets.append((local_descs, metadata))
                bucket_count += 1
                direct_bucket_count += 1
                direct_bytes += metadata["bucket_bytes"]

            groups = _publish_readable_operation_groups(
                self.agent,
                next_agents,
                direct_buckets,
            )
            metadata_publish_count += len(next_agents)
            direct_ops.extend(groups)

            if is_last:
                await wait_for_all_outstanding()
            while not is_last and len(direct_ops) >= buffer_count:
                await wait_for_operation(direct_ops.popleft())

        async def flush_bucket(*, is_last: bool) -> None:
            nonlocal bucket_meta, bucket_copy_streams, offset, buffer_index
            nonlocal bucket_count, staged_bucket_count, staged_bytes
            flush_pack()
            if offset == 0:
                return

            pending_staged_buckets.append(
                _PendingStagedBucket(
                    buffer_index=buffer_index,
                    local_descs=self._buffer_xfer_desc(buffer_index, offset),
                    metadata=_bucket_metadata(
                        bucket_meta,
                        bucket_bytes=offset,
                        buffer_index=buffer_index,
                        notify_key=self._notify_key(buffer_index),
                        is_last=is_last,
                    ),
                    copy_events=record_bucket_copy_events(),
                )
            )
            await publish_staged_buckets(block_all=False)

            bucket_count += 1
            staged_bucket_count += 1
            staged_bytes += offset

            if is_last:
                await wait_for_all_outstanding()
                return

            next_buffer_index = (buffer_index + 1) % buffer_count
            await publish_staged_buckets(block_until_buffer=next_buffer_index)
            previous_operation = staged_ops.pop(next_buffer_index, None)
            if previous_operation is not None:
                await wait_for_operation(previous_operation)

            buffer_index = next_buffer_index
            bucket_meta = {}
            bucket_copy_streams = set()
            offset = 0

        async def queue_direct_bucket(
            *,
            name: str,
            weight: torch.Tensor,
            buffer: torch.Tensor,
            chunk_offset: int,
            chunk_size: int,
        ) -> None:
            nonlocal pending_direct, transfer_entry_count
            nonlocal direct_registration_count, direct_registration_time
            tensor_meta = TensorMeta(
                name=name,
                shape=weight.shape,
                dtype=weight.dtype,
                chunk_offset=chunk_offset,
                chunk_size=chunk_size,
                offset=0,
            )
            transfer_entry_count += 1
            metadata = _bucket_metadata(
                {name: tensor_meta},
                bucket_bytes=chunk_size,
                buffer_index=-1,
                notify_key=uuid.uuid4().bytes,
                is_last=False,
            )
            metadata["remote_desc_key"] = (
                buffer.data_ptr(),
                chunk_offset,
                chunk_size,
            )
            reg_key = (buffer.data_ptr(), buffer.nbytes)
            direct_reg_cache = getattr(self, "_direct_reg_cache", {})
            had_registration = reg_key in direct_reg_cache
            direct_registration_start = time.perf_counter()
            local_descs = self._direct_xfer_desc(
                buffer, chunk_offset=chunk_offset, chunk_size=chunk_size
            )
            direct_registration_time += time.perf_counter() - direct_registration_start
            if not had_registration:
                direct_registration_count += 1
                self._send_agent_metadata_update(next_agents)
            pending_direct.append((local_descs, metadata))
            await publish_direct_pending(is_last=False)

        for name, weight in weights:
            if weight.nbytes == 0:
                continue
            tensor_count += 1
            total_bytes += weight.nbytes
            view_start = time.perf_counter()
            buffer, made_contiguous_copy = _as_contiguous_uint8(weight)
            source_view_time += time.perf_counter() - view_start
            if made_contiguous_copy:
                contiguous_copy_count += 1
            use_direct = can_direct_transfer(weight, buffer, made_contiguous_copy)
            use_pack = (
                not use_direct
                and weight.nbytes <= _SMALL_TENSOR_PACK_THRESHOLD_BYTES
                and weight.nbytes <= _SMALL_TENSOR_PACK_BYTES
                and weight.nbytes <= self.bucket_size
            )

            if not use_direct:
                await publish_direct_pending(is_last=False, force=True)

            if use_pack:
                if pack_children and (
                    pack_dtype != weight.dtype
                    or pack_bytes + weight.nbytes > _SMALL_TENSOR_PACK_BYTES
                ):
                    flush_pack()
                if offset + weight.nbytes > self.bucket_size:
                    await flush_bucket(is_last=False)
                if not pack_children:
                    pack_dtype = weight.dtype
                    pack_start_offset = offset

                enqueue_staged_copy(
                    buffers[buffer_index][offset : offset + weight.nbytes],
                    buffer,
                )
                pack_children.append(
                    TensorMeta(
                        name=name,
                        shape=weight.shape,
                        dtype=weight.dtype,
                        chunk_offset=0,
                        chunk_size=weight.nbytes,
                        offset=offset - pack_start_offset,
                    )
                )
                offset += weight.nbytes
                pack_bytes += weight.nbytes
                packed_tensor_count += 1
                continue

            flush_pack()
            chunk_offset = 0
            while chunk_offset < weight.nbytes:
                chunk_size = min(
                    direct_chunk_size if use_direct else self.bucket_size,
                    weight.nbytes - chunk_offset,
                )
                if use_direct:
                    if offset > 0:
                        await flush_bucket(is_last=False)
                    await queue_direct_bucket(
                        name=name,
                        weight=weight,
                        buffer=buffer,
                        chunk_offset=chunk_offset,
                        chunk_size=chunk_size,
                    )
                    chunk_offset += chunk_size
                    continue

                if offset + chunk_size > self.bucket_size:
                    await flush_bucket(is_last=False)

                tensor_meta = TensorMeta(
                    name=name,
                    shape=weight.shape,
                    dtype=weight.dtype,
                    chunk_offset=chunk_offset,
                    chunk_size=chunk_size,
                    offset=offset,
                )
                bucket_meta[name] = tensor_meta
                transfer_entry_count += 1
                enqueue_staged_copy(
                    buffers[buffer_index][offset : offset + chunk_size],
                    buffer[chunk_offset : chunk_offset + chunk_size],
                )
                offset += chunk_size
                chunk_offset += chunk_size

        if pending_direct:
            await publish_direct_pending(is_last=True)
        else:
            await flush_bucket(is_last=True)
        elapsed = time.perf_counter() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024) if elapsed > 0 else 0
        logger.info(
            "NIXL send_weights completed: tensors=%d packed=%d "
            "direct_registered=%d contiguous_copies=%d entries=%d "
            "buckets=%d staged_buckets=%d "
            "direct_buckets=%d bytes=%.2fGiB staged_bytes=%.2fGiB "
            "direct_bytes=%.2fGiB "
            "time=%.2fs logical_bandwidth=%.2fGiB/s direct_registration=%.2fs "
            "source_view=%.2fs "
            "copy_enqueue=%.2fs sync=%.2fs wait=%.2fs "
            "direct_stripe_count=%d metadata_batch_size=%d "
            "metadata_publishes=%d",
            tensor_count,
            packed_tensor_count,
            direct_registration_count,
            contiguous_copy_count,
            transfer_entry_count,
            bucket_count,
            staged_bucket_count,
            direct_bucket_count,
            total_bytes / (1024 * 1024 * 1024),
            staged_bytes / (1024 * 1024 * 1024),
            direct_bytes / (1024 * 1024 * 1024),
            elapsed,
            bandwidth,
            direct_registration_time,
            source_view_time,
            copy_time,
            sync_time,
            send_wait_time,
            getattr(self, "direct_stripe_count", 1),
            metadata_batch_size,
            metadata_publish_count,
        )

    async def receive_weight_batches(
        self,
    ) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        merge_name: str | None = None
        merge_weight: torch.Tensor | None = None
        merge_offset = 0
        pending_weight_batch: list[tuple[str, torch.Tensor]] = []
        pending_leases: list[_ReceivedChunkBatch] = []
        pending_bucket_count = 0
        chunk_batch_count = 0
        chunk_count = 0
        weight_count = 0
        logical_bytes = 0
        alloc_time = 0.0
        copy_time = 0.0
        view_time = 0.0
        release_time = 0.0
        release_tasks: set[asyncio.Task[None]] = set()

        async def release_leases(
            leases: list[_ReceivedChunkBatch], *, sync_devices: bool = True
        ) -> None:
            nonlocal release_time
            release_start = time.perf_counter()
            for lease in leases:
                await lease.release(sync_devices=sync_devices)
            release_time += time.perf_counter() - release_start

        async def release_weight_batch(
            weight_batch: _LoadCompleteWeightBatch,
            leases: list[_ReceivedChunkBatch],
        ) -> None:
            nonlocal release_time
            release_start = time.perf_counter()
            await weight_batch.wait_for_cuda_load_complete_async()
            for lease in leases:
                await lease.release(sync_devices=False)
            release_time += time.perf_counter() - release_start

        async def drain_finished_release_tasks() -> None:
            done_tasks = {task for task in release_tasks if task.done()}
            for task in done_tasks:
                release_tasks.remove(task)
                await task

        async def release_or_schedule_weight_batch(
            weight_batch: _LoadCompleteWeightBatch,
            leases: list[_ReceivedChunkBatch],
        ) -> None:
            if not leases:
                return
            if any(lease.release_blocks_next for lease in leases):
                await release_weight_batch(weight_batch, leases)
                return

            task = asyncio.create_task(release_weight_batch(weight_batch, leases))
            release_tasks.add(task)
            await drain_finished_release_tasks()

        async def process_lease(lease: _ReceivedChunkBatch) -> bool:
            nonlocal merge_name, merge_weight, merge_offset
            nonlocal chunk_count, weight_count, logical_bytes
            nonlocal alloc_time, copy_time, view_time
            lease_has_views = False
            chunk_count += len(lease.chunks)

            for tensor_meta, chunk in lease.chunks:
                if chunk.dtype != torch.uint8:
                    raise TypeError(
                        f"Checkpoint-engine chunks must be uint8, got {chunk.dtype}"
                    )

                if tensor_meta.children:
                    lease_has_views = True
                    for child_meta in tensor_meta.children:
                        if child_meta.offset is None:
                            raise RuntimeError(
                                f"Missing packed offset for {child_meta.name}."
                            )
                        weight = chunk[
                            child_meta.offset : child_meta.offset
                            + child_meta.chunk_size
                        ]
                        view_start = time.perf_counter()
                        pending_weight_batch.append(
                            (
                                child_meta.name,
                                weight.view(child_meta.dtype).view(child_meta.shape),
                            )
                        )
                        view_time += time.perf_counter() - view_start
                        weight_count += 1
                        logical_bytes += child_meta.nbytes
                    continue

                if (
                    tensor_meta.chunk_offset == 0
                    and tensor_meta.chunk_size == tensor_meta.nbytes
                ):
                    if merge_weight is not None:
                        raise RuntimeError(f"Unexpected open merge for {merge_name}.")
                    lease_has_views = True
                    view_start = time.perf_counter()
                    pending_weight_batch.append(
                        (
                            tensor_meta.name,
                            chunk.view(tensor_meta.dtype).view(tensor_meta.shape),
                        )
                    )
                    view_time += time.perf_counter() - view_start
                    weight_count += 1
                    logical_bytes += tensor_meta.nbytes
                    continue

                if merge_weight is None:
                    if tensor_meta.chunk_offset != 0:
                        raise RuntimeError(
                            f"First chunk for {tensor_meta.name} starts at "
                            f"{tensor_meta.chunk_offset}, expected 0."
                        )
                    merge_name = tensor_meta.name
                    alloc_start = time.perf_counter()
                    merge_weight = torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device=chunk.device,
                    )
                    alloc_time += time.perf_counter() - alloc_start
                    merge_offset = 0

                if tensor_meta.name != merge_name:
                    raise RuntimeError(
                        f"Expected chunk for {merge_name}, got {tensor_meta.name}."
                    )
                if merge_offset != tensor_meta.chunk_offset:
                    raise RuntimeError(
                        f"Expected chunk offset {merge_offset}, "
                        f"got {tensor_meta.chunk_offset}."
                    )

                copy_start = time.perf_counter()
                merge_weight.view(-1).view(torch.uint8)[
                    tensor_meta.chunk_offset : tensor_meta.chunk_offset
                    + tensor_meta.chunk_size
                ] = chunk
                copy_time += time.perf_counter() - copy_start
                merge_offset += tensor_meta.chunk_size

                if (
                    tensor_meta.chunk_offset + tensor_meta.chunk_size
                    == tensor_meta.nbytes
                ):
                    pending_weight_batch.append((merge_name, merge_weight))
                    weight_count += 1
                    logical_bytes += tensor_meta.nbytes
                    merge_name = None
                    merge_weight = None
                    merge_offset = 0

            return lease_has_views

        load_batch_bucket_count = getattr(self, "load_batch_bucket_count", 1)

        try:
            async for lease in self._receive_weight_chunk_batch_leases():
                await drain_finished_release_tasks()
                chunk_batch_count += 1
                pending_bucket_count += 1
                lease_has_views = await process_lease(lease)
                if lease_has_views:
                    pending_leases.append(lease)
                else:
                    await release_leases([lease])

                if (
                    pending_weight_batch
                    and pending_bucket_count >= load_batch_bucket_count
                ):
                    weight_batch = _LoadCompleteWeightBatch(pending_weight_batch)
                    leases = pending_leases
                    pending_weight_batch = []
                    pending_leases = []
                    pending_bucket_count = 0
                    try:
                        yield weight_batch
                    finally:
                        await release_or_schedule_weight_batch(weight_batch, leases)

            if merge_weight is not None:
                raise RuntimeError(f"Unfinished tensor merge for {merge_name}.")

            if pending_weight_batch:
                weight_batch = _LoadCompleteWeightBatch(pending_weight_batch)
                leases = pending_leases
                pending_weight_batch = []
                pending_leases = []
                try:
                    yield weight_batch
                finally:
                    await release_or_schedule_weight_batch(weight_batch, leases)
            elif pending_leases:
                await release_leases(pending_leases)
        finally:
            if release_tasks:
                await asyncio.gather(*release_tasks)

        if chunk_batch_count > 0:
            logger.info(
                "Checkpoint-engine merge completed: batches=%d chunks=%d "
                "weights=%d bytes=%.2fGiB alloc=%.2fs copy=%.2fs "
                "view=%.2fs release=%.2fs load_batch_bucket_count=%d",
                chunk_batch_count,
                chunk_count,
                weight_count,
                logical_bytes / (1024 * 1024 * 1024),
                alloc_time,
                copy_time,
                view_time,
                release_time,
                load_batch_bucket_count,
            )

    async def _receive_weight_chunk_batches(
        self,
    ) -> AsyncGenerator[list[tuple[TensorMeta, torch.Tensor]], None]:
        async for lease in self._receive_weight_chunk_batch_leases():
            try:
                yield lease.chunks
            finally:
                await lease.release()

    async def _receive_weight_chunk_batch_leases(
        self,
    ) -> AsyncGenerator[_ReceivedChunkBatch, None]:
        if self.prev_agent is None:
            raise RuntimeError("NIXL receiver rank has no previous peer.")

        buffers, _descs = self._prepared_buffer_lists()
        buffer_count = len(buffers)
        total_bytes = 0
        total_chunks = 0
        total_buckets = 0
        metadata_wait_time = 0.0
        read_wait_time = 0.0
        forward_wait_time = 0.0
        sync_time = 0.0
        start_time = time.perf_counter()

        async def read_metadata(*, block: bool) -> dict[str, Any] | None:
            nonlocal metadata_wait_time
            while True:
                metadata_start = time.perf_counter()
                if block:
                    metadata = await self.agent.read_message(self.prev_agent)
                else:
                    metadata = await self.agent.try_read_message(self.prev_agent)
                metadata_wait_time += time.perf_counter() - metadata_start
                if metadata is None:
                    return None
                if metadata.get("message_type") == _MESSAGE_AGENT_METADATA_UPDATE:
                    self._refresh_prev_agent(metadata["agent_metadata"])
                    continue
                return metadata

        async def wait_for_read(read_op: ReadOperation) -> None:
            nonlocal read_wait_time
            wait_start = time.perf_counter()
            await read_op.wait_for_complete()
            read_wait_time += time.perf_counter() - wait_start

        async def wait_for_forward(
            readable_op: ReadableOperationGroup | None,
        ) -> None:
            nonlocal forward_wait_time
            if readable_op is None:
                return
            wait_start = time.perf_counter()
            await readable_op.wait_for_complete()
            forward_wait_time += time.perf_counter() - wait_start

        buffer_order = list(range(1, buffer_count)) + [0]
        free_buffer_indices = deque(buffer_order)
        inflight_reads: deque[tuple[NixlBucketMetadata, ReadOperation, int]] = deque()
        saw_last_bucket = False

        async def prefetch_read(*, block: bool) -> bool:
            nonlocal saw_last_bucket
            if saw_last_bucket or not free_buffer_indices:
                return False

            metadata = await read_metadata(block=block)
            if metadata is None:
                return False

            buffer_index = free_buffer_indices.popleft()
            metadata, read_op = self._read_operation(
                local_descs=self._buffer_xfer_desc(
                    buffer_index, metadata["bucket_bytes"]
                ),
                local_buffer_index=buffer_index,
                metadata=metadata,
            )
            inflight_reads.append((metadata, read_op, buffer_index))
            saw_last_bucket = metadata["is_last"]
            return True

        async def fill_prefetch_window() -> None:
            while (
                not saw_last_bucket
                and free_buffer_indices
                and len(inflight_reads) < buffer_count - 1
            ):
                block = len(inflight_reads) == 0
                if not await prefetch_read(block=block):
                    break

        def release_bucket(
            *,
            readable_op: ReadableOperationGroup | None,
            buffer_index: int,
        ) -> Callable[[bool], Awaitable[None]]:
            async def release(sync_devices: bool = True) -> None:
                nonlocal sync_time
                await wait_for_forward(readable_op)
                if sync_devices:
                    sync_start = time.perf_counter()
                    _sync_devices((self.device,))
                    sync_time += time.perf_counter() - sync_start
                free_buffer_indices.append(buffer_index)
                await fill_prefetch_window()

            return release

        await prefetch_read(block=True)

        while inflight_reads:
            metadata, read_op, current_buffer_index = inflight_reads.popleft()
            await wait_for_read(read_op)
            self.agent.send_message(
                self.prev_agent,
                {
                    "message_type": _MESSAGE_BUCKET_COMPLETE,
                    "notify_key": metadata["notify_key"],
                },
            )
            total_bytes += metadata["bucket_bytes"]
            total_chunks += len(metadata["bucket_meta"])
            total_buckets += 1

            readable_op = None
            next_agents = getattr(self, "next_agents", [])
            if not next_agents and self.next_agent is not None:
                next_agents = [self.next_agent]
            if next_agents:
                readable_op = ReadableOperationGroup(
                    self.agent,
                    next_agents,
                    self._buffer_xfer_desc(
                        current_buffer_index, metadata["bucket_bytes"]
                    ),
                    self._forward_metadata(
                        metadata,
                        buffer_index=current_buffer_index,
                    ),
                )

            await fill_prefetch_window()
            release_blocks_next = not inflight_reads and not saw_last_bucket

            yield _ReceivedChunkBatch(
                _bucket_chunks(metadata, buffers[current_buffer_index]),
                metadata["bucket_bytes"],
                release_bucket(
                    readable_op=readable_op,
                    buffer_index=current_buffer_index,
                ),
                release_blocks_next=release_blocks_next,
            )

        wall_time = time.perf_counter() - start_time
        active_time = (
            metadata_wait_time + read_wait_time + forward_wait_time + sync_time
        )
        bandwidth = (
            total_bytes / active_time / (1024 * 1024 * 1024) if active_time > 0 else 0
        )
        logger.info(
            "NIXL receive_weights completed: buckets=%d chunks=%d bytes=%.2fGiB "
            "active=%.2fs wall=%.2fs logical_bandwidth=%.2fGiB/s "
            "metadata_wait=%.2fs read_wait=%.2fs forward_wait=%.2fs sync=%.2fs",
            total_buckets,
            total_chunks,
            total_bytes / (1024 * 1024 * 1024),
            active_time,
            wall_time,
            bandwidth,
            metadata_wait_time,
            read_wait_time,
            forward_wait_time,
            sync_time,
        )
