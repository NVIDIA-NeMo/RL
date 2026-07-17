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
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

import ray
import torch
import zmq
import zmq.asyncio

from nemo_rl.utils.checkpoint_engines.base import (
    CheckpointEngine,
    TensorMeta,
    merge_weight_chunk_batches,
    split_weight_chunks,
)

NixlAgentMetadata = dict[str, Any]
NIXL_DEFAULT_BACKEND_NAME = "UCX"


def _source_rank_for_rollout(
    rollout_rank: int,
    *,
    train_world_size: int,
    rollout_world_size: int,
) -> int:
    if rollout_rank < 0 or rollout_rank >= rollout_world_size:
        raise ValueError(
            f"rollout_rank must be in [0, {rollout_world_size}), got {rollout_rank}."
        )
    if train_world_size < 1:
        raise ValueError("train_world_size must be >= 1.")
    if rollout_world_size < 1:
        raise ValueError("rollout_world_size must be >= 1.")
    if train_world_size < rollout_world_size:
        raise ValueError(
            "NIXL checkpoint-engine refit requires train_world_size >= "
            f"rollout_world_size, got {train_world_size} < {rollout_world_size}."
        )

    return rollout_rank


def _create_nixl_agent(
    agent_name: str,
    backend_name: str,
    backend_init_params: dict[str, Any] | None = None,
) -> Any:  # pragma: no cover
    try:
        nixl_api = importlib.import_module("nixl._api")
    except ImportError as exc:
        raise ImportError("Install NIXL or disable checkpoint-engine refit.") from exc
    if backend_name == "UCX" and backend_init_params is None:
        return nixl_api.nixl_agent(agent_name)

    agent = nixl_api.nixl_agent(agent_name, nixl_api.nixl_agent_config(backends=[]))
    agent.create_backend(
        backend_name,
        {key: str(value) for key, value in (backend_init_params or {}).items()},
    )
    return agent


def resolve_nixl_backend_kwargs(
    nixl_kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """Resolve ``(backend_name, backend_init_params)`` from ``engine_kwargs.nixl``.

    Single source for the NIXL backend-name default so preinit call sites don't
    each repeat ``.get("backend_name", NIXL_DEFAULT_BACKEND_NAME)``.
    """
    return (
        nixl_kwargs.get("backend_name", NIXL_DEFAULT_BACKEND_NAME),
        nixl_kwargs.get("backend_init_params"),
    )


def preinit_nixl_agent(
    *,
    backend_name: str = NIXL_DEFAULT_BACKEND_NAME,
    backend_init_params: dict[str, Any] | None = None,
) -> Any:  # pragma: no cover
    agent = _create_nixl_agent(
        f"preinit-{uuid.uuid4()}", backend_name, backend_init_params
    )
    agent.get_agent_metadata()
    return agent


def _sync_device(device: torch.device) -> None:  # pragma: no cover
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.cuda.is_available():
        # Pinned host transfer buffers receive asynchronous device-to-host
        # copies (non_blocking=True from CUDA weight tensors). Flush the current
        # CUDA stream before the peer RDMA-reads the buffer, otherwise the reader
        # can observe stale/partial weights with no error.
        torch.cuda.current_stream().synchronize()


class NixlAgent:  # pragma: no cover
    def __init__(
        self,
        backend_name: str = NIXL_DEFAULT_BACKEND_NAME,
        backend_init_params: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = str(uuid.uuid4())
        self.agent = _create_nixl_agent(
            self.agent_name, backend_name, backend_init_params
        )
        self.messages: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self.notifications: dict[str, deque[bytes]] = defaultdict(deque)
        self.zmq_clients: dict[str, zmq.Socket] = {}
        self.zmq_client_context = zmq.Context()
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.zmq_context = zmq.asyncio.Context()
        self.socket = self.zmq_context.socket(zmq.PULL)
        self.listen_port = self.socket.bind_to_random_port(f"tcp://{self.ip}")

    def get_agent_metadata(self) -> NixlAgentMetadata:
        return {
            "agent_name": self.agent_name,
            "agent_metadata": self.agent.get_agent_metadata(),
            "zmq_ip": self.ip,
            "zmq_port": self.listen_port,
        }

    def add_remote_agent(self, metadata: NixlAgentMetadata) -> str:
        remote = self.agent.add_remote_agent(metadata["agent_metadata"])
        agent_name = remote.decode("utf-8") if isinstance(remote, bytes) else remote
        socket = self.zmq_client_context.socket(zmq.PUSH)
        socket.connect(f"tcp://{metadata['zmq_ip']}:{metadata['zmq_port']}")
        self.zmq_clients[agent_name] = socket
        return agent_name

    def remove_remote_agent(self, agent_name: str) -> None:
        self.agent.remove_remote_agent(agent_name)
        self.zmq_clients.pop(agent_name).close(linger=0)

    def send_message(self, agent_name: str, message: dict[str, Any]) -> None:
        self.zmq_clients[agent_name].send_pyobj(
            (self.agent_name, message), zmq.DONTWAIT
        )

    async def read_message(self, agent_name: str) -> dict[str, Any]:
        while not self.messages[agent_name]:
            if callable(progress := getattr(self.agent, "progress", None)):
                progress()
            try:
                remote_agent_name, message = await self.socket.recv_pyobj(zmq.DONTWAIT)
            except zmq.Again:
                await asyncio.sleep(0)
                continue
            self.messages[remote_agent_name].append(message)
        return self.messages[agent_name].popleft()

    async def wait_notification(self, agent_name: str, notify_key: bytes) -> None:
        while True:
            pending_notifications = self.notifications[agent_name]
            for notification in pending_notifications:
                if notification == notify_key:
                    pending_notifications.remove(notification)
                    return

            if callable(progress := getattr(self.agent, "progress", None)):
                progress()
            new_notifications = self.agent.get_new_notifs()
            for remote_agent_name, notifications in new_notifications.items():
                if isinstance(remote_agent_name, bytes):
                    remote_agent_name = remote_agent_name.decode("utf-8")
                self.notifications[remote_agent_name].extend(notifications)
            await asyncio.sleep(0)


class NIXLCheckpointEngine(CheckpointEngine):  # pragma: no cover
    def __init__(
        self,
        bucket_size: int,
        device: str | torch.device = "cuda",
        backend_name: str = NIXL_DEFAULT_BACKEND_NAME,
        backend_init_params: dict[str, Any] | None = None,
        shard_expert_weights: bool = False,
        release_after_refit: bool = False,
    ) -> None:
        if bucket_size < 1:
            raise ValueError("NIXL checkpoint-engine bucket_size must be >= 1 byte.")
        self.bucket_size = bucket_size
        self.shard_expert_weights = shard_expert_weights
        self.release_after_refit = release_after_refit
        self._target_weight_layout: dict[str, Any] | None = None
        self.agent = NixlAgent(backend_name, backend_init_params)
        self.prev_agent: str | None = None
        self.next_agent: str | None = None
        self.buffers: list[torch.Tensor] = []
        self.registration_descs: list[Any] = []
        self.xfer_descs: list[Any] = []
        transfer_device = torch.device(device)
        if transfer_device.type == "cuda" and transfer_device.index is None:
            transfer_device = torch.device("cuda", torch.cuda.current_device())
        self._transfer_device = transfer_device  # pyrefly: ignore[read-only]
        self._cupy_buffers: list[Any] = []
        self._cupy_memory_pool: Any | None = None
        self._uses_torch_cuda_buffers = False

    def _allocate_transfer_buffer(self) -> torch.Tensor:
        device = self._transfer_device
        if device.type != "cuda":
            return torch.zeros(
                self.bucket_size,
                dtype=torch.uint8,
                device=device,
                pin_memory=torch.cuda.is_available(),
            )

        torch.cuda.set_device(device)
        try:
            cupy = importlib.import_module("cupy")
        except ImportError:
            self._uses_torch_cuda_buffers = True
            return torch.zeros(self.bucket_size, dtype=torch.uint8, device=device)
        if self._cupy_memory_pool is None:
            self._cupy_memory_pool = cupy.cuda.MemoryPool()
        with (
            cupy.cuda.Device(device.index),
            cupy.cuda.using_allocator(self._cupy_memory_pool.malloc),
        ):
            cupy_buffer = cupy.zeros(self.bucket_size, dtype=cupy.uint8)
        self._cupy_buffers.append(cupy_buffer)
        return torch.as_tensor(cupy_buffer, dtype=torch.uint8, device=device)

    def prepare(self) -> NixlAgentMetadata:
        if not self.buffers:
            self.buffers = [self._allocate_transfer_buffer() for _ in range(2)]
            for buffer in self.buffers:
                self.registration_descs.append(self.agent.agent.register_memory(buffer))
                self.xfer_descs.append(self.agent.agent.get_xfer_descs(buffer))
        return self.agent.get_agent_metadata()

    def get_target_weight_layout(self) -> dict[str, Any] | None:
        return self._target_weight_layout

    def init_policy_process_group(
        self,
        *,
        worker_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[NixlAgentMetadata],
    ) -> None:
        self._disconnect_peers()
        source_to_rollout = {
            _source_rank_for_rollout(
                rollout_rank,
                train_world_size=train_world_size,
                rollout_world_size=rollout_world_size,
            ): rollout_rank
            for rollout_rank in range(rollout_world_size)
        }
        rollout_rank = source_to_rollout.get(worker_rank)
        if rollout_rank is not None:
            target_metadata = metadata[train_world_size + rollout_rank]
            if self.shard_expert_weights:
                self._target_weight_layout = target_metadata["weight_layout"]
            self.next_agent = self.agent.add_remote_agent(target_metadata)

    def init_rollout_process_group(
        self,
        *,
        rollout_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[NixlAgentMetadata],
    ) -> None:
        self._disconnect_peers()
        source_rank = _source_rank_for_rollout(
            rollout_rank,
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
        )
        self.prev_agent = self.agent.add_remote_agent(metadata[source_rank])

    def _disconnect_peers(self) -> None:
        if self.prev_agent is not None:
            self.agent.remove_remote_agent(self.prev_agent)
        if self.next_agent is not None:
            self.agent.remove_remote_agent(self.next_agent)
        self.prev_agent = None
        self.next_agent = None
        self._target_weight_layout = None

    def _release_transfer_buffers(self) -> None:
        _sync_device(self._transfer_device)
        for registration_desc in self.registration_descs:
            self.agent.agent.deregister_memory(registration_desc)
        self.registration_descs.clear()
        self.xfer_descs.clear()
        self.buffers.clear()
        self._cupy_buffers.clear()

        if self._cupy_memory_pool is not None:
            self._cupy_memory_pool.free_all_blocks()
            self._cupy_memory_pool = None
        elif self._uses_torch_cuda_buffers:
            torch.cuda.empty_cache()
        self._uses_torch_cuda_buffers = False

    def finalize(self) -> None:
        self._disconnect_peers()
        if self.release_after_refit:
            self._release_transfer_buffers()

    @torch.no_grad()
    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> None:
        if self.next_agent is None:
            # DTensor-backed iterators can run collectives while materializing
            # tensors. Ranks without rollout peers must still participate.
            for _ in weights:
                pass
            return
        next_agent = cast(str, self.next_agent)

        buffers = self.buffers
        descs = self.xfer_descs
        buffer_index = 0
        offset: int = 0
        bucket_meta: dict[str, TensorMeta] = {}
        pending_key: bytes | None = None

        async def wait_readers(notify_key: bytes | None) -> None:
            if notify_key is None:
                return
            await self.agent.wait_notification(next_agent, notify_key)

        async def flush_bucket(
            *,
            bucket_buffer_index: int,
            bucket_offset: int,
            bucket_metadata: dict[str, TensorMeta],
            previous_key: bytes | None,
            is_last: bool,
        ) -> tuple[int, int, dict[str, TensorMeta], bytes | None]:
            if bucket_offset == 0 and not is_last:
                return bucket_buffer_index, bucket_offset, bucket_metadata, previous_key
            _sync_device(self._transfer_device)
            await wait_readers(previous_key)
            notify_key = uuid.uuid4().bytes
            metadata = {
                "bucket_meta": bucket_metadata,
                "notify_key": notify_key,
                "is_last": is_last,
                "remote_descs": descs[bucket_buffer_index],
            }
            self.agent.send_message(next_agent, metadata)
            next_bucket_meta: dict[str, TensorMeta] = {}
            return 1 - bucket_buffer_index, 0, next_bucket_meta, notify_key

        for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            alignment = max(1, tensor_meta.dtype.itemsize)
            aligned_offset = ((offset + alignment - 1) // alignment) * alignment
            if aligned_offset + tensor_meta.chunk_size > self.bucket_size:
                buffer_index, offset, bucket_meta, pending_key = await flush_bucket(
                    bucket_buffer_index=buffer_index,
                    bucket_offset=offset,
                    bucket_metadata=bucket_meta,
                    previous_key=pending_key,
                    is_last=False,
                )
                aligned_offset = 0
            offset = aligned_offset
            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            buffers[buffer_index][offset : offset + tensor_meta.chunk_size].copy_(
                chunk, non_blocking=True
            )
            offset += tensor_meta.chunk_size

        buffer_index, offset, bucket_meta, pending_key = await flush_bucket(
            bucket_buffer_index=buffer_index,
            bucket_offset=offset,
            bucket_metadata=bucket_meta,
            previous_key=pending_key,
            is_last=True,
        )
        await wait_readers(pending_key)

    async def receive_weight_batches(
        self,
    ) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        # Keep full tensors off GPU when they span multiple RDMA buckets.
        async for batch in merge_weight_chunk_batches(
            self._receive_weight_chunk_batches(),
            merge_device="cpu" if self._transfer_device.type == "cuda" else None,
        ):
            yield batch

    async def _wait_read(self, xfer_handle: Any, remote_agent: str) -> None:
        while True:
            if callable(progress := getattr(self.agent.agent, "progress", None)):
                progress()
            state = self.agent.agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                self.agent.agent.release_xfer_handle(xfer_handle)
                return
            if state == "ERR":
                raise RuntimeError(f"NIXL read from {remote_agent} failed.")
            await asyncio.sleep(0)

    async def _receive_weight_chunk_batches(
        self,
    ) -> AsyncGenerator[list[tuple[TensorMeta, torch.Tensor]], None]:
        prev_agent = self.prev_agent
        if prev_agent is None:
            raise RuntimeError("NIXL rollout process group is not initialized.")

        buffers = self.buffers
        descs = self.xfer_descs
        buffer_index = 1
        while True:
            message = await self.agent.read_message(prev_agent)
            xfer_handle = self.agent.agent.initialize_xfer(
                "READ",
                descs[buffer_index],
                message["remote_descs"],
                prev_agent,
                message["notify_key"],
            )
            if self.agent.agent.transfer(xfer_handle) == "ERR":
                raise RuntimeError(f"NIXL read from {prev_agent} failed to start.")
            await self._wait_read(xfer_handle, prev_agent)

            chunks = [
                (
                    meta,
                    buffers[buffer_index][
                        int(meta.offset) : int(meta.offset) + meta.chunk_size
                    ],
                )
                for meta in message["bucket_meta"].values()
            ]
            if chunks:
                yield chunks
            _sync_device(self._transfer_device)
            if message["is_last"]:
                break
            buffer_index = 1 - buffer_index
