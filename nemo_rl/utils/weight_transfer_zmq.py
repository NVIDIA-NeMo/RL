# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Transactional ZeroMQ value plane for remote sparse vLLM refit."""

import json
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

import zmq

from nemo_rl.utils.weight_transfer_remote_sparse import (
    SparsePartitionMode,
    merge_vllm_refit_metrics,
    post_vllm_refit_endpoints,
    refit_env_int,
    sparse_payload_checksum,
    stream_sparse_delta_payloads,
    vllm_refit_api_key,
    vllm_refit_endpoints,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    NamedTensor,
)

G_VLLM_REFIT_ZMQ_PAYLOAD_PATH = "/nemo-rl/refit/zmq-payload"
G_VLLM_REFIT_TRANSFER_HEADER = "x-nemo-rl-refit-transfer"
G_VLLM_REFIT_PRODUCER_HEADER = "x-nemo-rl-refit-producer"
G_VLLM_REFIT_PAYLOAD_HEADER = "x-nemo-rl-refit-payload"
G_VLLM_REFIT_CHECKSUM_HEADER = "x-nemo-rl-refit-checksum"

_PROTOCOL = "nemo-rl-sparse-zmq-v1"
_DATA = b"DATA"
_ACK = b"ACK"
_NACK = b"NACK"
_ZMQ_LOCAL = threading.local()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _configure_socket(socket: zmq.Socket, high_water_mark: int) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, high_water_mark)
    socket.setsockopt(zmq.RCVHWM, high_water_mark)
    socket.setsockopt(zmq.TCP_KEEPALIVE, 1)


class ZmqSparseRefitClient:
    """One-thread DEALER client with retry-safe payload identifiers."""

    def __init__(
        self,
        address: str,
        *,
        timeout_s: float,
        producer_id: int,
        api_key: str | None = None,
    ) -> None:
        self._address = address
        self._timeout_ms = max(1, int(timeout_s * 1000))
        self._producer_id = producer_id
        self._api_key = api_key
        self._socket = zmq.Context.instance().socket(zmq.DEALER)
        _configure_socket(self._socket, 2)
        self._socket.setsockopt(
            zmq.IDENTITY, f"nrl-{producer_id}-{uuid.uuid4().hex}".encode()
        )
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(address)

    def send_payload(
        self,
        *,
        transfer_id: str,
        payload_id: int,
        checksum: str,
        body: bytes,
    ) -> dict[str, Any]:
        metadata = {
            "protocol": _PROTOCOL,
            "transfer_id": transfer_id,
            "producer_id": self._producer_id,
            "payload_id": payload_id,
            "checksum": checksum,
        }
        if self._api_key is not None:
            metadata["api_key"] = self._api_key
        metadata_frame = _json_bytes(metadata)
        retries = refit_env_int("NRL_REFIT_ZMQ_RETRIES", default=3, min_value=0)

        for attempt in range(retries + 1):
            try:
                self._socket.send_multipart(
                    [_DATA, metadata_frame, body],
                    copy=False,
                )
            except zmq.Again:
                if attempt == retries:
                    break
                continue

            deadline = time.monotonic() + self._timeout_ms / 1000
            while True:
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                if deadline <= time.monotonic() or not self._socket.poll(
                    remaining_ms, zmq.POLLIN
                ):
                    break
                frames = self._socket.recv_multipart()
                if len(frames) != 2:
                    continue
                kind, raw_reply = frames
                reply = json.loads(raw_reply)
                if kind == _NACK and "transfer_id" not in reply:
                    raise RuntimeError(f"ZeroMQ sparse refit rejected payload: {reply}")
                reply_key = (
                    reply.get("transfer_id"),
                    reply.get("producer_id"),
                    reply.get("payload_id"),
                )
                if reply_key != (transfer_id, self._producer_id, payload_id):
                    continue
                if kind == _ACK and reply.get("ok") is True:
                    return reply
                raise RuntimeError(f"ZeroMQ sparse refit rejected payload: {reply}")

            if attempt < retries:
                time.sleep(min(0.05 * 2**attempt, 0.5))

        raise TimeoutError(
            f"Timed out sending sparse refit payload {payload_id} to {self._address}."
        )

    def close(self) -> None:
        self._socket.close()


class ZmqSparseRefitServer:
    """Bounded ROUTER relay that fans each compressed payload to all replicas."""

    def __init__(
        self,
        refit_urls: Sequence[str],
        *,
        bind_address: str,
        api_key_env_var: str | None,
        timeout_s: float,
    ) -> None:
        self._refit_endpoints = vllm_refit_endpoints(
            refit_urls, G_VLLM_REFIT_ZMQ_PAYLOAD_PATH
        )
        if not self._refit_endpoints:
            raise ValueError("ZeroMQ sparse refit requires receiver HTTP URLs.")
        self._bind_address = bind_address
        self._token = vllm_refit_api_key(api_key_env_var)
        self._timeout_s = timeout_s
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._endpoint: str | None = None
        self._error: Exception | None = None
        self._payload_workers = refit_env_int(
            "NRL_REFIT_ZMQ_RELAY_PAYLOAD_WORKERS", default=16
        )
        self._fanout_workers = refit_env_int(
            "NRL_REFIT_ZMQ_RELAY_FANOUT_WORKERS",
            default=max(8, min(32, len(self._refit_endpoints) * self._payload_workers)),
        )

    def start(self) -> str:
        self._thread = threading.Thread(
            target=self._run,
            name="nrl-zmq-refit-relay",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("Timed out starting the ZeroMQ sparse refit relay.")
        if self._error is not None:
            raise RuntimeError(
                "Failed to start the ZeroMQ sparse refit relay."
            ) from self._error
        assert self._endpoint is not None
        return self._endpoint

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._timeout_s))
            if self._thread.is_alive():
                raise RuntimeError("Timed out stopping the ZeroMQ sparse refit relay.")
        self._thread = None

    def _fanout(
        self,
        body: bytes,
        metadata: Mapping[str, Any],
        http_executor: ThreadPoolExecutor,
    ) -> dict[str, Any]:
        headers = {
            "content-type": "application/octet-stream",
            G_VLLM_REFIT_TRANSFER_HEADER: str(metadata["transfer_id"]),
            G_VLLM_REFIT_PRODUCER_HEADER: str(metadata["producer_id"]),
            G_VLLM_REFIT_PAYLOAD_HEADER: str(metadata["payload_id"]),
            G_VLLM_REFIT_CHECKSUM_HEADER: str(metadata["checksum"]),
        }
        started = time.perf_counter()
        results = post_vllm_refit_endpoints(
            self._refit_endpoints,
            body,
            api_key=self._token,
            timeout_s=self._timeout_s,
            headers=headers,
            executor=http_executor,
        )
        merged = merge_vllm_refit_metrics({}, results, maximum=True)
        merged["receiver_relay_fanout_s"] = time.perf_counter() - started
        return merged

    @staticmethod
    def _send_reply(
        socket: zmq.Socket,
        identity: bytes,
        kind: bytes,
        reply: Mapping[str, Any],
    ) -> None:
        with suppress(zmq.ZMQError):
            socket.send_multipart(
                [identity, kind, _json_bytes(reply)], flags=zmq.NOBLOCK
            )

    def _parse_data_message(
        self,
        frames: list[bytes],
    ) -> tuple[bytes, tuple[str, int, int], bytes, dict[str, Any]]:
        if len(frames) != 4:
            raise ValueError(f"Expected 4 ZeroMQ frames, received {len(frames)}.")
        identity, kind, raw_metadata, body = frames
        if kind != _DATA:
            raise ValueError(f"Unsupported ZeroMQ sparse refit message {kind!r}.")
        metadata = json.loads(raw_metadata)
        if metadata.get("protocol") != _PROTOCOL:
            raise ValueError("Unsupported ZeroMQ sparse refit protocol.")
        if self._token is not None and metadata.get("api_key") != self._token:
            raise PermissionError("ZeroMQ sparse refit producer authentication failed.")
        transfer_id = str(metadata["transfer_id"])
        producer_id = int(metadata["producer_id"])
        payload_id = int(metadata["payload_id"])
        checksum = str(metadata["checksum"])
        if not transfer_id or producer_id < 0 or payload_id < 0:
            raise ValueError("Invalid ZeroMQ sparse refit payload identity.")
        return identity, (transfer_id, producer_id, payload_id), body, metadata

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        payload_executor = ThreadPoolExecutor(
            max_workers=self._payload_workers,
            thread_name_prefix="nrl-zmq-payload",
        )
        http_executor = ThreadPoolExecutor(
            max_workers=self._fanout_workers,
            thread_name_prefix="nrl-zmq-fanout",
        )
        pending: dict[Any, tuple[bytes, tuple[str, int, int]]] = {}
        try:
            _configure_socket(socket, 16)
            socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            socket.bind(self._bind_address)
            self._endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self._ready.set()

            while not self._stop.is_set() or pending:
                if (
                    not self._stop.is_set()
                    and len(pending) < self._payload_workers
                    and socket.poll(10, zmq.POLLIN)
                ):
                    frames = socket.recv_multipart()
                    identity = frames[0] if frames else b""
                    try:
                        identity, key, body, metadata = self._parse_data_message(frames)
                        future = payload_executor.submit(
                            self._fanout,
                            body,
                            metadata,
                            http_executor,
                        )
                        pending[future] = (identity, key)
                    except Exception as exc:
                        self._send_reply(
                            socket,
                            identity,
                            _NACK,
                            {"ok": False, "error": str(exc)},
                        )

                for future, (identity, key) in list(pending.items()):
                    if not future.done():
                        continue
                    transfer_id, producer_id, payload_id = key
                    reply: dict[str, Any] = {
                        "ok": True,
                        "transfer_id": transfer_id,
                        "producer_id": producer_id,
                        "payload_id": payload_id,
                    }
                    try:
                        reply.update(future.result())
                    except Exception as exc:
                        reply.update(ok=False, error=str(exc))
                        kind = _NACK
                    else:
                        kind = _ACK
                    self._send_reply(socket, identity, kind, reply)
                    del pending[future]
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            payload_executor.shutdown(wait=True, cancel_futures=True)
            http_executor.shutdown(wait=True, cancel_futures=True)
            socket.close()
            context.term()


def stream_sparse_delta_payloads_via_zmq(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker,
    refit_targets: Sequence[str],
    transfer_id: str,
    api_key_env_var: str | None,
    timeout_s: float,
    shard_rank: int,
    shard_count: int,
    partition: SparsePartitionMode = "chunks",
) -> dict[str, int]:
    addresses = [address.strip() for address in refit_targets if address.strip()]
    if not addresses:
        raise ValueError("At least one ZeroMQ sparse refit address is required.")
    address = addresses[shard_rank % len(addresses)]
    api_key = vllm_refit_api_key(api_key_env_var)

    def send_payload(body: bytes, payload_id: int) -> dict[str, Any]:
        clients = getattr(_ZMQ_LOCAL, "clients", None)
        if clients is None:
            clients = {}
            _ZMQ_LOCAL.clients = clients
        client_key = (address, shard_rank, api_key)
        client = clients.get(client_key)
        if client is None:
            client = ZmqSparseRefitClient(
                address,
                timeout_s=timeout_s,
                producer_id=shard_rank,
                api_key=api_key,
            )
            clients[client_key] = client
        started = time.perf_counter()
        reply = client.send_payload(
            transfer_id=transfer_id,
            payload_id=payload_id,
            checksum=sparse_payload_checksum(body),
            body=body,
        )
        return {
            "zmq_send_s": time.perf_counter() - started,
            "receiver": reply,
        }

    return stream_sparse_delta_payloads(
        iterator,
        delta_tracker=delta_tracker,
        transport="zmq",
        send_payload=send_payload,
        transfer_workers=refit_env_int("NRL_REFIT_ZMQ_SEND_WORKERS", default=4),
        shard_rank=shard_rank,
        shard_count=shard_count,
        partition=partition,
    )
