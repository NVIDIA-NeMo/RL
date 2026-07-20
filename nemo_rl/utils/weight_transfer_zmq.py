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

import hmac
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import zmq

from nemo_rl.models.generation.vllm.config import VllmRefitTuningConfig
from nemo_rl.utils.weight_transfer_http import (
    merge_vllm_refit_metrics,
    vllm_refit_api_key,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    NamedTensor,
)
from nemo_rl.utils.weight_transfer_stream import (
    SparseRefitTransport,
    sparse_payload_checksum,
    stream_sparse_delta_payloads,
)

_PROTOCOL = "nemo-rl-sparse-zmq-v1"
_DATA = b"DATA"
_ACK = b"ACK"
_NACK = b"NACK"


@dataclass
class _RelayTransfer:
    checksums: dict[tuple[int, int], str] = field(default_factory=dict)
    futures: list[Future[dict[str, Any]]] = field(default_factory=list)


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
        retries: int,
        api_key: str | None = None,
    ) -> None:
        self._address = address
        self._timeout_ms = max(1, int(timeout_s * 1000))
        self._producer_id = producer_id
        self._retries = retries
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
        verification_candidates: int,
        body: bytes,
        relay_root: str | None = None,
        producer_id: int | None = None,
    ) -> dict[str, Any]:
        producer_id = self._producer_id if producer_id is None else producer_id
        metadata = {
            "protocol": _PROTOCOL,
            "transfer_id": transfer_id,
            "producer_id": producer_id,
            "payload_id": payload_id,
            "checksum": checksum,
            "verification_candidates": verification_candidates,
        }
        if relay_root is not None:
            metadata["relay_root"] = relay_root
        if self._api_key is not None:
            metadata["api_key"] = self._api_key
        metadata_frame = _json_bytes(metadata)
        for attempt in range(self._retries + 1):
            try:
                self._socket.send_multipart(
                    [_DATA, metadata_frame, body],
                    copy=False,
                )
            except zmq.Again:
                if attempt == self._retries:
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
                if reply_key != (transfer_id, producer_id, payload_id):
                    continue
                if kind == _ACK and reply.get("ok") is True:
                    return reply
                raise RuntimeError(f"ZeroMQ sparse refit rejected payload: {reply}")

            if attempt < self._retries:
                time.sleep(min(0.05 * 2**attempt, 0.5))

        raise TimeoutError(
            f"Timed out sending sparse refit payload {payload_id} to {self._address}."
        )

    def close(self) -> None:
        self._socket.close()


class ZmqSparseRefitServer:
    """Bounded ROUTER relay that applies locally and fans out through a tree."""

    def __init__(
        self,
        apply_payload: Callable[[bytes, Mapping[str, Any]], dict[str, Any]],
        *,
        bind_address: str,
        api_key_env_var: str | None,
        timeout_s: float,
        tuning: VllmRefitTuningConfig,
    ) -> None:
        self._apply_payload = apply_payload
        self._bind_address = bind_address
        self._token = vllm_refit_api_key(api_key_env_var)
        self._timeout_s = timeout_s
        self._retries = tuning.zmq_retries
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._endpoint: str | None = None
        self._error: Exception | None = None
        self._payload_workers = tuning.zmq_relay_payload_workers
        self._transfer_lock = threading.Lock()
        self._transfer_condition = threading.Condition(self._transfer_lock)
        self._transfers: dict[str, _RelayTransfer] = {}
        self._flush_results: dict[str, Future[dict[str, Any]]] = {}
        self._tree: tuple[tuple[str, ...], int] | None = None
        self._forward_local = threading.local()
        self._forward_clients: list[ZmqSparseRefitClient] = []
        self._payload_executor = ThreadPoolExecutor(
            max_workers=self._payload_workers,
            thread_name_prefix="nrl-zmq-payload",
        )
        self._forward_executor = ThreadPoolExecutor(
            max_workers=tuning.zmq_relay_forward_workers,
            thread_name_prefix="nrl-zmq-forward",
        )

    def configure_tree(
        self,
        relay_addresses: Sequence[str],
        *,
        own_address: str,
    ) -> None:
        addresses = tuple(dict.fromkeys(relay_addresses))
        self._tree = (addresses, addresses.index(own_address))

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

    def flush(self, transfer_id: str, expected_payloads: int = 0) -> dict[str, Any]:
        """Wait for every staged fanout belonging to one transfer."""
        with self._transfer_condition:
            completion = self._flush_results.get(transfer_id)
            if completion is None:
                ready = self._transfer_condition.wait_for(
                    lambda: (
                        len(
                            self._transfers.get(transfer_id, _RelayTransfer()).checksums
                        )
                        >= expected_payloads
                    ),
                    timeout=self._timeout_s,
                )
                if not ready:
                    raise TimeoutError(
                        f"Timed out waiting for {expected_payloads} ZeroMQ payloads "
                        f"for transfer {transfer_id}."
                    )
                completion = self._flush_results.get(transfer_id)
            if completion is None:
                completion = Future()
                self._flush_results[transfer_id] = completion
                staged = self._transfers.pop(transfer_id, _RelayTransfer())
            else:
                staged = None
        if staged is None:
            return completion.result()

        try:
            started = time.perf_counter()
            results: list[dict[str, Any]] = []
            first_error: Exception | None = None
            for future in staged.futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                raise RuntimeError(
                    f"ZeroMQ relay fanout failed for transfer {transfer_id}: "
                    f"{first_error}"
                ) from first_error

            merged = merge_vllm_refit_metrics({}, results, maximum=False)
            merged.update(
                ok=True,
                payloads=len(staged.checksums),
                receiver_relay_flush_s=time.perf_counter() - started,
            )
        except Exception as exc:
            completion.set_exception(exc)
            raise
        completion.set_result(merged)
        return merged

    def _fanout(
        self,
        body: bytes,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = self._apply_payload(body, metadata)
        result["receiver_relay_fanout_s"] = time.perf_counter() - started
        return result

    def _forward(
        self,
        body: bytes,
        metadata: Mapping[str, Any],
        address: str,
        relay_root: str,
    ) -> dict[str, Any]:
        clients = getattr(self._forward_local, "clients", None)
        if clients is None:
            clients = {}
            self._forward_local.clients = clients
        client = clients.get(address)
        if client is None:
            client = ZmqSparseRefitClient(
                address,
                timeout_s=self._timeout_s,
                producer_id=0,
                retries=self._retries,
                api_key=self._token,
            )
            clients[address] = client
            with self._transfer_lock:
                self._forward_clients.append(client)
        started = time.perf_counter()
        client.send_payload(
            transfer_id=str(metadata["transfer_id"]),
            payload_id=int(metadata["payload_id"]),
            checksum=str(metadata["checksum"]),
            verification_candidates=int(metadata["verification_candidates"]),
            body=body,
            relay_root=relay_root,
            producer_id=int(metadata["producer_id"]),
        )
        return {"receiver_relay_forward_s": time.perf_counter() - started}

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
        supplied_token = metadata.get("api_key")
        if self._token is not None and (
            not isinstance(supplied_token, str)
            or not hmac.compare_digest(self._token, supplied_token)
        ):
            raise PermissionError("ZeroMQ sparse refit producer authentication failed.")
        transfer_id = str(metadata["transfer_id"])
        producer_id = int(metadata["producer_id"])
        payload_id = int(metadata["payload_id"])
        checksum = str(metadata["checksum"])
        verification_candidates = int(metadata["verification_candidates"])
        if (
            not transfer_id
            or producer_id < 0
            or payload_id < 0
            or not checksum
            or verification_candidates < 0
        ):
            raise ValueError("Invalid ZeroMQ sparse refit payload identity.")
        relay_root = metadata.get("relay_root")
        if relay_root is not None and (
            self._tree is None or relay_root not in self._tree[0]
        ):
            raise ValueError("Invalid ZeroMQ relay root.")
        return identity, (transfer_id, producer_id, payload_id), body, metadata

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        try:
            _configure_socket(socket, 16)
            socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            socket.bind(self._bind_address)
            self._endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self._ready.set()

            while not self._stop.is_set():
                if socket.poll(10, zmq.POLLIN):
                    frames = socket.recv_multipart()
                    identity = frames[0] if frames else b""
                    try:
                        identity, key, body, metadata = self._parse_data_message(frames)
                        transfer_id, producer_id, payload_id = key
                        payload_key = (producer_id, payload_id)
                        checksum = str(metadata["checksum"])
                        with self._transfer_lock:
                            if transfer_id in self._flush_results:
                                raise RuntimeError(
                                    "ZeroMQ sparse refit transfer is already flushed."
                                )
                            staged = self._transfers.setdefault(
                                transfer_id, _RelayTransfer()
                            )
                            previous = staged.checksums.get(payload_key)
                            if previous is not None and previous != checksum:
                                raise ValueError(
                                    "Conflicting ZeroMQ sparse refit payload checksum."
                                )
                            if previous is None:
                                staged.checksums[payload_key] = checksum
                                staged.futures.append(
                                    self._payload_executor.submit(
                                        self._fanout, body, metadata
                                    )
                                )
                                if self._tree is not None:
                                    addresses, own_index = self._tree
                                    relay_root = str(
                                        metadata.get("relay_root", addresses[own_index])
                                    )
                                    root_index = addresses.index(relay_root)
                                    node_index = (own_index - root_index) % len(
                                        addresses
                                    )
                                    for child_index in (
                                        2 * node_index + 1,
                                        2 * node_index + 2,
                                    ):
                                        if child_index < len(addresses):
                                            child = addresses[
                                                (root_index + child_index)
                                                % len(addresses)
                                            ]
                                            staged.futures.append(
                                                self._forward_executor.submit(
                                                    self._forward,
                                                    body,
                                                    metadata,
                                                    child,
                                                    relay_root,
                                                )
                                            )
                                self._transfer_condition.notify_all()
                        self._send_reply(
                            socket,
                            identity,
                            _ACK,
                            {
                                "ok": True,
                                "staged": True,
                                "transfer_id": transfer_id,
                                "producer_id": producer_id,
                                "payload_id": payload_id,
                            },
                        )
                    except Exception as exc:
                        self._send_reply(
                            socket,
                            identity,
                            _NACK,
                            {"ok": False, "error": str(exc)},
                        )
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._payload_executor.shutdown(wait=True, cancel_futures=True)
            self._forward_executor.shutdown(wait=True, cancel_futures=True)
            for client in self._forward_clients:
                client.close()
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
) -> dict[str, int]:
    addresses = [address.strip() for address in refit_targets if address.strip()]
    if not addresses:
        raise ValueError("At least one ZeroMQ sparse refit address is required.")
    address = addresses[shard_rank % len(addresses)]
    tuning = delta_tracker.refit_config.tuning
    api_key = vllm_refit_api_key(api_key_env_var)
    local = threading.local()

    def send(
        body: bytes, payload_id: int, verification_candidates: int
    ) -> dict[str, Any]:
        client = getattr(local, "client", None)
        if client is None:
            client = ZmqSparseRefitClient(
                address,
                timeout_s=timeout_s,
                producer_id=shard_rank,
                retries=tuning.zmq_retries,
                api_key=api_key,
            )
            local.client = client
        started = time.perf_counter()
        reply = client.send_payload(
            transfer_id=transfer_id,
            payload_id=payload_id,
            checksum=sparse_payload_checksum(body),
            verification_candidates=verification_candidates,
            body=body,
        )
        return {
            "zmq_send_s": time.perf_counter() - started,
            "receiver": reply,
        }

    def cleanup() -> None:
        client = getattr(local, "client", None)
        if client is not None:
            client.close()
            del local.client

    return stream_sparse_delta_payloads(
        iterator,
        delta_tracker=delta_tracker,
        transport=SparseRefitTransport(
            name="zmq",
            transfer_workers=tuning.transfer_workers["zmq"],
            send=send,
            cleanup=cleanup,
        ),
        shard_rank=shard_rank,
        shard_count=shard_count,
    )
