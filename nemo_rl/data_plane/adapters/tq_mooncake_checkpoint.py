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
"""Opt-in, owner-distributed Mooncake checkpoints for TransferQueue.

Normal Mooncake PUTs remain memory-only.  At an explicit TQ checkpoint, a
small Ray registry discovers the already-running Mooncake clients and a local
ZeroMQ endpoint asks each selected memory owner to copy its own objects to a
unique Lustre shard.  The TQ storage-manager process coordinates metadata but
never receives the checkpoint payload.

Restore reverses the process: the current Mooncake clients read size-balanced
sets of durable objects and upsert them into their own preferred segments
before TQ restores controller metadata.  Saved client identities are not
reused across restarts.

The caller must keep writers and clears quiescent while ``tq.save_checkpoint``
captures the controller and reads its referenced objects.  All intended
restore clients must connect before ``tq.load_checkpoint`` is called.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import mmap
import os
import pickle
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import zmq

_PLUGIN_MARKER = "_nemo_rl_mooncake_checkpoint_v3"
_PROTOCOL = "nemo-rl-tq-mooncake-checkpoint-v1"
_REGISTRY_NAME = "NeMoRLMooncakeCheckpointRegistry"
_REGISTRY_NAMESPACE = "transfer_queue"
_STORAGE_DIR = "mooncake_storage"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_VERSION = 3
_UNREGISTER_ATTEMPTS = 3
_DEFAULT_TIMEOUT_S = 1800.0
_DEFAULT_MAX_PARALLEL = 64

# An mmap must outlive its Mooncake registration. On a persistent unregister
# failure, retain it until process teardown instead of letting Python unmap
# memory that Mooncake or the NIC may still reference.
_QUARANTINED_BUFFERS: list[mmap.mmap] = []

_REGISTRY_ACTOR_CLASS: Any = None


def _checkpoint_settings(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("MooncakeStore config must be a mapping")
    checkpoint = config.get("checkpoint") or {}
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MooncakeStore.checkpoint must be a mapping")
    return checkpoint


def _checkpoint_enabled(config: Any) -> bool:
    return _checkpoint_settings(config).get("enabled") is True


def _checkpoint_timeout_s(config: Any) -> float:
    value = _checkpoint_settings(config).get("timeout_s", _DEFAULT_TIMEOUT_S)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError("MooncakeStore.checkpoint.timeout_s must be positive")
    return float(value)


def _checkpoint_max_parallel(config: Any) -> int:
    value = _checkpoint_settings(config).get("max_parallel", _DEFAULT_MAX_PARALLEL)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("MooncakeStore.checkpoint.max_parallel must be positive")
    return value


def _storage_layout(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("MooncakeStore config must be a mapping")
    return {
        "use_gdr": bool(config["use_gdr"]),
        "gdr_staging_buffer_mb": int(config["gdr_staging_buffer_mb"]),
    }


def _validate_metadata_mode(manager: Any) -> None:
    metadata_server = str(
        getattr(manager.storage_client, "metadata_server", "")
    ).strip()
    if metadata_server.upper() == "P2PHANDSHAKE":
        raise NotImplementedError(
            "Owner-distributed Mooncake checkpoints do not support "
            "P2PHANDSHAKE: Mooncake does not expose the local transfer endpoint "
            "needed to match this process to replica descriptors"
        )


def _validate_checkpoint_runtime(manager: Any) -> None:
    _validate_metadata_mode(manager)
    replica_config = manager.storage_client.replica_config
    if getattr(replica_config, "with_hard_pin", None) is not True:
        raise NotImplementedError(
            "Owner-distributed Mooncake checkpoints require hard-pinned memory replicas"
        )
    offload = manager.config.get("offload")
    if isinstance(offload, Mapping) and offload.get("enabled") is True:
        raise NotImplementedError(
            "Owner-distributed Mooncake checkpoints do not support enabled "
            "Mooncake offload"
        )


def _physical_keys(controller_state: Mapping[str, Any]) -> list[str]:
    """Return every produced Mooncake key referenced by a TQ controller cut."""
    partitions = controller_state.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("TQ controller checkpoint has no partitions mapping")

    keys: set[str] = set()
    for partition in partitions.values():
        indexes = getattr(partition, "global_indexes", None)
        fields = getattr(partition, "field_name_mapping", None)
        produced = getattr(partition, "production_status", None)
        backend_meta = getattr(partition, "field_custom_backend_meta", {})
        if not isinstance(indexes, (set, list, tuple)):
            raise ValueError("TQ partition has malformed global_indexes")
        if not isinstance(fields, Mapping) or produced is None:
            raise ValueError("TQ partition has malformed field metadata")
        if not isinstance(backend_meta, Mapping):
            raise ValueError("TQ partition has malformed backend metadata")

        for global_index in indexes:
            per_index_meta = backend_meta.get(global_index) or {}
            if not isinstance(per_index_meta, Mapping):
                raise ValueError(
                    "TQ partition has malformed per-index backend metadata"
                )
            for field_name, column in fields.items():
                status = produced[global_index, column]
                item = getattr(status, "item", None)
                if callable(item):
                    status = item()
                if status != 1:
                    continue

                key = f"{global_index}@{field_name}"
                field_meta = per_index_meta.get(field_name)
                if isinstance(field_meta, Mapping) and "n_chunks" in field_meta:
                    n_chunks = field_meta["n_chunks"]
                    if (
                        isinstance(n_chunks, bool)
                        or not isinstance(n_chunks, int)
                        or n_chunks <= 0
                    ):
                        raise ValueError(f"Invalid n_chunks for Mooncake key {key!r}")
                    keys.update(f"{key}:c{i}" for i in range(n_chunks))
                else:
                    keys.add(key)
    return sorted(keys)


@dataclass(frozen=True)
class _StoredObject:
    key: str
    size: int


@dataclass(frozen=True)
class _ParticipantInfo:
    participant_id: str
    incarnation: str
    controller_session: str
    control_endpoint: str
    segment_name: str
    transport_endpoint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> _ParticipantInfo:
        fields = {
            name: value.get(name)
            for name in (
                "participant_id",
                "incarnation",
                "controller_session",
                "control_endpoint",
                "segment_name",
                "transport_endpoint",
            )
        }
        if any(not isinstance(field, str) or not field for field in fields.values()):
            raise ValueError(f"Malformed Mooncake checkpoint participant: {value!r}")
        return cls(**fields)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _ParticipantRequest:
    participant: _ParticipantInfo
    body: dict[str, Any]


@dataclass(frozen=True)
class _ManifestObject:
    key: str
    shard: str
    offset: int
    size: int
    sha256: str
    saved_owner: str


def _stored_objects(store: Any, keys: list[str]) -> list[_StoredObject]:
    """Resolve the authoritative raw byte size for every checkpoint key."""
    objects: list[_StoredObject] = []
    for key in keys:
        size = store.get_size(key)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RuntimeError(
                f"Mooncake could not resolve a positive size for {key!r}: {size!r}"
            )
        objects.append(_StoredObject(key=key, size=size))
    return objects


def _controller_path(checkpoint_dir: Path) -> Path:
    # Optional dependency; importing it at module load would eagerly load TQ.
    from transfer_queue import interface as tq_interface

    return checkpoint_dir / tq_interface._CONTROLLER_FILE


def _controller_keys(checkpoint_dir: Path) -> list[str]:
    path = _controller_path(checkpoint_dir)
    with path.open("rb") as controller_file:
        state = pickle.load(controller_file)
    if not isinstance(state, Mapping):
        raise ValueError("TQ controller checkpoint must contain a mapping")
    return _physical_keys(state)


@dataclass
class _CheckpointBuffer:
    payload: mmap.mmap
    pointer: int
    quarantined: bool = False

    @classmethod
    def allocate(cls, size: int) -> _CheckpointBuffer:
        payload = mmap.mmap(-1, size)
        # ctypes' stub rejects the object returned by its own from_buffer.
        # pyrefly: ignore  # bad-argument-type
        pointer = ctypes.addressof(ctypes.c_char.from_buffer(payload))
        return cls(payload=payload, pointer=pointer)

    def close(self) -> None:
        if not self.quarantined:
            self.payload.close()


def _unregister_or_quarantine(
    store: Any, buffer: _CheckpointBuffer, *, label: str
) -> None:
    last_error: BaseException | None = None
    for _ in range(_UNREGISTER_ATTEMPTS):
        try:
            result = store.unregister_buffer(buffer.pointer)
        except BaseException as error:
            last_error = error
            continue
        if result == 0:
            return
        last_error = RuntimeError(f"status {result}")

    buffer.quarantined = True
    _QUARANTINED_BUFFERS.append(buffer.payload)
    raise RuntimeError(
        f"Mooncake buffer cleanup failed for {label} after "
        f"{_UNREGISTER_ATTEMPTS} attempts; the mapped buffer was retained"
    ) from last_error


@contextmanager
def _registered_buffer(
    store: Any, buffer: _CheckpointBuffer, *, size: int, label: str
) -> Iterator[int]:
    try:
        result = store.register_buffer(buffer.pointer, size)
    except BaseException as error:
        buffer.quarantined = True
        _QUARANTINED_BUFFERS.append(buffer.payload)
        error.add_note(
            f"The mapped buffer for {label} was retained because registration "
            "raised before its outcome could be determined"
        )
        raise
    if result != 0:
        raise RuntimeError(f"Mooncake buffer registration failed for {label}: {result}")
    try:
        yield buffer.pointer
    except BaseException as operation_error:
        try:
            _unregister_or_quarantine(store, buffer, label=label)
        except BaseException as cleanup_error:
            operation_error.add_note(str(cleanup_error))
        raise
    else:
        _unregister_or_quarantine(store, buffer, label=label)


def _write_buffer(output: Any, buffer: mmap.mmap, size: int, *, label: str) -> None:
    view = memoryview(buffer)
    try:
        offset = 0
        while offset < size:
            written = output.write(view[offset:size])
            if written is None or written <= 0:
                raise OSError(f"Short checkpoint write for {label}")
            offset += written
    finally:
        view.release()


def _read_buffer(source: Any, buffer: mmap.mmap, size: int, *, label: str) -> None:
    view = memoryview(buffer)
    try:
        offset = 0
        while offset < size:
            read = source.readinto(view[offset:size])
            if read is None or read <= 0:
                raise OSError(f"Short checkpoint read for {label}")
            offset += read
    finally:
        view.release()


def _save_object(store: Any, obj: _StoredObject, output: Any) -> str:
    """GET one local Mooncake object and append it to an owner's shard."""
    buffer = _CheckpointBuffer.allocate(obj.size)
    try:
        with _registered_buffer(
            store, buffer, size=obj.size, label=f"Mooncake key {obj.key!r}"
        ) as pointer:
            result = store.get_into(obj.key, pointer, obj.size)
            if result != obj.size:
                raise RuntimeError(
                    f"Mooncake checkpoint GET failed for {obj.key!r}: "
                    f"expected {obj.size} bytes, got {result}"
                )

        digest = hashlib.sha256(buffer.payload).hexdigest()
        _write_buffer(output, buffer.payload, obj.size, label=obj.key)
        return digest
    finally:
        buffer.close()


def _restore_object(
    store: Any,
    entry: _ManifestObject,
    replica_config: Any,
    payload_file: Any,
) -> None:
    """Read one durable object and upsert it into the current client."""
    buffer = _CheckpointBuffer.allocate(entry.size)
    try:
        payload_file.seek(entry.offset)
        _read_buffer(payload_file, buffer.payload, entry.size, label=entry.key)
        if hashlib.sha256(buffer.payload).hexdigest() != entry.sha256:
            raise ValueError(f"Corrupt Mooncake checkpoint payload for {entry.key!r}")

        # Register anonymous memory rather than a Lustre-backed mmap. RDMA
        # registration of network-filesystem mappings is not a Mooncake contract.
        with _registered_buffer(
            store, buffer, size=entry.size, label=f"Mooncake key {entry.key!r}"
        ) as pointer:
            result = store.upsert_from(entry.key, pointer, entry.size, replica_config)
            if result != 0:
                raise RuntimeError(
                    f"Mooncake restore failed for {entry.key!r}: {result}"
                )
    finally:
        buffer.close()


def _status_is_complete(status: Any) -> bool:
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name == "COMPLETE"
    return str(status).rsplit(".", 1)[-1] == "COMPLETE"


def _complete_memory_replicas(
    descriptors: Any,
) -> list[tuple[str, int]]:
    replicas: list[tuple[str, int]] = []
    if not isinstance(descriptors, (list, tuple)):
        return replicas
    for descriptor in descriptors:
        is_memory = getattr(descriptor, "is_memory_replica", None)
        if not callable(is_memory) or not is_memory():
            continue
        if not _status_is_complete(getattr(descriptor, "status", None)):
            continue
        memory = descriptor.get_memory_descriptor()
        buffer = memory.buffer_descriptor
        endpoint = getattr(buffer, "transport_endpoint", None)
        size = getattr(buffer, "size", None)
        if isinstance(endpoint, str) and endpoint and isinstance(size, int):
            replicas.append((endpoint, size))
    return replicas


class _MooncakeCheckpointRegistry:
    """Small named Ray actor containing control endpoints, never payloads."""

    def __init__(self) -> None:
        self._participants: dict[str, dict[str, str]] = {}

    def register(self, participant: dict[str, str]) -> None:
        info = _ParticipantInfo.from_mapping(participant)
        self._participants[info.participant_id] = asdict(info)

    def unregister(self, participant_id: str, incarnation: str) -> None:
        current = self._participants.get(participant_id)
        if current is not None and current.get("incarnation") == incarnation:
            self._participants.pop(participant_id, None)

    def participants(self, controller_session: str) -> list[dict[str, str]]:
        return [
            participant
            for participant in self._participants.values()
            if participant.get("controller_session") == controller_session
        ]


def _registry_actor() -> Any:
    import ray

    global _REGISTRY_ACTOR_CLASS
    if _REGISTRY_ACTOR_CLASS is None:
        _REGISTRY_ACTOR_CLASS = ray.remote(num_cpus=0)(_MooncakeCheckpointRegistry)
    return _REGISTRY_ACTOR_CLASS.options(
        name=_REGISTRY_NAME,
        namespace=_REGISTRY_NAMESPACE,
        get_if_exists=True,
    ).remote()


def _controller_session(manager: Any) -> str:
    """Stable identity for one live TQ controller, including its endpoints."""
    info = manager.controller_info
    controller_id = getattr(info, "id", None)
    ip = getattr(info, "ip", None)
    ports = getattr(info, "ports", None)
    if (
        not isinstance(controller_id, str)
        or not controller_id
        or not isinstance(ip, str)
        or not ip
        or not isinstance(ports, Mapping)
    ):
        raise RuntimeError("Mooncake checkpoint could not identify TQ controller")
    normalized_ports: list[tuple[str, int]] = []
    for name, port in ports.items():
        if (
            not isinstance(name, str)
            or isinstance(port, bool)
            or not isinstance(port, int)
        ):
            raise RuntimeError("TQ controller has malformed endpoint metadata")
        normalized_ports.append((name, port))
    payload = json.dumps(
        {
            "id": controller_id,
            "ip": ip,
            "ports": sorted(normalized_ports),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _request_body(manager: Any, operation: str, **payload: Any) -> dict[str, Any]:
    return {
        "protocol": _PROTOCOL,
        "controller_session": _controller_session(manager),
        "operation": operation,
        "request_id": uuid.uuid4().hex,
        **payload,
    }


def _safe_shard_name(value: Any) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"Invalid Mooncake checkpoint shard name: {value!r}")
    if not value.startswith("part-") or not value.endswith(".bin"):
        raise ValueError(f"Invalid Mooncake checkpoint shard name: {value!r}")
    return value


def _request_id(body: Mapping[str, Any]) -> str:
    value = body.get("request_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Mooncake checkpoint request has no request_id")
    return value


def _parse_stored_objects(values: Any) -> list[_StoredObject]:
    if not isinstance(values, list):
        raise ValueError("Mooncake checkpoint request has no object list")
    objects: list[_StoredObject] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Malformed Mooncake checkpoint object request")
        key = value.get("key")
        size = value.get("size")
        if (
            not isinstance(key, str)
            or not key
            or key in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError(f"Malformed Mooncake checkpoint object: {value!r}")
        seen.add(key)
        objects.append(_StoredObject(key, size))
    return objects


def _manifest_object_from_mapping(value: Mapping[str, Any]) -> _ManifestObject:
    key = value.get("key")
    shard = value.get("shard")
    offset = value.get("offset")
    size = value.get("size")
    digest = value.get("sha256")
    saved_owner = value.get("saved_owner")
    if not isinstance(key, str) or not key:
        raise ValueError(f"Malformed Mooncake checkpoint object: {value!r}")
    shard = _safe_shard_name(shard)
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or not isinstance(saved_owner, str)
        or not saved_owner
    ):
        raise ValueError(f"Malformed Mooncake checkpoint entry for {key!r}")
    return _ManifestObject(key, shard, offset, size, digest, saved_owner)


def _local_replica_config(manager: Any, segment_name: str) -> Any:
    from mooncake.store import ReplicateConfig

    source = manager.storage_client.replica_config
    config = ReplicateConfig()
    for name in (
        "replica_num",
        "with_soft_pin",
        "with_hard_pin",
        "prefer_alloc_in_same_node",
        "data_type",
    ):
        if hasattr(source, name):
            setattr(config, name, getattr(source, name))
    # `preferred_segment` is tried first and does not require a vector whose
    # length equals replica_num.  TQ's pinned Mooncake client uses replica_num=1.
    config.preferred_segment = segment_name
    return config


class _CheckpointParticipant:
    """Executes checkpoint file I/O inside one existing Mooncake client."""

    def __init__(self, manager: Any) -> None:
        _validate_checkpoint_runtime(manager)
        self._manager = manager
        self._store = manager.storage_client._store
        segment_name = self._store.get_hostname()
        if not isinstance(segment_name, str) or not segment_name:
            raise RuntimeError("Mooncake client did not expose its segment name")

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._registry: Any = None
        self.info = _ParticipantInfo(
            participant_id=str(manager.storage_manager_id),
            incarnation=uuid.uuid4().hex,
            controller_session=_controller_session(manager),
            control_endpoint="",
            segment_name=segment_name,
            # In HTTP/etcd metadata mode Mooncake writes local_hostname into
            # every mounted memory descriptor's transport_endpoint.
            transport_endpoint=segment_name,
        )

    def start_and_register(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.info.participant_id}-checkpoint",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self.close()
            raise RuntimeError("Timed out starting Mooncake checkpoint endpoint")
        if self._error is not None:
            error = self._error
            self.close()
            raise RuntimeError(
                "Failed to start Mooncake checkpoint endpoint"
            ) from error

        import ray

        self._registry = _registry_actor()
        ray.get(
            self._registry.register.remote(asdict(self.info)),
            timeout=30.0,
        )

    def close(self) -> None:
        if self._registry is not None:
            with suppress(Exception):
                import ray

                ray.get(
                    self._registry.unregister.remote(
                        self.info.participant_id, self.info.incarnation
                    ),
                    timeout=5.0,
                )
            self._registry = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                raise RuntimeError(
                    "Mooncake checkpoint endpoint is still executing; refusing "
                    "to release its storage client"
                )
            self._thread = None

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, 8)
            socket.setsockopt(zmq.RCVHWM, 8)
            socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            host = self.info.segment_name.rsplit(":", 1)[0]
            socket.bind(f"tcp://{host}:*")
            endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self.info = _ParticipantInfo(
                participant_id=self.info.participant_id,
                incarnation=self.info.incarnation,
                controller_session=self.info.controller_session,
                control_endpoint=endpoint,
                segment_name=self.info.segment_name,
                transport_endpoint=self.info.transport_endpoint,
            )
            self._ready.set()

            while not self._stop.is_set():
                if not socket.poll(100, zmq.POLLIN):
                    continue
                frames = socket.recv_multipart()
                identity = frames[0] if frames else b""
                request_id = ""
                try:
                    if len(frames) != 2:
                        raise ValueError("Malformed Mooncake checkpoint request")
                    body = json.loads(frames[1])
                    if not isinstance(body, Mapping):
                        raise ValueError("Malformed Mooncake checkpoint request")
                    request_id = _request_id(body)
                    response = self._dispatch(body)
                except Exception as error:
                    response = {
                        "ok": False,
                        "request_id": request_id,
                        "error": str(error),
                    }
                try:
                    socket.send_multipart(
                        [
                            identity,
                            json.dumps(
                                response, separators=(",", ":"), sort_keys=True
                            ).encode(),
                        ]
                    )
                except zmq.ZMQError:
                    # A coordinator can time out and disconnect while the local
                    # participant is still completing checkpoint I/O. Keep the
                    # endpoint alive so a later checkpoint can still use it.
                    continue
        except Exception as error:
            self._error = error
            self._ready.set()
        finally:
            socket.close(linger=0)
            context.term()

    def _dispatch(self, body: Mapping[str, Any]) -> dict[str, Any]:
        if body.get("protocol") != _PROTOCOL:
            raise ValueError("Unsupported Mooncake checkpoint protocol")
        if body.get("controller_session") != self.info.controller_session:
            raise ValueError("Mooncake checkpoint controller session mismatch")
        request_id = _request_id(body)
        operation = body.get("operation")
        if operation == "PING":
            return {
                "ok": True,
                "request_id": request_id,
                "participant": asdict(self.info),
            }
        if operation == "SAVE_SHARD":
            return self._save_shard(body)
        if operation == "LOAD_OBJECTS":
            return self._load_objects(body)
        raise ValueError(f"Unsupported Mooncake checkpoint operation: {operation!r}")

    def _checkpoint_root(self, body: Mapping[str, Any]) -> Path:
        value = body.get("checkpoint_root")
        if not isinstance(value, str) or not value:
            raise ValueError("Mooncake checkpoint request has no checkpoint_root")
        root = Path(value)
        if not root.is_absolute():
            raise ValueError("Mooncake checkpoint_root must be absolute")
        return root

    def _verify_local_owner(self, obj: _StoredObject) -> None:
        descriptors = self._store.get_replica_desc(obj.key)
        replicas = _complete_memory_replicas(descriptors)
        if (self.info.transport_endpoint, obj.size) not in replicas:
            raise RuntimeError(
                f"Mooncake key {obj.key!r} no longer has a complete memory "
                f"replica owned by {self.info.transport_endpoint!r}"
            )

    def _save_shard(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request_id = _request_id(body)
        root = self._checkpoint_root(body)
        shard = _safe_shard_name(body.get("shard_name"))
        objects = _parse_stored_objects(body.get("objects"))
        storage_dir = root / _STORAGE_DIR
        if not storage_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint storage directory is missing: {storage_dir}"
            )
        target = storage_dir / shard
        if target.exists():
            raise FileExistsError(f"Mooncake checkpoint shard already exists: {target}")
        partial = storage_dir / f".{shard}.{request_id}.partial"

        entries: list[dict[str, Any]] = []
        offset = 0
        with partial.open("xb") as output:
            for obj in objects:
                self._verify_local_owner(obj)
                digest = _save_object(self._store, obj, output)
                entries.append(
                    {
                        "key": obj.key,
                        "shard": shard,
                        "offset": offset,
                        "size": obj.size,
                        "sha256": digest,
                        "saved_owner": self.info.transport_endpoint,
                    }
                )
                offset += obj.size
            output.flush()
            os.fsync(output.fileno())
        partial.rename(target)
        return {
            "ok": True,
            "request_id": request_id,
            "participant_id": self.info.participant_id,
            "objects": entries,
        }

    def _load_objects(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request_id = _request_id(body)
        root = self._checkpoint_root(body)
        values = body.get("objects")
        if not isinstance(values, list):
            raise ValueError("Mooncake restore request has no object list")
        entries = [
            _manifest_object_from_mapping(value)
            for value in values
            if isinstance(value, Mapping)
        ]
        if len(entries) != len(values):
            raise ValueError("Malformed Mooncake restore object")
        if len({entry.key for entry in entries}) != len(entries):
            raise ValueError("Duplicate Mooncake restore key")

        config = _local_replica_config(self._manager, self.info.segment_name)
        storage_dir = root / _STORAGE_DIR
        restored: list[str] = []
        with ExitStack() as stack:
            payloads: dict[str, Any] = {}
            for entry in entries:
                payload = payloads.get(entry.shard)
                if payload is None:
                    payload = stack.enter_context(
                        (storage_dir / entry.shard).open("rb")
                    )
                    payloads[entry.shard] = payload
                _restore_object(self._store, entry, config, payload)
                restored.append(entry.key)

        return {
            "ok": True,
            "request_id": request_id,
            "participant_id": self.info.participant_id,
            "restored_keys": restored,
        }


def _request_participant(
    request: _ParticipantRequest, *, timeout_s: float
) -> dict[str, Any]:
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    try:
        timeout_ms = max(1, int(timeout_s * 1000))
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.RCVHWM, 2)
        socket.setsockopt(zmq.IMMEDIATE, 1)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.connect(request.participant.control_endpoint)
        socket.send(
            json.dumps(request.body, separators=(",", ":"), sort_keys=True).encode()
        )
        if not socket.poll(timeout_ms, zmq.POLLIN):
            raise TimeoutError(
                "Timed out waiting for Mooncake checkpoint participant "
                f"{request.participant.participant_id}"
            )
        frames = socket.recv_multipart()
        if len(frames) != 1:
            raise RuntimeError("Malformed Mooncake checkpoint participant response")
        response = json.loads(frames[0])
        if not isinstance(response, dict):
            raise RuntimeError("Malformed Mooncake checkpoint participant response")
        if response.get("request_id") != request.body.get("request_id"):
            raise RuntimeError("Mooncake checkpoint response ID mismatch")
        if response.get("ok") is not True:
            raise RuntimeError(
                "Mooncake checkpoint participant "
                f"{request.participant.participant_id} failed: "
                f"{response.get('error', 'unknown error')}"
            )
        return response
    finally:
        socket.close(linger=0)
        context.term()


def _fanout_requests(
    requests: list[_ParticipantRequest],
    *,
    timeout_s: float,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
    allow_failures: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run participant requests concurrently; tests can replace this seam."""
    if not requests:
        return {}
    responses: dict[str, dict[str, Any]] = {}
    errors: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(requests))) as executor:
        futures = {
            executor.submit(
                _request_participant, request, timeout_s=timeout_s
            ): request.participant.participant_id
            for request in requests
        }
        for future in as_completed(futures):
            participant_id = futures[future]
            try:
                responses[participant_id] = future.result()
            except Exception as error:
                errors[participant_id] = error
    if errors and not allow_failures:
        detail = "; ".join(
            f"{participant_id}: {error}"
            for participant_id, error in sorted(errors.items())
        )
        raise RuntimeError(f"Mooncake checkpoint fanout failed: {detail}") from next(
            iter(errors.values())
        )
    return responses


def _require_exact_responses(
    requests: list[_ParticipantRequest],
    responses: Mapping[str, Any],
    *,
    operation: str,
) -> None:
    expected = {request.participant.participant_id for request in requests}
    if set(responses) != expected:
        raise RuntimeError(
            f"Mooncake {operation} fanout response set does not match its requests"
        )


def _live_participants(manager: Any) -> list[_ParticipantInfo]:
    """Return responsive existing-client endpoints; tests can replace this seam."""
    import ray

    local = getattr(manager, "_checkpoint_participant", None)
    registry = getattr(local, "_registry", None) or _registry_actor()
    timeout_s = min(_checkpoint_timeout_s(manager.config), 30.0)
    controller_session = _controller_session(manager)
    raw_participants = ray.get(
        registry.participants.remote(controller_session), timeout=timeout_s
    )
    if not isinstance(raw_participants, list):
        raise RuntimeError("Mooncake checkpoint registry returned malformed data")

    participants: list[_ParticipantInfo] = []
    ids: set[str] = set()
    endpoints: set[str] = set()
    for raw in raw_participants:
        if not isinstance(raw, Mapping):
            continue
        participant = _ParticipantInfo.from_mapping(raw)
        if (
            participant.participant_id in ids
            or participant.transport_endpoint in endpoints
        ):
            raise RuntimeError("Mooncake checkpoint registry contains duplicate owners")
        ids.add(participant.participant_id)
        endpoints.add(participant.transport_endpoint)
        participants.append(participant)

    ping_requests = [
        _ParticipantRequest(
            participant,
            _request_body(manager, "PING"),
        )
        for participant in participants
    ]
    responses = _fanout_requests(
        ping_requests,
        timeout_s=timeout_s,
        max_parallel=_checkpoint_max_parallel(manager.config),
        allow_failures=True,
    )
    live = [
        participant
        for participant in participants
        if participant.participant_id in responses
        and responses[participant.participant_id].get("participant")
        == asdict(participant)
    ]
    stale = [participant for participant in participants if participant not in live]
    if stale:
        refs = [
            registry.unregister.remote(
                participant.participant_id, participant.incarnation
            )
            for participant in stale
        ]
        with suppress(Exception):
            ray.get(refs, timeout=timeout_s)
    return sorted(live, key=lambda participant: participant.participant_id)


def _owner_assignments(
    store: Any,
    objects: list[_StoredObject],
    participants: list[_ParticipantInfo],
) -> dict[str, list[_StoredObject]]:
    endpoint_to_participant = {
        participant.transport_endpoint: participant for participant in participants
    }
    if len(endpoint_to_participant) != len(participants):
        raise RuntimeError("Mooncake checkpoint participants have duplicate endpoints")
    descriptors = store.batch_get_replica_desc([obj.key for obj in objects])
    if not isinstance(descriptors, Mapping):
        raise RuntimeError("Mooncake replica query returned malformed data")

    assignments: dict[str, list[_StoredObject]] = {
        participant.participant_id: [] for participant in participants
    }
    assigned_bytes = {participant.participant_id: 0 for participant in participants}
    for obj in objects:
        candidates: list[_ParticipantInfo] = []
        for endpoint, descriptor_size in _complete_memory_replicas(
            descriptors.get(obj.key)
        ):
            if descriptor_size != obj.size:
                raise RuntimeError(
                    f"Mooncake replica size mismatch for {obj.key!r}: "
                    f"catalog={obj.size}, descriptor={descriptor_size}"
                )
            participant = endpoint_to_participant.get(endpoint)
            if participant is not None:
                candidates.append(participant)
        if not candidates:
            raise RuntimeError(
                f"Mooncake key {obj.key!r} has no COMPLETE memory replica "
                "owned by a live checkpoint participant; disk-only/offloaded "
                "objects are not supported"
            )
        owner = min(
            candidates,
            key=lambda participant: (
                assigned_bytes[participant.participant_id],
                participant.participant_id,
            ),
        )
        assignments[owner.participant_id].append(obj)
        assigned_bytes[owner.participant_id] += obj.size
    return {
        participant_id: assigned
        for participant_id, assigned in assignments.items()
        if assigned
    }


def _validate_save_response(
    request: _ParticipantRequest, response: Mapping[str, Any]
) -> list[_ManifestObject]:
    if response.get("participant_id") != request.participant.participant_id:
        raise RuntimeError("Mooncake checkpoint participant ACK identity mismatch")
    expected = _parse_stored_objects(request.body.get("objects"))
    values = response.get("objects")
    if not isinstance(values, list) or len(values) != len(expected):
        raise RuntimeError("Mooncake checkpoint participant ACK object mismatch")
    entries: list[_ManifestObject] = []
    expected_offset = 0
    shard_name = request.body.get("shard_name")
    for obj, value in zip(expected, values, strict=True):
        if not isinstance(value, Mapping):
            raise RuntimeError("Malformed Mooncake checkpoint participant ACK")
        entry = _manifest_object_from_mapping(value)
        if (
            entry.key != obj.key
            or entry.size != obj.size
            or entry.offset != expected_offset
            or entry.shard != shard_name
            or entry.saved_owner != request.participant.transport_endpoint
        ):
            raise RuntimeError(
                f"Mooncake checkpoint participant ACK mismatch for {obj.key!r}"
            )
        entries.append(entry)
        expected_offset += obj.size
    return entries


def _write_manifest(
    storage_dir: Path,
    *,
    config: Any,
    entries: list[_ManifestObject],
) -> None:
    manifest = {
        "version": _MANIFEST_VERSION,
        "storage_layout": _storage_layout(config),
        "objects": [asdict(entry) for entry in sorted(entries, key=lambda x: x.key)],
    }
    partial = storage_dir / f".{_MANIFEST_FILE}.{uuid.uuid4().hex}.partial"
    with partial.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    partial.rename(storage_dir / _MANIFEST_FILE)


def _save_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    """Coordinate owner-local saves without carrying payload through manager."""
    _validate_checkpoint_runtime(manager)
    checkpoint_root = Path(checkpoint_dir).resolve()
    objects = _stored_objects(
        manager.storage_client._store, _controller_keys(checkpoint_root)
    )
    storage_dir = checkpoint_root / _STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=False)

    if not objects:
        _write_manifest(
            storage_dir,
            config=manager.config,
            entries=[],
        )
        return

    participants = _live_participants(manager)
    if not participants:
        raise RuntimeError("No live Mooncake checkpoint participants")
    assignments = _owner_assignments(
        manager.storage_client._store, objects, participants
    )
    by_id = {participant.participant_id: participant for participant in participants}
    requests: list[_ParticipantRequest] = []
    for index, participant_id in enumerate(sorted(assignments)):
        participant = by_id[participant_id]
        requests.append(
            _ParticipantRequest(
                participant,
                _request_body(
                    manager,
                    "SAVE_SHARD",
                    checkpoint_root=str(checkpoint_root),
                    shard_name=f"part-{index:05d}.bin",
                    objects=[asdict(obj) for obj in assignments[participant_id]],
                ),
            )
        )
    responses = _fanout_requests(
        requests,
        timeout_s=_checkpoint_timeout_s(manager.config),
        max_parallel=_checkpoint_max_parallel(manager.config),
    )
    _require_exact_responses(requests, responses, operation="checkpoint")

    entries: list[_ManifestObject] = []
    for request in requests:
        response = responses.get(request.participant.participant_id)
        if response is None:
            raise RuntimeError(
                "Missing Mooncake checkpoint ACK from "
                f"{request.participant.participant_id}"
            )
        entries.extend(_validate_save_response(request, response))
    if sorted(entry.key for entry in entries) != [obj.key for obj in objects]:
        raise RuntimeError("Mooncake checkpoint ACKs do not cover the controller cut")
    packed_sizes: dict[str, int] = {}
    for entry in entries:
        packed_sizes[entry.shard] = packed_sizes.get(entry.shard, 0) + entry.size
    for shard, expected_size in packed_sizes.items():
        shard_path = storage_dir / shard
        if not shard_path.is_file() or shard_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Mooncake checkpoint shard {shard!r} was not durably packed as ACKed"
            )

    _write_manifest(
        storage_dir,
        config=manager.config,
        entries=entries,
    )


def _load_manifest(
    checkpoint_root: Path,
) -> tuple[list[_ManifestObject], dict[str, Any]]:
    storage_dir = checkpoint_root / _STORAGE_DIR
    manifest = json.loads((storage_dir / _MANIFEST_FILE).read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("version") != _MANIFEST_VERSION
    ):
        raise ValueError("Unsupported Mooncake checkpoint manifest")
    raw_layout = manifest.get("storage_layout")
    if (
        not isinstance(raw_layout, Mapping)
        or not isinstance(raw_layout.get("use_gdr"), bool)
        or isinstance(raw_layout.get("gdr_staging_buffer_mb"), bool)
        or not isinstance(raw_layout.get("gdr_staging_buffer_mb"), int)
    ):
        raise ValueError("Mooncake checkpoint manifest has an invalid storage layout")
    layout = {
        "use_gdr": raw_layout["use_gdr"],
        "gdr_staging_buffer_mb": raw_layout["gdr_staging_buffer_mb"],
    }

    values = manifest.get("objects")
    if not isinstance(values, list):
        raise ValueError("Mooncake checkpoint manifest has no object list")
    entries: list[_ManifestObject] = []
    keys: set[str] = set()
    next_offset: dict[str, int] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Malformed Mooncake checkpoint object")
        entry = _manifest_object_from_mapping(value)
        if entry.key in keys or entry.offset != next_offset.get(entry.shard, 0):
            raise ValueError(f"Malformed Mooncake checkpoint entry for {entry.key!r}")
        keys.add(entry.key)
        next_offset[entry.shard] = entry.offset + entry.size
        entries.append(entry)
    if [entry.key for entry in entries] != sorted(keys):
        raise ValueError("Mooncake checkpoint manifest keys are not sorted")
    for shard, size in next_offset.items():
        path = storage_dir / shard
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or corrupt Mooncake checkpoint shard: {shard}")
    return entries, layout


def _restore_assignments(
    entries: list[_ManifestObject], participants: list[_ParticipantInfo]
) -> dict[str, list[_ManifestObject]]:
    assignments = {participant.participant_id: [] for participant in participants}
    assigned_bytes = {participant.participant_id: 0 for participant in participants}
    for entry in sorted(entries, key=lambda value: (-value.size, value.key)):
        participant = min(
            participants,
            key=lambda value: (
                assigned_bytes[value.participant_id],
                value.participant_id,
            ),
        )
        assignments[participant.participant_id].append(entry)
        assigned_bytes[participant.participant_id] += entry.size
    return {
        participant_id: sorted(values, key=lambda value: value.key)
        for participant_id, values in assignments.items()
        if values
    }


def _validate_restore_response(
    request: _ParticipantRequest, response: Mapping[str, Any]
) -> list[str]:
    if response.get("participant_id") != request.participant.participant_id:
        raise RuntimeError("Mooncake restore ACK identity mismatch")
    values = request.body.get("objects")
    if not isinstance(values, list):
        raise RuntimeError("Malformed Mooncake restore request")
    expected = [value.get("key") for value in values if isinstance(value, Mapping)]
    restored = response.get("restored_keys")
    if restored != expected:
        raise RuntimeError(
            f"Mooncake restore ACK mismatch from {request.participant.participant_id}"
        )
    return expected


def _load_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    """Restore payload on current clients before TQ loads its controller."""
    _validate_checkpoint_runtime(manager)
    checkpoint_root = Path(checkpoint_dir).resolve()
    entries, saved_layout = _load_manifest(checkpoint_root)
    current_layout = _storage_layout(manager.config)
    if saved_layout != current_layout:
        raise ValueError(
            "Mooncake checkpoint storage layout does not match this runtime: "
            f"saved={saved_layout}, current={current_layout}"
        )
    expected_keys = _controller_keys(checkpoint_root)
    if [entry.key for entry in entries] != expected_keys:
        raise ValueError(
            "Mooncake payload manifest does not match the TQ controller checkpoint"
        )

    store = manager.storage_client._store
    if entries:
        existence = store.batch_is_exist([entry.key for entry in entries])
        if len(existence) != len(entries) or any(result != 0 for result in existence):
            raise RuntimeError("Mooncake storage restore requires a clean store")
    else:
        return

    participants = _live_participants(manager)
    if not participants:
        raise RuntimeError("No live Mooncake checkpoint participants")
    assignments = _restore_assignments(entries, participants)
    by_id = {participant.participant_id: participant for participant in participants}
    expected_restore_endpoints = {
        entry.key: by_id[participant_id].transport_endpoint
        for participant_id, assigned in assignments.items()
        for entry in assigned
    }
    requests = [
        _ParticipantRequest(
            by_id[participant_id],
            _request_body(
                manager,
                "LOAD_OBJECTS",
                checkpoint_root=str(checkpoint_root),
                objects=[asdict(entry) for entry in assignments[participant_id]],
            ),
        )
        for participant_id in sorted(assignments)
    ]
    responses = _fanout_requests(
        requests,
        timeout_s=_checkpoint_timeout_s(manager.config),
        max_parallel=_checkpoint_max_parallel(manager.config),
    )
    _require_exact_responses(requests, responses, operation="restore")
    restored: list[str] = []
    for request in requests:
        response = responses.get(request.participant.participant_id)
        if response is None:
            raise RuntimeError(
                f"Missing Mooncake restore ACK from {request.participant.participant_id}"
            )
        restored.extend(_validate_restore_response(request, response))
    if sorted(restored) != expected_keys:
        raise RuntimeError("Mooncake restore ACKs do not cover the controller cut")

    descriptors = store.batch_get_replica_desc(expected_keys)
    if not isinstance(descriptors, Mapping):
        raise RuntimeError("Mooncake restore replica query returned malformed data")
    for entry in entries:
        replicas = _complete_memory_replicas(descriptors.get(entry.key))
        if not any(
            endpoint == expected_restore_endpoints[entry.key] and size == entry.size
            for endpoint, size in replicas
        ):
            raise RuntimeError(
                f"Restored Mooncake key {entry.key!r} has no COMPLETE memory "
                "replica on its assigned checkpoint participant"
            )


def install_tq_mooncake_checkpoint_plugin() -> None:
    """Install the explicit-checkpoint storage manager once."""
    from transfer_queue.storage.managers import mooncake_manager
    from transfer_queue.storage.managers.base import StorageManagerFactory

    manager_registry = StorageManagerFactory._registry
    current_manager = manager_registry.get("MooncakeStore")
    if getattr(current_manager, _PLUGIN_MARKER, False):
        return
    if current_manager is not mooncake_manager.MooncakeStorageManager:
        raise RuntimeError("Unexpected TQ MooncakeStore manager registration")

    class CheckpointMooncakeStorageManager(mooncake_manager.MooncakeStorageManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._checkpoint_participant: _CheckpointParticipant | None = None
            if _checkpoint_enabled(self.config):
                participant = _CheckpointParticipant(self)
                try:
                    participant.start_and_register()
                except BaseException:
                    participant.close()
                    with suppress(Exception):
                        super().close()
                    raise
                self._checkpoint_participant = participant

        async def save_checkpoint(self, checkpoint_dir: str) -> None:
            if not _checkpoint_enabled(self.config):
                await super().save_checkpoint(checkpoint_dir)
                return
            await asyncio.to_thread(_save_storage_checkpoint, self, checkpoint_dir)

        async def load_checkpoint(self, checkpoint_dir: str) -> None:
            if not _checkpoint_enabled(self.config):
                await super().load_checkpoint(checkpoint_dir)
                return
            await asyncio.to_thread(_load_storage_checkpoint, self, checkpoint_dir)

        def close(self) -> None:
            participant = getattr(self, "_checkpoint_participant", None)
            if participant is not None:
                participant.close()
                self._checkpoint_participant = None
            super().close()

    CheckpointMooncakeStorageManager.__name__ = "CheckpointMooncakeStorageManager"
    setattr(CheckpointMooncakeStorageManager, _PLUGIN_MARKER, True)
    manager_registry["MooncakeStore"] = CheckpointMooncakeStorageManager


__all__ = ["install_tq_mooncake_checkpoint_plugin"]
