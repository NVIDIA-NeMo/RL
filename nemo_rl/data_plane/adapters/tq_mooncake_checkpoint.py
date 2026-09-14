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
"""Owner-distributed Mooncake checkpoints for TransferQueue.

Normal Mooncake PUTs remain memory-only. At an explicit TQ checkpoint, the
controller commands the existing workers through their Ray actor handles.
Workers discover disjoint key slices and route ownership metadata through Ray
object references. Each payload is written directly from its owning process's
CPU segment: no payload crosses Ray, no native GET or staging copy is needed,
and the coordinator never collects detailed object addresses/sizes. No new
actors, registry, or normal PUT/CLEAR bookkeeping are introduced.

Restore reverses the process: the current Mooncake clients read size-balanced
sets of durable objects and upsert them into their own preferred segments
before TQ restores controller metadata.  Saved client identities are not
reused across restarts.

During save, objects selected by the controller snapshot must remain unchanged
until every owner's ACK. Generation may continue writing unrelated fresh keys,
but overwrites and clears of selected objects must wait. Their borrowed CPU
allocations must remain valid: do not move their replicas, unmount their segments,
or close their store clients. Hard pinning prevents eviction, not these explicit
mutations; same-host peer addresses are never dereferenced. During restore, keep
writers and clears stopped through every owner's verification and ACK. All
intended restore clients must connect before ``tq.load_checkpoint`` is called.
Checkpoint files must remain immutable throughout restore. Each owner validates
its indexes and key absence before writing. Storage load completes before TQ
installs its controller; it does not independently reread the controller snapshot.
Failures may leave partial or overwritten objects and do not roll back peers'
restores. A failed restore is unusable; start with a fresh restore environment.

New checkpoints use checksum-free format v3. Legacy v1/v2 payload and index
layouts remain readable, but their historical checksum fields are ignored.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import mmap
import os
import pickle
import uuid
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import ray
import torch

_PLUGIN_MARKER = "_nemo_rl_mooncake_checkpoint"
_STORAGE_DIR = "mooncake_storage"
_MANIFEST_FILE = "manifest.json"
# Per Ray wait, not a whole-checkpoint deadline; match TQ Simple's default.
_DEFAULT_TIMEOUT_S = 200.0
_BATCH_KEYS = 400  # Match TQ's native Mooncake batch limit.
_BATCH_BYTES = 64 * 1024 * 1024
_BUFFER_ALIGNMENT = 256  # Match TQ's native transfer-buffer alignment.

# An mmap must outlive its Mooncake registration. On registration errors or a
# failed unregister, retain it until process teardown instead of
# letting Python unmap memory that Mooncake or the NIC may still reference.
_QUARANTINED_BUFFERS: list[mmap.mmap] = []


def _fsync_directory(path: Path) -> None:
    """Make prior entry creation/rename operations durable in ``path``."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint_settings(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("MooncakeStore config must be a mapping")
    checkpoint = config.get("checkpoint") or {}
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MooncakeStore.checkpoint must be a mapping")
    return checkpoint


def _checkpoint_enabled(config: Any) -> bool:
    # Internal TQ runtime mode, derived at bootstrap from NeMo-RL's existing
    # checkpointing settings and selected resume path.
    return _checkpoint_settings(config).get("enabled") is True


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
    if manager.config.get("protocol", "tcp") not in {"tcp", "rdma"}:
        raise NotImplementedError(
            "Direct Mooncake checkpoints require TCP/RDMA CPU storage segments"
        )
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

    # Preserve traversal order while deduplicating so the final sort can reuse runs.
    keys: dict[str, None] = {}
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

        selected_statuses: list[list[int]] | None = None
        if (
            type(produced) is torch.Tensor
            and produced.device.type == "cpu"
            and produced.dtype == torch.int8
            and produced.ndim == 2
            and produced.layout == torch.strided
            and produced.names == (None, None)
            and type(indexes) in (set, list, tuple)
            and type(fields) is dict
        ):
            rows = list(indexes)
            columns = list(fields.values())
            if all(
                type(index) is int and -produced.shape[0] <= index < produced.shape[0]
                for index in rows
            ) and all(
                type(column) is int and -produced.shape[1] <= column < produced.shape[1]
                for column in columns
            ):
                # Row selection temporarily materializes all allocated columns;
                # Python list storage scales with selected rows times fields.
                selected_statuses = produced[rows][:, columns].tolist()

        for row_position, global_index in enumerate(indexes):
            per_index_meta = backend_meta.get(global_index) or {}
            if not isinstance(per_index_meta, Mapping):
                raise ValueError(
                    "TQ partition has malformed per-index backend metadata"
                )
            for field_position, (field_name, column) in enumerate(fields.items()):
                if selected_statuses is not None:
                    status = selected_statuses[row_position][field_position]
                else:
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
                    keys.update((f"{key}:c{i}", None) for i in range(n_chunks))
                else:
                    keys[key] = None
    return sorted(keys)


@dataclass(frozen=True)
class _StoredObject:
    key: str
    size: int
    address: int


@dataclass(frozen=True)
class _SaveGroup:
    controller_session: str
    owner: str
    objects: list[_StoredObject]


@dataclass(frozen=True)
class _ParticipantInfo:
    participant_id: str
    controller_session: str
    segment_name: str
    transport_endpoint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> _ParticipantInfo:
        fields = {
            name: value.get(name)
            for name in (
                "participant_id",
                "controller_session",
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
    saved_owner: str


@dataclass(frozen=True)
class _ShardIndex:
    shard: str
    object_count: int
    payload_bytes: int
    saved_owner: str


def _controller_path(checkpoint_dir: Path) -> Path:
    # Optional dependency; importing it at module load would eagerly load TQ.
    from transfer_queue import interface as tq_interface

    return checkpoint_dir / tq_interface._CONTROLLER_FILE


def _controller_keys(checkpoint_dir: Path) -> list[str]:
    path = _controller_path(checkpoint_dir)
    with (
        path.open("rb") as controller_file,
    ):
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
    cleanup_error: BaseException
    try:
        result = store.unregister_buffer(buffer.pointer)
    except BaseException as error:
        cleanup_error = error
    else:
        if result == 0:
            return
        cleanup_error = RuntimeError(f"status {result}")

    buffer.quarantined = True
    _QUARANTINED_BUFFERS.append(buffer.payload)
    raise RuntimeError(
        f"Mooncake buffer cleanup failed for {label}; the mapped buffer was retained"
    ) from cleanup_error


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
        # A failed multi-NIC registration may still retain native registrations.
        buffer.quarantined = True
        _QUARANTINED_BUFFERS.append(buffer.payload)
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


def _write_buffer(
    output: Any,
    buffer: mmap.mmap | memoryview,
    size: int,
    *,
    label: str,
    offset: int = 0,
) -> None:
    view = memoryview(buffer)[offset : offset + size]
    try:
        offset = 0
        while offset < size:
            written = output.write(view[offset:size])
            if written is None or written <= 0:
                raise OSError(f"Short checkpoint write for {label}")
            offset += written
    finally:
        view.release()


def _read_buffer(
    source: Any, buffer: mmap.mmap, size: int, *, label: str, offset: int = 0
) -> None:
    view = memoryview(buffer)[offset : offset + size]
    try:
        offset = 0
        while offset < size:
            read = source.readinto(view[offset:size])
            if read is None or read <= 0:
                raise OSError(f"Short checkpoint read for {label}")
            offset += read
    finally:
        view.release()


def _checkpoint_batches(sizes: list[int]) -> Iterator[tuple[int, int, list[int]]]:
    """Bound native calls by keys and bytes; allow an oversized singleton."""
    start = 0
    offsets: list[int] = []
    total = 0
    for index, size in enumerate(sizes):
        if offsets and (len(offsets) == _BATCH_KEYS or total + size > _BATCH_BYTES):
            yield start, index, offsets
            start, offsets, total = index, [], 0
        offsets.append(total)
        total += (size + _BUFFER_ALIGNMENT - 1) // _BUFFER_ALIGNMENT * _BUFFER_ALIGNMENT
    if offsets:
        yield start, len(sizes), offsets


@contextmanager
def _checkpoint_buffer(
    store: Any, sizes: list[int], *, label: str
) -> Iterator[_CheckpointBuffer]:
    # One registration for every batch in this owner operation. An object larger
    # than the target gets enough space by itself, rather than being truncated.
    capacity = max(
        (
            offsets[-1] + sizes[stop - 1]
            for _, stop, offsets in _checkpoint_batches(sizes)
        ),
        default=1,
    )
    buffer = _CheckpointBuffer.allocate(capacity)
    try:
        with _registered_buffer(store, buffer, size=capacity, label=label):
            yield buffer
    finally:
        buffer.close()


def _check_batch_results(results: Any, expected: list[int], *, operation: str) -> None:
    """Native GET returns byte counts; UPSERT returns zero statuses, per key."""
    if (
        not isinstance(results, (list, tuple))
        or len(results) != len(expected)
        or any(
            type(result) is not int or result != size
            for result, size in zip(results, expected)
        )
    ):
        raise RuntimeError(
            f"Mooncake checkpoint {operation} returned incomplete or invalid results: {results!r}"
        )


def _batch_replicas(store: Any, keys: list[str]) -> Mapping[str, Any]:
    descriptors = store.batch_get_replica_desc(keys)
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(keys):
        raise RuntimeError("Mooncake replica query returned malformed data")
    return descriptors


def _status_is_complete(status: Any) -> bool:
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name == "COMPLETE"
    return str(status).rsplit(".", 1)[-1] == "COMPLETE"


def _complete_memory_replicas(
    descriptors: Any,
) -> list[tuple[str, int]]:
    return [
        (buffer.transport_endpoint, buffer.size)
        for buffer in _complete_memory_buffers(descriptors)
    ]


def _complete_memory_buffers(descriptors: Any) -> list[Any]:
    replicas: list[Any] = []
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
            replicas.append(buffer)
    return replicas


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
        "controller_session": _controller_session(manager),
        "operation": operation,
        **payload,
    }


def _safe_shard_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or (os.path.basename(value) if type(value) is str else Path(value).name)
        != value
    ):
        raise ValueError(f"Invalid Mooncake checkpoint shard name: {value!r}")
    if not value.startswith("part-") or not value.endswith(".bin"):
        raise ValueError(f"Invalid Mooncake checkpoint shard name: {value!r}")
    return value


def _manifest_object_from_mapping(value: Mapping[str, Any]) -> _ManifestObject:
    key = value.get("key")
    shard = value.get("shard")
    offset = value.get("offset")
    size = value.get("size")
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
        or not isinstance(saved_owner, str)
        or not saved_owner
    ):
        raise ValueError(f"Malformed Mooncake checkpoint entry for {key!r}")
    return _ManifestObject(key, shard, offset, size, saved_owner)


def _shard_index_from_mapping(value: Any) -> _ShardIndex:
    if not isinstance(value, Mapping):
        raise ValueError("Malformed Mooncake shard index descriptor")
    shard = _safe_shard_name(value.get("shard"))
    count = value.get("object_count")
    size = value.get("payload_bytes")
    owner = value.get("saved_owner")
    if (
        type(count) is not int
        or count <= 0
        or type(size) is not int
        or size <= 0
        or not isinstance(owner, str)
        or not owner
    ):
        raise ValueError(f"Malformed Mooncake shard index descriptor: {shard!r}")
    return _ShardIndex(shard, count, size, owner)


def _parse_shard_indexes(values: Any) -> list[_ShardIndex]:
    if not isinstance(values, list):
        raise ValueError("Mooncake checkpoint has no shard index list")
    shards = [_shard_index_from_mapping(value) for value in values]
    if len({shard.shard for shard in shards}) != len(shards):
        raise ValueError("Duplicate Mooncake shard index")
    return shards


def _read_shard_indexes(root: Path, values: Any) -> list[_ManifestObject]:
    storage_dir = root / _STORAGE_DIR
    entries: list[_ManifestObject] = []
    keys: set[str] = set()
    for descriptor in _parse_shard_indexes(values):
        index_path = storage_dir / f"{descriptor.shard}.index.json"
        if index_path.is_symlink():
            raise ValueError("Mooncake shard index must not be a symbolic link")
        data = index_path.read_bytes()
        index = json.loads(data)
        if not isinstance(index, Mapping) or not isinstance(index.get("objects"), list):
            raise ValueError("Malformed Mooncake shard index objects")
        objects = index["objects"]
        if len(objects) != descriptor.object_count:
            raise ValueError("Mooncake shard index object count mismatch")
        offset = 0
        previous_key: str | None = None
        for value in objects:
            if not isinstance(value, Mapping):
                raise ValueError("Malformed Mooncake shard index object")
            entry = _manifest_object_from_mapping(value)
            if (
                entry.shard != descriptor.shard
                or entry.saved_owner != descriptor.saved_owner
                or entry.offset != offset
                or entry.key in keys
                or (previous_key is not None and entry.key <= previous_key)
            ):
                raise ValueError(f"Malformed Mooncake shard index entry: {entry.key!r}")
            entries.append(entry)
            keys.add(entry.key)
            offset += entry.size
            previous_key = entry.key
        payload_path = storage_dir / descriptor.shard
        if (
            offset != descriptor.payload_bytes
            or payload_path.is_symlink()
            or not payload_path.is_file()
            or payload_path.stat().st_size != offset
        ):
            raise ValueError(
                f"Missing or corrupt Mooncake checkpoint shard: {descriptor.shard}"
            )
    return entries


def _write_shard_index(
    storage_dir: Path, *, shard: str, entries: list[dict[str, Any]], saved_owner: str
) -> _ShardIndex:
    target = storage_dir / f"{shard}.index.json"
    if target.exists():
        raise FileExistsError(f"Mooncake checkpoint index already exists: {target}")
    data = (json.dumps({"objects": entries}, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = _ShardIndex(
        shard,
        len(entries),
        sum(entry["size"] for entry in entries),
        saved_owner,
    )
    partial = storage_dir / f".{target.name}.{uuid.uuid4().hex}.partial"
    with partial.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    partial.rename(target)
    _fsync_directory(storage_dir)
    return descriptor


def _local_replica_config(manager: Any, segment_name: str) -> Any:
    # Optional native extension; it ships without Python type stubs.
    from mooncake.store import ReplicateConfig  # type: ignore[import-not-found]

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

        self.info = _ParticipantInfo(
            participant_id=str(manager.storage_manager_id),
            controller_session=_controller_session(manager),
            segment_name=segment_name,
            # In HTTP/etcd metadata mode Mooncake writes local_hostname into
            # every mounted memory descriptor's transport_endpoint.
            transport_endpoint=segment_name,
        )

    def _dispatch(self, body: Mapping[str, Any]) -> dict[str, Any]:
        if body.get("controller_session") != self.info.controller_session:
            raise ValueError("Mooncake checkpoint controller session mismatch")
        operation = body.get("operation")
        if operation == "DESCRIBE":
            return {
                "ok": True,
                "participant": asdict(self.info),
            }
        if operation == "SAVE_SHARD":
            return self._save_shard(body)
        if operation == "DISCOVER_SAVE":
            return self._discover_save(body)
        if operation == "LOAD_OBJECTS":
            return self._load_objects(body)
        if operation == "LOAD_SHARDS":
            return self._load_objects(body)
        raise ValueError(f"Unsupported Mooncake checkpoint operation: {operation!r}")

    def _discover_save(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Query each key once and leave detailed plans with their Ray owners."""
        keys = body.get("keys")
        endpoints = body.get("current_endpoints")
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key for key in keys)
            or any(left >= right for left, right in zip(keys, keys[1:]))
            or not isinstance(endpoints, list)
            or not endpoints
            or any(
                not isinstance(endpoint, str) or not endpoint for endpoint in endpoints
            )
            or len(set(endpoints)) != len(endpoints)
            or self.info.transport_endpoint not in endpoints
        ):
            raise ValueError("Malformed Mooncake save key-slice request")
        current_endpoints = set(endpoints)
        groups: dict[str, list[_StoredObject]] = {}
        for start in range(0, len(keys), _BATCH_KEYS):
            descriptors = _batch_replicas(
                self._store, keys[start : start + _BATCH_KEYS]
            )
            for key, values in descriptors.items():
                replicas = _complete_memory_buffers(values)
                eligible = [
                    replica
                    for replica in replicas
                    if replica.transport_endpoint in current_endpoints
                ]
                if not eligible:
                    raise RuntimeError(
                        f"Mooncake key {key!r} has no COMPLETE memory replica "
                        "owned by a live checkpoint participant"
                    )
                size = eligible[0].size
                if any(
                    type(replica.size) is not int
                    or replica.size <= 0
                    or replica.size != size
                    for replica in replicas
                ):
                    raise RuntimeError(
                        f"Mooncake replica sizes are invalid or inconsistent for {key!r}"
                    )
                # The usual single-replica path needs no placement search.
                # Replicated objects use a canonical live owner, not global
                # byte balancing: locality takes precedence over redistribution.
                replica = (
                    eligible[0]
                    if len(eligible) == 1
                    else min(eligible, key=lambda value: value.transport_endpoint)
                )
                address = getattr(replica, "buffer_address", None)
                if type(address) is not int or address <= 0:
                    raise RuntimeError(
                        f"Mooncake replica has no CPU address for {key!r}"
                    )
                groups.setdefault(replica.transport_endpoint, []).append(
                    _StoredObject(key, size, address)
                )
        return {
            "ok": True,
            "participant_id": self.info.participant_id,
            "groups": {
                owner: (
                    ray.put(_SaveGroup(self.info.controller_session, owner, objects)),
                    len(objects),
                    sum(obj.size for obj in objects),
                )
                for owner, objects in groups.items()
            },
        }

    def _resolve_save_objects(self, body: Mapping[str, Any]) -> list[_StoredObject]:
        references = body.get("object_groups")
        if not isinstance(references, list) or not references:
            raise ValueError("Mooncake save requires owner metadata references")
        # Discovery finishes everywhere before this call; no actor method must
        # execute recursively to produce a referenced group.
        groups = ray.get(references, timeout=_DEFAULT_TIMEOUT_S)
        objects: list[_StoredObject] = []
        for group in groups:
            if (
                not isinstance(group, _SaveGroup)
                or group.controller_session != self.info.controller_session
                or group.owner != self.info.transport_endpoint
            ):
                raise ValueError(
                    "Mooncake save metadata belongs to a different owner/session"
                )
            for obj in group.objects:
                if (
                    not isinstance(obj, _StoredObject)
                    or not isinstance(obj.key, str)
                    or not obj.key
                    or type(obj.size) is not int
                    or obj.size <= 0
                    or type(obj.address) is not int
                    or obj.address <= 0
                ):
                    raise ValueError("Malformed Mooncake owner memory descriptor")
                objects.append(obj)
        objects.sort(key=lambda obj: obj.key)
        if not objects or any(a.key == b.key for a, b in zip(objects, objects[1:])):
            raise ValueError("Empty or duplicate Mooncake owner save keys")
        return objects

    def _checkpoint_root(self, body: Mapping[str, Any]) -> Path:
        value = body.get("checkpoint_root")
        if not isinstance(value, str) or not value:
            raise ValueError("Mooncake checkpoint request has no checkpoint_root")
        root = Path(value)
        if not root.is_absolute():
            raise ValueError("Mooncake checkpoint_root must be absolute")
        return root

    def _verify_restored_objects(
        self, entries: list[_ManifestObject], current_endpoints: set[str]
    ) -> None:
        # The preferred segment may fall back to another current participant.
        # Keep the coordinator's original acceptance rule, not a self-only check.
        for start in range(0, len(entries), _BATCH_KEYS):
            batch = entries[start : start + _BATCH_KEYS]
            descriptors = _batch_replicas(self._store, [entry.key for entry in batch])
            for entry in batch:
                replicas = _complete_memory_replicas(descriptors[entry.key])
                if not any(
                    endpoint in current_endpoints and size == entry.size
                    for endpoint, size in replicas
                ):
                    raise RuntimeError(
                        f"Restored Mooncake key {entry.key!r} has no COMPLETE memory "
                        "replica on a current checkpoint participant"
                    )

    def _save_shard(self, body: Mapping[str, Any]) -> dict[str, Any]:
        root = self._checkpoint_root(body)
        shard = _safe_shard_name(body.get("shard_name"))
        objects = self._resolve_save_objects(body)
        storage_dir = root / _STORAGE_DIR
        if not storage_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint storage directory is missing: {storage_dir}"
            )
        target = storage_dir / shard
        if target.exists():
            raise FileExistsError(f"Mooncake checkpoint shard already exists: {target}")
        partial = storage_dir / f".{shard}.{uuid.uuid4().hex}.partial"

        entries: list[dict[str, Any]] = []
        offset = 0
        with partial.open("xb") as output:
            for obj in objects:
                # Borrow this process's hard-pinned CPU allocation, not a copy.
                # Selected objects must not be overwritten, cleared or moved,
                # and their stores must stay alive until all SAVE ACKs complete.
                allocation = (ctypes.c_ubyte * obj.size).from_address(obj.address)
                # ctypes arrays expose a buffer; their type stubs omit it.
                with memoryview(cast(Any, allocation)).cast("B") as view:
                    _write_buffer(output, view, obj.size, label=obj.key)
                entries.append(
                    {
                        "key": obj.key,
                        "shard": shard,
                        "offset": offset,
                        "size": obj.size,
                        "saved_owner": self.info.transport_endpoint,
                    }
                )
                offset += obj.size
            output.flush()
            os.fsync(output.fileno())
        partial.rename(target)
        _fsync_directory(storage_dir)
        descriptor = _write_shard_index(
            storage_dir,
            shard=shard,
            entries=entries,
            saved_owner=self.info.transport_endpoint,
        )
        return {
            "ok": True,
            "participant_id": self.info.participant_id,
            "shard": asdict(descriptor),
            "saved_keys": [obj.key for obj in objects],
        }

    def _load_objects(self, body: Mapping[str, Any]) -> dict[str, Any]:
        root = self._checkpoint_root(body)
        if body.get("operation") == "LOAD_SHARDS":
            entries = _read_shard_indexes(root, body.get("shards"))
            _require_clean_store(self._store, [entry.key for entry in entries])
        else:
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

        endpoint_values = body.get("current_endpoints")
        if (
            not isinstance(endpoint_values, list)
            or not endpoint_values
            or any(not isinstance(value, str) or not value for value in endpoint_values)
            or len(set(endpoint_values)) != len(endpoint_values)
            or self.info.transport_endpoint not in endpoint_values
        ):
            raise ValueError("Malformed Mooncake restore current endpoints")
        current_endpoints = set(endpoint_values)

        config = _local_replica_config(self._manager, self.info.segment_name)
        storage_dir = root / _STORAGE_DIR
        restored: list[str] = []
        sizes = [entry.size for entry in entries]
        with (
            ExitStack() as stack,
            _checkpoint_buffer(self._store, sizes, label="restore") as buffer,
        ):
            payloads: dict[str, Any] = {}
            for start, stop, offsets in _checkpoint_batches(sizes):
                batch = entries[start:stop]
                for entry, position in zip(batch, offsets, strict=True):
                    payload = payloads.get(entry.shard)
                    if payload is None:
                        payload = stack.enter_context(
                            (storage_dir / entry.shard).open("rb")
                        )
                        payloads[entry.shard] = payload
                    payload.seek(entry.offset)
                    _read_buffer(
                        payload,
                        buffer.payload,
                        entry.size,
                        label=entry.key,
                        offset=position,
                    )
                results = self._store.batch_upsert_from(
                    [entry.key for entry in batch],
                    [buffer.pointer + position for position in offsets],
                    sizes[start:stop],
                    config,
                )
                _check_batch_results(results, [0] * len(batch), operation="restore")
                restored.extend(entry.key for entry in batch)

        # Each owner verifies only its disjoint assignment before acknowledging;
        # quiescent writers/clears remain required until the whole restore ends.
        self._verify_restored_objects(entries, current_endpoints)
        return {
            "ok": True,
            "participant_id": self.info.participant_id,
            "restored_keys": restored,
        }


def _fanout_requests(
    requests: list[_ParticipantRequest],
    *,
    workers: Mapping[str, Any],
    local: _CheckpointParticipant | None,
    timeout_s: float,
) -> dict[str, dict[str, Any]]:
    """Send metadata-only commands to existing actors; never RPC back to self."""
    responses: dict[str, dict[str, Any]] = {}
    try:
        pending: dict[str, Any] = {}
        local_requests: list[_ParticipantRequest] = []
        for request in requests:
            participant_id = request.participant.participant_id
            if local is not None and participant_id == local.info.participant_id:
                local_requests.append(request)
            else:
                pending[participant_id] = workers[
                    participant_id
                ].mooncake_checkpoint.remote(body=request.body)
        # Remote I/O runs concurrently with this process's own shard I/O.
        for request in local_requests:
            assert local is not None
            responses[request.participant.participant_id] = local._dispatch(
                request.body
            )
        results = ray.get(list(pending.values()), timeout=timeout_s)
        responses.update(zip(pending, results, strict=True))
    except Exception as error:
        # An RPC may still be writing after a timeout. Do not let the periodic
        # checkpoint pump mistake that uncertainty for a retryable local timeout.
        raise RuntimeError(f"Mooncake checkpoint fanout failed: {error}") from error
    for response in responses.values():
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError("Malformed Mooncake checkpoint worker response")
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


def _live_participants(
    manager: Any,
) -> tuple[list[_ParticipantInfo], dict[str, Any]]:
    """Describe the explicitly supplied workers, plus the calling process."""
    workers = manager._checkpoint_workers
    body = _request_body(manager, "DESCRIBE")
    try:
        responses = ray.get(
            [worker.mooncake_checkpoint.remote(body=body) for worker in workers],
            timeout=_DEFAULT_TIMEOUT_S,
        )
    except Exception as error:
        raise RuntimeError("Mooncake checkpoint worker discovery failed") from error
    participants: dict[str, _ParticipantInfo] = {}
    handles: dict[str, Any] = {}
    local = manager._checkpoint_participant
    if local is not None:
        participants[local.info.participant_id] = local.info
    for worker, response in zip(workers, responses, strict=True):
        # Some generation ranks do not host token capture or a TQ client.
        if response is None:
            continue
        if not isinstance(response, Mapping) or response.get("ok") is not True:
            raise RuntimeError("Malformed Mooncake checkpoint worker description")
        raw = response.get("participant")
        if not isinstance(raw, Mapping):
            raise RuntimeError("Missing Mooncake checkpoint worker identity")
        participant = _ParticipantInfo.from_mapping(raw)
        if participant.controller_session != body["controller_session"]:
            raise RuntimeError("Mooncake checkpoint controller session mismatch")
        existing = participants.get(participant.participant_id)
        if existing is not None and existing != participant:
            raise RuntimeError("Conflicting Mooncake checkpoint worker identities")
        participants[participant.participant_id] = participant
        handles[participant.participant_id] = worker
    live = list(participants.values())
    if len({participant.transport_endpoint for participant in live}) != len(live):
        raise RuntimeError(
            "Live Mooncake checkpoint participants have duplicate endpoints"
        )
    return sorted(live, key=lambda participant: participant.participant_id), handles


def _write_manifest(
    storage_dir: Path,
    *,
    config: Any,
    entries: list[_ShardIndex],
) -> None:
    manifest = {
        "format_version": 3,
        "storage_layout": _storage_layout(config),
        "shards": [asdict(entry) for entry in sorted(entries, key=lambda x: x.shard)],
    }
    partial = storage_dir / f".{_MANIFEST_FILE}.{uuid.uuid4().hex}.partial"
    with partial.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2, sort_keys=True))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    partial.rename(storage_dir / _MANIFEST_FILE)
    _fsync_directory(storage_dir)


def _save_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    """Route metadata references; payload and detailed plans stay distributed."""
    _validate_checkpoint_runtime(manager)
    checkpoint_root = Path(checkpoint_dir).resolve()
    keys = _controller_keys(checkpoint_root)
    storage_dir = checkpoint_root / _STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(checkpoint_root)
    if not keys:
        _write_manifest(storage_dir, config=manager.config, entries=[])
        return

    participants, workers = _live_participants(manager)
    if not participants:
        raise RuntimeError("No live Mooncake checkpoint participants")
    endpoints = [participant.transport_endpoint for participant in participants]
    batch_count = (len(keys) + _BATCH_KEYS - 1) // _BATCH_KEYS
    discoverers = participants[:batch_count]
    discoveries = [
        _ParticipantRequest(
            participant,
            _request_body(
                manager,
                "DISCOVER_SAVE",
                keys=keys[
                    (index * batch_count // len(discoverers)) * _BATCH_KEYS : (
                        (index + 1) * batch_count // len(discoverers)
                    )
                    * _BATCH_KEYS
                ],
                current_endpoints=endpoints,
            ),
        )
        for index, participant in enumerate(discoverers)
    ]
    # Keep these responses (and nested refs) alive through the SAVE ACKs.
    plans = _fanout_requests(
        discoveries,
        workers=workers,
        local=manager._checkpoint_participant,
        timeout_s=_DEFAULT_TIMEOUT_S,
    )
    _require_exact_responses(discoveries, plans, operation="discovery")
    references: dict[str, list[Any]] = {endpoint: [] for endpoint in endpoints}
    counts = dict.fromkeys(endpoints, 0)
    sizes = dict.fromkeys(endpoints, 0)
    for request in discoveries:
        response = plans[request.participant.participant_id]
        groups = response.get("groups")
        if response.get(
            "participant_id"
        ) != request.participant.participant_id or not isinstance(groups, Mapping):
            raise RuntimeError("Malformed Mooncake save discovery")
        discovered_count = 0
        for owner, group in groups.items():
            if (
                owner not in references
                or not isinstance(group, (list, tuple))
                or len(group) != 3
            ):
                raise RuntimeError("Malformed Mooncake save owner group")
            reference, count, size = group
            if (
                type(count) is not int
                or count <= 0
                or type(size) is not int
                or size <= 0
            ):
                raise RuntimeError("Malformed Mooncake save group sizes")
            references[owner].append(reference)
            counts[owner] += count
            sizes[owner] += size
            discovered_count += count
        if discovered_count != len(request.body["keys"]):
            raise RuntimeError("Mooncake save discovery did not cover its keys")
    requests = [
        _ParticipantRequest(
            participant,
            _request_body(
                manager,
                "SAVE_SHARD",
                checkpoint_root=str(checkpoint_root),
                shard_name=f"part-{index:05d}.bin",
                object_groups=references[participant.transport_endpoint],
            ),
        )
        for index, participant in enumerate(participants)
        if references[participant.transport_endpoint]
    ]
    responses = _fanout_requests(
        requests,
        workers=workers,
        local=manager._checkpoint_participant,
        timeout_s=_DEFAULT_TIMEOUT_S,
    )
    _require_exact_responses(requests, responses, operation="checkpoint")
    entries: list[_ShardIndex] = []
    saved_keys: list[str] = []
    for request in requests:
        response = responses[request.participant.participant_id]
        descriptor = _shard_index_from_mapping(response.get("shard"))
        owner = request.participant.transport_endpoint
        acknowledged = response.get("saved_keys")
        if (
            response.get("participant_id") != request.participant.participant_id
            or not isinstance(acknowledged, list)
            or any(not isinstance(key, str) or not key for key in acknowledged)
            or descriptor.shard != request.body["shard_name"]
            or descriptor.saved_owner != owner
            or descriptor.object_count != counts[owner]
            or descriptor.payload_bytes != sizes[owner]
            or len(acknowledged) != counts[owner]
        ):
            raise RuntimeError("Mooncake checkpoint shard ACK mismatch")
        entries.append(descriptor)
        saved_keys.extend(acknowledged)
    if sorted(saved_keys) != keys:
        raise RuntimeError("Mooncake checkpoint ACKs do not cover the controller cut")
    for entry in entries:
        shard_path = storage_dir / entry.shard
        index_path = storage_dir / f"{entry.shard}.index.json"
        if (
            not shard_path.is_file()
            or shard_path.stat().st_size != entry.payload_bytes
            or not index_path.is_file()
        ):
            raise RuntimeError(
                f"Mooncake checkpoint shard {entry.shard!r} was not durably packed as ACKed"
            )
    _write_manifest(storage_dir, config=manager.config, entries=entries)


def _load_manifest(
    checkpoint_root: Path,
) -> tuple[list[_ManifestObject] | list[_ShardIndex], dict[str, Any]]:
    storage_dir = checkpoint_root / _STORAGE_DIR
    manifest_text = (storage_dir / _MANIFEST_FILE).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    del manifest_text
    if not isinstance(manifest, Mapping):
        raise ValueError("Mooncake checkpoint manifest must contain a mapping")
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

    if "format_version" in manifest:
        version = manifest["format_version"]
        if type(version) is not int or version not in (2, 3) or "objects" in manifest:
            raise ValueError("Unsupported Mooncake checkpoint manifest format version")
        shards = _parse_shard_indexes(manifest.get("shards"))
        return shards, layout

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
    shards: dict[str, list[_ManifestObject]] = {}
    shard_bytes: dict[str, int] = {}
    for entry in entries:
        shards.setdefault(entry.shard, []).append(entry)
        shard_bytes[entry.shard] = shard_bytes.get(entry.shard, 0) + entry.size
    assignments = {participant.participant_id: [] for participant in participants}
    assigned_bytes = {participant.participant_id: 0 for participant in participants}
    for shard in sorted(shards, key=lambda value: (-shard_bytes[value], value)):
        participant = min(
            participants,
            key=lambda value: (
                assigned_bytes[value.participant_id],
                value.participant_id,
            ),
        )
        assignments[participant.participant_id].extend(shards[shard])
        assigned_bytes[participant.participant_id] += shard_bytes[shard]
    return {
        participant_id: sorted(values, key=lambda value: (value.shard, value.offset))
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
    expected: list[str] = []
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get("key"), str):
            raise RuntimeError("Malformed Mooncake restore request key")
        expected.append(value["key"])
    restored = response.get("restored_keys")
    if restored != expected:
        raise RuntimeError(
            f"Mooncake restore ACK mismatch from {request.participant.participant_id}"
        )
    return expected


def _require_clean_store(
    store: Any,
    keys: list[str],
) -> None:
    for start in range(0, len(keys), _BATCH_KEYS):
        batch = keys[start : start + _BATCH_KEYS]
        existence = store.batch_is_exist(batch)
        if not isinstance(existence, (list, tuple)) or len(existence) != len(batch):
            raise RuntimeError("Malformed Mooncake existence response")
        for key, result in zip(batch, existence, strict=True):
            if type(result) is not int or result > 1:
                raise RuntimeError(
                    f"Malformed Mooncake existence response for key {key!r}: {result!r}"
                )
            if result < 0:
                raise RuntimeError(
                    f"Mooncake existence check failed for key {key!r}: error code {result}"
                )
            if result == 1:
                raise RuntimeError(
                    "Mooncake storage restore requires a clean store: "
                    f"key {key!r} already exists"
                )


def _load_sharded_checkpoint(
    manager: Any, root: Path, shards: list[_ShardIndex]
) -> None:
    participants, workers = _live_participants(manager)
    if not participants:
        raise RuntimeError("No live Mooncake checkpoint participants")
    assignments: dict[str, list[_ShardIndex]] = {
        participant.participant_id: [] for participant in participants
    }
    sizes = {participant.participant_id: 0 for participant in participants}
    for shard in sorted(shards, key=lambda item: (-item.payload_bytes, item.shard)):
        owner = min(
            participants,
            key=lambda item: (sizes[item.participant_id], item.participant_id),
        )
        assignments[owner.participant_id].append(shard)
        sizes[owner.participant_id] += shard.payload_bytes
    endpoints = sorted(participant.transport_endpoint for participant in participants)
    requests = [
        _ParticipantRequest(
            participant,
            _request_body(
                manager,
                "LOAD_SHARDS",
                checkpoint_root=str(root),
                shards=[
                    asdict(shard)
                    for shard in sorted(
                        assignments[participant.participant_id],
                        key=lambda item: item.shard,
                    )
                ],
                current_endpoints=endpoints,
            ),
        )
        for participant in participants
        if assignments[participant.participant_id]
    ]
    responses = _fanout_requests(
        requests,
        workers=workers,
        local=manager._checkpoint_participant,
        timeout_s=_DEFAULT_TIMEOUT_S,
    )
    _require_exact_responses(requests, responses, operation="restore")
    for request in requests:
        response = responses[request.participant.participant_id]
        keys = response.get("restored_keys")
        count = sum(
            shard.object_count
            for shard in assignments[request.participant.participant_id]
        )
        if (
            response.get("participant_id") != request.participant.participant_id
            or not isinstance(keys, list)
            or len(keys) != count
            or any(not isinstance(key, str) or not key for key in keys)
        ):
            raise RuntimeError("Malformed Mooncake restore ACK")


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
    if entries and isinstance(entries[0], _ShardIndex):
        _load_sharded_checkpoint(
            manager, checkpoint_root, cast(list[_ShardIndex], entries)
        )
        return
    entries = cast(list[_ManifestObject], entries)

    store = manager.storage_client._store
    if not entries:
        return
    _require_clean_store(store, [entry.key for entry in entries])
    participants, workers = _live_participants(manager)
    if not participants:
        raise RuntimeError("No live Mooncake checkpoint participants")
    assignments = _restore_assignments(entries, participants)
    by_id = {participant.participant_id: participant for participant in participants}
    current_endpoints = {participant.transport_endpoint for participant in participants}
    requests = [
        _ParticipantRequest(
            by_id[participant_id],
            _request_body(
                manager,
                "LOAD_OBJECTS",
                checkpoint_root=str(checkpoint_root),
                objects=[asdict(entry) for entry in assignments[participant_id]],
                current_endpoints=sorted(current_endpoints),
            ),
        )
        for participant_id in sorted(assignments)
    ]
    responses = _fanout_requests(
        requests,
        workers=workers,
        local=manager._checkpoint_participant,
        timeout_s=_DEFAULT_TIMEOUT_S,
    )
    _require_exact_responses(requests, responses, operation="restore")
    for request in requests:
        response = responses[request.participant.participant_id]
        _validate_restore_response(request, response)


def configure_checkpoint_workers(workers: list[Any]) -> None:
    """Bind existing actor handles for this process's checkpoint coordinator.

    Call after all intended owners have attached, before save or restore. Do
    not include the calling actor: its shard is executed directly, including
    when restoring inside SingleController's constructor.
    """
    # TransferQueue is optional outside this backend.
    import transfer_queue as tq

    manager = tq.get_client().storage_manager
    if not getattr(type(manager), _PLUGIN_MARKER, False):
        raise RuntimeError("Mooncake checkpoint manager is not installed")
    manager._checkpoint_workers = list(workers)


def run_checkpoint_command(body: Mapping[str, Any]) -> dict[str, Any] | None:
    """Execute an actor command using only its already-attached local store."""
    # TQ's public get_client() asserts when no client is attached. Read the
    # singleton so generation ranks without token capture can return None.
    from transfer_queue import interface as tq_interface

    client = tq_interface._TQ_CLIENT
    if client is None:
        return None
    manager = client.storage_manager
    if not getattr(type(manager), _PLUGIN_MARKER, False):
        return None
    participant = manager._checkpoint_participant
    return None if participant is None else participant._dispatch(body)


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
        def __init__(self, controller_info: Any, config: dict[str, Any]) -> None:
            if _checkpoint_enabled(config):
                config = dict(config)
                if ray.get_runtime_context().get_actor_id() is None:
                    # A driver/task has no actor command endpoint. It may use
                    # Mooncake, but must not own otherwise unreachable payload.
                    # Keep the controller's published config unchanged so actors
                    # still mount their configured storage capacity.
                    config["global_segment_size"] = 0
            super().__init__(controller_info, config)
            self._checkpoint_workers: list[Any] = []
            self._checkpoint_participant: _CheckpointParticipant | None = None
            if _checkpoint_enabled(self.config):
                _validate_checkpoint_runtime(self)
                if self.config["global_segment_size"] > 0:
                    self._checkpoint_participant = _CheckpointParticipant(self)

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

    CheckpointMooncakeStorageManager.__name__ = "CheckpointMooncakeStorageManager"
    setattr(CheckpointMooncakeStorageManager, _PLUGIN_MARKER, True)
    manager_registry["MooncakeStore"] = CheckpointMooncakeStorageManager


__all__ = [
    "configure_checkpoint_workers",
    "install_tq_mooncake_checkpoint_plugin",
    "run_checkpoint_command",
]
