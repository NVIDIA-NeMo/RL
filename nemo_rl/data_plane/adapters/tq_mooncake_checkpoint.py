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
"""Opt-in Mooncake storage checkpoints for the pinned TransferQueue.

Normal Mooncake PUTs remain memory-only.  When TransferQueue explicitly calls
the storage checkpoint hook, this module reads every controller-referenced raw
Mooncake object into the checkpoint directory and writes a verified manifest.
Restore upserts those raw bytes into a fresh Mooncake store before TQ restores
controller metadata.

The caller must keep writers and clears quiescent while ``tq.save_checkpoint``
captures the controller and reads its referenced objects.  Higher-level
checkpoint policy and recovery decisions intentionally remain outside this
module.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import mmap
import os
import pickle
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_PLUGIN_MARKER = "_nemo_rl_mooncake_checkpoint_v2"
_STORAGE_DIR = "mooncake_storage"
_PAYLOAD_FILE = "objects.bin"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_VERSION = 2
_UNREGISTER_ATTEMPTS = 3

# An mmap must outlive its Mooncake registration. On a persistent unregister
# failure, retain it until process teardown instead of letting Python unmap
# memory that Mooncake or the NIC may still reference.
_QUARANTINED_BUFFERS: list[mmap.mmap] = []


def _checkpoint_settings(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("MooncakeStore config must be a mapping")
    checkpoint = config.get("checkpoint") or {}
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MooncakeStore.checkpoint must be a mapping")
    return checkpoint


def _checkpoint_enabled(config: Any) -> bool:
    return _checkpoint_settings(config).get("enabled") is True


def _storage_layout(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("MooncakeStore config must be a mapping")
    return {
        "use_gdr": bool(config["use_gdr"]),
        "gdr_staging_buffer_mb": int(config["gdr_staging_buffer_mb"]),
    }


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


def _save_object(store: Any, obj: _StoredObject, output: Any) -> str:
    """GET one Mooncake object and append its raw bytes to a payload shard."""
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


def _save_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    checkpoint_root = Path(checkpoint_dir)
    config = manager.config
    keys = _controller_keys(checkpoint_root)
    store = manager.storage_client._store
    objects = _stored_objects(store, keys)

    storage_dir = checkpoint_root / _STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=False)
    manifest_objects: list[dict[str, Any]] = []
    offset = 0
    with (storage_dir / _PAYLOAD_FILE).open("xb") as payload_file:
        for obj in objects:
            digest = _save_object(store, obj, payload_file)
            manifest_objects.append(
                {
                    "key": obj.key,
                    "offset": offset,
                    "size": obj.size,
                    "sha256": digest,
                }
            )
            offset += obj.size
        payload_file.flush()
        os.fsync(payload_file.fileno())

    manifest = {
        "version": _MANIFEST_VERSION,
        "storage_layout": _storage_layout(config),
        "objects": manifest_objects,
    }
    with (storage_dir / _MANIFEST_FILE).open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())


@dataclass(frozen=True)
class _ManifestObject:
    key: str
    offset: int
    size: int
    sha256: str


def _load_manifest(
    checkpoint_root: Path,
) -> tuple[list[_ManifestObject], dict[str, Any], Path]:
    storage_dir = checkpoint_root / _STORAGE_DIR
    manifest_path = storage_dir / _MANIFEST_FILE
    payload_path = storage_dir / _PAYLOAD_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("version") != _MANIFEST_VERSION
    ):
        raise ValueError("Unsupported Mooncake checkpoint manifest")
    raw_objects = manifest.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("Mooncake checkpoint manifest has no object list")
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

    entries: list[_ManifestObject] = []
    seen: set[str] = set()
    expected_offset = 0
    for raw in raw_objects:
        if not isinstance(raw, Mapping):
            raise ValueError("Malformed Mooncake checkpoint object")
        key = raw.get("key")
        offset = raw.get("offset")
        size = raw.get("size")
        digest = raw.get("sha256")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"Invalid or duplicate Mooncake checkpoint key: {key!r}")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset != expected_offset
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"Malformed Mooncake checkpoint entry for {key!r}")
        seen.add(key)
        entries.append(
            _ManifestObject(key=key, offset=offset, size=size, sha256=digest)
        )
        expected_offset += size

    if not payload_path.is_file() or payload_path.stat().st_size != expected_offset:
        raise ValueError("Missing or corrupt Mooncake checkpoint payload shard")
    return entries, layout, payload_path


def _restore_object(
    store: Any, entry: _ManifestObject, replica_config: Any, payload_file: Any
) -> None:
    """Read one checkpoint payload and upsert it into Mooncake memory."""
    buffer = _CheckpointBuffer.allocate(entry.size)
    try:
        payload_file.seek(entry.offset)
        view = memoryview(buffer.payload)
        try:
            offset = 0
            while offset < entry.size:
                read = payload_file.readinto(view[offset : entry.size])
                if read is None or read <= 0:
                    raise OSError(
                        f"Short checkpoint read for Mooncake key {entry.key!r}"
                    )
                offset += read
        finally:
            view.release()
        if hashlib.sha256(buffer.payload).hexdigest() != entry.sha256:
            raise ValueError(f"Corrupt Mooncake checkpoint payload for {entry.key!r}")

        # Register anonymous memory rather than a Lustre-backed mmap. RDMA
        # registration of network-filesystem mappings is not a supported
        # Mooncake contract.
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


def _load_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    checkpoint_root = Path(checkpoint_dir)
    entries, saved_layout, payload_path = _load_manifest(checkpoint_root)
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

    with payload_path.open("rb") as payload_file:
        for entry in entries:
            _restore_object(
                store, entry, manager.storage_client.replica_config, payload_file
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


__all__ = ["install_tq_mooncake_checkpoint_plugin"]
