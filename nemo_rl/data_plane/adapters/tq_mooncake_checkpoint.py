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

Mooncake already has the data-path primitive this needs.  When
``mooncake_master`` is started with ``--root_fs_dir``, every producer client
writes a DISK replica of its own PUT to that shared filesystem.  This module
connects those replicas to TransferQueue's existing storage checkpoint hooks:

* checkpoint-enabled PUTs wait for their DISK replica before returning;
* the TQ storage manager copies controller-referenced replicas into the TQ
  checkpoint and writes a small manifest; and
* restore upserts the saved raw bytes before TQ restores controller metadata.

The caller must keep writers quiescent while ``tq.save_checkpoint`` captures
the controller and copies its referenced files.  Higher-level checkpoint
policy and recovery decisions intentionally remain outside this module.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import mmap
import pickle
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PLUGIN_MARKER = "_nemo_rl_mooncake_checkpoint_v1"
_STORAGE_DIR = "mooncake_storage"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_VERSION = 1


def _checkpoint_settings(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("MooncakeStore config must be a mapping")
    checkpoint = config.get("checkpoint") or {}
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MooncakeStore.checkpoint must be a mapping")
    return checkpoint


def _checkpoint_enabled(config: Any) -> bool:
    return _checkpoint_settings(config).get("enabled") is True


def _storage_root(config: Any) -> Path:
    raw = _checkpoint_settings(config).get("storage_root")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Mooncake checkpoint storage_root must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"Mooncake checkpoint storage_root must be absolute: {raw!r}")
    return path.resolve()


def _durability_timeout_s(config: Any) -> float:
    return float(_checkpoint_settings(config)["durability_timeout_s"])


def _poll_interval_s(config: Any) -> float:
    return float(_checkpoint_settings(config)["poll_interval_s"])


def _storage_layout(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("MooncakeStore config must be a mapping")
    return {
        "use_gdr": bool(config["use_gdr"]),
        "gdr_staging_buffer_mb": int(config["gdr_staging_buffer_mb"]),
    }


def _status_name(status: Any) -> str:
    name = getattr(status, "name", None)
    return name if isinstance(name, str) else str(status).rsplit(".", 1)[-1]


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
class _DiskObject:
    key: str
    path: Path
    size: int


def _wait_for_disk_objects(
    store: Any,
    keys: Sequence[str],
    *,
    live_root: Path,
    timeout_s: float,
    poll_interval_s: float,
) -> list[_DiskObject]:
    """Wait until every key has a complete direct-DISK replica."""
    pending = set(keys)
    found: dict[str, _DiskObject] = {}
    path_owners: dict[Path, str] = {}
    deadline = time.monotonic() + timeout_s
    resolved_live_root = live_root.resolve()

    while pending:
        descriptors = store.batch_get_replica_desc(sorted(pending))
        if not isinstance(descriptors, Mapping):
            raise RuntimeError(
                "Mooncake batch_get_replica_desc returned malformed data"
            )

        for key in list(pending):
            for descriptor in descriptors.get(key, []):
                if (
                    not descriptor.is_disk_replica()
                    or _status_name(descriptor.status) != "COMPLETE"
                ):
                    continue
                disk = descriptor.get_disk_descriptor()
                path = Path(disk.file_path).resolve()
                size = int(disk.object_size)
                if not path.is_relative_to(resolved_live_root):
                    raise RuntimeError(
                        f"Mooncake DISK path for {key!r} is outside {resolved_live_root}: {path}"
                    )
                if not path.is_file() or path.stat().st_size != size:
                    continue
                owner = path_owners.get(path)
                if owner is not None and owner != key:
                    raise RuntimeError(
                        "Mooncake mapped distinct keys to the same DISK path: "
                        f"{owner!r}, {key!r} -> {path}"
                    )
                path_owners[path] = key
                found[key] = _DiskObject(key=key, path=path, size=size)
                pending.remove(key)
                break

        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for Mooncake DISK replicas: "
                    + ", ".join(sorted(pending))
                )
            time.sleep(poll_interval_s)

    return [found[key] for key in sorted(found)]


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


def _validate_checkpoint_paths(checkpoint_root: Path, storage_root: Path) -> None:
    # TQ passes ``<destination>.tmp`` to the storage manager, then removes the
    # old destination before renaming the temporary directory into place.
    suffix = ".tmp"
    destination = checkpoint_root
    if destination.name.endswith(suffix):
        destination = destination.with_name(destination.name[: -len(suffix)])
    destination = destination.resolve()
    storage_root = storage_root.resolve()
    live_root = storage_root / "live"

    if storage_root.is_relative_to(destination):
        raise ValueError(
            "Mooncake checkpoint storage_root must not equal or be nested "
            f"inside the TQ checkpoint destination: {destination}"
        )
    if destination.is_relative_to(live_root):
        raise ValueError(
            "TQ checkpoint destination must not be nested inside Mooncake's "
            f"active live directory: {live_root}"
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as payload:
        return hashlib.file_digest(payload, "sha256").hexdigest()


def _save_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    checkpoint_root = Path(checkpoint_dir)
    config = manager.config
    storage_root = _storage_root(config)
    _validate_checkpoint_paths(checkpoint_root, storage_root)
    keys = _controller_keys(checkpoint_root)
    live_root = storage_root / "live"
    objects = _wait_for_disk_objects(
        manager.storage_client._store,
        keys,
        live_root=live_root,
        timeout_s=_durability_timeout_s(config),
        poll_interval_s=_poll_interval_s(config),
    )

    storage_dir = checkpoint_root / _STORAGE_DIR
    payload_dir = storage_dir / "objects"
    payload_dir.mkdir(parents=True, exist_ok=False)
    manifest_objects: list[dict[str, Any]] = []
    for index, obj in enumerate(objects):
        relative = Path("objects") / f"{index:08d}.bin"
        destination = storage_dir / relative
        shutil.copyfile(obj.path, destination)
        if destination.stat().st_size != obj.size:
            raise OSError(f"Short checkpoint copy for Mooncake key {obj.key!r}")
        manifest_objects.append(
            {
                "key": obj.key,
                "path": relative.as_posix(),
                "size": obj.size,
                "sha256": _sha256(destination),
            }
        )

    manifest = {
        "version": _MANIFEST_VERSION,
        "storage_layout": _storage_layout(config),
        "objects": manifest_objects,
    }
    (storage_dir / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


@dataclass(frozen=True)
class _ManifestObject:
    key: str
    path: Path
    size: int


def _load_manifest(
    checkpoint_root: Path,
) -> tuple[list[_ManifestObject], dict[str, Any]]:
    storage_dir = checkpoint_root / _STORAGE_DIR
    manifest_path = storage_dir / _MANIFEST_FILE
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
    resolved_storage_dir = storage_dir.resolve()
    for raw in raw_objects:
        if not isinstance(raw, Mapping):
            raise ValueError("Malformed Mooncake checkpoint object")
        key = raw.get("key")
        relative = raw.get("path")
        size = raw.get("size")
        digest = raw.get("sha256")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"Invalid or duplicate Mooncake checkpoint key: {key!r}")
        if not isinstance(relative, str) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Malformed Mooncake checkpoint entry for {key!r}")
        path = (storage_dir / relative).resolve()
        if not path.is_relative_to(resolved_storage_dir) or not path.is_file():
            raise ValueError(
                f"Unsafe or missing Mooncake checkpoint payload for {key!r}"
            )
        if (
            path.stat().st_size != size
            or not isinstance(digest, str)
            or _sha256(path) != digest
        ):
            raise ValueError(f"Corrupt Mooncake checkpoint payload for {key!r}")
        seen.add(key)
        entries.append(_ManifestObject(key=key, path=path, size=size))
    return entries, layout


def _restore_object(store: Any, entry: _ManifestObject, replica_config: Any) -> None:
    """Upsert one payload without copying it through Mooncake's local buffer."""
    with entry.path.open("rb") as payload_file:
        with mmap.mmap(
            payload_file.fileno(), entry.size, access=mmap.ACCESS_COPY
        ) as payload:
            # ctypes' stub rejects the object returned by its own from_buffer.
            # pyrefly: ignore  # bad-argument-type
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(payload))
            register_result = store.register_buffer(pointer, entry.size)
            if register_result != 0:
                raise RuntimeError(
                    f"Mooncake restore buffer registration failed for {entry.key!r}: "
                    f"{register_result}"
                )
            try:
                result = store.upsert_from(
                    entry.key, pointer, entry.size, replica_config
                )
            finally:
                unregister_result = store.unregister_buffer(pointer)

    if result != 0:
        raise RuntimeError(f"Mooncake restore failed for {entry.key!r}: {result}")
    if unregister_result != 0:
        raise RuntimeError(
            f"Mooncake restore buffer cleanup failed for {entry.key!r}: "
            f"{unregister_result}"
        )


def _load_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    checkpoint_root = Path(checkpoint_dir)
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

    for entry in entries:
        _restore_object(store, entry, manager.storage_client.replica_config)

    config = manager.config
    _wait_for_disk_objects(
        store,
        [entry.key for entry in entries],
        live_root=_storage_root(config) / "live",
        timeout_s=_durability_timeout_s(config),
        poll_interval_s=_poll_interval_s(config),
    )


def _server_address(value: str, *, name: str) -> tuple[str, str]:
    parsed = urlparse(value if "://" in value else f"//{value}")
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"Invalid {name}: {value!r}")
    return parsed.hostname, str(parsed.port)


def _master_command(
    *,
    executable: str,
    metadata_server: str,
    master_server_address: str,
    live_root: Path,
    session_id: str,
) -> list[str]:
    master_host, master_port = _server_address(
        master_server_address, name="master_server_address"
    )
    command = [
        executable,
        "-client_ttl=30",
        "-default_kv_lease_ttl=999999",
        "-default_kv_soft_pin_ttl=999999",
        "--allow_evict_soft_pinned_objects=false",
        f"--rpc_address={master_host}",
        f"--rpc_port={master_port}",
        "--enable_offload=false",
        "--enable_disk_eviction=false",
        f"--root_fs_dir={live_root}",
        f"--cluster_id={session_id}",
    ]
    if metadata_server.strip().upper() != "P2PHANDSHAKE":
        metadata_host, metadata_port = _server_address(
            metadata_server, name="metadata_server"
        )
        command.extend(
            [
                "--enable_http_metadata_server=true",
                f"--http_metadata_server_host={metadata_host}",
                f"--http_metadata_server_port={metadata_port}",
            ]
        )
    return command


@dataclass
class MooncakeCheckpointMaster:
    """Exact plugin-owned Mooncake master process."""

    process: subprocess.Popen[str]
    log_path: Path

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def start_mooncake_checkpoint_master(
    *,
    executable: str,
    metadata_server: str,
    master_server_address: str,
    storage_root: str,
) -> MooncakeCheckpointMaster:
    """Start one master that enables producer-side direct DISK replicas."""
    root = Path(storage_root)
    if not root.is_absolute():
        raise ValueError(
            f"Mooncake checkpoint storage_root must be absolute: {storage_root!r}"
        )
    root = root.resolve()
    live_root = root / "live"
    log_root = root / "logs"
    live_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    log_path = log_root / f"{session_id}.log"
    command = _master_command(
        executable=executable,
        metadata_server=metadata_server,
        master_server_address=master_server_address,
        live_root=live_root,
        session_id=session_id,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    time.sleep(3)
    if process.poll() is not None:
        output = log_path.read_text(encoding="utf-8")
        raise RuntimeError(f"mooncake_master exited during startup:\n{output}")
    return MooncakeCheckpointMaster(process=process, log_path=log_path)


def install_tq_mooncake_checkpoint_plugin() -> None:
    """Install the checkpoint manager and per-PUT durability fence once."""
    from transfer_queue.storage.clients import mooncake_client
    from transfer_queue.storage.managers import mooncake_manager
    from transfer_queue.storage.managers.base import StorageManagerFactory

    manager_registry = StorageManagerFactory._registry
    current_manager = manager_registry.get("MooncakeStore")
    if getattr(current_manager, _PLUGIN_MARKER, False):
        return
    if current_manager is not mooncake_manager.MooncakeStorageManager:
        raise RuntimeError("Unexpected TQ MooncakeStore manager registration")

    original_upsert = mooncake_client.MooncakeStoreClient._batch_upsert_with_retry
    if not getattr(original_upsert, _PLUGIN_MARKER, False):

        def durable_upsert(
            client: Any,
            keys: list[str],
            pointers: list[int],
            sizes: list[int],
        ) -> None:
            original_upsert(client, keys, pointers, sizes)
            if not _checkpoint_enabled(client.config):
                return
            _wait_for_disk_objects(
                client._store,
                keys,
                live_root=_storage_root(client.config) / "live",
                timeout_s=_durability_timeout_s(client.config),
                poll_interval_s=_poll_interval_s(client.config),
            )

        setattr(durable_upsert, _PLUGIN_MARKER, True)
        setattr(
            mooncake_client.MooncakeStoreClient,
            "_batch_upsert_with_retry",
            durable_upsert,
        )

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


__all__ = [
    "MooncakeCheckpointMaster",
    "install_tq_mooncake_checkpoint_plugin",
    "start_mooncake_checkpoint_master",
]
