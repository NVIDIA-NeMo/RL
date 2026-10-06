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
"""Checkpoint save / restore for the NixlStore backend (milestone A2).

TransferQueue saves the controller first and storage second, and restores
storage first. Storage state here is just *the bytes of every live blob plus
its live-entry count*. Placement is deliberately not saved: on restore the
blobs are re-placed on whatever units are ACTIVE now (any count, any nodes)
and the BlobDirectory is re-pointed. That is why the controller's location
meta ``{"b","o","n","k","d","s"}`` carries no unit id.

Layout under ``<checkpoint_dir>/nixl_store/``::

    manifest.json        {"version": 1, "store": "unit" | "file", "shards": [...]}
    unit-<id>.shard      live blobs of one unit, back to back          (unit store)
    files/<blob>.blob    copies of the blob files                       (file store)

Blobs are copied with plain file I/O on the unit (sequential dump of a pinned
slab; a few GB/s per unit on Lustre). A NIXL POSIX/GDS loopback path can
replace it later without changing the layout.

Callers must be quiescent: TQ's checkpoint runs under NeMo-RL's lifecycle
guard, so no put/get/clear is in flight while a shard is written.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import ray

from nemo_rl.data_plane.nixl.blobstore import FileStore, UnitSlabStore
from nemo_rl.data_plane.nixl.directory import ACTIVE
from nemo_rl.data_plane.nixl.errors import UnitFull

SUBDIR = "nixl_store"
MANIFEST = "manifest.json"
VERSION = 1


def _root(checkpoint_dir: str | os.PathLike) -> Path:
    return Path(checkpoint_dir) / SUBDIR


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ============================================================================ save
def save_storage_checkpoint(
    client: Any, checkpoint_dir: str | os.PathLike
) -> dict[str, Any]:
    """Dump every live blob reachable from ``client.store``; return the manifest."""
    root = _root(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    store = client.store
    if isinstance(store, UnitSlabStore):
        manifest = _save_units(store, root)
    elif isinstance(store, FileStore):
        manifest = _save_files(store, root)
    else:
        raise NotImplementedError(f"{type(store).__name__} does not support checkpoint")
    _write_json(root / MANIFEST, manifest)
    return manifest


def _unit_actor(store: UnitSlabStore, unit_id: int):
    """Checkpointing is an admin path, so it reaches units through Ray directly."""
    return ray.get_actor(f"NixlStorageUnit#{unit_id}", namespace=store.namespace)


def _active_units(store: UnitSlabStore) -> list[int]:
    store._refresh_units()
    return sorted(
        int(u["unit_id"])
        for u in store._units.values()
        if u.get("state", ACTIVE) == ACTIVE
    )


def _save_units(store: UnitSlabStore, root: Path) -> dict[str, Any]:
    units = _active_units(store)
    futs = {
        u: _unit_actor(store, u).save_shard.remote(str(root / f"unit-{u}.shard"))
        for u in units
    }
    shards = []
    for u, f in futs.items():
        r = ray.get(f)
        shards.append(
            {
                "unit_id": u,
                "file": f"unit-{u}.shard",
                "nbytes": r["nbytes"],
                "blobs": r["blobs"],
            }
        )
    return {"version": VERSION, "store": "unit", "shards": shards}


def _save_files(store: FileStore, root: Path) -> dict[str, Any]:
    snap = ray.get(store.directory.snapshot.remote())
    files = root / "files"
    files.mkdir(exist_ok=True)
    blobs: dict[str, list[int]] = {}
    for b, u in snap["blob_unit"].items():
        if int(u) != -1:
            continue
        shutil.copyfile(store.path(b), files / f"{b}.blob")
        blobs[b] = [int(snap["refs"].get(b, 0))]
    return {
        "version": VERSION,
        "store": "file",
        "root": store.root,
        "shards": [{"unit_id": -1, "file": "files", "blobs": blobs}],
    }


# ============================================================================ load
def load_storage_checkpoint(
    client: Any, checkpoint_dir: str | os.PathLike
) -> dict[str, Any]:
    """Re-place every blob of the manifest on the units alive now; return the manifest."""
    root = _root(checkpoint_dir)
    with open(root / MANIFEST) as f:
        manifest = json.load(f)
    if manifest.get("version") != VERSION:
        raise ValueError(
            f"unsupported nixl_store checkpoint version {manifest.get('version')!r}"
        )
    store = client.store
    kind = (
        "unit"
        if isinstance(store, UnitSlabStore)
        else "file"
        if isinstance(store, FileStore)
        else None
    )
    if kind != manifest["store"]:
        raise ValueError(
            f"checkpoint was taken with store={manifest['store']!r}, this client runs {kind!r}"
        )
    if kind == "unit":
        _load_units(store, root, manifest)
    else:
        _load_files(store, root, manifest)
    return manifest


def _load_units(store: UnitSlabStore, root: Path, manifest: dict[str, Any]) -> None:
    units = _active_units(store)
    if not units:
        raise RuntimeError("no ACTIVE NixlStorageUnit to restore into")
    placed: list[tuple[str, int]] = []
    # Shard i starts on unit i (mod n); whatever does not fit spills to the next
    # unit in order, so N units restore onto N±k units without a planner.
    for i, shard in enumerate(manifest["shards"]):
        pending = dict(shard["blobs"])
        path = str(root / shard["file"])
        order = units[i % len(units) :] + units[: i % len(units)]
        for u in order:
            if not pending:
                break
            r = ray.get(_unit_actor(store, u).load_shard.remote(path, pending))
            placed.extend((b, u) for b in r["loaded"])
            pending = {b: pending[b] for b in r["failed"]}
        if pending:
            raise UnitFull(
                f"{len(pending)} blob(s) of shard {shard['file']} fit on no ACTIVE unit"
            )
    ray.get(store.directory.put_blobs.remote(placed))
    for b, u in placed:
        store._blob_unit[b] = u


def _load_files(store: FileStore, root: Path, manifest: dict[str, Any]) -> None:
    os.makedirs(store.root, exist_ok=True)
    items: list[tuple[str, int, int]] = []
    for shard in manifest["shards"]:
        for b, (refs,) in shard["blobs"].items():
            dst = store.path(b)
            if not os.path.exists(dst):
                shutil.copyfile(root / shard["file"] / f"{b}.blob", dst)
            items.append((b, -1, int(refs)))
    ray.get(store.directory.put_blobs.remote(items))
