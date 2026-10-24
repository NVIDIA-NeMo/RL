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
"""Checkpoint save / restore for the NixlStore backend (``NixlStorageManager``).

TransferQueue saves the controller first and storage second, and restores
storage first. Storage state is each unit's live blobs plus, per blob, the
``gidx@field`` keys its row index points at it with. Rows are index-addressed
(row ``gidx`` lives on unit ``unit_ids[gidx % N]``), so:

- same unit count: unit ``k`` reloads ``unit-k.shard`` into itself;
- a different count: the driver re-puts every live key through the normal put
  path, which routes it to its new owner (slower: all bytes pass the driver).

Layout under ``<checkpoint_dir>/nixl_store/``::

    manifest.json        {"version": 2, "unit_ids": [...], "shards": [...]}
    unit-<id>.shard      live blobs of one unit, back to back
    unit-<id>.shard.rows msgpack {blob_id: [live gidx@field keys]}

Version 1 checkpoints (locations kept in the TQ controller) are rejected.

Callers must be quiescent: TQ's checkpoint runs under NeMo-RL's lifecycle
guard, so no put/get/clear is in flight while a shard is written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import msgspec
import ray

from nemo_rl.data_plane.nixl import blob_format
from nemo_rl.data_plane.nixl.blobstore import UnitSlabStore
from nemo_rl.data_plane.nixl.directory import ACTIVE
from nemo_rl.data_plane.nixl.errors import UnitFull

SUBDIR = "nixl_store"
MANIFEST = "manifest.json"
VERSION = 2


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
    manifest = _save_units(client.store, root)
    manifest["unit_ids"] = list(client.unit_ids)
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
                "clears_during_save": int(r.get("clears_during_save", 0)),
            }
        )
    return {"version": VERSION, "shards": shards}


# ============================================================================ load
def load_storage_checkpoint(
    client: Any, checkpoint_dir: str | os.PathLike
) -> dict[str, Any]:
    """Restore every live key of the manifest onto its owner unit now; return the manifest."""
    root = _root(checkpoint_dir)
    with open(root / MANIFEST) as f:
        manifest = json.load(f)
    if manifest.get("version") != VERSION:
        raise ValueError(
            f"unsupported nixl_store checkpoint version {manifest.get('version')!r} "
            f"(this build reads version {VERSION})"
        )
    if list(manifest["unit_ids"]) == list(client.unit_ids):
        _load_in_place(client.store, root, manifest)
    else:
        _load_rerouted(client, root, manifest)
    return manifest


def _load_in_place(store: UnitSlabStore, root: Path, manifest: dict[str, Any]) -> None:
    """Same unit table: unit ``k`` reloads its own shard (RAM first, then its SSD tier)."""
    shards = {int(s["unit_id"]): s for s in manifest["shards"]}
    pending = {u: dict(s["blobs"]) for u, s in shards.items()}
    for spill in (False, True):
        futs = {
            u: _unit_actor(store, u).load_shard.remote(
                str(root / shards[u]["file"]), pending[u], spill
            )
            for u in shards
            if pending[u]
        }
        for u, f in futs.items():
            r = ray.get(f)
            pending[u] = {b: pending[u][b] for b in r["failed"]}
    for u, left in pending.items():
        if left:
            raise UnitFull(
                f"{len(left)} blob(s) of {shards[u]['file']} fit on unit {u}"
            )


def _load_rerouted(client: Any, root: Path, manifest: dict[str, Any]) -> None:
    """A different unit table: re-put each shard's live keys through the client."""
    for s in manifest["shards"]:
        path = str(root / s["file"])
        with open(path + ".rows", "rb") as rf:
            live = {b: set(ks) for b, ks in msgspec.msgpack.decode(rf.read()).items()}
        with open(path, "rb") as f:
            for b, (off, nbytes, _refs) in s["blobs"].items():
                keep = live.get(b, set())
                if not keep:
                    continue
                f.seek(int(off))
                blob = memoryview(bytearray(f.read(int(nbytes))))
                entries = [e for e in blob_format.read_index(blob) if e.key in keep]
                values = [
                    blob_format.materialize(
                        blob_format.decode_entry(blob[e.off : e.off + e.len], e)
                    )
                    for e in entries
                ]
                client.put([e.key for e in entries], values)
