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

import contextlib
import copy
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import ray

from nemo_rl.data_plane.adapters.tq_simple_checkpoint import _wait_all
from nemo_rl.data_plane.checkpoint import CheckpointCapture, CheckpointReleaseError
from nemo_rl.data_plane.nixl import blob_format
from nemo_rl.data_plane.nixl.blobstore import UnitSlabStore
from nemo_rl.data_plane.nixl.directory import ACTIVE
from nemo_rl.data_plane.nixl.errors import UnitFull
from nemo_rl.telemetry.instrumentation import managed_span
from nemo_rl.telemetry.span_groups import RLSpanGroup

log = logging.getLogger(__name__)

SUBDIR = "nixl_store"
MANIFEST = "manifest.json"
# Checkpoint layout version: written to manifest.json, and load rejects any other
# value. Bump it when an older build could not read a new checkpoint correctly:
# a manifest key added, renamed or given a new meaning; a change to how
# unit-<id>.shard or its .rows file is laid out; or a change to where rows live
# (row gidx on unit_ids[gidx % N]). Changes inside one blob are covered by
# blob_format.VERSION instead.
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


def _controller_actor():
    return ray.get_actor("TransferQueueController", namespace="transfer_queue")


def _active_units(store: UnitSlabStore, *, timeout_s: float | None = None) -> list[int]:
    """Active unit ids; ``timeout_s`` bounds the directory query (GetTimeoutError)."""
    store._refresh_units(timeout_s=timeout_s)
    return sorted(
        int(u["unit_id"])
        for u in store._units.values()
        if u.get("state", ACTIVE) == ACTIVE
    )


def _shard_entry(unit_id: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "file": f"unit-{unit_id}.shard",
        "nbytes": result["nbytes"],
        "blobs": result["blobs"],
        "clears_during_save": int(result.get("clears_during_save", 0)),
    }


def _save_units(store: UnitSlabStore, root: Path) -> dict[str, Any]:
    units = _active_units(store)
    futs = {
        u: _unit_actor(store, u).save_shard.remote(str(root / f"unit-{u}.shard"))
        for u in units
    }
    shards = [_shard_entry(u, ray.get(f)) for u, f in futs.items()]
    return {"version": VERSION, "shards": shards}


# ============================================================================ two-phase save
@dataclass
class NixlCheckpointCapture(CheckpointCapture):
    """One held cut per unit; ``persist`` writes the files of a synchronous save."""

    checkpoint_dir: Path
    token: str
    controller: Any  # holds the controller bytes (see tq_controller_checkpoint)
    unit_ids: list[int]  # the client's routing table, saved in the manifest
    units: dict[int, Any]  # unit id -> actor
    metadata: dict[str, Any]
    timeout_s: float
    _persist_started: bool = field(default=False, init=False, repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _persist_refs: list[Any] = field(default_factory=list, init=False, repr=False)
    _operation_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @contextlib.contextmanager
    def _operation(self) -> Iterator[float]:
        deadline = time.monotonic() + self.timeout_s
        if not self._operation_lock.acquire(timeout=self.timeout_s):
            raise TimeoutError(
                "A checkpoint persist/release operation is still running"
            )
        try:
            yield deadline
        finally:
            self._operation_lock.release()

    def persist(self) -> None:
        with self._operation() as deadline:
            if self._released:
                raise RuntimeError("Cannot persist a released checkpoint capture")
            if self._persist_started:
                raise RuntimeError(
                    "persist() is single-attempt: release the capture, discard the "
                    "checkpoint directory and capture again"
                )
            self._persist_started = True
            root = _root(self.checkpoint_dir)
            refs = self._persist_refs
            try:
                refs.append(
                    self.controller.nrl_persist_snapshot.remote(
                        self.token, str(self.checkpoint_dir / "controller_state.pkl")
                    )
                )
                for u, actor in self.units.items():
                    refs.append(
                        actor.persist_shard.remote(
                            self.token, str(root / f"unit-{u}.shard")
                        )
                    )
            finally:
                # Partial submission still leaves owned writes to settle or release.
                # Unpack after this block: with no ref submitted, unpacking here
                # would raise ValueError and hide the submission error.
                values = _wait_all(
                    refs, timeout_s=max(0.0, deadline - time.monotonic())
                )
            _, *results = values
            shards = [
                _shard_entry(u, r) for u, r in zip(self.units, results, strict=True)
            ]
            _write_json(
                root / MANIFEST,
                {"version": VERSION, "shards": shards, "unit_ids": self.unit_ids},
            )
            # The completion marker is last, as in TQ's own save.
            _write_json(
                self.checkpoint_dir / "metadata.json",
                {"storage_saved": True, "user_metadata": self.metadata},
            )

    def release(self) -> None:
        with self._operation() as deadline:
            if self._released:
                return
            # Completed write failures were reported by persist; holds can retire.
            # A wait expiry is different: those RPCs must finish before release.
            _wait_all(
                self._persist_refs,
                timeout_s=max(0.0, deadline - time.monotonic()),
                raise_errors=False,
            )
            refs = []
            first_error: Exception | None = None
            methods = [
                self.controller.nrl_release_snapshot,
                *(a.release_shard for a in self.units.values()),
            ]
            for method in methods:
                try:
                    refs.append(method.remote(self.token))
                except Exception as error:  # noqa: BLE001 - re-raised below
                    first_error = first_error or error
            try:
                _wait_all(refs, timeout_s=max(0.0, deadline - time.monotonic()))
            except Exception as error:  # noqa: BLE001 - re-raised below
                first_error = first_error or error
            if first_error is not None:
                raise first_error
            self._released = True
            self._persist_refs.clear()


def capture_nixl_checkpoint(
    client: Any,
    checkpoint_dir: str | os.PathLike,
    *,
    metadata: dict[str, Any] | None,
    timeout_s: float,
) -> NixlCheckpointCapture:
    """Capture the controller and hold every unit's blobs; ``persist`` writes them later.

    The controller side is the SimpleStorage one (``tq_controller_checkpoint``), so
    ``configure_controller_checkpoint`` must have run before ``tq.init``. The caller
    must keep puts and clears out until this returns. After that it may put new
    keys, call ``persist``, and always call ``release``. It must not update or clear
    a key of the cut before ``release``: ``persist`` would raise.
    """
    meta = copy.deepcopy(metadata) if metadata is not None else {}
    json.dumps(meta)  # fail before touching disk
    storage = client.storage_manager.storage_client
    directory = Path(checkpoint_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)  # never overwrite a saved cut
    _root(directory).mkdir()
    # The caller holds its mutation barrier, so a frozen actor must not stall it:
    # one budget covers the unit lookup and every capture RPC.
    deadline = time.monotonic() + timeout_s
    try:
        active = _active_units(storage.store, timeout_s=timeout_s)
    except ray.exceptions.GetTimeoutError as error:
        raise TimeoutError(
            "nixl unit lookup timed out before any hold was taken"
        ) from error
    units = {u: _unit_actor(storage.store, u) for u in active}
    capture = NixlCheckpointCapture(
        directory,
        uuid.uuid4().hex,
        _controller_actor(),
        list(storage.unit_ids),
        units,
        meta,
        timeout_s,
    )
    try:
        _wait_all(
            [capture.controller.nrl_capture_snapshot.remote(capture.token)],
            timeout_s=max(0.0, deadline - time.monotonic()),
        )
        _wait_all(
            [a.capture_shard.remote(capture.token) for a in units.values()],
            timeout_s=max(0.0, deadline - time.monotonic()),
        )
        return capture
    except BaseException:
        # Release on every unit: one that did not capture ignores the token.
        # A unit runs one method at a time, in submission order (its actor sets
        # no max_concurrency), so this release runs after a capture_shard that
        # timed out but is still queued or running, and drops the hold it makes.
        try:
            capture.release()
        except Exception as error:
            raise CheckpointReleaseError(
                f"Cannot confirm release of nixl capture {capture.token}; "
                "storage actors may still retain it, so stop checkpointing"
            ) from error
        raise


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
    in_place = list(manifest["unit_ids"]) == list(client.unit_ids)
    attrs: dict[str, Any] = {
        "rl.data_plane.restore_path": "in_place" if in_place else "rerouted",
        "rl.data_plane.saved_units": len(manifest["unit_ids"]),
        "rl.data_plane.current_units": len(client.unit_ids),
        "rl.data_plane.bytes": sum(int(s["nbytes"]) for s in manifest["shards"]),
    }
    t0 = time.monotonic()
    with managed_span(RLSpanGroup.DATA_PLANE, "rl.data_plane.restore_shards", **attrs):
        if in_place:
            _load_in_place(client.store, root, manifest)
        else:
            _load_rerouted(client, root, manifest)
    log.info("nixl_store restore %s took %.1fs", attrs, time.monotonic() - t0)
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
                f"{len(left)} blob(s) of {shards[u]['file']} do not fit on unit {u}"
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
