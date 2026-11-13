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
"""NeMo-RL's two-phase compatibility layer for TQ 0.1.11 SimpleStorage.

Only storage dictionaries are copied during capture, not their tensor payloads.
The audited StorageUnitData API replaces/removes values; it never modifies a
stored value in place or exposes it to another in-process caller. Strong
references therefore preserve the cut across subsequent PUT/CLEAR operations.
The shared lock excludes the actor's ZMQ mutation thread while copying indexes.

Persistence runs on the Ray actor thread, concurrently with the ZMQ worker,
and writes TQ's ordinary checkpoint layout. No tensor payload travels via Ray.
SSD/hybrid storage is deliberately unsupported. Re-audit this contract when
updating TQ; this module must not silently run against another implementation.
"""

from __future__ import annotations

import copy
import contextlib
import json
import logging
import pickle
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ray
import transfer_queue.interface as interface
from transfer_queue.storage.bootstrap import simple_storage_bootstrap as bootstrap
from transfer_queue.storage.simple_storage import StorageUnitData

from nemo_rl.data_plane.adapters.tq_controller_checkpoint import (
    configure_controller_checkpoint,
)
from nemo_rl.data_plane.adapters.tq_controller_patch import assert_stock_tq_version
from nemo_rl.data_plane.checkpoint import CheckpointCapture, CheckpointReleaseError

LOGGER = logging.getLogger(__name__)


def _wait_all(
    refs: list[Any], *, timeout_s: float | None = None, raise_errors: bool = True
) -> list[Any]:
    """Observe every RPC within one budget; timeout never cancels remote work."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    results = []
    first_error: Exception | None = None
    first_timeout: Exception | None = None
    for ref in refs:
        try:
            if deadline is None:
                results.append(ray.get(ref))
            else:
                results.append(
                    ray.get(ref, timeout=max(0.0, deadline - time.monotonic()))
                )
        except ray.exceptions.GetTimeoutError as error:
            if first_timeout is None:
                first_timeout = error
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_timeout is not None:
        raise TimeoutError(
            "TQ checkpoint RPC wait timed out; remote writes may still be running. "
            "Retry release() before deleting the checkpoint directory or capturing again."
        ) from first_timeout
    if first_error is not None and raise_errors:
        raise first_error
    return results


class SnapshotStorageMixin:
    """Added before the pinned storage actor base; ZMQ and Ray share one lock."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Must exist before the base constructor starts its worker thread.
        self._snapshot_lock = threading.RLock()
        self._snapshot_views: dict[str, dict[str, Any]] = {}
        super().__init__(*args, **kwargs)

    def _handle_put(self, data_parts: Any) -> Any:
        with self._snapshot_lock:
            return super()._handle_put(data_parts)

    def _handle_clear(self, data_parts: Any) -> Any:
        with self._snapshot_lock:
            return super()._handle_clear(data_parts)

    def _handle_load_checkpoint(self, data_parts: Any) -> Any:
        with self._snapshot_lock:
            if self._snapshot_views:
                raise RuntimeError("Cannot restore while a snapshot is retained")
            return super()._handle_load_checkpoint(data_parts)

    def nrl_capture_snapshot(self, token: str) -> str:
        """Capture on this actor; return only its native storage ID."""
        started = time.monotonic()
        with self._snapshot_lock:
            if self._snapshot_views:
                raise RuntimeError("Only one SimpleStorage snapshot may be retained")
            storage = self.storage_data
            if type(storage) is not StorageUnitData:
                raise TypeError("Async snapshots require in-memory StorageUnitData")
            self._snapshot_views[token] = {
                "storage_unit_id": self.storage_unit_id,
                "storage_unit_size": storage.storage_size,
                "field_data": {
                    name: dict(values) for name, values in storage.field_data.items()
                },
                "active_keys": set(storage._active_keys),
            }
        LOGGER.info(
            "simple_snapshot captured unit=%s seconds=%.6f",
            self.storage_unit_id,
            time.monotonic() - started,
        )
        return self.storage_unit_id

    def nrl_persist_snapshot(self, token: str, path: str) -> int:
        """Serialize retained values, without locking live PUT/CLEAR traffic."""
        with self._snapshot_lock:
            state = self._snapshot_views[token]
        started = time.monotonic()
        with open(path, "xb") as output:
            pickle.dump(state, output, protocol=pickle.HIGHEST_PROTOCOL)
        size = Path(path).stat().st_size
        LOGGER.info(
            "simple_snapshot persisted unit=%s seconds=%.6f bytes=%d",
            self.storage_unit_id,
            time.monotonic() - started,
            size,
        )
        return size

    def nrl_release_snapshot(self, token: str) -> None:
        """Idempotent; ordered behind persistence RPCs on this Ray actor."""
        with self._snapshot_lock:
            state = self._snapshot_views.pop(token, None)
        # Reclaim retained-but-cleared tensor storage outside the mutation lock.
        del state


def configure_simple_checkpoint() -> None:
    """Patch bootstrap before tq.init; do not modify an already running actor."""
    assert_stock_tq_version()
    configure_controller_checkpoint()
    actor_class = bootstrap.SimpleStorageUnit
    base = actor_class.__ray_metadata__.modified_class
    if issubclass(base, SnapshotStorageMixin):
        return

    class SnapshotStorageUnit(SnapshotStorageMixin, base):  # pragma: no cover
        pass

    bootstrap.SimpleStorageUnit = ray.remote(**actor_class._default_options)(
        SnapshotStorageUnit
    )


@dataclass
class SimpleCheckpointCapture(CheckpointCapture):
    """Own one captured view on each shard until publication or abort."""

    checkpoint_dir: Path
    token: str
    controller: Any
    units: list[Any]
    storage_info: list[dict[str, Any]]
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
            refs = self._persist_refs
            try:
                refs.append(
                    self.controller.nrl_persist_snapshot.remote(
                        self.token, str(self.checkpoint_dir / "controller_state.pkl")
                    )
                )
                for unit, info in zip(self.units, self.storage_info, strict=True):
                    path = (
                        self.checkpoint_dir
                        / "simple_storage"
                        / (f"su_{info['position']}_{info['storage_unit_id']}.pkl")
                    )
                    refs.append(unit.nrl_persist_snapshot.remote(self.token, str(path)))
            finally:
                # Partial submission still leaves owned writes to settle or release.
                sizes = _wait_all(refs, timeout_s=max(0.0, deadline - time.monotonic()))
            (
                self.checkpoint_dir / "simple_storage" / "storage_unit_info.json"
            ).write_text(json.dumps(self.storage_info, indent=2))
            # Publication markers must never outlive an unfinished shard write.
            (self.checkpoint_dir / "metadata.json").write_text(
                json.dumps(
                    {"storage_saved": True, "user_metadata": self.metadata}, indent=2
                )
            )
            LOGGER.info(
                "simple_snapshot persisted total_bytes=%d controller_bytes=%d",
                sum(sizes),
                sizes[0],
            )

    def release(self) -> None:
        with self._operation() as deadline:
            if self._released:
                return
            # Completed write failures were reported by persist; views can retire.
            # A wait expiry is different: those RPCs must finish before release.
            _wait_all(
                self._persist_refs,
                timeout_s=max(0.0, deadline - time.monotonic()),
                raise_errors=False,
            )
            refs = []
            first_error: Exception | None = None
            for unit in [self.controller, *self.units]:
                try:
                    refs.append(unit.nrl_release_snapshot.remote(self.token))
                except Exception as error:
                    if first_error is None:
                        first_error = error
            try:
                _wait_all(refs, timeout_s=max(0.0, deadline - time.monotonic()))
            except Exception as error:
                if first_error is None:
                    first_error = error
            if first_error is None:
                self._released = True
                self._persist_refs.clear()
            else:
                raise first_error


def capture_simple_checkpoint(
    checkpoint_dir: str | Path,
    *,
    num_storage_units: int,
    metadata: dict[str, Any] | None,
    timeout_s: float,
) -> SimpleCheckpointCapture:
    """Detach controller and retain shards while the mutation cut is held.

    The native manager's ID order, not Ray actor lookup order, determines shard
    positions. Native restore maps those positions to its newly allocated IDs.
    """
    capture_started = time.monotonic()
    assert_stock_tq_version()
    directory = Path(checkpoint_dir).resolve()
    # Fail before modifying disk or retaining data on invalid user metadata.
    captured_metadata = copy.deepcopy(metadata) if metadata is not None else {}
    json.dumps(captured_metadata)
    units = [
        ray.get_actor(f"TransferQueueStorageUnit#{rank}")
        for rank in range(num_storage_units)
    ]
    client = interface._maybe_create_tq_client()
    expected_ids = list(client.storage_manager.storage_unit_infos)
    if len(expected_ids) != num_storage_units:
        raise RuntimeError("SimpleStorage shard count differs from checkpoint config")
    # The checkpoint path must be new; never overwrite a durable snapshot.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "simple_storage").mkdir()
    capture = SimpleCheckpointCapture(
        checkpoint_dir=directory,
        token=uuid.uuid4().hex,
        controller=ray.get_actor("TransferQueueController", namespace="transfer_queue"),
        units=units,
        storage_info=[],
        metadata=captured_metadata,
        timeout_s=timeout_s,
    )
    setup_finished = time.monotonic()
    # The caller holds its mutation barrier, so a frozen actor must not stall it.
    deadline = setup_finished + timeout_s
    try:
        _wait_all(
            [capture.controller.nrl_capture_snapshot.remote(capture.token)],
            timeout_s=timeout_s,
        )
        controller_finished = time.monotonic()
        refs = []
        try:
            for unit in units:
                refs.append(unit.nrl_capture_snapshot.remote(capture.token))
        finally:
            unit_ids = _wait_all(refs, timeout_s=max(0.0, deadline - time.monotonic()))
        storage_finished = time.monotonic()
        if len(set(unit_ids)) != len(units) or set(unit_ids) != set(expected_ids):
            raise RuntimeError("Captured SimpleStorage units differ from TQ manager")
        by_id = dict(zip(unit_ids, units, strict=True))
        capture.units = [by_id[unit_id] for unit_id in expected_ids]
        capture.storage_info = [
            {"position": position, "storage_unit_id": unit_id}
            for position, unit_id in enumerate(expected_ids)
        ]
        LOGGER.info(
            "simple_snapshot capture_summary token=%s units=%d "
            "setup_s=%.6f controller_capture_rpc_s=%.6f storage_capture_rpc_s=%.6f "
            "validate_s=%.6f total_s=%.6f",
            capture.token,
            len(units),
            setup_finished - capture_started,
            controller_finished - setup_finished,
            storage_finished - controller_finished,
            time.monotonic() - storage_finished,
            time.monotonic() - capture_started,
        )
        return capture
    except BaseException:
        try:
            capture.release()
        except Exception as error:
            raise CheckpointReleaseError(
                f"Cannot confirm release of SimpleStorage capture {capture.token}; "
                "storage actors may still retain it, so stop checkpointing"
            ) from error
        raise
