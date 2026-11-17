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
"""Immutable controller checkpoint bytes for pinned TransferQueue 0.1.11.

Capture serializes the native checkpoint layout under a lock shared with the
controller's ZMQ request dispatcher. Persistence writes only the retained bytes
without excluding live requests. Callers enforce a global data-plane mutation
cut across controller and storage capture. No metadata payload crosses Ray.
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from typing import Any

import ray
import transfer_queue.interface as interface
from transfer_queue.controller import DataPartitionStatus

from nemo_rl.data_plane.adapters.tq_controller_patch import assert_stock_tq_version

LOGGER = logging.getLogger(__name__)


class SnapshotControllerMixin:
    """Keep one immutable metadata cut while live request traffic continues."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._controller_snapshot_lock = threading.RLock()
        self._controller_snapshot_views: dict[str, bytes] = {}
        super().__init__(*args, **kwargs)

    def _handle_request(self, request_msg: Any, monitor: Any) -> Any:
        with self._controller_snapshot_lock:
            return super()._handle_request(request_msg, monitor)

    def load_checkpoint(self, path: str) -> None:
        with self._controller_snapshot_lock:
            if self._controller_snapshot_views:
                raise RuntimeError(
                    "Cannot restore while a controller snapshot is retained"
                )
            super().load_checkpoint(path)

    def nrl_capture_snapshot(self, token: str) -> str:
        """Serialize live metadata under the request lock; return only its ID."""
        started, cpu_started = time.monotonic(), time.thread_time()
        with self._controller_snapshot_lock:
            acquired = time.monotonic()
            if self._controller_snapshot_views:
                raise RuntimeError("Only one controller snapshot may be retained")
            state = {
                "controller_id": self.controller_id,
                "partitions": {
                    pid: partition
                    if type(partition) is DataPartitionStatus
                    else partition.to_snapshot()
                    for pid, partition in self.partitions.items()
                },
                "index_manager": {
                    "partition_to_indexes": dict(
                        self.index_manager.partition_to_indexes
                    ),
                    "reusable_indexes": list(self.index_manager.reusable_indexes),
                    "global_index_counter": self.index_manager.global_index_counter,
                    "allocated_indexes": self.index_manager.allocated_indexes,
                },
                "sampler": self.sampler.save_checkpoint(),
            }
            layout_finished = time.monotonic()
            payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            serialized = time.monotonic()
            self._controller_snapshot_views[token] = payload
        LOGGER.info(
            "controller_snapshot captured token=%s lock_wait_s=%.6f layout_s=%.6f "
            "serialize_s=%.6f thread_cpu_s=%.6f bytes=%d total_s=%.6f",
            token,
            acquired - started,
            layout_finished - acquired,
            serialized - layout_finished,
            time.thread_time() - cpu_started,
            len(payload),
            time.monotonic() - started,
        )
        return self.controller_id

    def nrl_persist_snapshot(self, token: str, path: str) -> int:
        """Write captured bytes without holding the live-request lock."""
        with self._controller_snapshot_lock:
            payload = self._controller_snapshot_views[token]
        started, cpu_started = time.monotonic(), time.thread_time()
        with open(path, "xb") as output:
            output.write(payload)
        LOGGER.info(
            "controller_snapshot persisted token=%s seconds=%.6f thread_cpu_s=%.6f bytes=%d",
            token,
            time.monotonic() - started,
            time.thread_time() - cpu_started,
            len(payload),
        )
        return len(payload)

    def nrl_release_snapshot(self, token: str) -> None:
        """Idempotently release the immutable payload after writes settle."""
        with self._controller_snapshot_lock:
            payload = self._controller_snapshot_views.pop(token, None)
        del payload


def configure_controller_checkpoint() -> None:
    """Wrap the controller before initialization, retaining existing patches."""
    assert_stock_tq_version()
    actor_class = interface.TransferQueueController
    base = actor_class.__ray_metadata__.modified_class
    if issubclass(base, SnapshotControllerMixin):
        return

    class SnapshotController(SnapshotControllerMixin, base):  # pragma: no cover
        pass

    interface.TransferQueueController = ray.remote(**actor_class._default_options)(
        SnapshotController
    )
