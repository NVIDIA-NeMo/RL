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
"""``mooncake_cpu.ssd_offload``: Mooncake's embedded offload in segment owners.

Mooncake's master offloads an object only to the SSD of the process owning
the segment that holds it, so every process that owns Mooncake memory runs
offload in its own directory. TQ's own ``MooncakeStore.offload`` is not used:
its single offload client on the driver owns none of the data.
"""

from __future__ import annotations

import _thread
import atexit
import logging
import os
import re
import shutil
import tempfile
import threading
import urllib.request
from collections.abc import Callable
from typing import Any

from nemo_rl.data_plane.interfaces import MooncakeCpuConfig

LOGGER = logging.getLogger(__name__)

# Read by mooncake at store setup; a launcher may still override them.
ENV_DEFAULTS = {
    # The default bucket backend writes nothing until a bucket holds 256 MB
    # or 500 keys; until then the queued objects stay pinned in memory, so an
    # owner whose eviction batch is smaller than one bucket fills up and every
    # put fails. One file per key writes each object as it is queued.
    "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR": "file_per_key_storage_backend",
    # Disk watermark eviction deletes the SSD copy, which is the only copy of
    # an object already evicted from memory. The per-key backend's own
    # eviction (MOONCAKE_OFFLOAD_ENABLE_EVICTION) stays on: in mooncake
    # 0.3.13 turning it off makes setup fail ("Metadata has not been loaded
    # yet"); it only deletes files once the disk is 90% full.
    "MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION": "false",
}

# This process's offload directory, once installed.
_dir: str | None = None

# The master counts every report of SSD copies its owners deleted for space
# (FileStorage::NotifyEvictedDiskReplicas -> BatchEvictDiskReplica). NeMo-RL
# and TQ never delete SSD copies themselves, so any count means data loss.
EVICTION_COUNTER = "master_evict_disk_replica_requests_total"
_COUNTER_LINE = re.compile(rf"^{EVICTION_COUNTER}(?:{{[^}}]*}})?\s+(\S+)", re.M)
# The running watcher in this process, if any.
_watch_stop: threading.Event | None = None


def master_args(cfg: MooncakeCpuConfig) -> tuple[str, ...]:
    """mooncake_master flags for ``ssd_offload``; empty when it is off.

    In Mooncake, offload happens on eviction: past the high watermark the
    master moves objects to SSD and frees ``eviction_ratio`` of the memory.
    These replace the no-eviction flags TQ passes (gflags keeps the last
    occurrence). ``offload_force_evict=false``: an object whose SSD write
    failed stays in memory rather than being dropped. TQ's 999999 ms lease
    would keep every object written in the last ~17 min from being evicted,
    so it goes back to mooncake's own 10 s default.
    """
    if not cfg.ssd_offload.enabled:
        return ()
    return (
        "--enable_offload=true",
        "--offload_on_evict=true",
        "--offload_force_evict=false",
        f"--eviction_high_watermark_ratio={cfg.ssd_offload.offload_threshold}",
        "--eviction_ratio=0.1",
        "--default_kv_lease_ttl=10000",
    )


def install(cfg: MooncakeCpuConfig, segment_size: int | None) -> None:
    """Make this process's Mooncake store offload to its own SSD directory.

    No-op unless offload is on and this process owns memory. TQ's client calls
    ``setup`` with no offload arguments, so this swaps the store class it
    builds for one that adds them. Each process gets a fresh, owner-only
    ``<path>/nemo_rl_mooncake_offload/<ray job>-<pid>-<random>``: no run can
    pick up, or wipe, another's files. Must run before the store is set up.
    """
    global _dir
    if (
        _dir is not None
        or not cfg.ssd_offload.enabled
        or not cfg.owned_segment_size(segment_size)
    ):
        return

    import ray
    from transfer_queue.storage.clients import mooncake_client

    from nemo_rl.data_plane.adapters.transfer_queue import _tq_shape_drift_error

    base = getattr(mooncake_client, "MooncakeDistributedStore", None)
    if base is None:
        raise _tq_shape_drift_error(
            "storage.clients.mooncake_client no longer imports "
            "MooncakeDistributedStore",
            "Mooncake SSD offload could not be enabled",
            "store class",
        )
    for name, value in ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
    parent = os.path.join(str(cfg.ssd_offload.path), "nemo_rl_mooncake_offload")
    os.makedirs(parent, exist_ok=True)
    job_id = ray.get_runtime_context().get_job_id()
    _dir = path = tempfile.mkdtemp(prefix=f"{job_id}-{os.getpid()}-", dir=parent)
    # Runs at a normal exit only: Mooncake replaces the SIGTERM handler with
    # one that ends in _Exit, so a process Ray stops with SIGTERM skips it
    # (storage units are closed explicitly: stop_storage_units).
    atexit.register(remove_dir)

    class _OffloadingStore(base):  # type: ignore[misc, valid-type]
        def setup(self, *args: Any, **kwargs: Any) -> Any:
            return super().setup(
                *args, enable_ssd_offload=True, ssd_offload_path=path, **kwargs
            )

    mooncake_client.MooncakeDistributedStore = _OffloadingStore
    LOGGER.info("Mooncake SSD offload for this process under %s", path)


def remove_dir() -> None:
    """Delete this process's offload directory, if it has one.

    Call only once nothing will read this process's data again: the files
    may be the only copy of offloaded objects.
    """
    if _dir is not None:
        shutil.rmtree(_dir, ignore_errors=True)


def disk_evictions(metrics_text: str) -> int:
    """The master's SSD-eviction count from its Prometheus ``/metrics`` text."""
    return int(sum(float(value) for value in _COUNTER_LINE.findall(metrics_text)))


def watch_disk_evictions(
    read: Callable[[], int],
    path: str,
    *,
    interval_s: float = 30.0,
    on_eviction: Callable[[], None] = _thread.interrupt_main,
) -> threading.Event:
    """Poll ``read`` and stop the run at the first SSD eviction.

    Mooncake's per-key backend deletes the oldest offloaded files once the disk
    is nearly full and cannot be told not to (0.3.13), and a deleted file may be
    the only copy of an object. The default ``on_eviction`` interrupts this
    process's main thread, ending the run. Returns an event that stops the
    watcher when set; the watcher also sets it once it has fired.
    """
    stop = threading.Event()

    def _watch() -> None:
        while not stop.wait(interval_s):
            try:
                count = read()
            except OSError:
                continue  # master briefly unreachable (startup, shutdown)
            if count > 0:
                LOGGER.error(
                    "Mooncake deleted SSD copies of offloaded objects (%d eviction "
                    "report(s)) because the disk under %s is nearly full. Those may "
                    "have been the only copies, so this run is stopping. Give "
                    "data_plane.mooncake_cpu.ssd_offload.path more free disk.",
                    count,
                    path,
                )
                on_eviction()
                stop.set()
                return

    threading.Thread(
        target=_watch, name="mooncake-ssd-eviction-watch", daemon=True
    ).start()
    return stop


def watch_master_disk_evictions(metrics_url: str, path: str) -> None:
    """Watch the master's ``/metrics`` at ``metrics_url`` from this process."""
    global _watch_stop
    if _watch_stop is not None:
        _watch_stop.set()  # a re-init in this process replaces the old watcher

    def read() -> int:
        with urllib.request.urlopen(metrics_url, timeout=5) as response:
            return disk_evictions(response.read().decode())

    _watch_stop = watch_disk_evictions(read, path)
