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
import fcntl
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from nemo_rl.data_plane.interfaces import MooncakeCpuConfig

LOGGER = logging.getLogger(__name__)

# Read by mooncake at store setup. install() refuses a launcher value that
# differs: either one would let mooncake delete the only copy of an object.
REQUIRED_ENV = {
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
# Held open for this process's lifetime: the kernel drops the lock when the
# process exits, even on SIGKILL or Mooncake's _Exit, which marks the dir stale.
_lock_fd: int | None = None
# Set once the offloading store's setup has run; per process, because a second
# client in one process reuses the first client's store.
_offloading_store_set_up = False

# Mooncake's ErrorCode::NO_AVAILABLE_HANDLE: no segment had room for the put.
NO_AVAILABLE_HANDLE = -200
# With offload on, a full segment is a wait, not an error: offload frees
# memory in the background. The bound only turns a stuck offload (disk full,
# SSD writes failing) into an error instead of a hang.
PUT_WAIT_S = 600.0
_FIRST_BACKOFF_S = 0.05
PUT_BACKOFF_MAX_S = 1.0

# The master counts every report of SSD copies its owners deleted for space
# (FileStorage::NotifyEvictedDiskReplicas -> BatchEvictDiskReplica). NeMo-RL
# and TQ never delete SSD copies themselves, so any count means data loss.
EVICTION_COUNTER = "master_evict_disk_replica_requests_total"
_COUNTER_LINE = re.compile(
    rf"^{re.escape(EVICTION_COUNTER)}(?:{{[^}}]*}})?\s+(\S+)", re.M
)
# The running watcher in this process, if any.
_watch_stop: threading.Event | None = None

# Mooncake's remove drops only the master's record: the owner keeps the file of
# a cleared key until the key is offloaded again or the owner exits. Each owner
# sweeps those files itself (Mooncake 0.3.13 has no per-key disk delete).
SWEEP_INTERVAL_S = 30.0
_SWEEP_BATCH = 1000
# Files store the tenant-scoped key, ``<tenant>\0<key>``; TQ uses the default
# tenant. Measured on 0.3.13 (job 1338878): b"\n\x0edefault\x0098@big...".
_TENANT_PREFIX = "default\0"
# The running sweep in this process, if any.
_sweep_stop: threading.Event | None = None


def master_args(cfg: MooncakeCpuConfig) -> tuple[str, ...]:
    """mooncake_master flags for ``ssd_offload``; empty when it is off.

    In Mooncake, offload happens on eviction: past the high watermark the
    master moves objects to SSD and frees ``eviction_ratio`` of the memory.
    These replace the no-eviction flags TQ passes (gflags keeps the last
    occurrence). ``offload_force_evict=false``: an object whose SSD write
    failed stays in memory rather than being dropped. A fresh object cannot
    be evicted until its lease expires: TQ's 999999 ms would pin ~17 min of
    writes, and even 10 s let a 512 MiB segment fill at 64 MiB/s before
    anything was evictable (job 1321643). The 1 s lease must cover one whole
    ``batch_get_into`` call (up to ``staging_buffer_size`` per call), not one
    key: a slower call fails with LEASE_EXPIRED.
    """
    if not cfg.ssd_offload.enabled:
        return ()
    return (
        "--enable_offload=true",
        "--offload_on_evict=true",
        "--offload_force_evict=false",
        f"--eviction_high_watermark_ratio={cfg.ssd_offload.offload_threshold}",
        "--eviction_ratio=0.1",
        "--default_kv_lease_ttl=1000",
    )


def install(cfg: MooncakeCpuConfig, segment_size: int | None) -> None:
    """Set up this process's Mooncake client for SSD offload.

    No-op unless offload is on. Every client's puts then wait for memory
    (:func:`wait_for_memory_on_put`). A process that owns memory also offloads
    to its own SSD directory: TQ's client calls ``setup`` with no offload
    arguments, so this swaps the store class it builds for one that adds them.
    Each owner gets a fresh, owner-only ``<path>/nemo_rl_mooncake_offload/<ray job>-<pid>-<random>``,
    locked for its lifetime: no live run can pick up, or wipe, another's files,
    and dirs whose owner died are removed here. Must run before the store is
    set up.
    """
    global _dir
    if not cfg.ssd_offload.enabled:
        return

    import ray
    from transfer_queue.storage.clients import mooncake_client

    from nemo_rl.data_plane.adapters.transfer_queue import _tq_shape_drift_error

    wait_for_memory_on_put(getattr(mooncake_client, "MooncakeStoreClient", None))
    if _dir is not None or not cfg.owned_segment_size(segment_size):
        return

    base = getattr(mooncake_client, "MooncakeDistributedStore", None)
    if base is None:
        raise _tq_shape_drift_error(
            "storage.clients.mooncake_client no longer imports "
            "MooncakeDistributedStore",
            "Mooncake SSD offload could not be enabled",
            "store class",
        )
    for name, value in REQUIRED_ENV.items():
        if os.environ.setdefault(name, value) != value:
            raise ValueError(
                f"Mooncake SSD offload needs {name}={value}, but the environment "
                f"sets {os.environ[name]!r}; unset it or turn ssd_offload off"
            )
    parent = os.path.join(str(cfg.ssd_offload.path), "nemo_rl_mooncake_offload")
    os.makedirs(parent, exist_ok=True)
    _remove_stale_dirs(parent)  # free disk left by earlier runs on this node
    job_id = ray.get_runtime_context().get_job_id()
    _dir = path = tempfile.mkdtemp(prefix=f"{job_id}-{os.getpid()}-", dir=parent)
    _hold_dir_lock(path)
    # Runs at a normal exit only: Mooncake replaces the SIGTERM handler with
    # one that ends in _Exit, so a process Ray stops with SIGTERM skips it
    # (storage units are closed explicitly: stop_storage_units).
    atexit.register(remove_dir)

    class _OffloadingStore(base):  # type: ignore[misc, valid-type]
        def setup(self, *args: Any, **kwargs: Any) -> Any:
            global _offloading_store_set_up
            _offloading_store_set_up = True
            result = super().setup(
                *args, enable_ssd_offload=True, ssd_offload_path=path, **kwargs
            )
            if result == 0:
                _start_sweep(self, path)
            return result

    mooncake_client.MooncakeDistributedStore = _OffloadingStore
    LOGGER.info("Mooncake SSD offload for this process under %s", path)


def wait_for_memory_on_put(client_cls: Any) -> None:
    """Make TQ's puts wait for offload to free memory instead of failing.

    TQ's ``_batch_upsert_with_retry`` gives up after 3 x 1 s. Raising TQ's
    ``MAX_RETRIES`` instead would also retry permanent errors, logging each
    at ERROR. Patches ``client_cls`` once.
    """
    if getattr(client_cls, "_nrl_waits_for_memory", False):
        return
    if not hasattr(client_cls, "_batch_upsert_with_retry"):
        from nemo_rl.data_plane.adapters.transfer_queue import _tq_shape_drift_error

        raise _tq_shape_drift_error(
            "MooncakeStoreClient._batch_upsert_with_retry is gone",
            "puts would fail instead of waiting while SSD offload frees memory",
            "put retry",
        )
    tq_upsert = client_cls._batch_upsert_with_retry

    def _batch_upsert_with_retry(self, keys, ptrs, sizes):  # type: ignore[no-untyped-def]
        upsert_waiting_for_memory(
            lambda k, p, s: self._store.batch_upsert_from(
                k, p, s, config=self.replica_config
            ),
            keys,
            ptrs,
            sizes,
            on_error=lambda k, p, s, _: tq_upsert(self, k, p, s),
        )

    client_cls._batch_upsert_with_retry = _batch_upsert_with_retry
    client_cls._nrl_waits_for_memory = True


def upsert_waiting_for_memory(
    upsert: Callable[[list[str], list[int], list[int]], Any],
    keys: list[str],
    ptrs: list[int],
    sizes: list[int],
    *,
    on_error: Callable[[list[str], list[int], list[int], list[int]], None],
) -> None:
    """Upsert, putting keys that find every segment full again until they fit.

    Mooncake admits a put only into free segment memory, while offload frees
    memory later, in the background. ``upsert`` returns one status per key;
    keys answered ``NO_AVAILABLE_HANDLE`` are put again with capped backoff
    for up to ``PUT_WAIT_S``, with one warning. ``on_error(keys, ptrs, sizes,
    statuses)`` gets the keys that failed any other way.
    """
    deadline = time.monotonic() + PUT_WAIT_S
    backoff = _FIRST_BACKOFF_S
    while True:
        results = upsert(keys, ptrs, sizes)
        if not isinstance(results, (list, tuple)) or len(results) != len(keys):
            raise RuntimeError(
                f"batch_upsert_from returned {results!r} for {len(keys)} keys"
            )
        failed = [i for i, r in enumerate(results) if r not in (0, NO_AVAILABLE_HANDLE)]
        if failed:
            on_error(*_pick((keys, ptrs, sizes, list(results)), failed))
        full = [i for i, r in enumerate(results) if r == NO_AVAILABLE_HANDLE]
        if not full:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Mooncake memory stayed full for {PUT_WAIT_S:.0f} s with SSD "
                f"offload on; {len(full)} key(s) not stored, e.g. "
                f"{keys[full[0]]!r}. Either offload is not freeing memory "
                "(check the disk under data_plane.mooncake_cpu.ssd_offload.path "
                "and the mooncake_master log) or one object is larger than any "
                "Mooncake segment."
            )
        if backoff == _FIRST_BACKOFF_S:
            LOGGER.warning(
                "Mooncake memory is full; %d key(s) (e.g. %r) waiting for SSD "
                "offload to free memory (up to %.0f s).",
                len(full),
                keys[full[0]],
                PUT_WAIT_S,
            )
        keys, ptrs, sizes = _pick((keys, ptrs, sizes), full)
        time.sleep(backoff)
        backoff = min(PUT_BACKOFF_MAX_S, 2 * backoff)


def _pick(columns: tuple[list[Any], ...], indexes: list[int]) -> list[list[Any]]:
    return [[column[i] for i in indexes] for column in columns]


def require_offloading_store() -> None:
    """Raise if this process owns memory but TQ never built the offloading store.

    The swap in :func:`install` works only while TQ looks the store class up by
    name; call this after connecting, when TQ has built its store.
    """
    if _dir is not None and not _offloading_store_set_up:
        raise RuntimeError(
            "Mooncake SSD offload is on, but TQ did not build its store through "
            "mooncake_client.MooncakeDistributedStore, so this process would "
            "never offload. The TQ pin may have moved."
        )


def _hold_dir_lock(path: str) -> None:
    global _lock_fd
    tmp = os.path.join(path, ".lock.tmp")
    _lock_fd = os.open(tmp, os.O_CREAT | os.O_WRONLY, 0o600)
    fcntl.flock(_lock_fd, fcntl.LOCK_EX)
    # Lock, then publish under the name other runs check, so no other process
    # ever sees this dir unlocked.
    os.rename(tmp, os.path.join(path, ".lock"))


def _remove_stale_dirs(parent: str) -> None:
    """Delete offload dirs whose owning process is gone (crash, kill, restart)."""
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        try:
            fd = os.open(os.path.join(path, ".lock"), os.O_RDONLY)
        except OSError:
            continue  # its owner is still starting, or it predates the lock
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            continue  # its owner is alive
        else:
            shutil.rmtree(path, ignore_errors=True)
        finally:
            os.close(fd)


def remove_dir() -> None:
    """Delete this process's offload directory, if it has one.

    Call only once nothing will read this process's data again: the files
    may be the only copy of offloaded objects.
    """
    if _sweep_stop is not None:
        _sweep_stop.set()
    if _dir is not None:
        shutil.rmtree(_dir, ignore_errors=True)


def _file_key(path: str) -> str | None:
    """The default tenant's key in a per-key offload file, or None.

    The file holds a protobuf KVEntry whose field 1 is the tenant, a NUL, then
    the key; the file name replaces the NUL and filesystem-unsafe characters
    with ``_``, so it cannot be mapped back. Other tenants' files are skipped.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if not head or head[0] != 0x0A:
        return None
    length, shift, i = 0, 0, 1
    while i < min(len(head), 11):
        byte = head[i]
        length |= (byte & 0x7F) << shift
        i += 1
        if not byte & 0x80:
            break
        shift += 7
    else:
        return None
    key = head[i : i + length]
    if len(key) != length:
        return None
    try:
        scoped = key.decode()
    except UnicodeDecodeError:
        return None
    if not scoped.startswith(_TENANT_PREFIX):
        return None
    return scoped[len(_TENANT_PREFIX) :] or None


def sweep_cleared_files(root: str, exists: Callable[[list[str]], list[int]]) -> int:
    """Delete offload files under ``root`` whose key the master no longer has.

    ``exists`` is ``batch_is_exist``: 1 present, 0 absent, negative on error.
    A file is deleted only on an explicit 0 and only if it is unchanged since
    it was listed (before the check): a key re-put and offloaded again after
    the check rewrites its file, which is then kept. Returns the count deleted.
    """
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            key = None if name.startswith(".lock") else _file_key(path)
            if key is not None and (stamp := _stamp(path)) is not None:
                files.append((path, key, stamp))
    removed = 0
    for start in range(0, len(files), _SWEEP_BATCH):
        batch = files[start : start + _SWEEP_BATCH]
        results = exists([key for _, key, _ in batch])
        if len(results) != len(batch):
            continue
        for (path, _, stamp), result in zip(batch, results):
            if result != 0 or _stamp(path) != stamp:
                continue
            try:
                os.unlink(path)
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def _stamp(path: str) -> tuple[int, int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


def _start_sweep(store: Any, root: str) -> None:
    global _sweep_stop
    stop = threading.Event()

    def _sweep() -> None:
        while not stop.wait(SWEEP_INTERVAL_S):
            try:
                removed = sweep_cleared_files(root, store.batch_is_exist)
            except Exception:
                LOGGER.warning("Mooncake SSD offload sweep failed", exc_info=True)
                continue
            if removed:
                LOGGER.debug("Deleted %d SSD files of cleared keys", removed)

    threading.Thread(target=_sweep, name="mooncake-ssd-sweep", daemon=True).start()
    _sweep_stop = stop


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
        failing = False
        while not stop.wait(interval_s):
            try:
                count = read()
            except Exception:  # keep watching: a dead watcher hides data loss
                if not failing:
                    LOGGER.warning(
                        "Cannot read the Mooncake SSD-eviction count; retrying",
                        exc_info=True,
                    )
                failing = True
                continue
            failing = False
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
