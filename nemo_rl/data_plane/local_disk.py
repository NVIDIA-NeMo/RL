# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""The node-local directory an SSD tier writes to: on NVMe, and a disk.

The three SSD tiers (``mooncake_cpu.ssd_offload``, ``nixl.ssd`` and
``simple.ssd_offload``) default to :func:`default_ssd_dir`: ``/raid``, where
DGX nodes mount their NVMe RAID, when the node has it, else ``/tmp``. On some hosts ``/tmp`` is a RAM-backed tmpfs; a
tier "spilling" there would consume the memory it is meant to relieve and lose
the data with the node. The owner of the directory calls
:func:`require_disk_backed` once, before it writes, on the node where the
directory lives: a default on RAM stops the run, a configured path only warns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_RAM_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})
NVME_MOUNT = "/raid"  # DGX nodes mount the NVMe RAID here
FALLBACK_BASE = "/tmp"


def default_ssd_dir(name: str, mounts_text: str | None = None) -> str:
    """``/raid/<name>-<uid>`` when ``/raid`` is a mounted disk this process can write to, else ``/tmp/<name>-<uid>``.

    Resolve it on the node that creates the directory; the result still goes
    through :func:`require_disk_backed` there. ``mounts_text`` as in
    :func:`filesystem_of`.
    """
    base = FALLBACK_BASE if nvme_fallback_reason(mounts_text) else NVME_MOUNT
    # One folder per user: the first run creates it 0755, so a shared name
    # would stop every other user's run on this node.
    return os.path.join(base, f"{name}-{os.getuid()}")


def nvme_fallback_reason(mounts_text: str | None = None) -> str | None:
    """Why :func:`default_ssd_dir` cannot use ``/raid`` here; ``None`` when it can."""
    # From /proc/mounts, not os.path.ismount: in a container whose root lives
    # on the same device (enroot keeps container roots under /raid), a bind
    # mount of /raid has the device of /, and ismount says False. Writable:
    # /raid is often root-owned, and a job then has no business there.
    mountpoint, fs = _mount_of(NVME_MOUNT, mounts_text)
    if mountpoint != NVME_MOUNT:
        return f"{NVME_MOUNT} is not mounted"
    if fs in _RAM_FILESYSTEMS:
        return f"{NVME_MOUNT} is a {fs} (RAM, not a disk)"
    if not os.access(NVME_MOUNT, os.W_OK):
        return f"{NVME_MOUNT} is not writable by this process"
    return None


_noted_fallback = False  # one note per process; the driver's is the run's


def note_ssd_dir_fallback_once(log: Any) -> None:
    """Log once why the SSD tiers default to ``/tmp`` and what ``/tmp`` is, if so."""
    global _noted_fallback
    if _noted_fallback:
        return
    _noted_fallback = True
    reason = nvme_fallback_reason()
    if reason is None:
        return
    mountpoint, fs = _mount_of(FALLBACK_BASE)
    log.warning(
        "%s on %s, so the data plane's SSD tiers default to %s (%s mounted at %s). "
        "If that is not the node's NVMe, mount the NVMe at %s (ray.sub does so when "
        "%s is writable on the node) or set the tier's path.",
        reason,
        os.uname().nodename,
        FALLBACK_BASE,
        fs or "unknown filesystem",
        mountpoint or "?",
        NVME_MOUNT,
        NVME_MOUNT,
    )


def filesystem_of(path: str, mounts_text: str | None = None) -> str:
    """The filesystem type that holds ``path``: its longest mount-point prefix.

    ``path`` need not exist yet; its nearest existing ancestor is resolved.
    ``mounts_text`` is ``/proc/mounts`` (one ``src mountpoint fstype ...`` line
    each); read from the host when omitted. ``""`` when nothing matches.
    """
    return _mount_of(path, mounts_text)[1]


def _mount_of(path: str, mounts_text: str | None = None) -> tuple[str, str]:
    """``(mountpoint, fstype)`` of the mount that holds ``path``; ``("", "")`` if none."""
    # Resolve symlinks in the part that exists; keep the rest as given, so a
    # directory that is not created yet still maps to the mount it will be on.
    existing = Path(os.path.abspath(path))
    rest: list[str] = []
    while not existing.exists() and existing.parent != existing:
        rest.append(existing.name)
        existing = existing.parent
    target = os.path.join(os.path.realpath(existing), *reversed(rest))
    if mounts_text is None:
        try:
            mounts_text = Path("/proc/mounts").read_text()
        except OSError:
            return "", ""
    best, best_len = ("", ""), -1
    for line in mounts_text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        if (
            target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/")
        ) and len(mountpoint) > best_len:
            best, best_len = (mountpoint, fields[2]), len(mountpoint)
    return best


def require_disk_backed(path: str, *, setting: str, explicit: bool = False) -> None:
    """Refuse a RAM-backed ``path``, or only warn when the config named it.

    A default that landed on RAM (``/tmp`` as tmpfs) would silently turn the
    SSD tier into a second copy in memory, so it stops the run. A path the
    config sets (``explicit``) is a choice, for example a small test
    filesystem: it runs, with a warning that says what it costs. ``setting``
    names the config key in the message.
    """
    fs = filesystem_of(path)
    if fs not in _RAM_FILESYSTEMS:
        return
    where = (
        f"{setting}={path!r} is on a {fs} filesystem (RAM, not a disk) on "
        f"host {os.uname().nodename}"
    )
    if explicit:
        LOGGER.warning("%s; the SSD tier there spends memory, not disk.", where)
        return
    raise ValueError(f"{where}; point it at a node-local SSD.")
