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
"""The node-local directory an SSD tier writes to must be a disk.

Both SSD tiers (``mooncake_cpu.ssd_offload`` and ``nixl.ssd``) default to a
directory under ``/tmp``. On some hosts ``/tmp`` is a RAM-backed tmpfs; a tier
"spilling" there would consume the memory it is meant to relieve and lose the
data with the node. The owner of the directory calls :func:`require_disk_backed`
once, before it writes, on the node where the directory lives.
"""

from __future__ import annotations

import os
from pathlib import Path

_RAM_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})


def filesystem_of(path: str, mounts_text: str | None = None) -> str:
    """The filesystem type that holds ``path``: its longest mount-point prefix.

    ``path`` need not exist yet; its nearest existing ancestor is resolved.
    ``mounts_text`` is ``/proc/mounts`` (one ``src mountpoint fstype ...`` line
    each); read from the host when omitted. ``""`` when nothing matches.
    """
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
            return ""
    best, best_len = "", -1
    for line in mounts_text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        if (
            target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/")
        ) and len(mountpoint) > best_len:
            best, best_len = fields[2], len(mountpoint)
    return best


def require_disk_backed(path: str, *, setting: str) -> None:
    """Refuse a RAM-backed ``path``; ``setting`` names the config key for the error."""
    fs = filesystem_of(path)
    if fs in _RAM_FILESYSTEMS:
        raise ValueError(
            f"{setting}={path!r} is on a {fs} filesystem (RAM, not a disk) on "
            f"host {os.uname().nodename}; point it at a node-local SSD."
        )
