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
"""Socket awareness, detected from sysfs (never from the GPU model).

A *socket* here is a NUMA node that has CPUs. NUMA nodes without CPUs (the
GPU-memory nodes Grace systems expose) are ignored.

| node type (examples)              | sockets | ``numa="auto"`` does            |
|-----------------------------------|---------|---------------------------------|
| GB200 / GB300 tray (2x Grace)     | 2       | bind per socket (below)         |
| DGX H100 / H200 (2x x86)          | 2       | bind per socket (below)         |
| GH200, one superchip per node     | 1       | nothing (no-op)                 |

Binding per socket, with ``numa="auto"`` and >= 2 sockets:

- storage unit: the k-th unit placed on a node takes socket ``k % sockets``;
  it pins its threads to that socket's CPUs and first-touches its slab there
  (so the pages live in that socket's memory).
- client: if the process is already pinned inside one socket, it writes to a
  unit on its socket first. An unpinned client changes nothing.
- NICs are NOT restricted per socket: doing so broke cross-node UCX endpoint
  creation at 86 nodes (job 1072049), so every process keeps UCX_NET_DEVICES.

``numa="off"`` disables all of it. Run one unit per socket per node
(``num_storage_units = nodes x sockets``) to keep every local write socket-local.
"""

from __future__ import annotations

import os
from pathlib import Path

NODE_ROOT = "/sys/devices/system/node"
IB_ROOT = "/sys/class/infiniband"


def _parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpu_nodes(root: str = NODE_ROOT) -> dict[int, set[int]]:
    """NUMA node id -> CPUs, for nodes that have CPUs (i.e. sockets)."""
    out: dict[int, set[int]] = {}
    for d in Path(root).glob("node[0-9]*"):
        cpus = (
            _parse_cpulist((d / "cpulist").read_text())
            if (d / "cpulist").exists()
            else set()
        )
        if cpus:
            out[int(d.name[4:])] = cpus
    return out


def socket_of_affinity(nodes: dict[int, set[int]] | None = None) -> int | None:
    """The socket this process is pinned inside, or ``None`` if it spans several."""
    nodes = cpu_nodes() if nodes is None else nodes
    aff = os.sched_getaffinity(0)
    for node, cpus in nodes.items():
        if aff and aff <= cpus:
            return node
    return None


def allowed_from_env(env: dict[str, str] | None = None) -> set[str] | None:
    """RDMA device names UCX_NET_DEVICES allows (``None`` = all)."""
    devs = (os.environ if env is None else env).get("UCX_NET_DEVICES", "all").strip()
    if devs in ("", "all"):
        return None
    return {d.split(":")[0] for d in devs.split(",") if d}


def rdma_nics(
    node: int, allowed: set[str] | None = None, root: str = IB_ROOT
) -> list[str]:
    """ACTIVE RDMA devices attached to ``node``, sorted, filtered by ``allowed``."""
    out = []
    for d in sorted(Path(root).glob("*")):
        try:
            if int((d / "device" / "numa_node").read_text()) != node:
                continue
            if "ACTIVE" not in (d / "ports" / "1" / "state").read_text():
                continue
        except (OSError, ValueError):
            continue
        if allowed is None or d.name in allowed:
            out.append(d.name)
    return out


def bind_process(cpus: set[int]) -> None:
    """Pin every thread of this process (and threads it creates later) to ``cpus``."""
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except OSError:
            pass


def first_touch(arr) -> None:
    """Fault every page of ``arr`` from the current (pinned) thread.

    The kernel's default first-touch policy places the pages on this socket.
    """
    step = os.sysconf("SC_PAGE_SIZE")
    flat = arr.reshape(-1).view("uint8")
    flat[::step] = 0
    flat[-1:] = 0


def device_list_param(nics: list[str]) -> str:
    """NIXL UCX ``device_list``: split on ", " and suffixed with ":1" by NIXL."""
    return ", ".join(nics)
