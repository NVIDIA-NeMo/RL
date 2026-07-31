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

"""NUMA-aware CPU affinity and memory binding for GPU workers.

Uses a GPU→cpulist mapping file written by topology_probe.sh (in ray.sub)
at node startup. The file path is communicated via the NRL_GPU_CPU_AFFINITY_FILE
environment variable. See ray.sub for the writer side.

Disable all binding with NRL_DISABLE_NUMA_BINDING=1.
Disable only memory policy with NRL_DISABLE_NUMA_MEMBIND=1.
"""

import ctypes
import ctypes.util
import logging
import os

logger = logging.getLogger(__name__)

# IMPORTANT: This default path must stay in sync with topology_probe.sh in ray.sub.
# The canonical path is set via the NRL_GPU_CPU_AFFINITY_FILE env var exported by ray.sub.
GPU_CPU_AFFINITY_PATH = os.environ.get(
    "NRL_GPU_CPU_AFFINITY_FILE", "/tmp/nrl_gpu_cpu_affinity"
)


def bind_to_gpu_numa(gpu_id: int) -> bool:
    """Pin the current process to the NUMA-local CPUs and memory of the given GPU.

    Reads the GPU→cpulist mapping written by topology_probe.sh at node
    startup, then calls os.sched_setaffinity() for CPU pinning and
    numa_set_membind() for memory policy. Best-effort: failures are
    logged, never raised.

    Args:
        gpu_id: Node-global physical GPU index (``nvidia-smi`` numbering), which
            is how the affinity file is keyed. Passed explicitly because
            ``CUDA_VISIBLE_DEVICES`` lists all devices on the node under
            ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` and so does not
            identify a single worker's GPU. In a Ray actor this is
            ``int(ray.get_gpu_ids()[0])``.

    Returns True if CPU binding succeeded, False if skipped or failed.
    Memory binding is attempted independently and logged separately.
    """
    if os.environ.get("NRL_DISABLE_NUMA_BINDING") == "1":
        return False

    gpu = str(gpu_id)
    try:
        with open(GPU_CPU_AFFINITY_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx, cpulist = line.split(":", 1)
                if idx == gpu:
                    cpus = _parse_cpulist(cpulist)
                    os.sched_setaffinity(0, cpus)
                    logger.info("NUMA CPU binding: GPU %s → CPUs %s", gpu, cpulist)
                    _set_numa_membind(cpus)
                    return True
        logger.debug("NUMA binding: GPU %s not found in %s", gpu, GPU_CPU_AFFINITY_PATH)
    except FileNotFoundError:
        logger.debug("NUMA binding skipped: %s not found", GPU_CPU_AFFINITY_PATH)
    except Exception as exc:
        logger.debug("NUMA binding skipped: %s", exc)
    return False


def resolve_visible_gpu_id(local_index: int) -> int | None:
    """Map a process-local CUDA device index to its node-global physical GPU id.

    ``CUDA_VISIBLE_DEVICES`` lists the physical GPU ids visible to this process
    in device-index order, and ``local_index`` (e.g.
    ``torch.cuda.current_device()``) indexes into that list. The affinity file is
    keyed by the physical id, so return ``CUDA_VISIBLE_DEVICES[local_index]``.

    ``CUDA_VISIBLE_DEVICES`` contents depend on the worker:
      - vLLM TP>1 (``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1``): the
        per-instance device subset, e.g. ``"4,5"``.
      - vLLM TP=1: a single isolated device, so ``local_index`` is 0.

    Returns the physical GPU id, or None if it cannot be resolved (unset CVD,
    index out of range, or non-integer entries such as MIG UUIDs).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        return None
    devices = cvd.split(",")
    if local_index < 0 or local_index >= len(devices):
        return None
    try:
        return int(devices[local_index])
    except ValueError:
        return None


def _load_libnuma() -> ctypes.CDLL | None:
    """Load libnuma, returning None if unavailable."""
    try:
        return ctypes.CDLL("libnuma.so.1")
    except OSError:
        return None


def _get_numa_node(libnuma: ctypes.CDLL, cpus: set[int]) -> int:
    """Return the NUMA node for the given CPU set, or -1 on failure."""
    libnuma.numa_node_of_cpu.restype = ctypes.c_int
    return libnuma.numa_node_of_cpu(min(cpus))


def _set_numa_membind(cpus: set[int]) -> bool:
    """Hard-bind memory allocations to the NUMA node of the given CPUs."""
    if os.environ.get("NRL_DISABLE_NUMA_MEMBIND") == "1":
        return False

    libnuma = _load_libnuma()
    if libnuma is None:
        logger.debug("NUMA membind skipped: libnuma.so.1 not available")
        return False

    try:
        numa_node = _get_numa_node(libnuma, cpus)
        if numa_node < 0:
            logger.debug(
                "NUMA membind skipped: numa_node_of_cpu(%d) returned %d",
                min(cpus),
                numa_node,
            )
            return False

        libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
        libnuma.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        libnuma.numa_bitmask_setbit.restype = ctypes.c_void_p
        libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
        libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]

        nodemask = libnuma.numa_allocate_nodemask()
        if not nodemask:
            logger.debug("NUMA membind skipped: numa_allocate_nodemask returned NULL")
            return False

        try:
            libnuma.numa_bitmask_setbit(nodemask, numa_node)
            libnuma.numa_set_membind(nodemask)
        finally:
            libnuma.numa_bitmask_free(nodemask)

        logger.info(
            "NUMA membind: hard-bound to node %d (from CPU %d)", numa_node, min(cpus)
        )
        return True
    except Exception as exc:
        logger.debug("NUMA membind skipped: %s", exc)
        return False


def _parse_cpulist(cpulist: str) -> set[int]:
    """Parse a Linux cpulist string like '0-71' into a set of ints."""
    cpus: set[int] = set()
    for part in cpulist.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus
