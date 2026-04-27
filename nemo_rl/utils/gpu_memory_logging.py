import os
from typing import Any, cast

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from nemo_rl.utils.nvml import get_full_nvml_snapshot


@ray.remote(num_cpus=0)
def _collect_nvml_snapshot() -> dict[str, Any]:
    """Return NVML snapshot for the local node."""
    return get_full_nvml_snapshot()


def _collect_all_nodes_nvml_snapshots() -> list[dict[str, Any]]:
    """Gather NVML snapshots from all alive Ray nodes."""
    nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    futures: list[ray.ObjectRef] = []
    for n in nodes:
        node_id = n["NodeID"]
        futures.append(
            _collect_nvml_snapshot.options(
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=True
                ),
            ).remote()
        )
    if not futures:
        return []
    ready, _ = ray.wait(futures, num_returns=len(futures), timeout=2.0)
    if not ready:
        return []
    return cast(list[dict[str, Any]], ray.get(ready))


def maybe_log_gpu_memory() -> None:
    """Print per-GPU/process memory for all processes & GPUs on all nodes."""
    if os.environ.get("NEMO_RL_LOG_GPU_MEMORY", "0") != "1":
        return
    try:
        snapshots = _collect_all_nodes_nvml_snapshots()
        for s in snapshots:
            node = s.get("node", "?")
            for d in s.get("devices", []):
                gpu_used = d.get("usedMiB", 0)
                gpu_total = d.get("totalMiB", 0)
                gpu_idx = d.get("index", -1)
                procs = d.get("processes", [])
                for proc in procs:
                    pid = proc.get("pid", -1)
                    name = proc.get("name", "")
                    used = proc.get("usedMiB", 0)
                    print(f"[GPU mem] {node} gpu {gpu_idx} {gpu_used:5.0f}/{gpu_total} MiB | {used:5.0f} MiB {pid:6d} {name}")
    except Exception as e:
        print(f"[GPU mem] failed to log GPU memory: {e}")
