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
"""CPU Ray actors the controller hands data-plane work to.

The advantage stage and the finalizers each run as a pool of these. A
``DataPlaneDelegate`` owns a data-plane client whose store holds rows, so it
answers the data-plane checkpoint and metrics calls. ``start_cpu_actors``
starts a pool and kills it if any actor fails to come up.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.data_plane.observability import is_metrics_client

# One CPU, no GPU, and no retries: a retried call could write twice.
CPU_ACTOR_OPTIONS: dict[str, Any] = {
    "num_cpus": 1,
    "num_gpus": 0,
    "max_restarts": 0,
    "max_task_retries": 0,
}


class DataPlaneDelegate:
    """Own a connect-only data-plane client whose store holds rows.

    Rows written through this client live in this process's Mooncake memory,
    so a data-plane checkpoint must reach it (``mooncake_checkpoint``).
    """

    def __init__(self, dp_config: DataPlaneConfig, *, label: str = "0") -> None:
        from nemo_rl.data_plane import build_data_plane_client

        self.label = label
        self.dp_client = build_data_plane_client(dp_config, bootstrap=False)

    def __repr__(self) -> str:
        # Ray prefixes this actor's log lines with its repr.
        return f"{type(self).__name__}[{self.label}]"

    def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run an owner-local checkpoint command; return metadata, never payloads."""
        return run_checkpoint_command(body)

    def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        """This client's data-plane I/O counters, if metrics are on."""
        if not is_metrics_client(self.dp_client):
            return None
        return self.dp_client.snapshot(reset_step_window=True)


def spread_off_head() -> Callable[[int], Any]:
    """Map an actor's index to a strategy that keeps it off a dedicated head.

    The head under ray.sub's DEDICATED_RAY_HEAD=1 hosts the controller and is
    compute-free. CPU-only actors that ask for plain CPUs would otherwise be
    packed onto the driver's node, the head. Soft, because a hard pin waits
    forever on a node that cannot fit ``num_cpus=1``. Without a dedicated head
    there is no node to single out, so actors spread.
    """
    if ray.cluster_resources().get("ray_head", 0) < 1:
        return lambda _: "SPREAD"
    node_ids = sorted(
        node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive") and not node.get("Resources", {}).get("ray_head")
    )
    if not node_ids:
        return lambda _: "SPREAD"
    return lambda index: NodeAffinitySchedulingStrategy(
        node_id=node_ids[index % len(node_ids)], soft=True
    )


def _kill_all(actors: list[Any]) -> None:
    # One failed kill must not skip the others or hide the startup error.
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception as error:
            print(f"cpu actor cleanup failed: {error}", flush=True)


def start_cpu_actors(
    make_actor: Callable[[int], Any],
    *,
    n: int,
    ready: str = "__ray_ready__",
    timeout_s: float | None = None,
) -> list[Any]:
    """Create ``n`` actors, wait for ``ready`` on each, kill all if one fails."""
    if n <= 0:
        raise ValueError(f"need at least one actor, got {n}")
    actors = [make_actor(index) for index in range(n)]
    try:
        ray.get([getattr(actor, ready).remote() for actor in actors], timeout=timeout_s)
    except BaseException:
        _kill_all(actors)
        raise
    return actors
