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
"""CPU storage-unit placement, and the actors that own Mooncake memory.

Storage units hold all data-plane memory so no compute process does. Both
backends place them the same way: ``storage_unit_placement`` names the
clusters whose nodes host them, and :func:`plan_storage_unit_nodes` turns
that into one Ray node ID per unit, round-robin over those nodes.

- ``mooncake_cpu`` with ``storage_unit_segment_size > 0``: every other process
  (trainers, generation, controller) is a client that owns nothing, so a
  checkpoint save calls only these :class:`MooncakeStorageUnit` actors.
- ``simple`` with ``storage_unit_placement`` set: TQ's own SimpleStorageUnits
  are pinned to the planned nodes instead of TQ's SPREAD placement group.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import ray
from ray.util.placement_group import placement_group_table
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.data_plane.interfaces import backend_config
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.utils.venvs import make_actor_runtime_env

logger = logging.getLogger(__name__)

# A unit hard-pinned to a node with no free CPU would otherwise wait forever.
_UNIT_STARTUP_TIMEOUT_S = 600


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class MooncakeStorageUnit:  # pragma: no cover
    """Own a Mooncake segment and serve checkpoint commands; nothing else."""

    def __init__(self, dp_config: DataPlaneConfig) -> None:
        self._dp_client = build_data_plane_client(
            dp_config,
            bootstrap=False,
            segment_size=backend_config(dp_config).storage_unit_segment_size,
        )

    def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run an owner-local checkpoint command; return metadata, never payloads."""
        return run_checkpoint_command(body)


def select_clusters(
    placement: Sequence[str] | str,
    clusters_by_name: Mapping[str, RayVirtualCluster],
) -> list[RayVirtualCluster]:
    """The clusters ``storage_unit_placement`` names (``"all"``: every one).

    Names are the keys of ``clusters_by_name`` (``train``, ``inference``,
    ``teacher:<name>``). A colocated run registers one cluster under two
    names, so excluding one of them excludes nothing; that is logged.
    """
    if placement == "all":
        return list(clusters_by_name.values())
    names = [placement] if isinstance(placement, str) else list(placement)
    unknown = sorted(set(names) - clusters_by_name.keys())
    if unknown:
        raise ValueError(
            f"storage_unit_placement {unknown} not in {sorted(clusters_by_name)}"
        )
    chosen = [clusters_by_name[name] for name in names]
    also = sorted(
        name
        for name, cluster in clusters_by_name.items()
        if name not in names and any(cluster is c for c in chosen)
    )
    if also:
        logger.warning(
            "storage_unit_placement %s also covers %s: they share nodes",
            names,
            also,
        )
    return chosen


def storage_node_ids(
    clusters: Sequence[RayVirtualCluster], count: int | None
) -> list[str]:
    """Ray node ID per storage unit, round-robin over the clusters' nodes.

    ``count`` None means 2 per selected node.
    """
    # Creates (and waits for) any placement group not yet reserved, so the
    # table read below sees them all.
    pgs = [pg for cluster in clusters for pg in cluster.get_placement_groups()]
    # One GCS read for every placement group; no actor RPC.
    table = placement_group_table()
    nodes = sorted(
        {
            node_id
            for pg in pgs
            for node_id in table[pg.id.hex()]["bundles_to_node_id"].values()
        }
    )
    # Every selected node gets count // len(nodes) units, or one more.
    return [nodes[i % len(nodes)] for i in range(count or 2 * len(nodes))]


def plan_storage_unit_nodes(
    dp_config: DataPlaneConfig,
    clusters_by_name: Mapping[str, RayVirtualCluster],
) -> list[str] | None:
    """One Ray node ID per storage unit, or None when storage units are off.

    Off unless ``mooncake_cpu.storage_unit_segment_size > 0`` or
    ``simple.storage_unit_placement`` is set.
    """
    backend = dp_config["backend"]
    if backend == "simple":
        if "simple" not in dp_config:
            return None
        placement = backend_config(dp_config).storage_unit_placement
        if placement is None:
            return None
    elif backend == "mooncake_cpu":
        if backend_config(dp_config).storage_unit_segment_size == 0:
            return None
        placement = backend_config(dp_config).storage_unit_placement
    else:
        return None
    return storage_node_ids(
        select_clusters(placement, clusters_by_name),
        backend_config(dp_config).num_storage_units,
    )


def start_storage_units(
    dp_config: DataPlaneConfig, node_ids: Sequence[str] | None
) -> tuple[Any, ...]:
    """Start one MooncakeStorageUnit per entry of ``node_ids``.

    Returns ``()`` for any other backend or when ``node_ids`` is None.
    """
    if dp_config["backend"] != "mooncake_cpu" or node_ids is None:
        return ()
    runtime_env = make_actor_runtime_env(
        "nemo_rl.data_plane.mooncake_storage_unit.MooncakeStorageUnit"
    )
    units = tuple(
        MooncakeStorageUnit.options(
            runtime_env=runtime_env,
            # Hard pin: the plan is the placement.
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        ).remote(dp_config)
        for node_id in node_ids
    )
    try:
        ray.get(
            [unit.__ray_ready__.remote() for unit in units],
            timeout=_UNIT_STARTUP_TIMEOUT_S,
        )
    except (ray.exceptions.RayError, TimeoutError) as e:
        for unit in units:
            ray.kill(unit)
        raise RuntimeError(
            f"MooncakeStorageUnit startup failed on nodes {sorted(set(node_ids))}: "
            f"{e}. Each unit needs 1 free CPU on its node."
        ) from e
    return units
