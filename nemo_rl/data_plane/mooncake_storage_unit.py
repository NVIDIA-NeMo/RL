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
"""CPU actors that are the only Mooncake memory owners -- SimpleStorageUnit's shape.

With ``mooncake_cpu.storage_unit_segment_size > 0`` every other process
(trainers, vLLM, controller) is a client, so a checkpoint save calls only
these units: ``num_storage_units`` of them, spread evenly over the nodes
``storage_unit_placement`` selects. vLLM capture puts prefer a unit on their
own node.
"""

from __future__ import annotations

from typing import Any

import ray
from ray.util.placement_group import placement_group_table
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import (
    local_segment_name,
    run_checkpoint_command,
)
from nemo_rl.data_plane.interfaces import backend_config
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.utils.venvs import make_actor_runtime_env


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class MooncakeStorageUnit:  # pragma: no cover
    """Own a Mooncake segment and serve checkpoint commands; nothing else."""

    def __init__(self, dp_config: DataPlaneConfig) -> None:
        self._dp_client = build_data_plane_client(
            dp_config,
            bootstrap=False,
            segment_size=backend_config(dp_config).storage_unit_segment_size,
        )
        self._segment = local_segment_name()

    def segment(self) -> str:
        """Mooncake segment name, for writers on the same node to prefer."""
        return self._segment

    def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run an owner-local checkpoint command; return metadata, never payloads."""
        return run_checkpoint_command(body)


def start_storage_units(
    dp_config: DataPlaneConfig,
    *,
    inference_cluster: RayVirtualCluster,
    train_cluster: RayVirtualCluster,
) -> tuple[tuple[Any, ...], dict[str, list[str]]]:
    """Start the units; return them and {Ray node ID: that node's segments}.

    Returns ``((), {})`` unless ``mooncake_cpu.storage_unit_segment_size > 0``.
    """
    if dp_config["backend"] != "mooncake_cpu":
        return (), {}
    mooncake_cfg = backend_config(dp_config)
    if mooncake_cfg.storage_unit_segment_size == 0:
        return (), {}
    clusters = {
        "inference": [inference_cluster],
        "train": [train_cluster],
        "all": [inference_cluster, train_cluster],
    }[mooncake_cfg.storage_unit_placement]
    # One GCS read for every placement group; no actor RPC.
    table = placement_group_table()
    nodes = sorted(
        {
            node_id
            for cluster in clusters
            for pg in cluster.get_placement_groups()
            for node_id in table[pg.id.hex()]["bundles_to_node_id"].values()
        }
    )
    count = mooncake_cfg.num_storage_units or 2 * len(nodes)
    # Round-robin: every selected node gets count // len(nodes) units, or one more.
    node_ids = [nodes[i % len(nodes)] for i in range(count)]
    runtime_env = make_actor_runtime_env(
        "nemo_rl.data_plane.mooncake_storage_unit.MooncakeStorageUnit"
    )
    units = tuple(
        MooncakeStorageUnit.options(
            runtime_env=runtime_env,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        ).remote(dp_config)
        for node_id in node_ids
    )
    segments = ray.get([unit.segment.remote() for unit in units])
    by_node: dict[str, list[str]] = {}
    for node_id, segment in zip(node_ids, segments, strict=True):
        by_node.setdefault(node_id, []).append(segment)
    return units, by_node
