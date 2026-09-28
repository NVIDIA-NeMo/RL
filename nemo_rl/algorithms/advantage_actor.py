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
"""CPU Ray actors that run the advantage stage off the controller."""

from __future__ import annotations

from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    AdvantageComputer,
    AdvantageOutcome,
    AdvantageRequest,
    AdvantageStageConfig,
)
from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.utils.rpc_guard import assert_metadata_only


@ray.remote(
    num_cpus=1,
    num_gpus=0,
    max_restarts=0,
    max_task_retries=0,
)
class AdvantageActor:  # pragma: no cover
    """Own a connect-only DataPlane client and one advantage computer."""

    def __init__(
        self,
        dp_config: DataPlaneConfig,
        config: AdvantageStageConfig,
        advantage_estimator: Any,
    ) -> None:
        self._computer = AdvantageComputer(
            build_data_plane_client(dp_config, bootstrap=False),
            config=config,
            advantage_estimator=advantage_estimator,
        )

    def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run an owner-local checkpoint command; return metadata, never payloads.

        This actor writes the ``advantages`` column through its own connect-only
        client, so its local store holds rows a data-plane snapshot must see.
        """
        return run_checkpoint_command(body)

    async def run(self, request: AdvantageRequest) -> AdvantageOutcome:
        """Run one advantage stage without letting payloads cross Ray RPC."""
        assert_metadata_only(request)
        outcome = await self._computer.run(request)
        assert_metadata_only(outcome)
        return outcome


def _placement_node_ids() -> list[str]:
    """Alive node ids that are not the dedicated Ray head, in a stable order.

    The head under ray.sub's DEDICATED_RAY_HEAD=1 is compute-free and hosts the
    controller, which claims the whole `ray_head` unit so nothing else lands
    there. That claim only excludes actors that request `ray_head`, and these
    actors request plain CPUs, so without an explicit strategy Ray's hybrid
    default would pack them onto the driver's node -- the head -- and put the
    advantage stage's host-memory peak right back beside the controller.

    Returns an empty list on a cluster with no dedicated head, where there is
    no node to single out; `create_advantage_actors` spreads instead.
    """
    if ray.cluster_resources().get("ray_head", 0) < 1:
        return []
    return sorted(
        node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive") and not node.get("Resources", {}).get("ray_head")
    )


def create_advantage_actors(
    dp_config: DataPlaneConfig,
    config: AdvantageStageConfig,
    advantage_estimator: Any,
    *,
    num_workers: int,
) -> list[Any]:
    """Construct the pool after TQ partitions are registered."""
    if num_workers <= 0:
        raise ValueError(f"num_advantage_workers must be positive, got {num_workers}")
    node_ids = _placement_node_ids()
    actors = []
    for index in range(num_workers):
        options: dict[str, Any] = {}
        if node_ids:
            # Round-robin so a pool larger than the cluster still spreads
            # evenly. Soft for the same reason SPREAD is below: steering off
            # the head is an optimization, not a requirement. A hard pin
            # deadlocks the untimed readiness wait with no diagnostic whenever
            # the chosen node cannot fit num_cpus=1 -- drained, or its CPUs
            # already taken by the GPU worker actors.
            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=node_ids[index % len(node_ids)], soft=True
            )
        else:
            # No head to steer away from, but Ray's hybrid default packs to 50%
            # of a node before spreading and a CPU-only actor sits far below
            # that, so every worker would land on the driver's node and stack
            # its host-memory peak beside the controller. SPREAD is soft, which
            # is what we want here: placement is an optimization, not a
            # requirement, and a pool larger than the cluster must still start.
            options["scheduling_strategy"] = "SPREAD"
        actors.append(
            AdvantageActor.options(**options).remote(
                dp_config, config, advantage_estimator
            )
        )
    try:
        ray.get([actor.__ray_ready__.remote() for actor in actors])
    except ray.exceptions.RayError:
        # Cleanup errors must not hide the startup failure or skip other actors.
        for actor in actors:
            try:
                ray.kill(actor)
            except Exception as error:
                print(f"advantage actor cleanup failed: {error}", flush=True)
        raise
    return actors
