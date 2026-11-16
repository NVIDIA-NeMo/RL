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

from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    AdvantageComputer,
    AdvantageOutcome,
    AdvantageRequest,
    AdvantageStageConfig,
)
from nemo_rl.data_plane import DataPlaneConfig
from nemo_rl.data_plane.delegate import (
    CPU_ACTOR_OPTIONS,
    DataPlaneDelegate,
    spread_off_head,
    start_cpu_actors,
)
from nemo_rl.utils.rpc_guard import assert_metadata_only


@ray.remote(**CPU_ACTOR_OPTIONS)
class AdvantageActor(DataPlaneDelegate):  # pragma: no cover
    """Run one advantage computer on its own connect-only data-plane client."""

    def __init__(
        self,
        dp_config: DataPlaneConfig,
        config: AdvantageStageConfig,
        advantage_estimator: Any,
        *,
        label: str = "0",
    ) -> None:
        super().__init__(dp_config, label=label)
        self._computer = AdvantageComputer(
            self.dp_client,
            config=config,
            advantage_estimator=advantage_estimator,
            shard_id=label,
        )

    async def run(self, request: AdvantageRequest) -> AdvantageOutcome:
        """Run one advantage stage without letting payloads cross Ray RPC."""
        assert_metadata_only(request)
        outcome = await self._computer.run(request)
        assert_metadata_only(outcome)
        return outcome


def create_advantage_actors(
    dp_config: DataPlaneConfig,
    config: AdvantageStageConfig,
    advantage_estimator: Any,
    *,
    num_workers: int,
) -> list[Any]:
    """Start the pool, kept off the controller's node, after TQ partitions exist."""
    placement = spread_off_head()
    return start_cpu_actors(
        lambda index: AdvantageActor.options(
            scheduling_strategy=placement(index)
        ).remote(dp_config, config, advantage_estimator, label=str(index)),
        n=num_workers,
    )
