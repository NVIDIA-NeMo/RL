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

"""Ray worker for trajectory Parquet logging."""

from __future__ import annotations

import contextlib

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import DataPlaneConfig, KVBatchMeta
from nemo_rl.experience.trajectory_logger import FETCH_FIELDS, TrajectoryLogger
from nemo_rl.utils.venvs import make_actor_runtime_env


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class TrajectoryLoggerActor:  # pragma: no cover
    """Fetch logged rows from the data plane and record trajectories to disk."""

    def __init__(self, dp_config: DataPlaneConfig, root_dir: str) -> None:
        self._client = build_data_plane_client(dp_config, bootstrap=False)
        self._writer = TrajectoryLogger(root_dir=root_dir)

    def record(self, meta: KVBatchMeta, *, step: int, chunk_index: int) -> None:
        fields = list(FETCH_FIELDS) + [
            name
            for name in ("prev_logprobs", "values", "returns")
            if name in (meta.fields or ())
        ]
        data = self._client.get_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=fields,
        )
        self._writer.record(meta, data, step=step, chunk_index=chunk_index)

    def commit_step(self, step: int) -> None:
        self._writer.commit_step(step)


def create_trajectory_logger_actor(
    dp_config: DataPlaneConfig, root_dir: str
) -> ray.actor.ActorHandle:
    """Create a log writer on the driver's node."""
    actor = TrajectoryLoggerActor.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.experience.trajectory_logger_actor.TrajectoryLoggerActor"
        ),
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        ),
    ).remote(dp_config, root_dir)
    try:
        ray.get(actor.__ray_ready__.remote())
    except ray.exceptions.RayError:
        with contextlib.suppress(ray.exceptions.RayError):
            ray.kill(actor, no_restart=True)
        raise
    return actor
