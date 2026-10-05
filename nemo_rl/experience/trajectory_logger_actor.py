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

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import DataPlaneConfig, KVBatchMeta
from nemo_rl.experience.trajectory_logger import TrajectoryLogWriter
from nemo_rl.utils.venvs import make_actor_runtime_env


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class TrajectoryLogActor:  # pragma: no cover
    """Fetch logged rows from the data plane and serialize writes in one process."""

    def __init__(
        self,
        dp_config: DataPlaneConfig,
        *,
        root_dir: str,
        policy_logprobs_required: bool,
    ) -> None:
        self._client = build_data_plane_client(dp_config, bootstrap=False)
        self._writer = TrajectoryLogWriter(root_dir=root_dir)
        self._policy_logprobs_required = policy_logprobs_required

    def record(self, meta: KVBatchMeta, *, step: int, chunk_index: int) -> None:
        """Read a chunk after advantage writeback, before its rows are cleared."""
        required = ["input_lengths", "advantages", "sample_mask"]
        # Policy workers write this column without updating the caller's metadata.
        if self._policy_logprobs_required:
            required.append("prev_logprobs")
        optional = [*TrajectoryLogWriter.FETCH_FIELDS, "values", "returns"]
        fields = required + [
            name
            for name in optional
            if name in (meta.fields or ()) and name not in required
        ]
        data = self._client.get_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=fields,
        )
        self._writer.record(
            meta,
            data,
            advantages=data["advantages"],
            final_sample_mask=data["sample_mask"],
            step=step,
            chunk_index=chunk_index,
            values=data.get("values"),
            returns=data.get("returns"),
        )

    def commit_step(self, step: int) -> None:
        """Publish only after the controller has checked every chunk's result."""
        self._writer.commit_step(step)


def create_trajectory_log_actor(
    dp_config: DataPlaneConfig,
    *,
    root_dir: str,
    policy_logprobs_required: bool,
) -> ray.actor.ActorHandle:
    """Create and validate the writer on the driver's node for local log paths."""
    actor = TrajectoryLogActor.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.experience.trajectory_logger_actor.TrajectoryLogActor"
        ),
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        ),
    ).remote(
        dp_config,
        root_dir=root_dir,
        policy_logprobs_required=policy_logprobs_required,
    )
    try:
        ray.get(actor.__ray_ready__.remote())
    except ray.exceptions.RayError:
        try:
            ray.kill(actor, no_restart=True)
        except ray.exceptions.RayError:
            pass
        raise
    return actor
