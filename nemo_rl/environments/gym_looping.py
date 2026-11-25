# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Optional passive Gym looping metrics, independent of reward processing."""

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class GymLoopingConfig(BaseModel, extra="forbid"):
    """Resolved training-side resource settings forwarded to Gym's analysis executor."""

    enabled: bool = False
    workers: int = Field(default=1, ge=1)
    max_pending: int = Field(default=16, ge=1)


def looping_worker_cpus(config: Mapping[str, Any]) -> int:
    """Return additional CPUs to reserve for enabled passive analysis workers."""
    looping_config = config.get("looping")
    if looping_config is None:
        return 0
    settings = GymLoopingConfig.model_validate(looping_config)
    if not settings.enabled:
        return 0
    return settings.workers


async def collect_looping_snapshots(
    actors: Mapping[str, Any], *, timeout_s: float, drain: bool = False
) -> dict[str, float]:
    """Collect actor snapshots concurrently within one telemetry deadline."""
    if not actors:
        return {}

    async def read_snapshot(
        actor_name: str, actor: Any
    ) -> tuple[str, Mapping[str, float | str | None] | None]:
        try:
            if drain:
                reference = actor.finish_looping_metrics.remote(timeout_s=timeout_s)
            else:
                reference = actor.looping_metrics_snapshot.remote()
            actor_snapshot = await asyncio.wait_for(reference, timeout=timeout_s)
        except Exception:
            return actor_name, None
        return actor_name, actor_snapshot

    snapshot_reads = [read_snapshot(name, actor) for name, actor in actors.items()]
    snapshots = await asyncio.gather(*snapshot_reads)
    metrics: dict[str, float] = {}
    pooled_counts: defaultdict[str, float] = defaultdict(float)
    unavailable_actors = 0
    for actor_name, snapshot in snapshots:
        if snapshot is None:
            unavailable_actors += 1
            metrics[f"looping/{actor_name}/snapshot_unavailable"] = 1.0
            continue

        for metric_name, value in snapshot.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            numeric_value = float(value)
            metrics[f"{actor_name}/{metric_name}"] = numeric_value
            if metric_name.endswith("_total"):
                pooled_counts[metric_name] += numeric_value

    metrics["looping/snapshot_unavailable_actors"] = float(unavailable_actors)
    if unavailable_actors:
        return metrics

    metrics.update(pooled_counts)
    for metric_name, assessed_count in pooled_counts.items():
        if not metric_name.endswith("/assessed_total") or not assessed_count:
            continue
        prefix = metric_name.removesuffix("/assessed_total")
        flagged_count = pooled_counts[f"{prefix}/flagged_total"]
        metrics[f"{prefix}/rate_assessed"] = flagged_count / assessed_count
    return metrics
