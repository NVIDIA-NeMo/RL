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

"""Shared aggregation helpers for rollout metrics."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class RolloutTelemetry:
    """Token-free observations for one completed sibling, safe to checkpoint.

    Store observations rather than derived statistics (a singleton's standard
    deviation is NaN). This also keeps duplicate-seal comparisons deterministic.
    Arbitrary numeric environment diagnostics remain supported by metric name.
    """

    environment: str
    observations: dict[str, float]
    scalars: dict[str, float]

    @classmethod
    def from_metrics(cls, environment: str, metrics: dict[str, Any]) -> Self:
        """Snapshot singleton distributions, excluding tables and payloads."""
        observations: dict[str, float] = {}
        scalars: dict[str, float] = {}
        summaries = {"mean", "min", "max", "median", "stddev"}
        for key, value in metrics.items():
            if key.endswith("/histogram"):
                if not isinstance(value, list) or len(value) != 1:
                    raise ValueError("Sibling telemetry requires singleton histograms")
                observations[key.removesuffix("/histogram")] = float(value[0])
            elif isinstance(value, (int, float)) and not (
                key.rsplit("/", 1)[-1] in summaries
                and f"{key.rsplit('/', 1)[0]}/histogram" in metrics
            ):
                scalars[key] = float(value)
        return cls(environment, observations, scalars)

    def to_metrics(self) -> dict[str, Any]:
        """Recreate reducer input; selected-step aggregation pools observations."""
        metrics: dict[str, Any] = dict(self.scalars)
        for name, value in self.observations.items():
            metrics.update(calculate_single_metric([value], 1, name))
        return metrics

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        """Validate a primitive-only recovery snapshot, without loading tokens."""
        if not isinstance(state, dict) or set(state) != {
            "environment",
            "observations",
            "scalars",
        }:
            raise ValueError("Invalid rollout telemetry snapshot fields")
        environment = state["environment"]
        if not isinstance(environment, str) or not environment:
            raise ValueError("Rollout telemetry requires an environment name")
        for name in ("observations", "scalars"):
            values = state[name]
            if not isinstance(values, dict) or not all(
                isinstance(key, str) and isinstance(value, (int, float))
                for key, value in values.items()
            ):
                raise ValueError(
                    f"Rollout telemetry {name} must contain numeric scalars"
                )
        return cls(environment, dict(state["observations"]), dict(state["scalars"]))


def is_histogram_metric(name: str) -> bool:
    """Return whether a metric key represents raw histogram observations."""
    return name.startswith("histogram/") or name.endswith("/histogram")


def calculate_single_metric(
    values: Sequence[float | int], batch_size: int, key_name: str
) -> dict:
    """Compute summary statistics for a metric as slash-prefixed keys.

    Args:
        values: Per-sample metric values to aggregate.
        batch_size: Denominator for the mean (sum(values) / batch_size, not len(values)); stddev still uses len(values).
        key_name: Prefix for the returned metric keys (e.g. "total_reward").

    Returns:
        Dict mapping "{key_name}/{stat}" to its value for stat in mean, max, min,
        median, stddev (nan for a single value), and histogram. Histogram values
        remain backend-agnostic raw observations until the logger serializes them.
    """
    return {
        f"{key_name}/mean": sum(values) / batch_size,
        f"{key_name}/max": max(values),
        f"{key_name}/min": min(values),
        f"{key_name}/median": statistics.median(values),
        f"{key_name}/stddev": statistics.stdev(values) if len(values) > 1 else math.nan,
        f"{key_name}/histogram": list(values),
    }


def pct(values: Sequence[float | int], p: float) -> float:
    """Percentile helper for buffer starvation diagnostics."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
    return float(sorted_v[idx])
