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
from typing import Any
from urllib.parse import quote

from nemo_rl.environments.nemo_gym_task import NemoGymHarnessRouting

HARNESS_ROUTING_METRIC_PREFIX = "nemo_gym/routing/"


def harness_routing_metrics(
    routing: NemoGymHarnessRouting,
    *,
    reward: float,
    valid: bool,
    masked: bool,
    generation_tokens: int,
    total_tokens: int,
) -> dict[str, float]:
    """Count one logical rollout using declared task and actual execution tags."""
    scopes = [
        f"benchmark/{quote(routing['benchmark'], safe='')}",
        f"env/{quote(routing['env'], safe='')}",
    ]
    if routing["executed_harness"] is not None:
        scopes.append(f"harness/{quote(routing['executed_harness'], safe='')}")
        scopes.append(
            f"benchmark/{quote(routing['benchmark'], safe='')}/harness/"
            f"{quote(routing['executed_harness'], safe='')}"
        )
    values = {
        "rollout_count": 1.0,
        "valid_rollout_count": float(valid),
        "invalid_rollout_count": float(not valid),
        "masked_rollout_count": float(masked),
        "reward_count": float(valid),
        "reward_sum": float(reward) if valid else 0.0,
        "generation_token_count": float(generation_tokens),
        "total_token_count": float(total_tokens),
        "masked_generation_token_count": float(generation_tokens if masked else 0),
    }
    return {
        f"{HARNESS_ROUTING_METRIC_PREFIX}{scope}/{name}": value
        for scope in scopes
        for name, value in values.items()
    }


def aggregate_harness_routing_metrics(
    per_group_metrics: dict[str, list[Any]],
) -> dict[str, float]:
    """Pool additive counters before computing reward means over unequal groups."""
    totals = {
        name: float(sum(values))
        for name, values in per_group_metrics.items()
        if name.startswith(HARNESS_ROUTING_METRIC_PREFIX)
    }
    for name, count in tuple(totals.items()):
        if name.endswith("/reward_count") and count:
            prefix = name.removesuffix("/reward_count")
            totals[f"{prefix}/reward_mean"] = totals[f"{prefix}/reward_sum"] / count
    return totals


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
