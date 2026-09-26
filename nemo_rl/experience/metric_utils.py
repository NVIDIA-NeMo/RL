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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

# Fixed scalar buckets match the miles telemetry convention.
_STALENESS_HISTOGRAM_MAX = 40
ROLLOUT_CATEGORY_TAG = "rollout_category"
READY_WEIGHT_VERSION_TAG = "ready_weight_version"


def resolve_rollout_category(
    *, extra_env_info: Mapping[str, Any] | None, task_name: str | None
) -> str:
    """Resolve a telemetry category from task source, agent name, or task name.

    Missing/empty labels fall back to the next source, then ``unknown``.
    Non-string labels raise ValueError rather than merging unrelated categories.
    """
    candidates = []
    if extra_env_info is not None:
        candidates.append(extra_env_info.get("task_source"))
        agent_ref = extra_env_info.get("agent_ref")
        if isinstance(agent_ref, Mapping):
            candidates.append(agent_ref.get("name"))
    candidates.append(task_name)
    for value in candidates:
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Rollout category must be a string, got {value!r}")
        if value.strip():
            return value.strip()
    return "unknown"


def is_histogram_metric(name: str) -> bool:
    """Return whether a metric key represents raw histogram observations."""
    return name.startswith("histogram/") or name.endswith("/histogram")


def calculate_staleness_metrics(
    sample_versions: Iterable[tuple[str, int]],
    *,
    train_weight_version: int,
    sample_categories: Mapping[str, str | None] | None = None,
    sample_ready_versions: Mapping[str, int | None] | None = None,
) -> dict[str, float]:
    """Summarize realized version lag once per selected prompt group.

    Args:
        sample_versions: Canonical sample IDs and their weight-version tags,
            pooled across all selected streaming chunks of one training step.
            IDs use the ``{group_id}_g{generation_index}`` payload convention;
            IDs without that suffix identify individual groups. Row filtering
            and loss masks do not change this selected-group population.
        train_weight_version: Trainer version before this step's update, using
            the same clock as the sampler (including PPO critic warmup).
        sample_categories: Optional category tags keyed by canonical sample ID.
            When provided, also report each category under
            ``staleness/category/<escaped label>/*``. Missing labels belong to
            ``unknown``; siblings must agree on their category.
        sample_ready_versions: Optional trainer versions at the instant groups
            became selectable, keyed by sample ID. Enables pre-queue (ready
            minus generation) and in-queue (train minus ready) statistics.
            Missing versions from older checkpoints contribute to total only;
            decomposition coverage reports the measured population.

    Returns:
        Fully qualified total and optional category metrics, or an empty dict
        when there are no groups. Variance is population variance (ddof=0).

    Raises:
        ValueError: A weight version is invalid or newer than the trainer,
            or sibling categories/ready versions disagree.
    """
    if (
        isinstance(train_weight_version, bool)
        or not isinstance(train_weight_version, int)
        or train_weight_version < 0
    ):
        raise ValueError("train_weight_version must be a non-negative integer")
    group_versions: dict[str, int] = {}
    group_categories: dict[str, str] = {}
    group_ready_versions: dict[str, int | None] = {}
    for sample_id, version in sample_versions:
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not 0 <= version <= train_weight_version
        ):
            raise ValueError(
                f"Invalid weight version {version!r} for sample {sample_id!r} "
                f"at trainer version {train_weight_version}"
            )
        # Keep the same canonical-ID convention as SC._group_ids_from_meta.
        group_id, separator, generation_index = sample_id.rpartition("_g")
        if not (group_id and separator and generation_index.isdigit()):
            group_id = sample_id
        if sample_ready_versions is not None:
            ready = sample_ready_versions.get(sample_id)
            if ready is not None and (
                isinstance(ready, bool)
                or not isinstance(ready, int)
                or not version <= ready <= train_weight_version
            ):
                raise ValueError(
                    f"Invalid ready weight version {ready!r} for sample "
                    f"{sample_id!r}: generation={version}, train={train_weight_version}"
                )
            if (
                group_id in group_ready_versions
                and group_ready_versions[group_id] != ready
            ):
                raise ValueError(f"Inconsistent ready versions for group {group_id!r}")
            group_ready_versions[group_id] = ready
        if sample_categories is not None:
            category = resolve_rollout_category(
                extra_env_info=None, task_name=sample_categories.get(sample_id)
            )
            if group_id in group_categories and group_categories[group_id] != category:
                raise ValueError(
                    f"Inconsistent rollout categories for group {group_id!r}: "
                    f"{group_categories[group_id]!r} and {category!r}"
                )
            group_categories[group_id] = category
        if group_id in group_versions:
            group_versions[group_id] = min(group_versions[group_id], version)
        else:
            group_versions[group_id] = version
    if not group_versions:
        return {}

    lags = [train_weight_version - version for version in group_versions.values()]
    metrics = _summarize_staleness(lags, prefix="staleness/total")
    populations = {"staleness": list(group_versions)}
    if sample_categories is not None:
        category_groups: dict[str, list[str]] = defaultdict(list)
        for group_id in group_versions:
            category_groups[group_categories[group_id]].append(group_id)
        for category, group_ids in sorted(category_groups.items()):
            # Preserve category hierarchy; encode literal '%' so "a/b" and
            # "a%2Fb" still have distinct metric keys.
            prefix = f"staleness/category/{quote(category, safe='/')}"
            values = [train_weight_version - group_versions[g] for g in group_ids]
            # Keep legacy aliases only for flat labels. A hierarchical alias
            # for "math/total" would overwrite "math"'s canonical total keys.
            if "/" not in category:
                metrics.update(_summarize_staleness(values, prefix=prefix))
            metrics.update(_summarize_staleness(values, prefix=f"{prefix}/total"))
            populations[prefix] = group_ids
    if sample_ready_versions is not None:
        for prefix, group_ids in populations.items():
            pre_queue: list[int] = []
            in_queue: list[int] = []
            for group_id in group_ids:
                ready = group_ready_versions[group_id]
                if ready is not None:
                    pre_queue.append(ready - group_versions[group_id])
                    in_queue.append(train_weight_version - ready)
            metrics.update(
                {
                    f"{prefix}/decomposition/num_groups": float(len(pre_queue)),
                    f"{prefix}/decomposition/missing_num_groups": float(
                        len(group_ids) - len(pre_queue)
                    ),
                    f"{prefix}/decomposition/frac_known": len(pre_queue)
                    / len(group_ids),
                }
            )
            if pre_queue:
                metrics.update(
                    _summarize_staleness(pre_queue, prefix=f"{prefix}/pre_queue")
                )
                metrics.update(
                    _summarize_staleness(in_queue, prefix=f"{prefix}/in_queue")
                )
    return metrics


def _summarize_staleness(lags: list[int], *, prefix: str) -> dict[str, float]:
    """Summarize a nonempty, group-weighted population of integer lags."""
    ordered = sorted(lags)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    counts = Counter(lags)
    variance = statistics.pvariance(lags)
    metrics: dict[str, float] = {
        "mean": statistics.fmean(lags),
        "variance": float(variance),
        "std": math.sqrt(variance),
        "max": float(max(lags)),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "frac_zero": counts[0] / len(lags),
        "num_groups": float(len(lags)),
    }
    metrics.update(
        {
            f"count_{level}": float(counts[level])
            for level in range(_STALENESS_HISTOGRAM_MAX + 1)
        }
    )
    metrics[f"count_ge_{_STALENESS_HISTOGRAM_MAX + 1}"] = float(
        sum(count for lag, count in counts.items() if lag > _STALENESS_HISTOGRAM_MAX)
    )
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


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
