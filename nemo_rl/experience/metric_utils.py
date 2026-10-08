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

import hashlib
import math
import re
import statistics
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Self, cast

_METRIC_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

Reduction = Literal[
    "sum", "mean", "min", "max", "median", "dist", "concat", "sum_by_key", "field_dist"
]


class Metric(NamedTuple):
    """Observations and their producer-declared, once-per-step reduction.

    Numeric reductions use scalar observations. "concat" preserves raw
    histograms or opt-in log tables; "sum_by_key" combines worker counters.
    These two explicit reductions keep non-scalar logging out of name heuristics.
    "field_dist" preserves optional-field rows, including empty rows, so means
    include groups where a field is entirely absent.
    """

    values: list[Any]
    reduce: Reduction


def distribution(
    name: str, values: Sequence[float | int], *, rows: int | None = None
) -> dict[str, Metric]:
    """Declare a distribution and its independently weighted mean.

    Args:
        name: Output metric family.
        values: Observed values, excluding rows without this field.
        rows: Total rows contributing to the mean. Missing observations count
            as zero for the mean, not for other distribution statistics.

    Returns:
        Distribution and mean declarations; neither computes group statistics.

    Raises:
        ValueError: If rows is negative or smaller than the observed population.
    """
    population = len(values) if rows is None else rows
    if population < len(values) or population < 0:
        raise ValueError("Distribution rows must cover all observed values")
    return {
        name: Metric(list(values), "dist"),
        f"{name}/mean": Metric(
            list(values) + [0.0] * (population - len(values)), "mean"
        ),
    }


def optional_distributions(
    prefix: str, rows: Sequence[dict[str, float | int]]
) -> dict[str, Metric]:
    """Declare optional numeric fields with a shared row population.

    Args:
        prefix: Output namespace chosen by the producer.
        rows: Numeric fields present on each row, including empty mappings.

    Returns:
        A field-distribution declaration containing fully qualified output names.
        Every row contributes to means; other statistics use present values only.
    """
    return {
        f"{prefix}/": Metric(
            [
                {f"{prefix}/{field}": value for field, value in row.items()}
                for row in rows
            ],
            "field_dist",
        )
    }


def _distribution_stats(
    name: str, values: Sequence[float | int], rows: int
) -> dict[str, Any]:
    result = calculate_single_metric(values, rows, name)
    result[f"{name}/p50"] = result[f"{name}/median"]
    result[f"{name}/p95"] = pct(values, 95)
    result[f"{name}/p99"] = pct(values, 99)
    return result


def reduce_step(groups: Sequence[dict[str, Metric]]) -> dict[str, Any]:
    """Pool observations across selected groups and apply declared reducers.

    No metric name determines a reduction. Distribution outputs are min, max,
    median, sample stddev, p50, discrete p95/p99, and raw histogram observations;
    their means are separate declarations so missing fields need not affect
    other statistics. Inputs are not mutated.

    Args:
        groups: Producer-declared metrics for the selected prompt groups.

    Returns:
        Scalars, raw histogram/table lists, and worker-counter maps for logging.

    Raises:
        TypeError: If a group contains an unconverted legacy value.
        ValueError: If reducers disagree, an unknown reducer is supplied, or
            two declarations produce the same output key.
    """
    joined: dict[str, Metric] = {}
    for group in groups:
        for name, metric in group.items():
            if not isinstance(metric, Metric):
                raise TypeError(f"Metric {name!r} has no declared reduction")
            pooled = joined.setdefault(name, Metric([], metric.reduce))
            if pooled.reduce != metric.reduce:
                raise ValueError(f"Groups disagree on how to reduce {name!r}")
            pooled.values.extend(metric.values)
    out: dict[str, Any] = {}
    reducers = {
        "sum": sum,
        "mean": statistics.fmean,
        "min": min,
        "max": max,
        "median": statistics.median,
    }
    optional_values: dict[str, list[float]] = {}
    optional_rows: dict[str, int] = {}
    for name, (values, reduction) in joined.items():
        if reduction == "concat":
            reduced = {name: values}
        elif reduction == "sum_by_key":
            counts: dict[Any, float] = {}
            for worker_counts in values:
                for worker, count in worker_counts.items():
                    counts[worker] = counts.get(worker, 0) + count
            reduced = {name: counts}
        elif reduction == "dist":
            if not values:
                continue
            reduced = _distribution_stats(name, values, len(values))
            del reduced[f"{name}/mean"]
        elif reduction == "field_dist":
            # Names and the population are explicit producer data, not inferred
            # from agent names or environment/sample_count output keys. Two raw
            # aliases may share an output name; pool both declared populations.
            fields = dict.fromkeys(field for row in values for field in row)
            for field in fields:
                optional_values.setdefault(field, []).extend(
                    row[field] for row in values if field in row
                )
                optional_rows[field] = optional_rows.get(field, 0) + len(values)
            continue
        elif reduction in reducers:
            if not values:
                continue
            reduced = {name: reducers[reduction](values)}
        else:
            raise ValueError(f"Unknown metric reduction {reduction!r} for {name!r}")
        collisions = out.keys() & reduced.keys()
        if collisions:
            raise ValueError(f"Metric declarations overlap at {sorted(collisions)}")
        out.update(reduced)
    for name, observations in optional_values.items():
        reduced = _distribution_stats(name, observations, optional_rows[name])
        collisions = out.keys() & reduced.keys()
        if collisions:
            raise ValueError(f"Metric declarations overlap at {sorted(collisions)}")
        out.update(reduced)
    return out


def metrics_from_plain(plain: dict[str, Any]) -> dict[str, Metric]:
    """Adapt native V1 producer output using the pre-PR numeric name rules.

    This adapter belongs at the native producer, never at the replay boundary:
    older replay rows lack the observations/denominators needed for safe mixing.
    Histogram-derived summaries are pooled, but numeric means preserve V1's
    average-of-group-means behavior. Non-numeric payloads are not inferred.

    Args:
        plain: One native prompt group's V1 rollout metrics.

    Returns:
        Explicit numeric and histogram declarations for the SC reducer.

    Raises:
        TypeError: If a native histogram is not a list of observations.
    """
    out: dict[str, Metric] = {}
    summaries = {"min", "max", "median", "stddev", "p50", "p95", "p99"}
    for name, value in plain.items():
        if is_histogram_metric(name):
            if not isinstance(value, list):
                raise TypeError(f"Native histogram {name!r} must contain a list")
            if name.startswith("histogram/"):
                out[name] = Metric(list(value), "concat")
            else:
                out[name.removesuffix("/histogram")] = Metric(list(value), "dist")
        elif not isinstance(value, (int, float)):
            continue
        elif (
            name.rsplit("/", 1)[-1] in summaries
            and f"{name.rsplit('/', 1)[0]}/histogram" in plain
        ):
            continue
        elif name.endswith("/min") or (
            name.startswith("min_") and not name.endswith("_rate")
        ):
            out[name] = Metric([value], "min")
        elif name.endswith("/max") or (
            name.startswith("max_") and not name.endswith("_rate")
        ):
            out[name] = Metric([value], "max")
        elif name == "total_turns":
            out[name] = Metric([value], "sum")
        else:
            out[name] = Metric([value], "mean")
    return out


def rollout_environment_metric_component(environment: str) -> str:
    """Return a readable metric path component without silent collisions."""
    sanitized = _METRIC_COMPONENT_PATTERN.sub("_", environment).strip("_.")
    if sanitized == environment:
        return sanitized
    digest = hashlib.blake2s(environment.encode(), digest_size=8).hexdigest()
    return f"{sanitized or 'unknown'}-{digest}"


@dataclass(frozen=True)
class RolloutTelemetry:
    """Token-free, producer-declared metrics for one completed sibling."""

    environment: str
    metrics: dict[str, Metric]

    @classmethod
    def from_metrics(cls, environment: str, metrics: dict[str, Metric]) -> Self:
        """Copy numeric declarations without inferring summary names.

        Args:
            environment: Resolved Gym agent name, before metric-path sanitization.
            metrics: One sibling's producer-declared observations and reductions.

        Returns:
            A primitive-only snapshot, excluding opt-in non-numeric log tables.

        Raises:
            ValueError: If a metric contains non-numeric payloads other than a
                declared concatenation, which is omitted from capture telemetry.
        """
        numeric = {}
        for name, metric in metrics.items():
            if not isinstance(metric, Metric):
                raise ValueError(
                    f"Capture telemetry requires declared metrics: {name!r}"
                )
            if metric.reduce != "field_dist" and not all(
                isinstance(value, (int, float)) for value in metric.values
            ):
                if metric.reduce == "concat":
                    continue
                raise ValueError(f"Non-numeric capture telemetry for {name!r}")
            numeric[name] = Metric(deepcopy(metric.values), metric.reduce)
        return cls.from_state(
            {
                "environment": environment,
                "metrics": {
                    name: [metric.values, metric.reduce]
                    for name, metric in numeric.items()
                },
            }
        )

    def to_metrics(self) -> dict[str, Metric]:
        """Return independent declarations for finalizer-side corrections."""
        return {
            name: Metric(deepcopy(metric.values), metric.reduce)
            for name, metric in self.metrics.items()
        }

    def to_state(self) -> dict[str, Any]:
        """Return plain lists and scalars accepted by weights-only checkpoint loading."""
        return {
            "environment": self.environment,
            "metrics": {
                name: [deepcopy(metric.values), metric.reduce]
                for name, metric in self.metrics.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        """Validate declared primitive metrics loaded from a recovery sidecar.

        Args:
            state: Serialized environment and metric declarations.

        Returns:
            A validated, independent telemetry snapshot.

        Raises:
            ValueError: If fields, environment, reductions, or observations are
                malformed. Legacy snapshots are handled by the recovery reader.
        """
        if not isinstance(state, dict) or set(state) != {"environment", "metrics"}:
            raise ValueError("Invalid rollout telemetry snapshot fields")
        environment = state["environment"]
        if not isinstance(environment, str) or not environment:
            raise ValueError("Rollout telemetry requires an environment name")
        raw_metrics = state["metrics"]
        if not isinstance(raw_metrics, dict):
            raise ValueError("Rollout telemetry metrics must be a mapping")
        metrics = {}
        for name, pair in raw_metrics.items():
            if (
                not isinstance(name, str)
                or not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not isinstance(pair[0], list)
                or not isinstance(pair[1], str)
                or pair[1]
                not in {
                    "sum",
                    "mean",
                    "min",
                    "max",
                    "median",
                    "dist",
                    "concat",
                    "field_dist",
                }
            ):
                raise ValueError("Invalid rollout telemetry metric declaration")
            if pair[1] == "field_dist":
                valid = all(
                    isinstance(row, dict)
                    and all(
                        isinstance(field, str) and isinstance(value, (int, float))
                        for field, value in row.items()
                    )
                    for row in pair[0]
                )
            else:
                valid = all(isinstance(value, (int, float)) for value in pair[0])
            if not valid:
                raise ValueError("Invalid rollout telemetry observations")
            metrics[name] = Metric(deepcopy(pair[0]), cast(Reduction, pair[1]))
        return cls(environment, metrics)


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
