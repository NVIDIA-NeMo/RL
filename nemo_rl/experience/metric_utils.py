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
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from wandb import Histogram

MetricSamples: TypeAlias = dict[str, list[float]]
TaskMetricSamples: TypeAlias = dict[str, MetricSamples]

NEMO_GYM_OPTIONAL_METRIC_KEYS = (
    "patch_exists",
    "agent_timed_out",
    "eval_timed_out",
    "ray_queue_time",
    "openhands_run_time",
    "generation_apptainer_spinup_time",
    "final_eval_apptainer_spinup_time",
)
NEMO_GYM_AGENT_ERROR_KINDS = (
    "max_iteration",
    "context_window",
    "stuck_in_loop",
    "other",
)
_SKIP_NEMO_GYM_SEARCH_KEYS = {
    "completion",
    "completions",
    "input",
    "messages",
    "output",
    "patch",
    "response",
    "trajectory",
    "trajectories",
}


def calculate_single_metric(
    values: Sequence[float | int], batch_size: int, key_name: str
) -> dict:
    """Compute summary statistics for a metric as slash-prefixed keys.

    Args:
        values: Per-sample metric values to aggregate.
        batch_size: Denominator for the mean (sum(values) / batch_size, not len(values)); stddev still uses len(values).
        key_name: Prefix for the returned metric keys (e.g. "total_reward").

    Returns:
        Dict mapping "{key_name}/{stat}" to its value for stat in mean, max, min, median, stddev (nan for a single value), and histogram (a wandb.Histogram).
    """
    return {
        f"{key_name}/mean": sum(values) / batch_size,
        f"{key_name}/max": max(values),
        f"{key_name}/min": min(values),
        f"{key_name}/median": statistics.median(values),
        f"{key_name}/stddev": statistics.stdev(values) if len(values) > 1 else math.nan,
        f"{key_name}/histogram": Histogram(values),
    }


def pct(values: Sequence[float | int], p: float) -> float:
    """Percentile helper for buffer starvation diagnostics."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
    return float(sorted_v[idx])


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _find_nemo_gym_value(payload: Any, key: str, depth: int = 0) -> Any:
    """Find a small scalar health field without traversing large response data."""
    if depth > 4:
        return None
    if isinstance(payload, Mapping):
        if key in payload:
            return payload[key]
        for nested_key, nested_value in payload.items():
            if nested_key in _SKIP_NEMO_GYM_SEARCH_KEYS:
                continue
            found = _find_nemo_gym_value(nested_value, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_nemo_gym_value(item, key, depth + 1)
            if found is not None:
                return found
    return None


def extract_nemo_gym_metric_values(result: Mapping[str, Any]) -> dict[str, float]:
    """Extract Megatron-compatible scalar health metrics from one Gym result."""
    response = result.get("response") or {}
    output_items = (
        (response.get("output") or []) if isinstance(response, Mapping) else []
    )
    trainable_items = 0
    generated_tokens = 0
    for item in output_items:
        if not isinstance(item, Mapping):
            continue
        token_ids = item.get("generation_token_ids")
        if token_ids is None:
            continue
        trainable_items += 1
        generated_tokens += len(token_ids)

    instance_config = result.get("instance_config") or {}
    mask_sample = (
        bool(instance_config.get("mask_sample", False))
        if isinstance(instance_config, Mapping)
        else False
    )
    metrics = {
        "resolved": float(result.get("resolved") is True),
        "mask_sample": float(mask_sample),
        "trainable": float(trainable_items > 0),
        "trainable_items": float(trainable_items),
        "generated_tokens": float(generated_tokens),
        "output_items": float(len(output_items)),
    }

    for key in NEMO_GYM_OPTIONAL_METRIC_KEYS:
        value = _finite_float(_find_nemo_gym_value(result, key))
        if value is not None:
            metrics[key] = value

    agent_error_kind = _find_nemo_gym_value(result, "agent_error_kind")
    if (
        agent_error_kind is not None
        and agent_error_kind not in NEMO_GYM_AGENT_ERROR_KINDS
    ):
        agent_error_kind = "other"
    for error_kind in NEMO_GYM_AGENT_ERROR_KINDS:
        metrics[f"agent_error_kind/{error_kind}"] = float(
            agent_error_kind == error_kind
        )
    return metrics


def collect_nemo_gym_metric_samples(
    results: Sequence[Mapping[str, Any]],
    *,
    message_logs: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> MetricSamples:
    """Collect raw per-result health values for lossless later aggregation.

    NeMo-RL removes generation token IDs from the retained Gym result after
    building its message log. When message logs are supplied, use their
    assistant messages as the authoritative trainable-item/token counts.
    """
    if message_logs is not None and len(message_logs) != len(results):
        raise ValueError("message_logs and results must have the same length")
    samples: MetricSamples = {}
    for index, result in enumerate(results):
        values = extract_nemo_gym_metric_values(result)
        if message_logs is not None:
            trainable_lengths = [
                len(message.get("token_ids", []))
                for message in message_logs[index]
                if message.get("role") == "assistant"
                and message.get("token_ids") is not None
            ]
            values["trainable_items"] = float(len(trainable_lengths))
            values["generated_tokens"] = float(sum(trainable_lengths))
            values["trainable"] = float(bool(trainable_lengths))
        for name, value in values.items():
            samples.setdefault(name, []).append(value)
    return samples


def merge_metric_samples(*sample_sets: MetricSamples) -> MetricSamples:
    """Merge raw metric samples without mutating any input mapping."""
    merged: MetricSamples = {}
    for samples in sample_sets:
        for name, values in samples.items():
            merged.setdefault(name, []).extend(values)
    return merged


def summarize_nemo_gym_metric_samples(
    samples: MetricSamples,
    *,
    prefix: str = "rollout_metrics/nemo_gym",
) -> dict[str, Any]:
    """Build the mean/min/max/histogram metrics used by Megatron RL."""
    metrics: dict[str, Any] = {}
    for name, values in samples.items():
        if not values:
            continue
        metric_name = f"{prefix}/{name}"
        metrics[f"{metric_name}/mean"] = sum(values) / len(values)
        metrics[f"{metric_name}/min"] = min(values)
        metrics[f"{metric_name}/max"] = max(values)
        metrics[f"{metric_name}/histogram"] = Histogram(values)
    return metrics
