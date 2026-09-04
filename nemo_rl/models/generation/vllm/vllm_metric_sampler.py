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

"""Low-overhead, auditable sampling of the vLLM metrics used by NeMo RL."""

import math
from typing import Any


# ``vllm.v1.metrics.reader.get_metrics_snapshot`` walks every collector in the
# process-wide registry and materializes every vLLM histogram bucket.  The
# rollout profiler only consumes these five series, so sampling the full
# registry at 4 Hz adds avoidable, highly variable Python work.
_SAMPLE_NAME_TO_FIELD_AND_TYPE: dict[str, tuple[str, type[int] | type[float]]] = {
    "vllm:num_requests_running": ("inflight_batch_size", int),
    "vllm:num_requests_waiting": ("num_pending", int),
    "vllm:kv_cache_usage_perc": ("kv_cache_usage_perc", float),
    "vllm:generation_tokens_total": ("generation_tokens", int),
    "vllm:num_preemptions_total": ("num_preemptions", int),
}

VLLM_METRIC_SAMPLE_NAMES = frozenset(_SAMPLE_NAME_TO_FIELD_AND_TYPE)
VLLM_METRIC_FIELDS = frozenset(
    field for field, _value_type in _SAMPLE_NAME_TO_FIELD_AND_TYPE.values()
)


def read_restricted_vllm_metrics(
    registry: Any,
) -> tuple[dict[str, int | float], dict[str, dict[str, Any]]]:
    """Read exactly the five required vLLM Prometheus series.

    The return value contains JSON-compatible field values plus source
    metadata that identifies the exact Prometheus family, sample, type, and
    labels used for each field.

    No full-registry fallback is intentional.  A fallback to ``collect()``
    would silently restore the expensive behavior this sampler is designed to
    avoid and would make profiler overhead dependent on unrelated histograms.
    """

    restricted_registry = getattr(registry, "restricted_registry", None)
    if not callable(restricted_registry):
        raise RuntimeError(
            "The Prometheus registry does not support restricted_registry(); "
            "refusing to fall back to a full registry scan"
        )

    values: dict[str, int | float] = {}
    sources: dict[str, dict[str, Any]] = {}
    for metric_family in restricted_registry(VLLM_METRIC_SAMPLE_NAMES).collect():
        for prometheus_sample in metric_family.samples:
            specification = _SAMPLE_NAME_TO_FIELD_AND_TYPE.get(
                prometheus_sample.name
            )
            if specification is None:
                continue
            field, value_type = specification
            source = {
                "metric_family_name": str(metric_family.name),
                "metric_type": str(metric_family.type),
                "sample_name": str(prometheus_sample.name),
                "labels": dict(prometheus_sample.labels),
            }
            if field in values:
                raise RuntimeError(
                    "Multiple Prometheus series matched required vLLM metric "
                    f"field {field!r}: {sources[field]!r} and {source!r}"
                )
            values[field] = value_type(prometheus_sample.value)
            sources[field] = source

    missing_fields = sorted(VLLM_METRIC_FIELDS.difference(values))
    if missing_fields:
        expected_names = sorted(
            sample_name
            for sample_name, (field, _value_type) in (
                _SAMPLE_NAME_TO_FIELD_AND_TYPE.items()
            )
            if field in missing_fields
        )
        raise RuntimeError(
            "Required vLLM Prometheus series are missing: "
            f"fields={missing_fields}, sample_names={expected_names}"
        )
    return values, sources


def metric_capture_timing(
    *,
    interval_s: float,
    scheduled_at_monotonic_s: float | None,
    attempted_at_monotonic_s: float,
    started_at_monotonic_s: float,
    finished_at_monotonic_s: float,
) -> dict[str, int | float | None]:
    """Return diagnostics that separate wake, lock, and capture delays.

    ``attempted`` is recorded when the sampler thread starts a pass, before it
    acquires the shared metrics lock. ``started`` is recorded immediately after
    acquiring that lock and before reading Prometheus. ``finished`` is recorded
    immediately after the registry read.

    ``capture_missed_periods`` counts additional nominal sample deadlines that
    elapsed from this pass's scheduled deadline through capture completion.  A
    value of zero means the pass finished before the next deadline.
    """

    if interval_s <= 0:
        raise ValueError(f"interval_s must be positive, got {interval_s}")
    if attempted_at_monotonic_s > started_at_monotonic_s:
        raise ValueError("capture attempt must not follow capture start")
    if started_at_monotonic_s > finished_at_monotonic_s:
        raise ValueError("capture start must not follow capture finish")

    result: dict[str, int | float | None] = {
        "capture_scheduled_at_monotonic_s": scheduled_at_monotonic_s,
        "capture_attempted_at_monotonic_s": attempted_at_monotonic_s,
        "capture_started_at_monotonic_s": started_at_monotonic_s,
        "capture_finished_at_monotonic_s": finished_at_monotonic_s,
        "capture_duration_s": (
            finished_at_monotonic_s - started_at_monotonic_s
        ),
        "capture_lock_wait_s": (
            started_at_monotonic_s - attempted_at_monotonic_s
        ),
        "capture_wake_lateness_s": None,
        "capture_deadline_lateness_s": None,
        "capture_finish_lateness_s": None,
        "capture_missed_periods": 0,
    }
    if scheduled_at_monotonic_s is None:
        return result

    result["capture_wake_lateness_s"] = max(
        0.0, attempted_at_monotonic_s - scheduled_at_monotonic_s
    )
    result["capture_deadline_lateness_s"] = max(
        0.0, started_at_monotonic_s - scheduled_at_monotonic_s
    )
    finish_lateness_s = max(
        0.0, finished_at_monotonic_s - scheduled_at_monotonic_s
    )
    result["capture_finish_lateness_s"] = finish_lateness_s
    result["capture_missed_periods"] = max(
        0, math.floor(finish_lateness_s / interval_s)
    )
    return result
