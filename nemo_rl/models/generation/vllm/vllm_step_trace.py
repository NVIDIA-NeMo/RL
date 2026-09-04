# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only, per-scheduler-step tracing for the pinned vLLM contract.

The adapter deliberately has no module-level vLLM imports.  Unit tests and
non-vLLM entry points can therefore import it without initializing vLLM.  The
custom ``AggregateStatLoggerBase`` subclass is created lazily in the worker.

``frontend_observed_cadence_ns`` is the interval between consecutive
``StatLogger.record`` callbacks in the AsyncLLM frontend.  It is useful for
ordering and diagnosing stalls, but it is *not* model execution latency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from array import array
from collections import Counter, deque
from functools import lru_cache
from typing import Any, Callable

VLLM_STEP_TRACE_SCHEMA_VERSION = 4
VLLM_STEP_TRACE_AGGREGATION_VERSION = 1
VLLM_STEP_TRACE_TIMELINE_COLUMN_ENCODING = "python-array-q-v1"
VLLM_STEP_TRACE_PHASE_CLASSIFIER = "vllm-v0.17.1-mfu-debug"
VLLM_STEP_TRACE_MODEL_STEP_DISCRIMINATOR = (
    "mfu_scheduled_work_then_cudagraph_required-v2"
)
VLLM_STEP_TRACE_DEFAULT_MAX_RECORDS = 100_000
VLLM_STEP_TRACE_MAX_RETAINED_ERRORS = 100

_CONTEXT_FIELDS = (
    "num_prefill_requests",
    "prefill_num_tokens",
    "prefill_context_len",
    "prefill_token_context_product",
    "num_decode_requests",
    "decode_num_tokens",
    "decode_context_len",
    "decode_token_context_product",
)

_TIMELINE_COLUMN_PATHS = {
    "recorded_at_monotonic_ns": ("frontend_recorded_at_monotonic_ns",),
    "scheduled_request_count": ("scheduled", "request_count"),
    "scheduled_token_count": ("scheduled", "token_count"),
    "decode_request_count": ("scheduled", "decode", "request_count"),
    "decode_token_count": ("scheduled", "decode", "token_count"),
    "decode_context_len_sum": ("scheduled", "decode", "context_len_sum"),
    "prefill_request_count": ("scheduled", "prefill", "request_count"),
    "prefill_token_count": ("scheduled", "prefill", "token_count"),
    "padding_token_count": ("cudagraph", "padding_tokens"),
    "estimated_flops_per_gpu": ("estimated_work_per_gpu", "flops"),
    "estimated_read_bytes_per_gpu": (
        "estimated_work_per_gpu",
        "read_bytes",
    ),
}


class VllmStepTraceContractError(RuntimeError):
    """The installed vLLM no longer satisfies the pinned trace contract."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _required_attr(value: Any, name: str, path: str) -> Any:
    if value is None or not hasattr(value, name):
        raise VllmStepTraceContractError(f"missing required field {path}.{name}")
    return getattr(value, name)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VllmStepTraceContractError(
            f"{path} must be an int, got {type(value).__name__}"
        )
    if value < 0:
        raise VllmStepTraceContractError(f"{path} must be nonnegative, got {value}")
    return value


def _nonnegative_number(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VllmStepTraceContractError(
            f"{path} must be numeric, got {type(value).__name__}"
        )
    if value < 0:
        raise VllmStepTraceContractError(f"{path} must be nonnegative, got {value}")
    return value


def _integer_breakdown(value: Any, path: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise VllmStepTraceContractError(
            f"{path} must be a dict, got {type(value).__name__}"
        )
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise VllmStepTraceContractError(
                f"{path} keys must be strings, got {type(key).__name__}"
            )
        result[key] = _nonnegative_int(item, f"{path}[{key!r}]")
    return result


def _optional_iteration_stats(iteration_stats: Any) -> dict[str, Any]:
    if iteration_stats is None:
        return {
            "present": False,
            "emitted_generation_tokens": None,
            "emitted_prompt_tokens": None,
            "computed_prompt_tokens": None,
            "preempted_requests": None,
            "finished_requests": None,
        }

    prompt_token_stats = _required_attr(
        iteration_stats,
        "prompt_token_stats",
        "iteration_stats",
    )
    finished_requests = _required_attr(
        iteration_stats,
        "finished_requests",
        "iteration_stats",
    )
    if not isinstance(finished_requests, list):
        raise VllmStepTraceContractError(
            "iteration_stats.finished_requests must be a list"
        )
    return {
        "present": True,
        "emitted_generation_tokens": _nonnegative_int(
            _required_attr(
                iteration_stats,
                "num_generation_tokens",
                "iteration_stats",
            ),
            "iteration_stats.num_generation_tokens",
        ),
        "emitted_prompt_tokens": _nonnegative_int(
            _required_attr(prompt_token_stats, "total", "prompt_token_stats"),
            "iteration_stats.prompt_token_stats.total",
        ),
        "computed_prompt_tokens": _nonnegative_int(
            _required_attr(prompt_token_stats, "computed", "prompt_token_stats"),
            "iteration_stats.prompt_token_stats.computed",
        ),
        "preempted_requests": _nonnegative_int(
            _required_attr(
                iteration_stats,
                "num_preempted_reqs",
                "iteration_stats",
            ),
            "iteration_stats.num_preempted_reqs",
        ),
        "finished_requests": len(finished_requests),
    }


def _required_mfu_context(perf_stats: Any) -> tuple[Any, dict[str, int]]:
    """Return the pinned MFU debug object and its validated context fields."""
    if perf_stats is None:
        raise VllmStepTraceContractError(
            "scheduler_stats.perf_stats is None for a purported model step"
        )
    debug_stats = _required_attr(perf_stats, "debug_stats", "perf_stats")
    if debug_stats is None:
        raise VllmStepTraceContractError(
            "perf_stats.debug_stats is None; require "
            "enable_mfu_metrics=True and VLLM_DEBUG_MFU_METRICS=1"
        )

    context_breakdown = _required_attr(
        debug_stats,
        "context_breakdown",
        "debug_stats",
    )
    if not isinstance(context_breakdown, dict):
        raise VllmStepTraceContractError(
            "debug_stats.context_breakdown must be a dict"
        )
    missing_context_fields = [
        field for field in _CONTEXT_FIELDS if field not in context_breakdown
    ]
    if missing_context_fields:
        raise VllmStepTraceContractError(
            "debug_stats.context_breakdown is missing "
            f"{missing_context_fields}; observed={sorted(context_breakdown)}"
        )
    context = {
        field: _nonnegative_int(
            context_breakdown[field],
            f"debug_stats.context_breakdown[{field!r}]",
        )
        for field in _CONTEXT_FIELDS
    }

    debug_prefill_requests = _nonnegative_int(
        _required_attr(
            debug_stats,
            "num_prefill_requests",
            "debug_stats",
        ),
        "debug_stats.num_prefill_requests",
    )
    debug_decode_requests = _nonnegative_int(
        _required_attr(
            debug_stats,
            "num_decode_requests",
            "debug_stats",
        ),
        "debug_stats.num_decode_requests",
    )
    if debug_prefill_requests != context["num_prefill_requests"]:
        raise VllmStepTraceContractError(
            "prefill request count disagrees between DebugPerfStats and "
            "context_breakdown"
        )
    if debug_decode_requests != context["num_decode_requests"]:
        raise VllmStepTraceContractError(
            "decode request count disagrees between DebugPerfStats and "
            "context_breakdown"
        )
    return debug_stats, context


def _classify_zero_work_mfu_estimate(
    perf_stats: Any,
    debug_stats: Any,
) -> str:
    """Validate and classify vLLM's terminal zero-work MFU estimate.

    vLLM 0.17.1 recomputes MFU stats for its terminal cleanup callback.  The
    context, FLOPs, and writes are zero, but the estimator can still account
    for a static read-only weight footprint.  This is not a model execution:
    the callback has no scheduled request/token and no CUDAGraphStat.  Accept
    only that narrowly observed, internally balanced artifact.
    """
    totals = {
        "num_flops_per_gpu": _nonnegative_int(
            _required_attr(
                perf_stats,
                "num_flops_per_gpu",
                "perf_stats",
            ),
            "perf_stats.num_flops_per_gpu",
        ),
        "num_read_bytes_per_gpu": _nonnegative_int(
            _required_attr(
                perf_stats,
                "num_read_bytes_per_gpu",
                "perf_stats",
            ),
            "perf_stats.num_read_bytes_per_gpu",
        ),
        "num_write_bytes_per_gpu": _nonnegative_int(
            _required_attr(
                perf_stats,
                "num_write_bytes_per_gpu",
                "perf_stats",
            ),
            "perf_stats.num_write_bytes_per_gpu",
        ),
    }
    breakdowns = {
        "num_flops_per_gpu_breakdown": _integer_breakdown(
            _required_attr(
                debug_stats,
                "num_flops_per_gpu_breakdown",
                "debug_stats",
            ),
            "debug_stats.num_flops_per_gpu_breakdown",
        ),
        "num_read_bytes_per_gpu_breakdown": _integer_breakdown(
            _required_attr(
                debug_stats,
                "num_read_bytes_per_gpu_breakdown",
                "debug_stats",
            ),
            "debug_stats.num_read_bytes_per_gpu_breakdown",
        ),
        "num_write_bytes_per_gpu_breakdown": _integer_breakdown(
            _required_attr(
                debug_stats,
                "num_write_bytes_per_gpu_breakdown",
                "debug_stats",
            ),
            "debug_stats.num_write_bytes_per_gpu_breakdown",
        ),
    }
    breakdown_sums = {
        name: sum(values.values())
        for name, values in breakdowns.items()
    }
    expected_sums = {
        "num_flops_per_gpu_breakdown": totals["num_flops_per_gpu"],
        "num_read_bytes_per_gpu_breakdown": totals[
            "num_read_bytes_per_gpu"
        ],
        "num_write_bytes_per_gpu_breakdown": totals[
            "num_write_bytes_per_gpu"
        ],
    }
    if breakdown_sums != expected_sums:
        raise VllmStepTraceContractError(
            "zero-work MFU cleanup totals disagree with breakdowns: "
            f"totals={totals}, breakdown_sums="
            f"{breakdown_sums}"
        )
    if (
        totals["num_flops_per_gpu"] != 0
        or totals["num_write_bytes_per_gpu"] != 0
    ):
        raise VllmStepTraceContractError(
            "zero-work MFU cleanup has compute or write work: "
            f"totals={totals}, breakdown_sums={breakdown_sums}"
        )
    if totals["num_read_bytes_per_gpu"] == 0:
        return "finished_request_cleanup_zero_scheduled_work"
    return (
        "finished_request_cleanup_zero_scheduled_work_"
        "with_static_read_estimate"
    )


def build_vllm_step_trace_record(
    scheduler_stats: Any,
    iteration_stats: Any,
    *,
    engine_idx: int,
    step_index: int,
    recorded_at_monotonic_ns: int,
    trace_window_started_monotonic_ns: int,
    previous_recorded_at_monotonic_ns: int | None,
) -> dict[str, Any]:
    """Convert one vLLM scheduler callback into a JSON-serializable record."""
    perf_stats = _required_attr(scheduler_stats, "perf_stats", "scheduler_stats")
    debug_stats, context = _required_mfu_context(perf_stats)

    flops_breakdown = _integer_breakdown(
        _required_attr(
            debug_stats,
            "num_flops_per_gpu_breakdown",
            "debug_stats",
        ),
        "debug_stats.num_flops_per_gpu_breakdown",
    )
    read_bytes_breakdown = _integer_breakdown(
        _required_attr(
            debug_stats,
            "num_read_bytes_per_gpu_breakdown",
            "debug_stats",
        ),
        "debug_stats.num_read_bytes_per_gpu_breakdown",
    )
    write_bytes_breakdown = _integer_breakdown(
        _required_attr(
            debug_stats,
            "num_write_bytes_per_gpu_breakdown",
            "debug_stats",
        ),
        "debug_stats.num_write_bytes_per_gpu_breakdown",
    )
    num_flops_per_gpu = _nonnegative_int(
        _required_attr(perf_stats, "num_flops_per_gpu", "perf_stats"),
        "perf_stats.num_flops_per_gpu",
    )
    num_read_bytes_per_gpu = _nonnegative_int(
        _required_attr(perf_stats, "num_read_bytes_per_gpu", "perf_stats"),
        "perf_stats.num_read_bytes_per_gpu",
    )
    num_write_bytes_per_gpu = _nonnegative_int(
        _required_attr(perf_stats, "num_write_bytes_per_gpu", "perf_stats"),
        "perf_stats.num_write_bytes_per_gpu",
    )
    flops_breakdown_sum = sum(flops_breakdown.values())
    read_bytes_breakdown_sum = sum(read_bytes_breakdown.values())
    write_bytes_breakdown_sum = sum(write_bytes_breakdown.values())

    prefill_requests = context["num_prefill_requests"]
    decode_requests = context["num_decode_requests"]
    prefill_tokens = context["prefill_num_tokens"]
    decode_tokens = context["decode_num_tokens"]
    scheduled_requests = prefill_requests + decode_requests
    scheduled_tokens = prefill_tokens + decode_tokens
    token_context_product = (
        context["prefill_token_context_product"]
        + context["decode_token_context_product"]
    )

    cudagraph_stats = _required_attr(
        scheduler_stats,
        "cudagraph_stats",
        "scheduler_stats",
    )
    if cudagraph_stats is None:
        raise VllmStepTraceContractError(
            "scheduler_stats.cudagraph_stats is None; require "
            "cudagraph_metrics=True"
        )
    unpadded_tokens = _nonnegative_int(
        _required_attr(
            cudagraph_stats,
            "num_unpadded_tokens",
            "cudagraph_stats",
        ),
        "cudagraph_stats.num_unpadded_tokens",
    )
    padded_tokens = _nonnegative_int(
        _required_attr(
            cudagraph_stats,
            "num_padded_tokens",
            "cudagraph_stats",
        ),
        "cudagraph_stats.num_padded_tokens",
    )
    paddings = _nonnegative_int(
        _required_attr(cudagraph_stats, "num_paddings", "cudagraph_stats"),
        "cudagraph_stats.num_paddings",
    )
    runtime_mode = _required_attr(
        cudagraph_stats,
        "runtime_mode",
        "cudagraph_stats",
    )
    if not isinstance(runtime_mode, str):
        raise VllmStepTraceContractError(
            "cudagraph_stats.runtime_mode must be a string"
        )
    if recorded_at_monotonic_ns < trace_window_started_monotonic_ns:
        raise VllmStepTraceContractError(
            "record timestamp precedes trace-window start"
        )
    cadence_ns: int | None = None
    if previous_recorded_at_monotonic_ns is not None:
        cadence_ns = recorded_at_monotonic_ns - previous_recorded_at_monotonic_ns
        if cadence_ns < 0:
            raise VllmStepTraceContractError(
                "frontend StatLogger record timestamps are not monotonic"
            )

    return {
        "schema_version": VLLM_STEP_TRACE_SCHEMA_VERSION,
        "phase_classifier": VLLM_STEP_TRACE_PHASE_CLASSIFIER,
        "engine_idx": _nonnegative_int(engine_idx, "engine_idx"),
        "step_index": _nonnegative_int(step_index, "step_index"),
        "frontend_recorded_at_monotonic_ns": recorded_at_monotonic_ns,
        "since_trace_window_start_ns": (
            recorded_at_monotonic_ns - trace_window_started_monotonic_ns
        ),
        "frontend_observed_cadence_ns": cadence_ns,
        "post_step_scheduler_state": {
            "running_requests": _nonnegative_int(
                _required_attr(
                    scheduler_stats,
                    "num_running_reqs",
                    "scheduler_stats",
                ),
                "scheduler_stats.num_running_reqs",
            ),
            "waiting_requests": _nonnegative_int(
                _required_attr(
                    scheduler_stats,
                    "num_waiting_reqs",
                    "scheduler_stats",
                ),
                "scheduler_stats.num_waiting_reqs",
            ),
            "kv_cache_usage": _nonnegative_number(
                _required_attr(
                    scheduler_stats,
                    "kv_cache_usage",
                    "scheduler_stats",
                ),
                "scheduler_stats.kv_cache_usage",
            ),
        },
        "scheduled": {
            "request_count": scheduled_requests,
            "token_count": scheduled_tokens,
            "token_context_product": token_context_product,
            "prefill": {
                "request_count": prefill_requests,
                "token_count": prefill_tokens,
                "context_len_sum": context["prefill_context_len"],
                "mean_context_len": (
                    context["prefill_context_len"] / prefill_requests
                    if prefill_requests
                    else None
                ),
                "token_context_product": context[
                    "prefill_token_context_product"
                ],
            },
            "decode": {
                "request_count": decode_requests,
                "token_count": decode_tokens,
                "context_len_sum": context["decode_context_len"],
                "mean_context_len": (
                    context["decode_context_len"] / decode_requests
                    if decode_requests
                    else None
                ),
                "token_context_product": context[
                    "decode_token_context_product"
                ],
            },
        },
        "estimated_work_per_gpu": {
            "flops": num_flops_per_gpu,
            "read_bytes": num_read_bytes_per_gpu,
            "write_bytes": num_write_bytes_per_gpu,
            "flops_breakdown": flops_breakdown,
            "flops_breakdown_sum": flops_breakdown_sum,
            "flops_total_matches_breakdown": (
                num_flops_per_gpu == flops_breakdown_sum
            ),
            "read_bytes_breakdown": read_bytes_breakdown,
            "read_bytes_breakdown_sum": read_bytes_breakdown_sum,
            "read_bytes_total_matches_breakdown": (
                num_read_bytes_per_gpu == read_bytes_breakdown_sum
            ),
            "write_bytes_breakdown": write_bytes_breakdown,
            "write_bytes_breakdown_sum": write_bytes_breakdown_sum,
            "write_bytes_total_matches_breakdown": (
                num_write_bytes_per_gpu == write_bytes_breakdown_sum
            ),
            "mfu_stats_calculation_seconds": _nonnegative_number(
                _required_attr(debug_stats, "calc_duration", "debug_stats"),
                "debug_stats.calc_duration",
            ),
        },
        "cudagraph": {
            "unpadded_tokens": unpadded_tokens,
            "padded_tokens": padded_tokens,
            "padding_tokens": paddings,
            "runtime_mode": runtime_mode,
            "unpadded_matches_mfu_scheduled_tokens": (
                unpadded_tokens == scheduled_tokens
            ),
            "padded_equals_unpadded_plus_padding": (
                padded_tokens == unpadded_tokens + paddings
            ),
        },
        "frontend_output": _optional_iteration_stats(iteration_stats),
    }


def _empty_engine_aggregate(engine_idx: int) -> dict[str, Any]:
    return {
        "engine_idx": int(engine_idx),
        "model_step_count": 0,
        "step_indexes_contiguous": True,
        "timestamps_are_internally_consistent": True,
        "record_schema_match_count": 0,
        "phase_classifier_match_count": 0,
        "first_records": [],
        "last_records": deque(maxlen=3),
        "timeline_columns": {
            name: array("q") for name in _TIMELINE_COLUMN_PATHS
        },
        # Runtime mode is a string in vLLM's CUDAGraphStat and therefore
        # cannot share the signed-int timeline encoding above.  Retain one
        # value per model step so an exact selected range can prove whether
        # CUDA graphs actually ran instead of merely being allowed.
        "runtime_mode_timeline": [],
        "distributions": {
            "scheduled_request_count": Counter(),
            "scheduled_token_count": Counter(),
            "decode_request_count": Counter(),
            "prefill_request_count": Counter(),
        },
        "scheduled": {
            "request_count_sum": 0,
            "token_count_sum": 0,
            "decode": {
                "request_count_sum": 0,
                "token_count_sum": 0,
                "context_len_sum": 0,
                "token_context_product_sum": 0,
            },
            "prefill": {
                "request_count_sum": 0,
                "token_count_sum": 0,
                "context_len_sum": 0,
                "token_context_product_sum": 0,
            },
        },
        "estimated_work_per_gpu": {
            "flops_sum": 0,
            "read_bytes_sum": 0,
            "write_bytes_sum": 0,
            "mfu_stats_calculation_seconds_sum": 0.0,
            "flops_breakdown_component_sums": Counter(),
            "read_bytes_breakdown_component_sums": Counter(),
            "write_bytes_breakdown_component_sums": Counter(),
            "flops_total_matches_breakdown_count": 0,
            "read_bytes_total_matches_breakdown_count": 0,
            "write_bytes_total_matches_breakdown_count": 0,
        },
        "cudagraph": {
            "unpadded_token_count_sum": 0,
            "padded_token_count_sum": 0,
            "padding_token_count_sum": 0,
            "unpadded_matches_mfu_scheduled_tokens_count": 0,
            "padded_equals_unpadded_plus_padding_count": 0,
            "runtime_mode_model_step_counts": Counter(),
        },
        "frontend_output": {
            "present_model_step_count": 0,
            "emitted_generation_tokens": 0,
            "emitted_prompt_tokens": 0,
            "computed_prompt_tokens": 0,
            "preempted_requests": 0,
            "finished_requests": 0,
        },
    }


def _nested_int(record: dict[str, Any], path: tuple[str, ...]) -> int:
    value: Any = record
    for key in path:
        value = value[key]
    return int(value)


def _update_component_sums(
    target: Counter[str], values: dict[str, int]
) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def _aggregate_record(
    aggregate: dict[str, Any],
    record: dict[str, Any],
    *,
    trace_window_started_monotonic_ns: int,
) -> None:
    step_count = int(aggregate["model_step_count"])
    step_index = int(record["step_index"])
    aggregate["step_indexes_contiguous"] = bool(
        aggregate["step_indexes_contiguous"] and step_index == step_count
    )
    aggregate["record_schema_match_count"] += int(
        record.get("schema_version") == VLLM_STEP_TRACE_SCHEMA_VERSION
    )
    aggregate["phase_classifier_match_count"] += int(
        record.get("phase_classifier") == VLLM_STEP_TRACE_PHASE_CLASSIFIER
    )

    timestamp_ns = int(record["frontend_recorded_at_monotonic_ns"])
    timestamp_column = aggregate["timeline_columns"][
        "recorded_at_monotonic_ns"
    ]
    previous_timestamp_ns = (
        int(timestamp_column[-1]) if timestamp_column else None
    )
    cadence_ns = record.get("frontend_observed_cadence_ns")
    timestamp_contract_valid = (
        int(record["since_trace_window_start_ns"])
        == timestamp_ns - trace_window_started_monotonic_ns
        and (
            previous_timestamp_ns is None
            and cadence_ns is None
            or previous_timestamp_ns is not None
            and cadence_ns is not None
            and timestamp_ns > previous_timestamp_ns
            and int(cadence_ns) == timestamp_ns - previous_timestamp_ns
        )
    )
    aggregate["timestamps_are_internally_consistent"] = bool(
        aggregate["timestamps_are_internally_consistent"]
        and timestamp_contract_valid
    )

    for name, path in _TIMELINE_COLUMN_PATHS.items():
        aggregate["timeline_columns"][name].append(_nested_int(record, path))

    if step_count < 3:
        aggregate["first_records"].append(record)
    aggregate["last_records"].append(record)

    scheduled = record["scheduled"]
    decode = scheduled["decode"]
    prefill = scheduled["prefill"]
    distributions = aggregate["distributions"]
    distributions["scheduled_request_count"].update(
        [int(scheduled["request_count"])]
    )
    distributions["scheduled_token_count"].update(
        [int(scheduled["token_count"])]
    )
    distributions["decode_request_count"].update(
        [int(decode["request_count"])]
    )
    distributions["prefill_request_count"].update(
        [int(prefill["request_count"])]
    )

    aggregate_scheduled = aggregate["scheduled"]
    aggregate_scheduled["request_count_sum"] += int(
        scheduled["request_count"]
    )
    aggregate_scheduled["token_count_sum"] += int(scheduled["token_count"])
    for phase, values in (("decode", decode), ("prefill", prefill)):
        phase_aggregate = aggregate_scheduled[phase]
        phase_aggregate["request_count_sum"] += int(values["request_count"])
        phase_aggregate["token_count_sum"] += int(values["token_count"])
        phase_aggregate["context_len_sum"] += int(values["context_len_sum"])
        phase_aggregate["token_context_product_sum"] += int(
            values["token_context_product"]
        )

    estimated = record["estimated_work_per_gpu"]
    aggregate_estimated = aggregate["estimated_work_per_gpu"]
    aggregate_estimated["flops_sum"] += int(estimated["flops"])
    aggregate_estimated["read_bytes_sum"] += int(estimated["read_bytes"])
    aggregate_estimated["write_bytes_sum"] += int(estimated["write_bytes"])
    aggregate_estimated["mfu_stats_calculation_seconds_sum"] += float(
        estimated["mfu_stats_calculation_seconds"]
    )
    for field in ("flops", "read_bytes", "write_bytes"):
        _update_component_sums(
            aggregate_estimated[f"{field}_breakdown_component_sums"],
            estimated[f"{field}_breakdown"],
        )
        aggregate_estimated[
            f"{field}_total_matches_breakdown_count"
        ] += int(estimated[f"{field}_total_matches_breakdown"])

    cudagraph = record["cudagraph"]
    aggregate_cudagraph = aggregate["cudagraph"]
    aggregate_cudagraph["unpadded_token_count_sum"] += int(
        cudagraph["unpadded_tokens"]
    )
    aggregate_cudagraph["padded_token_count_sum"] += int(
        cudagraph["padded_tokens"]
    )
    aggregate_cudagraph["padding_token_count_sum"] += int(
        cudagraph["padding_tokens"]
    )
    aggregate_cudagraph[
        "unpadded_matches_mfu_scheduled_tokens_count"
    ] += int(cudagraph["unpadded_matches_mfu_scheduled_tokens"])
    aggregate_cudagraph[
        "padded_equals_unpadded_plus_padding_count"
    ] += int(cudagraph["padded_equals_unpadded_plus_padding"])
    aggregate_cudagraph["runtime_mode_model_step_counts"].update(
        [str(cudagraph["runtime_mode"])]
    )
    aggregate["runtime_mode_timeline"].append(
        str(cudagraph["runtime_mode"])
    )

    frontend_output = record["frontend_output"]
    if frontend_output["present"]:
        aggregate_frontend = aggregate["frontend_output"]
        aggregate_frontend["present_model_step_count"] += 1
        for field in (
            "emitted_generation_tokens",
            "emitted_prompt_tokens",
            "computed_prompt_tokens",
            "preempted_requests",
            "finished_requests",
        ):
            aggregate_frontend[field] += int(frontend_output[field])

    aggregate["model_step_count"] = step_count + 1


def _snapshot_engine_aggregate(
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    step_count = int(aggregate["model_step_count"])
    return {
        "engine_idx": int(aggregate["engine_idx"]),
        "model_step_count": step_count,
        "first_step_index": 0 if step_count else None,
        "last_step_index": step_count - 1 if step_count else None,
        "step_indexes_contiguous": bool(
            aggregate["step_indexes_contiguous"]
        ),
        "timestamps_are_internally_consistent": bool(
            aggregate["timestamps_are_internally_consistent"]
        ),
        "record_schema_match_count": int(
            aggregate["record_schema_match_count"]
        ),
        "phase_classifier_match_count": int(
            aggregate["phase_classifier_match_count"]
        ),
        "first_records": copy.deepcopy(aggregate["first_records"]),
        "last_records": copy.deepcopy(list(aggregate["last_records"])),
        "timeline_columns": {
            name: copy.copy(values)
            for name, values in aggregate["timeline_columns"].items()
        },
        "runtime_mode_timeline": list(
            aggregate["runtime_mode_timeline"]
        ),
        "distributions": {
            name: dict(values)
            for name, values in aggregate["distributions"].items()
        },
        "scheduled": copy.deepcopy(aggregate["scheduled"]),
        "estimated_work_per_gpu": {
            key: (
                dict(value) if isinstance(value, Counter) else value
            )
            for key, value in aggregate["estimated_work_per_gpu"].items()
        },
        "cudagraph": {
            key: (
                dict(value) if isinstance(value, Counter) else value
            )
            for key, value in aggregate["cudagraph"].items()
        },
        "frontend_output": copy.deepcopy(aggregate["frontend_output"]),
    }


class VllmStepTraceBuffer:
    """Thread-safe online aggregate used by the vLLM custom stat logger."""

    def __init__(
        self,
        engine_indexes: list[int],
        *,
        max_records_per_engine: int = VLLM_STEP_TRACE_DEFAULT_MAX_RECORDS,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if max_records_per_engine <= 0:
            raise ValueError("max_records_per_engine must be positive")
        self._engine_indexes = [int(engine_idx) for engine_idx in engine_indexes]
        self._max_records_per_engine = int(max_records_per_engine)
        self._monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        self._clear_locked(self._monotonic_ns())

    def _clear_locked(self, started_at_ns: int) -> None:
        self._window_started_at_ns = started_at_ns
        self._aggregates: dict[int, dict[str, Any]] = {
            engine_idx: _empty_engine_aggregate(engine_idx)
            for engine_idx in self._engine_indexes
        }
        self._errors: list[dict[str, Any]] = []
        self._error_count = 0
        self._dropped_error_count = 0
        self._ignored_non_step_callbacks: dict[int, int] = {
            engine_idx: 0 for engine_idx in self._engine_indexes
        }
        self._ignored_non_step_callbacks_by_reason: dict[
            int, Counter[str]
        ] = {
            engine_idx: Counter() for engine_idx in self._engine_indexes
        }
        self._ignored_non_step_callbacks_with_frontend_token_activity: dict[
            int, int
        ] = {engine_idx: 0 for engine_idx in self._engine_indexes}
        self._ignored_non_step_callbacks_with_unverifiable_frontend_output: (
            dict[int, int]
        ) = {engine_idx: 0 for engine_idx in self._engine_indexes}
        self._dropped_records: dict[int, int] = {
            engine_idx: 0 for engine_idx in self._engine_indexes
        }
        self._previous_recorded_at_ns: dict[int, int | None] = {
            engine_idx: None for engine_idx in self._engine_indexes
        }
        self._next_step_index: dict[int, int] = {
            engine_idx: 0 for engine_idx in self._engine_indexes
        }

    def clear(self) -> None:
        with self._lock:
            self._clear_locked(self._monotonic_ns())

    def _ensure_engine(self, engine_idx: int) -> None:
        if engine_idx in self._aggregates:
            return
        self._aggregates[engine_idx] = _empty_engine_aggregate(engine_idx)
        self._ignored_non_step_callbacks[engine_idx] = 0
        self._ignored_non_step_callbacks_by_reason[engine_idx] = Counter()
        self._ignored_non_step_callbacks_with_frontend_token_activity[
            engine_idx
        ] = 0
        self._ignored_non_step_callbacks_with_unverifiable_frontend_output[
            engine_idx
        ] = 0
        self._dropped_records[engine_idx] = 0
        self._previous_recorded_at_ns[engine_idx] = None
        self._next_step_index[engine_idx] = 0

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        *,
        engine_idx: int,
    ) -> None:
        """Record a model step without allowing instrumentation to kill vLLM."""
        recorded_at_ns = self._monotonic_ns()
        engine_idx = int(engine_idx)
        with self._lock:
            self._ensure_engine(engine_idx)
            step_index = self._next_step_index[engine_idx]
            try:
                # vLLM emits stats-only control callbacks with no PerfStats.
                # It also emits one finished-request cleanup callback after the
                # final real step.  That cleanup gets a newly computed, zero-
                # work MFU context but no GPU-runner CUDAGraphStat.  Classify it
                # only from the exact scheduled-work context; positive work
                # without CUDAGraphStats remains a contract error.
                ignored_reason: str | None = None
                if scheduler_stats is None:
                    ignored_reason = "scheduler_stats_missing"
                elif getattr(scheduler_stats, "perf_stats", None) is None:
                    ignored_reason = "perf_stats_missing"
                else:
                    debug_stats, context = _required_mfu_context(
                        scheduler_stats.perf_stats
                    )
                    scheduled_requests = (
                        context["num_prefill_requests"]
                        + context["num_decode_requests"]
                    )
                    scheduled_tokens = (
                        context["prefill_num_tokens"]
                        + context["decode_num_tokens"]
                    )
                    if (scheduled_requests == 0) != (scheduled_tokens == 0):
                        raise VllmStepTraceContractError(
                            "MFU context has inconsistent zero scheduled "
                            f"work: requests={scheduled_requests}, "
                            f"tokens={scheduled_tokens}"
                        )
                    if scheduled_requests == 0:
                        if any(context.values()):
                            raise VllmStepTraceContractError(
                                "zero-work MFU cleanup context must have all "
                                "context_breakdown fields equal to zero"
                            )
                        cudagraph_stats = _required_attr(
                            scheduler_stats,
                            "cudagraph_stats",
                            "scheduler_stats",
                        )
                        if cudagraph_stats is not None:
                            raise VllmStepTraceContractError(
                                "zero-work MFU cleanup unexpectedly has "
                                "cudagraph_stats"
                            )
                        running_requests = _nonnegative_int(
                            _required_attr(
                                scheduler_stats,
                                "num_running_reqs",
                                "scheduler_stats",
                            ),
                            "scheduler_stats.num_running_reqs",
                        )
                        waiting_requests = _nonnegative_int(
                            _required_attr(
                                scheduler_stats,
                                "num_waiting_reqs",
                                "scheduler_stats",
                            ),
                            "scheduler_stats.num_waiting_reqs",
                        )
                        if running_requests or waiting_requests:
                            raise VllmStepTraceContractError(
                                "zero-work MFU cleanup has nonempty scheduler "
                                f"state: running={running_requests}, "
                                f"waiting={waiting_requests}"
                            )
                        ignored_reason = _classify_zero_work_mfu_estimate(
                            scheduler_stats.perf_stats,
                            debug_stats,
                        )

                if ignored_reason is not None:
                    self._ignored_non_step_callbacks[engine_idx] += 1
                    self._ignored_non_step_callbacks_by_reason[engine_idx][
                        ignored_reason
                    ] += 1
                    try:
                        frontend_output = _optional_iteration_stats(
                            iteration_stats
                        )
                    except Exception:
                        self._ignored_non_step_callbacks_with_unverifiable_frontend_output[
                            engine_idx
                        ] += 1
                    else:
                        if frontend_output["present"] and any(
                            int(frontend_output[field]) != 0
                            for field in (
                                "emitted_generation_tokens",
                                "emitted_prompt_tokens",
                                "computed_prompt_tokens",
                                "preempted_requests",
                            )
                        ):
                            self._ignored_non_step_callbacks_with_frontend_token_activity[
                                engine_idx
                            ] += 1
                    return

                record = build_vllm_step_trace_record(
                    scheduler_stats,
                    iteration_stats,
                    engine_idx=engine_idx,
                    step_index=step_index,
                    recorded_at_monotonic_ns=recorded_at_ns,
                    trace_window_started_monotonic_ns=self._window_started_at_ns,
                    previous_recorded_at_monotonic_ns=(
                        self._previous_recorded_at_ns[engine_idx]
                    ),
                )
                if (
                    int(self._aggregates[engine_idx]["model_step_count"])
                    >= self._max_records_per_engine
                ):
                    self._dropped_records[engine_idx] += 1
                else:
                    _aggregate_record(
                        self._aggregates[engine_idx],
                        record,
                        trace_window_started_monotonic_ns=(
                            self._window_started_at_ns
                        ),
                    )
            except Exception as error:
                self._error_count += 1
                if len(self._errors) < VLLM_STEP_TRACE_MAX_RETAINED_ERRORS:
                    self._errors.append(
                        {
                            "engine_idx": engine_idx,
                            "step_index": step_index,
                            "frontend_recorded_at_monotonic_ns": recorded_at_ns,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                else:
                    self._dropped_error_count += 1
                return

            self._next_step_index[engine_idx] += 1
            self._previous_recorded_at_ns[engine_idx] = recorded_at_ns

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": VLLM_STEP_TRACE_SCHEMA_VERSION,
                "capture_contract": {
                    "source": (
                        "SchedulerStats.perf_stats.debug_stats.context_breakdown"
                    ),
                    "model_step_discriminator": (
                        VLLM_STEP_TRACE_MODEL_STEP_DISCRIMINATOR
                    ),
                    "model_step_discriminator_semantics": (
                        "stats-only callbacks without PerfStats are control "
                        "events; a PerfStats callback is ignored as finished-"
                        "request cleanup only when every MFU context field is "
                        "zero, scheduler running/waiting counts are zero, and "
                        "CUDAGraphStat is absent. The terminal vLLM 0.17.1 "
                        "read-only static-weight MFU estimate is accepted only "
                        "when FLOPs/writes are zero and all totals equal their "
                        "breakdowns; every positive scheduled-work callback "
                        "requires CUDAGraphStat; ignored callbacks must report "
                        "no new frontend token or preemption activity"
                    ),
                    "phase_classifier": VLLM_STEP_TRACE_PHASE_CLASSIFIER,
                    "phase_classifier_rule": (
                        "new requests are prefill; cached requests with "
                        "num_scheduled_tokens > 1 are prefill; other cached "
                        "requests are decode"
                    ),
                    "scheduler_state_semantics": "post_step_queue_state",
                    "aggregation_version": (
                        VLLM_STEP_TRACE_AGGREGATION_VERSION
                    ),
                    "aggregation_semantics": (
                        "worker-side exact counters/distributions plus compact "
                        "per-step timeline columns and bounded first/last "
                        "record samples; full nested records are not retained"
                    ),
                    "timeline_column_encoding": (
                        VLLM_STEP_TRACE_TIMELINE_COLUMN_ENCODING
                    ),
                    "retained_record_samples_per_edge": 3,
                    "cadence_semantics": (
                        "consecutive AsyncLLM frontend StatLogger.record "
                        "callback interval; not model execution latency"
                    ),
                },
                "capture_hostname": os.uname().nodename,
                "trace_window_started_monotonic_ns": self._window_started_at_ns,
                "max_records_per_engine": self._max_records_per_engine,
                "aggregates_by_engine": {
                    engine_idx: _snapshot_engine_aggregate(aggregate)
                    for engine_idx, aggregate in self._aggregates.items()
                },
                "errors": copy.deepcopy(self._errors),
                "error_count": self._error_count,
                "dropped_error_count": self._dropped_error_count,
                "ignored_non_step_callbacks_by_engine": copy.deepcopy(
                    self._ignored_non_step_callbacks
                ),
                "ignored_non_step_callbacks_by_reason_by_engine": {
                    engine_idx: dict(sorted(reason_counts.items()))
                    for engine_idx, reason_counts in (
                        self._ignored_non_step_callbacks_by_reason.items()
                    )
                },
                "ignored_non_step_callbacks_with_frontend_token_activity_by_engine": (
                    copy.deepcopy(
                        self._ignored_non_step_callbacks_with_frontend_token_activity
                    )
                ),
                "ignored_non_step_callbacks_with_unverifiable_frontend_output_by_engine": (
                    copy.deepcopy(
                        self._ignored_non_step_callbacks_with_unverifiable_frontend_output
                    )
                ),
                "dropped_records_by_engine": copy.deepcopy(self._dropped_records),
            }


def summarize_exact_model_step_range(
    raw_trace_by_dp: dict[int, dict[str, Any]],
    *,
    start_step_index: int,
    stop_step_index: int,
) -> dict[str, Any]:
    """Select one exact, post-step-frontend-observed model-step range.

    Step indexes are the zero-based, contiguous indexes assigned by
    :class:`VllmStepTraceBuffer`.  The selected range is left-inclusive and
    right-exclusive.  Exact elapsed time is measured from the callback after
    ``start_step_index - 1`` through the callback after
    ``stop_step_index - 1``; this is why a start index of zero is rejected.

    The timestamps remain frontend observations and are not GPU timestamps.
    The selector is nevertheless exact about which model-step records and
    scheduler work enter the summary.
    """
    if (
        isinstance(start_step_index, bool)
        or not isinstance(start_step_index, int)
        or isinstance(stop_step_index, bool)
        or not isinstance(stop_step_index, int)
        or start_step_index < 1
        or stop_step_index <= start_step_index
    ):
        raise VllmStepTraceContractError(
            "exact model-step range must be positive and nonempty: "
            f"[{start_step_index!r}, {stop_step_index!r})"
        )
    if not isinstance(raw_trace_by_dp, dict) or not raw_trace_by_dp:
        raise VllmStepTraceContractError(
            "exact model-step selection requires at least one DP trace"
        )

    expected_columns = set(_TIMELINE_COLUMN_PATHS)
    summed_columns = tuple(
        name
        for name in _TIMELINE_COLUMN_PATHS
        if name != "recorded_at_monotonic_ns"
    )
    selected_step_count = stop_step_index - start_step_index
    per_engine: list[dict[str, Any]] = []

    for raw_dp_idx, snapshot in sorted(
        raw_trace_by_dp.items(), key=lambda item: int(item[0])
    ):
        dp_idx = int(raw_dp_idx)
        if not isinstance(snapshot, dict):
            raise VllmStepTraceContractError(
                f"DP {dp_idx} step trace is not a dictionary"
            )
        contract = snapshot.get("capture_contract")
        if (
            snapshot.get("schema_version") != VLLM_STEP_TRACE_SCHEMA_VERSION
            or not isinstance(contract, dict)
            or contract.get("aggregation_version")
            != VLLM_STEP_TRACE_AGGREGATION_VERSION
            or contract.get("timeline_column_encoding")
            != VLLM_STEP_TRACE_TIMELINE_COLUMN_ENCODING
        ):
            raise VllmStepTraceContractError(
                f"DP {dp_idx} step trace does not match the exact-range contract"
            )
        if (
            int(snapshot.get("error_count", -1)) != 0
            or int(snapshot.get("dropped_error_count", -1)) != 0
            or snapshot.get("errors") != []
        ):
            raise VllmStepTraceContractError(
                f"DP {dp_idx} step trace contains retained or dropped errors"
            )
        raw_drops = snapshot.get("dropped_records_by_engine")
        if (
            not isinstance(raw_drops, dict)
            or not raw_drops
            or any(int(value) != 0 for value in raw_drops.values())
        ):
            raise VllmStepTraceContractError(
                f"DP {dp_idx} step trace contains dropped model steps"
            )
        aggregates = snapshot.get("aggregates_by_engine")
        if not isinstance(aggregates, dict) or not aggregates:
            raise VllmStepTraceContractError(
                f"DP {dp_idx} step trace has no engine aggregates"
            )

        for raw_engine_idx, aggregate in sorted(
            aggregates.items(), key=lambda item: int(item[0])
        ):
            engine_idx = int(raw_engine_idx)
            if not isinstance(aggregate, dict):
                raise VllmStepTraceContractError(
                    f"DP {dp_idx}/engine {engine_idx} aggregate is invalid"
                )
            model_step_count = int(aggregate.get("model_step_count", -1))
            if (
                model_step_count < stop_step_index
                or aggregate.get("step_indexes_contiguous") is not True
                or int(aggregate.get("first_step_index", -1)) != 0
                or int(aggregate.get("last_step_index", -1))
                != model_step_count - 1
            ):
                raise VllmStepTraceContractError(
                    f"DP {dp_idx}/engine {engine_idx} does not cover contiguous "
                    f"steps [0, {stop_step_index})"
                )
            timeline = aggregate.get("timeline_columns")
            if not isinstance(timeline, dict) or set(timeline) != expected_columns:
                raise VllmStepTraceContractError(
                    f"DP {dp_idx}/engine {engine_idx} timeline columns differ"
                )
            columns = {
                name: [int(value) for value in values]
                for name, values in timeline.items()
            }
            if any(
                len(values) != model_step_count for values in columns.values()
            ):
                raise VllmStepTraceContractError(
                    f"DP {dp_idx}/engine {engine_idx} timeline is truncated"
                )
            runtime_mode_timeline = aggregate.get(
                "runtime_mode_timeline"
            )
            if (
                not isinstance(runtime_mode_timeline, list)
                or len(runtime_mode_timeline) != model_step_count
                or any(
                    not isinstance(mode, str) or not mode
                    for mode in runtime_mode_timeline
                )
            ):
                raise VllmStepTraceContractError(
                    f"DP {dp_idx}/engine {engine_idx} runtime-mode timeline "
                    "is truncated or invalid"
                )
            timestamps = columns["recorded_at_monotonic_ns"]
            if (
                aggregate.get("timestamps_are_internally_consistent") is not True
                or any(
                    right <= left
                    for left, right in zip(timestamps, timestamps[1:])
                )
            ):
                raise VllmStepTraceContractError(
                    f"DP {dp_idx}/engine {engine_idx} timestamps are invalid"
                )

            boundary_before_start_ns = timestamps[start_step_index - 1]
            boundary_after_stop_ns = timestamps[stop_step_index - 1]
            elapsed_ns = boundary_after_stop_ns - boundary_before_start_ns
            if elapsed_ns <= 0:
                raise VllmStepTraceContractError(
                    f"DP {dp_idx}/engine {engine_idx} selected duration is not "
                    "positive"
                )
            selected_columns = {
                name: values[start_step_index:stop_step_index]
                for name, values in columns.items()
                if name != "recorded_at_monotonic_ns"
            }
            selected_runtime_modes = runtime_mode_timeline[
                start_step_index:stop_step_index
            ]
            runtime_mode_counts = dict(
                sorted(Counter(selected_runtime_modes).items())
            )
            sums = {
                name: sum(selected_columns[name]) for name in summed_columns
            }
            decode_requests = sums["decode_request_count"]
            per_engine.append(
                {
                    "dp_idx": dp_idx,
                    "engine_idx": engine_idx,
                    "available_model_step_count": model_step_count,
                    "selected_model_step_count": selected_step_count,
                    "boundary_after_previous_step_monotonic_ns": (
                        boundary_before_start_ns
                    ),
                    "boundary_after_final_selected_step_monotonic_ns": (
                        boundary_after_stop_ns
                    ),
                    "frontend_observed_elapsed_ns": elapsed_ns,
                    "frontend_observed_model_steps_per_s": (
                        selected_step_count * 1_000_000_000 / elapsed_ns
                    ),
                    "minimum_scheduled_requests_per_step": min(
                        selected_columns["scheduled_request_count"]
                    ),
                    "maximum_prefill_requests_per_step": max(
                        selected_columns["prefill_request_count"]
                    ),
                    "mean_scheduled_requests_per_step": (
                        sums["scheduled_request_count"] / selected_step_count
                    ),
                    "mean_scheduled_tokens_per_step": (
                        sums["scheduled_token_count"] / selected_step_count
                    ),
                    "mean_decode_requests_per_step": (
                        decode_requests / selected_step_count
                    ),
                    "request_weighted_mean_decode_context_len": (
                        sums["decode_context_len_sum"] / decode_requests
                        if decode_requests
                        else None
                    ),
                    "prefill_requests_per_step": (
                        sums["prefill_request_count"] / selected_step_count
                    ),
                    "padding_tokens_per_step": (
                        sums["padding_token_count"] / selected_step_count
                    ),
                    "estimated_flops_per_gpu_per_step": (
                        sums["estimated_flops_per_gpu"] / selected_step_count
                    ),
                    "estimated_read_bytes_per_gpu_per_step": (
                        sums["estimated_read_bytes_per_gpu"]
                        / selected_step_count
                    ),
                    "aggregate_sums": sums,
                    "runtime_mode_model_step_counts": runtime_mode_counts,
                    "selected_runtime_mode_sha256": _canonical_sha256(
                        selected_runtime_modes
                    ),
                    "selected_work_columns_sha256": _canonical_sha256(
                        {
                            "start_step_index": start_step_index,
                            "stop_step_index": stop_step_index,
                            "columns": selected_columns,
                        }
                    ),
                }
            )

    return {
        "schema": "nrl-vllm-exact-model-step-range-v1",
        "step_index_semantics": (
            "zero-based contiguous worker-observed model-step index; "
            "left-inclusive and right-exclusive"
        ),
        "timing_semantics": (
            "elapsed frontend StatLogger callback time from immediately after "
            "step start-1 through immediately after step stop-1; exact record "
            "membership but not GPU execution latency"
        ),
        "start_step_index": start_step_index,
        "stop_step_index": stop_step_index,
        "selected_model_step_count_per_engine": selected_step_count,
        "engine_count": len(per_engine),
        "per_engine": per_engine,
        "all_selected_work_columns_sha256": _canonical_sha256(
            [
                {
                    "dp_idx": item["dp_idx"],
                    "engine_idx": item["engine_idx"],
                    "selected_work_columns_sha256": item[
                        "selected_work_columns_sha256"
                    ],
                }
                for item in per_engine
            ]
        ),
    }


@lru_cache(maxsize=1)
def get_vllm_step_trace_logger_class() -> type[Any]:
    """Create the custom logger only after the worker has imported vLLM."""
    from vllm.v1.metrics.loggers import AggregateStatLoggerBase

    class NrlVllmStepTraceStatLogger(AggregateStatLoggerBase):
        _nrl_vllm_step_trace_logger = True

        def __init__(self, vllm_config: Any, engine_indexes: list[int]) -> None:
            self.vllm_config = vllm_config
            raw_max_records = os.environ.get(
                "NRL_VLLM_STEP_TRACE_MAX_RECORDS",
                str(VLLM_STEP_TRACE_DEFAULT_MAX_RECORDS),
            )
            try:
                max_records = int(raw_max_records)
            except ValueError as error:
                raise ValueError(
                    "NRL_VLLM_STEP_TRACE_MAX_RECORDS must be an integer, "
                    f"got {raw_max_records!r}"
                ) from error
            self._buffer = VllmStepTraceBuffer(
                engine_indexes,
                max_records_per_engine=max_records,
            )

        def record(
            self,
            scheduler_stats: Any,
            iteration_stats: Any,
            mm_cache_stats: Any = None,
            engine_idx: int = 0,
        ) -> None:
            del mm_cache_stats
            self._buffer.record(
                scheduler_stats,
                iteration_stats,
                engine_idx=engine_idx,
            )

        def log_engine_initialized(self) -> None:
            pass

        def clear(self) -> None:
            self._buffer.clear()

        def snapshot(self) -> dict[str, Any]:
            return self._buffer.snapshot()

    NrlVllmStepTraceStatLogger.__name__ = "NrlVllmStepTraceStatLogger"
    NrlVllmStepTraceStatLogger.__qualname__ = "NrlVllmStepTraceStatLogger"
    return NrlVllmStepTraceStatLogger
