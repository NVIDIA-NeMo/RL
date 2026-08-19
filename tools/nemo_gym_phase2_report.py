#!/usr/bin/env python3
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

"""Build an auditable NeMo Gym Router Phase 2 Prometheus report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen


TIMING_MARKER = "NEMO_GYM_CHAT_COMPLETION_TIMING_JSON="
RESULT_MARKER = "ROLLOUT_BENCHMARK_RESULT_JSON="
WARMUP_RESULT_MARKER = "PHASE2_WARMUP_RESULT_JSON="
CONSISTENT_HASH_ROUTE_RE = re.compile(
    r"Consistent hash routing: key='(?P<key>.*?)' -> worker='(?P<worker>.*?)'"
)
ROUTER_NATIVE_METRICS_AUDITED = frozenset(
    {
        "vllm_router_active_workers",
        "vllm_router_cache_hits_total",
        "vllm_router_cache_misses_total",
        "vllm_router_cb_outcomes_total",
        "vllm_router_cb_state",
        "vllm_router_cb_state_transitions_total",
        "vllm_router_load_balancing_events_total",
        "vllm_router_policy_decisions_total",
        "vllm_router_processed_requests_total",
        "vllm_router_request_errors_total",
        "vllm_router_requests_total",
        "vllm_router_retries_exhausted_total",
        "vllm_router_retries_total",
        "vllm_router_running_requests",
        "vllm_router_tree_size",
        "vllm_router_worker_health",
        "vllm_router_worker_load",
    }
)
ROUTER_NATIVE_ACTIVITY_METRICS = frozenset(
    {
        "vllm_router_cb_outcomes_total",
        "vllm_router_cb_state",
        "vllm_router_policy_decisions_total",
        "vllm_router_processed_requests_total",
        "vllm_router_requests_total",
    }
)


@dataclass(frozen=True)
class QueryDefinition:
    """One named PromQL query used by the Phase 2 report."""

    name: str
    promql: str


@dataclass(frozen=True)
class RangeQueryDefinition:
    """One named PromQL range query used for time-series evidence."""

    name: str
    promql: str


@dataclass(frozen=True)
class PrometheusSample:
    """One instant-vector sample returned by Prometheus."""

    labels: Mapping[str, str]
    timestamp: float
    value: float


class PrometheusQueryError(RuntimeError):
    """Raised when Prometheus rejects or cannot answer a report query."""


class PrometheusClient:
    """Small client for Prometheus's stable instant-query HTTP API."""

    def __init__(self, base_url: str, *, timeout_s: float) -> None:
        normalized = base_url.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"expected an http(s) Prometheus URL, got {base_url!r}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("credentials must not be embedded in Prometheus URL")
        self.base_url = normalized
        self.timeout_s = timeout_s

    def query(self, promql: str, *, evaluation_time: float) -> dict[str, Any]:
        """Execute one instant query and return its complete JSON response."""
        query_string = urlencode({"query": promql, "time": evaluation_time})
        url = f"{self.base_url}/api/v1/query?{query_string}"
        try:
            with urlopen(url, timeout=self.timeout_s) as response:
                status_code = response.status
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PrometheusQueryError(
                f"Prometheus returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise PrometheusQueryError(
                f"Prometheus query failed at {url}: {exc}"
            ) from exc

        if not 200 <= status_code < 300:
            raise PrometheusQueryError(f"Prometheus returned HTTP {status_code}")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PrometheusQueryError("Prometheus returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise PrometheusQueryError(f"Prometheus query failed: {payload!r}")
        return payload

    def query_range(
        self,
        promql: str,
        *,
        start_time: float,
        end_time: float,
        step_seconds: float,
    ) -> dict[str, Any]:
        """Execute one range query and return its complete JSON response."""
        query_string = urlencode(
            {
                "query": promql,
                "start": start_time,
                "end": end_time,
                "step": step_seconds,
            }
        )
        url = f"{self.base_url}/api/v1/query_range?{query_string}"
        try:
            with urlopen(url, timeout=self.timeout_s) as response:
                status_code = response.status
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PrometheusQueryError(
                f"Prometheus returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise PrometheusQueryError(
                f"Prometheus range query failed at {url}: {exc}"
            ) from exc

        if not 200 <= status_code < 300:
            raise PrometheusQueryError(f"Prometheus returned HTTP {status_code}")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PrometheusQueryError("Prometheus returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise PrometheusQueryError(f"Prometheus range query failed: {payload!r}")
        return payload


def _promql_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f'"{escaped}"'


def build_query_definitions(run_id: str, window_seconds: int) -> list[QueryDefinition]:
    """Return the complete, version-tolerant Phase 2 query set."""
    run = _promql_string(run_id)
    window = f"{window_seconds}s"
    backend = f'run_id={run},component="vllm_backend"'
    router = f'run_id={run},component="vllm_router"'

    definitions = [
        QueryDefinition(
            "target_up_min_by_target",
            f"min by (component, replica) (min_over_time(up{{run_id={run}}}[{window}]))",
        ),
        QueryDefinition(
            "router_metrics_adapter_info",
            f"max by (source, policy) (nemo_rl_vllm_router_metrics_adapter_info{{{router}}})",
        ),
        QueryDefinition(
            "router_native_metric_present_by_metric",
            f"max by (metric) (nemo_rl_vllm_router_native_metric_present{{{router}}})",
        ),
        QueryDefinition(
            "router_worker_health_source",
            f"max by (source) (nemo_rl_vllm_router_worker_health_source_info{{{router}}})",
        ),
        QueryDefinition(
            "router_cache_hits",
            f"sum(increase(vllm_router_cache_hits_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_cache_misses",
            f"sum(increase(vllm_router_cache_misses_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_cache_metrics_provenance",
            f"max by (source) (nemo_rl_vllm_router_cache_metrics_info{{{router}}})",
        ),
        QueryDefinition(
            "router_cache_threshold",
            f"max(nemo_rl_vllm_router_cache_threshold{{{router}}})",
        ),
        QueryDefinition(
            "router_cache_log_observations",
            "sum(increase(nemo_rl_vllm_router_cache_log_observations_total{"
            f"{router}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_prefix_cache_hits_by_replica",
            "sum by (replica) (increase("
            f'{{__name__=~"vllm:prefix_cache_hits(_total)?",{backend}}}[{window}]'
            "))",
        ),
        QueryDefinition(
            "backend_prefix_cache_queries_by_replica",
            "sum by (replica) (increase("
            f'{{__name__=~"vllm:prefix_cache_queries(_total)?",{backend}}}[{window}]'
            "))",
        ),
        QueryDefinition(
            "backend_request_success_by_replica",
            f"sum by (replica) (increase(vllm:request_success_total{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_prompt_tokens_by_replica",
            f"sum by (replica) (increase(vllm:prompt_tokens_total{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_generation_tokens_by_replica",
            f"sum by (replica) (increase(vllm:generation_tokens_total{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_prefix_cache_hits_resets_by_replica",
            "sum by (replica) (resets("
            f'{{__name__=~"vllm:prefix_cache_hits(_total)?",{backend}}}[{window}]'
            "))",
        ),
        QueryDefinition(
            "backend_prefix_cache_queries_resets_by_replica",
            "sum by (replica) (resets("
            f'{{__name__=~"vllm:prefix_cache_queries(_total)?",{backend}}}[{window}]'
            "))",
        ),
        QueryDefinition(
            "backend_request_success_resets_by_replica",
            f"sum by (replica) (resets(vllm:request_success_total{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_prompt_tokens_resets_by_replica",
            f"sum by (replica) (resets(vllm:prompt_tokens_total{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_generation_tokens_resets_by_replica",
            f"sum by (replica) (resets(vllm:generation_tokens_total{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_preemptions_by_replica",
            f"sum by (replica) (increase(vllm:num_preemptions_total{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_running_max_by_replica",
            f"max by (replica) (max_over_time(vllm:num_requests_running{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_running_mean_by_replica",
            f"avg by (replica) (avg_over_time(vllm:num_requests_running{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_running_p95_by_replica",
            "max by (replica) (quantile_over_time(0.95, "
            f"vllm:num_requests_running{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_running_request_seconds_by_replica",
            "sum by (replica) (avg_over_time("
            f"vllm:num_requests_running{{{backend}}}[{window}])) * {window_seconds}",
        ),
        QueryDefinition(
            "backend_waiting_max_by_replica",
            f"max by (replica) (max_over_time(vllm:num_requests_waiting{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_waiting_mean_by_replica",
            f"avg by (replica) (avg_over_time(vllm:num_requests_waiting{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_waiting_p95_by_replica",
            "max by (replica) (quantile_over_time(0.95, "
            f"vllm:num_requests_waiting{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "backend_waiting_request_seconds_by_replica",
            "sum by (replica) (avg_over_time("
            f"vllm:num_requests_waiting{{{backend}}}[{window}])) * {window_seconds}",
        ),
        QueryDefinition(
            "backend_kv_usage_max_by_replica",
            f"max by (replica) (max_over_time(vllm:kv_cache_usage_perc{{{backend}}}[{window}]))",
        ),
        QueryDefinition(
            "router_policy_decisions_by_worker",
            f"sum by (worker) (increase(nemo_rl_vllm_router_policy_decisions_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_processed_requests_by_worker",
            f"sum by (worker) (increase(nemo_rl_vllm_router_processed_requests_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_requests",
            f"sum(increase(nemo_rl_vllm_router_requests_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_request_errors",
            f"sum(increase(nemo_rl_vllm_router_request_errors_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_retries",
            f"sum(increase(nemo_rl_vllm_router_retries_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_retries_exhausted",
            f"sum(increase(nemo_rl_vllm_router_retries_exhausted_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_requests_resets",
            f"sum(resets(nemo_rl_vllm_router_requests_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_cache_hits_resets",
            f"sum(resets(vllm_router_cache_hits_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_cache_misses_resets",
            f"sum(resets(vllm_router_cache_misses_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_load_balancing_events",
            f"sum(increase(nemo_rl_vllm_router_load_balancing_events_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_cb_transitions",
            f"sum(increase(nemo_rl_vllm_router_cb_state_transitions_total{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_cb_transitions_by_worker",
            "sum by (worker) (increase(nemo_rl_vllm_router_cb_state_transitions_total{"
            f"{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_cb_failures",
            "sum(increase(nemo_rl_vllm_router_cb_outcomes_total{"
            f'{router},outcome="failure"}}[{window}]))',
        ),
        QueryDefinition(
            "router_cb_failures_by_worker",
            "sum by (worker) (increase(nemo_rl_vllm_router_cb_outcomes_total{"
            f'{router},outcome="failure"}}[{window}]))',
        ),
        QueryDefinition(
            "router_cb_successes",
            "sum(increase(nemo_rl_vllm_router_cb_outcomes_total{"
            f'{router},outcome="success"}}[{window}]))',
        ),
        QueryDefinition(
            "router_cb_successes_by_worker",
            "sum by (worker) (increase(nemo_rl_vllm_router_cb_outcomes_total{"
            f'{router},outcome="success"}}[{window}]))',
        ),
        QueryDefinition(
            "router_cb_state_max_by_worker",
            f"max by (worker) (max_over_time(nemo_rl_vllm_router_cb_state{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_active_workers_min",
            f"min(min_over_time(vllm_router_active_workers{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_worker_health_min_by_worker",
            f"min by (worker) (min_over_time(nemo_rl_vllm_router_worker_health{{{router}}}[{window}]))",
        ),
        QueryDefinition(
            "router_tree_size_max_by_worker",
            f"max by (worker) (max_over_time(vllm_router_tree_size{{{router}}}[{window}]))",
        ),
    ]
    for metric_key, metric_name in (
        ("backend_ttft", "vllm:time_to_first_token_seconds_bucket"),
        ("backend_itl", "vllm:inter_token_latency_seconds_bucket"),
        ("backend_e2e", "vllm:e2e_request_latency_seconds_bucket"),
        ("backend_queue", "vllm:request_queue_time_seconds_bucket"),
    ):
        for percentile_name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
            definitions.append(
                QueryDefinition(
                    f"{metric_key}_{percentile_name}",
                    f"histogram_quantile({quantile}, sum by (le) "
                    f"(increase({metric_name}{{{backend}}}[{window}])))",
                )
            )
    for percentile_name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        definitions.append(
            QueryDefinition(
                f"router_request_duration_{percentile_name}",
                f"histogram_quantile({quantile}, sum by (le) "
                f"(increase(vllm_router_request_duration_seconds_bucket{{{router}}}[{window}])))",
            )
        )
    return definitions


def build_range_query_definitions(run_id: str) -> list[RangeQueryDefinition]:
    """Return time-series queries used for drain and replica heatmaps."""
    run = _promql_string(run_id)
    backend = f'run_id={run},component="vllm_backend"'
    return [
        RangeQueryDefinition(
            "target_up_by_target",
            f"min by (component, replica) (up{{run_id={run}}})",
        ),
        RangeQueryDefinition(
            "target_scrape_age_seconds_by_target",
            f"max by (component, replica) (time() - timestamp(up{{run_id={run}}}))",
        ),
        RangeQueryDefinition(
            "backend_running_by_replica",
            f"sum by (replica) (vllm:num_requests_running{{{backend}}})",
        ),
        RangeQueryDefinition(
            "backend_waiting_by_replica",
            f"sum by (replica) (vllm:num_requests_waiting{{{backend}}})",
        ),
    ]


def collect_prometheus_queries(
    client: PrometheusClient,
    *,
    run_id: str,
    start_time: float,
    end_time: float,
    range_step_seconds: float,
) -> dict[str, Any]:
    """Collect and retain every raw query response used by the report."""
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if range_step_seconds <= 0:
        raise ValueError("range_step_seconds must be positive")
    window_seconds = max(1, math.ceil(end_time - start_time))
    queries: dict[str, Any] = {}
    for definition in build_query_definitions(run_id, window_seconds):
        queries[definition.name] = {
            "promql": definition.promql,
            "response": client.query(
                definition.promql,
                evaluation_time=end_time,
            ),
        }
    range_queries: dict[str, Any] = {}
    for definition in build_range_query_definitions(run_id):
        range_queries[definition.name] = {
            "promql": definition.promql,
            "response": client.query_range(
                definition.promql,
                start_time=start_time,
                end_time=end_time,
                step_seconds=range_step_seconds,
            ),
        }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "start_time": start_time,
        "end_time": end_time,
        "window_seconds": window_seconds,
        "range_step_seconds": range_step_seconds,
        "queries": queries,
        "range_queries": range_queries,
    }


def parse_samples(
    query_archive: Mapping[str, Any], query_name: str
) -> list[PrometheusSample]:
    """Parse one archived Prometheus vector/scalar response."""
    query_entry = query_archive["queries"].get(query_name)
    if not isinstance(query_entry, dict):
        raise ValueError(f"query archive is missing {query_name!r}")
    response = query_entry.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"query {query_name!r} has no response object")
    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"query {query_name!r} has no data object")
    result_type = data.get("resultType")
    result = data.get("result")
    if result_type == "scalar":
        result = [{"metric": {}, "value": result}]
    elif result_type != "vector":
        raise ValueError(
            f"query {query_name!r} returned unsupported result type {result_type!r}"
        )
    if not isinstance(result, list):
        raise ValueError(f"query {query_name!r} returned a non-list result")

    samples: list[PrometheusSample] = []
    for item in result:
        if not isinstance(item, dict):
            raise ValueError(f"query {query_name!r} returned an invalid sample")
        labels = item.get("metric")
        value = item.get("value")
        if (
            not isinstance(labels, dict)
            or not isinstance(value, list)
            or len(value) != 2
        ):
            raise ValueError(f"query {query_name!r} returned an invalid sample")
        numeric_value = float(value[1])
        if not math.isfinite(numeric_value):
            continue
        samples.append(
            PrometheusSample(
                labels={str(key): str(label) for key, label in labels.items()},
                timestamp=float(value[0]),
                value=numeric_value,
            )
        )
    return samples


def parse_range_samples(
    query_archive: Mapping[str, Any], query_name: str
) -> dict[tuple[tuple[str, str], ...], list[tuple[float, float]]]:
    """Parse one archived Prometheus matrix response by its complete label set."""
    range_queries = query_archive.get("range_queries")
    if not isinstance(range_queries, dict):
        raise ValueError("query archive has no range_queries object")
    query_entry = range_queries.get(query_name)
    if not isinstance(query_entry, dict):
        raise ValueError(f"range query archive is missing {query_name!r}")
    response = query_entry.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"range query {query_name!r} has no response object")
    data = response.get("data")
    if not isinstance(data, dict) or data.get("resultType") != "matrix":
        raise ValueError(f"range query {query_name!r} did not return a matrix")
    result = data.get("result")
    if not isinstance(result, list):
        raise ValueError(f"range query {query_name!r} returned a non-list result")

    series: dict[tuple[tuple[str, str], ...], list[tuple[float, float]]] = {}
    for item in result:
        if not isinstance(item, dict):
            raise ValueError(f"range query {query_name!r} returned an invalid series")
        labels = item.get("metric")
        values = item.get("values")
        if not isinstance(labels, dict) or not isinstance(values, list):
            raise ValueError(f"range query {query_name!r} returned an invalid series")
        key = tuple(sorted((str(name), str(value)) for name, value in labels.items()))
        if key in series:
            raise ValueError(f"range query {query_name!r} returned duplicate labels")
        parsed_values = []
        for value in values:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(
                    f"range query {query_name!r} returned an invalid sample"
                )
            numeric_value = float(value[1])
            if math.isfinite(numeric_value):
                parsed_values.append((float(value[0]), numeric_value))
        series[key] = parsed_values
    return series


def scalar_value(query_archive: Mapping[str, Any], query_name: str) -> float | None:
    """Return one scalar query value, or None when no finite series exists."""
    samples = parse_samples(query_archive, query_name)
    if not samples:
        return None
    if len(samples) != 1:
        raise ValueError(f"query {query_name!r} returned {len(samples)} samples")
    return samples[0].value


def values_by_label(
    query_archive: Mapping[str, Any],
    query_name: str,
    label_name: str,
) -> dict[str, float]:
    """Map one Prometheus label to values, rejecting ambiguous duplicates."""
    values: dict[str, float] = {}
    for sample in parse_samples(query_archive, query_name):
        label = sample.labels.get(label_name)
        if label is None:
            raise ValueError(f"query {query_name!r} sample is missing {label_name!r}")
        if label in values:
            raise ValueError(
                f"query {query_name!r} returned duplicate {label_name}={label!r}"
            )
        values[label] = sample.value
    return dict(sorted(values.items()))


def summarize_cache_metrics(query_archive: Mapping[str, Any]) -> dict[str, Any]:
    """Compute Router and backend cache hit rates from counter increases."""
    router_hits = scalar_value(query_archive, "router_cache_hits")
    router_misses = scalar_value(query_archive, "router_cache_misses")
    provenance_by_source = values_by_label(
        query_archive,
        "router_cache_metrics_provenance",
        "source",
    )
    active_sources = [
        source for source, value in provenance_by_source.items() if value == 1
    ]
    router_source = active_sources[0] if len(active_sources) == 1 else None
    router_threshold = scalar_value(query_archive, "router_cache_threshold")
    router_log_observations = scalar_value(
        query_archive, "router_cache_log_observations"
    )
    router_denominator = (
        router_hits + router_misses
        if router_hits is not None and router_misses is not None
        else None
    )
    router_rate = (
        router_hits / router_denominator
        if router_hits is not None
        and router_denominator is not None
        and router_denominator > 0
        else None
    )

    hits_by_replica = values_by_label(
        query_archive,
        "backend_prefix_cache_hits_by_replica",
        "replica",
    )
    queries_by_replica = values_by_label(
        query_archive,
        "backend_prefix_cache_queries_by_replica",
        "replica",
    )
    replicas = sorted(set(hits_by_replica) | set(queries_by_replica))
    per_replica = {}
    for replica in replicas:
        hits = hits_by_replica.get(replica)
        queries = queries_by_replica.get(replica)
        per_replica[replica] = {
            "hits": hits,
            "queries": queries,
            "hit_rate": (
                hits / queries
                if hits is not None and queries is not None and queries > 0
                else None
            ),
        }
    global_hits = sum(hits_by_replica.values()) if hits_by_replica else None
    global_queries = sum(queries_by_replica.values()) if queries_by_replica else None
    global_rate = (
        global_hits / global_queries
        if global_hits is not None and global_queries is not None and global_queries > 0
        else None
    )
    return {
        "router_routing_cache": {
            "available": router_rate is not None,
            "hits": router_hits,
            "misses": router_misses,
            "hit_rate": router_rate,
            "source": router_source,
            "source_samples": provenance_by_source,
            "cache_threshold": router_threshold,
            "debug_log_observations": router_log_observations,
        },
        "backend_prefix_cache": {
            "available": global_rate is not None,
            "hits": global_hits,
            "queries": global_queries,
            "hit_rate": global_rate,
            "per_replica": per_replica,
        },
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _decode_after_marker(line: str, marker: str) -> dict[str, Any]:
    payload = line.split(marker, 1)[1]
    value, _ = json.JSONDecoder().raw_decode(payload)
    if not isinstance(value, dict):
        raise TypeError(f"expected an object after {marker}")
    return value


def summarize_driver_log(path: Path) -> dict[str, Any]:
    """Read the final benchmark record and optional legacy timing markers."""
    timings: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    warmup_results: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if TIMING_MARKER in line:
                timings.append(_decode_after_marker(line, TIMING_MARKER))
            if RESULT_MARKER in line:
                results.append(_decode_after_marker(line, RESULT_MARKER))
            if WARMUP_RESULT_MARKER in line:
                warmup_results.append(_decode_after_marker(line, WARMUP_RESULT_MARKER))
    if len(results) != 1:
        raise ValueError(
            f"{path}: expected one final benchmark result, found {len(results)}"
        )
    if len(warmup_results) > 1:
        raise ValueError(
            f"{path}: expected at most one Phase 2 warmup result, "
            f"found {len(warmup_results)}"
        )
    successful = [
        float(timing["elapsed_s"]) for timing in timings if timing.get("status") == "ok"
    ]
    status_counts: dict[str, int] = {}
    for timing in timings:
        status = str(timing.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    request_timing = None
    if successful:
        request_timing = {
            "attempts": len(timings),
            "successful": len(successful),
            "status_counts": dict(sorted(status_counts.items())),
            "p50_s": percentile(successful, 0.50),
            "p90_s": percentile(successful, 0.90),
            "p95_s": percentile(successful, 0.95),
            "p99_s": percentile(successful, 0.99),
            "max_s": max(successful),
            "p99_over_p50": percentile(successful, 0.99) / percentile(successful, 0.50),
        }
    return {
        "benchmark_result": results[0],
        "warmup_result": warmup_results[0] if warmup_results else None,
        "legacy_request_timing": request_timing,
    }


def _duration_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "p50_s": None,
            "p90_s": None,
            "p95_s": None,
            "p99_s": None,
            "max_s": None,
            "p99_over_p50": None,
        }
    p50 = percentile(values, 0.50)
    p99 = percentile(values, 0.99)
    return {
        "samples": len(values),
        "p50_s": p50,
        "p90_s": percentile(values, 0.90),
        "p95_s": percentile(values, 0.95),
        "p99_s": p99,
        "max_s": max(values),
        "p99_over_p50": p99 / p50 if p50 > 0 else None,
    }


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _contains_context_limit_signal(*values: Any) -> bool:
    text = " ".join(json.dumps(value, sort_keys=True) for value in values if value)
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "context length",
            "max_model_len",
            "maximum context",
            "prompt exceeds",
        )
    )


def _is_blocking_observability_gap(code: str) -> bool:
    """Return whether a Gym gap prevents request-level Phase 2 auditing."""
    normalized = code.lower()
    return any(
        marker in normalized
        for marker in (
            "model_call",
            "trajectory_projection",
            "trajectory_identity",
            "producer_trajectory",
        )
    ) or normalized in {"invalid_gap_record", "ng_trajectory_missing"}


def _trajectory_termination(last_call: Mapping[str, Any] | None) -> str:
    """Classify the terminal model response without inventing missing evidence."""
    if last_call is None:
        return "unknown"
    finish_reason = str(last_call.get("finish_reason") or "").lower()
    response_status = str(last_call.get("response_status") or "").lower()
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return "truncated"
    if response_status in {"incomplete", "cancelled", "canceled"}:
        return "truncated"
    if finish_reason in {"stop", "end_turn", "completed"}:
        return "natural"
    if response_status in {"completed", "succeeded"}:
        return "natural"
    if last_call.get("successful") is False:
        return "error"
    return "unknown"


def summarize_eval_results(path: Path) -> dict[str, Any]:
    """Summarize correctness and Gym's correlated model-call observations."""
    rewards: list[float] = []
    coverage: set[tuple[int, int]] = set()
    generations_by_prompt: dict[int, set[int]] = {}
    expected_generations: set[int] = set()
    outcomes: list[dict[str, Any]] = []
    request_records: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []
    response_errors = 0
    context_limit_events = 0
    response_status_counts: Counter[str] = Counter()
    model_call_status_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    observability_gap_counts: Counter[str] = Counter()
    trajectories_with_observability = 0
    calls_seen = 0
    calls_with_complete_timing = 0
    calls_with_reported_cache = 0
    calls_with_completion_tokens = 0
    reported_prompt_tokens = 0
    reported_cached_tokens = 0

    with path.open(encoding="utf-8") as json_lines:
        for line_number, line in enumerate(json_lines, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            prompt_index = int(record["prompt_index"])
            generation_index = int(record["generation_index"])
            pair = (prompt_index, generation_index)
            reward = float(record["reward"])
            rewards.append(reward)
            coverage.add(pair)
            generations_by_prompt.setdefault(prompt_index, set()).add(generation_index)
            expected_generations.add(int(record["num_generations_per_prompt"]))

            full_result = record.get("full_result")
            if not isinstance(full_result, dict):
                raise TypeError(f"{path}:{line_number}: full_result must be an object")
            response = full_result.get("response")
            if not isinstance(response, dict):
                raise TypeError(
                    f"{path}:{line_number}: full_result.response must be an object"
                )
            response_error = response.get("error") is not None
            response_errors += response_error
            response_status_counts[str(response.get("status", "unknown"))] += 1
            if _contains_context_limit_signal(
                response.get("error"), response.get("incomplete_details")
            ):
                context_limit_events += 1
            outcomes.append(
                {
                    "prompt_index": prompt_index,
                    "generation_index": generation_index,
                    "reward": reward,
                    "response_error": response_error,
                }
            )

            trajectory = full_result.get("ng_trajectory")
            if not isinstance(trajectory, dict):
                observability_gap_counts["ng_trajectory_missing"] += 1
                termination_counts["unknown"] += 1
                outcomes[-1]["termination"] = "unknown"
                continue
            trajectories_with_observability += 1
            for gap in trajectory.get("gaps") or []:
                if isinstance(gap, dict):
                    observability_gap_counts[str(gap.get("code", "unknown"))] += 1
                else:
                    observability_gap_counts["invalid_gap_record"] += 1
            raw_calls = trajectory.get("model_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                observability_gap_counts["model_calls_unavailable"] += 1
                termination_counts["unknown"] += 1
                outcomes[-1]["termination"] = "unknown"
                continue

            rollout_id = str(
                trajectory.get("rollout_id")
                or f"prompt-{prompt_index}-generation-{generation_index}"
            )
            session_starts: list[float] = []
            session_ends: list[float] = []
            session_timing_complete = True
            trajectory_call_records: list[dict[str, Any]] = []
            for call_index, call in enumerate(raw_calls):
                calls_seen += 1
                if not isinstance(call, dict):
                    observability_gap_counts["invalid_model_call"] += 1
                    session_timing_complete = False
                    continue
                metadata = call.get("response_metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                token_stats = call.get("token_stats")
                token_stats = token_stats if isinstance(token_stats, dict) else {}
                started_at = _as_optional_float(call.get("started_at"))
                completed_at = _as_optional_float(call.get("completed_at"))
                duration_ms = _as_optional_float(call.get("duration_ms"))
                exact_timing = (
                    started_at is not None
                    and completed_at is not None
                    and completed_at >= started_at
                    and duration_ms is not None
                    and duration_ms >= 0
                )
                if exact_timing:
                    assert started_at is not None and completed_at is not None
                    calls_with_complete_timing += 1
                    session_starts.append(started_at)
                    session_ends.append(completed_at)
                else:
                    session_timing_complete = False

                prompt_tokens = token_stats.get("prompt_tokens")
                completion_tokens = token_stats.get("completion_tokens")
                cached_tokens = token_stats.get("cached_tokens")
                if completion_tokens is not None:
                    calls_with_completion_tokens += 1
                if prompt_tokens is not None and cached_tokens is not None:
                    calls_with_reported_cache += 1
                    reported_prompt_tokens += int(prompt_tokens)
                    reported_cached_tokens += int(cached_tokens)

                status_code = metadata.get("status_code")
                error_category = metadata.get("error_category")
                request_record = {
                    "prompt_index": prompt_index,
                    "generation_index": generation_index,
                    "rollout_id": rollout_id,
                    "call_index": call_index,
                    "model_call_id": call.get("model_call_id"),
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_s": duration_ms / 1000
                    if duration_ms is not None
                    else None,
                    "status_code": status_code,
                    "response_status": metadata.get("response_status"),
                    "finish_reason": metadata.get("finish_reason"),
                    "error_category": error_category,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "successful": (
                        (status_code is None or int(status_code) < 400)
                        and error_category is None
                    ),
                    "timing_source": (
                        "gym_model_call_capture_monotonic"
                        if duration_ms is not None
                        else "unavailable"
                    ),
                }
                request_records.append(request_record)
                trajectory_call_records.append(request_record)
                model_call_status_counts[
                    "success" if request_record["successful"] else "error"
                ] += 1

            termination = _trajectory_termination(
                trajectory_call_records[-1] if trajectory_call_records else None
            )
            termination_counts[termination] += 1
            outcomes[-1]["termination"] = termination

            if session_timing_complete and session_starts and session_ends:
                session_start = min(session_starts)
                session_end = max(session_ends)
                session_records.append(
                    {
                        "prompt_index": prompt_index,
                        "generation_index": generation_index,
                        "rollout_id": rollout_id,
                        "started_at": session_start,
                        "completed_at": session_end,
                        "duration_s": session_end - session_start,
                        "model_calls": len(raw_calls),
                    }
                )

    if not rewards:
        raise ValueError(f"{path}: no evaluation records")
    if len(coverage) != len(rewards):
        raise ValueError(f"{path}: duplicate (prompt_index, generation_index) records")
    if len(expected_generations) != 1:
        raise ValueError(f"{path}: inconsistent num_generations_per_prompt values")
    generations_per_prompt = next(iter(expected_generations))
    expected_generation_indices = set(range(generations_per_prompt))
    coverage_complete = all(
        generations == expected_generation_indices
        for generations in generations_by_prompt.values()
    )

    binary = all(reward in {0.0, 1.0} for reward in rewards)
    correct = int(sum(rewards)) if binary else None
    request_durations = [
        float(record["duration_s"])
        for record in request_records
        if record["duration_s"] is not None
    ]
    session_durations = [float(record["duration_s"]) for record in session_records]
    all_starts = [
        float(record["started_at"])
        for record in request_records
        if record["started_at"] is not None
    ]
    all_ends = [
        float(record["completed_at"])
        for record in request_records
        if record["completed_at"] is not None
    ]
    measurement_start = min(all_starts) if all_starts else None
    measurement_end = max(all_ends) if all_ends else None
    makespan = (
        measurement_end - measurement_start
        if measurement_start is not None and measurement_end is not None
        else None
    )
    session_timing = _duration_summary(session_durations)
    tail_bubble = (
        makespan - float(session_timing["p90_s"])
        if makespan is not None and session_timing["p90_s"] is not None
        else None
    )
    successful_calls = sum(bool(record["successful"]) for record in request_records)
    reported_output_tokens = sum(
        int(record["completion_tokens"])
        for record in request_records
        if record["completion_tokens"] is not None
    )
    blocking_gap_counts = {
        code: count
        for code, count in observability_gap_counts.items()
        if _is_blocking_observability_gap(code)
    }
    observability_complete = (
        trajectories_with_observability == len(rewards)
        and calls_seen > 0
        and calls_with_complete_timing == calls_seen
        and len(session_records) == len(rewards)
        and not blocking_gap_counts
    )
    return {
        "records": len(rewards),
        "coverage_pairs": len(coverage),
        "coverage_complete": coverage_complete,
        "num_prompts": len(generations_by_prompt),
        "num_generations_per_prompt": generations_per_prompt,
        "response_error_records": response_errors,
        "response_status_counts": dict(sorted(response_status_counts.items())),
        "context_limit_events": context_limit_events,
        "termination_counts": dict(sorted(termination_counts.items())),
        "natural_termination_rate": (
            termination_counts["natural"] / len(rewards) if rewards else None
        ),
        "truncation_rate": (
            termination_counts["truncated"] / len(rewards) if rewards else None
        ),
        "reward_is_binary": binary,
        "correct": correct,
        "accuracy": correct / len(rewards) if correct is not None else None,
        "mean_reward": statistics.fmean(rewards),
        "outcomes": outcomes,
        "request_records": request_records,
        "session_records": session_records,
        "request_timing": _duration_summary(request_durations),
        "session_timing": session_timing,
        "makespan_s": makespan,
        "tail_bubble_s": tail_bubble,
        "measurement_start_time": measurement_start,
        "measurement_end_time": measurement_end,
        "throughput": {
            "successful_model_calls_per_s": (
                successful_calls / makespan if makespan and makespan > 0 else None
            ),
            "output_tokens_per_s": (
                reported_output_tokens / makespan
                if makespan
                and makespan > 0
                and calls_with_completion_tokens == calls_seen
                else None
            ),
            "successful_trajectories_per_s": (
                (len(rewards) - response_errors) / makespan
                if makespan and makespan > 0
                else None
            ),
        },
        "model_call_observability": {
            "complete": observability_complete,
            "trajectories": trajectories_with_observability,
            "calls": calls_seen,
            "calls_with_complete_timing": calls_with_complete_timing,
            "gap_counts": dict(sorted(observability_gap_counts.items())),
            "blocking_gap_counts": dict(sorted(blocking_gap_counts.items())),
            "status_counts": dict(sorted(model_call_status_counts.items())),
        },
        "output_token_observability": {
            "complete": calls_seen > 0 and calls_with_completion_tokens == calls_seen,
            "calls_with_data": calls_with_completion_tokens,
            "calls_total": calls_seen,
            "reported_output_tokens": reported_output_tokens,
        },
        "api_reported_cached_tokens": {
            "available": calls_with_reported_cache > 0,
            "calls_with_data": calls_with_reported_cache,
            "calls_total": calls_seen,
            "cached_tokens": reported_cached_tokens,
            "prompt_tokens": reported_prompt_tokens,
            "cached_token_share": (
                reported_cached_tokens / reported_prompt_tokens
                if reported_prompt_tokens > 0
                else None
            ),
        },
    }


def summarize_router_logs(paths: Sequence[Path]) -> dict[str, Any]:
    """Extract session-affinity and non-exported queue evidence from Router logs."""
    existing_paths = [path for path in paths if path.exists()]
    mappings: dict[str, set[str]] = {}
    decisions_by_key: Counter[str] = Counter()
    queue_events: Counter[str] = Counter()
    error_signals: Counter[str] = Counter()
    for path in existing_paths:
        with path.open(encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                route = CONSISTENT_HASH_ROUTE_RE.search(line)
                if route is not None:
                    key = route.group("key")
                    mappings.setdefault(key, set()).add(route.group("worker"))
                    decisions_by_key[key] += 1
                lowered = line.lower()
                if "request queue is full" in lowered:
                    queue_events["queue_full"] += 1
                if "timed out in queue" in lowered:
                    queue_events["queue_timeout"] += 1
                if "failed to send request to worker" in lowered:
                    error_signals["worker_request_failure"] += 1
                if "unhealthy" in lowered and "worker" in lowered:
                    error_signals["unhealthy_worker"] += 1

    repeated_keys = {key: count for key, count in decisions_by_key.items() if count > 1}
    violations = {
        hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]: sorted(workers)
        for key, workers in mappings.items()
        if len(workers) > 1
    }
    return {
        "available": bool(existing_paths),
        "paths": [str(path) for path in paths],
        "missing_paths": [str(path) for path in paths if not path.exists()],
        "session_affinity": {
            "available": bool(mappings),
            "keys": len(mappings),
            "routing_decisions": sum(decisions_by_key.values()),
            "repeated_keys": len(repeated_keys),
            "repeated_routing_decisions": sum(repeated_keys.values()),
            "violations": violations,
            "passed": bool(repeated_keys) and not violations,
            "key_values_archived_only_in_raw_logs": True,
        },
        "router_queue": {
            "status": "not_exposed_by_router_version",
            "router_version": "0.1.15",
            "log_event_counts": dict(sorted(queue_events.items())),
        },
        "error_signal_counts": dict(sorted(error_signals.items())),
    }


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _scalar_metrics(
    query_archive: Mapping[str, Any], names: Sequence[str]
) -> dict[str, float | None]:
    return {name: scalar_value(query_archive, name) for name in names}


def _distribution_statistics(values_by_replica: Mapping[str, float]) -> dict[str, Any]:
    values = list(values_by_replica.values())
    if not values:
        return {
            "coefficient_of_variation": None,
            "largest_replica_share": None,
            "max_to_mean": None,
        }
    total = sum(values)
    mean = statistics.fmean(values)
    return {
        "coefficient_of_variation": _coefficient_of_variation(values),
        "largest_replica_share": max(values) / total if total > 0 else None,
        "max_to_mean": max(values) / mean if mean > 0 else None,
    }


def _complete_replica_metric(
    values: Mapping[str, float], expected_replicas: set[str]
) -> bool:
    return set(values) == expected_replicas


def _map_backend_replicas_to_router_workers(
    manifest_targets: Sequence[Mapping[str, Any]],
    router_workers: set[str],
) -> dict[str, str]:
    """Map backend replica labels to exact Router worker URL labels."""
    mapping: dict[str, str] = {}
    for target in manifest_targets:
        labels = target.get("labels")
        if not isinstance(labels, Mapping) or labels.get("component") != "vllm_backend":
            continue
        replica = str(labels.get("replica", ""))
        candidates: set[str] = set()
        metrics_url = target.get("metrics_url")
        if isinstance(metrics_url, str):
            parsed = urlsplit(metrics_url)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                candidates.add(f"{parsed.scheme}://{parsed.netloc}")
        address = target.get("address")
        if isinstance(address, str) and address:
            candidates.update({f"http://{address}", f"https://{address}"})
        matches = candidates & router_workers
        if replica and len(matches) == 1:
            mapping[replica] = next(iter(matches))
    return mapping


def _clip_step_series(
    values: Sequence[tuple[float, float]], *, start_time: float, end_time: float
) -> list[tuple[float, float]]:
    """Clip a sampled gauge as a left-continuous step series."""
    if end_time <= start_time:
        raise ValueError("step-series end_time must be greater than start_time")
    ordered = sorted((float(timestamp), float(value)) for timestamp, value in values)
    if not ordered or ordered[0][0] > start_time or ordered[-1][0] < end_time:
        return []
    current = ordered[0][1]
    for timestamp, value in ordered:
        if timestamp > start_time:
            break
        current = value
    clipped = [(start_time, current)]
    for timestamp, value in ordered:
        if start_time < timestamp < end_time:
            clipped.append((timestamp, value))
    clipped.append((end_time, clipped[-1][1]))
    return clipped


def _step_series_statistics(
    values: Sequence[tuple[float, float]], *, start_time: float, end_time: float
) -> dict[str, float] | None:
    clipped = _clip_step_series(values, start_time=start_time, end_time=end_time)
    if not clipped:
        return None
    intervals = [
        (value, next_timestamp - timestamp)
        for (timestamp, value), (next_timestamp, _) in zip(clipped, clipped[1:])
        if next_timestamp > timestamp
    ]
    duration = end_time - start_time
    request_seconds = sum(value * interval_s for value, interval_s in intervals)
    threshold = duration * 0.95
    accumulated = 0.0
    p95 = 0.0
    for value, interval_s in sorted(intervals):
        accumulated += interval_s
        p95 = value
        if accumulated >= threshold:
            break
    return {
        "mean": request_seconds / duration,
        "p95": p95,
        "max": max(value for value, _ in intervals),
        "request_seconds": request_seconds,
    }


def _concurrency_statistics_by_replica(
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    start_time: float,
    end_time: float,
) -> tuple[dict[str, dict[str, float]], bool]:
    statistics_by_replica = {
        replica: _step_series_statistics(
            values, start_time=start_time, end_time=end_time
        )
        for replica, values in series.items()
    }
    complete = bool(statistics_by_replica) and all(
        value is not None for value in statistics_by_replica.values()
    )
    return (
        {
            statistic: {
                replica: values[statistic]
                for replica, values in statistics_by_replica.items()
                if values is not None
            }
            for statistic in ("mean", "p95", "max", "request_seconds")
        },
        complete,
    )


def build_summary(
    *,
    manifest: Mapping[str, Any],
    query_archive: Mapping[str, Any],
    driver_summary: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
    router_log_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine Prometheus, request, and evaluation evidence into one summary."""
    run_id = str(manifest["run_id"])
    if query_archive.get("run_id") != run_id:
        raise ValueError("Prometheus query archive run_id does not match manifest")
    manifest_targets = manifest.get("targets")
    if not isinstance(manifest_targets, list) or not manifest_targets:
        raise ValueError("manifest contains no Prometheus targets")
    policies = {
        str(target["labels"]["routing_policy"])
        for target in manifest_targets
        if isinstance(target, dict)
    }
    if len(policies) != 1:
        raise ValueError("manifest targets do not have one routing_policy")
    routing_policy = next(iter(policies))

    query_start = float(query_archive["start_time"])
    query_end = float(query_archive["end_time"])
    if query_end <= query_start:
        raise ValueError("Prometheus query archive has an invalid measurement window")

    expected_targets: set[tuple[str, str]] = set()
    backend_replicas: set[str] = set()
    for target in manifest_targets:
        if not isinstance(target, dict) or not isinstance(target.get("labels"), dict):
            raise ValueError("manifest contains an invalid Prometheus target")
        component = str(target["labels"].get("component", ""))
        replica = str(target["labels"].get("replica", ""))
        if not component or not replica:
            raise ValueError("manifest target is missing component or replica")
        identity = (component, replica)
        if identity in expected_targets:
            raise ValueError(f"manifest contains duplicate target {identity!r}")
        expected_targets.add(identity)
        if component == "vllm_backend":
            backend_replicas.add(replica)
    if not backend_replicas:
        raise ValueError("manifest contains no vLLM backend targets")

    up_samples = parse_samples(query_archive, "target_up_min_by_target")
    target_up = [
        {
            "component": sample.labels.get("component"),
            "replica": sample.labels.get("replica"),
            "min_up": sample.value,
        }
        for sample in up_samples
    ]
    observed_targets = {
        (str(target["component"]), str(target["replica"])) for target in target_up
    }
    all_targets_up = observed_targets == expected_targets and all(
        sample.value >= 1.0 for sample in up_samples
    )
    cache = summarize_cache_metrics(query_archive)
    request_success_by_replica = values_by_label(
        query_archive, "backend_request_success_by_replica", "replica"
    )
    prompt_tokens_by_replica = values_by_label(
        query_archive, "backend_prompt_tokens_by_replica", "replica"
    )
    generation_by_replica = values_by_label(
        query_archive, "backend_generation_tokens_by_replica", "replica"
    )
    counter_resets_by_replica = {
        name: values_by_label(
            query_archive, f"backend_{name}_resets_by_replica", "replica"
        )
        for name in (
            "prefix_cache_hits",
            "prefix_cache_queries",
            "request_success",
            "prompt_tokens",
            "generation_tokens",
        )
    }
    backend_counter_reset_coverage_complete = all(
        _complete_replica_metric(values, backend_replicas)
        for values in counter_resets_by_replica.values()
    )
    backend_counter_resets_total = sum(
        value
        for values in counter_resets_by_replica.values()
        for value in values.values()
    )
    preemptions_by_replica = values_by_label(
        query_archive, "backend_preemptions_by_replica", "replica"
    )
    kv_usage_by_replica = values_by_label(
        query_archive, "backend_kv_usage_max_by_replica", "replica"
    )
    backend_latency = _scalar_metrics(
        query_archive,
        [
            f"backend_{metric}_{percentile}"
            for metric in ("ttft", "itl", "e2e", "queue")
            for percentile in ("p50", "p90", "p99")
        ],
    )
    registration = manifest.get("registration")
    registration_confirmed = (
        isinstance(registration, dict) and registration.get("status") == "registered"
    )
    all_ready = all(
        bool(target.get("ready_at_registration")) for target in manifest_targets
    )
    accuracy = eval_summary.get("accuracy")
    benchmark_result = driver_summary["benchmark_result"]
    benchmark_average = float(benchmark_result["average_score"])
    accuracy_matches_benchmark = (
        math.isclose(float(accuracy), benchmark_average, rel_tol=1e-9, abs_tol=1e-12)
        if accuracy is not None
        else math.isclose(
            float(eval_summary["mean_reward"]),
            benchmark_average,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    )
    benchmark_prompts = int(benchmark_result["num_samples"])
    expected_evaluation_records = benchmark_prompts * int(
        eval_summary["num_generations_per_prompt"]
    )
    evaluation_coverage_complete = (
        bool(eval_summary["coverage_complete"])
        and int(eval_summary["num_prompts"]) == benchmark_prompts
        and int(eval_summary["records"]) == expected_evaluation_records
    )
    capture_start = eval_summary.get("measurement_start_time")
    capture_end = eval_summary.get("measurement_end_time")
    measurement_contains_model_calls = (
        capture_start is not None
        and capture_end is not None
        and query_start <= float(capture_start) <= float(capture_end) <= query_end
    )
    model_call_observability = eval_summary["model_call_observability"]
    model_call_errors = int(model_call_observability["status_counts"].get("error", 0))
    router_metrics_required = routing_policy != "direct"
    router_logs = dict(router_log_summary or {})
    router_logs_available = bool(router_logs.get("available"))
    router_adapter_samples = parse_samples(query_archive, "router_metrics_adapter_info")
    router_adapter_complete = (
        len(router_adapter_samples) == 1
        and router_adapter_samples[0].value == 1
        and router_adapter_samples[0].labels.get("source") == "native_aggregate_compat"
        and router_adapter_samples[0].labels.get("policy") == routing_policy
    )
    router_native_metric_presence = values_by_label(
        query_archive, "router_native_metric_present_by_metric", "metric"
    )
    router_health_sources = values_by_label(
        query_archive, "router_worker_health_source", "source"
    )
    router_request_errors = scalar_value(query_archive, "router_request_errors")
    router_retries_exhausted = scalar_value(query_archive, "router_retries_exhausted")
    router_worker_health = values_by_label(
        query_archive, "router_worker_health_min_by_worker", "worker"
    )
    router_active_workers_min = scalar_value(query_archive, "router_active_workers_min")
    router_policy_decisions = values_by_label(
        query_archive, "router_policy_decisions_by_worker", "worker"
    )
    router_processed_requests = values_by_label(
        query_archive, "router_processed_requests_by_worker", "worker"
    )
    router_cb_state = values_by_label(
        query_archive, "router_cb_state_max_by_worker", "worker"
    )
    router_cb_transitions_by_worker = values_by_label(
        query_archive, "router_cb_transitions_by_worker", "worker"
    )
    router_cb_successes_by_worker = values_by_label(
        query_archive, "router_cb_successes_by_worker", "worker"
    )
    router_cb_failures_by_worker = values_by_label(
        query_archive, "router_cb_failures_by_worker", "worker"
    )
    router_health_source_verified = (
        len(router_health_sources) == 1
        and next(iter(router_health_sources))
        in {
            "adapter_backend_health_probe",
            "native_and_adapter_probe",
            "partial_native_and_adapter_probe",
        }
        and next(iter(router_health_sources.values())) == 1
    )
    router_health_observations_valid = (
        all(value >= 1.0 for value in router_worker_health.values())
        and router_health_source_verified
        and router_active_workers_min is not None
        and router_active_workers_min >= len(backend_replicas)
    )
    target_up_series = parse_range_samples(query_archive, "target_up_by_target")
    range_target_ids = {
        (dict(labels).get("component", ""), dict(labels).get("replica", ""))
        for labels in target_up_series
    }
    scrape_age_series = parse_range_samples(
        query_archive, "target_scrape_age_seconds_by_target"
    )
    scrape_age_target_ids = {
        (dict(labels).get("component", ""), dict(labels).get("replica", ""))
        for labels in scrape_age_series
    }
    monitoring_config = manifest.get("monitoring_config")
    monitoring_config = monitoring_config if isinstance(monitoring_config, dict) else {}
    scrape_interval_s = float(monitoring_config.get("scrape_interval_s", 10.0))
    max_allowed_scrape_age_s = scrape_interval_s * 2.5
    range_step_s = float(query_archive["range_step_seconds"])
    measurement_duration_s = (
        float(capture_end) - float(capture_start)
        if capture_start is not None and capture_end is not None
        else 0.0
    )
    measurement_resolution_sufficient = (
        measurement_duration_s >= 10 * scrape_interval_s
        and range_step_s <= scrape_interval_s
    )
    warmup_evidence = driver_summary.get("warmup_result")
    warmup_settle_seconds = (
        _as_optional_float(warmup_evidence.get("settle_seconds"))
        if isinstance(warmup_evidence, dict)
        else None
    )
    warmup_execution_evidence_present = (
        isinstance(warmup_evidence, dict)
        and warmup_evidence.get("status") == "completed"
        and warmup_evidence.get("source") == "measurement_workload_prefix"
        and isinstance(warmup_evidence.get("requests"), int)
        and not isinstance(warmup_evidence.get("requests"), bool)
        and int(warmup_evidence["requests"]) > 0
        and warmup_evidence.get("results") == warmup_evidence.get("requests")
        and warmup_evidence.get("model_call_capture_reset") is True
        and isinstance(warmup_evidence.get("workload_sha256"), str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", warmup_evidence["workload_sha256"])
        is not None
        and warmup_settle_seconds is not None
        and warmup_settle_seconds
        >= float(monitoring_config.get("initial_scrape_wait_s", 0)) + scrape_interval_s
    )
    scrape_freshness_complete = scrape_age_target_ids == expected_targets and all(
        values and all(0 <= value <= max_allowed_scrape_age_s for _, value in values)
        for values in scrape_age_series.values()
    )
    running_range = _series_replica(
        parse_range_samples(query_archive, "backend_running_by_replica")
    )
    waiting_range = _series_replica(
        parse_range_samples(query_archive, "backend_waiting_by_replica")
    )
    prometheus_range_evidence_complete = (
        range_target_ids == expected_targets
        and all(
            values and all(value >= 1.0 for _, value in values)
            for values in target_up_series.values()
        )
        and set(running_range) == backend_replicas
        and set(waiting_range) == backend_replicas
        and all(running_range.values())
        and all(waiting_range.values())
    )
    if capture_start is not None and capture_end is not None:
        running_by_replica, running_measurement_complete = (
            _concurrency_statistics_by_replica(
                running_range,
                start_time=float(capture_start),
                end_time=float(capture_end),
            )
        )
        waiting_by_replica, waiting_measurement_complete = (
            _concurrency_statistics_by_replica(
                waiting_range,
                start_time=float(capture_start),
                end_time=float(capture_end),
            )
        )
    else:
        running_by_replica = {
            name: {} for name in ("mean", "p95", "max", "request_seconds")
        }
        waiting_by_replica = {
            name: {} for name in ("mean", "p95", "max", "request_seconds")
        }
        running_measurement_complete = False
        waiting_measurement_complete = False
    concurrency_measurement_complete = (
        running_measurement_complete
        and waiting_measurement_complete
        and all(
            _complete_replica_metric(values, backend_replicas)
            for values in (*running_by_replica.values(), *waiting_by_replica.values())
        )
    )
    backend_replica_metrics = {
        "request_success": request_success_by_replica,
        "prompt_tokens": prompt_tokens_by_replica,
        "generation_tokens": generation_by_replica,
        "preemptions": preemptions_by_replica,
        "kv_usage_max": kv_usage_by_replica,
        **{f"running_{name}": values for name, values in running_by_replica.items()},
        **{f"waiting_{name}": values for name, values in waiting_by_replica.items()},
    }
    backend_telemetry_complete = all(
        _complete_replica_metric(values, backend_replicas)
        for values in backend_replica_metrics.values()
    ) and all(value is not None for value in backend_latency.values())
    router_worker_labels = set(router_worker_health)
    backend_worker_mapping = _map_backend_replicas_to_router_workers(
        manifest_targets,
        router_worker_labels,
    )
    backend_worker_mapping_complete = (
        set(backend_worker_mapping) == backend_replicas
        and set(backend_worker_mapping.values()) == router_worker_labels
        and len(set(backend_worker_mapping.values())) == len(backend_replicas)
    )

    def load_by_worker(statistic: str) -> dict[str, float]:
        return {
            backend_worker_mapping[replica]: value
            for replica, value in running_by_replica[statistic].items()
            if replica in backend_worker_mapping
        }

    router_worker_load_mean = load_by_worker("mean")
    router_worker_load_p95 = load_by_worker("p95")
    router_worker_load_max = load_by_worker("max")
    router_worker_load_source = (
        "backend_prometheus_num_requests_running"
        if backend_worker_mapping_complete and concurrency_measurement_complete
        else "unavailable"
    )
    router_worker_load_complete = (
        router_worker_load_source == "backend_prometheus_num_requests_running"
        and all(
            set(values) == router_worker_labels
            for values in (
                router_worker_load_mean,
                router_worker_load_p95,
                router_worker_load_max,
            )
        )
    )

    gates = {
        "registration_confirmed": registration_confirmed,
        "all_targets_ready_at_registration": all_ready,
        "all_targets_up_during_measurement": all_targets_up,
        "prometheus_range_evidence_complete": prometheus_range_evidence_complete,
        "prometheus_scrape_freshness_complete": scrape_freshness_complete,
        "prometheus_measurement_resolution_sufficient": (
            measurement_resolution_sufficient
        ),
        "measurement_window_contains_model_calls": measurement_contains_model_calls,
        "evaluation_coverage_complete": evaluation_coverage_complete,
        "response_error_records_zero": eval_summary["response_error_records"] == 0,
        "model_call_observability_complete": bool(model_call_observability["complete"]),
        "model_call_errors_zero": model_call_errors == 0,
        "output_token_observability_complete": bool(
            eval_summary["output_token_observability"]["complete"]
        ),
        "context_limit_events_zero": eval_summary["context_limit_events"] == 0,
        "backend_per_replica_telemetry_complete": backend_telemetry_complete,
        "backend_concurrency_measurement_window_complete": (
            concurrency_measurement_complete
        ),
        "backend_counter_reset_coverage_complete": (
            backend_counter_reset_coverage_complete
        ),
        "backend_counter_resets_zero": backend_counter_resets_total == 0,
        "backend_cache_metrics_available": cache["backend_prefix_cache"]["available"],
        "backend_cache_replica_coverage_complete": set(
            cache["backend_prefix_cache"]["per_replica"]
        )
        == backend_replicas,
        "accuracy_matches_benchmark_result": accuracy_matches_benchmark,
        "rl_insight_target_lifecycle_isolated": monitoring_config.get(
            "target_lifecycle"
        )
        == "dedicated",
        "warmup_execution_evidence_present": warmup_execution_evidence_present,
    }
    router_counter_resets: dict[str, float | None] = {
        "router_requests_resets": None,
        "router_cache_hits_resets": None,
        "router_cache_misses_resets": None,
    }
    if router_metrics_required:
        reset_metrics = ["router_requests_resets"]
        if routing_policy == "cache_aware":
            reset_metrics.extend(
                ["router_cache_hits_resets", "router_cache_misses_resets"]
            )
        router_counter_resets.update(_scalar_metrics(query_archive, reset_metrics))
        router_operational_scalars = _scalar_metrics(
            query_archive,
            [
                "router_requests",
                "router_request_errors",
                "router_retries",
                "router_retries_exhausted",
                "router_load_balancing_events",
                "router_cb_transitions",
                "router_cb_failures",
                "router_cb_successes",
                "router_active_workers_min",
            ],
        )
        required_native_activity_metrics = set(ROUTER_NATIVE_ACTIVITY_METRICS)
        if routing_policy == "cache_aware":
            required_native_activity_metrics.add("vllm_router_running_requests")
        router_request_activity_observed = (
            (router_operational_scalars["router_requests"] or 0) > 0
            and sum(router_policy_decisions.values()) > 0
            and sum(router_processed_requests.values()) > 0
            and sum(router_cb_successes_by_worker.values()) > 0
        )
        gates.update(
            {
                "router_logs_archived": router_logs_available,
                "router_metrics_adapter_verified": router_adapter_complete,
                "router_native_metric_presence_complete": set(
                    router_native_metric_presence
                )
                == ROUTER_NATIVE_METRICS_AUDITED,
                "router_required_native_activity_metrics_present": all(
                    router_native_metric_presence.get(metric) == 1
                    for metric in required_native_activity_metrics
                ),
                "router_operational_counters_available": all(
                    value is not None for value in router_operational_scalars.values()
                ),
                "router_request_activity_observed": router_request_activity_observed,
                "router_request_errors_zero": router_request_errors == 0,
                "router_retries_exhausted_zero": router_retries_exhausted == 0,
                "router_backend_worker_mapping_complete": (
                    backend_worker_mapping_complete
                ),
                "router_worker_health_complete": (
                    backend_worker_mapping_complete and router_health_observations_valid
                ),
                "router_worker_load_complete": router_worker_load_complete,
                "router_policy_decision_worker_coverage_complete": set(
                    router_policy_decisions
                )
                == router_worker_labels,
                "router_processed_request_worker_coverage_complete": set(
                    router_processed_requests
                )
                == router_worker_labels,
                "router_circuit_breaker_worker_coverage_complete": all(
                    set(values) == router_worker_labels
                    for values in (
                        router_cb_state,
                        router_cb_transitions_by_worker,
                        router_cb_successes_by_worker,
                        router_cb_failures_by_worker,
                    )
                ),
                "router_counter_resets_zero": all(
                    router_counter_resets[name] == 0 for name in reset_metrics
                ),
            }
        )
    if routing_policy == "cache_aware":
        router_cache = cache["router_routing_cache"]
        router_cache_source = router_cache["source"]
        router_cache_observations = router_cache["debug_log_observations"]
        gates.update(
            {
                "router_cache_metrics_available": router_cache["available"],
                "router_cache_metrics_provenance_available": router_cache_source
                in {"native", "debug_log_compat"},
                "router_cache_threshold_available": router_cache["cache_threshold"]
                is not None,
                "router_cache_debug_log_observations_match": (
                    router_cache_source != "debug_log_compat"
                    or (
                        router_cache_observations is not None
                        and router_cache["hits"] is not None
                        and router_cache["misses"] is not None
                        and math.isclose(
                            router_cache_observations,
                            router_cache["hits"] + router_cache["misses"],
                        )
                    )
                ),
            }
        )
    if routing_policy == "consistent_hash":
        affinity = router_logs.get("session_affinity")
        gates["repeated_session_affinity_verified"] = (
            isinstance(affinity, dict) and affinity.get("passed") is True
        )

    evaluation_summary = {
        key: value
        for key, value in eval_summary.items()
        if key not in {"outcomes", "request_records", "session_records"}
    }
    request_distribution = _distribution_statistics(request_success_by_replica)
    generated_distribution = _distribution_statistics(generation_by_replica)
    router_scalars = _scalar_metrics(
        query_archive,
        [
            "router_requests",
            "router_request_errors",
            "router_retries",
            "router_retries_exhausted",
            "router_load_balancing_events",
            "router_cb_transitions",
            "router_cb_failures",
            "router_cb_successes",
            "router_active_workers_min",
            "router_request_duration_p50",
            "router_request_duration_p90",
            "router_request_duration_p99",
        ],
    )

    return {
        "schema_version": 1,
        "run_id": run_id,
        "routing_policy": routing_policy,
        "measurement": {
            "start_time": capture_start,
            "end_time": capture_end,
            "window_seconds": (
                float(capture_end) - float(capture_start)
                if capture_start is not None and capture_end is not None
                else None
            ),
            "prometheus_collection_start_time": query_start,
            "prometheus_collection_end_time": query_end,
            "prometheus_counter_window_seconds": query_archive["window_seconds"],
        },
        "monitoring": {
            "target_lifecycle": monitoring_config.get("target_lifecycle"),
            "scrape_interval_s": scrape_interval_s,
            "range_query_step_s": range_step_s,
            "expected_scrape_periods_during_measurement": (
                measurement_duration_s / scrape_interval_s
            ),
            "max_allowed_scrape_age_s": max_allowed_scrape_age_s,
            "max_observed_scrape_age_s": (
                max(
                    value
                    for values in scrape_age_series.values()
                    for _, value in values
                )
                if scrape_age_series and all(scrape_age_series.values())
                else None
            ),
        },
        "gates": {**gates, "passed": all(gates.values())},
        "accuracy": {
            "available": eval_summary["reward_is_binary"],
            "correct": eval_summary["correct"],
            "evaluated": eval_summary["records"],
            "value": eval_summary["accuracy"],
            "mean_reward": eval_summary["mean_reward"],
            "reward_is_binary": eval_summary["reward_is_binary"],
        },
        "evaluation": evaluation_summary,
        "request_timing": dict(eval_summary["request_timing"]),
        "session_timing": dict(eval_summary["session_timing"]),
        "timing_calculation": {
            "duration_unit": "seconds",
            "percentile_method": "linear_interpolation",
            "implementation": "tools.nemo_gym_phase2_report.percentile",
        },
        "makespan_s": eval_summary["makespan_s"],
        "tail_bubble_s": eval_summary["tail_bubble_s"],
        "throughput": dict(eval_summary["throughput"]),
        "legacy_request_timing": driver_summary.get("legacy_request_timing"),
        "warmup": warmup_evidence,
        "cache": cache,
        "targets": target_up,
        "backend": {
            "concurrency_source": "range_step_measurement_window",
            "request_success_by_replica": request_success_by_replica,
            "request_distribution": request_distribution,
            "prompt_tokens_by_replica": prompt_tokens_by_replica,
            "generation_tokens_by_replica": generation_by_replica,
            "generated_token_distribution": generated_distribution,
            "preemptions_by_replica": preemptions_by_replica,
            "counter_resets_by_replica": counter_resets_by_replica,
            "counter_resets_total": backend_counter_resets_total,
            "running_by_replica": running_by_replica,
            "waiting_by_replica": waiting_by_replica,
            "max_kv_usage_by_replica": kv_usage_by_replica,
            "latency_seconds": backend_latency,
        },
        "router": {
            "applicable": router_metrics_required,
            "metrics_adapter": {
                "verified": router_adapter_complete,
                "samples": [
                    {"labels": sample.labels, "value": sample.value}
                    for sample in router_adapter_samples
                ],
                "native_metric_presence": router_native_metric_presence,
                "worker_health_source": next(iter(router_health_sources), None),
            },
            **router_scalars,
            "backend_replica_to_worker": backend_worker_mapping,
            "worker_load_source": router_worker_load_source,
            "policy_decisions_by_worker": router_policy_decisions,
            "processed_requests_by_worker": router_processed_requests,
            "worker_health_min_by_worker": router_worker_health,
            "worker_load_mean_by_worker": router_worker_load_mean,
            "worker_load_p95_by_worker": router_worker_load_p95,
            "worker_load_max_by_worker": router_worker_load_max,
            "tree_size_max_by_worker": values_by_label(
                query_archive, "router_tree_size_max_by_worker", "worker"
            ),
            "circuit_breaker_state_max_by_worker": router_cb_state,
            "circuit_breaker_transitions_by_worker": router_cb_transitions_by_worker,
            "circuit_breaker_successes_by_worker": router_cb_successes_by_worker,
            "circuit_breaker_failures_by_worker": router_cb_failures_by_worker,
            "counter_resets": router_counter_resets,
            "log_evidence": router_logs,
            "queue": (
                router_logs.get("router_queue")
                if router_metrics_required
                else {"status": "not_applicable_direct"}
            ),
        },
        "raw_rollout_metrics": benchmark_result.get("rollout_metrics"),
    }


def _format_rate(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2%}"


def _format_number(value: float | int | None, digits: int = 4) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact human-readable view backed by ``summary.json``."""
    gates = summary["gates"]
    accuracy = summary["accuracy"]
    evaluation = summary["evaluation"]
    timing = summary["request_timing"]
    session_timing = summary["session_timing"]
    router_cache = summary["cache"]["router_routing_cache"]
    backend_cache = summary["cache"]["backend_prefix_cache"]
    backend = summary["backend"]
    router = summary["router"]
    lines = [
        "# NeMo Gym Router Phase 2 report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Routing policy: `{summary['routing_policy']}`",
        f"- Phase 2 gates: **{'PASS' if gates['passed'] else 'FAIL'}**",
        f"- RL-Insight target lifecycle: `{summary['monitoring']['target_lifecycle']}`",
        f"- Maximum observed scrape age: {_format_number(summary['monitoring']['max_observed_scrape_age_s'])} s",
        "- Percentiles: "
        f"`{summary['timing_calculation']['percentile_method']}` via "
        f"`{summary['timing_calculation']['implementation']}`",
        "",
        "## Correctness and latency",
        "",
        "|Metric|Value|",
        "|---|---:|",
        f"|Evaluated records|{accuracy['evaluated']}|",
        f"|Correct|{_format_number(accuracy['correct'], 0)}|",
        f"|Accuracy|{_format_rate(accuracy['value'])}|",
        f"|Mean reward|{_format_number(accuracy['mean_reward'])}|",
        f"|Response errors|{evaluation['response_error_records']}|",
        f"|Natural termination rate|{_format_rate(evaluation['natural_termination_rate'])}|",
        f"|Truncation rate|{_format_rate(evaluation['truncation_rate'])}|",
        f"|Context-limit events|{evaluation['context_limit_events']}|",
        f"|Model-call samples|{timing['samples']}|",
        f"|Model-call p50|{_format_number(timing['p50_s'])} s|",
        f"|Model-call p90|{_format_number(timing['p90_s'])} s|",
        f"|Model-call p95|{_format_number(timing['p95_s'])} s|",
        f"|Model-call p99|{_format_number(timing['p99_s'])} s|",
        f"|Model-call max|{_format_number(timing['max_s'])} s|",
        f"|Tail amplification (model-call p99 / p50)|{_format_number(timing['p99_over_p50'])}|",
        f"|Session samples|{session_timing['samples']}|",
        f"|Session p50|{_format_number(session_timing['p50_s'])} s|",
        f"|Session p90|{_format_number(session_timing['p90_s'])} s|",
        f"|Session p99|{_format_number(session_timing['p99_s'])} s|",
        f"|Session max|{_format_number(session_timing['max_s'])} s|",
        f"|Makespan|{_format_number(summary['makespan_s'])} s|",
        f"|Tail bubble (makespan - session p90)|{_format_number(summary['tail_bubble_s'])} s|",
        f"|Successful model calls/s|{_format_number(summary['throughput']['successful_model_calls_per_s'])}|",
        f"|Successful trajectories/s|{_format_number(summary['throughput']['successful_trajectories_per_s'])}|",
        f"|Output tokens/s|{_format_number(summary['throughput']['output_tokens_per_s'])}|",
        "",
        "## Cache",
        "",
        "|Layer|Hits|Queries/misses|Hit rate|",
        "|---|---:|---:|---:|",
        f"|Router routing cache|{_format_number(router_cache['hits'], 0)}|"
        f"{_format_number(router_cache['misses'], 0)}|"
        f"{_format_rate(router_cache['hit_rate'])}|",
        f"|Backend prefix cache|{_format_number(backend_cache['hits'], 0)}|"
        f"{_format_number(backend_cache['queries'], 0)}|"
        f"{_format_rate(backend_cache['hit_rate'])}|",
        "",
        f"Router cache metric source: `{router_cache.get('source') or 'unavailable'}`; "
        f"threshold: {_format_number(router_cache.get('cache_threshold'))}; "
        "DEBUG-log observations: "
        f"{_format_number(router_cache.get('debug_log_observations'), 0)}",
        "",
        f"Backend counter resets observed: {_format_number(backend['counter_resets_total'], 0)}",
        "",
        "### Backend prefix cache by replica",
        "",
        "|Replica|Requests|Prompt tokens|Generated tokens|Hits|Queries|Hit rate|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    replicas = sorted(
        set(backend["request_success_by_replica"]) | set(backend_cache["per_replica"])
    )
    for replica in replicas:
        cache_values = backend_cache["per_replica"].get(replica, {})
        lines.append(
            f"|{replica}|"
            f"{_format_number(backend['request_success_by_replica'].get(replica), 0)}|"
            f"{_format_number(backend['prompt_tokens_by_replica'].get(replica), 0)}|"
            f"{_format_number(backend['generation_tokens_by_replica'].get(replica), 0)}|"
            f"{_format_number(cache_values.get('hits'), 0)}|"
            f"{_format_number(cache_values.get('queries'), 0)}|"
            f"{_format_rate(cache_values.get('hit_rate'))}|"
        )
    lines.extend(
        [
            "",
            "## Load distribution",
            "",
            "|Distribution|CV|Largest share|Max / mean|",
            "|---|---:|---:|---:|",
            "|Processed requests|"
            f"{_format_number(backend['request_distribution']['coefficient_of_variation'])}|"
            f"{_format_rate(backend['request_distribution']['largest_replica_share'])}|"
            f"{_format_number(backend['request_distribution']['max_to_mean'])}|",
            "|Generated tokens|"
            f"{_format_number(backend['generated_token_distribution']['coefficient_of_variation'])}|"
            f"{_format_rate(backend['generated_token_distribution']['largest_replica_share'])}|"
            f"{_format_number(backend['generated_token_distribution']['max_to_mean'])}|",
            "",
            "### Backend concurrency by replica",
            "",
            "|Replica|Running mean|Running p95|Running max|Running request-s|Waiting mean|Waiting p95|Waiting max|Waiting request-s|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for replica in replicas:
        running = backend["running_by_replica"]
        waiting = backend["waiting_by_replica"]
        lines.append(
            f"|{replica}|{_format_number(running['mean'].get(replica))}|"
            f"{_format_number(running['p95'].get(replica))}|"
            f"{_format_number(running['max'].get(replica))}|"
            f"{_format_number(running['request_seconds'].get(replica))}|"
            f"{_format_number(waiting['mean'].get(replica))}|"
            f"{_format_number(waiting['p95'].get(replica))}|"
            f"{_format_number(waiting['max'].get(replica))}|"
            f"{_format_number(waiting['request_seconds'].get(replica))}|"
        )
    lines.extend(
        [
            "",
            "## Backend latency",
            "",
            "|Metric|p50|p90|p99|",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ("ttft", "itl", "e2e", "queue"):
        lines.append(
            f"|{metric.upper()}|"
            f"{_format_number(backend['latency_seconds'][f'backend_{metric}_p50'])} s|"
            f"{_format_number(backend['latency_seconds'][f'backend_{metric}_p90'])} s|"
            f"{_format_number(backend['latency_seconds'][f'backend_{metric}_p99'])} s|"
        )
    if router["applicable"]:
        queue = router.get("queue") or {}
        affinity = router.get("log_evidence", {}).get("session_affinity", {})
        adapter = router["metrics_adapter"]
        native_missing = sorted(
            name
            for name, present in adapter["native_metric_presence"].items()
            if present == 0
        )
        lines.extend(
            [
                "",
                "## Router",
                "",
                f"- Owned metrics adapter verified: `{adapter['verified']}`",
                f"- Worker-health source: `{adapter['worker_health_source'] or 'unavailable'}`",
                f"- Worker-load source: `{router['worker_load_source']}`",
                "- Native exporter metric families absent during final query: "
                + (", ".join(f"`{name}`" for name in native_missing) or "none"),
                f"- Requests: {_format_number(router['router_requests'], 0)}",
                f"- Request errors: {_format_number(router['router_request_errors'], 0)}",
                f"- Retries / exhausted: {_format_number(router['router_retries'], 0)} / {_format_number(router['router_retries_exhausted'], 0)}",
                f"- Active workers (minimum): {_format_number(router['router_active_workers_min'], 0)}",
                f"- Circuit-breaker transitions: {_format_number(router['router_cb_transitions'], 0)}",
                f"- Circuit-breaker outcomes (success/failure): {_format_number(router['router_cb_successes'], 0)} / {_format_number(router['router_cb_failures'], 0)}",
                f"- Router queue metric: `{queue.get('status', 'unavailable')}`",
                "- Router counter resets (requests/cache hits/cache misses): "
                f"{_format_number(router['counter_resets']['router_requests_resets'], 0)} / "
                f"{_format_number(router['counter_resets']['router_cache_hits_resets'], 0)} / "
                f"{_format_number(router['counter_resets']['router_cache_misses_resets'], 0)}",
                f"- Repeated session-affinity keys / violations: {affinity.get('repeated_keys', 0)} / {len(affinity.get('violations', {}))}",
            ]
        )
        router_workers = sorted(
            set(router["policy_decisions_by_worker"])
            | set(router["processed_requests_by_worker"])
            | set(router["worker_health_min_by_worker"])
            | set(router["worker_load_max_by_worker"])
        )
        lines.extend(
            [
                "",
                "### Router behavior by worker",
                "",
                "|Worker|Policy decisions|Processed|Health min|Running/load mean|Running/load p95|Running/load max|CB state max|CB transitions|CB successes|CB failures|",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for worker in router_workers:
            lines.append(
                f"|{worker}|"
                f"{_format_number(router['policy_decisions_by_worker'].get(worker), 0)}|"
                f"{_format_number(router['processed_requests_by_worker'].get(worker), 0)}|"
                f"{_format_number(router['worker_health_min_by_worker'].get(worker))}|"
                f"{_format_number(router['worker_load_mean_by_worker'].get(worker))}|"
                f"{_format_number(router['worker_load_p95_by_worker'].get(worker))}|"
                f"{_format_number(router['worker_load_max_by_worker'].get(worker))}|"
                f"{_format_number(router['circuit_breaker_state_max_by_worker'].get(worker))}|"
                f"{_format_number(router['circuit_breaker_transitions_by_worker'].get(worker), 0)}|"
                f"{_format_number(router['circuit_breaker_successes_by_worker'].get(worker), 0)}|"
                f"{_format_number(router['circuit_breaker_failures_by_worker'].get(worker), 0)}|"
            )
    lines.extend(
        [
            "",
            "## Gates",
            "",
        ]
    )
    for name, passed in gates.items():
        if name == "passed":
            continue
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _backend_csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    backend = summary["backend"]
    cache = summary["cache"]["backend_prefix_cache"]["per_replica"]
    replicas = sorted(
        set(backend["request_success_by_replica"])
        | set(backend["prompt_tokens_by_replica"])
        | set(backend["generation_tokens_by_replica"])
        | set(cache)
    )
    rows = []
    for replica in replicas:
        cache_values = cache.get(replica, {})
        rows.append(
            {
                "replica": replica,
                "processed_requests": backend["request_success_by_replica"].get(
                    replica
                ),
                "prompt_tokens": backend["prompt_tokens_by_replica"].get(replica),
                "generated_tokens": backend["generation_tokens_by_replica"].get(
                    replica
                ),
                "prefix_cache_hits": cache_values.get("hits"),
                "prefix_cache_queries": cache_values.get("queries"),
                "prefix_cache_hit_rate": cache_values.get("hit_rate"),
                "running_mean": backend["running_by_replica"]["mean"].get(replica),
                "running_p95": backend["running_by_replica"]["p95"].get(replica),
                "running_max": backend["running_by_replica"]["max"].get(replica),
                "running_request_seconds": backend["running_by_replica"][
                    "request_seconds"
                ].get(replica),
                "waiting_mean": backend["waiting_by_replica"]["mean"].get(replica),
                "waiting_p95": backend["waiting_by_replica"]["p95"].get(replica),
                "waiting_max": backend["waiting_by_replica"]["max"].get(replica),
                "waiting_request_seconds": backend["waiting_by_replica"][
                    "request_seconds"
                ].get(replica),
                "kv_usage_max": backend["max_kv_usage_by_replica"].get(replica),
                "preemptions": backend["preemptions_by_replica"].get(replica),
                "counter_resets": sum(
                    values.get(replica, 0)
                    for values in backend["counter_resets_by_replica"].values()
                ),
            }
        )
    return rows


def _write_backend_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows = _backend_csv_rows(summary)
    if not rows:
        raise ValueError("cannot write backend CSV without replica data")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg_frame(title: str, body: str, *, width: int = 960, height: int = 520) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="24" y="32" font-family="sans-serif" font-size="20">{html.escape(title)}</text>\n'
        f"{body}\n</svg>\n"
    )


def _write_ecdf_svg(path: Path, eval_summary: Mapping[str, Any]) -> None:
    series = {
        "model call": sorted(
            float(record["duration_s"])
            for record in eval_summary["request_records"]
            if record["duration_s"] is not None
        ),
        "session": sorted(
            float(record["duration_s"])
            for record in eval_summary["session_records"]
            if record["duration_s"] is not None
        ),
    }
    nonempty = [values for values in series.values() if values]
    body = ""
    if nonempty:
        x_max = max(values[-1] for values in nonempty)
        x_max = x_max if x_max > 0 else 1.0
        colors = {"model call": "#1f77b4", "session": "#d62728"}
        for series_index, (name, values) in enumerate(series.items()):
            if not values:
                continue
            points = []
            for index, value in enumerate(values, start=1):
                x = 72 + 840 * value / x_max
                y = 456 - 390 * index / len(values)
                points.append(f"{x:.2f},{y:.2f}")
            body += (
                f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2" '
                f'points="{" ".join(points)}"/>\n'
                f'<text x="760" y="{58 + 22 * series_index}" font-family="sans-serif" '
                f'font-size="13" fill="{colors[name]}">{html.escape(name)}</text>\n'
            )
        body += (
            '<line x1="72" y1="456" x2="912" y2="456" stroke="black"/>\n'
            '<line x1="72" y1="66" x2="72" y2="456" stroke="black"/>\n'
            f'<text x="820" y="490" font-family="sans-serif" font-size="13">seconds (max {x_max:.3f})</text>\n'
            '<text x="16" y="76" font-family="sans-serif" font-size="13">1.0</text>\n'
        )
    else:
        body = '<text x="24" y="72" font-family="sans-serif">No complete timing observations.</text>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_frame("Completion ECDF", body), encoding="utf-8")


def _series_replica(
    series: Mapping[tuple[tuple[str, str], ...], Sequence[tuple[float, float]]],
) -> dict[str, Sequence[tuple[float, float]]]:
    result = {}
    for labels, values in series.items():
        label_map = dict(labels)
        replica = label_map.get("replica")
        if replica is None or replica in result:
            raise ValueError("range series is missing or duplicates replica labels")
        result[replica] = values
    return dict(sorted(result.items()))


def _write_concurrency_drain_svg(
    path: Path,
    query_archive: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
) -> None:
    running = _series_replica(
        parse_range_samples(query_archive, "backend_running_by_replica")
    )
    waiting = _series_replica(
        parse_range_samples(query_archive, "backend_waiting_by_replica")
    )
    by_timestamp: dict[float, list[float]] = {}
    start_time = float(eval_summary["measurement_start_time"])
    end_time = float(eval_summary["measurement_end_time"])
    for value_by_replica in (running, waiting):
        for values in value_by_replica.values():
            for timestamp, value in _clip_step_series(
                values, start_time=start_time, end_time=end_time
            ):
                by_timestamp.setdefault(timestamp, []).append(value)
    points = sorted(
        (timestamp, sum(values)) for timestamp, values in by_timestamp.items()
    )
    if points:
        x_min, x_max = points[0][0], points[-1][0]
        x_span = max(x_max - x_min, 1.0)
        y_max = max(max(value for _, value in points), 1.0)
        coordinates = [
            f"{72 + 840 * (timestamp - x_min) / x_span:.2f},{456 - 390 * value / y_max:.2f}"
            for timestamp, value in points
        ]
        body = (
            '<polyline fill="none" stroke="#1f77b4" stroke-width="2" '
            f'points="{" ".join(coordinates)}"/>\n'
            '<line x1="72" y1="456" x2="912" y2="456" stroke="black"/>\n'
            '<line x1="72" y1="66" x2="72" y2="456" stroke="black"/>\n'
            '<text x="750" y="490" font-family="sans-serif" font-size="13">measurement time</text>\n'
            f'<text x="16" y="76" font-family="sans-serif" font-size="13">{y_max:.1f}</text>\n'
        )
    else:
        body = '<text x="24" y="72" font-family="sans-serif">No concurrency samples.</text>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _svg_frame("Backend running + waiting drain", body), encoding="utf-8"
    )


def _write_concurrency_heatmap_svg(
    path: Path,
    query_archive: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
) -> None:
    running = _series_replica(
        parse_range_samples(query_archive, "backend_running_by_replica")
    )
    start_time = float(eval_summary["measurement_start_time"])
    end_time = float(eval_summary["measurement_end_time"])
    running = {
        replica: _clip_step_series(values, start_time=start_time, end_time=end_time)
        for replica, values in running.items()
    }
    ranking = sorted(
        running,
        key=lambda replica: (
            -(
                _step_series_statistics(
                    running[replica], start_time=start_time, end_time=end_time
                )
                or {"request_seconds": 0.0}
            )["request_seconds"],
            replica,
        ),
    )
    all_values = [value for values in running.values() for _, value in values]
    maximum = max(all_values, default=0.0)
    body = ""
    if running:
        cell_width = 840 / max(max(len(values) for values in running.values()), 1)
        cell_height = min(52.0, 390 / len(running))
        for row, replica in enumerate(ranking):
            values = running[replica]
            y = 66 + row * cell_height
            body += (
                f'<text x="18" y="{y + cell_height * 0.68:.2f}" '
                f'font-family="sans-serif" font-size="12">{html.escape(replica)}</text>\n'
            )
            for column, (_, value) in enumerate(values):
                intensity = value / maximum if maximum > 0 else 0.0
                red = round(245 - 190 * intensity)
                green = round(248 - 100 * intensity)
                blue = round(255 - 25 * intensity)
                body += (
                    f'<rect x="{72 + column * cell_width:.2f}" y="{y:.2f}" '
                    f'width="{cell_width + 0.2:.2f}" height="{cell_height:.2f}" '
                    f'fill="rgb({red},{green},{blue})"/>\n'
                )
        body += f'<text x="760" y="490" font-family="sans-serif" font-size="13">max running {maximum:.1f}</text>\n'
    else:
        body = '<text x="24" y="72" font-family="sans-serif">No per-replica concurrency samples.</text>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_frame("Instance concurrency heatmap", body), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _require_object(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"experiment metadata {key!r} must be a JSON object")
    return value


def _require_text(parent: Mapping[str, Any], key: str, *, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"experiment metadata {context}.{key} must be non-empty")
    return value.strip()


def _require_positive_int(parent: Mapping[str, Any], key: str, *, context: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"experiment metadata {context}.{key} must be a positive integer"
        )
    return value


def _jsonl_record_count(path: Path) -> int:
    return len(_jsonl_records(path))


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: expected at least one JSONL record")
    return records


def validate_experiment_metadata(
    metadata: Mapping[str, Any],
    *,
    target_manifest: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
    workload_path: Path,
    warmup_workload_path: Path,
    workload_seed: str,
) -> dict[str, Any]:
    """Validate and normalize the formal Phase 2 experiment declaration."""
    if metadata.get("schema_version") != 1:
        raise ValueError("experiment metadata schema_version must be 1")

    engine = _require_object(metadata, "engine")
    if engine.get("fresh") is not True:
        raise ValueError("experiment metadata engine.fresh must be true")
    launch_id = _require_text(engine, "launch_id", context="engine")

    replay = _require_object(metadata, "workload_replay")
    if replay.get("faithful") is not True:
        raise ValueError("experiment metadata workload_replay.faithful must be true")
    replay_seed = _require_text(replay, "seed", context="workload_replay")
    if replay_seed != workload_seed:
        raise ValueError("workload replay seed differs from --workload-seed")
    workload_sha256 = _require_text(
        replay, "workload_sha256", context="workload_replay"
    )
    if workload_sha256 != _sha256(workload_path):
        raise ValueError("workload replay SHA-256 differs from --workload-file")
    num_prompts = _require_positive_int(
        replay, "num_prompts", context="workload_replay"
    )
    num_generations = _require_positive_int(
        replay, "num_generations_per_prompt", context="workload_replay"
    )
    if num_prompts != int(eval_summary["num_prompts"]):
        raise ValueError("workload replay num_prompts differs from evaluation coverage")
    if num_generations != int(eval_summary["num_generations_per_prompt"]):
        raise ValueError(
            "workload replay num_generations_per_prompt differs from evaluation coverage"
        )
    if _jsonl_record_count(workload_path) != num_prompts:
        raise ValueError("workload JSONL record count differs from num_prompts")

    warmup = _require_object(metadata, "warmup")
    if warmup.get("completed") is not True:
        raise ValueError("experiment metadata warmup.completed must be true")
    warmup_source = _require_text(warmup, "source", context="warmup")
    if warmup_source != "measurement_workload_prefix":
        raise ValueError(
            "experiment metadata warmup.source must be 'measurement_workload_prefix'"
        )
    warmup_requests = _require_positive_int(warmup, "requests", context="warmup")
    warmup_sha256 = _require_text(warmup, "workload_sha256", context="warmup")
    if warmup_sha256 != _sha256(warmup_workload_path):
        raise ValueError("warmup SHA-256 differs from --warmup-workload-file")
    if _jsonl_record_count(warmup_workload_path) != warmup_requests:
        raise ValueError("warmup JSONL record count differs from warmup.requests")
    if (
        _jsonl_records(warmup_workload_path)
        != _jsonl_records(workload_path)[:warmup_requests]
    ):
        raise ValueError(
            "warmup workload does not match the measured workload prefix executed "
            "by the Phase 2 warmup hook"
        )

    software = _require_object(metadata, "software")
    nemo_rl_commit = _require_text(software, "nemo_rl_commit", context="software")
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", nemo_rl_commit) is None:
        raise ValueError("software.nemo_rl_commit must be a full Git object ID")
    container_digest = _require_text(software, "container_digest", context="software")
    if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", container_digest) is None:
        raise ValueError("software.container_digest must be sha256:<64 hex digits>")

    model = _require_object(metadata, "model")
    model_name = _require_text(model, "name", context="model")
    for key in ("revision", "tokenizer", "tokenizer_revision"):
        _require_text(model, key, context="model")
    chat_template_sha256 = _require_text(model, "chat_template_sha256", context="model")
    if re.fullmatch(r"[0-9a-fA-F]{64}", chat_template_sha256) is None:
        raise ValueError("model.chat_template_sha256 must contain 64 hex digits")
    target_models = {
        str(target["labels"]["model"])
        for target in target_manifest.get("targets", [])
        if isinstance(target, dict) and isinstance(target.get("labels"), dict)
    }
    if target_models != {model_name}:
        raise ValueError("experiment model name differs from Prometheus target labels")

    topology = _require_object(metadata, "topology")
    for key in (
        "tensor_parallel_size",
        "data_parallel_size",
        "num_nodes",
        "gpus_per_node",
    ):
        _require_positive_int(topology, key, context="topology")
    backend_replicas = {
        str(target["labels"]["replica"])
        for target in target_manifest.get("targets", [])
        if isinstance(target, dict)
        and isinstance(target.get("labels"), dict)
        and target["labels"].get("component") == "vllm_backend"
    }
    if int(topology["data_parallel_size"]) != len(backend_replicas):
        raise ValueError(
            "topology.data_parallel_size differs from discovered backend replicas"
        )

    generation = _require_object(metadata, "generation")
    for key in ("concurrency", "max_context_tokens", "max_output_tokens"):
        _require_positive_int(generation, key, context="generation")
    sampling_parameters = generation.get("sampling_parameters")
    if not isinstance(sampling_parameters, dict) or not sampling_parameters:
        raise ValueError(
            "experiment metadata generation.sampling_parameters must be non-empty"
        )

    backend = _require_object(metadata, "backend")
    if not isinstance(backend.get("prefix_caching_enabled"), bool):
        raise ValueError("backend.prefix_caching_enabled must be boolean")
    for key in ("scheduler_parameters", "batching_parameters"):
        value = backend.get(key)
        if not isinstance(value, dict) or not value:
            raise ValueError(f"experiment metadata backend.{key} must be non-empty")

    router = _require_object(metadata, "router")
    routing_policy = _require_text(router, "policy", context="router")
    if routing_policy not in {"direct", "cache_aware", "consistent_hash"}:
        raise ValueError(
            f"experiment metadata router.policy is invalid: {routing_policy}"
        )
    target_policies = {
        str(target["labels"]["routing_policy"])
        for target in target_manifest.get("targets", [])
        if isinstance(target, dict) and isinstance(target.get("labels"), dict)
    }
    if target_policies != {routing_policy}:
        raise ValueError(
            "experiment router policy differs from Prometheus target labels"
        )
    affinity_header = _require_text(router, "session_affinity_header", context="router")
    if affinity_header != "X-Session-ID":
        raise ValueError("router.session_affinity_header must be X-Session-ID")
    router_enabled = router.get("enabled")
    if not isinstance(router_enabled, bool) or router_enabled != (
        routing_policy != "direct"
    ):
        raise ValueError("router.enabled does not match router.policy")
    cache_metrics_mode = _require_text(router, "cache_metrics_mode", context="router")
    expected_cache_metrics_mode = (
        "debug_log_compat" if routing_policy == "cache_aware" else "native"
    )
    if cache_metrics_mode != expected_cache_metrics_mode:
        raise ValueError(
            "router.cache_metrics_mode does not match the Phase 2 routing policy: "
            f"expected {expected_cache_metrics_mode!r}"
        )
    cache_threshold = _as_optional_float(router.get("cache_threshold"))
    if cache_threshold is None or not 0 <= cache_threshold <= 1:
        raise ValueError("router.cache_threshold must be a number between 0 and 1")

    normalized = json.loads(json.dumps(metadata))
    comparison_invariants = {
        "workload_replay": {
            "workload_sha256": workload_sha256,
            "seed": replay_seed,
            "num_prompts": num_prompts,
            "num_generations_per_prompt": num_generations,
        },
        "warmup": {
            "source": warmup_source,
            "workload_sha256": warmup_sha256,
            "requests": warmup_requests,
        },
        "software": dict(software),
        "model": dict(model),
        "topology": dict(topology),
        "generation": dict(generation),
        "backend": dict(backend),
        "router_contract": {
            "session_affinity_header": affinity_header,
            "cache_threshold": cache_threshold,
        },
    }
    normalized["derived"] = {
        "engine_launch_id": launch_id,
        "workload_file_sha256": workload_sha256,
        "warmup_workload_file_sha256": warmup_sha256,
        "comparison_invariants": comparison_invariants,
        "comparison_invariants_sha256": _canonical_sha256(comparison_invariants),
    }
    return normalized


def _write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "artifact_checksums.sha256"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _parse_key_value(values: Sequence[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise ValueError(f"{option} expects NAME=VALUE, got {value!r}")
        if key in result:
            raise ValueError(f"{option} repeats {key!r}")
        result[key] = item
    return dict(sorted(result.items()))


def write_report_artifacts(
    output_dir: Path,
    *,
    target_manifest_path: Path,
    target_manifest: Mapping[str, Any],
    query_archive: Mapping[str, Any],
    driver_log_path: Path,
    eval_results_path: Path,
    workload_path: Path,
    warmup_workload_path: Path,
    workload_seed: str,
    repeat_id: str,
    command_path: Path,
    experiment_metadata_path: Path,
    experiment_metadata: Mapping[str, Any],
    config_paths: Sequence[Path],
    versions: Mapping[str, str],
    backend_log_paths: Mapping[str, str],
    eval_summary: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the complete auditable per-run artifact tree."""
    validated_experiment = validate_experiment_metadata(
        experiment_metadata,
        target_manifest=target_manifest,
        eval_summary=eval_summary,
        workload_path=workload_path,
        warmup_workload_path=warmup_workload_path,
        workload_seed=workload_seed,
    )
    comparison_invariants = {
        "experiment": validated_experiment["derived"]["comparison_invariants"],
        "versions": dict(sorted(versions.items())),
        "monitoring": {
            "scrape_interval_s": target_manifest.get("monitoring_config", {}).get(
                "scrape_interval_s"
            ),
            "target_lifecycle": target_manifest.get("monitoring_config", {}).get(
                "target_lifecycle"
            ),
        },
    }
    comparison_invariants_sha256 = _canonical_sha256(comparison_invariants)
    validated_experiment["derived"]["matrix_comparison_invariants"] = (
        comparison_invariants
    )
    validated_experiment["derived"]["matrix_comparison_invariants_sha256"] = (
        comparison_invariants_sha256
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"report output directory must be empty to avoid stale artifacts: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_file(command_path, output_dir / "command.txt")
    _copy_file(target_manifest_path, output_dir / "prometheus-targets.json")
    _copy_file(eval_results_path, output_dir / "evaluation" / "results.jsonl")
    _copy_file(workload_path, output_dir / "evaluation" / "workload.jsonl")
    _copy_file(
        warmup_workload_path, output_dir / "evaluation" / "warmup-workload.jsonl"
    )
    _copy_file(
        experiment_metadata_path, output_dir / "experiment" / "input-metadata.json"
    )
    _write_json(output_dir / "experiment" / "metadata.json", validated_experiment)
    _copy_file(driver_log_path, output_dir / "logs" / "driver.log")
    for index, config_path in enumerate(config_paths):
        destination_name = f"{index:02d}-{config_path.name}"
        _copy_file(config_path, output_dir / "config" / destination_name)

    router_log_paths = [
        Path(value) for value in target_manifest.get("router_log_paths", [])
    ]
    if summary["router"]["applicable"]:
        by_stream = {
            "stdout": next(
                (path for path in router_log_paths if "stdout" in path.name), None
            ),
            "stderr": next(
                (path for path in router_log_paths if "stderr" in path.name), None
            ),
        }
        for stream, source in by_stream.items():
            if source is None:
                raise ValueError(
                    f"Router {stream} log path is missing from target manifest"
                )
            _copy_file(source, output_dir / "logs" / f"router.{stream}.log")
    else:
        for stream in ("stdout", "stderr"):
            (output_dir / "logs").mkdir(parents=True, exist_ok=True)
            (output_dir / "logs" / f"router.{stream}.log").write_text(
                "not applicable: direct arm has no Router process\n", encoding="utf-8"
            )

    expected_backend_replicas = {
        str(target["labels"]["replica"])
        for target in target_manifest["targets"]
        if target["labels"]["component"] == "vllm_backend"
    }
    if set(backend_log_paths) != expected_backend_replicas:
        raise ValueError(
            "backend logs must cover exactly the Prometheus backend replicas: "
            f"expected {sorted(expected_backend_replicas)}, got {sorted(backend_log_paths)}"
        )
    for replica, source in backend_log_paths.items():
        _copy_file(
            Path(source), output_dir / "logs" / "backends" / f"replica-{replica}.log"
        )

    _write_json(output_dir / "metrics" / "prometheus-query-results.json", query_archive)
    _write_backend_csv(output_dir / "metrics" / "backend-per-replica.csv", summary)
    _write_jsonl(
        output_dir / "requests" / "per-request.jsonl",
        eval_summary["request_records"],
    )
    _write_jsonl(output_dir / "evaluation" / "outcomes.jsonl", eval_summary["outcomes"])
    _write_ecdf_svg(output_dir / "figures" / "completion-ecdf.svg", eval_summary)
    _write_concurrency_drain_svg(
        output_dir / "figures" / "concurrency-drain.svg",
        query_archive,
        eval_summary,
    )
    _write_concurrency_heatmap_svg(
        output_dir / "figures" / "instance-concurrency-heatmap.svg",
        query_archive,
        eval_summary,
    )

    finalized_summary = dict(summary)
    finalized_summary["experiment"] = validated_experiment
    finalized_gates = dict(summary["gates"])
    finalized_gates["required_artifact_inputs_complete"] = bool(
        config_paths and versions and backend_log_paths and validated_experiment
    )
    finalized_gates["formal_experiment_metadata_complete"] = True
    finalized_gates["workload_replay_faithful"] = True
    warmup_evidence = finalized_summary.get("warmup")
    warmup_execution_matches_metadata = (
        isinstance(warmup_evidence, dict)
        and warmup_evidence.get("status") == "completed"
        and warmup_evidence.get("source") == validated_experiment["warmup"]["source"]
        and warmup_evidence.get("requests")
        == validated_experiment["warmup"]["requests"]
        and warmup_evidence.get("workload_sha256")
        == validated_experiment["warmup"]["workload_sha256"]
    )
    finalized_gates["warmup_execution_matches_metadata"] = (
        warmup_execution_matches_metadata
    )
    finalized_gates["warmup_completed"] = warmup_execution_matches_metadata
    finalized_gates["fresh_engines_declared"] = True
    if finalized_summary["routing_policy"] == "cache_aware":
        router_metadata = validated_experiment["router"]
        router_cache = finalized_summary["cache"]["router_routing_cache"]
        finalized_gates["router_cache_provenance_matches_experiment"] = (
            router_metadata["cache_metrics_mode"] == router_cache["source"]
            and router_cache["cache_threshold"] is not None
            and math.isclose(
                float(router_metadata["cache_threshold"]),
                float(router_cache["cache_threshold"]),
            )
        )
    finalized_gates["passed"] = all(
        value for name, value in finalized_gates.items() if name != "passed"
    )
    finalized_summary["gates"] = finalized_gates
    _write_json(output_dir / "summary.json", finalized_summary)
    (output_dir / "report.md").write_text(
        render_markdown(finalized_summary), encoding="utf-8"
    )

    artifact_paths = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    artifact_paths.extend(["artifact_checksums.sha256", "manifest.json"])
    artifact_paths = sorted(set(artifact_paths))
    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": finalized_summary["run_id"],
        "repeat_id": repeat_id,
        "engine_launch_id": validated_experiment["derived"]["engine_launch_id"],
        "comparison_invariants_sha256": comparison_invariants_sha256,
        "routing_policy": finalized_summary["routing_policy"],
        "measurement": finalized_summary["measurement"],
        "versions": dict(versions),
        "targets": target_manifest["targets"],
        "registration": target_manifest["registration"],
        "input_data": {
            "records": eval_summary["records"],
            "prompts": eval_summary["num_prompts"],
            "generations_per_prompt": eval_summary["num_generations_per_prompt"],
            "evaluation_results_sha256": _sha256(eval_results_path),
            "workload_sha256": _sha256(workload_path),
            "workload_seed": workload_seed,
            "workload_replay_faithful": True,
            "warmup_workload_sha256": _sha256(warmup_workload_path),
            "warmup_requests": validated_experiment["warmup"]["requests"],
        },
        "experiment": validated_experiment,
        "config_sha256": {
            f"{index:02d}-{path.name}": _sha256(path)
            for index, path in enumerate(config_paths)
        },
        "sources": {
            "command": str(command_path),
            "configs": [str(path) for path in config_paths],
            "driver_log": str(driver_log_path),
            "prometheus_targets": str(target_manifest_path),
            "workload": str(workload_path),
            "warmup_workload": str(warmup_workload_path),
            "experiment_metadata": str(experiment_metadata_path),
            "model_call_capture_dir": target_manifest.get("model_call_capture_dir"),
        },
        "artifacts": artifact_paths,
    }
    _write_json(output_dir / "manifest.json", run_manifest)
    _write_checksums(output_dir)
    required_artifacts = {
        "command.txt",
        "manifest.json",
        "prometheus-targets.json",
        "logs/router.stdout.log",
        "logs/router.stderr.log",
        "metrics/prometheus-query-results.json",
        "metrics/backend-per-replica.csv",
        "requests/per-request.jsonl",
        "evaluation/results.jsonl",
        "evaluation/workload.jsonl",
        "evaluation/warmup-workload.jsonl",
        "evaluation/outcomes.jsonl",
        "experiment/input-metadata.json",
        "experiment/metadata.json",
        "summary.json",
        "report.md",
        "artifact_checksums.sha256",
        "figures/completion-ecdf.svg",
        "figures/concurrency-drain.svg",
        "figures/instance-concurrency-heatmap.svg",
    }
    present_artifacts = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    missing_artifacts = sorted(required_artifacts - present_artifacts)
    if missing_artifacts:
        raise RuntimeError(
            "report writer did not produce required artifacts: "
            + ", ".join(missing_artifacts)
        )
    finalized_gates["required_artifacts_present"] = True
    finalized_gates["passed"] = all(
        value for name, value in finalized_gates.items() if name != "passed"
    )
    _write_json(output_dir / "summary.json", finalized_summary)
    (output_dir / "report.md").write_text(
        render_markdown(finalized_summary), encoding="utf-8"
    )
    _write_checksums(output_dir)
    return finalized_summary


def parse_timestamp(value: str) -> float:
    """Parse Unix seconds or an RFC 3339/ISO-8601 timestamp."""
    try:
        return float(value)
    except ValueError:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            raise ValueError("timestamps without a timezone are not allowed")
        return parsed.astimezone(timezone.utc).timestamp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prometheus-targets",
        "--manifest",
        dest="prometheus_targets",
        type=Path,
        required=True,
    )
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--workload-file", type=Path, required=True)
    parser.add_argument("--warmup-workload-file", type=Path, required=True)
    parser.add_argument("--workload-seed", required=True)
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--experiment-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument(
        "--version",
        action="append",
        default=[],
        help="Audited component version as NAME=VALUE; repeat for every component.",
    )
    parser.add_argument(
        "--backend-log",
        action="append",
        default=[],
        help="Backend evidence log as REPLICA=PATH; defaults to the target manifest.",
    )
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prometheus-url")
    source.add_argument("--prometheus-query-results", type=Path)
    parser.add_argument("--request-timeout-s", type=float, default=10.0)
    parser.add_argument("--range-step-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.prometheus_targets.read_text(encoding="utf-8"))
    experiment_metadata = json.loads(
        args.experiment_metadata.read_text(encoding="utf-8")
    )
    if not isinstance(experiment_metadata, dict):
        raise TypeError("--experiment-metadata must contain a JSON object")
    run_id = str(manifest["run_id"])
    driver_summary = summarize_driver_log(args.driver_log)
    eval_summary = summarize_eval_results(args.eval_results)
    if args.prometheus_query_results is not None:
        query_archive = json.loads(
            args.prometheus_query_results.read_text(encoding="utf-8")
        )
    else:
        monitoring_config = manifest.get("monitoring_config")
        monitoring_config = (
            monitoring_config if isinstance(monitoring_config, dict) else {}
        )
        if (args.start_time is None) != (args.end_time is None):
            raise ValueError("--start-time and --end-time must be provided together")
        if args.start_time is not None:
            start_time = parse_timestamp(args.start_time)
            end_time = parse_timestamp(args.end_time)
        else:
            captured_start = eval_summary.get("measurement_start_time")
            captured_end = eval_summary.get("measurement_end_time")
            if captured_start is None or captured_end is None:
                raise ValueError(
                    "cannot infer the query window without complete Gym model-call timing"
                )
            start_time = float(captured_start) - float(
                monitoring_config.get("initial_scrape_wait_s", 12.0)
            )
            end_time = float(captured_end) + float(
                monitoring_config.get("final_scrape_wait_s", 12.0)
            )
        query_archive = collect_prometheus_queries(
            PrometheusClient(
                args.prometheus_url,
                timeout_s=args.request_timeout_s,
            ),
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            range_step_seconds=args.range_step_s,
        )

    router_log_paths = [Path(value) for value in manifest.get("router_log_paths", [])]
    router_log_summary = summarize_router_logs(router_log_paths)
    summary = build_summary(
        manifest=manifest,
        query_archive=query_archive,
        driver_summary=driver_summary,
        eval_summary=eval_summary,
        router_log_summary=router_log_summary,
    )
    manifest_backend_logs = manifest.get("backend_log_paths")
    if args.backend_log:
        backend_log_paths = _parse_key_value(args.backend_log, option="--backend-log")
    elif isinstance(manifest_backend_logs, dict):
        backend_log_paths = {
            str(replica): str(path) for replica, path in manifest_backend_logs.items()
        }
    else:
        raise ValueError(
            "backend evidence logs are required; pass --backend-log REPLICA=PATH"
        )
    manifest_versions = manifest.get("versions")
    versions = (
        {
            str(component): str(version)
            for component, version in manifest_versions.items()
        }
        if isinstance(manifest_versions, dict)
        else {}
    )
    versions.update(_parse_key_value(args.version, option="--version"))
    required_versions = {
        "python",
        "nemo_rl",
        "nemo_gym",
        "vllm",
        "vllm_router",
        "uv",
        "rl_insight",
    }
    missing_versions = sorted(
        component
        for component in required_versions
        if versions.get(component) in {None, "", "unavailable"}
    )
    if missing_versions:
        raise ValueError(
            "formal Phase 2 report requires component versions; pass --version for: "
            + ", ".join(missing_versions)
        )
    finalized_summary = write_report_artifacts(
        args.output_dir,
        target_manifest_path=args.prometheus_targets,
        target_manifest=manifest,
        query_archive=query_archive,
        driver_log_path=args.driver_log,
        eval_results_path=args.eval_results,
        workload_path=args.workload_file,
        warmup_workload_path=args.warmup_workload_file,
        workload_seed=args.workload_seed,
        repeat_id=args.repeat_id,
        command_path=args.command_file,
        experiment_metadata_path=args.experiment_metadata,
        experiment_metadata=experiment_metadata,
        config_paths=args.config,
        versions=versions,
        backend_log_paths=backend_log_paths,
        eval_summary=eval_summary,
        summary=summary,
    )
    print(args.output_dir / "report.md")
    if finalized_summary["gates"]["passed"] is not True:
        raise SystemExit("Phase 2 single-run gates failed; see report.md")


if __name__ == "__main__":
    main()
