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

"""Prometheus polling lifecycle for managed Dynamo workers."""

import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from nemo_rl.models.generation.dynamo.refit import DynamoWorkerEndpoint

LOGGER = logging.getLogger(__name__)

DEFAULT_METRICS_EXCLUDE_PREFIXES = ("python_", "process_")
CURATED_METRICS_INCLUDE_PREFIXES = (
    "dynamo_component_gpu_cache_usage",
    "dynamo_component_inflight_requests",
    "dynamo_work_handler_queue_depth",
    "dynamo_component_requests_total",
    "dynamo_work_handler_time_to_first_response",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:generation_tokens",
    "vllm:prompt_tokens_total",
    "vllm:inter_token_latency",
)
CANONICAL_LOGGER_ALIASES = {
    "inflight_batch_sizes": [
        "dynamo_component_inflight_requests",
        "vllm_num_requests_running",
    ],
    "num_pending_samples": [
        "dynamo_work_handler_queue_depth",
        "vllm_num_requests_waiting",
    ],
    "kv_cache_usage_perc": [
        "dynamo_component_gpu_cache_usage_percent",
        "vllm_kv_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
    ],
    "generation_tokens": [
        "vllm_generation_tokens_total",
        "vllm_generation_tokens",
    ],
}


def _http_get_text(url: str, timeout_s: float) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def parse_prometheus_metrics(
    text: str,
    include_prefixes: tuple[str, ...] | None = None,
    exclude_prefixes: tuple[str, ...] = DEFAULT_METRICS_EXCLUDE_PREFIXES,
) -> dict[str, float]:
    """Parse Prometheus text exposition into summed scalar values."""
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name = line[: line.index("{")]
            try:
                value_text = line[line.rindex("}") + 1 :]
            except ValueError:
                continue
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, value_text = parts
        if name.endswith(("_bucket", "_created")):
            continue
        if include_prefixes and not name.startswith(include_prefixes):
            continue
        if exclude_prefixes and name.startswith(exclude_prefixes):
            continue
        value_parts = value_text.split()
        if not value_parts:
            continue
        try:
            value = float(value_parts[0])
        except ValueError:
            continue
        key = name.replace(":", "_")
        metrics[key] = metrics.get(key, 0.0) + value
    return metrics


class DynamoMetricsSampler:
    """Own polling thread, samples, and deterministic sampler shutdown."""

    def __init__(
        self,
        workers: Sequence[dict[str, Any] | DynamoWorkerEndpoint],
        *,
        interval_s: float,
        include_prefixes: list[str] | None,
        exclude_prefixes: list[str] | None,
    ) -> None:
        self._workers = tuple(
            DynamoWorkerEndpoint.from_metadata(worker)
            if isinstance(worker, dict)
            else worker
            for worker in workers
        )
        if not self._workers:
            raise ValueError("Managed Dynamo metrics require at least one worker")
        self._interval_s = interval_s
        self._include_prefixes = (
            tuple(include_prefixes)
            if include_prefixes is not None
            else CURATED_METRICS_INCLUDE_PREFIXES
        )
        self._exclude_prefixes = (
            tuple(exclude_prefixes)
            if exclude_prefixes is not None
            else DEFAULT_METRICS_EXCLUDE_PREFIXES
        )
        self._samples: dict[str, dict[int, list[float]]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Dynamo metrics sampler is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="dynamo-metrics-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        self._stop.wait(min(2.0, self._interval_s))
        while not self._stop.is_set():
            for ordinal, worker in enumerate(self._workers):
                text = _http_get_text(
                    f"{worker.system_url}/metrics",
                    timeout_s=self._interval_s + 2.0,
                )
                if self._stop.is_set():
                    break
                if not text:
                    continue
                metrics = parse_prometheus_metrics(
                    text,
                    self._include_prefixes,
                    self._exclude_prefixes,
                )
                with self._lock:
                    for name, value in metrics.items():
                        self._samples.setdefault(name, {}).setdefault(
                            ordinal, []
                        ).append(value)
            self._stop.wait(self._interval_s)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            metrics = {
                name: {
                    worker_id: list(samples)
                    for worker_id, samples in worker_metrics.items()
                }
                for name, worker_metrics in self._samples.items()
            }
        for alias, sources in CANONICAL_LOGGER_ALIASES.items():
            if alias in metrics:
                continue
            source = next((name for name in sources if name in metrics), None)
            metrics[alias] = dict(metrics[source]) if source is not None else {}
            if source is not None:
                del metrics[source]
        return metrics

    def clear(self) -> None:
        with self._lock:
            self._samples = {}

    def shutdown(self) -> None:
        self._stop.set()
        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=max(5.0, self._interval_s + 3.0))
            if thread.is_alive():
                LOGGER.warning("Dynamo metrics sampler did not stop before shutdown")
        self._thread = None
