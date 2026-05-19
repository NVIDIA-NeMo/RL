# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Forward Dynamo Prometheus metrics into W&B."""

import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

import requests
from prometheus_client.parser import text_string_to_metric_families

from nemo_rl.utils.k8s import read_pod_namespace

if TYPE_CHECKING:
    from nemo_rl.utils.logger import Logger

DEFAULT_METRICS_PORT = 9090
DEFAULT_COLLECTION_INTERVAL = 10.0
DEFAULT_TIMEOUT = 5.0
DEFAULT_METRIC_PREFIX = "dynamo_prometheus"
DEFAULT_SERVICE_NAMES = ("vllmdecodeworker",)
DEFAULT_METRIC_PREFIXES = ("dynamo_",)
DEFAULT_LABEL_KEYS = (
    "dynamo_component",
    "dynamo_endpoint",
    "model",
    "model_name",
    "worker_id",
)
COUNTER_SUFFIXES = ("_total", "_count", "_sum")

_SAFE_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_.:-]+")


def maybe_start_dynamo_prometheus_monitor(
    master_config: Mapping[str, Any],
    logger: "Logger",
) -> Optional["DynamoPrometheusMonitor"]:
    """Start Dynamo Prometheus forwarding when this run uses Dynamo + W&B.

    By default the monitor writes into the active W&B run. Set
    ``policy.generation.dynamo_cfg.prometheus_metrics.wandb_run: separate`` to
    write Dynamo telemetry into a dedicated W&B run grouped with the training
    run.
    """
    generation_config = (
        master_config.get("policy", {}).get("generation", {})  # type: ignore[union-attr]
        or {}
    )
    if generation_config.get("backend") != "dynamo":
        return None

    dynamo_cfg = generation_config.get("dynamo_cfg", {}) or {}
    prometheus_cfg = _get_prometheus_cfg(dynamo_cfg)
    if not prometheus_cfg.get("enabled", True):
        print("[Dynamo Prometheus] Disabled by policy.generation.dynamo_cfg.")
        return None

    if getattr(logger, "wandb_logger", None) is None:
        print("[Dynamo Prometheus] W&B logger is not enabled; skipping forwarding.")
        return None

    monitor = DynamoPrometheusMonitor(
        logger=logger,
        dynamo_cfg=dynamo_cfg,
        prometheus_cfg=prometheus_cfg,
        logger_config=master_config.get("logger", {}) or {},
    )
    monitor.start()
    return monitor


class DynamoPrometheusMonitor:
    """Poll Dynamo Prometheus endpoints and log scalar samples to W&B."""

    def __init__(
        self,
        logger: "Logger",
        dynamo_cfg: Mapping[str, Any],
        prometheus_cfg: Mapping[str, Any],
        logger_config: Mapping[str, Any] | None = None,
    ):
        self.logger = logger
        self.dynamo_cfg = dynamo_cfg
        self.prometheus_cfg = prometheus_cfg
        self.logger_config = logger_config or {}
        self.collection_interval = float(
            prometheus_cfg.get("collection_interval", DEFAULT_COLLECTION_INTERVAL)
        )
        self.timeout = float(prometheus_cfg.get("timeout", DEFAULT_TIMEOUT))
        self.metric_prefix = str(
            prometheus_cfg.get("metric_prefix", DEFAULT_METRIC_PREFIX)
        )
        self.elapsed_time_metric = f"{self.metric_prefix}/elapsed_seconds"
        self.wall_time_metric = f"{self.metric_prefix}/wall_time"
        self.metric_prefixes = tuple(
            str(prefix)
            for prefix in prometheus_cfg.get(
                "metric_prefixes", DEFAULT_METRIC_PREFIXES
            )
        )
        self.label_keys = tuple(
            str(label_key)
            for label_key in prometheus_cfg.get("label_keys", DEFAULT_LABEL_KEYS)
        )
        self.include_histogram_buckets = bool(
            prometheus_cfg.get("include_histogram_buckets", False)
        )
        self.log_counter_deltas = bool(prometheus_cfg.get("log_counter_deltas", True))
        self.endpoints = _resolve_metric_endpoints(dynamo_cfg, prometheus_cfg)
        self.previous_values: dict[tuple[str, tuple[tuple[str, str], ...], str], float] = {}
        self.samples: list[dict[str, Any]] = []
        self.samples_lock = threading.Lock()
        self.flushed_sample_count = 0
        self.log_live = bool(prometheus_cfg.get("log_live", False))
        self.separate_wandb_run = _uses_separate_wandb_run(prometheus_cfg)
        self.wandb_run: Any = None
        self.owns_wandb_run = False
        self.is_running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.start_time = float("-inf")
        self.last_error_log_time = 0.0

    def start(self) -> None:
        """Start the background polling thread."""
        if self.is_running:
            return

        if not self.endpoints:
            print("[Dynamo Prometheus] No metrics endpoints resolved; skipping.")
            return

        self.wandb_run = self._resolve_wandb_run()
        if self.wandb_run is None:
            print("[Dynamo Prometheus] No W&B run available; skipping forwarding.")
            return

        self._define_wandb_metric(self.elapsed_time_metric)
        self._define_wandb_metric(self.wall_time_metric)
        for endpoint_name, _ in self.endpoints:
            endpoint_prefix = (
                f"{self.metric_prefix}/{_sanitize_metric_part(endpoint_name)}/*"
            )
            self._define_wandb_metric(
                endpoint_prefix, step_metric=self.elapsed_time_metric
            )

        self.start_time = time.time()
        self.is_running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
        )
        self.collection_thread.start()
        endpoint_summary = ", ".join(f"{name}={url}" for name, url in self.endpoints)
        print(
            "[Dynamo Prometheus] Collecting metrics for W&B time series "
            f"every {self.collection_interval}s from {endpoint_summary}",
            flush=True,
        )

    def stop(self) -> None:
        """Stop the background polling thread."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=self.collection_interval * 2)
        flushed = self._flush_samples_to_wandb()
        print(
            f"[Dynamo Prometheus] Forwarding stopped; flushed {flushed} samples",
            flush=True,
        )
        if self.owns_wandb_run and self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
            self.owns_wandb_run = False

    def _collection_loop(self) -> None:
        while self.is_running:
            try:
                self._collect_and_log()
            except Exception as exc:
                self._log_fetch_error(f"unexpected collection error: {exc}")
            time.sleep(self.collection_interval)

    def _collect_and_log(self) -> None:
        now = time.time()
        elapsed_seconds = max(0.0, now - self.start_time)
        metrics: dict[str, Any] = {
            self.elapsed_time_metric: elapsed_seconds,
            self.wall_time_metric: now,
        }

        for endpoint_name, endpoint_url in self.endpoints:
            endpoint_metrics = self._fetch_endpoint_metrics(endpoint_name, endpoint_url)
            for name, value in endpoint_metrics.items():
                metrics[f"{self.metric_prefix}/{name}"] = value

        if len(metrics) == 2:
            return

        with self.samples_lock:
            self.samples.append(metrics)

        if self.log_live:
            self._log_wandb_sample(metrics)
            with self.samples_lock:
                self.flushed_sample_count = len(self.samples)

    def _flush_samples_to_wandb(self) -> int:
        with self.samples_lock:
            samples = self.samples[self.flushed_sample_count :]

        for sample in samples:
            self._log_wandb_sample(sample)

        with self.samples_lock:
            self.flushed_sample_count = len(self.samples)
        return len(samples)

    def _log_wandb_sample(self, sample: Mapping[str, Any]) -> None:
        # Use W&B directly here instead of Logger.log_metrics(step=...):
        # Prometheus samples are wall-clock telemetry, not training-step data.
        # Logging them after collection keeps the training step monotonic while
        # still giving the Dynamo panels a time-based x-axis.
        if self.wandb_run is None:
            self.wandb_run = self._resolve_wandb_run()
        if self.wandb_run is not None:
            self.wandb_run.log(dict(sample))

    def _resolve_wandb_run(self) -> Any:
        if self.separate_wandb_run:
            return self._start_separate_wandb_run()
        return self.logger.wandb_logger.run

    def _define_wandb_metric(
        self, name: str, step_metric: Optional[str] = None
    ) -> None:
        if self.owns_wandb_run:
            self.wandb_run.define_metric(name, step_metric=step_metric)
        else:
            self.logger.wandb_logger.define_metric(name, step_metric=step_metric)

    def _start_separate_wandb_run(self) -> Any:
        import wandb

        main_run = self.logger.wandb_logger.run
        wandb_cfg = dict(self.logger_config.get("wandb", {}) or {})
        main_run_id = getattr(main_run, "id", None)
        main_run_name = (
            wandb_cfg.get("name")
            or getattr(main_run, "name", None)
            or main_run_id
            or "training"
        )
        group = (
            self.prometheus_cfg.get("wandb_group")
            or wandb_cfg.get("group")
            or getattr(main_run, "group", None)
            or main_run_id
        )
        run_name = self.prometheus_cfg.get(
            "wandb_name", f"{main_run_name}-dynamo-prometheus"
        )
        init_kwargs = {
            "project": self.prometheus_cfg.get("wandb_project")
            or wandb_cfg.get("project")
            or getattr(main_run, "project", None),
            "entity": self.prometheus_cfg.get("wandb_entity")
            or wandb_cfg.get("entity")
            or getattr(main_run, "entity", None),
            "name": run_name,
            "group": group,
            "job_type": self.prometheus_cfg.get(
                "wandb_job_type", "dynamo-prometheus"
            ),
            "reinit": "create_new",
            "resume": "never",
            "config": {
                "source_wandb_run_id": main_run_id,
                "source_wandb_run_name": main_run_name,
                "metric_prefix": self.metric_prefix,
                "metric_prefixes": list(self.metric_prefixes),
                "endpoints": dict(self.endpoints),
            },
        }
        init_kwargs = {
            key: value for key, value in init_kwargs.items() if value is not None
        }
        run = wandb.init(**init_kwargs)
        self.owns_wandb_run = True
        print(
            "[Dynamo Prometheus] Logging telemetry to separate W&B run "
            f"{run_name} in group {group}",
            flush=True,
        )
        return run

    def _fetch_endpoint_metrics(
        self, endpoint_name: str, endpoint_url: str
    ) -> dict[str, float]:
        try:
            response = requests.get(endpoint_url, timeout=self.timeout)
            if response.status_code != 200:
                self._log_fetch_error(
                    f"{endpoint_name} returned HTTP {response.status_code}"
                )
                return {}
        except Exception as exc:
            self._log_fetch_error(f"{endpoint_name} fetch failed: {exc}")
            return {}

        return self._parse_prometheus_text(endpoint_name, response.text)

    def _parse_prometheus_text(
        self, endpoint_name: str, metrics_text: str
    ) -> dict[str, float]:
        parsed: dict[str, float] = {}
        delta_parts: dict[tuple[str, tuple[tuple[str, str], ...], str], dict[str, float]] = {}

        for family in text_string_to_metric_families(metrics_text):
            for sample in family.samples:
                sample_name = sample.name
                if not self._should_include_sample(sample_name):
                    continue

                value = sample.value
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue

                label_items = tuple(sorted((str(k), str(v)) for k, v in sample.labels.items()))
                formatted_name = self._format_sample_name(
                    endpoint_name, sample_name, sample.labels
                )
                parsed[formatted_name] = float(value)

                if not self.log_counter_deltas:
                    continue

                counter_suffix = _counter_suffix(sample_name)
                if counter_suffix is None:
                    continue

                previous_key = (sample_name, label_items, endpoint_name)
                previous_value = self.previous_values.get(previous_key)
                self.previous_values[previous_key] = float(value)
                if previous_value is None:
                    continue

                delta = float(value) - previous_value
                if delta < 0:
                    # Counter reset or worker restarted.
                    continue

                parsed[f"{formatted_name}_delta"] = delta

                base_name = sample_name[: -len(counter_suffix)]
                delta_key = (base_name, label_items, endpoint_name)
                part_name = counter_suffix[1:]
                delta_parts.setdefault(delta_key, {})[part_name] = delta

        for (base_name, label_items, endpoint_name), parts in delta_parts.items():
            count_delta = parts.get("count")
            sum_delta = parts.get("sum")
            if count_delta is None or sum_delta is None or count_delta <= 0:
                continue
            labels = dict(label_items)
            mean_name = self._format_sample_name(
                endpoint_name, f"{base_name}_mean_seconds", labels
            )
            parsed[mean_name] = sum_delta / count_delta

        return parsed

    def _should_include_sample(self, sample_name: str) -> bool:
        if not self.include_histogram_buckets and sample_name.endswith("_bucket"):
            return False
        if sample_name.endswith("_created"):
            return False
        return sample_name.startswith(self.metric_prefixes)

    def _format_sample_name(
        self, endpoint_name: str, sample_name: str, labels: Mapping[str, str]
    ) -> str:
        parts = [_sanitize_metric_part(endpoint_name), _sanitize_metric_part(sample_name)]
        for label_key in self.label_keys:
            label_value = labels.get(label_key)
            if label_value:
                parts.append(
                    f"{_sanitize_metric_part(label_key)}.{_sanitize_metric_part(label_value)}"
                )
        if self.include_histogram_buckets and "le" in labels:
            parts.append(f"le.{_sanitize_metric_part(labels['le'])}")
        return "/".join(parts)

    def _log_fetch_error(self, message: str) -> None:
        now = time.time()
        if now - self.last_error_log_time < 60:
            return
        self.last_error_log_time = now
        print(f"[Dynamo Prometheus] {message}", flush=True)


def _get_prometheus_cfg(dynamo_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    cfg = dynamo_cfg.get("prometheus_metrics", None)
    if cfg is None:
        cfg = dynamo_cfg.get("metrics", None)
    return cfg or {}


def _uses_separate_wandb_run(prometheus_cfg: Mapping[str, Any]) -> bool:
    if bool(prometheus_cfg.get("separate_wandb_run", False)):
        return True
    wandb_run = str(prometheus_cfg.get("wandb_run", "same")).lower()
    return wandb_run in {"separate", "dedicated", "new"}


def _resolve_metric_endpoints(
    dynamo_cfg: Mapping[str, Any],
    prometheus_cfg: Mapping[str, Any],
) -> list[tuple[str, str]]:
    explicit_endpoints = prometheus_cfg.get("endpoints")
    if explicit_endpoints:
        return _normalize_explicit_endpoints(explicit_endpoints)

    explicit_url = prometheus_cfg.get("url")
    if explicit_url:
        return [("dynamo", str(explicit_url))]

    dgd_name = dynamo_cfg.get("dgd_name")
    if not dgd_name:
        return []

    namespace = (
        prometheus_cfg.get("namespace")
        or dynamo_cfg.get("namespace")
        or read_pod_namespace()
        or "default"
    )
    port = int(prometheus_cfg.get("port", DEFAULT_METRICS_PORT))
    service_names = prometheus_cfg.get("service_names", DEFAULT_SERVICE_NAMES)
    if isinstance(service_names, str):
        service_names = [service_names]

    endpoints = []
    for service_name in service_names:
        service_name = str(service_name).lower()
        url = (
            f"http://{dgd_name}-{service_name}.{namespace}.svc.cluster.local:"
            f"{port}/metrics"
        )
        endpoints.append((service_name, url))
    return endpoints


def _normalize_explicit_endpoints(endpoints: Any) -> list[tuple[str, str]]:
    if isinstance(endpoints, Mapping):
        return [(str(name), str(url)) for name, url in endpoints.items()]

    if isinstance(endpoints, str):
        return [("dynamo", endpoints)]

    if isinstance(endpoints, Sequence):
        normalized = []
        for index, endpoint in enumerate(endpoints):
            if isinstance(endpoint, Mapping):
                name = endpoint.get("name", f"endpoint_{index}")
                url = endpoint.get("url")
                if not url:
                    continue
                normalized.append((str(name), str(url)))
            else:
                normalized.append((f"endpoint_{index}", str(endpoint)))
        return normalized

    return []


def _counter_suffix(sample_name: str) -> Optional[str]:
    for suffix in COUNTER_SUFFIXES:
        if sample_name.endswith(suffix):
            return suffix
    return None


def _sanitize_metric_part(value: str) -> str:
    sanitized = _SAFE_METRIC_CHARS.sub("_", value.strip())
    return sanitized.strip("_") or "unknown"
