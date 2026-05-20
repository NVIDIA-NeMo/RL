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
"""Export Dynamo Prometheus metrics as local replay artifacts."""

import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import requests
from prometheus_client.parser import text_string_to_metric_families

from nemo_rl.utils.dynamo_grafana import (
    DEFAULT_GRAFANA_DASHBOARD_FILENAME,
    build_grafana_dashboard,
)
from nemo_rl.utils.k8s import read_pod_namespace

if TYPE_CHECKING:
    from nemo_rl.utils.logger import Logger

DEFAULT_METRICS_PORT = 9090
DEFAULT_COLLECTION_INTERVAL = 10.0
DEFAULT_TIMEOUT = 5.0
DEFAULT_METRIC_PREFIX = "dynamo_prometheus"
DEFAULT_EXPORT_DIR_NAME = "dynamo_prometheus_export"
EXPORT_SAMPLES_FILENAME = "samples.jsonl"
EXPORT_RAW_SCRAPES_FILENAME = "raw_scrapes.jsonl"
EXPORT_OPENMETRICS_FILENAME = "data.openmetrics"
EXPORT_METADATA_FILENAME = "metadata.json"
EXPORT_README_FILENAME = "README.md"
EXPORT_PROMETHEUS_CONFIG_FILENAME = "prometheus-offline.yml"
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
    """Start local Dynamo Prometheus export when this run uses Dynamo."""
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

    export_enabled = _export_enabled(prometheus_cfg)
    if not export_enabled:
        print(
            "[Dynamo Prometheus] Local export is disabled; skipping collection."
        )
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
    """Poll Dynamo Prometheus endpoints and write local replay artifacts."""

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
        self.export_cfg = _get_export_cfg(prometheus_cfg)
        self.export_enabled = _export_enabled(prometheus_cfg)
        self.export_include_raw_scrapes = bool(
            self.export_cfg.get("include_raw_scrapes", True)
        )
        self.export_include_openmetrics = bool(
            self.export_cfg.get("include_openmetrics", True)
        )
        self.export_include_grafana_dashboard = bool(
            self.export_cfg.get("include_grafana_dashboard", True)
        )
        self.export_final_scrape = bool(self.export_cfg.get("final_scrape", True))
        self.export_dir: Optional[Path] = None
        self.export_samples_file: Any = None
        self.export_raw_scrapes_file: Any = None
        self.export_openmetrics_file: Any = None
        self.export_started_at: Optional[float] = None
        self.export_ended_at: Optional[float] = None
        self.export_lock = threading.Lock()
        self.exported_sample_count = 0
        self.exported_scrape_count = 0
        self.exported_openmetrics_sample_count = 0
        self.export_metric_names: set[str] = set()
        self.export_grafana_dashboard_written = False
        self.export_grafana_dashboard_panel_count = 0
        self.export_finalized = False
        self.endpoints = _resolve_metric_endpoints(dynamo_cfg, prometheus_cfg)
        self.previous_values: dict[tuple[str, tuple[tuple[str, str], ...], str], float] = {}
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

        if self.export_enabled:
            try:
                self._prepare_export()
            except Exception as exc:
                self.export_enabled = False
                print(
                    f"[Dynamo Prometheus] Local export setup failed: {exc}",
                    flush=True,
                )

        if not self.export_enabled:
            print("[Dynamo Prometheus] No local export available; skipping.")
            return

        self.start_time = time.time()
        self.is_running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
        )
        self.collection_thread.start()
        endpoint_summary = ", ".join(f"{name}={url}" for name, url in self.endpoints)
        print(
            "[Dynamo Prometheus] Exporting replay artifacts to "
            f"{self.export_dir} every {self.collection_interval}s from {endpoint_summary}",
            flush=True,
        )

    def stop(self) -> None:
        """Stop the background polling thread."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=self.collection_interval * 2)
        if self.export_enabled and self.export_final_scrape:
            try:
                self._collect_and_log()
            except Exception as exc:
                self._log_fetch_error(f"final export scrape failed: {exc}")
        export_dir = self._finalize_export()
        print("[Dynamo Prometheus] Collection stopped", flush=True)
        if export_dir is not None:
            print(
                "[Dynamo Prometheus] Exported replay artifacts to "
                f"{export_dir}",
                flush=True,
            )

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

        self._write_export_sample(metrics)

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

        self._write_raw_scrape(endpoint_name, endpoint_url, response.text, time.time())
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

    def _prepare_export(self) -> None:
        self.export_started_at = time.time()
        self.export_dir = Path(self._resolve_export_dir())
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._clear_stale_export_artifacts()
        self.export_samples_file = open(
            self.export_dir / EXPORT_SAMPLES_FILENAME, "w", encoding="utf-8"
        )
        if self.export_include_raw_scrapes:
            self.export_raw_scrapes_file = open(
                self.export_dir / EXPORT_RAW_SCRAPES_FILENAME, "w", encoding="utf-8"
            )
        if self.export_include_openmetrics:
            self.export_openmetrics_file = open(
                self.export_dir / EXPORT_OPENMETRICS_FILENAME, "w", encoding="utf-8"
            )
        self._write_export_readme()
        self._write_prometheus_offline_config()
        self._write_export_metadata()

    def _clear_stale_export_artifacts(self) -> None:
        if self.export_dir is None:
            return
        for filename in (
            EXPORT_SAMPLES_FILENAME,
            EXPORT_RAW_SCRAPES_FILENAME,
            EXPORT_OPENMETRICS_FILENAME,
            EXPORT_METADATA_FILENAME,
            EXPORT_README_FILENAME,
            EXPORT_PROMETHEUS_CONFIG_FILENAME,
            DEFAULT_GRAFANA_DASHBOARD_FILENAME,
        ):
            try:
                (self.export_dir / filename).unlink(missing_ok=True)
            except OSError as exc:
                print(
                    f"[Dynamo Prometheus] Could not remove stale {filename}: {exc}",
                    flush=True,
                )

    def _resolve_export_dir(self) -> str:
        configured_dir = (
            self.export_cfg.get("output_dir")
            or self.export_cfg.get("dir")
            or self.prometheus_cfg.get("export_dir")
        )
        base_log_dir = getattr(self.logger, "base_log_dir", None) or self.logger_config.get(
            "log_dir"
        )
        if configured_dir:
            configured_dir = str(configured_dir)
            if os.path.isabs(configured_dir):
                return configured_dir
            if base_log_dir:
                return os.path.join(str(base_log_dir), configured_dir)
            return configured_dir
        if base_log_dir:
            return os.path.join(str(base_log_dir), DEFAULT_EXPORT_DIR_NAME)
        return DEFAULT_EXPORT_DIR_NAME

    def _write_export_sample(self, metrics: Mapping[str, Any]) -> None:
        if not self.export_enabled or self.export_samples_file is None:
            return
        with self.export_lock:
            json.dump(metrics, self.export_samples_file, sort_keys=True)
            self.export_samples_file.write("\n")
            self.exported_sample_count += 1

    def _write_raw_scrape(
        self,
        endpoint_name: str,
        endpoint_url: str,
        metrics_text: str,
        scrape_time: float,
    ) -> None:
        if not self.export_enabled:
            return
        with self.export_lock:
            if self.export_raw_scrapes_file is not None:
                json.dump(
                    {
                        "endpoint_name": endpoint_name,
                        "endpoint_url": endpoint_url,
                        "scrape_time": scrape_time,
                        "scrape_time_iso": _format_unix_time(scrape_time),
                        "text": metrics_text,
                    },
                    self.export_raw_scrapes_file,
                    sort_keys=True,
                )
                self.export_raw_scrapes_file.write("\n")
            if self.export_openmetrics_file is not None:
                self.exported_openmetrics_sample_count += (
                    self._write_openmetrics_scrape(
                        endpoint_name, metrics_text, scrape_time
                    )
                )
            else:
                self.export_metric_names.update(
                    _extract_prometheus_text_metric_names(
                        metrics_text, self._should_include_sample
                    )
                )
            self.exported_scrape_count += 1

    def _write_openmetrics_scrape(
        self, endpoint_name: str, metrics_text: str, scrape_time: float
    ) -> int:
        assert self.export_openmetrics_file is not None
        sample_count = 0
        for family in text_string_to_metric_families(metrics_text):
            for sample in family.samples:
                sample_name = sample.name
                if not self._should_include_sample(sample_name):
                    continue
                value = sample.value
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                self.export_metric_names.add(sample_name)
                self.export_openmetrics_file.write(
                    _format_openmetrics_sample_line(
                        sample_name=sample_name,
                        labels={
                            **{str(k): str(v) for k, v in sample.labels.items()},
                            "nemo_rl_endpoint": endpoint_name,
                        },
                        value=float(value),
                        timestamp=scrape_time,
                    )
                )
                sample_count += 1
        return sample_count

    def _finalize_export(self) -> Optional[Path]:
        if not self.export_enabled or self.export_dir is None or self.export_finalized:
            return self.export_dir
        self.export_ended_at = time.time()
        with self.export_lock:
            if self.export_openmetrics_file is not None:
                self.export_openmetrics_file.write("# EOF\n")
            for file_obj in (
                self.export_samples_file,
                self.export_raw_scrapes_file,
                self.export_openmetrics_file,
            ):
                if file_obj is not None:
                    file_obj.flush()
                    file_obj.close()
            self.export_samples_file = None
            self.export_raw_scrapes_file = None
            self.export_openmetrics_file = None
            self.export_finalized = True
        self._write_grafana_dashboard()
        self._write_export_metadata()
        return self.export_dir

    def _write_export_metadata(self) -> None:
        if self.export_dir is None:
            return
        metadata = {
            "start_time_unix": self.export_started_at,
            "start_time_iso": _format_unix_time(self.export_started_at),
            "end_time_unix": self.export_ended_at,
            "end_time_iso": _format_unix_time(self.export_ended_at),
            "collection_interval": self.collection_interval,
            "timeout": self.timeout,
            "metric_prefixes": list(self.metric_prefixes),
            "include_histogram_buckets": self.include_histogram_buckets,
            "metric_names": sorted(self.export_metric_names),
            "endpoints": dict(self.endpoints),
            "files": {
                "samples_jsonl": EXPORT_SAMPLES_FILENAME,
                "raw_scrapes_jsonl": (
                    EXPORT_RAW_SCRAPES_FILENAME
                    if self.export_include_raw_scrapes
                    else None
                ),
                "openmetrics": (
                    EXPORT_OPENMETRICS_FILENAME
                    if self.export_include_openmetrics
                    else None
                ),
                "prometheus_offline_config": EXPORT_PROMETHEUS_CONFIG_FILENAME,
                "grafana_dashboard": (
                    DEFAULT_GRAFANA_DASHBOARD_FILENAME
                    if self.export_grafana_dashboard_written
                    else None
                ),
            },
            "counts": {
                "samples": self.exported_sample_count,
                "scrapes": self.exported_scrape_count,
                "openmetrics_samples": self.exported_openmetrics_sample_count,
                "grafana_dashboard_panels": self.export_grafana_dashboard_panel_count,
            },
        }
        with open(
            self.export_dir / EXPORT_METADATA_FILENAME, "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

    def _write_export_readme(self) -> None:
        if self.export_dir is None:
            return
        readme = f"""# Dynamo Prometheus Export

This directory contains Dynamo `/metrics` samples captured during the NeMo-RL run.

Files:
- `{EXPORT_SAMPLES_FILENAME}`: parsed scalar sample snapshots for offline inspection.
- `{EXPORT_RAW_SCRAPES_FILENAME}`: raw `/metrics` scrape text with scrape timestamps.
- `{EXPORT_OPENMETRICS_FILENAME}`: filtered OpenMetrics samples with explicit wall-clock timestamps.
- `{EXPORT_METADATA_FILENAME}`: endpoints, time range, and export counters.
- `{EXPORT_PROMETHEUS_CONFIG_FILENAME}`: minimal config for replaying generated TSDB blocks.
- `{DEFAULT_GRAFANA_DASHBOARD_FILENAME}`: importable Grafana dashboard, clipped to metrics that
  exist in this export.

To replay in Grafana, build Prometheus TSDB blocks from the OpenMetrics file:

```bash
promtool tsdb create-blocks-from openmetrics {EXPORT_OPENMETRICS_FILENAME} tsdb
prometheus \\
  --config.file={EXPORT_PROMETHEUS_CONFIG_FILENAME} \\
  --storage.tsdb.path=tsdb \\
  --web.listen-address=127.0.0.1:9092
```

Then point Grafana at `http://127.0.0.1:9092` and use an absolute time range
inside the exported start/end timestamps from `{EXPORT_METADATA_FILENAME}`. You can also import
`{DEFAULT_GRAFANA_DASHBOARD_FILENAME}`; its default time range is set to this export window.
The dashboard is derived from the Dynamo Grafana template and removes panels whose
PromQL metrics were not captured in `{EXPORT_OPENMETRICS_FILENAME}`.
"""
        with open(self.export_dir / EXPORT_README_FILENAME, "w", encoding="utf-8") as f:
            f.write(readme)

    def _write_prometheus_offline_config(self) -> None:
        if self.export_dir is None:
            return
        config = (
            "global:\n"
            "  scrape_interval: 15s\n"
            "  evaluation_interval: 15s\n"
            "\n"
            "scrape_configs: []\n"
        )
        with open(
            self.export_dir / EXPORT_PROMETHEUS_CONFIG_FILENAME, "w", encoding="utf-8"
        ) as f:
            f.write(config)

    def _write_grafana_dashboard(self) -> None:
        if self.export_dir is None or not self.export_include_grafana_dashboard:
            return
        try:
            dashboard = build_grafana_dashboard(
                start_time_iso=_format_unix_time(self.export_started_at),
                end_time_iso=_format_unix_time(self.export_ended_at),
                available_metric_names=(
                    self.export_metric_names if self.export_finalized else None
                ),
            )
        except Exception as exc:
            self.export_grafana_dashboard_written = False
            self.export_grafana_dashboard_panel_count = 0
            print(
                f"[Dynamo Prometheus] Grafana dashboard export failed: {exc}",
                flush=True,
            )
            return
        self.export_grafana_dashboard_panel_count = len(dashboard.get("panels", []))
        with open(
            self.export_dir / DEFAULT_GRAFANA_DASHBOARD_FILENAME,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dashboard, f, indent=2, sort_keys=True)
            f.write("\n")
        self.export_grafana_dashboard_written = True


def _get_prometheus_cfg(dynamo_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    cfg = dynamo_cfg.get("prometheus_metrics", None)
    if cfg is None:
        cfg = dynamo_cfg.get("metrics", None)
    return cfg or {}


def _get_export_cfg(prometheus_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    cfg = prometheus_cfg.get("export")
    if isinstance(cfg, Mapping):
        return cfg
    return {}


def _export_enabled(prometheus_cfg: Mapping[str, Any]) -> bool:
    export_cfg = _get_export_cfg(prometheus_cfg)
    if "enabled" in export_cfg:
        return bool(export_cfg["enabled"])
    return bool(prometheus_cfg.get("export_on_exit", False))


def _extract_prometheus_text_metric_names(
    metrics_text: str,
    should_include_sample: Any,
) -> set[str]:
    names = set()
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            sample_name = sample.name
            if should_include_sample(sample_name):
                names.add(sample_name)
    return names


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


def _format_unix_time(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _format_openmetrics_sample_line(
    sample_name: str,
    labels: Mapping[str, str],
    value: float,
    timestamp: float,
) -> str:
    label_text = ""
    if labels:
        label_parts = [
            f'{key}="{_escape_openmetrics_label_value(str(label_value))}"'
            for key, label_value in sorted(labels.items())
        ]
        label_text = "{" + ",".join(label_parts) + "}"
    return f"{sample_name}{label_text} {value:.17g} {timestamp:.6f}\n"


def _escape_openmetrics_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
