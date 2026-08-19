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

import math
import re
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Annotated, Literal, Mapping
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import BaseModel, Field, model_validator
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from nemo_rl.environments.prometheus import PrometheusTarget, target_from_base_url


_CACHE_MATCH_PREFIX = "Cache match for model"
_CACHE_MATCH_RE = re.compile(
    r"Cache match for model .*?: matched_chars=(?P<matched>[0-9]+), "
    r"input_chars=(?P<input>[0-9]+), match_rate="
)
_CACHE_HITS_SAMPLE_RE = re.compile(
    r"^vllm_router_cache_hits_total(?:\{|[ \t])", re.MULTILINE
)
_CACHE_MISSES_SAMPLE_RE = re.compile(
    r"^vllm_router_cache_misses_total(?:\{|[ \t])", re.MULTILINE
)
_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_AGGREGATE_COUNTER_METRICS = (
    "vllm_router_requests_total",
    "vllm_router_request_errors_total",
    "vllm_router_retries_total",
    "vllm_router_retries_exhausted_total",
    "vllm_router_load_balancing_events_total",
)
_WORKER_COUNTER_METRICS = (
    "vllm_router_policy_decisions_total",
    "vllm_router_processed_requests_total",
    "vllm_router_cb_state_transitions_total",
)
_WORKER_GAUGE_METRICS = ("vllm_router_cb_state",)
_CB_OUTCOMES = ("success", "failure")


@dataclass(frozen=True)
class _RouterCacheMetrics:
    hits: int
    misses: int
    malformed_lines: int

    @property
    def observations(self) -> int:
        return self.hits + self.misses


def _float32(value: float) -> float:
    """Round a Python float to the Router's IEEE-754 f32 representation."""
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _parse_router_cache_metrics(
    log_path: Path,
    *,
    cache_threshold: float,
) -> _RouterCacheMetrics:
    """Reconstruct vLLM Router 0.1.15 cache decisions from DEBUG logs."""
    hits = 0
    misses = 0
    malformed_lines = 0
    threshold = _float32(cache_threshold)
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            # Ignore a trailing partial write. The next scrape will see the complete line.
            if not line.endswith("\n") or _CACHE_MATCH_PREFIX not in line:
                continue
            match = _CACHE_MATCH_RE.search(line)
            if match is None:
                malformed_lines += 1
                continue
            matched_chars = int(match.group("matched"))
            input_chars = int(match.group("input"))
            if matched_chars > input_chars or (input_chars == 0 and matched_chars != 0):
                malformed_lines += 1
                continue
            match_rate = (
                _float32(_float32(float(matched_chars)) / _float32(float(input_chars)))
                if input_chars > 0
                else 0.0
            )
            if match_rate > threshold:
                hits += 1
            else:
                misses += 1
    return _RouterCacheMetrics(
        hits=hits,
        misses=misses,
        malformed_lines=malformed_lines,
    )


def _native_samples_by_name(native_metrics: str) -> dict[str, list[Sample]]:
    """Parse native Router exposition into samples keyed by metric name."""
    samples: dict[str, list[Sample]] = {}
    for family in text_string_to_metric_families(native_metrics):
        for sample in family.samples:
            samples.setdefault(sample.name, []).append(sample)
    return samples


def _metric_total(
    samples_by_name: Mapping[str, list[Sample]],
    metric_name: str,
    *,
    labels: Mapping[str, str] | None = None,
) -> float:
    required_labels = labels or {}
    return sum(
        float(sample.value)
        for sample in samples_by_name.get(metric_name, [])
        if all(
            sample.labels.get(name) == value for name, value in required_labels.items()
        )
    )


def _metric_by_worker(
    samples_by_name: Mapping[str, list[Sample]],
    metric_name: str,
    worker: str,
    *,
    default: float,
    reducer: Literal["sum", "max"],
) -> float:
    values = [
        float(sample.value)
        for sample in samples_by_name.get(metric_name, [])
        if sample.labels.get("worker") == worker
    ]
    if not values:
        return default
    return sum(values) if reducer == "sum" else max(values)


def _escape_prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _prometheus_sample(
    metric_name: str,
    value: float,
    *,
    labels: Mapping[str, str] | None = None,
) -> str:
    rendered_labels = ""
    if labels:
        rendered_labels = (
            "{"
            + ",".join(
                f'{name}="{_escape_prometheus_label(label)}"'
                for name, label in sorted(labels.items())
            )
            + "}"
        )
    if math.isnan(value):
        rendered_value = "NaN"
    elif value.is_integer():
        rendered_value = str(int(value))
    else:
        rendered_value = repr(value)
    return f"{metric_name}{rendered_labels} {rendered_value}\n"


def _probe_worker_health(worker_base_url: str, *, timeout_s: float) -> float:
    """Probe the same backend health endpoint used by Router 0.1.15."""
    try:
        with urlopen(f"{worker_base_url}/health", timeout=timeout_s) as response:
            return 1.0 if response.status == HTTPStatus.OK else 0.0
    except (OSError, TimeoutError, URLError):
        return 0.0


def _render_operational_compatibility_metrics(
    native_metrics: str,
    *,
    worker_base_urls: tuple[str, ...],
    policy: str,
    health_probe_timeout_s: float,
) -> str:
    """Stabilize Router 0.1.15 lazy metrics without hiding native absence."""
    samples_by_name = _native_samples_by_name(native_metrics)
    observed_metric_names = {
        *_AGGREGATE_COUNTER_METRICS,
        *_WORKER_COUNTER_METRICS,
        *_WORKER_GAUGE_METRICS,
        "vllm_router_active_workers",
        "vllm_router_cache_hits_total",
        "vllm_router_cache_misses_total",
        "vllm_router_cb_outcomes_total",
        "vllm_router_running_requests",
        "vllm_router_tree_size",
        "vllm_router_worker_health",
        "vllm_router_worker_load",
    }
    lines = [
        "# HELP nemo_rl_vllm_router_metrics_adapter_info Owned Router metrics adapter provenance.\n",
        "# TYPE nemo_rl_vllm_router_metrics_adapter_info gauge\n",
        _prometheus_sample(
            "nemo_rl_vllm_router_metrics_adapter_info",
            1.0,
            labels={"policy": policy, "source": "native_aggregate_compat"},
        ),
        "# HELP nemo_rl_vllm_router_native_metric_present Whether the native Router exporter currently exposes a metric family.\n",
        "# TYPE nemo_rl_vllm_router_native_metric_present gauge\n",
    ]
    for metric_name in sorted(observed_metric_names):
        lines.append(
            _prometheus_sample(
                "nemo_rl_vllm_router_native_metric_present",
                float(bool(samples_by_name.get(metric_name))),
                labels={"metric": metric_name},
            )
        )

    for native_name in _AGGREGATE_COUNTER_METRICS:
        compatibility_name = f"nemo_rl_{native_name}"
        lines.extend(
            [
                f"# HELP {compatibility_name} Stable aggregate of native {native_name}.\n",
                f"# TYPE {compatibility_name} counter\n",
                _prometheus_sample(
                    compatibility_name,
                    _metric_total(samples_by_name, native_name),
                ),
            ]
        )

    for native_name in _WORKER_COUNTER_METRICS:
        compatibility_name = f"nemo_rl_{native_name}"
        lines.extend(
            [
                f"# HELP {compatibility_name} Stable per-worker aggregate of native {native_name}.\n",
                f"# TYPE {compatibility_name} counter\n",
            ]
        )
        for worker in worker_base_urls:
            lines.append(
                _prometheus_sample(
                    compatibility_name,
                    _metric_by_worker(
                        samples_by_name,
                        native_name,
                        worker,
                        default=0.0,
                        reducer="sum",
                    ),
                    labels={"worker": worker},
                )
            )

    cb_outcomes_name = "nemo_rl_vllm_router_cb_outcomes_total"
    lines.extend(
        [
            f"# HELP {cb_outcomes_name} Stable per-worker aggregate of native circuit-breaker outcomes.\n",
            f"# TYPE {cb_outcomes_name} counter\n",
        ]
    )
    for worker in worker_base_urls:
        for outcome in _CB_OUTCOMES:
            lines.append(
                _prometheus_sample(
                    cb_outcomes_name,
                    _metric_total(
                        samples_by_name,
                        "vllm_router_cb_outcomes_total",
                        labels={"worker": worker, "outcome": outcome},
                    ),
                    labels={"worker": worker, "outcome": outcome},
                )
            )

    for native_name in _WORKER_GAUGE_METRICS:
        compatibility_name = f"nemo_rl_{native_name}"
        lines.extend(
            [
                f"# HELP {compatibility_name} Stable per-worker view of native {native_name}.\n",
                f"# TYPE {compatibility_name} gauge\n",
            ]
        )
        for worker in worker_base_urls:
            lines.append(
                _prometheus_sample(
                    compatibility_name,
                    _metric_by_worker(
                        samples_by_name,
                        native_name,
                        worker,
                        default=0.0,
                        reducer="max",
                    ),
                    labels={"worker": worker},
                )
            )

    expected_workers = set(worker_base_urls)
    native_health_workers = {
        str(sample.labels["worker"])
        for sample in samples_by_name.get("vllm_router_worker_health", [])
        if "worker" in sample.labels
    }
    native_health_available = bool(expected_workers) and (
        native_health_workers == expected_workers
    )
    if worker_base_urls:
        with ThreadPoolExecutor(
            max_workers=len(worker_base_urls),
            thread_name_prefix="vllm-router-health-probe",
        ) as executor:
            probed_health = dict(
                zip(
                    worker_base_urls,
                    executor.map(
                        lambda worker: _probe_worker_health(
                            worker,
                            timeout_s=health_probe_timeout_s,
                        ),
                        worker_base_urls,
                    ),
                    strict=True,
                )
            )
    else:
        probed_health = {}
    health_source = (
        "native_and_adapter_probe"
        if native_health_available
        else "partial_native_and_adapter_probe"
        if native_health_workers
        else "adapter_backend_health_probe"
    )
    lines.extend(
        [
            "# HELP nemo_rl_vllm_router_worker_health Router worker health with explicit compatibility provenance.\n",
            "# TYPE nemo_rl_vllm_router_worker_health gauge\n",
        ]
    )
    for worker in worker_base_urls:
        native_health = _metric_by_worker(
            samples_by_name,
            "vllm_router_worker_health",
            worker,
            default=1.0,
            reducer="max",
        )
        health = min(native_health, probed_health[worker])
        lines.append(
            _prometheus_sample(
                "nemo_rl_vllm_router_worker_health",
                health,
                labels={"worker": worker},
            )
        )
    lines.extend(
        [
            "# HELP nemo_rl_vllm_router_worker_health_source_info Source used for worker health.\n",
            "# TYPE nemo_rl_vllm_router_worker_health_source_info gauge\n",
            _prometheus_sample(
                "nemo_rl_vllm_router_worker_health_source_info",
                1.0,
                labels={"source": health_source},
            ),
        ]
    )
    return "".join(lines)


@dataclass(frozen=True)
class _RouterMetricsProxy:
    """Proxy native Router metrics and stabilize missing 0.1.15 observations."""

    native_metrics_url: str
    router_stdout_log_path: Path
    cache_threshold: float
    worker_base_urls: tuple[str, ...]
    policy: str
    cache_metrics_mode: Literal["native", "debug_log_compat"]
    request_timeout_s: float = 2.0

    def render(self) -> bytes:
        with urlopen(
            self.native_metrics_url, timeout=self.request_timeout_s
        ) as response:
            if response.status != HTTPStatus.OK:
                raise RuntimeError(
                    f"native Router metrics returned HTTP {response.status}"
                )
            native_metrics = response.read().decode("utf-8")

        has_native_hits = _CACHE_HITS_SAMPLE_RE.search(native_metrics) is not None
        has_native_misses = _CACHE_MISSES_SAMPLE_RE.search(native_metrics) is not None
        if has_native_hits != has_native_misses:
            raise RuntimeError(
                "native Router exporter exposed only one cache counter; refusing "
                "to publish an ambiguous metric pair"
            )
        cache_metrics = ""
        if has_native_hits:
            cache_metrics = self._provenance_metrics(source="native")
        elif self.cache_metrics_mode == "debug_log_compat":
            cache = _parse_router_cache_metrics(
                self.router_stdout_log_path,
                cache_threshold=self.cache_threshold,
            )
            if cache.malformed_lines:
                raise RuntimeError(
                    "failed to parse "
                    f"{cache.malformed_lines} Router cache decision log line(s)"
                )
            cache_metrics = (
                "# HELP vllm_router_cache_hits_total Cache-aware routing decisions "
                "whose prefix match exceeded the configured threshold.\n"
                "# TYPE vllm_router_cache_hits_total counter\n"
                f"vllm_router_cache_hits_total {cache.hits}\n"
                "# HELP vllm_router_cache_misses_total Cache-aware routing decisions "
                "whose prefix match did not exceed the configured threshold.\n"
                "# TYPE vllm_router_cache_misses_total counter\n"
                f"vllm_router_cache_misses_total {cache.misses}\n"
                "# HELP nemo_rl_vllm_router_cache_log_observations_total Router "
                "cache decisions reconstructed from archived DEBUG logs.\n"
                "# TYPE nemo_rl_vllm_router_cache_log_observations_total counter\n"
                "nemo_rl_vllm_router_cache_log_observations_total "
                f"{cache.observations}\n"
                + self._provenance_metrics(source="debug_log_compat")
            )
        operational_metrics = _render_operational_compatibility_metrics(
            native_metrics,
            worker_base_urls=self.worker_base_urls,
            policy=self.policy,
            health_probe_timeout_s=self.request_timeout_s,
        )
        extra_metrics = operational_metrics + cache_metrics
        return _append_prometheus_metrics(native_metrics, extra_metrics).encode("utf-8")

    def _provenance_metrics(self, *, source: str) -> str:
        return (
            "# HELP nemo_rl_vllm_router_cache_metrics_info Source used for Router "
            "cache hit and miss counters.\n"
            "# TYPE nemo_rl_vllm_router_cache_metrics_info gauge\n"
            "nemo_rl_vllm_router_cache_metrics_info"
            f'{{source="{source}"}} 1\n'
            "# HELP nemo_rl_vllm_router_cache_threshold Prefix-match threshold used "
            "by cache-aware routing.\n"
            "# TYPE nemo_rl_vllm_router_cache_threshold gauge\n"
            "nemo_rl_vllm_router_cache_threshold "
            f"{self.cache_threshold}\n"
        )


def _append_prometheus_metrics(native_metrics: str, extra_metrics: str) -> str:
    """Append metrics before an optional OpenMetrics EOF marker."""
    normalized = native_metrics.rstrip("\n")
    if normalized.endswith("# EOF"):
        normalized = normalized.removesuffix("# EOF").rstrip("\n")
        return f"{normalized}\n{extra_metrics}# EOF\n"
    return f"{normalized}\n{extra_metrics}"


def _metrics_handler(
    proxy: _RouterMetricsProxy,
) -> type[BaseHTTPRequestHandler]:
    class RouterMetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.partition("?")[0] != "/metrics":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = proxy.render()
            except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
                payload = f"Router metrics compatibility proxy failed: {exc}\n".encode()
                self.send_response(HTTPStatus.BAD_GATEWAY)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", _PROMETHEUS_CONTENT_TYPE)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return RouterMetricsHandler


class VllmRouterConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    policy: Literal[
        "random",
        "round_robin",
        "cache_aware",
        "power_of_two",
        "consistent_hash",
        "rendezvous_hash",
    ] = "consistent_hash"
    cache_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    cache_metrics_mode: Literal["native", "debug_log_compat"] = "native"

    @model_validator(mode="after")
    def _validate_cache_metrics_mode(self) -> "VllmRouterConfig":
        if (
            self.cache_metrics_mode == "debug_log_compat"
            and self.policy != "cache_aware"
        ):
            raise ValueError(
                "cache_metrics_mode=debug_log_compat requires policy=cache_aware"
            )
        return self


class VllmRouterProcess:
    def __init__(
        self,
        *,
        worker_base_urls: list[str],
        host: str,
        port: int,
        prometheus_port: int,
        native_prometheus_port: int | None = None,
        config: VllmRouterConfig,
        log_dir: str | Path,
    ) -> None:
        self.worker_base_urls = [
            base_url.rstrip("/").removesuffix("/v1") for base_url in worker_base_urls
        ]
        self.host = host
        self.port = port
        self.prometheus_port = prometheus_port
        self.native_prometheus_port = native_prometheus_port
        self.config = config
        self.log_dir = Path(log_dir)
        self.stdout_log_path = self.log_dir / "router.stdout.log"
        self.stderr_log_path = self.log_dir / "router.stderr.log"
        self._process: subprocess.Popen | None = None
        self._stdout_log: IO[str] | None = None
        self._stderr_log: IO[str] | None = None
        self._metrics_server: ThreadingHTTPServer | None = None
        self._metrics_thread: threading.Thread | None = None
        if self.cache_metrics_compatibility_enabled:
            if native_prometheus_port is None:
                raise ValueError(
                    "native_prometheus_port is required for debug_log_compat mode"
                )
            if native_prometheus_port == prometheus_port:
                raise ValueError(
                    "native_prometheus_port and prometheus_port must be distinct"
                )
        elif native_prometheus_port == prometheus_port:
            raise ValueError(
                "native_prometheus_port and prometheus_port must be distinct"
            )

    @property
    def cache_metrics_compatibility_enabled(self) -> bool:
        return self.config.cache_metrics_mode == "debug_log_compat"

    @property
    def metrics_proxy_enabled(self) -> bool:
        return self.native_prometheus_port is not None

    @property
    def router_prometheus_port(self) -> int:
        return self.native_prometheus_port or self.prometheus_port

    @property
    def command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "vllm_router.launch_router",
            "--worker-urls",
            *self.worker_base_urls,
            "--policy",
            self.config.policy,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--prometheus-port",
            str(self.router_prometheus_port),
            "--prometheus-host",
            "127.0.0.1" if self.metrics_proxy_enabled else self.host,
        ]
        if self.config.policy == "cache_aware":
            command.extend(["--cache-threshold", str(self.config.cache_threshold)])
        if self.cache_metrics_compatibility_enabled:
            command.extend(["--log-level", "debug"])
        return command

    @property
    def openai_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def readiness_url(self) -> str:
        return f"http://{self.host}:{self.port}/readiness"

    @property
    def metrics_url(self) -> str:
        return f"http://{self.host}:{self.prometheus_port}/metrics"

    @property
    def native_metrics_url(self) -> str:
        return f"http://127.0.0.1:{self.router_prometheus_port}/metrics"

    @property
    def log_paths(self) -> list[str]:
        return [str(self.stdout_log_path), str(self.stderr_log_path)]

    def prometheus_target(self, *, labels: Mapping[str, str]) -> PrometheusTarget:
        """Return the remotely reachable Router metrics target."""
        return target_from_base_url(
            f"http://{self.host}:{self.prometheus_port}",
            labels=labels,
        )

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("vLLM Router process has already been started")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout_log = self.stdout_log_path.open("w", encoding="utf-8", buffering=1)
        self._stderr_log = self.stderr_log_path.open("w", encoding="utf-8", buffering=1)
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=self._stdout_log,
                stderr=self._stderr_log,
            )
            if self.metrics_proxy_enabled:
                proxy = _RouterMetricsProxy(
                    native_metrics_url=self.native_metrics_url,
                    router_stdout_log_path=self.stdout_log_path,
                    cache_threshold=self.config.cache_threshold,
                    worker_base_urls=tuple(self.worker_base_urls),
                    policy=self.config.policy,
                    cache_metrics_mode=self.config.cache_metrics_mode,
                )
                self._metrics_server = ThreadingHTTPServer(
                    (self.host, self.prometheus_port),
                    _metrics_handler(proxy),
                )
                self._metrics_server.daemon_threads = True
                self._metrics_thread = threading.Thread(
                    target=self._metrics_server.serve_forever,
                    kwargs={"poll_interval": 0.1},
                    name="vllm-router-metrics-proxy",
                    daemon=True,
                )
                self._metrics_thread.start()
        except (OSError, RuntimeError):
            self.stop()
            raise

    def wait_until_ready(
        self,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        """Wait for the Router request-serving endpoint."""
        self._wait_until_url_ready(
            self.readiness_url,
            endpoint_name="request endpoint",
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def wait_until_metrics_ready(
        self,
        timeout: float,
        poll_interval: float = 1.0,
    ) -> None:
        """Wait for the Router Prometheus endpoint independently of serving."""
        self._wait_until_url_ready(
            self.metrics_url,
            endpoint_name="Prometheus endpoint",
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def _wait_until_url_ready(
        self,
        url: str,
        *,
        endpoint_name: str,
        timeout: float,
        poll_interval: float,
    ) -> None:
        process = self._process
        if process is None:
            raise RuntimeError("vLLM Router process has not been started")

        deadline = time.monotonic() + timeout
        while True:
            return_code = process.poll()
            if return_code is not None:
                self._process = None
                self._stop_metrics_proxy()
                self._close_logs()
                raise RuntimeError(
                    f"vLLM Router process exited with code {return_code} "
                    f"before its {endpoint_name} became ready"
                )

            try:
                with urlopen(
                    url,
                    timeout=poll_interval,
                ) as response:
                    if response.status == 200:
                        return
            except (URLError, TimeoutError):
                pass

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"vLLM Router {endpoint_name} did not become ready within "
                    f"{timeout} seconds"
                )

            time.sleep(poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_metrics_proxy()
        process = self._process
        if process is None:
            self._close_logs()
            return

        self._process = None
        if process.poll() is not None:
            self._close_logs()
            return

        try:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            self._close_logs()

    def _stop_metrics_proxy(self) -> None:
        server = self._metrics_server
        thread = self._metrics_thread
        self._metrics_server = None
        self._metrics_thread = None
        if server is None:
            return
        if thread is not None and thread.is_alive():
            server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def _close_logs(self) -> None:
        if self._stdout_log is not None:
            self._stdout_log.close()
            self._stdout_log = None
        if self._stderr_log is not None:
            self._stderr_log.close()
            self._stderr_log = None
