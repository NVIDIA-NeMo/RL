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

"""Prometheus target discovery and RL-Insight registration for NeMo Gym."""

from __future__ import annotations

import importlib.metadata
import ipaddress
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
    model_validator,
)


RL_INSIGHT_SERVER_URL_ENV = "RL_INSIGHT_SERVER_URL"
NEMO_RL_RUN_ID_ENV = "NEMO_RL_RUN_ID"
PROMETHEUS_MANIFEST_FILENAME = "prometheus-targets.json"

_RESERVED_TARGET_LABELS = frozenset(
    {"component", "model", "replica", "routing_policy", "run_id"}
)
_PROMETHEUS_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*\Z")


class NemoGymPrometheusConfig(BaseModel, extra="forbid"):
    """Configure Prometheus target registration for a NeMo Gym run.

    ``server_url`` takes precedence over ``RL_INSIGHT_SERVER_URL``. A null
    ``run_id`` is resolved from ``NEMO_RL_RUN_ID``, ``SLURM_JOB_ID``, or the
    current Ray job ID, in that order.
    """

    enabled: bool = False
    required: bool = False
    server_url: str | None = None
    job_name: str = "nemo_rl_vllm"
    run_id: str | None = None
    request_timeout_s: PositiveFloat = 10.0
    readiness_timeout_s: PositiveFloat = 30.0
    scrape_interval_s: PositiveFloat = 10.0
    initial_scrape_wait_s: NonNegativeFloat = 12.0
    final_scrape_wait_s: NonNegativeFloat = 12.0
    target_lifecycle: Literal["dedicated", "shared"] = "shared"
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("server_url")
    @classmethod
    def _validate_server_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_http_base_url(value)

    @field_validator("job_name")
    @classmethod
    def _validate_job_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("job_name must not be empty")
        return value

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("run_id must not be empty when set")
        return value

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, value: dict[str, str]) -> dict[str, str]:
        reserved = _RESERVED_TARGET_LABELS.intersection(value)
        if reserved:
            raise ValueError(
                "labels must not override NeMo RL target labels: "
                + ", ".join(sorted(reserved))
            )
        normalized = {str(key): str(label_value) for key, label_value in value.items()}
        invalid = sorted(
            key
            for key in normalized
            if not _PROMETHEUS_LABEL_NAME.fullmatch(key) or key.startswith("__")
        )
        if invalid:
            raise ValueError(
                "labels contain invalid Prometheus label names: " + ", ".join(invalid)
            )
        return normalized

    @model_validator(mode="after")
    def _validate_required_mode(self) -> NemoGymPrometheusConfig:
        if self.required and not self.enabled:
            raise ValueError("required=true requires enabled=true")
        if self.required and self.target_lifecycle != "dedicated":
            raise ValueError(
                "required=true requires target_lifecycle=dedicated because the "
                "supported RL-Insight API has no target TTL or unregister operation"
            )
        if self.required and (
            self.initial_scrape_wait_s < self.scrape_interval_s
            or self.final_scrape_wait_s < self.scrape_interval_s
        ):
            raise ValueError(
                "required=true requires initial_scrape_wait_s and "
                "final_scrape_wait_s to cover at least one scrape_interval_s"
            )
        return self


@dataclass(frozen=True)
class PrometheusTarget:
    """One Prometheus scrape target and the labels registered with it."""

    address: str
    metrics_url: str
    labels: Mapping[str, str]


@dataclass(frozen=True)
class PrometheusTargetStatus:
    """Readiness observed for one Prometheus target during NeMo Gym spinup."""

    target: PrometheusTarget
    ready: bool
    error: str | None


@dataclass(frozen=True)
class PrometheusRegistrationResult:
    """Auditable result of one RL-Insight target registration attempt."""

    status: Literal["registered", "failed", "skipped"]
    server_url: str | None
    response: Mapping[str, Any] | None
    error: str | None


class PrometheusRegistrationError(RuntimeError):
    """Raised when RL-Insight did not confirm registration and reload."""


def _normalize_http_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"expected an http(s) server URL, got {value!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials must not be embedded in server_url")
    if parsed.query or parsed.fragment:
        raise ValueError("server_url must not contain a query string or fragment")
    return normalized


def resolve_run_id(
    config: NemoGymPrometheusConfig,
    *,
    ray_job_id: str | None,
) -> str:
    """Resolve one stable run ID for target labels and the run manifest."""
    if config.run_id is not None:
        return config.run_id
    for env_name in (NEMO_RL_RUN_ID_ENV, "SLURM_JOB_ID"):
        value = os.environ.get(env_name)
        if value:
            return value
    if ray_job_id:
        return f"ray-{ray_job_id}"
    raise ValueError(
        "Prometheus monitoring requires a run ID; set prometheus.run_id, "
        f"{NEMO_RL_RUN_ID_ENV}, or run inside a Ray job"
    )


def resolve_rl_insight_server_url(config: NemoGymPrometheusConfig) -> str:
    """Resolve and validate the RL-Insight server URL."""
    value = config.server_url or os.environ.get(RL_INSIGHT_SERVER_URL_ENV)
    if not value:
        raise PrometheusRegistrationError(
            "RL-Insight server URL is required; set prometheus.server_url or "
            f"{RL_INSIGHT_SERVER_URL_ENV}"
        )
    try:
        return _normalize_http_base_url(value)
    except ValueError as exc:
        raise PrometheusRegistrationError(str(exc)) from exc


def target_from_base_url(
    base_url: str,
    *,
    labels: Mapping[str, str],
) -> PrometheusTarget:
    """Build a Prometheus target from an HTTP service base URL."""
    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError(f"expected an HTTP service URL, got {base_url!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials must not be embedded in service URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid port in service URL {base_url!r}") from exc
    if port is None:
        port = 80
    if port == 0:
        raise ValueError(f"service URL must use a non-zero port, got {base_url!r}")

    host = parsed.hostname
    if host.lower() == "localhost" or host.lower().endswith(".localhost"):
        raise ValueError(f"Prometheus scrape target must not be loopback: {host}")
    try:
        address_value = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        mapped_address = getattr(address_value, "ipv4_mapped", None)
        if (
            address_value.is_loopback
            or address_value.is_unspecified
            or (
                mapped_address is not None
                and (mapped_address.is_loopback or mapped_address.is_unspecified)
            )
        ):
            raise ValueError(
                f"Prometheus scrape target must be remotely reachable, got {host}"
            )
    address_host = f"[{host}]" if ":" in host else host
    address = f"{address_host}:{port}"
    metrics_url = f"http://{address}/metrics"
    return PrometheusTarget(
        address=address,
        metrics_url=metrics_url,
        labels=dict(sorted((str(key), str(value)) for key, value in labels.items())),
    )


def wait_for_prometheus_target(
    target: PrometheusTarget,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> PrometheusTargetStatus:
    """Wait until a target's ``/metrics`` endpoint responds with HTTP 200."""
    deadline = time.monotonic() + timeout_s
    last_error = "metrics endpoint did not respond"
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 0:
            return PrometheusTargetStatus(
                target=target,
                ready=False,
                error=last_error,
            )
        request_timeout = max(min(poll_interval_s, remaining), 0.001)
        try:
            with urlopen(target.metrics_url, timeout=request_timeout) as response:
                if response.status == 200:
                    return PrometheusTargetStatus(
                        target=target,
                        ready=True,
                        error=None,
                    )
                last_error = f"HTTP {response.status} from {target.metrics_url}"
        except (URLError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return PrometheusTargetStatus(
                target=target,
                ready=False,
                error=last_error,
            )
        time.sleep(min(poll_interval_s, remaining))


def register_prometheus_targets(
    config: NemoGymPrometheusConfig,
    target_statuses: list[PrometheusTargetStatus],
) -> PrometheusRegistrationResult:
    """Register targets through RL-Insight's Prometheus target HTTP API."""
    server_url = resolve_rl_insight_server_url(config)
    targets = _deduplicate_targets([status.target for status in target_statuses])
    if not targets:
        raise PrometheusRegistrationError("no Prometheus targets were discovered")

    payload = {
        "job_name": config.job_name,
        "targets": [
            {"target": target.address, "labels": dict(target.labels)}
            for target in targets
        ],
    }
    endpoint = f"{server_url}/api/v1/prometheus/targets"
    request = Request(
        endpoint,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(config.request_timeout_s)) as response:
            body = response.read().decode("utf-8")
            status_code = response.status
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PrometheusRegistrationError(
            f"RL-Insight target registration returned HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise PrometheusRegistrationError(
            f"RL-Insight target registration failed at {endpoint}: {exc}"
        ) from exc

    if not 200 <= status_code < 300:
        raise PrometheusRegistrationError(
            f"RL-Insight target registration returned HTTP {status_code}"
        )
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise PrometheusRegistrationError(
            "RL-Insight target registration returned invalid JSON"
        ) from exc
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise PrometheusRegistrationError(
            f"RL-Insight did not confirm target registration: {result!r}"
        )
    if result.get("prometheus_reloaded") is not True:
        raise PrometheusRegistrationError(
            "RL-Insight registered targets but did not reload Prometheus"
        )
    return PrometheusRegistrationResult(
        status="registered",
        server_url=server_url,
        response=result,
        error=None,
    )


def failed_registration_result(
    config: NemoGymPrometheusConfig,
    error: PrometheusRegistrationError,
) -> PrometheusRegistrationResult:
    """Create the manifest representation of a failed registration."""
    try:
        server_url = resolve_rl_insight_server_url(config)
    except PrometheusRegistrationError:
        server_url = None
    return PrometheusRegistrationResult(
        status="failed",
        server_url=server_url,
        response=None,
        error=str(error),
    )


def write_prometheus_manifest(
    path: str | Path,
    *,
    config: NemoGymPrometheusConfig,
    run_id: str,
    target_statuses: list[PrometheusTargetStatus],
    registration: PrometheusRegistrationResult,
    router_log_paths: list[str],
    backend_log_paths: Mapping[str, str],
    model_call_capture_dir: str | None,
) -> None:
    """Atomically archive discovered targets and their registration result."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "job_name": config.job_name,
        "required": config.required,
        "monitoring_config": config.model_dump(mode="json"),
        "targets": [
            {
                "address": status.target.address,
                "metrics_url": status.target.metrics_url,
                "labels": dict(status.target.labels),
                "ready_at_registration": status.ready,
                "readiness_error": status.error,
            }
            for status in target_statuses
        ],
        "registration": {
            "status": registration.status,
            "server_url": registration.server_url,
            "response": registration.response,
            "error": registration.error,
        },
        "versions": collect_monitoring_versions(),
        "router_log_paths": router_log_paths,
        "backend_log_paths": dict(sorted(backend_log_paths.items())),
        "model_call_capture_dir": model_call_capture_dir,
    }
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def collect_monitoring_versions() -> dict[str, str]:
    """Collect the local components needed to reproduce monitoring behavior."""
    versions = {"python": platform.python_version()}
    for key, distribution in (
        ("nemo_rl", "nemo-rl"),
        ("nemo_gym", "nemo-gym"),
        ("vllm", "vllm"),
        ("vllm_router", "vllm-router"),
    ):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    try:
        uv_version = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        versions["uv"] = "unavailable"
    else:
        versions["uv"] = uv_version.removeprefix("uv ")
    return versions


def _deduplicate_targets(targets: list[PrometheusTarget]) -> list[PrometheusTarget]:
    by_address: dict[str, PrometheusTarget] = {}
    for target in targets:
        previous = by_address.get(target.address)
        if previous is not None and dict(previous.labels) != dict(target.labels):
            raise PrometheusRegistrationError(
                f"target {target.address} was discovered with conflicting labels"
            )
        by_address[target.address] = target
    return [by_address[address] for address in sorted(by_address)]
