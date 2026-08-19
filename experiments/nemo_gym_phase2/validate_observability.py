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

"""Exercise the real RL-Insight target API and Prometheus scrape data path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener, install_opener, urlopen

import yaml
import rl_insight

from nemo_rl.environments.prometheus import (
    NemoGymPrometheusConfig,
    PrometheusRegistrationResult,
    PrometheusTarget,
    PrometheusTargetStatus,
    register_prometheus_targets,
    target_from_base_url,
    wait_for_prometheus_target,
)


READY_TIMEOUT_SECONDS = 45.0
POLL_INTERVAL_SECONDS = 0.25


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus-bin", type=Path, required=True)
    parser.add_argument("--prometheus-base-config", type=Path, required=True)
    parser.add_argument("--stack-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("0.0.0.0", 0))
        return int(listener.getsockname()[1])


def _advertise_host() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("198.51.100.1", 9))
            host = str(probe.getsockname()[0])
        except OSError:
            host = socket.gethostbyname(socket.gethostname())
    if host.startswith("127.") or host == "0.0.0.0":
        raise RuntimeError(f"could not resolve a remotely reachable node IP: {host}")
    return host


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_server_config(
    path: Path,
    *,
    stack_root: Path,
    prometheus_bin: Path,
    prometheus_base_config: Path,
    server_port: int,
    prometheus_port: int,
) -> None:
    payload = {
        "server": {
            "port": server_port,
            "install_dir": str(stack_root / "services"),
            "runtime_dir": str(stack_root / "runtime"),
            "data_dir": str(stack_root / "data"),
            "state_file": str(stack_root / "run" / "services.json"),
        },
        "prometheus": {
            "binary_path": str(prometheus_bin),
            "prometheus_port": prometheus_port,
            "retention_time": "1d",
            "config_file": str(prometheus_base_config),
        },
        "tempo": {"enable": False},
        "grafana": {"enable": False},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _json_request(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object from {url}, got {type(payload).__name__}")
    return payload


def _wait_for_json(url: str, *, ready: Any) -> dict[str, Any]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = _json_request(url)
            if ready(payload):
                return payload
        except (OSError, ValueError, TypeError) as exc:
            last_error = exc
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"{url} was not ready: {last_error}")


def _metrics_handler(body: str) -> type[BaseHTTPRequestHandler]:
    encoded = body.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/metrics":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


@contextmanager
def _metrics_server(body: str) -> Iterator[int]:
    server = ThreadingHTTPServer(("0.0.0.0", 0), _metrics_handler(body))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _prometheus_query(prometheus_url: str, promql: str) -> dict[str, Any]:
    query = urlencode({"query": promql})
    payload = _json_request(f"{prometheus_url}/api/v1/query?{query}")
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload!r}")
    return payload


def _matching_active_targets(
    payload: Mapping[str, Any], run_id: str
) -> list[dict[str, Any]]:
    data = payload.get("data")
    active = data.get("activeTargets") if isinstance(data, dict) else None
    if not isinstance(active, list):
        return []
    return [
        target
        for target in active
        if isinstance(target, dict)
        and isinstance(target.get("labels"), dict)
        and target["labels"].get("run_id") == run_id
        and target.get("health") == "up"
    ]


def _observed_probe_values(payload: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    data = payload.get("data")
    results = data.get("result") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise RuntimeError(f"Prometheus returned an invalid vector: {payload!r}")
    observed: dict[tuple[str, str], float] = {}
    for sample in results:
        metric = sample.get("metric") if isinstance(sample, dict) else None
        value = sample.get("value") if isinstance(sample, dict) else None
        if (
            not isinstance(metric, dict)
            or not isinstance(value, list)
            or len(value) != 2
        ):
            raise RuntimeError(f"Prometheus returned an invalid sample: {sample!r}")
        key = (str(metric.get("component")), str(metric.get("replica")))
        if key in observed:
            raise RuntimeError(f"duplicate probe series for labels {key!r}")
        observed[key] = float(value[1])
    return observed


def _registration_payload(result: PrometheusRegistrationResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "server_url": result.server_url,
        "response": result.response,
        "error": result.error,
    }


def _target_payload(status: PrometheusTargetStatus) -> dict[str, Any]:
    return {
        "address": status.target.address,
        "metrics_url": status.target.metrics_url,
        "labels": dict(status.target.labels),
        "ready": status.ready,
        "error": status.error,
    }


def _run_validation(args: argparse.Namespace) -> dict[str, Any]:
    prometheus_source_bin = args.prometheus_bin.expanduser().resolve(strict=True)
    prometheus_base_config = args.prometheus_base_config.expanduser().resolve(
        strict=True
    )
    stack_root = args.stack_root.expanduser().resolve()
    if stack_root.exists():
        raise FileExistsError(f"refusing to reuse validation stack root: {stack_root}")
    stack_root.mkdir(parents=True)

    prometheus_bin = stack_root / "services" / "prometheus" / "prometheus"
    prometheus_bin.parent.mkdir(parents=True)
    shutil.copy2(prometheus_source_bin, prometheus_bin)
    prometheus_bin.chmod(prometheus_bin.stat().st_mode | 0o111)
    prometheus_source_sha256 = _sha256(prometheus_source_bin)
    prometheus_staged_sha256 = _sha256(prometheus_bin)
    if prometheus_staged_sha256 != prometheus_source_sha256:
        raise RuntimeError("node-local Prometheus copy differs from the source binary")

    server_port = _free_port()
    prometheus_port = _free_port()
    while prometheus_port == server_port:
        prometheus_port = _free_port()
    config_path = stack_root / "config.yaml"
    _write_server_config(
        config_path,
        stack_root=stack_root,
        prometheus_bin=prometheus_bin,
        prometheus_base_config=prometheus_base_config,
        server_port=server_port,
        prometheus_port=prometheus_port,
    )

    host = _advertise_host()
    # Use loopback for the local validation control plane, while the scrape
    # targets below deliberately use the node's non-loopback address. Some
    # developer hosts do not reliably hairpin their externally advertised IP.
    server_url = f"http://127.0.0.1:{server_port}"
    prometheus_url = f"http://127.0.0.1:{prometheus_port}"
    bypass_proxy = build_opener(ProxyHandler({}))
    install_opener(bypass_proxy)
    run_id = f"phase2-observability-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    job_name = f"nemo_rl_phase2_validation_{os.getpid()}"
    command = [
        sys.executable,
        "-m",
        "rl_insight.cli",
        "server",
        "start",
        "--config",
        str(config_path),
        "--detach",
    ]
    start = subprocess.run(command, check=False, capture_output=True, text=True)
    if start.returncode != 0:
        raise RuntimeError(
            "RL-Insight start failed with exit code "
            f"{start.returncode}:\n{start.stdout}\n{start.stderr}"
        )

    stop_command = [
        sys.executable,
        "-m",
        "rl_insight.cli",
        "server",
        "stop",
        "--config",
        str(config_path),
    ]
    try:
        health = _wait_for_json(
            f"{server_url}/healthz",
            ready=lambda payload: payload == {"status": "ok"},
        )
        services = _wait_for_json(
            f"{server_url}/api/v1/services",
            ready=lambda payload: payload.get("prometheus_port") == prometheus_port,
        )
        _wait_for_json(
            f"{prometheus_url}/api/v1/status/runtimeinfo",
            ready=lambda payload: payload.get("status") == "success",
        )

        exporter_specs = [
            (
                "vllm_backend",
                "backend-0",
                11.0,
                "# TYPE phase2_observability_probe gauge\n"
                "phase2_observability_probe 11\n"
                "# TYPE vllm:prefix_cache_hits_total counter\n"
                "vllm:prefix_cache_hits_total 8\n"
                "# TYPE vllm:prefix_cache_queries_total counter\n"
                "vllm:prefix_cache_queries_total 10\n",
            ),
            (
                "vllm_backend",
                "backend-1",
                22.0,
                "# TYPE phase2_observability_probe gauge\n"
                "phase2_observability_probe 22\n"
                "# TYPE vllm:prefix_cache_hits_total counter\n"
                "vllm:prefix_cache_hits_total 15\n"
                "# TYPE vllm:prefix_cache_queries_total counter\n"
                "vllm:prefix_cache_queries_total 20\n",
            ),
            (
                "vllm_router",
                "router-0",
                33.0,
                "# TYPE phase2_observability_probe gauge\n"
                "phase2_observability_probe 33\n"
                "# TYPE vllm_router_cache_hits_total counter\n"
                "vllm_router_cache_hits_total 3\n"
                "# TYPE vllm_router_cache_misses_total counter\n"
                "vllm_router_cache_misses_total 1\n",
            ),
        ]
        expected_values = {
            (component, replica): value
            for component, replica, value, _ in exporter_specs
        }
        with ExitStack() as exporters:
            targets: list[PrometheusTarget] = []
            for component, replica, _, metrics in exporter_specs:
                port = exporters.enter_context(_metrics_server(metrics))
                targets.append(
                    target_from_base_url(
                        f"http://{host}:{port}/v1",
                        labels={
                            "component": component,
                            "model": "Qwen/Qwen2.5-1.5B-Instruct",
                            "replica": replica,
                            "routing_policy": "cache_aware",
                            "run_id": run_id,
                        },
                    )
                )
            statuses = [
                wait_for_prometheus_target(
                    target,
                    timeout_s=5,
                    poll_interval_s=POLL_INTERVAL_SECONDS,
                )
                for target in targets
            ]
            if not all(status.ready for status in statuses):
                raise RuntimeError(
                    "one or more validation exporters were unreachable: "
                    f"{[_target_payload(status) for status in statuses]!r}"
                )

            config = NemoGymPrometheusConfig(
                enabled=True,
                required=True,
                server_url=server_url,
                job_name=job_name,
                run_id=run_id,
                scrape_interval_s=1,
                initial_scrape_wait_s=1,
                final_scrape_wait_s=1,
                target_lifecycle="dedicated",
            )
            registration = register_prometheus_targets(config, statuses)
            if registration.response is None or registration.response.get(
                "target_count"
            ) != len(targets):
                raise RuntimeError(
                    f"unexpected registration response: {registration.response!r}"
                )

            targets_url = f"{prometheus_url}/api/v1/targets?state=active"
            active_targets = _wait_for_json(
                targets_url,
                ready=lambda payload: len(_matching_active_targets(payload, run_id))
                == len(targets),
            )
            query = f'phase2_observability_probe{{run_id="{run_id}"}}'
            samples = _wait_for_json(
                f"{prometheus_url}/api/v1/query?{urlencode({'query': query})}",
                ready=lambda payload: len(
                    payload.get("data", {}).get("result", [])
                    if isinstance(payload.get("data"), dict)
                    else []
                )
                == len(targets),
            )
            observed_values = _observed_probe_values(samples)
            if observed_values != expected_values:
                raise RuntimeError(
                    f"scraped probe values differ: {observed_values!r} != "
                    f"{expected_values!r}"
                )

            backend_hits = _prometheus_query(
                prometheus_url,
                f'vllm:prefix_cache_hits_total{{run_id="{run_id}"}}',
            )
            router_hits = _prometheus_query(
                prometheus_url,
                f'vllm_router_cache_hits_total{{run_id="{run_id}"}}',
            )
            backend_results = backend_hits.get("data", {}).get("result", [])
            router_results = router_hits.get("data", {}).get("result", [])
            if len(backend_results) != 2 or len(router_results) != 1:
                raise RuntimeError(
                    "cache metric series were not scraped per component: "
                    f"backend={len(backend_results)}, router={len(router_results)}"
                )
            matched_targets = _matching_active_targets(active_targets, run_id)
            required_label_sets = {
                (
                    str(target["labels"].get("component")),
                    str(target["labels"].get("replica")),
                    str(target["labels"].get("routing_policy")),
                    str(target["labels"].get("model")),
                    str(target["labels"].get("run_id")),
                )
                for target in matched_targets
            }
            expected_label_sets = {
                (
                    component,
                    replica,
                    "cache_aware",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    run_id,
                )
                for component, replica, _, _ in exporter_specs
            }
            if required_label_sets != expected_label_sets:
                raise RuntimeError(
                    f"scrape target labels differ: {required_label_sets!r} != "
                    f"{expected_label_sets!r}"
                )

        return {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "run_id": run_id,
            "scrape_advertise_host": host,
            "server_url": server_url,
            "prometheus_url": prometheus_url,
            "versions": {
                "python": sys.version.split()[0],
                "rl_insight": rl_insight.__version__,
                "prometheus": subprocess.run(
                    [str(prometheus_bin), "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()[0],
            },
            "prometheus_binary": {
                "source": str(prometheus_source_bin),
                "staged": str(prometheus_bin),
                "sha256": prometheus_staged_sha256,
            },
            "health": health,
            "services": services,
            "registration": _registration_payload(registration),
            "targets": [_target_payload(status) for status in statuses],
            "active_targets": matched_targets,
            "probe_values": {
                f"{component}/{replica}": value
                for (component, replica), value in sorted(observed_values.items())
            },
            "backend_cache_metric": backend_hits,
            "router_cache_metric": router_hits,
            "start_command": command,
            "start_stdout": start.stdout,
            "start_stderr": start.stderr,
        }
    finally:
        stop = subprocess.run(stop_command, check=False, capture_output=True, text=True)
        if stop.returncode != 0:
            print(
                "RL-Insight stop failed:\n" + stop.stdout + "\n" + stop.stderr,
                file=sys.stderr,
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _run_validation(args)
    _atomic_write_json(args.output.expanduser().resolve(), payload)
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
