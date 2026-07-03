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

"""Driver-owned lifecycle for a fixed Ray-managed Dynamo deployment."""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_frontend_argv,
    redact_argv,
    redact_environment,
    validate_managed_vllm_config,
)
from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo.worker_pool import FixedDynamoWorkerPool


def _managed_namespace(configured: str | None) -> str:
    raw = configured or f"nemo-rl-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    namespace = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw)).strip("-_").lower()
    if not namespace:
        raise ValueError(f"Could not derive a valid Dynamo namespace from {raw!r}.")
    return namespace


class ManagedDynamoRuntime:
    """Own etcd, NATS, frontend, and a fixed Ray actor worker fleet."""

    def __init__(
        self,
        *,
        cluster: RayVirtualCluster,
        config: dict[str, Any],
        dynamo_cfg: DynamoCfg,
    ) -> None:
        self._cluster = cluster
        self._config = config
        self._dynamo_cfg = dynamo_cfg
        self._namespace = _managed_namespace(dynamo_cfg.namespace)
        self._host = _get_node_ip_local()
        self._etcd_process: subprocess.Popen | None = None
        self._nats_process: subprocess.Popen | None = None
        self._frontend_process: subprocess.Popen | None = None
        self._etcd_data_dir: str | None = None
        self._nats_data_dir: str | None = None
        self._pool: FixedDynamoWorkerPool | None = None

        vllm_cfg = dict(config.get("vllm_cfg") or {})
        validate_managed_vllm_config(vllm_cfg)
        engine_world_size = int(dynamo_cfg.engine_world_size)
        configured_world_size = int(vllm_cfg.get("tensor_parallel_size", 1)) * int(
            vllm_cfg.get("pipeline_parallel_size", 1)
        )
        if configured_world_size != engine_world_size:
            raise ValueError(
                "dynamo_cfg.engine_world_size must equal "
                "vllm_cfg.tensor_parallel_size * vllm_cfg.pipeline_parallel_size: "
                f"{engine_world_size} != {configured_world_size}."
            )

        used_ports: set[int] = set()

        def allocate_port(configured: int) -> int:
            if configured:
                if configured in used_ports:
                    raise ValueError(
                        f"Managed Dynamo service port {configured} is configured more than once."
                    )
                used_ports.add(configured)
                return configured
            for _ in range(100):
                candidate = _get_free_port_local()
                if candidate not in used_ports:
                    used_ports.add(candidate)
                    return candidate
            raise RuntimeError(
                "Could not allocate a distinct managed Dynamo service port."
            )

        self._etcd_port = allocate_port(dynamo_cfg.etcd_port)
        self._etcd_peer_port = allocate_port(dynamo_cfg.etcd_peer_port)
        self._nats_port = allocate_port(dynamo_cfg.nats_port)
        self._frontend_port = allocate_port(dynamo_cfg.frontend_port)

        self._manager_env = {
            "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
            "NATS_SERVER": f"nats://{self._host}:{self._nats_port}",
            "DYN_NAMESPACE": self._namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
        }
        self._start()

    @property
    def frontend_url(self) -> str:
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{host}:{self._frontend_port}/v1"

    @property
    def namespace(self) -> str:
        return self._namespace

    def _start(self) -> None:
        print(
            f"  [Dynamo] managed environment={redact_environment(self._manager_env)!r}",
            flush=True,
        )
        try:
            self._start_etcd()
            self._start_nats()
            self._pool = FixedDynamoWorkerPool(
                cluster=self._cluster,
                config=self._config,
                namespace=self._namespace,
                engine_world_size=self._dynamo_cfg.engine_world_size,
                system_port_base=self._dynamo_cfg.system_port_base,
                manager_env=self._manager_env,
                startup_timeout_s=self._dynamo_cfg.startup_timeout_s,
            )
            self._pool.start()
            self._start_frontend()
            self._wait_for_frontend(self._pool.size)
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _stop_process(
        process: subprocess.Popen | None, label: str, timeout_s: float = 15
    ) -> None:
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        print(f"  [Dynamo] {label} stopped pid={process.pid}", flush=True)

    def _service_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("DYN_")
        }
        env.update(self._manager_env)
        env["ALLOW_NONE_AUTHENTICATION"] = "yes"
        return env

    def _start_etcd(self) -> None:
        self._etcd_data_dir = tempfile.mkdtemp(prefix="nemorl_dynamo_etcd_")
        peer_url = f"http://{self._host}:{self._etcd_peer_port}"
        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{self._etcd_port}",
            "--advertise-client-urls",
            f"http://{self._host}:{self._etcd_port}",
            "--listen-peer-urls",
            f"http://0.0.0.0:{self._etcd_peer_port}",
            "--initial-advertise-peer-urls",
            peer_url,
            "--initial-cluster",
            f"default={peer_url}",
            "--data-dir",
            self._etcd_data_dir,
        ]
        self._etcd_process = subprocess.Popen(
            command, env=self._service_env(), start_new_session=True
        )
        self._wait_for_etcd()
        print(f"  [Dynamo] etcd ready on {self._host}:{self._etcd_port}", flush=True)

    def _wait_for_etcd(self) -> None:
        url = f"http://127.0.0.1:{self._etcd_port}/health"
        deadline = time.monotonic() + min(self._dynamo_cfg.startup_timeout_s, 60)
        while time.monotonic() < deadline:
            if self._etcd_process is not None and self._etcd_process.poll() is not None:
                raise RuntimeError(
                    f"etcd exited with code {self._etcd_process.returncode}."
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        raise RuntimeError(f"etcd did not become healthy at {url}.")

    def _start_nats(self) -> None:
        self._nats_data_dir = tempfile.mkdtemp(prefix="nemorl_dynamo_nats_")
        self._nats_process = subprocess.Popen(
            [
                "nats-server",
                "-js",
                "-sd",
                self._nats_data_dir,
                "-p",
                str(self._nats_port),
            ],
            env=self._service_env(),
            start_new_session=True,
        )
        self._wait_for_port(self._nats_port, "NATS", self._nats_process)
        print(f"  [Dynamo] NATS ready on {self._host}:{self._nats_port}", flush=True)

    def _wait_for_port(
        self, port: int, label: str, process: subprocess.Popen | None = None
    ) -> None:
        deadline = time.monotonic() + min(self._dynamo_cfg.startup_timeout_s, 60)
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"{label} exited with code {process.returncode}.")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"{label} did not open port {port}.")

    def _start_frontend(self) -> None:
        argv = build_dynamo_frontend_argv(
            host="0.0.0.0",
            port=self._frontend_port,
            namespace=self._namespace,
            dynamo_cfg=self._dynamo_cfg,
        )
        command = [
            self._dynamo_cfg.dynamo_python,
            "-m",
            "dynamo.frontend",
            *argv,
        ]
        print(
            f"  [Dynamo] launching frontend argv={redact_argv(command)!r}", flush=True
        )
        self._frontend_process = subprocess.Popen(
            command, env=self._service_env(), start_new_session=True
        )

    def _wait_for_frontend(self, expected_workers: int) -> None:
        url = f"http://127.0.0.1:{self._frontend_port}/health"
        deadline = time.monotonic() + self._dynamo_cfg.startup_timeout_s
        last_counts = (0, 0)
        while time.monotonic() < deadline:
            if (
                self._frontend_process is not None
                and self._frontend_process.poll() is not None
            ):
                raise RuntimeError(
                    f"Dynamo frontend exited with code {self._frontend_process.returncode}."
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    payload = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(1)
                continue
            generate_ids: set[str] = set()
            rl_ids: set[str] = set()
            for instance in payload.get("instances", []):
                if not isinstance(instance, dict):
                    continue
                if instance.get("namespace") != self._namespace:
                    continue
                if instance.get("component") != "backend":
                    continue
                instance_id = instance.get("instance_id")
                if instance_id is None:
                    continue
                if instance.get("endpoint") == "generate":
                    generate_ids.add(str(instance_id))
                elif instance.get("endpoint") == "rl":
                    rl_ids.add(str(instance_id))
            last_counts = (len(generate_ids), len(rl_ids))
            if (
                last_counts == (expected_workers, expected_workers)
                and generate_ids == rl_ids
            ):
                print(
                    f"  [Dynamo] frontend ready with {expected_workers} generation "
                    "and RL workers",
                    flush=True,
                )
                return
            time.sleep(1)
        raise RuntimeError(
            "Dynamo frontend did not observe the fixed worker fleet within "
            f"{self._dynamo_cfg.startup_timeout_s}s: expected={expected_workers}, "
            f"last_generate={last_counts[0]}, last_rl={last_counts[1]}."
        )

    def refit_workers(self) -> list[dict[str, Any]]:
        self._assert_services_alive()
        if self._pool is None:
            raise RuntimeError("Managed Dynamo worker pool is not running.")
        return self._pool.refit_workers()

    def validate_workers(self, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._assert_services_alive()
        if self._pool is None:
            raise RuntimeError("Managed Dynamo worker pool is not running.")
        return self._pool.validate(expected)

    def _assert_services_alive(self) -> None:
        for label, process in (
            ("etcd", self._etcd_process),
            ("NATS", self._nats_process),
            ("frontend", self._frontend_process),
        ):
            if process is None or process.poll() is not None:
                code = None if process is None else process.returncode
                raise RuntimeError(
                    f"Managed Dynamo {label} is not alive (code={code})."
                )

    def shutdown(self) -> None:
        self._stop_process(self._frontend_process, "frontend")
        self._frontend_process = None
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self._stop_process(self._nats_process, "NATS")
        self._nats_process = None
        self._stop_process(self._etcd_process, "etcd")
        self._etcd_process = None
        if self._etcd_data_dir is not None:
            shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
            self._etcd_data_dir = None
        if self._nats_data_dir is not None:
            shutil.rmtree(self._nats_data_dir, ignore_errors=True)
            self._nats_data_dir = None
