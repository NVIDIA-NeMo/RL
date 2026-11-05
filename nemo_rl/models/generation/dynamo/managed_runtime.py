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

import atexit
import json
import logging
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
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_frontend_argv,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.models.generation.dynamo.venv import (
    get_dynamo_executable,
    get_dynamo_python,
)
from nemo_rl.models.generation.dynamo.worker_pool import FixedDynamoWorkerPool

LOGGER = logging.getLogger(__name__)


def _managed_namespace() -> str:
    raw = f"nemo-rl-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
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
    ) -> None:
        validated_config = DynamoConfig.model_validate(config)
        self._cluster = cluster
        self._config = validated_config.model_dump()
        self._dynamo_cfg = validated_config.dynamo_cfg
        self._engine_world_size = validated_config.engine_world_size
        if self._engine_world_size > cluster.num_gpus_per_node:
            raise ValueError(
                "Managed Dynamo requires each TP/PP engine group to fit on one "
                f"node: tp*pp={self._engine_world_size} exceeds "
                f"cluster.num_gpus_per_node={cluster.num_gpus_per_node}"
            )
        self._namespace = _managed_namespace()
        self._host = ""
        self._etcd_port = 0
        self._etcd_peer_port = 0
        self._nats_port = 0
        self._frontend_port = 0
        self._manager_env: dict[str, str] = {}
        self._started = False
        self._atexit_registered = False
        self._etcd_process: subprocess.Popen | None = None
        self._nats_process: subprocess.Popen | None = None
        self._frontend_process: subprocess.Popen | None = None
        self._etcd_data_dir: str | None = None
        self._nats_data_dir: str | None = None
        self._pool: FixedDynamoWorkerPool | None = None

    @property
    def frontend_url(self) -> str:
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{host}:{self._frontend_port}/v1"

    def start(self) -> None:
        """Start the complete managed service fleet."""
        if self._started:
            raise RuntimeError("Managed Dynamo runtime is already started")
        # Managed services use new process groups and would survive an
        # exception that escapes GRPO setup. Keep an interpreter-exit fallback
        # in addition to the normal explicit shutdown path.
        atexit.register(self.shutdown)
        self._atexit_registered = True
        self._host = _get_node_ip_local()
        used_ports: set[int] = set()

        def allocate_port(*, low: int, high: int) -> int:
            port = _get_free_port_local(
                low,
                high,
                max_retries=None,
                excluded_ports=used_ports,
            )
            used_ports.add(port)
            return port

        self._etcd_port = allocate_port(
            low=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
        )
        self._etcd_peer_port = allocate_port(
            low=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
        )
        self._nats_port = allocate_port(
            low=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
        )
        self._frontend_port = allocate_port(
            low=DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
            high=DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
        )
        self._manager_env = {
            "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
            "NATS_SERVER": f"nats://{self._host}:{self._nats_port}",
            "DYN_NAMESPACE": self._namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            # Dynamo 1.3.0's legacy tool jail rebuilds response chunks with
            # nvext=None. Its qwen3/deepseek v2 path preserves engine_data,
            # which NeMo-Gym needs for exact token IDs and log probabilities.
            "DYN_ENABLE_EXPERIMENTAL_PARSERS_V2": "1",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
            "DYN_RL_INIT_WEIGHTS_TIMEOUT_S": str(self._dynamo_cfg.control_timeout_s),
        }
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
                engine_world_size=self._engine_world_size,
                manager_env=self._manager_env,
                startup_timeout_s=self._dynamo_cfg.startup_timeout_s,
            )
            self._pool.start()
            self._start_frontend()
            self._wait_for_frontend(self._pool.size)
            self._started = True
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
            if not key.startswith(("DYN_", "ETCD_", "NATS_"))
        }
        env.update(self._manager_env)
        return env

    def _frontend_env(self) -> dict[str, str]:
        env = self._service_env()
        frontend_args = self._dynamo_cfg.frontend_args
        env["DYN_TOKENIZER"] = frontend_args.tokenizer
        if frontend_args.tokenizer_cache:
            env["DYN_TOKENIZER_CACHE"] = "1"
            env["DYN_TOKENIZER_CACHE_BYTES"] = str(frontend_args.tokenizer_cache_bytes)
        return env

    def _start_etcd(self) -> None:
        self._etcd_data_dir = tempfile.mkdtemp(prefix="nemorl_dynamo_etcd_")
        peer_url = f"http://{self._host}:{self._etcd_peer_port}"
        command = [
            get_dynamo_executable("etcd"),
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
                get_dynamo_executable("nats-server"),
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
            get_dynamo_python(),
            "-m",
            "dynamo.frontend",
            *argv,
        ]
        print(
            f"  [Dynamo] launching frontend argv={redact_argv(command)!r}", flush=True
        )
        frontend_env = self._frontend_env()
        tokenizer_env = {
            key: value
            for key, value in frontend_env.items()
            if key.startswith("DYN_TOKENIZER")
        }
        print(
            f"  [Dynamo] frontend tokenizer environment={tokenizer_env!r}",
            flush=True,
        )
        self._frontend_process = subprocess.Popen(
            command, env=frontend_env, start_new_session=True
        )

    def _wait_for_frontend(self, expected_workers: int) -> None:
        health_url = f"http://127.0.0.1:{self._frontend_port}/health"
        models_url = f"http://127.0.0.1:{self._frontend_port}/v1/models"
        expected_model = str(self._config["model_name"])
        deadline = time.monotonic() + self._dynamo_cfg.startup_timeout_s
        last_counts = (0, 0)
        last_models: set[str] = set()
        while time.monotonic() < deadline:
            if self._pool is None or not self._pool.is_alive():
                raise RuntimeError(
                    "A Ray-managed Dynamo vLLM worker exited while the frontend "
                    "was waiting for model registration."
                )
            if (
                self._frontend_process is not None
                and self._frontend_process.poll() is not None
            ):
                raise RuntimeError(
                    f"Dynamo frontend exited with code {self._frontend_process.returncode}."
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
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
                # /health reflects discovery registrations before the frontend's
                # model watcher has necessarily installed its OpenAI routes. Do
                # not expose frontend_url until the served model is visible;
                # otherwise an immediate /v1/completions request can race with
                # watcher setup and receive a transient 404.
                try:
                    with urllib.request.urlopen(models_url, timeout=5) as response:
                        models_payload = json.loads(response.read())
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    json.JSONDecodeError,
                ):
                    time.sleep(1)
                    continue
                last_models = {
                    str(model["id"])
                    for model in models_payload.get("data", [])
                    if isinstance(model, dict) and model.get("id") is not None
                }
                if expected_model not in last_models:
                    time.sleep(1)
                    continue
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
            f"last_generate={last_counts[0]}, last_rl={last_counts[1]}, "
            f"expected_model={expected_model!r}, last_models={sorted(last_models)!r}."
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
        """Best-effort, idempotent teardown of every owned resource."""
        try:
            self._stop_process(self._frontend_process, "frontend")
        except Exception:
            LOGGER.exception("Failed to stop the managed Dynamo frontend")
        self._frontend_process = None
        pool = self._pool
        self._pool = None
        if pool is not None:
            try:
                pool.shutdown()
            except Exception:
                LOGGER.exception("Failed to stop the managed Dynamo worker pool")
        try:
            self._stop_process(self._nats_process, "NATS")
        except Exception:
            LOGGER.exception("Failed to stop managed Dynamo NATS")
        self._nats_process = None
        try:
            self._stop_process(self._etcd_process, "etcd")
        except Exception:
            LOGGER.exception("Failed to stop managed Dynamo etcd")
        self._etcd_process = None
        if self._etcd_data_dir is not None:
            shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
            self._etcd_data_dir = None
        if self._nats_data_dir is not None:
            shutil.rmtree(self._nats_data_dir, ignore_errors=True)
            self._nats_data_dir = None
        if self._atexit_registered:
            atexit.unregister(self.shutdown)
            self._atexit_registered = False
        self._started = False
