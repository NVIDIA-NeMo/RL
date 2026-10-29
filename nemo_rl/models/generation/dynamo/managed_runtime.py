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

"""Driver-owned Dynamo services: etcd, NATS and the OpenAI frontend.

The vLLM engines are NeMo RL's own workers; each registers itself with these
services (see ``native_engine``), so the runtime only waits to see them.
"""

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
from typing import Callable

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_frontend_argv,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo.venv import get_dynamo_service_executable

# The engines register on this component and endpoint (see native_engine).
DYNAMO_COMPONENT = "backend"
DYNAMO_ENDPOINT = "generate"

LOGGER = logging.getLogger(__name__)
# The KV router's applied-event counter (Dynamo's kvrouter.KV_CACHE_EVENTS_APPLIED),
# one sample per label set.
_KV_EVENTS_APPLIED = re.compile(
    r'^\w*kv_cache_events_applied(?:_total)?\{(?=[^}]*status="(?P<status>[^"]*)")'
    r"[^}]*\}\s+(?P<value>\S+)$",
    re.MULTILINE,
)


def _managed_namespace() -> str:
    raw = f"nemo-rl-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    namespace = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw)).strip("-_").lower()
    if not namespace:
        raise ValueError(f"Could not derive a valid Dynamo namespace from {raw!r}.")
    return namespace


class ManagedDynamoRuntime:
    """Own etcd, NATS and the frontend that NeMo RL's vLLM engines register with."""

    def __init__(
        self, *, dynamo_cfg: DynamoCfg, model_name: str, frontend_python: str
    ) -> None:
        """Describe the services; nothing starts until :meth:`start`.

        Args:
            dynamo_cfg: The validated ``policy.generation.dynamo_cfg``.
            model_name: The served model; the frontend is ready once it lists it.
            frontend_python: An interpreter whose environment has ``ai-dynamo``:
                the vLLM workers' own.
        """
        self._dynamo_cfg = dynamo_cfg
        self._model_name = model_name
        self._frontend_python = frontend_python
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

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def frontend_url(self) -> str:
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{host}:{self._frontend_port}/v1"

    @property
    def engine_env(self) -> dict[str, str]:
        """The environment each engine's Dynamo worker needs to register."""
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        # An OS-assigned system port makes each engine serve its own model-card
        # files (a custom chat template included) to a frontend on another node.
        return {**self._manager_env, "DYN_SYSTEM_PORT": "0"}

    def start(self) -> None:
        """Start etcd, NATS and the frontend."""
        if self._started:
            raise RuntimeError("Managed Dynamo runtime is already started")
        # Managed services use new process groups and would survive an
        # exception that escapes setup. Keep an interpreter-exit fallback in
        # addition to the normal explicit shutdown path.
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
            # Dynamo's legacy tool jail rebuilds response chunks with nvext=None.
            # Its v2 parser path preserves engine_data, which NeMo-Gym needs for
            # exact token IDs and log probabilities.
            "DYN_ENABLE_EXPERIMENTAL_PARSERS_V2": "1",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
            # A canary request would stall behind a refit pause; engine liveness
            # is the Ray actor's, as for every other vLLM worker.
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
        }
        print(
            f"  [Dynamo] managed environment={redact_environment(self._manager_env)!r}",
            flush=True,
        )
        try:
            self._start_etcd()
            self._start_nats()
            self._start_frontend()
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
            get_dynamo_service_executable("etcd"),
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
                get_dynamo_service_executable("nats-server"),
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
            self._frontend_python,
            "-m",
            "dynamo.frontend",
            *argv,
        ]
        print(
            f"  [Dynamo] launching frontend argv={redact_argv(command)!r}", flush=True
        )
        frontend_env = self._frontend_env()
        dynamo_venv = os.path.dirname(os.path.dirname(self._frontend_python))
        frontend_env["VIRTUAL_ENV"] = dynamo_venv
        frontend_env["UV_PROJECT_ENVIRONMENT"] = dynamo_venv
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

    def wait_for_engines(
        self, expected_engines: int, check_engines: Callable[[], None]
    ) -> None:
        """Block until the frontend routes to ``expected_engines`` registered engines.

        ``/health`` lists discovery registrations before the frontend's model
        watcher has installed its OpenAI routes, so readiness also needs the
        served model on ``/v1/models``; otherwise an immediate request can race
        the watcher and get a transient 404.

        ``check_engines`` runs on every poll and raises when an engine exited or
        its endpoint stopped, so a failed registration fails at once.
        """
        health_url = f"http://127.0.0.1:{self._frontend_port}/health"
        models_url = f"http://127.0.0.1:{self._frontend_port}/v1/models"
        deadline = time.monotonic() + self._dynamo_cfg.startup_timeout_s
        registered = 0
        served: set[str] = set()
        while time.monotonic() < deadline:
            self._assert_services_alive()
            check_engines()
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    health = json.loads(response.read())
                with urllib.request.urlopen(models_url, timeout=5) as response:
                    models = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(1)
                continue
            registered = len(
                {
                    str(instance["instance_id"])
                    for instance in health.get("instances", [])
                    if isinstance(instance, dict)
                    and instance.get("namespace") == self._namespace
                    and instance.get("component") == DYNAMO_COMPONENT
                    and instance.get("endpoint") == DYNAMO_ENDPOINT
                    and instance.get("instance_id") is not None
                }
            )
            served = {
                str(model["id"])
                for model in models.get("data", [])
                if isinstance(model, dict) and model.get("id") is not None
            }
            if registered >= expected_engines and self._model_name in served:
                print(
                    f"  [Dynamo] frontend {self.frontend_url} routes to "
                    f"{registered} engine(s)",
                    flush=True,
                )
                return
            time.sleep(1)
        raise RuntimeError(
            "Dynamo frontend did not see every vLLM engine within "
            f"{self._dynamo_cfg.startup_timeout_s}s: expected={expected_engines}, "
            f"registered={registered}, model={self._model_name!r}, "
            f"served={sorted(served)!r}."
        )

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

    def _log_router_kv_events(self) -> None:
        """Log how many engine KV-cache events the KV router applied, by status.

        Read from the frontend's metrics before it stops. A KV router that never
        applied one routed on load alone.
        """
        if self._frontend_process is None or self._frontend_process.poll() is not None:
            return
        url = f"http://127.0.0.1:{self._frontend_port}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                metrics = response.read().decode()
        except (urllib.error.URLError, TimeoutError) as error:
            print(
                f"  [Dynamo] router KV events: metrics unavailable ({error})",
                flush=True,
            )
            return
        applied: dict[str, float] = {}
        for match in _KV_EVENTS_APPLIED.finditer(metrics):
            applied[match["status"]] = applied.get(match["status"], 0.0) + float(
                match["value"]
            )
        print(f"  [Dynamo] router applied KV events: {json.dumps(applied)}", flush=True)

    def shutdown(self) -> None:
        """Best-effort, idempotent teardown of every owned resource."""
        try:
            self._log_router_kv_events()
        except Exception:
            LOGGER.exception("Failed to read the Dynamo router's KV-event metrics")
        try:
            self._stop_process(self._frontend_process, "frontend")
        except Exception:
            LOGGER.exception("Failed to stop the managed Dynamo frontend")
        self._frontend_process = None
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
