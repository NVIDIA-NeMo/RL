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

The engines are NeMo RL's own workers; each registers itself with these
services from inside its worker (vLLM: ``native_engine``; Megatron:
``MegatronPolicyWorker.start_dynamo_endpoint``), so the runtime only launches
the services and waits to see the engines. The ``dynamo.vllm`` subprocess fleet
keeps ``legacy_managed_runtime.LegacyManagedDynamoRuntime``.
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
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)

# The component an aggregated or decode engine registers under; a prefill
# engine registers under "prefill".
DYNAMO_COMPONENT = "backend"
DYNAMO_ENDPOINT = "generate"

LOGGER = logging.getLogger(__name__)


class FrontendArgs(Protocol):
    """The ``dynamo_cfg.frontend_args`` fields the runtime reads."""

    @property
    def router_mode(self) -> str: ...

    @property
    def extra_cli_args(self) -> Sequence[str]: ...


class ManagedDynamoCfg(Protocol):
    """The ``dynamo_cfg`` fields the runtime reads, whichever backend owns the block."""

    @property
    def startup_timeout_s(self) -> float: ...

    @property
    def frontend_args(self) -> FrontendArgs: ...


def frontend_argv(
    *, host: str, port: int, namespace: str, dynamo_cfg: ManagedDynamoCfg
) -> list[str]:
    """The ``dynamo.frontend`` arguments; ``extra_cli_args`` may not repeat one."""
    argv = [
        "--http-host",
        host,
        "--http-port",
        str(port),
        "--namespace-prefix",
        namespace,
        "--discovery-backend",
        "etcd",
        "--request-plane",
        "tcp",
        "--event-plane",
        "nats",
        "--router-mode",
        dynamo_cfg.frontend_args.router_mode,
    ]
    extra = dynamo_cfg.frontend_args.extra_cli_args
    for token in extra:
        flag = token.split("=", 1)[0].replace("_", "-")
        if flag.startswith("--") and flag in argv:
            raise ValueError(
                f"Dynamo frontend option {flag} is set by both the managed runtime "
                "and dynamo_cfg.frontend_args.extra_cli_args."
            )
    return argv + list(extra)


def _service_executable(name: str) -> str:
    """The etcd or nats-server binary.

    Looked up in ``NEMO_RL_DYNAMO_BIN_DIR`` when set, else on ``PATH``, else in
    the image's Dynamo virtual environment when one is present.
    """
    bin_dir = os.environ.get("NEMO_RL_DYNAMO_BIN_DIR")
    found = shutil.which(name, path=os.path.expanduser(bin_dir) if bin_dir else None)
    if found is None and not bin_dir:
        try:
            # The image's separate Dynamo venv, when the build still ships one.
            from nemo_rl.models.generation.dynamo.venv import get_dynamo_venv_dir

            found = shutil.which(name, path=os.path.join(get_dynamo_venv_dir(), "bin"))
        except (ImportError, RuntimeError):
            found = None
    if found is None:
        where = f"NEMO_RL_DYNAMO_BIN_DIR={bin_dir}" if bin_dir else "PATH"
        raise FileNotFoundError(
            f"{name} is not an executable on {where}. Install etcd and nats-server "
            "and set NEMO_RL_DYNAMO_BIN_DIR to their directory."
        )
    return found


# The KV router's applied-event counter (Dynamo's kvrouter.KV_CACHE_EVENTS_APPLIED),
# one sample per label set.
_KV_EVENTS_APPLIED = re.compile(
    r'^\w*kv_cache_events_applied(?:_total)?\{(?=[^}]*status="(?P<status>[^"]*)")'
    r"[^}]*\}\s+(?P<value>\S+)$",
    re.MULTILINE,
)


def _managed_namespace() -> str:
    raw = f"nemo-rl-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-_").lower()


def wait_for_engine_registrations(
    *,
    frontend_url: str,
    namespace: str,
    expected_components: Mapping[str, int],
    model_name: str,
    timeout_s: float,
    check_alive: Callable[[], None],
) -> None:
    """Block until the frontend lists every expected engine and the model.

    ``/health`` lists discovery registrations; every component in
    ``expected_components`` must show at least its count of instance ids on
    the generate endpoint. ``/v1/models`` must then list ``model_name``: the
    frontend installs its OpenAI routes from a model watcher that can lag
    discovery, and exposing the URL before that races a transient 404. A
    prefill/decode frontend answers 503 until both roles are live and never
    falls back to aggregated serving, so every component must register.

    Runs without the service processes too, which is how a copy of the owner
    on another process (the SingleController actor's) waits.

    Args:
        frontend_url: The frontend's OpenAI base URL, ending in ``/v1``.
        namespace: Dynamo namespace the engines register under.
        expected_components: Component name to the number of engines in it.
        model_name: Served model name ``/v1/models`` must list.
        timeout_s: Total wait budget.
        check_alive: Runs on every poll; raises when an engine or service died,
            so a failed registration fails at once.

    Raises:
        RuntimeError: When the fleet is not fully registered within ``timeout_s``.
    """
    root = frontend_url.rstrip("/").removesuffix("/v1")
    health_url = f"{root}/health"
    models_url = f"{root}/v1/models"
    deadline = time.monotonic() + timeout_s
    registered: dict[str, int] = {component: 0 for component in expected_components}
    served: set[str] = set()
    while time.monotonic() < deadline:
        check_alive()
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                health = json.loads(response.read())
            with urllib.request.urlopen(models_url, timeout=5) as response:
                models = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
            continue
        instances: dict[str, set[str]] = {
            component: set() for component in expected_components
        }
        for instance in health.get("instances", []):
            if not isinstance(instance, dict):
                continue
            component = instance.get("component")
            if (
                instance.get("namespace") == namespace
                and component in instances
                and instance.get("endpoint") == DYNAMO_ENDPOINT
                and instance.get("instance_id") is not None
            ):
                instances[str(component)].add(str(instance["instance_id"]))
        registered = {component: len(ids) for component, ids in instances.items()}
        served = {
            str(model["id"])
            for model in models.get("data", [])
            if isinstance(model, dict) and model.get("id") is not None
        }
        if (
            all(
                registered[component] >= count
                for component, count in expected_components.items()
            )
            and model_name in served
        ):
            print(
                f"  [Dynamo] frontend ready: {root} routes to {registered} engine(s) "
                f"for {model_name!r}",
                flush=True,
            )
            return
        time.sleep(1)
    raise RuntimeError(
        "Dynamo frontend did not see every engine within "
        f"{timeout_s}s: expected={dict(expected_components)}, "
        f"registered={registered}, model={model_name!r}, served={sorted(served)!r}."
    )


class ManagedDynamoRuntime:
    """Own etcd, NATS and the frontend that NeMo RL's engines register with."""

    def __init__(
        self, *, dynamo_cfg: ManagedDynamoCfg, model_name: str, frontend_python: str
    ) -> None:
        """Describe the services; nothing starts until :meth:`start`.

        Args:
            dynamo_cfg: The validated ``policy.generation.dynamo_cfg``.
            model_name: The served model; the frontend is ready once it lists it.
            frontend_python: An interpreter whose environment has ``ai-dynamo``:
                the engine workers' own, so the frontend runs the same release
                as every engine.
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
        # files to a frontend on another node.
        return {**self._manager_env, "DYN_SYSTEM_PORT": "0"}

    def start(self) -> None:
        """Start etcd, NATS and the frontend."""
        if self._started:
            raise RuntimeError("Managed Dynamo runtime is already started")
        # Managed services use new process groups and would survive an
        # exception that escapes setup. Keep an interpreter-exit fallback in
        # addition to the normal explicit shutdown path.
        atexit.register(self.shutdown)
        self._host = _get_node_ip_local()
        control = (
            DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
        )
        http = (DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW, DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH)
        ports: list[int] = []
        for low, high in (control, control, control, http):
            ports.append(
                _get_free_port_local(
                    low, high, max_retries=None, excluded_ports=set(ports)
                )
            )
        self._etcd_port, self._etcd_peer_port, self._nats_port, self._frontend_port = (
            ports
        )
        self._manager_env = {
            "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
            "NATS_SERVER": f"nats://{self._host}:{self._nats_port}",
            "DYN_NAMESPACE": self._namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
            # A canary request would stall behind a refit pause; engine liveness
            # is the Ray actor's, as for every other NeMo RL worker.
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
        }
        print(f"  [Dynamo] managed environment={self._manager_env!r}", flush=True)
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

    def _start_etcd(self) -> None:
        self._etcd_data_dir = tempfile.mkdtemp(prefix="nemorl_dynamo_etcd_")
        peer_url = f"http://{self._host}:{self._etcd_peer_port}"
        command = [
            _service_executable("etcd"),
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
                _service_executable("nats-server"),
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
        argv = frontend_argv(
            host="0.0.0.0",
            port=self._frontend_port,
            namespace=self._namespace,
            dynamo_cfg=self._dynamo_cfg,
        )
        command = [self._frontend_python, "-m", "dynamo.frontend", *argv]
        print(f"  [Dynamo] launching frontend argv={command!r}", flush=True)
        frontend_env = self._service_env()
        dynamo_venv = os.path.dirname(os.path.dirname(self._frontend_python))
        frontend_env["VIRTUAL_ENV"] = dynamo_venv
        frontend_env["UV_PROJECT_ENVIRONMENT"] = dynamo_venv
        self._frontend_process = subprocess.Popen(
            command, env=frontend_env, start_new_session=True
        )

    def wait_for_engines(
        self,
        expected_engines: int | Mapping[str, int],
        check_engines: Callable[[], None],
    ) -> None:
        """Block until the frontend routes to every expected engine and the model.

        ``expected_engines`` is an engine count under ``DYNAMO_COMPONENT``, or a
        count per component (prefill engines register under their own).
        ``check_engines`` runs on every poll and raises when an engine exited or
        its endpoint stopped, so a failed registration fails at once; the owned
        service processes are checked alongside.
        """
        expected = (
            {DYNAMO_COMPONENT: expected_engines}
            if isinstance(expected_engines, int)
            else dict(expected_engines)
        )

        def check_alive() -> None:
            self._assert_services_alive()
            check_engines()

        wait_for_engine_registrations(
            frontend_url=f"http://127.0.0.1:{self._frontend_port}/v1",
            namespace=self._namespace,
            expected_components=expected,
            model_name=self._model_name,
            timeout_s=self._dynamo_cfg.startup_timeout_s,
            check_alive=check_alive,
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
        atexit.unregister(self.shutdown)
        self._started = False
