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

"""Driver-owned Dynamo services: etcd, NATS and the OpenAI frontends.

etcd and NATS run on the driver; each frontend runs on a generation node, in a
Ray actor. The vLLM engines are NeMo RL's own workers; each registers itself
with these services (see ``native_engine``), so the runtime only waits to see
them.
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
from typing import Callable, Sequence

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.config import DynamoCfg

# The engines register on this component and endpoint (see native_engine).
DYNAMO_COMPONENT = "backend"
DYNAMO_ENDPOINT = "generate"

LOGGER = logging.getLogger(__name__)


def frontend_argv(
    *, host: str, port: int, namespace: str, dynamo_cfg: DynamoCfg
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
    if dynamo_cfg.frontend_args.num_frontends > 1:
        # Each replica's router sees the requests the others have in flight.
        argv.append("--router-replica-sync")
    if dynamo_cfg.frontend_args.session_affinity_ttl_secs is not None:
        argv += [
            "--router-session-affinity-ttl-secs",
            str(dynamo_cfg.frontend_args.session_affinity_ttl_secs),
            "--router-session-affinity-max-entries",
            str(dynamo_cfg.frontend_args.session_affinity_max_entries),
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
    """The etcd or nats-server binary: in NEMO_RL_DYNAMO_BIN_DIR when set, else on PATH."""
    bin_dir = os.environ.get("NEMO_RL_DYNAMO_BIN_DIR")
    found = shutil.which(name, path=os.path.expanduser(bin_dir) if bin_dir else None)
    if found is None:
        where = f"NEMO_RL_DYNAMO_BIN_DIR={bin_dir}" if bin_dir else "PATH"
        raise FileNotFoundError(
            f"{name} is not an executable on {where}. Build the image with "
            "--build-arg BUILD_DYNAMO=1, or install etcd and nats-server with "
            "docker/dynamo/install.sh and set NEMO_RL_DYNAMO_BIN_DIR to their "
            "directory."
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


class FrontendHost:
    """One ``dynamo.frontend`` on the node a Ray actor (``_FrontendHostActor``) runs on.

    If the actor dies, Ray kills the frontend with it (Ray reaps an exiting
    worker's child processes).
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    def start(
        self,
        *,
        frontend_python: str,
        namespace: str,
        dynamo_cfg: DynamoCfg,
        env: dict[str, str],
    ) -> str:
        """Launch the frontend on a free port of this node; return its OpenAI base URL."""
        port = _get_free_port_local(
            DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
            max_retries=None,
        )
        argv = frontend_argv(
            host="0.0.0.0", port=port, namespace=namespace, dynamo_cfg=dynamo_cfg
        )
        command = [frontend_python, "-m", "dynamo.frontend", *argv]
        print(f"  [Dynamo] launching frontend argv={command!r}", flush=True)
        self._process = subprocess.Popen(
            command, env=_service_env(env), start_new_session=True
        )
        host = _get_node_ip_local()
        host = f"[{host}]" if ":" in host else host
        return f"http://{host}:{port}/v1"

    def returncode(self) -> int | None:
        """The frontend's exit code, or None while it runs."""
        if self._process is None:
            raise RuntimeError("The Dynamo frontend was never started")
        return self._process.poll()

    def stop(self) -> None:
        ManagedDynamoRuntime._stop_process(self._process, "frontend")
        self._process = None


_FrontendHostActor = ray.remote(num_cpus=0.001)(FrontendHost)


def _service_env(managed_env: dict[str, str]) -> dict[str, str]:
    """This process's environment with ``managed_env`` replacing every Dynamo setting."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DYN_", "ETCD_", "NATS_"))
    }
    env.update(managed_env)
    return env


class ManagedDynamoRuntime:
    """Own etcd, NATS and the frontends that NeMo RL's vLLM engines register with."""

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
        self._manager_env: dict[str, str] = {}
        self._started = False
        self._etcd_process: subprocess.Popen | None = None
        self._nats_process: subprocess.Popen | None = None
        self._frontend_hosts: list[ray.actor.ActorHandle] = []
        self._frontend_urls: list[str] = []
        self._etcd_data_dir: str | None = None
        self._nats_data_dir: str | None = None

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def frontend_urls(self) -> list[str]:
        """Every frontend's OpenAI base URL, one per ``num_frontends``."""
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        return list(self._frontend_urls)

    @property
    def engine_env(self) -> dict[str, str]:
        """The environment each engine's Dynamo worker needs to register."""
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        # An OS-assigned system port makes each engine serve its own model-card
        # files (a custom chat template included) to a frontend on another node.
        return {**self._manager_env, "DYN_SYSTEM_PORT": "0"}

    def start(self, *, frontend_node_ids: Sequence[str]) -> None:
        """Start etcd and NATS here, and the frontends on ``frontend_node_ids``.

        Frontend ``i`` runs on ``frontend_node_ids[i % len(frontend_node_ids)]``.
        """
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
        ports: list[int] = []
        for _ in range(3):
            ports.append(
                _get_free_port_local(
                    *control, max_retries=None, excluded_ports=set(ports)
                )
            )
        self._etcd_port, self._etcd_peer_port, self._nats_port = ports
        self._manager_env = {
            "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
            "NATS_SERVER": f"nats://{self._host}:{self._nats_port}",
            "DYN_NAMESPACE": self._namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            # Parse with Dynamo's unified v2 parsers wherever the parser pair has
            # a v2 family; they preserve engine_data, which NeMo-Gym needs for
            # exact token IDs and log probabilities. Other pairs (e.g. hermes)
            # keep the v1 parsers, as with 1.5.1's experimental flag.
            "DYN_PARSER_VERSION": "auto",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
            # NeMo-Gym sends media inline: raise the frontend's 45 MB HTTP body
            # limit, and the 32 MiB request-plane limit above it (the engine
            # request carries the same media plus token IDs).
            "DYN_HTTP_BODY_LIMIT_MB": "512",
            "DYN_TCP_MAX_MESSAGE_SIZE": str(1 << 30),
            # A canary request would stall behind a refit pause; engine liveness
            # is the Ray actor's, as for every other vLLM worker.
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
        }
        print(
            f"  [Dynamo] managed environment={self._manager_env!r}",
            flush=True,
        )
        try:
            self._start_etcd()
            self._start_nats()
            self._start_frontends(frontend_node_ids)
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
        return _service_env(self._manager_env)

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

    def _start_frontends(self, node_ids: Sequence[str]) -> None:
        dynamo_venv = os.path.dirname(os.path.dirname(self._frontend_python))
        env = {
            **self._manager_env,
            "VIRTUAL_ENV": dynamo_venv,
            "UV_PROJECT_ENVIRONMENT": dynamo_venv,
        }
        for index in range(self._dynamo_cfg.frontend_args.num_frontends):
            node_id = node_ids[index % len(node_ids)]
            self._frontend_hosts.append(
                _FrontendHostActor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id, soft=False
                    )
                ).remote()
            )
        self._frontend_urls = ray.get(
            [
                host.start.remote(
                    frontend_python=self._frontend_python,
                    namespace=self._namespace,
                    dynamo_cfg=self._dynamo_cfg,
                    env=env,
                )
                for host in self._frontend_hosts
            ]
        )

    def wait_for_engines(
        self, expected_engines: int, check_engines: Callable[[], None]
    ) -> None:
        """Block until every frontend routes to ``expected_engines`` registered engines.

        ``/health`` lists discovery registrations before the frontend's model
        watcher has installed its OpenAI routes, so readiness also needs the
        served model on ``/v1/models``; otherwise an immediate request can race
        the watcher and get a transient 404.

        ``check_engines`` runs on every poll and raises when an engine exited or
        its endpoint stopped, so a failed registration fails at once.
        """
        deadline = time.monotonic() + self._dynamo_cfg.startup_timeout_s
        registered = 0
        served: set[str] = set()
        pending = list(self._frontend_urls)
        while time.monotonic() < deadline:
            self._assert_services_alive()
            check_engines()
            url = pending[0]
            try:
                with urllib.request.urlopen(
                    f"{url.removesuffix('/v1')}/health?instances=true", timeout=5
                ) as response:
                    health = json.loads(response.read())
                with urllib.request.urlopen(f"{url}/models", timeout=5) as response:
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
                    f"  [Dynamo] frontend {url} routes to {registered} engine(s)",
                    flush=True,
                )
                pending.pop(0)
                if not pending:
                    return
                continue
            time.sleep(1)
        raise RuntimeError(
            f"Dynamo frontend {pending[0]} did not see every vLLM engine within "
            f"{self._dynamo_cfg.startup_timeout_s}s: expected={expected_engines}, "
            f"registered={registered}, model={self._model_name!r}, "
            f"served={sorted(served)!r}."
        )

    def _assert_services_alive(self) -> None:
        for label, process in (
            ("etcd", self._etcd_process),
            ("NATS", self._nats_process),
        ):
            if process is None or process.poll() is not None:
                code = None if process is None else process.returncode
                raise RuntimeError(
                    f"Managed Dynamo {label} is not alive (code={code})."
                )
        codes = ray.get([host.returncode.remote() for host in self._frontend_hosts])
        for url, code in zip(self._frontend_urls, codes):
            if code is not None:
                raise RuntimeError(
                    f"Managed Dynamo frontend {url} is not alive (code={code})."
                )

    def _log_router_kv_events(self) -> None:
        """Log how many engine KV-cache events each KV router applied, by status.

        Read from each frontend's metrics before it stops. A KV router that never
        applied one routed on load alone.
        """
        for url in self._frontend_urls:
            self._log_router_kv_events_of(url)

    def _log_router_kv_events_of(self, frontend_url: str) -> None:
        url = f"{frontend_url.removesuffix('/v1')}/metrics"
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
        print(
            f"  [Dynamo] router applied KV events: {json.dumps(applied)} ({url})",
            flush=True,
        )

    def shutdown(self) -> None:
        """Best-effort, idempotent teardown of every owned resource."""
        try:
            self._log_router_kv_events()
        except Exception:
            LOGGER.exception("Failed to read the Dynamo router's KV-event metrics")
        for host in self._frontend_hosts:
            try:
                ray.get(host.stop.remote())
                ray.kill(host)
            except Exception:
                LOGGER.exception("Failed to stop a managed Dynamo frontend")
        self._frontend_hosts = []
        self._frontend_urls = []
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
