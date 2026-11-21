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
Ray actor. The engines are NeMo RL's own workers; each registers itself with
these services from inside its worker (vLLM: see ``engine_adapter``; Megatron:
``MegatronDynamoGenerationMixin.start_dynamo_endpoint``), so the runtime only
launches the services and waits to see the engines.
"""

import atexit
import json
import logging
import os
import pickle
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, Protocol

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.config import trace_export_env

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
    def session_affinity_ttl_secs(self) -> int | None: ...

    @property
    def session_affinity_max_entries(self) -> int: ...

    @property
    def num_frontends(self) -> int: ...

    @property
    def render_workers(self) -> int: ...

    @property
    def extra_cli_args(self) -> Sequence[str]: ...

    @property
    def extra_env(self) -> Mapping[str, str]: ...


class ManagedDynamoCfg(Protocol):
    """The ``dynamo_cfg`` fields the runtime reads, whichever backend owns the block."""

    @property
    def startup_timeout_s(self) -> float: ...

    @property
    def frontend_args(self) -> FrontendArgs: ...


# NeMo RL's chat engine, as the custom chat processor Dynamo's frontend loads.
CHAT_PROCESSOR = "nemo_rl.models.generation.dynamo.vllm.chat_engine:provider"
# NeMo-Gym's generation-cut route, as the route extension the frontend loads.
CUT_ROUTES = "nemo_rl.models.generation.dynamo.vllm.generation_cut:routes"
# The frontend's HTTP body limit. NeMo-Gym sends media inline, and a generation
# cut lists every call in flight; Dynamo's default is 45 MB.
HTTP_BODY_LIMIT_MB = 512


class ChatProcessorOptions(NamedTuple):
    """What a frontend's chat engine starts with: Dynamo's chat processor config."""

    # Processes that render chat calls and build their replies.
    render_workers: int
    # The Ray cluster the chat engine's TQ reads go through.
    ray_address: str
    # Where setup_token_capture writes the capture config, on the frontend's node.
    capture_file: str
    # Where the driver writes the chat settings an engine sends before it loads.
    # Set only for a run that serves chat: the frontend then starts its render
    # workers at launch.
    chat_settings_file: str | None


def frontend_argv(
    *,
    host: str,
    port: int,
    namespace: str,
    dynamo_cfg: ManagedDynamoCfg,
    chat_options: ChatProcessorOptions,
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
        "--dyn-chat-processor",
        CHAT_PROCESSOR,
        "--dyn-chat-processor-config",
        json.dumps(chat_options._asdict()),
        "--frontend-route-extension",
        CUT_ROUTES,
    ]
    if dynamo_cfg.frontend_args.num_frontends > 1:
        argv.append("--router-replica-sync")
    if dynamo_cfg.frontend_args.session_affinity_ttl_secs is not None:
        argv += [
            "--router-session-affinity-ttl-secs",
            str(dynamo_cfg.frontend_args.session_affinity_ttl_secs),
            "--router-session-affinity-max-entries",
            str(dynamo_cfg.frontend_args.session_affinity_max_entries),
        ]
    extra = dynamo_cfg.frontend_args.extra_cli_args
    seen: set[str] = set()
    for token in extra:
        flag = token.split("=", 1)[0].replace("_", "-")
        if not flag.startswith("--"):
            continue
        if flag in argv:
            raise ValueError(
                f"Dynamo frontend option {flag} is set by both the managed runtime "
                "and dynamo_cfg.frontend_args.extra_cli_args."
            )
        if flag in seen:
            raise ValueError(
                f"Dynamo frontend option {flag} is repeated in "
                "dynamo_cfg.frontend_args.extra_cli_args."
            )
        seen.add(flag)
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
# one sample per label set. Stored-block events only: every prefix-cache reset
# emits a payload-free "cleared" event, which applies even when the router
# decodes none of vLLM's block events.
_KV_EVENTS_APPLIED = re.compile(
    r'^\w*kv_cache_events_applied(?:_total)?\{(?=[^}]*event_type="stored")'
    r'(?=[^}]*status="(?P<status>[^"]*)")'
    r"[^}]*\}\s+(?P<value>\S+)$",
    re.MULTILINE,
)


# The frontend's own gauges, one sample per model: calls that wait for their first
# response, and calls in flight.
_FRONTEND_LOAD = re.compile(
    r"^(?P<name>dynamo_frontend_(?:queued|inflight)_requests)(?:\{[^}]*\})?"
    r"\s+(?P<value>\S+)$",
    re.MULTILINE,
)


def _read_metrics(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


def _managed_namespace() -> str:
    raw = f"nemo-rl-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-_").lower()


# How long a managed service has after SIGTERM before it gets SIGKILL.
_STOP_TIMEOUT_S = 15
_WATCHDOG_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "service_watchdog.py"
)


def _stop_process(
    process: subprocess.Popen, label: str, timeout_s: float = _STOP_TIMEOUT_S
) -> None:
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


class _ServiceProcess:
    """A managed service in its own session, stopped even if this process is SIGKILLed.

    The session keeps a Ctrl-C at the driver's terminal from reaching the
    service. :meth:`stop` stops it on a clean shutdown; otherwise its watchdog
    (``service_watchdog.py``) stops it once this process is gone. Ray reaps a
    worker's children only when the worker exits through Ray, not when the
    worker is SIGKILLed. The watchdog is the service's sibling, not its parent,
    so Ray's reaping still kills the service itself.
    """

    def __init__(
        self, command: Sequence[str], *, env: dict[str, str], label: str
    ) -> None:
        self._label = label
        # This process holds the only write end and never writes to it, so the
        # watchdog reads EOF once this process is gone.
        lifeline_read, self._lifeline = os.pipe()
        try:
            self._process = subprocess.Popen(command, env=env, start_new_session=True)
            try:
                self._watchdog = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        _WATCHDOG_SCRIPT,
                        str(lifeline_read),
                        str(self._process.pid),
                        str(_STOP_TIMEOUT_S),
                    ],
                    pass_fds=(lifeline_read,),
                    start_new_session=True,
                )
            except BaseException:
                _stop_process(self._process, label)
                raise
        except BaseException:
            os.close(self._lifeline)
            raise
        finally:
            os.close(lifeline_read)

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def poll(self) -> int | None:
        """The service's exit code, or None while it runs."""
        return self._process.poll()

    def stop(self) -> None:
        """Stop the service, then its watchdog."""
        try:
            _stop_process(self._process, self._label)
        finally:
            self._watchdog.kill()
            self._watchdog.wait()
            os.close(self._lifeline)


# Where the frontend's generation-cut route (dynamo/vllm/generation_cut.py) finds
# the capture config, which holds a cut's bearer token and timeout. In the
# frontend's environment because Dynamo passes a route extension no settings.
CAPTURE_FILE_ENV = "NEMO_RL_DYNAMO_CAPTURE_FILE"


def read_chat_capture(path: str) -> dict[str, Any] | None:
    """What ``publish_chat_capture`` wrote to ``path``; None before it has."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as capture_file:
        return pickle.load(capture_file)


def read_chat_settings(path: str) -> dict[str, Any] | None:
    """What ``publish_chat_settings`` wrote to ``path``; None before it has."""
    if not os.path.exists(path):
        return None
    with open(path) as settings_file:
        return json.load(settings_file)


# Launches of one frontend: a launch is repeated when another process took the port.
_FRONTEND_LAUNCHES = 3


def _answers_health(port: int) -> bool:
    """Whether an HTTP server on this node's ``port`` answers ``/health``."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _frontend_answers(process: _ServiceProcess, port: int, deadline: float) -> bool:
    """Wait for a launched frontend: True once it answers, False if it exited."""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        # Whichever process holds the port answers, so ours must still run.
        if _answers_health(port) and process.poll() is None:
            return True
        time.sleep(0.2)
    raise RuntimeError(
        f"The Dynamo frontend did not answer on port {port} within its startup timeout."
    )


def wait_for_engine_registrations(
    *,
    frontend_urls: Sequence[str],
    namespace: str,
    expected_components: Mapping[str, int],
    model_name: str,
    timeout_s: float,
    check_alive: Callable[[], None],
) -> None:
    """Block until every frontend lists every expected engine and the model.

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
        frontend_urls: Each frontend's OpenAI base URL, ending in ``/v1``.
        namespace: Dynamo namespace the engines register under.
        expected_components: Component name to the number of engines in it.
        model_name: Served model name ``/v1/models`` must list.
        timeout_s: Total wait budget, shared by every frontend.
        check_alive: Runs on every poll; raises when an engine or service died,
            so a failed registration fails at once.

    Raises:
        RuntimeError: When a frontend does not see the whole fleet within
            ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    registered: dict[str, int] = {component: 0 for component in expected_components}
    served: set[str] = set()
    pending = list(frontend_urls)
    while pending and time.monotonic() < deadline:
        check_alive()
        url = pending[0].rstrip("/")
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
                f"  [Dynamo] frontend {url} routes to {registered} engine(s) "
                f"for {model_name!r}",
                flush=True,
            )
            pending.pop(0)
            # The timeout error reports the counts of the frontend it names.
            registered = {component: 0 for component in expected_components}
            served = set()
            continue
        time.sleep(1)
    if pending:
        raise RuntimeError(
            f"Dynamo frontend {pending[0]} did not see every engine within "
            f"{timeout_s}s: expected={dict(expected_components)}, "
            f"registered={registered}, model={model_name!r}, "
            f"served={sorted(served)!r}."
        )


class FrontendHost:
    """One frontend on the node a Ray actor (``_FrontendHostActor``) runs on.

    It runs Dynamo's frontend with NeMo RL's chat engine serving chat calls
    (``CHAT_PROCESSOR``) and NeMo-Gym's generation cuts (``CUT_ROUTES``). The
    frontend stops when the actor dies, however it dies (see
    ``_ServiceProcess``).
    """

    def __init__(self) -> None:
        self._process: _ServiceProcess | None = None
        self._chat_dir: str | None = None

    def start(
        self,
        *,
        frontend_python: str,
        namespace: str,
        dynamo_cfg: ManagedDynamoCfg,
        env: dict[str, str],
        ray_address: str,
        serves_chat: bool,
        port_range: tuple[int, int],
    ) -> str:
        """Launch the frontend on a free port of this node; return its base URL.

        ``port_range`` is this frontend's slice ``[low, high)`` of Dynamo's port band.
        The port is free when it is picked, not held until the frontend binds it:
        a frontend that exits before it answers lost the port to another process,
        and is launched again on another one.
        """
        chat_dir = tempfile.mkdtemp(prefix="nemorl_dynamo_chat_")
        self._chat_dir = chat_dir
        capture_file = os.path.join(chat_dir, "capture.pkl")
        frontend_env = _service_env(env)
        frontend_env[CAPTURE_FILE_ENV] = capture_file
        deadline = time.monotonic() + dynamo_cfg.startup_timeout_s
        tried: set[int] = set()
        for _ in range(_FRONTEND_LAUNCHES):
            port = _get_free_port_local(
                *port_range, max_retries=None, excluded_ports=tried
            )
            tried.add(port)
            argv = frontend_argv(
                host="0.0.0.0",
                port=port,
                namespace=namespace,
                dynamo_cfg=dynamo_cfg,
                chat_options=ChatProcessorOptions(
                    render_workers=dynamo_cfg.frontend_args.render_workers,
                    ray_address=ray_address,
                    capture_file=capture_file,
                    chat_settings_file=(
                        os.path.join(chat_dir, "chat_settings.json")
                        if serves_chat
                        else None
                    ),
                ),
            )
            command = [frontend_python, "-m", "dynamo.frontend", *argv]
            print(f"  [Dynamo] launching frontend argv={command!r}", flush=True)
            process = _ServiceProcess(command, env=frontend_env, label="frontend")
            self._process = process
            if _frontend_answers(process, port, deadline):
                host = _get_node_ip_local()
                host = f"[{host}]" if ":" in host else host
                return f"http://{host}:{port}/v1"
            print(
                f"  [Dynamo] frontend exited with code {process.returncode} "
                f"before it answered on port {port}",
                flush=True,
            )
            process.stop()
            self._process = None
        raise RuntimeError(
            "The Dynamo frontend exited before it answered on each of the ports "
            f"{sorted(tried)}."
        )

    def publish_chat_capture(self, capture: dict[str, Any]) -> None:
        """Write the token-capture config this frontend's chat engine reads."""
        # Pickled: the data-plane config holds pydantic storage configs.
        self._write_chat_file("capture.pkl", pickle.dumps(capture))

    def publish_chat_settings(self, settings: dict[str, Any]) -> None:
        """Write the chat settings this frontend builds its render workers from."""
        self._write_chat_file("chat_settings.json", json.dumps(settings).encode())

    def _write_chat_file(self, name: str, content: bytes) -> None:
        if self._chat_dir is None:
            raise RuntimeError("The Dynamo frontend was never started")
        with tempfile.NamedTemporaryFile(
            "wb", dir=self._chat_dir, delete=False
        ) as chat_file:
            chat_file.write(content)
        # Renamed into place, so the chat engine never reads a partial file.
        os.replace(chat_file.name, os.path.join(self._chat_dir, name))

    def returncode(self) -> int | None:
        """The frontend's exit code, or None while it runs."""
        if self._process is None:
            raise RuntimeError("The Dynamo frontend was never started")
        return self._process.poll()

    def stop(self) -> None:
        if self._process is not None:
            self._process.stop()
            self._process = None
        if self._chat_dir is not None:
            shutil.rmtree(self._chat_dir, ignore_errors=True)
            self._chat_dir = None


_FrontendHostActor = ray.remote(num_cpus=0)(FrontendHost)


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
    """Own etcd, NATS and the frontends that NeMo RL's engines register with."""

    def __init__(
        self,
        *,
        dynamo_cfg: ManagedDynamoCfg,
        model_name: str,
        frontend_python: str,
        ray_address: str,
        serves_chat: bool,
    ) -> None:
        """Describe the services; nothing starts until :meth:`start`.

        Args:
            dynamo_cfg: The validated ``policy.generation.dynamo_cfg``.
            model_name: The served model; the frontend is ready once it lists it.
            frontend_python: An interpreter whose environment has ``ai-dynamo``:
                the engine workers' own, so the frontend runs the same release
                as every engine.
            ray_address: The Ray cluster's GCS address; the frontend's chat
                engine reads token-capture prefixes from TQ through it.
            serves_chat: Whether the engines expose chat
                (``vllm_cfg.expose_http_server``, which NeMo-Gym requires). The
                frontend then starts its render workers at launch.
        """
        self._dynamo_cfg = dynamo_cfg
        self._model_name = model_name
        self._frontend_python = frontend_python
        self._ray_address = ray_address
        self.serves_chat = serves_chat
        self._namespace = _managed_namespace()
        self._host = ""
        self._etcd_port = 0
        self._etcd_peer_port = 0
        self._nats_port = 0
        self._manager_env: dict[str, str] = {}
        self._started = False
        self._etcd_process: _ServiceProcess | None = None
        self._nats_process: _ServiceProcess | None = None
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
        # An exception that escapes setup skips shutdown(): stop the services at
        # interpreter exit too. Their watchdogs cover a process killed outright.
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
            # A field the frontend doesn't know fails the call with a 400 rather
            # than vanishing; an inherited setting can't turn that off.
            "DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS": "0",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
            # The 32 MiB request-plane limit goes above the HTTP body limit: an
            # engine request carries the same media plus token IDs.
            "DYN_HTTP_BODY_LIMIT_MB": str(HTTP_BODY_LIMIT_MB),
            "DYN_TCP_MAX_MESSAGE_SIZE": str(1 << 30),
            # A canary request would stall behind a refit pause; engine liveness
            # is the Ray actor's, as for every other NeMo RL worker.
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
            **trace_export_env(),
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
        self._etcd_process = _ServiceProcess(
            command, env=_service_env(self._manager_env), label="etcd"
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
        self._nats_process = _ServiceProcess(
            [
                _service_executable("nats-server"),
                "-js",
                "-sd",
                self._nats_data_dir,
                "-p",
                str(self._nats_port),
            ],
            env=_service_env(self._manager_env),
            label="NATS",
        )
        self._wait_for_port(self._nats_port, "NATS", self._nats_process)
        print(f"  [Dynamo] NATS ready on {self._host}:{self._nats_port}", flush=True)

    def _wait_for_port(
        self, port: int, label: str, process: _ServiceProcess | None = None
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
        extra_env = self._dynamo_cfg.frontend_args.extra_env
        if clashing := sorted(extra_env.keys() & (env.keys() | {CAPTURE_FILE_ENV})):
            raise ValueError(
                f"Dynamo frontend environment {clashing} is set by both the managed "
                "runtime and dynamo_cfg.frontend_args.extra_env."
            )
        # The driver's environment as it is now, as the engines get it
        # (RayWorkerGroup): an actor has only what the driver had at init_ray().
        env = _service_env({**env, **extra_env})
        # ray.init() raises a driver's open-file limit to 65,536, but not a Ray
        # worker's: a frontend started from one would stop at Dynamo's 8,192.
        env.setdefault("DYN_FRONTEND_FD_LIMIT_TARGET", "65536")
        num_frontends = self._dynamo_cfg.frontend_args.num_frontends
        for index in range(num_frontends):
            node_id = node_ids[index % len(node_ids)]
            self._frontend_hosts.append(
                _FrontendHostActor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id, soft=False
                    )
                ).remote()
            )
        # Frontends that share a node pick from separate slices of Dynamo's port
        # band, so two of them never pick the same port.
        low = DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW
        ports = (DEFAULT_DYNAMO_CONTROL_PORT_RANGE_HIGH - low) // -(
            -num_frontends // len(node_ids)
        )
        self._frontend_urls = ray.get(
            [
                host.start.remote(
                    frontend_python=self._frontend_python,
                    namespace=self._namespace,
                    dynamo_cfg=self._dynamo_cfg,
                    env=env,
                    ray_address=self._ray_address,
                    serves_chat=self.serves_chat,
                    port_range=(
                        low + (index // len(node_ids)) * ports,
                        low + (index // len(node_ids) + 1) * ports,
                    ),
                )
                for index, host in enumerate(self._frontend_hosts)
            ]
        )
        # The telemetry names a frontend by its place in this list.
        print(f"  [Dynamo] frontends, by index: {self._frontend_urls}", flush=True)

    def publish_chat_capture(
        self,
        *,
        dp_cfg: dict[str, Any],
        staging_partition: str,
        capture_media: bool,
        generation_cut_control_token: str | None,
        generation_cut_control_timeout_s: float | None,
    ) -> None:
        """Tell every frontend's chat engine where token-capture prefixes are read.

        Capture is set up after the engines register, so each chat engine reads
        this on its first captured call. ``generation_cut_control_token`` is the
        bearer token a generation cut must carry, None to refuse every cut;
        ``generation_cut_control_timeout_s`` bounds one cut across the engines.
        """
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        capture = {
            "data_plane": dp_cfg,
            "staging_partition": staging_partition,
            "capture_media": capture_media,
            "generation_cut_control_token": generation_cut_control_token,
            "generation_cut_control_timeout_s": generation_cut_control_timeout_s,
        }
        ray.get(
            [host.publish_chat_capture.remote(capture) for host in self._frontend_hosts]
        )

    def publish_chat_settings(self, settings: dict[str, Any]) -> None:
        """Hand every frontend the chat settings an engine sent before it loaded.

        ``settings`` is what the engine publishes at registration
        (``chat_runtime_data``). Each frontend builds its render workers from
        them while the engines load, and registration confirms them.
        """
        if not self._started:
            raise RuntimeError("Managed Dynamo runtime has not been started")
        ray.get(
            [
                host.publish_chat_settings.remote(settings)
                for host in self._frontend_hosts
            ]
        )

    def wait_for_engines(
        self,
        expected_engines: int | Mapping[str, int],
        check_engines: Callable[[], None],
    ) -> None:
        """Block until every frontend routes to every expected engine and the model.

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
            frontend_urls=self._frontend_urls,
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
        """Log how many stored-block KV events each KV router applied, by status.

        Read from each frontend's metrics before it stops. A KV router that never
        applied one routed on load alone.
        """
        for url in self._frontend_urls:
            self._log_router_kv_events_of(url)

    def _log_router_kv_events_of(self, frontend_url: str) -> None:
        url = f"{frontend_url.removesuffix('/v1')}/metrics"
        try:
            metrics = _read_metrics(url)
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

    def frontend_load(self) -> dict[str, dict[int, list[float]]]:
        """Every frontend's queued and in-flight calls, shaped as the vLLM logger metrics.

        Keyed by metric name, then by frontend index. A frontend that does not
        answer is left out.
        """
        load: dict[str, dict[int, list[float]]] = {
            f"dynamo_frontend_{name}_requests": {} for name in ("queued", "inflight")
        }
        for index, frontend_url in enumerate(self._frontend_urls):
            try:
                metrics = _read_metrics(f"{frontend_url.removesuffix('/v1')}/metrics")
            except (urllib.error.URLError, TimeoutError):
                continue
            for match in _FRONTEND_LOAD.finditer(metrics):
                samples = load[match["name"]].setdefault(index, [0.0])
                samples[0] += float(match["value"])
        return load

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
            if self._nats_process is not None:
                self._nats_process.stop()
        except Exception:
            LOGGER.exception("Failed to stop managed Dynamo NATS")
        self._nats_process = None
        try:
            if self._etcd_process is not None:
                self._etcd_process.stop()
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
