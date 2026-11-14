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
import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.megatron import dynamo_worker_mixin
from nemo_rl.models.generation.megatron.dynamo_worker_mixin import (
    MegatronDynamoGenerationMixin,
    require_megatron_dynamo_integration,
)
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

pytestmark = pytest.mark.mcore


class _Worker(MegatronDynamoGenerationMixin, MegatronGenerationMixin):
    """MegatronDynamoPolicyWorker's generation bases, in the order it composes them."""


def _bare_worker(role, cls=_Worker):
    worker = object.__new__(cls)
    worker.cfg = {"generation": {}}
    if role is not None:
        worker.cfg["generation"]["_dynamo_role"] = role
    return worker


@pytest.fixture
def engine_modules(monkeypatch):
    """Fake the plain and the Dynamo handoff engine classes the hooks pick between."""
    plain = MagicMock(name="DynamicInferenceEngine")
    handoff = MagicMock(name="DynamoDynamicInferenceEngine")
    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.DynamicInferenceEngine", plain
    )
    for name in (
        "megatron.inference",
        "megatron.inference.integrations",
        "megatron.inference.integrations.dynamo",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("megatron.inference.integrations.dynamo.dynamic_engine")
    module.DynamoDynamicInferenceEngine = handoff
    monkeypatch.setitem(
        sys.modules, "megatron.inference.integrations.dynamo.dynamic_engine", module
    )
    monkeypatch.setattr(dynamo_worker_mixin, "_get_node_ip_local", lambda: "10.0.0.5")
    return plain, handoff


def test_plain_worker_builds_the_plain_engine(engine_modules):
    plain, handoff = engine_modules
    worker = _bare_worker(role=None, cls=MegatronGenerationMixin)
    controller, context = MagicMock(), MagicMock()

    assert worker._inference_config_overrides() == {}
    assert worker._coordinator_kwargs() == {}
    assert worker._build_dynamic_inference_engine(controller, context) is (
        plain.return_value
    )
    plain.assert_called_once_with(controller, context)
    handoff.assert_not_called()


def test_dynamo_worker_needs_its_role():
    worker = _bare_worker(role=None)

    with pytest.raises(RuntimeError, match="_dynamo_role"):
        worker._dynamo_role


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_dynamo_worker_builds_the_handoff_engine(engine_modules, role):
    plain, handoff = engine_modules
    worker = _bare_worker(role=role)
    controller, context = MagicMock(), MagicMock()

    assert worker._dynamo_role == role
    # The peer role lives in other processes: NIXL carries the handoff and the
    # coordinator must listen on a routable address.
    assert worker._inference_config_overrides() == {"kv_transport_backend": "nixl"}
    assert worker._coordinator_kwargs() == {"hostname": "10.0.0.5"}

    engine = worker._build_dynamic_inference_engine(controller, context)

    assert engine is handoff.return_value
    handoff.assert_called_once_with(controller, context)
    engine.setup_kv_transfer.assert_called_once_with(
        role=role, require_handoff_claim=True
    )
    plain.assert_not_called()


def test_engine_state_reset_clears_the_event_reporter_and_endpoint():
    worker = _bare_worker(role="prefill")
    worker._dynamo_event_reporter = MagicMock()
    worker._dynamo_endpoint_future = MagicMock()
    worker._dynamo_endpoint_metadata = {"event_port": 1, "handoff_owner": "o"}

    worker._init_inference_engine_state()

    assert worker._dynamo_event_reporter is None
    assert worker._dynamo_endpoint_future is None
    assert worker._dynamo_endpoint_metadata is None
    assert worker.dynamo_endpoint_metadata() is None
    assert worker.dynamic_inference_engine is None
    assert worker.coordinator_addr is None
    assert worker.dynamo_coordinator_address() is None


def test_pinned_handoff_block_count_is_zero_without_an_engine():
    worker = _bare_worker(role="prefill")
    worker._init_inference_engine_state()
    worker._inference_loop = None

    assert worker.pinned_handoff_block_count() == 0


def test_pinned_handoff_block_count_is_read_on_the_inference_loop(dynamo_endpoint_env):
    env = dynamo_endpoint_env
    reads = []

    class Engine:
        @property
        def pinned_handoff_block_count(self):
            reads.append(asyncio.get_running_loop())
            return 3

    env.worker.dynamic_inference_engine = Engine()

    assert env.worker.pinned_handoff_block_count() == 3
    # Read on the loop the engine mutates the pinned map from.
    assert reads == [env.worker._inference_loop]


def test_attach_dynamo_events_needs_a_built_engine():
    worker = _bare_worker(role="prefill")
    worker._init_inference_engine_state()
    worker._inference_engine_initialized = False

    with pytest.raises(RuntimeError, match="engine to be built"):
        worker.attach_dynamo_events(parent_event_address="tcp://10.0.0.5:4001")


# ---------------------------------------------------------------------------
# Rank zero hosts Megatron-LM's Dynamo adapter in-process
# ---------------------------------------------------------------------------


class _LoopThread:
    """A running asyncio loop on a thread, standing in for the worker's inference loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        async def _drain() -> None:
            # Run every cancelled or still-running task to completion, so none is
            # garbage-collected pending (and logged) during a later test.
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(_drain(), self.loop).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()


@pytest.fixture
def dynamo_endpoint_env(monkeypatch):
    """Fake ai-dynamo and Megatron-LM Dynamo modules, plus a bare rank-zero worker.

    Yields a namespace with the worker, the fakes' call records, and a knob that
    makes the fake ``Worker.run`` return (``finish``) or raise (``fail``).
    """
    records = SimpleNamespace(
        configs=[],
        engines=[],
        workers=[],
        fetched=[],
        attested=[],
        run_started=None,
        run_cancelled=False,
        reporters=[],
    )
    control = SimpleNamespace(finish=None, fail=None)

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            records.configs.append(self)

    class MegatronLLMEngine:
        def __init__(self, config, registration_model=None):
            self.config = config
            self.registration_model = registration_model
            records.engines.append(self)

        @staticmethod
        def worker_config(config, registration_model=None):
            return ("worker-config", config.role, registration_model)

    class Worker:
        def __init__(self, engine, config):
            self.engine = engine
            self.config = config
            records.workers.append(self)

        async def run(self):
            records.run_started = asyncio.get_running_loop()
            try:
                while control.finish is None and control.fail is None:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                records.run_cancelled = True
                raise
            if control.fail is not None:
                raise control.fail
            return None

    async def fetch_model(model, ignore_weights=False):
        records.fetched.append((model, ignore_weights))
        return f"/snapshots/{model.replace('/', '--')}"

    class EngineEventReporter:
        def __init__(self, engine, parent_address):
            self.parent_address = parent_address
            self.events = []
            self.started_on = None
            self.stopped = False
            records.reporters.append(self)

        def start(self):
            self.started_on = asyncio.get_running_loop()

        def stop(self):
            self.stopped = True

        def observe(self, kind, payload):
            self.events.append((kind, payload))

    class InferenceEngineEndpoint:
        def __init__(self, address):
            self.address = address

        @classmethod
        def from_engine(cls, address, engine, logical_data_parallel_size):
            assert logical_data_parallel_size == 1
            return cls(address)

        def to_dict(self):
            return {"coordinator_address": self.address}

    async def confirm_terminated(address, owner, timeout=30.0):
        records.attested.append((address, owner, timeout))
        if address.endswith(":dead"):
            raise ConnectionError(address)

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("dynamo")
    module("dynamo.common")
    module("dynamo.common.backend", Worker=Worker)
    module("dynamo.common.model_fetch", fetch_model=fetch_model)
    module("megatron.inference.integrations")
    module("megatron.inference.integrations.dynamo")
    module("megatron.inference.integrations.dynamo.args", Config=Config)
    module(
        "megatron.inference.integrations.dynamo.llm_engine",
        MegatronLLMEngine=MegatronLLMEngine,
    )
    module(
        "megatron.inference.integrations.dynamo.handoff_recovery",
        confirm_terminated=confirm_terminated,
    )
    module(
        "megatron.inference.integrations.dynamo.telemetry",
        EngineEventReporter=EngineEventReporter,
    )
    module(
        "megatron.core.inference.engine_endpoint",
        InferenceEngineEndpoint=InferenceEngineEndpoint,
    )
    ports = iter(range(4000, 4100))

    def free_port(low, high, *, max_retries=None, excluded_ports=()):
        while True:
            port = next(ports)
            if port not in excluded_ports:
                return port

    worker_module = "nemo_rl.models.generation.megatron.dynamo_worker_mixin"
    monkeypatch.setattr(f"{worker_module}._get_node_ip_local", lambda: "10.0.0.5")
    monkeypatch.setattr(f"{worker_module}._get_free_port_local", free_port)
    # start_dynamo_endpoint writes the manager env and DYN_SYSTEM_PORT into
    # os.environ; setenv records each key so teardown restores (or removes) it.
    for key in ("DYN_SYSTEM_PORT", "ETCD_ENDPOINTS", "DYN_NAMESPACE"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    loop_thread = _LoopThread()
    # A rank-zero worker whose engine the first refit already built.
    worker = object.__new__(_Worker)
    worker.rank = 0
    worker.cfg = {"generation": {"_dynamo_role": "prefill"}}
    worker._inference_engine_initialized = True
    worker._dynamo_endpoint_future = None
    worker._dynamo_endpoint_metadata = None
    worker._dynamo_event_reporter = None
    worker.coordinator_addr = "tcp://10.0.0.5:6000"
    worker.dynamic_inference_engine = MagicMock()
    worker._inference_loop = loop_thread.loop
    yield SimpleNamespace(worker=worker, records=records, control=control)
    worker.stop_dynamo_endpoint()
    loop_thread.close()


def _start(env, role="prefill", **overrides):
    env.worker.cfg["generation"]["_dynamo_role"] = role
    kwargs = dict(
        namespace="ns",
        manager_env={"ETCD_ENDPOINTS": "http://etcd:2379", "DYN_NAMESPACE": "ns"},
        model_name="Qwen/Qwen3-0.6B",
        served_model_name="Qwen/Qwen3-0.6B",
        endpoint_types=["completions"],
        engine_start_timeout_s=600.0,
        excluded_ports=[4000],
    )
    kwargs.update(overrides)
    return env.worker.start_dynamo_endpoint(**kwargs)


def _wait_until(predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_start_dynamo_endpoint_hosts_the_adapter_on_the_inference_loop(
    dynamo_endpoint_env, role
):
    env = dynamo_endpoint_env
    metadata = _start(env, role=role)

    # Ports skip what earlier engines on the node took; the event address is
    # what every rank's reporter connects to.
    assert metadata["node_ip"] == "10.0.0.5"
    assert metadata["event_port"] == 4001 and metadata["system_port"] == 4002
    assert metadata["parent_event_address"] == "tcp://10.0.0.5:4001"
    import os

    assert os.environ["DYN_SYSTEM_PORT"] == "4002"
    assert os.environ["ETCD_ENDPOINTS"] == "http://etcd:2379"

    # The adapter is Megatron-LM's own, in external mode: it starts no engine.
    _wait_until(lambda: env.records.run_started is not None)
    (config,) = env.records.configs
    assert config.engine_launch_mode == "external"
    assert config.role == role
    assert config.component == ("prefill" if role == "prefill" else "backend")
    assert config.endpoint == "generate" and config.endpoint_types == "completions"
    assert (config.parent_event_host, config.parent_event_port) == ("10.0.0.5", 4001)
    assert config.coordinator_host == "10.0.0.5"
    assert config.engine_start_timeout == 600.0
    assert config.namespace == "ns" and config.discovery_backend == "etcd"
    if role == "decode":
        assert config.handoff_owner == metadata["handoff_owner"]
        assert len(config.handoff_owner) == 32
    else:
        assert config.handoff_owner is None and metadata["handoff_owner"] is None
    # It runs on the inference loop, the loop the engine itself will use.
    assert env.records.run_started is env.worker._inference_loop
    assert not env.worker._dynamo_endpoint_future.done()
    assert env.worker.dynamo_endpoint_metadata() == metadata

    # A repeat start hands back the same endpoint and launches no second adapter.
    assert _start(env, role=role, excluded_ports=[]) == metadata
    assert len(env.records.configs) == 1


@pytest.mark.parametrize(
    ("model_name", "fetched", "registered"),
    [
        # A hub id registers as a metadata-only snapshot, never a second copy.
        pytest.param(
            "Qwen/Qwen3-0.6B",
            [("Qwen/Qwen3-0.6B", True)],
            "/snapshots/Qwen--Qwen3-0.6B",
            id="hub_id",
        ),
        pytest.param("{tmp_path}", [], "{tmp_path}", id="local_path"),
    ],
)
def test_dynamo_endpoint_registers_the_model_with_the_adapter(
    dynamo_endpoint_env, tmp_path, model_name, fetched, registered
):
    env = dynamo_endpoint_env
    local = str(tmp_path.resolve())
    _start(env, model_name=model_name.format(tmp_path=local))
    _wait_until(lambda: env.records.workers)

    assert env.records.fetched == fetched
    (engine,) = env.records.engines
    assert engine.registration_model == registered.format(tmp_path=local)
    (dynamo_worker,) = env.records.workers
    assert dynamo_worker.engine is engine
    assert dynamo_worker.config == (
        "worker-config",
        "prefill",
        engine.registration_model,
    )


def test_stop_dynamo_endpoint_cancels_the_adapter_task(dynamo_endpoint_env):
    env = dynamo_endpoint_env
    metadata = _start(env)
    future = env.worker._dynamo_endpoint_future
    _wait_until(lambda: env.records.run_started is not None)

    env.worker.attach_dynamo_events(
        parent_event_address=metadata["parent_event_address"]
    )
    (reporter,) = env.records.reporters

    env.worker.stop_dynamo_endpoint()

    # Worker.run itself is cancelled on the inference loop.
    _wait_until(lambda: env.records.run_cancelled)
    assert reporter.stopped and env.worker._dynamo_event_reporter is None
    assert env.worker.dynamo_endpoint_error() == "cancelled"
    # The stopped task stays recorded: Dynamo's runtime serves one Worker per
    # process, so a second start in this process is refused.
    assert env.worker._dynamo_endpoint_future is future
    with pytest.raises(RuntimeError, match=r"\(cancelled\).*restart the engine"):
        _start(env, excluded_ports=[])
    assert len(env.records.configs) == 1


def test_a_failed_endpoint_reports_its_error_and_refuses_a_restart(
    dynamo_endpoint_env,
):
    env = dynamo_endpoint_env
    metadata = _start(env)
    _wait_until(lambda: env.records.run_started is not None)
    assert env.worker.dynamo_endpoint_error() is None
    # A repeat start while it serves hands back the same endpoint.
    assert _start(env, excluded_ports=[]) == metadata

    env.control.fail = ConnectionError("etcd unreachable")
    _wait_until(env.worker._dynamo_endpoint_future.done)

    assert env.worker.dynamo_endpoint_error() == repr(env.control.fail)
    with pytest.raises(RuntimeError, match="etcd unreachable"):
        _start(env, excluded_ports=[])
    assert len(env.records.configs) == 1


def test_an_endpoint_that_returned_reports_exited(dynamo_endpoint_env):
    env = dynamo_endpoint_env
    _start(env)
    env.control.finish = True
    _wait_until(env.worker._dynamo_endpoint_future.done)

    assert env.worker.dynamo_endpoint_error() == "exited"
    with pytest.raises(RuntimeError, match=r"\(exited\)"):
        _start(env, excluded_ports=[])


def test_dynamo_endpoint_and_events_need_a_built_engine(dynamo_endpoint_env):
    env = dynamo_endpoint_env
    env.worker._inference_engine_initialized = False
    match = "needs the inference engine"
    with pytest.raises(RuntimeError, match=match):
        _start(env)
    with pytest.raises(RuntimeError, match=match):
        env.worker.attach_dynamo_events(parent_event_address="tcp://10.0.0.5:4001")
    assert env.records.configs == [] and env.records.reporters == []


def test_attach_dynamo_events_reports_ready_to_the_adapter_once(dynamo_endpoint_env):
    """Once the engine runs, rank zero tells the adapter where its coordinator is."""
    env = dynamo_endpoint_env
    metadata = _start(env)
    address = metadata["parent_event_address"]

    assert env.worker.attach_dynamo_events(parent_event_address=address) is None

    (reporter,) = env.records.reporters
    assert reporter.parent_address == address
    # Started on the inference loop, where the engine fires its callbacks.
    assert reporter.started_on is env.worker._inference_loop
    assert reporter.events == [
        ("ready", {"coordinator_address": "tcp://10.0.0.5:6000"})
    ]
    env.worker.dynamic_inference_engine.set_progress_callback.assert_called_once()
    # A second attach (a later refit, or the controller's copy) changes nothing.
    env.worker.attach_dynamo_events(parent_event_address=address)
    assert len(env.records.reporters) == 1 and len(reporter.events) == 1


def test_dynamo_worker_requires_the_megatron_dynamo_integration(monkeypatch):
    module = "megatron.inference.integrations.dynamo.dynamic_engine"
    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    require_megatron_dynamo_integration()

    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(RuntimeError, match="megatron.inference.integrations.dynamo"):
        require_megatron_dynamo_integration()


def test_attach_dynamo_events_retries_until_ready_and_pins_the_address(
    dynamo_endpoint_env,
):
    """A failed "ready" leaves nothing attached; a re-attach elsewhere is refused."""
    env = dynamo_endpoint_env
    address = _start(env)["parent_event_address"]

    # The coordinator is not listening yet, so "ready" cannot be sent.
    env.worker.coordinator_addr = None
    with pytest.raises(RuntimeError, match="not listening"):
        env.worker.attach_dynamo_events(parent_event_address=address)
    assert env.worker._dynamo_event_reporter is None
    assert env.records.reporters[0].stopped

    env.worker.coordinator_addr = "tcp://10.0.0.5:6000"
    env.worker.attach_dynamo_events(parent_event_address=address)
    assert env.worker._dynamo_event_reporter is env.records.reporters[1]
    assert [kind for kind, _ in env.records.reporters[1].events] == ["ready"]

    env.worker.attach_dynamo_events(parent_event_address=address)
    assert len(env.records.reporters) == 2
    with pytest.raises(ValueError, match="tcp://10.0.0.5:9999"):
        env.worker.attach_dynamo_events(parent_event_address="tcp://10.0.0.5:9999")


def test_attach_dynamo_events_on_a_non_leader_rank_sends_no_ready(
    dynamo_endpoint_env, monkeypatch
):
    """Only rank zero reports "ready"; other ranks attach without a coordinator."""
    env = dynamo_endpoint_env
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    env.worker.coordinator_addr = None

    env.worker.attach_dynamo_events(parent_event_address="tcp://10.0.0.5:4001")

    (reporter,) = env.records.reporters
    assert reporter.events == []
    assert env.worker._dynamo_event_reporter is reporter


@pytest.mark.parametrize(
    ("addresses", "confirmed"),
    [
        pytest.param([], True, id="no_prefill_coordinators"),
        pytest.param(
            ["tcp://10.0.0.1:5555", "tcp://10.0.0.2:5555"], True, id="every_one"
        ),
        # One unreachable coordinator makes the attestation False, not an exception.
        pytest.param(
            ["tcp://10.0.0.1:5555", "tcp://10.0.0.3:dead"], False, id="unreachable"
        ),
    ],
)
def test_confirm_decode_terminated_attests_to_every_prefill_coordinator(
    dynamo_endpoint_env, addresses, confirmed
):
    env = dynamo_endpoint_env
    assert env.worker.confirm_decode_terminated(addresses, "owner-1", 3.0) is confirmed
    assert env.records.attested == [(address, "owner-1", 3.0) for address in addresses]


# ---------------------------------------------------------------------------
# Rank zero serves NeMo-Gym from Megatron-LM's endpoints over the frontend
# ---------------------------------------------------------------------------


def test_setup_openai_api_server_needs_the_frontend_url():
    worker = _bare_worker(role="prefill")
    worker._inference_loop = object()

    with pytest.raises(RuntimeError, match="configure_dynamo_gym_proxy"):
        worker._setup_openai_api_server()


@pytest.mark.parametrize("reserved_port", [5000, None])
def test_setup_openai_api_server_serves_gym_over_the_frontend(
    monkeypatch, reserved_port
):
    import asyncio

    from nemo_rl.distributed import held_port
    from nemo_rl.models.generation.megatron import dynamo_gym_proxy as proxy

    records = SimpleNamespace(received=[], bound=[], apps=[], served=[], healthy=[])

    class _Sock:
        def __init__(self, port):
            self.port = port or 6000
            self.closed = False

        def getsockname(self):
            return ("0.0.0.0", self.port)

        def close(self):
            self.closed = True

    def receive_held_socket(port):
        records.received.append(port)
        return _Sock(port)

    def bind_server_socket(port):
        records.bound.append(port)
        return _Sock(port)

    def build_gym_proxy_app(client, **kwargs):
        records.apps.append((client, kwargs))
        return "app"

    async def serve_app(app, sock):
        records.served.append((app, sock, asyncio.get_running_loop()))
        await asyncio.Event().wait()

    def wait_until_healthy(base_url, *, server, timeout_s):
        records.healthy.append((base_url, server, timeout_s))

    monkeypatch.setattr(held_port, "receive_held_socket", receive_held_socket)
    monkeypatch.setattr(proxy, "bind_server_socket", bind_server_socket)
    monkeypatch.setattr(proxy, "build_gym_proxy_app", build_gym_proxy_app)
    monkeypatch.setattr(proxy, "serve_app", serve_app)
    monkeypatch.setattr(proxy, "wait_until_healthy", wait_until_healthy)
    monkeypatch.setattr(dynamo_worker_mixin, "_get_node_ip_local", lambda: "10.0.0.5")

    loop_thread = _LoopThread()
    worker = _bare_worker(role="decode")
    worker.rank = 0
    worker.cfg["generation"].update(
        model_name="Qwen/Qwen3-0.6B",
        temperature=0.6,
        top_p=0.95,
        top_k=None,
        mcore_generation_config={"parsers": ["nano_v3"], "expose_http_server": True},
    )
    worker.megatron_tokenizer = "tok"
    worker.inference_wrapped_model = SimpleNamespace(multimodal_prompt_config="mm")
    worker._reserved_http_server_port = reserved_port
    worker._inference_loop = loop_thread.loop
    worker._dynamo_endpoint_future = None
    worker._dynamo_event_reporter = None
    try:
        worker.configure_dynamo_gym_proxy(
            frontend_url="http://10.0.0.1:3000/v1", request_timeout_s=120.0
        )
        base_url = worker._setup_openai_api_server()

        if reserved_port is not None:
            # The driver's reservation is adopted: its socket is taken (and
            # released), and the server binds the same port.
            assert records.received == [5000] and records.bound == [5000]
            assert base_url == "http://10.0.0.5:5000/v1"
        else:
            assert records.received == [] and records.bound == [0]
            assert base_url == "http://10.0.0.5:6000/v1"
        ((client, app_kwargs),) = records.apps
        assert client.completions_url == "http://10.0.0.1:3000/v1/completions"
        assert client._request_timeout_s == 120.0
        assert client._model_name == "Qwen/Qwen3-0.6B"
        assert app_kwargs == {
            "tokenizer": "tok",
            "parsers": ["nano_v3"],
            "multimodal_prompt_config": "mm",
            "default_temperature": 0.6,
            "default_top_p": 0.95,
            "default_top_k": 0,
        }
        _wait_until(lambda: records.served)
        ((app, sock, loop),) = records.served
        assert app == "app" and loop is loop_thread.loop
        assert sock.port == (reserved_port or 6000)
        ((healthy_url, server, timeout_s),) = records.healthy
        assert healthy_url == base_url
        assert server is worker._dynamo_gym_proxy_future
        assert timeout_s == dynamo_worker_mixin.GYM_PROXY_START_TIMEOUT_S
        assert not server.done()

        # Stopping the endpoint stops the proxy server with it.
        worker.stop_dynamo_endpoint()
        _wait_until(lambda: server.done())
        assert server.cancelled()
        # Let the cancelled server task unwind before the loop closes.
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop_thread.loop).result()
    finally:
        loop_thread.close()
