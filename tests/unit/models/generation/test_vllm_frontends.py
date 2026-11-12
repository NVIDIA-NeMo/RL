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
"""CPU frontends of a vLLM engine (``vllm_cfg.num_frontend_workers``), without vLLM or Ray."""

import asyncio
import os
import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncFrontendWorkerImpl,
    VllmAsyncGenerationWorkerImpl,
)

WORKER_MODULE = "nemo_rl.models.generation.vllm.vllm_worker_async"


def _cfg(**vllm_cfg):
    return {"vllm_cfg": {"async_engine": True, "expose_http_server": True, **vllm_cfg}}


class _Actor:
    """A Ray actor handle that logs its calls; a call's future is its result."""

    def __init__(self, name="", calls=None, **results):
        self.name, self.results = name, results
        self.calls = [] if calls is None else calls

    def _future(self, result):
        return result

    def __getattr__(self, method):
        def remote(**kwargs):
            self.calls.append((self.name, method, kwargs))
            return self._future(self.results.get(method))

        return types.SimpleNamespace(remote=remote)


class _AsyncActor(_Actor):
    async def _future(self, result):
        return result


def _logger_report(inflight, changed_at):
    return {
        "inflight_batch_sizes": [inflight],
        "num_pending_samples": [0],
        "kv_cache_usage_perc": [0.5],
        "generation_tokens": [7],
        "gauges_changed_at": changed_at,
    }


@pytest.fixture
def generation(monkeypatch):
    """Two engines with two frontends each, without Ray."""
    monkeypatch.setattr("ray.get", lambda futures, **kwargs: futures)
    monkeypatch.setattr("ray.wait", lambda refs, **kwargs: (refs, []))
    gen = VllmGeneration.__new__(VllmGeneration)
    gen.cfg = _cfg(enable_vllm_metrics_logger=True)
    gen.shutdown = lambda: True  # called by __del__
    gen._frontends = [
        [
            _Actor(
                get_reserved_url=f"http://node{engine}:{9000 + index}/v1",
                _get_raw_spec_counters={"vllm:generation_tokens": 10.0},
                drain_latest_vllm_logger_metrics=_logger_report(
                    100 * (index + 1), float(index)
                ),
                get_data_plane_snapshot={"puts": index} if index else None,
                captured_call_progress={"weight_version": 3, "calls": [[index, 5]]},
            )
            for index in range(2)
        ]
        for engine in range(2)
    ]
    owners = [_Actor(_get_raw_spec_counters={"vllm:generation_tokens": 1.0})] * 2
    gen.worker_group = types.SimpleNamespace(
        workers=owners,
        run_all_workers_single_data=lambda method, **kwargs: [
            getattr(owner, method).remote() for owner in owners
        ],
    )
    return gen


def test_gym_gets_one_url_per_frontend_engine_by_engine(generation):
    assert generation._collect_reserved_urls() == [
        "http://node0:9000/v1",
        "http://node0:9001/v1",
        "http://node1:9000/v1",
        "http://node1:9001/v1",
    ]


def test_capture_and_the_weight_version_go_to_the_frontends(generation):
    generation.set_rollout_weight_version(3)
    generation.setup_token_capture({"backend": "simple"}, "staging", top_logprobs=32)

    for frontend in [f for engine in generation._frontends for f in engine]:
        (_, weight, version), (_, capture, kwargs) = frontend.calls
        assert (weight, version) == ("set_rollout_weight_version", {"version": 3})
        assert capture == "setup_token_capture" and kwargs["top_logprobs"] == 32
    assert generation.worker_group.workers[0].calls == []


@pytest.mark.parametrize(
    ("dp_cfg", "options", "unsupported"),
    [
        ({"backend": "simple"}, {"capture_media": True}, "capture_media"),
        (
            {"backend": "simple"},
            {"generation_prefix_cuts_enabled": True},
            "generation-prefix cuts",
        ),
        ({"backend": "mooncake_cpu"}, {}, "mooncake_cpu"),
    ],
)
def test_capture_the_frontends_cannot_host_is_refused(
    generation, dp_cfg, options, unsupported
):
    with pytest.raises(ValueError, match=unsupported):
        generation.setup_token_capture(dp_cfg, "staging", **options)


def test_frontends_that_are_slow_to_answer_are_named(generation, monkeypatch, capsys):
    flat = [frontend for engine in generation._frontends for frontend in engine]
    for ref, frontend in enumerate(flat):
        frontend.results["set_rollout_weight_version"] = ref
    # Frontend 0.1 answers one wait later than the others.
    rounds = iter([([0, 2, 3], [1]), ([1], [])])
    monkeypatch.setattr("ray.wait", lambda pending, **kwargs: next(rounds))

    generation.set_rollout_weight_version(4)

    out = capsys.readouterr().out
    assert "['0.1']" in out and "set_rollout_weight_version" in out


def test_engine_metrics_are_gathered_from_the_frontends(generation):
    # Two owners at 1 each and four frontends at 10 each.
    assert generation._get_raw_spec_counters() == {"vllm:generation_tokens": 42.0}
    assert generation.drain_latest_vllm_logger_metrics() == {
        "inflight_batch_sizes": {0: [200], 1: [200]},
        "num_pending_samples": {0: [0], 1: [0]},
        "kv_cache_usage_perc": {0: [0.5], 1: [0.5]},
        "generation_tokens": {0: [14], 1: [14]},
    }
    assert generation.get_data_plane_snapshots() == [{"puts": 1}] * 2
    assert [p["calls"] for p in generation.captured_call_progress()] == [
        [[0, 5]],
        [[1, 5]],
    ] * 2


def test_fleet_health_is_refused_with_frontends():
    master_config = types.SimpleNamespace(
        policy={"generation": _cfg(num_frontend_workers=2)},
        async_rl=types.SimpleNamespace(
            generation_fleet_health=types.SimpleNamespace(enabled=True)
        ),
    )
    with pytest.raises(ValueError, match="num_frontend_workers"):
        VllmGeneration.validate_settings(master_config)


def _engine_owner():
    """An engine owner with two frontends, and the log of their calls."""
    log = []
    owner = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    owner.cfg = _cfg()
    owner.llm = MagicMock(collective_rpc=AsyncMock(), vllm_config="vllm-config")
    owner.llm.model_config.multimodal_config = None
    owner.llm_async_engine_args = "engine-args"
    owner.routed_experts_dtype = "int16"
    owner.report_device_id_async = AsyncMock()
    owner._sparse_refit_receiver = owner.server_thread = None
    owner._mtp_speculative_enabled = owner._mtp_load_from_disk = False
    owner._setup_vllm_server = MagicMock()
    owner._frontends = [
        _AsyncActor(f"fe{i}", log, connect=f"http://node:{9000 + i}/v1")
        for i in range(2)
    ]
    owner._frontend_client_addresses = [
        {"input_address": f"ipc:///tmp/in{i}", "output_address": f"ipc:///tmp/out{i}"}
        for i in (1, 2)
    ]
    return owner, log


@pytest.mark.asyncio
async def test_an_engine_owner_connects_its_frontends_and_serves_nothing(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    owner, log = _engine_owner()

    await owner.post_init_async()

    assert [call[:2] for call in log] == [("fe0", "connect"), ("fe1", "connect")]
    for index, (_, _, kwargs) in enumerate(log):
        assert kwargs == {
            "vllm_config": "vllm-config",
            "engine_args": "engine-args",
            "routed_experts_dtype": "int16",
            "client_addresses": owner._frontend_client_addresses[index],
            "client_count": 3,
            "client_index": index + 1,  # the owner is client 0
            "env": {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
        }
    owner._setup_vllm_server.assert_not_called()


@pytest.mark.asyncio
async def test_a_frontend_that_does_not_connect_fails_the_engine_start(monkeypatch):
    monkeypatch.setattr(f"{WORKER_MODULE}._FRONTEND_CONNECT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(f"{WORKER_MODULE}._get_node_ip_local", lambda: "10.0.0.7")
    owner, _ = _engine_owner()

    async def refused(**kwargs):
        raise RuntimeError("Traceback ...\nRuntimeError: no ready message")

    owner._frontends = [
        types.SimpleNamespace(connect=types.SimpleNamespace(remote=remote))
        for remote in (refused, lambda **kwargs: asyncio.sleep(3600))
    ]
    with pytest.raises(RuntimeError) as failure:
        await owner.post_init_async()

    message = str(failure.value)
    assert "10.0.0.7" in message and f"pid {os.getpid()}" in message
    assert "ipc:///tmp/in1: RuntimeError: no ready message" in message
    assert "ipc:///tmp/in2: no answer in 0 s" in message
    assert isinstance(failure.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_multimodal_engines_are_refused_before_any_frontend_connects():
    owner, log = _engine_owner()
    owner.llm.model_config.multimodal_config = object()
    with pytest.raises(RuntimeError, match="multimodal"):
        await owner.post_init_async()

    owner.cfg["vllm_kwargs"] = {"language_model_only": True}
    await owner.post_init_async()
    assert len(log) == 2


@pytest.mark.asyncio
async def test_an_engine_owner_stops_its_frontends_before_its_engine(monkeypatch):
    owner, log = _engine_owner()
    owner.llm.shutdown = lambda: log.append(("engine", "shutdown", {}))
    monkeypatch.setattr(f"{WORKER_MODULE}.shutdown_telemetry", lambda: None)
    monkeypatch.setattr("ray.kill", lambda actor: log.append((actor.name, "kill", {})))

    assert await owner.shutdown()

    assert [call[:2] for call in log] == [
        ("fe0", "shutdown"),
        ("fe1", "shutdown"),
        ("fe0", "kill"),
        ("fe1", "kill"),
        ("engine", "shutdown"),
    ]


@pytest.fixture
def vllm_client_modules(monkeypatch):
    """Stand-ins for the vLLM modules a frontend imports to attach."""
    stubs = {
        "vllm.v1.engine": {},
        "vllm.v1.engine.core_client": {"VLLM_ENGINE_READY_TIMEOUT_S": 3600},
        "vllm.v1.engine.async_llm": {"AsyncLLM": MagicMock()},
        "vllm.v1.metrics.loggers": {"PrometheusStatLogger": None},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return sys.modules


@pytest.mark.asyncio
async def test_a_frontend_attaches_as_a_client_of_the_engine(
    monkeypatch, vllm_client_modules
):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    async_llm = vllm_client_modules["vllm.v1.engine.async_llm"].AsyncLLM
    frontend = VllmAsyncFrontendWorkerImpl.__new__(VllmAsyncFrontendWorkerImpl)
    frontend.cfg = _cfg()
    frontend._setup_vllm_server = lambda: ("thread", "http://node:9001/v1", "server")
    pair = {"input_address": "ipc:///tmp/in2", "output_address": "ipc:///tmp/out2"}
    vllm_config = MagicMock()
    connect = dict(
        vllm_config=vllm_config,
        engine_args=types.SimpleNamespace(
            disable_log_stats=False, enable_log_requests=False
        ),
        routed_experts_dtype="int16",
        client_addresses=pair,
        client_count=3,
        client_index=2,
        env={"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
    )

    url = await frontend.connect(**connect)

    assert url == frontend.base_url == "http://node:9001/v1"
    async_llm.from_vllm_config.assert_called_once_with(
        vllm_config,
        stat_loggers=[],
        enable_log_requests=False,
        disable_log_stats=False,
        client_addresses=pair,
        client_count=3,
        client_index=2,
    )
    assert vllm_config.cache_config.num_gpu_blocks is None
    core_client = vllm_client_modules["vllm.v1.engine.core_client"]
    assert core_client.VLLM_ENGINE_READY_TIMEOUT_S == 120
    assert frontend._engine_loop is asyncio.get_running_loop()
    assert frontend.routed_experts_dtype == "int16"
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    with pytest.raises(RuntimeError, match="loads no model"):
        frontend.load_model()

    async_llm.from_vllm_config.side_effect = TimeoutError
    with pytest.raises(RuntimeError, match=r"frontend 2\b.*120 s.*ipc:///tmp/in2"):
        await frontend.connect(**connect)


def test_a_frontend_stamps_when_its_engine_gauges_last_changed(monkeypatch):
    clock = iter([100.0, 300.0])
    monkeypatch.setattr("time.time", lambda: next(clock))
    frontend = VllmAsyncFrontendWorkerImpl.__new__(VllmAsyncFrontendWorkerImpl)
    frontend.cfg = _cfg(enable_vllm_metrics_logger=True)
    frontend._vllm_metrics_lock = threading.Lock()
    frontend._last_logger_gauges = None
    frontend._logger_gauges_changed_at = 0.0
    frontend.inflight_batch_sizes = [5]
    frontend.num_pending_samples = [0]
    frontend.kv_cache_usage_perc = [0.5]
    frontend.generation_tokens = [10]

    assert frontend.drain_latest_vllm_logger_metrics()["gauges_changed_at"] == 100.0
    frontend.generation_tokens.append(20)
    assert frontend.drain_latest_vllm_logger_metrics()["gauges_changed_at"] == 100.0
    frontend.kv_cache_usage_perc.append(0.6)
    assert frontend.get_vllm_logger_metrics()["gauges_changed_at"] == 300.0
