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
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load the real lightweight module without the generation package's GPU imports.
G_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/generation/vllm/output_loop_yield.py"
)
G_SPEC = importlib.util.spec_from_file_location(
    "_vllm_output_loop_yield_test", G_SOURCE
)
assert G_SPEC is not None and G_SPEC.loader is not None
output_loop_yield = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(output_loop_yield)
install_output_loop_yield = output_loop_yield.install_output_loop_yield


class _FakeAsyncMPClient:
    async def get_output_async(self):
        return "outputs"


class _FakeSocket:
    """Mimics zmq.asyncio's recv_multipart: returns a future, done if a message waits."""

    def __init__(self):
        self.message_waiting = True
        self.calls = 0

    def recv_multipart(self, *args, **kwargs):
        self.calls += 1
        future = asyncio.get_running_loop().create_future()
        if self.message_waiting:
            future.set_result(["frame"])
        else:
            asyncio.get_running_loop().call_soon(future.set_result, ["frame"])
        return future


@pytest.fixture
def fake_core_client(monkeypatch):
    """Stand-in for vllm.v1.engine.core_client with a fresh AsyncMPClient class."""
    client_cls = type("AsyncMPClient", (_FakeAsyncMPClient,), {})
    module = types.ModuleType("vllm.v1.engine.core_client")
    module.AsyncMPClient = client_cls
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core_client", module)
    return client_cls


def _fake_llm():
    socket = _FakeSocket()
    llm = SimpleNamespace(
        engine_core=SimpleNamespace(resources=SimpleNamespace(output_socket=socket))
    )
    return llm, socket


def _count_sleeps(monkeypatch):
    sleeps = []
    real_sleep = output_loop_yield.asyncio.sleep

    async def counting_sleep(delay):
        sleeps.append(delay)
        await real_sleep(delay)

    monkeypatch.setattr(output_loop_yield.asyncio, "sleep", counting_sleep)
    return sleeps


@pytest.mark.asyncio
async def test_output_handler_yields_once_per_step(fake_core_client, monkeypatch):
    llm, _ = _fake_llm()
    install_output_loop_yield(llm, reader_budget_s=0.02)
    sleeps = _count_sleeps(monkeypatch)

    assert await fake_core_client().get_output_async() == "outputs"
    assert sleeps == [0]


def test_class_patch_is_installed_once(fake_core_client):
    install_output_loop_yield(_fake_llm()[0], reader_budget_s=0.02)
    patched = fake_core_client.get_output_async
    install_output_loop_yield(_fake_llm()[0], reader_budget_s=0.02)

    assert fake_core_client.get_output_async is patched


@pytest.mark.asyncio
async def test_reader_yields_only_after_budget(fake_core_client, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(output_loop_yield.time, "monotonic", lambda: now[0])
    llm, socket = _fake_llm()
    install_output_loop_yield(llm, reader_budget_s=0.02)
    sleeps = _count_sleeps(monkeypatch)
    recv = llm.engine_core.resources.output_socket.recv_multipart

    # Messages already waiting: each receive returns a done future.
    now[0] += 0.01
    assert await recv() == ["frame"]
    assert sleeps == []  # 10 ms of continuous work, within the 20 ms budget

    now[0] += 0.015
    await recv()
    assert sleeps == [0]  # 25 ms since the budget started: yield and restart it

    now[0] += 0.01
    await recv()
    assert sleeps == [0]
    assert socket.calls == 3


@pytest.mark.asyncio
async def test_reader_budget_restarts_after_a_real_wait(fake_core_client, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(output_loop_yield.time, "monotonic", lambda: now[0])
    llm, socket = _fake_llm()
    install_output_loop_yield(llm, reader_budget_s=0.02)
    sleeps = _count_sleeps(monkeypatch)
    recv = llm.engine_core.resources.output_socket.recv_multipart

    # The socket is empty: the receive's future is pending, so the loop gets its turn.
    socket.message_waiting = False
    now[0] += 0.05
    assert await recv() == ["frame"]
    assert sleeps == []

    # The budget restarted when that message arrived, not when the reader started.
    socket.message_waiting = True
    now[0] += 0.01
    await recv()
    assert sleeps == []


def test_missing_vllm_internals_fail_loudly(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.core_client",
        types.ModuleType("vllm.v1.engine.core_client"),
    )
    with pytest.raises(RuntimeError, match="output_loop_yield_budget_ms"):
        install_output_loop_yield(_fake_llm()[0], reader_budget_s=0.02)


def test_missing_output_socket_fails_loudly(fake_core_client):
    with pytest.raises(RuntimeError, match="output_loop_yield_budget_ms"):
        install_output_loop_yield(SimpleNamespace(), reader_budget_s=0.02)


def test_non_positive_budget_is_rejected(fake_core_client):
    with pytest.raises(ValueError, match="positive"):
        install_output_loop_yield(_fake_llm()[0], reader_budget_s=0)


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_pinned_vllm_output_loops_match_the_patch() -> None:
    """Fail if the pinned vLLM moves the output-loop calls this patch relies on."""
    core_client = pytest.importorskip("vllm.v1.engine.core_client")
    async_llm = pytest.importorskip("vllm.v1.engine.async_llm")
    zmq = pytest.importorskip("zmq")
    zmq_asyncio = pytest.importorskip("zmq.asyncio")

    # output_handler awaits get_output_async once per engine step.
    assert inspect.iscoroutinefunction(core_client.AsyncMPClient.get_output_async)
    output_handler = inspect.getsource(async_llm.AsyncLLM._run_output_handler)
    assert "engine_core.get_output_async()" in output_handler, (
        "vLLM's output_handler no longer awaits get_output_async; "
        "the output_handler yield is dead"
    )

    # The socket reader looks recv_multipart up on the client's output socket on
    # every message, so the per-instance override takes effect.
    reader = inspect.getsource(core_client.AsyncMPClient._ensure_output_queue_task)
    assert "resources.output_socket" in reader
    assert "await output_socket.recv_multipart(" in reader, (
        "vLLM's output-socket reader no longer awaits output_socket.recv_multipart; "
        "the reader budget is dead"
    )

    # The reader budget tells a waited receive from a waiting message by future.done().
    context = zmq_asyncio.Context()
    socket = context.socket(zmq.PULL)
    try:
        future = socket.recv_multipart(copy=False)
        assert asyncio.isfuture(future)
        assert not future.done()
        future.cancel()
    finally:
        socket.close(linger=0)
        context.term()
