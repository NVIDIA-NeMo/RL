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

import sys
import types
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm import output_loop_yield
from nemo_rl.models.generation.vllm.output_loop_yield import install_output_loop_yield


class _FakeAsyncMPClient:
    async def get_output_async(self):
        return "outputs"


class _FakeSocket:
    def __init__(self):
        self.calls = 0

    async def recv_multipart(self, *args, **kwargs):
        self.calls += 1
        return ["frame"]


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

    # Messages already waiting: no time passes inside the receive itself.
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

    async def slow_recv(*args, **kwargs):
        now[0] += 0.05  # the receive itself waited for a message
        return ["frame"]

    socket.recv_multipart = slow_recv
    install_output_loop_yield(llm, reader_budget_s=0.02)
    sleeps = _count_sleeps(monkeypatch)

    await llm.engine_core.resources.output_socket.recv_multipart()
    assert sleeps == []  # the loop already got its turn while the receive waited


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
