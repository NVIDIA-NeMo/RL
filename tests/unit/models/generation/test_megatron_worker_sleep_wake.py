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
"""The Megatron worker's suspend/resume, against a fake engine with Megatron's state rules.

The fake engine follows Megatron-LM's dynamic engine: PAUSE is ignored outside
RUNNING, UNPAUSE and SUSPEND need PAUSED, RESUME needs SUSPENDED, a broken rule
ends the engine loop, and RESUMED is an event the engine sets on its way back to
PAUSED. The worker's real ``suspend_for_refit`` / ``resume_after_refit`` (and so
``_sleep`` / ``_wake``) run on a real event loop thread, with rank zero and the
barrier stubbed.
"""

import asyncio
import threading
from collections import deque

import pytest
import torch
from megatron.core.inference.engines.dynamic_engine import EngineState

from nemo_rl.models.generation.megatron.megatron_worker import (
    MegatronGenerationMixin,
    MegatronGenerationRefitMixin,
)

pytestmark = pytest.mark.mcore

_STABLE = (
    EngineState.RUNNING,
    EngineState.PAUSED,
    EngineState.SUSPENDED,
    EngineState.RESUMED,
    EngineState.STOPPED,
)
# Every test finishes far sooner; a hang fails the test instead of the run.
_BOUND_S = 10.0


class _Worker(MegatronGenerationMixin, MegatronGenerationRefitMixin):
    """The generation and refit halves of a Megatron worker, without the model."""


class _FakeEngine:
    """Megatron's engine state machine: one signal, then one transition, per loop step."""

    def __init__(self) -> None:
        self.state = EngineState.RUNNING
        self._state_events = {state: asyncio.Event() for state in _STABLE}
        self._state_events[EngineState.RUNNING].set()
        self.pending: deque[str] = deque()
        self.engine_loop_task = asyncio.get_running_loop().create_task(self._run())

    async def wait_until(self, state: EngineState) -> None:
        await self._state_events[state].wait()

    def _apply(self, header: str) -> None:
        if header == "PAUSE":
            if self.state == EngineState.RUNNING:
                self.state = EngineState.PAUSING
                self._state_events[EngineState.RUNNING].clear()
        elif header == "UNPAUSE":
            assert self.state == EngineState.PAUSED, f"UNPAUSE in {self.state}"
            self.state = EngineState.UNPAUSING
        elif header == "SUSPEND":
            assert self.state == EngineState.PAUSED, f"SUSPEND in {self.state}"
            self._state_events[EngineState.RESUMED].clear()
            self.state = EngineState.SUSPENDING
        elif header == "RESUME":
            assert self.state == EngineState.SUSPENDED, f"RESUME in {self.state}"
            self._state_events[EngineState.SUSPENDED].clear()
            self.state = EngineState.RESUMING

    def _transition(self) -> None:
        if self.state == EngineState.PAUSING:
            self.state = EngineState.PAUSED
            self._state_events[EngineState.PAUSED].set()
        elif self.state == EngineState.UNPAUSING:
            self.state = EngineState.RUNNING
            self._state_events[EngineState.PAUSED].clear()
            self._state_events[EngineState.RUNNING].set()
        elif self.state == EngineState.SUSPENDING:
            self.state = EngineState.SUSPENDED
            self._state_events[EngineState.SUSPENDED].set()
        elif self.state == EngineState.RESUMING:
            self.state = EngineState.PAUSED
            self._state_events[EngineState.RESUMED].set()

    async def _run(self) -> None:
        while True:
            if self.pending:
                self._apply(self.pending.popleft())
            await asyncio.sleep(0)
            self._transition()
            await asyncio.sleep(0)


class _FakeClient:
    """Rank zero's inference client: each call queues one signal on the engine."""

    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine
        self.sent: list[str] = []
        self.fail_on: str | None = None

    def _send(self, header: str) -> None:
        if header == self.fail_on:
            raise ConnectionError(f"{header} could not be sent")
        self.sent.append(header)
        self.engine.pending.append(header)

    def pause_engines(self) -> None:
        self._send("PAUSE")

    def unpause_engines(self) -> None:
        self._send("UNPAUSE")

    def suspend_engines(self) -> None:
        self._send("SUSPEND")

    def resume_engines(self) -> None:
        self._send("RESUME")


@pytest.fixture
def worker(monkeypatch):
    """A Megatron generation worker with a serving fake engine on its own loop thread."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args: None)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def make_engine():
        return _FakeEngine()

    engine = asyncio.run_coroutine_threadsafe(make_engine(), loop).result(_BOUND_S)
    worker = object.__new__(_Worker)
    worker.rank = 0
    worker.dynamic_inference_engine = engine
    worker.inference_client = _FakeClient(engine)
    worker._inference_loop = loop
    worker._inference_engine_initialized = True
    worker._inference_engine_asleep = False
    yield worker
    loop.call_soon_threadsafe(loop.stop)
    thread.join(_BOUND_S)


def _bounded(fn) -> None:
    """Run a blocking worker call on its own thread so a hang fails the test."""
    result: list[BaseException | None] = []

    def target():
        try:
            fn()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            result.append(error)
        else:
            result.append(None)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(_BOUND_S)
    assert result, f"{fn.__name__} hung"
    if result[0] is not None:
        raise result[0]


def _state(worker) -> EngineState:
    return worker.dynamic_inference_engine.state


def test_suspend_then_resume_round_trip(worker):
    _bounded(worker.suspend_for_refit)
    assert _state(worker) == EngineState.SUSPENDED
    assert worker._inference_engine_asleep
    _bounded(worker.resume_after_refit)
    assert _state(worker) == EngineState.RUNNING
    assert not worker._inference_engine_asleep
    assert worker.inference_client.sent == ["PAUSE", "SUSPEND", "RESUME", "UNPAUSE"]


def test_a_suspend_that_failed_after_pause_is_recovered_by_resume(worker):
    """PAUSE landed, SUSPEND failed: the resume unpauses the engine (it used to no-op)."""
    client = worker.inference_client
    client.fail_on = "SUSPEND"
    with pytest.raises(ConnectionError):
        _bounded(worker.suspend_for_refit)
    assert _state(worker) == EngineState.PAUSED
    assert not worker._inference_engine_asleep

    client.fail_on = None
    _bounded(worker.resume_after_refit)
    assert _state(worker) == EngineState.RUNNING
    assert client.sent == ["PAUSE", "UNPAUSE"]


def test_a_suspend_that_failed_after_pause_is_finished_by_the_next_suspend(worker):
    client = worker.inference_client
    client.fail_on = "SUSPEND"
    with pytest.raises(ConnectionError):
        _bounded(worker.suspend_for_refit)
    client.fail_on = None
    _bounded(worker.suspend_for_refit)
    assert _state(worker) == EngineState.SUSPENDED
    # PAUSE is not sent again to a PAUSED engine.
    assert client.sent == ["PAUSE", "SUSPEND"]


def test_an_engine_suspended_behind_a_stale_flag_is_not_suspended_twice(worker):
    """The engine reached SUSPENDED but the flag was not set (the barrier raised).

    A second SUSPEND outside PAUSED would end Megatron's engine loop.
    """
    _bounded(worker.suspend_for_refit)
    worker._inference_engine_asleep = False
    _bounded(worker.suspend_for_refit)
    assert _state(worker) == EngineState.SUSPENDED
    assert worker.dynamic_inference_engine.engine_loop_task.done() is False
    assert worker.inference_client.sent == ["PAUSE", "SUSPEND"]
    _bounded(worker.resume_after_refit)
    assert _state(worker) == EngineState.RUNNING


def test_resuming_a_running_engine_sends_nothing(worker):
    _bounded(worker.resume_after_refit)
    assert _state(worker) == EngineState.RUNNING
    assert worker.inference_client.sent == []


def test_an_uninitialized_engine_is_left_alone(worker):
    worker._inference_engine_initialized = False
    _bounded(worker.suspend_for_refit)
    _bounded(worker.resume_after_refit)
    assert worker.inference_client.sent == []
    assert _state(worker) == EngineState.RUNNING


def test_a_stopped_engine_loop_raises_instead_of_hanging(worker):
    engine = worker.dynamic_inference_engine
    # An UNPAUSE outside PAUSED breaks the engine's rule and ends its loop.
    engine.pending.append("UNPAUSE")
    with pytest.raises(RuntimeError, match="engine loop has stopped") as caught:
        _bounded(worker.suspend_for_refit)
    assert isinstance(caught.value.__cause__, AssertionError)
    with pytest.raises(RuntimeError, match="engine loop has stopped"):
        _bounded(worker.resume_after_refit)


def test_an_engine_loop_that_stops_during_a_wait_raises_instead_of_hanging(worker):
    client = worker.inference_client
    engine = worker.dynamic_inference_engine
    original = client.pause_engines

    def pause_then_break():
        original()
        # The loop dies before PAUSED is reached.
        engine.pending.appendleft("RESUME")

    client.pause_engines = pause_then_break
    with pytest.raises(RuntimeError, match="while waiting for PAUSED"):
        _bounded(worker.suspend_for_refit)
