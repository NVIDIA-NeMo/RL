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
"""Interop: the straggler-cutoff loop driving the installed NeMo Gym's real ``run_examples``.

Gym side: ``RolloutCollectionHelper.run_examples`` from the installed ``nemo_gym`` (the
3rdparty pin in the ``nemo_gym`` CI shard), driven through a fake server client whose
``/run`` handlers return after a short delay or hang until cancelled, the same pattern as
Gym's own tests for this iterator. RL side: ``_iter_gym_results_with_straggler_cutoff``
as imported, nothing extracted or copied.

Cancelling needs the iterator ``run_examples`` returns to have ``aclose_and_drain()`` and ``aclose()``
(NVIDIA-NeMo/Gym#4029). With a Gym that returns a plain ``map`` the actor runs uncut
(covered in ``test_nemo_gym_straggler_cutoff.py``), so these tests skip until the pin
has it; from then on they run in the ``nemo_gym`` shard. They exercise the RL/Gym
iterator contract, not the HTTP layer or Gym's server-side middleware.
"""

import asyncio
import inspect
from dataclasses import dataclass
from time import monotonic
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

pytest.importorskip("nemo_gym")

from nemo_gym import rollout_collection
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.rollout_collection import RolloutCollectionHelper

from nemo_rl.environments import nemo_gym as rl_gym
from nemo_rl.environments.nemo_gym import (
    StragglerCutoffConfig,
    _iter_gym_results_with_straggler_cutoff,
)
from nemo_rl.experience.failures import GymTransportError
from nemo_rl.utils.timer import Timer

pytestmark = pytest.mark.nemo_gym

AGENT_NAME = "my_agent"
# Pre-dispatch validation reads the running config: the agent must exist as an
# agent instance, fronted by one environment server.
GLOBAL_CONFIG = {
    AGENT_NAME: {"responses_api_agents": {"impl": {}}},
    "my_environment_server": {
        "environment_servers": {"legacy_agent": {"agent_server": {"name": AGENT_NAME}}}
    },
}
# How long to watch for a late handler completion or admission after the cut.
LATE_WRITE_WATCH_S = 0.3


class _FakeResponse:
    status = 200
    ok = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def release(self) -> None:
        pass


class _FakeRunServers:
    """``/run`` handlers: return after ``delays[task]`` seconds, or hang when it is None."""

    def __init__(self, delays: dict[int, float | None]) -> None:
        self.delays = delays
        self.started: list[int] = []
        self.finished: list[int] = []
        self.cancelled: list[int] = []

    async def post(self, server_name: str, url_path: str, json: dict) -> _FakeResponse:
        task = json[TASK_INDEX_KEY_NAME]
        self.started.append(task)
        delay = self.delays[task]
        try:
            if delay is None:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled.append(task)
            raise
        self.finished.append(task)
        return _FakeResponse({"reward": 1.0, "task": task})


def _helper(servers: _FakeRunServers) -> RolloutCollectionHelper:
    client = MagicMock()
    client.post = servers.post
    client.global_config_dict = OmegaConf.create(GLOBAL_CONFIG)

    class _Helper(RolloutCollectionHelper):
        def setup_server_client(self, *args, **kwargs):
            return client

    return _Helper()


def _rows(n: int) -> list[dict]:
    return [
        {
            AGENT_REF_KEY_NAME: {"name": AGENT_NAME},
            TASK_INDEX_KEY_NAME: i,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "_rowidx": i,
            "responses_create_params": {"input": []},
        }
        for i in range(n)
    ]


@pytest.fixture
def fake_run_transport(monkeypatch):
    """Route Gym's response handling to the fake responses."""

    async def raise_for_status(response):
        assert response.ok

    async def get_response_json(response):
        return response.payload

    monkeypatch.setattr(rollout_collection, "raise_for_status", raise_for_status)
    monkeypatch.setattr(rollout_collection, "get_response_json", get_response_json)


def _skip_unless_cancellable(completions) -> None:
    if not hasattr(completions, "aclose_and_drain"):
        pytest.skip(
            "the installed NeMo Gym's run_examples cannot cancel in-flight rollouts "
            "(no aclose_and_drain()); needs a Gym with NVIDIA-NeMo/Gym#4029"
        )


@dataclass(frozen=True)
class _Scenario:
    name: str
    num_rows: int
    hanging: frozenset[int]
    max_resident_tasks: int | None
    cutoff: StragglerCutoffConfig


GRACE_CUTOFF = StragglerCutoffConfig(done_fraction=0.75, grace_s=0.2)
SCENARIOS = [
    _Scenario("unbounded tasks, grace cut", 8, frozenset({6, 7}), None, GRACE_CUTOFF),
    # NeMo RL does not pass Gym's max_resident_tasks; these two cover the
    # bounded iterator's aclose() contract in case Gym's default ever changes.
    _Scenario("bounded tasks, grace cut", 8, frozenset({1, 2}), 4, GRACE_CUTOFF),
    _Scenario(
        "bounded tasks, max_wall cut with unstarted rows",
        8,
        frozenset({0, 1, 2}),
        3,
        StragglerCutoffConfig(done_fraction=0.9, grace_s=10.0, max_wall_s=0.3),
    ),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_the_cutoff_cancels_gyms_in_flight_rollouts(scenario, fake_run_transport):
    servers = _FakeRunServers(
        {
            i: (None if i in scenario.hanging else 0.01 * (i + 1))
            for i in range(scenario.num_rows)
        }
    )
    rows = _rows(scenario.num_rows)
    run_examples_kwargs = {}
    if scenario.max_resident_tasks is not None:
        if (
            "max_resident_tasks"
            not in inspect.signature(RolloutCollectionHelper.run_examples).parameters
        ):
            pytest.skip("this NeMo Gym's run_examples has no max_resident_tasks")
        run_examples_kwargs["max_resident_tasks"] = scenario.max_resident_tasks
    stats: dict[str, float] = {}

    async def collect():
        completions = _helper(servers).run_examples(rows, **run_examples_kwargs)
        _skip_unless_cancellable(completions)
        started = monotonic()
        out = [
            (row["_rowidx"], result, was_cut)
            async for row, result, was_cut in _iter_gym_results_with_straggler_cutoff(
                completions, rows, scenario.cutoff, Timer(), "timing/interop", stats
            )
        ]
        loop_s = monotonic() - started
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        started_at_cut = sorted(servers.started)
        finished_at_cut = sorted(servers.finished)
        with pytest.raises(StopIteration):
            next(completions)
        await asyncio.sleep(LATE_WRITE_WATCH_S)
        return out, loop_s, pending, started_at_cut, finished_at_cut

    out, loop_s, pending, started_at_cut, finished_at_cut = asyncio.run(collect())

    finished_rows = [i for i, _, cut in out if not cut]
    cut_rows = [i for i, _, cut in out if cut]
    never_started = sorted(set(range(scenario.num_rows)) - set(started_at_cut))
    first_cut = next((k for k, (_, _, cut) in enumerate(out) if cut), len(out))

    # Every row comes back exactly once: finished rows first, in completion
    # order, then the cut rows in input order with no result.
    assert sorted(i for i, _, _ in out) == list(range(scenario.num_rows))
    assert not any(cut for _, _, cut in out[:first_cut])
    assert cut_rows == sorted(cut_rows)
    assert all(result is None for _, result, cut in out if cut)
    assert sorted(finished_rows) == finished_at_cut
    # The hanging rollouts were really cancelled (their /run handlers saw it),
    # nothing else was, and rows never admitted were cut rather than started.
    assert set(scenario.hanging) <= set(cut_rows)
    assert set(never_started) <= set(cut_rows)
    assert sorted(servers.cancelled) == sorted(scenario.hanging)
    # Closed for good: no task left behind, no late write, no late admission.
    assert not pending
    assert sorted(servers.started) == started_at_cut
    assert sorted(servers.finished) == finished_at_cut
    assert loop_s < 5.0

    assert stats["straggler_cutoff/num_cut"] == float(len(cut_rows))
    cut_by_max_wall = scenario.cutoff.max_wall_s is not None
    assert stats["straggler_cutoff/cut_by_max_wall"] == float(cut_by_max_wall)
    assert stats["straggler_cutoff/armed"] == float(not cut_by_max_wall)
    if cut_by_max_wall:
        assert finished_rows == []
        assert never_started, "bounded admission should have left rows unstarted"
        assert stats["straggler_cutoff/cut_at_s"] >= scenario.cutoff.max_wall_s
    else:
        assert stats["straggler_cutoff/done_fraction_at_cut"] >= (
            scenario.cutoff.done_fraction
        )
        assert stats["straggler_cutoff/cut_at_s"] >= (
            stats["straggler_cutoff/armed_at_s"] + scenario.cutoff.grace_s
        )


@pytest.mark.parametrize("max_resident_tasks", [None, 2])
@pytest.mark.parametrize("exit_path", ["early_close", "cancellation"])
def test_non_cutoff_exit_closes_real_gym_tasks(
    max_resident_tasks, exit_path, fake_run_transport
):
    servers = _FakeRunServers({0: 0.01, 1: None, 2: None, 3: None})
    rows = _rows(4)
    kwargs = {}
    if max_resident_tasks is not None:
        if (
            "max_resident_tasks"
            not in inspect.signature(RolloutCollectionHelper.run_examples).parameters
        ):
            pytest.skip("this NeMo Gym's run_examples has no max_resident_tasks")
        kwargs["max_resident_tasks"] = max_resident_tasks

    async def collect():
        completions = _helper(servers).run_examples(rows, **kwargs)
        _skip_unless_cancellable(completions)
        stats: dict[str, float] = {}
        stream = _iter_gym_results_with_straggler_cutoff(
            completions, rows, StragglerCutoffConfig(), Timer(), "timing/interop", stats
        )
        try:
            row, _, event = await anext(stream)
            assert row["_rowidx"] == 0
            assert event is None
            if exit_path == "cancellation":
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                await stream.aclose()
            # Check before event-loop shutdown, which would cancel leaked tasks.
            assert asyncio.all_tasks() == {asyncio.current_task()}
            assert set(servers.cancelled) == set(servers.started) - {0}
            assert stats["straggler_cutoff/num_cut"] == 0
            with pytest.raises(StopIteration):
                next(completions)
            started_at_close = list(servers.started)
            await asyncio.sleep(LATE_WRITE_WATCH_S)
            assert servers.started == started_at_close
            assert servers.finished == [0]
        finally:
            await stream.aclose()
            await completions.aclose()

    asyncio.run(collect())


@pytest.mark.parametrize("max_resident_tasks", [None, 4])
@pytest.mark.parametrize("deadline", ["zero_grace", "positive_grace", "max_wall"])
def test_cutoff_preserves_ready_backlog(
    max_resident_tasks, deadline, fake_run_transport, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(rl_gym, "monotonic", lambda: clock[0])
    cutoff = StragglerCutoffConfig(
        done_fraction=0.5,
        grace_s=0 if deadline == "zero_grace" else 1,
        max_wall_s=1 if deadline == "max_wall" else None,
    )
    servers = _FakeRunServers({i: 0.0 for i in range(4)})
    rows = _rows(4)
    stats = {}

    async def collect():
        completions = _helper(servers).run_examples(
            rows, max_resident_tasks=max_resident_tasks
        )
        _skip_unless_cancellable(completions)
        out = []
        async for row, result, event in _iter_gym_results_with_straggler_cutoff(
            completions, rows, cutoff, Timer(), "timing/backlog", stats
        ):
            assert sorted(servers.finished) == [0, 1, 2, 3]
            out.append((row["_rowidx"], result, event))
            # Model slow consumption deterministically, after arming or before max-wall.
            if (
                deadline == "positive_grace" and len(out) == 3
            ) or deadline == "max_wall":
                clock[0] = 10.0
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return out

    out = asyncio.run(collect())
    assert sorted(idx for idx, _, _ in out) == [0, 1, 2, 3]
    assert all(result is not None and event is None for _, result, event in out)
    assert stats["straggler_cutoff/num_cut"] == 0
    assert servers.cancelled == []


@pytest.mark.parametrize("max_resident_tasks", [None, 4])
def test_cutoff_preserves_ready_results_with_pending_and_unadmitted_work(
    max_resident_tasks, fake_run_transport
):
    rows = _rows(7)
    servers = _FakeRunServers({i: 0.0 if i < 3 else None for i in range(7)})
    stats = {}

    async def collect():
        completions = _helper(servers).run_examples(
            rows, max_resident_tasks=max_resident_tasks
        )
        _skip_unless_cancellable(completions)
        out = [
            (row["_rowidx"], event is not None)
            async for row, _, event in _iter_gym_results_with_straggler_cutoff(
                completions,
                rows,
                StragglerCutoffConfig(done_fraction=0.1, grace_s=0),
                Timer(),
                "timing/mixed",
                stats,
            )
        ]
        assert asyncio.all_tasks() == {asyncio.current_task()}
        started = list(servers.started)
        await asyncio.sleep(0)
        assert servers.started == started
        return out

    out = asyncio.run(collect())
    assert sorted(index for index, cut in out if not cut) == [0, 1, 2]
    assert [index for index, cut in out if cut] == [3, 4, 5, 6]
    assert stats["straggler_cutoff/num_cut"] == 4
    assert set(servers.cancelled) == set(servers.started) - {0, 1, 2}
    if max_resident_tasks is not None:
        assert not set(servers.started) & {5, 6}


@pytest.mark.parametrize("max_resident_tasks", [None, 4])
def test_cutoff_propagates_a_queued_failure(max_resident_tasks, fake_run_transport):
    class HttpError(Exception):
        status = 503

    async def collect():
        release_failure = asyncio.Event()

        class FailureServer(_FakeRunServers):
            async def post(self, server_name, url_path, json):
                if json[TASK_INDEX_KEY_NAME] == 3:
                    await release_failure.wait()
                    raise HttpError("queued server failure")
                return await super().post(server_name, url_path, json)

        servers = FailureServer({0: 0.0, 1: 0.0, 2: 0.0})
        rows = _rows(4)
        completions = _helper(servers).run_examples(
            rows, max_resident_tasks=max_resident_tasks
        )
        _skip_unless_cancellable(completions)
        stream = _iter_gym_results_with_straggler_cutoff(
            completions,
            rows,
            StragglerCutoffConfig(done_fraction=0.25, grace_s=0),
            Timer(),
            "timing/failure",
            {},
        )
        await anext(stream)
        release_failure.set()
        await asyncio.sleep(0)
        with pytest.raises(GymTransportError, match="HTTP 503"):
            async for _, _, event in stream:
                assert event is None
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(collect())
