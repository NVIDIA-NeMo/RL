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

"""RolloutHealthMonitor lifecycle + SGLangGeneration monitor wiring.

No sglang extra, no Ray runtime and no GPU: engine actors are replaced by
fakes and the ``ray`` module used by ``fault_tolerance`` is monkeypatched,
so these run in the base (unmarked) unit-test shard.
"""

import threading
import time

import pytest

from nemo_rl.models.generation.sglang import fault_tolerance
from nemo_rl.models.generation.sglang.fault_tolerance import RolloutHealthMonitor
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration

CHECK_INTERVAL = 0.01
CHECK_TIMEOUT = 0.5
WAIT_TIMEOUT = 5.0
# ``stop`` joins for ``timeout + interval + 5``; the +5 is a fixed floor, so a
# test that needs the join to expire cannot run faster than this.
STOP_JOIN_TIMEOUT = CHECK_TIMEOUT + CHECK_INTERVAL + 5


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _RemoteMethod:
    """Stands in for a Ray actor method: ``.remote()`` returns a thunk."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return lambda: self._fn(*args, **kwargs)


class _FakeRay:
    def __init__(self):
        self.killed = []
        self.get_timeouts = []

    def get(self, ref, timeout=None):
        self.get_timeouts.append(timeout)
        return ref()

    def kill(self, actor):
        self.killed.append(actor)


class _FakeEngine:
    def __init__(self, health_fn=None, shutdown_fn=None):
        self.health_check_count = 0
        self.shutdown_count = 0
        self.health_generate = _RemoteMethod(health_fn or self._health_generate)
        self.shutdown = _RemoteMethod(shutdown_fn or self._shutdown)

    def _health_generate(self, timeout=None):
        self.health_check_count += 1
        return True

    def _shutdown(self):
        self.shutdown_count += 1


class _FakeGeneration:
    def __init__(self, engines, nodes_per_engine=1):
        self.all_engines = list(engines)
        self.nodes_per_engine = nodes_per_engine

    @property
    def engines(self):
        return self.all_engines[:: self.nodes_per_engine]


class _RecordingMonitor:
    def __init__(self):
        self.events = []

    def arm_first_wait(self):
        self.events.append("arm_first_wait")

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")

    def stop(self):
        self.events.append("stop")


def _cfg(first_wait=0.0, interval=CHECK_INTERVAL, timeout=CHECK_TIMEOUT):
    return {
        "sglang_cfg": {
            "rollout_health_check_interval": interval,
            "rollout_health_check_timeout": timeout,
            "rollout_health_check_first_wait": first_wait,
        }
    }


def _wait_until(predicate, timeout=WAIT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def fake_ray(monkeypatch):
    ray_stub = _FakeRay()
    monkeypatch.setattr(fault_tolerance, "ray", ray_stub)
    return ray_stub


@pytest.fixture
def monitor_factory(fake_ray):
    """Build monitors and guarantee their threads are stopped afterwards."""
    built = []

    def _build(generation, **cfg_kwargs):
        monitor = RolloutHealthMonitor(generation, _cfg(**cfg_kwargs))
        built.append(monitor)
        return monitor

    yield _build
    for monitor in built:
        monitor.stop()


# ---------------------------------------------------------------------------
# RolloutHealthMonitor lifecycle
# ---------------------------------------------------------------------------
def test_start_without_engines_returns_false(monitor_factory):
    monitor = monitor_factory(_FakeGeneration([]))
    assert monitor.start() is False
    assert monitor._thread is None


def test_start_is_idempotent(monitor_factory):
    monitor = monitor_factory(_FakeGeneration([_FakeEngine()]))
    assert monitor.start() is True
    thread = monitor._thread
    assert monitor.start() is True
    assert monitor._thread is thread


def test_monitor_stays_idle_until_resumed(monitor_factory):
    engine = _FakeEngine()
    monitor = monitor_factory(_FakeGeneration([engine]))

    monitor.start()
    assert monitor.is_checking_enabled() is False
    time.sleep(0.2)
    assert engine.health_check_count == 0

    monitor.resume()
    assert monitor.is_checking_enabled() is True
    assert _wait_until(lambda: engine.health_check_count > 0)


def test_pause_stops_further_checks(monitor_factory):
    engine = _FakeEngine()
    monitor = monitor_factory(_FakeGeneration([engine]))

    monitor.start()
    monitor.resume()
    assert _wait_until(lambda: engine.health_check_count > 0)

    monitor.pause()
    assert monitor.is_checking_enabled() is False
    settled = engine.health_check_count
    time.sleep(0.2)
    assert engine.health_check_count == settled


def test_pause_waits_for_in_flight_check(monitor_factory):
    entered = threading.Event()
    release = threading.Event()

    def _blocking_health(timeout=None):
        entered.set()
        release.wait(WAIT_TIMEOUT)
        return True

    engine = _FakeEngine(health_fn=_blocking_health)
    monitor = monitor_factory(_FakeGeneration([engine]))
    monitor.start()
    monitor.resume()
    assert entered.wait(WAIT_TIMEOUT)

    paused = threading.Event()
    waiter = threading.Thread(target=lambda: (monitor.pause(), paused.set()))
    waiter.start()
    try:
        assert not paused.wait(0.2)
        release.set()
        assert paused.wait(WAIT_TIMEOUT)
    finally:
        release.set()
        waiter.join(WAIT_TIMEOUT)


def test_first_wait_delays_the_initial_checks(monitor_factory):
    engine = _FakeEngine()
    monitor = monitor_factory(_FakeGeneration([engine]), first_wait=2.0)

    monitor.start()
    monitor.resume()
    time.sleep(0.3)
    assert engine.health_check_count == 0
    assert _wait_until(lambda: engine.health_check_count > 0, timeout=10.0)


def test_first_wait_is_not_restarted_by_every_resume(monitor_factory):
    """``resume`` runs once per training step. Re-arming the grace period there
    meant any generation phase shorter than ``first_wait`` left the monitor
    permanently inside it, so an enabled monitor never probed anything.
    """
    engine = _FakeEngine()
    monitor = monitor_factory(_FakeGeneration([engine]), first_wait=0.3)
    monitor.start()

    # Six training steps whose generation phases are all shorter than the
    # grace period, separated by paused (training) gaps.
    for _ in range(6):
        monitor.resume()
        time.sleep(0.1)
        monitor.pause()
        time.sleep(0.1)

    assert engine.health_check_count > 0


def test_stop_terminates_the_thread(monitor_factory):
    monitor = monitor_factory(_FakeGeneration([_FakeEngine()]))
    monitor.start()
    monitor.resume()
    thread = monitor._thread

    monitor.stop()
    assert monitor._thread is None
    assert monitor.is_checking_enabled() is False
    assert not thread.is_alive()


def test_stop_without_start_is_a_noop(monitor_factory):
    monitor = monitor_factory(_FakeGeneration([_FakeEngine()]))
    monitor.stop()
    assert monitor._thread is None


def test_stop_leaves_events_intact_when_the_join_times_out(monitor_factory):
    """A probe can outlive ``stop``'s join budget: it is bounded at
    ``2 * timeout``, and a failed probe then spends up to another ``timeout``
    in ``_kill_engine``, against a budget of ``timeout + interval + 5``.
    Clearing the events on that path killed the still-running thread with
    ``AttributeError: 'NoneType' object has no attribute 'wait'``.
    """
    entered = threading.Event()
    release = threading.Event()

    def _blocking_health(timeout=None):
        entered.set()
        # Must outlast ``stop``'s join budget, which has a fixed +5s floor.
        release.wait(STOP_JOIN_TIMEOUT + WAIT_TIMEOUT)
        return True

    engine = _FakeEngine(health_fn=_blocking_health)
    monitor = monitor_factory(_FakeGeneration([engine]))
    monitor.start()
    monitor.resume()
    assert entered.wait(WAIT_TIMEOUT)

    died = []
    previous_hook = threading.excepthook
    threading.excepthook = died.append
    try:
        monitor.stop()
        # The join could not reap the thread, so its events must survive.
        assert monitor._thread is not None
        assert monitor._stop_event is not None and monitor._stop_event.is_set()
        assert monitor.is_checking_enabled() is False

        release.set()
        assert _wait_until(lambda: not monitor._thread.is_alive())
        assert died == []
    finally:
        release.set()
        threading.excepthook = previous_hook


# ---------------------------------------------------------------------------
# Health checking / engine kill
# ---------------------------------------------------------------------------
def test_health_check_is_bounded_by_a_ray_level_timeout(monitor_factory, fake_ray):
    engine = _FakeEngine()
    monitor = monitor_factory(_FakeGeneration([engine]))

    monitor._check_engine_health(0, engine)

    assert fake_ray.get_timeouts == [pytest.approx(2 * CHECK_TIMEOUT)]


def test_unhealthy_engine_is_killed_and_slot_cleared(monitor_factory, fake_ray):
    def _raise(timeout=None):
        raise RuntimeError("engine is down")

    engine = _FakeEngine(health_fn=_raise)
    generation = _FakeGeneration([engine])
    monitor = monitor_factory(generation)

    monitor._check_engine_health(0, engine)

    assert engine.shutdown_count == 1
    assert fake_ray.killed == [engine]
    assert generation.all_engines == [None]


def test_engine_is_killed_even_when_graceful_shutdown_fails(monitor_factory, fake_ray):
    def _raise(*args, **kwargs):
        raise RuntimeError("actor is wedged")

    engine = _FakeEngine(shutdown_fn=_raise)
    generation = _FakeGeneration([engine])
    monitor = monitor_factory(generation)

    monitor._kill_engine(rollout_engine_id=0)

    assert fake_ray.killed == [engine]
    assert generation.all_engines == [None]


def test_kill_engine_clears_every_node_of_a_multi_node_engine(
    monitor_factory, fake_ray
):
    engines = [_FakeEngine() for _ in range(4)]
    generation = _FakeGeneration(engines, nodes_per_engine=2)
    monitor = monitor_factory(generation)

    monitor._kill_engine(rollout_engine_id=1)

    assert generation.all_engines == [engines[0], engines[1], None, None]
    assert fake_ray.killed == [engines[2], engines[3]]


def test_none_engine_is_skipped(monitor_factory, fake_ray):
    monitor = monitor_factory(_FakeGeneration([None]))
    monitor._check_engine_health(0, None)
    assert fake_ray.killed == []


# ---------------------------------------------------------------------------
# SGLangGeneration monitor wiring
# ---------------------------------------------------------------------------
def _make_generation(monitor):
    gen = SGLangGeneration.__new__(SGLangGeneration)
    gen.weight_synchronizer = None
    gen.all_engines = []
    gen.needs_offload = True
    gen.num_gpus_per_engine = 1
    gen.num_gpus_per_node = 1
    gen.gpu_offset = 0
    gen.num_new_engines = 0
    gen.rollout_engine_lock = None
    gen._health_monitor = monitor
    gen._router_actor = None
    gen._http_client = None
    gen._async_loop = None
    gen._recover = lambda: None
    return gen


@pytest.mark.parametrize(
    "tags,expected",
    [
        (None, ["resume"]),
        (["kv_cache"], ["resume"]),
        (["weights", "kv_cache"], ["resume"]),
        (["weights"], []),
    ],
)
def test_prepare_for_generation_resumes_only_once_generation_ready(tags, expected):
    monitor = _RecordingMonitor()
    gen = _make_generation(monitor)

    gen.prepare_for_generation(tags=tags)

    assert monitor.events == expected


def test_finish_generation_pauses_monitoring():
    monitor = _RecordingMonitor()
    gen = _make_generation(monitor)

    gen.finish_generation()

    assert monitor.events == ["pause"]


def test_monitoring_survives_a_full_offload_recover_onload_cycle():
    monitor = _RecordingMonitor()
    gen = _make_generation(monitor)

    gen.finish_generation()
    gen.recover_updatable_engines()
    gen.prepare_for_generation(tags=["weights"])
    gen.prepare_for_generation(tags=["kv_cache"])

    assert monitor.events == ["pause", "pause", "resume"]


def test_recover_updatable_engines_reports_engine_state():
    gen = _make_generation(_RecordingMonitor())
    gen.num_new_engines = 2

    engines, lock, num_new_engines, gpu_counts, gpu_offsets = (
        gen.recover_updatable_engines()
    )

    assert engines == []
    assert lock is None
    assert num_new_engines == 2
    assert gpu_counts == []
    assert gpu_offsets == []


def test_recover_leaves_num_new_engines_alone_when_nothing_died():
    """The common case: a refit runs, no engine died. ``_start_engines``
    rewrites ``num_new_engines`` unconditionally, and that count is the only
    gate on building the trainer-side weight transport, so a no-op recovery
    must not run at all -- otherwise the first refit clears the count and
    every later weight send silently no-ops.
    """
    gen = _make_generation(_RecordingMonitor())
    gen.all_engines = [_FakeEngine(), _FakeEngine()]
    gen.num_new_engines = 4

    def _explode(port_cursors):
        raise AssertionError("_start_engines must not run when nothing died")

    gen._start_engines = _explode

    SGLangGeneration._recover(gen)

    assert gen.num_new_engines == 4


def test_recover_rearms_the_grace_period_for_restarted_engines():
    monitor = _RecordingMonitor()
    gen = _make_generation(monitor)
    gen.all_engines = [None]
    gen.num_new_engines = 1
    gen.needs_offload = False
    gen._start_engines = lambda port_cursors: ([], {})

    SGLangGeneration._recover(gen)

    assert monitor.events == ["arm_first_wait"]


def test_generation_lifecycle_is_a_noop_without_fault_tolerance():
    gen = _make_generation(None)

    gen.prepare_for_generation()
    gen.finish_generation()
    assert gen.shutdown() is True


def test_monitor_names_the_missing_tuning_keys():
    """Every sglang recipe inherits ``grpo_math_1B.yaml``, which carries no
    sglang keys, so flipping ``use_fault_tolerance: true`` in one of them
    reaches this constructor with none of the three knobs set.
    """
    with pytest.raises(AssertionError) as excinfo:
        RolloutHealthMonitor(
            _FakeGeneration([_FakeEngine()]),
            {"sglang_cfg": {"use_fault_tolerance": True}},
        )

    message = str(excinfo.value)
    for key in (
        "rollout_health_check_interval",
        "rollout_health_check_timeout",
        "rollout_health_check_first_wait",
    ):
        assert key in message
    assert "use_fault_tolerance" in message


# ---------------------------------------------------------------------------
# Recovery cohort rollback (checkpoint-engine restart contract)
# ---------------------------------------------------------------------------


def _bare_generation_for_recover():
    gen = SGLangGeneration.__new__(SGLangGeneration)
    gen.all_engines = ["survivor", None]
    gen.num_new_engines = 0
    gen._health_monitor = None
    gen.needs_offload = False
    return gen


def test_recover_rolls_back_the_cohort_when_replacement_init_fails(monkeypatch):
    """A failed attempt must leave only ``None`` slots and the old count.

    ``_start_engines`` publishes replacement actors and rewrites
    ``num_new_engines`` before their init is awaited, so without rollback a
    partially initialized cohort stays visible and the refit dispatch would
    rebind against a broken actor.
    """
    from unittest.mock import MagicMock

    from nemo_rl.models.generation.sglang import sglang_generation

    gen = _bare_generation_for_recover()
    fake_actor = MagicMock()

    def fake_start_engines(port_cursors=None):
        gen.all_engines[1] = fake_actor
        gen.num_new_engines = 1
        return (["init-handle"], {})

    gen._start_engines = fake_start_engines
    killed = []
    monkeypatch.setattr(
        sglang_generation.ray,
        "get",
        MagicMock(side_effect=RuntimeError("replacement init died")),
    )
    monkeypatch.setattr(
        sglang_generation.ray, "kill", lambda actor: killed.append(actor)
    )

    with pytest.raises(RuntimeError, match="replacement init died"):
        gen._recover()

    assert gen.all_engines == ["survivor", None]
    assert gen.num_new_engines == 0
    assert killed == [fake_actor]
    # Rollback mirrors the health monitor's kill path: a best-effort graceful
    # shutdown (router deregistration + server process tree) precedes the
    # ray.kill. Here the graceful ray.get raises (same mock) — tolerated.
    fake_actor.shutdown.remote.assert_called_once_with()


def test_recover_escalates_when_the_rollback_itself_fails(monkeypatch):
    """Rollback failure means inconsistent engine state: terminal, not retry."""
    from unittest.mock import MagicMock

    from nemo_rl.models.generation.sglang import sglang_generation
    from nemo_rl.models.generation.sglang.fault_tolerance import (
        RecoveryRollbackError,
    )

    gen = _bare_generation_for_recover()

    def fake_start_engines(port_cursors=None):
        gen.all_engines[1] = MagicMock()
        gen.num_new_engines = 1
        return (["init-handle"], {})

    gen._start_engines = fake_start_engines
    monkeypatch.setattr(
        sglang_generation.ray,
        "get",
        MagicMock(side_effect=RuntimeError("replacement init died")),
    )

    def failing_kill(_actor):
        raise RuntimeError("kill failed")

    monkeypatch.setattr(sglang_generation.ray, "kill", failing_kill)

    with pytest.raises(RecoveryRollbackError, match="rollback"):
        gen._recover()


def test_recover_rolls_back_when_start_engines_itself_fails(monkeypatch):
    """A synchronous mid-start failure must roll back what was published.

    ``_start_engines`` mutates ``all_engines`` and ``num_new_engines`` while
    it runs, so an exception during actor creation or port allocation —
    before any init is awaited — already leaves a partial cohort visible.
    """
    from unittest.mock import MagicMock

    from nemo_rl.models.generation.sglang import sglang_generation

    gen = _bare_generation_for_recover()
    # A nonzero unconsumed count is real: the constructor's _start_engines
    # reports the whole startup fleet until the first successful communicator
    # setup consumes it. Rollback must RESTORE it, not zero it.
    gen.num_new_engines = 3
    fake_actor = MagicMock()

    def fake_start_engines(port_cursors=None):
        gen.all_engines[1] = fake_actor
        gen.num_new_engines = 1
        raise RuntimeError("port allocation died")

    gen._start_engines = fake_start_engines
    killed = []
    monkeypatch.setattr(sglang_generation.ray, "get", MagicMock())
    monkeypatch.setattr(
        sglang_generation.ray, "kill", lambda actor: killed.append(actor)
    )

    with pytest.raises(RuntimeError, match="port allocation died"):
        gen._recover()

    assert gen.all_engines == ["survivor", None]
    assert gen.num_new_engines == 3
    assert killed == [fake_actor]


def test_recover_rolls_back_when_the_offload_transition_fails(monkeypatch):
    """Post-init recovery work is part of the same atomic attempt.

    If the ``needs_offload`` release/resume RPCs fail after the replacement
    initialized, leaving the cohort published would make the next recovery
    see no dead slot and rebind a partially transitioned engine. The whole
    attempt must roll back — including a graceful shutdown of the (fully
    initialized) replacement so no orphan server or router entry survives.
    """
    from unittest.mock import MagicMock

    from nemo_rl.models.generation.sglang import sglang_generation

    gen = _bare_generation_for_recover()
    gen.needs_offload = True

    class _StubMonitor:
        check_timeout = 7.5

        def arm_first_wait(self):
            pass

    gen._health_monitor = _StubMonitor()
    fake_actor = MagicMock()

    def fake_start_engines(port_cursors=None):
        gen.all_engines[1] = fake_actor
        gen.num_new_engines = 1
        return (["init-handle"], {})

    gen._start_engines = fake_start_engines

    get_calls = []

    def fake_get(handles, timeout=None):
        get_calls.append((handles, timeout))
        if len(get_calls) == 2:  # the first offload release RPC batch
            raise RuntimeError("offload transition died")

    killed = []
    monkeypatch.setattr(sglang_generation.ray, "get", fake_get)
    monkeypatch.setattr(
        sglang_generation.ray, "kill", lambda actor: killed.append(actor)
    )

    with pytest.raises(RuntimeError, match="offload transition died"):
        gen._recover()

    assert gen.all_engines == ["survivor", None]
    assert gen.num_new_engines == 0
    assert killed == [fake_actor]
    fake_actor.shutdown.remote.assert_called_once_with()
    # The graceful step is BOUNDED by the monitor's configured per-RPC
    # timeout — an unbounded ray.get could hang the rollback forever on a
    # wedged replacement.
    assert get_calls[2][1] == 7.5
