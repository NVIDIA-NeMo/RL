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

"""Recovering the refit after a generation shard dies during or around it.

Two distinct failures land on this path, and job 5925668 hit one with each variant of
the functional test:

``RefitAborted``    a rank went silent inside the collective and a worker's watchdog
                    broke it. Survivors hold partial weights.
``RayActorError``   the collective completed and the shard died before its RPC
                    returned. Survivors hold complete weights.

What the tests here pin down is the ordering that made the first attempt fail on real
hardware: the repair has to establish who is gone *itself*, rather than reading a health
monitor whose verdict arrives on a probe schedule.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest
import ray.exceptions

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.distributed.refit_watchdog import (
    RefitAborted,
    is_refit_abort,
    is_refit_context_lost,
)
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetHealth,
    ShardState,
)


async def _completed(value=None):
    return value


async def _failed(error):
    raise error


class _Synchronizer:
    """Fails the first sync, then succeeds. Records what it was reconciled to."""

    def __init__(self, failure, *, rebuild_succeeds: bool = True) -> None:
        self._failure = failure
        self._rebuild_succeeds = rebuild_succeeds
        self.sync_calls = 0
        self.reconciled_with: list[list[int]] = []
        self.forced: list[bool] = []
        # State of the fleet at the moment the retry ran, which is the only place the
        # dead shard's exclusion can actually be observed.
        self.absent_at_retry: list[int] | None = None
        self._monitor = None

    def bind(self, monitor) -> None:
        self._monitor = monitor

    def sync_weights(self, *, kv_scales=None):
        del kv_scales
        self.sync_calls += 1
        if self.sync_calls == 1:
            raise self._failure
        if self._monitor is not None:
            self.absent_at_retry = self._monitor.absent_shards()

    def reconcile_communicator(self, absent, force=False):
        # `force` is how the recovery path says the communicator is gone rather than
        # merely unchanged; the real synchronizers skip an unchanged absent set.
        self.reconciled_with.append(sorted(absent))
        self.forced.append(force)
        return bool(absent) and self._rebuild_succeeds


def _make_controller(
    failure,
    *,
    shard_count: int = 2,
    dead_shards: tuple[int, ...] = (0,),
    rebuild_succeeds: bool = True,
    with_monitor: bool = True,
):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = SimpleNamespace(
        recompute_kv_cache_after_weight_updates=False,
        generation_fleet_health=SimpleNamespace(
            probe_timeout_s=1.0,
            probe_interval_s=0.01,
            # None keeps the controller's refit await unbounded, which is what every
            # test in this file was written against. A fixture standing in for a
            # constructed actor has to carry every field the methods under test read.
            refit_timeout_s=None,
        ),
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()

    monitor = None
    if with_monitor:
        # unhealthy_threshold well above anything a single probe round could reach, so a
        # DEAD verdict here can only have come from the conclusive path.
        monitor = GenerationFleetHealth(
            shard_count=shard_count,
            policy=FleetHealthPolicy(unhealthy_threshold=99, min_healthy_shards=1),
        )
    ctrl._gen_fleet = monitor

    sync = _Synchronizer(failure, rebuild_succeeds=rebuild_succeeds)
    sync.bind(monitor)
    ctrl._weight_synchronizer = sync

    ctrl._gen = SimpleNamespace(
        requires_kv_scale_sync=False,
        invalidate_kv_cache=MagicMock(),
        worker_group=SimpleNamespace(
            get_dp_leader_worker_idx=lambda shard: shard,
            workers=[
                SimpleNamespace(
                    is_alive=SimpleNamespace(
                        remote=(
                            (lambda: _failed(ray.exceptions.ActorDiedError()))
                            if idx in dead_shards
                            else (lambda: _completed(True))
                        )
                    )
                )
                for idx in range(shard_count)
            ],
        ),
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 7
    # _sync_weights now asks _should_use_nemo_gym before aborting stale in-flight
    # rollouts (upstream #3263). An empty env dict selects the native path, and an
    # empty registry makes _abort_stale_inflight a no-op -- neither is what this test
    # is about, but both have to exist for it to reach the refit.
    ctrl._master_config = SimpleNamespace(env={})
    ctrl._inflight_by_group_id = {}
    ctrl._rollout_recovery_enabled = False
    return ctrl, monitor, sync


ABORTED = RefitAborted("refit broadcast exceeded 60.0s and was aborted")


class TestDeathInsideTheCollective:
    """RefitAborted: the watchdog broke a collective a dead peer had stalled."""

    def test_the_dead_shard_is_identified_without_waiting_for_probe_counters(self):
        """The bug that made the first attempt fail on hardware.

        The handler used to read absent_shards() straight from the monitor, whose verdict
        is paced by probe rounds. The abort is an event and always won that race, so the
        corpse was still SUSPECT, absent came back empty, and nothing was rebuilt.
        unhealthy_threshold is 99 here: no amount of counting could produce this verdict.
        """
        ctrl, monitor, _ = _make_controller(ABORTED)
        asyncio.run(ctrl._sync_weights())
        assert monitor.state_of(0) is ShardState.DEAD

    def test_the_rebuild_excludes_the_dead_shard(self):
        ctrl, _, sync = _make_controller(ABORTED)
        asyncio.run(ctrl._sync_weights())
        assert sync.reconciled_with[-1] == [0]

    def test_the_retry_runs_against_the_smaller_fleet(self):
        """Ordering, not just the end state: the corpse must be gone *before* the retry.

        Rebuilding after retrying would hang on the same missing rank.
        """
        ctrl, _, sync = _make_controller(ABORTED)
        asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 2
        assert sync.absent_at_retry == [0]

    def test_survivors_are_pulled_from_service_then_given_back(self):
        """Partial weights must not serve -- and must not be stranded either.

        mark_weights_partial has no other exit, so without the promotion at the end of a
        successful sync the recovery would finish with an empty serving set and the run
        would die of an exhausted fleet: a worse failure than the one being repaired.
        """
        ctrl, monitor, _ = _make_controller(ABORTED)
        asyncio.run(ctrl._sync_weights())
        assert monitor.serving_shards() == [1]
        assert monitor.state_of(1) is ShardState.HEALTHY

    def test_the_dead_shard_is_not_laundered_into_stale(self):
        """Regression: marking the corpse's weights partial hid it from the rebuild.

        mark_weights_partial walks the *serving* shards, and a shard that has only just
        died is still serving. STALE is deliberately not an absent state -- refitting a
        STALE shard is how it stops being stale -- so a corpse moved there stays in the
        refit membership and the rebuild puts it straight back into the collective.
        """
        ctrl, monitor, sync = _make_controller(ABORTED)
        asyncio.run(ctrl._sync_weights())
        assert monitor.state_of(0) is not ShardState.STALE
        assert 0 not in monitor.serving_shards()
        assert sync.reconciled_with[-1] == [0]

    def test_a_recovery_does_not_clear_a_survivor_s_reported_failures(self):
        """Regression: the recovery laundered a SUSPECT survivor back to HEALTHY.

        consecutive_reported_failures is the only counter that can condemn a wedged
        engine, and report_failure keeps it separate from the probe streak precisely
        because such an engine still answers is_alive -- so an ok-probe must not clear
        it. A refit used to.

        The path: serving_shards() includes SUSPECT, so mark_weights_partial moves the
        suspect survivor to STALE, and _promote_refit_shards then promotes every STALE
        shard through report_refit, which zeroes the counters. Net effect, a shard
        failing real generations could never reach unhealthy_threshold for as long as
        refits kept happening.

        Driven through the whole recovery on purpose: asserting against
        _promote_refit_shards alone passes even with the bug present, because it is
        mark_weights_partial -- in the other function -- that relabels SUSPECT to STALE.
        """
        ctrl, monitor, _ = _make_controller(ABORTED, shard_count=3, dead_shards=(0,))
        # shard 2 is a survivor that was already failing generations
        monitor.report_failure(2, RuntimeError("router: failed generation"))
        monitor.report_failure(2, RuntimeError("router: failed generation"))
        assert monitor.state_of(2) is ShardState.SUSPECT
        before = monitor.snapshot()[2].consecutive_reported_failures
        assert before == 2

        asyncio.run(ctrl._sync_weights())

        assert monitor.snapshot()[2].consecutive_reported_failures == before, (
            "the refit cleared the streak that was about to condemn this shard"
        )
        assert monitor.state_of(2) is ShardState.SUSPECT
        # the untroubled survivor is still promoted normally
        assert monitor.state_of(1) is ShardState.HEALTHY

    def test_an_untroubled_survivor_is_still_promoted_to_healthy(self):
        """The restore must not become a blanket refusal to promote."""
        ctrl, monitor, _ = _make_controller(ABORTED)
        asyncio.run(ctrl._sync_weights())
        assert monitor.state_of(1) is ShardState.HEALTHY
        assert monitor.snapshot()[1].consecutive_reported_failures == 0


class TestDeathAfterTheCollective:
    """RayActorError: the broadcast completed and the shard died in the epilogue.

    ray.get(futures_train) returned and ray.get(futures_inference) raised. The data
    transfer had already succeeded, and the run died anyway.
    """

    def test_an_actor_death_is_recovered_rather_than_fatal(self):
        ctrl, monitor, sync = _make_controller(ray.exceptions.ActorDiedError())
        asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 2
        assert monitor.state_of(0) is ShardState.DEAD
        assert sync.reconciled_with[-1] == [0]

    def test_survivors_keep_serving_because_nothing_is_partial(self):
        """The distinction that makes this worth separating from the abort.

        The collective finished, so the survivors' weights are complete. Marking them
        partial would take a healthy fleet out of service over a transfer that worked.
        """
        ctrl, monitor, _ = _make_controller(ray.exceptions.ActorDiedError())
        asyncio.run(ctrl._sync_weights())
        assert monitor.state_of(1) is ShardState.HEALTHY


class TestWhenRecoveryIsNotPossible:
    def test_no_identifiable_absentee_fails_instead_of_retrying(self):
        """A rank alive but not participating is not a membership problem.

        There is no smaller fleet to rebuild over, so a retry would either die on the
        aborted communicator or rebuild the full one and hang on the same silent rank --
        recreating the wedge. Fail attributably instead.
        """
        ctrl, _, sync = _make_controller(ABORTED, dead_shards=())
        with pytest.raises(RuntimeError, match="could be identified as absent"):
            asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 1, "must not retry without a rebuilt communicator"

    def test_a_failed_rebuild_is_not_retried_over(self):
        ctrl, _, sync = _make_controller(ABORTED, rebuild_succeeds=False)
        with pytest.raises(RuntimeError, match="could be identified as absent"):
            asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 1

    def test_without_fleet_health_the_original_failure_propagates(self):
        """Inert by default: with no monitor there is nothing to reconcile against.

        The error the caller sees must be the refit's own, not a recovery failure
        layered on top of it.
        """
        ctrl, _, sync = _make_controller(
            ray.exceptions.ActorDiedError(), with_monitor=False
        )
        with pytest.raises(ray.exceptions.ActorDiedError):
            asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 1
        assert sync.reconciled_with == []


class TestTheRecoveryForcesTheRebuild:
    """The pre-refit reconciles may skip; the recovery's may not.

    The synchronizers skip a rebuild when the absent set matches what they last built
    with -- that is what stops a lost shard costing two full rebuilds on every subsequent
    step. After an abort the absent set is identical and the communicator is *gone*, so
    the recovery has to override the skip or it retries over a communicator that no longer
    exists and fails with "no generation shard could be identified as absent".
    """

    def test_the_recovery_reconcile_is_forced(self):
        ctrl, _, sync = _make_controller(RefitAborted("aborted"))

        asyncio.run(ctrl._sync_weights())

        assert any(sync.forced), (
            "the recovery must force its rebuild; skipping it there retries over an "
            "aborted communicator"
        )

    def test_the_pre_refit_reconciles_are_not_forced(self):
        """They run every step, and forcing them would reinstate the cost this removes."""
        ctrl, _, sync = _make_controller(None)
        sync._failure = None

        def _clean(*, kv_scales=None):
            del kv_scales
            sync.sync_calls += 1

        sync.sync_weights = _clean
        asyncio.run(ctrl._sync_weights())

        assert sync.forced and not any(sync.forced), (
            "a healthy step must never force a rebuild"
        )


class TestTheHappyPathIsUntouched:
    def test_a_clean_sync_neither_probes_nor_rebuilds(self):
        ctrl, monitor, sync = _make_controller(None)
        sync._failure = None

        def _clean(*, kv_scales=None):
            del kv_scales
            sync.sync_calls += 1

        sync.sync_weights = _clean
        asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 1
        assert monitor.serving_shards() == [0, 1]
        # Reconciled before the collective as always, but with nothing absent.
        assert sync.reconciled_with == [[], []]


class TestTheRefitHoldHook:
    """The fault-injection hook that makes "killed mid-refit" reproducible.

    Timing alone cannot reach that window -- a refit here takes ~0.10s -- so job 5925668
    aimed at the collective and hit the RPC epilogue instead, leaving the abort path
    untested while reporting a result.
    """

    @staticmethod
    def _hook():
        from nemo_rl.distributed.refit_watchdog import (
            hold_refit_for_fault_injection,
        )

        return hold_refit_for_fault_injection

    def test_it_is_inert_when_unset(self, monkeypatch):
        """Every real run takes this path, so it must cost nothing and change nothing."""
        monkeypatch.delenv("NRL_REFIT_HOLD_FILE", raising=False)
        self._hook()()  # must return immediately

    def test_it_is_inert_when_the_file_is_absent(self, monkeypatch, tmp_path):
        """Armed for the whole run, but only holds while the harness says so."""
        monkeypatch.setenv("NRL_REFIT_HOLD_FILE", str(tmp_path / "nope"))
        self._hook()()

    def test_it_blocks_while_the_file_exists_and_returns_when_removed(
        self, monkeypatch, tmp_path
    ):
        import threading

        hold = tmp_path / "hold_refit"
        hold.write_text("")
        monkeypatch.setenv("NRL_REFIT_HOLD_FILE", str(hold))
        monkeypatch.setenv("NRL_REFIT_HOLD_MAX_S", "10")

        released = threading.Event()

        def _run():
            self._hook()()
            released.set()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        assert not released.wait(0.3), "must still be holding while the file is there"

        hold.unlink()

        assert released.wait(5.0), "must return once the harness removes the file"

    def test_the_hold_is_bounded(self, monkeypatch, tmp_path):
        """A harness that dies mid-test must not wedge the worker forever."""
        hold = tmp_path / "hold_refit"
        hold.write_text("")
        monkeypatch.setenv("NRL_REFIT_HOLD_FILE", str(hold))
        monkeypatch.setenv("NRL_REFIT_HOLD_MAX_S", "0.2")
        self._hook()()
        assert hold.exists(), "returned on the deadline, not because the file went away"


class TestTheControllerStopsWaitingForASilentRank:
    """A frozen-but-alive rank is a Ray actor that never answers, and Ray never times out.

    Every worker-side bound can behave perfectly and the controller still waits forever.
    Job 6508251 measured that end state: the deadline fired, the workers aborted, the
    trainers returned, every actor was idle -- and the run sat for 1800s because this await
    had no bound. It is the last unbounded wait on the refit path.
    """

    @staticmethod
    def _ctrl(refit_timeout_s, sync_weights):
        # @ray.remote makes SingleControllerActor an ActorClass, not a class, so
        # object.__new__ rejects it. Same unwrap the fixture above uses.
        ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
        ctrl._async_cfg = SimpleNamespace(
            generation_fleet_health=SimpleNamespace(refit_timeout_s=refit_timeout_s)
        )
        ctrl._weight_synchronizer = SimpleNamespace(sync_weights=sync_weights)
        return ctrl

    def test_a_refit_that_never_returns_gives_up(self):
        never = threading.Event()
        ctrl = self._ctrl(0.2, lambda **_: never.wait())
        ctrl._REFIT_UNWIND_GRACE_S = 0.3

        started = time.monotonic()
        with pytest.raises(RefitAborted, match="did not return within"):
            asyncio.run(ctrl._sync_weights_within(None, "first"))
        elapsed = time.monotonic() - started

        never.set()
        assert elapsed < 10.0, "it must give up, not wait on a rank that never answers"

    def test_giving_up_is_recognised_as_an_abort(self):
        """So it lands in the existing `except (RefitAborted, RayActorError)` recovery."""
        never = threading.Event()
        ctrl = self._ctrl(0.1, lambda **_: never.wait())
        ctrl._REFIT_UNWIND_GRACE_S = 0.1

        with pytest.raises(RefitAborted) as caught:
            asyncio.run(ctrl._sync_weights_within(None, "first"))
        never.set()
        assert is_refit_abort(caught.value)

    def test_the_budget_leaves_room_for_the_workers_own_error(self):
        """If this bound fired first we would lose the attributable worker diagnosis."""
        ctrl = self._ctrl(12.5, lambda **_: None)
        assert ctrl._refit_await_budget_s() > 12.5

    def test_no_deadline_configured_keeps_the_original_unbounded_await(self):
        calls = []
        ctrl = self._ctrl(None, lambda **kw: calls.append(kw))
        assert ctrl._refit_await_budget_s() is None
        asyncio.run(ctrl._sync_weights_within({"k": 1}, "first"))
        assert calls == [{"kv_scales": {"k": 1}}]

    def test_a_refit_that_succeeds_is_untouched(self):
        calls = []
        ctrl = self._ctrl(30.0, lambda **kw: calls.append(kw))
        asyncio.run(ctrl._sync_weights_within({"k": 1}, "first"))
        assert calls == [{"kv_scales": {"k": 1}}], "kv_scales must still reach the sync"

    def test_a_real_failure_propagates_rather_than_becoming_a_timeout(self):
        """A genuine refit bug must not be relabelled as a silent rank and retried."""

        def _boom(**_):
            raise ValueError("shape mismatch")

        ctrl = self._ctrl(30.0, _boom)
        with pytest.raises(ValueError, match="shape mismatch"):
            asyncio.run(ctrl._sync_weights_within(None, "first"))

    def test_the_worker_thread_cannot_block_interpreter_exit(self):
        """asyncio.to_thread's pool is non-daemon and joined at exit, so it is not used.

        Otherwise a thread still parked on the frozen actor would hang shutdown -- trading
        a wedge in the refit for a wedge on the way out.
        """
        never = threading.Event()
        seen = []

        original = threading.Thread

        def _record(*args, **kwargs):
            thread = original(*args, **kwargs)
            seen.append(thread)
            return thread

        ctrl = self._ctrl(0.1, lambda **_: never.wait())
        ctrl._REFIT_UNWIND_GRACE_S = 0.1
        with mock.patch.object(threading, "Thread", _record):
            with pytest.raises(RefitAborted):
                asyncio.run(ctrl._sync_weights_within(None, "first"))
        never.set()

        assert seen, "the refit must run on a thread we control"
        assert all(t.daemon for t in seen), "and every one of them must be a daemon"


class TestAWedgedButAliveEngineIsAttributed:
    """The failure mode most likely to break a refit, and the one that used to end the run.

    `is_alive()` is answered by the Ray actor and never touches the engine, so the
    post-abort probe can only ever see a dead PROCESS. An engine that is wedged with its
    actor healthy stays serving, is never absent, and the recovery refuses -- while that
    same shard has been driven to SUSPECT by its own timing-out generations. The abort says
    something stopped participating; SUSPECT says which.
    """

    def test_a_single_suspect_is_condemned_and_the_run_continues(self):
        ctrl, monitor, sync = _make_controller(
            RefitAborted("aborted"), dead_shards=(), shard_count=2
        )
        # Wedged, not dead: driven to SUSPECT by failing generations while its actor keeps
        # answering. Nothing here makes it absent.
        monitor.report_failure(1, TimeoutError("generation timed out"))

        asyncio.run(ctrl._sync_weights())

        assert monitor.state_of(1) is ShardState.DEAD, "the suspect must be condemned"
        assert 1 in monitor.absent_shards(), "and thereby excluded from the rebuild"

    def test_the_condemnation_says_why_it_was_not_an_actor_death(self):
        ctrl, monitor, _ = _make_controller(
            RefitAborted("aborted"), dead_shards=(), shard_count=2
        )
        monitor.report_failure(1, TimeoutError("generation timed out"))

        asyncio.run(ctrl._sync_weights())

        reason = monitor.snapshot()[1].last_error
        assert "did not participate" in reason, (
            "the ledger must not claim Ray reported the process gone"
        )

    def test_two_suspects_still_end_the_run(self):
        """Condemning the wrong one costs a healthy shard AND leaves the culprit in."""
        ctrl, monitor, _ = _make_controller(
            RefitAborted("aborted"), dead_shards=(), shard_count=3
        )
        monitor.report_failure(1, TimeoutError("generation timed out"))
        monitor.report_failure(2, TimeoutError("generation timed out"))

        with pytest.raises(
            RuntimeError, match="could not be safely rebuilt|identified as absent"
        ):
            asyncio.run(ctrl._sync_weights())

        assert monitor.state_of(1) is not ShardState.DEAD
        assert monitor.state_of(2) is not ShardState.DEAD

    def test_no_suspect_still_ends_the_run(self):
        """Nothing to go on; guessing would be worse than stopping."""
        ctrl, monitor, _ = _make_controller(
            RefitAborted("aborted"), dead_shards=(), shard_count=2
        )

        with pytest.raises(RuntimeError, match="identified as absent"):
            asyncio.run(ctrl._sync_weights())

    def test_the_message_names_the_suspects_it_found(self):
        """'no shard was absent' alone gave the reader nothing to act on."""
        ctrl, monitor, _ = _make_controller(
            RefitAborted("aborted"), dead_shards=(), shard_count=3
        )
        monitor.report_failure(1, TimeoutError("t"))
        monitor.report_failure(2, TimeoutError("t"))

        with pytest.raises(RuntimeError, match=r"already suspect"):
            asyncio.run(ctrl._sync_weights())


CONTEXT_LOST = RefitAborted(
    "[refit-context-lost] refit: the bulk parameter transfer did not retire within 60.0s"
)


class TestContextLostIsNotRecoverable:
    """The known limitation: detected and ended fast, never recovered from.

    nccl_reshard only, and only when the fault lands while the bulk transfer is in flight.
    sync_stream_within gives up on kernels already enqueued on the trainers' streams, and
    aborting a communicator does not retire them -- so those CUDA contexts are unusable,
    ncclCommAbort never returns and no rebuild on that device can bootstrap. Entering the
    recovery does not fail there, it WEDGES: jobs 6521181, 6523731, 6582457 and 6584636 all
    ran to the 1800s harness deadline with no attribution.
    """

    def test_the_failure_propagates_instead_of_being_recovered(self):
        ctrl, _, _ = _make_controller(CONTEXT_LOST)
        with pytest.raises(RefitAborted) as caught:
            asyncio.run(ctrl._sync_weights())
        assert is_refit_context_lost(caught.value)

    def test_no_rebuild_is_attempted(self):
        """The rebuild is the wedge, so not attempting it is the entire fix."""
        ctrl, _, sync = _make_controller(CONTEXT_LOST)
        with pytest.raises(RefitAborted):
            asyncio.run(ctrl._sync_weights())
        # force=True is the recovery's marker: it tells the transport the communicator is
        # gone rather than merely unchanged. The False entries are the ordinary reconciles
        # that run before every refit and are not the path under test.
        assert True not in sync.forced, (
            "a forced reconcile is the recovery path; on a lost context it cannot "
            f"complete and wedges the run. Saw forced={sync.forced}"
        )

    def test_no_retry_is_attempted(self):
        ctrl, _, sync = _make_controller(CONTEXT_LOST)
        with pytest.raises(RefitAborted):
            asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 1

    def test_an_ordinary_abort_still_recovers(self):
        """The scope guard, from the other side.

        The packed-broadcast transport raises plain RefitAborted here and recovers from it
        -- job 6584636 measured null-refit and null-refit-frozen doing exactly that. If
        this ever fails, the fail-fast path has widened past its one scenario.
        """
        ctrl, _, sync = _make_controller(ABORTED)
        asyncio.run(ctrl._sync_weights())
        assert sync.sync_calls == 2 and sync.forced != []
