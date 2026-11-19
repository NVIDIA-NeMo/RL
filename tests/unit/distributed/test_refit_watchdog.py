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

"""The watchdog that breaks a refit collective a dead peer has left hanging.

Testable without NCCL because the watchdog's contract is entirely about *when* it calls
``abort()`` -- the NCCL behaviour it depends on (an aborted collective releases, and
returns without raising) was verified separately on real hardware.
"""

import asyncio
import pickle
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from nemo_rl.distributed.refit_watchdog import (
    REFIT_ABORTED_TOKEN,
    RefitAborted,
    RefitAbortWatchdog,
    await_off_loop,
    is_refit_abort,
    sync_stream_within,
)


class _FakeGroup:
    def __init__(self, fail: bool = False) -> None:
        self.abort_calls = 0
        self._fail = fail

    def abort(self) -> None:
        self.abort_calls += 1
        if self._fail:
            raise RuntimeError("abort failed")


class TestDisarmed:
    """No timeout means no thread and no behaviour change, which is the default."""

    @pytest.mark.parametrize("timeout", [None, 0, -1.0])
    def test_a_non_positive_timeout_never_arms(self, timeout):
        group = _FakeGroup()
        with RefitAbortWatchdog(group, timeout) as guard:
            assert not guard.armed
        assert guard.fired is False
        assert group.abort_calls == 0

    def test_no_group_never_arms(self):
        with RefitAbortWatchdog(None, 0.01) as guard:
            assert not guard.armed
        assert guard.fired is False

    def test_no_thread_is_started_when_disarmed(self):
        before = threading.active_count()
        with RefitAbortWatchdog(_FakeGroup(), None):
            assert threading.active_count() == before


class TestFires:
    def test_it_aborts_a_block_that_overruns(self):
        group = _FakeGroup()
        with RefitAbortWatchdog(group, 0.05) as guard:
            time.sleep(0.4)
        assert guard.fired is True
        assert group.abort_calls == 1

    def test_fired_survives_the_clean_return(self):
        """The whole point: an aborted collective returns normally.

        There is no exception to catch, so the flag has to outlive the guarded block or
        the caller cannot tell a completed refit from an aborted one.
        """
        group = _FakeGroup()
        with RefitAbortWatchdog(group, 0.05) as guard:
            time.sleep(0.4)
        assert guard.fired is True

    def test_a_failing_abort_does_not_escape(self):
        """The caller is already blocked; a raising watchdog thread helps nobody."""
        group = _FakeGroup(fail=True)
        with RefitAbortWatchdog(group, 0.05) as guard:
            time.sleep(0.4)
        assert guard.fired is True
        assert group.abort_calls == 1


class TestTheAbortSurvivesABoundaryThatDropsTheType:
    """vLLM's EngineCore RPC keeps the message and discards the exception class.

    ``v1/engine/core.py`` stringifies the worker exception into ``failure_message``;
    ``v1/engine/core_client.py`` re-raises it as ``Exception(failure_message)``. So a
    RefitAborted raised inside the engine reaches the Ray actor as a plain Exception, and
    every ``except RefitAborted`` downstream of a collective_rpc was dead code.

    Job 6484412 is what that cost: the deadline fired, the abort was named in the log, the
    handler did not match, and the run wedged at step 4 for the rest of its wall-clock.
    """

    def test_every_message_carries_the_token(self):
        assert REFIT_ABORTED_TOKEN in str(RefitAborted("a peer stopped participating"))

    def test_the_token_is_not_stacked_on_re_wrap(self):
        """Re-raising translates message to message; the prefix must not accumulate."""
        once = RefitAborted("aborted")
        twice = RefitAborted(str(once))
        assert str(twice).count(REFIT_ABORTED_TOKEN) == 1

    def test_it_survives_pickling(self):
        """Ray pickles exceptions across the actor boundary."""
        revived = pickle.loads(pickle.dumps(RefitAborted("aborted mid-collective")))
        assert is_refit_abort(revived)
        assert str(revived).count(REFIT_ABORTED_TOKEN) == 1

    def test_a_real_refit_aborted_is_recognised(self):
        assert is_refit_abort(RefitAborted("deadline exceeded"))

    def test_the_vllm_flattened_form_is_recognised(self):
        """The exact shape vLLM reconstructs: bare Exception, message preserved."""
        inside_the_engine = RefitAborted(
            "the refit was aborted after its 12.5s deadline"
        )
        flattened = Exception(
            f"Call to nccl_reshard_refit method failed: {inside_the_engine}"
        )

        assert not isinstance(flattened, RefitAborted), "premise: the type is gone"
        assert is_refit_abort(flattened), "but the abort must still be recognised"

    def test_an_unrelated_failure_is_not_mistaken_for_an_abort(self):
        """A real refit bug must not be relabelled as a deliberate abort and retried."""
        assert not is_refit_abort(RuntimeError("CUDA out of memory"))
        assert not is_refit_abort(
            Exception("Call to nccl_reshard_refit method failed: shape mismatch")
        )


class TestAnEscapeAfterTheAbortIsNamedAsOne:
    """Whatever a transport raises after its communicator is aborted must be RefitAborted.

    Only ``StatelessProcessGroup.broadcast`` names the abort, and the nccl_reshard bulk
    path never calls it -- it hands ``nccl_communicator`` straight to xferdtensor. So on
    that transport an abort landing on any parameter but the last escaped as an unrelated
    type, ``_sync_weights`` (which catches only ``RefitAborted`` and ``RayActorError``)
    missed it, and the rebuild-and-retry never ran.
    """

    def test_an_attribute_error_after_the_abort_becomes_refit_aborted(self):
        """The communicator is None post-abort, so the next use raises AttributeError."""
        with pytest.raises(RefitAborted):
            with RefitAbortWatchdog(_FakeGroup(), 0.05):
                time.sleep(0.4)
                raise AttributeError("'NoneType' object has no attribute 'split'")

    def test_the_original_error_is_kept_as_the_cause(self):
        """The abort is the cause; the transport error is still needed to diagnose it."""
        original = AttributeError("'NoneType' object has no attribute 'send'")
        with pytest.raises(RefitAborted) as caught:
            with RefitAbortWatchdog(_FakeGroup(), 0.05):
                time.sleep(0.4)
                raise original
        assert caught.value.__cause__ is original

    def test_an_arbitrary_transport_error_is_translated_too(self):
        """Stands in for nccl4py's NcclInvalid, which is not importable here."""

        class NcclInvalid(RuntimeError):
            pass

        with pytest.raises(RefitAborted):
            with RefitAbortWatchdog(_FakeGroup(), 0.05):
                time.sleep(0.4)
                raise NcclInvalid("communicator pointer was zeroed")

    def test_a_refit_aborted_is_not_wrapped_in_another_one(self):
        already = RefitAborted("named by broadcast")
        with pytest.raises(RefitAborted) as caught:
            with RefitAbortWatchdog(_FakeGroup(), 0.05):
                time.sleep(0.4)
                raise already
        assert caught.value is already

    def test_an_error_without_a_fired_deadline_passes_through(self):
        """The guard must not relabel failures it had nothing to do with."""
        with pytest.raises(ValueError, match="unrelated"):
            with RefitAbortWatchdog(_FakeGroup(), 30.0) as guard:
                raise ValueError("unrelated")
        assert guard.fired is False

    def test_a_disarmed_guard_never_translates(self):
        with pytest.raises(ValueError, match="unrelated"):
            with RefitAbortWatchdog(_FakeGroup(), None):
                raise ValueError("unrelated")

    def test_a_keyboard_interrupt_is_left_alone(self):
        """Not a consequence of the abort, and relabelling it hides why we are exiting."""
        with pytest.raises(KeyboardInterrupt):
            with RefitAbortWatchdog(_FakeGroup(), 0.05):
                time.sleep(0.4)
                raise KeyboardInterrupt

    def test_the_clean_return_still_reports_through_fired(self):
        """No exception to translate, so the existing `if guard.fired:` sites still work."""
        with RefitAbortWatchdog(_FakeGroup(), 0.05) as guard:
            time.sleep(0.4)
        assert guard.fired is True


class TestDoesNotFire:
    def test_a_block_that_finishes_in_time_is_left_alone(self):
        group = _FakeGroup()
        with RefitAbortWatchdog(group, 5.0) as guard:
            pass
        assert guard.fired is False
        assert group.abort_calls == 0

    def test_an_exception_in_the_block_still_disarms(self):
        group = _FakeGroup()
        with pytest.raises(ValueError):
            with RefitAbortWatchdog(group, 5.0):
                raise ValueError("boom")
        time.sleep(0.1)
        assert group.abort_calls == 0, "a failed refit must not also be aborted"


class TestThreadHygiene:
    def test_threads_do_not_accumulate_across_refits(self):
        """A run refits every step, so a leak here is unbounded over a long job."""
        group = _FakeGroup()
        before = threading.active_count()
        for _ in range(50):
            with RefitAbortWatchdog(group, 5.0):
                pass
        assert threading.active_count() <= before + 1
        assert group.abort_calls == 0


class TestEveryRefitEntrypointAcceptsTheDeadline:
    """EVERY refit entrypoint must take refit_timeout_s, or the run dies at the first sync.

    This exists because one of them did not. ``update_weights_from_collective`` got the
    parameter; ``update_weights_from_collective_async`` did not -- and the async engine is
    what the recovery test actually uses, so the very first refit failed with
    ``TypeError: got an unexpected keyword argument 'refit_timeout_s'`` from inside Ray's
    argument validation.

    No behavioural test caught it: the call crosses a Ray actor boundary, where the
    signature is checked at dispatch rather than by any import, and the fakes in these
    suites do not model that. A signature assertion is cheap and covers exactly the gap.

    Parametrized over BOTH transports because the same omission then repeated itself: the
    nccl_reshard path was plumbed nowhere at all, so ``recovery-reshard`` ran with no
    deadline and would have wedged on a mid-refit death exactly as before -- while
    passing, because its kill lands at a step boundary.
    """

    @pytest.mark.parametrize(
        ("module", "cls", "method"),
        [
            (
                "nemo_rl.models.generation.vllm.vllm_worker_async",
                "VllmAsyncGenerationWorker",
                "update_weights_from_collective_async",
            ),
            (
                "nemo_rl.models.generation.vllm.vllm_worker_async",
                "VllmAsyncGenerationWorker",
                "nccl_reshard_refit_async",
            ),
            (
                "nemo_rl.models.generation.vllm.vllm_generation",
                "VllmGeneration",
                "nccl_reshard_refit",
            ),
            (
                "nemo_rl.models.policy.lm_policy",
                "Policy",
                "nccl_reshard_refit",
            ),
            (
                "nemo_rl.models.policy.lm_policy",
                "Policy",
                "broadcast_weights_for_collective",
            ),
        ],
    )
    def test_entrypoint_accepts_the_deadline(self, module, cls, method):
        import importlib
        import inspect

        mod = importlib.import_module(module)
        fn = getattr(getattr(mod, cls), method)
        assert "refit_timeout_s" in inspect.signature(fn).parameters, (
            f"{cls}.{method} must accept refit_timeout_s; the controller passes it on "
            "every refit and Ray rejects the call otherwise"
        )

    def test_the_reshard_synchronizer_takes_it_too(self):
        """The factory constructs it by keyword; a missing parameter is a TypeError at setup.

        It was simply not passed -- the factory built the reshard synchronizer without it
        while passing it to the collective one two branches below, so the whole abort
        mechanism was absent on that transport with nothing to indicate it.
        """
        import inspect

        from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
            NcclReshardWeightSynchronizer,
        )

        params = inspect.signature(NcclReshardWeightSynchronizer.__init__).parameters
        assert "refit_timeout_s" in params
        assert params["refit_timeout_s"].default is None

    def test_the_factory_forwards_it_to_both_transports(self):
        """A parameter the factory accepts and drops is worse than one it never had."""
        import inspect

        from nemo_rl.weight_sync import factory

        source = inspect.getsource(factory)
        assert source.count("refit_timeout_s=refit_timeout_s") >= 2, (
            "both CollectiveWeightSynchronizer and NcclReshardWeightSynchronizer must "
            "be constructed with the deadline"
        )

    def test_the_two_entrypoints_agree(self):
        """The sync and async paths are chosen by config, so they must stay interchangeable."""
        import inspect

        from nemo_rl.models.generation.vllm.vllm_worker_async import (
            VllmAsyncGenerationWorker,
        )

        async_sig = inspect.signature(
            VllmAsyncGenerationWorker.update_weights_from_collective_async
        )
        assert "refit_timeout_s" in async_sig.parameters
        assert async_sig.parameters["refit_timeout_s"].default is None, (
            "must default to None so an unconfigured run is unchanged"
        )


class TestMultipleGroups:
    """The nccl_reshard transport blocks in one of two communicator families.

    Bulk weights move over per-PP-stage groups, then the remainder broadcasts over the
    shared model_update_group. Nothing at the watchdog's level can tell which one a hang
    is in, so it aborts all of them.
    """

    def test_every_group_is_aborted(self):
        groups = [_FakeGroup(), _FakeGroup(), _FakeGroup()]
        with RefitAbortWatchdog(groups, 0.05) as guard:
            time.sleep(0.4)
        assert guard.fired is True
        assert [g.abort_calls for g in groups] == [1, 1, 1]

    def test_one_failing_abort_does_not_strand_the_others(self):
        """The group that raises may not be the one the caller is blocked in.

        Giving up on the rest would leave it hung on a group that would have released.
        """
        groups = [_FakeGroup(fail=True), _FakeGroup(), _FakeGroup()]
        with RefitAbortWatchdog(groups, 0.05) as guard:
            time.sleep(0.4)
        assert guard.fired is True
        assert [g.abort_calls for g in groups] == [1, 1, 1]

    def test_none_entries_are_ignored(self):
        """A worker with no PP group passes None in the list rather than branching."""
        real = _FakeGroup()
        with RefitAbortWatchdog([None, real], 0.05) as guard:
            time.sleep(0.4)
        assert guard.fired is True
        assert real.abort_calls == 1

    def test_a_list_of_nothing_stays_disarmed(self):
        with RefitAbortWatchdog([None, None], 0.05) as guard:
            assert not guard.armed
        assert guard.fired is False

    def test_an_empty_list_stays_disarmed(self):
        with RefitAbortWatchdog([], 0.05) as guard:
            assert not guard.armed
        assert guard.fired is False


class TestBothTransportsCanBeHeldOpen:
    """The fault-injection hook must be reachable on BOTH refit receives.

    Source inspection rather than import: vllm_backend does `import vllm` at module
    scope, so the default unit lane cannot import it at all. The property being
    protected is structural -- "this call site exists inside the guarded block" -- and
    that is exactly what a source assertion can check.

    Worth guarding because the reshard abort path is otherwise invisible to unit tests:
    the deadline is plumbed and signature-tested, but whether a reshard refit can be
    made to abort at all depends on this one call, and only a GPU functional test
    (recovery-reshard-refit) can prove it end to end.
    """

    @staticmethod
    def _backend_source() -> str:
        from pathlib import Path

        import nemo_rl

        path = (
            Path(nemo_rl.__file__).parent
            / "models"
            / "generation"
            / "vllm"
            / "vllm_backend.py"
        )
        return path.read_text()

    def _guarded_body(self, method: str) -> str:
        """Return the text between `def <method>` and the next top-level def."""
        src = self._backend_source()
        start = src.index(f"    def {method}(")
        nxt = src.index("\n    def ", start + 1)
        return src[start:nxt]

    def test_the_collective_receive_can_be_held(self):
        body = self._guarded_body("update_weights_from_collective")
        assert "hold_refit_for_fault_injection()" in body

    def test_the_reshard_receive_can_be_held(self):
        """The gap this closes: reshard had the deadline but no way to aim at it."""
        body = self._guarded_body("nccl_reshard_refit")
        assert "hold_refit_for_fault_injection()" in body

    def test_the_hold_is_inside_the_watchdog_not_before_it(self):
        """Order matters. Held outside the guard, the deadline clock never starts and
        the victim is killed during an unguarded pause -- the run would hang exactly as
        it did before the watchdog existed, and the test would look like it passed."""
        for method in ("update_weights_from_collective", "nccl_reshard_refit"):
            body = self._guarded_body(method)
            guard_at = body.index("with RefitAbortWatchdog(")
            hold_at = body.index("hold_refit_for_fault_injection()")
            assert hold_at > guard_at, (
                f"{method}: the hold must be INSIDE the RefitAbortWatchdog block"
            )


class TestTheStreamWaitIsBounded:
    """A CUDA sync that never returns is the one wedge the watchdog cannot break.

    Aborting a communicator does not retire kernels already enqueued on a stream. Job
    6485245: both policy workers parked in torch.cuda.synchronize() at
    megatron_policy_worker.py:2940 for 1801s after their own abort had logged, while the
    generation workers had already unwound and gone idle. The guarded block never exits,
    so no exception translation and no `if guard.fired:` can ever run.
    """

    @staticmethod
    def _fake_torch(monkeypatch, *, completes: bool):
        """Stand in for torch.cuda, so this needs no GPU and no real stuck kernel."""
        calls = {"synchronize": 0, "recorded": []}

        class _Event:
            def record(self, stream):
                calls["recorded"].append(stream)

            def query(self):
                return completes

        cuda = SimpleNamespace(
            Event=_Event,
            synchronize=lambda: calls.__setitem__(
                "synchronize", calls["synchronize"] + 1
            ),
        )
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
        return calls

    def test_work_that_never_retires_gives_up_instead_of_blocking(self, monkeypatch):
        self._fake_torch(monkeypatch, completes=False)
        started = time.monotonic()

        with pytest.raises(RefitAborted, match="did not retire"):
            sync_stream_within(object(), 0.3, "the bulk parameter transfer")

        assert time.monotonic() - started < 5.0, "it must give up, not block"

    def test_the_failure_names_what_was_waiting(self, monkeypatch):
        """Two sites share this; the message has to say which one stalled."""
        self._fake_torch(monkeypatch, completes=False)
        with pytest.raises(RefitAborted, match="the misc broadcast"):
            sync_stream_within(object(), 0.1, "the misc broadcast")

    def test_it_is_recognised_as_an_abort_downstream(self, monkeypatch):
        """It must reach the same handler as any other abort, incl. across vLLM's RPC."""
        self._fake_torch(monkeypatch, completes=False)
        with pytest.raises(RefitAborted) as caught:
            sync_stream_within(object(), 0.1, "the bulk parameter transfer")
        assert is_refit_abort(caught.value)
        assert REFIT_ABORTED_TOKEN in str(caught.value)

    def test_the_happy_path_still_ends_in_a_device_wide_synchronize(self, monkeypatch):
        """Behaviour must be identical to before when nothing is wrong."""
        calls = self._fake_torch(monkeypatch, completes=True)
        sync_stream_within("a-stream", 30.0, "the bulk parameter transfer")
        assert calls["synchronize"] == 1
        assert calls["recorded"] == ["a-stream"]

    @pytest.mark.parametrize("budget", [None, 0, -1.0])
    def test_no_deadline_keeps_the_original_unbounded_wait(self, monkeypatch, budget):
        """A run that configures no refit deadline behaves exactly as it did."""
        calls = self._fake_torch(monkeypatch, completes=False)
        sync_stream_within(object(), budget, "the bulk parameter transfer")
        assert calls["synchronize"] == 1, "falls back to torch.cuda.synchronize()"
        assert calls["recorded"] == [], "and records no event at all"


class TestTheRefitRunsOffTheActorsEventLoop:
    """A blocking refit on a Ray actor starves every other call to that actor.

    Ray runs a sync actor method directly in the event loop -- sync_to_async wraps it as
    `async def wrapper: return func(...)`, with no executor -- so max_concurrency cannot
    help: it interleaves coroutines, and a coroutine blocked in C never yields.

    Job 6509685 is the cost. The controller gave up on the stuck refit and called
    init_collective to rebuild, that call queued behind the refit still holding the loop,
    rank 0 never created the rendezvous store, and the surviving generation worker timed
    out dialling it for 300s -- twice -- before the run ended at 690s.
    """

    def test_the_loop_stays_free_while_the_blocking_call_runs(self):
        """The property the rebuild depends on: another call can still be serviced."""
        release = threading.Event()
        serviced = []

        async def _scenario():
            refit = asyncio.ensure_future(await_off_loop(release.wait))
            # Stands in for the recovery's init_collective arriving while the refit is
            # still blocked. On the event loop, as Ray would run it.
            for _ in range(20):
                await asyncio.sleep(0.01)
                serviced.append(1)
            assert not refit.done(), "premise: the refit is still blocked"
            release.set()
            await refit

        asyncio.run(_scenario())
        assert serviced, "the loop was starved; the rebuild could never be serviced"

    def test_the_result_comes_back(self):
        assert asyncio.run(await_off_loop(lambda: "refit-done")) == "refit-done"

    def test_a_failure_propagates_rather_than_being_swallowed(self):
        def _boom():
            raise RefitAborted("aborted mid-collective")

        with pytest.raises(RefitAborted, match="aborted mid-collective"):
            asyncio.run(await_off_loop(_boom))

    def test_the_thread_cannot_block_interpreter_exit(self):
        """asyncio.to_thread's pool is non-daemon and joined at exit, so it is not used."""
        seen = []
        original = threading.Thread

        def _record(*args, **kwargs):
            thread = original(*args, **kwargs)
            seen.append(thread)
            return thread

        with mock.patch.object(threading, "Thread", _record):
            asyncio.run(await_off_loop(lambda: None))

        assert seen and all(t.daemon for t in seen)


def test_the_release_runs_on_the_callers_cuda_device(monkeypatch):
    """The CUDA device is thread-local, and a fresh thread starts on device 0.

    Job 6524733 is the regression this pins. recovery-reshard-refit went from passing to a
    1800s wedge, and its victim was SIGKILLed -- ActorDiedError, genuinely gone -- so the
    abort had no peer to wait on and still did not return within 30s. The same trap cost
    job 6510914 a run when the refit first moved off the event loop, which is why
    megatron_policy_worker asserts against device drift after setup.
    """
    import torch

    from nemo_rl.distributed.refit_watchdog import release_within

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    device_when_released = []
    current = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: current.append(d))

    release_within(
        lambda: device_when_released.append(current[-1] if current else None),
        5.0,
        "the test communicator",
    )

    assert device_when_released == [3], (
        "release_within must set the caller's device inside the release thread; "
        f"saw {device_when_released}. ncclCommAbort has to run against the "
        "communicator's own device or it does not retire."
    )


def test_the_release_still_runs_without_cuda(monkeypatch):
    """Unit tests and CPU-only deployments must not trip over the device pinning."""
    import torch

    from nemo_rl.distributed.refit_watchdog import release_within

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ran = []
    release_within(lambda: ran.append(True), 5.0, "the test communicator")
    assert ran == [True]


class TestStandDown:
    """The deadline is for a SILENT peer, not a dead one.

    A dead peer closes its sockets and NCCL unblocks the survivors by itself -- job 6405953
    recovered the reshard kill variant that way with RefitAborted appearing zero times,
    before any deadline existed. Once it existed it started winning that race, and the
    abort orphans kernels on the trainers' streams, so the rebuild that would have worked
    cannot. The controller therefore stands the deadline down on a confirmed actor death.
    """

    def test_standing_down_cancels_the_deadline_without_firing_it(self):
        from nemo_rl.distributed.refit_watchdog import (
            RefitAbortWatchdog,
            stand_down_armed_watchdogs,
        )

        group = _FakeGroup()
        with RefitAbortWatchdog(group, 0.05) as guard:
            assert stand_down_armed_watchdogs() == 1
            time.sleep(0.3)  # well past the deadline it would otherwise have fired on
            assert not guard.fired, (
                "a stood-down watchdog must not fire; if it does, the abort still orphans "
                "the trainers' streams and the rebuild it was meant to protect cannot run"
            )
        assert group.abort_calls == 0, (
            f"a stood-down watchdog must not abort anything; saw {group.abort_calls}"
        )

    def test_a_watchdog_that_has_exited_is_no_longer_standable_down(self):
        """The registry must not leak, or a later death would poke a finished guard."""
        from nemo_rl.distributed.refit_watchdog import (
            RefitAbortWatchdog,
            stand_down_armed_watchdogs,
        )

        with RefitAbortWatchdog(_FakeGroup(), 10.0):
            pass
        assert stand_down_armed_watchdogs() == 0

    def test_a_disarmed_watchdog_is_never_registered(self):
        from nemo_rl.distributed.refit_watchdog import (
            RefitAbortWatchdog,
            stand_down_armed_watchdogs,
        )

        with RefitAbortWatchdog(_FakeGroup(), None):
            assert stand_down_armed_watchdogs() == 0


class TestContextLost:
    """The one fault this stack detects but cannot recover from, and its exact scope.

    nccl_reshard + a fault while the bulk transfer is in flight. Nothing else: job 6584636
    measured the packed-broadcast transport taking the same fault, firing the same deadline
    4x, raising RefitAborted 10x, rebuilding and recovering with zero stuck aborts -- and
    reshard recovering too when the fault lands at a step boundary.
    """

    def test_a_stream_that_never_drains_is_marked_unrecoverable(self):
        import torch

        from nemo_rl.distributed.refit_watchdog import (
            RefitAborted,
            is_refit_context_lost,
            sync_stream_within,
        )

        never = mock.Mock()
        never.query.return_value = False
        with mock.patch.object(torch.cuda, "Event", return_value=never):
            with pytest.raises(RefitAborted) as caught:
                sync_stream_within(mock.Mock(), 0.01, "the bulk parameter transfer")
        assert is_refit_context_lost(caught.value), (
            "an abort that orphans enqueued kernels must be marked unrecoverable, or the "
            "controller enters a rebuild that cannot complete and wedges instead of failing"
        )

    def test_an_ordinary_abort_is_not_marked_unrecoverable(self):
        """The scope guard. A plain RefitAborted is recoverable and must stay so.

        The packed-broadcast transport raises these and recovers from them; marking them
        would turn four passing variants into fail-fast.
        """
        from nemo_rl.distributed.refit_watchdog import (
            RefitAborted,
            is_refit_context_lost,
        )

        assert not is_refit_context_lost(RefitAborted("a peer stopped participating"))

    def test_the_marker_survives_a_boundary_that_drops_the_type(self):
        """vLLM's EngineCore RPC stringifies and re-raises as a bare Exception."""
        from nemo_rl.distributed.refit_watchdog import (
            RefitAborted,
            is_refit_context_lost,
        )

        original = RefitAborted(
            "[refit-context-lost] the bulk parameter transfer did not retire"
        )
        flattened = Exception(f"Call to nccl_reshard_refit method failed: {original}")
        assert is_refit_context_lost(flattened)
