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

import threading
import pickle
import time

import pytest

from nemo_rl.distributed.refit_watchdog import (
    REFIT_ABORTED_TOKEN,
    RefitAborted,
    RefitAbortWatchdog,
    is_refit_abort,
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
