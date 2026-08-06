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
import time

import pytest

from nemo_rl.distributed.refit_watchdog import RefitAbortWatchdog


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
