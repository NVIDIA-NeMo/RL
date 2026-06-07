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

"""Unit tests for ``StatelessProcessGroup``'s interruptible stream wait.

The cross-cluster weight-sync comm uses the raw ``nccl`` bindings in
non-blocking mode with NO torch ``ProcessGroupNCCL`` watchdog. A blind
``cudaStreamSynchronize`` on a collective whose peer was SIGKILL'd
mid-broadcast therefore blocks UNINTERRUPTIBLY forever — nothing aborts
the spinning kernel. ``_poll_until_done`` is the missing watchdog,
inlined onto the calling thread: it polls a completion signal AND the
comm's async-error state, and on peer error / timeout it aborts the comm
(unblocking the hung kernel) and raises so the caller can re-init.

These tests pin that decision logic. The CUDA-event wiring lives in the
thin ``synchronize_or_abort`` wrapper (exercised e2e on a GPU); the core
loop takes the completion signal as an injected ``done_fn`` so it is
unit-testable on CPU with fakes for the nccl bindings.
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import nemo_rl.distributed.stateless_process_group as spg
from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup


def _make_group():
    """Construct a StatelessProcessGroup without touching the TCPStore."""
    group = object.__new__(StatelessProcessGroup)
    group.rank = 3
    group.world_size = 8
    return group


def _fake_bindings(async_error_value):
    """A stand-in for ``nccl.bindings`` with controllable async-error state.

    ``Success``/``InProgress`` mirror the real enum's role (healthy /
    still-running); any other value is treated as a peer/NCCL error.
    """
    comm_abort = MagicMock(name="comm_abort")
    return SimpleNamespace(
        Result=SimpleNamespace(Success=0, InProgress=7),
        comm_get_async_error=MagicMock(return_value=async_error_value),
        comm_abort=comm_abort,
        get_last_error=MagicMock(return_value="fake nccl error"),
    )


def test_poll_until_done_returns_when_complete_without_abort():
    """Healthy path: stream completes, comm never aborted, no raise."""
    group = _make_group()
    # done_fn flips to True on the 3rd poll; comm stays healthy (Success).
    calls = {"n": 0}

    def done_fn():
        calls["n"] += 1
        return calls["n"] >= 3

    fake = _fake_bindings(async_error_value=0)  # Success
    with patch.object(spg, "_nccl_bindings", fake):
        group._poll_until_done(
            done_fn=done_fn,
            comm_ptr=12345,
            phase="broadcast_sync",
            timeout=timedelta(seconds=5),
            poll_interval_s=0.001,
        )

    assert calls["n"] >= 3
    fake.comm_abort.assert_not_called()


def test_poll_until_done_aborts_and_raises_on_peer_error():
    """Peer death: async-error flips, comm is aborted, RuntimeError raised."""
    group = _make_group()
    fake = _fake_bindings(async_error_value=2)  # neither Success nor InProgress

    with patch.object(spg, "_nccl_bindings", fake):
        with pytest.raises(RuntimeError, match="async_state=2"):
            group._poll_until_done(
                done_fn=lambda: False,  # stream never completes (hung kernel)
                comm_ptr=999,
                phase="broadcast_sync",
                timeout=timedelta(seconds=5),
                poll_interval_s=0.001,
            )

    # The abort is what unblocks the hung kernel — it MUST be called, on
    # the wedged comm pointer.
    fake.comm_abort.assert_called_once_with(999)


def test_poll_until_done_aborts_and_raises_on_timeout():
    """Backstop: comm looks healthy but never completes -> timeout aborts."""
    group = _make_group()
    fake = _fake_bindings(async_error_value=0)  # Success the whole time

    with patch.object(spg, "_nccl_bindings", fake):
        with pytest.raises(RuntimeError, match="timed out"):
            group._poll_until_done(
                done_fn=lambda: False,
                comm_ptr=7,
                phase="broadcast_sync",
                timeout=timedelta(seconds=0.05),
                poll_interval_s=0.001,
            )

    fake.comm_abort.assert_called_once_with(7)


def test_poll_until_done_tolerates_in_progress_state():
    """``InProgress`` is not an error — keep polling until done."""
    group = _make_group()
    calls = {"n": 0}

    def done_fn():
        calls["n"] += 1
        return calls["n"] >= 4

    fake = _fake_bindings(async_error_value=7)  # InProgress
    with patch.object(spg, "_nccl_bindings", fake):
        group._poll_until_done(
            done_fn=done_fn,
            comm_ptr=1,
            phase="broadcast_sync",
            timeout=timedelta(seconds=5),
            poll_interval_s=0.001,
        )

    fake.comm_abort.assert_not_called()
