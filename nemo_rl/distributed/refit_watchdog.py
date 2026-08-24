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

"""Break a refit collective that a dead peer has left hanging.

WHY THIS HAS TO RUN INSIDE THE WORKER. A generation rank that dies mid-refit leaves the
surviving ranks blocked in NCCL with no timeout and no error -- observed directly, both
policy workers stuck in ``packed_broadcast_producer -> cuda stream synchronize`` while the
run sat wedged for 1801s. The controller cannot rescue them by RPC: the collective blocks
the worker actor's event loop, and the worker actors carry no ``max_concurrency``, so an
incoming ``abort`` call would queue behind the very operation it is meant to interrupt.
The abort must therefore come from a thread already inside the process, which is exactly
the arrangement the design's NCCL spike validated (a survivor released 0.15s after another
thread called ``abort()``).

TWO SEMANTICS THAT SHAPE THE API, both established by that spike:

1. **An aborted collective returns without raising.** So the caller cannot detect this with
   ``try``/``except``; it has to ask whether the abort fired. Hence :attr:`fired`.
2. **The destination buffers hold partial data afterwards.** A generation shard caught
   mid-refit holds a mix of old and new weights and must not serve until a later refit
   completes. Callers are responsible for propagating that -- see ``RefitAborted``.

Inert unless armed with a positive timeout, so a run that does not configure one behaves
exactly as before, down to not starting a thread.
"""

import threading
from collections.abc import Sequence
from types import TracebackType
from typing import Optional, Protocol, Union


class _Abortable(Protocol):
    def abort(self) -> None: ...


class RefitAborted(RuntimeError):
    """A refit was cut short because a peer stopped participating.

    Raised by the worker that armed the watchdog, not by NCCL -- the aborted call itself
    returns cleanly, so this is the only signal the caller gets.
    """


class RefitAbortWatchdog:
    """Abort the given group(s) if the guarded block outlives ``timeout_s``.

    Use as a context manager around the collective::

        with RefitAbortWatchdog(self.model_update_group, timeout_s) as guard:
            ...collective...
        if guard.fired:
            raise RefitAborted(...)

    A sequence may be passed instead of one group, and the nccl_reshard transport needs
    that: it moves weights over per-PP-stage bulk groups and then broadcasts the
    remainder over the shared ``model_update_group``, so a hang can be in either family
    and nothing at this level can tell which. Aborting all of them costs nothing --
    ``abort()`` is idempotent and safe on a group that never built a communicator -- and
    the recovery rebuilds every family regardless.

    ``timeout_s`` of ``None`` or ``<= 0`` disarms it entirely: no thread is started and
    ``fired`` stays False, so the default configuration is bit-for-bit the old behaviour.
    """

    def __init__(
        self,
        group: Optional[Union[_Abortable, Sequence[Optional[_Abortable]]]],
        timeout_s: Optional[float],
    ) -> None:
        if group is None:
            groups: list[_Abortable] = []
        elif isinstance(group, Sequence):
            groups = [g for g in group if g is not None]
        else:
            groups = [group]
        self._groups = groups
        self._timeout_s = timeout_s
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fired = False

    @property
    def armed(self) -> bool:
        return (
            bool(self._groups) and self._timeout_s is not None and self._timeout_s > 0
        )

    @property
    def fired(self) -> bool:
        """True if the deadline passed and abort() was called."""
        return self._fired

    def _watch(self) -> None:
        assert self._timeout_s is not None
        # wait() returns False on timeout, True if the guarded block finished first. The
        # normal path is therefore "wait, observe True, do nothing" -- the thread never
        # touches the group unless the collective genuinely overran.
        if self._done.wait(self._timeout_s):
            return
        self._fired = True
        # Printed because the abort is otherwise invisible until something happens to
        # raise RefitAborted, and "the deadline never fired" and "it fired but the
        # verdict was lost" are different bugs that look identical in a log. Three
        # hardware runs were spent unable to tell them apart.
        print(
            f"  refit: deadline exceeded after {self._timeout_s}s; "
            f"aborting {len(self._groups)} communicator group(s)",
            flush=True,
        )
        for group in self._groups:
            try:
                group.abort()
            except Exception:  # noqa: BLE001
                # A failed abort leaves the caller blocked, which is the situation we
                # were already in; swallowing keeps the watchdog thread from dying
                # silently mid-way and is strictly no worse than not having tried.
                #
                # Per group, not around the loop: with several families the blocked one
                # may not be the one that raised, and giving up on the rest would leave
                # the caller hung on a group that would have aborted cleanly.
                pass

    def __enter__(self) -> "RefitAbortWatchdog":
        if self.armed:
            print(
                f"  refit: watchdog armed, deadline {self._timeout_s}s over "
                f"{len(self._groups)} communicator group(s)",
                flush=True,
            )
            self._thread = threading.Thread(
                target=self._watch, name="refit-abort-watchdog", daemon=True
            )
            self._thread.start()
        else:
            # Says which of the two reasons, because they need opposite fixes: no
            # deadline configured is a config question, no groups is a plumbing one.
            print(
                "  refit: watchdog NOT armed ("
                + (
                    f"no deadline configured, timeout={self._timeout_s}"
                    if not (self._timeout_s and self._timeout_s > 0)
                    else "no communicator groups were passed"
                )
                + ")",
                flush=True,
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # Join, so a run that refits every step cannot accumulate one thread per step.
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # A refit is many operations. The abort releases the one in flight; a LATER one on
        # the aborted group fails with whatever that transport happens to raise. Only
        # StatelessProcessGroup.broadcast() names the abort, and the nccl_reshard bulk path
        # never calls it -- it hands nccl_communicator straight to xferdtensor -- so the
        # escape is an AttributeError (communicator now None) or an nccl4py NcclInvalid (a
        # local bound before the abort, used after). _sync_weights catches only
        # (RefitAborted, RayActorError), so the run died instead of rebuilding.
        #
        # Translated here rather than at each call site because this is the one boundary
        # every escape must cross, and because a per-site check cannot see an abort that
        # lands MID-call: _exchange_exact_overlaps binds the communicator into a parameter
        # and uses it several statements later.
        #
        # Cannot fire spuriously: _fired is set only after the deadline elapsed and abort()
        # ran. The `exc is None` path is untouched, so the existing `if guard.fired:` sites
        # still raise their own more specific messages.
        #
        # Exception, not BaseException: a KeyboardInterrupt or SystemExit that happens to
        # land inside a fired window is not a consequence of the abort, and relabelling it
        # would hide the real reason the process is going away.
        if (
            self._fired
            and isinstance(exc, Exception)
            and not isinstance(exc, RefitAborted)
        ):
            raise RefitAborted(
                "the refit was aborted after its "
                f"{self._timeout_s}s deadline; the error below is a consequence of the "
                "abort, not its cause"
            ) from exc


def hold_refit_for_fault_injection() -> None:
    """Block a refit receive while a test holds it open. Inert unless asked.

    Does nothing unless ``NRL_REFIT_HOLD_FILE`` names a path that exists, so a real run
    pays one ``os.path.exists`` per refit and behaves no differently.

    It exists because "kill a shard during the refit" is otherwise untestable. A refit on
    the functional test's model takes ~0.10s, and the harness has to notice one started
    and then find and kill a process: job 5925668 aimed at the collective and landed in
    the RPC epilogue instead. That is a real failure mode and worth handling, but it is
    not the one the test claimed to cover, so the abort-and-rebuild path went unexercised
    while the run still reported a result.

    A file rather than a fixed delay because the harness has to hold *one specific*
    refit -- the one after the step it kills at. A delay on every refit would slow the
    whole run for the sake of one moment and still not be aimed at it.

    Bounded by ``NRL_REFIT_HOLD_MAX_S`` so a harness that dies mid-test cannot wedge the
    worker it was holding.
    """
    import os

    hold_file = os.environ.get("NRL_REFIT_HOLD_FILE")
    if not hold_file or not os.path.exists(hold_file):
        return

    import time

    deadline = time.monotonic() + float(
        os.environ.get("NRL_REFIT_HOLD_MAX_S", "120") or 120
    )
    print(
        f"  refit: holding the receive open, waiting for {hold_file} to be removed "
        "(NRL_REFIT_HOLD_FILE fault-injection hook)",
        flush=True,
    )
    while os.path.exists(hold_file) and time.monotonic() < deadline:
        time.sleep(0.1)
    print("  refit: hold released; entering the receive", flush=True)
