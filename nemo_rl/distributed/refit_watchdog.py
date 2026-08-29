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

import asyncio
import threading
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Optional, Protocol, Union


class _Abortable(Protocol):
    def abort(self) -> None: ...


# Carried in every RefitAborted message so the abort survives a boundary that preserves
# the message but drops the type. vLLM's EngineCore RPC is exactly that boundary: it
# stringifies the worker's exception into ``failure_message`` (v1/engine/core.py) and
# re-raises it client-side as a bare ``Exception`` (v1/engine/core_client.py). Nothing we
# can configure changes that, so the signal has to travel in the text.
REFIT_ABORTED_TOKEN = "[refit-aborted]"

# A SECOND, NARROWER MARKER: the abort left this trainer's CUDA context unusable, so the
# run cannot be recovered and must end. Carried in the message for the same reason as the
# token above -- vLLM's EngineCore RPC preserves the text and drops the type.
#
# Only sync_stream_within raises with this, and it is reachable only from
# _nccl_reshard_refit. That is the whole scope of the limitation: the packed-broadcast
# transport takes the same fault, fires the same deadline, aborts, rebuilds and recovers
# with zero stuck aborts (job 6584636: null-refit and null-refit-frozen, deadline fired 4x,
# RefitAborted 10x, not one release timeout). Reshard recovers too when the fault lands at
# a step boundary rather than inside the bulk transfer.
REFIT_CONTEXT_LOST_TOKEN = "[refit-context-lost]"


class RefitAborted(RuntimeError):
    """A refit was cut short because a peer stopped participating.

    Raised by the worker that armed the watchdog, not by NCCL -- the aborted call itself
    returns cleanly, so this is the only signal the caller gets.

    The constructor prefixes :data:`REFIT_ABORTED_TOKEN` to the message. Idempotent, so a
    re-raise or an unpickle (``RuntimeError.__reduce__`` replays ``args``) does not stack
    prefixes.
    """

    def __init__(self, *args: object) -> None:
        if args and isinstance(args[0], str) and REFIT_ABORTED_TOKEN not in args[0]:
            args = (f"{REFIT_ABORTED_TOKEN} {args[0]}", *args[1:])
        super().__init__(*args)


def is_refit_context_lost(error: BaseException) -> bool:
    """True when the abort orphaned GPU work and no rebuild on that device can succeed.

    Recovery after this is not merely unlikely, it is impossible in-process, and four runs
    established that by elimination rather than argument. Job 6521181's py-spy dump caught
    both trainers in ``init_nccl_communicator`` with frame-less ``ncclCommAbort`` threads 25
    minutes on. Job 6523731 killed the frozen victim before the rebuild, closing its
    sockets, and they wedged identically. Job 6582457 pinned the release to the caller's
    CUDA device, with a SIGKILLed peer, and it still did not return. Job 6584636 stood the
    deadline down on confirmed death and lost the race to Ray's own detection.

    So the controller stops trying. Detecting this and ending the run in seconds is the
    supported behaviour; recovering from it is left to a future change -- see
    design_vllm_fault_tolerance.md section 8.5.7.
    """
    return REFIT_CONTEXT_LOST_TOKEN in str(error)


def is_refit_abort(error: BaseException) -> bool:
    """True for a RefitAborted, or for one flattened to a plain Exception in transit.

    The type check alone is not enough anywhere downstream of a vLLM ``collective_rpc``:
    the abort is raised inside the engine core process and arrives at the Ray actor as
    ``Exception("Call to <method> method failed: ...")``. A bare ``except RefitAborted``
    there is dead code, which is what job 6484412 demonstrated -- the deadline fired, the
    abort was named, and the run still wedged because the handler never matched.
    """
    return isinstance(error, RefitAborted) or REFIT_ABORTED_TOKEN in str(error)


async def await_off_loop(fn):
    """Run a blocking call on a daemon thread so this actor's event loop stays free.

    Ray runs a SYNC actor method directly in the event loop -- ``sync_to_async`` wraps it
    as ``async def wrapper: return func(...)``, with no executor -- so a refit that blocks
    in NCCL starves every other call to the same actor. ``max_concurrency`` cannot help;
    it interleaves coroutines, and a coroutine blocked in C never yields.

    That is why the recovery could not run in job 6509685. The controller gave up on the
    stuck refit and called ``init_collective`` to rebuild, but that call queued behind the
    refit still occupying this loop. Rank 0 is the rendezvous master, so the store was
    never created, and the surviving generation worker timed out dialling it for 300s,
    twice, before the run ended.

    Daemon, because ``asyncio.to_thread``'s default executor is non-daemon and joined at
    interpreter exit: a thread still parked in NCCL would hang shutdown, trading a wedge in
    the refit for a wedge on the way out.

    No timeout here on purpose. Bounding the wait is the controller's job
    (``_sync_weights_within``); this only decides which thread blocks.
    """
    loop = asyncio.get_running_loop()
    settled: asyncio.Future = loop.create_future()

    def _settle(setter, value) -> None:
        if not settled.done():
            setter(value)

    def _run() -> None:
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the loop below
            loop.call_soon_threadsafe(_settle, settled.set_exception, exc)
        else:
            loop.call_soon_threadsafe(_settle, settled.set_result, result)

    threading.Thread(target=_run, name="refit-off-loop", daemon=True).start()
    return await settled


def sync_stream_within(stream, budget_s: Optional[float], what: str) -> None:
    """Wait for ``stream``'s enqueued work, giving up after ``budget_s``.

    WHY THIS EXISTS, and why the watchdog above is not enough. Aborting a communicator
    does not retire work already enqueued on a CUDA stream. When a generation rank stops
    receiving mid-refit the sends sit on the stream forever, and ``torch.cuda.synchronize``
    -- which waits on the whole device -- never returns. The watchdog cannot help: the
    abort fires, the kernels do not retire, the guarded block never exits, and
    :attr:`RefitAbortWatchdog.fired` is never read. No exception-translation reaches a
    hang.

    Job 6485245 measured exactly that on 4xGB200: both policy workers parked in
    ``synchronize`` 1801s after their own abort had logged, while the generation workers
    had already unwound and gone idle.

    So the wait is bounded here rather than trusted to end. The event is POLLED, not waited
    on, so nothing can be left holding the GIL, and the happy path still finishes with the
    same device-wide ``synchronize()`` -- behaviour is identical when nothing is wrong.

    This does NOT recover the fleet. In-flight kernels are orphaned and the caller's CUDA
    context should not be trusted afterwards, so the ``RefitAborted`` raised here is
    expected to end the run -- attributably, in seconds, rather than after a 30-minute
    stall. Recovering a frozen-but-alive rank on this transport stays out of scope.

    ``budget_s`` of None or <= 0 keeps the original unbounded synchronize, so a run with no
    refit deadline configured behaves exactly as before.
    """
    import torch

    if budget_s is None or budget_s <= 0:
        torch.cuda.synchronize()
        return

    event = torch.cuda.Event()
    event.record(stream)
    deadline = time.monotonic() + budget_s
    while not event.query():
        if time.monotonic() >= deadline:
            raise RefitAborted(
                f"{REFIT_CONTEXT_LOST_TOKEN} refit: {what} did not retire within "
                f"{budget_s}s. A peer most likely stopped receiving; aborting the "
                "communicator does not retire work already enqueued on the stream, so "
                "this gives up rather than blocking in cudaDeviceSynchronize forever. "
                "The orphaned kernels are on THIS trainer's device, so its CUDA context "
                "cannot be trusted and no communicator rebuild on it can succeed -- the "
                "run ends here rather than wedging in the recovery."
            )
        time.sleep(0.05)
    torch.cuda.synchronize()


RELEASE_GRACE_S = 30.0


def release_within(release, budget_s: float, what: str) -> None:
    """Run a teardown that may never return, without letting it wedge the caller.

    THE SIXTH UNBOUNDED WAIT, and the one that only the reshard transport reaches.
    ``StatelessProcessGroup.abort()`` calls ``abort_xferdtensor_python_subcommunicators``
    before the parent, and those split children exist ONLY on the Python reshard path --
    on the packed-broadcast path that call finds no cache entry and returns immediately.
    ``ncclCommAbort`` joins the communicator's proxy thread, and a proxy thread blocked
    reading from a SIGSTOPped peer never returns: the socket is open and idle, so nothing
    errors and nothing times out. SIGKILL closes it and the proxy errors out at once,
    which is why the killed variants never see this.

    Job 6518381 measured the consequence. On the rebuild, train rank 0 -- the rendezvous
    store's master -- entered ``init_collective``, printed its line, and never bound the
    port, because the only statement between the two is the old group's release. The
    surviving generation rank then spent 600s (300s, an 89s backoff, 300s again) failing
    to connect to a store that was never created, and the run died at 700s having
    condemned the right shard and planned the right membership.

    Bounded, on a DAEMON thread, and deliberately not joined. ``asyncio.to_thread`` and a
    bare ``ThreadPoolExecutor`` both use non-daemon threads that the interpreter joins at
    exit, which would move the wedge to shutdown rather than remove it. Nothing downstream
    reads a result: the release exists to stop resources accumulating across rebuilds, so
    a release that never finishes costs one stuck thread, while waiting on it costs the run.
    """
    import torch

    # THE CUDA DEVICE IS THREAD-LOCAL, and a fresh thread starts on device 0. Job 6510914
    # cost a run to exactly this when the refit first moved off the event loop, and
    # megatron_policy_worker asserts against the same "device drift" after setup. Captured
    # here, on the caller's thread, and re-set inside the release -- ncclCommAbort has to
    # run against the communicator's own device or it does not retire.
    #
    # Job 6524733 is what happens without it: recovery-reshard-refit regressed from passing
    # to a wedge, and its victim was SIGKILLed -- ActorDiedError, genuinely gone -- so the
    # abort had nothing to wait for and still did not return in 30s.
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    finished = threading.Event()

    def _release() -> None:
        try:
            if device is not None:
                torch.cuda.set_device(device)
            release()
        except Exception:  # noqa: BLE001 - a failed release must not mask the rebuild
            pass
        finally:
            finished.set()

    threading.Thread(target=_release, name="refit-release", daemon=True).start()
    if not finished.wait(budget_s):
        print(
            f"  refit: {what} did not release within {budget_s}s; continuing without it. "
            "A peer that is frozen rather than dead leaves ncclCommAbort joining a proxy "
            "thread that never returns, so this is abandoned rather than waited on.",
            flush=True,
        )


_ARMED_LOCK = threading.Lock()
_ARMED: "set[RefitAbortWatchdog]" = set()


def stand_down_armed_watchdogs() -> int:
    """Disarm every refit deadline armed in this process, and say how many.

    THE DEADLINE IS FOR A SILENT PEER, NOT A DEAD ONE, and firing it on a dead one is
    strictly harmful. When a generation rank's process is gone its sockets close, NCCL's
    own error path unblocks the survivors, and the run recovers off the pre-existing
    actor-death route -- job 6405953 passed the reshard kill variant that way with
    ``RefitAborted`` appearing zero times, before any deadline existed.

    Once the deadline was added it started winning that race. It aborts at its timeout,
    ``sync_stream_within`` gives up on kernels already enqueued on the trainers' streams,
    and the CUDA context cannot be trusted afterwards -- so the rebuild that would have
    succeeded now cannot. ``recovery-reshard-refit`` has failed continuously since job
    6512153, which is the run where the deadline first began firing on that path, and jobs
    6521181/6523731/6582457 each confirmed the abort never retires: not with the peer
    SIGKILLed, not with the release pinned to the caller's device.

    So the controller stands the deadline down the moment a probe reports an actor DEATH,
    which is conclusive in a way a timeout is not. The frozen case is untouched: a frozen
    rank is alive, no death is ever recorded, and the deadline still fires and still ends
    the run attributably.

    The controller can reach this at all only because the refit runs off the actor's event
    loop (see ``await_off_loop``); a worker blocked in the loop could not service the call.

    Idempotent, and safe to call when nothing is armed.
    """
    with _ARMED_LOCK:
        guards = list(_ARMED)
    for guard in guards:
        guard.stand_down()
    return len(guards)


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
        self._stood_down = False

    @property
    def armed(self) -> bool:
        return (
            bool(self._groups) and self._timeout_s is not None and self._timeout_s > 0
        )

    def stand_down(self) -> None:
        """Cancel this deadline without firing it; see stand_down_armed_watchdogs.

        Sets the same event the guarded block sets on a clean exit, so the watch thread
        returns without aborting anything and ``fired`` stays False. Racing a watch thread
        that is already past its wait is harmless: ``_done`` is only ever set, never
        cleared, and the abort it performs is idempotent.
        """
        self._stood_down = True
        self._done.set()

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
            # Registered before the thread starts, so a stand-down that arrives in the
            # same instant cannot slip between arming and being reachable.
            with _ARMED_LOCK:
                _ARMED.add(self)
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
        with _ARMED_LOCK:
            _ARMED.discard(self)
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
