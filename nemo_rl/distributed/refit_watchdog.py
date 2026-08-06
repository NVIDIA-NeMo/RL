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
from types import TracebackType
from typing import Optional, Protocol


class _Abortable(Protocol):
    def abort(self) -> None: ...


class RefitAborted(RuntimeError):
    """A refit was cut short because a peer stopped participating.

    Raised by the worker that armed the watchdog, not by NCCL -- the aborted call itself
    returns cleanly, so this is the only signal the caller gets.
    """


class RefitAbortWatchdog:
    """Abort ``group`` if the guarded block outlives ``timeout_s``.

    Use as a context manager around the collective::

        with RefitAbortWatchdog(self.model_update_group, timeout_s) as guard:
            ...collective...
        if guard.fired:
            raise RefitAborted(...)

    ``timeout_s`` of ``None`` or ``<= 0`` disarms it entirely: no thread is started and
    ``fired`` stays False, so the default configuration is bit-for-bit the old behaviour.
    """

    def __init__(self, group: Optional[_Abortable], timeout_s: Optional[float]) -> None:
        self._group = group
        self._timeout_s = timeout_s
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fired = False

    @property
    def armed(self) -> bool:
        return (
            self._group is not None
            and self._timeout_s is not None
            and self._timeout_s > 0
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
        try:
            assert self._group is not None
            self._group.abort()
        except Exception:  # noqa: BLE001
            # A failed abort leaves the caller blocked, which is the situation we were
            # already in; swallowing keeps the watchdog thread from dying silently mid-way
            # and is strictly no worse than not having tried.
            pass

    def __enter__(self) -> "RefitAbortWatchdog":
        if self.armed:
            self._thread = threading.Thread(
                target=self._watch, name="refit-abort-watchdog", daemon=True
            )
            self._thread.start()
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
