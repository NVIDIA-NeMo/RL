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

"""Admission for bounded, independent generation-cut inventories.

Bookkeeping belongs to the HTTP event loop, not the executor. Waiting for an
overlapping inventory must never occupy a thread needed to finish its owner.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nemo_rl.utils.prefix_cut_diagnostics import prefix_cut_count


@dataclass(eq=False)
class _Inventory:
    key: tuple[str, str]
    call_ids: frozenset[str]
    task: asyncio.Task[Any] | None
    active: bool
    waiters: int


class GenerationCutControl:
    """Serialize overlapping calls and checkpoint epochs before executor entry.

    Independent inventories of the same checkpoint may run concurrently.
    A queued different checkpoint forms a barrier: later arrivals cannot
    overtake it. Caller cancellation never releases a running operation's
    reservations; its thread must actually finish first.

    ``run`` is confined to one event loop. ``close`` may be called from the
    actor's shutdown thread and refuses new work without cancelling active cuts.
    """

    def __init__(self, *, max_workers: int) -> None:
        if type(max_workers) is not int or max_workers not in (1, 2):
            raise ValueError("generation-cut max_workers must be 1 or 2")
        self._max_workers = max_workers
        self._loop: asyncio.AbstractEventLoop | None = None
        self._changed = asyncio.Event()
        self._jobs: dict[tuple[str, str], _Inventory] = {}
        self._pending: list[_Inventory] = []
        self._active: list[_Inventory] = []
        self._checkpoint_id: str | None = None
        self._retired_checkpoints: set[str] = set()
        self._closed = False

    async def run(
        self,
        *,
        checkpoint_id: str,
        inventory_digest: str,
        call_ids: frozenset[str],
        deadline: float | None,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Join a duplicate or admit an independent inventory before its deadline."""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("generation-cut admission belongs to another event loop")
        if self._closed:
            raise RuntimeError("generation-cut admission is closed")
        key = (checkpoint_id, inventory_digest)
        job = self._jobs.get(key)
        if job is not None and not job.active and job.task is not None:
            if job.task.cancelling():
                # The last caller cancelled before admission. A retry may
                # arrive before that task's done callback removes its entry.
                del self._jobs[key]
                job = None
        if job is None:
            job = _Inventory(key, call_ids, task=None, active=False, waiters=0)
            self._jobs[key] = job
            self._pending.append(job)
            job.task = loop.create_task(self._execute(job, operation, deadline))
            job.task.add_done_callback(functools.partial(self._finished, job))
        else:
            if job.call_ids != call_ids:
                raise ValueError(
                    "duplicate inventory has different model-call identities"
                )
            prefix_cut_count("control_duplicate_join", 1)
        assert job.task is not None
        job.waiters += 1
        try:
            # HTTP cancellation must not cancel the task awaiting the executor:
            # cancelling that await would abandon a still-running thread.
            return await asyncio.shield(job.task)
        finally:
            job.waiters -= 1
            if job.waiters == 0 and not job.active and not job.task.done():
                prefix_cut_count("control_cancelled_before_start", 1)
                job.task.cancel()

    def _can_start(self, job: _Inventory) -> bool:
        if len(self._active) >= self._max_workers:
            return False
        if any(active.key[0] != job.key[0] for active in self._active):
            return False
        if any(active.call_ids & job.call_ids for active in self._active):
            return False
        for earlier in self._pending:
            if earlier is job:
                return True
            if earlier.key[0] != job.key[0] or earlier.call_ids & job.call_ids:
                return False
        return False

    async def _execute(
        self,
        job: _Inventory,
        operation: Callable[[], Awaitable[Any]],
        deadline: float | None,
    ) -> Any:
        try:
            while True:
                if self._closed:
                    raise RuntimeError("generation-cut admission is closed")
                if job.key[0] in self._retired_checkpoints:
                    prefix_cut_count("control_stale_checkpoint_rejected", 1)
                    raise RuntimeError("generation-cut checkpoint has been superseded")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    prefix_cut_count("control_admission_deadline_expired", 1)
                    raise TimeoutError(
                        "generation-cut deadline expired before admission"
                    )
                if self._can_start(job):
                    if self._checkpoint_id != job.key[0]:
                        if self._checkpoint_id is not None:
                            self._retired_checkpoints.add(self._checkpoint_id)
                        self._checkpoint_id = job.key[0]
                    self._pending.remove(job)
                    self._active.append(job)
                    job.active = True
                    break
                # No await separates the predicate and clear: another operation
                # cannot release its reservation between them on this loop.
                self._changed.clear()
                try:
                    async with asyncio.timeout(remaining):
                        await self._changed.wait()
                except TimeoutError:
                    prefix_cut_count("control_admission_deadline_expired", 1)
                    raise TimeoutError(
                        "generation-cut deadline expired before admission"
                    ) from None
            return await operation()
        finally:
            if job in self._pending:
                self._pending.remove(job)
            if job in self._active:
                self._active.remove(job)
            self._changed.set()

    def _finished(self, job: _Inventory, task: asyncio.Task[Any]) -> None:
        # Also handles cancellation before _execute had its first turn.
        if self._jobs.get(job.key) is job:
            del self._jobs[job.key]
        if job in self._pending:
            self._pending.remove(job)
        self._changed.set()
        if not task.cancelled():
            # A disconnected caller might never retrieve a failed operation.
            task.exception()

    def close(self) -> None:
        """Refuse new admissions and cancel only operations not yet started."""
        self._closed = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(functools.partial(self._cancel_pending))

    def _cancel_pending(self) -> None:
        for job in tuple(self._pending):
            assert job.task is not None
            job.task.cancel()
        self._changed.set()
