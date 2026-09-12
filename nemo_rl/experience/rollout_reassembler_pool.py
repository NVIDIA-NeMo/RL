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

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import ray


@dataclass
class ReassemblerLease:
    """One exclusive actor checkout, including its publication outcome."""

    actor: Any
    queue_wait_ms: float
    queue_depth: int
    active_actor_count: int
    rpc_submitted: bool = False
    outcome_known: bool = False


class RolloutReassemblerPool:
    """Own actor availability while the controller owns publication and commits.

    An actor with an unknown RPC outcome is quarantined until shutdown. It must
    never be replaced or retried: its canonical rows may already be published.
    """

    def __init__(self, actors: list[Any]) -> None:
        self._actors = list(actors)
        self._available: asyncio.Queue[Any] = asyncio.Queue()
        for actor in actors:
            self._available.put_nowait(actor)
        self.active = 0
        self.waiters = 0
        self.unknown_outcomes = 0
        self._closed = False

    def __bool__(self) -> bool:
        return bool(self._actors)

    @property
    def available_count(self) -> int:
        return 0 if self._closed else self._available.qsize()

    async def acquire(self) -> ReassemblerLease:
        """Wait for an actor; cancellation never consumes a checkout."""
        if self._closed:
            raise RuntimeError("reassembler pool is closed")
        self.waiters += 1
        queue_depth = max(0, self.waiters - self.available_count)
        start = time.perf_counter()
        try:
            actor = await self._available.get()
        finally:
            self.waiters -= 1
        if actor is None:
            raise RuntimeError("reassembler pool is closed")
        self.active += 1
        return ReassemblerLease(
            actor=actor,
            queue_wait_ms=(time.perf_counter() - start) * 1000.0,
            queue_depth=queue_depth,
            active_actor_count=self.active,
        )

    def release(self, lease: ReassemblerLease) -> None:
        """Return a safe actor or quarantine an unresolved publication."""
        self.active -= 1
        if lease.rpc_submitted and not lease.outcome_known:
            self.unknown_outcomes += 1
        elif not self._closed:
            self._available.put_nowait(lease.actor)

    def shutdown(self) -> None:
        """Terminate all actors after controller tasks have drained."""
        if self._closed:
            return
        self._closed = True
        while not self._available.empty():
            self._available.get_nowait()
        for _ in range(self.waiters):
            self._available.put_nowait(None)
        for actor in self._actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as error:
                # Teardown must not mask the controller's original failure.
                print(f"reassembler actor termination failed: {error}", flush=True)
