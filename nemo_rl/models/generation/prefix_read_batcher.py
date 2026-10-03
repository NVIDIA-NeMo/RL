# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded, event-loop-local coalescing of blocking generation-prefix reads."""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any


LOGGER = logging.getLogger(__name__)


class PrefixReadBatcher:
    """Share GETs across requests without sharing request cancellation.

    The fetch callback must validate row identity and return requested order
    (TQTokenSource.fetch does both). Token estimates are upper bounds per key.
    A single oversized row is allowed alone: splitting a stored row is not
    supported. Shutdown fails waiters but cannot interrupt an executing GET.
    """

    def __init__(
        self,
        fetch: Callable[[list[str]], list[Any]],
        *,
        max_rows: int = 256,
        max_tokens: int = 4_194_304,
        max_pending: int = 1024,
        wait_seconds: float = 0.002,
    ) -> None:
        if min(max_rows, max_tokens, max_pending) < 1 or wait_seconds < 0:
            raise ValueError("prefix read limits must be positive and wait nonnegative")
        self._fetch = fetch
        self._max_rows = max_rows
        self._max_tokens = max_tokens
        self._wait_seconds = wait_seconds
        self._slots = asyncio.Semaphore(max_pending)
        self._queue: deque[tuple[str, int, asyncio.Future[Any]]] = deque()
        self._wake = asyncio.Event()
        self._pending: set[asyncio.Future[Any]] = set()
        self._runner: asyncio.Task[None] | None = None
        self._closed = False

    async def fetch(self, keys: list[str], *, tokens_per_key: int) -> list[Any]:
        """Read a chain in order; cancellation affects only this caller."""
        if tokens_per_key < 1 or len(keys) != len(set(keys)):
            raise ValueError("positive token estimate and unique keys required")
        if (
            self._runner is not None
            and self._runner.get_loop() is not asyncio.get_running_loop()
        ):
            raise RuntimeError("prefix reads must use one serving event loop")
        futures: list[asyncio.Future[Any]] = []
        try:
            for key in keys:
                await self._slots.acquire()
                if self._closed:
                    self._slots.release()
                    raise RuntimeError("prefix reader closed")
                future = asyncio.get_running_loop().create_future()
                self._pending.add(future)
                futures.append(future)
                self._queue.append((key, tokens_per_key, future))
                self._wake.set()
                if self._runner is None:
                    self._runner = asyncio.create_task(self._run())
            return list(await asyncio.gather(*futures))
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    # Retrieve failures even if enqueue was interrupted.
                    future.exception()

    def _release(self, future: asyncio.Future[Any]) -> None:
        if future in self._pending:
            self._pending.remove(future)
            self._slots.release()

    async def _read(self, keys: list[str]) -> dict[str, Any | Exception]:
        try:
            rows = await asyncio.to_thread(self._fetch, keys)
            if len(rows) != len(keys):
                raise KeyError("incomplete prefix read")
            return dict(zip(keys, rows))
        except KeyError as error:
            if len(keys) == 1:
                return {keys[0]: error}
            middle = len(keys) // 2
            left = await self._read(keys[:middle])
            left.update(await self._read(keys[middle:]))
            return left
        except Exception as error:
            return dict.fromkeys(keys, error)

    async def _run(self) -> None:
        try:
            while True:
                await self._wake.wait()
                await asyncio.sleep(self._wait_seconds)
                batch: dict[str, list[asyncio.Future[Any]]] = {}
                tokens = 0
                while self._queue:
                    key, estimate, future = self._queue[0]
                    if future.done():
                        self._queue.popleft()
                        self._release(future)
                        continue
                    if key not in batch:
                        if batch and (
                            len(batch) >= self._max_rows
                            or tokens + estimate > self._max_tokens
                        ):
                            break
                        tokens += estimate
                    self._queue.popleft()
                    batch.setdefault(key, []).append(future)
                if not self._queue:
                    self._wake.clear()
                if not batch:
                    continue
                started = time.monotonic()
                results = await self._read(list(batch))
                LOGGER.info(
                    "generation_prefix_read_batch keys=%d waiters=%d "
                    "estimated_tokens=%d read_seconds=%.6f failed_keys=%d",
                    len(batch),
                    sum(len(waiters) for waiters in batch.values()),
                    tokens,
                    time.monotonic() - started,
                    sum(isinstance(value, Exception) for value in results.values()),
                )
                for key, waiters in batch.items():
                    result = results[key]
                    for waiter in waiters:
                        if not waiter.done():
                            if isinstance(result, Exception):
                                waiter.set_exception(result)
                            else:
                                waiter.set_result(result)
                        self._release(waiter)
        finally:
            self._closed = True
            for future in tuple(self._pending):
                if not future.done():
                    future.set_exception(RuntimeError("prefix reader closed"))
                self._release(future)
            self._queue.clear()

    async def aclose(self) -> None:
        """Close on the serving loop, including when called by the actor loop."""
        if self._runner is not None:
            owner = self._runner.get_loop()
            if owner is not asyncio.get_running_loop():
                if owner.is_closed():
                    self._closed = True
                    return
                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(self._close_on_owner(), owner)
                )
                return
        await self._close_on_owner()

    async def _close_on_owner(self) -> None:
        """Stop accepting reads and wake queued/in-flight callers immediately."""
        self._closed = True
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        for future in tuple(self._pending):
            if not future.done():
                future.set_exception(RuntimeError("prefix reader closed"))
            self._release(future)
        self._queue.clear()
