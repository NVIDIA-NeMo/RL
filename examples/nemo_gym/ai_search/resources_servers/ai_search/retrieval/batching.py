# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous micro-batching for concurrent rollout search calls."""

import asyncio
import time
from dataclasses import dataclass, replace

from resources_servers.ai_search.retrieval.types import SearchProvider, SearchResult


@dataclass
class _PendingSearch:
    query: str
    top_k: int
    enqueued_at: float
    future: asyncio.Future[SearchResult]


class AsyncSearchBatcher:
    """Combine nearby requests into one encoder and cuVS batch."""

    def __init__(
        self,
        provider: SearchProvider,
        max_batch_size: int,
        wait_ms: float,
    ) -> None:
        self._provider = provider
        self._max_batch_size = max_batch_size
        self._wait_seconds = wait_ms / 1000.0
        self._queue: asyncio.Queue[_PendingSearch | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._worker_task is None:
                self._worker_task = asyncio.create_task(
                    self._run(), name="ai-search-microbatcher"
                )

    async def close(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    async def search(self, query: str, top_k: int) -> SearchResult:
        await self.start()
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _PendingSearch(
                query=query,
                top_k=top_k,
                enqueued_at=time.perf_counter(),
                future=future,
            )
        )
        return await future

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return

            pending = [first]
            if self._wait_seconds > 0:
                await asyncio.sleep(self._wait_seconds)
            while len(pending) < self._max_batch_size:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self._queue.task_done()
                    await self._finish_batch(pending)
                    return
                pending.append(item)
            await self._finish_batch(pending)

    async def _finish_batch(self, pending: list[_PendingSearch]) -> None:
        execution_started = time.perf_counter()
        max_top_k = max(item.top_k for item in pending)
        try:
            results = await asyncio.to_thread(
                self._provider.search_batch,
                [item.query for item in pending],
                max_top_k,
            )
            if len(results) != len(pending):  # pragma: no cover
                raise RuntimeError(
                    f"Search provider returned {len(results)} results for {len(pending)} queries"
                )
            for item, result in zip(pending, results):
                queue_ms = (execution_started - item.enqueued_at) * 1000.0
                sliced = replace(
                    result,
                    hits=result.hits[: item.top_k],
                    timings=replace(result.timings, queue_ms=queue_ms),
                )
                if not item.future.cancelled():
                    item.future.set_result(sliced)
        except Exception as error:
            for item in pending:
                if not item.future.cancelled():
                    item.future.set_exception(error)
        finally:
            for _ in pending:
                self._queue.task_done()
