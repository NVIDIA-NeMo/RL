# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Bounded async-iterator handoff between the engine and HTTP event loops."""

import asyncio
import threading
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, PositiveInt


class HTTPBridgeConfig(BaseModel, extra="allow"):
    """Optional HTTP output transport; defaults preserve the per-output bridge."""

    mode: Literal["per_output", "queued"] = "per_output"
    # Maximum queued outputs per request, plus one held by the producer.
    max_pending_outputs: PositiveInt = 16


def _wake(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


@dataclass
class _Mailbox[T]:
    capacity: int
    producer_loop: asyncio.AbstractEventLoop
    consumer_loop: asyncio.AbstractEventLoop
    stopped: asyncio.Future[None]
    # Each waiter belongs to its respective loop; only that loop resolves it.
    items: deque[T] = field(init=False, default_factory=deque)
    lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    reader: asyncio.Future[None] | None = field(init=False, default=None)
    writer: asyncio.Future[None] | None = field(init=False, default=None)
    finished: bool = field(init=False, default=False)
    error: BaseException | None = field(init=False, default=None)

    async def put(self, item: T) -> None:
        while True:
            with self.lock:
                if len(self.items) < self.capacity:
                    self.items.append(item)
                    reader, self.reader = self.reader, None
                    break
                self.writer = waiter = self.producer_loop.create_future()
            await waiter
        if reader is not None:
            self.consumer_loop.call_soon_threadsafe(_wake, reader)

    async def get(self) -> T:
        while True:
            with self.lock:
                if self.items:
                    item = self.items.popleft()
                    writer, self.writer = self.writer, None
                    break
                if self.finished:
                    if self.error is not None:
                        raise self.error
                    raise StopAsyncIteration
                self.reader = waiter = self.consumer_loop.create_future()
            await waiter
        if writer is not None:
            self.producer_loop.call_soon_threadsafe(_wake, writer)
        return item

    def finish(self, task: asyncio.Task[None]) -> None:
        # Retrieve the exception even if the consumer disconnected. Task.done()
        # runs after the producer's finally blocks, including generator close.
        try:
            error = task.exception()
        except asyncio.CancelledError as exc:
            error = exc
        with self.lock:
            self.finished, self.error = True, error
            reader, self.reader = self.reader, None

        def notify() -> None:
            if reader is not None:
                _wake(reader)
            _wake(self.stopped)

        self.consumer_loop.call_soon_threadsafe(notify)


async def queued_iterator[T](
    source: AsyncGenerator[T, None],
    producer_loop: asyncio.AbstractEventLoop,
    *,
    capacity: int,
) -> AsyncGenerator[T, None]:
    """Iterate on the owner loop with bounded read-ahead and shared references.

    One producer task owns the source for its whole lifetime. A lock protects
    the deque and waiter registration together, preventing lost wakeups. Only
    an actually waiting peer is notified; available outputs drain immediately
    without per-item tasks/futures, serialization, or a batch timer. Source
    outputs must remain stable after the next iteration (e.g. vLLM deltas).

    Closing/cancelling the consumer cancels the producer and waits for source
    cleanup. The queue holds at most ``capacity`` outputs, with at most one
    additional output held by the producer waiting for space. Upstream source
    buffers and the consumer's current output are outside this bound.
    """
    if capacity <= 0:
        raise ValueError("bridge capacity must be positive")
    consumer_loop = asyncio.get_running_loop()
    mailbox = _Mailbox[T](
        capacity, producer_loop, consumer_loop, consumer_loop.create_future()
    )
    producer: asyncio.Task[None] | None = None

    async def produce() -> None:
        try:
            async for item in source:
                await mailbox.put(item)
        finally:
            await source.aclose()

    def start() -> None:
        nonlocal producer
        producer = producer_loop.create_task(produce())
        producer.add_done_callback(mailbox.finish)

    def cancel() -> None:
        if producer is not None and not producer.done() and not producer.cancelling():
            producer.cancel()

    producer_loop.call_soon_threadsafe(start)
    delivered = 0
    try:
        while True:
            try:
                item = await mailbox.get()
            except StopAsyncIteration:
                return
            yield item
            delivered += 1
            if delivered == capacity:
                # ASGI writes need not suspend when the socket buffer has room.
                # Give other HTTP requests a turn even under a continuous burst.
                delivered = 0
                await asyncio.sleep(0)
    finally:
        if not mailbox.stopped.done():
            producer_loop.call_soon_threadsafe(cancel)
        # Do not cancel this acknowledgement: concurrent Future cancellation
        # would report completion before the owner-loop cleanup actually ran.
        await asyncio.shield(mailbox.stopped)
