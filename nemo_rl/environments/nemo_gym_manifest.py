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
"""Bounded, off-loop manifest reads for colocated Gym actors."""

from __future__ import annotations

import asyncio
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from typing import Callable, Protocol


class ManifestReader(Protocol):
    def read_manifest(
        self, rollout_id: str, *, deadline: float, cancel_event: threading.Event
    ) -> dict: ...


class ManifestReceiptReader:
    """Read and assemble receipts with bounded executor submissions.

    The semaphore covers running and queued jobs. Its permit belongs to the
    executor future, so cancelling or timing out a caller cannot free capacity
    while its thread is still working. Other callers wait asynchronously.
    """

    def __init__(
        self,
        reader: ManifestReader,
        assemble_receipt: Callable[..., dict],
        *,
        workers: int,
        queue_size: int,
        timeout_s: float,
    ) -> None:
        if (
            workers < 1
            or queue_size < 1
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(
                "manifest worker count, queue size and timeout must be positive and finite"
            )
        self._reader = reader
        self._assemble = assemble_receipt
        self._timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="manifest-reader")
        self._slots = asyncio.Semaphore(workers + queue_size)
        self._jobs: dict[asyncio.Future[dict], threading.Event] = {}
        self._requests: set[asyncio.Task] = set()
        self._closed = False

    async def read_receipt(
        self,
        rollout_id: str,
        *,
        terminal_response_id: str | None,
        scored_response: dict | None,
        reward: float,
    ) -> dict:
        """Use one deadline for admission, file read and receipt assembly."""
        if self._closed:
            raise RuntimeError("manifest reader is closed")
        task = asyncio.current_task()
        assert task is not None
        self._requests.add(task)
        deadline = monotonic() + self._timeout_s
        cancel = threading.Event()

        def check() -> None:
            if cancel.is_set() or monotonic() >= deadline:
                raise TimeoutError(
                    f"manifest read for {rollout_id} cancelled or exceeded its deadline"
                )

        def read() -> dict:
            check()
            manifest = self._reader.read_manifest(
                rollout_id, deadline=deadline, cancel_event=cancel
            )
            check()
            receipt = self._assemble(
                rollout_id,
                manifest,
                terminal_response_id=terminal_response_id,
                scored_response=scored_response,
                reward=reward,
            )
            check()
            return receipt

        try:
            async with asyncio.timeout(self._timeout_s):
                await self._slots.acquire()
                try:
                    future = asyncio.wrap_future(self._pool.submit(read))
                except BaseException:
                    self._slots.release()
                    raise
                self._jobs[future] = cancel
                future.add_done_callback(self._finished)
                receipt = await asyncio.shield(future)
                check()
        except TimeoutError as error:
            raise TimeoutError(
                f"manifest read for {rollout_id} exceeded {self._timeout_s}s"
            ) from error
        except Exception as error:
            # Isolate a failed read/assembly to this rollout; later reads still run.
            raise RuntimeError(
                f"local manifest read for {rollout_id} failed: {error}"
            ) from error
        finally:
            cancel.set()
            self._requests.discard(task)
        return receipt

    def _finished(self, future: asyncio.Future[dict]) -> None:
        self._jobs.pop(future)
        self._slots.release()
        if not future.cancelled():
            future.exception()  # Consume late failures after the caller timed out.

    async def close(self) -> None:
        """Cancel callers and queued work without blocking on running OS reads."""
        self._closed = True
        for cancel in self._jobs.values():
            cancel.set()
        requests = list(self._requests)
        for task in requests:
            task.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        await asyncio.gather(*requests, return_exceptions=True)
