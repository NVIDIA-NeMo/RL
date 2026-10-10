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
"""Bounded obsolete-prefix deletion coordinated with generation snapshot fences."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator

LOGGER = logging.getLogger(__name__)


class GenerationPrefixCleanupConfig(BaseModel, extra="forbid"):
    """Optional per-generation-owner cleanup limits; synchronous cleanup is default."""

    enabled: bool = False
    batch_size: Annotated[int, Field(gt=0)] = 32
    max_pending: Annotated[int, Field(gt=0)] = 128
    wait_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.01

    @model_validator(mode="after")
    def _validate_capacity(self) -> Self:
        if self.max_pending < self.batch_size:
            raise ValueError("max_pending must be at least batch_size")
        return self


@dataclass(frozen=True)
class PrefixCleanupStats:
    """Cumulative attempts and current per-worker queue occupancy."""

    requests: int
    batches: int
    keys: int
    pending_requests: int
    queued_requests: int
    peak_pending_requests: int
    active_batch: bool
    paused: bool
    failed_keys: int
    oldest_queue_age_seconds: float


class CleanupReservation:
    """Capacity acquired outside the terminal gate, then consumed or released."""

    def __init__(self, owner: BackgroundPrefixCleanup) -> None:
        self._owner = owner
        self._used = False

    def submit(self, keys: Sequence[str]) -> None:
        """Transfer obsolete keys after the replacement row is acknowledged."""
        copied_keys = list(dict.fromkeys(keys))
        if not copied_keys:
            self.close()
            return
        with self._owner._condition:
            if self._used:
                raise RuntimeError("cleanup reservation already consumed")
            self._owner._check_open()
            self._used = True
            self._owner._queue.append((copied_keys, time.monotonic()))
            self._owner._condition.notify_all()

    def close(self) -> None:
        """Release an unused slot, including on failed terminal staging."""
        with self._owner._condition:
            if not self._used:
                self._used = True
                self._owner._pending -= 1
                self._owner._condition.notify_all()


class BackgroundPrefixCleanup:
    """Batch already-obsolete keys without delaying successful terminal replies.

    Reserve outside the terminal gate; enqueue only after canonical staging.
    Pause before draining that gate and wait for the active CLEAR before saving
    TQ. Queued deletes remain deferred until resume. Released fence epochs cannot
    be reclosed by late control work. Transport errors are latched and never
    automatically retried because a CLEAR may have partially succeeded.

    The queue is process-local. Losing it leaves extra obsolete rows, rather than
    losing a required prefix. Restoring a checkpoint does not replay this queue.
    """

    def __init__(
        self,
        clear: Callable[[list[str]], None],
        *,
        config: GenerationPrefixCleanupConfig,
    ) -> None:
        self._clear = clear
        # Own limits even if the driver's model is subsequently modified.
        self._batch_size = config.batch_size
        self._max_pending = config.max_pending
        self._wait_seconds = config.wait_seconds
        self._condition = threading.Condition()
        self._queue: deque[tuple[list[str], float]] = deque()
        self._pending = 0
        self._peak_pending = 0
        self._active = False
        self._epoch = 0
        self._released_epoch = 0
        self._closed = False
        self._error: Exception | None = None
        self._failed_keys: list[str] = []
        self._requests = 0
        self._batches = 0
        self._keys = 0
        # Actor shutdown must not wait forever on a stalled transport.
        self._thread = threading.Thread(
            target=self._drain, name="nrl-background-prefix-cleanup", daemon=True
        )
        self._thread.start()

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "background prefix cleanup failed; no automatic retry"
            ) from self._error

    def _check_open(self) -> None:
        self._check_error()
        if self._closed:
            raise RuntimeError("prefix cleanup is closed")

    def reserve(self) -> CleanupReservation:
        """Bound reservations plus queued/active requests, outside the terminal gate."""
        with self._condition:
            self._check_open()
            while self._pending >= self._max_pending:
                self._condition.wait()
                self._check_open()
            self._pending += 1
            self._peak_pending = max(self._peak_pending, self._pending)
            return CleanupReservation(self)

    def pause(self, epoch: int) -> None:
        """Stop admitting new CLEAR batches for this snapshot epoch."""
        with self._condition:
            self._epoch = max(self._epoch, epoch)
            self._condition.notify_all()

    def wait_paused(self, epoch: int) -> None:
        """Wait for an active CLEAR only; queued cleanup is deferred, not drained."""
        with self._condition:
            while self._active and self._released_epoch < epoch:
                self._condition.wait()
            self._check_error()

    def resume(self) -> None:
        """Release the current epoch, including when its checkpoint is aborted."""
        with self._condition:
            self._released_epoch = self._epoch
            self._condition.notify_all()

    def _drain(self) -> None:
        while True:
            with self._condition:
                while not self._queue or self._epoch > self._released_epoch:
                    if self._closed and not self._queue:
                        return
                    self._condition.wait()
                deadline = self._queue[0][1] + self._wait_seconds
                while (
                    len(self._queue) < self._batch_size
                    and not self._closed
                    and self._epoch <= self._released_epoch
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._epoch > self._released_epoch:
                    continue
                batch = [
                    self._queue.popleft()
                    for _ in range(min(len(self._queue), self._batch_size))
                ]
                keys = list(dict.fromkeys(key for rows, _ in batch for key in rows))
                self._active = True
                self._batches += 1
                self._requests += len(batch)
                self._keys += len(keys)
            try:
                self._clear(keys)
            except Exception as error:  # noqa: BLE001 - surface arbitrary transport failures
                with self._condition:
                    self._error = error
                    self._failed_keys = keys
                    self._active = False
                    self._condition.notify_all()
                LOGGER.exception(
                    "background prefix cleanup failed for %d keys", len(keys)
                )
                return
            with self._condition:
                self._active = False
                self._pending -= len(batch)
                self._condition.notify_all()

    def stats(self) -> PrefixCleanupStats:
        """Return attempted batch counts and queue pressure without per-key logging."""
        with self._condition:
            return PrefixCleanupStats(
                requests=self._requests,
                batches=self._batches,
                keys=self._keys,
                pending_requests=self._pending,
                queued_requests=len(self._queue),
                peak_pending_requests=self._peak_pending,
                active_batch=self._active,
                paused=self._epoch > self._released_epoch,
                failed_keys=len(self._failed_keys),
                oldest_queue_age_seconds=(
                    time.monotonic() - self._queue[0][1] if self._queue else 0.0
                ),
            )

    def close(self, *, wait: bool = True) -> None:
        """Reject reservations and drain queued deletes after serving has stopped.

        Resume any checkpoint first. ``wait=False`` is for actor teardown: it
        leaves acknowledged terminal rows intact if the process exits before
        obsolete deletes complete. Unused reservations must still be released.
        """
        with self._condition:
            if self._epoch > self._released_epoch:
                raise RuntimeError("resume the cleanup fence before closing cleanup")
            self._closed = True
            self._condition.notify_all()
        if wait:
            self._thread.join()
            self._check_error()
