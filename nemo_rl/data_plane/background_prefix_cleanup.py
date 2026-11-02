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
"""Batched obsolete-prefix deletion fenced by the token-capture snapshot gate."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Protocol, Self

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


class SnapshotGate(Protocol):
    """The worker's token-capture snapshot gate (``_TokenCaptureSnapshotGate``)."""

    def enter(self) -> None: ...

    def exit(self) -> None: ...


class BackgroundPrefixCleanup:
    """Delete already-obsolete prefix keys off the completion path, in batches.

    Callers submit a call's keys only after its canonical row is staged and
    after leaving the snapshot gate, so a full queue blocks that completion
    without holding a permit the snapshot fence waits on. The cleanup thread
    holds a gate permit for each CLEAR: a fence waits for a running CLEAR, and
    no CLEAR starts while the gate is closed, so queued deletes wait for the
    fence to end. A failed CLEAR is logged and dropped. It is never retried,
    because it may have partly succeeded, and a lost delete only leaves extra
    rows.

    The queue is process-local and is not checkpointed, so a crash or actor
    shutdown can leave extra obsolete rows, never a missing required prefix.
    """

    def __init__(
        self,
        clear: Callable[[list[str]], None],
        gate: SnapshotGate,
        *,
        config: GenerationPrefixCleanupConfig,
    ) -> None:
        self._clear = clear
        self._gate = gate
        # Own limits even if the driver's model is subsequently modified.
        self._batch_size = config.batch_size
        self._wait_seconds = config.wait_seconds
        self._queue: queue.Queue[list[str]] = queue.Queue(maxsize=config.max_pending)
        # Daemon and never joined, so shutdown never waits on this thread itself.
        threading.Thread(
            target=self._drain, name="nrl-background-prefix-cleanup", daemon=True
        ).start()

    def submit(self, keys: Sequence[str]) -> None:
        """Queue one call's obsolete keys; blocks while ``max_pending`` calls wait."""
        owned = list(dict.fromkeys(keys))
        if owned:
            self._queue.put(owned)

    def _drain(self) -> None:
        while True:
            batch = [self._queue.get()]
            deadline = time.monotonic() + self._wait_seconds
            while len(batch) < self._batch_size:
                try:
                    batch.append(
                        self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
                    )
                except queue.Empty:
                    break
            keys = list(dict.fromkeys(key for call_keys in batch for key in call_keys))
            # A checkpoint fence waits for this CLEAR and blocks the next one.
            self._gate.enter()
            try:
                self._clear(keys)
            except Exception:  # noqa: BLE001 - a lost delete only leaves extra rows
                # Never retry: a CLEAR may have partly succeeded. Keep serving.
                LOGGER.exception(
                    "background prefix cleanup failed for %d keys", len(keys)
                )
            finally:
                self._gate.exit()
