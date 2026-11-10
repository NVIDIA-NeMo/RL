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
"""Two-phase checkpoint ownership and cancellation-safe persistence.

Capture runs under the caller's global mutation barrier. Once it returns, the
handle owns a stable view independent of live clears/replacements. Persistence
may run after that barrier opens. Publication remains the caller's responsibility.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Callable, Self, TypeVar

from nemo_rl.data_plane.async_utils import call_data_plane

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class CheckpointCapture(ABC):
    """One retained snapshot; callers must release it even after failed saves."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    @abstractmethod
    def persist(self) -> None:
        """Write the captured view, without reading mutable live state."""

    @abstractmethod
    def release(self) -> None:
        """Release after writers stop; retry release if its wait times out."""


async def _settle(task: asyncio.Task[T]) -> T:
    """Do not leave an uncancellable thread/RPC running past cleanup."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def checkpoint_io(operation: Callable[[], T]) -> T:
    """Run blocking snapshot I/O; cancellation waits for outstanding writes.

    Cancellation settles the local callable. A backend timeout can leave remote
    RPCs running; release must succeed before removing its temporary directory.
    """
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await _settle(task)
        except Exception:
            LOGGER.exception("Checkpoint I/O failed while settling cancellation")
        raise


async def capture_checkpoint(client: Any, **kwargs: Any) -> CheckpointCapture | None:
    """Capture via an adapter, releasing late results if capture is cancelled."""
    task = asyncio.create_task(
        call_data_plane(client, "capture_checkpoint", offload_sync=True, **kwargs)
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            captured = await _settle(task)
            if captured is not None:
                await checkpoint_io(captured.release)
        except Exception:
            LOGGER.exception("Checkpoint capture cleanup failed during cancellation")
        raise
