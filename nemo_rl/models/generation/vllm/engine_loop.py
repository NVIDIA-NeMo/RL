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

"""A dedicated event loop that owns one vLLM ``AsyncLLM`` engine.

``AsyncLLM`` binds its output handler and request state to the loop that
constructs it. With ``vllm_cfg.engine_owner_loop`` enabled, the async worker
creates the engine, runs the OpenAI HTTP server, and executes every engine
operation on this loop, so no engine call ever runs on the Ray actor loop or
a second server thread.
"""

import asyncio
import functools
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

ResultT = TypeVar("ResultT")
P = ParamSpec("P")


class EngineOwnerLoop:
    """Run one event loop forever on a dedicated daemon thread."""

    def __init__(self, start_timeout_s: float = 5.0) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="vllm-engine-owner-loop", daemon=True
        )
        self.thread.start()
        if not self._ready.wait(timeout=start_timeout_s):
            raise RuntimeError("vLLM engine owner loop failed to start")

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("vLLM engine owner loop is not initialized")
        return self._loop

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._thread_id = threading.get_ident()
        loop.call_soon(self._ready.set)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def is_owner_thread(self) -> bool:
        return threading.get_ident() == self._thread_id

    def assert_owner(self) -> None:
        """Raise if the caller is not running on this loop."""
        if not self.is_owner_thread() or asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("vLLM engine operation executed outside its owner loop")

    def submit(self, coro: Coroutine[Any, Any, ResultT]):
        if not self.loop.is_running():
            coro.close()
            raise RuntimeError("vLLM engine owner loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro: Coroutine[Any, Any, ResultT]) -> ResultT:
        """Block a non-owner thread until ``coro`` finishes on the owner loop."""
        if self.is_owner_thread():
            coro.close()
            raise RuntimeError("Cannot block the vLLM engine owner loop")
        return self.submit(coro).result()

    async def run_async(self, coro: Coroutine[Any, Any, ResultT]) -> ResultT:
        """Await ``coro`` on the owner loop from any loop.

        Cancelling the caller cancels the owner-loop task.
        """
        if self.is_owner_thread():
            return await coro
        future = self.submit(coro)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def shutdown(self, timeout_s: float = 10.0) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.thread.join(timeout=timeout_s)
        if self.thread.is_alive():
            raise RuntimeError("vLLM engine owner loop failed to stop")


def on_engine_loop(
    method: Callable[..., Awaitable[ResultT]],
) -> Callable[..., Awaitable[ResultT]]:
    """Run an async worker method on the engine owner loop when one is active.

    The worker exposes ``_engine_owner_loop`` (``None`` when the feature is
    off). Without an owner loop the method runs unchanged on the caller's loop.
    Nested calls already on the owner loop run inline.
    """

    @functools.wraps(method)
    async def wrapper(self, *args: Any, **kwargs: Any) -> ResultT:
        owner_loop: EngineOwnerLoop | None = getattr(self, "_engine_owner_loop", None)
        if owner_loop is None:
            return await method(self, *args, **kwargs)
        return await owner_loop.run_async(method(self, *args, **kwargs))

    return wrapper
