# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
import threading


class AsyncLoopThread:
    """A background event loop thread for running async operations in Ray actors.

    This class creates a dedicated thread with its own event loop, allowing
    synchronous Ray actor methods to execute async coroutines without blocking
    the main actor thread. This is necessary because run_coroutine_threadsafe
    requires the event loop to be in a different thread.
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Event loop thread failed to start within 5 seconds")

    def _start_loop(self):
        """Run the event loop in the background thread."""
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def run(self, coro):
        """Schedule a coroutine onto the loop and block until it's done.

        Args:
            coro: The coroutine to execute

        Returns:
            The result of the coroutine
        """
        if not self.loop.is_running():
            raise RuntimeError("Event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        result = future.result()
        return result

    def shutdown(self):
        """Shutdown the event loop and wait for the thread to finish."""
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)
        if not self.loop.is_closed():
            self.loop.close()
