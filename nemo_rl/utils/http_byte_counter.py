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
"""Env-gated per-route HTTP byte counter (``NRL_HTTP_BYTES_DIR``).

Measurement tooling for the token-capture perf comparison: sums request-body
and response-body bytes per path on the vLLM worker's in-process HTTP server
and periodically flushes an aggregate JSON. Mirrors Gym's
``HttpByteCounterMiddleware`` (separate copy: worker venvs cannot assume
``nemo_gym`` is installed on the legacy path). Never installed unless the env
var is set.
"""

import json
import os
from typing import Any, Awaitable, Callable

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class HttpByteCounterMiddleware:
    """Pure ASGI wrapper counting per-path request/response body bytes."""

    FLUSH_EVERY = 25

    def __init__(self, app: Any, server_name: str, out_dir: str) -> None:
        self.app = app
        self.out_path = os.path.join(out_dir, f"{server_name}_{os.getpid()}.json")
        os.makedirs(out_dir, exist_ok=True)
        self.counts: dict[str, list[int]] = {}
        self._events = 0

    def _flush(self) -> None:
        with open(self.out_path, "w") as f:
            json.dump(
                {
                    path: {
                        "requests": c[0],
                        "req_bytes": c[1],
                        "resp_bytes": c[2],
                    }
                    for path, c in self.counts.items()
                },
                f,
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        entry = self.counts.setdefault(scope["path"], [0, 0, 0])
        entry[0] += 1

        async def counting_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                entry[1] += len(message.get("body", b""))
            return message

        async def counting_send(message: Message) -> None:
            if message["type"] == "http.response.body":
                entry[2] += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, counting_receive, counting_send)
        finally:
            self._events += 1
            if self._events % self.FLUSH_EVERY == 0:
                self._flush()
