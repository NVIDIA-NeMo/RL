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

"""Test-only pass-through proxy between NeMo-Gym's policy model server and vLLM.

The turn-recovery functional test needs a published checkpoint that holds a
turn whose tool call has already changed the resources state. Gym counts a
session at its boundary while it waits on a policy call made right after that
boundary, and a held call never blocks a checkpoint prepare. So holding the
first policy request that carries a tool result leaves Gym parked at the
post-tool boundary through production code, until the test kills the process.

After a tool result, the proxy also decides the next call. With a
``follow_up_tool``, a request whose episode has not called it yet is sent with
that tool forced, and every later request with ``tool_choice: none``; without
one, every request after a tool result is sent with ``tool_choice: none``.
So no turn, including the one a restored episode re-issues, can call the
episode's first tool a second time, and a read-only follow-up tool can report
the resources state the episode continued from. Only ``tool_choice`` changes;
the tool list stays, so the rendered prompt keeps the prefix token capture
checks. The proxy logs each follow-up tool call it sees answered, with its
arguments and result.

Each backend gets its own listener, so Gym's per-session endpoint choice still
maps onto one backend. Listeners bind every interface and advertise this node's
address, so Gym replicas on other nodes reach them too.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Optional

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local
from tests.functional._single_controller_hook_utils import append_event

_SKIPPED_REQUEST_HEADERS = frozenset({"host", "content-length", "connection"})
_SKIPPED_RESPONSE_HEADERS = frozenset(
    {"content-length", "transfer-encoding", "connection", "content-encoding"}
)
_STREAM_CHUNK_BYTES = 64 * 1024


def _carries_tool_result(body: Any) -> bool:
    messages = body.get("messages") if isinstance(body, dict) else None
    return isinstance(messages, list) and any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in messages
    )


def _tool_calls(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The assistant tool calls in the conversation so far, by call ID."""
    calls = {}
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                calls[str(call.get("id"))] = call.get("function") or {}
    return calls


def _tool_results(body: dict[str, Any]) -> dict[str, Any]:
    """The tool results in the conversation so far, by the call ID they answer."""
    return {
        str(message.get("tool_call_id")): message.get("content")
        for message in body.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "tool"
    }


def _rollout_id(body: dict[str, Any]) -> Optional[str]:
    capture = body.get("ng_capture")
    if isinstance(capture, dict):
        rollout_id = capture.get("rollout_id")
        return str(rollout_id) if rollout_id is not None else None
    return None


class PolicyHoldProxy:
    """Forward Gym's policy calls to vLLM, holding at most one post-tool request."""

    def __init__(
        self,
        backend_urls: list[str],
        *,
        hold_first_tool_result: bool,
        follow_up_tool: Optional[str],
        events_path: Path,
    ) -> None:
        if not backend_urls:
            raise ValueError("the policy proxy needs at least one backend URL")
        self._backends = list(backend_urls)
        self._hold_first_tool_result = hold_first_tool_result
        self._follow_up_tool = follow_up_tool
        self._events_path = events_path
        self._host = _get_node_ip_local()
        self._ports: list[int] = []
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, name="gym-policy-hold-proxy", daemon=True
        )

    @property
    def base_urls(self) -> list[str]:
        """The proxied URLs, in backend order, in the ``http://host:port/v1`` form."""
        return [f"http://{self._host}:{port}/v1" for port in self._ports]

    def start(self) -> "PolicyHoldProxy":
        self._thread.start()
        if not self._started.wait(timeout=60):
            raise TimeoutError("the policy proxy did not start within 60s")
        if self._startup_error is not None:
            raise RuntimeError("the policy proxy failed to start") from (
                self._startup_error
            )
        append_event(
            self._events_path,
            "policy_proxy_started",
            backends=self._backends,
            proxied=self.base_urls,
            hold_first_tool_result=self._hold_first_tool_result,
            follow_up_tool=self._follow_up_tool,
        )
        return self

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as error:  # surfaced to start()
            self._startup_error = error
            self._started.set()
            return
        self._loop.run_forever()

    async def _serve(self) -> None:
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

        # No limit and no timeout: the rollout layer owns deadlines, and a held
        # request must stay open until the process is killed.
        self._session = ClientSession(
            connector=TCPConnector(limit=0), timeout=ClientTimeout(total=None)
        )
        self._held = False
        for backend in self._backends:
            app = web.Application(client_max_size=1024**3)

            async def handle(request: Any, backend: str = backend) -> Any:
                return await self._handle(request, backend)

            app.router.add_route("*", "/{tail:.*}", handle)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host="0.0.0.0", port=0)
            await site.start()
            self._ports.append(site._server.sockets[0].getsockname()[1])
        self._started.set()

    async def _handle(self, request: Any, backend: str) -> Any:
        from aiohttp import web

        payload = await request.read()
        if request.method == "POST" and request.path == "/v1/chat/completions":
            body = json.loads(payload)
            if _carries_tool_result(body):
                rollout_id = _rollout_id(body)
                if self._hold_first_tool_result and not self._held:
                    self._held = True
                    append_event(
                        self._events_path,
                        "policy_call_held",
                        rollout_id=rollout_id,
                        backend=backend,
                    )
                    # Never answered: Gym stays at the post-tool boundary until
                    # the process is killed.
                    await asyncio.Event().wait()
                calls = _tool_calls(body)
                results = _tool_results(body)
                for call_id, function in calls.items():
                    if (
                        function.get("name") == self._follow_up_tool
                        and call_id in results
                    ):
                        append_event(
                            self._events_path,
                            "follow_up_tool_result",
                            rollout_id=rollout_id,
                            tool=self._follow_up_tool,
                            arguments=function.get("arguments"),
                            result=results[call_id],
                        )
                followed_up = any(
                    function.get("name") == self._follow_up_tool
                    for function in calls.values()
                )
                tool_choice: Any = (
                    {"type": "function", "function": {"name": self._follow_up_tool}}
                    if self._follow_up_tool is not None and not followed_up
                    else "none"
                )
                body["tool_choice"] = tool_choice
                payload = json.dumps(body).encode()
                append_event(
                    self._events_path,
                    "policy_call_tool_choice",
                    rollout_id=rollout_id,
                    backend=backend,
                    tool_choice=tool_choice,
                )

        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _SKIPPED_REQUEST_HEADERS
        }
        target = backend.removesuffix("/v1") + request.rel_url.path_qs
        async with self._session.request(
            request.method, target, data=payload, headers=headers
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers={
                    name: value
                    for name, value in upstream.headers.items()
                    if name.lower() not in _SKIPPED_RESPONSE_HEADERS
                },
            )
            try:
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(_STREAM_CHUNK_BYTES):
                    await response.write(chunk)
                await response.write_eof()
            except ConnectionResetError:
                # The client gave up before the backend answered, as Gym's
                # startup probes do while vLLM loads; there is no one to tell.
                pass
            return response
