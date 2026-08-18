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

"""Tool-call scheduling for the central agent loop."""

import asyncio
import json
from typing import Any, Awaitable, Callable, Optional

CallTool = Callable[[str, dict[str, Any]], Awaitable[str]]


def function_call_output(call_id: Optional[str], output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


class BaseToolCallManager:
    """Runs a turn's tool calls and returns Responses-API ``function_call_output`` items.

    Failure handling matches ``simple_agent/app.py``: unparseable arguments
    become an error tool result, and a non-2xx tool response body is passed
    through as the tool output. Neither aborts the rollout. Transport errors
    still propagate.
    """

    def __init__(self, cfg: Optional[dict[str, Any]], call_tool: CallTool) -> None:
        self.cfg = cfg or {}
        self._call_tool = call_tool
        self._submitted = 0
        self._malformed = 0
        self._returned = 0

    async def submit_and_collect(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Take this turn's calls and return the results that are ready."""
        raise NotImplementedError

    async def drain(self) -> list[dict[str, Any]]:
        """Finish anything still outstanding. Nothing is, unless calls are queued."""
        return []

    def stats(self) -> dict[str, Any]:
        return {
            "tool_calls_total": self._submitted,
            "tool_calls_malformed": self._malformed,
            "tool_calls_returned": self._returned,
            "tool_calls_pending": self.pending_count,
        }

    @property
    def pending_count(self) -> int:
        return 0

    async def _call_and_wrap(
        self, call: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        # A nameless call POSTs to "/" and gets a 404 body back as its tool
        # output, which is the same path an unknown tool takes.
        name: Any = call.get("name") or ""
        output = await self._call_tool(name, arguments)
        return function_call_output(call.get("call_id"), output)

    def _parse_arguments(
        self, call: dict[str, Any]
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        """Parse tool arguments; on failure return the error result to send back."""
        self._submitted += 1
        arguments: Any = call.get("arguments")  # None is a TypeError, handled below
        try:
            return json.loads(arguments), None
        except (json.JSONDecodeError, TypeError) as e:
            # Surface the error as a tool response so the rollout continues (or
            # ends with a low reward) instead of crashing the batch. repr(e) so
            # the exception type is included even when str(e) is empty.
            self._malformed += 1
            return {}, function_call_output(
                call.get("call_id"),
                json.dumps({"error": f"Invalid tool call arguments: {e!r}"}),
            )


class SerialToolCallManager(BaseToolCallManager):
    """Executes every call of the batch in the model's original order."""

    async def submit_and_collect(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results = []
        for call in calls:
            arguments, error = self._parse_arguments(call)
            results.append(
                error
                if error is not None
                else await self._call_and_wrap(call, arguments)
            )
        self._returned += len(results)
        return results


class ParallelToolCallManager(BaseToolCallManager):
    """One queue per rollout, spanning turns.

    On a turn with ``n`` new calls it submits them, then returns every result
    finished so far as soon as ``finished >= min(min_returns, n)``; unfinished
    calls stay queued and are returned by a later turn, so ``n == 0`` returns
    immediately. Results are always ordered by submission index.
    """

    def __init__(self, cfg: Optional[dict[str, Any]], call_tool: CallTool) -> None:
        super().__init__(cfg, call_tool)
        self._min_returns = max(1, int(self.cfg.get("min_returns") or 1))
        self._pending: dict[int, "asyncio.Future[dict[str, Any]]"] = {}
        self._finished: dict[int, dict[str, Any]] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def submit_and_collect(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        threshold = min(self._min_returns, len(calls))
        for call in calls:
            self._submit(call)
        self._harvest()
        while len(self._finished) < threshold and self._pending:
            await asyncio.wait(
                self._pending.values(), return_when=asyncio.FIRST_COMPLETED
            )
            self._harvest()
        return self._take_finished()

    async def drain(self) -> list[dict[str, Any]]:
        if self._pending:
            await asyncio.wait(self._pending.values())
        self._harvest()
        return self._take_finished()

    def _submit(self, call: dict[str, Any]) -> None:
        index = self._submitted  # _parse_arguments increments it
        arguments, error = self._parse_arguments(call)
        if error is not None:
            self._finished[index] = error
            return
        self._pending[index] = asyncio.get_running_loop().create_task(
            self._call_and_wrap(call, arguments)
        )

    def _harvest(self) -> None:
        """Move completed calls out of the queue. Results accumulate until taken."""
        for index, task in list(self._pending.items()):
            if task.done():
                self._finished[index] = task.result()
                del self._pending[index]

    def _take_finished(self) -> list[dict[str, Any]]:
        """Drain everything finished, in the model's original call order."""
        ready = [self._finished.pop(index) for index in sorted(self._finished)]
        self._returned += len(ready)
        return ready
