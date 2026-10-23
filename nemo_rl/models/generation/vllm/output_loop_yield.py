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

"""Let vLLM's front-end output loops yield to other tasks under a generation backlog.

AsyncLLM runs two coroutines on the generation actor's event loop:

  * ``process_outputs_socket`` (``core_client.AsyncMPClient``) receives EngineCore's
    output messages from a ZMQ socket and decodes them;
  * ``output_handler`` (``async_llm.AsyncLLM``) takes the decoded outputs off a queue
    and does the per-token work: detokenization, logprob objects, stats.

Both ``await`` something that is already complete when a backlog exists -- a message
waiting in the socket, an item waiting in the queue -- and awaiting a completed future
does not hand control back to the event loop. ``output_handler`` only yields between
chunks of outputs within one engine step, never between steps. So once the actor falls
behind its GPU, either loop can run for minutes without letting any other task run:
refit calls, every other Ray call to the actor and new HTTP requests all wait. A
single-node benchmark with 550 requests in flight measured 85-401 s stops.

With this patch, ``output_handler`` yields once per engine step, and the socket reader
yields once it has run continuously for ``reader_budget_s``. Yielding the reader on
every message instead starves it against ``output_handler``'s larger per-step work: the
backlog moves into the socket and replies handled by the reader (such as a
``collective_rpc`` result) are picked up late. The budget gives the reader a fair share
of the loop while bounding how long any loop can hold it.

This bounds how long control calls wait behind generation; it does not make the
per-token work cheaper.
"""

import asyncio
import time
from typing import Any


def install_output_loop_yield(llm: Any, *, reader_budget_s: float) -> None:
    """Make ``llm``'s output loops yield to other tasks on the event loop.

    Args:
        llm: The ``vllm.v1.engine.async_llm.AsyncLLM`` whose output loops to patch.
        reader_budget_s: How long the socket reader may run without yielding.

    Raises:
        ValueError: If ``reader_budget_s`` is not positive.
        RuntimeError: If the installed vLLM does not have the internals this patches.
    """
    if reader_budget_s <= 0:
        raise ValueError(f"reader_budget_s must be positive, got {reader_budget_s}")
    try:
        from vllm.v1.engine.core_client import AsyncMPClient

        original_get = AsyncMPClient.get_output_async
        socket = llm.engine_core.resources.output_socket
        original_recv = socket.recv_multipart
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "output-loop yielding is enabled (vllm_cfg.output_loop_yield_budget_ms) but "
            "the installed vLLM does not have the AsyncMPClient output-loop internals it "
            "patches; disable it or update nemo_rl/models/generation/vllm/"
            "output_loop_yield.py for this vLLM version"
        ) from e

    # output_handler: one yield after each EngineCoreOutputs taken off the queue. Patched
    # on the class once per process; each output_handler task calls it.
    if not getattr(AsyncMPClient, "_nrl_output_loop_yield", False):

        async def get_output_async(self: Any) -> Any:
            outputs = await original_get(self)
            await asyncio.sleep(0)
            return outputs

        AsyncMPClient.get_output_async = get_output_async
        AsyncMPClient._nrl_output_loop_yield = True

    # process_outputs_socket: yield after reader_budget_s of continuous work. Patched on
    # this client's socket only (pyzmq allows overriding class-defined attributes on the
    # instance), so other ZMQ users in the process are unaffected.
    state = {"since": time.monotonic()}

    async def recv_multipart(*args: Any, **kwargs: Any) -> Any:
        # The zmq.asyncio socket returns a future that is already done when a message
        # was waiting. Awaiting a done future does not yield, so that is exactly the case
        # that can monopolize the loop.
        future = original_recv(*args, **kwargs)
        if not future.done():
            # The receive waits for a message, so the loop gets its turn: start a new
            # budget once it arrives.
            frames = await future
            state["since"] = time.monotonic()
            return frames
        frames = await future
        if time.monotonic() - state["since"] >= reader_budget_s:
            await asyncio.sleep(0)
            state["since"] = time.monotonic()
        return frames

    socket.recv_multipart = recv_multipart
