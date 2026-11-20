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

"""Checkpoint generation cuts for NeMo-Gym calls served through Dynamo.

At a checkpoint NeMo-Gym posts the calls it has in flight to
``/ng-control/v1/generation-cut`` beside the base URL they went to. Behind
Dynamo that URL is the frontend, and the calls run in the engines. So the
frontend process serves the route on a control port of its own and sends the
cut to every engine over Dynamo's request plane. Each engine answers for the
calls it cut, and the answers are merged into the one receipt Gym expects.

The control port is there because a Dynamo frontend route can only be a
synchronous GET (ai-dynamo/dynamo#18486). Once routes take POST and async
handlers, this route can move onto the frontend's own port.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Any

from nemo_rl.models.generation.dynamo.managed_runtime import (
    DYNAMO_COMPONENT,
    DYNAMO_ENDPOINT,
    GENERATION_CUT_PORT_ENV,
    read_chat_capture,
)

LOGGER = logging.getLogger(__name__)

# Marks an engine request as a generation cut; its inventory rides in extra_args.
GENERATION_CUT_ANNOTATION = "nemo_rl:generation_cut"
INVENTORY_KEY = "nemo_rl_generation_cut"
# The calls an engine cut, in its answer's engine_data.
CUT_PREFIXES_KEY = "prefixes"
# The frontend's deadline for the cut, in extra_args: wall-clock seconds, since
# an engine on another host does not share the frontend's monotonic clock.
DEADLINE_KEY = "nemo_rl_generation_cut_deadline"
# What an engine leaves of that deadline for its answer to reach the frontend.
CUT_REPLY_MARGIN_S = 1.0


def merge_cut_receipts(inventory: Any, engine_prefixes: Iterable[Sequence[Any]]) -> Any:
    """Build Gym's receipt for ``inventory`` from what each engine cut.

    An engine acknowledges only the calls it cut. A call no engine
    acknowledged gets Gym's failure ack, so it regenerates after a restore.
    """
    from nemo_gym._checkpoint.generation_cut import (
        GenerationCutPrefixAck,
        GenerationCutReceipt,
    )

    cut: dict[str, Any] = {}
    for prefixes in engine_prefixes:
        for prefix in prefixes:
            ack = GenerationCutPrefixAck.model_validate(prefix)
            cut.setdefault(ack.ticket_id, ack)
    return GenerationCutReceipt(
        checkpoint_id=inventory.checkpoint_id,
        cut_id=f"frontend-{inventory.inventory_digest}",
        inventory_digest=inventory.inventory_digest,
        inventory=inventory,
        backend_snapshot_id=f"tq-{inventory.inventory_digest}",
        prefixes=tuple(
            cut.get(prefix.ticket_id) or GenerationCutPrefixAck.failure(prefix)
            for prefix in inventory.active_prefixes
        ),
    )


class EngineCutFanOut:
    """Cuts a checkpoint's calls on every engine: what the cut route runs."""

    def __init__(self, runtime: Any, loop: asyncio.AbstractEventLoop) -> None:
        """Open the request-plane client. Call it on ``loop``, the frontend's.

        The client is opened now, at startup, so that by the first checkpoint
        it has discovered every engine.
        """
        # The frontend's loop, which the runtime's request-plane client runs on.
        self._loop = loop
        path = f"{os.environ['DYN_NAMESPACE']}.{DYNAMO_COMPONENT}.{DYNAMO_ENDPOINT}"
        # Dynamo returns a future here, not a coroutine.
        self._client = asyncio.ensure_future(runtime.endpoint(path).client())

    @property
    def control_token(self) -> str | None:
        """The bearer token a cut needs; None until capture is set up with cuts on."""
        capture = read_chat_capture()
        return None if capture is None else capture["generation_cut_control_token"]

    async def run_cut(self, inventory: Any) -> Any:
        """Cut ``inventory``'s calls. Safe to await from the control server's loop."""
        # One deadline, from the cut's arrival, for this wait and every engine:
        # an engine that started its own later would answer after this gave up.
        timeout_s = (read_chat_capture() or {}).get("generation_cut_control_timeout_s")
        deadline = None if timeout_s is None else time.time() + timeout_s
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(
                self._cut_on_every_engine(inventory, deadline), self._loop
            )
        )

    async def _cut_on_every_engine(self, inventory: Any, deadline: float | None) -> Any:
        client = await self._client
        request = {
            # Dynamo's request type requires the field: without it the request
            # never reaches the engine and the caller is left waiting.
            "token_ids": [],
            "annotations": [GENERATION_CUT_ANNOTATION],
            "extra_args": {
                INVENTORY_KEY: inventory.model_dump(mode="json"),
                DEADLINE_KEY: deadline,
            },
        }
        cuts = {
            asyncio.ensure_future(
                self._cut_on_engine(client, request, instance_id)
            ): instance_id
            for instance_id in client.instance_ids()
        }
        engine_prefixes = []
        if cuts:
            done, late = await asyncio.wait(
                cuts,
                timeout=None if deadline is None else max(0.0, deadline - time.time()),
            )
            for cut in late:
                cut.cancel()
            if late:
                LOGGER.warning(
                    "generation cut %s: engines %s did not answer in time; calls "
                    "they hold regenerate after a restore",
                    inventory.checkpoint_id,
                    sorted(cuts[cut] for cut in late),
                )
            engine_prefixes = [cut.result() for cut in done]
        return merge_cut_receipts(inventory, engine_prefixes)

    @staticmethod
    async def _cut_on_engine(
        client: Any, request: dict[str, Any], instance_id: int
    ) -> Sequence[Any]:
        prefixes: Sequence[Any] = ()
        try:
            stream = await client.direct(request, instance_id)
            async for item in stream:
                data = item.data()
                if data and data.get("engine_data"):
                    prefixes = data["engine_data"][CUT_PREFIXES_KEY]
        # One engine's failure must cost only the calls it holds, whatever it is.
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "generation cut failed on engine %s; calls it holds regenerate "
                "after a restore",
                instance_id,
                exc_info=True,
            )
        return prefixes


def start_generation_cut_server(runtime: Any, loop: asyncio.AbstractEventLoop) -> None:
    """Serve Gym's generation-cut route on this frontend process's control port."""
    # Deferred: only the frontend process serves the route.
    import uvicorn
    from fastapi import FastAPI

    from nemo_rl.models.generation.vllm.generation_cut_route import (
        register_generation_cut_route,
    )

    fan_out = EngineCutFanOut(runtime, loop)
    app = FastAPI()
    register_generation_cut_route(app, lambda: fan_out)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="0.0.0.0",
            port=int(os.environ[GENERATION_CUT_PORT_ENV]),
            log_level="warning",
        )
    )
    # A thread of its own, as each engine's HTTP server has: on the frontend's
    # loop uvicorn would take over the frontend's signal handlers.
    threading.Thread(
        target=server.run, name="nrl-generation-cut-server", daemon=True
    ).start()
