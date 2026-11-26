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
frontend serves the route itself, as a Dynamo route extension (``routes``), and
sends the cut to every engine over Dynamo's request plane. Each engine answers
for the calls it holds, and the answers are merged into the one receipt Gym
expects.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

from nemo_rl.models.generation.dynamo.managed_runtime import (
    CAPTURE_FILE_ENV,
    DYNAMO_COMPONENT,
    DYNAMO_ENDPOINT,
    HTTP_BODY_LIMIT_MB,
    read_chat_capture,
)
from nemo_rl.models.generation.vllm.generation_cut_route import (
    GENERATION_CUT_ROUTE,
    answer_generation_cut,
    fan_out_cut,
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
# A cut times itself (the deadline above), so the route's own timeout is only a
# backstop: the longest Dynamo accepts.
_ROUTE_TIMEOUT_S = 365 * 24 * 60 * 60


class EngineCutFanOut:
    """Cuts a checkpoint's calls on every engine: what the cut route runs."""

    def __init__(self, runtime: Any, capture_file: str) -> None:
        """Open the request-plane client. Call it on the frontend's event loop.

        Args:
            runtime: The frontend's ``DistributedRuntime``.
            capture_file: Where the driver publishes the capture config, which
                holds a cut's bearer token and timeout.
        """
        self._capture_file = capture_file
        path = f"{os.environ['DYN_NAMESPACE']}.{DYNAMO_COMPONENT}.{DYNAMO_ENDPOINT}"
        # Dynamo returns a future here, not a coroutine.
        self._client = asyncio.ensure_future(runtime.endpoint(path).client())

    @property
    def control_token(self) -> str | None:
        """The bearer token a cut needs; None until capture is set up with cuts on."""
        capture = read_chat_capture(self._capture_file)
        return None if capture is None else capture["generation_cut_control_token"]

    async def run_cut(self, inventory: Any) -> Any:
        """Cut ``inventory``'s calls on every engine and merge their answers."""
        # One deadline, from the cut's arrival, for this wait and every engine:
        # an engine that started its own later would answer after this gave up.
        timeout_s = (read_chat_capture(self._capture_file) or {}).get(
            "generation_cut_control_timeout_s"
        )
        deadline = None if timeout_s is None else time.time() + timeout_s
        return await self._cut_on_every_engine(inventory, deadline)

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
        return await fan_out_cut(
            inventory,
            {
                f"engine {instance_id}": self._cut_on_engine(
                    client, request, instance_id
                )
                for instance_id in client.instance_ids()
            },
            deadline=deadline,
        )

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


class CutRouteHandler:
    """Answers NeMo-Gym's cut call from the context Dynamo hands a route's handler."""

    def __init__(self, capture_file: str) -> None:
        self._capture_file = capture_file
        self._fan_out: EngineCutFanOut | None = None

    async def answer(self, context: Any) -> tuple[int, dict[str, Any]]:
        """The call's HTTP status and JSON body, as the native route answers it."""
        if self._fan_out is None:
            # Dynamo hands over its runtime only with a request. The client
            # shares the frontend's discovery watch, so it lists every engine.
            self._fan_out = EngineCutFanOut(context.runtime, self._capture_file)
        return await answer_generation_cut(
            self._fan_out,
            authorization=context.headers.get("authorization"),
            inventory=json.loads(context.body),
        )


def routes() -> list[Any]:
    """Dynamo's route extension provider: the cut route, on the frontend's own port."""
    # Deferred: ai-dynamo is an optional dependency, and only the frontend
    # process calls this.
    from dynamo.llm import FrontendResponse, FrontendRoute

    # Gym is imported at launch where it is installed: a lazy import that met
    # another thread's first import of Gym midway would get a half-built package.
    if importlib.util.find_spec("nemo_gym") is not None:
        # Gym points an unset HF_HOME at its own cache directory, where the
        # early render-worker build would then look for a Hub ID's files.
        hf_home = os.environ.get("HF_HOME")
        importlib.import_module("nemo_gym._checkpoint.generation_cut")
        if hf_home is None:
            os.environ.pop("HF_HOME", None)
    handler = CutRouteHandler(os.environ[CAPTURE_FILE_ENV])

    async def serve_cut(context: Any) -> Any:
        status, body = await handler.answer(context)
        return body if status == 200 else FrontendResponse(status, body)

    return [
        FrontendRoute(
            "POST",
            GENERATION_CUT_ROUTE,
            serve_cut,
            timeout_s=_ROUTE_TIMEOUT_S,
            max_body_bytes=HTTP_BODY_LIMIT_MB << 20,
        )
    ]
