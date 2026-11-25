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

"""The HTTP route Gym calls to cut this worker's live model calls at a checkpoint."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

import aiohttp
from fastapi import FastAPI, Header, HTTPException

LOGGER = logging.getLogger(__name__)

# Gym owns this route (nemo_gym._checkpoint.generation_cut.GENERATION_CUT_ROUTE),
# but nemo_gym is an optional extra. A drifted copy would 404 every cut, which Gym
# treats as "regenerate", so test_generation_cut_route_matches_gym pins it.
GENERATION_CUT_ROUTE = "/ng-control/v1/generation-cut"
# One HTTP frontend's share of a cut, requested by the frontend Gym reached.
FRONTEND_CUT_ROUTE = f"{GENERATION_CUT_ROUTE}/frontend"
# What a cut's participant leaves of the coordinator's deadline for its answer
# to reach the coordinator.
CUT_REPLY_MARGIN_S = 1.0


class GenerationCutBackend(Protocol):
    """What runs a cut: a worker's prefix recovery, or the Dynamo frontend's fan-out."""

    @property
    def control_token(self) -> str | None:
        """The bearer token a cut must carry; None refuses every cut."""
        ...

    async def run_cut(self, inventory: Any) -> Any:
        """Cut the calls Gym's inventory names and return Gym's receipt."""
        ...


class LocalCutBackend(GenerationCutBackend, Protocol):
    """A worker's own prefix recovery: cuts to a caller's deadline."""

    @property
    def control_timeout_s(self) -> float | None:
        """How long a cut may take from its arrival; None waits indefinitely."""
        ...

    async def run_cut(self, inventory: Any, *, deadline: float | None = None) -> Any:
        """Cut the calls Gym's inventory names, by a ``time.monotonic()`` deadline."""
        ...


def merge_cut_receipts(inventory: Any, engine_prefixes: Iterable[Sequence[Any]]) -> Any:
    """Build Gym's receipt for ``inventory`` from what each engine cut.

    An engine (a Dynamo engine, or one of a worker's HTTP frontends)
    acknowledges only the calls it cut. A call no engine acknowledged gets
    Gym's failure ack, so it regenerates after a restore.
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


def durable_prefixes(receipt: Any) -> list[dict[str, Any]]:
    """The acks of the calls a receipt cut, as one engine answers a fan-out."""
    return [
        ack.model_dump(mode="json")
        for ack in receipt.prefixes
        if ack.disposition == "durable_prefix"
    ]


def local_deadline(wall_deadline: float | None) -> float | None:
    """A coordinator's wall-clock cut deadline as this process's monotonic one.

    Less ``CUT_REPLY_MARGIN_S``, so the answer reaches the coordinator in time.
    """
    if wall_deadline is None:
        return None
    return time.monotonic() + (wall_deadline - time.time()) - CUT_REPLY_MARGIN_S


def _refusal(
    backend: GenerationCutBackend, authorization: str | None
) -> tuple[int, dict[str, Any]] | None:
    """The HTTP status and body refusing a cut call, or None to run it."""
    expected = backend.control_token
    supplied = (
        authorization.removeprefix("Bearer ")
        if authorization is not None and authorization.startswith("Bearer ")
        else ""
    )
    if expected is None:
        return 404, {"detail": "generation-prefix cuts are disabled"}
    if not secrets.compare_digest(supplied, expected):
        return 401, {"detail": "invalid control bearer"}
    return None


async def answer_generation_cut(
    backend: GenerationCutBackend,
    *,
    authorization: str | None,
    inventory: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Answer one generation-cut call with its HTTP status and JSON body.

    Shared by every server of the route: the worker's FastAPI app, and the
    Dynamo frontend's route (``dynamo/vllm/generation_cut.py``). ``inventory`` is
    the call's JSON body, Gym's ``GenerationCutInventory``.
    """
    # Deferred: nemo_gym is an optional extra.
    from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

    refusal = _refusal(backend, authorization)
    if refusal is not None:
        return refusal
    receipt = await backend.run_cut(GenerationCutInventory.model_validate(inventory))
    return 200, receipt.model_dump(mode="json")


class FrontendCutFanOut:
    """Cuts a checkpoint's calls on every HTTP frontend of one worker.

    With several frontends sharing one listening socket, Gym's cut reaches one
    of them, but a call's state lives only in the frontend that accepted its
    request. Like the Dynamo frontend's ``EngineCutFanOut``, this cuts here and
    on every other frontend, over their loopback control URLs, and merges the
    answers.
    """

    def __init__(self, recovery: LocalCutBackend, peer_urls: Sequence[str]) -> None:
        """Wrap this frontend's prefix recovery and the other frontends' URLs."""
        self._recovery = recovery
        self._peer_urls = peer_urls

    @property
    def control_token(self) -> str | None:
        """The bearer token a cut needs; every frontend shares it."""
        return self._recovery.control_token

    async def run_cut(self, inventory: Any) -> Any:
        """Cut ``inventory``'s calls on every frontend and merge their answers."""
        # One deadline, from the cut's arrival, for this wait and every frontend.
        timeout_s = self._recovery.control_timeout_s
        deadline = None if timeout_s is None else time.time() + timeout_s
        cuts = [
            asyncio.ensure_future(self._cut_here(inventory, deadline)),
            *(
                asyncio.ensure_future(self._cut_on_peer(url, inventory, deadline))
                for url in self._peer_urls
            ),
        ]
        done, late = await asyncio.wait(
            cuts,
            timeout=None if deadline is None else max(0.0, deadline - time.time()),
        )
        for cut in late:
            cut.cancel()
        if late:
            LOGGER.warning(
                "generation cut %s: %d of %d HTTP frontends did not answer in "
                "time; calls they hold regenerate after a restore",
                inventory.checkpoint_id,
                len(late),
                len(cuts),
            )
        return merge_cut_receipts(inventory, [cut.result() for cut in done])

    async def _cut_here(self, inventory: Any, deadline: float | None) -> list[Any]:
        receipt = await self._recovery.run_cut(
            inventory, deadline=local_deadline(deadline)
        )
        return durable_prefixes(receipt)

    async def _cut_on_peer(
        self, url: str, inventory: Any, deadline: float | None
    ) -> Sequence[Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}{FRONTEND_CUT_ROUTE}",
                    json={
                        "inventory": inventory.model_dump(mode="json"),
                        "deadline": deadline,
                    },
                    headers={"authorization": f"Bearer {self.control_token}"},
                ) as response:
                    response.raise_for_status()
                    return (await response.json())["prefixes"]
        # One frontend's failure must cost only the calls it holds.
        except (aiohttp.ClientError, KeyError, TypeError, ValueError):
            LOGGER.warning(
                "generation cut failed on HTTP frontend %s; calls it holds "
                "regenerate after a restore",
                url,
                exc_info=True,
            )
            return ()


def register_generation_cut_route(
    app: FastAPI,
    recovery: Callable[[], LocalCutBackend],
    peer_urls: Callable[[], Sequence[str]],
) -> None:
    """Serve Gym's generation-cut call from the worker's prefix recovery.

    Args:
        app: The frontend's FastAPI app.
        recovery: Returns the frontend's current ``PrefixRecovery`` or
            ``NoPrefixRecovery``. It is a callable because the server starts
            before ``setup_token_capture`` builds the recovery object.
        peer_urls: Returns the loopback control URLs of the worker's other HTTP
            frontends; empty with a single frontend.
    """

    @app.post(GENERATION_CUT_ROUTE)
    async def checkpoint_generation_cut(
        inventory: dict,
        authorization: str | None = Header(default=None),
    ):
        """Persist one checkpoint's frozen active-call prefixes to TQ."""
        peers = peer_urls()
        backend = FrontendCutFanOut(recovery(), peers) if peers else recovery()
        status, body = await answer_generation_cut(
            backend, authorization=authorization, inventory=inventory
        )
        if status != 200:
            raise HTTPException(status_code=status, detail=body["detail"])
        return body

    @app.post(FRONTEND_CUT_ROUTE)
    async def frontend_generation_cut(
        request: dict,
        authorization: str | None = Header(default=None),
    ):
        """Cut this frontend's share of a fan-out; answer with its durable acks."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

        current = recovery()
        refusal = _refusal(current, authorization)
        if refusal is not None:
            raise HTTPException(status_code=refusal[0], detail=refusal[1]["detail"])
        receipt = await current.run_cut(
            GenerationCutInventory.model_validate(request["inventory"]),
            deadline=local_deadline(request["deadline"]),
        )
        return {"prefixes": durable_prefixes(receipt)}
