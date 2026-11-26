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
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
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
# How many uncut model call IDs a merged cut's warning names.
_LOGGED_UNCUT_CALLS = 5


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

    async def run_cut(
        self, inventory: Any, *, deadline: float | None = None, held_only: bool = False
    ) -> Any:
        """Cut the calls Gym's inventory names, by a ``time.monotonic()`` deadline.

        With ``held_only``, the receipt names only the calls this process holds.
        """
        ...


def merge_cut_receipts(inventory: Any, engine_prefixes: Iterable[Sequence[Any]]) -> Any:
    """Build Gym's receipt for ``inventory`` from each engine's acks.

    An engine (a Dynamo engine, or one of a worker's HTTP frontends) acks only
    the calls it holds: the ones it cut, and failures it has already logged. A
    cut ack wins over a failure for the same call. A call no engine holds, say
    one whose request had not arrived, gets Gym's failure ack and one warning
    per cut; it regenerates after a restore.
    """
    from nemo_gym._checkpoint.generation_cut import (
        GenerationCutPrefixAck,
        GenerationCutReceipt,
    )

    held: dict[str, Any] = {}
    for prefixes in engine_prefixes:
        for prefix in prefixes:
            ack = GenerationCutPrefixAck.model_validate(prefix)
            previous = held.get(ack.ticket_id)
            if previous is None or (
                previous.disposition != "durable_prefix"
                and ack.disposition == "durable_prefix"
            ):
                held[ack.ticket_id] = ack
    unheld = [
        prefix.model_call_id
        for prefix in inventory.active_prefixes
        if prefix.ticket_id not in held
    ]
    if unheld:
        LOGGER.warning(
            "generation cut %s: no engine holds %d of %d model calls; they "
            "regenerate after a restore: %s",
            inventory.checkpoint_id,
            len(unheld),
            len(inventory.active_prefixes),
            unheld[:_LOGGED_UNCUT_CALLS],
        )
    return GenerationCutReceipt(
        checkpoint_id=inventory.checkpoint_id,
        cut_id=f"frontend-{inventory.inventory_digest}",
        inventory_digest=inventory.inventory_digest,
        inventory=inventory,
        backend_snapshot_id=f"tq-{inventory.inventory_digest}",
        prefixes=tuple(
            held.get(prefix.ticket_id) or GenerationCutPrefixAck.failure(prefix)
            for prefix in inventory.active_prefixes
        ),
    )


async def fan_out_cut(
    inventory: Any,
    cuts: Mapping[str, Awaitable[Sequence[Any]]],
    *,
    deadline: float | None,
) -> Any:
    """Wait for every target's cut until ``deadline`` and merge their acks.

    Each awaitable in ``cuts`` (keyed by a name for the logs) returns the
    acks of one target: a Dynamo engine, or one of a worker's HTTP frontends. A target that misses the wall-clock ``deadline`` is cancelled
    and its calls regenerate after a restore. A target that should cost only
    its own calls when it fails must catch its errors and return ``()``.
    """
    tasks = {asyncio.ensure_future(cut): name for name, cut in cuts.items()}
    done: set[asyncio.Task[Sequence[Any]]] = set()
    if tasks:
        done, late = await asyncio.wait(
            tasks,
            timeout=None if deadline is None else max(0.0, deadline - time.time()),
        )
        for task in late:
            task.cancel()
        if late:
            LOGGER.warning(
                "generation cut %s: %s did not answer in time; calls they hold "
                "regenerate after a restore",
                inventory.checkpoint_id,
                sorted(tasks[task] for task in late),
            )
    # In target order, so which ack wins for a call acked twice does not vary.
    return merge_cut_receipts(
        inventory, [task.result() for task in tasks if task in done]
    )


def held_acks(receipt: Any) -> list[dict[str, Any]]:
    """A ``held_only`` receipt's acks, as one engine answers a fan-out."""
    return [ack.model_dump(mode="json") for ack in receipt.prefixes]


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
        # This frontend's own cut is not wrapped: if it fails (say its prefix
        # writer died), the whole cut fails, as it does with one frontend.
        return await fan_out_cut(
            inventory,
            {
                "this frontend": self._cut_here(inventory, deadline),
                **{
                    url: self._cut_on_peer(url, inventory, deadline)
                    for url in self._peer_urls
                },
            },
            deadline=deadline,
        )

    async def _cut_here(self, inventory: Any, deadline: float | None) -> list[Any]:
        receipt = await self._recovery.run_cut(
            inventory, deadline=local_deadline(deadline), held_only=True
        )
        return held_acks(receipt)

    async def _cut_on_peer(
        self, url: str, inventory: Any, deadline: float | None
    ) -> Sequence[Any]:
        try:
            # fan_out_cut enforces the deadline; aiohttp's own 300 s default
            # would end a longer cut with an error instead.
            timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{url}{FRONTEND_CUT_ROUTE}",
                    json={
                        "inventory": inventory.model_dump(mode="json"),
                        "deadline": deadline,
                    },
                    headers={"authorization": f"Bearer {self.control_token}"},
                ) as response:
                    response.raise_for_status()
                    prefixes = (await response.json())["prefixes"]
            cut = sorted(
                prefix["model_call_id"]
                for prefix in prefixes
                if prefix["disposition"] == "durable_prefix"
            )
            LOGGER.info(
                "generation cut %s: HTTP frontend %s cut %d model calls: %s",
                inventory.checkpoint_id,
                url,
                len(cut),
                cut,
            )
            return prefixes
        # One frontend's failure must cost only the calls it holds, whatever it is.
        except Exception:  # noqa: BLE001
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
        """Cut this frontend's share of a fan-out; answer for the calls it holds."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

        current = recovery()
        refusal = _refusal(current, authorization)
        if refusal is not None:
            raise HTTPException(status_code=refusal[0], detail=refusal[1]["detail"])
        receipt = await current.run_cut(
            GenerationCutInventory.model_validate(request["inventory"]),
            deadline=local_deadline(request["deadline"]),
            held_only=True,
        )
        return {"prefixes": held_acks(receipt)}
