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
from collections.abc import Callable, Sequence
from typing import Any

import aiohttp
from fastapi import FastAPI, Header, HTTPException

from nemo_rl.models.generation.prefix_recovery import (
    CutClaim,
    NoPrefixRecovery,
    PrefixRecovery,
    merge_cut_claims,
)

LOGGER = logging.getLogger(__name__)

# Gym owns this route (nemo_gym._checkpoint.generation_cut.GENERATION_CUT_ROUTE),
# but nemo_gym is an optional extra. A drifted copy would 404 every cut, which Gym
# treats as "regenerate", so test_generation_cut_route_matches_gym pins it.
GENERATION_CUT_ROUTE = "/ng-control/v1/generation-cut"
# One HTTP frontend's share of a cut, requested by the frontend Gym reached.
GENERATION_CUT_CLAIMS_ROUTE = f"{GENERATION_CUT_ROUTE}/claims"


def register_generation_cut_route(
    app: FastAPI,
    recovery: Callable[[], PrefixRecovery | NoPrefixRecovery],
    peer_urls: Callable[[], Sequence[str]],
) -> None:
    """Serve Gym's generation-cut call from the worker's prefix recovery.

    With several HTTP frontends sharing one listening socket, Gym's call
    reaches one of them, but each call's state lives in the frontend that
    accepted its request. That frontend then asks every other one for its
    claims over their loopback control URLs and merges the answers.

    Args:
        app: The frontend's FastAPI app.
        recovery: Returns the frontend's current ``PrefixRecovery`` or
            ``NoPrefixRecovery``. It is a callable because the server starts
            before ``setup_token_capture`` builds the recovery object.
        peer_urls: Returns the control URLs of the worker's other HTTP
            frontends; empty with a single frontend.
    """

    @app.post(GENERATION_CUT_ROUTE)
    async def checkpoint_generation_cut(
        inventory: dict,
        authorization: str | None = Header(default=None),
    ):
        """Persist one checkpoint's frozen active-call prefixes to TQ."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

        current = _authorized(recovery(), authorization)
        typed_inventory = GenerationCutInventory.model_validate(inventory)
        peers = peer_urls()
        if not peers:
            receipt = await current.run_cut(typed_inventory)
        else:
            frontend_claims = await asyncio.gather(
                current.run_cut_claims(typed_inventory),
                *(
                    _peer_cut_claims(
                        url,
                        inventory,
                        calls=len(typed_inventory.active_prefixes),
                        control_token=current.control_token,
                        timeout_s=current.control_timeout_s,
                    )
                    for url in peers
                ),
            )
            receipt = merge_cut_claims(typed_inventory, frontend_claims)
        return receipt.model_dump(mode="json")

    @app.post(GENERATION_CUT_CLAIMS_ROUTE)
    async def generation_cut_claims(
        inventory: dict,
        authorization: str | None = Header(default=None),
    ):
        """Cut the inventory's calls this frontend serves; ``None`` for the rest."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

        current = _authorized(recovery(), authorization)
        claims = await current.run_cut_claims(
            GenerationCutInventory.model_validate(inventory)
        )
        return [
            None
            if claim is None
            else {"ack": claim.ack.model_dump(mode="json"), "active": claim.active}
            for claim in claims
        ]


def _authorized(
    current: PrefixRecovery | NoPrefixRecovery, authorization: str | None
) -> PrefixRecovery | NoPrefixRecovery:
    """Return ``current`` if the request carries its control bearer."""
    expected = current.control_token
    supplied = (
        authorization.removeprefix("Bearer ")
        if authorization is not None and authorization.startswith("Bearer ")
        else ""
    )
    if expected is None:
        raise HTTPException(
            status_code=404, detail="generation-prefix cuts are disabled"
        )
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid control bearer")
    return current


async def _peer_cut_claims(
    url: str,
    inventory: dict[str, Any],
    *,
    calls: int,
    control_token: str | None,
    timeout_s: float | None,
) -> list[CutClaim | None]:
    """Ask the frontend at ``url`` for its claims on the inventory's calls.

    A frontend that does not answer claims nothing, so its calls regenerate
    after a restore, as when Gym's own cut call fails.
    """
    from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as session:
            async with session.post(
                f"{url}{GENERATION_CUT_CLAIMS_ROUTE}",
                json=inventory,
                headers={"authorization": f"Bearer {control_token}"},
            ) as response:
                response.raise_for_status()
                answers = await response.json()
        claims = [
            None
            if answer is None
            else CutClaim(
                GenerationCutPrefixAck.model_validate(answer["ack"]),
                active=bool(answer["active"]),
            )
            for answer in answers
        ]
        if len(claims) != calls:
            raise ValueError(f"{len(claims)} claims for {calls} inventory calls")
        return claims
    except (aiohttp.ClientError, TimeoutError, KeyError, TypeError, ValueError):
        LOGGER.exception(
            "HTTP frontend %s did not answer the generation cut; its calls "
            "regenerate after a restore",
            url,
        )
        return [None for _ in range(calls)]
