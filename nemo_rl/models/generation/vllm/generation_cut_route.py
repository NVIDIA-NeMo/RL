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

import secrets
from collections.abc import Callable

from fastapi import FastAPI, Header, HTTPException

from nemo_rl.models.generation.prefix_recovery import NoPrefixRecovery, PrefixRecovery

# Gym owns this route (nemo_gym._checkpoint.generation_cut.GENERATION_CUT_ROUTE),
# but nemo_gym is an optional extra. A drifted copy would 404 every cut, which Gym
# treats as "regenerate", so test_generation_cut_route_matches_gym pins it.
GENERATION_CUT_ROUTE = "/ng-control/v1/generation-cut"


def register_generation_cut_route(
    app: FastAPI, recovery: Callable[[], PrefixRecovery | NoPrefixRecovery]
) -> None:
    """Serve Gym's generation-cut call from the worker's prefix recovery.

    ``recovery`` returns the worker's current ``PrefixRecovery`` or
    ``NoPrefixRecovery``. It is a callable because the server starts before
    ``setup_token_capture`` builds the recovery object.
    """

    @app.post(GENERATION_CUT_ROUTE)
    async def checkpoint_generation_cut(
        inventory: dict,
        authorization: str | None = Header(default=None),
    ):
        """Persist one checkpoint's frozen active-call prefixes to TQ."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

        current = recovery()
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
        typed_inventory = GenerationCutInventory.model_validate(inventory)
        receipt = await current.run_cut(typed_inventory)
        return receipt.model_dump(mode="json")
