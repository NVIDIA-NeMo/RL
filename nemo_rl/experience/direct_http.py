# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Direct capture's small HTTP control plane and request-scoped identity."""

import asyncio
import hmac
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from nemo_rl.experience.direct_capture import (
    DirectCapture,
    RolloutCapture,
    current_capture,
)


class RegisterRollout(BaseModel):
    api_key: str = Field(min_length=1, repr=False)
    rollout_id: str = Field(min_length=1)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)


class RolloutKey(BaseModel):
    api_key: str = Field(min_length=1, repr=False)


class DirectCaptureMiddleware:
    """Scope native capture across full HTTP/SSE lifetimes, without body parsing."""

    def __init__(self, app: Any, capture: Callable[[], DirectCapture | None]) -> None:
        self.app = app
        self.capture = capture

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        capture = self.capture()
        if (
            capture is None
            or scope["type"] != "http"
            or scope["path"] not in {"/v1/chat/completions", "/v1/responses"}
        ):
            await self.app(scope, receive, send)
            return
        authorization = (
            dict(scope["headers"]).get(b"authorization", b"").decode("latin1")
        )
        try:
            context = capture.enter(authorization.removeprefix("Bearer "))
        except KeyError:
            await JSONResponse({"error": "unknown rollout API key"}, status_code=401)(
                scope, receive, send
            )
            return
        except ValueError:
            await JSONResponse({"error": "rollout is closing"}, status_code=409)(
                scope, receive, send
            )
            return
        token = current_capture.set(context)
        try:
            await self.app(scope, receive, send)
        finally:
            current_capture.reset(token)
            capture.leave(context)


def validate_direct_request(request: Any, rollout: RolloutCapture) -> None:
    """Default omitted sampling to the rollout profile and reject off-policy calls."""
    if request.n not in (None, 1) or request.use_beam_search:
        raise HTTPException(400, "direct capture requires n=1")
    if request.ng_capture is not None or request.required_prefix_token_ids is not None:
        raise HTTPException(
            400, "direct capture does not accept token-prefix overrides"
        )
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            part.get("type") != "text" for part in content
        ):
            raise HTTPException(
                400, "direct capture currently supports text and tools only"
            )
    if request.temperature is None:
        request.temperature = rollout.temperature
    if request.top_p is None:
        request.top_p = rollout.top_p
    if (request.temperature, request.top_p) != (rollout.temperature, rollout.top_p):
        raise HTTPException(
            400, "sampling parameters differ from this rollout's registered profile"
        )


def install_direct_control(
    app: FastAPI,
    *,
    capture: Callable[[], DirectCapture | None],
    auth_token: Callable[[], str],
    timeout: Callable[[], float],
) -> None:
    app.add_middleware(DirectCaptureMiddleware, capture=capture)

    def authorized(request: Request) -> DirectCapture:
        owner = capture()
        if owner is None:
            raise HTTPException(503, "direct capture is disabled")
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {auth_token()}"
        ):
            raise HTTPException(401, "invalid control credential")
        return owner

    @app.post("/nrl/capture/register")
    async def register(body: RegisterRollout, request: Request) -> dict[str, bool]:
        owner = authorized(request)
        owner.register(
            body.api_key,
            body.rollout_id,
            temperature=body.temperature,
            top_p=body.top_p,
        )
        return {"registered": True}

    @app.post("/nrl/capture/finalize")
    async def finalize(body: RolloutKey, request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(
            authorized(request).finalize, body.api_key, timeout=timeout()
        )

    @app.post("/nrl/capture/release")
    async def release(body: RolloutKey, request: Request) -> dict[str, bool]:
        authorized(request).release(body.api_key)
        return {"released": True}

    @app.post("/nrl/capture/discard")
    async def discard(body: RolloutKey, request: Request) -> dict[str, bool]:
        await asyncio.to_thread(
            authorized(request).discard, body.api_key, timeout=timeout()
        )
        return {"discarded": True}
