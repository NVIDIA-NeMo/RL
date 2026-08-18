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

"""Environment-driven steering messages, injected at the end of a turn."""

from typing import Any, Awaitable, Callable, Optional

# (url_path, json body) -> parsed JSON response from the steering resources server
PostSteering = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class BaseSteeringMessageManager:
    def __init__(
        self, cfg: Optional[dict[str, Any]] = None, post: Optional[PostSteering] = None
    ) -> None:
        self.cfg = cfg or {}
        self._post = post
        self._step = 0

    async def get_steering_messages(
        self, latest_model_output: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Items to append after this turn's model output and tool results."""
        raise NotImplementedError


class NoOpSteeringMessageManager(BaseSteeringMessageManager):
    async def get_steering_messages(
        self, latest_model_output: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return []


class SteeringMessageManager(BaseSteeringMessageManager):
    """Asks the agent's own resources server what to inject next.

    POSTs ``{"latest_model_output": [...], "step": n}`` to ``url_path`` on the
    same resources server the agent seeds and verifies against, carrying the
    session cookies, and appends the returned ``messages``. The endpoint is
    opt-in per environment; no resources server ships one by default.
    """

    async def get_steering_messages(
        self, latest_model_output: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self._step += 1
        assert self._post is not None, "steering requires a bound post callable"
        body = await self._post(
            self.cfg.get("url_path") or "/steering",
            {"latest_model_output": latest_model_output, "step": self._step},
        )
        return list(body.get("messages") or [])
