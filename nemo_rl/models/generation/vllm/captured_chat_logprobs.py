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

"""Avoid constructing OpenAI logprob objects discarded by terminal capture."""

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, cast

_G_BYPASS_OWNER: ContextVar[tuple[object, asyncio.Task[Any] | None, int] | None] = (
    ContextVar("nemo_rl_captured_chat_logprobs_owner", default=None)
)


class _ChatLogprobsBuilder(Protocol):
    def _create_chat_logprobs(self, *args: Any, **kwargs: Any) -> Any: ...


class CapturedChatLogprobsMixin:
    """Scope a formatting-only bypass to one serving instance and request task.

    Place before vLLM's OpenAIServingChat in the MRO. Engine sampling parameters
    and the raw-array attachment/validation path remain unchanged.
    """

    @contextmanager
    def _bypass_chat_logprobs(self, *, enabled: bool) -> Iterator[None]:
        """Enable only around full-response formatting for admitted raw capture."""
        owner = (
            (self, asyncio.current_task(), threading.get_ident()) if enabled else None
        )
        token = _G_BYPASS_OWNER.set(owner)
        try:
            yield
        finally:
            _G_BYPASS_OWNER.reset(token)

    def _create_chat_logprobs(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate normally, or omit objects that capture will strip after PUT."""
        owner = _G_BYPASS_OWNER.get()
        if (
            owner is not None
            and owner[0] is self
            and owner[2] == threading.get_ident()
            and owner[1] is asyncio.current_task()
        ):
            return None
        return cast(_ChatLogprobsBuilder, super())._create_chat_logprobs(
            *args, **kwargs
        )
