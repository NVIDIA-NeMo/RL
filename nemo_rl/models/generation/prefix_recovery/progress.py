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

"""The backend-specific seam: reading how far each live call has decoded."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationProgress:
    """One call's generated tokens past a cut's start, read at an output boundary.

    ``logprobs`` is the engine's own append-only container; positions up to
    ``start + len(token_ids)`` were published when the tokens were copied.
    """

    token_ids: list[int]
    logprobs: Any
    finish_reason: str | None
    stop_reason: str | int | None


@dataclass(frozen=True)
class ProgressRead:
    """Progress per model call ID, and how long the read held the engine up.

    A call the engine does not hold (not yet submitted, aborted, or not a
    single engine request) is missing from ``progress``.
    """

    progress: Mapping[str, GenerationProgress]
    wait_seconds: float
    hold_seconds: float


class ProgressReader(Protocol):
    """Reads every named call's new tokens at one engine output boundary."""

    def read(
        self, calls: Sequence[tuple[str, int]], *, timeout_s: float | None
    ) -> ProgressRead:
        """Read ``(model_call_id, start)`` calls; raise TimeoutError past ``timeout_s``."""
        ...

    def logprobs(
        self, token_ids: Sequence[int], logprobs: Any, *, start: int
    ) -> list[float]:
        """The sampled log probability of each token, from the engine's container.

        Raises RuntimeError, TypeError or ValueError when they do not line up.
        """
        ...
