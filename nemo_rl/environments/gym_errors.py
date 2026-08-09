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
"""Picklable stand-ins for aiohttp errors raised inside the NeMo-Gym actor.

``aiohttp.ClientResponseError`` keeps its response ``headers`` as a
``multidict.CIMultiDictProxy``, which cloudpickle cannot serialize. When one
escapes a Ray actor, Ray drops the cause it cannot pickle and hands the caller
a bare ``RayTaskError``, so every Gym HTTP failure -- 400s, 500s, truncated
bodies -- collapses into a single bucket in the SingleController's
per-error-type rollout failure counts, which is what we use to tell one sick
environment server apart from a systemic backend failure.

Only ``ClientResponseError`` is affected. ``ClientOSError``,
``ServerDisconnectedError``, ``ClientPayloadError`` and ``ContentLengthError``
all round-trip through cloudpickle unchanged, so they are deliberately left
alone and keep crossing the boundary as their own, more specific types.
"""

from typing import Any, Optional

from aiohttp import ClientResponseError

# Enough of the body to identify the failure without copying a large payload
# into an exception that crosses the actor boundary.
MAX_BODY_CHARS = 512


def _truncate_body(content: Any) -> Optional[str]:
    """Render a response body as a short string, or None if there is none."""
    if content is None:
        return None
    text = (
        content.decode("utf-8", errors="replace")
        if isinstance(content, (bytes, bytearray))
        else str(content)
    )
    if len(text) > MAX_BODY_CHARS:
        return f"{text[:MAX_BODY_CHARS]}... (truncated from {len(text)} chars)"
    return text


def _rebuild_gym_http_status_error(
    status: Optional[int],
    url: str,
    method: str,
    original_type: str,
    message: str,
    body: Optional[str],
) -> "GymHTTPStatusError":
    """Reconstruct a `GymHTTPStatusError` while unpickling."""
    return GymHTTPStatusError(
        status=status,
        url=url,
        method=method,
        original_type=original_type,
        message=message,
        body=body,
    )


class GymHTTPStatusError(Exception):
    """A Gym HTTP status failure reduced to primitives so Ray can ship it.

    Attributes:
        status: HTTP status code, or None if aiohttp recorded none.
        url: Request URL.
        method: HTTP method.
        original_type: Class name of the aiohttp exception this replaced, kept
            so per-error-type failure counters keep their resolution.
        message: aiohttp's status message.
        body: Truncated response body, when one had already been read.
    """

    def __init__(
        self,
        *,
        status: Optional[int],
        url: str,
        method: str,
        original_type: str,
        message: str,
        body: Optional[str] = None,
    ) -> None:
        self.status = status
        self.url = url
        self.method = method
        self.original_type = original_type
        self.message = message
        self.body = body

        detail = f"{status} {message} for {method} {url} (original {original_type})"
        if body:
            detail = f"{detail}: {body}"
        super().__init__(detail)

    def __reduce__(self) -> tuple[Any, ...]:
        # The constructor is keyword-only, so the default exception reduce --
        # which replays self.args positionally -- cannot rebuild this.
        return (
            _rebuild_gym_http_status_error,
            (
                self.status,
                self.url,
                self.method,
                self.original_type,
                self.message,
                self.body,
            ),
        )


def to_picklable_gym_error(error: BaseException) -> BaseException:
    """Swap an unpicklable aiohttp error for one Ray can carry to the caller.

    Args:
        error: Exception on its way out of the Gym actor.

    Returns:
        A `GymHTTPStatusError` for `ClientResponseError`; otherwise `error`
        unchanged, so callers keep the most specific type available.
    """
    if not isinstance(error, ClientResponseError):
        return error

    request_info = getattr(error, "request_info", None)
    return GymHTTPStatusError(
        status=error.status,
        url=str(getattr(request_info, "real_url", "") or ""),
        method=str(getattr(request_info, "method", "") or ""),
        original_type=type(error).__name__,
        message=str(error.message),
        # Gym's raise_for_status attaches the body it already read; other call
        # sites raise straight from aiohttp and have nothing to attach.
        body=_truncate_body(getattr(error, "response_content", None)),
    )
