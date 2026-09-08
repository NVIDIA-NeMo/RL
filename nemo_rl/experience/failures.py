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

"""Rollout failure taxonomy for the SingleController path.

A failed rollout attempt is one of exactly two things, and the distinction drives
the whole retry policy:

  - **Infra** (:class:`RolloutInfraFailure`) — the prompt is fine, the fleet is not.
    A dead generation shard, a timeout, a dropped connection. Re-dispatching the same
    prompt is expected to succeed because the retry lands on a different shard.
  - **Data** (:class:`RolloutDataFailure`) — deterministic and prompt-specific. A prompt
    longer than the engine's ``max_model_len``, non-contiguous tokens after tokenization.
    Another shard fails the same way, so the retry budget is small and exhausting it is
    reported rather than absorbed.

:func:`classify_rollout_failure` maps an arbitrary exception onto that split. Anything
not recognized as infrastructure is treated as data — an unrecognized exception is more
likely a real bug than a transient blip, and the data path surfaces it loudly instead of
retrying it into silence.

**Classify where the information still exists.** An exception that crosses a Ray actor
boundary arrives stripped: Ray pickles the cause, and anything unpicklable (aiohttp's
``CIMultiDictProxy`` headers, for one) is replaced by a bare error carrying neither type
nor ``.status``. Driver-side classification is then guessing. So failures raised inside
an actor are converted to a picklable member of this taxonomy *before* they cross —
:func:`http_status_is_infra` exists to keep that decision identical on both sides. See
``_typed_gym_failure`` in ``nemo_rl/environments/nemo_gym.py``.
"""

from __future__ import annotations

import enum
from typing import Final, Optional

import ray.exceptions

# aiohttp is not a declared NeMo-RL dependency -- it arrives transitively with ray and
# vllm. It is present in every supported environment, but the import is guarded so this
# module stays usable if that ever stops being true; _INFRA_TYPE_NAMES below is the
# fallback.
_AIOHTTP_INFRA_TYPES: tuple[type[BaseException], ...]
_CLIENT_RESPONSE_ERROR: Optional[type[BaseException]]
try:
    from aiohttp import (
        ClientConnectionError as _ClientConnectionError,
    )
    from aiohttp import (
        ClientPayloadError as _ClientPayloadError,
    )
    from aiohttp import (
        ClientResponseError as _ClientResponseError,
    )
except ImportError:  # pragma: no cover - aiohttp ships with ray in supported images
    _AIOHTTP_INFRA_TYPES = ()
    _CLIENT_RESPONSE_ERROR = None
else:
    # ClientConnectionError is the transport-failure root: it covers ClientOSError,
    # ClientConnectorError, ServerDisconnectedError and ServerTimeoutError. Deliberately
    # NOT the broader ClientError, whose other branch (ClientResponseError) carries HTTP
    # status codes and is classified by status instead -- see _http_status.
    _AIOHTTP_INFRA_TYPES = (_ClientConnectionError, _ClientPayloadError)
    _CLIENT_RESPONSE_ERROR = _ClientResponseError

# Maximum ``__cause__`` links followed by :func:`classify_rollout_failure`. Bounded so a
# self-referential or pathologically deep chain cannot spin.
_MAX_CAUSE_DEPTH: Final[int] = 8

# Sub-500 HTTP statuses that still mean "try again", not "this prompt is bad".
_RETRIABLE_HTTP_STATUSES: Final[frozenset[int]] = frozenset({408, 429})


class RolloutFailure(Exception):
    """Base class for failures that terminate a single rollout attempt."""


class RolloutInfraFailure(RolloutFailure):
    """The generation or environment infrastructure could not serve this rollout.

    The prompt is not implicated. Re-dispatching it is expected to succeed once a
    healthy shard is selected.
    """


class RolloutTimeout(RolloutInfraFailure):
    """A rollout, generation turn, or environment step exceeded its deadline."""


class GenerationUnavailable(RolloutInfraFailure):
    """The selected generation shard could not serve the request."""


class NoHealthyShards(RolloutInfraFailure):
    """No generation shard is currently eligible to serve traffic."""


class GymTransportError(RolloutInfraFailure):
    """NeMo-Gym failed at the transport layer rather than returning a rollout."""


class RolloutDataFailure(RolloutFailure):
    """This prompt cannot be rolled out, and another shard would fail identically.

    Usually a configuration problem — most often ``policy.max_total_sequence_length``
    exceeding the generation engine's ``max_model_len``.
    """


class RolloutRedispatchExhausted(RuntimeError):
    """A prompt exhausted its infrastructure retry budget.

    Deliberately not a :class:`RolloutFailure`: it is terminal for the run rather than
    for one attempt, so the per-attempt retry loop must not catch it. Reaching it means
    the prompt failed across repeated shard selections, which indicates fleet-wide
    failure rather than a bad prompt.
    """


class RolloutStall(RuntimeError):
    """Rollouts are in flight but none has committed within the watchdog deadline."""


class FailureClass(str, enum.Enum):
    """Retry policy bucket for a failed rollout attempt."""

    INFRA = "infra"
    DATA = "data"


# Exception types that always indicate infrastructure failure. Ray's actor/node/RPC
# errors mean a worker or its host went away; the stdlib entries cover the timeout and
# connection paths the rollout code raises directly.
_INFRA_TYPES: Final[tuple[type[BaseException], ...]] = (
    RolloutInfraFailure,
    TimeoutError,  # asyncio.TimeoutError is an alias for this on Python 3.11+
    ConnectionError,
    ray.exceptions.RayActorError,  # parent of ActorDiedError
    ray.exceptions.ActorUnavailableError,
    ray.exceptions.ActorUnschedulableError,
    ray.exceptions.TaskUnschedulableError,
    ray.exceptions.WorkerCrashedError,
    ray.exceptions.NodeDiedError,
    ray.exceptions.LocalRayletDiedError,
    ray.exceptions.OwnerDiedError,
    ray.exceptions.RpcError,
    ray.exceptions.GetTimeoutError,
    ray.exceptions.ObjectFetchTimedOutError,
    # Cluster-resource and object-store failures. None of these say anything about the
    # prompt, and all of them were previously falling through to DATA -- which meant a
    # node running out of memory got two attempts and killed the run, while the bounded,
    # backed-off infra budget built for exactly that sat unused.
    # ObjectLostError is the parent of OwnerDiedError/ObjectFetchTimedOutError above;
    # both are kept listed because being explicit about what we have considered is worth
    # more here than a shorter tuple.
    ray.exceptions.ObjectLostError,
    ray.exceptions.OutOfMemoryError,
    ray.exceptions.ObjectStoreFullError,
    ray.exceptions.RaySystemError,
    *_AIOHTTP_INFRA_TYPES,
)

# Fallback for environments without aiohttp, matched against every class name in the
# exception's MRO. These are the transport errors NeMo-Gym surfaces when a vLLM endpoint
# dies mid-request or refuses a connection. Note ``ClientOSError`` derives from
# ``OSError`` but *not* from ``ConnectionError``, so the isinstance table above would
# not catch it on its own. ``ClientError`` and ``ClientResponseError`` are deliberately
# absent: they cover HTTP responses, which _http_status classifies by status code.
_INFRA_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ClientConnectionError",
        "ClientConnectorError",
        "ClientOSError",
        "ClientPayloadError",
        "ServerConnectionError",
        "ServerDisconnectedError",
        "ServerTimeoutError",
    }
)


def http_status_is_infra(status: int) -> bool:
    """Whether an HTTP status means the endpoint is unwell rather than the request bad.

    5xx and the retriable 4xx are infrastructure; any other 4xx describes the request
    itself, so another shard would answer it identically.

    Public because NeMo-Gym has to make this same call on the *raising* side of a Ray
    actor boundary (see ``nemo_rl/environments/nemo_gym.py``), and the two copies of the
    decision must not drift apart.
    """
    return status >= 500 or status in _RETRIABLE_HTTP_STATUSES


def _http_status(exc: BaseException) -> Optional[int]:
    """Return the HTTP status carried by an aiohttp response error, if this is one."""
    if _CLIENT_RESPONSE_ERROR is not None:
        if not isinstance(exc, _CLIENT_RESPONSE_ERROR):
            return None
    elif not any(
        cls.__name__ == "ClientResponseError" for cls in type(exc).__mro__
    ):  # pragma: no cover - only reachable without aiohttp
        return None
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def _is_infra(exc: BaseException) -> bool:
    """Return whether this single exception (ignoring its cause chain) is infra."""
    if isinstance(exc, RolloutDataFailure):
        return False

    # An HTTP error is classified by status, not by type. A 4xx from the generation
    # engine describes the request -- "This model's maximum context length is ..." comes
    # back as a 400 -- so retrying it on another shard would fail identically. 5xx and
    # the retriable 4xx mean the endpoint is unwell, which another shard can serve.
    status = _http_status(exc)
    if status is not None:
        return http_status_is_infra(status)

    if isinstance(exc, _INFRA_TYPES):
        return True
    return any(cls.__name__ in _INFRA_TYPE_NAMES for cls in type(exc).__mro__)


def classify_rollout_failure(exc: BaseException) -> FailureClass:
    """Bucket a rollout exception into ``INFRA`` or ``DATA``.

    An explicit :class:`RolloutDataFailure` always wins, so callers that know a failure
    is prompt-specific can say so and not have it re-read as infrastructure. Otherwise
    the exception and its ``__cause__`` chain are checked against the infrastructure
    table; anything unrecognized is ``DATA`` so that unexpected exceptions fail loudly
    instead of being retried into silence.

    Args:
        exc: The exception raised by a rollout attempt.

    Returns:
        The :class:`FailureClass` governing this failure's retry budget.
    """
    if isinstance(exc, RolloutDataFailure):
        return FailureClass.DATA

    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_CAUSE_DEPTH):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if _is_infra(current):
            return FailureClass.INFRA
        current = current.__cause__

    return FailureClass.DATA
