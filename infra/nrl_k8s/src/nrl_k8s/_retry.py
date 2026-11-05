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
"""Retry wrapper for transient Kubernetes API failures.

5xx responses and connection resets are common on busy clusters. Our
apply/get/list calls are side-effect-free or idempotent, so a short
exponential backoff keeps the CLI usable without masking real bugs.

4xx (bad manifest, missing RBAC) is not retried — those are our bugs.
"""

from __future__ import annotations

import socket
from functools import wraps
from typing import Callable, TypeVar

from kubernetes.client.exceptions import ApiException
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from urllib3.exceptions import ProtocolError, ReadTimeoutError

T = TypeVar("T")

_RETRIABLE_STATUSES = frozenset({500, 502, 503, 504})
_RETRIABLE_NETWORK = (
    ReadTimeoutError,
    ProtocolError,
    socket.timeout,
    ConnectionResetError,
    OSError,
)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ApiException):
        return exc.status in _RETRIABLE_STATUSES
    return isinstance(exc, _RETRIABLE_NETWORK)


def with_retries(
    func: Callable[[], T],
    *,
    retries: int = 3,
    max_wait: float = 2.0,
) -> T:
    """Call ``func`` up to ``retries`` times on transient k8s API errors.

    Backs off exponentially (0.5s, 1s, capped at ``max_wait``) between
    attempts. Re-raises the last exception once attempts are exhausted.
    """
    retrying = Retrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=1, min=0.5, max=max_wait),
        reraise=True,
    )
    return retrying(func)


def retry_transient(retries: int = 3, max_wait: float = 2.0):
    """Decorator flavour of :func:`with_retries` for module-level helpers."""

    def wrap(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def inner(*args, **kwargs):
            return with_retries(
                lambda: func(*args, **kwargs), retries=retries, max_wait=max_wait
            )

        return inner

    return wrap


__all__ = ["retry_transient", "with_retries"]
