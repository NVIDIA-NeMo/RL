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

"""Async dispatch helpers for local and Ray data-plane clients."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

_Result = TypeVar("_Result")


async def complete_before_cancellation(awaitable: Awaitable[_Result]) -> _Result:
    """Settle an operation before allowing caller cancellation to propagate.

    A dispatched write may still publish after its caller is cancelled. Keep
    the caller's ownership and checkpoint scopes open until the operation has
    settled, including when shutdown cancels the caller more than once. If the
    operation fails during cancellation, consume its exception and preserve
    the caller's original cancellation. Without cancellation, return or raise
    the operation's result normally.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except (asyncio.CancelledError, Exception):
                # Cancellation never transfers ownership back before the
                # operation finishes; its eventual failure is retrieved below.
                pass
        if not task.cancelled():
            task.exception()
        raise


async def call_data_plane(
    client: Any,
    method_name: str,
    *,
    offload_sync: bool = False,
    **kwargs: Any,
) -> Any:
    """Call a local data-plane client or a Ray actor exposing its methods.

    Synchronous offloading is opt-in because it allows the actor event loop to
    issue other calls while this one is running. Callers should enable it only
    when that concurrency is supported or externally serialized.

    Local clients that enable ``async_io_enabled`` use their async I/O methods
    instead. Checkpoint and other control operations retain their existing
    dispatch. The async implementation owns admission and cancellation safety;
    this helper awaits completion within the caller's existing mutation scope.

    Args:
        client: Local ``DataPlaneClient`` or Ray actor handle.
        method_name: Data-plane method to invoke.
        offload_sync: Run a synchronous local implementation in a worker
            thread. Ray methods and enabled async I/O methods ignore this flag.
        **kwargs: Keyword arguments forwarded to the data-plane method.

    Returns:
        The method result after awaiting Ray or coroutine results.
    """
    method = getattr(client, method_name)
    remote = getattr(method, "remote", None)
    if remote is not None:
        return await remote(**kwargs)
    if method_name in (
        "get_data",
        "put_samples",
        "get_samples",
        "list_sample_ids",
        "clear_samples",
    ) and getattr(client, "async_io_enabled", False):
        return await getattr(client, f"async_{method_name}")(**kwargs)
    if offload_sync:
        result = await asyncio.to_thread(method, **kwargs)
    else:
        result = method(**kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result
