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
from typing import Any


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

    Args:
        client: Local ``DataPlaneClient`` or Ray actor handle.
        method_name: Data-plane method to invoke.
        offload_sync: Run a synchronous local implementation in a worker
            thread. Ray methods are already asynchronous and ignore this flag.
        **kwargs: Keyword arguments forwarded to the data-plane method.

    Returns:
        The method result after awaiting Ray or coroutine results.
    """
    method = getattr(client, method_name)
    remote = getattr(method, "remote", None)
    if remote is not None:
        return await remote(**kwargs)
    if offload_sync:
        result = await asyncio.to_thread(method, **kwargs)
    else:
        result = method(**kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result
