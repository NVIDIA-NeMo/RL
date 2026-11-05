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

import asyncio
import concurrent.futures
import inspect
from typing import Any

import ray


async def resolve_collective_rpc_result(result: Any) -> Any:
    """Recursively resolve a vLLM collective-RPC result."""
    while inspect.isawaitable(result):
        result = await result
    if isinstance(result, concurrent.futures.Future):
        return await resolve_collective_rpc_result(await asyncio.wrap_future(result))
    if isinstance(result, ray.ObjectRef):
        return await asyncio.to_thread(ray.get, result)
    if isinstance(result, list | tuple):
        items = [await resolve_collective_rpc_result(item) for item in result]
        return tuple(items) if isinstance(result, tuple) else items
    return result
