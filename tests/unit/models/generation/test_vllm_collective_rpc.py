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

"""Tests for vLLM collective-RPC result resolution."""

import asyncio
import concurrent.futures

from nemo_rl.models.generation.vllm.collective_rpc import resolve_collective_rpc_result


def test_resolve_collective_rpc_result_passes_through_plain_values():
    assert asyncio.run(resolve_collective_rpc_result(7)) == 7
    assert asyncio.run(resolve_collective_rpc_result("x")) == "x"


def test_resolve_collective_rpc_result_unwraps_awaitables():
    async def make() -> int:
        async def inner() -> int:
            return 42

        # An awaitable that itself resolves to another awaitable.
        return inner()

    assert asyncio.run(resolve_collective_rpc_result(make())) == 42


def test_resolve_collective_rpc_result_unwraps_concurrent_future():
    async def run():
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result("done")
        return await resolve_collective_rpc_result(future)

    assert asyncio.run(run()) == "done"


def test_resolve_collective_rpc_result_preserves_list_and_tuple_shape():
    async def run():
        async def coro(v):
            return v

        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result("f")
        # Nested mix of awaitables, futures, and plain values; list outer,
        # tuple inner — shapes must be preserved.
        nested = [coro(1), (future, coro(2)), 3]
        return await resolve_collective_rpc_result(nested)

    result = asyncio.run(run())
    assert result == [1, ("f", 2), 3]
    assert isinstance(result, list)
    assert isinstance(result[1], tuple)
