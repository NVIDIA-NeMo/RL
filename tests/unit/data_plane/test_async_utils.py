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

"""Tests for local and Ray-style async data-plane dispatch."""

import asyncio
import threading

from nemo_rl.data_plane.async_utils import call_data_plane


class _LocalClient:
    def thread_id(self) -> int:
        return threading.get_ident()

    async def async_value(self, *, value: int) -> int:
        return value


class _RemoteMethod:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def remote(self, *, value: int) -> int:
        self.calls.append(value)
        return value


class _RemoteClient:
    def __init__(self) -> None:
        self.value = _RemoteMethod()


def test_sync_call_stays_inline_by_default() -> None:
    caller_thread_id = threading.get_ident()

    result = asyncio.run(call_data_plane(_LocalClient(), "thread_id"))

    assert result == caller_thread_id


def test_sync_call_can_be_offloaded() -> None:
    caller_thread_id = threading.get_ident()

    result = asyncio.run(
        call_data_plane(_LocalClient(), "thread_id", offload_sync=True)
    )

    assert result != caller_thread_id


def test_local_coroutine_result_is_awaited() -> None:
    result = asyncio.run(call_data_plane(_LocalClient(), "async_value", value=7))

    assert result == 7


def test_ray_style_remote_result_is_awaited() -> None:
    client = _RemoteClient()

    result = asyncio.run(call_data_plane(client, "value", value=11))

    assert result == 11
    assert client.value.calls == [11]
