# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for SGLang utilities.

These tests verify that the SGLang utilities work as expected.
"""

import pytest

from nemo_rl.models.generation.sglang.utils import AsyncLoopThread


def test_async_loop_thread_run_returns_result():
    loop_thread = AsyncLoopThread()

    async def sample():
        return 42

    try:
        assert loop_thread.run(sample()) == 42
    finally:
        loop_thread.shutdown()


def test_async_loop_thread_run_when_stopped_raises():
    loop_thread = AsyncLoopThread()
    loop_thread.shutdown()

    async def sample():
        return 1

    with pytest.raises(RuntimeError, match="Event loop is not running"):
        coro = sample()
        try:
            loop_thread.run(coro)
        finally:
            coro.close()
