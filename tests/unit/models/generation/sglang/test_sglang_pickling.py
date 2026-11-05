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

"""Pickling SGLangGeneration into the SingleController actor.

No sglang extra, no Ray runtime and no GPU: the engine handles are picklable
fakes, so these run in the base (unmarked) unit-test shard.
"""

import asyncio
import pickle

import httpx
import pytest

from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.sglang.utils.async_utils import AsyncLoopThread
from nemo_rl.models.generation.sglang.utils.http_utils import HttpClient


class _ExplodingRemote:
    """A Ray actor method the copy must never call."""

    def remote(self, *args, **kwargs):
        raise AssertionError("the client copy must not drive engine teardown")


class _PicklableEngine:
    def __init__(self):
        self.shutdown = _ExplodingRemote()


def _make_driver_generation() -> SGLangGeneration:
    gen = SGLangGeneration.__new__(SGLangGeneration)
    gen._owns_engines = True
    gen._health_monitor = None
    gen.weight_synchronizer = None
    gen._router_actor = None
    gen.rollout_engine_lock = None
    gen.all_engines = [_PicklableEngine(), _PicklableEngine()]
    gen.sglang_cfg = {"sglang_cfg": {"model_path": "Qwen/Qwen3-0.6B"}}
    gen.pause_generation_mode = "retract"
    gen._async_loop = AsyncLoopThread()
    gen._http_client = HttpClient()
    # A live httpx client is loop-bound and does not pickle on its own.
    gen._http_client._client = httpx.AsyncClient()
    return gen


def _shutdown_driver(gen: SGLangGeneration) -> None:
    """Owner teardown without the fake engines, which only exist to be left alone."""
    gen.all_engines = []
    assert gen.shutdown() is True


def test_round_trip_builds_a_client_copy_with_its_own_loop():
    gen = _make_driver_generation()
    try:
        copy = pickle.loads(pickle.dumps(gen))
        try:
            assert copy._owns_engines is False
            assert isinstance(copy._async_loop, AsyncLoopThread)
            assert copy._async_loop is not gen._async_loop
            assert copy._async_loop._thread.is_alive()
            assert copy._http_client._client is None
            assert copy._http_client._max_retries == gen._http_client._max_retries
            assert len(copy.all_engines) == 2
            assert copy.pause_generation_mode == "retract"
            # The driver's instance is untouched by being pickled.
            assert gen._owns_engines is True
            assert gen._http_client._client is not None
        finally:
            assert copy.shutdown() is True
        assert copy._async_loop is None
        assert copy._http_client is None
        # The copy's shutdown never reached the shared engines (they would
        # have raised) and never stopped the driver's loop.
        assert gen._async_loop._thread.is_alive()
    finally:
        _shutdown_driver(gen)


def test_pickling_rejects_a_running_health_monitor():
    gen = _make_driver_generation()
    try:
        gen._health_monitor = object()
        with pytest.raises(NotImplementedError, match="fault tolerance"):
            pickle.dumps(gen)
    finally:
        gen._health_monitor = None
        _shutdown_driver(gen)


def test_http_client_pickles_without_its_loop_bound_client():
    client = HttpClient()
    client._post_actors = ["poster-handle"]
    client._client = httpx.AsyncClient()
    try:
        copy = pickle.loads(pickle.dumps(client))
        assert copy._client is None
        assert copy._post_actors == ["poster-handle"]
        assert copy._max_retries == client._max_retries
        # Local close keeps the shared POST actors.
        asyncio.run(copy.aclose_local())
        assert copy._post_actors == ["poster-handle"]
    finally:
        asyncio.run(client.aclose_local())
