# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from tools.external_genrm.genrm_lb import (
    Backend,
    BackendPool,
    LoadBalancer,
    UpstreamRetryableStatus,
)


def test_backend_pool_reads_only_ready_registry_entries(tmp_path):
    registry = tmp_path / ".registry_test"
    registry.write_text(
        "\n".join(
            [
                "ready-backend 10.0.0.1 8000 123 ready",
                "starting-backend 10.0.0.2 8001 124 starting",
                "malformed",
            ]
        )
    )

    pool = BackendPool(str(tmp_path), "test")

    assert pool._read_registry() == {"ready-backend": ("10.0.0.1", 8000)}


def test_backend_pool_picks_least_loaded_healthy_backend():
    pool = BackendPool("/tmp", "test")
    first = Backend("first", "10.0.0.1", 8000)
    second = Backend("second", "10.0.0.2", 8000)
    first.inflight = 4
    second.inflight = 1
    pool.backends = {first.job_id: first, second.job_id: second}

    assert pool.pick() is second
    assert pool.pick(exclude={"second"}) is first

    first.healthy = False
    assert pool.pick(exclude={"second"}) is None


def test_affinity_key_is_stable_and_ignores_invalid_json():
    body = json.dumps({"messages": [{"role": "user", "content": "prompt"}]}).encode()

    assert LoadBalancer._extract_affinity_key(body) == (
        LoadBalancer._extract_affinity_key(body)
    )
    assert LoadBalancer._extract_affinity_key(b"not-json") is None


@pytest.mark.asyncio
async def test_proxy_retries_a_5xx_on_another_backend():
    pool = BackendPool("/tmp", "test")
    first = Backend("first", "10.0.0.1", 8000)
    second = Backend("second", "10.0.0.2", 8000)
    pool.backends = {first.job_id: first, second.job_id: second}
    load_balancer = LoadBalancer(pool, 9213)

    expected_response = web.Response(status=200, body=b"ok")
    load_balancer._proxy_once = AsyncMock(
        side_effect=[
            UpstreamRetryableStatus(500, b"engine failed", {}),
            expected_response,
        ]
    )
    request = MagicMock(spec=web.Request)
    request.read = AsyncMock(return_value=b"{}")
    request.method = "POST"
    request.path_qs = "/v1/chat/completions"
    request.headers = {}

    response = await load_balancer.handle_proxy(request)

    assert response is expected_response
    assert load_balancer._proxy_once.await_count == 2
    assert first.healthy is False
    assert second.healthy is True
