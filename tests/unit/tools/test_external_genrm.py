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
import os
import signal
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientPayloadError, web

from tools.external_genrm.genrm_lb import (
    SHUTDOWN_TIMEOUT_SECONDS,
    Backend,
    BackendPool,
    LoadBalancer,
    UpstreamRetryableStatus,
    _read_current_rss_mb,
)

REPO_ROOT = Path(__file__).parents[3]


def test_shutdown_timeout_bounds_watchdog_restart_outage():
    assert 0 < SHUTDOWN_TIMEOUT_SECONDS <= 120


def test_read_current_rss_uses_vmrss_instead_of_process_high_water_mark():
    status = textwrap.dedent(
        """\
        Name:   python
        VmHWM:  8388608 kB
        VmRSS:  315392 kB
        """
    )

    with patch(
        "tools.external_genrm.genrm_lb.Path.read_text",
        return_value=status,
    ):
        assert _read_current_rss_mb() == 308


@pytest.mark.asyncio
async def test_memory_watchdog_requests_graceful_shutdown():
    pool = BackendPool("/tmp", "test")
    pool._running = True

    with (
        patch(
            "tools.external_genrm.genrm_lb._read_current_rss_mb",
            return_value=4097,
        ),
        patch("tools.external_genrm.genrm_lb.os.kill") as kill,
    ):
        await pool._health_check_loop()

    assert pool._running is False
    kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


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


def test_read_registry_skips_bad_line_without_dropping_later_entries(tmp_path):
    registry = tmp_path / ".registry_test"
    registry.write_text(
        "\n".join(
            [
                "good-1 10.0.0.1 8000 123 ready",
                "bad-port 10.0.0.2 not-a-port 124 ready",
                "good-2 10.0.0.3 8002 125 ready",
            ]
        )
    )

    pool = BackendPool(str(tmp_path), "test")

    assert pool._read_registry() == {
        "good-1": ("10.0.0.1", 8000),
        "good-2": ("10.0.0.3", 8002),
    }


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


def test_extract_affinity_key_handles_json_that_is_not_an_object():
    assert LoadBalancer._extract_affinity_key(b"[1, 2]") is None
    assert LoadBalancer._extract_affinity_key(b"null") is None
    assert LoadBalancer._extract_affinity_key(b"123") is None


def test_pick_prefers_affinity_backend_until_it_becomes_a_hotspot():
    pool = BackendPool("/tmp", "test")
    first = Backend("first", "10.0.0.1", 8000)
    second = Backend("second", "10.0.0.2", 8000)
    pool.backends = {first.job_id: first, second.job_id: second}
    affinity_key = LoadBalancer._extract_affinity_key(
        json.dumps({"messages": [{"role": "user", "content": "prompt"}]}).encode()
    )

    preferred = pool.pick(affinity_key=affinity_key)
    assert pool.pick(affinity_key=affinity_key) is preferred

    other = next(
        backend for backend in pool.backends.values() if backend is not preferred
    )
    preferred.inflight = 2 * other.inflight + 11
    assert pool.pick(affinity_key=affinity_key) is other


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


@pytest.mark.asyncio
async def test_stream_failure_after_prepare_does_not_escape_for_retry():
    class FailingStreamContent:
        async def _iterate(self):
            yield b"first chunk"
            raise ClientPayloadError("upstream disconnected")

        def iter_any(self):
            return self._iterate()

    backend = Backend("first", "10.0.0.1", 8000)
    load_balancer = LoadBalancer(BackendPool("/tmp", "test"), 9213)
    upstream_response = MagicMock()
    upstream_response.status = 200
    upstream_response.headers = {"Content-Type": "text/event-stream"}
    upstream_response.content = FailingStreamContent()
    request_context = MagicMock()
    request_context.__aenter__ = AsyncMock(return_value=upstream_response)
    request_context.__aexit__ = AsyncMock(return_value=None)
    proxy_session = MagicMock()
    proxy_session.request.return_value = request_context
    load_balancer._proxy_session = proxy_session

    stream_response = MagicMock(spec=web.StreamResponse)
    stream_response.prepare = AsyncMock()
    stream_response.write = AsyncMock()
    stream_response.write_eof = AsyncMock()
    request = MagicMock(spec=web.Request)

    with patch(
        "tools.external_genrm.genrm_lb.web.StreamResponse",
        return_value=stream_response,
    ):
        result = await load_balancer._proxy_once(
            backend,
            "POST",
            "/v1/responses",
            {},
            b"{}",
            request,
        )

    assert result is stream_response
    stream_response.prepare.assert_awaited_once_with(request)
    stream_response.write.assert_awaited_once_with(b"first chunk")
    stream_response.write_eof.assert_awaited_once()
    assert backend.healthy is False
    assert backend.inflight == 0


@pytest.mark.asyncio
async def test_proxy_forwards_last_upstream_5xx_after_exhausting_backends():
    pool = BackendPool("/tmp", "test")
    first = Backend("first", "10.0.0.1", 8000)
    second = Backend("second", "10.0.0.2", 8000)
    pool.backends = {first.job_id: first, second.job_id: second}
    load_balancer = LoadBalancer(pool, 9213)
    load_balancer._proxy_once = AsyncMock(
        side_effect=UpstreamRetryableStatus(503, b"engine dead", {"X-Request-Id": "1"})
    )
    request = MagicMock(spec=web.Request)
    request.read = AsyncMock(return_value=b"{}")
    request.method = "POST"
    request.path_qs = "/v1/chat/completions"
    request.headers = {}

    response = await load_balancer.handle_proxy(request)

    assert response.status == 503
    assert response.body == b"engine dead"
    assert response.headers["X-Request-Id"] == "1"
    assert load_balancer._proxy_once.await_count == 2
    assert not first.healthy and not second.healthy


@pytest.mark.asyncio
async def test_proxy_returns_503_when_no_backend_is_available():
    load_balancer = LoadBalancer(BackendPool("/tmp", "test"), 9213)
    request = MagicMock(spec=web.Request)
    request.read = AsyncMock(return_value=b"{}")
    request.method = "POST"
    request.path_qs = "/v1/chat/completions"
    request.headers = {}

    response = await load_balancer.handle_proxy(request)

    assert response.status == 503


@pytest.mark.asyncio
async def test_health_reports_backend_counts():
    pool = BackendPool("/tmp", "test")
    healthy = Backend("healthy", "10.0.0.1", 8000)
    sick = Backend("sick", "10.0.0.2", 8000)
    sick.healthy = False
    pool.backends = {healthy.job_id: healthy, sick.job_id: sick}

    response = await LoadBalancer(pool, 9213).handle_health(MagicMock(spec=web.Request))

    assert isinstance(response.body, bytes)
    payload = json.loads(response.body)
    assert payload["status"] == "ok"
    assert payload["healthy_backends"] == 1
    assert payload["total_backends"] == 2


def test_registry_shell_helpers_add_replace_remove(tmp_path):
    script = REPO_ROOT / "tools/external_genrm/genrm_registry.sh"
    program = textwrap.dedent(
        f"""
        set -euo pipefail
        export GENRM_SERVING_DIR={tmp_path}
        export GENRM_GROUP_ID=test
        source {script}
        registry_add job-a 10.0.0.1 8000
        registry_add job-b 10.0.0.2 8001
        echo "count=$(registry_count_ready)"
        registry_add job-a 10.0.0.9 8009
        echo "count=$(registry_count_ready)"
        echo "ready=$(registry_list_ready | tr '\\n' ',')"
        registry_remove job-b
        echo "count=$(registry_count_ready)"
        """
    )
    result = subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "count=2",
        "count=2",
        "ready=10.0.0.2:8001,10.0.0.9:8009,",
        "count=1",
    ]
