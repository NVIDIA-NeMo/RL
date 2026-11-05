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

import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientPayloadError, web

from nemo_rl.environments.nemo_gym import (
    ExternalServiceReadinessTargetConfig,
    _probe_external_service,
)
from tools.external_gym_vllm.vllm_pool_lb import (
    SHUTDOWN_TIMEOUT_SECONDS,
    Backend,
    BackendPool,
    LoadBalancer,
    UpstreamRetryableStatus,
    _read_current_rss_mb,
)

REPO_ROOT = Path(__file__).parents[3]


def test_serve_wrapper_loads_patches_without_importing_nemo_rl_package():
    script = REPO_ROOT / "tools/external_gym_vllm/serve_vllm_on_ray.py"
    program = textwrap.dedent(
        f"""
        import runpy
        import sys
        import types

        sys.modules["ray"] = types.ModuleType("ray")
        namespace = runpy.run_path({str(script)!r})
        namespace["_load_apply_vllm_patches"]()
        leaked = [m for m in sys.modules if m.partition(".")[0] == "nemo_rl"]
        assert not leaked, leaked
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_vllm_patches_module_never_imports_nemo_rl():
    """serve_vllm_on_ray.py runs patches.py by path, then calls _apply_vllm_patches.

    Both happen in serving containers without NeMo RL, so an import of nemo_rl at
    ANY level -- including lazily inside a patch function -- breaks them. Relative
    imports break too: a module run by path has no parent package.
    """
    path = REPO_ROOT / "nemo_rl/models/generation/vllm/patches.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ["." * node.level + (node.module or "")]
        else:
            continue
        for module in modules:
            assert module.split(".")[0] not in ("", "nemo_rl"), (
                f"patches.py:{node.lineno} imports {module!r}; serving containers "
                "load this module by path without NeMo RL installed."
            )


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
        "tools.external_gym_vllm.vllm_pool_lb.Path.read_text",
        return_value=status,
    ):
        assert _read_current_rss_mb() == 308


@pytest.mark.asyncio
async def test_memory_watchdog_requests_graceful_shutdown():
    pool = BackendPool("/tmp", "test")
    pool._running = True

    with (
        patch(
            "tools.external_gym_vllm.vllm_pool_lb._read_current_rss_mb",
            return_value=4097,
        ),
        patch("tools.external_gym_vllm.vllm_pool_lb.os.kill") as kill,
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
    assert first.healthy is True
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
        "tools.external_gym_vllm.vllm_pool_lb.web.StreamResponse",
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
async def test_proxy_drops_stale_length_after_upstream_decompression():
    backend = Backend("first", "10.0.0.1", 8000)
    load_balancer = LoadBalancer(BackendPool("/tmp", "test"), 9213)
    upstream_response = MagicMock()
    upstream_response.status = 200
    upstream_response.headers = {
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",
        "Content-Length": "3",
        "X-Request-Id": "request-1",
    }
    upstream_response.read = AsyncMock(return_value=b"decompressed")
    request_context = MagicMock()
    request_context.__aenter__ = AsyncMock(return_value=upstream_response)
    request_context.__aexit__ = AsyncMock(return_value=None)
    proxy_session = MagicMock()
    proxy_session.request.return_value = request_context
    load_balancer._proxy_session = proxy_session
    request = MagicMock(spec=web.Request)
    expected_response = MagicMock(spec=web.Response)

    with patch(
        "tools.external_gym_vllm.vllm_pool_lb.web.Response",
        return_value=expected_response,
    ) as response_class:
        result = await load_balancer._proxy_once(
            backend,
            "POST",
            "/v1/responses",
            {},
            b"{}",
            request,
        )

    assert result is expected_response
    assert response_class.call_args.kwargs["headers"] == {
        "Content-Type": "application/json",
        "X-Request-Id": "request-1",
    }
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
    assert first.healthy and second.healthy


def test_load_balancer_accepts_payloads_larger_than_aiohttp_default():
    app = LoadBalancer(BackendPool("/tmp", "test"), 9213).make_app()

    assert app._client_max_size == 0


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("healthy", "override", "expected_backends", "ready"),
    [
        (1, {}, 2, False),
        (2, {}, 2, True),
        # expected_backends=1 so only the bool guard rejects True (== 1).
        (2, {"healthy_backends": True}, 1, False),
        (2, {"total_backends": 1}, 2, False),
        (2, {"status": "no_healthy_backends"}, 2, False),
    ],
    ids=["one-of-two", "two-of-two", "bool-count", "short-total", "status-not-ok"],
)
async def test_readiness_probe_reads_load_balancer_health(
    healthy, override, expected_backends, ready
):
    """Tie the Gym readiness probe to the reply this load balancer really sends."""
    pool = BackendPool("/tmp", "test")
    for i in range(2):
        backend = Backend(f"job{i}", f"10.0.0.{i}", 8000)
        backend.healthy = i < healthy
        pool.backends[backend.job_id] = backend
    health = await LoadBalancer(pool, 9213).handle_health(MagicMock(spec=web.Request))

    probe_response = MagicMock(status=health.status)
    probe_response.read.return_value = json.dumps(
        {**json.loads(health.body), **override}
    ).encode()
    probe_response.__enter__.return_value = probe_response
    service = ExternalServiceReadinessTargetConfig(
        name="GENRM",
        url="http://10.0.0.1:9213/health",
        expected_backends=expected_backends,
    )
    with patch(
        "nemo_rl.environments.nemo_gym.urllib.request.urlopen",
        return_value=probe_response,
    ):
        problem = _probe_external_service(service, request_timeout_seconds=1)

    assert (problem is None) == ready


def test_registry_shell_helpers_add_replace_remove(tmp_path):
    script = REPO_ROOT / "tools/external_gym_vllm/vllm_backend_registry.sh"
    program = textwrap.dedent(
        f"""
        set -euo pipefail
        export EXTERNAL_VLLM_STATE_DIR={tmp_path}
        export EXTERNAL_VLLM_GROUP_ID=test
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


def test_load_balancer_watchdog_forwards_term_to_child(tmp_path):
    watchdog = REPO_ROOT / "tools/external_gym_vllm/lb_watchdog.sh"
    fake_python = tmp_path / "fake-python"
    child_started = tmp_path / "child-started"
    child_stopped = tmp_path / "child-stopped"
    fake_python.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            touch {child_started}
            trap 'touch {child_stopped}; exit 0' TERM INT
            while true; do sleep 0.1; done
            """
        )
    )
    fake_python.chmod(0o755)
    process = subprocess.Popen(
        ["bash", str(watchdog), "9213", str(tmp_path), "test"],
        env={"PATH": os.environ["PATH"], "PYTHON": str(fake_python)},
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + 5
        while not child_started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_started.exists()

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        assert child_stopped.exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_launcher_requires_a_heterogeneous_job():
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"

    result = subprocess.run(
        ["bash", str(script)],
        env={"PATH": os.environ["PATH"], "SLURM_JOB_ID": "123"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "This script requires a Slurm heterogeneous job" in result.stderr


def test_launcher_rejects_more_than_two_hetgroups():
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": "123",
        "SLURM_HET_SIZE": "3",
        "SLURM_JOB_NODELIST_HET_GROUP_0": "ray[01-02]",
        "SLURM_JOB_NODELIST_HET_GROUP_1": "genrm[01-02]",
        "SLURM_JOB_ACCOUNT": "account",
        "SLURM_JOB_PARTITION": "partition",
        "SLURM_SUBMIT_DIR": "/tmp",
        "BASE_LOG_DIR": "/lustre/logs",
        "CONTAINER": "training.sqsh",
        "MOUNTS": "/lustre:/lustre",
        "COMMAND": "run __GENRM_BASE_URL__",
        "EXTERNAL_VLLM_POOLS": "GENRM",
        "EXTERNAL_VLLM_TOOLS_DIR_HOST": "/lustre/tools",
    }

    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Expected exactly two Slurm hetgroups, got 3" in result.stderr


def test_launcher_requires_nl2bash_placeholder_when_pool_is_enabled():
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": "123",
        "SLURM_HET_SIZE": "2",
        "SLURM_JOB_NODELIST_HET_GROUP_0": "ray[01-02]",
        "SLURM_JOB_NODELIST_HET_GROUP_1": "judge[01-02]",
        "SLURM_JOB_ACCOUNT": "account",
        "SLURM_JOB_PARTITION": "partition",
        "SLURM_SUBMIT_DIR": str(REPO_ROOT),
        "BASE_LOG_DIR": "/lustre/logs",
        "CONTAINER": "training.sqsh",
        "MOUNTS": "/lustre:/lustre",
        "COMMAND": "run __GENRM_BASE_URL__",
        "EXTERNAL_VLLM_POOLS": "GENRM NL2BASH",
        "EXTERNAL_VLLM_TOOLS_DIR_HOST": str(REPO_ROOT / "tools/external_gym_vllm"),
        "GENRM_CONTAINER": "genrm.sqsh",
        "GENRM_MODEL": "model-id",
        "GENRM_VLLM_PYTHON": "/opt/python",
        "GENRM_REPLICAS": "1",
        "GENRM_TENSOR_PARALLEL_SIZE": "4",
        "GENRM_LB_PORT": "9213",
        "GENRM_URL_PLACEHOLDER": "__GENRM_BASE_URL__",
        "NL2BASH_CONTAINER": "judge.sqsh",
        "NL2BASH_MODEL": "judge-model-id",
        "NL2BASH_VLLM_PYTHON": "/opt/python",
        "NL2BASH_REPLICAS": "4",
        "NL2BASH_TENSOR_PARALLEL_SIZE": "4",
        "NL2BASH_LB_PORT": "9214",
        "NL2BASH_URL_PLACEHOLDER": "__NL2BASH_BASE_URL__",
        "RAY_SUB": str(REPO_ROOT / "ray.sub"),
    }

    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Driver command is missing __NL2BASH_BASE_URL__" in result.stderr


def test_launcher_rejects_duplicate_url_placeholders():
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": "123",
        "SLURM_HET_SIZE": "2",
        "SLURM_JOB_NODELIST_HET_GROUP_0": "ray[01-02]",
        "SLURM_JOB_NODELIST_HET_GROUP_1": "judge[01-02]",
        "SLURM_JOB_ACCOUNT": "account",
        "SLURM_JOB_PARTITION": "partition",
        "SLURM_SUBMIT_DIR": str(REPO_ROOT),
        "BASE_LOG_DIR": "/lustre/logs",
        "CONTAINER": "training.sqsh",
        "MOUNTS": "/lustre:/lustre",
        "COMMAND": "run __SHARED_BASE_URL__",
        "EXTERNAL_VLLM_POOLS": "GENRM NL2BASH",
        "EXTERNAL_VLLM_TOOLS_DIR_HOST": str(REPO_ROOT / "tools/external_gym_vllm"),
        "GENRM_CONTAINER": "genrm.sqsh",
        "GENRM_MODEL": "model-id",
        "GENRM_VLLM_PYTHON": "/opt/python",
        "GENRM_REPLICAS": "1",
        "GENRM_TENSOR_PARALLEL_SIZE": "4",
        "GENRM_LB_PORT": "9213",
        "GENRM_URL_PLACEHOLDER": "__SHARED_BASE_URL__",
        "NL2BASH_CONTAINER": "judge.sqsh",
        "NL2BASH_MODEL": "judge-model-id",
        "NL2BASH_VLLM_PYTHON": "/opt/python",
        "NL2BASH_REPLICAS": "4",
        "NL2BASH_TENSOR_PARALLEL_SIZE": "4",
        "NL2BASH_LB_PORT": "9214",
        "NL2BASH_URL_PLACEHOLDER": "__SHARED_BASE_URL__",
        "RAY_SUB": str(REPO_ROOT / "ray.sub"),
    }

    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Multiple pools use URL placeholder __SHARED_BASE_URL__" in result.stderr


def test_launcher_requires_readiness_placeholder():
    """Without the token the gate has no safe place to go, so refuse to start."""
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": "123",
        "SLURM_HET_SIZE": "2",
        "SLURM_JOB_NODELIST_HET_GROUP_0": "ray[01-02]",
        "SLURM_JOB_NODELIST_HET_GROUP_1": "genrm01",
        "SLURM_JOB_ACCOUNT": "account",
        "SLURM_JOB_PARTITION": "partition",
        "SLURM_SUBMIT_DIR": str(REPO_ROOT),
        "BASE_LOG_DIR": "/lustre/logs",
        "CONTAINER": "training.sqsh",
        "MOUNTS": "/lustre:/lustre",
        # A trailing pipe is exactly where an appended override would be lost.
        "COMMAND": "run __GENRM_BASE_URL__ 2>&1 | tee run.log",
        "EXTERNAL_VLLM_POOLS": "GENRM",
        "EXTERNAL_VLLM_TOOLS_DIR_HOST": str(REPO_ROOT / "tools/external_gym_vllm"),
        "GENRM_CONTAINER": "genrm.sqsh",
        "GENRM_MODEL": "model-id",
        "GENRM_VLLM_PYTHON": "/opt/python",
        "GENRM_REPLICAS": "1",
        "GENRM_TENSOR_PARALLEL_SIZE": "4",
        "GENRM_LB_PORT": "9213",
        "GENRM_URL_PLACEHOLDER": "__GENRM_BASE_URL__",
        "RAY_SUB": str(REPO_ROOT / "ray.sub"),
    }

    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Driver command is missing __EXTERNAL_SERVICE_READINESS__" in result.stderr


def test_launcher_routes_generic_pools_to_explicit_hetgroups():
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
    source = script.read_text()

    srun_blocks = []
    current_block = []
    for line in source.splitlines():
        if line.lstrip() == "srun \\":
            current_block = [line]
        elif current_block:
            current_block.append(line)
            if line.rstrip().endswith("&"):
                srun_blocks.append("\n".join(current_block))
                current_block = []

    assert len(srun_blocks) == 2
    replica_launch = next(block for block in srun_blocks if "VLLM_SERVER_BODY" in block)
    lb_launch = next(
        block
        for block in srun_blocks
        if "lb_watchdog.sh" in block and "--output=" in block
    )

    assert '"${het_ext[@]}"' in replica_launch
    assert '-A "${SLURM_JOB_ACCOUNT}"' not in replica_launch
    assert '-p "${SLURM_JOB_PARTITION}"' not in replica_launch
    assert '"${het_ray[@]}"' in lb_launch
    # Inline, those select the two hetgroups; a serve job has a single component.
    assert "het_ray=(--het-group=0)" in source
    assert "het_ext=(--het-group=1)" in source
    assert '-A "${SLURM_JOB_ACCOUNT}"' in lb_launch
    assert '-p "${SLURM_JOB_PARTITION}"' in lb_launch

    assert "preflight" not in source.lower()
    assert "import ray, vllm" not in source
    assert "import aiohttp" not in source
    assert 'export "${pool}_ENV_VARS=$(pool_value "${pool}" ENV_VARS)"' in source
    assert 'export "${pool}_VLLM_ARGS=$(pool_value "${pool}" VLLM_ARGS)"' in source
    assert 'SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST_HET_GROUP_0}"' in source
    assert 'scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}"' in source
    assert 'for pool in "${pool_names[@]}"' in source
    assert "POOL_PREFIX=${pool}" in source
    assert (
        'COMMAND="${COMMAND//${placeholders[${pool}]}/${pool_urls[${pool}]}}"' in source
    )
    assert "++env.nemo_gym.external_service_readiness" in source
    assert (
        'COMMAND="${COMMAND//${EXTERNAL_SERVICE_READINESS_PLACEHOLDER}/${readiness_arg}}"'
        in source
    )
    assert 'COMMAND+=" ++env.nemo_gym' not in source
    assert "expected_backends:${replicas[${pool}]}" in source
    assert "external_service_readiness_json" not in source
    assert source.index('bash "${RAY_SUB}" &') < source.index(
        "deadline=$((SECONDS + max_startup_timeout))"
    )
    assert source.count("check_startup_steps") == 4
    assert "genrm" not in source.lower()
    assert "nl2bash" not in source.lower()
    assert "safety" not in source.lower()
    assert "RAY_NODELIST" not in source
    assert "external-vllm-lb-preflight" not in source
    assert "if ! ready=$(" in source
    assert (
        'env \\\n    SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST_HET_GROUP_0}"' in source
    )
    assert 'if [[ -n "${SLURM_RESTART_COUNT:-}" ]]; then' in source
    assert (
        'LOG_DIR="${BASE_LOG_DIR}/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT}-logs"'
        in source
    )
    assert 'rm -f "${pool_log_dirs[${pool}]}"/head_ip_*' in source
    assert (
        'echo "[${REPLICA_ID}] ERROR: vLLM exited with status ${vllm_status}"' in source
    )
    assert "if (( vllm_status == 0 )); then" in source


def test_load_balancers_use_lb_container_and_its_python():
    source = (REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh").read_text()

    assert (
        'EXTERNAL_VLLM_LB_CONTAINER="${EXTERNAL_VLLM_LB_CONTAINER:-${CONTAINER}}"'
        in source
    )
    assert '--container-image="${EXTERNAL_VLLM_LB_CONTAINER}"' in source
    assert "PYTHON='${lb_python}' /opt/external-vllm-tools/lb_watchdog.sh" in source


def test_private_ray_and_vllm_ports_match_sub_ephemeral_layout():
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
    source = script.read_text()

    assert "RAY_PORT=1200" in source
    assert "RAY_CLIENT_SERVER_PORT=1201" in source
    assert "MIN_WORKER_PORT=2000" in source
    assert "MAX_WORKER_PORT=2999" in source
    assert "VLLM_ENGINE_PORT=7000" in source
    assert source.count('--min-worker-port="${MIN_WORKER_PORT}"') == 2
    assert source.count('--max-worker-port="${MAX_WORKER_PORT}"') == 2
    assert 'export VLLM_PORT="${VLLM_ENGINE_PORT}"' in source
    assert '--port "${VLLM_HTTP_PORT}"' in source


def test_pool_config_interface_registers_an_arbitrary_third_pool():
    script = REPO_ROOT / "tools/external_gym_vllm/pool_config.sh"
    program = textwrap.dedent(
        f"""
        set -euo pipefail
        source {script}
        register_external_vllm_pool SAFETY \\
          --display-name "Safety judge" \\
          --model safety-model \\
          --container service.sqsh \\
          --python /opt/vllm/bin/python \\
          --replicas 2 \\
          --tensor-parallel-size 4 \\
          --lb-port 9215 \\
          --url-placeholder __SAFETY_BASE_URL__ \\
          --group-id safety-pool
        external_vllm_pool_env SAFETY NCCL_MNNVL_ENABLE=0
        external_vllm_pool_args SAFETY \\
          --dtype bfloat16 \\
          --attention-backend FLASH_ATTN
        printf 'pools=%s\n' "$EXTERNAL_VLLM_POOLS"
        printf 'name=%s\n' "$SAFETY_DISPLAY_NAME"
        printf 'env=%s\n' "$SAFETY_ENV_VARS"
        printf 'args=%s\n' "$(tr '\n' ',' <<< "$SAFETY_VLLM_ARGS")"
        printf 'group=%s\n' "$SAFETY_GROUP_ID"
        printf 'nodes=%s\n' "$EXTERNAL_VLLM_NUM_NODES"
        """
    )
    result = subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "pools=SAFETY",
        "name=Safety judge",
        "env=NCCL_MNNVL_ENABLE=0",
        "args=--dtype,bfloat16,--attention-backend,FLASH_ATTN,",
        "group=safety-pool",
        "nodes=2",
    ]


@pytest.mark.parametrize(
    ("option", "value", "expected_error"),
    [
        ("--replicas", "-5", "TEST_REPLICAS must be a positive integer"),
        (
            "--tensor-parallel-size",
            "0",
            "TEST_TENSOR_PARALLEL_SIZE must be a positive integer",
        ),
        ("--lb-port", "99999999", "TEST_LB_PORT must be at most 65535"),
        ("--vllm-port", "0", "TEST_VLLM_PORT must be a positive integer"),
        (
            "--startup-timeout",
            "nope",
            "TEST_STARTUP_TIMEOUT must be a positive integer",
        ),
    ],
)
def test_pool_registration_rejects_invalid_numeric_values(
    option, value, expected_error
):
    script = REPO_ROOT / "tools/external_gym_vllm/pool_config.sh"
    program = textwrap.dedent(
        f"""
        source {script}
        register_external_vllm_pool TEST \\
          --model model \\
          --container image \\
          --python /opt/python \\
          --replicas 1 \\
          --tensor-parallel-size 4 \\
          --lb-port 9213 \\
          --url-placeholder __TEST_URL__ \\
          {option} {value}
        """
    )

    result = subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    assert result.returncode == 2
    assert expected_error in result.stderr


@pytest.mark.parametrize(
    ("second_pool_args", "expected_error"),
    [
        ("--lb-port 9213 --url-placeholder __SECOND_URL__", "use LB port 9213"),
        (
            "--lb-port 9214 --url-placeholder __FIRST_URL__",
            "use URL placeholder __FIRST_URL__",
        ),
    ],
)
def test_pool_registration_rejects_duplicate_routing_keys(
    second_pool_args, expected_error
):
    script = REPO_ROOT / "tools/external_gym_vllm/pool_config.sh"
    program = textwrap.dedent(
        f"""
        source {script}
        register_external_vllm_pool FIRST \\
          --model model --container image --python /opt/python \\
          --replicas 1 --tensor-parallel-size 4 \\
          --lb-port 9213 --url-placeholder __FIRST_URL__
        register_external_vllm_pool SECOND \\
          --model model --container image --python /opt/python \\
          --replicas 1 --tensor-parallel-size 4 {second_pool_args}
        """
    )

    result = subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    assert result.returncode == 2
    assert expected_error in result.stderr


def test_pool_registration_rejects_partial_nodes_and_unsafe_group_id():
    script = REPO_ROOT / "tools/external_gym_vllm/pool_config.sh"
    command = textwrap.dedent(
        f"""
        source {script}
        register_external_vllm_pool TEST \\
          --model model --container image --python /opt/python \\
          --replicas 1 --tensor-parallel-size 2 \\
          --lb-port 9213 --url-placeholder __TEST_URL__
        """
    )
    unsafe_group_command = command.replace(
        "--tensor-parallel-size 2",
        "--tensor-parallel-size 4 --group-id bad/id",
    )

    partial = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    unsafe_group = subprocess.run(
        ["bash", "-c", unsafe_group_command], capture_output=True, text=True
    )

    assert partial.returncode == 2
    assert "must be divisible by GPUS_PER_NODE=4" in partial.stderr
    assert unsafe_group.returncode == 2
    assert "TEST_GROUP_ID may contain only" in unsafe_group.stderr


def test_submission_validation_checks_placeholders_paths_and_node_total():
    script = REPO_ROOT / "tools/external_gym_vllm/pool_config.sh"
    tools_dir = REPO_ROOT / "tools/external_gym_vllm"
    command = "'run __TEST_URL__ __EXTERNAL_SERVICE_READINESS__'"
    program = textwrap.dedent(
        f"""
        set -euo pipefail
        source {script}
        EXTERNAL_VLLM_SHARED_ROOT={REPO_ROOT}
        BASE_LOG_DIR={REPO_ROOT}/logs
        EXTERNAL_VLLM_TOOLS_DIR_HOST={tools_dir}
        register_external_vllm_pool TEST \\
          --model model --container image --python /opt/python \\
          --replicas 2 --tensor-parallel-size 4 \\
          --lb-port 9213 --url-placeholder __TEST_URL__
        validate_external_vllm_submission {command} 2
        """
    )

    def run_with(old: str, new: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", program.replace(old, new)],
            capture_output=True,
            text=True,
        )

    valid = subprocess.run(["bash", "-c", program], capture_output=True, text=True)
    wrong_nodes = run_with(f"{command} 2", f"{command} 3")
    missing_placeholder = run_with(
        command, "'run without endpoint __EXTERNAL_SERVICE_READINESS__'"
    )
    missing_readiness = run_with(command, "'run __TEST_URL__ 2>&1 | tee run.log'")
    missing_node_count = run_with(f"{command} 2", command)

    assert valid.returncode == 0, valid.stderr
    assert wrong_nodes.returncode == 2
    assert "expected 2 from registered pools" in wrong_nodes.stderr
    assert missing_placeholder.returncode == 2
    assert "submission command is missing __TEST_URL__" in missing_placeholder.stderr
    assert missing_readiness.returncode == 2
    assert (
        "submission command is missing __EXTERNAL_SERVICE_READINESS__"
        in missing_readiness.stderr
    )
    assert missing_node_count.returncode == 0, missing_node_count.stderr
    assert (
        "skipping external hetgroup node-count validation" in missing_node_count.stderr
    )


def _run_lightning_launcher(**overrides):
    launcher = (
        REPO_ROOT / "examples/nemo_gym/nemotron-3.5-lightning/lightning35_launch.sh"
    )
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        root = Path(temp_dir)
        gym_source = root / "Gym"
        gym_actor = (
            gym_source
            / "responses_api_models/local_vllm_model/local_vllm_model_actor.py"
        )
        gym_actor.parent.mkdir(parents=True)
        gym_actor.touch()
        env = {
            "HOME": str(root),
            "PATH": os.environ["PATH"],
            "DRY_RUN": "1",
            "USE_SNAPSHOT": "0",
            "EXP_NAME": "lightning-launcher-test",
            "MODEL_PATH": "test-policy-model",
            "TRAIN_PATH": str(root / "train.jsonl"),
            "VAL_PATH": str(root / "validation.jsonl"),
            "GENRM_MODEL": "test-genrm-model",
            "NL2BASH_JUDGE_MODEL": "test-nl2bash-model",
            "SAFETY_JUDGE_MODEL": "test-safety-model",
            "CONTAINER": "test-container",
            "SANDBOX_CONTAINER": "test-sandbox-container",
            "PERSISTENT_CACHE": str(root / "cache"),
            "RESULTS_DIR": str(root / "results"),
            "GYM_SOURCE": str(gym_source),
            "EXTERNAL_VLLM_SHARED_ROOT": str(REPO_ROOT),
            "SLURM_PARTITION": "test-partition",
            "SLURM_ACCOUNT": "test-account",
        }
        env.update(overrides)
        return subprocess.run(
            ["bash", str(launcher)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )


def test_lightning_launcher_dry_run_fills_external_service_placeholders():
    result = _run_lightning_launcher()

    assert result.returncode == 0, result.stderr
    assert "base_url=__GENRM_BASE_URL__" in result.stdout
    assert "base_url=__NL2BASH_BASE_URL__" in result.stdout
    assert "__EXTERNAL_SERVICE_READINESS__" in result.stdout


def test_lightning_launcher_rejects_invalid_external_pool_tp():
    result = _run_lightning_launcher(GENRM_TENSOR_PARALLEL_SIZE="6")

    assert result.returncode == 2
    assert "must be divisible by GPUS_PER_NODE=4" in result.stderr


# ---------------------------------------------------------------------------
# Persistent pools: serve mode, attach_and_run.sh and the pool_config.sh submit
# helpers, run end to end against stub Slurm and network commands. The stub srun
# registers each replica in the real backend registry, so the wrapper's
# readiness loop runs unchanged. The stub curl answers /health with
# healthy_backends from STUB_LOG/healthy_<port> (all healthy when absent), and
# the stub squeue fails like the real one when STUB_SQUEUE_FAIL is set.
# ---------------------------------------------------------------------------

_STUB_SRUN = r"""#!/bin/bash
export_arg=""
is_lb=0
for arg in "$@"; do
  case "${arg}" in --export=*) export_arg="${arg#--export=}" ;; esac
  [[ "${arg}" == *lb_watchdog.sh* ]] && is_lb=1
done
trap 'exit 0' TERM INT
if (( is_lb )); then
  echo lb >> "${STUB_LOG}/lb_launches"
  while true; do sleep 0.1; done
fi
IFS=, read -ra assignments <<< "${export_arg}"
for assignment in "${assignments[@]}"; do
  [[ "${assignment}" == *=* ]] && export "${assignment}"
done
echo "${REPLICA_ID}" >> "${STUB_LOG}/replica_launches"
source "${EXTERNAL_VLLM_TOOLS_DIR}/vllm_backend_registry.sh"
registry_add "${REPLICA_ID}" 127.0.0.1 8000
while true; do
  if [[ -f "${STUB_LOG}/crash_${REPLICA_ID}" ]]; then
    rm -f "${STUB_LOG}/crash_${REPLICA_ID}"
    exit 1
  fi
  sleep 0.1
done
"""

_STUB_CURL = r"""#!/bin/bash
url="${*: -1}"
if [[ "${url}" == */health ]]; then
  port="${url##*:}"
  port="${port%%/*}"
  healthy=$(cat "${STUB_LOG}/healthy_${port}" 2>/dev/null || echo 99)
  printf '{"status": "ok", "healthy_backends": %s, "total_backends": %s}\n' "${healthy}" "${healthy}"
fi
exit 0
"""

_STUB_SQUEUE = r"""#!/bin/bash
# Some Slurm configurations make every client call print a non-fatal error.
[[ -n "${STUB_SQUEUE_WARN:-}" ]] && echo "${STUB_SQUEUE_WARN}" >&2
if [[ -n "${STUB_SQUEUE_FAIL:-}" ]]; then
  echo "slurm_load_jobs error: ${STUB_SQUEUE_FAIL}" >&2
  exit 1
fi
echo "${STUB_SQUEUE_STATE:-}"
"""

_STUB_SCONTROL = r"""#!/bin/bash
if [[ "$1 $2" == "show hostnames" ]]; then
  tr , "\n" <<< "$3"
elif [[ "$1 $2" == "show job" ]]; then
  echo "JobId=$3 EndTime=${STUB_END_TIME:-Unknown}"
fi
"""

_STUB_SBATCH = r"""#!/bin/bash
echo "$*" >> "${STUB_LOG}/sbatch_calls"
echo "mode=${EXTERNAL_VLLM_MODE:-} pool_job=${EXTERNAL_VLLM_POOL_JOB:-} command=${COMMAND:-}" >> "${STUB_LOG}/sbatch_env"
echo "$(( $(wc -l < "${STUB_LOG}/sbatch_calls") + 700 ));cluster"
"""

_WRAPPER = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
_ATTACH = REPO_ROOT / "tools/external_gym_vllm/attach_and_run.sh"
_TRAIN_COMMAND = (
    "train a=__ALPHA_BASE_URL__ b=__BETA_BASE_URL__ __EXTERNAL_SERVICE_READINESS__"
)


def _stub_env(tmp_path, **extra):
    """Stub Slurm/network commands on PATH and a pool contract for two pools."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "srun": _STUB_SRUN,
        "scontrol": _STUB_SCONTROL,
        "getent": '#!/bin/bash\necho "127.0.0.1 STREAM $2"\n',
        "curl": _STUB_CURL,
        "squeue": _STUB_SQUEUE,
        "sbatch": _STUB_SBATCH,
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)
    # The wrapper requires the tools directory under EXTERNAL_VLLM_SHARED_ROOT
    # (replicas mount it), so work from a copy inside tmp_path.
    tools = tmp_path / "tools"
    shutil.copytree(REPO_ROOT / "tools/external_gym_vllm", tools)
    stub_log = tmp_path / "stub_log"
    stub_log.mkdir()
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_LOG": str(stub_log),
        "STUB_SQUEUE_STATE": "RUNNING",
        "SLURM_JOB_ID": "555",
        "SLURM_JOB_ACCOUNT": "account",
        "SLURM_JOB_PARTITION": "partition",
        "SLURM_SUBMIT_DIR": str(REPO_ROOT),
        "BASE_LOG_DIR": str(tmp_path / "logs"),
        "CONTAINER": "training.sqsh",
        "MOUNTS": f"{tmp_path}:{tmp_path}",
        "EXTERNAL_VLLM_SHARED_ROOT": str(tmp_path),
        "EXTERNAL_VLLM_TOOLS_DIR_HOST": str(tools),
        "EXTERNAL_VLLM_POOLS": "ALPHA BETA",
    }
    for pool, replicas, port in (("ALPHA", 2, 9213), ("BETA", 1, 9214)):
        env.update(
            {
                f"{pool}_CONTAINER": "judge.sqsh",
                f"{pool}_MODEL": "model-id",
                f"{pool}_VLLM_PYTHON": "/opt/python",
                f"{pool}_REPLICAS": str(replicas),
                f"{pool}_TENSOR_PARALLEL_SIZE": "4",
                f"{pool}_LB_PORT": str(port),
                f"{pool}_URL_PLACEHOLDER": f"__{pool}_BASE_URL__",
            }
        )
    env.update(extra)
    return env


def _serve_env(tmp_path, **extra):
    return _stub_env(
        tmp_path,
        EXTERNAL_VLLM_MODE="serve",
        EXTERNAL_VLLM_POOL_DIR=str(tmp_path / "pool"),
        EXTERNAL_VLLM_SERVE_POLL_S="0.2",
        SLURM_JOB_NODELIST="node1,node2,node3",
        **extra,
    )


def _wait_for(predicate, timeout_s, message):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(message)


def _start(script, env, output):
    return subprocess.Popen(
        ["bash", str(script)],
        env=env,
        stdout=open(output, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    # Reap anything the stubs left behind (each process runs in its own session).
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=5)


def test_serve_mode_rejects_a_heterogeneous_job(tmp_path):
    env = _serve_env(tmp_path, SLURM_HET_SIZE="2")
    result = subprocess.run(
        ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "serve mode runs as a plain (non-heterogeneous) job" in result.stderr


def test_serve_mode_requires_a_pool_dir_and_rejects_unknown_modes(tmp_path):
    env = _stub_env(tmp_path, EXTERNAL_VLLM_MODE="serve", SLURM_JOB_NODELIST="node1")
    result = subprocess.run(
        ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "EXTERNAL_VLLM_POOL_DIR is required in serve mode" in result.stderr

    env["EXTERNAL_VLLM_MODE"] = "attach"
    result = subprocess.run(
        ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "EXTERNAL_VLLM_MODE must be inline or serve, got: attach" in result.stderr


def test_serve_mode_publishes_relaunches_and_withdraws_ready_while_degraded(tmp_path):
    pool_dir = tmp_path / "pool"
    env = _serve_env(tmp_path, STUB_END_TIME="2026-10-14T06:00:00")
    stub_log = Path(env["STUB_LOG"])
    process = _start(_WRAPPER, env, tmp_path / "serve.out")
    try:
        # COMMAND is not required in serve mode, and no ray.sub is started.
        _wait_for((pool_dir / "READY").exists, 60, "pools never published READY")
        assert (pool_dir / "READY").read_text().startswith("555-0 ")
        assert (pool_dir / "owner").read_text().strip() == "555-0"
        assert (pool_dir / "generation").read_text().strip() == "555-0"
        assert (pool_dir / "job_id").read_text().strip() == "555"
        assert (pool_dir / "heartbeat").exists()
        assert (pool_dir / "end_time").read_text().startswith("2026-10-14T06:00:00")
        manifest = [
            line.split("\t")
            for line in (pool_dir / "manifest.tsv").read_text().splitlines()
            if not line.startswith("#")
        ]
        assert manifest == [
            [
                "ALPHA",
                "http://127.0.0.1:9213/v1",
                "http://127.0.0.1:9213/health",
                "2",
                "model",
                "3600",
            ],
            [
                "BETA",
                "http://127.0.0.1:9214/v1",
                "http://127.0.0.1:9214/health",
                "1",
                "model",
                "3600",
            ],
        ]
        assert (
            pool_dir / "alpha_url"
        ).read_text().strip() == "http://127.0.0.1:9213/v1"
        assert not (pool_dir / "STOPPED").exists()
        assert sorted((stub_log / "replica_launches").read_text().split()) == [
            "555-alpha-0",
            "555-alpha-1",
            "555-beta-0",
        ]

        # A replica that dies is relaunched; the published URLs stay as they were.
        (stub_log / "crash_555-alpha-1").touch()
        _wait_for(
            lambda: (stub_log / "replica_launches")
            .read_text()
            .split()
            .count("555-alpha-1")
            == 2,
            15,
            "dead replica was not relaunched",
        )
        assert (
            pool_dir / "alpha_url"
        ).read_text().strip() == "http://127.0.0.1:9213/v1"

        # While a load balancer counts fewer healthy backends than replicas (a relaunch
        # loading weights, or a wedged replica), READY is withdrawn; it comes back once
        # the pool is whole again.
        (stub_log / "healthy_9213").write_text("1\n")
        _wait_for(
            lambda: not (pool_dir / "READY").exists(), 15, "READY stayed while degraded"
        )
        (stub_log / "healthy_9213").unlink()
        _wait_for((pool_dir / "READY").exists, 15, "READY was not restored")

        # Attached jobs are logged by the serve loop.
        (pool_dir / "attached" / "900").write_text("555-0\n")
        _wait_for(
            lambda: "attached training jobs: 900"
            in (tmp_path / "serve.out").read_text(),
            15,
            "attached job not logged",
        )
    finally:
        _stop(process)

    output = (tmp_path / "serve.out").read_text()
    assert "ALPHA replica 1 exited; relaunch 1/5" in output
    assert "READY withdrawn; healthy backends: ALPHA=1/2" in output
    assert not (pool_dir / "READY").exists()
    assert (pool_dir / "STOPPED").read_text().startswith("555-0 ")
    assert not (pool_dir / "owner").exists()


def test_serve_mode_waits_for_every_load_balancer_to_count_every_replica(tmp_path):
    pool_dir = tmp_path / "pool"
    env = _serve_env(tmp_path)
    (Path(env["STUB_LOG"]) / "healthy_9214").write_text("0\n")
    process = _start(_WRAPPER, env, tmp_path / "serve.out")
    try:
        _wait_for(
            lambda: "BETA ready: 1/1" in (tmp_path / "serve.out").read_text(),
            60,
            "replicas never registered",
        )
        time.sleep(2)
        # Registered and answering /models, but the load balancer has not counted it.
        assert not (pool_dir / "READY").exists()
        (Path(env["STUB_LOG"]) / "healthy_9214").unlink()
        _wait_for((pool_dir / "READY").exists, 30, "READY not published once healthy")
    finally:
        _stop(process)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({}, "is owned by live pool job 444 (RUNNING)"),
        # squeue succeeds but prints a config error on stderr first.
        (
            {
                "STUB_SQUEUE_WARN": "squeue: error: Ignoring ControlMachine since SlurmctldHost is set."
            },
            "is owned by live pool job 444 (RUNNING)",
        ),
        # squeue fails: the owner's state is unknown, so the directory is not taken.
        (
            {"STUB_SQUEUE_FAIL": "Socket timed out on send/recv operation"},
            "Could not query pool job 444",
        ),
    ],
)
def test_serve_mode_refuses_a_pool_dir_owned_by_a_live_job(tmp_path, extra, message):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "owner").write_text("444-0\n")
    (pool_dir / "READY").write_text("444-0 then\n")
    env = _serve_env(tmp_path, **extra)
    result = subprocess.run(
        ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert message in result.stderr
    # The other job's state is left alone.
    assert (pool_dir / "owner").read_text().strip() == "444-0"
    assert (pool_dir / "READY").exists()
    assert not (pool_dir / "STOPPED").exists()


def test_replica_log_survives_a_relaunch():
    """A relaunched replica reuses vllm_<i>.log; appending keeps the crash it replaces."""
    source = _WRAPPER.read_text()
    assert '>> "${LOG_FILE}" 2>&1 &' in source
    assert '> "${LOG_FILE}" 2>&1 &' not in source.replace('>> "${LOG_FILE}"', "")


@pytest.mark.parametrize(
    ("owner", "extra", "taken_from"),
    [
        ("444-0", {"STUB_SQUEUE_STATE": "COMPLETED"}, "444-0"),  # owner finished
        ("444-0", {"STUB_SQUEUE_FAIL": "Invalid job id specified"}, "444-0"),  # purged
        ("555-0", {"SLURM_RESTART_COUNT": "1"}, "555-0"),  # this job, requeued
    ],
)
def test_serve_mode_takes_over_from_a_gone_owner_or_its_own_requeue(
    tmp_path, owner, extra, taken_from
):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "owner").write_text(f"{owner}\n")
    (pool_dir / "READY").write_text(f"{owner} then\n")  # stale; must not survive
    env = _serve_env(tmp_path, **extra)
    process = _start(_WRAPPER, env, tmp_path / "serve.out")
    try:
        generation = f"555-{extra.get('SLURM_RESTART_COUNT', '0')}"
        _wait_for(
            lambda: (pool_dir / "READY").exists()
            and (pool_dir / "READY").read_text().startswith(generation),
            60,
            "takeover did not publish READY",
        )
        assert (pool_dir / "generation").read_text().strip() == generation
    finally:
        _stop(process)
    assert (
        f"Taking over {pool_dir} from {taken_from}"
        in (tmp_path / "serve.out").read_text()
    )


@pytest.mark.parametrize(("reset_s", "survives"), [("1", True), ("600", False)])
def test_serve_mode_relaunch_count_resets_after_a_stable_run(
    tmp_path, reset_s, survives
):
    pool_dir = tmp_path / "pool"
    env = _serve_env(
        tmp_path, EXTERNAL_VLLM_MAX_RESTARTS="1", EXTERNAL_VLLM_RESTART_RESET_S=reset_s
    )
    stub_log = Path(env["STUB_LOG"])
    process = _start(_WRAPPER, env, tmp_path / "serve.out")
    try:
        _wait_for((pool_dir / "READY").exists, 60, "pools never published READY")
        for crash in (1, 2):
            time.sleep(1.5)  # outlive a 1 s reset window
            (stub_log / "crash_555-beta-0").touch()
            _wait_for(
                lambda: not (stub_log / "crash_555-beta-0").exists(),
                15,
                f"crash {crash} was not consumed",
            )
        time.sleep(1.5)
        assert (process.poll() is None) == survives
    finally:
        _stop(process)
    output = (tmp_path / "serve.out").read_text()
    if survives:
        assert output.count("BETA replica 0 exited; relaunch 1/1") == 2
    else:
        assert "BETA replica 0 exited again after 1 relaunch(es)" in output
        assert (pool_dir / "STOPPED").exists()


def _published_pool(tmp_path, generation="555-0"):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir(exist_ok=True)
    (pool_dir / "generation").write_text(f"{generation}\n")
    (pool_dir / "READY").write_text(f"{generation} 2026-01-01T00:00:00\n")
    (pool_dir / "heartbeat").write_text(f"{generation}\n")
    (pool_dir / "manifest.tsv").write_text(
        "# External vLLM pools\n"
        "ALPHA\thttp://10.0.0.1:9213/v1\thttp://10.0.0.1:9213/health\t2\tmodel\t600\n"
        "BETA\thttp://10.0.0.1:9214/v1\thttp://10.0.0.1:9214/health\t1\tmodel\t900\n"
    )
    return pool_dir


def _attach_env(tmp_path, ray_sleep="0", **extra):
    ray_sub = tmp_path / "ray.sub"
    ray_sub.write_text(
        f'#!/bin/bash\necho "$COMMAND" > {tmp_path / "command_seen"}\nsleep {ray_sleep}\n'
    )
    settings = {
        "SLURM_JOB_ID": "900",
        "EXTERNAL_VLLM_POOL_DIR": str(tmp_path / "pool"),
        "EXTERNAL_VLLM_POOL_JOB": "555",
        "COMMAND": _TRAIN_COMMAND,
        "RAY_SUB": str(ray_sub),
        "EXTERNAL_VLLM_ATTACH_POLL_S": "0.1",
        "EXTERNAL_VLLM_ATTACH_WATCH_S": "0.1",
    }
    settings.update(extra)
    return _stub_env(tmp_path, **settings)


def test_attach_and_run_resolves_urls_and_the_readiness_gate_from_the_manifest(
    tmp_path,
):
    pool_dir = _published_pool(tmp_path)
    env = _attach_env(tmp_path, ray_sleep="1")
    process = _start(_ATTACH, env, tmp_path / "attach.out")
    try:
        _wait_for((tmp_path / "command_seen").exists, 30, "ray.sub was not started")
        # Registered as attached while training runs, removed after.
        assert (pool_dir / "attached" / "900").read_text().startswith("555-0 ")
        process.wait(timeout=30)
    finally:
        _stop(process)
    assert process.returncode == 0, (tmp_path / "attach.out").read_text()
    assert (tmp_path / "command_seen").read_text().strip() == (
        "train a=http://10.0.0.1:9213/v1 b=http://10.0.0.1:9214/v1 "
        "++env.nemo_gym.external_service_readiness='{services:["
        '{name:ALPHA,url:"http://10.0.0.1:9213/health",expected_backends:2},'
        '{name:BETA,url:"http://10.0.0.1:9214/health",expected_backends:1}],'
        "timeout_seconds:900,poll_interval_seconds:5,request_timeout_seconds:10}'"
    )
    assert not (pool_dir / "attached" / "900").exists()


def test_attach_and_run_requires_the_readiness_placeholder(tmp_path):
    _published_pool(tmp_path)
    env = _attach_env(
        tmp_path, COMMAND="train a=__ALPHA_BASE_URL__ b=__BETA_BASE_URL__"
    )
    result = subprocess.run(
        ["bash", str(_ATTACH)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert "missing __EXTERNAL_SERVICE_READINESS__" in result.stderr


@pytest.mark.parametrize(
    "stale",
    ["generation_from_another_job", "heartbeat_too_old"],
)
def test_attach_and_run_does_not_trust_a_stale_ready(tmp_path, stale):
    pool_dir = _published_pool(tmp_path)
    extra = {"EXTERNAL_VLLM_ATTACH_TIMEOUT": "1"}
    if stale == "generation_from_another_job":
        (pool_dir / "generation").write_text("444-0\n")
    else:
        extra["EXTERNAL_VLLM_HEARTBEAT_STALE_S"] = "1"
        old = time.time() - 60
        os.utime(pool_dir / "heartbeat", (old, old))
    env = _attach_env(tmp_path, **extra)
    result = subprocess.run(
        ["bash", str(_ATTACH)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert "pool job 555 not READY after 1s" in result.stderr
    assert not (tmp_path / "command_seen").exists()


def test_attach_and_run_survives_transient_squeue_failures(tmp_path):
    # Not READY yet, and squeue failing: a transient slurmctld error must not be
    # read as "the pool job is gone". The real squeue exits non-zero here.
    env = _attach_env(
        tmp_path, STUB_SQUEUE_FAIL="Socket timed out on send/recv operation"
    )
    process = _start(_ATTACH, env, tmp_path / "attach.out")
    try:
        _wait_for(
            lambda: (tmp_path / "attach.out").read_text().count("squeue failed") >= 3,
            30,
            "squeue failures were not retried",
        )
        assert process.poll() is None
        _published_pool(tmp_path)
        process.wait(timeout=30)
    finally:
        _stop(process)
    assert process.returncode == 0, (tmp_path / "attach.out").read_text()


def test_attach_and_run_declares_a_purged_pool_job_gone_after_repeated_checks(tmp_path):
    env = _attach_env(tmp_path, STUB_SQUEUE_FAIL="Invalid job id specified")
    result = subprocess.run(
        ["bash", str(_ATTACH)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert result.stderr.count("pool job 555: GONE") == 3
    assert "[FATAL] pool job 555 is gone" in result.stderr
    assert not (tmp_path / "command_seen").exists()


@pytest.mark.parametrize(
    "failure", ["stopped", "generation_changed", "heartbeat_stale"]
)
def test_attach_and_run_stops_training_when_the_pool_fails(tmp_path, failure):
    pool_dir = _published_pool(tmp_path)
    env = _attach_env(tmp_path, ray_sleep="300", EXTERNAL_VLLM_HEARTBEAT_STALE_S="2")
    process = _start(_ATTACH, env, tmp_path / "attach.out")
    try:
        _wait_for((tmp_path / "command_seen").exists, 30, "ray.sub was not started")
        if failure == "stopped":
            (pool_dir / "STOPPED").write_text("555-0 now status=143\n")
        elif failure == "generation_changed":
            (pool_dir / "generation").write_text("555-1\n")
        else:
            pass  # nothing refreshes the heartbeat
        if failure != "heartbeat_stale":
            # Keep the heartbeat fresh so only the injected failure can trigger.
            (pool_dir / "heartbeat").touch()
        process.wait(timeout=30)
    finally:
        _stop(process)
    output = (tmp_path / "attach.out").read_text()
    assert process.returncode == 1
    assert {
        "stopped": "pool job stopped",
        "generation_changed": "pool generation changed from 555-0 to 555-1",
        "heartbeat_stale": "pool heartbeat is",
    }[failure] in output
    assert "stopping NeMo RL" in output
    assert not (pool_dir / "attached" / "900").exists()


@pytest.mark.parametrize("require", ["0", "1"])
def test_attach_and_run_checks_that_the_pool_outlives_the_job(tmp_path, require):
    pool_dir = _published_pool(tmp_path)
    (pool_dir / "end_time").write_text("2026-10-14T06:00:00\n")
    env = _attach_env(
        tmp_path,
        STUB_END_TIME="2026-10-14T08:00:00",
        EXTERNAL_VLLM_ATTACH_REQUIRE_LIFETIME=require,
    )
    result = subprocess.run(
        ["bash", str(_ATTACH)], env=env, capture_output=True, text=True, timeout=60
    )
    assert "120 min before this job's time limit" in result.stderr
    if require == "1":
        assert result.returncode != 0
        assert not (tmp_path / "command_seen").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "command_seen").exists()


def test_pool_config_submit_helpers_reuse_a_live_pool_and_depend_on_it_only_while_loading(
    tmp_path,
):
    env = _stub_env(tmp_path, RAY_SUB=str(REPO_ROOT / "ray.sub"))
    pool_dir = tmp_path / "pool"
    program = textwrap.dedent(
        f"""
        set -euo pipefail
        source {REPO_ROOT / "tools/external_gym_vllm/pool_config.sh"}
        export EXTERNAL_VLLM_NUM_NODES=3
        pool_job=$(submit_external_vllm_pool_job {pool_dir} --time=12:00:00)
        again=$(submit_external_vllm_pool_job {pool_dir} --time=12:00:00)
        loading=$(submit_external_vllm_attached_job {pool_dir} "$pool_job" "{_TRAIN_COMMAND}" --nodes=68)
        touch {pool_dir}/READY
        warm=$(submit_external_vllm_attached_job {pool_dir} "$pool_job" "{_TRAIN_COMMAND}" --nodes=68)
        printf 'pool=%s again=%s loading=%s warm=%s\\n' "$pool_job" "$again" "$loading" "$warm"
        if submit_external_vllm_attached_job {pool_dir} "$pool_job" "train __ALPHA_BASE_URL__" --nodes=68; then
          echo "accepted a command without every placeholder"
        fi
        """
    )
    result = subprocess.run(
        ["bash", "-c", program], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "pool=701 again=701 loading=702 warm=703" in result.stdout
    assert "accepted" not in result.stdout
    assert "missing __BETA_BASE_URL__ for pool BETA" in result.stderr
    assert (pool_dir / "submitted_job").read_text().strip() == "701"
    calls = (Path(env["STUB_LOG"]) / "sbatch_calls").read_text().splitlines()
    tools = env["EXTERNAL_VLLM_TOOLS_DIR_HOST"]
    assert calls == [
        f"--parsable --nodes=3 --exclusive --gres=gpu:4 --export=ALL --time=12:00:00 {tools}/run_in_allocation.sh",
        f"--parsable --export=ALL --dependency=after:701+15 --nodes=68 {tools}/attach_and_run.sh",
        f"--parsable --export=ALL --nodes=68 {tools}/attach_and_run.sh",
    ]
    submitted_env = (Path(env["STUB_LOG"]) / "sbatch_env").read_text().splitlines()
    assert submitted_env[0].startswith("mode=serve ")
    assert submitted_env[1].startswith(
        "mode= pool_job=701 command=train a=__ALPHA_BASE_URL__"
    )
