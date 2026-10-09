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

    # The component selectors are per-mode arrays: hetgroups inline, none in serve.
    assert '"${het_ext[@]}"' in replica_launch
    assert '-A "${SLURM_JOB_ACCOUNT}"' not in replica_launch
    assert '-p "${SLURM_JOB_PARTITION}"' not in replica_launch
    assert '"${het_ray[@]}"' in lb_launch
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
        validate_external_vllm_submission 'run __TEST_URL__' 2
        """
    )

    valid = subprocess.run(["bash", "-c", program], capture_output=True, text=True)
    wrong_nodes = subprocess.run(
        ["bash", "-c", program.replace("'run __TEST_URL__' 2", "'run __TEST_URL__' 3")],
        capture_output=True,
        text=True,
    )
    missing_placeholder = subprocess.run(
        [
            "bash",
            "-c",
            program.replace("'run __TEST_URL__' 2", "'run without endpoint' 2"),
        ],
        capture_output=True,
        text=True,
    )
    missing_node_count = subprocess.run(
        [
            "bash",
            "-c",
            program.replace(
                "validate_external_vllm_submission 'run __TEST_URL__' 2",
                "validate_external_vllm_submission 'run __TEST_URL__'",
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert valid.returncode == 0, valid.stderr
    assert wrong_nodes.returncode == 2
    assert "expected 2 from registered pools" in wrong_nodes.stderr
    assert missing_placeholder.returncode == 2
    assert "submission command is missing __TEST_URL__" in missing_placeholder.stderr
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


def test_lightning_launcher_dry_run_builds_reference_external_pool_topology():
    result = _run_lightning_launcher()

    assert result.returncode == 0, result.stderr
    assert "Nodes:       86 total" in result.stdout
    assert "Hetgroup 0: 66 NeMo RL nodes" in result.stdout
    assert "Hetgroup 1: 20 external-service nodes" in result.stdout
    assert "GenRM:    8 independent TP=8, DP=1 servers" in result.stdout
    assert "NL2Bash:  4 independent TP=4, DP=1 servers" in result.stdout
    assert "base_url=__GENRM_BASE_URL__" in result.stdout
    assert "base_url=__NL2BASH_BASE_URL__" in result.stdout
    assert "--reasoning-parser\n  nemotron_v3" in result.stdout
    assert "--reasoning-parser-plugin" not in result.stdout
    assert "--attention-backend\n  TRITON_ATTN" in result.stdout
    assert result.stdout.count("--enable-expert-parallel") == 2


def test_lightning_launcher_rejects_invalid_external_pool_tp():
    result = _run_lightning_launcher(GENRM_TENSOR_PARALLEL_SIZE="6")

    assert result.returncode == 2
    assert "must be divisible by GPUS_PER_NODE=4" in result.stderr


# ---------------------------------------------------------------------------
# Serve mode, early Ray, and attach_and_run.sh, exercised end to end against
# stub Slurm commands. The stub srun registers each replica in the real backend
# registry (or not, per test), so the wrapper's readiness loop runs unchanged.
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
if [[ -z "${STUB_NO_REGISTER:-}" ]]; then
  source "${EXTERNAL_VLLM_TOOLS_DIR}/vllm_backend_registry.sh"
  registry_add "${REPLICA_ID}" 127.0.0.1 8000
fi
while true; do
  if [[ -f "${STUB_LOG}/crash_${REPLICA_ID}" ]]; then
    rm -f "${STUB_LOG}/crash_${REPLICA_ID}"
    exit 1
  fi
  sleep 0.1
done
"""


def _stub_env(tmp_path, **extra):
    """Stub Slurm/network commands on PATH and a pool contract for two pools."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "srun": _STUB_SRUN,
        "scontrol": '#!/bin/bash\n[[ "$1 $2" == "show hostnames" ]] && tr , "\\n" <<< "$3"\n',
        "getent": '#!/bin/bash\necho "127.0.0.1 STREAM $2"\n',
        "curl": "#!/bin/bash\nexit 0\n",
        "squeue": '#!/bin/bash\necho "${STUB_SQUEUE_STATE:-}"\n',
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)
    tools = tmp_path / "tools"
    subprocess.run(
        ["cp", "-a", str(REPO_ROOT / "tools/external_gym_vllm"), str(tools)],
        check=True,
    )
    stub_log = tmp_path / "stub_log"
    stub_log.mkdir()
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_LOG": str(stub_log),
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


def _wait_for(predicate, timeout_s, message):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(message)


def _start(script, env, tmp_path):
    out = open(tmp_path / "wrapper.out", "w")
    return subprocess.Popen(
        ["bash", str(script)],
        env=env,
        stdout=out,
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
    # Reap anything the stubs left behind (each test runs in its own session).
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=5)


def test_serve_mode_rejects_a_heterogeneous_job(tmp_path):
    env = _stub_env(
        tmp_path,
        EXTERNAL_VLLM_MODE="serve",
        EXTERNAL_VLLM_POOL_DIR=str(tmp_path / "pool"),
        SLURM_HET_SIZE="2",
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "serve mode runs as a plain (non-heterogeneous) job" in result.stderr


def test_serve_mode_requires_a_pool_dir_and_rejects_unknown_modes(tmp_path):
    script = REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh"
    env = _stub_env(tmp_path, EXTERNAL_VLLM_MODE="serve")
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "EXTERNAL_VLLM_POOL_DIR is required in serve mode" in result.stderr

    env["EXTERNAL_VLLM_MODE"] = "attach"
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "EXTERNAL_VLLM_MODE must be inline or serve, got: attach" in result.stderr


def test_serve_mode_publishes_urls_relaunches_dead_replicas_and_withdraws_ready(
    tmp_path,
):
    pool_dir = tmp_path / "pool"
    env = _stub_env(
        tmp_path,
        EXTERNAL_VLLM_MODE="serve",
        EXTERNAL_VLLM_POOL_DIR=str(pool_dir),
        EXTERNAL_VLLM_SERVE_POLL_S="0.2",
        SLURM_JOB_NODELIST="node1,node2,node3",
    )
    stub_log = Path(env["STUB_LOG"])
    process = _start(
        REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh", env, tmp_path
    )
    try:
        # COMMAND is not required in serve mode, and no ray.sub is started.
        _wait_for((pool_dir / "READY").exists, 60, "pools never published READY")
        assert (pool_dir / "pools.env").read_text().splitlines() == [
            "ALPHA=http://127.0.0.1:9213/v1",
            "BETA=http://127.0.0.1:9214/v1",
        ]
        assert (
            pool_dir / "alpha_url"
        ).read_text().strip() == "http://127.0.0.1:9213/v1"
        assert (pool_dir / "beta_url").read_text().strip() == "http://127.0.0.1:9214/v1"
        assert (pool_dir / "job_id").read_text().strip() == "555"
        assert not (pool_dir / "STOPPED").exists()
        launches = (stub_log / "replica_launches").read_text().split()
        assert sorted(launches) == ["555-alpha-0", "555-alpha-1", "555-beta-0"]

        # A replica that dies is relaunched; the published URLs stay as they were.
        (stub_log / "crash_555-alpha-1").touch()
        _wait_for(
            lambda: (
                (stub_log / "replica_launches").read_text().split().count("555-alpha-1")
                == 2
            ),
            15,
            "dead replica was not relaunched",
        )
        assert (pool_dir / "READY").exists()
        assert (
            pool_dir / "alpha_url"
        ).read_text().strip() == "http://127.0.0.1:9213/v1"
    finally:
        _stop(process)

    output = (tmp_path / "wrapper.out").read_text()
    assert "ALPHA replica 1 exited; relaunch 1/5" in output
    assert not (pool_dir / "READY").exists()
    assert (pool_dir / "STOPPED").read_text().startswith("555 ")


def test_serve_mode_gives_up_on_a_crash_looping_replica(tmp_path):
    pool_dir = tmp_path / "pool"
    env = _stub_env(
        tmp_path,
        EXTERNAL_VLLM_MODE="serve",
        EXTERNAL_VLLM_POOL_DIR=str(pool_dir),
        EXTERNAL_VLLM_SERVE_POLL_S="0.2",
        EXTERNAL_VLLM_MAX_RESTARTS="1",
        SLURM_JOB_NODELIST="node1,node2,node3",
    )
    stub_log = Path(env["STUB_LOG"])
    process = _start(
        REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh", env, tmp_path
    )
    try:
        _wait_for((pool_dir / "READY").exists, 60, "pools never published READY")
        for launch in (1, 2):
            (stub_log / "crash_555-beta-0").touch()
            _wait_for(
                lambda: not (stub_log / "crash_555-beta-0").exists(),
                15,
                f"crash {launch} was not consumed",
            )
        process.wait(timeout=15)
    finally:
        _stop(process)

    assert process.returncode == 1
    assert (
        "BETA replica 0 exited 1 times; giving up"
        in (tmp_path / "wrapper.out").read_text()
    )
    assert not (pool_dir / "READY").exists()
    assert (pool_dir / "STOPPED").exists()


def test_early_ray_starts_nemo_rl_before_the_pools_are_healthy(tmp_path):
    ray_sub = tmp_path / "ray.sub"
    command_seen = tmp_path / "command_seen"
    ray_sub.write_text(f'#!/bin/bash\necho "$COMMAND" > {command_seen}\nsleep 600\n')
    env = _stub_env(
        tmp_path,
        SLURM_HET_SIZE="2",
        SLURM_JOB_NODELIST_HET_GROUP_0="ray1",
        SLURM_JOB_NODELIST_HET_GROUP_1="node1,node2,node3",
        COMMAND="train alpha=__ALPHA_BASE_URL__ beta=__BETA_BASE_URL__",
        RAY_SUB=str(ray_sub),
        EXTERNAL_VLLM_EARLY_RAY="1",
        STUB_NO_REGISTER="1",  # replicas never become ready
    )
    process = _start(
        REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh", env, tmp_path
    )
    try:
        _wait_for(command_seen.exists, 30, "ray.sub was not started while pools load")
        assert command_seen.read_text().strip() == (
            "train alpha=http://127.0.0.1:9213/v1 beta=http://127.0.0.1:9214/v1"
        )
        assert process.poll() is None  # still waiting on the pools
    finally:
        _stop(process)
    output = (tmp_path / "wrapper.out").read_text()
    assert "starting NeMo RL while external vLLM pools load" in output
    assert "ALPHA ready: 0/2" in output


def _attach_env(tmp_path, *, ready_job, squeue_state, **extra):
    env = _stub_env(tmp_path)
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "job_id").write_text(f"{ready_job}\n")
    (pool_dir / "READY").write_text("2026-01-01T00:00:00\n")
    (pool_dir / "alpha_url").write_text("http://10.0.0.1:9213/v1\n")
    (pool_dir / "beta_url").write_text("http://10.0.0.1:9214/v1\n")
    ray_sub = tmp_path / "ray.sub"
    ray_sub.write_text(f'#!/bin/bash\necho "$COMMAND" > {tmp_path / "command_seen"}\n')
    env.update(
        {
            "EXTERNAL_VLLM_POOL_DIR": str(pool_dir),
            "EXTERNAL_VLLM_POOL_JOB": "777",
            "COMMAND": "train alpha=__ALPHA_BASE_URL__ beta=__BETA_BASE_URL__",
            "RAY_SUB": str(ray_sub),
            "STUB_SQUEUE_STATE": squeue_state,
            "EXTERNAL_VLLM_ATTACH_POLL_S": "0.1",
            **extra,
        }
    )
    return env


def test_attach_and_run_substitutes_pool_urls_and_runs_ray_sub(tmp_path):
    env = _attach_env(tmp_path, ready_job="777", squeue_state="RUNNING")
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "tools/external_gym_vllm/attach_and_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "command_seen").read_text().strip() == (
        "train alpha=http://10.0.0.1:9213/v1 beta=http://10.0.0.1:9214/v1"
    )
    assert "attach: ALPHA -> http://10.0.0.1:9213/v1" in result.stdout


def test_attach_and_run_ignores_ready_left_by_another_pool_job(tmp_path):
    env = _attach_env(
        tmp_path,
        ready_job="666",
        squeue_state="RUNNING",
        EXTERNAL_VLLM_ATTACH_TIMEOUT="1",
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "tools/external_gym_vllm/attach_and_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "judge pool job 777 not READY after 1s" in result.stderr
    assert not (tmp_path / "command_seen").exists()


def test_attach_and_run_fails_when_the_pool_job_is_gone(tmp_path):
    env = _attach_env(tmp_path, ready_job="777", squeue_state="")
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "tools/external_gym_vllm/attach_and_run.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "judge pool job 777 is gone" in result.stderr
    assert not (tmp_path / "command_seen").exists()
