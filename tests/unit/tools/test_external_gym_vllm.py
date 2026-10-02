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
import shlex
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
from omegaconf import OmegaConf

from nemo_rl.utils.config import parse_hydra_overrides
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

    # The per-mode hetgroup, account, and partition routing of these two steps
    # is asserted on recorded srun argv by the fake-Slurm tests below.
    assert source.count('srun "${replica_step_args[@]}" \\') == 1
    assert source.count('srun "${lb_step_args[@]}" \\') == 1

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
    assert "external_vllm_readiness_override" in source
    assert (
        'readiness_targets+=("${pool}" "${health_urls[${pool}]}" "${replicas[${pool}]}")'
        in source
    )
    assert "external_service_readiness_json" not in source
    assert source.index('bash "${RAY_SUB}" &') < source.index(
        "deadline=$((SECONDS + max_startup_timeout))"
    )
    assert source.count("check_startup_steps") == 3
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
    services_only = subprocess.run(
        [
            "bash",
            "-c",
            program.replace(
                "validate_external_vllm_submission 'run __TEST_URL__' 2",
                "validate_external_vllm_services 2",
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
        "skipping external-service node-count validation" in missing_node_count.stderr
    )
    assert services_only.returncode == 0, services_only.stderr


def _write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text))
    path.chmod(0o755)


def _fake_slurm_env(tmp_path: Path) -> dict[str, str]:
    """Environment that runs run_in_allocation.sh against fake Slurm tools.

    Each srun records its argv. A replica step registers its backend as ready, as
    the real server body does once vLLM is healthy, then idles until killed.
    Hostnames resolve through a fixed table, and every sleep is shortened.
    """
    tools_dir = tmp_path / "tools"
    shutil.copytree(REPO_ROOT / "tools/external_gym_vllm", tools_dir)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "hosts").write_text(
        "ray-a 10.0.0.1\nray-b 10.0.0.2\n"
        "svc-a 10.0.1.1\nsvc-b 10.0.1.2\nsvc-c 10.0.1.3\nsvc-d 10.0.1.4\n"
    )
    _write_executable(
        bin_dir / "srun",
        """\
        #!/bin/bash
        printf '%s\\0' "$@" > "${FAKE_SLURM_DIR}/srun.$$"
        for argument in "$@"; do
          [[ "${argument}" == --export=* ]] || continue
          IFS=, read -r -a assignments <<< "${argument#--export=}"
          for assignment in "${assignments[@]}"; do
            case "${assignment}" in
              REPLICA_ID=*|EXTERNAL_VLLM_STATE_DIR=*|EXTERNAL_VLLM_GROUP_ID=*)
                export "${assignment}" ;;
            esac
          done
        done
        if [[ -n "${REPLICA_ID:-}" ]]; then
          source "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/vllm_backend_registry.sh"
          registry_add "${REPLICA_ID}" 10.9.9.9 8000
        fi
        exec "${FAKE_SLEEP}" 60
        """,
    )
    _write_executable(
        bin_dir / "scontrol",
        """\
        #!/bin/bash
        [[ "$1 $2" == "show hostnames" ]] && tr ',' '\\n' <<< "$3"
        """,
    )
    _write_executable(
        bin_dir / "getent",
        """\
        #!/bin/bash
        awk -v host="$2" '$1 == host { print $2 " STREAM " host }' "${FAKE_SLURM_DIR}/hosts"
        """,
    )
    _write_executable(bin_dir / "curl", "#!/bin/bash\nexit 0\n")
    _write_executable(bin_dir / "sleep", '#!/bin/bash\nexec "${FAKE_SLEEP}" 0.01\n')
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_SLURM_DIR": str(tmp_path),
        "FAKE_SLEEP": shutil.which("sleep"),
        "SLURM_JOB_ID": "4242",
        "BASE_LOG_DIR": str(tmp_path / "logs"),
        "CONTAINER": "nemo-rl.sqsh",
        "EXTERNAL_VLLM_SHARED_ROOT": str(tmp_path),
        "EXTERNAL_VLLM_TOOLS_DIR_HOST": str(tools_dir),
        "EXTERNAL_VLLM_POOLS": "GENRM JUDGE",
        "GENRM_MODEL": "genrm-model-id",
        "GENRM_CONTAINER": "vllm.sqsh",
        "GENRM_VLLM_PYTHON": "/opt/vllm/bin/python",
        "GENRM_REPLICAS": "2",
        "GENRM_TENSOR_PARALLEL_SIZE": "4",
        "GENRM_LB_PORT": "9213",
        "GENRM_URL_PLACEHOLDER": "__GENRM_URL__",
        "GENRM_SERVED_MODEL_NAME": "genrm",
        "JUDGE_MODEL": "judge-model-id",
        "JUDGE_CONTAINER": "vllm.sqsh",
        "JUDGE_VLLM_PYTHON": "/opt/vllm/bin/python",
        "JUDGE_REPLICAS": "1",
        "JUDGE_TENSOR_PARALLEL_SIZE": "8",
        "JUDGE_LB_PORT": "9214",
        "JUDGE_URL_PLACEHOLDER": "__JUDGE_URL__",
    }


def _recorded_srun_steps(tmp_path: Path) -> tuple[list[list[str]], list[list[str]]]:
    """Return the recorded (replica, load-balancer) srun argv lists."""
    steps = [
        path.read_text().rstrip("\0").split("\0")
        for path in sorted(tmp_path.glob("srun.*"))
    ]
    replica_steps = [
        step for step in steps if any("POOL_PREFIX=" in arg for arg in step)
    ]
    lb_steps = [step for step in steps if any("lb_watchdog.sh" in arg for arg in step)]
    assert len(replica_steps) + len(lb_steps) == len(steps)
    return replica_steps, lb_steps


def _parse_readiness_override(command: str) -> dict:
    """Apply a command's readiness override the way the NeMo RL driver does."""
    prefix = "++env.nemo_gym.external_service_readiness="
    overrides = [arg for arg in shlex.split(command) if arg.startswith(prefix)]
    assert len(overrides) == 1, command
    config = parse_hydra_overrides(
        OmegaConf.create({"env": {"nemo_gym": {}}}), overrides
    )
    return OmegaConf.to_container(config.env.nemo_gym.external_service_readiness)


def test_inline_job_routes_pools_and_gates_nemo_rl_on_their_readiness(tmp_path):
    log_dir = tmp_path / "logs/4242-logs"
    fake_ray_sub = tmp_path / "ray.sub"
    _write_executable(
        fake_ray_sub,
        """\
        #!/bin/bash
        printf '%s' "${COMMAND}" > "${FAKE_SLURM_DIR}/ray_sub_command"
        echo "${SLURM_JOB_NODELIST} ${SLURM_JOB_NUM_NODES}" > "${FAKE_SLURM_DIR}/ray_sub_nodes"
        # Like a real driver, outlive the wrapper's external readiness checks.
        for _ in $(seq 1 1000); do
          [[ -f "${FAKE_RAY_SUB_WAIT_FOR}" ]] && exit 0
          "${FAKE_SLEEP}" 0.01
        done
        exit 1
        """,
    )
    env = _fake_slurm_env(tmp_path)
    env.update(
        {
            "SLURM_HET_SIZE": "2",
            "SLURM_JOB_NODELIST_HET_GROUP_0": "ray-b,ray-a",
            "SLURM_JOB_NODELIST_HET_GROUP_1": "svc-a,svc-b,svc-c,svc-d",
            "SLURM_JOB_ACCOUNT": "account",
            "SLURM_JOB_PARTITION": "partition",
            "SLURM_SUBMIT_DIR": str(tmp_path),
            "MOUNTS": "/data:/data",
            "COMMAND": "run genrm=__GENRM_URL__ judge=__JUDGE_URL__",
            "RAY_SUB": str(fake_ray_sub),
            "FAKE_RAY_SUB_WAIT_FOR": str(log_dir / "judge_url"),
        }
    )

    result = subprocess.run(
        ["bash", str(tmp_path / "tools/run_in_allocation.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ray_sub_nodes").read_text() == "ray-b,ray-a 2\n"
    command = (tmp_path / "ray_sub_command").read_text()
    assert command.startswith(
        "run genrm=http://10.0.0.1:9213/v1 judge=http://10.0.0.1:9214/v1 "
    )
    assert _parse_readiness_override(command) == {
        "services": [
            {
                "name": "GENRM",
                "url": "http://10.0.0.1:9213/health",
                "expected_backends": 2,
            },
            {
                "name": "JUDGE",
                "url": "http://10.0.0.1:9214/health",
                "expected_backends": 1,
            },
        ],
        "timeout_seconds": 3600,
        "poll_interval_seconds": 5,
        "request_timeout_seconds": 10,
    }
    replica_steps, lb_steps = _recorded_srun_steps(tmp_path)
    assert sorted(
        arg for step in replica_steps for arg in step if arg.startswith("--nodelist=")
    ) == ["--nodelist=svc-a", "--nodelist=svc-b", "--nodelist=svc-c,svc-d"]
    for step in replica_steps:
        assert "--het-group=1" in step
        assert "-A" not in step and "-p" not in step
    assert len(lb_steps) == 2
    for step in lb_steps:
        assert "--het-group=0" in step
        assert step[step.index("-A") + 1] == "account"
        assert step[step.index("-p") + 1] == "partition"
        assert "--nodelist=ray-a" in step
        assert f"--container-workdir={tmp_path}" in step
        assert any(arg.startswith("--container-mounts=/data:/data,") for arg in step)
    assert not (log_dir / "external_vllm_services.tsv").exists()


def test_services_only_job_serves_pools_to_separately_submitted_nemo_rl(tmp_path):
    log_dir = tmp_path / "logs/4242-logs"
    manifest = log_dir / "external_vllm_services.tsv"
    env = _fake_slurm_env(tmp_path)
    # No hetgroups, COMMAND, MOUNTS, or ray.sub: the pools own the allocation.
    env.update(
        {
            "EXTERNAL_VLLM_SERVICES_ONLY": "1",
            "SLURM_JOB_NODELIST": "svc-d,svc-c,svc-b,svc-a",
        }
    )
    with open(tmp_path / "wrapper.log", "w") as wrapper_log:
        process = subprocess.Popen(
            ["bash", str(tmp_path / "tools/run_in_allocation.sh")],
            env=env,
            stdout=wrapper_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 30
        while (
            not manifest.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert manifest.exists(), (tmp_path / "wrapper.log").read_text()

        assert [
            line.split("\t")
            for line in manifest.read_text().splitlines()
            if not line.startswith("#")
        ] == [
            [
                "GENRM",
                "http://10.0.1.1:9213/v1",
                "http://10.0.1.1:9213/health",
                "2",
                "genrm",
            ],
            [
                "JUDGE",
                "http://10.0.1.1:9214/v1",
                "http://10.0.1.1:9214/health",
                "1",
                "model",
            ],
        ]
        replica_steps, lb_steps = _recorded_srun_steps(tmp_path)
        assert sorted(
            arg
            for step in replica_steps
            for arg in step
            if arg.startswith("--nodelist=")
        ) == ["--nodelist=svc-a", "--nodelist=svc-b", "--nodelist=svc-c,svc-d"]
        assert len(lb_steps) == 2
        for step in replica_steps + lb_steps:
            assert not any(arg.startswith("--het-group") for arg in step)
            assert "-A" not in step and "-p" not in step
        for step in lb_steps:
            assert "--nodelist=svc-a" in step
        assert not (tmp_path / "ray_sub_command").exists()

        # A NeMo RL launcher that uses only some of the pools attaches to them.
        attach = subprocess.run(
            [
                "bash",
                "-c",
                textwrap.dedent(
                    f"""\
                    set -euo pipefail
                    source {tmp_path}/tools/pool_config.sh
                    register_external_vllm_pool GENRM \\
                      --model genrm-model-id --container vllm.sqsh \\
                      --python /opt/vllm/bin/python --replicas 2 \\
                      --tensor-parallel-size 4 --lb-port 9213 \\
                      --served-model-name genrm --startup-timeout 600 \\
                      --url-placeholder __GENRM_URL__
                    resolve_external_vllm_services 'run genrm=__GENRM_URL__' {log_dir}
                    """
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert attach.returncode == 0, attach.stderr
        assert attach.stdout.startswith("run genrm=http://10.0.1.1:9213/v1 ")
        assert _parse_readiness_override(attach.stdout) == {
            "services": [
                {
                    "name": "GENRM",
                    "url": "http://10.0.1.1:9213/health",
                    "expected_backends": 2,
                }
            ],
            "timeout_seconds": 600,
            "poll_interval_seconds": 5,
            "request_timeout_seconds": 10,
        }

        # Stopping the job withdraws the manifest so no new job attaches.
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 143
        assert not manifest.exists()
        assert (log_dir / "ENDED").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        # Reap any fake srun steps the wrapper left behind.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_services_only_mode_rejects_a_heterogeneous_allocation():
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh")],
        env={
            "PATH": os.environ["PATH"],
            "SLURM_JOB_ID": "123",
            "SLURM_HET_SIZE": "2",
            "EXTERNAL_VLLM_SERVICES_ONLY": "1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires a single-component allocation, got 2 hetgroups" in result.stderr


_GENRM_SERVICE = "GENRM\thttp://10.0.1.1:9213/v1\thttp://10.0.1.1:9213/health"


@pytest.mark.parametrize(
    ("manifest", "expected_error"),
    [
        (None, "external_vllm_services.tsv does not exist"),
        (
            "JUDGE\thttp://10.0.1.1:9214/v1\thttp://10.0.1.1:9214/health\t1\tmodel\n",
            "do not include pool GENRM",
        ),
        (
            f"{_GENRM_SERVICE}\t2\tother\n",
            "pool GENRM serves model name 'other', but GENRM_SERVED_MODEL_NAME='genrm'",
        ),
        (
            "GENRM\t10.0.1.1:9213/v1\thttp://10.0.1.1:9213/health\t2\tgenrm\n",
            "malformed GENRM entry",
        ),
        (
            f"{_GENRM_SERVICE}\t0\tgenrm\n",
            "GENRM expected backends must be a positive integer",
        ),
    ],
)
def test_resolve_external_vllm_services_rejects_unusable_services(
    tmp_path, manifest, expected_error
):
    if manifest is not None:
        (tmp_path / "external_vllm_services.tsv").write_text(manifest)
    program = textwrap.dedent(
        f"""\
        source {REPO_ROOT}/tools/external_gym_vllm/pool_config.sh
        register_external_vllm_pool GENRM \\
          --model genrm-model-id --container vllm.sqsh \\
          --python /opt/vllm/bin/python --replicas 2 \\
          --tensor-parallel-size 4 --lb-port 9213 \\
          --served-model-name genrm --url-placeholder __GENRM_URL__
        resolve_external_vllm_services 'run genrm=__GENRM_URL__' {tmp_path}
        """
    )

    result = subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert result.stdout == ""


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


def test_lightning_launcher_dry_run_submits_only_the_external_pools():
    result = _run_lightning_launcher(EXTERNAL_VLLM_SERVICES_ONLY="1")

    assert result.returncode == 0, result.stderr
    assert "Job name:    lightning-launcher-test-services" in result.stdout
    assert "Nodes:       20 total" in result.stdout
    assert "Services only: 20 external-service nodes" in result.stdout
    assert "GenRM:    8 independent TP=8, DP=1 servers" in result.stdout
    assert "NL2Bash:  4 independent TP=4, DP=1 servers" in result.stdout
    assert "Hetgroup" not in result.stdout
    assert "Training:" not in result.stdout
    assert "--- TRAIN_CMD ---" not in result.stdout


def test_lightning_launcher_dry_run_attaches_to_running_services(tmp_path):
    (tmp_path / "external_vllm_services.tsv").write_text(
        "# External vLLM services from Slurm job 4242\n"
        "GENRM\thttp://10.0.1.1:9213/v1\thttp://10.0.1.1:9213/health\t8\tmodel\n"
        "NL2BASH\thttp://10.0.1.1:9214/v1\thttp://10.0.1.1:9214/health\t4\tmodel\n"
    )

    result = _run_lightning_launcher(EXTERNAL_VLLM_SERVICES_DIR=str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "Nodes:       66 total" in result.stdout
    assert "Hetgroup" not in result.stdout
    assert f"External services: {tmp_path}" in result.stdout
    batch_script_line = next(
        line for line in result.stdout.splitlines() if "Batch script:" in line
    )
    assert batch_script_line.endswith("/ray.sub")
    train_cmd = result.stdout.split("--- TRAIN_CMD ---\n")[1].split("\n--- end ---")[0]
    assert "__GENRM_BASE_URL__" not in train_cmd
    assert "__NL2BASH_BASE_URL__" not in train_cmd
    assert "genrm_model.base_url=http://10.0.1.1:9213/v1" in train_cmd
    assert "local_vllm_model.base_url=http://10.0.1.1:9214/v1" in train_cmd
    assert _parse_readiness_override(train_cmd)["services"] == [
        {
            "name": "GENRM",
            "url": "http://10.0.1.1:9213/health",
            "expected_backends": 8,
        },
        {
            "name": "NL2BASH",
            "url": "http://10.0.1.1:9214/health",
            "expected_backends": 4,
        },
    ]


def test_lightning_launcher_rejects_unusable_external_services_modes(tmp_path):
    missing = _run_lightning_launcher(EXTERNAL_VLLM_SERVICES_DIR=str(tmp_path))
    both = _run_lightning_launcher(
        EXTERNAL_VLLM_SERVICES_ONLY="1", EXTERNAL_VLLM_SERVICES_DIR=str(tmp_path)
    )

    assert missing.returncode == 2
    assert "external_vllm_services.tsv does not exist" in missing.stderr
    assert both.returncode == 2
    assert "mutually exclusive" in both.stderr


def _run_ultra_launcher(**overrides):
    launcher = REPO_ROOT / "examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh"
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        root = Path(temp_dir)
        env = {
            "HOME": str(root),
            "PATH": os.environ["PATH"],
            "DRY_RUN": "1",
            "USE_SNAPSHOT": "0",
            "EXP_NAME": "ultra-launcher-test",
            "CONFIG_PATH": "examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml",
            "MODEL_PATH": "test-policy-model",
            "TRAIN_PATH": str(root / "train.jsonl"),
            "VAL_PATH": str(root / "validation.jsonl"),
            "CONTAINER": "test-container",
            "SANDBOX_CONTAINER": "test-sandbox-container",
            "PERSISTENT_CACHE": str(root / "cache"),
            "RESULTS_DIR": str(root / "results"),
            "EXTERNAL_VLLM_SHARED_ROOT": str(REPO_ROOT),
            "SLURM_PARTITION": "test-partition",
            "SLURM_ACCOUNT": "test-account",
            "EXTERNAL_JUDGES": "1",
            "GENRM_MODEL": "test-genrm-model",
            "NL2BASH_JUDGE_MODEL": "test-nl2bash-model",
            "SAFETY_JUDGE_MODEL": "test-safety-model",
        }
        env.update(overrides)
        return subprocess.run(
            ["bash", str(launcher)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )


def test_ultra_launcher_dry_run_builds_inline_external_judges():
    result = _run_ultra_launcher()

    assert result.returncode == 0, result.stderr
    assert "Nodes:       264 total" in result.stdout
    assert "Hetgroup 0: 256 NeMo RL nodes" in result.stdout
    assert "Hetgroup 1: 8 external-service nodes" in result.stdout
    assert "genrm_model.base_url=__GENRM_BASE_URL__" in result.stdout
    assert "local_vllm_model.base_url=__NL2BASH_BASE_URL__" in result.stdout


def test_ultra_launcher_dry_run_submits_only_the_judge_pools():
    result = _run_ultra_launcher(EXTERNAL_VLLM_SERVICES_ONLY="1")

    assert result.returncode == 0, result.stderr
    assert "Job name:    ultra-launcher-test-services" in result.stdout
    assert "Nodes:       8 total" in result.stdout
    assert "Services only: 8 external-service nodes" in result.stdout
    assert "GenRM: 4 independent TP=4, DP=1 servers" in result.stdout
    assert "NL2Bash: 4 independent TP=4, DP=1 servers" in result.stdout
    assert "Hetgroup" not in result.stdout
    assert "Training:" not in result.stdout
    assert "--- TRAIN_CMD ---" not in result.stdout


def test_ultra_launcher_dry_run_attaches_to_running_services(tmp_path):
    # The services job may serve more pools than a stage uses.
    (tmp_path / "external_vllm_services.tsv").write_text(
        "GENRM\thttp://10.0.1.1:9213/v1\thttp://10.0.1.1:9213/health\t16\tmodel\n"
        "NL2BASH\thttp://10.0.1.1:9214/v1\thttp://10.0.1.1:9214/health\t4\tmodel\n"
    )

    result = _run_ultra_launcher(
        EXTERNAL_VLLM_SERVICES_DIR=str(tmp_path), NL2BASH_JUDGE_MODEL=""
    )

    assert result.returncode == 0, result.stderr
    assert "Nodes:       256 total" in result.stdout
    assert "Hetgroup" not in result.stdout
    assert f"External services: {tmp_path}" in result.stdout
    train_cmd = result.stdout.split("--- TRAIN_CMD ---\n")[1].split("\n--- end ---")[0]
    assert "genrm_model.base_url=http://10.0.1.1:9213/v1" in train_cmd
    assert "genrm_model.model=model" in train_cmd
    assert "nl2bash" not in train_cmd
    assert _parse_readiness_override(train_cmd)["services"] == [
        {
            "name": "GENRM",
            "url": "http://10.0.1.1:9213/health",
            "expected_backends": 16,
        }
    ]


def test_ultra_launcher_external_services_modes_require_external_judges(tmp_path):
    result = _run_ultra_launcher(
        EXTERNAL_JUDGES="0", EXTERNAL_VLLM_SERVICES_DIR=str(tmp_path)
    )

    assert result.returncode == 1
    assert "require EXTERNAL_JUDGES=1" in result.stderr
