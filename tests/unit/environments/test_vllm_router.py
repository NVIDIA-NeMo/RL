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

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from urllib.error import URLError

import pytest
from pydantic import ValidationError

from nemo_rl.environments.vllm_router import (
    VllmRouterConfig,
    VllmRouterProcess,
    _parse_router_cache_metrics,
    _probe_worker_health,
    _RouterMetricsProxy,
)


def test_router_config_rejects_unknown_policy() -> None:
    with pytest.raises(ValidationError, match="policy"):
        VllmRouterConfig(enabled=True, policy="typo")


def test_router_config_validates_cache_metrics_compatibility() -> None:
    with pytest.raises(ValidationError, match="requires policy=cache_aware"):
        VllmRouterConfig(
            enabled=True,
            policy="consistent_hash",
            cache_metrics_mode="debug_log_compat",
        )
    with pytest.raises(ValidationError, match="cache_threshold"):
        VllmRouterConfig(
            enabled=True,
            policy="cache_aware",
            cache_threshold=1.1,
        )


def test_reconstructs_cache_decisions_using_router_f32_threshold(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "router.stdout.log"
    log_path.write_text(
        "Cache match for model 'm': matched_chars=3, input_chars=10, "
        "match_rate=0.30\n"
        "Cache match for model 'm': matched_chars=4, input_chars=10, "
        "match_rate=0.40\n"
        "Cache match for model 'm': matched_chars=0, input_chars=0, "
        "match_rate=0.00\n"
        "Cache match for model 'm': matched_chars=9, input_chars=10, ",
        encoding="utf-8",
    )

    metrics = _parse_router_cache_metrics(log_path, cache_threshold=0.3)

    assert metrics.hits == 1
    assert metrics.misses == 2
    assert metrics.observations == 3
    assert metrics.malformed_lines == 0


def test_cache_metrics_proxy_preserves_native_metrics_and_adds_missing_pair(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "router.stdout.log"
    log_path.write_text(
        "Cache match for model 'm': matched_chars=8, input_chars=10, "
        "match_rate=0.80\n"
        "Cache match for model 'm': matched_chars=1, input_chars=10, "
        "match_rate=0.10\n",
        encoding="utf-8",
    )
    native_response = MagicMock()
    native_response.status = 200
    native_response.read.return_value = (
        b"# TYPE vllm_router_requests_total counter\nvllm_router_requests_total 2\n"
    )
    native_response.__enter__.return_value = native_response
    proxy = _RouterMetricsProxy(
        native_metrics_url="http://127.0.0.1:6601/metrics",
        router_stdout_log_path=log_path,
        cache_threshold=0.3,
        worker_base_urls=("http://worker-0:8000",),
        policy="cache_aware",
        cache_metrics_mode="debug_log_compat",
    )

    with patch(
        "nemo_rl.environments.vllm_router.urlopen",
        return_value=native_response,
    ):
        rendered = proxy.render().decode()

    assert "vllm_router_requests_total 2" in rendered
    assert "vllm_router_cache_hits_total 1" in rendered
    assert "vllm_router_cache_misses_total 1" in rendered
    assert "nemo_rl_vllm_router_cache_log_observations_total 2" in rendered
    assert (
        'nemo_rl_vllm_router_cache_metrics_info{source="debug_log_compat"} 1'
        in rendered
    )
    assert "nemo_rl_vllm_router_cache_threshold 0.3" in rendered


def test_cache_metrics_proxy_does_not_duplicate_native_cache_counters(
    tmp_path: Path,
) -> None:
    native_response = MagicMock()
    native_response.status = 200
    native_response.read.return_value = (
        b"# TYPE vllm_router_cache_hits_total counter\n"
        b"vllm_router_cache_hits_total 7\n"
        b"# TYPE vllm_router_cache_misses_total counter\n"
        b"vllm_router_cache_misses_total 3\n"
    )
    native_response.__enter__.return_value = native_response
    proxy = _RouterMetricsProxy(
        native_metrics_url="http://127.0.0.1:6601/metrics",
        router_stdout_log_path=tmp_path / "does-not-exist.log",
        cache_threshold=0.3,
        worker_base_urls=("http://worker-0:8000",),
        policy="cache_aware",
        cache_metrics_mode="debug_log_compat",
    )

    with patch(
        "nemo_rl.environments.vllm_router.urlopen",
        return_value=native_response,
    ):
        rendered = proxy.render().decode()

    assert rendered.count("vllm_router_cache_hits_total 7") == 1
    assert rendered.count("vllm_router_cache_misses_total 3") == 1
    assert 'nemo_rl_vllm_router_cache_metrics_info{source="native"} 1' in rendered
    assert "nemo_rl_vllm_router_cache_log_observations_total" not in rendered


def test_cache_metrics_proxy_rejects_partial_native_counter_pair(
    tmp_path: Path,
) -> None:
    native_response = MagicMock()
    native_response.status = 200
    native_response.read.return_value = b"vllm_router_cache_hits_total 7\n"
    native_response.__enter__.return_value = native_response
    proxy = _RouterMetricsProxy(
        native_metrics_url="http://127.0.0.1:6601/metrics",
        router_stdout_log_path=tmp_path / "router.stdout.log",
        cache_threshold=0.3,
        worker_base_urls=("http://worker-0:8000",),
        policy="cache_aware",
        cache_metrics_mode="debug_log_compat",
    )

    with (
        patch(
            "nemo_rl.environments.vllm_router.urlopen",
            return_value=native_response,
        ),
        pytest.raises(RuntimeError, match="only one cache counter"),
    ):
        proxy.render()


def test_cache_metrics_proxy_rejects_malformed_decision_log(tmp_path: Path) -> None:
    log_path = tmp_path / "router.stdout.log"
    log_path.write_text(
        "Cache match for model 'm': matched_chars=invalid, input_chars=10, "
        "match_rate=0.00\n",
        encoding="utf-8",
    )
    native_response = MagicMock()
    native_response.status = 200
    native_response.read.return_value = b"vllm_router_requests_total 1\n"
    native_response.__enter__.return_value = native_response
    proxy = _RouterMetricsProxy(
        native_metrics_url="http://127.0.0.1:6601/metrics",
        router_stdout_log_path=log_path,
        cache_threshold=0.3,
        worker_base_urls=("http://worker-0:8000",),
        policy="cache_aware",
        cache_metrics_mode="debug_log_compat",
    )

    with (
        patch(
            "nemo_rl.environments.vllm_router.urlopen",
            return_value=native_response,
        ),
        pytest.raises(RuntimeError, match="failed to parse 1"),
    ):
        proxy.render()


def test_metrics_proxy_stabilizes_lazy_operational_metrics(tmp_path: Path) -> None:
    native_response = MagicMock()
    native_response.status = 200
    native_response.read.return_value = (
        b"vllm_router_active_workers 2\n"
        b'vllm_router_requests_total{route="/v1/chat/completions"} 3\n'
        b'vllm_router_policy_decisions_total{policy="consistent_hash",worker="http://worker-0:8000"} 2\n'
        b'vllm_router_policy_decisions_total{policy="consistent_hash",worker="http://worker-1:8000"} 1\n'
        b'vllm_router_processed_requests_total{worker="http://worker-0:8000"} 2\n'
        b'vllm_router_processed_requests_total{worker="http://worker-1:8000"} 1\n'
        b'vllm_router_cb_outcomes_total{worker="http://worker-0:8000",outcome="success"} 2\n'
        b'vllm_router_cb_outcomes_total{worker="http://worker-1:8000",outcome="success"} 1\n'
    )
    native_response.__enter__.return_value = native_response
    proxy = _RouterMetricsProxy(
        native_metrics_url="http://127.0.0.1:6601/metrics",
        router_stdout_log_path=tmp_path / "does-not-exist.log",
        cache_threshold=0.3,
        worker_base_urls=("http://worker-0:8000", "http://worker-1:8000"),
        policy="consistent_hash",
        cache_metrics_mode="native",
    )

    with patch(
        "nemo_rl.environments.vllm_router.urlopen",
        return_value=native_response,
    ):
        rendered = proxy.render().decode()

    assert (
        'nemo_rl_vllm_router_metrics_adapter_info{policy="consistent_hash",source="native_aggregate_compat"} 1'
        in rendered
    )
    assert "nemo_rl_vllm_router_requests_total 3" in rendered
    assert "nemo_rl_vllm_router_request_errors_total 0" in rendered
    assert "nemo_rl_vllm_router_retries_total 0" in rendered
    assert (
        'nemo_rl_vllm_router_native_metric_present{metric="vllm_router_request_errors_total"} 0'
        in rendered
    )
    assert (
        'nemo_rl_vllm_router_policy_decisions_total{worker="http://worker-0:8000"} 2'
        in rendered
    )
    assert (
        'nemo_rl_vllm_router_cb_outcomes_total{outcome="failure",worker="http://worker-1:8000"} 0'
        in rendered
    )
    assert (
        'nemo_rl_vllm_router_worker_health{worker="http://worker-0:8000"} 1' in rendered
    )
    assert (
        'nemo_rl_vllm_router_worker_health_source_info{source="adapter_backend_health_probe"} 1'
        in rendered
    )


def test_worker_health_probe_reports_unreachable_backend() -> None:
    with patch(
        "nemo_rl.environments.vllm_router.urlopen",
        side_effect=URLError("unreachable"),
    ):
        assert _probe_worker_health("http://worker-0:8000", timeout_s=0.1) == 0


def test_metrics_proxy_does_not_mask_partial_native_worker_health(
    tmp_path: Path,
) -> None:
    native_response = MagicMock()
    native_response.status = 200
    native_response.read.return_value = (
        b"vllm_router_active_workers 2\n"
        b'vllm_router_worker_health{worker="http://worker-0:8000"} 0\n'
    )
    native_response.__enter__.return_value = native_response
    proxy = _RouterMetricsProxy(
        native_metrics_url="http://127.0.0.1:6601/metrics",
        router_stdout_log_path=tmp_path / "does-not-exist.log",
        cache_threshold=0.3,
        worker_base_urls=("http://worker-0:8000", "http://worker-1:8000"),
        policy="consistent_hash",
        cache_metrics_mode="native",
    )

    with patch(
        "nemo_rl.environments.vllm_router.urlopen",
        return_value=native_response,
    ):
        rendered = proxy.render().decode()

    assert (
        'nemo_rl_vllm_router_worker_health{worker="http://worker-0:8000"} 0' in rendered
    )
    assert (
        'nemo_rl_vllm_router_worker_health{worker="http://worker-1:8000"} 1' in rendered
    )
    assert (
        'nemo_rl_vllm_router_worker_health_source_info{source="partial_native_and_adapter_probe"} 1'
        in rendered
    )


def test_builds_cache_metrics_compatibility_command(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        native_prometheus_port=6601,
        config=VllmRouterConfig(
            enabled=True,
            policy="cache_aware",
            cache_threshold=0.3,
            cache_metrics_mode="debug_log_compat",
        ),
        log_dir=tmp_path,
    )

    assert router.command[-8:] == [
        "--prometheus-port",
        "6601",
        "--prometheus-host",
        "127.0.0.1",
        "--cache-threshold",
        "0.3",
        "--log-level",
        "debug",
    ]
    assert router.metrics_url == "http://10.0.0.5:6600/metrics"
    assert router.native_metrics_url == "http://127.0.0.1:6601/metrics"


def test_starts_and_stops_cache_metrics_compatibility_proxy(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        native_prometheus_port=6601,
        config=VllmRouterConfig(
            enabled=True,
            policy="cache_aware",
            cache_metrics_mode="debug_log_compat",
        ),
        log_dir=tmp_path,
    )
    process = MagicMock()
    process.poll.return_value = None
    server = MagicMock()
    thread = MagicMock()
    thread.is_alive.return_value = True

    with (
        patch(
            "nemo_rl.environments.vllm_router.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "nemo_rl.environments.vllm_router.ThreadingHTTPServer",
            return_value=server,
        ) as server_type,
        patch(
            "nemo_rl.environments.vllm_router.threading.Thread",
            return_value=thread,
        ) as thread_type,
    ):
        router.start()
        router.stop(timeout=2.0)

    assert server_type.call_args.args[0] == ("10.0.0.5", 6600)
    assert server_type.call_args.args[1].__name__ == "RouterMetricsHandler"
    assert thread_type.call_args.kwargs == {
        "target": server.serve_forever,
        "kwargs": {"poll_interval": 0.1},
        "name": "vllm-router-metrics-proxy",
        "daemon": True,
    }
    thread.start.assert_called_once_with()
    server.shutdown.assert_called_once_with()
    server.server_close.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=5.0)
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2.0)


def test_builds_static_router_command_and_openai_base_url(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=[
            "http://worker-0:8000/v1",
            "http://worker-1:8001/v1/",
        ],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=VllmRouterConfig(enabled=True),
        log_dir=tmp_path,
    )

    assert router.command == [
        sys.executable,
        "-m",
        "vllm_router.launch_router",
        "--worker-urls",
        "http://worker-0:8000",
        "http://worker-1:8001",
        "--policy",
        "consistent_hash",
        "--host",
        "10.0.0.5",
        "--port",
        "6100",
        "--prometheus-port",
        "6600",
        "--prometheus-host",
        "10.0.0.5",
    ]
    assert router.openai_base_url == "http://10.0.0.5:6100/v1"
    assert router.readiness_url == "http://10.0.0.5:6100/readiness"
    assert router.metrics_url == "http://10.0.0.5:6600/metrics"
    assert router.log_paths == [
        str(tmp_path / "router.stdout.log"),
        str(tmp_path / "router.stderr.log"),
    ]


def test_starts_and_stops_owned_router_process(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=VllmRouterConfig(enabled=True),
        log_dir=tmp_path,
    )
    process = MagicMock()
    process.poll.return_value = None

    with patch(
        "nemo_rl.environments.vllm_router.subprocess.Popen",
        return_value=process,
    ) as popen:
        router.start()
        popen.assert_called_once()
        assert popen.call_args.args == (router.command,)
        assert popen.call_args.kwargs["stdout"].name == str(
            tmp_path / "router.stdout.log"
        )
        assert popen.call_args.kwargs["stderr"].name == str(
            tmp_path / "router.stderr.log"
        )

        router.stop(timeout=2.0)
        router.stop(timeout=2.0)

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2.0)
    process.kill.assert_not_called()


def test_force_kills_router_process_when_shutdown_times_out(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=VllmRouterConfig(enabled=True),
        log_dir=tmp_path,
    )
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(router.command, 2.0),
        0,
    ]

    with patch(
        "nemo_rl.environments.vllm_router.subprocess.Popen",
        return_value=process,
    ):
        router.start()
        router.stop(timeout=2.0)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [
        call(timeout=2.0),
        call(),
    ]


def test_waits_until_router_is_ready(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=VllmRouterConfig(enabled=True),
        log_dir=tmp_path,
    )
    process = MagicMock()
    process.poll.return_value = None

    ready_response = MagicMock()
    ready_response.status = 200
    ready_response.__enter__.return_value = ready_response

    with (
        patch(
            "nemo_rl.environments.vllm_router.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "nemo_rl.environments.vllm_router.urlopen",
            side_effect=[URLError("not ready"), ready_response],
        ) as urlopen,
        patch("nemo_rl.environments.vllm_router.time.sleep") as sleep,
    ):
        router.start()
        router.wait_until_ready(timeout=10.0, poll_interval=0.25)

    assert urlopen.call_args_list == [
        call(router.readiness_url, timeout=0.25),
        call(router.readiness_url, timeout=0.25),
    ]
    sleep.assert_called_once_with(0.25)
    router.stop()


def test_waits_for_prometheus_endpoint_independently(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=VllmRouterConfig(enabled=True),
        log_dir=tmp_path,
    )
    process = MagicMock()
    process.poll.return_value = None
    ready_response = MagicMock()
    ready_response.status = 200
    ready_response.__enter__.return_value = ready_response

    with (
        patch(
            "nemo_rl.environments.vllm_router.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "nemo_rl.environments.vllm_router.urlopen",
            return_value=ready_response,
        ) as urlopen,
    ):
        router.start()
        router.wait_until_metrics_ready(timeout=10.0, poll_interval=0.25)

    urlopen.assert_called_once_with(router.metrics_url, timeout=0.25)
    router.stop()


def test_readiness_fails_when_router_process_exits(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=VllmRouterConfig(enabled=True),
        log_dir=tmp_path,
    )
    process = MagicMock()
    process.poll.return_value = 17

    with (
        patch(
            "nemo_rl.environments.vllm_router.subprocess.Popen",
            return_value=process,
        ),
        patch("nemo_rl.environments.vllm_router.urlopen") as urlopen,
    ):
        router.start()
        with pytest.raises(RuntimeError, match="exited with code 17"):
            router.wait_until_ready(timeout=10.0, poll_interval=0.25)

    urlopen.assert_not_called()


def test_readiness_times_out(tmp_path: Path) -> None:
    router = VllmRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=VllmRouterConfig(enabled=True),
        log_dir=tmp_path,
    )
    process = MagicMock()
    process.poll.return_value = None

    with (
        patch(
            "nemo_rl.environments.vllm_router.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "nemo_rl.environments.vllm_router.urlopen",
            side_effect=URLError("not ready"),
        ),
        patch("nemo_rl.environments.vllm_router.time.sleep") as sleep,
    ):
        router.start()
        with pytest.raises(TimeoutError, match="did not become ready"):
            router.wait_until_ready(timeout=0.0, poll_interval=0.25)

    sleep.assert_not_called()
    router.stop()
