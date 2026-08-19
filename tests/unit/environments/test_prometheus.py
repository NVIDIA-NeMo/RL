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

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from urllib.error import URLError

import pytest
from pydantic import ValidationError

from nemo_rl.environments.prometheus import (
    NemoGymPrometheusConfig,
    PrometheusRegistrationError,
    PrometheusRegistrationResult,
    PrometheusTargetStatus,
    register_prometheus_targets,
    resolve_run_id,
    target_from_base_url,
    wait_for_prometheus_target,
    write_prometheus_manifest,
)


def test_prometheus_config_rejects_reserved_labels_and_invalid_required_mode() -> None:
    with pytest.raises(ValidationError, match="must not override"):
        NemoGymPrometheusConfig(labels={"replica": "wrong"})

    with pytest.raises(ValidationError, match="required=true requires enabled=true"):
        NemoGymPrometheusConfig(required=True)

    with pytest.raises(ValidationError, match="target_lifecycle"):
        NemoGymPrometheusConfig(enabled=True, required=True)

    with pytest.raises(ValidationError, match="cover at least one scrape_interval"):
        NemoGymPrometheusConfig(
            enabled=True,
            required=True,
            target_lifecycle="dedicated",
            scrape_interval_s=10,
            initial_scrape_wait_s=9,
        )

    with pytest.raises(ValidationError, match="invalid Prometheus label"):
        NemoGymPrometheusConfig(labels={"bad-label": "value"})

    with pytest.raises(ValidationError, match="invalid Prometheus label"):
        NemoGymPrometheusConfig(labels={"__address__": "override"})


def test_target_from_base_url_strips_api_path_and_formats_ipv6() -> None:
    target = target_from_base_url(
        "http://[2001:db8::5]:8000/v1/",
        labels={"replica": "0", "component": "vllm_backend"},
    )

    assert target.address == "[2001:db8::5]:8000"
    assert target.metrics_url == "http://[2001:db8::5]:8000/metrics"
    assert dict(target.labels) == {
        "component": "vllm_backend",
        "replica": "0",
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
        "http://0.0.0.0:8000/v1",
        "http://localhost:8000/v1",
        "http://[::ffff:127.0.0.1]:8000/v1",
    ],
)
def test_target_from_base_url_rejects_non_remote_targets(base_url: str) -> None:
    with pytest.raises(ValueError, match="(loopback|remotely reachable)"):
        target_from_base_url(base_url, labels={"replica": "0"})


@pytest.mark.parametrize(
    "base_url, match",
    [
        ("http://user:secret@worker-0:8000/v1", "credentials"),
        ("http://worker-0:0/v1", "non-zero"),
    ],
)
def test_target_from_base_url_rejects_invalid_authority(
    base_url: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        target_from_base_url(base_url, labels={"replica": "0"})


def test_resolve_run_id_uses_explicit_then_environment_then_ray() -> None:
    with patch.dict(
        "os.environ",
        {"NEMO_RL_RUN_ID": "environment-run", "SLURM_JOB_ID": "123"},
        clear=True,
    ):
        assert (
            resolve_run_id(
                NemoGymPrometheusConfig(run_id="configured-run"),
                ray_job_id="abcd",
            )
            == "configured-run"
        )
        assert (
            resolve_run_id(NemoGymPrometheusConfig(), ray_job_id="abcd")
            == "environment-run"
        )

    with patch.dict("os.environ", {}, clear=True):
        assert (
            resolve_run_id(NemoGymPrometheusConfig(), ray_job_id="abcd") == "ray-abcd"
        )


def test_wait_for_prometheus_target_records_readiness() -> None:
    target = target_from_base_url(
        "http://worker-0:8000/v1",
        labels={"replica": "0"},
    )
    response = MagicMock(status=200)
    response.__enter__.return_value = response

    with (
        patch(
            "nemo_rl.environments.prometheus.urlopen",
            side_effect=[URLError("starting"), response],
        ) as urlopen,
        patch("nemo_rl.environments.prometheus.time.sleep") as sleep,
    ):
        status = wait_for_prometheus_target(
            target,
            timeout_s=10.0,
            poll_interval_s=0.25,
        )

    assert status == PrometheusTargetStatus(target=target, ready=True, error=None)
    assert urlopen.call_args_list == [
        call(target.metrics_url, timeout=0.25),
        call(target.metrics_url, timeout=0.25),
    ]
    sleep.assert_called_once_with(0.25)


def test_register_prometheus_targets_uses_rl_insight_contract() -> None:
    config = NemoGymPrometheusConfig(
        enabled=True,
        required=True,
        server_url="http://monitor:18080/",
        run_id="run-7",
        target_lifecycle="dedicated",
    )
    target = target_from_base_url(
        "http://worker-0:8000/v1",
        labels={"component": "vllm_backend", "replica": "0", "run_id": "run-7"},
    )
    status = PrometheusTargetStatus(target=target, ready=True, error=None)
    response = MagicMock(status=200)
    response.read.return_value = json.dumps(
        {
            "status": "ok",
            "prometheus_reloaded": True,
            "target_count": 1,
        }
    ).encode()
    response.__enter__.return_value = response

    with patch(
        "nemo_rl.environments.prometheus.urlopen",
        return_value=response,
    ) as urlopen:
        result = register_prometheus_targets(config, [status, status])

    assert result.status == "registered"
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://monitor:18080/api/v1/prometheus/targets"
    assert json.loads(request.data) == {
        "job_name": "nemo_rl_vllm",
        "targets": [
            {
                "target": "worker-0:8000",
                "labels": {
                    "component": "vllm_backend",
                    "replica": "0",
                    "run_id": "run-7",
                },
            }
        ],
    }
    assert urlopen.call_args.kwargs == {"timeout": 10.0}


def test_register_prometheus_targets_requires_confirmed_reload() -> None:
    config = NemoGymPrometheusConfig(
        enabled=True,
        server_url="http://monitor:18080",
        run_id="run-7",
        target_lifecycle="dedicated",
    )
    target = target_from_base_url(
        "http://worker-0:8000/v1",
        labels={"run_id": "run-7"},
    )
    response = MagicMock(status=200)
    response.read.return_value = json.dumps(
        {"status": "ok", "prometheus_reloaded": False}
    ).encode()
    response.__enter__.return_value = response

    with (
        patch("nemo_rl.environments.prometheus.urlopen", return_value=response),
        pytest.raises(PrometheusRegistrationError, match="did not reload"),
    ):
        register_prometheus_targets(
            config,
            [PrometheusTargetStatus(target=target, ready=True, error=None)],
        )


def test_write_prometheus_manifest_archives_targets_and_registration(
    tmp_path: Path,
) -> None:
    config = NemoGymPrometheusConfig(
        enabled=True,
        required=True,
        server_url="http://monitor:18080",
        run_id="run-7",
        target_lifecycle="dedicated",
    )
    target = target_from_base_url(
        "http://worker-0:8000/v1",
        labels={"component": "vllm_backend", "replica": "0", "run_id": "run-7"},
    )
    path = tmp_path / "monitoring" / "prometheus-targets.json"

    with patch(
        "nemo_rl.environments.prometheus.collect_monitoring_versions",
        return_value={"uv": "0.11.28", "vllm_router": "0.1.15"},
    ):
        write_prometheus_manifest(
            path,
            config=config,
            run_id="run-7",
            target_statuses=[
                PrometheusTargetStatus(target=target, ready=False, error="starting")
            ],
            registration=PrometheusRegistrationResult(
                status="failed",
                server_url="http://monitor:18080",
                response=None,
                error="reload failed",
            ),
            router_log_paths=["/logs/router.stdout.log", "/logs/router.stderr.log"],
            backend_log_paths={"0": "/logs/backends/replica-0.log"},
            model_call_capture_dir="/logs/model_call_capture",
        )

    manifest = json.loads(path.read_text())
    assert manifest["run_id"] == "run-7"
    assert manifest["registration"]["status"] == "failed"
    assert manifest["model_call_capture_dir"] == "/logs/model_call_capture"
    assert manifest["backend_log_paths"] == {"0": "/logs/backends/replica-0.log"}
    assert manifest["versions"]["uv"] == "0.11.28"
    assert manifest["monitoring_config"]["final_scrape_wait_s"] == 12.0
    assert manifest["targets"] == [
        {
            "address": "worker-0:8000",
            "labels": {
                "component": "vllm_backend",
                "replica": "0",
                "run_id": "run-7",
            },
            "metrics_url": "http://worker-0:8000/metrics",
            "readiness_error": "starting",
            "ready_at_registration": False,
        }
    ]
