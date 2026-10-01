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
import threading
import urllib.request
from typing import Any

import pytest
from pydantic import ValidationError
from ray import cloudpickle

from nemo_rl.models.generation.remote_vllm import (
    RemoteVllmClient,
    RemoteVllmGeneration,
    RemoteVllmServiceConfig,
    preflight_remote_vllm_service,
)


class _Response:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.status = 200
        self._payload = b"" if body is None else json.dumps(body).encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _config(**updates: Any) -> RemoteVllmServiceConfig:
    values: dict[str, Any] = {
        "base_url": "http://rollout.test:9210/v1",
        "served_model_name": "policy",
        "max_model_len": 4096,
        "refit": {"checkpoint_dir": "/shared/hf-exports"},
    }
    values.update(updates)
    return RemoteVllmServiceConfig.model_validate(values)


def test_remote_service_config_rejects_non_http_base_url() -> None:
    with pytest.raises(ValidationError, match="base_url must start"):
        _config(base_url="rollout.test:9210/v1")


def test_remote_service_config_rejects_non_http_control_base_url() -> None:
    with pytest.raises(ValidationError, match="control_base_url must start"):
        _config(control_base_url="rollout.test:9211/v1")


def test_remote_service_config_requires_shared_absolute_checkpoint_path() -> None:
    with pytest.raises(ValidationError, match="checkpoint_dir must be an absolute"):
        _config(refit={"checkpoint_dir": "relative/hf-exports"})


def test_preflight_uses_stock_vllm_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 5.0
        requested_urls.append(request.full_url)
        if request.full_url.endswith("/health"):
            return _Response()
        if request.full_url.endswith("/v1/models"):
            return _Response({"data": [{"id": "policy"}]})
        return _Response({"model_config": {"max_model_len": 4096}})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    info = preflight_remote_vllm_service(_config())

    assert info.model == "policy"
    assert info.dev_mode_control_api is True
    assert requested_urls == [
        "http://rollout.test:9210/health",
        "http://rollout.test:9210/v1/models",
        "http://rollout.test:9210/server_info",
    ]


def test_preflight_rejects_served_model_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _Response(),
            _Response({"data": [{"id": "another-model"}]}),
        ]
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: next(responses),
    )

    with pytest.raises(RuntimeError, match="model mismatch"):
        preflight_remote_vllm_service(_config())


def test_preflight_rejects_multiple_independent_lb_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            {"status": "ok", "healthy_backends": 2, "total_backends": 2}
        ),
    )

    with pytest.raises(RuntimeError, match="control-plane fan-out"):
        preflight_remote_vllm_service(_config())


def test_preflight_accepts_multiple_backends_with_control_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _Response(
                {
                    "status": "ok",
                    "healthy_backends": 4,
                    "total_backends": 4,
                    "control_fanout": True,
                }
            ),
            _Response({"data": [{"id": "policy"}]}),
            _Response({"model_config": {"max_model_len": 4096}}),
        ]
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: next(responses),
    )

    info = preflight_remote_vllm_service(_config())

    assert info.model == "policy"


def test_reload_uses_collective_rpc_with_weights_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 1800.0
        requests.append(request)
        return _Response({"result": [None, None]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    RemoteVllmClient(_config()).reload_weights("/shared/hf-exports/version_00000001")

    assert requests[0].full_url == "http://rollout.test:9210/collective_rpc"
    assert json.loads(requests[0].data or b"") == {
        "method": "reload_weights",
        "kwargs": {"weights_path": "/shared/hf-exports/version_00000001"},
    }


def test_generation_and_control_can_use_separate_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requested_urls.append(request.full_url)
        if request.full_url == "http://router.test:9210/health":
            return _Response()
        if request.full_url == "http://control.test:9211/health":
            return _Response({"total_backends": 4, "control_fanout": True})
        if request.full_url == "http://router.test:9210/v1/models":
            return _Response({"data": [{"id": "policy"}]})
        if request.full_url == "http://control.test:9211/server_info":
            return _Response({"model_config": {"max_model_len": 4096}})
        return _Response({"status": "ok"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    config = _config(
        base_url="http://router.test:9210/v1",
        control_base_url="http://control.test:9211/v1",
    )

    preflight_remote_vllm_service(config)
    RemoteVllmClient(config).reload_weights("/shared/hf-exports/version_00000001")

    assert requested_urls == [
        "http://router.test:9210/health",
        "http://control.test:9211/health",
        "http://router.test:9210/v1/models",
        "http://control.test:9211/server_info",
        "http://control.test:9211/collective_rpc",
    ]


def test_pause_keeps_inflight_requests_and_clears_their_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requests.append(request)
        return _Response({"status": "paused"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    config = _config()
    RemoteVllmClient(config).pause(
        mode=config.refit.inflight_policy,
        clear_cache=True,
    )

    assert requests[0].full_url == (
        "http://rollout.test:9210/pause?mode=keep&clear_cache=true"
    )


def test_token_capture_configuration_uses_control_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requests.append(request)
        return _Response({"status": "ok", "fanout_backends": 8})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    config = _config(control_base_url="http://control.test:9211/v1")

    RemoteVllmClient(config).configure_token_capture(
        bridge_url="http://10.0.0.1:12345",
        auth_token="secret",
    )

    assert requests[0].full_url == (
        "http://control.test:9211/v1/nemo-rl/token-capture/configure"
    )
    assert json.loads(requests[0].data or b"") == {
        "bridge_url": "http://10.0.0.1:12345",
        "auth_token": "secret",
    }


def test_generation_serializes_bridge_coordinates_not_live_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnpicklableBridge:
        def __init__(self) -> None:
            self.lock = threading.Lock()

    generation = RemoteVllmGeneration({"remote_vllm_cfg": _config().model_dump()})
    live_bridge = UnpicklableBridge()
    generation._token_capture_bridge = live_bridge  # type: ignore[assignment]
    generation._token_capture_bridge_url = "http://10.0.0.1:12345"
    generation._token_capture_auth_token = "secret"

    restored = cloudpickle.loads(cloudpickle.dumps(generation))

    assert generation._token_capture_bridge is live_bridge
    assert restored._token_capture_bridge is None
    assert restored._token_capture_bridge_url == "http://10.0.0.1:12345"
    assert restored._token_capture_auth_token == "secret"

    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 30.0
        requests.append(request)
        return _Response({"weight_version": 7})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    restored.set_rollout_weight_version(7)

    assert requests[0].full_url == (
        "http://10.0.0.1:12345/v1/nemo-rl/token-capture/weight-version"
    )
    assert requests[0].headers["Authorization"] == "Bearer secret"
    assert json.loads(requests[0].data or b"") == {"weight_version": 7}
