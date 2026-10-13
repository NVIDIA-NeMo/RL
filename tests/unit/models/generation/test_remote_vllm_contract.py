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
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError
from ray import cloudpickle

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.remote_vllm import (
    RemoteVllmClient,
    RemoteVllmGeneration,
    RemoteVllmServiceConfig,
    preflight_remote_vllm_service,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.remote_vllm_nccl_weight_synchronizer import (
    RemoteVllmNcclWeightSynchronizer,
)


@pytest.mark.parametrize(
    "base_url", ["http://rollout.test:9210/v1", "https://rollout.test/v1"]
)
def test_remote_base_url_accepts_openai_v1(base_url: str) -> None:
    assert _config(base_url=base_url).base_url == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "http://rollout.test:9210",
        "http://rollout.test:9210/",
        "http://rollout.test:9210/v1/",
        "http://rollout.test:9210/v2",
        "http://rollout.test:9210/prefix/v1",
        "http://rollout.test:9210/v1?key=value",
        "http://rollout.test:9210/v1#fragment",
    ],
)
def test_remote_base_url_rejects_incompatible_gym_paths(base_url: str) -> None:
    with pytest.raises(ValidationError, match="base_url must end in /v1"):
        _config(base_url=base_url)


def test_generation_config_builds_runtime_remote_service_schema() -> None:
    schema = TypeAdapter(GenerationConfig).json_schema()
    assert "RemoteVllmServiceConfig" in schema["$defs"]


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
        "refit": {},
    }
    values.update(updates)
    return RemoteVllmServiceConfig.model_validate(values)


def test_remote_service_config_rejects_non_http_base_url() -> None:
    with pytest.raises(ValidationError, match="base_url must start"):
        _config(base_url="rollout.test:9210/v1")


def test_external_backend_uses_nccl_without_ray_inference_cluster() -> None:
    generation = RemoteVllmGeneration({"remote_vllm_cfg": _config().model_dump()})
    policy = SimpleNamespace(
        cfg={"megatron_cfg": {"enabled": True}, "offload_policy_before_refit": False}
    )
    synchronizer = create_weight_synchronizer(
        policy, generation, "remote_vllm", False, train_cluster=object()
    )
    assert isinstance(synchronizer, RemoteVllmNcclWeightSynchronizer)
    synchronizer.shutdown()


def test_remote_service_config_rejects_unknown_refit_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(refit={"checkpoint_dir": "/shared/hf-exports"})


def test_gym_receives_external_service_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = RemoteVllmGeneration({"remote_vllm_cfg": _config().model_dump()})
    assert generation.http_server_api_key is None
    assert "Authorization" not in generation.client._headers()

    monkeypatch.setenv("TEST_EXTERNAL_VLLM_API_KEY", "test-service-key")
    config = _config(api_key_env_var="TEST_EXTERNAL_VLLM_API_KEY")
    generation = RemoteVllmGeneration({"remote_vllm_cfg": config.model_dump()})
    assert generation.http_server_api_key == "test-service-key"
    assert generation.client._headers()["Authorization"] == "Bearer test-service-key"
    assert "test-service-key" not in config.model_dump_json()


@pytest.mark.parametrize("value", [None, ""])
def test_gym_rejects_missing_external_service_api_key(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("TEST_EXTERNAL_VLLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TEST_EXTERNAL_VLLM_API_KEY", value)
    config = _config(api_key_env_var="TEST_EXTERNAL_VLLM_API_KEY")
    generation = RemoteVllmGeneration({"remote_vllm_cfg": config.model_dump()})
    with pytest.raises(RuntimeError, match="configured but unset or empty"):
        _ = generation.http_server_api_key


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
        if request.full_url.endswith("/version"):
            return _Response({"version": "0.29.0"})
        if request.full_url.endswith("/get_world_size"):
            return _Response({"world_size": 1})
        return _Response(
            {"vllm_config": {"weight_transfer_config": {"backend": "nccl"}}}
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    info = preflight_remote_vllm_service(_config())

    assert info.model == "policy"
    assert info.dev_mode_control_api is True
    assert requested_urls == [
        "http://rollout.test:9210/health",
        "http://rollout.test:9210/v1/models",
        "http://rollout.test:9210/version",
        "http://rollout.test:9210/server_info?config_format=json",
        "http://rollout.test:9210/get_world_size",
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

    with pytest.raises(RuntimeError, match="Control-plane fan-out"):
        preflight_remote_vllm_service(_config())


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


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"/version": {"version": "0.25.1"}}, "requires vLLM 0.29.0"),
        ({"/get_world_size": {"world_size": 2}}, "world size does not match"),
        (
            {
                "/server_info?config_format=json": {
                    "vllm_config": {"weight_transfer_config": None}
                }
            },
            "weight_transfer_config.backend=nccl",
        ),
        (
            {
                "/health": {
                    "total_backends": 2,
                    "healthy_backends": 2,
                    "control_fanout": True,
                }
            },
            "exactly one engine deployment",
        ),
        ({"/health": {"total_backends": "1"}}, "exactly one engine deployment"),
    ],
)
def test_preflight_rejects_incompatible_serving_before_nccl_init(
    monkeypatch, overrides, match
):
    responses = {
        "/health": None,
        "/v1/models": {"data": [{"id": "policy"}]},
        "/version": {"version": "0.29.0"},
        "/server_info?config_format=json": {
            "vllm_config": {"weight_transfer_config": {"backend": "nccl"}}
        },
        "/get_world_size": {"world_size": 1},
    }
    responses.update(overrides)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            responses[request.full_url.removeprefix("http://rollout.test:9210")]
        ),
    )
    with pytest.raises(RuntimeError, match=match):
        preflight_remote_vllm_service(_config())


def test_generation_refit_lifecycle_forwards_controller_deadlines() -> None:
    generation = RemoteVllmGeneration({"remote_vllm_cfg": _config().model_dump()})
    generation.client.pause = MagicMock()
    generation.client.resume = MagicMock()
    assert generation.pause_generation_for_refit(clear_cache=True, timeout_s=3.0)
    generation.client.pause.assert_called_once_with(
        mode="keep", clear_cache=True, timeout_s=3.0
    )
    assert generation.resume_generation_after_refit(timeout_s=4.0)
    generation.client.resume.assert_called_once_with(timeout_s=4.0)


def test_token_capture_configuration_uses_service_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requests.append(request)
        return _Response({"status": "ok", "fanout_backends": 8})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    config = _config()

    RemoteVllmClient(config).configure_token_capture(
        bridge_url="http://10.0.0.1:12345",
        auth_token="secret",
    )

    assert requests[0].full_url == (
        "http://rollout.test:9210/v1/nemo-rl/token-capture/configure"
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
