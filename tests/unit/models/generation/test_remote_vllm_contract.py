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
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.remote_vllm import (
    RemoteVllmClient,
    RemoteVllmGeneration,
    RemoteVllmServiceConfig,
    preflight_remote_vllm_service,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.hf_checkpoint_weight_synchronizer import (
    HfCheckpointWeightSynchronizer,
)


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


def test_external_backend_reuses_hf_transport_without_ray_inference_cluster(
    tmp_path: Path,
) -> None:
    generation = RemoteVllmGeneration(
        {
            "remote_vllm_cfg": _config().model_dump(),
            "refit_transport": "hf_checkpoint",
            "refit_cfg": {"hf_checkpoint": {"checkpoint_dir": str(tmp_path)}},
        }
    )
    policy = SimpleNamespace(cfg={"megatron_cfg": {"enabled": True}})
    synchronizer = create_weight_synchronizer(policy, generation, "remote_vllm", False)
    assert isinstance(synchronizer, HfCheckpointWeightSynchronizer)


def test_remote_service_config_rejects_unknown_refit_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(refit={"checkpoint_dir": "/shared/hf-exports"})


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
