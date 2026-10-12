# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from fastapi.testclient import TestClient

from nemo_rl.models.generation.remote_vllm.token_capture_bridge import (
    RemoteVllmTokenCaptureBridge,
)


def _client():
    bridge = RemoteVllmTokenCaptureBridge(
        dp_client=object(), staging_partition="staging", auth_token="secret"
    )
    bridge.set_weight_version(7)
    records = []

    def stage(record):
        records.append(record)
        return SimpleNamespace(ok=True, staging_key=record.staging_key)

    bridge._sink = SimpleNamespace(stage_wire=stage)
    bridge._source = SimpleNamespace(fetch_prefix_token_ids=lambda chain: [10, 11])
    client = TestClient(bridge._build_app())
    return bridge, records, client


def test_bridge_requires_credentials_before_staging():
    _, records, client = _client()
    response = client.post("/v1/nemo-rl/token-capture/commit", json={})
    assert response.status_code == 401
    assert records == []


def test_bridge_stages_exact_tokens_before_returning_versioned_coords():
    _, records, client = _client()
    response = client.post(
        "/v1/nemo-rl/token-capture/commit",
        headers={"Authorization": "Bearer secret"},
        json={
            "admission": {"rollout_id": "r1", "model_call_id": "c1", "mode": "text"},
            "prefix_token_ids": [],
            "prompt_token_ids": [10, 11],
            "generated_token_ids": [20],
            "generated_logprobs": [-0.1],
        },
    )
    assert response.status_code == 200
    assert len(records) == 1
    assert records[0].token_ids_delta == [10, 11, 20]
    assert records[0].token_mask_delta == [0.0, 0.0, 1.0]
    coords = response.json()["ng_commit_coords"]
    assert coords["weight_version"] == 7
    assert coords["disposition"] == "staged"
    assert coords["digest"] == records[0].digest


def test_bridge_rejected_write_returns_failed_receipt():
    bridge, _, client = _client()
    bridge._sink = SimpleNamespace(
        stage_wire=lambda record: SimpleNamespace(ok=False, error="storage unavailable")
    )
    response = client.post(
        "/v1/nemo-rl/token-capture/commit",
        headers={"Authorization": "Bearer secret"},
        json={
            "admission": {"rollout_id": "r1", "model_call_id": "c1", "mode": "text"},
            "prompt_token_ids": [10],
            "generated_token_ids": [20],
            "generated_logprobs": [-0.1],
        },
    )
    assert response.status_code == 200
    assert response.json()["ng_commit_coords"]["disposition"] == "capture_failed"


def test_bridge_fetches_prefix_and_validates_version_rotation():
    _, _, client = _client()
    headers = {"Authorization": "Bearer secret"}
    response = client.post(
        "/v1/nemo-rl/token-capture/prefix",
        headers=headers,
        json={"staging_chain": ["r1/c1"]},
    )
    assert response.json() == {"prefix_token_ids": [10, 11]}
    assert (
        client.post(
            "/v1/nemo-rl/token-capture/weight-version",
            headers=headers,
            json={"weight_version": True},
        ).status_code
        == 400
    )
    assert client.post(
        "/v1/nemo-rl/token-capture/weight-version",
        headers=headers,
        json={"weight_version": 8},
    ).json() == {"weight_version": 8}
