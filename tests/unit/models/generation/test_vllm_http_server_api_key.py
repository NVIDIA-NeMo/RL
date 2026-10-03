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

"""The exposed vLLM HTTP server's per-job bearer token.

The server binds every interface of its node, so a recipe can require a bearer
token on the OpenAI routes: the driver generates one key per job
(http_server_api_key_and_worker_config), the workers refuse requests without it
(install_http_server_api_key_check), and setup hands the same key to the
NeMo-Gym model server as policy_api_key.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_rl.models.generation.vllm.vllm_generation import (
    http_server_api_key_and_worker_config,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    install_http_server_api_key_check,
)


def _config(**vllm_cfg):
    return {"model_name": "m", "vllm_cfg": {"async_engine": True, **vllm_cfg}}


def test_no_key_without_the_knob_or_without_the_server():
    """Off by default: the workers get the config as is and no key exists."""
    config = _config(expose_http_server=True)
    assert http_server_api_key_and_worker_config(config) == (None, config)
    # The knob alone does nothing when no HTTP server is exposed.
    config = _config(http_server_api_key_required=True)
    assert http_server_api_key_and_worker_config(config) == (None, config)


def test_key_travels_only_in_the_worker_copy():
    config = _config(expose_http_server=True, http_server_api_key_required=True)
    api_key, worker_config = http_server_api_key_and_worker_config(config)
    assert api_key and len(api_key) >= 32
    assert worker_config["vllm_cfg"]["http_server_api_key"] == api_key
    # The driver's config, which is saved and logged, never carries the key.
    assert "http_server_api_key" not in config["vllm_cfg"]
    assert worker_config["vllm_cfg"]["expose_http_server"] is True
    # Every job gets its own key.
    other_key, _ = http_server_api_key_and_worker_config(config)
    assert other_key != api_key


def _app_with_check(api_key: str) -> TestClient:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"ok": "chat"}

    @app.post("/tokenize")
    async def tokenize():
        return {"ok": "tokenize"}

    @app.post("/refit/prepare")
    async def refit():
        return {"ok": "refit"}

    install_http_server_api_key_check(app, api_key)
    return TestClient(app)


def test_openai_routes_require_the_exact_bearer_token():
    client = _app_with_check("s3cret")
    for path in ("/v1/chat/completions", "/tokenize"):
        missing = client.post(path)
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == 401
        wrong = client.post(path, headers={"Authorization": "Bearer other"})
        assert wrong.status_code == 401
        # The scheme is part of the comparison: a bare key is not a bearer token.
        bare = client.post(path, headers={"Authorization": "s3cret"})
        assert bare.status_code == 401
        right = client.post(path, headers={"Authorization": "Bearer s3cret"})
        assert right.status_code == 200


def test_other_routes_keep_their_own_authentication():
    """The sparse-refit routes carry their own header token and are not covered."""
    client = _app_with_check("s3cret")
    assert client.post("/refit/prepare").status_code == 200
