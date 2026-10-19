# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native context measurement reports the engine limit without token echo."""

import asyncio
import inspect
import json
import sys
from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from tests.unit.models.generation.test_vllm_generation import (
    _FakeFastAPIApp,
    _install_fake_vllm_openai_modules,
)

pytestmark = pytest.mark.nemo_gym


@pytest.mark.parametrize("captured", [False, True])
def test_native_measurement_returns_actual_limit_without_capture_token_echo(
    monkeypatch, captured
):
    _install_fake_vllm_openai_modules(monkeypatch)
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"temperature": 1.0, "top_p": 1.0, "vllm_cfg": {}}
    worker.llm = MagicMock(model_config="model-config", renderer="renderer")
    worker._http_engine_client = worker.llm
    worker._capture_calls = {}
    worker.token_capture = None
    worker.llm_async_engine_args = MagicMock()
    worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
        served_model_name="served-model", model="model-path"
    )
    app = _FakeFastAPIApp()
    worker._setup_vllm_openai_api_server(app)
    handler = next(handler for path, handler in app.routes if path == "/tokenize")
    request_type = next(
        kind
        for kind in get_args(
            inspect.signature(handler).parameters["request"].annotation
        )
        if kind.__name__ == "NeMoRLTokenizeChatRequest"
    )
    request = request_type()
    request.ng_capture = {"mode": "text"} if captured else None
    protocol = sys.modules["vllm.entrypoints.serve.tokenize.protocol"]
    result = protocol.TokenizeResponse()
    result.count = 282833
    result.max_model_len = 262144
    original = {
        "count": result.count,
        "max_model_len": result.max_model_len,
        "tokens": [18, 19],
    }
    result.model_dump = lambda: original
    serving = sys.modules[
        "vllm.entrypoints.serve.tokenize.serving"
    ].ServingTokenization.instances[0]
    serving.create_tokenize = AsyncMock(return_value=result)
    response = asyncio.run(handler(request, MagicMock()))
    actual = json.loads(response.body)
    if captured:
        assert actual == {
            "prompt_token_count": 282833,
            "max_model_len": 262144,
            "ng_context_measured": True,
        }
        assert "tokens" not in actual
    else:
        assert actual == original
    assert worker._capture_calls == {}
