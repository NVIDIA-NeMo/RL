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
"""A chat call's ``priority`` reaches vLLM through NeMo RL's HTTP server."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.generation.vllm.chat_server import NeMoRLChatCompletionRequest
from nemo_rl.models.generation.vllm.vllm_worker_async import _AsyncLLMHTTPClient

pytestmark = pytest.mark.vllm


@pytest.mark.parametrize("priority", [0, -3])
def test_chat_request_keeps_the_priority_gym_sends(priority: int) -> None:
    request = NeMoRLChatCompletionRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "x"}],
            "priority": priority,
        }
    )

    assert request.priority == priority


def test_http_engine_client_forwards_the_priority_to_the_engine() -> None:
    engine_calls: list[dict[str, Any]] = []

    async def engine_generate(prompt, sampling_params, request_id, **kwargs):
        engine_calls.append(kwargs)
        yield "output"

    engine_client = SimpleNamespace(
        generate=engine_generate,
        model_config=None,
        renderer=None,
        input_processor=None,
        vllm_config=None,
    )
    sampling_params = SimpleNamespace(logprobs=None, extra_args=None)

    async def run() -> list[Any]:
        client = _AsyncLLMHTTPClient(
            engine_client, asyncio.get_running_loop(), flat_logprobs=False
        )
        return [
            output
            async for output in client.generate(
                "prompt", sampling_params, "req-0", priority=-3
            )
        ]

    assert asyncio.run(run()) == ["output"]
    assert engine_calls == [{"priority": -3}]
