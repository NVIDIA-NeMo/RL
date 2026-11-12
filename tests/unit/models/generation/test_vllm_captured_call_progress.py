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

"""How far each running captured call got, read while paused for a refit."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    _CAPTURED_MODEL_CALL,
    _AsyncLLMHTTPClient,
)

pytestmark = pytest.mark.vllm


class _FakeEngine:
    """Records each request in vLLM's output-processor shape, then waits."""

    model_config = renderer = input_processor = vllm_config = None

    def __init__(self, generated_tokens: int) -> None:
        self.output_processor = SimpleNamespace(request_states={})
        self.release = asyncio.Event()
        self._generated_tokens = generated_tokens

    async def generate(self, prompt: Any, sampling_params: Any, request_id: str):
        # vLLM keys its state by an internal ID and keeps the caller's as
        # external_req_id.
        self.output_processor.request_states[f"internal-{request_id}"] = (
            SimpleNamespace(
                external_req_id=request_id,
                detokenizer=SimpleNamespace(
                    num_output_tokens=lambda: self._generated_tokens
                ),
            )
        )
        await self.release.wait()
        yield "output"

    def abort(self, request_id: str) -> None:
        pass


def test_running_captured_calls_report_their_generated_tokens() -> None:
    async def exercise() -> None:
        engine = _FakeEngine(generated_tokens=200)
        client = _AsyncLLMHTTPClient(
            engine, asyncio.get_running_loop(), flat_logprobs=False
        )
        sampling_params = SimpleNamespace(logprobs=None, prompt_logprobs=None)

        async def serve(model_call_id: Optional[str], request_id: str) -> None:
            # What _begin_request_capture does in the request's task.
            if model_call_id is not None:
                _CAPTURED_MODEL_CALL.set(model_call_id)
            async for _ in client.generate("prompt", sampling_params, request_id):
                pass

        requests = [
            asyncio.create_task(serve("call-1", "req-1")),
            # An uncaptured request, such as a validation call.
            asyncio.create_task(serve(None, "req-2")),
        ]
        while len(engine.output_processor.request_states) < 2:
            await asyncio.sleep(0)

        assert await client.captured_call_progress() == [["call-1", 200]]

        engine.release.set()
        await asyncio.gather(*requests)
        assert client._captured_requests == {}

    asyncio.run(exercise())


def test_an_engine_without_output_processor_state_reports_nothing() -> None:
    async def exercise() -> list:
        engine = SimpleNamespace(
            model_config=None, renderer=None, input_processor=None, vllm_config=None
        )
        client = _AsyncLLMHTTPClient(
            engine, asyncio.get_running_loop(), flat_logprobs=False
        )
        return await client.captured_call_progress()

    assert asyncio.run(exercise()) == []


def test_vllm_still_has_the_request_state_we_read() -> None:
    """The fakes above would pass a vLLM that renamed these; this checks the real ones.

    The read goes engine.output_processor.request_states[*].external_req_id and
    .detokenizer.num_output_tokens(), none of which is public API.
    """
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from vllm.v1.engine.output_processor import OutputProcessor, RequestState

    assert "self.output_processor" in inspect.getsource(AsyncLLM.__init__)
    assert "self.request_states" in inspect.getsource(OutputProcessor.__init__)
    params = inspect.signature(RequestState.__init__).parameters
    assert "external_req_id" in params and "detokenizer" in params
    assert callable(IncrementalDetokenizer.num_output_tokens)
