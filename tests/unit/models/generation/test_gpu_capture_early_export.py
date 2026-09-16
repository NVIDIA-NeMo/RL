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

"""Real host and serving wrapper with CPU tensors and fake Ray/vLLM boundaries."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm import vllm_worker_async as serving_module
from tests.unit.models.generation import test_gpu_capture_host as fixtures
from tests.unit.models.generation import test_vllm_chat_template_wiring as wiring
from tests.unit.models.generation.test_gpu_capture_host import FakeObjectRef, _Case

case = fixtures.case
cpu_cuda = fixtures.cpu_cuda
pytestmark = [pytest.mark.nemo_gym, pytest.mark.asyncio]


def pending_reply(case: _Case):
    """Model Ray's synchronous submission separately from asynchronous response."""
    reply = FakeObjectRef()
    submissions = []
    if case.host._worker is not None:
        original = case.rpc.execute_method.remote

        def remote(method, *args):
            if method != "export_gpu_output_capture":
                return original(method, *args)
            submissions.append(args)
            case.rpc.calls.append((method, args))
            case.rpc.direct_calls.append(method)
            return reply

        case.rpc.execute_method.remote = remote
    else:

        async def collective(*args):
            submissions.append(args)
            return [await reply]

        case.rpc.responses["export_gpu_output_capture"] = collective
    return reply, submissions


async def test_start_submits_before_formatter_and_bind_exports_once(case, cpu_cuda):
    reply, submissions = pending_reply(case)
    case.host.start_export(case.state, generated_token_count=1)
    pending = case.state.export_task
    assert pending is not None and not pending.done()
    # No event-loop yield occurs before this assertion. Ray must already submit.
    assert len(submissions) == (1 if case.host._worker is not None else 0)
    assert reply.future_calls == (1 if case.host._worker is not None else 0)
    case.host.start_export(case.state, generated_token_count=1)
    assert case.state.export_task is pending
    assert not case.imports
    reply.set_result(case.lease)
    await case.bind()
    assert len(submissions) == 1 and reply.future_calls == 1
    assert case.state.export_task is None and not case.imports
    assert (await case.finish()).ok
    case.assert_released()


@pytest.mark.parametrize("export_fails", [False, True])
async def test_formatter_cancellation_drains_pending_before_cleanup(
    case, cpu_cuda, export_fails
):
    reply, submissions = pending_reply(case)
    formatting = asyncio.Event()

    async def endpoint():
        try:
            case.host.start_export(case.state, generated_token_count=1)
            formatting.set()
            await asyncio.Event().wait()
        finally:
            await case.host.release(case.state)

    task = asyncio.create_task(endpoint())
    await asyncio.wait_for(formatting.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not reply.cancelled()
    assert all(method == "export_gpu_output_capture" for method, _ in case.rpc.calls)
    if export_fails:
        reply.set_exception(RuntimeError("native export failed"))
    else:
        reply.set_result(case.lease)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(submissions) == 1 and not case.imports
    assert case.state.export_task is None
    if export_fails:
        assert case.rpc.calls[-1] == ("discard_gpu_output_capture", ("call",))
        assert not cpu_cuda
    else:
        case.assert_released("abandon_unimported_gpu_output_capture")
        assert cpu_cuda == ["sync"]


async def test_abort_before_any_formatter_yield_adopts_and_abandons(case, cpu_cuda):
    reply, submissions = pending_reply(case)
    case.host.start_export(case.state, generated_token_count=1)
    reply.set_result(case.lease)
    await case.host.release(case.state)
    assert len(submissions) == 1 and reply.future_calls == 1 and not case.imports
    assert case.state.export_task is None
    case.assert_released("abandon_unimported_gpu_output_capture")


def build_serving(case: _Case, monkeypatch: pytest.MonkeyPatch, formatter):
    """Execute actual setup/mixin/endpoint methods using the existing vLLM fixture."""
    wiring._install_fake_vllm(monkeypatch)
    worker = serving_module.VllmAsyncGenerationWorkerImpl.__new__(
        serving_module.VllmAsyncGenerationWorkerImpl
    )
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "vllm_cfg": {},
    }
    worker._http_engine_client = SimpleNamespace(
        model_config="model", renderer="renderer"
    )
    worker.llm_async_engine_args = SimpleNamespace(
        enable_return_routed_experts=False,
        create_model_config=lambda: SimpleNamespace(
            served_model_name="model", model="model"
        ),
    )
    worker._gpu_capture_host = case.host
    worker.token_capture = object()
    case.state.capture = SimpleNamespace(fail_call=MagicMock())
    request = SimpleNamespace(
        top_k=None,
        top_p=1.0,
        temperature=1.0,
        return_tokens_as_token_ids=False,
        logprobs=False,
        stream=False,
    )
    worker._capture_calls = {id(request): case.state}
    monkeypatch.setattr(
        wiring._OpenAIServingChat,
        "chat_completion_full_generator",
        formatter,
        raising=False,
    )
    app = wiring._FakeApp()
    worker._setup_vllm_openai_api_server(app)
    serving = wiring._BUILT["chat"][0]
    endpoint = dict(app.routes)["/v1/chat/completions"]
    return worker, request, serving, endpoint


async def outputs():
    yield SimpleNamespace(finished=False, outputs=[SimpleNamespace(token_ids=[])])
    yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=[21])])


async def test_real_serving_submits_only_final_before_upstream_formatter(
    case, cpu_cuda, monkeypatch
):
    reply, submissions = pending_reply(case)
    response = None

    async def formatter(self, request, result_generator, *args, **kwargs):
        seen = 0
        async for result in result_generator:
            seen += 1
            if not result.finished:
                assert not submissions and case.state.export_task is None
            else:
                assert len(submissions) == (1 if case.host._worker is not None else 0)
                assert case.state.export_task is not None
                reply.set_result(case.lease)
        assert seen == 2
        return response

    _worker, request, serving, _endpoint = build_serving(case, monkeypatch, formatter)
    response = sys.modules[
        "vllm.entrypoints.openai.chat_completion.protocol"
    ].ChatCompletionResponse()
    assert await serving.chat_completion_full_generator(request, outputs()) is response
    assert len(submissions) == 1 and not case.imports
    assert case.state.export_task is None and case.state.prepare_payload is not None
    assert (await case.finish()).ok
    case.assert_released()


@pytest.mark.parametrize(
    "failure", ["error_response", "formatter", "serializer", "cancel"]
)
async def test_actual_endpoint_aborts_pending_export_on_all_formatting_exits(
    case, cpu_cuda, monkeypatch, failure
):
    reply, submissions = pending_reply(case)
    formatting = asyncio.Event()
    response = None

    async def formatter(self, request, result_generator, *args, **kwargs):
        async for result in result_generator:
            if result.finished:
                assert case.state.export_task is not None
                formatting.set()
                if failure == "formatter":
                    raise RuntimeError("formatter failed")
                if failure == "cancel":
                    await asyncio.Event().wait()
        return response

    worker, request, serving, endpoint = build_serving(case, monkeypatch, formatter)
    if failure == "error_response":
        response = sys.modules[
            "vllm.entrypoints.openai.engine.protocol"
        ].ErrorResponse()
        response.error = SimpleNamespace(code=400)
        response.model_dump = lambda: {"error": "request rejected"}
    else:
        response = sys.modules[
            "vllm.entrypoints.openai.chat_completion.protocol"
        ].ChatCompletionResponse()

    async def create_chat_completion(_request, _raw_request):
        return await serving.chat_completion_full_generator(_request, outputs())

    monkeypatch.setattr(
        serving, "create_chat_completion", create_chat_completion, raising=False
    )
    if failure == "serializer":

        def serialize(_response):
            raise RuntimeError("serializer failed")

        monkeypatch.setattr(
            serving_module,
            "model_dump_chat_response_with_dynamic_message_fields",
            serialize,
        )

    task = asyncio.create_task(endpoint(request, None))
    await asyncio.wait_for(formatting.wait(), 5)
    if failure == "cancel":
        task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not reply.cancelled()
    reply.set_result(case.lease)
    if failure == "error_response":
        assert (await task).status_code == 400
    elif failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            await task
    assert len(submissions) == 1 and not case.imports
    assert case.state.export_task is None and not worker._capture_calls
    assert case.state.capture.fail_call.call_count == 1
    case.assert_released("abandon_unimported_gpu_output_capture")
