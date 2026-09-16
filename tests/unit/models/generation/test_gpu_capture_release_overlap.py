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

"""Post-fence release submission, response construction, and ACK ownership."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import sys
import threading
from typing import Any

import fastapi.responses
import pytest
from starlette.responses import JSONResponse
import torch

from tests.unit.models.generation import test_gpu_capture_early_export as early
from tests.unit.models.generation import test_gpu_capture_host as fixtures
from tests.unit.models.generation.test_gpu_capture_host import (
    FakeObjectRef,
    _Case,
    hosting,
)

case = fixtures.case
cpu_cuda = fixtures.cpu_cuda
pytestmark = [
    pytest.mark.nemo_gym,
    pytest.mark.asyncio,
    pytest.mark.parametrize("case", ["ray"], indirect=True),
]
CONTENT = {
    "choices": [
        {"routed_experts": "unchanged upstream NPY / é", "message": {"content": "done"}}
    ]
}


def pending_release(case: _Case, events: list[str]) -> FakeObjectRef:
    reply = FakeObjectRef()
    original = case.rpc.execute_method.remote

    def remote(method: str, *args: Any) -> Any:
        if method not in (
            "release_gpu_output_capture",
            "abandon_unimported_gpu_output_capture",
        ):
            return original(method, *args)
        events.append("submit")
        case.rpc.calls.append((method, args))
        case.rpc.direct_calls.append(method)
        return reply

    case.rpc.execute_method.remote = remote
    return reply


async def test_fence_submission_serialization_ack_order_and_exact_response(
    case: _Case, cpu_cuda: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    await case.bind()
    events: list[str] = []
    reply = pending_release(case, events)
    serialized = asyncio.Event()
    loop = asyncio.get_running_loop()

    def fence(device: torch.device) -> None:
        assert case.state.prepare_payload is None
        assert case.state.gpu_sink._payload is None
        events.append("fence")

    def operation() -> dict[str, Any]:
        case.put()
        events.append("put")
        return deepcopy(CONTENT)

    def finalize(content: dict[str, Any]) -> JSONResponse:
        assert case.state.release_reply is reply._future
        assert not reply._future.done()
        events.append("serialize")
        result = JSONResponse(content)
        loop.call_soon_threadsafe(serialized.set)
        return result

    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    task = asyncio.create_task(
        case.host.finish(case.state, operation, finalize=finalize)
    )
    await asyncio.wait_for(serialized.wait(), 5)
    await asyncio.sleep(0)
    assert not task.done() and case.state.lease is case.lease
    assert events == ["put", "fence", "submit", "serialize"]
    assert reply.future_calls == 1
    reply.set_result(None)
    result = await task
    expected = JSONResponse(CONTENT)
    assert (result.body, result.raw_headers) == (expected.body, expected.raw_headers)
    assert events.count("submit") == 1 and case.state.release_reply is None
    case.assert_released()


@pytest.mark.parametrize("failure", ["fence", "submit", "future", "ack"])
async def test_cleanup_failures_keep_successful_put_and_producer_lease(
    case: _Case,
    cpu_cuda: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    await case.bind()
    attempts = []
    original = case.rpc.execute_method.remote

    def fence(device: torch.device) -> None:
        attempts.append("fence")
        if failure == "fence":
            raise RuntimeError("fence failed")

    class BrokenRegistration:
        def future(self) -> Any:
            raise RuntimeError("future failed")

    def remote(method: str, *args: Any) -> Any:
        if method != "release_gpu_output_capture":
            return original(method, *args)
        attempts.append("submit")
        if failure == "submit":
            raise RuntimeError("submit failed")
        if failure == "future":
            return BrokenRegistration()
        reply = FakeObjectRef()
        reply.set_exception(RuntimeError("ack failed"))
        return reply

    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    case.rpc.execute_method.remote = remote
    result = await case.host.finish(
        case.state, case.put, finalize=lambda _: JSONResponse(CONTENT)
    )
    assert result.body == JSONResponse(CONTENT).body
    assert case.state.lease is case.lease and case.state.capture_key == "call"
    assert case.state.release_reply is None
    assert attempts == (["fence"] if failure == "fence" else ["fence", "submit"])
    assert "cleanup acknowledgement failed" in caplog.text


@pytest.mark.parametrize("consumed", [False, True])
async def test_failed_import_cpu_fallback_keeps_release_classification(
    case: _Case, cpu_cuda: list[str], monkeypatch: pytest.MonkeyPatch, consumed: bool
) -> None:
    await case.bind()

    def fail_import(*args: Any) -> Any:
        raise fixtures.GpuOutputImportError("import failed", handles_consumed=consumed)

    monkeypatch.setattr(hosting, "import_gpu_output_lease", fail_import)
    response = await case.host.finish(
        case.state, case.put, finalize=lambda _: JSONResponse(CONTENT)
    )
    assert response.body == JSONResponse(CONTENT).body
    assert case.sink.payloads == {"call": None} and cpu_cuda == ["sync"]
    case.assert_released(
        "release_gpu_output_capture"
        if consumed
        else "abandon_unimported_gpu_output_capture"
    )


async def test_serializer_error_still_drains_exact_pending_ack(
    case: _Case, cpu_cuda: list[str]
) -> None:
    await case.bind()
    events: list[str] = []
    reply = pending_release(case, events)
    serialized = asyncio.Event()
    loop = asyncio.get_running_loop()

    def fail_serialize(result: Any) -> Any:
        loop.call_soon_threadsafe(serialized.set)
        raise ValueError("JSON serialization failed")

    task = asyncio.create_task(
        case.host.finish(case.state, case.put, finalize=fail_serialize)
    )
    await asyncio.wait_for(serialized.wait(), 5)
    await asyncio.sleep(0)
    assert not task.done()
    reply.set_result(None)
    with pytest.raises(ValueError, match="JSON serialization failed"):
        await task
    assert events == ["submit"] and cpu_cuda == ["sync"]
    case.assert_released()


@pytest.mark.parametrize("ack_fails", [False, True])
async def test_repeated_cancellation_drains_serialization_and_ack(
    case: _Case, cpu_cuda: list[str], ack_fails: bool
) -> None:
    await case.bind()
    reply = pending_release(case, [])
    started, unblock = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def serialize(result: Any) -> JSONResponse:
        loop.call_soon_threadsafe(started.set)
        assert unblock.wait(timeout=5)
        return JSONResponse(CONTENT)

    task = asyncio.create_task(
        case.host.finish(case.state, case.put, finalize=serialize)
    )
    await asyncio.wait_for(started.wait(), 5)
    try:
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done() and not reply.cancelled()
    finally:
        unblock.set()
    await asyncio.sleep(0)
    assert not task.done()
    if ack_fails:
        reply.set_exception(RuntimeError("lost ACK"))
    else:
        reply.set_result(None)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reply.future_calls == 1 and case.state.release_reply is None
    assert cpu_cuda == ["sync"]
    if ack_fails:
        assert case.state.lease is case.lease
    else:
        case.assert_released()


@pytest.mark.parametrize("pending", [False, True])
async def test_unadopted_export_or_key_only_cleanup_precedes_finalization(
    case: _Case, cpu_cuda: list[str], pending: bool
) -> None:
    reply = None
    if pending:
        reply, _ = early.pending_reply(case)
        case.host.start_export(case.state, generated_token_count=1)
    finalized = []

    def finalize(result: Any) -> JSONResponse:
        assert case.state.capture_key is case.state.lease is None
        finalized.append(True)
        return JSONResponse(CONTENT)

    task = asyncio.create_task(
        case.host.finish(case.state, case.put, finalize=finalize)
    )
    if reply is not None:
        await asyncio.sleep(0)
        assert not finalized and not task.done()
        reply.set_result(case.lease)
    await task
    assert finalized == [True]
    assert case.state.export_task is case.state.release_reply is None
    expected = (
        "abandon_unimported_gpu_output_capture"
        if pending
        else "discard_gpu_output_capture"
    )
    assert case.rpc.calls[-1][0] == expected


async def test_collective_cleanup_still_precedes_finalization(
    case: _Case, cpu_cuda: list[str]
) -> None:
    case.host._worker = None
    await case.bind()

    def finalize(result: Any) -> JSONResponse:
        assert case.state.lease is None and cpu_cuda == ["sync"]
        return JSONResponse(CONTENT)

    await case.host.finish(case.state, case.put, finalize=finalize)
    assert not case.rpc.direct_calls
    case.assert_released()


async def test_actual_endpoint_does_not_return_before_release_ack(
    case: _Case, cpu_cuda: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    reply = pending_release(case, events)
    serialized = asyncio.Event()
    loop = asyncio.get_running_loop()

    class ObservedJSONResponse(JSONResponse):
        def render(self, content: Any) -> bytes:
            assert events == ["submit"] and cpu_cuda == ["sync"]
            assert case.state.release_reply is reply._future
            body = super().render(content)
            loop.call_soon_threadsafe(serialized.set)
            return body

    monkeypatch.setattr(fastapi.responses, "JSONResponse", ObservedJSONResponse)
    response = None

    async def formatter(
        self: Any, request: Any, results: Any, *args: Any, **kwargs: Any
    ) -> Any:
        async for _result in results:
            pass
        return response

    worker, request, serving, endpoint = early.build_serving(
        case, monkeypatch, formatter
    )
    response = sys.modules[
        "vllm.entrypoints.openai.chat_completion.protocol"
    ].ChatCompletionResponse()
    response.choices = []
    response.model_dump = lambda: deepcopy(CONTENT)

    def finish(req: Any, content: dict, **kwargs: Any) -> dict:
        assert worker._capture_calls.pop(id(req)) is case.state
        case.put()
        return content

    async def create(req: Any, raw: Any) -> Any:
        return await serving.chat_completion_full_generator(req, early.outputs())

    worker._finish_request_capture = finish
    monkeypatch.setattr(serving, "create_chat_completion", create, raising=False)
    task = asyncio.create_task(endpoint(request, None))
    await asyncio.wait_for(serialized.wait(), 5)
    assert not task.done()
    reply.set_result(None)
    actual = await task
    expected = JSONResponse(CONTENT)
    assert (actual.body, actual.raw_headers) == (expected.body, expected.raw_headers)
    assert not worker._capture_calls and reply.future_calls == 1
    case.assert_released()
