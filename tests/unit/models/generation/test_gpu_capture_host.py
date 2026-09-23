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

"""GPU capture frontend ownership, request isolation, and cancellation ordering."""

from __future__ import annotations

import asyncio
import inspect
import socket
import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse, Response

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.records import StageResult  # noqa: E402

from nemo_rl.data_plane.gpu_token_payload import (  # noqa: E402
    BoundGpuTokenSink,
    GpuTokenPayload,
)
from nemo_rl.models.generation.vllm import gpu_capture_host as hosting  # noqa: E402
from nemo_rl.models.generation.vllm.gpu_output_capture import (  # noqa: E402
    GpuOutputImportError,
    GpuOutputLease,
    GpuOutputTensors,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
    _AsyncLLMHTTPClient,
)
from tests.unit.models.generation import (
    test_vllm_chat_template_wiring as wiring,  # noqa: E402
)

pytestmark = [pytest.mark.nemo_gym, pytest.mark.asyncio]


async def _consume(response: Response) -> Response:
    async def send(message: dict) -> None:
        pass

    await response({"type": "http"}, None, send)
    return response


class _Rpc:
    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.pending: dict[str, Future[Any]] = {}

    def pause(self, method: str) -> Future[Any]:
        self.pending[method] = Future()
        return self.pending[method]

    async def collective_rpc(self, method: str, *, args: tuple[Any, ...]) -> Any:
        self.calls.append((method, args))
        if method in self.pending:
            return [await asyncio.wrap_future(self.pending[method])]
        response = self.responses.get(method)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(*args)
        return await response if inspect.isawaitable(response) else response


class _Sink:
    def __init__(self) -> None:
        self.payloads: dict[str, GpuTokenPayload | None] = {}

    def stage(
        self, record: Any, *, gpu_payload: GpuTokenPayload | None = None
    ) -> StageResult:
        self.payloads[record.staging_key] = gpu_payload
        return StageResult(ok=True, staging_key=record.staging_key)


@dataclass
class _Case:
    rpc: _Rpc
    sink: _Sink
    state: hosting.CapturedModelCall
    lease: GpuOutputLease
    tensors: GpuOutputTensors
    host: hosting.GpuCaptureHost
    imports: list[tuple[GpuOutputLease, int]]

    async def bind(self, *, count: int = 1) -> None:
        await self.host.bind(
            self.state,
            generated_token_count=count,
        )

    def put(self, record: Any = None) -> dict:
        result = self.state.gpu_sink.stage(
            record or SimpleNamespace(staging_key="call")
        )
        return {"ok": result.ok}

    def assert_released(self, method: str = "release_gpu_output_capture") -> None:
        assert self.rpc.calls[-1] == (method, ("call",))
        assert all(
            rpc_method != "discard_gpu_output_capture"
            for rpc_method, _args in self.rpc.calls
        )
        assert (
            self.state.lease
            is self.state.capture_key
            is self.state.prepare_payload
            is self.state.export_task
            is self.state.release_reply
            is None
        )
        assert not self.state.ipc_handles_consumed


@pytest.fixture
def case(monkeypatch: pytest.MonkeyPatch) -> _Case:
    rpc, sink = _Rpc(), _Sink()
    state = hosting.CapturedModelCall(
        capture=None,
        call=SimpleNamespace(admission=SimpleNamespace(prev_len=0)),
        prompt_token_ids=[11, 12],
        gpu_sink=BoundGpuTokenSink(sink),
        capture_key="call",
    )
    # Tensor imports are mocked; native tests exercise actual CUDA IPC descriptors.
    lease = GpuOutputLease("call", socket.gethostname(), "gpu-1", (), (), None)
    result = _Case(
        rpc,
        sink,
        state,
        lease,
        GpuOutputTensors(torch.tensor([21]), torch.tensor([-0.25]), None),
        hosting.GpuCaptureHost(rpc, torch.device("cuda:0")),
        [],
    )
    rpc.responses["export_gpu_output_capture"] = lambda *_: [None, result.lease]

    def import_payload(
        descriptor: GpuOutputLease, device: torch.device
    ) -> GpuOutputTensors:
        result.imports.append((descriptor, threading.get_ident()))
        return result.tensors

    monkeypatch.setattr(hosting, "import_gpu_output_lease", import_payload)
    return result


@pytest.fixture
def cpu_cuda(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub CUDA contexts only; host, binding, and tensors execute normally."""
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append("sync"))
    return events


@pytest.mark.parametrize(
    "topology",
    [
        "available",
        "nondefault_device",
        "invisible_device",
        "no_owner",
        "two_owners",
        "no_device",
        "unsupported_native",
    ],
)
async def test_create_uses_visible_owner_or_preserves_cpu_path(
    case: _Case,
    cpu_cuda: list[str],
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
) -> None:
    case.rpc.responses["configure_gpu_output_capture"] = {
        "available": [None, "gpu-0"],
        "nondefault_device": ["gpu-1"],
        "invisible_device": ["gpu-2"],
        "no_owner": [None],
        "two_owners": ["gpu-0", "gpu-0"],
        "no_device": ["gpu-0"],
        "unsupported_native": RuntimeError("unsupported native output"),
    }[topology]
    monkeypatch.setattr(
        torch.cuda, "device_count", lambda: 0 if topology == "no_device" else 2
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: SimpleNamespace(uuid=f"gpu-{i}")
    )
    host = await hosting.GpuCaptureHost.create(case.rpc, require_routed_experts=True)
    expected_device = {
        "available": torch.device("cuda:0"),
        "nondefault_device": torch.device("cuda:1"),
    }.get(topology)
    assert (host.device if host else None) == expected_device
    assert case.rpc.calls == [
        ("configure_gpu_output_capture", (socket.gethostname(), True))
    ]
    if host is not None:
        await host.bind(case.state, generated_token_count=1)
        await host.release(case.state)
        assert [method for method, _ in case.rpc.calls] == [
            "configure_gpu_output_capture",
            "export_gpu_output_capture",
            "abandon_unimported_gpu_output_capture",
        ]


@pytest.mark.parametrize("prev_len,backfill_ranges", [(0, ()), (1, ((0, 1),)), (2, ())])
async def test_bind_preserves_tensors_coordinates_and_put_thread(
    case: _Case,
    cpu_cuda: list[str],
    backfill_ranges: tuple[tuple[int, int], ...],
    prev_len: int,
) -> None:
    case.state.call.admission.prev_len = prev_len
    routes = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.int64)[prev_len:]
    case.tensors = replace(case.tensors, routed_experts=routes)
    case.lease = replace(
        case.lease, routed_experts_prefix_backfill_ranges=backfill_ranges
    )
    await case.bind()
    assert not case.imports and not case.state.ipc_handles_consumed
    assert case.state.prepare_payload is not None

    def put() -> dict:
        assert case.imports == [(case.lease, threading.get_ident())]
        return case.put()

    response = await case.host.finish(case.state, put, finalize=JSONResponse)
    assert (await _consume(response)).body == b'{"ok":true}'
    payload = case.sink.payloads["call"]
    assert payload.prompt_len == 2
    assert payload.generated_token_ids is case.tensors.generated_token_ids
    assert payload.generated_logprobs is case.tensors.generation_logprobs
    assert payload.routed_experts is routes
    assert payload.routed_experts_prefix_backfill_ranges == backfill_ranges
    assert case.rpc.calls[0] == (
        "export_gpu_output_capture",
        ("call", 1, 2, prev_len),
    )
    assert cpu_cuda == ["sync"]
    case.assert_released()


@pytest.mark.parametrize("response", ["foreign", "empty", "multiple"])
async def test_invalid_export_discards_request_without_import(
    case: _Case,
    cpu_cuda: list[str],
    response: str,
) -> None:
    case.rpc.responses["export_gpu_output_capture"] = {
        "foreign": [replace(case.lease, capture_key="other")],
        "empty": [],
        "multiple": [case.lease, case.lease],
    }[response]
    with pytest.raises(
        RuntimeError, match="different model call|exactly one payload lease"
    ):
        await case.bind()
    await case.host.release(case.state)
    assert case.rpc.calls[-1] == ("discard_gpu_output_capture", ("call",))
    assert not case.imports and not cpu_cuda


@pytest.mark.parametrize(
    "failure", ["export", "import_before_open", "import_after_open", "put"]
)
async def test_capture_failures_preserve_cpu_put_and_release_ownership(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    cpu_cuda: list[str],
    failure: str,
) -> None:
    if failure == "export":
        case.rpc.responses["export_gpu_output_capture"] = RuntimeError("export failed")

    def import_payload(*args: Any) -> GpuOutputTensors:
        if failure.startswith("import_"):
            raise GpuOutputImportError(
                failure, handles_consumed=failure == "import_after_open"
            )
        return case.tensors

    def put() -> dict:
        if failure == "put":
            raise RuntimeError("put failed")
        return case.put()

    monkeypatch.setattr(hosting, "import_gpu_output_lease", import_payload)
    try:
        await case.bind()
    except RuntimeError:
        assert failure == "export"
        case.state.gpu_sink.clear()  # Serving preserves the original CPU path.
    if failure == "put":
        with pytest.raises(RuntimeError, match="put failed"):
            await case.host.finish(case.state, put, finalize=JSONResponse)
    else:
        response = await case.host.finish(case.state, put, finalize=JSONResponse)
        assert (await _consume(response)).body == b'{"ok":true}'
        assert case.sink.payloads == {"call": None}
    if failure == "export":
        assert case.rpc.calls[-1] == ("discard_gpu_output_capture", ("call",))
    else:
        case.assert_released(
            "abandon_unimported_gpu_output_capture"
            if failure == "import_before_open"
            else "release_gpu_output_capture"
        )


@pytest.mark.parametrize("phase", ["import", "put", "serialize", "serialize_ack_error"])
async def test_cancellation_drains_allocation_users_and_ack(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    cpu_cuda: list[str],
    phase: str,
) -> None:
    started, unblock = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    serializing = phase.startswith("serialize")
    reply = case.rpc.pause("release_gpu_output_capture") if serializing else None

    def block() -> None:
        loop.call_soon_threadsafe(started.set)
        assert unblock.wait(timeout=5)
        cpu_cuda.append("completed")

    def import_payload(*args: Any) -> GpuOutputTensors:
        if phase == "import":
            block()
        return case.tensors

    def put() -> dict:
        if phase == "put":
            block()
        return case.put()

    def finalize(result: dict) -> JSONResponse:
        if serializing:
            block()
        return JSONResponse(result)

    monkeypatch.setattr(hosting, "import_gpu_output_lease", import_payload)
    await case.bind()
    task = asyncio.create_task(
        case.host.finish(
            case.state,
            put,
            finalize=finalize,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    try:
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert cpu_cuda == (["sync"] if serializing else [])
        if reply is not None:
            assert not reply.cancelled()
    finally:
        unblock.set()
    if reply is not None:
        await asyncio.sleep(0)
        assert not task.done()
        if phase == "serialize_ack_error":
            reply.set_exception(RuntimeError("lost ACK"))
        else:
            reply.set_result(None)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cpu_cuda == (["sync", "completed"] if serializing else ["completed", "sync"])
    if phase == "serialize_ack_error":
        assert case.state.lease is case.lease
    else:
        case.assert_released()


async def test_cancelled_export_failure_stays_cancelled(
    case: _Case, cpu_cuda: list[str]
) -> None:
    started, unblock = asyncio.Event(), asyncio.Event()

    async def export(*args: Any) -> None:
        started.set()
        await unblock.wait()
        raise RuntimeError("export failed after cancellation")

    case.rpc.responses["export_gpu_output_capture"] = export
    task = asyncio.create_task(case.bind())
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    unblock.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await case.host.release(case.state)
    assert not case.imports and not case.sink.payloads
    assert case.rpc.calls[-1] == ("discard_gpu_output_capture", ("call",))


async def test_concurrent_calls_keep_independent_payloads(
    case: _Case, monkeypatch: pytest.MonkeyPatch, cpu_cuda: list[str]
) -> None:
    second = replace(
        case.state, gpu_sink=BoundGpuTokenSink(case.sink), capture_key="second"
    )
    tokens = {"call": torch.tensor([21]), "second": torch.tensor([42])}
    case.rpc.responses["export_gpu_output_capture"] = lambda key, *_: [
        replace(case.lease, capture_key=key)
    ]
    monkeypatch.setattr(
        hosting,
        "import_gpu_output_lease",
        lambda lease, _: GpuOutputTensors(
            tokens[lease.capture_key], torch.tensor([-0.25]), None
        ),
    )

    async def complete(state: hosting.CapturedModelCall) -> None:
        key = state.capture_key
        await case.host.bind(state, generated_token_count=1)
        response = await case.host.finish(
            state,
            lambda: {"ok": state.gpu_sink.stage(SimpleNamespace(staging_key=key)).ok},
            finalize=JSONResponse,
        )
        await _consume(response)

    await asyncio.gather(complete(case.state), complete(second))
    for key, ids in tokens.items():
        assert case.sink.payloads[key].generated_token_ids is ids
    assert sorted(
        args[0]
        for method, args in case.rpc.calls
        if method == "release_gpu_output_capture"
    ) == ["call", "second"]
    assert case.state.lease is second.lease is None


@pytest.mark.parametrize("failure", ["fence", "submit", "ack"])
async def test_cleanup_failure_preserves_successful_put_and_lease(
    case: _Case,
    cpu_cuda: list[str],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    await case.bind()
    attempts: list[str] = []

    def fence(device: torch.device) -> None:
        attempts.append("fence")
        if failure == "fence":
            raise RuntimeError("fence failed")

    def fail_submission(coroutine: Any, loop: Any) -> None:
        attempts.append("submit")
        coroutine.close()
        raise RuntimeError("submit failed")

    async def fail_ack(*args: Any) -> None:
        attempts.append("ack")
        raise RuntimeError("ack failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    if failure == "submit":
        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fail_submission)
    case.rpc.responses["release_gpu_output_capture"] = fail_ack
    response = await case.host.finish(
        case.state,
        case.put,
        finalize=JSONResponse,
    )
    assert (await _consume(response)).body == b'{"ok":true}'
    assert "call" in case.sink.payloads
    assert case.state.lease is case.lease and case.state.capture_key == "call"
    assert case.state.release_reply is None
    assert attempts == (["fence"] if failure == "fence" else ["fence", failure])


async def test_json_failure_still_waits_for_release_ack(
    case: _Case, cpu_cuda: list[str]
) -> None:
    await case.bind()
    reply = case.rpc.pause("release_gpu_output_capture")
    serialized = asyncio.Event()
    loop = asyncio.get_running_loop()

    def fail_serialize(result: Any) -> None:
        loop.call_soon_threadsafe(serialized.set)
        raise ValueError("JSON failed")

    task = asyncio.create_task(
        case.host.finish(case.state, case.put, finalize=fail_serialize)
    )
    await asyncio.wait_for(serialized.wait(), 5)
    assert not task.done()
    reply.set_result(None)
    with pytest.raises(ValueError, match="JSON failed"):
        await task
    assert cpu_cuda == ["sync"]
    case.assert_released()


@pytest.mark.parametrize("path", ["key_only", "pending_export"])
async def test_nonoverlap_cleanup_precedes_finalization(
    case: _Case, cpu_cuda: list[str], path: str
) -> None:
    reply = None
    if path == "pending_export":
        reply = case.rpc.pause("export_gpu_output_capture")
        case.host.start_export(case.state, generated_token_count=1)
    finalized = []

    def finalize(result: dict) -> JSONResponse:
        assert case.state.capture_key is case.state.lease is None
        finalized.append(True)
        return JSONResponse(result)

    task = asyncio.create_task(
        case.host.finish(case.state, case.put, finalize=finalize)
    )
    if reply is not None:
        await asyncio.sleep(0)
        assert not finalized and not task.done()
        reply.set_result(case.lease)
    assert (await task).body == b'{"ok":true}' and finalized == [True]
    expected = {
        "key_only": "discard_gpu_output_capture",
        "pending_export": "abandon_unimported_gpu_output_capture",
    }[path]
    assert case.rpc.calls[-1][0] == expected
    assert case.state.export_task is case.state.release_reply is None


async def _outputs() -> Any:
    yield SimpleNamespace(finished=False, outputs=[SimpleNamespace(token_ids=[])])
    yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=[21])])


def _endpoint(
    case: _Case, formatter: Callable, monkeypatch: pytest.MonkeyPatch
) -> tuple[VllmAsyncGenerationWorkerImpl, Any, Callable]:
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._gpu_capture_host, worker.token_capture = case.host, object()
    request = SimpleNamespace(
        top_k=None,
        top_p=1.0,
        temperature=1.0,
        return_tokens_as_token_ids=False,
        logprobs=False,
        stream=False,
    )
    worker._capture_calls = {id(request): case.state}
    case.state.capture = SimpleNamespace(fail_call=MagicMock())
    serving, endpoint = wiring.build_serving(worker, request, formatter, monkeypatch)

    async def create(req: Any, raw: Any) -> Any:
        return await serving.chat_completion_full_generator(req, _outputs())

    monkeypatch.setattr(serving, "create_chat_completion", create, raising=False)
    return worker, request, endpoint


async def test_endpoint_sends_same_json_before_release_ack(
    case: _Case, cpu_cuda: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    exported = case.rpc.pause("export_gpu_output_capture")
    released = case.rpc.pause("release_gpu_output_capture")
    serialized = asyncio.Event()
    loop = asyncio.get_running_loop()
    content = {
        "choices": [
            {"routed_experts": "unchanged NPY / é", "message": {"content": "done"}}
        ]
    }

    async def formatter(
        self: Any, request: Any, results: Any, *args: Any, **kwargs: Any
    ) -> Any:
        async for output in results:
            if output.finished:
                # The wrapper scheduled export; a second start must reuse it.
                case.host.start_export(case.state, generated_token_count=1)
                await asyncio.sleep(0)
                assert case.rpc.calls == [
                    ("export_gpu_output_capture", ("call", 1, 2, 0))
                ]
                exported.set_result(case.lease)
            else:
                assert not case.rpc.calls
        return response

    class ObservedResponse(JSONResponse):
        def render(self, value: Any) -> bytes:
            assert "call" in case.sink.payloads and cpu_cuda == ["sync"]
            assert case.state.gpu_sink._payload is None
            assert case.state.release_reply is not None
            assert not released.done()
            body = super().render(value)
            loop.call_soon_threadsafe(serialized.set)
            return body

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.JSONResponse",
        ObservedResponse,
    )

    worker, request, endpoint = _endpoint(case, formatter, monkeypatch)
    response = sys.modules[
        "vllm.entrypoints.openai.chat_completion.protocol"
    ].ChatCompletionResponse()
    response.choices, response.model_dump = [], lambda: content

    def finish(req: Any, value: dict, **kwargs: Any) -> dict:
        assert worker._capture_calls.pop(id(req)) is case.state
        case.put()
        return value

    worker._finish_request_capture = finish
    task = asyncio.create_task(endpoint(request, None))
    await asyncio.wait_for(serialized.wait(), 5)
    actual = await asyncio.wait_for(task, 5)
    expected = JSONResponse(content)
    assert (actual.body, actual.raw_headers) == (expected.body, expected.raw_headers)
    body_sent = asyncio.Event()

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body":
            assert message["body"] == expected.body
            body_sent.set()

    serving = asyncio.create_task(actual({"type": "http"}, None, send))
    await asyncio.wait_for(body_sent.wait(), 5)
    assert not serving.done() and case.state.lease is case.lease
    released.set_result(None)
    await serving
    assert len(case.rpc.calls) == 2
    assert not worker._capture_calls
    case.assert_released()


@pytest.mark.parametrize("failure", ["send", "cancel_send", "cancel_ack", "ack"])
async def test_response_failure_drains_cleanup(
    case: _Case, cpu_cuda: list[str], failure: str
) -> None:
    await case.bind()
    reply = case.rpc.pause("release_gpu_output_capture")
    response = await case.host.finish(
        case.state, case.put, finalize=lambda _: JSONResponse({"ok": True})
    )
    entered_send = asyncio.Event()

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body":
            entered_send.set()
            if failure == "send":
                raise OSError("client disconnected")
            if failure == "cancel_send":
                await asyncio.Event().wait()

    serving = asyncio.create_task(response({"type": "http"}, None, send))
    await asyncio.wait_for(entered_send.wait(), 5)
    if failure.startswith("cancel"):
        for _ in range(2):
            serving.cancel()
            await asyncio.sleep(0)
    assert not serving.done() and not reply.cancelled()
    assert cpu_cuda == ["sync"] and case.state.gpu_sink._payload is None
    if failure == "ack":
        reply.set_exception(RuntimeError("lost ACK"))
        await serving  # Cleanup remains warning-only after the body was sent.
        assert case.state.lease is case.lease
    else:
        reply.set_result(None)
        with pytest.raises(
            asyncio.CancelledError if failure.startswith("cancel") else OSError
        ):
            await serving
        case.assert_released()


async def test_real_http_body_and_shutdown_overlap_collective_ack(
    case: _Case, cpu_cuda: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real ASGI/socket IO and separate engine/HTTP loops, with a held ACK."""
    engine_loop = asyncio.new_event_loop()
    engine_thread = threading.Thread(target=engine_loop.run_forever)
    engine_thread.start()
    release_started = threading.Event()
    collective_rpc = case.rpc.collective_rpc

    async def on_engine(method: str, *, args: tuple[Any, ...]) -> Any:
        assert asyncio.get_running_loop() is engine_loop
        if method == "release_gpu_output_capture":
            release_started.set()
        return await collective_rpc(method, args=args)

    monkeypatch.setattr(case.rpc, "collective_rpc", on_engine)
    for name in ("model_config", "renderer", "input_processor", "vllm_config"):
        monkeypatch.setattr(case.rpc, name, None, raising=False)
    case.host._rpc = _AsyncLLMHTTPClient(case.rpc, engine_loop)
    reply = case.rpc.pause("release_gpu_output_capture")
    app = FastAPI()

    def serialize(result: Any) -> JSONResponse:
        # Rendering is on the existing PUT thread, leaving both loops runnable.
        assert release_started.wait(timeout=5)
        assert cpu_cuda == ["sync"] and case.state.gpu_sink._payload is None
        assert not reply.done()
        return JSONResponse({"ok": result["ok"], "text": "é"})

    @app.get("/capture")
    async def capture() -> Any:
        return await case.host.finish(case.state, case.put, finalize=serialize)

    server = uvicorn.Server(
        uvicorn.Config(app, log_config=None, access_log=False, lifespan="off")
    )
    shutdown_waiting = asyncio.Event()
    wait_tasks = server._wait_tasks_to_complete

    async def wait_for_cleanup() -> None:
        shutdown_waiting.set()
        await wait_tasks()

    monkeypatch.setattr(server, "_wait_tasks_to_complete", wait_for_cleanup)
    server_task = None
    writer = None
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        sock.setblocking(False)
        try:
            await case.bind()
            server_task = asyncio.create_task(server.serve(sockets=[sock]))

            async def ready() -> None:
                while not server.started:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(ready(), 5)
            reader, writer = await asyncio.open_connection(*sock.getsockname())
            writer.write(b"GET /capture HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            expected = JSONResponse({"ok": True, "text": "é"}).body
            assert headers.startswith(b"HTTP/1.1 200 OK\r\n")
            assert f"content-length: {len(expected)}\r\n".encode() in headers.lower()
            assert (
                await asyncio.wait_for(reader.readexactly(len(expected)), 5) == expected
            )
            assert not reply.done() and case.state.lease is case.lease
            assert (
                case.sink.payloads["call"].generated_token_ids
                is case.tensors.generated_token_ids
            )

            writer.close()
            await writer.wait_closed()
            server.should_exit = True
            await asyncio.wait_for(shutdown_waiting.wait(), 5)
            assert not server_task.done()
            reply.set_result(None)
            await asyncio.wait_for(server_task, 5)
            case.assert_released()
        finally:
            if not reply.done():
                reply.set_result(None)
            if writer is not None:
                writer.close()
            server.should_exit = True
            if server_task is not None:
                await asyncio.wait_for(server_task, 5)
            engine_loop.call_soon_threadsafe(engine_loop.stop)
            await asyncio.to_thread(engine_thread.join)
            engine_loop.close()


@pytest.mark.parametrize(
    "failure",
    ["error_response", "formatter", "model_dump", "cancel", "cancel_export_error"],
)
async def test_endpoint_abort_drains_pending_export(
    case: _Case, cpu_cuda: list[str], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    reply = case.rpc.pause("export_gpu_output_capture")
    formatting = asyncio.Event()

    async def formatter(
        self: Any, request: Any, results: Any, *args: Any, **kwargs: Any
    ) -> Any:
        async for output in results:
            if output.finished:
                formatting.set()
                if failure == "formatter":
                    raise RuntimeError("formatter failed")
                if failure.startswith("cancel"):
                    await asyncio.Event().wait()
        return response

    worker, request, endpoint = _endpoint(case, formatter, monkeypatch)
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

    if failure == "model_dump":

        def fail_dump() -> None:
            raise RuntimeError("model_dump failed")

        response.model_dump = fail_dump

    task = asyncio.create_task(endpoint(request, None))
    await asyncio.wait_for(formatting.wait(), 5)
    if failure.startswith("cancel"):
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
    assert not task.done() and not reply.cancelled()
    if failure == "cancel_export_error":
        reply.set_exception(RuntimeError("export failed"))
    else:
        reply.set_result(case.lease)
    if failure == "error_response":
        assert (await task).status_code == 400
    else:
        error = asyncio.CancelledError if failure.startswith("cancel") else RuntimeError
        with pytest.raises(error):
            await task
    assert not case.imports and not case.sink.payloads and not worker._capture_calls
    assert case.state.capture.fail_call.call_count == 1
    if failure == "cancel_export_error":
        assert case.rpc.calls[-1] == ("discard_gpu_output_capture", ("call",))
    else:
        case.assert_released("abandon_unimported_gpu_output_capture")
