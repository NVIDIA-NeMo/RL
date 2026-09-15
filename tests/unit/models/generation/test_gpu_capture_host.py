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
import threading
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

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

pytestmark = [pytest.mark.nemo_gym, pytest.mark.asyncio]


class _Rpc:
    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def collective_rpc(self, method: str, *, args: tuple[Any, ...]) -> Any:
        self.calls.append((method, args))
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

    def put(self, record: Any = None) -> StageResult:
        return self.state.gpu_sink.stage(record or SimpleNamespace(staging_key="call"))

    async def finish(self) -> StageResult:
        return await self.host.finish(self.state, self.put)

    def assert_released(self, method: str = "release_gpu_output_capture") -> None:
        assert self.rpc.calls[-1] == (method, ("lease-call",))
        assert all(
            rpc_method != "discard_gpu_output_capture"
            for rpc_method, _args in self.rpc.calls
        )
        assert (
            self.state.lease
            is self.state.capture_key
            is self.state.prepare_payload
            is None
        )
        assert not self.state.ipc_handles_consumed


@pytest.fixture
def case(monkeypatch: pytest.MonkeyPatch) -> _Case:
    rpc, sink = _Rpc(), _Sink()
    state = hosting.CapturedModelCall(
        capture=None,
        call=None,
        prompt_token_ids=[11, 12],
        gpu_sink=BoundGpuTokenSink(sink),
        capture_key="call",
    )
    # Tensor imports are mocked; native tests exercise actual CUDA IPC descriptors.
    lease = GpuOutputLease(
        "lease-call", "call", socket.gethostname(), "gpu-1", (), (), None
    )
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


@pytest.mark.parametrize("backfill_ranges", [(), ((0, 1),)])
async def test_bind_preserves_tensors_coordinates_and_put_thread(
    case: _Case,
    cpu_cuda: list[str],
    backfill_ranges: tuple[tuple[int, int], ...],
) -> None:
    routes = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.int64)
    case.tensors = replace(case.tensors, routed_experts=routes)
    case.lease = replace(
        case.lease, routed_experts_prefix_backfill_ranges=backfill_ranges
    )
    await case.bind()
    assert not case.imports and not case.state.ipc_handles_consumed
    assert case.state.prepare_payload is not None

    def put() -> StageResult:
        assert case.imports == [(case.lease, threading.get_ident())]
        return case.put()

    assert (await case.host.finish(case.state, put)).ok
    payload = case.sink.payloads["call"]
    assert payload.prompt_len == 2
    assert payload.generated_token_ids is case.tensors.generated_token_ids
    assert payload.generated_logprobs is case.tensors.generation_logprobs
    assert payload.routed_experts is routes
    assert payload.routed_experts_prefix_backfill_ranges == backfill_ranges
    assert case.rpc.calls[0] == ("export_gpu_output_capture", ("call", 1, 2))
    assert cpu_cuda == ["sync"]
    case.assert_released()


async def test_abort_before_put_never_imports(case: _Case, cpu_cuda: list[str]) -> None:
    await case.bind()
    await case.host.release(case.state)
    assert not case.imports
    case.assert_released("abandon_unimported_gpu_output_capture")


async def test_failed_cleanup_ack_preserves_successful_put(
    case: _Case, cpu_cuda: list[str]
) -> None:
    case.rpc.responses["release_gpu_output_capture"] = RuntimeError("lost ACK")
    await case.bind()
    assert (await case.finish()).ok
    assert case.state.lease is case.lease and case.state.capture_key == "call"
    assert case.rpc.calls[-1] == ("release_gpu_output_capture", ("lease-call",))


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

    def put() -> StageResult:
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
            await case.host.finish(case.state, put)
    else:
        assert (await case.host.finish(case.state, put)).ok
        assert case.sink.payloads == {"call": None}
    if failure == "export":
        assert case.rpc.calls[-1] == ("discard_gpu_output_capture", ("call",))
    else:
        case.assert_released(
            "abandon_unimported_gpu_output_capture"
            if failure == "import_before_open"
            else "release_gpu_output_capture"
        )


@pytest.mark.parametrize("phase", ["import", "put"])
async def test_cancellation_waits_for_allocation_users_before_release(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    cpu_cuda: list[str],
    phase: str,
) -> None:
    started, unblock = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def block() -> None:
        loop.call_soon_threadsafe(started.set)
        assert unblock.wait(timeout=5)
        cpu_cuda.append(f"{phase}-completed")

    def import_payload(*args: Any) -> GpuOutputTensors:
        if phase == "import":
            block()
        return case.tensors

    monkeypatch.setattr(hosting, "import_gpu_output_lease", import_payload)
    await case.bind()
    task = asyncio.create_task(
        case.host.finish(case.state, block if phase == "put" else lambda: None)
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    try:
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        completed, _ = await asyncio.wait({task}, timeout=0.05)
        assert not completed and not cpu_cuda
    finally:
        unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cpu_cuda == [f"{phase}-completed", "sync"]
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
        replace(case.lease, capture_key=key, lease_id=f"lease-{key}")
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
        await case.host.finish(
            state, lambda: state.gpu_sink.stage(SimpleNamespace(staging_key=key))
        )

    await asyncio.gather(complete(case.state), complete(second))
    for key, ids in tokens.items():
        assert case.sink.payloads[key].generated_token_ids is ids
    assert sorted(
        args[0]
        for method, args in case.rpc.calls
        if method == "release_gpu_output_capture"
    ) == ["lease-call", "lease-second"]
    assert case.state.lease is second.lease is None
