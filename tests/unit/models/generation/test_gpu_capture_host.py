"""GPU capture frontend ownership, request isolation, and cancellation ordering."""

from __future__ import annotations

import asyncio
import socket
import threading
from contextlib import nullcontext
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
    CudaTensorIpc,
    GpuOutputCaptureCapabilities,
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
            return response(*args)
        return response


class _Sink:
    def __init__(self) -> None:
        self.payloads: dict[str, GpuTokenPayload] = {}

    def stage(self, record: Any, *, gpu_payload: GpuTokenPayload) -> StageResult:
        self.payloads[record.staging_key] = gpu_payload
        return StageResult(ok=True, staging_key=record.staging_key)


def _state(key: str, sink: _Sink | None = None) -> hosting.CapturedModelCall:
    return hosting.CapturedModelCall(
        capture=None,
        call=None,
        prompt_token_ids=[11, 12],
        gpu_sink=BoundGpuTokenSink(sink or _Sink()),
        capture_key=key,
    )


def _lease(key: str) -> GpuOutputLease:
    ipc = CudaTensorIpc(
        shape=(1,),
        stride=(1,),
        tensor_offset=0,
        dtype="int64",
        storage_handle=None,
        storage_size_bytes=8,
        storage_offset_bytes=0,
        ref_counter_handle=None,
        ref_counter_offset=0,
        event_handle=None,
        event_sync_required=False,
    )
    return GpuOutputLease(
        lease_id=f"lease-{key}",
        capture_key=key,
        hostname=socket.gethostname(),
        gpu_uuid="gpu-1",
        ready_event_handle=b"ready",
        generated_token_ids=ipc,
        generation_logprobs=ipc,
        routed_experts=None,
    )


@pytest.fixture
def cpu_cuda(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub CUDA contexts only; host, binding, and tensors execute normally."""
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )
    return events


async def test_create_resolves_physical_uuid_instead_of_worker_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rpc = _Rpc()
    rpc.responses["configure_gpu_output_capture"] = [
        GpuOutputCaptureCapabilities(False, socket.gethostname(), "gpu-1", 0),
        GpuOutputCaptureCapabilities(True, socket.gethostname(), "gpu-0", 1),
    ]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=f"gpu-{index}"),
    )
    host = await hosting.GpuCaptureHost.create(
        rpc, max_retained_bytes=4096, require_routed_experts=True
    )
    assert host.device == torch.device("cuda:0")
    assert rpc.calls == [
        ("configure_gpu_output_capture", (4096, socket.gethostname(), True))
    ]


async def test_create_rejects_visible_producer_on_nondefault_frontend_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rpc = _Rpc()
    rpc.responses["configure_gpu_output_capture"] = [
        GpuOutputCaptureCapabilities(True, socket.gethostname(), "gpu-1", 0)
    ]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=f"gpu-{index}"),
    )
    with pytest.raises(RuntimeError, match="producer mapped to frontend cuda:0"):
        await hosting.GpuCaptureHost.create(
            rpc, max_retained_bytes=4096, require_routed_experts=True
        )


@pytest.mark.parametrize("owner_count", [0, 2])
async def test_create_requires_exactly_one_producer(owner_count: int) -> None:
    rpc = _Rpc()
    rpc.responses["configure_gpu_output_capture"] = [
        GpuOutputCaptureCapabilities(True, socket.gethostname(), "gpu-1", 0)
    ] * owner_count
    with pytest.raises(RuntimeError, match="exactly one producer"):
        await hosting.GpuCaptureHost.create(
            rpc, max_retained_bytes=4096, require_routed_experts=False
        )


@pytest.mark.parametrize("remote_host", [False, True])
async def test_create_rejects_inaccessible_producer(
    monkeypatch: pytest.MonkeyPatch, remote_host: bool
) -> None:
    rpc = _Rpc()
    rpc.responses["configure_gpu_output_capture"] = [
        GpuOutputCaptureCapabilities(
            True, "remote-host" if remote_host else socket.gethostname(), "gpu-1", 0
        )
    ]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="frontend host|not visible"):
        await hosting.GpuCaptureHost.create(
            rpc, max_retained_bytes=4096, require_routed_experts=False
        )


async def test_bind_preserves_original_tensors_and_request_coordinates(
    monkeypatch: pytest.MonkeyPatch, cpu_cuda: list[str]
) -> None:
    rpc, sink = _Rpc(), _Sink()
    state = _state("call-a", sink)
    lease = _lease("call-a")
    rpc.responses["export_gpu_output_capture"] = [None, lease]
    ids = torch.tensor([21])
    logprobs = torch.tensor([-0.25])
    routes = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.int64)
    imports: list[GpuOutputLease] = []

    def import_payload(
        descriptor: GpuOutputLease, device: torch.device
    ) -> GpuOutputTensors:
        imports.append(descriptor)
        return GpuOutputTensors(ids, logprobs, routes)

    monkeypatch.setattr(hosting, "import_gpu_output_lease", import_payload)
    host = hosting.GpuCaptureHost(rpc, torch.device("cuda:0"))
    await host.bind(state, generated_token_count=1, routed_experts_dtype=torch.int16)
    assert state.ipc_handles_consumed
    result = await host.finish(
        state, lambda: state.gpu_sink.stage(SimpleNamespace(staging_key="call-a"))
    )
    assert result.ok
    payload = sink.payloads["call-a"]
    assert payload.prompt_len == 2
    assert payload.generated_token_ids is ids
    assert payload.generated_logprobs is logprobs
    assert payload.routed_experts.dtype == torch.int16
    assert torch.equal(payload.routed_experts, routes)
    assert imports == [lease]
    assert rpc.calls == [
        ("export_gpu_output_capture", ("call-a", 1, 2)),
        ("release_gpu_output_capture", ("lease-call-a",)),
        ("discard_gpu_output_capture", ("call-a",)),
    ]
    assert cpu_cuda == ["sync"]
    assert state.lease is None and state.capture_key is None
    assert not state.ipc_handles_consumed


async def test_bind_rejects_foreign_model_call_lease(cpu_cuda: list[str]) -> None:
    rpc = _Rpc()
    rpc.responses["export_gpu_output_capture"] = [_lease("other-call")]
    host = hosting.GpuCaptureHost(rpc, torch.device("cuda:0"))
    state = _state("call-a")
    try:
        with pytest.raises(RuntimeError, match="different model call"):
            await host.bind(
                state, generated_token_count=1, routed_experts_dtype=torch.int16
            )
    finally:
        await host.release(state)
    assert rpc.calls == [
        ("export_gpu_output_capture", ("call-a", 1, 2)),
        ("discard_gpu_output_capture", ("call-a",)),
    ]
    assert not cpu_cuda
    assert state.lease is None and state.capture_key is None


@pytest.mark.parametrize(
    "failure", ["export", "import_before_open", "import_after_open", "put"]
)
async def test_failures_release_owned_state(
    monkeypatch: pytest.MonkeyPatch, cpu_cuda: list[str], failure: str
) -> None:
    rpc, state = _Rpc(), _state("call-a")
    rpc.responses["export_gpu_output_capture"] = (
        RuntimeError("export failed") if failure == "export" else [_lease("call-a")]
    )
    host = hosting.GpuCaptureHost(rpc, torch.device("cuda:0"))

    def import_payload(lease: GpuOutputLease, device: torch.device) -> GpuOutputTensors:
        if failure.startswith("import_"):
            raise GpuOutputImportError(
                f"{failure} failed", handles_consumed=failure == "import_after_open"
            )
        return GpuOutputTensors(torch.tensor([21]), torch.tensor([-0.25]), None)

    def put() -> None:
        raise RuntimeError("put failed")

    monkeypatch.setattr(hosting, "import_gpu_output_lease", import_payload)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        try:
            await host.bind(
                state, generated_token_count=1, routed_experts_dtype=torch.int16
            )
            await host.finish(state, put)
        finally:
            await host.release(state)
    expected = [("export_gpu_output_capture", ("call-a", 1, 2))]
    if failure != "export":
        method = (
            "abandon_unimported_gpu_output_capture"
            if failure == "import_before_open"
            else "release_gpu_output_capture"
        )
        expected.append((method, ("lease-call-a",)))
    expected.append(("discard_gpu_output_capture", ("call-a",)))
    assert rpc.calls == expected
    assert state.lease is None and state.capture_key is None
    assert not state.ipc_handles_consumed


@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_cancelled_put_finishes_before_releasing_lease(
    cpu_cuda: list[str], cancel_count: int
) -> None:
    rpc, state = _Rpc(), _state("call-a")
    state.lease = _lease("call-a")
    state.ipc_handles_consumed = True
    host = hosting.GpuCaptureHost(rpc, torch.device("cuda:0"))
    started, unblock, finished = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def put() -> str:
        loop.call_soon_threadsafe(started.set)
        assert unblock.wait(timeout=5)
        cpu_cuda.append("put-completed")
        finished.set()
        return "done"

    task = asyncio.create_task(host.finish(state, put))
    await asyncio.wait_for(started.wait(), timeout=5)
    try:
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        await asyncio.wait({task}, timeout=0.05)
    finally:
        unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # This also waits for a miscancelled worker thread before test teardown.
        assert await asyncio.to_thread(finished.wait, 5)
    assert cpu_cuda == ["put-completed", "sync"]
    assert rpc.calls == [
        ("release_gpu_output_capture", ("lease-call-a",)),
        ("discard_gpu_output_capture", ("call-a",)),
    ]


async def test_concurrent_calls_bind_and_release_independent_payloads(
    monkeypatch: pytest.MonkeyPatch, cpu_cuda: list[str]
) -> None:
    rpc, sink = _Rpc(), _Sink()
    states = [_state("a", sink), _state("b", sink)]
    rpc.responses["export_gpu_output_capture"] = lambda key, *_: [_lease(key)]
    tokens = {"a": torch.tensor([21]), "b": torch.tensor([42])}
    monkeypatch.setattr(
        hosting,
        "import_gpu_output_lease",
        lambda lease, device: GpuOutputTensors(
            tokens[lease.capture_key], torch.tensor([-0.25]), None
        ),
    )
    host = hosting.GpuCaptureHost(rpc, torch.device("cuda:0"))

    async def complete(state: hosting.CapturedModelCall) -> None:
        key = state.capture_key
        await host.bind(
            state, generated_token_count=1, routed_experts_dtype=torch.int16
        )
        await host.finish(
            state, lambda: state.gpu_sink.stage(SimpleNamespace(staging_key=key))
        )

    await asyncio.gather(*(complete(state) for state in states))
    assert sink.payloads["a"].generated_token_ids is tokens["a"]
    assert sink.payloads["b"].generated_token_ids is tokens["b"]
    assert sorted(
        args[0] for method, args in rpc.calls if method == "release_gpu_output_capture"
    ) == ["lease-a", "lease-b"]
    assert all(state.lease is None and state.capture_key is None for state in states)


async def test_repeated_cancel_during_import_waits_before_releasing(
    monkeypatch: pytest.MonkeyPatch, cpu_cuda: list[str]
) -> None:
    rpc, state = _Rpc(), _state("call-a")
    rpc.responses["export_gpu_output_capture"] = [_lease("call-a")]
    host = hosting.GpuCaptureHost(rpc, torch.device("cuda:0"))
    started, unblock = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def import_payload(lease: GpuOutputLease, device: torch.device) -> GpuOutputTensors:
        loop.call_soon_threadsafe(started.set)
        assert unblock.wait(timeout=5)
        cpu_cuda.append("import-completed")
        return GpuOutputTensors(torch.tensor([21]), torch.tensor([-0.25]), None)

    monkeypatch.setattr(hosting, "import_gpu_output_lease", import_payload)

    async def bind_and_cleanup() -> None:
        try:
            await host.bind(
                state, generated_token_count=1, routed_experts_dtype=torch.int16
            )
        finally:
            await host.release(state)

    task = asyncio.create_task(bind_and_cleanup())
    await asyncio.wait_for(started.wait(), timeout=5)
    try:
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        completed, _ = await asyncio.wait({task}, timeout=0.05)
        assert not completed
        assert not cpu_cuda
    finally:
        unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cpu_cuda == ["import-completed", "sync"]
    assert state.lease is None and state.capture_key is None


@pytest.mark.parametrize("lease_count", [0, 2])
async def test_ambiguous_export_discards_request_without_importing(
    cpu_cuda: list[str], lease_count: int
) -> None:
    rpc, state = _Rpc(), _state("call-a")
    rpc.responses["export_gpu_output_capture"] = [_lease("call-a")] * lease_count
    host = hosting.GpuCaptureHost(rpc, torch.device("cuda:0"))
    try:
        with pytest.raises(RuntimeError, match="exactly one payload lease"):
            await host.bind(
                state, generated_token_count=1, routed_experts_dtype=torch.int16
            )
    finally:
        await host.release(state)
    assert rpc.calls[-1] == ("discard_gpu_output_capture", ("call-a",))
    assert not cpu_cuda
