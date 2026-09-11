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
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD  # noqa: E402
from nemo_rl.models.generation.vllm import gpu_capture_host as hosting  # noqa: E402
from nemo_rl.models.generation.vllm.gpu_output_capture import (  # noqa: E402
    GpuOutputCaptureCapabilities,
    GpuOutputImportError,
    GpuOutputLease,
    GpuOutputTensors,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts  # noqa: E402

pytestmark = [pytest.mark.nemo_gym, pytest.mark.asyncio]
CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


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

    async def bind(
        self,
        *,
        count: int = 1,
        dtype: torch.dtype = torch.int16,
        envelope: str | None = None,
    ) -> None:
        await self.host.bind(
            self.state,
            generated_token_count=count,
            routed_experts_dtype=dtype,
            routed_experts_cpu=envelope,
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
        "remote_host",
        "no_device",
        "unsupported_native",
    ],
)
async def test_create_uses_visible_owner_or_preserves_cpu_path(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
) -> None:
    owner = GpuOutputCaptureCapabilities(True, socket.gethostname(), "gpu-0", 1)
    case.rpc.responses["configure_gpu_output_capture"] = {
        "available": [replace(owner, owner=False), owner],
        "nondefault_device": [replace(owner, gpu_uuid="gpu-1")],
        "invisible_device": [replace(owner, gpu_uuid="gpu-2")],
        "no_owner": [None],
        "two_owners": [owner, owner],
        "remote_host": [replace(owner, hostname="remote-host")],
        "no_device": [owner],
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


async def test_bind_preserves_tensors_coordinates_and_put_thread(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    cpu_cuda: list[str],
) -> None:
    routes = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.int64)
    case.tensors = replace(case.tensors, routed_experts=routes)
    monkeypatch.setattr(
        hosting,
        "decode_routed_experts",
        lambda *_: pytest.fail("uncached request decoded CPU routes"),
    )
    await case.bind(envelope="unused-invalid-envelope")
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
    assert payload.routed_experts.dtype == torch.int16
    assert torch.equal(payload.routed_experts, routes)
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


@CUDA_REQUIRED
@pytest.mark.parametrize("routing_dtype", [torch.int8, torch.int16, torch.int32])
async def test_cached_prefix_uploads_only_missing_intervals(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    routing_dtype: torch.dtype,
) -> None:
    case.state.prompt_token_ids = [11, 12, 13, 14, 15]
    ranges = ((0, 2), (3, 4))
    case.lease = replace(case.lease, routed_experts_prefix_backfill_ranges=ranges)
    fresh = torch.arange(28, dtype=torch.int16).reshape(7, 2, 2)
    case.tensors = GpuOutputTensors(
        torch.tensor([21, 22], device="cuda"),
        torch.tensor([-0.25, -0.5], device="cuda"),
        fresh.to(device="cuda", dtype=torch.uint16),
    )
    cpu_routes = fresh + 50  # Fresh CPU rows deliberately disagree with GPU sources.
    cpu_routes[0, 0, 0] = -1
    expected = fresh.to(dtype=routing_dtype, copy=True)
    for start, end in ranges:
        expected[start:end] = cpu_routes[start:end].to(dtype=routing_dtype)
    transfers = []
    original_copy = torch.Tensor.copy_

    def copy_with_accounting(
        destination: torch.Tensor, source: torch.Tensor, non_blocking: bool = False
    ) -> torch.Tensor:
        if destination.is_cuda and source.device.type == "cpu":
            transfers.append(
                (
                    destination.storage_offset(),
                    tuple(source.shape),
                    source.numel() * source.element_size(),
                )
            )
        return original_copy(destination, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_with_accounting)
    await case.bind(
        count=2, dtype=routing_dtype, envelope=encode_routed_experts(cpu_routes)
    )
    assert (await case.finish()).ok
    payload = case.sink.payloads["call"]
    assert payload.generated_token_ids is case.tensors.generated_token_ids
    assert payload.generated_logprobs is case.tensors.generation_logprobs
    assert (
        payload.routed_experts.is_cuda and payload.routed_experts.dtype == routing_dtype
    )
    assert torch.equal(payload.routed_experts.cpu(), expected)
    assert torch.equal(case.tensors.routed_experts.cpu(), fresh.to(torch.uint16))
    assert transfers == [
        (0, (2, 2, 2), 8 * routing_dtype.itemsize),
        (12, (1, 2, 2), 4 * routing_dtype.itemsize),
    ]
    case.assert_released()


@CUDA_REQUIRED
@pytest.mark.parametrize("routing_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("prev_len", [0, 3], ids=["first-turn", "continuation"])
async def test_cached_prefix_put_fields_equal_cpu_bytes(
    case: _Case,
    routing_dtype: torch.dtype,
    prev_len: int,
) -> None:
    case.state.prompt_token_ids = [11, 12, 13, 14, 15]
    case.lease = replace(
        case.lease, routed_experts_prefix_backfill_ranges=((0, 2), (3, 4))
    )
    routes = torch.arange(28, dtype=routing_dtype).reshape(7, 2, 2)
    routes[0, 0, 0] = -1
    routes[-1] = torch.tensor([[0, 1], [0, 1]], dtype=routing_dtype)
    native_routes = routes.to(dtype=torch.uint16, device="cuda")
    native_routes[:2] = 0
    native_routes[3:4] = 0
    case.tensors = GpuOutputTensors(
        torch.tensor([21, 22], device="cuda"),
        torch.tensor([-0.25, -0.5], device="cuda"),
        native_routes,
    )
    cpu_fields = {
        "token_ids_delta": torch.tensor([[11, 12, 13, 14, 15, 21, 22]])[:, prev_len:],
        "token_mask_delta": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])[
            :, prev_len:
        ],
        "generation_logprobs_delta": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, -0.25, -0.5]]
        )[:, prev_len:],
        ROUTED_EXPERTS_FIELD: routes[prev_len:].unsqueeze(0),
    }
    record = SimpleNamespace(
        staging_key="call", prev_len=prev_len, delta_len=7 - prev_len, cum_len=7
    )
    staged_fields = {}

    def put() -> StageResult:
        result = case.put(record)
        staged_fields.update(
            case.sink.payloads["call"].staging_fields(record, cpu_fields)
        )
        return result

    await case.bind(
        count=2, dtype=routing_dtype, envelope=encode_routed_experts(routes)
    )
    assert (await case.host.finish(case.state, put)).ok
    assert staged_fields.keys() == cpu_fields.keys()
    for name, expected in cpu_fields.items():
        actual = staged_fields[name]
        assert (
            actual.is_cuda
            and actual.dtype == expected.dtype
            and actual.shape == expected.shape
        )
        assert torch.equal(
            actual.cpu().contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        ), name


@pytest.mark.parametrize(
    ("ranges", "count", "source_kind"),
    [
        (((0, 1),), 1, "missing"),
        (((0, 1),), 1, "wrong_shape"),
        (((0, 2), (1, 2)), 1, "valid"),
        (((2, 3),), 1, "valid"),
        (((1, 2),), 0, "valid"),
    ],
)
async def test_invalid_prefix_uses_cpu_put_and_releases_lease(
    case: _Case,
    cpu_cuda: list[str],
    ranges: tuple[tuple[int, int], ...],
    count: int,
    source_kind: str,
) -> None:
    case.lease = replace(case.lease, routed_experts_prefix_backfill_ranges=ranges)
    routes = torch.arange(2 + count, dtype=torch.int16).reshape(-1, 1, 1)
    original = routes.clone()
    case.tensors = replace(case.tensors, routed_experts=routes)
    envelope = {
        "missing": None,
        "wrong_shape": encode_routed_experts(routes[1:]),
        "valid": encode_routed_experts(routes),
    }[source_kind]
    await case.bind(count=count, envelope=envelope)
    assert (await case.finish()).ok and case.sink.payloads == {"call": None}
    assert torch.equal(routes, original) and cpu_cuda == ["sync"]
    case.assert_released()


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
        await case.host.bind(
            state, generated_token_count=1, routed_experts_dtype=torch.int16
        )
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
