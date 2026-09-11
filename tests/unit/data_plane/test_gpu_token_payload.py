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
"""GPU capture transport retains source storage and Gym's committed bytes."""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture  # noqa: E402
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StagedCallRecord,
    StageResult,
)

from nemo_rl.data_plane.gpu_token_payload import (  # noqa: E402
    BoundGpuTokenSink,
    GpuTokenPayload,
)
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    TQTokenSink,
    _row_to_base_snapshot,
    _row_to_route_fragment,
)
from nemo_rl.experience.route_assembly import verify_route_fragment_integrity  # noqa: E402
from nemo_rl.utils.routed_experts_codec import encode_routed_experts  # noqa: E402

pytestmark = pytest.mark.nemo_gym
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


class _RecordSink:
    def __init__(self) -> None:
        self.record: StagedCallRecord | None = None

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.record = record
        return StageResult(ok=True, staging_key=record.staging_key)


class _PutClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.puts: list[dict[str, Any]] = []
        self.fail = fail

    def put_samples(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("PUT rejected")
        self.puts.append(kwargs)


def _record(
    *, prev_len: int = 0, prompt_len: int = 2, routes: torch.Tensor | None = None
) -> StagedCallRecord:
    prompt = list(range(20, 20 + prompt_len))
    sink = _RecordSink()
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 7)
    admission = CaptureAdmission(
        rollout_id="rollout",
        model_call_id="call",
        mode="token_in" if prev_len else "text",
        prev_len=prev_len,
        parent_call_id="parent" if prev_len else None,
        parent_chain_hash="a" * 64 if prev_len else None,
        required_prefix_token_ids=prompt[:prev_len],
    )
    call = capture.begin_call(admission)
    result = capture.complete_call(
        call,
        prompt_token_ids=prompt,
        generated_token_ids=[31, 32],
        generated_logprobs=[-0.125, -0.0],
        extras={"routed_experts": encode_routed_experts(routes[prev_len:])}
        if routes is not None
        else None,
    )
    assert result.disposition == "staged"
    assert sink.record is not None
    return sink.record


def _cpu_payload(*, prompt_len: int = 2) -> GpuTokenPayload:
    return GpuTokenPayload(
        prompt_len=prompt_len,
        generated_token_ids=torch.tensor([31, 32]),
        generated_logprobs=torch.tensor([-0.125, -0.0]),
    )


def test_cpu_stage_keeps_existing_wire_and_digest() -> None:
    client = _PutClient()
    record = _record()
    assert TQTokenSink(client, staging_partition="staging").stage(record).ok
    assert len(client.puts) == 1
    fields = client.puts[0]["fields"]
    assert all(value.device.type == "cpu" for value in fields.values())
    assert _row_to_base_snapshot(fields).model_dump() == record.model_dump(
        exclude={"extras"}
    )


def test_bound_sink_requires_payload_and_reports_export_failure() -> None:
    client = _PutClient()
    sink = TQTokenSink(client, staging_partition="staging")
    record = _record()
    assert not BoundGpuTokenSink(sink).stage(record).ok
    bound = BoundGpuTokenSink(sink)
    bound.fail("worker export failed")
    assert bound.stage(record).error == "worker export failed"
    assert client.puts == []


def test_cpu_reconstruction_cannot_claim_gpu_capture() -> None:
    client = _PutClient()
    bound = BoundGpuTokenSink(TQTokenSink(client, staging_partition="staging"))
    bound.bind(_cpu_payload())
    result = bound.stage(_record())
    assert not result.ok
    assert "original CUDA tensors" in result.error
    assert client.puts == []


def test_bound_sink_releases_payload_after_failed_stage() -> None:
    class FailingSink:
        def stage(self, record: Any, *, gpu_payload: Any) -> StageResult:
            raise RuntimeError("staging failed")

    bound = BoundGpuTokenSink(FailingSink())
    payload = _cpu_payload()
    payload_ref = weakref.ref(payload)
    bound.bind(payload)
    del payload
    with pytest.raises(RuntimeError, match="staging failed"):
        bound.stage(_record())
    assert payload_ref() is None
    assert not bound.stage(_record()).ok
    bound.clear()


def test_request_bindings_are_isolated_and_single_use() -> None:
    barrier = threading.Barrier(2)

    class ConcurrentSink:
        def stage(self, record: Any, *, gpu_payload: GpuTokenPayload) -> StageResult:
            barrier.wait(timeout=5)
            return StageResult(ok=True, staging_key=str(gpu_payload.prompt_len))

    shared = ConcurrentSink()
    bindings = [BoundGpuTokenSink(shared), BoundGpuTokenSink(shared)]
    for index, bound in enumerate(bindings):
        bound.bind(_cpu_payload(prompt_len=index))
        with pytest.raises(RuntimeError, match="only bind once"):
            bound.bind(_cpu_payload())
    record = _record()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda bound: bound.stage(record), bindings))
    assert [result.staging_key for result in results] == ["0", "1"]
    assert all(not bound.stage(record).ok for bound in bindings)


@requires_cuda
@pytest.mark.parametrize("prev_len,prompt_len", [(0, 2), (2, 2), (2, 3)])
def test_gpu_fields_match_committed_wire_without_rebuilding_generated_values(
    prev_len: int, prompt_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    routes = torch.arange(
        (prompt_len + 2) * 4, dtype=torch.int16, device="cuda"
    ).reshape(-1, 2, 2)
    payload = GpuTokenPayload(
        prompt_len=prompt_len,
        generated_token_ids=torch.tensor([31, 32], dtype=torch.int64, device="cuda"),
        generated_logprobs=torch.tensor(
            [-0.125, -0.0], dtype=torch.float32, device="cuda"
        ),
        routed_experts=routes,
    )
    record = _record(prev_len=prev_len, prompt_len=prompt_len, routes=routes)
    cpu_client, gpu_client = _PutClient(), _PutClient()
    assert TQTokenSink(cpu_client, staging_partition="staging").stage(record).ok
    h2d_sizes = []
    copy = torch.Tensor.copy_

    def tracked_copy(
        destination: torch.Tensor, source: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if destination.is_cuda and not source.is_cuda:
            h2d_sizes.append(source.numel())
        return copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", tracked_copy)
    bound = BoundGpuTokenSink(TQTokenSink(gpu_client, staging_partition="staging"))
    bound.bind(payload)
    assert bound.stage(record).ok
    assert len(gpu_client.puts) == 1
    expected, actual = cpu_client.puts[0]["fields"], gpu_client.puts[0]["fields"]
    assert gpu_client.puts[0]["tags"] == cpu_client.puts[0]["tags"]
    for name in expected.keys():
        assert torch.equal(
            actual[name].cpu().contiguous().view(torch.uint8),
            expected[name].contiguous().view(torch.uint8),
        )
    for name in (
        "token_ids_delta",
        "generation_logprobs_delta",
        "token_mask_delta",
        "routed_experts",
    ):
        assert actual[name].is_cuda
    assert (
        actual["routed_experts"].untyped_storage().data_ptr()
        == routes.untyped_storage().data_ptr()
    )
    carry_len = prompt_len - prev_len
    assert h2d_sizes == ([carry_len] if carry_len else [])
    if not carry_len:
        assert (
            actual["token_ids_delta"].data_ptr()
            == payload.generated_token_ids.data_ptr()
        )
        assert (
            actual["generation_logprobs_delta"].data_ptr()
            == payload.generated_logprobs.data_ptr()
        )
    snapshot = _row_to_base_snapshot(actual)
    assert snapshot.model_dump() == record.model_dump(exclude={"extras"})
    fragment = _row_to_route_fragment(actual)
    assert fragment is not None
    assert verify_route_fragment_integrity(
        fragment,
        extras_digest_version=record.extras_digest_version,
        expected_extras_digest=record.extras_digest,
    )


@requires_cuda
@pytest.mark.parametrize(
    "corrupt", ["ids", "logprobs", "routes", "prompt_len", "missing_routes"]
)
def test_gpu_mirror_mismatch_fails_before_put(corrupt: str) -> None:
    routes = torch.arange(16, dtype=torch.int16, device="cuda").reshape(4, 2, 2)
    record = _record(routes=routes)
    ids = torch.tensor([31, 32], dtype=torch.int64, device="cuda")
    logprobs = torch.tensor([-0.125, -0.0], dtype=torch.float32, device="cuda")
    if corrupt == "ids":
        ids[0] = 99
    elif corrupt == "logprobs":
        logprobs[-1] = 0.0  # Equal numerically, but a different committed bit pattern.
    elif corrupt == "routes":
        routes[0, 0, 0] = 99
    client = _PutClient()
    payload = GpuTokenPayload(
        prompt_len=3 if corrupt == "prompt_len" else 2,
        generated_token_ids=ids,
        generated_logprobs=logprobs,
        routed_experts=None if corrupt == "missing_routes" else routes,
        validate_cpu_mirror=True,
    )
    result = TQTokenSink(client, staging_partition="staging").stage(
        record, gpu_payload=payload
    )
    assert not result.ok
    assert client.puts == []


@requires_cuda
def test_normal_gpu_stage_does_not_copy_payload_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = torch.arange(16, dtype=torch.int16, device="cuda").reshape(4, 2, 2)
    record = _record(routes=routes)
    payload = GpuTokenPayload(
        prompt_len=2,
        generated_token_ids=torch.tensor([31, 32], dtype=torch.int32, device="cuda"),
        generated_logprobs=torch.tensor(
            [-0.125, -0.0], dtype=torch.float16, device="cuda"
        ),
        routed_experts=routes,
    )
    original_cpu = torch.Tensor.cpu

    def reject_payload_copy(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if tensor.is_cuda:
            raise AssertionError("normal GPU staging must not add a payload D2H copy")
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", reject_payload_copy)
    client = _PutClient()
    result = TQTokenSink(client, staging_partition="staging").stage(
        record, gpu_payload=payload
    )
    assert result.ok, result.error
    assert len(client.puts) == 1
    fields = client.puts[0]["fields"]
    assert fields["token_ids_delta"].dtype == torch.int64
    assert fields["generation_logprobs_delta"].dtype == torch.float32
    assert all(
        fields[name].is_cuda
        for name in (
            "token_ids_delta",
            "generation_logprobs_delta",
            "token_mask_delta",
            "routed_experts",
        )
    )


@requires_cuda
def test_gpu_fields_are_ready_for_backend_executor_on_another_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(prev_len=2)
    payload = GpuTokenPayload(
        prompt_len=2,
        generated_token_ids=torch.tensor([31, 32], dtype=torch.int64, device="cuda"),
        generated_logprobs=torch.tensor(
            [-0.125, -0.0], dtype=torch.float32, device="cuda"
        ),
    )
    torch.cuda.synchronize()
    caller_thread = threading.get_ident()
    caller_stream = torch.cuda.Stream()
    ready_events: list[torch.cuda.Event] = []
    original_zeros = torch.zeros

    def delayed_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        assert torch.cuda.current_stream() == caller_stream
        # Delay the mask producer so a missing stream fence is observable before
        # the executor attempts its own GPU read and any implicit synchronization.
        torch.cuda._sleep(100_000_000)
        result = original_zeros(*args, **kwargs)
        ready = torch.cuda.Event()
        ready.record()
        ready_events.append(ready)
        return result

    monkeypatch.setattr(torch, "zeros", delayed_zeros)

    def backend_read(fields: Any) -> None:
        assert threading.get_ident() != caller_thread
        with torch.cuda.device(payload.device()):
            assert torch.cuda.current_stream() == torch.cuda.default_stream()
            assert len(ready_events) == 1
            assert ready_events[0].query(), "backend observed unfinished field writes"
            assert fields["token_mask_delta"].cpu().tolist() == [[1.0, 1.0]]
            assert fields["token_ids_delta"].cpu().tolist() == [[31, 32]]

    with ThreadPoolExecutor(max_workers=1) as executor:

        class ExecutorClient:
            def put_samples(self, **kwargs: Any) -> None:
                executor.submit(backend_read, kwargs["fields"]).result(timeout=10)

        with torch.cuda.stream(caller_stream):
            result = TQTokenSink(ExecutorClient(), staging_partition="staging").stage(
                record, gpu_payload=payload
            )
        assert result.ok, result.error
