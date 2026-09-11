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
from dataclasses import replace
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
from nemo_rl.experience.route_assembly import (
    verify_route_fragment_integrity,  # noqa: E402
)
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
        self.puts.append(kwargs)
        if self.fail:
            raise RuntimeError("PUT rejected")


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


def _payload(
    *, prompt_len: int = 2, routes: torch.Tensor | None = None, device: str = "cuda"
) -> GpuTokenPayload:
    return GpuTokenPayload(
        prompt_len=prompt_len,
        generated_token_ids=torch.tensor([31, 32], dtype=torch.int64, device=device),
        generated_logprobs=torch.tensor(
            [-0.125, -0.0], dtype=torch.float32, device=device
        ),
        routed_experts=routes,
    )


def _assert_same_put(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    assert actual["tags"] == expected["tags"]
    assert actual["sample_ids"] == expected["sample_ids"]
    assert actual["partition_id"] == expected["partition_id"]
    assert actual["fields"].keys() == expected["fields"].keys()
    for name, expected_tensor in expected["fields"].items():
        actual_tensor = actual["fields"][name]
        assert actual_tensor.dtype == expected_tensor.dtype, name
        assert actual_tensor.shape == expected_tensor.shape, name
        assert torch.equal(
            actual_tensor.cpu().contiguous().view(torch.uint8),
            expected_tensor.contiguous().view(torch.uint8),
        ), name


@pytest.mark.parametrize("cleared", [False, True])
def test_missing_gpu_payload_preserves_cpu_put(cleared: bool) -> None:
    original, fallback = _PutClient(), _PutClient()
    record = _record()
    assert TQTokenSink(original, staging_partition="staging").stage(record).ok
    bound = BoundGpuTokenSink(TQTokenSink(fallback, staging_partition="staging"))
    if cleared:
        bound.bind(_payload(device="cpu"))
        bound.clear()
    assert bound.stage(record).ok
    assert len(fallback.puts) == 1
    _assert_same_put(original.puts[0], fallback.puts[0])
    fields = fallback.puts[0]["fields"]
    assert all(value.device.type == "cpu" for value in fields.values())
    assert _row_to_base_snapshot(fields).model_dump() == record.model_dump(
        exclude={"extras"}
    )


def test_bound_sink_releases_payload_after_failed_stage() -> None:
    class FailingSink:
        def stage(self, record: Any, *, gpu_payload: Any) -> StageResult:
            raise RuntimeError("staging failed")

    bound = BoundGpuTokenSink(FailingSink())
    payload = _payload(device="cpu")
    payload_ref = weakref.ref(payload)
    bound.bind(payload)
    del payload
    with pytest.raises(RuntimeError, match="staging failed"):
        bound.stage(_record())
    assert payload_ref() is None
    bound.clear()


def test_concurrent_calls_have_independent_gpu_bindings() -> None:
    barrier = threading.Barrier(2)

    class ConcurrentSink:
        def stage(self, record: Any, *, gpu_payload: GpuTokenPayload) -> StageResult:
            barrier.wait(timeout=5)
            return StageResult(ok=True, staging_key=str(gpu_payload.prompt_len))

    shared = ConcurrentSink()
    bindings = [BoundGpuTokenSink(shared), BoundGpuTokenSink(shared)]
    for index, bound in enumerate(bindings):
        bound.bind(_payload(prompt_len=index, device="cpu"))
    record = _record()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda bound: bound.stage(record), bindings))
    assert [result.staging_key for result in results] == ["0", "1"]


@requires_cuda
@pytest.mark.parametrize(
    "prev_len,prompt_len,routing_dtype",
    [(0, 2, torch.int8), (2, 2, torch.int16), (2, 3, torch.int32)],
)
def test_gpu_fields_match_committed_wire_without_rebuilding_generated_values(
    prev_len: int,
    prompt_len: int,
    routing_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = torch.arange(
        (prompt_len + 2) * 4, dtype=routing_dtype, device="cuda"
    ).reshape(-1, 2, 2)
    payload = _payload(prompt_len=prompt_len, routes=routes)
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

    original_cpu = torch.Tensor.cpu

    def reject_payload_copy(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        assert not tensor.is_cuda, "GPU staging must not copy payloads back to CPU"
        return original_cpu(tensor, *args, **kwargs)

    bound = BoundGpuTokenSink(TQTokenSink(gpu_client, staging_partition="staging"))
    bound.bind(payload)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", tracked_copy)
        patch.setattr(torch.Tensor, "cpu", reject_payload_copy)
        assert bound.stage(record).ok
    assert len(gpu_client.puts) == 1
    _assert_same_put(cpu_client.puts[0], gpu_client.puts[0])
    actual = gpu_client.puts[0]["fields"]
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
    "invalid", ["cpu", "ids_dtype", "logprobs_dtype", "routes_shape", "prompt_len"]
)
def test_invalid_gpu_optimization_preserves_one_cpu_put(invalid: str) -> None:
    routes = torch.arange(16, dtype=torch.int16, device="cuda").reshape(4, 2, 2)
    record = _record(routes=routes)
    payload = _payload(routes=routes)
    if invalid == "cpu":
        payload = _payload(routes=routes.cpu(), device="cpu")
    elif invalid == "ids_dtype":
        payload = replace(
            payload, generated_token_ids=payload.generated_token_ids.int()
        )
    elif invalid == "logprobs_dtype":
        payload = replace(payload, generated_logprobs=payload.generated_logprobs.half())
    elif invalid == "routes_shape":
        payload = replace(payload, routed_experts=routes[:1])
    else:
        payload = replace(payload, prompt_len=3)
    original, fallback = _PutClient(), _PutClient()
    assert TQTokenSink(original, staging_partition="staging").stage(record).ok
    assert (
        TQTokenSink(fallback, staging_partition="staging")
        .stage(record, gpu_payload=payload)
        .ok
    )
    assert len(fallback.puts) == 1
    _assert_same_put(original.puts[0], fallback.puts[0])
    assert all(
        value.device.type == "cpu" for value in fallback.puts[0]["fields"].values()
    )


@requires_cuda
def test_failed_gpu_put_is_not_retried_as_cpu() -> None:
    client = _PutClient(fail=True)
    bound = BoundGpuTokenSink(TQTokenSink(client, staging_partition="staging"))
    bound.bind(_payload())
    result = bound.stage(_record())
    assert not result.ok
    assert "PUT rejected" in result.error
    assert len(client.puts) == 1


@requires_cuda
def test_gpu_fields_are_ready_for_backend_executor_on_another_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(prev_len=2)
    payload = _payload()
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
