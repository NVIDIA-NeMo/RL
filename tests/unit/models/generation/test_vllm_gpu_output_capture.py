# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import multiprocessing
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.vllm.gpu_output_capture import (
    GPU_CAPTURE_KEY,
    CudaTensorIpc,
    GpuOutputCapture,
    GpuOutputImportError,
    GpuOutputLease,
    import_gpu_output_lease,
)


def _runner(*, key: str = "call", routes: bool = True) -> SimpleNamespace:
    params = SimpleNamespace(n=1, extra_args={GPU_CAPTURE_KEY: key}, logprobs=0)
    request = SimpleNamespace(sampling_params=params, num_prompt_tokens=3)

    def bookkeeping(
        scheduler_output: Any,
        sampler_output: Any,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        num_scheduled_tokens: int,
    ) -> str:
        return "original-serving-output"

    return SimpleNamespace(
        device=torch.device("cuda", 0),
        _bookkeeping_sync=bookkeeping,
        input_batch=SimpleNamespace(
            req_ids=["native"], num_computed_tokens_cpu=np.array([0])
        ),
        requests={"native": request},
        discard_request_mask=SimpleNamespace(np=np.array([True])),
        routed_experts_initialized=routes,
    )


def _step(
    runner: SimpleNamespace,
    capture: GpuOutputCapture,
    *,
    start: int,
    count: int,
    token: int,
    logprob: float,
    route_values: list[int],
    discard: bool = False,
    call_hook: bool = False,
) -> None:
    runner.input_batch.num_computed_tokens_cpu[0] = start
    runner.discard_request_mask.np[0] = discard
    routes = torch.tensor(route_values, dtype=torch.uint16, device="cuda").reshape(
        count, 1, 2
    )
    runner.routed_experts_capturer = SimpleNamespace(get_device_buffer=lambda: routes)
    ids = torch.tensor([[token]], dtype=torch.int32, device="cuda")
    logs = torch.tensor([[logprob, -9.0]], device="cuda")
    sampler = SimpleNamespace(
        sampled_token_ids=ids, logprobs_tensors=SimpleNamespace(logprobs=logs)
    )
    scheduler = SimpleNamespace(num_scheduled_tokens={"native": count})
    if call_hook:
        result = runner._bookkeeping_sync(scheduler, sampler, None, None, count)
        assert result == "original-serving-output"
    else:
        capture.capture_step(scheduler, sampler)
    # Native sampler and router scratch are reused after the step.
    ids.fill_(999)
    logs.fill_(-99)
    routes.fill_(255)


def _ipc_child(lease: GpuOutputLease, queue: Any) -> None:
    try:
        tensors = import_gpu_output_lease(lease, torch.device("cuda", 0))
        queue.put(
            (
                tensors.generated_token_ids.cpu().tolist(),
                tensors.generation_logprobs.cpu().tolist(),
                tensors.routed_experts.cpu().tolist(),
            )
        )
        del tensors
        torch.cuda.synchronize()
    except Exception as error:
        queue.put((type(error).__name__, str(error)))


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@cuda_required
def test_native_payload_survives_scratch_reuse_and_cross_process_ipc() -> None:
    runner = _runner()
    capture = GpuOutputCapture(
        runner, max_retained_bytes=4096, require_routed_experts=True
    )
    capture.install()
    capture_stream = torch.cuda.Stream()
    export_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        _step(
            runner,
            capture,
            start=0,
            count=2,
            token=0,
            logprob=0,
            route_values=[10, 11, 12, 13],
            discard=True,
            call_hook=True,
        )
        _step(
            runner,
            capture,
            start=2,
            count=1,
            token=7,
            logprob=-0.7,
            route_values=[14, 15],
            call_hook=True,
        )
        # An extra async-scheduled step must be trimmed at the final stop.
        _step(
            runner,
            capture,
            start=3,
            count=1,
            token=8,
            logprob=-0.8,
            route_values=[16, 17],
            call_hook=True,
        )
    with torch.cuda.stream(export_stream):
        lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_ipc_child, args=(lease, queue))
    child.start()
    result = queue.get(timeout=30)
    child.join(timeout=30)
    assert not child.is_alive()
    assert child.exitcode == 0
    assert result[0] == [7]
    assert result[1] == pytest.approx([-0.7])
    assert result[2] == [[[10, 11]], [[12, 13]], [[14, 15]], [[0, 1]]]
    with pytest.raises(RuntimeError, match="outstanding IPC leases"):
        capture.clear()
    capture.release(lease.lease_id)
    assert capture.retained_bytes == 0
    # A queued step after finalization must not reallocate discarded output.
    _step(
        runner, capture, start=4, count=1, token=9, logprob=-0.9, route_values=[18, 19]
    )
    assert capture.retained_bytes == 0
    capture.clear()


@cuda_required
def test_budget_failure_and_abort_release_all_payloads() -> None:
    runner = _runner()
    capture = GpuOutputCapture(
        runner, max_retained_bytes=4, require_routed_experts=True
    )
    with pytest.raises(RuntimeError, match="budget exceeded"):
        _step(
            runner,
            capture,
            start=0,
            count=2,
            token=0,
            logprob=0,
            route_values=[1, 2, 3, 4],
            discard=True,
        )
    assert capture.retained_bytes == 0
    capture.max_retained_bytes = 4096
    _step(
        runner,
        capture,
        start=0,
        count=3,
        token=7,
        logprob=-0.7,
        route_values=[1, 2, 3, 4, 5, 6],
    )
    assert capture.retained_bytes > 0
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    # Cancellation before import explicitly releases unused IPC refcounters.
    capture.abandon_unimported(lease.lease_id)
    capture.discard("call")
    assert capture.retained_bytes == 0
    capture.clear()


@cuda_required
def test_missing_routes_are_rejected_instead_of_returning_incomplete_payload() -> None:
    runner = _runner()
    capture = GpuOutputCapture(
        runner, max_retained_bytes=4096, require_routed_experts=True
    )
    _step(runner, capture, start=2, count=1, token=7, logprob=-0.7, route_values=[1, 2])
    with pytest.raises(RuntimeError, match="does not cover"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)
    capture.discard("call")
    assert capture.retained_bytes == 0


@cuda_required
def test_import_rejects_wrong_host_and_physical_gpu_before_opening_ipc() -> None:
    runner = _runner()
    capture = GpuOutputCapture(
        runner, max_retained_bytes=4096, require_routed_experts=False
    )
    _step(
        runner,
        capture,
        start=0,
        count=3,
        token=7,
        logprob=-0.7,
        route_values=[1, 2, 3, 4, 5, 6],
    )
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    with pytest.raises(GpuOutputImportError, match="same-host") as error:
        import_gpu_output_lease(replace(lease, hostname="wrong-host"), 0)
    assert not error.value.handles_consumed
    with pytest.raises(GpuOutputImportError, match="UUID mismatch") as error:
        import_gpu_output_lease(replace(lease, gpu_uuid="wrong-gpu"), 0)
    assert not error.value.handles_consumed
    capture.abandon_unimported(lease.lease_id)
    capture.discard("call")


@cuda_required
def test_untagged_requests_do_not_allocate_training_payloads() -> None:
    runner = _runner()
    runner.requests["native"].sampling_params.extra_args = {}
    capture = GpuOutputCapture(
        runner, max_retained_bytes=1, require_routed_experts=False
    )
    _step(
        runner,
        capture,
        start=0,
        count=3,
        token=7,
        logprob=-0.7,
        route_values=[1, 2, 3, 4, 5, 6],
    )
    assert capture.retained_bytes == 0
    with pytest.raises(RuntimeError, match="No retained GPU output"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)


@cuda_required
def test_hook_rejects_changed_upstream_signature() -> None:
    runner = _runner()
    runner._bookkeeping_sync = lambda scheduler_output: None
    capture = GpuOutputCapture(
        runner, max_retained_bytes=100, require_routed_experts=False
    )
    with pytest.raises(RuntimeError, match="signature"):
        capture.install()


@cuda_required
def test_capture_failure_preserves_serving_and_fails_only_payload_export() -> None:
    runner = _runner()
    capture = GpuOutputCapture(
        runner, max_retained_bytes=1, require_routed_experts=True
    )
    capture.install()
    _step(
        runner,
        capture,
        start=0,
        count=3,
        token=7,
        logprob=-0.7,
        route_values=[1, 2, 3, 4, 5, 6],
        call_hook=True,
    )
    assert capture.retained_bytes == 0
    with pytest.raises(RuntimeError, match="capture failed.*budget exceeded"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)
    capture.discard("call")
    capture.clear()


@cuda_required
def test_reprefill_preserves_already_emitted_routes_and_logprobs() -> None:
    runner = _runner()
    # A per-request -1 becomes vocab_size at the native batch boundary.
    runner.requests["native"].sampling_params.logprobs = -1
    capture = GpuOutputCapture(
        runner, max_retained_bytes=4096, require_routed_experts=True
    )
    _step(
        runner,
        capture,
        start=0,
        count=3,
        token=7,
        logprob=-0.7,
        route_values=[1, 2, 3, 4, 5, 6],
    )
    # Recomputed prefix differs, but vLLM only emits the last/new route after
    # preemption. Frozen routes from the first output must remain unchanged.
    _step(
        runner,
        capture,
        start=0,
        count=4,
        token=8,
        logprob=-0.8,
        route_values=[91, 92, 93, 94, 95, 96, 7, 8],
    )
    lease = capture.export("call", generated_token_count=2, prompt_token_count=3)
    tensors = capture._leases[lease.lease_id].tensors
    assert tensors.generated_token_ids.tolist() == [7, 8]
    assert tensors.generation_logprobs.tolist() == pytest.approx([-0.7, -0.8])
    assert tensors.routed_experts.tolist() == [
        [[1, 2]],
        [[3, 4]],
        [[5, 6]],
        [[7, 8]],
        [[0, 1]],
    ]
    del tensors
    capture.abandon_unimported(lease.lease_id)
    assert capture.retained_bytes == 0


@cuda_required
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_import_never_decrements_the_failed_attempt_twice(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    from nemo_rl.models.generation.vllm import gpu_output_capture as module

    runner = _runner()
    capture = GpuOutputCapture(
        runner, max_retained_bytes=4096, require_routed_experts=True
    )
    _step(
        runner,
        capture,
        start=0,
        count=3,
        token=7,
        logprob=-0.7,
        route_values=[1, 2, 3, 4, 5, 6],
    )
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    release_counter = module._release_unopened_handle
    attempted = []
    cleanup = []

    def imported_then_failed(handle: CudaTensorIpc, device: int) -> torch.Tensor:
        attempted.append(handle)
        # Simulate successful counter consumption inside PyTorch, including
        # the failing second descriptor's post-import tensor-rebuild failure.
        release_counter(handle)
        if len(attempted) == 2:
            raise RuntimeError("failed after consuming counter")
        return torch.empty(1, device="cuda")

    def cleanup_unopened(handle: CudaTensorIpc) -> None:
        cleanup.append(handle)
        release_counter(handle)

    monkeypatch.setattr(module, "_import_tensor", imported_then_failed)
    monkeypatch.setattr(module, "_release_unopened_handle", cleanup_unopened)
    # This unit test simulates tensor-import failures in the producer process;
    # CUDA event IPC itself is exercised by the real spawned-consumer test.
    monkeypatch.setattr(
        torch.cuda.Event,
        "from_ipc_handle",
        staticmethod(lambda device, handle: capture._leases[lease.lease_id].ready),
    )
    if cleanup_fails:

        def failed_sync(device: int) -> None:
            raise RuntimeError("injected cleanup synchronization failure")

        monkeypatch.setattr(torch.cuda, "synchronize", failed_sync)
    with pytest.raises(GpuOutputImportError) as error:
        import_gpu_output_lease(lease, 0)
    assert error.value.handles_consumed
    if cleanup_fails:
        assert "IPC cleanup failed" in str(error.value)
        assert cleanup == []
        # This descriptor was never attempted; clean the injected-failure
        # test's remaining counter without touching either consumed one.
        release_counter(lease.routed_experts)
    else:
        assert cleanup == [lease.routed_experts]
    capture.release(lease.lease_id)
    assert capture.retained_bytes == 0


@cuda_required
@pytest.mark.parametrize("raw_logprob", [float("-inf"), -10000.0, -9999.0, -9998.5])
def test_gpu_logprobs_match_serving_floor_before_staging(raw_logprob: float) -> None:
    from nemo_rl.models.generation.vllm.utils import VLLM_LOGPROB_FLOOR

    runner = _runner()
    capture = GpuOutputCapture(
        runner, max_retained_bytes=4096, require_routed_experts=False
    )
    _step(
        runner,
        capture,
        start=0,
        count=3,
        token=7,
        logprob=raw_logprob,
        route_values=[1, 2, 3, 4, 5, 6],
    )
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    normalized = capture._leases[lease.lease_id].tensors.generation_logprobs
    assert normalized.is_cuda
    assert normalized.item() == max(raw_logprob, VLLM_LOGPROB_FLOOR)
    del normalized
    capture.abandon_unimported(lease.lease_id)
    assert capture.retained_bytes == 0
