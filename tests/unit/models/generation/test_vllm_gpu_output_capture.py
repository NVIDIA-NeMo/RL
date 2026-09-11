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
import socket
import sys
import weakref
from collections.abc import Sequence
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
    configure_gpu_output_capture,
    import_gpu_output_lease,
)


def _runner(
    *, key: str = "call", routes: bool = True, async_scheduling: bool = True
) -> SimpleNamespace:
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

    def sample_tokens(grammar_output: Any) -> SimpleNamespace:
        scheduler, sampler, scratch = runner.native_step
        serving_result = runner._bookkeeping_sync(
            scheduler, sampler, None, None, sum(scheduler.num_scheduled_tokens.values())
        )
        callback = getattr(runner, "after_bookkeeping", None)
        if callback is not None:
            callback()
        # This is the clone already made by native vLLM, after bookkeeping.
        snapshot = (
            SimpleNamespace(routing_data=scratch.clone())
            if async_scheduling and routes
            else None
        )
        return SimpleNamespace(
            _routed_experts=snapshot,
            _sampled_token_ids=sampler.sampled_token_ids,
            _logprobs_tensors=sampler.logprobs_tensors,
            serving_result=serving_result,
        )

    runner = SimpleNamespace(
        sample_tokens=sample_tokens,
        use_async_scheduling=async_scheduling,
        device=torch.device("cuda", 0),
        _bookkeeping_sync=bookkeeping,
        input_batch=SimpleNamespace(
            req_ids=["native"], num_computed_tokens_cpu=np.array([0])
        ),
        requests={"native": request},
        discard_request_mask=SimpleNamespace(np=np.array([True])),
        routed_experts_initialized=routes,
    )
    return runner


def _step(
    runner: SimpleNamespace,
    capture: GpuOutputCapture,
    *,
    start: int = 0,
    count: int = 3,
    token: int = 7,
    logprob: float = -0.7,
    route_values: Sequence[int] = (1, 2, 3, 4, 5, 6),
    discard: bool = False,
    initial_cached_prefix: int | None = None,
    resumed_cached_prefix: int | None = None,
    resumed_output_tokens: int = 0,
) -> SimpleNamespace:
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
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"native": count},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["native"] if resumed_cached_prefix is not None else [],
            resumed_req_ids={"native"} if resumed_cached_prefix is not None else set(),
            num_computed_tokens=[resumed_cached_prefix]
            if resumed_cached_prefix is not None
            else [],
            num_output_tokens=[resumed_output_tokens]
            if resumed_cached_prefix is not None
            else [],
        ),
        scheduled_new_reqs=(
            [
                SimpleNamespace(
                    req_id="native", num_computed_tokens=initial_cached_prefix
                )
            ]
            if initial_cached_prefix is not None
            else []
        ),
    )
    if runner._bookkeeping_sync is capture._original_bookkeeping:
        capture.install()
    runner.native_step = (scheduler, sampler, routes)
    result = runner.sample_tokens(None)
    assert result.serving_result == "original-serving-output"
    # Only the raw router scratch is reusable. The sampled IDs/logprobs and
    # native route snapshot are separate immutable output allocations.
    routes.fill_(255)
    return result


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


def _prepare_empty_batch(runner: SimpleNamespace, capture: GpuOutputCapture) -> None:
    runner.input_batch.req_ids = []
    capture.prepare_step(
        SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        ),
        SimpleNamespace(sampled_token_ids=None),
    )


def _batched_step(
    runner: SimpleNamespace, capture: GpuOutputCapture
) -> SimpleNamespace:
    """Uneven requests share native allocations with an untagged serving row."""
    runner.requests = {
        request_id: SimpleNamespace(
            num_prompt_tokens=prompt_count,
            sampling_params=SimpleNamespace(
                n=1, extra_args={GPU_CAPTURE_KEY: key} if key else {}
            ),
        )
        for request_id, key, prompt_count in (
            ("native-a", "a", 3),
            ("native-b", "b", 1),
            ("untagged", None, 2),
        )
    }
    runner.input_batch.req_ids = ["native-a", "native-b", "untagged"]
    runner.input_batch.num_computed_tokens_cpu = np.array([0, 0, 0])
    runner.discard_request_mask.np = np.array([False, False, False])
    scratch = torch.arange(12, device="cuda").to(torch.uint16).reshape(6, 1, 2)
    sampler = SimpleNamespace(
        sampled_token_ids=torch.tensor(
            [[7], [8], [9]], dtype=torch.int32, device="cuda"
        ),
        logprobs_tensors=SimpleNamespace(
            logprobs=torch.tensor([[-0.7, -9], [-0.8, -8], [-0.9, -7]], device="cuda")
        ),
    )
    # Native input-batch order, rather than dict insertion order, owns row offsets.
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"untagged": 2, "native-b": 1, "native-a": 3},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
    )
    runner.native_step = (scheduler, sampler, scratch)
    if runner._bookkeeping_sync is capture._original_bookkeeping:
        capture.install()
    output = runner.sample_tokens(None)
    assert output.serving_result == "original-serving-output"
    scratch.fill_(255)
    return output


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@cuda_required
def test_native_payload_survives_scratch_reuse_and_cross_process_ipc() -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    capture.install()
    capture_stream = torch.cuda.Stream()
    export_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        _step(
            runner,
            capture,
            count=2,
            token=0,
            logprob=0,
            route_values=[10, 11, 12, 13],
            discard=True,
        )
        _step(runner, capture, start=2, count=1, route_values=[14, 15])
        # An extra async-scheduled step must be trimmed at the final stop.
        _step(
            runner,
            capture,
            start=3,
            count=1,
            token=8,
            logprob=-0.8,
            route_values=[16, 17],
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
    capture.clear()
    assert lease.lease_id in capture._leases
    capture.release(lease.lease_id)
    assert not capture._requests and not capture._leases
    # A queued step after finalization must not reallocate discarded output.
    _step(
        runner, capture, start=4, count=1, token=9, logprob=-0.9, route_values=[18, 19]
    )
    assert not capture._requests and not capture._leases
    capture.clear()


@cuda_required
def test_missing_routes_are_rejected_instead_of_returning_incomplete_payload() -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(runner, capture, start=2, count=1, route_values=[1, 2])
    with pytest.raises(RuntimeError, match="does not cover"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)
    capture.discard("call")
    assert not capture._requests and not capture._leases


@cuda_required
def test_import_rejects_wrong_host_and_physical_gpu_before_opening_ipc() -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=False)
    _step(runner, capture)
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
    capture = GpuOutputCapture(runner, require_routed_experts=False)
    _step(runner, capture)
    assert not capture._requests and not capture._leases
    with pytest.raises(RuntimeError, match="No retained GPU output"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)


@cuda_required
def test_reprefill_preserves_already_emitted_routes_and_logprobs() -> None:
    runner = _runner()
    # A per-request -1 becomes vocab_size at the native batch boundary.
    runner.requests["native"].sampling_params.logprobs = -1
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(runner, capture)
    # An overlapping prefill fragment must preserve routes already emitted.
    _step(
        runner,
        capture,
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
    assert not capture._requests and not capture._leases


@cuda_required
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_import_never_decrements_the_failed_attempt_twice(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    from nemo_rl.models.generation.vllm import gpu_output_capture as module

    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(runner, capture)
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
    assert not capture._requests and not capture._leases


@cuda_required
@pytest.mark.parametrize("raw_logprob", [float("-inf"), -10000.0, -9999.0, -9998.5])
def test_gpu_logprobs_match_serving_floor_before_staging(raw_logprob: float) -> None:
    from nemo_rl.models.generation.vllm.utils import VLLM_LOGPROB_FLOOR

    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=False)
    output = _step(runner, capture, logprob=raw_logprob)
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    normalized = capture._leases[lease.lease_id].tensors.generation_logprobs
    assert normalized.is_cuda
    assert normalized.item() == max(raw_logprob, VLLM_LOGPROB_FLOOR)
    assert output._logprobs_tensors.logprobs[0, 0].item() == raw_logprob
    del normalized
    capture.abandon_unimported(lease.lease_id)
    assert not capture._requests and not capture._leases


@cuda_required
def test_capture_reuses_native_storage_across_uneven_request_lifetimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    native_clone = torch.Tensor.clone
    cloned = []

    def track_clone(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        cloned.append(tensor)
        return native_clone(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", track_clone)
        output = _batched_step(runner, capture)
    # Exactly the routing snapshot clone native vLLM already performs.
    assert len(cloned) == 1
    assert cloned[0] is runner.native_step[2]
    snapshot = output._routed_experts.routing_data
    sampled = output._sampled_token_ids
    logs = output._logprobs_tensors.logprobs
    for key in ("a", "b"):
        fragment = capture._requests[key].fragments[0]
        assert (
            fragment.routes.untyped_storage()._cdata
            == snapshot.untyped_storage()._cdata
        )
        assert (
            fragment.token_id.untyped_storage()._cdata
            == sampled.untyped_storage()._cdata
        )
        assert (
            fragment.logprob.untyped_storage()._cdata == logs.untyped_storage()._cdata
        )
    assert set(capture._requests) == {"a", "b"}

    # Serving drops its references after D2H; our views keep the snapshot alive.
    del output._routed_experts, output._sampled_token_ids, output._logprobs_tensors
    del runner.native_step
    del snapshot, sampled, logs, fragment
    lease_a = capture.export("a", generated_token_count=1, prompt_token_count=3)
    torch.cuda.synchronize()
    # Request B's one-row views still pin the complete old batch allocations.
    assert set(capture._requests) == {"b"}
    assert capture._requests["b"].fragments[0].routes.tolist() == [[[6, 7]]]
    capture.abandon_unimported(lease_a.lease_id)
    lease_b = capture.export("b", generated_token_count=1, prompt_token_count=1)
    torch.cuda.synchronize()
    tensors = capture._leases[lease_b.lease_id].tensors
    assert tensors.generated_token_ids.tolist() == [8]
    assert tensors.routed_experts.tolist() == [[[6, 7]], [[0, 1]]]
    del tensors
    capture.abandon_unimported(lease_b.lease_id)
    assert not capture._requests and not capture._leases


@cuda_required
def test_request_mapping_is_frozen_before_native_bookkeeping_mutation() -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)

    def mutate_live_batch() -> None:
        runner.input_batch.req_ids.reverse()
        runner.input_batch.num_computed_tokens_cpu.fill(999)
        runner.discard_request_mask.np.fill(True)
        for request in runner.requests.values():
            request.num_prompt_tokens = 999
            request.sampling_params.extra_args = {GPU_CAPTURE_KEY: "changed"}

    runner.after_bookkeeping = mutate_live_batch
    _batched_step(runner, capture)
    lease = capture.export("a", generated_token_count=1, prompt_token_count=3)
    tensors = capture._leases[lease.lease_id].tensors
    assert tensors.generated_token_ids.tolist() == [7]
    assert tensors.generation_logprobs.tolist() == pytest.approx([-0.7])
    assert tensors.routed_experts.tolist() == [[[0, 1]], [[2, 3]], [[4, 5]], [[0, 1]]]
    assert set(capture._requests) == {"b"}
    del tensors
    capture.abandon_unimported(lease.lease_id)
    capture.discard("b")
    torch.cuda.synchronize()
    assert not capture._requests and not capture._leases


@cuda_required
def test_discard_between_bookkeeping_and_snapshot_does_not_resurrect_request() -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    runner.after_bookkeeping = lambda: capture.discard("call")
    _step(runner, capture)
    assert not capture._requests and not capture._leases


@cuda_required
def test_native_sync_scheduler_keeps_original_sampled_outputs() -> None:
    runner = _runner(async_scheduling=False)
    capture = GpuOutputCapture(runner, require_routed_experts=False)
    output = _step(runner, capture)
    assert output._routed_experts is None
    fragment = capture._requests["call"].fragments[0]
    assert fragment.token_id.data_ptr() == output._sampled_token_ids.data_ptr()
    assert fragment.logprob.data_ptr() == output._logprobs_tensors.logprobs.data_ptr()
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    assert lease.routed_experts is None
    tensors = capture._leases[lease.lease_id].tensors
    assert tensors.generated_token_ids.tolist() == [7]
    del tensors
    capture.abandon_unimported(lease.lease_id)
    torch.cuda.synchronize()
    assert not capture._requests and not capture._leases


@cuda_required
def test_raw_vocabulary_logprob_selection_is_deferred_to_assembly() -> None:
    runner = _runner()
    runner.input_batch.sampling_metadata = SimpleNamespace(max_num_logprobs=-1)
    capture = GpuOutputCapture(runner, require_routed_experts=False)
    output = _step(runner, capture, token=1)
    fragment = capture._requests["call"].fragments[0]
    assert fragment.logprob.shape == (2,)
    assert fragment.logprob.data_ptr() == output._logprobs_tensors.logprobs.data_ptr()
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    assert capture._leases[lease.lease_id].tensors.generation_logprobs.item() == -9
    capture.abandon_unimported(lease.lease_id)


@cuda_required
def test_prefix_cached_by_untagged_request_needs_only_historical_route_backfill() -> (
    None
):
    runner = _runner()
    runner.requests["native"].sampling_params.extra_args = {}
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    historical = _step(
        runner, capture, route_values=[10, 11, 12, 13, 14, 15], initial_cached_prefix=0
    )
    assert not capture._requests and not capture._leases
    del historical, runner.native_step
    # A subsequent request hits two cached prompt positions. Their routing
    # exists only in the native CPU cache, with no previous tagged capture.
    runner.requests["native"] = SimpleNamespace(
        sampling_params=SimpleNamespace(n=1, extra_args={GPU_CAPTURE_KEY: "call"}),
        num_prompt_tokens=3,
    )
    output = _step(
        runner,
        capture,
        start=2,
        count=1,
        token=8,
        logprob=-0.8,
        route_values=[20, 21],
        initial_cached_prefix=2,
    )
    fragment = capture._requests["call"].fragments[0]
    assert fragment.routes.data_ptr() == output._routed_experts.routing_data.data_ptr()
    assert fragment.token_id.data_ptr() == output._sampled_token_ids.data_ptr()
    assert fragment.logprob.data_ptr() == output._logprobs_tensors.logprobs.data_ptr()
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    assert lease.routed_experts_prefix_backfill_ranges == ((0, 2),)
    tensors = capture._leases[lease.lease_id].tensors
    assert tensors.generated_token_ids.tolist() == [8]
    assert tensors.generation_logprobs.tolist() == pytest.approx([-0.8])
    assert tensors.routed_experts[2:].tolist() == [[[20, 21]], [[0, 1]]]
    del tensors
    capture.abandon_unimported(lease.lease_id)
    torch.cuda.synchronize()
    assert not capture._requests and not capture._leases


@cuda_required
def test_cached_prefix_length_is_frozen_before_bookkeeping_and_chunked_prefill() -> (
    None
):
    runner = _runner()
    runner.requests["native"].num_prompt_tokens = 5
    capture = GpuOutputCapture(runner, require_routed_experts=True)

    def mutate_admission_metadata() -> None:
        runner.native_step[0].scheduled_new_reqs[0].num_computed_tokens = 4
        runner.input_batch.num_computed_tokens_cpu[0] = 4

    runner.after_bookkeeping = mutate_admission_metadata
    _step(
        runner,
        capture,
        start=2,
        count=1,
        token=0,
        logprob=0,
        route_values=[20, 21],
        discard=True,
        initial_cached_prefix=2,
    )
    del runner.after_bookkeeping
    _step(runner, capture, start=3, count=2, route_values=[30, 31, 40, 41])
    assert capture._requests["call"].proven_cached_prefix_tokens == 2
    lease = capture.export("call", generated_token_count=1, prompt_token_count=5)
    assert lease.routed_experts_prefix_backfill_ranges == ((0, 2),)
    tensors = capture._leases[lease.lease_id].tensors
    assert tensors.routed_experts[2:].tolist() == [
        [[20, 21]],
        [[30, 31]],
        [[40, 41]],
        [[0, 1]],
    ]
    del tensors
    capture.abandon_unimported(lease.lease_id)


@cuda_required
@pytest.mark.parametrize(
    "gap_kind", ["uncached_prompt", "generated_route", "generated_id"]
)
def test_prefix_authorization_never_covers_missing_fresh_outputs(gap_kind: str) -> None:
    runner = _runner()
    runner.requests["native"].num_prompt_tokens = 5
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(
        runner,
        capture,
        start=2,
        count=1 if gap_kind == "uncached_prompt" else 3,
        route_values=[20, 21]
        if gap_kind == "uncached_prompt"
        else [20, 21, 30, 31, 40, 41],
        discard=gap_kind == "uncached_prompt",
        initial_cached_prefix=2,
    )
    if gap_kind == "uncached_prompt":
        # Position 3 was not in the cache and its scheduled chunk is missing.
        _step(runner, capture, start=4, count=1, route_values=[40, 41])
        generated = 1
    elif gap_kind == "generated_route":
        # Token IDs can be present while a post-prompt route fragment is lost.
        _step(
            runner,
            capture,
            start=5,
            count=1,
            token=8,
            logprob=-0.8,
            route_values=[50, 51],
        )
        capture._requests["call"].fragments[-1].routes = None
        generated = 2
    else:
        generated = 2
    with pytest.raises(RuntimeError, match="does not cover"):
        capture.export("call", generated_token_count=generated, prompt_token_count=5)
    capture.discard("call")
    torch.cuda.synchronize()
    assert not capture._requests and not capture._leases


@cuda_required
@pytest.mark.parametrize("cached_prefix", [-1, 4, 1])
def test_invalid_admission_cache_metadata_fails_capture_only(
    cached_prefix: int,
) -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(
        runner,
        capture,
        start=2,
        count=1,
        route_values=[20, 21],
        initial_cached_prefix=cached_prefix,
    )
    with pytest.raises(RuntimeError, match="cached-prefix metadata"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)
    assert not capture._requests and not capture._leases


@cuda_required
def test_later_positive_start_does_not_expand_proven_cached_prefix() -> None:
    runner = _runner()
    runner.requests["native"].num_prompt_tokens = 5
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(
        runner,
        capture,
        start=1,
        count=1,
        token=0,
        logprob=0,
        route_values=[10, 11],
        discard=True,
        initial_cached_prefix=1,
    )
    # A later chunk has no NewRequestData; its start cannot authorize missing
    # freshly computed positions 2 and 3 as cache holes.
    _step(runner, capture, start=4, count=1, route_values=[40, 41])
    assert capture._requests["call"].proven_cached_prefix_tokens == 1
    with pytest.raises(RuntimeError, match="outside the proven cached prefix"):
        capture.export("call", generated_token_count=1, prompt_token_count=5)
    capture.discard("call")


@cuda_required
def test_native_capture_rejects_truncated_canonical_cpu_route_envelope() -> None:
    runner = _runner()
    runner.requests["native"].sampling_params.routed_experts_prompt_start = 2
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(
        runner,
        capture,
        start=2,
        count=1,
        route_values=[20, 21],
        initial_cached_prefix=2,
    )
    with pytest.raises(RuntimeError, match="full canonical prompt routes"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)


@cuda_required
@pytest.mark.parametrize("release_mode", ["explicit", "lost_reply", "consumed"])
def test_abort_drops_native_views_and_unimported_lease(
    monkeypatch: pytest.MonkeyPatch, release_mode: str
) -> None:
    from nemo_rl.models.generation.vllm import gpu_output_capture as module

    released = []
    release_counter = module._release_unopened_handle

    def tracked_release(handle: CudaTensorIpc) -> None:
        released.append(handle)
        release_counter(handle)

    monkeypatch.setattr(module, "_release_unopened_handle", tracked_release)
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(runner, capture)
    views = [
        weakref.ref(tensor)
        for tensor in (
            capture._requests["call"].fragments[0].routes,
            capture._requests["call"].fragments[0].token_id,
            capture._requests["call"].fragments[0].logprob,
        )
    ]
    assert all(ref() is not None for ref in views)
    lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    assert all(ref() is None for ref in views)
    assembled = weakref.ref(capture._leases[lease.lease_id].tensors.generated_token_ids)
    if release_mode == "explicit":
        capture.abandon_unimported(lease.lease_id)
    elif release_mode == "consumed":
        # Model PyTorch's consumed counters, then the frontend release ACK.
        for handle in (
            lease.generated_token_ids,
            lease.generation_logprobs,
            lease.routed_experts,
        ):
            release_counter(handle)
        capture.release(lease.lease_id)
    # With a lost export reply, discard must release the unopened handles.
    capture.discard("call")
    capture.discard("call")
    assert len(released) == (0 if release_mode == "consumed" else 3)
    assert assembled() is None
    assert not capture._requests and not capture._leases


@cuda_required
def test_changed_bookkeeping_call_preserves_serving() -> None:
    runner = _runner()
    runner._bookkeeping_sync = lambda payload: payload
    capture = GpuOutputCapture(runner, require_routed_experts=False)
    capture.install()
    assert (
        runner._bookkeeping_sync("original-serving-output") == "original-serving-output"
    )
    assert not capture._requests
    with pytest.raises(RuntimeError, match="capture failed"):
        capture.export("call", generated_token_count=1, prompt_token_count=3)


@cuda_required
@pytest.mark.parametrize("output_count", [0, 1])
def test_resumed_requests_discard_gpu_history_for_original_cpu_put(
    output_count: int,
) -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(runner, capture, route_values=[10, 11, 20, 21, 30, 31])
    old_view = weakref.ref(capture._requests["call"].fragments[0].routes)
    result = _step(
        runner,
        capture,
        start=1,
        token=8,
        logprob=-0.8,
        route_values=[40, 41, 50, 51, 60, 61],
        resumed_cached_prefix=1,
        resumed_output_tokens=output_count,
    )
    assert result.serving_result == "original-serving-output"
    assert old_view() is None
    assert not capture._requests and not capture._leases
    with pytest.raises(RuntimeError, match="capture failed.*preempted GPU history"):
        capture.export(
            "call", generated_token_count=output_count + 1, prompt_token_count=3
        )
    _step(
        runner, capture, start=4, count=1, token=9, logprob=-0.9, route_values=[70, 71]
    )
    assert not capture._requests and not capture._leases


@cuda_required
@pytest.mark.parametrize(
    "path",
    [
        "supported",
        "sync_routes",
        "speculative",
        "pipeline",
        "context",
        "missing_hook",
        "remote",
        "non_owner",
    ],
)
def test_native_optimization_availability_preserves_existing_paths(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    runner = _runner(async_scheduling=path != "sync_routes")
    original = runner.sample_tokens
    runner.vllm_config = SimpleNamespace(
        speculative_config=object() if path == "speculative" else None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2 if path == "pipeline" else 1,
            decode_context_parallel_size=2 if path == "context" else 1,
        ),
    )
    if path == "missing_hook":
        del runner._bookkeeping_sync
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: 1 if path == "non_owner" else 0,
        ),
    )
    worker = SimpleNamespace(model_runner=runner)
    capability = configure_gpu_output_capture(
        worker,
        frontend_hostname="other-host" if path == "remote" else socket.gethostname(),
        require_routed_experts=True,
    )
    assert (capability is not None) == (path == "supported")
    assert (runner.sample_tokens is not original) == (path == "supported")


@cuda_required
def test_export_reads_survive_source_release_on_another_stream() -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    source_stream, export_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(source_stream):
        _step(runner, capture, route_values=[10, 11, 20, 21, 30, 31])
    del runner.native_step
    with torch.cuda.stream(export_stream):
        # Keep the reads pending while the source stream allocates replacements.
        torch.cuda._sleep(50_000_000)
        lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    assert not capture._requests
    with torch.cuda.stream(source_stream):
        replacements = [
            torch.full((3, 1, 2), 255, dtype=torch.uint16, device="cuda")
            for _ in range(64)
        ]
    tensors = capture._leases[lease.lease_id].tensors
    export_stream.synchronize()
    assert tensors.generated_token_ids.tolist() == [7]
    assert tensors.generation_logprobs.tolist() == pytest.approx([-0.7])
    assert tensors.routed_experts.tolist() == [
        [[10, 11]],
        [[20, 21]],
        [[30, 31]],
        [[0, 1]],
    ]
    del tensors, replacements
    capture.abandon_unimported(lease.lease_id)


@cuda_required
@pytest.mark.parametrize("completion", ["export", "failure", "resume", "abort"])
def test_finished_native_requests_retire_tracking_after_cleanup(
    completion: str,
) -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    _step(runner, capture)
    if completion == "export":
        lease = capture.export("call", generated_token_count=1, prompt_token_count=3)
    elif completion == "failure":
        capture.fail_keys((("call", "native"),), RuntimeError("capture unavailable"))
    elif completion == "resume":
        _step(
            runner,
            capture,
            start=1,
            count=2,
            route_values=[40, 41, 50, 51],
            resumed_cached_prefix=1,
        )
    else:
        capture.discard("call")

    # Preemption removes the input row, but keeps runner.requests for resumption.
    _prepare_empty_batch(runner, capture)
    assert capture._closed_keys == {"call": "native"}
    del runner.requests["native"]
    _prepare_empty_batch(runner, capture)
    if completion == "export":
        assert "call" in capture._closed_keys
        capture.abandon_unimported(lease.lease_id)
    elif completion in ("failure", "resume"):
        assert "call" in capture._errors
        with pytest.raises(RuntimeError, match="capture failed"):
            capture.export("call", generated_token_count=1, prompt_token_count=3)
        capture.discard("call")
    _prepare_empty_batch(runner, capture)
    assert not capture._closed_keys and not capture._errors


@cuda_required
def test_early_abort_suppresses_later_admission_before_tracking_retires() -> None:
    runner = _runner()
    capture = GpuOutputCapture(runner, require_routed_experts=True)
    request = runner.requests.pop("native")
    capture.discard("call")
    _prepare_empty_batch(runner, capture)
    assert capture._closed_keys == {"call": None}

    runner.requests["native"] = request
    runner.input_batch.req_ids = ["native"]
    _step(runner, capture)
    assert capture._closed_keys == {"call": "native"}
    assert not capture._requests and not capture._leases
    del runner.requests["native"]
    _prepare_empty_batch(runner, capture)
    assert not capture._closed_keys
