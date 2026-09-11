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
"""Retain native vLLM CUDA outputs until the existing frontend staging PUT.

This hook leaves vLLM's serving outputs untouched. Only explicitly tagged
requests retain a second, device-resident view of their training payload.
The worker owns immutable clones until the frontend finishes PUT and releases
the IPC lease. No tensor payload is serialized through Ray or engine IPC.
"""

from __future__ import annotations

import inspect
import socket
import threading
import types
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, cast

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from nemo_rl.models.generation.vllm.utils import VLLM_LOGPROB_FLOOR

GPU_CAPTURE_KEY = "nrl_gpu_output_capture_key"


@dataclass(frozen=True)
class GpuOutputCaptureCapabilities:
    owner: bool
    hostname: str
    gpu_uuid: str
    device_index: int


@dataclass(frozen=True)
class CudaTensorIpc:
    """CUDA IPC metadata only; device ordinals are resolved by physical UUID."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    tensor_offset: int
    dtype: str
    storage_handle: bytes | None
    storage_size_bytes: int
    storage_offset_bytes: int
    ref_counter_handle: bytes | None
    ref_counter_offset: int
    event_handle: bytes | None
    event_sync_required: bool


@dataclass(frozen=True)
class GpuOutputLease:
    lease_id: str
    capture_key: str
    hostname: str
    gpu_uuid: str
    ready_event_handle: bytes
    generated_token_ids: CudaTensorIpc
    generation_logprobs: CudaTensorIpc
    routed_experts: CudaTensorIpc | None


@dataclass(frozen=True)
class GpuOutputTensors:
    generated_token_ids: torch.Tensor
    generation_logprobs: torch.Tensor
    routed_experts: torch.Tensor | None


class GpuOutputImportError(RuntimeError):
    """Whether an unsuccessful import makes whole-lease abandonment unsafe."""

    def __init__(self, message: str, *, handles_consumed: bool) -> None:
        super().__init__(message)
        self.handles_consumed = handles_consumed


def _lease_handles(lease: GpuOutputLease) -> list[CudaTensorIpc]:
    handles = [lease.generated_token_ids, lease.generation_logprobs]
    if lease.routed_experts is not None:
        handles.append(lease.routed_experts)
    return handles


def _release_unopened_handle(handle: CudaTensorIpc) -> None:
    if handle.ref_counter_handle is not None and handle.storage_size_bytes:
        torch.UntypedStorage._release_ipc_counter_cuda(
            handle.ref_counter_handle, handle.ref_counter_offset
        )


def gpu_device_uuid(device: torch.device | int) -> str:
    identity = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if identity is None:
        raise RuntimeError("GPU output capture requires CUDA physical-device UUIDs")
    return str(identity)


def _export_tensor(tensor: torch.Tensor) -> CudaTensorIpc:
    if not tensor.is_cuda:
        raise RuntimeError("GPU output capture cannot export a CPU tensor")
    _, args = reduce_tensor(tensor.detach())
    if len(args) != 15:
        raise RuntimeError("Unsupported PyTorch CUDA IPC reduction signature")
    return CudaTensorIpc(
        tuple(args[1]),
        tuple(args[2]),
        int(args[3]),
        str(args[5]).removeprefix("torch."),
        args[7],
        args[8],
        args[9],
        args[11],
        args[12],
        args[13],
        args[14],
    )


def _import_tensor(handle: CudaTensorIpc, device_index: int) -> torch.Tensor:
    return rebuild_cuda_tensor(
        torch.Tensor,
        handle.shape,
        handle.stride,
        handle.tensor_offset,
        torch.storage.TypedStorage,
        getattr(torch, handle.dtype),
        device_index,
        handle.storage_handle,
        handle.storage_size_bytes,
        handle.storage_offset_bytes,
        False,
        handle.ref_counter_handle,
        handle.ref_counter_offset,
        handle.event_handle,
        handle.event_sync_required,
    )


def import_gpu_output_lease(
    lease: GpuOutputLease, device: torch.device | int
) -> GpuOutputTensors:
    """Import once, then keep these tensors alive through the blocking PUT.

    The caller must release the remote lease in a finally block after PUT and
    after disposing of imported tensors. Never forward imported CUDA storage
    to another process.
    """
    if socket.gethostname() != lease.hostname:
        raise GpuOutputImportError(
            "GPU output CUDA IPC requires a same-host producer", handles_consumed=False
        )
    device_index = device if isinstance(device, int) else device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if gpu_device_uuid(device_index) != lease.gpu_uuid:
        raise GpuOutputImportError(
            "GPU output CUDA IPC physical-device UUID mismatch", handles_consumed=False
        )
    with torch.cuda.device(device_index):
        try:
            ready = cast(
                torch.cuda.Event,
                torch.cuda.Event.from_ipc_handle(
                    device_index, lease.ready_event_handle
                ),
            )
            torch.cuda.current_stream().wait_event(ready)
        except Exception as error:
            raise GpuOutputImportError(str(error), handles_consumed=False) from error
        imported: list[torch.Tensor] = []
        handles = _lease_handles(lease)
        for index, handle in enumerate(handles):
            try:
                imported.append(_import_tensor(handle, device_index))
            except Exception as error:
                try:
                    imported.clear()
                    torch.cuda.synchronize(device_index)
                    # The attempted handle may already have consumed its counter
                    # inside PyTorch. Only decrement handles never attempted.
                    for unopened in handles[index + 1 :]:
                        _release_unopened_handle(unopened)
                except Exception as cleanup_error:
                    # Even failed cleanup cannot make whole-lease abandonment
                    # safe again after an import attempt consumed counters.
                    raise GpuOutputImportError(
                        f"{error}; IPC cleanup failed: {cleanup_error}",
                        handles_consumed=True,
                    ) from error
                raise GpuOutputImportError(str(error), handles_consumed=True) from error
        return GpuOutputTensors(
            imported[0], imported[1], imported[2] if len(imported) == 3 else None
        )


@dataclass
class _Fragment:
    start: int
    routes: torch.Tensor | None
    generated_position: int | None
    token_id: torch.Tensor | None
    logprob: torch.Tensor | None
    ready: Any

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.routes, self.token_id, self.logprob)
            if t is not None
        )


@dataclass
class _RequestCapture:
    native_request_id: str
    prompt_token_count: int
    fragments: list[_Fragment] = field(default_factory=list)
    committed_route_end: int = 0


@dataclass
class _OwnedLease:
    descriptor: GpuOutputLease
    tensors: GpuOutputTensors
    ready: Any
    nbytes: int


class GpuOutputCapture:
    """Budgeted device storage owned by exactly one TP rank per engine."""

    def __init__(
        self, runner: Any, *, max_retained_bytes: int, require_routed_experts: bool
    ) -> None:
        if max_retained_bytes <= 0:
            raise ValueError("GPU output capture budget must be positive")
        self.runner = runner
        self.max_retained_bytes = max_retained_bytes
        self.require_routed_experts = require_routed_experts
        device = torch.device(runner.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("GPU capture requires an explicit CUDA device ordinal")
        self._device_index = device.index
        self.hostname = socket.gethostname()
        self.gpu_uuid = gpu_device_uuid(self.device)
        self._requests: dict[str, _RequestCapture] = {}
        self._leases: dict[str, _OwnedLease] = {}
        # Late steps may already be queued when the frontend observes a stop.
        # Closed keys suppress those steps until the quiescent batch cleanup.
        self._closed_keys: set[str] = set()
        self._errors: dict[str, str] = {}
        self._retained_bytes = 0
        self._lock = threading.RLock()
        self._original_bookkeeping = runner._bookkeeping_sync

    @property
    def device(self) -> torch.device:
        """The native worker's CUDA device."""
        return torch.device("cuda", self._device_index)

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._retained_bytes

    def _check_budget(self, additional: int) -> None:
        if self._retained_bytes + additional > self.max_retained_bytes:
            raise RuntimeError(
                "GPU output capture retention budget exceeded: "
                f"retained={self._retained_bytes}, additional={additional}, "
                f"limit={self.max_retained_bytes}; reduce rollout concurrency "
                "or increase the configured budget"
            )

    def install(self) -> None:
        expected = (
            "scheduler_output",
            "sampler_output",
            "logits",
            "hidden_states",
            "num_scheduled_tokens",
        )
        if tuple(inspect.signature(self._original_bookkeeping).parameters) != expected:
            raise RuntimeError("Unsupported vLLM _bookkeeping_sync signature")
        capture = self
        original = self._original_bookkeeping

        @wraps(original)
        def bookkeeping(
            _runner: Any,
            scheduler_output: Any,
            sampler_output: Any,
            logits: torch.Tensor | None,
            hidden_states: torch.Tensor,
            num_scheduled_tokens: int,
        ) -> Any:
            try:
                capture.capture_step(scheduler_output, sampler_output)
            except Exception as error:
                # Capture failure poisons staging for this batch's tagged
                # calls; it must never fail ordinary vLLM model serving.
                capture.fail_step(error)
            return original(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                num_scheduled_tokens,
            )

        self.runner._bookkeeping_sync = types.MethodType(bookkeeping, self.runner)

    def fail_step(self, error: Exception) -> None:
        """Preserve a capture error until the frontend requests its payload."""
        with self._lock:
            for request_id in self.runner.input_batch.req_ids:
                request = self.runner.requests.get(request_id)
                params = getattr(request, "sampling_params", None)
                key = (getattr(params, "extra_args", None) or {}).get(GPU_CAPTURE_KEY)
                if isinstance(key, str) and key and key not in self._closed_keys:
                    self.discard(key)
                    self._errors[key] = f"{type(error).__name__}: {error}"

    def capture_step(self, scheduler_output: Any, sampler_output: Any) -> None:
        """Clone native tensors before vLLM reuses output or routing buffers."""
        runner = self.runner
        batch = runner.input_batch
        sampled = sampler_output.sampled_token_ids
        if sampled.ndim != 2 or sampled.shape[1] != 1:
            raise RuntimeError("GPU output capture requires non-speculative sampling")
        routes = None
        if self.require_routed_experts:
            if not runner.routed_experts_initialized:
                raise RuntimeError("GPU routed-expert capture is not initialized")
            routes = runner.routed_experts_capturer.get_device_buffer()
        offset = 0
        with self._lock:
            for index, request_id in enumerate(batch.req_ids):
                count = int(scheduler_output.num_scheduled_tokens[request_id])
                request = runner.requests[request_id]
                params = request.sampling_params
                key = (getattr(params, "extra_args", None) or {}).get(GPU_CAPTURE_KEY)
                row_start = offset
                offset += count
                if key is None:
                    continue
                if not isinstance(key, str) or not key:
                    raise ValueError("GPU output capture key must be a nonempty string")
                if key in self._closed_keys:
                    continue
                if params.n != 1:
                    raise RuntimeError("GPU output capture supports n=1 only")
                prompt_count = int(request.num_prompt_tokens)
                state = self._requests.get(key)
                if state is not None and state.native_request_id != request_id:
                    raise RuntimeError(
                        "GPU output capture key reused by another request"
                    )
                start = int(batch.num_computed_tokens_cpu[index])
                valid = not bool(runner.discard_request_mask.np[index])
                # vLLM emits prompt routes once, then appends new positions.
                # Re-prefill cannot replace routes the frontend already saw.
                committed_end = state.committed_route_end if state else 0
                route_skip = min(max(committed_end - start, 0), count)
                route_view = (
                    routes[row_start + route_skip : offset]
                    if routes is not None
                    else None
                )
                token_view = sampled[index, :1] if valid else None
                logprob_view = None
                generated_position = None
                if valid:
                    logprobs = sampler_output.logprobs_tensors
                    if logprobs is None or logprobs.logprobs.ndim != 2:
                        raise RuntimeError(
                            "GPU output capture requires sampled logprobs"
                        )
                    metadata = getattr(batch, "sampling_metadata", None)
                    if getattr(metadata, "max_num_logprobs", None) == -1:
                        logprob_view = logprobs.logprobs[index].gather(
                            0, sampled[index, :1].to(torch.int64)
                        )
                    else:
                        # Both standard and specific-token sampling put the
                        # sampled logprob in column zero. GPUInputBatch maps
                        # a requested -1 to vocab_size before calling sampler.
                        logprob_view = logprobs.logprobs[index, :1]
                    generated_position = start + count - prompt_count
                    if generated_position < 0:
                        raise RuntimeError("vLLM sampled before completing the prompt")
                views = (route_view, token_view, logprob_view)
                if any(t is not None and not t.is_cuda for t in views):
                    raise RuntimeError("GPU output capture received CPU payloads")
                required = sum(
                    t.numel() * t.element_size() for t in views if t is not None
                )
                self._check_budget(required)
                cloned = [t.detach().clone() if t is not None else None for t in views]
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
                fragment = _Fragment(
                    start + route_skip,
                    cloned[0],
                    generated_position,
                    cloned[1],
                    cloned[2],
                    ready,
                )
                if state is None:
                    state = self._requests[key] = _RequestCapture(
                        request_id, prompt_count
                    )
                state.fragments.append(fragment)
                if valid:
                    state.committed_route_end = max(
                        state.committed_route_end, start + count
                    )
                self._retained_bytes += fragment.nbytes

    def export(
        self, capture_key: str, *, generated_token_count: int, prompt_token_count: int
    ) -> GpuOutputLease:
        if generated_token_count < 0 or prompt_token_count < 0:
            raise ValueError("GPU output lengths must be nonnegative")
        with self._lock, torch.cuda.device(self.device):
            if capture_key in self._errors:
                raise RuntimeError(
                    f"GPU output capture failed: {self._errors[capture_key]}"
                )
            state = self._requests.get(capture_key)
            if state is None:
                raise RuntimeError(
                    f"No retained GPU output for capture key {capture_key!r}"
                )
            if state.prompt_token_count != prompt_token_count:
                raise RuntimeError(
                    "GPU output prompt length differs from final response"
                )
            total = prompt_token_count + generated_token_count
            expected_routes = max(total - 1, 0)
            route_fragments = [f for f in state.fragments if f.routes is not None]
            route_shape = None
            if self.require_routed_experts:
                if not route_fragments:
                    raise RuntimeError("GPU output is missing routed experts")
                route_shape = tuple(route_fragments[0].routes.shape[1:])
                if len(route_shape) != 2:
                    raise RuntimeError("Expected [tokens, layers, topk] GPU routes")
            additional = generated_token_count * (8 + 4)
            if route_shape is not None:
                additional += (
                    total
                    * route_shape[0]
                    * route_shape[1]
                    * route_fragments[0].routes.element_size()
                )
            self._check_budget(additional)
            ids = torch.empty(
                generated_token_count, dtype=torch.int64, device=self.device
            )
            logprobs = torch.empty(
                generated_token_count, dtype=torch.float32, device=self.device
            )
            assembled_routes = None
            if route_shape is not None:
                source = route_fragments[0].routes
                assembled_routes = (
                    torch.arange(route_shape[1], device=self.device)
                    .to(source.dtype)
                    .view(1, 1, -1)
                    .expand(total, *route_shape)
                    .clone()
                )
            token_coverage = [False] * generated_token_count
            route_coverage = [False] * expected_routes
            for fragment in state.fragments:
                export_stream = torch.cuda.current_stream(self.device)
                export_stream.wait_event(fragment.ready)
                # Fragments were allocated on the capture stream. The request
                # drops its references below, before export-stream copies may
                # finish, so the caching allocator must track these reads.
                for source in (fragment.routes, fragment.token_id, fragment.logprob):
                    if source is not None:
                        source.record_stream(export_stream)
                pos = fragment.generated_position
                if pos is not None and pos < generated_token_count:
                    if fragment.token_id is None or fragment.logprob is None:
                        raise RuntimeError(
                            "Generated GPU fragment lacks sampled IDs/logprobs"
                        )
                    ids[pos : pos + 1].copy_(fragment.token_id)
                    logprobs[pos : pos + 1].copy_(fragment.logprob)
                    token_coverage[pos] = True
                if assembled_routes is not None and fragment.routes is not None:
                    end = min(
                        fragment.start + fragment.routes.shape[0], expected_routes
                    )
                    if end > fragment.start:
                        assembled_routes[fragment.start : end].copy_(
                            fragment.routes[: end - fragment.start]
                        )
                        route_coverage[fragment.start : end] = [True] * (
                            end - fragment.start
                        )
            if not all(token_coverage) or (
                route_shape is not None and not all(route_coverage)
            ):
                raise RuntimeError(
                    "GPU output does not cover the final accepted token/route positions"
                )
            # The OpenAI serving adapter applies max(raw, VLLM_LOGPROB_FLOOR).
            # Match that normalization on device to preserve committed bytes.
            logprobs.clamp_min_(VLLM_LOGPROB_FLOOR)
            tensors = GpuOutputTensors(ids, logprobs, assembled_routes)
            ready = torch.cuda.Event(interprocess=True)
            ready.record(torch.cuda.current_stream(self.device))
            lease_id = uuid.uuid4().hex
            handles: list[CudaTensorIpc] = []
            try:
                for tensor in (ids, logprobs, assembled_routes):
                    if tensor is not None:
                        handles.append(_export_tensor(tensor))
            except Exception:
                # None of these descriptors has left the producer yet.
                for handle in handles:
                    _release_unopened_handle(handle)
                raise
            descriptor = GpuOutputLease(
                lease_id,
                capture_key,
                self.hostname,
                self.gpu_uuid,
                ready.ipc_handle(),
                handles[0],
                handles[1],
                handles[2] if assembled_routes is not None else None,
            )
            self._leases[lease_id] = _OwnedLease(descriptor, tensors, ready, additional)
            self._retained_bytes += additional
            self._retained_bytes -= sum(f.nbytes for f in state.fragments)
            del self._requests[capture_key]
            self._closed_keys.add(capture_key)
            self._errors.pop(capture_key, None)
            return descriptor

    def release(self, lease_id: str) -> None:
        """Release only after the frontend has finished PUT and dropped imports."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is not None:
                self._retained_bytes -= lease.nbytes

    def abandon_unimported(self, lease_id: str) -> None:
        """Release descriptors the frontend guarantees it never attempted.

        Never call this if any receiver may still import these descriptors or
        has already consumed their IPC refcounters.
        """
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is not None:
                for handle in _lease_handles(lease.descriptor):
                    _release_unopened_handle(handle)
                self.release(lease_id)

    def discard(self, capture_key: str) -> None:
        """Discard an aborted request, including an export whose reply was lost.

        The frontend must first dispose of and synchronize any imported views.
        """
        with self._lock:
            self._closed_keys.add(capture_key)
            self._errors.pop(capture_key, None)
            state = self._requests.pop(capture_key, None)
            if state is not None:
                self._retained_bytes -= sum(f.nbytes for f in state.fragments)
            for lease_id, lease in list(self._leases.items()):
                if lease.descriptor.capture_key == capture_key:
                    self.release(lease_id)

    def clear(self) -> None:
        """Clear unexported captures at a quiescent rollout boundary."""
        with self._lock:
            if self._leases:
                raise RuntimeError(
                    "Cannot clear GPU output capture with outstanding IPC leases"
                )
            self._requests.clear()
            self._closed_keys.clear()
            self._errors.clear()
            self._retained_bytes = 0


def configure_gpu_output_capture(
    worker: Any,
    *,
    max_retained_bytes: int,
    frontend_hostname: str,
    require_routed_experts: bool,
) -> GpuOutputCaptureCapabilities:
    """Install the version-checked, per-instance native worker hook."""
    # vLLM is optional outside the native generation-worker environment.
    import vllm
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    if vllm.__version__ != "0.25.1":
        raise RuntimeError("GPU output capture currently supports vLLM 0.25.1 only")
    runner = worker.model_runner
    config = runner.vllm_config
    parallel = config.parallel_config
    if parallel.pipeline_parallel_size != 1:
        raise RuntimeError("GPU output capture requires pipeline_parallel_size=1")
    if config.speculative_config is not None:
        raise RuntimeError(
            "GPU output capture does not yet support speculative decoding"
        )
    if require_routed_experts and config.cache_config.enable_prefix_caching:
        raise RuntimeError("GPU routed-expert capture requires prefix caching disabled")
    if any(
        getattr(parallel, name, 1) != 1
        for name in ("decode_context_parallel_size", "prefill_context_parallel_size")
    ):
        raise RuntimeError("GPU output capture does not support context parallelism")
    owner = get_tensor_model_parallel_rank() == 0
    device = torch.device(runner.device)
    capability = GpuOutputCaptureCapabilities(
        owner,
        socket.gethostname(),
        gpu_device_uuid(device),
        device.index,
    )
    if not owner:
        return capability
    if capability.hostname != frontend_hostname:
        raise RuntimeError("GPU output capture owner and frontend must share a host")
    existing = getattr(worker, "_gpu_output_capture", None)
    if existing is not None:
        if (
            existing.max_retained_bytes != max_retained_bytes
            or existing.require_routed_experts != require_routed_experts
        ):
            raise RuntimeError("GPU output capture is already configured differently")
        return capability
    capture = GpuOutputCapture(
        runner,
        max_retained_bytes=max_retained_bytes,
        require_routed_experts=require_routed_experts,
    )
    capture.install()
    worker._gpu_output_capture = capture
    return capability
