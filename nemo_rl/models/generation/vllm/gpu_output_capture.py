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
requests retain views of the original sampled outputs and vLLM's existing
async routing snapshot. The worker keeps their backing storage alive through
final assembly and owns the assembled output until the frontend releases its
IPC lease. No tensor payload is serialized through Ray or engine IPC.
"""

from __future__ import annotations

import inspect
import socket
import threading
import types
import uuid
from collections.abc import Iterable, Sequence
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
    logprob_by_token: bool = False


@dataclass(frozen=True)
class _StepRequest:
    """Positions and references frozen before native bookkeeping mutates them."""

    key: str
    native_request_id: str
    prompt_token_count: int
    start: int
    count: int
    row_start: int
    generated_position: int | None
    token_id: torch.Tensor | None
    logprob: torch.Tensor | None
    logprob_by_token: bool


@dataclass
class _BackingStorage:
    storage: torch.UntypedStorage
    references: int = 0
    pending_reads: list[Any] = field(default_factory=list)


@dataclass
class _RequestCapture:
    native_request_id: str
    prompt_token_count: int
    fragments: list[_Fragment] = field(default_factory=list)
    committed_route_end: int = 0
    storage_keys: set[int] = field(default_factory=set)


@dataclass
class _OwnedLease:
    descriptor: GpuOutputLease
    tensors: GpuOutputTensors
    ready: Any
    storage_keys: set[int]


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
        self._original_sample_tokens = runner.sample_tokens
        self._pending_step: tuple[_StepRequest, ...] | None = None
        self._storages: dict[int, _BackingStorage] = {}
        self._retiring_storages: set[int] = set()
        self._pending_read_storages: set[int] = set()

    @property
    def device(self) -> torch.device:
        """The native worker's CUDA device."""
        return torch.device("cuda", self._device_index)

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            self._reap_storages()
            return self._retained_bytes

    def _reap_storages(self) -> None:
        """Retire storage only after the last request and queued export read."""
        for key in self._retiring_storages | self._pending_read_storages:
            owned = self._storages[key]
            owned.pending_reads = [
                event for event in owned.pending_reads if not event.query()
            ]
            if not owned.pending_reads:
                self._pending_read_storages.discard(key)
            if not owned.references and not owned.pending_reads:
                self._retained_bytes -= owned.storage.nbytes()
                del self._storages[key]
                self._retiring_storages.remove(key)

    def _mark_storage_reads(self, keys: set[int], ready: Any) -> None:
        for key in keys:
            self._storages[key].pending_reads.append(ready)
        self._pending_read_storages.update(keys)

    def _release_storage_keys(self, keys: set[int], ready: Any = None) -> None:
        if ready is not None:
            self._mark_storage_reads(keys, ready)
        for key in keys:
            owned = self._storages[key]
            owned.references -= 1
            if not owned.references:
                self._retiring_storages.add(key)
        self._reap_storages()

    def _release_request(self, state: _RequestCapture, ready: Any = None) -> None:
        self._release_storage_keys(state.storage_keys, ready)

    def _retain_tensors(
        self, keys: set[int], tensors: Sequence[torch.Tensor | None]
    ) -> None:
        storages = {
            tensor.untyped_storage()._cdata: tensor.untyped_storage()
            for tensor in tensors
            if tensor is not None and tensor.numel()
        }
        self._check_budget(
            sum(
                storage.nbytes()
                for key, storage in storages.items()
                if key not in self._storages
            )
        )
        for key, storage in storages.items():
            if key not in self._storages:
                self._storages[key] = _BackingStorage(storage)
                self._retained_bytes += storage.nbytes()
            if key not in keys:
                self._storages[key].references += 1
                self._retiring_storages.discard(key)
                keys.add(key)

    def _retain_fragment(self, state: _RequestCapture, fragment: _Fragment) -> None:
        self._retain_tensors(
            state.storage_keys, (fragment.routes, fragment.token_id, fragment.logprob)
        )
        state.fragments.append(fragment)

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
        if tuple(inspect.signature(self._original_sample_tokens).parameters) != (
            "grammar_output",
        ):
            raise RuntimeError("Unsupported vLLM sample_tokens signature")
        if self.require_routed_experts and not self.runner.use_async_scheduling:
            raise RuntimeError(
                "GPU routed-expert capture requires native vLLM async_scheduling=True "
                "to retain its existing GPU routing snapshot"
            )
        capture = self
        original = self._original_bookkeeping
        original_sample_tokens = self._original_sample_tokens

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
                capture._pending_step = capture.prepare_step(
                    scheduler_output, sampler_output
                )
            except Exception as error:
                # Capture must never fail ordinary vLLM model serving.
                capture.fail_step(error)
            return original(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                num_scheduled_tokens,
            )

        @wraps(original_sample_tokens)
        def sample_tokens(_runner: Any, grammar_output: Any) -> Any:
            capture._pending_step = None
            try:
                output = original_sample_tokens(grammar_output)
                pending = capture._pending_step
                if pending:
                    try:
                        capture.finish_step(pending, output)
                    except Exception as error:
                        capture.fail_keys((entry.key for entry in pending), error)
                # The native async object has not reached get_output yet; our
                # tensor views survive when get_output drops its GPU refs.
                return output
            finally:
                capture._pending_step = None

        self.runner._bookkeeping_sync = types.MethodType(bookkeeping, self.runner)
        self.runner.sample_tokens = types.MethodType(sample_tokens, self.runner)

    def fail_keys(self, keys: Iterable[Any], error: Exception) -> None:
        """Preserve a capture error until the frontend requests its payload."""
        with self._lock:
            for key in keys:
                if isinstance(key, str) and key and key not in self._closed_keys:
                    self.discard(key)
                    self._errors[key] = f"{type(error).__name__}: {error}"

    def fail_step(self, error: Exception) -> None:
        keys = []
        for request_id in self.runner.input_batch.req_ids:
            request = self.runner.requests.get(request_id)
            params = getattr(request, "sampling_params", None)
            keys.append(
                (getattr(params, "extra_args", None) or {}).get(GPU_CAPTURE_KEY)
            )
        self.fail_keys(keys, error)

    def prepare_step(
        self, scheduler_output: Any, sampler_output: Any
    ) -> tuple[_StepRequest, ...]:
        """Retain immutable sampler views and request positions, without copies."""
        runner = self.runner
        batch = runner.input_batch
        sampled = sampler_output.sampled_token_ids
        offset = 0
        entries = []
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
                if sampled.ndim != 2 or sampled.shape[1] != 1:
                    raise RuntimeError(
                        "GPU output capture requires non-speculative sampling"
                    )
                prompt_count = int(request.num_prompt_tokens)
                start = int(batch.num_computed_tokens_cpu[index])
                valid = not bool(runner.discard_request_mask.np[index])
                token_view = sampled[index, :1] if valid else None
                logprob_view = None
                generated_position = None
                by_token = False
                if valid:
                    logprobs = sampler_output.logprobs_tensors
                    if logprobs is None or logprobs.logprobs.ndim != 2:
                        raise RuntimeError(
                            "GPU output capture requires sampled logprobs"
                        )
                    metadata = getattr(batch, "sampling_metadata", None)
                    by_token = getattr(metadata, "max_num_logprobs", None) == -1
                    # Keep even the rare raw-vocabulary row as a view; select
                    # its sampled token only while assembling final output.
                    logprob_view = (
                        logprobs.logprobs[index]
                        if by_token
                        else logprobs.logprobs[index, :1]
                    )
                    generated_position = start + count - prompt_count
                    if generated_position < 0:
                        raise RuntimeError("vLLM sampled before completing the prompt")
                if any(
                    t is not None and not t.is_cuda for t in (token_view, logprob_view)
                ):
                    raise RuntimeError("GPU output capture received CPU payloads")
                entries.append(
                    _StepRequest(
                        key,
                        request_id,
                        prompt_count,
                        start,
                        count,
                        row_start,
                        generated_position,
                        token_view,
                        logprob_view,
                        by_token,
                    )
                )
        return tuple(entries)

    def finish_step(self, entries: tuple[_StepRequest, ...], output: Any) -> None:
        """Attach views of the native async routing snapshot before returning it."""
        routes = None
        if self.require_routed_experts:
            snapshot = getattr(output, "_routed_experts", None)
            routes = getattr(snapshot, "routing_data", None)
            if routes is None or not routes.is_cuda or routes.ndim != 3:
                raise RuntimeError(
                    "Native async output lacks its GPU routed-expert snapshot"
                )
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        with self._lock:
            self._reap_storages()
            for entry in entries:
                if entry.key in self._closed_keys:
                    continue
                try:
                    state = self._requests.get(entry.key)
                    if (
                        state is not None
                        and state.native_request_id != entry.native_request_id
                    ):
                        raise RuntimeError(
                            "GPU output capture key reused by another request"
                        )
                    if state is None:
                        state = _RequestCapture(
                            entry.native_request_id, entry.prompt_token_count
                        )
                    # Preserve routes already emitted before preemption. Only
                    # pending prefill positions may be replaced by a later step.
                    skip = min(
                        max(state.committed_route_end - entry.start, 0), entry.count
                    )
                    end = entry.row_start + entry.count
                    if routes is not None and end > routes.shape[0]:
                        raise RuntimeError(
                            "Native GPU routing snapshot is shorter than scheduled tokens"
                        )
                    route_view = (
                        routes[entry.row_start + skip : end]
                        if routes is not None and skip < entry.count
                        else None
                    )
                    fragment = _Fragment(
                        entry.start + skip,
                        route_view,
                        entry.generated_position,
                        entry.token_id,
                        entry.logprob,
                        ready,
                        entry.logprob_by_token,
                    )
                    self._retain_fragment(state, fragment)
                    self._requests[entry.key] = state
                    if entry.generated_position is not None:
                        state.committed_route_end = max(
                            state.committed_route_end, entry.start + entry.count
                        )
                except Exception as error:
                    self.fail_keys((entry.key,), error)

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
            self._reap_storages()
            self._check_budget(additional)
            assembly_keys: set[int] = set()
            try:
                ids = torch.empty(
                    generated_token_count, dtype=torch.int64, device=self.device
                )
                self._retain_tensors(assembly_keys, (ids,))
                logprobs = torch.empty(
                    generated_token_count, dtype=torch.float32, device=self.device
                )
                self._retain_tensors(assembly_keys, (logprobs,))
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
                self._retain_tensors(assembly_keys, (assembled_routes,))
                token_coverage = [False] * generated_token_count
                route_coverage = [False] * expected_routes
                for fragment in state.fragments:
                    export_stream = torch.cuda.current_stream(self.device)
                    export_stream.wait_event(fragment.ready)
                    # Fragments were allocated on the capture stream. The request
                    # drops its references below, before export-stream copies may
                    # finish, so the caching allocator must track these reads.
                    for source in (
                        fragment.routes,
                        fragment.token_id,
                        fragment.logprob,
                    ):
                        if source is not None:
                            source.record_stream(export_stream)
                    pos = fragment.generated_position
                    if pos is not None and pos < generated_token_count:
                        if fragment.token_id is None or fragment.logprob is None:
                            raise RuntimeError(
                                "Generated GPU fragment lacks sampled IDs/logprobs"
                            )
                        ids[pos : pos + 1].copy_(fragment.token_id)
                        selected_logprob = (
                            fragment.logprob.gather(
                                0, fragment.token_id.to(torch.int64)
                            )
                            if fragment.logprob_by_token
                            else fragment.logprob
                        )
                        logprobs[pos : pos + 1].copy_(selected_logprob)
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
                except Exception:
                    # None of these descriptors has left the producer yet.
                    for handle in handles:
                        _release_unopened_handle(handle)
                    raise
                self._leases[lease_id] = _OwnedLease(
                    descriptor, tensors, ready, assembly_keys
                )
                self._release_request(state, ready)
                del self._requests[capture_key]
                self._closed_keys.add(capture_key)
                self._errors.pop(capture_key, None)
                return descriptor
            except Exception:
                # Allocation or validation may fail after copies were queued.
                # Keep both their inputs and partial assembly charged until
                # those reads/writes complete; never synchronize the worker.
                failed_ready = torch.cuda.Event()
                failed_ready.record(torch.cuda.current_stream(self.device))
                self._mark_storage_reads(state.storage_keys, failed_ready)
                self._release_storage_keys(assembly_keys, failed_ready)
                raise

    def release(self, lease_id: str) -> None:
        """Release only after the frontend has finished PUT and dropped imports."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is not None:
                self._release_storage_keys(lease.storage_keys, lease.ready)

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
                self._release_request(state)
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
            for state in self._requests.values():
                self._release_request(state)
            self._requests.clear()
            self._closed_keys.clear()
            self._errors.clear()
            self._reap_storages()


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
