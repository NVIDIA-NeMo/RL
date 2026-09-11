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
"""Frontend ownership of CUDA payloads through the existing staging PUT."""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import torch

from nemo_rl.data_plane.gpu_token_payload import BoundGpuTokenSink, GpuTokenPayload
from nemo_rl.models.generation.vllm.gpu_output_capture import (
    GpuOutputCaptureCapabilities,
    GpuOutputImportError,
    GpuOutputLease,
    import_gpu_output_lease,
)
from nemo_rl.utils.routed_experts_codec import decode_routed_experts

_Result = TypeVar("_Result")
LOGGER = logging.getLogger(__name__)


def _backfill_cached_prefix_routes(
    routes: torch.Tensor | None,
    ranges: tuple[tuple[int, int], ...],
    cpu_envelope: str | None,
    *,
    prompt_token_count: int,
    generated_token_count: int,
) -> None:
    """Fill only proven cached-prefix holes from the canonical CPU response."""
    if not ranges:
        return
    total = prompt_token_count + generated_token_count
    if routes is None or routes.ndim != 3 or routes.shape[0] != total:
        raise ValueError("Cached-prefix backfill requires full aligned GPU routes")
    previous_end = 0
    prefix_end = min(prompt_token_count, max(total - 1, 0))
    for start, end in ranges:
        if (
            type(start) is not int
            or type(end) is not int
            or not previous_end <= start < end <= prefix_end
        ):
            raise ValueError("GPU route backfill ranges must be ordered prompt slices")
        previous_end = end
    if not isinstance(cpu_envelope, str):
        raise ValueError("Cached-prefix GPU routes require the CPU routing record")
    cpu_routes = decode_routed_experts(cpu_envelope, routes.dtype)
    if cpu_routes.shape != routes.shape:
        raise ValueError("Cached-prefix CPU and GPU routing shapes do not match")
    for start, end in ranges:
        # Only history absent from the current GPU snapshots crosses H2D.
        # Fresh routes and sampled outputs retain their GPU sources.
        routes[start:end].copy_(cpu_routes[start:end])


async def _await_completion(task: asyncio.Task[_Result]) -> _Result:
    """Delay cancellation until work using a producer's allocation has stopped."""
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancellation = error
        except Exception:
            if cancellation is not None:
                raise cancellation
            raise
    if cancellation is not None:
        raise cancellation
    return result


class CaptureRpcClient(Protocol):
    """The engine-loop-safe subset of the serving engine client."""

    async def collective_rpc(self, method: str, *, args: tuple[Any, ...]) -> Any: ...


@dataclass(kw_only=True)
class CapturedModelCall:
    """A call's Gym lifecycle and optional retained GPU payload lease."""

    capture: Any
    call: Any
    prompt_token_ids: list[int]
    gpu_sink: BoundGpuTokenSink | None
    capture_key: str | None
    lease: GpuOutputLease | None = None
    ipc_handles_consumed: bool = False
    prepare_payload: Callable[[], GpuTokenPayload] | None = None


class GpuCaptureHost:
    """Import worker-owned buffers and release them only after staging ends."""

    def __init__(self, rpc: CaptureRpcClient, device: torch.device) -> None:
        self._rpc = rpc
        if device.type != "cuda" or device.index is None:
            raise ValueError("GPU capture requires an explicit CUDA device ordinal")
        self._device_index = device.index

    @property
    def device(self) -> torch.device:
        """The producer's physical device in the frontend's CUDA namespace."""
        return torch.device("cuda", self._device_index)

    @classmethod
    async def create(
        cls, rpc: CaptureRpcClient, *, require_routed_experts: bool
    ) -> GpuCaptureHost | None:
        """Use retained output when available; otherwise keep the existing PUT."""
        try:
            capabilities = await rpc.collective_rpc(
                "configure_gpu_output_capture",
                args=(socket.gethostname(), require_routed_experts),
            )
            owners = [
                capability
                for capability in capabilities
                if isinstance(capability, GpuOutputCaptureCapabilities)
                and capability.owner
            ]
            # The current TQ executor uses its thread-default CUDA device.
            # Resolve physical identity rather than comparing worker ordinals.
            if (
                len(owners) == 1
                and owners[0].hostname == socket.gethostname()
                and torch.cuda.device_count()
                and str(torch.cuda.get_device_properties(0).uuid) == owners[0].gpu_uuid
            ):
                return cls(rpc, torch.device("cuda", 0))
        except Exception as error:
            LOGGER.debug(
                "GPU output reuse unavailable; using existing CPU PUT: %s", error
            )
        return None

    async def bind(
        self,
        state: CapturedModelCall,
        *,
        generated_token_count: int,
        routed_experts_dtype: torch.dtype,
        routed_experts_cpu: str | None = None,
    ) -> None:
        """Bind original device tensors to the call's completion-time sink."""
        if state.capture_key is None or state.gpu_sink is None:
            raise ValueError(
                "GPU payload binding requires an admitted GPU capture call"
            )

        async def export() -> GpuOutputLease:
            leases = await self._rpc.collective_rpc(
                "export_gpu_output_capture",
                args=(
                    state.capture_key,
                    generated_token_count,
                    len(state.prompt_token_ids),
                ),
            )
            owned = [lease for lease in leases if lease is not None]
            if len(owned) != 1 or not isinstance(owned[0], GpuOutputLease):
                raise RuntimeError(
                    "GPU output capture did not return exactly one payload lease"
                )
            lease = owned[0]
            if lease.capture_key != state.capture_key:
                raise RuntimeError("GPU output lease belongs to a different model call")
            state.lease = lease
            return lease

        lease = await _await_completion(asyncio.create_task(export()))

        # Defer import until the existing completion executor performs PUT.
        # This avoids handing CUDA work between two frontend worker threads.
        def import_payload() -> GpuTokenPayload:
            try:
                tensors = import_gpu_output_lease(lease, self.device)
            except GpuOutputImportError as error:
                state.ipc_handles_consumed = error.handles_consumed
                raise
            state.ipc_handles_consumed = True
            routes = tensors.routed_experts
            if routes is not None:
                routes = routes.to(dtype=routed_experts_dtype)
            _backfill_cached_prefix_routes(
                routes,
                lease.routed_experts_prefix_backfill_ranges,
                routed_experts_cpu,
                prompt_token_count=len(state.prompt_token_ids),
                generated_token_count=generated_token_count,
            )
            return GpuTokenPayload(
                prompt_len=len(state.prompt_token_ids),
                generated_token_ids=tensors.generated_token_ids,
                generated_logprobs=tensors.generation_logprobs,
                routed_experts=routes,
            )

        state.prepare_payload = import_payload

    async def finish(
        self, state: CapturedModelCall, operation: Callable[[], _Result]
    ) -> _Result:
        """Complete PUT before releasing a lease, including HTTP cancellation."""

        def on_device() -> _Result:
            prepare = state.prepare_payload
            state.prepare_payload = None
            with torch.cuda.device(self.device):
                if prepare is not None and state.gpu_sink is not None:
                    try:
                        state.gpu_sink.bind(prepare())
                    except Exception as error:
                        state.gpu_sink.clear()
                        LOGGER.warning(
                            "GPU output import unavailable; using existing CPU PUT: %s",
                            error,
                        )
                return operation()

        task = asyncio.create_task(asyncio.to_thread(on_device))
        try:
            result = await _await_completion(task)
        finally:
            await self.release(state)
        return result

    async def release(self, state: CapturedModelCall) -> None:
        """Drop frontend views before acknowledging release to the producer."""
        try:
            await _await_completion(asyncio.create_task(self._release(state)))
        except Exception as error:
            # A failed cleanup ACK must not change an already completed PUT.
            # Keep the lease state; only a confirmed ACK relinquishes ownership.
            LOGGER.warning("GPU output cleanup acknowledgement failed: %s", error)

    async def _release(self, state: CapturedModelCall) -> None:
        state.prepare_payload = None
        if state.gpu_sink is not None:
            state.gpu_sink.clear()
        if state.lease is not None:
            # Sink failures can leave asynchronous copies in flight too.
            await asyncio.to_thread(torch.cuda.synchronize, self.device)
            release_method = (
                "release_gpu_output_capture"
                if state.ipc_handles_consumed
                else "abandon_unimported_gpu_output_capture"
            )
            await self._rpc.collective_rpc(release_method, args=(state.lease.lease_id,))
            state.lease = None
            state.capture_key = None
            state.ipc_handles_consumed = False
        if state.capture_key is not None:
            await self._rpc.collective_rpc(
                "discard_gpu_output_capture", args=(state.capture_key,)
            )
            state.capture_key = None
