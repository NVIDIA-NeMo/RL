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

_Result = TypeVar("_Result")


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
        cls,
        rpc: CaptureRpcClient,
        *,
        max_retained_bytes: int,
        require_routed_experts: bool,
    ) -> GpuCaptureHost:
        capabilities = await rpc.collective_rpc(
            "configure_gpu_output_capture",
            args=(max_retained_bytes, socket.gethostname(), require_routed_experts),
        )
        owners = [
            capability
            for capability in capabilities
            if isinstance(capability, GpuOutputCaptureCapabilities) and capability.owner
        ]
        if len(owners) != 1:
            raise RuntimeError(
                "GPU output capture requires exactly one producer per engine"
            )
        owner = owners[0]
        if owner.hostname != socket.gethostname():
            raise RuntimeError(
                "GPU output capture requires the producer on the frontend host"
            )
        # Device ordinals can differ across Ray workers. Resolve physical identity.
        for index in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(index).uuid) == owner.gpu_uuid:
                # TQ's backend executor uses its thread-default CUDA ordinal.
                # A caller-side device context cannot bind that other thread.
                if index != 0:
                    raise RuntimeError(
                        "GPU output capture requires the producer mapped to frontend cuda:0 "
                        "with the current TransferQueue executor"
                    )
                return cls(rpc, torch.device("cuda", index))
        raise RuntimeError(
            "GPU capture producer's device is not visible to the frontend"
        )

    async def bind(
        self,
        state: CapturedModelCall,
        *,
        generated_token_count: int,
        routed_experts_dtype: torch.dtype,
    ) -> None:
        """Bind original device tensors to the call's completion-time sink."""
        await _await_completion(
            asyncio.create_task(
                self._bind(
                    state,
                    generated_token_count=generated_token_count,
                    routed_experts_dtype=routed_experts_dtype,
                )
            )
        )

    async def _bind(
        self,
        state: CapturedModelCall,
        *,
        generated_token_count: int,
        routed_experts_dtype: torch.dtype,
    ) -> None:
        if state.capture_key is None or state.gpu_sink is None:
            raise ValueError(
                "GPU payload binding requires an admitted GPU capture call"
            )
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

        # IPC waits and dtype casts run off the serving loop. The lease remains
        # owned by state even when import or later integrity validation fails.
        def import_payload() -> GpuTokenPayload:
            with torch.cuda.device(self.device):
                try:
                    tensors = import_gpu_output_lease(lease, self.device)
                except GpuOutputImportError as error:
                    state.ipc_handles_consumed = error.handles_consumed
                    raise
                state.ipc_handles_consumed = True
                routes = tensors.routed_experts
                if routes is not None:
                    routes = routes.to(dtype=routed_experts_dtype)
                # Staging executes on a different executor thread. Complete
                # the IPC wait and casts before handing the payload to it.
                torch.cuda.current_stream(self.device).synchronize()
                return GpuTokenPayload(
                    prompt_len=len(state.prompt_token_ids),
                    generated_token_ids=tensors.generated_token_ids,
                    generated_logprobs=tensors.generation_logprobs,
                    routed_experts=routes,
                )

        payload = await asyncio.to_thread(import_payload)
        state.gpu_sink.bind(payload)

    async def finish(
        self, state: CapturedModelCall, operation: Callable[[], _Result]
    ) -> _Result:
        """Complete PUT before releasing a lease, including HTTP cancellation."""

        def on_device() -> _Result:
            with torch.cuda.device(self.device):
                result = operation()
            return result

        task = asyncio.create_task(asyncio.to_thread(on_device))
        try:
            result = await _await_completion(task)
        finally:
            await self.release(state)
        return result

    async def release(self, state: CapturedModelCall) -> None:
        """Drop frontend views before acknowledging release to the producer."""
        await _await_completion(asyncio.create_task(self._release(state)))

    async def _release(self, state: CapturedModelCall) -> None:
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
            state.ipc_handles_consumed = False
        if state.capture_key is not None:
            await self._rpc.collective_rpc(
                "discard_gpu_output_capture", args=(state.capture_key,)
            )
            state.capture_key = None
