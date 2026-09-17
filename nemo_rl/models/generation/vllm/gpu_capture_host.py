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
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import torch

from nemo_rl.data_plane.gpu_token_payload import BoundGpuTokenSink, GpuTokenPayload
from nemo_rl.models.generation.vllm.gpu_output_capture import (
    GpuOutputImportError,
    GpuOutputLease,
    import_gpu_output_lease,
)

if TYPE_CHECKING:
    from ray.actor import ActorHandle

_Result = TypeVar("_Result")
LOGGER = logging.getLogger(__name__)


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
    export_task: asyncio.Task[GpuOutputLease] | None = None
    release_reply: Future[Any] | None = None


class GpuCaptureHost:
    """Import worker-owned buffers and release them only after staging ends."""

    def __init__(
        self,
        rpc: CaptureRpcClient,
        device: torch.device,
        *,
        worker: ActorHandle | None = None,
    ) -> None:
        self._rpc = rpc
        self._worker = worker
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
            owners = [owner for owner in capabilities if owner is not None]
            if len(owners) == 1:
                # TP workers and the frontend can use different CUDA ordinals.
                # TQ binds its transfer threads to this device when attaching.
                for index in range(torch.cuda.device_count()):
                    if (
                        str(torch.cuda.get_device_properties(index).uuid)
                        == owners[0].gpu_uuid
                    ):
                        return cls(
                            rpc, torch.device("cuda", index), worker=owners[0].worker
                        )
        except Exception as error:
            LOGGER.debug(
                "GPU output reuse unavailable; using existing CPU PUT: %s", error
            )
        return None

    async def _call_worker(self, method: str, *, args: tuple[Any, ...]) -> Any:
        if self._worker is not None:
            # The Ray wrapper already exposes execute_method. Calling its owner
            # directly avoids the engine utility queue and other TP workers.
            return [await self._worker.execute_method.remote(method, *args)]
        return await self._rpc.collective_rpc(method, args=args)

    def start_export(
        self,
        state: CapturedModelCall,
        *,
        generated_token_count: int,
    ) -> None:
        """Submit once before CPU formatting, retaining ownership until adoption."""
        if state.export_task is not None or state.lease is not None:
            return
        if state.capture_key is None or state.gpu_sink is None:
            raise ValueError(
                "GPU payload binding requires an admitted GPU capture call"
            )
        routed_experts_start = state.call.admission.prev_len
        args = (
            state.capture_key,
            generated_token_count,
            len(state.prompt_token_ids),
            routed_experts_start,
        )
        # Submit and register result retrieval before the formatter blocks the
        # event loop; the owned task below adopts this same Ray reply once.
        reply = (
            self._worker.execute_method.remote(
                "export_gpu_output_capture", *args
            ).future()
            if self._worker is not None
            else None
        )

        async def export() -> GpuOutputLease:
            leases = (
                [await asyncio.wrap_future(reply)]
                if reply is not None
                else await self._rpc.collective_rpc(
                    "export_gpu_output_capture", args=args
                )
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

        state.export_task = asyncio.create_task(export())

    async def bind(
        self,
        state: CapturedModelCall,
        *,
        generated_token_count: int,
    ) -> None:
        """Adopt the pending export and bind it to the completion-time sink."""
        self.start_export(state, generated_token_count=generated_token_count)
        if state.export_task is not None:
            try:
                await _await_completion(state.export_task)
            finally:
                state.export_task = None
        adopted_lease = state.lease
        if adopted_lease is None:
            raise RuntimeError("GPU output capture has no adopted payload lease")
        lease: GpuOutputLease = adopted_lease

        # Defer import until the existing completion executor performs PUT.
        # This avoids handing CUDA work between two frontend worker threads.
        def import_payload() -> GpuTokenPayload:
            try:
                tensors = import_gpu_output_lease(lease, self.device)
            except GpuOutputImportError as error:
                state.ipc_handles_consumed = error.handles_consumed
                raise
            state.ipc_handles_consumed = True
            return GpuTokenPayload(
                prompt_len=len(state.prompt_token_ids),
                generated_token_ids=tensors.generated_token_ids,
                generated_logprobs=tensors.generation_logprobs,
                routed_experts=tensors.routed_experts,
                routed_experts_prefix_backfill_ranges=(
                    lease.routed_experts_prefix_backfill_ranges
                ),
            )

        state.prepare_payload = import_payload

    async def finish(
        self,
        state: CapturedModelCall,
        operation: Callable[[], _Result],
        *,
        finalize: Callable[[_Result], Any] | None = None,
    ) -> Any:
        """Complete PUT before releasing a lease, including HTTP cancellation."""
        overlap = (
            self._worker is not None
            and state.lease is not None
            and state.export_task is None
        )

        def on_device() -> Any:
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
                result = operation()
                if finalize is not None and overlap:
                    self._start_release(state)
                    return finalize(result)
                return result

        task = asyncio.create_task(asyncio.to_thread(on_device))
        try:
            result = await _await_completion(task)
        finally:
            await self.release(state)
        if finalize is not None and not overlap:
            return finalize(result)
        return result

    def _start_release(self, state: CapturedModelCall) -> None:
        """Fence all allocation users before overlapping ACK with CPU formatting."""
        if state.lease is None or state.release_reply is not None:
            return
        assert self._worker is not None
        state.prepare_payload = None
        if state.gpu_sink is not None:
            state.gpu_sink.clear()
        try:
            torch.cuda.synchronize(self.device)
            method = (
                "release_gpu_output_capture"
                if state.ipc_handles_consumed
                else "abandon_unimported_gpu_output_capture"
            )
            state.release_reply = self._worker.execute_method.remote(
                method, state.lease.capture_key
            ).future()
        except Exception as error:
            # Preserve release()'s warning-only error policy and producer lease.
            # The same cleanup attempt must not fence or dispatch a second time.
            state.release_reply = Future()
            state.release_reply.set_exception(error)

    async def release(self, state: CapturedModelCall) -> None:
        """Drop frontend views before acknowledging release to the producer."""
        try:
            await _await_completion(asyncio.create_task(self._release(state)))
        except Exception as error:
            # A failed cleanup ACK must not change an already completed PUT.
            # Keep the lease state; only a confirmed ACK relinquishes ownership.
            LOGGER.warning("GPU output cleanup acknowledgement failed: %s", error)

    async def _release(self, state: CapturedModelCall) -> None:
        if state.export_task is not None:
            try:
                await state.export_task
            except Exception as error:
                LOGGER.debug("GPU output export failed before cleanup: %s", error)
            finally:
                state.export_task = None
        state.prepare_payload = None
        if state.gpu_sink is not None:
            state.gpu_sink.clear()
        if state.lease is not None:
            if state.release_reply is not None:
                try:
                    await asyncio.wrap_future(state.release_reply)
                finally:
                    state.release_reply = None
            else:
                # Sink failures can leave asynchronous copies in flight too.
                await asyncio.to_thread(torch.cuda.synchronize, self.device)
                release_method = (
                    "release_gpu_output_capture"
                    if state.ipc_handles_consumed
                    else "abandon_unimported_gpu_output_capture"
                )
                await self._call_worker(release_method, args=(state.lease.capture_key,))
            state.lease = None
            state.capture_key = None
            state.ipc_handles_consumed = False
        if state.capture_key is not None:
            await self._call_worker(
                "discard_gpu_output_capture", args=(state.capture_key,)
            )
            state.capture_key = None
