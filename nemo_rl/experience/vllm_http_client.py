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

"""HTTP-to-engine loop bridge with optional native output capture."""

import asyncio
import copy
import logging
from collections.abc import Callable, Coroutine
from typing import Any, AsyncGenerator

from nemo_rl.experience.direct_capture import CaptureContext, current_capture
from nemo_rl.experience.packed_logprobs import PackedLogprobBuffer, PackedLogprobCapture

LOGGER = logging.getLogger(__name__)


class _AsyncLLMHTTPClient:
    """Keep HTTP generation on the loop that owns AsyncLLM request state.

    The engine-client surface is explicit. Do not add a ``__getattr__`` fallback.
    Add each new member here and decide whether it must run on the engine loop.
    """

    def __init__(self, engine_client: Any, engine_loop: asyncio.AbstractEventLoop):
        self._engine_client = engine_client
        self._engine_loop = engine_loop
        self.model_config = engine_client.model_config
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        self.vllm_config = engine_client.vllm_config
        self._packed_capture: PackedLogprobCapture | None = None

    def enable_packed_capture(self) -> None:
        """Install the version-checked frontend hook on the engine loop."""
        if asyncio.get_running_loop() is not self._engine_loop:
            raise RuntimeError("install packed capture on the engine loop")
        if self._packed_capture is None:
            self._packed_capture = PackedLogprobCapture(
                self._engine_client.output_processor
            )

    async def _run_on_engine_loop(
        self, operation: Callable[[], Coroutine[Any, Any, Any]]
    ) -> Any:
        if asyncio.get_running_loop() is self._engine_loop:
            return await operation()

        future = asyncio.run_coroutine_threadsafe(operation(), self._engine_loop)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        context = current_capture.get()
        # num_logprobs also covers explicit logprob_token_ids requests.
        export_logprobs = (
            context is not None and sampling_params.num_logprobs is not None
        )
        if context is not None:
            sampling_params = copy.copy(sampling_params)
            # Obtain only the selected-token probability, without adding HTTP
            # logprobs or token IDs to the user response.
            if not export_logprobs:
                sampling_params.logprobs = 0
        return self._generate(
            prompt, sampling_params, request_id, kwargs, context, export_logprobs
        )

    async def _generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        kwargs: dict[str, Any],
        context: CaptureContext | None,
        export_logprobs: bool,
    ) -> AsyncGenerator[Any, None]:
        iterator = None
        completed = False
        captured = []
        registered = False
        packed: PackedLogprobBuffer | None = None
        if context is not None and context.owner.config.logprob_capture == "packed":
            if self._packed_capture is None:
                raise RuntimeError("packed logprob capture hook is not installed")
            limit = context.owner.config.max_generated_tokens_per_call
            if sampling_params.max_tokens is not None:
                limit = min(limit, sampling_params.max_tokens)
            packed = PackedLogprobBuffer(limit)
        ticket = context.owner.begin(context) if context is not None else None
        # vLLM's DELTA output owns each chunk; cumulative outputs can mutate
        # until the terminal output, so retain only that final output.
        delta = (
            sampling_params.output_kind.name == "DELTA"
            if context is not None
            else False
        )

        async def next_output() -> Any:
            nonlocal iterator, registered
            if iterator is None:
                if packed is not None:
                    assert self._packed_capture is not None
                    self._packed_capture.register(
                        request_id, packed, export_logprobs=export_logprobs
                    )
                    registered = True
                iterator = self._engine_client.generate(
                    prompt, sampling_params, request_id, **kwargs
                )
            try:
                output = await anext(iterator)
            except StopAsyncIteration:
                if packed is not None:
                    assert self._packed_capture is not None
                    self._packed_capture.unregister(request_id)
                    registered = False
                raise
            return output

        async def abort() -> None:
            try:
                if iterator is not None:
                    await self._engine_client.abort(request_id)
            finally:
                if registered:
                    assert self._packed_capture is not None
                    self._packed_capture.unregister(request_id)

        try:
            while True:
                try:
                    output = await self._run_on_engine_loop(next_output)
                    if (
                        context is not None
                        and packed is None
                        and (delta or output.finished)
                    ):
                        captured.append(output)
                    if context is not None and any(
                        item.routed_experts is not None for item in output.outputs
                    ):
                        # Upstream vLLM also serializes routes in its serving
                        # layer. Retain the original for staging, but hide the
                        # tensor from the HTTP formatter without copying it.
                        output = copy.copy(output)
                        output.outputs = [copy.copy(item) for item in output.outputs]
                        for item in output.outputs:
                            item.routed_experts = None
                    yield output
                except StopAsyncIteration:
                    completed = True
                    return
        finally:
            if ticket is not None:
                if completed:
                    if packed is not None:
                        context.owner.submit(
                            context,
                            lambda: packed.extract(
                                context.rollout.rollout_id,
                                ordinal=ticket[0],
                                weight_version=ticket[1],
                                packing=context.owner.config.token_packing,
                            ),
                        )
                    else:
                        context.owner.complete(
                            context, request_id, *ticket, captured, delta=delta
                        )
                else:
                    context.owner.abort(context)
            if not completed:
                try:
                    await self._run_on_engine_loop(abort)
                except Exception:
                    LOGGER.exception("Failed to abort vLLM request %s", request_id)

    # These members only read engine status or immutable configuration. Running
    # them on the engine loop added a cross-thread wait to each HTTP request.
    @property
    def errored(self) -> bool:
        return self._engine_client.errored

    @property
    def dead_error(self) -> BaseException:
        return self._engine_client.dead_error

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        """Queue-limit admission check vLLM >= 0.29 runs before every response.

        ``OpenAIServing._preflight`` calls this (vllm-project/vllm#49445,
        ``max_num_queued_reqs`` / ``max_num_queued_tokens``); without it every
        chat completion 500s with ``'_AsyncLLMHTTPClient' object has no
        attribute 'check_admission'``. It only reads scheduler config and
        unfinished-request counters, so it stays off the engine loop like the
        other status reads above. Raises vLLM's HTTP-mapped overflow errors.
        """
        self._engine_client.check_admission(n, request_id=request_id)

    async def is_tracing_enabled(self) -> bool:
        return await self._engine_client.is_tracing_enabled()
