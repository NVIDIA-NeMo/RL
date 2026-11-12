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

"""Read how far each captured call has decoded straight from vLLM's request state."""

from __future__ import annotations

import importlib.metadata
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from packaging.version import Version

from nemo_rl.models.generation.prefix_recovery.progress import (
    GenerationProgress,
    ProgressRead,
)
from nemo_rl.models.generation.vllm.utils import extract_generation_logprobs

if TYPE_CHECKING:
    from nemo_rl.models.generation.vllm.vllm_worker_async import _AsyncLLMHTTPClient

# The vLLM release whose private OutputProcessor layout these reads were
# checked against; see VllmProgressReader.
_PROGRESS_READ_VLLM_RELEASE = (0, 29)


@dataclass(frozen=True)
class _FinishedEngineRequest:
    """A finished request's state, kept until its consumer has the final output."""

    request_state: Any
    finish_reason: str | None
    stop_reason: str | int | None


class VllmProgressReader:
    """Read captured calls' progress from vLLM's request state on the caller's thread.

    Queuing each read onto the engine loop waited behind everything else on it
    (router-replay route reads measured 52 ms median, 588 ms p95 and 40 s worst
    that way, against 1-4 ms read directly). Instead, each
    ``OutputProcessor.process_outputs`` call runs under ``_lock``, so a read
    under the same lock sees each call between output chunks (up to
    ``VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`` requests; different calls may be one
    decode step apart) and waits for at most one chunk. ``_finish_request``
    keeps a finished request's state until its consumer has the final output,
    so a call vLLM just finished is still read whole.

    Raises:
        RuntimeError: The interpreter or vLLM is not one these private reads
            were checked on, or another reader already wraps this engine's
            output processor.
    """

    def __init__(self, client: _AsyncLLMHTTPClient) -> None:
        if not (sys.implementation.name == "cpython" and sys._is_gil_enabled()):
            raise RuntimeError(
                "generation-prefix cuts read vLLM's request state from another "
                "thread, which needs GIL-enabled CPython"
            )
        installed = importlib.metadata.version("vllm")
        if Version(installed).release[:2] != _PROGRESS_READ_VLLM_RELEASE:
            expected = ".".join(map(str, _PROGRESS_READ_VLLM_RELEASE))
            raise RuntimeError(
                "generation-prefix cuts read vLLM's private OutputProcessor state, "
                f"checked on vLLM {expected}.x; vLLM {installed} is installed. "
                "Re-check nemo_rl.models.generation.vllm.progress_reader."
            )
        try:
            processor = client.output_processor
            process_outputs = processor.process_outputs
            finish_request = processor._finish_request
        except AttributeError as e:
            raise RuntimeError(
                "generation-prefix cuts wrap vLLM's OutputProcessor.process_outputs "
                "and _finish_request, which the installed vLLM does not have; "
                "update nemo_rl.models.generation.vllm.progress_reader"
            ) from e
        # Marked on the processor: a second wrap would read under a lock vLLM
        # no longer takes.
        if getattr(processor, "_nrl_progress_reads", False):
            raise RuntimeError(
                "another reader already wraps this vLLM OutputProcessor for "
                "generation-progress reads"
            )
        self._client = client
        # Reentrant: vLLM's finish hook runs inside the process_outputs call
        # that holds it.
        self._lock = threading.RLock()
        # Finished captured requests whose consumers have not yet taken the
        # final output, by engine request ID.
        self._finished_requests: dict[str, _FinishedEngineRequest | None] = {}

        def locked_process_outputs(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return process_outputs(*args, **kwargs)

        def retaining_finish_request(req_state: Any) -> Any:
            with self._lock:
                request_id = req_state.external_req_id
                if client.is_captured(request_id):
                    # vLLM queued the final output just before this hook.
                    output = getattr(req_state.queue, "output", None)
                    choices = getattr(output, "outputs", None) or ()
                    finished = (
                        _FinishedEngineRequest(
                            req_state,
                            getattr(choices[0], "finish_reason", None),
                            getattr(choices[0], "stop_reason", None),
                        )
                        if len(choices) == 1
                        else None
                    )
                    # A second engine request under one ID cannot be read.
                    self._finished_requests[request_id] = (
                        None if request_id in self._finished_requests else finished
                    )
                return finish_request(req_state)

        processor.process_outputs = locked_process_outputs
        processor._finish_request = retaining_finish_request
        processor._nrl_progress_reads = True
        client.add_request_done_listener(self._forget)

    def _forget(self, request_id: str) -> None:
        # The consumer took the final output before asking for the next one,
        # so its capture state has recorded the tokens.
        with self._lock:
            self._finished_requests.pop(request_id, None)

    def read(
        self, calls: Sequence[tuple[str, int]], *, timeout_s: float | None
    ) -> ProgressRead:
        """Copy each captured call's generated token IDs past ``start``, at one boundary.

        Args:
            calls: ``(model call ID, start)`` per call; ``start`` is the
                number of generated tokens the call has already recorded.
            timeout_s: How long to wait for vLLM to finish its current output
                chunk; ``None`` waits indefinitely.

        Raises:
            TimeoutError: vLLM held the lock past ``timeout_s``.
        """
        started = time.perf_counter()
        if not self._lock.acquire(
            timeout=-1 if timeout_s is None else max(0.0, timeout_s)
        ):
            raise TimeoutError("vLLM held its output lock past the cut's deadline")
        acquired = time.perf_counter()
        try:
            processor = self._client.output_processor
            engine_request_ids = {
                model_call_id: request_id
                for request_id, model_call_id in self._client.captured_request_ids().items()
            }
            progress = {}
            for model_call_id, start in calls:
                request_id = engine_request_ids.get(model_call_id)
                if request_id is None:
                    continue
                one = self._read_one_request(processor, request_id, start)
                if one is not None:
                    progress[model_call_id] = one
        finally:
            self._lock.release()
        return ProgressRead(
            progress, acquired - started, time.perf_counter() - acquired
        )

    def logprobs(
        self, token_ids: Sequence[int], logprobs: Any, *, start: int
    ) -> list[float]:
        return extract_generation_logprobs(token_ids, logprobs, start=start)

    def _read_one_request(
        self, processor: Any, request_id: str, start: int
    ) -> GenerationProgress | None:
        if request_id in self._finished_requests:
            finished = self._finished_requests[request_id]
            if finished is None:
                return None
            request_state = finished.request_state
            finish_reason, stop_reason = finished.finish_reason, finished.stop_reason
        else:
            internal_ids = processor.external_req_ids.get(request_id)
            if not internal_ids or len(internal_ids) != 1:
                return None
            request_state = processor.request_states.get(internal_ids[0])
            if request_state is None:
                return None
            finish_reason = stop_reason = None
        if request_state.detokenizer is None:
            return None
        logprobs_processor = request_state.logprobs_processor
        return GenerationProgress(
            token_ids=list(request_state.detokenizer.output_token_ids[start:]),
            logprobs=None
            if logprobs_processor is None
            else logprobs_processor.logprobs,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )
