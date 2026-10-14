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
"""Bounded selected-logprob capture before vLLM's Python output formatting.

The engine loop owns each accumulator until generation ends. Only then does the
staging pool read it. No per-step futures, retained NumPy views, or capture data
are passed through the HTTP iterator. The existing HTTP loop bridge is separate.
"""

from array import array
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from nemo_rl.experience.direct_capture import (
    NativeCall,
    PackedNativeCall,
    _pack_tokens,
)


class PackedLogprobBuffer:
    """Own flat token/logprob storage, capped before each append.

    Arrays grow amortized with generated tokens, never with the number of output
    objects. Source buffers may be reused immediately after append returns.
    Prompt packing, finite-value validation and tensor staging run in the pool.
    """

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.prompt: list[int] | None = None
        self.tokens = array("q")
        self.logprobs = array("f")
        self.failure: str | None = None
        self.final: Any = None

    def append(self, token_ids: list[int], logprobs: Any) -> None:
        """Copy accepted tokens and selected float32 values from EngineCore.

        Capture errors must not escape vLLM's shared output-handler task and
        take down unrelated requests. The staging barrier reports them instead.
        """
        if self.failure is not None:
            return
        try:
            self._append(token_ids, logprobs)
        except (ValueError, TypeError, OverflowError) as error:
            self.failure = str(error)
            self.tokens = array("q")
            self.logprobs = array("f")

    def _append(self, token_ids: list[int], logprobs: Any) -> None:
        count = len(token_ids)
        if len(self.tokens) + count > self.max_tokens:
            raise ValueError("packed capture generated-token limit exceeded")
        if count == 0 and logprobs is None:
            return
        if logprobs is None:
            raise ValueError("missing selected-token logprobs")
        ids, values, _, _ = logprobs
        if (
            not isinstance(ids, np.ndarray)
            or not isinstance(values, np.ndarray)
            or ids.ndim != 2
            or values.ndim != 2
            or ids.shape != values.shape
            or values.shape[0] != count
            or values.shape[1] < 1
            or values.dtype != np.dtype("float32")
            or ids.dtype.kind != "i"
        ):
            raise ValueError("invalid native selected-logprob layout")
        # vLLM 0.29 places the sampled token in column zero. Verify alignment,
        # including multi-token speculative batches, before accepting values.
        if count == 1:
            aligned = ids[0, 0] == token_ids[0]
        else:
            aligned = np.array_equal(ids[:, 0], token_ids)
        if not aligned:
            raise ValueError("selected logprobs do not match accepted tokens")
        self.tokens.extend(token_ids)
        if values.shape[1] == 1 and values.flags.c_contiguous:
            self.logprobs.frombytes(memoryview(values).cast("B"))
        else:
            # Public top-k requests still use the first (sampled) column only.
            self.logprobs.frombytes(values[:, 0].tobytes())

    def extract(
        self,
        rollout_id: str,
        *,
        ordinal: int,
        weight_version: int,
        packing: Literal["fused", "legacy"],
    ) -> NativeCall | PackedNativeCall:
        """Validate and pack after the engine has relinquished the buffer."""
        if self.failure is not None:
            raise ValueError(self.failure)
        if self.final is None or not self.final.finished:
            raise ValueError("generation did not finish")
        if not self.prompt:
            raise ValueError("empty prompt")
        if len(self.final.outputs) != 1 or self.final.outputs[0].index != 0:
            raise ValueError("direct capture requires one completion per request")
        completion = self.final.outputs[0]
        if not self.tokens or completion.finish_reason not in {"stop", "length"}:
            raise ValueError("empty or aborted generation")
        if len(self.tokens) != len(self.logprobs):
            raise ValueError("missing selected-token logprobs")
        if not np.isfinite(np.frombuffer(self.logprobs, dtype=np.float32)).all():
            raise ValueError("nonfinite selected-token logprob")
        call_id = f"{rollout_id}/{ordinal}"
        if packing == "fused":
            return PackedNativeCall(
                rollout_id,
                call_id,
                ordinal,
                weight_version,
                _pack_tokens(self.prompt, self.tokens),
                len(self.prompt),
                self.logprobs,
                completion.routed_experts,
            )
        return NativeCall(
            rollout_id,
            call_id,
            ordinal,
            weight_version,
            tuple(self.prompt),
            tuple(self.tokens),
            tuple(self.logprobs),
            completion.routed_experts,
        )


@dataclass
class _PackedLogprobsProcessor:
    """Explicit vLLM LogprobsProcessor surface; normal API formatting is opt-in."""

    original: Any
    buffer: PackedLogprobBuffer
    export_logprobs: bool

    @property
    def logprobs(self) -> Any:
        return self.original.logprobs if self.export_logprobs else None

    @property
    def cumulative_logprob(self) -> float | None:
        return self.original.cumulative_logprob if self.export_logprobs else None

    @property
    def prompt_logprobs(self) -> Any:
        return self.original.prompt_logprobs

    def pop_prompt_logprobs(self) -> Any:
        return self.original.pop_prompt_logprobs()

    def update_from_output(self, output: Any) -> None:
        self.buffer.append(output.new_token_ids, output.new_logprobs)
        if self.export_logprobs:
            self.original.update_from_output(output)
        elif output.new_prompt_logprobs_tensors is not None:
            self.original._update_prompt_logprobs(output.new_prompt_logprobs_tensors)


@dataclass(frozen=True)
class _Binding:
    buffer: PackedLogprobBuffer
    export_logprobs: bool


@dataclass
class _CaptureOutputQueue:
    queue: Any
    buffer: PackedLogprobBuffer

    def put(self, output: Any) -> None:
        # vLLM's DELTA collector can merge the terminal output into an earlier
        # chunk without copying routed_experts. Retain it before that merge.
        if not isinstance(output, Exception) and output.finished:
            self.buffer.final = output
        self.queue.put(output)


class PackedLogprobCapture:
    """Install a per-engine hook at request registration, before the first await.

    Only registered external request IDs are intercepted. This changes one
    OutputProcessor instance, not vLLM classes or files. All methods run on the
    owning engine loop. The private seam is deliberately pinned to vLLM 0.29.0.
    """

    def __init__(self, output_processor: Any) -> None:
        # vLLM is optional outside generation workers (including CPU tests).
        from vllm import __version__

        if __version__.split("+")[0] != "0.29.0":
            raise RuntimeError(
                "packed logprob capture requires vLLM 0.29.0; "
                "use direct.logprob_capture=python for other versions"
            )
        self._processor = output_processor
        self._add_original = output_processor.add_request
        self._bindings: dict[str, _Binding] = {}
        output_processor.add_request = self._add_request

    def register(
        self, request_id: str, buffer: PackedLogprobBuffer, *, export_logprobs: bool
    ) -> None:
        if request_id in self._bindings:
            raise ValueError("duplicate packed capture request ID")
        self._bindings[request_id] = _Binding(buffer, export_logprobs)

    def unregister(self, request_id: str) -> None:
        self._bindings.pop(request_id, None)

    def _add_request(
        self,
        request: Any,
        prompt: str | None,
        parent_req: Any = None,
        request_index: int = 0,
        queue: Any = None,
    ) -> None:
        binding = self._bindings.get(request.external_req_id)
        if binding is not None:
            if parent_req is not None or request_index != 0 or request.resumable:
                raise ValueError("packed capture requires one non-resumable completion")
            if binding.buffer.prompt is not None:
                raise ValueError("packed capture request was registered twice")
            if not request.prompt_token_ids or request.prompt_embeds is not None:
                raise ValueError("packed capture requires prompt token IDs")
        capture_queue = (
            _CaptureOutputQueue(queue, binding.buffer)
            if binding is not None and queue is not None
            else queue
        )
        self._add_original(request, prompt, parent_req, request_index, capture_queue)
        if binding is not None:
            state = self._processor.request_states[request.request_id]
            binding.buffer.prompt = state.prompt_token_ids
            state.logprobs_processor = _PackedLogprobsProcessor(
                state.logprobs_processor, binding.buffer, binding.export_logprobs
            )
