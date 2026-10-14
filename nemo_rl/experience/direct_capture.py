# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native generation capture, independent of Gym and training row layout.

The packed path accumulates native selected logprobs on the engine loop and
submits one bounded job per call. The comparison path retains engine outputs.
Prompt packing, validation, routing tensors and storage run in the pool.
Only rollout finalization waits for those jobs. An incomplete or failed capture
invalidates the rollout; it never becomes an apparently complete receipt.
"""

from __future__ import annotations

import math
import threading
import time
from array import array
from collections.abc import Buffer, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, PositiveInt


class DirectCaptureConfig(BaseModel, extra="allow"):
    """Limits apply per model-serving worker, including in-flight calls."""

    token_packing: Literal["fused", "legacy"] = "fused"
    logprob_capture: Literal["packed", "python"] = "packed"
    max_generated_tokens_per_call: PositiveInt = 262144
    workers: PositiveInt = 2
    max_pending_calls: PositiveInt = 256
    max_rollouts: PositiveInt = 4096
    max_calls_per_rollout: PositiveInt = 512


@dataclass(frozen=True)
class NativeCall:
    rollout_id: str
    call_id: str
    ordinal: int
    weight_version: int
    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    routed_experts: Any


@dataclass(frozen=True)
class PackedNativeCall:
    """Worker-owned int64 tokens for staging; training still reads NativeCall."""

    rollout_id: str
    call_id: str
    ordinal: int
    weight_version: int
    token_ids: array
    prompt_length: int
    logprobs: tuple[float, ...] | array
    routed_experts: Any


def _pack_tokens(prompt: Sequence[int] | Buffer, tokens: Sequence[int]) -> array:
    """Own a full sequence without an intermediate tuple or prefix inference."""
    if isinstance(prompt, Buffer):
        with memoryview(prompt) as source:
            if (
                source.ndim != 1
                or not source.c_contiguous
                or source.itemsize != 8
                or source.format not in {"q", "l", "@q", "@l", "=q", "=l"}
            ):
                raise ValueError(
                    "packed prompt tokens require contiguous native signed int64"
                )
            # Allocate the final size so appending the generation cannot copy
            # the prompt again. The producer must keep source stable until copied.
            packed = array("q", [0]) * (len(source) + len(tokens))
            memoryview(packed).cast("B")[: source.nbytes] = source.cast("B")
            packed[len(source) :] = array("q", tokens)
            return packed
    packed = array("q", prompt)
    packed.extend(tokens)
    return packed


@dataclass
class RolloutCapture:
    rollout_id: str
    temperature: float
    top_p: float
    active: int = 0
    next_ordinal: int = 0
    closing: bool = False
    failure: str | None = None
    jobs: list[Future[dict[str, Any]]] = field(default_factory=list)
    receipt: dict[str, Any] | None = None


@dataclass(frozen=True)
class CaptureContext:
    owner: DirectCapture
    rollout: RolloutCapture


current_capture: ContextVar[CaptureContext | None] = ContextVar(
    "nrl_direct_capture", default=None
)


def extract_call(
    rollout_id: str,
    call_id: str,
    ordinal: int,
    weight_version: int,
    outputs: list[Any],
    *,
    delta: bool,
    packing: Literal["fused", "legacy"],
) -> NativeCall | PackedNativeCall:
    """Read vLLM 0.29 RequestOutputs after they leave the HTTP path."""
    if not outputs or not outputs[-1].finished:
        raise ValueError("generation did not finish")
    prompt = outputs[-1].prompt_token_ids
    if prompt is None or len(prompt) == 0:
        raise ValueError("empty prompt")
    tokens: list[int] = []
    logprobs: list[float] = []
    selected = outputs if delta else outputs[-1:]
    for output in selected:
        if len(output.outputs) != 1 or output.outputs[0].index != 0:
            raise ValueError("direct capture requires one completion per request")
        completion = output.outputs[0]
        if completion.logprobs is None or len(completion.logprobs) != len(
            completion.token_ids
        ):
            raise ValueError("missing selected-token logprobs")
        for token, distribution in zip(completion.token_ids, completion.logprobs):
            value = float(distribution[token].logprob)
            if not math.isfinite(value):
                raise ValueError("nonfinite selected-token logprob")
            tokens.append(int(token))
            logprobs.append(value)
    final = outputs[-1].outputs[0]
    if not tokens or final.finish_reason not in {"stop", "length"}:
        raise ValueError("empty or aborted generation")
    if packing == "fused":
        return PackedNativeCall(
            rollout_id,
            call_id,
            ordinal,
            weight_version,
            _pack_tokens(prompt, tokens),
            len(prompt),
            tuple(logprobs),
            final.routed_experts,
        )
    return NativeCall(
        rollout_id,
        call_id,
        ordinal,
        weight_version,
        tuple(prompt),
        tuple(tokens),
        tuple(logprobs),
        final.routed_experts,
    )


class DirectCapture:
    """Own admission, asynchronous native staging, and the rollout drain barrier."""

    def __init__(
        self,
        config: DirectCaptureConfig,
        stage: Callable[[NativeCall | PackedNativeCall], dict[str, Any]],
        weight_version: Callable[[], int],
        *,
        clear: Callable[[list[str]], None],
    ) -> None:
        self.config = config
        self._stage = stage
        self._weight_version = weight_version
        self._clear = clear
        self._condition = threading.Condition()
        self._slots = threading.BoundedSemaphore(config.max_pending_calls)
        self._pool = ThreadPoolExecutor(
            config.workers, thread_name_prefix="nrl-capture"
        )
        self._rollouts: dict[str, RolloutCapture] = {}

    def register(
        self, key: str, rollout_id: str, *, temperature: float, top_p: float
    ) -> None:
        with self._condition:
            previous = self._rollouts.get(key)
            if previous is not None:
                if (previous.rollout_id, previous.temperature, previous.top_p) != (
                    rollout_id,
                    temperature,
                    top_p,
                ):
                    raise ValueError(
                        "API key already belongs to another rollout/profile"
                    )
                return
            if len(self._rollouts) >= self.config.max_rollouts:
                raise RuntimeError("direct capture rollout registry is full")
            self._rollouts[key] = RolloutCapture(rollout_id, temperature, top_p)

    def enter(self, key: str) -> CaptureContext:
        with self._condition:
            rollout = self._rollouts[key]
            if rollout.closing:
                raise ValueError("rollout is already closing")
            rollout.active += 1
            return CaptureContext(self, rollout)

    def leave(self, context: CaptureContext) -> None:
        with self._condition:
            context.rollout.active -= 1
            self._condition.notify_all()

    def begin(self, context: CaptureContext) -> tuple[int, int]:
        """Reserve memory before inference; overload is a visible capture failure."""
        with self._condition:
            if context.rollout.next_ordinal >= self.config.max_calls_per_rollout:
                context.rollout.failure = "capture_call_limit"
                raise RuntimeError("direct capture call limit exceeded")
            if not self._slots.acquire(blocking=False):
                context.rollout.failure = "capture_queue_full"
                raise RuntimeError("direct capture queue is full")
            ordinal = context.rollout.next_ordinal
            context.rollout.next_ordinal += 1
            return ordinal, self._weight_version()

    def complete(
        self,
        context: CaptureContext,
        call_id: str,
        ordinal: int,
        weight_version: int,
        outputs: list[Any],
        *,
        delta: bool,
    ) -> None:
        self.submit(
            context,
            lambda: extract_call(
                context.rollout.rollout_id,
                f"{context.rollout.rollout_id}/{ordinal}",
                ordinal,
                weight_version,
                outputs,
                delta=delta,
                packing=self.config.token_packing,
            ),
        )

    def submit(
        self,
        context: CaptureContext,
        extract: Callable[[], NativeCall | PackedNativeCall],
    ) -> None:
        """Transfer one completed call to the bounded staging pool."""

        def stage() -> dict[str, Any]:
            try:
                return self._stage(extract())
            finally:
                self._slots.release()

        # Submission and active-request accounting share the finalization lock.
        with self._condition:
            try:
                job = self._pool.submit(stage)
            except RuntimeError:
                context.rollout.failure = "capture_submission_failed"
                self._slots.release()
                raise
            context.rollout.jobs.append(job)

    def abort(self, context: CaptureContext) -> None:
        with self._condition:
            context.rollout.failure = "incomplete_generation"
        self._slots.release()

    def finalize(self, key: str, *, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            rollout = self._rollouts[key]
            if rollout.receipt is not None:
                return rollout.receipt
            rollout.closing = True
            if not self._condition.wait_for(lambda: rollout.active == 0, timeout):
                raise TimeoutError("model requests are still active")
            jobs = list(rollout.jobs)
        manifest = []
        failure = rollout.failure
        for job in jobs:
            try:
                manifest.append(job.result(timeout=max(0, deadline - time.monotonic())))
            except TimeoutError:
                raise
            except Exception as error:
                failure = f"staging_failed:{type(error).__name__}"
        manifest.sort(key=lambda entry: entry["ordinal"])
        receipt = {
            "format": "nrl_direct_v1",
            "rollout_id": rollout.rollout_id,
            "manifest": manifest,
            "failure": failure or (None if manifest else "no_calls"),
        }
        with self._condition:
            rollout.receipt = receipt
            rollout.jobs.clear()
        return receipt

    def release(self, key: str) -> None:
        with self._condition:
            rollout = self._rollouts.get(key)
            if rollout is None:
                return
            if rollout.receipt is None:
                raise ValueError("finalize the rollout before releasing its API key")
            del self._rollouts[key]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def discard(self, key: str, *, timeout: float) -> None:
        """Drain and delete an abandoned rollout before releasing its identity."""
        receipt = self.finalize(key, timeout=timeout)
        self._clear([entry["staging_key"] for entry in receipt["manifest"]])
        self.release(key)


def select_main_branch(
    calls: list[NativeCall],
) -> tuple[list[NativeCall], list[NativeCall]]:
    """Latest admitted call and its longest exact-prefix predecessors.

    Matching token content is a training layout choice, not a claim about
    semantic parentage or vLLM's cache ancestry. Equal prefixes choose the
    latest prior call. All other calls are explicitly excluded in PR1.
    """
    if not calls:
        return [], []
    remaining = sorted(calls, key=lambda call: call.ordinal)
    branch = [remaining.pop()]
    while remaining:
        child = branch[-1]
        candidates = [
            call
            for call in remaining
            if len(call.prompt) + len(call.tokens) <= len(child.prompt)
            and child.prompt[: len(call.prompt)] == call.prompt
            and child.prompt[len(call.prompt) : len(call.prompt) + len(call.tokens)]
            == call.tokens
        ]
        if not candidates:
            break
        parent = max(
            candidates,
            key=lambda call: (len(call.prompt) + len(call.tokens), call.ordinal),
        )
        branch.append(parent)
        remaining = [call for call in remaining if call.ordinal < parent.ordinal]
    branch.reverse()
    selected = {call.call_id for call in branch}
    return branch, [call for call in calls if call.call_id not in selected]
