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

"""Validate and time final decode on an existing vLLM input stream.

The production background-prefill path cancels its resumable request and sends
the authoritative model call as a separate request. Consequently only complete
APC pages survive. This benchmark tests whether appending the authoritative
tail and decoding on the same resumable request can safely retain a partial
page without changing the generated result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
import uuid
from collections import Counter
from collections.abc import AsyncGenerator, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from examples.swe_bench.benchmark_background_prefill_admission import (
    _warm_authoritative_prefix,
)
from examples.swe_bench.benchmark_streaming_tool_call_prefill import (
    PromptTrace,
    _build_prompt_trace,
    _make_prefill_manager,
    _summary,
)


@dataclass(frozen=True)
class FinalGenerationMeasurement:
    """One final generation, including exact output-parity evidence."""

    output_kind: str
    cached_tokens: int
    prompt_tokens: int
    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...]
    output_text: str
    decoded_text: str
    time_to_first_output_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SeparateFinalMeasurement:
    """Current page-gated prefill followed by a separate final request."""

    scheduled_chunks: int
    scheduled_tokens: int
    completed_chunks: int
    completed_tokens: int
    cancelled_chunks: int
    failed_chunks: int
    committed_tokens: int
    prefix_matched: bool
    enqueue_seconds: float
    close_seconds: float
    final_generation: FinalGenerationMeasurement


@dataclass(frozen=True)
class SameRequestFinalMeasurement:
    """Prefill dummy decode followed by authoritative decode on one request."""

    streamed_prefix_tokens: int
    final_suffix_tokens: int
    prefill_chunks: int
    dummy_token_ids: tuple[int, ...]
    dummy_logprobs: tuple[float, ...]
    dummy_text: str
    prefill_cached_tokens: int | None
    prefill_to_dummy_seconds: float | None
    prefill_output_suppressed: bool
    final_generation: FinalGenerationMeasurement


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def _parse_nonnegative_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be non-negative floats")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=131_072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8_480)
    parser.add_argument(
        "--mamba-cache-mode",
        choices=("all", "align"),
        default="align",
    )
    parser.add_argument(
        "--compilation-backend",
        choices=("eager", "inductor"),
        default="eager",
    )
    parser.add_argument("--immutable-prefix-tokens", type=int, default=32_768)
    parser.add_argument("--tool-output-tokens", type=int, default=4_096)
    parser.add_argument("--mutable-suffix-tokens", type=int, default=32)
    parser.add_argument(
        "--candidate-chunk-token-sweep",
        type=_parse_positive_ints,
        default=(256, 512, 1_024, 1_152),
    )
    parser.add_argument("--stability-margin-tokens", type=int, default=8)
    parser.add_argument("--cache-page-size-tokens", type=int, default=1_056)
    parser.add_argument("--background-priority", type=int, default=1)
    parser.add_argument("--background-overlap-seconds", type=float, default=0.1)
    parser.add_argument("--cleanup-delay-seconds", type=float, default=0.05)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--same-request-prefill-chunks", type=int, default=1)
    parser.add_argument(
        "--same-request-final-before-prefill-ack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Queue and promote the authoritative final chunk immediately after "
            "the background chunk is submitted, without waiting for its dummy."
        ),
    )
    parser.add_argument(
        "--same-request-final-delay-sweep",
        type=_parse_nonnegative_floats,
        default=(),
        help=(
            "Optional tool-overlap delays to measure before promoting the final "
            "chunk; intended for the one-chunk in-flight path."
        ),
    )
    parser.add_argument("--concurrent-same-request-sessions", type=int, default=0)
    parser.add_argument(
        "--prefill-logprobs",
        choices=("none", "zero"),
        default="none",
        help=(
            "Use no sampled-token logprobs for production background prefill, "
            "or request zero alternatives for transition diagnosis."
        ),
    )
    parser.add_argument(
        "--diagnose-nan-logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask vLLM to count model-logit NaNs for each generated token.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--strict-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _extract_completion_delta(
    request_output: Any,
    *,
    require_logprobs: bool = True,
) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    """Extract sampled token IDs/logprobs and emitted text from one output."""
    if len(request_output.outputs) != 1:
        raise RuntimeError(
            "final-decode benchmark requires exactly one completion output"
        )
    completion = request_output.outputs[0]
    token_ids = tuple(int(token_id) for token_id in completion.token_ids)
    if not token_ids:
        return (), (), str(completion.text or "")
    if not require_logprobs and completion.logprobs is None:
        return token_ids, (), str(completion.text or "")
    if completion.logprobs is None or len(completion.logprobs) != len(token_ids):
        raise RuntimeError("vLLM did not return one logprob entry per output token")

    sampled_logprobs = []
    for token_id, token_logprobs in zip(token_ids, completion.logprobs, strict=True):
        if token_logprobs is None or token_id not in token_logprobs:
            raise RuntimeError(f"vLLM logprobs omitted sampled output token {token_id}")
        sampled_logprob = float(token_logprobs[token_id].logprob)
        if not math.isfinite(sampled_logprob):
            raise RuntimeError(
                "vLLM returned a non-finite sampled-token logprob "
                f"for token {token_id}: {sampled_logprob}"
            )
        sampled_logprobs.append(sampled_logprob)
    return token_ids, tuple(sampled_logprobs), str(completion.text or "")


def _classify_streaming_output(
    request_output: Any,
    *,
    prefill_prompt_boundaries: set[int],
    final_prompt_tokens: int,
) -> str:
    """Classify output by the authoritative prompt boundary that produced it.

    A final chunk can reach vLLM while the prior one-token prefill decode is
    still completing. NeMo-RL suppresses that discarded transition dummy, so
    the first externally visible token can already belong to the final prompt.
    Counting outputs is therefore ambiguous; the accumulated prompt length is
    the engine-owned phase boundary used by the production manager as well.
    """
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if prompt_token_ids is None:
        raise RuntimeError("streaming output did not report prompt token IDs")
    prompt_tokens = len(prompt_token_ids)
    if prompt_tokens == final_prompt_tokens:
        return "final"
    if prompt_tokens in prefill_prompt_boundaries:
        return "prefill"
    raise RuntimeError(
        "streaming output reported an unexpected prompt boundary: "
        f"observed={prompt_tokens}, "
        f"prefill={sorted(prefill_prompt_boundaries)}, final={final_prompt_tokens}"
    )


def _optional_summary(values: Sequence[float | None]) -> dict[str, float] | None:
    observed = [value for value in values if value is not None]
    return _summary(observed) if observed else None


def _max_abs_logprob_delta(
    reference: FinalGenerationMeasurement, candidate: FinalGenerationMeasurement
) -> float:
    """Return sampled-token logprob drift, or infinity for incompatible outputs."""
    if len(reference.output_logprobs) != len(candidate.output_logprobs):
        return float("inf")
    return max(
        (
            abs(reference_logprob - candidate_logprob)
            for reference_logprob, candidate_logprob in zip(
                reference.output_logprobs,
                candidate.output_logprobs,
                strict=True,
            )
        ),
        default=0.0,
    )


def _parity(
    reference: FinalGenerationMeasurement,
    candidate: FinalGenerationMeasurement,
    *,
    logprob_atol: float,
) -> dict[str, bool | float]:
    """Return output parity relative to normal-request logprob drift."""
    max_abs_logprob_delta = _max_abs_logprob_delta(reference, candidate)
    details = {
        "output_token_ids": reference.output_token_ids == candidate.output_token_ids,
        "output_logprobs_exact": (
            reference.output_logprobs == candidate.output_logprobs
        ),
        "output_logprobs_within_atol": max_abs_logprob_delta <= logprob_atol,
        "max_abs_logprob_delta": max_abs_logprob_delta,
        "logprob_atol": logprob_atol,
        "output_text": reference.output_text == candidate.output_text,
        "decoded_text": reference.decoded_text == candidate.decoded_text,
    }
    details["all"] = bool(
        details["output_token_ids"]
        and details["output_logprobs_within_atol"]
        and details["output_text"]
        and details["decoded_text"]
    )
    return details


def _output_signature(
    measurement: FinalGenerationMeasurement,
) -> tuple[tuple[int, ...], str, str]:
    return (
        measurement.output_token_ids,
        measurement.output_text,
        measurement.decoded_text,
    )


def _output_histogram(
    measurements: Sequence[FinalGenerationMeasurement],
) -> list[dict[str, Any]]:
    counts = Counter(_output_signature(measurement) for measurement in measurements)
    return [
        {
            "count": count,
            "output_token_ids": signature[0],
            "output_text": signature[1],
            "decoded_text": signature[2],
        }
        for signature, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _modal_control_contract(
    controls: Sequence[FinalGenerationMeasurement],
    candidate: FinalGenerationMeasurement,
) -> dict[str, Any]:
    """Compare a candidate with the unique modal output of normal requests."""
    if len(controls) < 3:
        raise ValueError("modal output validation requires at least three controls")

    signature_counts = Counter(_output_signature(control) for control in controls)
    max_count = max(signature_counts.values())
    modal_signatures = [
        signature for signature, count in signature_counts.items() if count == max_count
    ]
    modal_available = max_count >= 2 and len(modal_signatures) == 1
    if not modal_available:
        return {
            "modal_available": False,
            "modal_count": max_count,
            "control_count": len(controls),
            "candidate_matches_modal": False,
            "normal_modal_logprob_drift": float("inf"),
            "candidate_modal_logprob_drift": float("inf"),
            "logprobs_within_modal_control_drift": False,
            "all": False,
        }

    modal_signature = modal_signatures[0]
    modal_controls = [
        control for control in controls if _output_signature(control) == modal_signature
    ]
    normal_modal_logprob_drift = max(
        (
            _max_abs_logprob_delta(left, right)
            for left, right in combinations(modal_controls, 2)
        ),
        default=0.0,
    )
    candidate_matches_modal = _output_signature(candidate) == modal_signature
    candidate_modal_logprob_drift = (
        min(_max_abs_logprob_delta(control, candidate) for control in modal_controls)
        if candidate_matches_modal
        else float("inf")
    )
    logprob_atol = max(normal_modal_logprob_drift, 1e-6)
    logprobs_within_modal_control_drift = candidate_modal_logprob_drift <= logprob_atol
    return {
        "modal_available": True,
        "modal_count": max_count,
        "control_count": len(controls),
        "candidate_matches_modal": candidate_matches_modal,
        "normal_modal_logprob_drift": normal_modal_logprob_drift,
        "candidate_modal_logprob_drift": candidate_modal_logprob_drift,
        "logprob_atol": logprob_atol,
        "logprobs_within_modal_control_drift": (logprobs_within_modal_control_drift),
        # Token IDs and decoded/emitted text are the semantic contract. GPU
        # reductions make sampled-token logprobs non-bitwise-repeatable even
        # across ordinary identical requests, so their envelope is diagnostic.
        "all": candidate_matches_modal,
    }


async def _generate_final(
    llm: Any,
    *,
    prompt_token_ids: list[int],
    sampling_params: Any,
    tokens_prompt_type: type,
    tokenizer: Any,
    output_kind: str,
) -> FinalGenerationMeasurement:
    request_id = f"streaming-final-decode-control-{uuid.uuid4()}"
    started_at = time.perf_counter()
    first_output_at: float | None = None
    cached_tokens: int | None = None
    output_token_ids: list[int] = []
    output_logprobs: list[float] = []
    output_text: list[str] = []

    result_generator = llm.generate(
        prompt=tokens_prompt_type(prompt_token_ids=prompt_token_ids),
        sampling_params=sampling_params,
        request_id=request_id,
    )
    async for request_output in result_generator:
        if request_output.num_cached_tokens is not None:
            cached_tokens = int(request_output.num_cached_tokens)
        token_ids, logprobs, text = _extract_completion_delta(request_output)
        if token_ids and first_output_at is None:
            first_output_at = time.perf_counter()
        output_token_ids.extend(token_ids)
        output_logprobs.extend(logprobs)
        output_text.append(text)

    finished_at = time.perf_counter()
    if cached_tokens is None:
        raise RuntimeError("vLLM did not report num_cached_tokens")
    if first_output_at is None or not output_token_ids:
        raise RuntimeError("vLLM returned no final output tokens")
    return FinalGenerationMeasurement(
        output_kind=output_kind,
        cached_tokens=cached_tokens,
        prompt_tokens=len(prompt_token_ids),
        output_token_ids=tuple(output_token_ids),
        output_logprobs=tuple(output_logprobs),
        output_text="".join(output_text),
        decoded_text=tokenizer.decode(output_token_ids),
        time_to_first_output_seconds=first_output_at - started_at,
        total_seconds=finished_at - started_at,
    )


def _stable_streamed_prefix(
    trace: PromptTrace, *, stability_margin_tokens: int
) -> list[int]:
    first_snapshot = trace.snapshots[0]
    stable_candidate_end = len(first_snapshot) - trace.mutable_suffix_tokens
    stable_prefix_end = stable_candidate_end - stability_margin_tokens
    if stable_prefix_end <= trace.immutable_prefix_tokens:
        raise ValueError("candidate chunk is not larger than the stability margin")
    stable_prefix = first_snapshot[:stable_prefix_end]
    if trace.final[:stable_prefix_end] != stable_prefix:
        raise RuntimeError("candidate's stable prefix does not match the final prompt")
    return stable_prefix


async def _generate_same_request_final(
    llm: Any,
    *,
    trace: PromptTrace,
    prefill_sampling_params: Any,
    final_sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    tokenizer: Any,
    stability_margin_tokens: int,
    background_priority: int,
    prefill_has_logprobs: bool,
    prefill_chunk_count: int,
    final_before_prefill_ack: bool,
    final_delay_seconds: float = 0.0,
) -> SameRequestFinalMeasurement:
    stable_prefix = _stable_streamed_prefix(
        trace, stability_margin_tokens=stability_margin_tokens
    )
    streamed_prefix_token_count = len(stable_prefix)
    dynamic_prefix = stable_prefix[trace.immutable_prefix_tokens :]
    if not 1 <= prefill_chunk_count <= len(dynamic_prefix):
        raise ValueError(
            "same-request prefill chunks must be between one and the stable "
            f"dynamic-prefix length ({len(dynamic_prefix)})"
        )
    if final_before_prefill_ack and prefill_chunk_count != 1:
        raise ValueError(
            "final-before-prefill-ack currently requires exactly one prefill chunk"
        )
    if final_delay_seconds < 0:
        raise ValueError("final_delay_seconds must be non-negative")
    if final_delay_seconds and not final_before_prefill_ack:
        raise ValueError("final delay requires final-before-prefill-ack mode")
    dynamic_chunk_size, larger_chunk_count = divmod(
        len(dynamic_prefix), prefill_chunk_count
    )
    prefill_chunks: list[list[int]] = []
    dynamic_offset = 0
    for chunk_index in range(prefill_chunk_count):
        chunk_size = dynamic_chunk_size + (chunk_index < larger_chunk_count)
        next_dynamic_offset = dynamic_offset + chunk_size
        if chunk_index == 0:
            prefill_chunks.append(
                stable_prefix[: trace.immutable_prefix_tokens + next_dynamic_offset]
            )
        else:
            prefill_chunks.append(dynamic_prefix[dynamic_offset:next_dynamic_offset])
        dynamic_offset = next_dynamic_offset
    assert sum(len(chunk) for chunk in prefill_chunks) == len(stable_prefix)
    prefill_prompt_boundaries: set[int] = set()
    accumulated_prefill_tokens = 0
    for prefill_chunk in prefill_chunks:
        accumulated_prefill_tokens += len(prefill_chunk)
        prefill_prompt_boundaries.add(accumulated_prefill_tokens)
    final_suffix = trace.final[len(stable_prefix) :]
    if not final_suffix:
        raise RuntimeError(
            "same-request final decode requires a non-empty final suffix"
        )

    prefill_chunk_observed = asyncio.Event()
    input_started_at = time.perf_counter()
    final_enqueued_at: float | None = None

    async def input_stream() -> AsyncGenerator[Any, None]:
        nonlocal final_enqueued_at
        for prefill_chunk in prefill_chunks:
            yield streaming_input_type(
                prompt=tokens_prompt_type(prompt_token_ids=prefill_chunk),
                sampling_params=prefill_sampling_params,
            )
            if not final_before_prefill_ack:
                await prefill_chunk_observed.wait()
                prefill_chunk_observed.clear()
            elif final_delay_seconds:
                await asyncio.sleep(final_delay_seconds)
        final_enqueued_at = time.perf_counter()
        yield streaming_input_type(
            prompt=tokens_prompt_type(prompt_token_ids=final_suffix),
            sampling_params=final_sampling_params,
            priority=0,
        )

    request_id = f"streaming-final-decode-{uuid.uuid4()}"
    result_generator = llm.generate(
        prompt=input_stream(),
        sampling_params=final_sampling_params,
        request_id=request_id,
        priority=background_priority,
    )

    dummy_token_ids: list[int] = []
    dummy_logprobs: list[float] = []
    dummy_text: list[str] = []
    prefill_cached_tokens: int | None = None
    dummy_observed_at: float | None = None
    final_cached_tokens: int | None = None
    final_first_output_at: float | None = None
    final_output_token_ids: list[int] = []
    final_output_logprobs: list[float] = []
    final_output_text: list[str] = []

    try:
        async for request_output in result_generator:
            output_phase = _classify_streaming_output(
                request_output,
                prefill_prompt_boundaries=prefill_prompt_boundaries,
                final_prompt_tokens=len(trace.final),
            )
            token_ids, logprobs, text = _extract_completion_delta(
                request_output,
                require_logprobs=(output_phase == "final" or prefill_has_logprobs),
            )
            if output_phase == "prefill":
                if not token_ids:
                    continue
                if len(dummy_token_ids) >= prefill_chunk_count:
                    raise RuntimeError("prefill phase produced an extra dummy token")
                if len(token_ids) != 1:
                    raise RuntimeError(
                        "prefill phase generated more than one dummy token"
                    )
                dummy_token_ids.extend(token_ids)
                dummy_logprobs.extend(logprobs)
                dummy_text.append(text)
                if request_output.num_cached_tokens is not None:
                    prefill_cached_tokens = int(request_output.num_cached_tokens)
                dummy_observed_at = time.perf_counter()
                prefill_chunk_observed.set()
                continue

            if request_output.num_cached_tokens is not None:
                final_cached_tokens = int(request_output.num_cached_tokens)
            if token_ids and final_first_output_at is None:
                final_first_output_at = time.perf_counter()
            final_output_token_ids.extend(token_ids)
            final_output_logprobs.extend(logprobs)
            final_output_text.append(text)
    finally:
        # Prevent the AsyncLLM input task from waiting forever if output
        # validation raises before the prefill dummy is observed.
        prefill_chunk_observed.set()

    finished_at = time.perf_counter()
    prefill_output_suppressed = not dummy_token_ids
    if not prefill_output_suppressed and len(dummy_token_ids) != prefill_chunk_count:
        raise RuntimeError(
            "same-request prefill did not produce exactly one dummy token per "
            f"chunk: {len(dummy_token_ids)} != {prefill_chunk_count}"
        )
    if prefill_output_suppressed and not final_before_prefill_ack:
        raise RuntimeError(
            "same-request prefill output was suppressed without an enqueued final chunk"
        )
    if final_enqueued_at is None:
        raise RuntimeError("same-request final suffix was never enqueued")
    if final_cached_tokens is None:
        raise RuntimeError("same-request final decode reported no cache count")
    if final_first_output_at is None or not final_output_token_ids:
        raise RuntimeError("same-request final decode produced no output")

    return SameRequestFinalMeasurement(
        # vLLM extends the input list in place as later streaming chunks arrive.
        # Report the pre-mutation length captured above.
        streamed_prefix_tokens=streamed_prefix_token_count,
        final_suffix_tokens=len(final_suffix),
        prefill_chunks=prefill_chunk_count,
        dummy_token_ids=tuple(dummy_token_ids),
        dummy_logprobs=tuple(dummy_logprobs),
        dummy_text="".join(dummy_text),
        prefill_cached_tokens=prefill_cached_tokens,
        prefill_to_dummy_seconds=(
            None if dummy_observed_at is None else dummy_observed_at - input_started_at
        ),
        prefill_output_suppressed=prefill_output_suppressed,
        final_generation=FinalGenerationMeasurement(
            output_kind="delta",
            cached_tokens=final_cached_tokens,
            prompt_tokens=len(trace.final),
            output_token_ids=tuple(final_output_token_ids),
            output_logprobs=tuple(final_output_logprobs),
            output_text="".join(final_output_text),
            decoded_text=tokenizer.decode(final_output_token_ids),
            time_to_first_output_seconds=(final_first_output_at - final_enqueued_at),
            total_seconds=finished_at - final_enqueued_at,
        ),
    )


async def _run_separate_final(
    llm: Any,
    *,
    trace: PromptTrace,
    prefill_sampling_params: Any,
    final_sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    tokenizer: Any,
    stability_margin_tokens: int,
    background_priority: int,
    background_overlap_seconds: float,
    cache_page_size_tokens: int,
) -> SeparateFinalMeasurement:
    manager = _make_prefill_manager(
        llm,
        sampling_params=prefill_sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(trace.final),
        priority=background_priority,
        cache_page_size_tokens=cache_page_size_tokens,
    )
    first_snapshot = trace.snapshots[0]
    stable_candidate_end = len(first_snapshot) - trace.mutable_suffix_tokens
    session_id = f"streaming-separate-final-{uuid.uuid4()}"
    start_result = await manager.start_background(
        session_id=session_id,
        prompt_token_ids=first_snapshot,
        initial_candidate_token_ids=first_snapshot[:stable_candidate_end],
        dynamic_token_baseline=trace.immutable_prefix_tokens,
    )
    if background_overlap_seconds:
        await asyncio.sleep(background_overlap_seconds)
    close_started_at = time.perf_counter()
    close_result = await manager.close(
        session_id=session_id,
        final_prompt_token_ids=trace.final,
    )
    close_seconds = time.perf_counter() - close_started_at
    final_generation = await _generate_final(
        llm,
        prompt_token_ids=trace.final,
        sampling_params=final_sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        tokenizer=tokenizer,
        output_kind="delta",
    )
    return SeparateFinalMeasurement(
        scheduled_chunks=start_result.scheduled_chunks,
        scheduled_tokens=start_result.scheduled_tokens,
        completed_chunks=close_result.background_completed_chunks,
        completed_tokens=close_result.background_completed_tokens,
        cancelled_chunks=close_result.background_cancelled_chunks,
        failed_chunks=close_result.background_failed_chunks,
        committed_tokens=close_result.committed_tokens,
        prefix_matched=close_result.prefix_matched,
        enqueue_seconds=start_result.enqueue_seconds,
        close_seconds=close_seconds,
        final_generation=final_generation,
    )


def _paired_summary(
    controls: Sequence[FinalGenerationMeasurement],
    candidates: Sequence[FinalGenerationMeasurement],
) -> dict[str, Any]:
    if len(controls) != len(candidates):
        raise ValueError("control and candidate measurements must be paired")
    ttft_savings = [
        control.time_to_first_output_seconds - candidate.time_to_first_output_seconds
        for control, candidate in zip(controls, candidates, strict=True)
    ]
    total_savings = [
        control.total_seconds - candidate.total_seconds
        for control, candidate in zip(controls, candidates, strict=True)
    ]
    return {
        "control_time_to_first_output_seconds": _summary(
            [item.time_to_first_output_seconds for item in controls]
        ),
        "candidate_time_to_first_output_seconds": _summary(
            [item.time_to_first_output_seconds for item in candidates]
        ),
        "paired_time_to_first_output_savings_seconds": _summary(ttft_savings),
        "paired_time_to_first_output_win_fraction": statistics.fmean(
            saving > 0 for saving in ttft_savings
        ),
        "control_total_seconds": _summary([item.total_seconds for item in controls]),
        "candidate_total_seconds": _summary(
            [item.total_seconds for item in candidates]
        ),
        "paired_total_savings_seconds": _summary(total_savings),
        "paired_total_win_fraction": statistics.fmean(
            saving > 0 for saving in total_savings
        ),
        "control_cached_tokens": _summary(
            [float(item.cached_tokens) for item in controls]
        ),
        "candidate_cached_tokens": _summary(
            [float(item.cached_tokens) for item in candidates]
        ),
    }


async def _main(args: argparse.Namespace) -> None:
    if args.diagnose_nan_logits:
        os.environ["VLLM_COMPUTE_NANS_IN_LOGITS"] = "1"

    from transformers import AutoTokenizer
    from vllm.logger import init_logger

    from nemo_rl.models.generation.vllm.patches import (
        _patch_vllm_streaming_session_max_tokens,
        _patch_vllm_streaming_session_output_state,
        _patch_vllm_streaming_session_priority,
        _patch_vllm_strict_priority_scheduling,
    )

    patch_logger = init_logger("streaming_final_decode_patch")
    if not _patch_vllm_streaming_session_max_tokens(patch_logger):
        raise RuntimeError(
            "installed vLLM does not support per-chunk max_tokens updates"
        )
    if not _patch_vllm_streaming_session_output_state(patch_logger):
        raise RuntimeError(
            "installed vLLM does not support streaming output-state updates"
        )
    if not _patch_vllm_streaming_session_priority(patch_logger):
        raise RuntimeError(
            "installed vLLM does not support streaming priority promotion"
        )
    if not _patch_vllm_strict_priority_scheduling(patch_logger):
        raise RuntimeError("installed vLLM does not support strict background priority")

    from vllm import TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    if args.repeats <= 0 or args.warmup_repeats < 0:
        raise ValueError("repeats must be positive and warmup_repeats non-negative")
    if args.same_request_prefill_chunks <= 0:
        raise ValueError("same_request_prefill_chunks must be positive")
    if args.concurrent_same_request_sessions < 0:
        raise ValueError("concurrent_same_request_sessions must be non-negative")
    for name in (
        "max_output_tokens",
        "cache_page_size_tokens",
        "max_num_batched_tokens",
        "tensor_parallel_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.background_priority <= 0:
        raise ValueError("background_priority must be positive")
    if args.background_overlap_seconds < 0 or args.cleanup_delay_seconds < 0:
        raise ValueError("overlap and cleanup delays must be non-negative")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    traces: dict[int, PromptTrace] = {}
    common_final: list[int] | None = None
    for candidate_tokens in args.candidate_chunk_token_sweep:
        trace_args = argparse.Namespace(**vars(args))
        trace_args.candidate_chunk_tokens = candidate_tokens
        trace = _build_prompt_trace(tokenizer, trace_args)
        if common_final is None:
            common_final = trace.final
        elif trace.final != common_final:
            raise RuntimeError("candidate sweep changed the authoritative final prompt")
        if len(trace.final) + args.max_output_tokens > args.max_model_len:
            raise ValueError("final prompt and output exceed max_model_len")
        _stable_streamed_prefix(
            trace, stability_margin_tokens=args.stability_margin_tokens
        )
        traces[candidate_tokens] = trace

    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_prefix_caching=True,
            enforce_eager=args.enforce_eager,
            trust_remote_code=True,
            scheduling_policy="priority",
            async_scheduling=args.async_scheduling,
            mamba_cache_mode=args.mamba_cache_mode,
            compilation_config={
                "backend": args.compilation_backend,
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
            },
        )
    )
    nan_logits_by_request: dict[str, int] = {}
    if args.diagnose_nan_logits:
        original_process_outputs = llm.output_processor.process_outputs

        def process_outputs_with_nan_diagnostics(
            engine_core_outputs: list[Any], *process_args: Any, **process_kwargs: Any
        ) -> Any:
            for engine_core_output in engine_core_outputs:
                if engine_core_output.num_nans_in_logits:
                    nan_logits_by_request[engine_core_output.request_id] = (
                        nan_logits_by_request.get(engine_core_output.request_id, 0)
                        + int(engine_core_output.num_nans_in_logits)
                    )
                    print(
                        "MODEL_LOGITS_NAN "
                        f"request_id={engine_core_output.request_id} "
                        f"count={engine_core_output.num_nans_in_logits}",
                        flush=True,
                    )
            return original_process_outputs(
                engine_core_outputs, *process_args, **process_kwargs
            )

        llm.output_processor.process_outputs = process_outputs_with_nan_diagnostics

    prefill_sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        seed=args.seed,
        max_tokens=1,
        logprobs=0 if args.prefill_logprobs == "zero" else None,
        output_kind=RequestOutputKind.DELTA,
    )
    final_delta_sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        seed=args.seed,
        max_tokens=args.max_output_tokens,
        logprobs=0,
        output_kind=RequestOutputKind.DELTA,
    )
    final_only_sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        seed=args.seed,
        max_tokens=args.max_output_tokens,
        logprobs=0,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    async def reset_cache() -> None:
        deadline = time.perf_counter() + 5.0
        while not await llm.reset_prefix_cache():
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "vLLM prefix cache remained busy for more than 5 seconds"
                )
            await asyncio.sleep(0.05)

    async def prepare_arm(trace: PromptTrace) -> None:
        await reset_cache()
        await _warm_authoritative_prefix(
            llm,
            trace=trace,
            sampling_params=prefill_sampling_params,
            tokens_prompt_type=TokensPrompt,
            streaming_input_type=StreamingInput,
            stability_margin_tokens=args.stability_margin_tokens,
        )
        if args.cleanup_delay_seconds:
            await asyncio.sleep(args.cleanup_delay_seconds)

    raw: dict[str, dict[str, list[Any]]] = {}
    reference_parity_observations: list[dict[str, Any]] = []
    modal_contract_observations: list[dict[str, Any]] = []
    modal_contract_failures: list[dict[str, Any]] = []
    try:
        for candidate_tokens, trace in traces.items():
            label = str(candidate_tokens)
            raw[label] = {
                "control_delta": [],
                "control_final_only": [],
                "separate_final": [],
                "same_request_final": [],
            }
            total_iterations = args.warmup_repeats + args.repeats
            for iteration in range(total_iterations):
                keep = iteration >= args.warmup_repeats
                arm_order = (
                    "control_delta",
                    "control_final_only",
                    "separate_final",
                    "same_request_final",
                )
                if iteration % 2:
                    arm_order = tuple(reversed(arm_order))

                iteration_results: dict[str, Any] = {}
                for arm in arm_order:
                    await prepare_arm(trace)
                    if arm == "control_delta":
                        measurement = await _generate_final(
                            llm,
                            prompt_token_ids=trace.final,
                            sampling_params=final_delta_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            tokenizer=tokenizer,
                            output_kind="delta",
                        )
                    elif arm == "control_final_only":
                        measurement = await _generate_final(
                            llm,
                            prompt_token_ids=trace.final,
                            sampling_params=final_only_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            tokenizer=tokenizer,
                            output_kind="final_only",
                        )
                    elif arm == "separate_final":
                        measurement = await _run_separate_final(
                            llm,
                            trace=trace,
                            prefill_sampling_params=prefill_sampling_params,
                            final_sampling_params=final_delta_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            streaming_input_type=StreamingInput,
                            tokenizer=tokenizer,
                            stability_margin_tokens=args.stability_margin_tokens,
                            background_priority=args.background_priority,
                            background_overlap_seconds=(
                                args.background_overlap_seconds
                            ),
                            cache_page_size_tokens=args.cache_page_size_tokens,
                        )
                    else:
                        measurement = await _generate_same_request_final(
                            llm,
                            trace=trace,
                            prefill_sampling_params=prefill_sampling_params,
                            final_sampling_params=final_delta_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            streaming_input_type=StreamingInput,
                            tokenizer=tokenizer,
                            stability_margin_tokens=args.stability_margin_tokens,
                            background_priority=args.background_priority,
                            prefill_has_logprobs=(args.prefill_logprobs == "zero"),
                            prefill_chunk_count=args.same_request_prefill_chunks,
                            final_before_prefill_ack=(
                                args.same_request_final_before_prefill_ack
                            ),
                        )
                    iteration_results[arm] = measurement

                if not keep:
                    continue
                for arm, measurement in iteration_results.items():
                    raw[label][arm].append(measurement)

                reference = iteration_results["control_delta"]
                final_only_control = iteration_results["control_final_only"]
                separate_control = iteration_results["separate_final"].final_generation
                normal_control_logprob_drift = max(
                    _max_abs_logprob_delta(reference, final_only_control),
                    _max_abs_logprob_delta(reference, separate_control),
                )
                parity_candidates = {
                    "control_final_only": final_only_control,
                    "separate_final": separate_control,
                    "same_request_final": iteration_results[
                        "same_request_final"
                    ].final_generation,
                }
                for arm, candidate in parity_candidates.items():
                    details = _parity(
                        reference,
                        candidate,
                        logprob_atol=normal_control_logprob_drift,
                    )
                    observation = {
                        "candidate_chunk_tokens": candidate_tokens,
                        "repeat": iteration - args.warmup_repeats,
                        "arm": arm,
                        "details": details,
                    }
                    reference_parity_observations.append(observation)

                modal_contract = {
                    "candidate_chunk_tokens": candidate_tokens,
                    "repeat": iteration - args.warmup_repeats,
                    "details": _modal_control_contract(
                        (reference, final_only_control, separate_control),
                        iteration_results["same_request_final"].final_generation,
                    ),
                }
                modal_contract_observations.append(modal_contract)
                if not modal_contract["details"]["all"]:
                    modal_contract_failures.append(modal_contract)

        concurrent_control_measurements: list[FinalGenerationMeasurement] = []
        concurrent_same_request_measurements: list[SameRequestFinalMeasurement] = []
        if args.concurrent_same_request_sessions:
            await reset_cache()
            diagnostic_trace = next(iter(traces.values()))
            concurrent_same_request_measurements = list(
                await asyncio.gather(
                    *(
                        _generate_same_request_final(
                            llm,
                            trace=diagnostic_trace,
                            prefill_sampling_params=prefill_sampling_params,
                            final_sampling_params=final_delta_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            streaming_input_type=StreamingInput,
                            tokenizer=tokenizer,
                            stability_margin_tokens=args.stability_margin_tokens,
                            background_priority=args.background_priority,
                            prefill_has_logprobs=(args.prefill_logprobs == "zero"),
                            prefill_chunk_count=args.same_request_prefill_chunks,
                            final_before_prefill_ack=(
                                args.same_request_final_before_prefill_ack
                            ),
                        )
                        for _ in range(args.concurrent_same_request_sessions)
                    )
                )
            )
            await reset_cache()
            concurrent_control_measurements = list(
                await asyncio.gather(
                    *(
                        _generate_final(
                            llm,
                            prompt_token_ids=list(diagnostic_trace.final),
                            sampling_params=final_delta_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            tokenizer=tokenizer,
                            output_kind="delta",
                        )
                        for _ in range(args.concurrent_same_request_sessions)
                    )
                )
            )

        concurrent_control_signatures = {
            _output_signature(measurement)
            for measurement in concurrent_control_measurements
        }
        concurrent_same_request_matches_control = [
            _output_signature(measurement.final_generation)
            in concurrent_control_signatures
            for measurement in concurrent_same_request_measurements
        ]

        final_delay_sweep: dict[str, Any] = {}
        if args.same_request_final_delay_sweep:
            delay_trace = next(iter(traces.values()))
            for delay_seconds in args.same_request_final_delay_sweep:
                delay_controls: list[FinalGenerationMeasurement] = []
                delay_candidates: list[SameRequestFinalMeasurement] = []
                total_iterations = args.warmup_repeats + args.repeats
                for iteration in range(total_iterations):
                    arm_order = ("control", "candidate")
                    if iteration % 2:
                        arm_order = tuple(reversed(arm_order))
                    iteration_results: dict[str, Any] = {}
                    for arm in arm_order:
                        await prepare_arm(delay_trace)
                        if arm == "control":
                            measurement = await _generate_final(
                                llm,
                                prompt_token_ids=delay_trace.final,
                                sampling_params=final_delta_sampling_params,
                                tokens_prompt_type=TokensPrompt,
                                tokenizer=tokenizer,
                                output_kind="delta",
                            )
                        else:
                            measurement = await _generate_same_request_final(
                                llm,
                                trace=delay_trace,
                                prefill_sampling_params=prefill_sampling_params,
                                final_sampling_params=final_delta_sampling_params,
                                tokens_prompt_type=TokensPrompt,
                                streaming_input_type=StreamingInput,
                                tokenizer=tokenizer,
                                stability_margin_tokens=(args.stability_margin_tokens),
                                background_priority=args.background_priority,
                                prefill_has_logprobs=(args.prefill_logprobs == "zero"),
                                prefill_chunk_count=1,
                                final_before_prefill_ack=True,
                                final_delay_seconds=delay_seconds,
                            )
                        iteration_results[arm] = measurement
                    if iteration < args.warmup_repeats:
                        continue
                    delay_controls.append(iteration_results["control"])
                    delay_candidates.append(iteration_results["candidate"])

                candidate_generations = [
                    item.final_generation for item in delay_candidates
                ]
                output_matches = [
                    _output_signature(control) == _output_signature(candidate)
                    for control, candidate in zip(
                        delay_controls, candidate_generations, strict=True
                    )
                ]
                final_delay_sweep[f"{delay_seconds:g}"] = {
                    "delay_seconds": delay_seconds,
                    "summary": _paired_summary(delay_controls, candidate_generations),
                    "exact_token_and_text_match_fraction": statistics.fmean(
                        output_matches
                    ),
                    "prefill_to_dummy_seconds": _optional_summary(
                        [item.prefill_to_dummy_seconds for item in delay_candidates]
                    ),
                    "control": [asdict(item) for item in delay_controls],
                    "candidate": [asdict(item) for item in delay_candidates],
                }

        measurements: dict[str, Any] = {}
        subpage_ttft_positive = []
        subpage_total_positive = []
        dummy_isolated_from_full_final_output = True
        for candidate_tokens, trace in traces.items():
            label = str(candidate_tokens)
            control_delta = raw[label]["control_delta"]
            control_final_only = raw[label]["control_final_only"]
            separate_final = raw[label]["separate_final"]
            same_request_final = raw[label]["same_request_final"]
            separate_generations = [item.final_generation for item in separate_final]
            same_generations = [item.final_generation for item in same_request_final]
            separate_summary = _paired_summary(control_delta, separate_generations)
            same_summary = _paired_summary(control_delta, same_generations)
            stable_dynamic_tokens = (
                len(
                    _stable_streamed_prefix(
                        trace,
                        stability_margin_tokens=args.stability_margin_tokens,
                    )
                )
                - trace.immutable_prefix_tokens
            )
            is_subpage = stable_dynamic_tokens < args.cache_page_size_tokens
            if is_subpage:
                subpage_ttft_positive.append(
                    same_summary["paired_time_to_first_output_savings_seconds"]["mean"]
                    > 0
                )
                subpage_total_positive.append(
                    same_summary["paired_total_savings_seconds"]["mean"] > 0
                )
            dummy_isolated_from_full_final_output &= all(
                (
                    len(item.dummy_token_ids) == args.same_request_prefill_chunks
                    or (
                        args.same_request_final_before_prefill_ack
                        and item.prefill_output_suppressed
                    )
                )
                and len(item.final_generation.output_token_ids)
                == len(reference.output_token_ids)
                for item, reference in zip(
                    same_request_final, control_delta, strict=True
                )
            )
            measurements[label] = {
                "first_candidate_tokens": len(trace.snapshots[0]),
                "stable_dynamic_tokens": stable_dynamic_tokens,
                "is_subpage_candidate": is_subpage,
                "separate_final_summary": separate_summary,
                "same_request_final_summary": same_summary,
                "final_only_total_seconds": _summary(
                    [item.total_seconds for item in control_final_only]
                ),
                "separate_scheduled_fraction": statistics.fmean(
                    item.scheduled_chunks > 0 for item in separate_final
                ),
                "separate_completed_fraction": statistics.fmean(
                    item.completed_chunks > 0 for item in separate_final
                ),
                "same_request_prefill_to_dummy_seconds": _optional_summary(
                    [item.prefill_to_dummy_seconds for item in same_request_final]
                ),
                "same_request_streamed_prefix_tokens": _summary(
                    [float(item.streamed_prefix_tokens) for item in same_request_final]
                ),
                "raw": {
                    arm: [asdict(item) for item in arm_measurements]
                    for arm, arm_measurements in raw[label].items()
                },
            }

        same_reference_observations = [
            observation
            for observation in reference_parity_observations
            if observation["arm"] == "same_request_final"
        ]
        exact_reference_token_and_text_parity = all(
            observation["details"][field]
            for observation in same_reference_observations
            for field in ("output_token_ids", "output_text", "decoded_text")
        )
        exact_reference_logprob_parity = all(
            observation["details"]["output_logprobs_exact"]
            for observation in same_reference_observations
        )
        modal_output_parity = all(
            observation["details"]["candidate_matches_modal"]
            for observation in modal_contract_observations
        )
        logprobs_within_modal_control_drift = all(
            observation["details"]["logprobs_within_modal_control_drift"]
            for observation in modal_contract_observations
        )
        modal_contracts_passed = not modal_contract_failures
        output_contracts_passed = (
            modal_contracts_passed and dummy_isolated_from_full_final_output
        )
        result = {
            "model": args.model,
            "engine": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "async_scheduling": args.async_scheduling,
                "mamba_cache_mode": args.mamba_cache_mode,
                "compilation_backend": args.compilation_backend,
                "scheduling_policy": "priority",
                "background_priority": args.background_priority,
                "configured_cache_page_size_tokens": args.cache_page_size_tokens,
            },
            "trace": {
                "immutable_prefix_tokens": args.immutable_prefix_tokens,
                "tool_output_tokens": args.tool_output_tokens,
                "mutable_suffix_tokens": args.mutable_suffix_tokens,
                "stability_margin_tokens": args.stability_margin_tokens,
                "candidate_chunk_token_sweep": args.candidate_chunk_token_sweep,
                "same_request_prefill_chunks": args.same_request_prefill_chunks,
                "same_request_final_before_prefill_ack": (
                    args.same_request_final_before_prefill_ack
                ),
            },
            "generation": {
                "max_output_tokens": args.max_output_tokens,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "seed": args.seed,
                "sampled_token_logprobs": 0,
                "prefill_sampled_token_logprobs": args.prefill_logprobs,
            },
            "diagnostics": {
                "nan_logits_enabled": args.diagnose_nan_logits,
                "nan_logits_by_request": nan_logits_by_request,
                "concurrent_same_request_sessions": (
                    args.concurrent_same_request_sessions
                ),
                "concurrent_control_measurements": [
                    asdict(item) for item in concurrent_control_measurements
                ],
                "concurrent_same_request_measurements": [
                    asdict(item) for item in concurrent_same_request_measurements
                ],
                "concurrent_control_output_histogram": _output_histogram(
                    concurrent_control_measurements
                ),
                "concurrent_same_request_output_histogram": _output_histogram(
                    [
                        item.final_generation
                        for item in concurrent_same_request_measurements
                    ]
                ),
                "concurrent_same_request_matches_control_fraction": (
                    statistics.fmean(concurrent_same_request_matches_control)
                    if concurrent_same_request_matches_control
                    else None
                ),
                "same_request_final_delay_sweep": final_delay_sweep,
            },
            "repeats": args.repeats,
            "warmup_repeats": args.warmup_repeats,
            "acceptance": {
                "output_contracts_passed": output_contracts_passed,
                "modal_contracts_passed": modal_contracts_passed,
                "modal_output_parity": modal_output_parity,
                "logprobs_within_modal_control_drift": (
                    logprobs_within_modal_control_drift
                ),
                "dummy_isolated_from_full_final_output": (
                    dummy_isolated_from_full_final_output
                ),
                "exact_reference_token_and_text_parity": (
                    exact_reference_token_and_text_parity
                ),
                "exact_reference_logprob_parity": exact_reference_logprob_parity,
                "all_subpage_mean_ttft_savings_positive": bool(subpage_ttft_positive)
                and all(subpage_ttft_positive),
                "all_subpage_mean_total_savings_positive": bool(subpage_total_positive)
                and all(subpage_total_positive),
                "modal_contract_observations": modal_contract_observations,
                "modal_contract_failures": modal_contract_failures,
                "reference_parity_observations": reference_parity_observations,
            },
            "measurements": measurements,
        }
        serialized = json.dumps(result, indent=2, sort_keys=True)
        print(serialized, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        if args.strict_parity and not output_contracts_passed:
            raise RuntimeError("same-request final decode failed output-contract gates")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
