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

"""Benchmark production-like streaming tool-call prefill against cold generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nemo_rl.models.generation.vllm.streaming_tool_call import (
    StreamingToolCallPrefillManager,
)


@dataclass(frozen=True)
class GenerationMeasurement:
    """Latency, cache reuse, and output from one final model request."""

    cached_tokens: int
    prompt_tokens: int
    output_token_ids: tuple[int, ...]
    time_to_first_token_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class PrefillStepMeasurement:
    """Cost and committed work from one candidate observation."""

    sequence_no: int
    candidate_tokens: int
    committed_tokens: int
    chunk_tokens: int
    seconds: float


@dataclass(frozen=True)
class PrefillMeasurement:
    """Prefill work followed by the authoritative final model request."""

    start_seconds: float
    close_seconds: float
    committed_tokens: int
    dynamic_committed_tokens: int
    final_tail_tokens: int
    completed_chunks: int
    dummy_tokens: int
    prefix_matched: bool
    steps: tuple[PrefillStepMeasurement, ...]
    final_generation: GenerationMeasurement


@dataclass(frozen=True)
class DirectPairMeasurement:
    """Cold and immediately repeated warm requests for the same prompt."""

    cold: GenerationMeasurement
    warm: GenerationMeasurement


@dataclass(frozen=True)
class PrefixWarmContinuationMeasurement:
    """Production-like continuation after the prior prompt populated APC."""

    candidate_count: int
    initial_candidate_mode: str
    prefix_warm_seconds: float
    prefix_warm_committed_tokens: int
    continuation: PrefillMeasurement


@dataclass(frozen=True)
class PromptTrace:
    """Tokenized empty, growing, and final tool-response prompts."""

    immutable_prefix_tokens: int
    tool_output_tokens: int
    mutable_suffix_tokens: int
    authoritative_prefix: list[int]
    initial: list[int]
    snapshots: tuple[list[int], ...]
    final: list[int]


def _parse_candidate_counts(value: str) -> tuple[int | None, ...]:
    """Parse unique positive snapshot counts, using ``all`` for no limit."""
    parsed = []
    seen: set[int | None] = set()
    for raw_count in value.split(","):
        count_text = raw_count.strip().lower()
        if not count_text:
            raise argparse.ArgumentTypeError("candidate counts cannot be empty")
        if count_text == "all":
            count = None
        else:
            try:
                count = int(count_text)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid candidate count: {raw_count!r}"
                ) from error
            if count <= 0:
                raise argparse.ArgumentTypeError("candidate counts must be positive")
        if count not in seen:
            parsed.append(count)
            seen.add(count)
    if None not in seen:
        raise argparse.ArgumentTypeError("candidate counts must include 'all'")
    return tuple(parsed)


def _candidate_count_label(candidate_count: int | None) -> str:
    return "all" if candidate_count is None else str(candidate_count)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether continuation prefill reduces final model-call latency "
            "enough to justify its engine-side request overhead."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--immutable-prefix-tokens", type=int, default=4_096)
    parser.add_argument("--tool-output-tokens", type=int, default=4_096)
    parser.add_argument("--mutable-suffix-tokens", type=int, default=32)
    parser.add_argument("--candidate-chunk-tokens", type=int, default=512)
    parser.add_argument(
        "--candidate-counts",
        type=_parse_candidate_counts,
        default=_parse_candidate_counts("1,2,4,all"),
        help=(
            "Comma-separated admitted snapshot counts to benchmark; include "
            "'all' for the full continuation."
        ),
    )
    parser.add_argument("--stability-margin-tokens", type=int, default=8)
    parser.add_argument("--cleanup-delay-seconds", type=float, default=0.05)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--stable-first-candidate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Seed prefill from the first snapshot's stable tool-output prefix "
            "instead of only the prior model prefix."
        ),
    )
    return parser.parse_args()


def _repeat_tokens(tokenizer: Any, text: str, count: int) -> list[int]:
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        raise RuntimeError(f"tokenizer returned no tokens for {text!r}")
    repetitions = (count + len(seed) - 1) // len(seed)
    return (seed * repetitions)[:count]


def _build_prompt_trace(tokenizer: Any, args: argparse.Namespace) -> PromptTrace:
    immutable_prefix = _repeat_tokens(
        tokenizer,
        "Conversation history and assistant tool call. ",
        args.immutable_prefix_tokens,
    )
    tool_output = _repeat_tokens(
        tokenizer,
        "test output line: passed elapsed=12.345 seconds\n",
        args.tool_output_tokens,
    )
    mutable_suffix = _repeat_tokens(
        tokenizer,
        "</tool_response> assistant generation prefix ",
        args.mutable_suffix_tokens,
    )
    if tool_output[0] == mutable_suffix[0]:
        tool_output = tool_output[1:] + tool_output[:1]

    snapshot_lengths = list(
        range(
            args.candidate_chunk_tokens,
            len(tool_output),
            args.candidate_chunk_tokens,
        )
    )
    if len(snapshot_lengths) < 2:
        raise ValueError(
            "tool output must contain at least two non-final candidate chunks"
        )
    snapshots = tuple(
        immutable_prefix + tool_output[:length] + mutable_suffix
        for length in snapshot_lengths
    )
    return PromptTrace(
        immutable_prefix_tokens=len(immutable_prefix),
        tool_output_tokens=len(tool_output),
        mutable_suffix_tokens=len(mutable_suffix),
        authoritative_prefix=immutable_prefix,
        initial=immutable_prefix + mutable_suffix,
        snapshots=snapshots,
        final=immutable_prefix + tool_output + mutable_suffix,
    )


async def _generate_one_token(
    llm: Any,
    *,
    prompt_token_ids: list[int],
    sampling_params: Any,
    tokens_prompt_type: type,
) -> GenerationMeasurement:
    request_id = f"streaming-tool-call-prefill-benchmark-{uuid.uuid4()}"
    start_time = time.perf_counter()
    first_token_time: float | None = None
    cached_tokens: int | None = None
    output_token_ids: list[int] = []

    result_generator = llm.generate(
        prompt=tokens_prompt_type(prompt_token_ids=prompt_token_ids),
        sampling_params=sampling_params,
        request_id=request_id,
    )
    async for request_output in result_generator:
        if request_output.num_cached_tokens is not None:
            cached_tokens = request_output.num_cached_tokens
        new_token_ids = [
            token_id
            for completion_output in request_output.outputs
            for token_id in completion_output.token_ids
        ]
        if new_token_ids and first_token_time is None:
            first_token_time = time.perf_counter()
        output_token_ids.extend(new_token_ids)

    end_time = time.perf_counter()
    if cached_tokens is None:
        raise RuntimeError("vLLM did not report num_cached_tokens")
    if first_token_time is None or not output_token_ids:
        raise RuntimeError("vLLM did not generate the expected output token")
    return GenerationMeasurement(
        cached_tokens=cached_tokens,
        prompt_tokens=len(prompt_token_ids),
        output_token_ids=tuple(output_token_ids),
        time_to_first_token_seconds=first_token_time - start_time,
        total_seconds=end_time - start_time,
    )


def _make_prefill_manager(
    llm: Any,
    *,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
    max_prompt_tokens: int,
    priority: int = 0,
    cache_page_size_tokens: int | None = None,
) -> StreamingToolCallPrefillManager:
    def generate_prefill(
        input_stream: AsyncIterator[Any], request_id: str
    ) -> AsyncGenerator[Any, None]:
        return llm.generate(
            prompt=input_stream,
            sampling_params=sampling_params,
            request_id=request_id,
            priority=priority,
        )

    def make_streaming_input(token_ids: list[int]) -> Any:
        return streaming_input_type(
            prompt=tokens_prompt_type(prompt_token_ids=token_ids),
            sampling_params=sampling_params,
        )

    return StreamingToolCallPrefillManager(
        generate=generate_prefill,
        make_streaming_input=make_streaming_input,
        count_output_tokens=lambda output: sum(
            len(completion_output.token_ids) for completion_output in output.outputs
        ),
        max_sessions=1,
        session_ttl_seconds=60,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=max_prompt_tokens,
        cache_page_size_tokens=cache_page_size_tokens,
    )


async def _run_direct_pair(
    llm: Any,
    *,
    final_prompt_token_ids: list[int],
    sampling_params: Any,
    tokens_prompt_type: type,
) -> DirectPairMeasurement:
    cold = await _generate_one_token(
        llm,
        prompt_token_ids=final_prompt_token_ids,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    warm = await _generate_one_token(
        llm,
        prompt_token_ids=final_prompt_token_ids,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    return DirectPairMeasurement(cold=cold, warm=warm)


async def _run_one_shot_prefill(
    llm: Any,
    *,
    trace: PromptTrace,
    cleanup_delay_seconds: float,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
) -> PrefillMeasurement:
    manager = _make_prefill_manager(
        llm,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(trace.final),
    )
    session_id = f"one-shot-{uuid.uuid4()}"
    started = time.perf_counter()
    prime_result = await manager.prime(
        session_id=session_id,
        prompt_token_ids=trace.initial,
    )
    start_seconds = time.perf_counter() - started
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)
    final_generation = await _generate_one_token(
        llm,
        prompt_token_ids=trace.final,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    committed_tokens = prime_result.committed_tokens
    return PrefillMeasurement(
        start_seconds=start_seconds,
        close_seconds=0.0,
        committed_tokens=committed_tokens,
        dynamic_committed_tokens=max(
            0, committed_tokens - trace.immutable_prefix_tokens
        ),
        final_tail_tokens=len(trace.final) - committed_tokens,
        completed_chunks=prime_result.completed_chunks,
        dummy_tokens=prime_result.dummy_tokens,
        prefix_matched=(
            trace.final[:committed_tokens] == trace.initial[:committed_tokens]
        ),
        steps=(),
        final_generation=final_generation,
    )


async def _run_continuation_prefill(
    llm: Any,
    *,
    trace: PromptTrace,
    cleanup_delay_seconds: float,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
    initial_candidate_token_ids: list[int] | None = None,
    candidate_count: int | None = None,
) -> PrefillMeasurement:
    manager = _make_prefill_manager(
        llm,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(trace.final),
    )
    session_id = f"continuation-{uuid.uuid4()}"
    steps = []
    if candidate_count is not None and candidate_count > len(trace.snapshots):
        raise ValueError(
            f"requested {candidate_count} candidates but trace has "
            f"{len(trace.snapshots)}"
        )
    candidates_to_submit = (
        trace.snapshots
        if candidate_count is None
        else trace.snapshots[:candidate_count]
    )
    if not candidates_to_submit:
        raise ValueError("continuation requires at least one candidate")
    started = time.perf_counter()
    if initial_candidate_token_ids is None:
        start_result = await manager.start(
            session_id=session_id,
            prompt_token_ids=trace.initial,
            sequence_no=0,
        )
        candidates = enumerate(candidates_to_submit, start=1)
    else:
        first_candidate = candidates_to_submit[0]
        start_result = await manager.start(
            session_id=session_id,
            prompt_token_ids=first_candidate,
            sequence_no=0,
            initial_candidate_token_ids=initial_candidate_token_ids,
        )
        candidates = enumerate(candidates_to_submit[1:], start=1)
    start_seconds = time.perf_counter() - started
    if initial_candidate_token_ids is None:
        if start_result.chunk_tokens != 0:
            raise RuntimeError("continuation start performed unproven engine work")
    else:
        if start_result.chunk_tokens == 0:
            raise RuntimeError("prefix-seeded start performed no proven engine work")
        steps.append(
            PrefillStepMeasurement(
                sequence_no=0,
                candidate_tokens=len(candidates_to_submit[0]),
                committed_tokens=start_result.committed_tokens,
                chunk_tokens=start_result.chunk_tokens,
                seconds=start_seconds,
            )
        )

    for sequence_no, candidate in candidates:
        step_started = time.perf_counter()
        append_result = await manager.append(
            session_id=session_id,
            prompt_token_ids=candidate,
            sequence_no=sequence_no,
        )
        steps.append(
            PrefillStepMeasurement(
                sequence_no=sequence_no,
                candidate_tokens=len(candidate),
                committed_tokens=append_result.committed_tokens,
                chunk_tokens=append_result.chunk_tokens,
                seconds=time.perf_counter() - step_started,
            )
        )

    close_started = time.perf_counter()
    close_result = await manager.close(
        session_id=session_id,
        final_prompt_token_ids=trace.final,
    )
    close_seconds = time.perf_counter() - close_started
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)
    final_generation = await _generate_one_token(
        llm,
        prompt_token_ids=trace.final,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    return PrefillMeasurement(
        start_seconds=start_seconds,
        close_seconds=close_seconds,
        committed_tokens=close_result.committed_tokens,
        dynamic_committed_tokens=max(
            0, close_result.committed_tokens - trace.immutable_prefix_tokens
        ),
        final_tail_tokens=len(trace.final) - close_result.committed_tokens,
        completed_chunks=close_result.completed_chunks,
        dummy_tokens=close_result.dummy_tokens,
        prefix_matched=close_result.prefix_matched,
        steps=tuple(steps),
        final_generation=final_generation,
    )


async def _run_prefix_warm_seeded_continuation(
    llm: Any,
    *,
    trace: PromptTrace,
    cleanup_delay_seconds: float,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
    candidate_count: int | None,
    stable_first_candidate: bool,
) -> PrefixWarmContinuationMeasurement:
    """Run prefix-seeded continuation with APC state from the prior prompt."""
    warm_manager = _make_prefill_manager(
        llm,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(trace.final),
    )
    warm_started = time.perf_counter()
    warm_result = await warm_manager.prime(
        session_id=f"prefix-warm-{uuid.uuid4()}",
        prompt_token_ids=trace.initial,
    )
    prefix_warm_seconds = time.perf_counter() - warm_started
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)
    if stable_first_candidate:
        stable_candidate_end = len(trace.snapshots[0]) - trace.mutable_suffix_tokens
        initial_candidate_token_ids = trace.snapshots[0][:stable_candidate_end]
        initial_candidate_mode = "stable_first_snapshot"
    else:
        initial_candidate_token_ids = trace.authoritative_prefix
        initial_candidate_mode = "authoritative_model_prefix"
    continuation = await _run_continuation_prefill(
        llm,
        trace=trace,
        cleanup_delay_seconds=cleanup_delay_seconds,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        initial_candidate_token_ids=initial_candidate_token_ids,
        candidate_count=candidate_count,
    )
    return PrefixWarmContinuationMeasurement(
        candidate_count=(
            len(trace.snapshots) if candidate_count is None else candidate_count
        ),
        initial_candidate_mode=initial_candidate_mode,
        prefix_warm_seconds=prefix_warm_seconds,
        prefix_warm_committed_tokens=warm_result.committed_tokens,
        continuation=continuation,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _prefill_request_seconds(measurement: PrefillMeasurement) -> float:
    return sum(step.seconds for step in measurement.steps)


def _prefill_engine_requests(measurement: PrefillMeasurement) -> int:
    return sum(step.chunk_tokens > 0 for step in measurement.steps)


def _prefix_warm_candidate_summary(
    measurements: Sequence[PrefixWarmContinuationMeasurement],
    *,
    direct_controls: Sequence[PrefillMeasurement],
) -> dict[str, Any]:
    """Summarize one admitted-snapshot budget against the warm direct control."""
    if len(measurements) != len(direct_controls):
        raise ValueError("candidate measurements and direct controls must be paired")
    direct_model_call_ttft = [
        measurement.final_generation.time_to_first_token_seconds
        for measurement in direct_controls
    ]
    model_call_ttft = [
        measurement.continuation.final_generation.time_to_first_token_seconds
        for measurement in measurements
    ]
    engine_request_seconds = [
        _prefill_request_seconds(measurement.continuation)
        for measurement in measurements
    ]
    direct_mean = statistics.fmean(direct_model_call_ttft)
    model_call_mean = statistics.fmean(model_call_ttft)
    engine_request_mean = statistics.fmean(engine_request_seconds)
    paired_model_call_savings = [
        direct - candidate
        for direct, candidate in zip(
            direct_model_call_ttft,
            model_call_ttft,
            strict=True,
        )
    ]
    paired_unoverlapped_deltas = [
        request + candidate - direct
        for request, candidate, direct in zip(
            engine_request_seconds,
            model_call_ttft,
            direct_model_call_ttft,
            strict=True,
        )
    ]
    model_call_savings = statistics.fmean(paired_model_call_savings)
    if model_call_savings <= 0:
        overlap_break_even_fraction = None
    else:
        overlap_break_even_fraction = max(
            0.0,
            1.0 - model_call_savings / engine_request_mean,
        )
    return {
        "candidate_count": measurements[0].candidate_count,
        "direct_control_model_call_ttft_seconds": _summary(direct_model_call_ttft),
        "model_call_ttft_seconds": _summary(model_call_ttft),
        "engine_request_seconds": _summary(engine_request_seconds),
        "prefix_warm_setup_seconds": _summary(
            [measurement.prefix_warm_seconds for measurement in measurements]
        ),
        "model_call_savings_seconds": model_call_savings,
        "paired_model_call_savings_seconds": _summary(paired_model_call_savings),
        "paired_model_call_win_fraction": statistics.fmean(
            savings > 0 for savings in paired_model_call_savings
        ),
        "model_call_reduction_pct": 100 * (1 - model_call_mean / direct_mean),
        "unoverlapped_total_seconds": engine_request_mean + model_call_mean,
        "unoverlapped_delta_vs_prefix_warm_direct_seconds": (
            engine_request_mean + model_call_mean - direct_mean
        ),
        "paired_unoverlapped_delta_seconds": _summary(paired_unoverlapped_deltas),
        "paired_unoverlapped_win_fraction": statistics.fmean(
            delta < 0 for delta in paired_unoverlapped_deltas
        ),
        "engine_overlap_break_even_fraction": overlap_break_even_fraction,
        "cached_tokens": _summary(
            [
                float(measurement.continuation.final_generation.cached_tokens)
                for measurement in measurements
            ]
        ),
        "committed_tokens": _summary(
            [
                float(measurement.continuation.committed_tokens)
                for measurement in measurements
            ]
        ),
        "dynamic_committed_tokens": _summary(
            [
                float(measurement.continuation.dynamic_committed_tokens)
                for measurement in measurements
            ]
        ),
        "engine_requests": _summary(
            [
                float(_prefill_engine_requests(measurement.continuation))
                for measurement in measurements
            ]
        ),
    }


def _validate_results(
    direct_pairs: Sequence[DirectPairMeasurement],
    one_shot: Sequence[PrefillMeasurement],
    continuation: Sequence[PrefillMeasurement],
    prefix_warm_seeded_candidates: Sequence[PrefixWarmContinuationMeasurement],
) -> None:
    streaming_measurements = list(continuation) + [
        measurement.continuation for measurement in prefix_warm_seeded_candidates
    ]
    generations = [
        generation for pair in direct_pairs for generation in (pair.cold, pair.warm)
    ] + [
        measurement.final_generation
        for measurement in (*one_shot, *streaming_measurements)
    ]
    output_token_ids = {generation.output_token_ids for generation in generations}
    if len(output_token_ids) != 1:
        raise RuntimeError(
            f"deterministic final output changed across cases: {output_token_ids}"
        )
    if any(pair.cold.cached_tokens != 0 for pair in direct_pairs):
        raise RuntimeError("cold control unexpectedly reused prefix-cache tokens")
    if any(pair.warm.cached_tokens == 0 for pair in direct_pairs):
        raise RuntimeError("warm control did not reuse prefix-cache tokens")
    if any(not measurement.prefix_matched for measurement in streaming_measurements):
        raise RuntimeError(
            "continuation committed a prefix absent from the final prompt"
        )
    if any(
        _prefill_engine_requests(measurement) == 0
        for measurement in streaming_measurements
    ):
        raise RuntimeError("continuation did not submit any proven prefill chunk")
    if any(
        measurement.final_generation.cached_tokens == 0
        for measurement in streaming_measurements
    ):
        raise RuntimeError("continuation prefill produced no reusable APC tokens")


async def _main(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    positive_values = {
        "immutable_prefix_tokens": args.immutable_prefix_tokens,
        "tool_output_tokens": args.tool_output_tokens,
        "mutable_suffix_tokens": args.mutable_suffix_tokens,
        "candidate_chunk_tokens": args.candidate_chunk_tokens,
        "repeats": args.repeats,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmup_repeats < 0:
        raise ValueError("warmup_repeats must be non-negative")
    if args.stability_margin_tokens < 0:
        raise ValueError("stability_margin_tokens must be non-negative")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    trace = _build_prompt_trace(tokenizer, args)
    for candidate_count in args.candidate_counts:
        if candidate_count is not None and candidate_count > len(trace.snapshots):
            raise ValueError(
                f"requested {candidate_count} candidates but trace has "
                f"{len(trace.snapshots)}"
            )
    if len(trace.final) >= args.max_model_len:
        raise ValueError(
            f"final prompt has {len(trace.final)} tokens but max_model_len is "
            f"{args.max_model_len}"
        )

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    llm = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=1,
        max_tokens=1,
        output_kind=RequestOutputKind.DELTA,
    )

    async def reset_cache() -> None:
        await llm.reset_prefix_cache()
        await asyncio.sleep(0)

    try:
        for _ in range(args.warmup_repeats):
            await reset_cache()
            await _run_direct_pair(
                llm,
                final_prompt_token_ids=trace.final,
                sampling_params=sampling_params,
                tokens_prompt_type=TokensPrompt,
            )
            await reset_cache()
            await _run_one_shot_prefill(
                llm,
                trace=trace,
                cleanup_delay_seconds=args.cleanup_delay_seconds,
                sampling_params=sampling_params,
                tokens_prompt_type=TokensPrompt,
                streaming_input_type=StreamingInput,
                stability_margin_tokens=args.stability_margin_tokens,
            )
            await reset_cache()
            await _run_continuation_prefill(
                llm,
                trace=trace,
                cleanup_delay_seconds=args.cleanup_delay_seconds,
                sampling_params=sampling_params,
                tokens_prompt_type=TokensPrompt,
                streaming_input_type=StreamingInput,
                stability_margin_tokens=args.stability_margin_tokens,
            )
            for candidate_count in args.candidate_counts:
                await reset_cache()
                await _run_one_shot_prefill(
                    llm,
                    trace=trace,
                    cleanup_delay_seconds=args.cleanup_delay_seconds,
                    sampling_params=sampling_params,
                    tokens_prompt_type=TokensPrompt,
                    streaming_input_type=StreamingInput,
                    stability_margin_tokens=args.stability_margin_tokens,
                )
                await reset_cache()
                await _run_prefix_warm_seeded_continuation(
                    llm,
                    trace=trace,
                    cleanup_delay_seconds=args.cleanup_delay_seconds,
                    sampling_params=sampling_params,
                    tokens_prompt_type=TokensPrompt,
                    streaming_input_type=StreamingInput,
                    stability_margin_tokens=args.stability_margin_tokens,
                    candidate_count=candidate_count,
                    stable_first_candidate=args.stable_first_candidate,
                )

        direct_pairs = []
        one_shot = []
        continuation = []
        prefix_warm_seeded_candidate_sweep = {
            _candidate_count_label(candidate_count): []
            for candidate_count in args.candidate_counts
        }
        prefix_warm_candidate_direct_controls = {
            _candidate_count_label(candidate_count): []
            for candidate_count in args.candidate_counts
        }

        async def run_direct_control(label: str) -> None:
            await reset_cache()
            prefix_warm_candidate_direct_controls[label].append(
                await _run_one_shot_prefill(
                    llm,
                    trace=trace,
                    cleanup_delay_seconds=args.cleanup_delay_seconds,
                    sampling_params=sampling_params,
                    tokens_prompt_type=TokensPrompt,
                    streaming_input_type=StreamingInput,
                    stability_margin_tokens=args.stability_margin_tokens,
                )
            )

        async def run_candidate(
            label: str,
            candidate_count: int | None,
        ) -> None:
            await reset_cache()
            prefix_warm_seeded_candidate_sweep[label].append(
                await _run_prefix_warm_seeded_continuation(
                    llm,
                    trace=trace,
                    cleanup_delay_seconds=args.cleanup_delay_seconds,
                    sampling_params=sampling_params,
                    tokens_prompt_type=TokensPrompt,
                    streaming_input_type=StreamingInput,
                    stability_margin_tokens=args.stability_margin_tokens,
                    candidate_count=candidate_count,
                    stable_first_candidate=args.stable_first_candidate,
                )
            )

        for repeat_index in range(args.repeats):
            await reset_cache()
            direct_pairs.append(
                await _run_direct_pair(
                    llm,
                    final_prompt_token_ids=trace.final,
                    sampling_params=sampling_params,
                    tokens_prompt_type=TokensPrompt,
                )
            )
            await reset_cache()
            one_shot.append(
                await _run_one_shot_prefill(
                    llm,
                    trace=trace,
                    cleanup_delay_seconds=args.cleanup_delay_seconds,
                    sampling_params=sampling_params,
                    tokens_prompt_type=TokensPrompt,
                    streaming_input_type=StreamingInput,
                    stability_margin_tokens=args.stability_margin_tokens,
                )
            )
            await reset_cache()
            continuation.append(
                await _run_continuation_prefill(
                    llm,
                    trace=trace,
                    cleanup_delay_seconds=args.cleanup_delay_seconds,
                    sampling_params=sampling_params,
                    tokens_prompt_type=TokensPrompt,
                    streaming_input_type=StreamingInput,
                    stability_margin_tokens=args.stability_margin_tokens,
                )
            )
            for candidate_count in args.candidate_counts:
                label = _candidate_count_label(candidate_count)
                if repeat_index % 2 == 0:
                    await run_direct_control(label)
                    await run_candidate(label, candidate_count)
                else:
                    await run_candidate(label, candidate_count)
                    await run_direct_control(label)

        prefix_warm_seeded_candidates = [
            measurement
            for measurements in prefix_warm_seeded_candidate_sweep.values()
            for measurement in measurements
        ]
        prefix_warm_direct_control_measurements = [
            measurement
            for measurements in prefix_warm_candidate_direct_controls.values()
            for measurement in measurements
        ]
        prefix_warm_seeded_continuation = prefix_warm_seeded_candidate_sweep["all"]

        _validate_results(
            direct_pairs,
            [*one_shot, *prefix_warm_direct_control_measurements],
            continuation,
            prefix_warm_seeded_candidates,
        )
        cold_ttft = [pair.cold.time_to_first_token_seconds for pair in direct_pairs]
        warm_ttft = [pair.warm.time_to_first_token_seconds for pair in direct_pairs]
        one_shot_ttft = [
            measurement.final_generation.time_to_first_token_seconds
            for measurement in one_shot
        ]
        continuation_ttft = [
            measurement.final_generation.time_to_first_token_seconds
            for measurement in continuation
        ]
        one_shot_setup = [measurement.start_seconds for measurement in one_shot]
        continuation_requests = [
            _prefill_request_seconds(measurement) for measurement in continuation
        ]
        prefix_warm_seeded_continuation_ttft = [
            measurement.continuation.final_generation.time_to_first_token_seconds
            for measurement in prefix_warm_seeded_continuation
        ]
        prefix_warm_seeded_continuation_requests = [
            _prefill_request_seconds(measurement.continuation)
            for measurement in prefix_warm_seeded_continuation
        ]
        prefix_warm_seconds = [
            measurement.prefix_warm_seconds
            for measurement in prefix_warm_seeded_continuation
        ]
        prefix_warm_candidate_summaries = {
            label: _prefix_warm_candidate_summary(
                measurements,
                direct_controls=prefix_warm_candidate_direct_controls[label],
            )
            for label, measurements in prefix_warm_seeded_candidate_sweep.items()
        }
        cold_mean = statistics.fmean(cold_ttft)
        one_shot_mean = statistics.fmean(one_shot_ttft)
        continuation_mean = statistics.fmean(continuation_ttft)
        continuation_request_mean = statistics.fmean(continuation_requests)
        prefix_warm_seeded_continuation_mean = statistics.fmean(
            prefix_warm_seeded_continuation_ttft
        )
        prefix_warm_seeded_continuation_request_mean = statistics.fmean(
            prefix_warm_seeded_continuation_requests
        )
        continuation_model_call_savings = cold_mean - continuation_mean
        continuation_break_even_overlap = max(
            0.0,
            1.0 - continuation_model_call_savings / continuation_request_mean,
        )
        prefix_warm_seeded_model_call_savings = (
            one_shot_mean - prefix_warm_seeded_continuation_mean
        )
        prefix_warm_seeded_break_even_overlap = max(
            0.0,
            1.0
            - prefix_warm_seeded_model_call_savings
            / prefix_warm_seeded_continuation_request_mean,
        )
        result = {
            "model": args.model,
            "engine": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
            },
            "trace": {
                "immutable_prefix_tokens": trace.immutable_prefix_tokens,
                "tool_output_tokens": trace.tool_output_tokens,
                "mutable_suffix_tokens": trace.mutable_suffix_tokens,
                "initial_prompt_tokens": len(trace.initial),
                "final_prompt_tokens": len(trace.final),
                "candidate_count": len(trace.snapshots),
                "candidate_chunk_tokens": args.candidate_chunk_tokens,
                "candidate_counts": [
                    _candidate_count_label(candidate_count)
                    for candidate_count in args.candidate_counts
                ],
                "paired_candidate_controls": True,
                "paired_order": "alternating_by_repeat",
                "stable_first_candidate": args.stable_first_candidate,
                "stability_margin_tokens": args.stability_margin_tokens,
            },
            "repeats": args.repeats,
            "warmup_repeats": args.warmup_repeats,
            "summary": {
                "cold_model_call_ttft_seconds": _summary(cold_ttft),
                "warm_model_call_ttft_seconds": _summary(warm_ttft),
                "one_shot_model_call_ttft_seconds": _summary(one_shot_ttft),
                "prefix_warm_direct_model_call_ttft_seconds": _summary(one_shot_ttft),
                "continuation_model_call_ttft_seconds": _summary(continuation_ttft),
                "prefix_warm_seeded_continuation_model_call_ttft_seconds": (
                    _summary(prefix_warm_seeded_continuation_ttft)
                ),
                "one_shot_engine_prefill_seconds": _summary(one_shot_setup),
                "prefix_warm_setup_seconds": _summary(prefix_warm_seconds),
                "continuation_engine_request_seconds": _summary(continuation_requests),
                "prefix_warm_seeded_continuation_engine_request_seconds": (
                    _summary(prefix_warm_seeded_continuation_requests)
                ),
                "continuation_model_call_savings_seconds": (
                    continuation_model_call_savings
                ),
                "continuation_unoverlapped_total_seconds": (
                    continuation_request_mean + continuation_mean
                ),
                "continuation_unoverlapped_delta_vs_cold_seconds": (
                    continuation_request_mean + continuation_mean - cold_mean
                ),
                "continuation_engine_overlap_break_even_fraction": (
                    continuation_break_even_overlap
                ),
                "prefix_warm_seeded_model_call_savings_seconds": (
                    prefix_warm_seeded_model_call_savings
                ),
                "prefix_warm_seeded_unoverlapped_total_seconds": (
                    prefix_warm_seeded_continuation_request_mean
                    + prefix_warm_seeded_continuation_mean
                ),
                "prefix_warm_seeded_unoverlapped_delta_vs_prefix_warm_direct_seconds": (
                    prefix_warm_seeded_continuation_request_mean
                    + prefix_warm_seeded_continuation_mean
                    - one_shot_mean
                ),
                "prefix_warm_seeded_engine_overlap_break_even_fraction": (
                    prefix_warm_seeded_break_even_overlap
                ),
                "one_shot_model_call_reduction_pct": 100
                * (1 - one_shot_mean / cold_mean),
                "continuation_model_call_reduction_pct": 100
                * (1 - continuation_mean / cold_mean),
                "continuation_vs_one_shot_model_call_reduction_pct": 100
                * (1 - continuation_mean / one_shot_mean),
                "prefix_warm_seeded_model_call_reduction_pct": 100
                * (1 - prefix_warm_seeded_continuation_mean / one_shot_mean),
                "one_shot_cached_tokens": _summary(
                    [
                        float(measurement.final_generation.cached_tokens)
                        for measurement in one_shot
                    ]
                ),
                "continuation_cached_tokens": _summary(
                    [
                        float(measurement.final_generation.cached_tokens)
                        for measurement in continuation
                    ]
                ),
                "continuation_committed_tokens": _summary(
                    [
                        float(measurement.committed_tokens)
                        for measurement in continuation
                    ]
                ),
                "continuation_dynamic_committed_tokens": _summary(
                    [
                        float(measurement.dynamic_committed_tokens)
                        for measurement in continuation
                    ]
                ),
                "continuation_engine_requests": _summary(
                    [
                        float(_prefill_engine_requests(measurement))
                        for measurement in continuation
                    ]
                ),
                "prefix_warm_seeded_continuation_cached_tokens": _summary(
                    [
                        float(measurement.continuation.final_generation.cached_tokens)
                        for measurement in prefix_warm_seeded_continuation
                    ]
                ),
                "prefix_warm_seeded_continuation_committed_tokens": _summary(
                    [
                        float(measurement.continuation.committed_tokens)
                        for measurement in prefix_warm_seeded_continuation
                    ]
                ),
                "prefix_warm_seeded_continuation_dynamic_committed_tokens": (
                    _summary(
                        [
                            float(measurement.continuation.dynamic_committed_tokens)
                            for measurement in prefix_warm_seeded_continuation
                        ]
                    )
                ),
                "prefix_warm_seeded_continuation_engine_requests": _summary(
                    [
                        float(_prefill_engine_requests(measurement.continuation))
                        for measurement in prefix_warm_seeded_continuation
                    ]
                ),
                "prefix_warm_seeded_candidate_sweep": (prefix_warm_candidate_summaries),
            },
            "raw": {
                "direct": [asdict(measurement) for measurement in direct_pairs],
                "one_shot": [asdict(measurement) for measurement in one_shot],
                "continuation": [asdict(measurement) for measurement in continuation],
                "prefix_warm_seeded_continuation": [
                    asdict(measurement)
                    for measurement in prefix_warm_seeded_continuation
                ],
                "prefix_warm_seeded_candidate_sweep": {
                    label: [asdict(measurement) for measurement in measurements]
                    for label, measurements in (
                        prefix_warm_seeded_candidate_sweep.items()
                    )
                },
                "prefix_warm_candidate_direct_controls": {
                    label: [asdict(measurement) for measurement in measurements]
                    for label, measurements in (
                        prefix_warm_candidate_direct_controls.items()
                    )
                },
            },
        }
        serialized = json.dumps(result, indent=2, sort_keys=True)
        print(serialized, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
