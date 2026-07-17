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

"""Measure background-prefill admission, overhead, and final model TTFT."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from examples.swe_bench.benchmark_streaming_tool_call_prefill import (
    GenerationMeasurement,
    PromptTrace,
    _build_prompt_trace,
    _generate_one_token,
    _make_prefill_manager,
    _summary,
)


@dataclass(frozen=True)
class ControlMeasurement:
    """Prefix-warm control followed by the authoritative model request."""

    prefix_warm_seconds: float
    final_generation: GenerationMeasurement


@dataclass(frozen=True)
class BackgroundMeasurement:
    """Background prefill with a fixed command-overlap budget."""

    overlap_seconds: float
    prefix_warm_seconds: float
    enqueue_seconds: float
    close_seconds: float
    scheduled_chunks: int
    scheduled_tokens: int
    completed_chunks: int
    completed_tokens: int
    effective_chunks: int
    dynamic_tokens: int
    cancelled_chunks: int
    cancelled_tokens: int
    failed_chunks: int
    failed_tokens: int
    engine_completion_seconds: float
    committed_tokens: int
    prefix_matched: bool
    final_generation: GenerationMeasurement


@dataclass(frozen=True)
class ForegroundContentionMeasurement:
    """Foreground request latency while one background prefill is admitted."""

    background_priority: int
    foreground_generation: GenerationMeasurement
    background_completed_chunks: int
    background_cancelled_chunks: int


def _parse_overlaps(value: str) -> tuple[float, ...]:
    overlaps = tuple(float(item.strip()) for item in value.split(","))
    if not overlaps or any(overlap < 0 for overlap in overlaps):
        raise argparse.ArgumentTypeError("overlaps must be non-negative seconds")
    if len(set(overlaps)) != len(overlaps):
        raise argparse.ArgumentTypeError("overlaps must be unique")
    return overlaps


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--immutable-prefix-tokens", type=int, default=4_096)
    parser.add_argument("--tool-output-tokens", type=int, default=4_096)
    parser.add_argument("--mutable-suffix-tokens", type=int, default=32)
    parser.add_argument("--candidate-chunk-tokens", type=int, default=512)
    parser.add_argument("--stability-margin-tokens", type=int, default=8)
    parser.add_argument("--background-priority", type=int, default=1)
    parser.add_argument("--cache-page-size-tokens", type=int)
    parser.add_argument("--overlap-seconds", type=_parse_overlaps, default=(0.0,))
    parser.add_argument("--cleanup-delay-seconds", type=float, default=0.05)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--contention-repeats", type=int, default=10)
    parser.add_argument("--contention-foreground-tokens", type=int, default=2_048)
    parser.add_argument(
        "--candidate-chunk-token-sweep",
        type=_parse_positive_ints,
        default=(),
    )
    parser.add_argument("--size-sweep-overlap-seconds", type=float, default=0.075)
    parser.add_argument("--size-sweep-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


async def _warm_authoritative_prefix(
    llm: Any,
    *,
    trace: PromptTrace,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
) -> float:
    manager = _make_prefill_manager(
        llm,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(trace.final),
    )
    started_at = time.perf_counter()
    result = await manager.prime(
        session_id=f"background-prefill-warm-{uuid.uuid4()}",
        prompt_token_ids=trace.initial,
    )
    if not result.prefix_matched or result.completed_chunks != 1:
        raise RuntimeError("authoritative prefix warmup did not complete")
    return time.perf_counter() - started_at


async def _run_control(
    llm: Any,
    *,
    trace: PromptTrace,
    cleanup_delay_seconds: float,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
) -> ControlMeasurement:
    prefix_warm_seconds = await _warm_authoritative_prefix(
        llm,
        trace=trace,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
    )
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)
    final_generation = await _generate_one_token(
        llm,
        prompt_token_ids=trace.final,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    return ControlMeasurement(
        prefix_warm_seconds=prefix_warm_seconds,
        final_generation=final_generation,
    )


async def _run_background(
    llm: Any,
    *,
    trace: PromptTrace,
    overlap_seconds: float,
    cleanup_delay_seconds: float,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
    background_priority: int,
    cache_page_size_tokens: int | None,
) -> BackgroundMeasurement:
    prefix_warm_seconds = await _warm_authoritative_prefix(
        llm,
        trace=trace,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
    )
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)

    manager = _make_prefill_manager(
        llm,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(trace.final),
        priority=background_priority,
        cache_page_size_tokens=cache_page_size_tokens,
    )
    first_snapshot = trace.snapshots[0]
    stable_candidate_end = len(first_snapshot) - trace.mutable_suffix_tokens
    session_id = f"background-prefill-{uuid.uuid4()}"
    start_result = await manager.start_background(
        session_id=session_id,
        prompt_token_ids=first_snapshot,
        initial_candidate_token_ids=first_snapshot[:stable_candidate_end],
        dynamic_token_baseline=trace.immutable_prefix_tokens,
    )
    if overlap_seconds:
        await asyncio.sleep(overlap_seconds)
    close_started_at = time.perf_counter()
    close_result = await manager.close(
        session_id=session_id,
        final_prompt_token_ids=trace.final,
    )
    close_seconds = time.perf_counter() - close_started_at
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)
    final_generation = await _generate_one_token(
        llm,
        prompt_token_ids=trace.final,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    return BackgroundMeasurement(
        overlap_seconds=overlap_seconds,
        prefix_warm_seconds=prefix_warm_seconds,
        enqueue_seconds=start_result.enqueue_seconds,
        close_seconds=close_seconds,
        scheduled_chunks=start_result.scheduled_chunks,
        scheduled_tokens=start_result.scheduled_tokens,
        completed_chunks=close_result.background_completed_chunks,
        completed_tokens=close_result.background_completed_tokens,
        effective_chunks=close_result.background_effective_chunks,
        dynamic_tokens=close_result.background_dynamic_tokens,
        cancelled_chunks=close_result.background_cancelled_chunks,
        cancelled_tokens=close_result.background_cancelled_tokens,
        failed_chunks=close_result.background_failed_chunks,
        failed_tokens=close_result.background_failed_tokens,
        engine_completion_seconds=close_result.background_completion_seconds,
        committed_tokens=close_result.committed_tokens,
        prefix_matched=close_result.prefix_matched,
        final_generation=final_generation,
    )


async def _run_foreground_contention(
    llm: Any,
    *,
    trace: PromptTrace,
    foreground_prompt_token_ids: list[int],
    background_priority: int,
    cleanup_delay_seconds: float,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
    stability_margin_tokens: int,
    cache_page_size_tokens: int | None,
) -> ForegroundContentionMeasurement:
    await _warm_authoritative_prefix(
        llm,
        trace=trace,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
    )
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)

    manager = _make_prefill_manager(
        llm,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
        streaming_input_type=streaming_input_type,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(trace.final),
        priority=background_priority,
        cache_page_size_tokens=cache_page_size_tokens,
    )
    first_snapshot = trace.snapshots[0]
    stable_candidate_end = len(first_snapshot) - trace.mutable_suffix_tokens
    session_id = f"background-prefill-contention-{uuid.uuid4()}"
    await manager.start_background(
        session_id=session_id,
        prompt_token_ids=first_snapshot,
        initial_candidate_token_ids=first_snapshot[:stable_candidate_end],
        dynamic_token_baseline=trace.immutable_prefix_tokens,
    )
    foreground_generation = await _generate_one_token(
        llm,
        prompt_token_ids=foreground_prompt_token_ids,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    closed = await manager.close(
        session_id=session_id,
        final_prompt_token_ids=trace.final,
    )
    return ForegroundContentionMeasurement(
        background_priority=background_priority,
        foreground_generation=foreground_generation,
        background_completed_chunks=closed.background_completed_chunks,
        background_cancelled_chunks=closed.background_cancelled_chunks,
    )


def _summarize_pair(
    controls: list[ControlMeasurement],
    backgrounds: list[BackgroundMeasurement],
) -> dict[str, Any]:
    control_ttft = [
        item.final_generation.time_to_first_token_seconds for item in controls
    ]
    background_ttft = [
        item.final_generation.time_to_first_token_seconds for item in backgrounds
    ]
    paired_savings = [
        control - background
        for control, background in zip(control_ttft, background_ttft, strict=True)
    ]
    control_plane_seconds = [
        item.enqueue_seconds + item.close_seconds for item in backgrounds
    ]
    observed_dynamic_cached_tokens = [
        max(
            0,
            background.final_generation.cached_tokens
            - control.final_generation.cached_tokens,
        )
        for control, background in zip(controls, backgrounds, strict=True)
    ]
    return {
        "acknowledged_admission_fraction": statistics.fmean(
            item.completed_chunks > 0 for item in backgrounds
        ),
        "observed_kv_cache_admission_fraction": statistics.fmean(
            cached_tokens > 0 for cached_tokens in observed_dynamic_cached_tokens
        ),
        "cancellation_fraction": statistics.fmean(
            item.cancelled_chunks > 0 for item in backgrounds
        ),
        "failure_fraction": statistics.fmean(
            item.failed_chunks > 0 for item in backgrounds
        ),
        "control_model_call_ttft_seconds": _summary(control_ttft),
        "background_model_call_ttft_seconds": _summary(background_ttft),
        "paired_model_call_savings_seconds": _summary(paired_savings),
        "paired_model_call_win_fraction": statistics.fmean(
            saving > 0 for saving in paired_savings
        ),
        "background_enqueue_seconds": _summary(
            [item.enqueue_seconds for item in backgrounds]
        ),
        "background_close_seconds": _summary(
            [item.close_seconds for item in backgrounds]
        ),
        "nonoverlapped_control_plane_seconds": _summary(control_plane_seconds),
        "paired_net_savings_after_control_plane_seconds": _summary(
            [
                saving - overhead
                for saving, overhead in zip(
                    paired_savings, control_plane_seconds, strict=True
                )
            ]
        ),
        "background_engine_completion_seconds": _summary(
            [item.engine_completion_seconds for item in backgrounds]
        ),
        "scheduled_tokens": _summary(
            [float(item.scheduled_tokens) for item in backgrounds]
        ),
        "completed_tokens": _summary(
            [float(item.completed_tokens) for item in backgrounds]
        ),
        "dynamic_tokens": _summary(
            [float(item.dynamic_tokens) for item in backgrounds]
        ),
        "cancelled_tokens": _summary(
            [float(item.cancelled_tokens) for item in backgrounds]
        ),
        "control_cached_tokens": _summary(
            [float(item.final_generation.cached_tokens) for item in controls]
        ),
        "background_cached_tokens": _summary(
            [float(item.final_generation.cached_tokens) for item in backgrounds]
        ),
        "observed_dynamic_cached_tokens": _summary(
            [float(tokens) for tokens in observed_dynamic_cached_tokens]
        ),
    }


async def _main(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    if (
        args.repeats <= 0
        or args.warmup_repeats < 0
        or args.contention_repeats <= 0
        or args.size_sweep_repeats <= 0
    ):
        raise ValueError("repeats must be positive and warmup_repeats non-negative")
    if args.size_sweep_overlap_seconds < 0:
        raise ValueError("size_sweep_overlap_seconds must be non-negative")
    if args.background_priority <= 0:
        raise ValueError("background_priority must be positive")
    if args.cache_page_size_tokens is not None and args.cache_page_size_tokens <= 0:
        raise ValueError("cache_page_size_tokens must be positive when set")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    trace = _build_prompt_trace(tokenizer, args)
    if len(trace.final) >= args.max_model_len:
        raise ValueError("final prompt exceeds max_model_len")
    if not 0 < args.contention_foreground_tokens <= len(trace.final):
        raise ValueError(
            "contention_foreground_tokens must be in [1, final prompt tokens]"
        )
    foreground_prompt_token_ids = list(
        reversed(trace.final[-args.contention_foreground_tokens :])
    )
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enable_prefix_caching=True,
            enforce_eager=args.enforce_eager,
            trust_remote_code=True,
            scheduling_policy="priority",
        )
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=1,
        max_tokens=1,
        output_kind=RequestOutputKind.DELTA,
    )

    async def reset_cache() -> None:
        deadline = time.perf_counter() + 5.0
        while not await llm.reset_prefix_cache():
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "vLLM prefix cache remained busy for more than 5 seconds"
                )
            await asyncio.sleep(0.05)

    controls = {str(overlap): [] for overlap in args.overlap_seconds}
    backgrounds = {str(overlap): [] for overlap in args.overlap_seconds}
    contention = {0: [], args.background_priority: []}

    async def run_control(overlap: float) -> None:
        await reset_cache()
        controls[str(overlap)].append(
            await _run_control(
                llm,
                trace=trace,
                cleanup_delay_seconds=args.cleanup_delay_seconds,
                sampling_params=sampling_params,
                tokens_prompt_type=TokensPrompt,
                streaming_input_type=StreamingInput,
                stability_margin_tokens=args.stability_margin_tokens,
            )
        )

    async def run_background(overlap: float) -> None:
        await reset_cache()
        backgrounds[str(overlap)].append(
            await _run_background(
                llm,
                trace=trace,
                overlap_seconds=overlap,
                cleanup_delay_seconds=args.cleanup_delay_seconds,
                sampling_params=sampling_params,
                tokens_prompt_type=TokensPrompt,
                streaming_input_type=StreamingInput,
                stability_margin_tokens=args.stability_margin_tokens,
                background_priority=args.background_priority,
                cache_page_size_tokens=args.cache_page_size_tokens,
            )
        )

    async def run_contention(priority: int) -> None:
        await reset_cache()
        contention[priority].append(
            await _run_foreground_contention(
                llm,
                trace=trace,
                foreground_prompt_token_ids=foreground_prompt_token_ids,
                background_priority=priority,
                cleanup_delay_seconds=args.cleanup_delay_seconds,
                sampling_params=sampling_params,
                tokens_prompt_type=TokensPrompt,
                streaming_input_type=StreamingInput,
                stability_margin_tokens=args.stability_margin_tokens,
                cache_page_size_tokens=args.cache_page_size_tokens,
            )
        )

    try:
        for _ in range(args.warmup_repeats):
            await run_control(args.overlap_seconds[-1])
            controls[str(args.overlap_seconds[-1])].clear()
            await run_background(args.overlap_seconds[-1])
            backgrounds[str(args.overlap_seconds[-1])].clear()

        for repeat_index in range(args.repeats):
            for overlap in args.overlap_seconds:
                if repeat_index % 2 == 0:
                    await run_control(overlap)
                    await run_background(overlap)
                else:
                    await run_background(overlap)
                    await run_control(overlap)

        for repeat_index in range(args.contention_repeats):
            if repeat_index % 2 == 0:
                await run_contention(0)
                await run_contention(args.background_priority)
            else:
                await run_contention(args.background_priority)
                await run_contention(0)

        size_sweep: dict[str, dict[str, Any]] = {}
        size_sweep_generations: list[GenerationMeasurement] = []
        size_sweep_prefix_matches: list[bool] = []
        for candidate_chunk_tokens in args.candidate_chunk_token_sweep:
            sweep_args = argparse.Namespace(**vars(args))
            sweep_args.candidate_chunk_tokens = candidate_chunk_tokens
            sweep_trace = _build_prompt_trace(tokenizer, sweep_args)
            if sweep_trace.final != trace.final:
                raise RuntimeError("candidate size sweep changed the final prompt")

            sweep_controls: list[ControlMeasurement] = []
            sweep_backgrounds: list[BackgroundMeasurement] = []

            async def run_sweep_control() -> None:
                await reset_cache()
                sweep_controls.append(
                    await _run_control(
                        llm,
                        trace=sweep_trace,
                        cleanup_delay_seconds=args.cleanup_delay_seconds,
                        sampling_params=sampling_params,
                        tokens_prompt_type=TokensPrompt,
                        streaming_input_type=StreamingInput,
                        stability_margin_tokens=args.stability_margin_tokens,
                    )
                )

            async def run_sweep_background() -> None:
                await reset_cache()
                sweep_backgrounds.append(
                    await _run_background(
                        llm,
                        trace=sweep_trace,
                        overlap_seconds=args.size_sweep_overlap_seconds,
                        cleanup_delay_seconds=args.cleanup_delay_seconds,
                        sampling_params=sampling_params,
                        tokens_prompt_type=TokensPrompt,
                        streaming_input_type=StreamingInput,
                        stability_margin_tokens=args.stability_margin_tokens,
                        background_priority=args.background_priority,
                        cache_page_size_tokens=args.cache_page_size_tokens,
                    )
                )

            for _ in range(args.warmup_repeats):
                await run_sweep_control()
                sweep_controls.clear()
                await run_sweep_background()
                sweep_backgrounds.clear()

            for repeat_index in range(args.size_sweep_repeats):
                if repeat_index % 2 == 0:
                    await run_sweep_control()
                    await run_sweep_background()
                else:
                    await run_sweep_background()
                    await run_sweep_control()

            size_sweep_generations.extend(
                item.final_generation for item in (*sweep_controls, *sweep_backgrounds)
            )
            size_sweep_prefix_matches.extend(
                item.prefix_matched for item in sweep_backgrounds
            )
            size_sweep[str(candidate_chunk_tokens)] = {
                "first_candidate_tokens": len(sweep_trace.snapshots[0]),
                "summary": _summarize_pair(sweep_controls, sweep_backgrounds),
                "raw": {
                    "control": [asdict(item) for item in sweep_controls],
                    "background": [asdict(item) for item in sweep_backgrounds],
                },
            }

        all_generations = (
            [
                measurement.final_generation
                for measurements in controls.values()
                for measurement in measurements
            ]
            + [
                measurement.final_generation
                for measurements in backgrounds.values()
                for measurement in measurements
            ]
            + size_sweep_generations
        )
        if len({item.output_token_ids for item in all_generations}) != 1:
            raise RuntimeError("deterministic final output changed across cases")
        if any(
            not item.prefix_matched
            for measurements in backgrounds.values()
            for item in measurements
        ) or not all(size_sweep_prefix_matches):
            raise RuntimeError("background prefill committed a non-final prefix")
        contention_outputs = {
            item.foreground_generation.output_token_ids
            for measurements in contention.values()
            for item in measurements
        }
        if len(contention_outputs) != 1:
            raise RuntimeError("deterministic contention output changed across cases")

        equal_priority_ttft = [
            item.foreground_generation.time_to_first_token_seconds
            for item in contention[0]
        ]
        low_priority_ttft = [
            item.foreground_generation.time_to_first_token_seconds
            for item in contention[args.background_priority]
        ]
        foreground_priority_savings = [
            equal - low
            for equal, low in zip(equal_priority_ttft, low_priority_ttft, strict=True)
        ]

        result = {
            "model": args.model,
            "engine": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "scheduling_policy": "priority",
                "background_priority": args.background_priority,
                "cache_page_size_tokens": args.cache_page_size_tokens,
            },
            "trace": {
                "immutable_prefix_tokens": trace.immutable_prefix_tokens,
                "tool_output_tokens": trace.tool_output_tokens,
                "mutable_suffix_tokens": trace.mutable_suffix_tokens,
                "first_candidate_tokens": len(trace.snapshots[0]),
                "final_prompt_tokens": len(trace.final),
                "candidate_chunk_tokens": args.candidate_chunk_tokens,
                "stability_margin_tokens": args.stability_margin_tokens,
            },
            "overlap_seconds": args.overlap_seconds,
            "repeats": args.repeats,
            "warmup_repeats": args.warmup_repeats,
            "candidate_chunk_token_sweep": {
                "overlap_seconds": args.size_sweep_overlap_seconds,
                "repeats": args.size_sweep_repeats,
                "measurements": size_sweep,
            },
            "contention": {
                "repeats": args.contention_repeats,
                "foreground_prompt_tokens": len(foreground_prompt_token_ids),
                "equal_priority_foreground_ttft_seconds": _summary(equal_priority_ttft),
                "low_priority_foreground_ttft_seconds": _summary(low_priority_ttft),
                "paired_foreground_ttft_savings_seconds": _summary(
                    foreground_priority_savings
                ),
                "low_priority_win_fraction": statistics.fmean(
                    saving > 0 for saving in foreground_priority_savings
                ),
                "raw": {
                    str(priority): [asdict(item) for item in measurements]
                    for priority, measurements in contention.items()
                },
            },
            "summary": {
                str(overlap): _summarize_pair(
                    controls[str(overlap)], backgrounds[str(overlap)]
                )
                for overlap in args.overlap_seconds
            },
            "raw": {
                "control": {
                    overlap: [asdict(item) for item in measurements]
                    for overlap, measurements in controls.items()
                },
                "background": {
                    overlap: [asdict(item) for item in measurements]
                    for overlap, measurements in backgrounds.items()
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
