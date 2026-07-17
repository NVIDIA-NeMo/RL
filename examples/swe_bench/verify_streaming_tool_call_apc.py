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

"""Verify live vLLM APC reuse after streaming tool-call prefill."""

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from nemo_rl.models.generation.vllm.streaming_tool_call import (
    StreamingToolCallPrefillManager,
)


@dataclass(frozen=True)
class GenerationMeasurement:
    """Cache and latency measurements from one authoritative generation."""

    cached_tokens: int
    prompt_tokens: int
    time_to_first_token_seconds: float


@dataclass(frozen=True)
class StreamingMeasurement:
    """Streaming-prefill state and its following generation measurement."""

    cleanup_delay_seconds: float
    committed_tokens: int
    completed_chunks: int
    dummy_tokens: int
    prefix_matched: bool
    final_generation: GenerationMeasurement


@dataclass(frozen=True)
class PrefixCacheCounters:
    """Cumulative vLLM prefix-cache token counters."""

    queries: int
    hits: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled live-vLLM experiment that measures the exact "
            "number of APC-cached tokens reused by a final request."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prompt-tokens", type=int, default=4097)
    parser.add_argument("--candidate-chunk-tokens", type=int, default=512)
    parser.add_argument("--stability-margin-tokens", type=int, default=8)
    parser.add_argument("--cleanup-delay-seconds", type=float, default=0.1)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


async def _generate_one_token(
    llm: Any,
    *,
    prompt_token_ids: list[int],
    sampling_params: Any,
    tokens_prompt_type: type,
) -> GenerationMeasurement:
    request_id = f"streaming-tool-call-apc-check-{uuid.uuid4()}"
    start_time = time.perf_counter()
    first_token_time: float | None = None
    cached_tokens: int | None = None

    result_generator = llm.generate(
        prompt=tokens_prompt_type(prompt_token_ids=prompt_token_ids),
        sampling_params=sampling_params,
        request_id=request_id,
    )
    async for request_output in result_generator:
        if request_output.num_cached_tokens is not None:
            cached_tokens = request_output.num_cached_tokens
        if first_token_time is None and any(
            completion_output.token_ids for completion_output in request_output.outputs
        ):
            first_token_time = time.perf_counter()

    if cached_tokens is None:
        raise RuntimeError("vLLM did not report num_cached_tokens")
    if first_token_time is None:
        raise RuntimeError("vLLM did not generate the expected dummy token")

    return GenerationMeasurement(
        cached_tokens=cached_tokens,
        prompt_tokens=len(prompt_token_ids),
        time_to_first_token_seconds=first_token_time - start_time,
    )


def _make_prompt_token_ids(tokenizer: Any, prompt_tokens: int) -> list[int]:
    seed_text = (
        "Streaming tool calls should reuse the exact committed token prefix. "
        "This sentence intentionally repeats to create a controlled long prompt. "
    )
    seed_token_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_token_ids:
        raise RuntimeError("tokenizer returned an empty seed")
    repetitions = (prompt_tokens + len(seed_token_ids) - 1) // len(seed_token_ids)
    return (seed_token_ids * repetitions)[:prompt_tokens]


def _read_prefix_cache_counters() -> PrefixCacheCounters:
    from vllm.v1.metrics.reader import Counter, get_metrics_snapshot

    counter_values: dict[str, int] = {}
    for metric in get_metrics_snapshot():
        if isinstance(metric, Counter):
            counter_values[metric.name] = counter_values.get(metric.name, 0) + int(
                metric.value
            )
    return PrefixCacheCounters(
        queries=counter_values.get("vllm:prefix_cache_queries", 0),
        hits=counter_values.get("vllm:prefix_cache_hits", 0),
    )


def _counter_delta(
    after: PrefixCacheCounters, before: PrefixCacheCounters
) -> PrefixCacheCounters:
    return PrefixCacheCounters(
        queries=after.queries - before.queries,
        hits=after.hits - before.hits,
    )


async def _run_streaming_case(
    llm: Any,
    *,
    final_prompt_token_ids: list[int],
    candidate_chunk_tokens: int,
    stability_margin_tokens: int,
    cleanup_delay_seconds: float,
    sampling_params: Any,
    tokens_prompt_type: type,
    streaming_input_type: type,
) -> StreamingMeasurement:
    prefill_sampling_params = sampling_params

    def generate_prefill(
        input_stream: AsyncIterator[Any], request_id: str
    ) -> AsyncGenerator[Any, None]:
        return llm.generate(
            prompt=input_stream,
            sampling_params=prefill_sampling_params,
            request_id=request_id,
        )

    def make_streaming_input(token_ids: list[int]) -> Any:
        return streaming_input_type(
            prompt=tokens_prompt_type(prompt_token_ids=token_ids),
            sampling_params=prefill_sampling_params,
        )

    manager = StreamingToolCallPrefillManager(
        generate=generate_prefill,
        make_streaming_input=make_streaming_input,
        count_output_tokens=lambda output: sum(
            len(completion_output.token_ids) for completion_output in output.outputs
        ),
        max_sessions=1,
        session_ttl_seconds=60,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=len(final_prompt_token_ids),
    )
    session_id = f"apc-check-{uuid.uuid4()}"
    candidate_lengths = list(
        range(
            candidate_chunk_tokens,
            len(final_prompt_token_ids),
            candidate_chunk_tokens,
        )
    )
    if len(candidate_lengths) < 2:
        raise ValueError(
            "prompt_tokens must contain at least two candidate chunks before close"
        )

    await manager.start(
        session_id=session_id,
        prompt_token_ids=final_prompt_token_ids[: candidate_lengths[0]],
        sequence_no=0,
    )
    for sequence_no, candidate_length in enumerate(candidate_lengths[1:], start=1):
        await manager.append(
            session_id=session_id,
            prompt_token_ids=final_prompt_token_ids[:candidate_length],
            sequence_no=sequence_no,
        )

    close_result = await manager.close(
        session_id=session_id,
        final_prompt_token_ids=final_prompt_token_ids,
    )
    if cleanup_delay_seconds:
        await asyncio.sleep(cleanup_delay_seconds)

    final_generation = await _generate_one_token(
        llm,
        prompt_token_ids=final_prompt_token_ids,
        sampling_params=sampling_params,
        tokens_prompt_type=tokens_prompt_type,
    )
    return StreamingMeasurement(
        cleanup_delay_seconds=cleanup_delay_seconds,
        committed_tokens=close_result.committed_tokens,
        completed_chunks=close_result.completed_chunks,
        dummy_tokens=close_result.dummy_tokens,
        prefix_matched=close_result.prefix_matched,
        final_generation=final_generation,
    )


async def _main(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.metrics.loggers import PrometheusStatLogger

    if args.prompt_tokens >= args.max_model_len:
        raise ValueError("prompt_tokens must be smaller than max_model_len")
    if args.candidate_chunk_tokens <= args.stability_margin_tokens:
        raise ValueError(
            "candidate_chunk_tokens must be larger than stability_margin_tokens"
        )

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
    )
    llm = AsyncLLM.from_engine_args(
        engine_args,
        stat_loggers=[PrometheusStatLogger],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = _make_prompt_token_ids(tokenizer, args.prompt_tokens)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=1,
        max_tokens=1,
        output_kind=RequestOutputKind.DELTA,
    )

    try:
        await llm.reset_prefix_cache()
        before_direct_cold = _read_prefix_cache_counters()
        direct_cold = await _generate_one_token(
            llm,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            tokens_prompt_type=TokensPrompt,
        )
        after_direct_cold = _read_prefix_cache_counters()
        direct_warm = await _generate_one_token(
            llm,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            tokens_prompt_type=TokensPrompt,
        )
        after_direct_warm = _read_prefix_cache_counters()

        await llm.reset_prefix_cache()
        before_streaming_immediate = _read_prefix_cache_counters()
        streaming_immediate = await _run_streaming_case(
            llm,
            final_prompt_token_ids=prompt_token_ids,
            candidate_chunk_tokens=args.candidate_chunk_tokens,
            stability_margin_tokens=args.stability_margin_tokens,
            cleanup_delay_seconds=0,
            sampling_params=sampling_params,
            tokens_prompt_type=TokensPrompt,
            streaming_input_type=StreamingInput,
        )
        after_streaming_immediate = _read_prefix_cache_counters()

        await llm.reset_prefix_cache()
        before_streaming_delayed = _read_prefix_cache_counters()
        streaming_delayed = await _run_streaming_case(
            llm,
            final_prompt_token_ids=prompt_token_ids,
            candidate_chunk_tokens=args.candidate_chunk_tokens,
            stability_margin_tokens=args.stability_margin_tokens,
            cleanup_delay_seconds=args.cleanup_delay_seconds,
            sampling_params=sampling_params,
            tokens_prompt_type=TokensPrompt,
            streaming_input_type=StreamingInput,
        )
        after_streaming_delayed = _read_prefix_cache_counters()

        prefix_cache_counter_deltas = {
            "direct_cold": _counter_delta(after_direct_cold, before_direct_cold),
            "direct_warm": _counter_delta(after_direct_warm, after_direct_cold),
            "streaming_immediate": _counter_delta(
                after_streaming_immediate, before_streaming_immediate
            ),
            "streaming_delayed": _counter_delta(
                after_streaming_delayed, before_streaming_delayed
            ),
        }

        result = {
            "model": args.model,
            "direct_cold": asdict(direct_cold),
            "direct_warm": asdict(direct_warm),
            "streaming_immediate": asdict(streaming_immediate),
            "streaming_delayed": asdict(streaming_delayed),
            "prefix_cache_counter_deltas": {
                case_name: asdict(counter_delta)
                for case_name, counter_delta in prefix_cache_counter_deltas.items()
            },
        }
        print(json.dumps(result, indent=2), flush=True)

        if direct_cold.cached_tokens != 0:
            raise RuntimeError("cold-cache control unexpectedly reported cached tokens")
        if direct_warm.cached_tokens == 0:
            raise RuntimeError("warm-cache control did not reuse any cached tokens")
        if not streaming_immediate.prefix_matched:
            raise RuntimeError("streaming prefill did not match the final prompt")
        immediate_cached_tokens = streaming_immediate.final_generation.cached_tokens
        delayed_cached_tokens = streaming_delayed.final_generation.cached_tokens
        if immediate_cached_tokens == 0:
            raise RuntimeError(
                "streaming prefill did not produce immediately reusable APC blocks"
            )
        if delayed_cached_tokens == 0:
            raise RuntimeError(
                "streaming prefill did not produce reusable APC blocks even after cleanup"
            )
        if immediate_cached_tokens != delayed_cached_tokens:
            raise RuntimeError(
                "APC reuse changed after the cleanup delay: "
                f"immediate={immediate_cached_tokens}, delayed={delayed_cached_tokens}"
            )
        if immediate_cached_tokens > streaming_immediate.committed_tokens:
            raise RuntimeError(
                "the final request reused more tokens than the streaming session committed"
            )
        if prefix_cache_counter_deltas["direct_cold"].hits != 0:
            raise RuntimeError("cold-cache control unexpectedly incremented cache hits")
        if prefix_cache_counter_deltas["direct_warm"].hits == 0:
            raise RuntimeError("warm-cache control did not increment cache hits")
        if prefix_cache_counter_deltas["streaming_immediate"].hits == 0:
            raise RuntimeError("streaming prefill did not increment cache hits")
        if (
            prefix_cache_counter_deltas["streaming_immediate"]
            != prefix_cache_counter_deltas["streaming_delayed"]
        ):
            raise RuntimeError(
                "streaming prefix-cache counters changed after the cleanup delay"
            )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
