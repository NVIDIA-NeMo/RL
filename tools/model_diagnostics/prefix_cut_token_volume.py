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
"""Read-only token-volume controls for the generation-only prefix benchmark."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaptureTokenHeader:
    """Small capture metadata copied without retaining engine token arrays."""

    engine_request_id: str | None
    prompt_tokens: int
    staged_generated_tokens: int
    has_staged_chunks: bool


def count_engine_token_volume(
    headers: Sequence[CaptureTokenHeader],
    processor: Any,
    retained: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Count lengths on the engine loop while its publication lock is held.

    This performs no token/logprob copies, route assembly, encoding or writes.
    Completed-but-not-staged requests use the same retained state as real cuts.
    The benchmark sends single-completion root calls and never restores them.
    """
    result = {
        "calls": len(headers),
        "positive_calls": 0,
        "not_in_engine": 0,
        "cumulative_generation_tokens": 0,
        "new_token_positions": 0,
        "unstaged_prompt_tokens": 0,
        "max_generated_tokens": 0,
    }
    for header in headers:
        request_id = header.engine_request_id
        if request_id is None:
            result["not_in_engine"] += 1
            continue
        completed = retained.get(request_id, {})
        internal_ids = (
            set(processor.external_req_ids.get(request_id, ())) | completed.keys()
        )
        if len(internal_ids) > 1:
            raise ValueError(
                "Token-volume benchmark requires one engine completion per call"
            )
        if not internal_ids:
            result["not_in_engine"] += 1
            continue
        internal_id = next(iter(internal_ids))
        saved = completed.get(internal_id)
        state = (
            saved.request_state
            if saved is not None
            else processor.request_states.get(internal_id)
        )
        if state is None:
            result["not_in_engine"] += 1
            continue
        generated = len(state.detokenizer.output_token_ids)
        logprobs = state.logprobs_processor.logprobs
        if generated != (len(logprobs) if logprobs is not None else 0):
            raise ValueError("Published engine token/logprob lengths differ")
        if generated < header.staged_generated_tokens:
            raise ValueError("Engine generation became shorter than its durable prefix")
        if generated == 0:
            continue
        prompt = 0 if header.has_staged_chunks else header.prompt_tokens
        result["positive_calls"] += 1
        result["cumulative_generation_tokens"] += generated
        result["new_token_positions"] += (
            generated - header.staged_generated_tokens + prompt
        )
        result["unstaged_prompt_tokens"] += prompt
        result["max_generated_tokens"] = max(result["max_generated_tokens"], generated)
    return result


async def wait_for_token_volume(
    collect: Callable[[], Awaitable[list[dict[str, int]]]],
    *,
    cumulative_tokens: int,
    new_token_positions: int,
    timeout_s: float,
    poll_s: float,
    check_healthy: Callable[[], None],
    report: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Wait for every engine to reach both targets before starting cut timing."""
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        check_healthy()
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Benchmark did not reach its token-volume targets")
        async with asyncio.timeout(remaining):
            workers = await collect()
        ready = bool(workers) and all(
            worker["cumulative_generation_tokens"] >= cumulative_tokens
            and worker["new_token_positions"] >= new_token_positions
            for worker in workers
        )
        observation = {
            "elapsed_s": time.monotonic() - started,
            "ready": ready,
            "cumulative_token_target": cumulative_tokens,
            "new_token_position_target": new_token_positions,
            "workers": workers,
        }
        report(observation)
        if ready:
            return observation
        await asyncio.sleep(min(poll_s, remaining))
    raise TimeoutError("Benchmark did not reach its token-volume targets")


def token_match_report(
    actual: Mapping[str, int],
    *,
    cumulative_tokens: int,
    new_token_positions: int,
    tolerance: float,
) -> dict[str, Any]:
    """Judge actual cut counters, rather than treating a readiness sample as a match."""
    targets = {
        "cumulative_generation_tokens": cumulative_tokens,
        "batch_token_positions": new_token_positions,
    }
    errors = {
        name: abs(actual.get(name, 0) / target - 1.0)
        for name, target in targets.items()
        if target > 0
    }
    return {
        "actual": dict(actual),
        "targets": targets,
        "relative_errors": errors,
        "tolerance": tolerance,
        "matched": bool(errors)
        and all(error <= tolerance for error in errors.values()),
    }
