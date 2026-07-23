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

"""Replay recorded SWE prompts to isolate same-request prefill model-call gains."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from examples.swe_bench.benchmark_streaming_final_decode import (
    FinalGenerationMeasurement,
    _generate_final,
    _generate_same_request_final,
    _modal_control_contract,
    _output_signature,
    _paired_summary,
    _parse_nonnegative_floats,
    _parse_positive_ints,
    _stable_streamed_prefix,
    _warm_authoritative_prefix,
)
from examples.swe_bench.benchmark_streaming_tool_call_prefill import (
    PromptTrace,
    _summary,
)


@dataclass(frozen=True)
class RecordedPromptTrace:
    """One redacted, tokenized tool-output-to-next-model-call transition."""

    label: str
    prompt_sha256: str
    tool_output_chars: int
    final_prompt_chars: int
    trace: PromptTrace


@dataclass(frozen=True)
class _RecordedPromptPair:
    trajectory_index: int
    label: str
    tool_output: str
    final_prompt: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
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
    parser.add_argument("--trace-limit", type=int, default=16)
    parser.add_argument("--max-traces-per-trajectory", type=int, default=1)
    parser.add_argument("--snapshot-chars", type=int, default=256)
    parser.add_argument(
        "--snapshot-chars-sweep",
        type=_parse_positive_ints,
        default=(),
        help=(
            "Comma-separated snapshot sizes. All settings replay the same "
            "transitions selected as replayable at every requested size."
        ),
    )
    parser.add_argument(
        "--final-delay-seconds-sweep",
        type=_parse_nonnegative_floats,
        default=(0.0,),
        help=(
            "Comma-separated intervals available for background prefill before "
            "the authoritative final chunk is enqueued."
        ),
    )
    parser.add_argument("--stability-margin-tokens", type=int, default=8)
    parser.add_argument("--background-priority", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
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


def _common_prefix_length(*token_lists: Sequence[int]) -> int:
    if not token_lists:
        return 0
    common = min(len(tokens) for tokens in token_lists)
    for index in range(common):
        token = token_lists[0][index]
        if any(tokens[index] != token for tokens in token_lists[1:]):
            return index
    return common


def _next_function_call(
    output_items: Sequence[dict[str, Any]], start: int
) -> dict[str, Any] | None:
    for item in output_items[start:]:
        if item.get("type") == "function_call_output":
            return None
        if item.get("type") == "function_call":
            return item
    return None


def _recorded_prompt_pairs(
    trajectory_jsonl: Path,
) -> Iterable[_RecordedPromptPair]:
    with trajectory_jsonl.open() as source:
        for trajectory_index, line in enumerate(source):
            trajectory = json.loads(line)
            output_items = (trajectory.get("response") or {}).get("output") or []
            for output_index, item in enumerate(output_items):
                if item.get("type") != "function_call_output":
                    continue
                next_call = _next_function_call(output_items, output_index + 1)
                if next_call is None:
                    continue
                tool_output = item.get("output")
                final_prompt = next_call.get("prompt_str")
                if not isinstance(tool_output, str) or not isinstance(
                    final_prompt, str
                ):
                    continue
                yield _RecordedPromptPair(
                    trajectory_index=trajectory_index,
                    label=f"trajectory-{trajectory_index}-output-{output_index}",
                    tool_output=tool_output,
                    final_prompt=final_prompt,
                )


def _build_recorded_prompt_trace(
    tokenizer: Any,
    *,
    label: str,
    tool_output: str,
    final_prompt: str,
    snapshot_chars: int,
    stability_margin_tokens: int,
    max_model_len: int,
) -> tuple[RecordedPromptTrace | None, str | None]:
    if len(tool_output) <= snapshot_chars:
        return None, "tool_output_not_longer_than_snapshot"
    tool_output_start = final_prompt.rfind(tool_output)
    if tool_output_start < 0:
        return None, "tool_output_not_in_next_prompt"

    prefix = final_prompt[:tool_output_start]
    suffix = final_prompt[tool_output_start + len(tool_output) :]
    initial_text = prefix + suffix
    snapshot_text = prefix + tool_output[:snapshot_chars] + suffix
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    initial = encode(initial_text)
    snapshot = encode(snapshot_text)
    final = encode(final_prompt)
    if not initial or not snapshot or not final:
        return None, "empty_tokenization"
    if len(final) + 1 > max_model_len:
        return None, "final_prompt_too_long"

    immutable_prefix_tokens = _common_prefix_length(initial, snapshot, final)
    stable_candidate_tokens = _common_prefix_length(snapshot, final)
    if stable_candidate_tokens - stability_margin_tokens <= immutable_prefix_tokens:
        return None, "no_stable_dynamic_tokens"
    mutable_suffix_tokens = len(snapshot) - stable_candidate_tokens
    trace = PromptTrace(
        immutable_prefix_tokens=immutable_prefix_tokens,
        tool_output_tokens=max(0, len(final) - len(initial)),
        mutable_suffix_tokens=mutable_suffix_tokens,
        authoritative_prefix=final[:immutable_prefix_tokens],
        initial=initial,
        snapshots=(snapshot,),
        final=final,
    )
    _stable_streamed_prefix(
        trace,
        stability_margin_tokens=stability_margin_tokens,
    )
    return (
        RecordedPromptTrace(
            label=label,
            prompt_sha256=hashlib.sha256(final_prompt.encode()).hexdigest(),
            tool_output_chars=len(tool_output),
            final_prompt_chars=len(final_prompt),
            trace=trace,
        ),
        None,
    )


def load_recorded_prompt_traces(
    tokenizer: Any,
    *,
    trajectory_jsonl: Path,
    trace_limit: int,
    max_traces_per_trajectory: int,
    snapshot_chars: int,
    stability_margin_tokens: int,
    max_model_len: int,
) -> tuple[list[RecordedPromptTrace], dict[str, int]]:
    if trace_limit <= 0:
        raise ValueError("trace_limit must be positive")
    if max_traces_per_trajectory <= 0:
        raise ValueError("max_traces_per_trajectory must be positive")
    skipped: Counter[str] = Counter()
    selected_per_trajectory: Counter[int] = Counter()
    traces = []
    for pair in _recorded_prompt_pairs(trajectory_jsonl):
        if selected_per_trajectory[pair.trajectory_index] >= max_traces_per_trajectory:
            skipped["per_trajectory_limit"] += 1
            continue
        recorded, reason = _build_recorded_prompt_trace(
            tokenizer,
            label=pair.label,
            tool_output=pair.tool_output,
            final_prompt=pair.final_prompt,
            snapshot_chars=snapshot_chars,
            stability_margin_tokens=stability_margin_tokens,
            max_model_len=max_model_len,
        )
        if recorded is None:
            assert reason is not None
            skipped[reason] += 1
            continue
        traces.append(recorded)
        selected_per_trajectory[pair.trajectory_index] += 1
        if len(traces) == trace_limit:
            break
    return traces, dict(sorted(skipped.items()))


def load_recorded_prompt_trace_sweep(
    tokenizer: Any,
    *,
    trajectory_jsonl: Path,
    trace_limit: int,
    max_traces_per_trajectory: int,
    snapshot_chars_values: Sequence[int],
    stability_margin_tokens: int,
    max_model_len: int,
) -> tuple[dict[int, list[RecordedPromptTrace]], dict[str, int]]:
    """Select one common set of transitions replayable at every snapshot size."""
    if not snapshot_chars_values or any(value <= 0 for value in snapshot_chars_values):
        raise ValueError("snapshot_chars_values must contain positive values")
    if trace_limit <= 0:
        raise ValueError("trace_limit must be positive")
    if max_traces_per_trajectory <= 0:
        raise ValueError("max_traces_per_trajectory must be positive")

    skipped: Counter[str] = Counter()
    selected_per_trajectory: Counter[int] = Counter()
    traces_by_snapshot = {value: [] for value in snapshot_chars_values}
    for pair in _recorded_prompt_pairs(trajectory_jsonl):
        if selected_per_trajectory[pair.trajectory_index] >= max_traces_per_trajectory:
            skipped["per_trajectory_limit"] += 1
            continue
        recorded_by_snapshot: dict[int, RecordedPromptTrace] = {}
        rejection: tuple[int, str] | None = None
        for snapshot_chars in snapshot_chars_values:
            recorded, reason = _build_recorded_prompt_trace(
                tokenizer,
                label=pair.label,
                tool_output=pair.tool_output,
                final_prompt=pair.final_prompt,
                snapshot_chars=snapshot_chars,
                stability_margin_tokens=stability_margin_tokens,
                max_model_len=max_model_len,
            )
            if recorded is None:
                assert reason is not None
                rejection = (snapshot_chars, reason)
                break
            recorded_by_snapshot[snapshot_chars] = recorded
        if rejection is not None:
            snapshot_chars, reason = rejection
            skipped[f"snapshot_{snapshot_chars}:{reason}"] += 1
            continue
        for snapshot_chars, recorded in recorded_by_snapshot.items():
            traces_by_snapshot[snapshot_chars].append(recorded)
        selected_per_trajectory[pair.trajectory_index] += 1
        if len(next(iter(traces_by_snapshot.values()))) == trace_limit:
            break
    return traces_by_snapshot, dict(sorted(skipped.items()))


def audit_recorded_prompt_coverage(
    tokenizer: Any,
    *,
    trajectory_jsonl: Path,
    snapshot_chars_values: Sequence[int],
    stability_margin_tokens: int,
    max_model_len: int,
) -> dict[str, Any]:
    """Count replay eligibility without recording prompt or tool-output text."""
    total_pairs = 0
    trajectories_with_pairs: set[int] = set()
    replayable_trajectories = {value: set() for value in snapshot_chars_values}
    replayable_pairs: Counter[int] = Counter()
    rejections = {value: Counter() for value in snapshot_chars_values}
    for pair in _recorded_prompt_pairs(trajectory_jsonl):
        total_pairs += 1
        trajectories_with_pairs.add(pair.trajectory_index)
        for snapshot_chars in snapshot_chars_values:
            recorded, reason = _build_recorded_prompt_trace(
                tokenizer,
                label=pair.label,
                tool_output=pair.tool_output,
                final_prompt=pair.final_prompt,
                snapshot_chars=snapshot_chars,
                stability_margin_tokens=stability_margin_tokens,
                max_model_len=max_model_len,
            )
            if recorded is None:
                assert reason is not None
                rejections[snapshot_chars][reason] += 1
                continue
            replayable_pairs[snapshot_chars] += 1
            replayable_trajectories[snapshot_chars].add(pair.trajectory_index)
    return {
        "next_model_call_pairs": total_pairs,
        "trajectories_with_pairs": len(trajectories_with_pairs),
        "by_snapshot_chars": {
            str(snapshot_chars): {
                "replayable_pairs": replayable_pairs[snapshot_chars],
                "replayable_pair_fraction": (
                    replayable_pairs[snapshot_chars] / total_pairs
                    if total_pairs
                    else 0.0
                ),
                "replayable_trajectories": len(replayable_trajectories[snapshot_chars]),
                "rejections": dict(sorted(rejections[snapshot_chars].items())),
            }
            for snapshot_chars in snapshot_chars_values
        },
    }


def _replay_output_contract(
    controls: Sequence[FinalGenerationMeasurement],
    candidates: Sequence[FinalGenerationMeasurement],
) -> dict[str, Any]:
    """Validate repeated candidates against normal requests' modal output."""
    if len(controls) != len(candidates) or not controls:
        raise ValueError("control and candidate measurements must be non-empty pairs")
    if len(controls) < 3:
        matches = [
            _output_signature(control) == _output_signature(candidate)
            for control, candidate in zip(controls, candidates, strict=True)
        ]
        return {
            "method": "paired_exact",
            "control_modal_available": False,
            "candidate_matches_control_modal": sum(matches),
            "candidate_count": len(candidates),
            "match_fraction": statistics.fmean(matches),
            "all": all(matches),
        }

    contracts = [
        _modal_control_contract(controls, candidate) for candidate in candidates
    ]
    modal_available = all(item["modal_available"] for item in contracts)
    match_count = sum(bool(item["candidate_matches_modal"]) for item in contracts)
    required_matches = len(candidates) // 2 + 1
    return {
        "method": "modal_control",
        "control_modal_available": modal_available,
        "control_modal_count": contracts[0]["modal_count"],
        "candidate_matches_control_modal": match_count,
        "candidate_count": len(candidates),
        "required_candidate_matches": required_matches,
        "match_fraction": match_count / len(candidates),
        "all": modal_available and match_count >= required_matches,
    }


async def _main(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.logger import init_logger
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    from nemo_rl.models.generation.vllm.patches import (
        _patch_vllm_streaming_session_max_tokens,
        _patch_vllm_streaming_session_output_state,
        _patch_vllm_streaming_session_priority,
        _patch_vllm_strict_priority_scheduling,
    )

    if args.repeats <= 0 or args.warmup_repeats < 0:
        raise ValueError("repeats must be positive and warmup_repeats non-negative")
    snapshot_chars_values = args.snapshot_chars_sweep or (args.snapshot_chars,)
    if any(value <= 0 for value in snapshot_chars_values):
        raise ValueError("snapshot chars must be positive")

    patch_logger = init_logger("streaming_trace_replay_patch")
    patch_results = (
        _patch_vllm_streaming_session_max_tokens(patch_logger),
        _patch_vllm_streaming_session_output_state(patch_logger),
        _patch_vllm_streaming_session_priority(patch_logger),
        _patch_vllm_strict_priority_scheduling(patch_logger),
    )
    if not all(patch_results):
        raise RuntimeError("installed vLLM lacks required streaming patches")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    coverage = audit_recorded_prompt_coverage(
        tokenizer,
        trajectory_jsonl=args.trajectory_jsonl,
        snapshot_chars_values=snapshot_chars_values,
        stability_margin_tokens=args.stability_margin_tokens,
        max_model_len=args.max_model_len,
    )
    traces_by_snapshot, skipped = load_recorded_prompt_trace_sweep(
        tokenizer,
        trajectory_jsonl=args.trajectory_jsonl,
        trace_limit=args.trace_limit,
        max_traces_per_trajectory=args.max_traces_per_trajectory,
        snapshot_chars_values=snapshot_chars_values,
        stability_margin_tokens=args.stability_margin_tokens,
        max_model_len=args.max_model_len,
    )
    if not traces_by_snapshot or not all(traces_by_snapshot.values()):
        raise RuntimeError("trajectory file yielded no replayable prompt traces")
    common_labels = [
        recorded.label for recorded in traces_by_snapshot[snapshot_chars_values[0]]
    ]
    if any(
        [recorded.label for recorded in traces] != common_labels
        for traces in traces_by_snapshot.values()
    ):
        raise RuntimeError("snapshot sweep did not select identical prompt transitions")

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
    prefill_sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        seed=args.seed,
        max_tokens=1,
        output_kind=RequestOutputKind.DELTA,
    )
    final_sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        seed=args.seed,
        max_tokens=1,
        logprobs=0,
        output_kind=RequestOutputKind.DELTA,
    )

    async def reset_cache() -> None:
        deadline = asyncio.get_running_loop().time() + 5
        while not await llm.reset_prefix_cache():
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("vLLM prefix cache remained busy")
            await asyncio.sleep(0.05)

    async def prepare(trace: PromptTrace) -> None:
        await reset_cache()
        await _warm_authoritative_prefix(
            llm,
            trace=trace,
            sampling_params=prefill_sampling_params,
            tokens_prompt_type=TokensPrompt,
            streaming_input_type=StreamingInput,
            stability_margin_tokens=args.stability_margin_tokens,
        )

    async def run_setting(
        traces: Sequence[RecordedPromptTrace],
        *,
        snapshot_chars: int,
        final_delay_seconds: float,
    ) -> dict[str, Any]:
        controls: list[FinalGenerationMeasurement] = []
        candidates: list[FinalGenerationMeasurement] = []
        measurements = []
        output_contracts = []
        contract_failures = []
        for trace_index, recorded in enumerate(traces):
            trace_controls: list[FinalGenerationMeasurement] = []
            trace_candidates: list[FinalGenerationMeasurement] = []
            trace_measurements = []
            total_iterations = args.warmup_repeats + args.repeats
            for iteration in range(total_iterations):
                arm_order = ("control", "candidate")
                if (trace_index + iteration) % 2:
                    arm_order = tuple(reversed(arm_order))
                iteration_results: dict[str, FinalGenerationMeasurement] = {}
                same_request_details: dict[str, Any] | None = None
                for arm in arm_order:
                    await prepare(recorded.trace)
                    if arm == "control":
                        result = await _generate_final(
                            llm,
                            prompt_token_ids=recorded.trace.final,
                            sampling_params=final_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            tokenizer=tokenizer,
                            output_kind="delta",
                        )
                    else:
                        same_request = await _generate_same_request_final(
                            llm,
                            trace=recorded.trace,
                            prefill_sampling_params=prefill_sampling_params,
                            final_sampling_params=final_sampling_params,
                            tokens_prompt_type=TokensPrompt,
                            streaming_input_type=StreamingInput,
                            tokenizer=tokenizer,
                            stability_margin_tokens=args.stability_margin_tokens,
                            background_priority=args.background_priority,
                            prefill_has_logprobs=False,
                            prefill_chunk_count=1,
                            final_before_prefill_ack=True,
                            final_delay_seconds=final_delay_seconds,
                        )
                        result = same_request.final_generation
                        same_request_details = {
                            "streamed_prefix_tokens": (
                                same_request.streamed_prefix_tokens
                            ),
                            "final_suffix_tokens": same_request.final_suffix_tokens,
                            "prefill_chunks": same_request.prefill_chunks,
                            "dummy_token_count": len(same_request.dummy_token_ids),
                            "prefill_cached_tokens": (
                                same_request.prefill_cached_tokens
                            ),
                            "prefill_to_dummy_seconds": (
                                same_request.prefill_to_dummy_seconds
                            ),
                            "prefill_output_suppressed": (
                                same_request.prefill_output_suppressed
                            ),
                        }
                    iteration_results[arm] = result
                if iteration < args.warmup_repeats:
                    continue

                control = iteration_results["control"]
                candidate = iteration_results["candidate"]
                pair_parity = _output_signature(control) == _output_signature(candidate)
                trace_controls.append(control)
                trace_candidates.append(candidate)
                controls.append(control)
                candidates.append(candidate)
                stable_prefix_tokens = len(
                    _stable_streamed_prefix(
                        recorded.trace,
                        stability_margin_tokens=args.stability_margin_tokens,
                    )
                )
                trace_measurements.append(
                    {
                        "repeat": iteration - args.warmup_repeats,
                        "pair_output_parity": pair_parity,
                        "control": asdict(control),
                        "candidate": asdict(candidate),
                        "same_request": same_request_details,
                    }
                )

            contract = _replay_output_contract(trace_controls, trace_candidates)
            output_contracts.append(contract)
            if not contract["all"]:
                contract_failures.append(recorded.label)
            measurements.append(
                {
                    "label": recorded.label,
                    "prompt_sha256": recorded.prompt_sha256,
                    "tool_output_chars": recorded.tool_output_chars,
                    "final_prompt_chars": recorded.final_prompt_chars,
                    "initial_prompt_tokens": len(recorded.trace.initial),
                    "final_prompt_tokens": len(recorded.trace.final),
                    "stable_streamed_prefix_tokens": stable_prefix_tokens,
                    "stable_dynamic_tokens": (
                        stable_prefix_tokens - recorded.trace.immutable_prefix_tokens
                    ),
                    "output_contract": contract,
                    "repeats": trace_measurements,
                }
            )

        ttft_savings = [
            control.time_to_first_output_seconds
            - candidate.time_to_first_output_seconds
            for control, candidate in zip(controls, candidates, strict=True)
        ]
        pair_parities = [
            repeat["pair_output_parity"]
            for measurement in measurements
            for repeat in measurement["repeats"]
        ]
        candidate_prefill_to_dummy = [
            repeat["same_request"]["prefill_to_dummy_seconds"]
            for measurement in measurements
            for repeat in measurement["repeats"]
            if repeat["same_request"]["prefill_to_dummy_seconds"] is not None
        ]
        output_contract_passes = [bool(item["all"]) for item in output_contracts]
        return {
            "snapshot_chars": snapshot_chars,
            "final_delay_seconds": final_delay_seconds,
            "loaded_traces": len(traces),
            "paired": _paired_summary(controls, candidates),
            "paired_ttft_savings_seconds": _summary(ttft_savings),
            "paired_ttft_win_fraction": statistics.fmean(
                saving > 0 for saving in ttft_savings
            ),
            "pair_output_parity_fraction": statistics.fmean(pair_parities),
            "output_contract_pass_fraction": statistics.fmean(output_contract_passes),
            "output_contract_failures": contract_failures,
            "prefill_to_dummy_seconds": (
                _summary(candidate_prefill_to_dummy)
                if candidate_prefill_to_dummy
                else None
            ),
            "measurements": measurements,
        }

    settings = []
    try:
        for snapshot_chars in snapshot_chars_values:
            for final_delay_seconds in args.final_delay_seconds_sweep:
                print(
                    "[SETTING] "
                    f"snapshot_chars={snapshot_chars} "
                    f"final_delay_seconds={final_delay_seconds}",
                    flush=True,
                )
                settings.append(
                    await run_setting(
                        traces_by_snapshot[snapshot_chars],
                        snapshot_chars=snapshot_chars,
                        final_delay_seconds=final_delay_seconds,
                    )
                )

        all_contract_failures = [
            {
                "snapshot_chars": setting["snapshot_chars"],
                "final_delay_seconds": setting["final_delay_seconds"],
                "labels": setting["output_contract_failures"],
            }
            for setting in settings
            if setting["output_contract_failures"]
        ]
        result = {
            "source": {
                "trajectory_jsonl": str(args.trajectory_jsonl),
                "trace_limit": args.trace_limit,
                "max_traces_per_trajectory": args.max_traces_per_trajectory,
                "loaded_traces": len(common_labels),
                "skipped_before_limit": skipped,
                "coverage": coverage,
                "common_trace_labels": common_labels,
                "snapshot_chars_values": snapshot_chars_values,
            },
            "model": args.model,
            "engine": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "enforce_eager": args.enforce_eager,
                "async_scheduling": args.async_scheduling,
                "mamba_cache_mode": args.mamba_cache_mode,
                "compilation_backend": args.compilation_backend,
                "background_priority": args.background_priority,
            },
            "repeats": args.repeats,
            "warmup_repeats": args.warmup_repeats,
            "settings": settings,
            "output_contract_failures": all_contract_failures,
        }
        serialized = json.dumps(result, indent=2, sort_keys=True)
        print(serialized, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        if args.strict_parity and all_contract_failures:
            raise RuntimeError("same-request replay changed generated output")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
