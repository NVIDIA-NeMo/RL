#!/usr/bin/env python3
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

"""Validate and summarize NeMo-RL Chrome-trace JSONL timeline files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Fields a successful model_call carries when the request went through the
# PD-disaggregated TRT-LLM server (frontend + context-engine leg).
DISAGG_CTX_FIELDS = (
    "frontend_recv_ts_us",
    "frontend_fwd_ts_us",
    "ctx_queued_ts_us",
    "ctx_first_scheduled_ts_us",
    "ctx_done_ts_us",
    "ctx_cached_prompt_tokens",
    "ctx_computed_tokens",
    "ctx_num_chunks",
)


def validate_file(
    path: Path,
) -> tuple[int, Counter[str], Counter[str], int, int, int, int, int, int, list[str]]:
    count = 0
    names: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    successful_model_calls = 0
    generation_replica_model_calls = 0
    token_count_model_calls = 0
    cache_metric_model_calls = 0
    scheduling_timestamp_model_calls = 0
    disagg_ctx_model_calls = 0
    generation_replica_indices: set[int] = set()
    errors: list[str] = []
    try:
        lines = path.open("r", encoding="utf-8")
    except OSError as exc:
        return (
            0,
            names,
            categories,
            successful_model_calls,
            generation_replica_model_calls,
            token_count_model_calls,
            cache_metric_model_calls,
            scheduling_timestamp_model_calls,
            disagg_ctx_model_calls,
            [f"{path}: cannot open: {exc}"],
        )

    with lines:
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(event, dict):
                errors.append(f"{path}:{lineno}: event is not an object")
                continue
            for field in ("name", "cat", "ph", "ts", "dur", "pid", "tid", "args"):
                if field not in event:
                    errors.append(f"{path}:{lineno}: missing {field}")
            if event.get("ph") != "X":
                errors.append(f"{path}:{lineno}: ph must be X")
            if not isinstance(event.get("name"), str) or not event.get("name"):
                errors.append(f"{path}:{lineno}: name must be non-empty")
            if not isinstance(event.get("cat"), str) or not event.get("cat"):
                errors.append(f"{path}:{lineno}: cat must be non-empty")
            for field in ("ts", "dur"):
                value = event.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"{path}:{lineno}: {field} must be numeric")
                elif field == "dur" and value < 0:
                    errors.append(f"{path}:{lineno}: dur must be nonnegative")
            if "args" in event and not isinstance(event["args"], dict):
                errors.append(f"{path}:{lineno}: args must be an object")
            if event.get("name") == "model_call" and isinstance(
                event.get("args"), dict
            ):
                cache_fields = (
                    "cached_prompt_tokens",
                    "uncached_prompt_tokens",
                    "kv_cache_hit_rate",
                )
                timing_fields = (
                    "llm_arrival_ts_us",
                    "queued_ts_us",
                    "first_scheduled_ts_us",
                    "first_token_ts_us",
                    "last_token_ts_us",
                    "llm_response_ready_ts_us",
                    "gym_llm_response_received_ts_us",
                    "gym_response_ready_ts_us",
                    "openhands_response_received_ts_us",
                )
                timing_duration_fields = (
                    "gym_tokenize_duration_us",
                    "openhands_parse_validate_duration_us",
                )
                event_args = event["args"]
                successful = event_args.get("status") == "ok"
                valid_complete_timing = False
                if successful:
                    successful_model_calls += 1
                generation_replica_idx = event_args.get(
                    "generation_replica_idx"
                )
                valid_generation_replica_idx = (
                    isinstance(generation_replica_idx, int)
                    and not isinstance(generation_replica_idx, bool)
                    and generation_replica_idx >= 0
                )
                if (
                    "generation_replica_idx" in event_args
                    and not valid_generation_replica_idx
                ):
                    errors.append(
                        f"{path}:{lineno}: generation_replica_idx must be a "
                        "nonnegative integer"
                    )
                if successful and valid_generation_replica_idx:
                    generation_replica_model_calls += 1
                    generation_replica_indices.add(generation_replica_idx)
                prompt_tokens = event_args.get("prompt_tokens")
                decode_tokens = event_args.get("decode_tokens")
                valid_prompt_tokens = (
                    isinstance(prompt_tokens, int)
                    and not isinstance(prompt_tokens, bool)
                    and prompt_tokens >= 0
                )
                valid_decode_tokens = (
                    isinstance(decode_tokens, int)
                    and not isinstance(decode_tokens, bool)
                    and decode_tokens >= 0
                )
                if "prompt_tokens" in event_args and not valid_prompt_tokens:
                    errors.append(
                        f"{path}:{lineno}: prompt_tokens must be a "
                        "nonnegative integer"
                    )
                if "decode_tokens" in event_args:
                    if not valid_decode_tokens:
                        errors.append(
                            f"{path}:{lineno}: decode_tokens must be a "
                            "nonnegative integer"
                        )
                if successful and valid_prompt_tokens and valid_decode_tokens:
                    token_count_model_calls += 1
                present_cache_fields = [
                    field for field in cache_fields if field in event_args
                ]
                if present_cache_fields and len(present_cache_fields) != len(
                    cache_fields
                ):
                    errors.append(
                        f"{path}:{lineno}: incomplete model_call cache metrics"
                    )
                elif present_cache_fields:
                    cache_metric_model_calls += 1
                    cached_tokens = event_args["cached_prompt_tokens"]
                    uncached_tokens = event_args["uncached_prompt_tokens"]
                    hit_rate = event_args["kv_cache_hit_rate"]
                    valid_token_counts = all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in (
                            prompt_tokens,
                            cached_tokens,
                            uncached_tokens,
                        )
                    )
                    if not valid_token_counts:
                        errors.append(
                            f"{path}:{lineno}: cache token counts must be integers"
                        )
                    elif (
                        prompt_tokens < 0
                        or cached_tokens < 0
                        or uncached_tokens < 0
                        or cached_tokens + uncached_tokens != prompt_tokens
                    ):
                        errors.append(
                            f"{path}:{lineno}: inconsistent cache token counts"
                        )
                    if (
                        not isinstance(hit_rate, (int, float))
                        or isinstance(hit_rate, bool)
                        or not 0 <= hit_rate <= 1
                    ):
                        errors.append(
                            f"{path}:{lineno}: kv_cache_hit_rate must be between 0 and 1"
                        )
                    elif valid_token_counts and prompt_tokens >= 0:
                        expected_hit_rate = (
                            cached_tokens / prompt_tokens if prompt_tokens else 0.0
                        )
                        if abs(hit_rate - expected_hit_rate) > 1e-9:
                            errors.append(
                                f"{path}:{lineno}: kv_cache_hit_rate is inconsistent"
                            )
                present_timing_fields = [
                    field for field in timing_fields if field in event_args
                ]
                if present_timing_fields and len(present_timing_fields) != len(
                    timing_fields
                ):
                    errors.append(
                        f"{path}:{lineno}: incomplete LLM request timestamps"
                    )
                elif present_timing_fields:
                    timestamps = [event_args[field] for field in timing_fields]
                    valid_timestamps = all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                        for value in timestamps
                    )
                    if not valid_timestamps:
                        errors.append(
                            f"{path}:{lineno}: LLM request timestamps must be "
                            "positive integers"
                        )
                    elif timestamps[:6] != sorted(timestamps[:6]):
                        errors.append(
                            f"{path}:{lineno}: LLM-server-owned request timestamps are out of order"
                        )
                    elif timestamps[6:8] != sorted(timestamps[6:8]):
                        errors.append(
                            f"{path}:{lineno}: Gym-owned request timestamps are out of order"
                        )
                    else:
                        valid_complete_timing = True
                present_duration_fields = [
                    field
                    for field in timing_duration_fields
                    if field in event_args
                ]
                if present_duration_fields and len(present_duration_fields) != len(
                    timing_duration_fields
                ):
                    errors.append(
                        f"{path}:{lineno}: incomplete detailed timing durations"
                    )
                elif present_duration_fields:
                    valid_durations = True
                    for field in timing_duration_fields:
                        value = event_args[field]
                        if (
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value < 0
                        ):
                            valid_durations = False
                            errors.append(
                                f"{path}:{lineno}: {field} must be a "
                                "nonnegative integer"
                            )
                    if successful and valid_complete_timing and valid_durations:
                        scheduling_timestamp_model_calls += 1
                # PD-disaggregation legs (TRT-LLM disagg server). Each owner's
                # stamps are ordered locally; cross-owner differences can
                # carry wall-clock skew and are not checked.
                disagg_errors_before = len(errors)
                for owner, ordered_fields in (
                    ("frontend", ("frontend_recv_ts_us", "frontend_fwd_ts_us")),
                    (
                        "context-engine",
                        (
                            "ctx_arrival_ts_us",
                            "ctx_queued_ts_us",
                            "ctx_first_scheduled_ts_us",
                            "ctx_done_ts_us",
                        ),
                    ),
                ):
                    present = [
                        event_args[field]
                        for field in ordered_fields
                        if field in event_args
                    ]
                    if not all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                        for value in present
                    ):
                        errors.append(
                            f"{path}:{lineno}: {owner} timestamps must be "
                            "positive integers"
                        )
                    elif present != sorted(present):
                        errors.append(
                            f"{path}:{lineno}: {owner} timestamps are out of order"
                        )
                for field, minimum in (
                    ("ctx_cached_prompt_tokens", 0),
                    ("ctx_computed_tokens", 0),
                    ("ctx_first_begin", 0),
                    ("ctx_num_chunks", 1),
                ):
                    if field in event_args:
                        value = event_args[field]
                        if (
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value < minimum
                        ):
                            errors.append(
                                f"{path}:{lineno}: {field} must be an integer "
                                f">= {minimum}"
                            )
                if (
                    successful
                    and len(errors) == disagg_errors_before
                    and all(field in event_args for field in DISAGG_CTX_FIELDS)
                ):
                    disagg_ctx_model_calls += 1
            count += 1
            if isinstance(event.get("name"), str):
                names[event["name"]] += 1
            if isinstance(event.get("cat"), str):
                categories[event["cat"]] += 1
    if len(generation_replica_indices) > 1:
        errors.append(
            f"{path}: model calls use multiple generation replicas "
            f"{sorted(generation_replica_indices)} despite sticky session routing"
        )
    return (
        count,
        names,
        categories,
        successful_model_calls,
        generation_replica_model_calls,
        token_count_model_calls,
        cache_metric_model_calls,
        scheduling_timestamp_model_calls,
        disagg_ctx_model_calls,
        errors,
    )


def validate_lifecycle_file(path: Path) -> tuple[int, Counter[str], list[str]]:
    count = 0
    events: Counter[str] = Counter()
    errors: list[str] = []
    try:
        lines = path.open("r", encoding="utf-8")
    except OSError as exc:
        return 0, events, [f"{path}: cannot open: {exc}"]

    with lines:
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"{path}:{lineno}: record is not an object")
                continue
            event = record.get("event")
            if event not in {"rollout_buffered", "rollout_consumed"}:
                errors.append(f"{path}:{lineno}: invalid lifecycle event {event!r}")
            elif isinstance(event, str):
                events[event] += 1
            if not isinstance(record.get("rollout_id"), str) or not record.get(
                "rollout_id"
            ):
                errors.append(f"{path}:{lineno}: rollout_id must be non-empty")
            if not isinstance(record.get("ts"), (int, float)) or isinstance(
                record.get("ts"), bool
            ):
                errors.append(f"{path}:{lineno}: ts must be numeric")
            for field in ("generation_weight_version", "target_weight_version"):
                if not isinstance(record.get(field), int) or isinstance(
                    record.get(field), bool
                ):
                    errors.append(f"{path}:{lineno}: {field} must be an integer")
            target = record.get("target_weight_version")
            if isinstance(target, int) and not isinstance(target, bool):
                if record.get("target_display_step") != target + 1:
                    errors.append(
                        f"{path}:{lineno}: target_display_step is inconsistent"
                    )
            if event == "rollout_consumed":
                consumed = record.get("consumed_weight_version")
                if not isinstance(consumed, int) or isinstance(consumed, bool):
                    errors.append(
                        f"{path}:{lineno}: consumed_weight_version must be an integer"
                    )
                else:
                    if record.get("consumed_display_step") != consumed + 1:
                        errors.append(
                            f"{path}:{lineno}: consumed_display_step is inconsistent"
                        )
                    if isinstance(target, int) and not isinstance(target, bool):
                        expected = (
                            "consumed_early"
                            if target > consumed
                            else (
                                "consumed_late"
                                if target < consumed
                                else "consumed_as_targeted"
                            )
                        )
                        if record.get("disposition") != expected:
                            errors.append(
                                f"{path}:{lineno}: disposition is inconsistent"
                            )
            count += 1
    return count, events, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-timeline", type=Path)
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--rollout-lifecycle", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow only the training or rollout trace layer to be present.",
    )
    parser.add_argument(
        "--require-generation-replica",
        action="store_true",
        help="Require generation_replica_idx on every successful model_call.",
    )
    parser.add_argument(
        "--require-model-token-counts",
        action="store_true",
        help="Require prompt_tokens and decode_tokens on every successful model_call.",
    )
    parser.add_argument(
        "--require-model-cache-metrics",
        action="store_true",
        help="Require cache metrics on every successful model_call event.",
    )
    parser.add_argument(
        "--require-model-scheduling-timestamps",
        action="store_true",
        help=(
            "Require complete LLM-server, Gym-proxy, and OpenHands detailed timing "
            "timestamps on every successful model_call event."
        ),
    )
    parser.add_argument(
        "--require-disagg-ctx-metrics",
        action="store_true",
        help=(
            "Require the PD-disaggregation frontend and context-engine fields "
            "(ctx_* prefix reuse, prefill work and stamps) on every successful "
            "model_call event."
        ),
    )
    args = parser.parse_args()

    if not args.training_timeline and not args.rollout_root:
        parser.error("provide --training-timeline and/or --rollout-root")

    errors: list[str] = []
    summary: dict[str, Any] = {}
    have_training = False
    have_rollout = False

    if args.training_timeline:
        (
            count,
            names,
            categories,
            _,
            _,
            _,
            _,
            _,
            _,
            file_errors,
        ) = validate_file(args.training_timeline)
        errors.extend(file_errors)
        have_training = count > 0
        summary["training"] = {
            "path": str(args.training_timeline),
            "events": count,
            "categories": dict(categories),
            "top_names": dict(names.most_common(20)),
        }
        if count and names["total_step_time"] == 0:
            errors.append(f"{args.training_timeline}: no total_step_time event found")

    if args.rollout_lifecycle:
        lifecycle_count, lifecycle_events, lifecycle_errors = (
            validate_lifecycle_file(args.rollout_lifecycle)
        )
        errors.extend(lifecycle_errors)
        summary["rollout_lifecycle"] = {
            "path": str(args.rollout_lifecycle),
            "records": lifecycle_count,
            "events": dict(lifecycle_events),
        }
        if lifecycle_count and lifecycle_events["rollout_consumed"] == 0:
            errors.append(
                f"{args.rollout_lifecycle}: no rollout_consumed record found"
            )

    rollout_files: list[Path] = []
    rollout_names: Counter[str] = Counter()
    rollout_categories: Counter[str] = Counter()
    rollout_events = 0
    successful_model_calls = 0
    generation_replica_model_calls = 0
    token_count_model_calls = 0
    cache_metric_model_calls = 0
    scheduling_timestamp_model_calls = 0
    disagg_ctx_model_calls = 0
    if args.rollout_root:
        if args.rollout_root.exists():
            rollout_files = sorted(args.rollout_root.rglob("timeline.jsonl"))
        else:
            errors.append(f"{args.rollout_root}: rollout root does not exist")
        for path in rollout_files:
            (
                count,
                names,
                categories,
                file_successful_model_calls,
                file_generation_replica_model_calls,
                file_token_count_model_calls,
                file_cache_metric_model_calls,
                file_scheduling_timestamp_model_calls,
                file_disagg_ctx_model_calls,
                file_errors,
            ) = validate_file(path)
            rollout_events += count
            successful_model_calls += file_successful_model_calls
            generation_replica_model_calls += (
                file_generation_replica_model_calls
            )
            token_count_model_calls += file_token_count_model_calls
            cache_metric_model_calls += file_cache_metric_model_calls
            scheduling_timestamp_model_calls += (
                file_scheduling_timestamp_model_calls
            )
            disagg_ctx_model_calls += file_disagg_ctx_model_calls
            rollout_names.update(names)
            rollout_categories.update(categories)
            errors.extend(file_errors)
        have_rollout = rollout_events > 0
        summary["rollouts"] = {
            "root": str(args.rollout_root),
            "files": len(rollout_files),
            "events": rollout_events,
            "successful_model_calls": successful_model_calls,
            "generation_replica_model_calls": generation_replica_model_calls,
            "token_count_model_calls": token_count_model_calls,
            "cache_metric_model_calls": cache_metric_model_calls,
            "scheduling_timestamp_model_calls": (
                scheduling_timestamp_model_calls
            ),
            "disagg_ctx_model_calls": disagg_ctx_model_calls,
            "categories": dict(rollout_categories),
            "top_names": dict(rollout_names.most_common(20)),
        }
        if rollout_events and not (
            rollout_names["model_call"] or rollout_categories["tool"]
        ):
            errors.append(
                f"{args.rollout_root}: no model_call or tool-category event found"
            )
        if (
            args.require_generation_replica
            and generation_replica_model_calls != successful_model_calls
        ):
            errors.append(
                f"{args.rollout_root}: generation replica attribution found on "
                f"{generation_replica_model_calls}/{successful_model_calls} "
                "successful model_call events"
            )
        if (
            args.require_model_token_counts
            and token_count_model_calls != successful_model_calls
        ):
            errors.append(
                f"{args.rollout_root}: input/decode token counts found on "
                f"{token_count_model_calls}/{successful_model_calls} "
                "successful model_call events"
            )
        if (
            args.require_model_cache_metrics
            and cache_metric_model_calls != successful_model_calls
        ):
            errors.append(
                f"{args.rollout_root}: cache metrics found on "
                f"{cache_metric_model_calls}/{successful_model_calls} "
                "successful model_call events"
            )
        if (
            args.require_model_scheduling_timestamps
            and scheduling_timestamp_model_calls != successful_model_calls
        ):
            errors.append(
                f"{args.rollout_root}: LLM scheduling timestamps found on "
                f"{scheduling_timestamp_model_calls}/{successful_model_calls} "
                "successful model_call events"
            )

    if (
        args.rollout_root
        and args.require_disagg_ctx_metrics
        and disagg_ctx_model_calls != successful_model_calls
    ):
        errors.append(
            f"{args.rollout_root}: disaggregation frontend/context metrics found on "
            f"{disagg_ctx_model_calls}/{successful_model_calls} "
            "successful model_call events"
        )

    if args.allow_partial:
        if not (have_training or have_rollout):
            errors.append("no nonempty timeline layer found")
    else:
        if not have_training:
            errors.append("training timeline is missing or empty")
        if not have_rollout:
            errors.append("rollout timelines are missing or empty")

    summary["valid"] = not errors
    summary["errors"] = errors[:100]
    print(json.dumps(summary, indent=2, sort_keys=True))
    if len(errors) > 100:
        print(f"... {len(errors) - 100} additional errors", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
