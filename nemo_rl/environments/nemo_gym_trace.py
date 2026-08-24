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

"""Validate Gym model-call continuity and plan physical trace boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _as_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list")
    return list(value)


def _token_ids(value: Any, *, field: str) -> list[int]:
    result = _as_list(value, field=field)
    if any(isinstance(token, bool) or not isinstance(token, int) for token in result):
        raise ValueError(f"{field} must contain integer token IDs")
    return result


def _logprobs(value: Any, *, field: str) -> list[float]:
    values = _as_list(value, field=field)
    result = [float(item) for item in values]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite log probabilities")
    return result


def _boundaries_by_turn(
    boundary_events: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    event_ids: set[str] = set()
    for index, event in enumerate(boundary_events):
        if not isinstance(event, Mapping):
            raise TypeError(f"boundary_events[{index}] must be a mapping")
        event_id = event.get("event_id")
        turn_id = event.get("applies_to_step")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"boundary_events[{index}] has no event_id")
        if event_id in event_ids:
            raise ValueError(f"Duplicate boundary event ID {event_id!r}")
        if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id < 1:
            raise ValueError(f"boundary_events[{index}] has invalid applies_to_step")
        if turn_id in result:
            raise ValueError(f"Multiple boundary events apply to turn {turn_id}")
        event_ids.add(event_id)
        result[turn_id] = dict(event)
    return result


def build_rollout_trace_plan(
    *,
    rollout_id: str,
    calls: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]] = (),
    media_assets: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate model calls and return a lightweight boundary plan.

    This is NeMo-RL's ingestion validation boundary. It validates the external
    evidence that affects training, then retains only the fields needed while
    constructing canonical physical message logs. Token IDs and logprobs stay
    authoritative in the Gym response and are not duplicated in this plan.
    """
    if not isinstance(rollout_id, str) or not rollout_id:
        raise ValueError("rollout_id must be non-empty")
    if strict and media_assets is None:
        raise ValueError("Exact-trace authority requires the media asset arena")

    boundaries = _boundaries_by_turn(boundary_events)
    used_boundary_turns: set[int] = set()
    seen_turn_ids: set[int] = set()
    seen_completion_ids: set[str] = set()
    seen_action_ids: set[str] = set()
    previous_context: list[int] = []
    previous_media_ids: list[str] = []
    model_calls: list[dict[str, Any]] = []
    physical_trace_ids: list[str] = []
    current_trace: dict[str, Any] | None = None
    declared_boundaries = 0
    inferred_boundaries = 0

    for call_index, raw_call in enumerate(calls):
        if not isinstance(raw_call, Mapping):
            raise TypeError(f"calls[{call_index}] must be a mapping")
        turn_id = raw_call.get("turn_id", call_index + 1)
        if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id < 1:
            raise ValueError(f"Invalid turn_id {turn_id!r}")
        if turn_id in seen_turn_ids:
            raise ValueError(f"Duplicate turn_id {turn_id}")
        if strict and turn_id != call_index + 1:
            raise ValueError("Exact trace calls must use consecutive turn IDs")
        seen_turn_ids.add(turn_id)

        completion_id = raw_call.get("completion_id")
        if not isinstance(completion_id, str) or not completion_id:
            raise ValueError(f"Call {turn_id} has no completion identity")
        if completion_id in seen_completion_ids:
            raise ValueError(f"Duplicate completion ID {completion_id!r}")
        seen_completion_ids.add(completion_id)

        raw_action_id = raw_call.get("action_id")
        if strict and (not isinstance(raw_action_id, str) or not raw_action_id):
            raise ValueError(f"Completion {completion_id!r} has no action identity")
        action_id = (
            raw_action_id
            if isinstance(raw_action_id, str) and raw_action_id
            else completion_id
        )
        if action_id in seen_action_ids:
            raise ValueError(f"Duplicate action ID {action_id!r}")
        seen_action_ids.add(action_id)

        prompt_token_ids = _token_ids(
            raw_call.get("prompt_token_ids"),
            field=f"calls[{call_index}].prompt_token_ids",
        )
        sampled_token_ids = _token_ids(
            raw_call.get("sampled_token_ids"),
            field=f"calls[{call_index}].sampled_token_ids",
        )
        sampled_logprobs = _logprobs(
            raw_call.get("sampled_logprobs"),
            field=f"calls[{call_index}].sampled_logprobs",
        )
        if len(sampled_token_ids) != len(sampled_logprobs):
            raise ValueError(
                f"Completion {completion_id!r} token/logprob lengths disagree"
            )
        media_ids = [
            str(media_id)
            for media_id in _as_list(
                raw_call.get("media_ids", []),
                field=f"calls[{call_index}].media_ids",
            )
        ]
        if strict and any(
            media_id not in (media_assets or {}) for media_id in media_ids
        ):
            raise ValueError(
                f"Completion {completion_id!r} references an unknown media asset"
            )

        token_contiguous = previous_context == prompt_token_ids[: len(previous_context)]
        media_contiguous = previous_media_ids == media_ids[: len(previous_media_ids)]
        append_compatible = token_contiguous and media_contiguous
        starts_trace = current_trace is None or not append_compatible
        expected_append_compatible = raw_call.get("expected_append_compatible")
        if strict and (
            not isinstance(expected_append_compatible, bool)
            or expected_append_compatible
            != (current_trace is not None and append_compatible)
        ):
            raise ValueError(
                "Gym append-compatibility declaration disagrees with token/media "
                f"evidence at turn {turn_id}"
            )

        boundary = boundaries.get(turn_id)
        if current_trace is not None and starts_trace:
            if boundary is None:
                if strict:
                    raise ValueError(
                        f"Material rewrite before turn {turn_id} has no boundary record"
                    )
                boundary = {
                    "event_id": f"{rollout_id}:inferred-boundary-{turn_id:06d}",
                    "applies_to_step": turn_id,
                }
                inferred_boundaries += 1
            else:
                declared_boundaries += 1
                used_boundary_turns.add(turn_id)
        elif boundary is not None and strict:
            raise ValueError(
                f"Boundary for turn {turn_id} does not correspond to a rewrite"
            )

        segment_index = raw_call.get("segment_index")
        context_epoch = raw_call.get("context_epoch")
        segment_id = raw_call.get("segment_id")
        if strict:
            expected_segment_index = (
                len(physical_trace_ids) if starts_trace else len(physical_trace_ids) - 1
            )
            if (
                isinstance(segment_index, bool)
                or segment_index != expected_segment_index
                or context_epoch != segment_index
                or not isinstance(segment_id, str)
                or not segment_id
            ):
                raise ValueError(
                    f"Completion {completion_id!r} has inconsistent segment identity"
                )
            expected_boundary_id = boundary.get("event_id") if boundary else None
            if raw_call.get("compaction_event_id") != expected_boundary_id:
                raise ValueError(
                    f"Completion {completion_id!r} compaction event is inconsistent"
                )
            source_rollout_id = raw_call.get("rollout_id")
            if source_rollout_id != rollout_id:
                raise ValueError(
                    f"Completion {completion_id!r} belongs to the wrong rollout"
                )
            if not starts_trace:
                assert current_trace is not None
                if (
                    segment_id != current_trace["segment_id"]
                    or segment_index != current_trace["segment_index"]
                    or context_epoch != current_trace["context_epoch"]
                ):
                    raise ValueError(
                        f"Completion {completion_id!r} changed segment identity "
                        "without a rewrite"
                    )

        if starts_trace:
            trace_index = len(physical_trace_ids)
            trace_id = f"{rollout_id}:trace-{trace_index:06d}"
            current_trace = {
                "trace_id": trace_id,
                "segment_id": segment_id,
                "segment_index": segment_index,
                "context_epoch": context_epoch,
            }
            physical_trace_ids.append(trace_id)
            new_media_ids = media_ids
        else:
            assert current_trace is not None
            new_media_ids = media_ids[len(previous_media_ids) :]

        assert current_trace is not None
        eligible = bool(raw_call.get("eligible", True))
        model_calls.append(
            {
                "trace_id": current_trace["trace_id"],
                "starts_physical_trace": starts_trace,
                "new_media_ids": new_media_ids,
                "finish_reason": raw_call.get("finish_reason"),
                "eligible": eligible,
            }
        )
        previous_context = [*prompt_token_ids, *sampled_token_ids]
        previous_media_ids = media_ids

    unused_boundaries = set(boundaries) - used_boundary_turns
    if strict and unused_boundaries:
        raise ValueError(
            "Boundary records do not correspond to material rewrites at turns "
            f"{sorted(unused_boundaries)}"
        )
    checks = {
        "ok": inferred_boundaries == 0,
        "model_call_count": len(model_calls),
        "physical_trace_count": len(physical_trace_ids),
        "declared_boundary_count": declared_boundaries,
        "inferred_boundary_count": inferred_boundaries,
    }
    return {
        "rollout_id": rollout_id,
        "model_calls": model_calls,
        "physical_trace_ids": physical_trace_ids,
        "checks": checks,
    }
