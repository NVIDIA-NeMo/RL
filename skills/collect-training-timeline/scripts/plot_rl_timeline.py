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

"""Plot NeMo-RL training timers and NeMo Gym rollout events together."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objects as go


TRAINING_COLORS = {
    "total_step_time": "#34495e",
    "exposed_generation": "#e74c3c",
    "reward_calculation": "#f4d03f",
    "data_processing": "#5dade2",
    "logprob_inference_prep": "#85c1e9",
    "policy_and_reference_logprobs": "#2e86c1",
    "advantage_calculation": "#af7ac5",
    "training_prep": "#82e0aa",
    "policy_training": "#1e8449",
    "weight_sync": "#d35400",
    "total_validation_time": "#8e44ad",
    "checkpointing": "#7f8c8d",
}

ROLLOUT_COLORS = {
    "ray_queue": "#95a5a6",
    "agent_container_spinup": "#7f8c8d",
    "create_runtime": "#85c1e9",
    "connect_runtime": "#3498db",
    "initialize_runtime": "#8e44ad",
    "prepare_instruction": "#af7ac5",
    "model_call": "#27ae60",
    # Keep shell-command execution visually distinct from the other tool actions.
    "CmdRunAction": "#d81b60",
    "FileReadAction": "#f7dc6f",
    "FileEditAction": "#e67e22",
    "FileWriteAction": "#d35400",
    "IPythonRunCellAction": "#f5b041",
    "extract_patch": "#ca6f1e",
    "runtime_cleanup": "#a04000",
    "final_eval": "#c0392b",
}

# Plotly paints later bar traces over earlier ones. Rollout events are nested, so
# draw broad agent/runtime spans first and detailed model/tool work on top.
ROLLOUT_CATEGORY_DRAW_ORDER = {
    "queue": 0,
    "rollout": 0,
    "runtime": 0,
    "agent": 1,
    "model": 2,
    "tool": 3,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            print(f"Skipping malformed event {path}:{line_number}: {error}")
            continue
        if event.get("ph") == "X" and "ts" in event and "dur" in event:
            events.append(event)
    return events


def load_rollout_lifecycle(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load the latest targeting and consumption metadata for each rollout."""
    if path is None or not path.exists():
        return {}

    lifecycle: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            print(f"Skipping malformed lifecycle record {path}:{line_number}: {error}")
            continue
        rollout_id = record.get("rollout_id")
        if not rollout_id:
            continue
        lifecycle.setdefault(str(rollout_id), {}).update(record)
    return lifecycle


def event_identity(event: dict[str, Any]) -> tuple[str, int]:
    return str(event.get("args", {}).get("hostname", "")), int(event.get("pid", -1))


def select_controller_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the process that owns the RL total_step_time spans."""
    owners = Counter(
        event_identity(event)
        for event in events
        if event.get("name") == "total_step_time"
    )
    if not owners:
        raise ValueError(
            "No total_step_time events were found in the training timeline"
        )
    owner, _ = owners.most_common(1)[0]
    return [event for event in events if event_identity(event) == owner]


def select_training_steps(
    controller_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select full RL steps, excluding short buffer-polling timer attempts."""
    policy_training_events = [
        event for event in controller_events if event.get("name") == "policy_training"
    ]
    total_step_events = [
        event for event in controller_events if event.get("name") == "total_step_time"
    ]
    completed_step_events = [
        step_event
        for step_event in total_step_events
        if any(
            int(step_event["ts"]) <= int(training_event["ts"])
            and int(training_event["ts"]) + int(training_event["dur"])
            <= int(step_event["ts"]) + int(step_event["dur"])
            for training_event in policy_training_events
        )
    ]
    return sorted(
        completed_step_events or total_step_events, key=lambda event: event["ts"]
    )


def overlaps(event: dict[str, Any], start_us: int, end_us: int) -> bool:
    event_start = int(event["ts"])
    event_end = event_start + int(event.get("dur", 0))
    return event_start < end_us and event_end > start_us


def clip_event(
    event: dict[str, Any], start_us: int, end_us: int
) -> tuple[float, float]:
    event_start = max(int(event["ts"]), start_us)
    event_end = min(int(event["ts"]) + int(event.get("dur", 0)), end_us)
    return (event_start - start_us) / 1e6, max(event_end - event_start, 0) / 1e6


def load_rollouts(
    root: Path, lifecycle_by_rollout: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    rollouts = []
    lifecycle_by_rollout = lifecycle_by_rollout or {}
    for path in root.rglob("timeline.jsonl"):
        events = load_jsonl(path)
        if not events:
            continue
        queue = next(
            (event for event in events if event.get("name") == "ray_queue"), None
        )
        first = min(events, key=lambda event: event["ts"])
        source = queue or first
        rollout_id = str(source.get("args", {}).get("rollout_id", path.parent.name))
        lifecycle = lifecycle_by_rollout.get(rollout_id, {})
        for event in events:
            event.setdefault("args", {}).update(
                {
                    key: lifecycle[key]
                    for key in (
                        "generation_weight_version",
                        "target_weight_version",
                        "target_display_step",
                        "consumed_weight_version",
                        "consumed_display_step",
                        "disposition",
                    )
                    if key in lifecycle
                }
            )
        start = min(int(event["ts"]) for event in events)
        end = max(int(event["ts"]) + int(event.get("dur", 0)) for event in events)
        rollouts.append(
            {
                "id": rollout_id,
                "events": events,
                "start": start,
                "end": end,
                "path": path,
                "lifecycle": lifecycle,
            }
        )
    return sorted(rollouts, key=lambda rollout: (rollout["start"], rollout["id"]))


def sample_rollouts_evenly(
    rollouts: list[dict[str, Any]],
    limit: int,
    include_finish_extremes: bool = False,
) -> list[dict[str, Any]]:
    """Keep representative rollouts across the full selected time window."""
    if limit <= 0 or len(rollouts) <= limit:
        return rollouts
    if limit == 1:
        return [rollouts[len(rollouts) // 2]]

    if include_finish_extremes:
        earliest_finish = min(rollouts, key=lambda rollout: rollout["end"])
        latest_finish = max(rollouts, key=lambda rollout: rollout["end"])
        pinned = {
            earliest_finish["path"]: earliest_finish,
            latest_finish["path"]: latest_finish,
        }
        remaining = [rollout for rollout in rollouts if rollout["path"] not in pinned]
        sampled = sample_rollouts_evenly(remaining, limit - len(pinned))
        return sorted(
            [*pinned.values(), *sampled],
            key=lambda rollout: (rollout["start"], rollout["id"]),
        )

    indices = {
        round(index * (len(rollouts) - 1) / (limit - 1)) for index in range(limit)
    }
    return [rollouts[index] for index in sorted(indices)]


def hover_text(
    event: dict[str, Any], relative_start: float, duration: float, source: str
) -> str:
    args = event.get("args", {})
    details = [
        f"<b>{event.get('name', 'event')}</b>",
        f"Source: {source}",
        f"Start: {relative_start:.3f} s",
        f"Duration: {duration:.3f} s",
    ]
    for key in (
        "rollout_id",
        "turn",
        "generation_replica_idx",
        "context_tokens",
        "decode_tokens",
        "total_tokens",
        "prompt_tokens",
        "cached_prompt_tokens",
        "uncached_prompt_tokens",
        "kv_cache_hit_rate",
        "ctx_cached_prompt_tokens",
        "ctx_computed_tokens",
        "ctx_first_begin",
        "ctx_num_chunks",
        "generation_tokens",
        "generation_weight_version",
        "target_weight_version",
        "target_display_step",
        "consumed_weight_version",
        "consumed_display_step",
        "disposition",
        "status",
        "action_type",
        "hostname",
    ):
        if key in args:
            value = args[key]
            if key == "kv_cache_hit_rate":
                value = f"{float(value):.1%}"
            details.append(f"{key.replace('_', ' ').title()}: {value}")
    timing_keys = (
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
    if all(
        isinstance(args.get(key), int)
        and not isinstance(args.get(key), bool)
        and args[key] > 0
        for key in timing_keys
    ):
        event_start_us = float(event.get("ts", 0))
        event_end_us = event_start_us + float(event.get("dur", 0))
        (
            arrival_us,
            queued_us,
            scheduled_us,
            first_token_us,
            last_token_us,
            llm_ready_us,
            gym_received_us,
            gym_ready_us,
            openhands_received_us,
        ) = (
            float(args[key]) for key in timing_keys
        )
        details.extend(
            (
                f"OpenHands To LLM Arrival: "
                f"{(arrival_us - event_start_us) / 1000:.3f} ms",
                f"LLM Frontend/Submit: {(queued_us - arrival_us) / 1000:.3f} ms",
                f"LLM Scheduler Queue: {(scheduled_us - queued_us) / 1000:.3f} ms",
                f"Scheduled To First Token: "
                f"{(first_token_us - scheduled_us) / 1000:.3f} ms",
                f"Decode: {(last_token_us - first_token_us) / 1000:.3f} ms",
                f"LLM Finalize Response: "
                f"{(llm_ready_us - last_token_us) / 1000:.3f} ms",
                f"LLM To Gym: "
                f"{(gym_received_us - llm_ready_us) / 1000:.3f} ms",
                f"Gym Postprocess: "
                f"{(gym_ready_us - gym_received_us) / 1000:.3f} ms",
                f"Gym To OpenHands: "
                f"{(openhands_received_us - gym_ready_us) / 1000:.3f} ms",
                f"OpenHands Post-response: "
                f"{(event_end_us - openhands_received_us) / 1000:.3f} ms",
            )
        )
        if isinstance(args.get("gym_tokenize_duration_us"), int):
            details.append(
                f"Gym /tokenize (inside postprocess): "
                f"{args['gym_tokenize_duration_us'] / 1000:.3f} ms"
            )
        if isinstance(args.get("openhands_parse_validate_duration_us"), int):
            details.append(
                f"OpenHands Parse/Validate: "
                f"{args['openhands_parse_validate_duration_us'] / 1000:.3f} ms"
            )
    # PD-disaggregation legs: owner-local differences only.
    def _stamp(key: str) -> float | None:
        value = args.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return float(value)
        return None

    for label, first, second in (
        ("Disagg Frontend Hold", "frontend_recv_ts_us", "frontend_fwd_ts_us"),
        ("Ctx Adapter/Submit", "ctx_arrival_ts_us", "ctx_queued_ts_us"),
        ("Ctx Scheduler Queue", "ctx_queued_ts_us", "ctx_first_scheduled_ts_us"),
        ("Ctx Prefill", "ctx_first_scheduled_ts_us", "ctx_done_ts_us"),
    ):
        begin_us, end_us = _stamp(first), _stamp(second)
        if begin_us is not None and end_us is not None:
            details.append(f"{label}: {(end_us - begin_us) / 1000:.3f} ms")
    return "<br>".join(details)


def add_event_bars(
    figure: go.Figure,
    entries: Iterable[tuple[dict[str, Any], str, str]],
    start_us: int,
    end_us: int,
    colors: dict[str, str],
    category_draw_order: dict[str, int] | None = None,
) -> None:
    grouped: dict[str, list[tuple[dict[str, Any], str, str]]] = {}
    for event, lane, source in entries:
        grouped.setdefault(str(event.get("name", "event")), []).append(
            (event, lane, source)
        )

    grouped_entries = list(grouped.items())
    if category_draw_order is not None:
        grouped_entries.sort(
            key=lambda item: max(
                category_draw_order.get(str(event.get("cat", "")), 1)
                for event, _, _ in item[1]
            )
        )

    for name, name_entries in grouped_entries:
        starts, durations, lanes, hovers = [], [], [], []
        for event, lane, source in name_entries:
            relative_start, duration = clip_event(event, start_us, end_us)
            starts.append(relative_start)
            durations.append(duration)
            lanes.append(lane)
            hovers.append(hover_text(event, relative_start, duration, source))
        figure.add_trace(
            go.Bar(
                name=name,
                y=lanes,
                x=durations,
                base=starts,
                orientation="h",
                marker_color=colors.get(name, "#5d6d7e"),
                text=hovers,
                textposition="none",
                hovertemplate="%{text}<extra></extra>",
            )
        )


def write_timeline(
    training_path: Path,
    rollout_root: Path | None,
    output: Path,
    steps: list[int] | None,
    max_rollouts: int,
    title: str,
    rollout_lifecycle_path: Path | None = None,
    rollouts_per_step: int | None = None,
    zero_at_first_rollout: bool = False,
    include_all_rollouts: bool = False,
    include_finish_extremes: bool = False,
) -> None:
    training_events = select_controller_events(load_jsonl(training_path))
    step_events = select_training_steps(training_events)
    selected_step_numbers = steps or list(range(1, len(step_events) + 1))
    if not selected_step_numbers:
        raise ValueError("No training steps were selected")
    if min(selected_step_numbers) < 1 or max(selected_step_numbers) > len(step_events):
        raise ValueError(
            f"Requested steps {selected_step_numbers}, but timeline has {len(step_events)} steps"
        )

    selected_steps = [step_events[number - 1] for number in selected_step_numbers]
    window_start = min(int(event["ts"]) for event in selected_steps)
    window_end = max(int(event["ts"]) + int(event["dur"]) for event in selected_steps)

    rollouts = []
    if rollout_root is not None:
        if rollout_lifecycle_path is None:
            candidate = training_path.with_name("rollout_lifecycle.jsonl")
            if candidate.exists():
                rollout_lifecycle_path = candidate
        lifecycle_by_rollout = load_rollout_lifecycle(rollout_lifecycle_path)
        rollouts = load_rollouts(rollout_root, lifecycle_by_rollout)
        # Keep only rollouts overlapping the plotted steps. An all-rollouts plot
        # of every step keeps every discovered trace, including rollouts that
        # start after the last step; with explicit --step it keeps those
        # steps' rollouts without sampling them.
        if not include_all_rollouts or steps:
            rollouts = [
                rollout
                for rollout in rollouts
                if rollout["start"] < window_end
                and (zero_at_first_rollout or rollout["end"] > window_start)
            ]
        if not include_all_rollouts:
            if rollouts_per_step is not None:
                selected_rollouts: dict[Path, dict[str, Any]] = {}
                for step_event in selected_steps:
                    step_start = int(step_event["ts"])
                    step_end = step_start + int(step_event["dur"])
                    candidates = [
                        rollout
                        for rollout in rollouts
                        if rollout["start"] < step_end and rollout["end"] > step_start
                    ]
                    for rollout in sample_rollouts_evenly(
                        candidates,
                        rollouts_per_step,
                        include_finish_extremes=include_finish_extremes,
                    ):
                        selected_rollouts[rollout["path"]] = rollout
                rollouts = sorted(
                    selected_rollouts.values(),
                    key=lambda rollout: (rollout["start"], rollout["id"]),
                )
            else:
                rollouts = sample_rollouts_evenly(
                    rollouts,
                    max_rollouts,
                    include_finish_extremes=include_finish_extremes,
                )
    if zero_at_first_rollout and rollouts:
        window_start = min(int(rollout["start"]) for rollout in rollouts)
    if include_all_rollouts and rollouts:
        window_end = max(window_end, max(int(rollout["end"]) for rollout in rollouts))

    training_entries = []
    lane_order = []
    for event in training_events:
        if not overlaps(event, window_start, window_end):
            continue
        if event.get("name") == "total_step_time":
            try:
                step_number = step_events.index(event) + 1
            except ValueError:
                continue
            if step_number not in selected_step_numbers:
                continue
        lane = f"Training · {event.get('name', 'event')}"
        if lane not in lane_order:
            lane_order.append(lane)
        training_entries.append((event, lane, "trainer"))

    rollout_entries = []
    for index, rollout in enumerate(rollouts, 1):
        short_id = rollout["id"].split("__")[-1]
        lifecycle = rollout["lifecycle"]
        target_step = lifecycle.get("target_display_step")
        consumed_step = lifecycle.get("consumed_display_step")
        if target_step is None:
            lifecycle_label = ""
        elif consumed_step is None:
            lifecycle_label = f" · T{target_step}→unconsumed"
        else:
            lifecycle_label = f" · T{target_step}→C{consumed_step}"
        lane = f"R{index} · {short_id}{lifecycle_label}"
        lane_order.append(lane)
        for event in rollout["events"]:
            if overlaps(event, window_start, window_end):
                rollout_entries.append((event, lane, rollout["id"]))

    figure = go.Figure()
    add_event_bars(
        figure,
        training_entries,
        window_start,
        window_end,
        TRAINING_COLORS,
    )
    add_event_bars(
        figure,
        rollout_entries,
        window_start,
        window_end,
        ROLLOUT_COLORS,
        category_draw_order=ROLLOUT_CATEGORY_DRAW_ORDER,
    )

    for number, event in zip(selected_step_numbers, selected_steps):
        x = (int(event["ts"]) - window_start) / 1e6
        figure.add_vline(x=x, line_dash="dash", line_color="#2c3e50")
        figure.add_annotation(
            x=x,
            y=1,
            yref="paper",
            yanchor="bottom",
            yshift=4,
            text=f"Step {number}",
            showarrow=False,
            xanchor="left",
        )

    subtitle = (
        f"Steps: {','.join(map(str, selected_step_numbers))} · "
        f"Window: {(window_end - window_start) / 1e6:.2f}s · "
        f"Overlapping rollouts shown: {len(rollouts)}"
    )
    lifecycle_rollouts = sum(bool(rollout["lifecycle"]) for rollout in rollouts)
    if lifecycle_rollouts:
        subtitle += f" · Lifecycle metadata: {lifecycle_rollouts}/{len(rollouts)}"
    if rollouts_per_step is not None:
        subtitle += f" · Sampled per step: {rollouts_per_step}"
    if include_all_rollouts:
        subtitle += " · Selection: all rollout traces"
    if zero_at_first_rollout and rollouts:
        subtitle += " · Origin: first rollout"
    height = max(850, 28 * len(lane_order) + 240)
    figure.update_layout(
        title={
            "text": f"{title}<br><sup>{subtitle}</sup>",
            "x": 0,
            "xanchor": "left",
            # Pin the title a fixed 12 px below the top edge: a fractional
            # container y drifts into the plot once the figure grows tall.
            "y": 1 - 12 / height,
            "yref": "container",
            "yanchor": "top",
        },
        barmode="overlay",
        height=height,
        template="plotly_white",
        xaxis={
            "title": (
                "Seconds from first rollout"
                if zero_at_first_rollout and rollouts
                else "Seconds from selected timeline window start"
            )
        },
        yaxis={
            "title": "",
            "categoryorder": "array",
            "categoryarray": list(reversed(lane_order)),
            "automargin": True,
        },
        # One entry per event name (tens of them): a horizontal legend above
        # the plot wraps upward into the title, so keep it in a right column.
        legend={
            "orientation": "v",
            "x": 1.01,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
        },
        margin={"l": 260, "r": 40, "t": 110, "b": 70},
        hovermode="closest",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-timeline", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument(
        "--rollout-lifecycle",
        type=Path,
        help=(
            "Rollout lifecycle JSONL; defaults to rollout_lifecycle.jsonl beside "
            "--training-timeline when present"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        help="1-indexed RL step to plot; repeat for multiple steps (default: all)",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=64,
        help="Maximum overlapping rollout lanes to show; 0 means unlimited",
    )
    parser.add_argument(
        "--rollouts-per-step",
        type=int,
        help=(
            "Evenly sample this many rollouts from each selected step; "
            "overrides --max-rollouts and deduplicates rollouts spanning steps"
        ),
    )
    parser.add_argument("--title", default="NeMo-RL training and rollout timeline")
    parser.add_argument(
        "--zero-at-first-rollout",
        action="store_true",
        help="Use the earliest included rollout start as time zero",
    )
    parser.add_argument(
        "--include-all-rollouts",
        action="store_true",
        help=(
            "Include rollouts without sampling and extend the time window to "
            "their end: every discovered rollout, or with --step every rollout "
            "overlapping those steps"
        ),
    )
    parser.add_argument(
        "--include-finish-extremes",
        action="store_true",
        help="Pin the earliest- and latest-finishing rollouts in the sampled lanes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_timeline(
        training_path=args.training_timeline,
        rollout_root=args.rollout_root,
        rollout_lifecycle_path=args.rollout_lifecycle,
        output=args.output,
        steps=args.step,
        max_rollouts=args.max_rollouts,
        title=args.title,
        rollouts_per_step=args.rollouts_per_step,
        zero_at_first_rollout=args.zero_at_first_rollout,
        include_all_rollouts=args.include_all_rollouts,
        include_finish_extremes=args.include_finish_extremes,
    )


if __name__ == "__main__":
    main()
