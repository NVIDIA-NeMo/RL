# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward-zeroing loop checks on root Responses output, excluding tool outputs."""

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field


class LoopPenaltyConfig(BaseModel):
    """Optional rollout penalties; defaults preserve existing training behavior."""

    penalize_tool_call_loops: bool = False
    penalize_reasoning_loops: bool = False
    tool_call_loop_min_calls: int = Field(default=12, ge=1)
    tool_call_loop_max_period: int = Field(default=4, ge=1)
    tool_call_loop_min_cycles: int = Field(default=3, ge=2)
    reasoning_loop_min_repeats: int = Field(default=20, ge=2)
    reasoning_loop_min_line_chars: int = Field(default=60, ge=1)
    reasoning_loop_min_distinct_lines: int = Field(default=2, ge=1)


# Matches the natural-language screen used to audit the training exports.
_NATURAL = re.compile(
    r"(?:Actually|Wait|But |However|Let me|I |Now |So |Hmm|OK|Maybe|We |The (?:issue|problem|task|requirement))"
)
_THINK = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)


def _text(parts: object) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    return "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))


def detect_output_loops(
    output: Sequence[Mapping[str, Any]], config: LoopPenaltyConfig
) -> dict[str, bool]:
    """Check contiguous exact tool-input cycles and repetition within one thought.

    Tool name and the complete parsed argument object form the signature. Text,
    reasoning and tool-result items do not interrupt the tool-call sequence.
    Reasoning lines are counted separately within each reasoning output item;
    counts never accumulate across turns. Tool outputs are never examined.
    """
    calls: list[tuple[str, str]] = []
    reasoning_hit = False
    for item in output:
        kind = item.get("type")
        if config.penalize_tool_call_loops and kind == "function_call":
            arguments = item.get("arguments", "")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass
            calls.append(
                (
                    str(item.get("name", "")),
                    json.dumps(arguments, sort_keys=True, ensure_ascii=False),
                )
            )
        if not config.penalize_reasoning_loops or reasoning_hit:
            continue
        if kind == "reasoning":
            thought = _text(item.get("summary"))
        elif kind == "message":
            # Some providers retain thinking tags instead of parsed reasoning.
            thought = "\n".join(_THINK.findall(_text(item.get("content"))))
        else:
            continue
        lines: Counter[str] = Counter()
        fence = None
        for line in thought.splitlines():
            text = line.strip()
            if text.startswith(("```", "~~~")):
                delimiter = text[:3]
                if fence is None:
                    fence = delimiter
                elif fence == delimiter:
                    fence = None
                continue
            if fence is not None or text.startswith(">"):
                continue
            if len(text) >= config.reasoning_loop_min_line_chars and _NATURAL.match(
                text
            ):
                lines[text] += 1
        reasoning_hit = (
            sum(n >= config.reasoning_loop_min_repeats for n in lines.values())
            >= config.reasoning_loop_min_distinct_lines
        )

    tool_hit = False
    for period in range(1, min(config.tool_call_loop_max_period, len(calls)) + 1):
        run = 0
        for i in range(period, len(calls)):
            run = run + 1 if calls[i] == calls[i - period] else 0
            if run and period + run >= max(
                config.tool_call_loop_min_calls,
                period * config.tool_call_loop_min_cycles,
            ):
                tool_hit = True
                break
        if tool_hit:
            break
    return {"tool_call_loop": tool_hit, "reasoning_loop": reasoning_hit}


def apply_loop_reward_penalties(
    results: list[dict], config: LoopPenaltyConfig
) -> dict[str, int]:
    """Zero shared reward and retain evidence across repeated conversions.

    Streaming recovery converts a result before the group metrics conversion.
    Report the same per-result zeroing event on both passes; the streaming
    caller discards its counts and the group caller publishes them.
    """
    if not (config.penalize_tool_call_loops or config.penalize_reasoning_loops):
        return {}
    counts = {"tool_call_loop": 0, "reasoning_loop": 0, "loop_reward_zeroed": 0}
    for result in results:
        full_result = result["full_result"]
        full_result.setdefault(
            "reward_before_loop_penalties", full_result.get("reward")
        )
        full_result.setdefault("loop_reward_zeroed", False)
        hits = detect_output_loops(
            (full_result.get("response") or {}).get("output") or [], config
        )
        # Scalar-only evidence stays with the completion, not in the transcript.
        full_result["loop_penalties"] = hits
        for name, hit in hits.items():
            counts[name] += int(hit)
        if any(hits.values()):
            if float(full_result.get("reward") or 0.0) != 0.0:
                full_result["loop_reward_zeroed"] = True
            counts["loop_reward_zeroed"] += bool(full_result.get("loop_reward_zeroed"))
            full_result["reward"] = 0.0
    return counts
