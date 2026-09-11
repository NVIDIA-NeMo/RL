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
"""Shared reward calculations and the small adapter for verified capture rows."""

from __future__ import annotations

import math
import statistics
from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import RewardPenaltyConfig
    from nemo_rl.experience.rollouts import EffortLevelsConfig


def has_duplicated_reasoning(output_items: list[dict[str, Any]]) -> bool:
    """Detect nonempty reasoning copied into immediately adjacent content."""
    for item1, item2 in zip(output_items, output_items[1:]):
        if item1.get("type") != "reasoning":
            continue
        summary = item1.get("summary", [])
        if not summary or "text" not in summary[0]:
            continue
        reasoning_text = summary[0]["text"].strip()
        content = item2.get("content", "")
        if isinstance(content, list) and content and "text" in content[0]:
            chat_text = content[0]["text"].strip()
        elif isinstance(content, str):
            chat_text = content.strip()
        else:
            continue
        if reasoning_text and chat_text and reasoning_text == chat_text:
            return True
    return False


def has_empty_final_answer(output_items: list[dict[str, Any]]) -> bool:
    """Detect missing supported final content, except a final function call."""
    # Skip if the last output item is a function_call — it is legit for model to
    # produce reasoning and then a function_call as the last output item in PivotRL
    if output_items and output_items[-1].get("type") == "function_call":
        return False
    final_answer_text = None
    for item in reversed(output_items):
        # Skip items without content (function_call, function_call_output, etc.)
        if "content" not in item:
            continue
        content = item["content"]
        if isinstance(content, list) and content and "text" in content[0]:
            final_answer_text = content[0]["text"].strip()
            break
        elif isinstance(content, str):
            final_answer_text = content.strip()
            break
    return final_answer_text is None or final_answer_text == ""


def effort_shaping_enabled(config: EffortLevelsConfig | None) -> bool:
    """Return whether the configured low-effort adjustment is active."""
    return config is not None and config.low_weight > 0 and bool(config.low_string)


def is_low_effort(row: dict[str, Any], config: EffortLevelsConfig) -> bool:
    """Classify the original rollout input using the standard last-user rule."""
    prompt = next(
        (
            msg["content"]
            for msg in reversed(row["responses_create_params"]["input"])
            if msg.get("role") == "user" and "content" in msg
        ),
        "",
    )
    return config.low_string in prompt


def shape_effort_reward(
    reward: float, *, length: int, config: EffortLevelsConfig
) -> tuple[float, float]:
    """Return shaped reward and length term, preserving negative reward semantics."""
    length_reward = min(1.0, config.low_weight * (1.0 - length / config.low_ub))
    return (
        reward
        + reward * max(length_reward, 0.0)
        + config.low_penalty * min(length_reward, 0.0),
        length_reward,
    )


def has_unwanted_tokens(
    generations: Iterable[Container[int]], unwanted_ids: Sequence[int]
) -> bool:
    """Check full generated sequences, including their terminal tokens."""
    return any(
        token in generation for generation in generations for token in unwanted_ids
    )


@dataclass(frozen=True)
class RewardChecks:
    """Scored-text and original-prompt checks saved with a rollout attempt."""

    duplicated_reasoning: bool
    empty_final_answer: bool
    low_effort: bool

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.duplicated_reasoning,
                self.empty_final_answer,
                self.low_effort,
            )
        ):
            raise ValueError("reward checks must be booleans")


def compute_reward_checks(
    result: dict[str, Any],
    row: dict[str, Any],
    effort_config: EffortLevelsConfig | None,
) -> RewardChecks:
    """Evaluate available checks before sealing, without changing the raw reward."""
    response = result.get("response")
    output = response.get("output", []) if isinstance(response, dict) else []
    return RewardChecks(
        has_duplicated_reasoning(output),
        has_empty_final_answer(output),
        is_low_effort(row, effort_config)
        if effort_shaping_enabled(effort_config) and effort_config is not None
        else False,
    )


def finalize_capture_reward(
    reward: float,
    *,
    checks: RewardChecks | None,
    penalty_config: RewardPenaltyConfig | None,
    effort_config: EffortLevelsConfig | None,
    token_ids: list[int],
    link_spans: list[tuple[str, int, int]],
) -> tuple[float, dict[str, int], dict[str, float]]:
    """Shape, then penalize. Token IDs and spans must already be verified by Gym."""
    counts = {"duplicated_reasoning": 0, "empty_final_answer": 0, "unwanted_token": 0}
    metrics: dict[str, float] = {}
    shaping = effort_shaping_enabled(effort_config)
    text_checks = penalty_config is not None and (
        penalty_config.penalize_duplicated_reasoning
        or penalty_config.penalize_empty_final_answer
    )
    if (shaping or text_checks) and checks is None:
        raise ValueError("missing_reward_checks")
    if shaping:
        assert effort_config is not None and checks is not None
        length = link_spans[-1][2]
        bucket = "low" if checks.low_effort else "high"
        metrics[f"finalize/effort/{bucket}/{length}"] = 1.0
        if checks.low_effort:
            reward, length_reward = shape_effort_reward(
                reward, length=length, config=effort_config
            )
            metrics["finalize/effort/length_reward_sum"] = length_reward
            metrics["finalize/effort/reward_sum"] = reward
    if penalty_config is not None:
        if checks is not None:
            counts["duplicated_reasoning"] = int(
                penalty_config.penalize_duplicated_reasoning
                and checks.duplicated_reasoning
            )
            counts["empty_final_answer"] = int(
                penalty_config.penalize_empty_final_answer and checks.empty_final_answer
            )
        if penalty_config.penalize_unwanted_tokens:
            assert penalty_config.token_ids is not None
            unwanted_ids = penalty_config.token_ids.unwanted
            assert unwanted_ids is not None
            offset = 0
            for _, carry, generated in link_spans:
                end = offset + carry + generated
                if has_unwanted_tokens(
                    (token_ids[offset + carry : end],),
                    unwanted_ids,
                ):
                    counts["unwanted_token"] = 1
                    break
                offset = end
    return (0.0 if any(counts.values()) else reward), counts, metrics


CAPTURE_PENALTY_METRICS = {
    "duplicated_reasoning": "reasoning_equal_to_final_answer_rate",
    "empty_final_answer": "empty_final_answer_rate",
    "unwanted_token": "unwanted_token_rate",
}


def aggregate_capture_reward_metrics(
    metrics: dict[str, list[float]],
) -> dict[str, float]:
    """Pool finalizer sufficient statistics over valid rows, never group rates."""
    count = sum(metrics.get("finalize/reward_count", []))
    if not count:
        return {}
    result = {"finalize/reward_count": count}
    for category, metric in CAPTURE_PENALTY_METRICS.items():
        key = f"finalize/penalty_count/{category}"
        if key in metrics:
            result[key] = sum(metrics[key])
            result[metric] = result[key] / count
    mean = sum(metrics["finalize/reward_sum"]) / count
    variance = max(0.0, sum(metrics["finalize/reward_sumsq"]) / count - mean * mean)
    result.update(
        {
            "total_reward/mean": mean,
            "total_reward/stddev": math.sqrt(variance * count / (count - 1))
            if count > 1
            else math.nan,
            "total_reward/min": min(metrics["finalize/reward_min"]),
            "total_reward/max": max(metrics["finalize/reward_max"]),
        }
    )
    for bucket in ("low", "high"):
        lengths = [
            int(name.rsplit("/", 1)[1])
            for name, values in metrics.items()
            if name.startswith(f"finalize/effort/{bucket}/")
            for _ in range(int(sum(values)))
        ]
        if lengths:
            result[f"mean_length_{bucket}"] = statistics.fmean(lengths)
            result[f"median_length_{bucket}"] = float(statistics.median(lengths))
            if bucket == "low":
                result["mean_length_reward_low"] = sum(
                    metrics["finalize/effort/length_reward_sum"]
                ) / len(lengths)
                result["mean_reward_low"] = sum(
                    metrics["finalize/effort/reward_sum"]
                ) / len(lengths)
    return result
