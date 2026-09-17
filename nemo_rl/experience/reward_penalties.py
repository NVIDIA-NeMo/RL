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
"""Shared reward calculations and logging for verified capture rows.

Capture observations retain each valid row's final reward until its group is
consumed, allowing exact pooled medians, histograms and agent reward logging.
Optional full-result JSON is retained only when full-result logging is enabled.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Container, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from nemo_rl.experience.metric_utils import calculate_single_metric

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


@dataclass(frozen=True)
class CapturePenalty:
    """One supported penalty's configuration, counter and logging contract."""

    name: str
    flag: str
    metric: str
    enabled: Callable[[RewardPenaltyConfig], bool]
    text_match: Callable[[RewardChecks], bool] | None


CAPTURE_PENALTIES = (
    CapturePenalty(
        "duplicated_reasoning",
        "penalize_duplicated_reasoning",
        "reasoning_equal_to_final_answer_rate",
        lambda config: config.penalize_duplicated_reasoning,
        lambda checks: checks.duplicated_reasoning,
    ),
    CapturePenalty(
        "empty_final_answer",
        "penalize_empty_final_answer",
        "empty_final_answer_rate",
        lambda config: config.penalize_empty_final_answer,
        lambda checks: checks.empty_final_answer,
    ),
    CapturePenalty(
        "unwanted_token",
        "penalize_unwanted_tokens",
        "unwanted_token_rate",
        lambda config: config.penalize_unwanted_tokens,
        None,
    ),
)


@dataclass(frozen=True)
class EffortRewardSettings:
    """Active effort parameters that affect saved reward interpretation."""

    low_weight: float
    low_penalty: float
    low_ub: int
    low_string: str


@dataclass(frozen=True)
class CaptureRewardSettings:
    """Semantic checkpoint identity; user-config extras are never persisted."""

    enabled_penalties: tuple[str, ...]
    unwanted_tokens: tuple[int, ...]
    effort: EffortRewardSettings | None

    @classmethod
    def from_configs(
        cls, penalties: RewardPenaltyConfig, effort: EffortLevelsConfig | None
    ) -> CaptureRewardSettings:
        if penalties.penalize_malformed_think_tag:
            raise ValueError("capture does not support penalize_malformed_think_tag")
        unwanted: tuple[int, ...] = ()
        if penalties.penalize_unwanted_tokens:
            assert penalties.token_ids is not None
            assert penalties.token_ids.unwanted is not None
            unwanted = tuple(sorted(set(penalties.token_ids.unwanted)))
        return cls(
            tuple(spec.flag for spec in CAPTURE_PENALTIES if spec.enabled(penalties)),
            unwanted,
            EffortRewardSettings(
                effort.low_weight, effort.low_penalty, effort.low_ub, effort.low_string
            )
            if effort is not None and effort_shaping_enabled(effort)
            else None,
        )

    def to_state(self) -> dict[str, Any]:
        """Serialize version 1 using the existing config field names."""
        penalties: dict[str, Any] = {flag: True for flag in self.enabled_penalties}
        if self.unwanted_tokens:
            penalties["token_ids"] = {"unwanted": list(self.unwanted_tokens)}
        return {
            "version": 1,
            "penalties": penalties,
            "effort": asdict(self.effort) if self.effort is not None else None,
        }

    @classmethod
    def from_state(cls, value: object) -> CaptureRewardSettings:
        """Validate version 1 or the original unversioned schema-5 settings.

        Existing model defaults fill absent config fields. A future change to
        reward semantics must explicitly version/migrate this projection.
        """
        # Config modules import this module for the shared calculations.
        from nemo_rl.algorithms.grpo import RewardPenaltyConfig
        from nemo_rl.experience.rollouts import EffortLevelsConfig

        if not isinstance(value, dict) or set(value) not in (
            {"penalties", "effort"},
            {"version", "penalties", "effort"},
        ):
            raise ValueError("invalid capture reward_settings record")
        if "version" in value and (
            type(value["version"]) is not int or value["version"] != 1
        ):
            raise ValueError(
                f"unsupported capture reward_settings version={value['version']!r}"
            )
        penalties = RewardPenaltyConfig.model_validate(value["penalties"])
        effort = (
            EffortLevelsConfig.model_validate(value["effort"])
            if value["effort"] is not None
            else None
        )
        return cls.from_configs(penalties, effort)

    def require_compatible(self, current: CaptureRewardSettings) -> None:
        """Reject changed reward rules with actionable field names."""
        differences = []
        for spec in CAPTURE_PENALTIES:
            if (spec.flag in self.enabled_penalties) != (
                spec.flag in current.enabled_penalties
            ):
                differences.append(f"penalties.{spec.flag}")
        if self.unwanted_tokens != current.unwanted_tokens:
            differences.append("penalties.token_ids.unwanted")
        if self.effort != current.effort:
            if self.effort is None or current.effort is None:
                differences.append("effort.enabled")
            else:
                previous = asdict(self.effort)
                for name, value in asdict(current.effort).items():
                    if value != previous[name]:
                        differences.append(f"effort.{name}")
        if differences:
            raise ValueError(
                "capture reward settings differ from the checkpoint: "
                + ", ".join(differences)
            )


@dataclass(frozen=True)
class RewardLogContext:
    """Agent identity and optional full-result table data, without raw reward."""

    agent_name: str
    full_result_json: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.agent_name, str) or not self.agent_name:
            raise ValueError("reward log agent_name must be a nonempty string")
        if self.full_result_json is not None:
            if not isinstance(self.full_result_json, str) or not isinstance(
                json.loads(self.full_result_json), dict
            ):
                raise ValueError("reward log full_result_json must encode an object")


@dataclass(frozen=True)
class FinalizedReward:
    """A valid finalized sample's reward, retained until group consumption."""

    sample_id: str
    rollout_id: str
    reward: float
    log_context: RewardLogContext | None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.sample_id, self.rollout_id)
        ):
            raise ValueError("finalized reward identifiers must be nonempty strings")
        if type(self.reward) not in (int, float):
            raise ValueError("finalized reward must be numeric")

    @classmethod
    def from_state(cls, value: object) -> FinalizedReward:
        """Validate checkpoint observations before using them for metrics."""
        if not isinstance(value, dict) or set(value) != {
            "sample_id",
            "rollout_id",
            "reward",
            "log_context",
        }:
            raise ValueError("invalid finalized reward observation")
        raw_context = value["log_context"]
        context = parse_reward_log_context(raw_context)
        return cls(value["sample_id"], value["rollout_id"], value["reward"], context)


def parse_reward_log_context(value: object) -> RewardLogContext | None:
    """Validate the optional log context stored with an unfinished attempt."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"agent_name", "full_result_json"}:
        raise ValueError("invalid reward log context")
    return RewardLogContext(value["agent_name"], value["full_result_json"])


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
    counts = {spec.name: 0 for spec in CAPTURE_PENALTIES}
    metrics: dict[str, float] = {}
    shaping = effort_shaping_enabled(effort_config)
    text_checks = penalty_config is not None and any(
        spec.text_match is not None and spec.enabled(penalty_config)
        for spec in CAPTURE_PENALTIES
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
            for spec in CAPTURE_PENALTIES:
                if spec.text_match is not None:
                    counts[spec.name] = int(
                        spec.enabled(penalty_config) and spec.text_match(checks)
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


CAPTURE_PENALTY_METRICS = {spec.name: spec.metric for spec in CAPTURE_PENALTIES}


def aggregate_capture_reward_metrics(
    metrics: dict[str, list[float]],
    observations: Sequence[FinalizedReward] | None = None,
) -> dict[str, Any]:
    """Pool finalizer sufficient statistics over valid rows, never group rates."""
    count = sum(metrics.get("finalize/reward_count", []))
    if observations is not None and len(observations) != count:
        raise ValueError("finalized reward observations must match the valid-row count")
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
    if observations is not None:
        result.update(
            calculate_single_metric(
                [row.reward for row in observations], len(observations), "total_reward"
            )
        )
        agent_rewards: dict[str, list[float]] = {}
        for row in observations:
            if row.log_context is not None:
                agent_rewards.setdefault(row.log_context.agent_name, []).append(
                    row.reward
                )
        for agent_name, values in agent_rewards.items():
            result.update(
                calculate_single_metric(values, len(values), f"{agent_name}/reward")
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


def capture_reward_result_tables(
    observations: Sequence[FinalizedReward],
) -> dict[str, list[list[str]]]:
    """Build the existing agent full-result table rows using final rewards."""
    tables: dict[str, list[list[str]]] = {}
    for row in observations:
        context = row.log_context
        if context is None or context.full_result_json is None:
            continue
        result = json.loads(context.full_result_json)
        result.update(
            reward=row.reward, ng_rollout_id=row.rollout_id, sample_id=row.sample_id
        )
        tables.setdefault(f"{context.agent_name}/full_result", []).append(
            [json.dumps(result, separators=(",", ":"))]
        )
    return tables
