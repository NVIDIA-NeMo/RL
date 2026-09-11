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

"""Shared effort shaping and durable context for verified capture finalization."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, model_validator

EFFORT_CONTEXT_SCHEMA_VERSION = 1


class EffortLevelsConfig(BaseModel, extra="allow"):
    """Controls length-based reward shaping for low-effort prompts.

    When a prompt contains ``low_string``, the final reward is adjusted by a
    length-reward term that penalises overly long responses.  The reward formula
    is::

        length_reward = min(1, low_weight * (1 - response_len / low_ub))
        new_reward    = orig_reward
                      + orig_reward * max(length_reward, 0)
                      + low_penalty * min(length_reward, 0)

    Setting ``low_weight = 0`` or leaving ``low_string`` empty disables the
    shaping entirely.
    """

    low_weight: float = 0.0
    """Weight applied to the length-reward term.  Set to 0 to disable."""
    low_penalty: float = 1.0
    """Coefficient for the negative length-reward penalty."""
    low_ub: int = 64000
    """Response-length upper bound (in tokens) used to normalise the term."""
    low_string: str = ""
    """Substring that must appear in the user prompt to trigger shaping."""

    @model_validator(mode="after")
    def validate_length_bound(self) -> EffortLevelsConfig:
        if self.low_weight > 0 and self.low_string and self.low_ub <= 0:
            raise ValueError("active effort shaping requires low_ub > 0")
        return self


@dataclass
class EffortShapingMetrics:
    length_rewards_low: list[float]
    rewards_low: list[float]
    low_lengths: list[int]
    high_lengths: list[int]


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


def effort_fingerprint(config: EffortLevelsConfig) -> str:
    """Bind classification and formula semantics to the effective configuration."""
    encoded = json.dumps(
        [
            EFFORT_CONTEXT_SCHEMA_VERSION,
            config.low_weight,
            config.low_penalty,
            config.low_ub,
            config.low_string,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class RolloutEffortContext:
    """Token-free original prompt classification for one physical rollout."""

    schema_version: int
    semantics_fingerprint: str
    rollout_id: str
    is_low_effort: bool

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or type(self.is_low_effort) is not bool:
            raise ValueError(
                "effort context requires integer version and boolean classification"
            )
        if not isinstance(self.semantics_fingerprint, str) or not isinstance(
            self.rollout_id, str
        ):
            raise ValueError("effort context requires string semantics and identity")

    def state_dict(self) -> dict[str, Any]:
        """Serialize the explicit context schema for checkpoint ownership."""
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RolloutEffortContext:
        """Restore context without silently dropping fields or coercing flags."""
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "semantics_fingerprint",
            "rollout_id",
            "is_low_effort",
        }:
            raise ValueError("invalid effort context fields")
        return cls(**state)


def compute_effort_context(
    rollout_id: str, row: dict[str, Any], config: EffortLevelsConfig | None
) -> RolloutEffortContext | None:
    """Capture original prompt classification before a completion is sealed."""
    if not effort_shaping_enabled(config):
        return None
    assert config is not None
    return RolloutEffortContext(
        EFFORT_CONTEXT_SCHEMA_VERSION,
        effort_fingerprint(config),
        rollout_id,
        is_low_effort(row, config),
    )


def finalize_effort_reward(
    rollout_id: str,
    reward: float,
    *,
    terminal_length: int,
    context: RolloutEffortContext | None,
    config: EffortLevelsConfig | None,
) -> tuple[float, EffortShapingMetrics]:
    """Shape a raw reward using verified terminal length and sealed classification."""
    metrics = EffortShapingMetrics([], [], [], [])
    if not effort_shaping_enabled(config):
        if context is not None:
            raise ValueError("effort_context_config_mismatch")
        return reward, metrics
    assert config is not None
    if context is None:
        raise ValueError("missing_effort_context")
    if context.rollout_id != rollout_id:
        raise ValueError("effort_context_identity_mismatch")
    if (
        context.schema_version != EFFORT_CONTEXT_SCHEMA_VERSION
        or context.semantics_fingerprint != effort_fingerprint(config)
    ):
        raise ValueError("incompatible_effort_context")
    if terminal_length <= 0:
        raise ValueError("invalid_terminal_generation_length")
    if context.is_low_effort:
        reward, length_reward = shape_effort_reward(
            reward, length=terminal_length, config=config
        )
        metrics.length_rewards_low.append(length_reward)
        metrics.rewards_low.append(reward)
        metrics.low_lengths.append(terminal_length)
    else:
        metrics.high_lengths.append(terminal_length)
    return reward, metrics


def effort_shaping_metrics(shaping: EffortShapingMetrics) -> dict[str, float]:
    """Build the rollout-metric entries for one group's effort-shaping lists.

    Shared by the batched v1 path and the SingleController rollout manager so the
    two cannot drift apart.

    Args:
        shaping: Per-sample tracking lists returned by ``_apply_effort_shaping``.

    Returns:
        Metric name to value. Empty only when shaping was disabled; callers
        ``update`` an existing dict, so an absent key leaves the metric unreported
        rather than reporting a zero.
    """
    metrics: dict[str, float] = {}
    if shaping.length_rewards_low:
        metrics["mean_length_reward_low"] = sum(shaping.length_rewards_low) / len(
            shaping.length_rewards_low
        )
    if shaping.rewards_low:
        metrics["mean_reward_low"] = sum(shaping.rewards_low) / len(shaping.rewards_low)
    if shaping.low_lengths:
        metrics["mean_length_low"] = sum(shaping.low_lengths) / len(shaping.low_lengths)
        metrics["median_length_low"] = float(statistics.median(shaping.low_lengths))
    if shaping.high_lengths:
        metrics["mean_length_high"] = sum(shaping.high_lengths) / len(
            shaping.high_lengths
        )
        metrics["median_length_high"] = float(statistics.median(shaping.high_lengths))
    return metrics


def capture_effort_statistics(shaping: EffortShapingMetrics) -> dict[str, float]:
    """Encode exact length frequencies and sums in checkpoint-owned scalar metrics."""
    metrics: dict[str, float] = {}
    if shaping.low_lengths:
        metrics["finalize/effort/length_reward_sum"] = sum(shaping.length_rewards_low)
        metrics["finalize/effort/reward_sum"] = sum(shaping.rewards_low)
    for bucket, lengths in (
        ("low", shaping.low_lengths),
        ("high", shaping.high_lengths),
    ):
        for length in lengths:
            key = f"finalize/effort/{bucket}/{length}"
            metrics[key] = metrics.get(key, 0.0) + 1.0
    return metrics


def aggregate_capture_effort_metrics(
    metrics: dict[str, list[float]],
) -> dict[str, float]:
    """Pool exact histograms across groups; never average group means or medians."""
    result: dict[str, float] = {}
    for bucket in ("low", "high"):
        prefix = f"finalize/effort/{bucket}/"
        frequencies = sorted(
            (int(name[len(prefix) :]), int(sum(values)))
            for name, values in metrics.items()
            if name.startswith(prefix)
        )
        count = sum(frequency for _, frequency in frequencies)
        if not count:
            continue
        result[f"mean_length_{bucket}"] = (
            sum(length * frequency for length, frequency in frequencies) / count
        )
        # The middle two ranks coincide for odd populations.
        ranks = ((count - 1) // 2, count // 2)
        cumulative = 0
        middle: list[int] = []
        for length, frequency in frequencies:
            middle.extend(
                length for rank in ranks if cumulative <= rank < cumulative + frequency
            )
            cumulative += frequency
        result[f"median_length_{bucket}"] = sum(middle) / 2
        if bucket == "low":
            result["mean_length_reward_low"] = (
                sum(metrics["finalize/effort/length_reward_sum"]) / count
            )
            result["mean_reward_low"] = (
                sum(metrics["finalize/effort/reward_sum"]) / count
            )
    return result
