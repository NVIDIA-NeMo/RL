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
"""Shared text detectors and durable token-capture reward penalty evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

TEXT_PENALTY_EVIDENCE_SCHEMA_VERSION = 1


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


@dataclass(frozen=True)
class CaptureRewardPenaltyConfig:
    """Resolved internal config shared by receipt producers and finalizers."""

    duplicated_reasoning: bool
    empty_final_answer: bool
    unwanted_token_ids: tuple[int, ...]

    @classmethod
    def from_resolved(cls, config: dict[str, Any] | None) -> CaptureRewardPenaltyConfig:
        """Consume the existing resolver's output without tokenizer inference."""
        if config is None:
            return cls(False, False, ())
        if config.get("penalize_malformed_think_tag"):
            raise ValueError(
                "token capture does not support penalize_malformed_think_tag"
            )
        unwanted: tuple[int, ...] = ()
        if config.get("penalize_unwanted_tokens"):
            ids = (config.get("token_ids") or {}).get("unwanted")
            if not ids:
                raise ValueError("reward_penalties.token_ids.unwanted must be set")
            unwanted = tuple(ids)
        return cls(
            bool(config.get("penalize_duplicated_reasoning")),
            bool(config.get("penalize_empty_final_answer")),
            unwanted,
        )

    @property
    def semantics_fingerprint(self) -> str:
        """Version text semantics and effective text flags, independent of tokens."""
        return f"gym-output-v1:duplicated={int(self.duplicated_reasoning)}:empty={int(self.empty_final_answer)}"


@dataclass(frozen=True)
class RolloutTextPenaltyEvidence:
    """Versioned, token-free evidence bound to one physical rollout attempt."""

    schema_version: int
    semantics_fingerprint: str
    rollout_id: str
    duplicated_reasoning: bool | None
    empty_final_answer: bool | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ValueError("text penalty evidence schema_version must be an integer")
        if not isinstance(self.semantics_fingerprint, str) or not isinstance(
            self.rollout_id, str
        ):
            raise ValueError(
                "text penalty evidence requires string semantics and identity"
            )
        for value in (self.duplicated_reasoning, self.empty_final_answer):
            if value is not None and type(value) is not bool:
                raise ValueError("text penalty evidence checks must be bool or None")

    def state_dict(self) -> dict[str, Any]:
        """Serialize only the explicit evidence schema into checkpoint metadata."""
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RolloutTextPenaltyEvidence:
        """Preserve unknown semantic versions for finalizer compatibility checks."""
        expected = {
            "schema_version",
            "semantics_fingerprint",
            "rollout_id",
            "duplicated_reasoning",
            "empty_final_answer",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("invalid text penalty evidence fields")
        return cls(**state)


def compute_text_penalty_evidence(
    rollout_id: str,
    output_items: list[dict[str, Any]],
    config: CaptureRewardPenaltyConfig,
) -> RolloutTextPenaltyEvidence:
    """Evaluate the scored output before it leaves the environment actor."""
    return RolloutTextPenaltyEvidence(
        schema_version=TEXT_PENALTY_EVIDENCE_SCHEMA_VERSION,
        semantics_fingerprint=config.semantics_fingerprint,
        rollout_id=rollout_id,
        duplicated_reasoning=has_duplicated_reasoning(output_items)
        if config.duplicated_reasoning
        else None,
        empty_final_answer=has_empty_final_answer(output_items)
        if config.empty_final_answer
        else None,
    )


def finalize_reward_penalties(
    rollout_id: str,
    reward: float,
    evidence: RolloutTextPenaltyEvidence | None,
    config: CaptureRewardPenaltyConfig,
    token_ids: list[int],
    link_spans: list[tuple[str, int, int]],
) -> tuple[float, dict[str, int]]:
    """Apply penalties to a verified selected chain and sealed shaped reward."""
    counts = {"duplicated_reasoning": 0, "empty_final_answer": 0, "unwanted_token": 0}
    if config.duplicated_reasoning or config.empty_final_answer:
        if evidence is None:
            raise ValueError("missing_text_evidence")
        if evidence.rollout_id != rollout_id:
            raise ValueError("text_evidence_identity_mismatch")
        if (
            evidence.schema_version != TEXT_PENALTY_EVIDENCE_SCHEMA_VERSION
            or evidence.semantics_fingerprint != config.semantics_fingerprint
        ):
            raise ValueError("incompatible_text_evidence")
        for name, enabled, value in (
            (
                "duplicated_reasoning",
                config.duplicated_reasoning,
                evidence.duplicated_reasoning,
            ),
            (
                "empty_final_answer",
                config.empty_final_answer,
                evidence.empty_final_answer,
            ),
        ):
            if enabled:
                if value is None:
                    raise ValueError(f"unevaluated_text_evidence:{name}")
                counts[name] = int(value)
    if config.unwanted_token_ids:
        unwanted = set(config.unwanted_token_ids)
        offset = 0
        for _, carry_len, generation_len in link_spans:
            end = offset + carry_len + generation_len
            if carry_len < 0 or generation_len < 0 or end > len(token_ids):
                raise ValueError("invalid_penalty_token_span")
            if unwanted.intersection(token_ids[offset + carry_len : end]):
                counts["unwanted_token"] = 1
            offset = end
        if offset != len(token_ids):
            raise ValueError("penalty_token_span_coverage")
    return (0.0 if any(counts.values()) else reward), counts


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
    return result
