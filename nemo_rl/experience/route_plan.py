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
"""Strict metadata contract for deferred routed-expert assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

ROUTE_PLAN_SCHEMA_VERSION = 2
_EXTRAS_DIGEST_VERSION = 1
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class RouteSpan:
    """One root-first route contribution to a canonical rollout row."""

    staging_key: str
    carry_len: int
    generation_len: int
    staged_route_len: int
    extras_digest_version: int
    extras_digest: str


@dataclass(frozen=True)
class RouteAssemblyPlan:
    """How a policy worker reconstructs routes from staged fragments."""

    schema_version: int
    staging_partition: str
    spans: tuple[RouteSpan, ...]
    cleanup_staging_keys: tuple[str, ...]
    expected_token_length: int


RouteSpanMode = Literal["full", "tail", "sentinel"]


def classify_route_span(span: RouteSpan) -> RouteSpanMode:
    """Apply Gym's route-linearization decision table using metadata only."""
    expected = span.carry_len + span.generation_len
    if span.staged_route_len > 0 and span.staged_route_len == expected:
        return "full"
    if span.staged_route_len > 0 and 0 < span.generation_len <= span.staged_route_len:
        return "tail"
    return "sentinel"


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, where: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} fields must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


def _require_int(value: Any, *, where: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{where} must be int, got {type(value).__name__}")
    return value


def _require_nonnegative_int(value: Any, *, where: str) -> int:
    parsed = _require_int(value, where=where)
    if parsed < 0:
        raise ValueError(f"{where} must be non-negative, got {parsed}")
    return parsed


def _require_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{where} must be a non-empty string")
    return value


def _validate_plan(plan: RouteAssemblyPlan) -> None:
    if plan.schema_version != ROUTE_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "unsupported route plan schema version "
            f"{plan.schema_version}; expected {ROUTE_PLAN_SCHEMA_VERSION}"
        )
    _require_string(plan.staging_partition, where="route_plan.staging_partition")
    _require_nonnegative_int(
        plan.expected_token_length,
        where="route_plan.expected_token_length",
    )
    cleanup_keys = set(plan.cleanup_staging_keys)
    if len(cleanup_keys) != len(plan.cleanup_staging_keys):
        raise ValueError("route_plan.cleanup_staging_keys contains duplicates")
    for index, key in enumerate(plan.cleanup_staging_keys):
        _require_string(key, where=f"route_plan.cleanup_staging_keys[{index}]")
    for index, span in enumerate(plan.spans):
        _require_string(
            span.staging_key, where=f"route_plan.spans[{index}].staging_key"
        )
        _require_nonnegative_int(
            span.carry_len, where=f"route_plan.spans[{index}].carry_len"
        )
        _require_nonnegative_int(
            span.generation_len,
            where=f"route_plan.spans[{index}].generation_len",
        )
        _require_nonnegative_int(
            span.staged_route_len,
            where=f"route_plan.spans[{index}].staged_route_len",
        )
        if (
            type(span.extras_digest_version) is not int
            or span.extras_digest_version != _EXTRAS_DIGEST_VERSION
        ):
            raise ValueError(
                f"route_plan.spans[{index}].extras_digest_version must be "
                f"{_EXTRAS_DIGEST_VERSION}"
            )
        if (
            not isinstance(span.extras_digest, str)
            or len(span.extras_digest) != _SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in span.extras_digest
            )
        ):
            raise ValueError(
                f"route_plan.spans[{index}].extras_digest must be a lowercase "
                "SHA-256 hex digest"
            )
        if span.staging_key not in cleanup_keys:
            raise ValueError(
                f"route_plan.spans[{index}] key {span.staging_key!r} is outside "
                "cleanup_staging_keys"
            )
        classify_route_span(span)
    if plan.spans:
        contribution = sum(span.carry_len + span.generation_len for span in plan.spans)
        if contribution != plan.expected_token_length:
            raise ValueError(
                f"route plan spans contribute {contribution} tokens, expected "
                f"{plan.expected_token_length}"
            )


def encode_route_plan(plan: RouteAssemblyPlan) -> dict[str, Any]:
    """Encode a validated plan into primitive ``KVBatchMeta.tags`` data."""
    _validate_plan(plan)
    return {
        "schema_version": plan.schema_version,
        "staging_partition": plan.staging_partition,
        "spans": [
            {
                "staging_key": span.staging_key,
                "carry_len": span.carry_len,
                "generation_len": span.generation_len,
                "staged_route_len": span.staged_route_len,
                "extras_digest_version": span.extras_digest_version,
                "extras_digest": span.extras_digest,
            }
            for span in plan.spans
        ],
        "cleanup_staging_keys": list(plan.cleanup_staging_keys),
        "expected_token_length": plan.expected_token_length,
    }


def decode_route_plan(value: Any) -> RouteAssemblyPlan:
    """Strictly decode a plan without defaults or compatibility guesses."""
    if not isinstance(value, dict):
        raise TypeError(f"route plan must be a dict, got {type(value).__name__}")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "staging_partition",
            "spans",
            "cleanup_staging_keys",
            "expected_token_length",
        },
        where="route_plan",
    )
    spans_value = value["spans"]
    if not isinstance(spans_value, list):
        raise TypeError("route_plan.spans must be a list")
    spans: list[RouteSpan] = []
    for index, span_value in enumerate(spans_value):
        if not isinstance(span_value, dict):
            raise TypeError(f"route_plan.spans[{index}] must be a dict")
        _require_exact_keys(
            span_value,
            {
                "staging_key",
                "carry_len",
                "generation_len",
                "staged_route_len",
                "extras_digest_version",
                "extras_digest",
            },
            where=f"route_plan.spans[{index}]",
        )
        spans.append(
            RouteSpan(
                staging_key=_require_string(
                    span_value["staging_key"],
                    where=f"route_plan.spans[{index}].staging_key",
                ),
                carry_len=_require_nonnegative_int(
                    span_value["carry_len"],
                    where=f"route_plan.spans[{index}].carry_len",
                ),
                generation_len=_require_nonnegative_int(
                    span_value["generation_len"],
                    where=f"route_plan.spans[{index}].generation_len",
                ),
                staged_route_len=_require_nonnegative_int(
                    span_value["staged_route_len"],
                    where=f"route_plan.spans[{index}].staged_route_len",
                ),
                extras_digest_version=_require_int(
                    span_value["extras_digest_version"],
                    where=f"route_plan.spans[{index}].extras_digest_version",
                ),
                extras_digest=_require_string(
                    span_value["extras_digest"],
                    where=f"route_plan.spans[{index}].extras_digest",
                ),
            )
        )
    cleanup_value = value["cleanup_staging_keys"]
    if not isinstance(cleanup_value, list):
        raise TypeError("route_plan.cleanup_staging_keys must be a list")
    cleanup_keys = tuple(
        _require_string(key, where=f"route_plan.cleanup_staging_keys[{index}]")
        for index, key in enumerate(cleanup_value)
    )
    plan = RouteAssemblyPlan(
        schema_version=_require_int(
            value["schema_version"], where="route_plan.schema_version"
        ),
        staging_partition=_require_string(
            value["staging_partition"], where="route_plan.staging_partition"
        ),
        spans=tuple(spans),
        cleanup_staging_keys=cleanup_keys,
        expected_token_length=_require_nonnegative_int(
            value["expected_token_length"],
            where="route_plan.expected_token_length",
        ),
    )
    _validate_plan(plan)
    return plan


def encoded_route_plan_size_bytes(plan: RouteAssemblyPlan) -> int:
    """Return the compact UTF-8 encoded size used for observability."""
    return len(
        json.dumps(encode_route_plan(plan), separators=(",", ":")).encode("utf-8")
    )
