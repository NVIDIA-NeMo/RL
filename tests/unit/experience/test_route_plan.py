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
"""Strict deferred-route plan contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
    decode_route_plan,
    encode_route_plan,
)

_DIGEST = "0" * 64


def _span(
    staging_key: str,
    carry_len: int,
    generation_len: int,
    staged_route_len: int,
) -> RouteSpan:
    return RouteSpan(
        staging_key,
        carry_len,
        generation_len,
        staged_route_len,
        extras_digest_version=1,
        extras_digest=_DIGEST,
    )


def _plan() -> RouteAssemblyPlan:
    return RouteAssemblyPlan(
        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
        staging_partition="staging",
        spans=(
            _span("r/c0", carry_len=2, generation_len=2, staged_route_len=4),
            _span("r/c1", carry_len=3, generation_len=1, staged_route_len=1),
            _span("r/c2", carry_len=0, generation_len=0, staged_route_len=0),
            _span("r/c3", carry_len=1, generation_len=1, staged_route_len=0),
        ),
        cleanup_staging_keys=("r/c0", "r/c1", "r/c2", "r/c3", "r/off_chain"),
        expected_token_length=10,
    )


def test_route_plan_round_trip_is_strict_and_lossless() -> None:
    plan = _plan()
    assert decode_route_plan(encode_route_plan(plan)) == plan


def test_route_span_classification_uses_gyms_decision_table() -> None:
    # The decision table is Gym-owned; RL applies it via span metadata.
    pytest.importorskip("nemo_gym.token_id_capture.staging")
    from nemo_gym.token_id_capture.staging.routes import classify_route_span

    spans = _plan().spans
    assert [
        classify_route_span(
            carry_len=span.carry_len,
            generation_len=span.generation_len,
            staged_route_len=span.staged_route_len,
        )
        for span in spans
    ] == [
        "full",
        "tail",
        "sentinel",
        "sentinel",
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"schema_version": 999}, "unsupported route plan schema"),
        ({"expected_token_length": -1}, "must be non-negative"),
        ({"expected_token_length": 9}, "spans contribute 10 tokens"),
    ],
)
def test_route_plan_rejects_invalid_top_level_values(mutation, match) -> None:
    encoded = encode_route_plan(_plan())
    encoded.update(mutation)
    with pytest.raises(ValueError, match=match):
        decode_route_plan(encoded)


def test_route_plan_rejects_unknown_or_missing_fields() -> None:
    encoded = encode_route_plan(_plan())
    encoded["compat_guess"] = True
    with pytest.raises(ValueError, match="fields must be exactly"):
        decode_route_plan(encoded)

    del encoded["compat_guess"]
    del encoded["spans"][0]["staged_route_len"]
    with pytest.raises(ValueError, match="fields must be exactly"):
        decode_route_plan(encoded)


def test_route_plan_rejects_keys_outside_full_cleanup_manifest() -> None:
    encoded = encode_route_plan(_plan())
    encoded["cleanup_staging_keys"].remove("r/c1")
    with pytest.raises(ValueError, match="outside cleanup_staging_keys"):
        decode_route_plan(encoded)


def test_placeholder_plan_can_carry_length_without_route_reads() -> None:
    placeholder = RouteAssemblyPlan(
        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
        staging_partition="staging",
        spans=(),
        cleanup_staging_keys=("r/rejected",),
        expected_token_length=7,
    )
    assert decode_route_plan(encode_route_plan(placeholder)) == placeholder


def test_route_plan_boundary_round_trip() -> None:
    plan = _plan()
    repaired = replace(
        plan.spans[1], boundary_token_index=3, repair_previous_token=True
    )
    plan = replace(plan, spans=(plan.spans[0], repaired, *plan.spans[2:]))
    assert decode_route_plan(encode_route_plan(plan)) == plan


def test_legacy_v2_plan_preserves_legacy_semantics() -> None:
    plan = replace(_plan(), schema_version=2)
    encoded = encode_route_plan(plan)
    assert all("repair_previous_token" not in span for span in encoded["spans"])
    decoded = decode_route_plan(encoded)
    assert decoded == plan
    assert all(not span.repair_previous_token for span in decoded.spans)


@pytest.mark.parametrize(
    "updates",
    [
        {"boundary_token_index": True},
        {"boundary_token_index": -1},
        {"boundary_token_index": 4},
        {"repair_previous_token": 1},
        {"repair_previous_token": True},
    ],
)
def test_invalid_boundary_plan_is_not_silently_legacy(updates: dict) -> None:
    encoded = encode_route_plan(_plan())
    encoded["spans"][1].update(updates)
    with pytest.raises((TypeError, ValueError), match="boundary|repair_previous_token"):
        decode_route_plan(encoded)


def test_root_cannot_repair_a_previous_token() -> None:
    encoded = encode_route_plan(_plan())
    encoded["spans"][0].update(boundary_token_index=0, repair_previous_token=True)
    with pytest.raises(ValueError, match="boundary token index"):
        decode_route_plan(encoded)
