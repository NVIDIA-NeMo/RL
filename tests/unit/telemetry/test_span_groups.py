# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RLSpanGroup presets and resolution."""

import pytest

# ``resolve()`` requires the real nemo-lens SpanGroup base class.
pytest.importorskip("nemo.lens")

from nemo_rl.telemetry.span_groups import RLSpanGroup

RL_GROUPS = frozenset(
    {
        "rollout",
        "generation",
        "logprob",
        "reward",
        "advantage",
        "policy_update",
        "reference_policy",
        "data_processing",
        "efficiency",
    }
)

# Every group NeMo-RL emits a span in, RL-specific and inherited alike. Keep in
# sync when instrumenting a new group -- that is the point of
# ``test_every_emitted_group_is_reachable_from_a_shipped_preset``.
#
# A superset on purpose: it also holds the groups that are defined and bucketed
# but have no call site yet (``reference_policy``; see the coverage gaps in
# docs/observability/span-groups.md), so the preset wiring is already correct
# when one of them is instrumented rather than needing a second edit here.
EMITTED_GROUPS = RL_GROUPS | frozenset(
    {"job", "step", "checkpoint", "evaluate", "model_init"}
)


def test_all_groups_includes_base_and_rl():
    assert RL_GROUPS <= RLSpanGroup.ALL_GROUPS
    assert {"job", "checkpoint", "evaluate", "step"} <= RLSpanGroup.ALL_GROUPS


def test_default_preset_is_coarse():
    assert RLSpanGroup.resolve("default") == frozenset(
        {"job", "checkpoint", "evaluate"}
    )


def test_per_step_has_step_and_phases_but_not_job():
    per_step = RLSpanGroup.resolve("per_step")
    assert "step" in per_step
    assert RL_GROUPS <= per_step
    # per_step deliberately omits JOB so each step is its own root trace.
    assert "job" not in per_step


def test_every_emitted_group_is_reachable_from_a_shipped_preset():
    """A group only in ``all`` is invisible to both presets users pick.

    ``model_init`` was in exactly that position: its one span,
    ``rl.vllm.load_model``, could not appear under ``default`` or ``per_step``,
    so the phase that explains a slow start was unobservable in practice.
    """
    reachable = RLSpanGroup.resolve("default") | RLSpanGroup.resolve("per_step")
    assert EMITTED_GROUPS <= reachable, (
        f"only reachable from 'all': {sorted(EMITTED_GROUPS - reachable)}"
    )


def test_all_preset_matches_all_groups():
    resolved = RLSpanGroup.resolve("all")
    assert "job" in resolved
    assert resolved == RLSpanGroup.ALL_GROUPS


def test_resolve_comma_list():
    assert RLSpanGroup.resolve("reward,generation") == frozenset(
        {"reward", "generation"}
    )


def test_resolve_is_case_insensitive():
    assert RLSpanGroup.resolve("DEFAULT") == RLSpanGroup.resolve("default")


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        RLSpanGroup.resolve("nonexistent_group")
