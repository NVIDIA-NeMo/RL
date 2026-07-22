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
    }
)


def test_all_groups_includes_base_and_rl():
    assert RL_GROUPS <= RLSpanGroup.ALL_GROUPS
    assert {"job", "checkpoint", "evaluate", "step"} <= RLSpanGroup.ALL_GROUPS


def test_default_preset_is_coarse():
    assert RLSpanGroup.resolve("default") == frozenset({"job", "checkpoint", "evaluate"})


def test_per_step_has_step_and_phases_but_not_job():
    per_step = RLSpanGroup.resolve("per_step")
    assert "step" in per_step
    assert RL_GROUPS <= per_step
    # per_step deliberately omits JOB so each step is its own root trace.
    assert "job" not in per_step


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
