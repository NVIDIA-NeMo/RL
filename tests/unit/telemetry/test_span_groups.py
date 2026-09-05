# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RLSpanGroup presets and resolution."""

import pytest

# ``resolve()`` goes through lens's real SpanRegistry.
pytest.importorskip("nemo.lens")

from nemo.lens.groups import SpanRegistry

from nemo_rl.telemetry.span_groups import NAMESPACE, RLSpanGroup, register_span_groups

RL_GROUPS = frozenset(
    {
        "setup",
        "rollout",
        "generation",
        "logprob",
        "reward",
        "advantage",
        "policy_update",
        "reference_policy",
        "data_processing",
        "data_plane",
        "efficiency",
    }
)

# Held apart from RL_GROUPS, which the assertions below treat as "every phase a
# shipped preset should reach". ``per_prompt`` is a cardinality axis, not a
# phase, and is the one group deliberately reachable only from ``all`` or an
# explicit list -- its span count scales with the dataset, so folding it into
# ``per_step`` would make that preset's cost scale with prompts rather than
# steps. Its own tests below pin that placement.
PER_PROMPT_GROUPS = frozenset({"per_prompt"})

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
    assert PER_PROMPT_GROUPS <= RLSpanGroup.ALL_GROUPS
    assert {"job", "checkpoint", "evaluate", "step"} <= RLSpanGroup.ALL_GROUPS


def test_default_preset_is_coarse():
    assert RLSpanGroup.resolve("default") == frozenset(
        {"job", "checkpoint", "evaluate", "model_init", "setup"}
    )


def test_default_preset_covers_startup():
    """``setup`` and ``model_init`` travel together, in both shipped presets.

    Startup without the model load shows the worker build as one opaque block
    with its largest part missing, which is worse than not asking.
    """
    for preset in ("default", "per_step"):
        resolved = RLSpanGroup.resolve(preset)
        assert {"setup", "model_init"} <= resolved, preset


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

    ``per_prompt`` is the deliberate exception and so is not in
    ``EMITTED_GROUPS``; see ``PER_PROMPT_GROUPS``.
    """
    reachable = RLSpanGroup.resolve("default") | RLSpanGroup.resolve("per_step")
    assert EMITTED_GROUPS <= reachable, (
        f"only reachable from 'all': {sorted(EMITTED_GROUPS - reachable)}"
    )


def test_per_prompt_is_absent_from_both_shipped_presets():
    """The one group a preset must not turn on for you.

    ``rl.sc.generate_and_push`` and the rollout's ``rl.data_plane.put`` are ~2
    spans per prompt, so a 10k-prompt rollout emits ~20k where every other
    group in ``per_step`` emits a fixed handful per step. A user who picks
    ``per_step`` for step detail should not silently get dataset-sized volume.
    """
    for preset in ("default", "per_step"):
        assert "per_prompt" not in RLSpanGroup.resolve(preset), preset


def test_per_prompt_is_reachable_when_asked_for():
    assert "per_prompt" in RLSpanGroup.resolve("all")
    assert "per_prompt" in RLSpanGroup.resolve("per_step,per_prompt")


def test_asking_for_per_prompt_keeps_the_step_phases():
    """The documented opt-in spelling has to be additive, not a replacement."""
    resolved = RLSpanGroup.resolve("per_step,per_prompt")
    assert RL_GROUPS <= resolved
    assert "step" in resolved


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


def test_resolve_unknown_is_pending_rather_than_fatal():
    """An unknown entry must not raise, and must not silently vanish either.

    Lens cannot treat one as an error: a registry is per-process while a
    ``span_groups`` spec is job-wide, so a spawned process that imports fewer
    libraries would die on a value that is perfectly valid in the trainer. It
    comes back as ``pending`` instead, which is what lets the driver report a
    NeMo-RL typo without ending the run.
    """
    enabled, pending = RLSpanGroup.resolve_with_pending("nonexistent_group")
    assert enabled == frozenset()
    assert pending == frozenset({"nonexistent_group"})


def test_an_unknown_entry_does_not_discard_the_valid_ones():
    """A typo should cost the user that one entry, not the whole spec."""
    enabled, pending = RLSpanGroup.resolve_with_pending("generation,nonsense,reward")
    assert enabled == frozenset({"generation", "reward"})
    assert pending == frozenset({"nonsense"})


def test_importing_the_module_registers_the_groups():
    """Registration is an import side effect, and the whole scheme rests on it.

    Lens ships no group names, so nothing is selectable until NeMo-RL registers;
    a spec resolved before that silently enables nothing. ``setup.py`` imports
    this module ahead of ``setup_telemetry`` for exactly this reason.
    """
    assert NAMESPACE in SpanRegistry.namespaces()
    assert RLSpanGroup.ALL_GROUPS <= SpanRegistry.groups()


def test_registration_is_repeatable():
    """A cleared registry, or a re-import, must be able to re-register.

    ``register`` refuses a namespace it already holds unless told otherwise, so
    without ``allow_override`` this would raise the second time -- which in a
    test suite means the first test to clear the registry breaks every later
    one.
    """
    register_span_groups()
    register_span_groups()
    assert RLSpanGroup.ALL_GROUPS <= SpanRegistry.groups()


def test_the_all_preset_is_not_registered_as_a_preset():
    """``all`` is lens's reserved wildcard; registering it raises.

    It still has to resolve, and to everything -- it is computed from the live
    registry rather than the snapshot NeMo-RL took at import, which is what
    makes it correct when another library registers alongside.
    """
    assert "all" not in RLSpanGroup._PRESETS
    assert RLSpanGroup.resolve("all") == RLSpanGroup.ALL_GROUPS
