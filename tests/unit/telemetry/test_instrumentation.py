# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo_rl.telemetry.instrumentation``.

Two layers:

* Goodput bucket classification (``Bucket`` / ``bucket_for_span_group`` /
  ``goodput_span_attributes`` / efficiency-category mapping) — pure functions,
  no nemo-lens required.
* End-to-end span emission via the ``managed_span`` / ``trace_fn`` primitives
  the algorithm loops use, asserting spans emit per group, gate off when the
  group is disabled, carry ``rl.bucket`` on leaf groups, and nest correctly —
  requires nemo-lens.
"""

import pytest

from nemo_rl.telemetry.instrumentation import (
    EFFICIENCY_CATEGORY_BUCKET,
    RL_BUCKET_ATTR,
    RL_EFFICIENCY_CATEGORY_ATTR,
    UMBRELLA_GROUPS,
    Bucket,
    bucket_for_efficiency_category,
    bucket_for_span_group,
    bucket_scope,
    current_trace_carrier,
    efficiency_span,
    goodput_span_attributes,
    managed_span,
    remote_trace_context,
    trace_fn,
)
from nemo_rl.telemetry.span_groups import RLSpanGroup

try:
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _HAS_LENS = True
except ImportError:
    _HAS_LENS = False

requires_lens = pytest.mark.skipif(
    not _HAS_LENS, reason="nemo-lens (+ opentelemetry sdk) not installed"
)


# --------------------------------------------------------------------------- #
# Goodput bucket classification (pure functions)                              #
# --------------------------------------------------------------------------- #
def test_shared_bucket_tokens():
    assert {b.value for b in Bucket} == {
        "productive",
        "overhead",
        "idle",
        "wasted",
    }


def test_umbrellas_have_no_bucket():
    for group in (
        RLSpanGroup.JOB,
        RLSpanGroup.STEP,
        RLSpanGroup.ROLLOUT,
        RLSpanGroup.MODEL_INIT,
        RLSpanGroup.EVALUATE,
    ):
        assert group in UMBRELLA_GROUPS
        assert bucket_for_span_group(group) is None
        assert goodput_span_attributes(group) == {}


def test_leaf_groups_map_to_expected_buckets():
    assert bucket_for_span_group(RLSpanGroup.GENERATION) is Bucket.PRODUCTIVE
    assert bucket_for_span_group(RLSpanGroup.REWARD) is Bucket.PRODUCTIVE
    assert bucket_for_span_group(RLSpanGroup.POLICY_UPDATE) is Bucket.PRODUCTIVE
    assert bucket_for_span_group(RLSpanGroup.DATA_PROCESSING) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.CHECKPOINT) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.LOGPROB) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.ADVANTAGE) is Bucket.OVERHEAD
    assert bucket_for_span_group(RLSpanGroup.REFERENCE_POLICY) is Bucket.OVERHEAD


def test_goodput_span_attributes_shape():
    attrs = goodput_span_attributes(RLSpanGroup.GENERATION)
    assert attrs == {RL_BUCKET_ATTR: "productive"}


def test_unknown_non_umbrella_defaults_to_overhead():
    assert bucket_for_span_group("brand_new_leaf") is Bucket.OVERHEAD
    assert goodput_span_attributes("brand_new_leaf")[RL_BUCKET_ATTR] == "overhead"


def test_efficiency_categories_mapped():
    assert bucket_for_efficiency_category("idle/buffer_starvation") is Bucket.IDLE
    assert bucket_for_efficiency_category("wasted/failed_trajectory") is Bucket.WASTED
    assert bucket_for_efficiency_category("init/total") is Bucket.OVERHEAD
    assert set(EFFICIENCY_CATEGORY_BUCKET)  # non-empty


def _efficiency_categories_from_algorithms_utils() -> set[str]:
    """Every category async GRPO records, read from the canonical source."""
    from tests.unit.telemetry.conftest import algorithms_utils_categories

    found = algorithms_utils_categories(
        "WALL_CLOCK_EFFICIENCY_CATEGORIES",
        "THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES",
    )
    return set().union(*found.values())


def test_efficiency_category_bucket_matches_production_categories():
    """EFFICIENCY_CATEGORY_BUCKET duplicates the category strings that async
    GRPO actually records, so a new ``idle/*`` timer must not silently land
    without a bucket.
    """
    assert set(EFFICIENCY_CATEGORY_BUCKET) == (
        _efficiency_categories_from_algorithms_utils()
    )


@requires_lens
def test_efficiency_span_carries_bucket_and_category():
    handle, exporter = _setup("all")
    with efficiency_span("idle/refit_bubble", tracer=handle.tracer) as span:
        assert span is not None
    handle.shutdown()

    (emitted,) = exporter.get_finished_spans()
    assert emitted.name == "rl.idle.refit_bubble"
    assert emitted.attributes[RL_BUCKET_ATTR] == "idle"
    assert emitted.attributes[RL_EFFICIENCY_CATEGORY_ATTR] == "idle/refit_bubble"


@requires_lens
def test_efficiency_span_wasted_category_is_not_tagged_idle():
    handle, exporter = _setup("all")
    with efficiency_span("wasted/failed_trajectory", tracer=handle.tracer):
        pass
    handle.shutdown()

    (emitted,) = exporter.get_finished_spans()
    assert emitted.attributes[RL_BUCKET_ATTR] == "wasted"


@requires_lens
def test_trace_carrier_reparents_spans_in_another_context():
    """A carrier moves the trace across a process boundary Ray does not.

    Simulates the collector: capture inside the driver's job span, then reopen
    it where no span is active and check the child joins the same trace.
    """
    handle, exporter = _setup("all")
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer) as parent:
        carrier = current_trace_carrier()
        parent_ctx = parent.get_span_context()

    # Outside the parent block: nothing is active, so this would be a root.
    with remote_trace_context(carrier):
        with managed_span(
            RLSpanGroup.ROLLOUT, "rl.grpo.generation", tracer=handle.tracer
        ):
            pass
    handle.shutdown()

    child = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == "rl.grpo.generation"
    )
    assert child.parent is not None
    assert child.parent.span_id == parent_ctx.span_id
    assert child.context.trace_id == parent_ctx.trace_id


@requires_lens
def test_span_is_a_root_without_a_carrier():
    """No job span (the per_step case) must degrade to a root, not an error."""
    handle, exporter = _setup("all")
    carrier = current_trace_carrier()
    assert carrier == {}
    with remote_trace_context(carrier):
        with managed_span(
            RLSpanGroup.ROLLOUT, "rl.grpo.generation", tracer=handle.tracer
        ):
            pass
    handle.shutdown()

    (emitted,) = exporter.get_finished_spans()
    assert emitted.parent is None


@requires_lens
def test_collector_efficiency_spans_are_trace_only():
    """Collector-thread waits are visible in a trace but absent from rollups.

    They are timed concurrently with the driver's timeline, so a bucket would
    be summed against a wall-clock denominator it does not belong to. Omitting
    the attribute excludes them by construction instead of by convention.
    """
    handle, exporter = _setup("all")
    with efficiency_span("idle/generation_limit_pause", tracer=handle.tracer):
        pass
    with efficiency_span("idle/refit_event_wait", tracer=handle.tracer):
        pass
    handle.shutdown()

    emitted = exporter.get_finished_spans()
    assert [span.name for span in emitted] == [
        "rl.idle.generation_limit_pause",
        "rl.idle.refit_event_wait",
    ]
    for span in emitted:
        assert RL_BUCKET_ATTR not in span.attributes
        # The category still identifies the phase, and still says "idle/…",
        # so the span is greppable without being summable.
        assert span.attributes[RL_EFFICIENCY_CATEGORY_ATTR].startswith("idle/")


@requires_lens
def test_efficiency_span_is_gated_by_span_group():
    # The efficiency group is absent from the coarse "default" preset, so idle
    # spans must not appear there.
    handle, exporter = _setup("default")
    with efficiency_span("idle/buffer_starvation", tracer=handle.tracer):
        pass
    handle.shutdown()
    assert exporter.get_finished_spans() == ()


def test_efficiency_group_is_in_per_step_preset():
    # Per-step goodput only adds up if idle is included alongside the phases.
    assert RLSpanGroup.EFFICIENCY in RLSpanGroup._PRESETS["per_step"]
    assert RLSpanGroup.EFFICIENCY in RLSpanGroup.ALL_GROUPS
    assert RLSpanGroup.EFFICIENCY not in RLSpanGroup._PRESETS["default"]


def test_bucket_scope_replaces_leaf_bucket_and_restores_it():
    assert goodput_span_attributes(RLSpanGroup.GENERATION) == {
        RL_BUCKET_ATTR: "productive"
    }
    with bucket_scope(Bucket.OVERHEAD):
        assert goodput_span_attributes(RLSpanGroup.GENERATION) == {
            RL_BUCKET_ATTR: "overhead"
        }
    assert goodput_span_attributes(RLSpanGroup.GENERATION) == {
        RL_BUCKET_ATTR: "productive"
    }


def test_bucket_scope_leaves_umbrellas_unbucketed():
    """An override must not start bucketing umbrellas.

    ``rl.<algo>.evaluate`` encloses the generate spans it reclassifies, so
    tagging it too would count the same interval twice.
    """
    with bucket_scope(Bucket.OVERHEAD):
        assert goodput_span_attributes(RLSpanGroup.EVALUATE) == {}


def test_every_rl_span_group_is_classified():
    """Every known RLSpanGroup is either umbrella or has an explicit/default bucket."""
    for group in RLSpanGroup.ALL_GROUPS:
        bucket = bucket_for_span_group(group)
        if group in UMBRELLA_GROUPS:
            assert bucket is None, group
        else:
            assert bucket in Bucket, group


# --------------------------------------------------------------------------- #
# Span emission via managed_span / trace_fn (in-memory exporter)              #
# --------------------------------------------------------------------------- #
def _setup(groups):
    exporter = InMemorySpanExporter()
    cfg = NemoLensConfig(enabled=True, span_groups=groups, _span_group_cls=RLSpanGroup)
    handle = setup_telemetry(cfg, rank=0, world_size=1, span_exporter=exporter)
    return handle, exporter


@requires_lens
def test_managed_span_emits_when_group_enabled():
    handle, exporter = _setup("generation")
    with managed_span(
        RLSpanGroup.GENERATION,
        "rl.vllm.generate",
        tracer=handle.tracer,
        **{"rl.backend": "vllm"},
    ) as span:
        assert span is not None
    handle.shutdown()
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["rl.vllm.generate"]
    assert spans[0].attributes["rl.backend"] == "vllm"
    # Leaf groups carry rl.bucket for offline goodput rollup.
    assert spans[0].attributes[RL_BUCKET_ATTR] == "productive"


@requires_lens
def test_umbrella_span_has_no_bucket():
    handle, exporter = _setup("all")
    with managed_span(RLSpanGroup.STEP, "rl.grpo.step", tracer=handle.tracer) as span:
        assert span is not None
    handle.shutdown()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert RL_BUCKET_ATTR not in spans[0].attributes


@requires_lens
def test_managed_span_noop_when_group_disabled():
    # "generation" is not part of the "default" preset.
    handle, exporter = _setup("default")
    with managed_span(
        RLSpanGroup.GENERATION, "rl.vllm.generate", tracer=handle.tracer
    ) as span:
        assert span is None
    handle.shutdown()
    assert len(exporter.get_finished_spans()) == 0


@requires_lens
def test_trace_fn_job_span():
    handle, exporter = _setup("all")

    @trace_fn(RLSpanGroup.JOB, "rl.grpo.job")
    def train():
        return 42

    assert train() == 42
    handle.shutdown()
    assert any(s.name == "rl.grpo.job" for s in exporter.get_finished_spans())


@requires_lens
def test_validation_generation_is_overhead_not_productive():
    """Mirror of ``validate()``: evaluate umbrella + a generate span inside it.

    The generate span is opened by a decorator on ``VllmGeneration.generate``
    that cannot see whether the caller is a training rollout or validation, so
    the reclassification has to come from the enclosing scope.
    """
    handle, exporter = _setup("all")

    @trace_fn(RLSpanGroup.GENERATION, "rl.vllm.generate", tracer=handle.tracer)
    def generate():
        return "tokens"

    with (
        managed_span(RLSpanGroup.EVALUATE, "rl.grpo.evaluate", tracer=handle.tracer),
        bucket_scope(Bucket.OVERHEAD),
    ):
        generate()
    generate()  # a training rollout, outside the scope
    handle.shutdown()

    spans = exporter.get_finished_spans()
    evaluate = next(s for s in spans if s.name == "rl.grpo.evaluate")
    validation_gen, train_gen = (s for s in spans if s.name == "rl.vllm.generate")
    assert RL_BUCKET_ATTR not in evaluate.attributes
    assert validation_gen.attributes[RL_BUCKET_ATTR] == "overhead"
    assert train_gen.attributes[RL_BUCKET_ATTR] == "productive"


@requires_lens
def test_bucket_scope_reaches_generation_under_asyncio_run():
    """``run_multi_turn_rollout`` drives generation through ``asyncio.run``.

    A ``ContextVar`` survives that (the task copies the current context), which
    is what makes the scope usable from the synchronous ``validate()``.
    """
    import asyncio

    handle, exporter = _setup("all")

    @trace_fn(RLSpanGroup.GENERATION, "rl.vllm.generate", tracer=handle.tracer)
    def generate():
        return "tokens"

    async def rollout():
        generate()

    with bucket_scope(Bucket.OVERHEAD):
        asyncio.run(rollout())
    handle.shutdown()

    (emitted,) = exporter.get_finished_spans()
    assert emitted.attributes[RL_BUCKET_ATTR] == "overhead"


@requires_lens
def test_explicit_bucket_wins_over_scope():
    handle, exporter = _setup("all")
    with bucket_scope(Bucket.OVERHEAD):
        with managed_span(
            RLSpanGroup.GENERATION,
            "rl.vllm.generate",
            tracer=handle.tracer,
            **{RL_BUCKET_ATTR: "wasted"},
        ):
            pass
    handle.shutdown()

    (emitted,) = exporter.get_finished_spans()
    assert emitted.attributes[RL_BUCKET_ATTR] == "wasted"


@requires_lens
def test_step_nests_under_job():
    handle, exporter = _setup("all")
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer):
        with managed_span(RLSpanGroup.STEP, "rl.grpo.step", tracer=handle.tracer):
            pass
    handle.shutdown()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "rl.grpo.job" in spans and "rl.grpo.step" in spans
    step, job = spans["rl.grpo.step"], spans["rl.grpo.job"]
    assert step.parent is not None
    assert step.parent.span_id == job.context.span_id
