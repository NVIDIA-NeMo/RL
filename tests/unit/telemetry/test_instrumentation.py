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
    UMBRELLA_GROUPS,
    Bucket,
    bucket_for_efficiency_category,
    bucket_for_span_group,
    goodput_span_attributes,
    managed_span,
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
    assert (
        bucket_for_efficiency_category("wasted/failed_trajectory") is Bucket.WASTED
    )
    assert bucket_for_efficiency_category("init/total") is Bucket.OVERHEAD
    assert set(EFFICIENCY_CATEGORY_BUCKET)  # non-empty


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
    with managed_span(
        RLSpanGroup.STEP, "rl.grpo.step", tracer=handle.tracer
    ) as span:
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
