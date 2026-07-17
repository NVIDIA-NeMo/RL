"""End-to-end span tests using an in-memory exporter.

Exercises the same primitives the algorithm loops use (``managed_span`` /
``trace_fn``) and asserts spans are emitted per group, gated off when the group
is disabled, and nest correctly.
"""

import pytest

pytest.importorskip("nemo.lens")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from nemo.lens import NemoLensConfig, setup_telemetry
from nemo.lens.helpers import managed_span, trace_fn
from nemo_rl.telemetry.span_groups import RLSpanGroup


def _setup(groups):
    exporter = InMemorySpanExporter()
    cfg = NemoLensConfig(enabled=True, span_groups=groups, _span_group_cls=RLSpanGroup)
    handle = setup_telemetry(cfg, rank=0, world_size=1, span_exporter=exporter)
    return handle, exporter


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


def test_managed_span_noop_when_group_disabled():
    # "generation" is not part of the "default" preset.
    handle, exporter = _setup("default")
    with managed_span(
        RLSpanGroup.GENERATION, "rl.vllm.generate", tracer=handle.tracer
    ) as span:
        assert span is None
    handle.shutdown()
    assert len(exporter.get_finished_spans()) == 0


def test_trace_fn_job_span():
    handle, exporter = _setup("all")

    @trace_fn(RLSpanGroup.JOB, "rl.grpo.job")
    def train():
        return 42

    assert train() == 42
    handle.shutdown()
    assert any(s.name == "rl.grpo.job" for s in exporter.get_finished_spans())


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
