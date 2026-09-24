# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace propagation through real Ray actors.

``test_instrumentation.py`` exercises the same code against fakes, which is
enough for the logic but not for the part that actually broke: Ray validates
``.remote()`` arguments on the *caller*, against a signature it extracts by
unwrapping, so a decorator that looks correct in-process can still have every
dispatch rejected. These tests put the methods on a real actor and dispatch
across a process boundary.
"""

import pytest
import ray

from nemo_rl.telemetry.instrumentation import (
    accepts_trace_context,
    dispatch_with_trace_context,
    managed_span,
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


def _current_trace_id() -> int:
    """Trace id active in this process, 0 when there is none."""
    from opentelemetry import trace

    return trace.get_current_span().get_span_context().trace_id


@ray.remote(num_cpus=0)
class _Worker:
    """Reports the trace it is running under, rather than emitting spans.

    Asserting on the trace id keeps the check in the driver: collecting spans
    out of an actor would need an exporter and a flush per test, and the thing
    under test is whether the context arrived at all.
    """

    @accepts_trace_context
    def trace_id(self, value):
        return value, _current_trace_id()

    @accepts_trace_context
    async def trace_id_async(self, value):
        return value, _current_trace_id()

    @accepts_trace_context
    async def trace_ids_streamed(self, count):
        for index in range(count):
            yield index, _current_trace_id()

    def undecorated(self, value):
        return value, _current_trace_id()


@pytest.fixture
def telemetry():
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups="all"),
        span_exporter=InMemorySpanExporter(),
    )
    yield handle
    handle.shutdown()


@requires_lens
def test_a_decorated_actor_method_runs_under_the_callers_trace(telemetry):
    worker = _Worker.remote()
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=telemetry.tracer) as span:
        expected = span.get_span_context().trace_id
        value, trace_id = ray.get(dispatch_with_trace_context(worker.trace_id, 7))

    assert value == 7
    assert trace_id == expected


@requires_lens
def test_a_decorated_coroutine_method_runs_under_the_callers_trace(telemetry):
    worker = _Worker.remote()
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=telemetry.tracer) as span:
        expected = span.get_span_context().trace_id
        value, trace_id = ray.get(dispatch_with_trace_context(worker.trace_id_async, 7))

    assert value == 7
    assert trace_id == expected


@requires_lens
def test_every_step_of_a_streamed_method_runs_under_the_callers_trace(telemetry):
    """The context is attached per ``__anext__``, not held across the yield.

    Holding it across the yield is what made Ray's abandoned generators detach
    in the wrong context; attaching per step has to leave every item parented
    all the same.
    """
    worker = _Worker.remote()
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=telemetry.tracer) as span:
        expected = span.get_span_context().trace_id
        stream = dispatch_with_trace_context(
            worker.trace_ids_streamed.options(num_returns="streaming"), 3
        )
        items = [ray.get(ref) for ref in stream]

    assert [index for index, _ in items] == [0, 1, 2]
    assert {trace_id for _, trace_id in items} == {expected}


@requires_lens
def test_an_undecorated_actor_method_is_dispatched_without_the_carrier(telemetry):
    """Ray rejects the kwarg on the caller; the dispatch has to survive it.

    The fake in ``test_instrumentation.py`` raises the ``TypeError`` by hand.
    This is the real one, from Ray's own signature validation.
    """
    worker = _Worker.remote()
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=telemetry.tracer):
        value, trace_id = ray.get(dispatch_with_trace_context(worker.undecorated, 7))

    assert value == 7
    # No carrier reached it, so it ran outside the caller's trace.
    assert trace_id == 0


@requires_lens
def test_the_carrier_survives_an_options_handle(telemetry):
    """Both NeMo-Gym dispatch sites call ``.options(...)`` before ``.remote``."""
    worker = _Worker.remote()
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=telemetry.tracer) as span:
        expected = span.get_span_context().trace_id
        value, trace_id = ray.get(
            dispatch_with_trace_context(worker.trace_id.options(name="rollout"), 7)
        )

    assert value == 7
    assert trace_id == expected
