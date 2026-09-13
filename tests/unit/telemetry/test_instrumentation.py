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

import asyncio
import inspect
import logging

import pytest

from nemo_rl.telemetry.instrumentation import (
    EFFICIENCY_CATEGORY_BUCKET,
    RL_BUCKET_ATTR,
    RL_EFFICIENCY_CATEGORY_ATTR,
    TRACE_CARRIER_KWARG,
    UMBRELLA_GROUPS,
    Bucket,
    accepts_trace_context,
    bucket_for_efficiency_category,
    bucket_for_span_group,
    bucket_scope,
    current_trace_carrier,
    dispatch_with_trace_context,
    efficiency_span,
    goodput_span_attributes,
    in_per_prompt_scope,
    managed_span,
    per_prompt_scope,
    remote_trace_context,
    trace_context_kwargs,
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
    assert bucket_for_span_group(RLSpanGroup.DATA_PLANE) is Bucket.OVERHEAD


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
    """No job span (a group list without ``job``) degrades to a root, not an error."""
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


class _FakeRemoteMethod:
    """Stands in for ``actor.method``, recording what ``.remote`` received."""

    def __init__(self):
        self.args: tuple = ()
        self.kwargs: dict = {}

    def remote(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return "objectref"


@requires_lens
def test_dispatch_attaches_the_carrier_to_a_remote_call():
    handle, _ = _setup("all")
    method = _FakeRemoteMethod()
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer):
        dispatch_with_trace_context(method, "positional", keyword=1)
    handle.shutdown()

    assert method.args == ("positional",)
    assert method.kwargs["keyword"] == 1
    assert "traceparent" in method.kwargs[TRACE_CARRIER_KWARG]


@requires_lens
def test_dispatch_leaves_the_signature_alone_with_no_active_span():
    """An undecorated method must keep working when telemetry has nothing to send.

    Passing ``_otel_carrier=None`` instead would raise ``TypeError`` on every
    Ray method that has not been wired yet, turning a tracing gap into a crash.
    """
    handle, _ = _setup("all")
    method = _FakeRemoteMethod()
    dispatch_with_trace_context(method, "positional")
    handle.shutdown()

    assert method.kwargs == {}


@requires_lens
def test_trace_context_kwargs_is_spreadable_into_a_worker_group_call():
    """The form ``run_all_workers_*`` needs, since it takes the method by name."""
    handle, _ = _setup("all")
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer):
        with_span = trace_context_kwargs()
    without_span = trace_context_kwargs()
    handle.shutdown()

    assert "traceparent" in with_span[TRACE_CARRIER_KWARG]
    assert without_span == {}


@requires_lens
def test_a_sync_method_is_parented_to_its_caller():
    handle, exporter = _setup("all")

    @accepts_trace_context
    def worker_method(value):
        with managed_span(
            RLSpanGroup.DATA_PLANE, "rl.data_plane.get", tracer=handle.tracer
        ):
            return value

    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer) as parent:
        carrier = current_trace_carrier()
        parent_ctx = parent.get_span_context()
    assert worker_method(7, **{TRACE_CARRIER_KWARG: carrier}) == 7
    handle.shutdown()

    child = _named(exporter, "rl.data_plane.get")
    assert child.parent.span_id == parent_ctx.span_id


@requires_lens
def test_an_async_method_is_parented_to_its_caller():
    handle, exporter = _setup("all")

    @accepts_trace_context
    async def worker_method(value):
        with managed_span(
            RLSpanGroup.DATA_PLANE, "rl.data_plane.get", tracer=handle.tracer
        ):
            return value

    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer) as parent:
        carrier = current_trace_carrier()
        parent_ctx = parent.get_span_context()
    got = asyncio.run(worker_method(7, **{TRACE_CARRIER_KWARG: carrier}))
    handle.shutdown()

    assert got == 7
    child = _named(exporter, "rl.data_plane.get")
    assert child.parent.span_id == parent_ctx.span_id


@requires_lens
def test_an_async_generator_is_parented_to_its_caller():
    """The case lens's own ``traced_remote_call`` cannot cover.

    A sync wrapper returns the async generator without awaiting it, so the
    context is detached before the first item is produced. ``NemoGym.run_rollouts``
    is exactly this shape, so getting it wrong loses every Gym span.
    """
    handle, exporter = _setup("all")

    @accepts_trace_context
    async def worker_method(count):
        for index in range(count):
            with managed_span(
                RLSpanGroup.DATA_PLANE, "rl.data_plane.get", tracer=handle.tracer
            ):
                yield index

    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer) as parent:
        carrier = current_trace_carrier()
        parent_ctx = parent.get_span_context()

    async def drain():
        return [
            item async for item in worker_method(2, **{TRACE_CARRIER_KWARG: carrier})
        ]

    assert asyncio.run(drain()) == [0, 1]
    handle.shutdown()

    children = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "rl.data_plane.get"
    ]
    assert len(children) == 2
    assert {span.parent.span_id for span in children} == {parent_ctx.span_id}


@requires_lens
def test_a_method_without_a_carrier_still_runs():
    """Dispatch from an uninstrumented caller must not break the callee."""
    handle, exporter = _setup("all")

    @accepts_trace_context
    def worker_method(value):
        with managed_span(
            RLSpanGroup.DATA_PLANE, "rl.data_plane.get", tracer=handle.tracer
        ):
            return value

    assert worker_method(7) == 7
    handle.shutdown()
    assert _named(exporter, "rl.data_plane.get").parent is None


def test_the_carrier_is_stripped_before_the_method_below_sees_it():
    """The presharded entrypoints stack this over ``wrap_with_nvtx_name``.

    That wrapper takes ``**kwargs`` and forwards them verbatim, so a carrier
    that survived the decorator would reach the real method and raise
    ``TypeError`` on every dispatch -- turning a tracing feature into a
    training outage. The local double keeps the test off ``torch``.
    """

    def nvtx(name):
        def decorate(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        return decorate

    class Worker:
        @accepts_trace_context
        @nvtx("policy_worker/train_presharded")
        def train_presharded(self, meta, gbs=None):
            return meta, gbs

    worker = Worker()
    assert worker.train_presharded("meta", gbs=4) == ("meta", 4)
    assert worker.train_presharded(
        "meta", gbs=4, **{TRACE_CARRIER_KWARG: {"traceparent": "00-a-b-01"}}
    ) == ("meta", 4)


def test_the_decorator_preserves_the_callable_kind():
    """Ray inspects the method to decide how to run it.

    A coroutine function flattened to a plain one would be scheduled on the
    actor's sync path and never awaited.
    """

    @accepts_trace_context
    async def coroutine_method():
        return None

    @accepts_trace_context
    async def generator_method():
        yield None

    @accepts_trace_context
    def sync_method():
        return None

    assert inspect.iscoroutinefunction(coroutine_method)
    assert inspect.isasyncgenfunction(generator_method)
    assert not inspect.iscoroutinefunction(sync_method)
    assert not inspect.isasyncgenfunction(sync_method)


def _signature_ray_validates_against(method):
    """The signature Ray records for an actor method.

    ``_ActorClassMethodMetadata.create`` unwraps before extracting, which is
    the whole difficulty: a ``functools.wraps`` wrapper's ``**kwargs`` is
    invisible to it. Reproduced with ``inspect`` alone so the test does not
    need Ray.
    """
    return inspect.signature(inspect.unwrap(method))


@pytest.mark.parametrize(
    "shape",
    ["sync", "coroutine", "async_generator"],
)
def test_ray_can_bind_the_carrier_to_a_decorated_method(shape):
    """The kwarg has to be visible on the signature Ray validates against.

    Ray checks ``.remote()`` arguments on the *caller*, against the signature
    of the unwrapped original, so a wrapper that merely accepts ``**kwargs``
    gets the carrier rejected with ``TypeError`` before the task is submitted.
    Every Gym rollout dispatch failed this way until the decorator started
    declaring the parameter, and the failure is invisible to a test that calls
    the decorated function directly -- which is what the rest of this file does.
    """
    if shape == "sync":

        def method(self, examples, timer_prefix, dedupe=False):
            return None

    elif shape == "coroutine":

        async def method(self, examples, timer_prefix, dedupe=False):
            return None

    else:

        async def method(self, examples, timer_prefix, dedupe=False):
            yield None

    decorated = accepts_trace_context(method)
    signature = _signature_ray_validates_against(decorated)

    signature.bind(
        None, [1, 2], "gym", **{TRACE_CARRIER_KWARG: {"traceparent": "00-a-b-01"}}
    )
    # Still callable the way every un-instrumented caller calls it.
    signature.bind(None, [1, 2], "gym")


def test_a_method_taking_kwargs_is_left_alone():
    """``wrap_with_nvtx_name`` builds exactly this, and it needs no help.

    Its wrapper takes ``(*args, **kwargs)`` and does not use
    ``functools.wraps``, so unwrapping stops there and the carrier already
    binds. Declaring the parameter on top of ``**kwargs`` is redundant, and on
    a signature that has it would be a duplicate.
    """

    def takes_kwargs(self, *args, **kwargs):
        return None

    decorated = accepts_trace_context(takes_kwargs)

    assert (
        TRACE_CARRIER_KWARG
        not in _signature_ray_validates_against(decorated).parameters
    )
    _signature_ray_validates_against(decorated).bind(
        None, **{TRACE_CARRIER_KWARG: {"traceparent": "00-a-b-01"}}
    )


class _RefusesCarrier:
    """A Ray method that was never decorated, so Ray rejects the extra kwarg."""

    def __init__(self, name):
        self._method_name = name
        self.calls = []

    def remote(self, *args, **kwargs):
        if TRACE_CARRIER_KWARG in kwargs:
            raise TypeError(
                f"got an unexpected keyword argument '{TRACE_CARRIER_KWARG}'"
            )
        self.calls.append((args, kwargs))
        return "objectref"


@requires_lens
def test_a_method_refusing_the_carrier_degrades_instead_of_failing(caplog):
    """A tracing kwarg must not be able to abort a rollout.

    Ray validates arguments against the callee's recorded signature on the
    caller, so dispatching to an undecorated method raises before the task is
    even submitted. Every other telemetry path in this package warns and
    continues, and this one has to as well -- losing the parent link degrades a
    trace, where raising would fail the run.
    """
    handle, _ = _setup("all")
    method = _RefusesCarrier("undecorated_method")
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer):
        with caplog.at_level(logging.WARNING):
            assert dispatch_with_trace_context(method, 1, key="value") == "objectref"
        # Retried without the carrier, and the real arguments survived.
        assert method.calls == [((1,), {"key": "value"})]
        assert "accepts_trace_context" in caplog.text

        # One warning per method, not one per dispatch.
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            dispatch_with_trace_context(method, 2)
        assert "accepts_trace_context" not in caplog.text
    handle.shutdown()


@requires_lens
def test_an_unrelated_type_error_still_propagates():
    """Only the carrier kwarg is swallowed -- a real signature mistake is not."""

    class Broken:
        _method_name = "broken"

        def remote(self, *args, **kwargs):
            raise TypeError("missing 1 required positional argument: 'meta'")

    handle, _ = _setup("all")
    with managed_span(RLSpanGroup.JOB, "rl.grpo.job", tracer=handle.tracer):
        with pytest.raises(TypeError, match="required positional"):
            dispatch_with_trace_context(Broken())
    handle.shutdown()


def test_declaring_the_carrier_twice_does_not_duplicate_it():
    """Decoration can be repeated -- a re-imported module, a subclass override."""

    def method(self, value):
        return value

    once = accepts_trace_context(method)
    twice = accepts_trace_context(once)

    parameters = _signature_ray_validates_against(twice).parameters
    assert list(parameters) == ["self", "value", TRACE_CARRIER_KWARG]


def _named(exporter, name):
    """The one finished span called *name*."""
    return next(span for span in exporter.get_finished_spans() if span.name == name)


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


def test_setting_attributes_on_a_disabled_span_is_a_noop():
    """A disabled group yields None, and lens's helper dereferences the span.

    So the raw helper turns "telemetry is off" -- the default for every run --
    into an AttributeError inside production code. Absorbed in nemo_rl's
    wrapper, since a missing guard is invisible until it reaches a real run.
    """
    from nemo_rl.telemetry.instrumentation import safe_set_span_attributes

    safe_set_span_attributes(None, {"rl.anything": "value"})


@requires_lens
def test_a_disabled_setup_span_yields_none_rather_than_a_stub():
    """Pins the premise of the test above at the helper the launchers call."""
    from nemo_rl.telemetry.instrumentation import safe_set_span_attributes, setup_span

    handle, _ = _setup("rollout")
    with setup_span("ray_init", tracer=handle.tracer) as span:
        assert span is None
        safe_set_span_attributes(span, {"rl.ray.cluster_source": "started_local"})
    handle.shutdown()


@requires_lens
def test_the_initial_buffer_fill_span_is_trace_only():
    """The driver waits, but the generation fleet is busy for the same window.

    Both sides land in one trace, so bucketing the wait would charge that wall
    clock twice. The ``init/total`` *metric* keeps its bucket.
    """
    from nemo_rl.telemetry.instrumentation import bucket_for_efficiency_category

    handle, exporter = _setup("all")
    with efficiency_span("init/total", tracer=handle.tracer):
        pass
    handle.shutdown()

    (emitted,) = exporter.get_finished_spans()
    assert emitted.name == "rl.init.total"
    assert RL_BUCKET_ATTR not in emitted.attributes
    assert emitted.attributes[RL_EFFICIENCY_CATEGORY_ATTR] == "init/total"
    assert bucket_for_efficiency_category("init/total") is Bucket.OVERHEAD


@requires_lens
def test_the_u_aliases_are_the_same_groups_under_another_name():
    """Aliases, so no config, preset or ``span_groups`` spec has to change."""
    assert RLSpanGroup.U_ROLLOUT == RLSpanGroup.ROLLOUT == "rollout"
    assert RLSpanGroup.U_SETUP == RLSpanGroup.SETUP == "setup"
    assert {
        RLSpanGroup.U_JOB,
        RLSpanGroup.U_STEP,
        RLSpanGroup.U_MODEL_INIT,
        RLSpanGroup.U_EVALUATE,
        RLSpanGroup.U_ROLLOUT,
        RLSpanGroup.U_SETUP,
        RLSpanGroup.U_PER_PROMPT,
    } == UMBRELLA_GROUPS


def test_an_umbrella_span_is_the_same_span_minus_the_bucket():
    from nemo_rl.telemetry.instrumentation import umbrella_span

    handle, exporter = _setup("all")
    with umbrella_span(RLSpanGroup.U_ROLLOUT, "rl.test.rollout", tracer=handle.tracer):
        pass
    handle.shutdown()

    (span,) = exporter.get_finished_spans()
    assert span.name == "rl.test.rollout"
    assert RL_BUCKET_ATTR not in span.attributes


def test_a_leaf_group_at_an_umbrella_call_warns_and_keeps_its_bucket(caplog):
    """Degrade, do not raise: telemetry must not end a training run.

    Falling back to leaf semantics keeps the bucket the phase actually has.
    Emitting it unbucketed would be the worse failure, because it is
    indistinguishable downstream from a phase that is legitimately uncounted.
    """
    from nemo_rl.telemetry import instrumentation
    from nemo_rl.telemetry.instrumentation import umbrella_span

    instrumentation._LEAF_GROUP_AT_UMBRELLA_CALL.clear()
    handle, exporter = _setup("all")
    with caplog.at_level(logging.WARNING, logger=instrumentation.__name__):
        with umbrella_span(
            RLSpanGroup.GENERATION, "rl.test.generate", tracer=handle.tracer
        ):
            pass
    handle.shutdown()

    (span,) = exporter.get_finished_spans()
    assert span.attributes[RL_BUCKET_ATTR] == Bucket.PRODUCTIVE.value
    assert "is a leaf" in caplog.text


def test_the_leaf_group_warning_is_not_repeated_every_step(caplog):
    from nemo_rl.telemetry import instrumentation
    from nemo_rl.telemetry.instrumentation import umbrella_span

    instrumentation._LEAF_GROUP_AT_UMBRELLA_CALL.clear()
    with caplog.at_level(logging.WARNING, logger=instrumentation.__name__):
        for _ in range(3):
            with umbrella_span(RLSpanGroup.GENERATION, "rl.test.generate"):
                pass

    assert caplog.text.count("is a leaf") == 1


def test_a_leaf_group_at_an_umbrella_call_still_runs_the_body():
    """The body is the training work; a bad group must not skip it."""
    from nemo_rl.telemetry import instrumentation
    from nemo_rl.telemetry.instrumentation import umbrella_trace_fn

    instrumentation._LEAF_GROUP_AT_UMBRELLA_CALL.clear()

    @umbrella_trace_fn(RLSpanGroup.POLICY_UPDATE, "rl.test.train")
    def train() -> str:
        return "done"

    assert train() == "done"


def test_umbrella_trace_fn_returns_what_it_wraps():
    from nemo_rl.telemetry.instrumentation import umbrella_trace_fn

    handle, exporter = _setup("all")

    @umbrella_trace_fn(RLSpanGroup.U_JOB, "rl.test.job", tracer=handle.tracer)
    def job(value: int) -> int:
        return value * 2

    assert job(21) == 42
    handle.shutdown()

    (span,) = exporter.get_finished_spans()
    assert span.name == "rl.test.job"
    assert RL_BUCKET_ATTR not in span.attributes


def test_startup_spans_nest_without_a_bucket():
    """Startup phases nest and overlap, so none of them may carry a bucket."""
    from nemo_rl.telemetry.instrumentation import setup_span, startup_span

    handle, exporter = _setup("all")
    with startup_span(tracer=handle.tracer):
        with setup_span("ray_init", tracer=handle.tracer):
            pass
        with setup_span("workers", tracer=handle.tracer):
            with setup_span("policy", tracer=handle.tracer):
                pass
    handle.shutdown()

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    assert set(by_name) == {
        "rl.startup",
        "rl.setup.ray_init",
        "rl.setup.workers",
        "rl.setup.policy",
    }
    for span in by_name.values():
        assert RL_BUCKET_ATTR not in span.attributes
    # One trace, so a UI shows the phases as a waterfall rather than as roots.
    trace_ids = {span.context.trace_id for span in by_name.values()}
    assert len(trace_ids) == 1
    assert by_name["rl.setup.policy"].parent.span_id == (
        by_name["rl.setup.workers"].context.span_id
    )


@requires_lens
def test_startup_spans_are_gated_by_span_group():
    from nemo_rl.telemetry.instrumentation import setup_span, startup_span

    handle, exporter = _setup("rollout")
    with startup_span(tracer=handle.tracer):
        with setup_span("ray_init", tracer=handle.tracer):
            pass
    handle.shutdown()

    assert exporter.get_finished_spans() == ()


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


def test_per_prompt_scope_is_off_unless_entered():
    assert not in_per_prompt_scope()
    with per_prompt_scope():
        assert in_per_prompt_scope()
    assert not in_per_prompt_scope()


def test_per_prompt_scope_nests_without_leaking():
    """Nested rollouts must not switch it off on the inner exit."""
    with per_prompt_scope():
        with per_prompt_scope():
            assert in_per_prompt_scope()
        assert in_per_prompt_scope()
    assert not in_per_prompt_scope()


def test_per_prompt_scope_survives_an_exception():
    with pytest.raises(RuntimeError):
        with per_prompt_scope():
            raise RuntimeError("rollout failed")
    assert not in_per_prompt_scope()


def test_per_prompt_spans_carry_no_bucket():
    """Rollout-shaped work overlaps itself, so no bucket can hold it.

    Up to ``max_inflight_prompts`` rollouts are in flight at once, so any
    bucket on these would sum to a multiple of the wall clock.
    """
    assert RLSpanGroup.U_PER_PROMPT in UMBRELLA_GROUPS
    assert bucket_for_span_group(RLSpanGroup.PER_PROMPT) is None
    assert goodput_span_attributes(RLSpanGroup.PER_PROMPT) == {}


def test_a_bucket_scope_cannot_bucket_per_prompt_work():
    """The rollout path runs inside no scope today, but must stay safe if it does."""
    with bucket_scope(Bucket.PRODUCTIVE):
        assert goodput_span_attributes(RLSpanGroup.PER_PROMPT) == {}


def test_the_u_alias_for_per_prompt_is_the_same_group():
    assert RLSpanGroup.U_PER_PROMPT == RLSpanGroup.PER_PROMPT


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
    # No span-group class to pass and no rank to declare: the spec resolves
    # against the SpanRegistry that importing RLSpanGroup populated, and rank is
    # a resource attribute rather than an argument.
    cfg = NemoLensConfig(enabled=True, span_groups=groups)
    handle = setup_telemetry(cfg, span_exporter=exporter)
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
