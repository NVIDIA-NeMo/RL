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

"""Instrumentation helpers that attach efficiency tags.

Algorithms should import ``managed_span`` / ``trace_fn`` from here (not raw
nemo-lens) so every leaf span gets ``rl.bucket`` when applicable.

Shared bucket tokens are ``productive`` | ``overhead`` | ``idle`` | ``wasted``.
Umbrella groups (``job``, ``step``, ``rollout``, …) are timed but not tagged.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, Iterator, Mapping, Optional

from nemo.lens import (
    is_span_group_enabled,
    span_cm,
)
from nemo.lens import (
    managed_span as _managed_span,
)

from nemo_rl.telemetry.span_groups import RLSpanGroup

# OTel / OneLogger-shared attribute key (flat sinks encode this in the name).
RL_BUCKET_ATTR = "rl.bucket"

# Raw efficiency-category label, so consumers can group idle time by cause
# without parsing it back out of the span name.
RL_EFFICIENCY_CATEGORY_ATTR = "rl.efficiency.category"

__all__ = [
    "managed_span",
    "trace_fn",
    "span_cm",
    "is_span_group_enabled",
    "RL_BUCKET_ATTR",
    "Bucket",
    "UMBRELLA_GROUPS",
    "EFFICIENCY_CATEGORY_BUCKET",
    "bucket_for_span_group",
    "bucket_for_efficiency_category",
    "current_trace_carrier",
    "remote_trace_context",
    "bucket_scope",
    "efficiency_span",
    "RL_EFFICIENCY_CATEGORY_ATTR",
]


class Bucket(str, Enum):
    """Shared goodput buckets."""

    PRODUCTIVE = "productive"
    OVERHEAD = "overhead"
    IDLE = "idle"
    WASTED = "wasted"


# Span groups that are umbrellas / lifecycle only — no rl.bucket tag.
UMBRELLA_GROUPS: frozenset[str] = frozenset(
    {
        RLSpanGroup.JOB,
        RLSpanGroup.STEP,
        RLSpanGroup.ROLLOUT,  # collect_rollouts umbrella (like Cosmos generate)
        RLSpanGroup.MODEL_INIT,
        RLSpanGroup.EVALUATE,  # eval pass; treat as umbrella unless timed as idle
    }
)

# Default classification for RLSpanGroup members that are leaf work.
# logprob / advantage / reference_policy count as overhead (prep), not the
# productive policy gradient update itself.
_DEFAULT_GROUP_BUCKET: Mapping[str, Bucket] = {
    RLSpanGroup.GENERATION: Bucket.PRODUCTIVE,
    RLSpanGroup.REWARD: Bucket.PRODUCTIVE,
    RLSpanGroup.POLICY_UPDATE: Bucket.PRODUCTIVE,
    RLSpanGroup.FORWARD_BACKWARD: Bucket.PRODUCTIVE,
    RLSpanGroup.OPTIMIZER: Bucket.PRODUCTIVE,
    RLSpanGroup.DATA_PROCESSING: Bucket.OVERHEAD,
    RLSpanGroup.CHECKPOINT: Bucket.OVERHEAD,
    RLSpanGroup.LOAD_CHECKPOINT: Bucket.OVERHEAD,
    RLSpanGroup.LOGPROB: Bucket.OVERHEAD,
    RLSpanGroup.ADVANTAGE: Bucket.OVERHEAD,
    RLSpanGroup.REFERENCE_POLICY: Bucket.OVERHEAD,
}

# Async efficiency category labels → bucket. Not RLSpanGroup members.
#
# The keys mirror ``WALL_CLOCK_EFFICIENCY_CATEGORIES`` +
# ``THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES`` in ``nemo_rl/algorithms/utils.py``;
# a test keeps the two lists from drifting apart.
#
# The two halves are NOT interchangeable as spans. The wall-clock half is
# driver-side and sequential, so :func:`efficiency_span` can emit it directly —
# ``idle/buffer_starvation`` / ``idle/refit_bubble`` do exactly that in
# ``nemo_rl/algorithms/grpo.py``.
#
# Two wall-clock categories stay Timer-only. ``idle/validation`` wraps
# ``validate()``, which is already accounted as ``overhead``: its
# ``rl.grpo.evaluate`` umbrella wraps the generate calls in
# :func:`bucket_scope`, so on the sync rollout path the ``rl.vllm.generate``
# spans inside it carry ``overhead``. A bucketed span over the same interval
# would be counted a second time by a rollup that sums durations by
# ``rl.bucket``, and as ``idle`` it would contradict the label its own children
# carry. (On the async path there are no such children — ``generate_async``
# carries no span today — but the window is the same one, so the same
# accounting applies.) ``init/total`` runs before the per-step loop, so it fills
# no step-level gap.
#
# The collector-side half cannot be summed against a driver-side denominator:
# it is timed in another process, concurrently with the driver's timeline, and
# the batch-worker categories accumulate across threads (thread-seconds), so
# they can exceed the wall time they happened in. Two of them are still worth
# seeing in a trace and are emitted as *unbucketed* spans — see
# :data:`COLLECTOR_LOOP_CATEGORIES`. The other two stay ``Timer``-only,
# reported as ``efficiency/*`` scalars from
# ``async_utils/trajectory_collector.py``: ``idle/buffer_full_backoff`` is a
# precomputed duration spanning a retry loop with no block to wrap, and
# ``wasted/failed_trajectory`` covers the same window as the enclosing
# ``rl.grpo.generation`` span.
#
# Every category below is exported as the ``rl.efficiency.seconds`` metric (see
# ``nemo_rl/telemetry/metrics.py``); the per-entry notes say which are *also*
# spans, and why the rest are metric-only.
EFFICIENCY_CATEGORY_BUCKET: Mapping[str, Bucket] = {
    "init/total": Bucket.OVERHEAD,  # metric only — pre-loop, fills no step gap
    "idle/buffer_starvation": Bucket.IDLE,  # metric + span rl.idle.buffer_starvation
    "idle/refit_bubble": Bucket.IDLE,  # metric + span rl.idle.refit_bubble
    "idle/validation": Bucket.IDLE,  # metric only — span double-counts generate
    "idle/buffer_full_backoff": Bucket.IDLE,  # metric only — thread-seconds
    # Metric + span, but the span is unbucketed (trace-only): timed on the
    # collector's loop thread, concurrently with the driver's timeline.
    "idle/generation_limit_pause": Bucket.IDLE,
    "idle/refit_event_wait": Bucket.IDLE,
    "wasted/failed_trajectory": Bucket.WASTED,  # metric only — thread-seconds
}


def bucket_for_span_group(group: str) -> Optional[Bucket]:
    """Return the goodput bucket for a span group, or None if umbrella / unknown.

    Unknown non-umbrella groups default to ``overhead`` so new leaves are not
    silently dropped from the denominator.
    """
    if group in UMBRELLA_GROUPS:
        return None
    if group in _DEFAULT_GROUP_BUCKET:
        return _DEFAULT_GROUP_BUCKET[group]
    return Bucket.OVERHEAD


def bucket_for_efficiency_category(category: str) -> Optional[Bucket]:
    """Return the bucket for an async efficiency category label, if known."""
    return EFFICIENCY_CATEGORY_BUCKET.get(category)


# The two waits on the collector's single collection-loop thread. Both consumers
# of this set follow from that one fact, so it is defined once:
#
# * As spans they are trace-only (no ``rl.bucket``). The phase is real and worth
#   seeing in a waterfall, but the collector's wall clock runs concurrently
#   with the driver's, so summing these against a driver-side denominator
#   overcounts. Leaving the attribute off means a bucket rollup skips them by
#   construction rather than by convention.
# * As metrics they are ``collector_wall_clock`` rather than ``thread_seconds``
#   (see ``nemo_rl/telemetry/metrics.py``): being single-threaded
#   ``Event.wait()`` calls, they cannot exceed the wall time they happened in,
#   unlike the batch-worker categories they otherwise sit beside.
COLLECTOR_LOOP_CATEGORIES: frozenset[str] = frozenset(
    {
        "idle/refit_event_wait",
        "idle/generation_limit_pause",
    }
)


# Caller-supplied reclassification for spans opened further down the stack.
# The same function can be productive or not depending on why it was called —
# a generate() during validation advances no weights — and the span is opened by
# a decorator that cannot see its caller, so the intent has to travel with the
# execution context rather than the argument list.
_BUCKET_OVERRIDE: ContextVar[Optional[Bucket]] = ContextVar(
    "nemo_rl_bucket_override", default=None
)


@contextmanager
def bucket_scope(bucket: Bucket) -> Iterator[None]:
    """Reclassify every leaf span opened inside this block as *bucket*.

    For phases whose goodput meaning is set by the caller, not by the callee's
    span group. Validation is the motivating case: it generates through the same
    :data:`RLSpanGroup.GENERATION` path as training rollouts, but the tokens are
    scored and discarded, so counting them as ``productive`` overstates goodput.

    Applies to the group-derived bucket only. Umbrella groups stay unbucketed,
    a span that passes ``rl.bucket`` explicitly keeps it, and an
    :func:`efficiency_span` keeps its category's bucket — that one names the
    phase it measures, so a caller cannot make ``idle/refit_bubble``
    productive. So wrapping a region cannot start double-counting an interval
    that its children already account for.

    Propagates like any :class:`~contextvars.ContextVar`: to nested calls,
    and to coroutines started inside the block (``asyncio.run`` copies the
    current context), but not to raw threads or other processes.
    """
    token = _BUCKET_OVERRIDE.set(bucket)
    try:
        yield
    finally:
        _BUCKET_OVERRIDE.reset(token)


def goodput_span_attributes(group: str) -> dict[str, str]:
    """Attributes to merge into ``managed_span`` for *group*.

    Empty when the group is an umbrella (no ``rl.bucket``). An enclosing
    :func:`bucket_scope` replaces the group's default bucket.
    """
    bucket = bucket_for_span_group(group)
    if bucket is None:
        return {}
    override = _BUCKET_OVERRIDE.get()
    return {RL_BUCKET_ATTR: (override or bucket).value}


def current_trace_carrier() -> dict[str, str]:
    """W3C ``traceparent`` carrier for the active span, to hand to another process.

    Ray does not propagate OTel context, so a worker's spans start their own
    trace unless the parent is passed explicitly. Capture this on the driver
    inside the span that should be the root, hand it to the actor, and reopen it
    there with :func:`remote_trace_context`.

    Returns an empty dict when there is no active recording span — which is the
    case whenever the enclosing span's group is disabled — so the caller needs
    no telemetry-specific branch.
    """
    # Via lens rather than opentelemetry.propagate directly: lens owns the
    # carrier format on both ends of a Ray hop, so a change there cannot leave
    # the two halves of this file's round-trip disagreeing.
    from nemo.lens.contrib.ray import inject_ray_context

    return inject_ray_context()


@contextmanager
def remote_trace_context(carrier: Optional[Mapping[str, str]]) -> Iterator[None]:
    """Parent every span opened in this block to the span in *carrier*.

    A no-op for an empty carrier, so an uninstrumented or job-span-disabled run
    keeps emitting root spans instead of failing.

    Attach per thread, not once per process: OTel context is a
    :class:`~contextvars.ContextVar`, and ``threading.Thread`` does not inherit
    them — a fire-and-forget worker thread starts with an empty context.
    """
    if not carrier:
        yield
        return
    # attach/detach come from opentelemetry because lens wraps the extraction
    # but not the activation.
    from nemo.lens.contrib.ray import extract_ray_context
    from opentelemetry import context as otel_ctx

    token = otel_ctx.attach(extract_ray_context(dict(carrier)))
    try:
        yield
    finally:
        otel_ctx.detach(token)


@contextmanager
def managed_span(
    group: str, name: str, tracer=None, **attributes: Any
) -> Iterator[Any]:
    """Like lens ``managed_span``, but injects ``rl.bucket`` for leaf groups.

    Callers may override by passing ``rl.bucket=...`` explicitly. Umbrella
    groups (job / step / rollout / …) receive no bucket attribute.
    """
    attrs = dict(attributes)
    if RL_BUCKET_ATTR not in attrs:
        attrs.update(goodput_span_attributes(group))
    with _managed_span(group, name, tracer=tracer, **attrs) as span:
        yield span


@contextmanager
def efficiency_span(category: str, tracer=None, **attributes: Any) -> Iterator[Any]:
    """Span for one efficiency category, tagged with that category's bucket.

    ``category`` is the same label the ``Timer`` uses (``"idle/refit_bubble"``,
    …), which keeps the span and the ``efficiency/*`` metric describing the
    identical phase. The bucket comes from
    :data:`EFFICIENCY_CATEGORY_BUCKET`, so ``idle/*`` lands in ``idle`` rather
    than defaulting to ``overhead`` the way an unknown leaf group would.

    Categories in :data:`COLLECTOR_LOOP_CATEGORIES` are emitted without a
    bucket — visible in a trace, invisible to a rollup. For the rest, two
    conditions have to hold at the call site. The phase must be measured on a
    single thread against wall time, since categories summed across concurrent
    threads are thread-seconds and would overcount (see
    :data:`EFFICIENCY_CATEGORY_BUCKET`). And the wrapped block must emit no
    bucketed child spans, because this span carries ``rl.bucket`` and a rollup
    that sums durations by bucket has no notion of nesting: a bucketed parent
    covering the same interval as its children is counted twice. Wrap a wait,
    not a phase that does instrumented work.
    """
    bucket = bucket_for_efficiency_category(category)
    if category in COLLECTOR_LOOP_CATEGORIES:
        bucket = None
    attrs: dict[str, Any] = {RL_EFFICIENCY_CATEGORY_ATTR: category}
    if bucket is not None:
        attrs[RL_BUCKET_ATTR] = bucket.value
    attrs.update(attributes)
    name = f"rl.{category.replace('/', '.')}"
    # The lens helper rather than the wrapper above: the bucket is decided here,
    # from the category, and the wrapper would fill in the EFFICIENCY group's
    # default (overhead) for the categories deliberately left unbucketed.
    with _managed_span(RLSpanGroup.EFFICIENCY, name, tracer=tracer, **attrs) as span:
        yield span


def trace_fn(group: str, name: str, tracer=None):
    """Decorator that wraps a function in a bucket-tagged ``managed_span``."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with managed_span(group, name, tracer=tracer):
                return func(*args, **kwargs)

        return wrapper

    return decorator
