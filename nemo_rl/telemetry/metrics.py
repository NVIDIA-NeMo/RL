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

"""Best-effort mirroring of NeMo-RL's per-step metrics into OTel.

Kept out of ``nemo_rl.utils.logger`` (which pulls in torch/ray/wandb/etc.) so
the mapping stays importable and testable without the heavy training stack.
``nemo_rl.utils.logger.Logger.log_metrics`` calls :func:`tee_rl_metrics_to_otel`
after its normal fan-out to the file/wandb/mlflow backends.

Every series is declared by NeMo-RL rather than by lens, via lens's
consumer-driven metric registry: NeMo-RL owns the ``rl.*`` metric names, so a
new series needs no lens release and no negotiation over field names. Four
families are teed — the async ``efficiency/*`` phase durations, the training
scalars, the ``vllm/*`` engine deltas, and the ``timing/setup`` startup phases.
The first three ride the per-step ``train`` dicts; the last arrives once, at
step 0.

This module names no logger key of its own beyond the ``efficiency/*`` family
it also produces. The scalar rows come from the modules that log them, through
:mod:`nemo_rl.telemetry.vocabulary`, so adding an algorithm does not mean
editing telemetry.

``Logger.log_metrics`` fans a step out as several dicts under different
prefixes, and a key is only reachable from the prefix its own dict carries — so
only the prefixes those dicts actually arrive under are teed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional

from nemo_rl.algorithms.metric_utils import SETUP_TIMING_PREFIX, SetupTimingMetrics
from nemo_rl.telemetry.instrumentation import (
    COLLECTOR_LOOP_CATEGORIES,
    RL_BUCKET_ATTR,
    RL_EFFICIENCY_CATEGORY_ATTR,
    bucket_for_efficiency_category,
)
from nemo_rl.telemetry.setup import get_telemetry_handle
from nemo_rl.telemetry.vocabulary import (
    RUN_WINDOW_WALL_CLOCK_CATEGORIES,
    as_scalar,
    recorded_metrics,
    registry_key,
    teed_metrics,
)

if TYPE_CHECKING:
    from nemo.lens.instruments import MetricSpec
    from opentelemetry.metrics import Meter

logger = logging.getLogger(__name__)

# Prefixes carrying the per-step families. The efficiency, training-scalar and
# vLLM dicts all arrive under the driver's train prefixes, and a key is only
# reachable from the prefix its own dict is logged under, so looking anywhere
# else would be dead work per step. ``SETUP_TIMING_PREFIX`` is the one
# non-per-step prefix teed.
_TRAIN_PREFIXES: tuple[Optional[str], ...] = ("train", "")

# One dimensioned gauge rather than an instrument per category, so adding a
# category needs no instrument change.
RL_EFFICIENCY_SECONDS_METRIC = "rl.efficiency.seconds"
RL_EFFICIENCY_PCT_METRIC = "rl.efficiency.pct"

# How a duration relates to wall time, which decides what a consumer may sum:
#
# * ``wall_clock`` — measured on the driver, sequentially, against the same
#   timeline as the step. Safe to sum against a driver-side denominator.
# * ``collector_wall_clock`` — measured on the collector's single collection-loop
#   thread, so the durations are sequential and honest, but they belong to the
#   collector's timeline, which runs concurrently with the driver's. Do not add
#   them to a driver-side denominator; read them per-phase.
# * ``thread_seconds`` — accumulated across concurrent batch-worker threads and
#   so able to exceed the wall time it happened in. A saturation signal, not a
#   duration.
RL_EFFICIENCY_MEASUREMENT_ATTR = "rl.efficiency.measurement"
WALL_CLOCK_MEASUREMENT = "wall_clock"
COLLECTOR_WALL_CLOCK_MEASUREMENT = "collector_wall_clock"
THREAD_SECONDS_MEASUREMENT = "thread_seconds"

# What period a value covers, which decides whether summing it *over time* is
# meaningful -- an orthogonal question to ``measurement`` above, and one a
# dashboard gets wrong silently:
#
# * ``step`` — a per-step delta, because the driver resets its Timer each step.
#   Sums across steps.
# * ``run`` — cumulative since the process started, so consecutive points
#   already contain each other. Summing across steps multiplies it by the step
#   count. Covers the collector's categories (its Timer is never reset) and
#   ``init/total``, which is measured once before the loop and then republished
#   unchanged so it does not vanish from the dashboard after step 1.
RL_EFFICIENCY_WINDOW_ATTR = "rl.efficiency.window"
STEP_WINDOW = "step"
RUN_WINDOW = "run"

# Startup phase durations, logged once at step 0 by every algorithm's setup
# (``SetupTimingMetrics.to_metrics_dict`` in grpo.py / single_controller.py, a
# plain dict of the same shape in ppo.py).
#
# Dimensioned like the efficiency series, and for a stronger reason: alongside
# its declared fields ``SetupTimingMetrics`` carries a free-form ``extras`` dict
# (the sparse-refit transports add ``vllm_<transport>_sparse_init_time_s`` at
# runtime), so the phase set is not knowable at declaration time. A row per
# phase would silently drop whatever it had not been taught about.
RL_SETUP_DURATION_METRIC = "rl.setup.duration"
RL_SETUP_PHASE_ATTR = "rl.setup.phase"

# Keys that ``print_efficiency_summary`` puts in the Logger dict.
_EFFICIENCY_KEY_PREFIX = "efficiency/"
_EFFICIENCY_SECONDS_SUFFIX = "_s"
_EFFICIENCY_PCT_KEY = "efficiency/efficiency_pct"
_EFFICIENCY_PCT_PER_STEP_KEY = "efficiency/efficiency_pct_is_per_step"

# --- lens metric registry -------------------------------------------------
#
# lens stopped shipping an ``rl`` instrument module: a consuming framework now
# declares the metric families whose *names* it owns and records against them
# (lens PR #46). Every name below is therefore NeMo-RL's to choose, which is
# what removed the blocker that previously kept the training scalars off OTel
# entirely -- they had to be mapped onto lens's fixed keyword fields, and that
# mapping was never settled.
RL_METRIC_GROUP = "rl"

# Registry keys for the dimensioned series this module produces itself. Unlike
# the teed rows these carry no logger key: one series covers every category,
# with the category in an attribute.
_EFFICIENCY_SECONDS_KEY = registry_key(RL_EFFICIENCY_SECONDS_METRIC)
_EFFICIENCY_PCT_KEY_REGISTRY = registry_key(RL_EFFICIENCY_PCT_METRIC)
_SETUP_DURATION_KEY = registry_key(RL_SETUP_DURATION_METRIC)

# Registration is process-global in lens, so it has to happen once per process
# and in every process that records -- driver and workers alike. Doing it lazily
# on the first tee rather than at import keeps this module importable when the
# installed lens predates the registry, which matters because
# nemo_rl.utils.logger imports the tee at module scope: an import-time failure
# here would take down training, not just telemetry.
_REGISTERED = False

#: Set when registration failed for a reason retrying cannot change, so the
#: per-step tee stops re-attempting it. Separate from _REGISTERED because the
#: two answer different questions: whether the group is usable, and whether
#: asking again is worth the work.
_REGISTRATION_FAILED = False

_WARNED: set[str] = set()


def warn_once(key: str, message: str) -> None:
    """Warn with a traceback the first time *key* fails, then stay quiet.

    Telemetry failures are typically per-step and deterministic -- a broken
    instrument fails identically on every step -- so warning each time would put
    thousands of identical tracebacks in a run's log while telling the reader
    nothing the first one did not. Warning level
    once, so a permanently dead sink is visible at default verbosity; debug
    afterwards, so the repetition is still recoverable when someone is looking
    for it.
    """
    if key in _WARNED:
        logger.debug(message, exc_info=True)
        return
    _WARNED.add(key)
    logger.warning(message, exc_info=True)


def _metric_specs() -> list[MetricSpec]:
    """Build the full ``MetricSpec`` list for the ``rl`` group."""
    from nemo.lens.instruments import MetricSpec

    specs = [
        MetricSpec(
            _EFFICIENCY_SECONDS_KEY,
            RL_EFFICIENCY_SECONDS_METRIC,
            "gauge",
            unit="s",
            description="Time attributed to one async efficiency category.",
        ),
        MetricSpec(
            _EFFICIENCY_PCT_KEY_REGISTRY,
            RL_EFFICIENCY_PCT_METRIC,
            "gauge",
            unit="%",
            description=(
                "Productive share of driver-side wall clock, over the window "
                "named by the rl.efficiency.window attribute."
            ),
        ),
    ]
    specs.append(
        MetricSpec(
            _SETUP_DURATION_KEY,
            RL_SETUP_DURATION_METRIC,
            "gauge",
            unit="s",
            description=(
                "Wall clock spent in one startup phase, named by the "
                "rl.setup.phase attribute."
            ),
        )
    )
    specs.extend(
        MetricSpec(
            row.key,
            row.name,
            row.kind,
            unit=row.unit,
            description=row.description,
        )
        for row in (*teed_metrics(), *recorded_metrics())
    )
    return specs


def ensure_metric_group_registered() -> bool:
    """Declare the ``rl`` metric group with lens, once per process.

    Returns:
        Whether the group is available to record against. ``False`` means the
        installed lens has no metric registry, which makes every tee below a
        no-op rather than an error.
    """
    global _REGISTERED, _REGISTRATION_FAILED
    if _REGISTERED:
        return True
    # Both failures below are permanent for the process, so the attempt is not
    # repeated: the tee runs once per log_metrics, and retrying would rebuild
    # every MetricSpec each step for a group that will never register.
    if _REGISTRATION_FAILED:
        return False
    try:
        from nemo.lens.instruments import register_metric_group

        register_metric_group(RL_METRIC_GROUP, _metric_specs())
    except (ImportError, AttributeError):
        _REGISTRATION_FAILED = True
        warn_once(
            "registry",
            "could not declare the 'rl' metric group; RL metrics will not be "
            "exported (a lens build with the metric registry is required)",
        )
        return False
    except ValueError as exc:
        # Separated because this is our declaration, not the dependency: lens
        # raises ValueError for a duplicate key, an empty spec list or a second
        # registration of the group. Reporting those as "install a newer lens"
        # sends the reader to the wrong file entirely.
        _REGISTRATION_FAILED = True
        warn_once(
            "registry",
            f"the '{RL_METRIC_GROUP}' metric group is declared invalidly, so RL "
            f"metrics will not be exported; fix _metric_specs(): {exc}",
        )
        return False
    _REGISTERED = True
    return True


def _record(
    meter: Meter,
    values: Mapping[str, float],
    attributes: Optional[Mapping[str, str]] = None,
) -> None:
    """Record against the ``rl`` group. ``record_metrics`` never raises."""
    from nemo.lens.instruments import record_metrics

    record_metrics(meter, RL_METRIC_GROUP, values, attributes=attributes)


def record_rl_metrics(
    values: Mapping[str, float],
    attributes: Optional[Mapping[str, str]] = None,
) -> None:
    """Record declared series by registry key. No-op unless exporting.

    For values that never pass through ``Logger.log_metrics`` and so have no
    teed row. Declare them with ``register_recorded_metrics`` first.
    """
    handle = get_telemetry_handle()
    if handle is None or not handle.is_exporting:
        return
    if not ensure_metric_group_registered():
        return
    _record(handle.meter, values, attributes)


def efficiency_measurements() -> dict[str, str]:
    """Map each canonical efficiency category to its measurement kind.

    Returns:
        ``{category: "wall_clock" | "collector_wall_clock" | "thread_seconds"}``,
        or an empty dict when the training stack is unavailable — which makes the
        efficiency tee a no-op instead of an import error.
    """
    # Deferred: nemo_rl.algorithms.utils pulls in torch, and this module is
    # deliberately importable without the training stack. Reading the canonical
    # lists rather than restating them keeps the two from drifting.
    try:
        from nemo_rl.algorithms.utils import (
            THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES,
            WALL_CLOCK_EFFICIENCY_CATEGORIES,
        )
    except ImportError:
        return {}

    measurements = {
        category: WALL_CLOCK_MEASUREMENT
        for category in WALL_CLOCK_EFFICIENCY_CATEGORIES
    }
    # The canonical list groups everything collector-side together, because the
    # W&B summary only needs "not the driver's clock". Here the split matters:
    # calling the collection-loop waits thread_seconds would tell a consumer
    # they can exceed wall time, which is untrue of a single-threaded
    # Event.wait().
    for category in THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES:
        measurements[category] = (
            COLLECTOR_WALL_CLOCK_MEASUREMENT
            if category in COLLECTOR_LOOP_CATEGORIES
            else THREAD_SECONDS_MEASUREMENT
        )
    return measurements


def efficiency_window(category: str, measurement: str) -> str:
    """Return the period one efficiency value covers (see the constants above)."""
    if measurement == WALL_CLOCK_MEASUREMENT:
        return (
            RUN_WINDOW if category in RUN_WINDOW_WALL_CLOCK_CATEGORIES else STEP_WINDOW
        )
    return RUN_WINDOW


def map_efficiency_seconds(
    metrics: dict[str, Any],
    measurement_by_category: Mapping[str, str],
) -> dict[str, float]:
    """Extract ``{category: seconds}`` from a raw Logger metrics dict.

    Looks up only the categories in *measurement_by_category*, which keeps the
    aggregate ``efficiency/*`` keys (``total_waste_s``, ``productive_time_s``,
    ``total_wall_time_s``, ``thread_seconds_total_s``) out of the per-category
    series even though they share the prefix and suffix.

    Pure function (no OTel side effects) so it is trivially unit-testable.
    """
    seconds: dict[str, float] = {}
    for category in measurement_by_category:
        key = f"{_EFFICIENCY_KEY_PREFIX}{category}{_EFFICIENCY_SECONDS_SUFFIX}"
        value = as_scalar(metrics.get(key))
        if value is None:
            continue
        seconds[category] = value
    return seconds


#: Which serialized key names a phase, and what a phase is called. Owned by
#: ``SetupTimingMetrics``, whose field names the rule describes.
setup_phase = SetupTimingMetrics.phase_name
map_setup_seconds = SetupTimingMetrics.phase_seconds


def _tee_setup_metrics(meter: Meter, metrics: dict[str, Any]) -> None:
    """Emit the startup phase durations as ``rl.setup.duration``."""
    for phase, value in map_setup_seconds(metrics).items():
        # No rl.bucket, unlike the efficiency series: these phases nest and
        # overlap by construction, so summing them by bucket would count
        # startup several times over. Read one phase at a time.
        _record(meter, {_SETUP_DURATION_KEY: value}, {RL_SETUP_PHASE_ATTR: phase})


def map_teed_scalars(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract ``{registry key: value}`` for every declared scalar present.

    Pure function (no OTel side effects) so it is trivially unit-testable.
    """
    values: dict[str, float] = {}
    for teed in teed_metrics():
        value = as_scalar(metrics.get(teed.logger_key))
        if value is None:
            continue
        values[teed.key] = value
    return values


def _tee_scalars(meter: Meter, metrics: dict[str, Any]) -> None:
    """Emit the declared training and vLLM scalars, unlabelled."""
    values = map_teed_scalars(metrics)
    if values:
        _record(meter, values)


def _tee_efficiency_metrics(meter: Meter, metrics: dict[str, Any]) -> None:
    """Emit the ``efficiency/*`` phase durations as ``rl.efficiency.*``."""
    measurement_by_category = efficiency_measurements()
    seconds = map_efficiency_seconds(metrics, measurement_by_category)
    pct = as_scalar(metrics.get(_EFFICIENCY_PCT_KEY))
    if not seconds and pct is None:
        return

    for category, value in seconds.items():
        measurement = measurement_by_category[category]
        attributes = {
            RL_EFFICIENCY_CATEGORY_ATTR: category,
            RL_EFFICIENCY_MEASUREMENT_ATTR: measurement,
            RL_EFFICIENCY_WINDOW_ATTR: efficiency_window(category, measurement),
        }
        bucket = bucket_for_efficiency_category(category)
        if bucket is not None:
            attributes[RL_BUCKET_ATTR] = bucket.value
        _record(meter, {_EFFICIENCY_SECONDS_KEY: value}, attributes)
    if pct is not None:
        # Tagged like the per-category points even though it is a single series:
        # a ratio needs its window stated more than a duration does, since a
        # reader cannot tell a per-step percentage from a run-to-date one by
        # looking at it. Derived rather than asserted -- print_efficiency_summary
        # falls back to a run-cumulative denominator when a caller passes no
        # per-step one -- and defaulting to the run window when the flag is
        # absent, since mislabelling a run ratio as per-step is the harmful
        # direction.
        is_per_step = as_scalar(metrics.get(_EFFICIENCY_PCT_PER_STEP_KEY))
        _record(
            meter,
            {_EFFICIENCY_PCT_KEY_REGISTRY: pct},
            {
                RL_EFFICIENCY_MEASUREMENT_ATTR: WALL_CLOCK_MEASUREMENT,
                RL_EFFICIENCY_WINDOW_ATTR: STEP_WINDOW if is_per_step else RUN_WINDOW,
            },
        )


def tee_rl_metrics_to_otel(metrics: dict[str, Any], prefix: Optional[str]) -> None:
    """Mirror the per-step metrics into OTel (no-op unless exporting).

    Three families ride the driver's per-step ``train`` dicts and are teed: the
    ``efficiency/*`` durations, the training scalars, and the ``vllm/*`` engine
    deltas. The OTel instruments are touched only when telemetry is actively
    exporting; everything else short-circuits to a no-op.

    Never raises. ``Logger.log_metrics`` calls this unguarded on every step, so
    the guarantee has to live here: each emit path below has its own handler, and
    this one covers everything around them -- reading the handle, dispatching on
    the prefix, declaring the group -- so no shape of telemetry failure can reach
    a training step.
    """
    try:
        _tee_rl_metrics_to_otel(metrics, prefix)
    except Exception:
        warn_once("tee", "failed to tee RL metrics to OTel")


def _tee_rl_metrics_to_otel(metrics: dict[str, Any], prefix: Optional[str]) -> None:
    """Body of :func:`tee_rl_metrics_to_otel`, inside its exception guard."""
    is_setup = prefix == SETUP_TIMING_PREFIX
    if not is_setup and prefix not in _TRAIN_PREFIXES:
        return
    telemetry = get_telemetry_handle()
    if telemetry is None or not telemetry.is_exporting:
        return
    if not ensure_metric_group_registered():
        return

    if is_setup:
        # Its own branch rather than a third handler below: this dict arrives
        # once, at step 0, and shares no key with the per-step families, so
        # scanning it for them (and them for it) would be dead work.
        try:
            _tee_setup_metrics(telemetry.meter, metrics)
        except Exception:
            warn_once("setup", "failed to tee setup timing metrics")
        return

    # Separate handlers so one broken family cannot silence the others: a bad
    # value in a training scalar should not cost the run its efficiency series.
    try:
        _tee_efficiency_metrics(telemetry.meter, metrics)
    except Exception:
        # Broad by intent: observability must not break a training step.
        warn_once("efficiency", "failed to tee efficiency metrics")

    try:
        _tee_scalars(telemetry.meter, metrics)
    except Exception:
        warn_once("scalars", "failed to tee training/vLLM scalars")
