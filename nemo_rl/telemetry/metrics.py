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

"""Best-effort mirroring of NeMo-RL scalar metrics into OTel.

Kept out of ``nemo_rl.utils.logger`` (which pulls in torch/ray/wandb/etc.) so
the mapping stays importable and testable without the heavy training stack.
``nemo_rl.utils.logger.Logger.log_metrics`` calls :func:`tee_rl_metrics_to_otel`
after its normal fan-out to the file/wandb/mlflow backends.

Two families are teed. Standard training scalars (reward, loss, …) go through
nemo-lens's ``record_rl_metrics``, whose instruments live in lens. The async
``efficiency/*`` phase durations are emitted from instruments owned here,
because they are keyed by NeMo-RL's own efficiency-category labels and lens has
no fixed field for them.

``Logger.log_metrics`` fans a step out as several dicts under different
prefixes, and a key is only reachable from the prefix its own dict carries — so
the field maps here are per-prefix, and adding a candidate key means checking
where that dict is actually logged.
"""

from __future__ import annotations

import inspect
import logging
import weakref
from typing import TYPE_CHECKING, Any, Mapping, Optional

from nemo_rl.telemetry.instrumentation import (
    COLLECTOR_LOOP_CATEGORIES,
    RL_BUCKET_ATTR,
    RL_EFFICIENCY_CATEGORY_ATTR,
    bucket_for_efficiency_category,
)
from nemo_rl.telemetry.setup import get_telemetry

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

logger = logging.getLogger(__name__)

# Logger prefixes this module tees, and the field map to use for each. A metric
# is only reachable from the prefix its own dict is logged under, so a candidate
# key listed against the wrong prefix can never match — see the throughput note
# on _PERFORMANCE_OTEL_METRIC_MAP.
_TRAIN_PREFIXES: tuple[Optional[str], ...] = ("train", "")
_PERFORMANCE_PREFIX = "performance"

# Map raw Logger metric keys (under the "train"/"" prefix) to
# ``record_rl_metrics`` gauge fields. The first present candidate key wins.
# Best-effort: unmatched keys and non-scalar values are silently skipped.
# Fields not yet accepted by the installed nemo-lens ``record_rl_metrics``
# are dropped at tee time so one unknown kwarg cannot abort the whole batch.
_RL_OTEL_METRIC_MAP: dict[str, tuple[str, ...]] = {
    "reward_mean": ("reward", "reward_mean", "mean_reward"),
    "kl_divergence": ("kl", "kl_divergence", "mean_kl"),
    "policy_loss": ("loss", "policy_loss"),
    "value_loss": ("value_loss", "critic_loss"),
    "entropy": ("entropy",),
    "response_length_mean": ("mean_gen_tokens_per_sample", "response_length_mean"),
    "grad_norm": ("grad_norm",),
    "learning_rate": ("lr", "learning_rate"),
}

# Same shape, for the dict ``print_performance_metrics`` returns. Throughput is
# logged under its own "performance" prefix rather than with the train scalars,
# so it needs its own entry here — listing it above would never match. Only the
# job-wide ``tokens_per_sec`` is a candidate: the per-GPU rates alongside it
# would land in the same gauge at 1/N of the value, and rl.tokens_per_sec is
# documented as a job-wide rate.
_PERFORMANCE_OTEL_METRIC_MAP: dict[str, tuple[str, ...]] = {
    "tokens_per_sec": ("tokens_per_sec",),
}


def _scalar(value: Any) -> Optional[float]:
    """Coerce a logged value to float, or None when it is not a usable scalar."""
    # bool is a subclass of int, so it has to be excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def map_rl_metrics(
    metrics: dict[str, Any],
    field_map: Optional[Mapping[str, tuple[str, ...]]] = None,
) -> dict[str, float]:
    """Extract the ``record_rl_metrics`` kwargs present in a raw metrics dict.

    Pure function (no OTel side effects) so it is trivially unit-testable.

    Args:
        metrics: Raw Logger metrics dict for one step.
        field_map: ``{lens field: candidate keys}``. Defaults to the train-scalar
            map; pass :data:`_PERFORMANCE_OTEL_METRIC_MAP` for the performance
            dict.

    Returns:
        ``{lens field: value}`` for every field with a usable scalar present.
    """
    if field_map is None:
        field_map = _RL_OTEL_METRIC_MAP
    kwargs: dict[str, float] = {}
    for field, candidates in field_map.items():
        for key in candidates:
            value = _scalar(metrics.get(key))
            if value is None:
                continue
            kwargs[field] = value
            break
    return kwargs


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
# Restated rather than imported: nemo_rl.algorithms.utils owns this split (it
# excludes these from its per-step efficiency ratio) but pulls in torch, and
# this module is deliberately importable without the training stack. A test
# keeps the two copies in lockstep.
_RUN_WINDOW_WALL_CLOCK_CATEGORIES: frozenset[str] = frozenset({"init/total"})

# Keys that ``print_efficiency_summary`` puts in the Logger dict.
_EFFICIENCY_KEY_PREFIX = "efficiency/"
_EFFICIENCY_SECONDS_SUFFIX = "_s"
_EFFICIENCY_PCT_KEY = "efficiency/efficiency_pct"

_EFFICIENCY_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Warn with a traceback the first time *key* fails, then stay quiet.

    Both tee failure modes are per-step and deterministic -- a broken instrument
    or a lens signature mismatch fails identically on every step -- so warning
    each time would put thousands of identical tracebacks in a run's log while
    telling the reader nothing the first one did not. Warning level once, so a
    permanently dead tee is visible at default verbosity; debug afterwards, so
    the repetition is still recoverable when someone is looking for it.
    """
    if key in _WARNED:
        logger.debug(message, exc_info=True)
        return
    _WARNED.add(key)
    logger.warning(message, exc_info=True)


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
            RUN_WINDOW if category in _RUN_WINDOW_WALL_CLOCK_CATEGORIES else STEP_WINDOW
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
        value = _scalar(metrics.get(key))
        if value is None:
            continue
        seconds[category] = value
    return seconds


def _get_efficiency_instruments(meter: Meter) -> dict[str, Any]:
    """Create (once per Meter) the efficiency gauges."""
    instruments = _EFFICIENCY_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "seconds": meter.create_gauge(
                name=RL_EFFICIENCY_SECONDS_METRIC,
                unit="s",
                description="Time attributed to one async efficiency category.",
            ),
            "pct": meter.create_gauge(
                name=RL_EFFICIENCY_PCT_METRIC,
                unit="%",
                description=("Productive share of one step's driver-side wall clock."),
            ),
        }
        _EFFICIENCY_INSTRUMENTS[meter] = instruments
    return instruments


def _tee_efficiency_metrics(meter: Meter, metrics: dict[str, Any]) -> None:
    """Emit the ``efficiency/*`` phase durations as ``rl.efficiency.*``."""
    measurement_by_category = efficiency_measurements()
    seconds = map_efficiency_seconds(metrics, measurement_by_category)
    pct = _scalar(metrics.get(_EFFICIENCY_PCT_KEY))
    if not seconds and pct is None:
        return

    instruments = _get_efficiency_instruments(meter)
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
        instruments["seconds"].set(value, attributes=attributes)
    if pct is not None:
        # Tagged like the per-category points even though it is a single series:
        # a ratio needs its window stated more than a duration does, since a
        # reader cannot tell a per-step percentage from a run-to-date one by
        # looking at it.
        instruments["pct"].set(
            pct,
            attributes={
                RL_EFFICIENCY_MEASUREMENT_ATTR: WALL_CLOCK_MEASUREMENT,
                RL_EFFICIENCY_WINDOW_ATTR: STEP_WINDOW,
            },
        )


def tee_rl_metrics_to_otel(metrics: dict[str, Any], prefix: Optional[str]) -> None:
    """Mirror standard RL scalar metrics into OTel (no-op unless exporting).

    Only the driver's per-step ``train`` scalars, the ``efficiency/*`` durations
    logged alongside them, and the ``performance`` throughput dict are teed. The
    OTel instruments are touched only when telemetry is actively exporting;
    everything else short-circuits to a no-op. The scalars additionally require
    nemo-lens to be installed, while the ``efficiency/*`` durations are emitted
    from instruments owned here and so need only the meter.

    Never raises. ``Logger.log_metrics`` calls this unguarded on every step, so
    the guarantee has to live here: the two emit paths below have their own
    handlers, and this one covers everything around them -- reading the handle,
    importing lens, inspecting its signature -- so no shape of telemetry failure
    can reach a training step.
    """
    try:
        _tee_rl_metrics_to_otel(metrics, prefix)
    except Exception:
        _warn_once("tee", "failed to tee RL metrics to OTel")


def _tee_rl_metrics_to_otel(metrics: dict[str, Any], prefix: Optional[str]) -> None:
    """Body of :func:`tee_rl_metrics_to_otel`, inside its exception guard."""
    is_train = prefix in _TRAIN_PREFIXES
    if is_train:
        field_map = _RL_OTEL_METRIC_MAP
    elif prefix == _PERFORMANCE_PREFIX:
        field_map = _PERFORMANCE_OTEL_METRIC_MAP
    else:
        return
    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return

    # Before the lens tee below, which returns early on an empty kwargs match:
    # the efficiency dict shares no keys with _RL_OTEL_METRIC_MAP, so it would
    # never reach this point otherwise.
    if is_train:
        try:
            _tee_efficiency_metrics(telemetry.meter, metrics)
        except Exception:
            # Broad by intent: observability must not break a training step.
            _warn_once("efficiency", "failed to tee efficiency metrics")

    try:
        from nemo.lens.instruments.rl import record_rl_metrics
    except ImportError:
        return

    kwargs = map_rl_metrics(metrics, field_map)
    if not kwargs:
        return
    # Drop fields the installed lens build does not accept yet (e.g. grad_norm
    # before the matching instruments land). Passing them as **kwargs raises
    # TypeError and — if unfiltered — would skip the whole tee batch.
    accepted = set(inspect.signature(record_rl_metrics).parameters) - {"meter"}
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    if not kwargs:
        return
    try:
        record_rl_metrics(telemetry.meter, **kwargs)
    except Exception:
        # See the efficiency handler above on breadth.
        _warn_once("record_rl_metrics", "nemo-lens: failed to tee RL metrics")
