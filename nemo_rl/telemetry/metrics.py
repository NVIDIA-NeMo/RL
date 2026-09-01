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

Every series is declared here rather than in lens, via lens's consumer-driven
metric registry: NeMo-RL owns the ``rl.*`` metric names, so a new series needs
no lens release and no negotiation over field names. Four families are teed —
the async ``efficiency/*`` phase durations, the training scalars, the ``vllm/*``
engine deltas, and the ``timing/setup`` startup phases. The first three ride the
per-step ``train`` dicts; the last arrives once, at step 0.

``Logger.log_metrics`` fans a step out as several dicts under different
prefixes, and a key is only reachable from the prefix its own dict carries — so
only the prefixes those dicts actually arrive under are teed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from nemo_rl.telemetry.instrumentation import (
    COLLECTOR_LOOP_CATEGORIES,
    RL_BUCKET_ATTR,
    RL_EFFICIENCY_CATEGORY_ATTR,
    bucket_for_efficiency_category,
)
from nemo_rl.telemetry.setup import get_telemetry_handle

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

logger = logging.getLogger(__name__)

# Prefixes carrying the per-step families. The efficiency, training-scalar and
# vLLM dicts all arrive under the driver's train prefixes, and a key is only
# reachable from the prefix its own dict is logged under, so looking anywhere
# else would be dead work per step. ``_SETUP_PREFIX`` below is the one
# non-per-step prefix teed.
_TRAIN_PREFIXES: tuple[Optional[str], ...] = ("train", "")


def _scalar(value: Any) -> Optional[float]:
    """Coerce a logged value to float, or None when it is not a usable scalar."""
    # bool is a subclass of int, so it has to be excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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

# Startup phase durations, logged once at step 0 by every algorithm's setup
# (``SetupTimingMetrics.to_metrics_dict`` in grpo.py / single_controller.py, a
# plain dict of the same shape in ppo.py).
#
# Dimensioned like the efficiency series, and for a stronger reason: alongside
# its declared fields ``SetupTimingMetrics`` carries a free-form ``extras`` dict
# (the sparse-refit transports add ``vllm_<transport>_sparse_init_time_s`` at
# runtime), so the phase set is not knowable at declaration time. A row per
# phase would silently drop whatever it had not been taught about.
_SETUP_PREFIX = "timing/setup"
RL_SETUP_DURATION_METRIC = "rl.setup.duration"
RL_SETUP_PHASE_ATTR = "rl.setup.phase"
_SETUP_DURATION_KEY = "setup_duration"
# Both suffixes are stripped so the attribute reads as the phase
# (``generation_init``), not as the key (``generation_init_time_s``). Requiring
# one of them is also the filter: it excludes non-durations that ride the same
# dict, such as the ``parallel_init_enabled`` flag.
_SETUP_TIME_SUFFIX = "_time_s"
_SETUP_SECONDS_SUFFIX = "_s"

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

# Registry keys for the two dimensioned efficiency series. Unlike the scalars
# below these carry no single logger key: one series covers every category, with
# the category in an attribute.
_EFFICIENCY_SECONDS_KEY = "efficiency_seconds"
_EFFICIENCY_PCT_KEY_REGISTRY = "efficiency_pct"


@dataclass(frozen=True)
class _TeedMetric:
    """One ``Logger`` key mirrored into OTel, and how it is declared to lens.

    Pairing the logger key with its own declaration in a single row is the point:
    the previous design kept a separate name-to-field map, and three of its
    entries pointed at keys nothing emitted, so those gauges were dead on
    arrival and reported a flat line rather than an error. Here a renamed logger
    key that is not renamed in the same row fails the drift test in
    ``tests/unit/telemetry/test_source_drift.py`` instead.

    Attributes:
        logger_key: Key as it appears in the dict handed to ``Logger.log_metrics``.
        key: Registry key recorded against; must be a Python identifier.
        name: OTel series name actually emitted.
        kind: One of lens's ``gauge`` / ``counter`` / ``histogram`` / ``up_down_counter``.
        unit: UCUM-ish unit string, or empty when dimensionless.
        description: Help text carried to the backend.
    """

    logger_key: str
    key: str
    name: str
    kind: str = "gauge"
    unit: str = ""
    description: str = ""


# Training scalars. Each logger key here was verified to be emitted by an
# algorithm or loss function; see the drift test.
_TRAIN_SCALARS: tuple[_TeedMetric, ...] = (
    _TeedMetric(
        "reward", "reward_mean", "rl.reward.mean", description="Mean rollout reward."
    ),
    # The logger key is named for the penalty, but the value has the coefficient
    # divided back out (loss_functions.py), so this really is the divergence.
    _TeedMetric(
        "kl_penalty",
        "kl_divergence",
        "rl.kl.divergence",
        description="KL divergence from the reference policy.",
    ),
    _TeedMetric(
        "loss", "policy_loss", "rl.policy.loss", description="Policy training loss."
    ),
    _TeedMetric(
        "critic/loss",
        "value_loss",
        "rl.value.loss",
        description="Value/critic training loss (PPO).",
    ),
    _TeedMetric(
        "approx_entropy",
        "entropy",
        "rl.entropy",
        description="Approximate policy entropy.",
    ),
    _TeedMetric(
        "mean_gen_tokens_per_sample",
        "response_length_mean",
        "rl.response.length.mean",
        unit="{token}",
        description="Mean generated tokens per sample.",
    ),
    _TeedMetric("grad_norm", "grad_norm", "rl.grad_norm", description="Gradient norm."),
    _TeedMetric(
        "lr",
        "learning_rate",
        "rl.learning_rate",
        description="Optimizer learning rate.",
    ),
)

# vLLM engine metrics, read from the engine's own Prometheus registry and
# delta'd per step (see nemo_rl/models/generation/vllm/utils.py). Distinct from
# the driver-side ``gen_ai.*`` series, which are derived from the tensors a
# generate() call returns: these are the engine's own accounting, and the
# sequence-length and abort counts cannot be reconstructed driver-side at all.
#
# Counters take the per-step delta, so adding deltas reconstructs the run total.
_VLLM_STEP_METRICS: tuple[_TeedMetric, ...] = (
    _TeedMetric(
        "vllm/prompt_tokens",
        "vllm_prompt_tokens",
        "rl.vllm.prompt_tokens",
        kind="counter",
        unit="{token}",
        description="Prompt tokens processed by the vLLM engine.",
    ),
    _TeedMetric(
        "vllm/generation_tokens",
        "vllm_generation_tokens",
        "rl.vllm.generation_tokens",
        kind="counter",
        unit="{token}",
        description="Tokens generated by the vLLM engine.",
    ),
    _TeedMetric(
        "vllm/prompt_length_mean",
        "vllm_prompt_length_mean",
        "rl.vllm.prompt_length.mean",
        unit="{token}",
        description="Mean prompt sequence length over the step.",
    ),
    _TeedMetric(
        "vllm/generation_length_mean",
        "vllm_generation_length_mean",
        "rl.vllm.generation_length.mean",
        unit="{token}",
        description="Mean generated sequence length over the step.",
    ),
    _TeedMetric(
        "vllm/generations_ok",
        "vllm_generations_ok",
        "rl.vllm.generations.ok",
        kind="counter",
        unit="{generation}",
        description="Generations the engine finished normally.",
    ),
    _TeedMetric(
        "vllm/generations_failed",
        "vllm_generations_failed",
        "rl.vllm.generations.failed",
        kind="counter",
        unit="{generation}",
        description="Generations the engine aborted rather than finished.",
    ),
)

_TEED_SCALARS: tuple[_TeedMetric, ...] = _TRAIN_SCALARS + _VLLM_STEP_METRICS

# Registration is process-global in lens, so it has to happen once per process
# and in every process that records -- driver and workers alike. Doing it lazily
# on the first tee rather than at import keeps this module importable when the
# installed lens predates the registry, which matters because
# nemo_rl.utils.logger imports the tee at module scope: an import-time failure
# here would take down training, not just telemetry.
_REGISTERED = False

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


def _metric_specs() -> list[Any]:
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
            teed.key,
            teed.name,
            teed.kind,
            unit=teed.unit,
            description=teed.description,
        )
        for teed in _TEED_SCALARS
    )
    return specs


def ensure_metric_group_registered() -> bool:
    """Declare the ``rl`` metric group with lens, once per process.

    Returns:
        Whether the group is available to record against. ``False`` means the
        installed lens has no metric registry, which makes every tee below a
        no-op rather than an error.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from nemo.lens.instruments import register_metric_group

        register_metric_group(RL_METRIC_GROUP, _metric_specs())
    except Exception:
        warn_once(
            "registry",
            "could not declare the 'rl' metric group; RL metrics will not be "
            "exported (a lens build with the metric registry is required)",
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


def setup_phase(key: str) -> Optional[str]:
    """Phase name for a ``timing/setup`` key, or None when it is not a duration.

    Pure function (no OTel side effects) so it is trivially unit-testable.
    """
    for suffix in (_SETUP_TIME_SUFFIX, _SETUP_SECONDS_SUFFIX):
        if key.endswith(suffix):
            phase = key[: -len(suffix)]
            # Guards against a bare "_s" or "_time_s" key naming nothing.
            return phase or None
    return None


def map_setup_seconds(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract ``{phase: seconds}`` from a raw ``timing/setup`` metrics dict.

    Pure function (no OTel side effects) so it is trivially unit-testable.
    """
    seconds: dict[str, float] = {}
    for key, raw in metrics.items():
        phase = setup_phase(key)
        if phase is None:
            continue
        value = _scalar(raw)
        if value is None:
            continue
        seconds[phase] = value
    return seconds


def _tee_setup_metrics(meter: Meter, metrics: dict[str, Any]) -> None:
    """Emit the startup phase durations as ``rl.setup.duration``."""
    for phase, value in map_setup_seconds(metrics).items():
        # No rl.bucket, unlike the efficiency series. These phases overlap each
        # other by construction -- total_setup contains the rest, parallel_wall
        # covers the generation and policy builds running concurrently, and
        # generation_init_{reserve,load} are parts of generation_init -- so
        # anything summing them by bucket would count startup several times
        # over. Read one phase at a time; the rl.setup.* spans carry the bucket
        # for the flat subset where summing is well defined.
        _record(meter, {_SETUP_DURATION_KEY: value}, {RL_SETUP_PHASE_ATTR: phase})


def map_teed_scalars(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract ``{registry key: value}`` for every declared scalar present.

    Pure function (no OTel side effects) so it is trivially unit-testable.
    """
    values: dict[str, float] = {}
    for teed in _TEED_SCALARS:
        value = _scalar(metrics.get(teed.logger_key))
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
    pct = _scalar(metrics.get(_EFFICIENCY_PCT_KEY))
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
        is_per_step = _scalar(metrics.get(_EFFICIENCY_PCT_PER_STEP_KEY))
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
    is_setup = prefix == _SETUP_PREFIX
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
