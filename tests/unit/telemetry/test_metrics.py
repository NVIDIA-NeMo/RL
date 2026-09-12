# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the async efficiency metrics -> OTel tee."""

import logging

import pytest

from nemo_rl.telemetry.metrics import map_efficiency_seconds


def test_tee_noop_when_not_exporting():
    # No telemetry handle set -> must be a silent no-op (no exception).
    from nemo_rl.telemetry.metrics import tee_rl_metrics_to_otel

    tee_rl_metrics_to_otel({"efficiency/idle/refit_bubble_s": 12.0}, "train")


_MEASUREMENTS = {
    "idle/refit_bubble": "wall_clock",
    "idle/buffer_full_backoff": "thread_seconds",
}


def test_map_efficiency_seconds_reads_per_category_keys():
    seconds = map_efficiency_seconds(
        {
            "efficiency/idle/refit_bubble_s": 12.0,
            "efficiency/idle/buffer_full_backoff_s": 80,
        },
        _MEASUREMENTS,
    )
    assert seconds == {"idle/refit_bubble": 12.0, "idle/buffer_full_backoff": 80.0}


def test_map_efficiency_seconds_ignores_aggregate_keys():
    # These share the efficiency/ prefix and _s suffix but are not categories,
    # so a prefix/suffix parse would invent bogus category series from them.
    seconds = map_efficiency_seconds(
        {
            "efficiency/total_waste_s": 30.0,
            "efficiency/productive_time_s": 70.0,
            "efficiency/total_wall_time_s": 100.0,
            "efficiency/thread_seconds_total_s": 400.0,
        },
        _MEASUREMENTS,
    )
    assert seconds == {}


def test_map_efficiency_seconds_skips_bool_and_non_numeric():
    seconds = map_efficiency_seconds(
        {
            "efficiency/idle/refit_bubble_s": True,
            "efficiency/idle/buffer_full_backoff_s": "nan",
        },
        _MEASUREMENTS,
    )
    assert seconds == {}


def test_efficiency_measurements_classifies_every_category():
    # Drift guard: a category added to algorithms/utils.py without a
    # classification would silently drop out of the OTel series.
    pytest.importorskip("nemo_rl.algorithms.utils")
    from nemo_rl.algorithms.utils import EFFICIENCY_CATEGORIES
    from nemo_rl.telemetry.metrics import (
        COLLECTOR_WALL_CLOCK_MEASUREMENT,
        THREAD_SECONDS_MEASUREMENT,
        WALL_CLOCK_MEASUREMENT,
        efficiency_measurements,
    )

    measurements = efficiency_measurements()
    assert set(measurements) == set(EFFICIENCY_CATEGORIES)
    assert set(measurements.values()) <= {
        WALL_CLOCK_MEASUREMENT,
        COLLECTOR_WALL_CLOCK_MEASUREMENT,
        THREAD_SECONDS_MEASUREMENT,
    }


def test_collector_loop_waits_are_not_labelled_thread_seconds():
    """The two loop waits are sequential, so they are real durations.

    They sit in ``THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES`` because the W&B
    summary only splits driver from collector, but labelling them
    ``thread_seconds`` would tell a consumer they can exceed wall time -- which
    is untrue of a single-threaded ``Event.wait()``.
    """
    pytest.importorskip("nemo_rl.algorithms.utils")
    from nemo_rl.telemetry.metrics import (
        COLLECTOR_WALL_CLOCK_MEASUREMENT,
        THREAD_SECONDS_MEASUREMENT,
        efficiency_measurements,
    )

    measurements = efficiency_measurements()
    assert measurements["idle/refit_event_wait"] == COLLECTOR_WALL_CLOCK_MEASUREMENT
    assert (
        measurements["idle/generation_limit_pause"] == COLLECTOR_WALL_CLOCK_MEASUREMENT
    )
    # The batch-worker categories keep the label that warns about summing.
    assert measurements["idle/buffer_full_backoff"] == THREAD_SECONDS_MEASUREMENT
    assert measurements["wasted/failed_trajectory"] == THREAD_SECONDS_MEASUREMENT


def _gauge_points(data, name):
    """Data points for gauge *name* in one already-collected metrics batch.

    ``InMemoryMetricReader.get_metrics_data()`` drains what it collects, so
    callers must collect once and filter the result rather than calling the
    reader per metric name.
    """
    if data is None:
        return []
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def _start_exporting_telemetry():
    """Install an exporting telemetry handle; returns its metric reader."""
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    import nemo_rl.telemetry.setup as setup_mod
    from nemo_rl.telemetry.span_groups import RLSpanGroup

    reader = InMemoryMetricReader()
    cfg = NemoLensConfig(enabled=True, _span_group_cls=RLSpanGroup)
    setup_mod._TELEMETRY_HANDLE = setup_telemetry(
        cfg, rank=0, world_size=1, metric_reader=reader
    )
    return reader


def test_tee_emits_efficiency_seconds_tagged_by_measurement(monkeypatch):
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.instrumentation import (
        RL_BUCKET_ATTR,
        RL_EFFICIENCY_CATEGORY_ATTR,
    )
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_MEASUREMENT_ATTR,
        RL_EFFICIENCY_SECONDS_METRIC,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    # Pin the classification so this test does not move when the canonical
    # category lists change.
    monkeypatch.setattr(metrics_mod, "efficiency_measurements", lambda: _MEASUREMENTS)

    tee_rl_metrics_to_otel(
        {
            "efficiency/idle/refit_bubble_s": 12.0,
            "efficiency/idle/buffer_full_backoff_s": 80.0,
            "efficiency/total_waste_s": 999.0,
        },
        "",
    )

    points = _gauge_points(reader.get_metrics_data(), RL_EFFICIENCY_SECONDS_METRIC)
    by_category = {
        point.attributes[RL_EFFICIENCY_CATEGORY_ATTR]: point for point in points
    }
    assert set(by_category) == set(_MEASUREMENTS)

    wall = by_category["idle/refit_bubble"]
    assert wall.value == 12.0
    assert wall.attributes[RL_EFFICIENCY_MEASUREMENT_ATTR] == "wall_clock"
    assert wall.attributes[RL_BUCKET_ATTR] == "idle"

    thread = by_category["idle/buffer_full_backoff"]
    assert thread.value == 80.0
    assert thread.attributes[RL_EFFICIENCY_MEASUREMENT_ATTR] == "thread_seconds"
    # Same bucket as the wall-clock point, which is exactly why the measurement
    # attribute has to be present for a bucket rollup to stay honest.
    assert thread.attributes[RL_BUCKET_ATTR] == "idle"


def test_efficiency_window_separates_per_step_from_cumulative():
    """Summing over time needs its own attribute, not the measurement kind.

    The driver resets its ``Timer`` each step and the collector never does, and
    ``init/total`` is a run constant republished every step -- so a dashboard
    summing ``wall_clock`` across steps would count startup once per step.
    """
    from nemo_rl.telemetry.metrics import (
        COLLECTOR_WALL_CLOCK_MEASUREMENT,
        RUN_WINDOW,
        STEP_WINDOW,
        THREAD_SECONDS_MEASUREMENT,
        WALL_CLOCK_MEASUREMENT,
        efficiency_window,
    )

    assert efficiency_window("idle/refit_bubble", WALL_CLOCK_MEASUREMENT) == STEP_WINDOW
    assert efficiency_window("init/total", WALL_CLOCK_MEASUREMENT) == RUN_WINDOW
    assert (
        efficiency_window("idle/refit_event_wait", COLLECTOR_WALL_CLOCK_MEASUREMENT)
        == RUN_WINDOW
    )
    assert (
        efficiency_window("wasted/failed_trajectory", THREAD_SECONDS_MEASUREMENT)
        == RUN_WINDOW
    )


def test_run_window_categories_match_the_efficiency_summary():
    """The window split is restated here; the summary excludes the same set.

    ``print_efficiency_summary`` keeps run-window categories out of its per-step
    waste ratio using its own copy of this set. If the two drifted, a category
    could be tagged ``step`` here while being excluded from the step ratio
    there, or charged to every step while advertised as a run constant.
    """
    from nemo_rl.telemetry.metrics import _RUN_WINDOW_WALL_CLOCK_CATEGORIES
    from tests.unit.telemetry.conftest import algorithms_utils_categories

    canonical = algorithms_utils_categories("RUN_WINDOW_WALL_CLOCK_CATEGORIES")

    assert (
        set(_RUN_WINDOW_WALL_CLOCK_CATEGORIES)
        == canonical["RUN_WINDOW_WALL_CLOCK_CATEGORIES"]
    )


def test_tee_tags_the_pct_as_a_per_step_wall_clock_ratio(monkeypatch):
    """The one aggregate point needs its window stated most.

    A percentage carries no unit to hint at what it covers, so an untagged
    point invites a reader to treat a per-step ratio as run-to-date.
    """
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_MEASUREMENT_ATTR,
        RL_EFFICIENCY_PCT_METRIC,
        RL_EFFICIENCY_WINDOW_ATTR,
        STEP_WINDOW,
        WALL_CLOCK_MEASUREMENT,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    monkeypatch.setattr(metrics_mod, "efficiency_measurements", lambda: _MEASUREMENTS)

    tee_rl_metrics_to_otel(
        {
            "efficiency/efficiency_pct": 91.5,
            "efficiency/efficiency_pct_is_per_step": 1.0,
        },
        "",
    )

    (point,) = _gauge_points(reader.get_metrics_data(), RL_EFFICIENCY_PCT_METRIC)
    assert point.value == 91.5
    assert point.attributes[RL_EFFICIENCY_WINDOW_ATTR] == STEP_WINDOW
    assert point.attributes[RL_EFFICIENCY_MEASUREMENT_ATTR] == WALL_CLOCK_MEASUREMENT


def test_a_run_cumulative_pct_is_not_published_as_per_step(monkeypatch):
    """print_efficiency_summary falls back to a run-to-date denominator.

    Callers that pass no per-step wall time get a ratio over the whole run, so
    the window has to be derived from what the summary did rather than asserted
    here -- and an absent flag has to mean ``run``, since mislabelling a
    cumulative ratio as per-step is the direction that misleads.
    """
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_PCT_METRIC,
        RL_EFFICIENCY_WINDOW_ATTR,
        RUN_WINDOW,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    monkeypatch.setattr(metrics_mod, "efficiency_measurements", lambda: _MEASUREMENTS)

    tee_rl_metrics_to_otel(
        {
            "efficiency/efficiency_pct": 91.5,
            "efficiency/efficiency_pct_is_per_step": 0.0,
        },
        "",
    )
    (point,) = _gauge_points(reader.get_metrics_data(), RL_EFFICIENCY_PCT_METRIC)
    assert point.attributes[RL_EFFICIENCY_WINDOW_ATTR] == RUN_WINDOW


def test_a_pct_with_no_window_flag_defaults_to_the_run_window(monkeypatch):
    """An absent flag must not be read as per-step.

    Any caller predating the flag, or one that builds the dict by hand, would
    otherwise have its run-to-date ratio published as though it covered a
    single step.
    """
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_PCT_METRIC,
        RL_EFFICIENCY_WINDOW_ATTR,
        RUN_WINDOW,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    monkeypatch.setattr(metrics_mod, "efficiency_measurements", lambda: _MEASUREMENTS)

    tee_rl_metrics_to_otel({"efficiency/efficiency_pct": 91.5}, "")

    (point,) = _gauge_points(reader.get_metrics_data(), RL_EFFICIENCY_PCT_METRIC)
    assert point.attributes[RL_EFFICIENCY_WINDOW_ATTR] == RUN_WINDOW


def test_tee_tags_efficiency_seconds_with_the_window(monkeypatch):
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.instrumentation import RL_EFFICIENCY_CATEGORY_ATTR
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_SECONDS_METRIC,
        RL_EFFICIENCY_WINDOW_ATTR,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    monkeypatch.setattr(
        metrics_mod,
        "efficiency_measurements",
        lambda: {**_MEASUREMENTS, "init/total": "wall_clock"},
    )

    tee_rl_metrics_to_otel(
        {
            "efficiency/idle/refit_bubble_s": 12.0,
            "efficiency/init/total_s": 300.0,
        },
        "",
    )

    points = _gauge_points(reader.get_metrics_data(), RL_EFFICIENCY_SECONDS_METRIC)
    windows = {
        point.attributes[RL_EFFICIENCY_CATEGORY_ATTR]: point.attributes[
            RL_EFFICIENCY_WINDOW_ATTR
        ]
        for point in points
    }
    assert windows["idle/refit_bubble"] == "step"
    assert windows["init/total"] == "run"


def test_tee_emits_efficiency_pct(monkeypatch):
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_PCT_METRIC,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    monkeypatch.setattr(metrics_mod, "efficiency_measurements", lambda: _MEASUREMENTS)

    tee_rl_metrics_to_otel({"efficiency/efficiency_pct": 87.5}, "")

    points = _gauge_points(reader.get_metrics_data(), RL_EFFICIENCY_PCT_METRIC)
    assert [point.value for point in points] == [87.5]


def test_tee_skips_prefixes_other_than_train():
    # The efficiency dict is logged under "train"/"" -- looking for it under the
    # other prefixes log_metrics fans a step out to would be dead work on every
    # step, and admitting one would attribute another dict's values to a
    # category series.
    pytest.importorskip("nemo.lens")
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_SECONDS_METRIC,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    tee_rl_metrics_to_otel({"efficiency/idle/refit_bubble_s": 12.0}, "performance")
    tee_rl_metrics_to_otel({"efficiency/idle/refit_bubble_s": 12.0}, "validation")

    assert _gauge_points(reader.get_metrics_data(), RL_EFFICIENCY_SECONDS_METRIC) == []


def test_tee_skips_efficiency_instruments_without_efficiency_keys(monkeypatch):
    # A plain train-metrics dict must not create the efficiency series at all.
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.metrics import (
        RL_EFFICIENCY_PCT_METRIC,
        RL_EFFICIENCY_SECONDS_METRIC,
        tee_rl_metrics_to_otel,
    )

    reader = _start_exporting_telemetry()
    monkeypatch.setattr(metrics_mod, "efficiency_measurements", lambda: _MEASUREMENTS)

    tee_rl_metrics_to_otel({"reward": 0.5}, "train")

    data = reader.get_metrics_data()
    assert _gauge_points(data, RL_EFFICIENCY_SECONDS_METRIC) == []
    assert _gauge_points(data, RL_EFFICIENCY_PCT_METRIC) == []


def test_tee_never_raises_into_the_training_step(monkeypatch, caplog):
    """Logger.log_metrics calls this unguarded, so it has to swallow everything.

    Exercised through a failure *outside* the inner handler -- reading the
    telemetry handle -- since that one is already covered separately.
    """
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.metrics import tee_rl_metrics_to_otel

    def _boom():
        raise RuntimeError("handle is wedged")

    monkeypatch.setattr(metrics_mod, "get_telemetry_handle", _boom)
    monkeypatch.setattr(metrics_mod, "_WARNED", set())

    with caplog.at_level(logging.WARNING, logger=metrics_mod.__name__):
        tee_rl_metrics_to_otel({"reward": 1.0}, "train")

    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_a_broken_tee_warns_once_not_once_per_step(monkeypatch, caplog):
    """A deterministic failure repeats every step, so it must not log every step."""
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.metrics as metrics_mod
    from nemo_rl.telemetry.metrics import tee_rl_metrics_to_otel

    _start_exporting_telemetry()
    monkeypatch.setattr(metrics_mod, "efficiency_measurements", lambda: _MEASUREMENTS)
    monkeypatch.setattr(metrics_mod, "_WARNED", set())

    def _boom(*args, **kwargs):
        raise RuntimeError("instrument is gone")

    monkeypatch.setattr(metrics_mod, "map_efficiency_seconds", _boom)

    with caplog.at_level(logging.DEBUG, logger=metrics_mod.__name__):
        for _ in range(3):
            tee_rl_metrics_to_otel({"efficiency/idle/refit_bubble_s": 1.0}, "train")

    records = [r for r in caplog.records if "efficiency" in r.message]
    assert [r.levelno for r in records] == [
        logging.WARNING,
        logging.DEBUG,
        logging.DEBUG,
    ]
    # The traceback survives the demotion, so the repetition stays diagnosable.
    assert all(r.exc_info is not None for r in records)
