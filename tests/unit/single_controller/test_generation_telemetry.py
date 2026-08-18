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

"""Engine saturation: telling a busy generation fleet from an under-subscribed one.

Both look the same from the trainer's side -- time spent in generation -- but only the
second is fixed by admitting more in-flight prompts, so this is the measurement that
decides whether a deeper lag setting can buy anything. The reporting has to survive the
case it is most needed in: a run wedged before its first step closes, where the
step-boundary reporter never runs at all.
"""

import asyncio
import threading
from types import SimpleNamespace

from nemo_rl.algorithms import single_controller as sc
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    _summarize_generation_saturation,
)
from nemo_rl.experience.rollout_timing import NemoGymRolloutTiming


class TestSummarizeGenerationSaturation:
    def test_samples_are_pooled_across_engines(self):
        summary = _summarize_generation_saturation(
            {"inflight_batch_sizes": {0: [2, 4], 1: [6, 8]}}
        )

        assert summary["requests_running_mean"] == 5.0
        assert summary["requests_running_max"] == 8.0

    def test_queue_busy_frac_counts_ticks_not_depth(self):
        """The mean hides the shape: a queue deep for one tick averages out to idle."""
        summary = _summarize_generation_saturation(
            {"num_pending_samples": {0: [0, 0, 0, 40]}}
        )

        assert summary["queue_busy_frac"] == 0.25
        assert summary["requests_waiting_max"] == 40.0

    def test_a_backend_that_collects_nothing_summarizes_to_nothing(self):
        """Every backend except vLLM with its metrics logger enabled lands here."""
        assert _summarize_generation_saturation({}) == {}
        assert _summarize_generation_saturation({"inflight_batch_sizes": {}}) == {}

    def test_a_partially_reporting_backend_still_summarizes(self):
        summary = _summarize_generation_saturation(
            {"kv_cache_usage_perc": {0: [0.25, 0.75]}}
        )

        assert summary == {"kv_cache_usage_mean": 0.5, "kv_cache_usage_max": 0.75}


def _make_controller(get_logger_metrics, rollout_timing=None):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._gen = SimpleNamespace(get_logger_metrics=get_logger_metrics)
    ctrl._rollout_manager = SimpleNamespace(
        rollout_timing=rollout_timing or NemoGymRolloutTiming()
    )
    ctrl._train_steps = 0
    return ctrl


def _gym_timing(await_seconds: float, postprocess_seconds: float):
    timing = NemoGymRolloutTiming()
    timing.add(
        {
            "timing/rollout/await_results": await_seconds,
            "timing/rollout/postprocess_results": postprocess_seconds,
        }
    )
    return timing


async def _run_ticks(ctrl, seconds: float):
    task = asyncio.ensure_future(ctrl._telemetry_report_pump())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task


class TestTelemetryReportPump:
    def test_saturation_is_reported_without_a_step_closing(self, capsys, monkeypatch):
        """train_steps stays at 0 here -- the wedge this exists for."""
        monkeypatch.setattr(sc, "_TELEMETRY_REPORT_SECONDS", 0.001)
        ctrl = _make_controller(lambda: {"num_pending_samples": {0: [12, 12]}})

        asyncio.run(_run_ticks(ctrl, 0.05))

        assert "engine saturation" in capsys.readouterr().out

    def test_a_failing_collection_is_reported_and_the_pump_survives(
        self, capsys, monkeypatch
    ):
        """Telemetry is one of the tasks run() waits on, so exiting would end the run."""
        monkeypatch.setattr(sc, "_TELEMETRY_REPORT_SECONDS", 0.001)

        def _unreachable():
            raise RuntimeError("worker is gone")

        ctrl = _make_controller(_unreachable)

        task = asyncio.run(_run_ticks(ctrl, 0.05))

        assert task.cancelled(), "the pump must still have been running to cancel"
        assert "engine saturation unavailable" in capsys.readouterr().out

    def test_a_wedged_collection_does_not_queue_another_behind_it(self, monkeypatch):
        """The failure mode this reporting exists to observe must not compound it.

        An unbounded await is how the environment health probe used to wedge the
        watchdog; issuing a fresh collection per tick instead would pile up one blocked
        thread every cadence for the rest of the run.
        """
        monkeypatch.setattr(sc, "_TELEMETRY_REPORT_SECONDS", 0.001)
        release = threading.Event()
        calls = []

        def _wedged():
            calls.append(1)
            release.wait(timeout=30)
            return {}

        ctrl = _make_controller(_wedged)
        try:
            asyncio.run(_run_ticks(ctrl, 0.05))
            assert len(calls) == 1, f"issued {len(calls)} collections behind a wedge"
        finally:
            # Let the blocked worker thread exit so it does not outlive the test.
            release.set()

    def test_rollout_phases_survive_an_unreachable_generation_fleet(
        self, capsys, monkeypatch
    ):
        """These totals are local state, so they must not ride on the engine fetch."""
        monkeypatch.setattr(sc, "_TELEMETRY_REPORT_SECONDS", 0.001)

        def _unreachable():
            raise RuntimeError("worker is gone")

        ctrl = _make_controller(
            _unreachable,
            rollout_timing=_gym_timing(await_seconds=75.0, postprocess_seconds=25.0),
        )

        asyncio.run(_run_ticks(ctrl, 0.05))

        out = capsys.readouterr().out
        assert "gym rollout phases" in out
        assert "postprocess_results_pct=25.00" in out

    def test_the_native_rollout_path_reports_no_phases(self, capsys, monkeypatch):
        """Only NeMo-Gym has an inline postprocess to account for."""
        monkeypatch.setattr(sc, "_TELEMETRY_REPORT_SECONDS", 0.001)
        ctrl = _make_controller(lambda: {"num_pending_samples": {0: [12, 12]}})

        asyncio.run(_run_ticks(ctrl, 0.05))

        out = capsys.readouterr().out
        assert "gym rollout phases" not in out
        assert "engine saturation" in out


class TestLogRolloutPostprocessShare:
    def _controller(self, rollout_timing):
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._rollout_manager = SimpleNamespace(rollout_timing=rollout_timing)
        ctrl._train_steps = 7
        ctrl._logger = SimpleNamespace(
            logged=[],
            log_metrics=lambda metrics, step, prefix: ctrl._logger.logged.append(
                (metrics, step, prefix)
            ),
        )
        return ctrl

    def test_the_share_is_logged_as_a_series(self):
        ctrl = self._controller(
            _gym_timing(await_seconds=80.0, postprocess_seconds=20.0)
        )

        ctrl._log_rollout_postprocess_share()

        metrics, step, prefix = ctrl._logger.logged[0]
        assert metrics["postprocess_results_pct"] == 20.0
        assert step == 7
        assert prefix == "rollout/nemo_gym"

    def test_nothing_is_logged_before_a_gym_group_lands(self):
        ctrl = self._controller(NemoGymRolloutTiming())

        ctrl._log_rollout_postprocess_share()

        assert ctrl._logger.logged == []
