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

"""Offline replay of recorded rollout durations through the elastic scheduler."""

import json
import random

import pytest

from tools.elastic_scheduler_replay import (
    GROUP_SPAN_NAME,
    ReplayConfig,
    load_group_durations,
    main,
    replay_all,
    replay_colocate,
    replay_elastic,
    replay_fixed_split,
)


def _otlp_span(name: str, start_s: float, end_s: float) -> dict:
    return {
        "traceId": "0" * 32,
        "spanId": "0" * 16,
        "name": name,
        "startTimeUnixNano": str(int(start_s * 1e9)),
        "endTimeUnixNano": str(int(end_s * 1e9)),
        "attributes": [{"key": "rl.rollout.attempt", "value": {"intValue": "0"}}],
    }


def _otlp_line(*spans: dict, legacy: bool = False) -> str:
    scope_key = "instrumentationLibrarySpans" if legacy else "scopeSpans"
    return json.dumps(
        {"resourceSpans": [{"resource": {}, scope_key: [{"spans": list(spans)}]}]}
    )


def _console_span(name: str, start: str, end: str) -> str:
    return json.dumps(
        {
            "name": name,
            "context": {"trace_id": "0x0", "span_id": "0x0"},
            "start_time": start,
            "end_time": end,
            "attributes": {"rl.rollout.attempt": 0},
        },
        indent=4,
    )


def _config(**overrides) -> ReplayConfig:
    values = {
        "num_cells": 2,
        "cell_capacity": 8,
        "group_size": 4,
        "batch_groups": 4,
        "train_step_s": 100.0,
        "num_steps": 3,
        "to_training_s": 2.0,
        "to_rollout_s": 1.0,
    }
    values.update(overrides)
    return ReplayConfig(**values)


def _heavy_tail(count: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [min(60.0 * rng.paretovariate(1.2), 1800.0) for _ in range(count)]


class TestLoadGroupDurations:
    def test_reads_otlp_json_lines_in_start_order(self, tmp_path):
        trace = tmp_path / "spans.jsonl"
        trace.write_text(
            "\n".join(
                [
                    _otlp_line(
                        _otlp_span(GROUP_SPAN_NAME, 10.0, 15.5),
                        _otlp_span("rl.sc.step", 0.0, 100.0),
                    ),
                    _otlp_line(_otlp_span(GROUP_SPAN_NAME, 2.0, 32.0), legacy=True),
                ]
            )
        )
        assert load_group_durations(trace) == pytest.approx([30.0, 5.5])

    def test_reads_console_exporter_output_between_log_lines(self, tmp_path):
        trace = tmp_path / "stdout.log"
        trace.write_text(
            "\n".join(
                [
                    "step 1 starting {not json",
                    _console_span(
                        GROUP_SPAN_NAME,
                        "2026-11-06T10:00:05.000000Z",
                        "2026-11-06T10:01:05.500000Z",
                    ),
                    "  target_step=3: substituting a spare prompt",
                    _console_span(
                        "rl.sc.policy_training",
                        "2026-11-06T10:00:00.000000Z",
                        "2026-11-06T10:10:00.000000Z",
                    ),
                    _console_span(
                        GROUP_SPAN_NAME,
                        "2026-11-06T10:00:00.000000Z",
                        "2026-11-06T10:00:10.000000Z",
                    ),
                ]
            )
        )
        assert load_group_durations(trace) == pytest.approx([10.0, 60.5])

    def test_rejects_a_trace_without_group_spans(self, tmp_path):
        trace = tmp_path / "spans.jsonl"
        trace.write_text(_otlp_line(_otlp_span("rl.sc.step", 0.0, 1.0)))
        with pytest.raises(ValueError, match="per_prompt"):
            load_group_durations(trace)


class TestReplayConfig:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"num_cells": 0},
            {"cell_capacity": 3},  # Smaller than one group.
            {"batch_groups": 0},
            {"num_steps": 0},
            {"train_step_s": -1.0},
            {"fixed_split_train_cells": 0},
        ],
    )
    def test_rejects_out_of_range_parameters(self, overrides):
        with pytest.raises(ValueError):
            _config(**overrides)

    def test_empty_trace_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            replay_colocate([], _config())


class TestBaselines:
    def test_colocate_waits_for_the_batch_then_trains_on_every_cell(self):
        # Capacity 2 cells * 8 / 4 = 4 groups, exactly one batch, all launched at once.
        result = replay_colocate([50.0], _config())
        step = 50.0 + 2.0 + 100.0 / 2
        assert result.step_end_s == pytest.approx(
            (step, 2 * step + 1.0, 3 * step + 2.0)
        )
        # The batch fills every slot and training uses every cell, so only switching
        # costs cell-time.
        assert result.idle_cell_s == pytest.approx(0.0)
        assert result.transition_cell_s == pytest.approx(2 * (3 * 2.0 + 2 * 1.0))

    def test_fixed_split_overlaps_the_next_rollout_with_training(self):
        config = _config(
            batch_groups=2, fixed_split_train_cells=1, refit_s=5.0, num_steps=2
        )
        result = replay_fixed_split([50.0], config)
        # Step 1 waits for its rollouts; step 2's were dispatched one step ahead and
        # finished during step 1's training, so step 2 starts right after the refit.
        assert result.train_cells == 1
        assert result.step_end_s == pytest.approx((150.0, 255.0))

    def test_fixed_split_without_lookahead_is_serial(self):
        config = _config(
            batch_groups=2,
            fixed_split_train_cells=1,
            refit_s=5.0,
            num_steps=2,
            max_lookahead_steps=0,
        )
        result = replay_fixed_split([50.0], config)
        assert result.step_end_s == pytest.approx((150.0, 305.0))

    def test_fixed_split_reports_the_fastest_split_when_unset(self):
        config = _config(num_cells=4, cell_capacity=8, batch_groups=4)
        durations = _heavy_tail(64)
        best = replay_fixed_split(durations, config)
        each = [
            replay_fixed_split(
                durations,
                _config(
                    num_cells=4,
                    cell_capacity=8,
                    batch_groups=4,
                    fixed_split_train_cells=cells,
                ),
            )
            for cells in (1, 2, 3)
        ]
        assert best.total_s == min(result.total_s for result in each)
        assert best.train_cells in (1, 2, 3)


class TestElasticReplay:
    def test_replays_the_requested_steps_and_cycles_a_short_trace(self):
        result = replay_elastic([40.0, 300.0, 60.0], _config(num_steps=5))
        assert len(result.step_end_s) == 5
        assert list(result.step_end_s) == sorted(result.step_end_s)
        assert result.total_s == result.step_end_s[-1]

    def test_beats_colocate_on_a_long_tail(self):
        config = _config(
            num_cells=4,
            cell_capacity=24,
            batch_groups=8,
            train_step_s=960.0,
            num_steps=6,
            to_training_s=8.52,
            to_rollout_s=3.46,
        )
        results = replay_all(_heavy_tail(400), config)
        assert results["elastic"].total_s < results["colocate"].total_s
        assert results["elastic"].idle_fraction < results["colocate"].idle_fraction

    def test_does_not_beat_colocate_without_a_tail(self):
        """Uniform durations leave nothing to fill: serial switching only costs time."""
        config = _config(
            num_cells=4,
            cell_capacity=24,
            batch_groups=8,
            train_step_s=960.0,
            num_steps=6,
            initial_dispatch_steps=0.0,
        )
        results = replay_all([300.0] * 64, config)
        assert results["elastic"].total_s >= results["colocate"].total_s


class TestCommandLine:
    def test_prints_a_comparison_table(self, tmp_path, capsys):
        trace = tmp_path / "spans.jsonl"
        trace.write_text(
            "\n".join(
                _otlp_line(_otlp_span(GROUP_SPAN_NAME, float(i), i + duration))
                for i, duration in enumerate(_heavy_tail(50))
            )
        )
        args = [
            "--trace",
            str(trace),
            "--num-cells",
            "4",
            "--cell-capacity",
            "8",
            "--group-size",
            "4",
            "--batch-groups",
            "4",
            "--train-step-s",
            "200",
            "--num-steps",
            "3",
        ]
        main(args)
        table = capsys.readouterr().out
        assert "groups in trace: 50" in table
        for strategy in ("elastic", "fixed_split(", "colocate"):
            assert strategy in table

        main([*args, "--json", "--fixed-split-train-cells", "2"])
        report = json.loads(capsys.readouterr().out)
        assert set(report) == {"elastic", "fixed_split", "colocate"}
        assert report["fixed_split"]["train_cells"] == 2
        assert len(report["elastic"]["step_end_s"]) == 3
