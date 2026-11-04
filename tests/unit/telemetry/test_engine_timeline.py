# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging

import pytest

import nemo_rl.telemetry.engine_timeline as engine_timeline


class TestTimelineSwitch:
    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_the_opt_in_values_turn_collection_on(self, monkeypatch, value):
        monkeypatch.setenv("NRL_COLLECT_TIMELINE_FIELDS", value)

        assert engine_timeline.timeline_collection_enabled()

    @pytest.mark.parametrize("value", [None, "0", "", "no"])
    def test_collection_is_off_unless_opted_in(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("NRL_COLLECT_TIMELINE_FIELDS", raising=False)
        else:
            monkeypatch.setenv("NRL_COLLECT_TIMELINE_FIELDS", value)

        assert not engine_timeline.timeline_collection_enabled()


class TestAppendRecord:
    def test_appends_one_json_line_per_call(self, tmp_path):
        import json

        path = tmp_path / "records.jsonl"
        engine_timeline.append_record(str(path), {"a": 1})
        engine_timeline.append_record(str(path), {"b": 2})

        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"a": 1},
            {"b": 2},
        ]

    def test_an_unwritable_path_is_dropped_not_raised(self, tmp_path):
        engine_timeline.append_record(
            str(tmp_path / "missing" / "records.jsonl"), {"a": 1}
        )

    def test_an_unserialisable_record_is_dropped_not_raised(self, tmp_path):
        path = tmp_path / "records.jsonl"
        engine_timeline.append_record(str(path), {"bad": object()})

        assert not path.exists()


class TestClockSample:
    def test_recovers_the_offset_and_half_the_round_trip(self):
        # The remote clock runs 5 ms ahead. Each way takes 100 us and the remote
        # holds the probe for 10 us.
        offset_ns, one_way_ns, hold_ns = 5_000_000, 100_000, 10_000
        t0_ns = 1_000
        arrival_ns = t0_ns + one_way_ns + offset_ns
        departure_ns = arrival_ns + hold_ns
        t3_ns = t0_ns + one_way_ns + hold_ns + one_way_ns

        offset_us, uncertainty_us = engine_timeline.clock_sample(
            t0_ns, t3_ns, arrival_ns, departure_ns
        )

        assert offset_us == pytest.approx(5_000.0)
        assert uncertainty_us == pytest.approx(100.0)

    def test_a_remote_clock_behind_the_caller_has_a_negative_offset(self):
        offset_us, _ = engine_timeline.clock_sample(
            t0_ns=0, t3_ns=200_000, arrival_ns=-2_900_000, departure_ns=-2_900_000
        )

        assert offset_us < 0

    def test_uncertainty_is_never_negative(self):
        # A remote hold longer than the whole round trip means the clocks
        # disagree about elapsed time; the bound must not go below zero.
        _, uncertainty_us = engine_timeline.clock_sample(
            t0_ns=0, t3_ns=100, arrival_ns=0, departure_ns=10_000
        )

        assert uncertainty_us == 0.0


class TestCalibrateClusterClocks:
    @staticmethod
    def _record(host, *, node_id):
        return {
            "event": "clock_calibration",
            "host_clock_id": host,
            "offset_us": 1.5,
            "uncertainty_us": 20.0,
            "measured_at_us": 7,
            "sample_count": 8,
            "ray_node_id": node_id,
        }

    def _calibrate(self, monkeypatch, tmp_path, nodes, probes):
        import json

        import ray

        monkeypatch.setattr(ray, "nodes", lambda: nodes)
        monkeypatch.setattr(
            engine_timeline,
            "_probe_node_clock",
            lambda node_id, *, samples, timeout_s: probes[node_id],
        )
        path = tmp_path / "clock_calibration.jsonl"
        recorded = engine_timeline.calibrate_cluster_clocks(str(path))
        rows = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        return recorded, rows

    def test_records_one_row_per_live_host_against_this_clock(
        self, monkeypatch, tmp_path
    ):
        nodes = [
            {"NodeID": "n0", "Alive": True},
            {"NodeID": "n1", "Alive": True},
            {"NodeID": "n2", "Alive": False},
        ]
        probes = {
            "n0": self._record("host-a", node_id="n0"),
            "n1": self._record("host-b", node_id="n1"),
        }

        recorded, rows = self._calibrate(monkeypatch, tmp_path, nodes, probes)

        assert recorded == 2
        assert [row["host_clock_id"] for row in rows] == ["host-a", "host-b"]
        assert {row["reference_clock_id"] for row in rows} == {
            engine_timeline.timeline_clock_id()
        }
        # Counts live nodes only, so a reader can tell a host went unprobed.
        assert {row["expected_ray_node_count"] for row in rows} == {2}

    def test_a_node_that_cannot_be_probed_is_skipped(self, monkeypatch, tmp_path):
        nodes = [{"NodeID": "n0", "Alive": True}, {"NodeID": "n1", "Alive": True}]
        probes = {"n0": None, "n1": self._record("host-b", node_id="n1")}

        recorded, rows = self._calibrate(monkeypatch, tmp_path, nodes, probes)

        assert recorded == 1
        assert [row["host_clock_id"] for row in rows] == ["host-b"]

    def test_a_clock_reported_twice_is_kept_once(self, monkeypatch, tmp_path, caplog):
        nodes = [{"NodeID": "n0", "Alive": True}, {"NodeID": "n1", "Alive": True}]
        probes = {
            "n0": self._record("host-a", node_id="n0"),
            "n1": self._record("host-a", node_id="n1"),
        }

        with caplog.at_level(logging.WARNING):
            recorded, rows = self._calibrate(monkeypatch, tmp_path, nodes, probes)

        assert recorded == 1
        assert [row["ray_node_id"] for row in rows] == ["n0"]
        assert "more than one Ray node" in caplog.text
