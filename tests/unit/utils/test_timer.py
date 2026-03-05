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
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from nemo_rl.utils.timer import ThreadSafeTimer, TimeoutChecker, Timer


class TestTimer:
    @pytest.fixture
    def timer(self):
        return Timer()

    def test_start_stop(self, timer):
        """Test basic start/stop functionality."""
        timer.start("test_label")
        time.sleep(0.01)  # Small sleep to ensure measurable time
        elapsed = timer.stop("test_label")

        # Check that elapsed time is positive
        assert elapsed > 0

        # Check that the timer recorded the measurement
        assert "test_label" in timer._timers
        assert len(timer._timers["test_label"]) == 1

        # Check that the start time was removed
        assert "test_label" not in timer._start_times

    def test_start_already_running(self, timer):
        """Test that starting an already running timer raises an error."""
        timer.start("test_label")
        with pytest.raises(ValueError):
            timer.start("test_label")

    def test_stop_not_running(self, timer):
        """Test that stopping a timer that isn't running raises an error."""
        with pytest.raises(ValueError):
            timer.stop("nonexistent_label")

    def test_context_manager(self, timer):
        """Test the context manager functionality."""
        with timer.time("test_context"):
            time.sleep(0.01)  # Small sleep to ensure measurable time

        # Check that the timer recorded the measurement
        assert "test_context" in timer._timers
        assert len(timer._timers["test_context"]) == 1

    def test_multiple_measurements(self, timer):
        """Test recording multiple measurements for the same label."""
        for _ in range(3):
            timer.start("multiple")
            time.sleep(0.01)  # Small sleep to ensure measurable time
            timer.stop("multiple")

        # Check that all measurements were recorded
        assert len(timer._timers["multiple"]) == 3

    def test_get_elapsed(self, timer):
        """Test retrieving elapsed times."""
        # Record some measurements
        for _ in range(3):
            timer.start("get_elapsed_test")
            time.sleep(0.01)  # Small sleep to ensure measurable time
            timer.stop("get_elapsed_test")

        # Get the elapsed times
        elapsed_times = timer.get_elapsed("get_elapsed_test")

        # Check that we got the right number of measurements
        assert len(elapsed_times) == 3

        # Check that all times are positive
        for t in elapsed_times:
            assert t > 0

    def test_get_elapsed_nonexistent(self, timer):
        """Test that getting elapsed times for a nonexistent label raises an error."""
        with pytest.raises(KeyError):
            timer.get_elapsed("nonexistent_label")

    def test_reduce_mean(self, timer):
        """Test the mean reduction."""
        # Create known measurements
        timer._timers["reduction_test"] = [1.0, 2.0, 3.0]

        # Get the mean
        mean = timer.reduce("reduction_test", "mean")

        # Check the result
        assert mean == 2.0

    def test_reduce_default(self, timer):
        """Test that the default reduction is mean."""
        # Create known measurements
        timer._timers["reduction_default"] = [1.0, 2.0, 3.0]

        # Get the reduction without specifying type
        result = timer.reduce("reduction_default")

        # Check that it's the mean
        assert result == 2.0

    def test_reduce_all_types(self, timer):
        """Test all reduction types."""
        # Create known measurements
        timer._timers["all_reductions"] = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Test each reduction type
        assert timer.reduce("all_reductions", "mean") == 3.0
        assert timer.reduce("all_reductions", "median") == 3.0
        assert timer.reduce("all_reductions", "min") == 1.0
        assert timer.reduce("all_reductions", "max") == 5.0
        assert timer.reduce("all_reductions", "sum") == 15.0

        # For std, just check it's a reasonable value (avoid floating point comparison issues)
        std = timer.reduce("all_reductions", "std")
        np_std = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(std - np_std) < 1e-6

    def test_reduce_invalid_type(self, timer):
        """Test that an invalid reduction type raises an error."""
        timer._timers["invalid_reduction"] = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError):
            timer.reduce("invalid_reduction", "invalid_type")

    def test_reduce_nonexistent_label(self, timer):
        """Test that getting a reduction for a nonexistent label raises an error."""
        with pytest.raises(KeyError):
            timer.reduce("nonexistent_label")

    def test_reset_specific_label(self, timer):
        """Test resetting a specific label."""
        # Create some measurements
        timer._timers["reset_test1"] = [1.0, 2.0]
        timer._timers["reset_test2"] = [3.0, 4.0]

        # Reset one label
        timer.reset("reset_test1")

        # Check that only that label was reset
        assert "reset_test1" not in timer._timers
        assert "reset_test2" in timer._timers

    def test_reset_all(self, timer):
        """Test resetting all labels."""
        # Create some measurements
        timer._timers["reset_all1"] = [1.0, 2.0]
        timer._timers["reset_all2"] = [3.0, 4.0]

        # Start a timer
        timer.start("running_timer")

        # Reset all
        timer.reset()

        # Check that everything was reset
        assert len(timer._timers) == 0
        assert len(timer._start_times) == 0

    @patch("time.perf_counter")
    def test_precise_timing(self, mock_perf_counter, timer):
        """Test that timing is accurate using mocked time."""
        # Set up mock time to return specific values
        mock_perf_counter.side_effect = [10.0, 15.0]  # Start time, stop time

        # Time something
        timer.start("precise_test")
        elapsed = timer.stop("precise_test")

        # Check the elapsed time
        assert elapsed == 5.0
        assert timer._timers["precise_test"][0] == 5.0

    def test_record(self, timer):
        """Test the record() method for appending pre-measured durations."""
        timer.record("manual", 1.5)
        timer.record("manual", 2.5)
        timer.record("manual", 3.0)

        assert timer._timers["manual"] == [1.5, 2.5, 3.0]
        assert timer.reduce("manual", "sum") == 7.0
        assert timer.reduce("manual", "mean") == pytest.approx(7.0 / 3)

    def test_record_mixed_with_start_stop(self, timer):
        """Test that record() works alongside start/stop."""
        timer.record("mixed", 1.0)
        timer.start("mixed")
        time.sleep(0.01)
        timer.stop("mixed")
        timer.record("mixed", 2.0)

        assert len(timer._timers["mixed"]) == 3
        assert timer._timers["mixed"][0] == 1.0
        assert timer._timers["mixed"][1] > 0  # from start/stop
        assert timer._timers["mixed"][2] == 2.0

    def test_get_latest_elapsed(self, timer):
        """Test get_latest_elapsed returns the most recent measurement."""
        timer._timers["latest"] = [1.0, 2.0, 3.0]
        assert timer.get_latest_elapsed("latest") == 3.0

    def test_get_latest_elapsed_nonexistent(self, timer):
        """Test get_latest_elapsed raises KeyError for missing label."""
        with pytest.raises(KeyError):
            timer.get_latest_elapsed("missing")

    def test_get_timing_metrics_sum(self, timer):
        """Test get_timing_metrics with sum reduction."""
        timer._timers["a"] = [1.0, 2.0]
        timer._timers["b"] = [3.0, 4.0]

        metrics = timer.get_timing_metrics(reduction_op="sum")
        assert metrics["a"] == 3.0
        assert metrics["b"] == 7.0

    def test_mark_basic(self, timer):
        """Test basic mark() records a Unix epoch timestamp."""
        before = time.time()
        ts = timer.mark("event/test")
        after = time.time()

        assert before <= ts <= after
        assert "event/test" in timer._markers
        assert len(timer._markers["event/test"]) == 1
        assert timer._markers["event/test"][0] == (ts, None)

    def test_mark_with_metadata(self, timer):
        """Test mark() with metadata dict."""
        meta = {"worker_id": 3, "error": "OOM"}
        ts = timer.mark("vllm/worker_crashed", metadata=meta)

        markers = timer.get_markers("vllm/worker_crashed")
        assert len(markers["vllm/worker_crashed"]) == 1
        recorded_ts, recorded_meta = markers["vllm/worker_crashed"][0]
        assert recorded_ts == ts
        assert recorded_meta == {"worker_id": 3, "error": "OOM"}

    def test_mark_multiple(self, timer):
        """Test multiple marks under the same label."""
        timer.mark("crash", metadata={"id": 1})
        timer.mark("crash", metadata={"id": 2})
        timer.mark("crash", metadata={"id": 3})

        markers = timer.get_markers("crash")
        assert len(markers["crash"]) == 3
        # Timestamps should be non-decreasing
        timestamps = [m[0] for m in markers["crash"]]
        assert timestamps == sorted(timestamps)
        # Metadata preserved
        ids = [m[1]["id"] for m in markers["crash"]]
        assert ids == [1, 2, 3]

    def test_get_markers_all(self, timer):
        """Test get_markers() with no label returns all markers."""
        timer.mark("a")
        timer.mark("b", metadata={"x": 1})

        all_markers = timer.get_markers()
        assert "a" in all_markers
        assert "b" in all_markers
        assert len(all_markers["a"]) == 1
        assert len(all_markers["b"]) == 1

    def test_get_markers_nonexistent(self, timer):
        """Test get_markers() for a label with no markers returns empty list."""
        markers = timer.get_markers("missing")
        assert markers == {"missing": []}

    def test_get_markers_returns_copy(self, timer):
        """Test that get_markers returns a copy, not a reference."""
        timer.mark("x")
        markers = timer.get_markers("x")
        markers["x"].clear()
        assert len(timer._markers["x"]) == 1  # original unaffected

    def test_reset_clears_markers(self, timer):
        """Test that reset() clears markers too."""
        timer.mark("event")
        timer.record("duration", 1.0)
        timer.reset()
        assert len(timer._markers) == 0
        assert len(timer._timers) == 0

    def test_reset_specific_label_clears_markers(self, timer):
        """Test that reset(label) clears markers for that label."""
        timer.mark("a")
        timer.mark("b")
        timer.reset("a")
        assert "a" not in timer._markers
        assert "b" in timer._markers


class TestTimerLogging:
    """Tests for DEBUG-level log messages emitted by Timer."""

    def test_start_stop_logs(self, caplog):
        timer = Timer(context={"rank": 2, "worker": "policy"})
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.start("idle/refit_bubble")
            timer.stop("idle/refit_bubble")

        assert len(caplog.records) == 2
        assert "rank=2" in caplog.records[0].message
        assert "worker=policy" in caplog.records[0].message
        assert "hostname=" in caplog.records[0].message
        assert "idle/refit_bubble start ts=" in caplog.records[0].message
        assert "idle/refit_bubble end elapsed=" in caplog.records[1].message
        assert " ts=" in caplog.records[1].message

    def test_record_logs(self, caplog):
        timer = Timer(context={"rank": 0, "worker": "generation"})
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.record("wasted/failed_trajectory", 3.14)

        assert len(caplog.records) == 1
        assert "rank=0" in caplog.records[0].message
        assert "worker=generation" in caplog.records[0].message
        assert (
            "wasted/failed_trajectory record elapsed=3.1400s ts="
            in caplog.records[0].message
        )

    def test_mark_logs(self, caplog):
        timer = Timer(context={"rank": 5, "worker": "generation"})
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.mark("vllm/worker_crashed", metadata={"error": "OOM"})

        assert len(caplog.records) == 1
        assert "rank=5" in caplog.records[0].message
        assert "worker=generation" in caplog.records[0].message
        assert "vllm/worker_crashed mark meta=" in caplog.records[0].message
        assert "'error': 'OOM'" in caplog.records[0].message
        assert " ts=" in caplog.records[0].message

    def test_mark_without_metadata_logs(self, caplog):
        timer = Timer(context={"worker": "collector"})
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.mark("heartbeat")

        assert len(caplog.records) == 1
        assert "worker=collector" in caplog.records[0].message
        assert "heartbeat mark ts=" in caplog.records[0].message
        assert "meta=" not in caplog.records[0].message

    def test_no_context_still_logs(self, caplog):
        """Without explicit context, log messages still contain hostname, label, event, and UTC timestamp."""
        timer = Timer()
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.start("x")
            timer.stop("x")
            timer.record("y", 1.0)
            timer.mark("z")

        assert len(caplog.records) == 4
        assert "hostname=" in caplog.records[0].message
        assert "x start ts=" in caplog.records[0].message
        assert "x end elapsed=" in caplog.records[1].message
        assert " ts=" in caplog.records[1].message
        assert "y record elapsed=1.0000s ts=" in caplog.records[2].message
        assert "z mark ts=" in caplog.records[3].message

    def test_no_logs_at_warning_level(self, caplog):
        """At default WARNING level, no timer messages should appear."""
        timer = Timer(context={"rank": 0, "worker": "test"})
        with caplog.at_level(logging.WARNING, logger="nemo_rl.utils.timer"):
            timer.start("x")
            timer.stop("x")
            timer.record("y", 1.0)
            timer.mark("z")

        assert len(caplog.records) == 0

    def test_context_manager_logs_start_end(self, caplog):
        timer = Timer(context={"rank": 1, "worker": "policy"})
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            with timer.time("idle/validation"):
                pass

        assert len(caplog.records) == 2
        assert "start" in caplog.records[0].message
        assert "end elapsed=" in caplog.records[1].message

    def test_context_with_extra_fields(self, caplog):
        """Context can hold arbitrary keys beyond rank/worker."""
        timer = Timer(
            context={
                "rank": 0,
                "worker": "collector",
                "node": "gpu-05",
                "job_id": "slurm-123",
            }
        )
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.mark("event")

        msg = caplog.records[0].message
        assert "rank=0" in msg
        assert "worker=collector" in msg
        assert "node=gpu-05" in msg
        assert "job_id=slurm-123" in msg
        assert "hostname=" in msg

    def test_utc_timestamp_format(self, caplog):
        """Verify the ts= field contains a valid ISO 8601 UTC timestamp."""
        import re

        timer = Timer(context={"worker": "test"})
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.start("op")
            timer.stop("op")

        iso_pattern = re.compile(r"ts=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
        for record in caplog.records:
            assert iso_pattern.search(record.message), (
                f"Expected ISO 8601 UTC timestamp in: {record.message}"
            )

    def test_hostname_auto_injected(self):
        """Hostname is automatically added to context when not provided."""
        import socket

        timer = Timer()
        assert "hostname" in timer._context
        assert timer._context["hostname"] == socket.gethostname()

        timer_with_context = Timer(context={"worker": "test"})
        assert "hostname" in timer_with_context._context
        assert timer_with_context._context["worker"] == "test"
        assert timer_with_context._context["hostname"] == socket.gethostname()

    def test_hostname_override(self, caplog):
        """Caller-provided hostname is preserved (not overwritten)."""
        timer = Timer(context={"worker": "test", "hostname": "custom-host"})
        assert timer._context["hostname"] == "custom-host"

        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.start("op")
        assert "hostname=custom-host" in caplog.records[0].message

    def test_thread_safe_timer_logs(self, caplog):
        """ThreadSafeTimer delegates to super() which has the logging calls."""
        timer = ThreadSafeTimer(context={"rank": 3, "worker": "collector"})
        with caplog.at_level(logging.INFO, logger="nemo_rl.utils.timer"):
            timer.record("idle/buffer_full_backoff", 1.23)

        assert len(caplog.records) == 1
        assert "rank=3" in caplog.records[0].message
        assert "worker=collector" in caplog.records[0].message
        assert (
            "idle/buffer_full_backoff record elapsed=1.2300s ts="
            in caplog.records[0].message
        )


class TestThreadSafeTimer:
    @pytest.fixture
    def timer(self):
        return ThreadSafeTimer()

    def test_basic_operations(self, timer):
        """Test that ThreadSafeTimer has the same API as Timer."""
        timer.start("test")
        time.sleep(0.01)
        elapsed = timer.stop("test")
        assert elapsed > 0

        timer.record("manual", 5.0)
        assert timer.reduce("manual", "sum") == 5.0

        with timer.time("ctx"):
            time.sleep(0.01)
        assert timer.reduce("ctx", "sum") > 0

    def test_concurrent_record(self, timer):
        """Test that concurrent record() calls don't lose data."""
        num_threads = 10
        records_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(records_per_thread):
                timer.record("concurrent", 1.0)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = num_threads * records_per_thread
        assert len(timer._timers["concurrent"]) == expected_total
        assert timer.reduce("concurrent", "sum") == pytest.approx(float(expected_total))

    def test_concurrent_start_stop(self, timer):
        """Test concurrent start/stop with distinct labels."""
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def worker(thread_id):
            barrier.wait()
            label = f"thread_{thread_id}"
            timer.start(label)
            time.sleep(0.01)
            timer.stop(label)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(num_threads):
            assert f"thread_{i}" in timer._timers
            assert len(timer._timers[f"thread_{i}"]) == 1
            assert timer._timers[f"thread_{i}"][0] > 0

    def test_get_timing_metrics_thread_safe(self, timer):
        """Test that get_timing_metrics is safe to call during concurrent writes."""
        stop_event = threading.Event()

        def writer():
            i = 0
            while not stop_event.is_set():
                timer.record("bg", float(i))
                i += 1
                time.sleep(0.001)

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()

        # Read metrics multiple times while writer is active
        for _ in range(10):
            metrics = timer.get_timing_metrics(reduction_op="sum")
            if "bg" in metrics:
                assert metrics["bg"] >= 0
            time.sleep(0.005)

        stop_event.set()
        writer_thread.join()

    def test_concurrent_mark(self, timer):
        """Test that concurrent mark() calls don't lose data."""
        num_threads = 10
        marks_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker(thread_id):
            barrier.wait()
            for i in range(marks_per_thread):
                timer.mark("event", metadata={"thread": thread_id, "i": i})

        threads = [
            threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        markers = timer.get_markers("event")
        assert len(markers["event"]) == num_threads * marks_per_thread

    def test_mark_basic(self, timer):
        """Test mark() works on ThreadSafeTimer."""
        ts = timer.mark("test_event", metadata={"key": "val"})
        markers = timer.get_markers("test_event")
        assert len(markers["test_event"]) == 1
        assert markers["test_event"][0] == (ts, {"key": "val"})

    def test_reset(self, timer):
        """Test reset clears all data including markers."""
        timer.record("a", 1.0)
        timer.record("b", 2.0)
        timer.mark("event")
        timer.reset()
        assert len(timer._timers) == 0
        assert len(timer._markers) == 0


class TestTimeoutChecker:
    def test_infinite_timeout(self):
        checker = TimeoutChecker(timeout=None)
        time.sleep(0.1)
        assert checker.check_save() is False

    def test_short_timeout(self):
        checker = TimeoutChecker(timeout="00:00:00:01")
        time.sleep(1.1)
        assert checker.check_save() is True

    def test_double_save_prevented(self):
        checker = TimeoutChecker(timeout="00:00:00:01")
        time.sleep(1.1)
        assert checker.check_save() is True
        assert checker.check_save() is False

    def test_fit_last_save_time_enabled(self):
        # Create a TimeoutChecker with a 3-second timeout and enable fit_last_save_time logic
        checker = TimeoutChecker(timeout="00:00:00:03", fit_last_save_time=True)
        checker.start_iterations()

        # Simulate 10 iterations, each taking about 0.1 seconds
        # This builds up a stable average iteration time
        for _ in range(10):
            time.sleep(0.1)
            checker.mark_iteration()

        # Wait an additional ~2.0 seconds so that:
        # elapsed time + avg iteration time >= timeout (3 seconds)
        time.sleep(2.0)

        result = checker.check_save()
        # Assert that the checker triggers a save due to timeout
        assert result is True

    def test_iteration_tracking(self):
        checker = TimeoutChecker()
        checker.start_iterations()
        time.sleep(0.05)
        checker.mark_iteration()
        assert len(checker.iteration_times) == 1
        assert checker.iteration_times[0] > 0
