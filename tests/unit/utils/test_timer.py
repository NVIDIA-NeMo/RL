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

import time
from unittest.mock import patch

import numpy as np
import pytest

from nemo_rl.utils.timer import TimeoutChecker, Timer


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
