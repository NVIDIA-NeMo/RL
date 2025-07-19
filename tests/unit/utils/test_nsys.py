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
from unittest.mock import Mock, patch

import pytest

from nemo_rl.utils.nsys import maybe_gpu_profile_step


class MockPolicy:
    """Mock implementation of ProfilablePolicy for testing."""

    def __init__(self):
        self.start_gpu_profiling = Mock()
        self.stop_gpu_profiling = Mock()

    def __repr__(self):
        return "MockPolicy"


class TestMaybeGpuProfileStep:
    """Test cases for the maybe_gpu_profile_step function."""

    def test_no_environment_variables_set(self):
        """Test that function returns early when no environment variables are set."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", ""),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", ""),
        ):
            maybe_gpu_profile_step(policy, 5)

        # Should not call any profiling methods
        policy.start_gpu_profiling.assert_not_called()
        policy.stop_gpu_profiling.assert_not_called()

    def test_only_worker_patterns_set_raises_assertion_error(self):
        """Test that setting only NRL_NSYS_WORKER_PATTERNS raises AssertionError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", ""),
        ):
            with pytest.raises(
                AssertionError,
                match="Either both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE must be set",
            ):
                maybe_gpu_profile_step(policy, 5)

    def test_only_step_range_set_raises_assertion_error(self):
        """Test that setting only NRL_NSYS_PROFILE_STEP_RANGE raises AssertionError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", ""),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1:10"),
        ):
            with pytest.raises(
                AssertionError,
                match="Either both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE must be set",
            ):
                maybe_gpu_profile_step(policy, 5)

    def test_invalid_step_range_format_missing_colon(self):
        """Test that invalid step range format raises ValueError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1-10"),
        ):
            with pytest.raises(ValueError, match="not enough values to unpack"):
                maybe_gpu_profile_step(policy, 5)

    def test_invalid_step_range_format_non_integer_start(self):
        """Test that non-integer start value raises ValueError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "abc:10"),
        ):
            with pytest.raises(
                ValueError, match="Error parsing NRL_NSYS_PROFILE_STEP_RANGE"
            ):
                maybe_gpu_profile_step(policy, 5)

    def test_invalid_step_range_format_non_integer_stop(self):
        """Test that non-integer stop value raises ValueError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1:xyz"),
        ):
            with pytest.raises(
                ValueError, match="Error parsing NRL_NSYS_PROFILE_STEP_RANGE"
            ):
                maybe_gpu_profile_step(policy, 5)

    def test_start_step_greater_than_stop_step(self):
        """Test that start >= stop raises AssertionError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "10:5"),
        ):
            with pytest.raises(AssertionError, match="must be a non-empty range"):
                maybe_gpu_profile_step(policy, 7)

    def test_start_step_equal_to_stop_step(self):
        """Test that start == stop raises AssertionError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:5"),
        ):
            with pytest.raises(AssertionError, match="must be a non-empty range"):
                maybe_gpu_profile_step(policy, 5)

    def test_start_step_less_than_one(self):
        """Test that start < 1 raises AssertionError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "0:10"),
        ):
            with pytest.raises(AssertionError, match="must be >= 1"):
                maybe_gpu_profile_step(policy, 5)

    def test_negative_start_step(self):
        """Test that negative start raises AssertionError."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "-5:10"),
        ):
            with pytest.raises(AssertionError, match="must be >= 1"):
                maybe_gpu_profile_step(policy, 5)

    @patch("nemo_rl.utils.nsys.rich.print")
    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_start_profiling_in_range(self, mock_atexit_register, mock_rich_print):
        """Test that profiling starts when step is in range."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            maybe_gpu_profile_step(policy, 5)

        # Should start profiling
        policy.start_gpu_profiling.assert_called_once()
        policy.stop_gpu_profiling.assert_not_called()

        # Should set the flag
        assert hasattr(policy, "__NRL_PROFILE_STARTED")
        assert getattr(policy, "__NRL_PROFILE_STARTED") is True

        # Should print status message
        mock_rich_print.assert_called_once()
        assert "Starting GPU profiling" in str(mock_rich_print.call_args)

        # Should register exit handler
        mock_atexit_register.assert_called_once()

    @patch("nemo_rl.utils.nsys.rich.print")
    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_start_profiling_at_start_boundary(
        self, mock_atexit_register, mock_rich_print
    ):
        """Test profiling starts at the exact start step."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "3:8"),
        ):
            maybe_gpu_profile_step(policy, 3)

        policy.start_gpu_profiling.assert_called_once()
        assert getattr(policy, "__NRL_PROFILE_STARTED") is True

    @patch("nemo_rl.utils.nsys.rich.print")
    def test_continue_profiling_when_already_started(self, mock_rich_print):
        """Test that profiling doesn't restart when already started."""
        policy = MockPolicy()
        setattr(policy, "__NRL_PROFILE_STARTED", True)

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            maybe_gpu_profile_step(policy, 6)

        # Should not call start again
        policy.start_gpu_profiling.assert_not_called()
        policy.stop_gpu_profiling.assert_not_called()

        # Should not print anything
        mock_rich_print.assert_not_called()

    @patch("nemo_rl.utils.nsys.rich.print")
    def test_stop_profiling_after_range(self, mock_rich_print):
        """Test that profiling stops when step goes beyond range."""
        policy = MockPolicy()
        setattr(policy, "__NRL_PROFILE_STARTED", True)

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            maybe_gpu_profile_step(policy, 10)  # Exclusive upper bound

        # Should stop profiling
        policy.start_gpu_profiling.assert_not_called()
        policy.stop_gpu_profiling.assert_called_once()

        # Should clear the flag
        assert getattr(policy, "__NRL_PROFILE_STARTED") is False

        # Should print status message
        mock_rich_print.assert_called_once()
        assert "Stopping GPU profiling" in str(mock_rich_print.call_args)

    @patch("nemo_rl.utils.nsys.rich.print")
    def test_stop_profiling_at_stop_boundary(self, mock_rich_print):
        """Test profiling stops at the exact stop step (exclusive)."""
        policy = MockPolicy()
        setattr(policy, "__NRL_PROFILE_STARTED", True)

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "3:8"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
        ):
            maybe_gpu_profile_step(policy, 8)  # Should stop at step 8 (exclusive)

        policy.stop_gpu_profiling.assert_called_once()
        assert getattr(policy, "__NRL_PROFILE_STARTED") is False

    def test_no_action_when_step_before_range(self):
        """Test that no action is taken when step is before range."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            maybe_gpu_profile_step(policy, 3)

        policy.start_gpu_profiling.assert_not_called()
        policy.stop_gpu_profiling.assert_not_called()
        assert not hasattr(policy, "__NRL_PROFILE_STARTED")

    def test_no_action_when_step_after_range_and_not_started(self):
        """Test that no action is taken when step is after range and profiling wasn't started."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            maybe_gpu_profile_step(policy, 15)

        policy.start_gpu_profiling.assert_not_called()
        policy.stop_gpu_profiling.assert_not_called()

    @patch("nemo_rl.utils.nsys.rich.print")
    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_atexit_handler_calls_stop_profiling(
        self, mock_atexit_register, mock_rich_print
    ):
        """Test that the registered atexit handler calls stop_gpu_profiling."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            maybe_gpu_profile_step(policy, 5)

        # Verify atexit handler was registered
        mock_atexit_register.assert_called_once()

        # Get the registered function and call it
        registered_function = mock_atexit_register.call_args[0][0]
        registered_function()

        # The atexit handler should print and call stop_gpu_profiling
        # Note: We called start once during the main call, and stop once in the exit handler
        assert policy.stop_gpu_profiling.call_count == 1

        # Should have printed twice: once for start, once for exit
        assert mock_rich_print.call_count == 2
        assert "Stopping GPU profiling on exit" in str(
            mock_rich_print.call_args_list[1]
        )

    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_single_step_range(self, mock_atexit_register):
        """Test profiling with a single-step range."""
        policy = MockPolicy()

        # Step 5 should start profiling
        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:6"),
        ):
            maybe_gpu_profile_step(policy, 5)
        assert getattr(policy, "__NRL_PROFILE_STARTED") is True
        policy.start_gpu_profiling.assert_called_once()

        # Step 6 should stop profiling
        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:6"),
        ):
            maybe_gpu_profile_step(policy, 6)
        assert getattr(policy, "__NRL_PROFILE_STARTED") is False
        policy.stop_gpu_profiling.assert_called_once()

    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_profiling_sequence_full_lifecycle(self, mock_atexit_register):
        """Test a complete profiling lifecycle from start to stop."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "3:7"),
        ):
            # Before range - no action
            maybe_gpu_profile_step(policy, 1)
            assert not hasattr(policy, "__NRL_PROFILE_STARTED")

            maybe_gpu_profile_step(policy, 2)
            assert not hasattr(policy, "__NRL_PROFILE_STARTED")

            # Start of range - start profiling
            maybe_gpu_profile_step(policy, 3)
            assert getattr(policy, "__NRL_PROFILE_STARTED") is True
            policy.start_gpu_profiling.assert_called_once()

            # Within range - continue profiling (no additional calls)
            maybe_gpu_profile_step(policy, 4)
            maybe_gpu_profile_step(policy, 5)
            maybe_gpu_profile_step(policy, 6)
            policy.start_gpu_profiling.assert_called_once()  # Still only called once

            # End of range - stop profiling
            maybe_gpu_profile_step(policy, 7)
            assert getattr(policy, "__NRL_PROFILE_STARTED") is False
            policy.stop_gpu_profiling.assert_called_once()

            # After range - no additional action
            maybe_gpu_profile_step(policy, 8)
            policy.stop_gpu_profiling.assert_called_once()  # Still only called once

    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_large_step_numbers(self, mock_atexit_register):
        """Test profiling with large step numbers."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1000000:1000002"),
        ):
            maybe_gpu_profile_step(policy, 1000000)

        assert getattr(policy, "__NRL_PROFILE_STARTED") is True
        policy.start_gpu_profiling.assert_called_once()

    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_whitespace_in_environment_variables(self, mock_atexit_register):
        """Test that whitespace in environment variables is handled correctly."""
        policy = MockPolicy()

        # This should work despite whitespace
        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", " worker* "),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", " 5:10 "),
        ):
            maybe_gpu_profile_step(policy, 5)

        policy.start_gpu_profiling.assert_called_once()

    def test_policy_without_profiling_methods_fails(self):
        """Test that a policy without required methods raises AttributeError."""

        class InvalidPolicy:
            pass

        policy = InvalidPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            with pytest.raises(AttributeError):
                maybe_gpu_profile_step(policy, 5)


class TestProfilablePolicy:
    """Test cases for the ProfilablePolicy protocol."""

    def test_mock_policy_implements_protocol(self):
        """Test that our MockPolicy correctly implements the ProfilablePolicy protocol."""
        policy = MockPolicy()

        # Should have the required methods
        assert hasattr(policy, "start_gpu_profiling")
        assert hasattr(policy, "stop_gpu_profiling")
        assert callable(policy.start_gpu_profiling)
        assert callable(policy.stop_gpu_profiling)

    def test_protocol_methods_can_be_called(self):
        """Test that protocol methods can be called."""
        policy = MockPolicy()

        # Should be able to call methods without errors
        policy.start_gpu_profiling()
        policy.stop_gpu_profiling()

        # Verify calls were made
        policy.start_gpu_profiling.assert_called_once()
        policy.stop_gpu_profiling.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_exception_in_start_profiling_propagates(self, mock_atexit_register):
        """Test that exceptions in start_gpu_profiling propagate correctly."""
        policy = MockPolicy()
        policy.start_gpu_profiling.side_effect = RuntimeError("GPU not available")

        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            with pytest.raises(RuntimeError, match="GPU not available"):
                maybe_gpu_profile_step(policy, 5)

    def test_exception_in_stop_profiling_propagates(self):
        """Test that exceptions in stop_gpu_profiling propagate correctly."""
        policy = MockPolicy()
        setattr(policy, "__NRL_PROFILE_STARTED", True)
        policy.stop_gpu_profiling.side_effect = RuntimeError("Profiler error")

        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            with pytest.raises(RuntimeError, match="Profiler error"):
                maybe_gpu_profile_step(policy, 10)

    @patch("nemo_rl.utils.nsys.atexit.register")
    def test_multiple_calls_to_same_step(self, mock_atexit_register):
        """Test calling maybe_gpu_profile_step multiple times with the same step."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.rich.print"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "5:10"),
        ):
            # First call should start profiling
            maybe_gpu_profile_step(policy, 5)
            policy.start_gpu_profiling.assert_called_once()

            # Second call to same step should not call start again
            maybe_gpu_profile_step(policy, 5)
            policy.start_gpu_profiling.assert_called_once()  # Still only called once

    def test_step_zero_handling(self):
        """Test handling of step 0."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1:5"),
        ):
            # Step 0 should not trigger profiling
            maybe_gpu_profile_step(policy, 0)

        policy.start_gpu_profiling.assert_not_called()
        policy.stop_gpu_profiling.assert_not_called()

    def test_negative_step_handling(self):
        """Test handling of negative steps."""
        policy = MockPolicy()

        with (
            patch("nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS", "worker*"),
            patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1:5"),
        ):
            # Negative step should not trigger profiling
            maybe_gpu_profile_step(policy, -5)

        policy.start_gpu_profiling.assert_not_called()
        policy.stop_gpu_profiling.assert_not_called()
