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

import sys
from pathlib import Path

import pytest

# Add the tests directory to the path so we can import check_metrics
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from check_metrics import evaluate_check, max, mean, median, min, ratio_above


class TestMeanFunction:
    """Test the mean function with various scenarios."""

    def test_basic_mean(self):
        """Test basic mean calculation without outlier filtering."""
        data = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}
        result = mean(data)
        assert result == 3.0

    def test_mean_with_ignore_top_p(self):
        """Test mean with ignore_top_p to filter outliers."""
        # Data with one clear outlier (100)
        data = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 100.0}

        # Without filtering
        result_no_filter = mean(data, ignore_top_p=0.0)
        assert result_no_filter == 22.0  # (1+2+3+4+100)/5

        # With 20% filtering (should remove the top value: 100)
        result_with_filter = mean(data, ignore_top_p=0.2)
        assert result_with_filter == 2.5  # (1+2+3+4)/4

    def test_mean_ignore_top_5_percent(self):
        """Test mean with 5% outlier filtering."""
        # Create data with 20 values where the top one is an outlier
        data = {str(i): float(i) for i in range(1, 20)}  # 1-19
        data["20"] = 1000.0  # outlier

        # With 5% filtering (should remove 1 value out of 20 = top 5%)
        result = mean(data, ignore_top_p=0.05)
        # Should be mean of 1-19 = 10.0
        assert result == 10.0

    def test_mean_ignore_multiple_outliers(self):
        """Test mean with filtering multiple outliers."""
        data = {str(i): float(i) for i in range(1, 11)}  # 1-10

        # With 20% filtering (should remove top 2 values: 9, 10)
        result = mean(data, ignore_top_p=0.2)
        # Mean of 1-8 = 4.5
        assert result == 4.5

    def test_mean_with_range_and_ignore_top_p(self):
        """Test that range_start and range_end work with ignore_top_p."""
        data = {str(i): float(i) for i in range(1, 11)}

        # Get mean of steps 3-7 (values 3,4,5,6) with 25% filtering
        # Should remove the top value (6), leaving 3,4,5
        result = mean(data, range_start=3, range_end=7, ignore_top_p=0.25)
        assert result == 4.0  # (3+4+5)/3

    def test_mean_ignore_top_p_edge_case_all_same(self):
        """Test with all same values (no outliers)."""
        data = {str(i): 5.0 for i in range(1, 11)}
        result = mean(data, ignore_top_p=0.1)
        assert result == 5.0

    def test_mean_ignore_top_p_edge_case_single_value(self):
        """Test with single value."""
        data = {"1": 42.0}
        result = mean(data, ignore_top_p=0.5)
        # Should keep at least one value
        assert result == 42.0

    def test_mean_ignore_top_p_edge_case_two_values(self):
        """Test with two values."""
        data = {"1": 1.0, "2": 10.0}
        result = mean(data, ignore_top_p=0.5)
        # Should remove top 50% (1 value), leaving just 1.0
        assert result == 1.0

    def test_mean_ignore_top_p_invalid_range(self):
        """Test that invalid ignore_top_p values raise an error."""
        data = {"1": 1.0, "2": 2.0, "3": 3.0}

        with pytest.raises(
            ValueError, match="ignore_top_p must be between 0.0 and 1.0"
        ):
            mean(data, ignore_top_p=1.5)

        with pytest.raises(
            ValueError, match="ignore_top_p must be between 0.0 and 1.0"
        ):
            mean(data, ignore_top_p=-0.1)

    def test_mean_with_offset(self):
        """Test mean calculation with step offset (from checkpoint resume)."""
        # Simulate a checkpoint resume scenario
        # Steps 101-105 (resumed from step 100)
        data = {"101": 1.0, "102": 2.0, "103": 3.0, "104": 4.0, "105": 5.0}
        result = mean(data)
        assert result == 3.0

    def test_mean_with_negative_range(self):
        """Test mean with negative range indices."""
        data = {str(i): float(i) for i in range(1, 11)}  # 1-10

        # Last 3 values (8, 9, 10)
        result = mean(data, range_start=-3, range_end=0)
        assert result == 9.0  # (8+9+10)/3

    def test_mean_with_floats_and_strings(self):
        """Test that string values are properly converted to floats."""
        data = {"1": "1.5", "2": "2.5", "3": "3.5"}
        result = mean(data)
        assert result == 2.5


class TestMinMaxFunctions:
    """Test the min and max helper functions."""

    def test_min_basic(self):
        """Test basic min functionality."""
        data = {"1": 5.0, "2": 2.0, "3": 8.0, "4": 1.0}
        result = min(data)
        assert result == 1.0

    def test_max_basic(self):
        """Test basic max functionality."""
        data = {"1": 5.0, "2": 2.0, "3": 8.0, "4": 1.0}
        result = max(data)
        assert result == 8.0

    def test_min_with_string_values(self):
        """Test min with string numeric values."""
        data = {"1": "5.5", "2": "2.2", "3": "8.8"}
        result = min(data)
        assert result == 2.2

    def test_max_with_string_values(self):
        """Test max with string numeric values."""
        data = {"1": "5.5", "2": "2.2", "3": "8.8"}
        result = max(data)
        assert result == 8.8


class TestRatioAboveFunction:
    """Test the ratio_above function."""

    def test_ratio_above_basic(self):
        """Test basic ratio_above calculation."""
        data = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}
        # Values >= 3.0 are: 3.0, 4.0, 5.0 (3 out of 5 = 0.6)
        result = ratio_above(data, 3.0)
        assert result == 0.6

    def test_ratio_above_none_above(self):
        """Test when no values are above threshold."""
        data = {"1": 1.0, "2": 2.0, "3": 3.0}
        result = ratio_above(data, 10.0)
        assert result == 0.0

    def test_ratio_above_all_above(self):
        """Test when all values are above threshold."""
        data = {"1": 5.0, "2": 6.0, "3": 7.0}
        result = ratio_above(data, 4.0)
        assert result == 1.0

    def test_ratio_above_equal_to_threshold(self):
        """Test that values equal to threshold are counted (>=)."""
        data = {"1": 1.0, "2": 2.0, "3": 2.0, "4": 3.0}
        # Values >= 2.0 are: 2.0, 2.0, 3.0 (3 out of 4 = 0.75)
        result = ratio_above(data, 2.0)
        assert result == 0.75

    def test_ratio_above_single_value(self):
        """Test with single value."""
        data = {"1": 5.0}
        result = ratio_above(data, 3.0)
        assert result == 1.0

        result = ratio_above(data, 10.0)
        assert result == 0.0

    def test_ratio_above_empty_dict(self):
        """Test with empty dictionary."""
        data = {}
        result = ratio_above(data, 1.0)
        assert result == 0.0

    def test_ratio_above_with_strings(self):
        """Test that string values are properly converted."""
        data = {"1": "1.0", "2": "2.0", "3": "3.0", "4": "4.0", "5": "5.0"}
        result = ratio_above(data, 3.0)
        assert result == 0.6

    def test_ratio_above_with_floats(self):
        """Test with float threshold and values."""
        data = {"1": 1.05, "2": 1.1, "3": 1.0, "4": 1.2, "5": 0.9}
        # Values >= 1.05: 1.05, 1.1, 1.2 (3 out of 5 = 0.6)
        result = ratio_above(data, 1.05)
        assert result == 0.6


class TestEvaluateCheck:
    """Test the evaluate_check function."""

    def test_evaluate_check_pass(self):
        """Test a passing check."""
        data = {"accuracy": {"1": 0.9, "2": 0.95}}
        passed, message, value = evaluate_check(data, "mean(data['accuracy']) > 0.85")
        assert passed is True
        assert "PASS" in message
        assert value == 0.925

    def test_evaluate_check_fail(self):
        """Test a failing check."""
        data = {"accuracy": {"1": 0.7, "2": 0.75}}
        passed, message, value = evaluate_check(data, "mean(data['accuracy']) > 0.85")
        assert passed is False
        assert "FAIL" in message
        assert value == 0.725

    def test_evaluate_check_with_ignore_top_p(self):
        """Test evaluate_check with ignore_top_p parameter."""
        data = {"error": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 10.0}}

        # Without filtering, mean would be 2.8, which is > 1.5 (should fail the < check)
        passed_no_filter, _, value_no_filter = evaluate_check(
            data, "mean(data['error']) < 1.5"
        )
        assert passed_no_filter is False
        assert value_no_filter == 2.8

        # With 20% filtering, mean should be 1.0, which is < 1.5 (should pass)
        passed_with_filter, _, value_with_filter = evaluate_check(
            data, "mean(data['error'], ignore_top_p=0.2) < 1.5"
        )
        assert passed_with_filter is True
        assert value_with_filter == 1.0

    def test_evaluate_check_key_error(self):
        """Test evaluate_check with missing key."""
        data = {"accuracy": {"1": 0.9}}
        passed, message, value = evaluate_check(data, "mean(data['missing']) > 0.5")
        assert passed is False
        assert "key not found" in message
        assert value is None

    def test_evaluate_check_multiple_conditions(self):
        """Test evaluate_check with complex conditions."""
        data = {
            "train_loss": {"1": 0.5, "2": 0.4, "3": 0.3},
            "val_loss": {"1": 0.6, "2": 0.5, "3": 0.4},
        }

        # Test less than
        passed, _, value = evaluate_check(data, "mean(data['train_loss']) < 0.5")
        assert passed is True
        assert value == 0.4

        # Test greater than
        passed, _, value = evaluate_check(data, "mean(data['val_loss']) > 0.4")
        assert passed is True
        assert value == 0.5

    def test_evaluate_check_with_min_max(self):
        """Test evaluate_check with min and max functions."""
        data = {"scores": {"1": 1.0, "2": 5.0, "3": 3.0}}

        passed, _, value = evaluate_check(data, "min(data['scores']) > 0.5")
        assert passed is True
        assert value == 1.0

        passed, _, value = evaluate_check(data, "max(data['scores']) < 10.0")
        assert passed is True
        assert value == 5.0

    def test_evaluate_check_with_ratio_above(self):
        """Test evaluate_check with ratio_above function."""
        data = {"error": {"1": 1.0, "2": 1.0, "3": 1.5, "4": 1.0, "5": 2.0}}

        # 2 out of 5 values are >= 1.5 (ratio = 0.4)
        passed, _, value = evaluate_check(data, "ratio_above(data['error'], 1.5) < 0.5")
        assert passed is True
        assert value == 0.4

        # Should fail when ratio is above threshold
        passed, _, value = evaluate_check(data, "ratio_above(data['error'], 1.5) < 0.3")
        assert passed is False
        assert value == 0.4


class TestRealWorldScenarios:
    """Test scenarios that match real-world usage patterns."""

    def test_token_prob_error_scenario(self):
        """Test the exact scenario from the user's example."""
        # Simulate token_mult_prob_error with some outliers
        data = {
            "train/token_mult_prob_error": {
                str(i): 1.0 + (i % 3) * 0.01 for i in range(1, 20)
            }
        }
        # Add a couple large outliers that will skew the mean
        data["train/token_mult_prob_error"]["20"] = 5.0

        # Without filtering, mean should be significantly above 1.1
        passed_no_filter, _, value_no_filter = evaluate_check(
            data, 'mean(data["train/token_mult_prob_error"]) < 1.1'
        )
        assert passed_no_filter is False  # Should fail due to outlier
        assert value_no_filter > 1.1

        # With 5% filtering (removes 1 out of 20 = top 5%)
        passed_with_filter, _, value_with_filter = evaluate_check(
            data, 'mean(data["train/token_mult_prob_error"], ignore_top_p=0.05) < 1.1'
        )
        assert passed_with_filter is True  # Should pass with outlier removed
        assert value_with_filter < 1.1

    def test_large_dataset_with_few_outliers(self):
        """Test with a large dataset containing a few outliers."""
        # Create 100 normal values around 1.0
        data = {"metric": {str(i): 1.0 + (i % 10) * 0.01 for i in range(1, 101)}}
        # Add 5 outliers
        for i in range(101, 106):
            data["metric"][str(i)] = 10.0

        # Without filtering
        mean_no_filter = mean(data["metric"], ignore_top_p=0.0)
        assert mean_no_filter > 1.4  # Significantly affected by outliers

        # With 5% filtering (should remove ~5 values, including the outliers)
        mean_with_filter = mean(data["metric"], ignore_top_p=0.05)
        assert mean_with_filter < 1.1  # Should be close to 1.0

    def test_robustness_to_varying_outlier_severity(self):
        """Test that filtering works with outliers of varying severity."""
        base_data = {str(i): 1.0 for i in range(1, 10)}

        # Test with mild outlier
        data_mild = base_data.copy()
        data_mild["10"] = 2.0
        result_mild = mean(data_mild, ignore_top_p=0.1)
        assert result_mild == 1.0  # Outlier removed

        # Test with severe outlier
        data_severe = base_data.copy()
        data_severe["10"] = 100.0
        result_severe = mean(data_severe, ignore_top_p=0.1)
        assert result_severe == 1.0  # Outlier removed

    def test_ratio_above_real_world_scenario(self):
        """Test the exact scenario from the user's example with ratio_above."""
        # Simulate token_mult_prob_error where most values are around 1.0
        # but a few are above 1.05
        data = {
            "train/token_mult_prob_error": {
                str(i): 1.0 + (i % 20) * 0.001 for i in range(1, 101)
            }
        }
        # Add a few values above 1.05 (should be 1 out of 100 = 1%)
        data["train/token_mult_prob_error"]["50"] = 1.06

        # Check that less than 2% of values are above 1.05
        passed, _, value = evaluate_check(
            data, 'ratio_above(data["train/token_mult_prob_error"], 1.05) < 0.02'
        )
        assert passed is True
        assert value == 0.01  # 1 out of 100

    def test_ratio_above_combined_with_mean_ignore_top_p(self):
        """Test combining ratio_above check with mean ignore_top_p."""
        # Create data where a few outliers would skew the mean
        data = {"metric": {str(i): 1.0 for i in range(1, 96)}}
        # Add 5 outliers (5%)
        for i in range(96, 101):
            data["metric"][str(i)] = 10.0

        # Without filtering, mean would be high
        mean_no_filter = mean(data["metric"], ignore_top_p=0.0)
        assert mean_no_filter > 1.4

        # With 5% filtering, mean should be close to 1.0
        mean_with_filter = mean(data["metric"], ignore_top_p=0.05)
        assert mean_with_filter < 1.1

        # Check that exactly 5% are above threshold
        ratio = ratio_above(data["metric"], 5.0)
        assert ratio == 0.05


class TestMedianFunction:
    """Test the median function with various scenarios."""

    def test_basic_median_odd_count(self):
        """Test basic median calculation with odd number of values."""
        data = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}
        result = median(data)
        assert result == 3.0

    def test_basic_median_even_count(self):
        """Test basic median calculation with even number of values."""
        data = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0}
        result = median(data)
        assert result == 2.5  # (2+3)/2

    def test_median_with_outliers(self):
        """Test that median is robust to outliers (unlike mean)."""
        # Data with one severe outlier
        data = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 100.0}

        # Median should be unaffected by the outlier
        result = median(data)
        assert result == 3.0  # Middle value, ignores the outlier

        # Compare with mean which would be 22.0
        result_mean = mean(data)
        assert result_mean == 22.0

    def test_median_with_range(self):
        """Test median with range_start and range_end."""
        data = {str(i): float(i) for i in range(1, 11)}  # 1-10

        # Get median of steps 3-7 (values 3, 4, 5, 6)
        result = median(data, range_start=3, range_end=7)
        assert result == 4.5  # (4+5)/2

    def test_median_with_offset(self):
        """Test median calculation with step offset (from checkpoint resume)."""
        # Simulate a checkpoint resume scenario
        # Steps 101-105 (resumed from step 100)
        data = {"101": 1.0, "102": 2.0, "103": 3.0, "104": 4.0, "105": 5.0}
        result = median(data)
        assert result == 3.0

    def test_median_with_negative_range(self):
        """Test median with negative range indices."""
        data = {str(i): float(i) for i in range(1, 11)}  # 1-10

        # Last 3 values (8, 9, 10)
        result = median(data, range_start=-3, range_end=0)
        assert result == 9.0

    def test_median_with_floats_and_strings(self):
        """Test that string values are properly converted to floats."""
        data = {"1": "1.5", "2": "2.5", "3": "3.5"}
        result = median(data)
        assert result == 2.5

    def test_median_single_value(self):
        """Test median with single value."""
        data = {"1": 42.0}
        result = median(data)
        assert result == 42.0

    def test_median_two_values(self):
        """Test median with two values."""
        data = {"1": 1.0, "2": 10.0}
        result = median(data)
        assert result == 5.5  # (1+10)/2

    def test_median_all_same_values(self):
        """Test median with all same values."""
        data = {str(i): 5.0 for i in range(1, 11)}
        result = median(data)
        assert result == 5.0

    def test_median_vs_mean_outlier_robustness(self):
        """Demonstrate why median is preferred for outlier-prone data."""
        # Simulate token_mult_prob_error with outliers
        data = {str(i): 1.0 for i in range(1, 20)}
        data["20"] = 100.0  # Large outlier

        # Median is robust
        median_result = median(data)
        assert median_result == 1.0

        # Mean is affected
        mean_result = mean(data)
        assert mean_result == 5.95  # (19 * 1.0 + 100) / 20

    def test_evaluate_check_with_median(self):
        """Test evaluate_check with median function."""
        data = {"accuracy": {"1": 0.8, "2": 0.9, "3": 0.95, "4": 0.85, "5": 100.0}}

        # Median should be robust to the outlier
        passed, _, value = evaluate_check(data, "median(data['accuracy']) < 1.0")
        assert passed is True
        assert value == 0.9  # Middle value

    def test_median_token_mult_prob_error_scenario(self):
        """Test the exact scenario for which median is being used."""
        # Simulate token_mult_prob_error with some outliers
        data = {
            "train/token_mult_prob_error": {
                str(i): 1.0 + (i % 3) * 0.01 for i in range(1, 20)
            }
        }
        # Add outlier
        data["train/token_mult_prob_error"]["20"] = 5.0

        # Median should pass the check even with the outlier
        passed, _, value = evaluate_check(
            data, 'median(data["train/token_mult_prob_error"]) < 1.05'
        )
        assert passed is True
        assert value < 1.05
