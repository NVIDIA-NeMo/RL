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
"""Tests for validation utilities."""

import pytest

from nemo_rl.config.validation import (
    ConfigValidationError,
    validate_config,
    validate_divisible_by,
    validate_non_negative,
    validate_one_of,
    validate_positive,
    validate_power_of_two,
    validate_probability,
    validate_range,
)
from nemo_rl.config.policy import PolicyConfig


class TestValidatePositive:
    """Tests for validate_positive."""

    def test_valid_positive_int(self):
        """Test valid positive integer."""
        assert validate_positive(5, "count") == 5

    def test_valid_positive_float(self):
        """Test valid positive float."""
        assert validate_positive(3.14, "rate") == 3.14

    def test_zero_raises_error(self):
        """Test that zero raises error."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive(0, "count")
        assert "count" in str(exc_info.value)
        assert "positive" in str(exc_info.value).lower()

    def test_negative_raises_error(self):
        """Test that negative value raises error."""
        with pytest.raises(ConfigValidationError):
            validate_positive(-1, "count")


class TestValidateNonNegative:
    """Tests for validate_non_negative."""

    def test_valid_positive(self):
        """Test valid positive value."""
        assert validate_non_negative(5, "count") == 5

    def test_valid_zero(self):
        """Test valid zero value."""
        assert validate_non_negative(0, "count") == 0

    def test_negative_raises_error(self):
        """Test that negative value raises error."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_non_negative(-1, "count")
        assert "non-negative" in str(exc_info.value).lower()


class TestValidateRange:
    """Tests for validate_range."""

    def test_value_in_range(self):
        """Test value within range."""
        assert validate_range(5, "value", min_value=0, max_value=10) == 5

    def test_value_at_min(self):
        """Test value at minimum."""
        assert validate_range(0, "value", min_value=0, max_value=10) == 0

    def test_value_at_max(self):
        """Test value at maximum."""
        assert validate_range(10, "value", min_value=0, max_value=10) == 10

    def test_value_below_min(self):
        """Test value below minimum."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_range(-1, "value", min_value=0, max_value=10)
        assert ">= 0" in str(exc_info.value)

    def test_value_above_max(self):
        """Test value above maximum."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_range(11, "value", min_value=0, max_value=10)
        assert "<= 10" in str(exc_info.value)

    def test_no_min_constraint(self):
        """Test with no minimum constraint."""
        assert validate_range(-100, "value", min_value=None, max_value=10) == -100

    def test_no_max_constraint(self):
        """Test with no maximum constraint."""
        assert validate_range(100, "value", min_value=0, max_value=None) == 100


class TestValidateOneOf:
    """Tests for validate_one_of."""

    def test_valid_value(self):
        """Test valid value in allowed list."""
        assert validate_one_of("a", "choice", ["a", "b", "c"]) == "a"

    def test_invalid_value(self):
        """Test invalid value not in allowed list."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_one_of("d", "choice", ["a", "b", "c"])
        assert "a" in str(exc_info.value)
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)


class TestValidateProbability:
    """Tests for validate_probability."""

    def test_valid_probability(self):
        """Test valid probability."""
        assert validate_probability(0.5, "prob") == 0.5

    def test_zero_probability(self):
        """Test zero probability."""
        assert validate_probability(0.0, "prob") == 0.0

    def test_one_probability(self):
        """Test probability of 1."""
        assert validate_probability(1.0, "prob") == 1.0

    def test_negative_probability(self):
        """Test negative probability."""
        with pytest.raises(ConfigValidationError):
            validate_probability(-0.1, "prob")

    def test_probability_greater_than_one(self):
        """Test probability greater than 1."""
        with pytest.raises(ConfigValidationError):
            validate_probability(1.1, "prob")


class TestValidatePowerOfTwo:
    """Tests for validate_power_of_two."""

    def test_valid_powers_of_two(self):
        """Test valid powers of two."""
        for value in [1, 2, 4, 8, 16, 32, 64, 128]:
            assert validate_power_of_two(value, "size") == value

    def test_invalid_not_power_of_two(self):
        """Test invalid non-power-of-two."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_power_of_two(3, "size")
        assert "power of two" in str(exc_info.value).lower()

    def test_zero_is_invalid(self):
        """Test that zero is invalid."""
        with pytest.raises(ConfigValidationError):
            validate_power_of_two(0, "size")

    def test_negative_is_invalid(self):
        """Test that negative is invalid."""
        with pytest.raises(ConfigValidationError):
            validate_power_of_two(-2, "size")


class TestValidateDivisibleBy:
    """Tests for validate_divisible_by."""

    def test_valid_divisible(self):
        """Test valid divisible value."""
        assert validate_divisible_by(12, "value", 4) == 12

    def test_invalid_not_divisible(self):
        """Test invalid non-divisible value."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_divisible_by(13, "value", 4)
        assert "divisible by 4" in str(exc_info.value)
        # Check that suggestion includes valid alternatives
        assert "12" in str(exc_info.value) or "16" in str(exc_info.value)


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self):
        """Test validating a valid config."""
        config = PolicyConfig(model_name="gpt2")
        assert validate_config(config) is True

    def test_already_validated_config(self):
        """Test validating an already validated config."""
        config = PolicyConfig(model_name="gpt2")
        # Should not raise any errors
        assert validate_config(config) is True


class TestFuzzyMatchingInValidation:
    """Tests for fuzzy matching in validation errors."""

    def test_fuzzy_match_suggests_correction(self):
        """Test that fuzzy matching suggests the correct value."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_one_of("tensor", "backend", ["dtensor", "megatron"])
        error_msg = str(exc_info.value)
        # Should suggest dtensor as correction for tensor
        assert "dtensor" in error_msg.lower()

    def test_fuzzy_match_no_match_shows_options(self):
        """Test that when no fuzzy match, all options are shown."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_one_of("xyz", "backend", ["dtensor", "megatron"])
        error_msg = str(exc_info.value)
        # Should show valid options
        assert "dtensor" in error_msg
        assert "megatron" in error_msg


class TestErrorMessageQuality:
    """Tests for comprehensive error message quality."""

    def test_error_includes_field_name(self):
        """Test that error message includes field name."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive(-1, "learning_rate")
        error_msg = str(exc_info.value)
        assert "learning_rate" in error_msg

    def test_error_includes_value(self):
        """Test that error message includes the invalid value."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive(-5, "batch_size")
        error_msg = str(exc_info.value)
        assert "-5" in error_msg

    def test_error_includes_suggestion(self):
        """Test that error message includes suggestion."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive(-1, "count")
        error_msg = str(exc_info.value)
        assert "suggestion" in error_msg.lower() or ">" in error_msg

    def test_range_error_shows_valid_range(self):
        """Test that range error shows the valid range."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_range(100, "rate", min_value=0, max_value=1)
        error_msg = str(exc_info.value)
        assert "1" in error_msg  # max value shown


class TestConfigValidationErrorClass:
    """Tests for ConfigValidationError class."""

    def test_error_with_all_fields(self):
        """Test error with all fields populated."""
        error = ConfigValidationError(
            message="Invalid configuration",
            field="model.learning_rate",
            value=-0.001,
            suggestion="Use a positive value",
        )
        error_msg = str(error)
        assert "Invalid configuration" in error_msg
        assert "model.learning_rate" in error_msg
        assert "-0.001" in error_msg
        assert "positive" in error_msg.lower()

    def test_error_with_only_message(self):
        """Test error with only message."""
        error = ConfigValidationError(message="Something went wrong")
        assert "Something went wrong" in str(error)

    def test_error_attributes_accessible(self):
        """Test that error attributes are accessible."""
        error = ConfigValidationError(
            message="Test error",
            field="test_field",
            value="bad_value",
            suggestion="Try something else",
        )
        assert error.field == "test_field"
        assert error.value == "bad_value"
        assert error.suggestion == "Try something else"


class TestValidateModelName:
    """Tests for validate_model_name."""

    def test_valid_model_name(self):
        """Test valid model name."""
        from nemo_rl.config.validation import validate_model_name
        assert validate_model_name("Qwen/Qwen2.5-1.5B") == "Qwen/Qwen2.5-1.5B"

    def test_valid_local_path(self):
        """Test valid local path as model name."""
        from nemo_rl.config.validation import validate_model_name
        assert validate_model_name("/path/to/model") == "/path/to/model"

    def test_empty_model_name_raises(self):
        """Test that empty model name raises error."""
        from nemo_rl.config.validation import validate_model_name
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_model_name("")
        assert "empty" in str(exc_info.value).lower()
        assert "HuggingFace" in str(exc_info.value)

    def test_whitespace_model_name_raises(self):
        """Test that whitespace-only model name raises error."""
        from nemo_rl.config.validation import validate_model_name
        with pytest.raises(ConfigValidationError):
            validate_model_name("   ")


class TestValidatePathExists:
    """Tests for validate_path_exists."""

    def test_valid_path_no_check(self):
        """Test valid path without existence check."""
        from nemo_rl.config.validation import validate_path_exists
        result = validate_path_exists("/some/path", "data_path", must_exist=False)
        assert result == "/some/path"

    def test_empty_path_raises(self):
        """Test that empty path raises error."""
        from nemo_rl.config.validation import validate_path_exists
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_path_exists("", "data_path")
        assert "empty" in str(exc_info.value).lower()

    def test_nonexistent_path_with_must_exist(self):
        """Test that nonexistent path raises when must_exist=True."""
        from nemo_rl.config.validation import validate_path_exists
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_path_exists("/nonexistent/path/12345", "data_path", must_exist=True)
        assert "does not exist" in str(exc_info.value)


class TestValidateBatchSizeConsistency:
    """Tests for validate_batch_size_consistency."""

    def test_valid_batch_sizes(self):
        """Test valid batch size configuration."""
        from nemo_rl.config.validation import validate_batch_size_consistency
        # Should not raise
        validate_batch_size_consistency(
            global_batch=64,
            micro_batch=8,
            num_gpus=8,
        )

    def test_global_smaller_than_micro_raises(self):
        """Test that global < micro raises error."""
        from nemo_rl.config.validation import validate_batch_size_consistency
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_batch_size_consistency(
                global_batch=4,
                micro_batch=8,
                num_gpus=1,
            )
        assert "smaller" in str(exc_info.value).lower()

    def test_indivisible_batch_sizes_raises(self):
        """Test that indivisible batch sizes raise error."""
        from nemo_rl.config.validation import validate_batch_size_consistency
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_batch_size_consistency(
                global_batch=100,  # Not divisible by 8*4=32
                micro_batch=8,
                num_gpus=4,
            )
        error_msg = str(exc_info.value)
        assert "divisible" in error_msg.lower()
        # Should suggest valid alternatives
        assert any(str(n) in error_msg for n in [32, 64, 96, 128])


class TestValidateParallelism:
    """Tests for validate_parallelism."""

    def test_valid_parallelism(self):
        """Test valid parallelism configuration."""
        from nemo_rl.config.validation import validate_parallelism
        # Should not raise
        validate_parallelism(
            tensor_parallel=2,
            pipeline_parallel=2,
            num_gpus=8,
        )

    def test_parallelism_exceeds_gpus_raises(self):
        """Test that parallelism exceeding GPUs raises error."""
        from nemo_rl.config.validation import validate_parallelism
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_parallelism(
                tensor_parallel=4,
                pipeline_parallel=4,
                num_gpus=8,
            )
        assert "exceeds" in str(exc_info.value).lower()

    def test_indivisible_parallelism_raises(self):
        """Test that indivisible parallelism raises error."""
        from nemo_rl.config.validation import validate_parallelism
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_parallelism(
                tensor_parallel=3,  # 3*1=3 doesn't divide 8
                pipeline_parallel=1,
                num_gpus=8,
            )
        assert "divisible" in str(exc_info.value).lower()
