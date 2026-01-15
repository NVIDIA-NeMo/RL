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
"""Tests for error handling utilities."""

import pytest

from nemo_rl.utils.errors import (
    BackendError,
    CheckpointError,
    ConfigError,
    DataError,
    EnvironmentError,
    NeMoRLError,
    TrainingError,
    format_options,
    fuzzy_match,
    get_common_suggestion,
    get_doc_link,
)


class TestFuzzyMatch:
    """Tests for fuzzy_match function."""

    def test_exact_match(self):
        """Test exact match returns the value."""
        result = fuzzy_match("dtensor", ["dtensor", "megatron"])
        assert result == "dtensor"

    def test_close_match(self):
        """Test close match returns suggestion."""
        result = fuzzy_match("tensor", ["dtensor", "megatron"])
        assert result == "dtensor"

    def test_typo_match(self):
        """Test typo correction."""
        result = fuzzy_match("meagtron", ["dtensor", "megatron"])
        assert result == "megatron"

    def test_no_match(self):
        """Test no match returns None."""
        result = fuzzy_match("xyz", ["dtensor", "megatron"])
        assert result is None

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        result = fuzzy_match("DTENSOR", ["dtensor", "megatron"])
        assert result == "dtensor"

    def test_empty_value(self):
        """Test empty value returns None."""
        result = fuzzy_match("", ["dtensor", "megatron"])
        assert result is None

    def test_empty_options(self):
        """Test empty options returns None."""
        result = fuzzy_match("tensor", [])
        assert result is None


class TestFormatOptions:
    """Tests for format_options function."""

    def test_few_options(self):
        """Test formatting few options."""
        result = format_options(["a", "b", "c"])
        assert "'a'" in result
        assert "'b'" in result
        assert "'c'" in result

    def test_many_options_truncated(self):
        """Test many options are truncated."""
        options = ["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7"]
        result = format_options(options, max_display=3)
        assert "..." in result
        assert "4 more" in result

    def test_empty_options(self):
        """Test empty options."""
        result = format_options([])
        assert "no valid options" in result.lower()

    def test_numeric_options(self):
        """Test numeric options."""
        result = format_options([1, 2, 3])
        assert "1" in result
        assert "2" in result


class TestGetDocLink:
    """Tests for get_doc_link function."""

    def test_known_topic(self):
        """Test known topic returns correct URL."""
        result = get_doc_link("config")
        assert "nvidia.github.io" in result
        assert "/api/config" in result

    def test_unknown_topic(self):
        """Test unknown topic returns base URL."""
        result = get_doc_link("unknown_topic")
        assert "nvidia.github.io" in result


class TestNeMoRLError:
    """Tests for base NeMoRLError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = NeMoRLError("Something went wrong")
        assert "Something went wrong" in str(error)

    def test_error_with_context(self):
        """Test error with context."""
        error = NeMoRLError(
            "Failed to load model",
            context="During trainer initialization",
        )
        error_msg = str(error)
        assert "Failed to load model" in error_msg
        assert "During trainer initialization" in error_msg

    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = NeMoRLError(
            "Invalid configuration",
            suggestion="Check the config file format",
        )
        error_msg = str(error)
        assert "Invalid configuration" in error_msg
        assert "Check the config file format" in error_msg

    def test_error_with_doc_link(self):
        """Test error with documentation link."""
        error = NeMoRLError(
            "Configuration error",
            doc_topic="config",
        )
        error_msg = str(error)
        assert "nvidia.github.io" in error_msg


class TestConfigError:
    """Tests for ConfigError class."""

    def test_basic_config_error(self):
        """Test basic config error."""
        error = ConfigError("Invalid value")
        assert "Configuration Error" in str(error)

    def test_config_error_with_field(self):
        """Test config error with field name."""
        error = ConfigError(
            "Invalid value",
            field="learning_rate",
        )
        error_msg = str(error)
        assert "learning_rate" in error_msg

    def test_config_error_with_value(self):
        """Test config error with invalid value."""
        error = ConfigError(
            "Invalid value",
            field="batch_size",
            value=-1,
        )
        error_msg = str(error)
        assert "-1" in error_msg
        assert "int" in error_msg.lower()  # Shows type

    def test_config_error_with_valid_options(self):
        """Test config error shows valid options."""
        error = ConfigError(
            "Invalid backend",
            field="backend",
            value="tensor",
            valid_options=["dtensor", "megatron"],
        )
        error_msg = str(error)
        assert "dtensor" in error_msg
        assert "megatron" in error_msg

    def test_config_error_fuzzy_suggestion(self):
        """Test config error generates fuzzy match suggestion."""
        error = ConfigError(
            "Invalid backend",
            field="backend",
            value="tensor",
            valid_options=["dtensor", "megatron"],
        )
        error_msg = str(error)
        # Should suggest dtensor as correction
        assert "dtensor" in error_msg

    def test_config_error_with_valid_range(self):
        """Test config error shows valid range."""
        error = ConfigError(
            "Value out of range",
            field="learning_rate",
            value=2.0,
            valid_range=(0.0, 1.0),
        )
        error_msg = str(error)
        assert "0.0" in error_msg or "0" in error_msg
        assert "1.0" in error_msg or "1" in error_msg

    def test_config_error_attributes(self):
        """Test config error attributes are accessible."""
        error = ConfigError(
            "Test error",
            field="test_field",
            value="bad_value",
            expected_type=str,
            valid_options=["a", "b"],
        )
        assert error.field == "test_field"
        assert error.value == "bad_value"
        assert error.expected_type == str
        assert error.valid_options == ["a", "b"]


class TestBackendError:
    """Tests for BackendError class."""

    def test_basic_backend_error(self):
        """Test basic backend error."""
        error = BackendError("Backend initialization failed")
        assert "Backend Error" in str(error)

    def test_backend_error_with_type(self):
        """Test backend error with type."""
        error = BackendError(
            "Invalid backend",
            backend_type="training",
            backend_name="tensor",
        )
        error_msg = str(error)
        assert "training" in error_msg
        assert "tensor" in error_msg.lower()

    def test_backend_error_shows_valid_options(self):
        """Test backend error shows valid backends."""
        error = BackendError(
            "Invalid backend",
            backend_type="training",
            backend_name="tensor",
        )
        error_msg = str(error)
        # Should show valid training backends
        assert "dtensor" in error_msg
        assert "megatron" in error_msg

    def test_backend_error_fuzzy_match(self):
        """Test backend error fuzzy matching."""
        error = BackendError(
            "Invalid backend",
            backend_type="training",
            backend_name="tensor",
        )
        error_msg = str(error)
        # Should suggest dtensor
        assert "dtensor" in error_msg


class TestTrainingError:
    """Tests for TrainingError class."""

    def test_basic_training_error(self):
        """Test basic training error."""
        error = TrainingError("Training failed")
        assert "Training Error" in str(error)

    def test_training_error_with_step(self):
        """Test training error with step info."""
        error = TrainingError(
            "Loss is NaN",
            step=100,
            epoch=2,
        )
        error_msg = str(error)
        assert "100" in error_msg
        assert "2" in error_msg

    def test_training_error_with_batch_info(self):
        """Test training error with batch info."""
        error = TrainingError(
            "Out of memory",
            step=50,
            batch_info={"batch_size": 64, "seq_len": 2048},
        )
        error_msg = str(error)
        assert "64" in error_msg
        assert "2048" in error_msg


class TestDataError:
    """Tests for DataError class."""

    def test_basic_data_error(self):
        """Test basic data error."""
        error = DataError("Invalid data format")
        assert "Data Error" in str(error)

    def test_data_error_with_path(self):
        """Test data error with file path."""
        error = DataError(
            "File not found",
            data_path="/path/to/data.jsonl",
        )
        error_msg = str(error)
        assert "/path/to/data.jsonl" in error_msg

    def test_data_error_with_row_info(self):
        """Test data error with row information."""
        error = DataError(
            "Missing required field",
            data_path="/data.jsonl",
            row_index=42,
            column="prompt",
        )
        error_msg = str(error)
        assert "42" in error_msg
        assert "prompt" in error_msg


class TestEnvironmentError:
    """Tests for EnvironmentError class."""

    def test_basic_environment_error(self):
        """Test basic environment error."""
        error = EnvironmentError("Reward computation failed")
        assert "Environment Error" in str(error)

    def test_environment_error_with_context(self):
        """Test environment error with prompt/response."""
        error = EnvironmentError(
            "Invalid score",
            environment_type="MathReward",
            prompt="What is 2+2?",
            response="The answer is four.",
        )
        error_msg = str(error)
        assert "MathReward" in error_msg
        assert "2+2" in error_msg

    def test_environment_error_truncates_long_text(self):
        """Test environment error truncates long prompts."""
        long_prompt = "x" * 200
        error = EnvironmentError(
            "Error",
            prompt=long_prompt,
        )
        error_msg = str(error)
        # Should be truncated with ...
        assert "..." in error_msg


class TestCheckpointError:
    """Tests for CheckpointError class."""

    def test_basic_checkpoint_error(self):
        """Test basic checkpoint error."""
        error = CheckpointError("Failed to save checkpoint")
        assert "Checkpoint Error" in str(error)

    def test_checkpoint_error_with_path(self):
        """Test checkpoint error with path."""
        error = CheckpointError(
            "Checkpoint corrupted",
            checkpoint_path="/checkpoints/step_1000",
            step=1000,
        )
        error_msg = str(error)
        assert "/checkpoints/step_1000" in error_msg
        assert "1000" in error_msg


class TestGetCommonSuggestion:
    """Tests for get_common_suggestion function."""

    def test_known_error_type(self):
        """Test known error type returns suggestion."""
        suggestion = get_common_suggestion("out_of_memory")
        assert suggestion is not None
        assert "batch size" in suggestion.lower()

    def test_unknown_error_type(self):
        """Test unknown error type returns None."""
        suggestion = get_common_suggestion("unknown_error_type")
        assert suggestion is None

    def test_cuda_not_available_suggestion(self):
        """Test CUDA not available suggestion."""
        suggestion = get_common_suggestion("cuda_not_available")
        assert "NVIDIA" in suggestion or "CUDA" in suggestion

    def test_invalid_model_suggestion(self):
        """Test invalid model suggestion."""
        suggestion = get_common_suggestion("invalid_model")
        assert "HuggingFace" in suggestion or "model" in suggestion.lower()
