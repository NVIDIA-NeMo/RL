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
"""Tests for base configuration class."""

import json
import tempfile
from pathlib import Path
from typing import Annotated

import pytest
import yaml
from pydantic import Field

from nemo_rl.config.base import BaseConfig, ConfigValidationError


class SimpleConfig(BaseConfig):
    """Simple test configuration."""

    name: str
    count: Annotated[int, Field(gt=0)] = 1
    rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    enabled: bool = True


class NestedConfig(BaseConfig):
    """Nested test configuration."""

    simple: SimpleConfig
    extra: str = "default"


class TestBaseConfig:
    """Tests for BaseConfig class."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = SimpleConfig(name="test")
        assert config.name == "test"
        assert config.count == 1
        assert config.rate == 0.5
        assert config.enabled is True

    def test_create_with_custom_values(self):
        """Test creating config with custom values."""
        config = SimpleConfig(name="test", count=5, rate=0.8, enabled=False)
        assert config.name == "test"
        assert config.count == 5
        assert config.rate == 0.8
        assert config.enabled is False

    def test_validation_positive_integer(self):
        """Test validation of positive integer constraint."""
        with pytest.raises(ConfigValidationError) as exc_info:
            SimpleConfig(name="test", count=0)
        assert "count" in str(exc_info.value).lower()

    def test_validation_range_constraint(self):
        """Test validation of range constraint."""
        with pytest.raises(ConfigValidationError) as exc_info:
            SimpleConfig(name="test", rate=1.5)
        assert "rate" in str(exc_info.value).lower()

    def test_validation_negative_rate(self):
        """Test validation of negative rate."""
        with pytest.raises(ConfigValidationError) as exc_info:
            SimpleConfig(name="test", rate=-0.1)
        assert "rate" in str(exc_info.value).lower()

    def test_validation_missing_required(self):
        """Test validation of missing required field."""
        with pytest.raises(ConfigValidationError) as exc_info:
            SimpleConfig()  # Missing required 'name'
        assert "name" in str(exc_info.value).lower()

    def test_immutable_after_creation(self):
        """Test that config is immutable after creation."""
        config = SimpleConfig(name="test")
        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen models
            config.name = "modified"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimpleConfig(name="test", count=3)
        data = config.to_dict()
        assert data == {"name": "test", "count": 3, "rate": 0.5, "enabled": True}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"name": "test", "count": 3}
        config = SimpleConfig.from_dict(data)
        assert config.name == "test"
        assert config.count == 3

    def test_from_dict_invalid(self):
        """Test from_dict with invalid data."""
        with pytest.raises(ConfigValidationError) as exc_info:
            SimpleConfig.from_dict({"name": "test", "count": -1})
        assert "count" in str(exc_info.value).lower()

    def test_to_yaml(self):
        """Test saving to YAML file."""
        config = SimpleConfig(name="test", count=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)
            assert path.exists()

            with open(path) as f:
                data = yaml.safe_load(f)
            assert data["name"] == "test"
            assert data["count"] == 3

    def test_from_yaml(self):
        """Test loading from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            with open(path, "w") as f:
                yaml.dump({"name": "from_yaml", "count": 5}, f)

            config = SimpleConfig.from_yaml(path)
            assert config.name == "from_yaml"
            assert config.count == 5

    def test_from_yaml_not_found(self):
        """Test from_yaml with non-existent file."""
        with pytest.raises(FileNotFoundError):
            SimpleConfig.from_yaml("/nonexistent/path.yaml")

    def test_to_json(self):
        """Test saving to JSON file."""
        config = SimpleConfig(name="test", count=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_json(path)
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "test"
            assert data["count"] == 3

    def test_from_json(self):
        """Test loading from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            with open(path, "w") as f:
                json.dump({"name": "from_json", "count": 7}, f)

            config = SimpleConfig.from_json(path)
            assert config.name == "from_json"
            assert config.count == 7

    def test_from_json_not_found(self):
        """Test from_json with non-existent file."""
        with pytest.raises(FileNotFoundError):
            SimpleConfig.from_json("/nonexistent/path.json")

    def test_nested_config(self):
        """Test nested configuration."""
        config = NestedConfig(simple=SimpleConfig(name="nested"))
        assert config.simple.name == "nested"
        assert config.extra == "default"

    def test_nested_from_dict(self):
        """Test nested configuration from dictionary."""
        data = {"simple": {"name": "nested", "count": 2}, "extra": "custom"}
        config = NestedConfig.from_dict(data)
        assert config.simple.name == "nested"
        assert config.simple.count == 2
        assert config.extra == "custom"

    def test_formats_produce_equal_configs(self):
        """Test that YAML, JSON, and dict formats produce equal configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = SimpleConfig(name="test", count=5, rate=0.3, enabled=False)

            yaml_path = Path(tmpdir) / "config.yaml"
            json_path = Path(tmpdir) / "config.json"

            original.to_yaml(yaml_path)
            original.to_json(json_path)

            from_yaml = SimpleConfig.from_yaml(yaml_path)
            from_json = SimpleConfig.from_json(json_path)
            from_dict = SimpleConfig.from_dict(original.to_dict())

            assert from_yaml.to_dict() == from_json.to_dict() == from_dict.to_dict()


class TestConfigValidationError:
    """Tests for ConfigValidationError class."""

    def test_basic_error(self):
        """Test basic error message."""
        error = ConfigValidationError("Test error")
        assert "Test error" in str(error)

    def test_error_with_field(self):
        """Test error with field information."""
        error = ConfigValidationError("Test error", field="my_field")
        assert "my_field" in str(error)

    def test_error_with_value(self):
        """Test error with value information."""
        error = ConfigValidationError("Test error", value=42)
        assert "42" in str(error)

    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = ConfigValidationError(
            "Test error", suggestion="Try using a positive value"
        )
        assert "positive" in str(error)

    def test_full_error_message(self):
        """Test complete error message."""
        error = ConfigValidationError(
            message="Invalid value",
            field="count",
            value=-1,
            suggestion="Use a positive integer",
        )
        msg = str(error)
        assert "Invalid value" in msg
        assert "count" in msg
        assert "-1" in msg
        assert "positive" in msg
