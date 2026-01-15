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
"""Tests for backward-compatible configuration utilities."""

import warnings
import tempfile
import pytest

from nemo_rl.compat.config import (
    load_yaml_config,
    convert_omegaconf_to_dict,
    convert_old_config_to_new,
    OmegaConfAdapter,
)


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_emits_deprecation_warning(self):
        """Test that load_yaml_config emits deprecation warning."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\n")
            temp_path = f.name
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_yaml_config(temp_path)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "load_yaml_config" in str(w[0].message)

    def test_loads_yaml_content(self):
        """Test that YAML content is loaded correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("policy:\n  model_name: test-model\n  lr: 0.001\n")
            temp_path = f.name
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = load_yaml_config(temp_path)
            
            assert result["policy"]["model_name"] == "test-model"
            assert result["policy"]["lr"] == 0.001


class TestConvertOmegaconfToDict:
    """Tests for convert_omegaconf_to_dict function."""

    def test_converts_dict_passthrough(self):
        """Test that plain dicts pass through."""
        data = {"key": "value"}
        result = convert_omegaconf_to_dict(data)
        assert result == data

    def test_converts_object_with_to_container(self):
        """Test conversion of objects with to_container method."""
        class MockOmegaConf:
            def to_container(self, resolve=True):
                return {"resolved": True}
        
        result = convert_omegaconf_to_dict(MockOmegaConf())
        assert result == {"resolved": True}

    def test_converts_object_with_to_dict(self):
        """Test conversion of objects with to_dict method."""
        class MockConfig:
            def to_dict(self):
                return {"dict_method": True}
        
        result = convert_omegaconf_to_dict(MockConfig())
        assert result == {"dict_method": True}

    def test_returns_empty_dict_for_unknown(self):
        """Test that unknown types return empty dict."""
        result = convert_omegaconf_to_dict(None)
        assert result == {}


class TestConvertOldConfigToNew:
    """Tests for convert_old_config_to_new function."""

    def test_emits_deprecation_warning(self):
        """Test that convert_old_config_to_new emits warning."""
        old_config = {"policy": {"model_name": "test"}}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            convert_old_config_to_new(old_config, "grpo")
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_converts_policy_config(self):
        """Test that policy config is converted."""
        old_config = {
            "policy": {
                "model_name": "test-model",
                "precision": "bfloat16",
                "train_global_batch_size": 64,
            }
        }
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            new_config = convert_old_config_to_new(old_config, "grpo")
            
            assert new_config["policy"]["model_name"] == "test-model"
            assert new_config["policy"]["precision"] == "bfloat16"

    def test_converts_grpo_config(self):
        """Test that GRPO-specific config is converted."""
        old_config = {
            "policy": {"model_name": "test"},
            "grpo": {
                "num_prompts_per_step": 32,
                "max_num_steps": 1000,
            }
        }
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            new_config = convert_old_config_to_new(old_config, "grpo")
            
            assert new_config["num_prompts_per_step"] == 32
            assert new_config["max_num_steps"] == 1000

    def test_converts_optimizer_config(self):
        """Test that optimizer config is converted."""
        old_config = {
            "policy": {
                "model_name": "test",
                "optimizer": {
                    "name": "torch.optim.AdamW",
                    "kwargs": {"lr": 1e-5}
                }
            }
        }
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            new_config = convert_old_config_to_new(old_config, "grpo")
            
            assert new_config["policy"]["optimizer"]["name"] == "adamw"
            assert new_config["policy"]["optimizer"]["kwargs"]["lr"] == 1e-5


class TestOmegaConfAdapter:
    """Tests for OmegaConfAdapter class."""

    def test_emits_deprecation_warning(self):
        """Test that OmegaConfAdapter emits warning on creation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = OmegaConfAdapter({"key": "value"})
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OmegaConfAdapter" in str(w[0].message)

    def test_attribute_access(self):
        """Test OmegaConf-style attribute access."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = OmegaConfAdapter({"key": "value", "nested": {"inner": 42}})
            
            assert adapter.key == "value"
            assert adapter.nested.inner == 42

    def test_attribute_setting(self):
        """Test OmegaConf-style attribute setting."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = OmegaConfAdapter({"key": "value"})
            adapter.key = "new_value"
            
            assert adapter.key == "new_value"

    def test_to_container(self):
        """Test to_container method for OmegaConf compatibility."""
        data = {"key": "value"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = OmegaConfAdapter(data)
            
            result = adapter.to_container()
            assert result == data

    def test_to_dict(self):
        """Test to_dict method."""
        data = {"key": "value"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = OmegaConfAdapter(data)
            
            result = adapter.to_dict()
            assert result == data
