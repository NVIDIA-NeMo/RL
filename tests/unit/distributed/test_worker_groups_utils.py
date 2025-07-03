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
"""Tests for worker groups utility functions."""

from copy import deepcopy

from nemo_rl.distributed.worker_group_utils import recursive_merge_options


class TestRecursiveMergeOptions:
    """Test cases for the recursive_merge_options function."""

    def test_simple_merge(self):
        """Test simple merging of two dictionaries."""
        default_options = {"a": 1, "b": 2}
        extra_options = {"c": 3, "d": 4}

        result = recursive_merge_options(default_options, extra_options)

        expected = {"a": 1, "b": 2, "c": 3, "d": 4}

        assert result == expected

    def test_override_values(self):
        """Test that extra_options override default_options."""
        default_options = {"a": 1, "b": 2}
        extra_options = {
            "a": 10,  # Override
            "c": 3,
        }

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "a": 10,  # Overridden value
            "b": 2,
            "c": 3,
        }

        assert result == expected

    def test_nested_merge(self):
        """Test recursive merging of nested dictionaries."""
        default_options = {"level1": {"a": 1, "b": 2, "nested": {"x": 10, "y": 20}}}
        extra_options = {
            "level1": {
                "b": 20,  # Override
                "c": 3,  # New key
                "nested": {
                    "y": 200,  # Override nested
                    "z": 30,  # New nested key
                },
            }
        }

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "level1": {
                "a": 1,
                "b": 20,  # Overridden
                "c": 3,  # Added
                "nested": {
                    "x": 10,
                    "y": 200,  # Overridden
                    "z": 30,  # Added
                },
            }
        }

        assert result == expected

    def test_scalar_replaces_dict(self):
        """Test that scalar values can replace dictionary values."""
        default_options = {"config": {"nested": {"key": "value"}}}
        extra_options = {
            "config": "simple_string"  # Scalar replaces entire dict
        }

        result = recursive_merge_options(default_options, extra_options)

        expected = {"config": "simple_string"}

        assert result == expected

    def test_dict_replaces_scalar(self):
        """Test that dictionary values can replace scalar values."""
        default_options = {"config": "simple_string"}
        extra_options = {
            "config": {  # Dict replaces scalar
                "nested": {"key": "value"}
            }
        }

        result = recursive_merge_options(default_options, extra_options)

        expected = {"config": {"nested": {"key": "value"}}}

        assert result == expected

    def test_nsight_transformation(self):
        """Test the special _nsight -> nsight transformation."""
        default_options = {
            "runtime_env": {
                "_nsight": {"profile": True, "output": "profile.nsys-rep"},
                "env_vars": {"CUDA_VISIBLE_DEVICES": "0"},
            }
        }
        extra_options = {"runtime_env": {"env_vars": {"PYTHONPATH": "/custom/path"}}}

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "runtime_env": {
                "nsight": {  # _nsight transformed to nsight
                    "profile": True,
                    "output": "profile.nsys-rep",
                },
                "env_vars": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/custom/path"},
            }
        }

        assert result == expected
        # Ensure _nsight is removed
        assert "_nsight" not in result["runtime_env"]

    def test_nsight_transformation_with_override(self):
        """Test that extra_options can override transformed nsight config."""
        default_options = {
            "runtime_env": {"_nsight": {"profile": True, "output": "default.nsys-rep"}}
        }
        extra_options = {
            "runtime_env": {
                "nsight": {  # Should override the transformed _nsight
                    "profile": True,
                    "output": "custom.nsys-rep",
                    "extra_flag": "--custom",
                }
            }
        }

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "runtime_env": {
                "nsight": {
                    "profile": True,
                    "output": "custom.nsys-rep",  # Overridden
                    "extra_flag": "--custom",  # Added
                }
            }
        }

        assert result == expected

    def test_no_nsight_transformation_when_nsight_exists(self):
        """Test that _nsight is not transformed when nsight already exists."""
        default_options = {
            "runtime_env": {
                "_nsight": {"profile": True},
                "nsight": {  # Already exists
                    "output": "existing.nsys-rep"
                },
            }
        }
        extra_options = {}

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "runtime_env": {
                "_nsight": {  # Should remain as _nsight
                    "profile": True
                },
                "nsight": {"output": "existing.nsys-rep"},
            }
        }

        assert result == expected

    def test_empty_dictionaries(self):
        """Test merging with empty dictionaries."""
        default_options = {}
        extra_options = {"key": "value"}

        result = recursive_merge_options(default_options, extra_options)
        assert result == {"key": "value"}

        default_options = {"key": "value"}
        extra_options = {}

        result = recursive_merge_options(default_options, extra_options)
        assert result == {"key": "value"}

    def test_deep_copy_behavior(self):
        """Test that the function doesn't modify original dictionaries."""
        default_options = {"nested": {"key": "original"}}
        extra_options = {"nested": {"key": "modified"}}

        original_default = deepcopy(default_options)
        original_extra = deepcopy(extra_options)

        result = recursive_merge_options(default_options, extra_options)

        # Original dictionaries should be unchanged
        assert default_options == original_default
        assert extra_options == original_extra

        # Result should have the merged values
        assert result["nested"]["key"] == "modified"

    def test_list_handling(self):
        """Test that lists are replaced entirely, not merged."""
        default_options = {"list_key": [1, 2, 3]}
        extra_options = {"list_key": [4, 5]}

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "list_key": [4, 5]  # Completely replaced
        }

        assert result == expected

    def test_complex_nested_structure(self):
        """Test merging of complex nested structures."""
        default_options = {
            "worker_config": {
                "resources": {"num_cpus": 4, "num_gpus": 1},
                "runtime_env": {
                    "_nsight": {"profile": False},
                    "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"},
                },
            },
            "other_config": {"timeout": 300},
        }

        extra_options = {
            "worker_config": {
                "resources": {
                    "num_gpus": 2,  # Override
                    "memory": "16GB",  # Add
                },
                "runtime_env": {
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": "2,3",  # Override
                        "PYTHONPATH": "/custom",  # Add
                    }
                },
            },
            "new_config": {"enabled": True},
        }

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "worker_config": {
                "resources": {
                    "num_cpus": 4,
                    "num_gpus": 2,  # Overridden
                    "memory": "16GB",  # Added
                },
                "runtime_env": {
                    "_nsight": {  # _nsight stays as _nsight in nested structure
                        "profile": False
                    },
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": "2,3",  # Overridden
                        "PYTHONPATH": "/custom",  # Added
                    },
                },
            },
            "other_config": {"timeout": 300},
            "new_config": {  # Added
                "enabled": True
            },
        }

        assert result == expected

    def test_top_level_runtime_env_nsight_transformation(self):
        """Test _nsight transformation when runtime_env is at the top level."""
        default_options = {
            "runtime_env": {
                "_nsight": {"profile": True, "output": "profile.nsys-rep"},
                "env_vars": {"CUDA_VISIBLE_DEVICES": "0"},
            },
            "other_key": "value",
        }
        extra_options = {"runtime_env": {"env_vars": {"PYTHONPATH": "/custom/path"}}}

        result = recursive_merge_options(default_options, extra_options)

        expected = {
            "runtime_env": {
                "nsight": {  # _nsight transformed to nsight at top level
                    "profile": True,
                    "output": "profile.nsys-rep",
                },
                "env_vars": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/custom/path"},
            },
            "other_key": "value",
        }

        assert result == expected
        # Ensure _nsight is removed
        assert "_nsight" not in result["runtime_env"]
