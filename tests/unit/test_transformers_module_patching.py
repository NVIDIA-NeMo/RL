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
"""Tests for the transformers module directory patching functionality."""

import os
import tempfile
from unittest.mock import patch

from nemo_rl import patch_transformers_module_dir


class TestPatchTransformersModuleDir:
    """Test cases for the patch_transformers_module_dir function."""

    def test_no_patching_when_hf_home_not_set(self):
        """Test that patching is skipped when HF_HOME is not set."""
        env_vars = {"OTHER_VAR": "value"}

        # Ensure HF_HOME is not set
        with patch.dict(os.environ, {}, clear=True):
            result = patch_transformers_module_dir(env_vars)

        # Should return the same dict without modifications
        assert result == {"OTHER_VAR": "value"}
        assert "PYTHONPATH" not in result

    def test_patching_adds_pythonpath_when_not_present(self):
        """Test that PYTHONPATH is added when it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the modules directory
            modules_dir = os.path.join(tmpdir, "modules")
            os.makedirs(modules_dir)

            env_vars = {"OTHER_VAR": "value"}

            with patch.dict(os.environ, {"HF_HOME": tmpdir}):
                result = patch_transformers_module_dir(env_vars)

            # Should add PYTHONPATH with the modules directory
            assert "PYTHONPATH" in result
            assert result["PYTHONPATH"] == modules_dir
            assert result["OTHER_VAR"] == "value"

    def test_patching_prepends_to_existing_pythonpath(self):
        """Test that modules directory is prepended to existing PYTHONPATH."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the modules directory
            modules_dir = os.path.join(tmpdir, "modules")
            os.makedirs(modules_dir)

            existing_path = "/some/other/path"
            env_vars = {"PYTHONPATH": existing_path}

            with patch.dict(os.environ, {"HF_HOME": tmpdir}):
                result = patch_transformers_module_dir(env_vars)

            # Should prepend modules_dir to existing PYTHONPATH
            assert result["PYTHONPATH"] == f"{modules_dir}:{existing_path}"

    def test_patching_returns_early_when_modules_dir_not_exist(self):
        """Test that function returns unchanged env_vars when modules directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create the modules directory
            env_vars = {"OTHER_VAR": "value"}

            with patch.dict(os.environ, {"HF_HOME": tmpdir}):
                result = patch_transformers_module_dir(env_vars)

            # Should return unchanged env_vars
            assert result == {"OTHER_VAR": "value"}
            assert "PYTHONPATH" not in result

    def test_patching_with_nested_hf_home(self):
        """Test patching works with nested HF_HOME path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested HF_HOME structure
            hf_home = os.path.join(tmpdir, "nested", "huggingface")
            modules_dir = os.path.join(hf_home, "modules")
            os.makedirs(modules_dir)

            env_vars = {}

            with patch.dict(os.environ, {"HF_HOME": hf_home}):
                result = patch_transformers_module_dir(env_vars)

            assert result["PYTHONPATH"] == modules_dir

    def test_patching_does_not_modify_original_dict(self):
        """Test that the function modifies the dictionary in place and returns it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the modules directory
            modules_dir = os.path.join(tmpdir, "modules")
            os.makedirs(modules_dir)

            env_vars = {"OTHER_VAR": "value"}
            original_id = id(env_vars)

            with patch.dict(os.environ, {"HF_HOME": tmpdir}):
                result = patch_transformers_module_dir(env_vars)

            # Should return the same object (modified in place)
            assert id(result) == original_id
            assert "PYTHONPATH" in result
            assert result["PYTHONPATH"] == modules_dir

    def test_multiple_calls_with_same_env_vars(self):
        """Test that calling the function multiple times with existing PYTHONPATH works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the modules directory
            modules_dir = os.path.join(tmpdir, "modules")
            os.makedirs(modules_dir)

            env_vars = {}

            with patch.dict(os.environ, {"HF_HOME": tmpdir}):
                # First call
                result1 = patch_transformers_module_dir(env_vars)
                assert result1["PYTHONPATH"] == modules_dir

                # Second call with the already modified env_vars
                result2 = patch_transformers_module_dir(result1)
                # Should prepend again
                assert result2["PYTHONPATH"] == f"{modules_dir}:{modules_dir}"

    def test_empty_env_vars_dict(self):
        """Test that function works with an empty env_vars dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the modules directory
            modules_dir = os.path.join(tmpdir, "modules")
            os.makedirs(modules_dir)

            env_vars = {}

            with patch.dict(os.environ, {"HF_HOME": tmpdir}):
                result = patch_transformers_module_dir(env_vars)

            assert result == {"PYTHONPATH": modules_dir}

    def test_hf_home_with_trailing_slash(self):
        """Test that function handles HF_HOME with trailing slash correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the modules directory
            modules_dir = os.path.join(tmpdir, "modules")
            os.makedirs(modules_dir)

            env_vars = {}

            # Add trailing slash to HF_HOME
            hf_home_with_slash = tmpdir + "/"

            with patch.dict(os.environ, {"HF_HOME": hf_home_with_slash}):
                result = patch_transformers_module_dir(env_vars)

            # os.path.join should handle the trailing slash correctly
            expected_path = os.path.join(hf_home_with_slash, "modules")
            assert result["PYTHONPATH"] == expected_path

    def test_preserves_other_env_vars(self):
        """Test that function preserves other environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the modules directory
            modules_dir = os.path.join(tmpdir, "modules")
            os.makedirs(modules_dir)

            env_vars = {
                "VAR1": "value1",
                "VAR2": "value2",
                "VAR3": "value3",
            }

            with patch.dict(os.environ, {"HF_HOME": tmpdir}):
                result = patch_transformers_module_dir(env_vars)

            # All original vars should be preserved
            assert result["VAR1"] == "value1"
            assert result["VAR2"] == "value2"
            assert result["VAR3"] == "value3"
            assert result["PYTHONPATH"] == modules_dir
