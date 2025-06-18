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
"""Tests for the nsight file patching functionality."""

import os
from unittest.mock import Mock, mock_open, patch

from nemo_rl import _patch_nsight_file


class TestNsightPatching:
    """Test cases for the _patch_nsight_file function."""

    def test_no_patching_when_env_var_not_set(self):
        """Test that patching is skipped when NRL_NSYS_WORKER_PATTERNS is not set."""
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            with patch("nemo_rl.logging") as mock_logging:
                _patch_nsight_file()
                # Should not log anything since function returns early
                mock_logging.info.assert_not_called()
                mock_logging.warning.assert_not_called()

    def test_patching_when_env_var_set(self):
        """Test that patching proceeds when NRL_NSYS_WORKER_PATTERNS is set."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging") as mock_logging:
                # Mock the ray import inside the function
                mock_nsight = Mock()
                mock_nsight.__file__ = "/fake/path/nsight.py"

                original_content = (
                    'context.py_executable = " ".join(self.nsight_cmd) + " python"'
                )

                # Use patch with create=True to mock the import
                with patch("ray._private.runtime_env.nsight", mock_nsight, create=True):
                    with patch("builtins.open", mock_open(read_data=original_content)):
                        _patch_nsight_file()

                        # Should have attempted to log success
                        mock_logging.info.assert_called()

    def test_successful_patching(self):
        """Test successful patching of a file."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging") as mock_logging:
                # Mock the ray import
                mock_nsight = Mock()
                mock_nsight.__file__ = "/fake/path/nsight.py"

                original_content = (
                    'context.py_executable = " ".join(self.nsight_cmd) + " python"'
                )
                expected_content = 'context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"'

                # Mock file operations
                mock_file = mock_open(read_data=original_content)

                # Use patch with create=True to mock the import
                with patch("ray._private.runtime_env.nsight", mock_nsight, create=True):
                    with patch("builtins.open", mock_file):
                        _patch_nsight_file()

                        # Verify file was read and written
                        mock_file.assert_any_call("/fake/path/nsight.py", "r")
                        mock_file.assert_any_call("/fake/path/nsight.py", "w")

                        # Verify patched content was written
                        handle = mock_file()
                        handle.write.assert_called_with(expected_content)

                        mock_logging.info.assert_called_with(
                            "Successfully patched Ray nsight plugin at /fake/path/nsight.py"
                        )

    def test_already_patched_file(self):
        """Test that already patched files are detected and skipped."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging") as mock_logging:
                mock_nsight = Mock()
                mock_nsight.__file__ = "/fake/path/nsight.py"

                # Content already contains the patched line
                already_patched_content = 'context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"'

                mock_file = mock_open(read_data=already_patched_content)

                # Use patch with create=True to mock the import
                with patch("ray._private.runtime_env.nsight", mock_nsight, create=True):
                    with patch("builtins.open", mock_file):
                        _patch_nsight_file()

                        # Should only read, not write
                        mock_file.assert_called_once_with("/fake/path/nsight.py", "r")
                        handle = mock_file()
                        handle.write.assert_not_called()

                        mock_logging.info.assert_called_with(
                            "Ray nsight plugin already patched at /fake/path/nsight.py"
                        )

    def test_expected_line_not_found(self):
        """Test handling when expected line is not found in file."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging") as mock_logging:
                mock_nsight = Mock()
                mock_nsight.__file__ = "/fake/path/nsight.py"

                # Content doesn't contain expected line
                different_content = "some other content"

                mock_file = mock_open(read_data=different_content)

                # Use patch with create=True to mock the import
                with patch("ray._private.runtime_env.nsight", mock_nsight, create=True):
                    with patch("builtins.open", mock_file):
                        _patch_nsight_file()

                        # Should only read, not write
                        mock_file.assert_called_once_with("/fake/path/nsight.py", "r")
                        handle = mock_file()
                        handle.write.assert_not_called()

                        mock_logging.warning.assert_called_with(
                            "Expected line not found in /fake/path/nsight.py - Ray version may have changed"
                        )

    def test_import_error_handling(self):
        """Test graceful handling of ImportError."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging") as mock_logging:
                # Mock the import to raise ImportError
                with patch(
                    "builtins.__import__", side_effect=ImportError("Ray not found")
                ):
                    _patch_nsight_file()

                    # Should not crash and should not log anything (silent failure)
                    mock_logging.info.assert_not_called()
                    mock_logging.warning.assert_not_called()

    def test_file_not_found_error_handling(self):
        """Test graceful handling of FileNotFoundError."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging") as mock_logging:
                mock_nsight = Mock()
                mock_nsight.__file__ = "/nonexistent/path/nsight.py"

                # Use patch with create=True to mock the import
                with patch("ray._private.runtime_env.nsight", mock_nsight, create=True):
                    with patch(
                        "builtins.open", side_effect=FileNotFoundError("File not found")
                    ):
                        _patch_nsight_file()

                        # Should not crash and should not log anything (silent failure)
                        mock_logging.info.assert_not_called()
                        mock_logging.warning.assert_not_called()

    def test_permission_error_handling(self):
        """Test graceful handling of PermissionError."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging") as mock_logging:
                mock_nsight = Mock()
                mock_nsight.__file__ = "/fake/path/nsight.py"

                # Use patch with create=True to mock the import
                with patch("ray._private.runtime_env.nsight", mock_nsight, create=True):
                    with patch(
                        "builtins.open",
                        side_effect=PermissionError("Permission denied"),
                    ):
                        _patch_nsight_file()

                        # Should not crash and should not log anything (silent failure)
                        mock_logging.info.assert_not_called()
                        mock_logging.warning.assert_not_called()

    def test_line_replacement_accuracy(self):
        """Test that the exact line replacement is accurate."""
        with patch.dict(os.environ, {"NRL_NSYS_WORKER_PATTERNS": "test_pattern"}):
            with patch("nemo_rl.logging"):
                mock_nsight = Mock()
                mock_nsight.__file__ = "/fake/path/nsight.py"

                # Content with the target line embedded in other code
                original_content = """
def some_function():
    context.py_executable = " ".join(self.nsight_cmd) + " python"
    return context
"""

                expected_content = """
def some_function():
    context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"
    return context
"""

                mock_file = mock_open(read_data=original_content)

                # Use patch with create=True to mock the import
                with patch("ray._private.runtime_env.nsight", mock_nsight, create=True):
                    with patch("builtins.open", mock_file):
                        _patch_nsight_file()

                        handle = mock_file()
                        handle.write.assert_called_with(expected_content)
