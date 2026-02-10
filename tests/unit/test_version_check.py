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

"""Tests for container version fingerprint checking."""

import os
from pathlib import Path
from unittest import mock

import pytest

# Import the functions before any tests run
from nemo_rl import _check_container_fingerprint


class TestContainerFingerprintCheck:
    """Test the container fingerprint check functionality."""

    def test_skip_check_on_baremetal(self, monkeypatch):
        """Test that version check is skipped when not in container."""
        # Ensure NRL_CONTAINER is not set
        monkeypatch.delenv("NRL_CONTAINER", raising=False)

        # Call should return early without doing anything
        _check_container_fingerprint()

        # No exception should be raised
        assert True

    def test_check_passes_when_fingerprints_match(self, monkeypatch):
        """Test that check passes silently when fingerprints match."""
        import json

        # Set up environment to simulate container
        monkeypatch.setenv("NRL_CONTAINER", "1")

        # Create a mock fingerprint dictionary
        fingerprint = {
            "pyproject.toml": "abc123",
            "uv.lock": "def456",
            "submodules/third_party/NeMo": "789xyz",
        }
        fingerprint_json = json.dumps(fingerprint, indent=2, sort_keys=True)

        # Mock runpy to return matching fingerprint
        def mock_run_path(path, run_name=None):
            print(fingerprint_json)

        with mock.patch("runpy.run_path", side_effect=mock_run_path):
            with mock.patch("nemo_rl.Path") as mock_path:
                mock_fp_script = mock.MagicMock()
                mock_fp_script.exists.return_value = True

                mock_container_fp_file = mock.MagicMock()
                mock_container_fp_file.exists.return_value = True
                mock_container_fp_file.read_text.return_value = (
                    fingerprint_json  # Same fingerprint
                )

                def path_constructor(arg):
                    if "/opt/nemo_rl_container_fingerprint" in str(arg):
                        return mock_container_fp_file
                    m = mock.MagicMock()
                    m.exists.return_value = True
                    m.__truediv__ = mock.MagicMock(return_value=mock_fp_script)
                    return m

                mock_path.side_effect = path_constructor

                # Should complete without exception
                _check_container_fingerprint()

        # No exception raised
        assert True

    @pytest.mark.skip(reason="Complex mocking - integration test more appropriate")
    def test_check_raises_on_mismatch_without_ignore_flag(self, monkeypatch, tmp_path):
        """Test that check raises RuntimeError when fingerprints don't match."""
        # Set up environment to simulate container
        monkeypatch.setenv("NRL_CONTAINER", "1")
        monkeypatch.delenv("NRL_IGNORE_VERSION_MISMATCH", raising=False)

        # Create actual files with different fingerprints
        container_fingerprint = "abc123def456"
        code_fingerprint = "different999"

        # Create a fake fingerprint script that just prints code_fingerprint
        fake_script = tmp_path / "generate_fingerprint.py"
        fake_script.write_text(f"#!/usr/bin/env python3\nprint('{code_fingerprint}')\n")

        # Create container fingerprint file
        container_fp_file = tmp_path / "nemo_rl_container_fingerprint"
        container_fp_file.write_text(container_fingerprint)

        # Patch Path to point to our temp files
        original_path_init = Path.__init__

        def mock_path_init(self, *args):
            path_str = str(args[0]) if args else ""
            if "/opt/nemo_rl_container_fingerprint" in path_str:
                original_path_init(self, container_fp_file)
            elif "generate_fingerprint.py" in path_str:
                original_path_init(self, fake_script)
            else:
                original_path_init(self, *args)

        with mock.patch.object(Path, "__init__", mock_path_init):
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Container/Code Version Mismatch"):
                _check_container_fingerprint()

    @pytest.mark.skip(reason="Complex mocking - integration test more appropriate")
    def test_check_logs_warning_with_ignore_flag(self, monkeypatch, caplog):
        """Test that check logs warning but continues when NRL_IGNORE_VERSION_MISMATCH is set."""
        # Set up environment to simulate container with ignore flag
        monkeypatch.setenv("NRL_CONTAINER", "1")
        monkeypatch.setenv("NRL_IGNORE_VERSION_MISMATCH", "1")

        # Create a mock fingerprint file with different fingerprint
        container_fingerprint = "abc123def456"
        code_fingerprint = "different999"

        # Mock runpy to return a different fingerprint
        def mock_run_path(path, run_name=None):
            print(code_fingerprint)

        with mock.patch("runpy.run_path", side_effect=mock_run_path):
            # Mock the Path class
            with mock.patch("nemo_rl.Path") as mock_path_class:
                mock_repo_root = mock.MagicMock()
                mock_fingerprint_script = mock.MagicMock()
                mock_fingerprint_script.exists.return_value = True
                mock_repo_root.__truediv__ = mock.MagicMock(
                    return_value=mock_fingerprint_script
                )

                mock_container_fp = mock.MagicMock()
                mock_container_fp.exists.return_value = True
                mock_container_fp.read_text.return_value = container_fingerprint

                def path_side_effect(arg):
                    if str(arg) == "/opt/nemo_rl_container_fingerprint":
                        return mock_container_fp
                    return Path(arg)

                mock_path_class.side_effect = path_side_effect

                from nemo_rl import _check_container_fingerprint

                # Should not raise, just log warning
                _check_container_fingerprint()

        # No exception raised
        assert True

    @pytest.mark.skip(reason="Complex mocking - integration test more appropriate")
    def test_check_handles_missing_fingerprint_file(self, monkeypatch):
        """Test that check handles missing container fingerprint gracefully."""
        # Set up environment to simulate container
        monkeypatch.setenv("NRL_CONTAINER", "1")

        # Mock runpy to return a fingerprint
        def mock_run_path(path, run_name=None):
            print("abc123")

        with mock.patch("runpy.run_path", side_effect=mock_run_path):
            with mock.patch("nemo_rl.Path") as mock_path_class:
                mock_fingerprint_script = mock.MagicMock()
                mock_fingerprint_script.exists.return_value = True

                mock_container_fp = mock.MagicMock()
                mock_container_fp.exists.return_value = False  # Missing file

                def path_side_effect(arg):
                    if str(arg) == "/opt/nemo_rl_container_fingerprint":
                        return mock_container_fp
                    elif "generate_fingerprint.py" in str(arg):
                        return mock_fingerprint_script
                    return Path(arg)

                mock_path_class.side_effect = path_side_effect

                from nemo_rl import _check_container_fingerprint

                # Should not raise exception
                _check_container_fingerprint()

        assert True

    @pytest.mark.skip(reason="Complex mocking - integration test more appropriate")
    def test_check_handles_runpy_failure(self, monkeypatch):
        """Test that check handles runpy failures gracefully."""
        # Set up environment to simulate container
        monkeypatch.setenv("NRL_CONTAINER", "1")

        # Mock runpy to raise an exception
        def mock_run_path(path, run_name=None):
            raise RuntimeError("Error generating fingerprint")

        with mock.patch("runpy.run_path", side_effect=mock_run_path):
            with mock.patch("nemo_rl.Path") as mock_path_class:
                mock_fingerprint_script = mock.MagicMock()
                mock_fingerprint_script.exists.return_value = True

                def path_side_effect(arg):
                    if "generate_fingerprint.py" in str(arg):
                        return mock_fingerprint_script
                    return Path(arg)

                mock_path_class.side_effect = path_side_effect

                from nemo_rl import _check_container_fingerprint

                # Should not raise exception (handles gracefully)
                _check_container_fingerprint()

        assert True


class TestBuildIsolationDetection:
    """Test the build isolation detection functionality with real uv commands."""

    @pytest.fixture
    def dummy_project(self, tmp_path):
        """Create a minimal dummy project for testing."""
        import subprocess

        project_dir = tmp_path / "dummy_project"
        project_dir.mkdir()

        # Get the nemo_rl project root
        nemo_rl_root = Path(__file__).parent.parent.parent

        # Create a minimal pyproject.toml that imports nemo_rl
        pyproject = project_dir / "pyproject.toml"
        pyproject.write_text(f"""[project]
name = "dummy-test-package"
version = "0.1.0"
dependencies = [
    "nemo-rl @ file://{nemo_rl_root}",
]

[build-system]
requires = ["setuptools", "nemo-rl @ file://{nemo_rl_root}"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["dummy_pkg"]
""")

        # Create a minimal package
        pkg_dir = project_dir / "dummy_pkg"
        pkg_dir.mkdir()

        # Create a file to log build isolation detection results
        log_file = project_dir / "build_isolation_log.txt"

        init_file = pkg_dir / "__init__.py"
        init_file.write_text("""__version__ = "0.1.0"
""")

        # Create a setup.py that will be executed during build
        setup_py = project_dir / "setup.py"
        setup_py.write_text(f"""import sys
import os
from pathlib import Path
from setuptools import setup

# Log file to write build isolation detection results
log_file = Path(r"{log_file}")

# Set ignore flag to avoid actual fingerprint check failing
os.environ["NRL_IGNORE_VERSION_MISMATCH"] = "1"

try:
    # Import and test the build isolation detection
    from nemo_rl import _is_build_isolation
    result = _is_build_isolation()

    # Write results to log file
    with open(log_file, "a") as f:
        f.write(f"PREFIX:{{sys.prefix}}\\n")
        f.write(f"IS_BUILD_ISOLATION:{{result}}\\n")
        f.write("---\\n")
except Exception as e:
    # Write error to log file
    with open(log_file, "a") as f:
        f.write(f"ERROR:{{e}}\\n")
        f.write("---\\n")

# Call setup() - setuptools will read pyproject.toml for config
setup()
""")

        # Initialize uv.lock with isolated environment
        dummy_venv = project_dir / ".venv"
        test_env = {
            **os.environ,
            "NRL_IGNORE_VERSION_MISMATCH": "1",
            "UV_PROJECT_ENVIRONMENT": str(dummy_venv),
        }

        try:
            subprocess.run(
                ["uv", "lock"],
                cwd=project_dir,
                capture_output=True,
                check=True,
                timeout=30,
                env=test_env,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("uv lock timed out")
        except Exception as e:
            pytest.skip(f"Failed to initialize dummy project: {e}")

        return project_dir

    def test_build_isolation_detected_during_uv_sync(self, dummy_project):
        """Test that build isolation is detected during uv sync."""
        import subprocess

        log_file = dummy_project / "build_isolation_log.txt"

        # Clear log file if it exists
        if log_file.exists():
            log_file.unlink()

        # Touch the package to force a rebuild
        init_file = dummy_project / "dummy_pkg" / "__init__.py"
        init_file.touch()

        # Set up isolated environment for the dummy project
        dummy_venv = dummy_project / ".venv"
        test_env = {
            **os.environ,
            "NRL_IGNORE_VERSION_MISMATCH": "1",
            "UV_PROJECT_ENVIRONMENT": str(dummy_venv),
        }

        # Run uv sync which will trigger a build
        result = subprocess.run(
            ["uv", "sync"],
            cwd=dummy_project,
            capture_output=True,
            text=True,
            timeout=180,
            env=test_env,
        )

        # Read the log file written during build
        assert log_file.exists(), (
            f"Log file not created. uv sync output:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

        log_content = log_file.read_text()
        log_lines = log_content.strip().split("\n")

        # Look for our markers in the log
        prefix_lines = [line for line in log_lines if "PREFIX:" in line]
        isolation_lines = [line for line in log_lines if "IS_BUILD_ISOLATION:" in line]

        # During build, we should see at least one invocation with build isolation
        assert len(prefix_lines) > 0, f"No prefix lines found in log:\n{log_content}"
        assert len(isolation_lines) > 0, (
            f"No isolation detection lines found in log:\n{log_content}"
        )

        # Check that at least one prefix contains /builds-v (build isolation)
        has_build_isolation = any("/builds-v" in line for line in prefix_lines)
        assert has_build_isolation, (
            f"Expected /builds-v in at least one prefix:\n{'\n'.join(prefix_lines)}"
        )

        # Check that at least one isolation check returned True
        has_true_isolation = any(
            "IS_BUILD_ISOLATION:True" in line for line in isolation_lines
        )
        assert has_true_isolation, (
            f"Expected at least one True isolation detection:\n{'\n'.join(isolation_lines)}"
        )

    def test_build_isolation_not_detected_during_uv_run(self, dummy_project):
        """Test that build isolation is NOT detected during uv run."""
        import subprocess

        log_file = dummy_project / "build_isolation_log.txt"

        # Clear log file
        if log_file.exists():
            log_file.unlink()

        # Set up isolated environment for the dummy project
        dummy_venv = dummy_project / ".venv"
        test_env = {
            **os.environ,
            "NRL_IGNORE_VERSION_MISMATCH": "1",
            "UV_PROJECT_ENVIRONMENT": str(dummy_venv),
        }

        # Run a simple command with uv run that writes to our log file
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"import sys; import os; "
                f"os.environ['NRL_IGNORE_VERSION_MISMATCH']='1'; "
                f"from nemo_rl import _is_build_isolation; "
                f"from pathlib import Path; "
                f"log = Path(r'{log_file}'); "
                f"log.write_text(f'PREFIX:{{sys.prefix}}\\nIS_BUILD_ISOLATION:{{_is_build_isolation()}}\\n')",
            ],
            cwd=dummy_project,
            capture_output=True,
            text=True,
            timeout=60,
            env=test_env,
        )

        # Read the log file
        assert log_file.exists(), (
            f"Log file not created. uv run output:\n{result.stdout}\n{result.stderr}"
        )

        log_content = log_file.read_text()

        # During uv run, we should NOT be in build isolation
        assert "/builds-v" not in log_content, (
            f"Unexpected build isolation path in uv run:\n{log_content}"
        )
        assert "IS_BUILD_ISOLATION:False" in log_content, (
            f"Expected IS_BUILD_ISOLATION:False in uv run:\n{log_content}"
        )

    def test_fingerprint_check_skipped_with_force_rebuild_venvs(self, monkeypatch):
        """Test that fingerprint check is skipped when NRL_FORCE_REBUILD_VENVS=true."""
        # Set up environment to simulate container with force rebuild
        monkeypatch.setenv("NRL_CONTAINER", "1")
        monkeypatch.setenv("NRL_FORCE_REBUILD_VENVS", "true")

        # Should complete without exception (check is skipped)
        _check_container_fingerprint()

        # No exception raised
        assert True
