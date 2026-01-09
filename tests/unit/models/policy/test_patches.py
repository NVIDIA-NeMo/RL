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

import os
import sys
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from nemo_rl.models.policy.workers.patches import (
    _get_transformer_engine_file,
    apply_torch_aten_alias_tensor_patch,
    apply_transformer_engine_patch,
)


class TestGetTransformerEngineFile:
    """Tests for _get_transformer_engine_file function."""

    def test_package_not_found(self):
        """Test that RuntimeError is raised when transformer_engine is not installed."""
        with patch(
            "nemo_rl.models.policy.workers.patches.find_spec", return_value=None
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _get_transformer_engine_file("pytorch/triton/permutation.py")

            assert "Transformer Engine package not found" in str(exc_info.value)
            assert "pytorch/triton/permutation.py" in str(exc_info.value)

    def test_package_no_submodule_locations(self):
        """Test that RuntimeError is raised when package has no submodule_search_locations."""
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = None

        with patch(
            "nemo_rl.models.policy.workers.patches.find_spec", return_value=mock_spec
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _get_transformer_engine_file("pytorch/triton/permutation.py")

            assert "Transformer Engine package not found" in str(exc_info.value)

    def test_package_empty_submodule_locations(self):
        """Test that RuntimeError is raised when submodule_search_locations is empty."""
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = []

        with patch(
            "nemo_rl.models.policy.workers.patches.find_spec", return_value=mock_spec
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _get_transformer_engine_file("pytorch/triton/permutation.py")

            assert "Transformer Engine package not found" in str(exc_info.value)

    def test_file_not_found(self):
        """Test that RuntimeError is raised when the target file doesn't exist."""
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ["/fake/path/to/transformer_engine"]

        with (
            patch(
                "nemo_rl.models.policy.workers.patches.find_spec",
                return_value=mock_spec,
            ),
            patch("os.path.exists", return_value=False),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _get_transformer_engine_file("pytorch/triton/permutation.py")

            assert "Failed to locate expected Transformer Engine file" in str(
                exc_info.value
            )
            assert "pytorch/triton/permutation.py" in str(exc_info.value)

    def test_successful_file_lookup(self):
        """Test successful file path resolution."""
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ["/fake/path/to/transformer_engine"]
        expected_path = os.path.join(
            "/fake/path/to/transformer_engine", "pytorch", "triton", "permutation.py"
        )

        with (
            patch(
                "nemo_rl.models.policy.workers.patches.find_spec",
                return_value=mock_spec,
            ),
            patch("os.path.exists", return_value=True),
        ):
            result = _get_transformer_engine_file("pytorch/triton/permutation.py")

            assert result == expected_path

    def test_path_construction_with_multiple_segments(self):
        """Test that paths with multiple segments are correctly constructed."""
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ["/base/dir"]

        with (
            patch(
                "nemo_rl.models.policy.workers.patches.find_spec",
                return_value=mock_spec,
            ),
            patch("os.path.exists", return_value=True),
        ):
            result = _get_transformer_engine_file("a/b/c/d.py")

            expected = os.path.join("/base/dir", "a", "b", "c", "d.py")
            assert result == expected


class TestApplyTransformerEnginePatch:
    """Tests for apply_transformer_engine_patch function."""

    UNPATCHED_CONTENT = """
import triton
from triton import language as core

@triton.jit
def some_kernel(x):
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    return x
"""

    ALREADY_PATCHED_CONTENT = """
import triton
from triton import language as core

get_int_dtype = core.get_int_dtype
get_int_dtype = triton.constexpr_function(get_int_dtype)

@triton.jit
def some_kernel(x):
    idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    return x
"""

    def test_patch_not_applied_when_already_patched(self, capsys):
        """Test that patch is not applied when file is already patched."""
        with (
            patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value="/fake/path/permutation.py",
            ),
            patch(
                "builtins.open",
                mock_open(read_data=self.ALREADY_PATCHED_CONTENT),
            ) as mock_file,
        ):
            apply_transformer_engine_patch()

            # Verify file was only opened for reading (not writing)
            mock_file.assert_called_once_with("/fake/path/permutation.py", "r")
            # No print about applying fix since already patched
            captured = capsys.readouterr()
            assert "Applying Triton fix" not in captured.out

    def test_patch_applied_when_needed(self, capsys):
        """Test that patch is correctly applied when file needs patching."""
        mock_file_handle = MagicMock()
        mock_file_handle.read.return_value = self.UNPATCHED_CONTENT
        mock_file_handle.__enter__ = MagicMock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = MagicMock(return_value=False)

        written_content = []

        def mock_write(content):
            written_content.append(content)

        mock_file_handle.write = mock_write

        call_count = [0]

        def mock_open_func(path, mode="r"):
            call_count[0] += 1
            if mode == "r":
                mock_file_handle.read.return_value = self.UNPATCHED_CONTENT
            return mock_file_handle

        with (
            patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value="/fake/path/permutation.py",
            ),
            patch("builtins.open", mock_open_func),
        ):
            apply_transformer_engine_patch()

            captured = capsys.readouterr()
            assert "Applying Triton fix to /fake/path/permutation.py" in captured.out
            assert "Successfully patched" in captured.out

            # Verify the content was modified
            assert len(written_content) > 0
            new_content = written_content[0]
            assert (
                "get_int_dtype = triton.constexpr_function(get_int_dtype)"
                in new_content
            )
            assert (
                "idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
                in new_content
            )

    def test_patch_handles_permission_error(self, capsys):
        """Test that permission errors when writing are handled gracefully."""
        with (
            patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value="/fake/path/permutation.py",
            ),
            patch("builtins.open") as mock_file,
        ):
            # First call (read) succeeds
            read_mock = MagicMock()
            read_mock.__enter__ = MagicMock(return_value=read_mock)
            read_mock.__exit__ = MagicMock(return_value=False)
            read_mock.read.return_value = self.UNPATCHED_CONTENT

            # Second call (write) fails with permission error
            write_mock = MagicMock()
            write_mock.__enter__ = MagicMock(
                side_effect=PermissionError("Permission denied")
            )

            mock_file.side_effect = [read_mock, write_mock]

            apply_transformer_engine_patch()

            captured = capsys.readouterr()
            # Should not crash, but print error message
            assert "Applying Triton fix" in captured.out

    def test_patch_handles_file_lookup_error(self, capsys):
        """Test that errors from _get_transformer_engine_file are handled."""
        with patch(
            "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
            side_effect=RuntimeError("Transformer Engine package not found"),
        ):
            # Should not raise, just print error
            apply_transformer_engine_patch()

            captured = capsys.readouterr()
            assert "Error checking/patching transformer_engine" in captured.out

    def test_module_reload_when_already_imported(self):
        """Test that the module is reloaded if already imported."""
        module_name = "transformer_engine.pytorch.triton.permutation"

        # Create a fake module to put in sys.modules
        fake_module = MagicMock()

        with (
            patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value="/fake/path/permutation.py",
            ),
            patch("builtins.open", mock_open(read_data=self.ALREADY_PATCHED_CONTENT)),
            patch.dict(sys.modules, {module_name: fake_module}),
            patch("importlib.reload") as mock_reload,
        ):
            apply_transformer_engine_patch()

            mock_reload.assert_called_once_with(fake_module)

    def test_no_reload_when_module_not_imported(self):
        """Test that no reload happens if module isn't imported."""
        module_name = "transformer_engine.pytorch.triton.permutation"

        # Ensure module is NOT in sys.modules
        modules_without_te = {k: v for k, v in sys.modules.items() if k != module_name}

        with (
            patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value="/fake/path/permutation.py",
            ),
            patch("builtins.open", mock_open(read_data=self.ALREADY_PATCHED_CONTENT)),
            patch.dict(sys.modules, modules_without_te, clear=True),
            patch("importlib.reload") as mock_reload,
        ):
            apply_transformer_engine_patch()

            mock_reload.assert_not_called()

    def test_patch_does_nothing_when_old_usage_not_found(self, capsys):
        """Test that patch does nothing when old_usage pattern is not in file."""
        content_without_old_usage = """
import triton
from triton import language as core

@triton.jit
def some_kernel(x):
    # Different usage pattern
    idtype = some_other_function()
    return x
"""
        with (
            patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value="/fake/path/permutation.py",
            ),
            patch(
                "builtins.open", mock_open(read_data=content_without_old_usage)
            ) as mock_file,
        ):
            apply_transformer_engine_patch()

            # Verify file was only opened for reading (not writing)
            mock_file.assert_called_once_with("/fake/path/permutation.py", "r")
            captured = capsys.readouterr()
            # Should print applying message but not success message since pattern not found
            assert "Applying Triton fix" in captured.out
            assert "Successfully patched" not in captured.out

    def test_patch_does_nothing_when_jit_anchor_not_found(self, capsys):
        """Test that patch does nothing when @triton.jit anchor is not found."""
        content_without_jit = """
import triton
from triton import language as core

def some_kernel(x):
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    return x
"""
        with (
            patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value="/fake/path/permutation.py",
            ),
            patch(
                "builtins.open", mock_open(read_data=content_without_jit)
            ) as mock_file,
        ):
            apply_transformer_engine_patch()

            # Verify file was only opened for reading (not writing)
            mock_file.assert_called_once_with("/fake/path/permutation.py", "r")
            captured = capsys.readouterr()
            assert "Applying Triton fix" in captured.out
            assert "Successfully patched" not in captured.out


class TestPatchIntegration:
    """Integration-style tests for the patch module."""

    def test_patch_with_real_temp_file(self, capsys):
        """Test patching with a real temporary file to verify file operations."""
        unpatched_content = """import triton
from triton import language as core

@triton.jit
def permutation_kernel(x):
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    return x
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(unpatched_content)
            tmp_path = tmp_file.name

        try:
            with patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value=tmp_path,
            ):
                apply_transformer_engine_patch()

                # Read the patched file
                with open(tmp_path, "r") as f:
                    patched_content = f.read()

                # Verify the patch was applied
                assert (
                    "get_int_dtype = triton.constexpr_function(get_int_dtype)"
                    in patched_content
                )
                assert (
                    "idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
                    in patched_content
                )
                # Verify old pattern is gone
                assert (
                    "idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
                    not in patched_content
                )

                captured = capsys.readouterr()
                assert "Successfully patched" in captured.out
        finally:
            os.unlink(tmp_path)

    def test_patch_idempotent(self, capsys):
        """Test that applying patch twice doesn't change already patched content."""
        unpatched_content = """import triton
from triton import language as core

@triton.jit
def permutation_kernel(x):
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    return x
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(unpatched_content)
            tmp_path = tmp_file.name

        try:
            with patch(
                "nemo_rl.models.policy.workers.patches._get_transformer_engine_file",
                return_value=tmp_path,
            ):
                # Apply patch first time
                apply_transformer_engine_patch()

                with open(tmp_path, "r") as f:
                    first_patched = f.read()

                # Apply patch second time
                apply_transformer_engine_patch()

                with open(tmp_path, "r") as f:
                    second_patched = f.read()

                # Content should be identical
                assert first_patched == second_patched

                captured = capsys.readouterr()
                # First application should succeed, second should skip
                assert captured.out.count("Successfully patched") == 1
        finally:
            os.unlink(tmp_path)


def build_sharded_3d(rank: int, world_size: int):
    """
    Build tensor, DTensor, and test the sharding rule for torch.ops.aten.alias.default.
    """
    from torch.distributed.tensor import DeviceMesh, DTensor

    mesh = DeviceMesh("cuda", list(range(world_size)))
    global_shape = (4, 2, 4)
    tensor = torch.arange(
        torch.tensor(global_shape).prod(), dtype=torch.float32
    ).reshape(global_shape)

    try:
        dtensor = DTensor.from_local(
            tensor,
            device_mesh=mesh,
        )
        alias_dtensor = torch.ops.aten.alias.default(dtensor)
        assert False, (
            "Torch==2.9 should raise 'NotImplementedError: Operator aten.alias.default does not have a sharding strategy registered', but it didn't."
            "You can:\n "
            "1. Check is you bump your torch version which contain the fix https://github.com/pytorch/pytorch/pull/166867\n"
            "2. If yes, remove patch apply_torch_aten_alias_tensor_patch in nemo_rl/models/policy/workers/patches.py \n"
            "3. Remove the patching call in nemo_rl/models/policy/workers/dtensor_policy_worker.py and nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py \n"
            "4. Delete this test"
        )
    except NotImplementedError:
        apply_torch_aten_alias_tensor_patch()
        alias_dtensor = torch.ops.aten.alias.default(dtensor)
        assert alias_dtensor.shape == global_shape
        assert torch.allclose(alias_dtensor.to_local().cpu(), tensor.cpu())


@pytest.mark.cuda
def test_aten_alias_sharding_still_missing_upstream(
    distributed_test_runner, world_size=2
):
    """Test that sharding rule for aten.alias.default is still missing upstream."""
    distributed_test_runner(build_sharded_3d, world_size=world_size)
