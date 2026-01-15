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
"""Tests for checkpoint abstraction layer."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from nemo_rl.infra.checkpointing import (
    CheckpointError,
    CheckpointFormat,
    CheckpointManager,
    CheckpointMetadata,
    PyTorchBackend,
)


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""

    def test_create_metadata(self):
        """Test creating checkpoint metadata."""
        metadata = CheckpointMetadata(
            format=CheckpointFormat.PYTORCH,
            version="1.0",
            step=100,
            model_name="test-model",
        )
        assert metadata.format == CheckpointFormat.PYTORCH
        assert metadata.step == 100
        assert metadata.model_name == "test-model"

    def test_default_values(self):
        """Test default metadata values."""
        metadata = CheckpointMetadata(format=CheckpointFormat.SAFETENSORS)
        assert metadata.version == "1.0"
        assert metadata.step is None
        assert metadata.extra == {}


class TestPyTorchBackend:
    """Tests for PyTorchBackend."""

    def test_save_and_load(self):
        """Test saving and loading a checkpoint."""
        backend = PyTorchBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Create test state dict
            state_dict = {
                "weight": torch.randn(10, 10),
                "bias": torch.randn(10),
            }

            # Save
            backend.save(path, state_dict)
            assert path.exists()

            # Load
            loaded = backend.load(path)
            assert "weight" in loaded
            assert "bias" in loaded
            assert torch.equal(loaded["weight"], state_dict["weight"])

    def test_save_with_metadata(self):
        """Test saving with metadata."""
        backend = PyTorchBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            state_dict = {"weight": torch.randn(5, 5)}
            metadata = CheckpointMetadata(
                format=CheckpointFormat.PYTORCH,
                step=50,
            )

            backend.save(path, state_dict, metadata)
            assert path.exists()

    def test_can_load_pt_file(self):
        """Test can_load for .pt files."""
        backend = PyTorchBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = Path(tmpdir) / "model.pt"
            pt_path.touch()

            pth_path = Path(tmpdir) / "model.pth"
            pth_path.touch()

            assert backend.can_load(pt_path) is True
            assert backend.can_load(pth_path) is True

    def test_cannot_load_directory(self):
        """Test can_load returns False for directories."""
        backend = PyTorchBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            assert backend.can_load(Path(tmpdir)) is False


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_init(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                format=CheckpointFormat.PYTORCH,
                keep_top_k=3,
            )
            assert manager.checkpoint_dir == Path(tmpdir)
            assert manager.format == CheckpointFormat.PYTORCH
            assert manager.keep_top_k == 3

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            state_dict = {"weight": torch.randn(10, 10)}
            path = manager.save(state_dict, step=100)

            assert path.exists()
            assert (path / "training_info.json").exists()

    def test_save_with_metrics(self):
        """Test saving a checkpoint with metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            state_dict = {"weight": torch.randn(10, 10)}
            metrics = {"loss": 0.5, "reward": 1.2}
            path = manager.save(state_dict, step=100, metrics=metrics)

            # Check training info
            info_path = path / "training_info.json"
            with open(info_path) as f:
                info = json.load(f)
            assert info["metrics"]["loss"] == 0.5
            assert info["metrics"]["reward"] == 1.2

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                format=CheckpointFormat.PYTORCH,
            )

            # Save
            original = {"weight": torch.randn(10, 10)}
            path = manager.save(original, step=100)

            # Load
            loaded = manager.load(path)
            assert "weight" in loaded
            assert torch.equal(loaded["weight"], original["weight"])

    def test_get_latest_checkpoint(self):
        """Test getting the latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            # Save multiple checkpoints
            manager.save({"step": 1}, step=10)
            manager.save({"step": 2}, step=20)
            manager.save({"step": 3}, step=30)

            latest = manager.get_latest_checkpoint_path()
            assert latest is not None
            assert "step_30" in latest

    def test_get_best_checkpoint_no_metric(self):
        """Test getting best checkpoint without metric (returns latest)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                metric_name=None,
            )

            manager.save({"step": 1}, step=10)
            manager.save({"step": 2}, step=20)

            best = manager.get_best_checkpoint_path()
            assert best is not None
            assert "step_20" in best

    def test_get_best_checkpoint_with_metric(self):
        """Test getting best checkpoint with metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                metric_name="reward",
                higher_is_better=True,
                keep_top_k=None,  # Keep all for test
            )

            manager.save({"step": 1}, step=10, metrics={"reward": 0.5})
            manager.save({"step": 2}, step=20, metrics={"reward": 0.8})
            manager.save({"step": 3}, step=30, metrics={"reward": 0.3})

            best = manager.get_best_checkpoint_path()
            assert best is not None
            assert "step_20" in best  # Highest reward

    def test_keep_top_k(self):
        """Test that only top-k checkpoints are kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                keep_top_k=2,
                metric_name=None,
            )

            # Save more than top-k checkpoints
            manager.save({"step": 1}, step=10)
            manager.save({"step": 2}, step=20)
            manager.save({"step": 3}, step=30)
            manager.save({"step": 4}, step=40)

            # Check that only top-k remain
            checkpoints = list(Path(tmpdir).glob("step_*"))
            assert len(checkpoints) == 2

            # Should have the latest ones
            steps = sorted([int(cp.name.split("_")[1]) for cp in checkpoints])
            assert steps == [30, 40]

    def test_checkpoint_exists(self):
        """Test checking if checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            assert manager.checkpoint_exists(100) is False
            manager.save({"data": torch.randn(5)}, step=100)
            assert manager.checkpoint_exists(100) is True

    def test_auto_format_detection(self):
        """Test automatic format detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                format=CheckpointFormat.PYTORCH,
            )

            # Save in PyTorch format
            state_dict = {"weight": torch.randn(5, 5)}
            path = manager.save(state_dict, step=100)

            # Load with auto-detection
            loaded = manager.load(path, format=CheckpointFormat.AUTO)
            assert "weight" in loaded

    def test_load_best_when_path_none(self):
        """Test loading best checkpoint when path is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            manager.save({"v": 1}, step=10, metrics={"loss": 0.5})

            # Load with path=None should load best
            loaded = manager.load(path=None)
            assert loaded is not None

    def test_load_raises_when_no_checkpoints(self):
        """Test that load raises error when no checkpoints exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            with pytest.raises(CheckpointError):
                manager.load(path=None)


class TestCheckpointFormat:
    """Tests for CheckpointFormat enum."""

    def test_format_values(self):
        """Test format enum values."""
        assert CheckpointFormat.PYTORCH.value == "pytorch"
        assert CheckpointFormat.SAFETENSORS.value == "safetensors"
        assert CheckpointFormat.HUGGINGFACE.value == "huggingface"
        assert CheckpointFormat.AUTO.value == "auto"
