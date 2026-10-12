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
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml

from nemo_rl.utils.checkpoint import CheckpointManager


@pytest.fixture
def checkpoint_dir(tmp_path):
    return tmp_path.resolve() / "checkpoints"


@pytest.fixture
def checkpoint_config(checkpoint_dir):
    return {
        "enabled": True,
        "checkpoint_dir": checkpoint_dir,
        "metric_name": "loss",
        "higher_is_better": False,
        "save_period": 1,
        "keep_top_k": 3,
    }


@pytest.fixture
def checkpoint_manager(checkpoint_config):
    return CheckpointManager(checkpoint_config)


def test_init_tmp_checkpoint(checkpoint_manager, checkpoint_dir):
    # Test creating a new checkpoint
    step = 1
    training_info = {"loss": 0.5, "tensor": torch.tensor(0.5), "numpy": np.array(0.5)}
    run_config = {"model": "test"}

    save_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info, run_config)

    # Check if directory was created
    assert save_dir.exists()
    assert save_dir.name.startswith("tmp_step_")

    # Check if training metadata was saved correctly
    with open(save_dir / "training_info.json", "r") as f:
        saved_metadata = json.load(f)
        assert saved_metadata["loss"] == 0.5
        assert isinstance(saved_metadata["tensor"], (int, float))
        assert isinstance(saved_metadata["numpy"], (int, float))

    # Check if config was saved
    with open(save_dir / "config.yaml", "r") as f:
        saved_config = yaml.safe_load(f)
        assert saved_config == run_config


def test_finalize_checkpoint(checkpoint_manager, checkpoint_dir):
    # Create a temporary checkpoint
    step = 1
    training_info = {"loss": 0.5}
    tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)

    # Complete the checkpoint
    checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Check if temporary directory was renamed correctly
    assert not tmp_dir.exists()
    assert (checkpoint_dir / f"step_{step}").exists()


def test_remove_old_checkpoints(checkpoint_manager, checkpoint_dir):
    # Create multiple checkpoints with different loss values
    steps = [1, 2, 3, 4, 5, 6]
    losses = [0.5, 0.3, 0.7, 0.2, 0.4, 0.8]

    for step, loss in zip(steps, losses):
        training_info = {"loss": loss}
        tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
        checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Check if only top-k checkpoints are kept
    remaining_dirs = list(checkpoint_dir.glob("step_*"))
    assert (
        len(remaining_dirs) == checkpoint_manager.keep_top_k + 1
    )  # +1 because we exclude the latest

    # Verify the remaining checkpoints are the ones with lowest loss
    remaining_losses = []
    for dir_path in remaining_dirs:
        with open(dir_path / "training_info.json", "r") as f:
            metadata = json.load(f)
            remaining_losses.append(metadata["loss"])

    assert sorted(remaining_losses) == sorted(losses)[
        : checkpoint_manager.keep_top_k
    ] + [0.8]  # exclude latest


def test_remove_old_checkpoints_topk_bias_recent_if_equal(
    checkpoint_manager, checkpoint_dir
):
    # Create multiple checkpoints with the same loss value
    # Create multiple checkpoints with the same loss value
    steps = [1, 2, 3, 4, 10, 12]
    losses = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # All checkpoints have the same loss

    for step, loss in zip(steps, losses):
        training_info = {"loss": loss}
        tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
        checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Check if only top-k checkpoints are kept
    remaining_dirs = list(checkpoint_dir.glob("step_*"))
    assert (
        len(remaining_dirs) == checkpoint_manager.keep_top_k
    )  # +1 because we exclude the latest

    # When all losses are equal, the most recent checkpoints should be kept
    # (excluding the latest which is always kept)
    remaining_steps = []
    for dir_path in remaining_dirs:
        step_num = int(dir_path.name.split("_")[1])
        remaining_steps.append(step_num)

    # Should keep the most recent checkpoints (highest step numbers)
    expected_steps = sorted(steps)[-checkpoint_manager.keep_top_k :]
    assert sorted(remaining_steps) == sorted(expected_steps)


def test_remove_old_checkpoints_topk_some_missing_val_metric(
    checkpoint_manager, checkpoint_dir
):
    # Create checkpoints where some have validation metrics and others don't
    steps = [1, 2, 3, 4, 10, 11, 12]
    # Some checkpoints have loss metrics, others don't have any validation metrics
    training_infos = [
        {"loss": 0.5},  # step 1 - has loss
        {"loss": 0.3},  # step 2 - has loss
        {"other_metric": 0.8},  # step 3 - missing loss metric
        {"loss": 0.2},  # step 4 - has loss
        {},  # step 10 - missing loss metric
        {"loss": 1.0},  # has loss but not in top-k
        {},  # step 12 - missing loss (latest)
    ]

    for step, training_info in zip(steps, training_infos):
        tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
        checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Check if only top-k checkpoints are kept
    remaining_dirs = list(checkpoint_dir.glob("step_*"))
    assert (
        len(remaining_dirs) == checkpoint_manager.keep_top_k + 1
    )  # +1 because we exclude the latest

    # Checkpoints with missing validation metrics should be treated as having the worst possible value
    # Since higher_is_better=False, missing metrics get float("inf") which is worst
    # So checkpoints with actual loss values should be preferred over those without
    remaining_steps = []
    for dir_path in remaining_dirs:
        step_num = int(dir_path.name.split("_")[1])
        remaining_steps.append(step_num)

    # Should keep checkpoints with actual loss values (steps 1, 2, 4, 12)
    # and exclude those without loss metrics (steps 3, 10)
    # The latest checkpoint (step 12) is always kept
    expected_steps = [1, 2, 4, 12]  # Steps with loss metrics, plus latest
    assert sorted(remaining_steps) == sorted(expected_steps)


def test_remove_old_checkpoints_topk_most_missing_val_metric(
    checkpoint_manager, checkpoint_dir
):
    # Create checkpoints where some have validation metrics and others don't
    steps = [1, 2, 3, 4, 10, 12]
    # Some checkpoints have loss metrics, others don't have any validation metrics
    training_infos = [
        {"loss": 0.2},  # step 1 - has loss
        {},  # step 2 - has loss
        {"other_metric": 0.8},  # step 3 - missing loss metric
        {},  # step 4 - has loss
        {},  # step 10 - missing loss metric
        {},  # step 12 - missing loss (latest)
    ]

    for step, training_info in zip(steps, training_infos):
        tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
        checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Check if only top-k checkpoints are kept
    remaining_dirs = list(checkpoint_dir.glob("step_*"))
    assert len(remaining_dirs) == checkpoint_manager.keep_top_k

    # Checkpoints with missing validation metrics should be treated as having the worst possible value
    # Since higher_is_better=False, missing metrics get float("inf") which is worst
    # So checkpoints with actual loss values should be preferred over those without
    remaining_steps = []
    for dir_path in remaining_dirs:
        step_num = int(dir_path.name.split("_")[1])
        remaining_steps.append(step_num)

    # Should keep checkpoints with actual loss values (step 1)
    # followed by the most recent steps
    # The latest checkpoint (step 12) is always kept
    expected_steps = [1, 10, 12]  # Steps with loss metrics, plus latest
    assert sorted(remaining_steps) == sorted(expected_steps)


def test_get_best_checkpoint_path(checkpoint_manager, checkpoint_dir):
    # Create multiple checkpoints with different loss values
    steps = [1, 2, 3]
    losses = [0.5, 0.3, 0.7]

    for step, loss in zip(steps, losses):
        training_info = {"loss": loss}
        tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
        checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Get best checkpoint path
    best_path = checkpoint_manager.get_best_checkpoint_path()

    # Verify it's the checkpoint with lowest loss
    with open(Path(best_path) / "training_info.json", "r") as f:
        metadata = json.load(f)
        assert metadata["loss"] == min(losses)


def test_get_latest_checkpoint_path(checkpoint_manager, checkpoint_dir):
    # Create multiple checkpoints
    steps = [1, 2, 3]

    for step in steps:
        training_info = {"loss": 0.5}
        tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
        checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Get latest checkpoint path
    latest_path = checkpoint_manager.get_latest_checkpoint_path()

    # Verify it's the checkpoint with highest step number
    assert Path(latest_path).name == f"step_{max(steps)}"


def test_get_latest_checkpoint_path_with_suffixes(checkpoint_manager, checkpoint_dir):
    """Test that having step_*-hf dirs alongside step_* checkpoints doesn't crash."""
    # Create a checkpoint
    step = 1
    training_info = {"loss": 0.5}
    tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
    checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Create pseudo-converted checkpoint folder
    (checkpoint_dir / "step_1-hf").mkdir()

    # Get latest checkpoint path
    latest_path = checkpoint_manager.get_latest_checkpoint_path()

    # Verify the -hf suffix didn't affect the get_latest_checkpoint func
    assert Path(latest_path).name == "step_1"


def test_load_training_metadata(checkpoint_manager, checkpoint_dir):
    # Create a checkpoint
    step = 1
    training_info = {"loss": 0.5}
    tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
    checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Load training metadata
    metadata = checkpoint_manager.load_training_info(checkpoint_dir / f"step_{step}")

    # Verify metadata was loaded correctly
    assert metadata == training_info


def test_checkpoint_without_keep_top_k(tmp_path):
    # Test checkpoint manager without keep_top_k
    config = {
        "enabled": True,
        "checkpoint_dir": str((tmp_path.resolve() / "checkpoints")),
        "metric_name": "loss",
        "higher_is_better": False,
        "save_period": 1,
        "keep_top_k": None,
    }
    manager = CheckpointManager(config)

    # Create multiple checkpoints
    steps = [1, 2, 3]
    for step in steps:
        training_info = {"loss": 0.5}
        tmp_dir = manager.init_tmp_checkpoint(step, training_info)
        manager.finalize_checkpoint(tmp_dir)

    # Verify all checkpoints are kept
    remaining_dirs = list(Path(tmp_path.resolve() / "checkpoints").glob("step_*"))
    assert len(remaining_dirs) == len(steps)


def test_load_checkpoint_empty_dir(checkpoint_manager, checkpoint_dir):
    """Test that loading from an empty checkpoint directory returns None."""
    # Get latest checkpoint path from empty directory
    latest_path = checkpoint_manager.get_latest_checkpoint_path()
    assert latest_path is None

    # Load training metadata from None path
    metadata = checkpoint_manager.load_training_info(None)
    assert metadata is None


def test_get_latest_checkpoint_path_across_digits(checkpoint_manager, checkpoint_dir):
    """Test that getting latest checkpoint works correctly when crossing digit boundaries.
    This ensures we're doing numerical comparison rather than string comparison,
    as string comparison would incorrectly order step_9 > step_10.
    """
    # Create checkpoints with steps that cross digit boundary
    steps = [8, 9, 10, 11]

    for step in steps:
        training_info = {"loss": 0.5}
        tmp_dir = checkpoint_manager.init_tmp_checkpoint(step, training_info)
        checkpoint_manager.finalize_checkpoint(tmp_dir)

    # Get latest checkpoint path
    latest_path = checkpoint_manager.get_latest_checkpoint_path()

    # Verify it's the checkpoint with highest numerical step (11)
    assert Path(latest_path).name == f"step_{max(steps)}"

    # Double check that all checkpoints exist and are properly ordered
    all_checkpoints = sorted(
        [d for d in Path(checkpoint_dir).glob("step_*")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    assert len(all_checkpoints) == checkpoint_manager.keep_top_k
    assert all_checkpoints[-1].name == f"step_{max(steps)}"


def test_get_best_checkpoint_path_no_checkpoints(checkpoint_manager, checkpoint_dir):
    """Test that get_best_checkpoint_path returns None when no checkpoints exist."""
    result = checkpoint_manager.get_best_checkpoint_path()
    assert result is None


def test_get_best_checkpoint_path_some_missing_metric(tmp_path):
    """Test that get_best_checkpoint_path filters out checkpoints missing the metric and warns."""
    # Use keep_top_k=None to keep all checkpoints for this test
    config = {
        "enabled": True,
        "checkpoint_dir": str((tmp_path.resolve() / "checkpoints")),
        "metric_name": "loss",
        "higher_is_better": False,
        "save_period": 1,
        "keep_top_k": None,  # Keep all checkpoints
    }
    manager = CheckpointManager(config)

    # Create checkpoints where some have the metric and others don't
    steps = [1, 2, 3, 4]
    training_infos = [
        {"loss": 0.5},  # step 1 - has loss
        {"other_metric": 0.8},  # step 2 - missing loss
        {"loss": 0.3},  # step 3 - has loss (best)
        {},  # step 4 - missing loss
    ]

    for step, training_info in zip(steps, training_infos):
        tmp_dir = manager.init_tmp_checkpoint(step, training_info)
        manager.finalize_checkpoint(tmp_dir)

    # Should warn about missing metrics but still return the best checkpoint
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        best_path = manager.get_best_checkpoint_path()

        # Should have warned about 2 checkpoints missing the metric
        assert len(w) == 1
        assert "Ignoring 2 checkpoint(s)" in str(w[0].message)
        assert "val_at_end" in str(w[0].message)

    # Should return the checkpoint with the best (lowest) loss
    with open(Path(best_path) / "training_info.json", "r") as f:
        metadata = json.load(f)
        assert metadata["loss"] == 0.3  # step 3 has the best loss


def test_get_best_checkpoint_path_all_missing_metric(tmp_path):
    """Test that get_best_checkpoint_path returns latest checkpoint when all are missing the metric."""
    # Use keep_top_k=None to keep all checkpoints for this test
    config = {
        "enabled": True,
        "checkpoint_dir": str((tmp_path.resolve() / "checkpoints")),
        "metric_name": "loss",
        "higher_is_better": False,
        "save_period": 1,
        "keep_top_k": None,  # Keep all checkpoints
    }
    manager = CheckpointManager(config)

    # Create checkpoints where none have the required metric
    steps = [1, 2, 3]
    training_infos = [
        {"other_metric": 0.5},  # step 1 - missing loss
        {},  # step 2 - missing loss
        {"different_metric": 0.3},  # step 3 - missing loss
    ]

    for step, training_info in zip(steps, training_infos):
        tmp_dir = manager.init_tmp_checkpoint(step, training_info)
        manager.finalize_checkpoint(tmp_dir)

    # Should warn and return latest checkpoint when no checkpoints have the metric
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        best_path = manager.get_best_checkpoint_path()

        # Should have warned twice: once about ignoring all checkpoints, once about returning latest
        assert len(w) == 2
        assert "Ignoring 3 checkpoint(s)" in str(w[0].message)
        assert "No checkpoints contain metric 'loss'" in str(w[1].message)
        assert "Returning latest checkpoint" in str(w[1].message)
        assert "val_at_end" in str(w[1].message)

    # Should return the latest checkpoint (step 3)
    assert Path(best_path).name == "step_3"


def test_get_best_checkpoint_path_higher_is_better(tmp_path):
    """Test get_best_checkpoint_path with higher_is_better=True."""
    config = {
        "enabled": True,
        "checkpoint_dir": str((tmp_path.resolve() / "checkpoints")),
        "metric_name": "accuracy",
        "higher_is_better": True,
        "save_period": 1,
        "keep_top_k": None,  # Keep all
    }
    manager = CheckpointManager(config)

    # Create checkpoints with different accuracy values
    steps = [1, 2, 3]
    accuracies = [0.7, 0.9, 0.8]  # step 2 has the best accuracy

    for step, acc in zip(steps, accuracies):
        training_info = {"accuracy": acc}
        tmp_dir = manager.init_tmp_checkpoint(step, training_info)
        manager.finalize_checkpoint(tmp_dir)

    # Get best checkpoint path
    best_path = manager.get_best_checkpoint_path()

    # Verify it's the checkpoint with highest accuracy
    with open(Path(best_path) / "training_info.json", "r") as f:
        metadata = json.load(f)
        assert metadata["accuracy"] == 0.9  # step 2


@pytest.fixture
def async_checkpoint_config(checkpoint_dir):
    return {
        "enabled": True,
        "checkpoint_dir": checkpoint_dir,
        "metric_name": None,
        "higher_is_better": False,
        "save_period": 1,
        "keep_top_k": None,
    }


@pytest.fixture
def async_manager(async_checkpoint_config):
    mgr = CheckpointManager(async_checkpoint_config)
    yield mgr
    mgr.shutdown()


class TestBeginFinalization:
    """Tests for CheckpointManager.begin_finalization()."""

    def test_basic_rename(self, async_manager, checkpoint_dir):
        """begin_finalization with no wait_fn renames immediately."""
        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=None)
        async_manager.finalize_pending()

        assert not Path(tmp).exists()
        assert (checkpoint_dir / "step_1").exists()

    def test_wait_fn_called_before_rename(self, async_manager, checkpoint_dir):
        """wait_fn is called and completes before the rename happens."""
        call_order = []

        def mock_wait():
            call_order.append("wait_start")
            time.sleep(0.05)
            call_order.append("wait_end")

        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=mock_wait)
        async_manager.finalize_pending()

        assert call_order == ["wait_start", "wait_end"]
        assert (checkpoint_dir / "step_1").exists()

    def test_begin_blocks_if_previous_pending(self, async_manager, checkpoint_dir):
        """Second begin_finalization blocks until first completes."""
        barrier = threading.Event()

        def slow_wait():
            barrier.wait(timeout=5)

        tmp1 = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp1, wait_fn=slow_wait)

        tmp2 = async_manager.init_tmp_checkpoint(2, {"loss": 0.2})
        barrier.set()
        async_manager.begin_finalization(tmp2, wait_fn=None)
        async_manager.finalize_pending()

        assert (checkpoint_dir / "step_1").exists()
        assert (checkpoint_dir / "step_2").exists()

    def test_sync_finalize_still_works(self, async_manager, checkpoint_dir):
        """Original synchronous finalize_checkpoint API still works."""
        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.finalize_checkpoint(tmp)

        assert (checkpoint_dir / "step_1").exists()
        assert not Path(tmp).exists()


class TestFinalizePending:
    """Tests for CheckpointManager.finalize_pending()."""

    def test_noop_when_nothing_pending(self, async_manager):
        """finalize_pending is a safe no-op when no finalization is active."""
        async_manager.finalize_pending()

    def test_idempotent_double_call(self, async_manager, checkpoint_dir):
        """Calling finalize_pending twice is harmless."""
        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=None)
        async_manager.finalize_pending()
        async_manager.finalize_pending()
        assert (checkpoint_dir / "step_1").exists()

    def test_propagates_error_from_wait_fn(self, async_manager):
        """Exceptions in wait_fn are re-raised from finalize_pending."""

        def failing_wait():
            raise ValueError("Simulated async write failure")

        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=failing_wait)

        with pytest.raises(RuntimeError, match="Background checkpoint finalization failed"):
            async_manager.finalize_pending()

    def test_error_cleared_after_raise(self, async_manager, checkpoint_dir):
        """After an error is raised, manager is in clean state for next save."""

        def failing_wait():
            raise ValueError("boom")

        tmp1 = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp1, wait_fn=failing_wait)

        with pytest.raises(RuntimeError):
            async_manager.finalize_pending()

        tmp2 = async_manager.init_tmp_checkpoint(2, {"loss": 0.2})
        async_manager.begin_finalization(tmp2, wait_fn=None)
        async_manager.finalize_pending()
        assert (checkpoint_dir / "step_2").exists()


class TestShutdown:
    """Tests for CheckpointManager.shutdown()."""

    def test_waits_for_rename_and_deletion(self, checkpoint_dir):
        """shutdown() blocks until both rename and deletion complete."""
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": 1,
        }
        manager = CheckpointManager(config)

        for step in [1, 2, 3]:
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.begin_finalization(tmp, wait_fn=None)

        manager.shutdown()

        remaining = sorted(checkpoint_dir.glob("step_*"))
        assert len(remaining) == 1
        assert remaining[0].name == "step_3"

    def test_noop_when_nothing_pending(self, async_manager):
        """shutdown is safe to call with no pending work."""
        async_manager.shutdown()

    def test_double_shutdown(self, async_manager, checkpoint_dir):
        """Calling shutdown twice is harmless."""
        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=None)
        async_manager.shutdown()
        async_manager.shutdown()
        assert (checkpoint_dir / "step_1").exists()

    def test_usable_after_shutdown(self, async_manager, checkpoint_dir):
        """Manager can be used for new saves after shutdown."""
        tmp1 = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp1, wait_fn=None)
        async_manager.shutdown()

        tmp2 = async_manager.init_tmp_checkpoint(2, {"loss": 0.2})
        async_manager.begin_finalization(tmp2, wait_fn=None)
        async_manager.finalize_pending()
        assert (checkpoint_dir / "step_2").exists()


class TestHasPendingFinalization:
    """Tests for CheckpointManager.has_pending_finalization property."""

    def test_false_initially(self, async_manager):
        assert not async_manager.has_pending_finalization

    def test_true_while_active(self, async_manager):
        barrier = threading.Event()

        def slow_wait():
            barrier.wait(timeout=5)

        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=slow_wait)

        assert async_manager.has_pending_finalization

        barrier.set()
        async_manager.finalize_pending()

    def test_false_after_finalize(self, async_manager, checkpoint_dir):
        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=None)
        async_manager.finalize_pending()

        assert not async_manager.has_pending_finalization

    def test_false_after_error(self, async_manager):
        """has_pending_finalization is False after a failed finalization."""

        def failing_wait():
            raise ValueError("boom")

        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager.begin_finalization(tmp, wait_fn=failing_wait)

        with pytest.raises(RuntimeError):
            async_manager.finalize_pending()

        assert not async_manager.has_pending_finalization


class TestDeletionSerialization:
    """Tests for deletion via ThreadPoolExecutor."""

    def test_keep_top_k_with_async_finalization(self, checkpoint_dir):
        """Old checkpoints are deleted without blocking the main thread."""
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": 2,
        }
        manager = CheckpointManager(config)

        for step in range(1, 6):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.begin_finalization(tmp, wait_fn=None)

        manager.shutdown()

        remaining = sorted(
            checkpoint_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])
        )
        assert len(remaining) == 2
        assert remaining[0].name == "step_4"
        assert remaining[1].name == "step_5"

    def test_deletion_does_not_block_next_save(self, checkpoint_dir):
        """Deletion runs asynchronously, so the next begin_finalization returns quickly."""
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": 1,
        }
        manager = CheckpointManager(config)

        tmp1 = manager.init_tmp_checkpoint(1, {"loss": 0.1})
        manager.begin_finalization(tmp1, wait_fn=None)
        manager.finalize_pending()

        tmp2 = manager.init_tmp_checkpoint(2, {"loss": 0.2})
        start = time.monotonic()
        manager.begin_finalization(tmp2, wait_fn=None)
        elapsed = time.monotonic() - start
        # begin_finalization should return quickly (< 1s) even if deletion is slow
        assert elapsed < 1.0

        manager.shutdown()
        remaining = list(checkpoint_dir.glob("step_*"))
        assert len(remaining) == 1
        assert remaining[0].name == "step_2"


class TestRenameCheckpoint:
    """Tests for _rename_checkpoint edge cases."""

    def test_rename_overwrites_existing_step(self, async_manager, checkpoint_dir):
        """If step_N already exists, it is swapped via old_step_N."""
        existing = checkpoint_dir / "step_1"
        existing.mkdir(parents=True)
        (existing / "stale_marker.txt").write_text("old")

        tmp = async_manager.init_tmp_checkpoint(1, {"loss": 0.1})
        async_manager._rename_checkpoint(tmp)

        assert (checkpoint_dir / "step_1").exists()
        assert not (checkpoint_dir / "old_step_1").exists()
        info = json.loads((checkpoint_dir / "step_1" / "training_info.json").read_text())
        assert info["loss"] == 0.1
        assert not (checkpoint_dir / "step_1" / "stale_marker.txt").exists()


class TestMultiStepSequence:
    """End-to-end multi-step save/finalize sequence mirroring the training loop."""

    def test_three_step_sequence(self, checkpoint_dir):
        """Simulate 3 training steps with async finalization."""
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": 2,
        }
        manager = CheckpointManager(config)

        wait_called = []

        def make_wait(step):
            def wait_fn():
                time.sleep(0.02)
                wait_called.append(step)
            return wait_fn

        for step in [1, 2, 3]:
            manager.finalize_pending()
            tmp = manager.init_tmp_checkpoint(step, {"loss": 1.0 / step})
            manager.begin_finalization(tmp, wait_fn=make_wait(step))

        manager.shutdown()

        assert wait_called == [1, 2, 3]

        remaining = sorted(
            checkpoint_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])
        )
        assert len(remaining) == 2
        assert remaining[0].name == "step_2"
        assert remaining[1].name == "step_3"

    def test_mixed_sync_and_async(self, checkpoint_dir):
        """Mix of synchronous finalize_checkpoint and async begin_finalization."""
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": None,
        }
        manager = CheckpointManager(config)

        tmp1 = manager.init_tmp_checkpoint(1, {"loss": 0.1})
        manager.finalize_checkpoint(tmp1)

        tmp2 = manager.init_tmp_checkpoint(2, {"loss": 0.2})
        manager.begin_finalization(tmp2, wait_fn=None)

        tmp3 = manager.init_tmp_checkpoint(3, {"loss": 0.3})
        manager.finalize_checkpoint(tmp3)

        manager.shutdown()

        for step in [1, 2, 3]:
            assert (checkpoint_dir / f"step_{step}").exists()

    def test_latest_checkpoint_visible_after_async_finalize(
        self, checkpoint_dir
    ):
        """get_latest_checkpoint_path reflects the async-finalized checkpoint."""
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": None,
        }
        manager = CheckpointManager(config)

        tmp = manager.init_tmp_checkpoint(5, {"loss": 0.1})
        manager.begin_finalization(tmp, wait_fn=None)
        manager.finalize_pending()

        latest = manager.get_latest_checkpoint_path()
        assert latest is not None
        assert Path(latest).name == "step_5"


# ---------------------------------------------------------------------------
# Fault tolerance (ft_keep_latest_k) tests
# ---------------------------------------------------------------------------


class TestFTKeepLatestK:
    """Tests for the ft_keep_latest_k union retention policy.

    A checkpoint survives if EITHER keep_top_k or ft_keep_latest_k retains it.
    keep_top_k=None means that policy is inactive (no protection).
    ft_keep_latest_k=None means that policy is inactive (no protection).
    Both None → nothing is deleted.
    """

    def _make_manager(self, checkpoint_dir, keep_top_k=None, ft_keep_latest_k=None, metric_name=None, higher_is_better=False, save_period=1):
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "save_period": save_period,
            "keep_top_k": keep_top_k,
            "ft_keep_latest_k": ft_keep_latest_k,
        }
        return CheckpointManager(config)

    def test_both_none_keeps_everything(self, checkpoint_dir):
        """When both policies are None, no checkpoints are deleted."""
        manager = self._make_manager(checkpoint_dir)
        for step in range(1, 6):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining = sorted(checkpoint_dir.glob("step_*"))
        assert len(remaining) == 5

    def test_ft_keep_1_only(self, checkpoint_dir):
        """ft_keep_latest_k=1 with no periodic checkpoints keeps only the most recent."""
        # save_period=100 so no steps in range are periodic — isolates ft behavior
        manager = self._make_manager(checkpoint_dir, ft_keep_latest_k=1, save_period=100)
        for step in range(1, 6):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        assert remaining_steps == [5]

    def test_ft_keep_2_only(self, checkpoint_dir):
        """ft_keep_latest_k=2 with no periodic checkpoints keeps the two most recent."""
        # save_period=100 so no steps in range are periodic — isolates ft behavior
        manager = self._make_manager(checkpoint_dir, ft_keep_latest_k=2, save_period=100)
        for step in range(1, 8):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        assert remaining_steps == [6, 7]

    def test_keep_top_k_only(self, checkpoint_dir):
        """keep_top_k=2 alone keeps the 2 most recent (no metric)."""
        manager = self._make_manager(checkpoint_dir, keep_top_k=2)
        for step in range(1, 6):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        assert remaining_steps == [4, 5]

    def test_union_of_both_policies(self, checkpoint_dir):
        """A checkpoint survives if either policy retains it."""
        # keep_top_k=1 (no metric → most recent) + ft_keep_latest_k=1
        # Both pick the same checkpoint (step 9), so result is just {9}.
        manager = self._make_manager(checkpoint_dir, keep_top_k=1, ft_keep_latest_k=1)
        for step in range(1, 10):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        assert remaining_steps == [9]

    def test_union_with_metric_picks_different_sets(self, checkpoint_dir):
        """When metric-based top-k and recency pick different checkpoints, the union is kept."""
        manager = self._make_manager(
            checkpoint_dir,
            keep_top_k=1,
            ft_keep_latest_k=1,
            metric_name="reward",
            higher_is_better=True,
        )

        data = [
            (1, {"reward": 0.5}),
            (2, {"reward": 0.9}),  # best by metric
            (3, {"reward": 0.3}),
            (4, {"reward": 0.1}),
            (5, {"reward": 0.2}),
            (6, {"reward": 0.4}),  # most recent
        ]
        for step, info in data:
            tmp = manager.init_tmp_checkpoint(step, info)
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=1 by metric → step 2 (reward=0.9)
        # ft_keep_latest_k=1 → step 6 (most recent)
        # exclude_latest also protects step 6
        assert remaining_steps == [2, 6]

    def test_union_with_overlap(self, checkpoint_dir):
        """When both policies protect the same checkpoint, no double-counting."""
        manager = self._make_manager(
            checkpoint_dir,
            keep_top_k=2,
            ft_keep_latest_k=2,
            metric_name="reward",
            higher_is_better=True,
        )

        data = [
            (1, {"reward": 0.1}),
            (2, {"reward": 0.2}),
            (3, {"reward": 0.3}),
            (4, {"reward": 0.9}),  # top-2 by metric
            (5, {"reward": 0.8}),  # top-2 by metric AND latest-2
            (6, {"reward": 0.4}),  # latest-2 AND most recent
        ]
        for step, info in data:
            tmp = manager.init_tmp_checkpoint(step, info)
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=2 by metric → steps 4, 5
        # ft_keep_latest_k=2 → steps 5, 6
        # union: {4, 5, 6}
        assert remaining_steps == [4, 5, 6]

    def test_ft_with_async_finalization(self, checkpoint_dir):
        """FT retention works correctly with begin_finalization + shutdown."""
        manager = self._make_manager(checkpoint_dir, keep_top_k=2, ft_keep_latest_k=1)
        for step in range(1, 7):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.begin_finalization(tmp, wait_fn=None)

        manager.shutdown()

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=2 (no metric → most recent 2): steps 5, 6
        # ft_keep_latest_k=1: step 6
        # union: {5, 6}
        assert remaining_steps == [5, 6]

    def test_backward_compat_no_ft_keep(self, checkpoint_dir):
        """Omitting ft_keep_latest_k preserves original keep_top_k behavior."""
        config = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": 2,
        }
        manager = CheckpointManager(config)
        assert manager.ft_keep_latest_k is None

        for step in range(1, 5):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=2 keeps latest 2
        assert remaining_steps == [3, 4]

    def test_exclude_latest_protects_most_recent(self, checkpoint_dir):
        """exclude_latest=True protects the most recent checkpoint even beyond both policies."""
        manager = self._make_manager(
            checkpoint_dir,
            keep_top_k=1,
            ft_keep_latest_k=1,
            metric_name="reward",
            higher_is_better=True,
        )

        data = [
            (1, {"reward": 0.9}),  # best by metric
            (2, {"reward": 0.1}),  # most recent, worst by metric
        ]
        for step, info in data:
            tmp = manager.init_tmp_checkpoint(step, info)
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=1 → step 1 (best metric)
        # ft_keep_latest_k=1 → step 2 (most recent)
        # Both are protected, both survive
        assert remaining_steps == [1, 2]

    def test_resume_picks_latest_checkpoint(self, checkpoint_dir):
        """get_latest_checkpoint_path returns the highest step number."""
        manager = self._make_manager(checkpoint_dir, ft_keep_latest_k=1)
        for step in [5, 6, 7]:
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        latest = manager.get_latest_checkpoint_path()
        assert Path(latest).name == "step_7"

    def test_keep_top_k_only_considers_periodic_checkpoints(self, checkpoint_dir):
        """keep_top_k=2 with save_period=4 only retains save_period-aligned checkpoints."""
        manager = self._make_manager(checkpoint_dir, keep_top_k=2, ft_keep_latest_k=2, save_period=4)
        # Steps 1-10: save_period-aligned are 4, 8
        for step in range(1, 11):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=2 among periodic (4, 8) → both retained
        # ft_keep_latest_k=2 → steps 9, 10
        # exclude_latest → step 10
        # union: {4, 8, 9, 10}
        assert remaining_steps == [4, 8, 9, 10]

    def test_save_period_non_aligned_only_kept_by_ft(self, checkpoint_dir):
        """Non-aligned checkpoints are only protected by ft_keep_latest_k, not keep_top_k."""
        manager = self._make_manager(checkpoint_dir, keep_top_k=3, ft_keep_latest_k=1, save_period=5)
        for step in range(1, 16):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=3 among periodic (5, 10, 15) → all 3 retained
        # ft_keep_latest_k=1 → step 15 (already in periodic set)
        # exclude_latest → step 15
        # union: {5, 10, 15}
        assert remaining_steps == [5, 10, 15]

    def test_save_period_with_metric_picks_best_periodic(self, checkpoint_dir):
        """keep_top_k with metric selects best among save_period-aligned checkpoints only."""
        manager = self._make_manager(
            checkpoint_dir,
            keep_top_k=1,
            ft_keep_latest_k=2,
            save_period=3,
            metric_name="reward",
            higher_is_better=True,
        )
        data = [
            (1, {"reward": 0.9}),   # non-aligned, high reward — not considered by keep_top_k
            (2, {"reward": 0.1}),   # non-aligned
            (3, {"reward": 0.5}),   # aligned
            (4, {"reward": 0.8}),   # non-aligned, high reward — not considered by keep_top_k
            (5, {"reward": 0.2}),   # non-aligned
            (6, {"reward": 0.7}),   # aligned, best periodic by metric
            (7, {"reward": 0.3}),   # non-aligned
            (8, {"reward": 0.4}),   # non-aligned, most recent
        ]
        for step, info in data:
            tmp = manager.init_tmp_checkpoint(step, info)
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=1 among periodic (3, 6): step 6 wins (reward=0.7 > 0.5)
        # ft_keep_latest_k=2: steps 7, 8
        # exclude_latest: step 8
        # union: {6, 7, 8}
        assert remaining_steps == [6, 7, 8]

    def test_ft_only_no_keep_top_k_with_save_period(self, checkpoint_dir):
        """With keep_top_k=None, all periodic checkpoints are retained plus ft rolling window."""
        manager = self._make_manager(checkpoint_dir, ft_keep_latest_k=2, save_period=4)
        for step in range(1, 11):
            tmp = manager.init_tmp_checkpoint(step, {"loss": float(step)})
            manager.finalize_checkpoint(tmp)

        remaining_steps = sorted(
            int(p.name.split("_")[1]) for p in checkpoint_dir.glob("step_*")
        )
        # keep_top_k=None → all periodic (4, 8) retained
        # ft_keep_latest_k=2 → steps 9, 10
        # exclude_latest → step 10
        # union: {4, 8, 9, 10}
        assert remaining_steps == [4, 8, 9, 10]
