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
from pathlib import Path

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
