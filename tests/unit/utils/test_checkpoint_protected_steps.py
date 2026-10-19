# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retention must preserve explicitly protected full recovery checkpoints."""

import json

import pytest

from nemo_rl.utils.checkpoint import CheckpointManager


@pytest.fixture
def config(tmp_path):
    for step in (20, 26, 30, 40, 50, 60):
        checkpoint = tmp_path / f"step_{step}"
        checkpoint.mkdir()
        (checkpoint / "training_info.json").write_text(
            json.dumps({"current_step": step})
        )
        (checkpoint / "weights-marker").write_text(str(step))
    (tmp_path / "tmp_step_70").mkdir()
    return {
        "enabled": True,
        "checkpoint_dir": tmp_path,
        "metric_name": None,
        "higher_is_better": True,
        "save_period": 20,
        "keep_top_k": 0,
        "ft_keep_latest_k": 2,
        "save_optimizer": True,
    }


def remaining(config):
    return sorted(
        int(p.name.removeprefix("step_"))
        for p in config["checkpoint_dir"].glob("step_*")
    )


def test_bounded_rotation_preserves_seed_and_two_latest(config):
    config["protected_steps"] = [20]
    manager = CheckpointManager(config)
    manager.remove_old_checkpoints()
    assert remaining(config) == [20, 50, 60]
    assert (config["checkpoint_dir"] / "step_20/weights-marker").read_text() == "20"
    assert (config["checkpoint_dir"] / "tmp_step_70").is_dir()


def test_protected_nonperiodic_seed_survives_top_k_and_recency(config):
    config["protected_steps"] = [26, 26, 999]
    manager = CheckpointManager(config)
    manager.remove_old_checkpoints()
    assert remaining(config) == [26, 50, 60]
    assert not (config["checkpoint_dir"] / "step_999").exists()


@pytest.mark.parametrize("value", [None, []])
def test_absent_or_empty_pins_preserve_existing_retention(config, value):
    config["protected_steps"] = value
    manager = CheckpointManager(config)
    manager.remove_old_checkpoints()
    assert remaining(config) == [50, 60]


@pytest.mark.parametrize("value", [[-1], [True], ["20"], "20", {20: True}, [1.5]])
def test_invalid_pins_fail_before_any_checkpoint_is_deleted(config, value):
    config["protected_steps"] = value
    with pytest.raises(ValueError, match="protected_steps"):
        CheckpointManager(config)
    assert remaining(config) == [20, 26, 30, 40, 50, 60]


def test_default_keep_all_periodic_checkpoints_remains_unchanged(config):
    config["keep_top_k"] = None
    manager = CheckpointManager(config)
    manager.remove_old_checkpoints()
    assert remaining(config) == [20, 40, 50, 60]
