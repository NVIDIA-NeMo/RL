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

from unittest.mock import MagicMock, patch

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.loss_functions import PreferenceLoss
from nemo_rl.algorithms.rm import _default_rm_save_state, rm_train


@pytest.fixture
def mock_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {
            "loss": [0.5],
            "accuracy": [1.0],
            "rewards_chosen_mean": [4.5],
            "rewards_rejected_mean": [3.5],
            "num_valid_samples": [1.0],
            "global_valid_toks": [10],
        },
    }

    # Create a proper message log structure with token_ids
    mock_batch = {
        "message_log": [
            [  # chosen
                {"role": "user", "token_ids": torch.tensor([1, 2, 3])},
                {"role": "assistant", "token_ids": torch.tensor([4, 5, 6])},
            ],
            [  # rejected
                {"role": "user", "token_ids": torch.tensor([1, 2, 3])},
                {"role": "assistant", "token_ids": torch.tensor([7, 8, 9, 10, 11])},
            ],
        ],
        "length": torch.tensor([6, 8]),
        "loss_multiplier": torch.tensor([1.0, 1.0]),
    }

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = PreferenceLoss()
    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock master config
    master_config = {
        "rm": {
            "max_num_steps": 5,
            "max_num_epochs": 2,
            "val_period": 100,
            "val_batches": 1,
            "val_global_batch_size": 1,
            "val_micro_batch_size": 1,
            "val_at_start": False,
        },
        "policy": {
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 1,
            "reward_model_cfg": {
                "enabled": True,
                "reward_model_type": "bradley_terry",
            },
            "train_micro_batch_size": 1,
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
        },
        "cluster": {
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
    }

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "master_config": master_config,
    }


def test_exit_on_max_steps(mock_components):
    """Test that training loop exits when max_num_steps is reached"""
    # Set max steps to 12, which is less than len(train_dataloader) * max_num_epochs
    mock_components["master_config"]["rm"]["max_num_steps"] = 12

    rm_save_state = _default_rm_save_state()

    # Run training
    rm_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        rm_save_state,
    )

    # Verify we only trained for 12 steps.
    assert mock_components["policy"].train.call_count == 12


def test_exit_on_max_epochs(mock_components):
    """Test that training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_components["master_config"]["rm"]["max_num_epochs"] = 2
    mock_components["master_config"]["rm"]["max_num_steps"] = 100

    rm_save_state = _default_rm_save_state()

    # Run training
    rm_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        rm_save_state,
    )

    # Verify we trained for exactly two epochs (20 batches).
    assert mock_components["policy"].train.call_count == 20


def test_exit_on_timeout(mock_components, capsys):
    """Test that training loop exits when timeout is reached"""
    # Set max steps and epochs to large numbers
    mock_components["master_config"]["rm"]["max_num_steps"] = 100
    mock_components["master_config"]["rm"]["max_num_epochs"] = 10

    rm_save_state = _default_rm_save_state()

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.rm.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        # Create a side_effect that returns False 7 times, then True
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Run training
        rm_train(
            mock_components["policy"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["master_config"],
            mock_components["logger"],
            mock_components["checkpointer"],
            rm_save_state,
        )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_components["policy"].train.call_count == 8

        # Verify the timeout message was printed and is near the end (not followed by more training)
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # Verify no new epoch started after timeout (which would indicate a bug where break was used instead of return)
        remaining_lines = output_lines[timeout_line_idx:]
        for line in remaining_lines:
            assert "Epoch" not in line or "Epoch 1/10" in line, (
                f"Training continued to next epoch after timeout: {line}"
            )
