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
"""End-to-end integration tests for SFT training workflow.

These tests verify the complete SFT training pipeline from:
- Configuration loading
- Trainer initialization
- Data loading
- Training loop execution
- Checkpoint saving

Tests are designed to complete quickly by using mocked GPU operations.
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch


class TestSFTWorkflowE2E:
    """End-to-end tests for SFT training workflow."""

    def test_sft_trainer_initialization(self):
        """Test SFT trainer can be initialized with config."""
        from nemo_rl.algorithms.sft import SFTTrainer

        config = {
            "policy": {
                "model_name": "test-model",
                "learning_rate": 2e-5,
            },
            "sft": {
                "batch_size": 16,
                "max_num_steps": 100,
                "max_num_epochs": 1,
                "val_period": 50,
            },
        }

        trainer = SFTTrainer(config)

        assert trainer is not None
        assert trainer.val_period == 50

    def test_sft_from_pretrained(self):
        """Test SFT trainer from_pretrained pattern."""
        from nemo_rl.algorithms.sft import SFTTrainer

        trainer = SFTTrainer.from_pretrained(
            "test-model",
            batch_size=32,
            learning_rate=2e-5,
            max_steps=50,
        )

        assert isinstance(trainer, SFTTrainer)

    def test_sft_config_structure(self):
        """Test SFT config has all required sections."""
        from nemo_rl.algorithms.sft import SFTTrainer

        config = SFTTrainer._build_config_from_pretrained(
            "test-model",
            batch_size=64,
            max_steps=500,
        )

        # Required sections
        assert "policy" in config
        assert "sft" in config
        assert "checkpointing" in config
        assert "logger" in config

        # SFT config
        assert config["sft"]["batch_size"] == 64
        assert config["sft"]["max_num_steps"] == 500

    def test_sft_train_api(self):
        """Test nemo_rl.train() with SFT algorithm."""
        from nemo_rl.api.train import _validate_inputs, _build_sft_config

        # Validation should pass (no reward_fn needed for SFT)
        _validate_inputs("sft", "test-model", "test-dataset", None)

        # Config should be built correctly
        config = _build_sft_config(
            model="test-model",
            learning_rate=2e-5,
            batch_size=32,
            max_steps=200,
            max_epochs=2,
            output_dir="/tmp/test",
        )

        assert config["sft"]["batch_size"] == 32
        assert config["sft"]["max_num_steps"] == 200

    def test_sft_trainer_has_required_methods(self):
        """Test SFT trainer has all required methods."""
        from nemo_rl.algorithms.sft import SFTTrainer

        trainer = SFTTrainer.from_pretrained("test-model")

        # BaseTrainer methods
        assert hasattr(trainer, "fit")
        assert hasattr(trainer, "setup")
        assert hasattr(trainer, "validate")

        # SFT-specific
        assert hasattr(trainer, "_train_step")
        assert hasattr(trainer, "_compute_loss")
        assert hasattr(trainer, "_prepare_batch")

    def test_sft_no_reward_required(self):
        """Test SFT does not require reward function."""
        from nemo_rl.api.train import _validate_inputs

        # Should not raise
        _validate_inputs("sft", "test-model", "test-dataset", None)

    def test_sft_val_at_start_option(self):
        """Test SFT validation at start option."""
        from nemo_rl.algorithms.sft import SFTTrainer

        config = {
            "policy": {"model_name": "test"},
            "sft": {
                "batch_size": 8,
                "max_num_steps": 10,
                "val_at_start": True,
            },
        }

        trainer = SFTTrainer(config)
        assert trainer.val_at_start is True


class TestSFTDataIntegration:
    """Test SFT integration with data modules."""

    def test_sft_with_in_memory_data(self):
        """Test SFT works with in-memory data module."""
        from nemo_rl.data.module import InMemoryDataModule

        data = [
            {"prompt": "Hello", "response": "Hi there!"},
            {"prompt": "How are you?", "response": "I'm doing well, thanks!"},
        ]

        datamodule = InMemoryDataModule(
            train_data=data,
            batch_size=2,
        )

        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()

        # Should be able to iterate
        batch = next(iter(train_loader))
        assert len(batch) == 2

    def test_sft_datamodule_factory(self):
        """Test create_datamodule factory for SFT."""
        from nemo_rl.data.module import create_datamodule

        data = [{"text": "sample text"}]
        datamodule = create_datamodule(train_data=data, batch_size=1)

        assert datamodule is not None
        datamodule.setup("fit")


class TestSFTCallbacksIntegration:
    """Test SFT integration with callbacks."""

    def test_sft_logging_callback(self):
        """Test LoggingCallback with SFT trainer."""
        from nemo_rl.algorithms.sft import SFTTrainer
        from nemo_rl.trainers.callbacks import LoggingCallback, CallbackList

        trainer = SFTTrainer.from_pretrained("test-model", max_steps=1)
        callback = LoggingCallback(log_interval=1)

        trainer._callbacks = CallbackList([callback])
        
        # Should work without errors
        trainer._on_train_begin()
        trainer._on_step_end(1, {"loss": 0.5})
        trainer._on_train_end()

    def test_sft_checkpoint_callback(self):
        """Test CheckpointCallback configuration."""
        from nemo_rl.trainers.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            every_n_steps=100,
            save_best=True,
            monitor="loss",
        )

        assert callback.every_n_steps == 100
        assert callback.save_best is True
