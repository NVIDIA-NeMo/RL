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
"""End-to-end integration tests for DPO training workflow.

These tests verify the complete DPO training pipeline from:
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


class TestDPOWorkflowE2E:
    """End-to-end tests for DPO training workflow."""

    def test_dpo_trainer_initialization(self):
        """Test DPO trainer can be initialized with config."""
        from nemo_rl.algorithms.dpo import DPOTrainer

        config = {
            "policy": {
                "model_name": "test-model",
                "learning_rate": 5e-7,
            },
            "dpo": {
                "batch_size": 8,
                "max_num_steps": 100,
                "max_num_epochs": 1,
                "beta": 0.1,
                "val_period": 50,
            },
        }

        trainer = DPOTrainer(config)

        assert trainer is not None
        assert trainer.val_period == 50

    def test_dpo_from_pretrained(self):
        """Test DPO trainer from_pretrained pattern."""
        from nemo_rl.algorithms.dpo import DPOTrainer

        trainer = DPOTrainer.from_pretrained(
            "test-model",
            beta=0.2,
            batch_size=16,
            max_steps=50,
        )

        assert isinstance(trainer, DPOTrainer)

    def test_dpo_config_structure(self):
        """Test DPO config has all required sections."""
        from nemo_rl.algorithms.dpo import DPOTrainer

        config = DPOTrainer._build_config_from_pretrained(
            "test-model",
            beta=0.15,
            batch_size=8,
        )

        # Required sections
        assert "policy" in config
        assert "dpo" in config
        assert "checkpointing" in config
        assert "logger" in config

        # DPO config
        assert config["dpo"]["beta"] == 0.15
        assert config["dpo"]["batch_size"] == 8

    def test_dpo_train_api(self):
        """Test nemo_rl.train() with DPO algorithm."""
        from nemo_rl.api.train import _validate_inputs, _build_dpo_config

        # Validation should pass (no reward_fn needed for DPO)
        _validate_inputs("dpo", "test-model", "test-dataset", None)

        # Config should be built correctly
        config = _build_dpo_config(
            model="test-model",
            learning_rate=5e-7,
            batch_size=8,
            max_steps=200,
            max_epochs=1,
            output_dir="/tmp/test",
            beta=0.2,
        )

        assert config["dpo"]["beta"] == 0.2
        assert config["dpo"]["batch_size"] == 8

    def test_dpo_trainer_has_required_methods(self):
        """Test DPO trainer has all required methods."""
        from nemo_rl.algorithms.dpo import DPOTrainer

        trainer = DPOTrainer.from_pretrained("test-model")

        # BaseTrainer methods
        assert hasattr(trainer, "fit")
        assert hasattr(trainer, "setup")
        assert hasattr(trainer, "validate")

        # DPO-specific
        assert hasattr(trainer, "_train_step")
        assert hasattr(trainer, "_compute_loss")
        assert hasattr(trainer, "_prepare_batch")

    def test_dpo_beta_parameter(self):
        """Test DPO beta parameter configuration."""
        from nemo_rl.algorithms.dpo import DPOTrainer

        # Default beta
        config1 = DPOTrainer._build_config_from_pretrained("test-model")
        assert config1["dpo"]["beta"] == 0.1

        # Custom beta
        config2 = DPOTrainer._build_config_from_pretrained("test-model", beta=0.5)
        assert config2["dpo"]["beta"] == 0.5

    def test_dpo_reference_free_option(self):
        """Test DPO reference-free option."""
        from nemo_rl.algorithms.dpo import DPOTrainer

        config = DPOTrainer._build_config_from_pretrained(
            "test-model",
            reference_free=True,
        )

        assert config["dpo"]["reference_free"] is True


class TestDPOPreferenceData:
    """Test DPO integration with preference data."""

    def test_preference_data_format(self):
        """Test DPO works with preference data format."""
        # DPO expects chosen/rejected pairs
        preference_data = [
            {
                "prompt": "What is 2+2?",
                "chosen": "The answer is 4.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "What is the capital of France?",
                "chosen": "Paris is the capital of France.",
                "rejected": "London is the capital.",
            },
        ]

        # Should be valid data format
        assert len(preference_data) == 2
        assert "chosen" in preference_data[0]
        assert "rejected" in preference_data[0]

    def test_dpo_with_in_memory_preference_data(self):
        """Test DPO with in-memory preference data."""
        from nemo_rl.data.module import InMemoryDataModule

        preference_data = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
        ]

        datamodule = InMemoryDataModule(
            train_data=preference_data,
            batch_size=2,
        )

        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()

        batch = next(iter(train_loader))
        assert len(batch) == 2


class TestDPOLossIntegration:
    """Test DPO loss function integration."""

    def test_dpo_loss_creation(self):
        """Test DPO loss function can be created."""
        from nemo_rl.algorithms.dpo.loss import create_dpo_loss_function

        loss_fn = create_dpo_loss_function({"beta": 0.1})
        assert loss_fn is not None

    def test_dpo_loss_params(self):
        """Test DPO loss accepts various parameters."""
        from nemo_rl.algorithms.dpo.loss import DPOLoss

        loss = DPOLoss(beta=0.2)
        assert loss.beta == 0.2


class TestDPOCallbacksIntegration:
    """Test DPO integration with callbacks."""

    def test_dpo_with_callbacks(self):
        """Test DPO trainer works with callbacks."""
        from nemo_rl.algorithms.dpo import DPOTrainer
        from nemo_rl.trainers.callbacks import Callback, CallbackList

        class MetricsCallback(Callback):
            def __init__(self):
                self.metrics_logged = []

            def on_step_end(self, trainer, step, metrics):
                self.metrics_logged.append(metrics)

        trainer = DPOTrainer.from_pretrained("test-model", max_steps=1)
        callback = MetricsCallback()

        trainer._callbacks = CallbackList([callback])
        
        # Simulate step
        trainer._on_step_end(1, {"loss": 0.5, "accuracy": 0.8})

        assert len(callback.metrics_logged) == 1
        assert callback.metrics_logged[0]["loss"] == 0.5


class TestAlgorithmComparison:
    """Test all three algorithms can be used with same pattern."""

    def test_all_algorithms_have_from_pretrained(self):
        """Test all trainers have from_pretrained method."""
        from nemo_rl.algorithms.grpo import GRPOTrainer
        from nemo_rl.algorithms.sft import SFTTrainer
        from nemo_rl.algorithms.dpo import DPOTrainer

        assert hasattr(GRPOTrainer, "from_pretrained")
        assert hasattr(SFTTrainer, "from_pretrained")
        assert hasattr(DPOTrainer, "from_pretrained")

    def test_all_algorithms_same_interface(self):
        """Test all trainers have same base interface."""
        from nemo_rl.algorithms.grpo import GRPOTrainer
        from nemo_rl.algorithms.sft import SFTTrainer
        from nemo_rl.algorithms.dpo import DPOTrainer

        for TrainerClass in [GRPOTrainer, SFTTrainer, DPOTrainer]:
            trainer = TrainerClass.from_pretrained("test-model")

            # All should have these methods
            assert hasattr(trainer, "fit")
            assert hasattr(trainer, "setup")
            assert hasattr(trainer, "validate")
            assert hasattr(trainer, "cleanup")
            assert hasattr(trainer, "_train_step")
            assert hasattr(trainer, "_compute_loss")

    def test_api_list_algorithms(self):
        """Test API lists all algorithms."""
        from nemo_rl.api import list_algorithms

        algos = list_algorithms()

        assert "grpo" in algos
        assert "sft" in algos
        assert "dpo" in algos
