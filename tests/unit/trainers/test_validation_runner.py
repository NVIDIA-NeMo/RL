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
"""Unit tests for ValidationRunner."""

import math
import pytest
from unittest.mock import MagicMock, Mock


class TestValidationRunner:
    """Tests for ValidationRunner class."""

    def test_import(self):
        """Test ValidationRunner can be imported."""
        from nemo_rl.trainers.validation import (
            ValidationRunner,
            ValidationConfig,
            ValidationResult,
            create_validation_runner,
        )
        assert ValidationRunner is not None
        assert ValidationConfig is not None
        assert ValidationResult is not None
        assert create_validation_runner is not None

    def test_default_initialization(self):
        """Test default ValidationRunner initialization."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner()
        assert runner.config.frequency == 100
        assert runner.config.mode == "steps"
        assert runner.config.metrics == ["loss"]
        assert runner.config.max_batches == -1

    def test_custom_initialization(self):
        """Test ValidationRunner with custom parameters."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner(
            metrics=["loss", "accuracy"],
            frequency=50,
            mode="epochs",
            max_batches=10,
            log_prefix="test",
        )
        assert runner.config.frequency == 50
        assert runner.config.mode == "epochs"
        assert runner.config.metrics == ["loss", "accuracy"]
        assert runner.config.max_batches == 10
        assert runner.config.log_prefix == "test"

    def test_should_validate_steps_mode(self):
        """Test should_validate in steps mode."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner(frequency=100, mode="steps")

        assert not runner.should_validate(step=0, epoch=0)
        assert not runner.should_validate(step=50, epoch=0)
        assert runner.should_validate(step=100, epoch=0)
        # Should not trigger again at same step
        assert not runner.should_validate(step=100, epoch=0)
        assert runner.should_validate(step=200, epoch=0)

    def test_should_validate_epochs_mode(self):
        """Test should_validate in epochs mode."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner(frequency=2, mode="epochs")

        assert runner.should_validate(step=0, epoch=0)
        assert not runner.should_validate(step=100, epoch=1)
        assert runner.should_validate(step=200, epoch=2)
        # Should not trigger again at same epoch
        assert not runner.should_validate(step=250, epoch=2)

    def test_should_validate_disabled(self):
        """Test validation disabled when frequency=0."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner(frequency=0)
        assert not runner.should_validate(step=100, epoch=0)
        assert not runner.should_validate(step=1000, epoch=10)

    def test_reset(self):
        """Test reset clears validation state."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner(frequency=100, mode="steps")

        # Trigger validation
        assert runner.should_validate(step=100, epoch=0)
        assert not runner.should_validate(step=100, epoch=0)

        # Reset and verify can validate again
        runner.reset()
        assert runner.should_validate(step=100, epoch=0)

    def test_register_metric(self):
        """Test registering custom metrics."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner()

        def custom_metric(batch, outputs):
            return {"custom_value": 1.0}

        runner.register_metric("custom", custom_metric)
        assert "custom" in runner._custom_metrics

    def test_run_basic(self):
        """Test run method with mock trainer."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner(metrics=["loss"], max_batches=3)

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer._validate_step.return_value = {"loss": 0.5}

        # Mock dataloader
        mock_batch = MagicMock()
        mock_batch.size = 32
        dataloader = [mock_batch] * 5

        result = runner.run(mock_trainer, iter(dataloader))

        # Should stop at max_batches=3
        assert result.num_batches == 3
        assert result.num_samples == 96  # 32 * 3
        assert "val_loss" in result.metrics
        assert result.metrics["val_loss"] == pytest.approx(0.5)
        assert "val_perplexity" in result.metrics

    def test_run_computes_perplexity(self):
        """Test that perplexity is computed from loss."""
        from nemo_rl.trainers.validation import ValidationRunner

        runner = ValidationRunner(metrics=["loss"], max_batches=1)

        mock_trainer = MagicMock()
        mock_trainer._validate_step.return_value = {"loss": 2.0}

        mock_batch = MagicMock()
        mock_batch.size = 1
        dataloader = [mock_batch]

        result = runner.run(mock_trainer, iter(dataloader))

        expected_ppl = math.exp(2.0)
        assert result.metrics["val_perplexity"] == pytest.approx(expected_ppl)

    def test_validation_result_repr(self):
        """Test ValidationResult string representation."""
        from nemo_rl.trainers.validation import ValidationResult

        result = ValidationResult(
            metrics={"val_loss": 0.5, "val_perplexity": 1.65},
            num_samples=100,
            num_batches=10,
        )
        repr_str = repr(result)
        assert "val_loss=0.5000" in repr_str
        assert "samples=100" in repr_str

    def test_create_validation_runner_factory(self):
        """Test factory function."""
        from nemo_rl.trainers.validation import create_validation_runner

        # With config dict
        config = {
            "val_period": 200,
            "val_batches": 5,
        }
        runner = create_validation_runner(config=config)
        assert runner.config.frequency == 200
        assert runner.config.max_batches == 5

        # Without config
        runner = create_validation_runner(frequency=50, mode="epochs")
        assert runner.config.frequency == 50
        assert runner.config.mode == "epochs"


class TestValidationRunnerIntegration:
    """Integration tests for ValidationRunner with real algorithms."""

    def test_import_from_trainers_package(self):
        """Test ValidationRunner is exported from trainers package."""
        from nemo_rl.trainers import ValidationRunner, create_validation_runner

        runner = create_validation_runner(frequency=100)
        assert isinstance(runner, ValidationRunner)

    def test_config_dataclass(self):
        """Test ValidationConfig dataclass."""
        from nemo_rl.trainers.validation import ValidationConfig

        config = ValidationConfig()
        assert config.frequency == 100
        assert config.mode == "steps"

        config = ValidationConfig(frequency=50, mode="epochs", metrics=["loss", "acc"])
        assert config.frequency == 50
        assert config.metrics == ["loss", "acc"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
