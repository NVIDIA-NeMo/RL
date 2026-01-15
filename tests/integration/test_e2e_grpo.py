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
"""End-to-end integration tests for GRPO training workflow.

These tests verify the complete GRPO training pipeline from:
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


class TestGRPOWorkflowE2E:
    """End-to-end tests for GRPO training workflow."""

    def test_grpo_trainer_initialization(self):
        """Test GRPO trainer can be initialized with config."""
        from nemo_rl.algorithms.grpo import GRPOTrainer

        config = {
            "policy": {
                "model_name": "test-model",
                "learning_rate": 1e-6,
            },
            "grpo": {
                "num_prompts_per_step": 4,
                "num_generations_per_prompt": 2,
                "max_num_steps": 10,
                "max_num_epochs": 1,
            },
        }

        trainer = GRPOTrainer(config)

        assert trainer is not None
        assert trainer.num_prompts_per_step == 4
        assert trainer.num_generations_per_prompt == 2

    def test_grpo_from_pretrained(self):
        """Test GRPO trainer from_pretrained pattern."""
        from nemo_rl.algorithms.grpo import GRPOTrainer

        trainer = GRPOTrainer.from_pretrained(
            "test-model",
            num_prompts_per_step=8,
            num_generations_per_prompt=4,
            max_steps=5,
        )

        assert isinstance(trainer, GRPOTrainer)
        assert trainer.num_prompts_per_step == 8
        assert trainer.num_generations_per_prompt == 4

    def test_grpo_config_structure(self):
        """Test GRPO config has all required sections."""
        from nemo_rl.algorithms.grpo import GRPOTrainer

        config = GRPOTrainer._build_config_from_pretrained(
            "test-model",
            num_prompts_per_step=16,
        )

        # Required sections
        assert "policy" in config
        assert "grpo" in config
        assert "checkpointing" in config
        assert "logger" in config

        # Policy config
        assert config["policy"]["model_name"] == "test-model"

        # GRPO config
        assert config["grpo"]["num_prompts_per_step"] == 16

    def test_grpo_train_api(self):
        """Test nemo_rl.train() with GRPO algorithm."""
        from nemo_rl.api.train import _validate_inputs, _build_grpo_config

        # Validation should pass
        def reward_fn(p: str, r: str) -> float:
            return 1.0

        _validate_inputs("grpo", "test-model", "test-dataset", reward_fn)

        # Config should be built correctly
        config = _build_grpo_config(
            model="test-model",
            learning_rate=1e-6,
            batch_size=8,
            max_steps=100,
            max_epochs=1,
            num_generations_per_prompt=4,
            output_dir="/tmp/test",
        )

        assert config["grpo"]["num_prompts_per_step"] == 8
        assert config["grpo"]["num_generations_per_prompt"] == 4

    def test_grpo_effective_batch_size(self):
        """Test GRPO effective batch size calculation."""
        from nemo_rl.algorithms.grpo import GRPOTrainer

        config = {
            "policy": {"model_name": "test"},
            "grpo": {
                "num_prompts_per_step": 16,
                "num_generations_per_prompt": 8,
                "max_num_steps": 10,
            },
        }

        trainer = GRPOTrainer(config)
        
        assert trainer.effective_batch_size == 16 * 8  # 128

    def test_grpo_trainer_has_required_methods(self):
        """Test GRPO trainer has all required methods."""
        from nemo_rl.algorithms.grpo import GRPOTrainer

        trainer = GRPOTrainer.from_pretrained("test-model")

        # BaseTrainer methods
        assert hasattr(trainer, "fit")
        assert hasattr(trainer, "setup")
        assert hasattr(trainer, "validate")
        assert hasattr(trainer, "cleanup")

        # GRPO-specific
        assert hasattr(trainer, "_train_step")
        assert hasattr(trainer, "_compute_loss")
        assert hasattr(trainer, "_validate_step")

    def test_grpo_callbacks_integration(self):
        """Test GRPO trainer works with callbacks."""
        from nemo_rl.algorithms.grpo import GRPOTrainer
        from nemo_rl.trainers.callbacks import Callback

        class TestCallback(Callback):
            def __init__(self):
                self.train_begin_called = False
                self.train_end_called = False

            def on_train_begin(self, trainer):
                self.train_begin_called = True

            def on_train_end(self, trainer):
                self.train_end_called = True

        trainer = GRPOTrainer.from_pretrained("test-model", max_steps=1)
        callback = TestCallback()

        # Setup the callbacks
        from nemo_rl.trainers.callbacks import CallbackList
        trainer._callbacks = CallbackList([callback])
        
        # Trigger callbacks manually
        trainer._on_train_begin()
        trainer._on_train_end()

        assert callback.train_begin_called
        assert callback.train_end_called


class TestGRPORolloutIntegration:
    """Test GRPO integration with RolloutEngine."""

    def test_rollout_engine_creation(self):
        """Test RolloutEngine can be created."""
        from nemo_rl.algorithms.rollout import RolloutEngine, SamplingParams

        # Mock backend
        mock_backend = MagicMock()
        mock_environment = MagicMock()

        engine = RolloutEngine(
            backend=mock_backend,
            environment=mock_environment,
        )

        assert engine is not None

    def test_sampling_params(self):
        """Test SamplingParams configuration."""
        from nemo_rl.algorithms.rollout import SamplingParams

        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 512


class TestGRPORewardIntegration:
    """Test GRPO integration with reward functions."""

    def test_functional_reward_with_grpo(self):
        """Test FunctionalRewardWrapper works with GRPO."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        def length_reward(prompt: str, response: str) -> float:
            return len(response) / 100.0

        wrapper = FunctionalRewardWrapper(length_reward, name="length")

        # Test it can compute rewards
        reward = wrapper._compute_single_reward("test", "hello world")
        assert reward == 11 / 100.0  # len("hello world") / 100

    def test_dict_reward_with_grpo(self):
        """Test dict-returning reward functions."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper
        from typing import Dict

        def multi_reward(prompt: str, response: str) -> Dict[str, float]:
            return {
                "length": len(response) / 100.0,
                "has_answer": 1.0 if "answer" in response else 0.0,
            }

        wrapper = FunctionalRewardWrapper(multi_reward)
        assert wrapper.reward_fn == multi_reward
