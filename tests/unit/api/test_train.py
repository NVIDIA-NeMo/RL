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
"""Tests for nemo_rl.api.train module."""

import pytest
from unittest.mock import MagicMock, patch


class TestTrainAPI:
    """Tests for the train() function."""

    def test_import_train_from_nemo_rl(self):
        """Test that train can be imported from nemo_rl."""
        import nemo_rl
        
        assert hasattr(nemo_rl, "train")
        assert callable(nemo_rl.train)

    def test_import_train_from_api(self):
        """Test that train can be imported from nemo_rl.api."""
        from nemo_rl.api import train, TrainResult
        
        assert callable(train)
        assert TrainResult is not None

    def test_train_requires_model(self):
        """Test that train() raises error without model."""
        from nemo_rl.api import train
        
        with pytest.raises(TypeError):
            train(dataset="some_dataset")

    def test_train_requires_dataset(self):
        """Test that train() raises error without dataset."""
        from nemo_rl.api import train
        
        with pytest.raises(ValueError, match="Dataset is required"):
            train(model="Qwen/Qwen2.5-1.5B", dataset=None)

    def test_train_grpo_requires_reward_fn(self):
        """Test that GRPO requires reward_fn."""
        from nemo_rl.api import train
        
        with pytest.raises(ValueError, match="reward_fn is required for GRPO"):
            train(
                model="Qwen/Qwen2.5-1.5B",
                dataset="nvidia/OpenMathInstruct-2",
                algorithm="grpo",
                reward_fn=None,
            )

    def test_train_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        from nemo_rl.api import train
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            train(
                model="Qwen/Qwen2.5-1.5B",
                dataset="nvidia/OpenMathInstruct-2",
                algorithm="invalid",
            )

    def test_train_result_structure(self):
        """Test TrainResult dataclass structure."""
        from nemo_rl.api import TrainResult
        from nemo_rl.trainers.base import BaseTrainer
        
        mock_trainer = MagicMock(spec=BaseTrainer)
        
        result = TrainResult(
            trainer=mock_trainer,
            metrics={"loss": 0.5},
            checkpoint_path="/path/to/checkpoint",
            total_steps=100,
        )
        
        assert result.trainer == mock_trainer
        assert result.metrics["loss"] == 0.5
        assert result.checkpoint_path == "/path/to/checkpoint"
        assert result.total_steps == 100

    def test_train_result_repr(self):
        """Test TrainResult string representation."""
        from nemo_rl.api import TrainResult
        
        mock_trainer = MagicMock()
        result = TrainResult(
            trainer=mock_trainer,
            metrics={"loss": 0.5},
            total_steps=100,
        )
        
        repr_str = repr(result)
        assert "100" in repr_str
        assert "0.5" in repr_str


class TestFunctionalAPI:
    """Tests for the functional API helpers."""

    def test_list_algorithms(self):
        """Test list_algorithms() returns expected algorithms."""
        from nemo_rl.api import list_algorithms
        
        algorithms = list_algorithms()
        
        assert isinstance(algorithms, list)
        assert "grpo" in algorithms
        assert "sft" in algorithms
        # Should be sorted
        assert algorithms == sorted(algorithms)

    def test_get_algorithm_grpo(self):
        """Test get_algorithm() returns GRPOTrainer."""
        from nemo_rl.api import get_algorithm
        from nemo_rl.algorithms.grpo import GRPOTrainer
        
        trainer_class = get_algorithm("grpo")
        assert trainer_class == GRPOTrainer

    def test_get_algorithm_sft(self):
        """Test get_algorithm() returns SFTTrainer."""
        from nemo_rl.api import get_algorithm
        from nemo_rl.algorithms.sft import SFTTrainer
        
        trainer_class = get_algorithm("sft")
        assert trainer_class == SFTTrainer

    def test_get_algorithm_case_insensitive(self):
        """Test get_algorithm() is case insensitive."""
        from nemo_rl.api import get_algorithm
        
        assert get_algorithm("grpo") == get_algorithm("GRPO")
        assert get_algorithm("sft") == get_algorithm("SFT")

    def test_get_algorithm_invalid(self):
        """Test get_algorithm() raises for unknown algorithm."""
        from nemo_rl.api import get_algorithm
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm("not_an_algorithm")

    def test_create_trainer_grpo(self):
        """Test create_trainer() creates GRPOTrainer."""
        from nemo_rl.api import create_trainer
        from nemo_rl.algorithms.grpo import GRPOTrainer
        
        trainer = create_trainer(
            "grpo",
            model="Qwen/Qwen2.5-1.5B",
            learning_rate=1e-6,
        )
        
        assert isinstance(trainer, GRPOTrainer)

    def test_create_trainer_sft(self):
        """Test create_trainer() creates SFTTrainer."""
        from nemo_rl.api import create_trainer
        from nemo_rl.algorithms.sft import SFTTrainer
        
        trainer = create_trainer(
            "sft",
            model="Qwen/Qwen2.5-1.5B",
        )
        
        assert isinstance(trainer, SFTTrainer)


class TestConfigBuilding:
    """Tests for config building helpers."""

    def test_grpo_config_structure(self):
        """Test GRPO config is built correctly."""
        from nemo_rl.api.train import _build_grpo_config
        
        config = _build_grpo_config(
            model="test-model",
            learning_rate=1e-6,
            batch_size=32,
            max_steps=1000,
            max_epochs=1,
            num_generations_per_prompt=16,
            output_dir="./output",
        )
        
        assert config["policy"]["model_name"] == "test-model"
        assert config["policy"]["learning_rate"] == 1e-6
        assert config["grpo"]["num_prompts_per_step"] == 32
        assert config["grpo"]["num_generations_per_prompt"] == 16
        assert config["grpo"]["max_num_steps"] == 1000

    def test_sft_config_structure(self):
        """Test SFT config is built correctly."""
        from nemo_rl.api.train import _build_sft_config
        
        config = _build_sft_config(
            model="test-model",
            learning_rate=1e-5,
            batch_size=16,
            max_steps=500,
            max_epochs=2,
            output_dir="./output",
        )
        
        assert config["policy"]["model_name"] == "test-model"
        assert config["policy"]["learning_rate"] == 1e-5
        assert config["sft"]["batch_size"] == 16
        assert config["sft"]["max_num_steps"] == 500

    def test_dpo_config_structure(self):
        """Test DPO config is built correctly."""
        from nemo_rl.api.train import _build_dpo_config
        
        config = _build_dpo_config(
            model="test-model",
            learning_rate=5e-7,
            batch_size=8,
            max_steps=2000,
            max_epochs=1,
            output_dir="./output",
            beta=0.2,
        )
        
        assert config["policy"]["model_name"] == "test-model"
        assert config["dpo"]["beta"] == 0.2
        assert config["dpo"]["max_num_steps"] == 2000


class TestValidation:
    """Tests for input validation."""

    def test_validate_empty_model(self):
        """Test validation rejects empty model."""
        from nemo_rl.api.train import _validate_inputs
        
        with pytest.raises(ValueError, match="Model is required"):
            _validate_inputs("grpo", "", "dataset", lambda p, r: 1.0)

    def test_validate_none_dataset(self):
        """Test validation rejects None dataset."""
        from nemo_rl.api.train import _validate_inputs
        
        with pytest.raises(ValueError, match="Dataset is required"):
            _validate_inputs("grpo", "model", None, lambda p, r: 1.0)

    def test_validate_grpo_missing_reward(self):
        """Test validation rejects GRPO without reward_fn."""
        from nemo_rl.api.train import _validate_inputs
        
        with pytest.raises(ValueError, match="reward_fn is required"):
            _validate_inputs("grpo", "model", "dataset", None)

    def test_validate_sft_no_reward_needed(self):
        """Test validation accepts SFT without reward_fn."""
        from nemo_rl.api.train import _validate_inputs
        
        # Should not raise
        _validate_inputs("sft", "model", "dataset", None)


class TestModuleImports:
    """Tests for module-level imports."""

    def test_nemo_rl_train_lazy_import(self):
        """Test that train is lazily imported from nemo_rl."""
        import nemo_rl
        
        # Access train via __getattr__
        train_fn = nemo_rl.train
        
        assert train_fn is not None
        assert callable(train_fn)

    def test_nemo_rl_trainers_lazy_import(self):
        """Test that trainers are lazily imported."""
        import nemo_rl
        
        # These should be accessible
        assert nemo_rl.GRPOTrainer is not None
        assert nemo_rl.SFTTrainer is not None
        assert nemo_rl.BaseTrainer is not None

    def test_nemo_rl_api_helpers(self):
        """Test that API helpers are accessible from nemo_rl."""
        import nemo_rl
        
        assert callable(nemo_rl.list_algorithms)
        assert callable(nemo_rl.get_algorithm)
        assert callable(nemo_rl.create_trainer)


class TestFiveLineScript:
    """Tests to verify 5-line training scripts are possible."""

    def test_minimal_api_signature(self):
        """Test that minimal API has expected signature."""
        import inspect
        from nemo_rl.api import train
        
        sig = inspect.signature(train)
        params = sig.parameters
        
        # Required params
        assert "model" in params
        
        # Optional params with defaults
        assert "dataset" in params
        assert "reward_fn" in params
        assert "algorithm" in params
        assert "max_steps" in params
        
        # Check defaults
        assert params["algorithm"].default == "grpo"
        assert params["max_steps"].default == 1000
        assert params["batch_size"].default == 32

    def test_docstring_has_five_line_example(self):
        """Test that docstring shows 5-line example."""
        from nemo_rl.api import train
        
        doc = train.__doc__
        assert doc is not None
        assert "5-line" in doc.lower() or "minimal" in doc.lower()
