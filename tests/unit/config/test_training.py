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
"""Tests for training configuration."""

import tempfile
from pathlib import Path

import pytest

from nemo_rl.config.base import ConfigValidationError
from nemo_rl.config.cluster import ClusterConfig
from nemo_rl.config.policy import PolicyConfig
from nemo_rl.config.training import (
    CheckpointingConfig,
    DataConfig,
    DPOConfig,
    DPOLossConfig,
    GRPOConfig,
    LoggerConfig,
    RewardScalingConfig,
    RewardShapingConfig,
    SFTConfig,
)


class TestLoggerConfig:
    """Tests for LoggerConfig."""

    def test_default_values(self):
        """Test default values."""
        config = LoggerConfig()
        assert config.wandb_enabled is False
        assert config.tensorboard_enabled is True
        assert config.log_interval == 10

    def test_wandb_enabled(self):
        """Test W&B enabled config."""
        config = LoggerConfig(
            wandb_enabled=True,
            wandb_project="test-project",
            wandb_entity="test-entity",
        )
        assert config.wandb_enabled is True
        assert config.wandb_project == "test-project"


class TestCheckpointingConfig:
    """Tests for CheckpointingConfig."""

    def test_default_values(self):
        """Test default values."""
        config = CheckpointingConfig()
        assert config.enabled is True
        assert config.save_period == 100
        assert config.keep_top_k == 5

    def test_custom_checkpoint_dir(self):
        """Test custom checkpoint directory."""
        config = CheckpointingConfig(checkpoint_dir="/custom/path")
        assert config.checkpoint_dir == "/custom/path"

    def test_invalid_save_period(self):
        """Test invalid save period."""
        with pytest.raises(ConfigValidationError):
            CheckpointingConfig(save_period=0)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default values."""
        config = DataConfig()
        assert config.num_workers == 4
        assert config.shuffle is True
        assert config.seed == 42

    def test_custom_paths(self):
        """Test custom data paths."""
        config = DataConfig(
            train_path="/data/train.json",
            val_path="/data/val.json",
        )
        assert config.train_path == "/data/train.json"
        assert config.val_path == "/data/val.json"


class TestRewardShapingConfig:
    """Tests for RewardShapingConfig."""

    def test_disabled_by_default(self):
        """Test disabled by default."""
        config = RewardShapingConfig()
        assert config.enabled is False
        assert config.kl_penalty_coeff == 0.0

    def test_enabled_with_kl_penalty(self):
        """Test enabled with KL penalty."""
        config = RewardShapingConfig(
            enabled=True,
            kl_penalty_coeff=0.1,
        )
        assert config.kl_penalty_coeff == 0.1


class TestRewardScalingConfig:
    """Tests for RewardScalingConfig."""

    def test_default_values(self):
        """Test default values."""
        config = RewardScalingConfig()
        assert config.enabled is False
        assert config.source_min == 0.0
        assert config.source_max == 1.0

    def test_custom_scaling(self):
        """Test custom scaling range."""
        config = RewardScalingConfig(
            enabled=True,
            source_min=-1.0,
            source_max=1.0,
            target_min=0.0,
            target_max=10.0,
        )
        assert config.source_min == -1.0
        assert config.target_max == 10.0


class TestGRPOConfig:
    """Tests for GRPOConfig."""

    def test_minimal_config(self):
        """Test minimal GRPO config."""
        config = GRPOConfig(
            policy=PolicyConfig(model_name="gpt2"),
        )
        assert config.policy.model_name == "gpt2"
        assert config.num_prompts_per_step == 32
        assert config.num_generations_per_prompt == 16

    def test_custom_training_params(self):
        """Test custom training parameters."""
        config = GRPOConfig(
            policy=PolicyConfig(model_name="gpt2"),
            num_prompts_per_step=64,
            num_generations_per_prompt=8,
            max_num_steps=5000,
        )
        assert config.num_prompts_per_step == 64
        assert config.num_generations_per_prompt == 8
        assert config.max_num_steps == 5000

    def test_with_cluster_config(self):
        """Test with cluster configuration."""
        config = GRPOConfig(
            policy=PolicyConfig(model_name="gpt2"),
            cluster=ClusterConfig(num_nodes=2, gpus_per_node=8),
        )
        assert config.cluster.num_nodes == 2
        assert config.cluster.gpus_per_node == 8

    def test_invalid_num_prompts(self):
        """Test invalid num_prompts_per_step."""
        with pytest.raises(ConfigValidationError):
            GRPOConfig(
                policy=PolicyConfig(model_name="gpt2"),
                num_prompts_per_step=0,
            )

    def test_invalid_num_generations(self):
        """Test invalid num_generations_per_prompt."""
        with pytest.raises(ConfigValidationError):
            GRPOConfig(
                policy=PolicyConfig(model_name="gpt2"),
                num_generations_per_prompt=0,
            )

    def test_yaml_roundtrip(self):
        """Test YAML save and load roundtrip."""
        original = GRPOConfig(
            policy=PolicyConfig(model_name="gpt2"),
            num_prompts_per_step=64,
            seed=123,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "grpo_config.yaml"
            original.to_yaml(path)

            loaded = GRPOConfig.from_yaml(path)
            assert loaded.policy.model_name == "gpt2"
            assert loaded.num_prompts_per_step == 64
            assert loaded.seed == 123


class TestSFTConfig:
    """Tests for SFTConfig."""

    def test_minimal_config(self):
        """Test minimal SFT config."""
        config = SFTConfig(
            policy=PolicyConfig(model_name="gpt2"),
        )
        assert config.policy.model_name == "gpt2"
        assert config.max_num_epochs == 1

    def test_custom_training_params(self):
        """Test custom training parameters."""
        config = SFTConfig(
            policy=PolicyConfig(model_name="gpt2"),
            max_num_epochs=3,
            val_period=50,
        )
        assert config.max_num_epochs == 3
        assert config.val_period == 50


class TestDPOConfig:
    """Tests for DPOConfig."""

    def test_minimal_config(self):
        """Test minimal DPO config."""
        config = DPOConfig(
            policy=PolicyConfig(model_name="gpt2"),
        )
        assert config.policy.model_name == "gpt2"
        assert config.loss_fn.beta == 0.1

    def test_custom_loss_fn(self):
        """Test custom loss function config."""
        config = DPOConfig(
            policy=PolicyConfig(model_name="gpt2"),
            loss_fn=DPOLossConfig(beta=0.5, label_smoothing=0.1),
        )
        assert config.loss_fn.beta == 0.5
        assert config.loss_fn.label_smoothing == 0.1


class TestDPOLossConfig:
    """Tests for DPOLossConfig."""

    def test_default_values(self):
        """Test default values."""
        config = DPOLossConfig()
        assert config.beta == 0.1
        assert config.label_smoothing == 0.0
        assert config.reference_free is False

    def test_invalid_beta(self):
        """Test invalid beta value."""
        with pytest.raises(ConfigValidationError):
            DPOLossConfig(beta=0)

    def test_invalid_label_smoothing(self):
        """Test invalid label smoothing value."""
        with pytest.raises(ConfigValidationError):
            DPOLossConfig(label_smoothing=1.5)
