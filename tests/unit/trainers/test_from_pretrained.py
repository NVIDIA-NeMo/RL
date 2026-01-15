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
"""Tests for Trainer.from_pretrained() pattern."""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock


class TestBaseTrainerFromPretrained:
    """Tests for BaseTrainer.from_pretrained() method."""

    def test_from_pretrained_exists(self):
        """Test that from_pretrained is a class method."""
        from nemo_rl.trainers.base import BaseTrainer
        
        assert hasattr(BaseTrainer, "from_pretrained")
        assert callable(BaseTrainer.from_pretrained)

    def test_load_local_model_config(self):
        """Test loading config from local path."""
        from nemo_rl.trainers.base import BaseTrainer
        
        # Create a temporary config.json
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "hidden_size": 2048,
                    "num_hidden_layers": 24,
                    "vocab_size": 50000,
                    "max_position_embeddings": 4096,
                }, f)
            
            # Load the config
            config = BaseTrainer._load_local_model_config(tmpdir)
            
            assert config["hidden_size"] == 2048
            assert config["num_layers"] == 24
            assert config["vocab_size"] == 50000
            assert config["max_seq_length"] == 4096

    def test_load_local_missing_config(self):
        """Test handling missing config.json."""
        from nemo_rl.trainers.base import BaseTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # No config.json file
            config = BaseTrainer._load_local_model_config(tmpdir)
            assert config == {}

    def test_extract_config_from_hf(self):
        """Test extracting config from HuggingFace format."""
        from nemo_rl.trainers.base import BaseTrainer
        
        hf_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 8192,
            "intermediate_size": 11008,  # Should be ignored
            "num_attention_heads": 32,  # Should be ignored
        }
        
        config = BaseTrainer._extract_config_from_hf(hf_config)
        
        assert config["hidden_size"] == 4096
        assert config["num_layers"] == 32
        assert config["vocab_size"] == 32000
        assert config["max_seq_length"] == 8192
        assert "intermediate_size" not in config
        assert "num_attention_heads" not in config

    def test_build_config_structure(self):
        """Test that built config has expected structure."""
        from nemo_rl.trainers.base import BaseTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal config
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"hidden_size": 1024}, f)
            
            config = BaseTrainer._build_config_from_pretrained(
                tmpdir,
                learning_rate=5e-6,
            )
            
            assert "policy" in config
            assert config["policy"]["model_name"] == tmpdir
            assert config["policy"]["learning_rate"] == 5e-6
            assert "checkpointing" in config
            assert "logger" in config


class TestGRPOTrainerFromPretrained:
    """Tests for GRPOTrainer.from_pretrained() method."""

    def test_grpo_from_pretrained_creates_trainer(self):
        """Test that from_pretrained creates a GRPOTrainer."""
        from nemo_rl.algorithms.grpo import GRPOTrainer
        
        trainer = GRPOTrainer.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            num_prompts_per_step=16,
        )
        
        assert isinstance(trainer, GRPOTrainer)

    def test_grpo_from_pretrained_config_has_grpo_section(self):
        """Test that GRPO config is properly created."""
        from nemo_rl.algorithms.grpo import GRPOTrainer
        
        config = GRPOTrainer._build_config_from_pretrained(
            "test-model",
            num_prompts_per_step=64,
            num_generations_per_prompt=8,
            normalize_rewards=False,
        )
        
        assert "grpo" in config
        assert config["grpo"]["num_prompts_per_step"] == 64
        assert config["grpo"]["num_generations_per_prompt"] == 8
        assert config["grpo"]["normalize_rewards"] is False

    def test_grpo_default_values(self):
        """Test GRPO default configuration values."""
        from nemo_rl.algorithms.grpo import GRPOTrainer
        
        config = GRPOTrainer._build_config_from_pretrained("test-model")
        
        # Check defaults
        assert config["grpo"]["num_prompts_per_step"] == 32
        assert config["grpo"]["num_generations_per_prompt"] == 16
        assert config["grpo"]["normalize_rewards"] is True
        assert config["grpo"]["use_leave_one_out_baseline"] is True


class TestSFTTrainerFromPretrained:
    """Tests for SFTTrainer.from_pretrained() method."""

    def test_sft_from_pretrained_creates_trainer(self):
        """Test that from_pretrained creates an SFTTrainer."""
        from nemo_rl.algorithms.sft import SFTTrainer
        
        trainer = SFTTrainer.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            batch_size=16,
        )
        
        assert isinstance(trainer, SFTTrainer)

    def test_sft_from_pretrained_config_has_sft_section(self):
        """Test that SFT config is properly created."""
        from nemo_rl.algorithms.sft import SFTTrainer
        
        config = SFTTrainer._build_config_from_pretrained(
            "test-model",
            batch_size=64,
            max_steps=5000,
        )
        
        assert "sft" in config
        assert config["sft"]["batch_size"] == 64
        assert config["sft"]["max_num_steps"] == 5000

    def test_sft_default_values(self):
        """Test SFT default configuration values."""
        from nemo_rl.algorithms.sft import SFTTrainer
        
        config = SFTTrainer._build_config_from_pretrained("test-model")
        
        # Check defaults
        assert config["sft"]["batch_size"] == 32
        assert config["sft"]["max_num_steps"] == 1000
        assert config["sft"]["val_period"] == 100


class TestDPOTrainerFromPretrained:
    """Tests for DPOTrainer.from_pretrained() method."""

    def test_dpo_from_pretrained_creates_trainer(self):
        """Test that from_pretrained creates a DPOTrainer."""
        from nemo_rl.algorithms.dpo import DPOTrainer
        
        trainer = DPOTrainer.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            beta=0.2,
        )
        
        assert isinstance(trainer, DPOTrainer)

    def test_dpo_from_pretrained_config_has_dpo_section(self):
        """Test that DPO config is properly created."""
        from nemo_rl.algorithms.dpo import DPOTrainer
        
        config = DPOTrainer._build_config_from_pretrained(
            "test-model",
            beta=0.2,
            batch_size=16,
        )
        
        assert "dpo" in config
        assert config["dpo"]["beta"] == 0.2
        assert config["dpo"]["batch_size"] == 16

    def test_dpo_default_values(self):
        """Test DPO default configuration values."""
        from nemo_rl.algorithms.dpo import DPOTrainer
        
        config = DPOTrainer._build_config_from_pretrained("test-model")
        
        # Check defaults
        assert config["dpo"]["beta"] == 0.1
        assert config["dpo"]["batch_size"] == 8
        assert config["dpo"]["reference_free"] is False


class TestFromPretrainedIntegration:
    """Integration tests for from_pretrained pattern."""

    def test_trainer_from_pretrained_can_be_used(self):
        """Test that trainer from from_pretrained has expected methods."""
        from nemo_rl.algorithms.grpo import GRPOTrainer
        
        trainer = GRPOTrainer.from_pretrained("test-model")
        
        # Should have standard trainer methods
        assert hasattr(trainer, "fit")
        assert hasattr(trainer, "setup")
        assert hasattr(trainer, "validate")
        assert callable(trainer.fit)

    def test_nemo_rl_exposes_trainers_with_from_pretrained(self):
        """Test that trainers are accessible from nemo_rl namespace."""
        import nemo_rl
        
        # These should work
        grpo = nemo_rl.GRPOTrainer
        sft = nemo_rl.SFTTrainer
        
        assert hasattr(grpo, "from_pretrained")
        assert hasattr(sft, "from_pretrained")
