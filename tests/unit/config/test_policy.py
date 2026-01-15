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
"""Tests for policy configuration."""

import pytest

from nemo_rl.config.base import ConfigValidationError
from nemo_rl.config.policy import (
    DTensorConfig,
    DynamicBatchingConfig,
    LoRAConfig,
    MegatronConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
    SequencePackingConfig,
    TokenizerConfig,
    TrainingBackend,
)


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_values(self):
        """Test default values."""
        config = LoRAConfig()
        assert config.enabled is False
        assert config.dim == 8
        assert config.alpha == 16
        assert config.dropout == 0.0

    def test_enabled_lora(self):
        """Test enabled LoRA config."""
        config = LoRAConfig(
            enabled=True,
            target_modules=["q_proj", "v_proj"],
            dim=16,
            alpha=32,
        )
        assert config.enabled is True
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.dim == 16

    def test_invalid_dim(self):
        """Test invalid dim value."""
        with pytest.raises(ConfigValidationError):
            LoRAConfig(dim=0)

    def test_invalid_dropout(self):
        """Test invalid dropout value."""
        with pytest.raises(ConfigValidationError):
            LoRAConfig(dropout=1.5)


class TestTokenizerConfig:
    """Tests for TokenizerConfig."""

    def test_simple_tokenizer(self):
        """Test simple tokenizer config."""
        config = TokenizerConfig(name="gpt2")
        assert config.name == "gpt2"
        assert config.chat_template is None

    def test_with_chat_template(self):
        """Test tokenizer with chat template."""
        config = TokenizerConfig(
            name="meta-llama/Llama-3.1-8B-Instruct",
            chat_template="{% for msg in messages %}{{ msg.content }}{% endfor %}",
        )
        assert config.chat_template is not None


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_optimizer(self):
        """Test default optimizer config."""
        config = OptimizerConfig()
        assert config.name == "adamw"
        assert "lr" in config.kwargs

    def test_custom_optimizer(self):
        """Test custom optimizer config."""
        config = OptimizerConfig(
            name="sgd",
            kwargs={"lr": 0.01, "momentum": 0.9},
        )
        assert config.name == "sgd"
        assert config.kwargs["momentum"] == 0.9


class TestDTensorConfig:
    """Tests for DTensorConfig."""

    def test_default_values(self):
        """Test default values."""
        config = DTensorConfig()
        assert config.enabled is True
        assert config.tensor_parallel_size == 1
        assert config.activation_checkpointing is True

    def test_custom_parallel_size(self):
        """Test custom tensor parallel size."""
        config = DTensorConfig(tensor_parallel_size=4)
        assert config.tensor_parallel_size == 4

    def test_invalid_parallel_size(self):
        """Test invalid parallel size (not power of 2)."""
        with pytest.raises(ConfigValidationError):
            DTensorConfig(tensor_parallel_size=3)

    def test_parallel_size_power_of_two(self):
        """Test valid power of two parallel sizes."""
        for size in [1, 2, 4, 8, 16]:
            config = DTensorConfig(tensor_parallel_size=size)
            assert config.tensor_parallel_size == size


class TestMegatronConfig:
    """Tests for MegatronConfig."""

    def test_default_values(self):
        """Test default values."""
        config = MegatronConfig()
        assert config.enabled is False
        assert config.tensor_model_parallel_size == 1
        assert config.pipeline_model_parallel_size == 1

    def test_enabled_megatron(self):
        """Test enabled Megatron config."""
        config = MegatronConfig(
            enabled=True,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=2,
        )
        assert config.enabled is True
        assert config.tensor_model_parallel_size == 4
        assert config.pipeline_model_parallel_size == 2


class TestPolicyConfig:
    """Tests for PolicyConfig."""

    def test_minimal_config(self):
        """Test minimal required config."""
        config = PolicyConfig(model_name="gpt2")
        assert config.model_name == "gpt2"
        assert config.tokenizer is not None
        assert config.tokenizer.name == "gpt2"

    def test_with_backend_selection(self):
        """Test backend selection."""
        config = PolicyConfig(
            model_name="gpt2",
            backend=TrainingBackend.DTENSOR,
        )
        assert config.backend == TrainingBackend.DTENSOR
        assert config.dtensor_cfg.enabled is True

    def test_megatron_backend(self):
        """Test Megatron backend selection."""
        config = PolicyConfig(
            model_name="gpt2",
            backend=TrainingBackend.MEGATRON,
        )
        assert config.backend == TrainingBackend.MEGATRON
        assert config.megatron_cfg.enabled is True

    def test_custom_tokenizer(self):
        """Test custom tokenizer config."""
        config = PolicyConfig(
            model_name="gpt2",
            tokenizer=TokenizerConfig(name="custom-tokenizer"),
        )
        assert config.tokenizer.name == "custom-tokenizer"

    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        with pytest.raises(ConfigValidationError):
            PolicyConfig(model_name="gpt2", train_global_batch_size=0)

    def test_tensor_parallel_property(self):
        """Test tensor_parallel_size property."""
        config = PolicyConfig(
            model_name="gpt2",
            dtensor_cfg=DTensorConfig(tensor_parallel_size=4),
        )
        assert config.tensor_parallel_size == 4

    def test_pipeline_parallel_property(self):
        """Test pipeline_parallel_size property."""
        config = PolicyConfig(
            model_name="gpt2",
            backend=TrainingBackend.MEGATRON,
            megatron_cfg=MegatronConfig(
                enabled=True,
                pipeline_model_parallel_size=2,
            ),
        )
        assert config.pipeline_parallel_size == 2


class TestDynamicBatchingConfig:
    """Tests for DynamicBatchingConfig."""

    def test_disabled_by_default(self):
        """Test disabled by default."""
        config = DynamicBatchingConfig()
        assert config.enabled is False

    def test_enabled_config(self):
        """Test enabled config."""
        config = DynamicBatchingConfig(
            enabled=True,
            train_mb_tokens=16384,
        )
        assert config.enabled is True
        assert config.train_mb_tokens == 16384


class TestSequencePackingConfig:
    """Tests for SequencePackingConfig."""

    def test_disabled_by_default(self):
        """Test disabled by default."""
        config = SequencePackingConfig()
        assert config.enabled is False

    def test_custom_algorithm(self):
        """Test custom packing algorithm."""
        config = SequencePackingConfig(
            enabled=True,
            algorithm="best_fit",
        )
        assert config.algorithm == "best_fit"
