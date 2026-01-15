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
"""Tests for generation configuration."""

import pytest

from nemo_rl.config.base import ConfigValidationError
from nemo_rl.config.generation import (
    GenerationConfig,
    SamplingStrategy,
    VLLMConfig,
    VLLMQuantizationConfig,
)


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_values(self):
        """Test default values."""
        config = GenerationConfig()
        assert config.max_new_tokens == 512
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.do_sample is True

    def test_custom_values(self):
        """Test custom values."""
        config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50

    def test_invalid_max_new_tokens(self):
        """Test invalid max_new_tokens."""
        with pytest.raises(ConfigValidationError):
            GenerationConfig(max_new_tokens=0)

    def test_invalid_temperature(self):
        """Test invalid temperature."""
        with pytest.raises(ConfigValidationError):
            GenerationConfig(temperature=-0.1)

    def test_invalid_top_p(self):
        """Test invalid top_p."""
        with pytest.raises(ConfigValidationError):
            GenerationConfig(top_p=1.5)

    def test_temperature_zero_greedy(self):
        """Test that temperature=0 is converted to greedy."""
        config = GenerationConfig(temperature=0.0)
        # Temperature 0 should be converted to greedy (do_sample=False)
        assert config.do_sample is False

    def test_sampling_strategy_greedy(self):
        """Test sampling strategy detection - greedy."""
        config = GenerationConfig(do_sample=False)
        assert config.sampling_strategy == SamplingStrategy.GREEDY

    def test_sampling_strategy_sampling(self):
        """Test sampling strategy detection - sampling."""
        config = GenerationConfig(do_sample=True, num_beams=1)
        assert config.sampling_strategy == SamplingStrategy.SAMPLING

    def test_sampling_strategy_beam_search(self):
        """Test sampling strategy detection - beam search."""
        config = GenerationConfig(do_sample=True, num_beams=4)
        assert config.sampling_strategy == SamplingStrategy.BEAM_SEARCH

    def test_stop_sequences(self):
        """Test stop sequences."""
        config = GenerationConfig(
            stop_sequences=["</s>", "<|endoftext|>"],
        )
        assert len(config.stop_sequences) == 2


class TestVLLMQuantizationConfig:
    """Tests for VLLMQuantizationConfig."""

    def test_default_disabled(self):
        """Test disabled by default."""
        config = VLLMQuantizationConfig()
        assert config.enabled is False
        assert config.method is None

    def test_enabled_fp8(self):
        """Test enabled FP8 quantization."""
        config = VLLMQuantizationConfig(
            enabled=True,
            method="fp8",
            kv_cache_dtype="fp8",
        )
        assert config.enabled is True
        assert config.method == "fp8"


class TestVLLMConfig:
    """Tests for VLLMConfig."""

    def test_default_values(self):
        """Test default values."""
        config = VLLMConfig()
        assert config.enabled is True
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85

    def test_custom_parallel_size(self):
        """Test custom tensor parallel size."""
        config = VLLMConfig(tensor_parallel_size=4)
        assert config.tensor_parallel_size == 4

    def test_invalid_parallel_size(self):
        """Test invalid tensor parallel size (not power of 2)."""
        with pytest.raises(ConfigValidationError):
            VLLMConfig(tensor_parallel_size=3)

    def test_valid_parallel_sizes(self):
        """Test valid power of two parallel sizes."""
        for size in [1, 2, 4, 8]:
            config = VLLMConfig(tensor_parallel_size=size)
            assert config.tensor_parallel_size == size

    def test_invalid_gpu_memory_utilization(self):
        """Test invalid GPU memory utilization."""
        with pytest.raises(ConfigValidationError):
            VLLMConfig(gpu_memory_utilization=1.5)

    def test_total_gpus_property(self):
        """Test total_gpus property."""
        config = VLLMConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        )
        assert config.total_gpus == 8

    def test_to_vllm_kwargs(self):
        """Test conversion to vLLM kwargs."""
        config = VLLMConfig(
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            seed=42,
        )
        kwargs = config.to_vllm_kwargs()

        assert kwargs["tensor_parallel_size"] == 2
        assert kwargs["max_model_len"] == 4096
        assert kwargs["gpu_memory_utilization"] == 0.9
        assert kwargs["seed"] == 42

    def test_to_vllm_kwargs_with_quantization(self):
        """Test conversion to vLLM kwargs with quantization."""
        config = VLLMConfig(
            quantization=VLLMQuantizationConfig(
                enabled=True,
                method="fp8",
                kv_cache_dtype="fp8",
            ),
        )
        kwargs = config.to_vllm_kwargs()

        assert kwargs["quantization"] == "fp8"
        assert kwargs["kv_cache_dtype"] == "fp8"
