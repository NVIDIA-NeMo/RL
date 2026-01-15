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
"""Generation configuration for inference and rollouts.

This module provides configuration classes for generation/inference,
including vLLM backend settings and sampling parameters.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator

from nemo_rl.config.base import BaseConfig


class GenerationBackendType(str, Enum):
    """Supported generation backends."""

    VLLM = "vllm"
    MEGATRON = "megatron"
    HUGGINGFACE = "huggingface"


class SamplingStrategy(str, Enum):
    """Sampling strategies for generation."""

    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"


class GenerationConfig(BaseConfig):
    """Configuration for text generation/inference.

    Defines sampling parameters and generation settings used during
    rollouts and inference.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        min_new_tokens: Minimum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Top-p (nucleus) sampling threshold.
        top_k: Top-k sampling threshold.
        repetition_penalty: Penalty for repeating tokens.
        stop_sequences: List of sequences that stop generation.
        do_sample: Whether to use sampling (vs greedy decoding).
        num_beams: Number of beams for beam search.
        seed: Random seed for reproducibility.
    """

    max_new_tokens: Annotated[int, Field(gt=0)] = 512
    min_new_tokens: Annotated[int, Field(ge=0)] = 0
    temperature: Annotated[float, Field(ge=0.0)] = 1.0
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    top_k: Annotated[int, Field(ge=0)] = 0  # 0 means disabled
    repetition_penalty: Annotated[float, Field(gt=0.0)] = 1.0
    stop_sequences: list[str] = Field(default_factory=list)
    do_sample: bool = True
    num_beams: Annotated[int, Field(gt=0)] = 1
    seed: int | None = None

    @model_validator(mode="after")
    def validate_sampling_settings(self) -> "GenerationConfig":
        """Validate sampling configuration consistency."""
        if not self.do_sample and self.temperature != 1.0:
            # When not sampling, temperature should be 1.0
            object.__setattr__(self, "temperature", 1.0)
        if self.temperature == 0.0:
            # Temperature 0 is effectively greedy
            object.__setattr__(self, "do_sample", False)
            object.__setattr__(self, "temperature", 1.0)
        return self

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Determine the sampling strategy from settings."""
        if not self.do_sample:
            return SamplingStrategy.GREEDY
        if self.num_beams > 1:
            return SamplingStrategy.BEAM_SEARCH
        return SamplingStrategy.SAMPLING


class VLLMQuantizationConfig(BaseConfig):
    """Configuration for vLLM quantization.

    Attributes:
        enabled: Whether quantization is enabled.
        method: Quantization method ('fp8', 'awq', 'gptq', etc.).
        kv_cache_dtype: Data type for KV cache.
    """

    enabled: bool = False
    method: str | None = None
    kv_cache_dtype: str = "auto"


class VLLMConfig(BaseConfig):
    """Configuration for vLLM generation backend.

    vLLM provides optimized inference with PagedAttention and
    continuous batching.

    Attributes:
        enabled: Whether vLLM backend is enabled.
        tensor_parallel_size: Tensor parallel size for vLLM.
        pipeline_parallel_size: Pipeline parallel size for vLLM.
        max_model_len: Maximum model sequence length.
        max_num_seqs: Maximum number of sequences in a batch.
        gpu_memory_utilization: Fraction of GPU memory to use.
        enforce_eager: Disable CUDA graph compilation.
        enable_prefix_caching: Enable automatic prefix caching.
        enable_chunked_prefill: Enable chunked prefill.
        max_num_batched_tokens: Maximum tokens per batch.
        seed: Random seed for reproducibility.
        quantization: Quantization configuration.
        generation: Default generation parameters.
        env_vars: Environment variables for vLLM workers.
    """

    enabled: bool = True
    tensor_parallel_size: Annotated[int, Field(gt=0)] = 1
    pipeline_parallel_size: Annotated[int, Field(gt=0)] = 1
    max_model_len: Annotated[int, Field(gt=0)] | None = None
    max_num_seqs: Annotated[int, Field(gt=0)] = 256
    gpu_memory_utilization: Annotated[float, Field(gt=0.0, le=1.0)] = 0.85
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: Annotated[int, Field(gt=0)] | None = None
    seed: int | None = None
    quantization: VLLMQuantizationConfig = Field(default_factory=VLLMQuantizationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    env_vars: dict[str, str] | None = None

    @field_validator("tensor_parallel_size", "pipeline_parallel_size")
    @classmethod
    def validate_parallel_size(cls, v: int) -> int:
        """Validate parallel sizes are power of 2 or 1."""
        if v != 1 and (v & (v - 1)) != 0:
            raise ValueError(f"Parallel size must be 1 or a power of 2, got {v}")
        return v

    @property
    def total_gpus(self) -> int:
        """Total GPUs required for vLLM."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    def to_vllm_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for vLLM LLM constructor.

        Returns:
            Dictionary of kwargs for vLLM initialization.
        """
        kwargs = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
        }

        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.quantization.enabled and self.quantization.method:
            kwargs["quantization"] = self.quantization.method
            kwargs["kv_cache_dtype"] = self.quantization.kv_cache_dtype

        return kwargs


class MegatronInferenceConfig(BaseConfig):
    """Configuration for Megatron-based inference.

    Attributes:
        enabled: Whether Megatron inference is enabled.
        tensor_parallel_size: Tensor parallel size.
        pipeline_parallel_size: Pipeline parallel size.
        micro_batch_size: Micro batch size for inference.
    """

    enabled: bool = False
    tensor_parallel_size: Annotated[int, Field(gt=0)] = 1
    pipeline_parallel_size: Annotated[int, Field(gt=0)] = 1
    micro_batch_size: Annotated[int, Field(gt=0)] = 1
