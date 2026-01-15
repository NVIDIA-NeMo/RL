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
"""Policy configuration for model training.

This module provides configuration classes for policy models including
training backends (DTensor, Megatron), optimizers, and schedulers.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from nemo_rl.config.base import BaseConfig


class TrainingBackend(str, Enum):
    """Supported training backends."""

    DTENSOR = "dtensor"
    MEGATRON = "megatron"


class PrecisionType(str, Enum):
    """Supported precision types."""

    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"


class LoRAConfig(BaseConfig):
    """Configuration for LoRA (Low-Rank Adaptation).

    Attributes:
        enabled: Whether LoRA is enabled.
        target_modules: List of module names to apply LoRA to.
        exclude_modules: List of module names to exclude from LoRA.
        dim: LoRA rank dimension.
        alpha: LoRA alpha parameter.
        dropout: Dropout rate for LoRA layers.
        dropout_position: Position of dropout ('pre' or 'post').
        lora_A_init: Initialization method for LoRA A matrix.
        match_all_linear: Whether to match all linear layers.
        use_triton: Whether to use Triton kernels.
    """

    enabled: bool = False
    target_modules: list[str] = Field(default_factory=list)
    exclude_modules: list[str] = Field(default_factory=list)
    match_all_linear: bool = False
    dim: Annotated[int, Field(gt=0)] = 8
    alpha: Annotated[int, Field(gt=0)] = 16
    dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init: str = "kaiming_uniform"
    use_triton: bool = False


class TokenizerConfig(BaseConfig):
    """Configuration for tokenizer.

    Attributes:
        name: Name or path of the tokenizer (HuggingFace model name or local path).
        chat_template: Optional custom chat template.
        chat_template_kwargs: Additional kwargs for apply_chat_template.
    """

    name: str
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class OptimizerConfig(BaseConfig):
    """Configuration for PyTorch optimizer.

    Attributes:
        name: Optimizer name (e.g., 'adamw', 'sgd', 'adam').
        kwargs: Additional kwargs passed to the optimizer constructor.
    """

    name: str = "adamw"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"lr": 1e-5, "weight_decay": 0.01}
    )


class SchedulerConfig(BaseConfig):
    """Configuration for learning rate scheduler.

    Attributes:
        name: Scheduler name (e.g., 'cosine', 'linear', 'constant').
        kwargs: Additional kwargs passed to the scheduler constructor.
    """

    name: str = "constant"
    kwargs: dict[str, Any] = Field(default_factory=dict)


class DynamicBatchingConfig(BaseConfig):
    """Configuration for dynamic batching.

    Dynamic batching improves performance by ensuring microbatches
    have sufficient tokens to maximize GPU utilization.

    Attributes:
        enabled: Whether dynamic batching is enabled.
        train_mb_tokens: Target token count per training microbatch.
        logprob_mb_tokens: Target token count per logprob microbatch.
        sequence_length_round: Round sequence length to this value.
    """

    enabled: bool = False
    train_mb_tokens: Annotated[int, Field(gt=0)] = 8192
    logprob_mb_tokens: Annotated[int, Field(gt=0)] | None = None
    sequence_length_round: Annotated[int, Field(gt=0)] = 64


class SequencePackingConfig(BaseConfig):
    """Configuration for sequence packing.

    Sequence packing combines multiple sequences into a single batch
    to improve GPU utilization.

    Attributes:
        enabled: Whether sequence packing is enabled.
        train_mb_tokens: Target token count for packed training batches.
        logprob_mb_tokens: Target token count for packed logprob batches.
        algorithm: Packing algorithm to use.
    """

    enabled: bool = False
    train_mb_tokens: Annotated[int, Field(gt=0)] = 16384
    logprob_mb_tokens: Annotated[int, Field(gt=0)] | None = None
    algorithm: str = "first_fit_decreasing"


class DTensorConfig(BaseConfig):
    """Configuration for DTensor (Distributed Tensor) training backend.

    DTensor provides automatic parallelization using PyTorch's DTensor API.

    Attributes:
        enabled: Whether DTensor backend is enabled.
        cpu_offload: Whether to offload optimizer states to CPU.
        sequence_parallel: Whether to enable sequence parallelism.
        activation_checkpointing: Whether to enable activation checkpointing.
        tensor_parallel_size: Tensor parallel size.
        context_parallel_size: Context parallel size (for long sequences).
        custom_parallel_plan: Path to custom parallel plan module.
        clear_cache_every_n_steps: Clear CUDA cache every N steps.
        lora_cfg: LoRA configuration.
        env_vars: Environment variables to set for workers.
    """

    enabled: bool = True
    cpu_offload: bool = False
    sequence_parallel: bool = False
    activation_checkpointing: bool = True
    tensor_parallel_size: Annotated[int, Field(gt=0)] = 1
    context_parallel_size: Annotated[int, Field(gt=0)] = 1
    custom_parallel_plan: str | None = None
    clear_cache_every_n_steps: Annotated[int, Field(gt=0)] | None = None
    lora_cfg: LoRAConfig = Field(default_factory=LoRAConfig)
    env_vars: dict[str, str] | None = None

    @field_validator("tensor_parallel_size", "context_parallel_size")
    @classmethod
    def validate_parallel_size(cls, v: int) -> int:
        """Validate parallel sizes are power of 2 or 1."""
        if v != 1 and (v & (v - 1)) != 0:
            raise ValueError(f"Parallel size must be 1 or a power of 2, got {v}")
        return v


class MegatronOptimizerConfig(BaseConfig):
    """Configuration for Megatron optimizer.

    Attributes:
        optimizer: Optimizer type ('adam', 'sgd').
        lr: Learning rate.
        min_lr: Minimum learning rate.
        weight_decay: Weight decay.
        bf16: Use bfloat16 precision.
        fp16: Use float16 precision.
        params_dtype: Parameter data type.
        adam_beta1: Adam beta1.
        adam_beta2: Adam beta2.
        adam_eps: Adam epsilon.
        sgd_momentum: SGD momentum.
        use_distributed_optimizer: Use distributed optimizer.
        use_precision_aware_optimizer: Use precision-aware optimizer.
        clip_grad: Gradient clipping norm.
        optimizer_cpu_offload: Offload optimizer to CPU.
        optimizer_offload_fraction: Fraction of optimizer to offload.
    """

    optimizer: str = "adam"
    lr: Annotated[float, Field(gt=0)] = 1e-5
    min_lr: Annotated[float, Field(ge=0)] = 1e-6
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    bf16: bool = True
    fp16: bool = False
    params_dtype: str = "bfloat16"
    adam_beta1: Annotated[float, Field(ge=0, lt=1)] = 0.9
    adam_beta2: Annotated[float, Field(ge=0, lt=1)] = 0.999
    adam_eps: Annotated[float, Field(gt=0)] = 1e-8
    sgd_momentum: Annotated[float, Field(ge=0, le=1)] = 0.0
    use_distributed_optimizer: bool = True
    use_precision_aware_optimizer: bool = False
    clip_grad: Annotated[float, Field(ge=0)] = 1.0
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: Annotated[float, Field(ge=0, le=1)] = 1.0


class MegatronSchedulerConfig(BaseConfig):
    """Configuration for Megatron learning rate scheduler.

    Attributes:
        lr_decay_style: LR decay style ('cosine', 'linear', 'constant').
        lr_warmup_iters: Number of warmup iterations.
        lr_warmup_init: Initial learning rate during warmup.
        lr_decay_iters: Number of decay iterations.
        start_weight_decay: Starting weight decay.
        end_weight_decay: Ending weight decay.
        weight_decay_incr_style: Weight decay increment style.
    """

    lr_decay_style: str = "cosine"
    lr_warmup_iters: Annotated[int, Field(ge=0)] = 0
    lr_warmup_init: Annotated[float, Field(ge=0)] = 0.0
    lr_decay_iters: Annotated[int, Field(ge=0)] | None = None
    start_weight_decay: Annotated[float, Field(ge=0)] = 0.01
    end_weight_decay: Annotated[float, Field(ge=0)] = 0.01
    weight_decay_incr_style: str = "constant"


class MegatronDDPConfig(BaseConfig):
    """Configuration for Megatron DDP (Distributed Data Parallel).

    Attributes:
        grad_reduce_in_fp32: Reduce gradients in FP32.
        overlap_grad_reduce: Overlap gradient reduction with backward.
        overlap_param_gather: Overlap parameter gathering.
        use_custom_fsdp: Use custom FSDP implementation.
        data_parallel_sharding_strategy: FSDP sharding strategy.
    """

    grad_reduce_in_fp32: bool = True
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    use_custom_fsdp: bool = False
    data_parallel_sharding_strategy: str = "FULL_SHARD"


class MegatronConfig(BaseConfig):
    """Configuration for Megatron training backend.

    Megatron provides optimized distributed training with tensor, pipeline,
    and expert parallelism.

    Attributes:
        enabled: Whether Megatron backend is enabled.
        activation_checkpointing: Enable activation checkpointing.
        tensor_model_parallel_size: Tensor parallel size.
        pipeline_model_parallel_size: Pipeline parallel size.
        context_parallel_size: Context parallel size.
        sequence_parallel: Enable sequence parallelism.
        expert_tensor_parallel_size: Expert tensor parallel size.
        expert_model_parallel_size: Expert model parallel size.
        pipeline_dtype: Pipeline data type.
        apply_rope_fusion: Apply RoPE fusion optimization.
        bias_activation_fusion: Apply bias-activation fusion.
        freeze_moe_router: Freeze MoE router weights.
        empty_unused_memory_level: Memory cleanup level.
        moe_per_layer_logging: Log per-layer MoE stats.
        defer_fp32_logits: Defer FP32 logits casting.
        force_overwrite_initial_ckpt: Force overwrite initial checkpoint.
        optimizer: Megatron optimizer config.
        scheduler: Megatron scheduler config.
        distributed_data_parallel_config: DDP config.
        env_vars: Environment variables for workers.
    """

    enabled: bool = False
    activation_checkpointing: bool = True
    tensor_model_parallel_size: Annotated[int, Field(gt=0)] = 1
    pipeline_model_parallel_size: Annotated[int, Field(gt=0)] = 1
    num_layers_in_first_pipeline_stage: int | None = None
    num_layers_in_last_pipeline_stage: int | None = None
    context_parallel_size: Annotated[int, Field(gt=0)] = 1
    sequence_parallel: bool = True
    expert_tensor_parallel_size: Annotated[int, Field(gt=0)] = 1
    expert_model_parallel_size: Annotated[int, Field(gt=0)] = 1
    pipeline_dtype: str = "bfloat16"
    apply_rope_fusion: bool = True
    bias_activation_fusion: bool = True
    freeze_moe_router: bool = False
    empty_unused_memory_level: Annotated[int, Field(ge=0, le=3)] = 1
    moe_per_layer_logging: bool = False
    defer_fp32_logits: bool = False
    force_overwrite_initial_ckpt: bool = False
    optimizer: MegatronOptimizerConfig = Field(default_factory=MegatronOptimizerConfig)
    scheduler: MegatronSchedulerConfig = Field(default_factory=MegatronSchedulerConfig)
    distributed_data_parallel_config: MegatronDDPConfig = Field(
        default_factory=MegatronDDPConfig
    )
    env_vars: dict[str, str] | None = None


class PolicyConfig(BaseConfig):
    """Configuration for policy model.

    This is the main configuration class for the model being trained,
    including backend selection, parallelism, and training parameters.

    Attributes:
        model_name: HuggingFace model name or path.
        tokenizer: Tokenizer configuration.
        precision: Training precision ('float32', 'float16', 'bfloat16').
        backend: Training backend ('dtensor' or 'megatron').
        train_global_batch_size: Global batch size for training.
        train_micro_batch_size: Micro batch size per GPU.
        logprob_batch_size: Batch size for log probability computation.
        logprob_chunk_size: Chunk size for log probability (None=disabled).
        generation_batch_size: Batch size for generation.
        max_total_sequence_length: Maximum sequence length.
        make_sequence_length_divisible_by: Round sequence length.
        max_grad_norm: Maximum gradient norm for clipping.
        refit_buffer_size_gb: Size of refit buffer in GB.
        dtensor_cfg: DTensor backend configuration.
        megatron_cfg: Megatron backend configuration.
        dynamic_batching: Dynamic batching configuration.
        sequence_packing: Sequence packing configuration.
        optimizer: Optimizer configuration.
        scheduler: Scheduler configuration.
        hf_config_overrides: Overrides for HuggingFace model config.
    """

    model_name: str
    tokenizer: TokenizerConfig | None = None
    precision: str = "bfloat16"
    backend: TrainingBackend = TrainingBackend.DTENSOR

    # Batch sizes
    train_global_batch_size: Annotated[int, Field(gt=0)] = 32
    train_micro_batch_size: Annotated[int, Field(gt=0)] = 4
    logprob_batch_size: Annotated[int, Field(gt=0)] | None = None
    logprob_chunk_size: Annotated[int, Field(gt=0)] | None = None
    generation_batch_size: Annotated[int, Field(gt=0)] | None = None

    # Sequence settings
    max_total_sequence_length: Annotated[int, Field(gt=0)] = 4096
    make_sequence_length_divisible_by: Annotated[int, Field(gt=0)] = 1

    # Training settings
    max_grad_norm: Annotated[float, Field(gt=0)] | None = 1.0
    refit_buffer_size_gb: Annotated[float, Field(ge=0)] = 0.0

    # Backend configurations
    dtensor_cfg: DTensorConfig = Field(default_factory=DTensorConfig)
    megatron_cfg: MegatronConfig = Field(default_factory=MegatronConfig)

    # Batching configurations
    dynamic_batching: DynamicBatchingConfig = Field(default_factory=DynamicBatchingConfig)
    sequence_packing: SequencePackingConfig = Field(default_factory=SequencePackingConfig)

    # Optimizer and scheduler
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | list[SchedulerConfig] | None = None

    # HuggingFace config overrides
    hf_config_overrides: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_backend_config(self) -> "PolicyConfig":
        """Ensure the selected backend configuration is enabled."""
        if self.backend == TrainingBackend.DTENSOR:
            if not self.dtensor_cfg.enabled:
                # Auto-enable DTensor if selected as backend
                object.__setattr__(
                    self,
                    "dtensor_cfg",
                    DTensorConfig(**{**self.dtensor_cfg.model_dump(), "enabled": True}),
                )
        elif self.backend == TrainingBackend.MEGATRON:
            if not self.megatron_cfg.enabled:
                # Auto-enable Megatron if selected as backend
                object.__setattr__(
                    self,
                    "megatron_cfg",
                    MegatronConfig(**{**self.megatron_cfg.model_dump(), "enabled": True}),
                )
        return self

    @model_validator(mode="after")
    def set_default_tokenizer(self) -> "PolicyConfig":
        """Set default tokenizer from model name if not specified."""
        if self.tokenizer is None:
            object.__setattr__(
                self, "tokenizer", TokenizerConfig(name=self.model_name)
            )
        return self

    @property
    def tensor_parallel_size(self) -> int:
        """Get tensor parallel size from active backend."""
        if self.backend == TrainingBackend.MEGATRON:
            return self.megatron_cfg.tensor_model_parallel_size
        return self.dtensor_cfg.tensor_parallel_size

    @property
    def pipeline_parallel_size(self) -> int:
        """Get pipeline parallel size (Megatron only, DTensor returns 1)."""
        if self.backend == TrainingBackend.MEGATRON:
            return self.megatron_cfg.pipeline_model_parallel_size
        return 1

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        precision: str = "bfloat16",
        tensor_parallel_size: int = 1,
        backend: str = "dtensor",
        **kwargs,
    ) -> "PolicyConfig":
        """Create a PolicyConfig from a pretrained model name.

        This is a convenience method that creates a PolicyConfig with
        sensible defaults inferred from the model name.

        Args:
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen2.5-1.5B").
            precision: Training precision ('float32', 'float16', 'bfloat16').
            tensor_parallel_size: Tensor parallel size for distributed training.
            backend: Training backend ('dtensor' or 'megatron').
            **kwargs: Additional config overrides.

        Returns:
            PolicyConfig configured for the model.

        Example:
            >>> policy = PolicyConfig.from_pretrained("Qwen/Qwen2.5-1.5B")
            >>> policy = PolicyConfig.from_pretrained(
            ...     "meta-llama/Llama-3.1-8B-Instruct",
            ...     tensor_parallel_size=2,
            ...     backend="megatron"
            ... )
        """
        # Determine backend enum
        backend_enum = (
            TrainingBackend.MEGATRON
            if backend.lower() == "megatron"
            else TrainingBackend.DTENSOR
        )

        # Build config dict with defaults
        config_dict = {
            "model_name": model_name,
            "precision": precision,
            "backend": backend_enum,
            **kwargs,
        }

        # Set tensor parallel size in the appropriate backend config
        if backend_enum == TrainingBackend.DTENSOR:
            config_dict.setdefault("dtensor_cfg", {})
            config_dict["dtensor_cfg"]["tensor_parallel_size"] = tensor_parallel_size
        else:
            config_dict.setdefault("megatron_cfg", {})
            config_dict["megatron_cfg"]["tensor_model_parallel_size"] = tensor_parallel_size

        return cls.model_validate(config_dict)
