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
"""Default configurations for common use cases.

This module provides pre-configured defaults for common training scenarios,
reducing the amount of configuration needed for typical use cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_rl.config.cluster import ClusterConfig
from nemo_rl.config.generation import GenerationConfig, VLLMConfig
from nemo_rl.config.policy import (
    DTensorConfig,
    MegatronConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
)
from nemo_rl.config.training import (
    CheckpointingConfig,
    DataConfig,
    DPOConfig,
    GRPOConfig,
    LoggerConfig,
    SFTConfig,
)


def get_default_optimizer() -> OptimizerConfig:
    """Get default optimizer configuration.

    Returns AdamW with sensible defaults for LLM fine-tuning.

    Returns:
        OptimizerConfig with AdamW defaults.
    """
    return OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
    )


def get_default_scheduler(warmup_steps: int = 100) -> SchedulerConfig:
    """Get default scheduler configuration.

    Returns cosine scheduler with warmup.

    Args:
        warmup_steps: Number of warmup steps.

    Returns:
        SchedulerConfig with cosine schedule and warmup.
    """
    return SchedulerConfig(
        name="cosine_with_warmup",
        kwargs={
            "num_warmup_steps": warmup_steps,
        },
    )


def get_default_policy_config(
    model_name: str,
    precision: str = "bfloat16",
    tensor_parallel_size: int = 1,
    use_megatron: bool = False,
) -> PolicyConfig:
    """Get default policy configuration.

    Args:
        model_name: HuggingFace model name or path.
        precision: Training precision.
        tensor_parallel_size: Tensor parallel size.
        use_megatron: Whether to use Megatron backend.

    Returns:
        PolicyConfig with sensible defaults.
    """
    dtensor_cfg = DTensorConfig(
        enabled=not use_megatron,
        tensor_parallel_size=tensor_parallel_size,
        activation_checkpointing=True,
        cpu_offload=False,
    )

    megatron_cfg = MegatronConfig(
        enabled=use_megatron,
        tensor_model_parallel_size=tensor_parallel_size,
        activation_checkpointing=True,
    )

    return PolicyConfig(
        model_name=model_name,
        precision=precision,
        train_global_batch_size=32,
        train_micro_batch_size=4,
        max_total_sequence_length=4096,
        dtensor_cfg=dtensor_cfg,
        megatron_cfg=megatron_cfg,
        optimizer=get_default_optimizer(),
    )


def get_grpo_config_for_1b_model(
    model_name: str,
    num_gpus: int = 8,
) -> GRPOConfig:
    """Get GRPO configuration optimized for ~1B parameter models.

    Args:
        model_name: HuggingFace model name or path.
        num_gpus: Number of GPUs available.

    Returns:
        GRPOConfig optimized for 1B models.
    """
    return GRPOConfig(
        policy=PolicyConfig(
            model_name=model_name,
            precision="bfloat16",
            train_global_batch_size=32,
            train_micro_batch_size=8,
            max_total_sequence_length=4096,
            dtensor_cfg=DTensorConfig(
                enabled=True,
                tensor_parallel_size=1,
                activation_checkpointing=True,
            ),
        ),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=num_gpus,
        ),
        vllm=VLLMConfig(
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
        ),
        num_prompts_per_step=32,
        num_generations_per_prompt=16,
        max_num_epochs=1,
        max_num_steps=1000,
    )


def get_grpo_config_for_8b_model(
    model_name: str,
    num_nodes: int = 1,
    gpus_per_node: int = 8,
) -> GRPOConfig:
    """Get GRPO configuration optimized for ~8B parameter models.

    Args:
        model_name: HuggingFace model name or path.
        num_nodes: Number of nodes.
        gpus_per_node: GPUs per node.

    Returns:
        GRPOConfig optimized for 8B models.
    """
    return GRPOConfig(
        policy=PolicyConfig(
            model_name=model_name,
            precision="bfloat16",
            train_global_batch_size=64,
            train_micro_batch_size=4,
            max_total_sequence_length=4096,
            dtensor_cfg=DTensorConfig(
                enabled=True,
                tensor_parallel_size=2,
                activation_checkpointing=True,
                cpu_offload=True,
            ),
        ),
        cluster=ClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
        ),
        vllm=VLLMConfig(
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.80,
        ),
        num_prompts_per_step=64,
        num_generations_per_prompt=8,
        max_num_epochs=1,
        max_num_steps=2000,
    )


def get_sft_config(
    model_name: str,
    num_gpus: int = 8,
    tensor_parallel_size: int = 1,
) -> SFTConfig:
    """Get SFT configuration with sensible defaults.

    Args:
        model_name: HuggingFace model name or path.
        num_gpus: Number of GPUs available.
        tensor_parallel_size: Tensor parallel size.

    Returns:
        SFTConfig with sensible defaults.
    """
    return SFTConfig(
        policy=PolicyConfig(
            model_name=model_name,
            precision="bfloat16",
            train_global_batch_size=32,
            train_micro_batch_size=4,
            max_total_sequence_length=4096,
            dtensor_cfg=DTensorConfig(
                enabled=True,
                tensor_parallel_size=tensor_parallel_size,
                activation_checkpointing=True,
            ),
            optimizer=get_default_optimizer(),
        ),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=num_gpus,
        ),
        max_num_epochs=3,
        val_period=100,
    )


def get_dpo_config(
    model_name: str,
    num_gpus: int = 8,
    tensor_parallel_size: int = 1,
    beta: float = 0.1,
) -> DPOConfig:
    """Get DPO configuration with sensible defaults.

    Args:
        model_name: HuggingFace model name or path.
        num_gpus: Number of GPUs available.
        tensor_parallel_size: Tensor parallel size.
        beta: DPO beta parameter.

    Returns:
        DPOConfig with sensible defaults.
    """
    from nemo_rl.config.training import DPOLossConfig

    return DPOConfig(
        policy=PolicyConfig(
            model_name=model_name,
            precision="bfloat16",
            train_global_batch_size=32,
            train_micro_batch_size=4,
            max_total_sequence_length=4096,
            dtensor_cfg=DTensorConfig(
                enabled=True,
                tensor_parallel_size=tensor_parallel_size,
                activation_checkpointing=True,
            ),
            optimizer=get_default_optimizer(),
        ),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=num_gpus,
        ),
        loss_fn=DPOLossConfig(beta=beta),
        max_num_epochs=1,
        val_period=100,
    )


# Template registry for easy access
_TEMPLATES: dict[str, callable] = {
    "grpo_1b": get_grpo_config_for_1b_model,
    "grpo_8b": get_grpo_config_for_8b_model,
    "sft": get_sft_config,
    "dpo": get_dpo_config,
}


def load_template(template_name: str, **kwargs) -> GRPOConfig | SFTConfig | DPOConfig:
    """Load a configuration template by name.

    Args:
        template_name: Name of the template ('grpo_1b', 'grpo_8b', 'sft', 'dpo').
        **kwargs: Arguments passed to the template function.

    Returns:
        Configuration object from the template.

    Raises:
        ValueError: If template name is not recognized.
    """
    if template_name not in _TEMPLATES:
        available = ", ".join(_TEMPLATES.keys())
        raise ValueError(
            f"Unknown template '{template_name}'. Available templates: {available}"
        )

    return _TEMPLATES[template_name](**kwargs)


def list_templates() -> list[str]:
    """List available configuration templates.

    Returns:
        List of template names.
    """
    return list(_TEMPLATES.keys())


__all__ = [
    "get_default_optimizer",
    "get_default_scheduler",
    "get_default_policy_config",
    "get_grpo_config_for_1b_model",
    "get_grpo_config_for_8b_model",
    "get_sft_config",
    "get_dpo_config",
    "load_template",
    "list_templates",
]
