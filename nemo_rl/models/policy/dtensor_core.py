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

"""Core functions for DTensor policy worker initialization."""

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from nemo_automodel import NeMoAutoModelForSequenceClassification
from nemo_automodel._transformers.utils import sliding_window_overwrite
from nemo_automodel.components.config.loader import _resolve_target
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.utils import TRANSFORMERS_CACHE

from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import configure_dynamo_cache, resolve_model_class


@dataclass
class ValidatedState:
    """Configuration values validated and derived from PolicyConfig."""

    is_vlm: bool
    is_generation_colocated: Optional[bool]
    dtype: torch.dtype
    cpu_offload: bool
    max_grad_norm: float
    enable_seq_packing: bool
    attn_impl: Optional[str]
    model_config: AutoConfig
    allow_flash_attn_args: bool
    is_reward_model: bool
    model_class: type
    hf_config_overrides: dict[str, Any]
    tp_size: int
    cp_size: int
    ep_size: int
    dp_size: Optional[int]
    sequence_parallel_enabled: bool


@dataclass
class DistributedState:
    """Distributed environment state."""

    rank: int
    world_size: int
    device_mesh: Any
    dp_cp_mesh: Any
    dp_mesh: Any
    tp_mesh: Any
    cp_mesh: Any
    moe_mesh: Optional[Any]
    dp_size: int
    tp_size: int
    cp_size: int
    manager: FSDP2Manager


@dataclass
class ModelAndOptimizerState:
    """Model and optimizer state."""

    model: torch.nn.Module
    model_state_dict_keys: list[str]
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[Any]
    reference_model_state_dict: Optional[dict[str, torch.Tensor]]


def validate_and_set_config(
    config: PolicyConfig,
    processor: Optional[AutoProcessor],
    rank: int,
) -> ValidatedState:
    """Validate configuration and set up derived configuration values.

    Args:
        config: Policy configuration
        processor: Optional processor for VLM models
        rank: Distributed rank

    Returns:
        ValidatedState: Dataclass containing validated configuration values

    Raises:
        ValueError: If configuration is invalid
        NotImplementedError: If unsupported configuration is requested
        RuntimeError: If known broken configuration is detected
    """
    # Set basic configuration
    is_vlm = processor is not None
    is_generation_colocated = None
    if "generation" in config and config["generation"] is not None:
        is_generation_colocated = config["generation"]["colocated"]["enabled"]

    # Set NCCL environment variable
    if not is_generation_colocated:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

    # Configure dynamo cache
    configure_dynamo_cache()

    # Parse precision
    precision_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    precision = config["precision"]
    if precision not in precision_map:
        raise ValueError(f"Unknown precision: {precision}")
    dtype = precision_map[precision]

    # Get other configuration values
    cpu_offload = config["dtensor_cfg"]["cpu_offload"]
    max_grad_norm = config["max_grad_norm"]
    enable_seq_packing = config["sequence_packing"]["enabled"]
    model_name = config["model_name"]

    # Validate sequence packing
    if enable_seq_packing:
        if is_vlm:
            raise ValueError(
                "Sequence packing is not supported for VLM models. "
                "Please set policy.sequence_packing.enabled = False to train VLM models."
            )
        print(f"[Rank {rank}] Sequence packing is enabled for model {model_name}")
        print(f"[Rank {rank}] Using FlashAttention2 for sequence packing")

    # Get HF config overrides
    hf_config_overrides = config.get("hf_config_overrides", {}) or {}

    # Determine attention implementation
    cp_size_cfg = config["dtensor_cfg"]["context_parallel_size"]
    attn_impl = (
        "flash_attention_2"
        if (enable_seq_packing and cp_size_cfg == 1)
        else ("sdpa" if cp_size_cfg > 1 else None)
    )

    # Load model config
    model_config = AutoConfig.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Always load in float32 for master weights
        trust_remote_code=True,
        **sliding_window_overwrite(model_name),
        attn_implementation=attn_impl,
        **hf_config_overrides,
    )

    # Check if model supports flash attention args
    allow_flash_attn_args = True
    if (
        model_config.architectures[0] == "DeciLMForCausalLM"
        and model_config.model_type == "nemotron-nas"
    ):
        allow_flash_attn_args = False

    # Determine if reward model
    is_reward_model = (
        "reward_model_cfg" in config and config["reward_model_cfg"]["enabled"]
    )

    if is_reward_model:
        # Validate reward model configuration
        if enable_seq_packing:
            raise NotImplementedError(
                "Sequence packing is not supported for reward models"
            )

        rm_type = config["reward_model_cfg"]["reward_model_type"]
        if rm_type == "bradley_terry":
            model_class = NeMoAutoModelForSequenceClassification
            if model_config.num_labels != 1:
                print(
                    "model_config.num_labels is not 1. Setting it to 1 since this value is used as the out_features "
                    "for the linear head of Bradley-Terry reward models."
                )
                model_config.num_labels = 1
        else:
            raise ValueError(f"Unknown reward model type: {rm_type}")
    else:
        model_class = resolve_model_class(model_config.model_type)

    # Get parallelization sizes
    tp_size = config["dtensor_cfg"].get("tensor_parallel_size", 1)
    cp_size = config["dtensor_cfg"].get("context_parallel_size", 1)
    ep_size = config["dtensor_cfg"].get("expert_parallel_size", 1)
    dp_size = config["dtensor_cfg"].get("data_parallel_size", None)
    sequence_parallel_enabled = config["dtensor_cfg"]["sequence_parallel"]

    # Validate parallelization configuration
    if cp_size > 1 and enable_seq_packing:
        raise ValueError(
            "Context parallel is not supported for sequence packing. "
            "Refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
        )

    if sequence_parallel_enabled and tp_size == 1:
        print(
            "[WARNING]: sequence_parallel=True, but tp_size=1 which has no effect. "
            "Enable tp_size > 1 to use sequence parallelism."
        )
    elif sequence_parallel_enabled and tp_size > 1:
        raise RuntimeError(
            "Sequence parallel + tp_size >1 is currently broken in torch==2.8.0. "
            "See https://github.com/NVIDIA-NeMo/Automodel/issues/652 for more details."
        )

    return ValidatedState(
        is_vlm=is_vlm,
        is_generation_colocated=is_generation_colocated,
        dtype=dtype,
        cpu_offload=cpu_offload,
        max_grad_norm=max_grad_norm,
        enable_seq_packing=enable_seq_packing,
        attn_impl=attn_impl,
        model_config=model_config,
        allow_flash_attn_args=allow_flash_attn_args,
        is_reward_model=is_reward_model,
        model_class=model_class,
        hf_config_overrides=hf_config_overrides,
        tp_size=tp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        sequence_parallel_enabled=sequence_parallel_enabled,
    )


def setup_distributed(
    config: PolicyConfig,
    validated_state: ValidatedState,
) -> DistributedState:
    """Set up distributed environment and create device mesh.

    Args:
        config: Policy configuration
        validated_state: ValidatedState from validate_and_set_config

    Returns:
        DistributedState: Dataclass containing distributed environment state
    """
    # Initialize process group
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Extract validated config values
    dtype = validated_state.dtype
    cpu_offload = validated_state.cpu_offload
    tp_size = validated_state.tp_size
    cp_size = validated_state.cp_size
    ep_size = validated_state.ep_size
    dp_size = validated_state.dp_size
    sequence_parallel_enabled = validated_state.sequence_parallel_enabled

    # Create FSDP2 manager
    manager = FSDP2Manager(
        dp_size=dp_size,
        dp_replicate_size=1,
        tp_size=tp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        pp_size=1,
        sequence_parallel=sequence_parallel_enabled,
        use_hf_tp_plan=config["dtensor_cfg"].get("use_hf_tp_plan", False),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        ),
        offload_policy=CPUOffloadPolicy(pin_memory=False) if cpu_offload else None,
        backend="nccl",
        world_size=world_size,
        activation_checkpointing=config["dtensor_cfg"]["activation_checkpointing"],
    )

    # Store mesh references
    device_mesh = manager.device_mesh
    dp_cp_mesh = device_mesh["dp_cp"]
    dp_mesh = device_mesh["dp"]
    tp_mesh = device_mesh["tp"]
    cp_mesh = device_mesh["cp"]
    moe_mesh = getattr(manager, "moe_mesh", None)

    return DistributedState(
        rank=rank,
        world_size=world_size,
        device_mesh=device_mesh,
        dp_cp_mesh=dp_cp_mesh,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
        cp_mesh=cp_mesh,
        moe_mesh=moe_mesh,
        dp_size=manager.dp_size,
        tp_size=manager.tp_size,
        cp_size=manager.cp_size,
        manager=manager,
    )


def setup_model_and_optimizer(
    config: PolicyConfig,
    tokenizer: AutoTokenizer,
    validated_state: ValidatedState,
    distributed_state: DistributedState,
    worker_instance: Any,
    init_optimizer: bool = True,
    init_reference_model: bool = True,
) -> ModelAndOptimizerState:
    """Initialize model, load weights, and set up optimizer.

    Args:
        config: Policy configuration
        tokenizer: Tokenizer instance
        validated_state: ValidatedState from validate_and_set_config
        distributed_state: DistributedState from setup_distributed
        worker_instance: The worker instance (for accessing helper methods)
        init_optimizer: Whether to initialize optimizer
        init_reference_model: Whether to initialize reference model

    Returns:
        ModelAndOptimizerState: Dataclass containing model and optimizer state
    """
    from typing import cast

    from nemo_automodel.components.distributed.tensor_utils import get_cpu_state_dict
    from nemo_automodel.components.moe.parallelizer import (
        parallelize_model as moe_parallelize_model,
    )

    from nemo_rl.models.policy.utils import import_class_from_path

    # Extract configuration values
    model_config = validated_state.model_config
    model_class = validated_state.model_class
    attn_impl = validated_state.attn_impl
    hf_config_overrides = validated_state.hf_config_overrides
    is_vlm = validated_state.is_vlm
    tp_size = validated_state.tp_size
    cp_size = validated_state.cp_size
    sequence_parallel_enabled = validated_state.sequence_parallel_enabled
    cpu_offload = validated_state.cpu_offload

    rank = distributed_state.rank
    device_mesh = distributed_state.device_mesh
    manager = distributed_state.manager
    moe_mesh = distributed_state.moe_mesh

    model_name = config["model_name"]

    print(f"[Rank {rank}] Initializing empty model for FSDP...")

    # Prepare automodel kwargs
    automodel_model_kwargs = config.get("automodel_model_kwargs", {})
    if automodel_model_kwargs.get("backend", None) is not None:
        backend_class = _resolve_target(
            automodel_model_kwargs.get("backend", None)["_target_"]
        )
        backend_kwargs = automodel_model_kwargs.get("backend")
        backend_kwargs.pop("_target_")
        backend = backend_class(**backend_kwargs)
        automodel_model_kwargs["backend"] = backend

    # Initialize empty model
    with init_empty_weights():
        model = model_class.from_config(
            model_config,
            attn_implementation=attn_impl,
            torch_dtype=str(model_config.torch_dtype),
            **automodel_model_kwargs,
        )

    # Store original state dict keys
    model_state_dict_keys = list(model.state_dict().keys())

    # Set pad token ID if needed
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Validate CP configuration with model type
    if cp_size > 1:
        if isinstance(model, Gemma3ForCausalLM):
            raise ValueError(
                "Context parallel is not supported for Gemma3ForCausalLM. "
                "Torch context parallel has many limitations. "
                "Please refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
            )

        if tp_size > 1 and sequence_parallel_enabled:
            raise ValueError(
                "It's a known issue that context parallel can't be used together with sequence parallel in DTensor worker. "
                "Please either set cp_size = 1 or disable sequence parallel. "
                "See https://github.com/NVIDIA-NeMo/RL/issues/659 for more details."
            )

        if is_vlm:
            raise ValueError(
                "Context parallel is yet not supported for VLM models. Please set cp_size = 1 to train VLM models."
            )

    # Parallelize model
    is_moe_model = any(["expert" in key for key in model_state_dict_keys])
    if not isinstance(model, PreTrainedModel) and is_moe_model:
        moe_parallelize_model(
            model=model,
            world_mesh=device_mesh,
            moe_mesh=moe_mesh,
            pp_enabled=False,
            dp_axis_names=(
                ("dp_replicate", "dp_shard_cp")
                if "dp_replicate" in device_mesh.mesh_dim_names
                and "dp_shard_cp" in device_mesh.mesh_dim_names
                else ("dp_shard_cp",)
            ),
            cp_axis_name="cp",
            tp_axis_name="tp",
            ep_axis_name="ep",
            ep_shard_axis_names=("ep_shard",),
            activation_checkpointing=config["dtensor_cfg"]["activation_checkpointing"],
        )
    else:
        model = manager.parallelize(model)

    print(model)

    # Ensure checkpointer exists
    worker_instance._ensure_checkpointer(
        config_updates={
            "model_repo_id": model_name,
            "dequantize_base_checkpoint": config.get(
                "dequantize_base_checkpoint", False
            ),
        },
        checkpoint_root=None,
    )
    worker_instance.checkpointer.config.model_state_dict_keys = model_state_dict_keys

    # Load base HF weights
    worker_instance.checkpointer.load_base_model(
        model,
        device=torch.cuda.current_device(),
        root_dir=hf_config_overrides.get("cache_dir", TRANSFORMERS_CACHE),
        model_name=model_name,
        peft_init_method=None,
        load_base_model=True,
    )

    # Handle tied word embeddings
    is_tied_lm_head = hasattr(model, "lm_head") and getattr(
        getattr(model, "config", {}), "tie_word_embeddings", False
    )
    if is_tied_lm_head:
        embed_tokens_weight = None
        for name, param in model.named_parameters():
            if "embed_tokens" in name and name.endswith(".weight"):
                embed_tokens_weight = param
                break

        if embed_tokens_weight is not None:
            model.lm_head.weight = embed_tokens_weight

    # CPU offload if needed
    if cpu_offload:
        model = worker_instance.move_to_device(model, "cpu")

    # Initialize reference model
    reference_model_state_dict = None
    if init_reference_model:
        reference_model_state_dict = get_cpu_state_dict(
            model.state_dict().items(), pin_memory=True
        )

    # Initialize optimizer
    optimizer = None
    if init_optimizer:
        optimizer_cls = import_class_from_path(config["optimizer"]["name"])
        optimizer = optimizer_cls(model.parameters(), **config["optimizer"]["kwargs"])

    # Initialize scheduler
    scheduler = None
    if "scheduler" in config and optimizer is not None:
        if isinstance(config["scheduler"], dict):
            scheduler_cls = import_class_from_path(
                cast(str, config["scheduler"]["name"])
            )
            scheduler = scheduler_cls(optimizer, **config["scheduler"]["kwargs"])
        else:
            schedulers = []
            for scheduler_cfg in config["scheduler"]:
                if "name" in scheduler_cfg:
                    schedulers.append(
                        import_class_from_path(scheduler_cfg["name"])(
                            optimizer, **scheduler_cfg["kwargs"]
                        )
                    )
                else:
                    assert "milestones" in scheduler_cfg, (
                        "unknown scheduler config: ",
                        scheduler_cfg,
                    )
                    milestones: list[int] = scheduler_cfg["milestones"]

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers, milestones
            )
    elif optimizer is not None:
        # Default to passthrough LR schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )

    return ModelAndOptimizerState(
        model=model,
        model_state_dict_keys=model_state_dict_keys,
        optimizer=optimizer,
        scheduler=scheduler,
        reference_model_state_dict=reference_model_state_dict,
    )
