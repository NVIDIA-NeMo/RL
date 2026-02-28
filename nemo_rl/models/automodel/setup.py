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

"""Setup utilities for automodel-based training in NeMo RL."""

import os
from typing import Any, Optional

import torch
from hydra.utils import get_class
from nemo_automodel import NeMoAutoModelForSequenceClassification
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.config.loader import _resolve_target
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh_utils import create_device_mesh
from nemo_automodel.components.distributed.tensor_utils import get_cpu_state_dict
from nemo_automodel.components.moe.config import MoEParallelizerConfig
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from nemo_rl.models.automodel.config import (
    DistributedContext,
    ModelAndOptimizerState,
    RuntimeConfig,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import configure_dynamo_cache, resolve_model_class

STRING_TO_DTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def validate_and_prepare_config(
    config: PolicyConfig,
    processor: Optional[AutoProcessor],
    rank: int,
) -> RuntimeConfig:
    """Validate configuration and prepare runtime settings.

    This function validates the policy configuration, sets environment variables,
    determines model configuration, and returns runtime settings as a named tuple.

    Args:
        config: Policy configuration dictionary
        processor: Optional processor for multimodal models
        rank: Current process rank

    Returns:
        RuntimeConfig named tuple containing validated configuration values

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If incompatible settings are detected
    """
    # Set basic configuration
    is_vlm = processor is not None
    is_generation_colocated = None
    if "generation" in config and config["generation"] is not None:
        is_generation_colocated = config["generation"]["colocated"]["enabled"]

    # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
    # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
    if not is_generation_colocated:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

    # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
    # with different order of node_bundles
    configure_dynamo_cache()

    # Parse precision
    precision = config["precision"]
    if precision not in STRING_TO_DTYPE:
        raise ValueError(f"Unknown precision: {precision}")
    dtype = STRING_TO_DTYPE[precision]

    # Get other configuration values
    cpu_offload = config["dtensor_cfg"]["cpu_offload"]
    offload_optimizer_for_logprob = config.get("offload_optimizer_for_logprob", False)
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

    # NeMoAutoModelForCausalLM uses flash_attention_2 by default
    # so we need to set it to None if sequence packing is disabled
    # See https://github.com/NVIDIA-NeMo/Automodel/blob/7e748be260651349307862426c0c168cebdeeec3/nemo_automodel/components/_transformers/auto_model.py#L180
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
        attn_implementation="flash_attention_2" if enable_seq_packing else None,
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

    return RuntimeConfig(
        model_class=model_class,
        model_config=model_config,
        hf_config_overrides=hf_config_overrides,
        allow_flash_attn_args=allow_flash_attn_args,
        attn_impl=attn_impl,
        dtype=dtype,
        enable_seq_packing=enable_seq_packing,
        max_grad_norm=max_grad_norm,
        cpu_offload=cpu_offload,
        offload_optimizer_for_logprob=offload_optimizer_for_logprob,
        is_generation_colocated=is_generation_colocated,
        is_reward_model=is_reward_model,
    )


def setup_reference_model_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Set up reference model state dict by creating a CPU copy of the model's state dict.

    This creates a reference copy of the model weights on CPU with pinned memory
    for efficient CPU-GPU transfers. The reference model is typically used to
    compute reference log probabilities during RL training.

    Args:
        model: The model to create a reference copy from

    Returns:
        Dictionary mapping parameter names to CPU tensors with pinned memory

    Example:
        >>> model = setup_model(...)
        >>> reference_model_state_dict = setup_reference_model_state(model)
    """
    return get_cpu_state_dict(model.state_dict().items(), pin_memory=True)


def setup_distributed(
    config: PolicyConfig,
    runtime_config: RuntimeConfig,
) -> DistributedContext:
    """Set up distributed training environment and create device meshes.

    Initializes torch.distributed process group and creates FSDP2Config,
    MoEParallelizerConfig, and device meshes for distributed training.

    Args:
        config: Policy configuration dictionary
        runtime_config: RuntimeConfig named tuple from validate_and_prepare_config

    Returns:
        DistributedContext containing device meshes and distributed configuration
    """
    # Initialize process group
    backend = "nccl" if not runtime_config.cpu_offload else "cuda:nccl,cpu:gloo"
    torch.distributed.init_process_group(backend=backend)
    world_size = torch.distributed.get_world_size()

    # Extract configuration values
    dtype = runtime_config.dtype
    cpu_offload = runtime_config.cpu_offload

    # Extract parallelization config
    tp_size = config["dtensor_cfg"].get("tensor_parallel_size", 1)
    cp_size = config["dtensor_cfg"].get("context_parallel_size", 1)
    ep_size = config["dtensor_cfg"].get("expert_parallel_size", 1)
    dp_size = config["dtensor_cfg"].get("data_parallel_size", None)
    sequence_parallel_enabled = config["dtensor_cfg"]["sequence_parallel"]

    # Create FSDP2Config
    fsdp2_config = FSDP2Config(
        sequence_parallel=sequence_parallel_enabled,
        tp_plan=config["dtensor_cfg"].get("custom_parallel_plan", None),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        ),
        offload_policy=CPUOffloadPolicy(pin_memory=False) if cpu_offload else None,
        activation_checkpointing=config["dtensor_cfg"]["activation_checkpointing"],
        defer_fsdp_grad_sync=config["dtensor_cfg"].get("defer_fsdp_grad_sync", True),
        backend="nccl",
    )

    # Create MoEParallelizerConfig from nested moe_parallelizer options
    moe_parallelizer_cfg = config["dtensor_cfg"].get("moe_parallelizer", {})
    moe_config = MoEParallelizerConfig(**moe_parallelizer_cfg)

    # Create device mesh
    device_mesh, moe_mesh = create_device_mesh(
        fsdp2_config,
        dp_size=dp_size,
        dp_replicate_size=1,
        tp_size=tp_size,
        pp_size=1,
        cp_size=cp_size,
        ep_size=ep_size,
        world_size=world_size,
    )

    # Derive sizes from mesh
    # Note: "dp" is a flattened submesh (dp_replicate + dp_shard) and may not appear
    # in mesh_dim_names directly, so we access it without the conditional check.
    resolved_dp_size = device_mesh["dp"].size()
    resolved_tp_size = device_mesh["tp"].size()
    resolved_cp_size = device_mesh["cp"].size()

    return DistributedContext(
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        fsdp2_config=fsdp2_config,
        moe_config=moe_config,
        dp_size=resolved_dp_size,
        tp_size=resolved_tp_size,
        cp_size=resolved_cp_size,
    )


def setup_model_and_optimizer(
    config: PolicyConfig,
    tokenizer: AutoTokenizer,
    runtime_config: RuntimeConfig,
    distributed_context: DistributedContext,
    checkpoint_manager: Any,
    is_vlm: bool = False,
    init_optimizer: bool = True,
    weights_path: Optional[str] = None,
    optimizer_path: Optional[str] = None,
) -> ModelAndOptimizerState:
    """Set up model, parallelization, and optimizer.

    Creates the model via from_pretrained with device meshes and distributed
    config, letting Automodel handle meta-device init, FSDP/EP sharding,
    and base weight loading internally.

    Args:
        config: Policy configuration dictionary
        tokenizer: Tokenizer for the model
        runtime_config: RuntimeConfig named tuple from validate_and_prepare_config
        distributed_context: DistributedContext from setup_distributed
        checkpoint_manager: Checkpoint manager for loading/saving weights
        is_vlm: Whether this is a vision-language model
        init_optimizer: Whether to initialize optimizer
        weights_path: Optional path to checkpoint weights to load
        optimizer_path: Optional path to optimizer state to load

    Returns:
        ModelAndOptimizerState containing model, optimizer, scheduler, and metadata
    """
    # Extract configuration values
    model_config = runtime_config.model_config
    model_class = runtime_config.model_class
    attn_impl = runtime_config.attn_impl
    hf_config_overrides = runtime_config.hf_config_overrides
    cpu_offload = runtime_config.cpu_offload
    is_reward_model = runtime_config.is_reward_model

    # Extract distributed configuration from context
    rank = torch.distributed.get_rank()
    device_mesh = distributed_context.device_mesh
    moe_mesh = distributed_context.moe_mesh
    fsdp2_config = distributed_context.fsdp2_config
    moe_config = distributed_context.moe_config
    tp_size = distributed_context.tp_size
    cp_size = distributed_context.cp_size
    ep_size = config["dtensor_cfg"].get("expert_parallel_size", 1)
    sequence_parallel_enabled = config["dtensor_cfg"]["sequence_parallel"]

    model_name = config["model_name"]

    # LoRA configuration
    lora_cfg = config["dtensor_cfg"].get("lora_cfg", None)
    peft_config = None
    lora_enabled = lora_cfg is not None and lora_cfg["enabled"]
    if lora_enabled:
        if tp_size > 1:
            assert not lora_cfg["use_triton"], (
                "Triton is not supported when tensor_parallel_size > 1"
            )
        # Always use float32 since FSDP requires all parameters to be in the same dtype
        cfg_dict_with_dtype = {**lora_cfg, "lora_dtype": "torch.float32"}
        peft_config = PeftConfig.from_dict(cfg_dict_with_dtype)

    print(f"[Rank {rank}] Initializing model for FSDP...")

    # Prepare automodel kwargs
    automodel_kwargs = config["dtensor_cfg"].get("automodel_kwargs", {})
    if automodel_kwargs.get("backend", None) is not None:
        backend_class = _resolve_target(
            automodel_kwargs.get("backend", None)["_target_"]
        )
        backend_kwargs = automodel_kwargs.get("backend")
        backend_kwargs.pop("_target_")
        backend = backend_class(**backend_kwargs)
        automodel_kwargs["backend"] = backend

    if "use_liger_kernel" not in automodel_kwargs:
        automodel_kwargs["use_liger_kernel"] = False

    # Determine SDPA method for activation checkpointing and CP
    from torch.nn.attention import SDPBackend

    if cp_size > 1:
        sdpa_method = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ]
    elif config["dtensor_cfg"]["activation_checkpointing"]:
        sdpa_method = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    else:
        sdpa_method = None

    # Validate CP configuration before model creation
    if cp_size > 1:
        if model_config.model_type == "gemma3":
            raise AssertionError(
                "Context parallel is not supported for Gemma3ForCausalLM. "
                "Please refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
            )

        if tp_size > 1 and sequence_parallel_enabled:
            raise AssertionError(
                "It's a known issue that context parallel can't be used together with sequence parallel in DTensor worker. "
                "Please either set cp_size = 1 or disable sequence parallel. "
                "See https://github.com/NVIDIA-NeMo/RL/issues/659 for more details."
            )

        if is_vlm:
            raise AssertionError(
                "Context parallel is yet not supported for VLM models. Please set cp_size = 1 to train VLM models."
            )

    # Build from_pretrained kwargs from hf_config_overrides so from_pretrained
    # applies them when loading the config internally (avoids passing config=
    # which causes duplicate 'config' arg for custom model implementations).
    from_pretrained_kwargs: dict[str, Any] = {
        **(hf_config_overrides or {}),
    }
    if is_reward_model and model_config.num_labels == 1:
        from_pretrained_kwargs["num_labels"] = 1

    # Create model via from_pretrained — Automodel handles meta device init,
    # FSDP/EP parallelization, and base weight loading internally.
    model = model_class.from_pretrained(
        model_name,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=fsdp2_config,
        moe_config=moe_config if ep_size > 1 else None,
        activation_checkpointing=config["dtensor_cfg"]["activation_checkpointing"],
        peft_config=peft_config,
        attn_implementation=attn_impl,
        torch_dtype=str(model_config.torch_dtype),
        trust_remote_code=True,
        sdpa_method=sdpa_method,
        **from_pretrained_kwargs,
        **automodel_kwargs,
    )

    # For activation checkpointing, we also must globally disable the cudnn SDPA backend
    # to ensure that cudnn does not get selected during recomputation.
    if config["dtensor_cfg"]["activation_checkpointing"]:
        from torch.backends import cuda

        cuda.enable_cudnn_sdp(False)

    # Compute model metadata after from_pretrained
    model_state_dict_keys = list(model.state_dict().keys())

    # Set model state dict keys in checkpoint manager for saving
    checkpoint_manager.set_model_state_dict_keys(model_state_dict_keys)

    # Set pad token ID if needed
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Determine model characteristics
    is_moe_model = any(["expert" in key for key in model_state_dict_keys])
    is_hf_model = (
        model_config.architectures[0] not in ModelRegistry.model_arch_name_to_cls
    )
    # Autocast is disabled for custom MoE models (non-HF) to avoid numerical issues
    autocast_enabled = not (is_moe_model and not is_hf_model)

    if rank == 0:
        print(model)

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
        # Move buffers to CPU for FSDP modules
        for v in model.buffers():
            v.data = v.data.to("cpu")
        model = model.to("cpu")

    # Initialize optimizer
    optimizer = None
    if init_optimizer:
        optimizer_cls = get_class(config["optimizer"]["name"])
        optimizer = optimizer_cls(model.parameters(), **config["optimizer"]["kwargs"])

    # Initialize scheduler
    scheduler = None
    if "scheduler" in config and optimizer is not None:
        if isinstance(config["scheduler"], dict):
            scheduler_cls = get_class(config["scheduler"]["name"])
            scheduler = scheduler_cls(optimizer, **config["scheduler"]["kwargs"])
        else:
            schedulers = []
            for scheduler_cfg in config["scheduler"]:
                if "name" in scheduler_cfg:
                    schedulers.append(
                        get_class(scheduler_cfg["name"])(
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

    # Load checkpoint if provided
    if weights_path:
        checkpoint_manager.load_checkpoint(
            model=model,
            weights_path=weights_path,
            optimizer=optimizer,
            optimizer_path=optimizer_path,
            scheduler=scheduler,
        )
    else:
        print(
            "No weights path provided. Loaded base HF weights via from_pretrained (default policy init)"
        )

    return ModelAndOptimizerState(
        model=model,
        model_state_dict_keys=model_state_dict_keys,
        optimizer=optimizer,
        scheduler=scheduler,
        is_hf_model=is_hf_model,
        is_moe_model=is_moe_model,
        is_reward_model=is_reward_model,
        model_class=type(model),
        model_config=model.config,
        peft_config=peft_config,
        autocast_enabled=autocast_enabled,
    )
