# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time
import warnings
from typing import Any, Optional, TypeVar

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _create_peft_pre_wrap_hook,
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PreTrainedTokenizerBase

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.megatron.config import ModelAndOptimizerState, RuntimeConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
)

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


def setup_distributed() -> None:
    """Handle NCCL settings, dtype mapping, and basic config setup."""
    # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
    # with different order of node_bundles
    configure_dynamo_cache()
    # Ensure clean slate before import
    destroy_parallel_state()
    # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
    torch.distributed.init_process_group("nccl")


def validate_and_set_config(
    config,
    rank,
    hf_model_name,
    pretrained_path,
    weights_path,
    tokenizer,
):
    # Handle generation colocation
    is_generation_colocated = None
    if "generation" in config and config["generation"] is not None:
        is_generation_colocated = config["generation"]["colocated"]["enabled"]

    # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
    # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
    if not is_generation_colocated:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

    # Setup data types
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[config["precision"]]

    # Optimizer configuration
    optimizer_cpu_offload = config["megatron_cfg"]["optimizer"]["optimizer_cpu_offload"]
    offload_optimizer_for_logprob = config["offload_optimizer_for_logprob"]

    # Reward models are not yet supported with Megatron.
    if "reward_model_cfg" in config and config["reward_model_cfg"]["enabled"]:
        raise NotImplementedError(
            "Reward models are not yet supported with the Megatron backend, this issue is "
            "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
        )

    megatron_cfg, model_cfg = setup_model_config(
        config, rank, dtype, hf_model_name, pretrained_path, weights_path
    )

    final_padded_vocab_size = calculate_padded_vocab_size(
        megatron_cfg.model.vocab_size,
        megatron_cfg.model.make_vocab_size_divisible_by,
        config["megatron_cfg"]["tensor_model_parallel_size"],
    )

    return RuntimeConfig(
        megatron_cfg,
        model_cfg,
        dtype,
        optimizer_cpu_offload,
        offload_optimizer_for_logprob,
        is_generation_colocated,
        final_padded_vocab_size,
    )


def validate_model_paths(config: PolicyConfig) -> tuple[str, str, bool]:
    """Validate and setup model paths."""
    # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
    hf_model_name = config["model_name"]

    # Check if the checkpoint already exists
    hf_model_subdir = hf_model_name
    if os.path.exists(hf_model_name):
        hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

    pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
    pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
        os.path.join(pretrained_path, "iter_0000000")
    )

    return hf_model_name, pretrained_path, pt_checkpoint_exists


def setup_model_config(
    config: PolicyConfig,
    rank,
    dtype,
    hf_model_name: str,
    pretrained_path: str,
    weights_path: Optional[str] = None,
) -> tuple[ConfigContainer, Any]:
    """Handle all the model configuration logic."""
    # Load pretrained run config
    pretrained_run_config = os.path.join(
        pretrained_path, "iter_0000000/run_config.yaml"
    )

    if not os.path.exists(pretrained_run_config):
        raise FileNotFoundError(
            f"Pretrained run config not found at {pretrained_run_config} on rank={rank}. "
            "This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory "
            "not being mounted on this node. Please check"
        )

    try:
        cfg_from_pretrained = ConfigContainer.from_yaml(
            pretrained_run_config, mode=InstantiationMode.STRICT
        )
    except Exception as e:
        # Add helpful context as a note to the exception
        e.add_note(
            f"\n{'=' * 80}\n"
            f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
            f"created with an older version of megatron-bridge.\n"
            f"If this checkpoint is old or was generated by a different code version,\n"
            f"try deleting it and rerunning the code.\n"
            f"The checkpoint will be automatically regenerated with the current version.\n\n"
            f"Checkpoint location: {pretrained_path}\n"
            f"{'=' * 80}"
        )
        raise

    model_cfg = cfg_from_pretrained.model
    cfg_from_pretrained.logger = LoggerConfig()

    # Apply parallelism settings
    _apply_parallelism_config(model_cfg, config)

    # Apply MoE settings
    _apply_moe_config(model_cfg, config)

    # Apply precision settings
    _apply_precision_config(model_cfg, config, dtype)

    # Apply performance settings
    _apply_performance_config(model_cfg, config)

    # Validate optimizer configuration
    _validate_optimizer_config(config)

    # Optional layernorm epsilon
    if "layernorm_epsilon" in config["megatron_cfg"]:
        model_cfg.layernorm_epsilon = config["megatron_cfg"]["layernorm_epsilon"]

    # Validate chunking configuration
    _validate_chunking_config(config)

    # Create checkpoint configs
    checkpoint_config = _create_checkpoint_config(pretrained_path, weights_path)

    # Validate training configuration
    _validate_training_config(config, model_cfg)

    # Create final megatron config
    megatron_cfg = _create_megatron_config(
        model_cfg, checkpoint_config, config, hf_model_name, dtype
    )

    _validate_dtype_config(dtype, megatron_cfg.model, megatron_cfg.optimizer)

    return megatron_cfg, model_cfg


def _apply_parallelism_config(model_cfg: Any, config: PolicyConfig) -> None:
    """Apply tensor/pipeline/context parallelism configuration."""
    model_cfg.tensor_model_parallel_size = config["megatron_cfg"][
        "tensor_model_parallel_size"
    ]
    model_cfg.pipeline_model_parallel_size = config["megatron_cfg"][
        "pipeline_model_parallel_size"
    ]
    model_cfg.num_layers_in_first_pipeline_stage = config["megatron_cfg"][
        "num_layers_in_first_pipeline_stage"
    ]
    model_cfg.num_layers_in_last_pipeline_stage = config["megatron_cfg"][
        "num_layers_in_last_pipeline_stage"
    ]
    model_cfg.sequence_parallel = config["megatron_cfg"]["sequence_parallel"]
    model_cfg.context_parallel_size = config["megatron_cfg"]["context_parallel_size"]

    if model_cfg.context_parallel_size > 1:
        assert config["sequence_packing"]["enabled"], (
            "Sequence Packing must be enabled to use Context Parallelism with MCore"
        )


def _apply_moe_config(model_cfg: Any, config: PolicyConfig) -> None:
    """Apply Mixture of Experts configuration."""
    model_cfg.expert_tensor_parallel_size = config["megatron_cfg"][
        "expert_tensor_parallel_size"
    ]
    model_cfg.expert_model_parallel_size = config["megatron_cfg"][
        "expert_model_parallel_size"
    ]

    # MoE stability settings

    # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
    # especially when using many experts.
    model_cfg.moe_router_dtype = config["megatron_cfg"]["moe_router_dtype"]

    # The below two configs (and "freeze_moe_router") are used to stabilize moe training
    # by preventing updates to the moe router. We found that this is helpful in reducing
    # logprob error during training.

    # Set this to "none" to disable load balancing loss.
    model_cfg.moe_router_load_balancing_type = config["megatron_cfg"][
        "moe_router_load_balancing_type"
    ]
    # Set this to 0.0 to disable updates to the moe router expert bias
    model_cfg.moe_router_bias_update_rate = config["megatron_cfg"][
        "moe_router_bias_update_rate"
    ]

    model_cfg.moe_enable_deepep = config["megatron_cfg"]["moe_enable_deepep"]
    model_cfg.moe_token_dispatcher_type = config["megatron_cfg"][
        "moe_token_dispatcher_type"
    ]
    model_cfg.moe_shared_expert_overlap = config["megatron_cfg"][
        "moe_shared_expert_overlap"
    ]

    model_cfg.moe_permute_fusion = config["megatron_cfg"]["moe_permute_fusion"]


def _apply_precision_config(
    model_cfg: Any, config: PolicyConfig, dtype: torch.dtype
) -> None:
    """Apply precision and dtype configuration."""
    model_cfg.bf16 = dtype == torch.bfloat16
    model_cfg.fp16 = dtype == torch.float16

    if model_cfg.fp16:
        assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
        model_cfg.params_dtype = torch.float16
    elif model_cfg.bf16:
        assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
        model_cfg.params_dtype = torch.bfloat16
    else:
        model_cfg.params_dtype = torch.float32

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    model_cfg.pipeline_dtype = dtype_map[config["megatron_cfg"]["pipeline_dtype"]]


def _apply_performance_config(model_cfg: Any, config: PolicyConfig) -> None:
    """Apply performance optimization configuration."""
    model_cfg.parallel_output = True

    # Activation checkpointing
    if config["megatron_cfg"]["activation_checkpointing"]:
        model_cfg.recompute_granularity = "full"
        model_cfg.recompute_method = "uniform"
        model_cfg.recompute_num_layers = 1

    # Activation function validation
    if not model_cfg.gated_linear_unit:
        assert model_cfg.activation_func is not None, (
            "activation_func must be set if not using gated_linear_unit. This likely "
            "indicates an issue in configuration conversion (e.g. activation func was "
            "a lambda and couldn't be serialized). This is based on this check "
            "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
        )

    # Fusion settings
    model_cfg.apply_rope_fusion = config["megatron_cfg"]["apply_rope_fusion"]
    model_cfg.bias_activation_fusion = config["megatron_cfg"]["bias_activation_fusion"]

    # FP8 configuration
    fp8_cfg = config["megatron_cfg"].get("fp8_cfg", None)
    if fp8_cfg is not None and fp8_cfg.get("enabled", False):
        try:
            model_cfg.fp8 = fp8_cfg["fp8"]
            model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
            model_cfg.fp8_param = fp8_cfg["fp8_param"]
        except KeyError as e:
            raise KeyError(f"Missing key in fp8_cfg: {e}")

        if model_cfg.fp8_param:
            warnings.warn(
                "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
            )


def _validate_optimizer_config(config: PolicyConfig) -> None:
    """Validate optimizer configuration."""
    optimizer_cpu_offload = config["megatron_cfg"]["optimizer"]["optimizer_cpu_offload"]
    optimizer_offload_fraction = config["megatron_cfg"]["optimizer"][
        "optimizer_offload_fraction"
    ]

    if optimizer_cpu_offload:
        # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
        # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
        assert optimizer_offload_fraction == 1.0, (
            "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
        )


def _validate_chunking_config(config: PolicyConfig) -> None:
    """Validate chunking configuration."""
    if (
        "logprob_chunk_size" in config
        and config["logprob_chunk_size"] is not None
        and config["logprob_chunk_size"] > 0
    ):
        assert config["megatron_cfg"]["defer_fp32_logits"], (
            "defer_fp32_logits must be True if logprob_chunk_size is set"
        )


def _create_checkpoint_config(
    pretrained_path: str, weights_path: Optional[str]
) -> CheckpointConfig:
    """Create checkpoint configurations."""
    return CheckpointConfig(
        save_interval=100,
        save=weights_path,
        load=weights_path,
        pretrained_checkpoint=pretrained_path,
        async_save=False,
        fully_parallel_save=True,
        fully_parallel_load=True,
        load_rng=False,
    )


def _validate_training_config(config: PolicyConfig, model_cfg: Any) -> None:
    """Validate training configuration."""
    assert "train_iters" in config["megatron_cfg"], (
        "train_iters must be set in megatron_cfg. For an example, see "
        "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
    )

    ## These settings are required for correct gradient computations in mcore
    ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
    ## so we handle the scaling in nemo-rl.
    ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
    ## on the TP-sharded parameters.
    model_cfg.calculate_per_token_loss = True
    model_cfg.perform_initialization = True

    # MoE aux loss validation
    assert (
        "aux_loss" not in model_cfg.moe_router_load_balancing_type
        or model_cfg.moe_aux_loss_coeff == 0
    ), (
        "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
        "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
    )


def _validate_dtype_config(
    dtype: torch.dtype, model_cfg: Any, optimizer_cfg: Any
) -> None:
    # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
    if dtype == torch.bfloat16:
        assert model_cfg.bf16 == True, (
            "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
        )
        assert (
            optimizer_cfg.use_precision_aware_optimizer == False
            or optimizer_cfg.bf16 == True
        ), (
            "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
        )
    elif dtype == torch.float16:
        assert model_cfg.fp16 == True, (
            "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
        )
        assert (
            optimizer_cfg.use_precision_aware_optimizer == False
            or optimizer_cfg.fp16 == True
        ), (
            "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
        )
    elif dtype == torch.float32:
        assert model_cfg.bf16 == False and model_cfg.fp16 == False, (
            "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
        )
        assert optimizer_cfg.bf16 == False and optimizer_cfg.fp16 == False, (
            "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
        )


def _create_megatron_config(
    model_cfg: Any,
    checkpoint_config: CheckpointConfig,
    config: PolicyConfig,
    hf_model_name: str,
    dtype: torch.dtype,
) -> ConfigContainer:
    """Create the final Megatron configuration container."""
    return ConfigContainer(
        model=model_cfg,
        checkpoint=checkpoint_config,
        logger=LoggerConfig(logging_level=0),
        train=TrainingConfig(
            micro_batch_size=1,  # ignored
            global_batch_size=config["train_global_batch_size"],  # ignored
            train_iters=config["megatron_cfg"]["train_iters"],
        ),
        optimizer=OptimizerConfig(**config["megatron_cfg"]["optimizer"]),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=config["megatron_cfg"][
                "distributed_data_parallel_config"
            ]["grad_reduce_in_fp32"],
            overlap_grad_reduce=config["megatron_cfg"][
                "distributed_data_parallel_config"
            ]["overlap_grad_reduce"],
            overlap_param_gather=config["megatron_cfg"][
                "distributed_data_parallel_config"
            ]["overlap_param_gather"],
            # we need to set average_in_collective=False with calculate_per_token_loss=T
            # otherwise, mcore throws an assertion error.
            average_in_collective=False,  # Required with calculate_per_token_loss=True
            use_distributed_optimizer=config["megatron_cfg"]["optimizer"][
                "use_distributed_optimizer"
            ],
            data_parallel_sharding_strategy=config["megatron_cfg"][
                "distributed_data_parallel_config"
            ]["data_parallel_sharding_strategy"],
        ),
        scheduler=SchedulerConfig(**config["megatron_cfg"]["scheduler"]),
        dataset=None,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        ),
    )


def setup_model_and_optimizer(
    policy_cfg: PolicyConfig,
    megatron_cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = megatron_cfg
    # TODO: Freeze state.cfg

    megatron_cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=megatron_cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if megatron_cfg.ft and megatron_cfg.ft.enable_ft_package:
        fault_tolerance.setup(megatron_cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(megatron_cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(megatron_cfg.model, megatron_cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(megatron_cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        megatron_cfg.tokenizer,
        make_vocab_size_divisible_by=megatron_cfg.model.make_vocab_size_divisible_by
        // megatron_cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=megatron_cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert megatron_cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []

    use_peft = policy_cfg["megatron_cfg"].get("peft", {}).get("enabled", False)

    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:
        if use_peft:
            raise ValueError(
                "Freezing the MOE router is not currently supported when using PEFT"
            )

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = MoEFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    if use_peft:
        peft_cfg = policy_cfg["megatron_cfg"].get("peft", {})
        peft = LoRA(
            target_modules=peft_cfg["target_modules"],
            exclude_modules=peft_cfg["exclude_modules"],
            dim=peft_cfg["dim"],
            alpha=peft_cfg["alpha"],
            dropout=peft_cfg["dropout"],
            dropout_position=peft_cfg["dropout_position"],
            lora_A_init_method=peft_cfg["lora_A_init_method"],
            lora_B_init_method=peft_cfg["lora_B_init_method"],
            a2a_experimental=peft_cfg["a2a_experimental"],
            lora_dtype=peft_cfg["lora_dtype"],
        )
    else:
        peft = None
    megatron_cfg.peft = peft

    if megatron_cfg.peft is not None:
        pre_peft_hook = _create_peft_pre_wrap_hook(megatron_cfg, state)
        megatron_cfg.model.register_pre_wrap_hook(pre_peft_hook)

        def composed_peft_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            model = pre_peft_hook(model)
            return model

        pre_wrap_hook.extend([composed_peft_hook])

    # Model, optimizer, and learning rate.
    model = get_model(
        megatron_cfg.model,
        megatron_cfg.ddp,
        use_torch_fsdp2=megatron_cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=megatron_cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=megatron_cfg.optimizer,
            scheduler_config=megatron_cfg.scheduler,
            model=model,
            use_gloo_process_groups=megatron_cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    if megatron_cfg.peft is not None:
        should_load_checkpoint = (
            megatron_cfg.checkpoint.load is not None
            and checkpoint_exists(megatron_cfg.checkpoint.load)
        )
        if should_load_checkpoint:
            # The finetune toggle is explicitly set to True in order to avoid loading optimizer and RNG states
            # This is switched off here in order to load these states from the checkpoint
            megatron_cfg.checkpoint.finetune = False
    else:
        should_load_checkpoint = (
            megatron_cfg.checkpoint.load is not None
            and checkpoint_exists(megatron_cfg.checkpoint.load)
        ) or (
            megatron_cfg.checkpoint.pretrained_checkpoint is not None
            and checkpoint_exists(megatron_cfg.checkpoint.pretrained_checkpoint)
        )

    # Load checkpoint if applicable
    if should_load_checkpoint:
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and megatron_cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    # Set the param sync function for the model
    param_sync_func = None
    if megatron_cfg.ddp.overlap_param_gather and megatron_cfg.ddp.align_param_gather:
        param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            param_sync_func = param_sync_func[0]

    # Get the first model from the list
    model = model[0]

    return ModelAndOptimizerState(
        state,
        model,
        optimizer,
        scheduler,
        checkpointing_context,
        param_sync_func,
    )


def handle_model_import(
    config: PolicyConfig,
    hf_model_name: str,
    pretrained_path: str,
    pt_checkpoint_exists: bool,
) -> None:
    """Handle HF model import if checkpoint doesn't exist."""
    if pt_checkpoint_exists:
        print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
    else:
        hf_config_overrides = config.get("hf_config_overrides", {}) or {}
        import_model_from_hf_name(
            hf_model_name,
            pretrained_path,
            config["megatron_cfg"],
            **hf_config_overrides,
        )

        if parallel_state.model_parallel_is_initialized():
            print("Reinitializing model parallel after loading model state.")
            parallel_state.destroy_model_parallel()


def setup_reference_model_state(
    config: PolicyConfig, megatron_cfg: ConfigContainer, pretrained_path: str
) -> dict:
    """Setup the reference model for inference and return its state dict."""
    # Create reference checkpoint config
    ref_checkpoint_config = CheckpointConfig(
        pretrained_checkpoint=pretrained_path,
        save=None,
        load=None,
        fully_parallel_load=True,
        load_rng=False,
    )

    ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

    # Create a separate megatron config for the reference model
    ref_megatron_cfg = ConfigContainer(
        model=megatron_cfg.model,
        checkpoint=ref_checkpoint_config,
        logger=megatron_cfg.logger,
        train=megatron_cfg.train,
        optimizer=megatron_cfg.optimizer,
        ddp=megatron_cfg.ddp,
        scheduler=megatron_cfg.scheduler,
        dataset=megatron_cfg.dataset,
        tokenizer=megatron_cfg.tokenizer,
    )

    # Create a separate state object for the reference model
    ref_state = GlobalState()
    ref_state.cfg = ref_megatron_cfg

    # Configure mixed precision wrapper for reference model
    ref_mixed_precision_wrapper = Float16Module
    if config["megatron_cfg"].get("freeze_moe_router", False):
        ref_mixed_precision_wrapper = MoEFloat16Module

    reference_model = get_model(
        megatron_cfg.model,
        megatron_cfg.ddp,
        use_torch_fsdp2=megatron_cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
        pre_wrap_hook=megatron_cfg.rng.data_parallel_random_init,
        mixed_precision_wrapper=ref_mixed_precision_wrapper,
    )

    print("Loading the Reference Model")
    reference_state_dict = {}

    if ref_checkpoint_config.pretrained_checkpoint is not None and checkpoint_exists(
        ref_checkpoint_config.pretrained_checkpoint
    ):
        load_checkpoint(
            ref_state,
            reference_model,
            None,  # no optimizer
            None,  # no scheduler
            checkpointing_context=ref_ckpt_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and megatron_cfg.dist.use_torch_fsdp2,
        )
        reference_model = reference_model[0]
        reference_model.eval()

        # Store reference state dict on CPU
        for name, item in reference_model.state_dict().items():
            if isinstance(item, torch.Tensor):
                cpu_item = item.detach().to(device="cpu", non_blocking=True, copy=True)
                del item
            else:
                cpu_item = item
            reference_state_dict[name] = cpu_item
        print("Reference model loaded")
    else:
        print("Reference model not loaded")

    return reference_state_dict


def finalize_megatron_setup(
    config: PolicyConfig,
    megatron_cfg: ConfigContainer,
    hf_model_name: str,
    worker_sharding_annotations: NamedSharding,
    model,
    optimizer,
) -> tuple:
    """Finalize the setup with remaining configurations.

    Returns:
        Tuple of (megatron_tokenizer, megatron_bridge, should_disable_forward_pre_hook, dp_size)
    """
    _update_model_config_funcs(
        [model],
        megatron_cfg.model,
        megatron_cfg.ddp,
        optimizer,
        align_grad_reduce=megatron_cfg.dist.align_grad_reduce,
    )

    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_model_name,
    )

    megatron_tokenizer = build_tokenizer(
        tokenizer_config,
        make_vocab_size_divisible_by=megatron_cfg.model.make_vocab_size_divisible_by
        // config["megatron_cfg"]["tensor_model_parallel_size"],
        tensor_model_parallel_size=config["megatron_cfg"]["tensor_model_parallel_size"],
        trust_remote_code=True,
    )

    dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
    megatron_bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True
    )

    should_disable_forward_pre_hook = (
        config["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
        and config["megatron_cfg"]["distributed_data_parallel_config"][
            "overlap_param_gather"
        ]
    )

    return megatron_tokenizer, megatron_bridge, should_disable_forward_pre_hook, dp_size


class MoEFloat16Module(Float16Module):
    """Float 16 Module with the ability to keep the expert bias in float32.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(MoEFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()
