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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
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
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    is_pipeline_last_stage,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import get_moe_metrics
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.megatron.data import (
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_obj_from_pp_rank,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    TopkLogitsPostProcessor,
    megatron_forward_backward,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

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
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

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

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


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
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
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


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
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

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
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

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
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

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                (
                    batch,
                    global_valid_seqs,
                    global_valid_toks,
                ) = process_global_batch(
                    data,
                    gb_idx,
                    local_gbs,
                    loss_fn,
                    dp_group=parallel_state.get_data_parallel_group(),
                )

                (
                    data_iterator,
                    num_microbatches,
                    micro_batch_size,
                    seq_length,
                    padded_seq_length,
                ) = get_microbatch_iterator(batch, self.cfg, mbs)
                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(num_microbatches)

                loss_fn_wrapped = LossPostProcessor(
                    loss_fn=loss_fn,
                    cfg=self.cfg,
                )

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    losses_reduced = megatron_forward_backward(
                        model=self.model,
                        cfg=self.cfg,
                        data_iterator=data_iterator,
                        num_microbatches=num_microbatches,
                        seq_length=padded_seq_length,
                        mbs=micro_batch_size,
                        post_processing_fn=loss_fn_wrapped,
                        forward_only=eval_mode,
                        defer_fp32_logits=self.defer_fp32_logits,
                        global_valid_seqs=global_valid_seqs,
                        global_valid_toks=global_valid_toks,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                else:
                    gb_loss_metrics = None

                # Broadcast loss metrics from last stage to all stages
                ## TODO: check with PP > 1
                gb_loss_metrics = broadcast_loss_metrics_from_last_stage(
                    gb_loss_metrics
                )
                if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        self.model.eval()

        pp_grp = get_pipeline_model_parallel_group()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(data, self.cfg, logprob_batch_size)

        list_of_logprobs = megatron_forward_backward(
            model=self.model,
            cfg=self.cfg,
            data_iterator=mb_iterator,
            seq_length=padded_seq_length,
            mbs=micro_batch_size,
            num_microbatches=num_microbatches,
            post_processing_fn=LogprobsPostProcessor(cfg=self.cfg),
            forward_only=True,
            defer_fp32_logits=self.defer_fp32_logits,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = seq_length - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            tensors = {"logprobs": logprobs}
        else:
            tensors = {"logprobs": None}
        logprobs = broadcast_tensors_from_last_stage(tensors)["logprobs"]

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        self.model.eval()

        pp_grp = get_pipeline_model_parallel_group()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(data, self.cfg, logprob_batch_size)

        list_of_outputs = megatron_forward_backward(
            model=self.model,
            cfg=self.cfg,
            data_iterator=mb_iterator,
            seq_length=padded_seq_length,
            mbs=micro_batch_size,
            num_microbatches=num_microbatches,
            post_processing_fn=TopkLogitsPostProcessor(cfg=self.cfg, k=k),
            forward_only=True,
            defer_fp32_logits=self.defer_fp32_logits,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = padded_seq_length - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            tensors_to_broadcast = {
                "topk_logits": topk_logits,
                "topk_indices": topk_indices,
            }
        else:
            tensors_to_broadcast = {
                "topk_logits": None,
                "topk_indices": None,
            }

        # Broadcast tensors from last stage to all stages
        broadcasted = broadcast_tensors_from_last_stage(tensors_to_broadcast)
        topk_logits = broadcasted["topk_logits"]
        topk_indices = broadcasted["topk_indices"]

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False
        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_obj_from_pp_rank(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
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
