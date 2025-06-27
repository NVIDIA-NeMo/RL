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
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar

import ray
import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.custom_fsdp import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.engines import (
    StaticInferenceEngine,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.inference.text_generation.mcore_engine_server import (
    run_mcore_engine,
)
from megatron.training.utils import get_ltor_masks_and_position_ids
from nemo.tron import fault_tolerance
from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint, save_checkpoint
from nemo.tron.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
from nemo.tron.model import get_model_from_config
from nemo.tron.optim import setup_optimizer
from nemo.tron.setup import (
    HAVE_FSDP2,
    _init_checkpointing_context,
    _update_model_config_funcs,
)
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.async_utils import maybe_finalize_async_save
from nemo.tron.utils.common_utils import get_rank_safe
from nemo.tron.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from ray.util.queue import Queue
from torch.distributed import get_process_group_ranks
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import from_parallel_logits_to_logprobs
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.megatron.common import (
    broadcast_tensor,
    forward_step_arbitrary_loss,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.megatron.converters.common import (
    MegatronToHFConverter,
)
from nemo_rl.models.megatron.refit_utils import (
    gather_params,
    get_tp_dim,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from nemo_rl.models.policy.utils import get_gpu_info

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

    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        gpu_visibility_externally_set=True,
    )

    if cfg.ft_config and cfg.ft_config.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft_config)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model_config, cfg.train_config.micro_batch_size)

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
    checkpointing_context = _init_checkpointing_context(cfg.checkpoint_config)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer_config,
        make_vocab_size_divisible_by=cfg.model_config.make_vocab_size_divisible_by
        // cfg.model_config.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model_config.tensor_model_parallel_size,
    )
    if not cfg.model_config.vocab_size:
        cfg.model_config.vocab_size = cfg.tokenizer_config.padded_vocab_size

    torch.distributed.barrier()

    # Model, optimizer, and learning rate.
    model = get_model_from_config(
        cfg.model_config,
        cfg.ddp_config,
        use_torch_fsdp2=cfg.dist_config.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer_config.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng_config.data_parallel_random_init,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer_config,
            scheduler_config=cfg.scheduler_config,
            model=model,
            use_gloo_process_groups=cfg.dist_config.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    _update_model_config_funcs(
        model,
        cfg.model_config,
        cfg.ddp_config,
        optimizer,
        align_grad_reduce=cfg.dist_config.align_grad_reduce,
    )
    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint_config.load is not None
        or cfg.checkpoint_config.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint_config.load)
        or checkpoint_exists(cfg.checkpoint_config.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist_config.use_torch_fsdp2,
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
        from nemo.tron.utils.async_utils import AsyncCallsQueue

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
    runtime_env={
        **get_nsight_config_if_pattern_matches("megatron_policy_worker"),
    }
)
class MegatronPolicyWorker:
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
        megatron_checkpoint_home: Optional[str] = None,
        **kwargs: Any,
    ):
        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        if megatron_checkpoint_home is not None:
            pretrained_path = f"{megatron_checkpoint_home}/{hf_model_subdir}"
        else:
            pretrained_path = f"/opt/checkpoints/tron/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        if get_rank_safe() == 0:
            if pt_checkpoint_exists:
                print(
                    f"Checkpoint already exists at {pretrained_path}. Skipping import."
                )
            else:
                try:
                    # Clean environment to prevent conflicts
                    env_backup = {}
                    env_vars_to_clean = [
                        "MASTER_ADDR",
                        "MASTER_PORT",
                        "WORLD_SIZE",
                        "LOCAL_RANK",
                    ]
                    for var in env_vars_to_clean:
                        if var in os.environ:
                            env_backup[var] = os.environ[var]
                            del os.environ[var]

                    import_model_from_hf_name(hf_model_name, pretrained_path)

                    # Restore environment
                    for var, val in env_backup.items():
                        os.environ[var] = val

                except Exception as e:
                    print(f"Error importing model: {e}")
                    raise
                finally:
                    # Force cleanup after import
                    destroy_parallel_state()
            pre_init_communication_queue.put(True)
        else:
            pre_init_communication_queue.get()
            pre_init_communication_queue.put(True)
        destroy_parallel_state()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg_from_pretrained = ConfigContainer.from_yaml(pretrained_run_config)
        model_cfg = cfg_from_pretrained.model_config
        cfg_from_pretrained.logger_config = LoggerConfig()

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
        ]  # not supported right now
        assert model_cfg.context_parallel_size == 1, (
            "Context parallel is not supported right now"
        )
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
            model_cfg.activations_checkpoint_granularity = "full"
            model_cfg.activations_checkpoint_method = "uniform"
            model_cfg.activations_checkpoint_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )

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
        self.megatron_cfg = ConfigContainer(
            model_config=model_cfg,
            checkpoint_config=checkpoint_config,
            logger_config=LoggerConfig(logging_level=0),
            train_config=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=1000,  # Default value for inference
            ),
            optimizer_config=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp_config=DistributedDataParallelConfig(
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
                average_in_collective=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["average_in_collective"],
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler_config=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset_config=None,
            tokenizer_config=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        self.megatron_cfg.validate()
        print(f"cfg: {self.megatron_cfg}")
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
            self.megatron_cfg.ddp_config.overlap_param_gather
            and self.megatron_cfg.ddp_config.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = _init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            ref_megatron_cfg = ConfigContainer(
                model_config=self.megatron_cfg.model_config,
                checkpoint_config=ref_checkpoint_config,  # Use the reference checkpoint config
                logger_config=self.megatron_cfg.logger_config,
                train_config=self.megatron_cfg.train_config,
                optimizer_config=self.megatron_cfg.optimizer_config,
                ddp_config=self.megatron_cfg.ddp_config,
                scheduler_config=self.megatron_cfg.scheduler_config,
                dataset_config=self.megatron_cfg.dataset_config,
                tokenizer_config=self.megatron_cfg.tokenizer_config,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            reference_model = get_model_from_config(
                self.megatron_cfg.model_config,
                self.megatron_cfg.ddp_config,
                use_torch_fsdp2=self.megatron_cfg.dist_config.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer_config.overlap_param_gather_with_optimizer_step,
                data_parallel_random_init=self.megatron_cfg.rng_config.data_parallel_random_init,
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
                    and self.megatron_cfg.dist_config.use_torch_fsdp2,
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

        from nemo.tron.tokenizers.tokenizer import build_tokenizer

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model_config.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
        )
        self.final_padded_vocab_size = tokenizer_config.padded_vocab_size
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self._held_gather_buffer = None
        self.megatron_to_hf_converter = MegatronToHFConverter(hf_model_name, self.model)

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

    def configure_worker(self, num_gpus: int, bundle_indices: Optional[tuple] = None):
        USE_EXPANDABLE_SEGMENTS = False  # Disabling this right now as it seems to cause vLLM refit issues with Ampere
        if USE_EXPANDABLE_SEGMENTS:
            return None, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, None
        else:
            return None, None, None

    def is_alive(self):
        return True

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def get_gpu_info(self):
        """Return information about the GPU being used by this worker."""
        return get_gpu_info(self.model)

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

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
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(forward_step_arbitrary_loss, loss_fn=loss_fn)
            all_mb_metrics = []
            losses = []
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

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

                # Update learning rate.
                if update_successful:
                    increment = total_dataset_size.item()
                    self.scheduler.step(increment=increment)
                    skipped_iter = 0
                    curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                    curr_wd = self.scheduler.get_wd()
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
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["grad_norm"] = grad_norm
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

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
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor(
                mb_metrics["grad_norm"][-1]
            ).cpu(),  # TODO @sahilj: return an average or something later
        }
        return metrics

    # Temporary fix, 'data' is a kwarg due to some sort of ray bug
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

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_rank = get_pipeline_model_parallel_rank()
        pp_grp = get_pipeline_model_parallel_group()
        pp_size = get_pipeline_model_parallel_world_size()

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            data_dict = next(data_iterator).to("cuda")
            input_ids = data_dict["input_ids"]
            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                input_ids, 0, False, False, False
            )

            output_tensor = model(
                input_ids,
                position_ids,
                attention_mask,
            )

            def collection_fn(output_tensor):
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                token_logprobs = from_parallel_logits_to_logprobs(
                    output_tensor.to(torch.float32),
                    target=input_ids,
                    vocab_start_index=tp_rank * output_tensor.shape[-1],
                    vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                    group=tp_grp,
                    inference_only=True,
                )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0), {"logprobs": token_logprobs}

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
        micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

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

                gc.collect()
                torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    # Temporary fix, 'data' is a kwarg due to some sort of ray bug
    def get_reference_policy_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from thereference policy for a batch of data.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(
                data=data, micro_batch_size=micro_batch_size
            )

        return_data = BatchedDataDict[ReferenceLogprobOutputSpec]()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

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
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = True
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

        model_cfg = self.megatron_cfg.model_config
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts import StaticInferenceContext
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )

        inference_context = StaticInferenceContext.from_config(inference_wrapper_config)

        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, inference_context
        )
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )
        inference_engine = StaticInferenceEngine(
            text_generation_controller=text_generation_controller,
            max_batch_size=self.cfg["generation_batch_size"],
        )

        # detokenize the prompts
        # detokenized_prompts = [
        # self.tokenizer.decode(prompt)
        # for prompt in data.get("input_ids")
        # ]
        # apply chat template
        out = run_mcore_engine(
            engine=inference_engine,
            # prompts = detokenized_prompts,
            prompt_tokens_tensor=data["input_ids"],
            prompt_lengths_tensor=data["input_lengths"],
            tokens_to_generate=self.cfg["generation"]["max_new_tokens"]  # type: ignore
            - data["input_ids"].size(1),
        )
        # print(out)

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_seq_len = max([len(tokens) for tokens in out["tokens"]])

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
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )

            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": torch.tensor(
                [len(o) - input_lengths[i] for i, o in enumerate(out["logprobs"])]
            ),
            "unpadded_sequence_lengths": torch.tensor(
                [len(o) for o in out["logprobs"]]
            ),
        }

        self.model.config.flash_decode = False
        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    def zero_out_weights(self):
        """Zero out the weights of the model."""
        pass

    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """
        from nemo_rl.utils.nvml import get_device_uuid

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    def prepare_weights_for_ipc(self) -> tuple[list[tuple[str, int]], float]:
        """Prepare Megatron model weights for IPC transfer to vLLM.

        Collects information about weight tensors (names and sizes).
        Returns a list of (parameter_name, size_in_bytes) tuples.
        """
        from nemo_rl.utils.nvml import get_free_memory_bytes

        no_grad = torch.no_grad()
        no_grad.__enter__()
        # Ensure model is in evaluation mode
        self.model.eval()

        # Get parallel info
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_world_size = torch.distributed.get_world_size(tp_group)
        tp_group_rank_ids = get_process_group_ranks(tp_group)

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_world_size = torch.distributed.get_world_size(pp_group)
        pp_group_rank_ids = get_process_group_ranks(pp_group)

        # Collect parameter info
        param_info = []

        # Dictionary of modules we can quickly look up to check if a module has TP
        named_modules_dict = dict(self.model.named_modules())

        # Process each parameter in the model
        # state_dict includes parameters and persistent buffers
        for name, param in self.model.state_dict().items():
            # Skip _extra_state entries (these are metadata, not actual weights)
            if "_extra_state" in name:
                continue

            shape = list(param.shape)
            tp_dim = get_tp_dim(self.model, name, named_modules_dict)
            if tp_dim is not None:
                tp_rank_ids = tuple(sorted(tp_group_rank_ids))
                shape[tp_dim] *= len(tp_rank_ids)
            else:
                tp_rank_ids = (torch.distributed.get_rank(),)

            pp_rank_ids = tuple(sorted(pp_group_rank_ids))

            # Calculate size for this parameter
            prec_to_bytes = {
                torch.bfloat16: 2,
                torch.float16: 2,
                torch.float32: 4,
            }
            scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
            size_in_bytes = (
                param.element_size()
                * param.numel()
                * len(tp_rank_ids)
                * len(pp_rank_ids)
                * scale
            )
            param_info.append(
                (
                    (
                        name,
                        tuple(shape),
                        param.dtype,
                    ),
                    size_in_bytes,
                )
            )
        # Gather parameter info from all pipeline parallel ranks to ensure complete coverage
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_world_size = torch.distributed.get_world_size(pp_group)

        # Gather all parameter info from all PP ranks
        pp_gathered_param_infos = [None] * pp_world_size
        torch.distributed.all_gather_object(
            pp_gathered_param_infos, param_info, group=pp_group
        )
        pp_gathered_param_infos = [x for y in pp_gathered_param_infos for x in y]  # type: ignore

        all_param_infos = pp_gathered_param_infos

        # Merge all parameter infos, keeping only unique parameter names
        merged_param_info = []
        seen_params = set()

        for name, size in all_param_infos:
            if name not in seen_params:
                merged_param_info.append((name, size))
                seen_params.add(name)

        # Update param_info with the merged information
        param_info = merged_param_info

        print(f"Prepared {len(param_info)} tensors for IPC transfer")
        no_grad.__exit__(None, None, None)

        # Collect current available memory for refit
        ## Get current device index from torch
        device_idx = torch.cuda.current_device()
        ## Get device free memory using NVML
        total_available_bytes = get_free_memory_bytes(device_idx)
        ## Use 80% of the free memory for safety
        total_available_bytes *= 0.8

        return param_info, total_available_bytes

    # Temporary fix, 'keys' is a kwarg due to some sort of ray bug
    def get_weights_ipc_handles(self, *, keys: list[str]) -> dict[str, Any]:
        """Get IPC handles for the requested Megatron model weights.

        Args:
            keys: List of parameter names to get handles for
        Returns:
            Dict mapping device UUID to list of (mapped_key, handle) tuples
        """
        gathered_megatron_params = gather_params(
            self.model,
            keys,
        )
        gathered_hf_params = self.megatron_to_hf_converter.convert(
            gathered_megatron_params, self.model.config
        )

        # Get device UUID for IPC handles
        device_uuid = self.report_device_id()
        from torch.multiprocessing.reductions import reduce_tensor

        # Create IPC handles for each parameter
        all_handles = []
        for key, tensor in gathered_hf_params.items():
            handle = reduce_tensor(tensor.detach())
            all_handles.append((key, handle))

        # Store references to avoid premature garbage collection
        self._held_gather_buffer = gathered_hf_params
        shapes = {}
        for key, tensor in gathered_hf_params.items():
            shapes[key] = tensor.shape

        return {device_uuid: all_handles}

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()
        self.offload_before_refit()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        if hasattr(self, "optimizer") and self.optimizer is not None:
            if isinstance(self.optimizer, ChainedOptimizer):
                optimizer_state = self.optimizer.state
            else:
                optimizer_state = self.optimizer._get_state()
            for _, state in optimizer_state.items():
                for k, v in state.items():
                    if torch.is_tensor(v) and not v.is_cuda:
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

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
        if hasattr(self, "optimizer") and self.optimizer is not None:
            # Iterate through the state dictionaries for each parameter group
            if isinstance(self.optimizer, ChainedOptimizer):
                optimizer_state = self.optimizer.state
            else:
                optimizer_state = self.optimizer._get_state()
            for _, state in optimizer_state.items():
                # Iterate through the state items (e.g., momentum, variance) for a parameter
                for k, v in state.items():
                    # Check if the item is a tensor and on the GPU
                    if torch.is_tensor(v) and v.is_cuda:
                        # Move the tensor to CPU and update the state dictionary
                        state[k] = v.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    def offload_after_refit(self):
        no_grad = torch.no_grad()
        no_grad.__enter__()
        # Offload as much as possible on the CPU
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        if self._held_gather_buffer is not None:
            del self._held_gather_buffer
            self._held_gather_buffer = None

        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(self, model, device: str, move_params=True, move_grads=True):
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffer_idx in range(len(model.buffers)):
                if device == "cpu":
                    model.buffers[buffer_idx].offload_to_cpu(
                        move_params=move_params, move_grads=move_grads
                    )
                elif device == "cuda":
                    model.buffers[buffer_idx].reload_from_cpu(
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

        original_save_path = self.mcore_state.cfg.checkpoint_config.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                ckpt_cfg=self.mcore_state.cfg.checkpoint_config, blocking=False
            )
            self.mcore_state.cfg.checkpoint_config.save = weights_path

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
                ckpt_cfg=self.mcore_state.cfg.checkpoint_config,
                blocking=True,
                terminate=True,
            )

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint_config.save = original_save_path

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

    def shutdown(self):
        """Shutdown the policy."""
        pass

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
