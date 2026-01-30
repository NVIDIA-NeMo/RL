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
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge.training.checkpointing import (
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
    _create_peft_pre_wrap_hook,
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
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
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training.utils import get_ltor_masks_and_position_ids
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.config import MegatronGenerationConfig
from nemo_rl.models.megatron.setup import (
    finalize_megatron_setup,
    handle_model_import,
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_set_config,
    validate_model_paths,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


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
        **kwargs: Any,
    ):
        """Initialize the MegatronPolicyWorker."""
        # Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files
        apply_transformer_engine_patch()

        self.cfg = config

        # Set rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()

        # Step 1: Setup distributed
        setup_distributed()

        # Step 2: Validate and setup model paths
        hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
            config
        )
        # Handle model import if needed
        handle_model_import(
            config, hf_model_name, pretrained_path, pt_checkpoint_exists
        )

        # Store tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Setup model configuration
        runtime_config = validate_and_set_config(
            config,
            self.rank,
            hf_model_name,
            pretrained_path,
            weights_path,
            tokenizer,
        )

        self.megatron_cfg = runtime_config.megatron_cfg
        self.dtype = runtime_config.dtype
        self.optimizer_cpu_offload = runtime_config.optimizer_cpu_offload
        self.offload_optimizer_for_logprob = (
            runtime_config.offload_optimizer_for_logprob
        )
        self.is_generation_colocated = runtime_config.is_generation_colocated
        self.final_padded_vocab_size = runtime_config.final_padded_vocab_size

        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (runtime_config.model_cfg.fp16 or runtime_config.model_cfg.bf16)

        # Store FP8 config for later use
        self.fp8_cfg = config["megatron_cfg"].get("fp8_cfg", None)

        # Validate configuration
        self.megatron_cfg.validate()

        # Step 4: Setup Megatron model and components
        model_and_optimizer_state = setup_model_and_optimizer(
            config, self.megatron_cfg, init_optimizer
        )

        self.mcore_state = model_and_optimizer_state.state
        self.model = model_and_optimizer_state.model
        self.optimizer = model_and_optimizer_state.optimizer
        self.scheduler = model_and_optimizer_state.scheduler
        self.checkpointing_context = model_and_optimizer_state.checkpointing_context
        param_sync_func = model_and_optimizer_state.param_sync_func

        # Set the param sync function for the model if needed
        if param_sync_func is not None:
            self.megatron_cfg.param_sync_func = param_sync_func

        # Step 5: Setup reference model if needed
        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            self.reference_state_dict = setup_reference_model_state(
                config, self.megatron_cfg, pretrained_path
            )
            self.model = self.move_model(self.model, "cuda")

        # Step 6: Finalize setup
        (
            self.megatron_tokenizer,
            self.megatron_bridge,
            self.should_disable_forward_pre_hook,
            self.dp_size,
        ) = finalize_megatron_setup(
            config,
            self.megatron_cfg,
            hf_model_name,
            worker_sharding_annotations,
            self.model,
            self.optimizer,
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = None
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
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
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
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

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
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
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
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                pg_collection = get_pg_collection(self.model)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful, mp_group=pg_collection.mp
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm, mp_group=pg_collection.mp
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad, mp_group=pg_collection.mp
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
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
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

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

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

        mcore_generation_config = cast(
            MegatronGenerationConfig, self.cfg["generation"]["mcore_generation_config"]
        )
        buffer_size_gb = mcore_generation_config["buffer_size_gb"]

        num_cuda_graphs = mcore_generation_config["num_cuda_graphs"]
        block_size_tokens = mcore_generation_config["block_size_tokens"]
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config[
            "use_cuda_graphs_for_non_decode_steps"
        ]
        enable_chunked_prefill = mcore_generation_config["enable_chunked_prefill"]
        unified_memory_level = mcore_generation_config["unified_memory_level"]
        buffer_guaranteed_fraction = mcore_generation_config[
            "buffer_guaranteed_fraction"
        ]
        max_tokens = mcore_generation_config["max_tokens"]

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
            return broadcast_object_across_pp_ranks(size_in_bytes)

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
        from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
            get_vllm_qkv_scale_names,
        )

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
        from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
            convert_calibration_to_vllm_format,
        )

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
