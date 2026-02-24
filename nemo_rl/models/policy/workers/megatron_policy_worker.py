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
import asyncio
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
from megatron.bridge.training.checkpointing import (
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.config import InferenceConfig, KVCacheManagementMode
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    is_pipeline_last_stage,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import get_moe_metrics
from nemo_rl.models.megatron.config import MegatronGenerationConfig
from nemo_rl.models.megatron.data import (
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_obj_from_pp_rank,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.setup import (
    finalize_megatron_setup,
    handle_model_import,
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_set_config,
    validate_model_paths,
)
from nemo_rl.models.megatron.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    TopkLogitsPostProcessor,
    aggregate_training_statistics,
    megatron_forward_backward,
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
        
        self.dynamic_inference_engine = None
        self.inference_client = None
        self.inference_context = None
        self.inference_wrapped_model = None
        self._inference_engine_initialized = False
        self._inference_engine_alseep = True  # Start paused since we begin with training
        self._inference_loop = None  # Event loop for inference operations
        self._inference_thread = None  # Thread running the event loop

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        for module, handle in list(self.model.remove_forward_pre_hook_handles.items()):
            handle.remove()
        self.model.remove_forward_pre_hook_handles.clear()
        if param_sync:
            self.model.start_param_sync(force_sync=True)
            
        # TODO : Check why this doesnt work. 
        #self.model.disable_forward_pre_hook(param_sync=param_sync)

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
        # Note: zero_grad_buffer is called at the start of each global batch iteration
        # in the loop below, so we don't need to call it here.
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
                gb_result = process_global_batch(
                    data,
                    loss_fn=loss_fn,
                    dp_group=parallel_state.get_data_parallel_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                (
                    data_iterator,
                    num_microbatches,
                    micro_batch_size,
                    seq_length,
                    padded_seq_length,
                ) = get_microbatch_iterator(
                    batch,
                    self.cfg,
                    mbs,
                    straggler_timer=self.mcore_state.straggler_timer,
                )
                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(num_microbatches)

                loss_post_processor = LossPostProcessor(
                    loss_fn=loss_fn,
                    cfg=self.cfg,
                    num_microbatches=num_microbatches,
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
                        post_processing_fn=loss_post_processor,
                        forward_only=eval_mode,
                        defer_fp32_logits=self.defer_fp32_logits,
                        global_valid_seqs=global_valid_seqs,
                        global_valid_toks=global_valid_toks,
                        straggler_timer=self.mcore_state.straggler_timer,
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
                            if "_min" in k or "_max" in k:
                                loss_metrics[k] = x[k]
                            else:
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
        mb_metrics, global_loss = aggregate_training_statistics(
            all_mb_metrics=all_mb_metrics,
            losses=losses,
            data_parallel_group=parallel_state.get_data_parallel_group(),
        )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": mb_metrics,
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
        ) = get_microbatch_iterator(
            data,
            self.cfg,
            logprob_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
        )

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
            straggler_timer=self.mcore_state.straggler_timer,
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
        ) = get_microbatch_iterator(
            data,
            self.cfg,
            logprob_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
        )

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
            straggler_timer=self.mcore_state.straggler_timer,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = seq_length - tk.shape[1]
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

    def _get_lang_module(self):
        """Get the underlying language module from the wrapped model."""
        return (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

    def _initialize_inference_engine(self, mcore_generation_config: dict):
        """Initialize the persistent inference engine and client.
        
        This method sets up the DynamicInferenceEngine, DynamicInferenceContext,
        and InferenceClient for coordinator-based inference. The engine is created
        once and reused across multiple generate() calls.
        """
        if self._inference_engine_initialized:
            return

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.text_generation_controllers.text_generation_controller import (
            TextGenerationController,
        )

        model_cfg = self.megatron_cfg.model
        

        from megatron.core.utils import get_attr_wrapped_model
        pg_collection = get_attr_wrapped_model(self.model, "pg_collection")
        
        buffer_size_gb = mcore_generation_config["buffer_size_gb"]
        num_cuda_graphs = mcore_generation_config["num_cuda_graphs"]
        block_size_tokens = mcore_generation_config["block_size_tokens"]
        enable_chunked_prefill = mcore_generation_config["enable_chunked_prefill"]
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config[
            "use_cuda_graphs_for_non_decode_steps"
        ]
        max_tokens = mcore_generation_config["max_tokens"]

        # Level 0: No unified memory, CUDA graphs are deleted/recreated on pause/resume
        # Level 1: Unified memory enabled, tensors maintain static addresses
        unified_memory_level = mcore_generation_config["unified_memory_level"]
        kv_cache_management_mode = mcore_generation_config["kv_cache_management_mode"]
        static_kv_memory_pointers = mcore_generation_config["static_kv_memory_pointers"]
        materialize_only_last_token_logits = mcore_generation_config["materialize_only_last_token_logits"]

        model_config = self.model.config

        inference_config = InferenceConfig(
            block_size_tokens=block_size_tokens,
            buffer_size_gb=buffer_size_gb,
            num_cuda_graphs=num_cuda_graphs,
            max_tokens=max_tokens,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            unified_memory_level=unified_memory_level,
            kv_cache_management_mode=KVCacheManagementMode(kv_cache_management_mode),
            static_kv_memory_pointers=static_kv_memory_pointers,
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            enable_chunked_prefill=enable_chunked_prefill,
            pg_collection=pg_collection,
         )

        # Create inference context
        self.inference_context = DynamicInferenceContext(model_config, inference_config)

        # Create inference wrapper
        self.inference_wrapped_model = GPTInferenceWrapper(
            self.model, self.inference_context
        )
        # Create text generation controller
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=self.inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Create the inference engine
        self.dynamic_inference_engine = DynamicInferenceEngine(
            text_generation_controller,
            self.inference_context
        )

        self._inference_engine_initialized = True
        self._inference_engine_alseep = True  # Engine starts in paused state
        print(f"[Rank {self.rank}] Initialized persistent inference engine")

    async def _start_inference_coordinator(self, coordinator_port: int):
        """Start the inference coordinator and engine loop.
        
        This is called once when the inference infrastructure is first needed.
        The engine's start_listening_to_data_parallel_coordinator returns the
        actual coordinator address (dp_addr) which is used to create the client.
        """
        dp_addr = await self.dynamic_inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=coordinator_port,
            launch_inference_coordinator=True,
        )

        dist_rank = torch.distributed.get_rank()
        if dist_rank == 0:
            from megatron.core.inference.inference_client import InferenceClient
            self.inference_client = InferenceClient(inference_coordinator_address=dp_addr)
            await self.inference_client.start()
        
        self._inference_engine_alseep = False

    def _sleep(self):
        """pause the inference engine to free GPU memory for training.
        
        This method should be called before training to:
        1. Deallocate KV cache and other inference-specific GPU memory
        2. Disable CUDA graphs for inference
        3. Toggle model configuration for training mode
        
        Uses the coordinator's pause mechanism to properly pause the engine loop
        and then pause the engine (deallocate tensors, etc.).
        
        For coordinator-based inference:
        - Only rank 0 sends pause signals via the coordinator
        - The coordinator broadcasts to all DP engines
        - Non-rank-0 workers wait for their engine to be paused via the event loop
        """

        future = asyncio.run_coroutine_threadsafe(
            self._sleep_engine(),
            self._inference_loop
        )
        future.result()
        # Synchronize all ranks
        torch.distributed.barrier()
        
        self._inference_engine_alseep = True
        print(f"[Rank {self.rank}] paused inference engine")

    async def _sleep_engine(self):
        """Send suspend signals via the coordinator and wait for acknowledgment.
        
        Mirrors MegatronLocal.suspend() from megatron/rl/inference/megatron.py:
        1. Rank 0 sends suspend (PAUSE + SUSPEND) to coordinator
        2. All ranks wait for engine to be paused
        3. All ranks call engine.suspend() to deallocate GPU state
        """ 
        if torch.distributed.get_rank() == 0:
            # Send PAUSE  signals
            self.inference_client.suspend_engines()
            # Wait for the engine to acknowledge the pause
        await self.dynamic_inference_engine.paused.wait()
        self.dynamic_inference_engine.suspend()

    def _wake(self):
        """Resume the inference engine after training.
        
        This method should be called before generation to:
        1. Reallocate KV cache and inference-specific GPU memory
        2. Enable CUDA graphs for inference
        3. Toggle model configuration for inference mode
        
        Uses the coordinator's resume mechanism to properly resume the engine loop.
        
        For coordinator-based inference:
        - Only rank 0 sends resume signals via the coordinator
        - The coordinator broadcasts to all DP engines
        - Non-rank-0 workers wait for their engine to be running via the event loop
        """
        
        # Use the coordinator-based resume mechanism
        # Only rank 0 sends the signal - coordinator broadcasts to all DP engines
        future = asyncio.run_coroutine_threadsafe(
            self._wake_engine(),
            self._inference_loop
        )
        future.result()
        # Synchronize all ranks
        torch.distributed.barrier()
        
        self._inference_engine_alseep = False
        print(f"[Rank {self.rank}] Resumed inference engine")

    async def _wake_engine(self):
        """Send resume signals via the coordinator and wait for acknowledgment.
        
        Mirrors MegatronLocal.resume() from megatron/rl/inference/megatron.py:
        1. Rank 0 sends resume (RESUME + UNPAUSE) to coordinator
        2. All ranks wait for engine to be running
        3. All ranks call engine.resume() to reallocate GPU state
        """
        if torch.distributed.get_rank() == 0:
            self.inference_client.resume_engines()
        await self.dynamic_inference_engine.running.wait()
        self.dynamic_inference_engine.resume()

    @contextmanager
    def inference_mode(self, mcore_generation_config: dict):
        """Context manager for inference mode, following Megatron RL's pattern.

        This mirrors megatron_rl_inference_mode from megatron/rl/rl_utils.py

        ENTER order:
        1. Put model in eval mode
        2. Clear rotary cache
        3. Toggle CUDA graphs ON
        4. Initialize inference engine (first time only)
        5. Resume engine (reallocates KV cache, recreates CUDA graphs as needed)

        EXIT order:
        1. Suspend engine (deallocates KV cache and GPU state)
        2. Toggle CUDA graphs OFF
        3. Clear rotary cache
        4. Put model back in train mode

        In non-colocated mode, the engine is NOT suspended/resumed between
        iterations because training happens on separate GPUs. Only weight values
        change (via swap_weights_via_reshard), not the compute graph structure,
        so CUDA graphs remain valid and do not need to be rebuilt.

        KV cache lifecycle is managed by the engine's suspend/resume mechanism
        via KVCacheManagementMode in InferenceConfig.

        Yields:
            The dynamic inference engine for use during inference.
        """
        # Get the language module (unwrap from precision wrappers if needed)
        lang_module = self._get_lang_module()

        # Get config settings
        cuda_graph_impl = mcore_generation_config.get("cuda_graph_impl", "local")

        # In non-colocated mode, we don't need to suspend/resume the engine
        # between iterations since training runs on separate GPUs. The CUDA
        # graphs and KV cache can stay allocated. Only weight values change.
        needs_suspend_resume = self.is_generation_colocated

        # Save training state
        was_training = lang_module.training

        # === ENTER INFERENCE MODE ===

        # 1. Put model in eval mode
        lang_module.eval()

        # 2. Clear rotary position embedding caches (Megatron RL does this)
        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        has_lru_cache = rotary_module is not None and hasattr(rotary_module.forward, "cache_parameters")
        if has_lru_cache:
            rotary_module.forward.cache_clear()

        if cuda_graph_impl != "none":
            toggle_cuda_graphs(lang_module, set_to=cuda_graph_impl)

        # 4. Initialize inference engine if not already done
        if not self._inference_engine_initialized:
            self._initialize_inference_engine(mcore_generation_config)
            # Start the coordinator and engine loop (first time only)
            coordinator_port = self.cfg["generation"].get(
                "inference_coordinator_port", 5995
            )
            self._run_async_coordinator_start(coordinator_port)

        if self._inference_engine_alseep:
            self._wake()

        try:
            # Yield the inference engine for use
            yield self.dynamic_inference_engine

        finally:

            # 1. pause the inference engine (skip in non-colocated mode to
            #    avoid deleting and recreating CUDA graphs unnecessarily)
            if needs_suspend_resume and self._inference_engine_initialized and not self._inference_engine_alseep:
                self._sleep()

            # 2. Toggle CUDA graphs OFF (skip in non-colocated mode to keep them alive)
            if needs_suspend_resume and cuda_graph_impl != "none":
                toggle_cuda_graphs(lang_module, set_to="none")

            # 3. Clear rotary embedding cache again (Megatron RL does this on exit too)
            if has_lru_cache:
                rotary_module.forward.cache_clear()

            # 4. Restore training state (skip in non-colocated mode - model stays in eval)
            if needs_suspend_resume and was_training:
                lang_module.train()

            # 5. Force garbage collection and CUDA memory cleanup
            if needs_suspend_resume:
                gc.collect()
                torch.cuda.empty_cache()


    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using Megatron Core inference with coordinator.

        This method uses the coordinator-based inference pattern from Megatron Core,
        which enables better parallelism across data-parallel ranks through a central
        coordinator that routes requests to available engines.

        The inference engine is created once and reused across generate() calls.
        The engine is paused between generate() calls to free GPU memory for training.

        For coordinator-based inference:
        - Only DP rank 0 receives actual data and submits requests to the coordinator
        - Other DP ranks receive data=None but still participate in the inference engine loop
        - The coordinator distributes work across all DP engines
        - Results are broadcast from rank 0 to all ranks

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors,
                  or None for non-DP-0 workers (they participate in engine loop only)
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        from megatron.core.inference.sampling_params import SamplingParams

        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        
        dist_rank = torch.distributed.get_rank()
        is_request_submitter = (dist_rank == 0)
        
        # For non-rank-0 workers, data may be None (they participate in engine loop only)
        if data is not None:
            # Verify input is right padded
            assert isinstance(data, BatchedDataDict), (
                f"data must be a BatchedDataDict, got type: {type(data)}"
            )
            is_right_padded, error_msg = verify_right_padding(
                data, pad_value=self.tokenizer.pad_token_id
            )
            if not is_right_padded:
                warnings.warn(
                    f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
                )
        

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        # Use inference_mode context manager (mirrors megatron_rl_inference_mode from Megatron RL)
        # This handles: eval mode, CUDA graph toggle, engine init/resume, and cleanup
        with torch.no_grad(), self.inference_mode(mcore_generation_config) as inference_engine:
            # Handle None values for top_k - convert to integer as required by Megatron
            top_k_cfg = self.cfg["generation"]["top_k"]
            top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

            top_p_cfg = self.cfg["generation"]["top_p"]
            top_p_val = (
                0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
            )

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

            # Only rank 0 has actual data to submit
            if is_request_submitter:
                input_ids = data["input_ids"]
                print(f"[Rank {dist_rank}] input_ids: {input_ids.shape}")
                prompt_tokens_tensor = input_ids.cuda()
                prompt_lengths_tensor = data["input_lengths"]
            else:
                print(f"[Rank {dist_rank}] Participating in engine loop (no data to submit)")
                prompt_tokens_tensor = torch.empty(0, dtype=torch.long, device="cuda")
                prompt_lengths_tensor = torch.empty(0, dtype=torch.long, device="cuda")

            # Run the coordinator-based generation using the persistent engine
            # Rank 0 submits requests, other ranks participate in engine loop
            # Results are broadcast to all ranks inside this method
            result = self._run_async_generation_with_persistent_engine(
                prompt_tokens_tensor,
                prompt_lengths_tensor,
                sampling_params,
            )

        self.model.config.flash_decode = False

        # Context manager has exited - CUDA graphs are now disabled, model is back in train mode

        # Only rank 0 needs to format and return results
        # Other ranks return None (their results are ignored due to output_is_replicated)
        if not is_request_submitter:
            # Return empty result for non-submitter ranks
            # Use BatchedDataDict directly instead of from_batches to avoid padding issues with empty tensors
            return BatchedDataDict({
                "output_ids": torch.empty(0, 0, dtype=torch.long),
                "logprobs": torch.empty(0, 0, dtype=torch.float),
                "generation_lengths": torch.empty(0, dtype=torch.long),
                "unpadded_sequence_lengths": torch.empty(0, dtype=torch.long),
            }).to("cpu")

        input_lengths = data["input_lengths"]
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
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

        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            tokens = result[i].prompt_tokens.tolist() + result[i].generated_tokens
            logprobs = result[i].prompt_log_probs + result[i].generated_log_probs
            seq_len = len(tokens)
            output_ids_padded[i, :seq_len] = torch.tensor(
                tokens, dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(logprobs)
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                logprobs,
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    def _start_inference_loop_thread(self):
        """Start a background thread with a persistent event loop for inference.
        
        This thread runs the event loop that hosts the engine loop task.
        The loop runs forever until explicitly stopped.
        """
        import threading
        
        def run_loop():
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            self._inference_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._inference_loop)
            # Run forever - the engine loop task will run in this loop
            self._inference_loop.run_forever()
        
        self._inference_thread = threading.Thread(target=run_loop, daemon=True)
        self._inference_thread.start()
        
        # Wait for the loop to be created
        while self._inference_loop is None:
            time.sleep(0.001)

    def _run_async_coordinator_start(self, coordinator_port: int):
        """Start the coordinator and engine loop in the background thread.
        
        This is called once during the first generate() call to initialize
        the persistent inference infrastructure.
        """
        import concurrent.futures
        
        # Start the background thread with the event loop if not already running
        if self._inference_loop is None:
            self._start_inference_loop_thread()
        
        # Schedule the coordinator start in the inference loop
        future = asyncio.run_coroutine_threadsafe(
            self._start_inference_coordinator(coordinator_port),
            self._inference_loop
        )
        # Wait for completion
        return future.result()

    def _run_async_generation_with_persistent_engine(
        self,
        prompt_tokens_tensor: torch.Tensor,
        prompt_lengths_tensor: torch.Tensor,
        sampling_params: "SamplingParams",
    ) -> list:
        """Run generation using the persistent inference engine.
        
        This method uses the pre-initialized engine and client to run generation.
        Unlike the original method, it doesn't start/stop the coordinator each time.
        The async operation runs in the persistent inference loop.
        """
        if self._inference_loop is None:
            raise RuntimeError("Inference loop not initialized. Call generate() first.")
        
        # Schedule the generation in the inference loop
        future = asyncio.run_coroutine_threadsafe(
            self._generate_with_persistent_engine(
                prompt_tokens_tensor,
                prompt_lengths_tensor,
                sampling_params,
            ),
            self._inference_loop
        )
        # Wait for completion and return the result
        return future.result()

    async def _generate_with_persistent_engine(
        self,
        prompt_tokens_tensor: torch.Tensor,
        prompt_lengths_tensor: torch.Tensor,
        sampling_params: "SamplingParams",
    ) -> list:
        """Run generation using the persistent coordinator-based inference.
        
        This method uses the already-running engine and submits requests through
        the persistent client. The engine loop continues running between calls.

        For coordinator-based inference with centralized request submission:
        - Only rank 0 (the request submitter) submits requests and collects results
        - Other ranks return early but their engine loops continue running in the
          background, processing requests distributed by the coordinator
        - No broadcast is needed since only rank 0's results are used by the caller

        Args:
            prompt_tokens_tensor: Tensor of prompt token IDs [batch_size, seq_len]
            prompt_lengths_tensor: Tensor of prompt lengths [batch_size]
            sampling_params: Sampling parameters for generation

        Returns:
            List of completed request records sorted by request_id (rank 0),
            or empty list (other ranks)
        """
        from megatron.core.inference.inference_request import DynamicInferenceRequestRecord

        dist_rank = torch.distributed.get_rank()
        
        if dist_rank == 0:
            assert self.inference_client is not None, "Inference client not initialized"
        
        # Non-rank-0 workers: return immediately with empty results
        # Their engine loops will continue processing requests from the coordinator
        # in the background (the engine loop runs as a separate task in _inference_loop)
        if dist_rank != 0:
            print(f"[Rank {dist_rank}] Participating in engine loop only (not submitting requests)")
            # Return empty results - the caller only uses rank 0's results
            return []
        
        # Rank 0: submit ALL requests and collect results
        print(f"[Rank {dist_rank}] Submitting {prompt_tokens_tensor.size(0)} requests to coordinator")
        
        futures = []
        for request_id, (prompt_tokens, prompt_len) in enumerate(
            zip(prompt_tokens_tensor, prompt_lengths_tensor, strict=True)
        ):
            # Extract the actual prompt tokens (without padding) and convert to list
            prompt = prompt_tokens[: prompt_len.item()].tolist()
            future = self.inference_client.add_request(prompt, sampling_params)
            futures.append(future)

        # Wait for all requests to complete
        # The coordinator distributes work to all DP engines, including this one
        completed_records: list[DynamicInferenceRequestRecord] = await asyncio.gather(
            *futures
        )

        # Extract the merged request from each record
        results = [record.merge() for record in completed_records]

        # Sort by request_id to maintain original batch order
        results.sort(key=lambda x: x.request_id)
        
        print(f"[Rank {dist_rank}] Completed {len(results)} requests")

        return results


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

    def init_refit_collective(self, ip, port, world_size, rank_offset, refit_backend="gloo"):
        """Initialize the refit collective for non-colocated Megatron weight transfer.

        Creates a Gloo-backed ProcessGroup spanning training and inference
        workers for metadata exchange (all_gather_object, broadcast), and a
        CopyService for the actual data transfer (GlooCopyService for
        CPU-staged P2P, or NVSHMEMCopyService for GPU-direct transfers).

        Args:
            ip: IP address for the process group rendezvous.
            port: Port for the process group rendezvous.
            world_size: Total world size (train + inference workers).
            rank_offset: Offset for this side's ranks (0 for training, train_ws for inference).
            refit_backend: Copy service backend ("gloo" or "nvshmem").
        """
        from torch.distributed.distributed_c10d import (
            PrefixStore,
            ProcessGroup,
            ProcessGroupGloo,
            _world,
        )

        local_rank = torch.distributed.get_rank()
        global_rank = local_rank + rank_offset
        self.refit_rank_offset = rank_offset

        # port+1 to avoid collision with the caller's rendezvous on `port`.
        store = torch.distributed.TCPStore(
            host_name=ip,
            port=port + 1,
            world_size=world_size,
            is_master=(global_rank == 0),
        )

        group_name = "refit"
        pg_prefix_store = PrefixStore(f"{group_name}/", store)

        # Training and inference workers run in separate torch.distributed worlds
        # (each has its own init_process_group). The public APIs (new_group,
        # init_process_group) assume all ranks belong to one world — new_group
        # validates ranks against the default PG, and init_process_group can only
        # be called once. We construct the PG manually using the same internal
        # pattern as _new_process_group_helper, skipping the single-world
        # assumptions.
        pg = ProcessGroup(pg_prefix_store, global_rank, world_size)
        gloo_store = PrefixStore("cpu/", pg_prefix_store)
        gloo_backend = ProcessGroupGloo(gloo_store, global_rank, world_size)
        gloo_backend._set_sequence_number_for_group()
        pg._register_backend(
            torch.device("cpu"),
            ProcessGroup.BackendType.GLOO,
            gloo_backend,
        )
        pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        pg._set_group_name(group_name)

        self.refit_pg = pg

        # Register in torch.distributed's global state so that high-level ops
        # (all_gather_object, broadcast_object_list) work with this PG.
        # These ops internally call get_rank(group) which looks up pg_group_ranks,
        # and use pg_map for backend dispatch. The identity mapping works because
        # our global_rank space (0..world_size-1) is already the group rank space.
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        _world.pg_map[pg] = ("gloo", pg_prefix_store)
        _world.pg_names[pg] = group_name

        if refit_backend == "nvshmem":
            from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
            self.refit_copy_service = NVSHMEMCopyService(group=self.refit_pg)
            self.refit_copy_service._ensure_initialized()
        else:
            from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
            self.refit_copy_service = GlooCopyService(group=self.refit_pg)

        print(f"[REFIT] rank={global_rank} init_refit_collective complete", flush=True)

    @torch.no_grad()
    def swap_weights_via_reshard(self, is_source: bool, dst_rank_offset: int = 0) -> bool:
        """Transfer weights using Megatron's swap_model_weights resharding API.

        Uses the CopyService initialized in init_refit_collective for data
        transfer and the refit ProcessGroup for metadata exchange.

        Args:
            is_source: True for training workers (weight source), False for inference
                       workers (weight destination).
            dst_rank_offset: Rank offset of the inference (destination) side.

        Returns:
            True on success.
        """
        from megatron.core.resharding.refit import swap_model_weights

        if is_source:
            src_model = self.model
            dst_model = None
        else:
            src_model = None
            dst_model = self.model

        swap_model_weights(
            src_model, dst_model,
            refit_method=self.refit_copy_service,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=dst_rank_offset,
        )
        return True

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
