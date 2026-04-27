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

import contextlib
import gc
import warnings
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generator, Optional

import ray
import torch
from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
)
from nemo_automodel.components.distributed.cp_utils import (
    get_train_context as get_train_context_automodel,
)
from nemo_automodel.components.distributed.tensor_utils import (
    get_cpu_state_dict,
    to_local_if_dtensor,
)
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager
from nemo_rl.models.automodel.data import (
    check_sequence_dim,
    get_microbatch_iterator,
    install_thd_squeeze_hook,
    process_global_batch,
)
from nemo_rl.models.automodel.pipeline_parallel import (
    PPLossAdapter,
    broadcast_tensors_from_last_pp_stage,
    pp_forward_loop,
    pp_train_forward_backward_loop,
)
from nemo_rl.models.automodel.setup import (
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_prepare_config,
)
from nemo_rl.models.automodel.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    ScorePostProcessor,
    TopkLogitsPostProcessor,
    aggregate_training_statistics,
    automodel_forward_backward,
    forward_with_post_processing_fn,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ScoreOutputSpec,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import (
    apply_transformer_engine_patch,
)
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer


def dtensor_params_generator(
    model: nn.Module, target_dtype: torch.dtype
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Generator that yields (name, tensor) pairs, converting DTensors to local tensors and adapting to HF format.

    Args:
        model: The model whose parameters to generate.
        target_dtype: The dtype to convert tensors to.
        peft_config: Optional LoRA config for filtering which layers to merge.

    Yields:
        Tuples of (fully_qualified_name, tensor) where tensors are converted to target dtype and made contiguous.
    """
    module_map = dict(model.named_modules())
    for name, tensor in model.state_dict().items():
        if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
            continue
        full_tensor = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        merged_tensor = _maybe_merge_lora_weight(module_map, name, full_tensor)

        adapted_fqn_tensors = _maybe_adapt_tensor_to_hf(model, name, merged_tensor)
        for adapted_fqn, adapted_tensor in adapted_fqn_tensors:
            # Convert to target dtype
            yield (
                adapted_fqn,
                adapted_tensor.to(target_dtype, non_blocking=True).contiguous(),
            )
            del adapted_tensor
        del adapted_fqn_tensors
        del merged_tensor
        del full_tensor


@torch.no_grad()
def _maybe_merge_lora_weight(
    module_map: dict[str, nn.Module],
    fqn: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if not fqn.endswith(".weight"):
        return tensor
    module_name = fqn[: -len(".weight")]
    module = module_map.get(module_name)
    if not isinstance(module, LinearLoRA):
        return tensor
    if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
        return tensor

    lora_a = (
        module.lora_A.weight.full_tensor()
        if isinstance(module.lora_A.weight, DTensor)
        else module.lora_A.weight
    )
    lora_b = (
        module.lora_B.weight.full_tensor()
        if isinstance(module.lora_B.weight, DTensor)
        else module.lora_B.weight
    )
    lora_a = lora_a.to(device=tensor.device, dtype=tensor.dtype)
    lora_b = lora_b.to(device=tensor.device, dtype=tensor.dtype)
    scale = getattr(module, "scale", None)

    if scale is None and hasattr(module, "alpha") and hasattr(module, "dim"):
        scale = module.alpha / module.dim
    if scale is None:
        scale = 1.0

    return tensor + torch.matmul(lora_b, lora_a) * scale


def _maybe_adapt_tensor_to_hf(
    model_part: nn.Module, fqn: str, tensor: torch.Tensor, quantization: bool = False
) -> list[tuple[str, torch.Tensor]]:
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        return adapter.convert_single_tensor_to_hf(
            fqn,
            tensor,
            exclude_key_regex=r".*_extra_state.*",
            quantization=quantization,
        )
    return [(fqn, tensor)]


@contextlib.contextmanager
def get_train_context(
    cp_size: int,
    cp_mesh: Any,
    cp_buffers: list,
    sequence_dim: int,
    dtype: torch.dtype,
    autocast_enabled: bool = True,
) -> Generator[None, None, None]:
    """Create combined context manager for training with context parallel and autocast."""
    with contextlib.ExitStack() as stack:
        context_parallel_ctx = None
        if cp_size > 1 and cp_buffers:
            # Create DTensor CP context. Skip when cp_buffers is empty.
            context_parallel_ctx = create_context_parallel_ctx(
                cp_mesh=cp_mesh,
                cp_buffers=cp_buffers,
                cp_seq_dims=[sequence_dim] * len(cp_buffers),
                cp_no_restore_buffers=set(cp_buffers),
            )

        stack.enter_context(
            get_train_context_automodel(False, False, context_parallel_ctx)()
        )
        if autocast_enabled:
            stack.enter_context(torch.autocast(device_type="cuda", dtype=dtype))
        yield


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
# This is useful when using worker extension classes.
class DTensorPolicyWorkerV2Impl(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self) -> str:
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
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        """Initialize the DTensorPolicyWorkerV2."""
        # Apply TE patch until TE is upgraded to 2.10.0
        apply_transformer_engine_patch()

        # Store configuration
        self.cfg = config

        # Reconstruct tokenizer/processor locally to avoid pickling across
        # incompatible transformers versions (v4 head node → v5 worker).
        from nemo_rl.models.automodel.setup import get_tokenizer

        use_processor = config["tokenizer"].get("use_processor", False)
        result = get_tokenizer(config["tokenizer"], get_processor=use_processor)
        if use_processor:
            self.processor = result
            self.tokenizer = result.tokenizer
        else:
            self.tokenizer = result
            self.processor = None
        self.is_vlm = self.processor is not None
        self.lora_enabled = (
            config["dtensor_cfg"].get("lora_cfg", {}).get("enabled", False)
        )

        print(f"Initializing DTensorPolicyWorkerV2 with is_vlm={self.is_vlm}")

        # Initialize checkpoint manager
        self.checkpoint_manager: Optional[AutomodelCheckpointManager] = None

        # Validate configuration and prepare runtime settings
        runtime_config = validate_and_prepare_config(
            config=config,
            processor=self.processor,
            rank=0,  # Temporary, will be updated after distributed init
        )

        # Set up distributed environment (returns DistributedContext)
        distributed_context = setup_distributed(
            config=config,
            runtime_config=runtime_config,
        )
        # Set instance attributes from distributed context
        self.rank = torch.distributed.get_rank()
        self.device_mesh = distributed_context.device_mesh
        self.dp_cp_mesh = self.device_mesh["dp_cp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_mesh = self.device_mesh["tp"]
        self.cp_mesh = self.device_mesh["cp"]
        self.moe_mesh = distributed_context.moe_mesh
        self.dp_size = distributed_context.dp_size
        self.tp_size = distributed_context.tp_size
        self.cp_size = distributed_context.cp_size
        self.pp_size = distributed_context.pp_size
        self.pp_mesh = distributed_context.pp_mesh
        self.pp_enabled = self.pp_size > 1

        # Initialize checkpoint manager now that distributed is set up
        self._init_checkpoint_manager(
            config_updates={
                "model_repo_id": config["model_name"],
                "dequantize_base_checkpoint": config.get(
                    "dequantize_base_checkpoint", False
                ),
                "is_peft": self.lora_enabled,
                "is_async": True,
            },
        )

        # Set up model and optimizer
        model_and_optimizer_state = setup_model_and_optimizer(
            config=config,
            tokenizer=self.tokenizer,
            runtime_config=runtime_config,
            distributed_context=distributed_context,
            checkpoint_manager=self.checkpoint_manager,
            is_vlm=self.is_vlm,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )

        # Set instance attributes from model and optimizer state
        (
            self.model_handle,
            self.optimizers,
            self.schedulers,
            self.is_hf_model,
            self.is_moe_model,
            self._is_reward_model,
            self.model_class,
            self.model_config,
            self.peft_config,
            self.autocast_enabled,
        ) = model_and_optimizer_state

        # Convenience aliases
        self.model = self.model_handle.raw  # raw AutoPipeline or nn.Module
        self.has_first_stage = self.model_handle.has_first_stage
        self.has_last_stage = self.model_handle.has_last_stage

        # Initialize reference model if requested
        self.reference_model_state_dict = None
        if init_reference_model:
            self.reference_model_state_dict = setup_reference_model_state(
                self.model_handle
            )

        # Set instance attributes from runtime config (tuple unpacking)
        (
            self.model_class,  # Already set above, but includes in tuple for completeness
            self.model_config,  # Already set above, but includes in tuple for completeness
            self.hf_config_overrides,
            self.allow_flash_attn_args,
            self.attn_impl,
            self.dtype,
            self.enable_seq_packing,
            self.max_grad_norm,
            self.cpu_offload,
            self.offload_optimizer_for_logprob,
            self.is_generation_colocated,
            self.sampling_params,
            _runtime_is_reward_model,  # Duplicate, already set as _is_reward_model
        ) = runtime_config

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dp_mesh.get_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        # Validate sequence dimension
        sequence_dim, _ = check_sequence_dim(data)

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model_handle.eval()
        else:
            ctx = nullcontext()
            self.model_handle.train()

        if self.pp_enabled:
            return self._train_pp(
                data=data,
                loss_fn=loss_fn,
                eval_mode=eval_mode,
                local_gbs=local_gbs,
                num_global_batches=num_global_batches,
                sequence_dim=sequence_dim,
                ctx=ctx,
                optimizers_list=self.optimizers,
                schedulers_list=self.schedulers,
            )

        # --- Non-PP path (existing) ---

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=loss_fn,
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            enable_seq_packing=self.enable_seq_packing,
            sampling_params=self.sampling_params,
        )

        # Create train context factory
        def train_context_fn(processed_inputs):
            return get_train_context(
                cp_size=self.cp_size,
                cp_mesh=self.cp_mesh,
                cp_buffers=processed_inputs.cp_buffers,
                sequence_dim=sequence_dim,
                dtype=self.dtype,
                autocast_enabled=self.autocast_enabled,
            )

        # Setup cache clearing callback if configured
        empty_cache_steps = self.cfg.get("dtensor_cfg", {}).get(
            "clear_cache_every_n_steps"
        )
        if empty_cache_steps:
            warnings.warn(
                f"Emptying cache every {empty_cache_steps} microbatches; doing so unnecessarily would incur a large performance overhead.",
            )

        def on_microbatch_start(mb_idx):
            if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                torch.cuda.empty_cache()

        with ctx:
            # Get data from batch and move to device
            data = data.to("cuda")

            losses = []
            all_mb_metrics = []
            for gb_idx in range(num_global_batches):
                # Process global batch and compute normalization factors
                gb_result = process_global_batch(
                    data,
                    loss_fn,
                    self.dp_mesh.get_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                for opt in self.optimizers:
                    opt.zero_grad()

                # Get microbatch iterator based on batching strategy
                processed_iterator, iterator_len = get_microbatch_iterator(
                    batch,
                    self.cfg,
                    mbs,
                    self.dp_mesh,
                    tokenizer=self.tokenizer,
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    device_mesh=self.device_mesh,
                )

                # Use automodel_forward_backward for the training loop
                mb_results = automodel_forward_backward(
                    model=self.model,
                    data_iterator=processed_iterator,
                    post_processing_fn=loss_post_processor,
                    forward_only=eval_mode,
                    is_reward_model=self._is_reward_model,
                    allow_flash_attn_args=self.allow_flash_attn_args,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                    sampling_params=self.sampling_params,
                    sequence_dim=sequence_dim,
                    dp_size=self.dp_size,
                    cp_size=self.cp_size,
                    num_global_batches=num_global_batches,
                    train_context_fn=train_context_fn,
                    num_valid_microbatches=iterator_len,
                    on_microbatch_start=on_microbatch_start,
                )

                # Extract losses and metrics from results
                mb_losses = []
                for mb_idx, (loss, loss_metrics) in enumerate(mb_results):
                    # Only process valid (non-dummy) batches for metrics
                    if mb_idx < iterator_len:
                        num_valid_samples = loss_metrics["num_valid_samples"]
                        loss_metrics["lr"] = self.optimizers[0].param_groups[0]["lr"]
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()

                        if num_valid_samples > 0:
                            mb_losses.append(loss.item())
                            all_mb_metrics.append(loss_metrics)

                grad_norm: Optional[float | torch.Tensor] = None
                if not eval_mode:
                    grad_norm = scale_grads_and_clip_grad_norm(
                        self.max_grad_norm,
                        [self.model],
                        norm_type=2.0,
                        pp_enabled=False,
                        device_mesh=self.device_mesh,
                        moe_mesh=self.moe_mesh,
                        ep_axis_name="ep"
                        if self.moe_mesh is not None
                        and "ep" in self.moe_mesh.mesh_dim_names
                        else None,
                        pp_axis_name=None,
                        foreach=True,
                        num_label_tokens=1,
                        dp_group_size=self.dp_size * self.cp_size,
                    )
                    grad_norm = torch.tensor(
                        grad_norm, device="cpu", dtype=torch.float32
                    )

                    # Update parameters
                    for opt in self.optimizers:
                        opt.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # release gradient memory before rollouts
            for opt in self.optimizers:
                opt.zero_grad()
            # increment scheduler after all batches in rollout are processed
            if not eval_mode:
                for sched in self.schedulers:
                    sched.step()
            # dynamic batch and sequence dims causes alot of fragmentation, so clear
            # the memory allocator before moving on
            torch.cuda.empty_cache()

            # Aggregate training statistics across microbatches and ranks
            metrics = aggregate_training_statistics(
                losses=losses,
                all_mb_metrics=all_mb_metrics,
                grad_norm=grad_norm,
                dp_group=self.dp_mesh.get_group(),
                dtype=self.dtype,
            )

            return metrics

    def _train_pp(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool,
        local_gbs: int,
        num_global_batches: int,
        sequence_dim: int,
        ctx: AbstractContextManager[Any],
        optimizers_list: list,
        schedulers_list: list,
    ) -> dict[str, Any]:
        """PP training path using pp_train_forward_backward_loop."""
        loss_adapter = PPLossAdapter(
            loss_fn=loss_fn,
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            enable_seq_packing=self.enable_seq_packing,
            sampling_params=self.sampling_params,
        )

        pp_batch_size = self.model.pp_batch_size
        n_microbatches = self.model.info.schedule._n_microbatches

        # Install hook to squeeze batch dim for THD format in PP microbatches.
        # The PP schedule splits [n_chunks, tokens_per_chunk] → [1, tokens_per_chunk]
        # per microbatch, but custom models need [tokens_per_chunk] (1D) for THD.
        thd_hooks: list = []
        if self.enable_seq_packing and not self.is_hf_model:
            thd_hooks = install_thd_squeeze_hook(self.model_handle.parts)

        with ctx:
            data = data.to("cuda")
            losses = []
            all_mb_metrics = []
            grad_norm: Optional[float | torch.Tensor] = None

            for gb_idx in range(num_global_batches):
                gb_result = process_global_batch(
                    data,
                    loss_fn,
                    self.dp_mesh.get_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                for opt in optimizers_list:
                    opt.zero_grad()

                gb_total_loss, gb_metrics = pp_train_forward_backward_loop(
                    model=self.model,
                    model_parts=self.model_handle.parts,
                    batch=gb_result["batch"],
                    loss_adapter=loss_adapter,
                    pp_batch_size=pp_batch_size,
                    n_microbatches=n_microbatches,
                    has_last_stage=self.has_last_stage,
                    pp_mesh=self.pp_mesh,
                    enable_seq_packing=self.enable_seq_packing,
                    is_hf_model=self.is_hf_model,
                    tokenizer_eos_id=self.tokenizer.eos_token_id,
                    train_mb_tokens=self.cfg["sequence_packing"]["train_mb_tokens"],
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    device_mesh=self.device_mesh,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                    num_global_batches=num_global_batches,
                    forward_only=eval_mode,
                )

                # Enrich metrics with lr and global counts
                if gb_metrics:
                    for m in gb_metrics:
                        lr = optimizers_list[0].param_groups[0]["lr"]
                        m["lr"] = lr
                        m["global_valid_seqs"] = global_valid_seqs.item()
                        m["global_valid_toks"] = global_valid_toks.item()
                        if m.get("num_valid_samples", 0) > 0:
                            all_mb_metrics.append(m)

                if not eval_mode:
                    # PP gradient scaling: each schedule.step() scales loss by
                    # dp_size*cp_size (cancelling FSDP averaging), matching the
                    # non-PP path. Set num_label_tokens = dp_group_size so the
                    # PP scaling in scale_grads_and_clip_grad_norm is a no-op
                    # (divides by 1). Gradient accumulation naturally sums across
                    # steps — this matches non-PP behavior where the full batch
                    # gradient is computed in one pass.
                    dp_group_size = self.dp_size * self.cp_size
                    grad_norm = scale_grads_and_clip_grad_norm(
                        self.max_grad_norm,
                        self.model_handle.parts,
                        norm_type=2.0,
                        pp_enabled=True,
                        device_mesh=self.device_mesh,
                        moe_mesh=self.moe_mesh,
                        ep_axis_name="ep"
                        if self.moe_mesh is not None
                        and "ep" in self.moe_mesh.mesh_dim_names
                        else None,
                        pp_axis_name="pp",
                        foreach=True,
                        num_label_tokens=dp_group_size,
                        dp_group_size=dp_group_size,
                    )
                    grad_norm = torch.tensor(
                        grad_norm, device="cpu", dtype=torch.float32
                    )
                    for opt in optimizers_list:
                        opt.step()

                # Broadcast loss from last PP stage so all ranks report correct value.
                # Undo dp*cp scaling from PPLossAdapter (needed for FSDP grad averaging
                # during training, but not for the reported loss).
                if self.pp_size > 1:
                    loss_tensor = gb_total_loss.unsqueeze(0)
                    broadcasted = broadcast_tensors_from_last_pp_stage(
                        {"loss": loss_tensor if self.has_last_stage else None},
                        self.pp_mesh,
                        self.has_last_stage,
                    )
                    gb_total_loss = broadcasted["loss"].squeeze(0) / (
                        self.dp_size * self.cp_size
                    )
                losses.append(gb_total_loss.item())

            for opt in optimizers_list:
                opt.zero_grad()
            if not eval_mode:
                for sched in schedulers_list:
                    sched.step()
            torch.cuda.empty_cache()

            # Remove THD squeeze hooks
            for h in thd_hooks:
                h.remove()

            metrics = aggregate_training_statistics(
                losses=losses,
                all_mb_metrics=all_mb_metrics,
                grad_norm=grad_norm,
                dp_group=self.dp_mesh.get_group(),
                dtype=self.dtype,
            )
            return metrics

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/get_logprobs")
    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # Validate sequence dimension. Sync across ranks for PP (different ranks
        # may have different max sequence lengths from GRPO generation).
        sequence_dim, seq_dim_size = check_sequence_dim(data)
        if self.pp_enabled:
            max_seq = torch.tensor([seq_dim_size], device="cuda")
            torch.distributed.all_reduce(max_seq, op=torch.distributed.ReduceOp.MAX)
            seq_dim_size = int(max_seq.item())
            return self._get_logprobs_pp(data, logprob_batch_size, seq_dim_size)

        all_log_probs = []
        self.model.eval()

        # Create logprobs post-processor
        logprobs_post_processor = LogprobsPostProcessor(
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            enable_seq_packing=self.enable_seq_packing,
            sampling_params=self.sampling_params,
        )

        with torch.no_grad():
            data.to("cuda")
            # Get microbatch iterator based on batching strategy
            processed_iterator, iterator_len = get_microbatch_iterator(
                data,
                self.cfg,
                logprob_batch_size,
                self.dp_mesh,
                tokenizer=self.tokenizer,
                cp_size=self.cp_size,
                cp_mesh=self.cp_mesh,
                device_mesh=self.device_mesh,
            )

            for batch_idx, processed_mb in enumerate(processed_iterator):
                processed_inputs = processed_mb.processed_inputs

                with get_train_context(
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    cp_buffers=processed_inputs.cp_buffers,
                    sequence_dim=sequence_dim,
                    dtype=self.dtype,
                    autocast_enabled=self.autocast_enabled,
                ):
                    # Use forward_with_post_processing_fn for forward pass and post-processing
                    token_logprobs, _metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        post_processing_fn=logprobs_post_processor,
                        processed_mb=processed_mb,
                        is_reward_model=False,
                        allow_flash_attn_args=self.allow_flash_attn_args,
                        sampling_params=self.sampling_params,
                        sequence_dim=sequence_dim,
                    )

                # skip keeping the logprobs for the dummy batches
                if batch_idx >= iterator_len:
                    continue

                all_log_probs.append(token_logprobs)

        # Concatenate all batches
        return_data = BatchedDataDict[LogprobOutputSpec]()

        all_log_probs_padded = []
        for lp in all_log_probs:
            padding_needed = seq_dim_size - lp.shape[1]
            if padding_needed > 0:
                lp = torch.nn.functional.pad(
                    lp, (0, padding_needed), mode="constant", value=0.0
                )
            all_log_probs_padded.append(lp)
        return_data["logprobs"] = torch.cat(all_log_probs_padded, dim=0).cpu()

        return return_data

    def _get_logprobs_pp(
        self,
        data: BatchedDataDict[Any],
        logprob_batch_size: int,
        seq_dim_size: int,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """PP path for logprob computation using pp_forward_loop."""
        for mp in self.model_handle.parts:
            mp.eval()

        post_processor = LogprobsPostProcessor(
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            enable_seq_packing=self.enable_seq_packing,
            sampling_params=self.sampling_params,
        )

        with torch.no_grad():
            result = pp_forward_loop(
                model=self.model,
                data=data,
                post_processor=post_processor,
                pp_batch_size=self.model.pp_batch_size,
                seq_dim_size=seq_dim_size,
                pp_mesh=self.pp_mesh,
                has_first_stage=self.has_first_stage,
                has_last_stage=self.has_last_stage,
            )

        return_data = BatchedDataDict[LogprobOutputSpec]()
        return_data["logprobs"] = result["logprobs"].cpu()
        return return_data

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/score")
    def score(self, data: BatchedDataDict) -> BatchedDataDict[ScoreOutputSpec]:
        global_batch_size = min(self.cfg["batch_size"], data.size)

        # Validate sequence dimension
        sequence_dim, _ = check_sequence_dim(data)

        self.model.eval()
        print("Begin to batch datas")

        # Create score post-processor
        score_post_processor = ScorePostProcessor(cfg=self.cfg)

        with torch.no_grad():
            data.to("cuda")
            # Get microbatch iterator based on batching strategy
            processed_iterator, iterator_len = get_microbatch_iterator(
                data,
                self.cfg,
                global_batch_size,
                self.dp_mesh,
                tokenizer=self.tokenizer,
                cp_size=self.cp_size,
                cp_mesh=self.cp_mesh,
                device_mesh=self.device_mesh,
            )

            all_rm_scores = []
            for batch_idx, processed_mb in enumerate(processed_iterator):
                processed_inputs = processed_mb.processed_inputs

                with get_train_context(
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    cp_buffers=processed_inputs.cp_buffers,
                    sequence_dim=sequence_dim,
                    dtype=self.dtype,
                    autocast_enabled=self.autocast_enabled,
                ):
                    # Use forward_with_post_processing_fn for forward pass and post-processing
                    rm_scores, _metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        post_processing_fn=score_post_processor,
                        processed_mb=processed_mb,
                        is_reward_model=True,
                        allow_flash_attn_args=False,
                        sampling_params=self.sampling_params,
                        sequence_dim=sequence_dim,
                    )

                # skip keeping the scores for the dummy batches
                if batch_idx >= iterator_len:
                    continue

                all_rm_scores.append(rm_scores)

        all_rm_scores = torch.cat(all_rm_scores, dim=0)
        all_rm_scores = all_rm_scores.squeeze(-1).cpu()
        return_data = BatchedDataDict[ScoreOutputSpec](
            {
                "scores": all_rm_scores,
            }
        )
        return return_data

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/get_topk_logits")
    def get_topk_logits(
        self,
        data: BatchedDataDict[Any],
        k: int,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Return per-position top-k logits and corresponding global indices.

        Notes:
        - Return shapes are [B, S, k].
        - Computes top-k over the full sequence (no trimming of the last position).
        - If alignment with next-token targets is required, the caller should handle it.
        - If logits are TP-sharded DTensor, performs distributed global top-k across TP.
        - Supports context parallelism with proper CP gather.
        - Otherwise, computes local top-k on full-vocab tensor.
        """
        topk_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # Validate sequence dimension
        sequence_dim, seq_dim_size = check_sequence_dim(data)

        if self.pp_enabled:
            return self._get_topk_logits_pp(data, k, topk_batch_size, seq_dim_size)

        out_topk_vals = []
        out_topk_idx = []
        self.model.eval()

        # Create top-k post-processor
        topk_post_processor = TopkLogitsPostProcessor(
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            k=k,
            enable_seq_packing=self.enable_seq_packing,
        )

        with torch.no_grad():
            data.to("cuda")
            # Get microbatch iterator based on batching strategy
            processed_iterator, iterator_len = get_microbatch_iterator(
                data,
                self.cfg,
                topk_batch_size,
                self.dp_mesh,
                tokenizer=self.tokenizer,
                cp_size=self.cp_size,
                cp_mesh=self.cp_mesh,
                device_mesh=self.device_mesh,
            )

            for batch_idx, processed_mb in enumerate(processed_iterator):
                processed_inputs = processed_mb.processed_inputs

                with get_train_context(
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    cp_buffers=processed_inputs.cp_buffers,
                    sequence_dim=sequence_dim,
                    dtype=self.dtype,
                    autocast_enabled=self.autocast_enabled,
                ):
                    # Use forward_with_post_processing_fn for forward pass and post-processing
                    (vals, idx), _metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        post_processing_fn=topk_post_processor,
                        processed_mb=processed_mb,
                        is_reward_model=False,
                        allow_flash_attn_args=self.allow_flash_attn_args,
                        sampling_params=self.sampling_params,
                        sequence_dim=sequence_dim,
                    )

                # skip keeping the topk values for the dummy batches
                if batch_idx >= iterator_len:
                    continue

                # Keep only real sequence tokens (no trimming here; padded positions can be masked downstream)
                # Shapes remain [B, S, k].
                out_topk_vals.append(vals.cpu())
                out_topk_idx.append(idx.cpu())

        ret = BatchedDataDict[Any]()
        # Pad each micro-batch result on sequence dim to common length (S), similar to get_logprobs
        all_topk_vals_padded = []
        all_topk_idx_padded = []
        target_seq_len = seq_dim_size
        for vals, idx in zip(out_topk_vals, out_topk_idx):
            pad_needed = target_seq_len - vals.shape[1]
            if pad_needed > 0:
                # pad along sequence dimension (second dim): (last_dim_pad_left, last_dim_pad_right, seq_pad_left, seq_pad_right, batch_pad_left, batch_pad_right)
                vals = torch.nn.functional.pad(
                    vals, (0, 0, 0, pad_needed, 0, 0), mode="constant", value=0.0
                )
                idx = torch.nn.functional.pad(
                    idx, (0, 0, 0, pad_needed, 0, 0), mode="constant", value=0
                )
            all_topk_vals_padded.append(vals)
            all_topk_idx_padded.append(idx)

        ret["topk_logits"] = (
            torch.cat(all_topk_vals_padded, dim=0)
            if len(all_topk_vals_padded) > 1
            else all_topk_vals_padded[0]
        ).cpu()
        ret["topk_indices"] = (
            torch.cat(all_topk_idx_padded, dim=0)
            if len(all_topk_idx_padded) > 1
            else all_topk_idx_padded[0]
        ).cpu()
        return ret

    def _get_topk_logits_pp(
        self,
        data: BatchedDataDict[Any],
        k: int,
        topk_batch_size: int,
        seq_dim_size: int,
    ) -> BatchedDataDict[Any]:
        """PP path for top-k logits using pp_forward_loop."""
        for mp in self.model_handle.parts:
            mp.eval()

        post_processor = TopkLogitsPostProcessor(
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            k=k,
            enable_seq_packing=self.enable_seq_packing,
        )

        with torch.no_grad():
            result = pp_forward_loop(
                model=self.model,
                data=data,
                post_processor=post_processor,
                pp_batch_size=self.model.pp_batch_size,
                seq_dim_size=seq_dim_size,
                pp_mesh=self.pp_mesh,
                has_first_stage=self.has_first_stage,
                has_last_stage=self.has_last_stage,
            )

        ret = BatchedDataDict[Any]()
        ret["topk_logits"] = result["topk_logits"].cpu()
        ret["topk_indices"] = result["topk_indices"].cpu()
        return ret

    def _all_params_generator(self):
        """Yield (name, tensor) pairs for ALL model params, gathering across PP ranks.

        For non-PP: yields from the single model's state dict.
        For PP: each rank has a subset of params. All PP ranks cooperate to
        broadcast each param one at a time (like megatron bridge's
        stream_weights_megatron_to_hf) so every rank can yield the full model
        without holding all remote params in memory.
        """
        if not self.pp_enabled:
            yield from dtensor_params_generator(self.model, self.dtype)
            return

        # Phase 1: build a lazy local param lookup (name → part index).
        # We don't materialise all tensors yet to save memory.
        local_names: set[str] = set()
        for part in self.model_handle.parts:
            for name in part.state_dict().keys():
                if not (
                    name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")
                ):
                    local_names.add(name)

        # Phase 2: all-gather the name lists to build a global ordering.
        pp_group = self.pp_mesh.get_group()
        pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
        all_names_list: list[list[str]] = [None] * self.pp_size  # type: ignore[list-item]
        torch.distributed.all_gather_object(
            all_names_list, sorted(local_names), group=pp_group
        )

        name_to_owner: dict[str, int] = {}
        for pp_r, names in enumerate(all_names_list):
            for n in names:
                name_to_owner[n] = pp_r

        my_pp_rank = self.pp_mesh.get_local_rank()

        # Phase 3: iterate all params in sorted order. Owner broadcasts one
        # tensor at a time; others allocate a receive buffer and receive.
        for name in sorted(name_to_owner.keys()):
            owner_pp_rank = name_to_owner[name]
            owner_global_rank = pp_ranks[owner_pp_rank]

            # Owner adapts the (name, tensor) pairs and broadcasts the adapted
            # list along with metadata + tensor data; non-owners receive and
            # yield without re-adapting (the owner's adapter is authoritative).
            if my_pp_rank == owner_pp_rank:
                tensor, owning_part = self._get_local_param_tensor(name)
                tensor = tensor.cuda()
                adapted_pairs = list(
                    _maybe_adapt_tensor_to_hf(owning_part, name, tensor)
                )
                meta = [(an, list(at.shape), str(at.dtype)) for an, at in adapted_pairs]
                torch.distributed.broadcast_object_list(
                    [meta], src=owner_global_rank, group=pp_group
                )
                for _, adapted_tensor in adapted_pairs:
                    contig = adapted_tensor.contiguous()
                    torch.distributed.broadcast(
                        contig, src=owner_global_rank, group=pp_group
                    )
                for adapted_name, adapted_tensor in adapted_pairs:
                    yield (
                        adapted_name,
                        adapted_tensor.to(self.dtype, non_blocking=True).contiguous(),
                    )
            else:
                meta_list = [None]
                torch.distributed.broadcast_object_list(
                    meta_list, src=owner_global_rank, group=pp_group
                )
                meta = meta_list[0]
                for adapted_name, shape, dtype_str in meta:
                    dtype = getattr(torch, dtype_str.replace("torch.", ""))
                    buf = torch.empty(shape, dtype=dtype, device="cuda")
                    torch.distributed.broadcast(
                        buf, src=owner_global_rank, group=pp_group
                    )
                    yield (
                        adapted_name,
                        buf.to(self.dtype, non_blocking=True).contiguous(),
                    )

    def _get_local_param_tensor(self, name: str) -> tuple[torch.Tensor, nn.Module]:
        """Get a single param tensor by name from local model parts, gathering DTensor.

        Returns:
            Tuple of (tensor, owning_part) so callers can pass the correct
            model part to state_dict adapters.
        """
        for part in self.model_handle.parts:
            sd = part.state_dict()
            if name in sd:
                tensor = sd[name]
                gathered = (
                    tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                )
                return gathered, part
        raise KeyError(f"Param {name} not found in any local model part")

    def _model_state_dict_items(self):
        """Iterate state_dict items (unified for PP and non-PP via ModelHandle)."""
        return self.model_handle.state_dict_items()

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references.
                  Also disables top-k/top-p filtering since the reference policy's distribution
                  is different from the current policy, making filtered logprobs incompatible.
        On exit: Restores original references and re-flips cuda/cpu, restores sampling_params.
        """
        with torch.no_grad():
            # Save train model state_dict
            curr_state_dict = get_cpu_state_dict(
                self._model_state_dict_items(), pin_memory=True
            )

            # Swap reference model state_dict to self.model
            for k, v in self._model_state_dict_items():
                val = to_local_if_dtensor(v)
                val.copy_(self.reference_model_state_dict[k])

            # Temporarily disable top-k/top-p filtering for reference policy logprobs.
            saved_sampling_params = self.sampling_params
            if saved_sampling_params is not None:
                self.sampling_params = TrainingSamplingParams(
                    top_k=None,
                    top_p=1.0,
                    temperature=saved_sampling_params.temperature,
                )
            else:
                self.sampling_params = None

            yield

            # Restore sampling_params
            self.sampling_params = saved_sampling_params

            # Restore train model state_dict
            for k, v in self._model_state_dict_items():
                val = to_local_if_dtensor(v)
                val.copy_(curr_state_dict[k])

    def _add_noise_to_weights(self) -> None:
        """Add small Gaussian noise to the weights of the model. Note that this is used for testing purposes only."""
        noise_std = 0.01  # Standard deviation for the noise
        for p in self.model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)  # Add noise in-place
        torch.cuda.synchronize()

    def return_state_dict(self):
        return self.model_handle.state_dict()

    def return_model_config(self) -> dict[str, Any]:
        """Return the model configuration as a dictionary."""
        return self.model_handle.config

    @torch.no_grad()
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        For PP: each rank only has a subset of params. All-gather param info
        across PP ranks so every rank returns the full model's metadata.
        """
        local_info = {}
        for part in self.model_handle.parts:
            for name, tensor in part.state_dict().items():
                if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
                    continue
                full_tensor = (
                    tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                )
                adapted_fqn_tensors = _maybe_adapt_tensor_to_hf(part, name, full_tensor)
                for adapted_fqn, adapted_tensor in adapted_fqn_tensors:
                    local_info[adapted_fqn] = (adapted_tensor.shape, self.dtype)

        if self.pp_enabled:
            # All-gather param info across PP ranks so all workers return full metadata
            pp_group = self.pp_mesh.get_group()
            all_infos = [None] * self.pp_size
            torch.distributed.all_gather_object(all_infos, local_info, group=pp_group)
            state_dict_info = {}
            for info in all_infos:
                state_dict_info.update(info)
            return state_dict_info

        return local_info

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """Placeholder for FP8 Q/K/V scale calibration, not implemented for DTensorPolicyWorkerV2."""
        raise NotImplementedError(
            "calibrate_qkv_fp8_scales is not implemented for DTensorPolicyWorkerV2"
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self,
        buffer_size_bytes: int = 0,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        if kv_scales is not None:
            raise NotImplementedError(
                "FP8 kvcache is not currently supported for DTensor path, we will support it in the future."
            )

        self.maybe_init_zmq()
        # Manually move model to cuda for cpu offload case
        if self.cpu_offload:
            self.model = self.move_to_cuda(self.model)

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        stream_weights_via_ipc_zmq_impl(
            params_generator=self._all_params_generator(),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/stream_weights_via_http")
    def stream_weights_via_http(
        self,
        sglang_url_to_gpu_uuids: dict[str, list[str]],
    ) -> None:
        """Stream model weights to SGLang servers via HTTP API.

        Args:
            sglang_url_to_gpu_uuids: Dict mapping SGLang server URL to list of GPU UUIDs it uses
        """
        # Manually move model to cuda for cpu offload case
        if self.cpu_offload:
            self.model = self.move_to_cuda(self.model)

        from nemo_rl.models.policy.utils import stream_weights_via_http_impl

        # Get current GPU UUID
        current_device_uuid = self.report_device_id()

        def dtensor_params_generator():
            """Generator that yields (name, tensor) pairs, converting DTensors to local tensors."""
            state_dict_items = sorted(
                self.model.state_dict().items(), key=lambda x: x[0]
            )
            for name, tensor in state_dict_items:
                if isinstance(tensor, DTensor):
                    # Convert DTensor to full tensor for streaming
                    full_tensor = tensor.full_tensor()
                    # Convert to target dtype
                    yield (
                        name,
                        full_tensor.to(self.dtype, non_blocking=True).contiguous(),
                    )
                else:
                    # Convert to target dtype
                    yield name, tensor.to(self.dtype, non_blocking=True).contiguous()

        # Use the HTTP implementation
        stream_weights_via_http_impl(
            params_generator=dtensor_params_generator(),
            sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
            rank=self.rank,
            worker_name=str(self),
            current_device_uuid=current_device_uuid,
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        if kv_scales is not None:
            raise NotImplementedError(
                "FP8 kvcache is not currently supported for DTensor path, we will support it in the future."
            )

        # Manually move model to cuda for cpu offload case
        if self.cpu_offload:
            print(
                "[WARNING]: Unless you are lacking of memory, it is not recommended to enable cpu_offload when "
                "using non-colocated generation since it will have an extra onload and offload at refit stage."
            )
            self.model = self.move_to_cuda(self.model)

        # param_iterator will return (name, tensor), we only need tensor
        dtensor_post_iter_func = lambda x: x[1]

        packed_broadcast_producer(
            iterator=self._all_params_generator(),
            group=self.model_update_group,
            src=0,
            post_iter_func=dtensor_post_iter_func,
        )

        # Manually move model to cpu for cpu offload case
        # cpu offload needs model on CPU before model forward
        if self.cpu_offload:
            self.model = self.move_to_cpu(self.model)

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/prepare_for_lp_inference")
    def prepare_for_lp_inference(self) -> None:
        # onload model to cuda
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model_handle.move_buffers_to("cuda")
        self.model_handle.eval()

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if self.optimizers is not None and self.offload_optimizer_for_logprob:
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model_handle.move_buffers_to("cuda")
        self.model_handle.train()
        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            self.optimizers is not None
            and not self.cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer_to_device("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/offload_before_refit")
    def offload_before_refit(self) -> None:
        """Offload the optimizer to the CPU."""
        torch.randn(1).cuda()  # wake up torch allocator
        if self.optimizers is not None:
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/offload_after_refit")
    def offload_after_refit(self) -> None:
        """Offload as much as possible on the CPU."""
        self.model = self.move_to_cpu(self.model)
        self.model_handle.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_optimizer_to_device(self, device: str | torch.device) -> None:
        opts = self.optimizers
        for opt in opts:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to(device)

    def move_to_device(self, model: nn.Module, device: str | torch.device) -> nn.Module:
        self.model_handle.move_buffers_to(device)
        self.model_handle.to(device)
        return model

    def move_buffer_to_device(
        self, model: nn.Module, device: str | torch.device
    ) -> nn.Module:
        self.model_handle.move_buffers_to(device)
        return model

    def move_to_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def move_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[CheckpointingConfig] = None,
    ) -> None:
        """Save a checkpoint of the model.

        the optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        For PP, passes model parts list and optimizer/scheduler lists so the
        automodel Checkpointer can save per-stage state.
        """
        # For PP, pass model parts list; for non-PP, pass single model.
        # The automodel Checkpointer's ModelState/OptimizerState wrappers
        # handle both single nn.Module and lists transparently.
        model_for_ckpt = self.model_handle.parts if self.pp_enabled else self.model
        optimizer_for_ckpt = (
            self.optimizers
            if self.pp_enabled
            else (self.optimizers[0] if self.optimizers else None)
        )
        scheduler_for_ckpt = (
            self.schedulers
            if self.pp_enabled
            else (self.schedulers[0] if self.schedulers else None)
        )
        self.checkpoint_manager.save_checkpoint(
            model=model_for_ckpt,
            weights_path=weights_path,
            optimizer=optimizer_for_ckpt,
            optimizer_path=optimizer_path,
            scheduler=scheduler_for_ckpt,
            tokenizer=self.tokenizer if tokenizer_path else None,
            tokenizer_path=tokenizer_path,
            checkpointing_cfg=checkpointing_cfg,
            lora_enabled=self.lora_enabled,
            peft_config=self.peft_config,
        )

    def load_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
    ) -> None:
        """Load a checkpoint into the model using Automodel Checkpointer."""
        model_for_ckpt = self.model_handle.parts if self.pp_enabled else self.model
        optimizer_for_ckpt = (
            self.optimizers
            if self.pp_enabled
            else (self.optimizers[0] if self.optimizers else None)
        )
        scheduler_for_ckpt = (
            self.schedulers
            if self.pp_enabled
            else (self.schedulers[0] if self.schedulers else None)
        )
        self.checkpoint_manager.load_checkpoint(
            model=model_for_ckpt,
            weights_path=weights_path,
            optimizer=optimizer_for_ckpt,
            optimizer_path=optimizer_path,
            scheduler=scheduler_for_ckpt,
        )

    def _init_checkpoint_manager(
        self,
        config_updates: Optional[dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        """Initialize the AutomodelCheckpointManager for this worker.

        This creates the checkpoint manager bound to this worker's device meshes
        and initializes its underlying checkpointer.

        Args:
            config_updates: Dict of CheckpointingConfig fields to set during initialization.
            checkpoint_root: Optional root directory for checkpoints.
        """
        if self.checkpoint_manager is None:
            self.checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=self.dp_mesh,
                tp_mesh=self.tp_mesh,
                moe_mesh=self.moe_mesh,
                pp_mesh=getattr(self, "pp_mesh", None),
            )
            self.checkpoint_manager.init_checkpointer(
                config_updates=config_updates,
                checkpoint_root=checkpoint_root,
            )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")
)  # pragma: no cover
class DTensorPolicyWorkerV2(DTensorPolicyWorkerV2Impl):
    pass
