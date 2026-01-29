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
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generator, Optional

import ray
import torch
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
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    check_sequence_dim,
    get_microbatch_iterator,
    process_global_batch,
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
    automodel_forward_backward,
    forward_with_post_processing_fn,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ScoreOutputSpec,
)
from nemo_rl.models.policy.utils import (
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import (
    apply_torch_aten_alias_tensor_patch,
    apply_transformer_engine_patch,
)
from nemo_rl.utils.automodel_checkpoint import AutomodelCheckpointManager
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

    Yields:
        Tuples of (fully_qualified_name, tensor) where tensors are converted to target dtype and made contiguous.
    """
    for name, tensor in model.state_dict().items():
        full_tensor = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        adapted_fqn_tensors = _maybe_adapt_tensor_to_hf(model, name, full_tensor)
        for adapted_fqn, adapted_tensor in adapted_fqn_tensors:
            # Convert to target dtype
            yield (
                adapted_fqn,
                adapted_tensor.to(target_dtype, non_blocking=True).contiguous(),
            )


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
        if cp_size > 1:
            # Create context parallel context
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


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")
)  # pragma: no cover
class DTensorPolicyWorkerV2(AbstractPolicyWorker, ColocatablePolicyInterface):
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
        tokenizer: AutoTokenizer,
        processor: Optional[AutoProcessor] = None,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        """Initialize the DTensorPolicyWorkerV2."""
        # Apply TE patch until TE is upgraded to 2.10.0
        apply_transformer_engine_patch()
        # Apply patch to work around 'NotImplementedError: Operator aten.alias.default does not have a sharding strategy registered'
        apply_torch_aten_alias_tensor_patch()

        # Store configuration and tokenizer/processor
        self.cfg = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_vlm = processor is not None
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

        # Set up distributed environment (returns FSDP2Manager)
        distributed_manager = setup_distributed(
            config=config,
            runtime_config=runtime_config,
        )
        # Set instance attributes from distributed manager (tuple unpacking for mesh attributes)
        self.rank = torch.distributed.get_rank()
        self.device_mesh = distributed_manager.device_mesh
        self.dp_cp_mesh = self.device_mesh["dp_cp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_mesh = self.device_mesh["tp"]
        self.cp_mesh = self.device_mesh["cp"]
        self.moe_mesh = distributed_manager.moe_mesh
        self.dp_size = distributed_manager.dp_size
        self.tp_size = distributed_manager.tp_size
        self.cp_size = distributed_manager.cp_size

        # Initialize checkpoint manager now that distributed is set up
        self._init_checkpoint_manager(
            config_updates={
                "model_repo_id": config["model_name"],
                "dequantize_base_checkpoint": config.get(
                    "dequantize_base_checkpoint", False
                ),
                "is_peft": self.lora_enabled,
            },
        )

        # Set up model and optimizer
        model_and_optimizer_state = setup_model_and_optimizer(
            config=config,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            distributed_manager=distributed_manager,
            checkpoint_manager=self.checkpoint_manager,
            is_vlm=self.is_vlm,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )

        # Set instance attributes from model and optimizer state (tuple unpacking)
        (
            self.model,
            self.model_state_dict_keys,
            self.optimizer,
            self.scheduler,
            self.is_hf_model,
            self.is_moe_model,
            self._is_reward_model,  # Note: using underscore prefix for internal naming
            self.model_class,
            self.model_config,
            self.peft_config,
            self.autocast_enabled,
        ) = model_and_optimizer_state

        # Initialize reference model if requested
        self.reference_model_state_dict = None
        if init_reference_model:
            self.reference_model_state_dict = setup_reference_model_state(self.model)

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
        sequence_dim, seq_dim_size = check_sequence_dim(data)

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=loss_fn,
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
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
                f"Emptying cache every {empty_cache_steps} microbatches, doing so unnnecessarily would incur a large performance overhead."
            )

        def on_microbatch_start(mb_idx):
            if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                torch.cuda.empty_cache()

        with ctx:
            # Get data from batch and move to device
            data.to("cuda")

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

                self.optimizer.zero_grad()

                # Get microbatch iterator based on batching strategy
                processed_iterator, iterator_len = get_microbatch_iterator(
                    batch,
                    self.cfg,
                    mbs,
                    self.dp_mesh,
                    tokenizer=self.tokenizer,
                    cp_size=self.cp_size,
                )

                # Use automodel_forward_backward for the training loop
                mb_results = automodel_forward_backward(
                    model=self.model,
                    cfg=self.cfg,
                    data_iterator=processed_iterator,
                    post_processing_fn=loss_post_processor,
                    forward_only=eval_mode,
                    is_reward_model=self._is_reward_model,
                    allow_flash_attn_args=self.allow_flash_attn_args,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                    sequence_dim=sequence_dim,
                    dp_size=self.dp_size,
                    cp_size=self.cp_size,
                    num_global_batches=num_global_batches,
                    train_context_fn=train_context_fn,
                    num_valid_microbatches=iterator_len,
                    on_microbatch_start=on_microbatch_start
                    if empty_cache_steps
                    else None,
                )

                # Extract losses and metrics from results
                mb_losses = []
                for mb_idx, (loss, loss_metrics) in enumerate(mb_results):
                    # Only process valid (non-dummy) batches for metrics
                    if mb_idx < iterator_len:
                        num_valid_samples = loss_metrics.get("num_valid_samples", 0)
                        loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
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
                    self.optimizer.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # release gradient memory before rollouts
            self.optimizer.zero_grad()
            # increment scheduler after all batches in rollout are processed
            if not eval_mode:
                self.scheduler.step()
            # dynamic batch and sequence dims causes alot of fragmentation, so clear
            # the memory allocator before moving on
            torch.cuda.empty_cache()

            # Compute global loss across all ranks
            with torch.no_grad():
                global_loss = torch.tensor(losses, device="cuda")
                torch.distributed.all_reduce(
                    global_loss, group=self.dp_mesh.get_group()
                )
            # Aggregate metrics across all microbatches
            mb_metrics = defaultdict(list)
            for m in all_mb_metrics:
                for k, v in m.items():
                    mb_metrics[k].append(v)

            metrics = {
                "global_loss": global_loss.cpu(),
                "grad_norm": grad_norm,
                "rank": torch.distributed.get_rank(),
                "gpu_name": torch.cuda.get_device_name(),
                "model_dtype": self.dtype,
                "all_mb_metrics": dict(mb_metrics),
            }

            return metrics

    # TODO @Rayen Tian: Related Issue: Refactor shared logic between score() and get_logprobs() (https://github.com/NVIDIA-NeMo/RL/issues/1094)
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

        # Validate sequence dimension
        sequence_dim, seq_dim_size = check_sequence_dim(data)

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
                    token_logprobs, metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        cfg=self.cfg,
                        post_processing_fn=logprobs_post_processor,
                        processed_mb=processed_mb,
                        is_reward_model=False,
                        allow_flash_attn_args=self.allow_flash_attn_args,
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

    # TODO @Rayen Tian: Related Issue: Refactor shared logic between score() and get_logprobs() (https://github.com/NVIDIA-NeMo/RL/issues/1094)
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/score")
    def score(self, data: BatchedDataDict) -> BatchedDataDict[ScoreOutputSpec]:
        global_batch_size = min(self.cfg["batch_size"], data.size)

        # Validate sequence dimension
        sequence_dim, seq_dim_size = check_sequence_dim(data)

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
                    rm_scores, metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        cfg=self.cfg,
                        post_processing_fn=score_post_processor,
                        processed_mb=processed_mb,
                        is_reward_model=True,
                        allow_flash_attn_args=False,
                        sequence_dim=sequence_dim,
                    )

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
            processed_iterator, iterator_len = get_microbatch_iterator(
                data,
                self.cfg,
                topk_batch_size,
                self.dp_mesh,
                tokenizer=self.tokenizer,
                cp_size=self.cp_size,
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
                    (vals, idx), metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        cfg=self.cfg,
                        post_processing_fn=topk_post_processor,
                        processed_mb=processed_mb,
                        is_reward_model=False,
                        allow_flash_attn_args=self.allow_flash_attn_args,
                        sequence_dim=sequence_dim,
                    )

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

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        with torch.no_grad():
            try:
                # Save train model state_dict
                curr_state_dict = get_cpu_state_dict(
                    self.model.state_dict().items(), pin_memory=True
                )

                # Swap reference model state_dict to self.model
                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    val.copy_(self.reference_model_state_dict[k])

                # - self.model is the original reference_model, now on CUDA
                # - curr_state_dict is the train model, now on CPU
                yield

            finally:
                # Restore train model state_dict
                for k, v in self.model.state_dict().items():
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
        return self.model.state_dict()

    def return_model_config(self) -> dict[str, Any]:
        """Return the model configuration as a dictionary.

        Returns:
            dict: Model configuration dictionary
        """
        return self.model.config

    @torch.no_grad()
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        state_dict_info = {}
        for name, tensor in self.model.state_dict().items():
            full_tensor = (
                tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            )
            # all tensor will be casted to self.dtype in stream_weights_via_ipc_zmq/broadcast_weights_for_collective
            adapted_fqn_tensors = _maybe_adapt_tensor_to_hf(
                self.model, name, full_tensor
            )
            for adapted_fqn, adapted_tensor in adapted_fqn_tensors:
                state_dict_info[adapted_fqn] = (adapted_tensor.shape, self.dtype)

        return state_dict_info

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

        # Use the shared implementation
        stream_weights_via_ipc_zmq_impl(
            params_generator=dtensor_params_generator(self.model, self.dtype),
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
            iterator=dtensor_params_generator(self.model, self.dtype),
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
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.eval()

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if self.optimizer is not None and self.offload_optimizer_for_logprob:
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        # onload models and optimizer state to cuda
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            # when cpu offload is enabled, the buffers do not get moved
            # to cuda automatically, so we need to do that manually
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            self.optimizer is not None
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
        if self.optimizer is not None:
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/offload_after_refit")
    def offload_after_refit(self) -> None:
        """Offload as much as possible on the CPU."""
        self.model = self.move_to_cpu(self.model)
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_optimizer_to_device(self, device: str | torch.device) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, (DTensor, torch.Tensor)):
                    state[k] = v.to(device)

    def move_to_device(self, model: nn.Module, device: str | torch.device) -> nn.Module:
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(
        self, model: nn.Module, device: str | torch.device
    ) -> nn.Module:
        # FSDP modules do not move buffers to the device automatically
        for v in model.buffers():
            torch.utils.swap_tensors(v, v.to(device))

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
        """
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer if tokenizer_path is None else None,
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
        self.checkpoint_manager.load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
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
                model_state_dict_keys=getattr(self, "model_state_dict_keys", None),
                moe_mesh=self.moe_mesh,
            )
            self.checkpoint_manager.init_checkpointer(
                config_updates=config_updates,
                checkpoint_root=checkpoint_root,
            )
