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

"""Training utilities for automodel (DTensor-based) policy workers.

This module provides post-processor classes and forward/backward functions
that follow the same pattern as nemo_rl/models/megatron/train.py.

Key differences from megatron approach:
- Post-processors compute results directly (no callable return pattern)
- forward_with_post_processing_fn calls post-processor directly
- automodel_forward_backward uses PyTorch autograd instead of Megatron's pipeline
"""

from typing import Any, Callable, Iterator, Optional, Tuple, Union

import torch
from nemo_automodel.components.distributed.tensor_utils import to_local_if_dtensor
from torch import nn
from torch.distributed.tensor import DTensor, Shard

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    get_logprobs_from_vocab_parallel_logits,
)
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch

# Union type for any post-processing function
PostProcessingFunction = Union[
    "LossPostProcessor",
    "LogprobsPostProcessor",
    "TopkLogitsPostProcessor",
    "ScorePostProcessor",
]


def model_forward(
    model: nn.Module,
    processed_inputs: ProcessedInputs,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
) -> torch.Tensor:
    """Perform a single forward pass through the model.

    Args:
        model: The model to run forward pass on
        processed_inputs: ProcessedInputs containing all tensors for forward pass
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether to pass flash_attn_kwargs to model

    Returns:
        torch.Tensor: Output tensor from the model (logits)
    """
    model_args = dict(
        input_ids=processed_inputs.input_ids,
        attention_mask=processed_inputs.attention_mask,
        position_ids=processed_inputs.position_ids,
        use_cache=False,
    )

    # Add flash attention kwargs if applicable
    if processed_inputs.has_flash_attention:
        model_args["flash_attn_kwargs"] = processed_inputs.flash_attn_kwargs

    # Add VLM kwargs if applicable
    if processed_inputs.is_multimodal:
        model_args.update(processed_inputs.vlm_kwargs)
        # flash_attn_kwargs is not supported for multimodal
        if "flash_attn_kwargs" in model_args:
            del model_args["flash_attn_kwargs"]

    # Reward models don't support flash_attn_kwargs
    if is_reward_model:
        if "flash_attn_kwargs" in model_args:
            del model_args["flash_attn_kwargs"]

    # Remove flash_attn_kwargs if not allowed
    if not allow_flash_attn_args and "flash_attn_kwargs" in model_args:
        del model_args["flash_attn_kwargs"]

    outputs = model(**model_args)
    return outputs


def extract_logits(
    model: nn.Module,
    outputs: Any,
) -> torch.Tensor:
    """Extract logits from model outputs.

    Args:
        model: The model (used for lm_head if needed)
        outputs: Model outputs (can be tensor, DTensor, or object with logits attribute)

    Returns:
        torch.Tensor: Logits tensor
    """
    if isinstance(outputs, (torch.Tensor, DTensor)):
        # Custom models can output logits directly
        return outputs
    elif not hasattr(outputs, "logits"):
        return model.lm_head(outputs.last_hidden_state)
    else:
        return outputs.logits


def apply_temperature_scaling(
    logits: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Logits tensor to scale
        cfg: Configuration dictionary containing generation settings

    Returns:
        torch.Tensor: Temperature-scaled logits
    """
    if "generation" in cfg and cfg["generation"] is not None:
        logits.div_(cfg["generation"]["temperature"])
    return logits


def redistribute_logits_for_cp(
    logits: torch.Tensor,
    device_mesh: Any,
    cp_mesh: Any,
    sequence_dim: int = 1,
) -> DTensor:
    """Redistribute logits for context parallel processing.

    Handles the case where logits may be TP-sharded DTensor or regular tensor,
    and converts them to CP+TP sharded DTensor.

    Args:
        logits: Logits tensor (may be DTensor or regular tensor)
        device_mesh: Full device mesh
        cp_mesh: Context parallel mesh
        sequence_dim: Dimension for sequence sharding

    Returns:
        DTensor sharded on both CP and TP dimensions
    """
    if isinstance(logits, DTensor):
        # Must be tp sharded
        assert (
            logits.device_mesh.ndim == 1
            and logits.device_mesh.mesh_dim_names[0] == "tp"
        ), "logits must be tp sharded"

        # CP is implicitly sharded on the seq dim, so we need to redistribute to the tp dim
        logits = DTensor.from_local(
            logits.to_local(),
            device_mesh=device_mesh[("cp", "tp")],
            placements=[Shard(sequence_dim), Shard(-1)],
        )
    else:
        logits = DTensor.from_local(
            logits,
            device_mesh=device_mesh[("cp", "tp")],
            placements=[Shard(sequence_dim), Shard(-1)],
        )
    return logits


def prepare_data_for_cp(
    mb: BatchedDataDict[Any],
    processed_inputs: ProcessedInputs,
    cp_mesh: Any,
    sequence_dim: int = 1,
) -> tuple[torch.Tensor, BatchedDataDict[Any]]:
    """Prepare data for context parallel processing.

    Converts seq_index to full tensor and wraps CP-sharded tensors in DTensor.

    Args:
        mb: Microbatch data dictionary
        processed_inputs: Processed inputs containing CP buffers
        cp_mesh: Context parallel mesh
        sequence_dim: Dimension for sequence sharding

    Returns:
        Tuple of (seq_index_dtensor, updated_mb)
    """
    seq_index_dtensor = (
        DTensor.from_local(
            processed_inputs.seq_index,
            device_mesh=cp_mesh,
            placements=[Shard(1)],
        )
        .full_tensor()
        .squeeze(0)
    )

    mb["seq_index"] = seq_index_dtensor

    for tensor_name in mb:
        current_tensor = mb[tensor_name]
        for buffer in processed_inputs.cp_buffers:
            if current_tensor is buffer:
                assert type(current_tensor) == torch.Tensor, (
                    f"tensor {tensor_name} is not a tensor"
                )
                mb[tensor_name] = DTensor.from_local(
                    current_tensor,
                    device_mesh=cp_mesh,
                    placements=[Shard(sequence_dim)],
                )
                break

    return seq_index_dtensor, mb


def forward_with_post_processing_fn(
    model: nn.Module,
    cfg: dict[str, Any],
    post_processing_fn: PostProcessingFunction,
    data_iterator: Optional[Iterator[ProcessedMicrobatch]] = None,
    processed_mb: Optional[ProcessedMicrobatch] = None,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sequence_dim: int = 1,
) -> Tuple[Any, dict[str, Any], ProcessedMicrobatch]:
    """Perform forward pass with pre-processed microbatch and apply post-processing.

    This function takes a pre-processed microbatch (with sequence packing already handled),
    runs the forward step through the model, and applies the post-processing function
    to compute the result.

    Unlike the megatron approach which returns a callable, this directly computes
    and returns the result since automodel uses PyTorch autograd.

    Args:
        model: The model to run forward pass on
        cfg: Configuration dictionary
        post_processing_fn: Post-processing function to apply to the logits
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed).
            Either data_iterator or processed_mb must be provided.
        processed_mb: Pre-fetched ProcessedMicrobatch to use directly instead of
            iterating. Useful when the caller needs to access cp_buffers before
            entering a context manager. Either data_iterator or processed_mb must be provided.
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether to pass flash_attn_kwargs to model
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sequence_dim: Sequence dimension

    Returns:
        tuple: (result, metrics, processed_microbatch)
            - result: Output from post-processing (loss, logprobs, topk, or scores)
            - metrics: Dictionary of metrics from post-processing
            - processed_microbatch: The ProcessedMicrobatch that was processed
    """
    # Get the pre-processed microbatch from iterator or use provided one
    if processed_mb is None:
        if data_iterator is None:
            raise ValueError("Either data_iterator or processed_mb must be provided")
        processed_mb = next(data_iterator)

    # Extract the processed components
    data_dict = processed_mb.data_dict
    processed_inputs = processed_mb.processed_inputs

    # Model forward pass
    outputs = model_forward(
        model,
        processed_inputs,
        is_reward_model=is_reward_model,
        allow_flash_attn_args=allow_flash_attn_args,
    )

    # Extract logits from model outputs
    logits = extract_logits(model, outputs)
    del outputs

    # Apply temperature scaling
    logits = apply_temperature_scaling(logits, cfg)

    # Apply the post-processing function directly based on type
    if isinstance(post_processing_fn, LossPostProcessor):
        result, metrics = post_processing_fn(
            logits=logits,
            mb=data_dict,
            processed_inputs=processed_inputs,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            sequence_dim=sequence_dim,
        )
    elif isinstance(
        post_processing_fn, (LogprobsPostProcessor, TopkLogitsPostProcessor)
    ):
        result = post_processing_fn(
            logits=logits,
            processed_inputs=processed_inputs,
            input_lengths=data_dict["input_lengths"],
            original_batch_size=processed_mb.original_batch_size,
            original_seq_len=processed_mb.original_seq_len,
            enable_seq_packing=post_processing_fn.enable_seq_packing,
            sequence_dim=sequence_dim,
        )
        if isinstance(post_processing_fn, LogprobsPostProcessor):
            metrics = {"logprobs": result}
        else:
            vals, idx = result
            metrics = {"topk_logits": vals, "topk_indices": idx}
    elif isinstance(post_processing_fn, ScorePostProcessor):
        result = post_processing_fn(logits=logits)
        metrics = {"scores": result}
    else:
        raise TypeError(
            f"Unknown post-processing function type: {type(post_processing_fn)}"
        )

    del logits
    return result, metrics, processed_mb


def automodel_forward_backward(
    model: nn.Module,
    cfg: dict[str, Any],
    data_iterator: Iterator[ProcessedMicrobatch],
    post_processing_fn: PostProcessingFunction,
    forward_only: bool = False,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sequence_dim: int = 1,
    dp_size: int = 1,
    cp_size: int = 1,
    num_global_batches: int = 1,
    train_context_fn: Optional[Callable[[ProcessedInputs], Any]] = None,
    num_valid_microbatches: Optional[int] = None,
    on_microbatch_start: Optional[Callable[[int], None]] = None,
) -> list[Tuple[Any, dict[str, Any]]]:
    """Execute forward and backward passes for automodel.

    This is the main training loop function that coordinates forward and backward
    passes across multiple microbatches using PyTorch autograd.

    Unlike megatron_forward_backward which uses Megatron's pipeline parallel
    framework, this uses standard PyTorch operations.

    Args:
        model: The model to train
        cfg: Configuration dictionary
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        num_microbatches: Number of microbatches to process
        post_processing_fn: Post-processing function to apply to the logits
        forward_only: If True, skip backward pass
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether to pass flash_attn_kwargs to model
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sequence_dim: Sequence dimension
        dp_size: Data parallel size
        cp_size: Context parallel size
        num_global_batches: Number of global batches (for metric scaling)
        train_context_fn: Optional callable that takes ProcessedInputs and returns
            a context manager for the forward/backward pass. If None, no context is used.
        num_valid_microbatches: Number of valid (non-dummy) microbatches. If provided,
            microbatches beyond this index are treated as dummy batches (loss *= 0).
            If None, all microbatches are considered valid.
        on_microbatch_start: Optional callback called at the start of each microbatch
            with the microbatch index. Useful for cache clearing, etc.

    Returns:
        List of (result, metrics) tuples from each microbatch
    """
    from contextlib import nullcontext

    results = []

    for mb_idx, processed_mb in enumerate(data_iterator):
        # Call optional callback at start of microbatch
        if on_microbatch_start is not None:
            on_microbatch_start(mb_idx)

        processed_inputs = processed_mb.processed_inputs

        # Create train context if factory provided, otherwise use nullcontext
        if train_context_fn is not None:
            ctx = train_context_fn(processed_inputs)
        else:
            ctx = nullcontext()

        with ctx:
            # Forward pass with post-processing
            result, metrics, _ = forward_with_post_processing_fn(
                model=model,
                cfg=cfg,
                post_processing_fn=post_processing_fn,
                processed_mb=processed_mb,
                is_reward_model=is_reward_model,
                allow_flash_attn_args=allow_flash_attn_args,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                sequence_dim=sequence_dim,
            )

            # Check if this is a dummy batch
            is_dummy = (
                num_valid_microbatches is not None and mb_idx >= num_valid_microbatches
            )

            # Scale metrics for aggregation (only for loss)
            if isinstance(post_processing_fn, LossPostProcessor):
                if not is_dummy:
                    for k in metrics.keys():
                        if k != "num_valid_samples":
                            metrics[k] /= num_global_batches
                else:
                    # Zero out loss for dummy batches
                    result = result * 0

                # Backward pass if training
                if not forward_only:
                    # When FSDP reduces gradients over DP dim, they're automatically averaged
                    # but we want to sum them so we cancel out the average here
                    loss = result * dp_size * cp_size
                    loss.backward()

        results.append((result, metrics))

    return results


class LossPostProcessor:
    """Post-processor for computing training loss from model outputs."""

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: dict[str, Any],
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        dp_size: int,
    ):
        """Initialize LossPostProcessor.

        Args:
            loss_fn: Loss function to compute loss
            cfg: Configuration dictionary
            device_mesh: Full device mesh
            cp_mesh: Context parallel mesh
            tp_mesh: Tensor parallel mesh
            cp_size: Context parallel size
            dp_size: Data parallel size
        """
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device_mesh = device_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh
        self.cp_size = cp_size
        self.dp_size = dp_size

    def __call__(
        self,
        logits: torch.Tensor,
        mb: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss from logits.

        Args:
            logits: Model output logits
            mb: Microbatch data
            processed_inputs: Processed inputs
            global_valid_seqs: Global valid sequence count
            global_valid_toks: Global valid token count
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (loss, metrics)
        """
        # Handle CP redistribution
        if self.cp_size > 1:
            _, mb = prepare_data_for_cp(
                mb, processed_inputs, self.cp_mesh, sequence_dim
            )
            logits = redistribute_logits_for_cp(
                logits, self.device_mesh, self.cp_mesh, sequence_dim
            )

        # Wrap loss function for sequence packing if needed
        if processed_inputs.has_flash_attention:
            loss_fn_ = SequencePackingLossWrapper(
                loss_fn=self.loss_fn,
                cu_seqlens_q=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
                cu_seqlens_q_padded=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
            )
        else:
            loss_fn_ = self.loss_fn

        loss, loss_metrics = loss_fn_(
            logits,
            mb,
            global_valid_seqs,
            global_valid_toks,
        )

        return loss, loss_metrics


class LogprobsPostProcessor:
    """Post-processor for computing log probabilities from model outputs."""

    def __init__(
        self,
        cfg: dict[str, Any],
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        enable_seq_packing: bool = False,
    ):
        """Initialize LogprobsPostProcessor.

        Args:
            cfg: Configuration dictionary
            device_mesh: Full device mesh
            cp_mesh: Context parallel mesh
            tp_mesh: Tensor parallel mesh
            cp_size: Context parallel size
            enable_seq_packing: Whether sequence packing is enabled
        """
        self.cfg = cfg
        self.device_mesh = device_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh
        self.cp_size = cp_size
        self.enable_seq_packing = enable_seq_packing
        self.logprob_chunk_size = cfg.get("logprob_chunk_size", None)

    def __call__(
        self,
        logits: torch.Tensor,
        processed_inputs: ProcessedInputs,
        input_lengths: torch.Tensor,
        original_batch_size: int,
        original_seq_len: int,
        enable_seq_packing: bool,
        sequence_dim: int = 1,
    ) -> torch.Tensor:
        """Compute token log probabilities from logits.

        Args:
            logits: Model output logits
            processed_inputs: Processed inputs
            input_lengths: Sequence lengths
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            enable_seq_packing: Whether sequence packing is enabled
            sequence_dim: Sequence dimension

        Returns:
            Token log probabilities tensor [batch_size, seq_length]
        """
        seq_len = processed_inputs.seq_len

        if self.cp_size > 1:
            seq_index_tensor = (
                DTensor.from_local(
                    processed_inputs.seq_index,
                    device_mesh=self.cp_mesh,
                    placements=[Shard(1)],
                )
                .full_tensor()
                .squeeze(0)
            )

            input_ids_dtensor = DTensor.from_local(
                processed_inputs.input_ids,
                device_mesh=self.cp_mesh,
                placements=[Shard(sequence_dim)],
            )

            logits = redistribute_logits_for_cp(
                logits, self.device_mesh, self.cp_mesh, sequence_dim
            )

            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                logits,
                input_ids_dtensor,
                seq_index_tensor,
                chunk_size=self.logprob_chunk_size,
            )

            assert token_logprobs.shape[1] == seq_len - 1
        else:
            if isinstance(logits, DTensor):
                token_logprobs = get_logprobs_from_vocab_parallel_logits(
                    logits,
                    processed_inputs.input_ids,
                    chunk_size=self.logprob_chunk_size,
                )
            else:
                token_logprobs = self._compute_local_logprobs(
                    logits, processed_inputs.input_ids
                )

        # Prepend 0 for first token to maintain sequence length
        token_logprobs = torch.cat(
            [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
        )

        # Handle sequence packing unpacking or mask application
        if enable_seq_packing:
            unpacked_logprobs = torch.zeros(
                (original_batch_size, original_seq_len),
                dtype=token_logprobs.dtype,
                device=token_logprobs.device,
            )
            cu_seqlens = processed_inputs.flash_attn_kwargs.cu_seqlens_q
            for i in range(original_batch_size):
                start = cu_seqlens[i].item() + 1
                end = cu_seqlens[i + 1].item()
                seq_len_actual = input_lengths[i].item()
                unpacked_logprobs[i, 1:seq_len_actual] = token_logprobs[0, start:end]
            token_logprobs = unpacked_logprobs
        else:
            # Apply mask to zero out padding tokens logprobs
            batch_size = processed_inputs.input_ids.shape[0]
            post_attention_mask = torch.zeros(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=token_logprobs.device,
            )
            for i, length in enumerate(input_lengths):
                post_attention_mask[i, :length] = 1
            token_logprobs = token_logprobs * post_attention_mask

        return token_logprobs

    def _compute_local_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logprobs locally without distributed processing.

        Args:
            logits: Model output logits
            input_ids: Input token IDs

        Returns:
            Token log probabilities
        """
        if self.logprob_chunk_size is not None:
            logits_seq_len = int(logits.shape[1])
            num_chunks = (
                logits_seq_len + self.logprob_chunk_size - 1
            ) // self.logprob_chunk_size
            chunked_log_probs = []
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * self.logprob_chunk_size
                chunk_end = min(
                    logits_seq_len,
                    (chunk_idx + 1) * self.logprob_chunk_size,
                )
                chunk_logits = logits[:, chunk_start:chunk_end, :].to(torch.float32)
                log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
                chunked_log_probs.append(log_probs)
            log_probs = torch.cat(chunked_log_probs, dim=1)
            del chunked_log_probs
        else:
            logits = logits.to(torch.float32)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract logprobs for each token in the sequence
        next_tokens = input_ids[:, 1:]
        log_probs = log_probs[:, :-1]
        token_logprobs = log_probs.gather(
            dim=-1, index=next_tokens.unsqueeze(-1)
        ).squeeze(-1)
        del log_probs

        return token_logprobs


class TopkLogitsPostProcessor:
    """Post-processor for computing top-k logits from model outputs."""

    def __init__(
        self,
        cfg: dict[str, Any],
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        k: int,
        enable_seq_packing: bool = False,
    ):
        """Initialize TopkLogitsPostProcessor.

        Args:
            cfg: Configuration dictionary
            device_mesh: Full device mesh
            cp_mesh: Context parallel mesh
            tp_mesh: Tensor parallel mesh
            cp_size: Context parallel size
            k: Number of top logits to return
            enable_seq_packing: Whether sequence packing is enabled
        """
        self.cfg = cfg
        self.device_mesh = device_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh
        self.cp_size = cp_size
        self.k = k
        self.enable_seq_packing = enable_seq_packing

    def __call__(
        self,
        logits: torch.Tensor,
        processed_inputs: ProcessedInputs,
        input_lengths: torch.Tensor,
        original_batch_size: int,
        original_seq_len: int,
        enable_seq_packing: bool,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k logits and indices from model outputs.

        Args:
            logits: Model output logits
            processed_inputs: Processed inputs
            input_lengths: Sequence lengths
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            enable_seq_packing: Whether sequence packing is enabled
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (top-k values, top-k indices) tensors
        """
        if self.cp_size > 1:
            logits = redistribute_logits_for_cp(
                logits, self.device_mesh, self.cp_mesh, sequence_dim
            )

            # Deal with TP first
            local_logits = logits.to_local()  # [B, S_cp, V_tp]

            tp_group = self.tp_mesh.get_group()
            tp_rank = torch.distributed.get_rank(tp_group)
            V_local = int(local_logits.shape[-1])
            vocab_start_index = tp_rank * V_local
            vocab_end_index = (tp_rank + 1) * V_local

            vals, idx = distributed_vocab_topk(
                local_logits,
                k=self.k,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
            )
            # [B, S_cp, k]

            cp_group = self.cp_mesh.get_group()

            vals = allgather_cp_sharded_tensor(vals, cp_group, seq_dim=sequence_dim)
            idx = allgather_cp_sharded_tensor(idx, cp_group, seq_dim=sequence_dim)
            # [B, S, k]
        else:
            # Compute top-k over full sequence length
            if isinstance(logits, DTensor):
                local_logits = logits.to_local()  # [B, S, V_local]
                tp_group = self.tp_mesh.get_group()
                tp_rank = torch.distributed.get_rank(tp_group)
                V_local = int(local_logits.shape[-1])
                vocab_start_index = tp_rank * V_local
                vocab_end_index = (tp_rank + 1) * V_local

                vals, idx = distributed_vocab_topk(
                    local_logits,
                    k=self.k,
                    tp_group=tp_group,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_end_index,
                )
            else:
                full_logits = logits.to(torch.float32)
                vals, idx = torch.topk(full_logits, k=self.k, dim=-1)

        # Handle sequence packing unpacking
        if enable_seq_packing:
            unpacked_vals = torch.zeros(
                (original_batch_size, original_seq_len, self.k),
                dtype=vals.dtype,
                device=vals.device,
            )
            unpacked_idx = torch.zeros(
                (original_batch_size, original_seq_len, self.k),
                dtype=idx.dtype,
                device=idx.device,
            )

            cu_seqlens = processed_inputs.flash_attn_kwargs.cu_seqlens_q

            for i in range(original_batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_len_actual = input_lengths[i].item()

                unpacked_vals[i, :seq_len_actual, :] = vals[0, start:end, :]
                unpacked_idx[i, :seq_len_actual, :] = idx[0, start:end, :]

            vals = unpacked_vals
            idx = unpacked_idx

        return vals, idx


class ScorePostProcessor:
    """Post-processor for computing reward model scores from model outputs."""

    def __init__(
        self,
        cfg: dict[str, Any],
    ):
        """Initialize ScorePostProcessor.

        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Extract scores from reward model outputs.

        Args:
            logits: Model output logits

        Returns:
            Scores tensor
        """
        logits = logits.to(torch.float32)
        rm_scores = to_local_if_dtensor(logits)
        rm_scores = rm_scores.squeeze(-1)

        return rm_scores
