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

"""Training utilities for automodel (DTensor-based) policy workers.

This module provides post-processor classes and forward/backward functions
that follow the same pattern as nemo_rl/models/megatron/train.py.

Key differences from megatron approach:
- Post-processors compute results directly (no callable return pattern)
- forward_with_post_processing_fn calls post-processor directly
- automodel_forward_backward uses PyTorch autograd instead of Megatron's pipeline
"""

import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import torch
from nemo_automodel.components.distributed.tensor_utils import to_local_if_dtensor
from torch import nn
from torch.distributed.tensor import DTensor, Shard
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    apply_top_k_top_p,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss import SequencePackingLossWrapper, prepare_loss_input
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    _compute_distributed_log_softmax,
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    get_logprobs_from_vocab_parallel_logits,
)
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import (
    get_handle_from_tensor,
    rebuild_cuda_tensor_from_ipc,
)

# Union type for any post-processing function
PostProcessingFunction = Union[
    "LossPostProcessor",
    "XTokenStudentIPCLossPostProcessor",
    "XTokenTeacherIPCExportPostProcessor",
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

    is_gemma3 = isinstance(model, Gemma3ForCausalLM) or isinstance(
        model, Gemma3ForConditionalGeneration
    )
    if is_gemma3 and "token_type_ids" not in model_args:
        model_args["token_type_ids"] = torch.zeros_like(processed_inputs.input_ids)

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
    logits: torch.Tensor, sampling_params: Optional[TrainingSamplingParams]
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Logits tensor to scale
        sampling_params: Sampling parameters

    Returns:
        torch.Tensor: Temperature-scaled logits
    """
    if sampling_params is not None and sampling_params.temperature != 1.0:
        logits.div_(sampling_params.temperature)
    return logits


def apply_top_k_top_p_filtering_for_local_logits(
    logits: torch.Tensor, sampling_params: Optional[TrainingSamplingParams]
) -> torch.Tensor:
    """Apply top-k and top-p filtering to the non-distributed logits.

    Args:
        logits: Logits tensor to filter
        sampling_params: Sampling parameters

    Returns:
        torch.Tensor: Filtered logits
    """
    if need_top_k_or_top_p_filtering(sampling_params):
        logits, _ = apply_top_k_top_p(
            logits,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )
    return logits


def redistribute_logits_for_cp(
    logits: torch.Tensor,
    device_mesh: Any,
    cp_mesh: Any,  # noqa: ARG001
    sequence_dim: int = 1,
) -> DTensor:
    """Redistribute logits for context parallel processing.

    Handles the case where logits may be TP-sharded DTensor or regular tensor,
    and converts them to CP+TP sharded DTensor.

    Args:
        logits: Logits tensor (may be DTensor or regular tensor)
        device_mesh: Full device mesh
        cp_mesh: Context parallel mesh (kept for signature compatibility)
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
    post_processing_fn: PostProcessingFunction,
    processed_mb: ProcessedMicrobatch,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
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
        post_processing_fn: Post-processing function to apply to the logits
        processed_mb: Pre-fetched ProcessedMicrobatch containing data and processed inputs
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether to pass flash_attn_kwargs to model
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
        sequence_dim: Sequence dimension

    Returns:
        tuple: (result, metrics, processed_microbatch)
            - result: Output from post-processing (loss, logprobs, topk, or scores)
            - metrics: Dictionary of metrics from post-processing
            - processed_microbatch: The ProcessedMicrobatch that was processed
    """
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

    # Apply temperature scaling only for sampling-oriented post-processors
    # Score computations should use unscaled logits
    if isinstance(
        post_processing_fn,
        (LossPostProcessor, LogprobsPostProcessor, TopkLogitsPostProcessor),
    ):
        # Temperature scaling is element-wise, directly applying it here.
        # Other sampling parameters like top-k and top-p need the logits from whole vocabulary,
        # so applying them when gathering logits from vocab parallel (called in LossPostProcessor and LogprobsPostProcessor).
        logits = apply_temperature_scaling(logits, sampling_params)

    # Apply the post-processing function directly based on type
    if isinstance(post_processing_fn, LossPostProcessor):
        result, metrics = post_processing_fn(
            logits=logits,
            data_dict=data_dict,
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
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=processed_mb.original_batch_size,
            original_seq_len=processed_mb.original_seq_len,
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
    data_iterator: Iterator[ProcessedMicrobatch],
    post_processing_fn: PostProcessingFunction,
    forward_only: bool = False,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
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
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        num_microbatches: Number of microbatches to process
        post_processing_fn: Post-processing function to apply to the logits
        forward_only: If True, skip backward pass
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether to pass flash_attn_kwargs to model
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
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
                post_processing_fn=post_processing_fn,
                processed_mb=processed_mb,
                is_reward_model=is_reward_model,
                allow_flash_attn_args=allow_flash_attn_args,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                sampling_params=sampling_params,
                sequence_dim=sequence_dim,
            )

            # Check if this is a dummy batch
            is_dummy = (
                num_valid_microbatches is not None and mb_idx >= num_valid_microbatches
            )

            # Scale metrics for aggregation (only for loss)
            if isinstance(post_processing_fn, LossPostProcessor):
                # skip the update for dummy batches
                if not is_dummy:
                    ## scale by the number of global batches so we get the correct
                    ## value when summing metrics across all microbatches
                    for k in metrics.keys():
                        if "_min" in k or "_max" in k:
                            continue

                        metrics[k] /= num_global_batches
                else:
                    # Zero out loss for dummy batches
                    result = result * 0

                # Backward pass if training
                if not forward_only:
                    ## NOTE: invalid samples should be multiplied
                    ## by zero in the loss function to prevent them
                    ## from affecting the gradient calculation

                    # when FSDP reduces the gradients over the DP dim, they're automatically averaged
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
        cfg: PolicyConfig,
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        dp_size: int,
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
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
            enable_seq_packing: Whether sequence packing is enabled
            sampling_params: Sampling parameters
        """
        self.loss_fn: LossFunction = loss_fn
        self.cfg: PolicyConfig = cfg
        self.device_mesh = device_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.enable_seq_packing = enable_seq_packing
        self.sampling_params = sampling_params

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss from logits.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            global_valid_seqs: Global valid sequence count
            global_valid_toks: Global valid token count
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (loss, metrics)
        """
        # Handle CP redistribution
        if self.cp_size > 1:
            _, data_dict = prepare_data_for_cp(
                data_dict, processed_inputs, self.cp_mesh, sequence_dim
            )
            logits = redistribute_logits_for_cp(
                logits, self.device_mesh, self.cp_mesh, sequence_dim
            )

        # Wrap prepare_loss_input with sampling_params
        prepare_loss_input_wrapped = partial(
            prepare_loss_input, sampling_params=self.sampling_params
        )
        # Wrap loss function for sequence packing if needed
        if self.enable_seq_packing:
            loss_fn = SequencePackingLossWrapper(
                loss_fn=self.loss_fn,
                prepare_fn=prepare_loss_input_wrapped,
                cu_seqlens_q=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
                cu_seqlens_q_padded=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
            )
            loss, loss_metrics = loss_fn(
                logits,
                data_dict,
                global_valid_seqs,
                global_valid_toks,
            )
        else:
            loss_input, data_dict = prepare_loss_input_wrapped(
                logits, data_dict, self.loss_fn
            )
            loss, loss_metrics = self.loss_fn(
                data=data_dict,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                **loss_input,
            )

        return loss, loss_metrics


class LogprobsPostProcessor:
    """Post-processor for computing log probabilities from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ):
        """Initialize LogprobsPostProcessor.

        Args:
            cfg: Configuration dictionary
            device_mesh: Full device mesh
            cp_mesh: Context parallel mesh
            tp_mesh: Tensor parallel mesh
            cp_size: Context parallel size
            enable_seq_packing: Whether sequence packing is enabled
            sampling_params: Sampling parameters
        """
        self.cfg = cfg
        self.device_mesh = device_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh
        self.cp_size = cp_size
        self.enable_seq_packing = enable_seq_packing
        self.sampling_params = sampling_params
        self.logprob_chunk_size = cfg.get("logprob_chunk_size", None)

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        original_batch_size: int,
        original_seq_len: int,
        sequence_dim: int = 1,
    ) -> torch.Tensor:
        """Compute token log probabilities from logits.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            sequence_dim: Sequence dimension

        Returns:
            Token log probabilities tensor [batch_size, seq_length]
        """
        seq_len = processed_inputs.seq_len
        input_lengths = data_dict["input_lengths"]

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
                sampling_params=self.sampling_params,  # top-k and top-p filtering
            )

            assert token_logprobs.shape[1] == seq_len - 1
        else:
            if isinstance(logits, DTensor):
                # DTensor path with TP sharding
                token_logprobs = get_logprobs_from_vocab_parallel_logits(
                    logits,
                    processed_inputs.input_ids,
                    chunk_size=self.logprob_chunk_size,
                    sampling_params=self.sampling_params,  # top-k and top-p filtering
                )
            else:
                # Non-DTensor path (no TP sharding)
                token_logprobs = self._compute_local_logprobs(
                    logits, processed_inputs.input_ids
                )

        # Prepend 0 for first token to maintain sequence length
        token_logprobs = torch.cat(
            [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
        )

        # Handle sequence packing unpacking or mask application
        if self.enable_seq_packing:
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
                # For right-padded sequence, set 1s at the beginning of the sequence
                post_attention_mask[i, :length] = 1
            token_logprobs = token_logprobs * post_attention_mask

        # handle top-k/top-p filtering for logprobs, only used for ClippedPGLossFn now
        if need_top_k_or_top_p_filtering(self.sampling_params):
            mask = data_dict["token_mask"] * data_dict["sample_mask"].unsqueeze(-1)
            token_logprobs = mask_out_neg_inf_logprobs(
                token_logprobs, mask, "prev_logprobs"
            )

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
                chunk_logits = apply_top_k_top_p_filtering_for_local_logits(
                    chunk_logits, self.sampling_params
                )
                log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
                chunked_log_probs.append(log_probs)
            log_probs = torch.cat(chunked_log_probs, dim=1)
            del chunked_log_probs
        else:
            logits = logits.to(torch.float32)
            logits = apply_top_k_top_p_filtering_for_local_logits(
                logits, self.sampling_params
            )
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract logprobs for each token in the sequence by gathering the logprob
        # corresponding to the next token at each position
        # Input shapes:
        #   log_probs: [batch_size, sequence_length, vocab_size] - logits for each position
        #   token_ids: [batch_size, sequence_length] - actual tokens
        # Output shape: [batch_size, sequence_length] - logprob of each token given previous
        # We get logprob of token[t+1] from logits[t], prepending 0 to maintain sequence length
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
        cfg: PolicyConfig,
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
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        original_batch_size: int,
        original_seq_len: int,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k logits and indices from model outputs.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (top-k values, top-k indices) tensors
        """
        input_lengths = data_dict["input_lengths"]

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
        if self.enable_seq_packing:
            # Unpack top-k results from packed format back to original batch format
            # vals: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]
            # idx: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]
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

                # Extract the corresponding portion from packed results
                # Note: vals and idx are [1, packed_seq_len, k] due to packing
                unpacked_vals[i, :seq_len_actual, :] = vals[0, start:end, :]
                unpacked_idx[i, :seq_len_actual, :] = idx[0, start:end, :]

            vals = unpacked_vals
            idx = unpacked_idx

        return vals, idx


class ScorePostProcessor:
    """Post-processor for computing reward model scores from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
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


def aggregate_training_statistics(
    losses: list[float],
    all_mb_metrics: list[dict[str, Any]],
    grad_norm: Optional[torch.Tensor],
    dp_group: Any,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Aggregate training statistics across microbatches and ranks.

    Args:
        losses: List of loss values from each microbatch
        all_mb_metrics: List of metrics dictionaries from each microbatch
        grad_norm: Gradient norm tensor (or None if eval mode)
        dp_group: Data parallel process group for all-reduce
        dtype: Model dtype for metrics

    Returns:
        Dictionary containing aggregated metrics including global_loss, grad_norm, etc.
    """
    # Compute global loss across all ranks
    with torch.no_grad():
        global_loss = torch.tensor(losses, device="cuda")
        torch.distributed.all_reduce(global_loss, group=dp_group)

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
        "model_dtype": dtype,
        "all_mb_metrics": dict(mb_metrics),
    }

    return metrics


class XTokenStudentIPCLossPostProcessor(LossPostProcessor):
    """Loss post-processor that injects teacher logits via CUDA IPC handles.

    Consumes records emitted by :class:`XTokenTeacherIPCExportPostProcessor`
    (schema_version=1). The current microbatch handle is matched against the
    worker's own ``(tp_rank, cp_rank)`` coordinates; payload is rebuilt from
    ``handle["payload_ipc"]`` rather than the deprecated ``handle[rank]`` key.
    """

    def __init__(
        self,
        *args: Any,
        teacher_result: Optional[dict[str, Any]] = None,
        teacher_tp_group_results: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._teacher_result = teacher_result
        # List of per-tp-rank teacher entries (ordered by tp_rank) for the
        # current worker's TP group. Used only on the cross-tokenizer full-
        # logits path to reconstruct the complete teacher vocab by concatenating
        # each TP rank's IPC buffer along the vocab dim.
        self._teacher_tp_group_results = teacher_tp_group_results
        self._microbatch_idx = 0
        # Cache parallel coordinates for fast-path handle validation.
        self.world_rank = torch.distributed.get_rank()
        self.tp_group = self.tp_mesh.get_group()
        self.tp_rank = torch.distributed.get_rank(self.tp_group)
        self.tp_size = torch.distributed.get_world_size(self.tp_group)
        if self.cp_size > 1 and self.cp_mesh is not None:
            self.cp_group = self.cp_mesh.get_group()
            self.cp_rank = torch.distributed.get_rank(self.cp_group)
        else:
            self.cp_group = None
            self.cp_rank = 0

    def set_microbatch_index(self, mb_idx: int) -> None:
        self._microbatch_idx = mb_idx

    def _get_current_microbatch_teacher_handle(self) -> Optional[dict[str, Any]]:
        """Resolve the teacher handle for the current microbatch and validate.

        Returns ``None`` when IPC teacher export is unavailable (e.g. no
        ``_teacher_result`` was provided or the current microbatch index is
        past the end of the exported handle list). Raises on structural
        mismatches between teacher and student shard coordinates.
        """
        if self._teacher_result is None:
            return None
        handles = self._teacher_result.get("microbatch_handles")
        if not handles:
            return None
        if self._microbatch_idx >= len(handles):
            return None
        handle = handles[self._microbatch_idx]

        # Strict shard-to-shard matching: minimal-change plan assumes teacher
        # and student worker topologies are aligned.
        assert handle.get("tp_rank") == self.tp_rank, (
            f"teacher handle tp_rank={handle.get('tp_rank')} != student "
            f"tp_rank={self.tp_rank} (mb_idx={self._microbatch_idx})"
        )
        assert handle.get("cp_rank") == self.cp_rank, (
            f"teacher handle cp_rank={handle.get('cp_rank')} != student "
            f"cp_rank={self.cp_rank} (mb_idx={self._microbatch_idx})"
        )
        assert handle.get("tp_size") == self.tp_size, (
            f"teacher handle tp_size={handle.get('tp_size')} != student "
            f"tp_size={self.tp_size}"
        )
        assert handle.get("cp_size") == self.cp_size, (
            f"teacher handle cp_size={handle.get('cp_size')} != student "
            f"cp_size={self.cp_size}"
        )
        handle_world_rank = handle.get("world_rank")
        if handle_world_rank != self.world_rank:
            # Soft warning: worker-layer selection should already have matched
            # by world_rank, but guard against silent misrouting.
            warnings.warn(
                "XToken teacher handle world_rank="
                f"{handle_world_rank} does not match student world_rank="
                f"{self.world_rank}",
                stacklevel=2,
            )
        return handle

    def _get_tp_group_handles_for_current_microbatch(
        self,
    ) -> Optional[list[dict[str, Any]]]:
        """Return one schema-v1 handle per TP rank for the current microbatch.

        Ordered by ``tp_rank`` (``0 .. tp_size-1``). Used by the full-logits
        path to reconstruct the complete teacher vocab by concatenating each
        TP rank's IPC payload along the vocab dim. Returns ``None`` when:

            - ``tp_size == 1`` (nothing to gather),
            - ``teacher_tp_group_results`` was not provided, or
            - the current microbatch index is past the end of any sibling's
              handle list (teacher produced fewer microbatches than expected).
        """
        if self.tp_size == 1 or self._teacher_tp_group_results is None:
            return None
        handles_per_tp: list[dict[str, Any]] = []
        for tp_r, entry in enumerate(self._teacher_tp_group_results):
            handles = (
                entry.get("microbatch_handles") if isinstance(entry, dict) else None
            )
            if not handles or self._microbatch_idx >= len(handles):
                return None
            h = handles[self._microbatch_idx]
            assert h.get("tp_rank") == tp_r, (
                f"TP-group entry at index {tp_r} has handle tp_rank={h.get('tp_rank')}"
            )
            assert h.get("cp_rank") == self.cp_rank, (
                f"TP sibling at tp_rank={tp_r} has cp_rank="
                f"{h.get('cp_rank')} != student cp_rank={self.cp_rank}"
            )
            assert h.get("is_topk") is False, (
                f"TP-group gather is only valid on the full-logits path; "
                f"handle at tp_rank={tp_r} reports is_topk=True"
            )
            handles_per_tp.append(h)
        return handles_per_tp

    @staticmethod
    def _rebuild_own_rank_payload(
        handle: dict[str, Any], current_device_id: int
    ) -> torch.Tensor:
        """Open the own-rank IPC view and return a locally-owned tensor.

        Slice is driven by ``handle["actual_shape"]`` so pre-allocated
        buffers that are larger than the current microbatch don't leak
        garbage dimensions into the student.
        """
        payload_ipc = handle.get("payload_ipc")
        assert payload_ipc is not None, (
            "teacher handle is missing payload_ipc (schema_version="
            f"{handle.get('schema_version')})"
        )
        aB, aS, aK = handle["actual_shape"]
        tensor = rebuild_cuda_tensor_from_ipc(payload_ipc, current_device_id).detach()
        return tensor[:aB, :aS, :aK].clone()

    @staticmethod
    def _rebuild_topk_indices(
        handle: dict[str, Any],
        current_device_id: int,
        expected_shape: torch.Size,
    ) -> torch.Tensor:
        """Open the top-k indices IPC view with schema validation.

        The indices tensor must share its shape with the values tensor (same
        batch/seq/K) and the handle must carry valid vocab bounds.
        """
        assert "topk_indices_ipc" in handle, (
            "top-k teacher handle must include topk_indices_ipc"
        )
        vocab_start_index = handle.get("vocab_start_index")
        vocab_end_index = handle.get("vocab_end_index")
        assert (
            vocab_start_index is not None
            and vocab_end_index is not None
            and vocab_start_index >= 0
            and vocab_end_index > vocab_start_index
        ), (
            "top-k handle requires valid vocab_start_index / vocab_end_index "
            f"(got {vocab_start_index}, {vocab_end_index})"
        )
        aB, aS, aK = handle["actual_shape"]
        indices = rebuild_cuda_tensor_from_ipc(
            handle["topk_indices_ipc"], current_device_id
        ).detach()
        indices = indices[:aB, :aS, :aK].clone()
        assert indices.shape == expected_shape, (
            f"teacher top-k indices shape {tuple(indices.shape)} does not "
            f"match values shape {tuple(expected_shape)}"
        )
        return indices

    def _reconstruct_full_teacher_vocab_across_tp(
        self,
        own_rank_tensor: torch.Tensor,
        own_handle: dict[str, Any],
        current_device_id: int,
    ) -> torch.Tensor:
        """Concatenate every TP rank's local vocab shard into a full vocab.

        Own-rank tensor is reused as-is. Sibling ``tp_rank`` entries are
        rebuilt from their IPC handles (which requires peer-to-peer CUDA
        access between the student's GPU and each sibling's GPU on the same
        node). No-op when ``tp_size == 1``.
        """
        if self.tp_size == 1:
            return own_rank_tensor

        tp_handles = self._get_tp_group_handles_for_current_microbatch()
        assert tp_handles is not None, (
            "tp_size > 1 requires teacher_tp_group_results to be provided "
            "for cross-tokenizer full-logits path"
        )
        assert len(tp_handles) == self.tp_size

        aB_own, aS_own, _ = own_handle["actual_shape"]
        tp_shards: list[torch.Tensor] = []
        for tp_r, sibling_handle in enumerate(tp_handles):
            if tp_r == self.tp_rank:
                tp_shards.append(own_rank_tensor)
                continue
            aB_s, aS_s, aV_s = sibling_handle["actual_shape"]
            assert (aB_s, aS_s) == (aB_own, aS_own), (
                f"TP sibling shape mismatch at tp_rank={tp_r}: "
                f"(B={aB_s},S={aS_s}) vs own (B={aB_own},S={aS_own})"
            )
            sibling = rebuild_cuda_tensor_from_ipc(
                sibling_handle["payload_ipc"], current_device_id
            ).detach()
            sibling = sibling[:aB_s, :aS_s, :aV_s].clone()
            tp_shards.append(sibling)
        return torch.cat(tp_shards, dim=-1)

    def _reconstruct_full_teacher_sequence_across_cp(
        self, teacher_logits_tensor: torch.Tensor
    ) -> torch.Tensor:
        """All-gather CP-sharded teacher sequence to reconstruct full ``(B, S, V)``.

        Teacher forward ran with CP, so each rank's IPC payload is only its
        local CP shard of the sequence — and, because NeMo-RL uses
        load-balanced CP chunking, that shard is **non-contiguous** in the
        global sequence (rank ``i`` holds chunks ``i`` and
        ``2*cp_size-1-i``). ``allgather_cp_sharded_tensor`` handles both the
        all-gather and the un-chunking to recover the global sequence order.

        No-op when ``cp_size == 1``.
        """
        if self.cp_size == 1 or self.cp_group is None:
            return teacher_logits_tensor
        return allgather_cp_sharded_tensor(
            teacher_logits_tensor, self.cp_group, seq_dim=1
        )

    def _inject_teacher_ipc_tensors(self, loss_kwargs: dict[str, Any]) -> None:
        """Populate ``loss_kwargs`` with teacher tensors from IPC for this mb.

        No-op when:
            - sequence packing is enabled (IPC path currently does not support
              sequence-packed microbatches),
            - no teacher handle is available for the current microbatch index,
            - teacher export wasn't set on this post-processor.

        On the full-logits path, reconstructs the complete teacher tensor
        ``(B, S_full, V_full)`` by concatenating TP shards along the vocab
        dim (when ``tp_size > 1``) and then all-gathering across CP to
        recover the full sequence (when ``cp_size > 1``). On the top-k path,
        populates ``teacher_topk_indices_ipc``; top-k values are already
        global across TP via ``distributed_vocab_topk`` in the teacher.
        """
        if self.enable_seq_packing:
            return
        handle = self._get_current_microbatch_teacher_handle()
        if handle is None:
            return

        current_device_id = torch.cuda.current_device()
        is_topk = bool(handle.get("is_topk", False))

        teacher_logits_tensor = self._rebuild_own_rank_payload(
            handle, current_device_id
        )

        if is_topk:
            loss_kwargs["teacher_topk_indices_ipc"] = self._rebuild_topk_indices(
                handle, current_device_id, teacher_logits_tensor.shape
            )
        else:
            # Full-logits path: own-rank payload is a local vocab shard when
            # tp_size > 1 and/or a local sequence shard when cp_size > 1.
            # First concat sibling TP shards along vocab (yields full V at
            # local CP seq-slice); then all-gather across CP to recover the
            # global sequence. After both steps teacher_logits_tensor has
            # shape (B, S_full, V_full).
            _, _, aK = handle["actual_shape"]
            assert aK > 0, f"full-logits handle has degenerate vocab dim aK={aK}"
            if self.tp_size > 1 and not handle.get("vocab_sharded"):
                warnings.warn(
                    "tp_size > 1 but teacher full-logits handle reports "
                    "vocab_sharded=False; cross-tokenizer loss may see only "
                    "a partial vocab.",
                    stacklevel=2,
                )
            if self.cp_size > 1 and not handle.get("sequence_sharded"):
                warnings.warn(
                    "cp_size > 1 but teacher full-logits handle reports "
                    "sequence_sharded=False; cross-tokenizer loss may see "
                    "only a partial sequence.",
                    stacklevel=2,
                )
            teacher_logits_tensor = self._reconstruct_full_teacher_vocab_across_tp(
                teacher_logits_tensor, handle, current_device_id
            )
            teacher_logits_tensor = self._reconstruct_full_teacher_sequence_across_cp(
                teacher_logits_tensor
            )

        loss_kwargs["teacher_logits"] = teacher_logits_tensor

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.cp_size > 1:
            _, data_dict = prepare_data_for_cp(
                data_dict, processed_inputs, self.cp_mesh, sequence_dim
            )
            logits = redistribute_logits_for_cp(
                logits, self.device_mesh, self.cp_mesh, sequence_dim
            )

        if self.enable_seq_packing:
            loss_fn_ = SequencePackingLossWrapper(
                loss_fn=self.loss_fn,
                cu_seqlens_q=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
                cu_seqlens_q_padded=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
            )
        else:
            loss_fn_ = self.loss_fn

        loss_kwargs: dict[str, Any] = {}
        self._inject_teacher_ipc_tensors(loss_kwargs)

        loss, loss_metrics = loss_fn_(
            logits,
            data_dict,
            global_valid_seqs,
            global_valid_toks,
            **loss_kwargs,
        )
        return loss, loss_metrics


class XTokenTeacherIPCExportPostProcessor(LossPostProcessor):
    """Teacher-side post-processor that exports per-microbatch logits via CUDA IPC.

    XToken IPC microbatch handle schema (schema_version=1):
        - schema_version (int): record format version.
        - world_rank (int): global rank that produced this record.
        - tp_rank, tp_size (int): tensor-parallel coordinates of the producer.
        - cp_rank, cp_size (int): context-parallel coordinates of the producer.
        - actual_shape (tuple): shape of the exported local shard (not the full
          tensor); student must slice with these dims before clone.
        - sequence_sharded (bool): True iff cp_size > 1 (sequence dim sharded).
        - vocab_sharded (bool): True iff the exported tensor's vocab dim is a
          local TP shard.
        - vocab_start_index, vocab_end_index (int): global vocab column bounds
          covered by this shard.
        - is_topk (bool): whether payload_ipc carries top-k values (True) or
          full local log-probs (False).
        - payload_ipc (tuple): IPC handle for the payload buffer (top-k values
          when is_topk=True, log-probs otherwise).
        - topk_indices_ipc (tuple, top-k path only): IPC handle for the
          top-k indices buffer. Indices are always in global vocab id space.
    """

    def __init__(
        self,
        *args: Any,
        tp_mesh: Any,
        topk_logits: Optional[int],
        is_mdlm: bool = False,
        **kwargs: Any,
    ):
        # Keep explicit tp_mesh for local use and also forward it to base init.
        super().__init__(*args, tp_mesh=tp_mesh, **kwargs)
        self.tp_mesh = tp_mesh
        # Cache TP/CP parallel metadata; reuse existing mesh wiring only.
        self.tp_group = self.tp_mesh.get_group()
        self.tp_rank = torch.distributed.get_rank(self.tp_group)
        self.tp_size = torch.distributed.get_world_size(self.tp_group)
        if self.cp_size > 1 and self.cp_mesh is not None:
            self.cp_group = self.cp_mesh.get_group()
            self.cp_rank = torch.distributed.get_rank(self.cp_group)
        else:
            self.cp_group = None
            self.cp_rank = 0
        assert self.tp_size >= 1, f"tp_size must be >= 1, got {self.tp_size}"
        assert self.cp_size >= 1, f"cp_size must be >= 1, got {self.cp_size}"

        self.topk_logits = topk_logits
        self.is_mdlm = is_mdlm
        self.microbatch_handles: list[dict[str, Any]] = []
        self._microbatch_idx = 0
        self._mb_vals_buffers: list[torch.Tensor] = []
        self._mb_vals_ipcs: list[tuple[Any]] = []
        self._mb_idx_buffers: list[torch.Tensor] = []
        self._mb_idx_ipcs: list[tuple[Any]] = []
        self._mb_logits_buffers: list[torch.Tensor] = []
        self._mb_logits_ipcs: list[tuple[Any]] = []

    def set_microbatch_index(self, mb_idx: int) -> None:
        self._microbatch_idx = mb_idx

    def _ensure_topk_buffer(
        self,
        buf_idx: int,
        B: int,
        S: int,
        K: int,
        vals_dtype: torch.dtype,
        idx_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        while len(self._mb_vals_buffers) <= buf_idx:
            vals_buf = torch.empty((B, S, K), dtype=vals_dtype, device=device)
            idx_buf = torch.empty((B, S, K), dtype=idx_dtype, device=device)
            self._mb_vals_buffers.append(vals_buf)
            self._mb_vals_ipcs.append(get_handle_from_tensor(vals_buf))
            self._mb_idx_buffers.append(idx_buf)
            self._mb_idx_ipcs.append(get_handle_from_tensor(idx_buf))
        vals_buf = self._mb_vals_buffers[buf_idx]
        idx_buf = self._mb_idx_buffers[buf_idx]
        needs_realloc = (
            vals_buf.shape[0] < B
            or vals_buf.shape[1] < S
            or vals_buf.shape[2] < K
            or vals_buf.dtype != vals_dtype
            or vals_buf.device != device
            or idx_buf.shape[0] < B
            or idx_buf.shape[1] < S
            or idx_buf.shape[2] < K
            or idx_buf.dtype != idx_dtype
            or idx_buf.device != device
        )
        if needs_realloc:
            vals_buf = torch.empty((B, S, K), dtype=vals_dtype, device=device)
            idx_buf = torch.empty((B, S, K), dtype=idx_dtype, device=device)
            self._mb_vals_buffers[buf_idx] = vals_buf
            self._mb_vals_ipcs[buf_idx] = get_handle_from_tensor(vals_buf)
            self._mb_idx_buffers[buf_idx] = idx_buf
            self._mb_idx_ipcs[buf_idx] = get_handle_from_tensor(idx_buf)

    def _ensure_logits_buffer(
        self,
        buf_idx: int,
        B: int,
        S: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        while len(self._mb_logits_buffers) <= buf_idx:
            buf = torch.empty((B, S, V), dtype=dtype, device=device)
            self._mb_logits_buffers.append(buf)
            self._mb_logits_ipcs.append(get_handle_from_tensor(buf))
        buf = self._mb_logits_buffers[buf_idx]
        needs_realloc = (
            buf.shape[0] < B
            or buf.shape[1] < S
            or buf.shape[2] < V
            or buf.dtype != dtype
            or buf.device != device
        )
        if needs_realloc:
            buf = torch.empty((B, S, V), dtype=dtype, device=device)
            self._mb_logits_buffers[buf_idx] = buf
            self._mb_logits_ipcs[buf_idx] = get_handle_from_tensor(buf)

    def _assert_logits_vocab_sharded_for_tp(self, logits: torch.Tensor) -> None:
        """Assert that TP>1 logits really have a Shard(vocab) placement.

        The downstream pipeline (``vocab_start_index`` /
        ``_compute_distributed_log_softmax`` / ``distributed_vocab_topk``) is
        only correct when logits are vocab-sharded across the TP group
        whenever ``tp_size > 1``. A replicated DTensor with ``tp_size > 1``
        would cause ``_compute_distributed_log_softmax`` to sum identical
        exp-logits across ranks and silently produce wrong results.
        """
        if self.tp_size == 1:
            return
        if isinstance(logits, DTensor):
            vocab_dim = logits.ndim - 1
            placements_vocab_sharded = any(
                isinstance(p, Shard) and p.dim == vocab_dim for p in logits.placements
            )
        else:
            placements_vocab_sharded = False
        assert placements_vocab_sharded, (
            f"tp_size={self.tp_size} requires logits to be sharded along the "
            f"vocab dim across TP, but the input is not. "
            f"logits type={type(logits).__name__}, "
            f"placements="
            f"{logits.placements if isinstance(logits, DTensor) else 'N/A'}. "
            f"_compute_distributed_log_softmax would produce incorrect results "
            f"on replicated logits."
        )

    def _compute_local_log_probs(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Return ``(local_log_probs, vocab_start_index, vocab_end_index)``.

        Unwraps a DTensor to its local shard if needed, computes the
        TP-aware log-softmax (reducing across ``self.tp_group``), and applies
        the MDLM shared-seq halving when enabled.
        """
        if isinstance(logits, DTensor):
            mb_logits_local = logits.to_local()
        else:
            mb_logits_local = logits

        V_local = int(mb_logits_local.shape[-1])
        vocab_start_index = self.tp_rank * V_local
        vocab_end_index = (self.tp_rank + 1) * V_local
        assert vocab_end_index > vocab_start_index, (
            f"vocab_end_index ({vocab_end_index}) must be > vocab_start_index "
            f"({vocab_start_index})"
        )

        mb_logits_local = mb_logits_local.to(torch.float32)
        mb_log_prob = _compute_distributed_log_softmax(
            mb_logits_local, group=self.tp_group
        )
        del mb_logits_local
        if isinstance(mb_log_prob, DTensor):
            mb_log_prob = mb_log_prob.to_local()

        if self.is_mdlm:
            shared_seq_len = int(mb_log_prob.shape[1] / 2)
            mb_log_prob = mb_log_prob[:, shared_seq_len:, :]

        return mb_log_prob, vocab_start_index, vocab_end_index

    def _base_handle_metadata(
        self,
        world_rank: int,
        vocab_start_index: int,
        vocab_end_index: int,
        sequence_sharded: bool,
    ) -> dict[str, Any]:
        """Return schema-v1 fields common to top-k and full-logits records.

        The per-path builders fill in ``actual_shape``, ``is_topk``,
        ``payload_ipc``, ``vocab_sharded`` (and ``topk_indices_ipc`` on the
        top-k path) on top of this base.
        """
        return {
            "schema_version": 1,
            "world_rank": world_rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "cp_rank": self.cp_rank,
            "cp_size": self.cp_size,
            "sequence_sharded": sequence_sharded,
            "vocab_start_index": vocab_start_index,
            "vocab_end_index": vocab_end_index,
        }

    def _export_topk_handle(
        self,
        mb_log_prob: torch.Tensor,
        buf_idx: int,
        vocab_start_index: int,
        vocab_end_index: int,
        sequence_sharded: bool,
        world_rank: int,
    ) -> dict[str, Any]:
        """Run TP-aware top-k, stage values + indices in IPC, build handle."""
        mb_topk_vals, mb_topk_idx = distributed_vocab_topk(
            mb_log_prob,
            k=self.topk_logits,
            tp_group=self.tp_group,
            vocab_start_index=vocab_start_index,
            vocab_end_index=vocab_end_index,
        )
        B_mb, S_mb, K_mb = mb_topk_vals.shape
        self._ensure_topk_buffer(
            buf_idx,
            B_mb,
            S_mb,
            K_mb,
            mb_topk_vals.dtype,
            mb_topk_idx.dtype,
            mb_topk_vals.device,
        )
        self._mb_vals_buffers[buf_idx][:B_mb, :S_mb, :K_mb].copy_(mb_topk_vals)
        self._mb_idx_buffers[buf_idx][:B_mb, :S_mb, :K_mb].copy_(mb_topk_idx)
        del mb_topk_vals, mb_topk_idx

        payload_ipc = self._mb_vals_ipcs[buf_idx]
        topk_indices_ipc = self._mb_idx_ipcs[buf_idx]
        assert payload_ipc is not None, "topk payload_ipc must be set"
        assert topk_indices_ipc is not None, (
            "topk_indices_ipc must be set on the top-k export path"
        )

        handle = self._base_handle_metadata(
            world_rank, vocab_start_index, vocab_end_index, sequence_sharded
        )
        handle.update(
            {
                "actual_shape": (B_mb, S_mb, K_mb),
                # Top-k values span the global vocab (indices are global), so
                # the payload itself is not vocab-sharded even when tp_size > 1.
                "vocab_sharded": False,
                "is_topk": True,
                "payload_ipc": payload_ipc,
                "topk_indices_ipc": topk_indices_ipc,
            }
        )
        return handle

    def _export_full_logits_handle(
        self,
        mb_log_prob: torch.Tensor,
        buf_idx: int,
        vocab_start_index: int,
        vocab_end_index: int,
        sequence_sharded: bool,
        vocab_sharded: bool,
        world_rank: int,
    ) -> dict[str, Any]:
        """Stage local log-probs in IPC, build handle."""
        B_mb, S_mb, V_mb = mb_log_prob.shape
        self._ensure_logits_buffer(
            buf_idx,
            B_mb,
            S_mb,
            V_mb,
            mb_log_prob.dtype,
            mb_log_prob.device,
        )
        self._mb_logits_buffers[buf_idx][:B_mb, :S_mb, :V_mb].copy_(mb_log_prob)

        payload_ipc = self._mb_logits_ipcs[buf_idx]
        assert payload_ipc is not None, "full-logits payload_ipc must be set"

        handle = self._base_handle_metadata(
            world_rank, vocab_start_index, vocab_end_index, sequence_sharded
        )
        handle.update(
            {
                "actual_shape": (B_mb, S_mb, V_mb),
                "vocab_sharded": vocab_sharded,
                "is_topk": False,
                "payload_ipc": payload_ipc,
            }
        )
        return handle

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],  # noqa: ARG002
        processed_inputs: ProcessedInputs,  # noqa: ARG002
        global_valid_seqs: torch.Tensor,  # noqa: ARG002
        global_valid_toks: torch.Tensor,  # noqa: ARG002
        sequence_dim: int = 1,  # noqa: ARG002
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self._assert_logits_vocab_sharded_for_tp(logits)

        mb_log_prob, vocab_start_index, vocab_end_index = self._compute_local_log_probs(
            logits
        )

        # Shared metadata for both topk and full-logits records.
        # ``_assert_logits_vocab_sharded_for_tp`` guarantees
        # ``tp_size > 1 => placements_vocab_sharded``, and ``tp_size == 1``
        # means the "shard" is the full vocab (no sharding observable by
        # the student), so we key purely on tp_size.
        vocab_sharded = self.tp_size > 1
        sequence_sharded = self.cp_size > 1
        world_rank = torch.distributed.get_rank()
        buf_idx = self._microbatch_idx

        if self.topk_logits is not None:
            handle = self._export_topk_handle(
                mb_log_prob,
                buf_idx,
                vocab_start_index,
                vocab_end_index,
                sequence_sharded,
                world_rank,
            )
        else:
            handle = self._export_full_logits_handle(
                mb_log_prob,
                buf_idx,
                vocab_start_index,
                vocab_end_index,
                sequence_sharded,
                vocab_sharded,
                world_rank,
            )

        self.microbatch_handles.append(handle)

        dummy_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        return dummy_loss, {"num_valid_samples": 1.0}
