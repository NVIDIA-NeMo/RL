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
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    get_logprobs_from_vocab_parallel_logits,
)
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.models.policy import PolicyConfig

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

    # THD packing (seq_packing + CP with TE): pass cu_seqlens directly.
    # Automodel models accept cu_seqlens via **attn_kwargs — the attention
    # utils (preprocess_args_and_kwargs_for_attn) detect "cu_seqlens" in kwargs
    # and route to TE's THD path with qkv_format="thd".
    if processed_inputs.packed_seq_params is not None:
        cu_padded = processed_inputs.cu_seqlens_padded
        model_args["cu_seqlens"] = cu_padded
        if processed_inputs.cu_seqlens_padded is not None:
            model_args["cu_seqlens_padded"] = cu_padded
        max_seqlen = int((cu_padded[1:] - cu_padded[:-1]).max().item())
        model_args["max_seqlen"] = max_seqlen
        model_args.pop("attention_mask", None)

    # Add flash attention kwargs if applicable
    elif processed_inputs.has_flash_attention:
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


# ---------------------------------------------------------------------------
# Pipeline Parallelism (PP) utilities
# ---------------------------------------------------------------------------


def broadcast_tensors_from_last_pp_stage(
    tensors: dict[str, Optional[torch.Tensor]],
    pp_mesh: Any,
    has_last_stage: bool,
) -> dict[str, torch.Tensor]:
    """Broadcast tensors from last PP stage to all stages.

    Adapted from nemo_rl/models/megatron/pipeline_parallel.py for device-mesh
    based PP (no Megatron parallel state).

    Args:
        tensors: Dict mapping names to tensors. On last stage, tensors are
            populated; on other stages they may be None.
        pp_mesh: Pipeline parallel DeviceMesh.
        has_last_stage: Whether this rank owns the last pipeline stage.

    Returns:
        Dict with the same keys, all tensors populated on every rank.
    """
    pp_group = pp_mesh.get_group()
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    last_global_rank = pp_ranks[-1]

    result = {}
    for name, tensor in tensors.items():
        # Broadcast metadata (shape, dtype)
        if has_last_stage:
            assert tensor is not None, f"Last stage must provide tensor '{name}'"
            meta = [list(tensor.shape), str(tensor.dtype)]
        else:
            meta = [None, None]
        meta_list = [meta]
        torch.distributed.broadcast_object_list(
            meta_list, src=last_global_rank, group=pp_group
        )
        shape, dtype_str = meta_list[0]

        if not has_last_stage:
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            tensor = torch.empty(shape, dtype=dtype, device="cuda")

        torch.distributed.broadcast(tensor, src=last_global_rank, group=pp_group)
        result[name] = tensor

    return result


def broadcast_loss_metrics_from_last_pp_stage(
    metrics: Optional[list[dict[str, Any]]],
    pp_mesh: Any,
    has_last_stage: bool,
) -> list[dict[str, Any]]:
    """Broadcast loss metrics from last PP stage to all stages.

    Args:
        metrics: List of metric dicts (populated on last stage, None elsewhere).
        pp_mesh: Pipeline parallel DeviceMesh.
        has_last_stage: Whether this rank owns the last pipeline stage.

    Returns:
        List of metric dicts on all ranks.
    """
    pp_group = pp_mesh.get_group()
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    last_global_rank = pp_ranks[-1]

    obj = [metrics]
    torch.distributed.broadcast_object_list(obj, src=last_global_rank, group=pp_group)
    return obj[0]


class PPLossAdapter:
    """Stateful loss adapter for pipeline parallel schedules.

    Bridges NeMo RL's LossFunction interface with the PP schedule's
    ``loss_fn(output, target)`` contract by pre-chunking RL data across
    microbatches.

    The PP schedule calls ``__call__(output, target)`` once per microbatch.
    This adapter indexes into pre-chunked RL tensors to compute the loss for
    each microbatch.

    **Critical**: PP schedules sum loss across microbatches, so the loss
    returned here must use ``reduction="sum"`` (not mean).
    """

    def __init__(
        self,
        loss_fn: "LossFunction",
        cfg: Any,
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        dp_size: int,
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ):
        self._loss_fn = loss_fn
        self._cfg = cfg
        self._device_mesh = device_mesh
        self._cp_mesh = cp_mesh
        self._tp_mesh = tp_mesh
        self._cp_size = cp_size
        self._dp_size = dp_size
        self._enable_seq_packing = enable_seq_packing
        self._sampling_params = sampling_params

        self._microbatches: list[dict[str, Any]] = []
        self._call_idx: int = 0
        self._all_metrics: list[dict[str, Any]] = []
        self._global_valid_seqs: Optional[torch.Tensor] = None
        self._global_valid_toks: Optional[torch.Tensor] = None
        self._num_global_batches: int = 1

    def set_microbatches(
        self,
        data_dict: "BatchedDataDict[Any]",
        n_microbatches: int,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        num_global_batches: int = 1,
    ) -> None:
        """Pre-chunk RL tensors along batch dim into n_microbatches."""
        self._call_idx = 0
        self._all_metrics = []
        self._global_valid_seqs = global_valid_seqs
        self._global_valid_toks = global_valid_toks
        self._num_global_batches = num_global_batches

        self._microbatches = []
        for i in range(n_microbatches):
            mb = {}
            for key in data_dict:
                val = data_dict[key]
                if torch.is_tensor(val) and val.shape[0] > 0:
                    chunks = torch.tensor_split(val, n_microbatches, dim=0)
                    mb[key] = chunks[i]
                else:
                    mb[key] = val
            self._microbatches.append(mb)

    def __call__(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        """Called by PP schedule per microbatch.

        Following the validated pattern from automodel-engine/test_rl_pp.py:
        compute logprobs directly from the schedule's output and target tensors,
        then use pre-chunked RL data for advantages/prev_logprobs.

        Args:
            output: Model output (has .logits attribute or is a tensor).
            target: Labels tensor for this microbatch (schedule-chunked).

        Returns:
            Scalar loss tensor with reduction="sum".
        """
        logits = getattr(output, "logits", output)
        mb_data = self._microbatches[self._call_idx]
        self._call_idx += 1

        # Compute next-token logprobs directly from schedule's logits and target.
        # Following test_rl_pp.py: use target (schedule-chunked labels) for gather.
        # Returns [B, S-1] matching the non-PP path's get_next_token_logprobs_from_logits.
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        curr_logprobs = (
            log_probs[:, :-1]
            .gather(dim=-1, index=target[:, 1:].unsqueeze(-1).clamp(min=0))
            .squeeze(-1)
        )

        # Build loss input with computed logprobs
        from nemo_rl.algorithms.loss.interfaces import LossInputType

        if self._loss_fn.input_type == LossInputType.LOGPROB:
            loss_input = {"next_token_logprobs": curr_logprobs}
        else:
            loss_input = {"logits": logits}

        loss, loss_metrics = self._loss_fn(
            data=mb_data,
            global_valid_seqs=self._global_valid_seqs,
            global_valid_toks=self._global_valid_toks,
            **loss_input,
        )

        # Scale metrics for aggregation
        for k in loss_metrics:
            if "_min" not in k and "_max" not in k:
                loss_metrics[k] /= self._num_global_batches

        self._all_metrics.append(loss_metrics)

        # Scale loss to cancel FSDP's automatic gradient averaging across DP ranks,
        # matching the non-PP path which does `loss * dp_size * cp_size` before backward.
        return loss * self._dp_size * self._cp_size

    def reset(self) -> None:
        """Reset state for the next forward-backward call."""
        self._call_idx = 0
        self._all_metrics = []


class PPLogprobsCapturer:
    """Pseudo-loss that captures logits from each PP microbatch.

    Used with ``schedule.eval()`` to collect logits on the last pipeline
    stage for logprob computation.
    """

    def __init__(self):
        self.captured_logits: list[torch.Tensor] = []

    def __call__(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        """Capture logits and return a dummy zero loss."""
        logits = getattr(output, "logits", output)
        self.captured_logits.append(logits.detach())
        return torch.tensor(0.0, device="cuda")

    def reset(self) -> None:
        self.captured_logits = []


class PPTopkCapturer:
    """Pseudo-loss that captures logits for top-k computation on last stage."""

    def __init__(self):
        self.captured_logits: list[torch.Tensor] = []

    def __call__(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        logits = getattr(output, "logits", output)
        self.captured_logits.append(logits.detach())
        return torch.tensor(0.0, device="cuda")

    def reset(self) -> None:
        self.captured_logits = []


def pp_forward_backward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_adapter: PPLossAdapter,
    *,
    forward_only: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Execute forward (and optionally backward) using the PP schedule.

    The PP schedule internally handles microbatch splitting, stage-to-stage
    communication, and gradient accumulation.

    Args:
        model: AutoPipeline model with .info and .parts attributes.
        batch: Dict with at least ``input_ids`` and ``labels``.
        loss_adapter: PPLossAdapter (already configured via set_microbatches).
        forward_only: If True, use schedule.eval() instead of schedule.step().

    Returns:
        Tuple of (total_loss, list_of_metric_dicts).
        total_loss is the summed loss on last stage, 0.0 on other stages.
    """
    schedule = model.info.schedule
    has_first = model.info.has_first_stage
    has_last = model.info.has_last_stage

    input_ids = batch.pop("input_ids")
    targets = batch.pop("labels", None) if has_last else None
    losses: Optional[list[torch.Tensor]] = [] if has_last else None

    # Inject the loss adapter into the schedule
    schedule._loss_fn = loss_adapter

    # Build args: first stage receives input_ids, others don't
    args = (input_ids,) if has_first else ()

    if forward_only:
        schedule.eval(*args, target=targets, losses=losses)
    else:
        schedule.step(*args, target=targets, losses=losses)

    if has_last and losses:
        total_loss = torch.sum(torch.stack(losses))
    else:
        total_loss = torch.tensor(0.0, device="cuda")

    return total_loss, loss_adapter._all_metrics
