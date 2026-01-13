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

from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from megatron.core.pipeline_parallel import get_forward_backward_func

from nemo_rl.algorithms.loss_functions import LossFunction, SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.models.megatron.data import ProcessedMicrobatch

# Union type for any post-processing function (defined after classes below)
PostProcessingFunction = Union[
    "LossPostProcessor",
    "LogprobsPostProcessor",
    "TopkLogitsPostProcessor",
]


def model_forward(
    model: GPTModel,
    data_dict: BatchedDataDict[Any],
    cfg: Dict[str, Any],
    input_ids_cp_sharded: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params: Optional[PackedSeqParams] = None,
    defer_fp32_logits: Optional[bool] = None,
) -> torch.Tensor:
    """Perform a single forward pass through the model.

    Args:
        model: The model to run forward pass on
        data_dict (BatchedDataDict): Dictionary containing batch data
        cfg (dict): Configuration dictionary
        input_ids_cp_sharded: Context-parallel sharded input token IDs
        position_ids: Position IDs for tokens
        attention_mask: Attention mask for the sequence
        packed_seq_params: Parameters for packed sequences (optional)
        defer_fp32_logits (Optional[bool]): Whether to skip the conversion of logits to fp32

    Returns:
        torch.Tensor: Output tensor from the model (logits)
    """
    multimodal_data = data_dict.get_multimodal_dict(
        as_tensors=True, device=input_ids_cp_sharded.device
    )
    if len(multimodal_data) > 0:
        position_ids = None

    additional_kwargs = {}
    # Mamba models currently do not support packed_seq_params
    if packed_seq_params is not None:
        additional_kwargs["packed_seq_params"] = packed_seq_params
    if defer_fp32_logits:
        additional_kwargs["fp32_output"] = False
    # with straggler_timer:
    output_tensor = model(
        input_ids=input_ids_cp_sharded,
        position_ids=position_ids,
        attention_mask=attention_mask,
        **additional_kwargs,
        **multimodal_data,
    )

    # Apply temperature scaling to logits for training
    # This matches the dtensor worker's _apply_temperature_scaling in the train method
    if "generation" in cfg and cfg["generation"] is not None:
        output_tensor.div_(cfg["generation"]["temperature"])

    return output_tensor


def forward_with_post_processing_fn(
    data_iterator: Iterator[ProcessedMicrobatch],
    model: GPTModel,
    cfg: Dict[str, Any],
    post_processing_fn: PostProcessingFunction,
    defer_fp32_logits: Optional[bool] = True,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Callable]:
    """Perform forward pass with pre-processed microbatch and return output tensor and post-processing function.

    This function takes a pre-processed microbatch (with sequence packing already handled),
    runs the forward step through the model, and prepares a post-processing function for
    post-processing the outputs.

    Args:
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        model: The model to run forward pass on
        cfg (dict): Configuration dictionary
        post_processing_fn: Post-processing function to post-process the logits
        defer_fp32_logits: Whether to defer FP32 conversion of logits
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization

    Returns:
        tuple: (output_tensor, post_processing_fn_wrapped)
            - output_tensor: Raw model outputs (logits)
            - post_processing_fn_wrapped: Function to create output post-processing function when called
    """
    # Get the pre-processed microbatch from the iterator
    processed_mb = next(data_iterator)

    # Extract the processed components
    data_dict = processed_mb.data_dict
    input_ids = processed_mb.input_ids
    input_ids_cp_sharded = processed_mb.input_ids_cp_sharded
    attention_mask = processed_mb.attention_mask
    position_ids = processed_mb.position_ids
    packed_seq_params = processed_mb.packed_seq_params
    cu_seqlens_padded = processed_mb.cu_seqlens_padded

    output_tensor = model_forward(
        model,
        data_dict,
        cfg,
        input_ids_cp_sharded,
        position_ids,
        attention_mask,
        packed_seq_params,
        defer_fp32_logits,
    )

    ## calling post_processing_fn will return a function that takes the output tensor and returns a tuple of (loss, metrics)
    # Use type checking to dispatch to the correct post-processing method
    if isinstance(post_processing_fn, LossPostProcessor):
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            packed_seq_params=packed_seq_params,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )
    elif isinstance(post_processing_fn, LogprobsPostProcessor):
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            input_ids=input_ids,
            cu_seqlens_padded=cu_seqlens_padded,
        )
    elif isinstance(post_processing_fn, TopkLogitsPostProcessor):
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            cu_seqlens_padded=cu_seqlens_padded,
        )
    else:
        raise TypeError(
            f"Unknown post-processing function type: {type(post_processing_fn)}"
        )

    return output_tensor, post_processing_fn_wrapped


def megatron_forward_backward(
    model: GPTModel,
    cfg: Dict[str, Any],
    data_iterator: Iterator[ProcessedMicrobatch],
    num_microbatches: int,
    seq_length: int,
    mbs: int,
    post_processing_fn: PostProcessingFunction,
    forward_only: bool = False,
    defer_fp32_logits: Optional[bool] = None,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    do_not_average_loss: bool = False,
) -> Any:
    """Execute forward and backward passes using Megatron's utilities.

    This is the main training loop function that coordinates forward and backward
    passes across multiple microbatches using Megatron's pipeline parallel
    execution framework.

    Args:
        model: The model to train
        cfg (dict): Configuration dictionary
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        num_microbatches (int): Number of microbatches to process
        seq_length (int): Sequence length
        mbs (int): Micro batch size
        post_processing_fn: Post-processing function to post-process the logits
        forward_only (bool): If True, skip backward pass
        defer_fp32_logits (Optional[bool]): Whether to skip the conversion of logits to fp32
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization

    Returns:
        BatchedDataDict: Results from the forward/backward execution
    """
    forward_step = partial(
        forward_with_post_processing_fn,
        cfg=cfg,
        post_processing_fn=post_processing_fn,
        defer_fp32_logits=defer_fp32_logits,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )
    forward_backward_func = get_forward_backward_func()
    return forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=mbs,
        decoder_seq_length=seq_length,
        forward_only=forward_only,
        do_not_average_loss=do_not_average_loss,
    )


class LossPostProcessor:
    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: Dict[str, Any],
        cp_normalize: bool = True,
    ):
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.cp_normalize = cp_normalize

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Optional[PackedSeqParams] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]]:
        """Create a loss post-processing function for training.

        This function wraps a loss function with the necessary context and parameters
        to compute loss and metrics from model outputs. It handles sequence packing
        and context parallelism normalization.

        Args:
            loss_fn: The base loss function to wrap
            cfg (dict): Configuration dictionary
            data_dict: Batched data dictionary
            packed_seq_params: Parameters for packed sequences (optional)
            cp_normalize (bool): Whether to normalize by context parallel size

        Returns:
            Callable: Function that takes output tensor and returns (loss, metrics) tuple
        """
        loss_fn = self.loss_fn
        pack_sequences = self.cfg["sequence_packing"]["enabled"]
        if pack_sequences and packed_seq_params is not None:
            # remove padding
            loss_fn = SequencePackingLossWrapper(
                loss_fn=loss_fn,
                cu_seqlens_q=packed_seq_params.cu_seqlens_q,
                cu_seqlens_q_padded=packed_seq_params.cu_seqlens_q_padded,
            )

        loss_fn_wrapped = partial(
            loss_fn,
            data=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            vocab_parallel_rank=get_tensor_model_parallel_rank(),
            vocab_parallel_group=get_tensor_model_parallel_group(),
            context_parallel_group=get_context_parallel_group(),
        )

        if self.cp_normalize:
            cp_size = get_context_parallel_world_size()
            orig_loss_fn_wrapped = loss_fn_wrapped

            def _div_by_cp_size(*args, **kwargs):
                loss, metrics = orig_loss_fn_wrapped(*args, **kwargs)
                return loss / cp_size, metrics

            loss_fn_wrapped = _div_by_cp_size

        return loss_fn_wrapped


class LogprobsPostProcessor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        input_ids: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Create a post-processing function that computes token log probabilities.

        This function returns a processor that takes model logits and converts them
        to token-level log probabilities, handling both packed and unpacked sequences.

        Args:
            data_dict: Batched data dictionary containing input sequences
            input_ids: Processed input token IDs
            cu_seqlens_padded: Cumulative sequence lengths for packed sequences

        Returns:
            Callable: Function that takes output tensor and returns (dummy_loss, {"logprobs": token_logprobs})
        """
        unpacked_input_ids = data_dict["input_ids"]
        original_seq_length = unpacked_input_ids.shape[1]

        def processor_fn_inner(output_tensor):
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

        return processor_fn_inner


class TopkLogitsPostProcessor:
    def __init__(self, cfg: Dict[str, Any], k: int):
        self.cfg = cfg
        self.k = k

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        cu_seqlens_padded: torch.Tensor,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Create a post-processing function that computes top-k logits and indices.

        This function returns a processor that extracts the top-k highest logits
        and their corresponding vocabulary indices from model outputs. It handles
        tensor parallelism, context parallelism, and sequence packing.

        Args:
            data_dict: Batched data dictionary
            cu_seqlens_padded: Cumulative sequence lengths for packed sequences

        Returns:
            Callable: Function that takes output tensor and returns
                      (dummy_loss, {"topk_logits": values, "topk_indices": indices})
        """
        pack = self.cfg["sequence_packing"]["enabled"]
        cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
        unpacked_seqlen = data_dict["input_ids"].shape[1]
        seq_lengths = data_dict["input_lengths"]

        def processor_fn_inner(output_tensor):
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
                self.k,
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
                        (1, total_packed_len, self.k),
                        dtype=topk_vals_local.dtype,
                        device=topk_vals_local.device,
                    )
                    topk_idx_full = torch.zeros(
                        (1, total_packed_len, self.k),
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
                    (batch_size, unpacked_seqlen, self.k),
                    dtype=topk_vals_full.dtype,
                    device=topk_vals_full.device,
                )
                out_idx = torch.zeros(
                    (batch_size, unpacked_seqlen, self.k),
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

        return processor_fn_inner
