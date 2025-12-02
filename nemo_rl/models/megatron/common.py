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
import time
from functools import partial
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.utils import get_ltor_masks_and_position_ids

from nemo_rl.algorithms.loss_functions import LossFunction, SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    _get_tokens_on_this_cp_rank,
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)

def _round_up_to_multiple(value: int, multiple: int) -> int:
    return (
        ((value + multiple - 1) // multiple * multiple)
        if value % multiple != 0
        else value
    )


def _pack_sequences_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_packed_seq_to: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> tuple[torch.Tensor, PackedSeqParams, torch.Tensor, Optional[torch.Tensor]]:
    """Pack sequences for Megatron model processing with optional context parallelism.

    Args:
        input_ids: Input token IDs [batch_size, seq_length]
        seq_lengths: Actual sequence lengths for each sample [batch_size]
        pad_individual_seqs_to_multiple_of: Pad individual sequences to a multiple of this value
        pad_packed_seq_to_multiple_of: Pad packed sequences to a multiple of this value
        pad_packed_seq_to: Pad packed sequences to this value (before CP)
            - The three parameters above can be calculated using _get_pack_sequence_parameters_for_megatron, we do not recommend users to set these parameters manually.
        cp_size: Context parallelism size

    Returns:
        Tuple of:
        - packed_input_ids: Packed input tensor [1, T]
        - input_ids_cp_sharded: Sharded input tensor [cp_size, T // cp_size]
        - packed_seq_params: PackedSeqParams object
        - cu_seqlens: Cumulative sequence lengths
        - cu_seqlens_padded: Padded cumulative sequence lengths
    """
    batch_size = input_ids.shape[0]

    # Build cumulative sequence lengths (cu_seqlens) and extract valid tokens
    needs_padding = (
        pad_individual_seqs_to_multiple_of > 1
        or pad_packed_seq_to_multiple_of > 1
        or pad_packed_seq_to is not None
    )

    cu_seqlens = [0]
    cu_seqlens_padded = [0] if needs_padding else None
    valid_tokens = []

    # Round up the pad_packed_seq_to to the nearest multiple of pad_packed_seq_to_multiple_of
    if pad_packed_seq_to is not None:
        pad_packed_seq_to = _round_up_to_multiple(
            pad_packed_seq_to, pad_packed_seq_to_multiple_of
        )

    pad_factor = pad_individual_seqs_to_multiple_of

    for b in range(batch_size):
        seq_len = (
            seq_lengths[b].item() if torch.is_tensor(seq_lengths[b]) else seq_lengths[b]
        )

        # Extract valid tokens for this sequence
        valid_tokens.append(input_ids[b, :seq_len])

        # Update cumulative sequence lengths
        cu_seqlens.append(cu_seqlens[-1] + seq_len)

        # For context parallelism, track padded sequence lengths
        if needs_padding:
            # Pad sequence length to multiple of (cp_size * 2)
            padded_seq_len = _round_up_to_multiple(seq_len, pad_factor)
            cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_seq_len)

    # Convert to tensors
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=input_ids.device)
    if needs_padding:
        cu_seqlens_padded = torch.tensor(
            cu_seqlens_padded, dtype=torch.int32, device=input_ids.device
        )
        if pad_packed_seq_to is not None:
            cu_seqlens_padded[-1] = pad_packed_seq_to
        elif pad_packed_seq_to_multiple_of > 1:
            cu_seqlens_padded[-1] = _round_up_to_multiple(
                cu_seqlens_padded[-1], pad_packed_seq_to_multiple_of
            )

    # Calculate max sequence length (padded if using CP)
    if needs_padding:
        seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
        max_seqlen = seq_lens_padded.max().item()
    else:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

    # Concatenate all valid tokens
    # If using individual padding, we need to pad individual sequences
    # CP will always need padding (of at least cp_size * 2)
    running_seq_len = 0
    if pad_factor > 1:
        all_input_ids = []
        padded_tokens = []
        for b in range(batch_size):
            seq_len = (
                seq_lengths[b].item()
                if torch.is_tensor(seq_lengths[b])
                else seq_lengths[b]
            )
            # if last element, pad to the max sequence length
            if b == batch_size - 1 and needs_padding:
                if pad_packed_seq_to is not None:
                    padded_seq_len = pad_packed_seq_to - running_seq_len
                elif pad_packed_seq_to_multiple_of > 1:
                    padded_seq_len = _round_up_to_multiple(seq_len, pad_factor)
                    padded_seq_len = (
                        _round_up_to_multiple(
                            running_seq_len + padded_seq_len,
                            pad_packed_seq_to_multiple_of,
                        )
                        - running_seq_len
                    )
                else:
                    padded_seq_len = _round_up_to_multiple(seq_len, pad_factor)
            else:
                padded_seq_len = _round_up_to_multiple(seq_len, pad_factor)

            running_seq_len += padded_seq_len

            # Pad this sequence to the required length
            seq_tokens = input_ids[b, :seq_len]
            if padded_seq_len > seq_len:
                # Pad with zeros (or use a padding token if available)
                seq_tokens = torch.nn.functional.pad(
                    seq_tokens, (0, padded_seq_len - seq_len), value=0
                )
            all_input_ids.append(seq_tokens)

            if cp_size > 1:
                seq_tokens = _get_tokens_on_this_cp_rank(
                    seq_tokens, cp_rank, cp_size, seq_dim=0
                )

            padded_tokens.append(seq_tokens)

        # Concatenate all padded tokens
        # For 'thd' format, the shape should be [1, T] where T is total tokens
        packed_input_ids = torch.cat(padded_tokens, dim=0).unsqueeze(0)
        all_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(0)
    else:
        # No individual padding, just concatenate valid tokens
        # For 'thd' format, the shape should be [1, T] where T is total tokens
        packed_input_ids = torch.cat(valid_tokens, dim=0).unsqueeze(0)
        all_input_ids = packed_input_ids
        if needs_padding:
            if pad_packed_seq_to is not None:
                pad_len = pad_packed_seq_to - packed_input_ids.shape[1]
            elif pad_packed_seq_to_multiple_of > 1:
                current_seq_len = packed_input_ids.shape[1]
                pad_this_seq_to = _round_up_to_multiple(
                    current_seq_len, pad_packed_seq_to_multiple_of
                )
                pad_len = pad_this_seq_to - current_seq_len
            else:
                pad_len = 0
            if pad_len > 0:
                packed_input_ids = torch.nn.functional.pad(
                    packed_input_ids, (0, pad_len), value=0
                )
                all_input_ids = torch.nn.functional.pad(
                    all_input_ids, (0, pad_len), value=0
                )

    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens.clone()

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max_seqlen),
        max_seqlen_kv=int(max_seqlen),
        qkv_format="thd",
    )

    return (
        all_input_ids.contiguous(),
        packed_input_ids.contiguous(),
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    )


def _get_pack_sequence_parameters_for_megatron(
    megatron_cfg: dict,
    max_seq_len_in_batch: int,
):
    """Get pack sequence parameters for Megatron model processing with optional context parallelism.

    Args:
        megatron_cfg: Megatron configuration
        max_seq_len_in_batch: Maximum sequence length in batch

    Returns:
        Tuple of:
        - pad_individual_seqs_to_multiple_of: Pad individual sequences to a multiple of this value
        - pad_packed_seq_to_multiple_of: Pad packed sequences to a multiple of this value
        - pad_packed_seq_to: Pad packed sequences to this value (before CP)
    """
    tp_size = megatron_cfg["tensor_model_parallel_size"]
    sp = megatron_cfg["sequence_parallel"]
    pp_size = megatron_cfg["pipeline_model_parallel_size"]
    cp_size = megatron_cfg["context_parallel_size"]
    fp8_cfg = megatron_cfg.get("fp8_cfg", None) or {}
    use_fp8 = fp8_cfg.get("enabled", False)
    use_blockwise_fp8 = fp8_cfg.get("fp8_recipe", None) == "blockwise"

    # individual sequence needs to be splitted to CP domain, and to TP domain when SP is enabled.
    pad_individual_seqs_to_multiple_of = 1
    if cp_size > 1:
        pad_individual_seqs_to_multiple_of *= cp_size * 2
    if tp_size > 1 and sp:
        pad_individual_seqs_to_multiple_of *= tp_size

    # packed sequence length, after splitted to TP and CP domains, needs to be divisible by 128 if using blockwise FP8, and divisible by 16 if using other FP8 recipes.
    if use_fp8:
        divisor = 128 if use_blockwise_fp8 else 16
        pad_packed_seq_to_multiple_of = divisor
        if cp_size > 1:
            pad_packed_seq_to_multiple_of *= cp_size * 2
        if tp_size > 1 and sp:
            pad_packed_seq_to_multiple_of *= tp_size
    else:
        pad_packed_seq_to_multiple_of = 1

    # when PP is used, all sequences must have the same length, so we need to pad the packed sequence to the max sequence length in the batch.
    if pp_size > 1:
        pad_packed_seq_to = max_seq_len_in_batch
    else:
        pad_packed_seq_to = None

    return (
        pad_individual_seqs_to_multiple_of,
        pad_packed_seq_to_multiple_of,
        pad_packed_seq_to,
    )


def _unpack_sequences_from_megatron(
    output_tensor: torch.Tensor,
    seq_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor],
    original_batch_size: int,
    original_seq_length: int,
) -> torch.Tensor:
    """Unpack sequences from Megatron output format.

    Args:
        output_tensor: Packed output tensor [1, T, vocab_size]
        seq_lengths: Actual sequence lengths for each sample
        cu_seqlens: Cumulative sequence lengths
        cu_seqlens_padded: Padded cumulative sequence lengths (if CP was used)
        original_batch_size: Original batch size
        original_seq_length: Original maximum sequence length

    Returns:
        Unpacked output tensor [batch_size, seq_length, vocab_size]
    """
    # Remove the batch dimension to get [T, vocab_size]
    output_tensor = output_tensor.squeeze(0)

    # Create a padded output tensor with original shape
    vocab_size = output_tensor.shape[-1]
    unpacked_output = torch.zeros(
        (original_batch_size, original_seq_length, vocab_size),
        dtype=output_tensor.dtype,
        device=output_tensor.device,
    )

    # Get context parallel size to determine which cu_seqlens to use
    cp_size = get_context_parallel_world_size()

    # Fill in the unpacked output tensor with valid tokens
    for b in range(original_batch_size):
        # Get actual sequence length for this sample
        seq_len = (
            seq_lengths[b].item() if torch.is_tensor(seq_lengths[b]) else seq_lengths[b]
        )

        if cp_size > 1 and cu_seqlens_padded is not None:
            # When using CP, we need to account for padding
            # Calculate the padded sequence boundaries
            pad_factor = cp_size * 2
            padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor
            start_idx = cu_seqlens_padded[b].item()

            # Only copy the valid tokens (not the padding)
            unpacked_output[b, :seq_len] = output_tensor[
                start_idx : start_idx + seq_len
            ]
        else:
            # No CP, use regular cu_seqlens
            start_idx = cu_seqlens[b].item()
            end_idx = cu_seqlens[b + 1].item()

            # Copy the valid tokens to the unpacked tensor
            unpacked_output[b, :seq_len] = output_tensor[start_idx:end_idx]

    return unpacked_output


def broadcast_tensor(
    tensor: torch.Tensor | None, src_rank: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Broadcasts a tensor from src_rank to all ranks in the group using broadcast_object_list for metadata.

    Handles the case where the input tensor might be None on non-source ranks.
    If the input tensor is provided on non-source ranks, it must have the
    correct shape and dtype matching the tensor on the source rank.

    Args:
        tensor: The tensor to broadcast on the source rank. Can be None on
                non-source ranks (will be created with correct shape/dtype).
                If not None on non-source ranks, it's used as the buffer
                for the broadcast and must match the source tensor's metadata.
        src_rank (int): The global rank of the source process.
        group: The process group for communication.

    Returns:
        torch.Tensor: The broadcasted tensor. On non-source ranks, this will
                      be the tensor received from the source.

    Raises:
        ValueError: If the tensor is None on the source rank, or if a tensor
                    provided on a non-source rank has mismatched shape/dtype/device.
        TypeError: If broadcasting metadata fails (e.g., due to pickling issues).
    """
    rank = dist.get_rank()
    # Assume operations happen on the default CUDA device for the rank
    # TODO: Consider making device explicit if needed, e.g., derive from tensor on src
    device = torch.cuda.current_device()

    # 1. Broadcast metadata (shape and dtype) using broadcast_object_list
    if rank == src_rank:
        if tensor is None:
            raise ValueError(f"Rank {rank} is source ({src_rank}) but tensor is None.")
        # Package metadata into a list containing shape and dtype
        metadata = [tensor.shape, tensor.dtype]
        object_list = [metadata]
    else:
        # Placeholder for receiving the object on non-source ranks
        object_list = [None]

    # Broadcast the list containing the metadata object
    # This relies on the underlying distributed backend supporting object serialization (pickle)
    try:
        dist.broadcast_object_list(object_list, src=src_rank, group=group)
    except Exception as e:
        # Catch potential issues with pickling or backend support
        raise TypeError(
            f"Failed to broadcast tensor metadata using broadcast_object_list: {e}"
        ) from e

    # All ranks now have the metadata in object_list[0]
    received_shape, received_dtype = object_list[0]

    # 2. Prepare tensor buffer on non-source ranks
    if rank != src_rank:
        if tensor is None:
            # Create tensor if it wasn't provided by the caller
            tensor = torch.empty(received_shape, dtype=received_dtype, device=device)
        else:
            # Validate the tensor provided by the caller on the non-source rank
            if tensor.shape != received_shape:
                raise ValueError(
                    f"Rank {rank}: Provided tensor has shape {tensor.shape}, "
                    f"but source rank {src_rank} is broadcasting shape {received_shape}."
                )
            if tensor.dtype != received_dtype:
                raise ValueError(
                    f"Rank {rank}: Provided tensor has dtype {tensor.dtype}, "
                    f"but source rank {src_rank} is broadcasting dtype {received_dtype}."
                )
            # Ensure the provided tensor is on the correct device
            # Compare torch.device objects directly for accuracy
            if tensor.device != torch.device(device):
                raise ValueError(
                    f"Rank {rank}: Provided tensor is on device {tensor.device}, "
                    f"but expected broadcast device is {device}."
                )

    # 3. Broadcast the actual tensor data
    # The tensor object (either original on src, newly created, or validated user-provided on non-src)
    # must exist on all ranks before calling broadcast.
    # `dist.broadcast` operates in-place on the provided tensor object.
    dist.broadcast(tensor, src=src_rank, group=group)

    return tensor

def preprocess_one_batch(
    data_iterator,
    seq_length_key: Optional[str] = None,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_full_seq_to: Optional[int] = None,
    pack_sequences: bool = False,

):
    #with straggler_timer(bdata=True):
    data_dict = next(data_iterator).to("cuda")
    input_ids = data_dict["input_ids"]
    attention_mask = None
    position_ids = None
    packed_seq_params = None

    original_batch_size = input_ids.shape[0]
    original_seq_length = input_ids.shape[1]
    seq_lengths = None  # Will be set if using packed sequences
    cu_seqlens = None
    cu_seqlens_padded = None

    if pack_sequences:
        # For packed sequences with padded input, we need sequence lengths
        assert seq_length_key is not None, (
            "seq_length_key must be provided for packed sequences"
        )
        assert seq_length_key in data_dict, (
            f"{seq_length_key} not found in data_dict"
        )

        # Get sequence lengths and context parallel size
        seq_lengths = data_dict[seq_length_key]

        # Pack sequences
        (
            input_ids,
            input_ids_cp_sharded,
            packed_seq_params,
            cu_seqlens,
            cu_seqlens_padded,
        ) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths,
            pad_individual_seqs_to_multiple_of,
            pad_full_seq_to or 1,
            cp_rank=get_context_parallel_rank(),
            cp_size=get_context_parallel_world_size(),
        )
    
        # For packed sequences, position_ids and attention_mask are typically None
        # The PackedSeqParams handles all necessary sequence information
        position_ids = None
        attention_mask = None
    else:
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
    return (
        data_dict,
        input_ids,
        input_ids_cp_sharded,
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
        position_ids,
        attention_mask,
    )

def forward_step(
    model,
    data_dict,
    cfg,
    input_ids_cp_sharded,
    position_ids,
    attention_mask,
    packed_seq_params,
):
    multimodal_data = data_dict.get_multimodal_dict(
        as_tensors=True, device=input_ids_cp_sharded.device
    )
    if len(multimodal_data) > 0:
        position_ids = None

    additional_kwargs = {}
    # Mamba models currently do not support packed_seq_params
    if packed_seq_params is not None:
        additional_kwargs["packed_seq_params"] = packed_seq_params
    #with straggler_timer:
    output_tensor = model(
        input_ids=input_ids_cp_sharded,
        position_ids=position_ids,
        attention_mask=attention_mask,
        **additional_kwargs,
        **multimodal_data,
    )

    # Apply temperature scaling to logits for training
    # This matches the dtensor worker's _apply_temperature_scaling in the train method
    ## TODO: make this work with current API
    if (
        "generation" in cfg
        and cfg["generation"] is not None
    ):
        output_tensor.div_(cfg["generation"]["temperature"])
        
    return output_tensor

def forward_step_with_collection_fn(
    data_iterator: Iterator[BatchedDataDict[Any]],
    model,
    cfg,
    seq_length_key: Optional[str] = None,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_full_seq_to: Optional[int] = None,
    collection_fn: Optional[Callable[Any, Callable]] = None,
):
    pack_sequences = cfg["sequence_packing"]["enabled"]
    (
        data_dict,
        input_ids,
        input_ids_cp_sharded,
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
        position_ids,
        attention_mask,
    ) = preprocess_one_batch(
        data_iterator,
        seq_length_key,
        pad_individual_seqs_to_multiple_of,
        pad_full_seq_to,
        pack_sequences=pack_sequences,
    )

    output_tensor = forward_step(
        model,
        data_dict,
        cfg,
        input_ids_cp_sharded,
        position_ids,
        attention_mask,
        packed_seq_params,
    )

    ## calling collection_fn will return a function that takes the output tensor and returns a tuple of (loss, metrics)
    #### NOTE: the collection_fn passed in here should take in the following kwargs!
    return output_tensor, collection_fn(
        data_dict=data_dict,
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids_cp_sharded,
        packed_seq_params=packed_seq_params,
        cu_seqlens=cu_seqlens,
        cu_seqlens_padded=cu_seqlens_padded,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

def forward_maybe_backward(
    model,
    cfg,
    data_iterator,
    seq_length_key,
    pad_individual_seqs_to_multiple_of,
    pad_full_seq_to,
    num_microbatches,
    seq_length,
    mbs,
    ## TODO: type hint
    collection_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    forward_only: bool = False,
) -> BatchedDataDict[Any]:
    forward_step = partial(
        forward_step_with_collection_fn,
        cfg=cfg,
        seq_length_key=seq_length_key,
        pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
        pad_full_seq_to=pad_full_seq_to,
        collection_fn=collection_fn,
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
    )

## collection_fn should return a callable that takes the output tensor and returns a tuple of (loss, metrics)
def loss_collection_fn(
    loss_fn: LossFunction, ## TODO: initialize using a partial within train
    cfg: dict,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
    ## the following args depend on the batch
    data_dict: BatchedDataDict[Any],
    packed_seq_params: Optional[PackedSeqParams] = None,
    cp_normalize: bool = True,
    **unused_kwargs,
):
    pack_sequences = cfg["sequence_packing"]["enabled"]

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

    if cp_normalize:
        cp_size = get_context_parallel_world_size()
        orig_loss_fn_wrapped = loss_fn_wrapped

        def _div_by_cp_size(*args, **kwargs):
            loss, metrics = orig_loss_fn_wrapped(*args, **kwargs)
            return loss / cp_size, metrics

        loss_fn_wrapped = _div_by_cp_size
    
    return loss_fn_wrapped

def logprobs_collection_fn(
    cfg: dict,
    ## the following args depend on the batch
    data_dict: BatchedDataDict[Any],
    input_ids,
    cu_seqlens_padded,
    **unused_kwargs,
):
    unpacked_input_ids = data_dict["input_ids"]
    original_seq_length = unpacked_input_ids.shape[1]
    
    def collection_fn_inner(output_tensor):
        stc = time.time()
        tp_grp = get_tensor_model_parallel_group()
        tp_rank = get_tensor_model_parallel_rank()
        logprob_chunk_size = cfg.get("logprob_chunk_size", None)
        if cfg["sequence_packing"]["enabled"]:
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
    return collection_fn_inner


def topk_logits_collection_fn(
    cfg: dict,
    k: int,
    ## arguments that depend on the batch
    data_dict,
    cu_seqlens_padded,
    **additional_kwargs,
):

    pack = cfg["sequence_packing"]["enabled"]
    cp_size = cfg["megatron_cfg"]["context_parallel_size"]
    unpacked_seqlen = data_dict["input_ids"].shape[1]
    seq_lengths = data_dict["input_lengths"]

    def collection_fn_inner(output_tensor):
        # Only the last PP stage produces final logits/top-k; earlier stages return empty
        # if not is_pipeline_last_stage(ignore_virtual=True):
        # return output_tensor.new_zeros(()), {}

        tp_grp = get_tensor_model_parallel_group()
        tp_rank = get_tensor_model_parallel_rank()
        vocab_shard_size = output_tensor.shape[-1]
        vocab_start_index = tp_rank * vocab_shard_size

        chunk_size = None
        if "logprob_chunk_size" in cfg:
            chunk_size = cfg["logprob_chunk_size"]

        topk_vals_local, topk_idx_local = distributed_vocab_topk(
            output_tensor,
            k,
            tp_grp,
            vocab_start_index=vocab_start_index,
            vocab_end_index=vocab_start_index + vocab_shard_size,
            chunk_size=chunk_size,
        )

        if cfg["megatron_cfg"]["context_parallel_size"] > 1:
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
    return collection_fn_inner

def check_sequence_dim(data: BatchedDataDict[Any]):
    # dim 1 is always assumed to be the sequence dim, sanity check this here
    sequence_dim = 1
    seq_dim_size = data["input_ids"].shape[sequence_dim]
    for k, v in data.items():
        if torch.is_tensor(v) and len(v.shape) > 1:
            assert v.shape[sequence_dim] == seq_dim_size, (
                f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
            )
    return sequence_dim, seq_dim_size

def get_mb_iterator(cfg: dict, data: BatchedDataDict[Any], mbs: int):
    sequence_dim, seq_dim_size = check_sequence_dim(data)
    if cfg["dynamic_batching"]["enabled"]:
        mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
        data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
        micro_batch_size = mbs
        ## TODO: handle other args
    elif cfg["sequence_packing"]["enabled"]:
        mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
        data_iterator_len, _ = (
            data.get_microbatch_iterator_for_packable_sequences_len()
        )
        (
            pad_factor,
            pad_packed_seq_to_multiple_of,
            pad_full_seq_to,
        ) = _get_pack_sequence_parameters_for_megatron(
            cfg["megatron_cfg"],
            seq_dim_size,
        )
        micro_batch_size = 1
    else:
        mb_iterator = data.make_microbatch_iterator(mbs)
        data_iterator_len = data.size // mbs
        micro_batch_size = mbs
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
    return (
        mb_iterator,
        data_iterator_len,
        micro_batch_size,
        pad_factor,
        pad_packed_seq_to_multiple_of,
        pad_full_seq_to,
    )

