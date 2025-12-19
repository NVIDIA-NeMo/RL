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
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from megatron.bridge.training.state import GlobalState
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)
from megatron.training.utils import get_ltor_masks_and_position_ids

from nemo_rl.algorithms.loss_functions import LossFunction, SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank


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
        assert pad_packed_seq_to % pad_packed_seq_to_multiple_of == 0, (
            f"pad_packed_seq_to ({pad_packed_seq_to}) is not a multiple of pad_packed_seq_to_multiple_of ({pad_packed_seq_to_multiple_of})."
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

    # make sure the pad_packed_seq_to is a multiple of the pad_packed_seq_to_multiple_of
    if pad_packed_seq_to is not None:
        pad_packed_seq_to = _round_up_to_multiple(
            pad_packed_seq_to, pad_packed_seq_to_multiple_of
        )

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


def forward_step_arbitrary_loss(
    state: GlobalState,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
    data_iterator: Iterator[BatchedDataDict[Any]],
    model: GPTModel,
    loss_fn: LossFunction,
    pack_sequences: bool = False,
    seq_length_key: Optional[str] = None,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_full_seq_to: Optional[int] = None,
    defer_fp32_logits: Optional[bool] = None,
    cp_normalize: bool = True,
    policy_cfg: Optional[dict] = None,
):
    """Forward training step with support for packed sequences and context parallelism.

    Args:
        state (GlobalState): Global state for the run
        global_valid_seqs: Global count of valid sequences
        global_valid_toks: Global count of valid tokens
        data_iterator: Input data iterator
        model (GPTModel): The GPT Model
        loss_fn (LossFunction): Loss function to apply
        pack_sequences (bool): Whether to pack sequences for efficiency
        seq_length_key (Optional[str]): Key in data_dict containing actual sequence lengths
        pad_individual_seqs_to_multiple_of (int): Pad individual sequences to a multiple of this value
        pad_full_seq_to (Optional[int]): Pad packed sequences to this value
        defer_fp32_logits (Optional[bool]): Whether to skip the conversion of logits to fp32
        cp_normalize (bool): Whether to normalize the loss by the cp_size
        policy_cfg (Optional[dict]): Policy configuration containing generation parameters

    Notes on packed sequences with context parallelism (CP):
        - When CP > 1, each sequence is padded to a multiple of (cp_size * 2)
        - The factor of 2 ensures load balancing for causal attention
        - cu_seqlens tracks actual sequence boundaries
        - cu_seqlens_padded tracks padded sequence boundaries for CP
        - Requires TransformerEngine >= 1.10 for CP support
    """
    straggler_timer = state.straggler_timer

    with straggler_timer(bdata=True):
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
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
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

    with straggler_timer:
        output_tensor = model(
            input_ids=input_ids_cp_sharded,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **additional_kwargs,
            **multimodal_data,
        )

        # Apply temperature scaling to logits for training
        # This matches the dtensor worker's _apply_temperature_scaling in the train method
        if (
            policy_cfg is not None
            and "generation" in policy_cfg
            and policy_cfg["generation"] is not None
        ):
            output_tensor.div_(policy_cfg["generation"]["temperature"])

        # Unpack the output tensor if we did packed sequences
        if pack_sequences and packed_seq_params is not None:
            # remove padding
            loss_fn = SequencePackingLossWrapper(
                loss_fn=loss_fn,
                cu_seqlens_q=packed_seq_params.cu_seqlens_q,
                cu_seqlens_q_padded=packed_seq_params.cu_seqlens_q_padded,
            )

        loss_data = data_dict

    loss_fn_wrapped = partial(
        loss_fn,
        data=loss_data,
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

    return output_tensor, loss_fn_wrapped


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


def get_moe_metrics(
    loss_scale: float,
    total_loss_dict: Optional[dict] = None,
    per_layer_logging: bool = False,
) -> dict[str, Any]:
    """Returns Mixture of Experts (MoE) auxiliary-loss metrics.

    This function reduces MoE auxiliary losses across ranks, aggregates them, and
    returns a dictionary of metrics.

    Args:
        loss_scale: Scale factor to apply to each auxiliary loss (e.g., 1/num_microbatches).
        total_loss_dict: If provided, accumulate means into this dict (by name).
        per_layer_logging: If True, include per-layer values in the returned dict.

    Returns:
        dict[str, Any]: A flat dict of aggregated metrics. For each aux loss name,
        the mean value is returned under the same key (e.g., "load_balancing_loss").
        If per_layer_logging is True, per-layer values are returned under keys of the
        form "moe/{name}_layer_{i}".
    """
    reduce_aux_losses_tracker_across_ranks()
    tracker = get_moe_layer_wise_logging_tracker()

    metrics: dict[str, Any] = {}
    if len(tracker) > 0:
        aux_losses = {k: v["values"].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            # Megatron-LM aggregates aux losses across layers and normalizes by number of MoE layers
            num_layers = int(loss_list.numel()) if loss_list.numel() > 0 else 1
            aggregated_value = loss_list.sum() / num_layers
            metrics[name] = float(aggregated_value.item())
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = aggregated_value
                else:
                    total_loss_dict[name] += aggregated_value

            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    metrics[f"moe/{name}_layer_{i}"] = float(loss)

    clear_aux_losses_tracker()
    return metrics
