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
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)

from nemo_rl.algorithms.loss_functions import LossFunction, SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return (
        ((value + multiple - 1) // multiple * multiple)
        if value % multiple != 0
        else value
    )


def forward_step_arbitrary_loss(
    state: GlobalState,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
    data_iterator: Iterator[BatchedDataDict[Any]],
    model: GPTModel,
    loss_fn: LossFunction,
    pack_sequences: bool = False,
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
