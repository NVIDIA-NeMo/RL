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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from megatron.core.utils import (
    StragglerDetector,
    get_batch_on_this_cp_rank,
    get_batch_on_this_hybrid_cp_rank,
    get_thd_batch_on_this_cp_rank,
)
from megatron.training.utils import get_ltor_masks_and_position_ids

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.common import _round_up_to_multiple


@dataclass
class ProcessedInputs:
    """Processed microbatch inputs used for model forward pass."""

    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    routed_experts: Optional[torch.Tensor] = None
    routed_experts_cp_sharded: Optional[torch.Tensor] = None


@dataclass
class ProcessedMicrobatch:
    """Container for a processed microbatch ready for model forward pass.

    This dataclass holds both the original data dictionary and the processed
    tensors needed for the Megatron model forward pass.

    Attributes:
        data_dict: The original BatchedDataDict containing raw batch data
        input_ids: Processed input token IDs (may be packed for sequence packing)
        input_ids_cp_sharded: Context-parallel sharded input token IDs
        attention_mask: Attention mask tensor (None for packed sequences)
        position_ids: Position IDs tensor (None for packed sequences)
        packed_seq_params: PackedSeqParams for sequence packing (None if not packing)
        cu_seqlens_padded: Padded cumulative sequence lengths (None if not packing)
        routed_experts: Optional token-aligned routed expert ids
        routed_experts_cp_sharded: Context-parallel sharded routed expert ids
    """

    data_dict: BatchedDataDict[Any]
    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    routed_experts: Optional[torch.Tensor] = None
    routed_experts_cp_sharded: Optional[torch.Tensor] = None


def make_processed_microbatch_iterator(
    raw_iterator: Iterator[BatchedDataDict[Any]],
    cfg: dict[str, Any],
    seq_length_key: Optional[str],
    pad_individual_seqs_to_multiple_of: int,
    pad_packed_seq_to_multiple_of: int,
    straggler_timer: StragglerDetector,
    pad_full_seq_to: Optional[int],
) -> Iterator[ProcessedMicrobatch]:
    """Wrap a raw microbatch iterator to yield processed microbatches.

    This function takes a raw iterator that yields BatchedDataDict objects and
    wraps it to yield ProcessedMicrobatch objects that contain both the original
    data and the processed tensors ready for model forward pass.

    Args:
        raw_iterator: Iterator yielding raw BatchedDataDict microbatches
        cfg: Configuration dictionary containing sequence_packing settings
        seq_length_key: Key for sequence length in data dict (required for packing)
        pad_individual_seqs_to_multiple_of: Padding multiple for individual sequences
        pad_packed_seq_to_multiple_of: Padding multiple for packed sequences
        pad_full_seq_to: Target length for full sequence padding (optional)

    Yields:
        ProcessedMicrobatch objects containing processed tensors ready for model forward
    """
    pack_sequences = cfg["sequence_packing"]["enabled"]

    for data_dict in raw_iterator:
        # Move to GPU
        data_dict = data_dict.to("cuda")

        # Process the microbatch
        processed_inputs = process_microbatch(
            data_dict=data_dict,
            seq_length_key=seq_length_key,
            pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
            pad_full_seq_to=pad_full_seq_to,
            pack_sequences=pack_sequences,
            straggler_timer=straggler_timer,
        )

        yield ProcessedMicrobatch(
            data_dict=data_dict,
            input_ids=processed_inputs.input_ids,
            input_ids_cp_sharded=processed_inputs.input_ids_cp_sharded,
            attention_mask=processed_inputs.attention_mask,
            position_ids=processed_inputs.position_ids,
            packed_seq_params=processed_inputs.packed_seq_params,
            cu_seqlens_padded=processed_inputs.cu_seqlens_padded,
            routed_experts=processed_inputs.routed_experts,
            routed_experts_cp_sharded=processed_inputs.routed_experts_cp_sharded,
        )


def get_microbatch_iterator(
    data: BatchedDataDict[Any],
    cfg: dict[str, Any],
    mbs: int,
    straggler_timer: StragglerDetector,
    seq_length_key: Optional[str] = None,
) -> Tuple[Iterator[ProcessedMicrobatch], int, int, int, int]:
    """Create a processed microbatch iterator from a batch of data.

    This function creates an iterator that yields ProcessedMicrobatch objects,
    which contain both the original data dictionary and the processed tensors
    ready for model forward pass.

    Args:
        data: The batch data to create microbatches from
        cfg: Configuration dictionary
        mbs: Microbatch size
        seq_length_key: Key for sequence lengths in data dict (auto-detected if None)

    Returns:
        Tuple containing the iterator and metadata
        - iterator: Iterator yielding ProcessedMicrobatch objects
        - data_iterator_len: Number of microbatches in the iterator
        - micro_batch_size: Size of each microbatch
        - seq_dim_size: Sequence length dimension size
        - padded_seq_length: Padded sequence length for pipeline parallelism (may differ from seq_length)
    """
    micro_batch_size = mbs
    pad_factor = 1
    pad_full_seq_to = None
    pad_packed_seq_to_multiple_of = 1

    _, seq_dim_size = get_and_validate_seqlen(data)

    # Auto-detect seq_length_key if not provided
    if seq_length_key is None and cfg["sequence_packing"]["enabled"]:
        seq_length_key = "input_lengths"

    if cfg["dynamic_batching"]["enabled"]:
        raw_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
        data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
    elif cfg["sequence_packing"]["enabled"]:
        raw_iterator = data.make_microbatch_iterator_for_packable_sequences()
        data_iterator_len, pack_seq_dim_size = (
            data.get_microbatch_iterator_for_packable_sequences_len()
        )
        (
            pad_factor,
            pad_packed_seq_to_multiple_of,
            pad_full_seq_to,
        ) = _get_pack_sequence_parameters_for_megatron(
            cfg["megatron_cfg"],
            cfg["make_sequence_length_divisible_by"],
            pack_seq_dim_size,
        )
        micro_batch_size = 1
    else:
        raw_iterator = data.make_microbatch_iterator(mbs)
        data_iterator_len = data.size // mbs

    # Wrap the raw iterator with processing
    processed_iterator = make_processed_microbatch_iterator(
        raw_iterator=raw_iterator,
        cfg=cfg,
        seq_length_key=seq_length_key,
        pad_individual_seqs_to_multiple_of=pad_factor,
        pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
        pad_full_seq_to=pad_full_seq_to,
        straggler_timer=straggler_timer,
    )

    # Compute padded sequence length for pipeline parallelism
    padded_seq_length = pad_full_seq_to if pad_full_seq_to is not None else seq_dim_size

    return (
        processed_iterator,
        data_iterator_len,
        micro_batch_size,
        seq_dim_size,
        padded_seq_length,
    )


def process_microbatch(
    data_dict: BatchedDataDict[Any],
    seq_length_key: Optional[str] = None,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_full_seq_to: Optional[int] = None,
    pack_sequences: bool = False,
    straggler_timer: Optional[StragglerDetector] = None,
) -> ProcessedInputs:
    """Process a microbatch for Megatron model forward pass."""
    ctx = straggler_timer(bdata=True) if straggler_timer is not None else nullcontext()
    with ctx:
        input_ids = data_dict["input_ids"]
        attention_mask = None
        position_ids = None
        packed_seq_params = None
        routed_experts = (
            data_dict["routed_experts"] if "routed_experts" in data_dict else None
        )
        if routed_experts is not None and routed_experts.dim() != 4:
            raise ValueError(
                "routed_experts must have shape [batch, seq, num_moe_layers, topk] "
                f"before Megatron packing; got {tuple(routed_experts.shape)}"
            )
        routed_experts_cp_sharded = routed_experts

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

            token_aligned_tensors = {"tokens": input_ids}
            if routed_experts is not None:
                token_aligned_tensors["routed_experts"] = routed_experts

            (
                packed_batch,
                cp_packed_batch,
                packed_seq_params,
                cu_seqlens,
                cu_seqlens_padded,
            ) = _pack_token_aligned_batch_for_megatron(
                token_aligned_tensors,
                seq_lengths,
                pad_individual_seqs_to_multiple_of,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
                cp_rank=get_context_parallel_rank(),
                cp_size=get_context_parallel_world_size(),
            )
            input_ids = packed_batch["tokens"]
            input_ids_cp_sharded = cp_packed_batch["tokens"]
            routed_experts = packed_batch.get("routed_experts")
            routed_experts_cp_sharded = cp_packed_batch.get("routed_experts")
            if (
                routed_experts_cp_sharded is not None
                and routed_experts_cp_sharded.dim() != 4
            ):
                raise ValueError(
                    "CP-sharded routed_experts must have shape [1, tokens, "
                    "num_moe_layers, topk] after Megatron packing; got "
                    f"{tuple(routed_experts_cp_sharded.shape)}"
                )

            # For packed sequences, position_ids and attention_mask are typically None
            # The PackedSeqParams handles all necessary sequence information
            position_ids = None
            attention_mask = None
        else:
            if routed_experts is not None:
                if "input_lengths" not in data_dict:
                    raise ValueError(
                        "routed_experts requires input_lengths when sequence packing "
                        "is disabled so padding rows can be repaired before router "
                        "replay."
                    )
                routed_experts = _fill_routed_experts_padding(
                    routed_experts,
                    data_dict["input_lengths"],
                )
                routed_experts_cp_sharded = routed_experts
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
    return ProcessedInputs(
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids_cp_sharded,
        attention_mask=attention_mask,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
        cu_seqlens_padded=cu_seqlens_padded,
        routed_experts=routed_experts,
        routed_experts_cp_sharded=routed_experts_cp_sharded,
    )


def _fill_routed_experts_padding(
    routed_experts: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Replace materialized jagged padding with a valid dummy top-k route."""
    if routed_experts.dim() != 4:
        raise ValueError(
            "routed_experts must have shape [batch, seq, num_moe_layers, topk]; "
            f"got {tuple(routed_experts.shape)}"
        )
    if seq_lengths.shape != (routed_experts.shape[0],):
        raise ValueError(
            "seq_lengths must have one entry per routed_experts row; "
            f"got {tuple(seq_lengths.shape)} for batch={routed_experts.shape[0]}"
        )

    seq_lengths = seq_lengths.to(device=routed_experts.device, dtype=torch.long)
    seq_positions = torch.arange(
        routed_experts.shape[1],
        device=routed_experts.device,
    ).unsqueeze(0)
    padding_mask = seq_positions >= seq_lengths.unsqueeze(1)
    if not bool(padding_mask.any().item()):
        return routed_experts

    repaired = routed_experts.clone()
    default_route = torch.arange(
        routed_experts.shape[-1],
        dtype=routed_experts.dtype,
        device=routed_experts.device,
    ).view(1, 1, 1, routed_experts.shape[-1])
    default_routes = default_route.expand_as(repaired)
    repaired[padding_mask] = default_routes[padding_mask]
    return repaired


def process_global_batch(
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
    dp_group: torch.distributed.ProcessGroup,
    *,
    batch_idx: int,
    batch_size: int,
) -> dict[str, Any]:
    """Process a global batch and compute normalization factors.

    Args:
        data: Full dataset to extract a batch from
        loss_fn: Loss function (used to check loss type for token-level validation)
        dp_group: Data parallel process group for all-reduce
        batch_idx: Index of batch to extract
        batch_size: Size of batch to extract

    Returns:
        Dictionary containing:
        - batch: The extracted batch
        - global_valid_seqs: Number of valid sequences across all ranks
        - global_valid_toks: Number of valid tokens across all ranks
    """
    batch = data.get_batch(batch_idx=batch_idx, batch_size=batch_size)

    assert "sample_mask" in batch, "sample_mask must be present in the data!"

    # Get the normalization factor for the loss
    local_valid_seqs = torch.sum(batch["sample_mask"])

    if "token_mask" not in batch:
        local_valid_toks = local_valid_seqs * batch["input_ids"].shape[1]
    else:
        local_valid_toks = torch.sum(
            batch["token_mask"][:, 1:] * batch["sample_mask"].unsqueeze(-1)
        )

    to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
    torch.distributed.all_reduce(to_reduce, group=dp_group)
    global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

    if hasattr(loss_fn, "loss_type") and loss_fn.loss_type == LossType.TOKEN_LEVEL:
        assert "token_mask" in batch, (
            "token_mask must be present in the data when using token-level loss"
        )

    return {
        "batch": batch,
        "global_valid_seqs": global_valid_seqs,
        "global_valid_toks": global_valid_toks,
    }


def _pack_token_aligned_tensors_for_megatron(
    tensors: dict[str, torch.Tensor],
    seq_lengths: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pack token-aligned ``[B, S, ...]`` tensors into Megatron THD batch form.

    This helper only pads and packs data. CP token ownership is delegated to
    Megatron's THD batch slicing helper so NeMo-RL does not encode CP topology.
    """
    total_tokens = int(cu_seqlens_padded[-1].item())
    packed_tensors = {}
    for key, tensor in tensors.items():
        if tensor.dim() < 2:
            raise ValueError(
                f"Expected token-aligned tensor '{key}' to have shape [B, S, ...], "
                f"got {tensor.shape}"
            )
        if tensor.shape[0] != seq_lengths.shape[0]:
            raise ValueError(
                f"Batch size mismatch for '{key}': tensor has {tensor.shape[0]} rows "
                f"but seq_lengths has {seq_lengths.shape[0]} entries."
            )

        packed_shape = (1, total_tokens, *tensor.shape[2:])
        if key == "routed_experts":
            default_route = torch.arange(
                tensor.shape[-1],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            packed = (
                default_route.view(
                    *([1] * (len(packed_shape) - 1)),
                    tensor.shape[-1],
                )
                .expand(packed_shape)
                .clone()
            )
        else:
            packed = torch.zeros(
                packed_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        for b in range(tensor.shape[0]):
            seq_len = int(seq_lengths[b].item())
            start = int(cu_seqlens_padded[b].item())
            packed[0, start : start + seq_len] = tensor[b, :seq_len]
        packed_tensors[key] = packed

    return packed_tensors


def _slice_batch_for_megatron_context_parallel(
    batch: dict[str, torch.Tensor],
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_padded: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    *,
    local_cp_size: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> tuple[dict[str, torch.Tensor], Optional[PackedSeqParams]]:
    """Delegate CP batch slicing to Megatron using Megatron's dispatch shape.

    NeMo-RL should construct the batch, then let Megatron choose how CP token
    ownership is mapped for dense, packed THD, or hybrid CP inputs.
    """
    if cu_seqlens is None and local_cp_size is None:
        if cp_rank != 0 or cp_size != 1:
            raise ValueError(
                "Dense CP slicing uses Megatron parallel_state. Explicit "
                "cp_rank/cp_size are only supported for packed THD slicing."
            )
        return get_batch_on_this_cp_rank(dict(batch)), None

    if local_cp_size is not None:
        if cp_rank != 0 or cp_size != 1:
            raise ValueError(
                "Hybrid CP slicing uses Megatron process groups. Explicit "
                "cp_rank/cp_size are only supported for packed THD slicing."
            )
        return get_batch_on_this_hybrid_cp_rank(dict(batch), local_cp_size)

    if cu_seqlens_padded is None or max_seqlen is None:
        raise ValueError(
            "Packed THD CP slicing requires cu_seqlens, cu_seqlens_padded, and max_seqlen."
        )

    max_seqlen_t = torch.tensor(
        [max_seqlen],
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    return get_thd_batch_on_this_cp_rank(
        dict(batch),
        cu_seqlens,
        cu_seqlens_padded,
        max_seqlen_t,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )


def _pack_token_aligned_batch_for_megatron(
    token_aligned_tensors: dict[str, torch.Tensor],
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_packed_seq_to: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    PackedSeqParams,
    torch.Tensor,
    torch.Tensor,
]:
    """Pack and CP-slice token-aligned tensors with Megatron's THD topology."""
    if "tokens" not in token_aligned_tensors:
        raise ValueError("token_aligned_tensors must include a 'tokens' entry")

    input_ids = token_aligned_tensors["tokens"]
    batch_size = input_ids.shape[0]

    needs_padding = (
        pad_individual_seqs_to_multiple_of > 1
        or pad_packed_seq_to_multiple_of > 1
        or pad_packed_seq_to is not None
    )

    cu_seqlens = [0]
    cu_seqlens_padded = [0] if needs_padding else None

    if pad_packed_seq_to is not None:
        assert pad_packed_seq_to % pad_packed_seq_to_multiple_of == 0, (
            f"pad_packed_seq_to ({pad_packed_seq_to}) is not a multiple of pad_packed_seq_to_multiple_of ({pad_packed_seq_to_multiple_of})."
        )

    pad_factor = pad_individual_seqs_to_multiple_of

    for b in range(batch_size):
        seq_len = (
            seq_lengths[b].item() if torch.is_tensor(seq_lengths[b]) else seq_lengths[b]
        )
        cu_seqlens.append(cu_seqlens[-1] + seq_len)

        if needs_padding:
            padded_seq_len = _round_up_to_multiple(seq_len, pad_factor)
            cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_seq_len)

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

    if needs_padding:
        seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
        max_seqlen = seq_lens_padded.max().item()
    else:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens.clone()

    packed_batch = _pack_token_aligned_tensors_for_megatron(
        token_aligned_tensors,
        seq_lengths,
        cu_seqlens_padded,
    )
    cp_packed_batch, packed_seq_params = _slice_batch_for_megatron_context_parallel(
        packed_batch,
        cu_seqlens,
        cu_seqlens_padded,
        int(max_seqlen),
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

    return (
        {key: value.contiguous() for key, value in packed_batch.items()},
        {key: value.contiguous() for key, value in cp_packed_batch.items()},
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    )


def _pack_sequences_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_packed_seq_to: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, PackedSeqParams, torch.Tensor, torch.Tensor]:
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

    # Build cumulative sequence lengths (cu_seqlens)
    needs_padding = (
        pad_individual_seqs_to_multiple_of > 1
        or pad_packed_seq_to_multiple_of > 1
        or pad_packed_seq_to is not None
    )

    cu_seqlens = [0]
    cu_seqlens_padded = [0] if needs_padding else None

    if pad_packed_seq_to is not None:
        assert pad_packed_seq_to % pad_packed_seq_to_multiple_of == 0, (
            f"pad_packed_seq_to ({pad_packed_seq_to}) is not a multiple of pad_packed_seq_to_multiple_of ({pad_packed_seq_to_multiple_of})."
        )

    pad_factor = pad_individual_seqs_to_multiple_of

    for b in range(batch_size):
        seq_len = (
            seq_lengths[b].item() if torch.is_tensor(seq_lengths[b]) else seq_lengths[b]
        )

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

    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens.clone()

    packed_batch = _pack_token_aligned_tensors_for_megatron(
        {"tokens": input_ids},
        seq_lengths,
        cu_seqlens_padded,
    )
    all_input_ids = packed_batch["tokens"]
    cp_packed_batch, packed_seq_params = _slice_batch_for_megatron_context_parallel(
        packed_batch,
        cu_seqlens,
        cu_seqlens_padded,
        int(max_seqlen),
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
    packed_input_ids = cp_packed_batch["tokens"]

    return (
        all_input_ids.contiguous(),
        packed_input_ids.contiguous(),
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    )


def _get_pack_sequence_parameters_for_megatron(
    megatron_cfg: dict,
    pad_individual_seqs_to_multiple_of: int,
    max_seq_len_in_batch: int,
):
    """Get pack sequence parameters for Megatron model processing with optional context parallelism.

    Args:
        megatron_cfg: Megatron configuration
        pad_individual_seqs_to_multiple_of: Pad individual sequences to a multiple of this value
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

    # individual sequence needs to be splitted to CP domain, and to TP domain when SP is enabled.
    minimum_pad_factor = 1
    if cp_size > 1:
        minimum_pad_factor *= cp_size * 2
    if tp_size > 1 and sp:
        minimum_pad_factor *= tp_size
    assert pad_individual_seqs_to_multiple_of % minimum_pad_factor == 0, (
        f"make_sequence_length_divisible_by ({pad_individual_seqs_to_multiple_of}) is not a multiple of minimum_pad_factor ({minimum_pad_factor}).\n"
        f"Please set policy.make_sequence_length_divisible_by to a multiple of {minimum_pad_factor}.\n"
        f"    - If CP is enabled, the minimum pad factor is `cp_size * 2`.\n"
        f"    - If TP+SP is enabled, the minimum pad factor is `tp_size`.\n"
        f"    - If both are enabled, the minimum pad factor is `cp_size * 2 * tp_size`."
    )

    # packed sequence length, after splitted to TP and CP domains, needs to be divisible by 128 if using blockwise FP8, and divisible by 16 if using other FP8 recipes.
    if use_fp8:
        divisor = 16
        if fp8_cfg["fp8_recipe"] == "blockwise":
            divisor = 128
        elif fp8_cfg["fp8_recipe"] == "mxfp8":
            divisor = 32
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


def get_and_validate_seqlen(data: BatchedDataDict[Any]):
    # dim 1 is always assumed to be the sequence dim, sanity check this here
    sequence_dim = 1
    seq_dim_size = data["input_ids"].shape[sequence_dim]
    for k, v in data.items():
        if torch.is_tensor(v) and len(v.shape) > 1:
            assert v.shape[sequence_dim] == seq_dim_size, (
                f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape} for key {k}"
            )
    return sequence_dim, seq_dim_size
