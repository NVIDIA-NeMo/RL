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
from megatron.core.utils import StragglerDetector
from megatron.training.utils import get_ltor_masks_and_position_ids

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
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
    """

    data_dict: BatchedDataDict[Any]
    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]


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
    delegate_pack_to_model = cfg["sequence_packing"].get(
        "delegate_pack_to_model", False
    )

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
            delegate_pack_to_model=delegate_pack_to_model,
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
    delegate_pack_to_model: bool = False,
    straggler_timer: Optional[StragglerDetector] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[PackedSeqParams],
    Optional[torch.Tensor],
]:
    """Process a microbatch for Megatron model forward pass."""
    ctx = straggler_timer(bdata=True) if straggler_timer is not None else nullcontext()
    with ctx:
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

            if delegate_pack_to_model:
                # VLM path: model (e.g. mbridge Qwen3VL) does its own
                # preprocess_packed_seqs; NeMo-RL must NOT pre-pack + CP-shard,
                # or the double-processing produces shape mismatches downstream
                # (GDN/RoPE/MoE). We only pad each sequence individually and
                # hand the model [B, max_seq] + bool attention_mask + cu_seqlens.
                (
                    input_ids,
                    input_ids_cp_sharded,
                    attention_mask,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _prepare_vlm_batch_for_megatron(
                    input_ids,
                    seq_lengths,
                    pad_individual_seqs_to_multiple_of,
                    pad_full_seq_to=pad_full_seq_to,
                )
                position_ids = None
            else:
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
    return ProcessedInputs(
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids_cp_sharded,
        attention_mask=attention_mask,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
        cu_seqlens_padded=cu_seqlens_padded,
    )


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


def _prepare_vlm_batch_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int,
    pad_full_seq_to: Optional[int] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PackedSeqParams,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare a [B, max_seq] batch for a model that does its own packing + CP sharding.

    Used with mbridge VLM wrappers (e.g. Qwen3VL). The model's forward calls
    preprocess_packed_seqs internally, which re-packs + CP-shards from
    attention_mask. So NeMo-RL must NOT pre-pack / CP-shard; it only:
      * pads each sequence (along dim 1) to pad_individual_seqs_to_multiple_of,
      * builds a bool attention_mask describing real token validity,
      * builds cu_seqlens_padded describing full (pre-shard) packed layout,
      * hands everything to the model as [B, max_seq].

    When ``pad_full_seq_to`` is set (PP>1 requires a constant total packed
    length across microbatches), the last sequence's effective length is
    extended so ``sum(padded_lens) == pad_full_seq_to``. These extra positions
    are treated as "valid" by the model (so mbridge's internal packing stays
    consistent) but should be masked out at the loss layer via token_mask.

    Returns:
        - input_ids: packed [1, T] view for downstream logprob/loss target slicing
        - input_ids_cp_sharded: [B, padded_max_seq] for the model forward
        - attention_mask: [B, padded_max_seq] bool (True for valid tokens)
        - packed_seq_params: PackedSeqParams(qkv_format="thd", cu_seqlens_*=padded)
        - cu_seqlens: None (unpadded cu_seqlens unused in this path)
        - cu_seqlens_padded: [B+1] int32 matching packed_seq_params
    """
    batch_size, _ = input_ids.shape
    device = input_ids.device
    align = max(1, pad_individual_seqs_to_multiple_of)

    # One CPU-GPU sync per call via .tolist(); per-seq arithmetic runs on CPU
    # ints (fast) instead of .item() in a loop (which sync'd per seq).
    if torch.is_tensor(seq_lengths):
        lengths_list = seq_lengths.tolist()
    else:
        lengths_list = list(seq_lengths)
    padded_lens = [((L + align - 1) // align) * align for L in lengths_list]

    # PP>1: force sum(padded_lens) to a fixed value so every microbatch produces
    # the same decoder-side packed length. We mirror _pack_sequences_for_megatron
    # by absorbing the deficit into the LAST sequence's effective length. The
    # extra positions look valid to the model but are zero-ed out at the loss
    # layer via token_mask (consistent with the non-VLM path).
    if pad_full_seq_to is not None and batch_size > 0:
        natural_sum = sum(padded_lens)
        deficit = pad_full_seq_to - natural_sum
        assert deficit >= 0, (
            f"pad_full_seq_to ({pad_full_seq_to}) < natural padded sum "
            f"({natural_sum}); increase pad_full_seq_to."
        )
        assert deficit % align == 0, (
            f"pad_full_seq_to deficit ({deficit}) must be a multiple of "
            f"pad_individual_seqs_to_multiple_of ({align})."
        )
        if deficit > 0:
            lengths_list[-1] += deficit
            padded_lens[-1] += deficit

    padded_max = max(padded_lens) if padded_lens else 0

    # Row-pad input_ids to padded_max so all sequences live in one rectangular tensor.
    if input_ids.shape[1] < padded_max:
        pad_amt = padded_max - input_ids.shape[1]
        input_ids_2d = torch.nn.functional.pad(input_ids, (0, pad_amt), value=0)
    elif input_ids.shape[1] > padded_max:
        input_ids_2d = input_ids[:, :padded_max].contiguous()
    else:
        input_ids_2d = input_ids

    # Vectorised attention_mask: positions < padded length, broadcast over batch.
    # We use padded_lens (not raw lengths) so mbridge's preprocess_packed_seqs,
    # which recomputes seqlens from attention_mask.sum, sees the same packed
    # total as our cu_seqlens_padded. Otherwise a mismatch between raw length
    # and align-padded length leads to GDN's cu_seqlens vs total_seq_len check
    # firing. Tokens in the padded tail are masked out at the loss layer.
    padded_lens_tensor = torch.tensor(padded_lens, dtype=torch.long, device=device)
    positions = torch.arange(padded_max, device=device)
    attention_mask = positions.unsqueeze(0) < padded_lens_tensor.unsqueeze(1)

    # Build cu_seqlens on CPU then H2D once.
    cu_vals = [0]
    for p in padded_lens:
        cu_vals.append(cu_vals[-1] + p)
    cu_seqlens_padded = torch.tensor(cu_vals, dtype=torch.int32, device=device)

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=padded_max,
        max_seqlen_kv=padded_max,
    )

    # Packed (unsharded) view for downstream logprob / loss code that slices
    # per-sequence targets via cu_seqlens_padded. If all sequences are already
    # padded to the same length (common case: input_ids_2d columns == padded_max
    # and every padded_len == padded_max), we can reshape instead of Python-
    # loop + cat, which avoids B separate GPU slice ops.
    if padded_lens and all(p == padded_max for p in padded_lens):
        packed_input_ids = input_ids_2d.reshape(1, -1)
    else:
        packed_segments = [input_ids_2d[i, :p] for i, p in enumerate(padded_lens)]
        packed_input_ids = (
            torch.cat(packed_segments, dim=0).unsqueeze(0)
            if packed_segments
            else input_ids_2d.new_zeros((1, 0))
        )

    # input_ids_cp_sharded keeps the [B, max_seq] layout: the model (mbridge
    # Qwen3VL) runs its own preprocess_packed_seqs to pack + CP-shard.
    # input_ids is the packed (but not CP-sharded) view for target/logprob
    # post-processing, which uses cu_seqlens_padded to slice per sequence.
    return (
        packed_input_ids,
        input_ids_2d,
        attention_mask,
        packed_seq_params,
        None,
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
