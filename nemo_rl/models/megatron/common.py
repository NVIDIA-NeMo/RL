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
from megatron.training.utils import get_ltor_masks_and_position_ids

from nemo_rl.algorithms.loss_functions import LossFunction, SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.models.megatron.multimodal import (
    collapse_multimodal_tokens,
    is_llava_model,
    prepare_multimodal_data,
)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return (
        ((value + multiple - 1) // multiple * multiple)
        if value % multiple != 0
        else value
    )


def _vlm_sp_repad_collapsed(
    input_ids: torch.Tensor,
    tokens_removed_per_sample: Optional[torch.Tensor],
    tp_size: int,
) -> torch.Tensor:
    """Re-pad collapsed VLM input_ids for sequence parallelism alignment.

    In non-packing mode all samples share the same tensor width. LLaVA
    re-expands image tokens internally, producing:
        combined_embeddings.shape[0] = collapsed_width + max(tokens_removed)
    With sequence_parallel=True, _calc_shard_factor asserts this is
    divisible by tp_size.

    Required invariant: (collapsed_width + max_removed) % tp_size == 0
    Solution: required_width = round_up(collapsed_width + max_removed, tp_size) - max_removed
    """
    if tokens_removed_per_sample is None or tp_size <= 1:
        return input_ids
    max_removed = int(tokens_removed_per_sample.max().item())
    collapsed_width = input_ids.shape[1]
    required_width = _round_up_to_multiple(collapsed_width + max_removed, tp_size) - max_removed
    if required_width > collapsed_width:
        input_ids = torch.nn.functional.pad(
            input_ids, (0, required_width - collapsed_width), value=0
        )
    return input_ids


def _pack_sequences_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_packed_seq_to: Optional[int] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    tokens_removed_per_sample: Optional[torch.Tensor] = None,
    skip_local_cp_sharding: bool = False,
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
        tokens_removed_per_sample: Per-sample count of tokens removed by VLM multimodal
            token collapsing (from collapse_multimodal_tokens). When provided, per-sequence
            padding ensures that expanded lengths (collapsed + removed) are multiples of
            pad_individual_seqs_to_multiple_of. None for non-VLM paths.
        skip_local_cp_sharding: Keep packed_input_ids unsharded because the downstream
            model will apply context parallel sharding after rebuilding embeddings.

    Returns:
        Tuple of:
        - packed_input_ids: Packed input tensor [1, T]
        - input_ids_cp_sharded: Sharded input tensor [cp_size, T // cp_size]
        - packed_seq_params: PackedSeqParams object
        - cu_seqlens: Cumulative sequence lengths
        - cu_seqlens_padded: Padded cumulative sequence lengths
    """
    batch_size = input_ids.shape[0]

    # --- Guard: PP + VLM + CP > 1 is explicitly unsupported ---
    # VLM token expansion causes per-microbatch total length variation that
    # breaks pipeline parallelism's uniform sequence length requirement.
    if tokens_removed_per_sample is not None and pad_packed_seq_to is not None and cp_size > 1:
        raise NotImplementedError(
            "PP > 1 with VLM sequence packing and CP > 1 is not yet supported. "
            "VLM token expansion causes per-microbatch total length variation "
            "that breaks pipeline parallelism's uniform sequence length requirement."
        )

    # --- Input validation ---
    if tokens_removed_per_sample is not None:
        assert tokens_removed_per_sample.shape[0] >= batch_size, (
            f"tokens_removed_per_sample has {tokens_removed_per_sample.shape[0]} entries "
            f"but batch_size is {batch_size}"
        )

    # Build cumulative sequence lengths (cu_seqlens) and extract valid tokens
    needs_padding = (
        pad_individual_seqs_to_multiple_of > 1
        or pad_packed_seq_to_multiple_of > 1
        or pad_packed_seq_to is not None
    )

    cu_seqlens = [0]
    valid_tokens = []

    # Round up the pad_packed_seq_to to the nearest multiple of pad_packed_seq_to_multiple_of
    if pad_packed_seq_to is not None:
        assert pad_packed_seq_to % pad_packed_seq_to_multiple_of == 0, (
            f"pad_packed_seq_to ({pad_packed_seq_to}) is not a multiple of pad_packed_seq_to_multiple_of ({pad_packed_seq_to_multiple_of})."
        )

    pad_factor = pad_individual_seqs_to_multiple_of

    # --- Loop 1: Build cu_seqlens and padded_seq_lens ---
    # padded_seq_lens is shared with Loop 2 to ensure dual-loop consistency
    # (both loops must agree on per-sequence padding).
    padded_seq_lens: list[int] = []

    for b in range(batch_size):
        seq_len = (
            seq_lengths[b].item() if torch.is_tensor(seq_lengths[b]) else seq_lengths[b]
        )

        # Extract valid tokens for this sequence
        valid_tokens.append(input_ids[b, :seq_len])

        # Update cumulative sequence lengths
        cu_seqlens.append(cu_seqlens[-1] + seq_len)

        # Compute padded sequence length
        if needs_padding:
            # VLM-aware padding: ensure expanded length (collapsed + removed)
            # is a multiple of pad_factor. When tokens_removed_per_sample is None,
            # removed=0 and the formula degenerates to the standard case.
            removed = (
                tokens_removed_per_sample[b].item()
                if tokens_removed_per_sample is not None
                else 0
            )
            padded_seq_len = _round_up_to_multiple(seq_len + removed, pad_factor) - removed
            padded_seq_lens.append(padded_seq_len)

    # --- Post-loop: adjust last sequence for PP or FP8 total alignment ---
    if needs_padding and batch_size > 0:
        running = sum(padded_seq_lens[:-1]) if batch_size > 1 else 0

        if pad_packed_seq_to is not None:
            # PP > 1: target collapsed total is pad_packed_seq_to.
            # VLM + CP > 1 is guarded above, so this handles non-VLM or CP=1 VLM.
            padded_seq_lens[-1] = pad_packed_seq_to - running
        elif pad_packed_seq_to_multiple_of > 1:
            if tokens_removed_per_sample is not None:
                # VLM + FP8: align total in expanded space, then derive collapsed padding.
                running_removed = (
                    sum(
                        tokens_removed_per_sample[b].item()
                        for b in range(batch_size - 1)
                    )
                    if batch_size > 1
                    else 0
                )
                running_expanded = running + running_removed
                last_removed = tokens_removed_per_sample[batch_size - 1].item()
                last_expanded = padded_seq_lens[-1] + last_removed  # already per-seq aligned
                total_expanded = _round_up_to_multiple(
                    running_expanded + last_expanded, pad_packed_seq_to_multiple_of
                )
                padded_seq_lens[-1] = total_expanded - running_expanded - last_removed
            else:
                # Non-VLM: current behavior -- align collapsed total.
                current = padded_seq_lens[-1]
                new_total = _round_up_to_multiple(
                    running + current, pad_packed_seq_to_multiple_of
                )
                padded_seq_lens[-1] = new_total - running

    # --- Build cu_seqlens_padded from padded_seq_lens ---
    cu_seqlens_padded = None
    if needs_padding:
        cu_seqlens_padded_list = [0]
        for psl in padded_seq_lens:
            cu_seqlens_padded_list.append(cu_seqlens_padded_list[-1] + psl)
        cu_seqlens_padded = torch.tensor(
            cu_seqlens_padded_list, dtype=torch.int32, device=input_ids.device
        )

    # Convert cu_seqlens to tensor
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=input_ids.device)

    # --- VLM assertions: verify expanded slot alignment ---
    if tokens_removed_per_sample is not None and pad_factor > 1 and needs_padding:
        for b in range(batch_size):
            removed = tokens_removed_per_sample[b].item()
            expanded_slot = padded_seq_lens[b] + removed
            assert expanded_slot % pad_factor == 0, (
                f"[VLM-CP] Expanded slot {b} = {expanded_slot} "
                f"(collapsed_padded={padded_seq_lens[b]}, removed={removed}) "
                f"not aligned to pad_factor={pad_factor}"
            )

    # Calculate max sequence length (padded if using CP)
    if needs_padding:
        seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
        max_seqlen = seq_lens_padded.max().item()
    else:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

    # --- Loop 2: Build padded token tensors ---
    # Uses padded_seq_lens[b] from Loop 1 for consistency (no re-computation).
    if pad_factor > 1:
        all_input_ids = []
        padded_tokens = []
        for b in range(batch_size):
            seq_len = (
                seq_lengths[b].item()
                if torch.is_tensor(seq_lengths[b])
                else seq_lengths[b]
            )

            # Use shared padded_seq_lens from Loop 1
            padded_seq_len = padded_seq_lens[b]

            # Pad this sequence to the required length
            seq_tokens = input_ids[b, :seq_len]
            if padded_seq_len > seq_len:
                # Pad with zeros (or use a padding token if available)
                seq_tokens = torch.nn.functional.pad(
                    seq_tokens, (0, padded_seq_len - seq_len), value=0
                )
            all_input_ids.append(seq_tokens)

            # Skip local CP-sharding when the downstream model will shard after
            # rebuilding its embedding sequence (for example, LLaVA) or when VLM
            # collapsing already moved padding/alignment decisions into expanded space.
            if (
                cp_size > 1
                and not skip_local_cp_sharding
                and tokens_removed_per_sample is None
            ):
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
        original_input_ids = data_dict["input_ids"]
        data_dict = collapse_multimodal_tokens(data_dict, model)
        use_llava_handoff = is_llava_model(model)

        # Extract per-sample token removal counts for VLM expanded↔collapsed conversion
        tokens_removed_per_sample = data_dict.pop("tokens_removed_per_sample", None)

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
                tokens_removed_per_sample=tokens_removed_per_sample,
                skip_local_cp_sharding=use_llava_handoff,
            )

            # --- Phase 2: Pre-set expanded cu_seqlens for VLM attention masking ---
            # When VLM + SP, the model re-expands collapsed image tokens internally
            # (LLaVA's _preprocess_data). The attention mechanism must see expanded
            # boundaries for ALL sequences, not just the last one.
            if tokens_removed_per_sample is not None:
                n_seqs = cu_seqlens_padded.shape[0] - 1
                cumulative_removed = torch.zeros(
                    n_seqs + 1, dtype=torch.int32, device=cu_seqlens_padded.device
                )
                cumulative_removed[1:] = (
                    tokens_removed_per_sample[:n_seqs].to(torch.int32).cumsum(0)
                )
                cu_seqlens_padded_expanded = cu_seqlens_padded.clone() + cumulative_removed
                cu_seqlens_expanded = cu_seqlens.clone() + cumulative_removed

                # Single clone, all four fields alias it (same as original packing).
                # TE relies on identity (cu_seqlens_q is cu_seqlens_kv) for
                # self-attention detection; separate clones break this.
                cu_seqlens_for_attn = cu_seqlens_padded_expanded.clone()
                packed_seq_params.cu_seqlens_q = cu_seqlens_for_attn
                packed_seq_params.cu_seqlens_kv = cu_seqlens_for_attn
                packed_seq_params.cu_seqlens_q_padded = cu_seqlens_for_attn
                packed_seq_params.cu_seqlens_kv_padded = cu_seqlens_for_attn

                # Update max_seqlen to reflect expanded individual sequence lengths
                expanded_slot_lengths = (
                    cu_seqlens_padded_expanded[1:] - cu_seqlens_padded_expanded[:-1]
                )
                packed_seq_params.max_seqlen_q = expanded_slot_lengths.max().item()
                packed_seq_params.max_seqlen_kv = expanded_slot_lengths.max().item()

            # For packed sequences, position_ids and attention_mask are typically None
            # The PackedSeqParams handles all necessary sequence information
            position_ids = None
            attention_mask = None
        else:
            if policy_cfg is not None:
                sp = policy_cfg["megatron_cfg"].get("sequence_parallel", False)
                tp_size = policy_cfg["megatron_cfg"].get("tensor_model_parallel_size", 1)
                if sp and tp_size > 1:
                    input_ids = _vlm_sp_repad_collapsed(
                        input_ids, tokens_removed_per_sample, tp_size
                    )
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

    additional_kwargs = {}
    # Mamba models currently do not support packed_seq_params
    if packed_seq_params is not None:
        additional_kwargs["packed_seq_params"] = packed_seq_params

    if defer_fp32_logits:
        additional_kwargs["fp32_output"] = False

    # LLaVA models must see the full packed input_ids so they can rebuild their
    # embedding sequence and apply CP internally, even for text-only microbatches.
    has_multimodal_payload = len(multimodal_data) > 0
    cp_size = get_context_parallel_world_size()

    # Assert that CP > 1 is only supported for LLaVA models
    if has_multimodal_payload and cp_size > 1:
        assert use_llava_handoff, (
            f"Context parallelism (CP > 1) with VLM is only supported for LLaVA models. "
            f"Got CP size {cp_size} with model type {type(model).__name__}. "
            f"Please set context_parallel_size to 1 for non-LLaVA VLM models."
        )

    model_input_ids = input_ids if use_llava_handoff else input_ids_cp_sharded

    with straggler_timer:
        prepare_multimodal_data(multimodal_data, model, model_input_ids.device)
        output_tensor = model(
            input_ids=model_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **additional_kwargs,
            **multimodal_data,
        )
        if type(output_tensor) == tuple:
            output_tensor = output_tensor[0]

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
        loss_data["input_ids"] = original_input_ids

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
