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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, TypeVar

import torch
from transformers import AutoConfig

Tensor = TypeVar("Tensor", bound=torch.Tensor)


@dataclass
class FlashAttentionKwargs:
    """Dataclass to hold FlashAttention v2 kwargs."""

    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    max_seqlen_q: int
    max_seqlen_k: int


class ModelFlag(Enum):
    """Enum that defines special flags for model-specific behaviors.

    This enum provides a way to identify models that require special handling or
    configuration in different parts of the NeMo RL codebase.

    Flags:
        SKIP_DTENSOR_TIED_WEIGHTS_CHECK: Models that should skip the tied weights check
                                 for the DTensor Policy even without setting the
                                 NRL_SKIP_TIED_WEIGHT_CHECK flag.
        VLLM_LOAD_FORMAT_AUTO: Models that should use the "auto" load format when initializing
                               VLLM.

    Each flag has a `matches` method that determines if the flag applies to a given model_name.
    """

    SKIP_DTENSOR_TIED_WEIGHTS_CHECK = auto()
    VLLM_LOAD_FORMAT_AUTO = auto()

    def matches(self, model_name: str) -> bool:
        match self:
            case ModelFlag.SKIP_DTENSOR_TIED_WEIGHTS_CHECK:
                return is_gemma_model(model_name)
            case ModelFlag.VLLM_LOAD_FORMAT_AUTO:
                return is_gemma_model(model_name)
            case _:
                raise ValueError(f"Unknown ModelFlag: {self}")


def is_gemma_model(model_name: str) -> bool:
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(hf_config, "model_type") and hf_config.model_type in [
        "gemma2",
        "gemma3",
        "gemma3_text",
    ]


def group_and_cat_tensors(
    tensors: list[torch.Tensor],
    group_sizes: list[int],
    padding_value: int = 0,
    min_seq_len: int = 0,
) -> torch.Tensor:
    """Groups and concatenates tensors according to group_sizes, then pads them to form a 2D tensor.

    Each group of 1D tensors is concatenated into a single 1D tensor, and all resulting
    group tensors are padded to the same length and stacked into a 2D tensor.

    Args:
        tensors: List of 1D tensors of varying lengths.
        group_sizes: List of integers. Each integer specifies how many tensors to group.
        padding_value: Integer used to pad shorter sequences.
        min_seq_len: Minimum sequence length.

    Returns:
        A 2D tensor where each row is a padded concatenation of the grouped tensors.

    Example:
        >>> tensors = [
        ...     torch.tensor([1, 2]),
        ...     torch.tensor([3]),
        ...     torch.tensor([4, 5, 6]),
        ...     torch.tensor([7])
        ... ]
        >>> group_sizes = [2, 2]
        >>> group_and_cat_tensors(tensors, group_sizes, padding_value=-1)
        tensor([[ 1,  2,  3, -1, -1],
                [ 4,  5,  6,  7, -1]])
    """
    grouped = []
    index = 0
    for size in group_sizes:
        group = tensors[index : index + size]
        concat = torch.cat(group, dim=0)
        grouped.append(concat)
        index += size

    # Compute the maximum length for padding
    max_len = max(t.size(0) for t in grouped)
    max_len = max(max_len, min_seq_len)

    # Pad each tensor to max_len
    padded = torch.stack(
        [
            torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=padding_value)
            for t in grouped
        ]
    )

    return padded


def pack_sequences(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    packed_sequence_size: list[int],
    padding_value: int = 0,
    return_attention_mask: bool = True,
    min_seq_len: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Packs sequences into rows where each row concatenates multiple sequences.

    Useful for sequence packing in transformer models (e.g. for SFT training). Returns:
    packed input_ids, packed position_ids, and optional attention_mask.

    Args:
        input_ids (torch.Tensor): Tensor of shape [num_sequences, max_seq_len]
        input_lengths (torch.Tensor): Tensor of shape [num_sequences], containing true lengths
        packed_sequence_size (List[int]): How many sequences to pack per row
        padding_value (int): Pad value for input_ids
        return_attention_mask (bool): Whether to return per-row causal attention mask
        min_seq_len (int): Minimum sequence length.

    Returns:
        Tuple:
            input_ids_packed (torch.Tensor): [batch_size, max_packed_seq_len]
            position_ids_packed (torch.Tensor): [batch_size, max_packed_seq_len]
            attention_mask (Optional[torch.Tensor]): [batch_size, max_len, max_len] if requested

    Example:
        >>> input_ids = torch.tensor([
        ...     [1, 2, 0, 0],   # len 2
        ...     [3, 4, 5, 0],   # len 3
        ...     [6, 0, 0, 0],   # len 1
        ...     [7, 8, 9, 9],   # len 4
        ...     [8, 7, 0, 0],   # len 2
        ...     [6, 0, 0, 0],   # len 1
        ...     [5, 4, 3, 0],   # len 3
        ... ])
        >>> input_lengths = torch.tensor([2, 3, 1, 4, 2, 1, 3])
        >>> packed_sequence_size = [3, 4]
        >>> input_ids_packed, position_ids_packed, attention_mask = pack_sequences(
        ...     input_ids, input_lengths, packed_sequence_size, padding_value=-1, return_attention_mask=True
        ... )
        >>> input_ids_packed
        tensor([
            [ 1,  2,  3,  4,  5,  6, -1, -1, -1, -1],
            [ 7,  8,  9,  9,  8,  7,  6,  5,  4,  3]
        ])
        >>> position_ids_packed
        tensor([
            [0, 1, 0, 1, 2, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0, 1, 0, 0, 1, 2]
        ])
        >>> attention_mask[0]
        tensor([
            [ True,  True, False, False, False, False, False, False, False, False],
            [False, False,  True,  True,  True, False, False, False, False, False],
            [False, False, False, False, False,  True, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
        ])
        >>> attention_mask[1]
        tensor([
            [ True,  True,  True,  True, False, False, False, False, False, False],
            [False, False, False, False,  True,  True,  True, False, False, False],
            [False, False, False, False, False, False,  True,  True,  True,  True],
            [False, False, False, False, False, False, False,  True,  True,  True],
        ])
    """
    flat_input_ids = []
    position_ids = []
    flat_lengths = input_lengths.tolist()

    for i, seq_len in enumerate(flat_lengths):
        flat_input_ids.append(input_ids[i, :seq_len])
        position_ids.append(
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        )

    # Group and pad
    input_ids_packed = group_and_cat_tensors(
        flat_input_ids, packed_sequence_size, padding_value, min_seq_len=min_seq_len
    )
    position_ids_packed = group_and_cat_tensors(
        position_ids, packed_sequence_size, padding_value=0, min_seq_len=min_seq_len
    )

    # Compute max length
    batch_size, max_seq_len = input_ids_packed.shape

    attention_mask = None
    if return_attention_mask:
        attention_mask = torch.zeros(
            (batch_size, max_seq_len, max_seq_len),
            dtype=torch.bool,
            device=input_ids.device,
        )
        index = 0
        for i, group_size in enumerate(packed_sequence_size):
            group_lengths = flat_lengths[index : index + group_size]
            total_len = sum(group_lengths)
            attention_mask[i, :total_len, :total_len] = torch.tril(
                torch.ones(
                    (total_len, total_len), dtype=torch.bool, device=input_ids.device
                )
            )
            index += group_size

    return input_ids_packed, position_ids_packed, attention_mask


# TODO(ahmadki): the function doesn't actually handle returning 2D tensors because none of the backends support this.
#  but we should support this anyways
def unpack_tensor(tensor, input_lengths):
    """Unpacks a packed tensor into individual sequences padded to the same length.

    Args:
        tensor (torch.Tensor): Packed tensor of shape [batch_size, packed_seq_len].
        packed_lengths (List[int]): Original sequence lengths in the order they were packed.

    Returns:
        torch.Tensor: [num_sequences, max_seq_len], each row is one unpacked and padded sequence.

    Example:
        >>> packed_tensor = torch.tensor([
        ...     [1, 2, 3, 4, 5, 6, -1, -1],
        ...     [7, 8, 9, 9, 8, 7, 6, -1]
        ... ])
        >>> packed_lengths = [2, 3, 1, 4, 2]
        >>> unpack_tensor(packed_tensor, packed_lengths)
        tensor([
            [1, 2, 0, 0],
            [3, 4, 5, 0],
            [6, 0, 0, 0],
            [7, 8, 9, 9],
            [8, 7, 0, 0],
        ])
    """
    packed_seqlen = tensor.shape[1]
    splitsizes = input_lengths.tolist()
    splitsizes.append(packed_seqlen - sum(splitsizes))
    tensor_split = torch.split(tensor, tuple(splitsizes), dim=1)

    max_len = max(input_lengths.tolist())  # max sequence length in the batch

    tensor_stacked = []
    for t in tensor_split[0:-1]:
        padding_needed = max_len - t.shape[1]
        tensor_stacked.append(
            torch.nn.functional.pad(
                t, (0, 0, 0, padding_needed), mode="constant", value=0.0
            )
        )
    return torch.cat(tensor_stacked, dim=0)


def get_flash_attention_kwargs(input_lengths: torch.Tensor) -> FlashAttentionKwargs:
    """Returns kwargs required for FlashAttention v2 forward functions.

    Args:
        input_lengths (torch.Tensor): [batch_size] containing lengths of each sequence

    Returns:
        Dict[str, torch.Tensor | int]:
            {
                "cu_seqlens_q": Tensor[int32],
                "cu_seqlens_k": Tensor[int32],
                "max_seqlen_q": int,
                "max_seqlen_k": int
            }
    """
    input_lengths_int32 = input_lengths.to(torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        input_lengths_int32.cumsum(dim=0), (1, 0)
    )  # prepend 0
    max_len = input_lengths.max().item()

    return FlashAttentionKwargs(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),  # same for self-attention
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
    )
