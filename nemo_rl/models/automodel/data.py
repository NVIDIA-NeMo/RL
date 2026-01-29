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

"""Data processing utilities for automodel training and inference."""

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Tuple

import torch
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.huggingface.common import (
    get_flash_attention_kwargs,
    pack_sequences,
)


@dataclass
class ProcessedInputs:
    """Processed microbatch inputs ready for model forward pass.

    This structure contains all necessary tensors and metadata for a forward pass,
    including context parallel buffers and flash attention configuration.
    """

    # Core inputs (always present)
    input_ids: torch.Tensor
    seq_len: int

    # Optional tensors (None when not applicable)
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None

    # Flash attention configuration
    flash_attn_kwargs: dict[str, Any] = field(default_factory=dict)

    # Multimodal (VLM) inputs
    vlm_kwargs: dict[str, Any] = field(default_factory=dict)

    # Context parallel support (cp_size > 1)
    cp_buffers: list[torch.Tensor] = field(default_factory=list)
    seq_index: Optional[torch.Tensor] = None

    @property
    def has_context_parallel(self) -> bool:
        """Check if context parallel is enabled."""
        return len(self.cp_buffers) > 0

    @property
    def has_flash_attention(self) -> bool:
        """Check if flash attention is configured.

        Works for both empty dict {} and dataclass objects like FlashAttnKwargs.
        """
        return bool(self.flash_attn_kwargs)

    @property
    def is_multimodal(self) -> bool:
        """Check if this is a multimodal input."""
        return len(self.vlm_kwargs) > 0


@dataclass
class ProcessedMicrobatch:
    """Container for a processed microbatch ready for model forward pass.

    This dataclass holds both the original data dictionary and the processed
    tensors needed for the automodel forward pass. It follows the same pattern
    as nemo_rl/models/megatron/data.py ProcessedMicrobatch.

    Attributes:
        data_dict: The original BatchedDataDict containing raw batch data
        processed_inputs: ProcessedInputs containing all tensors for forward pass
        original_batch_size: Original batch size before any packing
        original_seq_len: Original sequence length before any packing
    """

    data_dict: BatchedDataDict[Any]
    processed_inputs: ProcessedInputs
    original_batch_size: int
    original_seq_len: int


def make_processed_microbatch_iterator(
    raw_iterator: Iterator[BatchedDataDict[Any]],
    tokenizer: AutoTokenizer,
    cfg: dict[str, Any],
    cp_size: int,
) -> Iterator[ProcessedMicrobatch]:
    """Wrap a raw microbatch iterator to yield processed microbatches.

    This function takes a raw iterator that yields BatchedDataDict objects and
    wraps it to yield ProcessedMicrobatch objects that contain both the original
    data and the processed tensors ready for model forward pass.

    Args:
        raw_iterator: Iterator yielding raw BatchedDataDict microbatches
        tokenizer: Tokenizer for processing
        cfg: Configuration dictionary (enable_seq_packing is inferred from cfg["sequence_packing"]["enabled"])
        cp_size: Context parallel size

    Yields:
        ProcessedMicrobatch objects containing processed tensors ready for model forward
    """
    # Infer enable_seq_packing from config to mirror mcore pattern
    enable_seq_packing = cfg.get("sequence_packing", {}).get("enabled", False)

    for data_dict in raw_iterator:
        # Store original shapes before processing
        original_batch_size = data_dict.get("input_ids").shape[0]
        original_seq_len = data_dict.get("input_ids").shape[1]

        # Process the microbatch
        processed_inputs = process_microbatch(
            data_dict,
            tokenizer,
            enable_seq_packing,
            cfg,
            cp_size,
        )

        yield ProcessedMicrobatch(
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=original_batch_size,
            original_seq_len=original_seq_len,
        )


def get_microbatch_iterator(
    data: BatchedDataDict[Any],
    cfg: dict[str, Any],
    mbs: int,
    dp_mesh: Any,  # noqa: ARG001
    tokenizer: AutoTokenizer,
    cp_size: int = 1,
) -> tuple[Iterator[ProcessedMicrobatch], int]:
    """Create processed microbatch iterator based on batching strategy.

    Args:
        data: Full dataset to iterate over
        cfg: Configuration dictionary (enable_seq_packing is inferred from cfg["sequence_packing"]["enabled"])
        mbs: Microbatch size
        dp_mesh: Data parallel mesh
        tokenizer: Tokenizer for processing
        cp_size: Context parallel size

    Returns:
        Tuple of (processed_microbatch_iterator, iterator_length)
    """
    # Infer enable_seq_packing from config to mirror mcore pattern
    enable_seq_packing = cfg.get("sequence_packing", {}).get("enabled", False)

    dummy_iterator: Iterator[BatchedDataDict[Any]] = iter([])

    if cfg["dynamic_batching"]["enabled"]:
        mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
        iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
    elif enable_seq_packing:
        mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
        iterator_len, _ = data.get_microbatch_iterator_for_packable_sequences_len()
        max_batch_ct = torch.tensor([iterator_len], device="cuda")
        torch.distributed.all_reduce(max_batch_ct, op=torch.distributed.ReduceOp.MAX)

        # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
        # We add dummy batches to the end of the iterator to make the batch counts equal.
        dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
        dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
        dummy_iterator = itertools.islice(
            itertools.cycle(dummy_iterator), dummy_batch_ct
        )
    else:
        mb_iterator = data.make_microbatch_iterator(mbs)
        iterator_len = data.size // mbs

    # Wrap raw iterators to get processed microbatches
    processed_iterator = make_processed_microbatch_iterator(
        itertools.chain(mb_iterator, dummy_iterator),
        tokenizer,
        cfg,
        cp_size,
    )
    return processed_iterator, iterator_len


def process_microbatch(
    mb: BatchedDataDict[Any],
    tokenizer: AutoTokenizer,
    enable_seq_packing: bool,
    cfg: dict[str, Any],
    cp_size: int,
) -> ProcessedInputs:
    """Process a microbatch and prepare inputs for model forward.

    Args:
        mb: Microbatch data
        tokenizer: Tokenizer for padding value
        enable_seq_packing: Whether sequence packing is enabled
        cfg: Configuration dictionary
        cp_size: Context parallel size

    Returns:
        ProcessedInputs containing all tensors and metadata for forward pass
    """
    input_ids = mb.get("input_ids").cuda()

    if enable_seq_packing:
        input_ids, position_ids, _ = pack_sequences(
            input_ids=input_ids,
            input_lengths=mb["input_lengths"],
            packed_sequence_size=[
                len(mb["input_lengths"])
            ],  # flash attention 2 expects flattened input
            padding_value=tokenizer.eos_token_id,
            return_attention_mask=False,
            min_seq_len=cfg["sequence_packing"][
                "train_mb_tokens"
            ],  # TODO: this is a WAR for sequence packing, we should fix this. Without this, backward will fail when TP is enabled.
        )
        seq_len = input_ids.shape[1]
        attention_mask = None
        flash_attn_kwargs = get_flash_attention_kwargs(
            input_lengths=mb["input_lengths"],
        )
    else:
        batch_size, seq_len = input_ids.shape

        # DTensor requires the causal attention kernel to hit,
        # yet our post_attention_mask (used for masking after forward) is not always all 1s.
        # This is fine because we mask with the actual attention mask later,
        # but for input it has to be all 1s.
        attention_mask = torch.ones(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=input_ids.device,
        )
        # Explicitly create position ids for the input, otherwise the sharding
        # for DTensor will be incorrect.
        position_ids = torch.arange(seq_len, device=input_ids.device).repeat(
            batch_size, 1
        )
        flash_attn_kwargs = {}

    # Add vlm kwargs to model call
    vlm_kwargs = mb.get_multimodal_dict(as_tensors=True, device=input_ids.device)
    if len(vlm_kwargs) > 0:
        # if there are multimodal kwargs, we don't need to add position_ids (computed internally)
        position_ids = None
        assert not enable_seq_packing, (
            "multimodal kwargs are not supported for sequence packing"
        )
        assert not cfg["dtensor_cfg"]["sequence_parallel"], (
            "Sequence parallel is not supported with multimodal since there's an issue when you do not pass position_ids. See https://github.com/NVIDIA-NeMo/Automodel/issues/652"
        )

    # Prepare context parallel buffers if needed
    cp_buffers = []
    seq_index = None
    if cp_size > 1:
        assert len(vlm_kwargs) == 0, (
            f"multimodal kwargs={vlm_kwargs} are not supported for context parallel"
        )
        seq_index = torch.arange(seq_len, device=input_ids.device).repeat(1, 1)
        cp_buffers = [input_ids, position_ids, seq_index]

    return ProcessedInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        flash_attn_kwargs=flash_attn_kwargs,
        vlm_kwargs=vlm_kwargs,
        cp_buffers=cp_buffers,
        seq_index=seq_index,
        seq_len=seq_len,
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
        data: Full dataset
        loss_fn: Loss function (used to check loss type)
        dp_group: Data parallel process group (for consistency with Megatron naming)
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


def check_sequence_dim(data: BatchedDataDict[Any]) -> Tuple[int, int]:
    """Check and validate sequence dimension across all tensors.

    Verifies that dimension 1 is the sequence dimension for all tensors
    in the data dictionary that have more than one dimension.

    Args:
        data: BatchedDataDict to validate

    Returns:
        Tuple of (sequence_dim, seq_dim_size)

    Raises:
        AssertionError: If any tensor has inconsistent sequence dimension
    """
    sequence_dim = 1
    seq_dim_size = data.get("input_ids").shape[sequence_dim]
    for k, v in data.items():
        if torch.is_tensor(v) and len(v.shape) > 1:
            assert v.shape[sequence_dim] == seq_dim_size, (
                f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
            )
    return sequence_dim, seq_dim_size
