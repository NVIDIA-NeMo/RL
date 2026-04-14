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

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
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

    # THD batch for CP+seqpack path (bypasses FA2 and DTensor CP)
    thd_batch: Optional["THDBatch"] = None

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
    cp_mesh: Any = None,
    device_mesh: Any = None,
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
        cp_mesh: Optional CP device mesh (needed for seq_packing + CP)
        device_mesh: Full device mesh with "cp" dim (needed for CP+seqpack THD path)

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
            cp_mesh=cp_mesh,
            device_mesh=device_mesh,
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
    cp_mesh: Any = None,
    device_mesh: Any = None,
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
        cp_mesh=cp_mesh,
        device_mesh=device_mesh,
    )
    return processed_iterator, iterator_len


def process_microbatch(
    mb: BatchedDataDict[Any],
    tokenizer: AutoTokenizer,
    enable_seq_packing: bool,
    cfg: dict[str, Any],
    cp_size: int,
    cp_mesh: Any = None,
    device_mesh: Any = None,
) -> ProcessedInputs:
    """Process a microbatch and prepare inputs for model forward.

    Args:
        mb: Microbatch data
        tokenizer: Tokenizer for padding value
        enable_seq_packing: Whether sequence packing is enabled
        cfg: Configuration dictionary
        cp_size: Context parallel size
        cp_mesh: CP device mesh (for CP+seqpack THD path)
        device_mesh: Full device mesh with "cp" dim (for CP+seqpack THD path)

    Returns:
        ProcessedInputs containing all tensors and metadata for forward pass
    """
    input_ids = mb.get("input_ids").cuda()

    if enable_seq_packing and cp_size > 1:
        # CP+seqpack: use THD path with TE CP sharding.
        # pack_for_thd handles CP-padding, THD conversion, and CP sharding.
        # Pass token_mask so prompt tokens get labels=-100 in packed format.
        token_mask = mb.get("token_mask", None)
        thd_result = pack_for_thd(
            input_ids=input_ids,
            input_lengths=mb["input_lengths"],
            packed_sequence_size=[len(mb["input_lengths"])],
            padding_value=tokenizer.eos_token_id,
            min_seq_len=cfg["sequence_packing"]["train_mb_tokens"],
            cp_size=cp_size,
            cp_mesh=cp_mesh,
            device_mesh=device_mesh,
            token_mask=token_mask,
        )
        return ProcessedInputs(
            input_ids=thd_result.input_ids,
            position_ids=thd_result.position_ids,
            attention_mask=None,
            flash_attn_kwargs={},
            seq_len=thd_result.input_ids.shape[0],
            thd_batch=thd_result,
        )
    elif enable_seq_packing:
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
        # CP doesn't support attention_mask — torch's CP SDPA handler requires
        # is_causal=True (no explicit mask). Passing an unsplit mask causes a
        # DTensor redistribution assertion because the mask isn't in cp_buffers
        # and therefore keeps the full sequence length while Q/K/V are split.
        # Matches Automodel's cp_utils.py which does batch.pop("attention_mask").
        attention_mask = None
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


@dataclass
class THDBatch:
    """Packed sequence batch in THD-ready format for custom automodel models.

    Custom models (gpt-oss, qwen3-moe) with TE backend expect individual kwargs
    (cu_seqlens, qkv_format) instead of HF's bundled flash_attn_kwargs.
    """

    input_ids: torch.Tensor  # [packed_len] (1D THD) or [n_rows, packed_len] (2D for PP)
    position_ids: torch.Tensor  # same shape as input_ids
    labels: torch.Tensor  # same shape as input_ids
    cu_seqlens: torch.Tensor  # [num_seqs+1] or [n_rows, max_seqs+1] for PP
    cu_seqlens_per_row: list  # per-row clean cu_seqlens from actual lengths
    cu_seqlens_padded_per_row: list  # per-row clean cu_seqlens from CP-padded lengths
    n_packed_rows: int  # number of packed rows (= n_microbatches for PP)

    # CP-specific fields (set by pack_for_thd when cp_size > 1)
    cp_size: int = 1
    cp_rank: int = 0
    max_seqlen: Optional[torch.Tensor] = None

    def to_model_kwargs(self, device: torch.device) -> dict[str, Any]:
        """Build kwargs dict to pass to schedule.step() or model forward.

        For a single row (non-PP or PP with n_microbatches=1), tensors are 1D
        (THD format). For multiple rows (PP with n_microbatches > 1), tensors
        are 2D [n_rows, packed_len] so the PP schedule can split along dim 0.
        The _thd_squeeze_hook on model parts squeezes dim 0 after splitting.
        """
        result = {
            "input_ids": self.input_ids.to(device),
            "labels": self.labels.to(device),
            "position_ids": self.position_ids.to(device),
            "cu_seqlens": self.cu_seqlens.to(dtype=torch.int32, device=device),
            "qkv_format": "thd",
        }
        # CP size/rank are already configured on the model's attention modules
        # via apply_cp(). Don't pass them as kwargs — it can confuse TE backends.
        if self.max_seqlen is not None:
            result["max_seqlen"] = self.max_seqlen.to(device)
        return result


def install_thd_squeeze_hook(model_parts: list) -> list:
    """Install forward pre-hooks that squeeze cu_seqlens for PP microbatches.

    When the PP schedule splits [n_rows, packed_len] along dim 0, cu_seqlens
    becomes [1, max_seqs+1]. Custom models expect 1D cu_seqlens, so the hook
    squeezes dim 0. Input_ids stays [1, packed_len] — all custom models handle
    this via their internal THD unsqueeze logic.

    Returns list of hook handles for removal.
    """
    handles = []
    for part in model_parts:

        def _squeeze_hook(module, args, kwargs):
            if "cu_seqlens" not in kwargs:
                return args, kwargs
            kwargs = dict(kwargs)
            if kwargs["cu_seqlens"].ndim == 2 and kwargs["cu_seqlens"].shape[0] == 1:
                kwargs["cu_seqlens"] = kwargs["cu_seqlens"].squeeze(0)
            return args, kwargs

        h = part.register_forward_pre_hook(_squeeze_hook, with_kwargs=True)
        handles.append(h)
    return handles


def _cp_pad_length(length: int, cp_size: int) -> int:
    """Pad a sequence length to the nearest multiple of 2*cp_size."""
    divisor = 2 * cp_size
    return ((length + divisor - 1) // divisor) * divisor


def pack_for_thd(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    packed_sequence_size: list[int],
    padding_value: int,
    min_seq_len: int = 0,
    num_chunks: int = 1,
    cp_size: int = 1,
    cp_mesh: Any = None,
    device_mesh: Any = None,
    token_mask: Optional[torch.Tensor] = None,
) -> THDBatch:
    """Pack sequences into THD-format batch, optionally with CP sharding.

    For CP > 1, individual sequences are padded to multiples of ``2*cp_size``
    before packing, and ``make_cp_batch_and_ctx`` shards tokens across CP ranks.
    For PP, ``num_chunks`` controls how many microbatch rows are produced.

    Args:
        input_ids: [num_sequences, max_seq_len] raw input
        input_lengths: [num_sequences] actual lengths
        packed_sequence_size: How many sequences per packed row, e.g. [4, 4]
        padding_value: Pad token id
        min_seq_len: Minimum packed row length
        num_chunks: Number of THD chunks (= n_microbatches for PP). Default 1.
        cp_size: Context parallel size. Default 1.
        cp_mesh: CP device mesh (required when cp_size > 1).
        device_mesh: Full device mesh with "cp" dim (required when cp_size > 1).

    Returns:
        THDBatch with THD-format data, optionally CP-sharded.
    """
    from nemo_automodel.components.distributed.thd_utils import (
        split_batch_into_thd_chunks,
    )

    # When CP > 1, pad each individual sequence to 2*cp_size multiple.
    # This is done by adjusting input_lengths; pack_sequences will trim
    # each sequence to input_lengths[i] tokens, so we use the CP-padded
    # lengths as the actual lengths for packing.
    if cp_size > 1:
        cp_padded_lengths = torch.tensor(
            [_cp_pad_length(l.item(), cp_size) for l in input_lengths],
            dtype=input_lengths.dtype,
            device=input_lengths.device,
        )
    else:
        cp_padded_lengths = input_lengths

    packed_ids, packed_pos_ids, _ = pack_sequences(
        input_ids=input_ids,
        input_lengths=cp_padded_lengths,
        packed_sequence_size=packed_sequence_size,
        padding_value=padding_value,
        return_attention_mask=False,
        min_seq_len=min_seq_len,
    )
    n_rows = len(packed_sequence_size)
    row_len = packed_ids.shape[1]

    # Build seq_lens (actual) and seq_lens_padded (CP-padded per seq, last
    # seq absorbs remaining space to fill row_len — matching automodel's
    # packed_sequence_thd_collater convention).
    seq_lens_list = []
    seq_lens_padded_list = []
    cu_seqlens_per_row = []
    cu_seqlens_padded_per_row = []
    seq_idx = 0
    for row_size in packed_sequence_size:
        row_actual = input_lengths[seq_idx : seq_idx + row_size].clone()
        row_cp_padded = cp_padded_lengths[seq_idx : seq_idx + row_size].clone()

        # Last sequence absorbs remaining space to fill the row
        row_padded_for_thd = row_cp_padded.clone()
        total_cp_padded = int(row_cp_padded.sum().item())
        remaining = row_len - total_cp_padded
        if remaining > 0:
            row_padded_for_thd[-1] = row_padded_for_thd[-1] + remaining

        seq_lens_list.append(row_actual)
        seq_lens_padded_list.append(row_padded_for_thd)

        # Clean cu_seqlens from actual lengths (for data slicing in loss wrapper)
        cu = torch.nn.functional.pad(row_actual.to(torch.int32).cumsum(dim=0), (1, 0))
        cu_seqlens_per_row.append(cu)

        # Clean cu_seqlens from CP-padded lengths (for CP logit slicing)
        cu_padded = torch.nn.functional.pad(
            row_padded_for_thd.to(torch.int32).cumsum(dim=0), (1, 0)
        )
        cu_seqlens_padded_per_row.append(cu_padded)
        seq_idx += row_size

    # Pad to uniform number of sequences per row
    max_seqs = max(len(s) for s in seq_lens_list)
    seq_lens = torch.stack(
        [
            torch.nn.functional.pad(s, (0, max_seqs - len(s)), value=-1000)
            for s in seq_lens_list
        ]
    )
    seq_lens_padded = torch.stack(
        [
            torch.nn.functional.pad(s, (0, max_seqs - len(s)), value=-1000)
            for s in seq_lens_padded_list
        ]
    )

    # Build labels with -100 for non-trainable positions:
    # (a) CP padding between sequences (beyond actual but within CP-padded)
    # (b) End-of-row padding (beyond total valid tokens)
    # (c) Prompt tokens (where token_mask == 0, if provided)
    # This is critical for the CP loss path which uses cross_entropy directly
    # on all tokens (not bounded by cu_seqlens like SequencePackingLossWrapper).
    #
    # Also pack token_mask using the same CP-padded lengths so we can
    # mask prompt tokens in the packed format.
    if token_mask is not None:
        # token_mask: [batch_size, seq_len] with 0=prompt, 1=response.
        # Pack using the same CP-padded lengths to align with packed_ids.
        packed_mask, _, _ = pack_sequences(
            input_ids=token_mask.float(),
            input_lengths=cp_padded_lengths,
            packed_sequence_size=packed_sequence_size,
            padding_value=0,
            return_attention_mask=False,
            min_seq_len=min_seq_len,
        )

    labels = packed_ids.clone()
    for row_idx in range(n_rows):
        row_actual = seq_lens_list[row_idx]
        row_cp_padded = cp_padded_lengths[
            sum(packed_sequence_size[:row_idx]) : sum(
                packed_sequence_size[: row_idx + 1]
            )
        ]
        # Mark padding between each sequence's actual and CP-padded boundary
        pos = 0
        for seq_i in range(len(row_actual)):
            actual = int(row_actual[seq_i].item())
            padded = int(row_cp_padded[seq_i].item())
            if actual < padded:
                labels[row_idx, pos + actual : pos + padded] = -100
            pos += padded
        # Mark end-of-row padding
        if pos < row_len:
            labels[row_idx, pos:] = -100

    # Mask prompt tokens using token_mask
    if token_mask is not None:
        labels[packed_mask < 0.5] = -100

    thd_input = {
        "input_ids": packed_ids,
        "labels": labels,
        "position_ids": packed_pos_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    if cp_size > 1:
        # Use automodel's make_cp_batch_and_ctx for combined THD + CP sharding.
        from nemo_automodel.components.distributed.cp_utils import (
            make_cp_batch_and_ctx,
        )

        _, thd_batch = make_cp_batch_and_ctx(
            device_mesh,
            thd_input,
            use_te=True,
            padding_token_id=padding_value,
            num_chunks=num_chunks,
        )
    else:
        thd_batch = split_batch_into_thd_chunks(
            thd_input,
            num_chunks=num_chunks,
            padding_token_id=padding_value,
        )

    return THDBatch(
        input_ids=thd_batch["input_ids"],
        position_ids=thd_batch["position_ids"],
        labels=thd_batch["labels"],
        cu_seqlens=thd_batch["cu_seqlens"],
        cu_seqlens_per_row=cu_seqlens_per_row,
        cu_seqlens_padded_per_row=cu_seqlens_padded_per_row,
        n_packed_rows=n_rows,
        cp_size=thd_batch.get("cp_size", 1),
        cp_rank=thd_batch.get("cp_rank", 0),
        max_seqlen=thd_batch.get("max_seqlen"),
    )


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
    for _, v in data.items():
        if torch.is_tensor(v) and len(v.shape) > 1:
            assert v.shape[sequence_dim] == seq_dim_size, (
                f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
            )
    return sequence_dim, seq_dim_size
