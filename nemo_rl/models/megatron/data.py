# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from math import lcm
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import torch
from megatron.bridge.training.utils.packed_seq_utils import (
    get_packed_seq_cp_partition_indices,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from megatron.core.utils import StragglerDetector

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.data.multimodal_utils import PACKED_MULTIMODAL_FIELDS, PackedTensor
from nemo_rl.data.packing.algorithms import get_packer
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_EXECUTION_SLOT,
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane.schema import OPD_FULL_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.models.megatron.alignment import (
    get_fp8_token_alignment,
    get_parallel_token_alignment,
)
from nemo_rl.models.megatron.common import _round_up_to_multiple
from nemo_rl.models.megatron.hybridep import (
    get_packed_seq_padding_mask,
    pad_packed_seq_for_hybridep,
    uses_hybridep_flex_dispatcher,
)
from nemo_rl.models.policy import (
    MegatronConfig,
    PolicyConfig,
    get_shared_prefix_training_config,
)
from nemo_rl.utils.r3_trace import (
    r3_trace_verify_forward_enabled,
    trace_cp_routed_experts,
)
from nemo_rl.utils.sequence_lengths import to_cpu_int_tuple

if TYPE_CHECKING:
    # megatron.rl's shared-prefix modules exist only in Megatron-LM builds with
    # shared-prefix support; the shared-prefix functions below import them
    # lazily so dense runs work on any Megatron-LM.
    from megatron.rl.shared_prefix_execution import SharedPrefixExecutionUnit
    from megatron.rl.shared_prefix_packing import SharedPrefixRow
    from megatron.rl.shared_prefix_tensors import SharedPrefixTensorBin

SHARED_PREFIX_SOURCE_ROW_INDEX = "_shared_prefix_source_row_index"


@dataclass(frozen=True, slots=True)
class SharedPrefixForwardMetadata:
    """Structured global star metadata retained until model-output fan-out.

    For CP>1, the model input/output sequence is the standard two-chunk zigzag
    shard for ``cp_rank``. ``padding_multiple`` is the resolved physical packing
    contract ``M`` and is a multiple of the TP/CP topology quantum ``Q``.
    ``padded_total_length`` is the minimally ``M``-padded global star length.
    ``tensor_bin`` retains the global physical and logical correctness metadata
    used to route scalar next-token log-probabilities.
    ``exclude_sequence_padding_from_expert_bias`` states the dense path's MoE
    expert-bias convention for per-branch padding rows: packed dense units
    mask their padding exactly when this is true.
    """

    tensor_bin: "SharedPrefixTensorBin"
    source_sequence_length: int
    padding_multiple: int
    cp_rank: int
    cp_size: int
    padded_total_length: int
    exclude_sequence_padding_from_expert_bias: bool


@dataclass
class ProcessedInputs:
    """Processed microbatch inputs used for model forward pass."""

    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    mtp_loss_mask: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    routed_experts: Optional[torch.Tensor] = None
    routed_experts_cp_sharded: Optional[torch.Tensor] = None
    original_seq_length: Optional[int] = None
    media_token_validity_mask: Optional[torch.Tensor] = None


@dataclass
class ProcessedMicrobatch:
    """Container for a processed microbatch ready for model forward pass.

    This dataclass holds both the original data dictionary and the processed
    tensors needed for the Megatron model forward pass.

    Attributes:
        data_dict: The original BatchedDataDict containing raw batch data
        input_ids: Processed input token IDs (may be packed for sequence packing)
        input_ids_cp_sharded: Model-forward token IDs. Usually CP-sharded; models
            that insert media before CP selection receive the full packed THD row.
        attention_mask: Attention mask tensor (None for packed sequences)
        position_ids: Position IDs tensor. None for packed sequences unless the
            model runs MTP, in which case per-sample arange positions are packed
            like input_ids (full THD row for models that CP-slice their own
            inputs, CP-local shard otherwise)
        packed_seq_params: PackedSeqParams for sequence packing (None if not packing)
        cu_seqlens_padded: Padded cumulative sequence lengths (None if not packing)
        mtp_loss_mask: Pre-computed MTP loss mask (token_mask × sample_mask).
            None when MTP is disabled or token/sample masks are absent.
        padding_mask: Packed-sequence padding mask for MoE routing.
        routed_experts: Optional token-aligned routed expert ids
        routed_experts_cp_sharded: Context-parallel sharded routed expert ids
        media_token_validity_mask: Which media-token positions actually anchor a
            projected feature, in the model's own token layout. None when the
            batch needs no correction and the model should derive its own.
        shared_prefix: Shared-prefix star metadata; None for conventional units.
    """

    data_dict: BatchedDataDict[Any]
    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    cu_seqlens_padded: Optional[torch.Tensor]
    mtp_loss_mask: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    routed_experts: Optional[torch.Tensor] = None
    routed_experts_cp_sharded: Optional[torch.Tensor] = None
    original_seq_length: Optional[int] = None
    media_token_validity_mask: Optional[torch.Tensor] = None
    shared_prefix: Optional[SharedPrefixForwardMetadata] = None


def make_processed_microbatch_iterator(
    raw_iterator: Iterator[BatchedDataDict[Any]],
    cfg: PolicyConfig,
    seq_length_key: Optional[str],
    pad_individual_seqs_to_multiple_of: int,
    pad_packed_seq_to_multiple_of: int,
    straggler_timer: StragglerDetector,
    pad_full_seq_to: Optional[int],
    delegate_pack_to_model: bool = False,
    delegate_mtp_loss_mask_to_model: bool = False,
    model_slices_context_parallel_inputs: bool = False,
    create_packed_seq_padding_mask: bool = False,
    prepad_packed_seq_for_hybridep: bool = False,
    mtp_enabled: bool = False,
    shared_prefix_bin_capacity: Optional[int] = None,
    shared_prefix_execution_units: Optional[
        tuple["SharedPrefixExecutionUnit", ...]
    ] = None,
    shared_prefix_stage: Literal["train", "logprobs"] = "train",
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
        create_packed_seq_padding_mask: Whether to mask packed padding from MoE routing
        prepad_packed_seq_for_hybridep: Whether to align packed inputs across the
            HybridEP group before model forward
        mtp_enabled: Whether the model uses multi-token prediction layers.
        shared_prefix_execution_units: The planned shared-prefix forwards for
            the single raw batch this iterator wraps; required with sharing enabled.
        shared_prefix_stage: Consumer stage; logprobs may use evaluation packing
            and omit MTP training-head storage from the token budget.

    Yields:
        ProcessedMicrobatch objects containing processed tensors ready for model forward
    """
    pack_sequences = cfg["sequence_packing"]["enabled"]
    shared_prefix_enabled = get_shared_prefix_training_config(cfg).enabled_for(
        stage=shared_prefix_stage
    )

    for data_dict in raw_iterator:
        # Move to GPU
        data_dict = data_dict.to("cuda")

        if shared_prefix_enabled:
            if (
                shared_prefix_bin_capacity is None
                or shared_prefix_execution_units is None
            ):
                raise ValueError(
                    "shared-prefix train mode requires an explicit per-stage "
                    "token bin capacity and planned execution units"
                )
            yield from process_shared_prefix_microbatch(
                data_dict=data_dict,
                cfg=cfg,
                bin_capacity=shared_prefix_bin_capacity,
                execution_units=shared_prefix_execution_units,
                seq_length_key=seq_length_key,
                pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
                pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                pad_full_seq_to=pad_full_seq_to,
                straggler_timer=straggler_timer,
                mtp_enabled=mtp_enabled,
            )
            continue

        # Process the microbatch
        processed_inputs = process_microbatch(
            data_dict=data_dict,
            seq_length_key=seq_length_key,
            pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
            pad_full_seq_to=pad_full_seq_to,
            pack_sequences=pack_sequences,
            delegate_pack_to_model=delegate_pack_to_model,
            delegate_mtp_loss_mask_to_model=delegate_mtp_loss_mask_to_model,
            model_slices_context_parallel_inputs=model_slices_context_parallel_inputs,
            straggler_timer=straggler_timer,
            create_packed_seq_padding_mask=create_packed_seq_padding_mask,
            prepad_packed_seq_for_hybridep=prepad_packed_seq_for_hybridep,
            mtp_enabled=mtp_enabled,
        )

        yield ProcessedMicrobatch(
            data_dict=data_dict,
            input_ids=processed_inputs.input_ids,
            input_ids_cp_sharded=processed_inputs.input_ids_cp_sharded,
            attention_mask=processed_inputs.attention_mask,
            position_ids=processed_inputs.position_ids,
            packed_seq_params=processed_inputs.packed_seq_params,
            cu_seqlens_padded=processed_inputs.cu_seqlens_padded,
            mtp_loss_mask=processed_inputs.mtp_loss_mask,
            padding_mask=processed_inputs.padding_mask,
            routed_experts=processed_inputs.routed_experts,
            routed_experts_cp_sharded=processed_inputs.routed_experts_cp_sharded,
            original_seq_length=processed_inputs.original_seq_length,
            media_token_validity_mask=processed_inputs.media_token_validity_mask,
        )


def _get_non_packed_sequence_pad_factor(cfg: dict[str, Any]) -> int:
    """Combine user, parallelism, and FP8 alignment for dense batches."""
    megatron_cfg = cfg["megatron_cfg"]
    factor = lcm(
        cfg["make_sequence_length_divisible_by"],
        get_fp8_token_alignment(megatron_cfg),
    )
    cp_size = megatron_cfg["context_parallel_size"]
    if cp_size > 1:
        factor = lcm(factor, cp_size * 2)
    if (
        megatron_cfg["tensor_model_parallel_size"] > 1
        and megatron_cfg["sequence_parallel"]
    ):
        factor = lcm(factor, megatron_cfg["tensor_model_parallel_size"])
    return factor


def _pad_sequence_aligned_tensors(
    data_dict: BatchedDataDict[Any], multiple: int
) -> None:
    """Right-pad every dense sequence tensor in a microbatch in place."""
    if multiple <= 1:
        return
    batch_size, sequence_length = data_dict["input_ids"].shape[:2]
    padded_sequence_length = _round_up_to_multiple(sequence_length, multiple)
    padding = padded_sequence_length - sequence_length
    if padding == 0:
        return
    for key, value in list(data_dict.items()):
        if (
            not torch.is_tensor(value)
            or value.ndim < 2
            or value.shape[0] != batch_size
            or value.shape[1] != sequence_length
        ):
            continue
        pad_shape = list(value.shape)
        pad_shape[1] = padding
        data_dict[key] = torch.cat(
            (value, value.new_zeros(pad_shape)), dim=1
        ).contiguous()


def get_microbatch_iterator(
    data: BatchedDataDict[Any],
    cfg: PolicyConfig,
    mbs: int,
    straggler_timer: StragglerDetector,
    seq_length_key: Optional[str] = None,
    delegate_pack_to_model: bool = False,
    delegate_mtp_loss_mask_to_model: bool = False,
    model_slices_context_parallel_inputs: bool = False,
    mtp_enabled: bool = False,
    shared_prefix_bin_capacity: Optional[int] = None,
    shared_prefix_execution_units: Optional[
        tuple["SharedPrefixExecutionUnit", ...]
    ] = None,
    shared_prefix_stage: Literal["train", "logprobs"] = "train",
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
        mtp_enabled: Whether the model uses multi-token prediction layers.
        shared_prefix_execution_units: The output of
            :func:`plan_shared_prefix_execution_units` for ``data``; required in
            shared-prefix mode. The worker plans once under its own timer and
            reuses the units for the forward-count barrier and the iterator.
        shared_prefix_stage: Consumer stage, independent of model.train()/eval().
            Training APIs retain the train stage even during loss-only evaluation.

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
    create_packed_seq_padding_mask = False
    prepad_packed_seq_for_hybridep = False

    _, seq_dim_size = get_and_validate_seqlen(data)
    max_execution_length = seq_dim_size

    # Auto-detect seq_length_key if not provided
    if seq_length_key is None and cfg["sequence_packing"]["enabled"]:
        seq_length_key = "input_lengths"

    prepacked = "cu_seqlens" in data or "cu_seqlens_padded" in data
    if prepacked and not all(
        key in data for key in ("cu_seqlens", "cu_seqlens_padded")
    ):
        raise ValueError("Prepacked input requires both cumulative boundary fields.")
    if prepacked and (
        not cfg["sequence_packing"]["enabled"]
        or not cfg["sequence_packing"].get("fuse_loss", False)
        or cfg["dynamic_batching"]["enabled"]
    ):
        raise ValueError("Prepacked input requires fused sequence packing only.")
    if not cfg["sequence_packing"]["enabled"]:
        pad_factor = _get_non_packed_sequence_pad_factor(cfg)

    shared_prefix_enabled = get_shared_prefix_training_config(cfg).enabled_for(
        stage=shared_prefix_stage
    )
    shared_prefix_padding_multiple: Optional[int] = None
    if shared_prefix_execution_units is not None and not shared_prefix_enabled:
        raise ValueError(
            "shared_prefix_execution_units are only meaningful in shared-prefix "
            "train mode"
        )
    if shared_prefix_enabled and prepacked:
        raise ValueError(
            "Shared prefix requires unpacked sibling rows, not prepacked inputs"
        )
    if shared_prefix_enabled:
        if shared_prefix_bin_capacity is None or shared_prefix_execution_units is None:
            raise ValueError(
                "shared-prefix train mode requires shared_prefix_bin_capacity and "
                "the shared_prefix_execution_units planned for this batch"
            )
        if cfg["dynamic_batching"]["enabled"]:
            raise NotImplementedError(
                "shared-prefix train mode does not support dynamic batching"
            )
        if not cfg["sequence_packing"]["enabled"]:
            raise ValueError("shared-prefix train mode requires sequence packing")
        model_owned_inputs = [
            name
            for name, enabled in (
                ("delegate_pack_to_model", delegate_pack_to_model),
                ("delegate_mtp_loss_mask_to_model", delegate_mtp_loss_mask_to_model),
                (
                    "model_slices_context_parallel_inputs",
                    model_slices_context_parallel_inputs,
                ),
            )
            if enabled
        ]
        if model_owned_inputs:
            raise NotImplementedError(
                "shared-prefix train mode packs and CP-shards every unit itself "
                f"and does not support {', '.join(model_owned_inputs)}"
            )
        if uses_hybridep_flex_dispatcher(cfg["megatron_cfg"]) and cfg[
            "megatron_cfg"
        ].get("moe_hybridep_prepad_packed_inputs"):
            # Star units are padded only to M, not to the expert-group maximum,
            # and skip the prepad collective, so no shared-mode unit prepads.
            raise NotImplementedError(
                "shared-prefix train mode does not support "
                "moe_hybridep_prepad_packed_inputs"
            )
        if SHARED_PREFIX_SOURCE_ROW_INDEX in data:
            raise ValueError(
                f"input batch contains reserved field {SHARED_PREFIX_SOURCE_ROW_INDEX!r}"
            )
        # Lazy: megatron.rl ships only with shared-prefix Megatron-LM builds.
        from megatron.rl.shared_prefix_execution import (
            SharedPrefixExecutionPlan,
            validate_shared_prefix_execution_units,
        )

        # Units planned on ``data`` address its rows in order. Reject a plan for
        # a different batch, and a star whose rows disagree with its layout:
        # the logprob fan-out follows the layout's row order.
        validate_shared_prefix_execution_units(
            shared_prefix_execution_units,
            batch_size=data.size,
        )
        for unit in shared_prefix_execution_units:
            layout = unit.shared_layout
            if layout is not None and (
                tuple(unit.row_indices) != tuple(layout.row_indices)
                or unit.physical_length != layout.physical_total_length
            ):
                raise ValueError(
                    "precomputed shared-prefix execution unit disagrees with its "
                    f"layout: rows {tuple(unit.row_indices)} vs "
                    f"{tuple(layout.row_indices)}, physical length "
                    f"{unit.physical_length} vs {layout.physical_total_length}"
                )

        # Retain an explicit source-row index so expanded shared/fallback
        # forwards can be restored to the caller's conventional order. A
        # shallow copy keeps the caller's batch untouched.
        working_data = type(data)(dict(_normalize_shared_prefix_group_ids(data)))
        working_data[SHARED_PREFIX_SOURCE_ROW_INDEX] = torch.arange(
            data.size,
            dtype=torch.long,
        )
        *_, shared_prefix_padding_multiple = _resolve_shared_prefix_execution_topology(
            cfg
        )
        raw_iterator = iter((working_data,))
        execution_plan = SharedPrefixExecutionPlan(units=shared_prefix_execution_units)
        data_iterator_len = execution_plan.num_units
        max_execution_length = execution_plan.max_physical_length
        (
            pad_factor,
            pad_packed_seq_to_multiple_of,
            pad_full_seq_to,
        ) = _get_pack_sequence_parameters_for_megatron(
            cast(MegatronConfig, cfg["megatron_cfg"]),
            shared_prefix_padding_multiple,
            max_execution_length,
        )
        micro_batch_size = 1
    elif prepacked:
        create_packed_seq_padding_mask = bool(
            cfg["megatron_cfg"].get("moe_router_enable_expert_bias", False)
        )
        raw_iterator = data.make_microbatch_iterator(1)
        data_iterator_len = data.size
        micro_batch_size = 1
    elif cfg["dynamic_batching"]["enabled"]:
        raw_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
        data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
    elif cfg["sequence_packing"]["enabled"]:
        create_packed_seq_padding_mask = uses_hybridep_flex_dispatcher(
            cfg["megatron_cfg"]
        )
        prepad_packed_seq_for_hybridep = create_packed_seq_padding_mask and cfg[
            "megatron_cfg"
        ].get("moe_hybridep_prepad_packed_inputs")
        raw_iterator = data.make_microbatch_iterator_for_packable_sequences()
        data_iterator_len, pack_seq_dim_size = (
            data.get_microbatch_iterator_for_packable_sequences_len()
        )
        (
            pad_factor,
            pad_packed_seq_to_multiple_of,
            pad_full_seq_to,
        ) = _get_pack_sequence_parameters_for_megatron(
            cast(MegatronConfig, cfg["megatron_cfg"]),
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
        create_packed_seq_padding_mask=create_packed_seq_padding_mask,
        prepad_packed_seq_for_hybridep=prepad_packed_seq_for_hybridep,
        delegate_pack_to_model=delegate_pack_to_model,
        delegate_mtp_loss_mask_to_model=delegate_mtp_loss_mask_to_model,
        model_slices_context_parallel_inputs=model_slices_context_parallel_inputs,
        mtp_enabled=mtp_enabled,
        shared_prefix_bin_capacity=shared_prefix_bin_capacity,
        shared_prefix_execution_units=shared_prefix_execution_units,
        shared_prefix_stage=shared_prefix_stage,
    )

    # Compute padded sequence length for pipeline parallelism
    if shared_prefix_enabled:
        assert shared_prefix_padding_multiple is not None
        padded_seq_length = (
            pad_full_seq_to
            if pad_full_seq_to is not None
            else _round_up_to_multiple(
                max_execution_length,
                shared_prefix_padding_multiple,
            )
        )
    else:
        padded_seq_length = (
            pad_full_seq_to
            if pad_full_seq_to is not None
            else _round_up_to_multiple(seq_dim_size, pad_factor)
        )

    return (
        processed_iterator,
        data_iterator_len,
        micro_batch_size,
        seq_dim_size,
        padded_seq_length,
    )


def get_ltor_masks_and_position_ids(*args: Any, **kwargs: Any) -> Any:
    """Lazy proxy for `megatron.training.utils.get_ltor_masks_and_position_ids`.

    The underlying import is deferred to call time so that importing this module does
    not pull in `megatron.training` -> modelopt -> transformers -> torchvision, which
    can crash on a duplicate torchvision ``roi_align` meta-kernel registration in the mcore venv.
    """
    from megatron.training.utils import get_ltor_masks_and_position_ids as _impl

    return _impl(*args, **kwargs)


def _prepacked_boundary(
    data: BatchedDataDict[Any], key: str, device: torch.device
) -> torch.Tensor:
    value = data[key]
    if isinstance(value, PackedTensor):
        value = value.as_tensor()
    elif isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{key} must describe one physical pack.")
        value = value[0]
    elif torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional tensor.")
    return value.to(device=device, dtype=torch.int32)


def _slice_prepacked_for_cp(value: torch.Tensor, padded: torch.Tensor) -> torch.Tensor:
    """Apply Megatron's per-source zigzag CP slicing to a packed row."""
    if value.ndim < 2 or value.shape[:2] != (1, int(padded[-1])):
        raise ValueError(
            "Prepacked token-aligned tensors must have shape [1, pack length, ...]."
        )
    cp_rank = get_context_parallel_rank()
    cp_size = get_context_parallel_world_size()
    return torch.cat(
        [
            _get_tokens_on_this_cp_rank(
                value[:, int(start) : int(end)], cp_rank, cp_size, seq_dim=1
            )
            for start, end in zip(padded[:-1], padded[1:])
        ],
        dim=1,
    ).contiguous()


def _prepare_prepacked(
    data: BatchedDataDict[Any],
    *,
    model_slices_context_parallel_inputs: bool,
    create_padding_mask: bool = False,
    mtp_enabled: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    PackedSeqParams,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    input_ids = data["input_ids"]
    if not torch.is_tensor(input_ids) or input_ids.shape[0] != 1:
        raise ValueError("Prepacked input_ids must contain one physical row.")
    cu = _prepacked_boundary(data, "cu_seqlens", input_ids.device)
    padded = _prepacked_boundary(data, "cu_seqlens_padded", input_ids.device)
    source_lengths = cu[1:] - cu[:-1]
    padded_lengths = padded[1:] - padded[:-1]
    pack_length = int(padded[-1])
    if (
        cu.shape != padded.shape
        or cu.numel() < 2
        or int(cu[0]) != 0
        or int(padded[0]) != 0
        or pack_length > input_ids.shape[1]
        or bool((source_lengths <= 0).any())
        or bool((source_lengths > padded_lengths).any())
    ):
        raise ValueError("Invalid prepacked source boundaries.")
    batch_size, sequence_length = input_ids.shape[:2]
    for key, value in list(data.items()):
        if (
            key in {"cu_seqlens", "cu_seqlens_padded"}
            or not torch.is_tensor(value)
            or value.ndim < 2
            or value.shape[0] != batch_size
            or value.shape[1] != sequence_length
        ):
            continue
        data[key] = value[:, :pack_length].contiguous()
    input_ids = data["input_ids"]
    cp_size = get_context_parallel_world_size()
    if cp_size > 1 and bool((padded_lengths % (2 * cp_size) != 0).any()):
        raise ValueError(
            "Every prepacked padded source length must be divisible by 2 * "
            f"context_parallel_size ({2 * cp_size})."
        )
    local_input_ids = _slice_prepacked_for_cp(input_ids, padded)
    input_ids_cp_sharded = (
        input_ids if model_slices_context_parallel_inputs else local_input_ids
    )
    padding_mask = None
    if create_padding_mask:
        full_padding_mask = get_packed_seq_padding_mask(
            cu_seqlens=cu,
            cu_seqlens_padded=padded,
            total_tokens=pack_length,
        )
        padding_mask = (
            full_padding_mask
            if model_slices_context_parallel_inputs
            else _slice_prepacked_for_cp(full_padding_mask, padded)
        )
    position_ids = None
    if mtp_enabled:
        full_position_ids = torch.zeros_like(input_ids)
        for physical_start, source_length in zip(padded[:-1], source_lengths):
            source_length = int(source_length)
            physical_start = int(physical_start)
            full_position_ids[:, physical_start : physical_start + source_length] = (
                torch.arange(source_length, device=input_ids.device)
            )
        position_ids = (
            full_position_ids
            if model_slices_context_parallel_inputs
            else _slice_prepacked_for_cp(full_position_ids, padded)
        )
    params = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=int(padded_lengths.max()),
        max_seqlen_kv=int(padded_lengths.max()),
        pad_between_seqs=not torch.equal(cu, padded),
        qkv_format="thd",
        total_tokens=input_ids_cp_sharded.shape[1],
    )
    return (
        input_ids,
        input_ids_cp_sharded,
        params,
        padded,
        padding_mask,
        position_ids,
    )


def _normalize_shared_prefix_group_ids(
    data_dict: BatchedDataDict[Any],
) -> BatchedDataDict[Any]:
    """Return a list-backed group field suitable for BatchedDataDict slicing.

    TransferQueue deliberately materializes non-tensor columns as one-dimensional
    object arrays, while ``BatchedDataDict.select_indices`` only supports tensors,
    packed tensors, and lists. Normalize only the opted-in group field and leave
    the caller's batch untouched.
    """
    if SHARED_PREFIX_GROUP_ID not in data_dict:
        raise ValueError(
            f"shared-prefix train mode requires batch field {SHARED_PREFIX_GROUP_ID!r}"
        )
    group_ids = data_dict[SHARED_PREFIX_GROUP_ID]
    if isinstance(group_ids, list):
        return data_dict
    if isinstance(group_ids, np.ndarray):
        if group_ids.dtype != object or group_ids.ndim != 1:
            raise TypeError(
                "shared-prefix group IDs from TransferQueue must be a "
                "one-dimensional numpy object array"
            )
        normalized_group_ids = group_ids.tolist()
    elif isinstance(group_ids, tuple):
        normalized_group_ids = list(group_ids)
    else:
        raise TypeError(
            "shared-prefix group IDs must be a list, tuple, or one-dimensional "
            f"numpy object array, got {type(group_ids).__name__}"
        )
    if len(normalized_group_ids) != data_dict.size:
        raise ValueError("shared-prefix group IDs must have one entry per input row")
    normalized = type(data_dict)(dict(data_dict.items()))
    normalized[SHARED_PREFIX_GROUP_ID] = normalized_group_ids
    return normalized


def _build_shared_prefix_rows(
    data_dict: BatchedDataDict[Any],
) -> list["SharedPrefixRow"]:
    """Build validated CPU planner rows from conventional batch metadata."""
    # Lazy: megatron.rl ships only with shared-prefix Megatron-LM builds.
    from megatron.rl.shared_prefix_tensors import build_shared_prefix_rows

    required_fields = (
        "input_ids",
        "input_lengths",
        SHARED_PREFIX_PROMPT_LENGTHS,
        SHARED_PREFIX_GROUP_ID,
    )
    missing_fields = [field for field in required_fields if field not in data_dict]
    if missing_fields:
        raise ValueError(
            "shared-prefix train mode requires batch fields "
            f"{required_fields}; missing {tuple(missing_fields)}"
        )

    input_ids = data_dict["input_ids"]
    input_lengths = data_dict["input_lengths"]
    prompt_lengths = data_dict[SHARED_PREFIX_PROMPT_LENGTHS]
    group_ids = data_dict[SHARED_PREFIX_GROUP_ID]
    return build_shared_prefix_rows(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
    )


def _resolve_shared_prefix_execution_topology(
    cfg: PolicyConfig,
) -> tuple[int, int, int]:
    """Resolve the shared-prefix TP size, CP size and physical packing multiple.

    The topology must obey the shared-prefix SP contract, and
    ``make_sequence_length_divisible_by`` must be a multiple of its quantum;
    :func:`validate_shared_prefix_training_config` already enforces both.
    """
    # Lazy: megatron.rl ships only with shared-prefix Megatron-LM builds.
    from megatron.rl.shared_prefix_tensors import (
        resolve_shared_prefix_parallel_topology,
        resolve_shared_prefix_physical_padding_multiple,
    )

    megatron_cfg = cast(MegatronConfig, cfg["megatron_cfg"])
    tp_size, cp_size, _ = resolve_shared_prefix_parallel_topology(
        tp_size=megatron_cfg["tensor_model_parallel_size"],
        cp_size=megatron_cfg["context_parallel_size"],
        sequence_parallel=megatron_cfg["sequence_parallel"],
    )
    padding_multiple = resolve_shared_prefix_physical_padding_multiple(
        tp_size=tp_size,
        cp_size=cp_size,
        padding_multiple=cfg["make_sequence_length_divisible_by"],
    )
    return tp_size, cp_size, padding_multiple


def _get_prescribed_shared_prefix_slots(
    data_dict: BatchedDataDict[Any],
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic ``(group, slot)`` row sets and validate equal K."""
    # Lazy: megatron.rl ships only with shared-prefix Megatron-LM builds.
    from megatron.rl.shared_prefix_metadata import get_prescribed_shared_prefix_slots

    if SHARED_PREFIX_EXECUTION_SLOT not in data_dict:
        raise ValueError(
            "shared-prefix train mode requires a driver-prescribed execution-slot field"
        )
    raw_slots = data_dict[SHARED_PREFIX_EXECUTION_SLOT]
    if isinstance(raw_slots, torch.Tensor):
        if raw_slots.ndim != 1:
            raise ValueError("shared-prefix execution slots must have shape [batch]")
        slots = [int(value) for value in raw_slots.detach().cpu().tolist()]
    elif isinstance(raw_slots, np.ndarray):
        if raw_slots.ndim != 1:
            raise ValueError("shared-prefix execution slots must have shape [batch]")
        slots = [int(value) for value in raw_slots.tolist()]
    elif isinstance(raw_slots, (list, tuple)):
        slots = [int(value) for value in raw_slots]
    else:
        raise TypeError(
            "shared-prefix execution slots must be a tensor, ndarray, list, "
            f"or tuple, got {type(raw_slots).__name__}"
        )
    return get_prescribed_shared_prefix_slots(
        group_ids=data_dict[SHARED_PREFIX_GROUP_ID], slot_ids=slots
    )


def _make_shared_prefix_dense_packer(
    packing_cfg: Mapping[str, Any], bin_capacity: int
) -> Callable[[Sequence[int]], list[list[int]]]:
    """Return the conventional length packer for shared-prefix dense units.

    Dense units honor ``max_sequences_per_bin`` like ordinary sequence packing.
    Star and forest units are not row-capped: one forward holds every sibling.
    """
    if packing_cfg.get("pair_grouping_key") is not None:
        raise ValueError("shared-prefix train mode does not support pair_grouping_key")

    def pack(lengths: Sequence[int]) -> list[list[int]]:
        return get_packer(
            packing_cfg["algorithm"],
            bin_capacity=bin_capacity,
            max_sequences_per_bin=packing_cfg.get("max_sequences_per_bin"),
        ).pack(list(lengths))

    return pack


def plan_shared_prefix_execution_units(
    data: BatchedDataDict[Any],
    *,
    cfg: PolicyConfig,
    bin_capacity: int,
    forward_only: bool = False,
) -> tuple["SharedPrefixExecutionUnit", ...]:
    """Plan a local batch's shared-prefix forwards once for reuse by the iterator.

    Workers call this before :func:`get_microbatch_iterator` and pass the result
    back through ``shared_prefix_execution_units``, so the CPU planning (one host
    copy of ``input_ids`` plus the exact per-row prompt tuples) runs a single
    time per call. Each driver-prescribed slot resolves to exactly one star,
    forest or conventional unit. Row indices address ``data`` in its given
    order, which :func:`get_microbatch_iterator` preserves. ``forward_only``
    requires model evaluation mode, where shared-prefix MTP is skipped. It
    relaxes only the expanded MTP budget for merging stars; physical capacity
    and dense fallback validation remain unchanged.
    """
    # Lazy: megatron.rl ships only with shared-prefix Megatron-LM builds.
    from megatron.rl.shared_prefix_execution import (
        plan_shared_prefix_execution_units as plan_execution_units,
    )

    data = _normalize_shared_prefix_group_ids(data)
    rows = _build_shared_prefix_rows(data)
    *_, padding_multiple = _resolve_shared_prefix_execution_topology(cfg)
    row_slots = _get_prescribed_shared_prefix_slots(data)
    group_config = get_shared_prefix_training_config(cfg)
    packing_cfg = cfg["sequence_packing"]
    return plan_execution_units(
        rows,
        row_slots=row_slots,
        bin_capacity=bin_capacity,
        padding_multiple=padding_multiple,
        pack_groups=group_config.pack_groups,
        repack_groups=group_config.repack_groups,
        pack_dense_fallbacks=group_config.pack_dense_fallbacks,
        merge_dense_fallbacks=group_config.merge_dense_fallbacks,
        forward_only=forward_only,
        evaluation_packing=group_config.evaluation_packing,
        dense_packer=_make_shared_prefix_dense_packer(packing_cfg, bin_capacity),
        largest_first=packing_cfg.get("microbatch_order") == "largest_first",
    )


def process_shared_prefix_microbatch(
    *,
    data_dict: BatchedDataDict[Any],
    cfg: PolicyConfig,
    bin_capacity: int,
    execution_units: tuple["SharedPrefixExecutionUnit", ...],
    seq_length_key: Optional[str],
    pad_individual_seqs_to_multiple_of: int,
    pad_packed_seq_to_multiple_of: int,
    pad_full_seq_to: Optional[int],
    straggler_timer: Optional[StragglerDetector],
    mtp_enabled: bool = False,
) -> Iterator[ProcessedMicrobatch]:
    """Expand one conventional local batch into star and fallback forwards.

    ``execution_units`` is the plan :func:`plan_shared_prefix_execution_units`
    produced for this exact batch.
    """
    # Lazy: megatron.rl ships only with shared-prefix Megatron-LM builds.
    from megatron.rl.shared_prefix_tensors import (
        materialize_shared_prefix_layout,
        materialize_shared_prefix_token_aligned_tensor,
        shard_shared_prefix_tensor_bin_for_context_parallel,
    )

    data_dict = _normalize_shared_prefix_group_ids(data_dict)
    if data_dict.get_multimodal_dict():
        raise NotImplementedError(
            "shared-prefix train mode does not support multimodal/VLM batches"
        )
    unsupported_fields = [field for field in ("routed_experts",) if field in data_dict]
    if unsupported_fields:
        raise NotImplementedError(
            "shared-prefix train mode does not support batch fields "
            f"{tuple(unsupported_fields)}"
        )
    if seq_length_key != "input_lengths":
        raise ValueError(
            "shared-prefix train mode requires seq_length_key='input_lengths'"
        )

    tp_size, cp_size, padding_multiple = _resolve_shared_prefix_execution_topology(cfg)
    cp_rank = get_context_parallel_rank() if cp_size > 1 else 0
    source_sequence_length = data_dict["input_ids"].shape[1]
    # Packed dense units mask their padding from MoE routing with HybridEP;
    # star units state the same convention to the model.
    mask_sequence_padding = uses_hybridep_flex_dispatcher(cfg["megatron_cfg"])
    for unit in execution_units:
        unit_rows = list(unit.row_indices)
        unit_data = data_dict.select_indices(unit_rows)
        if unit.shared_layout is not None:
            tensor_bin = materialize_shared_prefix_layout(
                data_dict["input_ids"],
                input_lengths=data_dict["input_lengths"],
                layout=unit.shared_layout,
            )
            cp_shard = shard_shared_prefix_tensor_bin_for_context_parallel(
                tensor_bin,
                cp_rank=cp_rank,
                cp_size=cp_size,
                tp_size=tp_size,
                padding_multiple=padding_multiple,
            )
            if cp_shard.padded_total_length > bin_capacity:
                raise RuntimeError(
                    "context-parallel padding makes a shared-prefix star exceed "
                    "its bin capacity: "
                    f"physical_tokens={tensor_bin.layout.physical_total_length}, "
                    f"padded_tokens={cp_shard.padded_total_length}, "
                    f"capacity={bin_capacity}. Configure shared-prefix microbatch "
                    "token capacities as multiples of resolved padding "
                    f"M={padding_multiple}."
                )
            shared_prefix = SharedPrefixForwardMetadata(
                tensor_bin=tensor_bin,
                source_sequence_length=source_sequence_length,
                cp_rank=cp_rank,
                cp_size=cp_size,
                padded_total_length=cp_shard.padded_total_length,
                padding_multiple=padding_multiple,
                exclude_sequence_padding_from_expert_bias=mask_sequence_padding,
            )
            mtp_loss_mask = None
            if "mtp_loss_mask" in data_dict:
                global_mtp_loss_mask = materialize_shared_prefix_token_aligned_tensor(
                    data_dict["mtp_loss_mask"],
                    tensor_bin=tensor_bin,
                    padding_value=0,
                )
                topology_pad = (
                    cp_shard.padded_total_length - global_mtp_loss_mask.shape[0]
                )
                if topology_pad < 0:
                    raise RuntimeError(
                        "shared-prefix MTP mask is longer than the padded model input"
                    )
                if topology_pad:
                    global_mtp_loss_mask = torch.nn.functional.pad(
                        global_mtp_loss_mask,
                        (0, topology_pad),
                        value=0,
                    )
                mtp_loss_mask = global_mtp_loss_mask.index_select(
                    0, cp_shard.global_token_indices
                ).unsqueeze(0)
            yield ProcessedMicrobatch(
                data_dict=unit_data,
                input_ids=unit_data["input_ids"],
                input_ids_cp_sharded=cp_shard.packed_input_ids.unsqueeze(0),
                attention_mask=None,
                position_ids=cp_shard.position_ids.unsqueeze(0),
                packed_seq_params=None,
                cu_seqlens_padded=None,
                mtp_loss_mask=mtp_loss_mask,
                original_seq_length=source_sequence_length,
                shared_prefix=shared_prefix,
            )
            continue

        processed_inputs = process_microbatch(
            data_dict=unit_data,
            seq_length_key=seq_length_key,
            pad_individual_seqs_to_multiple_of=pad_individual_seqs_to_multiple_of,
            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
            pad_full_seq_to=pad_full_seq_to,
            pack_sequences=True,
            straggler_timer=straggler_timer,
            create_packed_seq_padding_mask=mask_sequence_padding,
            mtp_enabled=mtp_enabled,
        )
        yield ProcessedMicrobatch(
            data_dict=unit_data,
            input_ids=processed_inputs.input_ids,
            input_ids_cp_sharded=processed_inputs.input_ids_cp_sharded,
            attention_mask=processed_inputs.attention_mask,
            position_ids=processed_inputs.position_ids,
            packed_seq_params=processed_inputs.packed_seq_params,
            cu_seqlens_padded=processed_inputs.cu_seqlens_padded,
            mtp_loss_mask=processed_inputs.mtp_loss_mask,
            routed_experts=processed_inputs.routed_experts,
            routed_experts_cp_sharded=processed_inputs.routed_experts_cp_sharded,
            media_token_validity_mask=processed_inputs.media_token_validity_mask,
            padding_mask=processed_inputs.padding_mask,
            original_seq_length=processed_inputs.original_seq_length,
            shared_prefix=None,
        )


def process_microbatch(
    data_dict: BatchedDataDict[Any],
    seq_length_key: Optional[str] = None,
    pad_individual_seqs_to_multiple_of: int = 1,
    pad_packed_seq_to_multiple_of: int = 1,
    pad_full_seq_to: Optional[int] = None,
    pack_sequences: bool = False,
    delegate_pack_to_model: bool = False,
    delegate_mtp_loss_mask_to_model: bool = False,
    model_slices_context_parallel_inputs: bool = False,
    straggler_timer: Optional[StragglerDetector] = None,
    create_packed_seq_padding_mask: bool = False,
    prepad_packed_seq_for_hybridep: bool = False,
    mtp_enabled: bool = False,
) -> ProcessedInputs:
    """Process a microbatch for Megatron model forward pass."""
    prepacked = "cu_seqlens" in data_dict
    if (
        create_packed_seq_padding_mask
        and model_slices_context_parallel_inputs
        and not prepacked
    ):
        raise NotImplementedError(
            "HybridEP padding masks are not supported for models that perform "
            "context-parallel input slicing internally."
        )
    if prepad_packed_seq_for_hybridep and delegate_pack_to_model:
        raise NotImplementedError(
            "HybridEP input prepadding requires NeMo-owned sequence packing."
        )
    ctx = straggler_timer(bdata=True) if straggler_timer is not None else nullcontext()
    with ctx:
        input_ids = data_dict["input_ids"]
        original_seq_length = input_ids.shape[1]
        if not pack_sequences:
            _pad_sequence_aligned_tensors(data_dict, pad_individual_seqs_to_multiple_of)
            input_ids = data_dict["input_ids"]
        attention_mask = None
        position_ids = None
        packed_seq_params = None
        routed_experts = (
            data_dict["routed_experts"] if "routed_experts" in data_dict else None
        )
        token_identity_cp_sharded = None
        if routed_experts is not None and routed_experts.dim() != 4:
            raise ValueError(
                "routed_experts must have shape [batch, seq, num_moe_layers, topk] "
                f"before Megatron packing; got {tuple(routed_experts.shape)}"
            )
        routed_experts_cp_sharded = routed_experts

        original_batch_size = input_ids.shape[0]
        seq_lengths = None  # Will be set if using packed sequences
        cu_seqlens = None
        cu_seqlens_padded = None
        mtp_loss_mask = None
        padding_mask = None
        media_token_validity_mask = None

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

            prepacked = "cu_seqlens" in data_dict
            if prepacked:
                if delegate_pack_to_model:
                    raise ValueError("Prepacked input cannot use model-owned packing.")
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens_padded,
                    padding_mask,
                    position_ids,
                ) = _prepare_prepacked(
                    data_dict,
                    model_slices_context_parallel_inputs=(
                        model_slices_context_parallel_inputs
                    ),
                    create_padding_mask=create_packed_seq_padding_mask,
                    mtp_enabled=mtp_enabled,
                )
                original_seq_length = input_ids.shape[1]
                routed_experts = data_dict.get("routed_experts")
                routed_experts_cp_sharded = routed_experts
                if (
                    routed_experts is not None
                    and not model_slices_context_parallel_inputs
                ):
                    routed_experts_cp_sharded = _slice_prepacked_for_cp(
                        routed_experts, cu_seqlens_padded
                    )
                if "mtp_loss_mask" in data_dict:
                    mtp_loss_mask = data_dict["mtp_loss_mask"]
                    if not model_slices_context_parallel_inputs:
                        mtp_loss_mask = _slice_prepacked_for_cp(
                            mtp_loss_mask, cu_seqlens_padded
                        )
                if "media_token_validity_mask" in data_dict:
                    media_token_validity_mask = data_dict["media_token_validity_mask"]
                    if not model_slices_context_parallel_inputs:
                        media_token_validity_mask = _slice_prepacked_for_cp(
                            media_token_validity_mask, cu_seqlens_padded
                        )
                attention_mask = None
            elif delegate_pack_to_model:
                has_mtp_loss_mask = "mtp_loss_mask" in data_dict
                assert not has_mtp_loss_mask or delegate_mtp_loss_mask_to_model, (
                    "MTP training requires a self-packing VLM that advertises "
                    "model_owns_mtp_loss_mask_packing"
                )
                if "media_token_validity_mask" in data_dict:
                    # A self-packing model repacks internally, so a mask built
                    # against caller-side rows would reach the merge in a layout
                    # that no longer matches its tokens -- and a media mask that
                    # is merely misaligned silently attaches features to the
                    # wrong positions rather than failing.
                    raise NotImplementedError(
                        "media_token_validity_mask is not supported for models "
                        "that pack sequences internally (delegate_pack_to_model); "
                        "the mask would need to be packed inside the model."
                    )
                # VLM path: model (e.g. mbridge Qwen3VL) does its own
                # preprocess_packed_seqs; NeMo-RL must NOT pre-pack + CP-shard,
                # or the double-processing produces shape mismatches downstream
                # (GDN/RoPE/MoE). We only pad each sequence individually and
                # hand the model [B, max_seq] + bool attention_mask + cu_seqlens.
                if routed_experts is not None:
                    # Router replay needs routed_experts CP-sharded into the
                    # model's local token order, but a self-packing model packs
                    # and CP-shards internally, so NeMo-RL cannot build a matching
                    # layout here. Fail loudly rather than feed misaligned routes.
                    raise NotImplementedError(
                        "Router replay (routed_experts) is not supported with "
                        "models that pack and context-parallel shard internally "
                        "(delegate_pack_to_model=True)."
                    )
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
                if has_mtp_loss_mask:
                    source_mtp_loss_mask = data_dict["mtp_loss_mask"]
                    assert source_mtp_loss_mask.ndim == 2
                    assert (
                        source_mtp_loss_mask.shape[0] == input_ids_cp_sharded.shape[0]
                    )
                    mtp_loss_mask = source_mtp_loss_mask.new_zeros(
                        input_ids_cp_sharded.shape
                    )
                    copied_length = min(
                        source_mtp_loss_mask.shape[1],
                        input_ids_cp_sharded.shape[1],
                    )
                    mtp_loss_mask[:, :copied_length] = source_mtp_loss_mask[
                        :, :copied_length
                    ]
                    mtp_loss_mask = mtp_loss_mask * attention_mask.to(
                        dtype=mtp_loss_mask.dtype
                    )
                position_ids = None
            else:
                token_identity = None
                if routed_experts is not None and r3_trace_verify_forward_enabled():
                    token_identity = _make_r3_trace_token_identity(
                        input_ids, seq_lengths
                    )

                # Pack sequences on main's per-sequence zigzag CP layout.
                (
                    input_ids,
                    local_input_ids,
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
                if model_slices_context_parallel_inputs:
                    packed_seq_params = PackedSeqParams(
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_kv=cu_seqlens,
                        cu_seqlens_q_padded=cu_seqlens_padded,
                        cu_seqlens_kv_padded=cu_seqlens_padded,
                        max_seqlen_q=int(
                            (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1])
                            .max()
                            .item()
                        ),
                        max_seqlen_kv=int(
                            (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1])
                            .max()
                            .item()
                        ),
                        # TE's default inference excludes the final boundary, so
                        # it misses trailing-only padding for a single sequence.
                        # CP zigzag can move that padding to a rank-local seam.
                        pad_between_seqs=not torch.equal(cu_seqlens, cu_seqlens_padded),
                        qkv_format="thd",
                        total_tokens=input_ids.shape[1],
                    )
                    # This field is the model-forward input. For this capability
                    # the model needs the full THD row so it can insert media
                    # before selecting its CP-owned embeddings.
                    input_ids_cp_sharded = input_ids
                else:
                    input_ids_cp_sharded = local_input_ids
                if create_packed_seq_padding_mask:
                    if prepad_packed_seq_for_hybridep:
                        (
                            input_ids,
                            input_ids_cp_sharded,
                            packed_seq_params,
                            cu_seqlens_padded,
                        ) = pad_packed_seq_for_hybridep(
                            input_ids=input_ids,
                            input_ids_cp_sharded=input_ids_cp_sharded,
                            packed_seq_params=packed_seq_params,
                            cu_seqlens_padded=cu_seqlens_padded,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            cp_rank=get_context_parallel_rank(),
                            cp_size=get_context_parallel_world_size(),
                        )
                    full_padding_mask = get_packed_seq_padding_mask(
                        cu_seqlens=cu_seqlens,
                        cu_seqlens_padded=cu_seqlens_padded,
                        total_tokens=input_ids.shape[1],
                    )
                    if (
                        model_slices_context_parallel_inputs
                        or get_context_parallel_world_size() == 1
                    ):
                        padding_mask = full_padding_mask
                    else:
                        cp_partition_indices = get_packed_seq_cp_partition_indices(
                            packed_seq_params,
                            total_tokens=input_ids.shape[1],
                            cp_size=get_context_parallel_world_size(),
                            cp_rank=get_context_parallel_rank(),
                            device=input_ids.device,
                        )
                        padding_mask = full_padding_mask.index_select(
                            1, cp_partition_indices
                        ).contiguous()
                    assert padding_mask.shape == input_ids_cp_sharded.shape, (
                        f"padding_mask shape {padding_mask.shape} must match "
                        f"model input shape {input_ids_cp_sharded.shape}."
                    )
                # routed_experts and the R3 trace token identity ride the SAME
                # per-seq zigzag CP sharding as input_ids, re-derived from
                # cu_seqlens_padded.
                if routed_experts is not None:
                    (
                        routed_experts,
                        routed_experts_cp_sharded,
                        _token_identity_packed,
                        token_identity_cp_sharded,
                    ) = _shard_routed_experts_for_cp(
                        routed_experts,
                        token_identity,
                        seq_lengths,
                        cu_seqlens,
                        cu_seqlens_padded,
                        get_context_parallel_rank(),
                        get_context_parallel_world_size(),
                    )
                    if model_slices_context_parallel_inputs:
                        cp_partition_indices = get_packed_seq_cp_partition_indices(
                            packed_seq_params,
                            total_tokens=input_ids.shape[1],
                            cp_size=get_context_parallel_world_size(),
                            cp_rank=get_context_parallel_rank(),
                            device=input_ids.device,
                        )
                        routed_experts_cp_sharded = routed_experts.index_select(
                            1, cp_partition_indices
                        ).contiguous()
                        if _token_identity_packed is not None:
                            token_identity_cp_sharded = (
                                _token_identity_packed.index_select(
                                    1, cp_partition_indices
                                ).contiguous()
                            )
                if (
                    routed_experts_cp_sharded is not None
                    and routed_experts_cp_sharded.dim() != 4
                ):
                    raise ValueError(
                        "CP-sharded routed_experts must have shape [1, tokens, "
                        "num_moe_layers, topk] after Megatron packing; got "
                        f"{tuple(routed_experts_cp_sharded.shape)}"
                    )
                verified_token_count = _verify_r3_trace_cp_token_alignment(
                    source_input_ids=data_dict["input_ids"],
                    source_routed_experts=data_dict.get("routed_experts"),
                    input_ids_cp_sharded=(
                        local_input_ids
                        if model_slices_context_parallel_inputs
                        else input_ids_cp_sharded
                    ),
                    routed_experts_cp_sharded=routed_experts_cp_sharded,
                    token_identity_cp_sharded=token_identity_cp_sharded,
                )
                trace_cp_routed_experts(
                    routed_experts_cp_sharded=routed_experts_cp_sharded,
                    token_identity_cp_sharded=token_identity_cp_sharded,
                    input_ids_cp_sharded=(
                        local_input_ids
                        if model_slices_context_parallel_inputs
                        else input_ids_cp_sharded
                    ),
                    cp_token_identity_verified_count=verified_token_count,
                    cp_rank=get_context_parallel_rank(),
                    cp_size=get_context_parallel_world_size(),
                )

                # Pack pre-computed mtp_loss_mask the same way as input_ids
                if "mtp_loss_mask" in data_dict:
                    (
                        packed_mtp_loss_mask,
                        local_mtp_loss_mask,
                        _,
                        _,
                        _,
                    ) = _pack_sequences_for_megatron(
                        data_dict["mtp_loss_mask"],
                        seq_lengths,
                        pad_individual_seqs_to_multiple_of,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                        cp_rank=get_context_parallel_rank(),
                        cp_size=get_context_parallel_world_size(),
                    )
                    # Mirror the input_ids layout choice above. A model that
                    # slices CP itself receives the full THD row so it can insert
                    # media before selecting its CP-owned embeddings, so its MTP
                    # mask has to stay unsharded to line up with the labels.
                    # Every other model consumes the CP-local shard.
                    mtp_loss_mask = (
                        packed_mtp_loss_mask
                        if model_slices_context_parallel_inputs
                        else local_mtp_loss_mask
                    )

                # Pack the media-token validity mask the same way as input_ids.
                # The mask answers a per-token question, so it only means
                # anything while it sits in the same layout as the tokens the
                # model will compare it against. Packing is what destroys the
                # per-sample rows it was built from, so it has to travel through
                # the identical transform rather than be rebuilt afterwards.
                if "media_token_validity_mask" in data_dict:
                    (
                        packed_media_mask,
                        local_media_mask,
                        _,
                        _,
                        _,
                    ) = _pack_sequences_for_megatron(
                        # Pack in the token dtype: padding is filled with 0,
                        # which is a valid token id but not a valid bool. Read
                        # the dtype off the unpacked ids, since the local
                        # input_ids is already the packed tensor here.
                        data_dict["media_token_validity_mask"].to(
                            data_dict["input_ids"].dtype
                        ),
                        seq_lengths,
                        pad_individual_seqs_to_multiple_of,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                        cp_rank=get_context_parallel_rank(),
                        cp_size=get_context_parallel_world_size(),
                    )
                    # Mirror the input_ids layout choice above, for the same
                    # reason the MTP mask does: a model that slices CP itself
                    # merges media against the full THD row.
                    media_token_validity_mask = (
                        packed_media_mask
                        if model_slices_context_parallel_inputs
                        else local_media_mask
                    ).bool()

                # PackedSeqParams carries the sequence layout:
                # attention_mask and position_ids are normally None here.
                # The MTP block is different: it rolls and embeds position_ids per packed segment.
                # Pack per-sample arange positions like input_ids, on every forward.
                attention_mask = None
                if mtp_enabled:
                    position_ids_source = (
                        torch.arange(
                            original_seq_length,
                            dtype=torch.long,
                            device=input_ids.device,
                        )
                        .unsqueeze(0)
                        .expand(original_batch_size, -1)
                    )
                    (
                        packed_position_ids,
                        local_position_ids,
                        _,
                        _,
                        _,
                    ) = _pack_sequences_for_megatron(
                        position_ids_source,
                        seq_lengths,
                        pad_individual_seqs_to_multiple_of,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                        cp_rank=get_context_parallel_rank(),
                        cp_size=get_context_parallel_world_size(),
                    )
                    position_ids = (
                        packed_position_ids
                        if model_slices_context_parallel_inputs
                        else local_position_ids
                    )
                else:
                    position_ids = None
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
                if r3_trace_verify_forward_enabled():
                    token_identity_cp_sharded = _make_r3_trace_token_identity(
                        input_ids,
                        data_dict["input_lengths"],
                    )
            input_ids_cp_sharded = input_ids
            verified_token_count = _verify_r3_trace_cp_token_alignment(
                source_input_ids=data_dict["input_ids"],
                source_routed_experts=data_dict.get("routed_experts"),
                input_ids_cp_sharded=input_ids_cp_sharded,
                routed_experts_cp_sharded=routed_experts_cp_sharded,
                token_identity_cp_sharded=token_identity_cp_sharded,
            )
            trace_cp_routed_experts(
                routed_experts_cp_sharded=routed_experts_cp_sharded,
                token_identity_cp_sharded=token_identity_cp_sharded,
                input_ids_cp_sharded=input_ids_cp_sharded,
                cp_token_identity_verified_count=verified_token_count,
                cp_rank=get_context_parallel_rank(),
                cp_size=get_context_parallel_world_size(),
            )
            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                data=input_ids,
                eod_token=0,  # used for loss_mask, which we don't use
                pad_token=0,  # used for loss_mask, which we don't use
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                pad_mask_loss=False,
            )
            if "mtp_loss_mask" in data_dict:
                mtp_loss_mask = data_dict["mtp_loss_mask"]
            # Unpacked: rows still are samples, so the mask is already in the
            # layout the model will see.
            if "media_token_validity_mask" in data_dict:
                media_token_validity_mask = data_dict[
                    "media_token_validity_mask"
                ].bool()
    return ProcessedInputs(
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids_cp_sharded,
        attention_mask=attention_mask,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
        cu_seqlens_padded=cu_seqlens_padded,
        mtp_loss_mask=mtp_loss_mask,
        padding_mask=padding_mask,
        routed_experts=routed_experts,
        routed_experts_cp_sharded=routed_experts_cp_sharded,
        original_seq_length=original_seq_length,
        media_token_validity_mask=media_token_validity_mask,
    )


def _make_r3_trace_token_identity(
    input_ids: torch.Tensor,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build debug-only ``[batch_idx, token_pos, valid]`` token identities."""
    batch_size, seq_len = input_ids.shape[:2]
    batch_idx = torch.arange(
        batch_size,
        dtype=torch.int32,
        device=input_ids.device,
    ).view(batch_size, 1, 1)
    token_pos = torch.arange(
        seq_len,
        dtype=torch.int32,
        device=input_ids.device,
    ).view(1, seq_len, 1)
    if seq_lengths is None:
        valid = torch.ones(
            batch_size,
            seq_len,
            1,
            dtype=torch.int32,
            device=input_ids.device,
        )
    else:
        valid = (
            token_pos.expand(batch_size, seq_len, 1)
            < seq_lengths.to(device=input_ids.device, dtype=torch.int32).view(
                batch_size,
                1,
                1,
            )
        ).to(dtype=torch.int32)
    return torch.cat(
        (
            batch_idx.expand(batch_size, seq_len, 1),
            token_pos.expand(batch_size, seq_len, 1),
            valid,
        ),
        dim=-1,
    )


def _verify_r3_trace_cp_token_alignment(
    *,
    source_input_ids: torch.Tensor,
    source_routed_experts: Optional[torch.Tensor],
    input_ids_cp_sharded: torch.Tensor,
    routed_experts_cp_sharded: Optional[torch.Tensor],
    token_identity_cp_sharded: Optional[torch.Tensor],
) -> Optional[int]:
    """Verify debug identities line up with CP-local tokens and routed experts."""
    if not r3_trace_verify_forward_enabled() or token_identity_cp_sharded is None:
        return None
    if source_routed_experts is None or routed_experts_cp_sharded is None:
        raise RuntimeError(
            "R3 forward verifier expected routed_experts and token identity tensors "
            "to be present together."
        )
    if token_identity_cp_sharded.shape[-1] != 3:
        raise RuntimeError(
            "R3 token identity must have trailing [batch_idx, token_pos, valid] "
            f"dimension; got {tuple(token_identity_cp_sharded.shape)}"
        )

    flat_identity = token_identity_cp_sharded.reshape(-1, 3).to(dtype=torch.long)
    flat_tokens = input_ids_cp_sharded.reshape(-1)
    flat_routed = routed_experts_cp_sharded.reshape(
        -1,
        *routed_experts_cp_sharded.shape[2:],
    )
    if (
        flat_identity.shape[0] != flat_tokens.shape[0]
        or flat_identity.shape[0] != flat_routed.shape[0]
    ):
        raise RuntimeError(
            "R3 token identity, input_ids, and routed_experts CP slices have "
            "different token counts: "
            f"identity={flat_identity.shape[0]} tokens={flat_tokens.shape[0]} "
            f"routed={flat_routed.shape[0]}"
        )

    valid_mask = flat_identity[:, 2] == 1
    checked = int(valid_mask.sum().item())
    if checked == 0:
        return 0

    source_rows = flat_identity[valid_mask, 0]
    source_cols = flat_identity[valid_mask, 1]
    expected_tokens = source_input_ids[source_rows, source_cols].to(
        device=flat_tokens.device,
        dtype=flat_tokens.dtype,
    )
    actual_tokens = flat_tokens[valid_mask]
    if not bool(torch.equal(actual_tokens, expected_tokens)):
        raise RuntimeError(
            "R3 CP token identity verifier found input_ids that do not match "
            "their source [batch_idx, token_pos] identities."
        )

    expected_routed = source_routed_experts[source_rows, source_cols].to(
        device=flat_routed.device,
        dtype=flat_routed.dtype,
    )
    actual_routed = flat_routed[valid_mask]
    if not bool(torch.equal(actual_routed, expected_routed)):
        raise RuntimeError(
            "R3 CP token identity verifier found routed_experts that do not match "
            "their source [batch_idx, token_pos] identities."
        )

    return checked


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


def _prepare_vlm_batch_for_megatron(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor | Sequence[int],
    pad_individual_seqs_to_multiple_of: int,
    pad_full_seq_to: Optional[int] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PackedSeqParams,
    Optional[torch.Tensor],
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
    seq_lengths_cpu = list(to_cpu_int_tuple(seq_lengths))
    padded_lens = [_round_up_to_multiple(L, align) for L in seq_lengths_cpu]

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
            seq_lengths_cpu[-1] += deficit
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
    # per-sequence targets via cu_seqlens_padded.
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
    seq_lengths: torch.Tensor | Sequence[int],
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
    seq_lengths_cpu = to_cpu_int_tuple(seq_lengths)

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
        seq_len = seq_lengths_cpu[b]

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
            seq_len = seq_lengths_cpu[b]
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

    # total_tokens is required for PackedSeqParams.__post_init__ to build
    # seq_idx, which Mamba uses to reset SSM state at sample boundaries.
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max_seqlen),
        max_seqlen_kv=int(max_seqlen),
        qkv_format="thd",
        total_tokens=packed_input_ids.shape[1],
    )

    return (
        all_input_ids.contiguous(),
        packed_input_ids.contiguous(),
        packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    )


def _shard_routed_experts_for_cp(
    routed_experts: Optional[torch.Tensor],  # [B, S, L, K] or None
    token_identity: Optional[
        torch.Tensor
    ],  # [B, S, 3] or None (R3 forward verifier, debug)
    seq_lengths: torch.Tensor,  # [B]
    cu_seqlens: torch.Tensor,  # [B+1] valid cumulative (from _pack_sequences_for_megatron)
    cu_seqlens_padded: Optional[
        torch.Tensor
    ],  # [B+1] padded cumulative (None when no padding)
    cp_rank: int,
    cp_size: int,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """CP-shard routed_experts / token_identity onto main's per-seq packed layout.

    Mirrors _pack_sequences_for_megatron's per-sequence zigzag for input_ids: each
    sequence is padded to its padded length (from cu_seqlens_padded, so boundaries are
    IDENTICAL to input_ids) then sharded with _get_tokens_on_this_cp_rank(seq_dim=0).
    routed_experts pad rows use arange(topk) (a valid top-k route; mcore
    _validate_replay_tensor rejects 0/dup/-1). token_identity pads with 0 (verifier skips).
    No roll (these are per-token, not next-token targets).
    Returns (routed_packed, routed_cp_sharded, identity_packed, identity_cp_sharded),
    each [1, T(/cp), ...] or None.

    This additive helper is the only routed_experts-specific CP code path.
    """
    batch_size = seq_lengths.shape[0]

    all_routed = [] if routed_experts is not None else None
    cp_routed = [] if routed_experts is not None else None
    all_identity = [] if token_identity is not None else None
    cp_identity = [] if token_identity is not None else None
    topk = routed_experts.shape[-1] if routed_experts is not None else None

    for b in range(batch_size):
        seq_len = int(seq_lengths[b])
        if cu_seqlens_padded is not None:
            padded_len = int(cu_seqlens_padded[b + 1] - cu_seqlens_padded[b])
        else:
            padded_len = int(cu_seqlens[b + 1] - cu_seqlens[b])

        if routed_experts is not None:
            # [seq_len, num_moe_layers, topk] padded to the SAME padded_len boundary as
            # input_ids so the routes ride the same per-seq zigzag.
            re = routed_experts[b, :seq_len]
            if padded_len > seq_len:
                # mcore _validate_replay_tensor rejects zero/duplicate/-1 routes,
                # so pad each row with a valid top-k route arange(topk).
                default_route = torch.arange(
                    topk,
                    dtype=re.dtype,
                    device=re.device,
                ).view(1, 1, topk)
                pad_rows = default_route.expand(
                    padded_len - seq_len,
                    re.shape[1],
                    topk,
                )
                re = torch.cat((re, pad_rows), dim=0)
            all_routed.append(re)
            re_cp = (
                _get_tokens_on_this_cp_rank(re, cp_rank, cp_size, seq_dim=0)
                if cp_size > 1
                else re
            )
            cp_routed.append(re_cp)

        if token_identity is not None:
            # [seq_len, 3] padded to the SAME padded_len boundary
            id = token_identity[b, :seq_len]
            if padded_len > seq_len:
                # pad rows with valid=0 so the verifier skips them
                id = torch.nn.functional.pad(
                    id,
                    (0, 0, 0, padded_len - seq_len),
                    value=0,
                )
            all_identity.append(id)
            id_cp = (
                _get_tokens_on_this_cp_rank(id, cp_rank, cp_size, seq_dim=0)
                if cp_size > 1
                else id
            )
            cp_identity.append(id_cp)

    routed_packed = (
        torch.cat(all_routed, dim=0).unsqueeze(0)
        if routed_experts is not None
        else None
    )
    routed_cp_sharded = (
        torch.cat(cp_routed, dim=0).unsqueeze(0) if routed_experts is not None else None
    )
    identity_packed = (
        torch.cat(all_identity, dim=0).unsqueeze(0)
        if token_identity is not None
        else None
    )
    identity_cp_sharded = (
        torch.cat(cp_identity, dim=0).unsqueeze(0)
        if token_identity is not None
        else None
    )
    return routed_packed, routed_cp_sharded, identity_packed, identity_cp_sharded


def _get_pack_sequence_parameters_for_megatron(
    megatron_cfg: Mapping[str, Any],
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

    # individual sequence needs to be splitted to CP domain, and to TP domain when SP is enabled.
    minimum_pad_factor = get_parallel_token_alignment(megatron_cfg)
    assert pad_individual_seqs_to_multiple_of % minimum_pad_factor == 0, (
        f"make_sequence_length_divisible_by ({pad_individual_seqs_to_multiple_of}) is not a multiple of minimum_pad_factor ({minimum_pad_factor}).\n"
        f"Please set policy.make_sequence_length_divisible_by to a multiple of {minimum_pad_factor}.\n"
        f"    - If CP is enabled, the minimum pad factor is `cp_size * 2`.\n"
        f"    - If TP+SP is enabled, the minimum pad factor is `tp_size`.\n"
        f"    - If both are enabled, the minimum pad factor is `cp_size * 2 * tp_size`."
    )

    # packed sequence length, after sharding to TP and CP domains, needs to be divisible
    # by a recipe-dependent divisor:
    #   blockwise FP8 : 128  (cublas block size)
    #   MXFP8         :  32  (MXFP8 block size)
    #   other FP8     :  16
    #   HybridEP+flex : 128  (MAX_NUM_OF_TOKENS_PER_RANK must be divisible by
    #                         NUM_OF_TOKENS_PER_CHUNK=128 in deep_ep JIT kernels)
    # When multiple constraints apply, take the max (128 is a multiple of 32/16).
    divisor = get_fp8_token_alignment(megatron_cfg)
    if (
        megatron_cfg.get("moe_token_dispatcher_type") == "flex"
        and megatron_cfg.get("moe_flex_dispatcher_backend") == "hybridep"
    ):
        divisor = max(divisor, 128)
    if divisor > 1:
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
    seq_lengths: torch.Tensor | Sequence[int],
    cu_seqlens: torch.Tensor | Sequence[int],
    cu_seqlens_padded: Optional[torch.Tensor | Sequence[int]],
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
    seq_lengths_cpu = to_cpu_int_tuple(seq_lengths)
    cu_seqlens_cpu = to_cpu_int_tuple(cu_seqlens)
    cu_seqlens_padded_cpu = (
        to_cpu_int_tuple(cu_seqlens_padded) if cu_seqlens_padded is not None else None
    )

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
        seq_len = seq_lengths_cpu[b]

        if cp_size > 1 and cu_seqlens_padded_cpu is not None:
            # When using CP, we need to account for padding
            # Calculate the padded sequence boundaries
            pad_factor = cp_size * 2
            padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor
            start_idx = cu_seqlens_padded_cpu[b]

            # Only copy the valid tokens (not the padding)
            unpacked_output[b, :seq_len] = output_tensor[
                start_idx : start_idx + seq_len
            ]
        else:
            # No CP, use regular cu_seqlens
            start_idx = cu_seqlens_cpu[b]
            end_idx = cu_seqlens_cpu[b + 1]

            # Copy the valid tokens to the unpacked tensor
            unpacked_output[b, :seq_len] = output_tensor[start_idx:end_idx]

    return unpacked_output


def get_and_validate_seqlen(data: BatchedDataDict[Any]):
    # dim 1 is always assumed to be the sequence dim, sanity check this here.
    # Skip multimodal fields: their dim 1 is num_images / num_patches, not
    # seqlen. Skip opd_full's teacher payload columns too: materialize()
    # deliberately does not pad them to the forward pass's pad_to_seqlen
    # target (they never feed the model forward, only the loss, which reads
    # each sequence's own real length regardless of how far the column is
    # padded) -- see codec.py's OPD_FULL_FIELDS exclusion. Their dim 1 can
    # therefore legitimately be shorter than every other column's.
    sequence_dim = 1
    seq_dim_size = data["input_ids"].shape[sequence_dim]
    for k, v in data.items():
        if k in PACKED_MULTIMODAL_FIELDS or k in OPD_FULL_FIELDS:
            continue
        if torch.is_tensor(v) and len(v.shape) > 1:
            assert v.shape[sequence_dim] == seq_dim_size, (
                f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape} for key {k}"
            )
    return sequence_dim, seq_dim_size
