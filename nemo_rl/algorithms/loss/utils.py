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

from typing import TYPE_CHECKING, Any, Optional

import torch

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.algorithms.x_token.loss_utils import (
    prepare_xtoken_cross_tokenizer_loss_input,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    ChunkedDistributedCrossEntropyToFixedLogits,
    ChunkedDistributedEntropy,
    ChunkedDistributedReverseKLToFixedLogits,
    _get_tokens_on_this_cp_rank,
    allgather_cp_sharded_tensor,
    from_parallel_logits_to_logprobs_packed_sequences,
    get_cp_sharded_next_token_logprobs,
    get_distillation_topk_logprobs_from_logits,
    get_next_token_logprobs_from_logits,
)

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.context_parallel import (
        ContextParallelSharder,
    )


def map_teacher_logits_to_draft_vocab(
    teacher_logits: torch.Tensor,
    d2t: Optional[torch.Tensor],
    vocab_parallel_rank: Optional[int] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Restrict full-vocab teacher logits to the draft vocabulary via ``d2t``.

    ``d2t`` maps draft-vocab index ``i`` to target-vocab index ``i + d2t[i]``.
    Under tensor parallelism the teacher logits arrive vocab-sharded, so they
    are gathered to the full vocab and re-sliced to this rank's shard of the
    draft vocabulary (the draft output layer is sharded the same way). No-op
    when ``d2t`` is None (full-vocab drafts).
    """
    if d2t is None:
        return teacher_logits
    reverse_mapping = (
        torch.arange(len(d2t), device=teacher_logits.device, dtype=d2t.dtype) + d2t
    )
    if vocab_parallel_group is not None:
        from megatron.core.tensor_parallel import (
            gather_from_tensor_model_parallel_region,
        )

        teacher_logits = gather_from_tensor_model_parallel_region(
            teacher_logits, vocab_parallel_group
        )
        tp_size = torch.distributed.get_world_size(vocab_parallel_group)
        local_draft_size = len(d2t) // tp_size
        assert vocab_parallel_rank is not None
        start_index = vocab_parallel_rank * local_draft_size
        end_index = (vocab_parallel_rank + 1) * local_draft_size
        reverse_mapping = reverse_mapping[start_index:end_index]
    return teacher_logits[:, :, reverse_mapping]


def roll_packed_seq_dim(
    tensor: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    seq_dim: int,
) -> torch.Tensor:
    """Left-shift a packed tensor by one along ``seq_dim`` within each segment.

    Equivalent to a per-sequence ``torch.roll(shifts=-1)`` over the packed
    layout: one global roll followed by zeroing each segment's final slot (the
    only positions where the global roll would leak the next segment's first
    row). Segment boundaries come from ``cu_seqlens_padded``, the physical
    offsets of the packed layout.
    """
    rolled = torch.roll(tensor, shifts=-1, dims=seq_dim)
    boundary_index = (cu_seqlens_padded[1:] - 1).to(
        dtype=torch.long, device=rolled.device
    )
    index: list[Any] = [slice(None)] * rolled.dim()
    index[seq_dim] = boundary_index
    rolled[tuple(index)] = 0
    return rolled


def pack_rolled_draft_token_mask(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
) -> torch.Tensor:
    """Build the packed draft-loss mask ``[1, T_packed]`` from unpacked masks.

    Mirrors the non-packed DRAFT prepare (``token_mask`` left-shifted by one,
    scaled by ``sample_mask``), laid out at each sequence's padded offset.
    Each sequence's last real slot (whose shifted target would cross the
    boundary) and all padding slots stay zero.

    Reuses ``_pack_input_ids`` for the padded-offset layout (including its
    clamp for bin-alignment padding absorbed into the last sequence's
    effective length) and ``roll_packed_seq_dim`` for the boundary-safe
    per-segment left shift.
    """
    packed = _pack_input_ids(
        token_mask * sample_mask.unsqueeze(-1), cu_seqlens, cu_seqlens_padded
    )
    return roll_packed_seq_dim(packed, cu_seqlens_padded, seq_dim=1)


def reconstruct_opd_full_teacher_logits(
    payload: torch.Tensor,
    *,
    teacher_payload: str,
    student_logits: torch.Tensor,
    vocab_parallel_rank: Optional[int],
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
    teacher_output_layer_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """Turn the transported teacher payload into this rank's teacher logit shard.

    The payload is canonical ``[B, S, D]`` (already TP/CP-gathered on the teacher
    side), so it is re-sharded onto the student's own CP window here, exactly as
    ``from_parallel_logits_to_logprobs`` does for the student logits.

    Args:
        payload: Teacher payload ``[B, S, D]`` -- hidden states or full logits.
        teacher_payload: ``"hidden_states"`` or ``"logits"``.
        student_logits: Student logits ``[B, S_local, V_local]``, used for the
            target device, dtype-independent shapes, and CP geometry.
        vocab_parallel_rank: This rank's vocabulary-parallel rank.
        context_parallel_group: Context-parallel process group, if any.
        teacher_output_layer_weight: ``[V_local, H_teacher]`` teacher LM-head
            shard; required for the hidden-state path.

    Returns:
        Teacher logits ``[B, S_local, V_local]`` aligned with ``student_logits``.

    Raises:
        ValueError: If the payload and student shard cannot be aligned, or if the
            hidden-state path is used without a teacher LM-head shard.
    """
    payload = payload.to(device=student_logits.device)
    vocab_shard_size = int(student_logits.shape[-1])

    if teacher_payload == "hidden_states":
        if teacher_output_layer_weight is None:
            raise ValueError(
                "opd_full hidden-state reconstruction requires a loaded teacher "
                "output-layer weight shard on the training worker."
            )
        if int(payload.shape[-1]) != int(teacher_output_layer_weight.shape[1]):
            raise ValueError(
                "Teacher hidden states do not match the loaded teacher LM head: "
                f"payload width {payload.shape[-1]} vs LM-head input width "
                f"{teacher_output_layer_weight.shape[1]}."
            )
        teacher_logits = torch.matmul(
            payload.to(dtype=teacher_output_layer_weight.dtype),
            teacher_output_layer_weight.t(),
        )
    else:
        assert vocab_parallel_rank is not None, (
            "vocab_parallel_rank is required to slice the opd_full logits payload"
        )
        vocab_start_index = vocab_parallel_rank * vocab_shard_size
        vocab_end_index = vocab_start_index + vocab_shard_size
        if int(payload.shape[-1]) < vocab_end_index:
            raise ValueError(
                "Teacher logits payload is narrower than this rank's vocabulary "
                f"window: payload width {payload.shape[-1]} vs required "
                f"{vocab_end_index}."
            )
        teacher_logits = payload[..., vocab_start_index:vocab_end_index]

    if int(teacher_logits.shape[-1]) != vocab_shard_size:
        raise ValueError(
            "Reconstructed teacher logits must match the student vocabulary shard "
            f"width; got {teacher_logits.shape[-1]} vs {vocab_shard_size}."
        )

    cp_size = (
        1
        if context_parallel_group is None
        else torch.distributed.get_world_size(context_parallel_group)
    )
    target_seq_len = int(student_logits.shape[1]) * cp_size
    pad_len = target_seq_len - int(teacher_logits.shape[1])
    if pad_len < 0:
        raise ValueError(
            "Teacher payload is longer than the student forward window: "
            f"{teacher_logits.shape[1]} vs {target_seq_len}."
        )
    if pad_len > 0:
        teacher_logits = torch.nn.functional.pad(teacher_logits, (0, 0, 0, pad_len))
    if cp_size > 1:
        cp_rank = torch.distributed.get_rank(context_parallel_group)
        teacher_logits = _get_tokens_on_this_cp_rank(
            teacher_logits, cp_rank, cp_size, seq_dim=1
        )
    return teacher_logits


def prepare_opd_full_loss_input(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
    *,
    vocab_parallel_rank: Optional[int],
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup],
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
    sampling_params: Optional[TrainingSamplingParams],
    chunk_size: Optional[int],
    teacher_output_layer_weight: Optional[torch.Tensor],
) -> dict[str, Any]:
    """Build the full-vocabulary MOPD loss input from student logits + teacher payload.

    Runs the distributed reverse-KL kernel here rather than inside the loss so the
    loss stays free of process-group plumbing, mirroring the DISTILLATION branch.

    Args:
        logits: Student vocabulary-parallel logits ``[B, S_local, V_local]``.
        data: Microbatch carrying the teacher payload column.
        loss_fn: The ``opd_full``-configured loss function.
        vocab_parallel_rank: Vocabulary-parallel rank.
        vocab_parallel_group: Vocabulary-parallel process group.
        context_parallel_group: Context-parallel process group.
        sampling_params: Training sampling params for the sampled-token logprobs.
        chunk_size: Sequence-dim chunk size for the sampled-token logprobs.
        teacher_output_layer_weight: Teacher LM-head shard for the hidden path.

    Returns:
        Loss input dict with the per-token divergence and, when requested, the
        entropy/cross-entropy decomposition.

    Raises:
        ValueError: If the teacher payload column is missing.
        NotImplementedError: If no vocabulary-parallel group is available.
    """
    # Deferred: nemo_rl.algorithms.opd imports the data plane (tensordict), which
    # should not be pulled into every loss-function consumer.
    from nemo_rl.algorithms.opd import opd_full_payload_field

    full_cfg = loss_fn.opd_full  # type: ignore[attr-defined]
    if vocab_parallel_group is None:
        raise NotImplementedError(
            "opd_full currently requires the Megatron vocabulary-parallel path; "
            "the DTensor-only logit path is not supported."
        )

    payload_field = opd_full_payload_field(full_cfg)
    if payload_field not in data:
        raise ValueError(
            f"opd_full requires the teacher payload column {payload_field!r} in "
            "the training microbatch."
        )

    teacher_logits = reconstruct_opd_full_teacher_logits(
        data[payload_field],
        teacher_payload=full_cfg.teacher_payload,
        student_logits=logits,
        vocab_parallel_rank=vocab_parallel_rank,
        context_parallel_group=context_parallel_group,
        teacher_output_layer_weight=teacher_output_layer_weight,
    ).detach()

    divergence_chunk_size = full_cfg.chunk_size or int(logits.shape[1])
    reverse_kl = ChunkedDistributedReverseKLToFixedLogits.apply(  # type: ignore[misc]
        logits,
        teacher_logits,
        divergence_chunk_size,
        vocab_parallel_group,
        False,
    )
    entropy = None
    cross_entropy = None
    if full_cfg.validate_decomposition:
        entropy = ChunkedDistributedEntropy.apply(  # type: ignore[misc]
            logits,
            divergence_chunk_size,
            vocab_parallel_group,
            False,
        )
        cross_entropy = ChunkedDistributedCrossEntropyToFixedLogits.apply(  # type: ignore[misc]
            logits,
            teacher_logits,
            divergence_chunk_size,
            vocab_parallel_group,
            False,
        )

    if context_parallel_group is not None and (
        torch.distributed.get_world_size(context_parallel_group) > 1
    ):
        reverse_kl = allgather_cp_sharded_tensor(
            reverse_kl, context_parallel_group, seq_dim=1
        )
        if entropy is not None:
            entropy = allgather_cp_sharded_tensor(
                entropy, context_parallel_group, seq_dim=1
            )
        if cross_entropy is not None:
            cross_entropy = allgather_cp_sharded_tensor(
                cross_entropy, context_parallel_group, seq_dim=1
            )

    # Position t predicts token t+1 on both sides, so dropping the last position
    # matches the LOGPROB convention that pairs with token_mask[:, 1:].
    next_token_width = int(data["input_ids"].shape[1]) - 1
    loss_input: dict[str, Any] = {
        "opd_full_divergence": reverse_kl[:, :next_token_width],
        "opd_full_entropy": None if entropy is None else entropy[:, :next_token_width],
        "opd_full_cross_entropy": (
            None if cross_entropy is None else cross_entropy[:, :next_token_width]
        ),
    }
    if getattr(loss_fn, "reference_policy_kl_penalty", 0) != 0:
        loss_input["next_token_logprobs"] = get_next_token_logprobs_from_logits(
            input_ids=data["input_ids"],
            next_token_logits=logits,
            seq_index=data.get("seq_index", None),
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
            sampling_params=sampling_params,
            chunk_size=chunk_size,
        )
    return loss_input


def prepare_loss_input(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
    vocab_parallel_rank: Optional[int] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    d2t: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    cp_sharder: Optional["ContextParallelSharder"] = None,
    teacher_output_layer_weight: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], BatchedDataDict[Any]]:
    """Prepare loss input for a loss function.

    Args:
        logits: Logits from the model.
        data: Microbatch data. Will be updated if sampling_params is not None.
        loss_fn: Loss function.
        vocab_parallel_rank: Vocab parallel rank.
        vocab_parallel_group: Vocab parallel group.
        context_parallel_group: Context parallel group.
        sampling_params: Sampling parameters.
        d2t: Draft to target token mapping.
        chunk_size: Sequence-dim chunk size for the vocab-parallel logprob
            computation (policy.logprob_chunk_size); avoids materializing
            full-size float32 logits during training.
        cp_sharder: Automodel ``ContextParallelSharder`` owning this forward's
            sequence layout (V2 automodel worker with cp_size > 1); ``logits``
            are then this rank's CP-local shard while ``data`` stays canonical.
        teacher_output_layer_weight: This TP rank's ``[V_local, H_teacher]``
            teacher LM-head shard, used by the ``opd_full`` hidden-state path to
            project the teacher payload into teacher logits.

    Notes:
        vocab_parallel_rank, vocab_parallel_group, context_parallel_group are only used for megatron policy worker.
        sampling_params is only used for LossInputType.LOGPROB, and currently only supported for ClippedPGLossFn.
        d2t is only used for LossInputType.DRAFT.
        teacher_output_layer_weight is only used for LossInputType.OPD_FULL.

    Returns:
        tuple(loss_input, maybe_updated_data)
    """
    if loss_fn.input_type == LossInputType.LOGIT:
        loss_input = {"logits": logits}

    elif loss_fn.input_type == LossInputType.LOGPROB:
        # Linear CE fusion patch returns precomputed next-token logprobs (2D tensor).
        # Keep normal path unchanged for standard logits (3D tensor).
        if (
            hasattr(loss_fn, "use_fused_linear_logprobs")
            and loss_fn.use_fused_linear_logprobs
        ):
            logprobs = logits
            logprobs = logprobs.to(torch.float32)
            logprobs = logprobs[:, : data["input_ids"].shape[1] - 1]
        else:
            logprobs = get_next_token_logprobs_from_logits(
                input_ids=data["input_ids"],
                next_token_logits=logits,
                seq_index=data.get("seq_index", None),
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                sampling_params=sampling_params,
                chunk_size=chunk_size,
                cp_sharder=cp_sharder,
            )

        # handle top-k/top-p filtering for logprobs, only used for ClippedPGLossFn now
        if need_top_k_or_top_p_filtering(sampling_params):
            # mask out negative infinity logprobs
            # prev_logprobs is already masked out in the previous step
            mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
            logprobs = mask_out_neg_inf_logprobs(logprobs, mask[:, 1:], "curr_logprobs")

            # compute unfiltered logprobs for reference policy KL penalty
            if (
                hasattr(loss_fn, "reference_policy_kl_penalty")
                and loss_fn.reference_policy_kl_penalty != 0
            ):
                data["curr_logprobs_unfiltered"] = get_next_token_logprobs_from_logits(
                    input_ids=data["input_ids"],
                    next_token_logits=logits,
                    seq_index=data.get("seq_index", None),
                    vocab_parallel_rank=vocab_parallel_rank,
                    vocab_parallel_group=vocab_parallel_group,
                    context_parallel_group=context_parallel_group,
                    sampling_params=None,  # no filtering
                    # Only reachable with top-k/top-p sampling active that has its own kernel path so don't chunk here
                    chunk_size=None,
                    cp_sharder=cp_sharder,
                )

        loss_input = {"next_token_logprobs": logprobs}

    elif loss_fn.input_type == LossInputType.OPD_FULL:
        loss_input = prepare_opd_full_loss_input(
            logits,
            data,
            loss_fn,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
            sampling_params=sampling_params,
            chunk_size=chunk_size,
            teacher_output_layer_weight=teacher_output_layer_weight,
        )

    elif loss_fn.input_type == LossInputType.DISTILLATION:
        calculate_entropy = loss_fn.zero_outside_topk and loss_fn.kl_type != "forward"
        student_topk_logprobs, teacher_topk_logprobs, H_all = (
            get_distillation_topk_logprobs_from_logits(
                student_logits=logits,
                teacher_topk_logits=data["teacher_topk_logits"],
                teacher_topk_indices=data["teacher_topk_indices"],
                zero_outside_topk=loss_fn.zero_outside_topk,
                calculate_entropy=calculate_entropy,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                cp_sharder=cp_sharder,
            )
        )

        loss_input = {
            "student_topk_logprobs": student_topk_logprobs,
            "teacher_topk_logprobs": teacher_topk_logprobs,
            "H_all": H_all,
        }
    elif loss_fn.input_type == LossInputType.DISTILLATION_CROSS_TOKENIZER:
        # Rebuild each teacher's full-vocab logits from its per-rank CUDA IPC
        # handles and do the shared CP-resolution the loss needs; the loss fn
        # does the per-teacher projection / chunk-average / KL reductions and
        # aggregates them by ``kd_loss_mode``. ``projection_matrix_paths`` drives
        # the teacher count and which teachers are same-tokenizer (``None``). The
        # TP/CP groups are derived from the student logits' own device mesh.
        (
            student_logits_contig,
            teacher_full_logits_by_idx,
            aligns_by_idx,
            tp_group,
            cp_group,
        ) = prepare_xtoken_cross_tokenizer_loss_input(
            logits,
            data,
            projection_matrix_paths=loss_fn.projection_matrix_paths,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
            cp_sharder=cp_sharder,
        )
        loss_input = {
            "logits": logits,
            "student_logits_contig": student_logits_contig,
            "teacher_full_logits_by_idx": teacher_full_logits_by_idx,
            "aligns_by_idx": aligns_by_idx,
            "tp_group": tp_group,
            "cp_group": cp_group,
        }
        if cp_sharder is not None:
            next_token_logprobs = get_cp_sharded_next_token_logprobs(
                logits,
                data["input_ids"],
                cp_sharder,
                chunk_size=chunk_size,
            )
            # The sharder gathers canonical log-probabilities on every CP rank.
            # Give each rank one disjoint canonical window for CE backward so
            # every token contributes exactly once across the CP group. Append
            # the unused final-token slot first so partitioning uses the original
            # sequence length rather than the next-token length.
            full_logprobs = torch.cat(
                [next_token_logprobs, torch.zeros_like(next_token_logprobs[:, :1])],
                dim=1,
            )
            cp_size = (
                torch.distributed.get_world_size(context_parallel_group)
                if context_parallel_group is not None
                else 1
            )
            full_seq_len = full_logprobs.shape[1]
            if full_seq_len % cp_size != 0:
                raise ValueError(
                    "Student sequence length must be divisible by the student "
                    "context parallel size, but got "
                    f"sequence_length={full_seq_len}, cp_size={cp_size}. "
                    "Set policy.make_sequence_length_divisible_by to a multiple of "
                    "policy.dtensor_cfg.context_parallel_size."
                )
            cp_rank = (
                torch.distributed.get_rank(context_parallel_group)
                if context_parallel_group is not None
                else 0
            )
            local_seq_len = full_seq_len // cp_size
            seq_start = cp_rank * local_seq_len
            next_token_mask = (
                data["token_mask"].to(full_logprobs.device).roll(shifts=-1, dims=1)
            )
            next_token_mask[:, -1] = 0
            loss_input.update(
                student_next_token_logprobs=full_logprobs.narrow(
                    1, seq_start, local_seq_len
                ).contiguous(),
                student_next_token_mask=next_token_mask.narrow(
                    1, seq_start, local_seq_len
                ).contiguous(),
            )
    elif loss_fn.input_type == LossInputType.DRAFT:
        from megatron.core.transformer.multi_token_prediction import roll_tensor

        teacher_logits = roll_tensor(
            logits.detach(),
            shifts=-1,
            dims=1,
            cp_group=context_parallel_group,
        )[0]
        token_mask = roll_tensor(
            data["token_mask"], shifts=-1, dims=1, cp_group=context_parallel_group
        )[0]
        teacher_logits = map_teacher_logits_to_draft_vocab(
            teacher_logits,
            d2t,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
        )
        loss_input = {
            "teacher_logits": teacher_logits,
            "student_logits": data["student_logits"],
            "token_mask": token_mask,
        }

    else:
        raise ValueError(f"Unknown loss function input type: {loss_fn.input_type}")

    return loss_input, data


def _pack_input_ids(
    input_ids: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_padded: torch.Tensor,
    cp_rank: int = 0,
    cp_size: int = 1,
    roll_shift: int = 0,
) -> torch.Tensor:
    """Pack input_ids from [B, S] to [1, T_packed // CP] using sequence boundaries.

    Each sequence is individually padded to its padded length (from
    cu_seqlens_q_padded), optionally rolled, and CP-sharded at that padded
    length before being placed into the packed output.  This matches how
    Megatron packs and CP-shards sequences in _pack_sequences_for_megatron.

    Args:
        input_ids: Unpacked input IDs [B, S].
        cu_seqlens_q: Unpadded cumulative sequence lengths [B+1].
        cu_seqlens_q_padded: Padded cumulative sequence lengths [B+1].
        cp_rank: Context parallelism rank.
        cp_size: Context parallelism size.
        roll_shift: If non-zero, roll each padded sequence by this amount
            before CP-sharding.  Use -1 to build shifted targets for
            next-token prediction.
    """
    batch_size = input_ids.shape[0]
    total_packed_len = int(cu_seqlens_q_padded[-1].item()) // cp_size
    packed = torch.zeros(
        total_packed_len, dtype=input_ids.dtype, device=input_ids.device
    )
    for i in range(batch_size):
        actual_len = int((cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item())
        padded_len = int((cu_seqlens_q_padded[i + 1] - cu_seqlens_q_padded[i]).item())
        packed_start = int(cu_seqlens_q_padded[i].item())
        seq = torch.zeros(padded_len, dtype=input_ids.dtype, device=input_ids.device)
        # The packer absorbs bin-level alignment padding into the last
        # sequence's effective length (see _get_pack_sequence_parameters_for_megatron),
        # so cu_seqlens can exceed the unpacked row width. Copy only real
        # tokens; the tail stays zero and is excluded from the loss by token_mask.
        copy_len = min(actual_len, input_ids.shape[1])
        seq[:copy_len] = input_ids[i, :copy_len]
        if roll_shift != 0:
            seq = seq.roll(shifts=roll_shift, dims=0)
        sharded = _get_tokens_on_this_cp_rank(seq, cp_rank, cp_size, seq_dim=0)
        packed[packed_start // cp_size : (packed_start + padded_len) // cp_size] = (
            sharded
        )
    return packed.unsqueeze(0)


def prepare_packed_loss_input(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_padded: torch.Tensor,
    vocab_parallel_rank: Optional[int] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    chunk_size: Optional[int] = None,
) -> tuple[dict[str, Any], BatchedDataDict[Any]]:
    """Prepare loss input from packed logits in a single fused pass.

    Unlike prepare_loss_input which operates on a single (unpacked) sequence,
    this function computes log probabilities from packed logits across all
    sequences at once using from_parallel_logits_to_logprobs_packed_sequences.

    Currently only supports LossInputType.LOGPROB.

    Args:
        logits: Packed logits from the model [1, T_packed // CP, V // TP].
        data: Microbatch data (unpacked, [B, S]).
        loss_fn: Loss function (must have input_type == LossInputType.LOGPROB).
        cu_seqlens_q: Unpadded cumulative sequence lengths [B+1].
        cu_seqlens_q_padded: Padded cumulative sequence lengths [B+1].
        vocab_parallel_rank: Vocab parallel rank.
        vocab_parallel_group: Vocab parallel group.
        context_parallel_group: Context parallel group.
        sampling_params: Sampling parameters.
        chunk_size: Sequence-dim chunk size for the logprob computation
            (policy.logprob_chunk_size); avoids materializing full-size
            float32 logits during training.

    Returns:
        tuple(loss_input, maybe_updated_data)
    """
    if loss_fn.input_type != LossInputType.LOGPROB:
        raise ValueError(
            f"prepare_packed_loss_input only supports LossInputType.LOGPROB, "
            f"got {loss_fn.input_type}. Use SequencePackingLossWrapper with "
            f"prepare_loss_input for other types."
        )
    assert vocab_parallel_group is not None, (
        "prepare_packed_loss_input requires vocab_parallel_group (Megatron TP)."
    )
    assert vocab_parallel_rank is not None, (
        "vocab_parallel_rank must be provided with vocab_parallel_group."
    )

    input_ids = data["input_ids"]
    unpacked_seqlen = input_ids.shape[1]
    cp_size = (
        1
        if context_parallel_group is None
        else torch.distributed.get_world_size(context_parallel_group)
    )
    cp_rank = (
        0
        if context_parallel_group is None
        else torch.distributed.get_rank(context_parallel_group)
    )

    packed_rolled_targets = _pack_input_ids(
        input_ids,
        cu_seqlens_q,
        cu_seqlens_q_padded,
        cp_rank=cp_rank,
        cp_size=cp_size,
        roll_shift=-1,
    )

    # With chunking, keep logits in their original dtype: the chunked logprob
    # kernel casts each chunk to float32 internally.
    use_chunking = chunk_size is not None and not need_top_k_or_top_p_filtering(
        sampling_params
    )
    logits_for_logprobs = logits if use_chunking else logits.to(torch.float32)

    logprobs = from_parallel_logits_to_logprobs_packed_sequences(
        logits_for_logprobs,
        packed_rolled_targets,
        cu_seqlens_q_padded,
        unpacked_seqlen,
        vocab_start_index=vocab_parallel_rank * logits.shape[-1],
        vocab_end_index=(vocab_parallel_rank + 1) * logits.shape[-1],
        group=vocab_parallel_group,
        inference_only=False,
        cp_group=context_parallel_group,
        sampling_params=sampling_params,
        chunk_size=chunk_size if use_chunking else None,
        target_is_pre_rolled=True,
    )

    # Match prepare_loss_input behavior for top-k/top-p filtered training:
    # use filtered curr_logprobs for actor loss, but keep unfiltered values for KL.
    if need_top_k_or_top_p_filtering(sampling_params):
        mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        logprobs = mask_out_neg_inf_logprobs(logprobs, mask[:, 1:], "curr_logprobs")

        if (
            hasattr(loss_fn, "reference_policy_kl_penalty")
            and loss_fn.reference_policy_kl_penalty != 0
        ):
            data["curr_logprobs_unfiltered"] = (
                from_parallel_logits_to_logprobs_packed_sequences(
                    logits_for_logprobs,
                    packed_rolled_targets,
                    cu_seqlens_q_padded,
                    unpacked_seqlen,
                    vocab_start_index=vocab_parallel_rank * logits.shape[-1],
                    vocab_end_index=(vocab_parallel_rank + 1) * logits.shape[-1],
                    group=vocab_parallel_group,
                    inference_only=False,
                    cp_group=context_parallel_group,
                    sampling_params=None,
                    chunk_size=chunk_size if use_chunking else None,
                    target_is_pre_rolled=True,
                )
            )

    return {"next_token_logprobs": logprobs}, data
