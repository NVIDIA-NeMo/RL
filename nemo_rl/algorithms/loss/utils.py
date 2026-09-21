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

from typing import Any, Optional

import torch

from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank


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


def _project_hidden_states_per_teacher(
    payload: torch.Tensor,
    weight_by_index: dict[int, torch.Tensor],
    teacher_index: torch.Tensor,
) -> torch.Tensor:
    """Project each sample's hidden states through its own teacher's LM head.

    ``payload`` is ``[B, S, H]``; ``teacher_index`` is ``[B]`` int, naming which
    loaded teacher LM head produced each row (see ``OPD_FULL_TEACHER_INDEX_FIELD``).
    Grouped by teacher so this costs one matmul per distinct teacher present in
    the microbatch, not a per-sample loop -- when sequence packing is on, each
    call already has ``B == 1`` (one sequence at a time), so it takes the
    single-teacher path and never materializes the scatter buffer.

    Raises:
        ValueError: If ``teacher_index`` does not carry one entry per payload
            row, if a row's teacher_index has no loaded LM head, or if an LM
            head disagrees with the payload or with the other teachers'.
    """
    teacher_index = teacher_index.to(device=payload.device)
    if int(teacher_index.shape[0]) != int(payload.shape[0]):
        raise ValueError(
            "opd_full teacher_index must carry one entry per payload row: got "
            f"{tuple(teacher_index.shape)} against payload {tuple(payload.shape)}."
        )

    def _weight_for(idx: int) -> torch.Tensor:
        weight = weight_by_index.get(idx)
        if weight is None:
            raise ValueError(
                f"opd_full row tagged teacher_index={idx}, but no teacher LM "
                f"head is loaded for that index (loaded: {sorted(weight_by_index)})."
            )
        if int(payload.shape[-1]) != int(weight.shape[1]):
            raise ValueError(
                "Teacher hidden states do not match the loaded teacher LM head "
                f"for teacher_index={idx}: payload width {payload.shape[-1]} vs "
                f"LM-head input width {weight.shape[1]}."
            )
        return weight

    present = [int(idx) for idx in teacher_index.unique().tolist()]
    if len(present) == 1:
        # The sequence-packing path lands here on every call. Projecting
        # straight into the result skips the scatter buffer below, which would
        # otherwise double the peak ``[B, S, V_local]`` allocation for nothing.
        weight = _weight_for(present[0])
        return torch.matmul(payload.to(dtype=weight.dtype), weight.t())

    # One buffer holds every group, so the groups have to agree on the
    # vocabulary shard and dtype. They do by construction -- every shard is
    # requested at the student's own width and dtype -- but say so here, since
    # the alternative is an opaque RuntimeError out of the masked assignment.
    reference_index, reference = next(iter(weight_by_index.items()))
    teacher_logits = torch.empty(
        payload.shape[0],
        payload.shape[1],
        int(reference.shape[0]),
        dtype=reference.dtype,
        device=payload.device,
    )
    for idx in present:
        weight = _weight_for(idx)
        if (int(weight.shape[0]), weight.dtype) != (
            int(reference.shape[0]),
            reference.dtype,
        ):
            raise ValueError(
                "opd_full teacher LM heads must share one vocabulary shard and "
                f"dtype to be projected together: teacher_index={idx} is "
                f"[{weight.shape[0]}, ...] {weight.dtype}, but "
                f"teacher_index={reference_index} is [{reference.shape[0]}, ...] "
                f"{reference.dtype}."
            )
        mask = teacher_index == idx
        teacher_logits[mask] = torch.matmul(
            payload[mask].to(dtype=weight.dtype), weight.t()
        )
    return teacher_logits


def reconstruct_opd_full_teacher_logits(
    payload: torch.Tensor,
    *,
    teacher_payload: str,
    student_logits: torch.Tensor,
    vocab_parallel_rank: Optional[int],
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
    teacher_output_layer_weight_by_index: Optional[dict[int, torch.Tensor]] = None,
    teacher_index: Optional[torch.Tensor] = None,
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
        teacher_output_layer_weight_by_index: ``[V_local, H_teacher]`` teacher
            LM-head shards for the hidden-state path, keyed by the stable index
            rows are tagged with. Routed by ``teacher_index`` whenever that
            column is present.
        teacher_index: ``[B]`` int, this microbatch's per-row teacher index
            (see ``OPD_FULL_TEACHER_INDEX_FIELD``). Required once
            ``teacher_output_layer_weight_by_index`` holds more than one entry,
            and validated against the loaded heads whenever present.

    Returns:
        Teacher logits ``[B, S_local, V_local]`` aligned with ``student_logits``.

    Raises:
        ValueError: If the payload and student shard cannot be aligned, if the
            hidden-state path is used without a teacher LM-head shard, or if
            several shards are loaded without a ``teacher_index`` to route by.
    """
    vocab_shard_size = int(student_logits.shape[-1])

    # Every narrowing below happens before the device copy, and before the
    # hidden-state path widens the payload to the vocabulary.
    if teacher_payload == "hidden_states":
        if not teacher_output_layer_weight_by_index:
            raise ValueError(
                "opd_full hidden-state reconstruction requires a loaded teacher "
                "output-layer weight shard on the training worker."
            )
        if len(teacher_output_layer_weight_by_index) > 1 and teacher_index is None:
            # Refuse to guess rather than fall back to one arbitrary shard: that
            # keeps every shape right and silently distills the whole microbatch
            # toward whichever teacher the dict happens to yield first.
            raise ValueError(
                f"opd_full has {len(teacher_output_layer_weight_by_index)} "
                "teacher LM-head shards loaded, but this microbatch carries "
                "no per-row teacher index (OPD_FULL_TEACHER_INDEX_FIELD); "
                "there is no way to tell which teacher produced each row."
            )
        # Every loaded shard, not just the ones this microbatch routes to: the
        # widths are uniform by construction (the worker rejects a teacher whose
        # hidden size disagrees at load time), so a mismatch here is a real
        # inconsistency and should not depend on which rows happened to arrive.
        for idx, weight in sorted(teacher_output_layer_weight_by_index.items()):
            if int(payload.shape[-1]) != int(weight.shape[1]):
                raise ValueError(
                    "Teacher hidden states do not match the loaded teacher LM head "
                    f"for teacher_index={idx}: payload width {payload.shape[-1]} vs "
                    f"LM-head input width {weight.shape[1]}."
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
        payload = payload[..., vocab_start_index:vocab_end_index]

    # Narrow to this rank's sequence window *before* the payload is widened to
    # the vocabulary. Projecting first would do cp_size times the matmul and
    # allocate a full-sequence [B, S, V_local] tensor that the divergence
    # kernel's chunking cannot bound.
    cp_size = (
        1
        if context_parallel_group is None
        else torch.distributed.get_world_size(context_parallel_group)
    )
    target_seq_len = int(student_logits.shape[1]) * cp_size
    pad_len = target_seq_len - int(payload.shape[1])
    if pad_len < 0:
        raise ValueError(
            "Teacher payload is longer than the student forward window: "
            f"{payload.shape[1]} vs {target_seq_len}."
        )
    if pad_len > 0:
        # Zero-padding survives the projection (the LM head has no bias), so the
        # padded positions carry the same uniform logits either way. They are
        # masked out downstream regardless.
        payload = torch.nn.functional.pad(payload, (0, 0, 0, pad_len))
    if cp_size > 1:
        cp_rank = torch.distributed.get_rank(context_parallel_group)
        payload = _get_tokens_on_this_cp_rank(payload, cp_rank, cp_size, seq_dim=1)

    # contiguous() before the copy: the vocabulary slice above is a view, and the
    # result is handed to save_for_backward. Without it the whole-vocabulary
    # payload storage stays resident on the device until backward completes.
    payload = payload.contiguous().to(device=student_logits.device)

    if teacher_payload == "hidden_states":
        # Validated above, before any narrowing ran.
        assert teacher_output_layer_weight_by_index is not None
        if teacher_index is not None:
            # Route by tag whenever the column is present: a tag naming an
            # unloaded teacher must fail loud, not take the only head.
            teacher_logits = _project_hidden_states_per_teacher(
                payload, teacher_output_layer_weight_by_index, teacher_index
            )
        else:
            weight = next(iter(teacher_output_layer_weight_by_index.values()))
            teacher_logits = torch.matmul(
                payload.to(dtype=weight.dtype),
                weight.t(),
            )
    else:
        teacher_logits = payload

    if int(teacher_logits.shape[-1]) != vocab_shard_size:
        raise ValueError(
            "Reconstructed teacher logits must match the student vocabulary shard "
            f"width; got {teacher_logits.shape[-1]} vs {vocab_shard_size}."
        )
    return teacher_logits


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
