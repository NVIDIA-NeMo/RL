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
import torch.distributed

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
    _get_tokens_on_this_cp_rank,
    from_parallel_logits_to_logprobs_packed_sequences,
    get_cp_sharded_next_token_logprobs,
    get_distillation_topk_logprobs_from_logits,
    get_next_token_logprobs_from_logits,
)

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.context_parallel import (
        ContextParallelSharder,
    )


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

    Notes:
        vocab_parallel_rank, vocab_parallel_group, context_parallel_group are only used for megatron policy worker.
        sampling_params is only used for LossInputType.LOGPROB, and currently only supported for ClippedPGLossFn.
        d2t is only used for LossInputType.DRAFT.

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
        # TTT convention: pass-d student logits z^d_i predict x_{i+d+1}; the
        # matching teacher is the policy's logits at position i+d. The teacher
        # stays UNSHIFTED here. In the unpacked path the loss fn slices per
        # pass (pure views), which requires every rank to see the full
        # sequence (CP == 1). In the packed path the loss fn instead rolls
        # the teacher per subsequence with the CP-aware boundary exchange, so
        # CP > 1 is allowed there (the train loop stashes the packed zigzag
        # coords the loss needs).
        if (
            context_parallel_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(context_parallel_group) > 1
            and "draft_packed_pos_in_seq" not in data
        ):
            raise NotImplementedError(
                "Draft distillation loss requires context_parallel_size == 1 "
                "unless sequence packing is enabled."
            )
        teacher_logits = logits.detach()
        if d2t is not None:
            reverse_mapping = (
                torch.arange(len(d2t), device=teacher_logits.device, dtype=d2t.dtype)
                + d2t
            )
            if vocab_parallel_group is not None:
                from megatron.core.tensor_parallel import (
                    gather_from_tensor_model_parallel_region,
                )

                tp_size = torch.distributed.get_world_size(vocab_parallel_group)
                local_draft_size = len(d2t) // tp_size
                assert vocab_parallel_rank is not None
                start_index = vocab_parallel_rank * local_draft_size
                end_index = (vocab_parallel_rank + 1) * local_draft_size
                reverse_mapping = reverse_mapping[start_index:end_index]

                # Gather + d2t-subset in sequence chunks: gathering the whole
                # [B, S, V_full] teacher at once peaks at several GiB at long
                # sequence lengths, while only the [B, S, draft_vocab/tp]
                # subset survives.
                batch_size, seq_size = teacher_logits.shape[:2]
                subset_teacher = torch.empty(
                    batch_size,
                    seq_size,
                    reverse_mapping.shape[0],
                    dtype=teacher_logits.dtype,
                    device=teacher_logits.device,
                )
                for chunk_start in range(0, seq_size, loss_fn.seq_chunk_size):
                    chunk_end = min(seq_size, chunk_start + loss_fn.seq_chunk_size)
                    gathered_chunk = gather_from_tensor_model_parallel_region(
                        teacher_logits[:, chunk_start:chunk_end].contiguous(),
                        vocab_parallel_group,
                    )
                    subset_teacher[:, chunk_start:chunk_end] = gathered_chunk[
                        :, :, reverse_mapping
                    ]
                    del gathered_chunk
                teacher_logits = subset_teacher
            else:
                teacher_logits = teacher_logits[:, :, reverse_mapping]
        if "student_logits_by_pass" in data:
            student_logits_by_pass = list(data["student_logits_by_pass"])
        else:
            student_logits_by_pass = [data["student_logits"]]
        loss_input = {
            "teacher_logits": teacher_logits,
            "student_logits_by_pass": student_logits_by_pass,
        }

    else:
        raise ValueError(f"Unknown loss function input type: {loss_fn.input_type}")

    return loss_input, data


def roll_packed_left_cp(
    tensor: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Left-shift by one WITHIN each packed subsequence, zigzag-CP-aware.

    ``tensor``'s dim 0 is the rank-local packed (THD) dimension and
    ``cu_seqlens_local`` its subsequence boundaries. Without CP each
    subsequence rolls independently with its tail zero-filled. Under CP the
    local row of every subsequence is the zigzag pair ``[chunk_r,
    chunk_{2cp-1-r}]`` (the ``_get_tokens_on_this_cp_rank`` layout), so each
    chunk rolls locally and its tail element is the head of the globally NEXT
    chunk: chunk ``r``'s successor lives on rank ``r+1`` (or is this rank's
    own back chunk when ``r == cp-1``), chunk ``2cp-1-r``'s successor on rank
    ``r-1`` (or is the global sequence tail, zero, when ``r == 0``). All
    subsequences' boundary elements travel in ONE batched isend/irecv.
    """
    cp_size = 1 if cp_group is None else cp_group.size()
    cu = cu_seqlens_local.to(dtype=torch.long)
    seq_lens = cu[1:] - cu[:-1]

    # Global roll first, then patch every chunk-tail row (which the global
    # roll filled with the next chunk's head IN LOCAL LAYOUT — wrong across
    # subsequence/zigzag boundaries).
    rolled = torch.roll(tensor, shifts=-1, dims=0)

    if cp_size == 1:
        rolled[cu[1:] - 1] = 0
        return rolled

    cp_rank = cp_group.rank()
    global_ranks = torch.distributed.get_process_group_ranks(cp_group)
    next_rank = global_ranks[(cp_rank + 1) % cp_size]
    prev_rank = global_ranks[(cp_rank - 1) % cp_size]

    half = seq_lens // 2
    front_head = cu[:-1]
    back_head = cu[:-1] + half
    front_tail = back_head - 1
    back_tail = cu[1:] - 1

    send_to_prev = tensor[front_head]  # prev's front-chunk tails need these
    send_to_next = tensor[back_head]  # next's back-chunk tails need these
    recv_from_next = torch.empty_like(send_to_prev)
    recv_from_prev = torch.empty_like(send_to_next)

    ops = []
    if cp_rank > 0:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, send_to_prev, prev_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_from_prev, prev_rank, cp_group
            )
        )
    if cp_rank < cp_size - 1:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, send_to_next, next_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_from_next, next_rank, cp_group
            )
        )
    if ops:
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
    if cp_rank == cp_size - 1:
        # Chunk cp-1's global successor is chunk cp — this rank's own back chunk.
        recv_from_next = tensor[back_head]
    if cp_rank == 0:
        # Chunk 2cp-1 is the global sequence tail.
        recv_from_prev = torch.zeros_like(recv_from_prev)

    rolled[front_tail] = recv_from_next
    rolled[back_tail] = recv_from_prev
    return rolled


def packed_zigzag_token_coords(
    cu_seqlens_global: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each rank-local packed token to ``(subseq index, global position)``.

    ``cu_seqlens_global`` holds the PRE-CP padded boundaries. The local
    layout is the per-subsequence zigzag (`_get_tokens_on_this_cp_rank`):
    rank ``r`` holds chunks ``r`` and ``2*cp - 1 - r`` of every subsequence.
    Returns two ``[T_local]`` int64 tensors: the packed subsequence index
    (== the pre-packing batch row) and the token's position WITHIN its
    subsequence in the original (un-sharded) order. Pure local arithmetic,
    no communication — the basis for gathering per-token masks/labels from
    unpacked ``[B, S]`` tensors.
    """
    cu = cu_seqlens_global.to(dtype=torch.long)
    lens_local = (cu[1:] - cu[:-1]) // cp_size
    num_seqs = lens_local.numel()
    device = cu.device
    cu_local = torch.zeros(num_seqs + 1, dtype=torch.long, device=device)
    cu_local[1:] = torch.cumsum(lens_local, dim=0)
    seq_index = torch.repeat_interleave(
        torch.arange(num_seqs, device=device), lens_local
    )
    pos_local = (
        torch.arange(int(cu_local[-1].item()), device=device) - cu_local[:-1][seq_index]
    )
    if cp_size == 1:
        return seq_index, pos_local
    half = (lens_local // 2)[seq_index]
    is_front = pos_local < half
    pos_global = torch.where(
        is_front,
        cp_rank * half + pos_local,
        (2 * cp_size - 1 - cp_rank) * half + (pos_local - half),
    )
    return seq_index, pos_global


def packed_zigzag_local_index(
    seq_idx: torch.Tensor,
    pos: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Rank-local THD row of ``(subseq, global in-seq position)``.

    The caller guarantees this rank owns each position's zigzag chunk
    (``pos // half in {cp_rank, 2*cp - 1 - cp_rank}``).
    """
    cu = cu_seqlens_local.to(dtype=torch.long)
    half = ((cu[1:] - cu[:-1]) // 2)[seq_idx]
    front_off = pos - cp_rank * half
    is_front = (front_off >= 0) & (front_off < half)
    back_off = pos - (2 * cp_size - 1 - cp_rank) * half
    return cu[:-1][seq_idx] + torch.where(is_front, front_off, half + back_off)


def packed_zigzag_successor_halo(
    tensor: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    halo: int,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """First ``halo`` rows of each local chunk's globally NEXT chunk.

    ``tensor``'s dim 0 is the rank-local zigzag THD row. Returns
    ``[num_seqs, 2, halo, ...]`` (index 0 = front chunk's successor, 1 = back
    chunk's). Same neighbor pattern as :func:`roll_packed_left_cp`: front
    chunk ``r``'s successor is rank ``r+1``'s front chunk head (``r == cp-1``:
    this rank's own back chunk), back chunk ``2cp-1-r``'s successor is rank
    ``r-1``'s back chunk head (``r == 0``: past the sequence end, zeros).
    Requires ``halo <= min(half)`` — a successor chunk must cover the halo.
    """
    cp_size = 1 if cp_group is None else cp_group.size()
    cu = cu_seqlens_local.to(dtype=torch.long)
    half = (cu[1:] - cu[:-1]) // 2
    if halo > int(half.min().item()):
        raise ValueError(
            f"successor halo {halo} exceeds the smallest local chunk "
            f"{int(half.min().item())}; shorten gamma or lower CP for this "
            "sequence length."
        )
    num_seqs = half.numel()
    row_shape = tensor.shape[1:]
    head_index = torch.arange(halo, device=tensor.device).view(1, -1) + cu[:-1].view(
        -1, 1
    )
    front_head = tensor[head_index.reshape(-1)].reshape(num_seqs, halo, *row_shape)
    back_head = tensor[(head_index + half.view(-1, 1)).reshape(-1)].reshape(
        num_seqs, halo, *row_shape
    )

    if cp_size == 1:
        return torch.stack([back_head, torch.zeros_like(back_head)], dim=1)

    cp_rank = cp_group.rank()
    global_ranks = torch.distributed.get_process_group_ranks(cp_group)
    next_rank = global_ranks[(cp_rank + 1) % cp_size]
    prev_rank = global_ranks[(cp_rank - 1) % cp_size]

    recv_front_halo = torch.empty_like(front_head)
    recv_back_halo = torch.empty_like(back_head)
    ops = []
    if cp_rank > 0:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, front_head.contiguous(), prev_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_back_halo, prev_rank, cp_group
            )
        )
    if cp_rank < cp_size - 1:
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend, back_head.contiguous(), next_rank, cp_group
            )
        )
        ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_front_halo, next_rank, cp_group
            )
        )
    if ops:
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
    if cp_rank == cp_size - 1:
        recv_front_halo = back_head
    if cp_rank == 0:
        recv_back_halo = torch.zeros_like(back_head)
    return torch.stack([recv_front_halo, recv_back_halo], dim=1)


def draft_pass_token_mask(token_mask: torch.Tensor, ttt_pass: int) -> torch.Tensor:
    """Per-pass draft loss mask (before the sample_mask factor).

    Pass ``d`` position ``i`` predicts token ``x_{i+d+1}``, so its validity is
    ``token_mask[i + d + 1]``: a left shift by ``d + 1`` that also drops the
    out-of-bounds tail. Returned shape is ``[B, S - d - 1]``, aligned with the
    ``student[:, :S-d-1]`` / ``teacher[:, d:S-1]`` slices used by
    ``DraftCrossEntropyLossFn``.
    """
    return token_mask[:, ttt_pass + 1 :]


def compute_draft_pass_valid_counts(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    ttt_steps: int,
) -> torch.Tensor:
    """Local per-pass valid-token counts for the draft TTT loss.

    Returns a fp32 tensor of shape ``[ttt_steps]`` where entry ``d-1`` is this
    rank's ``#valid pass-d targets``. The caller must all-reduce it over the
    data-parallel group (and only DP: every CP/TP rank computes the loss over
    the full sequence/vocab, so reducing over those groups would double-count).

    The loss consumes the vector twice: the gradient denominator is
    ``sum_d alpha_d * counts[d-1]``, and the per-pass diagnostic metrics divide
    by ``counts[d-1]`` so that the driver's sum-over-microbatches aggregation
    (grpo.py logs mb-metric lists with ``np.sum``) reconstructs the global
    per-token mean instead of summing per-microbatch means.
    """
    counts = token_mask.new_zeros((ttt_steps,), dtype=torch.float32)
    for ttt_pass in range(1, ttt_steps + 1):
        pass_mask = draft_pass_token_mask(token_mask, ttt_pass) * sample_mask.unsqueeze(
            -1
        )
        counts[ttt_pass - 1] = pass_mask.sum().float()
    return counts


def block_draft_slot_mask(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    anchors: torch.Tensor,
    anchor_valid: torch.Tensor,
    *,
    gamma: int,
) -> torch.Tensor:
    """Validity mask ``[B, N, gamma]`` for DFlash/DSpark block draft slots.

    Slot ``j`` of a block anchored at ``p`` predicts ``x_{p + 1 + j}``; it is
    valid when the block is real (``anchor_valid``), the label position is in
    bounds, the label token is trained (``token_mask``), and the sample is
    valid. Anchors near the sequence end keep their in-bounds slots instead of
    being rejected outright (see the CP design note's packing issue ③).

    Validity is PREFIX-CONTIGUOUS along the slot axis: once a slot is invalid
    every later slot is too. A speculative block at serving time never spans a
    user/tool/prefill hole, so an anchor whose labels cross such a hole must
    not train the slots beyond it (multi-turn data would otherwise resurrect
    slots on the far side of the hole).
    """
    batch_size, num_anchors = anchors.shape
    seq_idx = (
        torch.arange(batch_size, device=anchors.device)
        .unsqueeze(1)
        .expand(-1, num_anchors)
        .reshape(-1)
    )
    return block_draft_slot_mask_packed(
        token_mask,
        sample_mask,
        seq_idx,
        anchors.reshape(-1),
        anchor_valid.reshape(-1),
        gamma=gamma,
    ).reshape(batch_size, num_anchors, gamma)


def block_draft_slot_mask_packed(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    seq_idx: torch.Tensor,
    anchors: torch.Tensor,
    anchor_valid: torch.Tensor,
    *,
    gamma: int,
) -> torch.Tensor:
    """Flat-block variant of :func:`block_draft_slot_mask`.

    ``seq_idx``/``anchors``/``anchor_valid`` are ``[NB]`` (one row per block,
    e.g. this CP rank's owned subset of a packed microbatch); the masks stay
    unpacked ``[B, S]``/``[B]``. Returns ``[NB, gamma]``.
    """
    seq_len = token_mask.shape[1]
    label_pos = (
        anchors.unsqueeze(-1)
        + 1
        + torch.arange(gamma, device=anchors.device).view(1, -1)
    )
    in_bounds = label_pos < seq_len
    label_token_mask = token_mask[
        seq_idx.unsqueeze(-1).expand(-1, gamma), label_pos.clamp(max=seq_len - 1)
    ]
    slot_mask = (
        anchor_valid.unsqueeze(-1)
        & in_bounds
        & (label_token_mask > 0.5)
        & (sample_mask[seq_idx].unsqueeze(-1) > 0.5)
    )
    return torch.cumprod(slot_mask.to(torch.int32), dim=-1).bool()


def compute_block_draft_slot_valid_counts(
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    anchors: torch.Tensor,
    anchor_valid: torch.Tensor,
    *,
    gamma: int,
) -> torch.Tensor:
    """Local per-slot valid counts ``[gamma]`` for the block draft loss.

    The block analogue of :func:`compute_draft_pass_valid_counts` (same
    DP-only all-reduce contract and the same denominator/metric consumption
    in the loss): entry ``j`` counts this rank's valid slot-``j`` CE targets.
    """
    slot_mask = block_draft_slot_mask(
        token_mask, sample_mask, anchors, anchor_valid, gamma=gamma
    )
    return slot_mask.float().sum(dim=(0, 1))


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
