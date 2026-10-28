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

"""W2S-OPD (arXiv:2607.26246), equations 2 and 7.

The three frozen models score identical student prefixes. Select the exact
global top-K of z_anchor + alpha * (z_positive - z_negative), then renormalize
both student and proxy on that support and differentiate KL(student || proxy).

The support depends only on the frozen models, so the negative teacher selects
it on GPU: it projects the anchor and positive hidden states with their LM
heads, contrasts them with its own logits, and publishes K token ids plus K
proxy log-probabilities per token (:func:`w2s_packed_proxy_support`). The
student then only gathers its own logits on that support
(:func:`w2s_reverse_kl_from_support`), and the data plane holds
``8 * K`` bytes per token until training (int32 ids, float32 log-probabilities);
the anchor and positive hidden states cross it only in temporary rows. Shipping
the three hidden states to the student instead would hold
``sum(H) * itemsize(payload_dtype)`` bytes per token, over the anchor, positive
and negative hidden widths: 21,760 against 256 (85x) for a 2688-wide anchor,
two 4096-wide teachers, bfloat16 and K = 32.
"""

from __future__ import annotations

import math

import torch

from nemo_rl.distributed.model_utils import (
    _get_tokens_on_this_cp_rank,
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    gather_logits_at_global_indices,
)


def validate_w2s_scoring(logprobs: torch.Tensor, input_lengths: torch.Tensor) -> None:
    """Fail collectively on nonfinite predicted-token scores before row masking.

    Call after the scoring schedule and pipeline broadcast have completed.
    Padding and the unpredicted first token do not belong to this check.
    """
    positions = torch.arange(logprobs.shape[1], device=logprobs.device)
    valid = (positions[None, :] >= 1) & (
        positions[None, :] < input_lengths.to(logprobs.device)[:, None]
    )
    bad_count = ((~torch.isfinite(logprobs)) & valid).sum()
    failed = (bad_count > 0).to(torch.int32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        raise FloatingPointError(
            "W2S scoring produced nonfinite predicted-token logprobs "
            f"(local_count={bad_count.item()}); refusing to mask and train on them."
        )


def _vocab_shard_start(
    width: int, vocab_parallel_group: torch.distributed.ProcessGroup | None
) -> tuple[int, int]:
    """Return ``(world, start)`` for this rank's ``width``-wide vocabulary shard."""
    if vocab_parallel_group is None:
        return 1, 0
    world = torch.distributed.get_world_size(vocab_parallel_group)
    return world, torch.distributed.get_rank(vocab_parallel_group) * width


@torch.no_grad()
def w2s_proxy_support(
    proxy_logits: torch.Tensor,
    *,
    top_k: int,
    vocab_size: int,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the proxy's exact global top-K and renormalize the proxy on it.

    Vocabulary padding is excluded before ranking. Every rank of the vocabulary
    group returns the same global token ids and log-probabilities.

    Returns:
        ``(support, target_logprobs)``: int64 global token ids and float32 proxy
        log-probabilities renormalized on that support, both ``[..., top_k]``.
    """
    width = proxy_logits.shape[-1]
    world, start = _vocab_shard_start(width, vocab_parallel_group)
    if not 2 <= top_k <= vocab_size <= width * world:
        raise ValueError(
            "W2S-OPD requires 2 <= top_k <= vocabulary <= padded vocabulary."
        )
    proxy = proxy_logits.detach().float()
    ids = torch.arange(start, start + width, device=proxy.device)
    proxy = proxy.masked_fill(ids >= vocab_size, float("-inf"))
    if vocab_parallel_group is None or world == 1:
        values, support = proxy.topk(top_k, dim=-1)
    else:
        values, support = distributed_vocab_topk(
            proxy,
            top_k,
            vocab_parallel_group,
            vocab_start_index=start,
            vocab_end_index=start + width,
        )
    return support, values.log_softmax(-1)


def w2s_reverse_kl_on_support(
    student_logits: torch.Tensor,
    support: torch.Tensor,
    target_logprobs: torch.Tensor,
    *,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    """KL(student || proxy), both renormalized on a fixed proxy support.

    Gradients flow only through the student. No sampled-token tail,
    teacher-mass weighting, PPO ratio, or reward advantage enters this loss.
    """
    width = student_logits.shape[-1]
    _, start = _vocab_shard_start(width, vocab_parallel_group)
    selected = gather_logits_at_global_indices(
        student_logits,
        support.long(),
        tp_group=vocab_parallel_group,
        vocab_start_index=start,
        vocab_end_index=start + width,
    )
    student_logprobs = selected.float().log_softmax(-1)
    target_logprobs = target_logprobs.detach().float()
    return (student_logprobs.exp() * (student_logprobs - target_logprobs)).sum(-1)


def _validate_arms_and_heads(
    arms: tuple[torch.Tensor, ...],
    heads: tuple[torch.Tensor, ...],
    negative_logits: torch.Tensor,
) -> None:
    if len(heads) != 2 or any(h.ndim != 2 for h in heads):
        raise ValueError("W2S-OPD needs the anchor and positive LM-head matrices.")
    if negative_logits.ndim != 3 or any(
        h.shape[0] != negative_logits.shape[-1] for h in heads
    ):
        raise ValueError(
            "W2S-OPD LM heads and negative logits must share one vocabulary shard."
        )
    if len(arms) != 2 or any(
        p.ndim != 3
        or p.shape[-1] != h.shape[1]
        or p.shape[:2] != negative_logits.shape[:2]
        for p, h in zip(arms, heads, strict=True)
    ):
        raise ValueError(
            "W2S-OPD anchor/positive payloads do not match their LM heads and "
            "the negative logits."
        )


@torch.no_grad()
def _project_proxy(
    arms: tuple[torch.Tensor, ...],
    heads: tuple[torch.Tensor, ...],
    negative_logits: torch.Tensor,
    alpha: float,
    start: int,
    end: int,
) -> torch.Tensor:
    """fp32 proxy logits for sequence positions ``[start, end)`` of aligned arms."""
    device = heads[0].device
    # Accumulate each projected arm in fp32 before applying the contrast.
    # Never subtract independently truncated top-K distributions.
    anchor, positive = (
        torch.matmul(
            arm[:, start:end].to(device=device, dtype=head.dtype),
            head.detach().t(),
        ).float()
        for arm, head in zip(arms, heads, strict=True)
    )
    negative = negative_logits[:, start:end].to(device=device, dtype=torch.float32)
    return anchor + alpha * (positive - negative)


@torch.no_grad()
def w2s_compact_proxy(
    arms: tuple[torch.Tensor, torch.Tensor],
    heads: tuple[torch.Tensor, torch.Tensor],
    negative_logits: torch.Tensor,
    *,
    alpha: float,
    top_k: int,
    vocab_size: int,
    chunk_size: int,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen-side proxy support for position-aligned arms.

    Runs on whichever rank layout holds the arms: each position is independent,
    so a context-parallel shard yields that shard's rows of the full result.

    Args:
        arms: Anchor and positive hidden states ``[B, S, H_i]``, in that order.
        heads: This rank's vocabulary shard of the anchor and positive LM heads.
        negative_logits: The negative teacher's logits ``[B, S, V_local]`` on the
            same vocabulary shard.

    Returns:
        ``(support, target_logprobs)``, int64 and float32 ``[B, S, top_k]``,
        identical on every rank of ``vocab_parallel_group``.
    """
    _validate_arms_and_heads(arms, heads, negative_logits)
    if chunk_size < 1:
        raise ValueError("W2S-OPD projection chunk_size must be positive.")
    supports, targets = [], []
    for start in range(0, negative_logits.shape[1], chunk_size):
        end = min(start + chunk_size, negative_logits.shape[1])
        support, target = w2s_proxy_support(
            _project_proxy(arms, heads, negative_logits, alpha, start, end),
            top_k=top_k,
            vocab_size=vocab_size,
            vocab_parallel_group=vocab_parallel_group,
        )
        supports.append(support)
        targets.append(target)
    return torch.cat(supports, dim=1), torch.cat(targets, dim=1)


@torch.no_grad()
def w2s_packed_proxy_support(
    arm_rows: tuple[torch.Tensor, torch.Tensor],
    negative_logits: torch.Tensor,
    heads: tuple[torch.Tensor, torch.Tensor],
    *,
    lengths: list[int],
    spans: list[tuple[int, int]],
    alpha: float,
    top_k: int,
    vocab_size: int,
    chunk_size: int,
    payload_dtype: torch.dtype,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
    context_parallel_group: torch.distributed.ProcessGroup | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Proxy support for one packed, context-parallel negative-teacher microbatch.

    Args:
        arm_rows: Anchor and positive hidden states ``[sum(lengths), H]``: each
            row's true-length positions back to back, identical on every rank.
        negative_logits: This rank's CP-local packed negative-teacher logits
            ``[1, S_local, V_local]``, as the model's output layer produced them.
        heads: This rank's vocabulary shard of the anchor and positive heads.
        lengths: True row lengths, in packing order.
        spans: Each row's padded ``[start, end)`` in the full packed sequence.
        payload_dtype: The anchor and positive arms are rounded to this dtype
            before projection, as their rows store it.

    Returns:
        Per row ``(support [length, K] int64, target_logprobs [length, K]
        float32)``, identical on every TP and CP rank.

    Raises:
        FloatingPointError: On every rank alike if any target is nonfinite, so
            no rank is left waiting in a collective.
    """
    cp_size = (
        1
        if context_parallel_group is None
        else torch.distributed.get_world_size(context_parallel_group)
    )
    cp_rank = 0 if cp_size == 1 else torch.distributed.get_rank(context_parallel_group)
    if len(lengths) != len(spans) or any(
        length > end - start or (end - start) % (2 * cp_size if cp_size > 1 else 1)
        for length, (start, end) in zip(lengths, spans)
    ):
        raise ValueError("W2S proxy rows do not fit their packed sequence spans.")
    local_length = negative_logits.shape[1]
    arms = []
    for rows, head in zip(arm_rows, heads, strict=True):
        if tuple(rows.shape) != (sum(lengths), head.shape[1]):
            raise ValueError(
                "W2S arm hidden states do not match the microbatch rows: "
                f"{tuple(rows.shape)} vs {(sum(lengths), head.shape[1])}."
            )
        # Rebuild this rank's packed CP layout: each padded span is split into
        # the load-balanced 2 x CP chunks the model consumed.
        local = rows.new_zeros((1, local_length, rows.shape[1]))
        offset = 0
        for length, (start, end) in zip(lengths, spans):
            span = rows.new_zeros((1, end - start, rows.shape[1]))
            span[0, :length] = rows[offset : offset + length]
            offset += length
            span = _get_tokens_on_this_cp_rank(span, cp_rank, cp_size, seq_dim=1)
            local[:, start // cp_size : end // cp_size] = span
        arms.append(local.to(payload_dtype))
    support, target = w2s_compact_proxy(
        (arms[0], arms[1]),
        heads,
        negative_logits,
        alpha=alpha,
        top_k=top_k,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        vocab_parallel_group=vocab_parallel_group,
    )
    rows_out = []
    nonfinite = 0
    for length, (start, end) in zip(lengths, spans):
        seq_support = support[:, start // cp_size : end // cp_size]
        seq_target = target[:, start // cp_size : end // cp_size]
        if cp_size > 1:
            seq_support = allgather_cp_sharded_tensor(
                seq_support, context_parallel_group, seq_dim=1
            )
            seq_target = allgather_cp_sharded_tensor(
                seq_target, context_parallel_group, seq_dim=1
            )
        seq_support, seq_target = seq_support[0, :length], seq_target[0, :length]
        nonfinite += int((~torch.isfinite(seq_target)).sum().item())
        rows_out.append((seq_support, seq_target))
    if nonfinite:
        raise FloatingPointError(
            f"W2S proxy produced {nonfinite} nonfinite target log-probabilities."
        )
    return rows_out


def _cp_shard_aligned(
    tensor: torch.Tensor,
    local_length: int,
    context_parallel_group: torch.distributed.ProcessGroup | None,
    pad_value: float,
) -> torch.Tensor:
    """Pad a position-aligned ``[B, S, D]`` tensor to the CP window and shard it."""
    cp_size = (
        1
        if context_parallel_group is None
        else torch.distributed.get_world_size(context_parallel_group)
    )
    pad = local_length * cp_size - tensor.shape[1]
    if pad < 0:
        raise ValueError("W2S-OPD teacher payload exceeds the student sequence window.")
    if pad:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad), value=pad_value)
    if cp_size > 1:
        tensor = _get_tokens_on_this_cp_rank(
            tensor,
            torch.distributed.get_rank(context_parallel_group),
            cp_size,
            seq_dim=1,
        )
    return tensor


def _gather_cp(
    result: torch.Tensor,
    context_parallel_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    if (
        context_parallel_group is None
        or torch.distributed.get_world_size(context_parallel_group) == 1
    ):
        return result
    return allgather_cp_sharded_tensor(result, context_parallel_group, seq_dim=1)


def w2s_reverse_kl_from_support(
    logits: torch.Tensor,
    support_ids: torch.Tensor,
    target_logprobs: torch.Tensor,
    *,
    chunk_size: int,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
    context_parallel_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    """Student-side W2S loss from a frozen-side proxy support.

    Args:
        logits: Student vocabulary-parallel logits ``[B, S_local, V_local]``.
        support_ids: Proxy top-K global token ids ``[B, S, K]``, aligned to
            prediction positions (not CP sharded).
        target_logprobs: Proxy log-probabilities on that support ``[B, S, K]``.
        chunk_size: Sequence chunk bounding the fp32 logit working set.
        vocab_parallel_group: Group the student vocabulary is sharded over.
        context_parallel_group: Group the student sequence is sharded over.

    Returns:
        Per-position reverse KL ``[B, S_local * CP]``, identical on all CP ranks.
    """
    if support_ids.ndim != 3 or support_ids.shape != target_logprobs.shape:
        raise ValueError("W2S-OPD proxy support and target must both be [B, S, K].")
    if support_ids.shape[0] != logits.shape[0]:
        raise ValueError("W2S-OPD proxy support does not match the student batch.")
    if chunk_size < 1:
        raise ValueError("W2S-OPD projection chunk_size must be positive.")
    top_k = support_ids.shape[-1]
    support = _cp_shard_aligned(
        support_ids.detach().to(device=logits.device, dtype=torch.long),
        logits.shape[1],
        context_parallel_group,
        pad_value=0,
    )
    # A uniform target keeps padded positions finite; masks drop them anyway.
    target = _cp_shard_aligned(
        target_logprobs.detach().to(device=logits.device, dtype=torch.float32),
        logits.shape[1],
        context_parallel_group,
        pad_value=-math.log(top_k),
    )
    divergences = [
        w2s_reverse_kl_on_support(
            logits[:, start : start + chunk_size],
            support[:, start : start + chunk_size],
            target[:, start : start + chunk_size],
            vocab_parallel_group=vocab_parallel_group,
        )
        for start in range(0, logits.shape[1], chunk_size)
    ]
    return _gather_cp(torch.cat(divergences, dim=1), context_parallel_group)


@torch.no_grad()
def w2s_proxy_agreement_on_support(
    student_logits: torch.Tensor,
    support: torch.Tensor,
    *,
    vocab_size: int,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """How closely the student's own ranking follows a fixed proxy support.

    ``support`` holds the proxy's top-K global token ids in descending proxy
    order, so ``support[..., 0]`` is the proxy's argmax and K is its width.
    Padded vocabulary ids never enter the student's top-K. Diagnostics only.

    Returns:
        ``(top1, topk_overlap)``, float32 ``support.shape[:-1]``: 1 where the
        student's argmax is the proxy's, and ``|top-K(student) ∩ support| / K``.
    """
    width = student_logits.shape[-1]
    world, start = _vocab_shard_start(width, vocab_parallel_group)
    top_k = support.shape[-1]
    if not 1 <= top_k <= vocab_size <= width * world:
        raise ValueError(
            "W2S-OPD proxy agreement requires 1 <= K <= vocabulary <= padded "
            "vocabulary."
        )
    student = student_logits.detach().float()
    ids = torch.arange(start, start + width, device=student.device)
    student = student.masked_fill(ids >= vocab_size, float("-inf"))
    if vocab_parallel_group is None or world == 1:
        student_top = student.topk(top_k, dim=-1).indices
    else:
        _, student_top = distributed_vocab_topk(
            student,
            top_k,
            vocab_parallel_group,
            vocab_start_index=start,
            vocab_end_index=start + width,
        )
    support = support.to(device=student_top.device, dtype=torch.long)
    top1 = (student_top[..., 0] == support[..., 0]).float()
    # The student's top-K ids are distinct, so counting the ones found in the
    # support counts the intersection.
    shared = (student_top.unsqueeze(-1) == support.unsqueeze(-2)).any(-1)
    return top1, shared.sum(-1).float() / top_k


@torch.no_grad()
def w2s_proxy_agreement_from_support(
    logits: torch.Tensor,
    support_ids: torch.Tensor,
    *,
    vocab_size: int,
    chunk_size: int,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
    context_parallel_group: torch.distributed.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student-vs-proxy agreement on a frozen-side proxy support; metrics only.

    Args:
        logits: Student vocabulary-parallel logits ``[B, S_local, V_local]``.
        support_ids: Proxy top-K global token ids ``[B, S, K]`` in descending
            proxy order, aligned to prediction positions (not CP sharded).
        vocab_size: Unpadded vocabulary size; larger ids are padding.
        chunk_size: Sequence chunk bounding the fp32 student-logit working set.

    Returns:
        Per-position ``(top1, topk_overlap)``, each ``[B, S_local * CP]`` and
        identical on all CP ranks; see :func:`w2s_proxy_agreement_on_support`.
    """
    if support_ids.ndim != 3 or support_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            "W2S-OPD proxy support must be [B, S, K] for the student batch."
        )
    if chunk_size < 1:
        raise ValueError("W2S-OPD agreement chunk_size must be positive.")
    support = _cp_shard_aligned(
        support_ids.detach().to(device=logits.device, dtype=torch.long),
        logits.shape[1],
        context_parallel_group,
        pad_value=0,
    )
    top1, overlap = [], []
    for start in range(0, logits.shape[1], chunk_size):
        chunk_top1, chunk_overlap = w2s_proxy_agreement_on_support(
            logits[:, start : start + chunk_size],
            support[:, start : start + chunk_size],
            vocab_size=vocab_size,
            vocab_parallel_group=vocab_parallel_group,
        )
        top1.append(chunk_top1)
        overlap.append(chunk_overlap)
    return (
        _gather_cp(torch.cat(top1, dim=1), context_parallel_group),
        _gather_cp(torch.cat(overlap, dim=1), context_parallel_group),
    )
