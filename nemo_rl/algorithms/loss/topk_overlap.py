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

"""Student/teacher top-k overlap diagnostics for on-policy distillation."""

from typing import NamedTuple

import torch


class TopkOverlapDiagnostics(NamedTuple):
    """Per-position top-k overlap diagnostics, each ``[B, S]``.

    ``I`` is the intersection of the student's and the teacher's top-k ids,
    leaving out ids either side gives zero probability. Teacher quantities use the teacher distribution renormalized over its
    top-k, the only part of it the student side receives.
    """

    # |I| / k.
    overlap_ratio: torch.Tensor
    # Mean over I of p̄ (log q̄ - log p̄), with p and q renormalized on I; 0 when
    # I is empty. Equals -KL(p̄ || q̄) / |I|, so it is never positive.
    overlap_token_advantage: torch.Tensor
    # Student full-vocabulary mass on I.
    overlap_student_mass: torch.Tensor
    # Teacher top-k-renormalized mass on I.
    overlap_teacher_mass: torch.Tensor
    # Entropy of the student renormalized over its own top-k.
    student_entropy: torch.Tensor
    # Entropy of the teacher renormalized over its top-k.
    teacher_entropy: torch.Tensor
    # |teacher_entropy - student_entropy|.
    entropy_gap: torch.Tensor


def _renormalized_entropy(logprobs: torch.Tensor) -> torch.Tensor:
    """Entropy of the distribution renormalized over the last dimension."""
    logprobs = logprobs.log_softmax(dim=-1)
    return -torch.where(logprobs.isneginf(), 0.0, logprobs.exp() * logprobs).sum(-1)


@torch.no_grad()
def compute_topk_overlap_diagnostics(
    student_topk_logprobs: torch.Tensor,
    student_topk_ids: torch.Tensor,
    *,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> TopkOverlapDiagnostics:
    """Top-k overlap diagnostics of arXiv:2604.13016 (Sec. 2.3, App. B.1).

    In a healthy on-policy distillation run the overlap ratio rises, the
    overlap-token advantage moves toward zero, the entropy gap narrows and the
    overlap mass stays near one. Diagnostics only.

    Args:
        student_topk_logprobs: Student full-vocabulary log-probs at its own
            top-k ids ``[B, S, k]``.
        student_topk_ids: The student's top-k token ids ``[B, S, k]``; distinct
            within a position.
        teacher_topk_logprobs: Teacher log-probs renormalized over its top-k
            ``[B, S, K]``.
        teacher_topk_ids: The teacher's top-k token ids ``[B, S, K]``.

    Returns:
        The per-position diagnostics; see ``TopkOverlapDiagnostics``.
    """
    student_topk_logprobs = student_topk_logprobs.float()
    teacher_topk_logprobs = teacher_topk_logprobs.float()
    student_topk_ids = student_topk_ids.contiguous()
    teacher_topk_ids = teacher_topk_ids.to(
        device=student_topk_ids.device, dtype=student_topk_ids.dtype
    )

    # Look each student id up in the sorted teacher ids: O(k log K) per
    # position instead of a [k, K] comparison.
    sorted_teacher_ids, teacher_order = teacher_topk_ids.sort(dim=-1)
    slot = torch.searchsorted(sorted_teacher_ids, student_topk_ids).clamp_(
        max=sorted_teacher_ids.shape[-1] - 1
    )
    teacher_logprobs_at_student = teacher_topk_logprobs.gather(
        -1, teacher_order.gather(-1, slot)
    )
    # A zero-probability id is not part of either top-k.
    shared = (
        (sorted_teacher_ids.gather(-1, slot) == student_topk_ids)
        & torch.isfinite(student_topk_logprobs)
        & torch.isfinite(teacher_logprobs_at_student)
    )
    num_shared = shared.sum(-1)
    has_shared = num_shared > 0

    # A position with no shared id keeps all of its slots so the renormalization
    # stays finite; its advantage is zeroed below.
    on_support = shared | ~has_shared.unsqueeze(-1)
    student_on_overlap = torch.where(
        on_support, student_topk_logprobs, float("-inf")
    ).log_softmax(dim=-1)
    teacher_on_overlap = torch.where(
        on_support, teacher_logprobs_at_student, float("-inf")
    ).log_softmax(dim=-1)
    advantage_terms = torch.where(
        shared,
        student_on_overlap.exp() * (teacher_on_overlap - student_on_overlap),
        0.0,
    )
    overlap_token_advantage = torch.where(
        has_shared, advantage_terms.sum(-1) / num_shared.clamp(min=1), 0.0
    )

    student_mass = torch.where(shared, student_topk_logprobs.exp(), 0.0)
    teacher_mass = torch.where(shared, teacher_logprobs_at_student.exp(), 0.0)
    student_entropy = _renormalized_entropy(student_topk_logprobs)
    teacher_entropy = _renormalized_entropy(teacher_topk_logprobs)
    return TopkOverlapDiagnostics(
        overlap_ratio=num_shared.float() / student_topk_ids.shape[-1],
        overlap_token_advantage=overlap_token_advantage,
        overlap_student_mass=student_mass.sum(-1),
        overlap_teacher_mass=teacher_mass.sum(-1),
        student_entropy=student_entropy,
        teacher_entropy=teacher_entropy,
        entropy_gap=(teacher_entropy - student_entropy).abs(),
    )
