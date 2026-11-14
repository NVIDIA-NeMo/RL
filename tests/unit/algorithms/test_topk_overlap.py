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

import math

import pytest
import torch

from nemo_rl.algorithms.loss.topk_overlap import compute_topk_overlap_diagnostics


def _heads(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int
) -> dict[str, torch.Tensor]:
    """Build the diagnostic inputs the way the distillation path does."""
    student_logprobs = student_logits.log_softmax(dim=-1)
    student_topk_logprobs, student_topk_ids = student_logprobs.topk(k, dim=-1)
    teacher_topk_logits, teacher_topk_ids = teacher_logits.topk(k, dim=-1)
    return {
        "student_topk_logprobs": student_topk_logprobs,
        "student_topk_ids": student_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logits.log_softmax(dim=-1),
        "teacher_topk_ids": teacher_topk_ids,
    }


def _diagnostics(heads: dict[str, torch.Tensor]):
    return compute_topk_overlap_diagnostics(
        heads["student_topk_logprobs"],
        heads["student_topk_ids"],
        teacher_topk_logprobs=heads["teacher_topk_logprobs"],
        teacher_topk_ids=heads["teacher_topk_ids"],
    )


def _reference(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int
) -> dict[str, float]:
    """arXiv:2604.13016 Eqs. 6-10 for one position, written out with sets."""
    p = student_logits.double().softmax(dim=-1).tolist()
    student_ids = set(student_logits.topk(k).indices.tolist())
    teacher_ids = teacher_logits.topk(k).indices.tolist()
    teacher_top = teacher_logits.double()[teacher_ids].softmax(dim=-1).tolist()
    q = dict(zip(teacher_ids, teacher_top))
    shared = student_ids & set(teacher_ids)

    student_head_mass = sum(p[v] for v in student_ids)
    student_entropy = -sum(
        p[v] / student_head_mass * math.log(p[v] / student_head_mass)
        for v in student_ids
    )
    teacher_entropy = -sum(q_v * math.log(q_v) for q_v in q.values())
    advantage = 0.0
    if shared:
        p_shared = sum(p[v] for v in shared)
        q_shared = sum(q[v] for v in shared)
        advantage = sum(
            p[v] / p_shared * (math.log(q[v] / q_shared) - math.log(p[v] / p_shared))
            for v in shared
        ) / len(shared)
    return {
        "overlap_ratio": len(shared) / k,
        "overlap_token_advantage": advantage,
        "overlap_student_mass": sum(p[v] for v in shared),
        "overlap_teacher_mass": sum(q[v] for v in shared),
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "entropy_gap": abs(teacher_entropy - student_entropy),
    }


@pytest.mark.parametrize("seed", range(4))
def test_topk_overlap_matches_the_set_reference(seed):
    torch.manual_seed(seed)
    vocab_size, k = 12, 4
    student_logits = torch.randn(2, 5, vocab_size) * 2
    # Correlated with the student so most positions overlap only partially.
    teacher_logits = student_logits + torch.randn(2, 5, vocab_size) * 1.5

    diagnostics = _diagnostics(_heads(student_logits, teacher_logits, k))

    ratios = set()
    for b in range(2):
        for s in range(5):
            expected = _reference(student_logits[b, s], teacher_logits[b, s], k)
            ratios.add(expected["overlap_ratio"])
            for name, value in expected.items():
                actual = getattr(diagnostics, name)[b, s].item()
                assert actual == pytest.approx(value, abs=1e-5), (name, b, s)
    assert ratios - {0.0, 1.0}, "the inputs should exercise a partial overlap"


def test_topk_overlap_of_identical_distributions():
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 10)

    diagnostics = _diagnostics(_heads(logits, logits.clone(), k=4))

    torch.testing.assert_close(diagnostics.overlap_ratio, torch.ones(3, 6))
    torch.testing.assert_close(
        diagnostics.overlap_token_advantage, torch.zeros(3, 6), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        diagnostics.entropy_gap, torch.zeros(3, 6), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(diagnostics.overlap_teacher_mass, torch.ones(3, 6))
    torch.testing.assert_close(
        diagnostics.overlap_student_mass,
        logits.softmax(dim=-1).topk(4).values.sum(-1),
    )


def test_topk_overlap_of_disjoint_heads_is_zero_and_finite():
    student_logits = torch.tensor([[[5.0, 4.0, 3.0, 0.0, 0.0, 0.0]]])
    teacher_logits = torch.tensor([[[0.0, 0.0, 0.0, 5.0, 4.0, 3.0]]])

    diagnostics = _diagnostics(_heads(student_logits, teacher_logits, k=3))

    for name in (
        "overlap_ratio",
        "overlap_token_advantage",
        "overlap_student_mass",
        "overlap_teacher_mass",
    ):
        assert getattr(diagnostics, name).item() == 0.0, name
    assert all(torch.isfinite(value).all() for value in diagnostics)


def test_topk_overlap_over_the_whole_vocabulary_is_exact():
    torch.manual_seed(1)
    student_logits = torch.randn(2, 3, 7)
    teacher_logits = torch.randn(2, 3, 7)

    diagnostics = _diagnostics(_heads(student_logits, teacher_logits, k=7))

    def entropy(logits: torch.Tensor) -> torch.Tensor:
        logprobs = logits.log_softmax(dim=-1)
        return -(logprobs.exp() * logprobs).sum(-1)

    torch.testing.assert_close(diagnostics.overlap_ratio, torch.ones(2, 3))
    torch.testing.assert_close(diagnostics.overlap_student_mass, torch.ones(2, 3))
    torch.testing.assert_close(diagnostics.overlap_teacher_mass, torch.ones(2, 3))
    torch.testing.assert_close(diagnostics.student_entropy, entropy(student_logits))
    torch.testing.assert_close(diagnostics.teacher_entropy, entropy(teacher_logits))
    # With I the whole vocabulary the advantage is -KL(p || q) / V.
    kl = (
        student_logits.softmax(dim=-1)
        * (student_logits.log_softmax(dim=-1) - teacher_logits.log_softmax(dim=-1))
    ).sum(-1)
    torch.testing.assert_close(diagnostics.overlap_token_advantage, -kl / 7)


def test_topk_overlap_ignores_zero_probability_ids():
    # Two of the student's four top-k slots have zero probability; one of them
    # is also in the teacher's top-k, where it must not count as shared.
    student_logits = torch.tensor([[[2.0, 1.0, float("-inf"), float("-inf"), -5.0]]])
    teacher_logits = torch.tensor([[[1.0, -3.0, 2.0, -9.0, 0.5]]])
    heads = _heads(student_logits, teacher_logits, k=4)
    assert heads["student_topk_logprobs"].isneginf().sum() == 1
    heads["student_topk_logprobs"][0, 0, 2] = float("-inf")

    diagnostics = _diagnostics(heads)

    assert all(torch.isfinite(value).all() for value in diagnostics)
    shared_ids = set(heads["student_topk_ids"][0, 0, :2].tolist()) & set(
        heads["teacher_topk_ids"][0, 0].tolist()
    )
    assert diagnostics.overlap_ratio.item() == len(shared_ids) / 4


def test_topk_overlap_of_a_padded_position_is_finite():
    # Padding positions carry zero teacher logits at id 0 in every slot.
    torch.manual_seed(2)
    heads = _heads(torch.randn(1, 2, 9), torch.randn(1, 2, 9), k=3)
    heads["teacher_topk_ids"][0, 1] = 0
    heads["teacher_topk_logprobs"][0, 1] = torch.zeros(3).log_softmax(dim=-1)

    diagnostics = _diagnostics(heads)

    assert all(torch.isfinite(value).all() for value in diagnostics)
    assert diagnostics.overlap_ratio[0, 1].item() <= 1 / 3
