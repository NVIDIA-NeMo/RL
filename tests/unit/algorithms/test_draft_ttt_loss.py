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

"""Index/mask/normalization tests for the multi-pass (TTT) draft loss.

Convention under test: pass-d student logits at position ``i`` predict token
``x_{i+d+1}``; the teacher is the policy's logits at position ``i + d``; the
mask is ``token_mask[i + d + 1] * sample_mask``.
"""

import torch

from nemo_rl.algorithms.loss.loss_functions import DraftCrossEntropyLossFn
from nemo_rl.algorithms.loss.utils import (
    compute_draft_pass_valid_counts,
    draft_pass_token_mask,
    prepare_loss_input,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _soft_ce(student_row: torch.Tensor, teacher_row: torch.Tensor) -> torch.Tensor:
    teacher_probs = torch.softmax(teacher_row, dim=-1)
    student_log_probs = torch.log_softmax(student_row, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1)


def _manual_draft_loss(
    teacher_logits: torch.Tensor,
    student_logits_by_pass: list[torch.Tensor],
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    pass_weights: list[float],
    denominator: float,
) -> float:
    """Direct positional transcription of the pass-d index convention."""
    batch_size, seq_len, _ = teacher_logits.shape
    total = 0.0
    for pass_index, student_logits in enumerate(student_logits_by_pass):
        ttt_pass = pass_index + 1
        for b in range(batch_size):
            for i in range(seq_len):
                target_index = i + ttt_pass + 1
                if target_index >= seq_len:
                    continue
                mask = float(token_mask[b, target_index]) * float(sample_mask[b])
                if mask == 0.0:
                    continue
                ce = _soft_ce(
                    student_logits[b, i], teacher_logits[b, i + ttt_pass]
                ).item()
                total += pass_weights[pass_index] * ce * mask
    return total / max(denominator, 1.0)


def test_draft_pass_token_mask_is_shifted_by_pass_plus_one():
    token_mask = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    assert torch.equal(draft_pass_token_mask(token_mask, 1), token_mask[:, 2:])
    assert torch.equal(draft_pass_token_mask(token_mask, 2), token_mask[:, 3:])
    assert draft_pass_token_mask(token_mask, 5).shape == (1, 0)


def test_compute_draft_pass_valid_counts_counts_shifted_masks():
    token_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    sample_mask = torch.tensor([1.0, 0.0])
    # Pass 1: token_mask[:, 2:] -> row0 [1,0,1]*1 = 2, row1 masked out = 0.
    # Pass 2: token_mask[:, 3:] -> row0 [0,1]*1 = 1, row1 = 0.
    counts = compute_draft_pass_valid_counts(token_mask, sample_mask, ttt_steps=2)
    assert counts.tolist() == [2.0, 1.0]


def test_single_pass_loss_matches_manual_soft_ce():
    torch.manual_seed(0)
    batch_size, seq_len, vocab = 2, 6, 7
    teacher = torch.randn(batch_size, seq_len, vocab)
    student = torch.randn(batch_size, seq_len, vocab)
    token_mask = torch.tensor(
        [[0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    sample_mask = torch.tensor([1.0, 1.0])
    pass_counts = compute_draft_pass_valid_counts(token_mask, sample_mask, ttt_steps=1)

    loss_fn = DraftCrossEntropyLossFn()
    loss, _ = loss_fn(
        teacher_logits=teacher,
        student_logits_by_pass=[student],
        data=BatchedDataDict({"token_mask": token_mask, "sample_mask": sample_mask}),
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
        global_draft_pass_counts=pass_counts,
    )
    expected = _manual_draft_loss(
        teacher, [student], token_mask, sample_mask, [1.0], pass_counts[0].item()
    )
    torch.testing.assert_close(loss.item(), expected, atol=1e-5, rtol=1e-5)


def test_multi_pass_loss_applies_weights_and_draft_denominator():
    torch.manual_seed(1)
    batch_size, seq_len, vocab = 2, 8, 5
    teacher = torch.randn(batch_size, seq_len, vocab)
    students = [torch.randn(batch_size, seq_len, vocab) for _ in range(3)]
    token_mask = (torch.rand(batch_size, seq_len) > 0.3).float()
    sample_mask = torch.tensor([1.0, 1.0])
    pass_weights = [1.0, 0.5, 0.25]
    pass_counts = compute_draft_pass_valid_counts(token_mask, sample_mask, ttt_steps=3)
    denominator = sum(
        w * c for w, c in zip(pass_weights, pass_counts.tolist(), strict=True)
    )

    loss_fn = DraftCrossEntropyLossFn(pass_weights=pass_weights)
    loss, metrics = loss_fn(
        teacher_logits=teacher,
        student_logits_by_pass=students,
        data=BatchedDataDict({"token_mask": token_mask, "sample_mask": sample_mask}),
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
        global_draft_pass_counts=pass_counts,
    )
    expected = _manual_draft_loss(
        teacher, students, token_mask, sample_mask, pass_weights, denominator
    )
    torch.testing.assert_close(loss.item(), expected, atol=1e-5, rtol=1e-5)
    # Per-pass metrics are pass_sum / global pass count so the driver's
    # sum-over-microbatches aggregation yields a global per-token mean.
    for ttt_pass in (1, 2, 3):
        expected_metric = _manual_draft_loss(
            teacher,
            [torch.zeros_like(students[0])] * (ttt_pass - 1) + [students[ttt_pass - 1]],
            token_mask,
            sample_mask,
            [0.0] * (ttt_pass - 1) + [1.0],
            pass_counts[ttt_pass - 1].item(),
        )
        torch.testing.assert_close(
            metrics[f"draft_loss_pass_{ttt_pass}"],
            expected_metric,
            atol=1e-5,
            rtol=1e-5,
        )


def test_masked_positions_do_not_affect_loss():
    torch.manual_seed(2)
    batch_size, seq_len, vocab = 1, 6, 5
    teacher = torch.randn(batch_size, seq_len, vocab)
    student = torch.randn(batch_size, seq_len, vocab)
    token_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    sample_mask = torch.tensor([1.0])
    data = BatchedDataDict({"token_mask": token_mask, "sample_mask": sample_mask})
    loss_fn = DraftCrossEntropyLossFn()
    kwargs = dict(
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
        global_draft_pass_counts=torch.tensor([3.0]),
    )

    loss_a, _ = loss_fn(
        teacher_logits=teacher, student_logits_by_pass=[student], data=data, **kwargs
    )

    # Pass 1 position i is masked iff token_mask[i + 2] == 0: perturb the
    # student at masked positions (i = 1, 3, 4, 5) and the loss must not move.
    student_perturbed = student.clone()
    student_perturbed[0, 1] += 100.0
    student_perturbed[0, 3:] -= 50.0
    loss_b, _ = loss_fn(
        teacher_logits=teacher,
        student_logits_by_pass=[student_perturbed],
        data=data,
        **kwargs,
    )
    torch.testing.assert_close(loss_a.item(), loss_b.item(), atol=1e-6, rtol=1e-6)


def test_out_of_bounds_pass_is_skipped():
    torch.manual_seed(3)
    seq_len = 3
    teacher = torch.randn(1, seq_len, 4)
    students = [torch.randn(1, seq_len, 4) for _ in range(3)]
    token_mask = torch.ones(1, seq_len)
    sample_mask = torch.ones(1)

    loss_fn = DraftCrossEntropyLossFn()
    loss, metrics = loss_fn(
        teacher_logits=teacher,
        student_logits_by_pass=students,
        data=BatchedDataDict({"token_mask": token_mask, "sample_mask": sample_mask}),
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
        global_draft_pass_counts=torch.tensor([1.0, 0.0, 0.0]),
    )
    # Passes 2 and 3 have no in-bounds target (S - d - 1 <= 0) and contribute 0.
    assert torch.isfinite(loss)
    assert metrics["draft_loss_pass_2"] == 0.0
    assert metrics["draft_loss_pass_3"] == 0.0
    expected = _manual_draft_loss(
        teacher, students, token_mask, sample_mask, [1.0, 1.0, 1.0], 1.0
    )
    torch.testing.assert_close(loss.item(), expected, atol=1e-5, rtol=1e-5)


def test_chunked_ce_matches_unchunked_loss_and_grads(monkeypatch):
    """Sequence-chunked soft CE must match the single-chunk result exactly.

    Covers both the loss value and the gradient w.r.t. the student logits
    (the chunked Function recomputes softmaxes in backward instead of saving
    the fp32 probability tensors).
    """
    import nemo_rl.algorithms.loss.loss_functions as loss_functions_module

    torch.manual_seed(4)
    batch_size, seq_len, vocab = 2, 11, 6
    teacher = torch.randn(batch_size, seq_len, vocab)
    students = [
        torch.randn(batch_size, seq_len, vocab, requires_grad=True) for _ in range(2)
    ]
    token_mask = (torch.rand(batch_size, seq_len) > 0.2).float()
    sample_mask = torch.ones(batch_size)
    data = BatchedDataDict({"token_mask": token_mask, "sample_mask": sample_mask})
    loss_fn = DraftCrossEntropyLossFn()
    kwargs = dict(
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
        global_draft_pass_counts=torch.tensor([4.0, 3.0]),
    )

    # Single chunk covers the whole sequence.
    monkeypatch.setattr(loss_functions_module, "DRAFT_LOSS_SEQ_CHUNK_SIZE", seq_len)
    loss_a, _ = loss_fn(
        teacher_logits=teacher,
        student_logits_by_pass=list(students),
        data=data,
        **kwargs,
    )
    grads_a = torch.autograd.grad(loss_a, students, retain_graph=False)

    # Uneven multi-chunk split (11 = 3 + 3 + 3 + 2).
    monkeypatch.setattr(loss_functions_module, "DRAFT_LOSS_SEQ_CHUNK_SIZE", 3)
    loss_b, _ = loss_fn(
        teacher_logits=teacher,
        student_logits_by_pass=list(students),
        data=data,
        **kwargs,
    )
    grads_b = torch.autograd.grad(loss_b, students)

    torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)
    for grad_a, grad_b in zip(grads_a, grads_b):
        torch.testing.assert_close(grad_a, grad_b, atol=1e-6, rtol=1e-6)


def test_chunked_ce_grad_matches_plain_autograd():
    """Hand-rolled chunked backward vs plain torch autograd on the same math."""
    from nemo_rl.distributed.model_utils import ChunkedDistributedCrossEntropy

    torch.manual_seed(5)
    student = torch.randn(1, 9, 5, requires_grad=True)
    teacher = torch.randn(1, 9, 5)
    weights = torch.rand(1, 9)

    per_token = ChunkedDistributedCrossEntropy.apply(student, teacher, 4, None, False)
    (per_token * weights).sum().backward()

    student_ref = student.detach().clone().requires_grad_()
    ce_ref = -(
        torch.softmax(teacher, dim=-1) * torch.log_softmax(student_ref, dim=-1)
    ).sum(-1)
    (ce_ref * weights).sum().backward()

    torch.testing.assert_close(student.grad, student_ref.grad, atol=1e-6, rtol=1e-6)


def test_prepare_loss_input_keeps_teacher_unshifted_and_detached():
    logits = torch.randn(1, 5, 6, requires_grad=True)
    student = torch.randn(1, 5, 6)
    data = BatchedDataDict(
        {
            "student_logits_by_pass": [student, student],
            "token_mask": torch.ones(1, 5),
            "sample_mask": torch.ones(1),
        }
    )
    loss_fn = DraftCrossEntropyLossFn()
    loss_input, _ = prepare_loss_input(logits=logits, data=data, loss_fn=loss_fn)

    assert torch.equal(loss_input["teacher_logits"], logits.detach())
    assert not loss_input["teacher_logits"].requires_grad
    assert len(loss_input["student_logits_by_pass"]) == 2


def test_prepare_loss_input_falls_back_to_single_pass_key():
    logits = torch.randn(1, 4, 6)
    student = torch.randn(1, 4, 6)
    data = BatchedDataDict(
        {
            "student_logits": student,
            "token_mask": torch.ones(1, 4),
            "sample_mask": torch.ones(1),
        }
    )
    loss_fn = DraftCrossEntropyLossFn()
    loss_input, _ = prepare_loss_input(logits=logits, data=data, loss_fn=loss_fn)

    assert len(loss_input["student_logits_by_pass"]) == 1
    assert loss_input["student_logits_by_pass"][0] is student
