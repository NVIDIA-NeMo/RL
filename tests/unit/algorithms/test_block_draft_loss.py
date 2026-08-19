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
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from nemo_rl.algorithms.loss.loss_functions import (
    BlockDraftLossFn,
    DSparkBlockLossFn,
    resolve_block_draft_slot_weights,
)
from nemo_rl.algorithms.loss.utils import (
    block_draft_slot_mask,
    compute_block_draft_slot_valid_counts,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import ChunkedDistributedLabelCEAndTV


def _make_block_data(batch_size=2, num_anchors=3, gamma=4, seq_len=16):
    torch.manual_seed(0)
    token_mask = torch.zeros(batch_size, seq_len)
    token_mask[:, 6:] = 1.0
    sample_mask = torch.ones(batch_size)
    anchors = torch.tensor([[5, 8, 13], [6, 10, 0]], dtype=torch.int64)
    anchor_valid = torch.tensor([[True, True, True], [True, True, False]])
    return token_mask, sample_mask, anchors, anchor_valid


def test_block_draft_slot_mask_semantics():
    token_mask, sample_mask, anchors, anchor_valid = _make_block_data()
    gamma = 4
    mask = block_draft_slot_mask(
        token_mask, sample_mask, anchors, anchor_valid, gamma=gamma
    )
    assert mask.shape == (2, 3, gamma)

    # Anchor p=5: labels at 6..9, all trained -> every slot valid.
    assert mask[0, 0].tolist() == [True, True, True, True]
    # Anchor p=13 (seq_len=16): labels at 14, 15 valid; 16, 17 out of bounds.
    assert mask[0, 2].tolist() == [True, True, False, False]
    # Invalid (dummy) anchor -> all slots masked.
    assert mask[1, 2].tolist() == [False, False, False, False]
    # sample_mask zeroes a whole row.
    sample_mask_zero = sample_mask.clone()
    sample_mask_zero[1] = 0.0
    mask_zeroed = block_draft_slot_mask(
        token_mask, sample_mask_zero, anchors, anchor_valid, gamma=gamma
    )
    assert not mask_zeroed[1].any()

    counts = compute_block_draft_slot_valid_counts(
        token_mask, sample_mask, anchors, anchor_valid, gamma=gamma
    )
    assert torch.equal(counts, mask.float().sum(dim=(0, 1)))


def test_block_draft_slot_mask_is_prefix_contiguous_across_holes():
    """Slots beyond a token_mask hole must stay invalid (a serving-time block
    never spans a user/tool span, so training must not resurrect the far
    side of the hole)."""
    seq_len = 12
    token_mask = torch.zeros(1, seq_len)
    # Labels for anchor p=2 sit at 3..8; punch a hole at 5..6:
    # per-slot validity would be [1, 1, 0, 0, 1, 1].
    token_mask[0, 3:5] = 1.0
    token_mask[0, 7:9] = 1.0
    sample_mask = torch.ones(1)
    anchors = torch.tensor([[2]], dtype=torch.int64)
    anchor_valid = torch.tensor([[True]])

    mask = block_draft_slot_mask(
        token_mask, sample_mask, anchors, anchor_valid, gamma=6
    )
    assert mask[0, 0].tolist() == [True, True, False, False, False, False]


def test_block_draft_loss_matches_naive_reference():
    torch.manual_seed(1)
    batch_size, num_anchors, gamma, seq_len, vocab = 2, 3, 4, 16, 13
    token_mask, sample_mask, anchors, anchor_valid = _make_block_data(
        batch_size, num_anchors, gamma, seq_len
    )
    teacher_logits = torch.randn(batch_size, seq_len, vocab)
    student = torch.randn(batch_size, num_anchors, gamma, vocab, requires_grad=True)
    slot_weights = [0.8**j for j in range(gamma)]

    slot_mask = block_draft_slot_mask(
        token_mask, sample_mask, anchors, anchor_valid, gamma=gamma
    ).float()
    local_counts = slot_mask.sum(dim=(0, 1))

    loss_fn = BlockDraftLossFn(vocab_parallel_group=None, slot_weights=slot_weights)
    data = BatchedDataDict(
        {
            "draft_anchor_positions": anchors,
            "draft_anchor_valid": anchor_valid,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )
    loss, metrics = loss_fn(
        teacher_logits=teacher_logits,
        student_block_logits=student,
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(100.0),
        global_draft_pass_counts=local_counts,
    )

    # Naive reference: soft CE against teacher at position p + j.
    weights = torch.tensor(slot_weights)
    numerator = torch.zeros(())
    for b in range(batch_size):
        for n in range(num_anchors):
            for j in range(gamma):
                if slot_mask[b, n, j] == 0:
                    continue
                teacher_pos = min(int(anchors[b, n]) + j, seq_len - 1)
                target = F.softmax(teacher_logits[b, teacher_pos], dim=-1)
                logprobs = F.log_softmax(student[b, n, j], dim=-1)
                numerator = numerator + weights[j] * -(target * logprobs).sum()
    denominator = (weights * local_counts).sum()
    expected = numerator / denominator

    assert torch.allclose(loss, expected, atol=1e-5), (loss, expected)
    # Per-slot metrics are normalized by the per-slot counts.
    assert f"draft_loss_slot_{gamma - 1}" in metrics
    loss.backward()
    assert torch.isfinite(student.grad).all()
    # Masked slots receive no gradient.
    assert torch.all(student.grad[1, 2] == 0)


def test_block_draft_loss_rejects_bad_weight_length():
    token_mask, sample_mask, anchors, anchor_valid = _make_block_data()
    loss_fn = BlockDraftLossFn(vocab_parallel_group=None, slot_weights=[1.0])
    data = BatchedDataDict(
        {
            "draft_anchor_positions": anchors,
            "draft_anchor_valid": anchor_valid,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )
    try:
        loss_fn(
            teacher_logits=torch.randn(2, 16, 13),
            student_block_logits=torch.randn(2, 3, 4, 13),
            data=data,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(1.0),
        )
        raise AssertionError("expected ValueError for mismatched slot_weights")
    except ValueError:
        pass


@patch("nemo_rl.algorithms.loss.loss_functions.BlockDraftLossFn")
def test_draft_loss_wrapper_block_dispatch(mock_block_loss_cls):
    """draft_method=dflash routes through BlockDraftLossFn without prepare_fn."""
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    policy_loss = torch.tensor(1.0)
    draft_loss = torch.tensor(2.0)
    next_token_logits = torch.randn(1, 4, 7)
    block_logits = torch.randn(1, 2, 3, 7)
    data = BatchedDataDict({"draft_block_logits": block_logits})
    global_valid = torch.tensor(1)

    policy_loss_fn = MagicMock(return_value=(policy_loss, {}))
    prepare_fn = MagicMock()
    block_loss_fn = MagicMock(return_value=(draft_loss, {"draft_loss_slot_0": 2.0}))
    mock_block_loss_cls.return_value = block_loss_fn

    wrapper = DraftLossWrapper(
        loss_fn=policy_loss_fn,
        prepare_fn=prepare_fn,
        data_dict=data,
        loss_weight=0.5,
        draft_loss_kwargs={"slot_weights": [1.0, 0.5, 0.25]},
        draft_method="dflash",
    )
    combined_loss, metrics = wrapper(
        next_token_logits=next_token_logits,
        data=data,
        global_valid_seqs=global_valid,
        global_valid_toks=global_valid,
    )

    assert combined_loss.item() == 2.0
    assert metrics["draft_loss"] == draft_loss.item()
    prepare_fn.assert_not_called()
    mock_block_loss_cls.assert_called_once_with(
        vocab_parallel_group=None, slot_weights=[1.0, 0.5, 0.25]
    )
    call_kwargs = block_loss_fn.call_args.kwargs
    assert torch.equal(call_kwargs["student_block_logits"], block_logits)
    assert not call_kwargs["teacher_logits"].requires_grad


def test_chunked_label_ce_and_tv_matches_reference():
    """Chunked custom Function == plain-autograd hard CE + TV (values + grads)."""
    torch.manual_seed(3)
    batch, rows, vocab = 2, 11, 17
    student = torch.randn(batch, rows, vocab, dtype=torch.bfloat16)
    teacher = torch.randn(batch, rows, vocab, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (batch, rows))

    student_a = student.clone().requires_grad_(True)
    ce, tv = ChunkedDistributedLabelCEAndTV.apply(
        student_a, teacher, labels, 0, vocab, 4, None
    )
    # Uneven downstream weights exercise both grad_output paths.
    ce_w = torch.linspace(0.1, 1.0, rows).repeat(batch, 1)
    tv_w = torch.linspace(1.0, 0.2, rows).repeat(batch, 1)
    (ce * ce_w + tv * tv_w).sum().backward()

    student_b = student.clone().requires_grad_(True)
    s32 = student_b.float()
    t32 = teacher.float()
    ce_ref = F.cross_entropy(
        s32.reshape(-1, vocab), labels.reshape(-1), reduction="none"
    ).reshape(batch, rows)
    tv_ref = (F.softmax(s32, dim=-1) - F.softmax(t32, dim=-1)).abs().sum(dim=-1)
    (ce_ref * ce_w + tv_ref * tv_w).sum().backward()

    assert torch.allclose(ce, ce_ref, atol=1e-4), (ce, ce_ref)
    assert torch.allclose(tv, tv_ref, atol=1e-4)
    assert torch.allclose(student_a.grad.float(), student_b.grad.float(), atol=2e-3), (
        (student_a.grad - student_b.grad).abs().max()
    )


def test_dspark_block_loss_matches_official_reference():
    """DSparkBlockLossFn == the DeepSpec loss math (hard CE + TV + confidence)."""
    torch.manual_seed(2)
    batch_size, num_anchors, gamma, seq_len, vocab = 2, 3, 4, 16, 13
    token_mask, sample_mask, anchors, anchor_valid = _make_block_data(
        batch_size, num_anchors, gamma, seq_len
    )
    input_ids = torch.randint(0, vocab, (batch_size, seq_len))
    teacher_logits = torch.randn(batch_size, seq_len, vocab)
    student = torch.randn(batch_size, num_anchors, gamma, vocab, requires_grad=True)
    confidence_pred = torch.randn(batch_size, num_anchors, gamma, requires_grad=True)
    # Official decay exp(-j/4) at block 7 == the "exp" scheme; any weights work.
    slot_weights = [math.exp(-j / 4.0) for j in range(gamma)]
    ce_alpha, tv_alpha, conf_alpha = 0.1, 0.9, 1.0

    slot_mask = block_draft_slot_mask(
        token_mask, sample_mask, anchors, anchor_valid, gamma=gamma
    ).float()
    local_counts = slot_mask.sum(dim=(0, 1))

    loss_fn = DSparkBlockLossFn(
        vocab_parallel_group=None,
        vocab_parallel_rank=0,
        slot_weights=slot_weights,
        ce_loss_alpha=ce_alpha,
        tv_loss_alpha=tv_alpha,
        confidence_head_alpha=conf_alpha,
    )
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "draft_anchor_positions": anchors,
            "draft_anchor_valid": anchor_valid,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "draft_confidence_pred": confidence_pred,
        }
    )
    loss, metrics = loss_fn(
        teacher_logits=teacher_logits,
        student_block_logits=student,
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(100.0),
        global_draft_pass_counts=local_counts,
    )

    weights = torch.tensor(slot_weights)
    denominator = (weights * local_counts).sum()
    ce_num = torch.zeros(())
    tv_num = torch.zeros(())
    conf_num = torch.zeros(())
    for b in range(batch_size):
        for n in range(num_anchors):
            for j in range(gamma):
                if slot_mask[b, n, j] == 0:
                    continue
                teacher_pos = min(int(anchors[b, n]) + j, seq_len - 1)
                label_pos = min(int(anchors[b, n]) + 1 + j, seq_len - 1)
                label = input_ids[b, label_pos]
                ce = F.cross_entropy(
                    student.detach()[b, n, j], label.unsqueeze(0).squeeze(0)
                )
                draft_probs = F.softmax(student.detach()[b, n, j], dim=-1)
                target_probs = F.softmax(teacher_logits[b, teacher_pos], dim=-1)
                tv = (draft_probs - target_probs).abs().sum()
                accept = (1.0 - 0.5 * tv).clamp(0.0, 1.0)
                bce = F.binary_cross_entropy_with_logits(
                    confidence_pred.detach()[b, n, j], accept
                )
                ce_num = ce_num + weights[j] * ce
                tv_num = tv_num + weights[j] * tv
                conf_num = conf_num + weights[j] * bce
    expected = (
        ce_alpha * ce_num + tv_alpha * tv_num + conf_alpha * conf_num
    ) / denominator

    assert torch.allclose(loss, expected, atol=1e-5), (loss, expected)
    assert math.isclose(
        metrics["dspark_ce_loss"], float((ce_num / denominator)), rel_tol=1e-4
    )
    assert math.isclose(
        metrics["dspark_tv_loss"], float((tv_num / denominator)), rel_tol=1e-4
    )
    for key in (
        "dspark_confidence_loss",
        "dspark_confidence_abs_error",
        "dspark_confidence_bias",
        "dspark_tau_probabilistic",
        "accept_rate_slot_0",
        f"draft_loss_slot_{gamma - 1}",
    ):
        assert key in metrics
    assert metrics["dspark_tau_probabilistic"] >= 1.0
    assert 0.0 <= metrics["accept_rate_slot_0"] <= 1.0

    loss.backward()
    assert torch.isfinite(student.grad).all()
    # Masked slots receive no gradient (block [1, 2] is a dummy anchor).
    assert torch.all(student.grad[1, 2] == 0)
    assert torch.all(confidence_pred.grad[1, 2] == 0)
    assert torch.isfinite(confidence_pred.grad).all()
    assert confidence_pred.grad.abs().sum() > 0


@patch("nemo_rl.algorithms.loss.loss_functions.DSparkBlockLossFn")
def test_draft_loss_wrapper_dspark_dispatch(mock_dspark_loss_cls):
    """draft_method=dspark routes through DSparkBlockLossFn with the alphas."""
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    data = BatchedDataDict({"draft_block_logits": torch.randn(1, 2, 3, 7)})
    dspark_loss_fn = MagicMock(return_value=(torch.tensor(2.0), {}))
    mock_dspark_loss_cls.return_value = dspark_loss_fn

    wrapper = DraftLossWrapper(
        loss_fn=MagicMock(return_value=(torch.tensor(1.0), {})),
        prepare_fn=MagicMock(),
        data_dict=data,
        loss_weight=1.0,
        draft_loss_kwargs={
            "slot_weights": [1.0, 0.5, 0.25],
            "ce_loss_alpha": 0.1,
            "tv_loss_alpha": 0.9,
            "confidence_head_alpha": 1.0,
        },
        draft_method="dspark",
    )
    wrapper(
        next_token_logits=torch.randn(1, 4, 7),
        data=data,
        global_valid_seqs=torch.tensor(1),
        global_valid_toks=torch.tensor(1),
    )
    mock_dspark_loss_cls.assert_called_once_with(
        vocab_parallel_group=None,
        vocab_parallel_rank=None,
        slot_weights=[1.0, 0.5, 0.25],
        ce_loss_alpha=0.1,
        tv_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    )


def test_resolve_block_draft_slot_weights_schemes():
    assert resolve_block_draft_slot_weights(None, 4) == [1.0] * 4
    assert resolve_block_draft_slot_weights("uniform", 3) == [1.0] * 3

    # gamma=15 <-> block 16: paper gamma_d = 7, so w_1 = e^{-1/7}, w_14 = e^{-2}.
    weights = resolve_block_draft_slot_weights("exp", 15)
    assert len(weights) == 15
    assert weights[0] == 1.0
    assert math.isclose(weights[1], math.exp(-1 / 7))
    assert math.isclose(weights[14], math.exp(-2))
    assert all(a > b for a, b in zip(weights, weights[1:]))

    # Tabulated block sizes: b8 -> gamma_d 4, b10 -> gamma_d 5.
    assert math.isclose(resolve_block_draft_slot_weights("exp", 7)[1], math.exp(-1 / 4))
    assert math.isclose(resolve_block_draft_slot_weights("exp", 9)[1], math.exp(-1 / 5))
    # Interpolated (b9 -> 4.5) and extrapolated (b22 -> 9) block sizes.
    assert math.isclose(
        resolve_block_draft_slot_weights("exp", 8)[1], math.exp(-1 / 4.5)
    )
    assert math.isclose(
        resolve_block_draft_slot_weights("exp", 21)[1], math.exp(-1 / 9)
    )


def test_resolve_block_draft_slot_weights_rejects_unknown_scheme():
    try:
        resolve_block_draft_slot_weights("linear", 8)
    except ValueError as err:
        assert "loss_weighting" in str(err)
    else:
        raise AssertionError("expected ValueError for unknown scheme")
