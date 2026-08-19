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
    resolve_block_draft_slot_weights,
)
from nemo_rl.algorithms.loss.utils import (
    block_draft_slot_mask,
    compute_block_draft_slot_valid_counts,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


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
