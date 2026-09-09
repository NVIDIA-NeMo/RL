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

from functools import partial
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import DraftCrossEntropyLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@patch("nemo_rl.algorithms.loss.wrapper.DraftCrossEntropyLossFn")
def test_draft_loss_wrapper_combines_policy_and_draft_loss(mock_draft_loss_cls):
    """DraftLossWrapper should add the weighted draft loss to the policy loss."""
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    policy_loss = torch.tensor(3.0)
    draft_loss = torch.tensor(2.0)
    metrics = {"policy_metric": 1.0}
    next_token_logits = torch.randn(1, 2, 3)
    data = BatchedDataDict({})
    global_valid = torch.tensor(1)

    policy_loss_fn = MagicMock(return_value=(policy_loss, metrics.copy()))
    prepare_fn = MagicMock(return_value=({"prepared": torch.tensor(1.0)}, data))
    draft_loss_fn = MagicMock(return_value=draft_loss)
    mock_draft_loss_cls.return_value = draft_loss_fn

    wrapper = DraftLossWrapper(
        loss_fn=policy_loss_fn,
        prepare_fn=prepare_fn,
        data_dict=data,
        loss_weight=0.5,
    )

    combined_loss, combined_metrics = wrapper(
        next_token_logits=next_token_logits,
        data=data,
        global_valid_seqs=global_valid,
        global_valid_toks=global_valid,
    )

    assert combined_loss.item() == 4.0
    assert combined_metrics["draft_loss"] == draft_loss.item()
    assert combined_metrics["policy_metric"] == metrics["policy_metric"]


@patch("nemo_rl.algorithms.loss.wrapper.DraftCrossEntropyLossFn")
def test_draft_loss_wrapper_reports_draft_loss_when_weight_is_zero(
    mock_draft_loss_cls,
):
    """A zero draft-loss weight should not suppress draft-loss reporting."""
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    policy_loss = torch.tensor(5.0)
    draft_loss = torch.tensor(1.5)
    next_token_logits = torch.randn(1, 2, 3)
    data = BatchedDataDict({})
    global_valid = torch.tensor(1)

    policy_loss_fn = MagicMock(return_value=(policy_loss, {}))
    prepare_fn = MagicMock(return_value=({"prepared": torch.tensor(1.0)}, data))
    draft_loss_fn = MagicMock(return_value=draft_loss)
    mock_draft_loss_cls.return_value = draft_loss_fn

    wrapper = DraftLossWrapper(
        loss_fn=policy_loss_fn,
        prepare_fn=prepare_fn,
        data_dict=data,
        loss_weight=0.0,
    )

    combined_loss, metrics = wrapper(
        next_token_logits=next_token_logits,
        data=data,
        global_valid_seqs=global_valid,
        global_valid_toks=global_valid,
    )

    assert combined_loss.item() == policy_loss.item()
    assert metrics["draft_loss"] == draft_loss.item()


@patch("nemo_rl.algorithms.loss.loss_functions.DistributedCrossEntropy.apply")
def test_draft_cross_entropy_loss_uses_distributed_path_for_tp(
    mock_distributed_ce,
):
    """DraftCrossEntropyLossFn should delegate to DistributedCrossEntropy under TP."""
    teacher_logits = torch.randn(2, 3, 5)
    student_logits = torch.randn(2, 3, 5)
    token_mask = torch.ones(2, 3)
    sample_mask = torch.ones(2)
    global_valid = torch.tensor(6.0)
    per_token_loss = torch.full((2, 3), 2.0)
    mock_distributed_ce.return_value = per_token_loss

    loss_fn = DraftCrossEntropyLossFn(vocab_parallel_group=MagicMock())
    loss = loss_fn(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        token_mask=token_mask,
        data=BatchedDataDict({"sample_mask": sample_mask}),
        global_valid_seqs=global_valid,
        global_valid_toks=global_valid,
    )

    mock_distributed_ce.assert_called_once()
    assert loss.item() == 2.0


def test_roll_packed_seq_dim_respects_segment_boundaries():
    """The packed roll must not leak rows across padded segment boundaries."""
    from nemo_rl.algorithms.loss.utils import roll_packed_seq_dim

    cu_seqlens_padded = torch.tensor([0, 4, 8])
    tensor = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1)

    rolled = roll_packed_seq_dim(tensor, cu_seqlens_padded, seq_dim=1)

    expected = torch.tensor([2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 0.0]).reshape(1, 8, 1)
    assert torch.equal(rolled, expected)


def test_pack_rolled_draft_token_mask_clamps_inflated_last_segment():
    """The packer absorbs bin-alignment padding into the last sequence's
    effective length, so cu_seqlens can exceed the unpacked row width; only
    the real tokens may carry mask."""
    from nemo_rl.algorithms.loss.utils import pack_rolled_draft_token_mask

    token_mask = torch.tensor([[0.0, 1.0, 1.0]])  # row width 3
    sample_mask = torch.ones(1)
    cu_seqlens = torch.tensor([0, 5])  # inflated past the row width
    cu_seqlens_padded = torch.tensor([0, 8])

    packed = pack_rolled_draft_token_mask(
        token_mask, sample_mask, cu_seqlens, cu_seqlens_padded
    )

    # Left-shifted real mask at the segment start; every slot past the real
    # tokens (including the inflated tail) stays zero.
    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.equal(packed, expected)


def _zero_policy_loss(
    next_token_logits, data, global_valid_seqs, global_valid_toks, **kwargs
):
    return torch.zeros(()), {}


def _build_draft_batch(draft_vocab_size, d2t):
    """Synthetic 3-sequence batch in both unpacked [B, S] and packed layouts."""
    torch.manual_seed(17)
    vocab_size = 11
    seq_lens = [5, 7, 4]
    padded_lens = [8, 8, 4]  # per-sequence pad to a multiple of 4
    max_len = max(seq_lens)

    cu_seqlens = torch.tensor([0, 5, 12, 16])
    cu_seqlens_padded = torch.tensor([0, 8, 16, 20])
    total_packed = int(cu_seqlens_padded[-1].item())

    batch_size = len(seq_lens)
    logits = torch.zeros(batch_size, max_len, vocab_size)
    student = torch.zeros(batch_size, max_len, draft_vocab_size)
    token_mask = torch.zeros(batch_size, max_len)
    for i, seq_len in enumerate(seq_lens):
        logits[i, :seq_len] = torch.randn(seq_len, vocab_size)
        student[i, :seq_len] = torch.randn(seq_len, draft_vocab_size)
        token_mask[i, 2:seq_len] = 1.0  # first two tokens are "prompt"
    sample_mask = torch.tensor([1.0, 1.0, 0.0])  # last sequence filtered out

    packed_logits = torch.zeros(1, total_packed, vocab_size)
    packed_student = torch.zeros(1, total_packed, draft_vocab_size)
    for i, seq_len in enumerate(seq_lens):
        start = int(cu_seqlens_padded[i].item())
        packed_logits[0, start : start + seq_len] = logits[i, :seq_len]
        packed_student[0, start : start + seq_len] = student[i, :seq_len]

    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "student_logits": student,
        }
    )
    global_valid_toks = (token_mask * sample_mask.unsqueeze(-1)).sum()
    return {
        "logits": logits,
        "data": data,
        "packed_logits": packed_logits,
        "packed_student": packed_student,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens_padded,
        "global_valid_toks": global_valid_toks,
        "d2t": d2t,
        "padded_lens": padded_lens,
    }


@pytest.mark.parametrize(
    "draft_vocab_size,d2t",
    [
        (11, None),  # full-vocab draft (scratch)
        (5, torch.tensor([0, 2, 4, 5, 6])),  # pruned draft vocab (d2t mapping)
    ],
)
def test_packed_draft_loss_matches_unpacked(draft_vocab_size, d2t):
    """The packed draft CE must equal the unpacked reference on the same batch."""
    from nemo_rl.algorithms.loss.utils import prepare_loss_input
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    batch = _build_draft_batch(draft_vocab_size, d2t)

    reference_wrapper = DraftLossWrapper(
        loss_fn=_zero_policy_loss,
        prepare_fn=partial(
            prepare_loss_input, sampling_params=None, d2t=d2t, chunk_size=None
        ),
        data_dict=batch["data"],
    )
    reference_loss, reference_metrics = reference_wrapper(
        next_token_logits=batch["logits"],
        data=batch["data"],
        global_valid_seqs=None,
        global_valid_toks=batch["global_valid_toks"],
    )

    packed_wrapper = DraftLossWrapper(
        loss_fn=_zero_policy_loss,
        prepare_fn=None,
        data_dict=batch["data"],
        cu_seqlens_q=batch["cu_seqlens"],
        cu_seqlens_q_padded=batch["cu_seqlens_padded"],
        d2t=d2t,
        student_logits=batch["packed_student"],
    )
    packed_loss, packed_metrics = packed_wrapper(
        next_token_logits=batch["packed_logits"],
        data=batch["data"],
        global_valid_seqs=None,
        global_valid_toks=batch["global_valid_toks"],
    )

    assert torch.allclose(packed_loss, reference_loss, rtol=1e-5, atol=1e-6)
    assert packed_metrics["draft_loss"] == pytest.approx(
        reference_metrics["draft_loss"], rel=1e-5
    )
    assert packed_metrics["draft_loss"] > 0.0
