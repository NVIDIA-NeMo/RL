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

from unittest.mock import MagicMock, patch

import torch

from nemo_rl.algorithms.loss.loss_functions import DraftCrossEntropyLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@patch("nemo_rl.algorithms.loss.loss_functions.DraftCrossEntropyLossFn")
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
    draft_loss_fn = MagicMock(return_value=(draft_loss, {"draft_loss_pass_1": 2.0}))
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
    assert combined_metrics["draft_loss_pass_1"] == 2.0
    assert combined_metrics["policy_metric"] == metrics["policy_metric"]


@patch("nemo_rl.algorithms.loss.loss_functions.DraftCrossEntropyLossFn")
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
    draft_loss_fn = MagicMock(return_value=(draft_loss, {}))
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


@patch("nemo_rl.algorithms.loss.loss_functions.DraftCrossEntropyLossFn")
def test_draft_loss_wrapper_threads_draft_pass_counts(mock_draft_loss_cls):
    """The wrapper must hand the global per-pass counts to the draft loss fn."""
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    data = BatchedDataDict({})
    global_valid = torch.tensor(1)
    draft_pass_counts = torch.tensor([42.0, 17.0])

    policy_loss_fn = MagicMock(return_value=(torch.tensor(1.0), {}))
    prepare_fn = MagicMock(return_value=({}, data))
    draft_loss_fn = MagicMock(return_value=(torch.tensor(0.5), {}))
    mock_draft_loss_cls.return_value = draft_loss_fn

    wrapper = DraftLossWrapper(
        loss_fn=policy_loss_fn,
        prepare_fn=prepare_fn,
        data_dict=data,
        loss_weight=1.0,
        draft_loss_kwargs={"pass_weights": [1.0, 0.5]},
        global_draft_pass_counts=draft_pass_counts,
    )
    wrapper(
        next_token_logits=torch.randn(1, 2, 3),
        data=data,
        global_valid_seqs=global_valid,
        global_valid_toks=global_valid,
    )

    mock_draft_loss_cls.assert_called_once_with(
        vocab_parallel_group=None,
        pass_weights=[1.0, 0.5],
    )
    call_kwargs = draft_loss_fn.call_args[1]
    assert call_kwargs["global_draft_pass_counts"] is draft_pass_counts


@patch("nemo_rl.algorithms.loss.loss_functions.ChunkedDistributedCrossEntropy.apply")
def test_draft_cross_entropy_loss_uses_distributed_path_for_tp(
    mock_distributed_ce,
):
    """DraftCrossEntropyLossFn should delegate to the chunked distributed CE."""
    seq_len = 4
    # Pass 1 slices to valid_len = S - 2 = 2 positions.
    teacher_logits = torch.randn(2, seq_len, 5)
    student_logits = torch.randn(2, seq_len, 5)
    token_mask = torch.ones(2, seq_len)
    sample_mask = torch.ones(2)
    global_valid = torch.tensor(4.0)
    per_token_loss = torch.full((2, 2), 2.0)
    mock_distributed_ce.return_value = per_token_loss

    loss_fn = DraftCrossEntropyLossFn(vocab_parallel_group=MagicMock())
    loss, metrics = loss_fn(
        teacher_logits=teacher_logits,
        student_logits_by_pass=[student_logits],
        data=BatchedDataDict({"sample_mask": sample_mask, "token_mask": token_mask}),
        global_valid_seqs=global_valid,
        global_valid_toks=global_valid,
    )

    mock_distributed_ce.assert_called_once()
    student_arg, teacher_arg = mock_distributed_ce.call_args[0][:2]
    assert student_arg.shape == (2, 2, 5)
    assert teacher_arg.shape == (2, 2, 5)
    assert torch.equal(student_arg, student_logits[:, :2])
    assert torch.equal(teacher_arg, teacher_logits[:, 1:3])
    # 4 valid positions x loss 2.0 / global_valid_toks 4.0
    assert loss.item() == 2.0
    assert metrics["draft_loss_pass_1"] == 2.0
