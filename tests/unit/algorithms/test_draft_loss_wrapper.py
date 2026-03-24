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
