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

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import DraftCrossEntropyLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

DRAFT_STEP_PAYLOAD_KEY = "_draft_step_payload"


def _mock_step_state_without_megatron() -> tuple[types.ModuleType, types.ModuleType]:
    module = types.ModuleType("nemo_rl.models.megatron.draft.step_state")
    module.DRAFT_STEP_PAYLOAD_KEY = DRAFT_STEP_PAYLOAD_KEY
    module.DraftStepState = MagicMock()
    package = types.ModuleType("nemo_rl.models.megatron.draft")
    package.step_state = module
    return package, module


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


@pytest.mark.parametrize("output_key", ["dflash_output", "dspark_output"])
def test_draft_loss_wrapper_hides_transient_provider_output_from_policy_loss(
    output_key: str,
) -> None:
    """Packed policy slicing must not index provider-owned transient outputs."""
    from nemo_rl.algorithms.loss.draft import DraftLossStats
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    provider_output = object()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            output_key: provider_output,
        }
    )

    def policy_loss(
        next_token_logits,
        policy_data,
        global_valid_seqs,
        global_valid_toks,
        **kwargs,
    ):
        del next_token_logits, global_valid_seqs, global_valid_toks, kwargs
        assert output_key not in policy_data
        policy_data.slice(0, 1)
        return torch.tensor(3.0), {}

    provider = MagicMock()
    provider.config.speculator_type = output_key.removesuffix("_output")
    provider.loss_stats.return_value = DraftLossStats(
        numerators=torch.tensor([2.0]),
        counts=torch.tensor([1.0]),
        weights=torch.ones(1),
    )
    wrapper = DraftLossWrapper(
        loss_fn=policy_loss,
        prepare_fn=MagicMock(),
        data_dict=data,
        loss_weight=0.5,
        draft_provider=provider,
        draft_normalization_counts=torch.tensor([1.0]),
    )

    combined_loss, _ = wrapper(
        next_token_logits=torch.randn(1, 2, 3),
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
    )

    assert combined_loss.item() == pytest.approx(4.0)
    assert data[output_key] is provider_output
    assert provider.loss_stats.call_args.kwargs["data"] is data


@patch("nemo_rl.algorithms.loss.loss_functions.streaming_vocab_parallel_soft_ce")
def test_draft_cross_entropy_loss_uses_streaming_path(
    mock_streaming_ce,
):
    """DraftCrossEntropyLossFn should consume one-bin streaming statistics."""
    teacher_logits = torch.randn(2, 3, 5)
    student_logits = torch.randn(2, 3, 5)
    token_mask = torch.ones(2, 3)
    sample_mask = torch.ones(2)
    global_valid = torch.tensor(6.0)
    stats = MagicMock()
    stats.normalized.return_value = torch.tensor(2.0)
    mock_streaming_ce.return_value = stats

    loss_fn = DraftCrossEntropyLossFn(vocab_parallel_group=MagicMock())
    loss = loss_fn(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        token_mask=token_mask,
        data=BatchedDataDict({"sample_mask": sample_mask}),
        global_valid_seqs=global_valid,
        global_valid_toks=global_valid,
    )

    mock_streaming_ce.assert_called_once()
    call_kwargs = mock_streaming_ce.call_args.kwargs
    assert call_kwargs["student_logits"] is student_logits
    assert call_kwargs["teacher_logits"] is teacher_logits
    assert call_kwargs["token_chunk_size"] == 4096
    assert call_kwargs["tp_group"] is loss_fn.vocab_parallel_group
    torch.testing.assert_close(
        call_kwargs["mask"],
        token_mask * sample_mask.unsqueeze(-1),
    )
    stats.normalized.assert_called_once()
    assert loss.item() == 2.0


@patch("nemo_rl.algorithms.loss.wrapper.DraftCrossEntropyLossFn")
def test_draft_loss_wrapper_defers_raw_stats_for_split_step(
    mock_draft_loss_cls,
) -> None:
    """Split loss keeps differentiable raw sums and emits detached counts."""
    from nemo_rl.algorithms.loss.draft import DraftLossStats
    from nemo_rl.algorithms.loss.wrapper import DraftLossWrapper

    policy_loss = torch.tensor(5.0)
    numerator = torch.tensor([12.0], requires_grad=True)
    stats = DraftLossStats(
        numerators=numerator,
        counts=torch.tensor([3.0]),
        weights=torch.ones(1),
    )
    draft_loss_fn = MagicMock()
    draft_loss_fn.loss_stats.return_value = stats
    mock_draft_loss_cls.return_value = draft_loss_fn
    data = BatchedDataDict({})
    wrapper = DraftLossWrapper(
        loss_fn=MagicMock(return_value=(policy_loss, {})),
        prepare_fn=MagicMock(return_value=({}, data)),
        data_dict=data,
        loss_weight=0.5,
        defer_normalization=True,
    )

    payload = object()
    draft_package, step_state_module = _mock_step_state_without_megatron()
    step_state_module.DraftStepState.metric_payload.return_value = payload
    with patch.dict(
        sys.modules,
        {
            "nemo_rl.models.megatron.draft": draft_package,
            "nemo_rl.models.megatron.draft.step_state": step_state_module,
        },
    ):
        combined_loss, metrics = wrapper(
            next_token_logits=torch.randn(1, 2, 3),
            data=data,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(1.0),
        )

    assert combined_loss.item() == pytest.approx(11.0)
    assert metrics["draft_loss"] == pytest.approx(12.0)
    assert metrics[DRAFT_STEP_PAYLOAD_KEY] is payload
    step_state_module.DraftStepState.metric_payload.assert_called_once_with(stats)
    draft_loss_fn.assert_not_called()
