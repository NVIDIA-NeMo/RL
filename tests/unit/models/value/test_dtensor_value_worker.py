# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for DTensor PPO value temporal alignment helpers."""

from typing import Any

import torch

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.value.temporal_shift import (
    RightShiftLossWrapper,
    right_shift_values,
)


def test_right_shift_values_aligns_value_predictions_to_state_tokens():
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    shifted = right_shift_values(values)

    expected = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 5.0, 6.0, 7.0],
        ]
    )
    torch.testing.assert_close(shifted, expected)
    assert shifted.shape == values.shape


def test_right_shift_loss_wrapper_shifts_logits_and_delegates_attributes():
    class RecordingLoss:
        loss_type = LossType.TOKEN_LEVEL
        input_type = LossInputType.LOGIT
        aggregation_type = "token_mean"

        def __init__(self):
            self.seen_logits = None

        def __call__(
            self,
            data: BatchedDataDict,
            global_valid_seqs: torch.Tensor,
            global_valid_toks: torch.Tensor,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            logits = kwargs["logits"]
            self.seen_logits = logits
            loss = logits.sum()
            return loss, {"loss": loss}

    inner = RecordingLoss()
    wrapper = RightShiftLossWrapper(inner)
    logits = torch.tensor([[10.0, 20.0, 30.0]])

    result, metrics = wrapper(
        BatchedDataDict({}),
        torch.tensor(1),
        torch.tensor(3),
        logits=logits,
    )

    expected_logits = torch.tensor([[0.0, 10.0, 20.0]])
    torch.testing.assert_close(inner.seen_logits, expected_logits)
    torch.testing.assert_close(result, expected_logits.sum())
    torch.testing.assert_close(metrics["loss"], expected_logits.sum())
    assert wrapper.input_type == inner.input_type
    assert wrapper.aggregation_type == inner.aggregation_type
