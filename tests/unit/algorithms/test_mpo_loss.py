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

from typing import Any

import pytest
import torch

from nemo_rl.algorithms.loss import MPOLossConfig, MPOLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _mpo_batch() -> tuple[torch.Tensor, BatchedDataDict]:
    # Deliberately reordered rows:
    # rejected(pair=1), chosen(pair=0), rejected(pair=0), chosen(pair=1).
    next_token_logprobs = torch.tensor(
        [
            [-2.0, -2.0],
            [-0.5, -0.5],
            [-1.5, -1.5],
            [-1.0, -1.0],
        ]
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(4, 3, dtype=torch.long),
            "reference_policy_logprobs": torch.zeros(4, 3),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0]] * 4),
            "sample_mask": torch.ones(4),
            "pair_index": torch.tensor([1, 0, 0, 1]),
            "is_chosen": torch.tensor([False, True, False, True]),
        }
    )
    return next_token_logprobs, data


def _loss_config(**overrides: Any) -> MPOLossConfig:
    config: dict[str, Any] = {
        "reference_policy_kl_penalty": 1.0,
        "preference_loss_weight": 1.0,
        "sft_loss_weight": 0.25,
        "bco_loss_weight": 0.5,
        "preference_average_log_probs": False,
        "sft_average_log_probs": False,
        "quality_average_log_probs": False,
        "reward_shift_momentum": 0.5,
        "reward_shift": 1.0,
    }
    config.update(overrides)
    return MPOLossConfig(**config)


def _call_loss(loss_fn: MPOLossFn, logprobs: torch.Tensor, data: BatchedDataDict):
    return loss_fn(
        next_token_logprobs=logprobs,
        data=data,
        global_valid_seqs=data["sample_mask"].sum(),
        global_valid_toks=data["token_mask"][:, 1:].sum(),
    )


def test_mpo_loss_is_invariant_to_pair_row_order():
    logprobs, data = _mpo_batch()
    loss_fn = MPOLossFn(_loss_config())
    reordered_loss, reordered_metrics = _call_loss(loss_fn, logprobs, data)

    order = torch.tensor([1, 2, 3, 0])
    interleaved_data = BatchedDataDict(
        {
            key: value.index_select(0, order)
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
        }
    )
    interleaved_loss, interleaved_metrics = _call_loss(
        loss_fn, logprobs.index_select(0, order), interleaved_data
    )

    torch.testing.assert_close(reordered_loss, interleaved_loss)
    for key in (
        "preference_loss",
        "bco_loss",
        "sft_loss",
        "accuracy",
        "bco_reward_sum",
        "bco_reward_count",
    ):
        assert reordered_metrics[key] == pytest.approx(interleaved_metrics[key])


def test_mpo_loss_does_not_mutate_reward_shift_on_worker_call():
    logprobs, data = _mpo_batch()
    loss_fn = MPOLossFn(_loss_config())

    _, metrics = _call_loss(loss_fn, logprobs, data)

    assert loss_fn.reward_shift == 1.0
    assert metrics["bco_reward_sum"] == pytest.approx(-10.0)
    assert metrics["bco_reward_count"] == pytest.approx(4.0)


def test_mpo_reward_shift_updates_once_from_global_statistics():
    loss_fn = MPOLossFn(_loss_config())

    updated = loss_fn.update_reward_shift(reward_sum=-10.0, reward_count=4.0)

    # momentum * old + (1 - momentum) * global batch mean
    assert updated == pytest.approx(0.5 * 1.0 + 0.5 * -2.5)
    assert loss_fn.reward_shift == pytest.approx(updated)


def test_mpo_reward_shift_ignores_empty_batch():
    loss_fn = MPOLossFn(_loss_config())
    assert loss_fn.update_reward_shift(0.0, 0.0) == 1.0


def test_mpo_rejects_malformed_pair_metadata():
    logprobs, data = _mpo_batch()
    data["pair_index"] = torch.tensor([1, 0, 2, 1])

    with pytest.raises(ValueError, match="exactly one chosen and one rejected"):
        _call_loss(MPOLossFn(_loss_config()), logprobs, data)


def test_mpo_validates_reward_shift_momentum():
    with pytest.raises(ValueError, match="reward_shift_momentum"):
        MPOLossFn(_loss_config(reward_shift_momentum=1.0))
