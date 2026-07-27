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

"""Unit tests for single_controller_utils.utils pure helpers."""

from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data_plane import KVBatchMeta


def _meta(size: int, sequence_lengths: list[int] | None = None) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"s{i}" for i in range(size)],
        sequence_lengths=sequence_lengths,
    )


class TestSqueezeTrailingUnitDim:
    def test_squeezes_trailing_unit_dim(self) -> None:
        out = squeeze_trailing_unit_dim(torch.zeros(4, 1))
        assert out.shape == (4,)

    def test_leaves_1d_untouched(self) -> None:
        out = squeeze_trailing_unit_dim(torch.zeros(4))
        assert out.shape == (4,)

    def test_leaves_non_unit_trailing_dim(self) -> None:
        out = squeeze_trailing_unit_dim(torch.zeros(4, 3))
        assert out.shape == (4, 3)


class TestTensorField:
    def test_returns_dense_tensor(self) -> None:
        td = TensorDict({"x": torch.arange(6).reshape(2, 3)}, batch_size=[2])
        out = tensor_field(td, "x")
        assert torch.equal(out, torch.arange(6).reshape(2, 3))

    def test_pads_nested_tensor(self) -> None:
        nested = torch.nested.as_nested_tensor(
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
            layout=torch.jagged,
        )
        td = TensorDict({"x": nested}, batch_size=[2])
        out = tensor_field(td, "x")
        assert not out.is_nested
        assert out.shape == (2, 3)
        assert out[1].tolist() == [4, 5, 0]

    def test_non_tensor_raises_type_error(self) -> None:
        td = TensorDict({"x": torch.zeros(2)}, batch_size=[2], non_blocking=False)
        td.set_non_tensor("meta", ["a", "b"])
        with pytest.raises(TypeError):
            tensor_field(td, "meta")


class TestAggregateStepMetrics:
    def test_scalar_loss_and_grad_norm_tensors(self) -> None:
        result = {
            "loss": torch.tensor([1.0, 3.0]),
            "grad_norm": torch.tensor(2.0),
        }
        out = aggregate_step_metrics(result)
        assert out["loss"] == pytest.approx(2.0)
        assert out["grad_norm"] == pytest.approx(2.0)

    def test_float_loss_and_optional_scalars(self) -> None:
        out = aggregate_step_metrics({"loss": 0.5, "total_flops": 10, "num_ranks": 4})
        assert out["loss"] == pytest.approx(0.5)
        assert out["total_flops"] == pytest.approx(10.0)
        assert out["num_ranks"] == 4
        assert "grad_norm" not in out

    def test_mb_metric_reduction_rules(self) -> None:
        result = {
            "all_mb_metrics": {
                "probs_ratio_min": [0.4, 0.2, 0.9],
                "probs_ratio_max": [0.4, 0.2, 0.9],
                "lr": [0.1, 0.3],
                "some_sum_metric": [1.0, 2.0, 3.0],
            }
        }
        out = aggregate_step_metrics(result)
        assert out["probs_ratio_min"] == pytest.approx(0.2)
        assert out["probs_ratio_max"] == pytest.approx(0.9)
        assert out["lr"] == pytest.approx(0.2)
        assert out["some_sum_metric"] == pytest.approx(6.0)

    def test_min_max_all_inf_falls_back_to_neg_one(self) -> None:
        result = {
            "all_mb_metrics": {
                "probs_ratio_min": [math.inf, math.inf],
                "probs_ratio_max": [math.inf],
            }
        }
        out = aggregate_step_metrics(result)
        assert out["probs_ratio_min"] == -1.0
        assert out["probs_ratio_max"] == -1.0

    def test_moe_and_mtp_metrics_are_prefixed(self) -> None:
        result = {
            "moe_metrics": {"load_balance": [1.0, 3.0]},
            "mtp_metrics": {"acc": [2.0, 2.0]},
        }
        out = aggregate_step_metrics(result)
        assert out["moe/load_balance"] == pytest.approx(4.0)
        assert out["mtp/acc"] == pytest.approx(4.0)


class TestReduceAdvantagePumpMetrics:
    def test_reward_and_advantages_and_tokens(self) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[torch.tensor([1.0, 3.0])],
            masked_advantages=[torch.tensor([-1.0, 0.0, 2.0])],
            sequence_lengths=[4, 6],
        )
        assert out["reward"] == pytest.approx(2.0)
        assert out["advantages/mean"] == pytest.approx(1.0 / 3.0)
        assert out["advantages/max"] == pytest.approx(2.0)
        assert out["advantages/min"] == pytest.approx(-1.0)
        assert out["total_num_tokens"] == pytest.approx(10.0)

    def test_empty_advantages_tensor_yields_zeros(self) -> None:
        out = reduce_advantage_pump_metrics(
            rewards=[],
            masked_advantages=[torch.empty(0)],
            sequence_lengths=[],
        )
        assert out["advantages/mean"] == 0.0
        assert out["advantages/max"] == 0.0
        assert out["advantages/min"] == 0.0
        assert "reward" not in out
        assert "total_num_tokens" not in out

    def test_all_empty_inputs_returns_empty_dict(self) -> None:
        assert reduce_advantage_pump_metrics([], [], []) == {}


class TestFieldsForPut:
    def test_no_sequence_lengths_packs_contiguous(self) -> None:
        meta = _meta(2, sequence_lengths=None)
        out = fields_for_put(meta, {"advantages": torch.zeros(2, 3)})
        assert out.batch_size == torch.Size([2])
        assert not out["advantages"].is_nested
        assert out["advantages"].shape == (2, 3)

    def test_renests_padded_rows_by_sequence_length(self) -> None:
        meta = _meta(2, sequence_lengths=[3, 2])
        value = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        out = fields_for_put(meta, {"advantages": value})
        assert out["advantages"].is_nested
        rows = out["advantages"].unbind()
        assert rows[0].tolist() == [1.0, 2.0, 3.0]
        assert rows[1].tolist() == [4.0, 5.0]

    def test_non_matching_width_stays_contiguous(self) -> None:
        meta = _meta(2, sequence_lengths=[3, 2])
        value = torch.zeros(2, 1)
        out = fields_for_put(meta, {"scalar": value})
        assert not out["scalar"].is_nested
        assert out["scalar"].shape == (2, 1)
