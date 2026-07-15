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

from argparse import Namespace

import pytest
import torch

from tools import nixl_elastic_rollout_demo as demo


def test_parse_sequence_accepts_comma_separated_counts():
    assert demo._parse_sequence("1, 3,2,,4") == [1, 3, 2, 4]


@pytest.mark.parametrize("value", ["", "0", "1,-1"])
def test_parse_sequence_rejects_empty_or_non_positive_counts(value):
    with pytest.raises(ValueError):
        demo._parse_sequence(value)


def test_phase_delta_adds_and_removes_ranks_for_contiguous_target():
    assert demo._phase_delta(active_ranks=[0, 2], target_count=3) == ([1], [])
    assert demo._phase_delta(active_ranks=[0, 1, 2, 3], target_count=2) == (
        [],
        [2, 3],
    )


def test_build_engine_kwargs_uses_current_checkpoint_engine_schema():
    args = Namespace(
        device="cuda",
        nixl_backend_name="UCX",
        ucx_error_handling_mode="peer",
    )

    assert demo._build_engine_kwargs(args) == {
        "device": "cuda",
        "backend_name": "UCX",
        "backend_init_params": {"ucx_error_handling_mode": "peer"},
        "cleanup_after_load": False,
    }


def test_tensor_summary_records_shape_dtype_and_sum():
    weights = [
        ("a", torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)),
        ("b", torch.tensor([5, 6], dtype=torch.int64)),
    ]

    assert demo._tensor_summary(weights) == {
        "a": {"shape": [2, 2], "dtype": "torch.int32", "sum": 10},
        "b": {"shape": [2], "dtype": "torch.int64", "sum": 11},
    }


def test_validate_results_accepts_rank_matched_results():
    expected = {1: {"weight": {"shape": [1], "dtype": "torch.int32", "sum": 7}}}

    demo._validate_results(
        expected_by_rank=expected,
        rollout_results=[{"rank": 1, "received": expected[1]}],
    )


def test_validate_results_rejects_unexpected_rank():
    with pytest.raises(RuntimeError, match="unexpected rollout rank 2"):
        demo._validate_results(expected_by_rank={}, rollout_results=[{"rank": 2}])


def test_validate_results_rejects_mismatch():
    expected = {0: {"weight": {"shape": [1], "dtype": "torch.int32", "sum": 7}}}

    with pytest.raises(RuntimeError, match="Rollout result mismatch"):
        demo._validate_results(
            expected_by_rank=expected,
            rollout_results=[
                {
                    "rank": 0,
                    "received": {
                        "weight": {
                            "shape": [1],
                            "dtype": "torch.int32",
                            "sum": 8,
                        }
                    },
                }
            ],
        )
