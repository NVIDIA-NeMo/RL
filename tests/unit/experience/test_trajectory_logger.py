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

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.experience.trajectory_logger import TrajectoryLogger, _token_column


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("layout", ["dense", "nested", "scalar", "extra_dim"])
def test_token_column_shapes_and_float_dtypes(dtype: torch.dtype, layout: str) -> None:
    lengths = np.array([2, 1], dtype=np.int64)
    if layout == "nested":
        expected = [[0.5, -1.0], [2.0]]
        value = torch.nested.as_nested_tensor(
            [torch.tensor(row, dtype=dtype) for row in expected], layout=torch.jagged
        )
    elif layout == "scalar":
        value = torch.tensor([0.5, 2.0], dtype=dtype)
        expected = [[0.5], [2.0]]
    else:
        value = torch.tensor([[0.5, -1.0, 99.0], [2.0, 99.0, 99.0]], dtype=dtype)
        if layout == "extra_dim":
            value = value.unsqueeze(-1)
        expected = [[0.5, -1.0], [2.0]]

    result = _token_column("advantages", value, lengths)
    assert result.type == pa.list_(pa.float32())
    assert result.to_pylist() == expected


@pytest.mark.parametrize(
    ("value", "lengths", "expected"),
    [
        (None, [2, 0], [None, None]),
        (torch.empty((0, 3), dtype=torch.int64), [], []),
        (torch.empty((2, 0), dtype=torch.int64), [0, 0], [[], []]),
        (torch.tensor([[10, 11], [20, 21]]), [0, 2], [[], [20, 21]]),
        (torch.tensor([[2**31, -(2**31) - 1]]), [2], [[-(2**31), 2**31 - 1]]),
    ],
)
def test_token_column_empty_rows_and_integer_casts(
    value: torch.Tensor | None, lengths: list[int], expected: list
) -> None:
    result = _token_column("input_ids", value, np.array(lengths, dtype=np.int64))
    assert result.type == pa.list_(pa.int32())
    assert result.to_pylist() == expected


def test_token_column_preserves_mask_cast_before_normalization() -> None:
    value = torch.tensor([[-1.0, 0.0, 0.5, 1.0, 127.0, 128.0, 256.0]])
    result = _token_column("token_mask", value, np.array([7], dtype=np.int64))
    assert result.type == pa.list_(pa.int8())
    assert result.to_pylist() == [[0, 0, 0, 1, 1, 0, 0]]


@pytest.mark.parametrize("with_prev_logprobs", [True, False])
@pytest.mark.parametrize("with_ppo", [True, False])
def test_trajectory_logger_preserves_rows_and_publishes_step(
    tmp_path: Path, with_prev_logprobs: bool, with_ppo: bool
) -> None:
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["group_g0", "group_g1"],
        tags=[
            {"prompt_idx": 7, "weight_version": 3},
            {"prompt_idx": 7, "weight_version": 3},
        ],
    )
    fields = {
        "input_lengths": torch.tensor([2, 3]),
        "advantages": torch.tensor([[0.5, 1.0, 0.0], [1.5, 2.0, 2.5]]),
        "sample_mask": torch.tensor([1.0, 0.0]),
        "total_reward": torch.tensor([1.0, -0.5]),
        "mask_sample": torch.tensor([False, True]),
        "truncated": torch.tensor([True, False]),
        "input_ids": torch.nested.as_nested_tensor(
            [torch.tensor([10, 11]), torch.tensor([20, 21, 22])],
            layout=torch.jagged,
        ),
        "token_mask": torch.tensor([[1, -1, 0], [1, 0, 1]]),
        "generation_logprobs": torch.tensor([[-0.5, -1.0, 0.0], [-1.5, -2.0, -2.5]]),
    }
    if with_prev_logprobs:
        fields["prev_logprobs"] = torch.tensor([[-1.0, -1.5, 0.0], [-2.0, -2.5, -3.0]])
    if with_ppo:
        fields["values"] = torch.tensor([[0.25, 0.5, 0.0], [0.75, 1.0, 1.25]])
        fields["returns"] = torch.tensor([[1.5, 1.75, 0.0], [2.0, 2.25, 2.5]])
    td = TensorDict(fields, batch_size=[2])
    writer = TrajectoryLogger(root_dir=str(tmp_path))
    for _ in range(2):
        writer.record(meta, td, step=1, chunk_index=0)

    step_dir = tmp_path / "step=00000001"
    assert list(step_dir.glob("*.parquet")) == []
    writer.commit_step(1)
    files = list(step_dir.glob("*.parquet"))
    assert len(files) == 1
    rows = pq.ParquetFile(files[0]).read().to_pylist()
    assert len(rows) == 4
    assert [row["sample_id"] for row in rows] == [
        "group_g0",
        "group_g1",
        "group_g0",
        "group_g1",
    ]
    assert [row["prompt_idx"] for row in rows] == [7] * 4
    assert [row["input_ids"] for row in rows[:2]] == [[10, 11], [20, 21, 22]]
    assert [row["token_mask"] for row in rows[:2]] == [[1, 0], [1, 0, 1]]
    assert [row["advantages"] for row in rows[:2]] == [[0.5, 1.0], [1.5, 2.0, 2.5]]
    assert [row["final_sample_mask"] for row in rows[:2]] == [1.0, 0.0]
    assert [row["reward"] for row in rows[:2]] == [1.0, -0.5]
    assert [row["mask_sample"] for row in rows[:2]] == [False, True]
    assert [row["truncated"] for row in rows[:2]] == [True, False]
    expected_values = [[0.25, 0.5], [0.75, 1.0, 1.25]] if with_ppo else [None, None]
    expected_returns = [[1.5, 1.75], [2.0, 2.25, 2.5]] if with_ppo else [None, None]
    assert [row["values"] for row in rows[:2]] == expected_values
    assert [row["returns"] for row in rows[:2]] == expected_returns
    expected_prev = (
        [[-1.0, -1.5], [-2.0, -2.5, -3.0]] if with_prev_logprobs else [None, None]
    )
    assert [row["prev_logprobs"] for row in rows[:2]] == expected_prev
