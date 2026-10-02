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

import pyarrow.parquet as pq
import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.experience.trajectory_logger import TrajectoryLogWriter


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
        "input_ids": torch.nested.as_nested_tensor(
            [torch.tensor([10, 11]), torch.tensor([20, 21, 22])],
            layout=torch.jagged,
        ),
        "token_mask": torch.tensor([[1, -1, 0], [1, 0, 1]]),
        "generation_logprobs": torch.tensor([[-0.5, -1.0, 0.0], [-1.5, -2.0, -2.5]]),
    }
    if with_prev_logprobs:
        fields["prev_logprobs"] = torch.tensor([[-1.0, -1.5, 0.0], [-2.0, -2.5, -3.0]])
    td = TensorDict(fields, batch_size=[2])
    writer = TrajectoryLogWriter(root_dir=str(tmp_path))
    for _ in range(2):
        writer.record(
            meta,
            td,
            advantages=torch.tensor([[0.5, 1.0, 0.0], [1.5, 2.0, 2.5]]),
            final_sample_mask=torch.tensor([1.0, 0.0]),
            step=1,
            chunk_index=0,
            values=torch.tensor([[0.25, 0.5, 0.0], [0.75, 1.0, 1.25]])
            if with_ppo
            else None,
            returns=torch.tensor([[1.5, 1.75, 0.0], [2.0, 2.25, 2.5]])
            if with_ppo
            else None,
        )

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
    expected_values = [[0.25, 0.5], [0.75, 1.0, 1.25]] if with_ppo else [None, None]
    expected_returns = [[1.5, 1.75], [2.0, 2.25, 2.5]] if with_ppo else [None, None]
    assert [row["values"] for row in rows[:2]] == expected_values
    assert [row["returns"] for row in rows[:2]] == expected_returns
    expected_prev = (
        [[-1.0, -1.5], [-2.0, -2.5, -3.0]] if with_prev_logprobs else [None, None]
    )
    assert [row["prev_logprobs"] for row in rows[:2]] == expected_prev
