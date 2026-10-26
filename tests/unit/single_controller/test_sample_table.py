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

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms.single_controller_utils.sample_table import (
    SampleTableConfig,
    merge_table_chunks,
    summarize_chunk,
)

NAN = float("nan")


def _chunk(**overrides):
    # Three 4-token rows; the last two positions are the response.
    args = dict(
        sample_ids=["g_g0", "g_g1", "g_g2"],
        tags=[
            {"group_id": "g", "rollout_id": "r0", "weight_version": 3},
            {"group_id": "g", "rollout_id": "r1", "weight_version": 3},
            {"group_id": "g", "weight_version": 3},
        ],
        input_lengths=torch.tensor([4, 4, 4]),
        token_mask=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]),
        sequences={
            # Prompt positions hold values that must not enter the mean.
            "generation_logprobs": torch.tensor(
                [[9.0, 9.0, -1.0, -3.0], [9.0, 9.0, NAN, -2.0], [9.0, 9.0, -1.0, -1.0]]
            ),
            "advantages": torch.tensor(
                [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, NAN, float("inf")], [0.0] * 4]
            ),
            "token_ids": torch.tensor([[1, 2, 3, 4]] * 3),
        },
        scalars={
            "rewards": torch.tensor([1.0, NAN, 0.0]),
            "sample_loss_mask": torch.tensor([1, 1, 0]),
            # Several values per sample: not a table column.
            "prompt_ids": torch.tensor([[1, 2]] * 3),
        },
    )
    return summarize_chunk(**{**args, **overrides})


def test_summary_columns_ids_scalars_and_response_means() -> None:
    chunk = _chunk()
    assert list(chunk) == [
        "sample_id",
        "group_id",
        "rollout_id",
        "weight_version",
        "input_lengths",
        "response_tokens",
        "mean_generation_logprobs",
        "mean_advantages",
        "rewards",
        "sample_loss_mask",
    ]
    assert chunk["rollout_id"] == ["r0", "r1", None]
    assert chunk["response_tokens"] == [2, 2, 0]
    # NaN and inf are skipped; nothing left (or a fully masked row) is None.
    assert chunk["mean_generation_logprobs"] == [-2.0, -2.0, None]
    assert chunk["mean_advantages"] == [0.5, None, None]
    assert chunk["rewards"] == [1.0, None, 0.0]


def test_positions_past_the_length_never_count() -> None:
    chunk = _chunk(input_lengths=torch.tensor([3, 4, 4]))
    assert chunk["response_tokens"][0] == 1
    assert chunk["mean_generation_logprobs"][0] == -1.0


def test_jagged_sequences_are_summarized_like_dense_ones() -> None:
    dense = _chunk()
    jagged = _chunk(
        sequences={
            "generation_logprobs": torch.nested.nested_tensor(
                [torch.tensor([9.0, 9.0, -1.0, -3.0])] * 3
            )
        },
        input_lengths=torch.tensor([4, 4, 4]),
    )
    assert jagged["mean_generation_logprobs"][0] == dense["mean_generation_logprobs"][0]


def test_merge_concatenates_chunks_in_arrival_order() -> None:
    first, second = _chunk(), _chunk(sample_ids=["h_g0", "h_g1", "h_g2"])
    columns, rows = merge_table_chunks([first, second])
    assert columns == list(first)
    assert [row[0] for row in rows] == ["g_g0", "g_g1", "g_g2", "h_g0", "h_g1", "h_g2"]
    assert merge_table_chunks([]) == ([], [])


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        SampleTableConfig(enabled=True, max_rows=10)
