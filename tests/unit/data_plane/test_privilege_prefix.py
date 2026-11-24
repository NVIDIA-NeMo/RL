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

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.privilege_prefix import (
    critic_view_meta,
    policy_view_meta,
    prepend_privilege_prefix,
    values_to_policy_layout,
)
from nemo_rl.data_plane.schema import (
    GROUP_ID_TAG,
    PRIVILEGE_KEY_TAG,
    PRIVILEGE_PREFIX_LEN_TAG,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _meta(group_id, n, lengths):
    return KVBatchMeta(
        partition_id="p",
        task_name="train",
        sample_ids=[f"{group_id}_g{i}" for i in range(n)],
        fields=["input_ids"],
        sequence_lengths=list(lengths),
        extra_info={},
        tags=[{"weight_version": 0, GROUP_ID_TAG: group_id} for _ in range(n)],
    )


def test_prepend_and_unshift_round_trip():
    meta = _meta("g", 2, [3, 2])
    meta.tags[0].update({PRIVILEGE_KEY_TAG: "a", PRIVILEGE_PREFIX_LEN_TAG: 2})
    meta.tags[1].update({PRIVILEGE_KEY_TAG: "b", PRIVILEGE_PREFIX_LEN_TAG: 1})
    prefixes = {
        "a": torch.tensor([90, 91], dtype=torch.int32),
        "b": torch.tensor([80], dtype=torch.int32),
    }
    width = 6
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0]]),
            "input_lengths": torch.tensor([3, 2]),
            "token_mask": torch.tensor([[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]),
            "returns": torch.tensor([[0.0, 7.0, 7.0, 0, 0, 0], [0.0, 5.0, 0, 0, 0, 0]]),
        }
    )

    out = prepend_privilege_prefix(data, meta, prefixes, shift_fields=("returns",))

    assert out["input_ids"][0, :5].tolist() == [90, 91, 1, 2, 3]
    assert out["input_ids"][1, :3].tolist() == [80, 4, 5]
    assert out["input_lengths"].tolist() == [5, 3]
    assert out["token_mask"][0].tolist() == [0, 0, 0, 1, 1, 0]
    assert out["token_mask"][1].tolist() == [0, 0, 1, 0, 0, 0]
    assert out["returns"][0].tolist() == [0, 0, 0, 7, 7, 0]
    assert critic_view_meta(meta).sequence_lengths == [5, 3]
    assert policy_view_meta(critic_view_meta(meta)).sequence_lengths == [3, 2]

    critic_values = torch.arange(2 * width, dtype=torch.float32).reshape(2, width)
    back = values_to_policy_layout(critic_values, meta, torch.tensor([3, 2]))
    assert back[0].tolist() == [2, 3, 4, 0, 0, 0]
    assert back[1].tolist() == [7, 8, 0, 0, 0, 0]


def test_prepend_rejects_a_batch_padded_for_policy_lengths():
    meta = _meta("g", 1, [4])
    meta.tags[0].update({PRIVILEGE_KEY_TAG: "a", PRIVILEGE_PREFIX_LEN_TAG: 3})
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "input_lengths": torch.tensor([4]),
            "token_mask": torch.ones(1, 4, dtype=torch.long),
        }
    )
    with pytest.raises(ValueError, match="pad target"):
        prepend_privilege_prefix(data, meta, {"a": torch.zeros(3, dtype=torch.int32)})


def test_untagged_rows_are_rejected():
    with pytest.raises(ValueError, match="no privilege tag"):
        critic_view_meta(_meta("g", 1, [4]))
