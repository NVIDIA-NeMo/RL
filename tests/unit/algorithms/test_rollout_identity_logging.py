# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json

import torch

from nemo_rl.algorithms.grpo import _rollout_identity_log_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.logger import Logger


def test_identical_prompts_keep_source_and_group_identity_in_jsonl(tmp_path):
    # A dynamic refill can return the same source again under a distinct group.
    batch = BatchedDataDict(
        idx=[17, 17, 23, 23, 17, 17],
        prompt_group_ids=torch.tensor([[0], [0], [1], [1], [2], [2]]),
    )
    data = _rollout_identity_log_data(batch)
    data["content"] = ["Transcribe the audio"] * batch.size
    data["rewards"] = [0.2, 0.8, 0.4, 0.9, 0.1, 0.7]
    logger = Logger.__new__(Logger)
    logger.base_log_dir = str(tmp_path)
    logger.gpu_monitor = None
    logger.log_batched_dict_as_jsonl(data, "train_data_step1.jsonl")
    rows = [
        json.loads(line)
        for line in (tmp_path / "train_data_step1.jsonl").read_text().splitlines()
    ]
    assert [row["idx"] for row in rows] == list(range(6))
    assert [row["source_idx"] for row in rows] == [[17], [17], [23], [23], [17], [17]]
    assert [row["prompt_group_ids"] for row in rows] == [
        [[0]],
        [[0]],
        [[1]],
        [[1]],
        [[2]],
        [[2]],
    ]
    assert [row["rewards"] for row in rows] == [
        [0.2],
        [0.8],
        [0.4],
        [0.9],
        [0.1],
        [0.7],
    ]


def test_identity_logging_accepts_batches_without_optional_keys():
    assert _rollout_identity_log_data(BatchedDataDict(content=["text"])) == {}
