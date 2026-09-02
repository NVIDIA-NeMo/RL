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

import json
import math
import tempfile

import pytest

from nemo_rl.data.datasets import load_oapl_dataset


def create_sample_data():
    data = [
        {
            "context": [{"role": "user", "content": "What is 2+2?"}],
            "completions": [
                {
                    "completion": [{"role": "assistant", "content": "4"}],
                    "reward": 1.0,
                    "reference_logprob": -1.0,
                },
                {
                    "completion": [{"role": "assistant", "content": "5"}],
                    "reward": 0.0,
                    "reference_logprob": -2.0,
                },
            ],
        },
        {
            "context": [{"role": "user", "content": "What is the capital of France?"}],
            "completions": [
                {
                    "completion": [
                        {"role": "assistant", "content": "The capital is Paris."}
                    ],
                    "reward": 1.0,
                    "reference_logprob": -3.0,
                },
                {
                    "completion": [
                        {"role": "assistant", "content": "The capital is London."}
                    ],
                    "reward": 0.0,
                    "reference_logprob": -4.0,
                },
                {
                    "completion": [
                        {"role": "assistant", "content": "The capital is Berlin."}
                    ],
                    "reward": 0.0,
                    "reference_logprob": -5.0,
                },
            ],
        },
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
        data_path = f.name

    return data_path


def test_oapl_dataset_flattening_and_log_z():
    data_path = create_sample_data()
    beta = 1.0
    data_config = {
        "dataset_name": "OAPLDataset",
        "data_path": data_path,
        "beta": beta,
    }
    dataset = load_oapl_dataset(data_config)

    # each group of 2 and 3 completions flattens to 5 total examples
    assert len(dataset.dataset) == 5

    first_group = dataset.dataset[0]
    assert first_group["context"] == [{"role": "user", "content": "What is 2+2?"}]
    assert first_group["completion"] == [{"role": "assistant", "content": "4"}]
    assert first_group["reward"] == 1.0
    assert first_group["reference_logprob"] == -1.0

    expected_log_z_group1 = math.log(
        (math.exp(1.0 / beta) + math.exp(0.0 / beta)) / 2
    )
    assert first_group["log_z"] == pytest.approx(expected_log_z_group1)
    # log_z is shared across every completion in the same group
    assert dataset.dataset[1]["log_z"] == pytest.approx(expected_log_z_group1)

    expected_log_z_group2 = math.log(
        (math.exp(1.0 / beta) + math.exp(0.0 / beta) + math.exp(0.0 / beta)) / 3
    )
    for idx in (2, 3, 4):
        assert dataset.dataset[idx]["log_z"] == pytest.approx(expected_log_z_group2)


def test_oapl_dataset_requires_at_least_two_completions():
    data = [
        {
            "context": [{"role": "user", "content": "What is 2+2?"}],
            "completions": [
                {
                    "completion": [{"role": "assistant", "content": "4"}],
                    "reward": 1.0,
                    "reference_logprob": -1.0,
                }
            ],
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
        data_path = f.name

    data_config = {
        "dataset_name": "OAPLDataset",
        "data_path": data_path,
        "beta": 1.0,
    }
    with pytest.raises(Exception):
        load_oapl_dataset(data_config)
