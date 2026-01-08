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
import tempfile

import pytest

from nemo_rl.data.datasets import load_preference_dataset


def create_sample_data(
    is_binary, prompt_key="prompt", chosen_key="chosen", rejected_key="rejected"
):
    data = [
        {
            prompt_key: "What is 2+2?",
            chosen_key: "The answer is 4.",
            rejected_key: "I don't know.",
        },
        {
            prompt_key: "What is the capital of France?",
            chosen_key: "The capital of France is Paris.",
            rejected_key: "The capital of France is London.",
        },
    ]

    if not is_binary:
        data = [
            {
                "context": [{"role": "user", "content": item[prompt_key]}],
                "completions": [
                    {
                        "rank": 0,
                        "completion": [
                            {"role": "assistant", "content": item[chosen_key]}
                        ],
                    },
                    {
                        "rank": 1,
                        "completion": [
                            {"role": "assistant", "content": item[rejected_key]}
                        ],
                    },
                ],
            }
            for item in data
        ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        data_path = f.name

    return data_path


@pytest.mark.parametrize(
    "prompt_key,chosen_key,rejected_key",
    [("prompt", "chosen", "rejected"), ("question", "answer1", "answer2")],
)
def test_binary_preference_dataset(prompt_key, chosen_key, rejected_key):
    # load the dataset
    data_path = create_sample_data(
        is_binary=True,
        prompt_key=prompt_key,
        chosen_key=chosen_key,
        rejected_key=rejected_key,
    )
    data_config = {
        "dataset_name": "BinaryPreferenceDataset",
        "data_path": data_path,
        "prompt_key": prompt_key,
        "chosen_key": chosen_key,
        "rejected_key": rejected_key,
    }
    dataset = load_preference_dataset(data_config)

    # check prompt, chosen and rejected keys
    assert dataset.prompt_key == prompt_key
    assert dataset.chosen_key == chosen_key
    assert dataset.rejected_key == rejected_key

    # check the first example
    first_example = dataset.dataset[0]

    # only contains context, completions and task_name
    assert len(first_example.keys()) == 3
    assert "context" in first_example
    assert "completions" in first_example
    assert "task_name" in first_example

    # check the content
    assert first_example["context"] == [{"role": "user", "content": "What is 2+2?"}]
    assert first_example["completions"] == [
        {
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
            "rank": 0,
        },
        {"completion": [{"role": "assistant", "content": "I don't know."}], "rank": 1},
    ]


def test_preference_dataset():
    # load the dataset
    data_path = create_sample_data(is_binary=False)
    data_config = {
        "dataset_name": "PreferenceDataset",
        "data_path": data_path,
    }
    dataset = load_preference_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]

    # only contains context, completions and task_name
    assert len(first_example.keys()) == 3
    assert "context" in first_example
    assert "completions" in first_example
    assert "task_name" in first_example

    # check the content
    assert first_example["context"] == [{"role": "user", "content": "What is 2+2?"}]
    assert first_example["completions"] == [
        {
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
            "rank": 0,
        },
        {"completion": [{"role": "assistant", "content": "I don't know."}], "rank": 1},
    ]
