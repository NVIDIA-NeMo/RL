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


import pytest

from nemo_rl.data.hf_datasets.helpsteer3 import (
    HelpSteer3Dataset,
    to_preference_data_format,
)


@pytest.fixture(scope="module")
def helpsteer3_dataset():
    try:
        dataset = HelpSteer3Dataset()
        yield dataset
    except Exception as e:
        print(f"Error during loading HelpSteer3Dataset: {e}")
        yield


def test_to_preference_data_format():
    """Test the `to_preference_data_format()` function with different preference values."""
    # Test case 1: response1 is preferred (overall_preference < 0)
    data1 = {
        "context": "What is 2+2?",
        "response1": "The answer is 4.",
        "response2": "I don't know.",
        "overall_preference": -1,
    }
    result1 = to_preference_data_format(data1)
    assert result1["context"] == [{"content": "What is 2+2?", "role": "user"}]
    assert result1["completions"] == [
        {"rank": 0, "completion": [{"role": "assistant", "content": "The answer is 4."}]},
        {"rank": 1, "completion": [{"role": "assistant", "content": "I don't know."}]
    }]

    # Test case 2: response2 is preferred (overall_preference > 0)
    data2 = {
        "context": "What is the capital of France?",
        "response1": "The capital of France is London.",
        "response2": "The capital of France is Paris.",
        "overall_preference": 1,
    }
    result2 = to_preference_data_format(data2)
    assert result2["context"] == [{"content": "What is the capital of France?", "role": "user"}]
    assert result2["completions"] == [
        {"rank": 0, "completion": [{"role": "assistant", "content": "The capital of France is Paris."}]},
        {"rank": 1, "completion": [{"role": "assistant", "content": "The capital of France is London."}]}
    ]

    # Test case 3: no preference (overall_preference = 0)
    data3 = {
        "context": "What is the weather like?",
        "response1": "It's sunny today.",
        "response2": "The weather is sunny.",
        "overall_preference": 0,
    }
    result3 = to_preference_data_format(data3)
    assert result3["context"] == [{"content": "What is the weather like?", "role": "user"}]
    # When preference is 0, neither response is preferred, so
    # response 1 is used for both chosen and rejected
    assert result3["completions"] == [
        {"rank": 0, "completion": [{"role": "assistant", "content": "It's sunny today."}]},
        {"rank": 1, "completion": [{"role": "assistant", "content": "It's sunny today."}]}
    ]

    # Test case 4: context is a list of dicts
    data1 = {
        "context": [
            {"role": "user", "content": "Can I ask you a question?"},
            {"role": "assistant", "content": "Sure, what do you want to know?"},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "response1": "4.",
        "response2": "I don't know.",
        "overall_preference": -1,
    }
    result1 = to_preference_data_format(data1)
    assert result1["context"] == [
        {"role": "user", "content": "Can I ask you a question?"},
        {"role": "assistant", "content": "Sure, what do you want to know?"},
        {"role": "user", "content": "What is 2+2?"}
    ]
    assert result1["completions"] == [
        {"rank": 0, "completion": [{"role": "assistant", "content": "4."}]},
        {"rank": 1, "completion": [{"role": "assistant", "content": "I don't know."}]
    }]

def test_helpsteer3_dataset_initialization(helpsteer3_dataset):
    """Test that HelpSteer3Dataset initializes correctly."""

    dataset = helpsteer3_dataset
    if dataset is None:
        pytest.skip("dataset download is flaky")

    # Verify dataset initialization
    assert dataset.task_spec.task_name == "HelpSteer3"


def test_helpsteer3_dataset_data_format(helpsteer3_dataset):
    """Test that HelpSteer3Dataset correctly formats the data."""

    dataset = helpsteer3_dataset
    if dataset is None:
        pytest.skip("dataset download is flaky")

    assert isinstance(dataset.formatted_ds, dict)
    assert "train" in dataset.formatted_ds
    assert "validation" in dataset.formatted_ds

    # Verify data format
    sample = dataset.formatted_ds["train"][0]
    assert "context" in sample
    assert "completions" in sample
