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
    format_helpsteer3,
)


@pytest.fixture(scope="module")
def helpsteer3_dataset():
    try:
        dataset = HelpSteer3Dataset()
        yield dataset
    except Exception as e:
        print(f"Error during loading HelpSteer3Dataset: {e}")
        yield


def test_format_helpsteer3():
    """Test the format_helpsteer3 function with different preference values."""
    # Test case 1: response1 is preferred (overall_preference < 0)
    data1 = {
        "context": "What is 2+2?",
        "response1": "The answer is 4.",
        "response2": "I don't know.",
        "overall_preference": -1,
    }
    result1 = format_helpsteer3(data1)
    assert result1["prompt"] == "What is 2+2?"
    assert result1["chosen_response"] == "The answer is 4."
    assert result1["rejected_response"] == "I don't know."

    # Test case 2: response2 is preferred (overall_preference > 0)
    data2 = {
        "context": "What is the capital of France?",
        "response1": "The capital of France is London.",
        "response2": "The capital of France is Paris.",
        "overall_preference": 1,
    }
    result2 = format_helpsteer3(data2)
    assert result2["prompt"] == "What is the capital of France?"
    assert result2["chosen_response"] == "The capital of France is Paris."
    assert result2["rejected_response"] == "The capital of France is London."

    # Test case 3: no preference (overall_preference = 0)
    data3 = {
        "context": "What is the weather like?",
        "response1": "It's sunny today.",
        "response2": "The weather is sunny.",
        "overall_preference": 0,
    }
    result3 = format_helpsteer3(data3)
    assert result3["prompt"] == "What is the weather like?"
    # When preference is 0, neither response is preferred, so
    # response 1 is used for both chosen and rejected
    assert result3["chosen_response"] == "It's sunny today."
    assert result3["rejected_response"] == "It's sunny today."


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
    assert "prompt" in sample
    assert "chosen_response" in sample
    assert "rejected_response" in sample
