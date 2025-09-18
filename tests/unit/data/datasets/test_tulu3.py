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

from nemo_rl.data.datasets.preference_datasets import Tulu3PreferenceDataset
from nemo_rl.data.datasets.preference_datasets.tulu3 import to_preference_data_format


@pytest.fixture(scope="module")
def tulu3_dataset():
    try:
        dataset = Tulu3PreferenceDataset()
        yield dataset
    except Exception as e:
        print(f"Error during loading Tulu3PreferenceDataset: {e}")
        yield


def test_to_preference_data_format():
    """Test the `to_preference_data_format()` function with different preference values."""
    data = {
        "prompt": "What is 2+2?",
        "chosen": [
            {"content": "What is 2+2?", "role": "user"},
            {"role": "assistant", "content": "The answer is 4."},
        ],
        "rejected": [
            {"content": "What is 2+2?", "role": "user"},
            {"role": "assistant", "content": "I don't know."},
        ],
    }
    result = to_preference_data_format(data)
    assert result["context"] == [{"content": "What is 2+2?", "role": "user"}]
    assert result["completions"] == [
        {
            "rank": 0,
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
        },
        {"rank": 1, "completion": [{"role": "assistant", "content": "I don't know."}]},
    ]


def test_tulu3_dataset_initialization(tulu3_dataset):
    """Test that Tulu3PreferenceDataset initializes correctly."""

    dataset = tulu3_dataset
    if dataset is None:
        pytest.skip("dataset download is flaky")

    # Verify dataset initialization
    assert dataset.task_spec.task_name == "Tulu3Preference"


def test_tulu3_dataset_data_format(tulu3_dataset):
    """Test that Tulu3PreferenceDataset correctly formats the data."""

    dataset = tulu3_dataset
    if dataset is None:
        pytest.skip("dataset download is flaky")

    assert isinstance(dataset.formatted_ds, dict)
    assert "train" in dataset.formatted_ds

    # Verify data format
    sample = dataset.formatted_ds["train"][0]
    assert "prompt" in sample
    assert "chosen" in sample
    assert "rejected" in sample
