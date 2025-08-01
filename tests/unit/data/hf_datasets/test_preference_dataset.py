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
import os
import tempfile

import pytest

from nemo_rl.data.hf_datasets.preference_dataset import PreferenceDataset


@pytest.fixture
def mock_preference_data():
    """Create temporary preference dataset files with sample data."""
    preference_data = [
        {
            "context": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "completions": [
                {
                    "rank": 1,
                    "completion": [
                        {"role": "assistant", "content": "The answer is 4."}
                    ]
                },
                {
                    "rank": 2,
                    "completion": [
                        {"role": "assistant", "content": "I don't know."}
                    ]
                }
            ]
        },
        {
            "context": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "completions": [
                {
                    "rank": 1,
                    "completion": [
                        {"role": "assistant", "content": "The capital of France is Paris."}
                    ]
                },
                {
                    "rank": 2,
                    "completion": [
                        {"role": "assistant", "content": "The capital of France is London."}
                    ]
                },
            ]
        }
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as preference_file:
        json.dump(preference_data, preference_file)
        preference_path = preference_file.name
    
    try:
        yield preference_path
    finally:
        # Cleanup
        os.unlink(preference_path)


def test_preference_dataset_initialization(mock_preference_data):
    """Test that PreferenceDataset initializes correctly with valid data files."""
    preference_path = mock_preference_data

    dataset = PreferenceDataset(dataset_path=preference_path, split="train")

    # Verify dataset initialization
    assert dataset.task_spec.task_name == "PreferenceData"

    # Verify formatted_ds structure
    assert "train" in dataset.formatted_ds
    assert len(dataset.formatted_ds["train"]) == 2


def test_preference_dataset_data_format(mock_preference_data):
    """Test that PreferenceDataset correctly loads and formats the data."""
    preference_path = mock_preference_data
    dataset = PreferenceDataset(dataset_path=preference_path, split="train")

    # Verify data format
    sample = dataset.formatted_ds["train"][0]
    assert "context" in sample
    assert "completions" in sample

    # Verify context structure
    assert isinstance(sample["context"], list)
    assert len(sample["context"]) == 1
    assert "role" in sample["context"][0]
    assert "content" in sample["context"][0]

    # Verify completions structure
    assert isinstance(sample["completions"], list)
    assert len(sample["completions"]) == 2

    for completion in sample["completions"]:
        assert "rank" in completion
        assert "completion" in completion
        assert isinstance(completion["rank"], int)
        assert isinstance(completion["completion"], list)
