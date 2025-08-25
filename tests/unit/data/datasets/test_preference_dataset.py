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

from nemo_rl.data.datasets.preference_datasets import LocalPreferenceDataset


@pytest.fixture
def mock_dpo_data():
    """Create temporary DPO dataset files with sample data."""
    train_data = [
        {
            "prompt": "What is 2+2?",
            "chosen_response": "The answer is 4.",
            "rejected_response": "I don't know.",
        },
        {
            "prompt": "What is the capital of France?",
            "chosen_response": "The capital of France is Paris.",
            "rejected_response": "The capital of France is London.",
        },
    ]

    val_data = [
        {
            "prompt": "What is 3*3?",
            "chosen_response": "The answer is 9.",
            "rejected_response": "The answer is 6.",
        }
    ]

    train_ctx = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    val_ctx = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as train_file:
        json.dump(train_data, train_file)
        train_path = train_file.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as val_file:
        json.dump(val_data, val_file)
        val_path = val_file.name
    yield train_path, val_path
    # Cleanup
    os.unlink(train_path)
    os.unlink(val_path)


def test_dpo_dataset_initialization(mock_dpo_data):
    """Test that LocalPreferenceDataset initializes correctly with valid data files."""
    train_path, val_path = mock_dpo_data

    dataset = LocalPreferenceDataset(train_data_path=train_path, val_data_path=val_path)

    # Verify dataset initialization
    assert dataset.task_spec.task_name == "DPO"

    # Verify formatted_ds structure
    assert "train" in dataset.formatted_ds
    assert "validation" in dataset.formatted_ds

    assert len(dataset.formatted_ds["train"]) == 2
    assert len(dataset.formatted_ds["validation"]) == 1


def test_dpo_dataset_invalid_files():
    """Test that LocalPreferenceDataset raises appropriate errors with invalid files."""
    with pytest.raises(FileNotFoundError):
        LocalPreferenceDataset(
            train_data_path="nonexistent.json", val_data_path="nonexistent.json"
        )


def test_dpo_dataset_data_format(mock_dpo_data):
    """Test that LocalPreferenceDataset correctly formats the data."""
    train_path, val_path = mock_dpo_data
    dataset = LocalPreferenceDataset(train_data_path=train_path, val_data_path=val_path)

    # Verify data format
    train_sample = dataset.formatted_ds["train"][0]
    assert "prompt" in train_sample
    assert "chosen_response" in train_sample
    assert "rejected_response" in train_sample

    # Verify data content
    assert train_sample["prompt"] == "What is 2+2?"
    assert train_sample["chosen_response"] == "The answer is 4."
    assert train_sample["rejected_response"] == "I don't know."
