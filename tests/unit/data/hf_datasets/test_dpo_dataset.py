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

from nemo_rl.data.hf_datasets.dpo import DPODataset


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
    """Test that DPODataset initializes correctly with valid data files."""
    train_path, val_path = mock_dpo_data

    dataset = DPODataset(train_data_path=train_path, val_data_path=val_path)

    # Verify dataset initialization
    assert dataset.task_spec.task_name == "DPO"

    # Verify formatted_ds structure
    assert "train" in dataset.formatted_ds
    assert "validation" in dataset.formatted_ds

    assert len(dataset.formatted_ds["train"]) == 2
    assert len(dataset.formatted_ds["validation"]) == 1


def test_dpo_dataset_invalid_files():
    """Test that DPODataset raises appropriate errors with invalid files."""
    with pytest.raises(FileNotFoundError):
        DPODataset(train_data_path="nonexistent.json", val_data_path="nonexistent.json")


def test_dpo_dataset_data_format(mock_dpo_data):
    """Test that DPODataset correctly formats the data."""
    train_path, val_path = mock_dpo_data
    dataset = DPODataset(train_data_path=train_path, val_data_path=val_path)

    # Verify data format
    train_sample = dataset.formatted_ds["train"][0]
    assert "context" in train_sample
    assert "completions" in train_sample

    # Verify data content
    print(train_sample["completions"])
    assert train_sample["context"] == [{'content': 'What is 2+2?', 'role': 'user'}]
    assert train_sample["completions"] == [
        {'completion': [{'content': 'The answer is 4.', 'role': 'assistant'}], 'rank': 0},
        {'completion': [{'content': "I don't know.", 'role': 'assistant'}], 'rank': 1}
    ]