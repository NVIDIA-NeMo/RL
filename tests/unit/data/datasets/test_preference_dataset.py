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

from nemo_rl.data.datasets import load_preference_dataset


@pytest.fixture
def mock_preference_data():
    """Create temporary preference dataset files with sample data."""
    preference_data = [
        {
            "context": [{"role": "user", "content": "What is 2+2?"}],
            "completions": [
                {
                    "rank": 1,
                    "completion": [
                        {"role": "assistant", "content": "The answer is 4."}
                    ],
                },
                {
                    "rank": 2,
                    "completion": [{"role": "assistant", "content": "I don't know."}],
                },
            ],
        },
        {
            "context": [{"role": "user", "content": "What is the capital of France?"}],
            "completions": [
                {
                    "rank": 1,
                    "completion": [
                        {
                            "role": "assistant",
                            "content": "The capital of France is Paris.",
                        }
                    ],
                },
                {
                    "rank": 2,
                    "completion": [
                        {
                            "role": "assistant",
                            "content": "The capital of France is London.",
                        }
                    ],
                },
            ],
        },
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
    # Load the dataset
    data_config = {
        "dataset_name": "PreferenceDataset",
        "train_data_path": mock_preference_data,
    }
    dataset = load_preference_dataset(data_config)

    # Verify dataset initialization
    assert dataset.task_spec.task_name == "PreferenceDataset"

    # Verify formatted_ds structure
    assert "train" in dataset.formatted_ds
    assert len(dataset.formatted_ds["train"]) == 2


def test_preference_dataset_data_format(mock_preference_data):
    """Test that PreferenceDataset correctly loads and formats the data."""
    # Load the dataset
    data_config = {
        "dataset_name": "PreferenceDataset",
        "train_data_path": mock_preference_data,
    }
    dataset = load_preference_dataset(data_config)

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


@pytest.fixture
def mock_binary_preference_data():
    """Create temporary chosen_rejected dataset files with sample data."""
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

    try:
        yield train_path, val_path
    finally:
        # Cleanup
        os.unlink(train_path)
        os.unlink(val_path)


def test_binary_preference_dataset_initialization(mock_binary_preference_data):
    """Test that PreferenceDataset initializes correctly with valid data files."""
    # Load the dataset
    train_path, val_path = mock_binary_preference_data
    data_config = {
        "dataset_name": "BinaryPreferenceDataset",
        "train_data_path": train_path,
        "val_data_path": val_path,
        "prompt_key": "prompt",
        "chosen_key": "chosen_response",
        "rejected_key": "rejected_response",
    }
    dataset = load_preference_dataset(data_config)

    # Verify dataset initialization
    assert dataset.task_spec.task_name == "BinaryPreferenceDataset"

    # Verify formatted_ds structure
    assert "train" in dataset.formatted_ds
    assert "validation" in dataset.formatted_ds

    assert len(dataset.formatted_ds["train"]) == 2
    assert len(dataset.formatted_ds["validation"]) == 1


def test_binary_preference_dataset_invalid_files():
    """Test that PreferenceDataset raises appropriate errors with invalid files."""
    with pytest.raises(FileNotFoundError):
        data_config = {
            "dataset_name": "BinaryPreferenceDataset",
            "train_data_path": "nonexistent.json",
            "val_data_path": "nonexistent.json",
            "prompt_key": "prompt",
            "chosen_key": "chosen_response",
            "rejected_key": "rejected_response",
        }
        load_preference_dataset(data_config)


def test_binary_preference_dataset_data_format(mock_binary_preference_data):
    """Test that PreferenceDataset correctly formats the data."""
    # Load the dataset
    train_path, val_path = mock_binary_preference_data
    data_config = {
        "dataset_name": "BinaryPreferenceDataset",
        "train_data_path": train_path,
        "val_data_path": val_path,
        "prompt_key": "prompt",
        "chosen_key": "chosen_response",
        "rejected_key": "rejected_response",
    }
    dataset = load_preference_dataset(data_config)

    # Verify data format
    train_sample = dataset.formatted_ds["train"][0]
    assert "context" in train_sample
    assert "completions" in train_sample

    # Verify data content
    print(train_sample["completions"])
    assert train_sample["context"] == [{"content": "What is 2+2?", "role": "user"}]
    assert train_sample["completions"] == [
        {
            "completion": [{"content": "The answer is 4.", "role": "assistant"}],
            "rank": 0,
        },
        {"completion": [{"content": "I don't know.", "role": "assistant"}], "rank": 1},
    ]
