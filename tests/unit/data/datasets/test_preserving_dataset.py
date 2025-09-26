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
from datasets import Dataset

from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
    PreservingDataset,
)


class TestPreservingDataset:
    """Test suite for PreservingDataset class."""

    def test_no_none_filling(self):
        """Test that PreservingDataset doesn't add None values for missing keys."""
        # Create data with heterogeneous structure
        data = [
            {"role": "user", "content": "Hello", "extra_key": "value1"},
            {"role": "assistant", "content": "Hi"},  # Missing 'extra_key'
            {"role": "user", "content": "How are you?", "another_key": "value2"},
        ]

        dataset = PreservingDataset(data)

        # Check that missing keys are not filled with None
        assert "extra_key" not in dataset[1]
        assert "another_key" not in dataset[0]
        assert "another_key" not in dataset[1]

        # Verify original structure is preserved
        assert dataset[0]["extra_key"] == "value1"
        assert dataset[2]["another_key"] == "value2"

    def test_indexing_operations(self):
        """Test various indexing operations."""
        data = [{"id": i, "value": f"item_{i}"} for i in range(5)]
        dataset = PreservingDataset(data)

        # Test integer indexing
        assert dataset[0]["id"] == 0
        assert dataset[2]["value"] == "item_2"

        # Test negative indexing
        assert dataset[-1]["id"] == 4
        assert dataset[-2]["value"] == "item_3"

        # Test slicing
        sliced = dataset[1:3]
        assert len(sliced) == 2
        assert sliced[0]["id"] == 1
        assert sliced[1]["id"] == 2

        # Test list indexing
        selected = dataset[[0, 2, 4]]
        assert len(selected) == 3
        assert selected[0]["id"] == 0
        assert selected[1]["id"] == 2
        assert selected[2]["id"] == 4

        # Test out of range
        with pytest.raises(IndexError):
            _ = dataset[10]

    def test_map_function(self):
        """Test the map function preserves structure."""
        data = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20, "extra": "data"},
        ]
        dataset = PreservingDataset(data)

        # Map without indices
        def double_value(item):
            item = item.copy()
            item["value"] *= 2
            return item

        mapped = dataset.map(double_value)
        assert mapped[0]["value"] == 20
        assert mapped[1]["value"] == 40
        assert "extra" not in mapped[0]  # Still no extra key
        assert mapped[1]["extra"] == "data"  # Extra key preserved

        # Map with indices
        def add_index(item, idx):
            item = item.copy()
            item["index"] = idx
            return item

        indexed = dataset.map(add_index, with_indices=True)
        assert indexed[0]["index"] == 0
        assert indexed[1]["index"] == 1

    def test_iteration(self):
        """Test iteration over dataset."""
        data = [{"id": i} for i in range(3)]
        dataset = PreservingDataset(data)

        items = list(dataset)
        assert len(items) == 3
        for i, item in enumerate(dataset):
            assert item["id"] == i

    def test_length(self):
        """Test len() operation."""
        dataset = PreservingDataset([])
        assert len(dataset) == 0

        dataset = PreservingDataset([{"a": 1}, {"b": 2}])
        assert len(dataset) == 2


class TestOpenAIFormatDatasetWithHeterogeneousTools:
    """Test OpenAIFormatDataset with heterogeneous tool calls."""

    @pytest.fixture
    def heterogeneous_data(self):
        """Create test data with varying tool call structures."""
        train_data = [
            {
                "messages": [
                    {"role": "user", "content": "Check the workspace and write a file"},
                    {
                        "role": "assistant",
                        "content": "Let me look at the workspace first",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "view_file",
                                    "arguments": {
                                        "path": "/workspace",
                                        "line_start": 1,
                                    },
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "workspace contents...",
                        "tool_call_id": "call_1",
                    },
                    {
                        "role": "assistant",
                        "content": "Now writing the file",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    # Different argument structure - has 'content' and 'mode' that view_file doesn't
                                    "arguments": {
                                        "path": "test.py",
                                        "content": "print('hello')",
                                        "mode": "w",
                                    },
                                },
                            }
                        ],
                    },
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Search for something"},
                    {
                        "role": "assistant",
                        "content": "Searching",
                        "tool_calls": [
                            {
                                "id": "call_3",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    # Yet another different structure
                                    "arguments": {
                                        "query": "test",
                                        "max_results": 10,
                                        "filter": "*.py",
                                    },
                                },
                            }
                        ],
                    },
                ]
            },
        ]

        val_data = [
            {
                "messages": [
                    {"role": "user", "content": "Delete a file"},
                    {
                        "role": "assistant",
                        "content": "Deleting",
                        "tool_calls": [
                            {
                                "id": "call_4",
                                "type": "function",
                                "function": {
                                    "name": "delete_file",
                                    # Simple structure with just path
                                    "arguments": {"path": "old.txt"},
                                },
                            }
                        ],
                    },
                ]
            }
        ]

        # Write to temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            for item in train_data:
                json.dump(item, f)
                f.write("\n")
            train_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            for item in val_data:
                json.dump(item, f)
                f.write("\n")
            val_path = f.name

        return train_path, val_path, train_data, val_data

    def test_preserves_tool_structure_without_none(self, heterogeneous_data):
        """Test that heterogeneous tool calls are handled correctly.
        Note: This test verifies the PreservingDataset behavior when it's triggered.
        In this test case, the standard loading may succeed, so we test the class directly.
        """
        train_path, val_path, original_train, original_val = heterogeneous_data

        # Test PreservingDataset directly to verify its behavior
        from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
            PreservingDataset,
        )

        # Simulate what happens in the exception handler
        with open(train_path, "r") as f:
            train_data = [json.loads(line) for line in f]

        # Create PreservingDataset
        preserving_dataset = PreservingDataset(train_data)

        # Verify no None-filling occurs
        sample_0 = preserving_dataset[0]
        sample_1 = preserving_dataset[1]

        # First sample has two assistant messages with different tool structures
        # First assistant message - view_file
        assert "tool_calls" in sample_0["messages"][1]
        view_args = sample_0["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert "path" in view_args
        assert "line_start" in view_args
        # These keys should NOT exist (not filled with None)
        assert "content" not in view_args
        assert "mode" not in view_args
        assert "query" not in view_args

        # Second assistant message in same sample - write_file
        assert "tool_calls" in sample_0["messages"][3]
        write_args = sample_0["messages"][3]["tool_calls"][0]["function"]["arguments"]
        assert "path" in write_args
        assert "content" in write_args
        assert "mode" in write_args
        # These keys should NOT exist
        assert "line_start" not in write_args
        assert "query" not in write_args

        # Second sample - search with different structure
        assert "tool_calls" in sample_1["messages"][1]
        search_args = sample_1["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert "query" in search_args
        assert "max_results" in search_args
        assert "filter" in search_args
        # These keys should NOT exist
        assert "path" not in search_args
        assert "content" not in search_args

    def test_comparison_with_standard_dataset(self):
        """Compare behavior with standard HuggingFace Dataset to show the difference."""
        # Data with heterogeneous structure
        data = [
            {"role": "user", "content": "Hello", "tool_id": "123"},
            {"role": "assistant", "content": "Hi"},  # Missing tool_id
        ]

        # Standard HuggingFace Dataset adds None
        hf_dataset = Dataset.from_list(data)
        assert hf_dataset[0]["tool_id"] == "123"
        assert hf_dataset[1]["tool_id"] is None  # HF adds None

        # PreservingDataset doesn't add None
        preserving_dataset = PreservingDataset(data)
        assert preserving_dataset[0]["tool_id"] == "123"
        assert "tool_id" not in preserving_dataset[1]  # Key doesn't exist
