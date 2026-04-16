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

from nemo_rl.data.datasets.response_datasets.mmpr_tiny import (
    MMPRTinyDataset,
    format_mmpr_tiny_dataset,
)


class TestFormatMMPRTinyDataset:
    """Tests for the MMPR-Tiny data formatting function."""

    def test_format_produces_correct_message_structure(self):
        sample = {
            "images": ["/path/to/image.png"],
            "question": "What is the angle?",
            "answer": "A",
            "task_name": "mmpr-tiny",
        }
        result = format_mmpr_tiny_dataset(sample)

        assert "messages" in result
        assert "task_name" in result
        assert result["task_name"] == "mmpr-tiny"

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        user_content = messages[0]["content"]
        assert len(user_content) == 2
        assert user_content[0]["type"] == "image"
        assert user_content[0]["image"] == "/path/to/image.png"
        assert user_content[1]["type"] == "text"
        assert user_content[1]["text"] == "What is the angle?"

        assert messages[1]["content"] == "A"

    def test_format_strips_image_tokens_from_question(self):
        sample = {
            "images": ["/path/to/image.png"],
            "question": "<image>\nWhat is the angle?",
            "answer": "32",
            "task_name": "mmpr-tiny",
        }
        result = format_mmpr_tiny_dataset(sample)
        text = result["messages"][0]["content"][1]["text"]
        assert "<image>" not in text
        assert "What is the angle?" in text

    def test_format_handles_numeric_answer(self):
        sample = {
            "images": ["/path/to/image.png"],
            "question": "How many circles?",
            "answer": "4",
            "task_name": "mmpr-tiny",
        }
        result = format_mmpr_tiny_dataset(sample)
        assert result["messages"][1]["content"] == "4"


class TestMMPRTinyDataset:
    """Tests for the MMPRTinyDataset class."""

    def test_invalid_split_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid split: valB"):
            MMPRTinyDataset(split="valB", download_dir="/tmp/fake")

    def test_invalid_split_test_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid split"):
            MMPRTinyDataset(split="test", download_dir="/tmp/fake")

    def test_missing_download_dir_raises_value_error(self):
        with pytest.raises(ValueError, match="download_dir is required"):
            MMPRTinyDataset(download_dir="")


class TestPromptFileFormatCompatibility:
    """Tests that the prompt file is compatible with str.format()."""

    def test_prompt_file_has_single_format_placeholder(self):
        with open("examples/prompts/mmpr_tiny_cot_nemotron_omni.txt") as f:
            prompt = f.read()

        result = prompt.format("What is the measure of angle BAC?")
        assert "\\boxed{}" in result
        assert "What is the measure of angle BAC?" in result

    def test_prompt_file_does_not_crash_with_format(self):
        with open("examples/prompts/mmpr_tiny_cot_nemotron_omni.txt") as f:
            prompt = f.read()

        # Should not raise IndexError or KeyError
        prompt.format("test question")
