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

import os
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

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


def _make_stub_nemotron_processor():
    """Build a minimal stub whose class name is NemotronNanoVLV2Processor.

    The stub implements just enough of the AutoProcessor interface for
    vlm_hf_data_processor to exercise the placeholder-style code path.
    """
    fake_input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    class _ImageProcessor:
        model_input_names = ["pixel_values"]

    class NemotronNanoVLV2Processor:
        image_token = "<image>"

        def __init__(self):
            self.image_processor = _ImageProcessor()
            self.tokenizer = MagicMock()
            self.tokenizer.model_input_names = ["input_ids"]

        def apply_chat_template(self, messages, **kwargs):
            # Flatten message content into a single string for the stub
            parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
            return " ".join(parts)

        def __call__(self, text=None, images=None, **kwargs):
            return {
                "input_ids": fake_input_ids,
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

    return NemotronNanoVLV2Processor()


@pytest.fixture()
def tiny_image_path(tmp_path):
    """Create a minimal 1x1 PNG for processor smoke tests."""
    img_path = tmp_path / "test_image.png"
    Image.new("RGB", (1, 1), color="red").save(img_path)
    return str(img_path)


class TestVLMProcessorMMPRTiny:
    """Smoke test: vlm_hf_data_processor with mmpr-tiny task through the
    NemotronNanoVLV2Processor placeholder-style code path."""

    def _make_sample(self, image_path):
        return {
            "images": [image_path],
            "question": "<image>\nWhat is the measure of angle BAC?",
            "answer": "A",
            "task_name": "mmpr-tiny",
        }

    def test_processor_produces_valid_datum_spec(self, tiny_image_path):
        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import vlm_hf_data_processor

        task_data_spec = TaskDataSpec(
            task_name="mmpr-tiny",
            prompt_file="examples/prompts/mmpr_tiny_cot_nemotron_omni.txt",
        )
        processor = _make_stub_nemotron_processor()
        sample = self._make_sample(tiny_image_path)

        result = vlm_hf_data_processor(
            datum_dict=sample,
            task_data_spec=task_data_spec,
            processor=processor,
            max_seq_length=8192,
            idx=0,
        )

        assert "message_log" in result
        assert "length" in result
        assert "extra_env_info" in result
        assert "ground_truth" in result["extra_env_info"]
        assert result["extra_env_info"]["ground_truth"] == "A"
        assert "vllm_content" in result
        assert "vllm_images" in result
        assert len(result["vllm_images"]) == 1
        assert result["task_name"] == "mmpr-tiny"

    def test_prompted_text_contains_boxed_literal(self, tiny_image_path):
        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import vlm_hf_data_processor

        task_data_spec = TaskDataSpec(
            task_name="mmpr-tiny",
            prompt_file="examples/prompts/mmpr_tiny_cot_nemotron_omni.txt",
        )
        processor = _make_stub_nemotron_processor()
        sample = self._make_sample(tiny_image_path)

        result = vlm_hf_data_processor(
            datum_dict=sample,
            task_data_spec=task_data_spec,
            processor=processor,
            max_seq_length=8192,
            idx=0,
        )

        vllm_content = result["vllm_content"]
        assert "\\boxed{}" in vllm_content

    def test_placeholder_conversion_for_nemotron_processor(self, tiny_image_path):
        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import vlm_hf_data_processor

        task_data_spec = TaskDataSpec(
            task_name="mmpr-tiny",
            prompt_file="examples/prompts/mmpr_tiny_cot_nemotron_omni.txt",
        )
        processor = _make_stub_nemotron_processor()
        sample = self._make_sample(tiny_image_path)

        result = vlm_hf_data_processor(
            datum_dict=sample,
            task_data_spec=task_data_spec,
            processor=processor,
            max_seq_length=8192,
            idx=0,
        )

        # The vllm_content should contain the <image> placeholder (added by
        # the processor path for NemotronNanoVLV2Processor) followed by the
        # prompted question text
        vllm_content = result["vllm_content"]
        assert "<image>" in vllm_content
        assert "What is the measure of angle BAC?" in vllm_content
