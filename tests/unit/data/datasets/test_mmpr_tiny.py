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
    It captures the exact text passed to __call__ for assertion.
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
            self.captured_call_text = None

        def apply_chat_template(self, messages, **kwargs):
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
            self.captured_call_text = text
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


# The raw question text as it appears in the MMPR dataset (with <image> token)
_RAW_QUESTION = "<image>\nWhat is the measure of angle BAC?"
# The clean question after <image> stripping by format_mmpr_tiny_dataset
_CLEAN_QUESTION = "What is the measure of angle BAC?"


def _run_processor(tiny_image_path):
    """Helper: run vlm_hf_data_processor on an MMPR sample and return
    (result DatumSpec, stub processor with captured_call_text)."""
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import vlm_hf_data_processor

    task_data_spec = TaskDataSpec(
        task_name="mmpr-tiny",
        prompt_file="examples/prompts/mmpr_tiny_cot_nemotron_omni.txt",
    )
    processor = _make_stub_nemotron_processor()
    sample = {
        "images": [tiny_image_path],
        "question": _RAW_QUESTION,
        "answer": "A",
        "task_name": "mmpr-tiny",
    }

    result = vlm_hf_data_processor(
        datum_dict=sample,
        task_data_spec=task_data_spec,
        processor=processor,
        max_seq_length=8192,
        idx=0,
    )
    return result, processor


class TestVLMProcessorMMPRTiny:
    """Smoke test: vlm_hf_data_processor with mmpr-tiny task through the
    NemotronNanoVLV2Processor placeholder-style code path."""

    def test_processor_produces_valid_datum_spec(self, tiny_image_path):
        result, _ = _run_processor(tiny_image_path)

        assert "message_log" in result
        assert "length" in result
        assert "extra_env_info" in result
        assert "ground_truth" in result["extra_env_info"]
        assert result["extra_env_info"]["ground_truth"] == "A"
        assert "vllm_content" in result
        assert "vllm_images" in result
        assert len(result["vllm_images"]) == 1
        assert result["task_name"] == "mmpr-tiny"

    def test_prompted_text_contains_boxed_literal_and_no_raw_dataset_string(
        self, tiny_image_path
    ):
        result, _ = _run_processor(tiny_image_path)
        vllm_content = result["vllm_content"]

        # Positive: literal \boxed{} must survive prompt formatting
        assert "\\boxed{}" in vllm_content

        # Negative: the raw dataset string (with <image> prefix) must NOT leak through
        assert _RAW_QUESTION not in vllm_content

    def test_placeholder_conversion_exact_string(self, tiny_image_path):
        """Verify the exact tokenizer input for the placeholder-style processor path.

        The processor stub captures the text argument passed to __call__.
        It must equal "<image>\\n<prompted_question>" where prompted_question
        is the prompt template formatted with the clean question text.
        """
        result, processor = _run_processor(tiny_image_path)

        # Build expected prompted question from the real prompt file
        with open("examples/prompts/mmpr_tiny_cot_nemotron_omni.txt") as f:
            prompt_template = f.read()
        prompted_question = prompt_template.format(_CLEAN_QUESTION)

        # The placeholder-style path should produce: "<image>\n<prompted_question>"
        expected_tokenizer_input = "<image>\n" + prompted_question

        # The stub's apply_chat_template joins message parts with spaces,
        # so the captured text passed to __call__ is the chat-templated string.
        # Verify the vllm_content (which is apply_chat_template output) matches.
        vllm_content = result["vllm_content"]
        assert vllm_content == expected_tokenizer_input

        # Verify exactly one <image> token in the final output
        assert vllm_content.count("<image>") == 1

        # Verify the question text is present
        assert _CLEAN_QUESTION in vllm_content

        # Verify the captured __call__ text also matches
        # (processor.__call__ receives the apply_chat_template output)
        assert processor.captured_call_text == expected_tokenizer_input
