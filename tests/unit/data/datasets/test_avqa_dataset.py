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

from nemo_rl.data.datasets.eval_datasets import (
    MULTIMODAL_DATASETS,
    _is_multimodal_dataset,
)


class TestIsMultimodalDataset:
    """Tests for _is_multimodal_dataset and MULTIMODAL_DATASETS."""

    def test_mmau_is_multimodal(self):
        assert _is_multimodal_dataset("mmau") is True

    def test_twinkstart_mmau_is_multimodal(self):
        assert _is_multimodal_dataset("TwinkStart/MMAU") is True

    def test_math_is_not_multimodal(self):
        assert _is_multimodal_dataset("math") is False

    def test_gpqa_is_not_multimodal(self):
        assert _is_multimodal_dataset("gpqa") is False

    def test_empty_string_is_not_multimodal(self):
        assert _is_multimodal_dataset("") is False

    def test_multimodal_datasets_is_a_set(self):
        assert isinstance(MULTIMODAL_DATASETS, set)

    def test_multimodal_datasets_contains_expected(self):
        assert "mmau" in MULTIMODAL_DATASETS
        assert "TwinkStart/MMAU" in MULTIMODAL_DATASETS


class TestAVQADataset:
    """Tests for AVQADataset loading and format_data."""

    def test_avqa_dataset_loads(self):
        from nemo_rl.data.datasets.response_datasets.avqa import AVQADataset

        dataset = AVQADataset(split="train", max_samples=2)

        assert dataset.task_name == "avqa"
        assert len(dataset.dataset) > 0
        assert dataset.preprocessor is not None
        assert dataset.val_dataset is None

    def test_avqa_dataset_with_split_validation(self):
        from nemo_rl.data.datasets.response_datasets.avqa import AVQADataset

        dataset = AVQADataset(
            split="train", split_validation_size=0.5, seed=42, max_samples=4
        )

        assert dataset.task_name == "avqa"
        assert len(dataset.dataset) > 0
        assert dataset.val_dataset is not None
        assert len(dataset.val_dataset) > 0

    def test_avqa_dataset_format_data(self):
        from nemo_rl.data.datasets.response_datasets.avqa import AVQADataset

        dataset = AVQADataset(split="train", max_samples=2)

        # Get a raw example and format it
        raw_example = dataset.dataset[0]
        formatted = dataset.preprocessor(raw_example)

        assert "messages" in formatted
        assert "task_name" in formatted
        assert formatted["task_name"] == "avqa"

        # Check message structure
        messages = formatted["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # Check user content is multimodal (has audio + text)
        user_content = messages[0]["content"]
        assert isinstance(user_content, list)
        content_types = [c["type"] for c in user_content]
        assert "audio" in content_types
        assert "text" in content_types

    def test_avqa_dataset_invalid_split(self):
        from nemo_rl.data.datasets.response_datasets.avqa import AVQADataset

        with pytest.raises(ValueError, match="Invalid split"):
            AVQADataset(split="test")

    def test_avqa_dataset_has_task_name_column(self):
        from nemo_rl.data.datasets.response_datasets.avqa import AVQADataset

        dataset = AVQADataset(split="train", max_samples=2)

        # Verify the task_name column was added
        raw_example = dataset.dataset[0]
        assert "task_name" in raw_example
        assert raw_example["task_name"] == "avqa"

    def test_avqa_load_via_registry(self):
        from nemo_rl.data.datasets import load_response_dataset

        data_config = {"dataset_name": "avqa", "split": "train", "max_samples": 2}
        dataset = load_response_dataset(data_config)

        assert dataset.task_name == "avqa"
        assert len(dataset.dataset) > 0
