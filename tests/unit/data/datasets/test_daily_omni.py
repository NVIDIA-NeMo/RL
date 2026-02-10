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

from nemo_rl.data.datasets import load_response_dataset


@pytest.fixture(scope="module")
def daily_omni_dataset():
    """Load Daily-Omni dataset from HuggingFace (downloads and caches locally)."""
    try:
        data_config = {"dataset_name": "daily-omni"}
        dataset = load_response_dataset(data_config)
        yield dataset
    except Exception as e:
        print(f"Error during loading DailyOmniDataset: {e}")
        yield None


def test_daily_omni_download_and_cache(daily_omni_dataset):
    """Test that the dataset can download the raw data from HuggingFace and cache it locally."""
    dataset = daily_omni_dataset
    if dataset is None:
        pytest.skip("dataset download is flaky or network unavailable")

    # Verify dataset was downloaded and cached locally
    assert dataset.hf_cache_dir is not None
    assert len(dataset.dataset) > 0
    assert dataset.task_name == "daily-omni"


def test_daily_omni_format_to_openai(daily_omni_dataset):
    """Test that once raw data is downloaded, the dataset formats it to OpenAI-compatible structure."""
    dataset = daily_omni_dataset
    if dataset is None:
        pytest.skip("dataset download is flaky or network unavailable")

    # Get first raw example and run preprocessor (format_data)
    first_raw = dataset.dataset[0]
    formatted = dataset.preprocessor(first_raw)

    # OpenAI-compatible structure: messages + task_name
    assert "messages" in formatted
    assert "task_name" in formatted
    assert formatted["task_name"] == "daily-omni"

    assert len(formatted["messages"]) == 2
    user_msg = formatted["messages"][0]
    assistant_msg = formatted["messages"][1]

    assert user_msg["role"] == "user"
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == first_raw["Answer"]

    # User message content: list of video + text (multimodal)
    user_content = user_msg["content"]
    assert isinstance(user_content, list)
    assert len(user_content) >= 1
    types = [c["type"] for c in user_content]
    assert "video" in types
    assert "text" in types
    assert user_content[0]["type"] == "video"
    assert "video" in user_content[0]
    # Text part contains question and choices
    text_parts = [c for c in user_content if c["type"] == "text"]
    assert len(text_parts) == 1
    assert first_raw["Question"] in text_parts[0]["text"]
