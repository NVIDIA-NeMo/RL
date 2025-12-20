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

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.chat_templates import COMMON_CHAT_TEMPLATES
from nemo_rl.data.datasets import load_response_dataset
from nemo_rl.data.datasets.response_datasets import OpenAIFormatDataset


@pytest.fixture
def sample_data(request):
    chat_key = request.param[0]
    system_key = request.param[1]

    data = {
        chat_key: [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    }

    if system_key is not None:
        data[system_key] = "You are a helpful assistant."

    # Create temporary files for train and validation data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        data_path = f.name

    return data_path


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer = get_tokenizer({"name": "Qwen/Qwen3-0.6B"})
    return tokenizer


@pytest.mark.parametrize("sample_data", [("messages", None)], indirect=True)
def test_dataset_initialization(sample_data):
    data_path = sample_data
    data_config = {
        "dataset_name": "openai_format",
        "data_path": data_path,
    }
    dataset = load_response_dataset(data_config)

    assert dataset.chat_key == "messages"
    assert len(dataset.dataset) == 1


@pytest.mark.parametrize("sample_data", [("conversations", None)], indirect=True)
def test_custom_keys(sample_data):
    data_path = sample_data
    data_config = {
        "dataset_name": "openai_format",
        "data_path": data_path,
        "chat_key": "conversations",
        "system_prompt": "You are a helpful assistant.",
    }
    dataset = load_response_dataset(data_config)

    assert dataset.chat_key == "conversations"
    assert dataset.system_prompt == "You are a helpful assistant."


@pytest.mark.parametrize("sample_data", [("messages", "system_key")], indirect=True)
def test_message_formatting(sample_data, tokenizer):
    # load the dataset
    data_path = sample_data
    dataset = OpenAIFormatDataset(
        data_path,
        chat_key="messages",
        system_key="system_key",
    )

    # check the first example
    first_example = dataset.dataset[0]

    assert "task_name" in first_example
    assert first_example["messages"][0]["role"] == "system"
    assert first_example["messages"][0]["content"] == "You are a helpful assistant."
    assert first_example["messages"][1]["role"] == "user"
    assert first_example["messages"][1]["content"] == "What is the capital of France?"
    assert first_example["messages"][2]["role"] == "assistant"
    assert first_example["messages"][2]["content"] == "The capital of France is Paris."

    # check the combined message
    chat_template = COMMON_CHAT_TEMPLATES.passthrough_prompt_response
    combined_message = tokenizer.apply_chat_template(
        first_example["messages"],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    assert combined_message == "".join(
        message["content"] for message in first_example["messages"]
    )
