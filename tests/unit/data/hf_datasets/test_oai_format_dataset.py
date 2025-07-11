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
from transformers import AutoTokenizer

from nemo_rl.data.hf_datasets.chat_templates import COMMON_CHAT_TEMPLATES
from nemo_rl.data.hf_datasets.oai_format_dataset import (
    OpenAIFormatDataset,
)


@pytest.fixture
def sample_data(request):
    chat_key = request.param[0]
    system_key = request.param[1]

    train_data = {
        chat_key: [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    }
    val_data = {
        chat_key: [
            {"role": "user", "content": "What is the capital of Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
        ],
    }

    if system_key is not None:
        train_data[system_key] = "You are a helpful assistant."
    if system_key is not None:
        val_data[system_key] = "You are a helpful assistant."

    # Create temporary files for train and validation data
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

    return train_path, val_path


@pytest.mark.parametrize("sample_data", [("messages", None)], indirect=True)
def test_dataset_initialization(sample_data):
    train_path, val_path = sample_data
    dataset = OpenAIFormatDataset(train_path, val_path)

    assert dataset.chat_key == "messages"
    assert "train" in dataset.formatted_ds
    assert "validation" in dataset.formatted_ds


@pytest.mark.parametrize("sample_data", [("conversations", None)], indirect=True)
def test_custom_keys(sample_data):
    train_path, val_path = sample_data
    dataset = OpenAIFormatDataset(
        train_path,
        val_path,
        chat_key="conversations",
        system_prompt="You are a helpful assistant.",
    )

    assert dataset.chat_key == "conversations"
    assert dataset.system_prompt == "You are a helpful assistant."


@pytest.mark.parametrize("sample_data", [("messages", "system_key")], indirect=True)
def test_message_formatting(sample_data):
    train_path, val_path = sample_data
    dataset = OpenAIFormatDataset(
        train_path, val_path, chat_key="messages", system_key="system_key"
    )

    first_example = dataset.formatted_ds["train"][0]

    assert first_example["messages"][0]["role"] == "system"
    assert first_example["messages"][0]["content"] == "You are a helpful assistant."
    assert first_example["messages"][1]["role"] == "user"
    assert first_example["messages"][1]["content"] == "What is the capital of France?"
    assert first_example["messages"][2]["role"] == "assistant"
    assert first_example["messages"][2]["content"] == "The capital of France is Paris."

    chat_template = COMMON_CHAT_TEMPLATES.passthrough_prompt_response
    tokenizer = AutoTokenizer.from_pretrained("Meta-Llama/Meta-Llama-3-8B-Instruct")

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
