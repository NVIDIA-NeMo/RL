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
from nemo_rl.data.hf_datasets.prompt_response_dataset import (
    PromptResponseDataset,
)


@pytest.fixture
def sample_data(request):
    input_key = request.param[0]
    output_key = request.param[1]

    train_data = [
        {input_key: "Hello", output_key: "Hi there!"},
        {input_key: "How are you?", output_key: "I'm good, thanks!"},
    ]
    val_data = [
        {input_key: "What's up?", output_key: "Not much!"},
        {input_key: "Bye", output_key: "Goodbye!"},
    ]

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


@pytest.mark.parametrize("sample_data", [("input", "output")], indirect=True)
def test_dataset_initialization(sample_data):
    train_path, val_path = sample_data
    dataset = PromptResponseDataset(train_path, val_path)

    assert dataset.input_key == "input"
    assert dataset.output_key == "output"
    assert "train" in dataset.formatted_ds
    assert "validation" in dataset.formatted_ds


@pytest.mark.parametrize("sample_data", [("question", "answer")], indirect=True)
def test_custom_keys(sample_data):
    train_path, val_path = sample_data
    dataset = PromptResponseDataset(
        train_path, val_path, input_key="question", output_key="answer"
    )

    assert dataset.input_key == "question"
    assert dataset.output_key == "answer"


@pytest.mark.hf_gated
@pytest.mark.parametrize("sample_data", [("question", "answer")], indirect=True)
def test_message_formatting(sample_data):
    train_path, val_path = sample_data
    dataset = PromptResponseDataset(
        train_path, val_path, input_key="question", output_key="answer"
    )

    first_example = dataset.formatted_ds["train"][0]

    assert first_example["messages"][0]["role"] == "user"
    assert first_example["messages"][0]["content"] == "Hello"
    assert first_example["messages"][1]["role"] == "assistant"
    assert first_example["messages"][1]["content"] == "Hi there!"

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
