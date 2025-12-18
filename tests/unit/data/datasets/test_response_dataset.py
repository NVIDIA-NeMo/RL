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

from nemo_rl.data.chat_templates import COMMON_CHAT_TEMPLATES
from nemo_rl.data.datasets import load_response_dataset


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
    # load the dataset
    train_path, val_path = sample_data
    data_config = {
        "dataset_name": "ResponseDataset",
        "train_data_path": train_path,
        "val_data_path": val_path,
    }
    dataset = load_response_dataset(data_config)

    assert dataset.input_key == "input"
    assert dataset.output_key == "output"
    assert "train" in dataset.formatted_ds
    assert "validation" in dataset.formatted_ds


@pytest.mark.parametrize("sample_data", [("question", "answer")], indirect=True)
def test_custom_keys(sample_data):
    # load the dataset
    train_path, val_path = sample_data
    data_config = {
        "dataset_name": "ResponseDataset",
        "train_data_path": train_path,
        "val_data_path": val_path,
        "input_key": "question",
        "output_key": "answer",
    }
    dataset = load_response_dataset(data_config)

    assert dataset.input_key == "question"
    assert dataset.output_key == "answer"


@pytest.mark.hf_gated
@pytest.mark.parametrize("sample_data", [("question", "answer")], indirect=True)
def test_message_formatting(sample_data):
    # load the dataset
    train_path, val_path = sample_data
    data_config = {
        "dataset_name": "ResponseDataset",
        "train_data_path": train_path,
        "val_data_path": val_path,
        "input_key": "question",
        "output_key": "answer",
    }
    dataset = load_response_dataset(data_config)

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


@pytest.mark.hf_gated
@pytest.mark.skip(reason="dataset download is flaky")
def test_squad_dataset():
    # load the dataset
    data_config = {
        "dataset_name": "squad",
        "prompt_file": None,
        "system_prompt_file": None,
    }
    squad_dataset = load_response_dataset(data_config)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # check that the dataset is formatted correctly
    for example in squad_dataset.formatted_ds["train"].take(5):
        assert "messages" in example
        assert len(example["messages"]) == 3

        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"

        template = "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer:'}}{%- elif message['role'] == 'assistant'  %}{{' ' + message['content'].strip()}}{%- endif %}{% endfor %}"

        ## check that applying chat template works as expected
        default_templated = tokenizer.apply_chat_template(
            example["messages"],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

        assert default_templated == (
            "Context: "
            + example["messages"][0]["content"]
            + " Question: "
            + example["messages"][1]["content"]
            + " Answer: "
            + example["messages"][2]["content"]
        )


def test_load_dataset_saved_with_save_to_disk():
    """Test loading a dataset that was saved using HuggingFace's save_to_disk().

    This tests the fix for datasets that already have a 'messages' column,
    which should be preserved without applying add_messages_key again.
    """
    from datasets import Dataset

    # Create a dataset with 'messages' column already present
    train_data = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ]
        },
    ]
    val_data = [
        {
            "messages": [
                {"role": "user", "content": "What is 3+3?"},
                {"role": "assistant", "content": "6"},
            ]
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create HF datasets and save using save_to_disk
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        train_path = f"{tmpdir}/train"
        val_path = f"{tmpdir}/val"

        train_dataset.save_to_disk(train_path)
        val_dataset.save_to_disk(val_path)

        # Load using load_response_dataset
        data_config = {
            "dataset_name": "ResponseDataset",
            "train_data_path": train_path,
            "val_data_path": val_path,
        }
        dataset = load_response_dataset(data_config)

        # Verify the dataset loaded correctly
        assert "train" in dataset.formatted_ds
        assert "validation" in dataset.formatted_ds
        assert len(dataset.formatted_ds["train"]) == 2
        assert len(dataset.formatted_ds["validation"]) == 1

        # Verify messages are preserved correctly
        first_train_example = dataset.formatted_ds["train"][0]
        assert "messages" in first_train_example
        assert len(first_train_example["messages"]) == 2
        assert first_train_example["messages"][0]["role"] == "user"
        assert first_train_example["messages"][0]["content"] == "What is 2+2?"
        assert first_train_example["messages"][1]["role"] == "assistant"
        assert first_train_example["messages"][1]["content"] == "4"

        # Verify validation data
        first_val_example = dataset.formatted_ds["validation"][0]
        assert first_val_example["messages"][0]["content"] == "What is 3+3?"
        assert first_val_example["messages"][1]["content"] == "6"
