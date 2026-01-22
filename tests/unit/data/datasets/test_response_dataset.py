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

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import load_response_dataset
from nemo_rl.data.datasets.response_datasets.clevr import format_clevr_cogent_dataset
from nemo_rl.data.datasets.response_datasets.geometry3k import format_geometry3k_dataset


def create_sample_data(input_key, output_key, is_save_to_disk=False):
    data = [
        {input_key: "Hello", output_key: "Hi there!"},
        {input_key: "How are you?", output_key: "I'm good, thanks!"},
    ]

    # Create temporary dataset file
    if is_save_to_disk:
        data_path = tempfile.mktemp()
        dataset = Dataset.from_list(data)
        dataset.save_to_disk(data_path)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            data_path = f.name

    return data_path


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer = get_tokenizer({"name": "Qwen/Qwen3-0.6B"})
    return tokenizer


@pytest.mark.parametrize(
    "input_key,output_key", [("input", "output"), ("question", "answer")]
)
@pytest.mark.parametrize("is_save_to_disk", [True, False])
def test_response_dataset(input_key, output_key, is_save_to_disk, tokenizer):
    # load the dataset
    data_path = create_sample_data(input_key, output_key, is_save_to_disk)
    data_config = {
        "dataset_name": "ResponseDataset",
        "data_path": data_path,
        "input_key": input_key,
        "output_key": output_key,
    }
    dataset = load_response_dataset(data_config)

    # check the input and output keys
    assert dataset.input_key == input_key
    assert dataset.output_key == output_key

    # check the first example
    first_example = dataset.dataset[0]

    # only contains messages and task_name
    assert len(first_example.keys()) == 2
    assert "messages" in first_example
    assert "task_name" in first_example

    # check the combined message
    chat_template = "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer:'}}{%- elif message['role'] == 'assistant'  %}{{' ' + message['content'].strip()}}{%- endif %}{% endfor %}"
    combined_message = tokenizer.apply_chat_template(
        first_example["messages"],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    assert combined_message == " Question: Hello Answer: Hi there!"


def test_helpsteer3_dataset():
    # load the dataset
    data_config = {"dataset_name": "HelpSteer3"}
    dataset = load_response_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]

    # only contains messages and task_name
    assert len(first_example.keys()) == 3
    assert "context" in first_example
    assert "response" in first_example
    assert "task_name" in first_example

    # check the content
    assert len(first_example["context"]) == 7
    assert first_example["response"][0]["role"] == "assistant"
    assert first_example["response"][0]["content"][:20] == "Yes, you are correct"


def test_open_assistant_dataset():
    # load the dataset
    data_config = {
        "dataset_name": "open_assistant",
        "split_validation_size": 0.05,
    }
    dataset = load_response_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]
    first_val_example = dataset.val_dataset[0]

    # only contains messages and task_name
    assert len(first_example.keys()) == 2
    assert "messages" in first_example
    assert "task_name" in first_example

    # check the content
    assert first_example["messages"][-1]["content"][:20] == "```\n    def forward("
    assert len(first_example["messages"]) == 7
    assert first_val_example["messages"][-1]["content"][:20] == "The colors you shoul"
    assert len(first_val_example["messages"]) == 5


@pytest.mark.parametrize(
    "dataset_name",
    ["DAPOMath17K", "DAPOMathAIME2024", "DeepScaler", "AIME2024", "squad"],
)
def test_build_in_dataset(dataset_name, tokenizer):
    # load the dataset
    data_config = {"dataset_name": dataset_name}
    dataset = load_response_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]

    # only contains messages and task_name
    assert len(first_example.keys()) == 2
    assert "messages" in first_example
    assert "task_name" in first_example

    # check the content
    if dataset_name == "DAPOMath17K":
        assert first_example["messages"][1]["content"] == "34"
    elif dataset_name == "DAPOMathAIME2024":
        assert first_example["messages"][1]["content"] == "540"
    elif dataset_name == "DeepScaler":
        assert first_example["messages"][1]["content"] == "-\\frac{2}{3}"
    elif dataset_name == "AIME2024":
        assert first_example["messages"][1]["content"] == "204"
        assert len(dataset.dataset) == 480
    elif dataset_name == "squad":
        assert first_example["messages"][2]["content"] == "Saint Bernadette Soubirous"

    # check the combined message
    chat_template = "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer:'}}{%- elif message['role'] == 'assistant'  %}{{' ' + message['content'].strip()}}{%- endif %}{% endfor %}"
    combined_message = tokenizer.apply_chat_template(
        first_example["messages"],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    if dataset_name == "squad":
        assert combined_message == (
            "Context: "
            + first_example["messages"][0]["content"]
            + " Question: "
            + first_example["messages"][1]["content"]
            + " Answer: "
            + first_example["messages"][2]["content"]
        )
    else:
        assert combined_message == (
            " Question: "
            + first_example["messages"][0]["content"]
            + " Answer: "
            + first_example["messages"][1]["content"]
        )


@pytest.mark.parametrize(
    "dataset_name,output_key",
    [
        ("OpenMathInstruct-2", "expected_answer"),
        ("OpenMathInstruct-2", "generated_solution"),
        ("tulu3_sft_mixture", None),
    ],
)
def test_build_in_dataset_with_split_validation(dataset_name, output_key, tokenizer):
    # load the dataset
    data_config = {
        "dataset_name": dataset_name,
        "output_key": output_key,
        "split_validation_size": 0.05,
    }
    dataset = load_response_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]
    first_val_example = dataset.val_dataset[0]

    # only contains messages and task_name
    assert len(first_example.keys()) == 2
    assert "messages" in first_example
    assert "task_name" in first_example

    # check the content
    if dataset_name == "OpenMathInstruct-2":
        if output_key == "expected_answer":
            assert first_example["messages"][1]["content"] == "\\frac{8\\sqrt{3}}{3}"
        elif output_key == "generated_solution":
            assert (
                first_example["messages"][1]["content"][:20] == "Let's denote the poi"
            )
    elif dataset_name == "tulu3_sft_mixture":
        assert first_example["messages"][1]["content"][:20] == "I'm sorry, but I can"

    # check the combined message
    messages = [first_example["messages"], first_val_example["messages"]]
    chat_template = "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer:'}}{%- elif message['role'] == 'assistant'  %}{{' ' + message['content'].strip()}}{%- endif %}{% endfor %}"
    combined_message = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    for i in range(2):
        assert combined_message[i] == (
            " Question: "
            + messages[i][0]["content"]
            + " Answer: "
            + messages[i][1]["content"]
        )


@pytest.mark.parametrize(
    "dataset_name,format_func",
    [
        ("clevr-cogent", format_clevr_cogent_dataset),
        ("geometry3k", format_geometry3k_dataset),
        # ("refcoco", format_refcoco_dataset), # this needs download 13.5G image
    ],
)
def test_vlm_dataset(dataset_name, format_func):
    # load the dataset
    data_config = {"dataset_name": dataset_name}
    dataset = load_response_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]
    first_example = format_func(first_example)

    # only contains messages and task_name
    assert len(first_example.keys()) == 2
    assert "messages" in first_example
    assert "task_name" in first_example

    # check the content
    assert first_example["messages"][0]["role"] == "user"
    assert first_example["messages"][0]["content"][0]["type"] == "image"
    assert first_example["messages"][0]["content"][1]["type"] == "text"
    assert first_example["messages"][1]["role"] == "assistant"

    if dataset_name == "clevr-cogent":
        assert first_example["messages"][1]["content"] == "3"
    elif dataset_name == "geometry3k":
        assert first_example["messages"][1]["content"] == "3"
    elif dataset_name == "refcoco":
        assert first_example["messages"][1]["content"] == "[243, 469, 558, 746]"
