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

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import load_response_dataset
from nemo_rl.data.datasets.response_datasets.clevr import format_clevr_cogent_dataset
from nemo_rl.data.datasets.response_datasets.geometry3k import format_geometry3k_dataset


def create_sample_data(input_key, output_key):
    data = [
        {input_key: "Hello", output_key: "Hi there!"},
        {input_key: "How are you?", output_key: "I'm good, thanks!"},
    ]

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


@pytest.mark.parametrize(
    "input_key,output_key", [("input", "output"), ("question", "answer")]
)
def test_response_dataset(input_key, output_key, tokenizer):
    # load the dataset
    data_path = create_sample_data(input_key, output_key)
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


@pytest.mark.parametrize("output_key", ["expected_answer", "generated_solution"])
def test_openmathinstruct2_dataset(output_key, tokenizer):
    # load the dataset
    data_config = {
        "dataset_name": "OpenMathInstruct-2",
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

    assert first_example["messages"][0]["content"][:20] == "An octahedron has ei"
    if output_key == "expected_answer":
        assert first_example["messages"][1]["content"][:20] == "\\frac{8\\sqrt{3}}{3}"
    elif output_key == "generated_solution":
        assert first_example["messages"][1]["content"][:20] == "Let's denote the poi"

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


@pytest.mark.parametrize(
    "dataset_name", ["DAPOMath17K", "DAPOMathAIME2024", "DeepScaler", "AIME2024"]
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

    if dataset_name == "DAPOMath17K":
        assert first_example["messages"][1]["content"] == "34"
    elif dataset_name == "DAPOMathAIME2024":
        assert first_example["messages"][1]["content"] == "540"
    elif dataset_name == "DeepScaler":
        assert first_example["messages"][1]["content"] == "-\\frac{2}{3}"
    elif dataset_name == "AIME2024":
        assert first_example["messages"][1]["content"] == "204"
        assert len(dataset.dataset) == 480

    # check the combined message
    chat_template = "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer:'}}{%- elif message['role'] == 'assistant'  %}{{' ' + message['content'].strip()}}{%- endif %}{% endfor %}"
    combined_message = tokenizer.apply_chat_template(
        first_example["messages"],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    assert combined_message == (
        " Question: "
        + first_example["messages"][0]["content"]
        + " Answer: "
        + first_example["messages"][1]["content"]
    )


@pytest.mark.parametrize(
    "dataset_name,format_func",
    [
        ("clevr-cogent", format_clevr_cogent_dataset),
        ("geometry3k", format_geometry3k_dataset),
        # this needs download 13.5G image
        # ("refcoco", format_refcoco_dataset),
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

    # check content
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
