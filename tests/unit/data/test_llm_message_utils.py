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
import torch
from transformers import AutoTokenizer

from nemo_rl.data.hf_datasets import COMMON_CHAT_TEMPLATES
from nemo_rl.data.interfaces import LLMMessageLogType, TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    _validate_tensor_consistency,
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
    get_first_index_that_differs,
    get_formatted_message_log,
    get_keys_from_message_log,
    message_log_to_flat_messages,
)


@pytest.fixture
def simple_message_log() -> LLMMessageLogType:
    """Fixture for a single message with tensor and text data."""
    return [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "text": "test",
        }
    ]


@pytest.fixture
def multiple_messages_log() -> LLMMessageLogType:
    """Fixture for multiple messages with tensor and text data."""
    return [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "text": "first",
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
            "text": "second",
        },
    ]


@pytest.fixture
def uneven_message_logs() -> list[LLMMessageLogType]:
    """Fixture for message logs of different lengths."""
    return [
        [  # First sequence (shorter)
            {
                "input_ids": torch.tensor([1, 2]),
                "role": "user",
            }
        ],
        [  # Second sequence (longer)
            {
                "input_ids": torch.tensor([3, 4, 5]),
                "role": "assistant",
            }
        ],
    ]


@pytest.fixture
def raw_chat_message_log() -> list[LLMMessageLogType]:
    """Fixture for chat message logs."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]


@pytest.fixture
def tokenized_non_chat_message_log() -> list[LLMMessageLogType]:
    return [
        [
            {
                "text": "some input text",
                "token_ids": torch.tensor([0, 1, 2, 3, 4, 5, 6]),
                "context_length": 3,
                "answer_length": 4,
            }
        ]
    ]


@pytest.fixture
def tokenized_chat_message_log() -> list[LLMMessageLogType]:
    return [
        [
            {
                "role": "system",
                "content": "system message",
                "token_ids": torch.tensor([0, 1, 2, 3, 4, 5]),
            },
            {
                "role": "user",
                "content": "user message",
                "token_ids": torch.tensor([6, 7, 8]),
            },
            {
                "role": "assistant",
                "content": "assistant message",
                "token_ids": torch.tensor([9, 10]),
            },
        ]
    ]


def test_message_log_to_flat_messages_empty() -> None:
    """Test message_log_to_flat_messages with empty input."""
    result = message_log_to_flat_messages([])
    assert result == {}, "Empty input should return empty dictionary"


def test_message_log_to_flat_messages_missing_keys() -> None:
    """Test message_log_to_flat_messages with messages having different keys."""
    message_log: LLMMessageLogType = [
        {"input_ids": torch.tensor([1, 2]), "text": "first"},
        {"input_ids": torch.tensor([3, 4]), "attention_mask": torch.tensor([1, 1])},
    ]
    result = message_log_to_flat_messages(message_log)
    assert torch.equal(result["input_ids"], torch.tensor([1, 2, 3, 4]))
    assert result["text"] == ["first"]
    assert torch.equal(result["attention_mask"], torch.tensor([1, 1]))


def test_concatenate_messages_different_shapes() -> None:
    """Test message_log_to_flat_messages with tensors of different shapes."""
    message_log: LLMMessageLogType = [
        {"input_ids": torch.tensor([[1, 2], [3, 4]])},  # 2D tensor
        {"input_ids": torch.tensor([5, 6])},  # 1D tensor
    ]
    with pytest.raises(
        RuntimeError,
        match=r"tensors for key='input_ids' must have same number of dimensions",
    ):
        message_log_to_flat_messages(message_log)


def test_get_keys_from_messages_empty() -> None:
    """Test get_keys_from_message_log with empty input."""
    assert get_keys_from_message_log([], ["key1"]) == []


def test_get_keys_from_messages_empty_keys() -> None:
    """Test get_keys_from_message_log with empty keys list."""
    message_log: LLMMessageLogType = [{"key1": "val1"}]
    assert get_keys_from_message_log(message_log, []) == [{}]


def test_get_keys_from_messages_all_missing() -> None:
    """Test get_keys_from_message_log when all requested keys are missing."""
    message_log: LLMMessageLogType = [{"key1": "val1"}]
    assert get_keys_from_message_log(message_log, ["nonexistent"]) == [{}]


def test_batch_pad_message_log_single_item() -> None:
    """Test batch_pad_message_log with single-item batch."""
    message_log_batch = [
        [{"input_ids": torch.tensor([1, 2, 3])}],
    ]
    result, input_lengths = batched_message_log_to_flat_message(message_log_batch)
    assert result["input_ids"].shape == (1, 3)
    assert input_lengths.shape == (1,)
    assert torch.equal(input_lengths, torch.tensor([3], dtype=torch.int32))


def test_batch_pad_message_log_empty_batch() -> None:
    """Test batch_pad_message_log with empty batch."""
    result, input_lengths = batched_message_log_to_flat_message([])
    assert len(result) == 0
    assert input_lengths.numel() == 0


def test_batch_pad_message_log_no_tensors() -> None:
    """Test batch_pad_message_log with messages containing no tensors."""
    message_log_batch = [
        [{"text": "first"}],
        [{"text": "second"}],
    ]
    result, input_lengths = batched_message_log_to_flat_message(message_log_batch)
    assert "text" in result
    assert isinstance(result["text"], list)
    assert result["text"] == ["first", "second"]
    assert input_lengths.numel() == 0


def test_batch_pad_messages_mixed_dtypes() -> None:
    """Test batch_pad_message_log with tensors of different dtypes."""
    message_log_batch = [
        [{"input_ids": torch.tensor([1, 2], dtype=torch.long)}],
        [{"input_ids": torch.tensor([3.0, 4.0, 5.0], dtype=torch.float)}],
    ]
    with pytest.raises(RuntimeError, match="expected consistent types"):
        batched_message_log_to_flat_message(message_log_batch)


@pytest.mark.parametrize("device", ["cuda", "meta"])
def test_batch_pad_message_log_different_devices(device: str) -> None:
    """Test batch_pad_message_log with tensors on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "meta" and not hasattr(torch.device(device), "type"):
        pytest.skip(f"Device {device} not available")

    message_log_batch = [
        [{"input_ids": torch.tensor([1, 2], device="cpu")}],
        [{"input_ids": torch.tensor([3, 4, 5], device=device)}],
    ]
    with pytest.raises(RuntimeError, match="expected tensors on the same device"):
        batched_message_log_to_flat_message(message_log_batch)


def test_message_log_to_flat_messages_single(
    simple_message_log: LLMMessageLogType,
) -> None:
    """Test message_log_to_flat_messages with a single message."""
    result = message_log_to_flat_messages(simple_message_log)
    assert torch.equal(result["input_ids"], simple_message_log[0]["input_ids"])
    assert torch.equal(
        result["attention_mask"], simple_message_log[0]["attention_mask"]
    )
    assert result["text"] == [simple_message_log[0]["text"]]


def test_message_log_to_flat_messages_multiple(
    multiple_messages_log: LLMMessageLogType,
) -> None:
    """Test message_log_to_flat_messages with multiple messages."""
    result = message_log_to_flat_messages(multiple_messages_log)
    assert torch.equal(result["input_ids"], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(result["attention_mask"], torch.tensor([1, 1, 1, 1]))
    assert result["text"] == ["first", "second"]


def test_get_keys_from_messages() -> None:
    """Test get_keys_from_message_log with various key combinations."""
    message_log: LLMMessageLogType = [
        {"key1": "val1", "key2": "val2", "key3": "val3"},
        {"key1": "val4", "key2": "val5", "key3": "val6"},
    ]

    # Test getting all keys
    result = get_keys_from_message_log(message_log, ["key1", "key2", "key3"])
    assert result == message_log

    # Test getting subset of keys
    result = get_keys_from_message_log(message_log, ["key1", "key2"])
    assert result == [
        {"key1": "val1", "key2": "val2"},
        {"key1": "val4", "key2": "val5"},
    ]

    # Test with non-existent key
    result = get_keys_from_message_log(message_log, ["key1", "nonexistent"])
    assert result == [{"key1": "val1"}, {"key1": "val4"}]


@pytest.mark.parametrize("make_sequence_length_divisible_by", [1, 8])
def test_batch_pad_message_log_divisible_by(
    uneven_message_logs: list[LLMMessageLogType], make_sequence_length_divisible_by: int
) -> None:
    """Test batch_pad_message_log padding to a multiple."""
    result, input_lengths = batched_message_log_to_flat_message(
        uneven_message_logs,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    batch_size, sequence_length = result["input_ids"].shape
    # Check shapes
    assert input_lengths.shape == (2,) == (batch_size,)
    assert sequence_length % make_sequence_length_divisible_by == 0


def test_batch_pad_message_log_basic(
    uneven_message_logs: list[LLMMessageLogType],
) -> None:
    """Test batch_pad_message_log with right padding."""
    result, input_lengths = batched_message_log_to_flat_message(uneven_message_logs)

    # Check shapes
    assert result["input_ids"].shape == (2, 3)
    assert input_lengths.shape == (2,)

    # Expected tensors for right padding
    expected_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    expected_lengths = torch.tensor([2, 3], dtype=torch.int32)

    assert torch.equal(result["input_ids"], expected_ids)
    assert torch.equal(input_lengths, expected_lengths)


def test_batch_pad_message_log_custom_pad_value(
    uneven_message_logs: list[LLMMessageLogType],
) -> None:
    """Test batch_pad_message_log with custom padding values."""
    pad_value_dict: dict[str, int] = {"input_ids": -100}
    result, input_lengths = batched_message_log_to_flat_message(
        uneven_message_logs, pad_value_dict=pad_value_dict
    )

    assert torch.equal(
        result["input_ids"],
        torch.tensor([[1, 2, -100], [3, 4, 5]]),
    )
    assert torch.equal(
        input_lengths,
        torch.tensor([2, 3], dtype=torch.int32),
    )


def test_get_formatted_message_log_llama(
    raw_chat_message_log: LLMMessageLogType,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    ## get expected result
    formatted_system_message = tokenizer.apply_chat_template(
        [raw_chat_message_log[0]],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    formatted_user_message = tokenizer.apply_chat_template(
        [raw_chat_message_log[1]],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    formatted_assistant_message = tokenizer.apply_chat_template(
        [raw_chat_message_log[2]],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    ## text should be equivalent to if we apply chat template
    ## to each turn separately and manually remove the bot string
    ## from the intermediate turns
    bot_str = "<|begin_of_text|>"
    expected_text = [
        formatted_system_message,
        formatted_user_message[len(bot_str) :],
        formatted_assistant_message[len(bot_str) :],
    ]

    task_data_spec = TaskDataSpec(
        task_name="test",
    )
    result = get_formatted_message_log(raw_chat_message_log, tokenizer, task_data_spec)
    actual_text = [m["content"] for m in result]

    assert actual_text == expected_text


def test_get_formatted_message_log_add_generation_prompt_llama(
    raw_chat_message_log: LLMMessageLogType,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    ## get expected result
    formatted_system_message = tokenizer.apply_chat_template(
        [raw_chat_message_log[0]],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    formatted_user_message = tokenizer.apply_chat_template(
        [raw_chat_message_log[1]],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    formatted_assistant_message = (
        raw_chat_message_log[2]["content"] + tokenizer.eos_token
    )

    ## text should be equivalent to if we apply chat template
    ## to each turn separately and manually remove the bot string
    ## from the intermediate turns
    bot_str = "<|begin_of_text|>"
    expected_text = [
        formatted_system_message,
        formatted_user_message[len(bot_str) :],
        formatted_assistant_message,
    ]

    task_data_spec = TaskDataSpec(
        task_name="test",
    )
    result = get_formatted_message_log(
        raw_chat_message_log,
        tokenizer,
        task_data_spec,
        add_generation_prompt=True,
    )
    actual_text = [m["content"] for m in result]

    assert actual_text == expected_text


def test_get_formatted_message_log_qwen(
    raw_chat_message_log: LLMMessageLogType,
) -> None:
    ## test using a tokenizer that does not have a bos token
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    assert tokenizer.bos_token is None

    ## get expected result
    ## result is equivalent to if we apply chat template to the full message log,
    ## remove the trailing newline, and then partition by the delimiter
    expected_text_string = tokenizer.apply_chat_template(
        [raw_chat_message_log],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )[0].rstrip("\n")  ## remove trailing newline

    delimiter = "<|im_end|>\n"
    split_text = expected_text_string.split(delimiter)
    expected_text = []
    for i in range(len(split_text)):
        if i == len(raw_chat_message_log) - 1:
            expected_text.append(split_text[i])
        else:
            expected_text.append(split_text[i] + delimiter)

    task_data_spec = TaskDataSpec(
        task_name="test",
    )
    result = get_formatted_message_log(raw_chat_message_log, tokenizer, task_data_spec)
    actual_text = [m["content"] for m in result]

    assert actual_text == expected_text


def test_get_formatted_message_log_add_generation_prompt_qwen(
    raw_chat_message_log: LLMMessageLogType,
) -> None:
    ## test using a tokenizer that does not have a bos token
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    assert tokenizer.bos_token is None

    ## get expected result
    ## result is equivalent to if we apply chat template to the full message log,
    ## remove the trailing newline, and then partition by the delimiter
    ## Separately handle the last message because of the generation prompt
    expected_text_string = tokenizer.apply_chat_template(
        [raw_chat_message_log[:2]],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )[0]

    delimiter = "<|im_end|>\n"
    split_text = expected_text_string.split(delimiter, 1)
    expected_text = []
    for i in range(len(split_text)):
        if i == len(split_text) - 1:
            expected_text.append(split_text[i])
        else:
            expected_text.append(split_text[i] + delimiter)

    formatted_assistant_message = (
        raw_chat_message_log[2]["content"] + tokenizer.eos_token
    )
    expected_text.append(formatted_assistant_message)

    task_data_spec = TaskDataSpec(
        task_name="test",
    )
    result = get_formatted_message_log(
        raw_chat_message_log,
        tokenizer,
        task_data_spec,
        add_generation_prompt=True,
    )
    actual_text = [m["content"] for m in result]

    assert actual_text == expected_text


def test_formatted_message_log_empty_message():
    message_logs = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ""},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    ]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.chat_template = COMMON_CHAT_TEMPLATES.passthrough_prompt_response
    task_data_spec = TaskDataSpec(task_name="test")
    result = [
        get_formatted_message_log(
            message_log,
            tokenizer,
            task_data_spec,
            add_bos_token=False,
            add_eos_token=False,
        )
        for message_log in message_logs
    ]
    flat_result = [message_log_to_flat_messages(m) for m in result]
    for k in flat_result[0].keys():
        if isinstance(flat_result[0][k], torch.Tensor):
            # make sure validate_tensor_consistency does not raise an error when one of the messages is empty
            _validate_tensor_consistency(
                [flat_result[i][k] for i in range(len(flat_result))]
            )


def test_add_loss_mask_to_chat_message_log(
    tokenized_chat_message_log: list[LLMMessageLogType],
):
    add_loss_mask_to_message_log(
        tokenized_chat_message_log, roles_to_train_on=["assistant"]
    )
    assert torch.equal(
        tokenized_chat_message_log[0][0]["token_loss_mask"],
        torch.tensor([0, 0, 0, 0, 0, 0]),
    )
    assert torch.equal(
        tokenized_chat_message_log[0][1]["token_loss_mask"], torch.tensor([0, 0, 0])
    )
    assert torch.equal(
        tokenized_chat_message_log[0][2]["token_loss_mask"], torch.tensor([1, 1])
    )

    ## test training on multiple roles
    add_loss_mask_to_message_log(
        tokenized_chat_message_log,
        roles_to_train_on=["assistant", "system"],
    )
    assert torch.equal(
        tokenized_chat_message_log[0][0]["token_loss_mask"],
        torch.tensor([1, 1, 1, 1, 1, 1]),
    )
    assert torch.equal(
        tokenized_chat_message_log[0][1]["token_loss_mask"], torch.tensor([0, 0, 0])
    )
    assert torch.equal(
        tokenized_chat_message_log[0][2]["token_loss_mask"], torch.tensor([1, 1])
    )

    ## test only unmasking final message
    add_loss_mask_to_message_log(
        tokenized_chat_message_log,
        only_unmask_final=True,
    )
    assert torch.equal(
        tokenized_chat_message_log[0][0]["token_loss_mask"],
        torch.tensor([0, 0, 0, 0, 0, 0]),
    )
    assert torch.equal(
        tokenized_chat_message_log[0][1]["token_loss_mask"], torch.tensor([0, 0, 0])
    )
    assert torch.equal(
        tokenized_chat_message_log[0][2]["token_loss_mask"], torch.tensor([1, 1])
    )


def test_get_first_index_that_differs():
    assert get_first_index_that_differs("hello", "hello") == 5
    assert get_first_index_that_differs("hello", "hello world") == 5
    assert get_first_index_that_differs("hello world", "hello") == 5
    assert get_first_index_that_differs("hi1", "hello2") == 1
    assert get_first_index_that_differs("hello2", "hi1") == 1
