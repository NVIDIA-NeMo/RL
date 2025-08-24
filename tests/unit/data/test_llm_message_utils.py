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


from typing import Any, Callable

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

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


@pytest.mark.parametrize(
    "model_id, chat_log_transform",
    [
        pytest.param(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            lambda raw: raw,
            marks=pytest.mark.hf_gated,
            id="llama",
        ),
        pytest.param(
            "google/gemma-3-27b-it",
            # Some Gemma chat templates (or versions) raise on system turns.
            # For portability across environments, test on user+assistant only.
            # If your tokenizer supports system turns, you can change this to `lambda raw: raw`.
            lambda raw: [raw[1], raw[2]],
            marks=pytest.mark.hf_gated,
            id="gemma",
        ),
        pytest.param(
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            lambda raw: raw,
            id="qwen",
        ),
    ],
)
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_get_formatted_message_log_models(
    raw_chat_message_log: LLMMessageLogType,
    model_id: str,
    chat_log_transform: Callable[[Any], Any],
    add_generation_prompt: bool,
) -> None:
    """Validate that get_formatted_message_log produces text consistent with the
    tokenizer's chat template across models.

    This test is parametrized over model/tokenizer and whether to include a
    generation prompt. For models like Gemma that error on system turns, the
    input chat log is transformed to exclude the system message.

    Expectations:
    - Require an EOS token for well-defined end-of-turn comparison.
    - When add_generation_prompt is False, the concatenated contents must match
      the tokenizer's apply_chat_template output; if the tokenizer omits a final
      EOS, accept the actual with EOS by appending EOS to the expected before
      comparison.
    - When add_generation_prompt is True and the last turn is an assistant
      message, accept either:
        (1) prefix built with add_generation_prompt=True followed by the raw
            assistant content plus EOS; or
        (2) the tokenizer's full non-generation template output plus EOS.
      This avoids hard-coding model-specific headers or delimiters while still
      verifying semantic equivalence.
    - Only normalization performed is trimming a trailing newline after EOS.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chat_log = chat_log_transform(raw_chat_message_log)
    # Ensure tokenizer defines an EOS token; otherwise the test logic is ill-defined
    assert tokenizer.eos_token, "Tokenizer must define eos_token for this test"
    eos = tokenizer.eos_token
    task_data_spec = TaskDataSpec(task_name="test")
    result = get_formatted_message_log(
        chat_log,
        tokenizer,
        task_data_spec,
        add_generation_prompt=add_generation_prompt,
    )
    actual_concat = "".join(m["content"] for m in result)

    def normalize(s: str) -> str:
        # Normalize EOS+newline quirk to EOS only
        if s.endswith(eos + "\n"):
            return s[:-1]
        return s

    if not add_generation_prompt:
        expected_concat = tokenizer.apply_chat_template(
            [chat_log],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )[0]
        # Accept EOS presence even if the tokenizer's template omits it
        if actual_concat.endswith(eos) and not expected_concat.endswith(eos):
            expected_concat = expected_concat + eos
        assert normalize(actual_concat) == normalize(expected_concat)
    else:
        if len(chat_log) > 0 and chat_log[-1].get("role") == "assistant":
            prefix_log = chat_log[:-1]
            # Some tokenizers include a role header when add_generation_prompt=True.
            # Accept either behavior without hard-coding model-specific strings.
            prefix_gen = tokenizer.apply_chat_template(
                [prefix_log],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )[0]
            assistant_suffix = chat_log[-1]["content"] + eos
            expected_concat_a = prefix_gen + assistant_suffix
            # Alternative: take the full non-generation template output and just append EOS
            full_no_gen = tokenizer.apply_chat_template(
                [chat_log],
                tokenize=False,
                add_generation_prompt=False,
                add_special_tokens=False,
            )[0]
            expected_concat_b = full_no_gen + eos
            actual_norm = normalize(actual_concat)
            assert actual_norm == normalize(
                expected_concat_a
            ) or actual_norm == normalize(expected_concat_b)
        else:
            expected_concat = tokenizer.apply_chat_template(
                [chat_log],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )[0]
            assert normalize(actual_concat) == normalize(expected_concat)


@pytest.mark.hf_gated
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


def test_message_log_to_flat_messages_with_packed_images() -> None:
    from nemo_rl.data.multimodal_utils import PackedTensor

    # two turns, each with an image tensor wrapped in PackedTensor
    img1 = torch.randn(2, 3, 8, 8)
    img2 = torch.randn(3, 3, 8, 8)
    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": "see image",
            "token_ids": torch.tensor([1, 2]),
            "images": PackedTensor(img1, dim_to_pack=0),
        },
        {
            "role": "assistant",
            "content": "ok",
            "token_ids": torch.tensor([3]),
            "images": PackedTensor(img2, dim_to_pack=0),
        },
    ]
    flat = message_log_to_flat_messages(message_log)
    assert isinstance(flat["images"], PackedTensor)
    assert tuple(flat["images"].as_tensor().shape) == (5, 3, 8, 8)
    assert torch.equal(flat["token_ids"], torch.tensor([1, 2, 3]))


def test_batched_message_log_to_flat_message_with_packed_images() -> None:
    from nemo_rl.data.multimodal_utils import PackedTensor

    img_a = torch.randn(1, 3, 4, 4)
    img_b = torch.randn(2, 3, 4, 4)
    img_c = torch.randn(1, 3, 4, 4)

    batch_logs = [
        [
            {
                "role": "user",
                "content": "prompt a",
                "token_ids": torch.tensor([1, 2, 3]),
                "images": PackedTensor(img_a, dim_to_pack=0),
            },
            {"role": "assistant", "content": "resp", "token_ids": torch.tensor([4])},
        ],
        [
            {
                "role": "user",
                "content": "prompt b",
                "token_ids": torch.tensor([5, 6]),
                "images": PackedTensor(img_b, dim_to_pack=0),
            },
            {
                "role": "assistant",
                "content": "resp2",
                "token_ids": torch.tensor([7, 8]),
            },
            {
                "role": "user",
                "content": "again",
                "token_ids": torch.tensor([9]),
                "images": PackedTensor(img_c, dim_to_pack=0),
            },
        ],
    ]

    batched, input_lengths = batched_message_log_to_flat_message(
        batch_logs, pad_value_dict={"token_ids": 0}
    )
    assert isinstance(batched["images"], PackedTensor)
    # flattened_concat keeps two packed tensors (one per convo)
    assert len(batched["images"]) == 2
    # total packed along dim 0 = 1 + (2 + 1) = 4
    assert tuple(batched["images"].as_tensor().shape) == (4, 3, 4, 4)
    assert torch.equal(input_lengths, torch.tensor([4, 5], dtype=torch.int32))


@pytest.mark.hf_gated
def test_get_formatted_message_log_multimodal_prompt_formatting() -> None:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    task_data_spec = TaskDataSpec(task_name="t")
    task_data_spec.prompt = "Question: {} Answer:"

    # one user turn with text+image, then assistant
    image = Image.new("RGB", (16, 16), color=(0, 0, 0))
    message_log: LLMMessageLogType = [
        {
            "role": "system",
            "content": "",  # to prevent Qwen's default system prompt taking over
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a cat?"},
                {"type": "image", "image": image},
            ],
        },
        {"role": "assistant", "content": "okay"},
    ]

    out = get_formatted_message_log(
        message_log, processor, task_data_spec, add_bos_token=False, add_eos_token=False
    )
    # First message text should be formatted by prompt
    assert isinstance(out[1]["content"], list)
    assert any(
        item["type"] == "text"
        and item["text"].startswith("<|im_start|>user\nQuestion: ")
        for item in out[1]["content"]
    )  # type: ignore[index]
    # pixel_values should be added as PackedTensor for the first message
    from nemo_rl.data.multimodal_utils import PackedTensor

    assert isinstance(out[1]["pixel_values"], PackedTensor)
    assert isinstance(out[1]["image_grid_thw"], PackedTensor)
    pv = out[1]["pixel_values"].as_tensor()
    grid_thw = out[1]["image_grid_thw"].as_tensor()
    assert pv.ndim == 2 and pv.shape[1] == 1176
    assert grid_thw.ndim == 2 and grid_thw.shape == torch.Size([1, 3])
    # token_ids should be non-empty tensors
    assert (
        isinstance(out[1]["token_ids"], torch.Tensor)
        and out[1]["token_ids"].numel() > 0
    )
    assert (
        isinstance(out[2]["token_ids"], torch.Tensor)
        and out[2]["token_ids"].numel() > 0
    )

    #### Case 2 : without system prompt
    image = Image.new("RGB", (16, 16), color=(0, 0, 0))
    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a cat?"},
                {"type": "image", "image": image},
            ],
        },
        {"role": "assistant", "content": "okay"},
    ]

    out = get_formatted_message_log(
        message_log, processor, task_data_spec, add_bos_token=False, add_eos_token=False
    )
    # First message text should be formatted by prompt
    assert isinstance(out[0]["content"], list)
    assert any(
        item["type"] == "text"
        and item["text"].startswith(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nQuestion: "
        )
        for item in out[0]["content"]
    )  # type: ignore[index]
    # pixel_values should be added as PackedTensor for the first message
    from nemo_rl.data.multimodal_utils import PackedTensor

    assert isinstance(out[0]["pixel_values"], PackedTensor)
    assert isinstance(out[0]["image_grid_thw"], PackedTensor)
    pv = out[0]["pixel_values"].as_tensor()
    grid_thw = out[0]["image_grid_thw"].as_tensor()
    assert pv.ndim == 2 and pv.shape[1] == 1176
    assert grid_thw.ndim == 2 and grid_thw.shape == torch.Size([1, 3])
    # token_ids should be non-empty tensors
    assert (
        isinstance(out[0]["token_ids"], torch.Tensor)
        and out[0]["token_ids"].numel() > 0
    )
    assert (
        isinstance(out[1]["token_ids"], torch.Tensor)
        and out[1]["token_ids"].numel() > 0
    )
