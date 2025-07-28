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

from datetime import datetime

import pytest

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.hf_datasets.chat_templates import COMMON_CHAT_TEMPLATES


@pytest.fixture
def conversation_messages():
    """Fixture providing a multi-turn conversation for testing chat templates"""
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the weather like today?"},
        {
            "role": "assistant",
            "content": "I don't have access to real-time weather data.",
        },
        {"role": "user", "content": "Can you help me with something else then?"},
        {"role": "assistant", "content": "Of course! What would you like help with?"},
    ]


def get_expected_llama_format(messages):
    """Generate the expected output format for Llama's chat template"""
    # Extract the date from the formatted output
    # Get current date
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %b %Y")

    # Extract system message if present
    if messages[0]["role"] == "system":
        system_message = messages[0]["content"].strip()
        messages = messages[1:]
    else:
        system_message = ""

    # Start with BOS token and system header
    expected = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    expected += "Cutting Knowledge Date: December 2023\n"
    expected += f"Today Date: {formatted_date}\n\n"
    expected += f"{system_message}<|eot_id|>"

    # Add each message
    for message in messages:
        if message["role"] not in ["ipython", "tool"]:
            expected += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            expected += f"{message['content'].strip()}<|eot_id|>"

    return expected


def get_format_with_simple_role_header(messages):
    message = "<|begin_of_text|>"
    for msg in messages:
        message += (
            "<|start_header_id|>"
            + msg["role"]
            + "<|end_header_id|>\n\n"
            + msg["content"].strip()
            + "<|eot_id|>"
        )
    return message


@pytest.mark.hf_gated
def test_get_tokenizer_no_chat_template(conversation_messages):
    """Test get_tokenizer when no chat template is specified in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct"}
    tokenizer = get_tokenizer(config)

    # Verify that the tokenizer's default template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)

    expected = get_expected_llama_format(conversation_messages)
    assert formatted == expected


@pytest.mark.hf_gated
def test_get_tokenizer_default_chat_template(conversation_messages):
    """Test get_tokenizer when chat_template is 'default' in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct", "chat_template": "default"}
    tokenizer = get_tokenizer(config)

    # Verify that the tokenizer's default template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)
    expected = get_expected_llama_format(conversation_messages)
    assert formatted == expected


@pytest.mark.hf_gated
def test_get_tokenizer_null_chat_template(conversation_messages):
    """Test get_tokenizer when chat_template is None in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct", "chat_template": None}
    tokenizer = get_tokenizer(config)

    # Verify that the passthrough template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)

    expected = "".join(msg["content"] for msg in conversation_messages)

    assert formatted == expected


@pytest.mark.hf_gated
def test_get_tokenizer_custom_jinja_template(conversation_messages):
    """Test get_tokenizer when a custom jinja template is specified"""
    custom_template = COMMON_CHAT_TEMPLATES.simple_role_header
    config = {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "chat_template": custom_template,
    }
    tokenizer = get_tokenizer(config)

    # Verify that the custom template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)
    expected = get_format_with_simple_role_header(conversation_messages)
    assert formatted == expected
