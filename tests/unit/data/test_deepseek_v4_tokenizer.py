# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pickle

import pytest

from nemo_rl.data.deepseek_v4_tokenizer import (
    get_deepseek_v4_tokenizer,
    should_use_deepseek_v4_chat_template,
)


class DummyTokenizer:
    """Minimal stand-in for a fast tokenizer, picklable for __reduce__ tests."""

    vocab_size = 100

    def get_added_vocab(self):
        return {"<|extra_0|>": 100, "<|extra_1|>": 101}

    def encode(self, text, add_special_tokens=False, **kwargs):
        assert add_special_tokens is False
        max_length = kwargs.get("max_length")
        token_ids = [ord(character) % 256 for character in text]
        if max_length is not None and kwargs.get("truncation"):
            token_ids = token_ids[:max_length]
        return token_ids


@pytest.mark.parametrize(
    ("tokenizer_config", "expected"),
    [
        ({"chat_template": "deepseek_v4"}, True),
        ({"chat_template": None}, False),
        ({"chat_template": "passthrough"}, False),
        ({}, False),
    ],
)
def test_should_use_deepseek_v4_chat_template(tokenizer_config, expected):
    assert should_use_deepseek_v4_chat_template(tokenizer_config) is expected


def test_chat_mode_closes_thinking_block():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}],
        tokenize=False,
    )

    assert (
        prompt == "<｜begin▁of▁sentence｜><｜User｜>Solve 1+1.<｜Assistant｜></think>"
    )


def test_thinking_mode_opens_thinking_block():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}],
        tokenize=False,
        enable_thinking=True,
    )

    assert prompt.endswith("<｜Assistant｜><think>")


def test_reasoning_effort_none_forces_chat_mode():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt.endswith("</think>")
    assert "Reasoning Effort" not in prompt


def test_reasoning_effort_max_injects_maximum_effort_preamble():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())

    max_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )
    high_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="medium",
    )

    assert "Reasoning Effort: Absolute maximum" in max_prompt
    assert "Reasoning Effort: Absolute maximum" not in high_prompt


def test_tools_are_rendered_as_a_leading_system_message():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=tools,
        tokenize=False,
    )

    assert "get_weather" in prompt
    assert prompt.index("get_weather") < prompt.index("<｜User｜>")


def test_tokenize_encodes_the_rendered_prompt_without_special_tokens():
    base_tokenizer = DummyTokenizer()
    tokenizer = get_deepseek_v4_tokenizer(base_tokenizer)
    messages = [{"role": "user", "content": "Solve 1+1."}]

    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer.apply_chat_template(messages)

    assert token_ids == base_tokenizer.encode(rendered)


def test_tokenize_forwards_truncation_kwargs():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())

    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}],
        truncation=True,
        max_length=5,
    )

    assert len(token_ids) == 5


def test_len_includes_added_vocab():
    base_tokenizer = DummyTokenizer()
    tokenizer = get_deepseek_v4_tokenizer(base_tokenizer)

    assert len(tokenizer) == base_tokenizer.vocab_size + len(
        base_tokenizer.get_added_vocab()
    )


def test_get_added_vocab_returns_a_defensive_copy():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())

    added_vocab = tokenizer.get_added_vocab()
    added_vocab["<|mutated|>"] = 999

    assert "<|mutated|>" not in tokenizer.get_added_vocab()


def test_num_special_tokens_to_add_is_zero_for_raw_encoding():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer())

    assert tokenizer.num_special_tokens_to_add() == 0


def test_wrapper_survives_pickling_for_ray_and_vllm_workers():
    """The tokenizer is shipped to Ray actors, so the dynamic class must pickle."""
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer(), {"enable_thinking": True})

    restored = pickle.loads(pickle.dumps(tokenizer))

    assert restored.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}], tokenize=False
    ) == tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}], tokenize=False
    )
    assert restored.apply_chat_template(
        [{"role": "user", "content": "Solve 1+1."}], tokenize=False
    ).endswith("<｜Assistant｜><think>")


def test_keyword_conversation_and_call_kwargs_override_defaults():
    tokenizer = get_deepseek_v4_tokenizer(DummyTokenizer(), {"enable_thinking": True})

    prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "Solve 1+1."}],
        tokenize=False,
        enable_thinking=False,
    )

    assert prompt.endswith("</think>")


def test_wrapping_does_not_mutate_the_base_tokenizer():
    base_tokenizer = DummyTokenizer()

    get_deepseek_v4_tokenizer(base_tokenizer)

    assert type(base_tokenizer) is DummyTokenizer
