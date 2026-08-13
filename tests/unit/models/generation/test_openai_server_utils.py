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
"""Tests for the shared on-policy prefix splice used by the vLLM/TRT-LLM OpenAI servers."""

from collections import UserDict
from copy import deepcopy
from types import MappingProxyType

import pytest

from nemo_rl.models.generation.openai_server_utils import (
    CompleteTokenInjectionError,
    _coerce_token_id_list,
    build_complete_prompt_token_ids,
    find_latest_tokenized_assistant,
    replace_prefix_tokens,
)


class _CompleteTokenTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        token_ids = []
        while text:
            if text.startswith(self.eos_token):
                token_ids.append(self.eos_token_id)
                text = text[len(self.eos_token) :]
            elif text.startswith("next"):
                token_ids.append(40)
                text = text[len("next") :]
            elif text.startswith("GEN"):
                token_ids.append(99)
                text = text[len("GEN") :]
            else:
                raise AssertionError(f"Unexpected text to encode: {text!r}")
        return token_ids


class _GemmaStyleTokenizer(_CompleteTokenTokenizer):
    eos_token = "<eos>"
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        token_ids = []
        while text:
            if text.startswith("<end_of_turn>"):
                token_ids.append(106)
                text = text[len("<end_of_turn>") :]
            elif text.startswith("\n"):
                token_ids.append(107)
                text = text[1:]
            elif text.startswith("next"):
                token_ids.append(40)
                text = text[len("next") :]
            elif text.startswith("GEN"):
                token_ids.append(99)
                text = text[len("GEN") :]
            else:
                raise AssertionError(f"Unexpected text to encode: {text!r}")
        return token_ids


def _render_marked_conversation(conversation):
    rendered = ""
    for message in conversation:
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", tool_call)
            assert isinstance(function.get("arguments"), dict)

        role = message["role"]
        content = message.get("content")
        if role == "user" and content == "hello":
            rendered += "hello"
        elif role == "user" and content == "next":
            rendered += "next"
        elif role == "assistant":
            rendered += f"{content}<eos>"
        else:
            rendered += "other"
    rendered += "GEN"
    return rendered


def test_build_complete_prompt_token_ids_preserves_prefix_and_suffix() -> None:
    tokenizer = _CompleteTokenTokenizer()
    conversation = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "older tool call",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": {"command": "pwd"},
                    },
                }
            ],
        },
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
    ]
    original_conversation = deepcopy(conversation)

    result = build_complete_prompt_token_ids(
        tokenizer=tokenizer,
        conversation=conversation,
        assistant_ordinal=1,
        model_prefix_token_ids=[10, 777, 2, 40, 31, 32, 2],
        render_prompt_text=_render_marked_conversation,
    )

    assert result == [10, 777, 2, 40, 31, 32, 2, 40, 99]
    assert conversation == original_conversation


@pytest.mark.parametrize(
    ("model_prefix_token_ids", "expected"),
    [
        pytest.param(
            [31, 32, 106, 107],
            [31, 32, 106, 107, 40, 99],
            id="close_present",
        ),
        pytest.param(
            [31, 32],
            [31, 32, 106, 107, 40, 99],
            id="close_missing",
        ),
    ],
)
def test_complete_token_injection_uses_template_assistant_close(
    model_prefix_token_ids, expected
) -> None:
    marker = "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E"

    result = build_complete_prompt_token_ids(
        tokenizer=_GemmaStyleTokenizer(),
        conversation=[
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "next"},
        ],
        assistant_ordinal=0,
        model_prefix_token_ids=model_prefix_token_ids,
        render_prompt_text=lambda messages: f"{marker}<end_of_turn>\nnextGEN",
        render_assistant_close_text=lambda messages: f"{marker}<end_of_turn>\n",
    )

    assert result == expected


@pytest.mark.parametrize(
    ("messages", "expected_ordinal", "expected_index"),
    [
        pytest.param([], None, None, id="empty"),
        pytest.param(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            None,
            None,
            id="no_token_metadata",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": "question one",
                    "prompt_token_ids": [90],
                    "generation_token_ids": [91],
                },
                {
                    "role": "assistant",
                    "content": "answer one",
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                },
                {
                    "role": "tool",
                    "content": "result",
                    "prompt_token_ids": [92],
                    "generation_token_ids": [93],
                },
                {"role": "user", "content": "question two"},
                {
                    "role": "assistant",
                    "content": "answer two",
                    "prompt_token_ids": [3],
                    "generation_token_ids": [4],
                },
                {"role": "user", "content": "question three"},
            ],
            1,
            4,
            id="latest_assistant_ignores_other_roles",
        ),
        pytest.param(
            [
                {
                    "role": "assistant",
                    "content": "partial",
                    "prompt_token_ids": [1],
                }
            ],
            None,
            None,
            id="partial_metadata",
        ),
    ],
)
def test_complete_token_injection_finds_latest_tokenized_assistant(
    messages, expected_ordinal, expected_index
) -> None:
    """The selector counts assistant messages and returns the latest full record."""
    result = find_latest_tokenized_assistant(messages)
    if expected_ordinal is None:
        assert result is None
    else:
        assert result is not None
        assistant_ordinal, message = result
        assert assistant_ordinal == expected_ordinal
        assert message is messages[expected_index]


@pytest.mark.parametrize("gap", ["", " \n\t"])
def test_complete_token_injection_tolerates_whitespace_before_assistant_eos(
    gap,
) -> None:
    tokenizer = _CompleteTokenTokenizer()
    marker = "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E"

    result = build_complete_prompt_token_ids(
        tokenizer=tokenizer,
        conversation=[
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "next"},
        ],
        assistant_ordinal=0,
        model_prefix_token_ids=[31, 32, 2],
        render_prompt_text=lambda messages: f"{marker}{gap}<eos>nextGEN",
    )

    assert result == [31, 32, 2, 40, 99]


def test_complete_token_injection_rejects_intervening_content_before_eos() -> None:
    tokenizer = _CompleteTokenTokenizer()

    with pytest.raises(CompleteTokenInjectionError, match="did not close.*immediately"):
        build_complete_prompt_token_ids(
            tokenizer=tokenizer,
            conversation=[
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "next"},
            ],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1Eintervening<eos>nextGEN"
            ),
        )


def test_complete_token_injection_requires_assistant_eos() -> None:
    tokenizer = _CompleteTokenTokenizer()
    conversation = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
    ]

    with pytest.raises(
        CompleteTokenInjectionError,
        match="did not close the tokenized assistant with EOS",
    ):
        build_complete_prompt_token_ids(
            tokenizer=tokenizer,
            conversation=conversation,
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E missing boundary close"
            ),
        )


@pytest.mark.parametrize(
    "value",
    [pytest.param((2, 40), id="non_list"), pytest.param(None, id="none")],
)
def test_complete_token_injection_rejects_non_list_token_ids(value) -> None:
    with pytest.raises(CompleteTokenInjectionError, match="must be a list"):
        _coerce_token_id_list(value, "Token IDs")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(["2"], id="string"),
        pytest.param([2.9], id="float"),
        pytest.param([True], id="bool"),
    ],
)
def test_complete_token_injection_rejects_non_integer_token_ids(value) -> None:
    with pytest.raises(CompleteTokenInjectionError, match="only integer token IDs"):
        _coerce_token_id_list(value, "Token IDs")


def test_complete_token_injection_copies_valid_token_ids() -> None:
    token_ids = [2, 40]

    result = _coerce_token_id_list(token_ids, "Token IDs")

    assert result == token_ids
    assert result is not token_ids


def test_complete_token_injection_wraps_copy_failure() -> None:
    conversation = [MappingProxyType({"role": "assistant", "content": "first"})]

    with pytest.raises(CompleteTokenInjectionError, match="could not be copied"):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=conversation,
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=_render_marked_conversation,
        )


def test_complete_token_injection_rejects_marker_collision() -> None:
    with pytest.raises(CompleteTokenInjectionError, match="marker collided"):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[
                {
                    "role": "user",
                    "content": "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E",
                },
                {
                    "role": "assistant",
                    "content": "first",
                },
            ],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=_render_marked_conversation,
        )


def test_complete_token_injection_rejects_rendered_marker_collision() -> None:
    marker = "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E"

    with pytest.raises(
        CompleteTokenInjectionError, match="collided with rendered request data"
    ):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: f"{marker}{marker}<eos>nextGEN",
        )


def test_complete_token_injection_rejects_missing_assistant_ordinal() -> None:
    with pytest.raises(
        CompleteTokenInjectionError, match="not present after chat parsing"
    ):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=1,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=_render_marked_conversation,
        )


def test_complete_token_injection_rejects_non_dict_assistant() -> None:
    with pytest.raises(CompleteTokenInjectionError, match="not a message object"):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[UserDict({"role": "assistant", "content": "first"})],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=_render_marked_conversation,
        )


def test_complete_token_injection_preserves_render_fallback() -> None:
    def render_prompt_text(messages):
        raise CompleteTokenInjectionError("renderer requested native preprocessing")

    with pytest.raises(
        CompleteTokenInjectionError,
        match="renderer requested native preprocessing",
    ):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=render_prompt_text,
        )


def test_complete_token_injection_wraps_render_failure() -> None:
    def render_prompt_text(messages):
        raise RuntimeError("render failed")

    with pytest.raises(CompleteTokenInjectionError, match="could not be rendered"):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=render_prompt_text,
        )


@pytest.mark.parametrize(
    ("rendered", "expected_error"),
    [
        pytest.param([], "must render as text", id="non_text"),
        pytest.param("missing marker<eos>", "did not preserve", id="missing_marker"),
    ],
)
def test_complete_token_injection_rejects_invalid_render_output(
    rendered, expected_error
) -> None:
    with pytest.raises(CompleteTokenInjectionError, match=expected_error):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: rendered,
        )


@pytest.mark.parametrize(
    ("render_assistant_close_text", "expected_error"),
    [
        pytest.param(
            lambda messages: (_ for _ in ()).throw(
                CompleteTokenInjectionError("probe requested native preprocessing")
            ),
            "probe requested native preprocessing",
            id="native_fallback",
        ),
        pytest.param(
            lambda messages: (_ for _ in ()).throw(RuntimeError("probe failed")),
            "probe could not be rendered",
            id="render_failure",
        ),
        pytest.param(lambda messages: [], "must render as text", id="non_text"),
        pytest.param(
            lambda messages: "missing marker<eos>",
            "did not preserve the boundary marker",
            id="missing_marker",
        ),
        pytest.param(
            lambda messages: "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E \n\t",
            "did not emit an assistant-close sequence",
            id="empty_close",
        ),
        pytest.param(
            lambda messages: "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E<eos>",
            "assistant-close sequence changed",
            id="changed_close",
        ),
    ],
)
def test_complete_token_injection_rejects_invalid_close_probe(
    render_assistant_close_text, expected_error
) -> None:
    marker = "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E"

    with pytest.raises(CompleteTokenInjectionError, match=expected_error):
        build_complete_prompt_token_ids(
            tokenizer=_GemmaStyleTokenizer(),
            conversation=[
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "next"},
            ],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 106, 107],
            render_prompt_text=lambda messages: f"{marker}<end_of_turn>\nnextGEN",
            render_assistant_close_text=render_assistant_close_text,
        )


@pytest.mark.hf_gated
def test_complete_token_injection_matches_real_gemma_template() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    prior_user = {"role": "user", "content": "hello"}
    prior_assistant = {"role": "assistant", "content": "answer"}
    next_user = {"role": "user", "content": "next"}
    conversation = [prior_user, prior_assistant, next_user]
    first_prompt_token_ids = tokenizer.apply_chat_template(
        [prior_user], tokenize=True, add_generation_prompt=True
    )
    generation_token_ids = tokenizer.encode("answer", add_special_tokens=False)
    native_token_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )

    injected_token_ids = build_complete_prompt_token_ids(
        tokenizer=tokenizer,
        conversation=conversation,
        assistant_ordinal=0,
        model_prefix_token_ids=first_prompt_token_ids + generation_token_ids,
        render_prompt_text=lambda messages: tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        render_assistant_close_text=lambda messages: tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ),
    )

    assert injected_token_ids == native_token_ids


def test_complete_token_injection_requires_tokenizer_eos() -> None:
    tokenizer = _CompleteTokenTokenizer()
    tokenizer.eos_token = None

    with pytest.raises(CompleteTokenInjectionError, match="requires tokenizer EOS"):
        build_complete_prompt_token_ids(
            tokenizer=tokenizer,
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E<eos>nextGEN"
            ),
        )


def test_complete_token_injection_preserves_encoding_fallback() -> None:
    tokenizer = _CompleteTokenTokenizer()

    def encode(text, add_special_tokens=False):
        raise CompleteTokenInjectionError("encoder requested native preprocessing")

    tokenizer.encode = encode
    with pytest.raises(
        CompleteTokenInjectionError,
        match="encoder requested native preprocessing",
    ):
        build_complete_prompt_token_ids(
            tokenizer=tokenizer,
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E<eos>nextGEN"
            ),
        )


def test_complete_token_injection_wraps_encoding_failure() -> None:
    tokenizer = _CompleteTokenTokenizer()

    def encode(text, add_special_tokens=False):
        raise RuntimeError("encode failed")

    tokenizer.encode = encode
    with pytest.raises(CompleteTokenInjectionError, match="could not be tokenized"):
        build_complete_prompt_token_ids(
            tokenizer=tokenizer,
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E<eos>nextGEN"
            ),
        )


def test_complete_token_injection_requires_encoded_assistant_close() -> None:
    tokenizer = _CompleteTokenTokenizer()
    tokenizer.encode = lambda text, add_special_tokens=False: (
        [2] if text == "<eos>" else [40]
    )

    with pytest.raises(
        CompleteTokenInjectionError, match="did not begin with the assistant-close"
    ):
        build_complete_prompt_token_ids(
            tokenizer=tokenizer,
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E<eos>nextGEN"
            ),
        )


def test_complete_token_injection_rejects_empty_model_prefix() -> None:
    with pytest.raises(
        CompleteTokenInjectionError, match="prefix token IDs were empty"
    ):
        build_complete_prompt_token_ids(
            tokenizer=_CompleteTokenTokenizer(),
            conversation=[{"role": "assistant", "content": "first"}],
            assistant_ordinal=0,
            model_prefix_token_ids=[],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E<eos>nextGEN"
            ),
        )


def test_complete_token_injection_appends_eos_when_prefix_has_no_eos() -> None:
    result = build_complete_prompt_token_ids(
        tokenizer=_CompleteTokenTokenizer(),
        conversation=[{"role": "assistant", "content": "first"}],
        assistant_ordinal=0,
        model_prefix_token_ids=[31, 32],
        render_prompt_text=lambda messages: (
            "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E<eos>nextGEN"
        ),
    )

    assert result == [31, 32, 2, 40, 99]


def test_replace_prefix_tokens_empty_model_prefix_returns_template():
    """Turn 1 has no prior model output; the full template is returned unchanged."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = []
    template_prefix_token_ids = [9, 2]
    template_token_ids = [9, 2, 33, 44]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == template_token_ids


def test_replace_prefix_tokens_missing_eos_in_template_prefix_raises():
    """A template prefix with no EOS has no valid splice boundary and must raise."""

    class _T:
        eos_token_id = 2

        def decode(self, *args, **kwargs):
            pass

    tokenizer = _T()
    model_prefix_token_ids = [7, 2]
    template_prefix_token_ids = [9, 9, 9]
    template_token_ids = [9, 9, 9, 2, 10]
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )


def test_replace_prefix_tokens_tokenizer_without_eos_raises():
    """A tokenizer that has no EOS token cannot locate the splice boundary."""

    class _T:
        eos_token_id = None

    tokenizer = _T()
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=[1],
            template_prefix_token_ids=[1, 2],
            template_token_ids=[1, 2],
        )


def test_replace_prefix_tokens_uses_last_eos_in_template_prefix():
    """When the prefix contains multiple EOS tokens, the splice cuts at the last one."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = [100, 2]
    template_prefix_token_ids = [9, 2, 9, 2]
    template_token_ids = [9, 2, 9, 2, 77, 88]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == [100, 2, 77, 88]


def test_replace_prefix_tokens_qwen3_think_shift_picks_assistant_eos_not_user_eos():
    """Non-strict-prefix: Qwen3 strips <think> from history when the last message is a
    user turn, so the template's prefix region is shorter and a later user-turn EOS
    lands within the first len(template_prefix) positions. The count-based algorithm
    must cut at the assistant EOS and preserve the intervening user turn.
    """

    class _T:
        eos_token_id = 2

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids)

    tokenizer = _T()
    model_prefix_token_ids = [11, 12, 99, 99, 99, 55, 2]
    template_prefix_token_ids = [11, 12, 88, 88, 88, 56, 2]
    template_token_ids = [11, 12, 56, 2, 70, 71, 2, 40, 41]

    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )

    assert result == [11, 12, 99, 99, 99, 55, 2, 70, 71, 2, 40, 41]
    assert 70 in result and 71 in result
