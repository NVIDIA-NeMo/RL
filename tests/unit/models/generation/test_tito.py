# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from transformers import AutoTokenizer

from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens
from nemo_rl.models.generation.tito import (
    QUALIFICATION_HISTORY,
    QUALIFICATION_OBSERVATIONS,
    TitoBridgeError,
    bridge_suffix,
    qualify_bridge,
)

CLOSE = 9


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def closes(tokenizer):
    return {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}


def test_bridge_keeps_tokens_after_the_parent_turn_close():
    suffix = bridge_suffix(
        parent_token_ids=[1, 2, CLOSE, 3, 4, CLOSE],
        rendered_token_ids=[1, 2, CLOSE, 3, 5, 6, CLOSE, 7, 8, CLOSE, 10],
        turn_close_token_ids={CLOSE},
    )

    assert suffix == [7, 8, CLOSE, 10]


def test_bridge_adds_the_template_close_after_an_unfinished_turn():
    suffix = bridge_suffix(
        parent_token_ids=[1, 2, CLOSE, 3, 4],
        rendered_token_ids=[1, 2, CLOSE, 3, 4, CLOSE, 7, 8],
        turn_close_token_ids={CLOSE},
    )

    assert suffix == [CLOSE, 7, 8]


def test_bridge_rejects_a_history_shorter_than_the_parent():
    with pytest.raises(TitoBridgeError, match="1 turn-close tokens"):
        bridge_suffix(
            parent_token_ids=[1, CLOSE, 2, CLOSE],
            rendered_token_ids=[1, CLOSE, 3],
            turn_close_token_ids={CLOSE},
        )


def _render(tokenizer, messages, *, add_generation_prompt, **template_kwargs):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("observations", QUALIFICATION_OBSERVATIONS)
@pytest.mark.parametrize("ending", ["<|im_end|>", "<|endoftext|>", ""])
def test_bridge_matches_the_prefix_splice(
    tokenizer, closes, thinking, observations, ending
):
    """The bridged prompt equals the splice of a parent the template re-tokenizes."""
    sampled = (
        "<think>\nrun the tests first\n</think>\n\n<tool_call>\n"
        '{"name": "bash", "arguments": {"cmd":  "pytest"}}\n</tool_call>' + ending
    )
    parent = _render(
        tokenizer,
        QUALIFICATION_HISTORY[:2],
        add_generation_prompt=True,
        enable_thinking=thinking,
    ) + tokenizer.encode(sampled, add_special_tokens=False)
    rendered = _render(
        tokenizer,
        QUALIFICATION_HISTORY + observations,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    bridged = parent + bridge_suffix(
        parent_token_ids=parent,
        rendered_token_ids=rendered,
        turn_close_token_ids=closes,
    )

    spliced = splice_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=parent,
        template_prefix_token_ids=_render(
            tokenizer,
            QUALIFICATION_HISTORY,
            add_generation_prompt=False,
            enable_thinking=thinking,
        ),
        template_token_ids=rendered,
        eos_token_id=closes,
    ).token_ids
    assert bridged == spliced
    assert bridged != rendered


@pytest.mark.parametrize("template_kwargs", [{}, {"enable_thinking": False}])
def test_qualification_accepts_the_qwen3_template(tokenizer, closes, template_kwargs):
    qualify_bridge(
        tokenizer=tokenizer,
        chat_template=None,
        template_kwargs=template_kwargs,
        turn_close_token_ids=closes,
    )


_NO_CLOSE_TEMPLATE = (
    "{% for message in messages %}{{ message.role }}: {{ message.content }}\n"
    "{% endfor %}"
)
_HISTORY_AFTER_CLOSE_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{{ message.content }}<|im_end|>\n{% endfor %}"
    "{{ messages[0].content }}<|im_start|>assistant\n"
)
_DROPS_OBSERVATIONS_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{% if message.role != 'tool' %}{{ message.content }}{% endif %}<|im_end|>\n"
    "{% endfor %}<|im_start|>assistant\n"
)


@pytest.mark.parametrize(
    "chat_template,match",
    [
        (_NO_CLOSE_TEMPLATE, "does not close an assistant turn"),
        (_HISTORY_AFTER_CLOSE_TEMPLATE, "repeat the history"),
        (_DROPS_OBSERVATIONS_TEMPLATE, "do not hold the appended messages"),
    ],
)
def test_qualification_rejects_a_template_the_bridge_cannot_continue(
    tokenizer, closes, chat_template, match
):
    with pytest.raises(ValueError, match=match):
        qualify_bridge(
            tokenizer=tokenizer,
            chat_template=chat_template,
            template_kwargs={},
            turn_close_token_ids=closes,
        )
