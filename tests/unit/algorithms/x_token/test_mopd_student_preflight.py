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

from typing import Any

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms.x_token import mopd_teacher_scoring as scoring

_CHATML_MESSAGES = """
{%- for message in messages -%}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' -}}
{%- endfor -%}
"""
_CHATML_TEMPLATE = (
    _CHATML_MESSAGES
    + "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
_CUSTOM_TEMPLATE = """
{%- for message in messages -%}
    {{- '<S ' + message['role'] + '>\n' + message['content'] + '<E>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ '<S assistant>\n' }}{%- endif -%}
"""
_MERGED_TOOLS_TEMPLATE = """
{%- for message in messages -%}
    {%- if message['role'] == 'tool' -%}
        {%- if loop.first or messages[loop.index0 - 1]['role'] != 'tool' -%}
            {{- '<|im_start|>user\n' -}}
        {%- endif -%}
        {{- '<tool_response>\n' + message['content'] + '\n</tool_response>\n' -}}
        {%- if loop.last or messages[loop.index0 + 1]['role'] != 'tool' -%}
            {{- '<|im_end|>\n' -}}
        {%- endif -%}
    {%- else -%}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\n' }}{%- endif -%}
"""


def _tokenizer(template: str = _CHATML_TEMPLATE) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        models.BPE(
            vocab={
                char: index
                for index, char in enumerate(
                    sorted(pre_tokenizers.ByteLevel.alphabet())
                )
            },
            merges=[],
        )
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        chat_template=template,
    )


def _student_config(**overrides: Any) -> dict[str, Any]:
    return {
        "name": "local-student",
        "chat_template": "default",
        "chat_template_kwargs": {},
        "tokenizer_kwargs": {},
        **overrides,
    }


def _use_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tokenizer: PreTrainedTokenizerFast
) -> None:
    monkeypatch.setattr(
        scoring, "_load_plain_auto_tokenizer", lambda name, kwargs: tokenizer
    )


def test_non_thinking_chatml_passes_student_format_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_tokenizer(monkeypatch, _tokenizer())
    assert scoring.validate_mopd_student_alignment_preflight(_student_config()) is False


def test_student_preflight_accepts_merged_tool_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _tokenizer(_MERGED_TOOLS_TEMPLATE)
    _use_tokenizer(monkeypatch, tokenizer)
    tools = [
        {"role": "tool", "content": "first tool result"},
        {"role": "tool", "content": "second tool result"},
    ]
    rendered = tokenizer.apply_chat_template(tools, tokenize=False)
    parsed = scoring.parse_qwen_chat_token_stream(
        tokenizer, tokenizer(rendered, add_special_tokens=False)["input_ids"]
    )
    assert parsed is not None
    assert len(parsed[0]) == 1
    assert parsed[0][0]["role"] == "user"
    assert all(message["content"] in parsed[0][0]["content"] for message in tools)
    assert scoring.validate_mopd_student_alignment_preflight(_student_config()) is False


@pytest.mark.parametrize("template", [_CUSTOM_TEMPLATE, _CHATML_MESSAGES])
def test_student_preflight_rejects_unsupported_format_or_missing_generation_header(
    monkeypatch: pytest.MonkeyPatch, template: str
) -> None:
    tokenizer = _tokenizer(template)
    _use_tokenizer(monkeypatch, tokenizer)
    assert scoring.validate_mopd_student_thinking_preflight(tokenizer) is False
    with pytest.raises(ValueError, match="conversation format|open assistant"):
        scoring.validate_mopd_student_alignment_preflight(_student_config())


@pytest.mark.parametrize("suffix", ["", "\n"])
def test_student_preflight_rejects_already_ended_empty_assistant(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    template = (
        _CHATML_MESSAGES
        + "{% if add_generation_prompt %}"
        + "<|im_start|>assistant\n<|im_end|>"
        + suffix
        + "{% endif %}"
    )
    _use_tokenizer(monkeypatch, _tokenizer(template))
    with pytest.raises(ValueError, match="open assistant message boundary"):
        scoring.validate_mopd_student_alignment_preflight(_student_config())


@pytest.mark.parametrize("enabled", [True, False])
def test_student_preflight_honors_configured_template_and_kwargs(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    _use_tokenizer(monkeypatch, _tokenizer(_CUSTOM_TEMPLATE))
    template = (
        "{% if use_chatml %}"
        + _CHATML_TEMPLATE
        + "{% else %}"
        + _CUSTOM_TEMPLATE
        + "{% endif %}"
    )
    config = _student_config(
        chat_template=template, chat_template_kwargs={"use_chatml": enabled}
    )
    if enabled:
        assert scoring.validate_mopd_student_alignment_preflight(config) is False
    else:
        with pytest.raises(ValueError, match="unsupported conversation format"):
            scoring.validate_mopd_student_alignment_preflight(config)


@pytest.mark.parametrize(
    "template",
    [
        _CHATML_TEMPLATE.replace("message['content']", "'missing content'"),
        _CHATML_TEMPLATE.replace("message in messages", "message in messages|reverse"),
        _CHATML_TEMPLATE.replace("<|im_end|>", ""),
    ],
    ids=["dropped-content", "reordered-content", "missing-message-end"],
)
def test_student_preflight_rejects_lost_probes_or_malformed_stream(
    monkeypatch: pytest.MonkeyPatch, template: str
) -> None:
    _use_tokenizer(monkeypatch, _tokenizer(template))
    with pytest.raises(ValueError, match="preserve.*probes|unsupported conversation"):
        scoring.validate_mopd_student_alignment_preflight(_student_config())


def test_student_probe_search_advances_past_each_match() -> None:
    tokenizer = _tokenizer(
        "{{ '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n' }}"
    )
    with pytest.raises(ValueError, match="preserve.*probes"):
        scoring._validate_student_chat_format(
            tokenizer,
            [
                {"role": "user", "content": "alpha beta"},
                {"role": "assistant", "content": "beta"},
            ],
            add_generation_prompt=False,
        )


@pytest.mark.parametrize("invalid_offsets", [False, True])
def test_student_preflight_retains_offset_checks(
    monkeypatch: pytest.MonkeyPatch, invalid_offsets: bool
) -> None:
    tokenizer = _tokenizer()
    _use_tokenizer(monkeypatch, tokenizer)
    original_call = PreTrainedTokenizerFast.__call__

    def broken_offsets(self: Any, text: str, **kwargs: Any) -> Any:
        encoding = original_call(self, text, **kwargs)
        if kwargs.get("return_offsets_mapping"):
            if invalid_offsets:
                encoding["offset_mapping"][0] = (-1, 1)
            else:
                encoding["offset_mapping"] = encoding["offset_mapping"][:-1]
        return encoding

    monkeypatch.setattr(PreTrainedTokenizerFast, "__call__", broken_offsets)
    with pytest.raises(ValueError, match="offset"):
        scoring.validate_mopd_student_alignment_preflight(_student_config())


def test_teacher_preflight_does_not_require_chatml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_tokenizer(monkeypatch, _tokenizer(_CUSTOM_TEMPLATE))
    scoring.validate_cross_tokenizer_preflight(
        {
            "tokenizer": _student_config(name="local-teacher"),
            "alignment_method": "offset_cluster_decode_fix",
            "mask_first_teacher_prefix_chunk": False,
            "exclude_proven_template_only_teacher_tokens": False,
            "missing_think_close_policy": "mask",
        }
    )
