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

"""Thinking classification is independent from student transcript compatibility."""

import os
from copy import deepcopy
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token import mopd_teacher_scoring as scoring

_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' "
    "+ message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
_NANO_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}{{ '<think>\n\n</think>\n\n' }}{% endif %}"
    "{{ message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"
)
_CUSTOM_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<S ' + message['role'] + '>\n' + message['content'] + '<E>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<S assistant>\n' }}{% endif %}"
)
_R1_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<｜' + message['role'] + '｜>' }}"
    "{% if message['role'] == 'assistant' %}{{ '<think></think>' }}{% endif %}"
    "{{ message['content'] }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<｜Assistant｜><think>\n' }}{% endif %}"
)


@pytest.fixture(scope="module")
def native_qwen3_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        os.environ.get("NRL_TEST_QWEN3_TOKENIZER", "Qwen/Qwen3-0.6B"),
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
    )


def _student_config(
    monkeypatch: pytest.MonkeyPatch,
    native_tokenizer: PreTrainedTokenizerBase,
    *,
    template: str = "default",
    template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    monkeypatch.setattr(
        scoring,
        "_load_plain_auto_tokenizer",
        lambda *_args, **_kwargs: deepcopy(native_tokenizer),
    )
    return {
        "name": "student-tokenizer",
        "chat_template": template,
        "chat_template_kwargs": template_kwargs or {},
    }


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_qwen3_thinking_preflight_honors_configured_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    enable_thinking: bool,
) -> None:
    config = _student_config(
        monkeypatch,
        native_qwen3_tokenizer,
        template_kwargs={"enable_thinking": enable_thinking},
    )
    tokenizer = scoring.load_mopd_student_alignment_tokenizer(config)

    assert (
        scoring.validate_mopd_student_thinking_preflight(tokenizer) is enable_thinking
    )
    assert scoring.validate_mopd_student_alignment_preflight(config) is enable_thinking


def test_nano_open_thinking_prefix_samples_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
) -> None:
    config = _student_config(
        monkeypatch, native_qwen3_tokenizer, template=_NANO_TEMPLATE
    )
    tokenizer = scoring.load_mopd_student_alignment_tokenizer(config)

    assert scoring.validate_mopd_student_thinking_preflight(tokenizer) is True
    assert scoring.validate_mopd_student_alignment_preflight(config) is True


def test_nonthinking_chatml_header_does_not_report_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
) -> None:
    config = _student_config(
        monkeypatch, native_qwen3_tokenizer, template=_CHATML_TEMPLATE
    )
    tokenizer = scoring.load_mopd_student_alignment_tokenizer(config)

    assert scoring.validate_mopd_student_thinking_preflight(tokenizer) is False
    assert scoring.validate_mopd_student_alignment_preflight(config) is False


def test_nonthinking_custom_format_still_fails_independent_format_preflight(
    monkeypatch: pytest.MonkeyPatch,
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
) -> None:
    config = _student_config(
        monkeypatch, native_qwen3_tokenizer, template=_CUSTOM_TEMPLATE
    )
    tokenizer = scoring.load_mopd_student_alignment_tokenizer(config)

    assert scoring.validate_mopd_student_thinking_preflight(tokenizer) is False
    with pytest.raises(ValueError, match="ChatML"):
        scoring.validate_mopd_student_alignment_preflight(config)


@pytest.mark.parametrize("template", ["<think>", "</think>", "plain"])
def test_thinking_preflight_requires_both_template_markers(
    monkeypatch: pytest.MonkeyPatch,
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    template: str,
) -> None:
    config = _student_config(monkeypatch, native_qwen3_tokenizer, template=template)
    tokenizer = scoring.load_mopd_student_alignment_tokenizer(config)

    assert scoring.validate_mopd_student_thinking_preflight(tokenizer) is False


def test_r1_distill_style_thinking_boundary_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
) -> None:
    config = _student_config(monkeypatch, native_qwen3_tokenizer, template=_R1_TEMPLATE)
    tokenizer = scoring.load_mopd_student_alignment_tokenizer(config)

    with pytest.raises(ValueError, match="thinking generation boundary is unsupported"):
        scoring.validate_mopd_student_thinking_preflight(tokenizer)
