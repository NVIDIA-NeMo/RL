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

import copy
from functools import partial
from typing import Any
from unittest.mock import patch

import pytest
import torch
from jinja2 import Template, TemplateError
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import render_jinja_template

from nemo_rl.data.full_conversation import (
    _has_generation_blocks,
    _render_with_message_spans,
    get_full_conversation_message_log,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
    get_formatted_message_log,
)
from nemo_rl.data.processors import PROCESSOR_REGISTRY

TEMPLATE = """{%- if tools %}{{ tools | tojson }}{% endif -%}
{%- for m in messages -%}
{%- if m.role == 'tool' -%}
{%- if loop.previtem.role != 'tool' %}{{ '<start>tool\\n' }}{% endif -%}
{{ m.content }}
{%- if loop.last or loop.nextitem.role != 'tool' %}{{ '<stop>\\n' }}{% endif -%}
{%- elif m.role == 'assistant' -%}
{{ '<start>assistant\\n' }}
{%- generation -%}
{%- if enable_thinking and m.reasoning_content %}{{ m.reasoning_content }}{% endif -%}
{{ m.content }}
{%- for call in m.tool_calls | default([]) %}{{ call | tojson }}{% endfor -%}
{{ '<stop>\\n' }}
{%- endgeneration -%}
{%- else -%}
{{ '<start>' + m.role + '\\n' + m.content + '<stop>\\n' }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}{{ '<start>assistant\\n' }}{% endif -%}"""

NEMOTRON_REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"


@pytest.mark.parametrize(
    "template,expected",
    [
        ("plain text", False),
        ("{# {% generation %} #}", False),
        ("{{ '{% generation %}' }}", False),
        ("{% raw %}{% generation %}{% endraw %}", False),
        ("{% generation %}answer{% endgeneration %}", True),
        ("{%- generation -%}answer{%- endgeneration -%}", True),
        ("{%\n generation\n%}answer{% endgeneration %}", True),
        (
            "{% if enabled %}{% generation %}answer{% endgeneration %}{% endif %}",
            True,
        ),
    ],
)
def test_generation_detection_uses_jinja_syntax(template: str, expected: bool) -> None:
    assert _has_generation_blocks(template) is expected


@pytest.mark.parametrize(
    "prefix",
    [
        "{# {% generation %} #}",
        "{{ '{% generation %}' }}",
        "{% raw %}{% generation %}{% endraw %}",
    ],
)
def test_literal_generation_tags_do_not_disable_automatic_tracking(
    tokenizer: PreTrainedTokenizerFast, prefix: str
) -> None:
    tokenizer.chat_template = prefix + TEMPLATE.replace(
        "{%- generation -%}", ""
    ).replace("{%- endgeneration -%}", "")
    log = format_messages(tokenizer)
    assert tokenizer.decode(torch.cat([m["token_ids"] for m in log])) == (
        tokenizer.apply_chat_template(messages(), tokenize=False)
    )
    assistant = [m for m in log if m["role"] == "assistant"]
    assert len(assistant) == 2
    assert all(m["content"].startswith("<start>assistant\n") for m in assistant)


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["hello hello hello hello world\n"] * 10,
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<bos>", "<start>", "<stop>", "<pad>"],
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<stop>",
        pad_token="<pad>",
        additional_special_tokens=["<start>"],
        chat_template=TEMPLATE,
    )


def messages(results: int = 2) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Check values"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "REASONING",
            "tool_calls": [
                {"name": "check", "arguments": {"n": i, "ok": True, "v": [None, False]}}
                for i in range(results)
            ],
        },
        *[{"role": "tool", "content": f"RESULT_{i}"} for i in reversed(range(results))],
        {"role": "assistant", "content": "All checked"},
    ]


def format_messages(
    tokenizer: PreTrainedTokenizerFast,
    source: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    options = {
        "prompt": None,
        "tools": None,
        "add_bos_token": False,
        "add_eos_token": True,
    }
    options.update(kwargs)
    return get_full_conversation_message_log(
        messages() if source is None else source, tokenizer, **options
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "defaults",
    [
        {},
        {"tools": None, "documents": None},
        {"tools": [], "documents": []},
        {"tools": [{"name": "configured"}]},
        {"documents": [{"title": "Reference", "text": "Document text"}]},
        {
            "tools": [{"name": "configured"}],
            "documents": [{"title": "Reference", "text": "Document text"}],
        },
    ],
)
@pytest.mark.parametrize("row_tools", [None, [], [{"name": "row"}]])
def test_standard_render_context_parity(
    tokenizer: PreTrainedTokenizerFast,
    explicit: bool,
    defaults: dict[str, Any],
    row_tools: list[dict[str, Any]] | None,
) -> None:
    target = "{{ m.content + '<stop>' }}"
    if explicit:
        target = "{% generation %}" + target + "{% endgeneration %}"
    tokenizer.chat_template = (
        "{% if tools is defined %}TOOLS_DEFINED:{% endif %}"
        "{% if tools is none %}TOOLS_NONE:{% endif %}"
        "{% if tools %}{{ tools | tojson }}{% endif %}"
        "{% if documents is defined %}DOCUMENTS_DEFINED:{% endif %}"
        "{% if documents is none %}DOCUMENTS_NONE:{% endif %}"
        "{% if documents %}{{ documents | tojson }}{% endif %}"
        "{% for m in messages %}{{ '<start>' + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}"
        + target
        + "{% else %}{{ m.content + '<stop>' }}{% endif %}{% endfor %}"
    )
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, **defaults)
    source = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    reference_options = {} if row_tools is None else {"tools": row_tools}
    reference = tokenizer.apply_chat_template(
        source, tokenize=False, add_generation_prompt=False, **reference_options
    )
    log = format_messages(tokenizer, source, tools=row_tools, add_eos_token=False)
    assert (
        torch.cat([m["token_ids"] for m in log]).tolist()
        == tokenizer(reference, add_special_tokens=False)["input_ids"]
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("content_state", ["missing", "null", "empty"])
def test_tool_call_content_state_parity(
    tokenizer: PreTrainedTokenizerFast, explicit: bool, content_state: str
) -> None:
    target = (
        "{% if m.content is not defined %}MISSING"
        "{% elif m.content is none %}NULL{% else %}EMPTY{% endif %}"
        "{{ m.tool_calls | tojson }}<stop>"
    )
    if explicit:
        target = "{% generation %}" + target + "{% endgeneration %}"
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<start>' + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}"
        + target
        + "{% else %}{{ m.content + '<stop>' }}{% endif %}{% endfor %}"
    )
    assistant: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [{"name": "check"}],
    }
    if content_state != "missing":
        assistant["content"] = None if content_state == "null" else ""
    source = [{"role": "user", "content": "question"}, assistant]
    original = copy.deepcopy(source)
    reference = tokenizer.apply_chat_template(
        source, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        reference, add_special_tokens=False, return_offsets_mapping=True
    )
    log = format_messages(tokenizer, source, add_eos_token=False)
    assert torch.cat([m["token_ids"] for m in log]).tolist() == encoded["input_ids"]
    assert source == original
    target_start = reference.index(
        content_state.upper() if explicit else "<start>assistant\n"
    )
    expected_mask = torch.tensor(
        [int(start >= target_start) for start, _ in encoded["offset_mapping"]]
    )
    add_loss_mask_to_message_log([log])
    assert torch.equal(torch.cat([m["token_loss_mask"] for m in log]), expected_mask)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("missing_content", [False, True])
def test_tool_call_template_errors_remain_visible(
    tokenizer: PreTrainedTokenizerFast, explicit: bool, missing_content: bool
) -> None:
    target = (
        "{% if m.content is not string %}"
        "{{ raise_exception('This template requires text content') }}{% endif %}"
        "{{ m.content }}<stop>"
    )
    if explicit:
        target = "{% generation %}" + target + "{% endgeneration %}"
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m.role == 'assistant' %}"
        + target
        + "{% else %}{{ m.content }}{% endif %}{% endfor %}"
    )
    assistant = {"role": "assistant", "tool_calls": [{"name": "check"}]}
    if not missing_content:
        assistant["content"] = None
    source = [{"role": "user", "content": "question"}, assistant]
    original = copy.deepcopy(source)
    with pytest.raises(TemplateError, match="requires text content"):
        tokenizer.apply_chat_template(source, tokenize=False)
    with pytest.raises(TemplateError, match="requires text content"):
        format_messages(tokenizer, source)
    assert source == original


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "options,error,match",
    [
        ({"documents": ["not a document"]}, TypeError, "Documents"),
        ({"tools": ["not a schema"]}, ValueError, "tool schemas"),
        ({"tools": [len]}, ValueError, "tool schemas"),
    ],
)
def test_invalid_render_inputs_fail_in_both_paths(
    tokenizer: PreTrainedTokenizerFast,
    explicit: bool,
    options: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    if not explicit:
        tokenizer.chat_template = TEMPLATE.replace("{%- generation -%}", "").replace(
            "{%- endgeneration -%}", ""
        )
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, **options)
    with pytest.raises(error, match=match):
        format_messages(tokenizer)


@pytest.mark.parametrize(
    "assistant",
    [
        {"role": "assistant"},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": [], "tool_calls": [{"name": "check"}]},
    ],
)
def test_nontext_content_is_not_silently_accepted(
    tokenizer: PreTrainedTokenizerFast, assistant: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="text-only"):
        format_messages(tokenizer, [{"role": "user", "content": "q"}, assistant])


@pytest.mark.parametrize("results", [1, 2, 3])
@pytest.mark.parametrize("thinking", [True, False])
def test_full_render_parity_and_loss_ownership(
    tokenizer: PreTrainedTokenizerFast, results: int, thinking: bool
) -> None:
    source = messages(results)
    original = copy.deepcopy(source)
    tools = [{"name": "check", "parameters": {"n": {"type": "integer"}}}]
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, enable_thinking=thinking
    )
    log = format_messages(tokenizer, source, tools=tools)
    reference = tokenizer.apply_chat_template(source, tools=tools, tokenize=False)
    ids = torch.cat([segment["token_ids"] for segment in log])
    assert tokenizer.decode(ids) == reference
    assert ids.tolist() == tokenizer(reference, add_special_tokens=False)["input_ids"]
    assert reference.count("<start>tool") == 1
    target = "".join(
        segment["content"] for segment in log if segment["role"] == "assistant"
    )
    assert target.count('"name": "check"') == results
    assert '"ok": true' in target and '"v": [null, false]' in target
    assert "RESULT_" not in target and "<start>" not in target
    assert ("REASONING" in target) == thinking
    assert target.count("<stop>") == 2
    assert source == original
    assert log[-1]["role"] == "assistant"


def test_one_render_and_one_tokenization(tokenizer: PreTrainedTokenizerFast) -> None:
    original_encode = type(tokenizer).__call__
    with (
        patch(
            "nemo_rl.data.full_conversation.render_jinja_template",
            wraps=render_jinja_template,
        ) as render,
        patch.object(
            type(tokenizer), "__call__", autospec=True, side_effect=original_encode
        ) as encode,
    ):
        format_messages(tokenizer)
    assert render.call_count == 1
    assert encode.call_count == 1


def test_bound_template_override(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.chat_template = "default without generation markers"
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, chat_template="CUSTOM:" + TEMPLATE
    )
    log = format_messages(tokenizer)
    actual = tokenizer.decode(torch.cat([m["token_ids"] for m in log]))
    assert actual == tokenizer.apply_chat_template(messages(), tokenize=False)
    assert actual.startswith("CUSTOM:")


@pytest.mark.parametrize("row_tools", [None, [], [{"name": "row_tool"}]])
def test_bound_tools_and_row_precedence(
    tokenizer: PreTrainedTokenizerFast, row_tools: list[dict[str, Any]] | None
) -> None:
    default_tools = [{"name": "default_tool"}]
    tokenizer.chat_template = {"default": "unmarked", "tool_use": TEMPLATE}
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, tools=default_tools
    )
    # An explicit empty list selects the default template, just as Transformers does.
    if row_tools == []:
        tokenizer.chat_template["default"] = "NO_TOOLS:" + TEMPLATE
    log = format_messages(tokenizer, tools=row_tools)
    options = {} if row_tools is None else {"tools": row_tools}
    expected = tokenizer.apply_chat_template(messages(), tokenize=False, **options)
    assert tokenizer.decode(torch.cat([m["token_ids"] for m in log])) == expected


@pytest.mark.parametrize("whitespace", ["", "\n"])
def test_eos_in_context_respects_explicit_spans(
    tokenizer: PreTrainedTokenizerFast, whitespace: str
) -> None:
    tokenizer.chat_template = (
        "prefix{% for m in messages %}{% if m.role == 'assistant' %}"
        "{% generation %}{{ m.content }}"
        "{% if loop.last %}{{ '<stop>' }}{% endif %}{% endgeneration %}"
        "{% if not loop.last %}{{ terminator_prefix + '<stop>' }}{% endif %}"
        "{% else %}{{ m.content }}{% endif %}{% endfor %}"
    )
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, terminator_prefix=whitespace
    )
    source = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "last"},
    ]
    reference = tokenizer.apply_chat_template(source, tokenize=False)
    encoded = tokenizer(
        reference, add_special_tokens=False, return_offsets_mapping=True
    )
    log = format_messages(tokenizer, source, add_eos_token=False)
    ids = torch.cat([m["token_ids"] for m in log])
    assert ids.tolist() == encoded["input_ids"]
    ranges = [
        (reference.index(text), reference.index(text) + len(text))
        for text in ("answer", "last<stop>")
    ]
    expected_mask = [
        int(any(start <= token_start < end for start, end in ranges))
        for token_start, _ in encoded["offset_mapping"]
    ]
    add_loss_mask_to_message_log([log])
    mask = torch.cat([m["token_loss_mask"] for m in log])
    assert mask.tolist() == expected_mask
    assert mask[ids == tokenizer.eos_token_id].tolist() == [0, 1]


@pytest.mark.parametrize(
    "trailing_whitespace,add_eos", [("\n", True), (" \n\t", False)]
)
@pytest.mark.parametrize("only_final", [False, True])
def test_shared_bos_eos_turn_headers(
    tokenizer: PreTrainedTokenizerFast,
    trailing_whitespace: str,
    add_eos: bool,
    only_final: bool,
) -> None:
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for m in messages %}{{ bos_token + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}{% generation %}"
        "{{ m.content + eos_token + target_whitespace }}{% endgeneration %}"
        "{% else %}{{ m.content + '\\n' }}{% endif %}{% endfor %}"
    )
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, target_whitespace=trailing_whitespace
    )
    source = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "last"},
    ]
    log = format_messages(tokenizer, source, add_bos_token=True, add_eos_token=add_eos)
    ids = torch.cat([segment["token_ids"] for segment in log])
    assert tokenizer.decode(ids) == tokenizer.apply_chat_template(
        source, tokenize=False
    )
    assistant_segments = [segment for segment in log if segment["role"] == "assistant"]
    assert [segment["content"] for segment in assistant_segments] == [
        answer + tokenizer.eos_token + trailing_whitespace
        for answer in ("answer", "last")
    ]
    add_loss_mask_to_message_log([log], only_unmask_final=only_final)
    for index, segment in enumerate(assistant_segments):
        is_eos = segment["token_ids"] == tokenizer.eos_token_id
        assert is_eos.sum() == 1
        should_train = not only_final or index == len(assistant_segments) - 1
        assert segment["token_loss_mask"][is_eos].item() == int(should_train)
    for segment in log:
        if segment["role"] == "context":
            assert not segment["token_loss_mask"].any()


@pytest.mark.parametrize(
    "content", ["", "caf\u00e9 \u4f60\u597d \U0001f600", "  answer\n\t "]
)
def test_text_and_unicode(tokenizer: PreTrainedTokenizerFast, content: str) -> None:
    log = format_messages(
        tokenizer,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": content},
        ],
    )
    assert log[-1]["content"] == content + "<stop>\n"


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("add_eos", [False, True])
@pytest.mark.parametrize("only_final", [False, True])
@pytest.mark.parametrize("shared_bos_eos", [False, True])
def test_eos_turn_headers_with_distinct_turn_terminator(
    tokenizer: PreTrainedTokenizerFast,
    explicit: bool,
    add_eos: bool,
    only_final: bool,
    shared_bos_eos: bool,
) -> None:
    if shared_bos_eos:
        tokenizer.bos_token = tokenizer.eos_token
    else:
        assert tokenizer.bos_token != tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<start>", "<turn_end>"]}
    )
    target = "{{ m.content + '<turn_end>\\n' }}"
    if explicit:
        target = "{% generation %}" + target + "{% endgeneration %}"
    tokenizer.chat_template = (
        "{% for m in messages %}{{ eos_token + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}"
        + target
        + "{% else %}{{ m.content + '<turn_end>\\n' }}{% endif %}{% endfor %}"
    )
    source = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "last"},
    ]
    reference = tokenizer.apply_chat_template(
        source, tokenize=False, add_generation_prompt=False
    )
    if add_eos:
        reference += tokenizer.eos_token
    encoded = tokenizer(
        reference, add_special_tokens=False, return_offsets_mapping=True
    )
    log = format_messages(tokenizer, source, add_bos_token=False, add_eos_token=add_eos)
    assert torch.cat([m["token_ids"] for m in log]).tolist() == encoded["input_ids"]
    targets = [answer + "<turn_end>\n" for answer in ("answer", "last")]
    if not explicit:
        targets = [tokenizer.eos_token + "assistant\n" + text for text in targets]
    if add_eos:
        targets[-1] += tokenizer.eos_token
    assert [m["content"] for m in log if m["role"] == "assistant"] == targets
    if only_final:
        targets = targets[-1:]
    ranges = [
        (reference.index(text), reference.index(text) + len(text)) for text in targets
    ]
    expected_mask = [
        int(any(start <= token_start < end for start, end in ranges))
        for token_start, _ in encoded["offset_mapping"]
    ]
    add_loss_mask_to_message_log([log], only_unmask_final=only_final)
    assert torch.cat([m["token_loss_mask"] for m in log]).tolist() == expected_mask


def test_prompt_and_bos(tokenizer: PreTrainedTokenizerFast) -> None:
    log = format_messages(tokenizer, prompt="TASK: {}", add_bos_token=True)
    assert log[0]["content"].startswith("<bos><start>user\nTASK: Check values")
    assert sum(segment["content"].count("<bos>") for segment in log) == 1


@pytest.mark.parametrize(
    "source,match",
    [
        ([], "end with"),
        ([{"role": "user", "content": "x"}], "end with"),
        ([{"role": "assistant", "content": "x"}], "user"),
        (
            [{"role": "user", "content": []}, {"role": "assistant", "content": "x"}],
            "text-only",
        ),
    ],
)
def test_invalid_input(
    tokenizer: PreTrainedTokenizerFast, source: list[dict[str, Any]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        format_messages(tokenizer, source)


@pytest.mark.parametrize(
    "template,match",
    [
        ("plain text", "generation"),
        ("prefix{% generation %}one<stop>{% endgeneration %}", "one generation"),
        (
            "prefix{% for m in messages if m.role == 'assistant' %}{% generation %}x<stop>{% endgeneration %}suffix{% endfor %}",
            "Non-whitespace",
        ),
    ],
)
def test_invalid_generation_templates(
    tokenizer: PreTrainedTokenizerFast, template: str, match: str
) -> None:
    tokenizer.chat_template = template
    with pytest.raises(ValueError, match=match):
        format_messages(tokenizer)


def test_boundary_crossing_uses_token_start(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.chat_template = "hel{% generation %}lo<stop>{% endgeneration %}"
    log = format_messages(
        tokenizer,
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "x"}],
    )
    assert log[0]["role"] == "context"
    assert log[0]["content"] == "hello"
    assert log[-1]["content"] == "<stop>"
    add_loss_mask_to_message_log([log])
    assert torch.cat([m["token_loss_mask"] for m in log]).tolist() == [0, 1]


def test_token_starting_in_assistant_can_cross_into_context(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokenizer.chat_template = (
        "x{% generation %}hel{% endgeneration %}lo"
        "{% generation %}<stop>{% endgeneration %}"
    )
    source = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "last"},
    ]
    log = format_messages(tokenizer, source)
    assert (
        torch.cat([m["token_ids"] for m in log]).tolist()
        == tokenizer("xhello<stop>", add_special_tokens=False)["input_ids"]
    )
    add_loss_mask_to_message_log([log])
    assert torch.cat([m["token_loss_mask"] for m in log]).tolist() == [0, 1, 1]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("only_final", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_reference_masks_and_normalization(
    tokenizer: PreTrainedTokenizerFast,
    explicit: bool,
    only_final: bool,
    normalize: bool,
) -> None:
    if normalize:
        tokenizer.backend_tokenizer.normalizer = normalizers.NFC()
    target = "{{ m.content + '<stop>' }}"
    if explicit:
        target = "{% generation %}" + target + "{% endgeneration %}"
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<start>' + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}"
        + target
        + "{% else %}{{ m.content + '<stop>' }}{% endif %}{% endfor %}"
    )
    source = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "CALL"},
        {"role": "tool", "content": "TOOL_RESULT"},
        {"role": "assistant", "content": "FINAL e\u0301"},
    ]
    reference = tokenizer.apply_chat_template(source, tokenize=False)
    encoded = tokenizer(
        reference, add_special_tokens=False, return_offsets_mapping=True
    )
    reference_ids = torch.tensor(encoded["input_ids"])
    if normalize:
        assert tokenizer.decode(reference_ids) != reference
    targets = ["CALL<stop>", "FINAL e\u0301<stop>"]
    if not explicit:
        targets = ["<start>assistant\n" + text for text in targets]
    if only_final:
        targets = targets[-1:]
    expected_ranges = [
        (reference.index(text), reference.index(text) + len(text)) for text in targets
    ]
    expected_mask = torch.tensor(
        [
            int(any(start <= token_start < end for start, end in expected_ranges))
            for token_start, _ in encoded["offset_mapping"]
        ]
    )
    log = format_messages(tokenizer, source, add_eos_token=False)
    assert torch.equal(torch.cat([m["token_ids"] for m in log]), reference_ids)
    add_loss_mask_to_message_log([log], only_unmask_final=only_final)
    flat, lengths = batched_message_log_to_flat_message(
        [log],
        {"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=8,
    )
    assert lengths.tolist() == [len(reference_ids)]
    assert torch.equal(flat["token_loss_mask"][0, : len(reference_ids)], expected_mask)
    assert not flat["token_loss_mask"][0, len(reference_ids) :].any()
    assert torch.equal(
        flat["token_ids"][0, 1:][flat["token_loss_mask"][0, 1:].bool()],
        reference_ids[1:][expected_mask[1:].bool()],
    )


def test_zero_next_token_targets_are_rejected(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokenizer.chat_template = "{% generation %}<stop>{% endgeneration %}"
    with pytest.raises(ValueError, match="no trainable next-token"):
        format_messages(
            tokenizer,
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "x"}],
        )


def test_missing_bos_eos(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.bos_token = None
    with pytest.raises(ValueError, match="no BOS"):
        format_messages(tokenizer, add_bos_token=True)
    tokenizer.eos_token = None
    with pytest.raises(ValueError, match="no EOS"):
        format_messages(tokenizer)


def test_eos_addition_stays_supervised(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.eos_token = "<bos>"
    log = format_messages(tokenizer)
    assert log[-1]["content"].endswith("<stop>\n<bos>")
    assert log[-1]["token_ids"][-1] == tokenizer.eos_token_id


@pytest.mark.parametrize("explicit", [False, True])
def test_adjacent_assistants_stay_separate(
    tokenizer: PreTrainedTokenizerFast, explicit: bool
) -> None:
    target = "{{ m.content + '<stop>' }}"
    if explicit:
        target = "{% generation %}" + target + "{% endgeneration %}"
    tokenizer.chat_template = (
        "prefix{% for m in messages if m.role == 'assistant' %}"
        + target
        + "{% endfor %}"
    )
    log = format_messages(
        tokenizer,
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "last"},
        ],
    )
    assert [m["content"] for m in log if m["role"] == "assistant"] == [
        "first<stop>",
        "last<stop>",
    ]
    add_loss_mask_to_message_log([log], only_unmask_final=True)
    assert not log[-2]["token_loss_mask"].any()
    assert log[-1]["token_loss_mask"].all()


def test_non_monotonic_tool_turn_regression(tokenizer: PreTrainedTokenizerFast) -> None:
    source = messages()
    incremental = get_formatted_message_log(
        source,
        tokenizer,
        TaskDataSpec(),
        add_bos_token=False,
        add_eos_token=True,
        add_generation_prompt=False,
    )
    one_shot = tokenizer.apply_chat_template(source, tokenize=False)
    assert (
        tokenizer.decode(torch.cat([m["token_ids"] for m in incremental])) != one_shot
    )
    full = format_messages(tokenizer, source)
    assert tokenizer.decode(torch.cat([m["token_ids"] for m in full])) == one_shot


@pytest.mark.parametrize("only_final", [False, True])
def test_actual_nemo_mask_padding_and_next_token_contract(
    tokenizer: PreTrainedTokenizerFast, only_final: bool
) -> None:
    source = [
        messages(),
        [{"role": "user", "content": "short"}, {"role": "assistant", "content": "yes"}],
    ]
    logs = [format_messages(tokenizer, row) for row in source]
    expected_targets = []
    for log in logs:
        assistant = [m["token_ids"] for m in log if m["role"] == "assistant"]
        expected_targets.append(torch.cat(assistant[-1:] if only_final else assistant))
    add_loss_mask_to_message_log(logs, only_unmask_final=only_final)
    flat, lengths = batched_message_log_to_flat_message(
        logs, {"token_ids": tokenizer.pad_token_id}, make_sequence_length_divisible_by=8
    )
    assert flat["token_ids"].shape[1] % 8 == 0
    for index, target in enumerate(expected_targets):
        ids = flat["token_ids"][index]
        mask = flat["token_loss_mask"][index]
        assert torch.equal(ids[1:][mask[1:].bool()], target)
        assert not mask[lengths[index] :].any()


@pytest.mark.parametrize("limit", [1, 4, 128, 16384])
def test_registered_processor_and_exclusion(
    tokenizer: PreTrainedTokenizerFast, limit: int, caplog: pytest.LogCaptureFixture
) -> None:
    processor = PROCESSOR_REGISTRY["sft_full_conversation_processor"]
    result = processor(
        {"messages": messages(), "task_name": "test"},
        TaskDataSpec(),
        tokenizer,
        limit,
        7,
        add_bos=False,
        add_eos=True,
        add_generation_prompt=False,
    )
    ids = torch.cat([m["token_ids"] for m in result["message_log"]])
    assert result["idx"] == 7
    if result["length"] >= limit:
        assert result["loss_multiplier"] == 0.0
        assert 0 < len(ids) <= limit
        assert "idx=7" in caplog.text
    else:
        assert result["loss_multiplier"] == 1.0
        assert tokenizer.decode(ids) == tokenizer.apply_chat_template(
            messages(), tokenize=False
        )


def test_registered_processor_rejects_generation_prompt(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    with pytest.raises(ValueError, match="add_generation_prompt"):
        PROCESSOR_REGISTRY["sft_full_conversation_processor"](
            {"messages": messages(), "task_name": "test"},
            TaskDataSpec(),
            tokenizer,
            16384,
            0,
            add_bos=False,
            add_eos=True,
            add_generation_prompt=True,
        )


def test_default_processor_is_unchanged(tokenizer: PreTrainedTokenizerFast) -> None:
    source = messages()
    result = PROCESSOR_REGISTRY["sft_processor"](
        {"messages": source, "task_name": "test"},
        TaskDataSpec(),
        tokenizer,
        16384,
        0,
        add_bos=False,
        add_eos=True,
        add_generation_prompt=False,
    )
    expected = get_formatted_message_log(
        source, tokenizer, TaskDataSpec(), add_bos_token=False, add_eos_token=True
    )
    assert torch.equal(
        torch.cat([m["token_ids"] for m in result["message_log"]]),
        torch.cat([m["token_ids"] for m in expected]),
    )


@pytest.mark.parametrize("suffix", ["", "\n", "\n\n", " ", "\t", " \t", "\n \t"])
@pytest.mark.parametrize("add_eos", [False, True])
def test_final_eos_whitespace_matches_incremental_convention(
    tokenizer: PreTrainedTokenizerFast, suffix: str, add_eos: bool
) -> None:
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<start>' + m.role + '\\n' }}"
        "{% if m.role == 'assistant' %}{% generation %}"
        "{{ m.content + eos_token + eos_suffix }}{% endgeneration %}"
        "{% else %}{{ m.content + '<stop>\\n' }}{% endif %}{% endfor %}"
    )
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, eos_suffix=suffix
    )
    source = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    log = format_messages(tokenizer, source, add_eos_token=add_eos)
    extra_eos = add_eos and bool(suffix.strip("\n"))
    expected_target = "answer<stop>" + suffix + ("<stop>" if extra_eos else "")
    assert log[-1]["content"] == expected_target
    incremental = get_formatted_message_log(
        source,
        tokenizer,
        TaskDataSpec(),
        add_bos_token=False,
        add_eos_token=add_eos,
    )
    ids = torch.cat([m["token_ids"] for m in log])
    assert tokenizer.decode(ids) == tokenizer.decode(
        torch.cat([m["token_ids"] for m in incremental])
    )
    add_loss_mask_to_message_log([log])
    stop = log[-1]["token_ids"] == tokenizer.eos_token_id
    assert stop.sum() == 1 + int(extra_eos)
    assert log[-1]["token_loss_mask"][stop].all()


def test_excluded_row_keeps_original_length_and_masked_placeholder(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    original = format_messages(tokenizer)
    original_ids = torch.cat([segment["token_ids"] for segment in original])
    original_length = len(original_ids)
    result = PROCESSOR_REGISTRY["sft_full_conversation_processor"](
        {"messages": messages(), "task_name": "test"},
        TaskDataSpec(),
        tokenizer,
        original_length,
        7,
        add_bos=False,
        add_eos=True,
        add_generation_prompt=False,
    )
    assert result["length"] == original_length
    assert result["loss_multiplier"] == 0.0
    log = result["message_log"]
    placeholder = torch.cat([segment["token_ids"] for segment in log])
    assert 0 < len(placeholder) <= 4 < original_length
    assert torch.equal(placeholder, original_ids[: len(placeholder)])
    add_loss_mask_to_message_log([log])
    flat, actual_lengths = batched_message_log_to_flat_message(
        [log], {"token_ids": tokenizer.pad_token_id}
    )
    assert actual_lengths.tolist() == [len(placeholder)]
    assert not flat["token_loss_mask"].any()


@pytest.mark.parametrize("missing_content", [False, True])
def test_null_tool_call_content(
    tokenizer: PreTrainedTokenizerFast, missing_content: bool
) -> None:
    source = messages()
    if missing_content:
        del source[1]["content"]
    else:
        source[1]["content"] = None
    original = copy.deepcopy(source)
    log = format_messages(tokenizer, source)
    assert source == original
    assert any('"name": "check"' in m["content"] for m in log)


@pytest.mark.parametrize("suffix", ["\n", " \n\t"])
def test_whitespace_after_final_generation_is_preserved(
    tokenizer: PreTrainedTokenizerFast, suffix: str
) -> None:
    tokenizer.chat_template = TEMPLATE + "{{ suffix }}"
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, suffix=suffix
    )
    log = format_messages(tokenizer, add_eos_token=False)
    assert tokenizer.decode(torch.cat([m["token_ids"] for m in log])) == (
        tokenizer.apply_chat_template(messages(), tokenize=False)
    )
    assert log[-1]["content"].endswith(suffix)


@pytest.mark.parametrize("only_final", [False, True])
def test_automatic_message_tracking(
    tokenizer: PreTrainedTokenizerFast, only_final: bool
) -> None:
    tokenizer.chat_template = TEMPLATE.replace("{%- generation -%}", "").replace(
        "{%- endgeneration -%}", ""
    )
    source = messages()
    expected = tokenizer.apply_chat_template(source, tokenize=False)
    original_encode = type(tokenizer).__call__
    with (
        patch.object(
            type(tokenizer), "__call__", autospec=True, side_effect=original_encode
        ) as encode,
        patch.object(
            Template, "generate", autospec=True, side_effect=Template.generate
        ) as render,
    ):
        log = format_messages(tokenizer)
    assert encode.call_count == 1
    assert render.call_count == 1
    assert (
        torch.cat([m["token_ids"] for m in log]).tolist()
        == tokenizer(expected, add_special_tokens=False)["input_ids"]
    )
    assistant = [m for m in log if m["role"] == "assistant"]
    assert len(assistant) == 2
    assert all(m["content"].startswith("<start>assistant\n") for m in assistant)
    assert sum(m["content"].count('"name": "check"') for m in assistant) == 2
    assert not any("RESULT_" in m["content"] for m in assistant)
    add_loss_mask_to_message_log([log], only_unmask_final=only_final)
    assert bool(assistant[0]["token_loss_mask"].any()) == (not only_final)
    assert assistant[-1]["token_loss_mask"].all()


@pytest.mark.parametrize(
    "body",
    [
        "{% set turns = messages[1:] %}{% for turn in turns %}{{ turn.content }}{% endfor %}",
        "{% for turn in messages if turn.role == 'assistant' %}{{ turn.content }}{% endfor %}",
        "{% macro body(turn) %}{{ turn.content }}{% endmacro %}{% for turn in messages %}{{ body(turn) }}{% endfor %}",
        "{% for turn in messages %}{% set text %}{{ turn.content }}{% endset %}{{ text }}{% endfor %}",
        "{% for turn in messages | reverse %}{% if turn.role == 'user' %}{% set ignored = turn.content %}{% endif %}{% endfor %}{% for turn in messages %}{{ turn.content }}{% endfor %}",
    ],
)
def test_message_tracking_preserves_jinja_output(
    tokenizer: PreTrainedTokenizerFast, body: str
) -> None:
    source = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": " answer\n"},
        {"role": "user", "content": "another"},
        {"role": "assistant", "content": " final "},
    ]
    expected = tokenizer.apply_chat_template(source, chat_template=body, tokenize=False)
    actual, spans = _render_with_message_spans(body, source, {})
    assert actual == expected
    assert [actual[start:end] for start, end in spans] == [" answer\n", " final "]


@pytest.mark.parametrize(
    "template",
    [
        "{% macro render() %}{% for m in messages %}{{ m.content }}{% endfor %}{% endmacro %}{{ render() }}",
        "{% set rendered %}{% for m in messages %}{{ m.content }}{% endfor %}{% endset %}{{ rendered }}",
        "{% filter trim %}{% for m in messages %}{{ m.content }}{% endfor %}{% endfilter %}",
        "{% for m in messages | reverse %}{{ m.content }}{% endfor %}",
        "{% for m in messages %}{% set copied = dict(m) %}{% for turn in [copied] %}{{ turn.content }}{% endfor %}{% endfor %}",
        "{% for m in messages %}{{ m.content }}{% if m.role == 'assistant' %}{% break %}{% endif %}{% endfor %}",
    ],
)
def test_ambiguous_automatic_ownership_is_rejected(
    tokenizer: PreTrainedTokenizerFast, template: str
) -> None:
    tokenizer.chat_template = template
    with pytest.raises(ValueError, match="generation"):
        format_messages(tokenizer)


@pytest.mark.parametrize(
    "model,revision,tool_count",
    [
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", NEMOTRON_REVISION, 2),
        ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 2),
        ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 2),
        (
            "unsloth/Llama-3.2-1B-Instruct",
            "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
            1,
        ),
        (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "c170c708c41dac9275d15a8fff4eca08d52bab71",
            2,
        ),
    ],
)
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize(
    "answer", ["Both checked.", "\nBoth checked. caf\u00e9 \u4f60\u597d \U0001f600\n"]
)
def test_stock_tokenizer_full_render_and_loss(
    model: str, revision: str, tool_count: int, with_tools: bool, answer: str
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check",
                "description": "Check a number",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
            },
        }
    ]
    source = [
        {"role": "user", "content": "Check two numbers"},
        {"role": "assistant", "content": answer},
    ]
    if with_tools:
        source[1:1] = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should check both numbers.",
                "tool_calls": [
                    {
                        "id": f"call_000{i}",
                        "type": "function",
                        "function": {"name": "check", "arguments": {"n": i}},
                    }
                    for i in range(1, tool_count + 1)
                ],
            },
            *[
                {
                    "role": "tool",
                    "name": "check",
                    "tool_call_id": f"call_000{i}",
                    "content": f"RESULT_{i}",
                }
                for i in range(1, tool_count + 1)
            ],
        ]
    original = copy.deepcopy(source)
    log = get_full_conversation_message_log(
        source,
        tokenizer,
        prompt=None,
        tools=tools if with_tools else None,
        add_bos_token=False,
        add_eos_token=False,
    )
    expected = tokenizer.apply_chat_template(
        source,
        tools=tools if with_tools else None,
        tokenize=False,
        add_generation_prompt=False,
    )
    ids = torch.cat([m["token_ids"] for m in log])
    assert ids.tolist() == tokenizer(expected, add_special_tokens=False)["input_ids"]
    assert source == original
    assistant = [m for m in log if m["role"] == "assistant"]
    assert len(assistant) == (2 if with_tools else 1)
    assert "Both checked." in assistant[-1]["content"]
    assert not any("RESULT_" in m["content"] for m in assistant)
    if with_tools:
        assert "check" in assistant[0]["content"]
    add_loss_mask_to_message_log([log], only_unmask_final=True)
    assert assistant[-1]["token_loss_mask"].all()
    assert not any(m["token_loss_mask"].any() for m in log[:-1])


@pytest.mark.parametrize("truncate_history", [False, True])
def test_stock_nemotron_history_reasoning(truncate_history: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", revision=NEMOTRON_REVISION
    )
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template,
        enable_thinking=True,
        truncate_history_thinking=truncate_history,
    )
    source = messages() + [
        {"role": "user", "content": "Summarize"},
        {
            "role": "assistant",
            "content": "Summary",
            "reasoning_content": "CURRENT_REASONING",
        },
    ]
    log = get_full_conversation_message_log(
        source,
        tokenizer,
        prompt=None,
        tools=None,
        add_bos_token=False,
        add_eos_token=False,
    )
    expected = tokenizer.apply_chat_template(source, tokenize=False)
    assert (
        torch.cat([m["token_ids"] for m in log]).tolist()
        == tokenizer(expected, add_special_tokens=False)["input_ids"]
    )
    assistant = [m for m in log if m["role"] == "assistant"]
    assert ("REASONING" in assistant[0]["content"]) == (not truncate_history)
    assert "CURRENT_REASONING" in assistant[-1]["content"]
    assert not any("RESULT_" in m["content"] for m in assistant)


@pytest.mark.parametrize("limit", [None, 0, -1])
def test_full_conversation_requires_positive_limit(
    tokenizer: PreTrainedTokenizerFast, limit: int | None
) -> None:
    with pytest.raises(ValueError, match="positive max_seq_length"):
        PROCESSOR_REGISTRY["sft_full_conversation_processor"](
            {"messages": messages(), "task_name": "test"},
            TaskDataSpec(),
            tokenizer,
            limit,
            0,
        )


def test_automatic_tracking_preserves_template_errors(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m.tool_calls | default([]) | length > 1 %}"
        "{{ raise_exception('Only one tool call supported') }}"
        "{% endif %}{{ m.content }}{% endfor %}"
    )
    with pytest.raises(TemplateError, match="Only one tool call supported"):
        format_messages(tokenizer)


def test_render_options_exclude_tokenizer_controls(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template,
        tokenize=True,
        return_dict=True,
        padding=True,
        truncation=True,
        max_length=10,
        return_tensors="pt",
        tokenizer_kwargs={"padding": True},
    )
    with patch(
        "nemo_rl.data.full_conversation.render_jinja_template",
        wraps=render_jinja_template,
    ) as render:
        format_messages(tokenizer)
    assert not {
        "tokenize",
        "return_dict",
        "padding",
        "truncation",
        "max_length",
        "return_tensors",
        "tokenizer_kwargs",
    }.intersection(render.call_args.kwargs)
