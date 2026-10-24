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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import render_jinja_template

from nemo_rl.data.full_conversation import get_full_conversation_message_log
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
def test_historical_eos_outside_generation_is_rejected(
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
    with pytest.raises(ValueError, match="EOS terminator is outside"):
        format_messages(
            tokenizer,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "last"},
            ],
        )


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
            "prefix{% for m in messages if m.role == 'assistant' %}{% generation %}x<stop>{% endgeneration %}\\n{% endfor %}",
            "trailing whitespace",
        ),
    ],
)
def test_invalid_generation_templates(
    tokenizer: PreTrainedTokenizerFast, template: str, match: str
) -> None:
    tokenizer.chat_template = template
    with pytest.raises(ValueError, match=match):
        format_messages(tokenizer)


def test_boundary_crossing_is_rejected(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.chat_template = "hel{% generation %}lo<stop>{% endgeneration %}"
    with pytest.raises(ValueError, match="crosses"):
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


def test_adjacent_assistants_stay_separate(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.chat_template = """prefix{% for m in messages if m.role == 'assistant' %}{% generation %}{{ m.content + '<stop>' }}{% endgeneration %}{% endfor %}"""
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
