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

"""Pinned real-tokenizer tests for the maintained Qwen history template."""

import copy
import json
import pickle
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
    OpenAIFormatDataset,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import chat_kd_processor
from nemo_rl.data.xtoken.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)
from nemo_rl.data.xtoken.native_chat import _render_and_tokenize_chat

ROOT = Path(__file__).resolve().parents[3]
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TEMPLATE = ROOT / "examples/chat_templates/qwen3_history.jinja"
FLAGS = {
    "enable_thinking": True,
    "preserve_thinking": True,
    "truncate_history_thinking": False,
}


def conversation(inline: bool = False) -> list[dict]:
    messages = [
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "content": "Earlier answer",
            "reasoning_content": "Earlier unique reasoning: café 中文 🌟",
        },
        {"role": "user", "content": "Later question"},
        {
            "role": "assistant",
            "content": "Latest answer",
            "reasoning_content": "Latest unique reasoning: naïve 日本語 🚀",
        },
    ]
    if inline:
        for message in messages:
            if message["role"] == "assistant":
                message["content"] = (
                    "<think>\n"
                    + message.pop("reasoning_content")
                    + "\n</think>\n\n"
                    + message["content"]
                )
    return messages


@pytest.fixture(scope="module")
def stock_qwen_tokenizer() -> PreTrainedTokenizerBase:
    return get_tokenizer(
        {
            "name": "Qwen/Qwen3-4B",
            "tokenizer_kwargs": {"revision": REVISION},
        }
    )


@pytest.fixture(scope="module")
def stock_qwen_thinking_tokenizer() -> PreTrainedTokenizerBase:
    return get_tokenizer(
        {
            "name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
            "tokenizer_kwargs": {
                "revision": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
            },
        }
    )


@pytest.fixture(scope="module")
def qwen_tokenizer():
    # Tokenizer assets only; the immutable revision prevents moving HF defaults.
    return get_tokenizer(
        {
            "name": "Qwen/Qwen3-4B",
            "tokenizer_kwargs": {"revision": REVISION},
            "chat_template": str(TEMPLATE),
            "chat_template_kwargs": FLAGS,
        }
    )


@pytest.mark.parametrize("inline", [False, True])
def test_configured_tokenizer_retains_both_reasoning_turns(qwen_tokenizer, inline):
    messages = conversation(inline)
    original = copy.deepcopy(messages)
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    assert "Earlier unique reasoning" in text
    assert "Latest unique reasoning" in text
    assert text.count("<think>") == 2
    assert messages == original
    restored = pickle.loads(pickle.dumps(qwen_tokenizer))
    assert (
        restored.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        == text
    )


@pytest.mark.parametrize(
    "flags,earlier",
    [
        ({}, False),
        ({"preserve_thinking": True}, True),
        ({"preserve_thinking": False}, False),
        ({"truncate_history_thinking": False}, True),
        ({"preserve_thinking": True, "truncate_history_thinking": True}, False),
        ({"preserve_thinking": False, "truncate_history_thinking": False}, True),
        ({"enable_thinking": False, "preserve_thinking": True}, True),
    ],
)
def test_flag_precedence(qwen_tokenizer, flags, earlier):
    # Invoke the class method to exercise omitted kwargs rather than the
    # configured instance partial that supplies the three production flags.
    text = type(qwen_tokenizer).apply_chat_template(
        qwen_tokenizer,
        conversation(),
        tokenize=False,
        add_generation_prompt=False,
        **flags,
    )
    assert ("Earlier unique reasoning" in text) is earlier
    assert "Latest unique reasoning" in text


@pytest.mark.parametrize("enable_thinking", [None, False, True])
def test_stock_generation_prompt_and_tool_serialization(
    qwen_tokenizer,
    stock_qwen_tokenizer: PreTrainedTokenizerBase,
    enable_thinking,
):
    messages = conversation()
    messages[1]["tool_calls"] = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "arguments": {
                    "query": "<think>literal</think>",
                    "enabled": True,
                    "optional": None,
                },
            },
        }
    ]
    messages.insert(2, {"role": "tool", "content": "Found result"})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    flags = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    render = type(qwen_tokenizer).apply_chat_template
    stock = render(
        qwen_tokenizer,
        messages,
        chat_template=stock_qwen_tokenizer.chat_template,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **flags,
    )
    adapted_default = render(
        qwen_tokenizer,
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **flags,
    )
    assert adapted_default == stock
    adapted = render(
        qwen_tokenizer,
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        preserve_thinking=True,
        **flags,
    )
    assert (
        adapted.rsplit("<|im_start|>assistant", 1)[1]
        == stock.rsplit("<|im_start|>assistant", 1)[1]
    )
    assert (
        adapted.split("<tools>", 1)[1].split("</tools>", 1)[0]
        == stock.split("<tools>", 1)[1].split("</tools>", 1)[0]
    )
    assert (
        adapted.split('<tool_call>\n{"name": "search"', 1)[1].split("</tool_call>", 1)[
            0
        ]
        == stock.split('<tool_call>\n{"name": "search"', 1)[1].split("</tool_call>", 1)[
            0
        ]
    )


@pytest.mark.parametrize("inline", [False, True])
def test_training_history_matches_history_preserving_reference(
    qwen_tokenizer: PreTrainedTokenizerBase,
    stock_qwen_thinking_tokenizer: PreTrainedTokenizerBase,
    inline: bool,
) -> None:
    messages = conversation(inline)
    ours = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    reference_template = stock_qwen_thinking_tokenizer.get_chat_template()
    # Preserve earlier reasoning by disabling only the upstream history guard.
    history_condition = "loop.index0 > ns.last_query_index"
    assert reference_template.count(history_condition) == 1
    reference_template = reference_template.replace(history_condition, "true")
    reference = stock_qwen_thinking_tokenizer.apply_chat_template(
        messages,
        chat_template=reference_template,
        tokenize=False,
        add_generation_prompt=False,
    )
    assert ours == reference


@pytest.mark.parametrize("teacher_kind", ["ce", "same_tokenizer", "cross_tokenizer"])
def test_stock_qwen_ordinary_chat_trains_only_final_repeated_answer(
    stock_qwen_tokenizer: PreTrainedTokenizerBase, teacher_kind: str
) -> None:
    student = copy.deepcopy(stock_qwen_tokenizer)
    teacher = None if teacher_kind == "ce" else copy.deepcopy(stock_qwen_tokenizer)
    messages = [
        {"role": "user", "content": "Please repeat Same answer"},
        {"role": "assistant", "content": "Same answer"},
        {"role": "user", "content": "Repeat it again"},
        {"role": "assistant", "content": "Same answer"},
    ]
    original = copy.deepcopy(messages)
    rendered = student.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    earlier_render = student.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=False
    )
    # Stock Qwen changes the earlier answer's scaffold once another user turn
    # follows it. This is the multi-turn case that prefix comparisons rejected.
    assert not rendered.startswith(earlier_render)
    selected = [0, 1, 0, 1]
    document = _render_and_tokenize_chat(
        student,
        messages,
        1024,
        message_loss_mask=selected,
        include_thinking_in_loss=False,
        native_thinking_alignment=False,
    )
    cross_tokenizer = teacher_kind == "cross_tokenizer"
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(mode="chat"),
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[
            TokenAligner(student, teacher, "unused.pt") if cross_tokenizer else None
        ],
        ctx_length_student=1024,
        ctx_length_teachers=[1024],
        drop_first_assistant_chunk_kl_by_teacher=[False],
        make_seq_div_by_student=64,
        make_seq_div_by_teachers=[64],
    )
    batch = collator(
        [
            {
                "message_log": messages,
                "message_loss_mask": selected,
                "loss_multiplier": 1.0,
                "idx": 71,
            }
        ]
    )
    assert messages == original
    assert document.rendered_text == rendered
    assert document.source_turn_indices == [3]
    answer_start = rendered.rindex("Same answer")
    answer_end = answer_start + len("Same answer")
    assert document.assistant_spans == [(answer_start, answer_end)]
    eot_id = student.convert_tokens_to_ids("<|im_end|>")
    final_eot = max(
        i for i, token_id in enumerate(document.input_ids) if token_id == eot_id
    )
    assert document.eot_indices == [final_eot]
    target_indices = [
        i
        for i, (start, end) in enumerate(document.offsets)
        if answer_start <= start < end <= answer_end
    ] + [final_eot]
    assert target_indices
    expected_ids = student(rendered, add_special_tokens=False)["input_ids"]
    for prefix in ["", "teacher_0_"] if cross_tokenizer else [""]:
        length = batch[f"{prefix}input_lengths"][0].item()
        assert batch[f"{prefix}input_ids"][0, :length].tolist() == expected_ids
        mask = batch[f"{prefix}token_mask"][0]
        assert mask.nonzero(as_tuple=True)[0].tolist() == target_indices
        if cross_tokenizer:
            side = "teacher" if prefix else "student"
            chunks = batch[f"alignment_0_{side}_chunk_id"][0]
            assert chunks[mask == 0].eq(-1).all()
            assert chunks[mask == 1].ge(0).all()
            assert batch["alignment_0_pair_valid"][0, chunks[mask == 1]].all()
            assert batch["alignment_0_pair_is_correct"][0, chunks[mask == 1]].all()
    if not cross_tokenizer:
        assert "teacher_0_input_ids" not in batch
        assert "alignment_0_pair_valid" not in batch


def _native_collator(student, teacher, *, same_tokenizer=False):
    return CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(
            mode="chat",
            include_thinking_in_loss=True,
            native_thinking_alignment=True,
            kd_alignment_regions=["reasoning", "close", "answer", "eot"],
        ),
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[
            None if same_tokenizer else TokenAligner(student, teacher, "unused.pt")
        ],
        ctx_length_student=1024,
        ctx_length_teachers=[1024],
        drop_first_assistant_chunk_kl_by_teacher=[False],
        make_seq_div_by_student=64,
        make_seq_div_by_teachers=[64],
    )


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("select_all", [False, True])
@pytest.mark.parametrize("same_tokenizer", [False, True])
def test_real_qwen_history_masks_and_kd(
    qwen_tokenizer, inline, select_all, same_tokenizer
):
    student = copy.deepcopy(qwen_tokenizer)
    teacher = copy.deepcopy(qwen_tokenizer)
    messages = conversation(inline)
    selected = [0, int(select_all), 0, 1]
    document = _render_and_tokenize_chat(
        student,
        messages,
        1024,
        message_loss_mask=selected,
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
    )
    collator = _native_collator(student, teacher, same_tokenizer=same_tokenizer)
    batch = collator(
        [
            {
                "message_log": messages,
                "message_loss_mask": selected,
                "loss_multiplier": 1.0,
                "idx": 17,
            }
        ]
    )
    assert document.source_turn_indices == [3]
    assert "Earlier unique reasoning" in document.rendered_text
    assert "Latest unique reasoning" in document.rendered_text
    for phrase, supervised in [
        (conversation()[1]["reasoning_content"], False),
        (conversation()[3]["reasoning_content"], True),
    ]:
        start = document.rendered_text.index(phrase)
        end = start + len(phrase)
        indices = [
            i for i, (a, b) in enumerate(document.offsets) if start <= a < b <= end
        ]
        assert indices
        assert all(batch["token_mask"][0, i].item() == int(supervised) for i in indices)
        if not same_tokenizer:
            chunks = batch["alignment_0_student_chunk_id"][0, indices]
            if supervised:
                assert (chunks >= 0).all()
                assert batch["alignment_0_pair_valid"][0, chunks].all()
                assert batch["alignment_0_pair_is_correct"][0, chunks].all()
                assert set(chunks.tolist()).issubset(
                    set(batch["alignment_0_teacher_chunk_id"][0].tolist())
                )
            else:
                assert (chunks == -1).all()
    for index in document.eot_indices:
        assert batch["token_mask"][0, index] == 1
    assert not batch["token_mask"][0, len(document.input_ids) :].any()
    if same_tokenizer:
        assert "teacher_0_input_ids" not in batch
    else:
        assert not (
            batch["alignment_0_student_chunk_id"][0, len(document.input_ids) :] >= 0
        ).any()
        assert batch["token_mask"][
            0, batch["alignment_0_student_chunk_id"][0] >= 0
        ].all()


def test_real_qwen_dataset_tools_to_collator(qwen_tokenizer, tmp_path):
    row = {"messages": conversation(), "message_loss_mask": [0, 1, 0, 1]}
    row["messages"][1].pop(
        "reasoning_content"
    )  # Valid context-dependent empty scaffold.
    row["messages"][1]["content"] = ""
    row["messages"][1]["tool_calls"] = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "arguments": {
                    "query": "<tool_call>nested</tool_call>",
                    "enabled": True,
                    "optional": None,
                },
            },
        }
    ]
    row["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    row["messages"].insert(2, {"role": "tool", "content": "A tool result"})
    row["message_loss_mask"].insert(2, 0)
    path = tmp_path / "tools.jsonl"
    path.write_text(json.dumps(row) + "\n")
    dataset = OpenAIFormatDataset(
        str(path), use_preserving_dataset=True, system_prompt="System insertion"
    )
    datum = chat_kd_processor(
        dataset.dataset[0], TaskDataSpec(), qwen_tokenizer, 1024, 0
    )
    assert datum["message_loss_mask"] == [0, 0, 1, 0, 0, 1]
    assert datum["tools"] == row["tools"]
    student, teacher = copy.deepcopy(qwen_tokenizer), copy.deepcopy(qwen_tokenizer)
    batch = _native_collator(student, teacher)([datum])
    decoded_targets = student.decode(
        batch["input_ids"][0, batch["token_mask"][0].bool()].tolist()
    )
    decoded_context = student.decode(batch["input_ids"][0].tolist())
    assert '"enabled": true' in decoded_context
    assert '"optional": null' in decoded_context
    assert "Earlier answer" not in decoded_targets
    assert "<tool_call>" not in decoded_targets
    assert "Latest unique reasoning" in decoded_targets
    assert "System insertion" not in decoded_targets
    assert "A tool result" not in decoded_targets
    assert batch["alignment_0_pair_valid"].any()


def test_stock_qwen_native_mode_accepts_template_defined_history(
    stock_qwen_tokenizer: PreTrainedTokenizerBase,
):
    student = copy.deepcopy(stock_qwen_tokenizer)
    batch = _native_collator(student, None, same_tokenizer=True)(
        [
            {
                "message_log": conversation(),
                "message_loss_mask": [0, 1, 0, 1],
                "loss_multiplier": 1.0,
                "idx": 71,
            }
        ]
    )
    context = student.decode(batch["input_ids"][0].tolist())
    targets = student.decode(
        batch["input_ids"][0, batch["token_mask"][0].bool()].tolist()
    )
    assert "Earlier answer" in context
    assert "Earlier unique reasoning" not in context
    assert "Earlier answer" not in targets
    assert "Latest unique reasoning" in targets
    assert "Latest answer" in targets


@pytest.mark.parametrize(
    "flags",
    [
        {
            "enable_thinking": True,
            "preserve_thinking": True,
            "truncate_history_thinking": True,
        },
        {"preserve_thinking": False},
    ],
)
def test_native_collator_respects_configured_history_rendering(flags):
    tokenizer = get_tokenizer(
        {
            "name": "Qwen/Qwen3-4B",
            "tokenizer_kwargs": {"revision": REVISION},
            "chat_template": str(TEMPLATE),
            "chat_template_kwargs": flags,
        }
    )
    batch = _native_collator(tokenizer, None, same_tokenizer=True)(
        [
            {
                "message_loss_mask": [0, 1, 0, 1],
                "message_log": conversation(),
                "loss_multiplier": 1.0,
                "idx": 29,
            }
        ]
    )
    context = tokenizer.decode(batch["input_ids"][0].tolist())
    targets = tokenizer.decode(
        batch["input_ids"][0, batch["token_mask"][0].bool()].tolist()
    )
    assert "Earlier answer" in context
    assert "Earlier unique reasoning" not in context
    assert "Earlier answer" not in targets
    assert "Latest unique reasoning" in targets
    assert "Latest answer" in targets
