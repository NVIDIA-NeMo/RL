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
from omegaconf import OmegaConf

from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
from nemo_rl.utils.config import register_omegaconf_resolvers, load_config
from nemo_rl.data.utils import setup_response_data
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)
from nemo_rl.data.native_chat import _render_and_tokenize_chat
from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
    OpenAIFormatDataset,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import chat_kd_processor

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures/xtoken"
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
    qwen_tokenizer, enable_thinking
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
        chat_template=(FIXTURES / "qwen3_4b_stock.jinja").read_text(),
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
def test_training_history_matches_history_preserving_reference(qwen_tokenizer, inline):
    messages = conversation(inline)
    ours = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    custom = qwen_tokenizer.apply_chat_template(
        messages,
        chat_template=(FIXTURES / "qwen_thinking_history_reference.jinja").read_text(),
        tokenize=False,
        add_generation_prompt=False,
    )
    assert ours == custom


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
        skip_overlength=True,
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
    assert document.source_turn_indices == ([1, 3] if select_all else [3])
    assert "Earlier unique reasoning" in document.rendered_text
    assert "Latest unique reasoning" in document.rendered_text
    for phrase, supervised in [
        (conversation()[1]["reasoning_content"], select_all),
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
    assert '"enabled": true' in decoded_targets
    assert '"optional": null' in decoded_targets
    assert "Latest unique reasoning" in decoded_targets
    assert "System insertion" not in decoded_targets
    assert "A tool result" not in decoded_targets
    assert batch["alignment_0_pair_valid"].any()


def test_stock_qwen_dropped_history_fails_with_sample_context(qwen_tokenizer):
    student = copy.deepcopy(qwen_tokenizer)
    student.chat_template = (FIXTURES / "qwen3_4b_stock.jinja").read_text()
    with pytest.raises(
        ValueError, match="student, sample idx=71.*(reasoning|historical)"
    ):
        _native_collator(student, None, same_tokenizer=True)(
            [
                {
                    "message_log": conversation(),
                    "message_loss_mask": [0, 0, 0, 1],
                    "loss_multiplier": 1.0,
                    "idx": 71,
                }
            ]
        )


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
def test_native_collator_rejects_configured_history_truncation(flags):
    tokenizer = get_tokenizer(
        {
            "name": "Qwen/Qwen3-4B",
            "tokenizer_kwargs": {"revision": REVISION},
            "chat_template": str(TEMPLATE),
            "chat_template_kwargs": flags,
        }
    )
    with pytest.raises(ValueError, match="student, sample idx=29.*reasoning"):
        _native_collator(tokenizer, None, same_tokenizer=True)(
            [{"message_log": conversation(), "loss_multiplier": 1.0, "idx": 29}]
        )


def test_native_recipe_loads_exact_tokenizers_and_dataset():
    register_omegaconf_resolvers()
    path = (
        ROOT
        / "examples/configs/recipes/llm/distillation-xtoken-qwen3-4b-to-qwen3-0.6b-1n8g-fsdp2tp1-native-chat.yaml"
    )
    config = MasterConfig.model_validate(
        OmegaConf.to_container(load_config(path), resolve=True)
    )
    student = get_tokenizer(config.policy["tokenizer"])
    teacher = get_tokenizer(config.teachers[0].policy_config()["tokenizer"])
    train, validation = setup_response_data(student, config.data, env_configs=None)
    assert validation is None
    rows = [train[index] for index in range(len(train))]
    assert len(rows) == 8
    collator = CrossTokenizerCollator(
        config=config.collator,
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[TokenAligner(student, teacher, "unused.pt")],
        ctx_length_student=config.policy["max_total_sequence_length"],
        ctx_length_teachers=[config.teachers[0].max_total_sequence_length],
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    batch = collator(rows)
    assert batch["input_ids"].shape[0] == 8
    assert batch["alignment_0_pair_valid"].any(dim=1).all()
    assert batch["token_mask"].sum(dim=1).min() > 0
