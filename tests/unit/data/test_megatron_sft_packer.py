# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import sys
from typing import Any

import pytest

from examples.converters import pack_megatron_sft_jsonl
from examples.converters.pack_megatron_sft_jsonl import (
    iter_conversations,
    pack_conversations,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.megatron_sft_packed import megatron_sft_packed_preprocessor


class _DummyTokenizer:
    pad_token_id = 99
    eos_token_id = 2

    def __init__(self, turn_tokens: dict[tuple[str, str], list[int]]) -> None:
        self.turn_tokens = turn_tokens

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> list[int]:
        return [
            token
            for message in messages
            for token in self.turn_tokens[(message["role"], message["content"])]
        ]

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<unk>"
        return self.pad_token_id


def _preprocess(
    messages: list[dict[str, str]],
    tokenizer: _DummyTokenizer,
    max_seq_length: int,
    context_parallel_size: int,
) -> dict[str, Any]:
    return megatron_sft_packed_preprocessor(
        {"packed_messages": messages},
        TaskDataSpec(),
        tokenizer,
        max_seq_length,
        idx=0,
        prompt_format="identity",
        context_parallel_size=context_parallel_size,
    )


def _conversation(name: str, length: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": name},
        {"role": "assistant", "content": "x" * (length - 1)},
    ]


def _tokenizer(conversations: list[list[dict[str, str]]]) -> _DummyTokenizer:
    return _DummyTokenizer(
        {
            (message["role"], message["content"]): list(
                range(10, 10 + len(message["content"]))
            )
            for conversation in conversations
            for message in conversation
        }
    )


def test_packer_groups_whole_conversations_at_loader_token_boundaries() -> None:
    conversations = [
        _conversation("a", 4),
        _conversation("b", 4),
        _conversation("c", 4),
    ]
    tokenizer = _tokenizer(conversations)

    rows = list(pack_conversations(conversations, tokenizer, 8, "identity", 1))

    assert rows == [conversations[0] + conversations[1], conversations[2]]
    assert _preprocess(rows[0], tokenizer, 8, 1)["packed_cu_seqlens"].tolist() == [
        0,
        4,
        8,
    ]


def test_packer_accounts_for_per_conversation_cp_padding() -> None:
    conversations = [
        _conversation("a", 5),
        _conversation("b", 5),
        _conversation("c", 5),
    ]
    tokenizer = _tokenizer(conversations)

    rows = list(pack_conversations(conversations, tokenizer, 16, "identity", 2))

    assert rows == [conversations[0] + conversations[1], conversations[2]]
    assert _preprocess(rows[0], tokenizer, 16, 2)["packed_cu_seqlens"].tolist() == [
        0,
        8,
        16,
    ]


def test_packer_keeps_oversized_conversation_in_its_own_row() -> None:
    conversations = [
        _conversation("a", 4),
        _conversation("b", 12),
        _conversation("c", 4),
    ]

    rows = list(
        pack_conversations(conversations, _tokenizer(conversations), 8, "identity", 1)
    )

    assert rows == conversations


def test_packer_rejects_invalid_cp_or_message_boundaries() -> None:
    conversations = [_conversation("a", 4)]
    tokenizer = _tokenizer(conversations)

    with pytest.raises(ValueError, match="multiple of"):
        list(pack_conversations(conversations, tokenizer, 7, "identity", 2))
    with pytest.raises(ValueError, match="start with a system"):
        list(
            pack_conversations(
                [[{"role": "assistant", "content": "bad"}]], tokenizer, 8, "identity", 1
            )
        )


def test_packer_cli_reads_and_writes_loader_compatible_jsonl(
    tmp_path, monkeypatch
) -> None:
    conversations = [
        _conversation("a", 4),
        _conversation("b", 4),
        _conversation("c", 4),
    ]
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "packed.jsonl.packed"
    input_path.write_text(
        "".join(
            json.dumps({"messages": conversation}) + "\n"
            for conversation in conversations
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pack_megatron_sft_jsonl.AutoTokenizer,
        "from_pretrained",
        lambda _: _tokenizer(conversations),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pack_megatron_sft_jsonl.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--tokenizer",
            "unused",
            "--max-seq-length",
            "8",
            "--prompt-format",
            "identity",
        ],
    )

    pack_megatron_sft_jsonl.main()

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {"messages": conversations[0] + conversations[1]},
        {"messages": conversations[2]},
    ]


def test_iter_conversations_rejects_malformed_boundary(tmp_path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "missing system"}]})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="start with a system"):
        list(iter_conversations(input_path))
