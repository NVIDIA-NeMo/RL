# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Greedily pack text SFT conversations for the Megatron-LM SFT dataset path."""

import argparse
import json
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from nemo_rl.data.megatron_sft_packed import (
    count_conversation_tokens,
    direct_packed_cp_granularity,
    split_megatron_sft_conversations,
    validate_megatron_sft_prompt_format,
)


def iter_conversations(input_path: Path) -> Iterator[list[dict[str, str]]]:
    """Read raw JSONL rows and split them at system-message boundaries."""
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"line {line_number}: messages must be a nonempty list"
                )
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(
                        f"line {line_number}: every message needs string role and content"
                    )
            for conversation in split_megatron_sft_conversations(messages):
                if (
                    conversation[0]["role"] != "system"
                    or conversation[-1]["role"] != "assistant"
                ):
                    raise ValueError(
                        f"line {line_number}: each conversation must start with a system "
                        "and end with an assistant message"
                    )
                yield conversation


def pack_conversations(
    conversations: Iterable[list[dict[str, str]]],
    tokenizer: Any,
    max_seq_length: int,
    prompt_format: str,
    context_parallel_size: int,
) -> Iterator[list[dict[str, str]]]:
    """Yield rows sized by the same tokenization and per-segment CP padding as the loader."""
    validate_megatron_sft_prompt_format(prompt_format)
    if max_seq_length < 1 or context_parallel_size < 1:
        raise ValueError("max_seq_length and context_parallel_size must be positive")
    granularity = direct_packed_cp_granularity(context_parallel_size)
    if context_parallel_size > 1 and max_seq_length % granularity != 0:
        raise ValueError(
            f"max_seq_length must be a multiple of 2 * context_parallel_size ({granularity})"
        )

    row: list[dict[str, str]] = []
    row_tokens = 0
    for conversation in conversations:
        if not conversation or conversation[0]["role"] != "system":
            raise ValueError("each conversation must start with a system message")
        if conversation[-1]["role"] != "assistant":
            raise ValueError("each conversation must end with an assistant message")
        token_count = count_conversation_tokens(conversation, tokenizer, prompt_format)
        if token_count == 0:
            raise ValueError("Megatron SFT conversation tokenized to zero tokens")
        segment_tokens = (
            (token_count + granularity - 1) // granularity * granularity
            if context_parallel_size > 1
            else token_count
        )
        if row and row_tokens + segment_tokens > max_seq_length:
            yield row
            row = []
            row_tokens = 0
        if segment_tokens > max_seq_length:
            warnings.warn(
                "Megatron SFT conversation exceeds max_seq_length and will be "
                "truncated by the loader; emitting it as a separate row",
                stacklevel=2,
            )
            yield conversation
            continue
        row.extend(conversation)
        row_tokens += segment_tokens
        if row_tokens == max_seq_length:
            yield row
            row = []
            row_tokens = 0
    if row:
        yield row


def main() -> None:
    """Pack a raw JSONL dataset into fixed-length Megatron SFT rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--prompt-format", required=True)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output paths must differ")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = pack_conversations(
        iter_conversations(args.input),
        tokenizer,
        args.max_seq_length,
        args.prompt_format,
        args.context_parallel_size,
    )
    with args.output.open("w", encoding="utf-8") as output:
        for messages in rows:
            output.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
