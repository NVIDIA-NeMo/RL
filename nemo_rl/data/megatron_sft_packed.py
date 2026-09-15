# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron-LM SFT packed JSONL preprocessing helpers."""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec

IGNORE_INDEX = -100
IDENTITY_TEMPLATE = (
    """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
)
NEMOTRON_H_ALIGNED_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}"""
NEMOTRON_NANO_V2_TEMPLATE = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}"""


@dataclass(frozen=True)
class _PromptConfig:
    assistant_prefix_len: int
    pad_token: str | None
    chat_template: str
    has_bos: bool = False
    has_system_role: bool = True


_PROMPT_CONFIGS = {
    "identity": _PromptConfig(
        assistant_prefix_len=0,
        pad_token="<unk>",
        chat_template=IDENTITY_TEMPLATE,
    ),
    "nemotron-nano-v2": _PromptConfig(
        assistant_prefix_len=3,
        pad_token="<unk>",
        chat_template=NEMOTRON_NANO_V2_TEMPLATE,
    ),
    "nemotron-h-aligned": _PromptConfig(
        assistant_prefix_len=0,
        pad_token="<SPECIAL_233>",
        chat_template=NEMOTRON_H_ALIGNED_TEMPLATE,
    ),
}


def validate_megatron_sft_prompt_format(prompt_format: str) -> None:
    if prompt_format not in _PROMPT_CONFIGS:
        raise NotImplementedError("unknown SFT prompt format", prompt_format)


class MegatronSFTPackedDatumSpec(DatumSpec):
    """One offline-packed SFT row, pre-shifted for the direct Megatron-LM path.

    A ``.jsonl.packed`` record is re-tokenized at load time into a single row of
    exactly ``max_seq_length`` target-aligned positions: ``input_ids`` is the
    pack without its last token and ``target_ids`` is the pack without its
    first, so nothing downstream shifts again.

    Attributes:
        input_ids: ``[max_seq_length]`` token ids, ``pack[:-1]``.
        target_ids: ``[max_seq_length]`` labels, ``pack[1:]``. Prompt positions
            and segment boundaries carry ``IGNORE_INDEX``; trailing padding
            carries the pad id.
        token_mask: ``[max_seq_length]`` float mask, ``0.0`` wherever
            ``target_ids`` is padding or ``IGNORE_INDEX``. This is not cosmetic:
            mcore's fused cross-entropy clamps ``IGNORE_INDEX`` before the vocab
            gather and forces ``predicted_logits = 0.0``, so an ignored position
            still emits ``log(sum_exp_logits)`` - a large finite loss with a
            non-zero gradient. ``token_mask`` is the only thing that zeroes both
            the loss and the gradient at those positions.
        position_ids: ``[max_seq_length]`` positions that restart at 0 on every
            packed segment boundary.
        packed_cu_seqlens: ``[num_segments + 1]`` int32 cumulative segment
            boundaries, from 0 through the packed row length. Its presence is
            the single marker that a row took this path; see
            :func:`is_direct_packed_row`.
        packed_max_seqlen: Longest packed segment, used to build
            ``PackedSeqParams``.
        packed_context_parallel_size: Context-parallel size the row was padded
            for. Every segment length is a multiple of
            :func:`direct_packed_cp_granularity` because mcore shards packed
            rows with *per-document* zigzag.
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    token_mask: torch.Tensor
    position_ids: torch.Tensor
    packed_cu_seqlens: torch.Tensor
    packed_max_seqlen: int
    packed_context_parallel_size: int


MEGATRON_SFT_PACKED_FIELDS = frozenset(
    MegatronSFTPackedDatumSpec.__annotations__
) - frozenset(DatumSpec.__annotations__)


# Fields a collated direct-packed microbatch carries. ``packed_context_parallel_size``
# is consumed and validated by the collate itself, and the collate is what adds
# ``sample_mask`` and ``packed_cu_seqlens_lengths``.
MEGATRON_SFT_PACKED_BATCH_FIELDS = (
    MEGATRON_SFT_PACKED_FIELDS - {"packed_context_parallel_size"}
) | {"sample_mask", "packed_cu_seqlens_lengths"}


def is_direct_packed_row(row: Mapping[str, Any]) -> bool:
    """Return whether ``row`` came from the direct Megatron-LM prepacked path.

    ``packed_cu_seqlens`` is the one authoritative marker. Every other field in
    :data:`MEGATRON_SFT_PACKED_FIELDS` is emitted alongside it by
    :func:`megatron_sft_packed_preprocessor`, so testing any other key answers a
    different question and the answers drift apart.
    """
    return "packed_cu_seqlens" in row


def direct_packed_cp_granularity(context_parallel_size: int) -> int:
    """Return the token multiple every direct-packed segment must satisfy.

    mcore shards packed rows with per-document zigzag, which cuts each document
    into ``2 * cp_size`` chunks, so a segment length that is not a multiple of
    this value cannot be sharded.
    """
    return 2 * context_parallel_size


def split_megatron_sft_conversations(
    merged_messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split a packed row whenever a new system message begins."""
    conversations: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in merged_messages:
        if message["role"] == "system":
            if current:
                conversations.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        conversations.append(current)
    return conversations


def _get_prompt_config(
    prompt_format: str | None,
    pad_token: str | None,
    assistant_prefix_len: int | None,
) -> _PromptConfig:
    if prompt_format is None:
        raise NotImplementedError("unknown SFT prompt format", prompt_format)
    validate_megatron_sft_prompt_format(prompt_format)

    default = _PROMPT_CONFIGS[prompt_format]
    resolved_assistant_prefix_len = (
        default.assistant_prefix_len
        if assistant_prefix_len is None
        else assistant_prefix_len
    )
    if resolved_assistant_prefix_len < 0:
        raise ValueError("assistant_prefix_len must be >= 0")
    if prompt_format == "identity" and resolved_assistant_prefix_len != 0:
        raise ValueError("identity prompt format does not support assistant_prefix_len")
    return _PromptConfig(
        assistant_prefix_len=resolved_assistant_prefix_len,
        pad_token=default.pad_token if pad_token is None else pad_token,
        chat_template=default.chat_template,
        has_bos=default.has_bos,
        has_system_role=default.has_system_role,
    )


def _normalize_token_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "get"):
        input_ids = token_ids.get("input_ids")
        if input_ids is not None:
            token_ids = input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    token_ids = list(token_ids)
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected one tokenized Megatron SFT conversation")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _tokenize_megatron_sft_conversation(
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    prompt_format: str,
    prompt_config: _PromptConfig,
) -> tuple[list[int], list[int]]:
    if not prompt_config.has_system_role and conversation[0]["role"] == "system":
        conversation = conversation[1:]

    tokens = _normalize_token_ids(
        tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_token_mask=False,
            return_tensors="np",
            chat_template=prompt_config.chat_template,
        )
    )
    targets = list(tokens)

    if prompt_format == "identity":
        return tokens, targets

    idx = 0
    for turn_idx, turn in enumerate(conversation):
        role = turn["role"].lower()
        if role == "assistant" and len(turn["content"]) == 0:
            raise ValueError(f"empty assistant turn in conversation: {conversation}.")
        if role == "assistant" and (
            turn_idx == 0
            or conversation[turn_idx - 1]["role"].lower() not in ("user", "tool")
        ):
            raise ValueError(
                "An assistant turn must follow a user or tool turn in "
                f"conversation: {conversation}."
            )

        turn_tokens = _normalize_token_ids(
            tokenizer.apply_chat_template(
                [turn],
                tokenize=True,
                chat_template=prompt_config.chat_template,
            )
        )
        if prompt_config.has_bos and turn_idx > 0:
            turn_tokens = turn_tokens[1:]
        turn_len = len(turn_tokens)

        if role in ("system", "user", "tool"):
            targets[idx : idx + turn_len] = [IGNORE_INDEX] * turn_len
        elif role == "assistant":
            prefix_len = prompt_config.assistant_prefix_len
            if prefix_len > turn_len:
                raise ValueError(
                    f"assistant_prefix_len={prefix_len} exceeds assistant turn "
                    f"length={turn_len}"
                )
            targets[idx : idx + prefix_len] = [IGNORE_INDEX] * prefix_len
        else:
            raise ValueError("Wrong role value.")

        if tokens[idx : idx + turn_len] != turn_tokens:
            raise ValueError(
                f"expected turn tokens to match tokens in conversation {conversation}"
            )
        idx += turn_len

    if idx != len(tokens):
        raise ValueError(f"mismatch in target masking the conversation {conversation}")
    return tokens, targets


def _resolve_pad_token_id(tokenizer: Any, prompt_config: _PromptConfig) -> int:
    # An AutoProcessor exposes the text tokenizer as ``.tokenizer`` and has no
    # ``convert_tokens_to_ids`` of its own.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if prompt_config.pad_token is not None:
        pad_token_id = int(
            text_tokenizer.convert_tokens_to_ids(prompt_config.pad_token)
        )
    elif text_tokenizer.pad_token_id is not None:
        pad_token_id = int(text_tokenizer.pad_token_id)
    else:
        raise ValueError(
            "Megatron SFT packed data requires a pad token distinct from EOS"
        )

    if text_tokenizer.eos_token_id is not None and pad_token_id == int(
        text_tokenizer.eos_token_id
    ):
        raise ValueError(
            "Megatron SFT packed data requires a pad token distinct from EOS"
        )
    return pad_token_id


def megatron_sft_packed_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int | None,
    idx: int,
    *,
    prompt_format: str,
    context_parallel_size: int,
    pad_token: str | None = None,
    assistant_prefix_len: int | None = None,
    **_unused_kwargs: Any,
) -> MegatronSFTPackedDatumSpec:
    """Build one direct tensor row with Megatron-LM ``SFTDataset`` semantics."""
    del task_data_spec
    if max_seq_length is None or max_seq_length < 1:
        raise ValueError("max_seq_length must be a positive integer")
    if context_parallel_size < 1:
        raise ValueError("context_parallel_size must be >= 1")
    cp_granularity = direct_packed_cp_granularity(context_parallel_size)
    if context_parallel_size > 1 and max_seq_length % cp_granularity != 0:
        raise ValueError(
            f"max_seq_length={max_seq_length} must be a multiple of "
            f"2 * context_parallel_size ({cp_granularity}) for Megatron SFT "
            "packed data; set data.max_input_seq_length accordingly"
        )

    prompt_config = _get_prompt_config(prompt_format, pad_token, assistant_prefix_len)
    pack_length = max_seq_length
    pad = _resolve_pad_token_id(tokenizer, prompt_config)
    conversations = split_megatron_sft_conversations(datum_dict["packed_messages"])
    if not conversations:
        raise ValueError("Megatron SFT packed data requires at least one conversation")

    pack_tokens: list[int] = []
    pack_targets: list[int] = []
    pack_positions: list[int] = []
    cu_seqlens = [0]

    def extend_with_padding(pad_len: int) -> None:
        start_position = pack_positions[-1] + 1
        pack_tokens.extend([pad] * pad_len)
        pack_targets.extend([pad] * pad_len)
        pack_positions.extend(range(start_position, start_position + pad_len))

    truncated_conversation = False
    dropped_conversations = 0
    for conversation_index, conversation in enumerate(conversations):
        tokens, targets = _tokenize_megatron_sft_conversation(
            conversation,
            tokenizer,
            prompt_format,
            prompt_config,
        )
        if not tokens:
            raise ValueError("Megatron SFT conversation tokenized to zero tokens")

        pack_tokens.extend(tokens)
        pack_targets.extend(targets)
        pack_positions.extend(range(len(tokens)))

        if context_parallel_size > 1:
            mod_token_count = len(pack_tokens) % cp_granularity
            if mod_token_count != 0:
                extend_with_padding(cp_granularity - mod_token_count)

        cu_seqlens.append(len(pack_tokens))

        if len(pack_tokens) == pack_length:
            dropped_conversations = len(conversations) - conversation_index - 1
            break

        if len(pack_tokens) >= pack_length + 1:
            pack_tokens = pack_tokens[:pack_length]
            pack_targets = pack_targets[:pack_length]
            pack_tokens.append(pad)
            pack_targets.append(pad)
            pack_positions = pack_positions[: pack_length + 1]
            cu_seqlens[-1] = len(pack_tokens) - 1
            truncated_conversation = True
            dropped_conversations = len(conversations) - conversation_index - 1
            break

    if truncated_conversation or dropped_conversations:
        # The offline packer already sized this row to max_seq_length. Overflow
        # here means retokenization was more verbose than the packer's, so the
        # tail of the row is silently thrown away.
        truncation_note = (
            "the last conversation was truncated and " if truncated_conversation else ""
        )
        warnings.warn(
            f"Megatron SFT packed row idx={idx} overflowed max_seq_length="
            f"{max_seq_length} during retokenization: {truncation_note}"
            f"{dropped_conversations} of {len(conversations)} conversations were "
            "dropped. The loader tokenizer and pack length must match the ones "
            "the offline packer used.",
            stacklevel=2,
        )

    if len(pack_tokens) < pack_length + 1:
        extend_with_padding(pack_length + 1 - len(pack_tokens))
        cu_seqlens[-1] = len(pack_tokens) - 1

    if not (
        len(pack_tokens) == len(pack_targets) == len(pack_positions) == pack_length + 1
    ):
        raise ValueError(
            "Megatron SFT packed preprocessing produced inconsistent pack lengths: "
            f"tokens={len(pack_tokens)} targets={len(pack_targets)} "
            f"positions={len(pack_positions)} expected={pack_length + 1}"
        )

    for boundary in cu_seqlens[1:-1]:
        pack_targets[boundary] = IGNORE_INDEX

    input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
    target_ids = torch.tensor(pack_targets[1:], dtype=torch.int64)
    position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

    token_mask = torch.ones(pack_length, dtype=torch.float32)
    token_mask[target_ids == pad] = 0.0
    token_mask[target_ids == IGNORE_INDEX] = 0.0

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)
    adjacent_diffs = cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]
    if context_parallel_size > 1:
        if bool((adjacent_diffs % cp_granularity != 0).any().item()):
            raise ValueError(
                "Megatron SFT packed segment lengths must be divisible by "
                f"{cp_granularity} when context_parallel_size={context_parallel_size}"
            )
    max_seqlen = int(adjacent_diffs.max().item())

    return {
        "message_log": [],
        "input_ids": input_ids,
        "target_ids": target_ids,
        "token_mask": token_mask,
        "position_ids": position_ids,
        "packed_cu_seqlens": cu_seqlens_tensor,
        "packed_max_seqlen": max_seqlen,
        "packed_context_parallel_size": int(context_parallel_size),
        "length": pack_length,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "megatron_sft_packed"),
    }
