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

"""Nemotron conversation tokenization and target-mask behavior."""

from __future__ import annotations

from typing import Any

import torch

# Unicode Private Use Area. Tokenizing prose never produces it, and text parts
# containing it are rejected, so a media placeholder cannot be forged by source
# text. Mirrors the reference's _MM_MARKER (multimodal_tokenizer.py:514).
# Defined here rather than in nemotron_visual because that module imports this
# one.
MM_MARKER = "\uE000"

# Media tags that must never appear verbatim in source text.
#
# The rule is "tags that tokenize to a single positive vocabulary id", because
# only those can forge a placeholder. Measured on the Nemotron tokenizer:
# "<image>" -> [18] and "<so_embedding>" -> [27], but "<video>" -> [1060,
# 24073, 1062].
#
# "<image>" is a real vocabulary token, so prose containing it tokenizes to
# exactly the id visual expansion emits. The row then reaches the model
# carrying a visual placeholder with no pixels behind it, and packing
# concatenates it into a sequence whose image-token count no longer matches its
# image features. "<so_embedding>" is the same hazard on the audio side:
# _expand_audio_placeholders locates sound slots by scanning for its positive
# vocabulary id. "<video>" is deliberately absent: three ordinary tokens cannot
# collide with a placeholder scan, so rejecting it would only discard rows that
# mention the HTML tag in prose.
#
# The Megatron reference guards exactly the tag that maps to a positive id and
# no others, which is the same rule stated the other way round. It asserts
# SOUND_TOKEN == "<so_embedding>" is absent from every text fragment
# (task_encoder.py:848, llava_model.py:63), but never checks "<image>",
# because images splice DEFAULT_IMAGE_TOKEN_INDEX = -200 (llava_model.py:59) --
# a negative sentinel that tokenizing prose cannot produce, so its value scans
# cannot collide. NeMo-RL splices the positive vocabulary id for images too,
# so it must reject where the reference is immune by construction.
RESERVED_MEDIA_TAGS = ("<image>", "<so_embedding>")

IGNORE_INDEX = -100

_MESSAGE_START_TOKEN_ID = 10
_MESSAGE_END_TOKEN_ID = 11
_TOOL_RESPONSE_TOKEN_ID = 16
_LINE_BREAK_TOKEN_ID = 1010
_ASSISTANT_ROLE_TOKEN_IDS = (1503, 19464)
_USER_ROLE_TOKEN_ID = 3263
_NEMOTRON6_TOKENIZER_LAYOUT = {
    (_MESSAGE_START_TOKEN_ID,): "<|im_start|>",
    (_MESSAGE_END_TOKEN_ID,): "<|im_end|>",
    (_TOOL_RESPONSE_TOKEN_ID,): "<tool_response>",
    (_LINE_BREAK_TOKEN_ID,): "\n",
    _ASSISTANT_ROLE_TOKEN_IDS: "assistant",
    (_USER_ROLE_TOKEN_ID,): "user",
}

_NEMOTRON_H_5P5_TEMPLATE = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}"""


class NoTrainableTokensError(ValueError):
    """Raised when an SFT sample has no labels that contribute to loss."""


def _text_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Nemotron messages require text content parts.")
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            raise ValueError(
                "Nemotron media must be lowered to text placeholders before "
                "conversation tokenization."
            )
        parts.append(str(part.get("text", "")))
    return "".join(parts)


def _renderable_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("Nemotron tokenization requires a non-empty conversation.")
    return [
        {**message, "role": str(message["role"]), "content": _text_content(message)}
        for message in messages
    ]


def _has_nonempty_thinking_trace(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if "<think>" not in content or "</think>" not in content:
            continue
        if content.split("<think>", 1)[1].split("</think>", 1)[0].strip():
            return True
    return False


def _native_tool_boundary_mask(messages: list[dict[str, Any]]) -> torch.Tensor:
    boundaries: list[bool] = []
    previous_role: str | None = None
    for message in messages:
        role = message["role"]
        if role == "user":
            boundaries.append(False)
        elif role == "tool" and previous_role != "tool":
            boundaries.append(True)
        previous_role = role
    return torch.tensor(boundaries, dtype=torch.bool)


def _nemotron6_assistant_indices(
    tokens: torch.Tensor,
    *,
    train_only_on_last_assistant_turn: bool,
    has_nonempty_thinking_trace: bool,
    tool_response_as_turn_boundary: bool,
    native_tool_boundaries: torch.Tensor,
) -> torch.Tensor:
    matches = torch.where(
        (tokens[:-2] == _MESSAGE_START_TOKEN_ID)
        & (tokens[1:-1] == _ASSISTANT_ROLE_TOKEN_IDS[0])
        & (tokens[2:] == _ASSISTANT_ROLE_TOKEN_IDS[1])
    )[0]
    assistant_indices = matches + 1
    if not train_only_on_last_assistant_turn:
        return assistant_indices

    use_native_boundaries = tool_response_as_turn_boundary and bool(
        native_tool_boundaries.any()
    )
    user_starts = torch.where(
        (tokens[:-1] == _MESSAGE_START_TOKEN_ID)
        & (tokens[1:] == _USER_ROLE_TOKEN_ID)
    )[0]
    if len(user_starts) == 0:
        if use_native_boundaries:
            raise ValueError("Native tool responses did not render as user boundaries.")
        return assistant_indices

    legacy_tool_responses = torch.zeros_like(user_starts, dtype=torch.bool)
    valid = user_starts + 3 < len(tokens)
    positions = user_starts[valid]
    legacy_tool_responses[valid] = (
        (tokens[positions + 2] == _LINE_BREAK_TOKEN_ID)
        & (tokens[positions + 3] == _TOOL_RESPONSE_TOKEN_ID)
    )
    if use_native_boundaries and len(native_tool_boundaries) != len(user_starts):
        raise ValueError("Tool-response roles do not match rendered user boundaries.")
    if not has_nonempty_thinking_trace and not use_native_boundaries:
        return assistant_indices

    boundary_mask = (
        ~legacy_tool_responses
        if has_nonempty_thinking_trace
        else torch.zeros_like(user_starts, dtype=torch.bool)
    )
    if use_native_boundaries:
        boundary_mask |= native_tool_boundaries
    boundary_positions = user_starts[boundary_mask]
    if len(boundary_positions) == 0:
        return assistant_indices
    return assistant_indices[assistant_indices > boundary_positions[-1]]


def _validate_nemotron6_tokenizer(tokenizer: Any) -> None:
    for token_ids, expected in _NEMOTRON6_TOKENIZER_LAYOUT.items():
        actual = tokenizer.decode(
            list(token_ids), clean_up_tokenization_spaces=False
        )
        if actual != expected:
            raise ValueError(
                f"Nemotron 6 tokenizer IDs {list(token_ids)} decode to "
                f"{actual!r}, expected {expected!r}."
            )


def validate_text_content(text: str, *, sample_key: Any) -> None:
    """Reject source text that would be mistaken for a media placeholder.

    Called on every text content part before rendering, so it sees what the
    conversation actually said rather than the placeholders the renderer
    substitutes for attached media.

    Raises:
        ValueError: The text carries the reserved marker or a literal media tag.
    """
    if MM_MARKER in text:
        raise ValueError(
            f"Nemotron sample {sample_key!r} contains the reserved multimodal "
            "marker in text content."
        )
    for tag in RESERVED_MEDIA_TAGS:
        if tag in text:
            raise ValueError(
                f"Nemotron sample {sample_key!r} contains a literal {tag!r} in "
                "text content. Media placeholders may only come from attached "
                "media, because packing cannot reconcile a placeholder that "
                "has no media behind it."
            )


def _raw_tokens_and_mask(
    messages: list[dict[str, Any]], tokenizer: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    turns = [message["content"] for message in messages]
    if any(not turn for turn in turns):
        raise ValueError("Nemotron skip-chat-template turns must be non-empty.")
    text = "".join(turns)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if (
        isinstance(token_ids, list)
        and isinstance(offsets, list)
        and len(token_ids) == len(offsets)
        and all(isinstance(offset, (tuple, list)) and len(offset) == 2 for offset in offsets)
    ):
        boundaries: list[int] = []
        offset = 0
        token_index = 0
        valid_offsets = True
        for turn in turns:
            offset += len(turn)
            while token_index < len(offsets) and int(offsets[token_index][1]) <= offset:
                if int(offsets[token_index][0]) == int(offsets[token_index][1]):
                    valid_offsets = False
                    break
                token_index += 1
            if not valid_offsets or (
                token_index < len(offsets) and int(offsets[token_index][0]) < offset
            ):
                valid_offsets = False
                break
            boundaries.append(token_index)
        valid_offsets = valid_offsets and boundaries[-1] == len(token_ids)
    else:
        valid_offsets = False
        boundaries = []

    if not valid_offsets:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        boundaries = [
            len(tokenizer.encode("".join(turns[: index + 1]), add_special_tokens=False))
            for index in range(len(turns))
        ]

    tokens = torch.tensor(token_ids, dtype=torch.long)
    mask = torch.zeros_like(tokens)
    start = 0
    for message, end in zip(messages, boundaries, strict=True):
        if message["role"] == "assistant":
            mask[start:end] = 1
        start = end
    return tokens, mask


def _encode_with_marker_splices(
    text: str, tokenizer: Any, *, image_token_id: int
) -> tuple[torch.Tensor, list[int]]:
    """Encode segment-wise, splicing the image token at each marker.

    Placeholder positions are known by construction, so a literal "<image>" --
    or even "<img><image></img>" -- written in prose is never mistaken for a
    media placeholder. Mirrors the reference's _encode_with_markers
    (multimodal_tokenizer.py:560).
    """
    segments = text.split(MM_MARKER)
    ids: list[int] = []
    positions: list[int] = []
    for index, segment in enumerate(segments):
        if segment:
            ids.extend(tokenizer.encode(segment, add_special_tokens=False))
        if index < len(segments) - 1:
            positions.append(len(ids))
            ids.append(image_token_id)
    return torch.tensor(ids, dtype=torch.long), positions


def tokenize_nemotron_conversation(
    messages: list[dict[str, Any]],
    *,
    processor: Any,
    prompt_format: str,
    skip_chat_template: bool,
    train_only_on_last_assistant_turn: bool,
    tool_response_as_turn_boundary: bool,
    assistant_turn_loss: list[bool] | None,
    complete_conversation: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Tokenize once and return the exact reference target as a binary mask."""
    tokenizer = processor.tokenizer
    rendered_messages = _renderable_messages(messages)
    if assistant_turn_loss is not None and tool_response_as_turn_boundary:
        raise ValueError(
            "Explicit assistant loss is incompatible with tool-response boundaries."
        )
    if tool_response_as_turn_boundary and not train_only_on_last_assistant_turn:
        raise ValueError(
            "Tool-response boundaries require train_only_on_last_assistant_turn."
        )
    if tool_response_as_turn_boundary and skip_chat_template:
        raise ValueError(
            "Tool-response boundaries are unsupported with skip_chat_template."
        )
    if train_only_on_last_assistant_turn and prompt_format != "nemotron6-moe":
        raise ValueError(
            "train_only_on_last_assistant_turn is supported only for nemotron6-moe."
        )
    if assistant_turn_loss is not None:
        if prompt_format != "nemotron6-moe":
            raise ValueError("Explicit assistant loss is supported only for nemotron6-moe.")
        if train_only_on_last_assistant_turn:
            raise ValueError(
                "Explicit assistant loss is incompatible with last-assistant-only loss."
            )
        if skip_chat_template:
            raise ValueError(
                "Explicit assistant loss is incompatible with skip_chat_template."
            )
        if any(type(value) is not bool for value in assistant_turn_loss):
            raise ValueError("Explicit assistant loss values must be booleans.")
        if not any(assistant_turn_loss):
            raise ValueError("Explicit assistant loss requires one selected turn.")

    if skip_chat_template:
        # This path bypasses the chat template and never splices markers. The
        # leaves that use it are text-only, so a marker here means a media
        # placeholder reached a path that cannot expand it.
        if any(
            MM_MARKER in str(message.get("content", ""))
            for message in rendered_messages
        ):
            raise ValueError(
                "Nemotron skip_chat_template does not support media "
                "placeholders; found the multimodal marker in the conversation."
            )
        placeholder_positions = []
        tokens, token_loss_mask = _raw_tokens_and_mask(rendered_messages, tokenizer)
    else:
        template = (
            _NEMOTRON_H_5P5_TEMPLATE
            if prompt_format == "nemotron-h-5p5-reasoning"
            else None
        )
        if prompt_format == "nemotron6-moe":
            _validate_nemotron6_tokenizer(tokenizer)
        elif prompt_format != "nemotron-h-5p5-reasoning":
            raise ValueError(f"Unsupported Nemotron prompt format {prompt_format!r}.")
        rendered_text = tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=template,
            truncate_history_thinking=False,
        )
        if isinstance(rendered_text, list):
            if len(rendered_text) != 1:
                raise ValueError("Nemotron chat template returned several conversations.")
            rendered_text = rendered_text[0]
        tokens, placeholder_positions = _encode_with_marker_splices(
            rendered_text,
            tokenizer,
            image_token_id=tokenizer.convert_tokens_to_ids("<image>"),
        )

        if prompt_format == "nemotron-h-5p5-reasoning":
            target = tokens.clone()
            boundaries = torch.where(tokens == 11)[0]
            if len(boundaries) < 2:
                raise ValueError("Nemotron-H conversation has incomplete turn boundaries.")
            target[: boundaries[1]] = IGNORE_INDEX
            for index in range(1, len(boundaries)):
                if index % 2 == 0:
                    target[boundaries[index] : boundaries[index + 1]] = IGNORE_INDEX
                else:
                    target[boundaries[index] : boundaries[index] + 3] = IGNORE_INDEX
            system_positions = torch.where(tokens == 10)[0]
            if len(system_positions) > 1:
                special_positions = torch.sort(
                    torch.cat(
                        [
                            torch.where(tokens == 10)[0],
                            torch.where(tokens == 11)[0],
                            torch.where(tokens == 12)[0],
                        ]
                    )
                ).values
                for system_position in system_positions[1:]:
                    next_special = special_positions[
                        special_positions > system_position
                    ]
                    if len(next_special):
                        target[system_position : next_special[0]] = IGNORE_INDEX
            token_loss_mask = (target != IGNORE_INDEX).to(dtype=torch.long)
        else:
            token_loss_mask = torch.zeros_like(tokens)
            assistant_indices = _nemotron6_assistant_indices(
                tokens,
                train_only_on_last_assistant_turn=train_only_on_last_assistant_turn,
                has_nonempty_thinking_trace=_has_nonempty_thinking_trace(
                    rendered_messages
                ),
                tool_response_as_turn_boundary=tool_response_as_turn_boundary,
                native_tool_boundaries=_native_tool_boundary_mask(rendered_messages),
            )
            if assistant_turn_loss is not None:
                if len(assistant_indices) != len(assistant_turn_loss):
                    raise ValueError(
                        "Explicit assistant loss count does not match rendered "
                        "assistant boundaries."
                    )
                assistant_indices = assistant_indices[
                    torch.tensor(assistant_turn_loss, dtype=torch.bool)
                ]
            end_indices = torch.where(tokens == _MESSAGE_END_TOKEN_ID)[0]
            for assistant_index in assistant_indices:
                lower = int(assistant_index)
                if lower + 2 >= len(tokens) or tokens[lower + 2] != _LINE_BREAK_TOKEN_ID:
                    raise ValueError("Invalid Nemotron 6 assistant start boundary.")
                following_ends = end_indices[end_indices > lower]
                if len(following_ends) == 0:
                    raise ValueError("Missing Nemotron 6 assistant end boundary.")
                upper = int(following_ends[0])
                if upper + 1 >= len(tokens) or tokens[upper + 1] != _LINE_BREAK_TOKEN_ID:
                    raise ValueError("Invalid Nemotron 6 assistant end boundary.")
                token_loss_mask[lower + 3 : upper + 1] = 1

    if len(tokens) == 0 or not bool(token_loss_mask.any()):
        target = tokens.clone()
        target[token_loss_mask == 0] = IGNORE_INDEX
        detokenized = tokenizer.decode(
            tokens.tolist(), clean_up_tokenization_spaces=False
        )
        conversation = "".join(
            f"{message.get('role')}: {message.get('content')}\n"
            for message in complete_conversation or rendered_messages
        )
        raise NoTrainableTokensError(
            f"target is empty: {target}, DETOKENIZED:\n\n{detokenized}"
            f"\n\nCONVERSATION:\n\n{conversation}"
        )
    return [
        {
            "role": "assistant",
            "content": "",
            "token_ids": tokens,
            "token_loss_mask": token_loss_mask,
            "visual_placeholder_positions": placeholder_positions,
        }
    ]


__all__ = [
    "IGNORE_INDEX",
    "MM_MARKER",
    "RESERVED_MEDIA_TAGS",
    "NoTrainableTokensError",
    "tokenize_nemotron_conversation",
    "validate_text_content",
]
