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

"""Pure transcript and alignment helpers for cross-tokenizer MOPD."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Literal

from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import AlignmentPair, TokenAligner

MOPD_ALIGNMENT_METHOD = "offset_cluster_decode_fix"
ThinkingState = Literal["open", "closed", "unknown"]


def normalize_alignment_method(method: str) -> str:
    """Validate the only public cross-tokenizer MOPD alignment method."""
    normalized = method.strip().lower()
    if normalized != MOPD_ALIGNMENT_METHOD:
        raise ValueError(
            "cross-tokenizer MOPD alignment_method must be "
            f"{MOPD_ALIGNMENT_METHOD!r}; got {method!r}"
        )
    return normalized


def _normalize_leading_instruction_messages(
    messages: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Map a leading OpenAI developer/system prefix to one system message."""
    normalized = [dict(message) for message in messages]
    prefix_length = 0
    for message in normalized:
        if message.get("role") not in {"system", "developer"}:
            break
        prefix_length += 1

    if prefix_length == 0 or not any(
        message.get("role") == "developer" for message in normalized[:prefix_length]
    ):
        return normalized, 0

    contents = [
        str(message.get("content") or "")
        for message in normalized[:prefix_length]
        if str(message.get("content") or "").strip()
    ]
    merged = dict(normalized[0])
    merged["role"] = "system"
    merged["content"] = "\n\n".join(contents)
    return [merged, *normalized[prefix_length:]], prefix_length


def _teacher_chat_template_kwargs(tokenizer: Any) -> dict[str, Any]:
    """Return compatibility kwargs that preserve prior Qwen reasoning."""
    template_kwargs: dict[str, Any] = {}
    chat_template = getattr(tokenizer, "chat_template", None)
    legacy_condition = "{%- if loop.index0 > ns.last_query_index %}"
    if (
        isinstance(chat_template, str)
        and "preserve_thinking" not in chat_template
        and legacy_condition in chat_template
    ):
        preserved_condition = (
            "{%- if (preserve_thinking is defined and preserve_thinking is true) "
            "or (loop.index0 > ns.last_query_index) %}"
        )
        template_kwargs["chat_template"] = chat_template.replace(
            legacy_condition,
            preserved_condition,
            1,
        )
    return template_kwargs


def _accepts_keyword_argument(callable_obj: Any, name: str) -> bool | None:
    """Return whether an inspectable callable accepts one keyword argument."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return None

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _render_requires_preserve_thinking(
    tokenizer: Any,
    template_kwargs: dict[str, Any],
    messages: Sequence[dict[str, str]],
) -> bool:
    """Return whether omitting ``preserve_thinking`` could drop reasoning."""
    effective_template = template_kwargs.get(
        "chat_template",
        getattr(tokenizer, "chat_template", None),
    )
    template_can_rewrite_reasoning = isinstance(effective_template, str) and (
        "preserve_thinking" in effective_template
        or (
            "last_query_index" in effective_template
            and "</think>" in effective_template
        )
    )
    sampled_reasoning_is_present = any(
        message.get("role") == "assistant"
        and any(
            marker in str(message.get("content") or "")
            for marker in ("<think>", "</think>")
        )
        for message in messages
    )
    return template_can_rewrite_reasoning or sampled_reasoning_is_present


def _apply_teacher_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> str:
    """Render a teacher transcript while preserving prior sampled reasoning."""
    teacher_messages, _ = _normalize_leading_instruction_messages(messages)
    template_kwargs = _teacher_chat_template_kwargs(tokenizer)
    renderer = tokenizer.apply_chat_template
    call_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }

    for name, value in template_kwargs.items():
        if _accepts_keyword_argument(renderer, name) is False:
            raise TypeError(
                "Teacher tokenizer cannot accept required chat-template "
                f"argument {name!r}."
            )
        call_kwargs[name] = value

    preserve_support = _accepts_keyword_argument(renderer, "preserve_thinking")
    if preserve_support is False:
        if _render_requires_preserve_thinking(
            tokenizer,
            template_kwargs,
            teacher_messages,
        ):
            raise TypeError(
                "Teacher tokenizer cannot accept required chat-template "
                "argument 'preserve_thinking'."
            )
    else:
        call_kwargs["preserve_thinking"] = True

    if enable_thinking is not None:
        enable_support = _accepts_keyword_argument(renderer, "enable_thinking")
        if enable_support is False:
            if not (add_generation_prompt and enable_thinking is True):
                raise TypeError(
                    "Teacher tokenizer cannot honor the requested "
                    "enable_thinking value."
                )
        else:
            call_kwargs["enable_thinking"] = enable_thinking

    # One call is intentional: an internal TypeError is a rendering failure,
    # not evidence that one of the semantic compatibility kwargs is invalid.
    rendered = renderer(teacher_messages, **call_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("Teacher chat template must render to text")
    return rendered


def apply_teacher_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
) -> str:
    """Render a complete teacher transcript without dropping prior reasoning."""
    return _apply_teacher_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )


def _tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer(
        text,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def classify_thinking_state_before_generated_span(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    generated_start: int,
) -> ThinkingState:
    """Classify the exact chat-template state at a generation boundary."""
    if generated_start < 0 or generated_start > len(token_ids):
        return "unknown"
    prefix_ids = [int(token_id) for token_id in token_ids[:generated_start]]
    try:
        open_ids = _tokenizer_ids(
            tokenizer,
            "<|im_start|>assistant\n<think>\n",
        )
        closed_id_variants = [
            _tokenizer_ids(
                tokenizer,
                "<|im_start|>assistant\n<think></think>",
            ),
            _tokenizer_ids(
                tokenizer,
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ),
        ]
    except (KeyError, TypeError, ValueError, RuntimeError):
        return "unknown"
    if open_ids and prefix_ids[-len(open_ids) :] == open_ids:
        return "open"
    for closed_ids in closed_id_variants:
        if closed_ids and prefix_ids[-len(closed_ids) :] == closed_ids:
            return "closed"
    return "unknown"


def generated_think_tool_marker_error(
    text: str,
    *,
    initial_state: ThinkingState,
) -> str | None:
    """Return the structural error in one sampled assistant span, if any."""
    if initial_state == "unknown":
        return "unknown_thinking_state"

    think_open_count = text.count("<think>")
    think_close_count = text.count("</think>")
    if think_open_count:
        return "generated_think_open"
    if initial_state == "open" and think_close_count > 1:
        return "multiple_think_close"
    if initial_state == "closed" and think_close_count:
        return "unexpected_think_close"

    if "<tool_response>" in text or "</tool_response>" in text:
        return "generated_tool_response"

    tool_depth = 0
    cursor = 0
    while cursor < len(text):
        open_position = text.find("<tool_call>", cursor)
        close_position = text.find("</tool_call>", cursor)
        if open_position < 0 and close_position < 0:
            break
        if open_position >= 0 and (
            close_position < 0 or open_position < close_position
        ):
            tool_depth += 1
            if tool_depth > 1:
                return "nested_tool_call"
            cursor = open_position + len("<tool_call>")
        else:
            tool_depth -= 1
            if tool_depth < 0:
                return "unmatched_tool_call_close"
            cursor = close_position + len("</tool_call>")
    if tool_depth:
        return "unclosed_tool_call"
    return None


def validate_generated_think_tool_markers(
    text: str,
    *,
    initial_state: ThinkingState,
) -> bool:
    """Return whether sampled thinking/tool markers are structurally safe."""
    return generated_think_tool_marker_error(text, initial_state=initial_state) is None


def render_teacher_open_thinking_prefix(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
) -> str:
    """Render history followed by the teacher's native open-thinking prompt."""
    rendered = _apply_teacher_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    rendered_ids = _tokenizer_ids(tokenizer, rendered)
    state = classify_thinking_state_before_generated_span(
        tokenizer,
        rendered_ids,
        generated_start=len(rendered_ids),
    )
    if state != "open":
        raise ValueError(
            "Teacher chat template did not produce a provable open-thinking "
            "generation prefix."
        )
    return rendered


def qwen_chat_boundaries_present(
    tokenizer: Any,
    token_ids: Sequence[int],
) -> bool:
    """Return whether a stream contains a provable Qwen message boundary."""
    try:
        im_start_ids = _tokenizer_ids(tokenizer, "<|im_start|>")
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return len(im_start_ids) == 1 and im_start_ids[0] in token_ids


def parse_qwen_chat_token_stream(
    tokenizer: Any,
    token_ids: Sequence[int],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]] | None:
    """Recover structural Qwen messages from an exact rendered token stream.

    NeMo Gym prompt deltas can contain several rendered roles. This parser is
    deliberately strict and returns ``None`` whenever boundaries are not
    provable from the exact token stream.
    """
    stream = [int(token_id) for token_id in token_ids]
    if not stream:
        return None
    try:
        im_start_ids = _tokenizer_ids(tokenizer, "<|im_start|>")
        im_end_ids = _tokenizer_ids(tokenizer, "<|im_end|>")
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    if len(im_start_ids) != 1 or len(im_end_ids) != 1:
        return None
    im_start_id = im_start_ids[0]
    im_end_id = im_end_ids[0]
    if im_start_id not in stream:
        return None

    messages: list[dict[str, str]] = []
    bounds: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(stream):
        try:
            message_start = stream.index(im_start_id, cursor)
        except ValueError:
            try:
                trailing_text = tokenizer.decode(
                    stream[cursor:], skip_special_tokens=True
                )
            except (TypeError, ValueError, RuntimeError):
                return None
            if trailing_text.strip():
                return None
            break

        try:
            leading_text = tokenizer.decode(
                stream[cursor:message_start], skip_special_tokens=True
            )
        except (TypeError, ValueError, RuntimeError):
            return None
        if leading_text.strip():
            return None

        header_start = message_start + 1
        try:
            message_end_token = stream.index(im_end_id, header_start)
        except ValueError:
            message_end_token = len(stream)
        try:
            nested_start = stream.index(im_start_id, header_start)
        except ValueError:
            nested_start = len(stream)
        if nested_start < message_end_token:
            return None

        try:
            message_text = tokenizer.decode(
                stream[header_start:message_end_token],
                skip_special_tokens=False,
            )
        except (TypeError, ValueError, RuntimeError):
            return None
        if "\n" not in message_text:
            return None
        role, content = message_text.split("\n", 1)
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            return None
        if message_end_token == len(stream) and role != "assistant":
            return None

        stripped_content = content.strip()
        if (
            role in {"user", "tool"}
            and stripped_content.startswith("<tool_response>")
            and stripped_content.endswith("</tool_response>")
        ):
            content = stripped_content

        message_full_end = min(message_end_token + 1, len(stream))
        messages.append({"role": role, "content": content})
        bounds.append(
            {
                "role": role,
                "full_start": message_start,
                "content_start": header_start,
                "content_end": message_end_token,
                "full_end": message_full_end,
            }
        )
        cursor = message_full_end

    if not messages:
        return None

    normalized_messages, instruction_prefix_length = (
        _normalize_leading_instruction_messages(messages)
    )
    if instruction_prefix_length:
        merged_bound = dict(bounds[0])
        merged_bound["role"] = "system"
        merged_bound["content_end"] = bounds[instruction_prefix_length - 1][
            "content_end"
        ]
        merged_bound["full_end"] = bounds[instruction_prefix_length - 1]["full_end"]
        messages = normalized_messages
        bounds = [merged_bound, *bounds[instruction_prefix_length:]]
    return messages, bounds


def map_generated_turns_to_parsed_messages(
    message_log: Sequence[dict[str, Any]],
    parsed_bounds: Sequence[dict[str, Any]],
    *,
    student_length: int,
) -> dict[int, int] | None:
    """Map each flattened generated span to one recovered assistant message."""
    mapping: dict[int, int] = {}
    used_parsed_indices: set[int] = set()
    student_cursor = 0
    for message_index, message in enumerate(message_log):
        message_token_ids = message.get("token_ids")
        if message_token_ids is None:
            continue
        message_length = len(message_token_ids)
        message_start = student_cursor
        message_end = message_start + message_length
        student_cursor = message_end
        if not (
            message.get("role") == "assistant"
            and "generation_logprobs" in message
            and message_length > 0
            and message_end <= student_length
        ):
            continue

        candidates = [
            parsed_index
            for parsed_index, bound in enumerate(parsed_bounds)
            if bound["role"] == "assistant"
            and parsed_index not in used_parsed_indices
            and message_start >= int(bound["content_start"])
            and message_end <= int(bound["full_end"])
        ]
        if len(candidates) != 1:
            return None
        parsed_index = candidates[0]
        mapping[message_index] = parsed_index
        used_parsed_indices.add(parsed_index)
    return mapping


def sampled_im_end_token_id(tokenizer: Any) -> int | None:
    """Return the EOS id only when EOS is exactly Qwen's assistant boundary."""
    if getattr(tokenizer, "eos_token", None) != "<|im_end|>":
        return None
    token_id = getattr(tokenizer, "eos_token_id", None)
    return int(token_id) if token_id is not None else None


def split_sampled_assistant_eot(
    tokenizer: Any,
    token_ids: Sequence[int],
) -> tuple[list[int], int | None]:
    """Remove and report an explicitly sampled terminal ``<|im_end|>``."""
    content_ids = [int(token_id) for token_id in token_ids]
    im_end_id = sampled_im_end_token_id(tokenizer)
    if not content_ids or im_end_id is None or content_ids[-1] != im_end_id:
        return content_ids, None
    content_ids.pop()
    return content_ids, len(content_ids)


def nearest_token_position(
    token_ids: Sequence[int],
    token_id: int | None,
    *,
    expected_position: int,
    minimum_position: int,
    radius: int = 8,
) -> int | None:
    """Find a template boundary near its standalone-tokenized position."""
    if token_id is None:
        return None
    lower = max(minimum_position, expected_position - radius, 0)
    upper = min(len(token_ids), expected_position + radius + 1)
    candidates = [
        position for position in range(lower, upper) if token_ids[position] == token_id
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda position: (abs(position - expected_position), position),
    )


def teacher_message_search_floor(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    message_index: int,
    teacher_text: str,
    teacher_ids: Sequence[int],
) -> int | None:
    """Return the proven teacher-token boundary before one rendered message."""
    if message_index <= 0:
        return 0
    try:
        prefix_text = apply_teacher_chat_template(
            tokenizer,
            messages[:message_index],
        )
        prefix_ids = _tokenizer_ids(tokenizer, prefix_text)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    if not teacher_text.startswith(prefix_text):
        return None
    if list(teacher_ids[: len(prefix_ids)]) != prefix_ids:
        return None
    return len(prefix_ids)


def compute_offsets_manual(
    tokenizer: PreTrainedTokenizerBase,
    ids: Sequence[int],
    reference_text: str,
    *,
    skip_special_tokens: bool = False,
    max_group_size: int = 16,
) -> list[tuple[int, int]] | None:
    """Compute strict offsets for an existing, possibly noncanonical stream."""
    if not ids or reference_text is None:
        return None
    if max_group_size < 1:
        raise ValueError("max_group_size must be at least 1")

    token_ids = [int(token_id) for token_id in ids]
    special_ids = {
        int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])
    }
    offsets: list[tuple[int, int]] = []
    cursor = 0
    token_index = 0
    while token_index < len(token_ids):
        token_id = token_ids[token_index]
        if skip_special_tokens and token_id in special_ids:
            offsets.append((cursor, cursor))
            token_index += 1
            continue

        matched_group_size = 0
        matched_piece = ""
        remaining = len(token_ids) - token_index
        for group_size in range(1, min(max_group_size, remaining) + 1):
            group = token_ids[token_index : token_index + group_size]
            try:
                piece = tokenizer.decode(
                    group,
                    skip_special_tokens=skip_special_tokens,
                )
            except (TypeError, ValueError, RuntimeError):
                return None
            if piece and reference_text.startswith(piece, cursor):
                matched_group_size = group_size
                matched_piece = piece
                break
        if matched_group_size == 0:
            return None

        span = (cursor, cursor + len(matched_piece))
        offsets.extend([span] * matched_group_size)
        cursor = span[1]
        token_index += matched_group_size

    if cursor != len(reference_text):
        return None
    return offsets


def teacher_template_trims_assistant_content(
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Return whether the teacher template trims rendered assistant content."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return False
    compact_template = "".join(template.split())
    return "setcontent=render_content(message.content,true)|trim" in compact_template


def qwen3_assistant_content_transform(
    content: str,
    *,
    trim_outer_whitespace: bool = False,
) -> tuple[str, bool]:
    """Apply the teacher template's assistant rewrite around ``</think>``."""
    if trim_outer_whitespace:
        content = content.strip()
    if "</think>" not in content:
        return content, False
    reasoning = content.split("</think>")[0].rstrip("\n")
    if "<think>" in reasoning:
        reasoning = reasoning.split("<think>")[-1]
    reasoning = reasoning.lstrip("\n")
    if trim_outer_whitespace:
        reasoning = reasoning.strip()
    body = content.split("</think>")[-1].lstrip("\n")
    return reasoning + "\n</think>\n\n" + body, True


def qwen3_leading_think_prefix_len(content: str) -> int:
    """Return the generated leading ``<think>`` wrapper length, if present."""
    if not content.startswith("<think>"):
        return 0
    position = len("<think>")
    while position < len(content) and content[position] == "\n":
        position += 1
    return position


def trim_outer_whitespace_tokens_for_alignment(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
    text: str,
) -> tuple[list[int], int, int, str] | None:
    """Trim outer whitespace only when it falls on clean token boundaries."""
    prepared = trim_outer_whitespace_with_offsets_for_alignment(
        tokenizer,
        token_ids,
        text,
    )
    if prepared is None:
        return None
    trimmed_ids, dropped_prefix, dropped_suffix, stripped_text, _ = prepared
    try:
        if tokenizer.decode(trimmed_ids, skip_special_tokens=True) != stripped_text:
            return None
    except (TypeError, ValueError, RuntimeError):
        return None
    return trimmed_ids, dropped_prefix, dropped_suffix, stripped_text


def trim_outer_whitespace_with_offsets_for_alignment(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
    text: str,
) -> tuple[list[int], int, int, str, list[tuple[int, int]]] | None:
    r"""Project a generated stream into a template-trimmed character frame."""
    token_ids = list(token_ids)
    stripped_text = text.strip()
    if not stripped_text:
        return None

    left_char = len(text) - len(text.lstrip())
    right_char = len(text.rstrip())
    try:
        encoding = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        canonical_ids = list(encoding["input_ids"])
        canonical_offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    except (NotImplementedError, TypeError, ValueError, RuntimeError):
        # Fastokens intentionally does not expose character offsets. Ordinary
        # tokenization still lets us verify the generated IDs before falling
        # back to the same strict decode-based reconstruction used elsewhere.
        try:
            encoding = tokenizer(
                text,
                return_tensors=None,
                add_special_tokens=False,
            )
            canonical_ids = list(encoding["input_ids"])
        except (TypeError, ValueError, RuntimeError):
            return None
        canonical_offsets = []

    if canonical_ids == token_ids and len(canonical_offsets) == len(token_ids):
        offsets = canonical_offsets
    else:
        try:
            if tokenizer.decode(token_ids, skip_special_tokens=True) != text:
                return None
        except (TypeError, ValueError, RuntimeError):
            return None
        offsets = compute_offsets_manual(
            tokenizer,
            token_ids,
            text,
            skip_special_tokens=True,
        )
        if offsets is None or len(offsets) != len(token_ids):
            return None

    dropped_prefix = 0
    for _start, end in offsets:
        if end <= left_char:
            dropped_prefix += 1
            continue
        break

    dropped_suffix = 0
    for start, _end in reversed(offsets):
        if start >= right_char:
            dropped_suffix += 1
            continue
        break

    kept_end = len(token_ids) - dropped_suffix if dropped_suffix else len(token_ids)
    trimmed_ids = token_ids[dropped_prefix:kept_end]
    trimmed_offsets = offsets[dropped_prefix:kept_end]
    projected_offsets = [
        (
            max(left_char, min(start, right_char)) - left_char,
            max(left_char, min(end, right_char)) - left_char,
        )
        for start, end in trimmed_offsets
    ]
    if len(projected_offsets) != len(trimmed_ids) or any(
        end <= start for start, end in projected_offsets
    ):
        return None
    try:
        retained_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    except (TypeError, ValueError, RuntimeError):
        return None
    if retained_text.strip() != stripped_text:
        return None
    return (
        trimmed_ids,
        dropped_prefix,
        dropped_suffix,
        stripped_text,
        projected_offsets,
    )


def trim_leading_text_tokens_for_alignment(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
    text: str,
    trim_chars: int,
) -> tuple[list[int], int, str]:
    """Trim a leading decoded-text prefix from token IDs when token-clean."""
    token_ids = list(token_ids)
    if trim_chars <= 0 or trim_chars >= len(text):
        return token_ids, 0, text

    try:
        try:
            encoding = tokenizer(
                text,
                return_tensors=None,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = list(encoding["input_ids"])
            offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
        except (NotImplementedError, TypeError, ValueError):
            encoding = tokenizer(text, return_tensors=None, add_special_tokens=False)
            ids = list(encoding["input_ids"])
            offsets = compute_offsets_manual(tokenizer, ids, text)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return token_ids, 0, text

    if not ids or offsets is None:
        return token_ids, 0, text

    dropped = 0
    for start, end in offsets:
        if end <= trim_chars:
            dropped += 1
            continue
        if start < trim_chars:
            return token_ids, 0, text
        break

    if dropped <= 0 or token_ids[:dropped] != ids[:dropped]:
        return token_ids, 0, text
    return token_ids[dropped:], dropped, text[trim_chars:]


def build_char_mapping(original: str, transformed: str) -> list[int] | None:
    """Map original character boundaries across template whitespace changes."""
    mapping: list[int] = []
    original_index = transformed_index = 0
    while original_index < len(original):
        if (
            transformed_index < len(transformed)
            and original[original_index] == transformed[transformed_index]
        ):
            mapping.append(transformed_index)
            original_index += 1
            transformed_index += 1
        elif (
            transformed_index < len(transformed)
            and transformed[transformed_index].isspace()
        ):
            transformed_index += 1
        elif original[original_index].isspace():
            mapping.append(transformed_index)
            original_index += 1
        else:
            break
    else:
        mapping.append(transformed_index)
        if all(char.isspace() for char in transformed[transformed_index:]):
            return mapping

    marker = "</think>"
    first_close = original.find(marker)
    last_close = original.rfind(marker)
    expected_transform, had_think = qwen3_assistant_content_transform(original)
    if (
        not had_think
        or first_close < 0
        or last_close == first_close
        or expected_transform != transformed
    ):
        return None

    reasoning_start = 0
    reasoning_end = first_close
    while reasoning_end > reasoning_start and original[reasoning_end - 1] == "\n":
        reasoning_end -= 1
    open_marker = original.rfind("<think>", reasoning_start, reasoning_end)
    if open_marker >= 0:
        reasoning_start = open_marker + len("<think>")
    while reasoning_start < reasoning_end and original[reasoning_start] == "\n":
        reasoning_start += 1

    body_start = last_close + len(marker)
    while body_start < len(original) and original[body_start] == "\n":
        body_start += 1

    reasoning = original[reasoning_start:reasoning_end]
    body = original[body_start:]
    transformed_close_start = len(reasoning) + 1
    transformed_body_start = transformed_close_start + len(marker) + 2
    retained_segments = [
        (reasoning_start, reasoning_end, 0),
        (first_close, first_close + len(marker), transformed_close_start),
        (body_start, len(original), transformed_body_start),
    ]
    if any(
        original[source_start:source_end]
        != transformed[target_start : target_start + source_end - source_start]
        for source_start, source_end, target_start in retained_segments
    ):
        return None

    retained_char_positions: dict[int, int] = {}
    for source_start, source_end, target_start in retained_segments:
        for source_position in range(source_start, source_end):
            retained_char_positions[source_position] = (
                target_start + source_position - source_start
            )

    mapping = [0] * (len(original) + 1)
    next_retained_position = len(transformed)
    for source_position in range(len(original) - 1, -1, -1):
        if source_position in retained_char_positions:
            next_retained_position = retained_char_positions[source_position]
        mapping[source_position] = next_retained_position
    mapping[-1] = len(transformed)
    return mapping


def proven_template_only_teacher_token_indices(
    original: str,
    transformed: str,
    teacher_offsets: Sequence[tuple[int, int]],
) -> tuple[set[int], set[int]] | None:
    r"""Prove teacher tokens containing only template-inserted whitespace."""
    if original.count("</think>") != 1:
        return None

    expected, had_think = qwen3_assistant_content_transform(original)
    if not had_think or expected != transformed:
        return None

    mapping = build_char_mapping(original, transformed)
    if mapping is None or len(mapping) != len(original) + 1:
        return None

    sampled_positions: set[int] = set()
    for source_position, source_char in enumerate(original):
        target_position = mapping[source_position]
        if (
            0 <= target_position < len(transformed)
            and transformed[target_position] == source_char
        ):
            sampled_positions.add(target_position)
        elif not source_char.isspace():
            return None

    inserted_positions = {
        position
        for position, char in enumerate(transformed)
        if position not in sampled_positions and char.isspace()
    }
    if any(
        position not in sampled_positions and position not in inserted_positions
        for position in range(len(transformed))
    ):
        return None

    template_only: set[int] = set()
    mixed: set[int] = set()
    for token_index, offset in enumerate(teacher_offsets):
        try:
            if len(offset) != 2:
                return None
            start, end = int(offset[0]), int(offset[1])
        except (TypeError, ValueError, IndexError):
            return None
        if start < 0 or end < start or end > len(transformed):
            return None
        if end == start:
            continue

        covered_positions = range(start, end)
        has_inserted = any(
            position in inserted_positions for position in covered_positions
        )
        if not has_inserted:
            continue
        if all(position in inserted_positions for position in covered_positions):
            template_only.add(token_index)
        else:
            mixed.add(token_index)

    return template_only, mixed


def teacher_prefix_before_span(
    tokenizer: PreTrainedTokenizerBase,
    teacher_ids: Sequence[int],
    teacher_start: int,
    student_text: str,
) -> tuple[bool, str]:
    """Detect a teacher-only prefix immediately before an aligned span."""
    if teacher_start <= 0 or student_text.startswith("\n"):
        return False, ""

    prefix_ids = list(teacher_ids[max(0, teacher_start - 8) : teacher_start])
    if not prefix_ids:
        return False, ""
    try:
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    except (TypeError, ValueError, RuntimeError):
        return False, ""

    if prefix_text.endswith("\n") or prefix_text.endswith("<think>"):
        return True, prefix_text[-80:]
    return False, prefix_text[-80:]


def _pair_fields(
    pair: AlignmentPair | tuple[int, int, int, int, bool],
) -> tuple[int, int, int, int, bool]:
    if isinstance(pair, AlignmentPair):
        return pair.s_start, pair.s_end, pair.t_start, pair.t_end, pair.is_correct
    return pair


def first_teacher_prefix_chunk_to_mask(
    chunks: Sequence[AlignmentPair | tuple[int, int, int, int, bool]],
    teacher_prefix_before_span: bool = False,
) -> int | None:
    """Return the first valid chunk index affected by a teacher-only prefix."""
    for index, pair in enumerate(chunks):
        student_start, student_end, teacher_start, teacher_end, is_correct = (
            _pair_fields(pair)
        )
        if (
            not is_correct
            or student_end <= student_start
            or teacher_end <= teacher_start
        ):
            continue
        if teacher_start > 0 or teacher_prefix_before_span:
            return index
        return None
    return None


def align_token_ids(
    student_ids: Sequence[int],
    teacher_ids: Sequence[int],
    *,
    aligner: TokenAligner,
    method: str,
    student_offsets: Sequence[tuple[int, int]],
    teacher_offsets: Sequence[tuple[int, int]],
) -> list[AlignmentPair]:
    """Align one exact sampled span using a cached #3286 ``TokenAligner``."""
    normalize_alignment_method(method)
    if len(student_ids) != len(student_offsets):
        raise ValueError("student token/offset lengths differ")
    if len(teacher_ids) != len(teacher_offsets):
        raise ValueError("teacher token/offset lengths differ")
    return aligner.align_one_offset_pairs(
        [int(token_id) for token_id in student_ids],
        [int(token_id) for token_id in teacher_ids],
        [(int(start), int(end)) for start, end in student_offsets],
        [(int(start), int(end)) for start, end in teacher_offsets],
    )
