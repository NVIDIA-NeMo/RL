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
"""Typed native ChatML rendering and assistant supervision for distillation.

The semantic parser supports the Qwen/Nano ChatML layouts. It deliberately
rejects changed or missing source regions instead of guessing a loss mask.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Tuple

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import (
    NativeAlignmentPart,
    NativeAlignmentRegions,
)
from nemo_rl.data.chat_templates import find_rendered_message_content_span
from nemo_rl.data.chat_utils import normalize_message_loss_mask

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_IM_START_ASSISTANT = "<|im_start|>assistant\n"
_IM_END = "<|im_end|>"


@dataclass(kw_only=True)
class RenderedChatDocument:
    """One unpadded native rendering with source-bound loss/alignment metadata."""

    rendered_text: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    assistant_mask: list[int]
    assistant_spans: list[tuple[int, int]]
    eot_indices: list[int]
    source_turn_indices: list[int]
    alignment_regions: list[NativeAlignmentRegions]
    answer_parts: list[list[NativeAlignmentPart] | None]


def _render_chat_text(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    preserve_thinking: bool,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Render training context without dropping required template controls."""
    kwargs: dict[str, Any] = {}
    if preserve_thinking:
        kwargs.update(
            enable_thinking=True,
            preserve_thinking=True,
            truncate_history_thinking=False,
        )
        # get_tokenizer binds explicitly configured template controls with a
        # partial. They take precedence over native defaults; validation below
        # will reject a template that consequently loses requested reasoning.
        if isinstance(tokenizer.apply_chat_template, partial):
            configured = tokenizer.apply_chat_template.keywords
            if (
                "preserve_thinking" in configured
                or "truncate_history_thinking" in configured
            ):
                kwargs.pop("preserve_thinking")
                kwargs.pop("truncate_history_thinking")
            for name in (
                "enable_thinking",
                "preserve_thinking",
                "truncate_history_thinking",
            ):
                if name in configured:
                    kwargs[name] = configured[name]
    if tools is not None:
        kwargs["tools"] = tools
    # Unsupported kwargs are errors: a compatibility retry could silently
    # discard tools or the history-retention contract.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **kwargs
    )
    if not isinstance(text, str):
        raise TypeError("Chat template must render a string")
    return text


def _prepare_native_thinking_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy source messages and expose leading reasoning separately from prose."""
    prepared = deepcopy(messages)
    for message in prepared:
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise TypeError("Native chat requires string assistant content")
        content = content.strip()
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise TypeError("reasoning_content must be a string")
        match = _THINK_PATTERN.match(content)
        if reasoning and reasoning.strip():
            message["reasoning_content"] = reasoning.strip()
            message["content"] = content
        elif match is not None:
            message["reasoning_content"] = match.group()[7:-8].strip()
            message["content"] = content[match.end() :].strip()
        else:
            message.pop("reasoning_content", None)
            message["content"] = content
    return prepared


def _turn_render(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    turn: int,
    text: str,
    *,
    preserve_thinking: bool,
    tools: list[dict[str, Any]] | None,
) -> str:
    prefix = _render_chat_text(
        tokenizer,
        messages[: turn + 1],
        preserve_thinking=preserve_thinking,
        tools=tools,
    )
    if not text.startswith(prefix):
        raise ValueError(
            f"turn {turn}: template changed historical content or reasoning; "
            "install a compatible history-preserving template"
        )
    return prefix


def _ordinary_content_span(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    turn: int,
    prefix: str,
    *,
    preserve_thinking: bool,
    tools: list[dict[str, Any]] | None,
) -> tuple[int, int, str] | None:
    """Bind content inside its template envelope, never role headers/suffixes."""
    marker = "__nemo_rl_content_span_probe__"
    while marker in prefix:
        marker += "_"
    probe_messages = deepcopy(messages[: turn + 1])
    probe_messages[-1]["content"] = marker
    probe = _render_chat_text(
        tokenizer, probe_messages, preserve_thinking=preserve_thinking, tools=tools
    )
    if probe.count(marker) != 1:
        raise ValueError(
            f"turn {turn}: missing canonical content span; template transformed the content probe"
        )
    before, after = probe.split(marker)
    if not prefix.startswith(before) or not prefix.endswith(after):
        raise ValueError(
            f"turn {turn}: missing canonical content span; template changed the content envelope"
        )
    content_start = len(before)
    content_end = len(prefix) - len(after)
    source = messages[turn].get("content") or ""
    if prefix[content_start:content_end].strip() != source.strip():
        raise ValueError(
            f"turn {turn}: missing canonical content span; rendered content differs from source"
        )
    return find_rendered_message_content_span(
        prefix[:content_end], source, content_start
    )


def _eot_index(offsets: list[tuple[int, int]], text: str, position: int) -> int:
    for index, (start, end) in enumerate(offsets):
        if start <= position < end and not text[start:position].strip():
            return index
    return -1


def _tokenize_chat_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    messages: list[dict[str, Any]],
    message_loss_mask: list[int],
    max_len: int,
    *,
    include_thinking_in_loss: bool,
    native_semantic_messages: list[dict[str, Any]] | None,
    skip_overlength: bool,
    tools: list[dict[str, Any]] | None = None,
) -> RenderedChatDocument:
    """Tokenize one rendered document and bind supervision to logical turns.

    Production callers reject overflow. The explicit lower-level truncation
    option remains available for inspecting retained token boundaries.
    """
    if include_thinking_in_loss and native_semantic_messages is None:
        for turn, message in enumerate(messages):
            if message.get("role") == "assistant" and message.get("reasoning_content"):
                raise ValueError(
                    f"turn {turn}: supervising separate reasoning_content requires "
                    "collator.native_thinking_alignment=true with a supported "
                    "Qwen/Nano ChatML template; ordinary chat supports inline "
                    "<think> reasoning in message.content"
                )
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len + int(skip_overlength),
    )
    ids = list(encoded["input_ids"])
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if skip_overlength and len(ids) > max_len:
        raise ValueError(
            f"Chat document exceeds context length {max_len}; overlength rows are rejected"
        )
    if not ids:
        raise ValueError("Chat template produced an empty tokenization")
    doc = RenderedChatDocument(
        rendered_text=text,
        input_ids=ids,
        offsets=offsets,
        assistant_mask=[0] * len(ids),
        assistant_spans=[],
        eot_indices=[],
        source_turn_indices=[],
        alignment_regions=[],
        answer_parts=[],
    )
    preserve = include_thinking_in_loss or native_semantic_messages is not None
    native_cursor: int = 0
    for turn, message in enumerate(messages):
        selected = bool(message_loss_mask[turn])
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise TypeError(f"turn {turn}: chat requires string content")
        if native_semantic_messages is not None:
            if message.get("role") != "assistant":
                continue
            marker_start = text.find(_IM_START_ASSISTANT, native_cursor)
            eot_start = (
                text.find(_IM_END, marker_start + len(_IM_START_ASSISTANT))
                if marker_start >= 0
                else -1
            )
            if marker_start < 0 or eot_start < marker_start:
                raise ValueError(
                    f"turn {turn}: unsupported native layout; expected Qwen/Nano ChatML markers"
                )
            content_start = marker_start + len(_IM_START_ASSISTANT)
            native_cursor = eot_start + len(_IM_END)
            regions: dict[str, tuple[int, int]] = {}
            reasoning = message.get("reasoning_content")
            answer_cursor = content_start
            if reasoning:
                open_start = text.find("<think>", content_start, eot_start)
                if open_start < 0:
                    raise ValueError(
                        f"turn {turn}: missing requested reasoning scaffold"
                    )
                reasoning_start = text.find(reasoning, open_start + 7, eot_start)
                if reasoning_start < 0:
                    raise ValueError(
                        f"turn {turn}: missing requested reasoning content"
                    )
                reasoning_end = reasoning_start + len(reasoning)
                close_start = text.find("</think>", reasoning_end, eot_start)
                if close_start < 0:
                    raise ValueError(f"turn {turn}: missing reasoning closing marker")
                regions["reasoning"] = (reasoning_start, reasoning_end)
                regions["close"] = (close_start, close_start + 8)
                answer_cursor = close_start + 8
            if content:
                answer_start = text.find(content, answer_cursor, eot_start)
                if answer_start < 0:
                    raise ValueError(f"turn {turn}: missing native answer content")
                regions["answer"] = (answer_start, answer_start + len(content))
                answer_cursor = answer_start + len(content)
            calls = message.get("tool_calls")
            if calls:
                first_call = _render_native_tool_call(tokenizer, calls[0])
                tool_start = text.find(first_call, answer_cursor, eot_start)
                if tool_start < 0:
                    raise ValueError(f"turn {turn}: missing native tool payload")
                last_call = _render_native_tool_call(tokenizer, calls[-1])
                tool_end = text.rfind(last_call, tool_start, eot_start) + len(last_call)
                regions["answer"] = (
                    regions.get("answer", (tool_start, tool_end))[0],
                    tool_end,
                )
            if not selected:
                continue
            if not regions:
                raise ValueError(
                    f"turn {turn}: selected assistant has empty supervision (whitespace-only content is unsupported)"
                )
            typed_regions = NativeAlignmentRegions(**regions)
            doc.alignment_regions.append(typed_regions)
            doc.assistant_spans.append((content_start, eot_start))
            doc.eot_indices.append(_eot_index(offsets, text, eot_start))
            doc.source_turn_indices.append(turn)
            for token_i, (start, end) in enumerate(offsets):
                if end > start and any(
                    a <= start and end <= b for a, b in regions.values()
                ):
                    doc.assistant_mask[token_i] = 1
        else:
            # Prefix renders anchor the lookup to this logical turn, including
            # user/context turns whose text may recur in an assistant answer.
            prefix = _turn_render(
                tokenizer, messages, turn, text, preserve_thinking=preserve, tools=tools
            )
            span = _ordinary_content_span(
                tokenizer,
                messages,
                turn,
                prefix,
                preserve_thinking=preserve,
                tools=tools,
            )
            if span is None:
                if content.strip() or selected:
                    raise ValueError(
                        f"turn {turn}: missing canonical content span or empty selected assistant"
                    )
                continue
            start, end, canonical = span
            if not selected:
                continue
            doc.assistant_spans.append((start, end))
            doc.source_turn_indices.append(turn)
            suffix = prefix[end:]
            terminator = suffix.lstrip()
            eot_position = end + len(suffix) - len(terminator)
            doc.eot_indices.append(
                _eot_index(offsets, text, eot_position) if terminator else -1
            )
            thinking = [
                (start + m.start(), start + m.end())
                for m in _THINK_PATTERN.finditer(canonical)
            ]
            openings = [
                (start + m.start(), start + m.start() + 7)
                for m in _THINK_PATTERN.finditer(canonical)
            ]
            for token_i, (a, b) in enumerate(offsets):
                if b <= a or b <= start or a >= end:
                    continue
                if text[a:start].strip() or text[end:b].strip():
                    continue
                in_think = any(x <= a and b <= y for x, y in thinking)
                in_open = any(x < b and a < y for x, y in openings)
                if (include_thinking_in_loss and not in_open) or not in_think:
                    doc.assistant_mask[token_i] = 1
    for index in doc.eot_indices:
        if index >= 0:
            doc.assistant_mask[index] = 1
    if not any(doc.assistant_mask) and skip_overlength:
        raise ValueError("Chat document has no selected assistant supervision")
    if native_semantic_messages is not None:
        doc.answer_parts = _native_answer_parts_for_document(
            tokenizer, messages, message_loss_mask, doc
        )
        for turn_regions, parts in zip(doc.alignment_regions, doc.answer_parts):
            if parts is None:
                continue
            assert turn_regions.answer is not None
            answer_start, answer_end = turn_regions.answer
            for token_i, (start, end) in enumerate(offsets):
                if answer_start <= start < end <= answer_end:
                    # Keep punctuation sharing a token with a separator, but
                    # exclude tokens containing only template separators.
                    doc.assistant_mask[token_i] = int(
                        any(
                            a < end and start < b
                            for a, b in (part.span for part in parts)
                        )
                    )
    return doc


def _reject_embedded_chatml_markers(value: Any) -> None:
    """Reject ambiguous literal turn delimiters in the supported native parser."""
    if isinstance(value, str):
        if "<|im_start|>" in value or _IM_END in value:
            raise ValueError(
                "Native source contains an embedded ChatML turn delimiter; this layout is unsupported"
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_embedded_chatml_markers(key)
            _reject_embedded_chatml_markers(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_embedded_chatml_markers(item)


def _render_and_tokenize_chat(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    max_len: int,
    *,
    tools: list[dict[str, Any]] | None = None,
    message_loss_mask: list[int] | None = None,
    include_thinking_in_loss: bool,
    native_thinking_alignment: bool,
    skip_overlength: bool,
) -> RenderedChatDocument:
    selected = normalize_message_loss_mask(messages, message_loss_mask)
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("tools must be a list of function schemas")
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not isinstance(
                function.get("name"), str
            ):
                raise ValueError(
                    "Each tool schema requires a function with a string name"
                )
            if "parameters" in function and not isinstance(
                function["parameters"], dict
            ):
                raise ValueError("Tool function parameters must be a schema object")
    if native_thinking_alignment:
        _reject_embedded_chatml_markers(messages)
        _reject_embedded_chatml_markers(tools)
        messages = _prepare_native_thinking_messages(messages)
    text = _render_chat_text(
        tokenizer,
        messages,
        preserve_thinking=include_thinking_in_loss or native_thinking_alignment,
        tools=tools,
    )
    return _tokenize_chat_text(
        tokenizer,
        text,
        messages,
        selected,
        max_len,
        include_thinking_in_loss=include_thinking_in_loss,
        native_semantic_messages=messages if native_thinking_alignment else None,
        skip_overlength=skip_overlength,
        tools=tools,
    )


def _render_native_tool_call(
    tokenizer: PreTrainedTokenizerBase, tool_call: Dict[str, Any]
) -> str:
    """Render one complete native call for exact binding to its original turn.

    No schema or prose is supplied, so literal tool markup inside argument
    values cannot be mistaken for the outer call's first/last delimiters.
    The auxiliary render never replaces the model's actual input.
    """
    rendered = _render_chat_text(
        tokenizer,
        [
            {"role": "user", "content": "Call the supplied tool."},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        ],
        preserve_thinking=True,
    )
    if rendered is None:
        raise ValueError("Unable to render a native tool call for KD alignment")
    assistant = rendered.find(_IM_START_ASSISTANT)
    start = rendered.find("<tool_call>", assistant + len(_IM_START_ASSISTANT))
    end = rendered.rfind("</tool_call>")
    if assistant < 0 or start < 0 or end < start:
        raise ValueError("Native tool rendering must contain a complete <tool_call>")
    return rendered[start : end + len("</tool_call>")]


def _native_scalar_spans(
    tokenizer: PreTrainedTokenizerBase,
    tool_call: Dict[str, Any],
    rendered_call: str,
) -> List[Tuple[str, Tuple[int, int]]]:
    """Locate direct bool/null values without parsing markup inside arguments."""
    function = tool_call.get("function", tool_call)
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        # OpenAI permits pre-serialized JSON. Preserve its exact spelling;
        # direct scalar compatibility applies only to typed argument values.
        if not isinstance(json.loads(arguments), dict):
            raise ValueError("Native tool arguments JSON must encode an object")
        return []
    if not isinstance(arguments, dict):
        raise ValueError(
            "Native tool arguments must be a dictionary or JSON object string"
        )
    spans: List[Tuple[str, Tuple[int, int]]] = []
    for name, value in arguments.items():
        if not isinstance(value, bool) and value is not None:
            continue
        marker = "__nemo_rl_native_scalar_probe__"
        while marker in rendered_call:
            marker += "_"
        changed_function = dict(function)
        changed_function["arguments"] = {**arguments, name: marker}
        changed_call = (
            {**tool_call, "function": changed_function}
            if "function" in tool_call
            else changed_function
        )
        probe = _render_native_tool_call(tokenizer, changed_call)
        if probe.count(marker) != 1:
            raise ValueError("Native template did not preserve the scalar probe")
        start = 0
        while (
            start < min(len(rendered_call), len(probe))
            and rendered_call[start] == probe[start]
        ):
            start += 1
        suffix = 0
        while (
            suffix < min(len(rendered_call), len(probe)) - start
            and rendered_call[-1 - suffix] == probe[-1 - suffix]
        ):
            suffix += 1
        end = len(rendered_call) - suffix
        expected = (
            {"True", "true"}
            if value is True
            else {"False", "false"}
            if value is False
            else {"None", "null"}
        )
        if rendered_call[start:end] not in expected:
            raise ValueError(
                f"Cannot bind native scalar argument {name!r} to its rendered value"
            )
        spans.append((name, (start, end)))
    return sorted(spans, key=lambda item: item[1][0])


def _native_answer_parts_for_document(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    selected: list[int],
    document: RenderedChatDocument,
) -> List[List[NativeAlignmentPart] | None]:
    """Bind structured calls to independent KD coordinates, preserving CE."""
    text = document.rendered_text
    turns: List[List[NativeAlignmentPart] | None] = []
    for message, selected_turn in zip(messages, selected):
        if message.get("role") != "assistant" or not selected_turn:
            continue
        regions = document.alignment_regions[len(turns)]
        calls = message.get("tool_calls")
        if not calls:
            turns.append(None)
            continue
        assert regions.answer is not None
        answer_start, answer_end = regions.answer
        cursor = answer_start
        prose = str(message.get("content", "") or "").strip()
        parts: List[NativeAlignmentPart] = []
        if prose:
            if not text.startswith(prose, cursor):
                raise ValueError(
                    "Cannot bind native assistant prose to its answer region"
                )
            parts.append(
                NativeAlignmentPart(
                    name="prose",
                    span=(cursor, cursor + len(prose)),
                    text=prose,
                    allow_native_difference=False,
                )
            )
            cursor += len(prose)
        for call_index, call in enumerate(calls):
            rendered_call = _render_native_tool_call(tokenizer, call)
            start = text.find(rendered_call, cursor, answer_end)
            if start < 0 or text[cursor:start].strip():
                raise ValueError(
                    f"Cannot bind native tool call {call_index} to its original answer"
                )
            local_cursor = 0
            for scalar_index, (name, (scalar_start, scalar_end)) in enumerate(
                _native_scalar_spans(tokenizer, call, rendered_call)
            ):
                if scalar_start < local_cursor:
                    raise ValueError("Native scalar argument spans overlap")
                parts.append(
                    NativeAlignmentPart(
                        name=f"tool_{call_index}/text_{scalar_index}",
                        span=(start + local_cursor, start + scalar_start),
                        text=rendered_call[local_cursor:scalar_start],
                        allow_native_difference=False,
                    )
                )
                parts.append(
                    NativeAlignmentPart(
                        name=f"tool_{call_index}/scalar/{name}",
                        span=(start + scalar_start, start + scalar_end),
                        text=rendered_call[scalar_start:scalar_end],
                        allow_native_difference=True,
                    )
                )
                local_cursor = scalar_end
            parts.append(
                NativeAlignmentPart(
                    name=f"tool_{call_index}/tail",
                    span=(start + local_cursor, start + len(rendered_call)),
                    text=rendered_call[local_cursor:],
                    allow_native_difference=False,
                )
            )
            cursor = start + len(rendered_call)
        if text[cursor:answer_end].strip():
            raise ValueError("Native answer contains unbound text after its tool calls")
        turns.append(parts)
    return turns
