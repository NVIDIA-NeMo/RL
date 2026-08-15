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

"""Convert OpenAI Responses trajectory items to chat-template messages."""

import json
from dataclasses import dataclass, field
from typing import Any


class ResponsesFormatError(ValueError):
    """Raised when a Responses trajectory cannot be represented safely."""


def _content_parts_for_chat(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesFormatError(
            f"message content must be a string or list, got {type(content).__name__}"
        )

    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ResponsesFormatError("message content parts must be objects")
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ResponsesFormatError(f"{part_type} part is missing string text")
            converted.append({"type": "text", "text": text})
        elif part_type == "input_image":
            image_url = part.get("image_url", "")
            if not isinstance(image_url, str):
                raise ResponsesFormatError("input_image.image_url must be a string")
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": part.get("detail", "auto"),
                    },
                }
            )
        else:
            raise ResponsesFormatError(f"unsupported message content type: {part_type}")
    return converted


def _assistant_content_text(content: Any) -> str:
    converted = _content_parts_for_chat(content)
    if isinstance(converted, str):
        return converted
    return "".join(part.get("text", "") for part in converted)


def _reasoning_text(item: dict[str, Any]) -> str:
    summary = item.get("summary") or []
    if not isinstance(summary, list):
        raise ResponsesFormatError("reasoning.summary must be a list")
    wrapped: list[str] = []
    for part in summary:
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            raise ResponsesFormatError(
                "reasoning summary parts must contain string text"
            )
        if part["text"]:
            wrapped.append(f"<think>{part['text']}</think>")
    return "".join(wrapped)


def _tool_arguments_for_chat_template(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"arguments": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"arguments": parsed}


@dataclass
class _AssistantBuffer:
    messages: list[dict[str, Any]] = field(default_factory=list)
    message_item_indices: list[list[int]] = field(default_factory=list)
    item_message_indices: list[int | None] = field(default_factory=list)
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending_item_indices: list[int] = field(default_factory=list)

    def append_message(self, message: dict[str, Any], item_indices: list[int]) -> None:
        message_index = len(self.messages)
        self.messages.append(message)
        self.message_item_indices.append(item_indices)
        for item_index in item_indices:
            if self.item_message_indices[item_index] is not None:
                raise ResponsesFormatError(
                    f"Responses item {item_index} was assigned to multiple messages"
                )
            self.item_message_indices[item_index] = message_index

    def flush(self) -> None:
        if not self.content and not self.tool_calls:
            if self.pending_item_indices:
                raise ResponsesFormatError(
                    "assistant items produced neither content nor tool calls"
                )
            return
        self.append_message(
            {
                "role": "assistant",
                "content": self.content or None,
                "tool_calls": self.tool_calls.copy(),
            },
            self.pending_item_indices.copy(),
        )
        self.content = ""
        self.tool_calls.clear()
        self.pending_item_indices.clear()


@dataclass(frozen=True)
class ResponsesMaterialization:
    """Chat-template inputs plus the originating Responses-item boundaries."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    item_message_indices: list[int]
    message_item_indices: list[list[int]]


def normalize_assistant_thinking_prefixes(
    materialized: ResponsesMaterialization,
) -> ResponsesMaterialization:
    r"""Make assistant messages share Nano's ``<think>\n`` generation prefix.

    This normalization is opt-in because it changes rendered v1 trajectories.
    Multi-target critic rows use it so a real assistant action and an appended
    generation prompt represent the same pre-action state boundary.
    """
    messages: list[dict[str, Any]] = []
    for original in materialized.messages:
        message = dict(original)
        if message.get("role") == "assistant":
            content = message.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise ResponsesFormatError("assistant content must be text")
            if content.startswith("<think>"):
                suffix = content[len("<think>") :]
                if not suffix.startswith("\n"):
                    content = "<think>\n" + suffix
            elif "</think>" in content:
                content = "<think>\n" + content
            else:
                content = "<think>\n</think>" + content
            message["content"] = content
        messages.append(message)
    return ResponsesMaterialization(
        messages=messages,
        tools=materialized.tools,
        item_message_indices=materialized.item_message_indices,
        message_item_indices=materialized.message_item_indices,
    )


def responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert Responses function definitions to HF chat-template tools."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ResponsesFormatError("responses_create_params.tools must be a list")

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ResponsesFormatError("only Responses function tools are supported")
        if isinstance(tool.get("function"), dict):
            function = dict(tool["function"])
        else:
            function = {key: value for key, value in tool.items() if key != "type"}
        function.pop("strict", None)
        if not isinstance(function.get("name"), str):
            raise ResponsesFormatError("function tool is missing a string name")
        converted.append({"type": "function", "function": function})
    return converted


def materialize_responses_create_params(
    responses_create_params: dict[str, Any],
) -> ResponsesMaterialization:
    """Materialize a Responses trajectory and retain exact item boundaries."""
    if not isinstance(responses_create_params, dict):
        raise ResponsesFormatError("responses_create_params must be an object")

    response_input = responses_create_params.get("input", [])
    if isinstance(response_input, str):
        input_items: list[dict[str, Any]] = [
            {"type": "message", "role": "user", "content": response_input}
        ]
    elif isinstance(response_input, list):
        input_items = response_input
    else:
        raise ResponsesFormatError(
            "responses_create_params.input must be a string or list"
        )

    item_message_indices: list[int | None] = []
    for _ in input_items:
        item_message_indices.append(None)
    state = _AssistantBuffer(item_message_indices=item_message_indices)
    instructions = responses_create_params.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ResponsesFormatError(
                "responses_create_params.instructions must be a string"
            )
        if instructions:
            state.append_message({"role": "system", "content": instructions}, [])

    for item_index, raw_item in enumerate(input_items):
        if not isinstance(raw_item, dict):
            raise ResponsesFormatError("Responses input items must be objects")
        item = dict(raw_item)
        item_type = item.get("type")
        if item_type is None and item.get("role"):
            item_type = "message"

        if item_type == "reasoning":
            state.content += _reasoning_text(item)
            state.pending_item_indices.append(item_index)
        elif item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, str)
            ):
                raise ResponsesFormatError(
                    "function_call requires string call_id, name, and arguments"
                )
            state.tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _tool_arguments_for_chat_template(arguments),
                    },
                }
            )
            state.pending_item_indices.append(item_index)
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise ResponsesFormatError(
                    "function_call_output requires a string call_id"
                )
            state.flush()
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, sort_keys=True)
            state.append_message(
                {"role": "tool", "tool_call_id": call_id, "content": output},
                [item_index],
            )
        elif item_type == "message":
            role = item.get("role")
            if role == "assistant":
                state.content += _assistant_content_text(item.get("content", ""))
                state.pending_item_indices.append(item_index)
            elif role in {"user", "system", "developer"}:
                state.flush()
                state.append_message(
                    {
                        "role": role,
                        "content": _content_parts_for_chat(item.get("content", "")),
                    },
                    [item_index],
                )
            else:
                raise ResponsesFormatError(f"unsupported message role: {role}")
        else:
            raise ResponsesFormatError(f"unsupported Responses item type: {item_type}")

    state.flush()
    if not state.messages:
        raise ResponsesFormatError("trajectory contains no materialized messages")
    if any(message_index is None for message_index in state.item_message_indices):
        missing = [
            index
            for index, message_index in enumerate(state.item_message_indices)
            if message_index is None
        ]
        raise ResponsesFormatError(
            f"Responses items have no message mapping: {missing}"
        )
    tools = responses_tools_to_chat_tools(responses_create_params.get("tools"))
    return ResponsesMaterialization(
        messages=state.messages,
        tools=tools,
        item_message_indices=[
            message_index
            for message_index in state.item_message_indices
            if message_index is not None
        ],
        message_item_indices=state.message_item_indices,
    )


def responses_to_chat_messages(
    responses_create_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize a Responses trajectory without executing any trajectory item."""
    materialized = materialize_responses_create_params(responses_create_params)
    return materialized.messages, materialized.tools
