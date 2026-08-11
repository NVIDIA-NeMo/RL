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
"""Shared contract for real vLLM/TRT-LLM chat-completions parity tests."""

import json
import os

MODEL = os.environ.get("PARITY_MODEL", "Qwen/Qwen3-0.6B")
MODEL_REVISION = os.environ.get(
    "PARITY_MODEL_REVISION", "c1899de289a04d12100db370d81485cdf75e47ca"
)

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Run a bash command and return its stdout.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                }
            },
            "required": ["command"],
        },
    },
}

# Cover thinking with both parsers and Qwen3 without thinking.
PARSER_SCENARIOS = {
    "qwen3": (("qwen3_with_thinking", True), ("qwen3_without_thinking", False)),
    "deepseek_r1": (("deepseek_r1_with_thinking", True),),
}

GENERATION_TOKEN_IDS_FIELD = "generation_token_ids"

# Normalize serving-generated tool-call IDs.
TOOL_CALL_ID = "chat-template-parity-tool-call"

USER_MSG = "List the files in /tmp using run_bash."
TOOL_RESULT = "file1.txt\nfile2.txt\nREADME.md"
FOLLOWUP_USER_MSG = "Without calling a tool, reply with exactly: Done."


def _tool_call_text(command: str) -> str:
    payload = {"name": "run_bash", "arguments": {"command": command}}
    return "<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"


# Deterministic cases exercise each backend's installed parsers.
REASONING_PARSER_CONTRACT_CASES = (
    {
        "name": "explicit_reasoning_answer",
        "raw_output": "<think>inspect files</think>Visible answer",
        "enable_thinking": True,
        "reasoning_at_start": False,
        "expected": {
            "reasoning_content": "inspect files",
            "content": "Visible answer",
            "tool_calls": [],
        },
    },
    {
        "name": "reasoning_then_tool_call",
        "raw_output": "<think>need shell</think>" + _tool_call_text("pwd"),
        "enable_thinking": True,
        "reasoning_at_start": False,
        "expected": {
            "reasoning_content": "need shell",
            "content": None,
            "tool_calls": [{"name": "run_bash", "arguments": {"command": "pwd"}}],
        },
    },
    {
        "name": "disabled_thinking_spontaneous_reasoning",
        "raw_output": "<think>unexpected thought</think>Visible answer",
        "enable_thinking": False,
        "reasoning_at_start": False,
        "expected": {
            "reasoning_content": "unexpected thought",
            "content": "Visible answer",
            "tool_calls": [],
        },
    },
    {
        "name": "prompt_injected_reasoning_start",
        "raw_output": "inspect files</think>Visible answer",
        "enable_thinking": True,
        "reasoning_at_start": True,
        "expected": {
            "reasoning_content": "inspect files",
            "content": "Visible answer",
            "tool_calls": [],
        },
    },
)

TOOL_PARSER_CONTRACT_CASES = (
    {
        "name": "single_tool_call",
        "raw_output": _tool_call_text("pwd"),
        "expected": {
            "reasoning_content": "",
            "content": None,
            "tool_calls": [{"name": "run_bash", "arguments": {"command": "pwd"}}],
        },
    },
    {
        "name": "multiple_tool_calls",
        "raw_output": _tool_call_text("pwd") + "\n" + _tool_call_text("whoami"),
        "expected": {
            "reasoning_content": "",
            "content": None,
            "tool_calls": [
                {"name": "run_bash", "arguments": {"command": "pwd"}},
                {"name": "run_bash", "arguments": {"command": "whoami"}},
            ],
        },
    },
    {
        "name": "text_before_and_after_tool_call",
        "raw_output": "before " + _tool_call_text("pwd") + " after",
        "expected": {
            "reasoning_content": "",
            "content": "before",
            "tool_calls": [{"name": "run_bash", "arguments": {"command": "pwd"}}],
        },
    },
    {
        "name": "malformed_tool_call_falls_back_to_content",
        "raw_output": "<tool_call>{bad json}</tool_call>",
        "expected": {
            "reasoning_content": "",
            "content": "<tool_call>{bad json}</tool_call>",
            "tool_calls": [],
        },
    },
)


def token_edit_similarity(left: list[int], right: list[int]) -> float:
    """Return normalized Levenshtein similarity for two token-ID sequences."""
    if not left and not right:
        return 1.0

    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current

    return 1.0 - previous[-1] / max(len(left), len(right))


def inclusive_token_span(
    token_ids: list[int], start_marker: list[int], end_marker: list[int]
) -> list[int]:
    """Return the first token span bounded by the supplied marker sequences."""
    assert start_marker, "start marker must not be empty"
    assert end_marker, "end marker must not be empty"

    def find_subsequence(needle: list[int], start: int) -> int:
        last_start = len(token_ids) - len(needle)
        for index in range(start, last_start + 1):
            if token_ids[index : index + len(needle)] == needle:
                return index
        return -1

    span_start = find_subsequence(start_marker, 0)
    assert span_start >= 0, f"start marker {start_marker!r} not found in generation"
    end_start = find_subsequence(end_marker, span_start + len(start_marker))
    assert end_start >= 0, f"end marker {end_marker!r} not found in generation"
    return token_ids[span_start : end_start + len(end_marker)]


def prompt_suffix_after_turn(turns: list[dict], turn_index: int) -> list[int]:
    """Return tokens appended after one turn's exact model prefix."""
    previous_turn = turns[turn_index]
    next_turn = turns[turn_index + 1]
    required_prefix = (
        previous_turn["prompt_token_ids"] + previous_turn["generation_token_ids"]
    )
    prompt = next_turn["prompt_token_ids"]
    assert prompt[: len(required_prefix)] == required_prefix
    return prompt[len(required_prefix) :]
