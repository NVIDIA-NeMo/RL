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
r"""Failed tool calls in token-capture rows, for ``<algo>.tool_failure_credit``.

A tool call *failed* when the tool itself refused or could not carry it out: the agent harness answered with one
of its own error replies instead of a result (the edit's search string was not found, the path does not exist, the
arguments did not match the tool's schema, a permission rule blocked the command, ...). A shell command that ran and
exited non-zero, or tests that failed, are not failed tool calls: that is the environment answering.

On the token-capture path a trainer row interleaves the model's turns (``token_mask`` == 1, one turn per model
call) with the environment's replies (``token_mask`` == 0). The chat template wraps every tool result in
``<tool_response>`` ... ``</tool_response>``, in the order of the turn's ``<tool_call>`` blocks, inside the stretch of
non-generated tokens that follows the turn. ``failed_tool_turns`` decodes the head of every tool response after a
turn (the harness's error replies are recognisable from their first line, so only ``RESPONSE_HEAD_TOKENS`` tokens
are decoded), classifies it with ``classify_tool_response``, and marks every generated token of a turn that has at
least one failed call in an allowed category.

The error replies are the harness's own strings (OpenCode's tools, the search harness's command filter, the GDPval
editor). They and the category rules follow a message-level audit of training rollouts, in which only
``VALIDATED_CATEGORIES`` were validated as the model's mistakes (precision 0.88-1.0); ``permission_denied`` is mostly
benign commands hitting over-broad deny rules, and ``search_tool_error`` / ``tool_error`` were not validated.
A generic ``Error`` head counts only for a non-shell tool, because a shell command's output often starts with a
program's own ``ERROR:``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

TOOL_CALL_OPEN = "<tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
THINK_CLOSE = "</think>"

# Tokens decoded after a <tool_response> / <tool_call> tag: enough for an error reply's first line / the
# "<function=NAME>" header, and a bound on the decode cost of long tool outputs.
RESPONSE_HEAD_TOKENS = 64
CALL_HEAD_TOKENS = 16

SCHEMA_ERROR = "schema_error"
EDIT_FAILED = "edit_failed"
READ_MISSING = "read_missing"
PERMISSION_DENIED = "permission_denied"
SEARCH_TOOL_ERROR = "search_tool_error"
TOOL_ERROR = "tool_error"
CATEGORIES = (
    SCHEMA_ERROR,
    EDIT_FAILED,
    READ_MISSING,
    PERMISSION_DENIED,
    SEARCH_TOOL_ERROR,
    TOOL_ERROR,
)
# The categories validated as the model's mistakes on training rollouts.
VALIDATED_CATEGORIES = (SCHEMA_ERROR, EDIT_FAILED, READ_MISSING)

SHELL_TOOLS = frozenset({"bash", "execute_bash", "bash_command"})
EDIT_TOOLS = frozenset({"edit", "multiedit", "write", "patch", "str_replace_editor"})
READ_TOOLS = frozenset({"read", "list", "glob"})

# Replies the harness itself gives for any tool, a shell tool included (anchored at the start of the response).
_HARNESS_REPLY = re.compile(
    r"(The \w+ tool was called with invalid arguments|The arguments provided to the tool are invalid|"
    r"The user has specified a rule which prevents|\[blocked: |NotFound: FileSystem)"
)
# The error replies of the harness's own (non-shell) tools; a shell command's output can start with these too.
_TOOL_ERROR_REPLY = re.compile(
    r"(Could not find oldString|Found multiple matches for oldString|No changes to apply|File not found:|"
    r"File \S+ not found|Offset \d+ is out of range|Cannot read binary file|ripgrep execution failed|"
    r"rg: regex parse error|Ripgrep JSON record exceeded|You must read (the )?file|File has been modified since|"
    r"No replacement was performed|Invalid `path` parameter)"
)
# GDPval's editor prefixes its error replies with this.
_ERROR_PREFIX = re.compile(r"ERROR:\s*")
# Generic error heads, trusted only for non-shell tools.
_GENERIC_ERROR = re.compile(
    r"(ERROR|Error|error)\b|(Unknown tool|Invalid arguments|Tool \S+ not found)"
)
_FUNCTION_NAME = re.compile(r"<function=([^>\n]+)>")


def classify_tool_response(tool: Optional[str], response: str) -> Optional[str]:
    """Category of a failed tool call from the tool's name and its response text, or None when it succeeded.

    For a shell tool, and for ``tool`` None (a response with no matching call), only the harness's own replies
    count (``_HARNESS_REPLY``): a program's output can start with any other error text.
    """
    head = response.lstrip()
    if tool is None or tool in SHELL_TOOLS:
        if not _HARNESS_REPLY.match(head):
            return None
    else:
        failed = _GENERIC_ERROR.match(head)
        head = _ERROR_PREFIX.sub("", head, count=1)
        if not (failed or _HARNESS_REPLY.match(head) or _TOOL_ERROR_REPLY.match(head)):
            return None
    start = head[:120]
    if (
        tool == "invalid"
        or "called with invalid arguments" in start
        or "arguments provided to the tool are invalid" in start
        or head.startswith(
            ("Unknown tool", "Invalid arguments", "Invalid `path` parameter")
        )
        or re.match(r"Tool \S+ not found", head)
    ):
        return SCHEMA_ERROR
    if head.startswith(
        (
            "Could not find oldString",
            "Found multiple matches",
            "No changes to apply",
            "No replacement was performed",
            "File has been modified since",
            "You must read",
        )
    ) or (
        tool in EDIT_TOOLS and ("not found" in head[:80] or head.startswith("Invalid"))
    ):
        return EDIT_FAILED
    if head.startswith(("The user has specified a rule", "[blocked: ")):
        return PERMISSION_DENIED
    if (
        tool in READ_TOOLS
        or head.startswith(
            ("File not found", "Offset", "Cannot read binary", "NotFound: FileSystem")
        )
        or re.match(r"File \S+ not found", head)
    ):
        return READ_MISSING
    if head.startswith(
        (
            "ripgrep execution failed",
            "rg: regex parse error",
            "Ripgrep JSON record exceeded",
        )
    ):
        return SEARCH_TOOL_ERROR
    return TOOL_ERROR


@dataclass(frozen=True)
class ToolTagIds:
    """Single-token ids of the chat-template tags ``failed_tool_turns`` scans for."""

    tool_call_open: int
    tool_response_open: int
    tool_response_close: int
    # ``THINK_CLOSE``; None when the tokenizer has no single-token thinking tag (no thinking span).
    think_close: Optional[int] = None


def single_token_id(tokenizer: Any, tag: str) -> Optional[int]:
    """The tokenizer's id for ``tag`` when it encodes to exactly that one token, else None."""
    tid = tokenizer.convert_tokens_to_ids(tag)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        return None
    if list(tokenizer.encode(tag, add_special_tokens=False)) != [tid]:
        return None
    return int(tid)


def resolve_tool_tag_ids(tokenizer: Any) -> ToolTagIds:
    """Ids of the tags ``failed_tool_turns`` scans for; the tool tags must each be one token."""
    tags = {
        "tool_call_open": TOOL_CALL_OPEN,
        "tool_response_open": TOOL_RESPONSE_OPEN,
        "tool_response_close": TOOL_RESPONSE_CLOSE,
    }
    ids = {key: single_token_id(tokenizer, tag) for key, tag in tags.items()}
    missing = [tags[key] for key, tid in ids.items() if tid is None]
    if missing:
        raise ValueError(
            "tool_failure_credit needs these tags to be single tokens of the policy tokenizer: "
            f"{missing}. Failed calls are found by scanning sampled token ids, so a tag split into "
            "several tokens cannot be located."
        )
    return ToolTagIds(
        tool_call_open=cast(int, ids["tool_call_open"]),
        tool_response_open=cast(int, ids["tool_response_open"]),
        tool_response_close=cast(int, ids["tool_response_close"]),
        think_close=single_token_id(tokenizer, THINK_CLOSE),
    )


@dataclass
class RowToolFailures:
    """``failed_tool_turns`` result for one row."""

    # Token-aligned bool array: True on every generated token of a turn with a failed call in an allowed category.
    mask: np.ndarray
    turns: int = 0
    failed_turns: int = 0
    tool_responses: int = 0
    # Failed calls per category (all categories, allowed or not, for monitoring).
    failed_calls: dict[str, int] = field(default_factory=dict)


def failed_tool_turns(
    token_ids: Sequence[int],
    token_mask: Sequence[float],
    *,
    ids: ToolTagIds,
    decode_many: Callable[[list[list[int]]], list[str]],
    categories: Collection[str],
) -> RowToolFailures:
    """Mark the generated tokens of every turn in one row whose tool calls include a failed one.

    A turn is a run of generated tokens; its replies are the non-generated tokens up to the next turn. The k-th
    ``<tool_response>`` in the replies answers the turn's k-th ``<tool_call>`` after the turn's first ``</think>``
    (vLLM's reasoning parser ends the thinking there; a tag inside the thinking is not a call). A turn with no tool response after it (the final answer, or a call cut off
    by the context limit) is never marked.

    Args:
        token_ids: the row's token ids (prompt, turns and environment replies).
        token_mask: 1 on generated tokens, 0 elsewhere.
        ids: the tag token ids (each tag is a single token on the capture path).
        decode_many: decodes a batch of token-id lists to text, special tokens kept.
        categories: the failure categories that mark a turn.

    Returns:
        The token-aligned mask and the row's counts.
    """
    tokens = np.asarray(token_ids, dtype=np.int64)
    generated = np.asarray(token_mask, dtype=np.float32) > 0
    n = int(tokens.shape[0])
    allowed = frozenset(categories)
    result = RowToolFailures(mask=np.zeros(n, dtype=bool))
    edges = np.diff(np.concatenate(([0], generated.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    result.turns = int(starts.shape[0])
    if not result.turns:
        return result
    call_pos = np.flatnonzero(generated & (tokens == ids.tool_call_open))
    think_close_pos = (
        np.flatnonzero(generated & (tokens == ids.think_close))
        if ids.think_close is not None
        else np.empty(0, dtype=np.int64)
    )
    open_pos = np.flatnonzero(~generated & (tokens == ids.tool_response_open))
    close_pos = np.flatnonzero(~generated & (tokens == ids.tool_response_close))
    reply_ends = np.append(starts[1:], n)
    call_slices: list[list[int]] = []
    response_slices: list[list[int]] = []
    plan: list[
        tuple[int, int, int, int, int]
    ] = []  # (turn, first call, n calls, first response, n responses)
    for t in range(result.turns):
        start, end, reply_end = int(starts[t]), int(ends[t]), int(reply_ends[t])
        lo, hi = np.searchsorted(open_pos, [end, reply_end])
        if hi <= lo:
            continue
        first_think = int(np.searchsorted(think_close_pos, start))
        calls_from = start
        if (
            first_think < think_close_pos.shape[0]
            and think_close_pos[first_think] < end
        ):
            calls_from = int(think_close_pos[first_think]) + 1
        c_lo, c_hi = np.searchsorted(call_pos, [calls_from, end])
        first_call = len(call_slices)
        for k in call_pos[c_lo:c_hi]:
            call_slices.append(
                tokens[k + 1 : min(end, k + 1 + CALL_HEAD_TOKENS)].tolist()
            )
        first_response = len(response_slices)
        next_free = end
        for o in open_pos[lo:hi]:
            o = int(o)
            if (
                o < next_free
            ):  # a tag inside the previous response's body, not a new response
                continue
            j = int(np.searchsorted(close_pos, o))
            close = (
                int(close_pos[j])
                if j < close_pos.shape[0] and close_pos[j] < reply_end
                else reply_end
            )
            response_slices.append(
                tokens[o + 1 : min(close, o + 1 + RESPONSE_HEAD_TOKENS)].tolist()
            )
            next_free = close + 1
        plan.append(
            (
                t,
                first_call,
                c_hi - c_lo,
                first_response,
                len(response_slices) - first_response,
            )
        )
    if not plan:
        return result
    texts = decode_many(call_slices + response_slices)
    call_texts, response_texts = texts[: len(call_slices)], texts[len(call_slices) :]
    for t, first_call, n_calls, first_response, n_responses in plan:
        failed = False
        for r in range(n_responses):
            tool = None
            if r < n_calls:
                match = _FUNCTION_NAME.search(call_texts[first_call + r])
                tool = match.group(1).strip() if match else None
            category = classify_tool_response(tool, response_texts[first_response + r])
            result.tool_responses += 1
            if category is None:
                continue
            result.failed_calls[category] = result.failed_calls.get(category, 0) + 1
            failed = failed or category in allowed
        if failed:
            result.mask[int(starts[t]) : int(ends[t])] = True
            result.failed_turns += 1
    return result
